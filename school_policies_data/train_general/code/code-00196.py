    # TODO Move this to C++ once the jit has better support for torch.Size.
    if not torch.jit.is_tracing():
        result = torch._refs._broadcast_shapes(*shapes)
        if result is None:
            return torch.Size([])
        return torch.Size(result)
    else:
        # with implementation above, torch.jit.trace hardcodes the sizes which makes subsequent replays fail
        with torch.no_grad():
            scalar = torch.zeros((), device="cpu")
            tensors = [scalar.expand(shape) for shape in shapes]
            tensors = broadcast_tensors(*tensors)
            return tensors[0].shape


def split(
    tensor: Tensor,
    split_size_or_sections: int | list[int],
    dim: int = 0,
) -> tuple[Tensor, ...]:
    r"""Splits the tensor into chunks. Each chunk is a view of the original tensor.

    If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will
    be split into equally sized chunks (if possible). Last chunk will be smaller if
    the tensor size along the given dimension :attr:`dim` is not divisible by
    :attr:`split_size`.

    If :attr:`split_size_or_sections` is a list, then :attr:`tensor` will be split
    into ``len(split_size_or_sections)`` chunks with sizes in :attr:`dim` according
    to :attr:`split_size_or_sections`.

    Args:
        tensor (Tensor): tensor to split.
        split_size_or_sections (int) or (list(int)): size of a single chunk or
            list of sizes for each chunk
        dim (int): dimension along which to split the tensor.

    Example::

        >>> a = torch.arange(10).reshape(5, 2)
        >>> a
        tensor([[0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9]])
        >>> torch.split(a, 2)
        (tensor([[0, 1],
                 [2, 3]]),
         tensor([[4, 5],
                 [6, 7]]),
         tensor([[8, 9]]))
        >>> torch.split(a, [1, 4])
        (tensor([[0, 1]]),
         tensor([[2, 3],
                 [4, 5],
                 [6, 7],
                 [8, 9]]))
    """
    if has_torch_function_unary(tensor):
        return handle_torch_function(
            split, (tensor,), tensor, split_size_or_sections, dim=dim
        )
    # Overwriting reason:
    # This dispatches to two ATen functions depending on the type of
    # split_size_or_sections. The branching code is in _tensor.py, which we
    # call here.
    return tensor.split(split_size_or_sections, dim)


def einsum(*args: Any) -> Tensor:
    r"""einsum(equation, *operands) -> Tensor

    Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation
    based on the Einstein summation convention.

    Einsum allows computing many common multi-dimensional linear algebraic array operations by representing them
    in a short-hand format based on the Einstein summation convention, given by :attr:`equation`. The details of
    this format are described below, but the general idea is to label every dimension of the input :attr:`operands`
    with some subscript and define which subscripts are part of the output. The output is then computed by summing
    the product of the elements of the :attr:`operands` along the dimensions whose subscripts are not part of the
    output. For example, matrix multiplication can be computed using einsum as `torch.einsum("ij,jk->ik", A, B)`.
    Here, j is the summation subscript and i and k the output subscripts (see section below for more details on why).

    Equation:

        The :attr:`equation` string specifies the subscripts (letters in `[a-zA-Z]`) for each dimension of
        the input :attr:`operands` in the same order as the dimensions, separating subscripts for each operand by a
        comma (','), e.g. `'ij,jk'` specify subscripts for two 2D operands. The dimensions labeled with the same subscript
        must be broadcastable, that is, their size must either match or be `1`. The exception is if a subscript is
        repeated for the same input operand, in which case the dimensions labeled with this subscript for this operand
        must match in size and the operand will be replaced by its diagonal along these dimensions. The subscripts that
        appear exactly once in the :attr:`equation` will be part of the output, sorted in increasing alphabetical order.
        The output is computed by multiplying the input :attr:`operands` element-wise, with their dimensions aligned based
        on the subscripts, and then summing out the dimensions whose subscripts are not part of the output.

        Optionally, the output subscripts can be explicitly defined by adding an arrow ('->') at the end of the equation
        followed by the subscripts for the output. For instance, the following equation computes the transpose of a
        matrix multiplication: 'ij,jk->ki'. The output subscripts must appear at least once for some input operand and
        at most once for the output.

        Ellipsis ('...') can be used in place of subscripts to broadcast the dimensions covered by the ellipsis.
        Each input operand may contain at most one ellipsis which will cover the dimensions not covered by subscripts,
        e.g. for an input operand with 5 dimensions, the ellipsis in the equation `'ab...c'` cover the third and fourth
        dimensions. The ellipsis does not need to cover the same number of dimensions across the :attr:`operands` but the
        'shape' of the ellipsis (the size of the dimensions covered by them) must broadcast together. If the output is not
        explicitly defined with the arrow ('->') notation, the ellipsis will come first in the output (left-most dimensions),
        before the subscript labels that appear exactly once for the input operands. e.g. the following equation implements
        batch matrix multiplication `'...ij,...jk'`.

        A few final notes: the equation may contain whitespaces between the different elements (subscripts, ellipsis,
        arrow and comma) but something like `'. . .'` is not valid. An empty string `''` is valid for scalar operands.

    .. note::

        ``torch.einsum`` handles ellipsis ('...') differently from NumPy in that it allows dimensions
        covered by the ellipsis to be summed over, that is, ellipsis are not required to be part of the output.

    .. note::

        Please install opt-einsum (https://optimized-einsum.readthedocs.io/en/stable/) in order to enroll into a more
        performant einsum. You can install when installing torch like so: `pip install torch[opt-einsum]` or by itself
        with `pip install opt-einsum`.

        If opt-einsum is available, this function will automatically speed up computation and/or consume less memory
        by optimizing contraction order through our opt_einsum backend :mod:`torch.backends.opt_einsum` (The _ vs - is
        confusing, I know). This optimization occurs when there are at least three inputs, since the order does not matter
        otherwise. Note that finding `the` optimal path is an NP-hard problem, thus, opt-einsum relies on different
        heuristics to achieve near-optimal results. If opt-einsum is not available, the default order is to contract
        from left to right.

        To bypass this default behavior, add the following to disable opt_einsum and skip path calculation:
        ``torch.backends.opt_einsum.enabled = False``

        To specify which strategy you'd like for opt_einsum to compute the contraction path, add the following line:
        ``torch.backends.opt_einsum.strategy = 'auto'``. The default strategy is 'auto', and we also support 'greedy' and
        'optimal'. Disclaimer that the runtime of 'optimal' is factorial in the number of inputs! See more details in
        the opt_einsum documentation (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html).

    .. note::

        As of PyTorch 1.10 :func:`torch.einsum` also supports the sublist format (see examples below). In this format,
        subscripts for each operand are specified by sublists, list of integers in the range [0, 52). These sublists
        follow their operands, and an extra sublist can appear at the end of the input to specify the output's
        subscripts., e.g. `torch.einsum(op1, sublist1, op2, sublist2, ..., [subslist_out])`. Python's `Ellipsis` object
        may be provided in a sublist to enable broadcasting as described in the Equation section above.

    Args:
        equation (str): The subscripts for the Einstein summation.
        operands (List[Tensor]): The tensors to compute the Einstein summation of.

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # trace
        >>> torch.einsum('ii', torch.randn(4, 4))
        tensor(-1.2104)

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # diagonal
        >>> torch.einsum('ii->i', torch.randn(4, 4))
        tensor([-0.1034,  0.7952, -0.2433,  0.4545])

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # outer product
        >>> x = torch.randn(5)
        >>> y = torch.randn(4)
        >>> torch.einsum('i,j->ij', x, y)
        tensor([[ 0.1156, -0.2897, -0.3918,  0.4963],
                [-0.3744,  0.9381,  1.2685, -1.6070],
                [ 0.7208, -1.8058, -2.4419,  3.0936],
                [ 0.1713, -0.4291, -0.5802,  0.7350],
                [ 0.5704, -1.4290, -1.9323,  2.4480]])

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # batch matrix multiplication
        >>> As = torch.randn(3, 2, 5)
        >>> Bs = torch.randn(3, 5, 4)
        >>> torch.einsum('bij,bjk->bik', As, Bs)
        tensor([[[-1.0564, -1.5904,  3.2023,  3.1271],
                [-1.6706, -0.8097, -0.8025, -2.1183]],

                [[ 4.2239,  0.3107, -0.5756, -0.2354],
                [-1.4558, -0.3460,  1.5087, -0.8530]],

                [[ 2.8153,  1.8787, -4.3839, -1.2112],
                [ 0.3728, -2.1131,  0.0921,  0.8305]]])

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # with sublist format and ellipsis
        >>> torch.einsum(As, [..., 0, 1], Bs, [..., 1, 2], [..., 0, 2])
        tensor([[[-1.0564, -1.5904,  3.2023,  3.1271],
                [-1.6706, -0.8097, -0.8025, -2.1183]],

                [[ 4.2239,  0.3107, -0.5756, -0.2354],
                [-1.4558, -0.3460,  1.5087, -0.8530]],

                [[ 2.8153,  1.8787, -4.3839, -1.2112],
                [ 0.3728, -2.1131,  0.0921,  0.8305]]])

        >>> # batch permute
        >>> A = torch.randn(2, 3, 4, 5)
        >>> torch.einsum('...ij->...ji', A).shape
        torch.Size([2, 3, 5, 4])

        >>> # equivalent to torch.nn.functional.bilinear
        >>> A = torch.randn(3, 5, 4)
        >>> l = torch.randn(2, 5)
        >>> r = torch.randn(2, 4)
        >>> torch.einsum('bn,anm,bm->ba', l, A, r)
        tensor([[-0.3430, -5.2405,  0.4494],
                [ 0.3311,  5.5201, -3.0356]])
    """
    import torch.backends.opt_einsum as opt_einsum

    # This wrapper exists to support variadic args.
    if len(args) < 2:
        raise ValueError(
            "einsum(): must specify the equation string and at least one operand, "
            "or at least one operand and its subscripts list"
        )

    equation = None
    operands = None

    if isinstance(args[0], torch.Tensor):
        # Convert the subscript list format which is an interleaving of operand and its subscripts
        # list with an optional output subscripts list at the end (see documentation for more details on this)
        # to the equation string format by creating the equation string from the subscripts list and grouping the
        # input operands into a tensorlist (List[Tensor]).
        def parse_subscript(n: int) -> str:
            if n == Ellipsis:
                return "..."
            if n >= 0 and n < 26:
                return chr(ord("A") + n)
            if n >= 26 and n < 52:
                return chr(ord("a") + n - 26)
            raise ValueError(
                "einsum(): subscript in subscript list is not within the valid range [0, 52)"
            )

        # Parse subscripts for input operands
        equation = ",".join("".join(parse_subscript(s) for s in l) for l in args[1::2])

        # Parse optional output subscripts (provided when the number of arguments is odd)
        if len(args) % 2 == 1:
            equation += "->" + "".join(parse_subscript(s) for s in args[-1])
            operands = args[:-1:2]
        else:
            operands = args[::2]
    else:
        equation = args[0]
        operands = args[1:]

    if has_torch_function(operands):
        return handle_torch_function(einsum, operands, equation, *operands)

    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        # the old interface of passing the operands as one list argument
        _operands = operands[0]
        # recurse in case operands contains value that has torch function
        # in the original implementation this line is omitted
        return einsum(equation, *_operands)

    if len(operands) <= 2 or not opt_einsum.enabled:
        # the path for contracting 0 or 1 time(s) is already optimized
        # or the user has disabled using opt_einsum
        return _VF.einsum(equation, operands)  # type: ignore[attr-defined]

    path = None
    if opt_einsum.is_available():
        _opt_einsum = opt_einsum.get_opt_einsum()
        tupled_path = _opt_einsum.contract_path(
            equation, *operands, optimize=opt_einsum.strategy
        )[0]
        # flatten path for dispatching to C++
        path = [*itertools.chain.from_iterable(tupled_path)]
    return _VF.einsum(equation, operands, path=path)  # type: ignore[attr-defined]


# This wrapper exists to support variadic args.
if TYPE_CHECKING:
    # The JIT doesn't understand Union, so only add type annotation for mypy
    def meshgrid(
        *tensors: Tensor | list[Tensor], indexing: str | None = None
    ) -> tuple[Tensor, ...]:
        return _meshgrid(*tensors, indexing=indexing)

else:

    def meshgrid(*tensors, indexing: str | None = None) -> tuple[Tensor, ...]:
        r"""Creates grids of coordinates specified by the 1D inputs in `attr`:tensors.

        This is helpful when you want to visualize data over some
        range of inputs. See below for a plotting example.

        Given :math:`N` 1D tensors :math:`T_0 \ldots T_{N-1}` as
        inputs with corresponding sizes :math:`S_0 \ldots S_{N-1}`,
        this creates :math:`N` N-dimensional tensors :math:`G_0 \ldots
        G_{N-1}`, each with shape :math:`(S_0, ..., S_{N-1})` where
        the output :math:`G_i` is constructed by expanding :math:`T_i`
        to the result shape.

        .. note::
            0D inputs are treated equivalently to 1D inputs of a
            single element.

        .. warning::
            `torch.meshgrid(*tensors)` currently has the same behavior
            as calling `numpy.meshgrid(*arrays, indexing='ij')`.

            In the future `torch.meshgrid` will transition to
            `indexing='xy'` as the default.

            https://github.com/pytorch/pytorch/issues/50276 tracks
            this issue with the goal of migrating to NumPy's behavior.

        .. seealso::

            :func:`torch.cartesian_prod` has the same effect but it
            collects the data in a tensor of vectors.

        Args:
            tensors (list of Tensor): list of scalars or 1 dimensional tensors. Scalars will be
                treated as tensors of size :math:`(1,)` automatically

            indexing: (str, optional): the indexing mode, either "xy"
                or "ij", defaults to "ij". See warning for future changes.

                If "xy" is selected, the first dimension corresponds
                to the cardinality of the second input and the second
                dimension corresponds to the cardinality of the first
                input.

                If "ij" is selected, the dimensions are in the same
                order as the cardinality of the inputs.

        Returns:
            seq (sequence of Tensors): If the input has :math:`N`
            tensors of size :math:`S_0 \ldots S_{N-1}``, then the
            output will also have :math:`N` tensors, where each tensor
            is of shape :math:`(S_0, ..., S_{N-1})`.

        Example::

            >>> x = torch.tensor([1, 2, 3])
            >>> y = torch.tensor([4, 5, 6])

            Observe the element-wise pairings across the grid, (1, 4),
            (1, 5), ..., (3, 6). This is the same thing as the
            cartesian product.
            >>> grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            >>> grid_x
            tensor([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3]])
            >>> grid_y
            tensor([[4, 5, 6],
                    [4, 5, 6],
                    [4, 5, 6]])

            This correspondence can be seen when these grids are
            stacked properly.
            >>> torch.equal(torch.cat(tuple(torch.dstack([grid_x, grid_y]))),
            ...             torch.cartesian_prod(x, y))
            True

            `torch.meshgrid` is commonly used to produce a grid for
            plotting.
            >>> # xdoctest: +REQUIRES(module:matplotlib)
            >>> # xdoctest: +REQUIRES(env:DOCTEST_SHOW)
            >>> import matplotlib.pyplot as plt
            >>> xs = torch.linspace(-5, 5, steps=100)
            >>> ys = torch.linspace(-5, 5, steps=100)
            >>> x, y = torch.meshgrid(xs, ys, indexing='xy')
            >>> z = torch.sin(torch.sqrt(x * x + y * y))
            >>> ax = plt.axes(projection='3d')
            >>> ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
            >>> plt.show()

        .. image:: ../_static/img/meshgrid.png
            :width: 512

        """
        return _meshgrid(*tensors, indexing=indexing)


def _meshgrid(*tensors, indexing: str | None):
    if has_torch_function(tensors):
        return handle_torch_function(meshgrid, tensors, *tensors, indexing=indexing)
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        # the old interface of passing the operands as one list argument
        tensors = tensors[0]  # type: ignore[assignment]

    # Continue allowing call of old method that takes no indexing
    # kwarg for forward compatibility reasons.
    #
    # Remove this two weeks after landing.
    kwargs = {} if indexing is None else {"indexing": indexing}
    return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]


def stft(
    input: Tensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: Tensor | None = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool | None = None,
    return_complex: bool | None = None,
    align_to_window: bool | None = None,
) -> Tensor:
    r"""Short-time Fourier transform (STFT).

    .. warning::
        From version 1.8.0, :attr:`return_complex` must always be given
        explicitly for real inputs and `return_complex=False` has been
        deprecated. Strongly prefer `return_complex=True` as in a future
        pytorch release, this function will only return complex tensors.

        Note that :func:`torch.view_as_real` can be used to recover a real
        tensor with an extra last dimension for real and imaginary components.

    .. warning::
        From version 2.1, a warning will be provided if a :attr:`window` is
        not specified. In a future release, this attribute will be required.
        Not providing a window currently defaults to using a rectangular window,
        which may result in undesirable artifacts. Consider using tapered windows,
        such as :func:`torch.hann_window`.

    The STFT computes the Fourier transform of short overlapping windows of the
    input. This gives frequency components of the signal as they change over
    time. The interface of this function is modeled after (but *not* a drop-in
    replacement for) librosa_ stft function.

    .. _librosa: https://librosa.org/doc/latest/generated/librosa.stft.html

    Ignoring the optional batch dimension, this method computes the following
    expression:

    .. math::
        X[\omega, m] = \sum_{k = 0}^{\text{win\_length-1}}%
                            \text{window}[k]\ \text{input}[m \times \text{hop\_length} + k]\ %
                            \exp\left(- j \frac{2 \pi \cdot \omega k}{\text{n\_fft}}\right),

    where :math:`m` is the index of the sliding window, and :math:`\omega` is
    the frequency :math:`0 \leq \omega < \text{n\_fft}` for ``onesided=False``,
    or :math:`0 \leq \omega < \lfloor \text{n\_fft} / 2 \rfloor + 1` for ``onesided=True``.

    * :attr:`input` must be either a 1-D time sequence or a 2-D batch of time
      sequences.

    * If :attr:`hop_length` is ``None`` (default), it is treated as equal to
      ``floor(n_fft / 4)``.

    * If :attr:`win_length` is ``None`` (default), it is treated as equal to
      :attr:`n_fft`.

    * :attr:`window` can be a 1-D tensor of size :attr:`win_length`, e.g., from
      :meth:`torch.hann_window`. If :attr:`window` is ``None`` (default), it is
      treated as if having :math:`1` everywhere in the window. If
      :math:`\text{win\_length} < \text{n\_fft}`, :attr:`window` will be padded on
      both sides to length :attr:`n_fft` before being applied.

    * If :attr:`center` is ``True`` (default), :attr:`input` will be padded on
      both sides so that the :math:`t`-th frame is centered at time
      :math:`t \times \text{hop\_length}`. Otherwise, the :math:`t`-th frame
      begins at time  :math:`t \times \text{hop\_length}`.

    * :attr:`pad_mode` determines the padding method used on :attr:`input` when
      :attr:`center` is ``True``. See :meth:`torch.nn.functional.pad` for
      all available options. Default is ``"reflect"``.

    * If :attr:`onesided` is ``True`` (default for real input), only values for
      :math:`\omega` in :math:`\left[0, 1, 2, \dots, \left\lfloor
      \frac{\text{n\_fft}}{2} \right\rfloor + 1\right]` are returned because
      the real-to-complex Fourier transform satisfies the conjugate symmetry,
      i.e., :math:`X[m, \omega] = X[m, \text{n\_fft} - \omega]^*`.
      Note if the input or window tensors are complex, then :attr:`onesided`
      output is not possible.

    * If :attr:`normalized` is ``True`` (default is ``False``), the function
      returns the normalized STFT results, i.e., multiplied by :math:`(\text{frame\_length})^{-0.5}`.

    * If :attr:`return_complex` is ``True`` (default if input is complex), the
      return is a ``input.dim() + 1`` dimensional complex tensor. If ``False``,
      the output is a ``input.dim() + 2`` dimensional real tensor where the last
      dimension represents the real and imaginary components.

    Returns either a complex tensor of size :math:`(* \times N \times T)` if
    :attr:`return_complex` is true, or a real tensor of size :math:`(* \times N
    \times T \times 2)`. Where :math:`*` is the optional batch size of
    :attr:`input`, :math:`N` is the number of frequencies where STFT is applied
    and :math:`T` is the total number of frames used.

    .. warning::
      This function changed signature at version 0.4.1. Calling with the
      previous signature may cause error or return incorrect result.

    Args:
        input (Tensor): the input tensor of shape `(B?, L)` where `B?` is an optional
            batch dimension
        n_fft (int): size of Fourier transform
        hop_length (int, optional): the distance between neighboring sliding window
            frames. Default: ``None`` (treated as equal to ``floor(n_fft / 4)``)
        win_length (int, optional): the size of window frame and STFT filter.
            Default: ``None``  (treated as equal to :attr:`n_fft`)
        window (Tensor, optional): the optional window function.
            Shape must be 1d and `<= n_fft`
            Default: ``None`` (treated as window of all :math:`1` s)
        center (bool, optional): whether to pad :attr:`input` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        pad_mode (str, optional): controls the padding method used when
            :attr:`center` is ``True``. Default: ``"reflect"``
        normalized (bool, optional): controls whether to return the normalized STFT results
             Default: ``False``
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy for real inputs.
            Default: ``True`` for real :attr:`input` and :attr:`window`, ``False`` otherwise.
        return_complex (bool, optional): whether to return a complex tensor, or
            a real tensor with an extra last dimension for the real and
            imaginary components.

            .. versionchanged:: 2.0
               ``return_complex`` is now a required argument for real inputs,
               as the default is being transitioned to ``True``.

            .. deprecated:: 2.0
               ``return_complex=False`` is deprecated, instead use ``return_complex=True``
               Note that calling :func:`torch.view_as_real` on the output will
               recover the deprecated output format.

    Returns:
        Tensor: A tensor containing the STFT result with shape `(B?, N, T, C?)` where
           - `B?` is an optional batch dimension from the input.
           - `N` is the number of frequency samples, `(n_fft // 2) + 1` for
             `onesided=True`, or otherwise `n_fft`.
           - `T` is the number of frames, `1 + L // hop_length`
             for `center=True`, or `1 + (L - n_fft) // hop_length` otherwise.
           - `C?` is an optional length-2 dimension of real and imaginary
             components, present when `return_complex=False`.

    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            stft,
            (input,),
            input,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            return_complex=return_complex,
            align_to_window=align_to_window,
        )
    if center and align_to_window is not None:
        raise RuntimeError(
            "stft align_to_window should only be set when center = false"
        )
    # NOTE: Do not edit. This code will be removed once the forward-compatibility
    #       period is over for PR #73432
    if center:
        signal_dim = input.dim()
        extended_shape = [1] * (3 - signal_dim) + list(input.size())
        pad = int(n_fft // 2)
        input = F.pad(input.view(extended_shape), [pad, pad], pad_mode)
        input = input.view(input.shape[-signal_dim:])
    return _VF.stft(  # type: ignore[attr-defined]
        input,
        n_fft,
        hop_length,
        win_length,
        window,
        normalized,
        onesided,
        return_complex,
        align_to_window,
    )


istft = _add_docstr(
    torch.istft,
    "istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, "
    "normalized=False, onesided=None, length=None, return_complex=False) -> Tensor:\n"
    r"""
Inverse short time Fourier Transform. This is expected to be the inverse of :func:`~torch.stft`.

.. warning::
    From version 2.1, a warning will be provided if a :attr:`window` is
    not specified. In a future release, this attribute will be required.
    Please provide the same window used in the stft call.

It has the same parameters (+ additional optional parameter of :attr:`length`) and it should return the
least squares estimation of the original signal. The algorithm will check using the NOLA condition (
nonzero overlap).

Important consideration in the parameters :attr:`window` and :attr:`center` so that the envelope
created by the summation of all the windows is never zero at certain point in time. Specifically,
:math:`\sum_{t=-\infty}^{\infty} |w|^2[n-t\times hop\_length] \cancel{=} 0`.

Since :func:`~torch.stft` discards elements at the end of the signal if they do not fit in a frame,
``istft`` may return a shorter signal than the original signal (can occur if :attr:`center` is False
since the signal isn't padded). If `length` is given in the arguments and is longer than expected,
``istft`` will pad zeros to the end of the returned signal.

If :attr:`center` is ``True``, then there will be padding e.g. ``'constant'``, ``'reflect'``, etc.
Left padding can be trimmed off exactly because they can be calculated but right padding cannot be
calculated without additional information.

Example: Suppose the last window is:
``[17, 18, 0, 0, 0]`` vs ``[18, 0, 0, 0, 0]``

The :attr:`n_fft`, :attr:`hop_length`, :attr:`win_length` are all the same which prevents the calculation
of right padding. These additional values could be zeros or a reflection of the signal so providing
:attr:`length` could be useful. If :attr:`length` is ``None`` then padding will be aggressively removed
(some loss of signal).

[1] D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform,"
IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr. 1984.

Args:
    input (Tensor): The input tensor. Expected to be in the format of :func:`~torch.stft`,
        output. That is a complex tensor of shape `(B?, N, T)` where

        - `B?` is an optional batch dimension
        - `N` is the number of frequency samples, `(n_fft // 2) + 1`
          for onesided input, or otherwise `n_fft`.
        - `T` is the number of frames, `1 + length // hop_length` for centered stft,
          or `1 + (length - n_fft) // hop_length` otherwise.

        .. versionchanged:: 2.0
            Real datatype inputs are no longer supported. Input must now have a
            complex datatype, as returned by ``stft(..., return_complex=True)``.
    n_fft (int): Size of Fourier transform
    hop_length (Optional[int]): The distance between neighboring sliding window frames.
        (Default: ``n_fft // 4``)
    win_length (Optional[int]): The size of window frame and STFT filter. (Default: ``n_fft``)
    window (Optional[torch.Tensor]): The optional window function.
        Shape must be 1d and `<= n_fft`
        (Default: ``torch.ones(win_length)``)
    center (bool): Whether :attr:`input` was padded on both sides so that the :math:`t`-th frame is
        centered at time :math:`t \times \text{hop\_length}`.
        (Default: ``True``)
    normalized (bool): Whether the STFT was normalized. (Default: ``False``)
    onesided (Optional[bool]): Whether the STFT was onesided.
        (Default: ``True`` if `n_fft != fft_size` in the input size)
    length (Optional[int]): The amount to trim the signal by (i.e. the
        original signal length). Defaults to `(T - 1) * hop_length` for
        centered stft, or `n_fft + (T - 1) * hop_length` otherwise, where `T`
        is the number of input frames.
    return_complex (Optional[bool]):
        Whether the output should be complex, or if the input should be
        assumed to derive from a real signal and window.
        Note that this is incompatible with ``onesided=True``.
        (Default: ``False``)

Returns:
    Tensor: Least squares estimation of the original signal of shape `(B?, length)` where
        `B?` is an optional batch dimension from the input tensor.
""",
)


if TYPE_CHECKING:
    # These _impl functions return a variable number of tensors as output with
    # __torch_function__; tuple unpacking is done already rather than being
    # done by the caller of the _impl function
    _unique_impl_out = Any
else:
    _unique_impl_out = tuple[Tensor, Tensor, Tensor]


def _unique_torch_function(
    input: Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> Any:
    return handle_torch_function(
        unique,
        (input,),
        input,
        sorted=sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
        dim=dim,
    )


def _unique_impl(
    input: Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> _unique_impl_out:
    r"""unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> tuple[Tensor, Tensor, Tensor]

    Returns the unique elements of the input tensor.

    .. note:: This function is different from :func:`torch.unique_consecutive` in the sense that
        this function also eliminates non-consecutive duplicate values.

    .. note:: Currently in the CUDA implementation and the CPU implementation,
        `torch.unique` always sort the tensor at the beginning regardless of the `sort` argument.
        Sorting could be slow, so if your input tensor is already sorted, it is recommended to use
        :func:`torch.unique_consecutive` which avoids the sorting.

    Args:
        input (Tensor): the input tensor
        sorted (bool): Whether to sort the unique elements in ascending order
            before returning as output.
        return_inverse (bool): Whether to also return the indices for where
            elements in the original input ended up in the returned unique list.
        return_counts (bool): Whether to also return the counts for each unique
            element.
        dim (int, optional): the dimension to operate upon. If ``None``, the
            unique of the flattened input is returned. Otherwise, each of the
            tensors indexed by the given dimension is treated as one of the
            elements to apply the unique operation upon. **Important:** when ``dim``
            is specified, the operation finds unique sub-tensors (e.g., unique rows
            or columns), not unique scalar values. This means individual values may
            appear multiple times in the output if they exist in different sub-tensors.
            See examples for more details. Default: ``None``

    Returns:
        (Tensor, Tensor (optional), Tensor (optional)): A tensor or a tuple of tensors containing

            - **output** (*Tensor*): the output list of unique scalar elements if :attr:`dim`
              is ``None``; otherwise, the unique sub-tensors along the specified dimension.
              Note that when :attr:`dim` is specified, scalar values may repeat in the output.
            - **inverse_indices** (*Tensor*): (optional) if
              :attr:`return_inverse` is True, there will be an additional
              returned tensor (same shape as input) representing the indices
              for where elements in the original input map to in the output;
              otherwise, this function will only return a single tensor.
            - **counts** (*Tensor*): (optional) if
              :attr:`return_counts` is True, there will be an additional
              returned tensor (same shape as output or output.size(dim),
              if dim was specified) representing the number of occurrences
              for each unique value or tensor.

    Example::

        >>> output = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long))
        >>> output
        tensor([1, 2, 3])

        >>> output, inverse_indices = torch.unique(
        ...     torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=True, return_inverse=True)
        >>> output
        tensor([1, 2, 3])
        >>> inverse_indices
        tensor([0, 2, 1, 2])

        >>> output, inverse_indices = torch.unique(
        ...     torch.tensor([[1, 3], [2, 3]], dtype=torch.long), sorted=True, return_inverse=True)
        >>> output
        tensor([1, 2, 3])
        >>> inverse_indices
        tensor([[0, 2],
                [1, 2]])

        >>> # When using dim, the operation finds unique sub-tensors, not unique values.
        >>> # Notice how values can repeat in the output:
        >>> x = torch.tensor([[1, 3, 2, 3], [1, 2, 1, 2]], dtype=torch.long)
        >>> torch.unique(x, dim=0)  # unique rows
        tensor([[1, 2, 1, 2],
                [1, 3, 2, 3]])
        >>> # Both rows are kept because they're different from each other,
        >>> # even though values 1, 2, 3 appear multiple times in the output.
        >>> torch.unique(x, dim=1)  # unique columns
        tensor([[1, 2, 3],
                [1, 1, 2]])
        >>> # The value 1 appears twice because we're comparing columns, not values.
        >>> # Compare with flattened (no dim):
        >>> torch.unique(x)
        tensor([1, 2, 3])

        >>> a = torch.tensor([
        ...     [
        ...         [1, 1, 0, 0],
        ...         [1, 1, 0, 0],
        ...         [0, 0, 1, 1],
        ...     ],
        ...     [
        ...         [0, 0, 1, 1],
        ...         [0, 0, 1, 1],
        ...         [1, 1, 1, 1],
        ...     ],
        ...     [
        ...         [1, 1, 0, 0],
        ...         [1, 1, 0, 0],
        ...         [0, 0, 1, 1],
        ...     ],
        ... ])

        >>> # If we call `torch.unique(a, dim=0)`, each of the tensors `a[idx, :, :]`
        >>> # will be compared. We can see that `a[0, :, :]` and `a[2, :, :]` match
        >>> # each other, so one of them will be removed.
        >>> (a[0, :, :] == a[2, :, :]).all()
        tensor(True)
        >>> a_unique_dim0 = torch.unique(a, dim=0)
        >>> a_unique_dim0
        tensor([[[0, 0, 1, 1],
                 [0, 0, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0],
                 [0, 0, 1, 1]]])

        >>> # Notice which sub-tensors from `a` match with the sub-tensors from
        >>> # `a_unique_dim0`:
        >>> (a_unique_dim0[0, :, :] == a[1, :, :]).all()
        tensor(True)
        >>> (a_unique_dim0[1, :, :] == a[0, :, :]).all()
        tensor(True)

        >>> # For `torch.unique(a, dim=1)`, each of the tensors `a[:, idx, :]` are
        >>> # compared. `a[:, 0, :]` and `a[:, 1, :]` match each other, so one of
        >>> # them will be removed.
        >>> (a[:, 0, :] == a[:, 1, :]).all()
        tensor(True)
        >>> torch.unique(a, dim=1)
        tensor([[[0, 0, 1, 1],
                 [1, 1, 0, 0]],
                [[1, 1, 1, 1],
                 [0, 0, 1, 1]],
                [[0, 0, 1, 1],
                 [1, 1, 0, 0]]])

        >>> # For `torch.unique(a, dim=2)`, the tensors `a[:, :, idx]` are compared.
        >>> # `a[:, :, 0]` and `a[:, :, 1]` match each other. Also, `a[:, :, 2]` and
        >>> # `a[:, :, 3]` match each other as well. So in this case, two of the
        >>> # sub-tensors will be removed.
        >>> (a[:, :, 0] == a[:, :, 1]).all()
        tensor(True)
        >>> (a[:, :, 2] == a[:, :, 3]).all()
        tensor(True)
        >>> torch.unique(a, dim=2)
        tensor([[[0, 1],
                 [0, 1],
                 [1, 0]],
                [[1, 0],
                 [1, 0],
                 [1, 1]],
                [[0, 1],
                 [0, 1],
                 [1, 0]]])
    """
    if dim is not None:
        output, inverse_indices, counts = _VF.unique_dim(
            input,
            dim,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )
    else:
        output, inverse_indices, counts = torch._unique2(
            input,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )
    return output, inverse_indices, counts


def _unique_consecutive_torch_function(
    input: Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> Any:
    return handle_torch_function(
        unique_consecutive,
        (input,),
        input,
        return_inverse=return_inverse,
        return_counts=return_counts,
        dim=dim,
    )


def _unique_consecutive_impl(
    input: Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> _unique_impl_out:
    r"""Eliminates all but the first element from every consecutive group of equivalent elements.

    .. note:: This function is different from :func:`torch.unique` in the sense that this function
        only eliminates consecutive duplicate values. This semantics is similar to `std::unique`
        in C++.

    Args:
        input (Tensor): the input tensor
        return_inverse (bool): Whether to also return the indices for where
            elements in the original input ended up in the returned unique list.
        return_counts (bool): Whether to also return the counts for each unique
            element.
        dim (int): the dimension to apply unique. If ``None``, the unique of the
            flattened input is returned. default: ``None``

    Returns:
        (Tensor, Tensor (optional), Tensor (optional)): A tensor or a tuple of tensors containing

            - **output** (*Tensor*): the output list of unique scalar elements.
            - **inverse_indices** (*Tensor*): (optional) if
              :attr:`return_inverse` is True, there will be an additional
              returned tensor (same shape as input) representing the indices
              for where elements in the original input map to in the output;
              otherwise, this function will only return a single tensor.
            - **counts** (*Tensor*): (optional) if
              :attr:`return_counts` is True, there will be an additional
              returned tensor (same shape as output or output.size(dim),
              if dim was specified) representing the number of occurrences
              for each unique value or tensor.

    Example::

        >>> x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
        >>> output = torch.unique_consecutive(x)
        >>> output
        tensor([1, 2, 3, 1, 2])

        >>> output, inverse_indices = torch.unique_consecutive(x, return_inverse=True)
        >>> output
        tensor([1, 2, 3, 1, 2])
        >>> inverse_indices
        tensor([0, 0, 1, 1, 2, 3, 3, 4])

        >>> output, counts = torch.unique_consecutive(x, return_counts=True)
        >>> output
        tensor([1, 2, 3, 1, 2])
        >>> counts
        tensor([2, 2, 1, 2, 1])
    """
    output, inverse_indices, counts = _VF.unique_consecutive(  # type: ignore[attr-defined]
        input, return_inverse=return_inverse, return_counts=return_counts, dim=dim
    )
    return output, inverse_indices, counts


def _return_inverse_and_counts(
    input,
    sorted=True,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, bool, Optional[int]) -> tuple[Tensor, Tensor, Tensor]

    if has_torch_function_unary(input):
        return _unique_torch_function(input, sorted, return_inverse, return_counts, dim)

    return _unique_impl(input, sorted, return_inverse, return_counts, dim)


def _return_counts(
    input,
    sorted=True,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, bool, Optional[int]) -> tuple[Tensor, Tensor]

    if has_torch_function_unary(input):
        return _unique_torch_function(input, sorted, return_inverse, return_counts, dim)

    output, _, counts = _unique_impl(input, sorted, return_inverse, return_counts, dim)
    return output, counts


def _return_output(
    input,
    sorted=True,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, bool, Optional[int]) -> Tensor

    if has_torch_function_unary(input):
        return _unique_torch_function(input, sorted, return_inverse, return_counts, dim)

    output, _, _ = _unique_impl(input, sorted, return_inverse, return_counts, dim)
    return output


def _return_inverse(
    input,
    sorted=True,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, bool, Optional[int]) -> tuple[Tensor, Tensor]

    if has_torch_function_unary(input):
        return _unique_torch_function(input, sorted, return_inverse, return_counts, dim)

    output, inverse_indices, _ = _unique_impl(
        input, sorted, return_inverse, return_counts, dim
    )
    return output, inverse_indices


_return_inverse_false = boolean_dispatch(
    arg_name="return_counts",
    arg_index=3,
    default=False,
    if_true=_return_counts,
    if_false=_return_output,
    module_name=__name__,
    func_name="unique",
)

_return_inverse_true = boolean_dispatch(
    arg_name="return_counts",
    arg_index=3,
    default=False,
    if_true=_return_inverse_and_counts,
    if_false=_return_inverse,
    module_name=__name__,
    func_name="unique",
)

# The return type of unique depends on `return_inverse`, and `return_counts` so in order to
# resolve the output type in TorchScript we need to statically know the value of both parameters

unique = boolean_dispatch(
    arg_name="return_inverse",
    arg_index=2,
    default=False,
    if_true=_return_inverse_true,
    if_false=_return_inverse_false,
    module_name=__name__,
    func_name="unique",
)
unique.__doc__ = _unique_impl.__doc__


def _consecutive_return_inverse_and_counts(
    input,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, Optional[int]) -> tuple[Tensor, Tensor, Tensor]
    if has_torch_function_unary(input):
        return _unique_consecutive_torch_function(
            input, return_inverse, return_counts, dim
        )

    return _unique_consecutive_impl(input, return_inverse, return_counts, dim)


def _consecutive_return_counts(
    input,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, Optional[int]) -> tuple[Tensor, Tensor]

    if has_torch_function_unary(input):
        return _unique_consecutive_torch_function(
            input, return_inverse, return_counts, dim
        )

    output, _, counts = _unique_consecutive_impl(
        input, return_inverse, return_counts, dim
    )
    return output, counts


def _consecutive_return_output(
    input,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, Optional[int]) -> Tensor

    if has_torch_function_unary(input):
        return _unique_consecutive_torch_function(
            input, return_inverse, return_counts, dim
        )

    output, _, _ = _unique_consecutive_impl(input, return_inverse, return_counts, dim)
    return output


def _consecutive_return_inverse(
    input,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, Optional[int]) -> tuple[Tensor, Tensor]

    if has_torch_function_unary(input):
        return _unique_consecutive_torch_function(
            input, return_inverse, return_counts, dim
        )

    output, inverse_indices, _ = _unique_consecutive_impl(
        input, return_inverse, return_counts, dim
    )
    return output, inverse_indices


_consecutive_return_inverse_false = boolean_dispatch(
    arg_name="return_counts",
    arg_index=1,
    default=False,
    if_true=_consecutive_return_counts,
    if_false=_consecutive_return_output,
    module_name=__name__,
    func_name="unique_consecutive",
)

_consecutive_return_inverse_true = boolean_dispatch(
    arg_name="return_counts",
    arg_index=1,
    default=False,
    if_true=_consecutive_return_inverse_and_counts,
    if_false=_consecutive_return_inverse,
    module_name=__name__,
    func_name="unique_consecutive",
)

# The return type of unique depends on `return_inverse`, and `return_counts` so in order to
# resolve the output type in TorchScript we need to statically know the value of both parameters

unique_consecutive = boolean_dispatch(
    arg_name="return_inverse",
    arg_index=2,
    default=False,
    if_true=_consecutive_return_inverse_true,
    if_false=_consecutive_return_inverse_false,
    module_name=__name__,
    func_name="unique_consecutive",
)
unique_consecutive.__doc__ = _unique_consecutive_impl.__doc__

if TYPE_CHECKING:
    pass
    # There's no good way to use this type annotation without breaking JIT
    # overloads. So leave untyped for mypy for now.
else:

    @overload
    def tensordot(
        a,
        b,
        dims: int = 2,
        out: torch.Tensor | None = None,
    ):
        pass

    @overload
    def tensordot(  # noqa: F811
        a,
        b,
        dims: tuple[list[int], list[int]],
        out: torch.Tensor | None = None,
    ):
        pass

    @overload
    def tensordot(  # noqa: F811
        a,
        b,
        dims: list[list[int]],
        out: torch.Tensor | None = None,
    ):
        pass

    @overload
    def tensordot(  # noqa: F811
        a,
        b,
        dims: torch.Tensor,
        out: torch.Tensor | None = None,
    ):
        pass


def tensordot(
    a,
    b,
    dims=2,
    out: torch.Tensor | None = None,
):
    r"""Returns a contraction of a and b over multiple dimensions.

    :attr:`tensordot` implements a generalized matrix product.

    Args:
      a (Tensor): Left tensor to contract
      b (Tensor): Right tensor to contract
      dims (int or Tuple[List[int], List[int]] or List[List[int]] containing two lists or Tensor): number of dimensions to
         contract or explicit lists of dimensions for :attr:`a` and
         :attr:`b` respectively

    When called with a non-negative integer argument :attr:`dims` = :math:`d`, and
    the number of dimensions of :attr:`a` and :attr:`b` is :math:`m` and :math:`n`,
    respectively, :func:`~torch.tensordot` computes the tensor :math:`r` of shape
    ``a.shape[:-dims] + b.shape[dims:]`` given by:

    .. math::
        r_{i_1,...,i_{m-d}, j_1,...,j_{n-d}}
          = \sum_{k_1,...,k_d} a_{i_1,...,i_{m-d},k_1,...,k_d} \times b_{k_1,...,k_d, j_1,...,j_{n-d}}.

    When called with :attr:`dims` of the list form, the given dimensions will be contracted
    in place of the last :math:`d` of :attr:`a` and the first :math:`d` of :math:`b`. The sizes
    in these dimensions must match, but :func:`~torch.tensordot` will deal with broadcasted
    dimensions.

    Examples::

        >>> a = torch.arange(60.).reshape(3, 4, 5)
        >>> b = torch.arange(24.).reshape(4, 3, 2)
        >>> torch.tensordot(a, b, dims=([1, 0], [0, 1]))
        tensor([[4400., 4730.],
                [4532., 4874.],
                [4664., 5018.],
                [4796., 5162.],
                [4928., 5306.]])

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> a = torch.randn(3, 4, 5, device='cuda')
        >>> b = torch.randn(4, 5, 6, device='cuda')
        >>> c = torch.tensordot(a, b, dims=2).cpu()
        tensor([[ 8.3504, -2.5436,  6.2922,  2.7556, -1.0732,  3.2741],
                [ 3.3161,  0.0704,  5.0187, -0.4079, -4.3126,  4.8744],
                [ 0.8223,  3.9445,  3.2168, -0.2400,  3.4117,  1.7780]])

        >>> a = torch.randn(3, 5, 4, 6)
        >>> b = torch.randn(6, 4, 5, 3)
        >>> torch.tensordot(a, b, dims=([2, 1, 3], [1, 2, 0]))
        tensor([[  7.7193,  -2.4867, -10.3204],
                [  1.5513, -14.4737,  -6.5113],
                [ -0.2850,   4.2573,  -3.5997]])
    """
    if has_torch_function_variadic(a, b):
        return handle_torch_function(tensordot, (a, b), a, b, dims=dims, out=out)

    if not isinstance(dims, (tuple, list, torch.Tensor, int, torch.SymInt)):
        raise RuntimeError(
            "tensordot expects dims to be int or "
            + "tuple[list[int], list[int]] or "
            + "list[list[int]] containing two lists, but got "
            + f"dims={dims}"
        )

    dims_a: list[int] = []
    dims_b: list[int] = []

    if isinstance(dims, (tuple, list)):
        dims_a, dims_b = dims

    if isinstance(dims, torch.Tensor):
        num_elements = dims.numel()
        if num_elements > 1:
            if dims.size()[0] != 2:
                raise AssertionError(
                    f"dims tensor must have size 2 in first dimension, got {dims.size()[0]}"
                )
            dims_a = torch.jit.annotate(list[int], dims[0].tolist())
            dims_b = torch.jit.annotate(list[int], dims[1].tolist())
        else:
            dims_val = int(dims.item())
            if dims_val < 0:
                raise RuntimeError(f"tensordot expects dims >= 0, but got dims={dims}")
            dims_a = list(range(-dims_val, 0))
            dims_b = list(range(dims_val))

    if isinstance(dims, (int, torch.SymInt)):
        if dims < 0:
            raise RuntimeError(f"tensordot expects dims >= 0, but got dims={dims}")
        if dims > min(a.dim(), b.dim()):
            raise RuntimeError(
                f"tensordot expects dims < ndim_a or ndim_b, but got dims={dims}"
            )
        dims_a = list(range(-dims, 0))
        dims_b = list(range(dims))

    if out is None:
        return _VF.tensordot(a, b, dims_a, dims_b)  # type: ignore[attr-defined]
    else:
        return _VF.tensordot(a, b, dims_a, dims_b, out=out)  # type: ignore[attr-defined]


def cartesian_prod(*tensors: Tensor) -> Tensor:
    """Do cartesian product of the given sequence of tensors. The behavior is similar to
    python's `itertools.product`.

    Args:
        *tensors: any number of 1 dimensional tensors.

    Returns:
        Tensor: A tensor equivalent to converting all the input tensors into lists,
        do `itertools.product` on these lists, and finally convert the resulting list
        into tensor.

    Example::

        >>> import itertools
        >>> a = [1, 2, 3]
        >>> b = [4, 5]
        >>> list(itertools.product(a, b))
        [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
        >>> tensor_a = torch.tensor(a)
        >>> tensor_b = torch.tensor(b)
        >>> torch.cartesian_prod(tensor_a, tensor_b)
        tensor([[1, 4],
                [1, 5],
                [2, 4],
                [2, 5],
                [3, 4],
                [3, 5]])
    """
    # This wrapper exists to support variadic args.
    if has_torch_function(tensors):
        return handle_torch_function(cartesian_prod, tensors, *tensors)
    return _VF.cartesian_prod(tensors)  # type: ignore[attr-defined]


def block_diag(*tensors):
    """Create a block diagonal matrix from provided tensors.

    Args:
        *tensors: One or more tensors with 0, 1, or 2 dimensions.

    Returns:
        Tensor: A 2 dimensional tensor with all the input tensors arranged in
        order such that their upper left and lower right corners are
        diagonally adjacent. All other elements are set to 0.

    Example::

        >>> import torch
        >>> A = torch.tensor([[0, 1], [1, 0]])
        >>> B = torch.tensor([[3, 4, 5], [6, 7, 8]])
        >>> C = torch.tensor(7)
        >>> D = torch.tensor([1, 2, 3])
        >>> E = torch.tensor([[4], [5], [6]])
        >>> torch.block_diag(A, B, C, D, E)
        tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 4, 5, 0, 0, 0, 0, 0],
                [0, 0, 6, 7, 8, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 3, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 6]])
    """
    # This wrapper exists to support variadic args.
    if has_torch_function(tensors):
        return handle_torch_function(block_diag, tensors, *tensors)
    return torch._C._VariableFunctions.block_diag(tensors)  # type: ignore[attr-defined]


def cdist(x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    # type: (Tensor, Tensor, float, str) -> (Tensor)
    r"""Computes the batched p-norm distance between each pair of the two collections of row vectors.

    Args:
        x1 (Tensor): input tensor where the last two dimensions represent the points and the feature dimension respectively.
            The shape can be :math:`D_1 \times D_2 \times \cdots \times D_n \times P \times M`,
            where :math:`P` is the number of points and :math:`M` is the feature dimension.
        x2 (Tensor): input tensor where the last two dimensions also represent the points and the feature dimension respectively.
            The shape can be :math:`D_1' \times D_2' \times \cdots \times D_m' \times R \times M`,
            where :math:`R` is the number of points and :math:`M` is the feature dimension,
            which should match the feature dimension of `x1`.
        p: p value for the p-norm distance to calculate between each vector pair
            :math:`\in [0, \infty]`.
        compute_mode:
            'use_mm_for_euclid_dist_if_necessary' - will use matrix multiplication approach to calculate
            euclidean distance (p = 2) if P > 25 or R > 25
            'use_mm_for_euclid_dist' - will always use matrix multiplication approach to calculate
            euclidean distance (p = 2)
            'donot_use_mm_for_euclid_dist' - will never use matrix multiplication approach to calculate
            euclidean distance (p = 2)
            Default: use_mm_for_euclid_dist_if_necessary.

    If x1 has shape :math:`B \times P \times M` and x2 has shape :math:`B \times R \times M` then the
    output will have shape :math:`B \times P \times R`.

    This function is equivalent to `scipy.spatial.distance.cdist(input,'minkowski', p=p)`
    if :math:`p \in (0, \infty)`. When :math:`p = 0` it is equivalent to
    `scipy.spatial.distance.cdist(input, 'hamming') * M`. When :math:`p = \infty`, the closest
    scipy function is `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.

    Example:

        >>> a = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
        >>> a
        tensor([[ 0.9041,  0.0196],
                [-0.3108, -2.4423],
                [-0.4821,  1.0590]])
        >>> b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
        >>> b
        tensor([[-2.1763, -0.4713],
                [-0.6986,  1.3702]])
        >>> torch.cdist(a, b, p=2)
        tensor([[3.1193, 2.0959],
                [2.7138, 3.8322],
                [2.2830, 0.3791]])
    """
    if has_torch_function_variadic(x1, x2):
        return handle_torch_function(
            cdist, (x1, x2), x1, x2, p=p, compute_mode=compute_mode
        )
    if compute_mode == "use_mm_for_euclid_dist_if_necessary":
        return _VF.cdist(x1, x2, p, None)  # type: ignore[attr-defined]
    elif compute_mode == "use_mm_for_euclid_dist":
        return _VF.cdist(x1, x2, p, 1)  # type: ignore[attr-defined]
    elif compute_mode == "donot_use_mm_for_euclid_dist":
        return _VF.cdist(x1, x2, p, 2)  # type: ignore[attr-defined]
    else:
        raise ValueError(f"{compute_mode} is not a valid value for compute_mode")


def atleast_1d(*tensors):
    r"""
    Returns a 1-dimensional view of each input tensor with zero dimensions.
    Input tensors with one or more dimensions are returned as-is.

    Args:
        input (Tensor or sequence of Tensors): tensor(s) to be converted to at least 1-dimensional.

    Returns:
        output (Tensor or tuple of Tensors)

    Example::

        >>> x = torch.arange(2)
        >>> x
        tensor([0, 1])
        >>> torch.atleast_1d(x)
        tensor([0, 1])
        >>> x = torch.tensor(1.)
        >>> x
        tensor(1.)
        >>> torch.atleast_1d(x)
        tensor([1.])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.)
        >>> torch.atleast_1d((x, y))
        (tensor([0.5000]), tensor([1.]))
        >>> torch.atleast_1d()
        ()
    """
    # This wrapper exists to support variadic args.
    if has_torch_function(tensors):
        return handle_torch_function(atleast_1d, tensors, *tensors)
    if len(tensors) == 1:
        tensors = tensors[0]
    return _VF.atleast_1d(tensors)  # type: ignore[attr-defined]


def atleast_2d(*tensors):
    r"""
    Returns a 2-dimensional view of each input tensor with zero dimensions.
    Input tensors with two or more dimensions are returned as-is.

    Args:
        input (Tensor or sequence of Tensors): tensor(s) to be converted to at least 2-dimensional.

    Returns:
        output (Tensor or tuple of Tensors)

    Example::

        >>> x = torch.tensor(1.)
        >>> x
        tensor(1.)
        >>> torch.atleast_2d(x)
        tensor([[1.]])
        >>> x = torch.arange(4).view(2, 2)
        >>> x
        tensor([[0, 1],
                [2, 3]])
        >>> torch.atleast_2d(x)
        tensor([[0, 1],
                [2, 3]])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.)
        >>> torch.atleast_2d((x, y))
        (tensor([[0.5000]]), tensor([[1.]]))
        >>> torch.atleast_2d()
        ()
    """
    # This wrapper exists to support variadic args.
    if has_torch_function(tensors):
        return handle_torch_function(atleast_2d, tensors, *tensors)
    if len(tensors) == 1:
        tensors = tensors[0]
    return _VF.atleast_2d(tensors)  # type: ignore[attr-defined]


def atleast_3d(*tensors):
    r"""
    Returns a 3-dimensional view of each input tensor with zero dimensions.
    Input tensors with three or more dimensions are returned as-is.

    Args:
        input (Tensor or sequence of Tensors): tensor(s) to be converted to at least 3-dimensional.

    Returns:
        output (Tensor or tuple of Tensors)

    Example:

        >>> x = torch.tensor(0.5)
        >>> x
        tensor(0.5000)
        >>> torch.atleast_3d(x)
        tensor([[[0.5000]]])
        >>> y = torch.arange(4).view(2, 2)
        >>> y
        tensor([[0, 1],
                [2, 3]])
        >>> torch.atleast_3d(y)
        tensor([[[0],
                 [1]],
                <BLANKLINE>
                [[2],
                 [3]]])
        >>> x = torch.tensor(1).view(1, 1, 1)
        >>> x
        tensor([[[1]]])
        >>> torch.atleast_3d(x)
        tensor([[[1]]])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.0)
        >>> torch.atleast_3d((x, y))
        (tensor([[[0.5000]]]), tensor([[[1.]]]))
        >>> torch.atleast_3d()
        ()
    """
    # This wrapper exists to support variadic args.
    if has_torch_function(tensors):
        return handle_torch_function(atleast_3d, tensors, *tensors)
    if len(tensors) == 1:
        tensors = tensors[0]
    return _VF.atleast_3d(tensors)  # type: ignore[attr-defined]


if TYPE_CHECKING:
    pass
    # There's no good way to use this type annotation; cannot rename norm() to
    # _norm_impl() in a way that doesn't break JIT overloads. So leave untyped
    # for mypy for now.
    #    def norm(input: Tensor,
    #             p: Optional[Union[str, Number]] = "fro",
    #             dim: Optional[Union[int, List[int]]] = None,
    #             keepdim: bool = False,
    #             out: Optional[Tensor] = None,
    #             dtype: _dtype = None) -> Tensor:
    #        return _norm_impl(input, p, dim, keepdim, out, dtype)
else:
    # TODO: type dim as BroadcastingList when
    # https://github.com/pytorch/pytorch/issues/33782 is fixed
    @overload
    def norm(
        input,
        p="fro",
        dim=None,
        keepdim=False,
        out=None,
        dtype=None,
    ):
        # type: (Tensor, str, Optional[List[int]], bool, Optional[Tensor], Optional[int]) -> Tensor
        pass

    @overload
    def norm(  # noqa: F811
        input,
        p="fro",
        dim=None,
        keepdim=False,
        out=None,
        dtype=None,
    ):
        # type: (Tensor, Optional[number], Optional[List[int]], bool, Optional[Tensor], Optional[int]) -> Tensor
        pass

    @overload
    def norm(  # noqa: F811
        input,
        p="fro",
        dim=None,
        keepdim=False,
        out=None,
        dtype=None,
    ):
        # type: (Tensor, Optional[number], Optional[int], bool, Optional[Tensor], Optional[int]) -> Tensor
        pass

    @overload
    def norm(  # noqa: F811
        input,
        p="fro",
        dim=None,
        keepdim=False,
        out=None,
        dtype=None,
    ):
        # type: (Tensor, str, Optional[int], bool, Optional[Tensor], Optional[int]) -> Tensor
        pass


def norm(
    input,
    p: float | str | None = "fro",
    dim=None,
    keepdim=False,
    out=None,
    dtype=None,
):
    r"""Returns the matrix norm or vector norm of a given tensor.

    .. warning::

        torch.norm is deprecated and may be removed in a future PyTorch release.
        Its documentation and behavior may be incorrect, and it is no longer
        actively maintained.

        Use :func:`torch.linalg.vector_norm` when computing vector norms and
        :func:`torch.linalg.matrix_norm` when computing matrix norms.
        For a function with a similar behavior as this one see :func:`torch.linalg.norm`.
        Note, however, the signature for these functions is slightly different than the
        signature for ``torch.norm``.

    Args:
        input (Tensor): The input tensor. Its data type must be either a floating
            point or complex type. For complex inputs, the norm is calculated using the
            absolute value of each element. If the input is complex and neither
            :attr:`dtype` nor :attr:`out` is specified, the result's data type will
            be the corresponding floating point type (e.g. float if :attr:`input` is
            complexfloat).

        p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm. Default: ``'fro'``
            The following norms can be calculated:

            ======  ==============  ==========================
            ord     matrix norm     vector norm
            ======  ==============  ==========================
            'fro'   Frobenius norm  --
            'nuc'   nuclear norm    --
            Number  --              sum(abs(x)**ord)**(1./ord)
            ======  ==============  ==========================

            The vector norm can be calculated across any number of dimensions.
            The corresponding dimensions of :attr:`input` are flattened into
            one dimension, and the norm is calculated on the flattened
            dimension.

            Frobenius norm produces the same result as ``p=2`` in all cases
            except when :attr:`dim` is a list of three or more dims, in which
            case Frobenius norm throws an error.

            Nuclear norm can only be calculated across exactly two dimensions.

        dim (int, tuple of ints, list of ints, optional):
            Specifies which dimension or dimensions of :attr:`input` to
            calculate the norm across. If :attr:`dim` is ``None``, the norm will
            be calculated across all dimensions of :attr:`input`. If the norm
            type indicated by :attr:`p` does not support the specified number of
            dimensions, an error will occur.
        keepdim (bool, optional): whether the output tensors have :attr:`dim`
            retained or not. Ignored if :attr:`dim` = ``None`` and
            :attr:`out` = ``None``. Default: ``False``
        out (Tensor, optional): the output tensor. Ignored if
            :attr:`dim` = ``None`` and :attr:`out` = ``None``.
        dtype (:class:`torch.dtype`, optional): the desired data type of
            returned tensor. If specified, the input tensor is casted to
            :attr:`dtype` while performing the operation. Default: None.

    .. note::
        Even though ``p='fro'`` supports any number of dimensions, the true
        mathematical definition of Frobenius norm only applies to tensors with
        exactly two dimensions. :func:`torch.linalg.matrix_norm` with ``ord='fro'``
        aligns with the mathematical definition, since it can only be applied across
        exactly two dimensions.

    Example::

        >>> import torch
        >>> a = torch.arange(9, dtype= torch.float) - 4
        >>> b = a.reshape((3, 3))
        >>> torch.norm(a)
        tensor(7.7460)
        >>> torch.norm(b)
        tensor(7.7460)
        >>> torch.norm(a, float('inf'))
        tensor(4.)
        >>> torch.norm(b, float('inf'))
        tensor(4.)
        >>> c = torch.tensor([[ 1, 2, 3], [-1, 1, 4]] , dtype=torch.float)
        >>> torch.norm(c, dim=0)
        tensor([1.4142, 2.2361, 5.0000])
        >>> torch.norm(c, dim=1)
        tensor([3.7417, 4.2426])
        >>> torch.norm(c, p=1, dim=1)
        tensor([6., 6.])
        >>> d = torch.arange(8, dtype=torch.float).reshape(2, 2, 2)
        >>> torch.norm(d, dim=(1, 2))
        tensor([ 3.7417, 11.2250])
        >>> torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
        (tensor(3.7417), tensor(11.2250))
    """

    if has_torch_function_unary(input):
        return handle_torch_function(
            norm, (input,), input, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype
        )

    # NB. All the repeated code and weird python is to please TorchScript.
    #     For a more compact implementation see the relevant function in `_refs/__init__.py`

    # We don't do this for MPS or sparse tensors
    if input.layout == torch.strided and input.device.type in (
        "cpu",
        "cuda",
        "xpu",
        "meta",
        torch.utils.backend_registration._privateuse1_backend_name,
    ):
        if dim is not None:
            if isinstance(dim, (int, torch.SymInt)):
                _dim = [dim]
            else:
                _dim = dim
        else:
            _dim = None  # type: ignore[assignment]

        if isinstance(p, str):
            if p == "fro" and (
                dim is None
                or isinstance(dim, (int, torch.SymInt))
                or len(dim) <= 2  # pyrefly: ignore  # bad-argument-type
            ):
                if out is None:
                    return torch.linalg.vector_norm(
                        input, 2, _dim, keepdim, dtype=dtype
                    )
                else:
                    return torch.linalg.vector_norm(
                        input, 2, _dim, keepdim, dtype=dtype, out=out
                    )

            # Here we either call the nuclear norm, or we call matrix_norm with some arguments
            # that will throw an error
            if _dim is None:
                _dim = list(range(input.ndim))
            if out is None:
                return torch.linalg.matrix_norm(input, p, _dim, keepdim, dtype=dtype)
            else:
                return torch.linalg.matrix_norm(
                    input, p, _dim, keepdim, dtype=dtype, out=out
                )
        else:
            # NB. p should be Union[str, number], not Optional!
            _p = 2.0 if p is None else p
            if out is None:
                return torch.linalg.vector_norm(input, _p, _dim, keepdim, dtype=dtype)
            else:
                return torch.linalg.vector_norm(
                    input, _p, _dim, keepdim, dtype=dtype, out=out
                )

    ndim = input.dim()

    # catch default case
    if dim is None and out is None and dtype is None and p is not None:
        if isinstance(p, str):
            if p == "fro":
                return _VF.frobenius_norm(input, dim=(), keepdim=keepdim)
        if not isinstance(p, str):
            _dim = list(range(ndim))
            return _VF.norm(input, p, dim=_dim, keepdim=keepdim)  # type: ignore[attr-defined]

    # TODO: when https://github.com/pytorch/pytorch/issues/33782 is fixed
    # remove the overloads where dim is an int and replace with BroadcastingList1
    # and remove next four lines, replace _dim with dim
    if dim is not None:
        if isinstance(dim, (int, torch.SymInt)):
            _dim = [dim]
        else:
            _dim = dim
    else:
        _dim = None  # type: ignore[assignment]

    if isinstance(p, str):
        if p == "fro":
            if dtype is not None:
                raise ValueError("dtype argument is not supported in frobenius norm")

            if _dim is None:
                _dim = list(range(ndim))
            if out is None:
                return _VF.frobenius_norm(input, _dim, keepdim=keepdim)  # type: ignore[arg-type]
            else:
                return _VF.frobenius_norm(input, _dim, keepdim=keepdim, out=out)  # type: ignore[arg-type]
        elif p == "nuc":
            if dtype is not None:
                raise ValueError("dtype argument is not supported in nuclear norm")
            if _dim is None:
                if out is None:
                    return _VF.nuclear_norm(input, keepdim=keepdim)  # type: ignore[arg-type]
                else:
                    return _VF.nuclear_norm(input, keepdim=keepdim, out=out)  # type: ignore[arg-type]
            else:
                if out is None:
                    return _VF.nuclear_norm(input, _dim, keepdim=keepdim)  # type: ignore[arg-type]
                else:
                    return _VF.nuclear_norm(input, _dim, keepdim=keepdim, out=out)  # type: ignore[arg-type]
        raise RuntimeError(f"only valid string values are 'fro' and 'nuc', found {p}")
    else:
        if _dim is None:
            _dim = list(range(ndim))

        if out is None:
            if dtype is None:
                return _VF.norm(input, p, _dim, keepdim=keepdim)  # type: ignore[attr-defined]
            else:
                return _VF.norm(input, p, _dim, keepdim=keepdim, dtype=dtype)  # type: ignore[attr-defined]
        else:
            if dtype is None:
                return _VF.norm(input, p, _dim, keepdim=keepdim, out=out)  # type: ignore[attr-defined]
            else:
                return _VF.norm(input, p, _dim, keepdim=keepdim, dtype=dtype, out=out)  # type: ignore[attr-defined]


def unravel_index(
    indices: Tensor,
    shape: int | Sequence[int] | torch.Size,
) -> tuple[Tensor, ...]:
    r"""Converts a tensor of flat indices into a tuple of coordinate tensors that
    index into an arbitrary tensor of the specified shape.

    Args:
        indices (Tensor): An integer tensor containing indices into the
            flattened version of an arbitrary tensor of shape :attr:`shape`.
            All elements must be in the range ``[0, prod(shape) - 1]``.

        shape (int, sequence of ints, or torch.Size): The shape of the arbitrary
            tensor. All elements must be non-negative.

    Returns:
        tuple of Tensors: Each ``i``-th tensor in the output corresponds with
        dimension ``i`` of :attr:`shape`. Each tensor has the same shape as
        ``indices`` and contains one index into dimension ``i`` for each of the
        flat indices given by ``indices``.

    Example::

        >>> import torch
        >>> torch.unravel_index(torch.tensor(4), (3, 2))
        (tensor(2),
         tensor(0))

        >>> torch.unravel_index(torch.tensor([4, 1]), (3, 2))
        (tensor([2, 0]),
         tensor([0, 1]))

        >>> torch.unravel_index(torch.tensor([0, 1, 2, 3, 4, 5]), (3, 2))
        (tensor([0, 0, 1, 1, 2, 2]),
         tensor([0, 1, 0, 1, 0, 1]))

        >>> torch.unravel_index(torch.tensor([1234, 5678]), (10, 10, 10, 10))
        (tensor([1, 5]),
         tensor([2, 6]),
         tensor([3, 7]),
         tensor([4, 8]))

        >>> torch.unravel_index(torch.tensor([[1234], [5678]]), (10, 10, 10, 10))
        (tensor([[1], [5]]),
         tensor([[2], [6]]),
         tensor([[3], [7]]),
         tensor([[4], [8]]))

        >>> torch.unravel_index(torch.tensor([[1234], [5678]]), (100, 100))
        (tensor([[12], [56]]),
         tensor([[34], [78]]))
    """
    if has_torch_function_unary(indices):
        return handle_torch_function(unravel_index, (indices,), indices, shape=shape)
    res_tensor = _unravel_index(indices, shape)
    return res_tensor.unbind(-1)


def _unravel_index(indices: Tensor, shape: int | Sequence[int]) -> Tensor:
    torch._check_type(
        not indices.is_complex()
        and not indices.is_floating_point()
        and indices.dtype != torch.bool,
        lambda: f"expected 'indices' to be integer dtype, but got {indices.dtype}",
    )

    torch._check_type(
        isinstance(shape, (int, torch.SymInt, Sequence)),
        lambda: f"expected 'shape' to be int or sequence of ints, but got {type(shape)}",
    )

    if isinstance(shape, (int, torch.SymInt)):
        shape = torch.Size([shape])  # pyrefly: ignore [bad-argument-type]
    else:
        for dim in shape:
            torch._check_type(
                isinstance(dim, (int, torch.SymInt)),
                lambda: f"expected 'shape' sequence to only contain ints, but got {type(dim)}",
            )
        shape = torch.Size(shape)

    torch._check_value(
        all(dim >= 0 for dim in shape),
        lambda: f"'shape' cannot have negative values, but got {tuple(shape)}",
    )

    torch._check_value(
        all(dim > 0 for dim in shape) or indices.numel() == 0,
        lambda: f"'shape' cannot have zero dimensions when 'indices' is non-empty, but got {tuple(shape)}",
    )

    coefs = list(
        reversed(
            list(
                itertools.accumulate(
                    reversed(shape[1:] + torch.Size([1])), func=operator.mul
                )
            )
        )
    )
    return indices.unsqueeze(-1).floor_divide(
        torch.tensor(coefs, device=indices.device, dtype=torch.int64)
    ) % torch.tensor(shape, device=indices.device, dtype=torch.int64)


def chain_matmul(*matrices, out=None):
    r"""Returns the matrix product of the :math:`N` 2-D tensors. This product is efficiently computed
    using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms
    of arithmetic operations (`[CLRS]`_). Note that since this is a function to compute the product, :math:`N`
    needs to be greater than or equal to 2; if equal to 2 then a trivial matrix-matrix product is returned.
    If :math:`N` is 1, then this is a no-op - the original matrix is returned as is.

    .. warning::

        :func:`torch.chain_matmul` is deprecated and will be removed in a future PyTorch release.
        Use :func:`torch.linalg.multi_dot` instead, which accepts a list of two or more tensors
        rather than multiple arguments.

    Args:
        matrices (Tensors...): a sequence of 2 or more 2-D tensors whose product is to be determined.
        out (Tensor, optional): the output tensor. Ignored if :attr:`out` = ``None``.

    Returns:
        Tensor: if the :math:`i^{th}` tensor was of dimensions :math:`p_{i} \times p_{i + 1}`, then the product
        would be of dimensions :math:`p_{1} \times p_{N + 1}`.

    Example::

        >>> # xdoctest: +SKIP
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> a = torch.randn(3, 4)
        >>> b = torch.randn(4, 5)
        >>> c = torch.randn(5, 6)
        >>> d = torch.randn(6, 7)
        >>> # will raise a deprecation warning
        >>> torch.chain_matmul(a, b, c, d)
        tensor([[ -2.3375,  -3.9790,  -4.1119,  -6.6577,   9.5609, -11.5095,  -3.2614],
                [ 21.4038,   3.3378,  -8.4982,  -5.2457, -10.2561,  -2.4684,   2.7163],
                [ -0.9647,  -5.8917,  -2.3213,  -5.2284,  12.8615, -12.2816,  -2.5095]])

    .. _`[CLRS]`: https://mitpress.mit.edu/books/introduction-algorithms-third-edition
    """
    # This wrapper exists to support variadic args.
    if has_torch_function(matrices):
        return handle_torch_function(chain_matmul, matrices, *matrices)

    if out is None:
        return _VF.chain_matmul(matrices)  # type: ignore[attr-defined]
    else:
        return _VF.chain_matmul(matrices, out=out)  # type: ignore[attr-defined]


def _lu_impl(A, pivot=True, get_infos=False, out=None):
    # type: (Tensor, bool, bool, Any) -> tuple[Tensor, Tensor, Tensor]
    r"""Computes the LU factorization of a matrix or batches of matrices
    :attr:`A`. Returns a tuple containing the LU factorization and
    pivots of :attr:`A`.  Pivoting is done if :attr:`pivot` is set to
    ``True``.

    .. warning::

        :func:`torch.lu` is deprecated in favor of :func:`torch.linalg.lu_factor`
        and :func:`torch.linalg.lu_factor_ex`. :func:`torch.lu` will be removed in a
        future PyTorch release.
        ``LU, pivots, info = torch.lu(A, compute_pivots)`` should be replaced with

        .. code:: python

            LU, pivots = torch.linalg.lu_factor(A, compute_pivots)

        ``LU, pivots, info = torch.lu(A, compute_pivots, get_infos=True)`` should be replaced with

        .. code:: python

            LU, pivots, info = torch.linalg.lu_factor_ex(A, compute_pivots)

    .. note::
        * The returned permutation matrix for every matrix in the batch is
          represented by a 1-indexed vector of size ``min(A.shape[-2], A.shape[-1])``.
          ``pivots[i] == j`` represents that in the ``i``-th step of the algorithm,
          the ``i``-th row was permuted with the ``j-1``-th row.
        * LU factorization with :attr:`pivot` = ``False`` is not available
          for CPU, and attempting to do so will throw an error. However,
          LU factorization with :attr:`pivot` = ``False`` is available for
          CUDA.
        * This function does not check if the factorization was successful
          or not if :attr:`get_infos` is ``True`` since the status of the
          factorization is present in the third element of the return tuple.
        * In the case of batches of square matrices with size less or equal
          to 32 on a CUDA device, the LU factorization is repeated for
          singular matrices due to the bug in the MAGMA library
          (see magma issue 13).
        * ``L``, ``U``, and ``P`` can be derived using :func:`torch.lu_unpack`.

    .. warning::
        The gradients of this function will only be finite when :attr:`A` is full rank.
        This is because the LU decomposition is just differentiable at full rank matrices.
        Furthermore, if :attr:`A` is close to not being full rank,
        the gradient will be numerically unstable as it depends on the computation of :math:`L^{-1}` and :math:`U^{-1}`.

    Args:
        A (Tensor): the tensor to factor of size :math:`(*, m, n)`
        pivot (bool, optional): Whether to compute the LU decomposition with partial pivoting, or the regular LU
                                decomposition. :attr:`pivot`\ `= False` not supported on CPU. Default: `True`.
        get_infos (bool, optional): if set to ``True``, returns an info IntTensor.
                                    Default: ``False``
        out (tuple, optional): optional output tuple. If :attr:`get_infos` is ``True``,
                               then the elements in the tuple are Tensor, IntTensor,
                               and IntTensor. If :attr:`get_infos` is ``False``, then the
                               elements in the tuple are Tensor, IntTensor. Default: ``None``

    Returns:
        (Tensor, IntTensor, IntTensor (optional)): A tuple of tensors containing

            - **factorization** (*Tensor*): the factorization of size :math:`(*, m, n)`

            - **pivots** (*IntTensor*): the pivots of size :math:`(*, \text{min}(m, n))`.
              ``pivots`` stores all the intermediate transpositions of rows.
              The final permutation ``perm`` could be reconstructed by
              applying ``swap(perm[i], perm[pivots[i] - 1])`` for ``i = 0, ..., pivots.size(-1) - 1``,
              where ``perm`` is initially the identity permutation of :math:`m` elements
              (essentially this is what :func:`torch.lu_unpack` is doing).

            - **infos** (*IntTensor*, *optional*): if :attr:`get_infos` is ``True``, this is a tensor of
              size :math:`(*)` where non-zero values indicate whether factorization for the matrix or
              each minibatch has succeeded or failed

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> warnings.filterwarnings("ignore", message=".*torch.lu is deprecated")  # docs: hide
        >>> A = torch.randn(2, 3, 3)
        >>> A_LU, pivots = torch.lu(A)
        >>> A_LU
        tensor([[[ 1.3506,  2.5558, -0.0816],
                 [ 0.1684,  1.1551,  0.1940],
                 [ 0.1193,  0.6189, -0.5497]],

                [[ 0.4526,  1.2526, -0.3285],
                 [-0.7988,  0.7175, -0.9701],
                 [ 0.2634, -0.9255, -0.3459]]])
        >>> pivots
        tensor([[ 3,  3,  3],
                [ 3,  3,  3]], dtype=torch.int32)
        >>> A_LU, pivots, info = torch.lu(A, get_infos=True)
        >>> if info.nonzero().size(0) == 0:
        ...     print('LU factorization succeeded for all samples!')
        LU factorization succeeded for all samples!
    """
    # If get_infos is True, then we don't need to check for errors and vice versa
    return torch._lu_with_info(A, pivot=pivot, check_errors=(not get_infos))


if TYPE_CHECKING:
    _ListOrSeq = Sequence[Tensor]
else:
    _ListOrSeq = list[Tensor]


def _check_list_size(out_len: int, get_infos: bool, out: _ListOrSeq) -> None:
    get_infos_int = 1 if get_infos else 0
    if out_len - get_infos_int != 2:
        raise TypeError(
            f"expected tuple of {2 + int(get_infos)} elements but got {out_len}"
        )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            f"argument 'out' must be tuple of Tensors, not {type(out).__name__}"
        )


def _lu_with_infos(A, pivot=True, get_infos=False, out=None):
    # type: (Tensor, bool, bool, Optional[tuple[Tensor, Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor]
    if has_torch_function_unary(A):
        return handle_torch_function(
            lu, (A,), A, pivot=pivot, get_infos=get_infos, out=out
        )
    result = _lu_impl(A, pivot, get_infos, out)
    if out is not None:
        _check_list_size(len(out), get_infos, out)
        for i in range(len(out)):
            out[i].resize_as_(result[i]).copy_(result[i])
        return out
    else:
        return result  # A_LU, pivots, infos


def _lu_no_infos(A, pivot=True, get_infos=False, out=None):
    # type: (Tensor, bool, bool, Optional[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]
    # need to check for torch_function here so that we exit if
    if has_torch_function_unary(A):
        return handle_torch_function(
            lu, (A,), A, pivot=pivot, get_infos=get_infos, out=out
        )
    result = _lu_impl(A, pivot, get_infos, out)
    if out is not None:
        _check_list_size(len(out), get_infos, out)
        for i in range(len(out)):
            out[i].resize_as_(result[i]).copy_(result[i])
        return out
    else:
        return result[0], result[1]  # A_LU, pivots


# The return type of lu depends on `get_infos`, so in order to resolve the output type
# of lu in TorchScript we need to statically know the value of `get_infos`
lu = boolean_dispatch(
    arg_name="get_infos",
    arg_index=2,
    default=False,
    if_true=_lu_with_infos,
    if_false=_lu_no_infos,
    module_name=__name__,
    func_name="lu",
)
lu.__doc__ = _lu_impl.__doc__
<|endoftext|># mypy: allow-untyped-defs
"""Adds docstrings to Storage functions"""

import torch._C
from torch._C import _add_docstr as add_docstr


storage_classes = ["StorageBase"]


def add_docstr_all(method, docstr):
    for cls_name in storage_classes:
        cls = getattr(torch._C, cls_name)
        try:
            add_docstr(getattr(cls, method), docstr)
        except AttributeError:
            pass


add_docstr_all(
    "from_file",
    """
from_file(filename, shared=False, nbytes=0) -> Storage

Creates a CPU storage backed by a memory-mapped file.

If ``shared`` is ``True``, then memory is shared between all processes.
All changes are written to the file. If ``shared`` is ``False``, then the changes on
the storage do not affect the file.

``nbytes`` is the number of bytes of storage. If ``shared`` is ``False``,
then the file must contain at least ``nbytes`` bytes. If ``shared`` is
``True`` the file will be created if needed. (Note that for ``UntypedStorage``
this argument differs from that of ``TypedStorage.from_file``)

Args:
    filename (str): file name to map
    shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                    underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
    nbytes (int): number of bytes of storage
""",
)
<|endoftext|># mypy: allow-untyped-defs
import _compat_pickle
import copyreg
import functools
import importlib
import logging
import math
import os
import pickle
import re
import sys
import traceback
import warnings
from collections import defaultdict
from collections.abc import Callable
from types import ModuleType
from typing import Any, cast, Generic, TYPE_CHECKING, TypedDict
from typing_extensions import deprecated, NotRequired, ParamSpec

import torch


def _type(self, dtype=None, non_blocking=False, **kwargs):
    """Returns the type if `dtype` is not provided, else casts this object to
    the specified type.

    If this is already of the correct type, no copy is performed and the
    original object is returned.

    Args:
        dtype (type or string): The desired type
        non_blocking (bool): If ``True``, and the source is in pinned memory
            and destination is on the GPU or vice versa, the copy is performed
            asynchronously with respect to the host. Otherwise, the argument
            has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument. The ``async`` arg is deprecated.
    """
    non_blocking = _get_async_or_non_blocking("type", non_blocking, kwargs)
    if dtype is None:
        return self.__module__ + "." + self.__class__.__name__

    if isinstance(dtype, str):
        dtype = _import_dotted_name(dtype)
    if dtype is type(self):
        return self
    if self.is_sparse:
        if not dtype.is_sparse:
            raise RuntimeError("Cannot cast sparse tensor to dense tensor")
        new_module_name = dtype.__module__.replace(".sparse", "")
        new_values_type_name = new_module_name + "." + dtype.__name__
        new_values = torch.Tensor._values(self).type(new_values_type_name, non_blocking)
        new_indices_type_name = new_module_name + ".LongTensor"
        new_indices = torch.Tensor._indices(self).type(
            new_indices_type_name, non_blocking
        )
        return dtype(new_indices, new_values, self.size())
    if dtype.is_sparse:
        raise RuntimeError("Cannot cast dense tensor to sparse tensor")
    return dtype(self.size()).copy_(self, non_blocking)


def _to(self, device, non_blocking=False):
    """Returns a copy of this object in device memory.

    If this object is already on the correct device, then no copy is performed
    and the original object is returned.

    Args:
        device (int): The destination device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
    """
    if self.device == device:
        return self

    if device.type == "cpu":
        pin_memory = non_blocking and self.device.type in (
            "cuda",
            "xpu",
            "mtia",
            torch._C._get_privateuse1_backend_name(),
        )
        untyped_storage = torch.empty(
            self.nbytes(), dtype=torch.uint8, device=device, pin_memory=pin_memory
        ).untyped_storage()
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage

    device_module = getattr(torch, device.type, None)
    if device_module is None:
        raise AssertionError(f"{device.type.upper()} device module is not loaded")
    with device_module.device(device):
        if self.is_sparse and hasattr(device_module, "sparse"):
            new_type = getattr(device_module.sparse, self.__class__.__name__)
            indices = getattr(torch.Tensor._indices(self), device.type)(
                device, non_blocking
            )
            values = getattr(torch.Tensor._values(self), device.type)(
                device, non_blocking
            )
            return new_type(indices, values, self.size())
        else:
            if self.is_sparse:
                raise AssertionError(
                    f"sparse storage is not supported for {device.type.upper()} tensors"
                )
            untyped_storage = torch.UntypedStorage(self.size(), device=device)
            untyped_storage.copy_(self, non_blocking)
            return untyped_storage


def _get_async_or_non_blocking(function_name, non_blocking, kwargs):
    """Return the non-blocking flag given the function name and kwargs.

    Args:
        function_name (str): the name of the function being used.
        non_blocking (bool): the default value.
        **kwargs (dict): the kwargs passed to the function.
    """
    if not kwargs:
        return non_blocking
    if len(kwargs) != 1 or "async" not in kwargs:
        message = "{}() got an unexpected keyword argument '{}'"
        argument = list(kwargs.keys()).pop()
        raise TypeError(message.format(function_name, argument))
    warnings.warn("'async' is deprecated; use 'non_blocking'", stacklevel=2)
    return kwargs["async"]


def _get_restore_location(device):
    """Return the map_location location.

    Used for rebuild functions where the tensor device is distinct from the storage
    """

    map_location = torch.serialization._serialization_tls.map_location
    if map_location is None:
        return device
    else:
        if isinstance(map_location, dict):
            return map_location.get(device, device)
        elif isinstance(map_location, (str, torch.device)):
            return map_location
        else:
            if not callable(map_location):
                raise AssertionError(
                    f"expected callable map_location, got {type(map_location).__name__}"
                )
            raise RuntimeError(
                "Callable map_location not supported with _rebuild_wrapper_subclass "
                "or _rebuild_device_tensor_from_numpy"
            )


# Note [Don't serialize hooks]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Since time immemorial, we have serialized the backward hooks associated with
# variables.  This kind of half-worked--Python can pickle global functions
# (but not closures!)--but there were problems.
#
#   - It's fragile.  If you serialize a backward hook into a saved
#     model, and then you rename the function associated with the hook,
#     now your saved model is broken and you can't load it anymore.
#
#   - It's not actually used.  The standard recommendation is to
#     serialize the *state_dict* of a model, not the model itself
#     (since this is more stable to code changes affecting the model
#     serialization), and the state dict saves "data" only, thus
#     stripping the backward hooks.  In some cases, hooks are
#     essential to the well-functioning of a model (e.g., DDP),
#     but DDP already manages re-adding the hooks!
#
#   - We didn't serialize them in many cases.  Prior to #10220, we
#     were dropping backward hooks in ForkingPickler.  We "fixed" this
#     to be convenient with other serialization sites, but lack of
#     serializing backward hooks wasn't actually the root cause of
#     the bug.
#
# With these cases in mind, we have decided that a better strategy
# is to just NOT serialize hooks at all.
#
# Since this is a BC-breaking change, we should warn when we previously
# serialized a hook, but no longer do so. This will be done by adding a special
# sentinel property to hooks will be used to suppress this warning. If a hook
# has the property _torch_serialize_ignore, we will not emit a warning if we
# attempt to serialize a Tensor with this hook attached to it.
#
# By the way, when _backward_hooks is skipped, we must give an EMPTY
# OrderedDict(), if you pass a None you'll run afoul #12219.


# TODO: Once we decide to break serialization FC, `storage` no longer needs to
# be a TypedStorage
def _rebuild_tensor(storage, storage_offset, size, stride):
    # first construct a tensor with the correct dtype/device
    t = torch.empty((0,), dtype=storage.dtype, device=storage._untyped_storage.device)
    return t.set_(storage._untyped_storage, storage_offset, size, stride)


def get_tensor_metadata(tensor):
    # Tensor's Metadata for serializing.
    # Currently, this only returns a dict[string, bool] specifying whether
    # `conj` or `neg` bit is set.
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError(f"expected torch.Tensor, got {type(tensor).__name__}")
    return torch._C._get_tensor_metadata(tensor)  # type: ignore[attr-defined]


def set_tensor_metadata(tensor, metadata):
    # See `get_tensor_metadata` above
    if not isinstance(metadata, dict):
        raise AssertionError(f"expected dict, got {type(metadata).__name__}")
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError(f"expected torch.Tensor, got {type(tensor).__name__}")
    torch._C._set_tensor_metadata(tensor, metadata)  # type: ignore[attr-defined]


def _restore_device_fake_mode(tensor):
    if torch._guards.detect_fake_mode(None) is not None:
        if tensor.untyped_storage()._fake_device is not None:
            device = _get_restore_location(tensor.untyped_storage()._fake_device)
            if not isinstance(device, torch.device):
                device = torch.device(device)
            tensor.fake_device = torch.device(device)
    return tensor


def _rebuild_tensor_v2(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
    metadata=None,
):
    tensor = _rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    if metadata:
        set_tensor_metadata(tensor, metadata)

    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    tensor._backward_hooks = backward_hooks

    tensor = _restore_device_fake_mode(tensor)
    return tensor


def _rebuild_tensor_v3(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
    dtype,
    metadata=None,
):
    t = torch.empty(
        (0,),
        dtype=dtype,
        device=storage._untyped_storage.device,
        requires_grad=requires_grad,
    )
    t.set_(storage._untyped_storage, storage_offset, size, stride)
    if metadata:
        set_tensor_metadata(t, metadata)
    t._backward_hooks = backward_hooks
    t = _restore_device_fake_mode(t)
    return t


_sparse_tensors_to_validate: list["torch.Tensor"] = []


# In _legacy_load() in serialization.py we unpickle storages after the sparse
# tensors have been already unpickled. Those storages contain data necessary for
# validating sparse tensors: indices and values. That's why sparse tensors are
# first unpickled without any validation, and then this function is called just
# before _legacy_load() returns, so that all the sparse tensors can be validated
# in bulk.
#
# The same procedure must be followed by _load() in serialization.py because due
# to Pickler semantics, we have to use the same (non-validating) function for
# unpickling sparse tensors, regardless of the caller.
def _validate_loaded_sparse_tensors(weights_only=False):
    # In weights_only mode we always validate: malformed sparse indices can
    # cause out-of-bounds reads later (e.g. in to_dense()), and an untrusted
    # checkpoint must not be able to slip those through. Otherwise we fall back
    # to the global check_sparse_tensor_invariants setting.
    try:
        if not _sparse_tensors_to_validate:
            return
        if weights_only:
            warnings.warn(
                "Validating sparse tensor invariants because weights_only=True; "
                "this is an O(nnz) scan per sparse tensor and may be slow for "
                "large checkpoints.",
                stacklevel=2,
            )
        elif not torch.sparse.check_sparse_tensor_invariants().is_enabled():
            return
        # We disable pinning check (see check_pinning=False below) to
        # avoid gh-153143. In fact, pinning check is unnecessary
        # anyway when loading sparse data from external sources.
        for t in _sparse_tensors_to_validate:
            if t.layout is torch.sparse_coo:
                torch._validate_sparse_coo_tensor_args(
                    t._indices(),
                    t._values(),
                    t.size(),
                    t.is_coalesced(),
                    check_pinning=False,
                )
            elif t.layout in {
                torch.sparse_csr,
                torch.sparse_csc,
                torch.sparse_bsr,
                torch.sparse_bsc,
            }:
                # TODO: Validation currently involves an expensive traversal
                # on CPU, which may include a device transfer.
                if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    compressed_indices, plain_indices = (
                        t.crow_indices(),
                        t.col_indices(),
                    )
                else:
                    compressed_indices, plain_indices = (
                        t.ccol_indices(),
                        t.row_indices(),
                    )
                torch._validate_sparse_compressed_tensor_args(
                    compressed_indices,
                    plain_indices,
                    t.values(),
                    t.size(),
                    t.layout,
                    check_pinning=False,
                )
            else:
                raise NotImplementedError(
                    f"_validate_loaded_sparse_tensors for layout `{t.layout}`"
                )

    finally:
        _sparse_tensors_to_validate.clear()


def _rebuild_sparse_tensor(layout, data):
    """
    Rebuilds a sparse tensor from its sparse storage representation.

    Args:
        layout (str): The sparse storage layout of the tensor.
        data (tuple): The tensor's sparse storage representation.
    """
    if layout == torch.sparse_coo:
        if len(data) == 3:
            # For BC:
            indices, values, size = data
            is_coalesced = None
        else:
            indices, values, size, is_coalesced = data
        result = torch.sparse_coo_tensor(
            indices, values, size, check_invariants=False, is_coalesced=is_coalesced
        )
        _sparse_tensors_to_validate.append(result)
        return result

    elif layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        compressed_indices, plain_indices, values, size = data
        result = torch.sparse_compressed_tensor(
            compressed_indices,
            plain_indices,
            values,
            size,
            layout=layout,
            check_invariants=False,
        )
        _sparse_tensors_to_validate.append(result)
        return result

    raise NotImplementedError(f"rebuilding sparse tensor for layout {layout}")


def _rebuild_nested_tensor(buffer, sizes, strides, storage_offsets):
    return torch._nested_view_from_buffer(buffer, sizes, strides, storage_offsets)


def _rebuild_device_tensor_from_cpu_tensor(data, dtype, device, requires_grad):
    device = _get_restore_location(device)
    tensor = data.to(dtype=dtype, device=device)
    tensor.requires_grad = requires_grad
    return tensor


def _rebuild_device_tensor_from_numpy(data, dtype, device, requires_grad):
    device = _get_restore_location(device)
    tensor = torch.from_numpy(data).to(dtype=dtype, device=device)
    tensor.requires_grad = requires_grad
    return tensor


# Should not be used, only here to be able to load Tensors serialized with older versions of pytorch
_rebuild_xla_tensor = _rebuild_device_tensor_from_numpy


def _rebuild_meta_tensor_no_storage(dtype, size, stride, requires_grad):
    return torch.empty_strided(
        size, stride, dtype=dtype, device="meta", requires_grad=requires_grad
    )


def _rebuild_wrapper_subclass(
    cls,
    dtype,
    size,
    stride,
    storage_offset,
    layout,
    device,
    requires_grad,
):
    device = _get_restore_location(device)
    return torch.Tensor._make_wrapper_subclass(
        cls,
        size,
        strides=stride,
        dtype=dtype,
        storage_offset=storage_offset,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


# TODO: Once we decide to break serialization FC, `storage` no longer needs to
# be a TypedStorage
def _rebuild_qtensor(
    storage,
    storage_offset,
    size,
    stride,
    quantizer_params,
    requires_grad,
    backward_hooks,
):
    qscheme = quantizer_params[0]
    if qscheme == torch.per_tensor_affine:
        _, scale, zero_point = quantizer_params
        tensor = torch._empty_affine_quantized(
            size,
            scale=scale,
            zero_point=zero_point,
            dtype=storage.dtype,
            device=storage.device,
        )
    elif qscheme in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
        _, scales, zero_points, axis = quantizer_params
        # The C++ constructor does not bound-check axis or the
        # scales/zero_points length against size; values arriving from an
        # untrusted weights_only checkpoint can otherwise be arbitrary.
        if not 0 <= axis < len(size):
            raise ValueError(
                f"_rebuild_qtensor: per_channel axis {axis} out of range for size {tuple(size)}"
            )
        expected_len = int(size[axis])
        scales_len = len(scales) if isinstance(scales, list) else scales.numel()
        zero_points_len = (
            len(zero_points) if isinstance(zero_points, list) else zero_points.numel()
        )
        if scales_len != expected_len or zero_points_len != expected_len:
            raise ValueError(
                "_rebuild_qtensor: per_channel scales/zero_points length must equal "
                f"size[axis]={expected_len}, got scales={scales_len}, "
                f"zero_points={zero_points_len}"
            )
        if type(scales) is list and type(zero_points) is list:
            if qscheme == torch.per_channel_affine:
                scales = torch.tensor(scales, dtype=torch.double, device=storage.device)
                zero_points = torch.tensor(
                    zero_points, dtype=torch.long, device=storage.device
                )
            else:
                scales = torch.tensor(scales, dtype=torch.float, device=storage.device)
                zero_points = torch.tensor(
                    zero_points, dtype=torch.float, device=storage.device
                )
        tensor = torch._empty_per_channel_affine_quantized(
            size,
            scales=scales,
            zero_points=zero_points,
            axis=axis,
            dtype=storage.dtype,
            device=storage.device,
        )
    else:
        raise RuntimeError(f"Can't deserialize quantized tensor with qscheme {qscheme}")
    tensor.set_(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    tensor._backward_hooks = backward_hooks
    return tensor


def _rebuild_parameter(data, requires_grad, backward_hooks):
    param = torch.nn.Parameter(data, requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks

    return param


def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state):
    param = torch.nn.Parameter(data, requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks

    # Restore state on Parameter like python attr.
    param = _set_obj_state(param, state)
    return param


def _get_obj_state(obj):
    # Get the state of the python subclass
    # This loosely mimics the function on the object class but since Tensor do not inherit
    # from it, we cannot call that function directly
    # https://github.com/python/cpython/blob/c83919bd635f4433f1c6ae8504996a9fe3c215e5/Objects/typeobject.c#L4891
    # Note that starting with Python 3.11, this `__getstate__` is always defined and thus
    # the else branch will never be taken.
    getstate_fn = getattr(obj, "__getstate__", None)
    if getstate_fn:
        state = getstate_fn()
    else:
        slots_to_save = copyreg._slotnames(obj.__class__)  # type: ignore[attr-defined]
        if slots_to_save:
            state = (
                obj.__dict__,
                {
                    name: getattr(obj, name)
                    for name in slots_to_save
                    if hasattr(obj, name)
                },
            )
        else:
            state = obj.__dict__

    return state


def _set_obj_state(obj, state):
    if isinstance(state, tuple):
        if not len(state) == 2:
            raise RuntimeError(f"Invalid serialized state: {state}")
        dict_state = state[0]
        slots_state = state[1]
    else:
        dict_state = state
        slots_state = None

    # Starting with Python 3.11, the __dict__ attribute is lazily created
    # and is serialized as None when not needed.
    if dict_state:
        for k, v in dict_state.items():
            setattr(obj, k, v)

    if slots_state:
        for k, v in slots_state.items():
            setattr(obj, k, v)
    return obj


def _import_dotted_name(name):
    components = name.split(".")
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj


def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    return torch._C._nn.flatten_dense_tensors(tensors)


def _flatten_sparse_tensors(tensors):
    """Flatten sparse tensors into two contiguous 1D buffers, one of indices and
    one of values. Assume tensors are of same sparse type.

    Args:
        tensors (Iterable[Tensor]): sparse tensors to flatten.

    Returns:
        A tuple of two contiguous 1D buffers, one containing input tensors'
        indices and the other containing the values.
    """
    flat_indices = torch._C._nn.flatten_dense_tensors(
        [torch.Tensor._indices(t) for t in tensors]
    )
    flat_values = torch._C._nn.flatten_dense_tensors(
        [torch.Tensor._values(t) for t in tensors]
    )
    return flat_indices, flat_values


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    return torch._C._nn.unflatten_dense_tensors(flat, tensors)


def _unflatten_sparse_tensors(flat, tensors):
    """View flat buffer (containing indices and values) using the sizes of
    tensors. Assume that tensors are of same sparse type, and that flat is given
    by _flatten_sparse_tensors.

    Args:
        flat (tuple(Tensor, Tensor)): flattened indices and values of sparse
          tensors to unflatten.
        tensors (Iterable[Tensor]): sparse tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened sparse tensors with sizes same as tensors and values from
        flat.
    """
    flat_indices, flat_values = flat
    indices = torch._C._nn.unflatten_dense_tensors(
        flat_indices, [torch.Tensor._indices(t) for t in tensors]
    )
    values = torch._C._nn.unflatten_dense_tensors(
        flat_values, [torch.Tensor._values(t) for t in tensors]
    )
    outputs = []
    for t, i, v in zip(tensors, indices, values):
        outputs.append(t.new(i, v, t.size()))
    return tuple(outputs)


def _reorder_tensors_as(tensors, ordered_tensors):
    """Assume that tensors are of same order as ordered_tensors within their
    types, e.g., from _take_tensors. Reorder them to be of same order as
    ordered_tensors.

    Args:
        tensors (Iterable[Tensor]): tensors to be reordered. They should be of
          the same order as ordered_tensors within their own types.
        ordered_tensors (Iterable[Tensor]): tensors whose order will be the
          reference.

    Returns:
        Ordered tuple of tensors with contents from tensors and order of
        ordered_tensors.
    """
    type_dict = defaultdict(list)
    for tensor in tensors:
        type_dict[tensor.type()].append(tensor)
    type_dict_ = {t: iter(coll) for t, coll in type_dict.items()}
    return tuple(next(type_dict_[tensor.type()]) for tensor in ordered_tensors)


def _take_tensors(tensors, size_limit):
    """Group tensors into chunks. This generator yields a chunk at each time,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Yields:
        Blocks of tensors of same type and within size_limit. The yielded
        tensors are only ordered as the original sequence within its types.
    """
    buf_dict: defaultdict[str, list] = defaultdict(lambda: [[], 0])
    for tensor in tensors:
        t = tensor.type()
        if tensor.is_sparse:
            indices = torch.Tensor._indices(tensor)
            values = torch.Tensor._values(tensor)
            size = (
                indices.numel() * indices.element_size()
                + values.numel() * values.element_size()
            )
        else:
            size = tensor.numel() * tensor.element_size()
        buf_and_size = buf_dict[t]
        if buf_and_size[1] + size > size_limit and buf_and_size[1] > 0:
            yield buf_and_size[0]
            buf_and_size = buf_dict[t] = [[], 0]
        buf_and_size[0].append(tensor)  # pyrefly: ignore [missing-attribute]
        buf_and_size[1] += size  # pyrefly: ignore [unsupported-operation]
    for buf, _ in buf_dict.values():
        if len(buf) > 0:
            yield buf


def render_call(fn, args, kwargs):
    str_fn = torch.overrides.resolve_name(fn)
    if str_fn is None:
        str_fn = str(fn)

    str_args: list[str] = []
    with torch._tensor_str.printoptions(threshold=0, edgeitems=0):
        str_args.extend(repr(a) for a in args)
        str_args.extend(f"{k}={repr(v)}" for k, v in kwargs.items())
        r = f"{str_fn}({', '.join(str_args)})"
    return r


# NOTE [ Python Traceback Reference Cycle Problem ]
#
# When using sys.exc_info(), it is important to **not** store the exc_info[2],
# which is the traceback, because otherwise you will run into the traceback
# reference cycle problem, i.e., the traceback holding reference to the frame,
# and the frame (which holds reference to all the object in its temporary scope)
# holding reference the traceback.


class KeyErrorMessage(str):
    r"""str subclass that returns itself in repr"""

    __slots__ = ()

    def __repr__(self):
        return self


class ExceptionWrapper:
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info=None, where="in background"):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]

        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        r"""Reraises the wrapped exception in the current thread"""
        # Format a message such as: "Caught ValueError in DataLoader worker
        # process 2. Original Traceback:", followed by the traceback.
        msg = f"Caught {self.exc_type.__name__} {self.where}.\nOriginal {self.exc_msg}"  # pyrefly: ignore [missing-attribute]
        if self.exc_type is KeyError:
            # KeyError calls repr() on its argument (usually a dict key). This
            # makes stack traces unreadable. It will not be changed in Python
            # (https://bugs.python.org/issue2651), so we work around it.
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            # Some exceptions have first argument as non-str but explicitly
            # have message field
            # pyrefly: ignore [not-callable]
            raise self.exc_type(
                # pyrefly: ignore [unexpected-keyword]
                message=msg
            )
        try:
            exception = self.exc_type(msg)  # pyrefly: ignore [not-callable]
        except Exception:
            # If the exception takes multiple arguments or otherwise can't
            # be constructed, don't try to instantiate since we don't know how to
            raise RuntimeError(msg) from None
        raise exception


def cpu_count() -> int | None:
    """Return the number of CPUs available to the current process.

    Prefers ``os.sched_getaffinity`` (respects cgroups / taskset) and
    falls back to ``os.cpu_count``.
    """
    # os.process_cpu_count was added in CPython 3.13, see
    # https://docs.python.org/3/library/os.html#os.process_cpu_count
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def _get_available_device_type():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
        return "xpu"
    if hasattr(torch, "mtia") and torch.mtia.is_available():
        return "mtia"
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, custom_backend_name, None)
    if custom_device_mod and custom_device_mod.is_available():
        return custom_backend_name
    # add more available device types here
    return None


def _get_device_attr(get_member):
    device_type = _get_available_device_type()
    if device_type and device_type.lower() == "cuda":
        return get_member(torch.cuda)
    if device_type and device_type.lower() == "mps":
        return get_member(torch.mps)
    if device_type and device_type.lower() == "xpu":
        return get_member(torch.xpu)  # type: ignore[attr-defined]
    if device_type and device_type.lower() == "mtia":
        return get_member(torch.mtia)
    if device_type == torch._C._get_privateuse1_backend_name():
        return get_member(getattr(torch, device_type))
    # add more available device types here
    return None


def _get_current_device_index():
    # current device index
    return _get_device_attr(lambda m: m.current_device())


def _get_all_device_indices():
    # all device index
    return _get_device_attr(lambda m: list(range(m.device_count())))


def _get_devices_properties(device_ids):
    # all device properties
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]


def get_current_device_index() -> int:
    r"""Checks if there are CUDA devices available and
    returns the device index of the current default CUDA device.
    Returns -1 in case there are no CUDA devices available.
    Arguments: ``None``
    """
    if torch.cuda.device_count() > 0:
        return torch.cuda.current_device()
    return -1


def _get_device_index(
    device: Any,
    optional: bool = False,
    allow_cpu: bool = False,
) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    has index. Note that for a device without a specified index,
    i.e., ``torch.device('xxx')``, this will return the current default
    device of that type if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default
    device of the supported runtime platform if :attr:`optional` is ``True``.
    i.e., the current default CUDA device will be returned if CUDA runtime is supported.
    """
    if isinstance(device, str):
        device = torch.device(device)
    device_idx: int | None = None
    if isinstance(device, torch.device):
        if not allow_cpu and device.type == "cpu":
            raise ValueError(f"Expected a non cpu device, but got: {device}")
        device_idx = -1 if device.type == "cpu" else device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        if optional:
            # The eager API _get_current_device_index uses `lambda` functions which are
            # not supported in JIT and hence not scriptable. The JIT equivalent API to get
            # the current device index is `get_current_device_index()` which can
            # be scripted. We use is_scripting to check the mode we are in and call the
            # appropriate API.
            if torch.jit.is_scripting():
                device_idx = get_current_device_index()
            else:
                device_idx = _get_current_device_index()
        else:
            raise ValueError(
                f"Expected a torch.device with a specified index or an integer, but got:{device}"
            )
    return device_idx


def _handle_complex(tensor):
    """
    Returns a real view of a tensor if complex dtype else just the tensor
    need to check if a UninitializedParameter because otherwise checking is_complex is an error for a LazyModule
    """
    return (
        torch.view_as_real(tensor)
        if not isinstance(tensor, torch.nn.UninitializedParameter)
        and tensor.is_complex()
        else tensor
    )


def _element_size(dtype):
    """
    Returns the element size for a dtype, in bytes
    """
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"expected torch.dtype, but got {type(dtype)}")

    if dtype.is_complex:
        return torch.finfo(dtype).bits >> 2
    elif dtype.is_floating_point:
        return torch.finfo(dtype).bits >> 3
    elif dtype == torch.bool:
        # NOTE: torch.bool is not supported in torch.iinfo()
        return 1
    else:
        return torch.iinfo(dtype).bits >> 3


class _ClassPropertyDescriptor:
    def __init__(self, fget, fset=None):
        self.fget = fget

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return self.fget.__get__(instance, owner)()


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)


if TYPE_CHECKING:
    # TorchScript does not support `@deprecated`
    # This is a workaround to avoid breaking TorchScript
    @deprecated(
        "`torch._utils.is_compiling` is deprecated. Use `torch.compiler.is_compiling` instead.",
        category=FutureWarning,
    )
    def is_compiling() -> bool:
        return torch.compiler.is_compiling()

else:

    def is_compiling() -> bool:
        """
        Indicates whether we are tracing/compiling with torch.compile() or torch.export().
        """
        warnings.warn(  # use `warnings.warn` instead of `@deprecated`
            "`torch._utils.is_compiling` is deprecated. Use `torch.compiler.is_compiling` instead.",
            # FutureWarning,  # TorchScript does not support Warning type
            stacklevel=2,
        )
        return torch.compiler.is_compiling()


def _functionalize_sync(t):
    # This code lives in python instead of C++ since conditioning on a certain python subclass
    # is much more of a pain in C++.
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(t, FunctionalTensor):
        # If a FunctionalTensorMode is active while syncing, we don't want it to intercept any ops that get called
        # when we sync our inner tensor.
        # Why?
        # (1) If there are input mutations in the graph, then they will be re-applied during
        #     AOTAutograd when we call _sync() from inside of our functionalization kernels.
        # (2) _sync() causes us to regenerate our updated the tensor from the updated base,
        #     which dispatches to a bunch of view ops
        # (3) The input to these view ops is our inner FunctionalTensorWrapper
        #     (since the sync was called from C++), not the python FunctionalTensor
        # (4) if a python FunctionalTensorMode is active, it will complain when it intercepts
        #     the view op, since it will see an input that is a C++ FunctionalTensorWrapper
        #     (aka a normal torch.Tensor) instead of a python `FunctionalTensor).
        maybe_functional_mode = torch._C._unset_dispatch_mode(
            torch._C._TorchDispatchModeKey.FUNCTIONAL
        )
        try:
            torch._functionalize_sync(t.elem)  # type: ignore[attr-defined]
        finally:
            if maybe_functional_mode is not None:
                torch._C._set_dispatch_mode(maybe_functional_mode)
    else:
        torch._functionalize_sync(t)  # type: ignore[attr-defined]


@functools.lru_cache(2)
def _get_device_module(device_type: str):
    device_module = getattr(torch, device_type, None)
    if device_module is None:
        raise RuntimeError(
            f"Device '{device_type}' does not have a corresponding module registered as 'torch.{device_type}'."
        )
    return device_module


def _dummy_type(name: str) -> type:
    def get_err_fn(is_init: bool):
        def err_fn(obj, *args, **kwargs):
            if is_init:
                class_name = obj.__class__.__name__
            else:
                class_name = obj.__name__
            raise RuntimeError(f"Tried to instantiate dummy base class {class_name}")

        return err_fn

    return type(
        name, (object,), {"__init__": get_err_fn(True), "__new__": get_err_fn(False)}
    )


class _LazySeedTracker:
    # Since seeding is memory-less, only track the latest seed.
    # Note: `manual_seed_all` followed by `manual_seed` overwrites
    # the seed on current device. We track the order of **latest**
    # calls between these two API.
    def __init__(self):
        self.manual_seed_all_cb = None
        self.manual_seed_cb = None
        self.call_order = []

    def queue_seed_all(self, cb, traceback):
        self.manual_seed_all_cb = (cb, traceback)  # pyrefly: ignore [bad-assignment]
        # update seed_all to be latest
        self.call_order = [self.manual_seed_cb, self.manual_seed_all_cb]

    def queue_seed(self, cb, traceback):
        self.manual_seed_cb = (cb, traceback)  # pyrefly: ignore [bad-assignment]
        # update seed to be latest
        self.call_order = [self.manual_seed_all_cb, self.manual_seed_cb]

    def get_calls(self) -> list:
        return self.call_order


logger = logging.getLogger(__name__)
P = ParamSpec("P")


class CallbackRegistry(Generic[P]):
    def __init__(self, name: str):
        self.name = name
        self.callback_list: list[Callable[P, None]] = []

    def add_callback(self, cb: Callable[P, None]) -> None:
        self.callback_list.append(cb)

    def fire_callbacks(self, *args: P.args, **kwargs: P.kwargs) -> None:
        for cb in self.callback_list:
            try:
                cb(*args, **kwargs)
            except Exception:
                logger.exception(
                    "Exception in callback for %s registered with gpu trace", self.name
                )


def try_import(module_name: str) -> ModuleType | None:
    # Implementation based on
    # https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
    if (module := sys.modules.get(module_name, None)) is not None:
        return module

    if (spec := importlib.util.find_spec(module_name)) is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # https://docs.python.org/3/library/importlib.html#importlib.machinery.ModuleSpec.loader
        # "The finder should always set this attribute"
        if spec.loader is None:
            raise AssertionError("The loader attribute should always be set")
        spec.loader.exec_module(module)
        return module

    return None


# IMPORT_MAPPING and NAME_MAPPING are adapted from https://github.com/python/cpython/blob/main/Lib/_compat_pickle.py
# for use in the weights_only Unpickler.

IMPORT_MAPPING = {
    "__builtin__": "builtins",
    "copy_reg": "copyreg",
    "Queue": "queue",
    "repr": "reprlib",
    "_abcoll": "collections.abc",
    # Non-mutual mappings.
    "UserDict": "collections",
    "UserList": "collections",
    "UserString": "collections",
    "whichdb": "dbm",
    "StringIO": "io",
    "cStringIO": "io",
}


# This contains rename rules that are easy to handle.  We ignore the more
# complex stuff (e.g. mapping the names in the urllib and types modules).
# These rules should be run before import names are fixed.
NAME_MAPPING = {
    ("__builtin__", "xrange"): ("builtins", "range"),
    ("__builtin__", "reduce"): ("functools", "reduce"),
    ("__builtin__", "intern"): ("sys", "intern"),
    ("__builtin__", "unichr"): ("builtins", "chr"),
    ("__builtin__", "unicode"): ("builtins", "str"),
    ("__builtin__", "long"): ("builtins", "int"),
    ("itertools", "izip"): ("builtins", "zip"),
    ("itertools", "imap"): ("builtins", "map"),
    ("itertools", "ifilter"): ("builtins", "filter"),
    ("itertools", "ifilterfalse"): ("itertools", "filterfalse"),
    ("itertools", "izip_longest"): ("itertools", "zip_longest"),
    ("UserDict", "IterableUserDict"): ("collections", "UserDict"),
    ("UserList", "UserList"): ("collections", "UserList"),
    ("UserString", "UserString"): ("collections", "UserString"),
    # Non-mutual mappings.
    ("__builtin__", "basestring"): ("builtins", "str"),
    ("exceptions", "StandardError"): ("builtins", "Exception"),
    ("UserDict", "UserDict"): ("collections", "UserDict"),
}

# Protocol 2 pickle (torch.save's default) maps builtin exceptions to the
# Python 2 "exceptions" module via REVERSE_NAME_MAPPING; map them back so
# allowlisted exception types resolve under their builtins.* names.
NAME_MAPPING.update(
    {
        ("exceptions", name): ("builtins", name)
        for name in _compat_pickle.PYTHON2_EXCEPTIONS
    }
)


def _chunk_or_narrow_cat(
    tensor: "torch.Tensor",
    num_chunks: int,
    narrow_dim: int,
    cat_dim: int = 0,
) -> "torch.Tensor":
    """
    Splits tensor along narrow_dim into num_chunks and concatenates along cat_dim.
    Uses torch.chunk in eager mode, but torch.narrow under tracing to be unbacked-symint safe.
    """
    if torch.distributed.is_available():
        from torch.distributed._functional_collectives import _are_we_tracing
        from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

        # TODO(pianpwk): remove the unbacked symbols check and fix AsyncTP pattern matching
        # for test_micro_pipeline_tp.py.
        if _are_we_tracing() and has_free_unbacked_symbols(tensor):
            chunk_size = tensor.size(narrow_dim) // num_chunks
            chunks = [
                torch.narrow(tensor, narrow_dim, i * chunk_size, chunk_size)
                for i in range(num_chunks)
            ]
            return torch.cat(chunks, dim=cat_dim)
    return torch.cat(torch.chunk(tensor, num_chunks, dim=narrow_dim), dim=cat_dim)


def _maybe_view_chunk_cat(
    res: "torch.Tensor", group_size: int, gather_dim: int
) -> "torch.Tensor":
    """
    This is intuitively the same as torch.cat(torch.chunk(res, group_size,
    dim=0), dim=gather_dim), but returns a view if data movement is not
    necessary.  This operation arises in NCCL all_gather, where you always get
    a result which is concatenated on dim=0, even though actually you may need
    to undo this concatenation and then re-cat on the gather dim.

    When is data-movement not necessary?  Intuitively, we need to understand if
    the unflatten in this reference implementation of this code triggers a
    copy or not:

        chunks = torch.unflatten(res, 0, [group_size, -1])
        return torch.flatten(torch.movedim(chunks, 0, gather_dim), gather_dim, gather_dim + 1)

    Assume res is contiguous (it will be coming out of the collective).  We
    essentially need to know if the movedim maintains the contiguity of the
    tensor.  Moving a dimension typically does NOT preserve contiguity, unless
    EVERY dimension it is moved across is size 1.

    Example: shape [4, d1, d2] with group_size=4, gather_dim=1 -> [1, 4*d1, d2]

        [4, d1, d2] -> [4, 1, d1, d2] -> [1, 4, d1, d2] (contiguous!)

    Example: shape [4, 2, d2] with group_size=4, gather_dim=2 -> [1, 2, 4*d2]

        [4, 2, d2] -> [4, 1, 2, d2] -> [1, 2, 4, d2] (not contiguous!)

    Args:
        res: Tensor with gathered data in dim 0, shape [group_size, ...]
        group_size: Number of ranks in the group
        gather_dim: Dimension to gather along in the output

    Returns:
        Tensor with data rearranged to gather along gather_dim
    """

    if gather_dim == 0:
        # When gather_dim is 0, chunk+cat is a no-op
        return res

    shape = list(res.shape)

    # Optimization: Can use view instead of split+cat when:
    # 1. res.shape[0] == group_size (invariant after all_gather)
    # 2. All dims between 0 and gather_dim (exclusive) have size 1
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    numel_between = math.prod(shape[1:gather_dim]) if gather_dim > 1 else 1

    if guard_or_false(shape[0] == group_size) and guard_or_false(numel_between == 1):
        # View optimization: reshape to collapse dim 0 into gather_dim
        final_shape = (
            [1]  # Dim 0 becomes 1
            + shape[1:gather_dim]  # Dims 1 to gather_dim-1 unchanged
            + [shape[0] * shape[gather_dim]]  # gather_dim gets multiplied by group_size
            + shape[gather_dim + 1 :]  # Rest unchanged
        )
        return res.view(final_shape)
    else:
        # General case: fall back to split + cat
        # This is better than torch.flatten as cat can be vectorized, whereas
        # the contiguous kernel is always bad.
        return _chunk_or_narrow_cat(res, group_size, narrow_dim=0, cat_dim=gather_dim)


class _Frame(TypedDict):
    """Frame information from memory profiler snapshots."""

    filename: str
    line: int
    name: str
    # Fields added by FX augmentation (optional)
    fx_node_op: NotRequired[str]
    fx_node_name: NotRequired[str]
    fx_node_target: NotRequired[str]
    fx_original_trace: NotRequired[str]


class _Block(TypedDict):
    """Memory block information."""

    size: int
    requested_size: int
    address: int
    state: str
    frames: list[_Frame]


class _Segment(TypedDict):
    """Memory segment information."""

    address: int
    total_size: int
    stream: int
    segment_type: str
    allocated_size: int
    active_size: int
    blocks: list[_Block]


class _TraceEntry(TypedDict):
    """Memory trace entry information."""

    action: str
    addr: NotRequired[int]
    frames: list[_Frame]
    size: int
    stream: int
    device_free: NotRequired[int]


class _Snapshot(TypedDict):
    """Memory snapshot structure."""

    segments: list[_Segment]
    device_traces: NotRequired[list[list[_TraceEntry]]]


def _augment_frames(frames: list[_Frame]) -> int:
    """
    Augment a list of frames with FX debug information. For each frame corresponding
    to an FX-generated Python file, this function attaches additional FX node
    metadata (op, name, target, and original trace).

    Args:
        frames (list[_Frame]): List of frame dictionaries to augment

    Returns:
        int: The count of frames that were augmented.
    """
    from torch.fx.graph_module import FX_GRAPH_MODULE_FILE_PREFIX
    from torch.fx.traceback import _FX_METADATA_REGISTRY

    # Regex pattern to match FX generated files
    _FX_GENERATED_PATTERN = re.compile(
        rf"{re.escape(FX_GRAPH_MODULE_FILE_PREFIX)}.*\.py$"
    )

    count = 0

    for frame in frames:
        filename = frame.get("filename")
        lineno = frame.get("line")
        if not filename or not lineno:
            continue

        # Check if this looks like an FX generated file
        if not _FX_GENERATED_PATTERN.search(os.path.basename(filename)):
            continue

        metadata = _FX_METADATA_REGISTRY.get(filename)
        if metadata is None:
            continue

        lineno_map = metadata.get("lineno_map", {})
        node_metadata = metadata.get("node_metadata", {})
        prologue_start = metadata.get("prologue_start", 0)

        # Get the node index for this line
        node_idx = lineno_map.get(lineno - prologue_start)
        if node_idx is None:
            continue

        node_info = node_metadata.get(node_idx)
        if node_info is None:
            continue

        # Populate FX metadata fields
        frame["fx_node_op"] = node_info.get("op")
        frame["fx_node_name"] = node_info.get("name")
        frame["fx_node_target"] = str(node_info.get("target"))

        # Attach original stack trace if available
        original_trace = node_info.get("stack_trace")
        if original_trace:
            frame["fx_original_trace"] = original_trace

        count += 1

    return count


def _augment_memory_snapshot_stack_traces(
    snapshot: str | _Snapshot,
) -> _Snapshot:
    """
    Augment a memory snapshot with original source stack traces from FX metadata.

    IMPORTANT: This function reads from a global in-memory registry (_FX_METADATA_REGISTRY)
    that is populated during graph module compilation. It must be called in the same
    Python process where the FX graphs were compiled. It cannot be used to augment
    snapshots loaded from disk in a different process.

    Args:
        snapshot (str or _Snapshot): Either a memory snapshot dict or path to a snapshot pickle file

    Returns:
        _Snapshot: The augmented snapshot dictionary with fx_node_op, fx_node_name,
            fx_original_trace, and fx_node_info fields added to frames
    """

    snapshot_dict: _Snapshot
    if isinstance(snapshot, str):
        # Load the memory snapshot
        with open(snapshot, "rb") as f:
            snapshot_dict = cast(_Snapshot, pickle.load(f))
    else:
        snapshot_dict = snapshot

    # Process blocks in segments (for regular allocations)
    for segment in snapshot_dict.get("segments", []):
        for block in segment.get("blocks", []):
            if "frames" in block:
                _augment_frames(block["frames"])

    # Process device traces (for memory history)
    for trace_list in snapshot_dict.get("device_traces", []):
        for trace_entry in trace_list:
            if isinstance(trace_entry, dict) and "frames" in trace_entry:
                _augment_frames(trace_entry["frames"])

    return snapshot_dict


def _is_privateuse1_backend_available():
    """
    Determines whether the privateuse1 backend is registered and available.

    Returns:
        Return True if the privateuse1 backend is registered and available.
    """
    privateuse1_backend_name = torch._C._get_privateuse1_backend_name()
    privateuse1_backend_module = getattr(torch, privateuse1_backend_name, None)
    return (
        is_available := getattr(privateuse1_backend_module, "is_available", None)
    ) and is_available()


def getenv(name: str) -> "str | None":
    """Read an environment variable through torch's serialized env access.

    Prefer this over :func:`os.getenv` when torch is loaded: it shares c10's
    environment mutex, so the read is consistent with concurrent C++ code and
    other :func:`torch._utils.setenv` calls that also go through c10.

    Args:
        name: Name of the environment variable.

    Returns:
        The variable's value, or ``None`` if it is not set.
    """
    return torch._C._getenv(name)


def setenv(name: str, value: str, overwrite: bool = True) -> None:
    """Set an environment variable through torch's serialized env access.

    Prefer this over assigning to :data:`os.environ` when torch is loaded: it
    shares c10's environment mutex, so the write is consistent with concurrent
    C++ code and other :func:`torch._utils.getenv` calls that also go through
    c10.

    Args:
        name: Name of the environment variable.
        value: Value to set.
        overwrite: If ``False`` and the variable is already set, leave it
            unchanged. Defaults to ``True``.
    """
    torch._C._setenv(name, value, overwrite)


def unsetenv(name: str) -> None:
    """Remove an environment variable through torch's serialized env access.

    The counterpart to :func:`setenv`; shares c10's environment mutex.

    Args:
        name: Name of the environment variable to remove.
    """
    torch._C._unsetenv(name)


# Saved originals so os.environ interception can be reverted; ``None`` means the
# hook is not currently installed.
_original_os_putenv: "Callable[..., None] | None" = None
_original_os_unsetenv: "Callable[..., None] | None" = None


def _torch_putenv(key: "str | bytes", value: "str | bytes") -> None:
    setenv(os.fsdecode(key), os.fsdecode(value))


def _torch_unsetenv(key: "str | bytes") -> None:
    unsetenv(os.fsdecode(key))


def install_os_environ_hook() -> None:
    """Route ``os.environ`` mutations through torch's serialized env access.

    CPython's ``os.environ.__setitem__`` / ``__delitem__`` call the module-level
    ``os.putenv`` / ``os.unsetenv`` and then update ``os.environ``'s cached dict.
    This replaces those two functions so that mutating ``os.environ`` (and any
    direct ``os.putenv`` / ``os.unsetenv`` call) goes through c10's environment
    mutex, keeping Python and C++ env access consistent. The cached dict is still
    updated by CPython afterwards, so ``os.environ`` reads stay correct.

    Idempotent: calling it again while installed is a no-op. Use
    :func:`remove_os_environ_hook` to revert.
    """
    global _original_os_putenv, _original_os_unsetenv
    if _original_os_putenv is not None:
        return
    _original_os_putenv = os.putenv
    _original_os_unsetenv = os.unsetenv
    os.putenv = _torch_putenv  # type: ignore[assignment]
    os.unsetenv = _torch_unsetenv  # type: ignore[assignment]


def remove_os_environ_hook() -> None:
    """Undo :func:`install_os_environ_hook`; a no-op if it was not installed."""
    global _original_os_putenv, _original_os_unsetenv
    if _original_os_putenv is None or _original_os_unsetenv is None:
        return
    os.putenv = _original_os_putenv
    os.unsetenv = _original_os_unsetenv
    _original_os_putenv = None
    _original_os_unsetenv = None
<|endoftext|>"""Python access to ATen's TensorIterator build pipeline.

:class:`TensorIterator` builds an iterator from a set of operands and flags
that mirror ``at::TensorIteratorConfig``, then exposes the post-build
shape / dtype / device / stride information for inspection.

This is a build-pipeline-only surface: there is no ``for_each`` here. Use it
to debug shape and dtype inference, validate custom-op contracts, or inspect
how ATen would lay out a kernel's iteration.

Construction is canonical, *not* a faithful replay of arbitrary
``TensorIteratorConfig`` call sequences. Operands are always registered in
the order ``outputs -> inputs -> const_inputs`` and setters are applied in
one fixed order. Notably:

* The C++ builder distinguishes ``add_input(a); add_const_input(b)`` from
  ``add_const_input(b); add_input(a)`` -- ``input(0)`` refers to different
  operands. This Python surface cannot express that distinction: every
  ``inputs[i]`` precedes every ``const_inputs[j]``.
* Some C++ setters have order-dependent side effects (e.g.
  ``promote_inputs_to_common_dtype(true)`` also flips
  ``check_all_same_dtype`` to ``false``). The Python surface materializes
  the *final* boolean state of each knob, not the call order, so it can't
  reproduce a sequence where an intermediate setter observed a
  since-overwritten value.

Every in-tree caller of ``at::TensorIteratorConfig`` fits the canonical-
recipe shape; the lossiness is theoretical, not practical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._C import (
    _TensorIterator as _CTensorIterator,
    _TensorIteratorSpec as _CTensorIteratorSpec,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "TensorIterator",
    "binary_op",
    "binary_float_op",
    "comparison_op",
    "unary_op",
    "unary_float_op",
    "nullary_op",
    "reduce_op",
]


class TensorIterator:
    """A built TensorIterator. Read-only view of the build-pipeline result.

    Constructor kwargs mirror ``at::TensorIteratorConfig``; defaults match
    the C++ defaults. See module docstring for canonical-recipe caveats.

    Examples
    --------
    Build a TI for a binary op with type promotion and inspect the result::

        >>> import torch
        >>> from torch._tensor_iterator import TensorIterator
        >>> a = torch.zeros(3, 4, dtype=torch.float32)
        >>> b = torch.zeros(3, 4, dtype=torch.float64)
        >>> it = TensorIterator(
        ...     outputs=[None],
        ...     const_inputs=[a, b],
        ...     promote_inputs_to_common_dtype=True,
        ...     cast_common_dtype_to_outputs=True,
        ... )
        >>> it.common_dtype
        torch.float64
        >>> it.numel
        12
        >>> it.ndim       # contiguous (3, 4) collapses to a single dim
        1

    Declare a static shape (the iterator skips broadcast/coalesce)::

        >>> a = torch.zeros(2, 6)
        >>> out = torch.empty(2, 6)
        >>> it = TensorIterator(
        ...     outputs=[out],
        ...     const_inputs=[a],
        ...     resize_outputs=False,
        ...     static_shape=(2, 6),
        ... )

    Build a reduction TI (output must be pre-allocated to the reduced shape)::

        >>> a = torch.zeros(3, 4)
        >>> out = torch.empty(3, 1)
        >>> it = TensorIterator(
        ...     outputs=[out],
        ...     const_inputs=[a],
        ...     resize_outputs=False,
        ...     is_reduction=True,
        ... )

    Inspect post-coalesce strides (in bytes; use ``element_strides`` for
    element units)::

        >>> a = torch.zeros(3, 4)
        >>> b = torch.zeros(3, 4)
        >>> it = TensorIterator(outputs=[None], const_inputs=[a, b])
        >>> tuple(it.strides(0))           # output byte strides
        (4,)
        >>> it.element_strides(0)          # ... in elements
        (1,)
    """

    def __init__(
        self,
        *,
        outputs: list[torch.Tensor | None] | None = None,
        inputs: list[torch.Tensor] | None = None,
        const_inputs: list[torch.Tensor] | None = None,
        check_all_same_dtype: bool = True,
        check_all_same_device: bool = True,
        promote_inputs_to_common_dtype: bool = False,
        promote_integer_inputs_to_float: bool = False,
        cast_common_dtype_to_outputs: bool = False,
        enforce_safe_casting_to_output: bool = False,
        enforce_linear_iteration: bool = False,
        resize_outputs: bool = True,
        check_mem_overlap: bool = True,
        allow_cpu_scalars: bool = False,
        is_reduction: bool = False,
        static_dtype: torch.dtype | None = None,
        static_device: torch.device | None = None,
        static_shape: Sequence[int] | None = None,
        squash_dims: Sequence[int] = (),
    ) -> None:
        spec = _CTensorIteratorSpec()
        spec.outputs = list(outputs) if outputs is not None else []
        spec.inputs = list(inputs) if inputs is not None else []
        spec.const_inputs = list(const_inputs) if const_inputs is not None else []

        spec.check_all_same_dtype = check_all_same_dtype
        spec.check_all_same_device = check_all_same_device
        spec.promote_inputs_to_common_dtype = promote_inputs_to_common_dtype
        spec.promote_integer_inputs_to_float = promote_integer_inputs_to_float
        spec.cast_common_dtype_to_outputs = cast_common_dtype_to_outputs
        spec.enforce_safe_casting_to_output = enforce_safe_casting_to_output
        spec.enforce_linear_iteration = enforce_linear_iteration
        spec.resize_outputs = resize_outputs
        spec.check_mem_overlap = check_mem_overlap
        spec.allow_cpu_scalars = allow_cpu_scalars
        spec.is_reduction = is_reduction

        if static_dtype is not None:
            spec.static_dtype = static_dtype
        if static_device is not None:
            spec.static_device = static_device
        if static_shape is not None:
            spec.static_shape = list(static_shape)
            spec.squash_dims = list(squash_dims)

        self._impl: _CTensorIterator = spec.build()

    @property
    def ndim(self) -> int:
        return self._impl.ndim

    @property
    def shape(self) -> memoryview:
        """Iterator shape (post coalesce/reorder), as a zero-copy
        ``memoryview`` of ``int64`` elements. The view holds a reference
        to this iterator and keeps it alive for as long as the view is
        reachable; copy via ``tuple(it.shape)`` if you need a snapshot
        you can outlive the iterator with. Hot-path readers (dispatch
        conditionals) get index access without an allocation."""
        return self._impl.shape

    @property
    def numel(self) -> int:
        return self._impl.numel

    @property
    def ntensors(self) -> int:
        return self._impl.ntensors

    @property
    def ninputs(self) -> int:
        return self._impl.ninputs

    @property
    def noutputs(self) -> int:
        return self._impl.noutputs

    @property
    def is_contiguous(self) -> bool:
        return self._impl.is_contiguous

    @property
    def is_trivial_1d(self) -> bool:
        return self._impl.is_trivial_1d

    @property
    def common_dtype(self) -> torch.dtype | None:
        """The inferred computation dtype, or ``None`` if no single dtype
        was inferred. Populated whenever TensorIterator can resolve a
        common dtype -- under promotion flags, or when every input
        already shares a dtype. ``None`` does not mean "promotion was
        not requested"; it means inference produced no answer."""
        return self._impl.common_dtype

    def tensor(self, index: int) -> torch.Tensor:
        """Return the iterator's current operand at ``index``.

        Note: under ``promote_inputs_to_common_dtype`` /
        ``cast_common_dtype_to_outputs`` (CPU paths), this may be a
        promoted/cast kernel temporary rather than the tensor that was
        registered with the config -- it's the iterator's view of the
        operand a kernel would actually iterate over."""
        return self._impl.tensor(index)

    def input(self, index: int = 0) -> torch.Tensor:
        """Return the iterator's current input. See :meth:`tensor` for the
        promoted-temporary caveat."""
        return self._impl.input(index)

    def output(self, index: int = 0) -> torch.Tensor:
        """Return the iterator's current output. See :meth:`tensor` for the
        cast-temporary caveat."""
        return self._impl.output(index)

    def dtype(self, index: int = 0) -> torch.dtype:
        return self._impl.dtype(index)

    def device(self, index: int = 0) -> torch.device:
        return self._impl.device(index)

    def strides(self, index: int) -> memoryview:
        """Per-operand strides in bytes (post reorder/coalesce), as a
        zero-copy ``memoryview`` of ``int64`` elements. The view holds a
        reference to this iterator and keeps it alive for as long as the
        view is reachable; copy via ``tuple(it.strides(i))`` if you need
        a snapshot you can outlive the iterator with."""
        return self._impl.strides(index)

    def element_strides(self, index: int) -> tuple[int, ...]:
        """Per-operand strides in elements (byte stride / element size).

        Allocates a fresh tuple on every call. Don't use on a hot path:
        cache the result, or read :meth:`strides` once and divide by
        :meth:`element_size` of the operand inline."""
        return self._impl.element_strides(index)

    def __repr__(self) -> str:
        return repr(self._impl)


# --- Factory shortcuts. These mirror the C++ named constructors at
# aten/src/ATen/TensorIterator.cpp:1069+. Pass ``out=None`` to ask the iterator
# to allocate a fresh output tensor of the inferred shape/dtype/device.


def binary_op(
    out: torch.Tensor | None, a: torch.Tensor, b: torch.Tensor
) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::binary_op``."""
    return TensorIterator(
        outputs=[out],
        const_inputs=[a, b],
        allow_cpu_scalars=True,
        promote_inputs_to_common_dtype=True,
        cast_common_dtype_to_outputs=True,
        enforce_safe_casting_to_output=True,
    )


def binary_float_op(
    out: torch.Tensor | None, a: torch.Tensor, b: torch.Tensor
) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::binary_float_op``."""
    return TensorIterator(
        outputs=[out],
        const_inputs=[a, b],
        allow_cpu_scalars=True,
        promote_inputs_to_common_dtype=True,
        cast_common_dtype_to_outputs=True,
        enforce_safe_casting_to_output=True,
        promote_integer_inputs_to_float=True,
    )


def comparison_op(
    out: torch.Tensor | None, a: torch.Tensor, b: torch.Tensor
) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::comparison_op``.

    When ``out`` is ``None``, the output dtype is forced to bool. When ``out``
    is a defined non-bool tensor, the common dtype is cast back to its dtype
    via ``cast_common_dtype_to_outputs``. The bool-output case skips that cast
    as a performance optimization.
    """
    static_dtype = torch.bool if out is None else None
    cast_to_outputs = out is not None and out.dtype != torch.bool
    return TensorIterator(
        outputs=[out],
        const_inputs=[a, b],
        allow_cpu_scalars=True,
        promote_inputs_to_common_dtype=True,
        cast_common_dtype_to_outputs=cast_to_outputs,
        static_dtype=static_dtype,
    )


def unary_op(out: torch.Tensor | None, a: torch.Tensor) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::unary_op``."""
    return TensorIterator(outputs=[out], const_inputs=[a])


def unary_float_op(out: torch.Tensor | None, a: torch.Tensor) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::unary_float_op``."""
    return TensorIterator(
        outputs=[out],
        const_inputs=[a],
        promote_inputs_to_common_dtype=True,
        cast_common_dtype_to_outputs=True,
        enforce_safe_casting_to_output=True,
        promote_integer_inputs_to_float=True,
    )


def nullary_op(out: torch.Tensor) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::nullary_op``.

    Unlike the binary/unary factories, ``out`` must be a defined tensor;
    the C++ named constructor takes a non-undefined output.
    """
    if out is None:
        raise TypeError(
            "nullary_op requires a defined output tensor; None is not allowed."
        )
    return TensorIterator(
        outputs=[out],
        check_all_same_dtype=False,
        resize_outputs=False,
    )


def reduce_op(
    out: torch.Tensor,
    a: torch.Tensor,
    *,
    out2: torch.Tensor | None = None,
) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::reduce_op``.

    Pass ``out2`` for the two-output reduction overload (e.g. ``min`` returning
    values + indices). The output tensor(s) must be pre-allocated and shaped
    correctly: this factory does not allocate or resize. With ``out2``, both
    outputs must live on ``a``'s device and share its sizes/strides -- the
    C++ named constructor asserts this and the same checks are mirrored
    here.
    """
    # Mirror the TORCH_INTERNAL_ASSERTs in TensorIterator::reduce_op. We
    # surface them as RuntimeError up front since the binding doesn't
    # rebind the named constructor itself; without this, mismatched
    # out1/out2 would build a kernel-shape that doesn't match the
    # tensors and corrupt memory once a kernel is invoked.
    if out2 is not None:
        if not out.device == a.device == out2.device:
            raise RuntimeError(
                "reduce_op: out, out2, and the input must share a device, "
                f"got out={out.device}, out2={out2.device}, a={a.device}"
            )
        if out.dim() != out2.dim():
            raise RuntimeError(
                f"reduce_op: out and out2 must have the same dim, got "
                f"{out.dim()} and {out2.dim()}"
            )
        if out.shape != out2.shape or out.stride() != out2.stride():
            raise RuntimeError(
                "reduce_op: out and out2 must have identical sizes and strides"
            )
        return TensorIterator(
            outputs=[out, out2],
            const_inputs=[a],
            check_all_same_dtype=False,
            resize_outputs=False,
            check_mem_overlap=False,
            is_reduction=True,
        )
    return TensorIterator(
        outputs=[out],
        const_inputs=[a],
        promote_inputs_to_common_dtype=True,
        resize_outputs=False,
        check_mem_overlap=False,
        is_reduction=True,
    )
<|endoftext|>from collections.abc import Iterable
from typing import Any

from torch._vendor.packaging.version import InvalidVersion, Version
from torch.version import __version__ as internal_version


__all__ = ["TorchVersion"]


class TorchVersion(str):
    """A string with magic powers to compare to both Version and iterables!
    Prior to 1.10.0 torch.__version__ was stored as a str and so many did
    comparisons against torch.__version__ as if it were a str. In order to not
    break them we have TorchVersion which masquerades as a str while also
    having the ability to compare against both packaging.version.Version as
    well as tuples of values, eg. (1, 2, 1)
    Examples:
        Comparing a TorchVersion object to a Version object
            TorchVersion('1.10.0a') > Version('1.10.0a')
        Comparing a TorchVersion object to a Tuple object
            TorchVersion('1.10.0a') > (1, 2)    # 1.2
            TorchVersion('1.10.0a') > (1, 2, 1) # 1.2.1
        Comparing a TorchVersion object against a string
            TorchVersion('1.10.0a') > '1.2'
            TorchVersion('1.10.0a') > '1.2.1'
    """

    __slots__ = ()

    # fully qualified type names here to appease mypy
    def _convert_to_version(self, inp: Any) -> Any:
        if isinstance(inp, Version):
            return inp
        elif isinstance(inp, str):
            return Version(inp)
        elif isinstance(inp, Iterable):
            # Ideally this should work for most cases by attempting to group
            # the version tuple, assuming the tuple looks (MAJOR, MINOR, ?PATCH)
            # Examples:
            #   * (1)         -> Version("1")
            #   * (1, 20)     -> Version("1.20")
            #   * (1, 20, 1)  -> Version("1.20.1")
            return Version(".".join(str(item) for item in inp))
        else:
            raise InvalidVersion(inp)

    def _cmp_wrapper(self, cmp: Any, method: str) -> bool:
        try:
            return getattr(Version(self), method)(self._convert_to_version(cmp))
        except BaseException as e:
            if not isinstance(e, InvalidVersion):
                raise
            # Fall back to regular string comparison if dealing with an invalid
            # version like 'parrot'
            return getattr(super(), method)(cmp)


for cmp_method in ["__gt__", "__lt__", "__eq__", "__ge__", "__le__"]:
    setattr(
        TorchVersion,
        cmp_method,
        lambda x, y, method=cmp_method: x._cmp_wrapper(y, method),
    )

__version__ = TorchVersion(internal_version)
<|endoftext|># mypy: allow-untyped-defs
import functools
from collections.abc import Callable
from typing import Any
from typing_extensions import deprecated

import torch
from torch import Tensor
from torch.utils._pytree import _broadcast_to_and_flatten, tree_flatten, tree_unflatten


in_dims_t = int | tuple
out_dims_t = int | tuple[int, ...]


# Checks that all args-to-be-batched have the same batch dim size
def _validate_and_get_batch_size(
    flat_in_dims: list[int | None],
    flat_args: list,
) -> int:
    batch_sizes = [
        arg.size(in_dim)
        for in_dim, arg in zip(flat_in_dims, flat_args)
        if in_dim is not None
    ]
    if batch_sizes and any(size != batch_sizes[0] for size in batch_sizes):
        raise ValueError(
            f"vmap: Expected all tensors to have the same size in the mapped "
            f"dimension, got sizes {batch_sizes} for the mapped dimension"
        )
    return batch_sizes[0]


def _num_outputs(batched_outputs: Tensor | tuple[Tensor, ...]) -> int:
    if isinstance(batched_outputs, tuple):
        return len(batched_outputs)
    return 1


# If value is a tuple, check it has length `num_elements`.
# If value is not a tuple, make a tuple with `value` repeated `num_elements` times
def _as_tuple(
    value: Any,
    num_elements: int,
    error_message_lambda: Callable[[], str],
) -> tuple:
    if not isinstance(value, tuple):
        return (value,) * num_elements
    if len(value) != num_elements:
        raise ValueError(error_message_lambda())
    return value


# Creates BatchedTensors for every Tensor in arg that should be batched.
# Returns the (potentially) batched arguments and the batch_size.
def _create_batched_inputs(
    in_dims: in_dims_t,
    args: tuple,
    vmap_level: int,
    func: Callable,
) -> tuple[tuple, int]:
    if not isinstance(in_dims, int) and not isinstance(in_dims, tuple):
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"expected `in_dims` to be int or a (potentially nested) tuple "
            f"matching the structure of inputs, got: {type(in_dims)}."
        )
    if len(args) == 0:
        raise ValueError(
            f"vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot to add "
            f"inputs, or you are trying to vmap over a function with no inputs. "
            f"The latter is unsupported."
        )

    flat_args, args_spec = tree_flatten(args)
    flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
    if flat_in_dims is None:
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"in_dims is not compatible with the structure of `inputs`. "
            f"in_dims has structure {tree_flatten(in_dims)[1]} but inputs "
            f"has structure {args_spec}."
        )

    for arg, in_dim in zip(flat_args, flat_in_dims):
        if not isinstance(in_dim, int) and in_dim is not None:
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but in_dim must be either "
                f"an integer dimension or None."
            )
        if isinstance(in_dim, int) and not isinstance(arg, Tensor):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but the input is of type "
                f"{type(arg)}. We cannot vmap over non-Tensor arguments, "
                f"please use None as the respective in_dim"
            )
        if in_dim is not None and (in_dim < 0 or in_dim >= arg.dim()):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for some input, but that input is a Tensor "
                f"of dimensionality {arg.dim()} so expected in_dim to satisfy "
                f"0 <= in_dim < {arg.dim()}."
            )

    batch_size = _validate_and_get_batch_size(flat_in_dims, flat_args)
    # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    batched_inputs = [
        arg if in_dim is None else torch._add_batch_dim(arg, in_dim, vmap_level)
        for in_dim, arg in zip(flat_in_dims, flat_args)
    ]
    return tree_unflatten(batched_inputs, args_spec), batch_size


# Undoes the batching (and any batch dimensions) associated with the `vmap_level`.
def _unwrap_batched(
    batched_outputs: Tensor | tuple[Tensor, ...],
    out_dims: out_dims_t,
    vmap_level: int,
    batch_size: int,
    func: Callable,
    allow_none_pass_through: bool = False,
) -> tuple:
    num_outputs = _num_outputs(batched_outputs)
    out_dims_as_tuple = _as_tuple(
        out_dims,
        num_outputs,
        lambda: f"vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must "
        f"have one dim per output (got {num_outputs} outputs) of {_get_name(func)}.",
    )

    # NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    # There is something wrong with our type bindings for functions that begin
    # with '_', see #40397.
    if isinstance(batched_outputs, Tensor):
        out_dim = out_dims_as_tuple[0]
        return torch._remove_batch_dim(batched_outputs, vmap_level, batch_size, out_dim)  # type: ignore[return-value]
    if allow_none_pass_through:
        return tuple(
            (
                torch._remove_batch_dim(out, vmap_level, batch_size, out_dim)
                if out is not None
                else None
            )
            for out, out_dim in zip(batched_outputs, out_dims_as_tuple)
        )
    else:
        return tuple(
            torch._remove_batch_dim(out, vmap_level, batch_size, out_dim)
            for out, out_dim in zip(batched_outputs, out_dims_as_tuple)
        )


# Checks that `fn` returned one or more Tensors and nothing else.
# NB: A python function that return multiple arguments returns a single tuple,
# so we are effectively checking that `outputs` is a single Tensor or a tuple of
# Tensors.
def _validate_outputs(outputs: Any, func: Callable) -> None:
    if isinstance(outputs, Tensor):
        return
    if not isinstance(outputs, tuple):
        raise ValueError(
            f"vmap({_get_name(func)}, ...): `{_get_name(func)}` must only return "
            f"Tensors, got type {type(outputs)} as the return."
        )
    for idx, output in enumerate(outputs):
        if isinstance(output, Tensor):
            continue
        raise ValueError(
            f"vmap({_get_name(func)}, ...): `{_get_name(func)}` must only return "
            f"Tensors, got type {type(output)} for return {idx}."
        )


def _check_out_dims_is_int_or_int_tuple(out_dims: out_dims_t, func: Callable) -> None:
    if isinstance(out_dims, int):
        return
    if not isinstance(out_dims, tuple) or not all(
        isinstance(out_dim, int) for out_dim in out_dims
    ):
        raise ValueError(
            f"vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must be "
            f"an int or a tuple of int representing where in the outputs the "
            f"vmapped dimension should appear."
        )


def _get_name(func: Callable):
    if hasattr(func, "__name__"):
        return func.__name__

    # Not all callables have __name__, in fact, only static functions/methods do.
    # A callable created via functools.partial or an nn.Module, to name some
    # examples, don't have a __name__.
    return repr(func)


# vmap(func)(inputs) wraps all Tensor inputs to be batched in BatchedTensors,
# sends those into func, and then unwraps the output BatchedTensors. Operations
# on BatchedTensors perform the batched operations that the user is asking for.
@deprecated(
    "Please use `torch.vmap` instead of `torch._vmap_internals.vmap`.",
    category=FutureWarning,
)
def vmap(func: Callable, in_dims: in_dims_t = 0, out_dims: out_dims_t = 0) -> Callable:
    """
    Please use torch.vmap instead of this API.
    """
    return _vmap(func, in_dims, out_dims)


# A version of vmap but without the initial "experimental prototype" warning
def _vmap(
    func: Callable,
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    allow_none_pass_through: bool = False,
) -> Callable:
    # The `allow_none_pass_through` argument is a temporary workaround may be removed.
    # Currently it enables us to wrap the call in `autograd.grad` to the autograd engine,
    # which may return None if any of the inputs are unused. See the issue discussing this:
    # https://github.com/pytorch/functorch/issues/159.
    @functools.wraps(func)
    def wrapped(*args):
        _check_out_dims_is_int_or_int_tuple(out_dims, func)
        vmap_level = torch._C._vmapmode_increment_nesting()
        try:
            batched_inputs, batch_size = _create_batched_inputs(
                in_dims, args, vmap_level, func
            )
            batched_outputs = func(*batched_inputs)
            if not allow_none_pass_through:
                _validate_outputs(batched_outputs, func)
            return _unwrap_batched(
                batched_outputs,
                out_dims,
                vmap_level,
                batch_size,
                func,
                allow_none_pass_through=allow_none_pass_through,
            )
        finally:
            torch._C._vmapmode_decrement_nesting()

    return wrapped
<|endoftext|># mypy: allow-untyped-defs
import ast
import functools
import inspect
from textwrap import dedent
from typing import Any, NamedTuple

from torch._C import ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory


def get_source_lines_and_file(
    obj: Any,
    error_msg: str | None = None,
) -> tuple[list[str], int, str | None]:
    """
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lineno, filename)
    """
    filename = None  # in case getsourcefile throws
    try:
        filename = inspect.getsourcefile(obj)
        sourcelines, file_lineno = inspect.getsourcelines(obj)
    except OSError as e:
        msg = (
            f"Can't get source for {obj}. TorchScript requires source access in "
            "order to carry out compilation, make sure original .py files are "
            "available."
        )
        if error_msg:
            msg += "\n" + error_msg
        raise OSError(msg) from e

    return sourcelines, file_lineno, filename


def normalize_source_lines(sourcelines: list[str]) -> list[str]:
    """
    This helper function accepts a list of source lines. It finds the
    indentation level of the function definition (`def`), then it indents
    all lines in the function body to a point at or greater than that
    level. This allows for comments and continued string literals that
    are at a lower indentation than the rest of the code.
    Args:
        sourcelines: function source code, separated into lines by
                        the '\n' character
    Returns:
        A list of source lines that have been correctly aligned
    """

    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix) :]

    # Find the line and line number containing the function definition
    idx = None
    for i, l in enumerate(sourcelines):
        if l.lstrip().startswith("def"):
            idx = i
            break

    # This will happen when the function is a lambda- we won't find "def" anywhere in the source
    # lines in that case. Currently trying to JIT compile a lambda will throw an error up in
    # `parse_def()`, but we might want to handle this case in the future.
    if idx is None:
        return sourcelines

    # Get a string representing the amount of leading whitespace
    fn_def = sourcelines[idx]
    whitespace = fn_def.split("def")[0]

    # Add this leading whitespace to all lines before and after the `def`
    aligned_prefix = [
        whitespace + remove_prefix(s, whitespace) for s in sourcelines[:idx]
    ]
    aligned_suffix = [
        whitespace + remove_prefix(s, whitespace) for s in sourcelines[idx + 1 :]
    ]

    # Put it together again
    aligned_prefix.append(fn_def)
    return aligned_prefix + aligned_suffix


# Thin wrapper around SourceRangeFactory to store extra metadata
# about the function-to-be-compiled.
class SourceContext(SourceRangeFactory):
    def __init__(
        self,
        source,
        filename,
        file_lineno,
        leading_whitespace_len,
        uses_true_division=True,
        funcname=None,
    ):
        super().__init__(source, filename, file_lineno, leading_whitespace_len)
        self.uses_true_division = uses_true_division
        self.filename = filename
        self.funcname = funcname


@functools.cache
def make_source_context(*args):
    return SourceContext(*args)


def fake_range():
    return SourceContext("", None, 0, 0).make_raw_range(0, 1)


class ParsedDef(NamedTuple):
    ast: ast.Module
    ctx: SourceContext
    source: str
    filename: str | None
    file_lineno: int


def parse_def(fn):
    sourcelines, file_lineno, filename = get_source_lines_and_file(
        fn, ErrorReport.call_stack()
    )
    sourcelines = normalize_source_lines(sourcelines)
    source = "".join(sourcelines)
    dedent_src = dedent(source)
    py_ast = ast.parse(dedent_src)
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise RuntimeError(
            f"Expected a single top-level function: {filename}:{file_lineno}"
        )
    leading_whitespace_len = len(source.split("\n", 1)[0]) - len(
        dedent_src.split("\n", 1)[0]
    )
    ctx = make_source_context(
        source, filename, file_lineno, leading_whitespace_len, True, fn.__name__
    )
    return ParsedDef(py_ast, ctx, source, filename, file_lineno)
<|endoftext|># mypy: allow-untyped-defs
import functools
import logging
import os
import sys
import tempfile
import typing_extensions
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler


_T = TypeVar("_T")
_P = ParamSpec("_P")

log = logging.getLogger(__name__)

if os.environ.get("TORCH_COMPILE_STROBELIGHT", False):
    import shutil

    if not shutil.which("strobeclient"):
        log.info(
            "TORCH_COMPILE_STROBELIGHT is true, but seems like you are not on a FB machine."
        )
    else:
        log.info("Strobelight profiler is enabled via environment variable")
        StrobelightCompileTimeProfiler.enable()

# this arbitrary-looking assortment of functionality is provided here
# to have a central place for overridable behavior. The motivating
# use is the FB build environment, where this source file is replaced
# by an equivalent.


def _compute_torch_parent() -> str:
    torch_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(torch_dir) == "shared":
        return os.path.dirname(os.path.dirname(torch_dir))
    # In scikit-build-core editable installs with redirect mode, binary
    # artifacts (bin/, lib/) are installed to the dist package directory
    # rather than the source tree. Fall back to the installed package
    # location for get_file_path.
    if not os.path.isdir(os.path.join(torch_dir, "bin")):
        try:
            from importlib.metadata import distribution

            installed = str(distribution("torch").locate_file("torch"))
            if os.path.isdir(os.path.join(installed, "bin")):
                return os.path.dirname(installed)
        except Exception:
            pass
    return os.path.dirname(torch_dir)


torch_parent = _compute_torch_parent()


def get_file_path(*path_components: str) -> str:
    return os.path.join(torch_parent, *path_components)


def get_file_path_2(*path_components: str) -> str:
    return os.path.join(*path_components)


def get_writable_path(path: str) -> str:
    if os.access(path, os.W_OK):
        return path
    return tempfile.mkdtemp(suffix=os.path.basename(path))


def prepare_multiprocessing_environment(path: str) -> None:
    pass


def resolve_library_path(path: str) -> str:
    return os.path.realpath(path)


def throw_abstract_impl_not_imported_error(opname, module, context):
    if module in sys.modules:
        raise NotImplementedError(
            f"{opname}: We could not find the fake impl for this operator. "
        )
    else:
        raise NotImplementedError(
            f"{opname}: We could not find the fake impl for this operator. "
            f"The operator specified that you may need to import the '{module}' "
            f"Python module to load the fake impl. {context}"
        )


# NB!  This treats "skip" kwarg specially!!
def compile_time_strobelight_meta(
    phase_name: str,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def compile_time_strobelight_meta_inner(
        function: Callable[_P, _T],
    ) -> Callable[_P, _T]:
        @functools.wraps(function)
        def wrapper_function(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            if "skip" in kwargs and isinstance(
                skip := kwargs["skip"],
                int,
            ):
                kwargs["skip"] = skip + 1

            # This is not needed but we have it here to avoid having profile_compile_time
            # in stack traces when profiling is not enabled.
            if not StrobelightCompileTimeProfiler.enabled:
                return function(*args, **kwargs)

            return StrobelightCompileTimeProfiler.profile_compile_time(
                function, phase_name, *args, **kwargs
            )

        return wrapper_function

    return compile_time_strobelight_meta_inner


# Meta only, see
# https://www.internalfb.com/intern/wiki/ML_Workflow_Observability/User_Guides/Adding_instrumentation_to_your_code/
#
# This will cause an event to get logged to Scuba via the signposts API.  You
# can view samples on the API at https://fburl.com/scuba/workflow_signpost/zh9wmpqs
# we log to subsystem "torch", and the category and name you provide here.
# Each of the arguments translate into a Scuba column.  We're still figuring
# out local conventions in PyTorch, but category should be something like
# "dynamo" or "inductor", and name should be a specific string describing what
# kind of event happened.
#
# Killswitch is at
# https://www.internalfb.com/intern/justknobs/?name=pytorch%2Fsignpost#event
def signpost_event(category: str, name: str, parameters: dict[str, Any]):
    log.info("%s %s: %r", category, name, parameters)


def add_mlhub_insight(category: str, insight: str, insight_description: str):
    pass


def log_compilation_event(metrics):
    log.info("%s", metrics)


def upload_graph(graph):
    pass


def set_pytorch_distributed_envs_from_justknobs():
    pass


def log_export_usage(**kwargs):
    pass


def log_draft_export_usage(**kwargs):
    pass


def log_trace_structured_event(*args, **kwargs) -> None:
    pass


def log_cache_bypass(*args, **kwargs) -> None:
    pass


def log_torchscript_usage(api: str, **kwargs):
    _ = api
    return


def check_if_torch_exportable():
    return False


def export_training_ir_rollout_check() -> bool:
    return True


def full_aoti_runtime_assert() -> bool:
    return True


def log_torch_jit_trace_exportability(
    api: str,
    type_of_export: str,
    export_outcome: str,
    result: str,
):
    _, _, _, _ = api, type_of_export, export_outcome, result
    return


DISABLE_JUSTKNOBS = True


def justknobs_check(name: str, default: bool = True) -> bool:
    """
    This function can be used to killswitch functionality in FB prod,
    where you can toggle this value to False in JK without having to
    do a code push.  In OSS, we always have everything turned on all
    the time, because downstream users can simply choose to not update
    PyTorch.  (If more fine-grained enable/disable is needed, we could
    potentially have a map we lookup name in to toggle behavior.  But
    the point is that it's all tied to source code in OSS, since there's
    no live server to query.)

    This is the bare minimum functionality I needed to do some killswitches.
    We have a more detailed plan at
    https://docs.google.com/document/d/1Ukerh9_42SeGh89J-tGtecpHBPwGlkQ043pddkKb3PU/edit
    In particular, in some circumstances it may be necessary to read in
    a knob once at process start, and then use it consistently for the
    rest of the process.  Future functionality will codify these patterns
    into a better high level API.

    WARNING: Do NOT call this function at module import time, JK is not
    fork safe and you will break anyone who forks the process and then
    hits JK again.
    """
    return default


def justknobs_getval_int(name: str) -> int:
    """
    Read warning on justknobs_check
    """
    return 0


def is_fb_unit_test() -> bool:
    return False


@functools.cache
def max_clock_rate():
    """
    unit: MHz
    """
    if not torch.version.hip:
        from triton.testing import nvsmi

        try:
            return nvsmi(["clocks.max.sm"])[0]
        except FileNotFoundError:
            import pynvml  # type: ignore[import]

            handle = torch.cuda._get_pynvml_handler()
            try:
                return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
            finally:
                pynvml.nvmlShutdown()
    else:
        # Manually set max-clock speeds on ROCm until equivalent nvmsi
        # functionality in triton.testing or via pyamdsmi enablement. Required
        # for test_snode_runtime unit tests.
        gcn_arch = str(torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0])
        if "gfx94" in gcn_arch:
            return 1700
        elif "gfx90a" in gcn_arch:
            return 1700
        elif "gfx908" in gcn_arch:
            return 1502
        elif "gfx12" in gcn_arch:
            return 1700
        elif "gfx11" in gcn_arch:
            return 1700
        elif "gfx103" in gcn_arch:
            return 1967
        elif "gfx101" in gcn_arch:
            return 1144
        elif "gfx95" in gcn_arch:
            return 1700  # TODO: placeholder, get actual value
        else:
            return 1100


def get_mast_job_name_version() -> tuple[str, int] | None:
    return None


TEST_MASTER_ADDR = "127.0.0.1"
TEST_MASTER_PORT = 29500
# USE_GLOBAL_DEPS controls whether __init__.py tries to load
# libtorch_global_deps, see Note [Global dependencies]
USE_GLOBAL_DEPS = True
# USE_RTLD_GLOBAL_WITH_LIBTORCH controls whether __init__.py tries to load
# _C.so with RTLD_GLOBAL during the call to dlopen.
USE_RTLD_GLOBAL_WITH_LIBTORCH = False
# If an op was defined in C++ and extended from Python using the
# torch.library.register_fake, returns if we require that there be a
# m.set_python_module("mylib.ops") call from C++ that associates
# the C++ op with a python module.
REQUIRES_SET_PYTHON_MODULE = False


def maybe_upload_prof_stats_to_manifold(profile_path: str) -> str | None:
    print("Uploading profile stats (fb-only otherwise no-op)")
    return None


def log_chromium_event_internal(
    event: dict[str, Any],
    stack: list[str],
    logger_uuid: str,
    start_time_ns: int,
):
    return None


def record_chromium_event_internal(
    event: dict[str, Any],
):
    return None


def profiler_allow_cudagraph_cupti_lazy_reinit_cuda12():
    return True


def deprecated():
    """
    When we deprecate a function that might still be in use, we make it internal
    by adding a leading underscore. This decorator is used with a private function,
    and creates a public alias without the leading underscore, but has a deprecation
    warning. This tells users "THIS FUNCTION IS DEPRECATED, please use something else"
    without breaking them, however, if they still really really want to use the
    deprecated function without the warning, they can do so by using the internal
    function name.
    """

    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        # Validate naming convention - single leading underscore, not dunder
        if not (func.__name__.startswith("_")):
            raise ValueError(
                "@deprecate must decorate a function whose name "
                "starts with a single leading underscore (e.g. '_foo') as the api should be considered internal for deprecation."
            )

        public_name = func.__name__[1:]  # drop exactly one leading underscore
        module = sys.modules[func.__module__]

        # Don't clobber an existing symbol accidentally.
        if hasattr(module, public_name):
            raise RuntimeError(
                f"Cannot create alias '{public_name}' -> symbol already exists in {module.__name__}. \
                 Please rename it or consult a pytorch developer on what to do"
            )

        warning_msg = f"{func.__name__[1:]} is DEPRECATED, please consider using an alternative API(s). "

        # public deprecated alias
        alias = typing_extensions.deprecated(
            # pyrefly: ignore [bad-argument-type]
            warning_msg,
            category=UserWarning,
            stacklevel=1,
        )(func)

        alias.__name__ = public_name

        # Adjust qualname if nested inside a class or another function
        if "." in func.__qualname__:
            alias.__qualname__ = func.__qualname__.rsplit(".", 1)[0] + "." + public_name
        else:
            alias.__qualname__ = public_name

        setattr(module, public_name, alias)

        return func

    return decorator


def get_default_numa_options():
    """
    When using elastic agent, if no numa options are provided, we will use these
    as the default.

    For external use cases, we return None, i.e. no numa binding. If you would like
    to use torch's automatic numa binding capabilities, you should provide
    NumaOptions to your launch config directly or use the numa binding option
    available in torchrun.

    Must return None or NumaOptions, but not specifying to avoid circular import.
    """
    return None


def log_triton_builds(fail: str | None):
    pass


def find_compile_subproc_binary() -> str | None:
    """
    Allows overriding the binary used for subprocesses
    """
    return None


def get_torch_source_version() -> str:
    """Return the source commit hash for the current PyTorch build."""
    import torch.version as torch_version

    return getattr(torch_version, "git_version", "")
<|endoftext|># mypy: allow-untyped-defs
import re

import torch._C as C


"""
PythonDispatcher class is a thin python-binding to C++ dispatcher and it
is designed to show how dispatcher precompute works. In particular,
it shows for a certain op `foo`, what the computed dispatch table looks
like after user register their kernels to certain dispatch keys.

In the real C++ dispatcher we support many dispatch keys for different
functionalities. For simplicity PythonDispatcher only supports dispatch
keys for a single example of each use case. These use cases are listed below:

- CPU/AutogradCPU: represents in-tree backends which we usually have dedicated inference &
    autograd kernel in pytorch core library.
    E.g. CPU, CUDA
- FPGA/AutogradOther: represents in-tree backends which we usually have backend specific
    inference kernels, but they share the same autograd kernel specified in AutogradOther.
    E.g. FPGA, SparseCsrCPU
- XLA/AutogradXLA: represents out-of-tree backends which we don't have either inference or autograd
    kernel defined in pytorch core library. Backend owner is responsible for registering both
    inference & autograd kernels in their extensions(e.g. torch-xla) for the operators they support.
    E.g. XLA, XPU, MPS
- CompositeExplicitAutograd: alias key mapped to inference kernels of all backends like CPU, CUDA, XLA etc.
    Kernels registered to this key MUST work for inference for all backends.
- Autograd: alias key mapped to autograd of all backends like AutogradCPU, AutogradXLA, AutogradOther.
    Kernels registered to this key MUST work for autograd for all backends.
- CompositeImplicitAutograd: alias key CompositeImplicitAutograd = CompositeExplicitAutograd + Autograd
    Kernels registered to this key MUST work for both inference + autograd for all backends.

Note we only allow registrations to alias keys inside pytorch core library. E.g
you shouldn't register a CompositeImplicitAutograd or CompositeExplicitAutograd
kernel from torch-xla extension, instead you should upstream the kernel into
pytorch/pytorch repo so that it's available for all backends and continuously
tested even without the extension.

Usage:
  dispatcher = PythonDispatcher()
  dispatcher.register(["CPU", "XLA", "CompositeImplicitAutograd"])
  print(dispatcher.dispatchTable()) # This tells you exactly which kernel is used for certain backend.
  # For more debugging information
  # print(dispatcher.keys())
  # print(dispatcher.registrations())
  # print(dispatcher.rawRegistrations())
  # print(dispatcher.rawDispatchTable())
PythonDispatcher calls C++ dispatcher under the hood to precompute dispatch table.
This file only provides the simplified API for developers, relevant test code is located in
test/test_dispatch.py
"""


class PythonDispatcher:
    namespace = "__test__"
    name = "foo"
    # fmt: off
    runtime_keys = [
        "CPU", "AutogradCPU",
        "FPGA", "AutogradOther",
        "XLA", "AutogradXLA",
        "Lazy", "AutogradLazy",
    ]
    # fmt: on
    alias_keys = [
        "CompositeExplicitAutograd",
        "Autograd",
        "CompositeImplicitAutograd",
    ]
    supported_keys = runtime_keys + alias_keys

    def __init__(self) -> None:
        C._dispatch_check_invariants(self.name)  # type: ignore[attr-defined]
        self.ref = C._dispatch_library("FRAGMENT", self.namespace, "")
        self.ref.def_("foo(Tensor x) -> Tensor")

    """
    Returns a list of dispatch keys supported by PythonDispatcher.
    You can register kernels to these keys.
    """

    def keys(self):
        return self.supported_keys

    """
    Register kernels to the target dispatchKeys.
    dispatchKeys(list[str]): a list of dispatch keys that you want to register
      your own kernel. Note that you don't need to write the kernel yourself in
      this PythonDispatcher.E.g. for CPU key, a kernel(e.g fn_CPU for CPU) is
      automatically generated and registered.
    """

    def register(self, dispatchKeys):
        # Overridden is not supported and triggers a warning in C++ dispatcher.
        if len(set(dispatchKeys)) != len(dispatchKeys):
            raise RuntimeError(
                f"Overridden is not allowed but found duplicates in {dispatchKeys}."
            )
        # We currently forbid this in codegen instead of C++ dispatcher.
        if (
            "CompositeImplicitAutograd" in dispatchKeys
            and "CompositeExplicitAutograd" in dispatchKeys
        ):
            raise RuntimeError(
                "Registration to both CompositeImplicitAutograd and CompositeExplicitAutograd is not allowed."
            )
        for key in dispatchKeys:
            if key not in self.supported_keys:
                raise RuntimeError(
                    f"{key} is not supported, please select a dispatch key in {self.supported_keys}."
                )
            self.ref.impl_t_t("foo", dispatch=key, debug="fn_" + key)

    """
    Helper function to format (key, kernel).
    """

    def _format_line(self, key, kernel):
        return f"{key:<15} {kernel}\n"

    """
    Helper function to print a table header.
    """

    def _format_header(self, header):
        s = f"""
{header}
"""
        s += self._format_line("key", "kernel")
        s += "---------------------------\n"
        return s

    """
    Returns raw output of all registration info for debugging only.
    Use registrations() for a simplified version.
    """

    def rawRegistrations(self):
        return C._dispatch_dump(f"{self.namespace}::{self.name}")  # type: ignore[attr-defined]

    """
    Returns raw output of computed dispatch table for debugging only.
    Use dispatchTable() for a simplified version.
    """

    def rawDispatchTable(self):
        return C._dispatch_dump_table(f"{self.namespace}::{self.name}")  # type: ignore[attr-defined]

    """
    Returns a table(str) including all the registrations from users.
    Note this includes registrations to both runtime keys and alias keys.
    """

    def registrations(self):
        output = self._format_header("Registered Kernels")
        state = self.rawRegistrations()
        state_entries = state.split("\n")
        for line in state_entries:
            first = line.split(":")[0]
            if any(first.startswith(k) for k in self.supported_keys):
                kernel = line.split("::")[0].split(" ")[1]
                output += self._format_line(first, kernel)
        return output

    """
    Returns the computed dispatch table(str). Note this only include
    runtime keys, registrations to alias keys have been decoded to their
    mapped runtime keys.
    """

    def dispatchTable(self):
        output = self._format_header("Computed Dispatch Table")
        table = self.rawDispatchTable()
        table_entries = table.split("\n")
        regex = re.compile(r"registered at .*FallbackKernel\.cpp.*(\[)")
        for line in table_entries:
            k = line.split(":")[0]
            if k in self.runtime_keys:
                entry = regex.sub("[", line)
                output += self._format_line(k, entry.split(": ")[1])
        return output
<|endoftext|># mypy: allow-untyped-defs
# Unpickler restricted to loading only state dicts
# Restrict constructing types to a list defined in _get_allowed_globals()
# Restrict BUILD operation to `Tensor`, `Parameter` and `OrderedDict` types only
# Restrict APPEND/APPENDS to `list` (and allowlisted list subclasses)
# In `GLOBALS` operation do not do class lookup by name, but rather rely on dictionary
# defined by `_get_allowed_globals()` method, that contains:
# - torch types (Storage, dtypes, Tensor, `torch.Size`),
# - `torch._utils._rebuild` functions.
# - `torch.nn.Parameter`
# - `collections.Counter`
# - `collections.OrderedDict`
# Additionally, users can use an allowlist for adding classes they have deemed as safe using
# `_add_safe_globals()` (`torch.serialization.add_safe_globals`)
# `_clear_safe_globals()` (`torch.serialization.clear_safe_globals`)
# `_get_safe_globals()` (`torch.serialization.get_safe_globals`)

# Based of https://github.com/python/cpython/blob/main/Lib/pickle.py
# Expected to be useful for loading PyTorch model weights
# For example:
# data = urllib.request.urlopen('https://download.pytorch.org/models/resnet50-0676ba61.pth').read()
# buf = io.BytesIO(data)
# weights = torch.load(buf, weights_only = True)

import functools as _functools
import warnings

from _codecs import encode
from collections import Counter, OrderedDict
from collections.abc import Callable
from pickle import (
    APPEND,
    APPENDS,
    BINFLOAT,
    BINGET,
    BININT,
    BININT1,
    BININT2,
    BINPERSID,
    BINPUT,
    BINUNICODE,
    BUILD,
    bytes_types,
    decode_long,
    EMPTY_DICT,
    EMPTY_LIST,
    EMPTY_SET,
    EMPTY_TUPLE,
    GLOBAL,
    LONG1,
    LONG_BINGET,
    LONG_BINPUT,
    MARK,
    NEWFALSE,
    NEWOBJ,
    NEWTRUE,
    NONE,
    PROTO,
    REDUCE,
    SETITEM,
    SETITEMS,
    SHORT_BINSTRING,
    STOP,
    TUPLE,
    TUPLE1,
    TUPLE2,
    TUPLE3,
    UnpicklingError,
)
from struct import unpack
from sys import maxsize
from typing import Any

import torch
from torch._utils import _sparse_tensors_to_validate, IMPORT_MAPPING, NAME_MAPPING


# modules in this list are never allowed, even if the user attempts to allowlist
# functions/classes from them
_blocklisted_modules = [
    "sys",
    "os",
    "posix",
    "nt",
]

_marked_safe_globals_set: set[Callable | tuple[Callable, str]] = set()


def _add_safe_globals(safe_globals: list[Callable | tuple[Callable, str]]):
    global _marked_safe_globals_set
    _marked_safe_globals_set = _marked_safe_globals_set.union(set(safe_globals))


def _get_safe_globals() -> list[Callable | tuple[Callable, str]]:
    global _marked_safe_globals_set
    return list(_marked_safe_globals_set)


def _clear_safe_globals():
    global _marked_safe_globals_set
    _marked_safe_globals_set = set()


def _remove_safe_globals(
    globals_to_remove: list[Callable | tuple[Callable, str]],
):
    global _marked_safe_globals_set
    _marked_safe_globals_set = _marked_safe_globals_set - set(globals_to_remove)


class _safe_globals:
    def __init__(self, safe_globals: list[Callable | tuple[Callable, str]]):
        self.safe_globals = safe_globals

    def __enter__(self):
        _add_safe_globals(self.safe_globals)

    def __exit__(self, type, value, tb):
        _remove_safe_globals(self.safe_globals)


# Separate from _get_allowed_globals because of the lru_cache on _get_allowed_globals
# For example if user had a script like
#   torch.load(file_a)
#   torch.serialization._add_safe_globals([torch.foo])
#   torch.load(file_b)
# the dynamic additions to safe_globals would not be picked up by
# _get_allowed_globals due to the lru_cache
def _get_user_allowed_globals():
    rc: dict[str, Any] = {}
    for f in _marked_safe_globals_set:
        if isinstance(f, tuple):
            if len(f) != 2:
                raise ValueError(
                    f"Expected tuple of length 2 (global, str of callable full path), but got tuple of length: {len(f)}"
                )
            if type(f[1]) is not str:
                raise TypeError(
                    f"Expected second item in tuple to be str of callable full path, but got: {type(f[1])}"
                )
            f, name = f
            rc[name] = f
        else:
            module, name = f.__module__, f.__qualname__
            rc[f"{module}.{name}"] = f
    return rc


def _tensor_rebuild_functions():
    return {
        torch._utils._rebuild_parameter,
        torch._utils._rebuild_parameter_with_state,
        torch._utils._rebuild_qtensor,
        torch._utils._rebuild_tensor,
