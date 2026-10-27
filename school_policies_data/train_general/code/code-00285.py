    all_named_tensors.update(module.named_buffers(remove_duplicate=False))

    # A map of {tensor: set(all_tied_names)} for all tensor names in the module.
    tensor_to_tied_names_map: dict[Tensor, set[str]] = {}
    for name, tensor in all_named_tensors.items():
        if tensor not in tensor_to_tied_names_map:
            tensor_to_tied_names_map[tensor] = set()
        tensor_to_tied_names_map[tensor].add(name)

    # A map of {tied_name: set(all_tied_names)} for all tensor names in the module.
    # If a name is not tied, it will not be in this map.
    tied_names_map: dict[str, set[str]] = {}
    for tied_names in tensor_to_tied_names_map.values():
        if len(tied_names) > 1:
            for tied_name in tied_names:
                tied_names_map[tied_name] = tied_names

    # Make sure the user didn't pass multiple values for the same tied tensor.
    given_names = set(parameters_and_buffers.keys())
    # same as given_names.intersection(tied_names_map.keys()) but dynamo can't
    # handle that
    given_names_for_tied_tensors: set[str] = set()
    for name in given_names:
        if name in tied_names_map:
            given_names_for_tied_tensors.add(name)

    for given_name in given_names_for_tied_tensors:
        tied_names = tied_names_map[given_name]
        if (
            # Detect if there are multiple keys present for the same tied tensor.
            len(tied_names.intersection(given_names_for_tied_tensors)) > 1
            # Only raise an error if the user passed multiple values for the same tied tensor.
            # If all given values are the same, don't raise.
            and len({parameters_and_buffers[tied_name] for tied_name in tied_names})
            != 1
        ):
            raise ValueError(
                f"functional_call got multiple values for keys {sorted(tied_names)}, "
                f"which are tied. Consider using tie_weights=False"
            )

    # Untie the given named tensor map
    # Make a copy for not modifying the original dict
    untied_parameters_and_buffers = parameters_and_buffers.copy()
    for given_name in given_names_for_tied_tensors:
        for tied_name in tied_names_map[given_name]:
            untied_parameters_and_buffers[tied_name] = parameters_and_buffers[
                given_name
            ]
    return untied_parameters_and_buffers


@contextlib.contextmanager
def _reparametrize_module(
    module: "torch.nn.Module",
    parameters_and_buffers: dict[str, Tensor],
    tie_weights: bool = False,
    strict: bool = False,
    stack_weights: bool = False,
):
    if tie_weights:
        untied_parameters_and_buffers = _untie_named_tensors_map(
            module, parameters_and_buffers
        )
    else:
        untied_parameters_and_buffers = parameters_and_buffers

    accessor = NamedMemberAccessor(module)
    if strict:
        missing_keys, unexpected_keys = accessor.check_keys(
            untied_parameters_and_buffers
        )
        error_msgs = []
        if len(unexpected_keys) > 0:
            error_msgs.append(
                f"Unexpected key(s): {', '.join(map(repr, unexpected_keys))}."
            )
        if len(missing_keys) > 0:
            error_msgs.append(f"Missing key(s): {', '.join(map(repr, missing_keys))}.")
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in reparametrizing for {}:\n\t{}".format(
                    module._get_name(), "\n\t".join(error_msgs)
                )
            )

    orig_parameters_and_buffers: dict[str, Tensor] = {}
    try:
        orig_parameters_and_buffers, _ = accessor.swap_tensors_dict(
            untied_parameters_and_buffers, allow_missing=True
        )
        yield
    finally:
        if stack_weights:
            # When stacking is enabled, we will restore the weights in LIFO order.
            orig_parameters_and_buffers = dict(
                reversed(orig_parameters_and_buffers.items())
            )
        new_parameters_and_buffers, _ = accessor.swap_tensors_dict(
            orig_parameters_and_buffers, allow_missing=True
        )
        # Sometimes the module is not completely stateless and has some in-place modifications on
        # the _parameters and _buffers dictionaries.
        # Write the changed parameters and buffers back to the original dict.
        parameters_and_buffers.update(
            {
                k: new_parameters_and_buffers[k]
                for k in parameters_and_buffers
                if k in new_parameters_and_buffers
            }
        )


@deprecated(
    "`torch.nn.utils.stateless.functional_call` is deprecated as of PyTorch 2.0 "
    "and will be removed in a future version of PyTorch. "
    "Please use `torch.func.functional_call` instead which is a drop-in replacement.",
    category=FutureWarning,
)
def functional_call(
    module: "torch.nn.Module",
    parameters_and_buffers: dict[str, Tensor],
    args: Any | tuple | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
):
    r"""Perform a functional call on the module by replacing the module parameters and buffers with the provided ones.

    .. warning::

        This API is deprecated as of PyTorch 2.0 and will be removed in a future
        version of PyTorch. Please use :func:`torch.func.functional_call` instead,
        which is a drop-in replacement for this API.

    .. note:: If the module has active parametrizations, passing a value in the
        :attr:`parameters_and_buffers` argument with the name set to the regular parameter
        name will completely disable the parametrization.
        If you want to apply the parametrization function to the value passed
        please set the key as ``{submodule_name}.parametrizations.{parameter_name}.original``.

    .. note:: If the module performs in-place operations on parameters/buffers, these will be reflected
        in the `parameters_and_buffers` input.

        Example::

            >>> a = {'foo': torch.zeros(())}
            >>> # xdoctest: +SKIP
            >>> mod = Foo()  # does self.foo = self.foo + 1
            >>> print(mod.foo)  # tensor(0.)
            >>> functional_call(mod, a, torch.ones(()))
            >>> print(mod.foo)  # tensor(0.)
            >>> print(a['foo'])  # tensor(1.)

    .. note:: If the module has tied weights, whether or not functional_call respects the tying is determined by the
        tie_weights flag.

        Example::

            >>> a = {'foo': torch.zeros(())}
            >>> # xdoctest: +SKIP
            >>> mod = Foo()  # has both self.foo and self.foo_tied which are tied. Returns x + self.foo + self.foo_tied
            >>> print(mod.foo)  # tensor(1.)
            >>> mod(torch.zeros(()))  # tensor(2.)
            >>> functional_call(mod, a, torch.zeros(()))  # tensor(0.) since it will change self.foo_tied too
            >>> functional_call(mod, a, torch.zeros(()), tie_weights=False)  # tensor(1.)--self.foo_tied is not updated
            >>> new_a = {'foo': torch.zeros(()), 'foo_tied': torch.zeros(())}
            >>> functional_call(mod, new_a, torch.zeros()) # tensor(0.)

    Args:
        module (torch.nn.Module): the module to call
        parameters_and_buffers (dict of str and Tensor): the parameters that will be used in
            the module call.
        args (Any or tuple): arguments to be passed to the module call. If not a tuple, considered a single argument.
        kwargs (dict): keyword arguments to be passed to the module call
        tie_weights (bool, optional): If True, then parameters and buffers tied in the original model will be treated as
            tied in the reparameterized version. Therefore, if True and different values are passed for the tied
            parameters and buffers, it will error. If False, it will not respect the originally tied parameters and
            buffers unless the values passed for both weights are the same. Default: True.
        strict (bool, optional): If True, then the parameters and buffers passed in must match the parameters and
            buffers in the original module. Therefore, if True and there are any missing or unexpected keys, it will
            error. Default: False.

    Returns:
        Any: the result of calling ``module``.
    """
    return _functional_call(
        module,
        parameters_and_buffers,
        args,
        kwargs,
        tie_weights=tie_weights,
        strict=strict,
    )


def _functional_call(
    module: "torch.nn.Module",
    parameters_and_buffers: dict[str, Tensor],
    args: Any | tuple | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
):
    # TODO allow kwargs such as unsafe and others for parametrization
    if (
        torch.jit.is_tracing()
        or torch.jit.is_scripting()
        or isinstance(
            module,
            (
                torch.jit.RecursiveScriptModule,
                torch.jit.ScriptModule,
                torch.jit.ScriptFunction,
            ),
        )
    ):
        raise RuntimeError("The stateless API can't be used with Jitted modules")
    if isinstance(module, torch.nn.DataParallel):
        raise RuntimeError(
            "The stateless API can't be used with nn.DataParallel module"
        )
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)
    with _reparametrize_module(
        module, parameters_and_buffers, tie_weights=tie_weights, strict=strict
    ):
        return module(*args, **kwargs)
<|endoftext|>import warnings
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, TypeVar
from typing_extensions import Self

import torch
from torch import _VF, Tensor
from torch.utils._typing_utils import copy_method_params


__all__ = [
    "PackedSequence",
    "invert_permutation",
    "pack_padded_sequence",
    "pad_packed_sequence",
    "pad_sequence",
    "unpad_sequence",
    "pack_sequence",
    "unpack_sequence",
]

_T = TypeVar("_T")
_R = TypeVar("_R")


class PackedSequence_(NamedTuple):
    data: torch.Tensor
    batch_sizes: torch.Tensor
    sorted_indices: torch.Tensor | None
    unsorted_indices: torch.Tensor | None


def bind(optional: _T | None, fn: Callable[[_T], _R]) -> _R | None:
    if optional is None:
        return None
    return fn(optional)


class PackedSequence(PackedSequence_):
    r"""Holds the data and list of :attr:`batch_sizes` of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data ``abc`` and ``x``
        the :class:`PackedSequence` would contain data ``axbc`` with
        ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Tensor): Tensor containing packed sequence
        batch_sizes (Tensor): Tensor of integers holding
            information about the batch size at each sequence step
        sorted_indices (Tensor, optional): Tensor of integers holding how this
            :class:`PackedSequence` is constructed from sequences.
        unsorted_indices (Tensor, optional): Tensor of integers holding how this
            to recover the original sequences with correct order.

    .. note::
        :attr:`data` can be on arbitrary device and of arbitrary dtype.
        :attr:`sorted_indices` and :attr:`unsorted_indices` must be ``torch.int64``
        tensors on the same device as :attr:`data`.

        However, :attr:`batch_sizes` should always be a CPU ``torch.int64`` tensor.

        This invariant is maintained throughout :class:`PackedSequence` class,
        and all functions that construct a :class:`PackedSequence` in PyTorch
        (i.e., they only pass in tensors conforming to this constraint).
    """

    def __new__(
        cls,
        data: Tensor,
        batch_sizes: Tensor | None = None,
        sorted_indices: Tensor | None = None,
        unsorted_indices: Tensor | None = None,
    ) -> Self:
        return super().__new__(
            cls,
            *_packed_sequence_init_args(
                data, batch_sizes, sorted_indices, unsorted_indices
            ),
        )

    # NOTE [ device and dtype of a PackedSequence ]
    #
    # See the note above in doc string (starting with ":attr:`data` can be on
    # arbitrary device...").
    def pin_memory(self) -> Self:
        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(
            self.data.pin_memory(),
            self.batch_sizes,
            bind(self.sorted_indices, lambda t: t.pin_memory()),
            bind(self.unsorted_indices, lambda t: t.pin_memory()),
        )

    @copy_method_params(torch.Tensor.to)
    def to(self, *args: Any, **kwargs: Any) -> Self:
        r"""Perform dtype and/or device conversion on `self.data`.

        It has similar signature as :meth:`torch.Tensor.to`

        .. note::

            If the ``self.data`` Tensor already has the correct :class:`torch.dtype`
            and :class:`torch.device`, then ``self`` is returned.
            Otherwise, returns a copy with the desired configuration.
        """

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        data = self.data.to(*args, **kwargs)
        if data is self.data:
            return self
        else:
            _device, _dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
                *args, **kwargs
            )

            # Does not forward device or dtype arg/kwargs, device is set from data.device
            def call_to(t: torch.Tensor) -> torch.Tensor:
                return t.to(
                    data.device,
                    non_blocking=non_blocking,
                    memory_format=convert_to_format,
                )

            sorted_indices = bind(self.sorted_indices, call_to)
            unsorted_indices = bind(self.unsorted_indices, call_to)
            return type(self)(data, self.batch_sizes, sorted_indices, unsorted_indices)

    @copy_method_params(torch.Tensor.cuda)
    def cuda(self, *args: Any, **kwargs: Any) -> Self:
        # Tests to see if 'cuda' should be added to kwargs
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args, **kwargs
        )
        if ex.is_cuda:
            return self.to(*args, **kwargs)
        kwargs["device"] = "cuda"
        return self.to(*args, **kwargs)

    @copy_method_params(torch.Tensor.cpu)
    def cpu(self, *args: Any, **kwargs: Any) -> Self:
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args, **kwargs
        )
        if ex.device.type == "cpu":
            return self.to(*args, **kwargs)
        kwargs["device"] = "cpu"
        return self.to(*args, **kwargs)

    def double(self) -> Self:
        return self.to(dtype=torch.double)

    def float(self) -> Self:
        return self.to(dtype=torch.float)

    def half(self) -> Self:
        return self.to(dtype=torch.half)

    def long(self) -> Self:
        return self.to(dtype=torch.long)

    def int(self) -> Self:
        return self.to(dtype=torch.int)

    def short(self) -> Self:
        return self.to(dtype=torch.short)

    def char(self) -> Self:
        return self.to(dtype=torch.int8)

    def byte(self) -> Self:
        return self.to(dtype=torch.uint8)

    @property
    def is_cuda(self) -> bool:
        r"""Return true if `self.data` stored on a gpu."""
        return self.data.is_cuda

    def is_pinned(self) -> bool:
        r"""Return true if `self.data` stored on in pinned memory."""
        return self.data.is_pinned()


# TorchScript doesn't support constructors on named tuples, so we use this helper
# method to construct PackedSequence
def _packed_sequence_init_args(
    data: Tensor,
    batch_sizes: Tensor | None = None,
    sorted_indices: Tensor | None = None,
    unsorted_indices: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    # NB: if unsorted_indices is provided, it should be the inverse permutation
    # to sorted_indices. Don't assert it here because the PackedSequence ctor
    # should only be used internally.

    if unsorted_indices is None:
        unsorted_indices = invert_permutation(sorted_indices)

    # support being called as `PackedSequence(data, batch_sizes, sorted_indices)`
    if batch_sizes is not None:
        # TODO: Re-enable this check (.type isn't supported in TorchScript)
        if batch_sizes.device.type != "cpu":
            raise ValueError(
                "batch_sizes should always be on CPU. "
                "Instances of PackedSequence should never be created manually. "
                "They should be instantiated by functions like pack_sequence "
                "and pack_padded_sequences in nn.utils.rnn. "
                "https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_sequence"
            )
        return data, batch_sizes, sorted_indices, unsorted_indices

    # support being called as `PackedSequence((data, batch_sizes), *, sorted_indices)`
    else:
        if not (isinstance(data, (list, tuple)) and len(data) == 2):
            raise AssertionError("Expected data to be a list or tuple of length 2")
        return data[0], data[1], sorted_indices, unsorted_indices


def _packed_sequence_init(
    data: Tensor,
    batch_sizes: Tensor | None = None,
    sorted_indices: Tensor | None = None,
    unsorted_indices: Tensor | None = None,
) -> PackedSequence:
    data, batch_sizes, sorted_indices, unsorted_indices = _packed_sequence_init_args(
        data, batch_sizes, sorted_indices, unsorted_indices
    )
    return PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)


def invert_permutation(permutation: Tensor | None) -> Tensor | None:
    """Returns the inverse of ``permutation``.

    This is useful for converting between sorted and unsorted indices in
    a :class:`~nn.utils.rnn.PackedSequence`.

    Args:
        permutation (Tensor, optional): a 1-D tensor of indices to invert
    """
    if permutation is None:
        return None
    output = torch.empty_like(permutation, memory_format=torch.legacy_contiguous_format)
    output.scatter_(
        0, permutation, torch.arange(0, permutation.numel(), device=permutation.device)
    )
    return output


def pack_padded_sequence(
    input: Tensor,
    lengths: Tensor | list[int],
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    r"""Packs a Tensor containing padded sequences of variable length.

    :attr:`input` can be of size ``T x B x *`` (if :attr:`batch_first` is ``False``)
    or ``B x T x *`` (if :attr:`batch_first` is ``True``) where ``T`` is the length
    of the longest sequence, ``B`` is the batch size, and ``*`` is any number of dimensions
    (including 0).

    For unsorted sequences, use `enforce_sorted = False`. If :attr:`enforce_sorted` is
    ``True``, the sequences should be sorted by length in a decreasing order, i.e.
    ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the shortest
    one. `enforce_sorted = True` is only necessary for ONNX export.

    It is an inverse operation to :func:`pad_packed_sequence`, and hence :func:`pad_packed_sequence`
    can be used to recover the underlying tensor packed in :class:`PackedSequence`.

    Note:
        This function accepts any input that has at least two dimensions. You
        can apply it to pack the labels, and use the output of the RNN with
        them to compute the loss directly. A Tensor can be retrieved from
        a :class:`PackedSequence` object by accessing its ``.data`` attribute.

    Args:
        input (Tensor): padded batch of variable length sequences.
        lengths (Tensor or list(int)): list of sequence lengths of each batch
            element (must be on the CPU if provided as a tensor).
        batch_first (bool, optional): if ``True``, the input is expected in ``B x T x *``
            format, ``T x B x *`` otherwise. Default: ``False``.
        enforce_sorted (bool, optional): if ``True``, the input is expected to
            contain sequences sorted by length in a decreasing order. If
            ``False``, the input will get sorted unconditionally. Default: ``True``.

    .. warning::
        The dim of ``input`` tensor will be truncated if its length larger than
        correspond value in ``length``.

    Returns:
        a :class:`PackedSequence` object
    """
    if not isinstance(lengths, torch.Tensor):
        if torch._C._get_tracing_state():
            warnings.warn(
                "pack_padded_sequence has been called with a Python list of "
                "sequence lengths. The tracer cannot track the data flow of Python "
                "values, and it will treat them as constants, likely rendering "
                "the trace incorrect for any other combination of lengths.",
                stacklevel=2,
            )
        lengths = torch.as_tensor(lengths, dtype=torch.int64, device="cpu")
    else:
        lengths = lengths.to(dtype=torch.int64)

    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = _VF._pack_padded_sequence(input, lengths, batch_first)
    return _packed_sequence_init(data, batch_sizes, sorted_indices, None)


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: int | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Pad a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Tensor's data will be of size ``T x B x *`` (if :attr:`batch_first` is ``False``)
    or ``B x T x *`` (if :attr:`batch_first` is ``True``) , where ``T`` is the length of the longest
    sequence and ``B`` is the batch size.

    Example:
        >>> from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        >>> seq = torch.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]])
        >>> lens = [2, 1, 3]
        >>> packed = pack_padded_sequence(
        ...     seq, lens, batch_first=True, enforce_sorted=False
        ... )
        >>> packed
        PackedSequence(data=tensor([4, 1, 3, 5, 2, 6]), batch_sizes=tensor([3, 2, 1]),
                       sorted_indices=tensor([2, 0, 1]), unsorted_indices=tensor([1, 2, 0]))
        >>> seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
        >>> seq_unpacked
        tensor([[1, 2, 0],
                [3, 0, 0],
                [4, 5, 6]])
        >>> lens_unpacked
        tensor([2, 1, 3])

    .. note::
        :attr:`total_length` is useful to implement the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`this FAQ section <pack-rnn-unpack-with-data-parallelism>` for
        details.

    Args:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise.
        padding_value (float, optional): values for padded elements.
        total_length (int, optional): if not ``None``, the output will be padded to
            have length :attr:`total_length`. This method will throw :class:`ValueError`
            if :attr:`total_length` is less than the max sequence length in
            :attr:`sequence`.

    Returns:
        Tuple of Tensor containing the padded sequence, and a Tensor
        containing the list of lengths of each sequence in the batch.
        Batch elements will be re-ordered as they were ordered originally when
        the batch was passed to ``pack_padded_sequence`` or ``pack_sequence``.
    """
    max_seq_length = sequence.batch_sizes.size(0)
    if total_length is not None:
        if total_length < max_seq_length:
            raise ValueError(
                "Expected total_length to be at least the length "
                "of the longest sequence in input, but got "
                f"total_length={total_length} and max sequence length being {max_seq_length}"
            )
        max_seq_length = total_length
    padded_output, lengths = _VF._pad_packed_sequence(
        sequence.data, sequence.batch_sizes, batch_first, padding_value, max_seq_length
    )
    unsorted_indices = sequence.unsorted_indices
    if unsorted_indices is not None:
        batch_dim = 0 if batch_first else 1
        return (
            padded_output.index_select(batch_dim, unsorted_indices),
            lengths[unsorted_indices.cpu()],
        )
    return padded_output, lengths


# NOTE: for JIT-compatibility, we need to be more restrictive here and use specific types instead of Iterable.
def pad_sequence(
    sequences: Tensor | list[Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = "right",
) -> Tensor:
    r"""Pad a list of variable length Tensors with :attr:`padding_value`.

    ``pad_sequence`` stacks a list of Tensors along a new dimension, and pads them
    to equal length. :attr:`sequences` can be list of sequences with size ``L x *``,
    where `L` is length of the sequence and ``*`` is any number of dimensions
    (including ``0``). If :attr:`batch_first` is ``False``, the output is of size
    ``T x B x *``, and ``B x T x *`` otherwise, where ``B`` is the batch size
    (the number of elements in :attr:`sequences`), ``T`` is the length of the longest
    sequence.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise.
        padding_value (float, optional): value for padded elements. Default: ``0``.
        padding_side (str, optional): the side to pad the sequences on.
            Default: ``'right'``.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """
    if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        # JIT doesn't support `Iterable`
        if not isinstance(sequences, Iterable):
            msg = (
                "pad_sequence: Expected iterable for input sequences, but got arg of type: "
                f"{type(sequences)}"
            )
            raise RuntimeError(msg)

        # In JIT context this leads to,
        # RuntimeError: cannot statically infer the expected size of a list in this context
        sequences = tuple(sequences)  # type: ignore[assignment]
    else:
        # For JIT, we only support Union[Tensor, Tuple[Tensor]]
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.unbind(0)  # type: ignore[assignment]

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    return torch._C._nn.pad_sequence(
        sequences,  # type: ignore[arg-type]
        batch_first,
        padding_value,
        padding_side,  # type: ignore[arg-type]
    )


def unpad_sequence(
    padded_sequences: Tensor,
    lengths: Tensor,
    batch_first: bool = False,
) -> list[Tensor]:
    r"""Unpad padded Tensor into a list of variable length Tensors.

    ``unpad_sequence`` unstacks padded Tensor into a list of variable length Tensors.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> sequences = [a, b, c]
        >>> padded_sequences = pad_sequence(sequences)
        >>> lengths = torch.as_tensor([v.size(0) for v in sequences])
        >>> unpadded_sequences = unpad_sequence(padded_sequences, lengths)
        >>> torch.allclose(sequences[0], unpadded_sequences[0])
        True
        >>> torch.allclose(sequences[1], unpadded_sequences[1])
        True
        >>> torch.allclose(sequences[2], unpadded_sequences[2])
        True

    Args:
        padded_sequences (Tensor): padded sequences.
        lengths (Tensor): length of original (unpadded) sequences.
        batch_first (bool, optional): whether batch dimension first or not. Default: ``False``.

    Returns:
        a list of :class:`Tensor` objects
    """
    unpadded_sequences = []

    if not batch_first:
        padded_sequences.transpose_(0, 1)

    max_length = padded_sequences.shape[1]
    idx = torch.arange(max_length, device=lengths.device)

    for seq, length in zip(padded_sequences, lengths, strict=True):
        mask = idx < length
        unpacked_seq = seq[mask]
        unpadded_sequences.append(unpacked_seq)

    return unpadded_sequences


def pack_sequence(
    sequences: list[Tensor],
    enforce_sorted: bool = True,
) -> PackedSequence:
    r"""Packs a list of variable length Tensors.

    Consecutive call of the next functions: ``pad_sequence``, ``pack_padded_sequence``.

    ``sequences`` should be a list of Tensors of size ``L x *``, where `L` is
    the length of a sequence and `*` is any number of trailing dimensions,
    including ``0``.

    For unsorted sequences, use `enforce_sorted = False`. If ``enforce_sorted``
    is ``True``, the sequences should be sorted in the order of decreasing length.
    ``enforce_sorted = True`` is only necessary for ONNX export.

    Example:
        >>> from torch.nn.utils.rnn import pack_sequence
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5])
        >>> c = torch.tensor([6])
        >>> pack_sequence([a, b, c])
        PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)

    Args:
        sequences (list[Tensor]): A list of sequences of decreasing length.
        enforce_sorted (bool, optional): if ``True``, checks that the input
            contains sequences sorted by length in a decreasing order. If
            ``False``, this condition is not checked. Default: ``True``.

    Returns:
        a :class:`PackedSequence` object
    """
    lengths = torch.as_tensor([v.size(0) for v in sequences])
    return pack_padded_sequence(
        pad_sequence(sequences), lengths, enforce_sorted=enforce_sorted
    )


def unpack_sequence(packed_sequences: PackedSequence) -> list[Tensor]:
    r"""Unpack PackedSequence into a list of variable length Tensors.

    ``packed_sequences`` should be a PackedSequence object.

    Example:
        >>> from torch.nn.utils.rnn import pack_sequence, unpack_sequence
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5])
        >>> c = torch.tensor([6])
        >>> sequences = [a, b, c]
        >>> print(sequences)
        [tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]
        >>> packed_sequences = pack_sequence(sequences)
        >>> print(packed_sequences)
        PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)
        >>> unpacked_sequences = unpack_sequence(packed_sequences)
        >>> print(unpacked_sequences)
        [tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]

    Args:
        packed_sequences (PackedSequence): A PackedSequence object.

    Returns:
        a list of :class:`Tensor` objects
    """
    padded_sequences, lengths = pad_packed_sequence(packed_sequences, batch_first=True)
    unpacked_sequences = unpad_sequence(padded_sequences, lengths, batch_first=True)
    return unpacked_sequences
<|endoftext|># This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable

import torch


_MISSING: torch.Tensor = object()  # type: ignore[assignment]


def set_tensor(module: "torch.nn.Module", name: str, tensor: torch.Tensor) -> None:
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{module} is not an instance of torch.nn.Module")
    if not isinstance(tensor, torch.Tensor) and tensor is not None:
        raise TypeError(f"{tensor} is not an instance of torch.Tensor")
    if "." in name:
        raise KeyError('tensor name can\'t contain "."')
    if name == "":
        raise KeyError('tensor name can\'t be empty string ""')
    if name in module._parameters:
        module._parameters[name] = tensor  # type: ignore[assignment]
    elif name in module._buffers:
        module._buffers[name] = tensor
    else:
        setattr(module, name, tensor)


def swap_tensor(
    module: "torch.nn.Module",
    name: str,
    tensor: torch.Tensor,
    allow_missing: bool = False,
) -> torch.Tensor:
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{module} is not an instance of torch.nn.Module")
    if (
        tensor is not _MISSING
        and not isinstance(tensor, torch.Tensor)
        and tensor is not None
    ):
        raise TypeError(f"{tensor} is not an instance of torch.Tensor")
    if "." in name:
        raise KeyError('tensor name can\'t contain "."')
    if name == "":
        raise KeyError('tensor name can\'t be empty string ""')

    orig_tensor: torch.Tensor
    if name in module._parameters:
        orig_tensor = module._parameters[name]  # type: ignore[assignment]
        if tensor is not _MISSING:
            module._parameters[name] = tensor  # type: ignore[assignment]
        else:
            del module._parameters[name]
    elif name in module._buffers:
        orig_tensor = module._buffers[name]  # type: ignore[assignment]
        if tensor is not _MISSING:
            module._buffers[name] = tensor
        else:
            del module._buffers[name]
    else:
        if hasattr(module, name):
            orig_tensor = getattr(module, name)
        else:
            if not allow_missing:
                raise AttributeError(f"{module._get_name()} has no attribute `{name}`")
            orig_tensor = _MISSING
        if (
            orig_tensor is not _MISSING
            and not isinstance(orig_tensor, torch.Tensor)
            and orig_tensor is not None
        ):
            raise TypeError(
                f"attribute `{name}`: {orig_tensor} is not an instance of torch.Tensor"
            )
        if tensor is not _MISSING:
            setattr(module, name, tensor)
        elif hasattr(module, name):
            delattr(module, name)
    # pyrefly: ignore [bad-return]
    return orig_tensor


def swap_submodule(
    module: "torch.nn.Module",
    name: str,
    submodule: "torch.nn.Module",
) -> "torch.nn.Module":
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{module} is not an instance of torch.nn.Module")
    if not isinstance(submodule, torch.nn.Module):
        raise TypeError(f"{submodule} is not an instance of torch.nn.Module")
    if "." in name:
        raise KeyError('submodule name can\'t contain "."')
    if name == "":
        raise KeyError('submodule name can\'t be empty string ""')
    if name not in module._modules:
        raise KeyError(f"submodule {name} does not exist")

    orig_submodule = module._modules[name]
    if not isinstance(orig_submodule, torch.nn.Module):
        raise TypeError(f"{name} attribute is not an instance of torch.nn.Module")
    module._modules[name] = submodule
    return orig_submodule


class NamedMemberAccessor:
    """
    A class that provides a way to access the submodules and parameters/buffers of a module.

    It provides caching mechanism to speed up submodule lookups.
    This is useful for functional programming to manipulate the module state.
    """

    def __init__(self, module: "torch.nn.Module") -> None:
        self.module = module
        self.memo: dict[str, torch.nn.Module] = {}

    # Nested attribute access

    def get_submodule(self, name: str) -> "torch.nn.Module":
        """
        Return the submodule specified by the given path.

        For example, to get the submodule mod.layer1.conv1,
        use accessor.get_submodule("layer1.conv1")

        Compare to mod.get_submodule("layer1.conv1"), this method will cache the
        intermediate submodule access to speed up future lookups.
        """
        if not name:
            return self.module

        if name in self.memo:
            return self.memo[name]
        else:
            prefix, dot, attr = name.rpartition(".")
            if dot:
                module = self.get_submodule(prefix)
            else:
                module = self.module
            try:
                submodule = getattr(module, attr)
            except AttributeError as ex:
                raise AttributeError(
                    f"{module._get_name()} has no attribute `{attr}`"
                ) from ex
            if not isinstance(submodule, torch.nn.Module):
                raise TypeError(
                    f"submodule `{name}`: {submodule} is not an instance of torch.nn.Module"
                )
            self.memo[name] = submodule
            return submodule

    def swap_submodule(self, path: str, value: "torch.nn.Module") -> "torch.nn.Module":
        """
        Swap the submodule specified by the given ``path`` to ``value``.

        For example, to swap the attribute mod.layer1.conv1 use
        ``accessor.swap_submodule("layer1.conv1", conv2)``.
        """
        prefix, _, attr = path.rpartition(".")
        return swap_submodule(self.get_submodule(prefix), attr, value)

    def get_tensor(self, name: str) -> torch.Tensor:
        """
        Get the tensor specified by the given path to value.

        For example, to get the attribute mod.layer1.conv1.weight,
        use accessor.get_tensor('layer1.conv1.weight')

        Compare to mod.get_parameter("layer1.conv1.weight"), this method will
        cache the intermediate submodule access to speed up future lookups.
        """
        prefix, _, attr = name.rpartition(".")
        submodule = self.get_submodule(prefix)
        try:
            tensor = getattr(submodule, attr)
        except AttributeError as ex:
            raise AttributeError(
                f"{submodule._get_name()} has no attribute `{name}`"
            ) from ex
        if not isinstance(tensor, torch.Tensor) and tensor is not None:
            raise TypeError(f"{tensor} is not an instance of torch.Tensor")
        return tensor  # type: ignore[return-value]

    def set_tensor(self, name: str, value: torch.Tensor) -> None:
        """
        Set the attribute specified by the given path to value.

        For example, to set the attribute mod.layer1.conv1.weight,
        use accessor.set_tensor("layer1.conv1.weight", value)
        """
        prefix, _, attr = name.rpartition(".")
        set_tensor(self.get_submodule(prefix), attr, value)

    def del_tensor(self, name: str) -> None:
        """
        Delete the attribute specified by the given path.

        For example, to delete the attribute mod.layer1.conv1.weight,
        use accessor.del_tensor("layer1.conv1.weight")
        """
        prefix, _, attr = name.rpartition(".")
        submodule = self.get_submodule(prefix)
        try:
            delattr(submodule, attr)
        except AttributeError as ex:
            raise AttributeError(
                f"{submodule._get_name()} has no attribute `{name}`"
            ) from ex

    def swap_tensor(
        self, name: str, value: torch.Tensor, allow_missing: bool = False
    ) -> torch.Tensor:
        """
        Swap the attribute specified by the given path to value.

        For example, to swap the attribute mod.layer1.conv1.weight,
        use accessor.swap_tensor("layer1.conv1.weight", value)
        """
        prefix, _, attr = name.rpartition(".")
        return swap_tensor(
            self.get_submodule(prefix), attr, value, allow_missing=allow_missing
        )

    # Batched operations

    def get_tensors(self, names: Iterable[str]) -> list[torch.Tensor]:
        """
        Get the tensors specified by the given paths.

        For example, to get the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.get_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"])
        """
        return [self.get_tensor(name) for name in names]

    def set_tensors(self, names: Iterable[str], values: Iterable[torch.Tensor]) -> None:
        """
        Set the attributes specified by the given paths to values.

        For example, to set the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.set_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"], [weight, bias])
        """
        if not isinstance(names, (list, tuple)):
            names = list(names)
        if not isinstance(values, (list, tuple)):
            values = list(values)
        if len(names) != len(values):
            raise AssertionError(
                f"names and values must have the same length, "
                f"got {len(names)} names and {len(values)} values"
            )

        for name, value in zip(names, values, strict=True):
            self.set_tensor(name, value)

    def set_tensors_dict(self, named_tensors: dict[str, torch.Tensor]) -> None:
        """
        Set the attributes specified by the given paths to values.

        For example, to set the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.set_tensors_dict({
            "layer1.conv1.weight": weight,
            "layer1.conv1.bias": bias,
        })
        """
        for name, value in named_tensors.items():
            self.set_tensor(name, value)

    def del_tensors(self, names: Iterable[str]) -> None:
        """
        Delete the attributes specified by the given paths.

        For example, to delete the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.del_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"])
        """
        for name in names:
            self.del_tensor(name)

    def swap_tensors(
        self,
        names: Iterable[str],
        values: Iterable[torch.Tensor],
        allow_missing: bool = False,
    ) -> list[torch.Tensor]:
        """
        Swap the attributes specified by the given paths to values.

        For example, to swap the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.swap_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"], [weight, bias])
        """
        if not isinstance(names, (list, tuple)):
            names = list(names)
        if not isinstance(values, (list, tuple)):
            values = list(values)
        if len(names) != len(values):
            raise AssertionError(
                f"names and values must have the same length, "
                f"got {len(names)} names and {len(values)} values"
            )

        return [
            self.swap_tensor(name, value, allow_missing=allow_missing)
            for name, value in zip(names, values, strict=True)
        ]

    def swap_tensors_dict(
        self, named_tensors: dict[str, torch.Tensor], allow_missing: bool = False
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """
        Swap the attributes specified by the given paths to values.

        For example, to swap the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.swap_tensors_dict({
            "layer1.conv1.weight": weight,
            "layer1.conv1.bias": bias,
        })
        """
        orig_named_tensors = {}
        missing_keys = []
        try:
            for name, tensor in named_tensors.items():
                orig_tensor = self.swap_tensor(name, tensor, allow_missing=True)
                if orig_tensor is _MISSING:
                    missing_keys.append(name)
                orig_named_tensors[name] = orig_tensor
        except Exception:
            # Swap back if any exception occurs
            for name, orig_tensor in orig_named_tensors.items():
                self.swap_tensor(name, orig_tensor, allow_missing=True)
            raise
        if missing_keys and not allow_missing:
            # Swap back if any key is missing when allow_missing is False
            for name, orig_tensor in orig_named_tensors.items():
                self.swap_tensor(name, orig_tensor, allow_missing=True)
            raise RuntimeError(f"Missing key(s): {', '.join(map(repr, missing_keys))}.")
        return orig_named_tensors, missing_keys

    def check_keys(self, keys: Iterable[str]) -> tuple[list[str], list[str]]:
        """Check that the given keys are valid."""
        keys = set(keys)
        valid_keys = {name for name, _ in self.named_tensors(remove_duplicate=False)}
        missing_keys = valid_keys - keys
        unexpected_keys = keys - valid_keys
        return sorted(missing_keys), sorted(unexpected_keys)

    # Shortcut methods

    def named_parameters(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Iterate over all the parameters in the module."""
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)

    def named_buffers(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Iterate over all the buffers in the module."""
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_tensors(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Iterate over all the tensors in the module."""
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_modules(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, "torch.nn.Module"]]:
        """Iterate over all the modules in the module."""
        yield from self.module.named_modules(remove_duplicate=remove_duplicate)
<|endoftext|>from __future__ import annotations

import copy
from typing import TypeVar

import torch


__all__ = [
    "fuse_conv_bn_eval",
    "fuse_conv_bn_weights",
    "fuse_linear_bn_eval",
    "fuse_linear_bn_weights",
]

ConvT = TypeVar("ConvT", bound="torch.nn.modules.conv._ConvNd")
LinearT = TypeVar("LinearT", bound="torch.nn.Linear")


def fuse_conv_bn_eval(
    conv: ConvT,
    bn: torch.nn.modules.batchnorm._BatchNorm,
    transpose: bool = False,
) -> ConvT:
    r"""Fuse a convolutional module and a BatchNorm module into a single, new convolutional module.

    Args:
        conv (torch.nn.modules.conv._ConvNd): A convolutional module.
        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.
        transpose (bool, optional): If True, transpose the convolutional weight. Defaults to False.

    Returns:
        torch.nn.modules.conv._ConvNd: The fused convolutional module.

    .. note::
        Both ``conv`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
    if conv.training or bn.training:
        raise AssertionError("Fusion only for eval!")
    fused_conv = copy.deepcopy(conv)

    if bn.running_mean is None or bn.running_var is None:
        raise AssertionError("bn.running_mean and bn.running_var must not be None")
    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
        fused_conv.weight,
        fused_conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
        transpose,
    )

    return fused_conv


def fuse_conv_bn_weights(
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None,
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_eps: float,
    bn_w: torch.Tensor | None,
    bn_b: torch.Tensor | None,
    transpose: bool = False,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    r"""Fuse convolutional module parameters and BatchNorm module parameters into new convolutional module parameters.

    Args:
        conv_w (torch.Tensor): Convolutional weight.
        conv_b (Optional[torch.Tensor]): Convolutional bias.
        bn_rm (torch.Tensor): BatchNorm running mean.
        bn_rv (torch.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (Optional[torch.Tensor]): BatchNorm weight.
        bn_b (Optional[torch.Tensor]): BatchNorm bias.
        transpose (bool, optional): If True, transpose the conv weight. Defaults to False.

    Returns:
        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused convolutional weight and bias.
    """
    conv_weight_dtype = conv_w.dtype
    conv_bias_dtype = conv_b.dtype if conv_b is not None else conv_weight_dtype
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

    fused_conv_w = (conv_w * (bn_w * bn_var_rsqrt).reshape(shape)).to(
        dtype=conv_weight_dtype
    )
    fused_conv_b = ((conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b).to(
        dtype=conv_bias_dtype
    )

    return (
        torch.nn.Parameter(fused_conv_w, conv_w.requires_grad),
        torch.nn.Parameter(fused_conv_b, conv_b.requires_grad),
    )


def fuse_linear_bn_eval(
    linear: LinearT,
    bn: torch.nn.modules.batchnorm._BatchNorm,
) -> LinearT:
    r"""Fuse a linear module and a BatchNorm module into a single, new linear module.

    Args:
        linear (torch.nn.Linear): A Linear module.
        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.

    Returns:
        torch.nn.Linear: The fused linear module.

    .. note::
        Both ``linear`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
    if linear.training or bn.training:
        raise AssertionError("Fusion only for eval!")
    fused_linear = copy.deepcopy(linear)

    """
    Linear-BN needs to be fused while preserving the shapes of linear weight/bias.
    To preserve the shapes of linear weight/bias, the channel dim of bn needs to be broadcastable with the last dim of linear,
    because bn operates over the channel dim, (N, C_in, H, W) while linear operates over the last dim, (*, H_in).
    To be broadcastable, the number of features in bn and
    the number of output features from linear must satisfy the following condition:
    1. they are equal, or
    2. the number of features in bn is 1
    Otherwise, skip the folding path
    """
    if linear.out_features != bn.num_features and bn.num_features != 1:
        raise AssertionError(
            f"To fuse, linear.out_features == bn.num_features or bn.num_features == 1, "
            f"got linear.out_features={linear.out_features} and bn.num_features={bn.num_features}"
        )

    if bn.running_mean is None or bn.running_var is None:
        raise AssertionError("bn.running_mean and bn.running_var must not be None")
    fused_linear.weight, fused_linear.bias = fuse_linear_bn_weights(
        fused_linear.weight,
        fused_linear.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )

    return fused_linear


def fuse_linear_bn_weights(
    linear_w: torch.Tensor,
    linear_b: torch.Tensor | None,
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_eps: float,
    bn_w: torch.Tensor,
    bn_b: torch.Tensor,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    r"""Fuse linear module parameters and BatchNorm module parameters into new linear module parameters.

    Args:
        linear_w (torch.Tensor): Linear weight.
        linear_b (Optional[torch.Tensor]): Linear bias.
        bn_rm (torch.Tensor): BatchNorm running mean.
        bn_rv (torch.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (torch.Tensor): BatchNorm weight.
        bn_b (torch.Tensor): BatchNorm bias.

    Returns:
        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused linear weight and bias.
    """
    linear_weight_dtype = linear_w.dtype
    linear_bias_dtype = linear_b.dtype if linear_b is not None else linear_weight_dtype
    if linear_b is None:
        linear_b = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)

    fused_w = linear_w * bn_scale.unsqueeze(-1).to(dtype=linear_weight_dtype)
    fused_b = ((linear_b - bn_rm) * bn_scale + bn_b).to(dtype=linear_bias_dtype)

    return torch.nn.Parameter(fused_w, linear_w.requires_grad), torch.nn.Parameter(
        fused_b, linear_b.requires_grad
    )
<|endoftext|># mypy: allow-untyped-defs

import torch

from .expanded_weights_impl import ExpandedWeight


def is_batch_first(expanded_args_and_kwargs):
    batch_first = None
    # pyrefly: ignore [bad-assignment]
    for arg in expanded_args_and_kwargs:
        if not isinstance(arg, ExpandedWeight):
            continue

        if not batch_first:
            batch_first = arg.batch_first
        elif arg.batch_first != batch_first:
            raise RuntimeError(
                "Got conflicting batch_first arguments in the same layer"
            )
    return batch_first


def standard_kwargs(kwarg_names, expanded_args):
    r"""Separate args and kwargs from `__torch_function__`s that standardize kwargs.

    Most `__torch_function__`s standardize the kwargs that they give, so this will separate
    the args and kwargs they pass. Functions that don't are linear and convAND.
    """
    kwarg_values = expanded_args[len(expanded_args) - len(kwarg_names) :]
    expanded_args_without_kwargs = expanded_args[
        : len(expanded_args) - len(kwarg_names)
    ]
    expanded_kwargs = dict(zip(kwarg_names, kwarg_values, strict=True))
    return expanded_args_without_kwargs, expanded_kwargs


def forward_helper(func, expanded_args, expanded_kwargs):
    r"""Compute the forward pass for a function that has expanded weight(s) passed to it.

    It will run the forward pass where all ExpandedWeights are their original
    weight. It runs checks on the given arguments and detaches the outputs.

    .. note:: First argument in :attr:`expanded_args` must be the input with the batch
    dimension as the first element of the shape

    .. note:: :attr:`func` must return a Tensor or tuple of Tensors

    Args:
        func: The function to be called
        expanded_args: Arguments to be passed to :attr:`func`. Will include arguments
          that need to be unpacked because they are ExpandedWeights
        expanded_kwargs: Keyword arguments to be passed to :attr:`func`.
          Similar to :attr:`expanded_args`.
    """
    unexpanded_args, unexpanded_kwargs = _check_and_unexpand_args(
        func, expanded_args, expanded_kwargs
    )
    return func(*unexpanded_args, **unexpanded_kwargs)


def _check_and_unexpand_args(func, expanded_args, expanded_kwargs):
    # input must be the first argument passed
    input = expanded_args[0]
    if isinstance(input, ExpandedWeight):
        raise RuntimeError(
            "Expanded Weights do not support inputs that are also ExpandedWeights. "
            f"Input must be a Tensor, got {type(input).__name__} in function {func.__name__}"
        )
    if not isinstance(input, torch.Tensor):
        raise RuntimeError(
            "Expanded Weights requires a Tensor as the first input to get the batch dimension, "
            f"got {type(input).__name__} in function {func.__name__}"
        )
    if len(input.shape) == 0:
        raise RuntimeError(
            f"Expanded Weights requires a batch dimension but got an input of size 0 in function {func.__name__}"
        )
    if input.shape[0] == 0:
        raise RuntimeError(
            "0 is not a valid batch size for Expanded Weights but got input tensor of "
            f"{input} in function {func.__name__}"
        )
    for arg in expanded_args + tuple(expanded_kwargs.values()):
        if not isinstance(arg, ExpandedWeight):
            continue
        batch_size = input.shape[0] if arg.batch_first else input.shape[1]
        if (arg.allow_smaller_batches and batch_size > arg.batch_size) or (
            not arg.allow_smaller_batches and arg.batch_size != batch_size
        ):
            raise RuntimeError(
                "Expected ExpandedWeights to have batch size matching input but got "
                f"input batch size of {batch_size} with ExpandedWeight of batch size {arg.batch_size}"
            )

    loss_reduction: str | None = None
    for arg in expanded_args + tuple(expanded_kwargs.values()):
        if isinstance(arg, ExpandedWeight):
            if loss_reduction is None:
                loss_reduction = arg.loss_reduction
            elif loss_reduction != arg.loss_reduction:
                raise RuntimeError(
                    "Expected ExpandedWeights to all have the same loss_reduction argument but got one"
                    f"with {loss_reduction} and one with {arg.loss_reduction}"
                )

    unexpanded_args = tuple(
        arg.orig_weight if isinstance(arg, ExpandedWeight) else arg
        for arg in expanded_args
    )
    unexpanded_kwargs = {
        name: arg.orig_weight if isinstance(arg, ExpandedWeight) else arg
        for (name, arg) in expanded_kwargs.items()
    }
    return unexpanded_args, unexpanded_kwargs


def maybe_scale_by_batch_size(grad_sample, expanded_weight):
    if expanded_weight.loss_reduction == "mean":
        return grad_sample * expanded_weight.batch_size
    else:
        return grad_sample


def set_grad_sample_if_exists(maybe_expanded_weight, per_sample_grad_fn) -> None:
    unpacked = unpack_expanded_weight_or_tensor(maybe_expanded_weight)
    if isinstance(maybe_expanded_weight, ExpandedWeight):
        grad_sample_contribution = maybe_scale_by_batch_size(
            per_sample_grad_fn(unpacked), maybe_expanded_weight
        )

        if maybe_expanded_weight.batch_size > grad_sample_contribution.shape[0]:
            # this only passes the other checks if the arg allows smaller batch sizes
            intermediate = torch.zeros(
                maybe_expanded_weight.batch_size,
                *grad_sample_contribution.shape[1:],
                dtype=grad_sample_contribution.dtype,
                device=grad_sample_contribution.device,
            )
            intermediate[: grad_sample_contribution.shape[0]] = grad_sample_contribution
            grad_sample_contribution = intermediate

        if hasattr(unpacked, "grad_sample") and unpacked.grad_sample is not None:
            unpacked.grad_sample = unpacked.grad_sample + grad_sample_contribution
        else:
            unpacked.grad_sample = grad_sample_contribution


def unpack_expanded_weight_or_tensor(maybe_expanded_weight, func=lambda x: x):
    if isinstance(maybe_expanded_weight, ExpandedWeight):
        orig_weight = maybe_expanded_weight.orig_weight
        return func(orig_weight)
    elif (
        isinstance(maybe_expanded_weight, torch.Tensor)
        and not maybe_expanded_weight.requires_grad
    ):
        return func(maybe_expanded_weight)
    elif isinstance(maybe_expanded_weight, torch.Tensor):
        raise RuntimeError(
            "ExpandedWeights currently does not support a mixture of ExpandedWeight parameters "
            "and normal Parameters. Please file and issue with pytorch/pytorch"
        )


def sum_over_all_but_batch_and_last_n(
    tensor: torch.Tensor,
    n_dims: int,
) -> torch.Tensor:
    r"""
    Calculate the sum over all dimensions, except the first (batch dimension), and excluding the last n_dims.

    This function will ignore the first dimension and it will
    not aggregate over the last n_dims dimensions.
    Args:
        tensor: An input tensor of shape ``(B, ..., X[n_dims-1])``.
        n_dims: Number of dimensions to keep.
    Example:
        >>> tensor = torch.ones(1, 2, 3, 4, 5)
        >>> sum_over_all_but_batch_and_last_n(tensor, n_dims=2).shape
        torch.Size([1, 4, 5])
    Returns:
        A tensor of shape ``(B, ..., X[n_dims-1])``
    """
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)
<|endoftext|>from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.nn.functional as F


_P = ParamSpec("_P")
_R = TypeVar("_R")

from .conv_utils import (
    conv_args_and_kwargs,
    conv_backward,
    conv_input_for_string_padding,
    conv_picker,
)
from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import forward_helper


@implements_per_sample_grads(F.conv1d)
@implements_per_sample_grads(F.conv2d)
@implements_per_sample_grads(F.conv3d)
class ConvPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: Any,
        kwarg_names: list[str],
        conv_fn: Callable[_P, _R],
        *expanded_args_and_kwargs: Any,
    ) -> torch.Tensor:
        expanded_args, expanded_kwargs = conv_args_and_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        orig_input = expanded_args[0]
        was_same_padding = expanded_kwargs["padding"] == "same"

        if isinstance(expanded_kwargs["padding"], str):
            # if padding is a string, we'll do the necessary padding (slowly) using F.pad
            kernel_size = expanded_args[1].shape[2:]
            padding, dilation = expanded_kwargs["padding"], expanded_kwargs["dilation"]
            input = conv_input_for_string_padding(
                conv_fn, padding, expanded_args[0], dilation, kernel_size
            )
            expanded_args = (input, expanded_args[1])
            # since we've already done the padding, don't need any more
            expanded_kwargs["padding"] = 0

        output = forward_helper(conv_fn, expanded_args, expanded_kwargs)
        input, weight = expanded_args
        batched_dim_size = conv_picker(conv_fn, 3, 4, 5)
        if input.dim() != batched_dim_size:
            raise RuntimeError(
                f"Expanded Weights only support convolution with batched input, got {conv_fn} with an"
                f"unbatched input of dim {input.dim()}, expected input of dim {batched_dim_size}"
            )

        # pyrefly: ignore [invalid-type-var]
        ctx.conv_fn = conv_fn

        ctx.batch_size = orig_input.shape[0]
        ctx.input_required_grad = orig_input.requires_grad
        ctx.orig_input_shape = orig_input.shape
        ctx.was_same_padding = was_same_padding
        ctx.stride, ctx.padding = expanded_kwargs["stride"], expanded_kwargs["padding"]
        ctx.dilation, ctx.groups = (
            expanded_kwargs["dilation"],
            expanded_kwargs["groups"],
        )

        if isinstance(weight, ExpandedWeight):
            ctx.input = input
        ctx.weight = weight
        ctx.bias = expanded_kwargs["bias"]

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return conv_backward(ctx.conv_fn, ctx, grad_outputs[0])
<|endoftext|>from .conv_expanded_weights import ConvPerSampleGrad
from .embedding_expanded_weights import EmbeddingPerSampleGrad
from .expanded_weights_impl import ExpandedWeight
from .group_norm_expanded_weights import GroupNormPerSampleGrad
from .instance_norm_expanded_weights import InstanceNormPerSampleGrad
from .layer_norm_expanded_weights import LayerNormPerSampleGrad
from .linear_expanded_weights import LinearPerSampleGrad


__all__ = ["ExpandedWeight"]
<|endoftext|># mypy: allow-untyped-defs
import functools
from collections.abc import Callable
from contextlib import contextmanager

import torch
from torch._decomp import decomposition_table
from torch.utils._pytree import tree_map_only


HANDLED_FUNCTIONS: dict[Callable, torch.autograd.Function] = {}

aten = torch._ops.ops.aten
# __torch_function__ runs before the pydispatcher so we need to manually use the same
# decompositions indexed by their torch equivalent
expanded_weights_rnn_decomps = {
    # func: (input_decomp, data_decomp)
    torch.rnn_relu: (
        decomposition_table[aten.rnn_relu.input],
        decomposition_table[aten.rnn_relu.data],
    ),
    torch.rnn_tanh: (
        decomposition_table[aten.rnn_tanh.input],
        decomposition_table[aten.rnn_tanh.data],
    ),
    torch.lstm: (
        decomposition_table[aten.lstm.input],
        decomposition_table[aten.lstm.data],
    ),
    torch.gru: (
        decomposition_table[aten.gru.input],
        decomposition_table[aten.gru.data],
    ),
}


# all of the RNN decomps run linear with the batch dimension second, even if batch_first was set
@contextmanager
def batch_second(args, kwargs):
    def set_batch_second(ew) -> None:
        ew.set_batch_first(False)

    def reset_batch_first(ew) -> None:
        ew.set_batch_first(True)

    tree_map_only(ExpandedWeight, set_batch_second, args)
    tree_map_only(ExpandedWeight, set_batch_second, kwargs)
    try:
        yield
    finally:
        tree_map_only(ExpandedWeight, reset_batch_first, args)
        tree_map_only(ExpandedWeight, reset_batch_first, kwargs)


# to support packed sequences, we need to allow for smaller batches. Expanded weights represents the largest batch
@contextmanager
def allow_smaller_batches(args, kwargs):
    def allow(ew) -> None:
        ew.set_allow_smaller_batches(True)

    def reset(ew) -> None:
        ew.set_allow_smaller_batches(False)

    tree_map_only(ExpandedWeight, allow, args)
    tree_map_only(ExpandedWeight, allow, kwargs)
    try:
        yield
    finally:
        tree_map_only(ExpandedWeight, reset, args)
        tree_map_only(ExpandedWeight, reset, kwargs)


@contextmanager
def setup_rnn(use_input_variant, args, kwargs):
    with (
        batch_second(args, kwargs)
        if use_input_variant
        else allow_smaller_batches(args, kwargs)
    ):
        yield


def implements_per_sample_grads(torch_function):
    @functools.wraps(torch_function)
    def decorator(autograd_func):
        HANDLED_FUNCTIONS[torch_function] = autograd_func
        return autograd_func

    return decorator


# ExpandedWeight represents a weight (parameter) Tensor that has an expanded
# batch dimension. Operations on the ExpandedWeight Tensor act exactly like
# those without an expanded batch dimension but a call to .backward() populates
# the original (unexpanded) tensor with per-sample-gradients in the grad_sample field
#
# ExpandedWeight has a fallback that always fails since we cannot know what the batch
# dimension of the input tensor is and therefore cannot know if this is a valid call
#
# This is a __torch_function__ object but it could have also been a Tensor Extension
# with a dispatch key.
#
# Needs to be a tensor subclass to allow reparameterization
class ExpandedWeight(torch.Tensor):
    def __init__(self, orig_weight, batch_size, loss_reduction) -> None:
        self.batch_size = batch_size
        self.batch_first = True
        self.allow_smaller_batches = False
        self.orig_weight = orig_weight
        self.loss_reduction = loss_reduction

    handled_functions = HANDLED_FUNCTIONS

    def __new__(cls, orig_weight, batch_size, loss_reduction):
        if not isinstance(orig_weight, torch.Tensor):
            raise RuntimeError(
                f"Can only make Expanded Weights of Tensors, got {type(orig_weight).__name__}"
            )
        if not orig_weight.requires_grad:
            raise RuntimeError(
                "Can only build ExpandedWeights objects of tensors that requires_grad"
            )
        ret = torch.Tensor._make_subclass(cls, orig_weight, True)
        return ret

    @classmethod
    def __torch_function__(cls, func, _, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in expanded_weights_rnn_decomps:
            # in aten, choosing the input or data variants is done by parsing logic. This mimics some of that
            decomp_opts = expanded_weights_rnn_decomps[func]
            use_input_variant = isinstance(
                # pyrefly: ignore [bad-index]
                args[2],
                list,
            )  # data variant uses a list here
            decomp = decomp_opts[0] if use_input_variant else decomp_opts[1]

            if decomp is not None:
                with setup_rnn(use_input_variant, args, kwargs):
                    return decomp(*args, **kwargs)
        if func is torch._cudnn_rnn_flatten_weight:
            # since we aren't using the fused cuda kernels for RNNs, don't do this
            return
        if func in cls.handled_functions:
            return cls.handled_functions[func].apply(
                tuple(kwargs.keys()), func, *(args + tuple(kwargs.values()))
            )
        # We cannot use a fallback here because we do not know the batch dimension for any regular tensor inputs,
        # i.e. torch.add(torch.Tensor, ExpandedWeight)
        raise RuntimeError(
            f"Expanded Weights encountered but cannot handle function {func.__name__}"
        )

    @property
    def dtype(self):  # type: ignore[override]
        return self.orig_weight.dtype

    @property
    def data(self):  # type: ignore[override]
        return self.orig_weight.data

    @property
    def shape(self):  # type: ignore[override]
        return self.orig_weight.shape

    @property
    def device(self):  # type: ignore[override]
        return self.orig_weight.device

    @property
    def is_cuda(self):  # type: ignore[override]
        return self.orig_weight.is_cuda

    def data_ptr(self):
        return self.orig_weight.data_ptr()

    def get_device(self):
        return self.orig_weight.get_device()

    def set_allow_smaller_batches(self, is_allow_smaller_batches) -> None:
        self.allow_smaller_batches = is_allow_smaller_batches

    def set_batch_first(self, is_batch_first=True) -> None:
        self.batch_first = is_batch_first
<|endoftext|>from typing import Any

import torch
import torch.nn.functional as F

from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
)


@implements_per_sample_grads(F.embedding)
class EmbeddingPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: Any, kwarg_names: list[str], _: Any, *expanded_args_and_kwargs: Any
    ) -> torch.Tensor:
        expanded_args, expanded_kwargs = standard_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        if len(expanded_args[0].shape) == 1:
            raise RuntimeError(
                f"Expanded Weights needs an input with a batch size, got a 1D tensor, {expanded_args[0]}"
            )
        output = forward_helper(F.embedding, expanded_args, expanded_kwargs)
        ctx.input, ctx.weight = expanded_args
        ctx.padding_idx, ctx.scale_grad_by_freq = (
            expanded_kwargs["padding_idx"],
            expanded_kwargs["scale_grad_by_freq"],
        )
        ctx.sparse = expanded_kwargs["sparse"]
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        input, weight = ctx.input, ctx.weight
        padding_idx, scale_grad_by_freq, sparse = (
            ctx.padding_idx,
            ctx.scale_grad_by_freq,
            ctx.sparse,
        )

        def weight_per_sample_grad(weight: torch.Tensor) -> torch.Tensor:
            batch_size = input.shape[0]
            embedding_dim = weight.shape[1]
            index = (
                input.unsqueeze(-1)
                .expand(*input.shape, embedding_dim)
                .reshape(batch_size, -1, embedding_dim)
            )
            grad_sample = torch.zeros(  # type: ignore[attr-defined]
                batch_size, *weight.shape, device=weight.device, dtype=grad_output.dtype
            )
            return grad_sample.scatter_add_(
                1, index, grad_output.reshape(batch_size, -1, embedding_dim)
            )

        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference

        if input.requires_grad:
            bw_fn = torch.ops.aten.embedding_backward
            results.append(
                bw_fn(
                    grad_output,
                    input,
                    weight.shape[0],
                    padding_idx,
                    scale_grad_by_freq,
                    sparse,
                )
            )
        else:
            results.append(None)

        # weight doesn't compute batched gradients; no other arguments are differentiable (2 not saved from forward)
        results = results + [None] * 6

        # set grad_sample field for weight with per sample gradients
        set_grad_sample_if_exists(weight, weight_per_sample_grad)
        return tuple(results)
<|endoftext|># mypy: allow-untyped-defs
from functools import partial

import torch
import torch.nn.functional as F

from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
    unpack_expanded_weight_or_tensor,
)


@implements_per_sample_grads(F.instance_norm)
class InstanceNormPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
        instance_norm = partial(torch.instance_norm, cudnn_enabled=True)
        expanded_args, expanded_kwargs = standard_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        output = forward_helper(instance_norm, expanded_args, expanded_kwargs)
        ctx.input = expanded_args[0]
        ctx.running_mean, ctx.running_var = (
            expanded_kwargs["running_mean"],
            expanded_kwargs["running_var"],
        )
        ctx.weight, ctx.bias, ctx.eps = (
            expanded_kwargs["weight"],
            expanded_kwargs["bias"],
            expanded_kwargs["eps"],
        )
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        input, running_mean, running_var = ctx.input, ctx.running_mean, ctx.running_var
        weight, bias, eps = ctx.weight, ctx.bias, ctx.eps

        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference
        if input.requires_grad:
            b = input.shape[0]
            c = input.shape[1]
            new_shape = (1, b * c, *input.shape[2:])

            weight_ = unpack_expanded_weight_or_tensor(
                weight, lambda orig_weight: orig_weight.repeat(b)
            )
            running_mean_ = running_mean.repeat(b) if running_mean is not None else None
            running_var_ = running_var.repeat(b) if running_var is not None else None
            input_reshaped = input.reshape(new_shape)
            grad_output_reshaped = grad_output.reshape(new_shape)
            var, mean = torch.var_mean(
                input_reshaped,
                (0,) + tuple(range(2, input.dim())),
                correction=0,
                keepdim=False,
            )
            rstd = torch.rsqrt(var + eps)

            # must use native batch norm since it supports all inputs. This may have used cuda or openmp during the forward but
            # it didn't save the metadata, so we don't know during the backward
            res = torch.ops.aten.native_batch_norm_backward(
                grad_output_reshaped,
                input_reshaped,
                weight_,
                running_mean_,
                running_var_,
                mean,
                rstd,
                True,
                eps,
                (True, False, False),
            )
            results.append(res[0].reshape(input.shape))
        else:
            results.append(None)

        # weight and bias don't compute batched gradients; no other arguments are differentiable (2 are not saved from the forward)
        results = results + [None] * 7

        # set grad_sample field for weight and bias with per sample gradients
        set_grad_sample_if_exists(
            weight,
            lambda _: torch.einsum(
                "ni...->ni", F.instance_norm(input, eps=eps) * grad_output
            ),
        )
        set_grad_sample_if_exists(
            bias, lambda _: torch.einsum("ni...->ni", grad_output)
        )
        return tuple(results)
<|endoftext|># mypy: allow-untyped-defs
import operator
from functools import reduce

import torch
import torch.nn.functional as F

from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
    unpack_expanded_weight_or_tensor,
)


@implements_per_sample_grads(F.group_norm)
class GroupNormPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
        expanded_args, expanded_kwargs = standard_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        input, num_groups = expanded_args
        N = input.shape[0]
        C = input.shape[1]
        HxW = reduce(operator.mul, input.shape[2:], 1)
        weight, bias, eps = (
            expanded_kwargs["weight"],
            expanded_kwargs["bias"],
            expanded_kwargs["eps"],
        )
        output, mean, rstd = forward_helper(
            torch.native_group_norm,
            (input, weight, bias, N, C, HxW, num_groups, eps),
            {},
        )
        ctx.input, ctx.num_groups = input, num_groups
        ctx.weight, ctx.eps = weight, eps
        ctx.mean, ctx.rstd = mean, rstd
        if isinstance(bias, ExpandedWeight):
            ctx.bias = bias
        if input.requires_grad and isinstance(weight, ExpandedWeight):
            ctx.weight = weight
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        input, num_groups = ctx.input, ctx.num_groups
        weight, bias, eps = ctx.weight, ctx.bias, ctx.eps
        mean, rstd = ctx.mean, ctx.rstd

        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference

        if input.requires_grad:
            weight_c = unpack_expanded_weight_or_tensor(
                weight, lambda t: t.contiguous()
            )
            input_c = input.contiguous()
            grad_output_c = (
                grad_output.contiguous() if grad_output is not None else None
            )
            N = input.shape[0]
            C = input.shape[1]
            HxW = 1
            for s in input.shape[2:]:
                HxW *= s
            bw_fn = torch.ops.aten.native_group_norm_backward
            results.append(
                bw_fn(
                    grad_output_c,
                    input_c,
                    mean,
                    rstd,
                    weight_c,
                    N,
                    C,
                    HxW,
                    num_groups,
                    (True, False, False),
                )[0]
            )
        else:
            results.append(None)

        # weight and bias don't compute batched gradients; no other arguments are differentiable
        results = results + [None] * 4

        # set grad_sample field for weight and bias with per sample gradients
        if hasattr(ctx, "weight"):
            set_grad_sample_if_exists(
                weight,
                lambda _: torch.einsum(
                    "ni...->ni",
                    F.group_norm(input, num_groups, eps=eps) * grad_output,
                ),
            )
        if hasattr(ctx, "bias"):
            set_grad_sample_if_exists(
                bias, lambda _: torch.einsum("ni...->ni", grad_output)
            )
        return tuple(results)
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.nn.functional as F

from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
    sum_over_all_but_batch_and_last_n,
    unpack_expanded_weight_or_tensor,
)


@implements_per_sample_grads(F.layer_norm)
class LayerNormPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
        expanded_args, expanded_kwargs = standard_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        input = expanded_args[0]
        normalized_shape = expanded_args[1]
        if len(input.shape) <= len(normalized_shape):
            raise RuntimeError(
                "Expanded Weights: Layer norm should not normalize over batch dimension for per sample gradient"
                f"computations but got that normalized shape, {normalized_shape}, matched input shape."
            )
        output, mean, rstd = forward_helper(
            torch.native_layer_norm, expanded_args, expanded_kwargs
        )
        ctx.args = expanded_args

        if input.requires_grad or isinstance(expanded_kwargs["weight"], ExpandedWeight):
            ctx.weight = expanded_kwargs["weight"]
        if input.requires_grad or isinstance(expanded_kwargs["bias"], ExpandedWeight):
            ctx.bias = expanded_kwargs["bias"]
        ctx.eps = expanded_kwargs["eps"]
        ctx.mean, ctx.rstd = mean, rstd
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        def weight_per_sample_grad(weight):
            return sum_over_all_but_batch_and_last_n(
                F.layer_norm(input, normalized_shape, eps=ctx.eps) * grad_output,
                weight.dim(),
            )

        input, normalized_shape = ctx.args
        mean, rstd = ctx.mean, ctx.rstd

        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference
        if input.requires_grad:
            weight_ = unpack_expanded_weight_or_tensor(ctx.weight)
            bias_ = unpack_expanded_weight_or_tensor(ctx.bias)
            results.append(
                torch.ops.aten.native_layer_norm_backward(
                    grad_output,
                    input,
                    normalized_shape,
                    mean,
                    rstd,
                    weight_,
                    bias_,
                    (True, False, False),
                )[0]
            )
        else:
            results.append(None)

        # weight and bias don't compute batched gradients; no other arguments are differentiable
        results = results + [None] * 4

        # set grad_sample field for weight and bias with per sample gradients
        if hasattr(ctx, "weight"):
            set_grad_sample_if_exists(ctx.weight, weight_per_sample_grad)
        if hasattr(ctx, "bias"):
            set_grad_sample_if_exists(
                ctx.bias,
                lambda bias: sum_over_all_but_batch_and_last_n(grad_output, bias.dim()),
            )
        return tuple(results)
<|endoftext|># mypy: allow-untyped-defs

import math

import torch
import torch.nn.functional as F

from .expanded_weights_utils import (
    set_grad_sample_if_exists,
    unpack_expanded_weight_or_tensor,
)


THRESHOLD = 32


def conv_picker(func, conv1dOpt, conv2dOpt, conv3dOpt):
    if func is F.conv1d:
        return conv1dOpt
    if func is F.conv2d:
        return conv2dOpt
    else:
        if func is not F.conv3d:
            raise AssertionError(
                f"Expected func to be F.conv1d, F.conv2d, or F.conv3d, got {func}"
            )
        return conv3dOpt


def conv_args_and_kwargs(kwarg_names, expanded_args_and_kwargs):
    args = expanded_args_and_kwargs[: len(expanded_args_and_kwargs) - len(kwarg_names)]
    kwargs = expanded_args_and_kwargs[
        len(expanded_args_and_kwargs) - len(kwarg_names) :
    ]
    kwargs = dict(zip(kwarg_names, kwargs, strict=True))

    return conv_normalizer(*args, **kwargs)


def conv_normalizer(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    return (input, weight), {
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }


def conv_input_for_string_padding(func, padding_style, input, dilation, kernel_size):
    if padding_style == "valid":
        return input
    else:
        padding = int_padding_for_string_padding(
            func, padding_style, dilation, kernel_size
        )
        return F.pad(input, padding)


def int_padding_for_string_padding(func, padding_style, dilation, kernel_size):
    def get_dilation(i):
        return dilation[i] if isinstance(dilation, tuple) else dilation

    if padding_style == "same":
        padding: list[int] = []
        # F.pad needs the padding in reverse order from what conv expects
        for i in range(conv_picker(func, 0, 1, 2), -1, -1):
            padding += conv_padding_for_same(get_dilation(i), kernel_size[i])
        return padding
    elif padding_style == "valid":
        return conv_picker(func, 2, 4, 6) * (0,)
    else:
        raise RuntimeError(
            f"got padding type of {padding_style}, only accept 'same' or 'valid'"
        )


def conv_padding_for_same(dilation, kernel_size):
    total_pad = dilation * (kernel_size - 1)
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return left_pad, right_pad


def conv_backward(func, ctx, grad_output):
    def weight_grad_sample(weight):
        if batch_size < THRESHOLD and groups == 1:
            return conv_group_weight_grad_sample(
                ctx.input,
                grad_output,
                weight_shape,
                stride,
                padding,
                dilation,
                batch_size,
                func,
            )
        else:
            return conv_unfold_weight_grad_sample(
                ctx.input,
                grad_output,
                weight_shape,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                func,
            )

    def expand(param):
        if isinstance(param, int):
            return conv_picker(func, (param,), (param, param), (param, param, param))
        else:
            return param

    def calc_total_padding(func, was_same, padding, dilation, kernel_size):
        if was_same:
            all_padding = int_padding_for_string_padding(
                func, "same", dilation, kernel_size
            )
            # F.pad needs the padding in reverse order from what conv expects
            total_padding = tuple(
                all_padding[i] + all_padding[i - 1]
                for i in range(len(all_padding) - 1, -1, -2)
            )
            return total_padding
        else:
            return tuple(2 * pad for pad in padding)

    weight_shape = ctx.weight.shape
    stride, padding, dilation, groups = (
        expand(ctx.stride),
        expand(ctx.padding),
        expand(ctx.dilation),
        ctx.groups,
    )

    kernel_size = [weight_shape[i] for i in range(2, conv_picker(func, 3, 4, 5))]

    batch_size = ctx.batch_size
    results: list[torch.Tensor | None] = []
    results.append(None)  # for kwarg names
    results.append(None)  # for op reference

    # "same" padding may give uneven padding on either side so we need to separate the "padding" attr and total padding
    total_padding = calc_total_padding(
        func, ctx.was_same_padding, padding, dilation, kernel_size
    )

    if ctx.input_required_grad:
        output_padding = []
        input_dims = conv_picker(func, 1, 2, 3)
        for i in range(input_dims):
            input_dim = ctx.orig_input_shape[2 + i]
            output_padding.append(
                (
                    total_padding[i]
                    + input_dim
                    - (kernel_size[i] * dilation[i] - dilation[i] + 1)
                )
                % stride[i]
            )
        weight_ = unpack_expanded_weight_or_tensor(ctx.weight)
        transpose_func = conv_picker(
            func, F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d
        )
        out = transpose_func(
            grad_output,
            weight_,
            None,
            stride,
            padding,
            tuple(output_padding),
            groups,
            dilation,
        )

        if ctx.was_same_padding:
            for i in range(len(total_padding)):
                out = torch.narrow(
                    out, 2 + i, total_padding[i] // 2, ctx.orig_input_shape[2 + i]
                )

        results.append(out)
    else:
        results.append(None)
    # weight and bias don't compute batched gradients; no other arguments are differentiable
    results = results + [None] * 6

    # set grad_sample field for weight and bias with per sample gradients
    set_grad_sample_if_exists(ctx.weight, weight_grad_sample)
    set_grad_sample_if_exists(
        ctx.bias, lambda _: grad_output.reshape(*grad_output.shape[:2], -1).sum(dim=2)
    )
    return tuple(results)


def conv_unfold_weight_grad_sample(
    input,
    grad_output,
    weight_shape,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    func,
):
    n = input.shape[0]
    in_channels = input.shape[1]

    unfold_func = conv_picker(
        func,
        lambda: F.unfold(
            input.unsqueeze(-2),
            kernel_size=(1, kernel_size[0]),
            dilation=(1, dilation[0]),
            padding=(0, padding[0]),
            stride=(1, stride[0]),
        ),
        lambda: F.unfold(
            input, kernel_size, dilation=dilation, padding=padding, stride=stride
        ),
        lambda: unfold3d(input, kernel_size, padding, stride, dilation),
    )

    input = unfold_func()
    grad_output = grad_output.reshape(n, -1, input.shape[-1])

    # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
    weight_grad_sample = torch.einsum("noq,npq->nop", grad_output, input)
    # rearrange the above tensor and extract diagonals.

    weight_grad_sample = weight_grad_sample.view(
        n,
        groups,
        -1,
        groups,
        int(in_channels / groups),
        math.prod(kernel_size),
    )
    weight_grad_sample = torch.einsum(
        "ngrg...->ngr...", weight_grad_sample
    ).contiguous()
    shape = [n] + list(weight_shape)
    weight_grad_sample = weight_grad_sample.view(shape)
    return weight_grad_sample


def conv_group_weight_grad_sample(
    input,
    grad_output,
    weight_shape,
    stride,
    padding,
    dilation,
    batch_size,
    func,
):
    I = input.shape[1]
    O = grad_output.shape[1]

    input_ = input.transpose(0, 1)
    grad_output_ = grad_output.view(
        grad_output.shape[0] * grad_output.shape[1], 1, *grad_output.shape[2:]
    )

    weight_grad_sample = func(
        input_,
        grad_output_,
        None,
        stride=dilation,
        padding=padding,
        dilation=stride,
        groups=batch_size,
    )
    input_dims = conv_picker(func, 3, 4, 5)
    for i in range(2, input_dims):
        weight_grad_sample = weight_grad_sample.narrow(i, 0, weight_shape[i])
    weight_grad_sample = weight_grad_sample.view(
        I, batch_size, O, *weight_grad_sample.shape[2:]
    )
    weight_grad_sample = weight_grad_sample.movedim(0, 2)
    return weight_grad_sample


def unfold3d(
    tensor,
    kernel_size,
    padding,
    stride,
    dilation,
):
    r"""
    Extract sliding local blocks from a batched input tensor.

    :class:`torch.nn.Unfold` only supports 4D inputs (batched image-like tensors).
    This method implements the same action for 5D inputs
    Args:
        tensor: An input tensor of shape ``(B, C, D, H, W)``.
        kernel_size: the size of the sliding blocks
        padding: implicit zero padding to be added on both sides of input
        stride: the stride of the sliding blocks in the input spatial dimensions
        dilation: the spacing between the kernel points.
    Returns:
        A tensor of shape ``(B, C * math.prod(kernel_size), L)``, where L - output spatial dimensions.
        See :class:`torch.nn.Unfold` for more details
    Example:
        >>> # xdoctest: +SKIP
        >>> B, C, D, H, W = 3, 4, 5, 6, 7
        >>> tensor = torch.arange(1, B * C * D * H * W + 1.0).view(B, C, D, H, W)
        >>> unfold3d(tensor, kernel_size=2, padding=0, stride=1).shape
        torch.Size([3, 32, 120])
    """

    if len(tensor.shape) != 5:
        raise ValueError(
            f"Input tensor must be of the shape [B, C, D, H, W]. Got{tensor.shape}"
        )

    if dilation != (1, 1, 1):
        raise NotImplementedError(f"dilation={dilation} not supported.")

    batch_size, channels, _, _, _ = tensor.shape

    # Input shape: (B, C, D, H, W)
    tensor = F.pad(
        tensor, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
    )
    # Output shape: (B, C, D+2*padding[2], H+2*padding[1], W+2*padding[0])

    tensor = tensor.unfold(dimension=2, size=kernel_size[0], step=stride[0])
    tensor = tensor.unfold(dimension=3, size=kernel_size[1], step=stride[1])
    tensor = tensor.unfold(dimension=4, size=kernel_size[2], step=stride[2])
    # Output shape: (B, C, D_out, H_out, W_out, kernel_size[0], kernel_size[1], kernel_size[2])
    # For D_out, H_out, W_out definitions see :class:`torch.nn.Unfold`

    tensor = tensor.permute(0, 2, 3, 4, 1, 5, 6, 7)
    # Output shape: (B, D_out, H_out, W_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    tensor = tensor.reshape(
        batch_size, -1, channels * math.prod(kernel_size)
    ).transpose(1, 2)
    # Output shape: (B, D_out * H_out * W_out, C * kernel_size[0] * kernel_size[1] * kernel_size[2]

    return tensor
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.nn.functional as F

from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    is_batch_first,
    set_grad_sample_if_exists,
    unpack_expanded_weight_or_tensor,
)


@implements_per_sample_grads(F.linear)
class LinearPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, _, __, *expanded_args_and_kwargs):
        if len(expanded_args_and_kwargs[0].shape) <= 1:
            raise RuntimeError(
                "Input does not have a batch dimension. Expanded Weights expected input "
                f"of at least rank 2, got of rank {len(expanded_args_and_kwargs[0].shape)}"
            )
        expanded_kwargs = {
            "bias": expanded_args_and_kwargs[2]
            if len(expanded_args_and_kwargs) == 3
            else None
        }
        expanded_args = expanded_args_and_kwargs[:2]
        ctx.batch_first = is_batch_first(expanded_args_and_kwargs)
        output = forward_helper(F.linear, expanded_args, expanded_kwargs)
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        input, weight = ctx.args
        bias = ctx.kwargs["bias"]
        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg_names
        results.append(None)  # for op reference

        if input.requires_grad:
            results.append(grad_output.matmul(unpack_expanded_weight_or_tensor(weight)))
        else:
            results.append(None)
        results.extend([None] * 2)  # weight and bias don't compute batched gradients

        if not ctx.batch_first:
            grad_output = grad_output.transpose(0, 1)
            input = input.transpose(0, 1)

        # weight and bias get their grad_sample fields set directly if they exist
        set_grad_sample_if_exists(
            weight, lambda _: torch.einsum("n...i,n...j->nij", grad_output, input)
        )
        set_grad_sample_if_exists(
            bias, lambda _: torch.einsum("n...k->nk", grad_output)
        )
        return tuple(results)
<|endoftext|># mypy: allow-untyped-defs
import operator
import warnings
from collections.abc import Sequence
from itertools import chain
from typing import Any, Generic, TypeVar

import torch
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
    _get_devices_properties,
)
from torch.nn.modules import Module
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter_kwargs


__all__ = ["DataParallel", "data_parallel"]


def _check_balance(device_ids: Sequence[int | torch.device]) -> None:
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = [_get_device_index(x, True) for x in device_ids]
    dev_props = _get_devices_properties(device_ids)

    def warn_imbalance(get_prop) -> bool:
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(
                imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]),
                stacklevel=2,
            )
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


T = TypeVar("T", bound=Module)


class DataParallel(Module, Generic[T]):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given :attr:`module` by
    splitting the input across the specified devices by chunking in the batch
    dimension (other objects will be copied once per device). In the forward
    pass, the module is replicated on each device, and each replica handles a
    portion of the input. During the backwards pass, gradients from each replica
    are summed into the original module.

    The batch size should be larger than the number of GPUs used.

    .. warning::
        It is recommended to use :class:`~torch.nn.parallel.DistributedDataParallel`,
        instead of this class, to do multi-GPU training, even if there is only a single
        node. See: :ref:`cuda-nn-ddp-instead` and :ref:`ddp`.

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel but some types are specially handled. tensors will be
    **scattered** on dim specified (default 0). tuple, list and dict types will
    be shallow copied. The other types will be shared among different threads
    and can be corrupted if written to in the model's forward pass.

    The parallelized :attr:`module` must have its parameters and buffers on
    ``device_ids[0]`` before running this :class:`~torch.nn.DataParallel`
    module.

    .. warning::
        In each forward, :attr:`module` is **replicated** on each device, so any
        updates to the running module in ``forward`` will be lost. For example,
        if :attr:`module` has a counter attribute that is incremented in each
        ``forward``, it will always stay at the initial value because the update
        is done on the replicas which are destroyed after ``forward``. However,
        :class:`~torch.nn.DataParallel` guarantees that the replica on
        ``device[0]`` will have its parameters and buffers sharing storage with
        the base parallelized :attr:`module`. So **in-place** updates to the
        parameters or buffers on ``device[0]`` will be recorded. E.g.,
        :class:`~torch.nn.BatchNorm2d` and :func:`~torch.nn.utils.spectral_norm`
        rely on this behavior to update the buffers.

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        will be invoked ``len(device_ids)`` times, each with inputs located on
        a particular device. Particularly, the hooks are only guaranteed to be
        executed in correct order with respect to operations on corresponding
        devices. For example, it is not guaranteed that hooks set via
        :meth:`~torch.nn.Module.register_forward_pre_hook` be executed before
        `all` ``len(device_ids)`` :meth:`~torch.nn.Module.forward` calls, but
        that each such hook be executed before the corresponding
        :meth:`~torch.nn.Module.forward` call of that device.

    .. warning::
        When :attr:`module` returns a scalar (i.e., 0-dimensional tensor) in
        :func:`forward`, this wrapper will return a vector of length equal to
        number of devices used in data parallelism, containing the result from
        each device.

    .. note::
        There is a subtlety in using the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`pack-rnn-unpack-with-data-parallelism` section in FAQ for
        details.


    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices (default: all devices)
        output_device (int or torch.device): device location of output (default: device_ids[0])

    Attributes:
        module (Module): the module to be parallelized

    Example::

        >>> # xdoctest: +SKIP
        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)  # input_var can be on any device, including CPU
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(
        self,
        module: T,
        device_ids: Sequence[int | torch.device] | None = None,
        output_device: int | torch.device | None = None,
        dim: int = 0,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once("torch.nn.parallel.DataParallel")
        device_type = _get_available_device_type()
        if device_type is None or device_type == "mps":
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = _get_all_device_indices()

        if device_ids is None:
            raise RuntimeError("no available devices were found")

        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = [_get_device_index(x, True) for x in device_ids]
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        if device_type == "cuda":
            _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        with torch.autograd.profiler.record_function("DataParallel.forward"):
            if not self.device_ids:
                return self.module(*inputs, **kwargs)

            # pyrefly: ignore [bad-argument-type]
            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError(
                        "module must have its parameters and buffers "
                        f"on device {self.src_device_obj} (device_ids[0]) but found one of "
                        f"them on device: {t.device}"
                    )

            inputs, module_kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # for forward function without any inputs, empty list and dict will be created
            # so the module can be executed on one device which is the first one in device_ids
            if not inputs and not module_kwargs:
                inputs = ((),)
                module_kwargs = ({},)

            if len(self.device_ids) == 1:
                return self.module(*inputs[0], **module_kwargs[0])
            replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, module_kwargs)
            return self.gather(outputs, self.output_device)

    def replicate(self, module: T, device_ids: Sequence[int | torch.device]) -> list[T]:
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(
        self,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
        device_ids: Sequence[int | torch.device],
    ) -> Any:
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(
        self, replicas: Sequence[T], inputs: Sequence[Any], kwargs: Any
    ) -> list[Any]:
        return parallel_apply(
            replicas, inputs, kwargs, self.device_ids[: len(replicas)]
        )

    def gather(self, outputs: Any, output_device: int | torch.device) -> Any:
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(
    module: Module,
    inputs: Any,
    device_ids: Sequence[int | torch.device] | None = None,
    output_device: int | torch.device | None = None,
    dim: int = 0,
    module_kwargs: Any | None = None,
) -> torch.Tensor:
    r"""Evaluate module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,) if inputs is not None else ()

    device_type = _get_available_device_type()

    if device_type is None:
        raise RuntimeError("device type could not be determined")

    if device_ids is None:
        device_ids = _get_all_device_indices()

    if device_ids is None:
        raise RuntimeError("no available devices were found")

    if output_device is None:
        output_device = device_ids[0]

    device_ids = [_get_device_index(x, True) for x in device_ids]
    output_device = _get_device_index(output_device, True)
    # pyrefly: ignore [bad-argument-type, no-matching-overload]
    src_device_obj = torch.device(device_type, device_ids[0])

    # pyrefly: ignore [bad-argument-type]
    for t in chain(module.parameters(), module.buffers()):
        if t.device != src_device_obj:
            raise RuntimeError(
                "module must have its parameters and buffers "
                f"on device {src_device_obj} (device_ids[0]) but found one of "
                f"them on device: {t.device}"
            )

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    # for module without any inputs, empty list and dict will be created
    # so the module can be executed on one device which is the first one in device_ids
    if not inputs and not module_kwargs:
        inputs = ((),)
        module_kwargs = ({},)

    if module_kwargs is None:
        raise AssertionError("module_kwargs should not be None after scatter_kwargs")

    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[: len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Sequence
from typing import Any, overload, TypeVar
from typing_extensions import deprecated

import torch
from torch.nn.parallel._functions import Gather, Scatter


__all__ = ["scatter", "scatter_kwargs", "gather"]


@deprecated(
    "`is_namedtuple` is deprecated, please use the python checks instead",
    category=FutureWarning,
)
def is_namedtuple(obj: Any) -> bool:
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return _is_namedtuple(obj)


def _is_namedtuple(obj: Any) -> bool:
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


T = TypeVar("T", dict, list, tuple)


# For some reason, 'scatter' returns a tuple when given a single Tensor input but a list otherwise.
@overload
def scatter(
    inputs: torch.Tensor,
    target_gpus: Sequence[int | torch.device],
    dim: int = ...,
) -> tuple[torch.Tensor, ...]: ...


@overload
def scatter(
    inputs: T,
    target_gpus: Sequence[int | torch.device],
    dim: int = ...,
) -> list[T]: ...


def scatter(inputs, target_gpus, dim=0):
    r"""Slice tensors into approximately equal chunks and distributes them across given GPUs.

    Duplicates references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if _is_namedtuple(obj):
            return [
                type(obj)(*args)
                # pyrefly: ignore [bad-argument-type, no-matching-overload]
                for args in zip(*map(scatter_map, obj), strict=False)
            ]
        if isinstance(obj, tuple) and len(obj) > 0:
            # pyrefly: ignore [bad-argument-type, no-matching-overload]
            return list(zip(*map(scatter_map, obj), strict=False))
        if isinstance(obj, list) and len(obj) > 0:
            # pyrefly: ignore [bad-argument-type, no-matching-overload]
            return [list(i) for i in zip(*map(scatter_map, obj), strict=False)]
        if isinstance(obj, dict) and len(obj) > 0:
            return [
                type(obj)(i)
                # pyrefly: ignore [bad-argument-type, no-matching-overload]
                for i in zip(*map(scatter_map, obj.items()), strict=False)
            ]
        return [obj for _ in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore[assignment]
    return res


def scatter_kwargs(
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    target_gpus: Sequence[int | torch.device],
    dim: int = 0,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    r"""Scatter with support for kwargs dictionary."""
    scattered_inputs = scatter(inputs, target_gpus, dim) if inputs else []
    scattered_kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(scattered_inputs) < len(scattered_kwargs):
        scattered_inputs.extend(
            () for _ in range(len(scattered_kwargs) - len(scattered_inputs))
        )
    elif len(scattered_kwargs) < len(inputs):
        scattered_kwargs.extend(
            {} for _ in range(len(scattered_inputs) - len(scattered_kwargs))
        )
    return tuple(scattered_inputs), tuple(scattered_kwargs)


def gather(outputs: Any, target_device: int | torch.device, dim: int = 0) -> Any:
    r"""Gather tensors from different GPUs on a specified device.

    This function is useful for gathering the results of a distributed computation.
    It takes a sequence of objects, one for each GPU, and returns a single object
    on the specified device.

    Args:
        outputs (Any): A sequence of objects (potentially tensors) to gather.
        target_device (Union[int, torch.device]): The device to gather the tensors to.
            Use 'cpu' for CPU to avoid a deprecation warning.
        dim (int, optional): The dimension along which to gather. Default: 0.

    Returns:
        Any: A gathered object (potentially tensor) on the specified device.
    """

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError("All dicts must have the same number of keys")
            # pyrefly: ignore [not-callable]
            return type(out)((k, gather_map([d[k] for d in outputs])) for k in out)
        if _is_namedtuple(out):
            # pyrefly: ignore [bad-argument-type]
            return type(out)._make(map(gather_map, zip(*outputs, strict=True)))
        # pyrefly: ignore [bad-argument-type]
        return type(out)(map(gather_map, zip(*outputs, strict=True)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None  # type: ignore[assignment]
    return res
<|endoftext|>import threading
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch._utils import ExceptionWrapper
from torch.cuda._utils import _get_device_index
from torch.nn.modules import Module


__all__ = ["get_a_var", "parallel_apply"]


def get_a_var(
    obj: torch.Tensor | list[Any] | tuple[Any, ...] | dict[Any, Any],
) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(
    modules: Sequence[Module],
    inputs: Sequence[Any],
    kwargs_tup: Sequence[dict[str, Any]] | None = None,
    devices: Sequence[int | torch.device | None] | None = None,
) -> list[Any]:
    r"""Apply each `module` in :attr:`modules` in parallel on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    if len(modules) != len(inputs):
        raise AssertionError(
            f"The number of modules {len(modules)} is not equal to "
            f"the number of inputs {len(inputs)}"
        )
    if kwargs_tup is not None:
        if len(modules) != len(kwargs_tup):
            raise AssertionError(
                f"The number of modules {len(modules)} is not equal to "
                f"the number of kwargs_tup {len(kwargs_tup)}"
            )
    else:
        kwargs_tup = (cast(dict[str, Any], {}),) * len(modules)
    if devices is not None:
        if len(modules) != len(devices):
            raise AssertionError(
                f"The number of modules {len(modules)} is not equal to "
                f"the number of devices {len(devices)}"
            )
    else:
        devices = [None] * len(modules)
    devices = [_get_device_index(x, True) for x in devices]
    streams = [torch.accelerator.current_stream(x) for x in devices]
    if not torch.accelerator.is_available():
        raise AssertionError("No available accelerator found.")
    device_type = torch.accelerator.current_accelerator().type  # type: ignore[union-attr]
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = (
        torch.is_grad_enabled(),
        torch.is_autocast_enabled(),
    )

    def _worker(
        i: int,
        module: Module,
        input: Any,
        kwargs: dict[str, Any],
        device: int | torch.device | None = None,
        stream: torch.Stream | None = None,
    ) -> None:
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            t = get_a_var(input)
            if t is None:
                with lock:
                    results[i] = ExceptionWrapper(
                        where=f"in replica {i}, no device was provided and no tensor input was found; "
                        "device cannot be resolved"
                    )
                return
            device = t.get_device()
        if isinstance(device, torch.device):
            device = device.index
        if stream is None:
            stream = torch.accelerator.current_stream(device)
        try:
            with (
                torch.accelerator.device_index(device),
                stream,
                torch.amp.autocast(device_type, enabled=autocast_enabled),
            ):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where=f"in replica {i} on device {device}"
                )

    if len(modules) > 1:
        threads = [
            threading.Thread(
                target=_worker, args=(i, module, input, kwargs, device, stream)
            )
            for i, (module, input, kwargs, device, stream) in enumerate(
                zip(modules, inputs, kwargs_tup, devices, streams, strict=True)
            )
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs
<|endoftext|>from typing_extensions import deprecated

from torch.nn.parallel.data_parallel import data_parallel, DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter


__all__ = [
    "replicate",
    "scatter",
    "parallel_apply",
    "gather",
    "data_parallel",
    "DataParallel",
    "DistributedDataParallel",
]


@deprecated(
    "`torch.nn.parallel.DistributedDataParallelCPU` is deprecated, "
    "please use `torch.nn.parallel.DistributedDataParallel` instead.",
    category=FutureWarning,
)
class DistributedDataParallelCPU(DistributedDataParallel):
    pass
<|endoftext|># mypy: allow-untyped-defs
import copy
import functools
import inspect
import itertools
import logging
import os
import sys
import warnings
import weakref
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import auto, Enum
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch._utils import _get_device_index
from torch.autograd import Function, Variable
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather, scatter_kwargs
from torch.utils._pytree import tree_flatten, tree_unflatten


RPC_AVAILABLE = False

# Default bucket size in MiB for gradient reduction
_DEFAULT_BUCKET_CAP_MB = 25
# Conversion factor from MiB to bytes
_MB_TO_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _BucketCapacityConfig:
    """Configuration for DDP gradient reduction bucket capacities.

    This immutable dataclass encapsulates bucket size settings for
    DistributedDataParallel. Use the `create()` factory method to construct.

    The bucket capacity determines how parameters are grouped for AllReduce:
    - Smaller buckets: more frequent, smaller AllReduce operations
    - Larger buckets: less frequent, larger AllReduce operations

    Attributes:
        bucket_bytes_cap: The primary bucket size limit in bytes. This is passed
            to the C++ Reducer as the default bucket size.

        per_bucket_bytes_caps: Per-bucket size limits in bytes when user specifies
            bucket_cap_mb_list. Empty tuple means use uniform bucket_bytes_cap.

        first_bucket_bytes_cap: Size limit for the first bucket in bytes. Can be
            smaller than bucket_bytes_cap to reduce latency (first bucket contains
            parameters whose gradients complete last, so a smaller first bucket
            allows AllReduce to start sooner).
    """

    bucket_bytes_cap: int
    per_bucket_bytes_caps: tuple[int, ...]
    first_bucket_bytes_cap: int

    @classmethod
    def create(
        cls,
        bucket_cap_mb: int | None,
        bucket_cap_mb_list: list[int] | None,
        use_python_reducer: bool,
    ) -> "_BucketCapacityConfig":
        """Factory method to create a BucketCapacityConfig from user inputs.

        Args:
            bucket_cap_mb: Single bucket size limit in MiB. If None, uses default
                of 25 MiB. Ignored if bucket_cap_mb_list is provided.
            bucket_cap_mb_list: Per-bucket size limits in MiB. When provided, each
                bucket gets its own size limit.
            use_python_reducer: Whether Python reducer is being used (for validation).

        Returns:
            A new BucketCapacityConfig instance.

        Raises:
            AssertionError: If bucket_cap_mb_list is provided with Python reducer.
        """
        is_using_default = bucket_cap_mb is None
        effective_bucket_cap_mb = (
            bucket_cap_mb if bucket_cap_mb is not None else _DEFAULT_BUCKET_CAP_MB
        )

        # Process per-bucket size list if provided
        per_bucket_bytes_caps: tuple[int, ...] = ()
        if bucket_cap_mb_list:
            if use_python_reducer:
                raise AssertionError(
                    "when using bucket_cap_mb_list, python reducer is not supported"
                )
            per_bucket_bytes_caps = tuple(
                int(cap_mb * _MB_TO_BYTES) for cap_mb in bucket_cap_mb_list
            )
            effective_bucket_cap_mb = max(bucket_cap_mb_list)
            is_using_default = False

        bucket_bytes_cap = int(effective_bucket_cap_mb * _MB_TO_BYTES)

        # First bucket optimization: use smaller size when using defaults
        # to reduce latency for early-computed gradients
        first_bucket_bytes_cap = (
            dist._DEFAULT_FIRST_BUCKET_BYTES if is_using_default else bucket_bytes_cap
        )

        return cls(
            bucket_bytes_cap=bucket_bytes_cap,
            per_bucket_bytes_caps=per_bucket_bytes_caps,
            first_bucket_bytes_cap=first_bucket_bytes_cap,
        )

    @property
    def bucket_cap_mb(self) -> int:
        """Return bucket size in MiB (for backward compatibility)."""
        return self.bucket_bytes_cap // _MB_TO_BYTES

    @property
    def has_custom_per_bucket_caps(self) -> bool:
        """Return True if user specified per-bucket size limits."""
        return len(self.per_bucket_bytes_caps) > 0

    def compute_bucket_size_limits(
        self,
        static_graph: bool,
        find_unused_parameters: bool,
    ) -> tuple[list[int], list[int]]:
        """Compute bucket size limits for initial bucketing and rebuilding.

        Args:
            static_graph: Whether the computation graph is static.
            find_unused_parameters: Whether to find unused parameters.

        Returns:
            A tuple of (bucket_size_limits, bucket_size_limits_for_rebuilding):
            - bucket_size_limits: Used for initial bucket assignment
            - bucket_size_limits_for_rebuilding: Passed to C++ Reducer (empty list
              means use C++ Reducer's default logic)
        """
        # Case 1: User provided explicit per-bucket size limits
        if self.has_custom_per_bucket_caps:
            limits = list(self.per_bucket_bytes_caps)
            return (limits, limits)

        # Case 2: Compute default bucket size limits
        # When static_graph or not finding unused params, disable initial bucketing
        # by using maxsize (put all params in one bucket initially)
        if static_graph or not find_unused_parameters:
            bucket_size_limits = [sys.maxsize]
        elif self.first_bucket_bytes_cap < self.bucket_bytes_cap:
            # Use smaller first bucket for latency optimization
            bucket_size_limits = [self.first_bucket_bytes_cap, self.bucket_bytes_cap]
        else:
            bucket_size_limits = [self.bucket_bytes_cap]

        # Empty list lets C++ Reducer use its default logic
        return (bucket_size_limits, [])


if dist.is_available():
    from torch.distributed.distributed_c10d import (
        _get_default_group,
        _rank_not_in_group,
        ReduceOp,
    )
    from torch.distributed.utils import (
        _alloc_storage,
        _cast_forward_inputs,
        _free_storage,
        _sync_module_states,
        _to_kwargs,
        _verify_param_shape_across_processes,
    )
if dist.rpc.is_available():
    RPC_AVAILABLE = True
    from torch.distributed.rpc import RRef

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


__all__ = ["DistributedDataParallel"]

logger = logging.getLogger(__name__)


@dataclass
class _MixedPrecision:
    """
    This configures DDP-native mixed precision training.

    Attributes:
        param_dtype (torch.dtype): This specifies the dtype for model
            parameters, inputs (when ``cast_forward_inputs`` is set to
            ``True``), and therefore the dtype for computation.
            However, outside the forward and backward passes, parameters are in
            full precision. Model checkpointing always happens in full
            precision.
        reduce_dtype (torch.dtype): This specifies the dtype for gradient
            reduction, which is permitted to differ from ``param_dtype``.
        buffer_dtype (torch.dtype): This specifies the dtype for buffers.

    .. note:: This API is experimental and subject to change.

    .. note:: Only floating point tensors are cast to their specified dtypes.

    .. note:: ``state_dict`` checkpoints parameters and buffers in full
        precision.

    .. note:: Each low precision dtype must be specified explicitly. For
        example, ``_MixedPrecision(reduce_dtype=torch.float16)`` only specifies
        the reduction dtype to be low precision, and DDP will not cast
        parameters or buffers.

    .. note:: If a ``reduce_dtype`` is not specified, then gradient reduction
        happens in ``param_dtype`` if specified or the original parameter dtype
        otherwise. For example, ``_MixedPrecision(param_dtype=torch.float16)``
        would result in communication occurring in fp16.
    """

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    buffer_dtype: torch.dtype | None = None
    # TODO (rohan-varma): keep_low_precision_grads: bool = False
    # TODO (rohan-varma): APIs to allow users to run batchnorm and layernorm
    # in full precision. For DDP, this can be implemented by not performing the
    # parameter cast for BN and LN units.


def _cast_buffers(mixed_precision_config, root_module):
    """Casts buffers to the given ``buffer_dtype``."""
    for buf in root_module.buffers():
        if hasattr(buf, "_ddp_ignored") and buf._ddp_ignored:
            continue

        buf.data = buf.to(dtype=mixed_precision_config.buffer_dtype)


def _setup_mixed_precision_params(mixed_precision_config, root_module):
    """Create and free storage for the mixed precision parameters."""
    for param in root_module.parameters():
        # Do not setup mixed precision for DDP ignored parameters.
        if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
            continue

        if not hasattr(param, "_mp_param"):
            param._mp_param = torch.zeros_like(
                param,
                device=param.device,
                dtype=mixed_precision_config.param_dtype,
                requires_grad=param.requires_grad,
            )
            _free_storage(param._mp_param)
            # _fp_param will point to the full precision param so it can be switched
            # back to at the end of forward / backward.
            param._fp_param = param.data


def _tree_flatten_with_rref(output):
    output_is_rref = RPC_AVAILABLE and isinstance(output, RRef)
    if output_is_rref:
        output_tensor_list, treespec = tree_flatten(output.local_value())
    else:
        output_tensor_list, treespec = tree_flatten(output)
    # Need to return flattened tensors, spec to re-pack them, as well
    # as if the return type was actually an RRef to reconstruct.
    return output_tensor_list, treespec, output_is_rref


def _tree_unflatten_with_rref(output, treespec, output_is_rref):
    output = tree_unflatten(output, treespec)
    if output_is_rref:
        output = RRef(output)
    return output


def _find_tensors(obj):
    r"""Recursively find all tensors contained in the specified object."""
    if RPC_AVAILABLE and isinstance(obj, RRef):
        # If the current node is the owner of the RRef, unwrap it and try to
        # find Tensors.
        # TODO: Expand to remote RRefs.
        if obj.is_owner():
            return _find_tensors(obj.local_value())
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain.from_iterable(map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain.from_iterable(map(_find_tensors, obj.values()))
    if is_dataclass(obj):
        return itertools.chain.from_iterable(
            map(_find_tensors, (getattr(obj, f.name) for f in fields(obj)))
        )

    return []


def _dump_DDP_relevant_env_vars():
    relevant_env_vars = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_PORT",
        "MASTER_ADDR",
        "CUDA_VISIBLE_DEVICES",
        "GLOO_SOCKET_IFNAME",
        "GLOO_DEVICE_TRANSPORT",
        "NCCL_SOCKET_IFNAME",
        "TORCH_NCCL_BLOCKING_WAIT",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_IB_DISABLE",
        # More NCCL env vars:
        "NCCL_P2P_DISABLE",
        "NCCL_P2P_LEVEL",
        "NCCL_SHM_DISABLE",
        "NCCL_SOCKET_NTHREADS",
        "NCCL_NSOCKS_PERTHREAD",
        "NCCL_BUFFSIZE",
        "NCCL_NTHREADS",
        "NCCL_RINGS",
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_CHECKS_DISABLE",
        "NCCL_CHECK_POINTERS",
        "NCCL_LAUNCH_MODE",
        "NCCL_IB_HCA",
        "NCCL_IB_TIMEOUT",
        "NCCL_IB_RETRY_CNT",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_SL",
        "NCCL_IB_TC",
        "NCCL_IB_AR_THRESHOLD",
        "NCCL_IB_CUDA_SUPPORT",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_NET_GDR_READ",
        "NCCL_SINGLE_RING_THRESHOLD",
        "NCCL_LL_THRESHOLD",
        "NCCL_TREE_THRESHOLD",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_IGNORE_CPU_AFFINITY",
        "NCCL_DEBUG_FILE",
        "NCCL_COLLNET_ENABLE",
        "NCCL_TOPO_FILE",
        "NCCL_TOPO_DUMP_FILE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ]
    formatted_output = ""
    for var in relevant_env_vars:
        value = os.environ.get(var, "N/A")
        formatted_output += f"env:{var}={value}\n"
    print(formatted_output)


class _BufferCommHookLocation(Enum):
    PRE_FORWARD = auto()
    POST_FORWARD = auto()


@dataclass
class _BufferCommHook:
    buffer_comm_hook: Callable
    buffer_comm_hook_state: Any
    buffer_comm_hook_location: _BufferCommHookLocation


# Add a DDPSink to run various functions when backwards starts, such as
# queueing call back of out-most backward/graph task,
# this helps call back is fired after all gradients' calculation
# is completed.
class _DDPSink(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, ddp_weakref, *inputs):
        # set_materialize_grads(False) will ensure that None gradients stay as
        # None and are not filled with zeros.
        ctx.set_materialize_grads(False)
        ctx.ddp_weakref = ddp_weakref
        ret = inputs
        if ddp_weakref()._ddp_sink_clone:
            ret = tuple(
                inp.clone() if isinstance(inp, torch.Tensor) else inp for inp in inputs
            )
        return ret

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Enqueue delay allreduce for static graph training on the first
        # iteration.
        ddp_weakref = ctx.ddp_weakref()
        reducer = ddp_weakref.reducer
        static_graph = ddp_weakref.static_graph
        delay_ar_enqueued = (
            static_graph and ddp_weakref._static_graph_delay_allreduce_enqueued
        )
        if static_graph and not delay_ar_enqueued:
            Variable._execution_engine.queue_callback(  # type: ignore[call-arg,misc]
                reducer._delay_all_reduce
            )
            ddp_weakref._static_graph_delay_allreduce_enqueued = True

        return (None, *grad_outputs)


class _DDPJoinHook(JoinHook):
    def __init__(self, ddp, divide_by_initial_world_size):
        """Set config variables for internal usage."""
        if not isinstance(ddp, DistributedDataParallel):
            raise AssertionError(
                "DDP join hook requires passing in a DistributedDataParallel "
                f"instance as the state, got {type(ddp).__name__}"
            )
        if ddp.logger is None:
            raise AssertionError("ddp.logger must not be None")
        ddp.logger._set_uneven_input_join()
        self.ddp = ddp
        self.ddp._divide_by_initial_world_size = divide_by_initial_world_size
        super().__init__()

    def main_hook(self):
        """Shadow the DDP collective communication operations in the forward and backward passes."""
        ddp = self.ddp
        # Buckets are rebuilt only once during a training period
        ddp.reducer._rebuild_buckets()

        # Schedule a broadcast if we are syncing module buffers in the
        # forward pass
        # TODO: make DDP uneven inputs context manager support buffer
        # comm hook (https://github.com/pytorch/pytorch/issues/65436)
        ddp._check_and_sync_module_buffers()

        # Check if need to sync in the backward pass
        should_sync_backwards = ddp._check_global_requires_backward_grad_sync(
            is_joined_rank=True
        )
        # Forward parameter sync is disabled in the next iteration if we
        # are skipping gradient sync this iteration, so set
        # `require_forward_param_sync` accordingly
        ddp.require_forward_param_sync = should_sync_backwards
        if not should_sync_backwards:
            return

        # Schedule one allreduce per gradient bucket to match the backward
        # pass allreduce
        ddp._match_all_reduce_for_bwd_pass()

        # Check if we need to allreduce locally unused parameters
        if ddp.find_unused_parameters:
            ddp._match_unused_params_allreduce()

        # Rebuilt parameters are pushed only once during a training period
        ddp.reducer._push_all_rebuilt_params()

    def post_hook(self, is_last_joiner: bool):
        """Sync the final model to ensure that the model is the same across all processes."""
        self.ddp._sync_final_model(is_last_joiner)


class DistributedDataParallel(Module, Joinable):
    r"""Implement distributed data parallelism based on ``torch.distributed`` at module level.

    This container provides data parallelism by synchronizing gradients
    across each model replica. The devices to synchronize across are
    specified by the input ``process_group``, which is the entire world
    by default. Note that ``DistributedDataParallel`` does not chunk or
    otherwise shard the input across participating GPUs; the user is
    responsible for defining how to do so, for example through the use
    of a :class:`DistributedSampler`.

    See also: :ref:`distributed-basics` and :ref:`cuda-nn-ddp-instead`.
    The same constraints on input as in :class:`torch.nn.DataParallel` apply.

    Creation of this class requires that ``torch.distributed`` to be already
    initialized, by calling :func:`torch.distributed.init_process_group`.

    ``DistributedDataParallel`` is proven to be significantly faster than
    :class:`torch.nn.DataParallel` for single-node multi-GPU data
    parallel training.

    To use ``DistributedDataParallel`` on a host with N GPUs, you should spawn
    up ``N`` processes, ensuring that each process exclusively works on a single
    GPU from 0 to N-1. This can be done by either setting
    ``CUDA_VISIBLE_DEVICES`` for every process or by calling the following API for GPUs,

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.cuda.set_device(i)

    or calling the unified API for :ref:`accelerator<accelerators>`,

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.accelerator.set_device_index(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> # xdoctest: +SKIP("undefined variables")
        >>> if torch.accelerator.is_available():
        >>>     device_type = torch.accelerator.current_accelerator().type
        >>>     vendor_backend = torch.distributed.get_default_backend_for_device(device_type)
        >>>
        >>> torch.distributed.init_process_group(
        >>>     backend=vendor_backend, world_size=N, init_method='...'
        >>> )
        >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    Or you can use the latest API for initialization:

        >>> torch.distributed.init_process_group(device_id=i)

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``.

    .. note::
        Please refer to `PyTorch Distributed Overview <https://pytorch.org/tutorials/beginner/dist_overview.html>`__
        for a brief introduction to all features related to distributed training.

    .. note::
        ``DistributedDataParallel`` can be used in conjunction with
        :class:`torch.distributed.optim.ZeroRedundancyOptimizer` to reduce
        per-rank optimizer states memory footprint. Please refer to
        `ZeroRedundancyOptimizer recipe <https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html>`__
        for more details.

    .. note:: ``nccl`` backend is currently the fastest and highly recommended
        backend when using GPUs. This applies to both single-node and
        multi-node distributed training.

    .. note:: This module also supports mixed-precision distributed training.
        This means that your model can have different types of parameters such
        as mixed types of ``fp16`` and ``fp32``, the gradient reduction on these
        mixed types of parameters will just work fine.

    .. note:: If you use ``torch.save`` on one process to checkpoint the module,
        and ``torch.load`` on some other processes to recover it, make sure that
        ``map_location`` is configured properly for every process. Without
        ``map_location``, ``torch.load`` would recover the module to devices
        where the module was saved from.

    .. note:: When a model is trained on ``M`` nodes with ``batch=N``, the
        gradient will be ``M`` times smaller when compared to the same model
        trained on a single node with ``batch=M*N`` if the loss is summed (NOT
        averaged as usual) across instances in a batch (because the gradients
        between different nodes are averaged). You should take this into
        consideration when you want to obtain a mathematically equivalent
        training process compared to the local training counterpart. But in most
        cases, you can just treat a DistributedDataParallel wrapped model, a
        DataParallel wrapped model and an ordinary model on a single GPU as the
        same (E.g. using the same learning rate for equivalent batch size).

    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in process of rank
        0, to all other replicas in the system in every iteration.

    .. note::
        If you are using DistributedDataParallel in conjunction with the
        :ref:`distributed-rpc-framework`, you should always use
        :meth:`torch.distributed.autograd.backward` to compute gradients and
        :class:`torch.distributed.optim.DistributedOptimizer` for optimizing
        parameters.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> import torch.distributed.autograd as dist_autograd
            >>> from torch.nn.parallel import DistributedDataParallel as DDP
            >>> import torch
            >>> from torch import optim
            >>> from torch.distributed.optim import DistributedOptimizer
            >>> import torch.distributed.rpc as rpc
            >>> from torch.distributed.rpc import RRef
            >>>
            >>> t1 = torch.rand((3, 3), requires_grad=True)
            >>> t2 = torch.rand((3, 3), requires_grad=True)
            >>> rref = rpc.remote("worker1", torch.add, args=(t1, t2))
            >>> ddp_model = DDP(my_model)
            >>>
            >>> # Setup optimizer
            >>> optimizer_params = [rref]
            >>> for param in ddp_model.parameters():
            >>>     optimizer_params.append(RRef(param))
            >>>
            >>> dist_optim = DistributedOptimizer(
            >>>     optim.SGD,
            >>>     optimizer_params,
            >>>     lr=0.05,
            >>> )
            >>>
            >>> with dist_autograd.context() as context_id:
            >>>     pred = ddp_model(rref.to_here())
            >>>     loss = loss_func(pred, target)
            >>>     dist_autograd.backward(context_id, [loss])
            >>>     dist_optim.step(context_id)

    .. note::
        DistributedDataParallel currently offers limited support for gradient
        checkpointing with :meth:`torch.utils.checkpoint`.
        If the checkpoint is done with use_reentrant=False (recommended), DDP
        will work as expected without any limitations.
        If, however, the checkpoint is done with use_reentrant=True (the default),
        DDP will work as expected when there are no unused parameters in the model
        and each layer is checkpointed at most once (make sure you are not passing
        `find_unused_parameters=True` to DDP). We currently do not support the
        case where a layer is checkpointed multiple times, or when there unused
        parameters in the checkpointed model.

    .. note::
        To let a non-DDP model load a state dict from a DDP model,
        :meth:`~torch.nn.modules.utils.consume_prefix_in_state_dict_if_present`
        needs to be applied to strip the prefix "module." in the DDP state dict before loading.

    .. warning::
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) are distributed synchronization
        points. Take that into account in case different processes might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.
        Same applies to buffers.

    .. warning::
        This module assumes all parameters are registered in the model of each
        distributed processes are in the same order. The module itself will
        conduct gradient ``allreduce`` following the reverse order of the
        registered parameters of the model. In other words, it is users'
        responsibility to ensure that each distributed process has the exact
        same model and thus the exact same parameter registration order.

    .. warning::
        This module allows parameters with non-rowmajor-contiguous strides.
        For example, your model may contain some parameters whose
        :class:`torch.memory_format` is ``torch.contiguous_format``
        and others whose format is ``torch.channels_last``.  However,
        corresponding parameters in different processes must have the
        same strides.

    .. warning::
        This module doesn't work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. warning::
        If you plan on using this module with a ``nccl`` backend or a ``gloo``
        backend (that uses Infiniband), together with a DataLoader that uses
        multiple workers, please change the multiprocessing start method to
        ``forkserver`` or ``spawn``. Unfortunately
        Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will
        likely experience deadlocks if you don't change this setting.

    .. warning::
        You should never try to change your model's parameters after wrapping
        up your model with ``DistributedDataParallel``. Because, when
        wrapping up your model with ``DistributedDataParallel``, the constructor
        of ``DistributedDataParallel`` will register the additional gradient
        reduction functions on all the parameters of the model itself at the
        time of construction. If you change the model's parameters afterwards,
        gradient reduction functions no longer match the correct set of
        parameters.

    .. warning::
        Using ``DistributedDataParallel`` in conjunction with the
        :ref:`distributed-rpc-framework` is experimental and subject to change.

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices.
                   1) For single-device modules, ``device_ids`` can
                   contain exactly one device id, which represents the only
                   CUDA device where the input module corresponding to this process resides.
                   Alternatively, ``device_ids`` can also be ``None``.
                   2) For multi-device modules and CPU modules,
                   ``device_ids`` must be ``None``.

                   When ``device_ids`` is ``None`` for both cases,
                   both the input data for the forward pass and the actual module
                   must be placed on the correct device.
                   (default: ``None``)
        output_device (int or torch.device): Device location of output for
                      single-device CUDA modules. For multi-device modules and
                      CPU modules, it must be ``None``, and the module itself
                      dictates the output location. (default: ``device_ids[0]``
                      for single-device modules)
        broadcast_buffers (bool or None): Flag that enables syncing (broadcasting)
                          buffers of the module at beginning of the ``forward``
                          function. (default: ``None``)

                          .. deprecated:: 2.13
                              Use ``forward_sync_buffers`` instead.
        init_sync (bool): Whether to sync during initialization to verify param
                          shapes and broadcast parameters and buffers.
                          Note: the deprecated ``broadcast_buffers=False``
                          excludes buffers from this init sync. The replacement
                          ``forward_sync_buffers`` does not affect init sync.
                          WARNING: if this is set to False the user is required
                          to ensure themselves that the weights are the same on
                          all ranks.
                          (default: ``True``)
        process_group: The process group to be used for distributed data
                       all-reduction. If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)
        bucket_cap_mb: ``DistributedDataParallel`` will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       :attr:`bucket_cap_mb` controls the bucket size in
                       MebiBytes (MiB). If ``None``, a default size of 25 MiB
                       will be used. (default: ``None``)
        find_unused_parameters (bool): Traverse the autograd graph from all
                               tensors contained in the return value of the
                               wrapped module's ``forward`` function. Parameters
                               that don't receive gradients as part of this
                               graph are preemptively marked as being ready to
                               be reduced. In addition, parameters that may have
                               been used in the wrapped module's ``forward``
                               function but were not part of loss computation and
                               thus would also not receive gradients are
                               preemptively marked as ready to be reduced.
                               (default: ``False``)
        check_reduction: This argument is deprecated.
        gradient_as_bucket_view (bool): When set to ``True``, gradients will be views
                      pointing to different offsets of ``allreduce`` communication
                      buckets. This can reduce peak memory usage, where the
                      saved memory size will be equal to the total gradients
                      size. Moreover, it avoids the overhead of copying between
                      gradients and ``allreduce`` communication buckets. When
                      gradients are views, ``detach_()`` cannot be called on the
                      gradients. If hitting such errors, please fix it by
                      referring to the :meth:`~torch.optim.Optimizer.zero_grad`
                      function in ``torch/optim/optimizer.py`` as a solution.
                      Note that gradients will be views after first iteration, so
                      the peak memory saving should be checked after first iteration.
        static_graph (bool): When set to ``True``, DDP knows the trained graph is
                     static. Static graph means 1) The set of used and unused
                     parameters will not change during the whole training loop; in
                     this case, it does not matter whether users set
                     ``find_unused_parameters = True`` or not. 2) How the graph is trained
                     will not change during the whole training loop (meaning there is
                     no control flow depending on iterations).
                     When static_graph is set to be ``True``, DDP will support cases that
                     can not be supported in the past:
                     1) Reentrant backwards.
                     2) Activation checkpointing multiple times.
                     3) Activation checkpointing when model has unused parameters.
                     4) There are model parameters that are outside of forward function.
                     5) Potentially improve performance when there are unused parameters,
                     as DDP will not search graph in each iteration to detect unused
                     parameters when static_graph is set to be ``True``.
                     To check whether you can set static_graph to be ``True``, one way is to
                     check ddp logging data at the end of your previous model training,
                     if ``ddp_logging_data.get("can_set_static_graph") == True``, mostly you
                     can set ``static_graph = True`` as well.

                     Example::
                         >>> # xdoctest: +SKIP("undefined variables")
                         >>> model_DDP = torch.nn.parallel.DistributedDataParallel(model)
                         >>> # Training loop
                         >>> ...
                         >>> ddp_logging_data = model_DDP._get_ddp_logging_data()
                         >>> static_graph = ddp_logging_data.get("can_set_static_graph")
        delay_all_reduce_named_params (list of tuple of str and torch.nn.Parameter): a list
                    of named parameters whose all reduce will be delayed when the gradient of
                    the parameter specified in ``param_to_hook_all_reduce`` is ready. Other
                    arguments of DDP do not apply to named params specified in this argument
                    as these named params will be ignored by DDP reducer.
        param_to_hook_all_reduce (torch.nn.Parameter): a parameter to hook delayed all reduce
                    of parameters specified in ``delay_all_reduce_named_params``.
        skip_all_reduce_unused_params: When set to True, DDP will skip reducing unused parameters.
                    This requires that unused parameters remain the same across all ranks throughout
                    the entire training process. If this condition is not met, it may cause
                    desynchronization and result in training hang.
        batched_grad_copy (bool): When set to ``True``, individual per-parameter
                    gradient-to-bucket copy and division operations are deferred
                    and flushed as a single ``_foreach_copy_`` plus one flat
                    ``div_`` when a bucket becomes ready. This reduces per-parameter
                    kernel launches down to 2 kernels per bucket, which can improve
                    throughput for models with many small parameters. The
                    optimization is most effective with
                    ``optimizer.zero_grad(set_to_none=True)`` (the default), where
                    ``gradient_as_bucket_view`` alone cannot avoid copies because
                    the bucket view alias is destroyed every iteration.
                    (default: ``False``)
        forward_sync_buffers (bool or None): Flag that enables syncing
                    (broadcasting) buffers of the module at runtime, including
                    at the beginning of ``forward`` and after uneven-input
                    joins. Does not affect initialization sync (see
                    ``init_sync``); buffers are always synced at init
                    regardless of this flag. Replaces the deprecated
                    ``broadcast_buffers`` argument. When ``None``, defaults
                    to ``True``. (default: ``None``)


    Attributes:
        module (Module): the module to be parallelized.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = torch.nn.parallel.DistributedDataParallel(model)
    """

    # used to track whether the given thread is inside ddp forward for torchdynamo purposes
    _active_ddp_module: Optional["DistributedDataParallel"] = None

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=None,
        init_sync=True,
        process_group=None,
        bucket_cap_mb=None,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        static_graph=False,
        delay_all_reduce_named_params=None,
        param_to_hook_all_reduce=None,
        mixed_precision: _MixedPrecision | None = None,
        device_mesh=None,
        skip_all_reduce_unused_params=False,
        bucket_cap_mb_list: list[int] | None = None,
        batched_grad_copy=False,
        forward_sync_buffers: bool | None = None,
    ):
        super().__init__()
        Joinable.__init__(self)
        self._use_python_reducer = (
            torch._dynamo.utils.get_optimize_ddp_mode() == "python_reducer"
        )
        self.logger: dist.Logger | None = None
        if bool(delay_all_reduce_named_params is not None) != bool(
            param_to_hook_all_reduce is not None
        ):
            self._log_and_throw(
                ValueError,
                "delay_all_reduce_named_params and param_to_hook_all_reduce "
                "need to be set at the same time.",
            )

        if process_group and device_mesh is not None:
            raise RuntimeError(
                "Cannot specify both process_group and device_mesh arguments."
            )
        elif process_group is None and device_mesh is None:
            self.process_group = _get_default_group()
        elif device_mesh is None:
            # pyrefly: ignore [bad-assignment]
            self.process_group = process_group
        else:
            if device_mesh.ndim != 1:
                raise RuntimeError(
                    f"Only 1D device mesh is supported, but got {device_mesh}."
                )
            self.device_mesh = device_mesh
            self.process_group = device_mesh.get_group(mesh_dim=0)

            root_mesh = device_mesh._get_root_mesh()
            # if a root mesh is not the same as device_mesh,
            # meaning the device_mesh is sliced out from the root mesh.
            if root_mesh != device_mesh:
                # TODO: This is a temporary work around to enable DDP + TP.
                # We should do the logic in DDP so that the 2D implementation is
                # sound and the state_dict works out of the box.
                # This has to be done before check UninitializedParameter.
                from torch.distributed.tensor.parallel.ddp import (
                    _pre_dp_module_transform,
                )

                _pre_dp_module_transform(module)

        self._delay_all_reduce_params = []
        if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = set(module._ddp_params_and_buffers_to_ignore)
        else:
            self.parameters_to_ignore = set()
        if delay_all_reduce_named_params is not None:
            for name, param in delay_all_reduce_named_params:
                self.parameters_to_ignore.add(name)
                self._delay_all_reduce_params.append(param)

        self._module_parameters = [
            p
            for n, p in module.named_parameters()
            if n not in self.parameters_to_ignore
        ]
        if not any(p.requires_grad for p in self._module_parameters):
            if len(self._delay_all_reduce_params):
                logger.info("Delay the AllReduce of all parameters.")
            else:
                self._log_and_throw(
                    RuntimeError,
                    "DistributedDataParallel is not needed when a module "
                    "doesn't have any parameter that requires a gradient.",
                )

        if device_ids is not None and len(device_ids) > 1:
            self._log_and_throw(
                ValueError,
                "device_ids can only be None or contain a single element.",
            )

        self.is_multi_device_module = (
            len({p.device for p in self._module_parameters}) > 1
        )
        distinct_device_types = {
            p.device.type for p in self._module_parameters if p.device is not None
        }
        if len(distinct_device_types) != 1:
            self._log_and_throw(
                ValueError,
                "DistributedDataParallel's input module must be on "
                f"the same type of devices, but input module parameters locate in {distinct_device_types}.",
            )

        self.device_type = next(iter(distinct_device_types))

        if (
            device_ids is None
            or len(device_ids) == 0  # For backward compatibility.
            or self.device_type == "cpu"
            or self.is_multi_device_module
        ):
            if device_ids or output_device:
                devices = {p.device for p in self._module_parameters}
                self._log_and_throw(
                    ValueError,
                    "DistributedDataParallel device_ids and output_device arguments "
                    "only work with single-device/multiple-device GPU modules or CPU modules, "
                    f"but got device_ids {device_ids}, output_device {output_device}, "
                    f"and module parameters {devices}.",
                )

            self.device_ids = None
            self.output_device = None
        else:
            self.device_ids = [_get_device_index(x, True) for x in device_ids]

            if output_device is None:
                output_device = device_ids[0]

            self.output_device = _get_device_index(output_device, True)

        self.static_graph = False
        self.dim = dim
        self.module = module
        self.device = next(iter(self._module_parameters)).device
        if broadcast_buffers is not None:
            if forward_sync_buffers is not None:
                warnings.warn(
                    "Both `broadcast_buffers` and `forward_sync_buffers` were "
                    "specified. `broadcast_buffers` is deprecated and its value "
                    "is ignored when `forward_sync_buffers` is provided.",
                    FutureWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "`broadcast_buffers` is deprecated. Use `forward_sync_buffers` "
                    "instead. IMPORTANT: unlike `broadcast_buffers=False`, "
                    "`forward_sync_buffers=False` still syncs buffers at "
                    "init. If you rely on buffers NOT being synced at "
                    "init (e.g. rank-local buffers), keep using "
                    "`broadcast_buffers=False` until `init_sync_buffers` "
                    "is available.",
                    FutureWarning,
                    stacklevel=2,
                )
        if forward_sync_buffers is not None:
            self.forward_sync_buffers = forward_sync_buffers
            init_sync_buffers = True
        else:
            effective_broadcast = (
                broadcast_buffers if broadcast_buffers is not None else True
            )
            self.forward_sync_buffers = effective_broadcast
            init_sync_buffers = effective_broadcast
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.batched_grad_copy = batched_grad_copy
        self.mixed_precision = mixed_precision
        if self.mixed_precision is not None:
            logger.warning("Received mixed precision config %s", self.mixed_precision)

        if check_reduction:
            # This argument is no longer used since the reducer
            # will ensure reduction completes even if some parameters
            # do not receive gradients.
            warnings.warn(
                "The `check_reduction` argument in `DistributedDataParallel` "
                "module is deprecated. Please avoid using it.",
                FutureWarning,
                stacklevel=2,
            )

        # Check that a module does not have Uninitialized parameters
        for param in self._module_parameters:
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                self._log_and_throw(
                    RuntimeError,
                    "Modules with uninitialized parameters can't be used with `DistributedDataParallel`. "
                    "Run a dummy forward pass to correctly initialize the modules",
                )
        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = 250 * 1024 * 1024

        # Initialize bucket capacity configuration using the factory method
        self._bucket_config = _BucketCapacityConfig.create(
            bucket_cap_mb=bucket_cap_mb,
            bucket_cap_mb_list=bucket_cap_mb_list,
            use_python_reducer=self._use_python_reducer,
        )
        # Expose config values as instance attributes for backward compatibility
        # TODO: Remove in the future
        self.bucket_bytes_cap = self._bucket_config.bucket_bytes_cap
        self.bucket_bytes_cap_list = list(self._bucket_config.per_bucket_bytes_caps)
        self.bucket_bytes_cap_default = (
            self._bucket_config.first_bucket_bytes_cap
            < self._bucket_config.bucket_bytes_cap
        )
        bucket_cap_mb = self._bucket_config.bucket_cap_mb

        # Whether to perform input tensor CPU to GPU copies on a side-stream
        self.use_side_stream_for_tensor_copies = (
            os.environ.get("PYTORCH_DDP_USE_SIDE_STREAM", "1") == "1"
        )

        # Initialize gradient buffers and register all reduce hook
        self._delay_grad_buffer: torch.Tensor | None = None
        self._delay_grad_views: list[torch.Tensor] = []
        self._delay_all_reduce_all_params = False
        if len(self._delay_all_reduce_params) != 0:
            self._register_delay_all_reduce_hook(
                bucket_cap_mb=bucket_cap_mb,
                param_to_hook_all_reduce=param_to_hook_all_reduce,
                device_ids=device_ids,
            )
            if self._delay_all_reduce_all_params:
                return

        self.skip_all_reduce_unused_params = skip_all_reduce_unused_params

        # Build parameters for reducer.
        parameters, expect_sparse_gradient = self._build_params_for_reducer()

        # All collectives during initialization are gated by this flag.
        if init_sync:
            # Verify model equivalence.
            _verify_param_shape_across_processes(self.process_group, parameters)
            # Sync params and buffers. Ensures all DDP models start off at the same value.
            _sync_module_states(
                module=self.module,
                process_group=self.process_group,
                broadcast_bucket_size=self.broadcast_bucket_size,
                src=0,
                params_and_buffers_to_ignore=self.parameters_to_ignore,
                broadcast_buffers=init_sync_buffers,
            )

        # In debug mode, build a mapping of parameter index -> parameter.
        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)

        # Builds reducer.
        self._ddp_init_helper(
            parameters,
            expect_sparse_gradient,
            param_to_name_mapping,
            static_graph,
        )
        self._comm_hooks: list[tuple[Callable, object]] = []

        if self.mixed_precision is not None:
            _setup_mixed_precision_params(self.mixed_precision, self.module)
            _cast_buffers(self.mixed_precision, self.module)
            # Stream used for async low precision copies.
            self._mp_stream = torch.Stream()
            self._submodule_to_event = defaultdict(deque)  # type: ignore[var-annotated]
            # Add forward pre-hook to root module to kick off copies to lower
            # precision.
            self.module.register_forward_pre_hook(
                self._root_copy_hook, prepend=False, with_kwargs=True
            )
            # Add forward pre hook to all submodules to wait for copy events
            # before running computation.
            for module in self.module.modules():
                module.register_forward_pre_hook(
                    self._module_wait_for_copy_hook,
                    prepend=False,
                    with_kwargs=True,
                )
            # Set up callbacks in backward to upcast and use full precision
            # params. TODO (rohan-varma): Make this compose with general
            # comm hooks and apply_optimizer_in_backward. Importing inline to
            # avoid circular import issue.
            from torch.distributed.algorithms.ddp_comm_hooks.mixed_precision_hooks import (
                _AllreduceUpcastHookState,
                _reducer_allreduce_and_upcast_hook,
            )

            upcast_hook_state = _AllreduceUpcastHookState(
                ddp_weakref=weakref.ref(self),
                upcast_stream=torch.Stream(),
            )
            self.register_comm_hook(
                upcast_hook_state,
                _reducer_allreduce_and_upcast_hook,
            )
            # Inform reducer of reduced precision param dtype for correctness
            # of type checks between gradient and bucket.
            self.reducer._set_mixed_precision_param_dtype(  # type: ignore[attr-defined]
                self.mixed_precision.param_dtype
            )

        self._has_rebuilt_buckets = False

        if static_graph:
            self._set_static_graph()

        self._lazy_init_ran = False

        # Register the AccumulateGrad post hooks if optimize_ddp is
        # True. The hooks will be deregistered if compiled_autograd is not
        # enabled.
        self._accum_grad_hooks: list[RemovableHandle] = []
        if self._use_python_reducer:
            # pyrefly: ignore [bad-assignment]
            torch._inductor.config._fuse_ddp_communication = True
            # pyrefly: ignore [bad-assignment]
            torch._inductor.config._fuse_ddp_bucket_size = bucket_cap_mb
            # Directly adding this to the trace rule will disturb the users
            # who are using DDPOptimizer.
            torch._dynamo.trace_rules.LEGACY_MOD_INLINELIST.add(
                "torch.nn.parallel.distributed"
            )
            torch._dynamo.trace_rules.get_legacy_mod_inlinelist.cache_clear()
            # NOTE: we should init these lazily
            self._register_accum_grad_hook()

        # Whether or not DDPSink performs a clone.
        self._ddp_sink_clone = True

    @property
    def broadcast_buffers(self) -> bool:
        return self.forward_sync_buffers

    @broadcast_buffers.setter
    def broadcast_buffers(self, value: bool) -> None:
        self.forward_sync_buffers = value

    def _register_accum_grad_hook(self):
        import torch.distributed._functional_collectives as fcol

        def compiled_accum_grad_hook(
            param,
            *,
            param_index: int,
        ):
            if not self.require_backward_grad_sync:
                return

            if param.grad is None:
                return

            if self._comm_hooks:
                for hook, state in self._comm_hooks:
                    hook(state, (param.grad, param))
            else:
                gradient = param.grad / self.process_group.size()
                gradient = fcol.all_reduce(gradient, "sum", self.process_group)
                param.grad.copy_(gradient)

        for index, param in enumerate(self._module_parameters):
            if not param.requires_grad:
                continue
            self._accum_grad_hooks.append(
                param.register_post_accumulate_grad_hook(
                    functools.partial(
                        compiled_accum_grad_hook,
                        param_index=index,
                    )
                )
            )

    def _delayed_all_reduce_hook(self, grad):
        world_size = dist.get_world_size(self.process_group)

        delay_grad_buffer = self._delay_grad_buffer
        if delay_grad_buffer is None:
            raise AssertionError("Delayed gradient buffer is not initialized")
        delay_grad_buffer.div_(world_size)
        _ = dist.all_reduce(delay_grad_buffer, group=self.process_group, async_op=True)
        return grad

    def _register_delay_all_reduce_hook(
        self,
        bucket_cap_mb,
        param_to_hook_all_reduce,
        device_ids,
    ):
        # 1. Create gradient buffer
        device = torch.device("cpu") if device_ids is None else device_ids[0]
        self._delay_grad_buffer = torch.zeros(
            sum(p.numel() for p in self._delay_all_reduce_params),
            device=device,
        )

        # 2. Broadcast the parameters
        detached_params = [p.detach() for p in self._delay_all_reduce_params]
        dist._broadcast_coalesced(self.process_group, detached_params, bucket_cap_mb, 0)

        # 3. Hook all reduce to the specified parameter
        param_to_hook_all_reduce.register_hook(self._delayed_all_reduce_hook)

        # 4. Build tensor views for gradients
        offset = 0
        for param in self._delay_all_reduce_params:
            grad_view = self._delay_grad_buffer[offset : (offset + param.numel())].view(
                param.shape
            )
            self._delay_grad_views.append(grad_view)
            offset = offset + param.numel()

        # 5. Check whether the all reduce of all params requiring grad is delayed.
        for module_name, module in self.module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    full_name = f"{module_name}.{param_name}"
                    if full_name not in self.parameters_to_ignore:
                        # There is at least a param whose all reduce will not be delayed.
                        # In this case, we should not set self._delay_all_reduce_all_params
                        # to True.
                        return
        self._delay_all_reduce_all_params = True

    def _setup_in_backward_optimizers(self):
        # Check if user has used apply_optim_in_backward to overlap optimizer
        # step + DDP backward. Current constraints:
        # 1. Only allreduce is supported at the moment, no custom communication.
        # 2. For DDP-managed parameters that have their optimizer run in
        # backward, their gradients are set to ``None``. If your use case
        # requires DDP parameters grad not to be set to ``None`` after their
        # in-backward optimizer runs, please ping
        # https://github.com/pytorch/pytorch/issues/90052.
        # NOTE: we use self._module_parameters instead of .parameters() since
        # the former excludes ignored (non-DDP managed) parameters.
        if any(hasattr(p, "_in_backward_optimizers") for p in self._module_parameters):
            torch._C._log_api_usage_once("ddp.optimizer_in_backward")
            # Remove hooks that apply_optim_in_backward had registered because
            # DDP customizes how optimizer is overlapped with backward due to
            # the allreduce.
            param_to_handle_map = (
                dist.optim.apply_optimizer_in_backward.param_to_optim_hook_handle_map
            )
            for p in self._module_parameters:
                for handle in param_to_handle_map.get(p, []):
                    handle.remove()

            # Need a weakref to DDP instance to run all_reduce (from reducer)
            # and get managed DDP parameters.
            ddp_weakref = weakref.ref(self)
            # Note: importing in function, otherwise this will cause a circular
            # import.
            from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
                _apply_optim_in_backward_hook,
            )

            self.register_comm_hook(
                ddp_weakref,
                _apply_optim_in_backward_hook(
                    gradient_is_bucket_view=self.gradient_as_bucket_view
                ),
            )

            self.reducer._set_optimizer_in_backward()  # type: ignore[attr-defined]

    def _fire_reducer_autograd_hook(self, idx, *unused):
        """
        Fire the reducer's autograd hook to allreduce params in a Reducer bucket.

        Note that this is only used during mixed precision training as the
        Reducer's hooks installed during construction time would not be called
        as we're working in the low precision parameter setting.
        """
        self.reducer._autograd_hook(idx)  # type: ignore[attr-defined]

    def _root_copy_hook(self, *args: Any, **kwargs: Any) -> None:
        """
        For DDP mixed precision, put low precision copies on separate stream and create events to wait for them.

        When training with DDP mixed precision, this root pre-forward hook kicks
        off low precision copies on a separate stream and creates respective
        events to wait for them.
        """
        # Clear out previous iteration submodule to event. This is because we
        # may have populated some events for modules that didn't end up being
        # used.
        self._submodule_to_event = defaultdict(deque)  # type: ignore[var-annotated]
        with self._mp_stream:
            for submodule in self.module.modules():
                for param in submodule.parameters(recurse=False):
                    # Do not cast DDP ignored parameters.
                    if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
                        continue
                    _alloc_storage(param._mp_param, param.size())
                    # copy() implicitly casts to low precision
                    with torch.no_grad():
                        param._mp_param.copy_(param.data)
                        # TODO: when zero_grad(set_to_none=False) or in grad
                        # accumulation case, accumulated grads can be in fp32
                        # which can cause errors when running DDP backwards due
                        # to mismatched incoming and accumulated gradient types.
                        # So we manually cast the accumulated grad down for now,
                        # in the future we may shift to FSDP style gradient
                        # accumulation management where the accumulated gradient
                        # is saved and .grad field is set to None, bypassing
                        # this issue.
                        if param.grad is not None:
                            param.grad.data = param.grad.to(
                                self.mixed_precision.param_dtype  # type: ignore[union-attr]
                            )
                    param.data = param._mp_param
                copy_event = torch.Event()
                copy_event.record()
                self._submodule_to_event[submodule].append(copy_event)

    def _module_wait_for_copy_hook(
        self,
        module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Before carrying out computation, wait on the appropriate event to ensure low precision copies have finished."""
        try:
            event = self._submodule_to_event[module].popleft()
        except IndexError:
            # copy event has already been waited on
            return

        event.wait(stream=torch.accelerator.current_stream())
        for p in module.parameters(recurse=False):
            # Don't register hooks if param does not require grad
            if not p.requires_grad or (hasattr(p, "_ddp_ignored") and p._ddp_ignored):
                continue
            # We need to register autograd hook here instead of DDP's ctor
            # since we're working with the low precision param. Register them
            # via obtaining the gradient accumulator.
            tmp = p.expand_as(p)
            grad_acc = tmp.grad_fn.next_functions[0][0]

            hook = grad_acc.register_hook(
                functools.partial(self._fire_reducer_autograd_hook, p._idx)
            )
            p._ddp_mp_hook_state = (grad_acc, hook)

    def _log_and_throw(self, err_type, err_msg):
        if self.logger is not None:
            self.logger.set_error_and_log(f"{str(err_type)}: {err_msg}")
        raise err_type(err_msg)

    def _ddp_init_helper(
        self,
        parameters,
        expect_sparse_gradient,
        param_to_name_mapping,
        static_graph,
    ):
        """
        DDP init helper function to manage parameters, grad hooks, logging, and SyncBatchNorm.

        Initialization helper function that does the following:
        (1) bucketing the parameters for reductions
        (2) resetting the bucketing states
        (3) registering the grad hooks
        (4) Logging construction-time DDP logging data
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """
        # Notice, the parameters order is not in the order in which they are used,
        # especially in models with control flow.
        #
        # Alongside parameters are not presented in the real execution order,
        # if a certain model happens to also
        #   1) have other collectives comm ops in its backward graph.
        #   2) have unused parameter in subset ranks of the whole world.
        # bucketing could insert ALL-REDUCE comm op too early on the rank with unused parameter,
        # matching up with other collectives comm ops on other ranks unexpectedly.
        #
        # In order to handle this corner case, when the parameters are not in the real execution order,
        # we don't do bucketing, thus only one ALL-REDUCE is inserted after all the gradients
        # of the whole graph are computed.
        #
        # Notice, here we only disable bucketing for the first iteration.
        # After the first iteration, it's OK to rebuild buckets,
        # because "bucket rebuild" bucketizes parameters based on its real execution order in backward graph.

        # Compute bucket size limits for initial bucketing and rebuilding
        # See BucketCapacityConfig.compute_bucket_size_limits for detailed explanation
        (
            bucket_size_limits,
            bucket_size_limits_for_rebuilding,
        ) = self._bucket_config.compute_bucket_size_limits(
            static_graph=static_graph,
            find_unused_parameters=self.find_unused_parameters,
        )

        (
            bucket_indices,
            per_bucket_size_limits,
        ) = dist._compute_bucket_assignment_by_size(
            parameters,
            bucket_size_limits,
            expect_sparse_gradient,
        )

        # Remember index for parameters if we are in mixed precision, as we
        # need to pass in index to Reducer's autograd hook via python.
        if self.mixed_precision is not None:
            for i, p in enumerate(parameters):
                p._idx = i

        # Note: reverse list of buckets because we want to approximate the
        # order in which their gradients are produced, and assume they
        # are used in the forward pass in the order they are defined.
        self.reducer = dist.Reducer(
            parameters,
            list(reversed(bucket_indices)),
            list(reversed(per_bucket_size_limits)),
            self.process_group,
            expect_sparse_gradient,
            # The bucket size limit is specified in the constructor.
            # Additionally, we allow for a single small bucket for parameters
            # that are defined first, such that their gradients don't spill into
            # a much larger bucket, adding unnecessary latency after gradient
            # computation finishes. Experiments showed 1MB is a reasonable value.
            self.bucket_bytes_cap,
            self.find_unused_parameters,
            self.gradient_as_bucket_view,
            param_to_name_mapping,
            # First bucket can be smaller for reduced latency. See
            # BucketCapacityConfig.first_bucket_bytes_cap for details.
            self._bucket_config.first_bucket_bytes_cap,
            self.skip_all_reduce_unused_params,
            self._use_python_reducer,
            bucket_size_limits_for_rebuilding,
            self.batched_grad_copy,
        )

        self.logger = dist.Logger(self.reducer)
        # Set as a weak reference to avoid reference cycle between
        # logger and reducer.
        self.reducer.set_logger(self.logger)

        has_sync_bn = False
        for submodule in self.module.modules():
            if isinstance(submodule, torch.nn.SyncBatchNorm):
                has_sync_bn = True
                break

        # Set logging data that can be got during construction time.
        self.logger.set_construction_data_and_log(
            self.module.__class__.__name__,
            [] if self.device_ids is None else self.device_ids,
            -1 if self.output_device is None else self.output_device,
            self.forward_sync_buffers,
            has_sync_bn,
            static_graph,
        )

        # passing a handle to torch.nn.SyncBatchNorm layer
        self._passing_sync_batchnorm_handle(self.module)

    def __getstate__(self):
        self._check_default_group()
        attrs = copy.copy(self.__dict__)
        del attrs["process_group"]
        del attrs["reducer"]
        del attrs["logger"]
        # broadcast_buffers is a property; include it for backward compat.
        attrs["broadcast_buffers"] = self.forward_sync_buffers
        return attrs

    def __setstate__(self, state):
        # If serializable, then the process group should be the default one
        self.process_group = _get_default_group()
        super().__setstate__(state)
        self.__dict__.setdefault("require_forward_param_sync", True)
        self.__dict__.setdefault("require_backward_grad_sync", True)
        # broadcast_buffers is now a property; pop it so it doesn't shadow.
        # Old pickles may only have broadcast_buffers; use it to seed forward_sync_buffers.
        old_broadcast = self.__dict__.pop("broadcast_buffers", True)
        self.__dict__.setdefault("forward_sync_buffers", old_broadcast)
        parameters, expect_sparse_gradient = self._build_params_for_reducer()
        # In debug mode, build a mapping of parameter index -> parameter.
        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)
        # Builds reducer.
        self._ddp_init_helper(
            parameters,
            expect_sparse_gradient,
            param_to_name_mapping,
            self.static_graph,
        )
        if self.static_graph:
            self.reducer._set_static_graph()
            if self.logger is None:
                raise AssertionError(
                    "self.logger must not be None when static_graph is True"
                )
            self.logger._set_static_graph()

    def _build_params_for_reducer(self):
        # Build tuple of (module, parameter) for all parameters that require grads.
        modules_and_parameters = [
            (module, parameter)
            for module_name, module in self.module.named_modules()
            for parameter in [
                param
                # Note that we access module.named_parameters instead of
                # parameters(module). parameters(module) is only needed in the
                # single-process multi device case, where it accesses replicated
                # parameters through _former_parameters.
                for param_name, param in module.named_parameters(recurse=False)
                if param.requires_grad
                and f"{module_name}.{param_name}" not in self.parameters_to_ignore
            ]
        ]

        # Deduplicate any parameters that might be shared across child modules.
        memo = set()
        modules_and_parameters = [
            # "p not in memo" is the deduplication check.
            # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
            (m, p)
            for m, p in modules_and_parameters
            if p not in memo and not memo.add(p)  # type: ignore[func-returns-value]
        ]

        # Build list of parameters.
        parameters = [parameter for _, parameter in modules_and_parameters]

        # Checks if a module will produce a sparse gradient.
        def produces_sparse_gradient(module):
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                return module.sparse
            return False

        # Build list of booleans indicating whether or not to expect sparse
        # gradients for the corresponding parameters.
        expect_sparse_gradient = [
            produces_sparse_gradient(module) for module, _ in modules_and_parameters
        ]

        self._assign_modules_buffers()

        return parameters, expect_sparse_gradient

    def _assign_modules_buffers(self):
        """
        Assign self.module.named_buffers to self.modules_buffers.

        Assigns module buffers to self.modules_buffers which are then used to
        broadcast across ranks when forward_sync_buffers=True. Note that this
        must be called every time buffers need to be synced because buffers can
        be reassigned by user module,
        see https://github.com/pytorch/pytorch/issues/63916.
        """
        # Collect buffers for modules, filtering out buffers that should be ignored.
        named_module_buffers = [
            (buffer, buffer_name)
            for buffer_name, buffer in self.module.named_buffers()
            if buffer_name not in self.parameters_to_ignore
        ]
        self.modules_buffers = [
            buffer for (buffer, buffer_name) in named_module_buffers
        ]
        # Dict[str, tensor] representing module buffers not ignored by DDP.
        self.named_module_buffers = {
            buffer_name: buffer for (buffer, buffer_name) in named_module_buffers
        }

    def _build_debug_param_to_name_mapping(self, parameters):
        param_to_param_index = {parameters[i]: i for i in range(len(parameters))}
        param_set = set(parameters)
        param_index_to_param_fqn = {}
        for module_name, module in self.module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fqn = f"{module_name}.{param_name}"
                # Bypass ignored parameters since those are not reduced by DDP
                # to begin with.
                if fqn not in self.parameters_to_ignore and param.requires_grad:
                    if param not in param_set:
                        self._log_and_throw(
                            ValueError,
                            f"Param with name {fqn} found in module parameters, but not DDP parameters."
                            " This indicates a bug in DDP, please report an issue to PyTorch.",
                        )
                    param_index = param_to_param_index[param]
                    param_index_to_param_fqn[param_index] = fqn

        # Ensure we covered all parameters
        if len(param_set) != len(param_index_to_param_fqn):
            self._log_and_throw(
                ValueError,
                (
                    "Expected param to name mapping to cover all parameters, but"
                    f" got conflicting lengths: {len(param_set)} vs "
                    f"{len(param_index_to_param_fqn)}. This indicates a bug in DDP"
                    ", please report an issue to PyTorch."
                ),
            )

        return param_index_to_param_fqn

    def _get_parameters(self, m, recurse=True):
        """Return a generator of module parameters."""

        def model_parameters(m):
            ps = (
                m._former_parameters.values()
                if hasattr(m, "_former_parameters")
                else m.parameters(recurse=False)
            )
            yield from ps

        for mod in m.modules() if recurse else [m]:
            yield from model_parameters(mod)

    def _check_default_group(self):
        pickle_not_supported = False
        try:
            if self.process_group != _get_default_group():
                pickle_not_supported = True
        except RuntimeError:
            pickle_not_supported = True

        if pickle_not_supported:
            self._log_and_throw(
                RuntimeError,
                "DDP Pickling/Unpickling are only supported "
                "when using DDP with the default process "
                "group. That is, when you have called "
                "init_process_group and have not passed "
                "process_group argument to DDP constructor",
            )

    @contextmanager
    def no_sync(self):
        r"""
        Context manager to disable gradient synchronizations across DDP processes.

        Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> ddp = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            >>>     for input in inputs:
            >>>         ddp(input).backward()  # no synchronization, accumulate grads
            >>> ddp(another_input).backward()  # synchronize grads

        .. warning::
            The forward pass should be included inside the context manager, or
            else gradients will still be synchronized.
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync

    @classmethod
    def _get_active_ddp_module(cls):
        """`TorchDynamo` requires DDP's status and module for cooperative optimization."""
        return cls._active_ddp_module

    # note, this ctxmgr function is marked 'skip' in torchdynamo, so dynamo only kicks in
    # for the 'module_to_run' underneath
    # see torch._dynamo/eval_frame.py TorchPatcher.patch for more details
    @contextmanager
    @torch._disable_dynamo(recursive=False)
    def _inside_ddp_forward(self):
        # Save and restore the previous _active_ddp_module to handle nested
        # DDP correctly (e.g., TorchRec wraps embeddings in an inner DDP inside
        # an outer DDP).  Without this, the inner DDP's exit would clear the
        # flag to None, causing DDPOptimizer to miss compiled regions that run
        # after the inner forward.
        old = DistributedDataParallel._active_ddp_module
        DistributedDataParallel._active_ddp_module = self
        try:
            yield
        finally:
            DistributedDataParallel._active_ddp_module = old

    def _run_ddp_forward(self, *inputs, **kwargs):
        if self._use_python_reducer:
            return self.module(*inputs, **kwargs)  # type: ignore[index]
        else:
            with self._inside_ddp_forward():
                return self.module(*inputs, **kwargs)  # type: ignore[index]

    def _clear_grad_buffer(self):
        # Making param.grad points to the grad buffers before backward is based on the
        # assumption that the grad accumulation is done in place in autograd engine,
        # for some edge cases, if the grad accumulation in autograd engine is not in
        # place, then the param.grad and grad buffers are detached.
        if self._delay_grad_buffer is not None:
            # We batch zero_grad for all params by resetting the whole grad
            # buffer when the grad of all params is set to None.
            all_param_grad_none = all(
                param.grad is None for param in self._delay_all_reduce_params
            )

            for index, param in enumerate(self._delay_all_reduce_params):
                if param.grad is None:
                    param.grad = self._delay_grad_views[index]
                    if not all_param_grad_none:
                        param.grad.zero_()

            if all_param_grad_none:
                self._delay_grad_buffer.zero_()

    def _lazy_init(self):
        # Initialization for DDP that occurs after construction, but lazily
        # before the first forward pass.
        self._setup_in_backward_optimizers()
        self._lazy_init_ran = True

    def _pre_forward(self, *inputs, **kwargs):
        if self._use_python_reducer:
            return inputs, kwargs

        if not self._lazy_init_ran and not torch.compiler.is_compiling():
            self._lazy_init()

        if self._delay_all_reduce_all_params:
            return inputs, kwargs

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            if self.logger is None:
                raise AssertionError("self.logger must not be None")
            self.logger.set_runtime_stats_and_log()
            self.reducer.prepare_for_forward()

        # Notify the join context that this process has not joined, if
        # needed
