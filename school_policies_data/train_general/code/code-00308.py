                                    the only argument.

        Returns:
            A new ``Future`` object that holds the return value of the
            ``callback`` and will be marked as completed when the given
            ``callback`` finishes.

        .. note:: Note that if the callback function throws, either
            through the original future being completed with an exception and
            calling ``fut.wait()``, or through other code in the callback, the
            future returned by ``then`` will be marked appropriately with the
            encountered error. However, if this callback later completes
            additional futures, those futures are not marked as completed with
            an error and the user is responsible for handling completion/waiting
            on those futures independently.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> def callback(fut):
            ...     print(f"RPC return value is {fut.wait()}.")
            >>> fut = torch.futures.Future()
            >>> # The inserted callback will print the return value when
            >>> # receiving the response from "worker1"
            >>> cb_fut = fut.then(callback)
            >>> chain_cb_fut = cb_fut.then(
            ...     lambda x : print(f"Chained cb done. {x.wait()}")
            ... )
            >>> fut.set_result(5)
            RPC return value is 5.
            Chained cb done. None
        """
        return cast(Future[S], super().then(callback))

    def add_done_callback(self, callback: Callable[[Future[T]], None]) -> None:
        r"""
        Append the given callback function to this ``Future``, which will be run
        when the ``Future`` is completed.  Multiple callbacks can be added to
        the same ``Future``, but the order in which they will be executed cannot
        be guaranteed. The callback must take one argument, which is the
        reference to this ``Future``. The callback function can use the
        :meth:`value` method to get the value. Note that if this ``Future`` is
        already completed, the given callback will be run inline.

        We recommend that you use the :meth:`then` method as it provides a way
        to synchronize after your callback has completed. ``add_done_callback``
        can be cheaper if your callback does not return anything. But both
        :meth:`then` and ``add_done_callback`` use the same callback
        registration API under the hood.

        With respect to GPU tensors, this method behaves in the same way as
        :meth:`then`.

        Args:
            callback(``Future``): a ``Callable`` that takes in one argument,
                which is the reference to this ``Future``.

        .. note:: Note that if the callback function throws, either
            through the original future being completed with an exception and
            calling ``fut.wait()``, or through other code in the callback,
            error handling must be carefully taken care of. For example, if
            this callback later completes additional futures, those futures are
            not marked as completed with an error and the user is responsible
            for handling completion/waiting on those futures independently.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> def callback(fut):
            ...     print("This will run after the future has finished.")
            ...     print(fut.wait())
            >>> fut = torch.futures.Future()
            >>> fut.add_done_callback(callback)
            >>> fut.set_result(5)
            This will run after the future has finished.
            5
        """
        super().add_done_callback(callback)

    def set_result(self, result: T) -> None:
        r"""
        Set the result for this ``Future``, which will mark this ``Future`` as
        completed and trigger all attached callbacks. Note that a ``Future``
        cannot be marked completed twice.

        If the result contains tensors that reside on GPUs, this method can be
        called even if the asynchronous kernels that are populating those
        tensors haven't yet completed running on the device, provided that the
        streams on which those kernels were enqueued are set as the current ones
        when this method is called. Put simply, it's safe to call this method
        immediately after launching those kernels, without any additional
        synchronization, as long as one doesn't change streams in between. This
        method will record events on all the relevant current streams and will
        use them to ensure proper scheduling for all the consumers of this
        ``Future``.

        Args:
            result (object): the result object of this ``Future``.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> import threading
            >>> import time
            >>> def slow_set_future(fut, value):
            ...     time.sleep(0.5)
            ...     fut.set_result(value)
            >>> fut = torch.futures.Future()
            >>> t = threading.Thread(
            ...     target=slow_set_future,
            ...     args=(fut, torch.ones(2) * 3)
            ... )
            >>> t.start()
            >>> print(fut.wait())
            tensor([3., 3.])
            >>> t.join()
        """
        super().set_result(result)

    def set_exception(self, result: T) -> None:
        r"""
        Set an exception for this ``Future``, which will mark this ``Future`` as
        completed with an error and trigger all attached callbacks. Note that
        when calling wait()/value() on this ``Future``, the exception set here
        will be raised inline.

        Args:
            result (BaseException): the exception for this ``Future``.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> fut = torch.futures.Future()
            >>> fut.set_exception(ValueError("foo"))
            >>> fut.wait()
            Traceback (most recent call last):
            ...
            ValueError: foo
        """
        if not isinstance(result, Exception):
            raise AssertionError(
                f"{result} is of type {type(result)}, not an Exception."
            )

        def raise_error(fut_result):
            raise fut_result

        super()._set_unwrap_func(raise_error)
        self.set_result(result)  # type: ignore[arg-type]


def collect_all(futures: list[Future]) -> Future[list[Future]]:
    r"""
    Collects the provided :class:`~torch.futures.Future` objects into a single
    combined :class:`~torch.futures.Future` that is completed when all of the
    sub-futures are completed.

    Args:
        futures (list): a list of :class:`~torch.futures.Future` objects.

    Returns:
        Returns a :class:`~torch.futures.Future` object to a list of the passed
        in Futures.

    Example::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
        >>> fut0 = torch.futures.Future()
        >>> fut1 = torch.futures.Future()
        >>> fut = torch.futures.collect_all([fut0, fut1])
        >>> fut0.set_result(0)
        >>> fut1.set_result(1)
        >>> fut_list = fut.wait()
        >>> print(f"fut0 result = {fut_list[0].wait()}")
        fut0 result = 0
        >>> print(f"fut1 result = {fut_list[1].wait()}")
        fut1 result = 1
    """
    return cast(
        Future[list[Future]],
        torch._C._collect_all(cast(list[torch._C.Future], futures)),
    )


def wait_all(futures: list[Future]) -> list:
    r"""
    Waits for all provided futures to be complete, and returns
    the list of completed values. If any of the futures encounters an error,
    the method will exit early and report the error not waiting for other
    futures to complete.

    Args:
        futures (list): a list of :class:`~torch.futures.Future` object.

    Returns:
        A list of the completed :class:`~torch.futures.Future` results. This
        method will throw an error if ``wait`` on any
        :class:`~torch.futures.Future` throws.
    """
    return [
        fut.wait()
        for fut in torch._C._collect_all(cast(list[torch._C.Future], futures)).wait()
    ]
<|endoftext|>from torch.masked._ops import (
    _canonical_dim,
    _combine_input_and_mask,
    _generate_docstring,
    _input_mask,
    _output_mask,
    _reduction_identity,
    _where,
    amax,
    amin,
    argmax,
    argmin,
    cumprod,
    cumsum,
    log_softmax,
    logaddexp,
    logsumexp,
    mean,
    median,
    norm,
    normalize,
    prod,
    softmax,
    softmin,
    std,
    sum,
    var,
)
from torch.masked.maskedtensor.core import is_masked_tensor, MaskedTensor
from torch.masked.maskedtensor.creation import as_masked_tensor, masked_tensor


__all__ = [
    "amax",
    "amin",
    "argmax",
    "argmin",
    "as_masked_tensor",
    "cumprod",
    "cumsum",
    "is_masked_tensor",
    "log_softmax",
    "logaddexp",
    "logsumexp",
    "masked_tensor",
    "MaskedTensor",
    "mean",
    "median",
    "norm",
    "normalize",
    "prod",
    "softmax",
    "softmin",
    "std",
    "sum",
    "var",
]
<|endoftext|># mypy: allow-untyped-defs
import warnings
from collections.abc import Callable
from typing import Any, TYPE_CHECKING, TypeAlias, TypeVar
from typing_extensions import ParamSpec

import torch
from torch import sym_float, Tensor
from torch._prims_common import corresponding_real_dtype
from torch.masked import _docs
from torch.masked.maskedtensor.core import is_masked_tensor, MaskedTensor
from torch.masked.maskedtensor.creation import as_masked_tensor


if TYPE_CHECKING:
    from torch._prims_common import DimsType
    from torch.types import _dtype as DType

    DimOrDims: TypeAlias = DimsType | None
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int
    DimOrDims = tuple[int, ...] | None


__all__: list[str] = []

_T = TypeVar("_T")
_P = ParamSpec("_P")

# All masked reduction/normalization operations have the same
# signatures. Here we introduce docstring templates that are applied
# to docstrings of reduction/normalization functions via
# _apply_docstring_templates decorator.


def _apply_docstring_templates(func: Callable[_P, _T]) -> Callable[_P, _T]:
    """Decorator that applies docstring templates to function docstring
    and returns the function instance.
    """

    doc_string = getattr(_docs, f"{func.__name__}_docstring", None)
    if doc_string is None:
        warnings.warn(
            f"No documentation string available for {func.__name__}."
            " PyTorch team should run `python tools/update_masked_docs.py`"
            " to generate the missing docstrings.",
            stacklevel=2,
        )
    else:
        func.__doc__ = doc_string

    # Expose function as public symbol
    __all__.append(func.__name__)

    return func


def _generate_docstring(func):
    """A utility function called from tools/update_masked_docs.py
    script to update the module torch.masked._docs.py
    """
    docstring_templates = dict(
        reduction_signature="""\
{function_name}(input, {operation_args}, *, {operation_kwargs}) -> Tensor""",
        reduction_descr="""\
Returns {operation name} of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.""",
        reduction_args="""\
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in {operation name} computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of {operation name} operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    {args_declarations}

Keyword args:
    {kwargs_declarations}""",
        reduction_example="""\
Example::

    >>> input = {example_input}
    >>> input
    {indent_example_input}
    >>> mask = {example_mask}
    >>> mask
    {indent_example_mask}
    >>> {full_function_name}(input, {example_args}, mask=mask)
    {indent_example_output}
""",
        reduction_identity="""\
The identity value of {operation name} operation, which is used to start the reduction, is ``{identity_int32}``.""",
        reduction_identity_dtype="""\
The identity value of {operation name} operation, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are ``{identity_float32}``, ``{identity_uint8}``, and ``{identity_int32}``, respectively.""",
        normalization_signature="""\
{function_name}(input, {operation_args}, *, {operation_kwargs}) -> Tensor""",
        normalization_descr="""\
Returns {operation name} of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

{definition}""",
        normalization_args="""\
The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
{operation name} computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the {operation name} output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    {args_declarations}

Keyword args:
    {kwargs_declarations}""",
        normalization_example="""\
Example::

    >>> input = {example_input}
    >>> input
    {indent_example_input}
    >>> mask = {example_mask}
    >>> mask
    {indent_example_mask}
    >>> {full_function_name}(input, {example_args}, mask=mask)
    {indent_example_output}
""",
    )

    args_and_kwargs = {
        # argument name suffixes separated by double underscore will
        # be removed in the final documentation string.
        "sum": (("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        "prod": (("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        "cumsum": (("dim__as_int",), ("dtype=None", "mask=None")),
        "cumprod": (("dim__as_int",), ("dtype=None", "mask=None")),
        "amin": (("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        "amax": (("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        "argmin": (("dim__as_int",), ("keepdim=False", "dtype=None", "mask=None")),
        "argmax": (("dim__as_int",), ("keepdim=False", "dtype=None", "mask=None")),
        "mean": (("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        "median": (("dim__as_int",), ("keepdim=False", "dtype=None", "mask=None")),
        "norm": (
            (
                "ord",
                "dim",
            ),
            ("keepdim=False", "dtype=None", "mask=None"),
        ),
        "var": (("dim", "unbiased"), ("keepdim=False", "dtype=None", "mask=None")),
        "std": (("dim", "unbiased"), ("keepdim=False", "dtype=None", "mask=None")),
        "logsumexp": (("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        "softmax": (("dim__as_int",), ("dtype=None", "mask=None")),
        "log_softmax": (("dim__as_int",), ("dtype=None", "mask=None")),
        "softmin": (("dim__as_int",), ("dtype=None", "mask=None")),
        "normalize": (
            (
                "ord__required",
                "dim__as_int",
            ),
            ("eps=1e-12", "dtype=None", "mask=None"),
        ),
    }

    argument_declarations = {
        "dim": """\
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
    Default: None that is equivalent to ``tuple(range(input.ndim))``.""",
        "dim__as_int": """\
    dim (int): the dimension along which {operation name} is computed.""",
        "ord": """\
    ord (int, float, optional): the order of vector norm. Default: 2.
    See :func:`torch.linalg.vector_norm` for a list of supported norms.""",
        "ord__required": """\
    ord (int, float): the order of vector norm. Default: 2.
    See :func:`torch.linalg.vector_norm` for a list of supported norms.""",
        "unbiased": """\
    unbiased (bool): when True, use Bessel's correction, otherwise, compute
    the uncorrected sample variance.""",
        "eps": """\
    eps (float, optional): small value to avoid division by zero. Default: {default}.""",
        "keepdim": """\
    keepdim (bool, optional): whether the output tensor has
    :attr:`dim` retained or not. Default: {default}.""",
        "dtype": """\
    dtype (:class:`torch.dtype`, optional): the desired data type
    of returned tensor.  If specified, the input tensor is
    casted to :attr:`dtype` before the operation is
    performed. Default: {default}.""",
        "mask": """\
    mask (:class:`torch.Tensor`, optional): the boolean tensor
    containing the binary mask of validity of input tensor
    elements.
    Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.""",
    }

    definitions = {
        "softmax": """\
    Let ``x`` be a sequence of unmasked elements of one-dimensional slice
    of the :attr:`input` tensor. Softmax of i-th element in ``x`` is
    defined as ``exp(x[i])/sum(exp(x))``.""",
        "log_softmax": """\
    Let ``x`` be a sequence of unmasked elements of one-dimensional slice
    of the :attr:`input` tensor. LogSoftmax of i-th element in ``x`` is
    defined as ``log(exp(x[i])/sum(exp(x)))``.""",
        "softmin": """\
    Let ``x`` be a sequence of unmasked elements of one-dimensional slice
    of the :attr:`input` tensor. Softmin of i-th element in ``x`` is
    defined as ``exp(-x[i])/sum(exp(-x))``.""",
        "normalize": """\
    Let ``x`` be a sequence of unmasked elements of one-dimensional slice
    of the :attr:`input` tensor. Normalize of i-th element in ``x`` is
    defined as ``x[i]/max(norm(x, p), eps)``.""",
        "cumsum": """\
    Let ``x`` be a sequence of unmasked elements of one-dimensional slice
    of the :attr:`input` tensor. Cumsum of i-th element in ``x`` is
    defined as ``sum(x[:i])``.""",
        "cumprod": """\
    Let ``x`` be a sequence of unmasked elements of one-dimensional slice
    of the :attr:`input` tensor. Cumsum of i-th element in ``x`` is
    defined as ``prod(x[:i])``.""",
    }

    reduction_names = {
        "sum": "sum",
        "prod": "product",
        "amax": "maximum",
        "amin": "minimum",
        "argmax": "argmax",
        "argmin": "argmin",
        "mean": "mean",
        "median": "median",
        "norm": "norm",
        "var": "variance",
        "std": "standard_deviation",
        "logsumexp": "logsumexp",
    }

    normalization_names = {
        "softmax": "softmax",
        "log_softmax": "log_softmax",
        "softmin": "softmin",
        "normalize": "normalize",
        "cumsum": "cumulative_sum",
        "cumprod": "cumulative_prod",
    }

    operation_names = {}
    operation_names.update(reduction_names)
    operation_names.update(normalization_names)

    # Default example data:
    example_dim = 1
    example_input = torch.tensor([[-3, -2, -1], [0, 1, 2]])
    example_mask = torch.tensor([[True, False, True], [False, False, False]])
    example_args: tuple[Any, ...]
    if func.__name__ in {"norm", "normalize"}:
        example_args = (2.0, example_dim)
        example_input = example_input.to(dtype=torch.float32)
    elif func.__name__ in {"var", "std"}:
        example_args = (example_dim, False)
    elif func.__name__ == "median":
        example_args = (example_dim,)
        example_input = example_input.to(dtype=torch.float32)
    else:
        example_args = (example_dim,)

    operation_args: tuple[str, ...]
    operation_kwargs: tuple[str, ...]
    operation_args, operation_kwargs = args_and_kwargs[func.__name__]
    arg_declarations = [
        "\n    ".join(
            argument_declarations.get(a, f"{a.split('__', 1)[0]}: TBD.").splitlines()
        )
        for a in operation_args
    ]
    kwarg_declarations = [
        "\n    ".join(
            argument_declarations.get(
                a.split("=", 1)[0], f"{a.split('__', 1)[0]}: TBD."
            )
            .format(default=a.split("=", 1)[1])
            .splitlines()
        )
        for a in operation_kwargs
    ]

    if func.__name__ in reduction_names:
        op_kind = "reduction"
        doc_sections = ["signature", "descr", "identity", "args", "example"]
    elif func.__name__ in normalization_names:
        op_kind = "normalization"
        doc_sections = ["signature", "descr", "args", "example"]
        example_input = example_input.to(dtype=torch.float32)
    else:
        # add function name to operation names dictionaries
        raise AssertionError(f"unknown function {func.__name__}")
    example_output = func(example_input, *example_args, mask=example_mask)

    template_data = {
        "function_name": func.__name__,
        "full_function_name": func.__module__ + "." + func.__name__,
        "operation name": operation_names[func.__name__],
        "operation_args": ", ".join(a.split("__", 1)[0] for a in operation_args),
        "operation_kwargs": ", ".join(a.split("__", 1)[0] for a in operation_kwargs),
        # one-line representation of a tensor:
        "example_input": " ".join(str(example_input).split()),
        "example_args": ", ".join(map(str, example_args)),
        "example_mask": " ".join(str(example_mask).split()),
        # multi-line representation of a tensor with indent
        "indent_example_input": ("\n    ").join(str(example_input).splitlines()),
        "indent_example_mask": ("\n    ").join(str(example_mask).splitlines()),
        "indent_example_output": ("\n    ").join(str(example_output).splitlines()),
    }

    if func.__name__ in reduction_names:
        template_data.update(
            identity_uint8=_reduction_identity(
                func.__name__, torch.tensor(0, dtype=torch.uint8)
            ),
            identity_int32=_reduction_identity(
                func.__name__, torch.tensor(0, dtype=torch.int32)
            ),
            identity_float32=_reduction_identity(
                func.__name__, torch.tensor(0, dtype=torch.float32)
            ),
        )
        if func.__name__ == "norm":
            template_data.update(
                identity_ord_ninf=_reduction_identity(
                    func.__name__, torch.tensor(0, dtype=torch.float32), float("-inf")
                )
            )
    elif func.__name__ in normalization_names:
        template_data.update(definition=definitions[func.__name__])
    else:
        # add function name to operation names dictionaries
        raise AssertionError(f"unknown function {func.__name__}")
    template_data.update(
        args_declarations=("\n    ".join(arg_declarations)).format_map(template_data)
    )
    template_data.update(
        kwargs_declarations=("\n    ".join(kwarg_declarations)).format_map(
            template_data
        )
    )

    # Apply function name info to docstring templates:
    templates = {
        k: v.format_map(template_data)
        for k, v in docstring_templates.items()
        if k.startswith(op_kind)
    }
    templates.update(
        (k, v.format_map(template_data) if isinstance(v, str) else v)
        for k, v in template_data.items()
    )

    # Apply docstring templates to function docstring:
    if func.__doc__ is None:
        doc_template = "\n\n".join([f"{{{op_kind}_{sec}}}" for sec in doc_sections])
    else:
        doc_template = func.__doc__
    return doc_template.format_map(templates)


def _reduction_identity(op_name: str, input: Tensor, *args):
    """Return identity value as scalar tensor of a reduction operation on
    given input, or None, if the identity value cannot be uniquely
    defined for the given input.

    The identity value of the operation is defined as the initial
    value to reduction operation that has a property ``op(op_identity,
    value) == value`` for any value in the domain of the operation.
    Or put it another way, including or excluding the identity value in
    a list of operands will not change the reduction result.

    See https://github.com/pytorch/rfcs/pull/27 for more information.

    """
    dtype: DType = input.dtype
    device = input.device
    op_name = op_name.rsplit(".", 1)[-1]  # lstrip module name when present
    if op_name in {"sum", "cumsum"}:
        return torch.tensor(0, dtype=dtype, device=device)
    elif op_name in {"prod", "cumprod"}:
        return torch.tensor(1, dtype=dtype, device=device)
    elif op_name in {"amax", "argmax", "logaddexp"}:
        if torch.is_floating_point(input):
            return torch.tensor(-torch.inf, dtype=dtype, device=device)
        elif torch.is_signed(input) or dtype == torch.uint8:
            return torch.tensor(torch.iinfo(dtype).min, dtype=dtype, device=device)
    elif op_name == "logsumexp":
        if torch.is_floating_point(input):
            return torch.tensor(-torch.inf, dtype=dtype, device=device)
        elif torch.is_complex(input):
            return torch.tensor(-torch.inf + 0j, dtype=dtype, device=device)
        elif torch.is_signed(input) or dtype == torch.uint8:
            return torch.tensor(torch.iinfo(dtype).min, dtype=dtype, device=device)
    elif op_name in {"amin", "argmin"}:
        if torch.is_floating_point(input):
            return torch.tensor(torch.inf, dtype=dtype, device=device)
        elif torch.is_signed(input) or dtype == torch.uint8:
            return torch.tensor(torch.iinfo(dtype).max, dtype=dtype, device=device)
    elif op_name == "mean":
        # Strictly speaking, the identity value of the mean operation
        # is the mean of the input. Since the mean value depends on
        # the dim argument and it may be a non-scalar tensor, we
        # consider the identity value of the mean operation ambiguous.
        # Moreover, the mean value of empty input is undefined.
        return None
    elif op_name == "norm":
        ord = args[0] if args else 2
        if ord == float("-inf"):
            if not torch.is_floating_point(input):
                raise AssertionError(f"input must be floating point, got {input.dtype}")
            return torch.tensor(torch.inf, dtype=dtype, device=device)
        return torch.tensor(0, dtype=dtype, device=device)
    elif op_name == "median":
        # We use NaN for now because the implementation is currently using torch.nanmedian
        # and NaN is the identity for that function since it gets ignored
        dtype = input.dtype if torch.is_floating_point(input) else torch.float
        return torch.tensor(torch.nan, dtype=dtype, device=device)
    elif op_name in {"var", "std"}:
        return None
    raise NotImplementedError(f"identity of {op_name} on {dtype} input")


def _canonical_dim(dim: DimOrDims, ndim: int) -> tuple[int, ...]:
    """Return dim argument as a tuple of sorted dim values."""
    dims: list[int] = []
    if dim == ():
        # Currently, `dim=()` in reductions operations means "reduce
        # over all dimensions" while in future, it will read "no
        # reduce". See https://github.com/pytorch/pytorch/issues/29137
        # When gh-29137 is resolved, this if-block must be deleted.
        dim = None
    if dim is None:
        return tuple(range(ndim))
    ndim = max(ndim, 1)
    dim_ = (dim,) if isinstance(dim, (int, torch.SymInt)) else dim
    for d in dim_:
        if d in dims:
            raise RuntimeError(f"dim={d} appears multiple times in the list of dims")
        if d >= ndim or d < -ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})"
            )
        # pyrefly: ignore [bad-argument-type]
        dims.append(d % ndim)
    return tuple(sorted(dims))


def _sparse_coo_flatten_indices(indices: Tensor, shape: tuple):
    # Flatted N-D indices to 1-D indices
    flat_indices = indices.new_zeros(indices.size(1))
    for d, sz in enumerate(shape):
        flat_indices.mul_(sz)
        flat_indices.add_(indices[d])
    return flat_indices


def _any(input: Tensor, dim: tuple, keepdim: bool):
    # Support torch.any with tuple dim argument.
    # Workaround of https://github.com/pytorch/pytorch/issues/56586
    r = input
    for d in reversed(dim):
        r = r.any(dim=d, keepdim=keepdim)
    return r


def _sparse_coo_where(mask: Tensor, input: Tensor, fill_value: Tensor) -> Tensor:
    """Sparse variant of torch.where. Supports sparse COO and hybrid sparse COO tensors.

    _sparse_coo_where implements the following invariant:

      _sparse_coo_where(mask, input, fill_value).to_dense(fill_value) ==
        torch.where(mask.to_dense(), input.to_dense(), torch.full(input.shape, fill_value))

    where `a == b` means `assertEqual(a, b)`, mask is boolean sparse
    tensor, and `to_dense(fill_value)` is like `to_dense()` except
    that the unspecified elements are mapped to `fill_value` rather
    than to `0`.

    Returns a sparse COO tensor with the following features:

    - all specified elements correspond to masked-in elements that
      have the values of the input tensor. If there exists a masked-in
      element (as specified by mask) that is not specified in the
      input, in the result tensor, the corresponding element has value
      0. In the dense part of the sparse tensor, the masked-out
      elements are replaced with fill_value.

    - all unspecified elements correspond to masked-out elements.
    """

    if input.layout != torch.sparse_coo:
        raise AssertionError(f"input.layout must be sparse_coo, got {input.layout}")
    if mask.layout != input.layout:
        raise AssertionError(f"mask.layout must match input.layout, got {mask.layout}")
    if mask.shape != input.shape:
        raise AssertionError(
            f"mask.shape must match input.shape: {mask.shape} vs {input.shape}"
        )
    if mask.dense_dim() != input.dense_dim():
        # TODO: eliminate this restriction
        raise AssertionError(
            f"mask.dense_dim() must match input.dense_dim(): "
            f"{mask.dense_dim()} vs {input.dense_dim()}"
        )

    input = input.coalesce()

    # For set operations on sparse tensor indices, we'll convert
    # multi-dimensional indices to 1-D indices for efficiency.
    input_flat_indices = _sparse_coo_flatten_indices(
        input.indices(), input.shape[: input.sparse_dim()]
    )
    mask_flat_indices = _sparse_coo_flatten_indices(
        mask.indices(), mask.shape[: mask.sparse_dim()]
    )

    # the set of mask flat indices that define masked-in elements:
    if mask.dense_dim() > 0:
        mask_values = _any(
            mask.values(), tuple(range(1, input.sparse_dim() + 1)), False
        )
    else:
        mask_values = mask.values()
    maskin_flat_indices = mask_flat_indices[mask_values.nonzero()[:, 0]]

    def intersection(i1, i2):
        union, counts = torch.cat([i1, i2]).unique(return_counts=True)
        return union, torch.where(counts.gt(1))

    def minus(i1, i2):
        union, counts = torch.cat([i1, i2]).unique(return_counts=True)
        return intersection(union[torch.where(counts.eq(1))], i1)

    def _apply(a):
        obj, w = a
        return obj[w]

    # the set of input flat indices of specified and masked-in elements:
    maskin_input_flat_indices = _apply(
        intersection(maskin_flat_indices, input_flat_indices)
    )
    _, w = intersection(input_flat_indices, maskin_input_flat_indices)

    # the indices and values of masked-in elements
    where_input_indices = input.indices()[(slice(None),) + w]
    where_input_values = input.values()[w]

    if mask.dense_dim() > 0:
        # apply mask to the dense part of the input values:
        _, w1 = intersection(mask_flat_indices, maskin_input_flat_indices)
        where_mask_values = mask.values()[w1]
        where_input_values = torch.where(
            where_mask_values, where_input_values, fill_value
        )

    # the set of flat indices of unspecified input and masked-in elements:
    maskin_zero_flat_indices = _apply(
        minus(maskin_flat_indices, maskin_input_flat_indices)
    )

    # the indices of masked-in zero elements
    _, w = intersection(mask_flat_indices, maskin_zero_flat_indices)
    where_zero_indices = mask.indices()[(slice(None),) + w]

    # construct result
    n = where_zero_indices.size(1)
    if n == 0:
        # the input is coalesced, hence input_flat_indices are ordered
        # and the result is guaranteed to be coalesced:
        result = torch.sparse_coo_tensor(
            where_input_indices, where_input_values, input.shape
        )
        return result._coalesced_(True)

    where_indices = torch.cat([where_input_indices, where_zero_indices], dim=1)
    where_values = torch.cat(
        [
            where_input_values,
            where_input_values.new_zeros((n,) + where_input_values.shape[1:]),
        ]
    )
    result = torch.sparse_coo_tensor(where_indices, where_values, input.shape)

    # appending zero elements leads to uncoalesced sparse tensor
    return result.coalesce()


def _sparse_coo_scatter_reduction_helper(
    op,
    mask_input: Tensor,
    dims: tuple[int, ...],
    keepdim: bool,
    dtype: DType | None = None,
) -> Tensor:
    reduce = op.__name__
    valid_reductions = ["sum", "prod", "amax", "amin"]
    if reduce not in valid_reductions:
        raise ValueError(
            f"op must be one of {' '.join(valid_reductions)}, but got {reduce} instead"
        )

    output_dtype = dtype
    values, indices = mask_input._values(), mask_input._indices()
    input_dims = mask_input.dim()
    num_sparse_dims = mask_input.sparse_dim()
    reduced_sparse_dims = []
    retained_sparse_dims = []
    reduced_dense_dims = []

    # promote dtype if specified
    if values.dtype != output_dtype:
        values = values.to(output_dtype)

    if keepdim:
        output_shape = tuple(
            1 if i in dims else si for (i, si) in enumerate(mask_input.shape)
        )
    else:
        output_shape = tuple(
            si for (i, si) in enumerate(mask_input.shape) if i not in dims
        )

    for d in dims:
        if d >= input_dims:
            continue

        if d < num_sparse_dims:
            reduced_sparse_dims.append(d)
        else:
            reduced_dense_dims.append(d + 1 - num_sparse_dims)

    # Reduce dense dimensions
    if len(reduced_dense_dims) > 0:
        if reduce == "sum":
            new_values = values
            new_values = op(new_values, dim=reduced_dense_dims, keepdim=bool(keepdim))
        else:
            # FIXME: Implement reductions for dense dimensions for ops with non-zero reduction identities
            return NotImplemented
    else:
        new_values = values.clone()

    # Reduce sparse dimensions
    if len(reduced_sparse_dims) == num_sparse_dims:
        if reduce in {"amax", "amin"} and new_values.size(0) == 0:
            # IndexError: amax(): Expected reduction dim 0 to have non-zero size.
            # sum()/prod() return the reduction identity when dim has size 0 but amax()/amin() do not
            # See https://github.com/pytorch/pytorch/issues/61901
            new_values = _reduction_identity(reduce, new_values)
        else:
            new_values = op(new_values, dim=0)
        if keepdim:
            for _ in range(num_sparse_dims):
                new_values = new_values.unsqueeze(0)
        return new_values.to(dtype=output_dtype).to_sparse()
    else:
        new_indices = indices.clone()
        if keepdim:
            # zero out reduced sparse dimensions if keepdim = True
            # ensures that the call to torch.unique folds duplicated indices together while preserving the dimension
            new_indices[reduced_sparse_dims, :] = 0
        else:
            # remove reduced sparse dimensions if keepdim = False
            if len(reduced_sparse_dims) > 0:
                retained_sparse_dims = [
                    i
                    for i in range(num_sparse_dims)
                    if i not in set(reduced_sparse_dims)
                ]
                new_indices = new_indices.index_select(
                    0, torch.tensor(retained_sparse_dims).to(mask_input.device)
                )

    # Use scatter_reduce to reduce items in the new_values tensor that correspond to the same indices in new_indices
    if new_indices.numel() > 0:
        # lexsort indices and get index tensor for scatter reduction
        new_indices, inverse_indices = torch.unique(
            new_indices, return_inverse=True, dim=1
        )
        out_shape = list(new_values.shape)
        out_shape[0] = new_indices.shape[1]
        for _ in range(new_values.ndim - 1):
            inverse_indices = inverse_indices.unsqueeze(-1)
        scatter_indices = inverse_indices.expand(new_values.shape)
        # FIXME: temporary workaround for issue with bfloat16/float16 remove when acctype is implemented for scatter_reduce
        if output_dtype in {torch.bfloat16, torch.float16}:
            new_values = new_values.to(torch.float)
            out = new_values.new_empty(out_shape)
            new_values = out.scatter_reduce_(
                0, scatter_indices, new_values, reduce=reduce, include_self=False
            )
            new_values = new_values.to(dtype=output_dtype)
        else:
            out = new_values.new_empty(out_shape)
            new_values = out.scatter_reduce_(
                0, scatter_indices, new_values, reduce=reduce, include_self=False
            )

    return torch.sparse_coo_tensor(
        new_indices,
        new_values,
        output_shape,
        dtype=output_dtype,
        device=mask_input.device,
    )


def _sparse_csr_segment_reduction_helper(
    op,
    mask_input: Tensor,
    dims: tuple[int, ...],
    keepdim: bool,
    dtype: DType | None = None,
) -> Tensor:
    # Currently, while sparse CSR is always 2D with no dense dimensions keepdim must be True
    # FIXME: when dense dimensions are implemented for CSR tensors
    if not keepdim:
        raise AssertionError(
            "reduction operations on CSR tensors with keepdim=False is unsupported"
        )
    reduce = op.__name__
    valid_reductions = ["sum", "prod", "mean", "amax", "amin"]
    if reduce not in valid_reductions:
        raise ValueError(
            f"op must be one of {' '.join(valid_reductions)}, but got {reduce} instead"
        )
    device = mask_input.device
    output_dtype = dtype
    values, crow_indices, col_indices = (
        mask_input.values(),
        mask_input.crow_indices(),
        mask_input.col_indices(),
    )

    # promote dtype if specified
    if values.dtype != output_dtype:
        values = values.to(output_dtype)

    if len(dims) == 0:
        return mask_input
    if len(dims) == 1:
        if dims[0] == 0:
            new_col_indices, scatter_indices = torch.unique(
                col_indices, return_inverse=True
            )
            new_nnz = new_col_indices.shape[0]
            new_crow_indices = torch.tensor([0, new_nnz])
            new_values = values.new_empty(new_col_indices.shape)
            new_values.scatter_reduce_(
                0, scatter_indices, values, reduce, include_self=False
            )
            new_shape = [1, mask_input.size(1)]
        else:
            if dims[0] != 1:
                raise AssertionError(
                    "Sparse CSR tensors are 2D and only support reduction along dim 0 or 1."
                )
            # all intervals new_crow_indices[i] - new_crow_indices[i-1] are 1
            # except for where crow_indices[i] == crow_indices[i-1] where the interval remains as 0
            new_crow_indices = torch.cat(
                (
                    crow_indices.new_zeros(1),
                    torch.cumsum(torch.diff(crow_indices) != 0, 0),
                ),
                0,
            )
            new_nnz = new_crow_indices[-1]
            new_col_indices = col_indices.new_zeros(new_nnz)  # type: ignore[call-overload]
            new_values = torch._segment_reduce(values, reduce, offsets=crow_indices)  # type: ignore[attr-defined]
            new_shape = [mask_input.size(0), 1]
    else:
        if len(dims) != 2:
            raise AssertionError(f"expected len(dims) == 2, got {len(dims)}")
        nnz = min(1, values.numel())
        if nnz == 1:
            op_kwargs = {"keepdim": True, "dtype": output_dtype}
            # amax and amin do not support dtype kwarg
            if reduce in ["amax", "amin"]:
                del op_kwargs["dtype"]
            new_values = op(values, 0, **op_kwargs)
        else:
            new_values = torch.empty(0, dtype=output_dtype)
        new_col_indices = col_indices.new_zeros(nnz)
        new_crow_indices = torch.tensor([0, nnz])
        new_shape = [1, nnz]

    return torch.sparse_csr_tensor(
        new_crow_indices,
        new_col_indices,
        new_values,
        new_shape,
        dtype=output_dtype,
        device=device,
    )


def _sparse_csr_where(mask: Tensor, input: Tensor, fill_value: Tensor) -> Tensor:
    """Sparse variant of torch.where. Supports sparse CSR tensors."""
    # TODO: implement sparse CSR specific where operator for efficiency
    return _sparse_coo_where(
        mask.to_sparse_coo(), input.to_sparse_coo(), fill_value
    ).to_sparse_csr()


def _where(mask: Tensor, input: Tensor, fill_value: Tensor) -> Tensor:
    """torch.where with sparse inputs support.

    _where implements the following invariant:

      _where(mask, input, fill_value).to_dense(fill_value) ==
        torch.where(mask.to_dense(), input.to_dense(), torch.full(input.shape, fill_value))

    where `a == b` means `assertEqual(a, b)`, mask is boolean sparse
    tensor, and `to_dense(fill_value)` is like `to_dense()` except
    that the unspecified elements are mapped to `fill_value` rather
    than to `0`.

    Returns a sparse tensor with the following features:

    - all specified elements correspond to masked-in elements that
      have the values of the input tensor. If there exists a masked-in
      element (as specified by mask) that is not specified in the
      input, in the result tensor, the corresponding element has value
      0. In the dense part of the sparse tensor, the masked-out
      elements are replaced with fill_value.

    - all unspecified elements correspond to masked-out elements.
    """
    if mask.layout == torch.strided:
        return torch.where(mask, input, fill_value)
    elif mask.layout == torch.sparse_coo:
        return _sparse_coo_where(mask, input, fill_value)
    elif mask.layout == torch.sparse_csr:
        return _sparse_csr_where(mask, input, fill_value)
    else:
        raise ValueError(
            f"_where expects strided or sparse COO or sparse CSR tensor but got {mask.layout}"
        )


def _input_mask(input: Tensor | MaskedTensor, *args, **kwargs) -> Tensor:
    """Return canonical input mask.

    A canonical input mask is defined as a boolean mask tensor that
    shape and layout matches with the shape and the layout of the
    input.

    The canonical input mask is computed from the :attr:`mask` tensor
    content to meet the following criteria:

    1. The shape of the canonical input mask is the same as the shape
       of :attr:`input` tensor. If the mask tensor has a smaller shape
       than the shape of the :attr:`input`, broadcasting rules will be
       applied. Downcasting of mask is not supported.

    2. The layout of the canonical input mask is the same as the
       layout of the :attr:`input` tensor. If the mask has different
       layout, it will be converted to the expected layout.  In the
       case of sparse COO layout, the canonical input mask will be
       coalesced.

    3. The dtype of the canonical input mask is torch.bool. If the
       mask dtype is not bool then it will be converted to bool dtype
       using `.to(dtype=bool)` method call.

    4. The elements of the canonical input mask have boolean values
       copied from the content of the :attr:`mask` tensor (after
       possible broadcasting and dtype conversion transforms).  In
       general, the sparsity pattern of the sparse canonical input
       mask need not to be the same as the sparsity pattern of the
       sparse :attr:`input` tensor.

    """
    if input.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:
        raise ValueError(
            f"_input_mask expects strided or sparse COO or sparse CSR tensor but got {input.layout}"
        )

    mask = kwargs.get("mask")

    # default mask
    if mask is None:
        raise ValueError("_input_mask requires explicit mask")

    # mask shape must match with input shape
    if mask.shape != input.shape:
        if mask.ndim > input.ndim:
            raise IndexError(
                "_input_mask expected broadcastable mask (got mask dimensionality higher than of the input)"
            )
        if mask.layout == torch.strided:
            mask = torch.broadcast_to(mask.clone(), input.shape).to(dtype=torch.bool)
        elif mask.layout == torch.sparse_coo:
            mask = torch._sparse_broadcast_to(mask, input.shape)
        else:
            if mask.layout != torch.sparse_csr:
                raise AssertionError(f"expected sparse_csr layout, got {mask.layout}")
            # Broadcasting of CSR tensors is not implemented. Working
            # around by using COO layout.
            mask = torch._sparse_broadcast_to(
                mask.to_sparse(), input.shape
            ).to_sparse_csr()

    # mask layout must match with input layout
    if mask.layout != input.layout:
        if input.layout == torch.strided:
            mask = mask.to_dense()
        elif input.layout == torch.sparse_coo:
            if mask.layout == torch.strided:
                mask = mask.to_sparse(input.sparse_dim())
            else:
                mask = mask.to_sparse()
        else:
            if input.layout != torch.sparse_csr:
                raise AssertionError(f"expected sparse_csr layout, got {input.layout}")
            mask = mask.to_sparse_csr()

    # sparse mask must be coalesced
    if mask.layout == torch.sparse_coo:
        mask = mask.coalesce()

    # mask is a boolean tensor
    mask = mask.to(dtype=torch.bool)

    return mask


def _output_mask(op, input: Tensor, *args, **kwargs) -> Tensor:
    """Return output mask of masked operation applied to given arguments."""
    if callable(op):
        is_reduction = op.__name__ in {
            "sum",
            "prod",
            "amax",
            "amin",
            "argmax",
            "argmin",
            "mean",
            "median",
            "norm",
            "var",
            "std",
            "logsumexp",
        }
        is_normalization = op.__name__ in {
            "softmax",
            "log_softmax",
            "softmin",
            "normalize",
            "cumsum",
            "cumprod",
        }
        if is_reduction:
            if op.__name__ == "norm":
                if args:
                    args = args[1:]  # lstrip ord argument
            dim = args[0] if args else kwargs.get("dim")
            outmask = _input_mask(input, *args, **kwargs)
            keepdim = kwargs.get("keepdim", False)
            dim_ = _canonical_dim(dim, input.ndim)
            return _any(outmask, dim_, bool(keepdim))
        elif is_normalization:
            return _input_mask(input, *args, **kwargs)
        else:
            raise ValueError(
                f"_output_mask expected masked operation (got callable {op.__module__}.{op.__name__})"
            )
    else:
        raise ValueError(
            f"_output_mask expected masked operation (got {type(op).__name__} object)"
        )


def _combine_input_and_mask(op, input: MaskedTensor | Tensor, mask, *args) -> Tensor:
    def helper(input, mask):
        if mask is None:
            return input
        canonical_mask = _input_mask(input, mask=mask)
        if callable(op):
            fill_value = _reduction_identity(op.__name__, input, *args)
            return _where(canonical_mask, input, fill_value)
        else:
            raise ValueError(
                f"_combine_input_and_mask expected masked operation (got {type(op).__name__} object)"
            )

    class Combine(torch.autograd.Function):
        @staticmethod
        # pyrefly: ignore [bad-override]
        def forward(ctx, input, mask):
            """Return input with masked-out elements eliminated for the given operations."""
            ctx.save_for_backward(mask)

            if mask is not None:
                ctx.mark_non_differentiable(mask)

            return helper(input, mask)

        @staticmethod
        # pyrefly: ignore [bad-override]
        def backward(ctx, grad_output):
            (mask,) = ctx.saved_tensors
            grad_data = (
                grad_output.get_data() if is_masked_tensor(grad_output) else grad_output
            )
            result = as_masked_tensor(grad_data, mask)
            return result, None

    return (
        Combine.apply(input.get_data(), input.get_mask())  # type: ignore[union-attr]
        if is_masked_tensor(input)
        else helper(input, mask)
    )


@_apply_docstring_templates
def sum(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = None,
    *,
    keepdim: bool | None = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    # __doc__ is generated by _apply_docstring_templates decorator
    if dtype is None:
        # promote integer types to int64 when output dtype is not specified
        if input.layout == torch.sparse_csr:
            if input.dtype in {
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
            }:
                # csr.to(dtype=torch.int64) is not implemented, so
                # using coo.to on input to ensure the promoted dtype
                input = input.to_sparse_coo().to(dtype=torch.int64).to_sparse_csr()
            else:
                dtype = input.dtype
        else:
            dtype = input.dtype
            if input.dtype in {
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
            }:
                dtype = torch.int64
    dim_ = _canonical_dim(dim, input.ndim)
    mask_input = _combine_input_and_mask(sum, input, mask)
    if mask_input.layout == torch.strided:
        return torch.sum(mask_input, dim_, bool(keepdim), dtype=dtype)
    elif mask_input.layout == torch.sparse_coo:
        return _sparse_coo_scatter_reduction_helper(
            torch.sum, mask_input, dim_, bool(keepdim), dtype
        )
    elif mask_input.layout == torch.sparse_csr:
        return torch._sparse_csr_sum(
            mask_input, dim=list(dim_), keepdim=bool(keepdim), dtype=dtype
        )
    else:
        raise ValueError(
            f"masked sum expects strided, sparse_coo or sparse_csr tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def prod(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = None,
    *,
    keepdim: bool | None = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    # __doc__ is generated by _apply_docstring_templates decorator
    if dtype is None:
        # promote integer types to int64 when output dtype is not specified
        if input.layout == torch.sparse_csr:
            if input.dtype in {
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
            }:
                # csr.to(dtype=torch.int64) is not implemented, so
                # using coo.to on input to ensure the promoted dtype
                input = input.to_sparse_coo().to(dtype=torch.int64).to_sparse_csr()
            else:
                dtype = input.dtype
        else:
            dtype = input.dtype
            if input.dtype in {
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
            }:
                dtype = torch.int64
    dim_ = _canonical_dim(dim, input.ndim)
    mask_input = _combine_input_and_mask(prod, input, mask)
    if mask_input.layout == torch.strided:
        # Workaround https://github.com/pytorch/pytorch/issues/56586
        result = mask_input
        result = result.to(dtype=dtype)
        for d in reversed(dim_):
            result = result.prod(dim=d, keepdim=bool(keepdim))
        return result
    elif mask_input.layout == torch.sparse_coo:
        if mask is None:
            # See comment in the sparse_csr branch, the same issue arises for sparse_coo tensors
            raise ValueError(
                "masked prod expects explicit mask for sparse_coo tensor input"
            )
        return _sparse_coo_scatter_reduction_helper(
            torch.prod, mask_input, dim_, bool(keepdim), dtype
        )
    elif mask_input.layout == torch.sparse_csr:
        if mask is None:
            # mask is None corresponds to all-True mask. The
            # unspecified elements in the CSR tensor correspond to
            # zero values. Hence, the prod reduction result is
            # automatically zero unless all elements are specified.
            # A semi-optimal way to take this into account is to use:
            #
            #   masked_prod(csr, ..., mask=None) == torch._sparse_csr_prod(csr, ...) * all(csr.nonzero(), ...)
            #
            # but that requires implementing `all` and `nonzero`
            # support for sparse csr tensors.
            raise ValueError(
                "masked prod expects explicit mask for sparse_csr tensor input"
            )
        return torch._sparse_csr_prod(
            mask_input, dim=list(dim_), keepdim=bool(keepdim), dtype=dtype
        )
    else:
        raise ValueError(
            f"masked prod expects strided, sparse_coo or sparse_csr tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def cumsum(
    input: Tensor,
    dim: int,
    *,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)[0]
    mask_input = _combine_input_and_mask(sum, input, mask)
    if mask_input.layout == torch.strided:
        return torch.cumsum(mask_input, dim_, dtype=dtype).to(dtype=dtype)
    else:
        raise ValueError(
            f"masked cumsum expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def cumprod(
    input: Tensor,
    dim: int,
    *,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)[0]
    mask_input = _combine_input_and_mask(prod, input, mask)
    if mask_input.layout == torch.strided:
        return torch.cumprod(mask_input, dim_, dtype=dtype).to(dtype=dtype)
    else:
        raise ValueError(
            f"masked cumprod expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def amax(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = None,
    *,
    keepdim: bool | None = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """\
{reduction_signature}

{reduction_descr}

{reduction_identity_dtype}

{reduction_args}

{reduction_example}"""
    if dtype is None:
        dtype = input.dtype

    mask_input = _combine_input_and_mask(amax, input, mask)
    dim_ = _canonical_dim(dim, mask_input.ndim)
    if mask_input.layout == torch.strided:
        return torch.amax(mask_input, dim_, bool(keepdim)).to(dtype=dtype)
    elif mask_input.layout == torch.sparse_coo:
        if mask is None:
            # See comment in the sparse_csr branch of prod, a similar issue arises here
            # where unspecified elements along a dimension may need to be reduced with the result
            raise ValueError(
                "masked amax expects explicit mask for sparse_coo tensor input"
            )
        return _sparse_coo_scatter_reduction_helper(
            torch.amax, mask_input, dim_, bool(keepdim), dtype
        )
    elif mask_input.layout == torch.sparse_csr:
        if mask is None:
            raise ValueError(
                "masked amax expects explicit mask for sparse_csr tensor input"
            )
        return _sparse_csr_segment_reduction_helper(
            torch.amax, mask_input, dim_, bool(keepdim), dtype
        )
    else:
        raise ValueError(
            f"masked amax expects strided, sparse_coo or sparse_csr tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def amin(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = None,
    *,
    keepdim: bool | None = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """\
{reduction_signature}

{reduction_descr}

{reduction_identity_dtype}

{reduction_args}

{reduction_example}"""
    if dtype is None:
        dtype = input.dtype

    mask_input = _combine_input_and_mask(amin, input, mask)
    dim_ = _canonical_dim(dim, mask_input.ndim)
    if mask_input.layout == torch.strided:
        return torch.amin(mask_input, dim_, bool(keepdim)).to(dtype=dtype)
    elif mask_input.layout == torch.sparse_coo:
        if mask is None:
            # See comment in the sparse_csr branch of prod, a similar issue arises here
            # where unspecified elements along a dimension may need to be reduced with the result
            raise ValueError(
                "masked amax expects explicit mask for sparse_coo tensor input"
            )
        return _sparse_coo_scatter_reduction_helper(
            torch.amin, mask_input, dim_, bool(keepdim), dtype
        )
    elif mask_input.layout == torch.sparse_csr:
        if mask is None:
            raise ValueError(
                "masked amin expects explicit mask for sparse_csr tensor input"
            )
        return _sparse_csr_segment_reduction_helper(
            torch.amin, mask_input, dim_, bool(keepdim), dtype
        )
    else:
        raise ValueError(
            f"masked amin expects strided, sparse_coo or sparse_csr tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def argmax(
    input: Tensor | MaskedTensor,
    dim: int | None = None,
    *,
    keepdim: bool | None = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """\
{reduction_signature}
{reduction_descr}
{reduction_identity_dtype}
{reduction_args}
{reduction_example}"""
    if dtype is None:
        dtype = input.dtype
    mask_input = _combine_input_and_mask(argmax, input, mask)
    if mask_input.layout == torch.strided:
        return torch.argmax(mask_input, dim, bool(keepdim)).to(dtype=dtype)
    else:
        raise ValueError(
            f"masked argmax expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def argmin(
    input: Tensor | MaskedTensor,
    dim: int | None = None,
    *,
    keepdim: bool | None = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """\
{reduction_signature}
{reduction_descr}
{reduction_identity_dtype}
{reduction_args}
{reduction_example}"""
    if dtype is None:
        dtype = input.dtype
    mask_input = _combine_input_and_mask(argmin, input, mask)
    if mask_input.layout == torch.strided:
        return torch.argmin(mask_input, dim, bool(keepdim)).to(dtype=dtype)
    else:
        raise ValueError(
            f"masked argmin expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def mean(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = None,
    *,
    keepdim: bool | None = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """\
{reduction_signature}

{reduction_descr}

By definition, the identity value of a mean operation is the mean
value of the tensor. If all elements of the input tensor along given
dimension(s) :attr:`dim` are masked-out, the identity value of the
mean is undefined.  Due to this ambiguity, the elements of output
tensor with strided layout, that correspond to fully masked-out
elements, have ``nan`` values.

{reduction_args}

{reduction_example}"""
    dtype_source = "Optional"
    if dtype is None:
        dtype = input.dtype
        dtype_source = "Input"

    if not (dtype.is_floating_point or dtype.is_complex):
        raise ValueError(
            f"mean(): Could not infer output dtype. {dtype_source} dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        )
    if input.layout == torch.strided:
        if mask is None:
            # TODO: compute count analytically
            # pyrefly: ignore [no-matching-overload]
            count = sum(
                torch.ones(input.shape, dtype=torch.int64, device=input.device),
                dim,
                keepdim=keepdim,
            )
            # pyrefly: ignore [no-matching-overload]
            total = sum(input, dim, keepdim=keepdim, dtype=dtype)
        else:
            inmask = _input_mask(input, mask=mask)
            count = inmask.sum(dim=dim, keepdim=bool(keepdim))
            # pyrefly: ignore [no-matching-overload]
            total = sum(input, dim, keepdim=keepdim, dtype=dtype, mask=inmask)
        return total / count
    elif input.layout == torch.sparse_csr:
        mask_input = _combine_input_and_mask(mean, input, mask)
        dim_ = _canonical_dim(dim, mask_input.ndim)
        if mask is None:
            raise ValueError(
                "masked mean expects explicit mask for sparse_csr tensor input"
            )
        return _sparse_csr_segment_reduction_helper(
            torch.mean, mask_input, dim_, bool(keepdim), dtype
        )
    else:
        raise ValueError(
            f"masked mean expects strided or sparse_csr tensor (got {input.layout} tensor)"
        )


@_apply_docstring_templates
def median(
    input: Tensor | MaskedTensor,
    dim: int = -1,
    *,
    keepdim: bool = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """\
{reduction_signature}
{reduction_descr}
By definition, the identity value of a median operation is the median
value of the tensor. If all elements of the input tensor along given
dimension(s) :attr:`dim` are masked-out, the identity value of the
median is undefined.  Due to this ambiguity, the elements of output
tensor with strided layout, that correspond to fully masked-out
elements, have ``nan`` values.
{reduction_args}
{reduction_example}"""
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)[0]
    is_float = torch.is_floating_point(input)
    if not is_float:
        input = input.to(dtype=torch.float)
    mask_input = _combine_input_and_mask(median, input, mask)
    if mask_input.layout == torch.strided:
        output = torch.nanmedian(mask_input, dim_, keepdim).values
        if is_float:
            return output
        elif not is_float and not torch.isnan(output).any():
            return output.to(dtype=dtype)
        else:
            raise ValueError(
                "masked median expects no fully masked out rows if dtype is not floating point"
            )
    else:
        raise ValueError(
            f"masked median expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def logsumexp(
    input: Tensor,
    dim: DimOrDims = None,
    *,
    keepdim: bool = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)
    mask_input = _combine_input_and_mask(logsumexp, input, mask)
    if mask_input.layout == torch.strided:
        return torch.logsumexp(mask_input, dim_, keepdim=keepdim).to(dtype=dtype)
    else:
        raise ValueError(
            f"masked logsumexp expects strided tensor (got {mask_input.layout} tensor)"
        )


# Cannot use _apply_docstring_templates as it is only set up for reductions and normalizations
def logaddexp(
    input: Tensor | MaskedTensor,
    other: Tensor | MaskedTensor,
    *,
    dtype: DType | None = None,
    input_mask: Tensor | None = None,
    other_mask: Tensor | None = None,
) -> Tensor:
    """logaddexp(input, other, *, dtype=None, input_mask=None, other_mask=None) -> Tensor

    Returns logaddexp of all the elements in the :attr:`input` and the :attr:`other`
    tensor. The :attr:`input` elements are masked out according to the boolean tensor
    :attr:`input_mask` and the attr:`other` elements are masked out according to the boolean tensor
    :attr:`other_mask`.

    The shapes of a mask tensor and the tensor to be masked
    don't need to match, but they must be :ref:`broadcastable
    <broadcasting-semantics>` and the dimensionality of the mask
    tensor must not be greater than of the tensor to be masked.

    Args:
        input (Tensor): the input tensor
        other (Tensor): the second input tensor

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type
          of returned tensor.  If specified, the output tensor is
          casted to :attr:`dtype` after the operation is
          performed. Default: None.
        input_mask (:class:`torch.Tensor`, optional): the boolean tensor
          containing the binary mask of validity of :attr:`input` tensor elements.
          Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.
        other_mask (:class:`torch.Tensor`, optional): the boolean tensor
          containing the binary mask of validity of :attr:`other` tensor elements.
          Default: None that is equivalent to ``torch.ones(other.shape, dtype=torch.bool)``.

    Example::

        >>> input = torch.tensor([-100.0, -200, -300])
        >>> input
        tensor([-100., -200., -300.])
        >>> other = torch.tensor([-1.0, -2, -3])
        >>> other
        tensor([-1., -2., -3.])
        >>> mask = torch.tensor([True, False, True])
        >>> mask
        tensor([ True, False,  True])
        >>> torch.masked._ops.logaddexp(input, other, input_mask=mask, other_mask=mask)
        tensor([-1., -inf, -3.])"""
    if dtype is None:
        dtype = input.dtype
    if input.layout == torch.strided and other.layout == torch.strided:
        mask_input = _combine_input_and_mask(logaddexp, input, input_mask)
        mask_other = _combine_input_and_mask(logaddexp, other, other_mask)
        return torch.logaddexp(mask_input, mask_other).to(dtype=dtype)
    else:
        raise ValueError(
            f"masked logaddexp expects strided tensors (got {input.layout} tensor for input, {other.layout} for other)"
        )


@_apply_docstring_templates
def norm(
    input: Tensor | MaskedTensor,
    ord: float | None = 2.0,
    dim: DimOrDims = None,
    *,
    keepdim: bool | None = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """\
{reduction_signature}

{reduction_descr}

The identity value of norm operation, which is used to start the
reduction, is ``{identity_float32}``, except for ``ord=-inf`` it is
``{identity_ord_ninf}``.

{reduction_args}

{reduction_example}"""
    if dtype is None:
        dtype = input.dtype
    mask_input = _combine_input_and_mask(norm, input, mask, ord)
    if mask_input.layout == torch.strided:
        dim_ = _canonical_dim(dim, input.ndim)
        return torch.linalg.vector_norm(
            mask_input, ord, dim_, bool(keepdim), dtype=dtype
        )
    else:
        raise ValueError(
            f"masked norm expects strided tensor (got {mask_input.layout} tensor)"
        )


def _std_var(
    input: Tensor | MaskedTensor,
    dim: DimOrDims,
    unbiased: bool | None,
    *,
    correction_opt: int | float | None,
    keepdim: bool | None,
    dtype: DType | None,
    mask: Tensor | None,
    take_sqrt: bool | None,
) -> Tensor:
    if unbiased is not None and correction_opt is not None:
        raise AssertionError("Only one of unbiased and correction may be given")
    correction = 1.0
    if unbiased is not None:
        correction = 1.0 if unbiased else 0.0
    if correction_opt is not None:
        correction = sym_float(correction_opt)

    if dtype is None:
        dtype = input.dtype
        if not (dtype.is_floating_point or dtype.is_complex):
            dtype = torch.float32
    compute_dtype = dtype
    if not (compute_dtype.is_floating_point or compute_dtype.is_complex):
        compute_dtype = torch.float32
    if input.layout == torch.strided:
        if mask is None:
            # TODO: compute count analytically
            # pyrefly: ignore [no-matching-overload]
            count = sum(
                torch.ones(input.shape, dtype=torch.int64, device=input.device),
                dim,
                keepdim=True,
            )
            # pyrefly: ignore [no-matching-overload]
            sample_total = sum(input, dim, keepdim=True, dtype=dtype)
        else:
            inmask = _input_mask(input, mask=mask)
            count = inmask.sum(dim=dim, keepdim=True)
            # pyrefly: ignore [no-matching-overload]
            sample_total = sum(input, dim, keepdim=True, dtype=dtype, mask=inmask)
        # TODO: replace torch.subtract/divide/square/maximum with
        # masked subtract/divide/square/maximum when these will be
        # available.
        sample_mean = torch.divide(sample_total, count)
        x = torch.subtract(input, sample_mean)
        if mask is None:
            # pyrefly: ignore [no-matching-overload]
            total = sum(x * x.conj(), dim, keepdim=keepdim, dtype=compute_dtype)
        else:
            # pyrefly: ignore [no-matching-overload]
            total = sum(
                x * x.conj(),
                dim,
                keepdim=keepdim,
                dtype=compute_dtype,
                mask=inmask,  # type: ignore[possibly-undefined]
            )
        if not keepdim:
            count = count.reshape(total.shape)
        if correction != 0:
            real_dtype = (
                corresponding_real_dtype(compute_dtype)
                if compute_dtype.is_complex
                else compute_dtype
            )
            count = count.to(real_dtype)
            # pyrefly: ignore [no-matching-overload]
            count = torch.subtract(count, correction)
            count = torch.maximum(count, count.new_zeros([]))
        output = torch.divide(total, count).to(dtype=dtype)
        if take_sqrt:
            output = torch.sqrt(output)
        return output
    else:
        raise ValueError(
            f"masked std/var expects strided tensor (got {input.layout} tensor)"
        )


@_apply_docstring_templates
def var(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = None,
    unbiased: bool | None = None,
    *,
    correction: int | float | None = None,
    keepdim: bool | None = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """\
{reduction_signature}
{reduction_descr}
The identity value of sample variance operation is undefined. The
elements of output tensor with strided layout, that correspond to
fully masked-out elements, have ``nan`` values.
{reduction_args}
{reduction_example}"""
    return _std_var(
        input=input,
        dim=dim,
        unbiased=unbiased,
        correction_opt=correction,
        keepdim=keepdim,
        dtype=dtype,
        mask=mask,
        take_sqrt=False,
    )


@_apply_docstring_templates
def std(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = None,
    unbiased: bool | None = None,
    *,
    correction: int | None = None,
    keepdim: bool | None = False,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """\
{reduction_signature}
{reduction_descr}
The identity value of sample standard deviation operation is undefined. The
elements of output tensor with strided layout, that correspond to
fully masked-out elements, have ``nan`` values.
{reduction_args}
{reduction_example}"""
    return _std_var(
        input=input,
        dim=dim,
        unbiased=unbiased,
        correction_opt=correction,
        keepdim=keepdim,
        dtype=dtype,
        mask=mask,
        take_sqrt=True,
    )


@_apply_docstring_templates
def softmax(
    input: Tensor | MaskedTensor,
    dim: int,
    *,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)[0]
    mask_input = _combine_input_and_mask(amax, input, mask)
    if mask_input.layout == torch.strided:
        return torch.nn.functional.softmax(mask_input, dim_, dtype=dtype)
    else:
        raise ValueError(
            f"masked softmax expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def log_softmax(
    input: Tensor | MaskedTensor,
    dim: int,
    *,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)[0]
    mask_input = _combine_input_and_mask(amax, input, mask)
    if mask_input.layout == torch.strided:
        return torch.nn.functional.log_softmax(mask_input, dim_, dtype=dtype)
    else:
        raise ValueError(
            f"masked log_softmax expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def softmin(
    input: Tensor | MaskedTensor,
    dim: int,
    *,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)[0]
    mask_input = _combine_input_and_mask(amin, input, mask)
    if mask_input.layout == torch.strided:
        return torch.nn.functional.softmin(mask_input, dim_, dtype=dtype)
    else:
        raise ValueError(
            f"masked softmin expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def normalize(
    input: Tensor | MaskedTensor,
    ord: float,
    dim: int,
    *,
    eps: float = 1e-12,
    dtype: DType | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    # TODO: eliminate mask_input as unnecessary when using masked divide.
    mask_input = _combine_input_and_mask(sum, input, mask)
    if mask_input.layout == torch.strided:
        nrm_ = norm(input, ord, dim, keepdim=True, dtype=dtype, mask=mask)
        # TODO: replace torch.maximum with masked maximum when available.
        denom = torch.maximum(nrm_, nrm_.new_full([], eps))
        # TODO: replace torch.divide with masked divide when available.
        return torch.divide(mask_input, denom)
    else:
        raise ValueError(
            f"masked normalize expects strided tensor (got {mask_input.layout} tensor)"
        )
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import warnings

import torch

from .core import is_masked_tensor
from .creation import as_masked_tensor, masked_tensor


__all__ = []  # type: ignore[var-annotated]


def _masked_all_all(data, mask=None):
    if mask is None:
        return data.all()
    return data.masked_fill(~mask, True).all()


def _masked_all_dim(data, dim, keepdim=False, mask=None):
    if mask is None:
        return torch.all(data, dim=dim, keepdim=keepdim)
    return torch.all(data.masked_fill(~mask, True), dim=dim, keepdim=keepdim)


def _masked_all(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 1:
        return _masked_all_all(args[0], mask=kwargs["mask"])
    return _masked_all_dim(*args, **kwargs)


def _multidim_any(mask, dim, keepdim):
    if isinstance(dim, int):
        return _multidim_any(mask, [dim], keepdim)
    for d in sorted(dim, reverse=True):
        mask = torch.any(mask, dim=d, keepdim=keepdim)
    return mask


def _get_masked_fn(fn):
    if fn == "all":
        return _masked_all
    return getattr(torch.masked, fn)


def _torch_reduce_all(fn):
    def reduce_all(self):
        masked_fn = _get_masked_fn(fn)
        data = self.get_data()
        mask = self.get_mask().values() if self.is_sparse else self.get_mask()
        # When reduction is "all", then torch.argmin/torch.argmax needs to return the index of the
        # element corresponding to the min/max, but this operation isn't supported correctly for sparse layouts.
        # Therefore, this implementation calculates it using the strides.
        if fn == "all":
            result_data = masked_fn(data, mask=mask)

        elif fn in {"argmin", "argmax"} and self.is_sparse_coo():
            sparse_idx = masked_fn(data.values(), mask=mask).to(dtype=torch.int)
            indices = (
                data.to_sparse_coo().indices()
                if not self.is_sparse_coo()
                else data.indices()
            )
            idx = indices.unbind(1)[sparse_idx]
            stride = data.size().numel() / torch.tensor(
                data.size(), device=data.device
            ).cumprod(0)
            result_data = torch.sum(idx * stride)

        # we simply pass in the values for sparse COO/CSR tensors
        elif self.is_sparse:
            result_data = masked_fn(masked_tensor(data.values(), mask))

        else:
            result_data = masked_fn(self, mask=mask)

        return as_masked_tensor(result_data, torch.any(mask))

    return reduce_all


def _torch_reduce_dim(fn):
    def reduce_dim(self, dim, keepdim=False, dtype=None):
        if self.is_sparse:
            msg = (
                f"The sparse version of {fn} is not implemented in reductions.\n"
                "If you would like this operator to be supported, please file an issue for a feature request at "
                "https://github.com/pytorch/maskedtensor/issues with a minimal reproducible code snippet.\n"
                "In the case that the semantics for the operator are not trivial, it would be appreciated "
                "to also include a proposal for the semantics."
            )
            warnings.warn(msg, stacklevel=2)
            return NotImplemented
        if not is_masked_tensor(self):
            raise TypeError("Input to reduce_dim must be a MaskedTensor")

        masked_fn = _get_masked_fn(fn)
        data = self.get_data()
        mask = self.get_mask()
        if fn == "all":
            result_data = masked_fn(data, dim=dim, keepdim=keepdim, mask=mask)
        else:
            result_data = masked_fn(
                self, dim=dim, keepdim=keepdim, dtype=dtype, mask=self.get_mask()
            )
        return as_masked_tensor(result_data, _multidim_any(mask, dim, keepdim))

    return reduce_dim


def _torch_reduce(fn):
    def reduce_fn(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return _torch_reduce_all(fn)(args[0])
        return _torch_reduce_dim(fn)(*args, **kwargs)

    return reduce_fn


def _reduce_dim_args(input, dim, keepdim=False, dtype=None):
    return input, dim, keepdim, dtype


def _torch_grad_reduce(fn):
    def grad_reduce(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return _torch_reduce_all(fn)(args[0])
        # TODO: autograd.Function doesn't support kwarg
        input, dim, keepdim, dtype = _reduce_dim_args(*args, **kwargs)
        return _torch_reduce_dim(fn)(input, dim, keepdim, dtype)

    return grad_reduce


REDUCE_NAMES = [
    "sum",
    "mean",
    "amin",
    "amax",
    "argmin",
    "argmax",
    "prod",
    "all",
    "norm",
    "var",
    "std",
]

NATIVE_REDUCE_MAP = {
    getattr(torch.ops.aten, name): _torch_reduce(name) for name in REDUCE_NAMES
}
TORCH_REDUCE_MAP = {
    getattr(torch, name): _torch_grad_reduce(name) for name in REDUCE_NAMES
}
TENSOR_REDUCE_MAP = {
    getattr(torch.Tensor, name): _torch_grad_reduce(name) for name in REDUCE_NAMES
}

NATIVE_REDUCE_FNS = list(NATIVE_REDUCE_MAP.keys())
TORCH_REDUCE_FNS = list(TORCH_REDUCE_MAP.keys())
TENSOR_REDUCE_FNS = list(TENSOR_REDUCE_MAP.keys())


def _is_reduction(fn):
    return fn in NATIVE_REDUCE_MAP or fn in TORCH_REDUCE_MAP or fn in TENSOR_REDUCE_MAP


def _apply_reduction(fn, *args, **kwargs):
    if fn in NATIVE_REDUCE_MAP:
        return NATIVE_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TORCH_REDUCE_MAP:
        return TORCH_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TENSOR_REDUCE_MAP:
        return TENSOR_REDUCE_MAP[fn](*args, **kwargs)
    return NotImplemented
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch

from .core import (
    _map_mt_args_kwargs,
    _masks_match,
    _tensors_match,
    _wrap_result,
    is_masked_tensor,
)


__all__ = []  # type: ignore[var-annotated]

BINARY_NAMES = [
    "add",
    "atan2",
    "arctan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "div",
    "divide",
    "floor_divide",
    "fmod",
    "logaddexp",
    "logaddexp2",
    "mul",
    "multiply",
    "nextafter",
    "remainder",
    "sub",
    "subtract",
    "true_divide",
    "eq",
    "ne",
    "le",
    "ge",
    "greater",
    "greater_equal",
    "gt",
    "less_equal",
    "lt",
    "less",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "not_equal",
]

INPLACE_BINARY_NAMES = [
    n + "_"
    for n in (
        list(
            set(BINARY_NAMES)
            - {
                "logaddexp",
                "logaddexp2",
                "equal",
                "fmin",
                "minimum",
                "maximum",
                "fmax",
            }
        )
    )
]


def _get_at_least_one_mask(a, b):
    if not is_masked_tensor(a) and not is_masked_tensor(b):
        raise TypeError("At least one of `a` and `b` must be a MaskedTensor")
    if not _masks_match(a, b):
        raise ValueError("a and b must have matching masks")
    if is_masked_tensor(a):
        return a.get_mask()
    return b.get_mask()


def _binary_helper(fn, args, kwargs, inplace):
    if len(kwargs) != 0:
        raise ValueError("len(kwargs) must equal 0")
    for a in args[2:]:
        if torch.is_tensor(a):
            raise TypeError(
                "MaskedTensor binary ops do not support Tensor arguments aside from the lhs and rhs"
            )

    if not _masks_match(*args[:2]):
        raise ValueError(
            "Input masks must match. If you need support for this, please open an issue on Github."
        )

    data_args, _data_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_data())
    mask_args, _mask_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_mask())

    args0_layout = data_args[0].layout
    same_layout = (
        torch.is_tensor(data_args[1]) or is_masked_tensor(data_args[1])
    ) and (args0_layout == data_args[1].layout)

    if args0_layout == torch.sparse_coo:
        if same_layout:
            if not _tensors_match(data_args[0].indices(), data_args[1].indices()):
                raise ValueError(
                    "sparse_coo indices must match. If you need support for this, please open an issue on Github."
                )
            if data_args[0].size() != data_args[1].size():
                raise ValueError(
                    "input1 and input2 must have the same size for binary functions."
                )

            data_args[1] = data_args[1].values()

        i = data_args[0].indices()
        size = data_args[0].size()
        data_args[0] = data_args[0].values()
        v = fn(*data_args)
        result_data = torch.sparse_coo_tensor(i, v, size)

    elif args0_layout == torch.sparse_csr:
        if same_layout:
            if not (
                _tensors_match(data_args[0].crow_indices(), data_args[1].crow_indices())
                and _tensors_match(
                    data_args[0].col_indices(), data_args[1].col_indices()
                )
            ):
                raise ValueError(
                    "sparse_csr indices must match. If you need support for this, please open an issue on Github."
                )

            data_args[1] = data_args[1].values()

        crow = data_args[0].crow_indices()
        col = data_args[0].col_indices()
        size = data_args[0].size()
        data_args[0] = data_args[0].values()
        v = fn(*data_args)
        result_data = torch.sparse_csr_tensor(crow, col, v, size)

    else:
        result_data = fn(*data_args)

    if inplace:
        args[0]._set_data_mask(result_data, mask_args[0])
        return args[0]
    else:
        result_mask = _get_at_least_one_mask(*args[:2])
        # sparse tensors don't have strides so we can only expand if the layout is strided
        if args0_layout == torch.strided:
            result_mask = result_mask.expand_as(result_data)
        return _wrap_result(result_data, result_mask)


def _torch_binary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    def binary_fn(*args, **kwargs):
        return _binary_helper(fn, args, kwargs, inplace=False)

    return binary_fn


def _torch_inplace_binary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    def binary_fn(*args, **kwargs):
        return _binary_helper(fn, args, kwargs, inplace=True)

    return binary_fn


NATIVE_BINARY_MAP = {
    getattr(torch.ops.aten, name): _torch_binary(name) for name in BINARY_NAMES
}
NATIVE_INPLACE_BINARY_MAP = {
    getattr(torch.ops.aten, name): _torch_inplace_binary(name)
    for name in INPLACE_BINARY_NAMES
}

NATIVE_BINARY_FNS = list(NATIVE_BINARY_MAP.keys())
NATIVE_INPLACE_BINARY_FNS = list(NATIVE_INPLACE_BINARY_MAP.keys())


def _is_native_binary(fn):
    return fn in NATIVE_BINARY_FNS or fn in NATIVE_INPLACE_BINARY_FNS


def _apply_native_binary(fn, *args, **kwargs):
    if fn in NATIVE_BINARY_FNS:
        return NATIVE_BINARY_MAP[fn](*args, **kwargs)
    if fn in NATIVE_INPLACE_BINARY_FNS:
        return NATIVE_INPLACE_BINARY_MAP[fn](*args, **kwargs)
    return NotImplemented
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch

from .core import _map_mt_args_kwargs, _wrap_result


__all__ = []  # type: ignore[var-annotated]


UNARY_NAMES = [
    "abs",
    "absolute",
    "acos",
    "arccos",
    "acosh",
    "arccosh",
    "angle",
    "asin",
    "arcsin",
    "asinh",
    "arcsinh",
    "atan",
    "arctan",
    "atanh",
    "arctanh",
    "bitwise_not",
    "ceil",
    "clamp",
    "clip",
    "conj_physical",
    "cos",
    "cosh",
    "deg2rad",
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "fix",
    "floor",
    "frac",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "logit",
    "i0",
    "isnan",
    "nan_to_num",
    "neg",
    "negative",
    "positive",
    "pow",
    "rad2deg",
    "reciprocal",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sgn",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
]

INPLACE_UNARY_NAMES = [
    n + "_"
    for n in (list(set(UNARY_NAMES) - {"angle", "positive", "signbit", "isnan"}))
]

# Explicitly tracking functions we know are currently not supported
# This might be due to missing code gen or because of complex semantics
UNARY_NAMES_UNSUPPORTED = [
    "atan2",
    "arctan2",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "copysign",
    "float_power",
    "fmod",
    "frexp",
    "gradient",
    "imag",
    "ldexp",
    "lerp",
    "logical_not",
    "hypot",
    "igamma",
    "igammac",
    "mvlgamma",
    "nextafter",
    "polygamma",
    "real",
    "remainder",
    "true_divide",
    "xlogy",
]


def _unary_helper(fn, args, kwargs, inplace):
    if len(kwargs) != 0:
        raise ValueError(
            "MaskedTensor unary ops require that len(kwargs) == 0. "
            "If you need support for this, please open an issue on Github."
        )
    for a in args[1:]:
        if torch.is_tensor(a):
            raise TypeError(
                "MaskedTensor unary ops do not support additional Tensor arguments"
            )

    mask_args, _mask_kwargs = _map_mt_args_kwargs(
        args, kwargs, lambda x: x._masked_mask
    )
    data_args, _data_kwargs = _map_mt_args_kwargs(
        args, kwargs, lambda x: x._masked_data
    )

    if args[0].layout == torch.sparse_coo:
        data_args[0] = data_args[0].coalesce()
        s = data_args[0].size()
        i = data_args[0].indices()
        data_args[0] = data_args[0].coalesce().values()
        v = fn(*data_args)
        result_data = torch.sparse_coo_tensor(i, v, size=s)

    elif args[0].layout == torch.sparse_csr:
        crow = data_args[0].crow_indices()
        col = data_args[0].col_indices()
        data_args[0] = data_args[0].values()
        v = fn(*data_args)
        result_data = torch.sparse_csr_tensor(crow, col, v)

    else:
        result_data = fn(*data_args)

    if inplace:
        args[0]._set_data_mask(result_data, mask_args[0])
        return args[0]
    else:
        return _wrap_result(result_data, mask_args[0])


def _torch_unary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    def unary_fn(*args, **kwargs):
        return _unary_helper(fn, args, kwargs, inplace=False)

    return unary_fn


def _torch_inplace_unary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    def unary_fn(*args, **kwargs):
        return _unary_helper(fn, args, kwargs, inplace=True)

    return unary_fn


NATIVE_UNARY_MAP = {
    getattr(torch.ops.aten, name): _torch_unary(name) for name in UNARY_NAMES
}
NATIVE_INPLACE_UNARY_MAP = {
    getattr(torch.ops.aten, name): _torch_inplace_unary(name)
    for name in INPLACE_UNARY_NAMES
}

NATIVE_UNARY_FNS = list(NATIVE_UNARY_MAP.keys())
NATIVE_INPLACE_UNARY_FNS = list(NATIVE_INPLACE_UNARY_MAP.keys())


def _is_native_unary(fn):
    return fn in NATIVE_UNARY_FNS or fn in NATIVE_INPLACE_UNARY_FNS


def _apply_native_unary(fn, *args, **kwargs):
    if fn in NATIVE_UNARY_FNS:
        return NATIVE_UNARY_MAP[fn](*args, **kwargs)
    if fn in NATIVE_INPLACE_UNARY_FNS:
        return NATIVE_INPLACE_UNARY_MAP[fn](*args, **kwargs)
    return NotImplemented
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates

from .binary import _apply_native_binary, _is_native_binary
from .core import is_masked_tensor, MaskedTensor
from .passthrough import _apply_pass_through_fn, _is_pass_through_fn
from .reductions import _apply_reduction, _is_reduction
from .unary import _apply_native_unary, _is_native_unary
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates

from .core import MaskedTensor


__all__ = [
    "as_masked_tensor",
    "masked_tensor",
]


# These two factory functions are intended to mirror
#     torch.tensor - guaranteed to be a leaf node
#     torch.as_tensor - differentiable constructor that preserves the autograd history


def masked_tensor(
    data: object, mask: object, requires_grad: bool = False
) -> MaskedTensor:
    return MaskedTensor(data, mask, requires_grad)


def as_masked_tensor(data: object, mask: object) -> MaskedTensor:
    return MaskedTensor._from_values(data, mask)
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import warnings
from typing import Any
from typing_extensions import TypeIs

import torch
from torch.overrides import get_default_nowrap_functions


__all__ = [
    "MaskedTensor",
    "is_masked_tensor",
]


def is_masked_tensor(obj: Any, /) -> TypeIs["MaskedTensor"]:
    r"""Returns True if the input is a MaskedTensor, else False

    Args:
        a: any input

    Examples:

        >>> # xdoctest: +SKIP
        >>> from torch.masked import MaskedTensor
        >>> data = torch.arange(6).reshape(2, 3)
        >>> mask = torch.tensor([[True, False, False], [True, True, False]])
        >>> mt = MaskedTensor(data, mask)
        >>> is_masked_tensor(mt)
        True
    """
    return isinstance(obj, MaskedTensor)


def _tensors_match(a, b, exact=True, rtol=1e-05, atol=1e-08):
    if is_masked_tensor(a) or is_masked_tensor(b):
        raise ValueError("Neither `a` nor `b` can be a MaskedTensor.")
    if a.layout != b.layout:
        raise ValueError(
            f"`a` and `b` must have the same layout. Got {a.layout} and {b.layout}"
        )

    if a.dtype != b.dtype:
        b = b.type(a.dtype)
    if a.layout == b.layout == torch.sparse_coo:
        return _tensors_match(a.values(), b.values(), exact) and _tensors_match(
            a.indices(), b.indices(), exact
        )
    elif a.layout == b.layout == torch.sparse_csr:
        return (
            _tensors_match(a.crow_indices(), b.crow_indices(), exact)
            and _tensors_match(a.col_indices(), b.col_indices(), exact)
            and _tensors_match(a.values(), b.values(), exact)
        )
    if exact:
        return (a.dim() == b.dim()) and torch.eq(a, b).all().item()
    return (a.dim() == b.dim()) and torch.allclose(a, b, rtol=rtol, atol=atol)


def _masks_match(a, b):
    if is_masked_tensor(a) and is_masked_tensor(b):
        mask_a = a.get_mask()
        mask_b = b.get_mask()
        return _tensors_match(mask_a, mask_b, exact=True)
    return True


def _map_mt_args_kwargs(args, kwargs, map_fn):
    def _helper(a, map_fn):
        if is_masked_tensor(a):
            return map_fn(a)
        elif torch.is_tensor(a):
            return a
        elif isinstance(a, list):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return a_impl
        elif isinstance(a, tuple):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return tuple(a_impl)
        else:
            return a

    if kwargs is None:
        kwargs = {}
    impl_args = []
    for a in args:
        impl_args.append(_helper(a, map_fn))
    impl_kwargs = {}
    for k in kwargs:
        impl_kwargs[k] = _helper(a, map_fn)
    return impl_args, impl_kwargs


def _wrap_result(result_data, result_mask):
    if isinstance(result_data, list):
        return [_wrap_result(r, m) for (r, m) in zip(result_data, result_mask)]
    if isinstance(result_data, tuple):
        return tuple(_wrap_result(r, m) for (r, m) in zip(result_data, result_mask))
    if torch.is_tensor(result_data):
        return MaskedTensor(result_data, result_mask)
    # Expect result_data and result_mask to be Tensors only
    return NotImplemented


def _masked_tensor_str(data, mask, formatter):
    if data.layout in {torch.sparse_coo, torch.sparse_csr}:
        data = data.to_dense()
        mask = mask.to_dense()
    if data.dim() == 1:
        formatted_elements = [
            formatter.format(d.item()) if isinstance(d.item(), float) else str(d.item())
            for d in data
        ]
        max_len = max(8 if x[1] else len(x[0]) for x in zip(formatted_elements, ~mask))
        return (
            "["
            + ", ".join(
                [
                    "--".rjust(max_len) if m else e
                    for (e, m) in zip(formatted_elements, ~mask)
                ]
            )
            + "]"
        )
    sub_strings = [_masked_tensor_str(d, m, formatter) for (d, m) in zip(data, mask)]
    sub_strings = ["\n".join(["  " + si for si in s.split("\n")]) for s in sub_strings]
    return "[\n" + ",\n".join(sub_strings) + "\n]"


def _get_data(a):
    if is_masked_tensor(a):
        return a._masked_data
    return a


def _maybe_get_mask(a):
    if is_masked_tensor(a):
        return a.get_mask()
    return None


class MaskedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, mask, requires_grad=False):
        if is_masked_tensor(data) or not torch.is_tensor(data):
            raise TypeError("data must be a Tensor")
        if is_masked_tensor(mask) or not torch.is_tensor(mask):
            raise TypeError("mask must be a Tensor")
        # Use a Tensor that of the give size for the wrapper.
        kwargs = {
            "device": data.device,
            "dtype": data.dtype,
            "layout": data.layout,
            "requires_grad": requires_grad,
            "dispatch_sizes_strides_policy": "strides",
            "dispatch_layout": True,
        }
        warnings.warn(
            (
                "The PyTorch API of MaskedTensors is in prototype stage "
                "and will change in the near future. Please open a Github issue "
                "for feature requests and see our documentation on the torch.masked "
                "module for further information about the project."
            ),
            UserWarning,
            stacklevel=2,
        )
        if data.requires_grad:
            warnings.warn(
                "It is not recommended to create a MaskedTensor with a tensor that requires_grad. "
                "To avoid this, you can use data.detach().clone()",
                UserWarning,
                stacklevel=2,
            )
        # pyrefly: ignore [bad-argument-type]
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)

    def _preprocess_data(self, data, mask):
        from .._ops import _sparse_coo_where, _sparse_csr_where

        if data.layout != mask.layout:
            raise TypeError("data and mask must have the same layout.")
        if data.layout == torch.sparse_coo:
            data = data.coalesce()
            mask = mask.coalesce()
            if data._nnz() != mask._nnz():
                data = _sparse_coo_where(mask, data, torch.tensor(0))
        elif data.layout == torch.sparse_csr:
            if data._nnz() != mask._nnz():
                data = _sparse_csr_where(mask, data, torch.tensor(0))

        # Have to pick awkward names to not conflict with existing fields such as data
        self._masked_data = data.clone()
        self._masked_mask = mask.clone()

    def _validate_members(self):
        data = self._masked_data
        mask = self.get_mask()
        if type(data) is not type(mask):
            raise TypeError(
                f"data and mask must have the same type. Got {type(data)} and {type(mask)}"
            )
        if data.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:
            raise TypeError(f"data layout of {data.layout} is not supported.")
        if data.layout == torch.sparse_coo:
            if not _tensors_match(data.indices(), mask.indices(), exact=True):
                raise ValueError(
                    "data and mask are both sparse COO tensors but do not have the same indices."
                )
        elif data.layout == torch.sparse_csr:
            if not _tensors_match(
                data.crow_indices(), mask.crow_indices(), exact=True
            ) or not _tensors_match(data.col_indices(), mask.col_indices(), exact=True):
                raise ValueError(
                    "data and mask are both sparse CSR tensors but do not share either crow or col indices."
                )
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype bool.")
        if not (
            data.dtype == torch.float16
            or data.dtype == torch.float32
            or data.dtype == torch.float64
            or data.dtype == torch.bool
            or data.dtype == torch.int8
            or data.dtype == torch.int16
            or data.dtype == torch.int32
            or data.dtype == torch.int64
        ):
            raise TypeError(f"{data.dtype} is not supported in MaskedTensor.")
        if data.dim() != mask.dim():
            raise ValueError("data.dim() must equal mask.dim()")
        if data.size() != mask.size():
            raise ValueError("data.size() must equal mask.size()")

    def __init__(self, data, mask, requires_grad=False):
        self._preprocess_data(data, mask)
        self._validate_members()

    @staticmethod
    def _from_values(data, mask):
        """Differentiable constructor for MaskedTensor"""

        class Constructor(torch.autograd.Function):
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, data, mask):
                return MaskedTensor(data, mask)

            @staticmethod
            # pyrefly: ignore [bad-override]
            def backward(ctx, grad_output):
                return grad_output, None

        result = Constructor.apply(data, mask)
        return result

    def _set_data_mask(self, data, mask):
        self._masked_data = data
        self._masked_mask = mask
        self._validate_members()

    def __repr__(self):  # type: ignore[override]
        formatter = "{0:8.4f}"
        if self.dim() == 0:
            scalar_data = self.get_data().item()
            data_formatted = (
                formatter.format(scalar_data)
                if isinstance(scalar_data, float)
                else str(scalar_data)
            )
            if not self.get_mask().item():
                data_formatted = "--"
            return (
                "MaskedTensor("
                + data_formatted
                + ", "
                + str(self.get_mask().item())
                + ")"
            )
        s = _masked_tensor_str(self.get_data(), self.get_mask(), formatter)
        s = "\n".join("  " + si for si in s.split("\n"))
        return "MaskedTensor(\n" + s + "\n)"

    # Seems like this needs to be defined before torch_dispatch to work
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        from ._ops_refs import _MASKEDTENSOR_FUNCTION_TABLE

        if func in _MASKEDTENSOR_FUNCTION_TABLE:
            return _MASKEDTENSOR_FUNCTION_TABLE[func](*args, **kwargs)

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return ret
            else:
                return torch._tensor._convert(ret, cls)

    @classmethod
    def unary(cls, fn, data, mask):
        return MaskedTensor(fn(data), mask)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):  # type: ignore[override]
        func = func.overloadpacket

        from ._ops_refs import _MASKEDTENSOR_DISPATCH_TABLE

        if func in _MASKEDTENSOR_DISPATCH_TABLE:
            return _MASKEDTENSOR_DISPATCH_TABLE[func](*args, **kwargs)

        msg = (
            f"{func.__name__} is not implemented in __torch_dispatch__ for MaskedTensor.\n"
            "If you would like this operator to be supported, please file an issue for a feature request at "
            "https://github.com/pytorch/maskedtensor/issues with a minimal reproducible code snippet.\n"
            "In the case that the semantics for the operator are not trivial, it would be appreciated "
            "to also include a proposal for the semantics."
        )
        warnings.warn(msg, stacklevel=2)
        return NotImplemented

    def __lt__(self, other):
        if is_masked_tensor(other):
            return MaskedTensor(self.get_data() < _get_data(other), self.get_mask())
        return MaskedTensor(self.get_data() < other, self.get_mask())

    def to_tensor(self, value):
        return self.get_data().masked_fill(~self.get_mask(), value)

    def get_data(self):
        class GetData(torch.autograd.Function):
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, self):
                return self._masked_data.detach()

            @staticmethod
            # pyrefly: ignore [bad-override]
            def backward(ctx, grad_output):
                if is_masked_tensor(grad_output):
                    return grad_output
                return MaskedTensor(grad_output, self.get_mask())

        return GetData.apply(self)

    def get_mask(self):
        return self._masked_mask

    def is_sparse_coo(self):
        return self.layout == torch.sparse_coo

    def is_sparse_csr(self):  # type: ignore[override]
        return self.layout == torch.sparse_csr

    # Update later to support more sparse layouts
    @property
    def is_sparse(self):  # type: ignore[override]
        return self.is_sparse_coo() or self.is_sparse_csr()
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from collections.abc import Callable
from functools import partial
from typing import Any, TYPE_CHECKING

import torch

from .binary import _apply_native_binary, NATIVE_BINARY_FNS, NATIVE_INPLACE_BINARY_FNS
from .core import (
    _get_data,
    _masks_match,
    _maybe_get_mask,
    is_masked_tensor,
    MaskedTensor,
)
from .passthrough import _apply_pass_through_fn, PASSTHROUGH_FNS
from .reductions import (
    _apply_reduction,
    NATIVE_REDUCE_FNS,
    TENSOR_REDUCE_FNS,
    TORCH_REDUCE_FNS,
)
from .unary import _apply_native_unary, NATIVE_INPLACE_UNARY_FNS, NATIVE_UNARY_FNS


if TYPE_CHECKING:
    from torch._ops import OpOverload


__all__ = []  # type: ignore[var-annotated]


def _check_args_kwargs_length(
    args, kwargs, error_prefix, len_args=None, len_kwargs=None
):
    if len_args is not None and len_args != len(args):
        raise ValueError(
            f"{error_prefix}: len(args) must be {len_args} but got {len(args)}"
        )
    if len_kwargs is not None and len_kwargs != len(kwargs):
        raise ValueError(
            f"{error_prefix}: len(kwargs) must be {len_kwargs} but got {len(kwargs)}"
        )


class _MaskedContiguous(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedContiguous forward: input must be a MaskedTensor.")

        if input.is_contiguous():
            return input

        data = input.get_data()
        mask = input.get_mask()

        return MaskedTensor(data.contiguous(), mask.contiguous())

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        return grad_output


class _MaskedToDense(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedToDense forward: input must be a MaskedTensor.")

        if input.layout == torch.strided:
            return input

        ctx.layout = input.layout
        data = input.get_data()
        mask = input.get_mask()

        return MaskedTensor(data.to_dense(), mask.to_dense())

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        layout = ctx.layout

        if layout == torch.sparse_coo:
            return grad_output.to_sparse_coo()
        elif layout == torch.sparse_csr:
            return grad_output.to_sparse_csr()
        elif layout == torch.strided:
            return grad_output.to_dense()
        raise ValueError(f"to_dense: Unsupported input layout: {layout}")


class _MaskedToSparse(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedToSparse forward: input must be a MaskedTensor.")

        # Following the convention from sparse tensors that to_sparse always means that we convert to sparse_coo
        if input.layout == torch.sparse_coo:
            return input

        data = input.get_data()
        mask = input.get_mask()
        sparse_mask = mask.to_sparse_coo().coalesce()
        sparse_data = data.sparse_mask(sparse_mask)

        return MaskedTensor(sparse_data, sparse_mask)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        return grad_output.to_dense()


class _MaskedToSparseCsr(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedToSparseCsr forward: input must be a MaskedTensor.")

        if input._masked_data.ndim != 2:
            raise ValueError(
                f"Only 2D tensors can be converted to the SparseCsr layout but got shape: {input._masked_data.size()}"
            )

        if input.layout == torch.sparse_csr:
            return input

        data = input.get_data()
        mask = input.get_mask()
        sparse_mask = mask.to_sparse_csr()
        sparse_data = data.sparse_mask(sparse_mask)

        return MaskedTensor(sparse_data, sparse_mask)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        return grad_output.to_dense()


class _MaskedWhere(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, cond, self, other):
        ctx.mark_non_differentiable(cond)
        ctx.save_for_backward(cond)
        return torch.ops.aten.where(cond, self, other)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        (cond,) = ctx.saved_tensors

        def masked_out_like(mt):
            return MaskedTensor(mt.get_data(), torch.zeros_like(mt.get_mask()).bool())

        return (
            None,
            torch.ops.aten.where(cond, grad_output, masked_out_like(grad_output)),
            torch.ops.aten.where(cond, masked_out_like(grad_output), grad_output),
        )


_MASKEDTENSOR_FUNCTION_TABLE = {}

_function_fn_apply_map = {
    (
        tuple(NATIVE_REDUCE_FNS),
        tuple(TORCH_REDUCE_FNS),
        tuple(TENSOR_REDUCE_FNS),
    ): _apply_reduction,
}

for fn_map_list, apply_fn in _function_fn_apply_map.items():
    for fn_map in fn_map_list:
        for fn in fn_map:
            _MASKEDTENSOR_FUNCTION_TABLE[fn] = partial(apply_fn, fn)


def register_function_func(ops):
    """
    Used for registering a new __torch_function__ function to MaskedTensor
    Called via _MASKEDTENSOR_FUNCTION_TABLE[func](*args, **kwargs)

    The code to register a new function looks like:

    @register_function_func(list_of_ops)
    def foo(func, *args, **kwargs):
        <implementation>
    """

    def wrapper(func):
        for op in ops:
            _MASKEDTENSOR_FUNCTION_TABLE[op] = partial(func, op)

    return wrapper


@register_function_func(NATIVE_REDUCE_FNS + TORCH_REDUCE_FNS + TENSOR_REDUCE_FNS)
def _general_function_reductions(func, *args, **kwargs):
    return _apply_reduction(func, *args, **kwargs)


@register_function_func([torch.Tensor.where, torch.where])
def _function_where(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, "__torch_function__, torch.where", len_args=3, len_kwargs=0
    )
    return _MaskedWhere.apply(*args)


@register_function_func([torch.Tensor.contiguous])
def _function_contiguous(func, *args, **kwargs):
    return _MaskedContiguous.apply(args[0])


@register_function_func([torch.Tensor.to_dense])
def _function_to_dense(func, *args, **kwargs):
    return _MaskedToDense.apply(args[0])


@register_function_func([torch.Tensor.to_sparse])
def _function_to_sparse(func, *args, **kwargs):
    return _MaskedToSparse.apply(args[0])


@register_function_func([torch.Tensor.to_sparse_csr])
def _function_to_sparse_csr(func, *args, **kwargs):
    return _MaskedToSparseCsr.apply(args[0])


_MASKEDTENSOR_DISPATCH_TABLE: dict["OpOverload", Callable[..., Any]] = {}


def register_dispatch_func(aten_ops):
    """
    Used for registering a new __torch_dispatch__ function to MaskedTensor
    Called via _MASKEDTENSOR_DISPATCH_TABLE[func](*args, **kwargs)

    The code to register a new function looks like:

    @register_dispatch_func(list_of_ops)
    def foo(func, *args, **kwargs):
        <implementation>
    """

    def wrapper(func):
        for aten_op in aten_ops:
            _MASKEDTENSOR_DISPATCH_TABLE[aten_op] = partial(func, aten_op)

    return wrapper


@register_dispatch_func(NATIVE_REDUCE_FNS + TORCH_REDUCE_FNS + TENSOR_REDUCE_FNS)
def _general_reduction(func, *args, **kwargs):
    return _apply_reduction(func, *args, **kwargs)


@register_dispatch_func(PASSTHROUGH_FNS)
def _general_passthrough(func, *args, **kwargs):
    return _apply_pass_through_fn(func, *args, **kwargs)


@register_dispatch_func(NATIVE_UNARY_FNS + NATIVE_INPLACE_UNARY_FNS)
def _general_unary(func, *args, **kwargs):
    return _apply_native_unary(func, *args, **kwargs)


@register_dispatch_func(NATIVE_BINARY_FNS + NATIVE_INPLACE_BINARY_FNS)
def _general_binary(func, *args, **kwargs):
    return _apply_native_binary(func, *args, **kwargs)


@register_dispatch_func([torch.ops.aten.stride])
def stride(func, *args, **kwargs):
    return None


@register_dispatch_func([torch.ops.aten.sym_stride])
def sym_stride(func, *args, **kwargs):
    return None


@register_dispatch_func([torch.ops.prim.layout])
def layout(func, *args, **kwargs):
    return _get_data(args[0]).layout


@register_dispatch_func(
    [torch.ops.aten.is_contiguous, torch.ops.aten.sym_is_contiguous]
)
def is_contiguous(func, *args, **kwargs):
    data = _get_data(args[0])
    if data.is_sparse:
        raise ValueError("MaskedTensors with sparse data do not have is_contiguous")
    return func(data, *args[1:], **kwargs)


@register_dispatch_func([torch.ops.aten.is_strides_like_format])
def is_strides_like_format(func, *args, **kwargs):
    data = _get_data(args[0])
    if data.is_sparse:
        raise ValueError(
            "MaskedTensors with sparse data do not have is_strides_like_format"
        )
    return func(data, *args[1:], **kwargs)


@register_dispatch_func([torch.ops.aten.is_non_overlapping_and_dense])
def is_non_overlapping_and_dense(func, *args, **kwargs):
    data = _get_data(args[0])
    if data.is_sparse:
        raise ValueError(
            "MaskedTensors with sparse data do not have is_non_overlapping_and_dense"
        )
    return func(data, *args[1:], **kwargs)


@register_dispatch_func([torch.ops.aten.contiguous])
def contiguous(func, *args, **kwargs):
    if _get_data(args[0]).is_sparse:
        raise ValueError("MaskedTensors with sparse data do not have contiguous")
    return _MaskedContiguous.apply(args[0])


@register_dispatch_func([torch.ops.aten.new_empty_strided])
def new_empty_strided(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=3)
    data = _get_data(args[0])
    mask = _maybe_get_mask(args[0])
    if tuple(args[1]) != tuple(data.size()):
        raise ValueError(
            f"__torch_dispatch__, {func}: args[1] expected to be the same as data.size()"
        )
    if tuple(args[2]) != tuple(data.stride()):
        raise ValueError(
            f"__torch_dispatch__, {func}: args[2] expected to be the same as data.stride()"
        )
    return MaskedTensor(func(data, args[1], args[2], **kwargs), mask)


@register_dispatch_func([torch.ops.aten._local_scalar_dense])
def _local_scalar_dense(func, *args, **kwargs):
    if not _maybe_get_mask(args[0]):
        raise ValueError(f"__torch_dispatch__, {func}: expected a mask tensor")
    return torch.ops.aten._local_scalar_dense(_get_data(args[0]))


@register_dispatch_func([torch.ops.aten.detach, torch.ops.aten.clone])
def _apply_fn_on_data(func, *args, **kwargs):
    return MaskedTensor(func(_get_data(args[0])), _maybe_get_mask(args[0]))


@register_dispatch_func([torch.ops.aten._to_copy])
def _to_copy(func, *args, **kwargs):
    new_data = func(_get_data(args[0]), *args[1:], **kwargs)
    cloned_kwargs = kwargs.copy()
    cloned_kwargs["dtype"] = torch.bool
    new_mask = func(_maybe_get_mask(args[0]), *args[1:], **cloned_kwargs)
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._softmax])
def _softmax(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=3, len_kwargs=0
    )
    data = _get_data(args[0])
    mask = _maybe_get_mask(args[0])
    result_data = torch.ops.aten._masked_softmax(data, ~mask, args[1], 2)
    return MaskedTensor(result_data, mask)


@register_dispatch_func([torch.ops.aten.ones_like])
def ones_like(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1)
    result_data = func(_get_data(args[0]), **kwargs)
    return MaskedTensor(result_data, _maybe_get_mask(args[0]))


@register_dispatch_func([torch.ops.aten._softmax_backward_data])
def _softmax_backward_data(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=4)
    grad, output, dim, _input_dtype = args
    if is_masked_tensor(grad) and is_masked_tensor(output):
        if not _masks_match(grad, output):
            raise ValueError(
                f"__torch_dispatch__, {func}: expected the masks of grad and output to match"
            )
        grad_data = _get_data(grad)
        new_grad_data = torch.ops.aten._masked_softmax_backward(
            grad_data,
            _get_data(output),
            ~_maybe_get_mask(grad),
            dim % grad_data.ndim,
        )
        res = MaskedTensor(new_grad_data, _maybe_get_mask(grad))
        return res
    else:
        raise ValueError(
            f"__torch_dispatch__, {func}: grad and output must both be MaskedTensors"
        )


@register_dispatch_func([torch.ops.aten.copy_])
def copy_(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=2)
    if not _masks_match(_maybe_get_mask(args[0]), _maybe_get_mask(args[1])):
        raise ValueError("args[0] mask and args[1] mask must match but do not")
    func(_get_data(args[0]), _get_data(args[1]))
    return args[0]


@register_dispatch_func([torch.ops.aten.where])
def where(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=3, len_kwargs=0
    )
    if not torch.is_tensor(args[0]):
        raise ValueError(f"__torch_dispatch__, {func}: expected args[0] to be a tensor")
    mx = args[1]
    my = args[2]
    if not is_masked_tensor(mx):
        mx = MaskedTensor(mx, torch.ones_like(mx, dtype=torch.bool))
    if not is_masked_tensor(my):
        my = MaskedTensor(my, torch.ones_like(my, dtype=torch.bool))
    new_data = func(args[0], mx.get_data(), my.get_data())
    new_mask = func(args[0], mx.get_mask(), my.get_mask())
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._to_sparse])
def _to_sparse(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    if not torch.is_tensor(args[0]):
        raise TypeError(f"__torch_dispatch__, {func}: expected args[0] to be a tensor")
    mt = args[0]
    if not is_masked_tensor(mt):
        mt = MaskedTensor(mt, torch.ones_like(mt, dtype=torch.bool))
    if mt.is_sparse_coo():
        return mt
    new_mask = func(_maybe_get_mask(args[0])).coalesce()
    new_data = _get_data(args[0]).sparse_mask(new_mask)
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._to_sparse_csr])
def _to_sparse_csr(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    if not torch.is_tensor(args[0]):
        raise ValueError(f"__torch_dispatch__, {func}: expected args[0] to be a tensor")
    mt = args[0]
    if not is_masked_tensor(mt):
        mt = MaskedTensor(mt, torch.ones_like(mt).bool())
    if mt.is_sparse_csr():
        return mt
    new_mask = func(_maybe_get_mask(args[0]))
    new_data = _get_data(args[0]).sparse_mask(new_mask)
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._to_dense])
def _to_dense(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    if not torch.is_tensor(args[0]):
        raise ValueError(f"__torch_dispatch__, {func}: expected args[0] to be a tensor")
    mt = args[0]
    if not is_masked_tensor(mt):
        mt = MaskedTensor(mt, torch.ones_like(mt).bool())
    new_data = func(_get_data(args[0]))
    new_mask = func(_maybe_get_mask(args[0]))
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._indices])
def _indices(func, *args, **kwargs):
    # Assumes data is sparse
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    data = _get_data(args[0]).indices()
    return MaskedTensor(data, torch.ones_like(data).bool())


@register_dispatch_func([torch.ops.aten._values])
def _values(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    data = _get_data(args[0]).values()
    return MaskedTensor(data, torch.ones_like(data).bool())


@register_dispatch_func([torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors])
def _sparse_coo_tensor_with_dims_and_tensors(func, *args, **kwargs):
    new_args = list(args)
    if is_masked_tensor(args[-1]):
        new_args[-1] = args[-1].get_data()
    if is_masked_tensor(args[-2]):
        new_args[-2] = args[-2].get_data()

    new_data = func(*new_args, **kwargs)
    new_args[-1] = torch.ones_like(new_args[-1])
    new_mask = func(*new_args, **kwargs).bool()

    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten.is_same_size])
def is_same_size(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=2)
    return _get_data(args[0]).is_same_size(_get_data(args[1]))


@register_dispatch_func([torch.ops.aten._is_any_true])
def _is_any_true(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    data = _get_data(args[0])
    mask = _maybe_get_mask(args[0])
    if mask is None:
        raise ValueError(
            f"__torch_dispatch__, {func}: expected args[0] to be a MaskedTensor"
        )
    if data.dtype != torch.bool:
        raise ValueError(f"__torch_dispatch__, {func}: expected a boolean tensor")
    if data.is_sparse:
        raise ValueError(f"MaskedTensors with sparse data do not have {func}")

    return MaskedTensor(func(data & mask), torch.tensor(True))
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
"""
These are functions that should simply be applied to both mask and data.
Take select or stack as an example. This operation can be applied to
both the mask and data of a MaskedTensor and the result wrapped into
a new MaskedTensor as a result.
"""

import torch

from .core import _map_mt_args_kwargs, _wrap_result


__all__ = []  # type: ignore[var-annotated]


PASSTHROUGH_FNS = [
    torch.ops.aten.select,
    torch.ops.aten.transpose,
    torch.ops.aten.split,
    torch.ops.aten.t,
    torch.ops.aten.slice,
    torch.ops.aten.slice_backward,
    torch.ops.aten.select_backward,
    torch.ops.aten.index,
    torch.ops.aten.expand,
    torch.ops.aten.view,
    torch.ops.aten._unsafe_view,
    torch.ops.aten._reshape_alias,
    torch.ops.aten.cat,
    torch.ops.aten.unsqueeze,
    torch.ops.aten.unfold,
    torch.ops.aten.unfold_backward,
    torch.ops.aten.im2col,
    torch.ops.aten.col2im,
    torch.ops.aten.stack,
]


def _is_pass_through_fn(fn):
    return fn in PASSTHROUGH_FNS


def _apply_pass_through_fn(fn, *args, **kwargs):
    data_args, data_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_data())
    result_data = fn(*data_args, **data_kwargs)
    mask_args, mask_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_mask())
    result_mask = fn(*mask_args, **mask_kwargs)
    return _wrap_result(result_data, result_mask)
<|endoftext|>from __future__ import annotations

import inspect
import logging
import sys
import traceback
import types
from collections import namedtuple
from contextlib import nullcontext
from typing import Any, cast, TYPE_CHECKING, TypeVar

import sympy

import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.convert_frame import CaptureOutput, fullgraph_capture, get_traced_fn
from torch._dynamo.decorators import disable as dynamo_disable
from torch._dynamo.eval_frame import argument_names, check_user_input_output
from torch._dynamo.exc import UserErrorType
from torch._dynamo.source import GetItemSource
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._export.utils import _compiling_state_context
from torch._guards import detect_fake_mode, Source, TracingContext
from torch.export.dynamic_shapes import _RelaxedConstraint, Constraint
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    DimDynamic,
    StatelessSymbolicContext,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch._dynamo.output_graph import OutputReturnInfo
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.export._trace import _DynamicShapesInput
    from torch.fx import Node
    from torch.fx.node import Argument, Target

from torch.fx.experimental._spec_binding import _bind_spec_to_args
from torch.fx.experimental.dynamic_spec import (
    _coerce_to_shapes_spec,
    IntermediateSpec,
    ParamsSpec,
    ShapesSpec,
)


T = TypeVar("T")
log = logging.getLogger(__name__)


def post_process_error_msg(
    constraint_violation_error: ConstraintViolationError,
    func: Callable[..., Any],
    args: Any,
    kwargs: Any,
) -> ConstraintViolationError:
    """
    Because we trace a different callable, the sources are all messed up.
    Manually patch them so the error message looks correct.
    """
    from torch.export._unlift import _get_input_paths, _replace_sources

    orig_sig = inspect.signature(func)
    flat_input_paths = _get_input_paths((args, kwargs), orig_sig)
    if constraint_violation_error.args:
        constraint_violation_error.args = (
            _replace_sources(constraint_violation_error.args[0], flat_input_paths),
        )
    return constraint_violation_error


EXPORT_ROOT_REPLACEMENTS = [
    ("__export_root_", "_"),
    ("_export_root.", ""),
    ("._export_root", ""),
]


def clean_export_root_string(text: str) -> str:
    """Generic utility to clean export_root patterns from strings."""
    result = text
    for pattern, replacement in EXPORT_ROOT_REPLACEMENTS:
        result = result.replace(pattern, replacement)
    return result


def clean_nn_module_stack_and_source_fn(
    graph_module: torch.fx.GraphModule, is_inline_builtin: bool = False
) -> torch.fx.GraphModule:
    """
    Clean up nn_module_stack metadata by removing export_root references.

    Removes the _export_root module references from nn_module_stack metadata
    in graph nodes, which are artifacts from the export process. Fixes two patterns:

    1. Keys: Removes "__export_root_" and "__modules['_export_root']_" prefixes
       - Normal case: "L__self____export_root_child" -> "L__self__child"
       - inline_builtin case: Uses numeric ID strings like "140468831433840"

    2. Values: Removes "._export_root" and "._modules['_export_root']" from child names
       e.g., "L['self']._export_root.child" -> "L['self'].child"
       e.g., "L['self']._modules['_export_root'].child" -> "L['self'].child"

    Also removes the root export entry "L__self____export_root" entirely.

    Args:
        graph_module: The GraphModule to clean up
        is_inline_builtin: If True, keys are numeric ID strings and self references
                          (L['self']) are filtered out

    Returns:
        The cleaned GraphModule (modified in-place)
    """

    def _process_nn_module_stack(
        nn_module_stack: dict[str, tuple[str, T]],
    ) -> dict[str, tuple[str, T]]:
        if "L__self____export_root" in nn_module_stack:
            del nn_module_stack["L__self____export_root"]

        # Clean up remaining entries
        cleaned_stack = {}
        for key, (child_name, child_class) in nn_module_stack.items():
            # Clean key by removing export_root patterns
            clean_key = clean_export_root_string(key)

            # Clean child_name by removing export_root patterns
            clean_name = clean_export_root_string(child_name)

            # Skip self reference for inline builtin case
            if is_inline_builtin and clean_name == "L['self']":
                continue

            cleaned_stack[clean_key] = (clean_name, child_class)
        return cleaned_stack

    def _process_source_fn(source_fn_stack: Iterable[T]) -> Iterable[T]:
        cleaned_stack = []
        for item in source_fn_stack:
            if isinstance(item, tuple) and len(item) == 2:
                name, cls = item
                if isinstance(name, str):
                    clean_name = clean_export_root_string(name)
                    cleaned_stack.append((clean_name, cls))
                else:
                    cleaned_stack.append(item)
            else:
                # pyrefly: ignore [bad-argument-type]
                cleaned_stack.append(item)
        # pyrefly: ignore [bad-return]
        return cleaned_stack

    for node in graph_module.graph.nodes:
        if "nn_module_stack" in node.meta:
            node.meta["nn_module_stack"] = _process_nn_module_stack(
                node.meta["nn_module_stack"].copy()
            )

        source_fn_stack = node.meta.get("source_fn_stack", None)
        if source_fn_stack:
            node.meta["source_fn_stack"] = _process_source_fn(source_fn_stack.copy())

    if "dynamo_flat_name_to_original_fqn" in graph_module.meta:
        # Clean up flat name to original fqn mapping
        clean_name_to_original_fqn = {}
        for flat_name, original_fqn in graph_module.meta[
            "dynamo_flat_name_to_original_fqn"
        ].items():
            clean_name_to_original_fqn[clean_export_root_string(flat_name)] = (
                clean_export_root_string(original_fqn)
            )
        graph_module.meta["dynamo_flat_name_to_original_fqn"] = (
            clean_name_to_original_fqn
        )

    return graph_module


def clean_export_root(graph_module: torch.fx.GraphModule) -> None:
    """Remove export_root artifacts from FX graph in-place"""

    # Unlike getattr node, call_module can be invoked multiple times
    # In those cases, we should fix all invocations of call_module
    clean_named_module_map: dict[str, str] = {}

    # Update get_attr nodes in-place
    for node in graph_module.graph.nodes:
        if node.op == "get_attr":
            old_target = node.target
            new_target = clean_export_root_string(old_target)
            if new_target != old_target:
                node.target = new_target
                if not hasattr(graph_module, old_target):
                    raise AssertionError(
                        f"graph_module missing attribute {old_target!r}"
                    )
                # Move the parameter to the new name
                param = torch.fx.graph_module._get_attr(graph_module, old_target)
                torch.fx.graph_module._assign_attr(param, graph_module, new_target)
                torch.fx.graph_module._del_attr(graph_module, old_target)
        # Dynamo will only have one nested level
        if node.op == "call_module":
            old_target = node.target
            if not isinstance(old_target, str):
                raise AssertionError(
                    f"expected old_target to be str, got {type(old_target)}"
                )
            new_target = clean_export_root_string(old_target)
            if not isinstance(new_target, str):
                raise AssertionError(
                    f"expected new_target to be str, got {type(new_target)}"
                )
            new_name = clean_export_root_string(node.name)
            if new_target == old_target:
                continue

            # if this module has already been cleaned before, just lookup from map.
            if old_target in clean_named_module_map:
                node.target = clean_named_module_map[old_target]
                node.name = new_name
                continue
            target = graph_module.get_submodule(old_target)
            graph_module.delete_submodule(old_target)
            graph_module.add_submodule(new_target, target)
            node.target = new_target
            node.name = new_name
            clean_named_module_map[old_target] = new_target


class ModuleToTrace(torch.nn.Module):
    def __init__(self, foo: Any, in_spec: Any) -> None:
        super().__init__()
        self._export_root = foo
        self.in_spec = in_spec

    def forward(self, *flat_args: Any) -> ExportTracerOutput:
        args, kwargs = pytree.tree_unflatten(flat_args, self.in_spec)
        res = self._export_root(*args, **kwargs)
        out_flat, out_spec = pytree.tree_flatten(res)
        return ExportTracerOutput(out_flat, out_spec)


ExportTracerOutput = namedtuple("ExportTracerOutput", ["flat_args", "out_spec"])


# mypy: disable-error-code="no-untyped-def,var-annotated,assignment,index,operator"
class DynamoGraphTransformer(torch.fx.Transformer):
    """Graph transformer for dynamo export that flattens inputs/outputs without complex matching."""

    def __init__(
        self,
        module: torch.fx.GraphModule,
        flat_inputs: list[Any],
        flat_args_dynamic_dims: list[set[int]],
        graph_input_order: dict[int, int],
        graph_output_map: dict[int, OutputReturnInfo],
        fake_mode: FakeTensorMode | None = None,
        graph_inputs: dict[int, Source | None] | None = None,
    ) -> None:
        super().__init__(module)

        if len(flat_args_dynamic_dims) != len(flat_inputs):
            raise AssertionError(
                f"flat_args_dynamic_dims length ({len(flat_args_dynamic_dims)}) "
                f"must match flat_inputs length ({len(flat_inputs)})"
            )

        self.flat_inputs = flat_inputs
        self.flat_args_dynamic_dims = flat_args_dynamic_dims
        self.graph_input_order = graph_input_order
        self.graph_output_map = graph_output_map
        self.fake_mode = fake_mode
        self.graph_inputs = graph_inputs or {}

        # Get original placeholders and output
        self.placeholders = [n for n in module.graph.nodes if n.op == "placeholder"]
        self.output_node = next(n for n in module.graph.nodes if n.op == "output")

        # Create new flattened input placeholders
        self.new_input_nodes: dict[int, torch.fx.Node] = {}
        self._create_flattened_inputs()

        # Iterator for replacing old placeholders
        self.old_to_new_mapping = {}
        self._create_placeholder_mapping()

    def _create_flattened_inputs(self) -> None:
        """Create new placeholder nodes for flattened inputs with proper fake tensors."""
        for i in range(len(self.flat_inputs)):
            placeholder = super().placeholder(f"arg_{i}", (), {})

            # Check if this user input (index i) maps to a graph placeholder
            if i in self.graph_input_order:
                # graph_input_order[i] gives us which graph placeholder this user input corresponds to
                graph_placeholder_idx = self.graph_input_order[i]
                if graph_placeholder_idx < len(self.placeholders):
                    orig_placeholder = self.placeholders[graph_placeholder_idx]
                    # Copy other metadata but not "val" yet
                    for key, value in orig_placeholder.meta.items():
                        if key != "val":
                            placeholder.node.meta[key] = value

            # Always ensure we have proper "val" metadata from fake tensor
            if self.fake_mode is not None and isinstance(
                self.flat_inputs[i], torch.Tensor
            ):
                placeholder.node.meta["val"] = self.fake_mode.from_tensor(
                    self.flat_inputs[i],
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[
                            (
                                DimDynamic.DYNAMIC
                                if d in self.flat_args_dynamic_dims[i]
                                else DimDynamic.STATIC
                            )
                            for d in range(len(self.flat_inputs[i].shape))
                        ],
                        constraint_sizes=[None] * len(self.flat_inputs[i].shape),
                    ),
                )
            elif hasattr(self.flat_inputs[i], "val"):  # _IntWrapper case
                placeholder.node.meta["val"] = self.flat_inputs[i].val
            else:
                placeholder.node.meta["val"] = self.flat_inputs[i]

            # pyrefly: ignore [unsupported-operation]
            self.new_input_nodes[i] = placeholder

    def _create_placeholder_mapping(self) -> None:
        """Create mapping from old placeholders to new ones."""
        # graph_input_order maps: user_input_index -> graph_placeholder_index
        # We need to create: old_graph_placeholder -> new_user_input_placeholder
        for user_input_idx, graph_placeholder_idx in self.graph_input_order.items():
            if graph_placeholder_idx < len(self.placeholders):
                old_placeholder = self.placeholders[graph_placeholder_idx]
                new_placeholder = self.new_input_nodes[user_input_idx]
                self.old_to_new_mapping[old_placeholder] = new_placeholder

    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Replace old placeholders with new flattened ones."""
        # Return the corresponding new placeholder
        if self.current_node in self.old_to_new_mapping:
            new_arg = self.old_to_new_mapping[self.current_node]

            # Copy over additional metadata from current node, but don't overwrite "val"
            for key in ["tensor_dict", "example_value", "unbacked_bindings"]:
                if key in self.current_node.meta:
                    new_arg.node.meta[key] = self.current_node.meta[key]

            # Only copy "val" if we don't already have a good one
            if "val" in self.current_node.meta and "val" not in new_arg.node.meta:
                new_arg.node.meta["val"] = self.current_node.meta["val"]

            return new_arg
        else:
            # Convert captured objects (e.g., opaque objects from closures) to
            # get_attr nodes
            placeholder_idx = self.placeholders.index(self.current_node)
            if placeholder_idx in self.graph_inputs:
                source = self.graph_inputs[placeholder_idx]
                if not isinstance(source, GetItemSource):
                    example_val = self.current_node.meta.get(
                        "val"
                    ) or self.current_node.meta.get("example_value")
                    if example_val is not None:
                        attr_name = f"_captured_{placeholder_idx}"
                        if isinstance(example_val, torch.Tensor):
                            self.module.register_buffer(attr_name, example_val)
                        else:
                            setattr(self.module, attr_name, example_val)
                        result = self.tracer.create_proxy("get_attr", attr_name, (), {})
                        result.node.meta = self.current_node.meta.copy()
                        result.node.meta["val"] = example_val
                        return result
            return super().placeholder(target, args, kwargs)

    def output(
        self, target: Target, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> Any:
        """Transform output according to graph_output_map."""
        original_outputs = args[0]

        # Build new output list based on graph_output_map
        new_outputs = []
        for i in sorted(self.graph_output_map.keys()):
            output_type, val = self.graph_output_map[i]

            if output_type == "graph_out":
                new_outputs.append(original_outputs[cast(int, val)])
            elif output_type == "input":
                input_idx = cast(GetItemSource, val).index
                new_outputs.append(self.new_input_nodes[input_idx])
            elif output_type == "constant":
                new_outputs.append(val)

        return super().output(target, (tuple(new_outputs),), {})

    def run_node(self, n: Node) -> Any:
        """Run node transformation and preserve metadata."""
        self.current_node = n
        result = super().run_node(n)

        # Copy important metadata
        if hasattr(result, "node") and result.node is not n:
            for key in ["val", "example_value", "unbacked_bindings"]:
                if key in n.meta:
                    result.node.meta[key] = n.meta[key]

            # Preserve node names (except output)
            if n.op != "output" and hasattr(n, "name"):
                result.node._rename(n.name)

        return result

    def transform(self) -> torch.fx.GraphModule:
        """Perform the graph transformation and copy module metadata."""
        result_gm = super().transform()

        # Copy module metadata like the original implementation
        if hasattr(self.module, "meta"):
            # pyrefly: ignore [unsupported-operation]
            if "dynamo_flat_name_to_original_fqn" in self.module.meta:
                # pyrefly: ignore [bad-index]
                result_gm.meta["dynamo_flat_name_to_original_fqn"] = self.module.meta[
                    # pyrefly: ignore [bad-index]
                    "dynamo_flat_name_to_original_fqn"
                ]
            # pyrefly: ignore [unsupported-operation]
            if "dynamo_compile_id" in self.module.meta:
                # pyrefly: ignore [bad-index]
                result_gm.meta["dynamo_compile_id"] = self.module.meta[
                    # pyrefly: ignore [bad-index]
                    "dynamo_compile_id"
                ]

        return result_gm


def _suggest_or_raise_constraint_violation(
    module_to_trace: torch.nn.Module,
    orig_callable: Callable[..., Any],
    fake_mode: FakeTensorMode | None,
    graph_capture_output: CaptureOutput,
    args: Any,
    kwargs: Any,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
) -> None:
    constraint_violation_error = None
    try:
        # Check if we have any constraint violations
        fn, _ = get_traced_fn(module_to_trace)
        graph_capture_output.graph_capture_output.build_guards(fn.__code__)
    except ConstraintViolationError as e:
        constraint_violation_error = e

    if (
        (shape_env := getattr(fake_mode, "shape_env", None)) is not None
        and (dim_constraints := shape_env.dim_constraints) is not None
        and not isinstance(
            module_to_trace.forward,
            torch._ops.OpOverloadPacket | torch._ops.OpOverload,
        )
    ):
        dim_constraints.solve()

        forced_specializations = dim_constraints.forced_specializations()

        msg = dim_constraints.prettify_results(
            inspect.signature(orig_callable),  # type: ignore[attr-defined]
            dynamic_shapes,
            constraint_violation_error,
            forced_specializations,
        )
        if constraint_violation_error:
            if constraint_violation_error.args:
                constraint_violation_error.args = (
                    constraint_violation_error.args[0] + msg,
                )
            else:
                constraint_violation_error.args = (msg,)
        else:
            if forced_specializations:
                constraint_violation_error = ConstraintViolationError(msg)
            else:
                log.info(
                    "Summary of dimension constraints:%s",
                    msg,
                )

        # Error if we have any constraints on static values

        for k in shape_env.var_to_range:
            if isinstance(k, sympy.Integer):
                constraint_violation_error = ConstraintViolationError(
                    f"{''.join(traceback.format_list(shape_env.var_to_stack[k]))}\n"
                    "It appears that you're trying to set a constraint on a "
                    f"value which we evaluated to have a static value of {k}. "
                    'Set TORCH_LOGS="+export" for more information.'
                )
    if constraint_violation_error:
        constraint_violation_error = post_process_error_msg(
            constraint_violation_error, orig_callable, args, kwargs
        )
        raise constraint_violation_error


def _normalize_shuffle_graph(shuffle_gm: torch.fx.GraphModule) -> None:
    shuffle_gm.graph.eliminate_dead_code()
    shuffle_gm.recompile()
    for name, buffer in list(shuffle_gm.named_buffers()):
        delattr(shuffle_gm, name)
        setattr(shuffle_gm, name, buffer)


def normalize_graph_module(gm: torch.fx.GraphModule) -> None:
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = node.meta["example_value"]


class InputProcessor:
    def __init__(
        self,
        root: object,
        num_args: int,
        kwarg_names: list[str],
    ) -> None:
        self.root = root
        self.num_args = num_args
        self.kwarg_names = kwarg_names

    def __call__(
        self, inputs: tuple[object, ...]
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        args = inputs
        # pyrefly: ignore [implicit-any]
        kwargs = {}
        if len(args) > self.num_args:
            kwargs = dict(zip(self.kwarg_names, args[self.num_args :]))
            args = args[: self.num_args]
        if self.root is not None:
            if isinstance(self.root, torch.fx.GraphModule):
                if not isinstance(self.root.graph._codegen, _DynamoBytecodeCodeGen):
                    raise AssertionError(
                        f"expected _DynamoBytecodeCodeGen, got {type(self.root.graph._codegen)}"
                    )
                if not hasattr(
                    self.root.graph._codegen.dynamo_bytecode_flatten, "input_processor"
                ):
                    raise AssertionError(
                        "dynamo_bytecode_flatten missing input_processor attribute"
                    )
                if (
                    self.root.graph._codegen.dynamo_bytecode_flatten.input_processor
                    is not self
                ):
                    raise AssertionError(
                        "dynamo_bytecode_flatten.input_processor is not self"
                    )
            args = (self.root, *args)
        return args, kwargs


class Yield(Exception):
    pass


class DynamoBytecodeFlatten:
    def __init__(
        self,
        input_processor: InputProcessor,
        out: CaptureOutput,
        f_globals: dict[str, object],
    ) -> None:
        self.input_processor = input_processor
        self.out = out
        self.f_globals = f_globals
        self.gm_inputs: tuple[Any, ...] | None = None

    @dynamo_disable(reason="do not trace internal dynamo graph capture")  # type: ignore[misc]
    def __call__(self, *inputs: object) -> object:
        def backend_dummy(*example_inputs: object) -> None:
            self.gm_inputs = example_inputs
            raise Yield

        args, kwargs = self.input_processor(inputs)
        try:
            self.out.forward_callable(
                compiled_fn=backend_dummy, extra_globals=self.f_globals
            )(*args, **kwargs)
        except Yield:
            if self.gm_inputs is None:
                raise AssertionError("gm_inputs must not be None after Yield") from None
            return self.gm_inputs
        raise RuntimeError


class DynamoBytecodeUnflatten:
    def __init__(
        self,
        input_processor: InputProcessor,
        out: CaptureOutput,
        f_globals: dict[str, object],
    ) -> None:
        self.input_processor = input_processor
        self.out = out
        self.f_globals = f_globals

    @dynamo_disable(reason="do not trace internal dynamo graph capture")  # type: ignore[misc]
    def __call__(
        self, flat_outs: Sequence[object], inputs: tuple[object, ...]
    ) -> object:
        def backend_dummy(*example_inputs: object) -> Sequence[object]:
            return flat_outs

        args, kwargs = self.input_processor(inputs)
        with torch._C._DisableTorchDispatch():
            results = self.out.forward_callable(
                compiled_fn=backend_dummy, extra_globals=self.f_globals
            )(*args, **kwargs)
        return results


def create_fx_graph_from_captured_output(
    out: CaptureOutput, mod: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> torch.fx.GraphModule:
    if out.backend_input is None:
        raise AssertionError("out.backend_input must not be None")
    backend_input = out.backend_input

    _, root = torch._dynamo.convert_frame.get_traced_fn(mod)

    flat_real_args = pytree.tree_leaves((args, kwargs))
    torch._dynamo.eval_frame.check_user_input_output(
        flat_real_args, UserErrorType.INVALID_INPUT
    )
    f_globals = out.graph_capture_output.f_globals

    graph_module = backend_input.graph_module
    if isinstance(root, torch.nn.Module):
        graph_module._parameters = root._parameters
        graph_module._buffers = root._buffers
        if not all(not hasattr(graph_module, m) for m in root._modules):
            raise AssertionError(
                "graph_module already has attributes conflicting with root._modules"
            )
        graph_module._modules.update(root._modules)
        graph_module._non_persistent_buffers_set = root._non_persistent_buffers_set
        if sys.version_info >= (3, 14):
            import annotationlib  # added in 3.14

            annotations = annotationlib.get_annotations(torch.nn.Module)
        else:
            annotations = getattr(torch.nn.Module, "__annotations__", None)
        for name, value in root.__dict__.items():
            if annotations and name not in annotations:
                graph_module.__dict__[name] = value
        graph_module._forward_hooks = root._forward_hooks.copy()
        graph_module._forward_pre_hooks = root._forward_pre_hooks.copy()
        graph_module._backward_hooks = root._backward_hooks.copy()
        graph_module._backward_pre_hooks = root._backward_pre_hooks.copy()
        if graph_module._forward_hooks or graph_module._forward_pre_hooks:
            # Even forward hooks are traced through, they still capture a bunch
            # of state through closure. We need to make sure these data are
            # accessible through the captured module (but the hooks should be
            # disabled).
            if getattr(graph_module, "_wrapped_call", None) is None:
                raise AssertionError("graph_module._wrapped_call must not be None")
            if not isinstance(
                graph_module._wrapped_call, torch.fx.graph_module._WrappedCall
            ):
                raise AssertionError(
                    f"expected _WrappedCall, got {type(graph_module._wrapped_call)}"
                )
            if graph_module._wrapped_call.cls_call is not None:
                raise AssertionError("graph_module._wrapped_call.cls_call must be None")

            def dynamo_wrapped_call(self, *args: object, **kwargs: object) -> object:
                if "forward" in self.__dict__:
                    raise AssertionError("'forward' should not be in self.__dict__")

                fwd_hooks = self._forward_hooks
                fwd_pre_hooks = self._forward_pre_hooks
                original_forward = type(self).forward

                def patched_forward(self, *args: object, **kwargs: object) -> object:
                    self._forward_hooks = fwd_hooks
                    self._forward_pre_hooks = fwd_pre_hooks
                    return original_forward(self, *args, **kwargs)

                try:
                    self.forward = types.MethodType(patched_forward, self)
                    # pyrefly: ignore [implicit-any]
                    self._forward_hooks = {}
                    # pyrefly: ignore [implicit-any]
                    self._forward_pre_hooks = {}
                    # pyrefly: ignore [invalid-argument]
                    return super(type(self), self).__call__(*args, **kwargs)
                finally:
                    self.__dict__.pop("forward")
                    self._forward_hooks = fwd_hooks
                    self._forward_pre_hooks = fwd_pre_hooks

            # pyrefly: ignore [bad-assignment]
            graph_module._wrapped_call.cls_call = dynamo_wrapped_call

    root = graph_module if isinstance(root, torch.nn.Module) else root
    input_processor = InputProcessor(root, len(args), list(kwargs.keys()))
    dynamo_bytecode_flatten = DynamoBytecodeFlatten(input_processor, out, f_globals)
    dynamo_bytecode_unflatten = DynamoBytecodeUnflatten(input_processor, out, f_globals)

    graph_module.graph._codegen = _DynamoBytecodeCodeGen(
        argument_names(inspect.signature(mod), args, kwargs),
        dynamo_bytecode_flatten,
        dynamo_bytecode_unflatten,
    )  # type: ignore[attr-defined]
    normalize_graph_module(graph_module)
    if hasattr(graph_module, "_dynamo_bytecode_flatten"):
        raise AssertionError(
            "graph_module already has _dynamo_bytecode_flatten attribute"
        )
    if hasattr(graph_module, "_dynamo_bytecode_unflatten"):
        raise AssertionError(
            "graph_module already has _dynamo_bytecode_unflatten attribute"
        )
    # pyrefly: ignore [bad-argument-type]
    graph_module._dynamo_bytecode_flatten = dynamo_bytecode_flatten
    # pyrefly: ignore [bad-argument-type]
    graph_module._dynamo_bytecode_unflatten = dynamo_bytecode_unflatten
    delattr(graph_module, "_param_name_to_source")
    graph_module.recompile()
    graph_module.meta["module_call_specs"] = (
        out.graph_capture_output.output_graph.export_metadata.module_call_spec
    )
    if out.backend_input is None:
        raise AssertionError("out.backend_input must not be None for fake_mode")
    graph_module.meta["fake_mode"] = out.backend_input.fake_mode  # type: ignore[attr-defined]
    graph_module.meta["fake_mode"].allow_non_fake_inputs = True
    tracing_context = TracingContext(graph_module.meta["fake_mode"])
    tracing_context.tensor_to_context = out.backend_input.tensor_to_context  # type: ignore[attr-defined]
    graph_module.meta["tracing_context"] = tracing_context
    return graph_module


class _DynamoBytecodeCodeGen(torch.fx.graph.CodeGen):
    def __init__(
        self,
        orig_arg_names: list[str],
        dynamo_bytecode_flatten: DynamoBytecodeFlatten,
        dynamo_bytecode_unflatten: DynamoBytecodeUnflatten,
    ) -> None:
        super().__init__()
        self.orig_arg_names = orig_arg_names
        self.dynamo_bytecode_flatten = dynamo_bytecode_flatten
        self.dynamo_bytecode_unflatten = dynamo_bytecode_unflatten
        self.wrap_tuple = False
        self._inputs: tuple[Any, ...] | None = None

    def process_inputs(self, *inputs: Any) -> Any:
        self._inputs = inputs
        results = self.dynamo_bytecode_flatten(*inputs)
        fake_mode = detect_fake_mode()
        if fake_mode is not None and pytree.tree_any(
            lambda x: isinstance(x, torch.Tensor) and not fake_mode.is_our_fake(x),
            results,
        ):
            # Bytecode replay can recover tensors captured from module
            # attributes as extra FX placeholder values. These tensors are not
            # visible in AOTAutograd's user-facing call signature, so allow the
            # active FakeTensorMode to convert them at dispatch time. This must
            # remain enabled for AOT's backward trace too, since autograd can
            # save the recovered tensors from the forward.
            fake_mode.allow_non_fake_inputs = True
        return results

    def process_outputs(self, outputs: Any) -> Any:
        if self._inputs is None:
            raise AssertionError("process_outputs called before process_inputs")
        results = self.dynamo_bytecode_unflatten(outputs, self._inputs)
        if self.wrap_tuple:
            results = (results,)
        self._inputs = None
        return results

    def gen_fn_def(
        self,
        free_vars: list[str],
        maybe_return_annotation: str,
        *,
        expanded_def: bool = False,
    ) -> str:
        fn_args = self.orig_arg_names
        has_orig_self = (fn_args[0] == "self") if len(fn_args) > 0 else False
        if has_orig_self:
            free_vars.insert(0, "self")
        # Rename any non-first `self` in fn_args to a unique name. The base
        # CodeGen.gen_fn_def prepends `"self"` for the GraphModule's bound-
        # method receiver whenever fn_args[0] != "self", which collides with
        # a schema param literally named `self` (e.g. `aten.where.self(Tensor
        # cond, Tensor self, Tensor other)` -> `def forward(self, cond, self,
        # other)` -> `SyntaxError: duplicate argument 'self' in function
        # definition`). Both the function-def arg list (via super().gen_fn_def)
        # and the body binding (via gen_var_bindings) reference fn_args, so
        # the rename is consistent end-to-end.
        fn_args = list(fn_args)
        first_pos = 1 if has_orig_self else 0
        for i in range(first_pos, len(fn_args)):
            if fn_args[i] == "self":
                new_name = "self_"
                while new_name in fn_args:
                    new_name += "_"
                fn_args[i] = new_name
        fn_definition = super().gen_fn_def(
            fn_args[:], maybe_return_annotation, expanded_def=expanded_def
        )

        if len(free_vars) > 0:  # pytree has placeholders in it
            fn_definition += self.gen_var_bindings(fn_args, free_vars, expanded_def)
        return fn_definition

    def gen_var_bindings(
        self, fn_args: list[str], free_vars: list[str], expanded_def: bool
    ) -> str:
        without_annotation = [x.split(":")[0].split("#")[0] for x in free_vars]
        if len(fn_args) == 0:
            fn_signature = ""
        elif len(fn_args) == 1:
            fn_signature = f"{fn_args[0]}, "
        else:
            fn_signature = f"{', '.join(fn_args)}"
        return f"""
    _fn_args = ({fn_signature})
    {", ".join(without_annotation)}, = self._dynamo_bytecode_flatten(*_fn_args)"""

    def generate_output(
        self,
        output_args: torch.fx.node.Argument,
        *,
        descs: object | None = None,
        repr_fn: Any | None = None,
    ) -> str:
        if repr_fn is None:
            repr_fn = repr
        # pyrefly: ignore [not-iterable]
        returned = f"self._dynamo_bytecode_unflatten(({', '.join([repr_fn(a) for a in output_args])},), _fn_args)"
        if self.wrap_tuple:
            returned = f"({returned},)"
        return f"return {returned}"


def dynamo_graph_capture_for_export(
    fn: Callable[..., Any],
    constraints: list[Constraint] | None = None,
) -> Callable[..., Any]:
    if isinstance(fn, torch._ops.OpOverload):

        def default_annotation(arg: torch.Argument) -> str:
            if arg.has_default_value():
                return f"={arg.default_value!r}"
            return ""

        has_kwarg_only = False
        arg_list = []
        for arg in fn._schema.arguments:
            if arg.kwarg_only and not has_kwarg_only:
                has_kwarg_only = True
                arg_list.append("*")
            arg_list.append(arg.name + default_annotation(arg))
        func_str = f"""
def op_overload_wrapper({", ".join(arg_list)}):
    return op({", ".join([f"{arg.name}={arg.name}" for arg in fn._schema.arguments])})
"""
        out = {}
        exec(func_str, {"op": fn}, out)
        fn = out["op_overload_wrapper"]  # type: ignore[assignment]

    def inner(*args: Any, **kwargs: Any) -> Any:
        if torch._dynamo.config.install_free_tensors:
            raise AssertionError("install_free_tensors must be False")
        with (
            _compiling_state_context(),
            torch._dynamo.config.patch(
                replay_side_effects=False, side_effect_replay_policy="warn"
            ),
            get_metrics_context(),
            dynamo_timed("fullgraph_capture"),
        ):
            out = fullgraph_capture(
                fn,
                args,
                kwargs,
                constraints=constraints,
            )
        graph_module = create_fx_graph_from_captured_output(out, fn, args, kwargs)
        return graph_module

    return inner


def _dynamo_graph_capture_for_export(
    mod: Callable[..., Any],
    *,
    constraints: list[Constraint] | None = None,
    dynamic_shapes: _DynamicShapesInput = None,
) -> Callable[..., torch.fx.GraphModule]:
    """
    Improved dynamo graph capture using transformer approach with proper fake tensor handling.

    This function creates a capture instance that handles:
    1. PyTree flattening/unflattening with proper input ordering
    2. Dynamo graph capture with export-specific context
    3. FX graph transformation for export compatibility
    4. Proper fake tensor metadata preservation
    5. Dynamic dimension constraint handling

    Notable improvements over manual approach:
    - Uses FX Transformer for cleaner graph manipulation
    - Properly handles fake tensor metadata and dynamic dimensions
    - Preserves all necessary metadata for export
    - More robust error handling and edge case management

    TODO:
    1. Are we actually gonna run the bytecode?
    2. Need to attach guards
    """

    _constraints = constraints

    def inner(*args: Any, **kwargs: Any) -> torch.fx.GraphModule:
        # This sets the is_exporting flag when building guards.
        with _compiling_state_context():
            # Only a ShapesSpec/ParamsSpec object opts into the new spec API.
            # A bare dict/tuple in export's `dynamic_shapes` is now bound to
            # the Dim export API.
            # TODO: inspect dictionary leaves and enable dict for ParamsSpec
            # here too, like make_fx and compile.
            user_spec: ShapesSpec | None = None
            if isinstance(dynamic_shapes, (ShapesSpec, ParamsSpec)):
                user_spec = _coerce_to_shapes_spec(dynamic_shapes)
            flattened_spec: ShapesSpec | None = None
            if user_spec is not None:
                # _bind_spec_to_args also flattens (args, kwargs) for us.
                leaf_specs, flat_inputs, in_spec = _bind_spec_to_args(
                    mod, args, kwargs, user_spec
                )
                flattened_spec = ShapesSpec(
                    # leaf_specs is list[LeafSpec]; cast around list invariance
                    # (ParamsSpec wants list[IntermediateSpec]).
                    ParamsSpec({"*args": cast(list[IntermediateSpec], leaf_specs)}),
                    assumptions=user_spec._assumptions or None,
                )
            else:
                flat_inputs, in_spec = pytree.tree_flatten((args, kwargs))
            check_user_input_output(flat_inputs, UserErrorType.INVALID_INPUT)
            module_to_trace = ModuleToTrace(mod, in_spec)
            orig_callable = mod.forward if isinstance(mod, torch.nn.Module) else mod

            constraints: list[Constraint] | None = _constraints

            from . import reset  # type: ignore[attr-defined]

            reset()

            dynamo_config_ctx = torch._dynamo.config.patch(
                specialize_int=True,
                specialize_float=True,
                assume_static_by_default=True,
                automatic_dynamic_shapes=False,
                capture_dynamic_output_shape_ops=True,
                capture_scalar_outputs=True,
                constant_fold_autograd_profiler_enabled=True,
                log_graph_in_out_metadata=True,
                # install_free_tensors ensures that params and buffers are still
                # added as graph attributes, and makes Dynamo emits graphs that
                # follow export pytree-able input requirements In future, if we
                # fully rely on bytecode for the runtime, we can turn this flag
                # off.
                install_free_tensors=torch._dynamo.config.install_free_tensors_for_export,
            )

            # If a user spec was supplied, re-key it to the intermediate
            # ModuleToTrace.forward(*flat_args) layout and expose via
            # ``torch._dynamo.config._dynamic_shapes_spec`` for the variable
            # builder.
            shapes_spec_in_use = user_spec is not None
            shapes_spec_ctx = nullcontext()
            if flattened_spec is not None:
                shapes_spec_ctx = torch._dynamo.config.patch(
                    _dynamic_shapes_spec=flattened_spec
                )

            with (
                get_metrics_context(),
                dynamo_timed("fullgraph_capture"),
                dynamo_config_ctx,
                shapes_spec_ctx,
            ):
                out = fullgraph_capture(
                    module_to_trace,
                    tuple(flat_inputs),
                    constraints=_constraints,
                    _is_export_deprecated_do_not_use=True,
                )

                if out.graph_capture_output.output_graph is None:
                    raise AssertionError(
                        "output_graph must not be None after fullgraph_capture"
                    )

                example_inputs: list[Any] = []
                if out.backend_input is not None:
                    graph = out.backend_input.graph_module
                    fake_mode = out.backend_input.fake_mode
                    example_inputs = out.backend_input.example_inputs
                else:
                    graph = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
                    graph.graph.output(None)
                    graph.recompile()
                    fake_mode = None

                # ShapesSpec is unbacked based, constraint violations are not
                # relevant.
                if not shapes_spec_in_use:
                    _suggest_or_raise_constraint_violation(
                        module_to_trace,
                        orig_callable,
                        fake_mode,
                        out,
                        args,
                        kwargs,
                        dynamic_shapes,  # type: ignore[arg-type]
                    )

                # Extract export metadata from the new location
                export_metadata = out.graph_capture_output.output_graph.export_metadata
                graph_inputs = export_metadata.graph_input_idx_to_local_source
                graph_output_map = export_metadata.output_return_type
                out_spec = export_metadata.out_spec
                module_call_spec = export_metadata.module_call_spec

            # Compute dynamic dimensions for each input based on constraints
            flat_args_dynamic_dims = [
                {
                    c.dim
                    for c in (constraints or ())
                    if (
                        c.t_id == id(x)
                        and not isinstance(c, _RelaxedConstraint)
                        and c.constraint_range.vr.lower != c.constraint_range.vr.upper
                    )
                }
                for x in flat_inputs
            ]

            # Create input order mapping from dynamo's internal order to user order
            # Only process inputs that come from function arguments (GetItemSource).
            # Skip inputs that come from other sources like closures (e.g., captured
            # opaque objects like DeviceMesh).
            graph_input_order: dict[int, int] = {}
            for inp in graph_inputs:
                source = graph_inputs[inp]
                if isinstance(source, torch._dynamo.source.GetItemSource):
                    graph_input_order[source.index] = len(graph_input_order)

            for real_idx, graph_idx in graph_input_order.items():
                flat_inputs[real_idx] = example_inputs[graph_idx]

            # Use FX transformer to rebuild the graph cleanly
            transformed_graph = DynamoGraphTransformer(
                graph,
                flat_inputs,
                flat_args_dynamic_dims,
                graph_input_order,
                graph_output_map,
                fake_mode,
                graph_inputs,
            ).transform()

            # Set up PyTree codegen for proper input/output handling
            transformed_graph.graph._codegen = _PyTreeCodeGen(
                _PyTreeInfo(
                    argument_names(inspect.signature(orig_callable), args, kwargs),  # type: ignore[attr-defined, arg-type]
                    in_spec,
                    out_spec,
                )
            )
            transformed_graph.recompile()

            clean_nn_module_stack_and_source_fn(transformed_graph, True)
            clean_export_root(transformed_graph)

            transformed_graph.meta["module_call_specs"] = module_call_spec
            transformed_graph.meta["fake_mode"] = fake_mode

            return transformed_graph

    return inner
<|endoftext|>"""
This module provides the public comptime interface to TorchDynamo, enabling users to execute
arbitrary Python code during symbolic evaluation of their programs.

The comptime interface allows inspection and modification of TorchDynamo's compilation
process while it is running. This can be useful for:

- Debugging compilation issues
- Inspecting intermediate state
- Adding custom guards or graph breaks
- Analyzing symbolic shapes and values

Example usage:

    import torch
    from torch._dynamo.comptime import comptime

    def my_model(x):
        # Print the compile-time known information about x
        comptime.print(x)

        # Print the current FX graph being constructed
        comptime.print_graph()

        # Force a value to be treated as static
        if comptime(lambda ctx: ctx.get_local("x").is_dynamic()):
            comptime.force_static(x)

        # Add a manual graph break
        comptime.graph_break()

Note: While this API provides significant flexibility, it intentionally avoids
exposing internal implementation details of TorchDynamo to maintain compatibility
across versions.
"""

import builtins
import dis
import time
import traceback
from collections.abc import Callable
from typing import Any, TextIO

import torch
from torch._dynamo.symbolic_convert import InstructionTranslatorBase
from torch._dynamo.variables.base import VariableTracker
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import free_symbols

from .exc import unimplemented
from .variables import CellVariable
from .variables.tensor import SymNodeVariable


class ComptimeVar:
    """
    A ComptimeVar represents a Python value, at some particular point
    in time, in the Python code we are symbolically evaluating with
    torchdynamo.  This must be distinguished from a runtime value, as
    at compile-time there are some properties of the variable we
    do not know (for example, if the ComptimeVar represents a Tensor,
    we only know metadata about the tensor; we do NOT know what the
    actual data in the Tensor is.)
    """

    def __init__(self, v: VariableTracker) -> None:
        self.__variable = v

    def as_proxy(self) -> VariableTracker | list[VariableTracker]:
        """
        Returns an fx.Proxy (or tuple/list of fx.Proxy) representing
        this variable in the FX graph we are assembling to pass
        to the user compiler.

        This method only works for variables we actually track in
        the FX graph, aka Tensors (and ints, if you are compiling
        with dynamic shapes).  In particular, if you have a list
        or tuple of tensors, you will get a list/tuple of proxies
        (not a single proxy representing the entire list/tuple).
        """
        return self.__variable.as_proxy()

    def is_proxy(self) -> bool:
        """
        Returns True if as_proxy() would succeed.
        """
        return self.__variable.is_proxy()

    def as_fake(self) -> FakeTensor | torch.SymInt:
        """
        Returns a "fake" value (either a FakeTensor or a SymInt)
        representing the variable in question.  This only works
        for variables that denote Tensor or int.  You can use
        this to query metadata; e.g., v.as_fake().size(0) will
        tell you the compile-time known size of the tensor.

        WARNING: Do NOT mutate the returned tensor.
        """
        return self.__variable.as_proxy().node.meta["example_value"]

    def size(self, dim: int | None = None) -> int | torch.SymInt:
        """
        Returns the size of the tensor (if dim is None) or the size
        at the dimension dim.  The returned size may be a SymInt.
        """
        return self.as_fake().size(dim)  # type: ignore[union-attr, return-value]

    def python_type(self) -> type:
        """
        Returns what type(v) would have returned for the variable
        at compile time.
        """
        return self.__variable.python_type()

    def as_python_constant(self) -> Any:
        """
        Returns the Python value this variable would have, but only if it is
        completely known at compile-time (e.g., it is constant).

        WARNING: Do NOT mutate the returned constant.  The returned constant
        may or may not correspond to the actual value this variable may take
        on at runtime; for example, if the variable in question is a constant
        list, we may return a copy of that list.
        """
        return self.__variable.as_python_constant()

    def is_python_constant(self) -> bool:
        """
        Returns True if as_python_constant would succeed.
        """
        return self.__variable.is_python_constant()

    def is_dynamic(self) -> bool:
        if isinstance(self.__variable, SymNodeVariable):
            fs = free_symbols(self.__variable.sym_num)
            return bool(fs)
        return False

    def force_static(self) -> None:
        """
        Forces that a value is static, inducing a guard on its specific value
        """
        if isinstance(self.__variable, SymNodeVariable):
            self.__variable.evaluate_expr()
        elif self.__variable.is_python_constant():
            # TODO: Maybe complain if this isn't a int/bool/float variable
            pass
        else:
            raise AssertionError(
                f"cannot force {self.__variable} ({type(self.__variable)}) static"
            )

    def _i_will_not_complain_if_bc_breaks_VariableTracker(self) -> VariableTracker:
        """
        Returns the internal data structure VariableTracker that Dynamo uses
        to represent variables at compile time.  There are no BC guarantees on
        this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if you rely on
        it.
        """
        return self.__variable

    def __repr__(self) -> str:
        return self.__variable.debug_repr()

    # TODO: API for adding a custom guard


class ComptimeContext:
    """
    This context class provides access to a public API for Dynamo's internals.
    If there is something here you would find useful that is missing, please
    file a feature request at https://github.com/pytorch/pytorch/
    """

    def __init__(self, tx: InstructionTranslatorBase) -> None:
        self.__tx = tx

    def get_local(self, name: str, *, stacklevel: int = 0) -> ComptimeVar:
        """
        Retrieve the compile-time known information about a local.
        """
        tx = self.__get_tx(stacklevel)
        var = tx.symbolic_locals[name]

        # Auto-dereference when accessing cell locals in python.
        if isinstance(var, CellVariable):
            return ComptimeVar(tx.output.side_effects.load_cell(var))

        return ComptimeVar(var)

    def graph_break(self, msg: str = "ComptimeContext.graph_break") -> None:
        """
        Manually trigger a graph break
        """
        unimplemented(
            gb_type="ComptimeContext graph break",
            context=msg,
            explanation=f"Manually triggered ComptimeContext graph break with message {msg}.",
            hints=[],
        )

    def graph(self) -> torch.fx.Graph:
        """
        Retrieve the partially constructed FX graph that would be
        passed to the user compiler after compilation.
        """
        return self.__tx.output.graph

    def assert_static(self, val: ComptimeVar) -> None:
        """
        Asserts that the int is static (and not dynamic, per dynamic shapes)
        """
        if val.is_dynamic():
            raise AssertionError(
                "expected static but got dynamic (run with TORCH_LOGS=dynamic for more info)"
            )

    def print_graph(self, *, verbose: bool = True, file: TextIO | None = None) -> None:
        """
        Print the partially constructed FX graph that would be passed
        to the user compiler after compilation.
        """
        print(
            self.__tx.output.graph.python_code("self", verbose=verbose).src, file=file
        )

    def parent(self) -> "ComptimeContext":
        return ComptimeContext(self.__tx.parent)  # type: ignore[arg-type]

    def __get_tx(self, stacklevel: int) -> Any:
        tx = self.__tx
        # pyrefly: ignore [bad-assignment, non-convergent-recursion]
        for _ in range(stacklevel):
            tx = tx.parent  # type: ignore[assignment]
        return tx

    def print(self, val: Any, *, file: TextIO | None = None) -> None:
        print(repr(val), file=file)

    def print_disas(self, *, file: TextIO | None = None, stacklevel: int = 0) -> None:
        """
        Print the current series of opcodes being executed (not including
        parent frames), including where you are in the particular opcode
        stream.
        """
        tx = self.__get_tx(stacklevel)
        print(
            dis.Bytecode(
                tx.f_code,
                current_offset=tx.instructions[tx.instruction_pointer].offset,
            ).dis(),
            file=file,
        )

    def print_value_stack(
        self, *, file: TextIO | None = None, stacklevel: int = 0
    ) -> None:
        """
        Print the current Python value stack.  Note that this is NOT the same
        as the traceback; use print_bt() to print that.  Note that at
        stacklevel=0, this will typically be empty, as comptime cannot
        currently be used in an expression context where there would be
        intermediates on the stack.  If you would find this useful, please
        file a bug at https://github.com/pytorch/pytorch/

        NB: Stack grows downwards in our print
        """
        tx = self.__get_tx(stacklevel)
        for s in tx.stack:
            print(f"- {s.debug_repr()}", file=file)

    def print_locals(self, *, file: TextIO | None = None, stacklevel: int = 0) -> None:
        """
        Print all of the locals available in the current context.
        By default this view is very limited; you can get more information
        about any individual local using get_local().
        """
        tx = self.__get_tx(stacklevel)
        for k, v in tx.symbolic_locals.items():
            print(f"{k} = {v.debug_repr()}", file=file)

    def print_bt(self, *, file: TextIO | None = None, stacklevel: int = 0) -> None:
        """
        Print the user code backtrace, starting at the beginning of the
        frame Dynamo started evaluating.  Note that this MAY NOT go all
        the way to the torch.compile invocation, as we may have done
        a graph break and are compiling an intermediate frame as the
        starting point.  If you think the other behavior would be better,
        file a bug at https://github.com/pytorch/pytorch/
        """
        stack = []
        tx = self.__get_tx(stacklevel)
        while tx is not None:
            stack.append(tx.frame_summary())
            tx = getattr(tx, "parent", None)
        print(
            "".join(traceback.StackSummary.from_list(reversed(stack)).format()),
            file=file,
        )

    def print_guards(self, *, file: TextIO | None = None) -> None:
        """
        Print the currently installed guards for the Dynamo context.
        This does NOT include guards associated with variables that
        may or may not be installed in the future if those variables
        are used.
        """
        # TODO: improve print format, current guard format is extremely
        # verbose
        print(
            "\n".join(f"{repr(guard)}" for guard in sorted(self.__tx.output.guards)),
            file=file,
        )

    def _i_will_not_complain_if_bc_breaks_InstructionTranslator(
        self,
    ) -> InstructionTranslatorBase:
        """
        Returns the internal data structure InstructionTranslator that Dynamo
        uses to track state of symbolic evaluation.  There are no BC
        guarantees on this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if
        you rely on it.
        """
        return self.__tx

    def sleep(self, sec: int | float) -> None:
        time.sleep(sec)


class _Comptime:
    @staticmethod
    def __call__(
        fn: Callable[[ComptimeContext], Any],
        fallback_fn: Callable[[], Any] = lambda: None,
    ) -> Any:
        """fn gets called at compile time in TorchDynamo, calls fallback_fn otherwise"""
        fallback_fn()

    # Convenience wrappers that are more compact to use

    @staticmethod
    def graph_break() -> None:
        comptime(lambda ctx: ctx.graph_break())

    @staticmethod
    def print(e: Any) -> None:
        comptime(lambda ctx: ctx.print(ctx.get_local("e")), lambda: print(e))

    @staticmethod
    def print_graph() -> None:
        comptime(lambda ctx: ctx.print_graph())

    @staticmethod
    def print_disas(*, stacklevel: int = 0) -> None:
        comptime(
            lambda ctx: ctx.print_disas(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    @staticmethod
    def print_value_stack(*, stacklevel: int = 0) -> None:
        comptime(
            lambda ctx: ctx.print_value_stack(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    # This is a more useful variant of print_value_stack that can be used
    # in an expression context; e.g., x + print_value_stack_and_return(y + z),
    # you will see x on the stack prior to the addition operation
    @staticmethod
    def print_value_stack_and_return(e: Any, *, stacklevel: int = 0) -> Any:
        comptime(
            lambda ctx: ctx.print_value_stack(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )
        return e

    @staticmethod
    def print_locals(*, stacklevel: int = 0) -> None:
        comptime(
            lambda ctx: ctx.print_locals(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    @staticmethod
    def print_bt(*, stacklevel: int = 0) -> None:
        comptime(
            lambda ctx: ctx.print_bt(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    @staticmethod
    def print_guards() -> None:
        comptime(lambda ctx: ctx.print_guards())

    @staticmethod
    def assert_static(val: Any) -> None:
        comptime(lambda ctx: ctx.assert_static(ctx.get_local("val")))

    @staticmethod
    def force_static(val: Any) -> None:
        comptime(lambda ctx: ctx.get_local("val").force_static())

    @staticmethod
    def breakpoint() -> None:
        """
        Like pdb breakpoint(), but drop into pdb whenever this line
        of code is compiled by dynamo.  Use it by putting
        this in your model code::

            from torch._dynamo.comptime import comptime

            comptime.breakpoint()

        And then, inside pdb, you can access 'ctx' to query things
        about the compilation context::

            (Pdb) !ctx.print_bt()
            (Pdb) !ctx.print_locals()
            (Pdb) p ctx.get_local("attention").as_fake()
        """

        def inner(inner_ctx: ComptimeContext) -> None:
            ctx = inner_ctx.parent()  # noqa: F841
            builtins.breakpoint()

        comptime(inner)

    @staticmethod
    def sleep(sec: int | float) -> None:
        comptime(lambda ctx: ctx.sleep(ctx.get_local("sec").as_python_constant()))


comptime = _Comptime()
<|endoftext|>"""
This module provides Source classes that track the origins of values in PyTorch Dynamo.
Sources represent where values come from (e.g. local variables, globals, attributes) and
are used for guard generation and code reconstruction during compilation.

The module includes specialized sources for:
- Local variables and synthetic locals
- Global variables and constants
- Object attributes and method calls
- NN module specialization (specialized vs unspecialized)
- Random values and tensor properties
- Default argument handling
- FSDP (Fully Sharded Data Parallel) modules

Sources play a key role in Dynamo's guard system by tracking value origins for
guard generation, and in code reconstruction by providing methods to rebuild
the code needed to recreate values.
"""

import dataclasses
import enum
import functools
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from torch import device as device_type
from torch._guards import (
    ChainedSource,
    dataclass_with_cached_hash,
    Guard,
    GuardSource,
    Source,
)

from . import utils
from .bytecode_transformation import (
    create_binary_subscr,
    create_build_tuple,
    create_call_function,
)


if TYPE_CHECKING:
    from .codegen import PyCodegen

# It shouldn't be supported to construct an NNModuleVariable inside an FSDP module,
# so those cases are omitted intentionally

# represents nn.Modules tracked with NNModuleVariable (specialized is implicit in the variable name)
_GUARD_SOURCE_SPECIALIZED_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
    # Just to ensure that guard_source() works
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}

# represents nn.Modules tracked with UnspecializedNNModuleVariable
_GUARD_SOURCE_UNSPECIALIZED_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    # this happens for an UnspecializedNNModule submodule on a NNModuleVariable
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    # Just to ensure that guard_source() works
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}

# represents nn.Modules tracked with UnspecializedBuiltinNNModuleVariable
_GUARD_SOURCE_UNSPECIALIZED_BUILTIN_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    # Just to ensure that guard_source() works
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}

_GUARD_SOURCE_FSDP_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}


def is_constant_source(source: Source) -> bool:
    if isinstance(source, ConstantSource):
        return True
    try:
        if source.guard_source == GuardSource.CONSTANT:
            return True
    except NotImplementedError:
        pass

    return False


def _get_source_debug_name(source: Source | None) -> str:
    if source is None:
        return "<unknown source>"
    else:
        try:
            return source.name
        except NotImplementedError:
            return "<unknown source>"


def _esc_str(s: Any, apply_repr: bool = False) -> str:
    """
    Escapes curly brackets for format strings.
    e.g. "frozenset({0})" becomes "frozenset({{0}})".
    This is used by _name_template for example, because it's
    expected to return a format string, but we may wish to include
    strings that should not be accidentally formatted.
    """
    if apply_repr:
        s = repr(s)
    else:
        s = str(s)
    return s.replace("{", "{{").replace("}", "}}")


@dataclass_with_cached_hash(frozen=True)
class LocalSource(Source):
    local_name: str

    # Whether this local is an input to the root frame.
    is_input: bool = False

    # Whether we know this input is dynamic (based on example_inputs)
    # For non tensors, we simply look at the first index of the tuple
    dynamism: frozenset[str] | None = None

    # Whether the item at this source is the _content_ of a cell that is
    # dereferenced from the root frame, i.e., it's a part of the `co_cellvars`
    # or `co_freevars`.
    is_derefed_cell_contents: bool = False

    # Whether this local is the function's varargs (``*args``) parameter.
    # Set from ``co_flags & CO_VARARGS`` at frame-entry time. Element accesses
    # like ``args[N]`` produce a ``GetItemSource`` whose base has this flag.
    # Useful for distinguishing ``*args`` from a regular list-typed input.
    # ``repr=False`` so the vast majority of locals (which are not varargs) do
    # not get a noisy ``is_varargs=False`` in every debug string.
    is_varargs: bool = dataclasses.field(default=False, repr=False)

    # Whether this local is the function's varkw (``**kwargs``) parameter.
    # Set from ``co_flags & CO_VARKEYWORDS`` at frame-entry time. Element
    # accesses like ``kwargs["k"]`` produce a ``DictGetItemSource`` whose base
    # has this flag. ``repr=False`` for the same reason as ``is_varargs``.
    is_varkw: bool = dataclasses.field(default=False, repr=False)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self.is_derefed_cell_contents:
            codegen.load_deref(self.local_name)
        else:
            codegen.append_output(codegen.create_load(self.local_name))

    def reconstruct_pycode(self, codegen: "PyCodegen") -> str:
        return self.local_name

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.LOCAL

    @functools.cached_property
    def _name_template(self) -> str:
        return f"L[{_esc_str(self.local_name, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class TempLocalSource(Source):
    # like LocalSource, but cannot be guarded on
    local_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.append_output(codegen.create_load(self.local_name))

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.TEMP_LOCAL

    @property
    def _name_template(self) -> str:
        raise NotImplementedError(
            "Cannot create guard on TempLocalSource - this is an internal Dynamo bug. Please file an issue on GitHub."
        )


@dataclass_with_cached_hash(frozen=True)
class SyntheticLocalSource(Source):
    local_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.append_output(codegen.create_load(self.local_name))

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.SYNTHETIC_LOCAL

    @functools.cached_property
    def _name_template(self) -> str:
        return f"SYNTHETIC_LOCAL[{_esc_str(self.local_name, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class RandomValueSource(Source):
    random_call_index: int

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.RANDOM_VALUE

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.append_output(codegen.create_load(codegen.tx.output.random_values_var))
        codegen.append_output(codegen.create_load_const(self.random_call_index))
        codegen.append_output(create_binary_subscr())

    @functools.cached_property
    def _name_template(self) -> str:
        return f"random_value_{_esc_str(self.random_call_index)}"


@dataclass_with_cached_hash(frozen=True)
class GlobalSource(Source):
    global_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.append_output(codegen.create_load_global(self.global_name, add=True))

    def reconstruct_pycode(self, codegen: "PyCodegen") -> str:
        return self.global_name

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.GLOBAL

    @functools.cached_property
    def _name_template(self) -> str:
        return f"G[{_esc_str(self.global_name, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class GlobalWeakRefSource(Source):
    global_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.append_output(
                codegen.create_load_global(self.global_name, add=True)
            )
        )
        codegen.extend_output(create_call_function(0, False))

    @property
    def guard_source(self) -> GuardSource:
        return GuardSource.GLOBAL

    @functools.cached_property
    def _name_template(self) -> str:
        return f"G[{_esc_str(self.global_name, apply_repr=True)}]()"


@dataclass_with_cached_hash(frozen=True)
class WeakRefCallSource(ChainedSource):
    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen(self.base))
        codegen.extend_output(create_call_function(0, False))

    @property
    def _name_template(self) -> str:
        return "{0}()"


@dataclass_with_cached_hash(frozen=True)
class CallFunctionNoArgsSource(WeakRefCallSource):
    pass


@dataclass_with_cached_hash(frozen=True)
class AttrSource(ChainedSource):
    member: str

    def __post_init__(self) -> None:
        if not self.base:
            raise AssertionError(
                "Can't construct an AttrSource without a valid base source"
            )
        if "." in self.member:
            raise AssertionError(
                f"AttrSource member must not contain '.', got {self.member!r}. "
                "Use OutputGraph.get_chained_attr_source() for dotted paths."
            )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs(self.member))

    def reconstruct_pycode(self, codegen: "PyCodegen") -> str:
        base = self.base.reconstruct_pycode(codegen)
        if self.member.isidentifier():
            return f"{base}.{self.member}"
        else:
            return f"getattr({base}, {self.member!r})"

    @functools.cached_property
    def _name_template(self) -> str:
        if not self.member.isidentifier():
            return f"getattr({{0}}, {_esc_str(self.member, apply_repr=True)})"
        return f"{{0}}.{_esc_str(self.member)}"


@dataclass_with_cached_hash(frozen=True)
class CellContentsSource(AttrSource):
    """
