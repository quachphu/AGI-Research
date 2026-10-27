                self.model.graph.inputs.extend(original_inputs)

    def apply_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Apply the weights from the specified state dict to the ONNX model.

        Use this method to replace FakeTensors or other weights.

        Args:
            state_dict: The state dict containing the weights to apply to the ONNX model.
        """
        from torch.onnx._internal.exporter import _core

        for name, tensor in state_dict.items():
            if name in self.model.graph.initializers:
                self.model.graph.initializers[name].const_value = _core.TorchTensor(
                    tensor, name
                )
            else:
                warnings.warn(
                    f"Weight '{name}' not found in the model. Skipped applying.",
                    category=torch.onnx.errors.OnnxExporterWarning,
                    stacklevel=1,
                )

    def initialize_inference_session(
        self,
        initializer: Callable[
            [str | bytes], ort.InferenceSession
        ] = _ort_session_initializer,
    ) -> None:
        """Initialize the ONNX Runtime inference session.

        Args:
            initializer: The function to initialize the ONNX Runtime inference
                session with the specified model. By default, it uses the
                :func:`_ort_session_initializer` function.
        """
        # TODO(justinchuby): Allow different inference options
        logger.debug("Initializing the inference session.")
        if (
            byte_size := _count_initializer_size(self.model.graph)
        ) > _LARGE_MODEL_THRESHOLD:
            logger.debug("The model initializers is larger than 1.5GB (%s).", byte_size)
            # Save the model to a temporary file if too large
            self._tempdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            model_path = os.path.join(self._tempdir.name, "model.onnx")
            self.save(model_path, external_data=True)
            model = model_path
        else:
            model = self.model_proto.SerializeToString()  # type: ignore[assignment]

        self._inference_session = initializer(model)
        logger.debug("Inference session initialized.")

    def release(self) -> None:
        """Release the inference session.

        You may call this method to release the resources used by the inference session.
        """
        # Release the inference session first so that the model file can be deleted
        if self._inference_session is not None:
            self._inference_session = None
        gc.collect()
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def rename_axes(self, rename_mapping: dict[str | ir.SymbolicDim, str]) -> None:
        """Rename axes in a model according to the specified rename mapping.

        Example::

            batch = onnx_program.model.graph.inputs[0].shape[0]
            seq_len = onnx_program.model.graph.inputs[0].shape[2]
            rename_mapping = {
                batch: "batch",
                seq_len: "seq_len",
            }
            onnx_program.rename_axes(rename_mapping)

        Args:
            rename_mapping: A dictionary mapping old axes to new axis names.
                Keys can be either:

                * String axis names (e.g., "s1", "s2")
                * SymbolicDim objects obtained from the model
                  (e.g., onnx_program.model.graph.inputs[0].shape[0])

                Values must be strings representing the new axis names.
        """
        _ir_passes.rename_axis(self.model, rename_mapping)

    def _rename_dynamic_axes(
        self,
        dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any],
    ) -> None:
        """Rename dynamic axes in a model according to the specified dynamic_axes names."""
        rename_mapping = _dynamic_shapes.create_rename_mapping(
            self.model.graph.inputs, dynamic_shapes
        )
        _ir_passes.rename_axis(self.model, rename_mapping)


def _process_args(args, kwargs) -> tuple[torch.Tensor, ...]:
    """Process input arguments for the ONNX model."""
    args = _flatten_inputs(args, kwargs)
    args = _remove_none_from_inputs(args)
    args = _convert_complex_to_real_representation(args)
    return args


def _flatten_inputs(model_args, model_kwargs):
    flattened_args, _ = _pytree.tree_flatten((model_args, model_kwargs))
    return flattened_args


def _remove_none_from_inputs(model_args):
    return tuple(arg for arg in model_args if arg is not None)


def _convert_complex_to_real_representation(model_args):
    """Convert complex dtype tensors to real representation tensors.

    ONNX does not support complex dtype tensors. Thus, we convert complex dtype tensors
    to real representation tensors (i.e., float dtype tensors with an extra dimension
    representing the real and imaginary parts of the complex number).
    """
    return tuple(
        torch.view_as_real(arg.resolve_conj())
        if isinstance(arg, torch.Tensor) and arg.is_complex()
        else arg
        for arg in model_args
    )
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.onnx._internal.exporter import _registration


def _arg_has_complex_dtype(arg) -> bool:
    """Check if the node has complex dtype recursively."""
    if (
        isinstance(arg, torch.fx.Node)
        and "val" in arg.meta
        and isinstance(arg.meta["val"], torch.Tensor)
        and torch.is_complex(arg.meta["val"])
    ):
        return True
    elif isinstance(arg, list):
        return any(_arg_has_complex_dtype(item) for item in arg)
    return False


def dispatch(
    node: torch.fx.Node, registry: _registration.ONNXRegistry
) -> tuple[Callable | None, str]:
    """Dispatch a node to an ONNX function based on the node's target and the ONNX registry.

    Args:
        node: The node to dispatch.
        registry: The ONNX registry to use for dispatching.

    Returns:
        A tuple containing the matched ONNX function and a string describing the reason for failure or success.
    """
    decomp_metas = registry.get_decomps(node.target)  # type: ignore[arg-type]
    # Determine if the node has complex inputs.
    is_complex = any(_arg_has_complex_dtype(arg) for arg in node.args) or any(
        _arg_has_complex_dtype(arg) for arg in node.kwargs.values()
    )
    if is_complex:
        decomp_metas = [decomp for decomp in decomp_metas if decomp.is_complex]
        if not decomp_metas:
            return None, "No decompositions registered for the complex-valued input"
    else:
        decomp_metas = [decomp for decomp in decomp_metas if not decomp.is_complex]
        if not decomp_metas:
            return None, "No decompositions registered for the real-valued input"

    # NOTE: Complex overload type matching logic has been removed to keep this simple
    # There should no longer be overloads (for the same opset version) in torchlib anymore
    return (decomp_metas[0].onnx_function, "The first implementation is used")
<|endoftext|>import numpy as np

import torch


def unpack_float4x2_as_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a float4x2 tensor to unpacked uint8 np array."""
    if tensor.dtype != torch.float4_e2m1fn_x2:
        raise AssertionError(f"Expected float4_e2m1fn_x2, got {tensor.dtype}")
    data = tensor.view(torch.uint8).numpy(force=True).flatten()
    result_size = tensor.numel() * 2
    result = np.empty([result_size], dtype=np.uint8)
    array_low = data & np.uint8(0x0F)
    array_high = data & np.uint8(0xF0)
    array_high >>= np.uint8(4)
    result[0::2] = array_low
    result[1::2] = array_high
    result.resize(get_float4_shape(tensor), refcheck=False)
    return result


def get_float4_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    """Get the shape of an unpacked float4 tensor.

    The float4_e2m1fn_x2 type is a shell type described in
    https://github.com/pytorch/pytorch/issues/146414.

    the shell dtype is takes up 1 byte per element and semantically represents
    two fp4 values packed into 1 byte. Semantically it represents (*tensor.shape[:-1], tensor.shape[-1]*2)
    fp4 elements.
    """
    if tensor.dtype != torch.float4_e2m1fn_x2:
        raise AssertionError(f"Expected float4_e2m1fn_x2, got {tensor.dtype}")
    return (*tensor.shape[:-1], tensor.shape[-1] * 2)
<|endoftext|>"""Strategies for capturing ExportedPrograms."""

# mypy: allow-untyped-defs
from __future__ import annotations

import abc
import contextlib
import dataclasses
import datetime
import logging
import pathlib
from typing import Any, TYPE_CHECKING

import torch
from torch.onnx import _flags


if TYPE_CHECKING:
    import os
    from collections.abc import Callable


logger = logging.getLogger(__name__)


def _verbose_printer(verbose: bool | None) -> Callable[..., None]:
    """Prints messages based on `verbose`."""
    if verbose is False:
        return lambda *_, **__: None

    return lambda *args, **kwargs: print("[torch.onnx]", *args, **kwargs)


def _take_first_line(text: str) -> str:
    """Take the first line of a text."""
    lines = text.split("\n", maxsplit=1)
    first_line = lines[0]
    if len(lines) > 1:
        first_line += "[...]"
    return first_line


@contextlib.contextmanager
def _patch_dynamo_unsupported_functions():
    """Patch PyTorch to bypass some functions torch.export.export does not support."""
    # TODO: Remove the patches once dynamo supports these functions.
    import torch.jit

    # Replace torch.jit.isinstance with isinstance
    jit_isinstance = torch.jit.isinstance
    # pyrefly: ignore [bad-assignment]
    torch.jit.isinstance = isinstance
    logger.info("Replaced torch.jit.isinstance with isinstance to allow dynamo tracing")
    try:
        yield
    finally:
        torch.jit.isinstance = jit_isinstance


@contextlib.contextmanager
def _patch_dynamic_shape_rnn_decompositions(dynamic_shapes):
    if not dynamic_shapes:
        yield
        return

    # Import lazily to avoid loading decomposition registrations on torch.onnx import.
    from torch.export._patches import (
        register_gru_while_loop_decomposition,
        register_lstm_while_loop_decomposition,
    )

    with (
        register_lstm_while_loop_decomposition(),
        register_gru_while_loop_decomposition(),
    ):
        yield


@dataclasses.dataclass
class Result:
    exported_program: torch.export.ExportedProgram | None
    strategy: str
    exception: Exception | None = None

    @property
    def success(self) -> bool:
        """Whether the capture was successful.

        An exception can still be recorded even if the capture was successful. In
        this case the exception is informational only. For example, draft_export
        can record an exception if there are warnings during the export. The exceptions
        will go into the onnx export report when report=True.
        """
        return self.exported_program is not None


class CaptureStrategy(abc.ABC):
    """Strategy for capturing a module as ExportedProgram.

    To use a strategy, create an instance and call it with the model, args, kwargs, and dynamic_shapes.
    Example::

        strategy = TorchExportNonStrictStrategy(verbose=True)
        result = strategy(model, args, kwargs, dynamic_shapes)
    """

    def __init__(
        self,
        *,
        verbose: bool = False,
        dump: bool = False,
        artifacts_dir: str | os.PathLike = ".",
        timestamp: str | None = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            verbose: Whether to print verbose messages.
            dump: Whether to dump the intermediate artifacts to a file.
        """
        self._verbose_print = _verbose_printer(verbose)
        self._dump = dump
        self._artifacts_dir = pathlib.Path(artifacts_dir)
        self._timestamp = timestamp or datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S-%f"
        )
        self._exception: Exception | None = None

    def __call__(
        self,
        model: torch.nn.Module | torch.jit.ScriptFunction,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
        dynamic_shapes,
    ) -> Result:
        self._enter(model)
        if kwargs is None:
            kwargs = {}
        try:
            exported_program = self._capture(model, args, kwargs, dynamic_shapes)
        except Exception as e:
            self._failure(model, e)
            return Result(
                exported_program=None,
                strategy=self.__class__.__name__,
                exception=e,
            )
        self._success(model)
        return Result(
            exported_program,
            strategy=self.__class__.__name__,
            exception=self._exception,
        )

    @abc.abstractmethod
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        raise NotImplementedError

    def _enter(self, model: torch.nn.Module | torch.jit.ScriptFunction) -> None:
        return

    def _success(self, model: torch.nn.Module | torch.jit.ScriptFunction) -> None:
        return

    def _failure(
        self, model: torch.nn.Module | torch.jit.ScriptFunction, e: Exception
    ) -> None:
        return


class TorchExportStrictStrategy(CaptureStrategy):
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        with (
            _patch_dynamo_unsupported_functions(),
            _patch_dynamic_shape_rnn_decompositions(dynamic_shapes),
            # Support the dynamism with 0/1 input dim
            torch.fx.experimental._config.patch(backed_size_oblivious=True),  # type: ignore[attr-defined]
        ):
            try:
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=True,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )
            except torch._dynamo.exc.UserError as exc:
                # Refine the dynamic shapes based on the suggested fixes.
                try:
                    new_shapes = torch.export.dynamic_shapes.refine_dynamic_shapes_from_suggested_fixes(
                        exc.msg, dynamic_shapes
                    )
                except Exception:
                    # If the dynamic shapes cannot be refined, re-raise the exception.
                    raise exc from None
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=new_shapes,
                    strict=True,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )

    def _enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=True)`..."
        )

    def _success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=True)`... ✅"
        )

    def _failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=True)`... ❌"
        )


class TorchExportNonStrictStrategy(CaptureStrategy):
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        with (
            _patch_dynamic_shape_rnn_decompositions(dynamic_shapes),
            # Support the dynamism with 0/1 input dim
            torch.fx.experimental._config.patch(backed_size_oblivious=True),  # type: ignore[attr-defined]
        ):
            try:
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )
            except torch._dynamo.exc.UserError as exc:
                # Refine the dynamic shapes based on the suggested fixes.
                try:
                    new_shapes = torch.export.dynamic_shapes.refine_dynamic_shapes_from_suggested_fixes(
                        exc.msg, dynamic_shapes
                    )
                except Exception:
                    # If the dynamic shapes cannot be refined, re-raise the exception.
                    raise exc from None
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=new_shapes,
                    strict=False,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )

    def _enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`..."
        )

    def _success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`... ✅"
        )

    def _failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`... ❌"
        )


class TorchExportDraftExportStrategy(CaptureStrategy):
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        ep = torch.export.draft_export(
            model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
        )
        report = ep._report  # type: ignore[attr-defined]
        if not report.successful():
            self._exception = RuntimeError(str(report))
            self._verbose_print(f"Draft Export report:\n{report}")
        return ep

    def _enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.draft_export`..."
        )

    def _success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.draft_export`... ✅"
        )

    def _failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.draft_export`... ❌"
        )


CAPTURE_STRATEGIES: tuple[type[CaptureStrategy], ...] = (
    TorchExportNonStrictStrategy,  # strict=False is preferred over strict=True because it does not have dynamo issues
    TorchExportStrictStrategy,
)

if _flags.ENABLE_DRAFT_EXPORT:
    CAPTURE_STRATEGIES = (*CAPTURE_STRATEGIES, TorchExportDraftExportStrategy)
<|endoftext|>from __future__ import annotations

import torch
import torch.export
import torch.fx
from torch.onnx._internal.exporter import _decomp, _registration
from torch.onnx._internal.fx import passes


def decompose_with_registry(
    exported_program: torch.export.ExportedProgram, registry: _registration.ONNXRegistry
) -> torch.export.ExportedProgram:
    """Decompose the exported program with the given registry.

    This function is needed so it shows clearly on the profiler results.
    """
    onnx_registered_ops = set(_decomp.get_onnx_implemented_overloads(registry))
    decomp_table = _decomp.create_onnx_friendly_decomposition_table(onnx_registered_ops)
    return exported_program.run_decompositions(decomp_table)


def insert_type_promotion_nodes(
    graph_module: torch.fx.GraphModule,
) -> None:
    """Inplace pass to insert explicit type promotion nodes, recursively through nested modules."""
    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            raise AssertionError(f"Expected GraphModule, got {type(module)}")
        passes.InsertTypePromotion(module).run()


def remove_assertion_nodes(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Remove all assertion and check nodes from the FX graph"""
    aten_assertion_targets = {
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten._assert_async.default,
        torch.ops.aten._assert_async.msg,
        torch.ops.aten._assert_scalar.default,
        torch.ops.aten._assert_tensor_metadata.default,
    }
    for gm in graph_module.modules():
        for node in gm.graph.nodes:  # type: ignore[union-attr]
            if node.op == "call_function" and node.target in aten_assertion_targets:
                gm.graph.erase_node(node)  # type: ignore[operator, union-attr]
        gm.recompile()  # type: ignore[operator]
    return graph_module
<|endoftext|>"""Internal flags for ONNX export."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec


if TYPE_CHECKING:
    from collections.abc import Callable


_is_onnx_exporting = False

# Use ParamSpec to preserve parameter types instead of erasing to Any
_P = ParamSpec("_P")
_R = TypeVar("_R")


def set_onnx_exporting_flag(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        global _is_onnx_exporting
        _is_onnx_exporting = True
        try:
            return func(*args, **kwargs)
        finally:
            # Ensure it resets even if an exception occurs
            _is_onnx_exporting = False

    return wrapper
<|endoftext|>"""NOTES:

We need a typing module that will handling Python to ONNX type promotion for use.
For example, if we have torch.ops.aten.add(Tensor, 1.0), we need to promote 1.0
to the same type as Tensor. The same thing needs to work for
torch.ops.aten.add(1.0, Tensor) as well, which means we need a mechanism to`
"""

# mypy: allow-untyped-defs
# mypy: disable-error-code=union-attr
from __future__ import annotations

import copy
import inspect
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

from onnxscript import evaluator

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir, onnxscript
from torch.onnx._internal.exporter import _errors, _schemas, _tensors


if TYPE_CHECKING:
    import onnx


logger = logging.getLogger(__name__)

ValidAttributeType = (
    ir.TensorProtocol
    | int
    | float
    | bool
    | str
    | Sequence[int]
    | Sequence[float]
    | None
)

AllowedArgType = ir.Value | Sequence[ir.Value | ValidAttributeType] | ValidAttributeType


# Logic for adapting inputs from general Python or PyTorch inputs to ONNX ir.Value
def _construct_named_inputs_and_attrs(
    signature: ir.schemas.OpSignature,
    args: Sequence[AllowedArgType],
    kwargs: Mapping[str, AllowedArgType],
) -> tuple[dict[str, AllowedArgType], dict[str, ValidAttributeType]]:
    """Construct two mappings: name to inputs and named to attributes based on the signature and args/kwargs.

    This function uses the OpSignature to determine which argument in args and kwargs corresponds to
    which parameter in the signature. ONNX node inputs are stored in named_inputs, and attributes are
    stored in named_attrs. If an _optional input_ is not provided, it is filled with None.

    Args:
        signature: The OpSignature for the node.
        args: The positional arguments for the node.
        kwargs: The keyword arguments for the node.

    Returns:
        A tuple of two mappings: named_inputs and named_attrs.

    Raises:
        ValueError: If a required parameter is not provided.
    """
    # 1. Construct the (named_inputs, named_attrs) mapping based on (args, kwargs) and the signature.
    #   a. Loop over all parameters in the signature and args together
    #   b. Depending on param.is_input, Record named_inputs[param.name] = arg or named_attrs[param.name] = arg
    #   c. Handle kwargs as well
    #   d. Fill in None if the input is not provided
    named_inputs: dict[str, Any] = {}
    named_attrs: dict[str, Any] = {}
    reversed_args_stack = list(reversed(args))
    for param in signature.params:
        if isinstance(param, ir.schemas.Parameter):
            # Handle inputs
            if reversed_args_stack:
                # First exhaust the positional arguments
                if param.variadic:
                    # Handle variadic arguments
                    named_inputs[param.name] = tuple(args)
                    reversed_args_stack.clear()
                else:
                    named_inputs[param.name] = reversed_args_stack.pop()  # type: ignore[assignment]
            elif param.name in kwargs:
                named_inputs[param.name] = kwargs[param.name]  # type: ignore[assignment]
            elif param.required:
                raise ValueError(
                    f"Required parameter '{param.name}' is not provided. "
                    f"Signature: {signature}. Args: {args}. Kwargs: {kwargs}."
                )
            else:
                logger.debug(
                    "Optional parameter '%s' is not provided. Added as None. Signature: %s",
                    param.name,
                    signature,
                )
                named_inputs[param.name] = None  # type: ignore[assignment]
        else:
            # Handle attributes
            attribute: ValidAttributeType | ir.Attr
            if not isinstance(param, ir.schemas.AttributeParameter):
                raise AssertionError(f"Expected AttributeParameter, got {type(param)}")
            if reversed_args_stack:
                # First exhaust the positional arguments
                attribute = reversed_args_stack.pop()  # type: ignore[assignment]
            elif param.name in kwargs:
                attribute = kwargs[param.name]  # type: ignore[assignment]
            elif param.default is not None:
                attribute = param.default
            else:
                attribute = None

            if attribute is None:
                if param.required:
                    raise ValueError(
                        f"Required attribute '{param.name}' is not provided. "
                        f"Signature: {signature}. Args: {args}. Kwargs: {kwargs}."
                    )
                else:
                    logger.debug(
                        "Optional attribute '%s' is None. Dropped. Signature: %s",
                        param.name,
                        signature,
                    )
                    continue

            if isinstance(attribute, ir.Attr):
                # Turn the attribute from a default value into an actual parameter for the node
                attr_copied = copy.copy(attribute)
                # Make sure the name is the same as the parameter name and not the name of the default parameter
                attr_copied.name = param.name
                attribute = attr_copied

            if isinstance(attribute, int) and param.type == ir.AttributeType.FLOAT:
                # Convert the attribute to float if needed. This happens in PyTorch
                # where an attribute marked as float can be passed as an int.
                attribute = float(attribute)
            named_attrs[param.name] = attribute
    return named_inputs, named_attrs  # type: ignore[return-value]


def _resolve_parameter_dtypes(
    signature: ir.schemas.OpSignature, named_inputs: Mapping[str, AllowedArgType]
) -> Mapping[ir.schemas.TypeConstraintParam, ir.TypeProtocol]:
    """Determine which parameter takes which type.

    Handle non-tensor input corner cases and type promotion.

    Requires:
        All ir.Value in name_inputs should have type set. Their type should be
        compatible with the type_constraint of the corresponding parameter in the signature.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments.

    Returns:
        A mapping of Constraint names to ir.TypeProtocol.
    """
    #   a. Create type_binding: dict[str, ir.TypeProtocol]
    #   b. Iterate over all named_inputs
    #   b0. Find the corresponding parameter in the signature
    #   b1. If the argument is a Python constant, skip.
    #   b2. If the argument is a ir.Value, Bind {constraint: arg.type}.
    type_binding = {}
    for name, arg in named_inputs.items():
        param = signature.params_map[name]
        if not isinstance(param, ir.schemas.Parameter):
            raise AssertionError(f"Expected Parameter, got {type(param)}")
        if isinstance(arg, (int, float, bool, str, Sequence, torch.Tensor)):
            # Skip the Python constants because we do not know what dtype they should take yet
            continue
        elif isinstance(arg, ir.Value):
            if arg.type is None:
                # Skip the ir.Value if the type is not set
                continue
            # NOTE: We assume arg.type is compatible with the type_constraint
            if arg.type is None:
                raise AssertionError(f"Expected type to be set for {arg}")
            # TODO(justinchuby): Implement type promotion logic here.
            type_binding[param.type_constraint] = arg.type
    return type_binding


def _determine_input_dtype(
    param: ir.schemas.Parameter,
    arg: AllowedArgType,
    type_binding: Mapping[ir.schemas.TypeConstraintParam, ir.TypeProtocol],
) -> ir.DataType:
    """Determine the dtype of the input that is a mix of Python constants and ir.Value."""
    if param.type_constraint in type_binding:
        # A known dtype is available because it was resolved
        return type_binding[param.type_constraint].dtype
    if len(param.type_constraint.allowed_types) == 1:
        # Only one type is allowed by the type constraint
        return next(iter(param.type_constraint.allowed_types)).dtype

    # No dtype information available. Infer from the Python constant or (in the Sequence case)
    # from a mix of Python constants and ir.Value
    if isinstance(arg, bool):
        return ir.DataType.BOOL
    if isinstance(arg, float):
        return ir.DataType.FLOAT
    if isinstance(arg, int):
        return ir.DataType.INT64
    if isinstance(arg, str):
        return ir.DataType.STRING
    if isinstance(arg, (ir.Tensor, ir.TensorProtocol)):
        return arg.dtype
    if isinstance(arg, complex):
        return ir.DataType.FLOAT
    if arg is None:
        return ir.DataType.UNDEFINED

    # Handle sequences
    if isinstance(arg, (tuple, list)):
        if len(arg) == 0:
            # Special case: Treat empty sequence as INT64 as they are typically used for shape
            return ir.DataType.INT64

        # Try to obtain the dtype from one of the values
        for val in arg:
            if isinstance(val, ir.Value) and val.dtype is not None:
                return val.dtype

        if any(isinstance(val, float) for val in arg):
            # If any float is present, the dtype is float
            return ir.DataType.FLOAT
        elif any(isinstance(val, int) for val in arg):
            # Otherwise if any int is present, the dtype is int
            return ir.DataType.INT64

    raise ValueError(
        f"Could not determine the dtype for the input '{param.name}'. "
        f"param={param}, arg={arg}, param_type_constraint={param.type_constraint}, "
        f"type_binding={type_binding}"
    )


def _allowed_types_are_sequence_types(allowed_types: Iterable[ir.TypeProtocol]) -> bool:
    """Check if all allowed types are Sequence types."""
    return all(isinstance(t, ir.SequenceType) for t in allowed_types)


def _get_or_create_constant(
    constant_farm: dict[
        tuple[
            bool
            | int
            | float
            | str
            | tuple[int, ...]
            | tuple[float, ...]
            | tuple[bool, ...],
            ir.DataType,
        ],
        ir.Value,
    ],
    arg: bool
    | int
    | float
    | str
    | tuple[int, ...]
    | tuple[float, ...]
    | tuple[bool, ...]
    | list[int]
    | list[float]
    | list[bool],
    dtype: ir.DataType,
    opset: onnxscript.values.Opset,
) -> ir.Value:
    # float representation of complex numbers
    if isinstance(arg, complex):
        # Convert the complex number to a float
        arg = (arg.real, arg.imag)

    if isinstance(arg, list):
        # Make the arg hashable
        # pyrefly: ignore [bad-argument-type]
        arg = tuple(arg)

    constant_value = constant_farm.get((arg, dtype))  # type: ignore[arg-type]
    if constant_value is None:
        constant_tensor = ir.tensor(value=arg, dtype=dtype)
        constant_value = opset.Constant(value=constant_tensor)
        constant_farm[(arg, dtype)] = constant_value  # type: ignore[arg-type,index]
    return constant_value  # type: ignore[return-value]


def _process_python_constants(
    signature: ir.schemas.OpSignature,
    named_inputs: dict[str, AllowedArgType],
    type_binding: Mapping[ir.schemas.TypeConstraintParam, ir.TypeProtocol],
    constant_farm: dict[
        tuple[
            bool | int | float | str | tuple[int, ...] | tuple[float, ...],
            ir.DataType,
        ],
        ir.Value,
    ],
    opset: onnxscript.values.Opset,
) -> dict[str, ir.Value | None]:
    """Convert Python constants to Constant nodes and list to Sequence nodes based on the dtype information.

    The added constants will be replacing values in named_inputs in place.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments.
        type_binding: A mapping of Constraint names to ir.DataType.
        constant_farm: A dictionary of {(py_value, ir.DataType): ir.Value} to store the deduplicated constants.
        opset: The Opset to use for creating Constant nodes.

    Returns:
        A mapping of parameter names to Python constants converted to constant Nodes.
    """
    # 3. Convert Python constants to Constant nodes based on the dtype information;
    #    construct sequences
    #   a. Iterate over all parameters in the signature the second time
    #   b. If the parameter is in to_resolve_type:
    #       - If param.constraint in type_binding,
    #         Get the constant from constant_farm (deduplicated);
    #            otherwise set named_inputs[param.name] = Constant(value, dtype=type_binding[param.constraint])
    #       - Otherwise, set named_inputs[param.name] = Constant(value)
    for name, arg in named_inputs.items():
        param = signature.params_map[name]
        if not isinstance(param, ir.schemas.Parameter):
            raise AssertionError(f"Expected Parameter, got {type(param)}")

        if isinstance(arg, ir.Value):
            # TODO(justinchuby): Cast the ir.Value here if needed
            continue

        if (
            isinstance(arg, Sequence)
            and len(arg) > 0
            and any(isinstance(val, ir.Value) for val in arg)
        ):
            # Skip the sequence of ir.Value. This is a variadic input or a Sequence input
            # It will be handled by _process_python_sequences
            continue
        if param.variadic:
            # Handled by _process_python_sequences
            continue
        if _allowed_types_are_sequence_types(param.type_constraint.allowed_types):
            # Handled by _process_python_sequences
            continue

        dtype = _determine_input_dtype(param, arg, type_binding)

        if arg is None:
            constant_value = None
        elif isinstance(arg, (ir.Tensor, ir.TensorProtocol)):
            constant_value = opset.Constant(value=arg)
        else:
            # Deduplicate the constants
            constant_value = _get_or_create_constant(constant_farm, arg, dtype, opset)  # type: ignore[arg-type]

        named_inputs[param.name] = constant_value
    return named_inputs  # type: ignore[return-value]


def _reshape_to_1d_tensor(opset: onnxscript.values.Opset, arg: ir.Value) -> ir.Value:
    """Reshape the input to a 1D tensor."""

    return opset.Reshape(
        arg, opset.Constant(value=ir.tensor([-1], dtype=ir.DataType.INT64))
    )


def _process_python_sequences(
    signature: ir.schemas.OpSignature,
    named_inputs: dict[str, AllowedArgType],
    type_binding: Mapping[ir.schemas.TypeConstraintParam, ir.TypeProtocol],
    constant_farm: dict[
        tuple[
            bool
            | int
            | float
            | str
            | ir.TensorProtocol
            | tuple[bool, ...]
            | tuple[int, ...]
            | tuple[float, ...],
            ir.DataType,
        ],
        ir.Value,
    ],
    opset: onnxscript.values.Opset,
):
    """Handle three types of sequences.

    1. Variadic inputs
    2. Sequence input of ir.Value,
    3. Sequence of Python constants that contains ir.Value
    """
    for name, arg in named_inputs.items():
        param = signature.params_map[name]
        if not isinstance(param, ir.schemas.Parameter):
            raise AssertionError(f"Expected Parameter, got {type(param)}")

        if not isinstance(arg, (tuple, list)):
            continue

        if len(arg) == 0:
            # Skip empty sequences
            continue

        # 1. Sequence input of ir.Value
        if _allowed_types_are_sequence_types(param.type_constraint.allowed_types):
            # Turn the list into a Sequence node
            # Constant op creation will be handled by the variadic case below when calling
            # the SequenceConstruct op.
            named_inputs[name] = opset.SequenceConstruct(*arg)
            continue

        # 2. Variadic inputs
        # NOTE: Variadic operators like Max can be called with mixed ir.Value and Python constants
        # like `Max(0, ir.Value())`
        # We need to convert the Python constants to Constant nodes
        if param.variadic:
            if all(isinstance(val, ir.Value) for val in arg):
                # Skip the variadic input if all values are ir.Value
                continue

            dtype = _determine_input_dtype(param, arg, type_binding)
            new_args = []
            for val in arg:
                if isinstance(val, ir.Value):
                    new_args.append(val)
                else:
                    constant_tensor = ir.tensor(value=val, dtype=dtype)  # type: ignore[arg-type]
                    constant_value = opset.Constant(value=constant_tensor)
                    new_args.append(constant_value)
            named_inputs[name] = new_args
            continue
        else:
            # 3. Concat the list as a single input
            # E.g. [Value, 42] should be converted to op.Concat(Value, Constant(42))
            # when the expected input type is INT64
            # We assume this only happens for 0D cases
            if all(isinstance(val, ir.Value) for val in arg):
                expanded_args = [_reshape_to_1d_tensor(opset, val) for val in arg]
                named_inputs[name] = opset.Concat(*expanded_args, axis=0)
                continue

            dtype = _determine_input_dtype(param, arg, type_binding)
            new_args = []
            for val in arg:
                if isinstance(val, ir.Value):
                    new_args.append(_reshape_to_1d_tensor(opset, val))
                elif val is None:
                    # Skip None values
                    continue
                elif isinstance(val, (ir.Tensor, ir.TensorProtocol)):
                    new_args.append(
                        _reshape_to_1d_tensor(opset, opset.Constant(value=val))
                    )
                else:
                    # Turn the Python constant into 1D tensor for the constant
                    if not isinstance(val, (bool, int, float)):
                        raise AssertionError(f"Expected int or float, got {type(val)}")
                    new_args.append(
                        _get_or_create_constant(constant_farm, [val], dtype, opset)  # type: ignore[arg-type]
                    )
            named_inputs[name] = opset.Concat(*new_args, axis=0)
            continue
    return named_inputs


def _determine_output_number(
    signature: ir.schemas.OpSignature, named_attrs: Mapping[str, ValidAttributeType]
) -> int:
    """Determine the number of outputs for the node with heuristics."""
    if signature.domain == "":
        if signature.name == "BatchNormalization":
            if not named_attrs.get("training_mode", 0):
                return 1
        if signature.name == "Split":
            num_outputs = named_attrs.get("num_outputs")
            if num_outputs is not None and isinstance(num_outputs, int):
                return num_outputs
            else:
                raise ValueError(
                    "Could not determine the number of outputs for Split. "
                    "num_outputs must be provided"
                )
    return len(signature.outputs)


def _construct_node(
    signature: ir.schemas.OpSignature,
    named_inputs: Mapping[str, ir.Value | None],
    named_attrs: Mapping[str, ValidAttributeType],
    opset: onnxscript.values.Opset,
    num_outputs: int,
) -> ir.Node:
    """Construct the node with the inputs and attributes.

    Variadic inputs are flattened.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments. When we
            do not have the schema of an operator, we do not know the names of
            the inputs, in which case the names can be anything because they
            are not used in this function. The data structure is passed in for
            consistency with the other functions.
        named_attrs: The mapping of attribute names to their values.
        num_outputs: The number of outputs for the node.
    """
    inputs: list[ir.Value | None] = []
    # Flatten variadic inputs
    for value in named_inputs.values():
        if isinstance(value, Sequence):
            inputs.extend(value)
        else:
            inputs.append(value)

    # If final inputs are None, strip them from the node inputs
    for input in reversed(inputs):
        if input is not None:
            break
        inputs.pop()

    # Construct and filter out None attributes
    attributes = [
        attr
        for attr in ir.convenience.convert_attributes(named_attrs)
        if attr.value is not None
    ]
    outputs = [_tensors.SymbolicTensor(opset) for _ in range(num_outputs)]
    return ir.Node(
        signature.domain,
        signature.name,
        inputs=inputs,
        attributes=attributes,
        outputs=outputs,
        version=signature.since_version,
    )


class OpRecorder(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into ONNX IR."""

    def __init__(
        self, opset: onnxscript.values.Opset, constant_farm: dict[Any, ir.Value]
    ) -> None:
        self.nodes: list[ir.Node] = []
        self.opset = opset
        self.functions: dict[
            ir.OperatorIdentifier, onnxscript.OnnxFunction | ir.Function
        ] = {}
        self.constant_farm = constant_farm

    def _call_op(
        self,
        op_signature: ir.schemas.OpSignature,
        named_inputs: dict[str, AllowedArgType],
        named_attrs: dict[str, ValidAttributeType],
        num_outputs: int,
    ) -> Sequence[_tensors.SymbolicTensor]:
        """Record nodes for the given opschema and arguments.

        Args:
            op_signature: The OpSchema containing the node signature.
            named_inputs: The mapping of parameter names to their arguments.
            named_attrs: The mapping of attribute names to their values.
        """
        type_binding = _resolve_parameter_dtypes(op_signature, named_inputs)
        try:
            converted_named_inputs = _process_python_constants(
                op_signature, named_inputs, type_binding, self.constant_farm, self.opset
            )
            converted_named_inputs = _process_python_sequences(
                op_signature,
                converted_named_inputs,  # type: ignore[arg-type]
                type_binding,
                self.constant_farm,
                self.opset,
            )

        except Exception as e:
            raise _errors.GraphConstructionError(
                f"Error processing Python constants for operator '{op_signature.domain}::{op_signature.name}'. "
                f"named_inputs={named_inputs}, named_attrs={named_attrs}, opset={self.opset}, op_signature={op_signature}."
            ) from e

        try:
            self.nodes.append(
                node := _construct_node(
                    op_signature,
                    converted_named_inputs,
                    named_attrs,
                    self.opset,
                    num_outputs,
                )
            )
        except Exception as e:
            raise _errors.GraphConstructionError(
                f"Error constructing node for operator '{op_signature.domain}::{op_signature.name}'. "
                f"named_inputs={named_inputs}, converted_named_inputs={converted_named_inputs}, "
                f"named_attrs={named_attrs}, opset={self.opset}, op_signature={op_signature}."
            ) from e
        return node.outputs  # type: ignore[return-value]

    def eval(
        self,
        schema: onnx.defs.OpSchema,
        args: Sequence[AllowedArgType],  # type: ignore[override]
        kwargs: Mapping[str, AllowedArgType],
    ) -> _tensors.SymbolicTensor | Sequence[_tensors.SymbolicTensor]:
        try:
            op_signature = ir.schemas.OpSignature.from_op_schema(schema)
            named_inputs, named_attrs = _construct_named_inputs_and_attrs(
                op_signature, args, kwargs
            )
            num_outputs = _determine_output_number(op_signature, named_attrs)
            outputs = self._call_op(
                op_signature, named_inputs, named_attrs, num_outputs
            )
            if len(outputs) == 1:
                return outputs[0]
            return outputs
        except Exception as e:
            raise _errors.GraphConstructionError(
                f"Error calling operator '{schema.name}' with args {args} and kwargs {kwargs}."
            ) from e

    def eval_function(  # type: ignore[override]
        self,
        function: onnxscript.OnnxFunction,
        args: Sequence[AllowedArgType],
        kwargs: Mapping[str, AllowedArgType],
    ) -> _tensors.SymbolicTensor | Sequence[_tensors.SymbolicTensor] | bool | int:
        try:
            # NOTE: signature should be written to function in the registration process
            if hasattr(function, "_pt_onnx_signature"):
                op_signature = function._pt_onnx_signature  # type: ignore[attr-defined]
            else:
                op_signature = _schemas.op_signature_from_function(
                    function,
                    function.function_ir.domain,
                    function.name,
                    since_version=function.opset.version,
                )
                function._pt_onnx_signature = op_signature  # type: ignore[attr-defined]

            named_inputs, named_attrs = _construct_named_inputs_and_attrs(
                op_signature, args, kwargs
            )

            # TODO(after torchlib migration): Remove traceable function handling
            # NOTE: We need to call traceable functions after the _construct_named_inputs_and_attrs
            # call because it will filter out the unexpected kwargs for us.
            if function.traceable:
                # Trace the function call instead of adding the function as a node
                # Turn the ir.Attr objects into Python constants first
                named_attrs = {
                    name: attr.value if isinstance(attr, ir.Attr) else attr
                    for name, attr in named_attrs.items()
                }

                # Use the type binding to resolve the dtypes of the inputs, and
                # convert Python constants to Constant nodes
                type_binding = _resolve_parameter_dtypes(op_signature, named_inputs)
                try:
                    # _process_python_sequences is not here because we want to preserve python list
                    # properties for the function call
                    converted_named_inputs = _process_python_constants(
                        op_signature,
                        named_inputs,
                        type_binding,
                        self.constant_farm,
                        self.opset,
                    )

                except Exception as e:
                    raise _errors.GraphConstructionError(
                        f"Error processing Python constants for operator '{op_signature.domain}::{op_signature.name}'. "
                        f"named_inputs={named_inputs}, named_attrs={named_attrs}, opset={self.opset}, op_signature={op_signature}."
                    ) from e

                return function.function(**converted_named_inputs, **named_attrs)

            outputs = self._call_op(
                op_signature,
                named_inputs,
                named_attrs,
                len(op_signature.outputs),
            )

            self.functions[(function.function_ir.domain, function.name, "")] = function
            if len(outputs) == 1:
                return outputs[0]
            return outputs
        except Exception as e:
            try:
                source_file = inspect.getsourcefile(function.function)
                _, lineno = inspect.getsourcelines(function.function)
            except Exception:
                source_file = lineno = None
            raise _errors.GraphConstructionError(
                f"Error calling function '{function.name}' with args {args} and kwargs {kwargs}."
                + f" The function is defined at '{source_file}:{lineno}'."
                if source_file
                else ""
            ) from e
<|endoftext|>"""Compatibility functions for the torch.onnx.export API."""

# mypy: allow-untyped-defs
from __future__ import annotations

import inspect
import warnings
from typing import Any, TYPE_CHECKING

import torch
from torch.export.dynamic_shapes import _DimHint, Dim
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.utils import _pytree


if TYPE_CHECKING:
    from collections.abc import Sequence


def from_dynamic_axes_to_dynamic_shapes(
    model,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    *,
    dynamic_axes=None,
    output_names: set[str],
    input_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any | None] | None, tuple[Any, ...], dict[str, Any] | None]:
    """
    Converts dynamic_axes into dynamic_shapes by wrapping the axis names with ``torch.export.Dim.DYNAMIC``.

    dynamic_axes examples:
    (1) dynamic_axes = {"x": {0: "my_custom_axis_name_1"}, "y": {1: "my_custom_axis_name_2"}}
    (2) dynamic_axes = {"x": [0], "y": [1]}

    these will be converted to dynamic_shapes respectively:
    (1) dynamic_shapes = {"x": {0: Dim.DYNAMIC}, "y": {1: Dim.DYNAMIC}}
    (2) dynamic_shapes = {"x": {0: Dim.DYNAMIC}, "y": {1: Dim.DYNAMIC}}

    Detail on Dim.DYNAMIC: `#133620 <https://github.com/pytorch/pytorch/pull/133620>`_
    """

    warnings.warn(
        "from_dynamic_axes_to_dynamic_shapes is deprecated and will be removed in a future release. "
        "This function converts 'dynamic_axes' format (including custom axis names) to 'dynamic_shapes' format. "
        "Instead of relying on this conversion, provide 'dynamic_shapes' directly with custom names.",
        DeprecationWarning,
        stacklevel=2,
    )

    # https://github.com/pytorch/pytorch/pull/128371
    # 1. The function does not need to provide dynamic_shapes to torch.export.export
    if dynamic_axes is None:
        return None, args, kwargs

    if input_names is None:
        input_names = []

    if kwargs is None:
        kwargs = {}

    dynamic_shapes: dict[str, Any | None] = {}
    for input_name, axes in dynamic_axes.items():
        # NOTE: torch.export.Dim.DYNAMIC does its best to infer the min and max values
        # from the model, but it's not guaranteed to be dynamic.
        if input_name in output_names:
            # output names are not needed for dynamic_shapes
            continue
        if isinstance(axes, dict):
            if any(not isinstance(k, int) for k in axes):
                raise ValueError(
                    "The axis in dynamic_axes must be in the form of: dict[int, str] or list[int]."
                )
            # str will be converted to Dim.DYNAMIC in convert_str_to_export_dim
            dynamic_shapes[input_name] = axes
        elif isinstance(axes, list):
            if any(not isinstance(k, int) for k in axes):
                raise ValueError(
                    "The axis in dynamic_axes must be in the form of: dict[int, str] or list[int]."
                )
            dynamic_shapes[input_name] = dict.fromkeys(axes, torch.export.Dim.DYNAMIC)
        elif axes is None:
            dynamic_shapes[input_name] = None
        else:
            raise ValueError(
                "Unsupported dynamic_axes format. Please provide a dict or a list."
            )

    for input_name in input_names:
        if input_name not in dynamic_shapes:
            dynamic_shapes[input_name] = None

    # Order the inputs according to the signature of the model
    sig = _signature(model)
    inputs = []
    for idx, param_name in enumerate(sig.parameters):
        if idx < len(args):
            inputs.append(args[idx])
        elif param_name in kwargs:
            inputs.append(kwargs[param_name])

    # We need tree structure to represent dynamic_shapes
    dynamic_shapes = _unflatten_dynamic_shapes_with_inputs_tree(inputs, dynamic_shapes)

    # Since the dynamic_shapes are now in the order of the model parameters,
    # we need to convert args and kwargs to the order of the model parameters.
    return dynamic_shapes, tuple(inputs), {}


def from_dynamic_shapes_to_dynamic_axes(
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any],
    input_names: Sequence[str],
    exception: Exception,
) -> dict[str, Any] | None:
    """
    Converts dynamic_shapes into dynamic_axes by removing torch.export.Dim wrapping
    and converting to list or dict form based on whether dimension names are present.

    dynamic_shapes examples:
    (1) dynamic_shapes = {"x": {0: Dim("my_custom_axis_name_1")}, "y": {1: Dim("my_custom_axis_name_2")}}
    (2) dynamic_shapes = ({0: Dim("my_custom_axis_name_1"}, {1: Dim("my_custom_axis_name_2")})

    these will be converted to dynamic_axes respectively:
    (1) dynamic_axes = {"x": [0], "y": [1]}
    (2) dynamic_axes = {"x": [0], "y": [1]}

    NOTE: If the model input is nested, so is the dynamic_shapes, we need to flatten the dynamic_shapes,
    and then assign the axes to the input names in the order they are provided.

    NOTE: input_names are used to assign the axes to the correct input names. If the input names are not
    provided, or less than the dynamic inputs/axes, it raises an error.
    """

    flat_dynamic_shapes, _ = _flatten_dynamic_shapes_to_axes(dynamic_shapes)

    if len(input_names) < len(flat_dynamic_shapes):
        raise ValueError(
            "To construct dynamic_axes from dynamic_shapes, "
            f"number of input names ({len(input_names)}) should be greater than or equal to "
            f"the number of graph inputs(flat) ({len(flat_dynamic_shapes)})"
        ) from exception

    dynamic_axes: dict[str, list[int]] = {}
    # input names are assigned in order
    for input_name, axes in zip(input_names, flat_dynamic_shapes):
        if axes is None:
            continue

        converted_axes: list[int] = []
        if isinstance(axes, dict):
            for axis, dim in axes.items():
                if dim is None:
                    continue
                converted_axes.append(axis)
            dynamic_axes[input_name] = converted_axes
        elif isinstance(axes, (list, tuple)):
            for idx, dim in enumerate(axes):
                if dim is None:
                    continue
                converted_axes.append(idx)
            dynamic_axes[input_name] = converted_axes
    return dynamic_axes


def _any_str_or_dim_in_dynamic_shapes(
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any],
) -> bool:
    """Check if there is any string or Dim in the dynamic_shapes."""
    flat_dynamic_shapes, _ = _flatten_dynamic_shapes_to_axes(dynamic_shapes)
    # This indicates the dynamic_shapes includes something we don't support in axes, and it's flattened
    # to itself. Otherwise, flat_dynamic_shapes should be a list of dict/list/tuple (or None).
    if any(
        not isinstance(axes, (dict, list, tuple)) and axes is not None
        for axes in flat_dynamic_shapes
    ):
        return False
    # both str and Dim can provide custom names
    for axes in flat_dynamic_shapes:
        if isinstance(axes, dict):
            for dim in axes.values():
                if isinstance(dim, (str, Dim)):
                    return True
        elif isinstance(axes, (list, tuple)):
            for dim in axes:
                if isinstance(dim, (str, Dim)):
                    return True
    return False


def convert_str_to_export_dim(
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
) -> tuple[dict[str, Any] | tuple[Any, ...] | list[Any] | None, bool]:
    # 1. If there is no string in dynamic_shapes, we do not touch dynamic_shapes
    if dynamic_shapes is None or not _any_str_or_dim_in_dynamic_shapes(dynamic_shapes):
        return dynamic_shapes, False
    # 2. Convert "name" to Dim.DYNAMIC with flattening and identify if there is any string
    #    to be replaced with Dim.DYNAMIC, and then unflatten it back to the original structure.
    #    for example: {"y": {0: "dim_0"}, "x": {1: "dim_1"}}
    #    to {"y": {0: Dim.DYNAMIC}, "x": {1: Dim.DYNAMIC}}
    dynamic_shapes_with_export_dim: list[
        list[Dim | _DimHint | None] | dict[int, Dim | _DimHint | None] | None
    ] = []
    flat_dynamic_shapes, tree_structure = _flatten_dynamic_shapes_to_axes(
        dynamic_shapes
    )
    for axes in flat_dynamic_shapes:
        if axes is None:
            dynamic_shapes_with_export_dim.append(None)
        elif isinstance(axes, dict):
            converted_axes_dict: dict[int, Dim | _DimHint | None] = {}
            for axis, dim in axes.items():
                if isinstance(dim, str):
                    converted_axes_dict[axis] = torch.export.Dim.DYNAMIC
                else:
                    converted_axes_dict[axis] = dim
            dynamic_shapes_with_export_dim.append(converted_axes_dict)
        elif isinstance(axes, (list, tuple)):
            converted_axes_list: list[Dim | _DimHint | None] = []
            for dim in axes:
                if isinstance(dim, str):
                    converted_axes_list.append(torch.export.Dim.DYNAMIC)
                else:
                    converted_axes_list.append(dim)
            dynamic_shapes_with_export_dim.append(converted_axes_list)

    dynamic_shapes_with_export_dim = _pytree.tree_unflatten(
        dynamic_shapes_with_export_dim, tree_structure
    )
    return (
        dynamic_shapes_with_export_dim,
        True,
    )


def create_rename_mapping(
    inputs, dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any]
) -> dict[str, str]:
    """Create a mapping from old names to new names for dynamic axes."""

    # NOTE: There's no need to handle cases where kwargs are out of order with the model signature,
    # as torch.export.export supports dynamism only when kwargs and dynamic_shapes are provided in order.
    # Reference: https://github.com/pytorch/pytorch/blob/49082f9dba3b79a344cb03652972ddbe7c3729cc/torch/export/_trace.py#L2034

    flat_dynamic_shapes, _ = _flatten_dynamic_shapes_to_axes(dynamic_shapes)
    if len(inputs) != len(flat_dynamic_shapes):
        warnings.warn(
            "# ONNX model has different number of inputs than the flatten dynamic_shapes. "
            "The dynamic axes will not be renamed.",
            UserWarning,
            stacklevel=3,
        )
        return {}
    rename_mapping: dict[str, str] = {}
    # NOTE: We assume that the flat_dynamic_shapes is in the same order as the inputs
    # When the axis is static, or it connects to _DimHint in dynamic shapes, we skip renaming
    for idx, axes in enumerate(flat_dynamic_shapes):
        input = inputs[idx]
        if isinstance(axes, dict):
            for dim, axis in axes.items():
                if not isinstance(input.shape[dim], ir.SymbolicDim):
                    continue
                old_name = input.shape[dim].value
                if old_name is None:
                    continue
                # _DimHint, int and None exists in dynamic shapes, we skip renaming
                if isinstance(axis, (_DimHint, int)) or axis is None:
                    continue
                # NOTE: ExportedProgram could give the axes the same name if they share
                # the same shape constraints.
                custom_name = _get_custom_axis_name(axis)
                if input.shape[dim].value in rename_mapping:
                    warnings.warn(
                        f"# The axis name: {custom_name} will not be used, since it shares "
                        f"the same shape constraints with another axis: {rename_mapping[input.shape[dim].value]}.",
                        stacklevel=2,
                    )
                    continue
                rename_mapping[input.shape[dim].value] = custom_name
        elif isinstance(axes, (list, tuple)):
            for dim, axis in enumerate(axes):
                if not isinstance(input.shape[dim], ir.SymbolicDim):
                    continue
                old_name = input.shape[dim].value
                if old_name is None:
                    continue
                # _DimHint, int and None exists in dynamic shapes, we skip renaming
                if isinstance(axis, (_DimHint, int)) or axis is None:
                    continue
                # NOTE: ExportedProgram could give the axes the same name if they share
                # the same shape constraints.
                custom_name = _get_custom_axis_name(axis)
                if input.shape[dim].value in rename_mapping:
                    warnings.warn(
                        f"# The axis name: {custom_name} will not be used, since it shares "
                        f"the same shape constraints with another axis: {rename_mapping[input.shape[dim].value]}.",
                        UserWarning,
                        stacklevel=3,
                    )
                    continue
                rename_mapping[input.shape[dim].value] = _get_custom_axis_name(axis)
    return rename_mapping


def _get_custom_axis_name(axis: Dim | str) -> str:
    """Get the custom axis name from a torch.export.Dim."""
    if isinstance(axis, Dim):
        return axis.__name__
    return axis


def _unflatten_dynamic_shapes_with_inputs_tree(
    inputs: list[Any],
    dynamic_shapes: dict[str, Any],
) -> dict[str, Any | None]:
    _, tree_structure = _pytree.tree_flatten(inputs)
    return _pytree.tree_unflatten(dynamic_shapes.values(), tree_structure)


def _flatten_dynamic_shapes_to_axes(
    dynamic_shapes: dict[str, Any | None] | tuple[Any, ...] | list[Any],
) -> tuple[list[Any], _pytree.TreeSpec]:
    # If it's a dict/list/tuple with torch.export.Dim, we consider it's an axis to dim mapping
    def is_axes(x) -> bool:
        return (
            isinstance(x, dict)
            and all(
                isinstance(k, int)
                and (v is None or isinstance(v, (Dim, _DimHint, str, int)))
                for k, v in x.items()
            )
        ) or (
            isinstance(x, (list, tuple))
            and all(v is None or isinstance(v, (Dim, _DimHint, str, int)) for v in x)
        )

    return _pytree.tree_flatten(dynamic_shapes, is_leaf=is_axes)


def _signature(model) -> inspect.Signature:
    should_be_callable = getattr(model, "forward", model)
    if callable(should_be_callable):
        return inspect.signature(should_be_callable)
    raise ValueError("model has no forward method and is not callable")
<|endoftext|># mypy: allow-untyped-defs
"""Helpers for constructing ONNX operator signatures from Python functions."""

from __future__ import annotations

import collections.abc
import inspect
import logging
import types
import typing
from collections.abc import Sequence
from typing import Any, Optional, TypeVar, Union

from torch.onnx._internal._lazy_import import onnx_ir as ir, onnxscript


logger = logging.getLogger(__name__)


# Map from python type to corresponding ONNX AttributeProto type
_PY_TYPE_TO_ATTR_TYPE = {
    float: ir.AttributeType.FLOAT,
    int: ir.AttributeType.INT,
    str: ir.AttributeType.STRING,
    bool: ir.AttributeType.INT,
    ir.Tensor: ir.AttributeType.TENSOR,
    ir.TensorProtocol: ir.AttributeType.TENSOR,
    ir.Graph: ir.AttributeType.GRAPH,
    ir.GraphProtocol: ir.AttributeType.GRAPH,
}

# Map from python type to corresponding ONNX AttributeProto type,
# for repeated (i.e., list of) values
_LIST_TYPE_TO_ATTR_TYPE = {
    float: ir.AttributeType.FLOATS,
    int: ir.AttributeType.INTS,
    str: ir.AttributeType.STRINGS,
    bool: ir.AttributeType.INTS,
    ir.Tensor: ir.AttributeType.TENSORS,
    ir.TensorProtocol: ir.AttributeType.TENSORS,
    ir.Graph: ir.AttributeType.GRAPHS,
    ir.GraphProtocol: ir.AttributeType.GRAPHS,
}

_ALL_VALUE_TYPES = (
    {ir.TensorType(dtype) for dtype in ir.DataType}
    | {ir.SequenceType(ir.TensorType(dtype)) for dtype in ir.DataType}
    | {ir.OptionalType(ir.TensorType(dtype)) for dtype in ir.DataType}
)

# TypeAnnotationValue represents the (value of) valid type-annotations recognized
# by ONNX Script. Currently, it supports
# - float, int, str (primitive attribute types)
# - Sequence[float], Sequence[int], Sequence[str] (attribute types)
# - Tensor types
# - Sequence[Tensor] types
# - Union of above 2
# - TypeVars with above bounds
# - Above types with annotation attached
TypeAnnotationValue = Any


def _is_optional(type_: type) -> bool:
    """Returns whether a type_ is an Optional."""
    origin_type = typing.get_origin(type_)
    if origin_type is Union and type(None) in typing.get_args(type_):
        # Python < 3.10
        return True
    if origin_type is Optional:
        # Python >= 3.10
        return True
    if (
        hasattr(types, "UnionType")
        and origin_type is types.UnionType
        and type(None) in typing.get_args(type_)
    ):
        # Python >= 3.10
        return True
    return False


def _get_attr_type(type_: type) -> ir.AttributeType:
    """Obtain the type of the attribute from a Python class."""
    try:
        if type_ in _PY_TYPE_TO_ATTR_TYPE:
            return _PY_TYPE_TO_ATTR_TYPE[type_]
        origin_type = typing.get_origin(type_)
        if origin_type is None:
            return ir.AttributeType.UNDEFINED
        if origin_type in (
            collections.abc.Sequence,
            Sequence,
            list,
            list,
            tuple,
            tuple,
        ):
            inner_type = typing.get_args(type_)[0]
            if inner_type in _LIST_TYPE_TO_ATTR_TYPE:
                return _LIST_TYPE_TO_ATTR_TYPE[inner_type]
    except TypeError:
        logger.warning("TypeError when checking %s.", type_, exc_info=True)
    return ir.AttributeType.UNDEFINED


def _get_type_constraint_name(type_: TypeAnnotationValue) -> str | None:
    """Returns the name of the type constraint for a given type annotation.

    Args:
        type_: A Python type.

    Returns:
        The name of the type constraint if it is a TypeVar.
        - Prefixes the name with "Sequence_" if the type annotation is a Sequence[].
    """
    if isinstance(type_, TypeVar):
        return type_.__name__
    if _is_optional(type_):
        subtypes = typing.get_args(type_)
        for subtype in subtypes:
            if subtype is type(None):
                continue
            type_param_name = _get_type_constraint_name(subtype)
            return type_param_name if type_param_name else None
    origin_type = typing.get_origin(type_)
    if isinstance(origin_type, type) and issubclass(origin_type, Sequence):
        subtypes = typing.get_args(type_)
        type_param_name = _get_type_constraint_name(subtypes[0])
        return f"Sequence_{type_param_name}" if type_param_name else None
    return None


def _get_allowed_types_from_type_annotation(
    type_: TypeAnnotationValue,
) -> set[ir.TypeProtocol]:
    """Obtain the allowed types from a type annotation."""
    if type_ is onnxscript.onnx_types.TensorType:
        # Any tensor type
        return {ir.TensorType(dtype) for dtype in ir.DataType}

    allowed_types: set[ir.TypeProtocol]

    if isinstance(type_, TypeVar):
        allowed_types = set()
        if constraints := type_.__constraints__:
            for constraint in constraints:
                allowed_types.update(
                    _get_allowed_types_from_type_annotation(constraint)
                )
        else:
            bound = type_.__bound__
            if bound is None:
                allowed_types = _ALL_VALUE_TYPES  # type: ignore[assignment]
            else:
                allowed_types.update(_get_allowed_types_from_type_annotation(bound))
        return allowed_types
    if hasattr(type_, "dtype"):
        # A single tensor type like INT64, FLOAT, etc.
        return {ir.TensorType(ir.DataType(type_.dtype))}
    if _is_optional(type_):
        allowed_types = set()
        subtypes = typing.get_args(type_)
        for subtype in subtypes:
            if subtype is type(None):
                continue
            allowed_types.update(_get_allowed_types_from_type_annotation(subtype))
        # NOTE: We do not consider dynamic optional types like optional(float) because they are not very useful.
        return allowed_types

    origin_type = typing.get_origin(type_)
    if origin_type is Union:
        allowed_types = set()
        subtypes = typing.get_args(type_)
        for subtype in subtypes:
            if subtype is type(None):
                raise AssertionError(
                    "Union should not contain None type because it is handled by _is_optional."
                )
            allowed_types.update(_get_allowed_types_from_type_annotation(subtype))
        return allowed_types

    if isinstance(origin_type, type) and issubclass(origin_type, Sequence):
        subtypes = typing.get_args(type_)
        return {
            ir.SequenceType(t)
            for t in _get_allowed_types_from_type_annotation(subtypes[0])
        }

    # Allow everything by default
    return _ALL_VALUE_TYPES  # type: ignore[return-value]


def op_signature_from_function(
    func,
    domain: str,
    name: str | None = None,
    overload: str = "",
    *,
    since_version: int = 1,
) -> ir.schemas.OpSignature:
    """Produce an OpSignature from a function using type annotation."""

    py_signature = inspect.signature(func)
    # Not using inspect.get_annotations because typing.get_type_hints seems to handle more cases
    # https://github.com/python/cpython/issues/102405
    type_hints = typing.get_type_hints(func)

    params: list[ir.schemas.Parameter | ir.schemas.AttributeParameter] = []
    # Create a mapping from type to a unique name
    type_constraints: dict[str, ir.schemas.TypeConstraintParam] = {}

    for param in py_signature.parameters.values():
        if param.name not in type_hints:
            logger.debug(
                "Missing annotation for parameter '%s' from %s. Treating as an Input.",
                param.name,
                py_signature,
            )
            type_constraint = ir.schemas.TypeConstraintParam.any_value(
                f"T_{param.name}"
            )
            type_constraints[param.name] = type_constraint
            kwargs: dict[str, Any] = {}
            if param.default is not inspect.Parameter.empty:
                kwargs["default"] = param.default
            params.append(
                ir.schemas.Parameter(
                    name=param.name,
                    type_constraint=type_constraint,
                    required=param.default is inspect.Parameter.empty,
                    # TODO: Handle variadic
                    variadic=False,
                    **kwargs,
                )
            )
        else:
            type_ = type_hints[param.name]
            if (attr_type := _get_attr_type(type_)) != ir.AttributeType.UNDEFINED:
                # Construct the default attribute
                if param.default is not inspect.Parameter.empty:
                    # TODO: Use ir_convenience instead to handle int as float
                    default = ir.Attr(param.name, attr_type, param.default)
                else:
                    default = None
                params.append(
                    ir.schemas.AttributeParameter(
                        name=param.name,
                        type=attr_type,
                        required=param.default is inspect.Parameter.empty,
                        default=default,
                    )
                )
            else:
                # Obtain the type constraint from the type annotation

                # 1. Get a type constraint name from the type annotation
                # If the type annotation is a TypeVar or Optional[TypeVar], get its name
                # Otherwise, name it T_{param.name}
                type_constraint_name = _get_type_constraint_name(type_)
                if type_constraint_name is None:
                    type_constraint_name = f"T_{param.name}"

                # 2. If the type constraint param is already initialized, use it
                if type_constraint_name in type_constraints:
                    type_constraint = type_constraints[type_constraint_name]
                else:
                    # 3. Otherwise, create a new TypeConstraintParam
                    type_constraint = ir.schemas.TypeConstraintParam(
                        name=type_constraint_name,
                        allowed_types=_get_allowed_types_from_type_annotation(type_),
                    )
                    type_constraints[type_constraint_name] = type_constraint
                # 4. Create Parameter
                kwargs: dict[str, Any] = {}
                if param.default is not inspect.Parameter.empty:
                    kwargs["default"] = param.default
                params.append(
                    ir.schemas.Parameter(
                        name=param.name,
                        type_constraint=type_constraint,
                        required=param.default is inspect.Parameter.empty,
                        # TODO: Handle variadic
                        variadic=False,
                        **kwargs,
                    )
                )

    return_type = type_hints.get("return")

    outputs = []
    if return_type is None:
        # No returns
        pass
    else:
        if typing.get_origin(return_type) is tuple:
            # Multiple returns
            return_types = typing.get_args(return_type)
        else:
            return_types = [return_type]  # type: ignore[assignment]

        for i, return_type_i in enumerate(return_types):
            if (
                return_param_name := _get_type_constraint_name(return_type_i)
            ) in type_constraints:
                # pyrefly: ignore [bad-index]
                type_constraint = type_constraints[return_param_name]
            else:
                return_param_name = f"TReturn{i}"
                type_constraint = ir.schemas.TypeConstraintParam(
                    name=return_param_name,
                    allowed_types=_get_allowed_types_from_type_annotation(
                        return_type_i
                    ),
                )
                type_constraints[return_param_name] = type_constraint
            outputs.append(
                ir.schemas.Parameter(
                    name=return_param_name,
                    type_constraint=type_constraint,
                    required=True,
                    variadic=False,
                )
            )

    return ir.schemas.OpSignature(
        domain=domain,
        name=name or func.__name__,
        overload=overload,
        params=params,
        outputs=outputs,
        since_version=since_version,
    )
<|endoftext|>from __future__ import annotations


__all__ = [
    "VerificationInfo",
    "verify_onnx_program",
]

import dataclasses
import logging
import math
from typing import Any, TYPE_CHECKING

import torch
from torch.utils import _pytree


if TYPE_CHECKING:
    from torch.onnx._internal._lazy_import import onnx_ir as ir
    from torch.onnx._internal.exporter import _onnx_program


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VerificationInfo:
    """Verification information for a value in the ONNX program.

    This class contains the maximum absolute difference, maximum relative difference,
    and histograms of absolute and relative differences between the expected and actual
    values. It also includes the expected and actual data types.

    The histograms are represented as tuples of tensors, where the first tensor is the
    histogram counts and the second tensor is the bin edges.

    Attributes:
        name: The name of the value (output or intermediate).
        max_abs_diff: The maximum absolute difference between the expected and actual values.
        max_rel_diff: The maximum relative difference between the expected and actual values.
        abs_diff_hist: A tuple of tensors representing the histogram of absolute differences.
            The first tensor is the histogram counts and the second tensor is the bin edges.
        rel_diff_hist: A tuple of tensors representing the histogram of relative differences.
            The first tensor is the histogram counts and the second tensor is the bin edges.
        expected_dtype: The data type of the expected value.
        actual_dtype: The data type of the actual value.
    """

    name: str
    max_abs_diff: float
    max_rel_diff: float
    abs_diff_hist: tuple[torch.Tensor, torch.Tensor]
    rel_diff_hist: tuple[torch.Tensor, torch.Tensor]
    expected_dtype: torch.dtype
    actual_dtype: torch.dtype
    # NOTE: We don't need to include shape because the expected shape is already known
    # and checked by the runtime

    @classmethod
    def from_tensors(
        cls,
        name: str,
        expected: torch.Tensor | float | int | bool,
        actual: torch.Tensor | float | int | bool,
    ) -> VerificationInfo:
        """Create a VerificationInfo object from two tensors.

        Args:
            name: The name of the value.
            expected: The expected tensor.
            actual: The actual tensor.

        Returns:
            VerificationInfo: The VerificationInfo object.
        """
        if not isinstance(expected, torch.Tensor):
            expected = torch.tensor(expected)
        if not isinstance(actual, torch.Tensor):
            actual = torch.tensor(actual)

        max_abs_diff, max_rel_diff, abs_diff, rel_diff = _compare_tensors(
            expected, actual
        )
        bins = torch.tensor(
            [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 1000000],
            dtype=torch.float,
        )
        abs_diff_hist = torch.histogram(abs_diff.float(), bins=bins)
        rel_diff_hist = torch.histogram(rel_diff.float(), bins=bins)
        return cls(
            name=name,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            abs_diff_hist=abs_diff_hist,
            rel_diff_hist=rel_diff_hist,
            expected_dtype=expected.dtype,
            actual_dtype=actual.dtype,
        )

    def asdict(self) -> dict[str, Any]:
        """Convert the VerificationInfo object to a dictionary.

        Returns:
            A dictionary representation of the VerificationInfo object.
        """
        return {
            "name": self.name,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "abs_diff_hist": [
                self.abs_diff_hist[0].tolist(),
                self.abs_diff_hist[1].tolist(),
            ],
            "rel_diff_hist": [
                self.rel_diff_hist[0].tolist(),
                self.rel_diff_hist[1].tolist(),
            ],
            "expected_dtype": str(self.expected_dtype),
            "actual_dtype": str(self.actual_dtype),
        }


def _compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    # Move tensors to the same device
    expected = expected.detach().cpu()
    actual = actual.detach().cpu()
    if expected.numel() == 0 or actual.numel() == 0:
        return math.inf, math.inf, torch.tensor(math.inf), torch.tensor(math.inf)
    if expected.dtype == torch.bool:
        expected = expected.to(torch.float32)
        actual = actual.to(torch.float32)
    if torch.is_complex(expected):
        expected = torch.view_as_real(expected)
    abs_diff = torch.abs(expected - actual)
    eps = 1e-7
    normalizer = torch.abs(expected) + eps
    rel_diff = abs_diff / normalizer

    max_absolute_difference = abs_diff.max().item()
    max_relative_difference = rel_diff.max().item()

    return max_absolute_difference, max_relative_difference, abs_diff, rel_diff


def verify_onnx_program(
    onnx_program: _onnx_program.ONNXProgram,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    compare_intermediates: bool = False,
) -> list[VerificationInfo]:
    """Verify the ONNX model by comparing the values with the expected values from ExportedProgram.

    Args:
        onnx_program: The ONNX program to verify.
        args: The input arguments for the model.
        kwargs: The keyword arguments for the model.
        compare_intermediates: Whether to verify intermediate values. This is going
            to take longer time, so it is disabled by default.

    Returns:
        VerificationInfo objects containing the verification information for each value.
    """
    exported_program = onnx_program.exported_program
    if exported_program is None:
        raise ValueError(
            "The ONNX program does not contain an exported_program. "
            "Please provide an exported_program to verify the ONNX program."
        )
    if args is None and kwargs is None:
        # User did not provide example inputs, use the default example inputs
        if exported_program.example_inputs is None:
            raise ValueError(
                "No example inputs provided and the exported_program does not contain example inputs. "
                "Please provide arguments to verify the ONNX program."
            )
        args, kwargs = exported_program.example_inputs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    # Flatten args for ONNX program and the VerificationInterpreter
    flat_args, _ = exported_program._get_flat_args_with_check(args, kwargs)

    if not compare_intermediates:
        # Compare the output values
        torch_outputs, _ = _pytree.tree_flatten(
            exported_program.module()(*args, **kwargs)
        )
        onnx_outputs = onnx_program(*flat_args)
        results = []
        for torch_output, onnx_output, output_val in zip(
            torch_outputs, onnx_outputs, onnx_program.model.graph.outputs
        ):
            results.append(
                VerificationInfo.from_tensors(
                    name=str(output_val.name),
                    expected=torch_output,
                    actual=onnx_output,
                )
            )
        return results

    # Use the _VerificationInterpreter to get the intermediate values
    # By design the output values are included too
    interpreter = _VerificationInterpreter(onnx_program)
    interpreter.run(*flat_args)

    return interpreter.verification_infos


def _create_value_mapping(graph: ir.Graph) -> dict[str, ir.Value]:
    """Return a dictionary mapping names to values in the graph.

    The mapping does not include values from subgraphs.

    Args:
        graph: The graph to extract the mapping from.

    Returns:
        A dictionary mapping names to values.
    """
    values: dict[str, ir.Value] = {}
    values.update(graph.initializers)
    # The names of the values can be None or "", which we need to exclude
    for input in graph.inputs:
        if not input.name:
            continue
        values[input.name] = input
    for node in graph:
        for value in node.outputs:
            if not value.name:
                continue
            values[value.name] = value
    return values


class _VerificationInterpreter(torch.fx.Interpreter):
    """Interpreter for verifying converted ONNX model accuracy by comparing intermediate values.

    To compare models, first initialize the interpreter with an ONNX program.
    Then, call the :meth:`run` method with the input arguments to execute the model.
    The :meth:`run` method will execute the model and populate the
    :attr:`verification_infos` attribute with the verification information for each value.

    ::
        onnx_program = torch.onnx.export(model, args, dynamo=True)
        interpreter = _VerificationInterpreter(onnx_program)
        interpreter.run(*args)
        verification_infos = interpreter.verification_infos
        for info in verification_infos:
            print("value name:", info.name, info)

    The verification information includes the maximum absolute difference, maximum relative
    difference, and histograms of absolute and relative differences between the expected
    and actual values. See :class:`VerificationInfo` for more details.

    Attributes:
        verification_infos: A list of verification information for each value.
            It is populated when the `run` method is called.
    """

    def __init__(self, onnx_program: torch.onnx.ONNXProgram) -> None:
        """Initialize the _VerificationInterpreter with an ONNX program.

        Args:
            onnx_program: The ONNX program to verify.
        """
        if onnx_program.exported_program is None:
            raise ValueError(
                "The ONNX program does not contain an exported_program. "
                "Please provide an exported_program to verify the ONNX program."
            )
        super().__init__(onnx_program.exported_program.module())
        self._onnx_program = onnx_program
        self._onnx_values = _create_value_mapping(onnx_program.model.graph)
        self._args: tuple[Any, ...] = ()
        self.verification_infos: list[VerificationInfo] = []

    def run(
        self,
        *args: Any,
        initial_env: dict[torch.fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        """Run the interpreter with the given input arguments.

        This method executes the model and populates the :attr:`verification_infos` attribute
        with the verification information for each value.

        Args:
            args: The input arguments for the model.
            initial_env: The initial environment for the interpreter.
            enable_io_processing: Whether to enable IO processing.

        Returns:
            Any: The result of executing the model.
        """
        self.verification_infos = []
        self._args = args
        return super().run(
            *args,
            initial_env=initial_env,
            enable_io_processing=enable_io_processing,
        )

    def run_node(self, n: torch.fx.Node) -> Any:
        result = super().run_node(n)
        if n.op != "call_function":
            return result
        node_name = n.name
        if node_name not in self._onnx_values:
            return result
        try:
            (onnx_result,) = self._onnx_program.compute_values([node_name], self._args)
        except Exception:
            logger.warning(
                "Failed to compute value for node %s", node_name, exc_info=True
            )
            return result
        info = VerificationInfo.from_tensors(
            name=node_name,
            expected=result,
            actual=onnx_result,
        )
        self.verification_infos.append(info)
        if info.max_abs_diff > 0.01 or info.max_rel_diff > 0.1:
            logger.warning(
                "Verification info for node %s: max_abs_diff: %s, max_rel_diff: %s",
                node_name,
                info.max_abs_diff,
                info.max_rel_diff,
            )
        else:
            logger.info(
                "Verification info for node %s: max_abs_diff: %s, max_rel_diff: %s",
                node_name,
                info.max_abs_diff,
                info.max_rel_diff,
            )
        return result
<|endoftext|>"""Test utilities for ONNX export."""

from __future__ import annotations


__all__ = ["assert_onnx_program"]

from typing import Any, Literal, TYPE_CHECKING

import torch
from torch.utils import _pytree


if TYPE_CHECKING:
    from torch.onnx._internal.exporter import _onnx_program


def assert_onnx_program(
    program: _onnx_program.ONNXProgram,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    strategy: str | None = "TorchExportNonStrictStrategy",
    backend: Literal["onnxruntime", "reference"] = "onnxruntime",
) -> None:
    """Assert that the ONNX model produces the same output as the PyTorch ExportedProgram.

    Args:
        program: The ``ONNXProgram`` to verify.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        args: The positional arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
        kwargs: The keyword arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
        strategy: Assert the capture strategy used to export the program. Values can be
            class names like "TorchExportNonStrictStrategy".
            If None, the strategy is not asserted.
        backend: The backend to use for evaluating the ONNX program.
            Supported values are "onnxruntime" and "reference".
    """
    if strategy is not None:
        if program._capture_strategy != strategy:
            raise ValueError(
                f"Expected strategy '{strategy}' is used to capture the exported program, "
                f"but got '{program._capture_strategy}'."
            )
    exported_program = program.exported_program
    if exported_program is None:
        raise ValueError(
            "The ONNXProgram does not contain an ExportedProgram. "
            "To verify the ONNX program, initialize ONNXProgram with an ExportedProgram, "
            "or assign the ExportedProgram to the ONNXProgram.exported_program attribute."
        )
    if args is None and kwargs is None:
        # User did not provide example inputs, use the default example inputs
        if exported_program.example_inputs is None:
            raise ValueError(
                "No example inputs provided and the exported_program does not contain example inputs. "
                "Please provide arguments to verify the ONNX program."
            )
        args, kwargs = exported_program.example_inputs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    torch_module = exported_program.module()
    torch_outputs, _ = _pytree.tree_flatten(torch_module(*args, **kwargs))
    # ONNX outputs are always real, so we need to convert torch complex outputs to real representations
    torch_outputs_adapted = []
    for output in torch_outputs:
        # ONNX graph does not support None outputs, so we skip them
        if output is None:
            continue
        if not isinstance(output, torch.Tensor):
            torch_outputs_adapted.append(torch.tensor(output))
        elif torch.is_complex(output):
            torch_outputs_adapted.append(torch.view_as_real(output))
        else:
            torch_outputs_adapted.append(output)

    # Obtain the ONNX outputs using the specified backend
    if backend == "onnxruntime":
        onnx_outputs = program(*args, **kwargs)
    elif backend == "reference":
        onnx_outputs = program.call_reference(*args, **kwargs)
    else:
        raise ValueError(
            f"Unsupported backend '{backend}'. Supported backends are 'onnxruntime' and 'reference'."
        )

    # TODO(justinchuby): Include output names in the error message
    torch.testing.assert_close(
        tuple(onnx_outputs),
        tuple(torch_outputs_adapted),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        check_device=False,
    )
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

from torch.onnx._internal.exporter import _analysis, _registration, _verification


if TYPE_CHECKING:
    import os

    import onnx_ir as ir  # type: ignore[import-untyped]

    import torch


@dataclasses.dataclass
class ExportStatus:
    # Whether torch.export.export(..., strict=True) succeeds
    torch_export_strict: bool | None = None
    # Whether torch.export.export(..., strict=False) succeeds
    torch_export_non_strict: bool | None = None
    # Whether torch.export.draft_export() succeeds
    torch_export_draft_export: bool | None = None
    # Whether decomposition succeeds
    decomposition: bool | None = None
    # Whether ONNX translation succeeds
    onnx_translation: bool | None = None
    # Whether ONNX model passes onnx.checker.check_model
    onnx_checker: bool | None = None
    # Whether ONNX model runs successfully with ONNX Runtime
    onnx_runtime: bool | None = None
    # Whether the output of the ONNX model is accurate
    output_accuracy: bool | None = None


def _status_emoji(status: bool | None) -> str:
    if status is None:
        return "⚪"
    return "✅" if status else "❌"


def _format_export_status(status: ExportStatus) -> str:
    return (
        f"```\n"
        f"{_status_emoji(status.torch_export_non_strict)} Obtain model graph with `torch.export.export(..., strict=False)`\n"
        f"{_status_emoji(status.torch_export_strict)} Obtain model graph with `torch.export.export(..., strict=True)`\n"
        f"{_status_emoji(status.torch_export_draft_export)} Obtain model graph with `torch.export.draft_export`\n"
        f"{_status_emoji(status.decomposition)} Decompose operators for ONNX compatibility\n"
        f"{_status_emoji(status.onnx_translation)} Translate the graph into ONNX\n"
        f"{_status_emoji(status.onnx_checker)} Run `onnx.checker` on the ONNX model\n"
        f"{_status_emoji(status.onnx_runtime)} Execute the model with ONNX Runtime\n"
        f"{_status_emoji(status.output_accuracy)} Validate model output accuracy\n"
        f"```\n\n"
    )


def _strip_color_from_string(text: str) -> str:
    # This regular expression matches ANSI escape codes
    # https://github.com/pytorch/pytorch/blob/9554a9af8788c57e1c5222c39076a5afcf0998ae/torch/_dynamo/utils.py#L2785-L2788
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def _format_exported_program(exported_program: torch.export.ExportedProgram) -> str:
    # Adapted from https://github.com/pytorch/pytorch/pull/128476
    # to remove colors
    # Even though we can call graph_module.print_readable directly, since the
    # colored option was added only recently, we can't guarantee that the
    # version of PyTorch used by the user has this option. Therefore, we
    # still call str(ExportedProgram)
    text = f"```python\n{_strip_color_from_string(str(exported_program))}\n```\n\n"
    return text


def construct_report_file_name(timestamp: str, status: ExportStatus) -> str:
    # Status could be None. So we need to check for False explicitly.
    if not (
        status.torch_export_non_strict
        or status.torch_export_strict
        or status.torch_export_draft_export
    ):
        # All strategies failed
        postfix = "pt_export"
    elif status.decomposition is False:
        postfix = "decomp"
    elif status.onnx_translation is False:
        postfix = "conversion"
    elif status.onnx_checker is False:
        postfix = "checker"
    elif status.onnx_runtime is False:
        postfix = "runtime"
    elif status.output_accuracy is False:
        postfix = "accuracy"
    elif (
        status.torch_export_strict is False
        or status.torch_export_non_strict is False
        or status.torch_export_draft_export is False
    ):
        # Some strategies failed
        postfix = "strategies"
    else:
        postfix = "success"
    return f"onnx_export_{timestamp}_{postfix}.md"


def format_decomp_comparison(
    pre_decomp_unique_ops: set[str],
    post_decomp_unique_ops: set[str],
) -> str:
    """Format the decomposition comparison result.

    Args:
        pre_decomp_unique_ops: The unique ops in the first program.
        post_decomp_unique_ops: The unique ops in the second program.

    Returns:
        The formatted comparison result.
    """
    return (
        f"Ops exist only in the ExportedProgram before decomposition: `{sorted(pre_decomp_unique_ops)}`\n\n"
        f"Ops exist only in the ExportedProgram after decomposition: `{sorted(post_decomp_unique_ops)}`\n"
    )


def format_verification_infos(
    verification_infos: list[_verification.VerificationInfo],
) -> str:
    """Format the verification result.

    Args:
        verification_infos: The verification result.

    Returns:
        The formatted verification result.
    """
    return "\n".join(
        f"`{info.name}`: `max_abs_diff={info.max_abs_diff:e}`, `max_rel_diff={info.max_rel_diff:e}`, "
        f"`abs_diff_hist={info.abs_diff_hist}`, `rel_diff_hist={info.rel_diff_hist}`"
        for info in verification_infos
    )


def create_torch_export_error_report(
    filename: str | os.PathLike,
    formatted_traceback: str,
    *,
    export_status: ExportStatus,
    profile_result: str | None,
) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Error Report\n\n")
        f.write(_format_export_status(export_status))
        f.write("Error message:\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("```\n\n")
        if profile_result is not None:
            f.write("## Profiling result\n\n")
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")


def create_onnx_export_report(
    filename: str | os.PathLike,
    formatted_traceback: str,
    program: torch.export.ExportedProgram,
    *,
    decomp_comparison: str | None = None,
    export_status: ExportStatus,
    profile_result: str | None,
    model: ir.Model | None = None,
    registry: _registration.ONNXRegistry | None = None,
    verification_result: str | None = None,
) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Report\n\n")
        f.write(_format_export_status(export_status))
        f.write("## Error messages\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("\n```\n\n")
        f.write("## Exported program\n\n")
        f.write(_format_exported_program(program))
        if model is not None:
            f.write("## ONNX model\n\n")
            f.write("```python\n")
            f.write(str(model))
            f.write("\n```\n\n")
        f.write("## Analysis\n\n")
        _analysis.analyze(program, file=f, registry=registry)
        if decomp_comparison is not None:
            f.write("\n## Decomposition comparison\n\n")
            f.write(decomp_comparison)
            f.write("\n")
        if verification_result is not None:
            f.write("\n## Verification results\n\n")
            f.write(verification_result)
            f.write("\n")
        if profile_result is not None:
            f.write("\n## Profiling result\n\n")
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")
<|endoftext|>"""Typings for function definitions."""

from __future__ import annotations

from typing import TypeVar, Union

from onnxscript import (
    BFLOAT16,
    BOOL,
    COMPLEX128,
    COMPLEX64,
    DOUBLE,
    FLOAT,
    FLOAT16,
    INT16,
    INT32,
    INT64,
    INT8,
    STRING,
    UINT8,
)


# NOTE: We do not care about unsigned types beyond UINT8 because PyTorch does not use them.
# More detail can be found: https://pytorch.org/docs/stable/tensors.html

TensorType = Union[  # noqa: UP007
    BFLOAT16,
    BOOL,
    COMPLEX64,
    COMPLEX128,
    DOUBLE,
    FLOAT,
    FLOAT16,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
]
_FloatType = Union[FLOAT16, FLOAT, DOUBLE, BFLOAT16]  # noqa: UP007
IntType = Union[INT8, INT16, INT32, INT64]  # noqa: UP007
RealType = Union[  # noqa: UP007
    BFLOAT16,
    FLOAT16,
    FLOAT,
    DOUBLE,
    INT8,
    INT16,
    INT32,
    INT64,
]

TTensor = TypeVar("TTensor", bound=TensorType)
# Duplicate TTensor for inputs/outputs that accept the same set of types as TTensor
# but do not constrain the type to be the same as the other inputs/outputs
TTensor2 = TypeVar("TTensor2", bound=TensorType)
TTensorOrString = TypeVar("TTensorOrString", bound=Union[TensorType, STRING])  # noqa: UP007
TFloat = TypeVar("TFloat", bound=_FloatType)
TFloatOrUInt8 = TypeVar(
    "TFloatOrUInt8",
    bound=Union[FLOAT, FLOAT16, DOUBLE, INT8, UINT8],  # noqa: UP007
)
TInt = TypeVar("TInt", bound=IntType)
TReal = TypeVar("TReal", bound=RealType)
TRealUnlessInt16OrInt8 = TypeVar(
    "TRealUnlessInt16OrInt8",
    bound=Union[FLOAT16, FLOAT, DOUBLE, BFLOAT16, INT32, INT64],  # noqa: UP007
)
TRealUnlessFloat16OrInt8 = TypeVar(
    "TRealUnlessFloat16OrInt8",
    bound=Union[DOUBLE, FLOAT, INT16, INT32, INT64],  # noqa: UP007
)
TRealOrUInt8 = TypeVar("TRealOrUInt8", bound=Union[RealType, UINT8])  # noqa: UP007
TFloatHighPrecision = TypeVar("TFloatHighPrecision", bound=Union[FLOAT, DOUBLE])  # noqa: UP007
<|endoftext|>"""Registry for aten functions."""

from __future__ import annotations


__all__ = ["onnx_impl", "get_torchlib_ops"]

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import onnxscript

import torch
from torch.onnx._internal.exporter import _constants, _registration


# Use ParamSpec for better type preservation instead of bound Callable TypeVar
_P = ParamSpec("_P")
_R = TypeVar("_R")

logger = logging.getLogger("__name__")


_registry: list[_registration.OnnxDecompMeta] = []


def onnx_impl(
    target: _registration.TorchOp | tuple[_registration.TorchOp, ...],
    *,
    trace_only: bool = False,
    complex: bool = False,
    opset_introduced: int = 18,
    no_compile: bool = False,
    private: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Register an ONNX implementation of a torch op."""

    if isinstance(target, torch._ops.OpOverloadPacket):
        raise TypeError(
            f"Target '{target}' should be provided as an OpOverload instead of an "
            "OpOverloadPacket. You can get the default overload with "
            "<op>.default"
        )

    def wrapper(
        func: Callable[_P, _R],
    ) -> Callable[_P, _R]:
        processed_func: Any
        if no_compile:
            processed_func = func
        else:
            torchlib_opset = onnxscript.values.Opset(
                domain=_constants.TORCHLIB_DOMAIN, version=1
            )

            if not trace_only:
                # Compile the function
                processed_func = onnxscript.script(opset=torchlib_opset)(func)
            else:
                processed_func = onnxscript.TracedOnnxFunction(torchlib_opset, func)

        if not private:
            # TODO(justinchuby): Simplify the logic and remove the private attribute
            # Skip registration if private
            if not isinstance(target, Sequence):
                targets = (target,)
            else:
                targets = target  # type: ignore[assignment]

            for t in targets:
                _registry.append(
                    _registration.OnnxDecompMeta(
                        onnx_function=processed_func,
                        fx_target=t,
                        signature=None,
                        is_complex=complex,
                        opset_introduced=opset_introduced,
                        skip_signature_inference=no_compile,
                    )
                )
        return processed_func  # type: ignore[return-value]

    return wrapper


def get_torchlib_ops() -> tuple[_registration.OnnxDecompMeta, ...]:
    # Trigger op registration
    from torch.onnx._internal.exporter._torchlib import ops

    del ops
    if len(_registry) == 0:
        raise AssertionError("_registry must not be empty")
    return tuple(_registry)
<|endoftext|>"""Implementation for torch.sym* ops."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TC003

from __future__ import annotations

from collections.abc import Sequence

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import (
    BOOL,
    FLOAT,
    IntType,
    TensorType,
    TTensor,
)
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


@onnx_impl(torch.sym_float, trace_only=True)
def sym_float(self: TensorType) -> FLOAT:
    """sym_float(SymInt self) -> SymFloat"""
    return op.Cast(self, to=FLOAT.dtype)


@onnx_impl(torch.sym_max, trace_only=True)
def sym_max(x: IntType, y: IntType) -> IntType:
    """sym_max(SymInt x, SymInt y) -> SymInt"""
    return op.Max(x, y)


@onnx_impl(torch.sym_min, trace_only=True)
def sym_min(x: IntType, y: IntType) -> IntType:
    """sym_min(SymInt x, SymInt y) -> SymInt"""
    return op.Min(x, y)


@onnx_impl(torch.sym_not, trace_only=True)
def sym_not(self: BOOL) -> BOOL:
    """sym_not(SymBool self) -> SymBool"""
    return op.Not(self)


@onnx_impl(torch.sym_sum, trace_only=True)
def sym_sum(args: Sequence[IntType]) -> IntType:
    """sym_sum(SymInt[] args) -> SymInt"""
    if len(args) == 0:
        return op.Constant(value_int=0)
    if len(args) == 1:
        return args[0]
    result = op.Add(args[0], args[1])
    for i in range(2, len(args)):
        result = op.Add(result, args[i])
    return result


@onnx_impl(torch.sym_ite, trace_only=True)
def sym_ite(b: BOOL, t: TTensor, f: TTensor) -> TTensor:
    """sym_ite(SymBool b, Tensor t, Tensor f) -> Tensor"""
    return op.Where(b, t, f)
<|endoftext|>"""Implementation for higher-order operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter import _core
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from collections.abc import Sequence


def call_op(
    op_type: str,
    *args: ir.Value | None,
    _num_outputs: int = 1,
    _domain: str = "",
    **kwargs: int | float | str | bool | ir.Graph | ir.TensorProtocol | Sequence[int],
) -> Sequence[ir.Value]:
    """Call an operator with the given arguments and keyword arguments.

    Arguments are always inputs, while keyword arguments are attributes.
    """
    # This is a wrapper around the IR node creation that hooks into the _builder.OpRecorder
    # tracer so that all nodes created are recorded the same way as if we were to use
    # onnxscript ops directly.
    from onnxscript.ir import convenience as ir_convenience

    if _core.current_tracer is None:
        raise AssertionError("current_tracer must be non-None")
    tracer = _core.current_tracer

    inputs = list(args)

    # If final inputs are None, strip them from the node inputs
    for input in reversed(inputs):
        if input is not None:
            break
        inputs.pop()

    # Construct and filter out None attributes
    attributes = [
        attr
        for attr in ir_convenience.convert_attributes(kwargs)
        if attr.value is not None  # type: ignore[union-attr]
    ]
    tracer.nodes.append(
        node := ir.Node(
            _domain,
            op_type,
            inputs=inputs,
            attributes=attributes,
            num_outputs=_num_outputs,
            version=tracer.opset.version,
        )
    )
    return node.outputs


@onnx_impl(torch.ops.higher_order.cond, no_compile=True)
def higher_order_cond(
    cond: ir.Value,
    true_func: ir.Function,
    false_func: ir.Function,
    inputs: Sequence[ir.Value],
) -> Sequence[ir.Value]:
    then_node = ir.Node(
        true_func.domain, true_func.name, inputs, num_outputs=len(true_func.outputs)
    )
    else_node = ir.Node(
        false_func.domain, false_func.name, inputs, num_outputs=len(false_func.outputs)
    )

    # ONNX Runtime complains about duplicate output names if we don't rename them.
    # But the doesn't seem to be an actual violation of SSA form without renaming.
    for func_out, out in zip(true_func.outputs, then_node.outputs):
        out.name = f"{func_out.name}_{true_func.name}"
    for func_out, out in zip(false_func.outputs, else_node.outputs):
        out.name = f"{func_out.name}_{false_func.name}"

    return call_op(
        "If",
        cond,
        _num_outputs=len(true_func.outputs),
        then_branch=ir.Graph(
            (), then_node.outputs, nodes=[then_node], name=true_func.name
        ),
        else_branch=ir.Graph(
            (), else_node.outputs, nodes=[else_node], name=false_func.name
        ),
    )


@onnx_impl(torch.ops.higher_order.scan, no_compile=True)
def higher_order_scan(
    body_func: ir.Function,
    scan_inits: Sequence[ir.Value],
    scan_inputs: Sequence[ir.Value],
    additional_inputs: Sequence[ir.Value] | None,
    reverse: bool = False,
) -> Sequence[ir.Value]:
    """https://github.com/pytorch/pytorch/blob/66ac724b56e6c37a534f3e066423ef2f41d7477f/torch/_higher_order_ops/scan.py#L109"""
    subgraph_inputs = [
        *[
            ir.Value(
                name=f"{inp.name}_{body_func.name}__subgraph_in",
                shape=inp.shape,
                type=ir.TensorType(inp.dtype),  # type: ignore[arg-type]
            )
            for inp in scan_inits
        ],
        *[
            ir.Value(
                name=f"{inp.name}_{body_func.name}__subgraph_in",
                # The iterated element passed to the body subgraph does not have a sequence axis.
                # It will have a rank one less than the rank of the corresponding scan_input.
                shape=ir.Shape(inp.shape[1:]),  # type: ignore[index]
                type=ir.TensorType(inp.dtype),  # type: ignore[arg-type]
            )
            for inp in scan_inputs
        ],
    ]
    # The one and only node in the Scan subgraph that calls the body_func
    body_node = ir.Node(
        body_func.domain,
        body_func.name,
        [
            *subgraph_inputs,
            *(additional_inputs or []),
        ],
        num_outputs=len(body_func.outputs),
    )

    # ONNX Runtime complains about duplicate output names if we don't rename them.
    # But the doesn't seem to be an actual violation of SSA form without renaming.
    for func_out, out in zip(body_func.outputs, body_node.outputs):
        out.name = f"{func_out.name}_{body_func.name}"

    n_outputs = len(body_func.outputs) - len(scan_inits)
    return call_op(
        "Scan",
        *scan_inits,
        *scan_inputs,
        _num_outputs=len(body_func.outputs),
        body=ir.Graph(
            subgraph_inputs,
            body_node.outputs,
            nodes=[body_node],
            name=body_func.name,
        ),
        num_scan_inputs=len(scan_inputs),
        scan_input_directions=[(1 if reverse else 0) for _ in scan_inputs],
        scan_output_directions=[(1 if reverse else 0) for _ in range(n_outputs)],
    )


@onnx_impl(torch.ops.higher_order.while_loop, no_compile=True)
def higher_order_while_loop(
    cond_func: ir.Function,
    body_func: ir.Function,
    carried_inputs: Sequence[ir.Value | int | float],
    additional_inputs: Sequence[ir.Value],
) -> Sequence[ir.Value]:
    """Implementation of while_loop using ONNX Loop operator.

    The ONNX Loop operator implements a generic looping construct with the signature:
    Loop(M, cond, v_initial) -> (v_final_and_scan_outputs)

    For while_loop, we use:
    - M: None (no trip count limit)
    - cond: initial condition value
    - v_initial: carried_inputs (loop-carried dependencies)

    The body subgraph takes:
    - iteration_num (int): current iteration number
    - condition_in (bool): loop continuation condition from previous iteration
    - loop_carried_dependencies: the carried values
    - additional_inputs: any additional inputs (constants/parameters)

    The body subgraph returns:
    - condition_out (bool): whether to continue looping
    - loop_carried_dependencies: updated carried values
    """

    # Create subgraph inputs for the Loop body
    # ONNX Loop body signature: (iter_num, cond_in, loop_carried_deps..., additional_inputs...)

    # Start subgraph construction
    subgraph_carried_inputs = []

    for i, inp in enumerate(carried_inputs):
        if isinstance(inp, ir.Value):
            subgraph_carried_inputs.append(
                ir.Value(
                    name=f"{inp.name}_{body_func.name}__subgraph_in",
                    shape=inp.shape,
                    type=ir.TensorType(inp.dtype),  # type: ignore[arg-type]
                )
            )
        elif isinstance(inp, int):
            subgraph_carried_inputs.append(
                ir.Value(
                    name=f"carried_input_{i}_{body_func.name}__subgraph_in",
                    shape=ir.Shape([]),
                    type=ir.TensorType(ir.DataType.INT64),
                )
            )
        elif isinstance(inp, float):
            subgraph_carried_inputs.append(
                ir.Value(
                    name=f"carried_input_{i}_{body_func.name}__subgraph_in",
                    shape=ir.Shape([]),
                    type=ir.TensorType(ir.DataType.FLOAT),
                )
            )
        else:
            raise NotImplementedError(
                f"Unsupported type for carried input: {type(inp)} ({inp}). "
                "Expected ir.Value, int, or float."
            )

    subgraph_inputs = [
        # Iteration number (int scalar, unused)
        ir.Value(
            name=f"iter_num_{body_func.name}",
            shape=ir.Shape([]),
            type=ir.TensorType(ir.DataType.INT64),
        ),
        # Condition input (bool scalar, unused)
        ir.Value(
            name=f"cond_in_{body_func.name}",
            shape=ir.Shape([]),
            type=ir.TensorType(ir.DataType.BOOL),
        ),
        # Loop-carried dependencies
        *subgraph_carried_inputs,
    ]

    # Create the combined body function that handles both condition and body logic
    # First, call the body function with the same inputs
    body_node = ir.Node(
        body_func.domain,
        body_func.name,
        [
            *subgraph_carried_inputs,  # carried inputs
            *additional_inputs,
        ],
        num_outputs=len(body_func.outputs),  # carried inputs
    )

    # Then call the condition function with carried inputs + additional inputs
    cond_node = ir.Node(
        cond_func.domain,
        cond_func.name,
        [
            *body_node.outputs,  # updated carried inputs from body
            *additional_inputs,
        ],
        num_outputs=len(cond_func.outputs),
    )

    if len(cond_func.outputs) != 1:
        raise AssertionError("Condition function must return a single boolean value.")

    # ONNX Runtime complains about duplicate output names if we don't rename them
    for func_out, out in zip(body_func.outputs, body_node.outputs):
        out.name = f"{func_out.name}_{body_func.name}"
    for func_out, out in zip(cond_func.outputs, cond_node.outputs):
        out.name = f"{func_out.name}_{cond_func.name}"

    # The Loop body must return: (cond_out, loop_carried_deps...)
    # We use the condition output and the body outputs
    loop_body_outputs = [
        cond_node.outputs[0],  # condition output (bool)
        *body_node.outputs,  # updated carried inputs
    ]

    body_graph = ir.Graph(
        subgraph_inputs,
        loop_body_outputs,
        nodes=[body_node, cond_node],
        name=f"{body_func.name}_loop_body",
    )

    # End subgraph construction

    carried_inputs_values: list[ir.Value] = []
    for inp in carried_inputs:
        if isinstance(inp, ir.Value):
            carried_inputs_values.append(inp)
        elif isinstance(inp, int):
            const = call_op("Constant", value=ir.tensor(inp))[0]
            carried_inputs_values.append(const)
        elif isinstance(inp, float):
            const = call_op("Constant", value=ir.tensor(inp))[0]
            carried_inputs_values.append(const)
        else:
            raise NotImplementedError(
                f"Unsupported type for carried input: {type(inp)} ({inp}). "
                "Expected ir.Value, int, or float."
            )

    # Get initial condition by calling cond_func with initial inputs
    initial_outputs = call_op(
        cond_func.name,
        *carried_inputs_values,
        *additional_inputs,
        _num_outputs=len(cond_func.outputs),
        _domain=cond_func.domain,
    )

    if len(initial_outputs) != 1:
        raise AssertionError("Condition function must return a single boolean value.")

    # Create the Loop operator call
    # Loop(M, cond, v_initial) where M is empty (no trip count limit)
    loop_outputs = call_op(
        "Loop",
        # M (trip count) - empty string means no limit
        None,
        # cond - initial condition
        initial_outputs[0],
        # v_initial - carried inputs (loop-carried dependencies)
        *carried_inputs_values,
        _num_outputs=len(carried_inputs_values),
        body=body_graph,
    )

    return loop_outputs


@onnx_impl(torch.ops.higher_order.invoke_subgraph, no_compile=True)
def higher_order_invoke_subgraph(
    subgraph: ir.Function,
    identifier: str | None,
    *operands: ir.Value,
) -> Sequence[ir.Value]:
    """Export invoke_subgraph HOP by creating a direct function call.

    This preserves the function as a separate entity in the ONNX graph
    instead of inlining it, which is the purpose of invoke_subgraph.

    Note: The onnxscript optimizer should be configured to not inline functions
    created by invoke_subgraph to preserve the intended structure.

    Args:
        subgraph: The function to invoke
        identifier: Optional identifier for the subgraph (used for caching in PyTorch,
            not needed for ONNX export as the function reference provides all necessary information)
        *operands: Input values to pass to the function

    Returns:
        Sequence of output values from the function call
    """
    # This key can be used by downstream to avoid inlining
    subgraph.metadata_props["pkg.torch.ops.higher_order.invoke_subgraph.identifier"] = (
        str(identifier)
    )

    # Create the function call node
    return call_op(
        subgraph.name,
        *operands,
        _num_outputs=len(subgraph.outputs),
        _domain=subgraph.domain,
    )
<|endoftext|>"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TCH001

from __future__ import annotations

import operator

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import TReal, TRealOrUInt8
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten


@onnx_impl((aten.abs.default, operator.abs), trace_only=True)
def aten_abs(self: TRealOrUInt8) -> TRealOrUInt8:
    """abs(Tensor self) -> Tensor"""

    return op.Abs(self)


@onnx_impl(aten.abs.default, complex=True, trace_only=True)
def aten_abs_complex(self: TRealOrUInt8) -> TRealOrUInt8:
    """abs(Tensor self) -> Tensor"""

    return op.ReduceL2(self, [-1], keepdims=False)


@onnx_impl((aten.add.Tensor, aten.add.Scalar, operator.add), trace_only=True)
def aten_add(self: TReal, other: TReal, alpha: float = 1.0) -> TReal:
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    if alpha != 1.0:
        alpha = op.CastLike(alpha, other)
        other = op.Mul(other, alpha)
    return op.Add(self, other)


@onnx_impl((aten.add.Tensor, aten.add.Scalar), trace_only=True, complex=True)
def aten_add_complex(self: TReal, other: TReal, alpha: float = 1.0) -> TReal:
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""

    return aten_add(self, other, alpha=alpha)
<|endoftext|>"""Implementation for higher-order operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript.ir import convenience as ir_convenience

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter import _core
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx.ops import _symbolic_impl


if TYPE_CHECKING:
    from collections.abc import Sequence


def _call_symbolic_op(
    op_type: str,
    domain: str,
    args: Sequence[ir.Value | None],
    kwargs: dict[str, int | float | str | bool | list[int] | list[float] | list[str]],
    dtypes: Sequence[int],
    version: int | None,
    metadata_props: dict[str, str] | None,
) -> Sequence[ir.Value]:
    """Call an operator with the given arguments and keyword arguments.

    Arguments are always inputs, while keyword arguments are attributes.
    """
    # This is a wrapper around the IR node creation that hooks into the _builder.OpRecorder
    # tracer so that all nodes created are recorded the same way as if we were to use
    # onnxscript ops directly.

    if _core.current_tracer is None:
        raise AssertionError("current_tracer must be non-None")
    tracer = _core.current_tracer

    inputs = list(args)

    # If final inputs are None, strip them from the node inputs
    for input in reversed(inputs):
        if input is not None:
            break
        inputs.pop()

    # Construct and filter out None attributes
    attributes = [
        attr
        for attr in ir_convenience.convert_attributes(kwargs)  # type: ignore[arg-type]
        if attr.value is not None  # type: ignore[union-attr]
    ]
    tracer.nodes.append(
        node := ir.Node(
            domain,
            op_type,
            inputs=inputs,
            attributes=attributes,
            num_outputs=len(dtypes),
            version=version,
            metadata_props=metadata_props,
        )
    )
    # Set the dtypes for the outputs. We set them here because the graph builder
    # Uses PyTorch types which are sometimes inaccurate when they are ONNX only
    # types like float4e2m1.
    for value, dtype in zip(node.outputs, dtypes):
        value.dtype = ir.DataType(dtype)
        # The shape is set by the graph builder. We don't need to set it here.
    return node.outputs


@onnx_impl(torch.ops.onnx_symbolic._symbolic.default, no_compile=True)
def onnx_symbolic_symbolic(
    inputs: Sequence[ir.Value | None],
    op_type: str,
    onnx_dtype: int,
    *,
    shape: Sequence[int | ir.Value],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: int | None = None,
) -> ir.Value:
    del shape  # Unused. The shapes are set by the graph builder
    encoded = _symbolic_impl.EncodedAttrs(
        attr_keys=list(attr_keys),
        attr_types=list(attr_types),
        attr_pos=list(attr_pos),
        attr_ints=list(attr_ints),
        attr_floats=list(attr_floats),
        attr_strs=list(attr_strs),
    )
    attrs = encoded.to_dict()
    return _call_symbolic_op(
        op_type,
        domain,
        inputs,
        attrs,
        dtypes=[onnx_dtype],
        version=version,
        metadata_props=dict(zip(metadata_props_keys, metadata_props_values)),
    )[0]


@onnx_impl(torch.ops.onnx_symbolic._symbolic_multi_out.default, no_compile=True)
def onnx_symbolic_symbolic_multi_out(
    inputs: Sequence[ir.Value | None],
    op_type: str,
    onnx_dtypes: Sequence[int],
    *,
    shapes: Sequence[Sequence[int | ir.Value]],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: int | None = None,
) -> Sequence[ir.Value]:
    del shapes  # Unused. The shapes are set by the graph builder
    encoded = _symbolic_impl.EncodedAttrs(
        attr_keys=list(attr_keys),
        attr_types=list(attr_types),
        attr_pos=list(attr_pos),
        attr_ints=list(attr_ints),
        attr_floats=list(attr_floats),
        attr_strs=list(attr_strs),
    )
    attrs = encoded.to_dict()
    return _call_symbolic_op(
        op_type,
        domain,
        inputs,
        attrs,
        dtypes=onnx_dtypes,
        version=version,
        metadata_props=dict(zip(metadata_props_keys, metadata_props_values)),
    )
<|endoftext|>"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TC001

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING  # noqa: UP035

from onnxscript.onnx_opset import (  # type: ignore[attr-defined]
    opset20 as op20,
    opset21 as op21,
    opset23 as op23,
)

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter._torchlib._tensor_typing import TFloat, TReal
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from onnxscript.values import Opset

aten = torch.ops.aten


@onnx_impl(aten.gelu.default, trace_only=True, opset_introduced=20)
def aten_gelu_opset20(
    self: TReal,
    approximate: str = "none",
) -> TReal:
    """gelu(Tensor self, *, str approximate="none") -> Tensor"""
    return op20.Gelu(self, approximate=approximate)


@onnx_impl(aten.group_norm.default, trace_only=True, opset_introduced=21)
def aten_group_norm(
    input: TFloat,
    num_groups: int,
    weight: TFloat | None = None,
    bias: TFloat | None = None,
    eps: float = 1e-05,
    cudnn_enabled: bool = True,
) -> TFloat:
    """group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor"""

    c = op21.Shape(input, start=1, end=2)
    if weight is None:
        weight = op21.ConstantOfShape(c, value=ir.tensor([1.0], dtype=input.dtype))
    if bias is None:
        bias = op21.ConstantOfShape(c, value=ir.tensor([0.0], dtype=input.dtype))
    return op21.GroupNormalization(
        input, weight, bias, epsilon=eps, num_groups=num_groups
    )


@onnx_impl(aten.rms_norm.default, trace_only=True, opset_introduced=23)
def aten_rms_norm(
    input: TFloat,
    normalized_shape: Sequence[int],
    weight: TFloat | None = None,
    eps: float | None = None,
) -> TFloat:
    """rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor"""

    # Default eps value if not provided
    if eps is None:
        eps = torch.finfo(torch.float).eps  # Observed from decomp

    # Calculate axis: the first normalization dimension
    # For normalized_shape with D dimensions, normalize over last D dimensions
    # Since ONNX RMSNormalization supports negative axis values, we use -len(normalized_shape)
    # which correctly maps to the first axis of the normalized dimensions
    normalized_dims = len(normalized_shape)
    axis = -normalized_dims

    # Create weight tensor if not provided
    if weight is None:
        weight = op23.ConstantOfShape(
            op23.Shape(input), value=ir.tensor([1], dtype=input.dtype)
        )

    return op23.RMSNormalization(input, weight, axis=axis, epsilon=eps)


@onnx_impl(
    aten.scaled_dot_product_attention.default, trace_only=True, opset_introduced=23
)
def aten_scaled_dot_product_attention_23(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    attn_mask: TFloat | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> TFloat:
    """scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None, bool enable_gqa=False) -> Tensor

    Reference:
        1. https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        2. https://onnx.ai/onnx/operators/onnx__Attention.html

    Attempts to convert SDPA to Attention onnx op and fallbacks to an onnx graph equivalent to the following PyTorch code::
        scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
        attn_mask = (
            torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            if is_causal
            else attn_mask
        )
        attn_mask = (
            attn_mask.masked_fill(not attn_mask, -float("inf"))
            if attn_mask.dtype == torch.bool
            else attn_mask
        )
        attn_weight = torch.softmax(
            (Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1
        )
        attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ V

    where Q, K, V are the query, key, and value tensors, respectively.
    L is the target sequence length, S is the source sequence length, and E is the embedding size.
    """
    if is_causal and attn_mask is not None:
        raise AssertionError("is_causal and attn_mask cannot be set at the same time")
    if not (len(query.shape) == 4 and len(key.shape) == 4 and len(value.shape) == 4):
        raise AssertionError("only 4D query, key, and value are supported")

    # Attention onnx op can only handle non-training scenarios where dropout is disabled.
    if dropout_p == 0:
        if enable_gqa:
            if not (
                query.shape[1] > key.shape[1] == value.shape[1]
                and query.shape[1] % key.shape[1] == 0
            ):
                raise AssertionError(
                    "SDPA (GQA or MQA) requires q_num_heads > kv_num_heads & "
                    "q_num_heads % kv_num_heads == 0"
                )
        else:
            if not (query.shape[1] == key.shape[1] == value.shape[1]):
                raise AssertionError("SDPA (MHA) requires q_num_heads = kv_num_heads")

        # NOTE: num_heads attributes (q_num_heads/kv_num_heads) should not be specified for 4D.
        # They are not populated with 4D inputs because this information directly comes from input shapes:
        # `q_num_heads=query.shape[1]` and `kv_num_heads=key.shape[1]`.
        # This dimension is usually static but it could not be dynamic if also given as an attribute.
        # num_heads attributes are needed for 3D attention inputs:
        # (shape: [B, S, N*H]), 4D shape is ([B, N, S, H]).

        Y, _, _, _ = op23.Attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            scale=scale,
            is_causal=is_causal,
        )
        return Y

    if scale is None:
        scale = _attention_scale(query, op23)
    scale = op23.CastLike(scale, query)

    if is_causal:
        attn_mask = _causal_attention_mask(query, key, op23)

    if enable_gqa:
        key, value = _attention_repeat_kv_for_group_query(query, key, value, op23)

    if attn_mask is None:
        return _aten_scaled_dot_product_attention_no_mask_onnx(
            query, key, value, scale, dropout_p, op23
        )

    return _aten_scaled_dot_product_attention_float_mask_onnx(
        query, key, value, attn_mask, scale, dropout_p, op23
    )


def _attention_repeat_kv_for_group_query(
    query: TFloat, key: TFloat, value: TFloat, op: Opset
) -> tuple[TFloat, TFloat]:
    """Expand key and value for group query attention.

    repeat_interleave is applied on key and value to match the number of heads in query.

    Args:
        query: Tensor of shape [B, q_num_heads, q_S, E]
        key: Tensor of shape [B, k_num_heads, kv_S, E]
        value: Tensor of shape [B, v_num_heads, kv_S, E]

    Returns:
        Tuple of (expanded_key, expanded_value) where:
            - expanded_key: Tensor of shape [B, q_num_heads, kv_S, E]
            - expanded_value: Tensor of shape [B, q_num_heads, kv_S, E]
    """

    if not (
        query.shape[1] > key.shape[1] == value.shape[1]
        and query.shape[1] % key.shape[1] == 0
    ):
        raise AssertionError(
            "SDPA (GQA or MQA) requires q_num_heads > kv_num_heads & "
            "q_num_heads % kv_num_heads == 0"
        )

    # NOTE: QKV are expected to be 4D tensors

    batch_size = op.Shape(query, start=0, end=1)  # [B]
    q_num_heads = op.Shape(query, start=1, end=2)  # [Hq]
    kv_num_heads = op.Shape(key, start=1, end=2)  # [Hk]
    qk_head_size = op.Shape(key, start=3, end=4)  # [Dk]
    v_head_size = op.Shape(value, start=3, end=4)  # [Dv]
    new_kv_seq_len = op.Shape(key, start=2, end=3)  # [T]

    interleave_dim = op.Div(q_num_heads, kv_num_heads)  # Hq / Hk
    two = op.Constant(value_int=2)
    k_unsqueezed = op.Unsqueeze(key, two)  # [B, Hk, 1, T, Dk]
    v_unsqueezed = op.Unsqueeze(value, two)  # [B, Hv, 1, T, Dv]

    k_expand_shape = op.Concat(
        batch_size, kv_num_heads, interleave_dim, new_kv_seq_len, qk_head_size, axis=0
    )
    k_expand = op.Expand(k_unsqueezed, k_expand_shape)
    v_expand_shape = op.Concat(
        batch_size, kv_num_heads, interleave_dim, new_kv_seq_len, v_head_size, axis=0
    )
    v_expand = op.Expand(v_unsqueezed, v_expand_shape)

    k_attention_shape = op.Concat(
        batch_size, q_num_heads, new_kv_seq_len, qk_head_size, axis=0
    )
    v_attention_shape = op.Concat(
        batch_size, q_num_heads, new_kv_seq_len, v_head_size, axis=0
    )

    expanded_key = op.Reshape(k_expand, k_attention_shape)
    expanded_value = op.Reshape(v_expand, v_attention_shape)

    return expanded_key, expanded_value


def _attention_scale(query: TFloat, op: Opset) -> TFloat:
    """Calculate the scale factor for the attention result.

    Args:
        query: Tensor of shape [..., L, E]

    Returns:
        Scalar scale factor := 1 / math.sqrt(query.size(-1))
    """
    q_shape = op.Shape(query)
    q_last_dim = op.Gather(q_shape, op.Constant(value_ints=[-1]))
    embedding_size = op.CastLike(q_last_dim, query)
    one = op.Constant(value_float=1.0)
    cast_one = op.CastLike(one, query)
    scale = op.Div(cast_one, op.Sqrt(embedding_size))
    return scale


def _causal_attention_mask(query: TFloat, key: TFloat, op: Opset) -> TFloat:
    """Create a causal mask for the given query and key tensors.

    Equivalent to::
        mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_mask = torch.zeros(L, S, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(not mask, -float("inf"))

    Args:
        query: Tensor of shape [..., L, E]
        key: Tensor of shape [..., S, E]

    Returns:
        Tensor of shape [L, S]
    """
    q_shape = op.Shape(query)
    k_shape = op.Shape(key)

    target_length = op.Slice(
        q_shape, op.Constant(value_ints=[-2]), op.Constant(value_ints=[-1])
    )
    source_length = op.Slice(
        k_shape, op.Constant(value_ints=[-2]), op.Constant(value_ints=[-1])
    )
    # attn_mask = torch.ones(L, S) := {
    size = op.Concat(target_length, source_length, axis=0)
    attn_mask = op.Expand(op.Constant(value_float=1.0), size)
    # }
    attn_mask = op.Trilu(attn_mask, upper=0)
    # The causal mask has 0s in the lower triangle and -inf in the upper triangle.
    attn_mask = op.Where(
        op.Equal(attn_mask, op.Constant(value_float=0.0)),
        op.Constant(value_float=-float("inf")),
        op.Constant(value_float=0.0),
    )
    attn_mask = op.CastLike(attn_mask, query)
    return attn_mask


def _aten_scaled_dot_product_attention_no_mask_onnx(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    scale: TFloat,
    dropout_p: float,
    op: Opset,
) -> TFloat:
    # Swap the last two axes of key
    key_last_dim = op.Shape(key, start=-1)
    key_second_last_dim = op.Shape(key, start=-2, end=-1)
    key_first_dims = op.Shape(key, end=-2)
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(
        key_first_dims, key_last_dim, key_second_last_dim, axis=0
    )
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(
        key_transposed, op.CastLike(op.Sqrt(scale), key_transposed)
    )
    attn_weight = op.Softmax(
        op.MatMul(query_scaled, key_transposed_scaled),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p)
    return op.MatMul(attn_weight, value)


def _aten_scaled_dot_product_attention_float_mask_onnx(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    attn_mask: TFloat,
    scale: TFloat,
    dropout_p: float,
    op: Opset,
) -> TFloat:
    # Swap the last two axes of key
    key_last_dim = op.Shape(key, start=-1)
    key_second_last_dim = op.Shape(key, start=-2, end=-1)
    key_first_dims = op.Shape(key, end=-2)
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(
        key_first_dims, key_last_dim, key_second_last_dim, axis=0
    )
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(key_transposed, op.Sqrt(scale))
    attn_weight = op.Softmax(
        op.Add(op.MatMul(query_scaled, key_transposed_scaled), attn_mask),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p)
    return op.MatMul(attn_weight, value)
<|endoftext|># mypy: allow-untyped-defs
import copy
import io
import math
import weakref
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, cast, NamedTuple, TYPE_CHECKING

import torch
import torch.cuda._pin_memory_utils as pin_memory_utils
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import AsyncCollectiveTensor


if dist.is_available() or TYPE_CHECKING:
    from torch.distributed import distributed_c10d
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed.tensor import distribute_tensor, DTensor, Replicate
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset


def _identity_func(
    obj: torch.Tensor,
    pg: dist.ProcessGroup | None,
    device: torch.device | None,
    companion_obj: Any,
) -> torch.Tensor:
    return obj


def _all_gather_sharded_tensor(
    sharded_tensor: "ShardedTensor",
    pg: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if pg is None:
        pg = distributed_c10d._get_default_group()
    world_size = dist.get_world_size(pg)
    shards = sharded_tensor.local_shards()
    dim_0_size = sharded_tensor.size()[0]  # type: ignore[index]
    tensor_numel = sharded_tensor.size().numel()  # type: ignore[union-attr]
    chunk_size = math.ceil(dim_0_size / world_size) * tensor_numel // dim_0_size
    pg_device = (
        distributed_c10d._get_pg_default_device(pg) if device is None else device
    )
    if shards:
        local_tensor = shards[0].tensor.flatten()
        if local_tensor.device.type != pg_device.type:
            local_tensor = local_tensor.to(pg_device)
        num_padding = chunk_size - local_tensor.numel()
        if num_padding > 0:
            local_tensor = F.pad(local_tensor, [0, num_padding])
    else:
        local_tensor = torch.zeros(
            chunk_size, dtype=sharded_tensor.dtype, device=pg_device
        )

    tensor = torch.empty(
        chunk_size * world_size,
        dtype=local_tensor.dtype,
        device=pg_device,
    )
    dist.all_gather_single(tensor, local_tensor, group=pg)

    tensor = tensor.narrow(0, 0, tensor_numel).reshape(sharded_tensor.size())
    return tensor


class CompanionMismatch(Exception):
    pass


def _iterate_state_dict(
    iter_object: Any,
    sharded_tensor_func: Callable,
    dtensor_func: Callable,
    tensor_func: Callable,
    *,
    pg: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
    cpu_offload: bool = False,
    companion_obj: Any = None,
    ranks_only: tuple[int, ...] = (),
    type_check: bool = True,
    non_blocking: bool = True,
) -> dict[str, Any]:
    """Iterate through the state dict, applying the given functions to each tensor type.

    Args:
        iter_object (Any): the target state_dict.
        sharded_tensor_func (Callable): the function to apply to ShardedTensor
        dtensor_func (Callable): the function to apply to DTensor
        tensor_func (Callable): the function to apply to Tensor
        pg (Optional[dist.ProcessGroup]): process group passed to tensor functions
        device (Optional[torch.device]): device passed to tensor functions
        cpu_offload (bool): whether to offload the tensors to CPU memory. This option is ignored
            if a companion_obj is supplied.
        companion_obj (Any): A companion object to the state dict. If this object
            is supplied, we attempt to copy the tensor to the companion object.
        ranks_only (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.
        non_blocking (bool): whether to use non-blocking copy when copying to the companion object.
    """
    # TODO: should we use pytree?
    cpu_device = torch.device("cpu")
    if isinstance(iter_object, ShardedTensor):
        ret = sharded_tensor_func(iter_object, pg, device, companion_obj)
    elif isinstance(iter_object, DTensor):
        ret = dtensor_func(iter_object, pg, device, companion_obj)
    elif isinstance(iter_object, torch.Tensor):
        ret = tensor_func(iter_object, pg, device, companion_obj)
    elif (
        isinstance(iter_object, (int, float, str, bytes, io.BytesIO))
        or iter_object is None
    ):
        ret = iter_object
    elif isinstance(iter_object, dict):
        if companion_obj is not None and (
            not isinstance(companion_obj, dict)
            or set(companion_obj.keys()) != set(iter_object.keys())
        ):
            msg = (
                ""
                if isinstance(companion_obj, dict)
                else f"{set(companion_obj.keys())=} {set(iter_object.keys())=}"
            )
            raise CompanionMismatch(msg)

        ret = {
            key: _iterate_state_dict(
                value,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[key] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for key, value in iter_object.items()
        }
    elif isinstance(iter_object, (list, tuple)):
        if companion_obj is not None and (
            not isinstance(companion_obj, (list, tuple))
            or len(companion_obj) != len(iter_object)
        ):
            raise CompanionMismatch

        ret = [
            _iterate_state_dict(
                v,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[idx] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for idx, v in enumerate(iter_object)
        ]
        if isinstance(iter_object, tuple):
            ret = tuple(ret)
    elif not type_check:
        ret = copy.deepcopy(iter_object)
    else:
        raise ValueError(f"Unexpected value type {type(iter_object)}")

    if not ranks_only or dist.get_rank(pg) in ranks_only:
        if isinstance(ret, torch.Tensor):
            if cpu_offload and companion_obj is None:
                ret = ret.to(cpu_device)

            if companion_obj is not None:
                if isinstance(companion_obj, DTensor):
                    if not isinstance(ret, DTensor):
                        raise AssertionError(
                            "ret must be a DTensor when companion_obj is a DTensor"
                        )
                    companion_obj._local_tensor.copy_(
                        ret._local_tensor, non_blocking=non_blocking
                    )
                elif isinstance(companion_obj, ShardedTensor):
                    if not isinstance(ret, ShardedTensor):
                        raise AssertionError(
                            "ret must be a ShardedTensor when companion_obj is a ShardedTensor"
                        )
                    for idx, shard in enumerate(companion_obj.local_shards()):
                        shard.tensor.copy_(
                            ret.local_shards()[idx].tensor, non_blocking=non_blocking
                        )
                else:
                    companion_obj.copy_(ret, non_blocking=non_blocking)
                ret = companion_obj
    else:
        ret = {} if isinstance(ret, dict) else None

    # pyrefly: ignore [bad-return]
    return ret


def _gather_state_dict(
    state_dict: dict[str, Any],
    *,
    pg: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
    cpu_offload: bool = False,
    ranks_only: tuple[int, ...] = (),
    type_check: bool = True,
) -> dict[str, Any]:
    """
    Given a state_dict, this API gathers all the ShardedTensors or DTensors in
    the state_dict.


    Args:
        state_dict (Dict[str, Any]): the target sharded state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        device: (Optional[torch.device]): the device that is used to
            perform allgather for ShardedTensor. Note that gathering a DTensor
            will use the DeviceMesh. So this argument will be ignored when
            gathering a DTensor.
        cpu_offload (bool): whether to offload the tensors to CPU memory. The
            default value is False.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check: (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        The gathered state dictionary.
    """

    def sharded_tensor_func(value, pg, device, companion_obj):
        # ShardedTensor does not seem to record the original device type.
        # So if the tensor is moved to CPU, we won't know the original type.
        # As a result, we have to rely on the user to tell us the correct one.
        cpu_device = torch.device("cpu")
        output_tensor = _all_gather_sharded_tensor(value, pg, device)
        local_shard_device = (
            value.local_shards()[0].tensor.device
            if value.local_shards()
            else cpu_device
        )
        if output_tensor.device != local_shard_device:
            value = output_tensor.to(local_shard_device)
        else:
            value = output_tensor
        return value

    def dtensor_func(value, pg, device, companion_obj):
        if value.device != value.device_mesh.device_type:
            value = value.to(value.device_mesh.device_type)
        # FSDP all_gather: [Shard(0)] -> [Replicate()]
        # HSDP all_gather: [Replicate(), Shard(0)] -> [Replicate(), Replicate()]
        # 2D FSDP + TP all_gather:
        # - [Shard(0), Shard(n)] -> [Replicate(), Replicate()]
        # - [Shard(0), Replicate()] -> [Replicate(), Replicate()]
        placements = [Replicate() for _ in value.placements]
        value = value.redistribute(
            device_mesh=value.device_mesh,
            placements=placements,
        )
        # Call `wait()` to force the tensor to be synchronous with respect
        # to the main stream.
        # See the discussion in https://github.com/pytorch/pytorch/pull/117799.
        value = value.to_local()
        if isinstance(value, AsyncCollectiveTensor):
            value = value.wait()
        return value

    return _iterate_state_dict(
        state_dict,
        sharded_tensor_func,
        dtensor_func,
        _identity_func,
        pg=pg,
        device=device,
        cpu_offload=cpu_offload,
        ranks_only=ranks_only,
        type_check=type_check,
    )


def _offload_state_dict_to_cpu(
    state_dict: dict[str, Any],
    *,
    ranks_only: tuple[int, ...] = (),
    type_check: bool = True,
) -> dict[str, Any]:
    """
    Given a state_dict, this API offload all the tensors to CPU memory.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check: (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        The gathered state dictionary.
    """

    ret = _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        _identity_func,
        pg=None,
        device=None,
        cpu_offload=True,
        ranks_only=ranks_only,
        type_check=type_check,
    )
    return ret


@torch.no_grad()
def _copy_state_dict(
    state_dict: dict[str, Any],
    copy_state_dict: dict[str, Any],
    non_blocking: bool = False,
    type_check: bool = True,
) -> dict[str, Any]:
    """
    Copies all tensors in a given state dict into a different state_dict with the
    same structure. Additionally, a copied state dict with the same value references
    is returned. Editing the keys on this state dict will not affect the
    passed in copy_state_dict (but the value references are the same).

    .. warning::
        It is expected by this function that state_dict and copy_state_dict share
        the same structure and data types.

    .. warning::
        The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        copy_state_dict (Dict[str, Any]):
            The state dict we are copying into. This state_dict must have exactly
             the same structure as the source `state_dict`.
        non_blocking: (bool): Whether copy ops should be performed asynchronously
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP. The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        State Dict copy
    """

    return _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        _identity_func,
        pg=None,
        device=None,
        cpu_offload=False,
        ranks_only=(),
        companion_obj=copy_state_dict,
        type_check=type_check,
        non_blocking=non_blocking,
    )


@torch.no_grad()
def _create_cpu_state_dict(
    state_dict: dict[str, Any], pin_memory: bool = False, share_memory: bool = False
) -> dict[str, Any]:
    """
    Given a state_dict, create another state_dict with the same structure and elements.
    However, all tensors in the returned state_dict are new tensors on CPU. These
    tensors can be placed on pin_memory or share_memory based on the provided arguments.

    .. warning::
        Setting both `pin_memory` and `share_memory` to True significantly increases the
        latency of this method because of the nuances which require us to register memory
        as pinned directly as opposed to relying on the pin_memory cache allocator. This
        option should only be used for long lived tensors which are required to be shared.
        This is not the case as long as at least one of `pin_memory` or `share_memory` is
         set to False.

    """

    def tensor_func(
        obj: torch.Tensor,
        pg: dist.ProcessGroup | None,
        device: torch.device | None,
        _: Any,
    ) -> torch.Tensor:
        if len(obj.size()) == 0:
            return torch.tensor(0, dtype=obj.dtype)

        # sometimes, a tensor might have non-zero size and 0 numel. In this case, pinning memory will fail
        # so we take a best guess at how to replicate the tensor below to maintain symmetry in the returned
        # state dict.
        if obj.numel() == 0 or obj.data_ptr() == 0:
            t = torch.zeros_like(obj, device="cpu")
            if share_memory:
                t = t.share_memory_()
            return t

        if share_memory:
            t = torch.empty(*tuple(obj.size()), dtype=obj.dtype)
            t = t.share_memory_()
            if pin_memory:
                pin_memory_utils.pin_memory(t.data_ptr(), t.numel() * t.element_size())
                weakref.finalize(t, pin_memory_utils.unpin_memory, t.data_ptr())

            return t
        elif pin_memory:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype).pin_memory()
        else:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype)

    def dtensor_func(
        obj: DTensor,
        pg: dist.ProcessGroup | None,
        device: torch.device | None,
        _: Any,
    ) -> DTensor:
        if len(obj.size()) == 0:
            return obj

        if obj.device != torch.device("cpu"):
            ret = cast(DTensor, obj.to(device="cpu"))
        else:
            ret = copy.deepcopy(obj)
        ret._local_tensor = tensor_func(ret._local_tensor, pg, device, None)
        return ret

    def sharded_tensor_func(
        obj: ShardedTensor,
        pg: dist.ProcessGroup | None,
        device: torch.device | None,
        _: Any,
    ) -> ShardedTensor:
        if not obj.local_shards():
            return obj

        if obj.device != torch.device("cpu"):
            ret = obj.to(device="cpu")
        else:
            ret = copy.deepcopy(obj)

        for shards in ret.local_shards():
            shards.tensor = tensor_func(shards.tensor, pg, device, None)

        return ret

    ret = _iterate_state_dict(
        state_dict,
        sharded_tensor_func,
        dtensor_func,
        tensor_func,
        pg=None,
        device=None,
        cpu_offload=False,
        ranks_only=(),
        type_check=False,
    )
    return ret


def _check_state_dict_similarity(
    state_dict: dict[str, Any],
    compared_state_dict: dict[str, Any],
) -> bool:
    """
    Given two state_dicts, check if the structures are the same. And
    if a [key, tensor] pair exist in one state_dict there must be
    a corresponding pair, [key, other_tensor], in the other state_dict,
    where tensor and other_tensor have the same size and dtype.

    Return the check result.
    """

    def tensor_func(
        obj: torch.Tensor,
        pg: dist.ProcessGroup | None,
        device: torch.device | None,
        companion_obj: Any,
    ) -> torch.Tensor:
        if companion_obj.dtype != obj.dtype or companion_obj.size() != obj.size():
            raise CompanionMismatch
        return obj

    try:
        _iterate_state_dict(
            state_dict,
            _identity_func,
            _identity_func,
            tensor_func,
            pg=None,
            device=None,
            cpu_offload=False,
            ranks_only=(),
            companion_obj=compared_state_dict,
            type_check=False,
        )
    except CompanionMismatch:
        return False

    return True


class _TensorInfo(NamedTuple):
    size: torch.Size
    dtype: torch.dtype


def _broadcast_tensors(
    full_state_dict: dict[str, Any],
    local_state_dict: dict[str, Any],
    keys: list[str],
    device: torch.device,
    pg: dist.ProcessGroup | None = None,
) -> None:
    if pg is None:
        pg = dist.distributed_c10d._get_default_group()
    pg_device = (
        device
        if device.type in {pg_device.type for pg_device in pg._device_types}
        else pg._device_types[0]
    )

    tensors: list[torch.Tensor] = []
    for key in keys:
        if dist.get_rank() == 0:
            full_state = full_state_dict[key]
            if not isinstance(full_state, torch.Tensor):
                raise AssertionError("full_state must be a torch.Tensor")
            full_tensor = full_state.detach().to(pg_device)
        else:
            tensor_info = full_state_dict[key]
            full_tensor = torch.empty(
                size=tensor_info.size,
                device=pg_device,
                dtype=tensor_info.dtype,
            )

        tensors.append(full_tensor)

        if (local_state := local_state_dict.get(key)) is None:
            continue

        local_state_dict[key] = (
            (local_state, full_tensor)
            if isinstance(local_state, DTensor)
            else full_tensor
        )

    if len(tensors) > 1:
        dist._broadcast_coalesced(pg, tensors, 500, 0)
    else:
        dist.broadcast(tensors[0], src=0, group=pg)

    if pg_device != device:
        for key, full_tensor in zip(keys, tensors):
            if (local_state := local_state_dict.get(key)) is not None:
                local_state_dict[key] = (
                    (local_state[0], full_tensor.to(device))
                    if (
                        isinstance(local_state, tuple)
                        and isinstance(local_state[0], DTensor)
                    )
                    else full_tensor.to(device)
                )

    _distribute_tensors(local_state_dict, keys, device, pg)


def _distribute_tensors(
    local_state_dict: dict[str, Any],
    keys: list[str],
    device: torch.device,
    pg: dist.ProcessGroup | None = None,
) -> None:
    if pg is None:
        pg = dist.distributed_c10d._get_default_group()
    for key in keys:
        _local_state = local_state_dict.get(key)
        if _local_state is None or torch.is_tensor(_local_state):
            continue

        local_state = _local_state[0]
        full_tensor = _local_state[1]

        shape, offset = compute_local_shape_and_global_offset(
            full_tensor.shape, local_state.device_mesh, local_state.placements
        )
        slices = [
            slice(cur_offset, cur_offset + cur_shape)
            for cur_shape, cur_offset in zip(shape, offset)
        ]
        if local_state.is_meta:
            # Use .clone() here rather than view to clone and return only the sliced portion, minimizing memory access and cost.
            local_tensor = full_tensor[tuple(slices)].detach().clone()
            # TODO: currently, we cannot handle strided sharding if the dp dimension is not even. For example,
            # one of the case that is not yet supported is when placements = (Shard(0), _StridedShard(0, sf=2)).
            ret = DTensor.from_local(
                local_tensor,
                local_state.device_mesh,
                local_state.placements,
                shape=local_state.shape,
                stride=local_state.stride(),
            )
        else:
            ret = local_state
            # Copy full_tensor[slices] into local_state.to_local() to reduce memory footprint.
            ret.to_local().copy_(full_tensor[tuple(slices)])
        local_state_dict[key] = ret


def _broadcast_state_dict(
    full_state_dict: dict[str, Any],
    local_state_dict: dict[str, Any],
    device: torch.device,
    pg: dist.ProcessGroup | None = None,
    strict: bool = False,
    cpu_offload: bool = False,
) -> None:
    # Broadcast from rank0's `full_state_dict` to all ranks' `local_state_dict`.
    # If strict is True, any keys in `local_state_dict` but not in `full_state_dict`
    # will be removed from `local_state_dict`.
    ret = {}
    if dist.get_rank() == 0:
        for key, value in full_state_dict.items():
            if not torch.is_tensor(value):
                ret[key] = value
            elif value.dim() == 0:
                ret[key] = value.cpu()
            else:
                ret[key] = _TensorInfo(value.size(), value.dtype)

    broadcast_list = [ret]
    dist.broadcast_object_list(broadcast_list, src=0, group=pg)
    ret = broadcast_list[0]
    # Gather values
    keys = []
    local_state_dict_keys = set(local_state_dict.keys())
    global_keys = set()
    for key, value in ret.items():
        global_keys.add(key)
        if not isinstance(value, _TensorInfo):
            if key in local_state_dict:
                local_state_dict[key] = value
            continue

        if dist.get_rank() == 0:
            ret[key] = full_state_dict[key]

        keys.append(key)
        # Broadcast every tensor to avoid OOM for now.
        if len(keys) >= 1:
            _broadcast_tensors(ret, local_state_dict, keys, device, pg)
            if cpu_offload:
                for key in keys:
                    local_state_dict[key] = local_state_dict[key].cpu()
            keys.clear()

    if strict:
        if missing_keys := (local_state_dict_keys - global_keys):
            for key in missing_keys:
                local_state_dict.pop(key)

    if keys:
        _broadcast_tensors(ret, local_state_dict, keys, device, pg)
        if cpu_offload:
            for key in keys:
                local_state_dict[key] = local_state_dict[key].cpu()


def _distribute_state_dict(
    full_state_dict: dict[str, Any],
    local_state_dict: dict[str, Any],
    device: torch.device,
    pg: dist.ProcessGroup | None = None,
) -> None:
    # Full_state_dict = True, broadcast_from_rank0 = False here. Each rank has
    # full_state_dict. Skip the broadcast in ``_broadcast_state_dict`` and
    # distribute tensors in each rank
    for key, value in full_state_dict.items():
        if key not in full_state_dict:
            continue
        if not torch.is_tensor(value):
            local_state_dict[key] = value
        elif value.dim() == 0:
            local_state_dict[key] = value.cpu()
        else:
            if not isinstance(value, torch.Tensor):
                raise AssertionError("value must be a torch.Tensor")
            local_state = local_state_dict.get(key)
            if local_state is None:
                continue
            elif isinstance(local_state, DTensor):
                local_state_dict[key] = distribute_tensor(
                    value.detach().to(device),
                    local_state.device_mesh,
                    local_state.placements,
                )
            else:
                local_state_dict[key] = value.detach().to(device)


# These APIs are from torch.distributed.checkpoint.
# TODO: We should consolidate the code here as some not all modules can depend on
# DCP.
PATH_ITEM = str | int
OBJ_PATH = tuple[PATH_ITEM, ...]
FLATTEN_MAPPING = dict[str, OBJ_PATH]
STATE_DICT_TYPE = dict[str, Any]
CONTAINER_TYPE = MutableMapping[PATH_ITEM, Any]


def _traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, Any], None],
) -> None:
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Mapping, list, and tuple will be flattened and other value types are treated
    as the terminal values and will invoke ``visitor``.
    """

    def _traverse_obj(path: OBJ_PATH, value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)
        else:
            visitor(path, value)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


def _flatten_state_dict(
    state_dict: STATE_DICT_TYPE,
) -> tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    """
    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.

    Use ``unflatten_state_dict`` to revert this process.
    Returns:
        A tuple with the flatten state_dict and a mapping from original to new state_dict.
    N.B. The new keys are derived from the object paths, joined by dot.
        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.
    """
    flattened: STATE_DICT_TYPE = {}
    mappings: FLATTEN_MAPPING = {}

    def flat_copy(path: OBJ_PATH, value: Any) -> None:
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    _traverse_state_dict(state_dict, flat_copy)
    return flattened, mappings


def _set_element(root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: Any) -> None:
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
    cur_container = cast(CONTAINER_TYPE, root_dict)

    def extend_list(lst: list[Any], idx: int) -> None:
        while len(lst) <= idx:
            lst.append(None)

    for i in range(1, len(path)):
        prev_key = path[i - 1]
        key = path[i]
        def_val: CONTAINER_TYPE | list[Any] = {} if type(key) is str else []

        if isinstance(cur_container, Mapping):
            cur_container = cast(
                CONTAINER_TYPE, cur_container.setdefault(prev_key, def_val)
            )
        else:
            # pyrefly: ignore [bad-argument-type]
            extend_list(cur_container, prev_key)
            if cur_container[prev_key] is None:
                cur_container[prev_key] = def_val
            cur_container = cur_container[prev_key]

    key = path[-1]
    if type(key) is int:
        extend_list(cast(list[Any], cur_container), key)

    cur_container[key] = value


def _unflatten_state_dict(
    state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING
) -> STATE_DICT_TYPE:
    """Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``."""
    nested: STATE_DICT_TYPE = {}
    for key, value in state_dict.items():
        _set_element(nested, mapping[key], value)
    return nested
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from argparse import Action


class env(Action):
    """
    Get argument values from ``PET_{dest}`` before defaulting to the given ``default`` value.

    For flags (e.g. ``--standalone``)
    use ``check_env`` instead.

    .. note:: when multiple option strings are specified, ``dest`` is
              the longest option string (e.g. for ``"-f", "--foo"``
              the env var to set is ``PET_FOO`` not ``PET_F``)

    Example:
    ::

     parser.add_argument("-f", "--foo", action=env, default="bar")

     ./program                                      -> args.foo="bar"
     ./program -f baz                               -> args.foo="baz"
     ./program --foo baz                            -> args.foo="baz"
     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"
     PET_FOO="env_bar" ./program --foo baz -> args.foo="baz"
     PET_FOO="env_bar" ./program           -> args.foo="env_bar"

     parser.add_argument("-f", "--foo", action=env, required=True)

     ./program                                      -> fails
     ./program -f baz                               -> args.foo="baz"
     PET_FOO="env_bar" ./program           -> args.foo="env_bar"
     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"
    """

    def __init__(self, dest, default=None, required=False, **kwargs) -> None:
        env_name = f"PET_{dest.upper()}"
        default = os.environ.get(env_name, default)

        # ``required`` means that it NEEDS to be present  in the command-line args
        # rather than "this option requires a value (either set explicitly or default"
        # so if we found default then we don't "require" it to be in the command-line
        # so set it to False
        if default:
            required = False

        super().__init__(dest=dest, default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class check_env(Action):
    """
    Check whether the env var ``PET_{dest}`` exists before defaulting to the given ``default`` value.

    Equivalent to
    ``store_true`` argparse built-in action except that the argument can
    be omitted from the commandline if the env var is present and has a
    non-zero value.

    .. note:: it is redundant to pass ``default=True`` for arguments
              that use this action because a flag should be ``True``
              when present and ``False`` otherwise.

    Example:
    ::

     parser.add_argument("--verbose", action=check_env)

     ./program                                  -> args.verbose=False
     ./program --verbose                        -> args.verbose=True
     PET_VERBOSE=1 ./program           -> args.verbose=True
     PET_VERBOSE=0 ./program           -> args.verbose=False
     PET_VERBOSE=0 ./program --verbose -> args.verbose=True

    Anti-pattern (don't do this):

    ::

     parser.add_argument("--verbose", action=check_env, default=True)

     ./program                                  -> args.verbose=True
     ./program --verbose                        -> args.verbose=True
     PET_VERBOSE=1 ./program           -> args.verbose=True
     PET_VERBOSE=0 ./program           -> args.verbose=False

    """

    def __init__(self, dest, default=False, **kwargs) -> None:
        env_name = f"PET_{dest.upper()}"
        default = bool(int(os.environ.get(env_name, "1" if default else "0")))
        super().__init__(dest=dest, const=True, default=default, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)
<|endoftext|>#!/usr/bin/env python3


"""
A set of primitive functions for performing collective ops.

Each should also handle single rank scenario.
"""

from __future__ import annotations

import importlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast, Generic, TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist


__all__: list[str] = [
    "SyncPayload",
    "broadcast",
    "all_gather",
    "all_gather_object_enforce_type",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SyncPayload(Generic[T]):
    stage_name: str | None
    success: bool
    payload: T
    exception: Exception | None = None


def broadcast(
    data_or_fn: T | Callable[[], T],
    *,
    success: bool = True,
    stage_name: str | None = None,
    rank: int = 0,
    pg: dist.ProcessGroup | None = None,
) -> T:
    """
    Broadcasts the data payload from rank 0 to all other ranks.
    Or if a function is passed, execute it in rank 0 and broadcast result to all other ranks.

    Can be used to broadcast a failure signal to stop all ranks.

    If the function raises an exception, all ranks will raise.

    Args:
        data_or_fn: the data to broadcast or function to execute and broadcast result.
        success: False to stop all ranks.
        stage_name: the name of the logical stage for synchronization and debugging
        rank: rank to broadcast data or execute function and broadcast results.
        pg: the process group for sync
    Throws:
        RuntimeError from original exception trace
    Returns:
        the value after synchronization

    Example usage:
    >> id = broadcast(data_or_fn=allocate_id, rank=0, pg=ext_pg.my_pg)
    """

    if not success and data_or_fn is not None:
        raise AssertionError(
            "Data or Function is expected to be None if not successful"
        )

    payload: T | None = None
    exception: Exception | None = None
    # if no pg is passed then execute if rank is 0
    if (pg is None and rank == 0) or (pg is not None and pg.rank() == rank):
        # determine if it is an executable function or data payload only
        if callable(data_or_fn):
            try:
                payload = data_or_fn()
            except Exception as e:
                success = False
                exception = e
        else:
            payload = data_or_fn

    # broadcast the exception type if any to all ranks for failure categorization
    sync_obj = SyncPayload(
        stage_name=stage_name,
        success=success,
        payload=payload,
        exception=exception,
    )

    if pg is not None:
        broadcast_list = [sync_obj]
        dist.broadcast_object_list(broadcast_list, src=rank, group=pg)
        if len(broadcast_list) != 1:
            raise AssertionError(
                f"Expected broadcast_list to have exactly 1 element, got {len(broadcast_list)}"
            )
        sync_obj = broadcast_list[0]

    # failure in any rank will trigger a throw in every rank.
    if not sync_obj.success:
        error_msg = f"Rank {rank} failed"
        if stage_name is not None:
            error_msg += f": stage {sync_obj.stage_name}"
        if sync_obj.exception is not None:
            error_msg += f": exception {sync_obj.exception}"

        raise RuntimeError(error_msg) from sync_obj.exception

    return cast(T, sync_obj.payload)


def all_gather(
    data_or_fn: T | Callable[[], T],
    stage_name: str | None = None,
    pg: dist.ProcessGroup | None = None,
) -> list[T]:
    """
    A simple all_gather primitive with basic synchronization guard logic,
    by checking payload from all ranks has the same stage name.

    Args:
        data_or_fn: the data to be all gathered across ranks or function to be executed
        stage_name: the sync stage name for out-of-sync protection
        pg: the process group for sync
    Throws:
        RuntimeError from original exception trace
    Returns:
        a list of synced data from all ranks

    Example usage:
    >> all_ids = all_gather(data_or_fn=allocate_id, pg=ext_pg.my_pg)
    """
    payload: T | None = None
    exception: Exception | None = None
    success = True
    # determine if it is an executable function or data payload only
    if callable(data_or_fn):
        try:
            payload = data_or_fn()
        except Exception as e:
            success = False
            exception = e
    else:
        payload = data_or_fn

    sync_obj = SyncPayload(
        stage_name=stage_name,
        success=success,
        payload=payload,
        exception=exception,
    )

    if pg is not None:
        # List of success/failure across all ranks.
        total_list = [None] * dist.get_world_size(pg)
        all_gather_object_enforce_type(pg, total_list, sync_obj)
        # Each rank will throw RuntimeError in case of failure on any rank.
        stage_name = cast(SyncPayload[T], total_list[0]).stage_name
        exception_list: list[tuple[int, Exception]] = []
        ret_list: list[T] = []
        error_msg: str = ""

        for i, sp in enumerate(cast(list[SyncPayload[T]], total_list)):
            if sp.stage_name != stage_name:
                error_msg += (
                    f"Unexpected stage name received from rank {i}: {sp.stage_name} "
                )
                continue
            if not sp.success and sp.exception is not None:
                exception_list.append((i, sp.exception))
                continue
            ret_list.append(sp.payload)

        if len(exception_list) > 0:
            raise RuntimeError(  # type: ignore[misc]
                error_msg,
                exception_list,
            ) from exception_list[0]  # pyrefly: ignore [bad-raise]
        return ret_list
    else:
        if not sync_obj.success:
            raise RuntimeError(
                f"all_gather failed with exception {sync_obj.exception}",
            ) from sync_obj.exception
        return [sync_obj.payload]  # type: ignore[list-item]


# Note: use Any for typing for now so users can pass in
# either a list of None or target type placeholders
# otherwise pyre would complain
def all_gather_object_enforce_type(
    pg: dist.ProcessGroup,
    # pyre-fixme[2]: Parameter must have a type that does not contain `Any`
    object_list: list[Any],
    # pyre-fixme[2]: Parameter must have a type other than `Any`
    obj: Any,
    # pyre-fixme[2]: Parameter must have a type that does not contain `Any`
    type_checker: Callable[[Any, Any], bool] = lambda x, y: type(x) is type(y),
) -> None:
    """
    Similar to plain all_gather_object but with additional type checking
    AFTER gather is done to ensure basic consistency.
    If check does not pass, all ranks will fail with exception.

    This is generally to prevent conditional logic leading to
    unexpected messages being received. This is considered fatal code error,
    but due to logic stacks this might happen implicitly in practice.

    The default check does not check sub type (considered different)
    or covariance (considered same) but users can pass in custom checker
    if more complicated check is needed.
    """
    dist.all_gather_object(object_list, obj, group=pg)

    # conservative check
    list_len = len(object_list)
    if list_len == 0:
        return
    first_obj = object_list[0]
    for i in range(1, list_len):
        if not type_checker(first_obj, object_list[i]):
            raise TypeError(
                f"Object type at index {i} is {type(object_list[i])}, "
                f"while first object type is {type(first_obj)}"
            )


def _summarize_ranks(ranks: Iterable[int]) -> str:
    ranks = sorted(ranks)
    if min(ranks) < 0:
        raise AssertionError("ranks should all be positive")
    if len(set(ranks)) != len(ranks):
        raise AssertionError("ranks should not contain duplicates")
    curr: int | range | None = None
    ranges = []
    while ranks:
        x = ranks.pop(0)
        if curr is None:
            curr = x
        elif isinstance(curr, int):
            if x == curr + 1:
                curr = range(curr, x + 1, 1)
            else:
                step = x - curr
                curr = range(curr, x + step, step)
        else:
            if not isinstance(curr, range):
                raise AssertionError("curr must be an instance of range")
            if x == curr.stop:
                curr = range(curr.start, curr.stop + curr.step, curr.step)
            else:
                ranges.append(curr)
                curr = x

    if isinstance(curr, int):
        ranges.append(range(curr, curr + 1, 1))
    elif isinstance(curr, range):
        ranges.append(curr)

    result = []
    for r in ranges:
        if len(r) == 1:
            result.append(f"{r.start}")
        elif r.step == 1:
            result.append(f"{r.start}:{r.stop}")
        else:
            result.append(f"{r.start}:{r.stop}:{r.step}")
    return ",".join(result)


def _check_philox_rng_sync(
    generator: torch.Generator, group: dist.ProcessGroup
) -> tuple[dict[Any, set], str]:
    local_state = generator.get_state()
    all_states = [torch.empty_like(local_state) for _ in range(group.size())]
    torch.distributed.all_gather(all_states, local_state)
    seeds_offsets = [
        (state[:8].view(torch.uint64).item(), state[8:].view(torch.uint64).item())
        for state in all_states
    ]
    seed_offset_ranks = defaultdict(set)
    for rank, (seed, offset) in enumerate(seeds_offsets):
        seed_offset_ranks[(seed, offset)].add(rank)
    return seed_offset_ranks, "(Seed, Offset)"


def _check_cpu_rng_sync(
    generator: torch.Generator, group: dist.ProcessGroup
) -> tuple[dict[Any, set], str]:
    # seed is returned as uint64_t from C impl, so may not fit in torch int64 tensor directly.
    state_tensor = generator.get_state()
    all_state_tensors = [torch.empty_like(state_tensor) for _ in range(group.size())]
    torch.distributed.all_gather(all_state_tensors, state_tensor)
    state_ranks = defaultdict(set)
    for rank, state_tensor in enumerate(all_state_tensors):
        # Summarize the state vector of the CPU rng.
        # The properties that matter most are (1) its different if there is a state difference, (2) its printable
        # (see desync table- not viable to print whole state vector of size 5k)
        state_ranks[torch.hash_tensor(state_tensor).item()].add(rank)
    return state_ranks, "Generator state hash"


def _check_rng_sync_internal(
    generator: torch.Generator, group: dist.ProcessGroup
) -> tuple[dict[Any, set], str]:
    if generator.device.type in {"cuda", "xpu"}:
        return _check_philox_rng_sync(generator, group)
    elif generator.device.type == "cpu":
        return _check_cpu_rng_sync(generator, group)
    else:
        raise NotImplementedError(
            f"Unsupported generator device: {generator.device.type}"
        )


def _desync_table_str(tag: str, value_ranks: dict[Any, set[int]]) -> str:
    headers = ["Ranks", f"{tag} values"]
    rank_values = [
        [_summarize_ranks(ranks), str(value)] for value, ranks in value_ranks.items()
    ]
    if importlib.util.find_spec("tabulate"):
        from tabulate import tabulate

        return tabulate(rank_values, headers=headers)
    row_str = "\n".join([str(row) for row in rank_values])
    return str(f"{headers}\n{row_str}")


def _check_rng_sync(generator: torch.Generator, group: dist.ProcessGroup) -> str | None:
    value_ranks, value_header = _check_rng_sync_internal(generator, group)
    log_str = None
    if len(value_ranks) > 1:
        log_str = f"Generator desync detected:\n{_desync_table_str(value_header, value_ranks)}"
        logger.error(log_str)
    return log_str
<|endoftext|># mypy: allow-untyped-defs
r"""
Module ``torch.distributed.launch``.

``torch.distributed.launch`` is a module that spawns up multiple distributed
training processes on each of the training nodes.

.. warning::

    This module is going to be deprecated in favor of :ref:`torchrun <launcher-api>`.

The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned. The utility can be used for either
CPU training or GPU training. If the utility is used for GPU training,
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be beneficial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.

In both cases of single-node distributed training or multi-node distributed
training, this utility will launch the given number of processes per node
(``--nproc-per-node``). If used for GPU training, this number needs to be less
or equal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.

**How to use this module:**

1. Single-Node multi-process distributed training

::

    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

2. Multi-Node multi-process distributed training: (e.g. two nodes)


Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

::

    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node-rank=0 --master-addr="192.168.1.1"
               --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

Node 2:

::

    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node-rank=1 --master-addr="192.168.1.1"
               --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

3. To look up what optional arguments this module offers:

::

    python -m torch.distributed.launch --help


**Important Notices:**

1. This utility and multi-process distributed (single-node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.

2. In your training program, you must parse the command-line argument:
``--local-rank=LOCAL_PROCESS_RANK``, which will be provided by this module.
If your training program uses GPUs, you should ensure that your code only
runs on the GPU device of LOCAL_PROCESS_RANK. This can be done by:

Parsing the local_rank argument

::

    >>> # xdoctest: +SKIP
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--local-rank", "--local_rank", type=int)
    >>> args = parser.parse_args()

Set your device to local rank using either

::

    >>> torch.cuda.set_device(args.local_rank)  # before your code runs

or

::

    >>> with torch.cuda.device(args.local_rank):
    >>>    # your code to run
    >>>    ...

.. versionchanged:: 2.0.0

    The launcher will passes the ``--local-rank=<rank>`` argument to your script.
    From PyTorch 2.0.0 onwards, the dashed ``--local-rank`` is preferred over the
    previously used underscored ``--local_rank``.

    For backward compatibility, it may be necessary for users to handle both
    cases in their argument parsing code. This means including both ``"--local-rank"``
    and ``"--local_rank"`` in the argument parser. If only ``"--local_rank"`` is
    provided, the launcher will trigger an error: "error: unrecognized arguments:
    --local-rank=<rank>". For training code that only supports PyTorch 2.0.0+,
    including ``"--local-rank"`` should be sufficient.

3. In your training program, you are supposed to call the following function
at the beginning to start the distributed backend. It is strongly recommended
that ``init_method=env://``. Other init methods (e.g. ``tcp://``) may work,
but ``env://`` is the one that is officially supported by this module.

::

    >>> torch.distributed.init_process_group(backend='YOUR BACKEND',
    >>>                                      init_method='env://')

4. In your training program, you can either use regular distributed functions
or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
training program uses GPUs for training and you would like to use
:func:`torch.nn.parallel.DistributedDataParallel` module,
here is how to configure it.

::

    >>> model = torch.nn.parallel.DistributedDataParallel(model,
    >>>                                                   device_ids=[args.local_rank],
    >>>                                                   output_device=args.local_rank)

Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[args.local_rank]``,
and ``output_device`` needs to be ``args.local_rank`` in order to use this
utility

5. Another way to pass ``local_rank`` to the subprocesses via environment variable
``LOCAL_RANK``. This behavior is enabled when you launch the script with
``--use-env=True``. You must adjust the subprocess example above to replace
``args.local_rank`` with ``os.environ['LOCAL_RANK']``; the launcher
will not pass ``--local-rank`` when you specify this flag.

.. warning::

    ``local_rank`` is NOT globally unique: it is only unique per process
    on a machine.  Thus, don't use it to decide if you should, e.g.,
    write to a networked filesystem.  See
    https://github.com/pytorch/pytorch/issues/12042 for an example of
    how things can go wrong if you don't do this correctly.



"""

from typing_extensions import deprecated as _deprecated

from torch.distributed.run import get_args_parser, run


def parse_args(args):
    parser = get_args_parser()
    parser.add_argument(
        "--use-env",
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local-rank as argument, and will instead set LOCAL_RANK.",
    )
    return parser.parse_args(args)


def launch(args):
    if args.no_python and not args.use_env:
        raise ValueError(
            "When using the '--no-python' flag, you must also set the '--use-env' flag."
        )
    run(args)


@_deprecated(
    "The module torch.distributed.launch is deprecated\n"
    "and will be removed in future. Use torchrun.\n"
    "Note that --use-env is set by default in torchrun.\n"
    "If your script expects `--local-rank` argument to be set, please\n"
    "change it to read from `os.environ['LOCAL_RANK']` instead. See \n"
    "https://pytorch.org/docs/stable/distributed.html#launch-utility for \n"
    "further instructions\n",
    category=FutureWarning,
)
def main(args=None):
    args = parse_args(args)
    launch(args)


if __name__ == "__main__":
    main()
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.distributed.distributed_c10d as c10d


"""
This file contains the op impls for the legacy (c10d_functional) functional collectives.
These impls simply call into the native (_c10d_functional) functional collectives.
"""


def _broadcast(input, src, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.broadcast(
        input,
        src,
        group_name,
    )


def _all_reduce(input, reduce_op, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_reduce(
        input,
        reduce_op,
        group_name,
    )


def _all_reduce_coalesced(inputs, reduce_op, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_reduce_coalesced(
        inputs,
        reduce_op,
        group_name,
    )


def _all_gather_into_tensor(input, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_gather_into_tensor(
        input,
        group_size,
        group_name,
    )


def _all_gather_into_tensor_coalesced(input, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
        input,
        group_size,
        group_name,
    )


def _reduce_scatter_tensor(
    input: torch.Tensor,
    reduce_op: str,
    tag: str,
    ranks: list[int],
    group_size: int,
):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.reduce_scatter_tensor(
        input,
        reduce_op,
        group_size,
        group_name,
    )


def _reduce_scatter_tensor_coalesced(
    inputs: list[torch.Tensor],
    reduce_op: str,
    tag: str,
    ranks: list[int],
    group_size: int,
):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
        inputs,
        reduce_op,
        group_size,
        group_name,
    )


def _all_to_all_single(
    input: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    tag: str,
    ranks: list[int],
    group_size: int,
):
    if output_split_sizes is None or input_split_sizes is None:
        if not (output_split_sizes is None and input_split_sizes is None):
            raise AssertionError(
                "output_split_sizes and input_split_sizes must either be "
                "specified together or both set to None"
            )
        output_split_sizes = [input.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_to_all_single(
        input,
        output_split_sizes,
        input_split_sizes,
        group_name,
    )


def _wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.ops._c10d_functional.wait_tensor(tensor)


def _isend(tensor: torch.Tensor, dst: int, tag: str, group_name):
    return torch.ops._c10d_functional.isend(tensor, dst, tag, group_name)


def _irecv(tensor: torch.Tensor, src: int, tag: str, group_name):
    return torch.ops._c10d_functional.irecv(tensor, src, tag, group_name)


def _batch_p2p_ops(
    op_list: list[str],
    peer_list: list[int],
    tag_list: list[int],
    tensors: list[torch.Tensor],
    group_name: str,
):
    return torch.ops._c10d_functional.batch_p2p_ops(
        op_list, peer_list, tag_list, tensors, group_name
    )
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import math
import sys
import warnings
from collections.abc import Callable, Sequence
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.compiler.config
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch._utils import _maybe_view_chunk_cat
from torch.distributed.device_mesh import DeviceMesh
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from . import _functional_collectives_impl as fun_col_impl


try:
    from torch.utils._cxx_pytree import tree_map_only
except ImportError:
    from torch.utils._pytree import tree_map_only  # type: ignore[no-redef]


try:
    from torch.compiler import is_dynamo_compiling as is_torchdynamo_compiling
except Exception:
    warnings.warn(
        "Unable to import torchdynamo util `is_torchdynamo_compiling`, so won't support torchdynamo correctly",
        stacklevel=2,
    )

    def is_torchdynamo_compiling():  # type: ignore[misc]
        return False


"""
New traceable, functional collectives.
RFC: https://github.com/pytorch/pytorch/issues/93173

  compiler: trace these ops with plain-old-data schemas, then choose how to lower them.
  eager: execute these 'functional' ops which in eager return AsyncCollectiveTensor subclasses,
         automatically calling .wait() on underlying/hidden async 'work' obj only when fed to
         a downstream op.

