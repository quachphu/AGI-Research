        self._overrides: dict[_K, _V] = {}
        self._merged: dict[_K, _V] = {}

    def set_base(self, key: _K, value: _V) -> None:
        self._base[key] = value
        if key not in self._overrides:
            self._merged[key] = value

    def in_base(self, key: _K) -> bool:
        """Checks if a key is in the base dictionary."""
        return key in self._base

    def override(self, key: _K, value: _V) -> None:
        """Overrides a base key-value with a new pair."""
        self._overrides[key] = value
        self._merged[key] = value

    def remove_override(self, key: _K) -> None:
        """Un-overrides a key-value pair."""
        self._overrides.pop(key, None)  # type: ignore[arg-type]
        self._merged.pop(key, None)  # type: ignore[arg-type]
        if key in self._base:
            self._merged[key] = self._base[key]

    def overridden(self, key: _K) -> bool:
        """Checks if a key-value pair is overridden."""
        return key in self._overrides

    def __getitem__(self, key: _K) -> _V:
        return self._merged[key]

    def get(self, key: _K, default: _V | None = None):
        return self._merged.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._merged

    def __iter__(self):
        return iter(self._merged)

    def __len__(self) -> int:
        return len(self._merged)

    def __repr__(self) -> str:
        return f"OverrideDict(base={self._base}, overrides={self._overrides})"

    def __bool__(self) -> bool:
        return bool(self._merged)


class _SymbolicFunctionGroup:
    """Different versions of symbolic functions registered to the same name.

    O(number of registered versions of an op) search is performed to find the most
    recent version of the op.

    The registration is delayed until op is used to improve startup time.

    Function overloads with different arguments are not allowed.
    Custom op overrides are supported.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        # A dictionary of functions, keyed by the opset version.
        self._functions: OverrideDict[OpsetVersion, Callable] = OverrideDict()

    def __repr__(self) -> str:
        return f"_SymbolicFunctionGroup({self._name}, registered={self._functions})"

    def __getitem__(self, key: OpsetVersion) -> Callable:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    # TODO(justinchuby): Add @functools.lru_cache(maxsize=None) if lookup time becomes
    # a problem.
    def get(self, opset: OpsetVersion) -> Callable | None:
        """Find the most recent version of the function."""
        version = _dispatch_opset_version(opset, self._functions)
        if version is None:
            return None

        return self._functions[version]

    def add(self, func: Callable, opset: OpsetVersion) -> None:
        """Adds a symbolic function.

        Args:
            func: The function to add.
            opset: The opset version of the function to add.
        """
        if self._functions.in_base(opset):
            warnings.warn(
                f"Symbolic function '{self._name}' already registered for opset {opset}. "
                f"Replacing the existing function with new function. This is unexpected. "
                f"Please report it on {_constants.PYTORCH_GITHUB_ISSUES_URL}.",
                errors.OnnxExporterWarning,
                stacklevel=2,
            )
        self._functions.set_base(opset, func)

    def add_custom(self, func: Callable, opset: OpsetVersion) -> None:
        """Adds a custom symbolic function.

        Args:
            func: The symbolic function to register.
            opset: The corresponding opset version.
        """
        self._functions.override(opset, func)

    def remove_custom(self, opset: OpsetVersion) -> None:
        """Removes a custom symbolic function.

        Args:
            opset: The opset version of the custom function to remove.
        """
        if not self._functions.overridden(opset):
            warnings.warn(
                f"No custom function registered for '{self._name}' opset {opset}",
                stacklevel=2,
            )
            return
        self._functions.remove_override(opset)

    def get_min_supported(self) -> OpsetVersion:
        """Returns the lowest built-in opset version supported by the function."""
        return min(self._functions)


class SymbolicRegistry:
    """Registry for symbolic functions.

    The registry maintains a mapping from qualified names to symbolic functions.
    It is used to register new symbolic functions and to dispatch calls to
    the appropriate function.
    """

    def __init__(self) -> None:
        self._registry: dict[str, _SymbolicFunctionGroup] = {}

    def register(
        self, name: str, opset: OpsetVersion, func: Callable, custom: bool = False
    ) -> None:
        """Registers a symbolic function.

        Args:
            name: The qualified name of the function to register. In the form of 'domain::op'.
                E.g. 'aten::add'.
            opset: The opset version of the function to register.
            func: The symbolic function to register.
            custom: Whether the function is a custom function that overrides existing ones.

        Raises:
            ValueError: If the separator '::' is not in the name.
        """
        if "::" not in name:
            raise ValueError(
                f"The name must be in the form of 'domain::op', not '{name}'"
            )
        symbolic_functions = self._registry.setdefault(
            name, _SymbolicFunctionGroup(name)
        )
        if custom:
            symbolic_functions.add_custom(func, opset)
        else:
            symbolic_functions.add(func, opset)

    def unregister(self, name: str, opset: OpsetVersion) -> None:
        """Unregisters a symbolic function.

        Args:
            name: The qualified name of the function to unregister.
            opset: The opset version of the function to unregister.
        """
        if name not in self._registry:
            return
        self._registry[name].remove_custom(opset)

    def get_function_group(self, name: str) -> _SymbolicFunctionGroup | None:
        """Returns the function group for the given name."""
        return self._registry.get(name)

    def is_registered_op(self, name: str, version: int) -> bool:
        """Returns whether the given op is registered for the given opset version."""
        functions = self.get_function_group(name)
        if functions is None:
            return False
        return functions.get(version) is not None

    def all_functions(self) -> set[str]:
        """Returns the set of all registered function names."""
        return set(self._registry)


def onnx_symbolic(
    name: str,
    opset: OpsetVersion | Sequence[OpsetVersion],
    decorate: Sequence[Callable] | None = None,
    custom: bool = False,
) -> Callable:
    """Registers a symbolic function.

    Usage::

    ```
    @onnx_symbolic(
        "aten::symbolic_b",
        opset=10,
        decorate=[quantized_aten_handler(scale=1 / 128, zero_point=0)],
    )
    @symbolic_helper.parse_args("v", "v", "b")
    def symbolic_b(g: _C.Graph, x: _C.Value, y: _C.Value, arg1: bool) -> _C.Value: ...
    ```

    Args:
        name: The qualified name of the function in the form of 'domain::op'.
            E.g. 'aten::add'.
        opset: The opset versions of the function to register at.
        decorate: A sequence of decorators to apply to the function.
        custom: Whether the function is a custom symbolic function.

    Raises:
        ValueError: If the separator '::' is not in the name.
    """

    def wrapper(func: Callable[_P, _R]) -> Callable[_P, _R]:
        decorated = func
        if decorate is not None:
            for decorate_func in decorate:
                decorated = decorate_func(decorated)

        global registry
        nonlocal opset
        if isinstance(opset, OpsetVersion):
            opset = (opset,)
        for opset_version in opset:
            registry.register(name, opset_version, decorated, custom=custom)

        # Return the original function because the decorators in "decorate" are only
        # specific to the instance being registered.
        return func

    return wrapper


def custom_onnx_symbolic(
    name: str,
    opset: OpsetVersion | Sequence[OpsetVersion],
    decorate: Sequence[Callable] | None = None,
) -> Callable:
    """Registers a custom symbolic function.

    Args:
        name: the qualified name of the function.
        opset: the opset version of the function.
        decorate: a sequence of decorators to apply to the function.

    Returns:
        The decorator.

    Raises:
        ValueError: If the separator '::' is not in the name.
    """
    return onnx_symbolic(name, opset, decorate, custom=True)


# The registry for all symbolic functions.
registry = SymbolicRegistry()
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
import torch._ops


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.onnx._internal.exporter import _registration


def get_onnx_implemented_overloads(
    registry: _registration.ONNXRegistry,
) -> list[_registration.TorchOp]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry: The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
    registered_ops: list[_registration.TorchOp] = []
    for onnx_decomp_meta in registry.functions.values():
        if len(onnx_decomp_meta) == 0:
            raise AssertionError("onnx_decomp_meta must not be empty")
        # Different OnnxDecompMeta for the same TorchOp should
        # have the same fx_target.
        fx_target = onnx_decomp_meta[0].fx_target
        registered_ops.append(fx_target)
    return registered_ops


def create_onnx_friendly_decomposition_table(
    onnx_registered_ops: set[_registration.TorchOp],
) -> dict[_registration.TorchOp, Callable]:
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of PyTorch's
    built-in aten-to-aten decomposition.

    Args:
        onnx_registered_ops: All ops that have an ONNX decomposition implemented.

    Returns:
        Dict[torch._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
    decomposition_table: dict[_registration.TorchOp, Callable] = {}

    for op_overload, decomp_fn in itertools.chain(
        torch.export.default_decompositions().items(),  # type: ignore[attr-defined]
        torch._decomp.decomposition_table.items(),  # type: ignore[attr-defined]
    ):
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX
        # symbolic function.
        # NOTE: Do not skip torch._refs decomps. They are fine because otherwise the model is
        # not exportable anyways.
        if op_overload in onnx_registered_ops:
            continue
        # If it is HOP, we filter those out as well.
        if not hasattr(op_overload, "_schema"):
            continue
        # NOTE: torch._decomp.decomposition_table covers more ops
        # than torch.export.default_decompositions, but the latter is
        # more critical to torch.onnx.export.
        if op_overload in decomposition_table:
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table
<|endoftext|>"""Module for handling ATen to ONNX functions registration.

https://github.com/pytorch/pytorch/blob/6aa5bb1a76dee8112f1a9e7c194c790b5cdc6462/torch/onnx/_internal/fx/registration.py
"""

# NOTE: Why do we need a different registry than the one in torchlib?
# The registry in torchlib is used to register functions that are already implemented in
# torchlib, and is designed to be a static singleton. It does not take into account custom ops or different
# opsets etc. The registry implemented for the exporter is designed to be modifiable at
# export time by users, and is designed with dispatching in mind.

# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import importlib.util
import logging
import math
import operator
import types
from collections.abc import Callable
from typing import Literal, TypeAlias

import torch
import torch._ops
from torch.onnx._internal._lazy_import import onnx_ir as ir, onnxscript, onnxscript_apis
from torch.onnx._internal.exporter import _constants, _schemas
from torch.onnx._internal.exporter._torchlib import _torchlib_registry


TorchOp: TypeAlias = torch._ops.OpOverload | types.BuiltinFunctionType | Callable

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OnnxDecompMeta:
    """A wrapper of onnx-script function with additional metadata.

    onnx_function: The onnx-script function from torchlib.
    fx_target: The PyTorch node callable target.
    signature: The ONNX signature of the function. When None, the signature is inferred.
    is_custom: Whether the function is a custom function.
    is_complex: Whether the function is a function that handles complex valued inputs.
    opset_introduced:
        The ONNX opset version in which the function was introduced.
        Its specifies the minimum ONNX opset version required to use the function.
    device: The device the function is registered to. If None, it is registered to all devices.
    skip_signature_inference: Whether to skip signature inference for the function.
    """

    onnx_function: Callable
    fx_target: TorchOp
    signature: ir.schemas.OpSignature | None
    is_custom: bool = False
    is_complex: bool = False
    opset_introduced: int = 18
    device: Literal["cuda", "cpu"] | str | None = None  # noqa: PYI051
    skip_signature_inference: bool = False

    def __post_init__(self) -> None:
        if self.signature is None and not self.skip_signature_inference:
            try:
                if isinstance(self.onnx_function, onnxscript.OnnxFunction):
                    signature = _schemas.op_signature_from_function(
                        self.onnx_function,
                        self.onnx_function.function_ir.domain,
                        self.onnx_function.name,
                        since_version=self.onnx_function.opset.version,
                    )
                else:
                    signature = _schemas.op_signature_from_function(
                        self.onnx_function, "__traced", self.onnx_function.__name__
                    )
            except Exception as e:
                # Log a warning if the op is custom. Raise exception for builtin ops.
                if not self.is_custom:
                    raise
                else:
                    # When the function is targeting an HOP, for example, it will accept
                    # functions as arguments and fail to generate an ONNX signature.
                    # In this case we set signature to None and dispatch to this function always.
                    logger.warning(
                        "Failed to infer the signature for function '%s' because '%s'"
                        "All nodes targeting `%s` will be dispatched to this function",
                        self.onnx_function,
                        e,
                        self.fx_target,
                    )
            else:
                self.signature = signature
                self.onnx_function._pt_onnx_signature = signature  # type: ignore[attr-defined]


def _get_overload(qualified_name: str) -> torch._ops.OpOverload | None:
    """Obtain the torch op from <namespace>::<op_name>[.<overload>]"""
    # TODO(justinchuby): Handle arbitrary custom ops
    namespace, opname_overload = qualified_name.split("::")
    op_name, *maybe_overload = opname_overload.split(".", 1)
    if namespace == "_operator":
        # Builtin functions
        return getattr(operator, op_name)
    if namespace == "math":
        return getattr(math, op_name)
    if namespace == "torchvision":
        if importlib.util.find_spec("torchvision") is None:
            logger.warning("torchvision is not installed. Skipping %s", qualified_name)
            return None
    try:
        op_packet = getattr(getattr(torch.ops, namespace), op_name)
        if maybe_overload:
            overload = maybe_overload[0]
        elif "default" in op_packet._overload_names or "" in op_packet._overload_names:
            # Has a default overload
            overload = "default"
        else:
            logger.warning(
                "'%s' does not have a 'default' overload. This could be an error in specifying the op name. Ignoring.",
                qualified_name,
                stacklevel=1,
            )
            return None

        return getattr(op_packet, overload)  # type: ignore[call-overload]
    except AttributeError:
        if qualified_name.endswith("getitem"):
            # This is a special case where we registered the function incorrectly,
            # but for BC reasons (pt<=2.4) we need to keep it.
            return None
        logger.info("'%s' is not found in this version of PyTorch.", qualified_name)
        return None
    except Exception:
        logger.exception("Failed to find torch op '%s'", qualified_name)
        return None


class ONNXRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    """

    def __init__(self) -> None:
        """Initializes the registry"""
        self._opset_version = _constants.TORCHLIB_OPSET
        self.functions: dict[TorchOp | str, list[OnnxDecompMeta]] = {}

    @property
    def opset_version(self) -> int:
        """The ONNX opset version the exporter should target."""
        return self._opset_version

    @classmethod
    def from_torchlib(cls, opset_version=_constants.TORCHLIB_OPSET) -> ONNXRegistry:
        """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
        registry = cls()
        registry._opset_version = opset_version
        for meta in _torchlib_registry.get_torchlib_ops():
            registry._register(meta.fx_target, meta)

        # TODO(justinchuby): Remove this once torchlib is migrated to PyTorch
        torchlib_ops = onnxscript_apis.get_torchlib_ops()

        for torchlib_meta in torchlib_ops:
            qualified_name = torchlib_meta.qualified_name
            overload_func = torchlib_meta.function
            try:
                # NOTE: This is heavily guarded with try-except because we don't want
                # to fail the entire registry population if one function fails.
                target = _get_overload(qualified_name)
                if target is None:
                    continue

                meta = OnnxDecompMeta(
                    onnx_function=overload_func,
                    fx_target=target,
                    signature=None,
                    is_custom=False,
                    is_complex=torchlib_meta.is_complex,
                )
                registry._register(target, meta)
            except Exception:
                logger.exception("Failed to register '%s'. Skipped", qualified_name)
                continue

        registry._cleanup_registry_based_on_opset_version()
        return registry

    def _register(
        self,
        target: TorchOp,
        onnx_decomposition: OnnxDecompMeta,
    ) -> None:
        """Registers a OnnxDecompMeta to an operator.

        Args:
            target: The PyTorch node callable target.
            onnx_decomposition: The OnnxDecompMeta to register.
        """
        target_or_name: str | TorchOp
        if isinstance(target, torch._ops.OpOverload):
            # Get the qualified name of the aten op because torch._ops.OpOverload lookup in
            # a dictionary is unreliable for some reason.
            target_or_name = target.name()
        else:
            target_or_name = target
        if onnx_decomposition.is_custom:
            self.functions.setdefault(target_or_name, []).insert(0, onnx_decomposition)
        else:
            self.functions.setdefault(target_or_name, []).append(onnx_decomposition)

    def register_op(
        self,
        target: TorchOp,
        function: Callable,
        is_complex: bool = False,
    ) -> None:
        """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            target: The PyTorch node callable target.
            function: The onnx-script function to register.
            is_complex: Whether the function is a function that handles complex valued inputs.
        """
        if isinstance(target, torch._ops.OpOverloadPacket):
            raise TypeError(
                f"Target '{target}' should be provided as an OpOverload instead of an "
                "OpOverloadPacket. You can get the default overload with "
                "<op>.default"
            )

        self._register(
            target,
            OnnxDecompMeta(
                onnx_function=function,
                fx_target=target,
                signature=None,
                is_custom=True,
                is_complex=is_complex,
            ),
        )

    def get_decomps(self, target: TorchOp) -> list[OnnxDecompMeta]:
        """Returns a list of OnnxDecompMeta for the given op: torch.ops.<namespace>.<op_name>.<overload>.

        The list is ordered by the time of registration. The custom operators should come
        first in the list.

        Args:
            target: The PyTorch node callable target.
        Returns:
            A list of OnnxDecompMeta corresponding to the given name, or None if
            the name is not in the registry.
        """
        target_or_name: str | TorchOp
        if isinstance(target, torch._ops.OpOverload):
            # Get the qualified name of the aten op because torch._ops.OpOverload lookup in
            # a dictionary is unreliable for some reason.
            target_or_name = target.name()
        else:
            target_or_name = target
        decomps = self.functions.get(target_or_name, [])
        return sorted(decomps, key=lambda x: x.is_custom, reverse=True)

    def is_registered(self, target: TorchOp) -> bool:
        """Returns whether the given op is registered: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            target: The PyTorch node callable target.

        Returns:
            True if the given op is registered, otherwise False.
        """
        return bool(self.get_decomps(target))

    def _cleanup_registry_based_on_opset_version(self) -> None:
        """Pick the implementation with the highest opset version valid until the current opset version."""
        cleaned_functions = {}
        for target_or_name, decomps in self.functions.items():
            # Filter decompositions to only include those with opset_introduced <= opset_version
            decomps = [d for d in decomps if d.opset_introduced <= self.opset_version]

            # Keep only the decomposition with the highest opset_introduced
            if decomps:
                # Find the maximum opset_introduced
                max_opset = max(d.opset_introduced for d in decomps)

                # Keep all decompositions with the maximum opset_introduced
                cleaned_functions[target_or_name] = [
                    d for d in decomps if d.opset_introduced == max_opset
                ]

        self.functions = cleaned_functions

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(functions={self.functions})"
<|endoftext|>"""Subclass of ir.Value that supports Python operators."""

# mypy: allow-untyped-defs
from __future__ import annotations

from typing import TYPE_CHECKING

from torch.onnx._internal._lazy_import import onnx_ir as ir


if TYPE_CHECKING:
    import onnxscript


class SymbolicTensor(ir.Value):
    """A subclass of ir.Value that supports Python operators."""

    def __init__(
        self,
        opset: onnxscript.values.Opset,
        name: str | None = None,
        shape: ir.Shape | None = None,
        type: ir.TypeProtocol | None = None,
        doc_string: str | None = None,
        const_value: ir.TensorProtocol | None = None,
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            type=type,
            doc_string=doc_string,
            const_value=const_value,
        )
        self._opset = opset

    @property
    def rank(self) -> int | None:
        if self.shape is None:
            return None
        return len(self.shape)

    # TODO: Implement indexing

    def __mod__(self, other):
        if self.dtype in {
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
            ir.DataType.FLOAT16,
            ir.DataType.BFLOAT16,
        }:
            return self._opset.Mod(self, other, fmod=1)
        return self._opset.Mod(self, other)

    def __ne__(self, other):
        return self._opset.Not(self._opset.Equal(self, other))

    def __neg__(self):
        return self._opset.Neg(self)

    def __add__(self, other):
        return self._opset.Add(self, other)

    def __radd__(self, other):
        return self._opset.Add(other, self)

    def __rand__(self, other):
        return self._opset.And(other, self)

    def __mul__(self, other):
        return self._opset.Mul(self, other)

    def __rmul__(self, other):
        return self._opset.Mul(other, self)

    def __matmul__(self, other):
        return self._opset.MatMul(self, other)

    def __pow__(self, other):
        return self._opset.Pow(self, other)

    def __sub__(self, other):
        return self._opset.Sub(self, other)

    def __rsub__(self, other):
        return self._opset.Sub(other, self)

    def __truediv__(self, other):
        return self._opset.Div(self, other)

    def __lt__(self, other):
        return self._opset.Less(self, other)

    def __le__(self, other):
        return self._opset.LessOrEqual(self, other)

    def __ge__(self, other):
        return self._opset.GreaterOrEqual(self, other)

    def __gt__(self, other):
        return self._opset.Greater(self, other)
<|endoftext|>from __future__ import annotations


__all__ = ["InputObserver"]

import contextlib
import inspect
import time
from typing import Any, TYPE_CHECKING

import torch
from torch.onnx._internal.exporter import _onnx_program


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import onnxruntime as ort


def _flatten_unflatten_for_dynamic_shapes(
    obj: Any,
    change_function: Callable[[torch.Tensor], Any] | None = None,
) -> Any:
    """Returns the object in a different structure similar to what
    the definition of the dynamic shapes should use.

    Args:
        obj: Object from a custom class.
        change_function: If not None, this function is called to modify the tensors
            in the structure itself, like replace them by a shape.

    Returns:
        The flattened object.
    """
    if isinstance(obj, torch.Tensor):
        return change_function(obj) if change_function else obj
    flat, spec = torch.utils._pytree.tree_flatten(obj)
    start = 0
    end = 0
    subtrees = []
    for subspec in (
        spec.children() if hasattr(spec, "children") else spec.children_specs
    ):
        end += subspec.num_leaves
        value = subspec.unflatten(flat[start:end])
        value = _flatten_unflatten_for_dynamic_shapes(
            value, change_function=change_function
        )
        subtrees.append(value)
        start = end
    if spec.type is dict:
        # This is a dictionary.
        return dict(zip(spec.context, subtrees))
    if spec.type is tuple:
        return tuple(subtrees)
    if spec.type is list:
        return list(subtrees)
    if spec.type is None and not subtrees:
        return None
    if spec.context:
        # This is a custom class with attributes.
        # It is returned as a list.
        return list(subtrees)
    raise ValueError(
        f"Unable to interpret spec type {spec.type} "
        f"(type is {type(spec.type)}, context is {spec.context}), "
        f"spec={spec}, subtrees={subtrees}"
    )


def _infer_dynamic_dimensions(
    shape_list: Sequence[tuple[int, ...]], set_batch_dimension: bool = False
) -> list[int]:
    """Returns the list of dynamic dimensions given a list of shapes
    corresponding to the same tensor.

    Args:
        shape_list:
            List of shapes, they must all have the same length.
        set_batch_dimension:
            Forces the first dimension to be treated as dynamic,
            even if all shapes have the same value for that dimension.

    Returns:
        list of dynamic dimensions
    """
    unique_ranks = {len(shape) for shape in shape_list}
    torch._check(
        len(unique_ranks) == 1,
        lambda: f"All shapes in shape_list must have the same rank but {shape_list=}.",
    )
    rank = unique_ranks.pop()
    dynamic = []
    for i in range(rank):
        dims = [shape[i] for shape in shape_list]
        if len(set(dims)) > 1 or (i == 0 and set_batch_dimension):
            dynamic.append(i)
    return dynamic


class InputCandidate:
    """Retains one set of inputs given to the forward method or any
    other method the class :class:`InputObserver` is stealing from.
    Any class is allowed as long as it can be flattened.

    Args:
        args: Positional arguments.
        kwargs: Optional arguments.
        clone: Clone the inputs before storing them. Some tensors
            may be modified inplace, the original value must be retained.
        cst_kwargs: Any optional arguments constant over multiple calls.
            int, float, str, bool values must be stored here.

    The constructor flattens the received arguments.
    Any necessary flattening function should have been registered first.
    """

    def __init__(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        clone: bool,
        cst_kwargs: dict[str, int | str | float | bool],
    ):
        self.args = args
        self.kwargs = kwargs
        self.flat_list, self.spec = torch.utils._pytree.tree_flatten((args, kwargs))
        self._position_to_args_kwargs: list[int | str] | None = None
        self._n_tensors_for_args_kwargs: dict[int | str, int] | None = None
        self.cst_kwargs = cst_kwargs.copy()

        if clone:
            self.flat_list = [
                (None if not isinstance(t, torch.Tensor) else t.clone().detach())
                for t in self.flat_list
            ]
            self.args, self.kwargs = torch.utils._pytree.tree_unflatten(
                self.flat_list, self.spec
            )

        self.aligned_spec: torch.utils._pytree.PyTreeSpec | None = None
        self.aligned_flat_list: list[torch.Tensor | None] | None = None

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self.args)} args, "
            f"{len(self.kwargs)} kwargs, {len(self.flat_list)} tensors, "
            f"{len(self.aligned_flat_list or [])} aligned tensors)"
        )

    def __len__(self) -> int:
        """Returns the number of flattened tensors, None tensors are included."""
        return len(self.flat_list)

    def str_obs(self) -> str:
        """Prints out some information about the observations."""
        return (
            f"InputCandidate(args=#{len(self.args)}(...), "
            f"kwargs=#{len(self.kwargs)}{{...}}, cst_kwargs={self.cst_kwargs})"
        )

    def build_mappings(self) -> list[int | str]:
        if self._position_to_args_kwargs is not None:
            return self._position_to_args_kwargs
        self._n_tensors_for_args_kwargs = {}

        flat_index_to_args: list[int | str] = []
        for index_args, a in enumerate(self.args):
            size = len(torch.utils._pytree.tree_flatten(a)[0])
            self._n_tensors_for_args_kwargs[index_args] = size
            flat_index_to_args.extend([index_args] * size)
        for k, v in self.kwargs.items():
            size = len(torch.utils._pytree.tree_flatten(v)[0])
            self._n_tensors_for_args_kwargs[k] = size
            flat_index_to_args.extend([k] * size)

        self._position_to_args_kwargs = flat_index_to_args
        return self._position_to_args_kwargs

    @property
    def position_to_args_kwargs(self) -> list[int | str]:
        """Returns the corresponding args or kwargs
        for every tensor in the flattened inputs.
        """
        if self._position_to_args_kwargs is None:
            self.build_mappings()
        # pyrefly: ignore [bad-return]
        return self._position_to_args_kwargs

    @property
    def n_tensors_for_args_kwargs(self) -> dict[int | str, int]:
        """Returns the number of flat tensors in every args or kwargs."""
        if self._n_tensors_for_args_kwargs is None:
            self.build_mappings()
        # pyrefly: ignore [bad-return]
        return self._n_tensors_for_args_kwargs

    def _set_aligned_flat_list(
        self,
        aligned_flat_list: list[torch.Tensor | None],
        aligned_spec: torch.utils._pytree.PyTreeSpec,
    ):
        self.aligned_flat_list = aligned_flat_list
        self.aligned_spec = aligned_spec

    def align_with(
        self,
        best_candidate: InputCandidate,
        captured_inputs: dict[int | str, int],
        signature_names: list[str],
    ):
        """Two candidates are considered as aligned if after being flattened
        if they have the same number of tensors (None allowed)."""
        if self.cst_kwargs != best_candidate.cst_kwargs:
            raise RuntimeError(
                f"Two calls were made with different constant values, "
                f"{self.cst_kwargs} != {best_candidate.cst_kwargs}"
            )

        args = self.args
        if len(self.args) > len(best_candidate.args):
            # We need to move some args to kwargs as the best_candidate does.
            new_kwargs = {}
            for i in range(len(best_candidate.args), len(self.args)):
                new_kwargs[signature_names[i]] = args[i]
            args = args[: len(best_candidate.args)]
            kwargs = {**new_kwargs, **self.kwargs}
        else:
            kwargs = self.kwargs

        flat = []
        for i in range(len(best_candidate.args)):
            if i < len(args) and (isinstance(args[i], torch.Tensor) or args[i]):
                ts = torch.utils._pytree.tree_flatten(self.args[i])[0]
                if i in captured_inputs and captured_inputs[i] != len(ts):
                    raise RuntimeError(
                        f"Positional argument {i} has {len(ts)} tensors "
                        f"but previously got {captured_inputs[i]} tensors. "
                        f"Inference is impossible in that case."
                    )
                captured_inputs[i] = len(ts)
                flat.extend(ts)
                continue
            # If the argument i is not specified or is None or an empty container.
            flat.extend(
                [None for _ in range(best_candidate.n_tensors_for_args_kwargs[i])]
            )

        for k in best_candidate.kwargs:
            if k in kwargs and (isinstance(kwargs[k], torch.Tensor) or kwargs[k]):
                ts = torch.utils._pytree.tree_flatten(kwargs[k])[0]
                if k in captured_inputs and captured_inputs[k] != len(ts):
                    raise RuntimeError(
                        f"Named argument {k!r} has {len(ts)} tensors "
                        f"but previously got {captured_inputs[k]} tensors in "
                        f"kwargs={list(kwargs)}. "
                        f"Inference is impossible in that case."
                    )
                captured_inputs[k] = len(ts)
                flat.extend(ts)
                continue
            # If the argument k is not specified or is None or an empty container.
            flat.extend(
                [None for _ in range(best_candidate.n_tensors_for_args_kwargs[k])]
            )

        self._set_aligned_flat_list(flat, best_candidate.spec)

    @property
    def n_aligned_tensors(self) -> int:
        if self.aligned_flat_list is None:
            raise RuntimeError("This input was not aligned with the others.")
        return len(self.aligned_flat_list)


class InputObserverInfo:
    """Contains all the necessary information to infer dynamic shapes
    and the arguments to send to :func:`torch.export.export`.

    Args:
        signature_names: Names of the arguments of the method
            the collector tensors come from. They are used if it becomes
            necessary to move positional arguments to named ones.
            They are used a second time because :func:`torch.export.export`
            cares about the order in kwargs and dynamic shapes, it needs
            to be the same in the ordered dictionaries `add_inputs` receive.
        default_values: Default values defined by the signature of the function,
            any value equal to that is ignored to simplify the export.
        value_if_missing: If an argument is missing,
            a default value will be taken in this dictionary,
            this is used when after the prefill step, an argument
            disappears (such as `pixel_values`) and another one
            is added (such as `past_key_values`).
            The values are only to infer dynamic shapes and arguments,
            not to run the model.
        args_name_and_position: Name of parameter `*args`
            and its position if it exists.
        kwargs_name: Name of the variable keyword parameter `**kwargs` if it exists.

    This is used by class :class:`InputObserver`.
    """

    def __init__(
        self,
        signature_names: list[str],
        default_values: dict[str, int | bool | str | float],
        value_if_missing: dict[str | int, Any],
        args_name_and_position: tuple[str, int] | None,
        kwargs_name: str | None,
    ):
        self.default_values = default_values
        self.value_if_missing = value_if_missing
        self.inputs: list[InputCandidate] = []
        self.outputs_specs: list[torch.utils._pytree.PyTreeSpec] = []
        self.flat_outputs: list[list[torch.Tensor | None]] = []
        self.latencies: list[float] = []
        self.args_name_and_position = args_name_and_position
        self.kwargs_name = kwargs_name
        self.signature_names = signature_names
        self._best_candidate: InputCandidate | None = None
        self._captured_inputs: dict[int | str, int] | None = None

    def __len__(self) -> int:
        """Returns the number of collected set of inputs/outputs."""
        return len(self.inputs)

    def add_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        """Stores one set of inputs. They are deepcopied.

        Args:
            args: Positional arguments.
            kwargs: Named arguments.
        """
        cst_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self.signature_names
            and isinstance(v, (int, float, bool, str))
            and v != self.default_values.get(k, None)
            and self.default_values.get(k, None) is not None
        }
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if v is not None and not isinstance(v, (int, float, bool, str))
        }

        # adds value_if_missing attributes
        for k, v in self.value_if_missing.items():
            if isinstance(k, str):
                if k not in kwargs:
                    # Validate that `value_if_missing` keys are compatible
                    # with the observed signature.
                    # If the function does not accept **kwargs,
                    # all value_if_missing keys must be
                    # present in the observed signature names.
                    if k not in self.signature_names and not self.kwargs_name:
                        raise ValueError(
                            f"Unexpected keyword argument {k!r} "
                            f"provided as a value_if_missing input "
                            "for a function that does not accept it. "
                            f"All value_if_missing keys must "
                            f"be in the observed signature: {tuple(self.signature_names)}."
                        )
                    kwargs[k] = v
            elif isinstance(k, int):
                if k >= len(self.signature_names):
                    raise ValueError(
                        f"Unexpected keyword argument {k=} "
                        f"provided as a value_if_missing input "
                        "for a function that does not accept it. "
                        f"All value_if_missing indices must "
                        f"be in the observed signature: {tuple(self.signature_names)}."
                    )
                if k >= len(args):
                    raise NotImplementedError(
                        f"Unexpected keyword argument {k=} "
                        f"provided as a value_if_missing input "
                        "for a function that does not accept it. "
                        f"All value_if_missing indices must "
                        f"be in the observed signature: {tuple(self.signature_names)}, "
                        f"only {len(args)} were given."
                    )
                list_args = list(args)
                list_args[k] = v
                args = tuple(list_args)
            else:
                raise TypeError(
                    f"Unexpected type {type(k)} for a missing value. The key is {k!r}."
                )

        # kwargs may come in a different order each time.
        # dictionaries are ordered and torch.export.export expects
        # dynamic shapes and kwargs to follow the same order.

        ordered_kwargs = {k: kwargs[k] for k in self.signature_names if k in kwargs}
        for k, v in kwargs.items():
            if k not in ordered_kwargs:
                ordered_kwargs[k] = v

        candidate = InputCandidate(
            args, ordered_kwargs, clone=True, cst_kwargs=cst_kwargs
        )
        self.inputs.append(candidate)
        if self._best_candidate is None or len(self._best_candidate) < len(candidate):
            self._best_candidate = candidate

    def add_outputs(self, res: torch.Tensor | tuple[torch.Tensor, ...], latency: float):
        """Stores outputs. They are deepcopied."""
        flat_res, spec = torch.utils._pytree.tree_flatten(res)
        self.outputs_specs.append(spec)
        self.flat_outputs.append(
            [(None if t is None else t.clone().detach()) for t in flat_res]
        )
        self.latencies.append(latency)

    def align_inputs_none_values(self):
        """Once the best candidate is chosen, this method aligns every set of inputs
        on the best candidate, it inserts None at the right position when
        optional inputs are not specified. We consider a set of inputs is aligned
        if this method does not change the original flattened inputs.
        """
        if not self.inputs or self._best_candidate is None:
            raise RuntimeError("No inputs were captured.")

        if all(candidate.aligned_flat_list is not None for candidate in self.inputs):
            # No new inputs, no alignment is necessary.
            return

        # Let's reprocess everything.
        self._captured_inputs = {}
        for candidate in self.inputs:
            if len(set(candidate.kwargs) | set(self._best_candidate.kwargs)) > len(
                self._best_candidate.kwargs
            ):
                raise RuntimeError(
                    f"At least one call to the observed model "
                    f"must contain all the named arguments. "
                    f"candidate kwargs={list(candidate.kwargs)}, "
                    f"best candidate kwargs={list(self._best_candidate.kwargs)}."
                )
            candidate.align_with(
                self._best_candidate, self._captured_inputs, self.signature_names
            )

    def infer_dynamic_shapes(
        self,
        set_batch_dimension_for: set[int | str] | bool | None = None,
        return_flat: bool = False,
    ) -> tuple[dict[int, Any] | None, ...] | dict[str, dict[int, Any] | None]:
        """Infers dynamic shapes based on the collected tensors.
        Most of the time, models do support a batch dimension
        but this batch dimension has the same value for every input sample.
        Instead of running inference on new samples, argument `set_batch_dimension_for`
        can be used to tell the first dimension is a dynamic dimension for a particular
        set of inputs referenced by their name (str) or their position (int).

        Args:
            set_batch_dimension_for (set[int | str] | bool | None): Set of input identifiers,
                by name (``str``) or position (``int``), for which the first dimension
                should be treated as a dynamic batch dimension. If ``None`` or empty,
                no additional batch dimensions are marked as dynamic.
            return_flat: Tells the function to return a flat tuple instead of
                nested structured. This option is used internally to infer arguments.
        """
        self.align_inputs_none_values()
        if self._best_candidate is None:
            raise AssertionError("expected self._best_candidate to be not None")
        if self._best_candidate.flat_list is None:
            raise AssertionError(
                "expected self._best_candidate.flat_list to be not None"
            )
        if self._best_candidate.aligned_flat_list is None:
            raise AssertionError(
                "expected self._best_candidate.aligned_flat_list to be not None"
            )

        def _set_batch_dimension(name_or_position) -> bool:
            if not set_batch_dimension_for:
                return False
            if (
                isinstance(set_batch_dimension_for, bool) and set_batch_dimension_for
            ) or name_or_position in set_batch_dimension_for:
                return True
            if isinstance(name_or_position, int):
                torch._check(
                    name_or_position < len(self.signature_names),
                    lambda: f"argument at position {name_or_position} is out of boundary",
                )
                if self.signature_names[name_or_position] in set_batch_dimension_for:
                    return True
            return False

        def _set_batch_dimension_for_flat_index(index) -> bool:
            return _set_batch_dimension(
                # pyrefly: ignore[missing-attribute]
                self._best_candidate.position_to_args_kwargs[index]
            )

        if len(self._best_candidate.flat_list) != len(
            self._best_candidate.aligned_flat_list
        ):
            raise NotImplementedError(
                "infer_dynamic_shapes is not implemented "
                "when the best candidate is not 'aligned'. "
                "This happens when there is no stored set of inputs where "
                "all optional inputs showing in other sets are defined."
            )

        if len({inputs.n_aligned_tensors for inputs in self.inputs}) != 1:
            raise NotImplementedError(
                f"infer_dynamic_shapes is not implemented "
                f"when the number of input tensors are not the same in "
                f"every set of inputs "
                f"{[inputs.n_aligned_tensors for inputs in self.inputs]}."
            )
        shape_lists = [
            [(None if t is None else t.shape) for t in candidate.aligned_flat_list]
            for candidate in self.inputs
            if candidate.aligned_flat_list is not None
        ]
        n_tensors = len(shape_lists[0])
        dynamic_shapes = [
            _infer_dynamic_dimensions(
                [s for s in [shapes[index] for shapes in shape_lists] if s is not None],
                set_batch_dimension=_set_batch_dimension_for_flat_index(index),
            )
            for index in range(n_tensors)
        ]
        cst = torch.export.Dim.DYNAMIC
        flat_dynamic_shapes = [dict.fromkeys(dims, cst) for dims in dynamic_shapes]
        if return_flat:
            return tuple(flat_dynamic_shapes)

        # Let's regroup.
        if len(flat_dynamic_shapes) == len(self._best_candidate.args) + len(
            self._best_candidate.kwargs
        ):
            # It means forward method is called with tensors only.
            if (
                not self._best_candidate.kwargs
                and not self._best_candidate.cst_kwargs
                and not self.args_name_and_position
            ):
                # only positional arguments
                return tuple(flat_dynamic_shapes)
            if not self._best_candidate.args:
                # only named arguments
                ds = dict(zip(list(self._best_candidate.kwargs), flat_dynamic_shapes))
                return self._post_process_for_kwargs(
                    {**ds, **dict.fromkeys(self._best_candidate.cst_kwargs, None)}
                )
            if not self.args_name_and_position:
                # positional arguments needs to be moved to the named arguments
                n_args = len(self._best_candidate.args)
                pos_names = self.signature_names[:n_args]
                return self._post_process_for_kwargs(
                    {
                        **dict(zip(pos_names, flat_dynamic_shapes[:n_args])),
                        **dict(
                            zip(
                                list(self._best_candidate.kwargs),
                                flat_dynamic_shapes[n_args:],
                            )
                        ),
                        **dict.fromkeys(self._best_candidate.cst_kwargs, None),
                    }
                )
            # positional arguments needs to be moved to the named arguments
            n_args = min(len(self._best_candidate.args), self.args_name_and_position[1])
            i_kwargs = max(
                len(self._best_candidate.args), self.args_name_and_position[1]
            )
            var_pos = self.args_name_and_position[0]
            pos_names = self.signature_names[:n_args]
            return self._post_process_for_kwargs(
                {
                    **dict(zip(pos_names, flat_dynamic_shapes[:n_args])),
                    var_pos: tuple(flat_dynamic_shapes[n_args:i_kwargs]),
                    **dict(
                        zip(
                            list(self._best_candidate.kwargs),
                            flat_dynamic_shapes[i_kwargs:],
                        )
                    ),
                    **dict.fromkeys(self._best_candidate.cst_kwargs, None),
                }
            )

        # nested types, here comes the fun part because the shapes cannot be unflattened,
        # custom classes must appear in their flattened shape.
        # This does not work in all cases but every time every available argument is flattened
        # with the same number of tensors. The function does not check
        # if that assumption is true.
        flat_inputs, _max_spec = torch.utils._pytree.tree_flatten(
            (self._best_candidate.args, self._best_candidate.kwargs)
        )
        torch._check(
            len(flat_inputs) == len(flat_dynamic_shapes),
            (
                # pyrefly: ignore [bad-argument-type]
                f"Length mismatch len(flat_inputs)={len(flat_inputs)}, "
                f"len(flat_dynamic_shapes)={len(flat_dynamic_shapes)}"
            ),
        )

        index = 0

        def change_function(t):
            nonlocal index
            if index >= len(flat_dynamic_shapes):
                raise RuntimeError(
                    f"Flattened {index} tensors when there are only "
                    f"{len(flat_dynamic_shapes)}."
                )
            res = flat_dynamic_shapes[index]
            index += 1
            return res

        ds_args, ds_kwargs = _flatten_unflatten_for_dynamic_shapes(
            (self._best_candidate.args, self._best_candidate.kwargs),
            change_function=change_function,
        )
        if self._best_candidate.cst_kwargs:
            ds_kwargs = {
                **ds_kwargs,
                **dict.fromkeys(self._best_candidate.cst_kwargs, None),
            }
        if not ds_kwargs and not self.args_name_and_position:
            return tuple(ds_args)
        if not ds_args:
            return self._post_process_for_kwargs(ds_kwargs)

        if not self.args_name_and_position:
            pos_names = self.signature_names[: len(ds_args)]
            return self._post_process_for_kwargs(
                {**dict(zip(pos_names, ds_args)), **ds_kwargs}
            )

        n_args = min(len(ds_args), self.args_name_and_position[1])
        pos_names = self.signature_names[:n_args]
        return self._post_process_for_kwargs(
            {
                **dict(zip(pos_names, ds_args[:n_args])),
                self.args_name_and_position[0]: tuple(ds_args[n_args:]),
                **ds_kwargs,
            }
        )

    def infer_arguments(
        self,
        index_or_candidate: InputCandidate | int | None = None,
        /,
        flat: bool = False,
        as_args_kwargs: bool = False,
    ) -> (
        list[torch.Tensor | None]
        | tuple[torch.Tensor, ...]
        | dict[str, torch.Tensor]
        | tuple[list[torch.Tensor] | tuple[torch.Tensor, ...], dict[str, torch.Tensor]]
    ):
        """Infers arguments based on the collected tensors.

        Args:
            index_or_candidate: If missing, the method selects one set of inputs
                among the available ones, usually the set of inputs containing
                with the highest number of tensors.
                It then replaces None values and missing tensors with empty tensors.
                If not missing, it can be an integer to fetch one of the stored set
                or some inputs.
            flat: If True, it returns a flattened list of tensors,
                if False, it returns a tuple or a dictionary preserving
                the nested structures. The flat version is used internally.
                It produces a single list of tensors easier to process or modify
                rather than a nested structure holding the same tensors.
                The original structure can be restored with
                ``torch.utils._pytree.tree_unflatten(flat_list, self.aligned_spec)``.
                This mechanism is used to replace None values by empty tensors.
            as_args_kwargs: If True, the method always returns `(args, kwargs)`,
                otherwise, it returns either a tuple (only args) or a dictionary
                (only kwargs) or raises an exception if it cannot do so.
        Returns:
            Inferred arguments, every optional tensor is replaced by an empty tensor.
        """
        # This is already checked by _build_inputs_completed_with_none_values
        # but this is not always well captured by tools checking types.
        self.align_inputs_none_values()
        if self._best_candidate is None:
            raise AssertionError("expected self._best_candidate to be not None")
        candidate = None
        if index_or_candidate is None:
            for cand in self.inputs:
                args, kwargs = cand.args, cand.kwargs
                if len(args) == len(self._best_candidate.args or ()) and len(
                    kwargs
                ) == len(self._best_candidate.kwargs or {}):
                    candidate = cand
                    break
        elif isinstance(index_or_candidate, int):
            torch._check(
                index_or_candidate < len(self.inputs),
                lambda: (
                    f"No stored input set for index="
                    f"{index_or_candidate}<{len(self.inputs)}."
                ),
            )
            candidate = self.inputs[index_or_candidate]
        else:
            candidate = index_or_candidate

        if candidate is None:
            raise AssertionError("expected candidate to be not None")
        if candidate.aligned_flat_list is None:
            raise RuntimeError(
                f"Candidate {candidate} has no aligned flat list of tensors, "
                f"index_or_candidate={index_or_candidate}. You should call "
                f"method 'align_with'."
            )

        aligned_flat_list = candidate.aligned_flat_list
        if aligned_flat_list is None:
            raise AssertionError("expected aligned_flat_list to be not None")
        if any(t is None for t in aligned_flat_list):
            dynamic_shapes = self.infer_dynamic_shapes(return_flat=True)
            if not isinstance(dynamic_shapes, tuple):
                raise AssertionError(f"expected tuple, got {type(dynamic_shapes)}")
            aligned_flat_list = list(aligned_flat_list)
            for index in range(len(aligned_flat_list)):
                if aligned_flat_list[index] is not None:
                    continue
                shape = dynamic_shapes[index]
                all_non_empty_tensors = [
                    c.aligned_flat_list[index]
                    for c in self.inputs
                    if c.aligned_flat_list is not None
                ]
                all_non_empty_tensors_not_none = [
                    t for t in all_non_empty_tensors if t is not None
                ]
                if not all_non_empty_tensors_not_none:
                    raise RuntimeError(
                        f"There is no tensor at position {index} in any flattened inputs."
                    )
                tensor = all_non_empty_tensors_not_none.pop()
                if tensor.numel() == 0:
                    aligned_flat_list[index] = tensor
                    continue
                if not shape:
                    aligned_flat_list[index] = torch.zeros(
                        tensor.shape, dtype=tensor.dtype, device=tensor.device
                    )
                    continue
                dim = max(shape)
                torch._check(
                    dim < tensor.ndim,
                    lambda index=index, shape=shape, tshape=tensor.shape: (
                        f"Tensor shape {tshape} does not match the "
                        f"dynamic shape {shape} at position {index}."
                    ),
                )
                new_shape = list(tensor.shape)
                new_shape[dim] = 0
                aligned_flat_list[index] = torch.empty(
                    tuple(new_shape), dtype=tensor.dtype, device=tensor.device
                )
        if flat:
            return aligned_flat_list
        args, kwargs = torch.utils._pytree.tree_unflatten(
            aligned_flat_list,
            # pyrefly: ignore[bad-argument-type]
            candidate.aligned_spec,
        )
        if self._best_candidate.cst_kwargs:
            kwargs = {**kwargs, **self._best_candidate.cst_kwargs}

        if not as_args_kwargs:
            if not kwargs:
                return args
            if not args:
                return kwargs

            # We need to move args to kwargs
            if self.args_name_and_position:
                raise RuntimeError(
                    "Cannot return arguments "
                    "as a single tuple or a single dictionary "
                    "because of '*args' in the function signature. "
                    "You need to set `as_args_kwargs=True`."
                )
            n_args = len(args)
            pos_names = self.signature_names[:n_args]
            return {**dict(zip(pos_names, args[:n_args])), **kwargs}

        # Generic case.
        return tuple(args), kwargs

    def _post_process_for_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """:func:`torch.export.export` requires dynamic shapes and keyword arguments
        that are not part of the explicit function signature but are collected via
        ``**<kwargs_name>`` to be wrapped under the corresponding parameter name
        (``self.kwargs_name``) as ``{<kwargs_name>: {'param': shape or tensor}}``.
        This function ensures this wrapping is performed when ``self.kwargs_name`` is set.
        """
        if not self.kwargs_name:
            # Nothing to do here.
            return kwargs
        to_be_moved = {k for k in kwargs if k not in self.signature_names}
        if not to_be_moved:
            return kwargs
        keywords = {k: v for k, v in kwargs.items() if k in to_be_moved}
        new_kwargs = {k: v for k, v in kwargs.items() if k not in to_be_moved}
        if self.kwargs_name in new_kwargs:
            raise ValueError(
                f"Keyword argument name collision: received a keyword argument "
                f"'{self.kwargs_name}' which conflicts with the **{self.kwargs_name} "
                "parameter used to collect extra keyword arguments. "
                "Passing a keyword argument with this name is not supported."
            )
        return {**new_kwargs, self.kwargs_name: keywords}


class InputObserver:
    """Steals forward method to collect inputs and outputs.
    This information is used to infer dynamic shapes and
    export arguments.

    Args:
        value_if_missing: If an argument is missing,
            a default value will be taken in this dictionary,
            this is used when after the prefill step, an argument
            disappears (such as `pixel_values`) and another one
            is added (such as `past_key_values`).
            The values are only to infer dynamic shapes and arguments,
            not to run the model.

    Examples
    --------
    >>> input_observer = InputObserver()
    >>> with input_observer(model):
    >>>     model(x1, y1)
    >>>     model(x2, y2)
    >>> ep = torch.export.export(  # or torch.onnx.export
    >>>     model,
    >>>     input_observer.infer_arguments(),
    >>>     dynamic_shapes.input_observer.infer_dynamic_shapes(),
    >>> )

    With LLM:

    >>> input_observer = InputObserver()
    >>> with input_observer(model):
    >>>     model.generate(input_ids)
    >>> ep = torch.export.export(  # or torch.onnx.export
    >>>     model,
    >>>     (),
    >>>     kwargs=input_observer.infer_arguments(),
    >>>     dynamic_shapes.input_observer.infer_dynamic_shapes(),
    >>> )

    The last example considers an LLM taking images and text as inputs.
    The first call to the forward method which we try to export has `pixel_values`
    but no `past_key_values`. The next calls do not have `pixel_values` but
    `past_key_values`. The observer understands `pixel_values` and `past_key_values`
    are needed but they may not be both specified at the same time.
    Since `pixel_values` only appears in the first call, the observer cannot
    tell how to infer an empty tensor for this argument. That's what the argument
    `value_if_missing` is for. The following example is more than a dummy example
    but shows how to use it with ``transformers``.

    .. code-block:: python

        from transformers import pipeline

        model_id = "tiny-random/gemma-3"
        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            device="cpu",
            trust_remote_code=True,
            max_new_tokens=3,
            dtype=torch.float16,
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                    },
                    {"type": "text", "text": "What animal is on the candy?"},
                ],
            },
        ]
        observer = InputObserver(
            value_if_missing=dict(
                pixel_values=torch.empty((0, 3, 896, 896), dtype=torch.float16)
            )
        )
        with observer(pipe.model):
            pipe(text=messages, max_new_tokens=4)

    .. versionadded:: 2.11.0
    """

    def __init__(self, value_if_missing: dict[str | int, Any] | None = None):
        self.info: InputObserverInfo | None = None
        self.value_if_missing = value_if_missing or {}

    def _replaced_method(
        self,
        *args,
        _captured_method: Callable | None = None,
        _store_n_calls: int = 3,
        **kwargs,
    ):
        if _captured_method is None:
            raise AssertionError("_captured_forward cannot be None")
        if self.info is None:
            raise AssertionError("info cannot be None")
        n_stored = len(self.info)
        if n_stored < _store_n_calls:
            self.info.add_inputs(args, kwargs)
        begin = time.perf_counter()
        res = _captured_method(*args, **kwargs)
        duration = time.perf_counter() - begin
        if n_stored < _store_n_calls:
            self.info.add_outputs(res, latency=duration)
        return res

    def num_obs(self) -> int:
        """Returns the number of stored set of inputs."""
        return 0 if not self.info else len(self.info)

    @contextlib.contextmanager
    def __call__(
        self,
        model: torch.nn.Module,
        store_n_calls: int = 3,
        method_name: str = "forward",
    ):
        """Starts collecting inputs and outputs of a specific method.
        The model method is replaced by a new one collecting tensors
        before and after the inner one is called.
        The original method is restored after the collection.

        Args:
            model: Model
            store_n_calls: The collection stops after this many calls
                to avoid taking too much memory.
            method_name: Method name to spy on.
        """
        if not hasattr(model, method_name):
            raise ValueError(
                f"Model type {model} does not have a method {method_name!r}."
            )
        captured_method = getattr(model, method_name)
        sig = inspect.signature(captured_method)
        if self.info is None:
            kwargs_names = [
                p
                for p in sig.parameters
                if sig.parameters[p].kind == inspect.Parameter.VAR_KEYWORD
            ]
            args_names = [
                (p, i)
                for (i, p) in enumerate(sig.parameters)
                if sig.parameters[p].kind == inspect.Parameter.VAR_POSITIONAL
            ]
            self.info = InputObserverInfo(
                signature_names=list(sig.parameters),
                default_values={
                    p.name: p.default
                    for p in sig.parameters.values()
                    if p.default != inspect.Parameter.empty
                    and isinstance(p.default, (int, bool, str, float))
                },
                value_if_missing=self.value_if_missing,
                args_name_and_position=args_names[0] if args_names else None,
                kwargs_name=kwargs_names[0] if kwargs_names else None,
            )
        n_already_stored = len(self.info)
        lambda_method = lambda *args, _cm=captured_method, _snc=(  # noqa: E731
            store_n_calls + n_already_stored
        ), **kwargs: self._replaced_method(
            *args, _captured_method=_cm, _store_n_calls=_snc, **kwargs
        )

        # It may happen that the signature of the forward is used to trigger a preprocessing.
        # This is used in GenerationMixin (transformers):
        #   position_ids_key = "decoder_position_ids" if ... else "position_ids"
        #   if position_ids_key in set(inspect.signature(self.forward).parameters.keys()):
        lambda_method.__signature__ = sig  # type: ignore[attr-defined]

        setattr(model, method_name, lambda_method)

        try:
            yield self
        finally:
            setattr(model, method_name, captured_method)

    def _check_captured(self):
        if self.info is None:
            raise RuntimeError("No inputs were captured.")

    def infer_dynamic_shapes(
        self, set_batch_dimension_for: set[int | str] | bool | None = None
    ) -> tuple[dict[int, Any] | None, ...] | dict[str, dict[int, Any] | None]:
        """
        Infers dynamic shapes. Most of the time, models do support a batch dimension
        but this batch dimension has the same value for every input sample.
        Instead of running inference on new samples, argument `set_batch_dimension_for`
        can be used to tell the first dimension is a dynamic dimension for a particular
        set of inputs referenced by their name (str) or their position (int).

        Args:
            set_batch_dimension_for (set[int | str] | bool | None): A set of input
                identifiers (by position as ``int`` or by name as ``str``) for
                which the first dimension should be treated as a dynamic batch
                dimension. If ``None``, no dimensions are explicitly marked as
                dynamic.
        """
        self._check_captured()
        if self.info is None:
            raise AssertionError("info must be non-None")
        return self.info.infer_dynamic_shapes(
            set_batch_dimension_for=set_batch_dimension_for
        )

    def infer_arguments(
        self,
        index_or_args_or_kwargs: tuple[Any] | dict[str, Any] | int | None = None,
        flat: bool = False,
        as_args_kwargs: bool = False,
    ) -> (
        list[torch.Tensor | None]
        | tuple[torch.Tensor, ...]
        | dict[str, torch.Tensor]
        | tuple[list[torch.Tensor] | tuple[torch.Tensor, ...], dict[str, torch.Tensor]]
    ):
        """Infers arguments based on the collected tensors.

        Args:
            index_or_args_or_kwargs: If missing, the method selects one set of inputs
                among the available ones, usually the set of inputs containing
                with the highest number of tensors.
                It then replaces None values and missing tensors with empty tensors.
                If not missing, it can be an integer to fetch one of the stored set
                or some inputs.
            flat: If True, it returns a flattened list of tensors,
                if False, it returns a tuple or a dictionary preserving
                the nested structures. The flat version is used internally.
                It produces a single list of tensors easier to process or modify
                rather than a nested structure holding the same tensors.
                The original structure can be restored with
                ``torch.utils._pytree.tree_unflatten(flat_list, self.aligned_spec)``.
                This mechanism is used to replace None values by empty tensors.
            as_args_kwargs: If True, the method always returns `(args, kwargs)`,
                otherwise, it returns either a tuple (only args) or a dictionary
                (only kwargs) or raises an exception if it cannot do so.
        Returns:
            Inferred arguments, every optional tensor is replaced by an empty tensor.
        """
        self._check_captured()
        if self.info is None:
            raise AssertionError("expected self.info to be not None")
        index_or_candidate: int | InputCandidate | None = None
        if index_or_args_or_kwargs is None or isinstance(index_or_args_or_kwargs, int):
            index_or_candidate = index_or_args_or_kwargs
        else:
            if isinstance(index_or_args_or_kwargs, tuple):
                index_or_candidate = InputCandidate(
                    args=index_or_args_or_kwargs, kwargs={}, clone=False, cst_kwargs={}
                )
            elif isinstance(index_or_args_or_kwargs, dict):
                index_or_candidate = InputCandidate(
                    args=(),
                    kwargs={
                        k: v
                        for k, v in index_or_args_or_kwargs.items()
                        if k not in self.info.default_values
                    },
                    clone=False,
                    cst_kwargs={
                        k: v
                        for k, v in index_or_args_or_kwargs.items()
                        if k in self.info.default_values
                    },
                )
            else:
                raise ValueError(
                    f"Unexpected type {type(index_or_args_or_kwargs)} "
                    f"for index_or_args_or_kwargs."
                )
            self.info.align_inputs_none_values()
            index_or_candidate.align_with(
                # pyrefly: ignore[bad-argument-type]
                self.info._best_candidate,
                # pyrefly: ignore[bad-argument-type]
                self.info._captured_inputs,
                self.info.signature_names,
            )
        return self.info.infer_arguments(
            index_or_candidate,
            flat=flat,
            as_args_kwargs=as_args_kwargs,
        )

    def check_discrepancies(
        self,
        onnx_program: torch.onnx.ONNXProgram,
        atol: float = 1e-4,
        rtol: float = 0.1,
        progress_bar: bool = False,
        initializer: Callable[
            [str | bytes], ort.InferenceSession
        ] = _onnx_program._ort_session_initializer,
        skip_none: bool = True,
    ) -> list[dict[str, str | int | float | bool]]:
        """Computes the discrepancies between the saved inputs and outputs
        with the saved onnx model.

        Args:
            onnx_program: Exported Model to verify.
            atol: Absolute tolerance, recommended values, 1e-4 for float, 1e-2 for float16.
            rtol: Relative tolerance.
            progress_bar: Shows a progress bar (requires `tqdm`).
            initializer: The function called to initialize the ONNX Runtime inference
                session with the specified model. By default, it uses the
                `_ort_session_initializer` function.
            skip_none: Does not check discrepancies when an output is None.

        Returns:
            A list of dictionaries, ready to be consumed by a dataframe.

        The function catches exceptions, it shows the error in the returned
        summary.
        """
        # For big models, we should consider taking a filename to avoid the users
        # creating the model proto twice.
        self._check_captured()
        if self.info is None:
            raise AssertionError("expected self.info to be not None")

        onnx_program.initialize_inference_session(initializer)

        input_names = [i.name for i in onnx_program.model.graph.inputs]
        io_sets = list(
            zip(self.info.inputs, self.info.flat_outputs, self.info.latencies)
        )
        if progress_bar:
            from tqdm import tqdm

            loop = tqdm(io_sets)
        else:
            loop = io_sets
        data: list[dict[str, Any]] = []
        for inputs, outputs, latency in loop:
            if inputs.aligned_flat_list is None:
                raise AssertionError("expected inputs.aligned_flat_list to be not None")
            if len(input_names) != len(inputs.aligned_flat_list):
                raise RuntimeError(
                    f"There are ({len(inputs.aligned_flat_list)}) "
                    f"tensors but the model expects {len(input_names)}."
                )
            n_none = sum(t is None for t in inputs.aligned_flat_list)
            n_empty = sum(t is None or t.numel() == 0 for t in inputs.aligned_flat_list)

            feeds = dict(zip(input_names, self.info.infer_arguments(inputs, flat=True)))

            begin = time.perf_counter()
            try:
                ort_outputs = onnx_program(**feeds)
                error = None
            except Exception as e:
                error = str(e)
                ort_outputs = None

            duration = time.perf_counter() - begin
            if error:
                diff: dict[str, str | int | float | bool] = dict(
                    error=error, SUCCESS=False
                )
            elif ort_outputs is None or len(outputs) != len(ort_outputs):
                diff = dict(SUCCESS=False, error="not the same number of outputs")
            else:
                success = True
                err_abs = 0.0
                err_rel = 0.0
                error = ""
                for torch_tensor, ort_tensor in zip(outputs, ort_outputs):
                    if torch_tensor is None or ort_tensor is None:
                        if type(torch_tensor) is not type(ort_tensor) and not skip_none:
                            success = False
                            error = "missing output"
                            break
                        continue
                    if torch_tensor.shape != ort_tensor.shape:
                        success = False
                        error = "not the same shape"
                        break
                    if torch_tensor.dtype != ort_tensor.dtype:
                        success = False
                        error = "not the same type"
                        break
                    err = (torch_tensor - ort_tensor).abs().max().item()
                    err_abs = max(err_abs, err)
                    if err_abs > atol:
                        success = False
                    err = (
                        (
                            (torch_tensor - ort_tensor).abs()
                            / (torch_tensor.abs() + rtol)
                        )
                        .max()
                        .item()
                    )
                    err_rel = max(err_rel, err)
                    if err_rel > rtol:
                        success = False
                diff = dict(SUCCESS=success, abs=err_abs, rel=err_rel)
            diff.update(
                dict(
                    index=len(data),
                    duration_torch=latency,
                    ort_duration=duration,
                    n_inputs=len(input_names),
                    n_none=n_none,
                    n_empty=n_empty,
                )
            )
            data.append(diff)
        onnx_program.release()
        return data
<|endoftext|>"""Isolated calls to methods that may segfault."""

from __future__ import annotations

import multiprocessing
import os
import warnings
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec, TypeVarTuple, Unpack


if TYPE_CHECKING:
    from collections.abc import Callable


_P = ParamSpec("_P")
_R = TypeVar("_R")
_Ts = TypeVarTuple("_Ts")

_IS_WINDOWS = os.name == "nt"


def _call_function_and_return_exception(
    func: Callable[[Unpack[_Ts]], _R], args: tuple[Unpack[_Ts]], kwargs: dict[str, Any]
) -> _R | Exception:
    """Call function and return an exception if there is one."""

    try:
        return func(*args, **kwargs)
    except Exception as e:
        return e


def safe_call(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    """Call a function in a separate process.

    Args:
        func: The function to call.
        args: The positional arguments to pass to the function.
        kwargs: The keyword arguments to pass to the function.

    Returns:
        The return value of the function.

    Raises:
        Exception: If the function raised an exception.
    """
    if _IS_WINDOWS:
        # On Windows, we cannot create a new process with fork.
        warnings.warn(
            f"A new process is not created for {func} on Windows.", stacklevel=1
        )
        return func(*args, **kwargs)

    with multiprocessing.get_context("fork").Pool(1) as pool:
        # It is important to fork a process here to prevent the main logic from
        # running again when the user does not place it under a `if __name__ == "__main__":`
        # block.
        result = pool.apply_async(
            _call_function_and_return_exception, (func, args, kwargs)
        )
        result = result.get(timeout=5)
    if isinstance(result, Exception):
        raise result
    return result
<|endoftext|>"""Compatibility functions for the torch.onnx.export API."""

# mypy: allow-untyped-defs
# mypy: disable-error-code=attr-defined
from __future__ import annotations

import io
import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

import torch
from torch.onnx import _constants as onnx_constants
from torch.onnx._internal._lazy_import import onnx
from torch.onnx._internal.exporter import (
    _constants,
    _core,
    _dynamic_shapes,
    _exportable_module,
    _onnx_program,
    _registration,
)


if TYPE_CHECKING:
    import os

logger = logging.getLogger(__name__)


def _get_torch_export_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    """Obtain the arguments for torch.onnx.export from the model and the input arguments."""
    if not kwargs and args and isinstance(args[-1], dict):
        kwargs = args[-1]
        args = args[:-1]
    return args, kwargs


def export_compat(
    model: torch.nn.Module
    | torch.export.ExportedProgram
    | torch.jit.ScriptModule
    | torch.jit.ScriptFunction,
    args: tuple[Any, ...],
    f: str | os.PathLike | None = None,
    *,
    kwargs: dict[str, Any] | None = None,
    export_params: bool = True,
    verbose: bool | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = onnx_constants.ONNX_DEFAULT_OPSET,
    custom_translation_table: dict[Callable, Callable] | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    keep_initializers_as_inputs: bool = False,
    external_data: bool = True,
    report: bool = False,
    optimize: bool = True,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
) -> _onnx_program.ONNXProgram:
    if opset_version is None:
        opset_version = onnx_constants.ONNX_DEFAULT_OPSET

    if isinstance(model, torch.nn.Module):
        if model.training:
            warnings.warn(
                "Exporting a model while it is in training mode. "
                "Please ensure that this is intended, as it may lead to "
                "different behavior during inference. "
                "Calling model.eval() before export is recommended.",
                UserWarning,
                stacklevel=3,
            )

    if isinstance(model, _exportable_module.ExportableModule):
        # Skip argument extraction if args or kwargs are provided
        if not args and not kwargs:
            args, kwargs = model.example_arguments()
            if input_names is None:
                input_names = model.input_names()
            if output_names is None:
                output_names = model.output_names()
            if dynamic_shapes is None:
                dynamic_shapes = model.dynamic_shapes()

    if isinstance(model, torch.export.ExportedProgram):
        # We know the model is already exported program, so the args, kwargs, and dynamic_shapes
        # are not used
        dynamic_shapes = dynamic_shapes or {}
    else:
        args, kwargs = _get_torch_export_args(args, kwargs)
        if dynamic_shapes is None and dynamic_axes is not None:
            warnings.warn(
                "# 'dynamic_axes' is not recommended when dynamo=True, "
                "and may lead to 'torch._dynamo.exc.UserError: Constraints violated.' "
                "Supply the 'dynamic_shapes' argument instead if export is unsuccessful.",
                UserWarning,
                stacklevel=3,
            )
            try:
                dynamic_shapes, args, kwargs = (
                    _dynamic_shapes.from_dynamic_axes_to_dynamic_shapes(
                        model,
                        args,
                        kwargs,
                        dynamic_axes=dynamic_axes,
                        input_names=input_names,
                        output_names=set(output_names or ()),
                    )
                )
            except Exception as e:
                raise RuntimeError(
                    "# Failed to convert 'dynamic_axes' to 'dynamic_shapes'. "
                    "Please provide 'dynamic_shapes' directly. "
                    "Refer to the documentation for 'torch.export.export' for more information on dynamic shapes."
                ) from e

    dynamic_shapes_with_export_dim, need_axis_mapping = (
        _dynamic_shapes.convert_str_to_export_dim(dynamic_shapes)
    )

    if opset_version < _constants.TORCHLIB_OPSET:
        logger.warning(
            "Setting ONNX exporter to use operator set version %s because "
            "the requested opset_version %s is a lower version than we have implementations for. "
            "Automatic version conversion will be performed, which may not be successful "
            "at converting to the requested version. If version conversion is unsuccessful, "
            "the opset version of the exported model will be kept at %s. "
            "Please consider setting opset_version >=%s to leverage latest ONNX features",
            _constants.TORCHLIB_OPSET,
            opset_version,
            _constants.TORCHLIB_OPSET,
            _constants.TORCHLIB_OPSET,
        )
        registry_opset_version = _constants.TORCHLIB_OPSET
    else:
        registry_opset_version = opset_version

    registry = _registration.ONNXRegistry().from_torchlib(
        opset_version=registry_opset_version
    )
    if custom_translation_table is not None:
        for torch_op, onnx_op in custom_translation_table.items():
            # TODO(justinchuby): Support complex inputs with annotations
            if isinstance(onnx_op, Sequence):
                raise TypeError(
                    "The value in custom_translation_table should be a single callable, not a sequence"
                )
            registry.register_op(torch_op, onnx_op, is_complex=False)

    onnx_program = _core.export(
        model,
        args,
        kwargs,
        registry=registry,
        dynamic_shapes=dynamic_shapes_with_export_dim,
        input_names=input_names,
        output_names=output_names,
        profile=profile,
        report=report,
        verify=verify,
        dump_exported_program=dump_exported_program,
        artifacts_dir=artifacts_dir,
        verbose=verbose,
        optimize=optimize,
        opset_version=opset_version,
    )

    if need_axis_mapping and dynamic_shapes is not None:
        onnx_program._rename_dynamic_axes(dynamic_shapes)

    if f is not None:
        if isinstance(f, io.BytesIO):
            # For legacy export compatibility, we allow f to be a BytesIO object.
            # This is not explicitly supported but we may need to maintain the
            # behavior indefinitely.
            warnings.warn(
                "Saving ONNX model to a BytesIO object is deprecated. "
                "Please use a file path instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            onnx.save(onnx_program.model_proto, f)
        else:
            onnx_program.save(
                f,
                include_initializers=export_params,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
                external_data=external_data,
            )

    return onnx_program
<|endoftext|>"""Compatibility analyzer for PyTorch models."""

# mypy: allow-untyped-defs

from __future__ import annotations

import dataclasses
import operator
import textwrap
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
import torch._export.serde.schema
from torch.export import graph_signature
from torch.onnx._internal.exporter import _dispatching, _registration


if TYPE_CHECKING:
    import torch.fx


@dataclasses.dataclass
class ModelInfo:
    """Information about the model."""

    parameter_count: defaultdict[torch.dtype, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    buffer_count: defaultdict[torch.dtype, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    fx_node_count: int = 0
    fx_node_op_count: defaultdict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    fx_node_target_count: defaultdict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    dispatch_failures: list[tuple[torch.fx.Node, str]] = dataclasses.field(
        default_factory=list
    )
    inputs: dict[str, torch._export.serde.schema.TensorMeta] = dataclasses.field(
        default_factory=dict
    )
    outputs: dict[str, torch._export.serde.schema.TensorMeta] = dataclasses.field(
        default_factory=dict
    )


def _count_weights(
    exported_program: torch.export.ExportedProgram,
) -> tuple[defaultdict[torch.dtype, int], defaultdict[torch.dtype, int]]:
    """Count the size of the parameters in the exported program."""

    parameter_count: defaultdict[torch.dtype, int] = defaultdict(int)
    buffer_count: defaultdict[torch.dtype, int] = defaultdict(int)
    for parameter in exported_program.parameters():
        dtype = parameter.dtype
        parameter_count[dtype] += parameter.numel()

    for buffer in exported_program.buffers():
        dtype = buffer.dtype
        buffer_count[dtype] += buffer.numel()

    return parameter_count, buffer_count


def _format_model_info(model_info: ModelInfo) -> str:
    """Format the information about the model."""
    lines = [
        textwrap.dedent(
            f"""\
            PyTorch ONNX Conversion Analysis

            ## Model Information

            The model has {sum(model_info.parameter_count.values())} parameters and {sum(model_info.buffer_count.values())} buffers (non-trainable parameters).
            Number of parameters per dtype:
            ```python
            {model_info.parameter_count}
            ```
            Number of buffers per dtype:
            ```python
            {model_info.buffer_count}
            ```
            """
        ),
        "Inputs:",
        *[f"- `{name}`: `{meta}`" for name, meta in model_info.inputs.items()],
        "",
        "Outputs:",
        *[f"- `{name}`: `{meta}`" for name, meta in model_info.outputs.items()],
        "",
        f"The FX graph has {model_info.fx_node_count} nodes in total. Number of FX nodes per op:",
    ]
    for op, count in model_info.fx_node_op_count.items():
        lines.append(f"- `{op}`: {count}")
    lines.append("\n")
    lines.append("Of the call_function nodes, the counts of operators used are:\n")
    sorted_targets = sorted(
        model_info.fx_node_target_count.items(),
        key=operator.itemgetter(1),
        reverse=True,
    )
    for target, count in sorted_targets:
        lines.append(f"- `{target}`: {count}")

    lines.append("")
    lines.append("## ONNX Conversion Information")
    lines.append("")

    if model_info.dispatch_failures:
        lines.append(
            "The model contains operators the dispatcher could not find registered ONNX decompositions for. "
            "This may be due to missing implementations, decompositions not registered "
            "correctly, or a bug in the dispatcher."
        )
        lines.append("")
        lines.append("Errors grouped by operator:\n")

        target_to_nodes = defaultdict(list)
        for node, _ in model_info.dispatch_failures:
            target_to_nodes[str(node.target)].append(node)

        target_to_messages = {}
        for node, message in model_info.dispatch_failures:
            if str(node.target) not in target_to_messages:
                target_to_messages[str(node.target)] = message

        for target, nodes in sorted(
            target_to_nodes.items(), key=operator.itemgetter(0), reverse=True
        ):
            message = textwrap.indent(
                f"{target_to_messages[target]}. Example node: `{nodes[0].format_node()}`. All nodes: `{nodes}`",
                "    ",
            )
            lines.append(f"- `{target}`: {message}")
    else:
        lines.append("All operators in the model have registered ONNX decompositions.")

    return "\n".join(lines)


def _get_io_specs(exported_program: torch.export.ExportedProgram) -> tuple[dict, dict]:
    """Get the input and output specs of the exported program."""

    nodes: dict[str, torch.fx.Node] = {
        node.name: node for node in exported_program.graph.nodes
    }
    user_inputs = [
        spec
        for spec in exported_program.graph_signature.input_specs
        if spec.kind == graph_signature.InputKind.USER_INPUT
    ]
    user_outputs = [
        spec
        for spec in exported_program.graph_signature.output_specs
        if spec.kind == graph_signature.OutputKind.USER_OUTPUT
    ]
    inputs: dict[str, torch._export.serde.schema.TensorMeta | str] = {}
    outputs: dict[str, torch._export.serde.schema.TensorMeta | str] = {}
    for spec in user_inputs:
        inputs = _log_spec_into_io_specs(spec, nodes, inputs)
    for spec in user_outputs:
        outputs = _log_spec_into_io_specs(spec, nodes, outputs)
    return inputs, outputs


def _log_spec_into_io_specs(
    spec: graph_signature.InputSpec,
    nodes: dict[str, torch.fx.Node],
    inputs_or_outputs: dict[str, torch._export.serde.schema.TensorMeta | str],
) -> dict[str, torch._export.serde.schema.TensorMeta | str]:
    # If dynamic is set to a constant input, it becomes a
    # symbolic argument, which is not a tensor.
    if isinstance(spec.arg, graph_signature.ConstantArgument):
        # Constant input does not have tensor_meta.
        return inputs_or_outputs
    # Symbolic arguments are not tensors, so it does not have tensor_meta,
    # but we need to provide a string representation for them to inform users.
    name = spec.arg.name
    if isinstance(
        spec.arg,
        (
            graph_signature.SymIntArgument,
            graph_signature.SymFloatArgument,
            graph_signature.SymBoolArgument,
        ),
    ):
        argument_to_str: dict[type[graph_signature.ArgumentSpec], str] = {
            graph_signature.SymIntArgument: "SymInt",
            graph_signature.SymFloatArgument: "SymFloat",
            graph_signature.SymBoolArgument: "SymBool",
        }
        inputs_or_outputs[name] = argument_to_str[type(spec.arg)]
        return inputs_or_outputs
    # FIXME: tensor_meta is None sometimes when the exported program still knows the shape/type
    inputs_or_outputs[name] = nodes[name].meta["tensor_meta"]
    return inputs_or_outputs


def _count_fx_targets(
    exported_program: torch.export.ExportedProgram,
) -> defaultdict[str, int]:
    """Count the number of targets for each node in the exported program."""
    fx_node_target_count: defaultdict[str, int] = defaultdict(int)
    for node in exported_program.graph.nodes:
        if node.op == "call_function":
            fx_node_target_count[str(node.target)] += 1
    return fx_node_target_count


def analyze(
    exported_program: torch.export.ExportedProgram,
    registry: _registration.ONNXRegistry | None = None,
    file=None,
) -> None:
    """Analyze the compatibility of the exported program."""
    # Get basic information about the model
    model_info = ModelInfo()
    model_info.parameter_count, model_info.buffer_count = _count_weights(
        exported_program
    )
    model_info.fx_node_count = len(exported_program.graph.nodes)
    model_info.fx_node_target_count = _count_fx_targets(exported_program)
    inputs, outputs = _get_io_specs(exported_program)
    model_info.inputs = inputs
    model_info.outputs = outputs

    if registry is None:
        registry = _registration.ONNXRegistry.from_torchlib()

    # Try to find ops for every node in the graph
    for node in exported_program.graph.nodes:
        model_info.fx_node_op_count[node.op] += 1
        if node.op == "call_function":
            try:
                onnx_function, message = _dispatching.dispatch(node, registry)
            except Exception as e:
                message = "Critical Error in dispatcher:\n"
                formatted_exception = "\n".join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                message += f"```pytb\n{formatted_exception}\n```\n"
                onnx_function = None
            if onnx_function is None:
                model_info.dispatch_failures.append((node, message))

    # Print the results
    report = _format_model_info(model_info)
    print(report, file=file, flush=True)


def compare_ops(
    program_a: torch.export.ExportedProgram, program_b: torch.export.ExportedProgram
) -> tuple[set[str], set[str]]:
    """Compare and get unique ops in two exported programs.

    Args:
        program_a: The first exported program.
        program_b: The second exported program.

    Returns:
        A tuple of two sets, where the first set contains the unique ops in the first program
        and the second set contains the unique ops in the second program.
    """
    program_a_ops = set(_count_fx_targets(program_a))
    program_b_ops = set(_count_fx_targets(program_b))
    return program_a_ops - program_b_ops, program_b_ops - program_a_ops
<|endoftext|>"""Abstract interface for ONNX exportable modules."""

from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Sequence


class ExportableModule(torch.nn.Module, abc.ABC):
    """Abstract interface for ONNX exportable modules.

    Inherit from this class and implement the defined abstract methods
    to create a module that can be exported to ONNX format.

    Example::

        class Model(torch.nn.Module):
            def forward(self, x):
                return x * 2


        class MyExportableModule(torch.onnx.ExportableModule):
            def __init__(self):
                super().__init__()
                self.model = Model()

            def forward(self, x):
                return self.model(x)

            def example_arguments(self):
                return (torch.randn(2, 3, 224, 224),), None

            def input_names(self):
                return ("input",)

            def output_names(self):
                return ("output",)

            def dynamic_shapes(self):
                return ({0: "batch_size"},)


        exportable_module = MyExportableModule()
        onnx_program = exportable_module.to_onnx()
        # The model can also be supplied directly to torch.onnx.export
        onnx_program = torch.onnx.export(exportable_module)
    """

    @abc.abstractmethod
    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        """Return example arguments for the model's forward method.

        This method must be implemented by subclasses to provide sample inputs
        that can be used for tracing, testing, and ONNX export. The returned
        arguments should be representative of the expected input shapes and types
        during inference.

        Example::

            def example_arguments(self):
                # For a model expecting a single tensor input
                return (torch.randn(1, 3, 224, 224),), None


            def example_arguments(self):
                # For a model with multiple inputs and keyword arguments
                return (torch.randn(1, 3, 224, 224), torch.randn(1, 512)), {
                    "temperature": 1.0
                }

        Returns:
            A tuple containing:

            - A tuple of positional arguments to pass to the forward method
            - A dictionary of keyword arguments (or None if no kwargs are needed)
        """
        raise NotImplementedError

    def dynamic_shapes(self) -> Any:
        """Return dynamic shape specifications for the model's inputs.

        Override this method to specify which dimensions of the input tensors
        should be treated as dynamic during ONNX export. This allows the exported
        model to accept inputs with varying sizes along the specified dimensions.

        Example::

            def dynamic_shapes(self):
                # Specify batch dimension as dynamic for input named 'x'
                return {"x": {0: "batch_size"}}


            def dynamic_shapes(self):
                # Multiple dynamic dimensions
                return {
                    "input": {0: "batch_size", 2: "height", 3: "width"},
                    "mask": {0: "batch_size"},
                }

        Note:
            The default implementation returns None, indicating all dimensions are static.

        Returns:
            Dynamic shape specification compatible with ``torch.export.export``.
            Return None if all input dimensions should be static. The format can be:

            - A dictionary mapping input names to dimension specifications
            - A tuple/list of dimension specifications corresponding to inputs
            - Any format accepted by the ``dynamic_shapes`` parameter of ``torch.export.export``
        """
        return None

    def input_names(self) -> Sequence[str] | None:
        """Return names for the model's input tensors.

        Override this method to provide custom names for the input tensors in the
        exported ONNX model. These names will be used as identifiers in the ONNX
        graph and can be useful for debugging and model inspection.

        Example::

            def input_names(self):
                return ["image", "mask"]


            def input_names(self):
                # For a single input
                return ["input_tensor"]

        Note:
            The default implementation returns None, which results in auto-generated names.

        Returns:
            A sequence of strings representing input names, or None to use default names.
            The number of names should match the number of positional arguments in the
            forward method.
        """
        return None

    def output_names(self) -> Sequence[str] | None:
        """Return names for the model's output tensors.

        Override this method to provide custom names for the output tensors in the
        exported ONNX model. These names will be used as identifiers in the ONNX
        graph and can be useful for debugging and model inspection.

        Example::

            def output_names(self):
                return ["logits", "probabilities"]


            def output_names(self):
                # For a single output
                return ["prediction"]

        Note:
            The default implementation returns None, which results in auto-generated names.

        Returns:
            A sequence of strings representing output names, or None to use default names.
            The number of names should match the number of outputs from the forward method.
            For models returning multiple outputs, provide a name for each output.
        """
        return None

    def to_onnx(self, **kwargs: Any) -> torch.onnx.ONNXProgram:
        """Export the module to ONNX format.

        This method provides a convenient wrapper around ``torch.onnx.export`` that
        automatically uses the example arguments, dynamic shapes, and input/output
        names defined by the module. Additional export options can be specified via
        keyword arguments.

        See Also: ``torch.onnx.export`` for complete documentation of export options.

        Args:
            **kwargs: Additional keyword arguments to pass to ``torch.onnx.export``.
                Common options include:

                - ``opset_version`` (int): The ONNX opset version to target
                - ``optimize`` (bool): Whether to apply optimizations to the exported model

        Returns:
            An ONNXProgram object containing the exported model and metadata.
        """
        result = torch.onnx.export(self, **kwargs)
        if result is None:
            raise AssertionError("result must be non-None")
        return result
<|endoftext|>"""Error classes for the ONNX exporter."""

from __future__ import annotations

import torch.onnx.errors


class TorchExportError(torch.onnx.errors.OnnxExporterError):
    """Error during graph capturing using torch.export."""


class ConversionError(torch.onnx.errors.OnnxExporterError):
    """Error during ExportedProgram to ONNX conversion."""


class DispatchError(ConversionError):
    """Error during ONNX Function dispatching."""


class GraphConstructionError(ConversionError):
    """Error during ONNX graph construction."""
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter import _constants


if TYPE_CHECKING:
    from collections.abc import Sequence


# The opset domain for ONNX operators
_ONNX_DOMAIN = ""

logger = logging.getLogger(__name__)


def rename_inputs(model: ir.Model, new_names: Sequence[str]) -> None:
    unique_names = frozenset(new_names)
    if len(unique_names) != len(new_names):
        seen = set()
        duplicates = []
        for name in new_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        raise ValueError(f"Input names cannot be duplicated: {duplicates}")

    for input, new_name in zip(model.graph.inputs, new_names):
        input.metadata_props["pkg.torch.onnx.original_node_name"] = str(input.name)
        input.name = new_name


def rename_outputs(model: ir.Model, new_names: Sequence[str]) -> None:
    unique_names = frozenset(new_names)
    if len(unique_names) != len(new_names):
        seen = set()
        duplicates = []
        for name in new_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        raise ValueError(f"Output names cannot be duplicated: {duplicates}")

    for output, new_name in zip(model.graph.outputs, new_names):
        output.metadata_props["pkg.torch.onnx.original_node_name"] = str(output.name)
        output.name = new_name


def _all_values(model: ir.Model):
    """Yield all values in a model."""
    # Yield all values in the model
    yield from model.graph.inputs
    yield from model.graph.initializers.values()
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        yield from node.outputs
    # Yield all values in functions
    for function in model.functions.values():
        yield from function.inputs
        for node in ir.traversal.RecursiveGraphIterator(function):
            yield from node.outputs


def _replace_names(shape_expr: str, rename_mapping: dict[str, str]) -> str:
    """Replace all known names in a shape expression with new names."""
    for old_name, new_name in rename_mapping.items():
        shape_expr = re.sub(
            rf"(?<!\w){re.escape(old_name)}(?!\w)", new_name, shape_expr
        )
    return shape_expr


def rename_axis(
    model: ir.Model, rename_mapping: dict[str | ir.SymbolicDim, str]
) -> None:
    """Rename dynamic axes in a model according to the specified dynamic_axes names."""

    # Create a mapping from string to string for easier replacement
    string_mapping: dict[str, str] = {}
    for key, value in tuple(rename_mapping.items()):
        if isinstance(key, ir.SymbolicDim):
            if isinstance(key.value, str):
                string_mapping[key.value] = value
            else:
                raise ValueError(
                    f"Invalid SymbolicDim value in rename_mapping: {key.value!r}. "
                    "Expected str."
                )
        elif isinstance(key, str):
            string_mapping[key] = value
        else:
            raise ValueError(
                f"Invalid key type in rename_mapping: {type(key)}({key!r}). Expected "
                "str or ir.SymbolicDim."
            )

    # NOTE: Mapping needs to be sorted by length because the shape expression
    # could have multiple ways to be expressed, for example,
    # {"s1": sequence_length, "s11": "past_sequence_length", "s1 + s11": "masked_sequence_length"}
    # We prefer the replacement starts from the longest match.
    sorted_rename_mapping = dict(
        sorted(string_mapping.items(), key=lambda item: len(item[0]), reverse=True)
    )
    for value in _all_values(model):
        if value.shape is None:
            continue
        new_shape = []
        changed = False
        for dim in value.shape:
            if not isinstance(dim, ir.SymbolicDim):
                new_shape.append(dim)
                continue
            dim_name = dim.value
            if dim_name in sorted_rename_mapping:
                new_shape.append(sorted_rename_mapping[dim_name])
                changed = True
            elif dim_name is not None:
                # For example: "2*s1", "s1+1", "s1-1", "s1*s2", "s1/s2"
                new_name = _replace_names(dim_name, sorted_rename_mapping)
                new_shape.append(new_name)
                if new_name != dim_name:
                    changed = True
            else:
                new_shape.append(None)
        if changed:
            value.shape = ir.Shape(new_shape)


def _maybe_set_opset_version(
    opset_imports: dict[str, int], domain: str, version: int | None
) -> None:
    """Set the opset version for the domain."""
    if domain in opset_imports and opset_imports[domain] != 1:
        # Already set
        return
    if domain == _ONNX_DOMAIN:
        opset_imports[domain] = _constants.TORCHLIB_OPSET
        return
    if version is None:
        # We don't know the opset version, so set it to 1
        # This is valid for the custom function domains like "pkg.torch.__subgraph__"
        opset_imports[domain] = 1
        return
    # Set the known opset version for the domain
    opset_imports[domain] = version


def add_opset_imports(model: ir.Model) -> None:
    """Collect all opsets used and add opset imports to the model and functions."""
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        domain = node.domain
        _maybe_set_opset_version(model.opset_imports, domain, node.version)

    for function in model.functions.values():
        for node in ir.traversal.RecursiveGraphIterator(function):
            domain = node.domain
            _maybe_set_opset_version(function.opset_imports, domain, node.version)
        for domain, version in function.opset_imports.items():
            # Add all opsets used in the function to the model, because ONNX Runtime
            # does not handle adding the opset imports to the model after inlining during inference.
            # This should happen after all opsets are collected for the function from its nodes.
            _maybe_set_opset_version(model.opset_imports, domain, version)
<|endoftext|># mypy: allow-untyped-defs

from __future__ import annotations

import ctypes
import datetime
import inspect
import itertools
import logging
import operator
import pathlib
import sys
import textwrap
import traceback
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import onnxscript
import onnxscript.evaluator
from onnxscript import ir
from onnxscript.ir import convenience as ir_convenience

import torch
import torch.fx
from torch.export import graph_signature
from torch.onnx._internal._lazy_import import onnxscript_apis
from torch.onnx._internal.exporter import (
    _analysis,
    _building,
    _capture_strategies,
    _constants,
    _dispatching,
    _errors,
    _flags,
    _fx_passes,
    _ir_passes,
    _onnx_program,
    _registration,
    _reporting,
    _tensors,
    _type_casting,
    _verification,
)


if typing.TYPE_CHECKING:
    import os

    import numpy.typing as npt


# Define utilities to convert PyTorch data types so users do not need to specify manually
_TORCH_DTYPE_TO_ONNX: dict[torch.dtype, ir.DataType] = {
    torch.bfloat16: ir.DataType.BFLOAT16,
    torch.bool: ir.DataType.BOOL,
    torch.complex128: ir.DataType.COMPLEX128,
    torch.complex64: ir.DataType.COMPLEX64,
    torch.float16: ir.DataType.FLOAT16,
    torch.float32: ir.DataType.FLOAT,
    torch.float64: ir.DataType.DOUBLE,
    torch.float8_e4m3fn: ir.DataType.FLOAT8E4M3FN,
    torch.float8_e4m3fnuz: ir.DataType.FLOAT8E4M3FNUZ,
    torch.float8_e5m2: ir.DataType.FLOAT8E5M2,
    torch.float8_e5m2fnuz: ir.DataType.FLOAT8E5M2FNUZ,
    torch.float4_e2m1fn_x2: ir.DataType.FLOAT4E2M1,
    torch.int16: ir.DataType.INT16,
    torch.int32: ir.DataType.INT32,
    torch.int64: ir.DataType.INT64,
    torch.int8: ir.DataType.INT8,
    torch.uint8: ir.DataType.UINT8,
    torch.uint16: ir.DataType.UINT16,
    torch.uint32: ir.DataType.UINT32,
    torch.uint64: ir.DataType.UINT64,
}
_BLUE = "\033[96m"
_END = "\033[0m"

_STEP_ONE_ERROR_MESSAGE = textwrap.dedent(
    f"""\
    Failed to export the model with torch.export. {_BLUE}This is step 1/3{_END} of exporting the model to ONNX. Next steps:
    - Modify the model code for `torch.export.export` to succeed. Refer to https://pytorch.org/docs/stable/generated/exportdb/index.html for more information.
    - Debug `torch.export.export` and submit a PR to PyTorch.
    - Create an issue in the PyTorch GitHub repository against the {_BLUE}*torch.export*{_END} component and attach the full error stack as well as reproduction scripts."""
)

_STEP_TWO_ERROR_MESSAGE = textwrap.dedent(
    f"""\
    Failed to decompose the FX graph for ONNX compatibility. {_BLUE}This is step 2/3{_END} of exporting the model to ONNX. Next steps:
    - Create an issue in the PyTorch GitHub repository against the {_BLUE}*torch.export*{_END} component and attach the full error stack as well as reproduction scripts.
    - Create an error report with `torch.onnx.export(..., report=True)`, and save the ExportedProgram as a pt2 file. Create an issue in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component. Attach the error report and the pt2 model."""
)

_STEP_THREE_ERROR_MESSAGE = textwrap.dedent(
    f"""\
    Failed to convert the exported program to an ONNX model. {_BLUE}This is step 3/3{_END} of exporting the model to ONNX. Next steps:
    - If there is a missing ONNX function, implement it and register it to the registry.
    - If there is an internal error during ONNX conversion, debug the error and submit a PR to PyTorch.
    - Create an error report with `torch.onnx.export(..., report=True)`, and save the ExportedProgram as a pt2 file. Create an issue in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component. Attach the error report and the pt2 model."""
)

logger = logging.getLogger(__name__)
# The current tracer that is being used to trace the operators,
# used by torch/onnx/_internal/exporter/_torchlib/ops/hop.py
current_tracer: _building.OpRecorder | None = None


def torch_dtype_to_onnx_dtype(dtype: torch.dtype) -> ir.DataType:
    return _TORCH_DTYPE_TO_ONNX[dtype]


class TorchTensor(ir.Tensor):
    def __init__(self, tensor: torch.Tensor, name: str | None = None) -> None:
        # Pass the tensor as the raw data to ir.Tensor's constructor
        if tensor.dtype == torch.float4_e2m1fn_x2:
            # Change the shape to the unpacked shape
            shape = ir.Shape(_type_casting.get_float4_shape(tensor), frozen=True)
        else:
            # The base class will set the shape to the tensor's shape
            shape = None
        super().__init__(
            tensor,
            dtype=torch_dtype_to_onnx_dtype(tensor.dtype),
            shape=shape,
            name=name,
        )

    def numpy(self) -> npt.NDArray:
        self.raw: torch.Tensor

        # Handle dtypes that are not natively supported by NumPy:
        # We pick an uint dtype that has the same size as the original dtype,
        # view the tensor as that dtype so that it is convertible to NumPy,
        # and then view it back to the proper dtype (using ml_dtypes obtained by
        # calling dtype.numpy()).
        if self.dtype == ir.DataType.BFLOAT16:
            return (
                self.raw.view(torch.uint16).numpy(force=True).view(self.dtype.numpy())
            )
        if self.dtype in {
            ir.DataType.FLOAT8E4M3FN,
            ir.DataType.FLOAT8E4M3FNUZ,
            ir.DataType.FLOAT8E5M2,
            ir.DataType.FLOAT8E5M2FNUZ,
        }:
            return self.raw.view(torch.uint8).numpy(force=True).view(self.dtype.numpy())
        if self.dtype == ir.DataType.FLOAT4E2M1:
            return _type_casting.unpack_float4x2_as_uint8(self.raw).view(
                self.dtype.numpy()
            )

        return self.raw.numpy(force=True)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> npt.NDArray:
        del copy  # Unused, but needed for the signature
        if dtype is None:
            return self.numpy()
        return self.numpy().__array__(dtype)

    def _get_cbytes(self):
        """Get a ctypes byte array pointing to the tensor data."""
        import torch._subclasses.fake_tensor

        with torch._subclasses.fake_tensor.unset_fake_temporarily():
            # Disable any fake mode so calling detach() etc. will return a real tensor
            tensor = self.raw.detach().cpu().contiguous()

        if torch._subclasses.fake_tensor.is_fake_tensor(tensor):
            raise TypeError(
                f"Cannot take content out from the FakeTensor ('{self.name}'). Please replace the tensor "
                "with a tensor backed by real data using ONNXProgram.apply_weights() "
                "or save the model without initializers by setting include_initializers=False."
            )

        # Return the tensor to ensure it is not garbage collected while the ctypes array is in use
        return tensor, (
            ctypes.c_ubyte * tensor.element_size() * tensor.numel()
        ).from_address(tensor.data_ptr())

    def tobytes(self) -> bytes:
        # On big-endian machines, call the super's tobytes() which returns a little-endian result.
        if sys.byteorder == "big":
            return super().tobytes()
        # Implement tobytes to support native PyTorch types so we can use types like bfloat16
        # Reading from memory directly is also more efficient because
        # it avoids copying to a NumPy array
        _, data = self._get_cbytes()
        return bytes(data)

    def tofile(self, file) -> None:
        # On big-endian machines, call the super's tofile() which returns a little-endian result.
        if sys.byteorder == "big":
            return super().tofile(file)
        _, data = self._get_cbytes()
        return file.write(data)


# https://github.com/pytorch/pytorch/blob/ee6cb6daa173896f8ea1876266a19775aaa4f610/torch/export/graph_signature.py#L56C1-L62C19
# class InputKind(Enum):
#     USER_INPUT = auto()
#     PARAMETER = auto()
#     BUFFER = auto()
#     CONSTANT_TENSOR = auto()
#     CUSTOM_OBJ = auto()
#     TOKEN = auto()

# https://github.com/pytorch/pytorch/blob/ee6cb6daa173896f8ea1876266a19775aaa4f610/torch/export/graph_signature.py#L89C1-L96C19
# class OutputKind(Enum):
#     USER_OUTPUT = auto()
#     LOSS_OUTPUT = auto()
#     BUFFER_MUTATION = auto()
#     GRADIENT_TO_PARAMETER = auto()
#     GRADIENT_TO_USER_INPUT = auto()
#     USER_INPUT_MUTATION = auto()
#     TOKEN = auto()


def _set_shape_types(
    values: Sequence[ir.Value],
    meta_vals: Sequence[torch.Tensor],
    complex_to_float: bool = True,
) -> None:
    if not isinstance(meta_vals, Sequence):
        logger.warning(
            "Expected meta_vals to be a sequence, but got %s. There may be an internal error.",
            meta_vals,
        )
        meta_vals = (meta_vals,)
    for value, meta_val in zip(values, meta_vals):
        _set_shape_type(value, meta_val, complex_to_float=complex_to_float)


def _set_shape_type(
    value: ir.Value,
    meta_val: torch.Tensor
    | torch.SymBool
    | torch.SymInt
    | torch.SymFloat
    | tuple[torch.Tensor],
    complex_to_float: bool,
) -> None:
    if isinstance(meta_val, tuple):
        logger.warning("Setting shape and type of tensors is not supported yet")
    if isinstance(meta_val, torch.Tensor):
        dims = []
        shape: tuple[int, ...]
        if meta_val.dtype == torch.float4_e2m1fn_x2:
            # Change the shape to the unpacked shape
            shape = _type_casting.get_float4_shape(meta_val)
        else:
            shape = meta_val.shape
        for dim in shape:
            if isinstance(dim, int):
                dims.append(dim)
            else:
                # pyrefly: ignore [bad-argument-type]
                dims.append(str(dim.node))

        # If the dtype is set already (e.g. by the onnx_symbolic ops),
        # we don't need to set it again.
        #
        # When a user specifies complex in onnx_symbolic, we consider that to
        # be the intention even though none of the ONNX ops deals with complex values.
        # In this case, we don't change the dtype or the shape of the tensor.
        if value.dtype is None:
            value.dtype = torch_dtype_to_onnx_dtype(meta_val.dtype)
            if complex_to_float and meta_val.dtype.is_complex:
                value.dtype = torch_dtype_to_onnx_dtype(meta_val.dtype.to_real())
                # Add 2 as the last dimension if the tensor is complex to hold the real/imag parts
                dims.append(2)

        value.shape = ir.Shape(dims)
    elif isinstance(meta_val, (int, torch.SymInt)):
        # aten::sym_size output is an int, not a tensor, which stands
        # for the size of one dim. We treat it as a scalar.
        value.dtype = ir.DataType.INT64
        value.shape = ir.Shape([])
    elif isinstance(meta_val, (bool, torch.SymBool)):
        value.dtype = ir.DataType.BOOL
        value.shape = ir.Shape([])
    elif isinstance(meta_val, (float, torch.SymFloat)):
        value.dtype = ir.DataType.FLOAT
        value.shape = ir.Shape([])


def _get_qualified_module_name(cls: Any) -> str:
    if isinstance(cls, str):
        return cls
    module = cls.__module__
    if module is None or module == str.__class__.__module__:
        return cls.__name__
    return module + "." + cls.__name__


def _get_node_namespace(node: torch.fx.Node) -> tuple[str, list[str], list[str]]:
    """Get the namespace and scope of the node.

    Example::

        {
            'L__self__': ('', <class 'torchvision.models.resnet.ResNet'>),
            'L__self___avgpool': ('avgpool', <class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>)
        }

    Will yield

    namespace: ": torchvision.models.resnet.ResNet/avgpool: torch.nn.modules.pooling.AdaptiveAvgPool2d/node_name: node_target"
    class_hierarchy: ["torchvision.models.resnet.ResNet", "torch.nn.modules.pooling.AdaptiveAvgPool2d", <node_target>]
    name_scopes: ["", "avgpool", <node_name>]

    Args:
        node: The node to get the namespace and scope of.

    Returns:
        (namespace, class_hierarchy, name_scope)
    """
    nn_module_stack = node.meta.get("nn_module_stack")
    logger.debug("%s", nn_module_stack)
    if nn_module_stack is None:
        logger.warning(
            "nn_module_stack not found for node '%s'. Skip adding metadata...",
            node.name,
        )
        return f"{node.name}: {node.target}", [str(node.target)], [node.name]
    namespaces = []
    class_hierarchy = []
    name_scopes = []
    for name, nn_module in nn_module_stack.values():
        name_scopes.append(name)
        nn_module_name = _get_qualified_module_name(nn_module)
        class_hierarchy.append(nn_module_name)
        namespaces.append(f"{name}: {_get_qualified_module_name(nn_module)}")
    namespaces.append(f"{node.name}: {node.target}")
    class_hierarchy.append(str(node.target))
    name_scopes.append(node.name)

    return "/".join(namespaces), class_hierarchy, name_scopes


def _set_node_metadata(fx_node: torch.fx.Node, ir_node: ir.Node) -> None:
    """Adds namespace and other node metadata to the ONNX node."""
    namespace, class_hierarchy, name_scopes = _get_node_namespace(fx_node)
    ir_node.metadata_props["namespace"] = namespace
    ir_node.metadata_props["pkg.torch.onnx.class_hierarchy"] = repr(class_hierarchy)
    ir_node.metadata_props["pkg.torch.onnx.name_scopes"] = repr(name_scopes)
    ir_node.metadata_props["pkg.torch.onnx.fx_node"] = str(fx_node.format_node())
    ir_node.metadata_props["pkg.torch.onnx.stack_trace"] = fx_node.meta.get(
        "stack_trace", ""
    )


def _handle_getitem_node(
    node: torch.fx.Node, node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]]
) -> ir.Value:
    """Handle a getitem node.

    Add the input value it is getting to the mapping, then return the value.

    There are two cases for this node:
    1. The output is a Sequence (traced), we can simply get the value from the sequence
    2. The output is produced by a SplitToSequence node, we need to get the value from the sequence value
    This function only handles the first case
    """
    if len(node.all_input_nodes) != 1:
        raise AssertionError(f"Expected 1 input node, got {len(node.all_input_nodes)}")
    source = node.all_input_nodes[0]
    source_outputs = node_name_to_values[source.name]
    if not isinstance(source_outputs, Sequence):
        raise AssertionError(
            f"Expected {source.name} to output sequence, got {node_name_to_values[source.name]}"
        )
    index = typing.cast(int, node.args[1])
    value = source_outputs[index]
    # Save the getitem value to the values mapping to in case
    # it is one of the graph outputs
    node_name_to_values[node.name] = value
    # Rename the name of value with the getitem name.
    value.name = node.name
    return value


def _handle_call_function_node(
    graph_like: ir.Graph | ir.Function,
    node: torch.fx.Node,
    node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]],
) -> None:
    """Handle a call_function node.

    Args:
        graph: The ONNX graph at construction.
        node: The FX node to translate.
        node_name_to_values: A mapping of FX node names to their produced ir.Value.
    """
    if node.target is operator.getitem:
        _handle_getitem_node(node, node_name_to_values)
    # Add op to the graph
    op = str(node.target)
    fx_inputs, attributes, input_names, output_names = _get_inputs_and_attributes(node)
    inputs: list[ir.Value | None] = []
    for i, input_ in enumerate(fx_inputs):
        if input_ is None:
            inputs.append(None)
        elif hasattr(input_, "name"):
            if isinstance(input_, torch.fx.Node) and input_.target is operator.getitem:
                actual_input = _handle_getitem_node(input_, node_name_to_values)
                inputs.append(actual_input)
            else:
                value = node_name_to_values[input_.name]
                if isinstance(value, Sequence):
                    raise AssertionError(f"Unexpected sequence value for {input_.name}")
                inputs.append(value)
        else:
            attributes[f"arg_{i}"] = input_

    outputs = [ir.Value(name=name) for name in output_names]
    if len(outputs) > 1:
        _set_shape_types(outputs, node.meta["val"], complex_to_float=False)
        node_name_to_values[node.name] = outputs
    else:
        _set_shape_type(outputs[0], node.meta["val"], complex_to_float=False)
        node_name_to_values[node.name] = outputs[0]
    ir_node = ir.Node(
        "pkg.torch.ops",
        op,
        inputs,
        attributes=ir_convenience.convert_attributes(attributes),
        outputs=outputs,
        name=node.name,
    )
    ir_node.meta["node"] = node
    ir_node.metadata_props["pkg.torch.onnx.input_names"] = repr(input_names)
    # Record the nn.Module stack for the node
    _set_node_metadata(node, ir_node)

    graph_like.append(ir_node)


def _convert_fx_arg_to_onnx_arg(
    arg,
    node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]],
    node_name_to_local_functions: dict[str, ir.Function],
) -> Any:
    """Convert an FX argument to an ONNX compatible argument.

    This function
    - Converts a torch dtype to an integer
    - Converts a torch device/memory_format/layout to a string
    - Converts a torch.fx.Node to an ir.Value
    - Converts a sequence of torch.fx.Node to a sequence of ir.Value
    - Converts a get_attr node to an ir.Function
    """
    if arg is None:
        # None arguments are not modified because when the arg is an ONNX input
        # we need to preserve the None value; when the arg is an ONNX attribute,
        # we want to drop the value.
        # The actual dropping of a None attribute value is done by OpRecorder
        return None
    if hasattr(arg, "name"):
        if isinstance(arg, torch.fx.Node) and arg.target is operator.getitem:
            source = arg.all_input_nodes[0]
            source_outputs = node_name_to_values[source.name]
            if isinstance(source_outputs, Sequence):
                # If the node is getting an input from another node, get the actual value the node is retrieving
                return _handle_getitem_node(arg, node_name_to_values)
            else:
                # `source_outputs` is a sequence(tensor()) value and we need to
                # use SequenceAt to get the value. This is handled by torchlib
                pass
        if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
            # A get_attr node may refer to either a subgraph function
            # (handled via node_name_to_local_functions) or, for lifted
            # tensor constants materialized from nested invoke_subgraph
            # tracing, an initializer value in node_name_to_values.
            if arg.name in node_name_to_values:
                return node_name_to_values[arg.name]
            return node_name_to_local_functions[arg.name]
        # If the input is a node, get the value from the mapping
        return node_name_to_values[arg.name]
    if isinstance(arg, (list, tuple)):
        return [
            _convert_fx_arg_to_onnx_arg(
                elem, node_name_to_values, node_name_to_local_functions
            )
            for elem in arg
        ]
    if isinstance(arg, (torch.device, torch.memory_format, torch.layout)):
        return str(arg)
    if isinstance(arg, torch.dtype):
        return torch_dtype_to_onnx_dtype(arg)
    # Maybe a Python value
    return arg


def _get_onnxscript_opset(opset_version: int) -> onnxscript.values.Opset:
    return onnxscript.values.Opset("", opset_version)


def _is_onnx_op(op: Any) -> bool:
    """Whether the op overload is an ONNX custom op implemented with PyTorch."""
    if not isinstance(op, torch._ops.OpOverload):
        return False
    return op.name().startswith("onnx::")


def _parse_onnx_op(op: torch._ops.OpOverload) -> tuple[str, int]:
    """Parse the ONNX custom op overload name to get the op type and opset version."""
    name = op.name()[len("onnx::") :]
    name, _, opset = name.partition(".opset")
    return name, int(opset)


def _handle_call_function_node_with_lowering(
    model: ir.Model,
    node: torch.fx.Node,
    node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]],
    *,
    graph_like: ir.Graph | ir.Function,
    constant_farm: dict[Any, ir.Value],
    registry: _registration.ONNXRegistry,
    opset: onnxscript.values.Opset,
    node_name_to_local_functions: dict[str, ir.Function],
) -> None:
    """Translate a call_function node to an ONNX node.

    Args:
        model: The ONNX model at construction.
        node: The FX node to translate.
        node_name_to_values: A mapping of FX node names to their produced ONNX ``Value``.
        graph_like: The current ONNX graph at construction.
            Must add nodes to this graph because it can be a subgraph that is currently being constructed.
        constant_farm: A mapping of constant values to existing ONNX ``Value``s.
        registry: The registry of all aten to ONNX decomposition functions.
        opset: The ONNX Script opset object for constructing ONNX nodes.
        node_name_to_local_functions: A mapping of subgraph names to the corresponding ONNX functions.
    """
    if node.target is operator.getitem:
        source = node.all_input_nodes[0]
        source_outputs = node_name_to_values[source.name]
        if isinstance(source_outputs, Sequence):
            _handle_getitem_node(node, node_name_to_values)
            return
        else:
            # `source_outputs` is a sequence(tensor()) value and we need to
            # use SequenceAt to get the value. This is handled by torchlib
            pass

    # Map FX inputs to ONNX inputs and fill optional inputs.
    # torch_args and torch_kwargs are for op-level validation
    fx_args = node.args
    fx_kwargs = node.kwargs

    # Replace the input FX nodes with ONNX values
    onnx_args = [
        _convert_fx_arg_to_onnx_arg(
            input_, node_name_to_values, node_name_to_local_functions
        )
        for input_ in fx_args
    ]

    onnx_kwargs = {}
    for key, value in fx_kwargs.items():
        onnx_kwargs[key] = _convert_fx_arg_to_onnx_arg(
            value, node_name_to_values, node_name_to_local_functions
        )
        if key == "dtype" and onnx_kwargs[key] is None:
            # Set dtype to -1 if it is None
            # TODO(justinchuby): Maybe keep it as None?
            onnx_kwargs[key] = -1

    if _is_onnx_op(node.target):
        # Handle torch.ops.onnx.* ops. These ops can be directly added to the graph
        op_type, opset_version = _parse_onnx_op(node.target)  # type: ignore[arg-type]
        # If final inputs are None, strip them from the node inputs
        for input_ in reversed(onnx_args):
            if input_ is not None:
                break
            onnx_args.pop()
        onnx_node = ir.Node(
            "",
            op_type,
            onnx_args,
            ir.convenience.convert_attributes(onnx_kwargs),
            name=node.name,
            num_outputs=len(node.target._schema.returns),  # type: ignore[union-attr]
            version=opset_version,
        )
        # Store the single node in a list to be consistent with the rest of the code for further processing
        onnx_nodes = [onnx_node]
        if len(onnx_node.outputs) == 1:
            outputs = onnx_node.outputs[0]
        else:
            outputs = onnx_node.outputs  # type: ignore[assignment]
    else:
        # Find the matching ONNX overload for the node
        # TODO: Log the message here to expose false positives
        onnx_function, message = _dispatching.dispatch(node, registry)

        if onnx_function is None:
            raise _errors.DispatchError(
                f"No ONNX function found for {node.target!r}. Failure message: {message}"
            )

        with onnxscript.evaluator.default_as(
            tracer := _building.OpRecorder(opset, constant_farm)
        ):
            global current_tracer
            current_tracer = tracer
            try:
                outputs = onnx_function(*onnx_args, **onnx_kwargs)
            except Exception as e:
                raise _errors.GraphConstructionError(
                    f"Error when calling function '{onnx_function}' with args '{onnx_args}' and kwargs '{onnx_kwargs}'"
                ) from e
            finally:
                current_tracer = None

        # Add the defined functions to the model
        for identifier, onnxscript_function in tracer.functions.items():
            if identifier in model.functions:
                continue
            if isinstance(onnxscript_function, ir.Function):
                ir_function = onnxscript_function
            else:
                # TODO: Get IR function directly when onnxscript is updated
                proto = onnxscript_function.to_function_proto()
                ir_function = ir.serde.deserialize_function(proto)
            model.functions[identifier] = ir_function
            # Opset imports are added to the model in the final add_opset_imports pass

        onnx_nodes = tracer.nodes
        del tracer  # tracer is no longer needed

    # NOTE: Instead of using the output names from node.target._schema,
    # we always use the index if there are more than one outputs so the
    # names can be programmatically reconstructed. This is useful for
    # comparing values from the ONNX graph with those from the FX graph.
    #
    # When there are multiple outputs, the output names will be
    # node_name__0, node_name__1, etc.
    if isinstance(outputs, Sequence):
        _set_shape_types(outputs, node.meta["val"], complex_to_float=True)
        node_name_to_values[node.name] = outputs
        for i, output in enumerate(outputs):
            output.name = f"{node.name}__{i}"
            # Set the name of the producing node using the value name for correspondence
            producer = output.producer()
            if producer is not None:
                producer.name = f"node_{output.name}"
    else:
        _set_shape_type(outputs, node.meta["val"], complex_to_float=True)
        node_name_to_values[node.name] = outputs
        outputs.name = node.name
        producer = outputs.producer()
        if producer is not None:
            producer.name = f"node_{outputs.name}"

    for ir_node in onnx_nodes:
        ir_node.meta["node"] = node
        # Record the nn.Module stack for the node
        _set_node_metadata(node, ir_node)

    # Add the traced nodes to the current graph
    # Must add nodes to this graph, not model.graph, because it can be a subgraph that is currently being constructed
    graph_like.extend(onnx_nodes)


def _handle_placeholder_node(
    node: torch.fx.Node,
    node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]],
    *,
    graph_like: ir.Graph | ir.Function,
    lower: str,
    opset: onnxscript.values.Opset,
) -> None:
    # Placeholder nodes are user inputs
    # We need to create a new tensor for each user input
    # and add it to the graph's inputs
    name = node.name
    input_ = _tensors.SymbolicTensor(opset, name=name)
    input_.meta["node"] = node
    _set_shape_type(input_, node.meta["val"], complex_to_float=lower != "none")
    node_name_to_values[name] = input_
    # The inputs should be add to the graph here
    graph_like.inputs.append(input_)


def _handle_get_attr_node(
    node: torch.fx.Node,
    *,
    owned_graphs: Mapping[str, ir.Function],
    node_name_to_local_functions: dict[str, ir.Function],
) -> None:
    """Handle a get_attr node by assigning the corresponding ONNX function to the node name.

    An example ExportedProgram that has uses get_attr nodes is:

        ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: "f32[5]"):
                true_graph_0 = self.true_graph_0  # get_attr
                false_graph_0 = self.false_graph_0  # get_attr
                conditional = torch.ops.higher_order.cond(False, true_graph_0, false_graph_0, [arg0_1]);  true_graph_0 = false_graph_0 = arg0_1 = None
                getitem: "f32[5]" = conditional[0];  conditional = None
                return (getitem,)

            class <lambda>(torch.nn.Module):
                def forward(self, arg0_1: "f32[5]"):
                    cos: "f32[5]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
                    return (cos,)

            class <lambda>(torch.nn.Module):
                def forward(self, arg0_1: "f32[5]"):
                    sin: "f32[5]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
                    return (sin,)

    Args:
        node: The FX node to translate.
        owned_graphs: A mapping of subgraph names to the corresponding ONNX functions.
        node_name_to_local_functions: A mapping of local function names to their corresponding ONNX functions.
    """
    if not isinstance(node.target, str):
        logger.warning(
            "Expected node.target for the node %s to be a string, but got '%s'. There may be an internal error.",
            node,
            type(node.target),
        )
        return
    function = owned_graphs[node.target]
    node_name_to_local_functions[node.name] = function


def _handle_output_node(
    node: torch.fx.Node,
    node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]],
    graph_like: ir.Graph | ir.Function,
) -> None:
    """Handle an output node by adding the output to the graph's outputs.

    Args:
        node: The FX node to translate.
        node_name_to_values: A mapping of FX node names to their produced ONNX ``Value``.
        graph_like: The ONNX graph at construction.
    """
    if not isinstance(node.args[0], Sequence):
        output_nodes = (node.args[0],)
    else:
        # node.args[0] can be a tuple with more than one elements. This happens when,
        # for example, a subgraph has multiple outputs. We flatten them all as ONNX graph outputs
        output_nodes = node.args[0]
    for output in output_nodes:
        if output is None:
            logger.warning(
                "Output node %s has None output. The output is ignored in the exported graph. Please ensure the graph output order is expected",
                node.name,
            )
            continue
        output_value_name = output.name  # type: ignore[union-attr]
        if not isinstance(output_value_name, str):
            raise AssertionError(f"Bug: Expected {output_value_name!r} to be a string")
        values = node_name_to_values[output_value_name]
        if isinstance(values, Sequence):
            graph_like.outputs.extend(values)
            return
        graph_like.outputs.append(values)


def _translate_fx_graph(
    fx_graph: torch.fx.Graph,
    model: ir.Model,
    *,
    graph_like: ir.Graph | ir.Function,
    owned_graphs: Mapping[str, ir.Function],
    lower: Literal["at_conversion", "none"],
    registry: _registration.ONNXRegistry,
) -> dict[str, ir.Value | Sequence[ir.Value]]:
    """Translate a submodule to an ONNX function.

    Any functions used by the traced functions will be added to the model.

    Args:
        fx_graph: The FX graph module to translate.
        model: The ONNX model at construction.
        current_scope: The current name scope of the submodule, excluding the current module name.
            E.g. "true_graph_0.false_graph_0".
        graph_name: The name of the submodule. E.g. "true_graph_0".
        graph: The ONNX graph at construction.
        owned_graphs: The subgraphs owned by the current graph.
        lower: The lowering strategy to use.
        registry: The registry of all aten to ONNX decomposition functions.

    Returns:
        A mapping of FX node names to their produced ONNX ``Value``.
    """
    node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]] = {}
    # The reason we need node_name_to_local_functions in addition to owned_graphs
    # is because the get_attr nodes may assign a different name than the GraphModule name
    # to the subgraph. This is not typical but is valid Python.
    node_name_to_local_functions: dict[str, ir.Function] = {}
    constant_farm: dict[Any, ir.Value] = {}
    opset = _get_onnxscript_opset(registry.opset_version)

    for node in fx_graph.nodes:
        logger.debug(
            "%s", (node.name, node.args, node.target, node.op, node.type, node.kwargs)
        )
        try:
            if node.op == "placeholder":
                _handle_placeholder_node(
                    node,
                    node_name_to_values,
                    graph_like=graph_like,
                    lower=lower,
                    opset=opset,
                )
            elif node.op == "call_function":
                if lower == "at_conversion":
                    _handle_call_function_node_with_lowering(
                        model,
                        node,
                        node_name_to_values,
                        graph_like=graph_like,
                        constant_farm=constant_farm,
                        registry=registry,
                        opset=opset,
                        node_name_to_local_functions=node_name_to_local_functions,
                    )
                else:
                    # No lowering
                    _handle_call_function_node(graph_like, node, node_name_to_values)
            elif node.op == "get_attr":
                if isinstance(node.target, str) and node.target not in owned_graphs:
                    # Nested invoke_subgraph tracing can leave lifted tensor
                    # constants attached as get_attr nodes that are not
                    # registered subgraphs. Materialize them so the exported
                    # model has the data available at save time.
                    tensor_value = node.meta.get("val", None)
                    if tensor_value is None:
                        tensor_value = node.meta.get("example_value", None)
                    if not isinstance(tensor_value, torch.Tensor):
                        raise KeyError(
                            f"get_attr node {node.name!r} is neither an owned "
                            f"subgraph nor a materializable tensor constant "
                            f"(target={node.target!r}, meta val type="
                            f"{type(tensor_value).__name__})"
                        )
                    value = ir.Value(name=node.name)
                    value.const_value = TorchTensor(tensor_value, name=node.name)
                    _set_shape_type(
                        value, tensor_value, complex_to_float=lower != "none"
                    )
                    if isinstance(graph_like, ir.Graph):
                        # Root graph: add as a named initializer.
                        model.graph.initializers[node.name] = value
                    else:
                        # ONNX function scope: ONNX functions cannot reference
                        # outer-scope initializers, so emit a Constant node
                        # inline so the tensor is self-contained.
                        const_node = ir.Node(
                            "",
                            "Constant",
                            inputs=[],
                            attributes=[
                                ir.Attr(
                                    "value",
                                    ir.AttributeType.TENSOR,
                                    value.const_value,
                                )
                            ],
                            outputs=[value],
                            name=node.name,
                        )
                        graph_like.append(const_node)
                    node_name_to_values[node.name] = value
                else:
                    _handle_get_attr_node(
                        node,
                        owned_graphs=owned_graphs,
                        node_name_to_local_functions=node_name_to_local_functions,
                    )
            elif node.op == "output":
                _handle_output_node(
                    node,
                    node_name_to_values,
                    graph_like=graph_like,
                )
        except Exception as e:
            raise _errors.ConversionError(
                f"Error when translating node {node.format_node()}. See the stack trace for more information."
            ) from e
    return node_name_to_values


def _get_inputs_and_attributes(
    node: torch.fx.Node,
) -> tuple[list[torch.fx.Node | None], dict[str, Any], list[str], list[str]]:
    """Find and Fill in the not provided kwargs with default values.

    Returns:
        (inputs, attributes, input_names, output_names)
    """
    if inspect.isbuiltin(node.target) or isinstance(node.target, str):
        inputs = list(node.args)
        return inputs, {}, [], [node.name]  # type: ignore[return-value]

    # The target should be an ATen operator now
    if not hasattr(node.target, "_schema"):
        raise AssertionError(
            f"The target should be an ATen operator now, but node target {node.target} has no schema"
        )
    node_schema: torch.FunctionSchema = node.target._schema

    # This function assumes the order of arguments in FX op is the
    # same as the order of arguments in TorchScript op.
    inputs: list[Any] = []  # type: ignore[no-redef]
    input_names: list[str] = []
    attributes: dict[str, Any] = {}

    if inspect.isbuiltin(node.target):
        inputs = list(node.args)
    else:
        for arg, schema_arg in zip(node.args, node_schema.arguments):
            if arg is None or isinstance(arg, torch.fx.Node):
                inputs.append(arg)
                input_names.append(schema_arg.name)
            elif isinstance(arg, Sequence) and all(
                elem is None or isinstance(elem, torch.fx.Node) for elem in arg
            ):
                inputs.extend(arg)
                input_names.extend([schema_arg.name] * len(arg))
            elif isinstance(arg, torch.device):
                attributes[schema_arg.name] = str(arg)
            elif isinstance(arg, torch.dtype):
                attributes[schema_arg.name] = torch_dtype_to_onnx_dtype(arg)
            else:
                attributes[schema_arg.name] = arg
        for schema_arg in node_schema.arguments:
            if schema_arg.name not in node.kwargs:
                continue
            kwarg = node.kwargs[schema_arg.name]
            if schema_arg.name in {
                "layout",
                "device",
                "requires_grad",
                "memory_format",
                "implicit",
            } or isinstance(kwarg, torch.device):
                attr = str(kwarg)
            elif isinstance(kwarg, torch.dtype):
                attr = torch_dtype_to_onnx_dtype(kwarg)  # type: ignore[assignment]
            else:
                attr = kwarg  # type: ignore[assignment]

            attributes[schema_arg.name] = attr

    output_names = [f"{node.name}_{output.name}" for output in node_schema.returns]

    return inputs, attributes, input_names, output_names  # type: ignore[return-value]


def _maybe_start_profiler(should_profile: bool) -> Any:
    if should_profile:
        import pyinstrument  # type: ignore[import-not-found]

        profiler = pyinstrument.Profiler(async_mode="disabled")
        profiler.start()
        return profiler
    return None


def _maybe_stop_profiler_and_get_result(profiler) -> str | None:
    if profiler is None:
        return None
    profiler.stop()
    return profiler.output_text(unicode=True)


def _format_exception(e: Exception) -> str:
    """Format the full traceback as Python would show it."""
    return "\n".join(traceback.format_exception(type(e), e, e.__traceback__))


def _summarize_exception_stack(e: BaseException) -> str:
    """Format the exception stack by showing the text of each exception."""
    causes = [e]
    while e.__cause__ is not None:
        causes.append(e.__cause__)
        e = e.__cause__
    return (
        "\n\n## Exception summary\n\n"
        + "⬆️\n".join([f"{type(e)}: {e}\n" for e in reversed(causes)])
        + "\n(Refer to the full stack trace above for more information.)"
    )


def _format_exceptions_for_all_strategies(
    results: list[_capture_strategies.Result],
) -> str:
    """Format all the exceptions from the capture strategies."""
    return "\n".join(
        [
            f"# ⚠️ Errors from strategy '{result.strategy}': -----------------------\n\n"
            f"{_format_exception(result.exception)}\n"
            for result in results
            if result.exception is not None
        ]
    )


def exported_program_to_ir(
    exported_program: torch.export.ExportedProgram,
    *,
    registry: _registration.ONNXRegistry | None = None,
    lower: Literal["at_conversion", "none"] = "at_conversion",
) -> ir.Model:
    """Convert an exported program to an ONNX IR model.

    Reference:
        - ExportedProgram spec: https://pytorch.org/docs/stable/export.html

    Args:
        exported_program: The exported program to convert.
        lower: Whether to lower the graph to core ONNX operators.
            at_conversion: Lower when translating the FX graph to ONNX IR.
            none: Do not lower the graph.
        registry: The registry of all ONNX Script decomposition.
    """
    if registry is None:
        registry = _registration.ONNXRegistry.from_torchlib()
    if lower != "none":
        exported_program = _prepare_exported_program_for_export(
            exported_program, registry=registry
        )
    return _exported_program_to_onnx_program(
        exported_program, registry=registry, lower=lower
    ).model


def _prepare_exported_program_for_export(
    exported_program: torch.export.ExportedProgram,
    *,
    registry: _registration.ONNXRegistry,
) -> torch.export.ExportedProgram:
    """Decompose and apply pre-export transformations to the exported program."""

    with (
        # Support the dynamism with 0/1 input dim
        torch.fx.experimental._config.patch(backed_size_oblivious=True),  # type: ignore[attr-defined]
    ):
        # Decompose the graph given the implemented torch ops in ONNX
        exported_program = _fx_passes.decompose_with_registry(
            exported_program, registry
        )

        graph_module = exported_program.graph_module
        # Include explicit type promotion nodes
        _fx_passes.insert_type_promotion_nodes(graph_module)
        graph_module = _fx_passes.remove_assertion_nodes(graph_module)
        # Reassign the graph module to save some runtime.
        exported_program._graph_module = graph_module
        return exported_program


def _get_scope_name(scoped_name: str) -> tuple[str, str]:
    """Get the scope and name of a node.

    Examples::
        >>> _get_scope_name('')
        ('', '')
        >>> _get_scope_name('true_graph')
        ('', 'true_graph')
        >>> _get_scope_name('true_graph.false_graph')
        ('true_graph', 'false_graph')
        >>> _get_scope_name('true_graph.false_graph.some_graph')
        ('true_graph.false_graph', 'some_graph')

    Args:
        scoped_name: The scoped name of the node.

    Returns:
        (scope, name)
    """
    if "." in scoped_name:
        scope, name = scoped_name.rsplit(".", 1)
    else:
        scope, name = "", scoped_name
    return scope, name


def _exported_program_to_onnx_program(
    exported_program: torch.export.ExportedProgram,
    *,
    registry: _registration.ONNXRegistry,
    lower: Literal["at_conversion", "none"] = "at_conversion",
) -> _onnx_program.ONNXProgram:
    """Convert an exported program to an ONNX Program.

    The exported_program field in the returned ONNXProgram is one that is after
    decompositions have been applied.

    Reference:
        - ExportedProgram spec: https://pytorch.org/docs/stable/export.html

    Args:
        exported_program: The exported program to convert. The exported program
            should be the one that is after decompositions have been applied.
        lower: Whether to lower the graph to core ONNX operators.
            at_conversion: Lower when translating the FX graph to ONNX IR.
            none: Do not lower the graph.
        registry: The registry of all ONNX Script decomposition.
    """
    model = ir.Model(
        graph=ir.Graph(
            [],
            [],
            nodes=[],
            # Opset imports are added to the model in the final add_opset_imports pass
            name="main_graph",
            metadata_props={
                "pkg.torch.export.ExportedProgram.graph_signature": str(
                    exported_program.graph_signature
                ),
                "pkg.torch.export.ExportedProgram.range_constraints": str(
                    exported_program.range_constraints
                ),
            },
        ),
        ir_version=_constants.ONNX_IR_VERSION,
        producer_name="pytorch",
        producer_version=torch.__version__,
    )

    # A dictionary storing the translated subgraphs as ONNX functions made available to outer graphs
    # {<subgraph_scope>: {<subgraph_name>: <IR function>}}
    scoped_subgraphs: dict[str, dict[str, ir.Function]] = {}
    values = None

    # 1. Translate all nodes in all subgraphs and the main graph
    # Create a dictionary of values for the main graph for step 2-3 to add inputs and outputs
    module: torch.fx.GraphModule
    # Reverse the order of the modules so that the innermost module is processed first
    # and made available to the outer module
    for name, module in reversed(
        tuple(exported_program.graph_module.named_modules(remove_duplicate=False))
    ):
        # Obtain the graphs (previously built) owned by the current module
        owned_graphs = scoped_subgraphs.setdefault(name, {})
        fx_graph = module.graph

        graph_like: ir.Graph | ir.Function
        if name == "":
            # Root graph
            graph_like = model.graph
        else:
            function_name = name.replace(".", "__")
            # Inputs and outputs will be created within _translate_fx_graph
            func = ir.Function(
                domain=_constants.LOCAL_FUNCTION_DOMAIN,
                name=function_name,
                graph=ir.Graph((), (), nodes=()),
                attributes=(),
            )
            # Make this function available to the outer graph
            scope, subgraph_name = _get_scope_name(name)
            scoped_subgraphs.setdefault(scope, {})[subgraph_name] = func
            model.functions[func.identifier()] = func
            graph_like = func

        values = _translate_fx_graph(
            fx_graph,
            model,
            graph_like=graph_like,
            owned_graphs=owned_graphs,
            lower=lower,
            registry=registry,
        )

    if name != "":
        raise AssertionError("The last module processed should be the root module")
    if values is None:
        raise AssertionError("values must be non-None")

    # Clear the input/output of the main graph and add them back in step 2-3
    # using the more accurate graph signature
    model.graph.inputs.clear()
    model.graph.outputs.clear()

    # 2. Add user inputs and all parameters/buffers to the graph.
    # Since the node names and the tensor names are different, we need to rename
    # the nodes to match the tensor names later. For now we will just use the node names.
    user_inputs = [
        spec
        for spec in exported_program.graph_signature.input_specs
        if spec.kind == graph_signature.InputKind.USER_INPUT
    ]
    non_user_inputs = [
        spec
        for spec in exported_program.graph_signature.input_specs
        if spec.kind != graph_signature.InputKind.USER_INPUT
    ]

    for spec in itertools.chain(user_inputs, non_user_inputs):
        # Put the user inputs first and then the parameters/buffers
        if isinstance(spec.arg, graph_signature.ConstantArgument):
            logger.debug("Skipping constant argument %s", spec.arg)
            continue
        value_name = spec.arg.name
        input_kind = spec.kind
        persistent = spec.persistent
        value = values[value_name]

        if isinstance(value, Sequence):
            raise AssertionError(
                f"Input '{value_name}' should not be a sequence. This is unexpected."
            )

        value.metadata_props["pkg.torch.export.graph_signature.InputSpec.kind"] = (
            input_kind.name
        )
        value.metadata_props[
            "pkg.torch.export.graph_signature.InputSpec.persistent"
        ] = str(persistent)

        if input_kind == graph_signature.InputKind.USER_INPUT:
            # Add only user inputs to the graph
            # Subsequent passes can decide if they want to add initializers as inputs
            model.graph.inputs.append(value)
        else:
            model.graph.initializers[value_name] = value

    # 3. Add user outputs to the graph and assign metadata to all outputs
    user_outputs = [
        spec
        for spec in exported_program.graph_signature.output_specs
        if spec.kind == graph_signature.OutputKind.USER_OUTPUT
    ]
    non_user_outputs = [
        spec
        for spec in exported_program.graph_signature.output_specs
        if spec.kind != graph_signature.OutputKind.USER_OUTPUT
    ]
    for spec in itertools.chain(user_outputs, non_user_outputs):
        if isinstance(spec.arg, graph_signature.ConstantArgument):
            logger.warning("Skipping constant argument %s", spec.arg)
            continue
        value_name = spec.arg.name
        output_kind = spec.kind
        value = values[value_name]

        if not isinstance(value, (ir.Value, Sequence)):
            raise TypeError(
                f"Output '{value_name}' should be an ir.Value. Actual type is '{type(value)}': {value!r}. "
                "This may be due to an incorrect implementation of the ONNX function that produced this output."
            )

        # The output value may be a sequence, meaning the operator has multiple outputs
        _values = (value,) if not isinstance(value, Sequence) else value

        if len(_values) > 1:
            logger.warning(
                "Model output '%s' has multiple values: %s (output spec: %s). Please make sure this is expected.",
                value_name,
                _values,
                spec,
            )

        for value in _values:
            value.metadata_props["pkg.torch.export.graph_signature.OutputSpec.kind"] = (
                output_kind.name
            )
            if output_kind == graph_signature.OutputKind.USER_OUTPUT:
                model.graph.outputs.append(value)

    # 4. Rename the initializers to match the tensor names
    for name, param_name in itertools.chain(
        exported_program.graph_signature.inputs_to_parameters.items(),
        exported_program.graph_signature.inputs_to_buffers.items(),
        exported_program.graph_signature.inputs_to_lifted_tensor_constants.items(),
    ):
        initializer = model.graph.initializers.pop(name)
        initializer.name = param_name
        # Record the original name so users can search the metadata and correspond
        # with the FX graph
        initializer.metadata_props["pkg.torch.onnx.original_node_name"] = name
        model.graph.initializers[param_name] = initializer

    # 5. Add initializers to the graph
    # ExportedProgram stores parameters and buffers in state_dict,
    # but non_persistent_buffers and lifted_tensor_constants are not there
    # so we need to get them from the name_* apis.
    for name, torch_tensor in itertools.chain(
        exported_program.named_parameters(),
        # pyrefly: ignore [bad-argument-type]
        exported_program.named_buffers(),
        exported_program.constants.items(),
    ):
        initializer = model.graph.initializers.get(name)  # type: ignore[assignment]
        if initializer is None:
            logger.warning("Tensor '%s' is not one of the initializers", name)
            continue
        if not isinstance(torch_tensor, torch.Tensor):
            raise NotImplementedError(
                f"Tensor '{name}' should be a torch.Tensor. Actual type is '{type(torch_tensor)}': {torch_tensor!r}. "
                "This is unexpected and not yet supported."
            )

        # Turn complex tensors into float tensors when converting to ONNX
        complex_to_float = lower != "none"
        if complex_to_float:
            if torch_tensor.dtype.is_complex:
                torch_tensor = torch.view_as_real(torch_tensor)

        ir_tensor = TorchTensor(torch_tensor, name=name)
        initializer.const_value = ir_tensor
        _set_shape_type(
            initializer,
            torch_tensor,
            complex_to_float=complex_to_float,
        )

    # TODO: Decide if we should keep mutated buffers as inputs/outputs

    # Collect and add opset imports to the model
    _ir_passes.add_opset_imports(model)

    return _onnx_program.ONNXProgram(model, exported_program)


def _verbose_printer(verbose: bool | None) -> Callable[..., None]:
    """Prints messages based on `verbose`."""
    if verbose is False:
        return lambda *_, **__: None

    return lambda *args, **kwargs: print("[torch.onnx]", *args, **kwargs)


@_flags.set_onnx_exporting_flag
def export(
    model: torch.nn.Module
    | torch.export.ExportedProgram
    | torch.fx.GraphModule
    | torch.jit.ScriptModule
    | torch.jit.ScriptFunction,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    registry: _registration.ONNXRegistry | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    report: bool = False,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    verbose: bool | None = None,
    optimize: bool = True,
    opset_version: int | None = None,
) -> _onnx_program.ONNXProgram:
    """Export a PyTorch model to ONNXProgram.

    Args:
        model: The model to export. This can be a PyTorch nn.Module or an ExportedProgram.
        args: The arguments to pass to the model.
        kwargs: The keyword arguments to pass to the model.
        registry: The registry of all ONNX decompositions.
        dynamic_shapes: Dynamic shapes in the graph.
        input_names: If provided, rename the inputs.
        output_names: If provided, rename the outputs.
        report: Whether to generate an error report if the export fails.
        verify: Whether to verify the ONNX model after exporting.
        profile: Whether to profile the export process. When report is True,
            the profile result will be saved in the report. Otherwise, the profile
            result will be printed.
        dump_exported_program: Whether to save the exported program to a file.
        artifacts_dir: The directory to save the exported program and error reports.
        verbose: Whether to print verbose messages. If None (default), some messages will be printed.
        optimize: Whether to optimize the exported ONNX graph.
        opset_version: The ONNX opset version to use. If None, use the default opset version
            from the registry.

    Returns:
        The ONNXProgram with the exported IR graph.

    Raises:
        TorchExportError: If the export process fails with torch.export.
        ConversionError: If the ExportedProgram to ONNX translation fails.
    """
    # Set up the error reporting facilities
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    profiler = _maybe_start_profiler(profile)

    # Create the artifacts directory if it does not exist
    artifacts_dir = pathlib.Path(artifacts_dir)
    if report or profile or dump_exported_program:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    verbose_print = _verbose_printer(verbose)
    export_status = _reporting.ExportStatus()
    failed_results: list[_capture_strategies.Result] = []

    program: torch.export.ExportedProgram | None = None
    capture_strategy: str | None = None
    # Step 1: Export the model with torch.export.export if the model is not already an ExportedProgram
    if isinstance(model, torch.export.ExportedProgram):
        # We know the model is already exported program, so the args, kwargs, and dynamic_shapes
        # are not used.
        program = model
        # torch.export.export has strict default to False
        export_status.torch_export_non_strict = True
    else:
        # Convert an nn.Module to an ExportedProgram
        # Try everything 🐰 (all paths for getting an ExportedProgram)
        result: _capture_strategies.Result | None = None
        for strategy_class in _capture_strategies.CAPTURE_STRATEGIES:
            strategy = strategy_class(  # type: ignore[abstract]
                verbose=verbose is not False,  # Treat None as verbose
                dump=dump_exported_program,
                artifacts_dir=artifacts_dir,
                timestamp=timestamp,
            )
            result = strategy(model, args, kwargs, dynamic_shapes=dynamic_shapes)

            # Record the status
            if strategy_class is _capture_strategies.TorchExportNonStrictStrategy:
                export_status.torch_export_non_strict = result.success
            elif strategy_class is _capture_strategies.TorchExportStrictStrategy:
                export_status.torch_export_strict = result.success
            elif strategy_class is _capture_strategies.TorchExportDraftExportStrategy:
                export_status.torch_export_draft_export = result.success

            if result.exception is not None:
                failed_results.append(result)
            if result.success:
                if result.exported_program is None:
                    raise AssertionError("exported_program must be non-None on success")
                program = result.exported_program
                break

        if result is None:
            raise AssertionError("result must be non-None")
        capture_strategy = result.strategy
        if result.exported_program is None:
            # If all strategies fail, produce an error report and raise the first error
            profile_result = _maybe_stop_profiler_and_get_result(profiler)

            if report:
                report_path = artifacts_dir / _reporting.construct_report_file_name(
                    timestamp, export_status
                )

                try:
                    _reporting.create_torch_export_error_report(
                        report_path,
                        _format_exceptions_for_all_strategies(failed_results),
                        export_status=export_status,
                        profile_result=profile_result,
                    )
                except Exception as e_report:
                    verbose_print(
                        f"Failed to save error report due to an error: {e_report}"
                    )
            else:
                report_path = None

            first_error = failed_results[0].exception
            if first_error is None:
                raise AssertionError("first_error must be non-None")

            # NOTE: We only throw the torch.export (first) exception because we want to
            # focus on the torch.export.export error. Errors from other strategies like
            # torch.jit.trace is due to the fallback and can be confusing to users.
            # We save all errors in the error report.
            raise _errors.TorchExportError(
                _STEP_ONE_ERROR_MESSAGE
                + (
                    f"\nError report has been saved to '{report_path}'."
                    if report
                    else ""
                )
                + _summarize_exception_stack(first_error)
            ) from first_error

    if program is None:
        raise AssertionError("program must be non-None")

    if dump_exported_program:
        verbose_print("Dumping ExportedProgram because `dump_exported_program=True`...")
        program_path = artifacts_dir / f"onnx_export_{timestamp}.pt2"
        try:
            torch.export.save(program, program_path)
        except Exception as e:
            verbose_print(f"Failed to save ExportedProgram due to an error: {e}")
        else:
            verbose_print(f"ExportedProgram has been saved to '{program_path}'.")

    # Step 2: Decompose the exported program and insert type promotion nodes
    verbose_print("Run decompositions...")

    try:
        # Build the ONNX function registry
        if registry is None:
            registry = _registration.ONNXRegistry.from_torchlib()

        # Process the exported program to run decompositions and type promotions etc.
        decomposed_program = _prepare_exported_program_for_export(
            program, registry=registry
        )
    except Exception as e:
        export_status.decomposition = False
        verbose_print("Run decompositions... ❌")
        profile_result = _maybe_stop_profiler_and_get_result(profiler)

        if report:
            report_path = artifacts_dir / _reporting.construct_report_file_name(
                timestamp, export_status
            )

            # Run the analysis to get the error report
            try:
                _reporting.create_onnx_export_report(
                    report_path,
                    f"{_format_exceptions_for_all_strategies(failed_results)}\n\n{_format_exception(e)}",
                    program,
                    export_status=export_status,
                    profile_result=profile_result,
                    registry=registry,
                )
            except Exception:
                logger.exception("Failed to save report due to an error.")
        else:
            report_path = None

        raise _errors.ConversionError(
            _STEP_TWO_ERROR_MESSAGE
            + (f"\nError report has been saved to '{report_path}'." if report else "")
            + _summarize_exception_stack(e)
        ) from e
    else:
        export_status.decomposition = True
        verbose_print("Run decompositions... ✅")

    # Step 3: Translate the decomposed program to ONNX and produce ONNXProgram
    verbose_print("Translate the graph into ONNX...")
    if report or profile:
        pre_decomp_unique_ops, post_decomp_unique_ops = _analysis.compare_ops(
            program, decomposed_program
        )
    else:
        pre_decomp_unique_ops = None
        post_decomp_unique_ops = None

    try:
        # Convert the exported program to an ONNX model
        onnx_program = _exported_program_to_onnx_program(
            decomposed_program, registry=registry
        )
        # Record the strategy used for getting the exported program for unit test assertions
        onnx_program._capture_strategy = capture_strategy

        export_status.onnx_translation = True
        verbose_print("Translate the graph into ONNX... ✅")
    except Exception as e:
        export_status.onnx_translation = False
        verbose_print("Translate the graph into ONNX... ❌")
        profile_result = _maybe_stop_profiler_and_get_result(profiler)

        if report:
            report_path = artifacts_dir / _reporting.construct_report_file_name(
                timestamp, export_status
            )

            try:
                if pre_decomp_unique_ops is None:
                    raise AssertionError("pre_decomp_unique_ops must be non-None")
                if post_decomp_unique_ops is None:
                    raise AssertionError("post_decomp_unique_ops must be non-None")

                # Run the analysis to get the error report
                _reporting.create_onnx_export_report(
                    report_path,
                    f"{_format_exceptions_for_all_strategies(failed_results)}\n\n{_format_exception(e)}",
                    decomposed_program,
                    decomp_comparison=_reporting.format_decomp_comparison(
                        pre_decomp_unique_ops, post_decomp_unique_ops
                    ),
                    export_status=export_status,
                    profile_result=profile_result,
                    registry=registry,
                )
                verbose_print(f"Export report has been saved to '{report_path}'.")
            except Exception:
                logger.exception("Failed to save report due to an error.")
        else:
            report_path = None

        raise _errors.ConversionError(
            _STEP_THREE_ERROR_MESSAGE
            + (f"\nError report has been saved to '{report_path}'." if report else "")
            + _summarize_exception_stack(e)
        ) from e

    profile_result = _maybe_stop_profiler_and_get_result(profiler)

    if onnx_program.exported_program is None:
        raise AssertionError("exported_program must be non-None")

    # Converter opset version and optimize
    if opset_version is not None:
        onnx_program.model = onnxscript_apis.convert_version(
            onnx_program.model, opset_version
        )

    if optimize:
        verbose_print("Optimize the ONNX graph...")
        onnx_program.optimize()
        verbose_print("Optimize the ONNX graph... ✅")

    # Run the ONNX passes
    if input_names:
        _ir_passes.rename_inputs(onnx_program.model, input_names)
    if output_names:
        _ir_passes.rename_outputs(onnx_program.model, output_names)

    if not verify:
        # Return if verification is not requested
        if report:
            try:
                if pre_decomp_unique_ops is None:
                    raise AssertionError("pre_decomp_unique_ops must be non-None")
                if post_decomp_unique_ops is None:
                    raise AssertionError("post_decomp_unique_ops must be non-None")
                report_path = artifacts_dir / _reporting.construct_report_file_name(
                    timestamp, export_status
                )
                _reporting.create_onnx_export_report(
                    report_path,
                    "No errors"
                    if not failed_results
                    else _format_exceptions_for_all_strategies(failed_results),
                    onnx_program.exported_program,
                    decomp_comparison=_reporting.format_decomp_comparison(
                        pre_decomp_unique_ops, post_decomp_unique_ops
                    ),
                    export_status=export_status,
                    profile_result=profile_result,
                    model=onnx_program.model,
                    registry=registry,
                )
                verbose_print(f"Export report has been saved to '{report_path}'.")
            except Exception:
                logger.exception("Failed to save report due to an error.")
        elif profile and profile_result is not None:
            verbose_print("Profile result:")
            verbose_print(profile_result)
        return onnx_program

    # Step 4: (verify=True) Check the ONNX model with ONNX checker
    try:
        verbose_print("Check the ONNX model...")
        onnxscript_apis.check_model(onnx_program.model)
        export_status.onnx_checker = True
        verbose_print("Check the ONNX model... ✅")
    except Exception as e:
        export_status.onnx_checker = False
        verbose_print("Check the ONNX model... ❌")
        if report:
            try:
                if pre_decomp_unique_ops is None:
                    raise AssertionError("pre_decomp_unique_ops must be non-None")
                if post_decomp_unique_ops is None:
                    raise AssertionError("post_decomp_unique_ops must be non-None")
                report_path = artifacts_dir / _reporting.construct_report_file_name(
                    timestamp, export_status
                )
                _reporting.create_onnx_export_report(
                    report_path,
                    f"{_format_exceptions_for_all_strategies(failed_results)}\n\n{_format_exception(e)}",
                    onnx_program.exported_program,
                    decomp_comparison=_reporting.format_decomp_comparison(
                        pre_decomp_unique_ops, post_decomp_unique_ops
                    ),
                    export_status=export_status,
                    profile_result=profile_result,
                    model=onnx_program.model,
                    registry=registry,
                )
                verbose_print(f"Export report has been saved to '{report_path}'.")
            except Exception:
                logger.exception("Failed to save report due to an error.")
        logger.warning(
            "Conversion successful but the ONNX model fails ONNX checker. "  # noqa: G004
            "Please create an issue "
            f"in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component and "
            "attach the full error stack as well as reproduction scripts. ",
            exc_info=e,
        )
        return onnx_program

    # Step 5: (verify=True) Execute the model with ONNX Runtime
    try:
        verbose_print("Execute the model with ONNX Runtime...")
        verification_results = _verification.verify_onnx_program(onnx_program)
        verbose_print("Execute the model with ONNX Runtime... ✅")
        export_status.onnx_runtime = True
        onnx_runtime_error_message = None
    except Exception as e:
        verbose_print("Execute the model with ONNX Runtime... ❌")
        export_status.onnx_runtime = False
        onnx_runtime_error_message = _format_exception(e)
        verification_message = None

    else:
        # Step 6: (verify=True) Validate the output values
        verbose_print("Verify output accuracy...")
        export_status.output_accuracy = True
        for verification_result in verification_results:
            # TODO(justinchuby): The threshold is arbitrary right now
            if verification_result.max_abs_diff >= 5e-3:
                logger.warning(
                    "Output '%s' has a large absolute difference of %f. ",
                    verification_result.name,
                    verification_result.max_abs_diff,
                )
                export_status.output_accuracy = False
            if verification_result.max_rel_diff >= 1e-1:
                logger.warning(
                    "Output '%s' has a large relative difference of %f. ",
                    verification_result.name,
                    verification_result.max_rel_diff,
                )
                export_status.output_accuracy = False
        if export_status.output_accuracy:
            verbose_print("Verify output accuracy... ✅")
        else:
            verbose_print("Verify output accuracy... ❌")
        verification_message = _reporting.format_verification_infos(
            verification_results
        )

    if report:
        try:
            if pre_decomp_unique_ops is None:
                raise AssertionError("pre_decomp_unique_ops must be non-None")
            if post_decomp_unique_ops is None:
                raise AssertionError("post_decomp_unique_ops must be non-None")

            traceback_lines = []
            if failed_results:
                traceback_lines.append(
                    _format_exceptions_for_all_strategies(failed_results)
                )
            if onnx_runtime_error_message:
                traceback_lines.append("# ⚠️ ONNX Runtime error -----------------------")
                traceback_lines.append(onnx_runtime_error_message)
            if not traceback_lines:
                traceback_lines.append("No errors")

            report_path = artifacts_dir / _reporting.construct_report_file_name(
                timestamp, export_status
            )
            _reporting.create_onnx_export_report(
                report_path,
                "\n\n".join(traceback_lines),
                onnx_program.exported_program,
                profile_result=profile_result,
                export_status=export_status,
                decomp_comparison=_reporting.format_decomp_comparison(
                    pre_decomp_unique_ops, post_decomp_unique_ops
                ),
                model=onnx_program.model,
                registry=registry,
                verification_result=verification_message,
            )
            verbose_print(f"Export report has been saved to '{report_path}'.")
        except Exception:
            logger.exception("Failed to save report due to an error.")

    # Release the inference session created during verification
    onnx_program.release()
    return onnx_program
<|endoftext|># mypy: allow-untyped-defs
# mypy: disable-error-code="attr-defined,name-defined"
from __future__ import annotations


__all__ = ["ONNXProgram"]

import contextlib
import copy
import gc
import logging
import os
import tempfile
import textwrap
import warnings
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

import torch
from torch.onnx._internal._lazy_import import onnx, onnx_ir as ir, onnxscript_apis
from torch.onnx._internal.exporter import _dynamic_shapes, _ir_passes
from torch.utils import _pytree


# NOTE: DO NOT import module from torch.onnx._internal to this module in the global scope
# because ONNXProgram is exposed to the public API

if TYPE_CHECKING:
    import numpy as np
    import onnxruntime as ort

_LARGE_MODEL_THRESHOLD = 1536 * 1024 * 1024  # 1536MB
_NP_UNSUPPORTED_DTYPES_8BIT = frozenset(
    {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
)

logger = logging.getLogger(__name__)


def _ort_session_initializer(model: str | bytes) -> ort.InferenceSession:
    """Initialize an ONNX Runtime inference session with the specified model."""
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # 3: Error
    possible_providers = (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    available_providers = set(ort.get_available_providers())
    providers = [
        provider for provider in possible_providers if provider in available_providers
    ]
    return ort.InferenceSession(
        model, providers=providers, sess_options=session_options
    )


def _count_initializer_size(graph: ir.Graph) -> int:
    """Count the total size of the initializers in bytes."""
    return sum(
        v.const_value.nbytes
        for v in graph.initializers.values()
        if v.const_value is not None
    )


@contextlib.contextmanager
def _set_graph_outputs(
    graph: ir.Graph,
    outputs: list[ir.Value],
):
    """Temporarily set the outputs of the graph.

    Args:
        graph: The graph to set the outputs for.
        outputs: The outputs to set.
    """
    original_outputs = list(graph.outputs)
    graph.outputs.clear()
    graph.outputs.extend(outputs)
    try:
        yield
    finally:
        graph.outputs.clear()
        graph.outputs.extend(original_outputs)


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


def _to_numpy_array(input: torch.Tensor | int | float | str | bool) -> np.ndarray:
    if isinstance(input, (int, float, str, bool)):
        return ir.tensor(input).numpy()

    from torch.onnx._internal.exporter import _core

    return _core.TorchTensor(input).numpy()


def _from_numpy_array(array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor."""
    import ml_dtypes  # type: ignore[import-not-found]
    import numpy as np

    if array.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    if array.dtype == ml_dtypes.float8_e4m3fn:
        return torch.from_numpy(array.view(np.uint8)).view(torch.float8_e4m3fn)
    if array.dtype == ml_dtypes.float8_e4m3fnuz:
        return torch.from_numpy(array.view(np.uint8)).view(torch.float8_e4m3fnuz)
    if array.dtype == ml_dtypes.float8_e5m2:
        return torch.from_numpy(array.view(np.uint8)).view(torch.float8_e5m2)
    if array.dtype == ml_dtypes.float8_e5m2fnuz:
        return torch.from_numpy(array.view(np.uint8)).view(torch.float8_e5m2fnuz)
    return torch.from_numpy(array)


def _to_ort_value(input: torch.Tensor | int | float | str | bool) -> ort.OrtValue:
    """Convert a PyTorch tensor to an ONNX Runtime OrtValue."""
    import numpy as np
    import onnxruntime as ort

    from torch.onnx._internal.exporter import _core

    if isinstance(input, (int, float, str, bool)):
        # Convert scalar values to OrtValue
        dtype_mapping = {
            int: np.int64,
            float: np.float32,
        }
        # pyrefly: ignore [bad-argument-type, no-matching-overload]
        dtype = dtype_mapping.get(type(input))
        return ort.OrtValue.ortvalue_from_numpy(np.array(input, dtype=dtype))

    if input.dtype == torch.bfloat16 or input.dtype in _NP_UNSUPPORTED_DTYPES_8BIT:
        if hasattr(ort.OrtValue, "ortvalue_from_numpy_with_onnx_type"):
            # This requires ONNX Runtime 1.21 or newer
            if input.dtype == torch.bfloat16:
                uint_type = torch.uint16
            else:
                uint_type = torch.uint8
            onnx_type = _core.torch_dtype_to_onnx_dtype(input.dtype)
            # Make tensor contiguous to ensure view() works
            input = input.contiguous()
            return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                input.view(uint_type).numpy(force=True), onnx_element_type=onnx_type
            )
        raise RuntimeError(
            f"Failed to convert tensor of type '{input.dtype}' to OrtValue. "
            "Please ensure that ONNX Runtime is built with DLPack support or is the latest version"
        )
    # TODO(#151064): Use dlpack when ORT properly supports it
    return ort.OrtValue.ortvalue_from_numpy(input.numpy(force=True))


def _from_ort_value(value: ort.OrtValue) -> torch.Tensor:
    if value.element_type() in (
        ir.DataType.BFLOAT16,
        ir.DataType.FLOAT8E4M3FN,
        ir.DataType.FLOAT8E4M3FNUZ,
        ir.DataType.FLOAT8E5M2,
        ir.DataType.FLOAT8E5M2FNUZ,
    ):
        # This requires ONNX Runtime 1.21 or newer
        try:
            return torch.from_dlpack(value._get_c_value())
        except Exception as e:
            raise RuntimeError(
                "Failed to convert OrtValue to torch.Tensor. "
                "Please ensure that ONNX Runtime is built with DLPack support or is the latest version"
            ) from e
    return torch.from_numpy(value.numpy())


class ONNXProgram:
    """A class to represent an ONNX program that is callable with torch tensors.

    Attributes:
        model: The ONNX model as an ONNX IR model object.
        exported_program: The exported program that produced the ONNX model.
    """

    def __init__(
        self, model: ir.Model, exported_program: torch.export.ExportedProgram | None
    ) -> None:
        """Initialize the ONNX program with the specified model and exported program.
        Args:
            model: The ONNX model.
            exported_program: The exported program that produced the ONNX model. Optional.
        """
        self.model: ir.Model = model
        self.exported_program = exported_program
        self._inference_session: ort.InferenceSession | None = None
        self._tempdir: tempfile.TemporaryDirectory | None = None
        # Strategy used to capture the exported program
        self._capture_strategy: str | None = None

    def __repr__(self) -> str:
        return f"""\
ONNXProgram(
    model=
{textwrap.indent(str(self.model), " " * 8)}
    ,
    exported_program=
{textwrap.indent(str(self.exported_program), " " * 8)}
)
"""

    def __call__(self, *args, **kwargs) -> Sequence[torch.Tensor]:
        """Run the ONNX model with the same arguments you would provide to the GraphModule."""
        import onnxruntime as ort

        flatten_args = _process_args(args, kwargs)

        if self._inference_session is None:
            self.initialize_inference_session()

        if self._inference_session is None:
            raise AssertionError("_inference_session must be non-None")

        ort_input = {
            k.name: _to_ort_value(v)
            for k, v in zip(self.model.graph.inputs, flatten_args)
        }
        run_options = ort.RunOptions()
        run_options.log_severity_level = 3  # 3: Error
        logger.debug("Running the inference session with %s arguments.", len(ort_input))
        outputs = self._inference_session.run_with_ort_values(
            None, ort_input, run_options=run_options
        )
        logger.debug("Inference session run completed.")
        return tuple(_from_ort_value(output) for output in outputs)

    def call_reference(self, *args, **kwargs) -> Sequence[torch.Tensor]:
        """Run the ONNX model using the reference backend."""
        import onnx.reference

        evaluator = onnx.reference.ReferenceEvaluator(self.model_proto)

        flatten_args = _process_args(args, kwargs)
        ref_input = {
            k.name: _to_numpy_array(v)
            for k, v in zip(self.model.graph.inputs, flatten_args)
        }
        outputs = evaluator.run(None, ref_input)  # type: ignore[arg-type]
        if not isinstance(outputs, Sequence):
            raise AssertionError(f"Expected Sequence, got {type(outputs)}")
        return tuple(_from_numpy_array(output) for output in outputs)

    def compute_values(
        self, value_names: Sequence[str], args=(), kwargs=None
    ) -> Sequence[torch.Tensor]:
        """Compute the values of the specified names in the ONNX model.

        This method is used to compute the values of the specified names in the ONNX model.
        The values are returned as a dictionary mapping names to tensors.

        Args:
            value_names: The names of the values to compute.

        Returns:
            A dictionary mapping names to tensors.
        """
        if kwargs is None:
            kwargs = {}
        self.release()
        values = _create_value_mapping(self.model.graph)
        for name in value_names:
            if name not in values:
                raise ValueError(
                    f"Value '{name}' not found in the model. "
                    "Please provide a valid value name."
                )
        temporary_outputs = [values[name] for name in value_names]
        with _set_graph_outputs(self.model.graph, temporary_outputs):
            try:
                result = self(*args, **kwargs)
            finally:
                self.release()
        return result

    @property
    def model_proto(self) -> onnx.ModelProto:
        """Return the ONNX ``ModelProto`` object."""
        return ir.serde.serialize_model(self.model)

    def optimize(self) -> None:
        """Optimize the ONNX model.

        This method optimizes the ONNX model by performing constant folding and
        eliminating redundancies in the graph. The optimization is done in-place.
        """
        self.model = onnxscript_apis.optimize(self.model)

    def save(
        self,
        destination: str | os.PathLike,
        *,
        include_initializers: bool = True,
        keep_initializers_as_inputs: bool = False,
        external_data: bool | None = None,
    ) -> None:
        """Save the ONNX model to the specified destination.

        When ``external_data`` is ``True`` or the model is larger than 2GB,
        the weights are saved as external data in a separate file.

        Initializer (model weights) serialization behaviors:

        * ``include_initializers=True``, ``keep_initializers_as_inputs=False`` (default):
          The initializers are included in the saved model.
        * ``include_initializers=True``, ``keep_initializers_as_inputs=True``:
          The initializers are included in the saved model and kept as model inputs.
          Choose this option if you want the ability to override the model weights
          during inference.
        * ``include_initializers=False``, ``keep_initializers_as_inputs=False``:
          The initializers are not included in the saved model and are not listed
          as model inputs. Choose this option if you want to attach the initializers
          to the ONNX model in a separate, post-processing, step.
        * ``include_initializers=False``, ``keep_initializers_as_inputs=True``:
          The initializers are not included in the saved model but are listed as model
          inputs. Choose this option if you want to supply the initializers during
          inference and want to minimize the size of the saved model.

        Args:
            destination: The path to save the ONNX model to.
            include_initializers: Whether to include the initializers in the saved model.
            keep_initializers_as_inputs: Whether to keep the initializers as inputs in the saved model.
                If `True`, the initializers are added as inputs to the model which means they can be overwritten.
                by providing the initializers as model inputs.
            external_data: Whether to save the weights as external data in a separate file.

        Raises:
            TypeError: If ``external_data`` is ``True`` and ``destination`` is not a file path.
        """
        original_initializers = copy.copy(self.model.graph.initializers)
        original_inputs = copy.copy(self.model.graph.inputs)

        # Adjust the model based on options
        if not include_initializers:
            self.model.graph.initializers.clear()
        if keep_initializers_as_inputs:
            self.model.graph.inputs.extend(original_initializers.values())  # type: ignore[arg-type]

        try:
            # Save the model to disk
            if (
                external_data
                or _count_initializer_size(self.model.graph) > _LARGE_MODEL_THRESHOLD
            ):
                onnxscript_apis.save_model_with_external_data(self.model, destination)
            else:
                ir.save(self.model, destination)
        finally:
            # Revert the changes to the model
            if not include_initializers:
                self.model.graph.initializers.update(original_initializers)
            if keep_initializers_as_inputs:
                self.model.graph.inputs.clear()
