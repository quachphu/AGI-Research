                    are_same_graph_modules,
                )

                if are_same_graph_modules(
                    fn_name, previous_mod, current_mod, tx.fake_mode
                ):
                    return submodule_name

        body_name = super().install_subgraph_in_output_graph(
            tx, fn_vt, fn_args_vt, kwargs, body_gmod, "subgraph"
        )
        hc_log.debug(
            "%s: Installing subgraph with identifier '%s', bringing total count for '%s' function to %s",
            fn_name,
            body_name,
            fn_name,
            len(previously_installed_submodules) + 1,
        )
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_dynamo_installed_submodule(fn_code, body_name)

        return body_name

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: "list[VariableTracker]",
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from torch._dynamo.utils import dynamo_timed
        from torch._dynamo.variables.higher_order_ops import (
            _call_function_with_auto_output_flattening,
        )

        fn_var = args[0]
        fn_args_vt = args[1:]

        config = None
        max_reuse_entries = 8
        reuse_hash_fn = None
        if hasattr(fn_var, "get_function"):
            try:
                fn = fn_var.get_function()
                config = getattr(fn, "__marked_compile_region_config__", None)
                max_reuse_entries = getattr(
                    fn, "__marked_compile_region_max_reuse_entries__", 8
                )
                reuse_hash_fn = getattr(
                    fn, "__marked_compile_region_reuse_hash_fn__", None
                )
            except Exception:
                log.warning(
                    "Failed to extract nested_compile_region() config from InvokeSubgraphHigherOrderVariable. ",
                    exc_info=True,
                )
                raise

        # TODO (anijain2305) - Collect issues why this does not work for export,
        # and enable if request arises.
        reuse = not tx.output.export

        # User-provided reuse_hash_fn path: hash key determines cache lookup.
        if reuse and reuse_hash_fn is not None:
            with dynamo_timed("invoke_subgraph_reuse_hash_fn"):
                hash_key = trace_reuse_hash_fn(tx, reuse_hash_fn, fn_args_vt, kwargs)

            cached = find_reuse_entry_by_key(tx, fn_var, hash_key)
            if cached is not None:
                hc_log.debug(
                    "subgraph_reuse: hash key %d hit for '%s', reusing subgraph '%s'",
                    hash_key,
                    fn_var,
                    cached.body_name,
                )
                fingerprint = build_input_fingerprint(tx, fn_args_vt, kwargs)
                with dynamo_timed("invoke_subgraph_reuse_stamp_out"):
                    return stamp_out_subgraph(tx, fingerprint, cached)

        # Automatic reuse lookup (guard-based): check fn_code first (cheap) to
        # avoid the expensive pytree flatten in build_input_fingerprint on
        # the first call when there's nothing in the cache yet.
        elif reuse and has_reuse_entries(tx, fn_var):
            with dynamo_timed("invoke_subgraph_reuse_lookup"):
                fingerprint = build_input_fingerprint(tx, fn_args_vt, kwargs)
                match = find_reuse_match(
                    tx,
                    fn_var,
                    fingerprint,
                )
            if match is not None:
                hc_log.debug(
                    "subgraph_reuse: cache hit for '%s', reusing subgraph '%s'",
                    fn_var,
                    match.body_name,
                )
                with dynamo_timed("invoke_subgraph_reuse_stamp_out"):
                    return stamp_out_subgraph(tx, fingerprint, match)

        if self._HOP_NAME is None:
            raise AssertionError("_HOP_NAME must not be None")
        with dynamo_timed("invoke_subgraph_trace"):
            (
                p_args,
                p_kwargs,
                example_value,
                body_r,
                body_gmod,
                body_name,
                body_graph_output_vts,
                tracing_info,
            ) = self.create_wrapped_node(tx, fn_var, fn_args_vt, kwargs, self._HOP_NAME)

        if len(p_kwargs) > 0:
            unimplemented(
                gb_type="invoke_subgraph: kwargs unexpected",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation="kwargs should have been flattened into lifted args.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        # Store config in the body graph module meta
        if isinstance(config, NestedCompileRegionOptions):
            body_gmod.meta["nested_region_config"] = config

        p_args = (
            p_args[0],
            body_name,
            *p_args[1:],
        )

        # Subgraph reuse: save entry for future cache hits
        if reuse:
            fingerprint = build_input_fingerprint(tx, fn_args_vt, kwargs)
            if reuse_hash_fn is not None:
                traced_sources = tracing_info.traced_sources
                if not is_reuse_eligible(
                    tx,
                    body_r,
                    fingerprint,
                    tracing_info,
                    traced_sources,
                    has_reuse_hash_fn=True,
                ):
                    raise RuntimeError(
                        "reuse_hash_fn was provided but the subgraph is not "
                        "eligible for reuse. Check the logs with "
                        "TORCH_LOGS='+hierarchical_compile' for details."
                    )
                save_reuse_entry(
                    tx,
                    fn_var,
                    fingerprint,
                    body_name,
                    body_gmod,
                    config,
                    p_args,
                    body_r,
                    example_value,
                    max_reuse_entries,
                    hash_key=hash_key,  # type: ignore[possibly-undefined]
                )
            else:
                traced_sources = tracing_info.traced_sources
                if is_reuse_eligible(
                    tx, body_r, fingerprint, tracing_info, traced_sources
                ):
                    condition = build_reuse_condition(
                        tx,
                        fingerprint,
                        traced_sources,
                    )
                    if condition is not None:
                        save_reuse_entry(
                            tx,
                            fn_var,
                            fingerprint,
                            body_name,
                            body_gmod,
                            config,
                            p_args,
                            body_r,
                            example_value,
                            max_reuse_entries,
                            condition=condition,
                        )

        return _call_function_with_auto_output_flattening(  # type: ignore[return-value]
            tx,
            torch._higher_order_ops.invoke_subgraph,
            tuple(p_args),
            p_kwargs,
            example_value,
            body_r,
            body_graph_output_vts,
            config=config,
        )
<|endoftext|>"""
Dynamo implementations of CPython's PyObject_* default slot algorithms.

Analogous to CPython's Objects/object.c, this module holds the general
dispatch machinery that is independent of any specific type.
Per-type hook implementations (nb_bool_impl, tp_richcompare_impl, tp_getattro_impl,
etc.) live in their respective VT files.
"""

import abc
import collections
import enum
import sys
import types
import typing
from functools import lru_cache, partial
from typing import NoReturn, TYPE_CHECKING

import torch
from torch._C._dynamo import (
    get_type_slots,
    has_slot,
    PyMappingSlots,
    PyNumberSlots,
    PySequenceSlots,
    PyTypeSlots,
)

from .. import graph_break_hints, polyfills, variables
from ..exc import (
    handle_observed_exception,
    ObservedTypeError,
    raise_observed_exception,
    raise_type_error,
    UnhandledDescriptorError,
    unimplemented,
)
from ..source import AttrSource, Source
from ..utils import istype
from .base import (
    AsPythonConstantNotImplementedError,
    AttrMutationKind,
    maybe_get_python_type,
    NO_SUCH_SUBOBJ,
    VariableTracker,
)
from .constant import ConstantVariable


if TYPE_CHECKING:
    from ..symbolic_convert import InstructionTranslatorBase


def vt_identity_compare(
    left: VariableTracker,
    right: VariableTracker,
) -> "VariableTracker | None":
    """Try to determine Python identity (left is right) at trace time.

    Returns ConstantVariable(True/False) if determinable, else None.
    Mirrors the logic in BuiltinVariable's handle_is handler.
    """
    if left is right:
        return ConstantVariable.create(True)

    left_val = left.get_real_python_backed_value()
    right_val = right.get_real_python_backed_value()
    left_known = left_val is not NO_SUCH_SUBOBJ
    right_known = right_val is not NO_SUCH_SUBOBJ

    if left_known and right_known:
        return (
            ConstantVariable.create(True)
            if left_val is right_val
            else ConstantVariable.create(False)
        )

    # One side has a concrete backing object, the other doesn't — they can't
    # be the same object.
    if left_known != right_known:
        return ConstantVariable.create(False)

    # Objects created during tracing: VT identity = Python identity.
    from .dicts import ConstDictVariable
    from .lists import ListVariable
    from .misc import TracebackVariable
    from .sets import SetVariable

    if isinstance(
        left, (ConstDictVariable, ListVariable, SetVariable, TracebackVariable)
    ):
        return ConstantVariable.create(False)

    # Different Python types can never be the same object.
    try:
        if left.python_type() is not right.python_type():
            return ConstantVariable.create(False)
    except NotImplementedError:
        pass

    # Different exception types are never identical.
    if (
        istype(left, variables.ExceptionVariable)
        and istype(right, variables.ExceptionVariable)
        and left.exc_type is not right.exc_type  # type: ignore[attr-defined]
    ):
        return ConstantVariable.create(False)

    return None


def binop_type_error(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    op_symbol: str,
) -> NoReturn:
    raise_type_error(
        tx,
        f"unsupported operand type(s) for {op_symbol}: '{v.python_type_name()}' and '{w.python_type_name()}'",
    )


@lru_cache(maxsize=256)
def _get_cached_slots(obj_type: type) -> tuple[int, int, int, int]:
    """Get all type slots for a type (cached)."""
    return get_type_slots(obj_type)


def type_implements_sq_slot(obj_type: type, slot: int) -> bool:
    """Check whether obj_type implements the given sq slot."""
    seq_slots, _, _, _ = _get_cached_slots(obj_type)
    return has_slot(seq_slots, slot)


def type_implements_mp_slot(obj_type: type, slot: int) -> bool:
    """Check whether obj_type implements the given mp slot."""
    _, map_slots, _, _ = _get_cached_slots(obj_type)
    return has_slot(map_slots, slot)


# PySequenceSlots
type_implements_sq_item = partial(type_implements_sq_slot, slot=PySequenceSlots.SQ_ITEM)
type_implements_sq_length = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_LENGTH
)
type_implements_sq_concat = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_CONCAT
)
type_implements_sq_inplace_concat = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_INPLACE_CONCAT
)
type_implements_sq_contains = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_CONTAINS
)
type_implements_sq_ass_item = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_ASS_ITEM
)
type_implements_sq_repeat = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_REPEAT
)

type_implements_sq_inplace_repeat = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_INPLACE_REPEAT
)

# PyMappingSlots
type_implements_mp_length = partial(
    type_implements_mp_slot, slot=PyMappingSlots.MP_LENGTH
)
type_implements_mp_subscript = partial(
    type_implements_mp_slot, slot=PyMappingSlots.MP_SUBSCRIPT
)
type_implements_mp_ass_subscript = partial(
    type_implements_mp_slot, slot=PyMappingSlots.MP_ASS_SUBSCRIPT
)
type_implements_mp_length = partial(
    type_implements_mp_slot, slot=PyMappingSlots.MP_LENGTH
)


def type_implements_nb_slot(obj_type: type, slot: int) -> bool:
    """Check whether obj_type implements the nb slot."""
    _, _, number_slots, _ = _get_cached_slots(obj_type)
    return has_slot(number_slots, slot)


type_implements_nb_add = partial(type_implements_nb_slot, slot=PyNumberSlots.NB_ADD)
type_implements_nb_subtract = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_SUBTRACT
)
type_implements_nb_multiply = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_MULTIPLY
)
type_implements_nb_remainder = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_REMAINDER
)
type_implements_nb_power = partial(type_implements_nb_slot, slot=PyNumberSlots.NB_POWER)
type_implements_nb_negative = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_NEGATIVE
)
type_implements_nb_positive = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_POSITIVE
)
type_implements_nb_absolute = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_ABSOLUTE
)
type_implements_nb_bool = partial(type_implements_nb_slot, slot=PyNumberSlots.NB_BOOL)
type_implements_nb_invert = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INVERT
)
type_implements_nb_lshift = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_LSHIFT
)
type_implements_nb_rshift = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_RSHIFT
)
type_implements_nb_and = partial(type_implements_nb_slot, slot=PyNumberSlots.NB_AND)
type_implements_nb_xor = partial(type_implements_nb_slot, slot=PyNumberSlots.NB_XOR)
type_implements_nb_or = partial(type_implements_nb_slot, slot=PyNumberSlots.NB_OR)
type_implements_nb_int = partial(type_implements_nb_slot, slot=PyNumberSlots.NB_INT)
type_implements_nb_float = partial(type_implements_nb_slot, slot=PyNumberSlots.NB_FLOAT)
type_implements_nb_inplace_add = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_ADD
)
type_implements_nb_inplace_subtract = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_SUBTRACT
)
type_implements_nb_inplace_multiply = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_MULTIPLY
)
type_implements_nb_inplace_remainder = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_REMAINDER
)
type_implements_nb_inplace_power = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_POWER
)
type_implements_nb_inplace_lshift = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_LSHIFT
)
type_implements_nb_inplace_rshift = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_RSHIFT
)
type_implements_nb_inplace_and = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_AND
)
type_implements_nb_inplace_xor = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_XOR
)
type_implements_nb_inplace_or = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_OR
)
type_implements_nb_floor_divide = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_FLOOR_DIVIDE
)
type_implements_nb_true_divide = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_TRUE_DIVIDE
)
type_implements_nb_inplace_floor_divide = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_FLOOR_DIVIDE
)
type_implements_nb_inplace_true_divide = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_TRUE_DIVIDE
)
type_implements_nb_index = partial(type_implements_nb_slot, slot=PyNumberSlots.NB_INDEX)
type_implements_nb_matrix_multiply = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_MATRIX_MULTIPLY
)
type_implements_nb_inplace_matrix_multiply = partial(
    type_implements_nb_slot, slot=PyNumberSlots.NB_INPLACE_MATRIX_MULTIPLY
)


def type_implements_tp_iter(obj_type: type) -> bool:
    _, _, _, type_slot = _get_cached_slots(obj_type)
    return has_slot(type_slot, PyTypeSlots.TP_ITER)


def type_implements_tp_iternext(obj_type: type) -> bool:
    _, _, _, type_slot = _get_cached_slots(obj_type)
    return has_slot(type_slot, PyTypeSlots.TP_ITERNEXT)


def type_implements_tp_repr(obj_type: type) -> bool:
    """Check whether obj_type implements the tp_repr slot."""
    _, _, _, type_slot = _get_cached_slots(obj_type)
    return has_slot(type_slot, PyTypeSlots.TP_REPR)


def type_implements_tp_str(obj_type: type) -> bool:
    """Check whether obj_type implements the tp_str slot."""
    _, _, _, type_slot = _get_cached_slots(obj_type)
    return has_slot(type_slot, PyTypeSlots.TP_STR)


def type_implements_tp_call(obj_type: type) -> bool:
    """Check whether obj_type implements the tp_call slot."""
    _, _, _, type_slot = _get_cached_slots(obj_type)
    return has_slot(type_slot, PyTypeSlots.TP_CALL)


def pyiter_check(obj_type: type) -> bool:
    # ref: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L2891-L2897
    # CPython checks if tp_iternext != _PyObject_NextNotImplemented
    # Dynamo only sets the bit if __next__ is actually defined
    return type_implements_tp_iternext(obj_type)


def pysequence_check(obj_type: type) -> bool:
    """Implements PySequence_Check semantics for VariableTracker objects."""
    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1714-L1721
    if issubclass(obj_type, dict):
        return False
    return type_implements_sq_item(obj_type)


def pyindex_check(obj_type: type) -> bool:
    """Implements _PyIndex_Check semantics for VariableTracker objects."""
    # ref: https://github.com/python/cpython/blob/3.13/Include/internal/pycore_abstract.h#L11-L17
    return type_implements_nb_index(obj_type)


def pycallable_check(obj_type: type) -> bool:
    """Implements PyCallable_Check: type(x)->tp_call != NULL.

    obj_type is the object's Python type (Py_TYPE(x)); a non-NULL tp_call
    slot on it means instances are callable.

    ref: https://github.com/python/cpython/blob/v3.13.0/Objects/call.c#L52-L57
    """
    return type_implements_tp_call(obj_type)


def pyiter_send(
    tx: "InstructionTranslatorBase", iter_: VariableTracker, arg: VariableTracker
) -> VariableTracker:
    """Implements PyIter_Send semantics for VariableTracker objects.

    ref: https://github.com/python/cpython/blob/51b511d7299f91a458e40d1ea997bd7e6cd3deef/Objects/abstract.c#L2930-L2953
    """

    tp_iternext = iter_.tp_iternext
    if arg.is_constant_none() and tp_iternext is not None:
        return iter_.tp_iternext_impl(tx)
    else:
        return iter_.call_method(tx, "send", [arg], {})


def pymapping_size(
    tx: "InstructionTranslatorBase", obj: "VariableTracker"
) -> "VariableTracker":
    # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L2308-L2330
    if obj.tp_as_mapping.mp_length:
        return obj.mp_length_impl(tx)

    if obj.tp_as_sequence.sq_length is not None:
        raise_type_error(tx, f"{obj.python_type_name()} is not a mapping")

    raise_type_error(tx, f"object of type {obj.python_type_name()} has no len()")


def generic_size(
    tx: "InstructionTranslatorBase", obj: "VariableTracker"
) -> "VariableTracker":
    # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L53-L69
    """
    Implements PyObject_Size/PyObject_Length semantics for VariableTracker objects.
    Dispatches to sq_length (sequences) or mp_length (mappings) depending on the VT type.
    """

    if obj.tp_as_sequence.sq_length:
        return obj.sq_length_impl(tx)
    return pymapping_size(tx, obj)


def generic_is_true(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyObject_IsTrue.

    https://github.com/python/cpython/blob/c09ccd9c429/Objects/object.c#L2135-L2158

    Resolution order: constants → nb_bool → mp_length/sq_length → truthy.
    """
    from .constant import ConstantVariable

    if obj.is_python_constant():
        try:
            return ConstantVariable.create(bool(obj.as_python_constant()))
        except Exception as e:
            raise_observed_exception(type(e), tx, args=[str(e)])

    if obj.tp_as_number.nb_bool:
        result = obj.nb_bool_impl(tx)
        if result is not None:
            return result

    try:
        length = generic_size(tx, obj)
        from .tensor import SymNodeVariable

        if isinstance(length, SymNodeVariable):
            return SymNodeVariable.create(tx, length.as_proxy() > 0)
        length_val = length.as_python_constant()
        if length_val < 0:
            raise_observed_exception(
                ValueError, tx, args=["__len__() should return >= 0"]
            )
        return ConstantVariable.create(length_val > 0)
    except ObservedTypeError:
        handle_observed_exception(tx)

    return ConstantVariable.create(True)


_repr_running: set[int] = set()


def generic_repr(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyObject_Repr with Py_ReprEnter/Py_ReprLeave cycle detection.

    https://github.com/python/cpython/blob/v3.13.3/Objects/object.c#L745-L778

    Resolution order: tp_repr -> TypeError if the result is not str.
    """
    obj_type = maybe_get_python_type(obj)

    tp_repr = obj.tp_repr
    if tp_repr is not None:
        obj_id = id(obj)
        if obj_id in _repr_running:
            sentinel = {list: "[...]", dict: "{...}", collections.deque: "[...]"}
            return ConstantVariable.create(sentinel.get(obj_type, "..."))
        _repr_running.add(obj_id)
        try:
            result = obj.tp_repr_impl(tx)
        finally:
            _repr_running.discard(obj_id)
        result_type = maybe_get_python_type(result)
        if not issubclass(result_type, str):
            if sys.version_info >= (3, 15):
                err_str = f"{obj.python_qualified_name()}.__repr__() must return a str, not int"
            else:
                err_str = f"__repr__ returned non-string (type {result_type.__name__})"
            raise_type_error(tx, err_str)
        return result

    raise_type_error(tx, f"object of type '{obj.python_type_name()}' has no repr")


def generic_str(
    tx: "InstructionTranslatorBase", obj: "VariableTracker"
) -> "VariableTracker":
    """Mirrors PyObject_Str semantics in Dynamo.

    https://github.com/python/cpython/blob/v3.13.3/Objects/object.c#L781-L829

    Resolution order: str identity check -> tp_str (tp_str_impl) -> tp_repr fallback.
    """
    from ..exc import TorchDynamoException

    if maybe_get_python_type(obj) is str:
        return obj

    try:
        if obj.tp_str and type(obj).tp_str_impl is not VariableTracker.tp_str_impl:
            result = obj.tp_str_impl(tx)
        else:
            result = generic_repr(tx, obj)
    except TorchDynamoException:
        raise
    except Exception as exc:
        raise_observed_exception(type(exc), tx, args=list(exc.args))

    result_type = maybe_get_python_type(result)
    if not issubclass(result_type, str):
        if sys.version_info >= (3, 15):
            err_str = f"{obj.python_qualified_name()}.__str__() must return a str, not {result.python_qualified_name()}"
        else:
            err_str = f"__str__ returned non-string (type {result_type.__name__})"
        raise_type_error(tx, err_str)
    return result


def generic_getitem(
    tx: "InstructionTranslatorBase",
    obj: VariableTracker,
    key: VariableTracker,
) -> VariableTracker:
    """CPython's PyObject_GetItem — dispatch to the type's mp_subscript/sq_item.

    PyObject_GetItem: https://github.com/python/cpython/blob/62a6e898e01/Objects/abstract.c#L155-L206

    CPython checks three branches in order:
      1. tp_as_mapping->mp_subscript  (L161-166)
      2. tp_as_sequence->sq_item      (L168-181) — only if key passes _PyIndex_Check
      3. PyType_Check(o)              (L183-203) — type[int] → GenericAlias/__class_getitem__

    Branch 1 is the common path (list, tuple, dict, range all have mp_subscript).
    Branch 2 fires for types with only sq_item (e.g. deque).
    Branch 3 delegates to mp_subscript_impl for type objects (__class_getitem__).
    """
    obj_type = maybe_get_python_type(obj)
    # Branch 1: mp_subscript
    if obj.tp_as_mapping.mp_subscript:
        return obj.mp_subscript_impl(tx, key)
    # Branch 2: sq_item (only if mp_subscript is absent)
    # CPython: abstract.c L168-181 — _PyIndex_Check(key) → PyNumber_AsSsize_t
    #          → PySequence_GetItem (wraps negative, calls sq_item)
    if obj.tp_as_sequence.sq_item is not None:
        key_type = maybe_get_python_type(key)
        if pyindex_check(key_type):
            key = pynumber_as_ssize_t(tx, key, IndexError)
            return pysequence_getitem(tx, obj, key)
        raise_type_error(
            tx,
            f"{obj_type.__name__} indices must be integers, not {key_type.__name__}",
        )
    # Branch 3: PyType_Check → __class_getitem__ (abstract.c L183-203)
    # In 3.10+ type.__getitem__ sets mp_subscript so this is normally caught
    # by Branch 1, but we check explicitly for safety.
    if issubclass(obj_type, type):
        return obj.mp_subscript_impl(tx, key)
    # CPython: abstract.c L205
    raise_type_error(tx, f"'{obj_type.__name__}' object is not subscriptable")


def pysequence_getitem(
    tx: "InstructionTranslatorBase",
    obj: VariableTracker,
    index: VariableTracker,
) -> VariableTracker:
    """CPython's PySequence_GetItem — always sq_item, never mp_subscript.

    ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L1874-L1902

    Called by PyObject_GetItem branch 2, reversed() fallback, and the old
    iteration protocol.  Wraps negative indices via sq_length before
    dispatching to sq_item.
    """
    sq_item = obj.tp_as_sequence.sq_item
    if sq_item is not None:
        # Negative index wrapping (abstract.c L2175-2183)
        if isinstance(index, ConstantVariable):
            index_val = index.as_python_constant()
            if isinstance(index_val, int) and index_val < 0:
                if obj.tp_as_sequence.sq_length is not None:
                    length = obj.sq_length_impl(tx)
                    index = ConstantVariable.create(
                        index_val + length.as_python_constant()
                    )
        return obj.sq_item_impl(tx, index)

    if obj.tp_as_mapping.mp_subscript is not None:
        raise_type_error(tx, f"'{obj.python_type_name()}' is not a sequence")

    raise_type_error(tx, f"'{obj.python_type_name()}' object does not support indexing")


def pysequence_setitem(
    tx: "InstructionTranslatorBase",
    s: VariableTracker,
    i: VariableTracker,
    o: VariableTracker,
) -> VariableTracker:
    # ref: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L1926-L1957 (PySequence_SetItem)
    sq_ass_item = s.tp_as_sequence.sq_ass_item
    if sq_ass_item is not None:
        # Negative index wrapping (abstract.c L1944-1952)
        if isinstance(i, ConstantVariable):
            index_val = i.as_python_constant()
            if isinstance(index_val, int) and index_val < 0:
                if s.tp_as_sequence.sq_length is not None:
                    length = s.sq_length_impl(tx)
                    i = ConstantVariable.create(index_val + length.as_python_constant())
        return s.sq_ass_item_impl(tx, i, o)

    if s.tp_as_mapping.mp_ass_subscript is not None:
        raise_type_error(tx, f"'{s.python_type_name()}' is not a sequence")

    raise_type_error(
        tx, f"'{s.python_type_name()}' object does not support item assignment"
    )


def generic_setitem(
    tx: "InstructionTranslatorBase",
    o: VariableTracker,
    key: VariableTracker,
    value: VariableTracker,
) -> VariableTracker:
    # ref: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L222-L254
    mp_ass_subscript = o.tp_as_mapping.mp_ass_subscript
    if mp_ass_subscript is not None:
        return o.mp_ass_subscript_impl(tx, key, value)

    if o.tp_as_sequence.sq_ass_item is not None:
        key_type = maybe_get_python_type(key)
        if pyindex_check(key_type):
            key_value = pynumber_as_ssize_t(tx, key, err=IndexError)
            return pysequence_setitem(tx, o, key_value, value)
        raise_type_error(
            tx, f"sequence index must be integer, not '{key.python_type_name()}'"
        )
    raise_type_error(
        tx, f"'{o.python_type_name()}' object does not support item assignment"
    )


def pysequence_delitem(
    tx: "InstructionTranslatorBase",
    s: VariableTracker,
    i: VariableTracker,
) -> VariableTracker:
    # ref: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L1959-L1990

    sq_ass_item = s.tp_as_sequence.sq_ass_item
    if sq_ass_item is not None:
        if isinstance(i, ConstantVariable):
            idx = i.as_python_constant()
            if idx < 0:
                if s.tp_as_sequence.sq_length is not None:
                    length = s.sq_length_impl(tx)
                    i = pynumber_add(tx, i, length)
        return s.sq_ass_item_impl(tx, i, None)

    if s.tp_as_mapping.mp_ass_subscript is not None:
        raise_type_error(tx, f"'{s.python_type_name()}' is not a sequence")

    raise_type_error(
        tx, f"'{s.python_type_name()}' object does not support item deletion"
    )


def generic_delitem(
    tx: "InstructionTranslatorBase",
    o: VariableTracker,
    key: VariableTracker,
) -> VariableTracker:
    # ref: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L256-L288

    mp_ass_subscript = o.tp_as_mapping.mp_ass_subscript
    if mp_ass_subscript is not None:
        return o.mp_ass_subscript_impl(tx, key, None)

    key_type = maybe_get_python_type(key)
    if pyindex_check(key_type):
        key_value = key.nb_index_impl(tx)
        return pysequence_delitem(tx, o, key_value)
    elif o.tp_as_sequence.sq_ass_item is not None:
        raise_type_error(
            tx, f"sequence index must be integer, not {key.python_type_name()}"
        )

    raise_type_error(tx, f"'{o.python_type_name()}' does not support item deletion")


def pynumber_int(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyNumber_Long (int(x) dispatch).

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1520-L1632

    Resolution: nb_int → nb_index → str/bytes/bytearray parsing → TypeError.
    """
    from .constant import ConstantVariable

    # Fast path for int (sub)class instances — mirrors PyLong_Check at the
    # top of PyNumber_Long (abstract.c:1531). Avoids infinite recursion for
    # int subclasses like IntEnum whose __int__ calls int() again.
    if obj.is_python_constant() and isinstance(obj.as_python_constant(), int):
        return ConstantVariable.create(int(obj.as_python_constant()))

    if obj.tp_as_number.nb_int is not None:
        res = obj.nb_int_impl(tx)
        if res.python_type() is not int:
            raise_type_error(
                tx,
                f"__int__ returned non-int (type {res.python_type_name()})",
            )
        return res

    if obj.tp_as_number.nb_index is not None:
        return obj.nb_index_impl(tx)

    # String/bytes/bytearray parsing fallback.
    # https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1598-L1612
    if obj.is_python_constant() and isinstance(
        obj.as_python_constant(), (str, bytes, bytearray)
    ):
        try:
            return ConstantVariable.create(int(obj.as_python_constant()))
        except ValueError as e:
            raise_observed_exception(ValueError, tx, args=[str(e)])

    raise_type_error(
        tx,
        f"int() argument must be a string, a bytes-like object "
        f"or a real number, not '{obj.python_type_name()}'",
    )


def pylong_from_base(
    tx: "InstructionTranslatorBase", x: VariableTracker, obase: VariableTracker
) -> VariableTracker | None:
    """Mirrors the explicit-base path of long_new_impl (int(x, base)).

    https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5879-L5922

    The base is resolved via PyNumber_AsSsize_t, which consults __index__, so
    any __index__-able object is accepted. Returns None (graph break) when the
    inputs cannot be resolved to Python constants.
    """
    # base = PyNumber_AsSsize_t(obase, NULL) -> nb_index.
    if obase.is_python_constant() and issubclass(obase.python_type(), int):
        base_vt = obase
    elif obase.tp_as_number.nb_index is not None:
        base_vt = obase.nb_index_impl(tx)
    else:
        raise_type_error(
            tx,
            f"'{obase.python_type_name()}' object cannot be interpreted as an integer",
        )
    if not (base_vt.is_python_constant() and issubclass(base_vt.python_type(), int)):
        return None
    base = base_vt.as_python_constant()
    if (base != 0 and base < 2) or base > 36:
        raise_observed_exception(
            ValueError, tx, args=["int() base must be >= 2 and <= 36, or 0"]
        )
    if not x.is_python_constant():
        return None
    xval = x.as_python_constant()
    if not isinstance(xval, (str, bytes, bytearray)):
        raise_type_error(tx, "int() can't convert non-string with explicit base")
    try:
        return ConstantVariable.create(int(xval, base))
    except ValueError as e:
        raise_observed_exception(ValueError, tx, args=list(e.args))


def pynumber_float(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyNumber_Float (float(x) dispatch).

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1635-L1692

    Resolution: nb_float → nb_index → str parsing → TypeError.
    """
    from .constant import ConstantVariable

    # Fast path: if the value is already a float constant, return it directly.
    # Mirrors PyFloat_CheckExact fast path at the top of PyNumber_Float
    # (abstract.c:1641-1643).
    if obj.is_python_constant() and isinstance(obj.as_python_constant(), float):
        return ConstantVariable.create(float(obj.as_python_constant()))

    if obj.tp_as_number.nb_float is not None:
        res = obj.nb_float_impl(tx)
        if res.python_type() is not float:
            raise_type_error(
                tx,
                f"__float__ returned non-float (type {res.python_type_name()})",
            )
        return res

    # https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1674-L1685
    if obj.tp_as_number.nb_index is not None:
        return obj.nb_index_impl(tx)

    # PyFloat_FromString fallback — handles str and bytes.
    # https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1691
    if issubclass(obj.python_type(), (str, bytes)):
        try:
            return ConstantVariable.create(float(obj.as_python_constant()))
        except ValueError as e:
            raise_observed_exception(ValueError, tx, args=[str(e)])

    raise_type_error(
        tx,
        f"float() argument must be a string or a real number, "
        f"not '{obj.python_type_name()}'",
    )


def getindex(
    tx: "InstructionTranslatorBase",
    obj: VariableTracker,
    arg: VariableTracker,
) -> VariableTracker:
    """Mirrors typeobject.c::getindex: calls PyNumber_AsSsize_t then tp_as_sequence.sq_length"""
    obj_type = maybe_get_python_type(obj)

    i = pynumber_as_ssize_t(tx, arg, err=OverflowError)
    if i.as_python_constant() < 0:
        if type_implements_sq_length(obj_type):
            length = obj.sq_length_impl(tx)
            i = pynumber_add(tx, i, length)
    return i


def pylong_as_ssize_t(tx: "InstructionTranslatorBase", obj: VariableTracker) -> int:
    """Mirrors PyLong_AsSsize_t: requires an int (or subclass).
    values outside the Py_ssize_t range raise OverflowError.

    https://github.com/python/cpython/blob/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Objects/longobject.c#L576
    """
    # Starting on Python 3.16, this will explicitly require an integer instance
    # https://docs.python.org/3/deprecations/index.html#pending-removal-in-python-3-16
    if not issubclass(obj.python_type(), int):
        raise_type_error(tx, "an integer is required")
    val = obj.as_python_constant()
    if not -sys.maxsize - 1 <= val <= sys.maxsize:
        raise_observed_exception(
            OverflowError,
            tx,
            args=["Python int too large to convert to C ssize_t"],
        )
    return val


def pynumber_as_ssize_t(
    tx: "InstructionTranslatorBase",
    item: VariableTracker,
    err: type[Exception] | None = IndexError,
) -> VariableTracker:
    """Mirrors PyNumber_AsSsize_t: _PyNumber_Index(item) then PyLong_AsSsize_t.

    On overflow (value outside the Py_ssize_t range) CPython remaps the
    OverflowError to `err`, or clips to the Py_ssize_t bounds when err is None.

    https://github.com/python/cpython/blob/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Objects/abstract.c#L1469
    """
    from .tensor import SymNodeVariable

    value = pynumber_index(tx, item)

    # PyLong_AsSsize_t: a symbolic int must be specialized to a concrete
    # ssize_t (with guard) to be usable as a C index.
    if isinstance(value, SymNodeVariable):
        val = value.evaluate_expr(tx.output)
    else:
        val = value.as_python_constant()

    if not isinstance(val, int):
        raise AssertionError("pynumber_index did not return an int-like value")

    if -sys.maxsize - 1 <= val <= sys.maxsize:
        return ConstantVariable.create(int(val))
    if err is None:
        r = sys.maxsize if val > 0 else -sys.maxsize - 1
        return ConstantVariable.create(r)
    raise_observed_exception(
        err,
        tx,
        args=[f"cannot fit '{item.python_type_name()}' into an index-sized integer"],
    )


def pynumber_index(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> "VariableTracker":
    """Mirrors PyNumber_Index (index(x) dispatch)."""

    if obj.tp_as_number.nb_index is None:
        raise_type_error(
            tx,
            f"'{obj.python_type_name()}' object cannot be interpreted as an integer",
        )

    result = obj.nb_index_impl(tx)

    if not issubclass(result.python_type(), int):
        raise_type_error(
            tx,
            f"__index__ returned non-int (type {result.python_type_name()})",
        )

    return result


def pynumber_tobase(
    tx: "InstructionTranslatorBase", obj: VariableTracker, base: int
) -> VariableTracker | None:
    """Mirrors PyNumber_ToBase (bin/oct/hex dispatch).

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1653-L1666

    Resolves __index__ (raising TypeError if absent), then formats the
    resulting int in the requested base. Returns None (graph break) when the
    index result is not a Python constant.
    """
    index = pynumber_index(tx, obj)
    format_fn = {2: bin, 8: oct, 16: hex}[base]
    return ConstantVariable.create(format_fn(index.as_python_constant()))


def pyiter_next(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> "VariableTracker":
    """
    Implements PyIter_Next / tp_iternext semantics for VariableTracker objects.

    Calls obj.tp_iternext_impl(tx) if the object is an iterator, otherwise raises
    TypeError. StopIteration propagation is left to the caller (mirrors
    CPython's iternext contract where NULL return signals exhaustion).
    """
    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L2865

    tp_iternext = obj.tp_iternext
    if tp_iternext is None:
        raise_type_error(tx, f"expected an iterator, got '{obj.python_type_name()}'")

    return obj.tp_iternext_impl(tx)


def pynumber_negative(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyNumber_Negative.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1375-L1392

    Algorithm:
    1. If type has nb_negative slot, call obj.nb_negative_impl(tx)
    2. Otherwise, raise TypeError
    """

    nb_negative = obj.tp_as_number.nb_negative
    if nb_negative is not None:
        return obj.nb_negative_impl(tx)

    raise_type_error(
        tx,
        f"bad operand type for unary -: '{obj.python_type_name()}'",
    )


def pynumber_positive(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyNumber_Positive.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1375-L1393

    Algorithm:
    1. If type has nb_positive slot, call obj.nb_positive_impl(tx)
    2. Otherwise, raise TypeError
    """

    nb_positive = obj.tp_as_number.nb_positive
    if nb_positive is not None:
        return obj.nb_positive_impl(tx)

    raise_type_error(
        tx,
        f"bad operand type for unary +: '{obj.python_type_name()}'",
    )


def pynumber_absolute(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyNumber_Absolute.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1375-L1395

    Algorithm:
    1. If type has nb_absolute slot, call obj.nb_absolute_impl(tx)
    2. Otherwise, raise TypeError
    """

    nb_absolute = obj.tp_as_number.nb_absolute
    if nb_absolute is not None:
        return obj.nb_absolute_impl(tx)

    raise_type_error(
        tx,
        f"bad operand type for abs(): '{obj.python_type_name()}'",
    )


def vt_is_iterable(obj: VariableTracker) -> bool:
    """Check if the object supports iteration (i.e. has tp_iter or sequence protocol)."""
    T = maybe_get_python_type(obj)
    return obj.tp_iter is not None or pysequence_check(T)


def pynumber_invert(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyNumber_Invert.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1375-L1394

    Algorithm:
    1. If type has nb_invert slot, call obj.nb_invert_impl(tx)
    2. Otherwise, raise TypeError
    """

    nb_invert = obj.tp_as_number.nb_invert
    if nb_invert is not None:
        return obj.nb_invert_impl(tx)

    raise_type_error(
        tx,
        f"bad operand type for unary ~: '{obj.python_type_name()}'",
    )


def generic_getiter(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> "VariableTracker":
    """
    Implements PyObject_GetIter semantics for VariableTracker objects.
    Routes to obj.tp_iter_impl(tx), the tp_iter slot on the object's type.
    """

    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L2847-L2870
    # The algorithm for PyObject_GetIter works as follows: Steps:
    # 1. If the object has tp_iter slot, call it and return the result. The
    #    return object must be an iterator (it must have a tp_iternext slot)
    # 2. If the object implements the sequence protocol - implements __getitem__
    #    then create a sequence iterator for the object and return it
    # 3. Otherwise, raise a TypeError

    T = maybe_get_python_type(obj)
    tp_iter = obj.tp_iter
    if tp_iter is not None:
        res = obj.tp_iter_impl(tx)
        res_T = maybe_get_python_type(res)
        if not pyiter_check(res_T):
            if sys.version_info >= (3, 15):
                err_str = f"{obj.python_qualified_name()}.__iter__() must return an iterator, not {res.python_qualified_name()}"
            else:
                err_str = (
                    f"iter() returned non-iterator of type '{res.python_type_name()}'"
                )
            raise_type_error(tx, err_str)
        return res
    elif pysequence_check(T):
        from .functions import UserFunctionVariable

        return UserFunctionVariable(polyfills.builtins.sequence_iterator).call_function(
            tx, [obj], {}
        )
    else:
        raise_type_error(tx, f"'{obj.python_type_name()}' object is not iterable")


# ---------------------------------------------------------------------------
# Binary-op dispatch (CPython's abstract.c: binary_op1 / binary_op)
# https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L927 (binary_op1)
# ---------------------------------------------------------------------------

NB_SLOT_MAPPING = {
    "nb_lshift": PyNumberSlots.NB_LSHIFT,
    "nb_inplace_lshift": PyNumberSlots.NB_INPLACE_LSHIFT,
    "nb_inplace_rshift": PyNumberSlots.NB_INPLACE_RSHIFT,
    "nb_rshift": PyNumberSlots.NB_RSHIFT,
    "nb_or": PyNumberSlots.NB_OR,
    "nb_inplace_or": PyNumberSlots.NB_INPLACE_OR,
    "nb_subtract": PyNumberSlots.NB_SUBTRACT,
    "nb_inplace_subtract": PyNumberSlots.NB_INPLACE_SUBTRACT,
    "nb_add": PyNumberSlots.NB_ADD,
    "nb_inplace_add": PyNumberSlots.NB_INPLACE_ADD,
    "nb_multiply": PyNumberSlots.NB_MULTIPLY,
    "nb_inplace_multiply": PyNumberSlots.NB_INPLACE_MULTIPLY,
    "nb_matrix_multiply": PyNumberSlots.NB_MATRIX_MULTIPLY,
    "nb_inplace_matrix_multiply": PyNumberSlots.NB_INPLACE_MATRIX_MULTIPLY,
    "nb_and": PyNumberSlots.NB_AND,
    "nb_inplace_and": PyNumberSlots.NB_INPLACE_AND,
    "nb_xor": PyNumberSlots.NB_XOR,
    "nb_inplace_xor": PyNumberSlots.NB_INPLACE_XOR,
    "nb_floor_divide": PyNumberSlots.NB_FLOOR_DIVIDE,
    "nb_inplace_floor_divide": PyNumberSlots.NB_INPLACE_FLOOR_DIVIDE,
    "nb_true_divide": PyNumberSlots.NB_TRUE_DIVIDE,
    "nb_inplace_true_divide": PyNumberSlots.NB_INPLACE_TRUE_DIVIDE,
    "nb_remainder": PyNumberSlots.NB_REMAINDER,
    "nb_inplace_remainder": PyNumberSlots.NB_INPLACE_REMAINDER,
    "nb_divmod": PyNumberSlots.NB_DIVMOD,
    "nb_power": PyNumberSlots.NB_POWER,
    "nb_inplace_power": PyNumberSlots.NB_INPLACE_POWER,
}


def is_nb_not_implemented(result: VariableTracker) -> bool:
    return result.is_constant_match(NotImplemented)


def is_python_subtype(w: VariableTracker, v: VariableTracker) -> bool:
    """Check if w's underlying Python type is a proper subtype of v's."""
    try:
        return issubclass(w.python_type(), v.python_type())
    except NotImplementedError:
        return False


#   Calling scheme used for binary operations:
#
#   Order operations are tried until either a valid result or error:
#     w.op(v,w)[*], v.op(v,w), w.op(v,w)
#
#   [*] only when Py_TYPE(v) != Py_TYPE(w) && Py_TYPE(w) is a subclass of
#       Py_TYPE(v)


def binary_op1(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    op_slot: str,
) -> VariableTracker:
    """CPython's binary_op1: try v's slot, then w's slot with subclass priority.

    Each VT that participates provides a ``<op_slot>_impl(self, tx, other,
    reverse)`` method. ``reverse=False`` means "self is left operand" (forward,
    e.g. ``__or__``), ``reverse=True`` means "self is right operand" (reverse,
    e.g. ``__ror__``). For built-in types the flag is ignored because their
    slots check both operands symmetrically.

    CPython splits the check into two steps: ``Py_TYPE(v)->tp_as_number !=
    NULL`` (does the type have a number-protocol struct), then read the
    slot pointer (which may itself be NULL).  We collapse both into a
    single :func:`type_implements_nb_slot` query — its bit is set only
    when the specific slot is non-NULL, which already implies
    ``tp_as_number`` is non-NULL.  Treating a missing slot as "no impl"
    is what keeps the base ``VariableTracker.nb_*_impl`` graph-break
    out of the path for types that genuinely lack the slot in C.

    https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L926-L977
    """
    impl_attr = f"{op_slot}_impl"
    nb_slot_bit = NB_SLOT_MAPPING[op_slot]

    v_type = maybe_get_python_type(v)
    w_type = maybe_get_python_type(w)

    v_slot = (
        getattr(type(v), impl_attr, None)
        if type_implements_nb_slot(v_type, nb_slot_bit)
        else None
    )
    w_slot = (
        getattr(type(w), impl_attr, None)
        if type_implements_nb_slot(w_type, nb_slot_bit)
        else None
    )

    # CPython skips slotw if Py_TYPE(w) == Py_TYPE(v) (one C type, one slot
    # function).  In Dynamo two VT subclasses can share a Python type — e.g.
    # ConstantVariable(3) and SymNodeVariable both report ``int`` — yet have
    # different ``nb_*_impl`` methods.  Comparing the slots themselves
    # captures both "literally the same function" (CPython's check) and
    # "different VT subclasses sharing a python_type", so we drop the type
    # equality check.
    if v_slot is w_slot:
        w_slot = None

    if v_slot is not None:
        # Subclass priority: if w's Python type is a proper subtype of v's
        # Python type and overrides the slot, try w first (CPython abstract.c:952-960).
        if w_slot is not None and is_python_subtype(w, v):
            # CPython ALWAYS calls the slot with (v, w), even for reverse slots.
            # Since w_slot is a method call, we use reverse=True to indicate w
            # is the right operand, matching CPython's semantics.
            result = w_slot(w, tx, v, True)
            if not is_nb_not_implemented(result):
                return result
            w_slot = None
        result = v_slot(v, tx, w, False)
        if not is_nb_not_implemented(result):
            return result
    if w_slot is not None:
        result = w_slot(w, tx, v, True)
        if not is_nb_not_implemented(result):
            return result
    return ConstantVariable.create(NotImplemented)


def binary_op(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    op_slot: str,
    op_symbol: str,
) -> VariableTracker:
    """CPython's binary_op: binary_op1 + TypeError fallback.
    https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L997-L1020
    """

    result = binary_op1(tx, v, w, op_slot)
    if is_nb_not_implemented(result):
        binop_type_error(tx, v, w, op_symbol)
    return result


def ternary_op(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    z: VariableTracker | None,
    op_slot: str,
    op_name: str,
) -> VariableTracker:
    impl_attr = f"{op_slot}_impl"
    nb_slot_bit = NB_SLOT_MAPPING[op_slot]

    v_type = maybe_get_python_type(v)
    w_type = maybe_get_python_type(w)

    v_slot = (
        getattr(type(v), impl_attr, None)
        if type_implements_nb_slot(v_type, nb_slot_bit)
        else None
    )

    w_slot = (
        getattr(type(w), impl_attr, None)
        if type_implements_nb_slot(w_type, nb_slot_bit)
        else None
    )

    if v_slot is w_slot:
        w_slot = None

    if v_slot:
        if w_slot and is_python_subtype(w, v):
            result = w_slot(w, tx, v, z, True)
            if not is_nb_not_implemented(result):
                return result
            w_slot = None
        result = v_slot(v, tx, w, z, False)
        if not is_nb_not_implemented(result):
            return result
    if w_slot:
        result = w_slot(w, tx, v, z, True)
        if not is_nb_not_implemented(result):
            return result

    if z:
        z_type = maybe_get_python_type(z)
        if type_implements_nb_slot(z_type, nb_slot_bit):
            z_impl_attr = f"{op_slot}_z_impl"
            z_slot = getattr(type(z), z_impl_attr, None)
            if z_slot in (v_slot, w_slot):
                z_slot = None
            if z_slot:
                result = z_slot(z, tx, v, w)
                if not is_nb_not_implemented(result):
                    return result

    if z is None:
        binop_type_error(tx, v, w, op_name)
    else:
        raise_type_error(
            tx,
            f"unsupported operand type(s) for {op_name}: "
            f"'{v.python_type_name()}', '{w.python_type_name()}', '{z.python_type_name()}'",
        )


def ternary_iop(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    z: VariableTracker | None,
    iop_slot: str,
    op_slot: str,
    op_name: str,
) -> VariableTracker:
    v_type = maybe_get_python_type(v)
    if type_implements_nb_slot(v_type, NB_SLOT_MAPPING[iop_slot]):
        impl_attr = f"{iop_slot}_impl"
        slot = getattr(type(v), impl_attr)
        result = slot(v, tx, w, z)
        if not is_nb_not_implemented(result):
            return result

    return ternary_op(tx, v, w, z, op_slot, op_name)


#  Binary in-place operators
#
#    The in-place operators are defined to fall back to the 'normal', non
#    in-place operations, if the in-place methods are not in place.
#
#    - If the left hand object has the appropriate struct members, and they are
#      filled, call the appropriate function and return the result.  No coercion
#      is done on the arguments; the left-hand object is the one the operation
#      is performed on, and it's up to the function to deal with the right-hand
#      object.
#
#    - Otherwise, in-place modification is not supported. Handle it exactly as a
#      non in-place operation of the same kind.


def binary_iop1(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    iop_slot: str,
    op_slot: str,
) -> VariableTracker:
    v_type = maybe_get_python_type(v)

    if type_implements_nb_slot(v_type, NB_SLOT_MAPPING[iop_slot]):
        impl_attr = f"{iop_slot}_impl"
        slot = getattr(type(v), impl_attr)
        result = slot(v, tx, w)
        if not is_nb_not_implemented(result):
            return result

    return binary_op1(tx, v, w, op_slot)


def binary_iop(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    iop_slot: str,
    op_slot: str,
    op_symbol: str,
) -> VariableTracker:
    """CPython's binary_iop: try inplace slot, fallback to binary_op1.

    Combines binary_iop1 + TypeError fallback from binary_iop.
    https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L1229-L1270 (binary_iop1, binary_iop)
    """
    result = binary_iop1(tx, v, w, iop_slot, op_slot)
    if is_nb_not_implemented(result):
        binop_type_error(tx, v, w, op_symbol)
    return result


# add / inplace add needs special handling
def pynumber_add(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
) -> VariableTracker:
    """Implements addition via nb_add / nb_inplace_add with binary_op dispatch."""
    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1138-L1155
    result = binary_op1(tx, v, w, "nb_add")
    if not is_nb_not_implemented(result):
        return result

    sq_concat = v.tp_as_sequence.sq_concat
    if sq_concat is not None:
        return v.sq_concat_impl(tx, w)
    binop_type_error(tx, v, w, "+")


def pynumber_inplace_add(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
) -> VariableTracker:
    """Implements in-place addition via nb_inplace_add with binary_iop dispatch."""
    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1307-L1328
    result = binary_iop1(tx, v, w, "nb_inplace_add", "nb_add")
    if is_nb_not_implemented(result):
        sq_inplace_concat = v.tp_as_sequence.sq_inplace_concat
        sq_concat = v.tp_as_sequence.sq_concat
        if sq_inplace_concat is not None:
            return v.sq_inplace_concat_impl(tx, w)
        elif sq_concat is not None:
            return v.sq_concat_impl(tx, w)
        else:
            binop_type_error(tx, v, w, "+=")
    return result


# ---------------------------------------------------------------------------
# Multiplication: PyNumber_Multiply / PyNumber_InPlaceMultiply
#
# Multiplication is special because numeric types implement nb_multiply but
# sequence types (list, tuple, str, bytes, bytearray) implement only sq_repeat.
# CPython's PyNumber_Multiply tries nb_multiply first (via binary_op1) and on
# NotImplemented falls back to sq_repeat on either operand.
#
# https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1156-L1193
# ---------------------------------------------------------------------------


def pysequence_repeat(
    tx: "InstructionTranslatorBase",
    seq: VariableTracker,
    n: VariableTracker,
) -> VariableTracker:
    """Mirrors CPython's sequence_repeat helper.

    Validates that ``n`` is index-like, converts it to an int, and dispatches
    to ``seq.sq_repeat_impl(tx, count)``.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1156-L1174
    """
    n_type = maybe_get_python_type(n)
    if not pyindex_check(n_type):
        raise_type_error(
            tx,
            f"can't multiply sequence by non-int of type '{n.python_type_name()}'",
        )
    count = n.nb_index_impl(tx)
    validate_sequence_repeat_count(tx, count)
    return seq.sq_repeat_impl(tx, count)


def pysequence_inplace_repeat(
    tx: "InstructionTranslatorBase",
    seq: VariableTracker,
    n: VariableTracker,
) -> VariableTracker:
    """pysequence_repeat using sq_inplace_repeat.

    The validation step is identical to ``pysequence_repeat``; only the
    target slot differs.
    """
    n_type = maybe_get_python_type(n)
    if not pyindex_check(n_type):
        raise_type_error(
            tx,
            f"can't multiply sequence by non-int of type '{n.python_type_name()}'",
        )
    count = n.nb_index_impl(tx)
    validate_sequence_repeat_count(tx, count)
    return seq.sq_inplace_repeat_impl(tx, count)


def validate_sequence_repeat_count(
    tx: "InstructionTranslatorBase",
    count: VariableTracker,
) -> None:
    n = count.as_python_constant()
    if n < -sys.maxsize - 1 or n > sys.maxsize:
        raise_observed_exception(
            OverflowError,
            tx,
            args=["cannot fit 'int' into an index-sized integer"],
        )


def pynumber_multiply(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
) -> VariableTracker:
    """Mirrors CPython's PyNumber_Multiply.

    Try nb_multiply via binary_op1; on NotImplemented fall back to sq_repeat
    on either operand.  TypeError if no path works.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1176-L1193
    """
    result = binary_op1(tx, v, w, "nb_multiply")
    if not is_nb_not_implemented(result):
        return result

    if v.tp_as_sequence.sq_repeat is not None:
        return pysequence_repeat(tx, v, w)
    if w.tp_as_sequence.sq_repeat is not None:
        return pysequence_repeat(tx, w, v)

    raise_type_error(
        tx,
        f"unsupported operand type(s) for *: "
        f"'{v.python_type_name()}' and '{w.python_type_name()}'",
    )


def pynumber_inplace_multiply(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
) -> VariableTracker:
    """Mirrors CPython's PyNumber_InPlaceMultiply.

    Try nb_inplace_multiply / nb_multiply via binary_iop1; on NotImplemented
    fall back to sq_inplace_repeat (preferred), then sq_repeat.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1330-L1357
    """
    result = binary_iop1(tx, v, w, "nb_inplace_multiply", "nb_multiply")
    if not is_nb_not_implemented(result):
        return result

    if v.tp_as_sequence.sq_inplace_repeat is not None:
        return pysequence_inplace_repeat(tx, v, w)
    if v.tp_as_sequence.sq_repeat is not None:
        return pysequence_repeat(tx, v, w)
    # Cannot mutate w in-place — abstract.c L1348-1352 explicitly avoids
    # sq_inplace_repeat on the right-hand operand.
    if w.tp_as_sequence.sq_repeat is not None:
        return pysequence_repeat(tx, w, v)

    raise_type_error(
        tx,
        f"unsupported operand type(s) for *=: "
        f"'{v.python_type_name()}' and '{w.python_type_name()}'",
    )


def pynumber_matrix_multiply(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
) -> VariableTracker:
    """Mirrors CPython's PyNumber_MatrixMultiply."""
    return binary_op(tx, v, w, "nb_matrix_multiply", "@")


def pynumber_inplace_matrix_multiply(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
) -> VariableTracker:
    """Mirrors CPython's PyNumber_InPlaceMatrixMultiply."""
    return binary_iop(
        tx,
        v,
        w,
        "nb_inplace_matrix_multiply",
        "nb_matrix_multiply",
        "@=",
    )


# ---------------------------------------------------------------------------
# Type-object slot wrappers for ``__mul__`` / ``__rmul__`` / ``__imul__``
#
# These mirror the *type object*'s installation of ``__mul__`` etc. as a
# method, not the operator dispatch.  In CPython the user-visible
# ``int.__mul__`` and ``list.__mul__`` are different functions, each
# generated at type-construction time from ``Objects/typeobject.c``'s
# ``slotdefs[]`` table:
#
#   slotdefs[]:
#     BINSLOT(__mul__, nb_multiply, slot_nb_multiply, "*")    [L10323]
#     SQSLOT (__mul__, sq_repeat,   NULL, wrap_indexargfunc)  [L10419]
#     RBINSLOT(__rmul__, nb_multiply, slot_nb_multiply, "*")  [L10325]
#     SQSLOT (__rmul__, sq_repeat,   NULL, wrap_indexargfunc) [L10421]
#     IBSLOT (__imul__, nb_inplace_multiply, ...)             [L10364]
#     SQSLOT (__imul__, sq_inplace_repeat, NULL, ...)         [L10434]
#
# https://github.com/python/cpython/blob/v3.13.13/Objects/typeobject.c#L10244 (slotdefs[])
# https://github.com/python/cpython/blob/v3.13.13/Objects/typeobject.c#L10412-L10416 (rationale comment)
#
# ``add_operators`` walks ``slotdefs[]`` and inserts a method into the
# type's ``__dict__`` for whichever slot the type fills.  As the comment
# at L10412 notes, types fill at most one of ``nb_multiply`` /
# ``sq_repeat`` per op; nb_* takes priority because its slotdef appears
# first.  ``slot_wrapper_mul``/``slot_wrapper_imul`` reproduce that
# selection so that ``call_method`` routing for direct dunder calls
# (``[1, 2].__mul__(3)``) reaches the correct slot.
#
# Distinct from ``pynumber_multiply`` / ``pynumber_inplace_multiply``, which
# mirror the operator-level algorithm in ``Objects/abstract.c``
# (``PyNumber_Multiply``) — cross-operand subclass priority and the
# ``sq_repeat`` fallback when ``nb_multiply`` returns ``NotImplemented``.
# ---------------------------------------------------------------------------


def slot_wrapper_mul(
    tx: "InstructionTranslatorBase",
    self: VariableTracker,
    other: VariableTracker,
    reverse: bool = False,
) -> VariableTracker:
    """``self.__mul__(other)`` / ``self.__rmul__(other)`` slot wrapper."""
    nb_multiply = self.tp_as_number.nb_multiply
    if nb_multiply is not None:
        return self.nb_multiply_impl(tx, other, reverse=reverse)
    if self.tp_as_sequence.sq_repeat is not None:
        # SQSLOT for __mul__ and __rmul__ both use ``wrap_indexargfunc`` —
        # the wrapper ignores the reverse flag because sq_repeat takes
        # ``(seq, count)`` regardless of which side ``self`` is on.
        return pysequence_repeat(tx, self, other)
    raise_type_error(
        tx,
        f"unsupported operand type(s) for *: "
        f"'{self.python_type_name()}' and '{other.python_type_name()}'",
    )


def slot_wrapper_imul(
    tx: "InstructionTranslatorBase",
    self: VariableTracker,
    other: VariableTracker,
) -> VariableTracker:
    """``self.__imul__(other)`` slot wrapper.

    When neither ``nb_inplace_multiply`` nor ``sq_inplace_repeat`` is
    installed, the slotdef machinery doesn't generate ``__imul__`` at all
    — but the operator-level fallback in ``slot_nb_inplace_multiply``
    (typeobject.c) does try the non-inplace slot.  We mirror that here so
    method lookups don't graph-break on (e.g.) tuple, even though tuple
    has no ``__imul__`` attribute in standard CPython.
    """
    nb_inplace_multiply = self.tp_as_number.nb_inplace_multiply
    if nb_inplace_multiply is not None:
        return self.nb_inplace_multiply_impl(tx, other)
    if self.tp_as_sequence.sq_inplace_repeat is not None:
        return pysequence_inplace_repeat(tx, self, other)
    return slot_wrapper_mul(tx, self, other)


# ---------------------------------------------------------------------------
# Type-object slot wrappers for ``__add__`` / ``__radd__`` / ``__iadd__``

#   BINSLOT (__add__,  nb_add,      slot_nb_add, "+")           [typeobject.c L10915]
#   RBINSLOT(__radd__, nb_add,      slot_nb_add, "+")           [L10917]
#   SQSLOT  (__add__,  sq_concat,   NULL, wrap_binaryfunc)      [L11017]
#   IBSLOT  (__iadd__, nb_inplace_add, ...)                     [L10960]
#   SQSLOT  (__iadd__, sq_inplace_concat, NULL, ...)            [L11031]
#
# Distinct from ``pynumber_add`` / ``pynumber_inplace_add``, which mirror the
# operator-level ``PyNumber_Add`` (cross-operand priority, sq_concat fallback).
# ---------------------------------------------------------------------------


def slot_wrapper_add(
    tx: "InstructionTranslatorBase",
    self: VariableTracker,
    other: VariableTracker,
    reverse: bool = False,
) -> VariableTracker:
    """``self.__add__(other)`` / ``self.__radd__(other)`` slot wrapper."""
    nb_add = self.tp_as_number.nb_add
    if nb_add is not None:
        return self.nb_add_impl(tx, other, reverse=reverse)
    # No SQSLOT(__radd__): sq_concat backs only the forward __add__.
    sq_concat = self.tp_as_sequence.sq_concat
    if not reverse and sq_concat is not None:
        return self.sq_concat_impl(tx, other)
    raise_type_error(
        tx,
        f"unsupported operand type(s) for +: "
        f"'{self.python_type_name()}' and '{other.python_type_name()}'",
    )


def slot_wrapper_iadd(
    tx: "InstructionTranslatorBase",
    self: VariableTracker,
    other: VariableTracker,
) -> VariableTracker:
    """``self.__iadd__(other)`` slot wrapper.

    Mirrors ``slot_wrapper_imul``: when neither ``nb_inplace_add`` nor
    ``sq_inplace_concat`` is installed, fall back to the non-inplace slot.
    """
    nb_inplace_add = self.tp_as_number.nb_inplace_add
    if nb_inplace_add is not None:
        return self.nb_inplace_add_impl(tx, other)
    sq_inplace_concat = self.tp_as_sequence.sq_inplace_concat
    if sq_inplace_concat is not None:
        return self.sq_inplace_concat_impl(tx, other)
    return slot_wrapper_add(tx, self, other)


# ---------------------------------------------------------------------------
# tp_richcompare -- comparison dispatch
#
# CPython comparison architecture (Objects/object.c, Objects/typeobject.c):
#
#   a == b  (COMPARE_OP bytecode)
#     -> PyObject_RichCompare(a, b, Py_EQ)
#       -> do_richcompare(a, b, Py_EQ)            # the 4-step algorithm
#         -> type(a)->tp_richcompare(a, b, Py_EQ)  # per-type slot
#
#   a.__eq__(b)  (attribute access)
#     -> type(a)->tp_getattro(a, "__eq__")          # descriptor protocol
#     -> returns wrapper bound to tp_richcompare
#     -> calling wrapper invokes tp_richcompare(a, b, Py_EQ) directly
#
# do_richcompare algorithm (Objects/object.c#L901-L955):
#   1. Subclass priority: if type(b) is a proper subclass of type(a),
#      try type(b)->tp_richcompare(b, a, swapped_op) first
#   2. Forward: type(a)->tp_richcompare(a, b, op)
#   3. Reflected: type(b)->tp_richcompare(b, a, swapped_op)
#   4. Fallback: identity for eq/ne, TypeError for ordering
#
# Dynamo implementation:
#
#   tp_richcompare_impl(self, tx, other, op) -- per-VT slot, analogous to
#     tp_richcompare.  Returns ConstantVariable(NotImplemented) when the
#     type does not handle the comparison.
#
#   generic_richcompare(tx, lhs, rhs, op) -- analogous to do_richcompare.
#     Implements the 4-step algorithm directly using tp_richcompare_impl
#     slots.  If a user comparison method graph-breaks, the Unsupported
#     exception propagates to COMPARE_OP (which has
#     @break_graph_if_unsupported) and runs the comparison eagerly.
#     UDOV.tp_richcompare_impl disables nested graph breaks on the resolved
#     funcvar so the InliningInstructionTranslator does not try to split
#     the inlined user method mid-function.
#
# Two entry points converge on tp_richcompare_impl:
#
#   COMPARE_OP (a == b):
#     -> BuiltinVariable dispatch -> generic_richcompare
#       -> tp_richcompare_impl (4-step: subclass priority, forward, reflected, fallback)
#
#   call_method("__eq__") (a.__eq__(b) in user code):
#     -> base.py call_method -> tp_richcompare_impl directly
#
# The call_method path calls tp_richcompare_impl directly (not
# generic_richcompare) to match CPython semantics: a.__eq__(b) invokes
# the type's tp_richcompare slot without do_richcompare's reflected-
# operand protocol, and may return NotImplemented.
# ---------------------------------------------------------------------------


is_richcompare_not_implemented = is_nb_not_implemented


def object_richcompare(
    self: VariableTracker,
    tx: "InstructionTranslatorBase",
    other: VariableTracker,
    op: str,
) -> VariableTracker:
    """object's tp_richcompare.

    https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L6263-L6305
    - __eq__: identity check, else NotImplemented
    - __ne__: delegates to tp_richcompare(self, other, Py_EQ) and inverts
    - ordering
    """
    if op == "__eq__":
        identity = vt_identity_compare(self, other)
        if identity is not None and identity.as_python_constant():
            return ConstantVariable.create(True)
        return ConstantVariable.create(NotImplemented)
    elif op == "__ne__":
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L6279-L6298
        # Safe to call as_python_constant(): only identity-based types use
        # object_richcompare, so eq_result is always True or NotImplemented.
        eq_result = self.tp_richcompare_impl(tx, other, "__eq__")
        if is_richcompare_not_implemented(eq_result):
            return eq_result
        return ConstantVariable.create(not eq_result.as_python_constant())
    else:
        return ConstantVariable.create(NotImplemented)


def python_constant_richcompare_impl(
    self: VariableTracker,
    tx: "InstructionTranslatorBase",
    other: VariableTracker,
    op: str,
) -> VariableTracker:
    """Constant-fold comparison for types with as_python_constant()."""
    if not self.is_python_constant() or not other.is_python_constant():
        return ConstantVariable.create(NotImplemented)
    self_val = self.as_python_constant()
    other_val = other.as_python_constant()
    try:
        result = getattr(type(self_val), op)(self_val, other_val)
    except TypeError as e:
        raise_observed_exception(TypeError, tx, args=list(e.args))
    return ConstantVariable.create(result)


# _Py_SwappedOp: https://github.com/python/cpython/blob/e76aa128fe/Objects/object.c#L987
_REFLECTED_OP: dict[str, str] = {
    "__lt__": "__gt__",
    "__gt__": "__lt__",
    "__le__": "__ge__",
    "__ge__": "__le__",
    "__eq__": "__eq__",
    "__ne__": "__ne__",
}

_OP_STR: dict[str, str] = {
    "__lt__": "<",
    "__le__": "<=",
    "__eq__": "==",
    "__ne__": "!=",
    "__gt__": ">",
    "__ge__": ">=",
}


def generic_richcompare(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    op: str,
) -> VariableTracker:
    """Dynamo's do_richcompare.

    https://github.com/python/cpython/blob/e76aa128fe/Objects/object.c#L994-L1039

    Implements the 4-step algorithm directly using tp_richcompare_impl slots.
    Graph breaks inside user comparison methods propagate to COMPARE_OP
    (which runs eagerly) because UDOV.tp_richcompare_impl disables nested
    graph breaks on the resolved funcvar.
    """
    reflected = _REFLECTED_OP[op]

    try:
        v_type = v.python_type()
    except NotImplementedError:
        v_type = None
    try:
        w_type = w.python_type()
    except NotImplementedError:
        w_type = None

    checked_reverse = False

    # Step 1: subclass priority
    if (
        v_type is not None
        and w_type is not None
        and v_type is not w_type
        and issubclass(w_type, v_type)
    ):
        checked_reverse = True
        result = w.tp_richcompare_impl(tx, v, reflected)
        if not is_richcompare_not_implemented(result):
            return result

    # Step 2: forward
    result = v.tp_richcompare_impl(tx, w, op)
    if not is_richcompare_not_implemented(result):
        return result

    # Step 3: reflected (if not already tried)
    if not checked_reverse:
        result = w.tp_richcompare_impl(tx, v, reflected)
        if not is_richcompare_not_implemented(result):
            return result

    # Step 4: fallback
    if op in ("__eq__", "__ne__"):
        identity = vt_identity_compare(v, w)
        if identity is not None:
            if op == "__ne__":
                return ConstantVariable.create(not identity.as_python_constant())
            return identity
        unimplemented(
            gb_type="richcompare identity fallback undetermined",
            context=f"generic_richcompare({v}, {w}, {op})",
            explanation="Cannot determine object identity for comparison fallback.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )
    else:
        raise_type_error(
            tx,
            f"'{_OP_STR[op]}' not supported between instances of "
            f"'{v.python_type_name()}' and '{w.python_type_name()}'",
        )


def generic_richcompare_bool(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    op: str,
) -> VariableTracker:
    """Dynamo's PyObject_RichCompareBool for eq/ne.

    https://github.com/python/cpython/blob/e76aa128fe/Objects/object.c#L1046-L1080

    Like generic_richcompare, but with an identity shortcut first: if v
    and w are the same Python object, eq is True and ne is False. This
    matters for NaN (nan is nan -> True, but nan == nan -> False). Used by
    container comparisons (list_richcompare, tuplerichcompare) which call
    PyObject_RichCompareBool per element in CPython.
    """
    if op not in ("__eq__", "__ne__"):
        raise AssertionError(f"generic_richcompare_bool only supports eq/ne, got {op}")
    identity = vt_identity_compare(v, w)
    if identity is not None and identity.as_python_constant():
        return ConstantVariable.create(op == "__eq__")
    return generic_richcompare(tx, v, w, op)


def generic_hash_impl(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> tuple[int, bool]:
    """Internal API: compute hash as (value, is_fake).

    Dispatches to the VT's hash_impl.  Called by generic_hash (which wraps
    the result in a VT), container hash_impls (which propagate is_fake),
    and HashableTracker (which just needs the int).
    """
    return obj.hash_impl(tx)


def generic_hash(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """User-facing API: mirrors PyObject_Hash, returns a VariableTracker.

    https://github.com/python/cpython/blob/e76aa128fe/Objects/object.c#L1101-L1115

    Wraps the result in ConstantVariable or FakeIdVariable depending on
    whether the hash depends on a sourceless object's identity.
    """
    from .constant import ConstantVariable, FakeIdVariable, FakeValueKind

    h, is_fake = generic_hash_impl(tx, obj)
    if is_fake:
        return FakeIdVariable(h, kind=FakeValueKind.HASH)
    return ConstantVariable.create(h)


def pysequence_contains(
    tx: "InstructionTranslatorBase", obj: "VariableTracker", item: "VariableTracker"
) -> "VariableTracker":
    """
    Implements PySequence_Contains semantics for VariableTracker objects.

    If the object has sq_contains (i.e., __contains__), calls obj.sq_contains_impl(tx, item).
    Otherwise falls back to iterating over obj and comparing each element.
    """
    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L2272-L2283
    sq_contains = obj.tp_as_sequence.sq_contains
    if sq_contains is not None:
        return obj.sq_contains_impl(tx, item)
    else:
        # iter fallback handles both __iter__ and __getitem__ sequence protocol cases
        it = generic_getiter(tx, obj)
        return VariableTracker.build(
            tx, polyfills.impl_CONTAINS_OP_fallback
        ).call_function(tx, [item, it], {})


# Metaclasses whose __subclasscheck__ Dynamo can't trace but whose
# behavior we're willing to observe at trace time via Python's issubclass.
# Each entry trades fidelity to the metaclass's side effects (e.g. ABC's
# subclass cache mutation) for coverage of the common case.
_CONSTANT_FOLD_SUBCLASSCHECK_METACLASSES: tuple[type, ...] = (
    abc.ABCMeta,
    torch._C._TensorMeta,  # actually just type.__subclasscheck__, but easier to list it here
    enum.EnumMeta,
)


def generic_issubclass(
    tx: "InstructionTranslatorBase",
    derived: VariableTracker,
    cls: VariableTracker,
) -> VariableTracker:
    """Mirrors CPython's PyObject_IsSubclass / object_issubclass.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L2766-L2823

    This only attempts to replicate object_issubclass, otherwise we delegate to cpython
    """
    derived_py = derived.get_real_python_backed_value()
    cls_py = cls.get_real_python_backed_value()
    if derived_py is NO_SUCH_SUBOBJ or cls_py is NO_SUCH_SUBOBJ:
        unimplemented(
            gb_type="issubclass() with unsupported arguments",
            context=f"issubclass({derived}, {cls})",
            explanation="Arguments to issubclass() must be backed by python values.",
            hints=[
                "Make sure your arguments are types.",
                *graph_break_hints.USER_ERROR,
                *graph_break_hints.SUPPORTABLE,
            ],
        )
    cls_type = maybe_get_python_type(cls)

    # Step 1: PyType_CheckExact fast path — abstract.c L2772
    if cls_type is type:
        try:
            return ConstantVariable.create(
                issubclass(
                    derived_py,  # pyrefly: ignore [bad-argument-type]
                    cls_py,  # pyrefly: ignore [invalid-argument]
                )
            )
        except TypeError as e:
            raise_observed_exception(TypeError, tx, args=list(e.args))

    # Step 2: PEP 604 Union (e.g. ``int | str``) — abstract.c L2779-2781.
    union_types = {types.UnionType}
    if sys.version_info < (3, 14):
        union_types.add(
            typing._UnionGenericAlias  # pyrefly: ignore [missing-attribute]
        )
    if cls_type in union_types:
        # TODO can trace this once TypingVariable is removed
        args = typing.get_args(cls_py)
        cls = VariableTracker.build(tx, args)

    # Step 3: tuple of classes — abstract.c L2783-2799.  Check for
    # TupleVariable instead of tuple to make the type checker happy.
    from .lists import TupleVariable

    if isinstance(cls, TupleVariable):
        for item in cls.items:
            r = generic_issubclass(tx, derived, item)
            if isinstance(r, ConstantVariable) and r.value:
                return ConstantVariable.create(True)
        return ConstantVariable.create(False)

    # Allowlist short-circuit for Step 4: constant-fold via Python's
    # issubclass for metaclasses whose ``__subclasscheck__`` Dynamo can't
    # trace (see _CONSTANT_FOLD_SUBCLASSCHECK_METACLASSES).  Note that ABCMeta
    # is problematic in particular since it caches registered subclasses.
    # Ideally this should be traced or guarded
    if isinstance(cls_py, type) and issubclass(
        type(cls_py), _CONSTANT_FOLD_SUBCLASSCHECK_METACLASSES
    ):
        try:
            return ConstantVariable.create(
                issubclass(
                    derived_py,  # pyrefly: ignore [bad-argument-type]
                    cls_py,
                )
            )
        except TypeError as e:
            raise_observed_exception(TypeError, tx, args=list(e.args))

    # TypeError gate, mirroring abstract.c L2822 ``recursive_issubclass``:
    # CPython reaches that fallback when ``_PyObject_LookupSpecial`` for
    # ``__subclasscheck__`` returns NULL, and its first action is
    # ``check_class(cls, ...)`` which raises this TypeError.  We check
    # eagerly because Dynamo's ``call_method`` below would graph-break
    # rather than cleanly signal "no such method".
    if not isinstance(cls_py, type):
        raise_type_error(
            tx,
            "issubclass() arg 2 must be a class, a tuple of classes, or a union",
        )

    # Step 4: general case — call ``__subclasscheck__`` on cls's metaclass
    # (abstract.c L2801-2815).  Runs user code on a custom metaclass.
    result = cls.call_method(tx, "__subclasscheck__", [derived], {})

    # Coerce to bool (PyObject_IsTrue, abstract.c L2812).
    return generic_is_true(tx, result)


# ── tp_getattro ──────────────────────────────────────────────────────
#
# Dynamo's PyObject_GetAttr / PyObject_GenericGetAttr.
#
# CPython references:
#   PyObject_GetAttr:        Objects/object.c#L1259-L1283
#   PyObject_GenericGetAttr: Objects/object.c#L1611-L1683
#   _PyType_LookupRef:       Objects/typeobject.c#L5208-L5262
#
# PyObject_GetAttr(obj, name):
#     tp = Py_TYPE(obj)
#     if tp->tp_getattro:
#         return tp->tp_getattro(obj, name)
#     ...
#     raise AttributeError
#
# Dispatch path:
#     LOAD_ATTR / getattr() -> GetAttrBuiltinVariable -> generic_getattr
#         -> obj.tp_getattro_impl(tx, name)
#
# The base VariableTracker.tp_getattro_impl tries object_generic_getattr()
# first (MRO walk + descriptor protocol), falling back to const_getattr
# on _UnhandledDescriptorError.  Callers of object_generic_getattr
# directly must handle _UnhandledDescriptorError at every step (2, 4,
# and 7), not just for unrecognized descriptor types.
#
# VTs with custom tp_getattro (TensorVariable, NNModuleVariable,
# UserDefinedClassVariable, SuperVariable) override tp_getattro_impl.

_NO_DEFAULT = object()


_UnhandledDescriptorError = UnhandledDescriptorError


def mro_lookup(py_type: type, name: str) -> object:
    """Walk py_type.__mro__ to find *name* in the class hierarchy.

    Mirrors CPython's _PyType_LookupRef (Objects/typeobject.c).
    Searches only the class chain (py_type.__mro__), NOT the metaclass
    chain.  Returns the raw descriptor/value from the class __dict__,
    or NO_SUCH_SUBOBJ if not found.
    """
    for base in py_type.__mro__:
        if name in base.__dict__:
            return base.__dict__[name]
    return NO_SUCH_SUBOBJ


def _resolve_descriptor_get(
    tx: "InstructionTranslatorBase",
    type_attr: object,
    obj: VariableTracker,
    class_vt: VariableTracker,
    source: "Source | None",
) -> "VariableTracker | None":
    """Invoke tp_descr_get on a type attribute if it's a descriptor.

    Handles all descriptor types that have dedicated VTs with
    tp_descr_get_impl.  Returns None if type_attr is not a recognized
    descriptor type (plain class variable).
    """
    import types as _types

    if isinstance(type_attr, property):
        prop_vt = variables.PropertyVariable(type_attr, source=source)
        return prop_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.MemberDescriptorType):
        md_vt = variables.MemberDescriptorVariable(type_attr, source=source)
        return md_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.GetSetDescriptorType):
        gs_vt = variables.GetSetDescriptorVariable(type_attr, source=source)
        return gs_vt.tp_descr_get_impl(tx, obj, class_vt)
    _tuplegetter = collections._tuplegetter  # pyrefly: ignore[missing-attribute]
    if isinstance(type_attr, _tuplegetter):
        tg_vt = variables.TupleGetterVariable(type_attr, source=source)
        return tg_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, staticmethod):
        sm_vt = variables.StaticMethodVariable(type_attr, source=source)
        return sm_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, classmethod):
        cm_vt = variables.ClassMethodVariable(type_attr, source=source)
        return cm_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.ClassMethodDescriptorType):
        cmd_vt = variables.ClassMethodDescriptorVariable(type_attr, source=source)
        return cmd_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.WrapperDescriptorType):
        wd_vt = variables.WrapperDescriptorVariable(
            type_attr, owner=class_vt, source=source
        )
        return wd_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.MethodDescriptorType):
        md_vt = variables.MethodDescriptorVariable(
            type_attr, owner=class_vt, source=source
        )
        return md_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.FunctionType):
        return variables.UserMethodVariable(type_attr, obj, source=source)

    return None


# BuiltinFunctionType is intentionally excluded: _resolve_descriptor_get
# does not handle it, so it falls through to _UnhandledDescriptorError
# and generic_getattr's GetAttrVariable fallback.
_METHOD_TYPES = (
    types.FunctionType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
)


def _is_method_type(type_attr: object) -> bool:
    return isinstance(type_attr, _METHOD_TYPES)


def _has_custom_call_method(obj: VariableTracker) -> bool:
    for cls in type(obj).__mro__:
        if cls is VariableTracker:
            return False
        if "call_method" in cls.__dict__:
            return True
    return False


def object_generic_getattr(
    tx: "InstructionTranslatorBase",
    obj: VariableTracker,
    name: str,
) -> VariableTracker:
    """Dynamo's PyObject_GenericGetAttr.

    https://github.com/python/cpython/blob/e76aa128fe/Objects/object.c#L1611-L1683

    Implements the standard attribute lookup algorithm using the VT's
    python_type() for MRO walking, and hooks on the VT for instance dict
    and __getattr__ fallback.

    Steps:
      1. MRO walk on python_type() for type_attr
      2. Data descriptor -> invoke tp_descr_get
      3. Instance dict -> obj.lookup_instance_dict(tx, name)
      4. Non-data descriptor -> invoke tp_descr_get
      5. Plain class variable -> wrap in VT
      6. __getattr__ fallback -> obj.call_getattr_fallback(tx, name)
      7. AttributeError
    """
    from .user_defined import is_data_descriptor

    py_type = obj.python_type()
    source = obj.source and AttrSource(obj.source, name)

    # Step 1: MRO walk.
    type_attr = mro_lookup(py_type, name)

    # Step 2: Data descriptor takes priority over instance dict.
    if type_attr is not NO_SUCH_SUBOBJ and is_data_descriptor(type_attr):
        class_vt = VariableTracker.build(tx, py_type)
        result = _resolve_descriptor_get(tx, type_attr, obj, class_vt, source)
        if result is not None:
            return result
        raise _UnhandledDescriptorError(
            f"object_generic_getattr: unhandled data descriptor "
            f"{type(type_attr)} for {name}"
        )

    # Step 3: Instance dict (tp_dictoffset).
    instance_result = obj.lookup_instance_dict(tx, name)
    if instance_result is not None:
        return instance_result

    # Step 4: Non-data descriptor with __get__.
    if type_attr is not NO_SUCH_SUBOBJ and hasattr(type(type_attr), "__get__"):
        # If the VT dispatches this method through call_method (a hand-written
        # override or a tp_methods entry), return a CallMethodVariable that
        # dispatches through call_method instead of inlining the resolved
        # method directly.  This preserves custom tracing logic (side effects,
        # graph nodes, suppression) that MRO-based resolution via
        # UserMethodVariable would bypass.
        if _is_method_type(type_attr) and (
            obj._lookup_tp_table(name, "tp_methods") is not None
            or _has_custom_call_method(obj)
        ):
            return variables.CallMethodVariable(obj, name, source=source)

        class_vt = VariableTracker.build(tx, py_type)
        result = _resolve_descriptor_get(tx, type_attr, obj, class_vt, source)
        if result is not None:
            return result
        raise _UnhandledDescriptorError(
            f"object_generic_getattr: unhandled non-data descriptor "
            f"{type(type_attr)} for {name}"
        )

    # Step 5: Plain class variable (no __get__).
    if type_attr is not NO_SUCH_SUBOBJ:
        return VariableTracker.build(tx, type_attr, source)

    # Step 6: __getattr__ fallback.
    getattr_result = obj.call_getattr_fallback(tx, name)
    if getattr_result is not None:
        return getattr_result

    # Step 7: Attribute not found -- signal to caller to fall back.
    raise _UnhandledDescriptorError(
        f"object_generic_getattr: '{py_type.__name__}' has no attribute '{name}'"
    )


def generic_getattr(
    tx: "InstructionTranslatorBase",
    obj: VariableTracker,
    name: str,
    default: "VariableTracker | object" = _NO_DEFAULT,
) -> VariableTracker:
    """Dynamo's PyObject_GetAttr: attribute access dispatch.

    Checks side effects for pending attribute mutations, then dispatches
    to obj.tp_getattro_impl(tx, name).  On NotImplementedError, falls back
    to GetAttrVariable (deferred resolution).
    """
    from .user_defined import is_data_descriptor

    # NOTE [Tensor "grad" and "_grad" attr]
    if obj.is_tensor() and name == "_grad":
        name = "grad"

    # Side effects: check for pending attribute mutations.
    if tx.output.side_effects.has_pending_mutation_of_attr(obj, name):
        if not isinstance(obj, variables.UserDefinedObjectVariable):
            return tx.output.side_effects.load_attr(obj, name)
        if tx.output.side_effects.has_pending_mutation_of_attr(
            obj, name, AttrMutationKind.INSTANCE_DICT
        ):
            value = tx.output.side_effects.load_attr(obj, name, deleted_ok=True)
            type_attr = obj.lookup_class_mro_attr(name)
            if not isinstance(value, variables.DeletedVariable) and (
                type_attr is NO_SUCH_SUBOBJ or not is_data_descriptor(type_attr)
            ):
                return value

    # Handle default for getattr(obj, name, default).
    if default is not _NO_DEFAULT:
        hasattr_var = obj.call_obj_hasattr(tx, name)
        if not hasattr_var.is_constant_match(True, False):
            raise AssertionError(
                f"hasattr_var must be a constant True or False, got {hasattr_var}"
            )
        if not hasattr_var.as_python_constant():
            return default  # type: ignore[return-value]

    # tp_getset/tp_members are data descriptors: resolve ahead of the VT's
    # tp_getattro so a tp_getattro_impl override need not repeat the consult.
    getset = obj.lookup_tp_getset_member(name)
    if getset is not None:
        result = getset.getter(obj, tx)
        if result is not None:
            return result

    # Core dispatch: call the VT's tp_getattro_impl (tp_getattro).
    source = obj.source and AttrSource(obj.source, name)
    try:
        return obj.tp_getattro_impl(tx, name)
    except AsPythonConstantNotImplementedError:
        raise
    except NotImplementedError:
        return variables.GetAttrVariable(obj, name, source=source)
<|endoftext|>from contextlib import ExitStack
from typing import Any, TYPE_CHECKING

import torch
from torch.fx import has_side_effect, Proxy

from ..bytecode_transformation import create_call_function
from ..exc import raise_type_error, unimplemented
from ..graph_bytecode_inputs import get_external_object_by_index
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, CallFunctionNoArgsSource, ImportSource
from .base import Member, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import ContextWrappingVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from ..codegen import PyCodegen

from torch._library.custom_ops import custom_op


def _get_mempool_by_index(index: int) -> torch.cuda.MemPool:
    mempool = get_external_object_by_index(index)
    if not isinstance(mempool, torch.cuda.MemPool):
        raise RuntimeError(
            f"use_mem_pool expected a torch.cuda.MemPool object at index {index}"
        )
    return mempool


def _current_cuda_device_source() -> CallFunctionNoArgsSource:
    return CallFunctionNoArgsSource(
        AttrSource(AttrSource(ImportSource("torch"), "cuda"), "current_device")
    )


# These marker ops are intentional even though Inductor also annotates enclosed
# FX nodes. The annotations tell Inductor which scheduler nodes allocate in a
# user pool, while begin/end keep the pool lifetime visible as ordered FX
# side-effects for non-Inductor backends.
@custom_op("mempool::begin", mutates_args=())
def begin_mempool(device_index: int, mempool_index: int) -> None:
    mempool = _get_mempool_by_index(mempool_index)
    torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool(device_index, mempool.id)


@begin_mempool.register_fake
def _(device_index: int, mempool_index: int) -> None:
    pass


has_side_effect(torch.ops.mempool.begin.default)


@custom_op("mempool::end", mutates_args=())
def end_mempool(device_index: int, mempool_index: int) -> None:
    mempool = _get_mempool_by_index(mempool_index)
    torch.cuda.memory._cuda_endAllocateToPool(device_index, mempool.id)
    torch.cuda.memory._cuda_releasePool(device_index, mempool.id)


@end_mempool.register_fake
def _(device_index: int, mempool_index: int) -> None:
    pass


has_side_effect(torch.ops.mempool.end.default)


class CUDAMemPoolVariable(VariableTracker):
    """Represents a torch.cuda.MemPool object."""

    def __init__(
        self,
        proxy: Proxy,
        value: torch.cuda.MemPool,
        user_object_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.user_object_index = user_object_index

    def python_type(self) -> type:
        return torch.cuda.MemPool

    def get_real_python_backed_value(self) -> object:
        return self.value

    def _id(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if self.source:
            install_guard(self.source.make_guard(GuardBuilder.ID_MATCH))
        return ConstantVariable.create(self.value.id)

    tp_members = {
        "id": Member(_id),
    }

    def as_proxy(self) -> Proxy:
        return self.proxy

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self.source:
            return super().reconstruct(codegen)

        codegen.add_push_null(
            lambda: codegen.load_import_from(
                torch._dynamo.graph_bytecode_inputs.__name__,
                "get_external_object_by_index",
            )
        )
        codegen.append_output(codegen.create_load_const(self.user_object_index))
        codegen.extend_output(create_call_function(1, False))


class CUDAMemPoolContextVariable(ContextWrappingVariable):
    """Represents the torch.cuda.use_mem_pool context manager."""

    def __init__(
        self,
        mempool: CUDAMemPoolVariable,
        device_index: int,
        **kwargs: Any,
    ) -> None:
        self.mempool = mempool
        self.device_index = device_index
        super().__init__(target_values=(), initial_values=None, **kwargs)

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        mempool: VariableTracker,
        device: VariableTracker | None = None,
        **kwargs: Any,
    ) -> "CUDAMemPoolContextVariable":
        if not isinstance(mempool, CUDAMemPoolVariable):
            raise_type_error(
                tx,
                "torch.cuda.use_mem_pool() expected a torch.cuda.MemPool argument",
            )

        if device is None or (
            isinstance(device, ConstantVariable) and device.value is None
        ):
            install_guard(
                _current_cuda_device_source().make_guard(GuardBuilder.EQUALS_MATCH)
            )
            device_index = torch.cuda.current_device()
        elif device.is_python_constant():
            device_index = torch.cuda._utils._get_device_index(
                device.as_python_constant()
            )
        else:
            unimplemented(
                gb_type="torch.cuda.use_mem_pool with non-constant device",
                context=f"device={device}",
                explanation="Dynamo requires the device argument to torch.cuda.use_mem_pool to be a Python constant.",
                hints=[],
            )

        return CUDAMemPoolContextVariable(mempool, device_index, **kwargs)

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        stack = ExitStack()
        stack.enter_context(
            torch.fx.traceback.annotate(
                {
                    "mempool": self.mempool.user_object_index,
                    "mempool_device": self.device_index,
                }
            )
        )
        stack.enter_context(torch.fx.traceback.preserve_node_meta())
        self.set_cleanup_hook(tx, lambda: stack.close())
        tx.output.create_proxy(
            "call_function",
            torch.ops.mempool.begin,
            (self.device_index, self.mempool.user_object_index),
            {},
        )
        return ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        tx.output.create_proxy(
            "call_function",
            torch.ops.mempool.end,
            (self.device_index, self.mempool.user_object_index),
            {},
        )
        self.cleanup_assert()
        return ConstantVariable.create(None)

    def module_name(self) -> str:
        return "torch.cuda"

    def fn_name(self) -> str:
        return "use_mem_pool"

    def reconstruct_type(self, codegen: "PyCodegen") -> None:
        # Supporting graph breaks safely requires reconstructing the eager
        # torch.cuda.use_mem_pool context on the resume path while keeping the
        # compiled begin/end marker ops balanced around each compiled segment.
        unimplemented(
            gb_type="torch.cuda.use_mem_pool graph break",
            context=str(self),
            explanation="Dynamo doesn't support graph breaks inside torch.cuda.use_mem_pool.",
            hints=[],
        )
<|endoftext|>"""
Constant variable tracking in Dynamo.

This module is fundamental to Dynamo's ability to track and propagate constant
values during compilation, ensuring proper handling of Python literals and
maintaining type safety through the compilation process.
"""

from __future__ import annotations

import enum
import operator
from collections.abc import Iterable
from typing import Any, Literal, overload, TYPE_CHECKING

import torch
from torch._dynamo.source import GetItemSource

from .. import graph_break_hints, variables
from ..exc import raise_observed_exception, raise_type_error, unimplemented
from ..utils import (
    common_constant_types,
    istype,
    np,
    raise_args_mismatch,
    unpack_iterable,
)
from .base import ValueMutationNew, VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from .functions import UserFunctionVariable


class ConstantVariable(VariableTracker):
    """
    Variable tracker for Python literals and basic immutable types, with automatic
    routing support for collection types (lists, tuples, sets, etc.).

    The create() method intelligently constructs appropriate variable types for
    nested collections.
    """

    # PyLong_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L6585
    # PyFloat_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L1880
    # PyBool_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/boolobject.c#L171
    # PyUnicode_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/unicodeobject.c#L14931
    # PyBytes_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/bytesobject.c#L3017
    # PyComplex_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L1099
    # _PyNone_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/object.c#L2022
    _cpython_type = (int, float, str, bytes, bool, type(None), complex, type(...))

    @overload
    @staticmethod
    def create(value: None) -> ConstantVariable: ...

    @overload
    @staticmethod
    def create(value: bool) -> ConstantVariable: ...

    @overload
    @staticmethod
    def create(value: Any, **kwargs: Any) -> VariableTracker: ...

    @staticmethod
    def create(value: Any, **kwargs: Any) -> VariableTracker:
        """
        Create a `ConstantVariable` based on the given value, and supports
        automatic routing for collection types like `tuple` (in which case we'd
        create `ConstantVariable` for the leaf items).

        NOTE: the caller must install the proper guards if needed; most often
        the guard will be `CONSTANT_MATCH`.
        """
        # Return pre-allocated sentinels for None/True/False when there are
        # no extra kwargs (source, etc.) that would differentiate the instance.
        if not kwargs:
            match value:
                case None:
                    return CONSTANT_VARIABLE_NONE
                case True:
                    return CONSTANT_VARIABLE_TRUE
                case False:
                    return CONSTANT_VARIABLE_FALSE

        source = kwargs.get("source")

        # Routing for supported collection literals.
        if isinstance(value, set):
            items = [ConstantVariable.create(x) for x in value]
            return variables.SetVariable(items, **kwargs)  # type: ignore[arg-type]
        elif isinstance(value, frozenset):
            items = [ConstantVariable.create(x) for x in value]
            return variables.FrozensetVariable(items, **kwargs)  # type: ignore[arg-type]
        elif isinstance(value, slice):
            slice_args = (value.start, value.stop, value.step)
            slice_args_vars: list[VariableTracker] = [
                ConstantVariable.create(arg) for arg in slice_args
            ]
            return variables.SliceVariable(slice_args_vars, **kwargs)
        elif isinstance(value, (list, tuple)):
            items = []
            for i, x in enumerate(value):
                item_source = GetItemSource(source, i) if source else None
                items.append(
                    ConstantVariable.create(
                        x,
                        source=item_source,
                    )
                )
            return variables.BaseListVariable.cls_for(type(value))(items, **kwargs)

        return ConstantVariable(value, **kwargs)

    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not ConstantVariable.is_base_literal(value):
            raise AssertionError(
                f"Cannot construct `ConstantVariable` for value of type {type(value)}.\n"
                "\n"
                "This failure likely due to PyTorch-internal use of `ConstantVariable` on\n"
                "non-literal python values, please try using `VariableTracker.build` instead. If\n"
                "you believe it's a necessary and legitimate use case (the value is immutable and\n"
                "can't easily be represented with another `VariableTracker` class), please add\n"
                "its type to `common_constant_types`."
            )
        if np is not None and isinstance(value, np.number):
            self.value = value.item()
        else:
            self.value = value

    def as_proxy(self) -> Any:
        return self.value

    def __repr__(self) -> str:
        return f"ConstantVariable({type(self.value).__name__}: {repr(self.value)})"

    def as_python_constant(self) -> Any:
        return self.value

    def is_python_constant(self) -> Literal[True]:
        return True

    def tp_repr_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        return ConstantVariable.create(repr(self.value))

    def is_symnode_like(self) -> bool:
        return isinstance(self.value, (int, bool))

    def is_constant_match(self, *values: Any) -> bool:
        return self.value in values

    def is_constant_none(self) -> bool:
        return self.value is None

    @property
    def items(self) -> list[VariableTracker]:
        """
        Need this when adding a BaseListVariable and a ConstantVariable together.
        Happens in detectron2.
        """
        try:
            return [ConstantVariable.create(x) for x in self.value]
        except TypeError as e:
            raise NotImplementedError from e

    def getitem_const(
        self, tx: InstructionTranslatorBase, arg: VariableTracker
    ) -> VariableTracker:
        # bytes_subscript: https://github.com/python/cpython/blob/62a6e898e017c9878490544f6a227b8a187a949c/Objects/bytesobject.c#L1732
        # unicode_subscript: https://github.com/python/cpython/blob/62a6e898e017c9878490544f6a227b8a187a949c/Objects/unicodeobject.c#L13738
        if isinstance(self.value, (str, bytes)):
            from .lists import SliceVariable
            from .object_protocol import maybe_get_python_type, pyindex_check

            # _PyIndex_Check first, then slice, mirroring unicode_subscript /
            # bytes_subscript.
            if pyindex_check(maybe_get_python_type(arg)):
                arg = arg.nb_index_impl(tx)
            elif isinstance(arg, SliceVariable):
                return ConstantVariable.create(self.value[arg.as_index_slice(tx)])
            else:
                raise_type_error(
                    tx,
                    f"{self.python_type_name()} indices must be integers, "
                    f"not {arg.python_type_name()}",
                )
        return ConstantVariable.create(
            self.value[arg.as_python_constant()],
        )

    def sq_item_impl(
        self, tx: InstructionTranslatorBase, key: VariableTracker
    ) -> VariableTracker:
        # unicode_getitem: https://github.com/python/cpython/blob/62a6e898e01/Objects/unicodeobject.c#L13777
        # bytes_item: https://github.com/python/cpython/blob/62a6e898e01/Objects/bytesobject.c#L319
        # CPython's sq_item takes Py_ssize_t (already int from generic_getitem's
        # nb_index_impl).  Unlike mp_subscript, sq_item never handles slices.
        index = key.as_python_constant()
        try:
            return ConstantVariable.create(self.value[index])
        except IndexError as e:
            raise_observed_exception(IndexError, tx, args=list(e.args))

    @staticmethod
    def is_base_literal(obj: object) -> bool:
        return type(obj) in common_constant_types

    @staticmethod
    def is_literal(obj: object, cache: dict[int, object] | None = None) -> bool:
        if cache is None:
            cache = {}
        if id(obj) in cache:
            # no-op if there is a cyclical reference
            return True
        if type(obj) in (list, tuple, set, frozenset, torch.Size):
            cache[id(obj)] = obj
            return all(ConstantVariable.is_literal(x, cache) for x in obj)  # type: ignore[attr-defined]
        return ConstantVariable.is_base_literal(obj)

    def unpack_var_sequence(
        self, tx: InstructionTranslatorBase | None
    ) -> list[VariableTracker]:
        try:
            return [ConstantVariable.create(x) for x in self.as_python_constant()]
        except TypeError as e:
            raise NotImplementedError from e

    def hash_impl(self, tx: InstructionTranslatorBase) -> tuple[int, bool]:
        """Dynamo tracing rule for long_hash, float_hash, unicode_hash, etc."""
        return hash(self.value), False

    def tp_richcompare_impl(
        self, tx: InstructionTranslatorBase, other: VariableTracker, op: str
    ) -> VariableTracker:
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def tp_str_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        return ConstantVariable.create(str(self.value))

    def len_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        """Generic len for any constant value (sequence or mapping)."""
        try:
            return ConstantVariable.create(len(self.value))
        except TypeError as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def sq_length_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        """Sequence length - delegates to len_impl for constants."""
        return self.len_impl(tx)

    def mp_length_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        """Mapping length - delegates to len_impl for constants."""
        return self.len_impl(tx)

    def const_getattr(
        self, tx: InstructionTranslatorBase, name: str
    ) -> VariableTracker:
        if not hasattr(self.value, name):
            raise_observed_exception(AttributeError, tx, args=[name])
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError
        return member

    def sq_contains_impl(self, tx: InstructionTranslatorBase, item: VariableTracker):
        """Sequence contains for constants."""
        if item.is_python_constant():
            search = item.as_python_constant()
            try:
                result = search in self.value
                return ConstantVariable.create(result)
            except TypeError as e:
                raise_observed_exception(
                    type(e),
                    tx,
                    args=list(e.args),
                )
        return super().sq_contains_impl(tx, item)

    def tp_iter_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        from .lists import ListIteratorVariable

        if isinstance(self.value, Iterable):
            try:
                return ListIteratorVariable(
                    [ConstantVariable.create(c) for c in self.value],
                    mutation_type=ValueMutationNew(),
                )
            except NotImplementedError:
                pass
        return super().tp_iter_impl(tx)

    def call_method(
        self,
        tx: InstructionTranslatorBase,
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .tensor import SymNodeVariable

        if name == "format" and istype(self.value, str):
            return variables.BuiltinVariable(str.format).call_function(
                tx,
                [self, *args],
                kwargs,
            )
        elif name == "join" and istype(self.value, str):
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            arg_unpacked = unpack_iterable(tx, args[0])
            try:
                arg_const = [x.as_python_constant() for x in arg_unpacked]
                return ConstantVariable.create(self.value.join(arg_const))
            except NotImplementedError:
                return super().call_method(tx, name, args, kwargs)

        if any(isinstance(x, SymNodeVariable) for x in args):
            # Promote to SymNodeVariable for operations involving dynamic shapes.
            return variables.SymNodeVariable.create(
                tx, self.as_proxy(), self.value
            ).call_method(tx, name, args, kwargs)

        try:
            const_args = [a.as_python_constant() for a in args]
            const_kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
        except NotImplementedError:
            return super().call_method(tx, name, args, kwargs)

        if name == "__iter__":
            return self.tp_iter_impl(tx)

        if isinstance(self.value, str) and name in str.__dict__:
            method = getattr(self.value, name)
            try:
                result = method(*const_args, **const_kwargs)
            except Exception as e:
                raise_observed_exception(type(e), tx)
            # str.split/rsplit/splitlines return a fresh caller-owned list;
            # mark it mutable so in-place ops (.sort(), shuffle, etc.) are tracked.
            if name in ("split", "rsplit", "splitlines"):
                return ConstantVariable.create(result, mutation_type=ValueMutationNew())
            return ConstantVariable.create(result)
        elif isinstance(self.value, (float, int)) and hasattr(self.value, name):
            if not (args or kwargs):
                try:
                    return ConstantVariable.create(getattr(self.value, name)())
                except (OverflowError, ValueError) as exc:
                    raise_observed_exception(
                        type(exc),
                        tx,
                        args=list(exc.args),
                    )
            if (
                hasattr(operator, name)
                and len(args) == 1
                and args[0].is_python_constant()
            ):
                add_target = const_args[0]
                op = getattr(operator, name)
                if isinstance(
                    add_target, (torch.SymBool, torch.SymFloat, torch.SymInt)
                ):
                    # Addition between a non sym and sym makes a sym
                    proxy = tx.output.create_proxy(
                        "call_function", op, (self.value, add_target), {}
                    )
                    return SymNodeVariable.create(tx, proxy, add_target)
                else:
                    try:
                        return ConstantVariable.create(op(self.value, add_target))
                    except Exception as e:
                        raise_observed_exception(type(e), tx, args=list(e.args))
        elif isinstance(self.value, bytes) and name == "decode":
            method = getattr(self.value, name)
            return ConstantVariable.create(method(*const_args, **const_kwargs))
        elif type(self.value) is complex and name in complex.__dict__:
            method = getattr(self.value, name)
            try:
                return ConstantVariable.create(method(*const_args, **const_kwargs))
            except Exception as e:
                raise_observed_exception(type(e), tx)

        if name == "__round__" and len(args) == 1 and args[0].is_python_constant():
            try:
                return ConstantVariable.create(
                    round(self.value, args[0].as_python_constant())
                )
            except Exception as e:
                raise_observed_exception(type(e), tx, args=list(e.args))
        return super().call_method(tx, name, args, kwargs)

    def call_tree_map(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if self.value is None:
            none_is_leaf_var = tree_map_kwargs.get("none_is_leaf")
            if none_is_leaf_var is not None:
                try:
                    none_is_leaf = bool(none_is_leaf_var.as_python_constant())
                except NotImplementedError:
                    return self._tree_map_fallback(
                        tx,
                        tree_map_fn,
                        map_fn,
                        rest,
                        tree_map_kwargs,
                    )
            else:
                tree_map_module = getattr(
                    getattr(tree_map_fn, "fn", None), "__module__", ""
                )
                # torch.utils._pytree and torch.utils._cxx_pytree treat None as a leaf
                # by default, while optree keeps it as an internal node unless
                # none_is_leaf=True is provided.
                none_is_leaf = not tree_map_module.startswith("optree")
            if none_is_leaf:
                return map_fn.call_function(tx, [self, *rest], {})
            else:
                for other in rest:
                    if not other.is_constant_none():
                        return self._tree_map_fallback(
                            tx,
                            tree_map_fn,
                            map_fn,
                            rest,
                            tree_map_kwargs,
                        )
                return self.clone()
        if isinstance(self.value, (int, float, bool, complex, str, bytes, torch.dtype)):
            return map_fn.call_function(tx, [self, *rest], {})
        return super().call_tree_map(
            tx,
            tree_map_fn,
            map_fn,
            rest,
            tree_map_kwargs,
        )

    def reconstruct_pycode(self, codegen) -> str:
        return repr(self.value)

    def get_id(self, tx: InstructionTranslatorBase) -> int | None:
        # Singletons have guaranteed stable identity across the process lifetime.
        if self.value is None or self.value is True or self.value is False:
            return id(self.value)
        # Sourceful constants resolve via source like any other sourceful VT.
        # Sourceless non-singleton constants (e.g. literal 42 in compiled code)
        # get FakeIdVariable — CPython interning of small ints/strings is an
        # implementation detail users shouldn't rely on.
        return super().get_id(tx)

    def get_real_python_backed_value(self) -> object:
        return self.value

    def nb_index_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # CPython: int and bool define nb_index (returns self for int,
        # int(self) for bool). All other constant types do not.
        from .object_protocol import type_implements_nb_index

        if type_implements_nb_index(type(self.value)):
            return ConstantVariable.create(operator.index(self.value))
        return super().nb_index_impl(tx)

    def nb_int_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # CPython: int defines nb_int (long_long, returns copy).
        # bool inherits nb_int from int via slot inheritance.
        # float defines nb_int (truncates toward zero via PyLong_FromDouble).
        return ConstantVariable.create(int(self.value))

    def nb_float_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # CPython: float defines nb_float (float_float, returns copy).
        # int defines nb_float (long_float, converts to float).
        # bool inherits nb_float from int via slot inheritance.
        return ConstantVariable.create(float(self.value))

    def _nb_binary_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        op: Any,
        type_check: Any,
        reverse: bool,
    ) -> VariableTracker:
        # Shared body for ConstantVariable's binary nb_* slots. Mirrors the
        # CPython contract: if the type doesn't implement the slot, return
        # NotImplemented so dispatch can try the reverse slot; otherwise run
        # ``op(v, w)`` and re-raise arithmetic exceptions as observed.
        if not type_check(type(self.value)):
            return ConstantVariable.create(NotImplemented)
        if not other.is_python_constant():
            return ConstantVariable.create(NotImplemented)
        self_, other_ = (other, self) if reverse else (self, other)
        v, w = self_.as_python_constant(), other_.as_python_constant()
        try:
            result = op(v, w)
        except (TypeError, ValueError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))
        if result is NotImplemented:
            return ConstantVariable.create(NotImplemented)
        return VariableTracker.build(tx, result)

    def nb_lshift_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: only int defines nb_lshift; bool inherits via slot inheritance.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5489 (long_lshift)
        from .object_protocol import type_implements_nb_lshift

        return self._nb_binary_impl(
            tx, other, operator.lshift, type_implements_nb_lshift, reverse
        )

    def nb_rshift_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: only int defines nb_rshift; bool inherits via slot inheritance.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5526 (long_rshift)
        from .object_protocol import type_implements_nb_rshift

        return self._nb_binary_impl(
            tx, other, operator.rshift, type_implements_nb_rshift, reverse
        )

    def nb_floor_divide_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int and float define nb_floor_divide; bool inherits int's.
        # complex does NOT (complex floor division raises TypeError).
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L4057 (long_floor_div)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L660 (float_floor_div)
        if not isinstance(self.value, (int, float)):
            return ConstantVariable.create(NotImplemented)
        if not other.is_python_constant():
            return ConstantVariable.create(NotImplemented)
        self_, other_ = (other, self) if reverse else (self, other)
        v, w = self_.as_python_constant(), other_.as_python_constant()
        try:
            return VariableTracker.build(tx, v // w)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def nb_true_divide_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int, float, and complex define nb_true_divide; bool inherits int's.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L4146 (long_true_divide)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L618 (float_div)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L518 (complex_div)
        if not isinstance(self.value, (int, float, complex)):
            return ConstantVariable.create(NotImplemented)
        if not other.is_python_constant():
            return ConstantVariable.create(NotImplemented)
        self_, other_ = (other, self) if reverse else (self, other)
        v, w = self_.as_python_constant(), other_.as_python_constant()
        try:
            return VariableTracker.build(tx, v / w)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def nb_remainder_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int and float define nb_remainder (bool inherits int's); complex
        # does NOT. str/bytes/bytearray define nb_remainder for %-formatting.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L4116 (long_mod)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L640 (float_rem)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/unicodeobject.c#L15170 (unicode_mod)
        if not isinstance(self.value, (int, float, str, bytes, bytearray)):
            return ConstantVariable.create(NotImplemented)
        if not other.is_python_constant():
            return ConstantVariable.create(NotImplemented)
        self_, other_ = (other, self) if reverse else (self, other)
        v, w = self_.as_python_constant(), other_.as_python_constant()
        try:
            return VariableTracker.build(tx, v % w)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def nb_divmod_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int and float define nb_divmod (bool inherits int's); complex
        # does NOT. Returns a (quotient, remainder) tuple.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L4126 (long_divmod)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L680 (float_divmod)
        if not isinstance(self.value, (int, float)):
            return ConstantVariable.create(NotImplemented)
        if not other.is_python_constant():
            return ConstantVariable.create(NotImplemented)
        self_, other_ = (other, self) if reverse else (self, other)
        v, w = self_.as_python_constant(), other_.as_python_constant()
        try:
            return VariableTracker.build(tx, divmod(v, w))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def nb_or_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int, frozenset, and type all define nb_or.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5606 (long_or)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L1319 (set_or)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/typeobject.c#L6028-L6030 (type_as_number.nb_or)
        # bool inherits int's nb_or via slot inheritance.
        from .object_protocol import type_implements_nb_or

        return self._nb_binary_impl(
            tx, other, operator.or_, type_implements_nb_or, reverse
        )

    def nb_subtract_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int, float, and complex define nb_subtract; bool inherits int's.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L3819-L3824 (long_sub_method)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L598-L606 (float_sub)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L494-L503 (COMPLEX_BINOP(sub, diff))
        from .object_protocol import type_implements_nb_subtract

        return self._nb_binary_impl(
            tx, other, operator.sub, type_implements_nb_subtract, reverse
        )

    def nb_multiply_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # int, float, and complex all define nb_multiply (bool inherits int's slot).
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L4242-L4260 (long_mul)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L608-L616 (float_mul)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L506 (complex_mul)
        # str/bytes/bytearray do NOT have nb_multiply — they go through sq_repeat,
        # so this method should not see them as ``self``.
        from .object_protocol import type_implements_nb_multiply

        if not other.is_python_constant():
            return ConstantVariable.create(NotImplemented)
        other_val = other.as_python_constant()
        # CPython's nb_multiply (e.g. long_mul, float_mul) returns NotImplemented
        # whenever the other operand isn't numeric — sequence repetition is
        # then handled by PyNumber_Multiply's sq_repeat fallback. We mirror that
        # here because ``operator.mul`` performs the full protocol (including
        # sq_repeat) and would incorrectly short-circuit the dispatch.
        if not isinstance(other_val, (int, float, complex)):
            return ConstantVariable.create(NotImplemented)
        return self._nb_binary_impl(
            tx, other, operator.mul, type_implements_nb_multiply, reverse
        )

    def sq_repeat_impl(
        self,
        tx: InstructionTranslatorBase,
        count: VariableTracker,
    ) -> VariableTracker:
        # Only str / bytes are reachable via ConstantVariable since list, tuple,
        # bytearray have their own VTs.  ``count`` was already validated as an
        # index by pysequence_repeat -> nb_index_impl.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/unicodeobject.c#L12371 (unicode_repeat)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/bytesobject.c#L1448 (bytes_repeat)
        if not isinstance(self.value, (str, bytes)):
            raise AssertionError("Expected str or bytes in sq_repeat_impl")
        n = count.as_python_constant()
        try:
            return ConstantVariable.create(self.value * n)
        except (MemoryError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def nb_and_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int, frozenset, and type all define nb_and.
        # https://github.com/python/cpython/blob/3.13/Objects/longobject.c#L5574 (long_and)
        # https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1506-L1518 (set_and)
        # bool inherits int's nb_and via slot inheritance.
        from .object_protocol import type_implements_nb_and

        return self._nb_binary_impl(
            tx, other, operator.and_, type_implements_nb_and, reverse
        )

    def nb_xor_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int and frozenset define nb_xor.
        # https://github.com/python/cpython/blob/3.13/Objects/longobject.c#L5587 (long_xor)
        # https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1984-L1990 (set_xor)
        # bool inherits int's nb_xor via slot inheritance.
        from .object_protocol import type_implements_nb_xor

        return self._nb_binary_impl(
            tx, other, operator.xor, type_implements_nb_xor, reverse
        )

    def nb_negative_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # int: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5179-L5189
        # float: https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L839-L849
        # complex: https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L569-L575
        # bool inherits nb_negative from int via slot inheritance.
        return ConstantVariable.create(-self.value)

    def nb_positive_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # int: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5619 (long_long)
        # float: https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L1114 (float_float)
        # complex: https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L578 (complex_pos)
        # bool inherits nb_positive from int via slot inheritance.
        return ConstantVariable.create(+self.value)

    def nb_add_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int, float, and complex define nb_add; bool inherits int's.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L3800 (long_add)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L559 (float_add)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L720 (COMPLEX_BINOP(add, sum))
        from .object_protocol import type_implements_nb_add

        return self._nb_binary_impl(
            tx, other, operator.add, type_implements_nb_add, reverse
        )

    def nb_power_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        z: VariableTracker | None,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int, float, and complex all define nb_power (bool inherits int's).
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c (long_pow)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c (float_pow)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c (complex_pow)

        if z is not None:
            z = z.as_python_constant()

        if not other.is_python_constant():
            return ConstantVariable.create(NotImplemented)

        self_, other_ = (other, self) if reverse else (self, other)
        v, w = (
            self_.as_python_constant(),
            other_.as_python_constant(),
        )

        try:
            return VariableTracker.build(tx, pow(v, w, z))
        except TypeError:
            return ConstantVariable.create(NotImplemented)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def nb_power_z_impl(
        self,
        tx: InstructionTranslatorBase,
        v: VariableTracker,
        w: VariableTracker,
    ) -> VariableTracker:
        if not isinstance(self.value, int):
            return ConstantVariable.create(NotImplemented)
        if not v.is_python_constant() or not w.is_python_constant():
            return ConstantVariable.create(NotImplemented)
        v_val, w_val = v.as_python_constant(), w.as_python_constant()
        if not isinstance(v_val, int) or not isinstance(w_val, int):
            return ConstantVariable.create(NotImplemented)
        try:
            return VariableTracker.build(tx, pow(v_val, w_val, self.value))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def nb_absolute_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # int: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5184-L5190
        # float: https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L847-L850
        # complex: https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L588-L600
        #   _Py_c_abs can set errno=ERANGE on overflow, which complex_abs
        #   converts to OverflowError("absolute value too large").
        # bool inherits nb_absolute from int via slot inheritance.
        try:
            return ConstantVariable.create(abs(self.value))
        except OverflowError as e:
            raise_observed_exception(OverflowError, tx, args=list(e.args))

    def nb_invert_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        # int: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5163-L5177
        #   long_invert implements ~x as -(x+1).
        # bool inherits nb_invert from int via slot inheritance.
        return ConstantVariable.create(~self.value)


CONSTANT_VARIABLE_NONE = ConstantVariable(None)
CONSTANT_VARIABLE_TRUE = ConstantVariable(True)
CONSTANT_VARIABLE_FALSE = ConstantVariable(False)


class FakeValueKind(enum.Enum):
    ID = "id"
    HASH = "hash"


class FakeIdVariable(VariableTracker):
    """A compile-time-only id or hash value that can be used as a dict key but
    cannot be reconstructed across graph breaks.

    When dynamo evaluates ``id(x)`` or ``hash(x)`` on a variable tracker that
    has no corresponding runtime object, we mint a fake integer.  The ``kind``
    field tracks which builtin produced the value so that same-kind comparisons
    (e.g. ``id(a) != id(b)``) can be resolved at compile time while cross-kind
    comparisons graph-break.
    """

    # PyLong_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L6585
    _cpython_type = int

    _nonvar_fields = {
        "kind",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self, value: int, *, kind: FakeValueKind = FakeValueKind.ID, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.kind = kind

    def as_python_constant(self) -> int:
        return self.value

    def is_python_constant(self) -> bool:
        return False

    def python_type(self) -> type:
        return int

    def hash_impl(self, tx: InstructionTranslatorBase) -> tuple[int, bool]:
        return hash(self.value), True

    def tp_repr_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        # Mirrors int.__repr__: the value is an int, so str()/repr() yield its
        # decimal string. The distinct-but-compile-time-only identity carried by
        # the fake id is preserved in the resulting string, matching how
        # FakeIdVariable already resolves same-kind id()/hash() comparisons.
        return ConstantVariable.create(repr(self.value))

    def tp_richcompare_impl(
        self, tx: InstructionTranslatorBase, other: VariableTracker, op: str
    ) -> VariableTracker:
        if (
            isinstance(other, FakeIdVariable)
            and self.kind == other.kind
            and op in ("__eq__", "__ne__")
        ):
            result = (
                (self.value == other.value)
                if op == "__eq__"
                else (self.value != other.value)
            )
            return ConstantVariable.create(result)
        unimplemented(
            gb_type="Comparison on compile-time-only id or hash value",
            context=f"FakeIdVariable({self.value}) {op} {type(other).__name__}",
            explanation="Cannot compare a compile-time-only id() or hash() "
            "value. The comparison will run eagerly.",
            hints=[
                "Avoid comparing id() or hash() of objects created inside "
                "the compiled region against other values.",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    # Arithmetic on a fake id/hash (e.g. ``id(self) & 0x7fffffff`` inside a
    # custom __hash__) stays compile-time-only: CPython computes int op int,
    # but the operand is a fake address so the result must not be baked into
    # the graph. We mirror int's nb_* slots and return another fake int.
    def _operand_int(self, other: VariableTracker) -> int | None:
        if isinstance(other, FakeIdVariable):
            return other.value
        if other.is_python_constant():
            val = other.as_python_constant()
            if isinstance(val, int):
                return val
        return None

    def _nb_binary_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        op: Any,
        reverse: bool,
    ) -> VariableTracker:
        other_val = self._operand_int(other)
        if other_val is None:
            return ConstantVariable.create(NotImplemented)
        lhs, rhs = (other_val, self.value) if reverse else (self.value, other_val)
        try:
            result = op(lhs, rhs)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))
        return FakeIdVariable(result)

    def nb_add_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.add, reverse)

    def nb_subtract_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.sub, reverse)

    def nb_multiply_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.mul, reverse)

    def nb_floor_divide_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.floordiv, reverse)

    def nb_remainder_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.mod, reverse)

    def nb_and_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.and_, reverse)

    def nb_or_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.or_, reverse)

    def nb_xor_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.xor, reverse)

    def nb_lshift_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.lshift, reverse)

    def nb_rshift_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.rshift, reverse)

    def nb_negative_impl(self, tx):  # type: ignore[no-untyped-def]
        return FakeIdVariable(-self.value)

    def nb_positive_impl(self, tx):  # type: ignore[no-untyped-def]
        return FakeIdVariable(+self.value)

    def nb_absolute_impl(self, tx):  # type: ignore[no-untyped-def]
        return FakeIdVariable(abs(self.value))

    def nb_invert_impl(self, tx):  # type: ignore[no-untyped-def]
        return FakeIdVariable(~self.value)

    def nb_int_impl(self, tx):  # type: ignore[no-untyped-def]
        return FakeIdVariable(int(self.value), kind=self.kind)

    def nb_index_impl(self, tx):  # type: ignore[no-untyped-def]
        return FakeIdVariable(self.value, kind=self.kind)

    def reconstruct(self, codegen: Any) -> None:
        unimplemented(
            gb_type="Reconstruction of FakeIdVariable",
            context=str(self.value),
            explanation=(
                "A fake id produced by id() on a compile-time container "
                "cannot be reconstructed across a graph break."
            ),
            hints=[
                "Avoid using id() on containers in code that may graph-break.",
            ],
        )
<|endoftext|>"""
This module implements variable tracking for PyTorch optimizers during Dynamo tracing.

The OptimizerVariable class provides specialized handling for optimizer instances by:
- Optimizing the tracing of expensive optimizer initialization
- Managing optimizer state and parameter group tracking
- Handling tensor sources and guards for optimizer state tensors
- Supporting CUDA graph execution through static tensor address management
- Providing special handling for parameter gradients and optimizer state tensors

Key features include:
- Efficient initialization tracing via _init_group optimization
- Automatic marking of optimizer state tensors as static for CUDA graphs
- Proper source tracking for parameter groups, gradients, and state tensors
- Guard installation for optimizer state structure
- Support for both CPU and GPU tensor handling
- Cleanup of static tensor references via finalizers

The module integrates with Dynamo's broader tracing system while providing
optimizer-specific optimizations and safety guarantees.
"""

import logging
import weakref
from collections.abc import Iterable
from typing import Any, TYPE_CHECKING

import torch
from torch._dynamo.variables.tensor import TensorVariable
from torch._guards import Source
from torch._logging import getArtifactLogger
from torch.utils._pytree import tree_map_only

from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    ConstDictKeySource,
    DictGetItemSource,
    GetItemSource,
    GlobalWeakRefSource,
    GradSource,
)
from ..utils import GLOBAL_KEY_PREFIX, unpack_iterable
from .base import GetSet, VariableTracker
from .constant import ConstantVariable
from .dicts import ConstDictVariable
from .hashable import HashableTracker
from .lists import ListVariable
from .misc import GetAttrVariable
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


class ArgMappingException(Exception):
    pass


class GuardInstallException(Exception):
    pass


perf_hint_log = getArtifactLogger(__name__, "perf_hints")


def _is_static_for_cudagraphs(x: torch.Tensor) -> bool:
    from torch._inductor.cudagraph_trees import get_manager

    if x.is_cuda:
        manager = get_manager(x.device.index, False)
        is_static_address = torch._dynamo.utils.get_static_address_type(x) is not None
        if manager:
            if manager.current_node is None:
                raise AssertionError("CUDA graph manager has no current node")
            return (
                is_static_address
                or manager.current_node._is_cuda_graph_recorded_tensor(x)
            )
        else:
            return is_static_address
    else:
        # Don't print a warning for non-cuda tensors
        return True


class OptimizerVariable(UserDefinedObjectVariable):
    _nonvar_fields = {
        "grad_to_source",
        "tensor_to_source",
        "static_tensor_names",
        *UserDefinedObjectVariable._nonvar_fields,
    }

    def __init__(
        self,
        value: torch.optim.Optimizer,
        grad_to_source: dict[Any, GradSource] | None = None,
        static_tensor_names: set[str] | None = None,
        tensor_to_source: dict[torch.Tensor, Source] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(value, **kwargs)
        # pyrefly: ignore [bad-override]
        self.value: torch.optim.Optimizer = value
        self.grad_to_source = grad_to_source or {}
        self.tensor_to_source = tensor_to_source or {}
        self.static_tensor_names = static_tensor_names or set()

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        """This is an optimization to avoid tracing the very slow initialization of the optimizer"""
        if name == "_init_group":
            if not hasattr(self.value, "_init_group"):
                # Fallback: if the optimizer does not have _init_group, trace normally
                return super().call_method(tx, name, args, kwargs)
            try:
                self.graph_break_if_pending_mutation(tx)
                self.move_step_if_cpu()
                py_args, py_kwargs = self.get_python_args(*args, **kwargs)
                ret_val = self.value._init_group(*py_args, **py_kwargs)
                self.map_sources_and_install_guards(tx)
                self.update_list_args(tx, args, kwargs, py_args, py_kwargs)
                # stash a weak_ptr to optimizer to invalidate code
                # if the optimizer object dies
                mangled_name = f"__optimizer_{id(self.value)}"
                tx.store_global_weakref_by_id(mangled_name, self.value)
                self.create_finalizer(tx)

                # This is currently safe only because the only actual `ret_val`s returned
                # by the `_init_group` of existing optimizers are properties that are invariant
                # to the input tensors (e.g. dtype, layout). Changing these would trigger a
                # recompilation and hence never result in the wrong specialization of `ret_val`.
                return ConstantVariable.create(ret_val)
            except (ArgMappingException, GuardInstallException) as _:
                # trace normally if we can't map args or install guards correctly
                pass

        return super().call_method(tx, name, args, kwargs)

    # _init_group resolves to a GetAttrVariable so the call is intercepted in
    # call_method (in the typical case, a UserMethodVariable would inline it).
    def _get_init_group(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if not self.source:
            raise AssertionError("OptimizerVariable requires a source for _init_group")
        name = "_init_group"
        py_type = type(getattr(self.value, name))
        return GetAttrVariable(
            self, name, py_type=py_type, source=AttrSource(self.source, name)
        )

    # param_groups only runs setup side effects (static addresses, capturable
    # guards) and declines, falling through to the generic protocol.
    def _get_param_groups(self, tx: "InstructionTranslatorBase") -> None:
        from ..decorators import mark_static_address

        for group in self.value.param_groups:
            for p in group["params"]:
                mark_static_address(p, guard=True)

        self._set_capturable(tx)
        return None

    tp_getset = {
        "_init_group": GetSet(_get_init_group),
        "param_groups": GetSet(_get_param_groups),
    }

    def graph_break_if_pending_mutation(self, tx: "InstructionTranslatorBase") -> None:
        # If there are pending mutations on a parameter (due to using closure)
        # then we need to graph break to allow the python version of the parameter
        # to update, so that running _init_group will initialize the states with
        # the correct values
        for g in self.value.param_groups:
            for p in g["params"]:
                side_effects = tx.output.side_effects
                variable = side_effects.id_to_variable.get(id(p), None)
                if variable and side_effects.has_pending_mutation(variable):
                    from ..exc import unimplemented

                    unimplemented(
                        gb_type="optimizer: pending mutation on parameter",
                        context=f"variable: {variable}, parameter: {p}",
                        explanation="Pending mutations on a parameter (e.g. due to using closure) require a graph break.",
                        hints=[],
                    )

    def _set_capturable(self, tx: "InstructionTranslatorBase") -> None:
        from . import LazyVariableTracker

        # We only set capturable if params are on cuda
        # and the state is not initialized
        def safe_to_set_capturable(group: dict[str, Any]) -> bool:
            all_uninitialized = True
            all_gpu = True

            for p in group.get("params", []):
                all_gpu &= p.is_cuda or p.is_xpu
                all_uninitialized &= p not in self.value.state

            return "capturable" in group and all_uninitialized and all_gpu

        # track indices to not set so we don't need to
        # in the variable tracker realize the whole state
        # we handle guarding the state specially
        for group in self.value.param_groups:
            if safe_to_set_capturable(group):
                group["capturable"] = True

        source = self.source and AttrSource(self.source, "param_groups")
        param_groups_vt = LazyVariableTracker.realize_all(
            VariableTracker.build(tx, self.value.param_groups, source)
        )
        for param_group_vt in param_groups_vt.items:
            key = HashableTracker(ConstantVariable.create("capturable"))
            param_group_vt.items[key] = ConstantVariable.create(True)

    def get_python_args(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[Any], dict[str, Any]]:
        """Get python values equivalent to the variable tracker args"""

        def map_arg(arg: Any) -> Any:
            if isinstance(arg, VariableTracker) and arg.is_python_constant():
                return arg.as_python_constant()
            elif isinstance(arg, ListVariable) and not arg.items:
                # pyrefly: ignore [implicit-any]
                return []
            elif (
                isinstance(arg, ConstDictVariable)
                and isinstance(arg.source, GetItemSource)
                and isinstance(arg.source.base, AttrSource)
                and arg.source.base.member == "param_groups"
            ):
                return self.value.param_groups[arg.source.index]

            raise ArgMappingException

        new_args = [map_arg(arg) for arg in args]
        new_kwargs = {k: map_arg(v) for k, v in kwargs.items()}

        return new_args, new_kwargs

    # If users load an old state dictionary,
    # it's possible that step could be on the cpu
    # if this is the case, move it to the GPU
    # corresponding to the parameter
    # in most cases this is a no-op because the state is empty
    def move_step_if_cpu(self) -> None:
        for p, state in self.value.state.items():
            if "step" in state and state["step"].is_cpu:
                state["step"] = state["step"].to(p.device)

    def map_sources_and_install_guards(self, tx: "InstructionTranslatorBase") -> None:
        from ..decorators import mark_static_address
        from .lazy import LazyVariableTracker

        self.grad_to_source = {}
        self.tensor_to_source = {}

        def mark_static(x: Any) -> None:
            mark_static_address(x, guard=True)

        tree_map_only(torch.Tensor, mark_static, self.value.state)

        # Recursively realize the variable trackers for optim.state and
        # optim.param_groups, which recursively install the necessary guards.
        params_groups_source = self.source and AttrSource(self.source, "param_groups")
        param_groups_vt = LazyVariableTracker.realize_all(
            VariableTracker.build(tx, self.value.param_groups, params_groups_source)
        )

        state_source = self.source and AttrSource(self.source, "state")
        state_vt = VariableTracker.build(tx, self.value.state, state_source)

        # We need to realize the top level state dict to populate
        # the guard locals
        state_vt.realize()
        if state_source is None:
            raise AssertionError("state_source must not be None")
        tx.output.guard_on_key_order.add(state_source)

        # Populate self.grad_to_source and self.tensor_to_source so that we can
        # manually update_list_args
        for group, group_vt in zip(self.value.param_groups, param_groups_vt.items):
            # we assume here that all params within a param group
            # are initialized similarly
            if len(group["params"]) > 0:
                for param in group["params"]:
                    if param.grad is not None:
                        key_index = None
                        for i, k in enumerate(self.value.state.keys()):
                            if k is param:
                                key_index = i
                                break
                        if key_index:
                            LazyVariableTracker.realize_all(
                                VariableTracker.build(
                                    tx,
                                    self.value.state[param],
                                    DictGetItemSource(
                                        state_source,
                                        ConstDictKeySource(state_source, key_index),
                                    ),
                                )
                            )
                            break

            params_vt = group_vt.getitem_const(tx, ConstantVariable.create("params"))
            all_static = True
            non_static_grads = []
            for p, p_vt in zip(group["params"], unpack_iterable(tx, params_vt)):
                param_source = p_vt.source
                if param_source is None:
                    raise AssertionError(f"param {p} does not have a source")
                self.tensor_to_source[p] = param_source
                grad_source = GradSource(
                    param_source,
                    "grad",
                )

                if p.grad is not None:
                    self.grad_to_source[p.grad] = grad_source
                    if not _is_static_for_cudagraphs(p.grad):
                        all_static = False
                        non_static_grads.append(grad_source)
                else:
                    install_guard(grad_source.make_guard(GuardBuilder.CONSTANT_MATCH))

            # Note: to avoid spam logs only warn if perf hint artifact is enabled
            # (NB: artifacts are only enabled at the debug or warning level)
            if not all_static and perf_hint_log.isEnabledFor(logging.DEBUG):
                non_static_grad_names = [src.name for src in non_static_grads]
                perf_hint_log.warning(
                    (
                        "Grad tensors %s will be copied during cudagraphs execution."
                        "If using cudagraphs and the grad tensor addresses will be the same across runs,"
                        " use torch._dynamo.decorators.mark_static_address to elide this copy.",
                    ),
                    non_static_grad_names,
                )

        # We have to again iterate over the state dict to collect the
        # tensor_to_source dict. This is used for the finalizer.
        for idx, value in enumerate(self.value.state.values()):
            p_state_source = DictGetItemSource(
                state_source, ConstDictKeySource(state_source, idx)
            )
            tx.output.guard_on_key_order.add(p_state_source)
            for inner_idx, v in enumerate(value.values()):
                if (
                    isinstance(v, torch.Tensor)
                    and v not in self.grad_to_source
                    and v not in self.tensor_to_source
                ):
                    self.tensor_to_source[v] = DictGetItemSource(
                        p_state_source, ConstDictKeySource(p_state_source, inner_idx)
                    )

    def wrap_tensor(
        self, tx: "InstructionTranslatorBase", tensor_value: torch.Tensor
    ) -> TensorVariable:
        """Wrap state tensor in a TensorVariable"""
        from ..decorators import mark_static_address

        # If we have a source for a tensor already use it,
        # if we have not seen a tensor before, stash and use a
        # global weak ref source, since it must be an optimizer tensor
        # that we have missed

        if tensor_value in self.tensor_to_source:
            # mark these tensors as static for cudagraphs
            mark_static_address(tensor_value, guard=True)
            source = self.tensor_to_source[tensor_value]
            self.static_tensor_names.add(tx.output.module_key_name(source.name))
        elif tensor_value in self.grad_to_source:
            source = self.grad_to_source[tensor_value]
        else:
            # mark these tensors as static for cudagraphs
            mark_static_address(tensor_value, guard=True)

            global_name = tx.store_global_weakref_by_id(GLOBAL_KEY_PREFIX, tensor_value)
            source = GlobalWeakRefSource(global_name)
            self.static_tensor_names.add(tx.output.module_key_name(source.name))

        return VariableTracker.build(tx, tensor_value, source)

    def update_list_args(
        self,
        tx: "InstructionTranslatorBase",
        args: Iterable[VariableTracker],
        kwargs: Any,
        py_args: Iterable[Any],
        py_kwargs: Any,
    ) -> None:
        """Update the args and kwargs to the traced optimizer call"""
        for arg, py_arg in zip(args, py_args):
            if isinstance(arg, ListVariable):
                if not isinstance(py_arg, list):
                    raise AssertionError(
                        "py_arg should be a list in optimizer variable"
                    )
                for i, val in enumerate(py_arg):
                    tx.output.side_effects.mutation(arg)
                    if isinstance(val, torch.Tensor):
                        arg.items.append(self.wrap_tensor(tx, val))
                    else:
                        source = arg.source and GetItemSource(arg.source, i)
                        arg.items.append(VariableTracker.build(tx, val, source))

    def create_finalizer(self, tx: "InstructionTranslatorBase") -> None:
        names_to_delete = self.static_tensor_names
        value = self.value
        tc = tx.output.tracing_context

        def init_finalizer(gm: torch.fx.GraphModule) -> None:
            def clear_static_tensor_refs() -> None:
                for name in names_to_delete:
                    gm._buffers.pop(name, None)
                    gm._parameters.pop(name, None)
                    if tc.params_flat:
                        tc.params_flat.clear()
                    if tc.params_flat_unwrap_subclasses:
                        tc.params_flat_unwrap_subclasses.clear()

            weakref.finalize(value, clear_static_tensor_refs)

        tx.output.add_graph_finalizer(init_finalizer)
<|endoftext|>"""TorchDynamo support for __torch_function__ tensor subclasses.

This module implements support for tensor subclasses with __torch_function__ overrides.
A tensor subclass instance is represented as a TensorWithTFOverrideVariable, which handles
dispatching __torch_function__ on attribute accesses, method calls, and torch API calls.

Unsupported features:
- Triggering __torch_function__ on tensor subclass non-tensor custom attributes
- Graph breaking on mutating guardable tensor properties within a __torch_function__ context
  (can cause excessive recompiles in certain cases)
- Matching exact eager behavior of ignoring __torch_function__ objects in non-tensor
  argument positions of Torch API calls

Supported features:
- Static method implementations of __torch_function__ on custom objects (triggers on torch
  API calls with the object as any argument)
- Triggering __torch_function__ on torch API calls with tensor subclass arguments
- __torch_function__ calls on base tensor attribute access and method calls for tensor
  subclass instances
- Matches dispatch ordering behavior of eager __torch_function__ with subclass/object
  arguments in any position

See https://docs.google.com/document/d/1WBxBSvW3NXhRp9ncmtokJloMLCtF4AYNhJaffvHe8Kw/edit#heading=h.vacn73lozd9w
for more information on the design.
"""

import collections
import contextlib
import inspect
import operator
from collections.abc import Generator, Iterable
from types import TracebackType
from typing import Any, TYPE_CHECKING

import torch._C
import torch.utils._pytree as pytree
from torch._guards import Source
from torch.overrides import (
    _get_overloaded_args,
    get_default_nowrap_functions,
    TorchFunctionMode,
)
from torch.utils._device import DeviceContext

from .. import graph_break_hints
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..polyfills import NoEnterTorchFunctionMode
from ..source import AttrSource, GlobalSource, TorchFunctionModeStackSource, TypeSource
from ..utils import (
    class_has_getattribute,
    clear_torch_function_mode_stack,
    get_safe_global_name,
    has_torch_function,
    is_tensor_base_attr_getter,
    set_torch_function_mode_stack,
)
from .base import VariableTracker
from .constant import ConstantVariable
from .ctx_manager import GenericContextWrappingVariable
from .functions import UserMethodVariable
from .lazy import LazyVariableTracker
from .lists import TupleVariable
from .tensor import TensorSubclassVariable, TensorVariable
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


bin_ops = [
    operator.pow,
    operator.mul,
    operator.matmul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.add,
    operator.lt,
    operator.gt,
    operator.ge,
    operator.le,
    operator.ne,
    operator.eq,
    operator.sub,
    operator.ipow,
    operator.imul,
    operator.imatmul,
    operator.ifloordiv,
    operator.itruediv,
    operator.imod,
    operator.iadd,
    operator.isub,
]

bin_int_ops = [
    operator.and_,
    operator.or_,
    operator.xor,
    operator.iand,
    operator.ixor,
    operator.ior,
]

un_int_ops = [operator.invert]

tensor_and_int_ops = [
    operator.lshift,
    operator.rshift,
    operator.ilshift,
    operator.irshift,
    operator.getitem,
]

un_ops = [
    operator.abs,
    operator.pos,
    operator.neg,
    operator.not_,  # Note: this has a local scalar dense call
    operator.length_hint,
]


banned_attrs = [
    fn.__self__.__name__  # type: ignore[attr-defined]
    for fn in get_default_nowrap_functions()
    if is_tensor_base_attr_getter(fn)
]


class TorchFunctionModeVariable(GenericContextWrappingVariable):
    @staticmethod
    def is_supported_torch_function_mode(ty: type[TorchFunctionMode]) -> bool:
        # Supported in this sense means we can support graph breaks under the
        # context.
        # We are able to trace custom modes but if there are graph breaks under them
        # and they have a custom __enter__/__exit__ we don't handle this for the
        # same reason we don't handle generic context managers: there may be side effects
        # that are now affected by executing the function across two frames instead of one
        # Today we support the enter/exit of the default TorchFunctionMode as well as
        # DeviceContext (which is used for set_default_device)
        return issubclass(ty, (NoEnterTorchFunctionMode, DeviceContext)) or (
            not class_has_getattribute(ty)
            and inspect.getattr_static(ty, "__enter__") is TorchFunctionMode.__enter__
            and inspect.getattr_static(ty, "__exit__") is TorchFunctionMode.__exit__
        )

    def __init__(
        self,
        value: TorchFunctionMode | None,
        source: Source | None = None,
        **kwargs: Any,
    ) -> None:
        if value is not None:
            super().__init__(value, **kwargs)
        self.value = value
        # needed for BC with calling enter from CM code
        self.cm_obj = value  # type: ignore[assignment]
        self.source = source  # type: ignore[assignment]

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # This shouldn't be called unless we have a source
        if not self.source:
            raise AssertionError(
                "TorchFunctionModeVariable requires a source for reconstruct"
            )
        self.source.reconstruct(codegen)

    def module_name(self) -> str:
        return self.value.__module__

    def fn_name(self) -> str:
        return type(self.value).__name__

    def python_type(self) -> type:
        return type(self.value)

    def call_torch_function(
        self,
        tx: "InstructionTranslatorBase",
        fn: VariableTracker,
        types: TupleVariable,
        args: Iterable[Any],
        kwargs: dict[str, Any],
    ) -> VariableTracker:
        return call_torch_function(
            tx,
            get_torch_function_fn(tx, self),  # type: ignore[arg-type]
            fn,
            types,
            args,
            kwargs,
        )

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .torch import TorchInGraphFunctionVariable

        if isinstance(self.value, NoEnterTorchFunctionMode):
            return ConstantVariable.create(None)

        TorchInGraphFunctionVariable(
            torch._C._push_on_torch_function_stack
        ).call_function(tx, [self], {})
        return ConstantVariable.create(None)

    def exit(self, tx: "InstructionTranslatorBase", *args: Any) -> VariableTracker:
        from .torch import TorchInGraphFunctionVariable

        TorchInGraphFunctionVariable(torch._C._pop_torch_function_stack).call_function(
            tx, [], {}
        )
        return ConstantVariable.create(None)

    def reconstruct_type(self, codegen: "PyCodegen") -> None:
        ty = NoEnterTorchFunctionMode
        codegen(
            AttrSource(
                codegen.tx.import_source(ty.__module__),
                ty.__name__,
            )
        )

    def supports_graph_breaks(self) -> bool:
        return True

    def exit_on_graph_break(self) -> bool:
        return False


# Used to clear/restore the python torch function mode stack and temporarily restore it as needed
class TorchFunctionModeStackStateManager:
    def __init__(self) -> None:
        self.stack: list[Any] = []

    def __enter__(self) -> None:
        self.stack = torch.overrides._get_current_function_mode_stack()
        clear_torch_function_mode_stack()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        set_torch_function_mode_stack(self.stack)
        self.stack = []

    @contextlib.contextmanager
    def temp_restore_stack(self) -> Generator[None, None, None]:
        prev = torch.overrides._get_current_function_mode_stack()
        set_torch_function_mode_stack(self.stack)
        try:
            yield
        finally:
            set_torch_function_mode_stack(prev)


torch_function_mode_stack_state_mgr = TorchFunctionModeStackStateManager()


class SymbolicTorchFunctionState:
    def __init__(self, py_stack: Iterable[Any], skip_next: bool) -> None:
        # This is annoyingly complicated because of how the torch function subclass + mode C API was designed
        # There are two exposed C knobs here as contexts: torch._C.DisableTorchFunction and torch._C.DisableTorchFunctionSubclass
        # These are their definitions:
        # 1) torch._C._is_torch_function_enabled indicates that neither of the above knobs have been entered
        # (if either are entered, this will be False)
        # 2) torch._C._is_torch_function_mode_enabled indicates that either the torch mode stack is empty OR
        # torch._C.DisableTorchFunction has been entered
        # To disambiguate these and keep myself sane I added a C API to check whether all torch function
        # concepts (modes and subclasses) are enabled.
        # This only returns true iff we have not entered torch._C.DisableTorchFunction and allows us to separate
        # the stack length from the enablement state of torch function modes.
        # This is important because now if a mode is pushed while dynamo is tracing, we know whether
        # or not torch function modes are enabled and whether we should trace it.
        self.torch_function_subclass_enabled = torch._C._is_torch_function_enabled()

        # Mirrors PythonTorchFunctionTLS::skip_next in C++.
        # Set by _skip_one_hop_torch_function, consumed by has_torch_function.
        # The real TLS flag is consumed by save_global_state in
        # OutputGraph.__init__ (which runs before this) and passed in here.
        # On tracing failure, restore_graphstate restores the real TLS.
        self.skip_next = skip_next

        # This differs from the C API of the same name
        # this will only be false iff we have entered torch._C.DisableTorchFunction
        # and does not take into account the mode stack length, while the C API bundles these
        # two concepts
        self.torch_function_mode_enabled = (
            not torch._C._is_torch_function_all_disabled()
        )

        self.cur_mode = None

        TorchFunctionModeStackVariable.reset()

        self.mode_stack: collections.deque[TorchFunctionModeVariable] = (
            collections.deque()
        )

        for i, val in enumerate(py_stack):
            self.mode_stack.append(
                LazyVariableTracker.create(val, source=TorchFunctionModeStackSource(i))  # type: ignore[arg-type]
            )

    def in_torch_function_mode(self) -> bool:
        return len(self.mode_stack) > 0

    def pop_torch_function_mode(self) -> TorchFunctionModeVariable:
        return self.mode_stack.pop()

    def push_torch_function_mode(self, mode_var: TorchFunctionModeVariable) -> None:
        self.mode_stack.append(mode_var)

    def call_torch_function_mode(
        self,
        tx: "InstructionTranslatorBase",
        fn: VariableTracker,
        types: TupleVariable,
        args: Iterable[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        with self._pop_mode_for_inlining() as cur_mode:
            return cur_mode.call_torch_function(tx, fn, types, args, kwargs)

    @contextlib.contextmanager
    def _pop_mode_for_inlining(
        self,
    ) -> Generator[TorchFunctionModeVariable, None, None]:
        old_mode = self.cur_mode
        self.cur_mode = self.pop_torch_function_mode()  # type: ignore[assignment]
        try:
            yield self.cur_mode  # type: ignore[misc]
        finally:
            mode = self.cur_mode
            self.cur_mode = old_mode
            self.push_torch_function_mode(mode)  # type: ignore[arg-type]


class TorchFunctionModeStackVariable(VariableTracker):
    """Fake VT to use as a dummy object, indicating the presence of torch function mode stack mutation"""

    # singleton value representing the global torch function mode stack
    # singleton (it exists in C++)
    stack_value_singleton = object()

    # offset is used to track if we have inserted/removed a
    # device context which is always placed at the bottom of the stack
    # if a device context is inserted, the graph will run this mutation
    # so when we want to reconstruct any other modes on the stack
    # their indices should be shifted right by 1 (+1)
    # Conversely, if there was a device context on the stack, and the graph
    # mutates the stack to remove that context (set default device to None)
    # each of the indices of other modes should be shifted left by 1 (-1)
    offset = 0

    def __init__(
        self,
        source: Source,
        symbolic_stack: collections.deque[TorchFunctionModeVariable],
    ) -> None:
        self.source = source
        self.symbolic_stack = symbolic_stack

    @classmethod
    def reset(cls) -> None:
        cls.offset = 0

    @classmethod
    def register_mutation(cls, tx: "InstructionTranslatorBase") -> None:
        if cls.stack_value_singleton not in tx.output.side_effects:
            var = cls(
                source=Source(),
                symbolic_stack=tx.symbolic_torch_function_state.mode_stack,
            )
            tx.output.side_effects.track_mutable(cls.stack_value_singleton, var)
            tx.output.side_effects.mutation(var)

    @classmethod
    def register_device_context_insertion(cls, tx: "InstructionTranslatorBase") -> None:
        stack = tx.symbolic_torch_function_state.mode_stack
        if stack and cls.is_device_context(stack[0]):
            return
        else:
            cls.offset += 1
            stack.insert(
                0,
                TorchFunctionModeVariable(
                    None, source=TorchFunctionModeStackSource(-cls.offset)
                ),
            )

    @classmethod
    def clear_default_device(cls, tx: "InstructionTranslatorBase") -> None:
        stack = tx.symbolic_torch_function_state.mode_stack
        if stack and cls.is_device_context(stack[0]):
            stack.popleft()
            cls.offset -= 1

    @staticmethod
    def is_device_context(var: TorchFunctionModeVariable) -> bool:
        return isinstance(var.value, DeviceContext) or var.value is None

    @classmethod
    def get_mode_index(cls, ind: int) -> int:
        return ind + cls.offset


def _get_all_args(
    args: Iterable[Any], kwargs: dict[str, Any]
) -> Iterable[VariableTracker]:
    return _flatten_vts(pytree.arg_tree_leaves(*args, **kwargs))


def _flatten_vts(vts: Iterable[VariableTracker]) -> list[VariableTracker]:
    from collections import deque

    from .dicts import ConstDictVariable
    from .lists import ListVariable

    vts = deque(vts)
    output = []

    while vts:
        vt = vts.popleft()

        if not vt.is_realized() and vt.peek_type() in (dict, list, tuple):  # type: ignore[attr-defined]
            vt.realize()

        if vt.is_realized():
            if isinstance(vt, ListVariable):
                vts.extend(vt.items)
                continue
            elif isinstance(vt, ConstDictVariable):
                vts.extend(vt.items.values())
                continue

        output.append(vt)

    return output


def _get_subclass_type(var: VariableTracker) -> type:
    if not isinstance(var, (TensorWithTFOverrideVariable, UserDefinedObjectVariable)):
        raise AssertionError(f"Unexpected type {type(var)}")
    return var.python_type()


def _get_subclass_type_var(
    tx: "InstructionTranslatorBase", var: VariableTracker
) -> VariableTracker:
    if isinstance(var, TensorWithTFOverrideVariable):
        return var.class_type_var(tx)
    elif isinstance(var, UserDefinedObjectVariable):
        source = var.source and TypeSource(var.source)
        return VariableTracker.build(tx, var.python_type(), source)
    else:
        raise AssertionError(f"Unexpected type {type(var)}")


def _is_attr_overridden(
    tx: "InstructionTranslatorBase", var: VariableTracker, name: str
) -> bool:
    if not isinstance(var, (TensorWithTFOverrideVariable, UserDefinedObjectVariable)):
        return False
    import torch

    overridden = False
    try:
        attr_val = inspect.getattr_static(var.python_type(), name)
        overridden |= attr_val != getattr(torch.Tensor, name)
    except AttributeError:
        pass

    return overridden


def call_torch_function(
    tx: "InstructionTranslatorBase",
    torch_function_var: VariableTracker,
    fn: VariableTracker,
    types: TupleVariable,
    args: Iterable[Any],
    kwargs: dict[str, Any],
    *,
    is_subclass_dispatch: bool = False,
) -> Any:
    # This emulates calling __torch_function__, which has a signature
    #   def __torch_function__(cls, func, types, args=(), kwargs=None):
    #
    # Also notice the `cls` is not explicitly passed in the reference
    # implementations:
    # 1. https://github.com/pytorch/pytorch/blob/8d81806211bc3c0ee6c2ef235017bacf1d775a85/torch/csrc/utils/python_arg_parser.cpp#L368-L374
    # 2. https://github.com/pytorch/pytorch/blob/8d81806211bc3c0ee6c2ef235017bacf1d775a85/torch/overrides.py#L1741-L1743
    tf_args = [
        fn,
        types,
        VariableTracker.build(tx, tuple(args)),
        VariableTracker.build(tx, kwargs),
    ]
    # Mirror the C++ THPModule_disable_torch_function behavior: disable
    # __torch_function__ subclass dispatch during the call to prevent
    # re-entrant dispatch on operations inside __torch_function__.
    # Only do this for subclass dispatch, not mode dispatch. Modes need
    # subclass dispatch to remain enabled because the mode's
    # __torch_function__ may re-dispatch to the subclass.
    tf_state = tx.symbolic_torch_function_state
    old_subclass_enabled = tf_state.torch_function_subclass_enabled
    if is_subclass_dispatch and old_subclass_enabled:
        tf_state.torch_function_subclass_enabled = False
    try:
        return torch_function_var.call_function(tx, tf_args, {})
    finally:
        tf_state.torch_function_subclass_enabled = old_subclass_enabled


def get_torch_function_fn(
    tx: "InstructionTranslatorBase", vt: VariableTracker
) -> VariableTracker:
    # The underlying function could be a classmethod, staticmethod, regular
    # function or a function with C-implementation. It doesn't matter as long as
    # they satisfy the calling convention in `call_torch_function`.

    args = [vt, VariableTracker.build(tx, "__torch_function__")]
    func_vt = VariableTracker.build(tx, getattr).call_function(tx, args, {})
    return func_vt


def can_dispatch_torch_function(
    tx: "InstructionTranslatorBase", args: Iterable[Any], kwargs: dict[str, Any]
) -> bool:
    has_overridden_args = any(
        has_torch_function(arg) for arg in _get_all_args(args, kwargs)
    )
    tf_state = tx.symbolic_torch_function_state
    return (has_overridden_args and tf_state.torch_function_subclass_enabled) or (
        tf_state.torch_function_mode_enabled and tf_state.in_torch_function_mode()
    )


def dispatch_torch_function(
    tx: "InstructionTranslatorBase",
    fn: VariableTracker,
    args: Iterable[Any],
    kwargs: dict[str, Any],
) -> Any:
    """Gathers all args that are TensorWithTFOverrideVariable and dispatches based on the ordering in _get_overloaded_args"""

    all_args = _get_all_args(args, kwargs)
    overloaded_args = _get_overloaded_args(
        [arg for arg in all_args if has_torch_function(arg)],
        _get_subclass_type,
    )

    types = TupleVariable([_get_subclass_type_var(tx, arg) for arg in overloaded_args])

    if tx.symbolic_torch_function_state.in_torch_function_mode():
        res = tx.symbolic_torch_function_state.call_torch_function_mode(
            tx, fn, types, args, kwargs
        )
        if not res.is_constant_match(NotImplemented):
            return res

    for arg in overloaded_args:
        res = arg.call_torch_function(
            tx,
            fn,
            types,
            args,
            kwargs,
        )

        if not res.is_constant_match(NotImplemented):
            tx.output.torch_function_subclass_inlined = True
            return res

    unimplemented(
        gb_type="All __torch_function__ overrides returned NotImplemented due to TypeError from user code",
        context=f"{fn=}, {args=}, {kwargs=}",
        explanation=f"All __torch_function__ overrides for function {fn} returned NotImplemented",
        hints=[
            *graph_break_hints.USER_ERROR,
        ],
    )


class TensorWithTFOverrideVariable(TensorVariable):
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """

    @classmethod
    def from_tensor_var(
        cls,
        tx: "InstructionTranslatorBase",
        tensor_var: VariableTracker,
        class_type: type,
        cls_source: Source | None,
    ) -> "TensorWithTFOverrideVariable":
        # [Note: __torch_function__] coerce `tensor_var` into a
        # TensorWithTFOverrideVariable. In eager, this is just a type change.
        import torch

        tensor_var = tensor_var.realize()
        # This simulates shallow-copying the tensor object.
        kwargs = dict(tensor_var.__dict__)
        input_tensor_type = kwargs.pop("class_type")
        if not (
            input_tensor_type in (torch.Tensor, torch.nn.Parameter)
            or issubclass(input_tensor_type, torch.Tensor)
        ):
            raise AssertionError(
                f"invalid class type {input_tensor_type} in TensorWithTFOverrideVariable.from_tensor_var"
            )
        var = cls(class_type=class_type, **kwargs)
        var.install_global(tx)
        return var

    def install_global(self, tx: "InstructionTranslatorBase") -> None:
        # stash the subclass type to rewrap an output tensor if needed
        # this is needed because the actual type needs to be available
        # each time the compiled artifact is run and outputs a wrapped tensor.
        if self.global_mangled_class_name(tx) not in tx.output.global_scope:
            # Safe because global_mangled_class_name figures it out
            tx.output.install_global_unsafe(
                self.global_mangled_class_name(tx), self.class_type
            )

    def python_type(self) -> type:
        return self.class_type

    def class_type_var(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return TensorSubclassVariable(
            self.class_type, source=GlobalSource(self.global_mangled_class_name(tx))
        )

    def global_mangled_class_name(self, tx: "InstructionTranslatorBase") -> str:
        return get_safe_global_name(
            tx, f"__subclass_{self.class_type.__name__}", self.class_type
        )

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [Note: __torch_function__] We currently only support attributes that are defined on
        # base tensors, custom attribute accesses will graph break.
        import torch

        # I think only `_base` is breaking because we aren't modelling view
        # relationship perfectly in some scenarios.
        if name in banned_attrs:
            unimplemented(
                gb_type="Unsupported tensor subclass attribute access",
                context=f"{name}",
                explanation="`torch.compile` currently can't trace this",
                hints=[
                    f"Avoid accessing {name} of tensor subclass in torch.compile region",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # Handle non-overridden attributes inherited from `torch.Tensor`.
        attr_is_overridden = _is_attr_overridden(tx, self, name)
        if (
            hasattr(torch.Tensor, name)
            and not attr_is_overridden
            and not inspect.ismethoddescriptor(getattr(torch.Tensor, name))
        ):
            args = [self]
            kwargs: dict[Any, Any] = {}
            if can_dispatch_torch_function(tx, args, kwargs):
                get_fn = VariableTracker.build(tx, getattr(torch.Tensor, name).__get__)

                return self.call_torch_function(
                    tx,
                    get_fn,
                    TupleVariable([self.class_type_var(tx)]),
                    args,
                    kwargs,
                )
        else:
            # `TensorVariable.tp_getattro_impl` doesn't handle user-defined
            # function/attribute well, so we explicitly handle them here.
            #
            # TODO move this logic into `TensorVariable`, or try to merge it
            # with similar logic in `UserDefinedObjectVariable`.
            try:
                attr = inspect.getattr_static(self.class_type, name)
            except AttributeError:
                pass
            else:
                import types

                cls_source = GlobalSource(self.global_mangled_class_name(tx))
                attr_source = AttrSource(cls_source, name)
                if isinstance(attr, types.FunctionType):
                    install_guard(attr_source.make_guard(GuardBuilder.CLOSURE_MATCH))
                    return UserMethodVariable(attr, self)

                elif isinstance(attr, property):
                    getter_source = AttrSource(attr_source, "fget")
                    getter = attr.fget
                    getter_var = VariableTracker.build(
                        tx, getter, source=getter_source, realize=True
                    )
                    return getter_var.call_function(tx, [self], {})

                elif isinstance(attr, classmethod):
                    return UserMethodVariable(
                        attr.__func__, self.class_type_var(tx), source=attr_source
                    )

                elif attr_is_overridden:
                    unimplemented(
                        gb_type="Unsupported tensor subclass overridden attribute access",
                        context=f"{name}",
                        explanation="`torch.compile` only support tracing certain types of overridden tensor subclass attributes",
                        hints=[
                            f"Avoid accessing {name} of tensor subclass in torch.compile region",
                            f"Renaming attribute `{name}` of type {self.class_type}",
                            *graph_break_hints.SUPPORTABLE,
                        ],
                    )

        return super().tp_getattro_impl(tx, name)

    def call_torch_function(
        self,
        tx: "InstructionTranslatorBase",
        fn: VariableTracker,
        types: TupleVariable,
        args: Iterable[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        # NOTE this assumes `__torch_function__` isn't modified during tracing.
        if not hasattr(self, "torch_function_fn"):
            self.torch_function_fn = get_torch_function_fn(tx, self)

        return call_torch_function(
            tx,
            self.torch_function_fn,
            fn,
            types,
            args,
            kwargs,
            is_subclass_dispatch=True,
        )

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # This code block implements inlining the __torch_function__ override
        # of `call_method`.
        tf_args = [self] + list(args)
        if can_dispatch_torch_function(tx, tf_args, kwargs):
            import torch

            if _is_attr_overridden(tx, self, name):
                unimplemented(
                    gb_type="Tensor subclass overridden method call",
                    context=f"{name}",
                    explanation="`torch.compile` currently can't trace this",
                    hints=[
                        f"Avoid calling {name} of tensor subclass in torch.compile region",
                        f"Renaming method `{name}` of type {self.class_type}",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            # [Note: __torch_function__] Currently we only support methods that are defined on tensor
            # we will graph break in other cases this will need a bigger overhaul of extracting methods/comparing them for equality
            # We've established with the above check that the method is not overridden, so we guard that the method is the same
            # as the impl defined on tensor and retrieve it
            if self.source:
                source = AttrSource(AttrSource(self.source, "__class__"), name)
                value = inspect.getattr_static(self.python_type(), name)
            else:
                source = None
                value = getattr(torch.Tensor, name)
            func_var = VariableTracker.build(tx, value, source)
            return dispatch_torch_function(tx, func_var, tf_args, kwargs)
        else:
            return super().call_method(tx, name, args, kwargs)
<|endoftext|>"""
This package implements variable tracking and symbolic execution capabilities for Dynamo,
which are essential for converting Python code into FX graphs. It provides a comprehensive
set of variable types that handle different Python constructs during tracing.

Each variable type (like BuiltinVariable, TensorVariable, NNModuleVariable, etc.) is responsible
for tracking and symbolically executing operations on specific Python objects. This enables
Dynamo to:
- Track the flow of values through Python code
- Maintain correct semantics during graph conversion
- Handle complex Python features like context managers, iterators, and custom objects
- Support both eager and symbolic execution modes

The VariableTracker base class provides the foundation for all variable types, with each
subclass implementing specific behavior for different Python constructs. This modular design
allows Dynamo to accurately trace and optimize Python code while preserving its semantics.
"""

from .base import VariableTracker
from .builtin import (
    BaseBuiltinVariable,
    BuiltinVariable,
    DictBuiltinVariable,
    GetAttrBuiltinVariable,
    HasAttrBuiltinVariable,
    IterBuiltinVariable,
    ListBuiltinVariable,
    SetAttrBuiltinVariable,
)
from .constant import ConstantVariable
from .ctx_manager import (
    AcceleratorDeviceIndexVariable,
    CatchWarningsCtxManagerVariable,
    ContextWrappingVariable,
    CUDADeviceVariable,
    CudagraphOverrideVariable,
    DisabledSavedTensorsHooksVariable,
    DualLevelContextManager,
    DynamoConfigPatchVariable,
    ErrorOnGraphBreakVariable,
    FSDPParamGroupUseTrainingStateVariable,
    FxTracebackAnnotateVariable,
    GenericContextWrappingVariable,
    get_device_context_manager,
    GradIncrementNestingCtxManagerVariable,
    GradInplaceRequiresGradCtxManagerVariable,
    GradModeVariable,
    InferenceModeVariable,
    JvpIncrementNestingCtxManagerVariable,
    register_device_context_manager,
    SDPAKernelVariable,
    SetFwdGradEnabledContextManager,
    TemporarilyPopInterpreterStackCtxManagerVariable,
    VmapIncrementNestingCtxManagerVariable,
    WithEnterFunctionVariable,
    WithExitFunctionVariable,
    XPUDeviceVariable,
)
from .dicts import (
    ConstDictVariable,
    DictItemsVariable,
    DictViewVariable,
    DunderDictVariable,
    MappingProxyVariable,
    NNModuleHooksDictVariable,
    OrderedDictVariable,
)
from .distributed import BackwardHookVariable, DistributedVariable
from .functions import (
    BaseUserFunctionVariable,
    BoundBuiltinMethodVariable,
    ClassMethodDescriptorVariable,
    ClassMethodVariable,
    CollectionsNamedTupleFunction,
    CreateTMADescriptorExperimentalVariable,
    CreateTMADescriptorStableVariable,
    FunctionDecoratedByContextlibContextManagerVariable,
    FunctoolsPartialVariable,
    GetSetDescriptorVariable,
    InspectSignatureVariable,
    LocalGeneratorFunctionVariable,
    LocalGeneratorObjectVariable,
    MemberDescriptorVariable,
    MethodDescriptorVariable,
    MethodWrapperVariable,
    NestedUserFunctionVariable,
    PolyfilledFunctionVariable,
    PropertyVariable,
    PyTreeGetNodeTypeFunctionVariable,
    PyTreeTreeIsLeafFunctionVariable,
    SkipFunctionVariable,
    SparseTensorCreationSkipVariable,
    StaticMethodVariable,
    TMADescriptorExperimentalVariable,
    TMADescriptorStableVariable,
    TritonSetAllocatorVariable,
    TupleGetterVariable,
    UserFunctionVariable,
    UserMethodVariable,
    WrapperDescriptorVariable,
    WrapperUserFunctionVariable,
    WrapperUserMethodVariable,
)
from .higher_order_ops import (
    FunctionalCallVariable,
    FunctorchHigherOrderVariable,
    ReparametrizeModuleCallVariable,
    TorchHigherOrderOperatorVariable,
)
from .iter import (
    ChainVariable,
    CountIteratorVariable,
    FilterVariable,
    IteratorVariable,
    ItertoolsVariable,
    MapVariable,
    RepeatIteratorVariable,
    ZipLongestVariable,
    ZipVariable,
)
from .lazy import LazyConstantVariable, LazyVariableTracker
from .lists import (
    BaseListVariable,
    DequeVariable,
    ListIteratorVariable,
    ListVariable,
    RangeVariable,
    SliceVariable,
    TupleIteratorVariable,
    TupleVariable,
)
from .memory import CUDAMemPoolContextVariable, CUDAMemPoolVariable
from .misc import (
    AttributeErrorVariable,
    AutogradFunctionContextVariable,
    AutogradFunctionVariable,
    CallMethodVariable,
    CellVariable,
    ContextVarVariable,
    DeletedVariable,
    ExceptionVariable,
    GetAttrVariable,
    LambdaVariable,
    NameErrorVariable,
    NewGlobalVariable,
    NumpyVariable,
    ObjectVariable,
    PythonModuleVariable,
    RandomClassVariable,
    RandomVariable,
    StopIterationVariable,
    StringFormatVariable,
    SuperVariable,
    TracebackVariable,
    TypingVariable,
    UnknownVariable,
    WeakRefVariable,
)
from .nn_module import (
    FSDPManagedNNModuleVariable,
    NNModuleVariable,
    UnspecializedBuiltinNNModuleVariable,
    UnspecializedNNModuleVariable,
)
from .optimizer import OptimizerVariable
from .sdpa import SDPAParamsVariable
from .sets import (
    DictKeySetVariable,
    FrozensetVariable,
    OrderedSetClassVariable,
    OrderedSetVariable,
    SetVariable,
)
from .streams import (
    CudaStreamVariable,
    EventVariable,
    StreamContextVariable,
    StreamVariable,
)
from .tensor import (
    DataPtrVariable,
    FakeItemVariable,
    NumpyNdarrayVariable,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
    UntypedStorageVariable,
)
from .torch import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable
from .user_defined import (
    DefaultDictVariable,
    FrozenDataClassVariable,
    InspectVariable,
    MutableMappingVariable,
    NamedTupleVariable,
    RemovableHandleVariable,
    StructSequenceVariable,
    UserDefinedClassVariable,
    UserDefinedConstantVariable,
    UserDefinedDequeVariable,
    UserDefinedDictVariable,
    UserDefinedExceptionClassVariable,
    UserDefinedExceptionObjectVariable,
    UserDefinedListVariable,
    UserDefinedObjectVariable,
    UserDefinedSetVariable,
    UserDefinedTupleVariable,
    UserDefinedVariable,
)


__all__ = [
    "AcceleratorDeviceIndexVariable",
    "AttributeErrorVariable",
    "AutogradFunctionContextVariable",
    "AutogradFunctionVariable",
    "BackwardHookVariable",
    "BaseBuiltinVariable",
    "BaseListVariable",
    "CallMethodVariable",
    "BuiltinVariable",
    "CatchWarningsCtxManagerVariable",
    "CellVariable",
    "ConstantVariable",
    "ConstDictVariable",
    "ContextVarVariable",
    "ContextWrappingVariable",
    "ChainVariable",
    "CountIteratorVariable",
    "CreateTMADescriptorExperimentalVariable",
    "CreateTMADescriptorStableVariable",
    "CUDADeviceVariable",
    "CUDAMemPoolContextVariable",
    "CUDAMemPoolVariable",
    "CudagraphOverrideVariable",
    "DataPtrVariable",
    "DefaultDictVariable",
    "DeletedVariable",
    "DictBuiltinVariable",
    "DictKeySetVariable",
    "DynamoConfigPatchVariable",
    "ErrorOnGraphBreakVariable",
    "FakeItemVariable",
    "get_device_context_manager",
    "GetAttrBuiltinVariable",
    "GetAttrVariable",
    "GradModeVariable",
    "HasAttrBuiltinVariable",
    "InspectSignatureVariable",
    "InspectVariable",
    "IterBuiltinVariable",
    "IteratorVariable",
    "ItertoolsVariable",
    "LambdaVariable",
    "LazyConstantVariable",
    "LazyVariableTracker",
    "ListBuiltinVariable",
    "ListIteratorVariable",
    "ListVariable",
    "MappingProxyVariable",
    "NameErrorVariable",
    "NamedTupleVariable",
    "NestedUserFunctionVariable",
    "NewGlobalVariable",
    "NNModuleVariable",
    "NumpyNdarrayVariable",
    "NumpyVariable",
    "OptimizerVariable",
    "OrderedDictVariable",
    "PolyfilledFunctionVariable",
    "PythonModuleVariable",
    "RangeVariable",
    "register_device_context_manager",
    "RemovableHandleVariable",
    "RepeatIteratorVariable",
    "SDPAParamsVariable",
    "SetAttrBuiltinVariable",
    "SkipFunctionVariable",
    "SliceVariable",
    "StopIterationVariable",
    "StringFormatVariable",
    "StructSequenceVariable",
    "SuperVariable",
    "TemporarilyPopInterpreterStackCtxManagerVariable",
    "TensorVariable",
    "TMADescriptorExperimentalVariable",
    "TMADescriptorStableVariable",
    "TorchCtxManagerClassVariable",
    "TorchInGraphFunctionVariable",
    "TupleVariable",
    "UnknownVariable",
    "UnspecializedNNModuleVariable",
    "UnspecializedPythonVariable",
    "UntypedStorageVariable",
    "UserDefinedClassVariable",
    "UserDefinedObjectVariable",
