    const_mod_name, non_const_mod_name = "submod_0", "submod_1"
    # Safely get submod_1 in case there are no non-const nodes
    const_gm = getattr(split, const_mod_name)
    if not isinstance(const_gm, torch.fx.GraphModule):
        raise AssertionError(
            f"Expected GraphModule for {const_mod_name}, got {type(const_gm)}"
        )
    non_const_mod = getattr(split, non_const_mod_name, None)
    non_const_gm: torch.fx.GraphModule | None = None
    if non_const_mod is not None:
        if not isinstance(non_const_mod, torch.fx.GraphModule):
            raise AssertionError(
                f"Expected GraphModule for {non_const_mod_name}, got {type(non_const_mod)}"
            )
        non_const_gm = non_const_mod

    # The module that a call_module node refers to gets copied to submodules during split.
    # The path to the module also gets inlined, i.e. mod.a.b -> mod_a_b. Here we need to
    # attach inlined modules to `split` as it's the owning module now.
    if non_const_gm is not None:
        for node in non_const_gm.graph.nodes:
            if node.op == "call_module":
                setattr(split, node.target, getattr(non_const_gm, node.target))
    for node in const_gm.graph.nodes:
        if node.op == "call_module":
            setattr(split, node.target, getattr(const_gm, node.target))

    # split_module currently does not use get_attrs for attrs. Instead it passes
    # them in as args from the parent module, which used get_attrs. Here we set
    # them as get_attrs inside const_gm, allowing for running folding without
    # somehow a priori knowing the attrs that should be passed as args. We can
    # unconditionally do this for all placeholders because we know all
    # placeholders to const_gm must be constants accessible via get_attr.
    call_const_gm_args = None
    for node in split.graph.nodes:
        if node.op == "call_module":
            if node.target == const_mod_name:
                call_const_gm_args = node.args
                break
    if call_const_gm_args is None:
        raise AssertionError("Could not find call_module node for const_gm")

    # Here we do the actual replacement of placeholders to get_attrs. Note that here we
    # set the const_gm.graph into a new root_const_gm with split as the root module,
    # because we are fetching attributes directly from the root module, instead of
    # fetching them from const_gm. Example: The const_gm must have some format like:
    # graph():
    #    %inp : [num_users=1] = placeholder[target=const_inp]
    #    %add : [num_users=1] = call_function[target=operator.add](args = (%inp, %inp), kwargs = {})
    #    return add
    # We replace that with the following, which does not have any placeholders:
    # graph():
    #    %inp_1 : [num_users=1] = get_attr[target=const_inp]
    #    %add : [num_users=1] = call_function[target=operator.add](args = (%inp_1, %inp_1), kwargs = {})
    #    return add
    root_const_gm = torch.fx.GraphModule(split, const_gm.graph)

    # The order of placeholders in the const_gm graph should match the order of
    # args in the outer module, so we can simply use an index for the
    # placeholder mapping
    ph_idx = 0
    for node in root_const_gm.graph.nodes:
        if node.op == "output":
            multiple_outputs = isinstance(node.args[0], tuple)
            continue
        if node.op != "placeholder":
            continue
        if ph_idx >= len(call_const_gm_args):
            raise AssertionError(
                f"Placeholder index {ph_idx} out of range for args "
                f"(len={len(call_const_gm_args)})"
            )
        in_node = call_const_gm_args[ph_idx]
        ph_idx += 1
        if in_node.op != "get_attr":
            raise AssertionError(f"Expected get_attr, got {in_node.op}")
        with root_const_gm.graph.inserting_before(node):
            new_node = root_const_gm.graph.get_attr(in_node.target)
        new_node.meta = node.meta.copy()
        node.replace_all_uses_with(new_node)
        root_const_gm.graph.erase_node(node)
    if "multiple_outputs" not in locals():
        raise AssertionError("multiple_outputs not set in loop")

    # Now find the call to const_gm inside split, and replace it with a getattr to the
    # folded tensor(s) that result from constant folding. Note that we don't need to
    # worry about whether this is one or more tensors because the original graph
    # correctly uses getitem to extract individual tensors if there are multiple folded.
    fx_const_folded_attrs_name = get_unique_attr_name_in_module(
        mod_traced, "_FX_CONST_FOLDED_ATTRS"
    )
    setattr(
        split,
        fx_const_folded_attrs_name,
        torch.nn.ParameterList() if multiple_outputs else torch.nn.Parameter(),  # type: ignore[possibly-undefined]
    )
    for node in split.graph.nodes:
        if node.op == "call_module" and node.target == const_mod_name:
            with node.graph.inserting_before(node):
                folded_attrs = node.graph.get_attr(fx_const_folded_attrs_name)
            folded_attrs.meta = node.meta.copy()
            node.replace_all_uses_with(folded_attrs)
            break

    # Finally, inline the non-constant submod (if it exists) into the split submod.
    # This is so that the original caller who may have passed in a graph module will
    # get back out a graph module whose graph is traced to the same granularity.
    if hasattr(split, non_const_mod_name):
        _inline_module(split, non_const_mod_name, is_impure_node=is_impure_node)

    split.graph.eliminate_dead_code(is_impure_node=is_impure_node)

    return FoldedGraphModule(
        split,
        split.graph,
        root_const_gm.graph,
        fx_const_folded_attrs_name,
        device_for_folded_attrs,
    )
<|endoftext|>"""Bind a ``ShapesSpec``/``ParamsSpec`` to a callable's flat ``(args, kwargs)``.

These helpers are pure spec->args binding (``pytree`` + ``inspect`` + the spec
types); they do NOT touch sympy or the ShapeEnv, so they live at the fx layer
and are imported by make_fx, strict export, and non-strict export alike (instead
of make_fx depending on the dynamo export module).
"""

from __future__ import annotations

import inspect
from itertools import zip_longest
from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.dynamic_spec import (
    DictSpec,
    IntermediateSpec,
    IntVar,
    LeafSpec,
    ObjectSpec,
    SeqSpec,
    ShapesSpec,
    TensorSpec,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def _walk_spec(
    user_spec: IntermediateSpec | None,
    arg_value: Any,
    where: str,
) -> list[LeafSpec]:
    """Return the flat leaf-spec list for ``arg_value`` (one entry per
    ``pytree.tree_flatten`` leaf, ``None`` = static), pairing each leaf with its
    ``user_spec`` counterpart.

    Container specs (``SeqSpec`` / ``DictSpec`` / ``ObjectSpec``) recurse and
    concatenate their children's lists, so leaves land in the same order
    ``pytree.tree_flatten`` produces:

    - **list / tuple** (``SeqSpec`` or no-spec subtree): iterate via plain
      ``enumerate(seq)`` -- matches pytree's ``_list_flatten`` /
      ``_tuple_flatten`` (a namedtuple, being a tuple, is handled by position
      here).
    - **dict** (``DictSpec`` or no-spec subtree): iterate ``arg_value.items()``
      (insertion order); pytree's ``_dict_flatten`` does ``list(d.values())``.
    - **pytree-registered objects** (``ObjectSpec``): use the type's registered
      ``flatten_with_keys_fn`` -- the same function pytree dispatches to.

    No flat-index bookkeeping: parents just concatenate child lists. ``where`` is
    a human-readable path string used solely for error messages.
    """
    # No spec for this subtree (top-level None or unspecified slot) => everything
    # below it is static.
    if user_spec is None:
        return [None] * len(pytree.tree_leaves(arg_value))

    if isinstance(user_spec, SeqSpec):
        if not isinstance(arg_value, (list, tuple)):
            raise ValueError(
                f"{where}: SeqSpec expected list/tuple, got {type(arg_value).__name__}"
            )
        entries = user_spec._entries
        if len(entries) > len(arg_value):
            raise ValueError(
                f"{where}: SeqSpec has {len(entries)} entries beyond runtime "
                f"sequence length {len(arg_value)}"
            )
        out: list[LeafSpec] = []
        for i, (value, sub_spec) in enumerate(zip_longest(arg_value, entries)):
            out += _walk_spec(sub_spec, value, where=f"{where}[{i}]")
        return out

    if isinstance(user_spec, DictSpec):
        if not isinstance(arg_value, dict):
            raise ValueError(
                f"{where}: DictSpec expected dict, got {type(arg_value).__name__}"
            )
        unmatched = set(user_spec) - set(arg_value)
        if unmatched:
            raise ValueError(
                f"{where}: DictSpec has entries {sorted(unmatched, key=repr)!r} "
                f"that do not match any key in the runtime dict. "
                f"Runtime keys: {sorted(arg_value.keys(), key=repr)!r}"
            )
        # Walk runtime ordering so positions align with pytree.tree_flatten
        # (insertion order for plain dicts).
        out = []
        for key, value in arg_value.items():
            sub_spec = user_spec._entries[key] if key in user_spec else None
            out += _walk_spec(sub_spec, value, where=f"{where}[{key!r}]")
        return out

    if isinstance(user_spec, ObjectSpec):
        node_type = pytree._get_node_type(arg_value)
        handler = pytree.SUPPORTED_NODES.get(node_type)
        if handler is None:
            raise ValueError(
                f"{where}: ObjectSpec requires the runtime value's type "
                f"to be pytree-registered, but {type(arg_value).__name__} "
                f"is not registered. Register it via "
                f"`torch.export.register_dataclass(<cls>)` (for dataclasses) "
                f"or `pytree.register_pytree_node(...)`."
            )
        if handler.flatten_with_keys_fn is None:
            raise ValueError(
                f"{where}: export requires "
                f"`flatten_with_keys_fn` to be registered for type "
                f"{type(arg_value).__name__}, but none was found. "
                f"Re-register via "
                f"`pytree.register_pytree_node(..., flatten_with_keys_fn=...)` "
                f"or use `torch.export.register_dataclass` for dataclasses."
            )
        key_children, _ = handler.flatten_with_keys_fn(arg_value)
        # Fail-fast on unmatched attrs before recursing. Non-attribute keys
        # (SequenceKey / MappingKey) contribute no matchable names.
        available_names = {
            ke.name for ke, _ in key_children if isinstance(ke, pytree.GetAttrKey)
        }
        unmatched = set(user_spec) - available_names
        if unmatched:
            raise ValueError(
                f"{where}: ObjectSpec has entries {sorted(unmatched)!r} "
                f"that do not match any attribute on the runtime object "
                f"of type {type(arg_value).__name__}. Available "
                f"attributes: {sorted(available_names)!r}"
            )
        out = []
        for key_entry, child in key_children:
            # Only ``GetAttrKey`` entries can match an ObjectSpec entry; any
            # other key shape contributes a static subtree.
            if isinstance(key_entry, pytree.GetAttrKey) and key_entry.name in user_spec:
                out += _walk_spec(
                    user_spec._fields[key_entry.name],
                    child,
                    where=f"{where}.{key_entry.name}",
                )
            else:
                out += [None] * len(pytree.tree_leaves(child))
        return out

    # Leaf spec -- a single flat slot. Type-check against the runtime value.
    if isinstance(user_spec, TensorSpec):
        if not isinstance(arg_value, torch.Tensor):
            raise ValueError(
                f"{where}: spec is TensorSpec but the actual arg is "
                f"{type(arg_value).__name__}, not a Tensor."
            )
    elif isinstance(user_spec, (IntVar, int)):
        if not isinstance(arg_value, int):
            raise ValueError(
                f"{where}: spec is {type(user_spec).__name__} "
                f"(scalar spec) but the actual arg is "
                f"{type(arg_value).__name__}, not int."
            )
    else:
        raise AssertionError(
            f"{where}: unexpected leaf spec type {type(user_spec).__name__}"
        )
    return [user_spec]


def _bind_spec_to_args(
    f: Callable[..., Any],
    args: Any,
    kwargs: dict[str, Any] | None,
    shapes_spec: ShapesSpec,
) -> tuple[list[LeafSpec], list[Any], pytree.TreeSpec]:
    """Bind a user-provided ``ShapesSpec`` to the actual ``(args, kwargs)``
    by inspecting ``f``'s signature, producing:
      - an aligned leaf-spec list ordered to match
        ``pytree.tree_flatten((args, kwargs))``,
      - the flat args list, and
      - the pytree TreeSpec for ``(args, kwargs)``,
    all from a single ``pytree.tree_flatten`` call (callers don't need to
    re-flatten).

    The leaf-spec list has length equal to the flat leaf count, with
    ``None`` at every position that should be treated as static.

    Consumers: the strict export tracer re-wraps this list into a
    ``ShapesSpec``; the non-strict tracer and ``make_fx`` iterate it and
    pair each entry with its flat input.
    """
    params_spec = shapes_spec._params
    kwargs = kwargs or {}
    flat_args, in_spec = pytree.tree_flatten((args, kwargs))
    if params_spec is None:
        # Empty params spec — no per-arg binding to do. Return an
        # all-None list aligned to the flat input layout.
        return [None] * in_spec.num_leaves, flat_args, in_spec

    params_spec_named_args = params_spec._named_args
    params_spec_varargs = params_spec._varargs  # may be None

    sig = (
        inspect.signature(f.forward)
        if isinstance(f, torch.nn.Module)
        else inspect.signature(f)
    )
    pos_params = list(sig.parameters.values())

    # The signature has up to four param regions, bound from the user's
    # call as follows (spec lookup in parens):
    #   1) named-positional: params before `*args`, from args[i] where
    #      i < varargs_idx            (params_spec_named_args[name])
    #   2) varargs (`*args`): from args[i] where i >= varargs_idx
    #                                 (params_spec_varargs[i - varargs_idx])
    #   3) keyword-named (KEYWORD_ONLY, or pos-or-kw passed by name): from
    #      kwargs[name]               (params_spec_named_args[name])
    #   4) var-keyword (`**kwargs`): from kwargs[name]
    #                                 (params_spec_varkw[name])
    # Python guarantees the call layout is [positionals][kwargs], so we
    # walk `args` first (regions 1 + 2), then `kwargs.items()` (regions
    # 3 + 4, same loop).
    #
    # Example: `def forward(self, x, y, *args, **kwargs)` called as
    # `mod(T1, T2, T3, T4, foo=T5, bar=T6)` → varargs_idx=2;
    # args[:2]=(T1,T2) named; args[2:]=(T3,T4) varargs; kwargs={foo,bar}.

    # `varargs_idx` is how many positional params come before `*args` in the
    # signature: the first `varargs_idx` of the caller's positional `args`
    # bind to those named params, and any extras spill into `*args`. It is
    # len(pos_params) when there is no `*args`.
    varargs_idx = len(pos_params)
    for i, p in enumerate(pos_params):
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            varargs_idx = i
            break

    # Walk the user's actual call structure.
    total_leaves = in_spec.num_leaves
    # out_leaf_specs[i] = leaf-spec for flat_args[i] (None = static); keyed
    # under "*args" since ModuleToTrace.forward only has a varargs signature.
    # Built by concatenating each arg's leaf-spec list, in call order (which
    # matches pytree.tree_flatten((args, kwargs))).
    out_leaf_specs: list[LeafSpec] = []

    # Track which named / **kwargs spec entries get bound; leftovers (typo
    # or spec for an omitted defaulted param) are rejected below.
    matched_named_keys: set[str] = set()
    matched_varkw_keys: set[str] = set()

    # Loop 1: named-positional. Spec keyed by signature param name. A key
    # present with value None means "explicitly static" — it skips binding
    # but still counts as matched (not flagged as unmatched below).
    for i, arg_value in enumerate(args[:varargs_idx]):
        arg_name = pos_params[i].name
        if arg_name in params_spec_named_args:
            matched_named_keys.add(arg_name)
            user_spec = params_spec_named_args[arg_name]
        else:
            user_spec = None
        out_leaf_specs += _walk_spec(
            user_spec, arg_value, where=f"shapes_spec[{arg_name!r}]"
        )

    # Pad varargs spec to the actual `*args` count (missing tail = static)
    # so Loop 2 can index uniformly without a bounds check.
    n_actual_varargs = len(args) - varargs_idx
    params_spec_varargs = list(params_spec_varargs or [])
    params_spec_varargs += [None] * max(0, n_actual_varargs - len(params_spec_varargs))

    # Loop 2: varargs. Spec keyed by position within `*args`.
    for user_idx, arg_value in enumerate(args[varargs_idx:]):
        user_spec = params_spec_varargs[user_idx]
        out_leaf_specs += _walk_spec(
            user_spec, arg_value, where=f"shapes_spec['*args'][{user_idx}]"
        )

    # Loop 3: kwargs. Spec comes from named_args[name] (kwarg matches a
    # named param) or varkw[name] (kwarg flows through `**kwargs`).
    params_spec_varkw = params_spec._varkw  # may be None
    for arg_name, arg_value in kwargs.items():
        if arg_name in params_spec_named_args:
            user_spec = params_spec_named_args[arg_name]
            matched_named_keys.add(arg_name)
        elif params_spec_varkw is not None and arg_name in params_spec_varkw:
            user_spec = params_spec_varkw[arg_name]
            matched_varkw_keys.add(arg_name)
        else:
            user_spec = None
        out_leaf_specs += _walk_spec(
            user_spec, arg_value, where=f"shapes_spec[{arg_name!r}]"
        )

    # Every named / **kwargs spec entry must bind to a passed argument; a
    # leftover is almost always a typo or a spec for an omitted default.
    unmatched = set(params_spec_named_args) - matched_named_keys
    if params_spec_varkw is not None:
        unmatched |= set(params_spec_varkw) - matched_varkw_keys
    if unmatched:
        n_named_positional = len(args[:varargs_idx])
        passed = [p.name for p in pos_params[:n_named_positional]] + list(kwargs)
        raise ValueError(
            f"ParamsSpec has entries {sorted(unmatched)!r} that do not match "
            f"any argument passed to export(). Spec keys must be forward "
            f"parameter names that were actually passed. Inputs received: "
            f"{passed!r}."
        )

    # Sanity: per-arg leaf count must equal the whole-tree flatten, else
    # our per-arg walk drifted from the tracer's layout.
    if len(out_leaf_specs) != total_leaves:
        raise AssertionError(
            f"_bind_spec_to_args leaf-count drift: walked {len(out_leaf_specs)} "
            f"leaves but pytree.tree_flatten((args, kwargs)) yields {total_leaves}. "
            f"This means the translator and the export tracer disagree on the "
            f"flat input layout."
        )

    return out_leaf_specs, flat_args, in_spec
<|endoftext|>from typing import Any

import torch
import torch.fx
from torch.fx.experimental.graph_gradual_typechecker import Refine
from torch.fx.experimental.refinement_types import Equality
from torch.fx.experimental.unification import unify, Var  # type: ignore[attr-defined]
from torch.fx.tensor_type import TensorType


__all__ = [
    "check_for_type_equality",
    "convert_eq",
    "infer_symbolic_types",
    "infer_symbolic_types_single_pass",
    "substitute_all_types",
    "substitute_solution_one_type",
    "unify_eq",
]


def infer_symbolic_types_single_pass(traced: torch.fx.GraphModule) -> None:
    """
    Calls our symbolic inferencer once.
    """
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)


def infer_symbolic_types(traced: torch.fx.GraphModule) -> None:
    """
    Calls our symbolic inferencer twice.
    This is useful when one pass is not enough
    to infer all the information such as the case
    for broadcasting.
    """
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)

    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)

    r.symbolic_relations()


def convert_eq(list_of_eq: list[Equality]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """
    Convert equality constraints in the right format
    to be used by unification library.
    """
    lhs = []
    rhs = []
    for eq in list_of_eq:
        lhs.append(eq.lhs)
        rhs.append(eq.rhs)
    return tuple(lhs), tuple(rhs)


def unify_eq(list_of_eq: list[Equality]) -> Any:
    """
    Apply unification to a set of
    equality constraints
    """
    lhs, rhs = convert_eq(list_of_eq)
    return unify(lhs, rhs)


def substitute_solution_one_type(mapping: dict[object, object], t: object) -> Any:
    """
    Apply the most general unifier to a type
    """
    if isinstance(t, Var):
        if t in mapping:
            return mapping[t]
        else:
            return t

    elif isinstance(t, TensorType):
        new_type = []
        for typ in t.dims:
            if typ in mapping:
                new_type.append(mapping[typ])
            else:
                new_type.append(typ)
        return TensorType(tuple(new_type))

    elif isinstance(t, list):
        new_type = []
        for typ in t:
            new_type.append(substitute_solution_one_type(mapping, typ))
        return new_type

    elif isinstance(t, tuple):
        new_type = []
        for typ in t:
            new_type.append(substitute_solution_one_type(mapping, typ))
        return tuple(new_type)

    else:
        return t


def substitute_all_types(graph: torch.fx.Graph, mapping: dict[object, object]) -> None:
    """
    Apply the most general unifier to all types in a graph
    till reaching a fixed point. If the input and output graph
    are the same, we converge.
    """
    flag = True
    while flag:
        flag = False
        for k in mapping:
            old_mapping_val = mapping[k]
            if mapping[k] in mapping:
                new_key = mapping[k]
                mapping[k] = mapping[new_key]
            if old_mapping_val != mapping[k]:
                flag = True

    for n in graph.nodes:
        n.type = substitute_solution_one_type(mapping, n.type)


def check_for_type_equality(g1: torch.fx.Graph, g2: torch.fx.Graph) -> bool:
    """
    A check equality to be used in fixed points.
    We do not use graph equality but instead type
    equality.
    """
    for n, m in zip(g1.nodes, g2.nodes):
        if n.type != m.type:
            return False
    return True
<|endoftext|>import enum
import os
import sys

from torch.utils._config_module import Config, install_config_module


# [@compile_ignored: debug] Fails hard instead of graph breaking on guard on data dependent errors.
no_data_dependent_graph_break = (
    os.environ.get("TORCHDYNAMO_NO_DATA_DEPENDENT_GRAPH_BREAK", "0") == "1"
)
# [@compile_ignored: debug] Uses z3 for validating the guard optimizations transformations.
translation_validation = (
    os.environ.get("TORCHDYNAMO_TRANSLATION_VALIDATION", "0") == "1"
)
# Timeout (in milliseconds) for z3 finding a solution.
# [@compile_ignored: debug]
translation_validation_timeout = int(
    os.environ.get("TORCHDYNAMO_TRANSLATION_VALIDATION_TIMEOUT", "600000")
)
# Disables bisection for translation validation.
#
# Translation validation bisection is enabled by default, if translation validation
# is also enabled. This should help finding guard simplification issues. However,
# since validation uses Z3 for bisecting, it might take a lot of time.
#
# Set this configuration option so as to avoid bisecting.
# [@compile_ignored: debug]
translation_validation_no_bisect = (
    os.environ.get("TORCHDYNAMO_TRANSLATION_NO_BISECT", "0") == "1"
)
# Checks whether replaying ShapeEnv events on a freshly constructed one yields
# a ShapeEnv with the same state. This should be used only in testing.
check_shape_env_recorded_events = False

# TODO: Perhaps consider allowing unions for the configs below (so you can hit
# multiple reps at the same time)

# Give extended debug information if the string representation of a guard
# matches this.  For example, set this to "Ne(s0, 10)" and whenever we issue
# this guard, we will generate full Python and C++ backtrace
# [@compile_ignored: debug]
extended_debug_guard_added = os.environ.get(
    "TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED", None
)

# Give extended debug information when a particular symbol is allocated.  For
# example, set this to "u2" and whenever we create this symbol, we will
# generate full Python and C++ backtrace
# [@compile_ignored: debug]
extended_debug_create_symbol = os.environ.get(
    "TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL", None
)

# Give extended debug information (C++ backtrace) for all extended debug
# settings as well as errors.  The C++ backtrace is slow and very spammy so we
# don't include it by default even when you're requesting extended debug.
# [@compile_ignored: debug]
extended_debug_cpp = os.environ.get("TORCHDYNAMO_EXTENDED_DEBUG_CPP", "") != ""

# Give extended debug information (line of code) when a torch function
# is called during export.  This is useful for showing progress and detecting
# where export might be stuck. Currently only works for strict=False.
# [@compile_ignored: debug]
extended_debug_current_loc = (
    os.environ.get("TORCHEXPORT_EXTENDED_DEBUG_CURRENT_LOC", "0") == "1"
)

# [@compile_ignored: debug] Show a warning for every specialization
print_specializations = False

# wraps (un)equalities with 'Not' class after recording the correct expression
# in the FX graph. This should incorrectly construct the divisible and replacement
# lists, and incorrectly issue guards.
inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY = False

# [@compile_ignored: debug] Validate that ShapeEnv's version key is updated correctly
validate_shape_env_version_key = False

# If we produce more than this many guards on a symbol, force the symbol to
# get specialized and bail out if this many guards mention this particular
# symbol.  This may be slightly more aggressive than the true number of guards
# issued (as we test if we've hit the limit on-the-fly, whereas we may
# do further simplifications at final guard issuance time that make guards
# irrelevant.)
symbol_guard_limit_before_specialize: int | None = None

# This flag changes whether we should use the same symbolic variable to represent input sizes that are the same.
use_duck_shape = True

# Controls the registration of torch.nonzero() on the meta device.
# When True, nonzero returns a tensor with shape (self.numel(), self.dim())
# assuming all elements are none-zero.
# Default is False to prevent unintended registration. Set to True to enable.
meta_nonzero_assume_all_nonzero = False

# Applies size-oblivious reasoning to backed symbols. This allocates a [0, inf] range for backed size symbols,
# and relies on size-oblivious semantics to avoid 0/1 specialization guards by marking them size-like.
# Currently an experimental option for export.
backed_size_oblivious = False

# When True, the size-oblivious fallback in `_view_unbacked_meta` (used when
# `backed_size_oblivious=True`) will, before falling through to the specializing
# recursion that emits Eq(s, 1) via eval_eager, try to discover and commit
# cross-symbol equalities (Eq(x, y)) that satisfy the view's numel constraint.
# Only applies when `backed_size_oblivious=True`. Defaults to False.
unify_view_symbols_bso_meta = False

# Skip dtype check in meta registrations. Only used for systems that does its own dtype checking.
skip_dtype_check_in_meta_registrations = False

# Experimental: If True, graph module will register fx metadata during recompile()
enrich_profiler_metadata: bool = Config(  # type: ignore[var-annotated]
    default=False,
    env_name_default="TORCH_ENRICH_RPOFILER_STACK_TRACE",
)

# When True, log a warning instead of raising PendingUnbackedSymbolNotFound exception
# when pending unbacked symbols are not found in returned outputs.
# The worst that can happen is an error somewhere else in the stack where we expect
# to locate an unbacked binding. Or a runtime assertion not being lowered in the output
# code.
soft_pending_unbacked_not_found_error = False

# When True, aggressively return fallback values in guard_or opting into
# guard-free semantics. This optimizes tracing time when symbolic reasoning
# is expensive. Since guard_or_X already have a general path to take, we
# can skip expensive static evaluation and just return the fallback value directly.
# This is usually safe because the fallback represents a valid code path that
# could be taken anyway.
# See AggressiveGuardFreeMode below for valid values.
aggressive_guard_free_semantics = 0


install_config_module(sys.modules[__name__])


class AggressiveGuardFreeMode(enum.IntEnum):
    DISABLED = 0
    VALUE_RANGE_ANALYSIS = 1  # use bound_sympy before returning fallback
    SKIP_RANGE_ANALYSIS = 2  # skip range analysis entirely, just return fallback_value
<|endoftext|>from typing import *  # noqa: F403


# Python version of c10/core/ConstantSymNodeImpl.cpp
# This needs to exist because the Python version of nested int is not compatible
# with the C++ version of constant symnode.
class ConstantIntNode:
    def __init__(self, val: int) -> None:
        self.val = val

    def is_constant(self) -> bool:
        return True

    def maybe_as_int(self) -> int:
        return self.val

    def is_int(self) -> bool:
        return True

    def is_float(self) -> bool:
        return False

    def is_bool(self) -> bool:
        return False

    def is_nested_int(self) -> bool:
        return False

    def clone(self) -> "ConstantIntNode":
        return self

    def _str(self) -> str:
        return str(self.val)

    def __str__(self) -> str:
        return self._str()

    def __repr__(self) -> str:
        return self._str()

    def _graph_repr(self) -> str:
        return self._str()

    def add(self, other: Any) -> Any:
        return other.add(self)

    def sub(self, other: Any) -> Any:
        return other.neg().add(self.val)

    def mul(self, other: Any) -> Any:
        return other.mul(self)

    def eq(self, other: Any) -> Any:
        return other.eq(self)

    def ne(self, other: Any) -> Any:
        return other.ne(self)

    def gt(self, other: Any) -> Any:
        return other.lt(self)

    def lt(self, other: Any) -> Any:
        return other.gt(self)

    def le(self, other: Any) -> Any:
        return other.ge(self)

    def ge(self, other: Any) -> Any:
        return other.le(self)

    def is_symbolic(self) -> bool:
        return False

    def constant_int(self) -> int:
        return self.val

    def guard_int(self, file: str, line: int) -> int:
        return self.val
<|endoftext|>import operator
from collections.abc import Callable
from typing import Any

import torch
import torch.fx
import torch.fx as fx
from torch.fx import Proxy, Transformer
from torch.fx.node import Argument, map_aggregate, Node, Target
from torch.fx.operator_schemas import (
    create_type_hint,
    normalize_function,
    normalize_module,
)

from .schema_type_annotation import AnnotateTypesWithSchema


class NormalizeArgs(Transformer):
    """
    Normalize arguments to Python targets. This means that
    `args/kwargs` will be matched up to the module/functional's
    signature and rewritten to exclusively kwargs in positional order
    if `normalize_to_only_use_kwargs` is true. Also populates default
    values. Does not support positional-only parameters or varargs
    parameters (*args, **kwargs).

    If the nodes have 'type' metadata, it will use it to disambiguate
    overloads. Otherwise, it will throw an error.

    Example usage:
        m = torchvision.models.resnet18()
        traced = torch.fx.symbolic_trace(m)
        traced = NormalizeArgs(traced).transform()
    """

    def __init__(
        self, module: torch.fx.GraphModule, normalize_to_only_use_kwargs: bool = True
    ) -> None:
        super().__init__(module)
        self.node_map: dict[Proxy, Node] = {}
        self.normalize_to_only_use_kwargs = normalize_to_only_use_kwargs

    def run_node(self, n: Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        def get_type(arg: object) -> Any:
            if isinstance(arg, fx.Node):
                return n.meta.get("type")
            return type(arg)

        arg_types = map_aggregate(n.args, get_type)
        if not isinstance(arg_types, tuple):
            raise AssertionError(f"Expected tuple, got {type(arg_types)}")
        arg_types = tuple(create_type_hint(i) for i in arg_types)
        kwarg_types = {k: get_type(v) for k, v in kwargs.items()}
        if n.op == "call_function":
            out = self.call_function(n.target, args, kwargs, arg_types, kwarg_types)
        else:
            out = super().run_node(n)
        if n.op != "output":
            self.node_map[out] = n
            out.node.meta = n.meta
            out.node.type = n.type
        return out

    def call_function(
        self,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Any],
        arg_types: tuple[Any, ...] | None = None,
        kwarg_types: dict[str, Any] | None = None,
    ) -> Proxy:
        if not callable(target):
            raise AssertionError(f"Expected callable target, got {type(target)}")
        new_args_and_kwargs = normalize_function(
            target,
            args,  # type: ignore[arg-type]
            kwargs,
            arg_types,  # type: ignore[arg-type]
            kwarg_types,
            self.normalize_to_only_use_kwargs,
        )
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            return self.tracer.create_proxy(
                "call_function", target, new_args, new_kwargs
            )
        else:
            return super().call_function(target, args, kwargs)

    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        if not isinstance(target, str):
            raise AssertionError(f"Expected str target, got {type(target)}")
        new_args_and_kwargs = normalize_module(
            self.module,
            target,
            args,  # type: ignore[arg-type]
            kwargs,
            self.normalize_to_only_use_kwargs,
        )
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            return super().call_module(target, new_args, new_kwargs)
        else:
            return super().call_module(target, args, kwargs)


class NormalizeOperators(AnnotateTypesWithSchema):
    """
    Normalize callsites that are different ways of "spelling" the same
    invocation into a single, canonical call. Currently supports:

    1. Normalize operators (e.g. operator.add) to the `torch` ops they
       ultimately invoke (e.g. torch.add) when it is possible to statically
       reason that

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = NormalizeOperators(traced).transform()
    """

    binary_magic_method_remap: dict[
        Callable[[Any, Any], Any], Callable[[Any, Any], Any]
    ] = {
        torch.add: operator.add,
        torch.mul: operator.mul,
        torch.sub: operator.sub,
        torch.div: operator.truediv,
        torch.floor_divide: operator.floordiv,
        torch.remainder: operator.mod,
        torch.eq: operator.eq,
        torch.ne: operator.ne,
        torch.lt: operator.lt,
        torch.le: operator.le,
        torch.gt: operator.gt,
        torch.ge: operator.ge,
    }

    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        # Normalize operators according to the magic methods implemented on tensors here:
        # https://github.com/pytorch/pytorch/blob/28c5d90b679c6b38bf4183ec99f16d933c2f1bcd/tools/autograd/templates/python_variable_methods.cpp#L1137

        if not callable(target):
            raise AssertionError(f"Expected callable target, got {type(target)}")

        if target in self.binary_magic_method_remap:
            if len(args) != 2:
                return super().call_function(target, args, kwargs)
            lhs, rhs = args

            return super().call_function(
                target=self.binary_magic_method_remap[target],
                args=(lhs, rhs),
                kwargs={},
            )

        return super().call_function(target, args, kwargs)
<|endoftext|>class Equality:
    def __init__(self, lhs: object, rhs: object) -> None:
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.lhs} = {self.rhs}"

    def __repr__(self) -> str:
        return f"{self.lhs} = {self.rhs}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Equality):
            return self.lhs == other.lhs and self.rhs == other.rhs
        else:
            return False
<|endoftext|>import itertools
import operator
from collections.abc import Callable
from functools import reduce
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import sympy

import torch
from torch.fx.experimental.refinement_types import Equality
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
from torch.fx.node import Node, Target
from torch.fx.tensor_type import Dyn, is_consistent, is_more_precise, TensorType
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d


_T = TypeVar("_T")
_P = ParamSpec("_P")

_INFERENCE_RULES: dict[Target, Callable[..., Any]] = {}
_REFINEMENT_RULES: dict[Target, Callable[..., Any]] = {}
_RULES: dict[Target, Callable[..., Any]] = {}

__all__ = [
    "GraphTypeChecker",
    "Refine",
    "adaptiveavgpool2d_check",
    "adaptiveavgpool2d_inference_rule",
    "add_inference_rule",
    "all_eq",
    "bn2d_inference_rule",
    "broadcast_types",
    "calculate_out_dimension",
    "conv2d_inference_rule",
    "conv_refinement_rule",
    "conv_rule",
    "element_wise_eq",
    "expand_to_tensor_dim",
    "first_two_eq",
    "flatten_check",
    "flatten_inference_rule",
    "flatten_refinement_rule",
    "get_attr_inference_rule",
    "get_greatest_upper_bound",
    "get_parameter",
    "linear_check",
    "linear_inference_rule",
    "linear_refinement_rule",
    "maxpool2d_check",
    "maxpool2d_inference_rule",
    "register_algebraic_expressions_inference_rule",
    "register_inference_rule",
    "register_refinement_rule",
    "relu_inference_rule",
    "reshape_inference_rule",
    "transpose_inference_rule",
]


# TODO: narrow t to TensorType | _DynType once Node.type is narrowed
def expand_to_tensor_dim(t: Any, n: int) -> TensorType:
    """
    Expand a type to the desired tensor dimension if possible
    Raise an error otherwise.
    - t is the given type
    - n is a number of dimensions to expand to
    """
    if t == Dyn:
        dims = [Dyn] * n
        return TensorType(tuple(dims))
    elif isinstance(t, TensorType):
        if len(t.dims) != n:
            raise TypeError(
                f"Cannot extend tensor. Tensor {t} has rank {len(t.dims)}. It should have rank {n}"
            )
        return t
    else:
        raise TypeError(f"Cannot match the type {t}")


def broadcast_types(t1: Any, t2: Any) -> tuple[Any, Any]:
    """
    Applies broadcasting to both given types such that they
    become consistent with each other and returns two new
    resulting types
    """

    # if either type is Dyn, do nothing since the types are already consistent
    if t1 == Dyn or t2 == Dyn or isinstance(t1, Var) or isinstance(t2, Var):
        return t1, t2

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        s1 = len(t1.dims)
        s2 = len(t2.dims)

        new_t1 = list(t1.dims)
        new_t2 = list(t2.dims)

        # We make the types the same length which is the first requirement
        # for consistency
        if s1 > s2:
            for _ in range(s1 - s2):
                new_t2.insert(0, 1)

        elif s2 > s1:
            for _ in range(s2 - s1):
                new_t1.insert(0, 1)

        # we replace occurrences of "1" with each tensor with
        # the corresponding type from the other tensor
        for i, (x, y) in enumerate(zip(new_t1, new_t2)):
            if x == 1:
                new_t1[i] = y
            elif y == 1:
                new_t2[i] = x

        # at this point our tensors should be consistent
        # and we can apply the element-wise operation and find the right dimension
        # for the output of the operation
        (t1, t2) = TensorType(tuple(new_t1)), TensorType(tuple(new_t2))
        return (t1, t2)
    else:
        raise TypeError(f"Cannot broadcast types {t1} and {t2}")


def register_inference_rule(
    call_target: Target,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def register(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        if call_target in _INFERENCE_RULES:
            raise RuntimeError(f"Inference rule already registered for {call_target}!")
        _INFERENCE_RULES[call_target] = fn
        return fn

    return register


def register_refinement_rule(
    call_target: Target,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def register(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        if call_target in _REFINEMENT_RULES:
            raise RuntimeError(f"Refinement rule already registered for {call_target}!")
        _REFINEMENT_RULES[call_target] = fn
        return fn

    return register


def register_algebraic_expressions_inference_rule(
    call_target: Target,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def register(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        if call_target in _RULES:
            raise RuntimeError(f"Rule already registered for {call_target}!")
        _RULES[call_target] = fn
        return fn

    return register


@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def add_inference_rule(n: Node) -> Any:
    """
    Apply the addition inference rule. This includes:
    - scalar addition
    - broadcasting semantics

    Note that we always return the least precise type between
    the operands (after applying broadcasting) to be the final type of the operation

    Note that we do not modify the operand types themselves after applying broadcasting
    to them. We only use them to calculate the final type
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[1])}")
    t1 = n.args[0].type
    t2 = n.args[1].type

    # handle scalar addition
    if t1 is int and isinstance(t2, TensorType):
        n.type = t2
        return n.type

    # handle scalar addition
    elif t2 is int and isinstance(t1, TensorType):
        n.type = t1
        return n.type

    # we bring the new types to the point where
    # we can check for consistency
    # any inconsistency would not have been caused
    # by broadcasting at this point
    (new_t1, new_t2) = broadcast_types(t1, t2)

    if new_t1 != t1 or new_t2 != t2:
        n.meta["broadcast"] = True
        n.meta[str(n.args[0])] = new_t1
        n.meta[str(n.args[1])] = new_t2

    else:
        n.meta["broadcast"] = False

    new_t1 = t1 if not n.meta["broadcast"] else new_t1
    new_t2 = t2 if not n.meta["broadcast"] else new_t2

    # we check for consistency between the new types
    if is_consistent(new_t1, new_t2):
        # we return the less precise type because
        # broadcasting may have happened
        # for operands with shape [1,2,Dyn] and [1,2,1]
        # we have to assign the node [1,2,Dyn]
        if is_more_precise(new_t1, new_t2):
            n.type = new_t2
        else:
            n.type = new_t1
        return n.type
    else:
        raise TypeError(
            f"Cannot add arguments {n.args[0]} ({n.args[0].type}) and {n.args[1]} ({n.args[1].type}) in node {n}."
            f" Types should match "
        )


@register_inference_rule(getattr)
def get_attr_inference_rule(n: Node, traced: Any) -> Any:
    """
    The current getattr rule only handles the shape attribute
    Can be extended to other attributes
    The most representative type we have is "Dyn" but the system
    can be extended with more types, such as a type to represent shapes
    """
    attr_name = n.args[1]

    if attr_name == "shape":
        n.type = Dyn
    else:
        raise TypeError("Not yet implemented")

    # TODO. We leave it like this till we add a type to represent tensor sizes
    return n.type


@register_inference_rule(torch.transpose)
def transpose_inference_rule(n: Node) -> Any:
    """
    We check that dimensions for the transpose operations
    are within range of the tensor type of the node
    """
    if n.target is torch.transpose:
        if not isinstance(n.args[0], Node):
            raise AssertionError(f"Expected Node, got {type(n.args[0])}")
        t = n.args[0].type

        if not isinstance(n.args[1], int):
            raise AssertionError(f"Expected int, got {type(n.args[1])}")
        if not isinstance(n.args[2], int):
            raise AssertionError(f"Expected int, got {type(n.args[2])}")
        dim1, dim2 = n.args[1], n.args[2]

        if t == Dyn:
            n.type = Dyn
            return n.type

        elif isinstance(t, TensorType):
            if 0 <= dim1 < len(t.dims) and 0 <= dim2 < len(t.dims):
                new_type = list(t.dims)
                new_type[dim1], new_type[dim2] = new_type[dim2], new_type[dim1]
                final = TensorType(new_type)
                n.type = get_greatest_upper_bound(n.type, final)
                return n.type
            else:
                raise TypeError(
                    f"Cannot transpose {dim1} and {dim2} in type {t} for node {n}"
                )
        else:
            raise TypeError(
                f"Cannot transpose {dim1} and {dim2} in type {t} for node {n}"
            )


@register_inference_rule(torch.reshape)
def reshape_inference_rule(n: Node) -> TensorType:
    """
    Without dynamism, the rule checks that the
    product of the elements of the argument tensor
    type is equal to the product of the elements
    of the required shape. We gradualize this rule
    by adding a case to handle fully dynamic input
    as well as input where some of the tensor dimensions
    are unknown. In this case we check for divisibility
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    t1 = n.args[0].type

    if not isinstance(n.args[1], list):
        raise AssertionError(f"Expected list, got {type(n.args[1])}")
    t2 = n.args[1]
    t2_type = TensorType([Dyn if elem == -1 else elem for elem in t2])

    # if we do not know the original tensor dimension,
    # we return the required dimension
    if t1 == Dyn:
        n.type = t2_type
        return t2_type

    # if any of the dimensions are unknown,
    # we check for divisibility
    elif isinstance(t1, TensorType):
        if not isinstance(t1, TensorType):
            raise AssertionError(f"Expected TensorType, got {type(t1)}")
        a = [e if e != Dyn else 1 for e in t1.dims]
        p1 = reduce(operator.mul, a)
        p2 = reduce(operator.mul, t2)
        if p1 % p2 == 0 or p2 % p1 == 0:
            n.type = t2_type
            return t2_type
        else:
            raise TypeError(f"Cannot reshape in node {n} from {t1} to {t2_type}")
    else:
        raise TypeError(f"Cannot reshape in node {n} from {t1} to {t2_type}")


@register_inference_rule(BatchNorm2d)
def bn2d_inference_rule(n: Node, module_instance: Any) -> Any:
    """
    Given a BatchNorm2D instance and a node check the following conditions:
    - the input type can be expanded to a size 4 tensor: t =  (x_1, x_2, x_3, x_4)
    - the current node type can be expanded to a size 4 tensor: t' =  (x_1', x_2', x_3', x_4')
    - t is consistent with t'
    - x_2 is consistent with the module's num_features
    - x_2' is consistent with the module's num_features
    output type: the more precise type of t and t'
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    n.args[0].type = expand_to_tensor_dim(n.args[0].type, 4)
    arg_type = n.args[0].type
    n.type = expand_to_tensor_dim(n.type, 4)

    # we check the conditions on the incoming argument
    # and any existing annotation
    # we also check for consistency between both annotations
    if (
        is_consistent(arg_type.dims[1], module_instance.num_features)
        and is_consistent(n.type.dims[1], module_instance.num_features)
        and is_consistent(arg_type, n.type)
    ):
        # we choose the more precise type
        # to be the node type
        # so if an incoming argument has more type information
        # we set this node's type to be the argument type
        n.type = get_greatest_upper_bound(arg_type, n.type)
        return n.type
    else:
        raise TypeError(
            f"Cannot apply {module_instance} with input type {arg_type} and existing type {n.type} on {n}"
        )


def calculate_out_dimension(d_in: Any, module_instance: Any, index: int) -> Any:
    """
    For calculating h_in and w_out according to the conv2D documentation
    """
    padding = (
        (module_instance.padding, module_instance.padding)
        if isinstance(module_instance.padding, int)
        else module_instance.padding
    )
    kernel_size = (
        (module_instance.kernel_size, module_instance.kernel_size)
        if isinstance(module_instance.kernel_size, int)
        else module_instance.kernel_size
    )
    stride = (
        (module_instance.stride, module_instance.stride)
        if isinstance(module_instance.stride, int)
        else module_instance.stride
    )
    dilation = (
        (module_instance.dilation, module_instance.dilation)
        if isinstance(module_instance.dilation, int)
        else module_instance.dilation
    )

    DIMENSION_TYPES = (int, sympy.Symbol)

    if d_in == Dyn:
        return Dyn

    elif isinstance(d_in, DIMENSION_TYPES):
        n = d_in + 2 * padding[index] - dilation[index] * (kernel_size[index] - 1) - 1

        return (n // stride[0]) + 1

    else:
        raise TypeError(
            f"{d_in} in {module_instance} must be a number or Dyn. Received {type(d_in)}"
        )


# TODO: narrow params/return to TensorType | _DynType once Node.type is narrowed
def get_greatest_upper_bound(type1: Any, type2: Any) -> Any:
    """
    Get the most precise type that's consistent with the given types
    """
    if type1 == Dyn:
        return type2
    elif type2 == Dyn:
        return type1
    elif isinstance(type1, TensorType) and isinstance(type2, TensorType):
        if not is_consistent(type1, type2):
            raise TypeError(f"Inconsistent types {type1}, {type2}")
        gub = [
            t1 if is_more_precise(t1, t2) else t2
            for (t1, t2) in zip(type1.dims, type2.dims)
        ]
        return TensorType(tuple(gub))


@register_inference_rule(Conv2d)
def conv2d_inference_rule(n: Node, module_instance: Any) -> Any:
    """
    Given a Conv2D instance and a node check the following conditions:
    - the input type can be expanded to a size 4 tensor: t =  (x_1, x_2, H, W)
    - the current node type can be expanded to a size 4 tensor: t' =  (x_1', x_2', x_3', x_4')
    - x_2 is consistent with the module's in_channels
    - let o = (x_1, out_channels, H_out, W_out)
    then the output is the greatest upper bound of o and the existing node type t'.
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    n.args[0].type = expand_to_tensor_dim(n.args[0].type, 4)
    arg_type = n.args[0].type
    curr_node_type = expand_to_tensor_dim(n.type, 4)

    if is_consistent(arg_type.dims[1], module_instance.in_channels):
        w_in = arg_type.dims[3]
        h_in = arg_type.dims[2]
        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)
        new_type = TensorType(
            (arg_type.dims[0], module_instance.out_channels, h_out, w_out)
        )
        gub = get_greatest_upper_bound(new_type, curr_node_type)
        n.type = gub
        return n.type
    else:
        raise TypeError(
            f"Cannot apply {module_instance} with input type {arg_type} and existing type {n.type} on {n}"
        )


@register_inference_rule(torch.nn.ReLU)
def relu_inference_rule(n: Node, module_instance: Any) -> Any:
    """
    Input and output shapes should be equal.
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.dims))

    if isinstance(n.args[0].type, TensorType):
        n.type = get_greatest_upper_bound(n.args[0].type, n.type)
    return n.type


def maxpool2d_check(typ: Any, module_instance: Any) -> TensorType:
    """
    Applies the maxpool2d shape information to the input
    this affects the last two dimensions
    """
    new_type_list = list(typ.dims)
    if len(new_type_list) == 4 or len(new_type_list) == 3:
        w_in = new_type_list[-1]
        h_in = new_type_list[-2]

        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)

        new_type_list[-1] = w_out
        new_type_list[-2] = h_out
        return TensorType(tuple(new_type_list))

    else:
        raise TypeError(f"Wrong size {typ} for {module_instance}")


@register_inference_rule(torch.nn.MaxPool2d)
def maxpool2d_inference_rule(n: Node, module_instance: Any) -> Any:
    """
    Given a MaxPool2D instance and a node check the following conditions:

    - Input size matches size 3 or 4
    - Current node type is consistent with the output type we will calculate
    - Input size matches output size and the last two dimensions of the output
      are w_out and h_out. The remaining dimensions are the same as the input
    - Our final result is the greatest upper bound of the output we calculate
      and the current node type.
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.dims))
    if isinstance(n.args[0].type, TensorType):
        output = maxpool2d_check(n.args[0].type, module_instance)
        n.type = get_greatest_upper_bound(output, n.type)
    return n.type


def linear_check(tensor_type: Any, module_instance: Any) -> TensorType:
    """
    Checks that an input tensor type satisfies the conditions for linear operation
    and returns the output type based on in and out features given by module_instance
    """
    if len(tensor_type.dims) >= 2:
        if is_consistent(module_instance.in_features, tensor_type.dims[-1]):
            new_type_args = list(tensor_type.dims)
            new_type_args[-1] = module_instance.out_features
            return TensorType(tuple(new_type_args))
        else:
            raise TypeError(
                f"Inconsistent {module_instance.in_features} and {tensor_type.dims[-1]} in {module_instance}"
            )
    else:
        raise TypeError(f"Type {tensor_type} must have rank 2 or more.")


@register_inference_rule(torch.nn.Linear)
def linear_inference_rule(n: Node, module_instance: Any) -> Any:
    """
    Applies the shape information to the input then gets the greatest upper bound
    of the resulting type and the existing type
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.dims))
    if isinstance(n.args[0].type, TensorType):
        output_type = linear_check(n.args[0].type, module_instance)
        n.type = get_greatest_upper_bound(output_type, n.type)
    return n.type


def adaptiveavgpool2d_check(tensor_type: Any, module_instance: Any) -> TensorType:
    output_size = module_instance.output_size
    if isinstance(output_size, int):
        output_size = [output_size, output_size]
    elif isinstance(output_size, tuple):
        output_size = list(output_size)
        if output_size[0] is None:
            output_size[0] = output_size[1]
        if output_size[1] is None:
            output_size[1] = output_size[0]

    new_type_list = list(tensor_type.dims)

    if len(tensor_type.dims) == 4 or len(tensor_type.dims) == 3:
        new_type_list[-1] = output_size[1]
        new_type_list[-2] = output_size[0]

        return TensorType(tuple(new_type_list))

    else:
        raise TypeError(f"Tensor ranks must be 3 or 4. Got {tensor_type}")


@register_inference_rule(torch.nn.AdaptiveAvgPool2d)
def adaptiveavgpool2d_inference_rule(n: Node, module_instance: Any) -> Any:
    """
    The input and output sizes should be the same except for the last
    two dimensions taken from the input, which represent width and height
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.dims))
    if isinstance(n.args[0].type, TensorType):
        output_type = adaptiveavgpool2d_check(n.args[0].type, module_instance)
        n.type = get_greatest_upper_bound(n.type, output_type)
    return n.type


def flatten_check(tensor_type: Any, start_dim: int, end_dim: int) -> TensorType:
    l = len(tensor_type.dims)

    start_dim = l if start_dim == -1 else abs(start_dim)
    end_dim = l + end_dim + 1 if end_dim < 0 else end_dim + 1

    if 0 <= start_dim <= (l - 1) and 0 <= end_dim <= l and start_dim < end_dim:
        my_args = list(tensor_type.dims)
        lhs = my_args[0:start_dim]
        rhs = my_args[end_dim:]
        mid = my_args[start_dim:end_dim]
        if Dyn in mid:
            mid = [Dyn]
        else:
            mid = [reduce(operator.mul, my_args[start_dim:end_dim])]
        new_type_list = lhs + mid + rhs
        return TensorType(tuple(new_type_list))
    else:
        raise TypeError(
            f"Incompatible dimensions {start_dim}, {end_dim - 1} in type {tensor_type}"
        )


@register_inference_rule(torch.flatten)
def flatten_inference_rule(n: Node) -> Any:
    """
    Applies the flatten shape information to the input then gets the
    greatest upper bound of the resulting type and the existing type
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # set the default start and end dims
    start_dim = 1
    end_dim = -1

    if len(n.args) > 1:
        if not isinstance(n.args[1], int):
            raise AssertionError(f"Expected int, got {type(n.args[1])}")
        start_dim = n.args[1]

    if len(n.args) > 2:
        if not isinstance(n.args[2], int):
            raise AssertionError(f"Expected int, got {type(n.args[2])}")
        end_dim = n.args[2]

    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.dims))

    if isinstance(n.args[0].type, TensorType):
        output_type = flatten_check(n.args[0].type, start_dim, end_dim)
        n.type = get_greatest_upper_bound(output_type, n.type)

    return n.type


class GraphTypeChecker:
    def __init__(self, env: dict[str, Any], traced: torch.fx.GraphModule) -> None:
        self.env = env
        self.traced = traced

    def type_check(self) -> bool:
        """
        A gradual type checker for graphs
        Effect: every node's field type will be
        populated with a type after type-checking is done
        """
        graph = self.traced.graph

        # type check every node with gradual type rules
        # if any node does not type check return false
        for n in graph.nodes:
            self.type_check_node(n)
        return True

    def type_check_node(self, n: Node) -> Any:
        """
        Type check a given fx node.
        Current operations:
        - Reshape
        - Transpose
        - Add
        - Relu
        - conv2d
        - batchnorm2d
        - flatten
        - maxpool2d
        - adaptiveavgpool2d
        - linear
        """
        if n.type is None:
            n.type = Dyn

        if n.op == "placeholder":
            return n.type

        elif n.op == "get_attr":
            t = get_parameter(self.traced, n.target)  # type: ignore[arg-type]
            if isinstance(t.data, torch.Tensor):
                n.type = TensorType(t.data.shape)
            return n.type

        elif n.op == "call_function":
            if n.target is getattr:
                if getattr not in _INFERENCE_RULES:
                    raise AssertionError("getattr not in _INFERENCE_RULES")
                return _INFERENCE_RULES[n.target](n, self.traced)

            elif n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](n)
            else:
                raise RuntimeError(
                    f"No inference rule registered for target {n.target}!"
                )

        elif n.op == "call_module":
            # pyrefly: ignore[bad-argument-type]
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _INFERENCE_RULES:
                return _INFERENCE_RULES[type(module_instance)](n, module_instance)
            else:
                raise RuntimeError(
                    f"No inference rule registered for class {type(module_instance)}!"
                )

        elif n.op == "output":

            def get_node_type(a: Any) -> Any:
                return a.type

            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            return n.type

        else:
            raise NotImplementedError(f"Method {n.op} not yet implemented")


@register_refinement_rule(Conv2d)
def conv_refinement_rule(n: Node) -> list[Any] | None:
    """
    The equality constraints are between the first dimension of
    the input and output
    """
    res: list[Any] = []
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        res = [Equality(arg_type.dims[0], n.type.dims[0])]
        return res


@register_refinement_rule(torch.nn.Linear)
def linear_refinement_rule(n: Node) -> list[Any]:
    """
    The equality constraints are between the first dimension of
    the input and output
    """
    res: list[Any] = []
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        res = [Equality(arg_type.dims[0], n.type.dims[0])]
    return res


@register_refinement_rule(BatchNorm2d)
@register_refinement_rule(torch.nn.ReLU)
def all_eq(n: Node) -> list[Any]:
    """
    For operations where the input shape is equal to the output shape
    """
    res: list[Any] = []
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        args1 = arg_type.dims
        args2 = n.type.dims
        res = [Equality(args1[i], args2[i]) for i in range(len(args1))]
    return res


@register_refinement_rule(torch.nn.AdaptiveAvgPool2d)
@register_refinement_rule(torch.nn.MaxPool2d)
def first_two_eq(n: Node) -> list[Any]:
    """
    For operations where the first two dimensions of the input and output shape
    are equal
    """
    res: list[Any] = []
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        args1 = arg_type.dims
        args2 = n.type.dims
        res = [Equality(args1[0], args2[0]), Equality(args1[1], args2[1])]
    return res


@register_refinement_rule(torch.add)
@register_refinement_rule(operator.add)
def element_wise_eq(n: Node) -> list[Any]:
    """
    For element-wise operations and handles broadcasting.
    Note that after applying broadcasting to the arguments
    we are able to determine if certain dimensions have not been broadcast
    if they are symbolically equal.

    in this case, we can establish equality between those dimensions and the
    corresponding output dimensions.

    Note that it takes two iterations for this result. One iteration to establish
    equality between certain dimensions of the operands (requiring the whole solver
    including unification) and another iteration to establish equality between the operands
    and the resulting type, requiring another round of constraint generation and unification.
    """
    res: list[Any] = []
    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        arg_type1 = n.args[0].type
        arg_type2 = n.args[1].type
        if (
            isinstance(arg_type1, TensorType)
            and isinstance(arg_type2, TensorType)
            and isinstance(n.type, TensorType)
        ):
            args1, args2 = broadcast_types(arg_type1, arg_type2)
            # by this point, we know that args1 and args2 are the same size.
            a1 = args1.dims
            a2 = args2.dims
            a3 = n.type.dims

            # we would be here in the second iteration where we establish equality
            # between operand type dimensions and the resulting type dimensions
            r = []
            for x, y, z in zip(a1, a2, a3):
                if x == y:
                    r.append(Equality(x, z))
            res = r
    return res


@register_refinement_rule(torch.flatten)
def flatten_refinement_rule(n: Node) -> list[Any]:
    """
    Generates equality constraints between the dimensions of the input and output
    that will not be involved in the flatten operation
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    eq_const = []

    start_dim = 1
    end_dim = -1

    if len(n.args) > 1:
        if not isinstance(n.args[1], int):
            raise AssertionError(f"Expected int, got {type(n.args[1])}")
        start_dim = n.args[1]

    if len(n.args) > 2:
        if not isinstance(n.args[2], int):
            raise AssertionError(f"Expected int, got {type(n.args[2])}")
        end_dim = n.args[2]

    if isinstance(n.type, TensorType) and isinstance(n.args[0].type, TensorType):
        l = len(n.type.dims)
        arg_type = n.args[0].type
        start_dim = l if start_dim == -1 else start_dim
        end_dim = l + end_dim + 1 if end_dim < 0 else end_dim + 1

        for t1, t2 in zip(n.type.dims[0:start_dim], arg_type.dims[0:start_dim]):
            eq_const.append(Equality(t1, t2))

        for t1, t2 in zip(n.type.dims[end_dim:], arg_type.dims[end_dim:]):
            eq_const.append(Equality(t1, t2))
    return eq_const


@register_algebraic_expressions_inference_rule(Conv2d)
def conv_rule(n: Node, module_instance: Any) -> TensorType | None:
    """
    Represents the output in terms of an algebraic expression w.r.t
    the input when possible
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        w_in = arg_type.dims[3]
        h_in = arg_type.dims[2]
        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)
        new_type = TensorType((n.type.dims[0], n.type.dims[1], h_out, w_out))
        n.type = new_type
        return new_type


class Refine:
    """
    Symbolic shape inference.
    Generates constraints over type variables.
    Currently all constraints are equality constraints.
    """

    def __init__(self, traced: Any) -> None:
        self.constraints = []
        self.traced = traced
        self.symbol_iter = itertools.count(start=0, step=1)

    def refine(self) -> bool:
        """
        Generates constraints for
        every node in the graph based on
        the operation.
        """
        graph = self.traced.graph
        for n in graph.nodes:
            self.refine_node(n)
        return True

    def symbolic_relations(self) -> bool:
        """
        Infers algebraic relations
        """
        graph = self.traced.graph
        for n in graph.nodes:
            self.infer_symbolic_relations(n)
        return True

    def replace_dyn_with_fresh_var(self, typ: Any) -> Any:
        """
        Replace all unknown types with fresh type variables.
        """
        if typ == Dyn:
            new_symbol = Var(next(self.symbol_iter))
            return new_symbol
        elif isinstance(typ, TensorType):
            new_args = [self.replace_dyn_with_fresh_var(a) for a in typ.dims]
            return TensorType(tuple(new_args))
        elif isinstance(typ, list):
            return [self.replace_dyn_with_fresh_var(t) for t in typ]
        elif isinstance(typ, tuple):
            return (self.replace_dyn_with_fresh_var(t) for t in typ)
        else:
            return typ

    def convert_to_sympy_symbols(self, typ: Any) -> Any:
        """
        Replace all unknown types with fresh type variables.
        """
        if isinstance(typ, Var):
            return sympy.symbols(str(typ))
        elif isinstance(typ, TensorType):
            new_args = [self.convert_to_sympy_symbols(a) for a in typ.dims]
            return TensorType(tuple(new_args))
        elif isinstance(typ, list):
            return [self.convert_to_sympy_symbols(t) for t in typ]
        elif isinstance(typ, tuple):
            return (self.convert_to_sympy_symbols(t) for t in typ)
        else:
            return typ

    def refine_node(self, n: Node) -> Any:
        """
        Returns a list of equality constraints for
        call_module and call_function nodes.
        Models the relation between input and output dimensions
        using constraints in case they are both tensors.
        All operations used in resnet50 are defined.
        """
        if n.type is None:
            n.type = Dyn

        n.type = self.replace_dyn_with_fresh_var(n.type)

        if n.op == "call_function":
            if n.target in _REFINEMENT_RULES:
                self.constraints += _REFINEMENT_RULES[n.target](n)

        if n.op == "call_module":
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _REFINEMENT_RULES:
                self.constraints += _REFINEMENT_RULES[type(module_instance)](n)

        if n.op == "output":

            def get_node_type(a: Any) -> Any:
                return a.type

            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            return n.type

    def infer_symbolic_relations(self, n: Node) -> Any:
        n.type = self.convert_to_sympy_symbols(n.type)
        if n.op == "call_function":
            if n.target in _RULES:
                return _RULES[n.target](n)

        if n.op == "call_module":
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _RULES:
                return _RULES[type(module_instance)](n, module_instance)

        if n.op == "output":

            def get_node_type(a: Any) -> Any:
                return a.type

            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            return n.type


def get_parameter(traced: Any, target: str) -> torch.nn.Parameter:
    """
    Returns the parameter given by ``target`` if it exists,
    otherwise throws an error.

    See the docstring for ``get_submodule`` for a more detailed
    explanation of this method's functionality as well as how to
    correctly specify ``target``.

    Args:
        target: The fully-qualified string name of the Parameter
            to look for. (See ``get_submodule`` for how to specify a
            fully-qualified string.)

    Returns:
        torch.nn.Parameter: The Parameter referenced by ``target``

    Raises:
        AttributeError: If the target string references an invalid
            path or resolves to something that is not an
            ``nn.Parameter``
    """
    module_path, _, param_name = target.rpartition(".")

    mod: torch.nn.Module = traced.get_submodule(module_path)

    if not hasattr(mod, param_name):
        raise AttributeError(mod._get_name() + " has no attribute `" + param_name + "`")

    param: torch.nn.Parameter = getattr(mod, param_name)

    return param
<|endoftext|>import re
from collections.abc import Callable
from typing import Any

import torch
from torch.utils._pytree import tree_flatten_with_path, tree_map


KeyPath = tuple[Any, ...]
NonTensorShapeFn = Callable[[int | float], tuple[Any, ...]]

__all__ = [
    "normalize_source_name",
    "module_to_nested_dict",
    "track_dynamism_across_examples",
    "clone_and_convert_to_meta",
]


def normalize_source_name(name: str) -> str:
    # Match attribute access like .x and replace with ['x']
    return re.sub(r"\.([a-zA-Z_][a-zA-Z0-9_]*)", r"['\1']", name)


def module_to_nested_dict(module: torch.nn.Module) -> dict[str, Any]:
    """Recursively converts an nn.Module into a nested dictionary with explicit 'parameters' and 'modules' keys."""
    self_dict: dict[str, Any] = {}

    parameters: dict[str, torch.Tensor] = {}
    modules: dict[str, dict[str, Any]] = {}
    self_dict["_parameters"] = parameters
    self_dict["_modules"] = modules

    for attr_name in dir(module):
        try:
            if not attr_name.startswith("_") and not callable(
                getattr(module, attr_name)
            ):
                attr_value = getattr(module, attr_name)
                if (
                    not isinstance(attr_value, torch.nn.Module)
                    and isinstance(attr_value, (int, float, torch.Tensor))
                    and type(attr_value) is not bool
                ):
                    self_dict[attr_name] = attr_value
        except NotImplementedError:
            # Skip attributes that raise NotImplementedError since they won't
            # contain any dynamism anyways.
            continue

    for name, param in module.named_parameters(recurse=False):
        self_dict["_parameters"][name] = param
    for name, buffer in module.named_buffers(recurse=False):
        self_dict["_parameters"][name] = buffer

    for name, submodule in module.named_children():
        self_dict["_modules"][name] = module_to_nested_dict(submodule)

    return self_dict


def track_dynamism_across_examples(
    example_inputs: list[Any],
) -> dict[Any, Any]:
    """
    This function analyzes a list of example inputs to determine the dynamism of their shapes.
    It tracks whether the dimensions of tensors or non-tensor values change across
    different examples. The function returns a dictionary where each key represents
    a path to a value in the input examples, and the corresponding value is a tuple
    indicating which dimensions are dynamic (i.e., change across examples). This
    helps in understanding how the structure of data varies across different instances.
    """
    tracking: dict[KeyPath, tuple[list[set[Any]], bool]] = {}

    for ex in example_inputs:
        if "self" in ex and isinstance(ex["self"], torch.nn.Module):
            ex["self"] = module_to_nested_dict(ex["self"])
        leaves_with_paths, _ = tree_flatten_with_path(ex)
        for key_path, value in leaves_with_paths:
            if not isinstance(value, (int, float, torch.Tensor)):
                continue
            if isinstance(value, torch.Tensor):
                shape: tuple[int | float, ...] = tuple(value.shape)
                is_tensor = True
            else:
                shape = (value,)
                is_tensor = False
            if key_path not in tracking:
                tracking[key_path] = ([set() for _ in range(len(shape))], is_tensor)
            else:
                dim_sets, flag = tracking[key_path]
                if flag != is_tensor:
                    pass
                while len(dim_sets) < len(shape):
                    dim_sets.append(set())
            for i, dim in enumerate(shape):
                tracking[key_path][0][i].add(dim)

    output: dict[Any, Any] = {}
    for key_path, (dim_sets, _is_tensor) in tracking.items():
        final_dyn = tuple(len(s) > 1 for s in dim_sets)
        key_str = "L" + "".join(f"{str(k)}" for k in key_path)
        key = key_path[0].key  # type: ignore[attr-defined]
        if key not in output:
            output[key] = {}
        output[key][key_str] = final_dyn
    return output


def clone_and_convert_to_meta(example_input: Any) -> Any:
    """
    This function takes a list of example inputs and for each tensor, clones it and converts it to device=meta.
    For non-tensor values, it keeps the reference. It uses pytree to handle nested structures recursively.
    """

    def transform_fn(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.clone().to(device="meta")
        return value

    return tree_map(transform_fn, example_input)
<|endoftext|>from typing import Any, TypeAlias

import torch


__all__ = [
    "evaluate_conditional_with_constraints",
    "iterate_till_fixed_point",
    "transform_algebraic_expression",
    "transform_all_constraints",
    "transform_all_constraints_trace_time",
    "transform_dimension",
    "transform_to_z3",
    "transform_var",
]


# z3 is an optional dependency with no type stubs, so we use aliases for its types.
_Z3Expr: TypeAlias = Any
_Z3Result: TypeAlias = Any
from torch.fx.experimental.migrate_gradual_types.constraint import (
    BinConstraintD,
    BinConstraintT,
    BVar,
    Conj,
    Constraint,
    Disj,
    DVar,
    F,
    is_algebraic_expression,
    is_bool_expr,
    is_dim,
    Prod,
    T,
    TVar,
)
from torch.fx.experimental.migrate_gradual_types.constraint_generator import (
    ConstraintGenerator,
)
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import (
    transform_constraint,
)
from torch.fx.experimental.migrate_gradual_types.operation import (
    op_add,
    op_div,
    op_eq,
    op_gt,
    op_leq,
    op_lt,
    op_mod,
    op_mul,
    op_neq,
    op_sub,
)
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx.tensor_type import _DynType, Dyn, TensorType


try:
    import z3  # type: ignore[import]

    from torch.fx.experimental.migrate_gradual_types.z3_types import (
        D,
        tensor_type,
        z3_dyn,
    )

    HAS_Z3 = True

    def transform_to_z3(
        constraint: Constraint, counter: int, dimension_dict: dict[int, int]
    ) -> tuple[_Z3Expr, int]:
        if isinstance(constraint, Conj):
            conjuncts = []
            for c in constraint.conjucts:
                new_c, counter = transform_to_z3(c, counter, dimension_dict)
                conjuncts.append(new_c)
            return z3.And(conjuncts), counter

        elif isinstance(constraint, Disj):
            disjuncts = []
            for c in constraint.disjuncts:
                new_c, counter = transform_to_z3(c, counter, dimension_dict)
                disjuncts.append(new_c)
            return z3.Or(disjuncts), counter

        elif isinstance(constraint, T):
            return True, counter

        elif isinstance(constraint, F):
            return False, counter

        elif isinstance(constraint, BinConstraintT):
            if constraint.op == op_eq:
                lhs, counter = transform_var(
                    constraint.lhs,  # pyrefly: ignore[bad-argument-type]
                    counter,
                    dimension_dict,
                )
                rhs, counter = transform_var(
                    constraint.rhs,  # pyrefly: ignore[bad-argument-type]
                    counter,
                    dimension_dict,
                )
                return (lhs == rhs), counter

            else:
                raise NotImplementedError("Method not yet implemented")

        elif isinstance(constraint, BinConstraintD):
            if constraint.op == op_eq:
                if isinstance(constraint.lhs, BVar) and is_bool_expr(constraint.rhs):
                    transformed_rhs, counter = transform_to_z3(
                        constraint.rhs,  # pyrefly: ignore[bad-argument-type]
                        counter,
                        dimension_dict,
                    )
                    transformed_lhs = z3.Bool(constraint.lhs.c)
                    return transformed_lhs == transformed_rhs, counter

                elif is_dim(constraint.lhs) and is_dim(constraint.rhs):
                    # with dimension transformations we consider the encoding
                    lhs, counter = transform_dimension(
                        constraint.lhs,  # pyrefly: ignore[bad-argument-type]
                        counter,
                        dimension_dict,
                    )
                    rhs, counter = transform_dimension(
                        constraint.rhs,  # pyrefly: ignore[bad-argument-type]
                        counter,
                        dimension_dict,
                    )
                    return lhs == rhs, counter

                else:
                    # then we have an algebraic expression which means that we disregard the
                    # first element of the encoding
                    lhs, counter = transform_algebraic_expression(
                        constraint.lhs,  # pyrefly: ignore[bad-argument-type]
                        counter,
                        dimension_dict,
                    )
                    rhs, counter = transform_algebraic_expression(
                        constraint.rhs,  # pyrefly: ignore[bad-argument-type]
                        counter,
                        dimension_dict,
                    )
                    return lhs == rhs, counter

            # The assumption here is that the LHS and RHS must be dimensions
            elif constraint.op == op_neq:
                if not is_dim(constraint.lhs):
                    raise AssertionError("Expected lhs to be a dimension")
                if not is_dim(constraint.rhs):
                    raise AssertionError("Expected rhs to be a dimension")
                lhs, counter = transform_dimension(
                    constraint.lhs,  # pyrefly: ignore[bad-argument-type]
                    counter,
                    dimension_dict,
                )
                rhs, counter = transform_dimension(
                    constraint.rhs,  # pyrefly: ignore[bad-argument-type]
                    counter,
                    dimension_dict,
                )
                if constraint.rhs == Dyn or constraint.lhs == Dyn:
                    if constraint.rhs == Dyn:
                        return lhs.arg(0) == 1, counter
                    else:
                        return rhs.arg(0) == 1, counter

                # if one of the instances is a number
                elif isinstance(constraint.lhs, int) or isinstance(constraint.rhs, int):
                    if isinstance(constraint.lhs, int):
                        return (
                            z3.Or(
                                [
                                    rhs.arg(0) == 0,
                                    z3.And([rhs.arg(0) == 1, lhs.arg(1) != rhs.arg(1)]),
                                ]
                            ),
                            counter,
                        )

                    else:
                        return (
                            z3.Or(
                                [
                                    lhs.arg(0) == 0,
                                    z3.And([lhs.arg(0) == 1, lhs.arg(1) != rhs.arg(1)]),
                                ]
                            ),
                            counter,
                        )

                else:
                    return (
                        z3.Or(
                            [
                                z3.And([lhs.arg(0) == 0, rhs.arg(0) != 0]),
                                z3.And([lhs.arg(0) != 0, rhs.arg(0) == 0]),
                                z3.And(
                                    [
                                        lhs.arg(0) != 0,
                                        rhs.arg(0) != 0,
                                        lhs.arg(1) != rhs.arg(1),
                                    ]
                                ),
                            ]
                        ),
                        counter,
                    )

            elif constraint.op == op_leq:
                # if the dimensions are not dyn, this will come into effect
                # there would have been another constraint specifying if a given dimension
                # is dyn or not
                if not (is_dim(constraint.lhs) and is_dim(constraint.rhs)):
                    raise AssertionError("Expected both lhs and rhs to be dimensions")
                lhs, counter = transform_algebraic_expression(
                    constraint.lhs,  # pyrefly: ignore[bad-argument-type]
                    counter,
                    dimension_dict,
                )
                rhs, counter = transform_algebraic_expression(
                    constraint.rhs,  # pyrefly: ignore[bad-argument-type]
                    counter,
                    dimension_dict,
                )
                return lhs <= rhs, counter

            elif constraint.op == op_gt:
                if not (is_dim(constraint.lhs) and is_dim(constraint.rhs)):
                    raise AssertionError("Expected both lhs and rhs to be dimensions")
                lhs, counter = transform_algebraic_expression(
                    constraint.lhs,  # pyrefly: ignore[bad-argument-type]
                    counter,
                    dimension_dict,
                )
                rhs, counter = transform_algebraic_expression(
                    constraint.rhs,  # pyrefly: ignore[bad-argument-type]
                    counter,
                    dimension_dict,
                )
                return lhs > rhs, counter

            elif constraint.op == op_lt:
                if not (is_dim(constraint.lhs) and is_dim(constraint.rhs)):
                    raise AssertionError("Expected both lhs and rhs to be dimensions")
                lhs, counter = transform_algebraic_expression(
                    constraint.lhs,  # pyrefly: ignore[bad-argument-type]
                    counter,
                    dimension_dict,
                )
                rhs, counter = transform_algebraic_expression(
                    constraint.rhs,  # pyrefly: ignore[bad-argument-type]
                    counter,
                    dimension_dict,
                )
                return lhs < rhs, counter

            else:
                raise NotImplementedError("operation not yet implemented")

        else:
            raise NotImplementedError("Operation not yet implemented")

    def transform_var(
        tensor: TVar | TensorType | _DynType,
        counter: int,
        dimension_dict: dict[int, int],
    ) -> tuple[_Z3Expr, int]:
        """
        Transforms tensor variables to a format understood by z3
        Args:
            tensor: Tensor variable or a tensor type potentially with variable dimensions
        Returns: Transformed variable to a z3 format

        """
        if isinstance(tensor, TensorType):
            res: list[_Z3Expr] = []
            for t in tensor.dims:
                transformed, counter = transform_dimension(t, counter, dimension_dict)
                res.append(transformed)

            if len(res) > 4:
                raise AssertionError(f"Expected res length <= 4, got {len(res)}")
            if len(tensor.dims) == 1:
                return tensor_type.tensor1(res[0]), counter
            elif len(tensor.dims) == 2:
                return tensor_type.tensor2(res[0], res[1]), counter
            elif len(tensor.dims) == 3:
                return tensor_type.tensor3(res[0], res[1], res[2]), counter
            elif len(tensor.dims) == 4:
                return tensor_type.tensor4(res[0], res[1], res[2], res[3]), counter
            else:
                raise AssertionError(
                    f"Unexpected tensor args length: {len(tensor.dims)}"
                )

        elif tensor == Dyn:
            return z3_dyn, counter

        elif isinstance(tensor, TVar):
            return z3.Const(tensor.tvar, tensor_type), counter

        else:
            raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")

    def transform_dimension(
        dimension: DVar | int | _DynType, counter: int, dimension_dict: dict[int, int]
    ) -> tuple[_Z3Expr, int]:
        """
        Takes a dimension variable or a number and transforms it to a tuple
        according to our scheme
        Args:
            dimension: The dimension to be transformed
            counter: variable tracking

        Returns:  tuple and the current counter

        """
        if dimension == Dyn:
            counter += 1
            return D(0, z3.Int(counter)), counter
        elif isinstance(dimension, int):
            return D(1, dimension), counter
        elif isinstance(dimension, DVar):
            if dimension.c in dimension_dict:
                return (
                    D(z3.Int(dimension_dict[dimension.c]), z3.Int(dimension.c)),
                    counter,
                )
            else:
                counter += 1
                dimension_dict[dimension.c] = counter
                return D(z3.Int(counter), z3.Int(dimension.c)), counter

        else:
            raise NotImplementedError(f"Unsupported dimension type: {type(dimension)}")

    def transform_algebraic_expression(
        expr: DVar | int | _DynType | Prod | BinConstraintD,
        counter: int,
        dimension_dict: dict[int, int],
    ) -> tuple[_Z3Expr, int]:
        """
        Transforms an algebraic expression to z3 format
        Args:
            expr: An expression is either a dimension variable or an algebraic-expression


        Returns: the transformed expression

        """
        if not (is_algebraic_expression(expr) or is_dim(expr)):
            raise AssertionError("Expected algebraic expression or dimension")

        if is_dim(expr):
            transformed, counter = transform_dimension(
                expr,  # pyrefly: ignore[bad-argument-type]
                counter,
                dimension_dict,
            )
            return transformed.arg(1), counter

        elif isinstance(expr, Prod):
            dims = []
            for dim in expr.products:
                if not is_dim(dim):
                    raise AssertionError("Expected dimension in Prod")
                d, counter = transform_dimension(dim, counter, dimension_dict)
                dims.append(d.arg(1))
            return z3.Product(dims), counter

        elif is_algebraic_expression(expr):
            lhs, counter = transform_algebraic_expression(
                expr.lhs,  # pyrefly: ignore[missing-attribute]
                counter,
                dimension_dict,
            )
            rhs, counter = transform_algebraic_expression(
                expr.rhs,  # pyrefly: ignore[missing-attribute]
                counter,
                dimension_dict,
            )

            if expr.op == op_sub:  # pyrefly: ignore[missing-attribute]
                c = lhs - rhs

            elif expr.op == op_add:
                c = lhs + rhs

            elif expr.op == op_div:
                c = lhs / rhs

            elif expr.op == op_mul:
                c = lhs * rhs

            elif expr.op == op_mod:
                c = lhs % rhs

            else:
                raise NotImplementedError("operation not yet implemented")

            return c, counter

        else:
            raise RuntimeError

    def transform_all_constraints(traced: torch.nn.Module, counter: int = 0) -> _Z3Expr:
        """
        Given a trace, generates constraints and transforms them to z3 format

        """
        dimension_dict: dict[int, int] = {}

        generator = ConstraintGenerator(traced)
        new_constraints, counter = generator.generate_constraints(counter)

        new_constraints, counter = iterate_till_fixed_point(new_constraints, counter)

        transformed, counter = transform_to_z3(new_constraints, counter, dimension_dict)
        # print(transformed)
        return transformed

    def iterate_till_fixed_point(
        constraints: Constraint, counter: int
    ) -> tuple[Constraint, int]:
        """
        Transform constraints till reaching a fixed point
        """
        old_c = None
        while old_c != constraints:
            old_c = constraints
            constraints, counter = transform_constraint(constraints, counter)
        return constraints, counter

    def transform_all_constraints_trace_time(
        tracer_root: torch.nn.Module, graph: Graph, node: Node, counter: int = 0
    ) -> tuple[_Z3Expr, _Z3Expr]:
        """
        Takes a node and a graph and generates two sets of constraints.
        One set constraints the node's constraints and another set
        constraints the negation of the node's constraints
        Args:
            tracer_root: the root for getting the module instances
            graph: the graph so far in the tracing process
            node: node that represents a conditional
            counter: variable tracking

        Returns: Two sets of constraints. One with a conjunction with the
        conditional constraint and the other with a conjunction with
        its negation.

        """
        dimension_dict: dict[int, int] = {}

        generator = ConstraintGenerator(tracer_root, graph)
        new_constraints, counter = generator.generate_constraints(counter)

        condition_constraint = new_constraints.conjucts[-1]

        # we know the constraint is a conjunction where the last constraint is about the conditional
        # so remove the last constraint
        new_constraints.conjucts = new_constraints.conjucts[:-1]

        # transform precision, matching, consistency till obtaining a fixed point
        new_constraints, counter = iterate_till_fixed_point(new_constraints, counter)

        # since the function returns a list of one element, we get the first element
        # we are only interested in the RHS in this case because the LHS just stores
        # the result

        # we make sure the constraint is of the form:
        # c = b where b is a boolean expression
        # and we consider b (constraint.rhs) for transformation
        if not isinstance(condition_constraint, BinConstraintD):
            raise TypeError(type(condition_constraint))
        if not isinstance(condition_constraint.lhs, BVar):
            raise AssertionError(f"Expected BVar, got {type(condition_constraint.lhs)}")
        if not is_bool_expr(condition_constraint.rhs):
            raise AssertionError("Expected bool expression for rhs")
        if not isinstance(condition_constraint.rhs, Constraint):
            raise TypeError(type(condition_constraint.rhs))
        condition_constraint_rhs = condition_constraint.rhs

        # transform the condition constraint
        condition_constraint_rhs, counter = iterate_till_fixed_point(
            condition_constraint_rhs, counter
        )

        transformed, counter = transform_to_z3(new_constraints, counter, dimension_dict)

        transformed_condition_constraint, counter = transform_to_z3(
            condition_constraint_rhs, counter, dimension_dict
        )

        negation_transformed_condition_constraint = z3.Not(
            transformed_condition_constraint
        )

        return z3.And([transformed, transformed_condition_constraint]), z3.And(
            [transformed, negation_transformed_condition_constraint]
        )

    def evaluate_conditional_with_constraints(
        tracer_root: torch.nn.Module,
        graph: Graph,
        node: Node,
        counter: int = 0,
        user_constraints: _Z3Expr | None = None,
    ) -> tuple[_Z3Result, _Z3Result]:
        """
        Given an IR and a node representing a conditional, evaluate the conditional
        and its negation
        Args:
            tracer_root: Tracer root for module instances
            node: The node to be evaluated

        Returns: the results of evaluating the condition and the negation with
        the rest of the constraints

        """

        (
            transformed_positive,
            transformed_negative,
        ) = transform_all_constraints_trace_time(tracer_root, graph, node, counter)

        s = z3.Solver()
        s.add(transformed_positive)
        if user_constraints is not None:
            s.add(user_constraints)
        condition = s.check()

        s = z3.Solver()
        s.add(transformed_negative)
        if user_constraints is not None:
            s.add(user_constraints)
        negation = s.check()
        return condition, negation

except ImportError:
    HAS_Z3 = False
<|endoftext|>import operator
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TypeAlias, TypeVar
from typing_extensions import ParamSpec

import torch
from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.experimental.migrate_gradual_types.constraint import (
    ApplyBroadcasting,
    BinConstraintD,
    BinConstraintT,
    BVar,
    CalcConv,
    CalcMaxPool,
    CalcProduct,
    CanReshape,
    Conj,
    Constraint,
    DGreatestUpperBound,
    Disj,
    DVar,
    F,
    GetItem,
    GetItemTensor,
    IndexSelect,
    T,
    TGreatestUpperBound,
    Transpose,
    TVar,
)
from torch.fx.experimental.migrate_gradual_types.operation import (
    op_add,
    op_consistency,
    op_div,
    op_eq,
    op_gt,
    op_leq,
    op_lt,
    op_matching,
    op_mul,
    op_neq,
    op_precision,
    op_sub,
)
from torch.fx.experimental.migrate_gradual_types.util import (
    gen_bvar,
    gen_dvar,
    gen_nat_constraints,
    gen_tensor_dims,
    gen_tvar,
)
from torch.fx.graph import Graph
from torch.fx.node import Node, Target
from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d


_T = TypeVar("_T")
_P = ParamSpec("_P")

_SymbolDict: TypeAlias = dict[Node, TVar | DVar | BVar]

_INFERENCE_RULES: dict[Target, Callable[..., tuple[list[Constraint], int]]] = {}

MAX_TENSOR_RANK = 4

__all__ = [
    "ConstraintGenerator",
    "adaptive_inference_rule",
    "add_layer_norm_constraints",
    "add_linear_constraints",
    "arange_inference_rule",
    "assert_inference_rule",
    "batchnorm_inference_rule",
    "bmm_inference_rule",
    "broadcasting_inference_rule",
    "conv2d_inference_rule",
    "cumsum_inference_rule",
    "embedding_inference_rule",
    "embedding_inference_rule_functional",
    "eq_inference_rule",
    "equality_inference_rule",
    "expand_inference_rule",
    "flatten_inference_rule",
    "full_inference_rule",
    "gen_broadcasting_constraints",
    "gen_embedding_rules",
    "gen_layer_norm_constraints",
    "generate_flatten_constraints",
    "get_attr_inference_rule",
    "getitem_inference_rule",
    "gt_inference_rule",
    "index_select_inference_rule",
    "layer_norm_functional",
    "layer_norm_inference_rule",
    "linear_constraints",
    "linear_inference_rule",
    "lt_inference_rule",
    "masked_fill_inference_rule",
    "maxpool_inference_rule",
    "neq_inference_rule",
    "range_check",
    "register_inference_rule",
    "relu_inference_rule",
    "reshape_inference_rule",
    "size_inference_rule",
    "tensor_inference_rule",
    "torch_dim_inference_rule",
    "torch_linear_inference_rule",
    "transpose_inference_rule",
    "type_inference_rule",
    "view_inference_rule",
]


def register_inference_rule(
    call_target: Target,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def register(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        if call_target in _INFERENCE_RULES:
            raise RuntimeError(f"Inference rule already registered for {call_target}!")
        _INFERENCE_RULES[call_target] = fn  # pyrefly: ignore[unsupported-operation]
        return fn

    return register


def generate_flatten_constraints(
    start_dim: int, end_dim: int, input: TVar, flattened: TVar, n: int, counter: int
) -> tuple[Conj, int]:
    d, counter = gen_tensor_dims(n, counter)
    c1 = BinConstraintT(input, TensorType(d), op_eq)
    start_dim = n if start_dim == -1 else abs(start_dim)
    end_dim = n + end_dim + 1 if end_dim < 0 else end_dim + 1
    c2 = CalcProduct(start_dim, end_dim, flattened, d)
    nat_constraints = gen_nat_constraints(d)
    return Conj([c1, c2, *nat_constraints]), counter


@register_inference_rule(getattr)
def get_attr_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    If the attribute is "device" then the tensor shape is preserved
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], str):
        raise AssertionError(f"Expected str, got {type(n.args[1])}")
    output, counter = gen_tvar(counter)
    symbols[n] = output

    input = symbols[n.args[0]]
    attr = n.args[1]

    if attr == "device":
        return [BinConstraintT(input, output, op_eq)], counter
    else:
        raise NotImplementedError("Not yet implemented")


@register_inference_rule(torch.bmm)
def bmm_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    Constraints that match the input to a size 3 tensor
    and switch the dimensions according to the rules
    of batch multiplication
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[1])}")

    bmm_output, counter = gen_tvar(counter)
    symbols[n] = bmm_output

    bmm_input1 = symbols[n.args[0]]
    bmm_input2 = symbols[n.args[1]]

    dims_input1, counter = gen_tensor_dims(3, counter)
    dims_input2, counter = gen_tensor_dims(3, counter)

    inputs_dyn = Conj(
        [
            BinConstraintT(bmm_input1, Dyn, op_eq),
            BinConstraintT(bmm_input2, Dyn, op_eq),
            BinConstraintT(bmm_output, Dyn, op_eq),
        ]
    )

    input1_dyn = Conj(
        [
            BinConstraintT(bmm_input1, Dyn, op_eq),
            BinConstraintT(bmm_input2, TensorType(dims_input2), op_eq),
            BinConstraintT(
                bmm_output, TensorType([dims_input2[0], Dyn, dims_input2[2]]), op_eq
            ),
        ]
    )

    input2_dyn = Conj(
        [
            BinConstraintT(bmm_input2, Dyn, op_eq),
            BinConstraintT(bmm_input1, TensorType(dims_input1), op_eq),
            BinConstraintT(
                bmm_output, TensorType([dims_input1[0], dims_input1[1], Dyn]), op_eq
            ),
        ]
    )

    consistency_constraints = [
        BinConstraintD(dims_input1[0], dims_input2[0], op_consistency)
    ]

    batch_size, counter = gen_dvar(counter)

    inputs_are_tensors = Conj(
        [
            BinConstraintT(bmm_input1, TensorType(dims_input1), op_eq),
            BinConstraintT(bmm_input2, TensorType(dims_input2), op_eq),
            BinConstraintT(
                bmm_output,
                TensorType([batch_size, dims_input1[1], dims_input2[2]]),
                op_eq,
            ),
            *consistency_constraints,
            DGreatestUpperBound(batch_size, dims_input1[0], dims_input2[0]),
        ]
    )

    return [Disj([inputs_dyn, input1_dyn, input2_dyn, inputs_are_tensors])], counter


@register_inference_rule("index_select")
def index_select_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    We constrain the second argument to a vector or Dyn.
    The output replaces the input with the shape of the vector
    at the position given by the index (first argument)
    """
    # print(n.args)
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], int):
        raise AssertionError(f"Expected int, got {type(n.args[1])}")
    if not isinstance(n.args[2], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[2])}")

    index_select, counter = gen_tvar(counter)
    symbols[n] = index_select

    dims, counter = gen_tensor_dims(1, counter)

    # equality constraint
    is_size_1 = BinConstraintT(symbols[n.args[2]], TensorType(dims), op_eq)
    is_dyn = BinConstraintT(symbols[n.args[2]], Dyn, op_eq)

    c2 = Conj(
        [
            is_size_1,
            Disj(
                [
                    IndexSelect(
                        i + 1,
                        symbols[  # pyrefly: ignore[bad-argument-type, bad-index]
                            n.args[0]
                        ],
                        dims[0],
                        n.args[1],
                        index_select,
                    )
                    for i in range(MAX_TENSOR_RANK)
                ]
            ),
        ]
    )
    c3 = Conj(
        [
            is_dyn,
            Disj(
                [
                    IndexSelect(
                        i + 1,
                        symbols[  # pyrefly: ignore[bad-argument-type, bad-index]
                            n.args[0]
                        ],
                        Dyn,
                        n.args[1],
                        index_select,
                    )
                    for i in range(MAX_TENSOR_RANK)
                ]
            ),
        ]
    )

    return [Disj([c2, c3])], counter


@register_inference_rule("expand")
def expand_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    We generate the exact constraints as we do for tensor additions but we constraint
    the rank of this expression to be equal to len(n.args[1:]) so that only
    those cases get considered for the output
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # define the output for expand
    expand, counter = gen_tvar(counter)
    symbols[n] = expand

    # since we do not have two nodes here, we will construct an argument variable
    e1 = symbols[n.args[0]]
    e2, counter = gen_tvar(counter)

    e2_nat_constraints = []
    for arg in n.args[1:]:
        if not isinstance(arg, (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(arg)}")
        if isinstance(arg, Node):
            if not isinstance(symbols[arg], DVar):
                raise AssertionError(f"Expected DVar, got {type(symbols[arg])}")
            e2_nat_constraints.append(BinConstraintD(0, symbols[arg], op_leq))

    e2_constraint = BinConstraintT(
        e2,
        TensorType(
            [
                arg
                if isinstance(arg, int)
                else symbols[arg]  # pyrefly: ignore[bad-index]
                for arg in n.args[1:]
            ]
        ),
        op_eq,
    )

    constraints, counter = gen_broadcasting_constraints(
        e1,  # pyrefly: ignore[bad-argument-type]
        e2,
        symbols,
        counter,
        expand,
    )

    # constraint the output size
    dims, counter = gen_tensor_dims(len(n.args[1:]), counter)
    nat_constraints = gen_nat_constraints(dims)
    c = [
        BinConstraintT(expand, TensorType(dims), op_eq),
        *nat_constraints,
        e2_constraint,
        *e2_nat_constraints,
    ]
    constraints += c

    return constraints, counter


@register_inference_rule(torch.nn.functional.gelu)
@register_inference_rule(torch.nn.functional.dropout)
@register_inference_rule(torch.nn.functional.softmax)
@register_inference_rule("detach")
@register_inference_rule("to")
@register_inference_rule("int")
@register_inference_rule("long")
@register_inference_rule("contiguous")
@register_inference_rule(torch.ones)
@register_inference_rule(torch.zeros)
def equality_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    We generate the constraint: input = output
    """
    output, counter = gen_tvar(counter)
    symbols[n] = output

    if isinstance(n.args[0], Node):
        input = symbols[n.args[0]]
        if isinstance(input, TVar):
            return [BinConstraintT(input, output, op_eq)], counter

        # then we have dimension variables
        else:
            for arg in n.args:
                if not isinstance(symbols[arg], DVar):  # pyrefly: ignore[bad-index]
                    raise AssertionError(
                        f"Expected DVar, got {type(symbols[arg])}"  # pyrefly: ignore[bad-index]
                    )
        my_size = [symbols[arg] for arg in n.args]  # pyrefly: ignore[bad-index]
        return [BinConstraintT(output, TensorType(my_size), op_eq)], counter

    elif isinstance(n.args[0], tuple):
        # then the tuple is the size
        if len(n.args[0]) > 4:
            raise AssertionError(f"Expected len <= 4, got {len(n.args[0])}")
        my_size = [symbols[arg] for arg in n.args[0]]  # pyrefly: ignore[bad-index]
        return [BinConstraintT(output, TensorType(my_size), op_eq)], counter
    else:
        raise NotImplementedError("Method not yet implemented")


@register_inference_rule("transpose")
def transpose_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    Can be considered as a sequence of two index selects, so we generate constraints accordingly
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], int):
        raise AssertionError(f"Expected int, got {type(n.args[1])}")
    if not isinstance(n.args[2], int):
        raise AssertionError(f"Expected int, got {type(n.args[2])}")

    output, counter = gen_tvar(counter)
    symbols[n] = output

    from_arg = symbols[n.args[0]]
    if not isinstance(from_arg, TVar):
        raise AssertionError(f"Expected TVar, got {type(from_arg)}")

    # input and output are dyn
    is_dyn = Conj(
        [BinConstraintT(from_arg, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)]
    )

    # or input is a tensor and we actually do the replacement
    c3 = Disj(
        [
            Transpose(i + 1, from_arg, n.args[1], n.args[2], output)
            for i in range(MAX_TENSOR_RANK)
        ]
    )

    return [Disj([is_dyn, c3])], counter


@register_inference_rule("type_as")
def type_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    We generate the constraint: input = output
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[1])}")

    output, counter = gen_tvar(counter)
    symbols[n] = output

    from_arg = symbols[n.args[0]]
    to_arg = symbols[n.args[1]]

    if not isinstance(from_arg, TVar):
        raise AssertionError(f"Expected TVar, got {type(from_arg)}")
    if not isinstance(to_arg, TVar):
        raise AssertionError(f"Expected TVar, got {type(to_arg)}")

    return [
        BinConstraintT(from_arg, to_arg, op_consistency),
        BinConstraintT(output, to_arg, op_eq),
    ], counter


@register_inference_rule("masked_fill_")
def masked_fill_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    Similar to addition. For now we implement the constraints when
    the argument is a boolean tensor. There is also a case for when
    it is a condition. We will leave this out for now.
    """

    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[1])}")

    # We will retrieve the type variables from the symbol table
    # and confirm they are tensor variables

    e1 = symbols[n.args[0]]
    e2 = symbols[n.args[1]]

    if isinstance(e1, TVar) and isinstance(e2, TVar):
        masked_fill_tensor, counter = gen_tvar(counter)
        symbols[n] = masked_fill_tensor
        return gen_broadcasting_constraints(
            e1, e2, symbols, counter, masked_fill_tensor
        )
    else:
        raise NotImplementedError("Not yet implemented")


@register_inference_rule(torch.nn.functional.embedding)
def embedding_inference_rule_functional(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    embedding_dim_weights = symbols[n.args[1]]  # pyrefly: ignore[bad-index]

    # will treat this as a static shape. So we will not use matching.
    weight_dims, counter = gen_tensor_dims(2, counter)
    equality_constraint = BinConstraintT(
        embedding_dim_weights, TensorType(weight_dims), op_eq
    )
    embedding_dim = weight_dims[1]
    constraints, counter = gen_embedding_rules(n, symbols, embedding_dim, counter)
    return [equality_constraint] + constraints, counter


@register_inference_rule(torch.nn.modules.sparse.Embedding)
def embedding_inference_rule(
    n: Node,
    module_instance: torch.nn.Embedding,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    """
    The output shape differs from the input shape in the last dimension
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    return gen_embedding_rules(n, symbols, module_instance.embedding_dim, counter)


def gen_embedding_rules(
    n: Node, symbols: _SymbolDict, embedding_dim: int | DVar, counter: int
) -> tuple[list[Constraint], int]:
    embedding_output, counter = gen_tvar(counter)
    symbols[n] = embedding_output
    embedding_input = symbols[n.args[0]]  # pyrefly: ignore[bad-index]

    input_dyn = BinConstraintT(embedding_input, Dyn, op_eq)
    output_dyn = BinConstraintT(embedding_output, Dyn, op_eq)

    c1 = Conj([input_dyn, output_dyn])
    c2 = []

    for i in range(1, MAX_TENSOR_RANK):
        new_dims, counter = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(new_dims)

        # we consider all tensor sizes and append embedding_dim to the end of the output dimension in all cases
        c_tensor_i = Conj(
            [
                BinConstraintT(embedding_input, TensorType(new_dims), op_eq),
                BinConstraintT(
                    embedding_output, TensorType(new_dims + [embedding_dim]), op_eq
                ),
            ]
            + nat_constraints
        )
        c2.append(c_tensor_i)

    return [Disj([c1, Disj(c2)])], counter


@register_inference_rule(torch.tensor)
def tensor_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    If the tensor is a scalar, we will skip it since we
    do not support scalars yet. We will add support in the future
    if it's needed. For our examples so far, scalars are not needed.
    """
    return [], counter  # pyrefly: ignore[implicit-any]


@register_inference_rule("reshape")
@register_inference_rule("view")
def view_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    Similar to reshape but with an extra condition on the strides
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # generate the new variable
    my_view, counter = gen_tvar(counter)
    symbols[n] = my_view

    src_var = symbols[n.args[0]]
    t2 = [
        symbols[elem] if isinstance(elem, Node) else elem for elem in n.args[1:]
    ]  # target shape
    t2_type = []
    num_constraints = []

    for t in t2:
        if t == -1:
            var, counter = gen_dvar(counter)
            t2_type.append(var)
            # pyrefly: ignore [bad-argument-type]
            num_constraints.append(BinConstraintD(var, Dyn, op_neq))

        else:
            # pyrefly: ignore [bad-argument-type]
            num_constraints.append(BinConstraintD(t, Dyn, op_neq))
            t2_type.append(t)  # type: ignore[arg-type]

    t2_type = TensorType(t2_type)  # type: ignore[assignment]

    c1 = BinConstraintT(my_view, t2_type, op_eq)
    c2 = CanReshape(src_var, t2_type)  # pyrefly: ignore[bad-argument-type]

    # TODO: add the extra check mentioned here:
    # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view

    return [c1, c2] + num_constraints, counter  # type: ignore[operator]


@register_inference_rule("size")
def size_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    The constraint is just lhs = rhs.
    Ex: size = input_ids.size()
    """

    if len(n.args) == 1:
        # generate the new variable
        size, counter = gen_tvar(counter)
        symbols[n] = size
        input = symbols[n.args[0]]  # pyrefly: ignore[bad-index]
        c = BinConstraintT(input, size, op_eq)
        return [c], counter

    elif len(n.args) == 2:
        # TODO: review this rule; should input = dyn; output = dyn be included here?
        if isinstance(n.args[1], int):
            # generate the new variable
            size_index, counter = gen_dvar(counter)
            symbols[n] = size_index
            input = symbols[n.args[0]]  # pyrefly: ignore[bad-index]
            c2 = [
                GetItem(
                    i + 1,
                    n.args[1],
                    size_index,
                    input,  # pyrefly: ignore[bad-argument-type]
                )
                for i in range(MAX_TENSOR_RANK)
            ]
            c3 = BinConstraintD(0, size_index, op_leq)

            input_dyn = BinConstraintT(input, Dyn, op_eq)
            output_dyn = BinConstraintD(size_index, Dyn, op_eq)
            c1 = Conj([input_dyn, output_dyn])

            return [Disj([c1, Conj([Disj(c2), c3])])], counter

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError


def range_check(i: int, n: int) -> T | F:
    """
    Checks if an index i is within range of a size n list
    Args:
        i: index
        n: list size

    Returns: Boolean
    """
    if i >= 0:
        return T() if i < n else F()
    else:
        return T() if i >= n else F()


@register_inference_rule(torch.cumsum)
def cumsum_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    Input and output shapes should be equal
    We should verify that the index is valid
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_1 = n.args[1] if len(n.args) > 1 else n.kwargs["dim"]
    if not isinstance(arg_1, int):
        raise AssertionError(f"Expected int, got {type(arg_1)}")

    output, counter = gen_tvar(counter)
    symbols[n] = output
    input = symbols[n.args[0]]

    input_dyn = BinConstraintT(input, Dyn, op_eq)
    output_dyn = BinConstraintT(output, Dyn, op_eq)
    c1 = Conj([input_dyn, output_dyn])
    c2 = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims, counter = gen_tensor_dims(i, counter)

        nat_constraints = gen_nat_constraints(new_dims)

        c_tensor_i = Conj(
            [
                BinConstraintT(input, TensorType(new_dims), op_eq),
                BinConstraintT(output, TensorType(new_dims), op_eq),
            ]
            + [range_check(arg_1, i)]
            + nat_constraints
        )

        c2.append(c_tensor_i)
    dyn_or_tensor = Disj([c1, Disj(c2)])
    return [dyn_or_tensor], counter


@register_inference_rule(_assert_is_none)
def assert_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if len(n.users) != 0:
        raise AssertionError(f"Expected no users, got {len(n.users)}")
    return [], counter  # pyrefly: ignore[implicit-any]


@register_inference_rule(operator.getitem)
def getitem_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # dimension output case
    if isinstance(n.args[1], int):
        # create and store the new dimension variable
        get_item_output, counter = gen_dvar(counter)
        symbols[n] = get_item_output

        # retrieve arg variables
        get_item_arg = symbols[n.args[0]]
        if not isinstance(get_item_arg, TVar):
            raise AssertionError(f"Expected TVar, got {type(get_item_arg)}")

        # if the input is dynamic, we accept any index and return
        # a dynamic dimension as output
        input_dyn = BinConstraintT(get_item_arg, Dyn, op_eq)
        output_dyn = BinConstraintD(get_item_output, Dyn, op_eq)
        c1 = Conj([input_dyn, output_dyn])

        # if the input is a tensor,
        # generate a getItem constraint which will be expanded based on the
        # tensor dimension.

        c2 = [
            GetItem(i + 1, n.args[1], get_item_output, get_item_arg)
            for i in range(MAX_TENSOR_RANK)
        ]

        # since the output is a dimension, we make sure it's a natural number
        # added as a conjunction to the disjunction of c2
        c3 = BinConstraintD(0, get_item_output, op_leq)
        return [Disj([c1, Conj([Disj(c2), c3])])], counter

    # tensor output case
    elif isinstance(n.args[1], tuple):
        # create and store the new tensor variable
        get_item_output, counter = gen_tvar(counter)  # type: ignore[arg-type,assignment]
        symbols[n] = get_item_output

        # retrieve arg variables
        if n.args[0] in symbols:
            get_item_arg = symbols[n.args[0]]
            if not isinstance(get_item_arg, TVar):
                raise AssertionError(f"Expected TVar, got {type(get_item_arg)}")

            input_dyn = BinConstraintT(get_item_arg, Dyn, op_eq)
            output_dyn = BinConstraintT(get_item_output, Dyn, op_eq)  # type: ignore[assignment]
            c1 = Conj([input_dyn, output_dyn])

            c2 = [
                GetItemTensor(i + 1, n.args[1], get_item_output, get_item_arg)  # type: ignore[misc]
                for i in range(MAX_TENSOR_RANK)
            ]
        else:
            # TODO: we should figure out why there is a key-error here.
            return [], counter  # pyrefly: ignore[implicit-any]

        return [Disj([c1, *c2])], counter

    else:
        raise RuntimeError("Method not yet implemented")


@register_inference_rule(operator.gt)
def gt_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[0])}")
    if not isinstance(n.args[1], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[1])}")

    # We make sure this node will not be used again. We do not
    # generate a constraint about that node. Only about the operands.

    e1 = symbols[n.args[0]] if isinstance(n.args[0], Node) else n.args[0]
    e2 = symbols[n.args[1]] if isinstance(n.args[1], Node) else n.args[1]

    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(e1, TVar) and isinstance(e2, TVar):
            gt_tensor, counter = gen_tvar(counter)
            symbols[n] = gt_tensor
            return gen_broadcasting_constraints(e1, e2, symbols, counter, gt_tensor)

        elif isinstance(e1, DVar) and isinstance(e2, DVar):
            # This is meant to be used for flow analysis only
            gt_constraint = BinConstraintD(e1, e2, op_gt)

            my_gt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_gt, gt_constraint, op_eq)
            return [equality_constraint], counter

        else:
            raise RuntimeError("Sort Mismatch")

    elif isinstance(n.args[0], Node) and not isinstance(n.args[1], Node):
        if isinstance(e1, DVar):
            # This is meant to be used for flow analysis only
            gt_constraint = BinConstraintD(e1, e2, op_gt)

            my_gt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_gt, gt_constraint, op_eq)
            return [equality_constraint], counter

        elif isinstance(e1, TVar) and isinstance(e2, int):
            # then we made the wrong assumption about the argument being a tensor
            # so we should fix the assumption
            warnings.warn(
                f"Made the wrong assumption for node {n}. Correctness not guaranteed."
            )

            new_e1, counter = gen_dvar(counter)
            symbols[n.args[0]] = new_e1
            symbols[n.args[0]]

            gt_constraint = BinConstraintD(new_e1, e2, op_gt)

            my_gt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_gt, gt_constraint, op_eq)
            return [equality_constraint], counter

        else:
            raise NotImplementedError("Method not yet implemented")

    else:
        raise NotImplementedError("Method not yet implemented")


@register_inference_rule(operator.eq)
def eq_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[0])}")
    if not isinstance(n.args[1], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[1])}")

    e1 = symbols[n.args[0]] if isinstance(n.args[0], Node) else n.args[0]
    e2 = symbols[n.args[1]] if isinstance(n.args[1], Node) else n.args[1]

    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(e1, TVar) and isinstance(e2, TVar):
            eq_tensor, counter = gen_tvar(counter)
            symbols[n] = eq_tensor
            return gen_broadcasting_constraints(e1, e2, symbols, counter, eq_tensor)

        elif isinstance(e1, DVar) and isinstance(e2, DVar):
            # This is meant to be used for flow analysis only
            eq_constraint = BinConstraintD(e1, e2, op_eq)

            my_eq, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_eq, eq_constraint, op_eq)
            return [equality_constraint], counter

        else:
            raise RuntimeError("Sort Mismatch")

    elif isinstance(n.args[0], Node) and not isinstance(n.args[1], Node):
        if isinstance(e1, DVar):
            # This is meant to be used for flow analysis only
            eq_constraint = BinConstraintD(e1, e2, op_eq)

            my_eq, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_eq, eq_constraint, op_eq)
            return [equality_constraint], counter
        else:
            raise NotImplementedError("Method not yet implemented")
    else:
        raise NotImplementedError("Method not yet implemented")


@register_inference_rule(operator.ne)
def neq_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    Translates to inconsistent in gradual types.
    To prove inequality, we should prove that
    tensors are either different sizes or
    disagree on at least one dimension

    This is a WIP (works when the condition
    is false. We are working on making this operation work
    when the condition is true as well)
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], tuple):
        raise AssertionError(f"Expected tuple, got {type(n.args[1])}")

    # implementing for size 3 and 4
    if len(n.args[1]) == 3:
        if not isinstance(n.args[1][0], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][0])}")
        if not isinstance(n.args[1][1], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][1])}")
        if not isinstance(n.args[1][2], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][2])}")

        lhs = symbols[n.args[0]]

        b, counter = gen_tensor_dims(4, counter)
        input_is_size3 = BinConstraintT(lhs, TensorType([b[0], b[1], b[2]]), op_eq)

        d1 = n.args[1][0] if isinstance(n.args[1][0], int) else symbols[n.args[1][0]]
        d2 = n.args[1][1] if isinstance(n.args[1][1], int) else symbols[n.args[1][1]]
        d3 = n.args[1][2] if isinstance(n.args[1][2], int) else symbols[n.args[1][2]]

        # dimensions not equal
        my_ne, counter = gen_bvar(counter)
        neq_1 = BinConstraintD(d1, b[0], op_neq)
        neq_2 = BinConstraintD(d2, b[1], op_neq)
        neq_3 = BinConstraintD(d3, b[2], op_neq)

        # dimensions inconsistent
        dims_inconsistent1 = Conj(
            [BinConstraintD(d1, Dyn, op_neq), BinConstraintD(b[0], Dyn, op_neq), neq_1]
        )
        dims_inconsistent2 = Conj(
            [BinConstraintD(d2, Dyn, op_neq), BinConstraintD(b[1], Dyn, op_neq), neq_2]
        )
        dims_inconsistent3 = Conj(
            [BinConstraintD(d3, Dyn, op_neq), BinConstraintD(b[2], Dyn, op_neq), neq_3]
        )

        dims_inconsistent = Disj(
            [dims_inconsistent1, dims_inconsistent2, dims_inconsistent3]
        )

        # we are covering size 3 and 4 only for now
        ne_constraint = Conj([input_is_size3, dims_inconsistent])

        my_ne, counter = gen_bvar(counter)
        equality_constraint = BinConstraintD(my_ne, ne_constraint, op_eq)

    elif len(n.args[1]) == 4:
        if not isinstance(n.args[1][0], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][0])}")
        if not isinstance(n.args[1][1], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][1])}")
        if not isinstance(n.args[1][2], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][2])}")
        if not isinstance(n.args[1][3], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][3])}")

        lhs = symbols[n.args[0]]

        b1, counter = gen_dvar(counter)
        b2, counter = gen_dvar(counter)
        b3, counter = gen_dvar(counter)
        b4, counter = gen_dvar(counter)

        input_is_size4 = BinConstraintT(lhs, TensorType([b1, b2, b3, b4]), op_eq)

        d1 = n.args[1][0] if isinstance(n.args[1][0], int) else symbols[n.args[1][0]]
        d2 = n.args[1][1] if isinstance(n.args[1][1], int) else symbols[n.args[1][1]]
        d3 = n.args[1][2] if isinstance(n.args[1][2], int) else symbols[n.args[1][2]]
        d4 = n.args[1][3] if isinstance(n.args[1][3], int) else symbols[n.args[1][3]]

        # dimensions not equal
        my_ne, counter = gen_bvar(counter)
        neq_1 = BinConstraintD(d1, b1, op_neq)
        neq_2 = BinConstraintD(d2, b2, op_neq)
        neq_3 = BinConstraintD(d3, b3, op_neq)
        neq_4 = BinConstraintD(d4, b4, op_neq)

        # dimensions to inconsistent
        dims_inconsistent1 = Conj(
            [BinConstraintD(d1, Dyn, op_neq), BinConstraintD(b1, Dyn, op_neq), neq_1]
        )
        dims_inconsistent2 = Conj(
            [BinConstraintD(d2, Dyn, op_neq), BinConstraintD(b2, Dyn, op_neq), neq_2]
        )
        dims_inconsistent3 = Conj(
            [BinConstraintD(d3, Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq), neq_3]
        )
        dims_inconsistent4 = Conj(
            [BinConstraintD(d4, Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq), neq_4]
        )

        dims_inconsistent = Disj(
            [
                dims_inconsistent1,
                dims_inconsistent2,
                dims_inconsistent3,
                dims_inconsistent4,
            ]
        )

        ne_constraint = Conj([input_is_size4, dims_inconsistent])

        my_ne, counter = gen_bvar(counter)

        equality_constraint = BinConstraintD(my_ne, ne_constraint, op_eq)

    else:
        raise NotImplementedError("Method not yet implemented")

    return [equality_constraint], counter


@register_inference_rule(operator.lt)
def lt_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[0])}")
    if not isinstance(n.args[1], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[1])}")

    # We make sure this node will not be used again. We do not
    # generate a constraint about that node. Only about the operands.

    e1 = symbols[n.args[0]] if isinstance(n.args[0], Node) else n.args[0]
    e2 = symbols[n.args[1]] if isinstance(n.args[1], Node) else n.args[1]

    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(e1, TVar) and isinstance(e2, TVar):
            lt_tensor, counter = gen_tvar(counter)
            symbols[n] = lt_tensor
            return gen_broadcasting_constraints(e1, e2, symbols, counter, lt_tensor)

        elif isinstance(e1, DVar) and isinstance(e2, DVar):
            # This is meant to be used for flow analysis only
            lt_constraint = BinConstraintD(e1, e2, op_lt)

            my_lt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_lt, lt_constraint, op_eq)
            return [equality_constraint], counter

        else:
            raise RuntimeError("Sort Mismatch")

    elif isinstance(n.args[0], Node) and not isinstance(n.args[1], Node):
        if isinstance(e1, DVar):
            # This is meant to be used for flow analysis only
            lt_constraint = BinConstraintD(e1, e2, op_lt)

            my_lt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_lt, lt_constraint, op_eq)
            return [equality_constraint], counter
        else:
            raise NotImplementedError("Method not yet implemented")

    else:
        raise NotImplementedError("Method not yet implemented")


@register_inference_rule(torch.full)
def full_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    full, counter = gen_tvar(counter)
    symbols[n] = full
    res = []

    if not isinstance(n.args[0], Iterable):
        raise AssertionError(f"Expected Iterable, got {type(n.args[0])}")
    for arg in n.args[0]:
        dim = (
            arg if isinstance(arg, int) else symbols[arg]  # pyrefly: ignore[bad-index]
        )
        res.append(dim)
    c = BinConstraintT(full, TensorType(list(res)), op_eq)  # type: ignore[arg-type]
    return [c], counter


# TODO normalize index
@register_inference_rule(torch.arange)
def arange_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    start = 0
    step = 1

    if len(n.args) == 1:
        end = symbols[n.args[0]]  # pyrefly: ignore[bad-index]
    else:
        raise NotImplementedError("Not yet implemented")

    # int((end - start) / step)
    d1, counter = gen_dvar(counter)
    size_constraint = BinConstraintD(
        d1, BinConstraintD(BinConstraintD(end, start, op_sub), step, op_div), op_eq
    )
    arange, counter = gen_tvar(counter)
    symbols[n] = arange

    # either a parameter is a number or it is Dyn
    c1 = Disj(
        [
            BinConstraintD(end, Dyn, op_eq),
            BinConstraintD(start, Dyn, op_eq),
            BinConstraintD(step, Dyn, op_eq),
        ]
    )
    c2 = BinConstraintD(d1, Dyn, op_eq)
    both_dyn = Conj([c1, c2])

    c11 = Conj(
        [
            BinConstraintD(end, Dyn, op_neq),
            BinConstraintD(start, Dyn, op_neq),
            BinConstraintD(step, Dyn, op_neq),
        ]
    )
    c22 = BinConstraintD(d1, Dyn, op_neq)
    both_numbers = Conj([c11, c22, size_constraint])

    return [
        BinConstraintT(arange, TensorType([d1]), op_eq),
        Disj([both_dyn, both_numbers]),
    ], counter


def gen_broadcasting_constraints(
    e1: TVar, e2: TVar, symbols: _SymbolDict, counter: int, output_var: TVar
) -> tuple[list[Constraint], int]:
    # additional vars that don't correspond to expressions
    e11, counter = gen_tvar(counter)
    e22, counter = gen_tvar(counter)

    # generate constraints
    c1 = TGreatestUpperBound(output_var, e11, e22)
    c2 = ApplyBroadcasting(e11, e22, e1, e2)
    c3 = BinConstraintT(e11, e22, op_consistency)
    return [c1, c2, c3], counter


@register_inference_rule(operator.mul)
@register_inference_rule(torch.ne)
@register_inference_rule("ne")
@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def broadcasting_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:  # pyrefly: ignore[bad-return]
    op_code = None
    if n.target is operator.add or n.target is torch.add:
        op_code = op_add
    elif n.target is operator.mul:
        op_code = op_mul

    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(symbols[n.args[0]], TVar) and isinstance(
            symbols[n.args[1]], TVar
        ):
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e1 = symbols[n.args[0]]
            e2 = symbols[n.args[1]]

            return gen_broadcasting_constraints(
                e1,  # pyrefly: ignore[bad-argument-type]
                e2,  # pyrefly: ignore[bad-argument-type]
                symbols,
                counter,
                my_output,
            )
        else:
            raise NotImplementedError("Method not yet implemented")

    elif isinstance(n.args[0], Node) and isinstance(n.args[1], (int, float)):
        if isinstance(symbols[n.args[0]], TVar):
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e1 = symbols[n.args[0]]
            return [BinConstraintT(my_output, e1, op_eq)], counter
        elif isinstance(symbols[n.args[0]], DVar):
            my_output, counter = gen_dvar(counter)  # type: ignore[arg-type,assignment]
            symbols[n] = my_output
            e1 = symbols[n.args[0]]

            # we will propagate the runtime value here since this is regular addition
            c = Conj(
                [
                    BinConstraintD(
                        my_output, BinConstraintD(e1, n.args[1], op_code), op_eq
                    ),
                    BinConstraintD(0, my_output, op_leq),
                ]
            )
            return [c], counter

    elif isinstance(n.args[1], Node) and isinstance(n.args[0], (int, float)):
        if isinstance(symbols[n.args[1]], TVar):
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e2 = symbols[n.args[1]]
            return [BinConstraintT(my_output, e2, op_eq)], counter
        elif isinstance(symbols[n.args[1]], DVar):
            my_output, counter = gen_dvar(counter)  # type: ignore[arg-type,assignment]
            symbols[n] = my_output
            e2 = symbols[n.args[1]]

            # we will propagate the runtime value here since this is regular addition
            c = Conj(
                [
                    BinConstraintD(
                        my_output, BinConstraintD(e2, n.args[0], op_code), op_eq
                    ),
                    BinConstraintD(0, my_output, op_leq),
                ]
            )
            return [c], counter

        else:
            raise NotImplementedError("Method not yet implemented")

    else:
        # TODO generate add constraints for scalar addition
        raise NotImplementedError("Addition not yet implemented")


@register_inference_rule(torch.flatten)
def flatten_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # generate the new variable
    flattened, counter = gen_tvar(counter)
    symbols[n] = flattened

    input = symbols[n.args[0]]

    # set the default start and end dims
    start_dim = 1
    end_dim = -1

    if len(n.args) > 1:
        if not isinstance(n.args[1], int):
            raise AssertionError(f"Expected int, got {type(n.args[1])}")
        start_dim = n.args[1]

    if len(n.args) > 2:
        if not isinstance(n.args[2], int):
            raise AssertionError(f"Expected int, got {type(n.args[2])}")
        end_dim = n.args[2]

    c1 = BinConstraintT(input, Dyn, op_eq)
    c2 = BinConstraintT(flattened, Dyn, op_eq)
    both_dyn = Conj([c1, c2])

    const = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        c, counter = generate_flatten_constraints(
            start_dim,
            end_dim,
            input,  # pyrefly: ignore[bad-argument-type]
            flattened,
            i,
            counter,
        )
        const.append(c)

    return [Disj([both_dyn, *const])], counter


@register_inference_rule(torch.nn.functional.layer_norm)
def layer_norm_functional(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    """
    We generate the constraint: input = output
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    return gen_layer_norm_constraints(
        n,
        n.args[1],  # pyrefly: ignore[bad-argument-type]
        symbols,
        counter,
    )


@register_inference_rule(torch.nn.LayerNorm)
def layer_norm_inference_rule(
    n: Node,
    module_instance: torch.nn.LayerNorm,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    """
    Input and output shapes should be equal.
    Input should be consistent with the normalized_shape
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    return gen_layer_norm_constraints(
        n, module_instance.normalized_shape, symbols, counter
    )


def gen_layer_norm_constraints(
    n: Node, normalized_shape: Sequence[int], symbols: _SymbolDict, counter: int
) -> tuple[list[Constraint], int]:
    output, counter = gen_tvar(counter)
    symbols[n] = output
    input = symbols[n.args[0]]  # pyrefly: ignore[bad-index]

    input_dyn = BinConstraintT(input, Dyn, op_eq)
    output_dyn = BinConstraintT(output, Dyn, op_eq)

    c1 = Conj([input_dyn, output_dyn])

    c2 = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs, counter = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(new_dims_rhs)

        c_tensor_i = Conj(
            [
                BinConstraintT(input, TensorType(new_dims_rhs), op_eq),
                BinConstraintT(output, TensorType(new_dims_rhs), op_eq),
            ]
            + add_layer_norm_constraints(new_dims_rhs, list(normalized_shape))
            + nat_constraints
        )
        c2.append(c_tensor_i)
    return [Disj([c1, Disj(c2)])], counter


@register_inference_rule(torch.nn.Dropout)
@register_inference_rule(torch.nn.ReLU)
def relu_inference_rule(
    n: Node,
    module_instance: torch.nn.Module,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    """
    Input and output shapes should be equal.
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    output, counter = gen_tvar(counter)
    symbols[n] = output
    input = symbols[n.args[0]]
    if not isinstance(input, TVar):
        raise AssertionError(f"Expected TVar, got {type(input)}")
    return [BinConstraintT(input, output, op_eq)], counter


@register_inference_rule(torch.nn.Linear)
def linear_inference_rule(
    n: Node,
    module_instance: torch.nn.Linear,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    """
    Input and output sizes should be the same except for the last dimension
    If the input is Dyn, then so should the output
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    return linear_constraints(
        n, module_instance.in_features, module_instance.out_features, symbols, counter
    )


@register_inference_rule("dim")
def torch_dim_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    my_dim, counter = gen_dvar(counter)
    symbols[n] = my_dim
    input = symbols[n.args[0]]

    input_dyn = BinConstraintT(input, Dyn, op_eq)
    output_dyn = BinConstraintD(my_dim, Dyn, op_eq)

    c1 = []

    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)

        c_tensor_i = Conj(
            [
                BinConstraintT(input, TensorType(new_dims_rhs_1), op_eq),
                BinConstraintD(my_dim, i, op_eq),
            ]
        )
        c1.append(c_tensor_i)

    return [Disj([Conj([input_dyn, output_dyn]), Disj(c1)])], counter


@register_inference_rule(torch._C._nn.linear)
def torch_linear_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    weight_dims, counter = gen_tensor_dims(2, counter)
    equality_constraint = BinConstraintT(
        symbols[n.args[1]],  # pyrefly: ignore[bad-index]
        TensorType(weight_dims),
        op_eq,
    )
    constraints, counter = linear_constraints(
        n, weight_dims[1], weight_dims[0], symbols, counter
    )
    return [equality_constraint] + constraints, counter


def linear_constraints(
    n: Node,
    in_features: int | DVar,
    out_features: int | DVar,
    symbols: _SymbolDict,
    counter: int,
) -> tuple[list[Constraint], int]:
    linear_output, counter = gen_tvar(counter)
    symbols[n] = linear_output
    linear_input = symbols[n.args[0]]  # pyrefly: ignore[bad-index]

    input_dyn = BinConstraintT(linear_input, Dyn, op_eq)
    output_dyn = BinConstraintT(linear_output, Dyn, op_eq)

    c1 = Conj([input_dyn, output_dyn])

    c2 = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)
        new_dims_rhs_2, counter = gen_tensor_dims(i, counter)

        nat_constraints = gen_nat_constraints(new_dims_rhs_1 + new_dims_rhs_2)

        c_tensor_i = Conj(
            [
                BinConstraintT(linear_input, TensorType(new_dims_rhs_1), op_eq),
                BinConstraintT(linear_output, TensorType(new_dims_rhs_2), op_eq),
            ]
            + add_linear_constraints(
                new_dims_rhs_1, new_dims_rhs_2, in_features, out_features
            )
            + nat_constraints
        )
        c2.append(c_tensor_i)
    return [Disj([c1, Disj(c2)])], counter


def add_layer_norm_constraints(
    input_dim: list[DVar], normalized_dim: list[int]
) -> list[Constraint]:
    """
    The constraints say that the type has the form: ``[*, 1024, 1024]``
    while the normalized_dim have the form ``[1024, 1024]``.

    Args:
        input_dim: Input shape of layer norm
        normalized_dim: normalized_dim parameter of the module instance
    """

    # in this case we return false since there's a pattern mismatch
    if len(normalized_dim) > len(input_dim):
        return [F()]

    else:
        constraints: list[Constraint] = []
        for i, n in zip(reversed(input_dim), reversed(normalized_dim)):
            constraints.append(BinConstraintD(i, n, op_consistency))
        return constraints


def add_linear_constraints(
    dims1: list[DVar],
    dims2: list[DVar],
    in_features: int | DVar,
    out_features: int | DVar,
) -> list[Constraint]:
    if len(dims1) != len(dims2):
        raise AssertionError(f"Expected same length, got {len(dims1)} vs {len(dims2)}")
    constraints: list[Constraint] = []
    for i in range(len(dims1)):
        if i == len(dims1) - 1:
            constraints.append(BinConstraintD(dims1[i], in_features, op_consistency))
            constraints.append(BinConstraintD(dims2[i], out_features, op_eq))
        else:
            constraints.append(BinConstraintD(dims1[i], dims2[i], op_eq))

    return constraints


@register_inference_rule(torch.reshape)
def reshape_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # generate the new variable
    my_reshape, counter = gen_tvar(counter)
    symbols[n] = my_reshape

    src_var = symbols[n.args[0]]
    t2 = n.args[1]
    t2_type = TensorType([Dyn if elem == -1 else elem for elem in t2])  # type: ignore[union-attr]
    c1 = BinConstraintT(my_reshape, t2_type, op_eq)  # type: ignore[union-attr]
    c2 = CanReshape(src_var, t2_type)  # pyrefly: ignore[bad-argument-type]

    return [c1, c2], counter


@register_inference_rule(BatchNorm2d)
def batchnorm_inference_rule(
    n: Node,
    module_instance: BatchNorm2d,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # generate the new variable
    batchnorm_output, counter = gen_tvar(counter)
    symbols[n] = batchnorm_output
    batchnorm_input = symbols[n.args[0]]

    # dim vars
    d1, counter = gen_dvar(counter)
    d2, counter = gen_dvar(counter)
    d3, counter = gen_dvar(counter)
    d4, counter = gen_dvar(counter)

    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    c1 = BinConstraintT(batchnorm_input, TensorType([d1, d2, d3, d4]), op_matching)
    c2 = BinConstraintT(batchnorm_input, batchnorm_output, op_eq)
    return [c1, c2, *nat_constraints], counter


@register_inference_rule(torch.nn.AdaptiveAvgPool2d)
def adaptive_inference_rule(
    n: Node,
    module_instance: torch.nn.AdaptiveAvgPool2d,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    avg_pool, counter = gen_tvar(counter)

    symbols[n] = avg_pool
    input_var = symbols[n.args[0]]

    # dim vars
    d1, counter = gen_dvar(counter)
    d2, counter = gen_dvar(counter)
    d3, counter = gen_dvar(counter)
    d4, counter = gen_dvar(counter)
    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])
    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)
    c2 = BinConstraintT(
        avg_pool,
        TensorType(
            [
                d1,
                d2,
                module_instance.output_size[  # pyrefly: ignore[bad-index, unsupported-operation]
                    0
                ],
                module_instance.output_size[  # pyrefly: ignore[bad-index, unsupported-operation]
                    1
                ],
            ]
        ),
        op_eq,
    )

    return [c1, c2, *nat_constraints], counter


@register_inference_rule(Conv2d)
def conv2d_inference_rule(
    n: Node,
    module_instance: Conv2d,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    my_conv, counter = gen_tvar(counter)
    symbols[n] = my_conv
    input_var = symbols[n.args[0]]

    # dim vars
    [d1, d2, d3, d4], counter = gen_tensor_dims(MAX_TENSOR_RANK, counter)

    # c1 = Matching(input_var, TensorType([d1, d2, d3, d4]))
    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)

    # c2 = DConsistency(module_instance.in_channels, d2)
    c2 = BinConstraintD(module_instance.in_channels, d2, op_consistency)

    c3 = CalcConv(
        my_conv,
        input_var,  # pyrefly: ignore[bad-argument-type]
        module_instance.out_channels,
        module_instance.kernel_size,  # pyrefly: ignore[bad-argument-type]
        module_instance.padding,  # pyrefly: ignore[bad-argument-type]
        module_instance.stride,  # pyrefly: ignore[bad-argument-type]
        module_instance.dilation,  # pyrefly: ignore[bad-argument-type]
        [d1, d2, d3, d4],
    )

    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    return [c1, c2, c3, *nat_constraints], counter


@register_inference_rule(torch.nn.MaxPool2d)
def maxpool_inference_rule(
    n: Node,
    module_instance: torch.nn.MaxPool2d,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    maxpool, counter = gen_tvar(counter)
    symbols[n] = maxpool
    input_var = symbols[n.args[0]]

    # dim vars
    [d1, d2, d3, d4], counter = gen_tensor_dims(MAX_TENSOR_RANK, counter)

    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)

    c2 = CalcMaxPool(
        maxpool,
        input_var,  # pyrefly: ignore[bad-argument-type]
        module_instance.kernel_size,
        module_instance.padding,
        module_instance.stride,
        module_instance.dilation,
        [d1, d2, d3, d4],
    )

    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    return [c1, c2, *nat_constraints], counter


class ConstraintGenerator:
    def __init__(self, traced: torch.nn.Module, graph: Graph | None = None) -> None:
        self.traced = traced  # traced or tracer.root
        self.traced_params = dict(self.traced.named_parameters())
        self.constraints = []
        self.symbol_dict = {}
        self.graph = traced.graph if hasattr(traced, "graph") else graph

    def generate_constraints(self, counter: int = 0) -> tuple[Conj, int]:
        """
        Iterate through every node and generate constraints
        Effect: self.constraints will be populated with the final constraints
        """
        graph = self.graph

        all_constraints: list[Constraint] = []

        # pyrefly: ignore [missing-attribute]
        for n in graph.nodes:
            (constraints, counter) = self.generate_constraints_node(n, counter)
            all_constraints += constraints

        return Conj(all_constraints), counter

    def generate_constraints_node(
        self, n: Node, counter: int
    ) -> tuple[list[Constraint], int]:
        """
        Generate constraints the given node:
        Currently supported operations:
        - Reshape
        - Add
        - conv2d
        """

        if n.op == "placeholder":
            x, counter = gen_tvar(counter)
            self.symbol_dict[n] = x

            my_type = n.type

            if n.type != Dyn and (not isinstance(n.type, TensorType)):
                if n.type == torch.nn.parameter.Parameter:
                    # since we have a parameter, the shape must be static
                    if "example_value" not in n.meta:
                        raise AssertionError("example_value not in n.meta")
                    my_type = TensorType(n.meta["example_value"].size())
                else:
                    my_type = Dyn

            c1 = BinConstraintT(my_type, x, op_precision)
            c2 = BinConstraintT(x, MAX_TENSOR_RANK, op_leq)
            return [c1, c2], counter

        elif n.op == "call_function":
            if n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](
                    n, self.symbol_dict, self.constraints, counter
                )
            else:
                raise RuntimeError(
                    f"No inference rule registered for target {n.target}!"
                )

        elif n.op == "call_module":
            module_instance = self.traced.get_submodule(
                n.target  # pyrefly: ignore[bad-argument-type]
            )
            if type(module_instance) in _INFERENCE_RULES:
                return _INFERENCE_RULES[type(module_instance)](
                    n, module_instance, self.symbol_dict, self.constraints, counter
                )
            else:
                raise RuntimeError(
                    f"No inference rule registered for class {type(module_instance)}!"
                )

        elif n.op == "call_method":
            if n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](
                    n, self.symbol_dict, self.constraints, counter
                )
            else:
                raise RuntimeError(
                    f"No inference rule registered for target {n.target}!"
                )

        elif n.op == "get_attr":
            t = self.traced_params.get(  # pyrefly: ignore[no-matching-overload]
                n.target, None
            )

            if isinstance(t, torch.Tensor):
                if len(t.shape) > 0:
                    res = list(t.shape)
                    attr_type = TensorType(res)
                    output, counter = gen_tvar(counter)
                    self.symbol_dict[n] = output
                    return [BinConstraintT(output, attr_type, op_eq)], counter
                else:
                    # scalar?
                    return [], counter  # pyrefly: ignore[implicit-any]
            else:
                return [], counter  # pyrefly: ignore[implicit-any]

        elif n.op == "output":
            return [], counter  # pyrefly: ignore[implicit-any]

        else:
            raise NotImplementedError(f"Method {n.op} not yet implemented")
<|endoftext|>from torch.fx.experimental.migrate_gradual_types.constraint import (
    BinConstraintD,
    BVar,
    DVar,
    TVar,
)
from torch.fx.experimental.migrate_gradual_types.operation import op_leq


def gen_tvar(curr: int) -> tuple[TVar, int]:
    """
    Generate a tensor variable
    :param curr: The current counter
    :return: a tensor variable and the updated counter
    """
    curr += 1
    return TVar(curr), curr


def gen_dvar(curr: int) -> tuple[DVar, int]:
    """
    Generate a dimension variable
    :param curr: the current counter
    :return: a dimension variable and an updated counter
    """
    curr += 1
    return DVar(curr), curr


def gen_bvar(curr: int) -> tuple[BVar, int]:
    """
    Generate a boolean variable
    :param curr: the current counter
    :return: a boolean variable and an updated counter
    """
    curr += 1
    return BVar(curr), curr


def gen_tensor_dims(n: int, curr: int) -> tuple[list[DVar], int]:
    """
    Generate a list of tensor dimensions
    :param n:  the number of dimensions
    :param curr: the current counter
    :return: a list of dimension variables and an updated counter
    """
    dims = []
    for _ in range(n):
        dvar, curr = gen_dvar(curr)
        dims.append(dvar)
    return dims, curr


def gen_nat_constraints(list_of_dims: list[DVar]) -> list[BinConstraintD]:
    """
    Generate natural number constraints for dimensions
    """
    return [BinConstraintD(0, d, op_leq) for d in list_of_dims]
<|endoftext|>import copy
import itertools
from collections.abc import Callable, Sequence

from torch.fx.experimental.migrate_gradual_types.constraint import (
    ApplyBroadcasting,
    BinConstraintD,
    CalcConv,
    CalcMaxPool,
    CalcProduct,
    CanReshape,
    Conj,
    Constraint,
    DGreatestUpperBound,
    Disj,
    DVar,
    F,
    GetItem,
    GetItemTensor,
    IndexSelect,
    Prod,
    T,
    TGreatestUpperBound,
    Transpose,
    TVar,
)
from torch.fx.experimental.migrate_gradual_types.constraint_generator import (
    BinConstraintT,
    MAX_TENSOR_RANK,
)
from torch.fx.experimental.migrate_gradual_types.operation import (
    op_add,
    op_consistency,
    op_div,
    op_eq,
    op_leq,
    op_matching,
    op_mod,
    op_mul,
    op_neq,
    op_precision,
    op_sub,
)
from torch.fx.experimental.migrate_gradual_types.util import (
    gen_dvar,
    gen_nat_constraints,
    gen_tensor_dims,
)
from torch.fx.tensor_type import _DynType, Dyn, TensorType


__all__ = [
    "apply_padding",
    "broadcast_dim",
    "calc_last_two_dims",
    "create_equality_constraints_for_broadcasting",
    "gen_all_reshape_possibilities",
    "gen_broadcasting_constraints",
    "gen_consistency_constraints",
    "gen_greatest_upper_bound",
    "gen_lists_of_dims",
    "generate_all_broadcasting_possibilities_no_padding",
    "generate_all_int_dyn_dim_possibilities",
    "generate_binconstraint_d",
    "generate_binconstraint_t",
    "generate_broadcasting",
    "generate_calc_conv",
    "generate_calc_maxpool",
    "generate_calc_product",
    "generate_conj",
    "generate_d_gub",
    "generate_disj",
    "generate_gub",
    "generate_reshape",
    "is_dim_div_by_target",
    "is_target_div_by_dim",
    "no_broadcast_dim_with_index",
    "register_transformation_rule",
    "transform_constraint",
    "transform_get_item",
    "transform_get_item_tensor",
    "transform_index_select",
    "transform_transpose",
    "valid_index",
    "valid_index_tensor",
]


_TransformFn = Callable[[Constraint, int], tuple[Constraint, int]]
_TRANSFORMATION_RULES: dict[type, _TransformFn] = {}


def register_transformation_rule(
    call_target: type[Constraint],
) -> Callable[[_TransformFn], _TransformFn]:
    def register(fn: _TransformFn) -> _TransformFn:
        if call_target in _TRANSFORMATION_RULES:
            raise RuntimeError(
                f"Transformation rule already registered for {call_target}!"
            )
        _TRANSFORMATION_RULES[call_target] = fn
        return fn

    return register


def valid_index(index: int, dims: list[DVar]) -> Constraint:
    """
    Given a list of dimensions, checks if an index is valid in the list
    """
    try:
        dims[index]
        return T()
    except IndexError:
        return F()


@register_transformation_rule(Transpose)
def transform_transpose(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    """
    Similar to a sequence of two index-selects
    """
    if not isinstance(constraint, Transpose):
        raise TypeError(type(constraint))
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    is_valid_index1 = valid_index(constraint.index1, dims)
    is_valid_index2 = valid_index(constraint.index2, dims)
    new_dims = copy.deepcopy(dims)
    nat_constraints = gen_nat_constraints(dims)

    if is_valid_index1 == T() and is_valid_index2 == T():
        new_dims[constraint.index1] = dims[constraint.index2]
        new_dims[constraint.index2] = dims[constraint.index1]

    transformed_constraint = Conj(
        [
            BinConstraintT(constraint.input_var, TensorType(dims), op_eq),
            *nat_constraints,
            is_valid_index1,
            is_valid_index2,
            BinConstraintT(constraint.output, TensorType(new_dims), op_eq),
        ]
    )
    return transformed_constraint, counter


@register_transformation_rule(IndexSelect)
def transform_index_select(
    constraint: Constraint, counter: int
) -> tuple[Constraint, int]:
    """
    The constraints consider the given tensor size, checks if the index is valid
    and if so, generates a constraint for replacing the input dimension
    with the required dimension
    """
    if not isinstance(constraint, IndexSelect):
        raise TypeError(type(constraint))
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    is_valid_index = valid_index(constraint.index, dims)
    nat_constraints = gen_nat_constraints(dims)

    # if the index is valid then replace the input dimension with the new dimension
    # otherwise the dimension will not be replaced and the clause will contain False
    new_dims = copy.deepcopy(dims)
    if is_valid_index == T():
        new_dims[constraint.index] = constraint.dim_replace  # type: ignore[unsupported-operation]

    transformed_constraint = Conj(
        [
            BinConstraintT(constraint.input_var, TensorType(dims), op_eq),
            *nat_constraints,
            is_valid_index,
            BinConstraintT(constraint.output, TensorType(new_dims), op_eq),
        ]
    )

    # print(constraints)
    return transformed_constraint, counter


@register_transformation_rule(GetItem)
def transform_get_item(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    """
    generate an equality of the form:
    t = [a1, ..., an]
    then generate constraints that check if the given index is valid
    given this particular tensor size.
    If the index is valid, generate a constraint to get the item
    Note that we already handled the Dyn input case in the previous
    step.
    Args:
        constraint: GetItem which assumes we are getting an item from a tensor (not Dyn)
        counter: variable tracking
    Returns: simplified constraints for GetItem

    """
    if not isinstance(constraint, GetItem):
        raise TypeError(type(constraint))
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    nat_constraints = gen_nat_constraints(dims)

    is_valid_index = valid_index(constraint.index, dims)

    all_constraints = [
        BinConstraintT(constraint.input_var, TensorType(dims), op_eq),
        *nat_constraints,
        is_valid_index,
    ]

    # if the index is valid, we generate a constraint for getting an item
    # otherwise this clause will have been UNSAT due to the wrong index
    if is_valid_index == T():
        all_constraints.append(
            BinConstraintD(constraint.res, dims[constraint.index], op_eq)
        )

    return Conj(all_constraints), counter


def valid_index_tensor(index: tuple[None | slice, ...], dims: list[DVar]) -> Constraint:
    """
    if the slice instances exceed the length of the dimensions
    then this is a type error so we return False
    """
    slice_count = 0
    for s in index:
        if isinstance(s, slice):
            slice_count += 1
    if slice_count > len(dims):
        return F()
    else:
        return T()


@register_transformation_rule(GetItemTensor)
def transform_get_item_tensor(
    constraint: Constraint, counter: int
) -> tuple[Constraint, int]:
    """
    When the index is a tuple, then the output will be a tensor
    TODO: we have to check if this is the case for all HF models

    The cases we are covering here are a tuple with one of:
     - slice with default argument
     - None

     None appends 1 to the input tensor dimensions
     so each occurrence of 'None' increases the rank by 1

     slice with default arguments does not change the rank
    """
    if not isinstance(constraint, GetItemTensor):
        raise TypeError(type(constraint))
    if not isinstance(constraint.index_tuple, tuple):
        raise AssertionError(
            f"Expected tuple for index_tuple, got {type(constraint.index_tuple)}"
        )

    # generate a result tensor of the expected size
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    nat_constraints = gen_nat_constraints(dims)

    # generate a place-holder list of the right rank
    # where "slice" does not contribute to the rank and "None" does
    none_c = constraint.index_tuple.count(None)
    # list invariance: [None] * n types as list[None], but elements are reassigned to int/DVar
    resulting_tensor_dims: list[int | DVar | None] = [None] * (none_c + len(dims))  # type: ignore[assignment]

    dim_index = 0
    for i in range(len(constraint.index_tuple)):
        # append 1 to the right location of the resulting tensor
        if constraint.index_tuple[i] is None:
            resulting_tensor_dims[i] = 1

        elif constraint.index_tuple[i] == slice(None, None, None):
            pass

        else:
            raise NotImplementedError("Method not yet implemented")

    # append the remaining dimensions to the right location
    dim_index = 0
    for i in range(len(resulting_tensor_dims)):
        if resulting_tensor_dims[i] is None:
            resulting_tensor_dims[i] = dims[dim_index]
            dim_index += 1

    # check if the index is valid
    is_valid_index = valid_index_tensor(constraint.index_tuple, dims)

    # check if the resulting tensor is within bounds
    if len(resulting_tensor_dims) > 4:
        return F(), counter

    else:
        constraints = [
            BinConstraintT(constraint.input_var, TensorType(dims), op_eq),
            BinConstraintT(constraint.res, TensorType(resulting_tensor_dims), op_eq),
            *nat_constraints,
            is_valid_index,
        ]
        return Conj(constraints), counter


@register_transformation_rule(BinConstraintT)
def generate_binconstraint_t(
    constraint: Constraint, counter: int
) -> tuple[Constraint, int]:
    """
    Transform binary constraints for tensors
    """
    if not isinstance(constraint, BinConstraintT):
        raise TypeError(type(constraint))

    # precision constraints
    if constraint.op == op_precision:
        if constraint.lhs == Dyn:
            return T(), counter
        elif isinstance(constraint.lhs, TensorType):
            is_fully_static = all(d != Dyn for d in constraint.lhs.dims)
            if is_fully_static:
                return BinConstraintT(constraint.lhs, constraint.rhs, op_eq), counter
            else:
                new_dims = []

                for _ in range(len(constraint.lhs.dims)):
                    dim, counter = gen_dvar(counter)
                    new_dims.append(dim)

                new_dim_constraints = (
                    [
                        BinConstraintD(old_dim, new_dim, op_precision)
                        for new_dim, old_dim in zip(new_dims, constraint.lhs.dims)
                    ]
                    + [BinConstraintT(constraint.rhs, TensorType(new_dims), op_eq)]
                    + [BinConstraintD(1, new_dim, op_leq) for new_dim in new_dims]
                )
                return Conj(new_dim_constraints), counter
        else:
            return constraint, counter

    # matching
    elif constraint.op == op_matching:
        if not isinstance(constraint.rhs, TensorType):
            raise AssertionError(f"Expected TensorType, got {type(constraint.rhs)}")
        d1 = constraint.rhs.dims[0]
        d2 = constraint.rhs.dims[1]
        d3 = constraint.rhs.dims[2]
        d4 = constraint.rhs.dims[3]

        conj = [
            BinConstraintT(constraint.lhs, Dyn, op_eq),
            BinConstraintD(d1, Dyn, op_eq),
            BinConstraintD(d2, Dyn, op_eq),
            BinConstraintD(d3, Dyn, op_eq),
            BinConstraintD(d4, Dyn, op_eq),
        ]
        return (
            Disj(
                [
                    Conj(conj),
                    BinConstraintT(constraint.lhs, TensorType([d1, d2, d3, d4]), op_eq),
                ]
            ),
            counter,
        )

    elif constraint.op == op_consistency:
        c_dyn = Disj(
            [
                BinConstraintT(constraint.lhs, Dyn, op_eq),
                BinConstraintT(constraint.rhs, Dyn, op_eq),
            ]
        )
        (
            (
                c_tensor_1,
                c_tensor_2,
                c_tensor_3,
                c_tensor_4,
            ),
            counter,
        ) = gen_consistency_constraints(constraint, counter)

        return Disj([c_dyn, c_tensor_1, c_tensor_2, c_tensor_3, c_tensor_4]), counter

    elif constraint.op == op_leq:
        if not isinstance(constraint.rhs, int):
            raise AssertionError(f"Expected int, got {type(constraint.rhs)}")
        disj = [BinConstraintT(constraint.lhs, Dyn, op_eq)]
        for i in range(1, constraint.rhs + 1):
            dims = []
            for _ in range(1, i + 1):
                dim_var, counter = gen_dvar(counter)
                dims.append(dim_var)
            disj.append(BinConstraintT(constraint.lhs, TensorType(dims), op_eq))
        return Disj(disj), counter
    else:
        return constraint, counter


@register_transformation_rule(BinConstraintD)
def generate_binconstraint_d(
    constraint: Constraint, counter: int
) -> tuple[Constraint, int]:
    """
    Transform binary constraints for dimensions
    """
    if not isinstance(constraint, BinConstraintD):
        raise TypeError(type(constraint))
    if constraint.op == op_precision:
        if isinstance(constraint.lhs, int):
            return BinConstraintD(constraint.lhs, constraint.rhs, op_eq), counter
        elif constraint.lhs == Dyn:
            return T(), counter
        else:
            return constraint, counter

    elif constraint.op == op_consistency:
        return (
            Disj(
                [
                    BinConstraintD(constraint.lhs, constraint.rhs, op_eq),
                    BinConstraintD(constraint.rhs, Dyn, op_eq),
                    BinConstraintD(constraint.lhs, Dyn, op_eq),
                ]
            ),
            counter,
        )

    else:
        return constraint, counter


@register_transformation_rule(Conj)
def generate_conj(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    """
    Transform conjunctions
    """
    if not isinstance(constraint, Conj):
        raise TypeError(type(constraint))
    new = []
    for c in constraint.conjucts:
        new_c, counter = transform_constraint(c, counter)
        new.append(new_c)
    return Conj(new), counter


@register_transformation_rule(Disj)
def generate_disj(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    """
    Transform disjunctions
    """
    if not isinstance(constraint, Disj):
        raise TypeError(type(constraint))
    new = []
    for c in constraint.disjuncts:
        new_c, counter = transform_constraint(c, counter)
        new.append(new_c)
    return Disj(new), counter


@register_transformation_rule(TGreatestUpperBound)
def generate_gub(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    """
    Transform greatest upper bound for tensors. Results in equality and Greatest Upper Bound
    on dimensions
    """
    if not isinstance(constraint, TGreatestUpperBound):
        raise TypeError(type(constraint))
    c1 = Conj(
        [
            Disj(
                [
                    BinConstraintT(constraint.rhs1, Dyn, op_eq),
                    BinConstraintT(constraint.rhs2, Dyn, op_eq),
                ]
            ),
            BinConstraintT(constraint.res, Dyn, op_eq),
        ]
    )

    [c2, c3, c4, c5], counter = gen_greatest_upper_bound(constraint, counter)

    return Disj([c1, c2, c3, c4, c5]), counter


@register_transformation_rule(DGreatestUpperBound)
def generate_d_gub(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    """
    Transform greatest upper bound for dimensions into equality constraints
    """
    if not isinstance(constraint, DGreatestUpperBound):
        raise TypeError(type(constraint))
    c1 = Conj(
        [
            BinConstraintD(constraint.rhs1, Dyn, op_eq),
            BinConstraintD(constraint.res, constraint.rhs2, op_eq),
        ]
    )
    c2 = Conj(
        [
            BinConstraintD(constraint.rhs2, Dyn, op_eq),
            BinConstraintD(constraint.res, constraint.rhs1, op_eq),
        ]
    )
    c3 = Conj(
        [
            BinConstraintD(constraint.rhs2, constraint.rhs1, op_eq),
            BinConstraintD(constraint.res, constraint.rhs1, op_eq),
        ]
    )
    return Disj([c1, c2, c3]), counter


@register_transformation_rule(CalcConv)
def generate_calc_conv(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    if not isinstance(constraint, CalcConv):
        raise TypeError(type(constraint))
    d, counter = gen_tensor_dims(4, counter)
    conv_result = TensorType([d[0], d[1], d[2], d[3]])

    # the convolution result is a tensor of size 4
    c1 = BinConstraintT(constraint.conv_result, conv_result, op_eq)

    # the second dimension of the output is equal to the output channels
    c2 = Conj(
        [
            BinConstraintD(d[1], constraint.c_out, op_eq),
            BinConstraintD(d[1], Dyn, op_neq),
        ]
    )

    # the input corresponds to the output in the first dimension of the convolution
    c3 = BinConstraintD(constraint.matching_constraint[0], d[0], op_eq)

    c4, c5 = calc_last_two_dims(constraint, d)

    leq_constraints = Conj(
        [
            BinConstraintD(0, d[0], op_leq),
            BinConstraintD(0, d[1], op_leq),
            BinConstraintD(0, d[2], op_leq),
            BinConstraintD(0, d[3], op_leq),
        ]
    )

    return Conj([c1, c2, c3, c4, c5, leq_constraints]), counter


@register_transformation_rule(CalcMaxPool)
def generate_calc_maxpool(
    constraint: Constraint, counter: int
) -> tuple[Constraint, int]:
    """
    Transform maxpool constraints
    """
    if not isinstance(constraint, CalcMaxPool):
        raise TypeError(type(constraint))
    d, counter = gen_tensor_dims(4, counter)
    maxpool_result = TensorType([d[0], d[1], d[2], d[3]])

    # the maxpool result is a tensor of size 4
    c1 = BinConstraintT(constraint.maxpool_result, maxpool_result, op_eq)

    # the input corresponds to the output in the first and second dimension of maxpool
    c2 = BinConstraintD(constraint.matching_constraint[1], d[1], op_eq)
    c3 = BinConstraintD(constraint.matching_constraint[0], d[0], op_eq)
    c4, c5 = calc_last_two_dims(constraint, d)

    leq_constraints = Conj(
        [
            BinConstraintD(0, d[0], op_leq),
            BinConstraintD(0, d[1], op_leq),
            BinConstraintD(0, d[2], op_leq),
            BinConstraintD(0, d[3], op_leq),
        ]
    )

    return Conj([c1, c2, c3, c4, c5, leq_constraints]), counter


@register_transformation_rule(CalcProduct)
def generate_calc_product(
    constraint: Constraint, counter: int
) -> tuple[Constraint, int]:
    """
    Transform flatten constraints
    """
    if not isinstance(constraint, CalcProduct):
        raise TypeError(type(constraint))
    start = constraint.start
    end = constraint.end
    dims = constraint.dims_to_flatten
    flattened = constraint.flattened
    n = len(constraint.dims_to_flatten)

    # this will be evaluated right here
    boundary_check = 0 <= start and start < end and end <= n

    c_boundary = T() if boundary_check else F()

    lhs = dims[0:start]
    rhs = dims[end:]
    mid = dims[start:end]

    all_possibilities = generate_all_int_dyn_dim_possibilities(mid)

    all_constraints: list[Constraint] = []

    for p in all_possibilities:
        p = list(p)
        # this tells us there is a dynamic variable
        contains_dyn = not all(constraint.op == op_neq for constraint in p)
        if contains_dyn:
            mid_var = [Dyn]
            total_constraints = lhs + mid_var + rhs
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                all_constraints.append(
                    Conj(
                        [
                            BinConstraintT(
                                flattened, TensorType(lhs + mid_var + rhs), op_eq
                            )
                        ]
                        + p
                    )
                )
        else:
            new_var, counter = gen_dvar(counter)
            mid_eq_prod = Conj(
                [
                    BinConstraintD(new_var, Prod(mid), op_eq),
                    BinConstraintD(new_var, Dyn, op_neq),
                ]
            )
            mid_var = [new_var]
            total_constraints = lhs + mid_var + rhs
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                all_constraints.append(
                    Conj(
                        [
                            BinConstraintT(
                                flattened, TensorType(lhs + mid_var + rhs), op_eq
                            ),
                            mid_eq_prod,
                        ]
                        + p
                    )
                )

    return Conj([Disj(all_constraints), c_boundary]), counter


@register_transformation_rule(CanReshape)
def generate_reshape(constraint: Constraint, counter: int) -> tuple[Constraint, int]:
    """
    Transform reshape constraints
    """
    if not isinstance(constraint, CanReshape):
        raise TypeError(type(constraint))
    d, counter = gen_tensor_dims(4, counter)

    d1 = d[0]
    d2 = d[1]
    d3 = d[2]
    d4 = d[3]

    target = constraint.target.dims

    is_fully_static = all(d != Dyn for d in target)

    # dynamic tensor
    c1_dyn = BinConstraintT(constraint.src, Dyn, op_eq)
    c2_tensor1 = BinConstraintT(constraint.src, TensorType([d1]), op_eq)
    c2_tensor2 = BinConstraintT(constraint.src, TensorType([d1, d2]), op_eq)
    c2_tensor3 = BinConstraintT(constraint.src, TensorType([d1, d2, d3]), op_eq)
    c2_tensor4 = BinConstraintT(constraint.src, TensorType([d1, d2, d3, d4]), op_eq)

    d1_eq_dyn = BinConstraintD(d1, Dyn, op_eq)
    d1_neq_dyn = BinConstraintD(d1, Dyn, op_neq)

    d2_eq_dyn = BinConstraintD(d2, Dyn, op_eq)
    d2_neq_dyn = BinConstraintD(d2, Dyn, op_neq)

    d3_eq_dyn = BinConstraintD(d3, Dyn, op_eq)
    d3_neq_dyn = BinConstraintD(d3, Dyn, op_neq)

    d4_eq_dyn = BinConstraintD(d3, Dyn, op_eq)
    d4_neq_dyn = BinConstraintD(d3, Dyn, op_neq)

    nat_d1 = BinConstraintD(0, d1, op_leq)
    nat_d2 = BinConstraintD(0, d2, op_leq)
    nat_d3 = BinConstraintD(0, d3, op_leq)
    nat_d4 = BinConstraintD(0, d4, op_leq)

    if is_fully_static:
        # size 1 tensor
        c3_tensor1 = Disj(
            [d1_eq_dyn, (Conj([d1_neq_dyn, BinConstraintD(d1, Prod(target), op_eq)]))]
        )
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])

        # size 2 tensor
        all_tensor_2 = Conj(
            [c2_tensor2, gen_all_reshape_possibilities([d1, d2], target)]
        )

        # size 3 tensor
        all_tensor_3 = Conj(
            [c2_tensor3, gen_all_reshape_possibilities([d1, d2, d3], target)]
        )

        # size 4 tensor
        all_tensor_4 = Conj(
            [c2_tensor4, gen_all_reshape_possibilities([d1, d2, d3, d4], target)]
        )

        return (
            Conj(
                [
                    Disj(
                        [c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]
                    ),
                    nat_d1,
                    nat_d2,
                    nat_d3,
                    nat_d4,
                ]
            ),
            counter,
        )

    # then there must be exactly one occurrence of dyn
    else:
        new_target = [n for n in target if n != Dyn]

        # tensor 1
        c3_tensor1 = Disj(
            [d1_eq_dyn, (Conj([d1_neq_dyn, is_dim_div_by_target(new_target, d1)]))]
        )
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])

        # tensor 2
        c21 = Disj([d1_eq_dyn, d2_eq_dyn])
        c22 = Conj(
            [d1_neq_dyn, d2_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2]))]
        )
        all_tensor_2 = Conj([c2_tensor2, Disj([c21, c22])])

        # tensor 3
        c31 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn])
        c32 = Conj(
            [
                d1_neq_dyn,
                d2_neq_dyn,
                d3_neq_dyn,
                is_dim_div_by_target(new_target, Prod([d1, d2, d3])),
            ]
        )
        all_tensor_3 = Conj([c2_tensor3, Disj([c31, c32])])

        # tensor 4
        c41 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn, d4_eq_dyn])
        c42 = Conj(
            [
                d1_neq_dyn,
                d2_neq_dyn,
                d3_neq_dyn,
                d4_neq_dyn,
                is_dim_div_by_target(new_target, Prod([d1, d2, d3, d4])),
            ]
        )
        all_tensor_4 = Conj([c2_tensor4, Disj([c41, c42])])

        return (
            Conj(
                [
                    Disj(
                        [c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]
                    ),
                    nat_d1,
                    nat_d2,
                    nat_d3,
                    nat_d4,
                ]
            ),
            counter,
        )


@register_transformation_rule(ApplyBroadcasting)
def generate_broadcasting(
    constraint: Constraint, counter: int
) -> tuple[Constraint, int]:
    """
    Transform broadcasting constraints
    """
    if not isinstance(constraint, ApplyBroadcasting):
        raise TypeError(type(constraint))
    e11, e12 = constraint.res1, constraint.res2
    e1, e2 = constraint.input1, constraint.input2

    e1_dyn = BinConstraintT(e1, Dyn, op_eq)
    e2_dyn = BinConstraintT(e2, Dyn, op_eq)

    # Introduce dimensions
    e1_equal_e11 = BinConstraintT(e1, e11, op_eq)
    e2_equal_e12 = BinConstraintT(e2, e12, op_eq)

    # dyn possibility
    e1_dyn_constraint = Conj([e1_dyn, e1_equal_e11, e2_equal_e12])
    e2_dyn_constraint = Conj([e2_dyn, e1_equal_e11, e2_equal_e12])

    # tensor possibility
    # generate dimensions to create tensors of size 1
    final_tensor_1_constraint, _, _, nat_dims_1, counter = gen_broadcasting_constraints(
        e1, e2, e11, e12, 1, counter
    )

    # generate dimensions to create tensors of size 2
    (
        final_tensor_2_constraint_no_padding,
        final_tensor_2_constraint_padding_arg1,
        final_tensor_2_constraint_padding_arg2,
        nat_dims_2,
        counter,
    ) = gen_broadcasting_constraints(e1, e2, e11, e12, 2, counter)

    # generate dimensions to create tensors of size 3
    (
        final_tensor_3_constraint_no_padding,
        final_tensor_3_constraint_padding_arg1,
        final_tensor_3_constraint_padding_arg2,
        nat_dims_3,
        counter,
    ) = gen_broadcasting_constraints(e1, e2, e11, e12, 3, counter)

    # generate dimensions to create tensors of size 4
    (
        final_tensor_4_constraint_no_padding,
        final_tensor_4_constraint_padding_arg1,
        final_tensor_4_constraint_padding_arg2,
        nat_dims_4,
        counter,
    ) = gen_broadcasting_constraints(e1, e2, e11, e12, 4, counter)

    final_result = Disj(
        [
            e1_dyn_constraint,
            e2_dyn_constraint,
            final_tensor_1_constraint,
            final_tensor_2_constraint_no_padding,
            final_tensor_2_constraint_padding_arg1,
            final_tensor_2_constraint_padding_arg2,
            final_tensor_3_constraint_no_padding,
            final_tensor_3_constraint_padding_arg1,
            final_tensor_3_constraint_padding_arg2,
            final_tensor_4_constraint_no_padding,
            final_tensor_4_constraint_padding_arg1,
            final_tensor_4_constraint_padding_arg2,
        ]
    )

    return (
        Conj([final_result, *nat_dims_1, *nat_dims_2, *nat_dims_3, *nat_dims_4]),
        counter,
    )


def transform_constraint(
    constraint: Constraint, counter: int
) -> tuple[Constraint, int]:
    """
    Transforms a constraint into a simpler constraint.
    Ex: precision and consistency are transformed to equality
    Args:
        constraint: constraint to be transformed
        counter: for variable tracking

    Returns: Constraint

    """
    if type(constraint) in _TRANSFORMATION_RULES:
        return _TRANSFORMATION_RULES[type(constraint)](constraint, counter)

    else:
        return constraint, counter


def calc_last_two_dims(
    constraint: CalcConv | CalcMaxPool, d: list[DVar]
) -> tuple[Constraint, Constraint]:
    """
    Generates constraints for the last two dimensions of a convolution or a maxpool output
    Args:
        constraint: CalcConv or CalcMaxPool
        d: The list of output dimensions

    Returns: Constraints for calculating the last two dimensions of the output

    """

    if not isinstance(constraint, (CalcConv, CalcMaxPool)):
        raise AssertionError(
            f"Expected CalcConv or CalcMaxPool, got {type(constraint)}"
        )

    b3 = constraint.matching_constraint[2]
    b4 = constraint.matching_constraint[3]

    b3_dyn = Conj([BinConstraintD(d[2], Dyn, op_eq), BinConstraintD(b3, Dyn, op_eq)])
    b4_dyn = Conj([BinConstraintD(d[3], Dyn, op_eq), BinConstraintD(b4, Dyn, op_eq)])

    d3_not_dyn = Conj(
        [BinConstraintD(d[2], Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq)]
    )
    d4_not_dyn = Conj(
        [BinConstraintD(d[3], Dyn, op_neq), BinConstraintD(b4, Dyn, op_neq)]
    )

    # transform parameters into tuples in case they are not already
    padding = (
        (constraint.padding, constraint.padding)
        if isinstance(constraint.padding, int)
        else constraint.padding
    )
    kernel = (
        (constraint.kernel, constraint.kernel)
        if isinstance(constraint.kernel, int)
        else constraint.kernel
    )
    stride = (
        (constraint.stride, constraint.stride)
        if isinstance(constraint.stride, int)
        else constraint.stride
    )
    dilation = (
        (constraint.dilation, constraint.dilation)
        if isinstance(constraint.dilation, int)
        else constraint.dilation
    )

    f1 = BinConstraintD(b3, BinConstraintD(2, padding[0], op_mul), op_add)
    f2 = BinConstraintD(dilation[0], BinConstraintD(kernel[0], 1, op_sub), op_mul)
    f3 = BinConstraintD(
        BinConstraintD(BinConstraintD(f1, f2, op_sub), 1, op_sub), stride[0], op_div
    )
    f4 = BinConstraintD(f3, 1, op_add)

    c4 = Disj([b3_dyn, Conj([d3_not_dyn, BinConstraintD(d[2], f4, op_eq)])])

    f11 = BinConstraintD(b4, BinConstraintD(2, padding[1], op_mul), op_add)
    f22 = BinConstraintD(dilation[1], BinConstraintD(kernel[1], 1, op_sub), op_mul)
    f33 = BinConstraintD(
        BinConstraintD(BinConstraintD(f11, f22, op_sub), 1, op_sub), stride[1], op_div
    )
    f44 = BinConstraintD(f33, 1, op_add)

    c5 = Disj([b4_dyn, Conj([d4_not_dyn, BinConstraintD(d[3], f44, op_eq)])])

    return c4, c5


def generate_all_int_dyn_dim_possibilities(
    my_list: list[DVar],
) -> list[tuple[BinConstraintD, ...]]:
    """
    Generate all possibilities of being equal or not equal to dyn for my_list
    Args:
        my_list: List of tensor dimensions

    Returns: A list of a list of constraints. Each list of constraints corresponds to
    one possibility about the values of the dimension variables
    """
    # generate all possibilities of being equal or not equal to dyn for my_list
    eq_possibilities = [
        BinConstraintD(my_list[i], Dyn, op_eq) for i in range(len(my_list))
    ]
    neq_possibilities = [
        BinConstraintD(my_list[i], Dyn, op_neq) for i in range(len(my_list))
    ]

    d_possibilities = [list(i) for i in zip(eq_possibilities, neq_possibilities)]
    all_possibilities = list(itertools.product(*d_possibilities))
    return all_possibilities


def is_target_div_by_dim(
    target: Sequence[DVar | int | _DynType], dim: DVar | Prod
) -> BinConstraintD:
    """
    Generate constraints to check if the target dimensions are divisible by the input dimensions
    Args:
        target: Target dimensions
        dim: Input dimensions

    Returns: Constraints to check divisibility

    """
    return BinConstraintD(BinConstraintD(Prod(target), dim, op_mod), 0, op_eq)


def is_dim_div_by_target(
    target: Sequence[DVar | int | _DynType], dim: DVar | Prod
) -> BinConstraintD:
    """
    Generate constraints to check if the input dimensions is divisible by the target dimensions
    Args:
        target: Target dimensions
        dim:  Input dimensions

    Returns: Constraints to check divisibility

    """
    return BinConstraintD(BinConstraintD(dim, Prod(target), op_mod), 0, op_eq)


def gen_all_reshape_possibilities(
    list_of_dims: list[DVar], target: Sequence[DVar | int | _DynType]
) -> Constraint:
    """
    Consider all possibilities what the input dimensions could be (number or dynamic)
    Then generate the appropriate constraints using multiplication or mod depending on the possibility
    The possibilities we consider here are the cross product of being equal to dyn or not equal to dyn
    for the input. Target is fixed because at most one dimension could be dyn.
    We have different cases for this.

    Args:
        list_of_dims: The input list of dimensions
        target: The tensor we want to reshape to

    Returns: A disjunction of transformed reshape constraints

    """
    all_possibilities = generate_all_int_dyn_dim_possibilities(list_of_dims)

    all_constraints = []

    for p in all_possibilities:
        to_multiply: list[DVar] = []

        p = list(p)

        for constraint in p:
            if not isinstance(constraint, BinConstraintD):
                raise AssertionError(f"Expected BinConstraintD, got {type(constraint)}")
            if constraint.op == op_neq:
                to_multiply.append(constraint.lhs)  # type: ignore[arg-type]

        if not to_multiply:
            all_constraints.append(Conj(p))

        elif len(to_multiply) < len(list_of_dims):
            all_constraints.append(
                Conj(p + [is_target_div_by_dim(target, Prod(to_multiply))])
            )
        else:
            all_constraints.append(
                Conj(p + [BinConstraintD(Prod(list_of_dims), Prod(target), op_eq)])
            )

    return Disj(all_constraints)


def broadcast_dim(
    tensor_input1: Sequence[DVar | None],
    tensor_input2: Sequence[DVar],
    res1: Sequence[DVar],
    res2: Sequence[DVar],
    index: int,
    padding: bool = False,
) -> Constraint:
    """
    Apply broadcasting to the 'index' dimension of tensor_input1.
    Args:
        tensor_input1: should represent [d1, ..., d_index, ...] where d_index = 1
        tensor_input2: represents the second input
        res1: broadcasted result 1
        res2: broadcasted result 2
        index: the index to broadcast
        padding: If padding was used, then tensor_input1[index] does not exist

    Returns:

    """
    if tensor_input1[index] is None:
        if not padding:
            raise AssertionError("Expected padding when tensor_input1[index] is None")

    if not padding:
        # then the inputs are the same length so they all have dimensions at "index"
        return Conj(
            [
                BinConstraintD(tensor_input1[index], 1, op_eq),  # type: ignore[arg-type]
                BinConstraintD(res1[index], res2[index], op_eq),
                BinConstraintD(res2[index], tensor_input2[index], op_eq),
            ]
        )

    else:
        # we don't set the input dimension to 1, since it doesn't exist.
        return Conj(
            [
                BinConstraintD(res1[index], res2[index], op_eq),
                BinConstraintD(res2[index], tensor_input2[index], op_eq),
            ]
        )


def apply_padding(
    e1_var: TVar,
    e11: BinConstraintT,
    e2: BinConstraintT,
    e12: BinConstraintT,
    d2: list[DVar],
    d11: list[DVar],
    d12: list[DVar],
    counter: int,
) -> tuple[Constraint, int]:
    """
    We are considering the possibility where one input has less dimensions than
    another input, so we apply padding to the broadcasted results

    Args:
        e1_var: Variable representing the first input where padding will be
        e11: constraint of the form e11 = Tensortype[d1, ..., dn]
        e2:  constraint of the form e2 = Tensortype[d1, ..., dn]
        e12: constraint of the form e11 = Tensortype[d1, ..., dn]
        d2: Tensor variables for the second input
        d11: Tensor variables for the broadcasted first input
        d12: Tensor variables for the broadcasted second input
        counter: variable tracking

    Returns: A new constraint whose goal is to apply padding to the broadcasted result

    """

    res = []

    # pad the shorter input with None so we can pass it to the broadcasting helper function
    for i in range(1, len(d2)):
        d1, counter = gen_tensor_dims(i, counter)

        nat_constraints = gen_nat_constraints(d1 + d2 + d11 + d12)

        e1 = BinConstraintT(e1_var, TensorType(d1), op_eq)

        simulate_padding = [None] * (len(d2) - i)

        if len(simulate_padding + d1) != len(d2):
