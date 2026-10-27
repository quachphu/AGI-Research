            # Compute 1-w in target dtype to match CUDA rounding.
            d0 = self[0]
            neg_omw = -(1.0 - torch.tensor(weight, dtype=d0.dtype, device=d0.device))
            neg_omw_list = [neg_omw] * len(diff)
            for s, e in zip(self, end):
                s.copy_(e)
            torch._foreach_addcmul_(self, neg_omw_list, diff)
        else:
            # Low formula: start + w * diff  →  fma(w, diff, start)
            weights = [torch.full_like(d, weight) for d in diff]
            torch._foreach_addcmul_(self, weights, diff)
    return self


def foreach_pow_scalar(
    scalar: Any, exps: Sequence[bool | complex | float | int]
) -> tuple[torch.Tensor, ...]:
    return torch._foreach_pow([scalar for _ in exps], exps)


def predicate(obj: object) -> bool:
    # This will cause the rest of dynamo to handle the if statement correctly, so we don't have to rewrite it here.
    # We can't just use bool() here since we can't trace into that in general.
    if obj:
        return True
    return False


def group_tensors_by_device_and_dtype(
    tensorlistlist: list[list[torch.Tensor | None]], with_indices: bool = False
) -> dict[tuple[torch.device, torch.dtype], tuple[list[list[Any]], list[int]]]:
    """Pure Python implementation of torch._C._group_tensors_by_device_and_dtype.

    Groups tensors by their device and dtype. This is useful before sending
    tensors off to a foreach implementation, which requires tensors to be on
    one device and dtype.

    Args:
        tensorlistlist: A list of lists of tensors (tensors can be None).
        with_indices: If True, track original indices in the output.

    Returns:
        A dict mapping (device, dtype) tuples to (grouped_tensorlistlist, indices).
    """
    # Result dict: (device, dtype) -> (list of lists, indices)
    result: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Any]], list[int]]
    ] = {}

    if not tensorlistlist or not tensorlistlist[0]:
        return result

    num_lists = len(tensorlistlist)
    num_tensors = len(tensorlistlist[0])

    for idx in range(num_tensors):
        # Find the first non-None tensor at this index to get device and dtype
        first_tensor = None
        for tlist in tensorlistlist:
            if tlist is not None and idx < len(tlist) and tlist[idx] is not None:
                first_tensor = tlist[idx]
                break

        if first_tensor is None:
            # All tensors at this index are None, skip
            continue

        key = (first_tensor.device, first_tensor.dtype)

        if key not in result:
            # Initialize empty lists for each tensorlist
            result[key] = ([[] for _ in range(num_lists)], [])

        grouped_lists, indices = result[key]

        # Add tensors from each list at this index
        for list_idx, tlist in enumerate(tensorlistlist):
            if tlist is not None and idx < len(tlist):
                grouped_lists[list_idx].append(tlist[idx])
            else:
                grouped_lists[list_idx].append(None)

        if with_indices:
            indices.append(idx)

    return result
<|endoftext|>from typing import Any

import torch

from ..decorators import substitute_in_graph


@substitute_in_graph(  # type: ignore[arg-type]
    torch.Tensor._make_subclass
)
def make_subclass(
    cls: type[Any], data: torch.Tensor, requires_grad: bool = False, **kwargs: Any
) -> Any:
    with torch._C.DisableTorchFunctionSubclass():
        # This is a rough approximation of `THPVariable_make_subclass`. It should
        # suffice for most of Dynamo tracing purposes.
        # https://github.com/pytorch/pytorch/blob/ccfde4dadfa3c342076a1ee387017f84dd4ad2f7/torch/csrc/autograd/python_variable.cpp#L597-L650
        if len(kwargs) != 0:
            raise AssertionError(
                "_make_subclass only supports requires_grad as keyword arg"
            )
        data = data.detach()

        # Avoid unnecessary `requires_grad` mutation, which isn't supported in Dynamo.
        if data.requires_grad != requires_grad:
            data.requires_grad = requires_grad

        # Dynamo can't yet handle upcasting to base tensor type via `as_subclass`.
        if cls is torch.Tensor:
            return torch.Tensor(data)

        # Calling `as_subclass` because
        # 1. Dynamo knows how to handle it
        # 2. the C impls match at this point -- both `THPVariable_make_subclass` and
        #    `THPVariable_as_subclass` calls `THPVariable_NewWithVar`.
        return data.as_subclass(cls)


@substitute_in_graph(  # type: ignore[arg-type]
    torch.Tensor.__new__
)
def tensor_pynew(
    cls: type[Any], data: torch.Tensor, requires_grad: bool = False, **kwargs: Any
) -> Any:
    return make_subclass(cls, data, requires_grad, **kwargs)


__all__ = [
    "make_subclass",
    "tensor_pynew",
]
<|endoftext|># Owner(s): ["module: pytree"]

"""
Python polyfills for torch.utils.pytree
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, TypeVar

import optree
import optree._C
import optree.utils
from optree import (
    is_namedtuple,
    is_namedtuple_class,
    is_namedtuple_instance,
    is_structseq,
    is_structseq_class,
    is_structseq_instance,
    namedtuple_fields,
    structseq_fields,
)

import torch.utils._cxx_pytree as cxx_pytree  # noqa: F401  # load the C++ extension module
import torch.utils._pytree as python_pytree
from torch.utils._pytree import BUILTIN_TYPES, STANDARD_DICT_TYPES

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    import builtins
    from collections.abc import Callable, Iterable, Mapping
    from typing_extensions import Self, TypeIs

    from torch.utils._cxx_pytree import PyTree


__all__ = [
    "is_namedtuple",
    "is_namedtuple_class",
    "is_namedtuple_instance",
    "is_structseq",
    "is_structseq_class",
    "is_structseq_instance",
    "namedtuple_fields",
    "structseq_fields",
    "treespec_leaf",
    "treespec_tuple",
    "treespec_dict",
    "tree_is_leaf",
    "tree_iter",
    "tree_leaves",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_structure",
    "tree_unflatten",
]


_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


@substitute_in_graph(
    optree._C.is_dict_insertion_ordered,
    can_constant_fold_through=True,
)
def _(*args: Any, **kwargs: Any) -> bool:
    # In namespace 'torch', the dictionary is always traversed in insertion order.
    # This function returns True.
    raise ValueError(
        "Should not be called directly "
        "because the original function will be called in the constant fold path."
    )


__name = ""
for __name, __func in (
    ("is_namedtuple", is_namedtuple),
    ("is_namedtuple_class", is_namedtuple_class),
    ("is_namedtuple_instance", is_namedtuple_instance),
    ("is_structseq", is_structseq),
    ("is_structseq_class", is_structseq_class),
    ("is_structseq_instance", is_structseq_instance),
    ("namedtuple_fields", namedtuple_fields),
    ("structseq_fields", structseq_fields),
):
    globals()[__name] = substitute_in_graph(
        __func,  # type: ignore[arg-type]
        can_constant_fold_through=True,
    )(__func.__python_implementation__)  # type: ignore[attr-defined]
    del __func
del __name


@substitute_in_graph(optree.tree_is_leaf, can_constant_fold_through=True)  # type: ignore[arg-type]
def tree_is_leaf(
    tree: PyTree,
    /,
    is_leaf: Callable[[PyTree], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> bool:
    if (tree is None and none_is_leaf) or (is_leaf is not None and is_leaf(tree)):
        return True
    if optree.register_pytree_node.get(type(tree), namespace=namespace) is None:
        return True
    return False


@substitute_in_graph(optree.tree_iter, can_constant_fold_through=False)  # type: ignore[arg-type]
def tree_iter(
    tree: PyTree,
    /,
    is_leaf: Callable[[PyTree], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> Iterable[Any]:
    stack = [tree]
    while stack:
        node = stack.pop()
        if tree_is_leaf(
            node,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ):
            yield node
            continue

        children, *_ = optree.tree_flatten_one_level(
            node,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        stack.extend(reversed(children))


@substitute_in_graph(optree.tree_leaves, can_constant_fold_through=True)  # type: ignore[arg-type]
def tree_leaves(
    tree: PyTree,
    /,
    is_leaf: Callable[[PyTree], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> list[Any]:
    return list(
        tree_iter(
            tree,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
    )


class _Asterisk(str):
    __slots__ = ()

    def __new__(cls) -> Self:
        return super().__new__(cls, "*")

    def __repr__(self) -> str:
        return "*"  # no quotes


_asterisk = _Asterisk()
del _Asterisk


@dataclass(frozen=True, repr=False, eq=False, unsafe_hash=False, slots=True)
class PyTreeSpec:
    """Analog for :class:`optree.PyTreeSpec` in Python."""

    _children: tuple[PyTreeSpec, ...]
    _type: builtins.type | None
    _metadata: Any
    _entries: tuple[Any, ...]
    _unflatten_func: Callable[[Any | None, Iterable[PyTree]], PyTree] | None
    none_is_leaf: bool
    namespace: str

    num_nodes: int = field(init=False)
    num_leaves: int = field(init=False)
    num_children: int = field(init=False)

    def __post_init__(self, /) -> None:
        if self._type is None:
            if len(self._children) != 0:
                raise AssertionError("Leaf node must have no children")
            if self._metadata is not None:
                raise AssertionError("Leaf node must have no metadata")
            if self._entries != ():
                raise AssertionError("Leaf node must have no entries")
            if self._unflatten_func is not None:
                raise AssertionError("Leaf node must have no unflatten_func")
            num_nodes = 1
            num_leaves = 1
            num_children = 0
        else:
            if not callable(self._unflatten_func):
                raise AssertionError(
                    "Non-leaf node must have a callable unflatten_func"
                )
            num_nodes = 1
            num_leaves = 0
            for child in self._children:
                num_nodes += child.num_nodes
                num_leaves += child.num_leaves
            num_children = len(self._children)

        object.__setattr__(self, "num_nodes", num_nodes)
        object.__setattr__(self, "num_leaves", num_leaves)
        object.__setattr__(self, "num_children", num_children)

    def __repr__(self, /) -> str:
        def helper(treespec: PyTreeSpec) -> str:
            if treespec.is_leaf():
                if treespec.type is not None:
                    raise AssertionError("Leaf treespec must have type None")
                return _asterisk

            if treespec.type is None:
                raise AssertionError("Non-leaf treespec must have a type")
            if not callable(treespec._unflatten_func):
                raise AssertionError(
                    "Non-leaf treespec must have a callable unflatten_func"
                )
            children_representations = [
                helper(subspec) for subspec in treespec._children
            ]
            if (
                treespec.type in BUILTIN_TYPES
                or (treespec.type is type(None) and not self.none_is_leaf)
                or optree.is_namedtuple_class(treespec.type)
                or optree.is_structseq_class(treespec.type)
            ):
                return treespec._unflatten_func(
                    treespec._metadata,
                    children_representations,
                )
            return (
                f"CustomTreeNode({treespec.type.__name__}[{treespec._metadata!r}], "
                f"[{', '.join(children_representations)}])"
            )

        inner = [
            str(helper(self)),
            *(["NoneIsLeaf"] if self.none_is_leaf else []),
            f"namespace={self.namespace!r}",
        ]
        return f"PyTreeSpec({', '.join(inner)})"

    def __len__(self, /) -> int:
        return self.num_leaves

    def __eq__(self, other: Any, /) -> bool:
        if not isinstance(other, PyTreeSpec):
            return NotImplemented
        if (
            self.none_is_leaf != other.none_is_leaf
            or self.num_nodes != other.num_nodes
            or self.num_leaves != other.num_leaves
            or self._type is not other._type
            or self._metadata != other._metadata
        ):
            return False
        # A namespace mismatch is ignored if either side has an empty namespace, mirroring optree's
        # `PyTreeSpec::EqualTo` (src/treespec/richcomparison.cpp).
        if self.namespace and other.namespace and self.namespace != other.namespace:
            return False
        return self._children == other._children

    def __hash__(self, /) -> int:
        node_type = self._type
        if node_type in STANDARD_DICT_TYPES:
            # Dict metadata is the (unhashable) list of keys; hash the keys instead.
            node_data: Any = tuple(self._entries)
        elif (
            node_type is None
            or node_type in BUILTIN_TYPES
            or optree.is_namedtuple_class(node_type)
            or optree.is_structseq_class(node_type)
        ):
            node_data = self._metadata
        else:
            # Custom node metadata may be unhashable; hash the node type only, mirroring optree's
            # `PyTreeSpec::HashValueImpl` (src/treespec/hashing.cpp).
            node_data = None
        # `namespace` is intentionally excluded: `__eq__` treats an empty namespace as a wildcard,
        # so specs that compare equal must hash the same.
        return hash(
            (
                self.none_is_leaf,
                node_type,
                self.num_leaves,
                self.num_nodes,
                node_data,
                self._children,
            )
        )

    @property
    def type(self, /) -> builtins.type | None:
        return self._type

    def is_leaf(self, /) -> bool:
        return self.num_nodes == 1 and self.num_leaves == 1

    def paths(self, /) -> list[tuple[Any, ...]]:
        def helper(treespec: PyTreeSpec, path_prefix: list[Any]) -> None:
            if treespec.is_leaf():
                paths.append(path_prefix)
                return

            for entry, subspec in zip(
                treespec._entries,
                treespec._children,
                strict=True,
            ):
                helper(subspec, path_prefix + [entry])

        paths: list[list[Any]] = []
        helper(self, [])
        return [tuple(path) for path in paths]

    def accessors(self, /) -> list[optree.PyTreeAccessor]:
        def helper(
            treespec: PyTreeSpec,
            entry_path_prefix: list[optree.PyTreeEntry],
        ) -> None:
            if treespec.is_leaf():
                entry_paths.append(entry_path_prefix)
                return

            node_type = treespec.type
            if node_type is None:
                raise AssertionError("Non-leaf treespec must have a type")
            handler = optree.register_pytree_node.get(
                node_type, namespace=treespec.namespace
            )
            if handler is None:
                raise AssertionError(f"No pytree handler registered for {node_type}")
            kind: optree.PyTreeKind = handler.kind
            path_entry_type: type[optree.PyTreeEntry] = handler.path_entry_type

            for entry, subspec in zip(
                treespec._entries,
                treespec._children,
                strict=True,
            ):
                helper(
                    subspec,
                    entry_path_prefix + [path_entry_type(entry, node_type, kind)],
                )

        entry_paths: list[list[optree.PyTreeEntry]] = []
        helper(self, [])
        return [optree.PyTreeAccessor(path) for path in entry_paths]

    def children(self, /) -> list[PyTreeSpec]:
        return list(self._children)

    def child(self, index: int, /) -> PyTreeSpec:
        return self._children[index]

    def entries(self, /) -> list[Any]:
        return list(self._entries)

    def entry(self, index: int, /) -> Any:
        return self._entries[index]

    def flatten_up_to(self, tree: PyTree, /) -> list[PyTree]:
        def helper(
            treespec: PyTreeSpec,
            node: PyTree,
            subtrees: list[PyTree],
        ) -> None:
            if treespec.is_leaf():
                subtrees.append(node)
                return

            node_type = type(node)
            if treespec.type not in BUILTIN_TYPES:
                # Always require custom node types to match exactly
                if node_type != treespec.type:
                    raise ValueError(
                        f"Type mismatch; "
                        f"expected {treespec.type!r}, but got {node_type!r}.",
                    )

                children, metadata, *_ = optree.tree_flatten_one_level(
                    node,
                    none_is_leaf=self.none_is_leaf,
                    namespace=self.namespace,
                )
                if len(children) != treespec.num_children:
                    raise ValueError(
                        f"Node arity mismatch; "
                        f"expected {treespec.num_children}, but got {len(children)}.",
                    )
                if metadata != treespec._metadata:
                    raise ValueError(
                        f"Node context mismatch for custom node type {treespec.type!r}.",
                    )
            else:
                # For builtin dictionary types, we allow some flexibility
                # Otherwise, we require exact matches
                both_standard_dict = (
                    treespec.type in STANDARD_DICT_TYPES
                    and node_type in STANDARD_DICT_TYPES
                )
                if not both_standard_dict and node_type != treespec.type:
                    raise ValueError(
                        f"Node type mismatch; "
                        f"expected {treespec.type!r}, but got {node_type!r}.",
                    )
                if len(node) != treespec.num_children:
                    raise ValueError(
                        f"Node arity mismatch; "
                        f"expected {treespec.num_children}, but got {len(node)}.",
                    )

                if both_standard_dict:
                    # dictionary types are compatible with each other
                    expected_keys = treespec.entries()
                    got_key_set = set(node)
                    expected_key_set = set(expected_keys)
                    if got_key_set != expected_key_set:
                        missing_keys = expected_key_set.difference(got_key_set)
                        extra_keys = got_key_set.difference(expected_key_set)
                        message = ""
                        if missing_keys:
                            message += f"; missing key(s): {missing_keys}"
                        if extra_keys:
                            message += f"; extra key(s): {extra_keys}"
                        raise ValueError(f"Node keys mismatch{message}.")
                    children = [node[key] for key in expected_keys]
                else:
                    # node_type is treespec.type
                    children, metadata, *_ = optree.tree_flatten_one_level(
                        node,
                        none_is_leaf=self.none_is_leaf,
                        namespace=self.namespace,
                    )
                    if (
                        node_type is not deque  # ignore mismatch of `maxlen` for deque
                    ) and metadata != treespec._metadata:
                        raise ValueError(
                            f"Node metadata mismatch for node type {treespec.type!r}; "
                            f"expected {treespec._metadata!r}, but got {metadata!r}.",  # namedtuple type mismatch
                        )

            for subtree, subspec in zip(children, treespec._children, strict=True):
                helper(subspec, subtree, subtrees)

        subtrees: list[PyTree] = []
        helper(self, tree, subtrees)
        return subtrees

    def unflatten(self, leaves: Iterable[Any], /) -> PyTree:
        if not isinstance(leaves, (list, tuple)):
            leaves = list(leaves)
        if len(leaves) != self.num_leaves:
            raise ValueError(
                f"treespec.unflatten(leaves): `leaves` has length {len(leaves)} "
                f"but the spec refers to a pytree that holds {self.num_leaves} "
                f"items ({self}).",
            )
        if self.is_leaf():
            return leaves[0]

        # Recursively unflatten the children
        start = 0
        end = 0
        subtrees = []
        for subspec in self._children:
            end += subspec.num_leaves
            subtrees.append(subspec.unflatten(leaves[start:end]))
            start = end

        if not callable(self._unflatten_func):
            raise AssertionError("Non-leaf node must have a callable unflatten_func")
        return self._unflatten_func(self._metadata, subtrees)


def _is_pytreespec_instance(obj: Any, /) -> TypeIs[PyTreeSpec | python_pytree.TreeSpec]:
    return isinstance(obj, (PyTreeSpec, python_pytree.TreeSpec))


@substitute_in_graph(  # type: ignore[arg-type]
    optree.treespec_leaf,
    # We need to disable constant folding here because we want the function to reference the
    # PyTreeSpec class defined above, not the one in the C++ module.
    can_constant_fold_through=False,
)
def treespec_leaf(
    *,
    none_is_leaf: bool = False,
    namespace: str = "",  # unused
) -> PyTreeSpec:
    return PyTreeSpec(
        (),
        None,
        None,
        (),
        None,
        none_is_leaf=none_is_leaf,
        namespace="",
    )


@substitute_in_graph(  # type: ignore[arg-type]
    optree.treespec_tuple,
    # We need to disable constant folding here because we want the function to reference the
    # PyTreeSpec class defined above, not the one in the C++ module.
    can_constant_fold_through=False,
)
def treespec_tuple(
    iterable: Iterable[PyTreeSpec] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> PyTreeSpec:
    children = tuple(iterable)
    if any(not _is_pytreespec_instance(child) for child in children):
        raise ValueError(f"Expected a tuple of PyTreeSpecs, got: {children!r}.")
    if any(child.none_is_leaf != none_is_leaf for child in children):
        raise ValueError(
            "All children PyTreeSpecs must have the same `none_is_leaf` value "
            f"as the parent; expected {none_is_leaf}, got: {children!r}.",
        )
    if any(child.namespace not in (namespace, "") for child in children):
        raise ValueError(
            "All children PyTreeSpecs must have the same `namespace` value "
            f"as the parent; expected {namespace!r}, got: {children!r}.",
        )
    handler = optree.register_pytree_node.get(tuple, namespace=namespace)
    if handler is None:
        raise AssertionError("No pytree handler registered for tuple")
    return PyTreeSpec(
        tuple(children),
        tuple,
        None,
        tuple(range(len(children))),
        handler.unflatten_func,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


@substitute_in_graph(  # type: ignore[arg-type]
    optree.treespec_dict,
    # We need to disable constant folding here because we want the function to reference the
    # PyTreeSpec class defined above, not the one in the C++ module.
    can_constant_fold_through=False,
)
def treespec_dict(
    mapping: Mapping[Any, PyTreeSpec] | Iterable[tuple[Any, PyTreeSpec]] = (),
    /,
    *,
    none_is_leaf: bool = False,
    namespace: str = "",
    **kwargs: PyTreeSpec,
) -> PyTreeSpec:
    dct = dict(mapping, **kwargs)
    if any(not _is_pytreespec_instance(child) for child in dct.values()):
        raise ValueError(f"Expected a dictionary of TreeSpecs, got: {dct!r}.")
    if any(child.none_is_leaf != none_is_leaf for child in dct.values()):
        raise ValueError(
            "All children PyTreeSpecs must have the same `none_is_leaf` value "
            f"as the parent; expected {none_is_leaf}, got: {dct!r}.",
        )
    if any(child.namespace not in (namespace, "") for child in dct.values()):
        raise ValueError(
            "All children PyTreeSpecs must have the same `namespace` value "
            f"as the parent; expected {namespace!r}, got: {dct!r}.",
        )

    (
        children,
        metadata,
        entries,
        unflatten_func,
    ) = optree.tree_flatten_one_level(  # type: ignore[assignment,var-annotated]
        dct,  # type: ignore[arg-type]
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    return PyTreeSpec(
        tuple(children),  # type: ignore[arg-type]
        dict,
        metadata,
        entries,
        unflatten_func,  # type: ignore[arg-type]
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


@substitute_in_graph(  # type: ignore[arg-type]
    optree.tree_flatten,
    # We need to disable constant folding here because we want the function to reference the
    # PyTreeSpec class defined above, not the one in the C++ module.
    can_constant_fold_through=False,
)
def tree_flatten(
    tree: PyTree,
    /,
    is_leaf: Callable[[PyTree], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> tuple[list[Any], PyTreeSpec]:
    def helper(node: PyTree, leaves: list[Any]) -> PyTreeSpec:
        if tree_is_leaf(
            node,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        ):
            leaves.append(node)
            return PyTreeSpec(
                (),
                None,
                None,
                (),
                None,
                none_is_leaf=none_is_leaf,
                namespace="",
            )

        (
            children,
            metadata,
            entries,
            unflatten_func,
        ) = optree.tree_flatten_one_level(
            node,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )

        # Recursively flatten the children
        subspecs = tuple(helper(child, leaves) for child in children)
        return PyTreeSpec(
            subspecs,
            type(node),
            metadata,
            entries,
            unflatten_func,  # type: ignore[arg-type]
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )  # type: ignore[arg-type]

    leaves: list[Any] = []
    treespec = helper(tree, leaves)
    return leaves, treespec


@substitute_in_graph(  # type: ignore[arg-type]
    optree._C.flatten,
    # We need to disable constant folding here because we want the function to reference the
    # PyTreeSpec class defined above, not the one in the C++ module.
    can_constant_fold_through=False,
)
def _C_flatten(
    tree: PyTree,
    /,
    leaf_predicate: Callable[[PyTree], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> tuple[list[Any], PyTreeSpec]:
    return tree_flatten(  # type: ignore[return-value]
        tree,
        is_leaf=leaf_predicate,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


@substitute_in_graph(  # type: ignore[arg-type]
    optree.tree_flatten_with_path,
    # We need to disable constant folding here because we want the function to reference the
    # PyTreeSpec class defined above, not the one in the C++ module.
    can_constant_fold_through=False,
)
def tree_flatten_with_path(
    tree: PyTree,
    /,
    is_leaf: Callable[[PyTree], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> tuple[list[tuple[Any, ...]], list[Any], PyTreeSpec]:
    leaves, treespec = tree_flatten(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    return treespec.paths(), leaves, treespec  # type: ignore[return-value]


@substitute_in_graph(  # type: ignore[arg-type]
    optree._C.flatten_with_path,
    # We need to disable constant folding here because we want the function to reference the
    # PyTreeSpec class defined above, not the one in the C++ module.
    can_constant_fold_through=False,
)
def _C_flatten_with_path(
    tree: PyTree,
    /,
    leaf_predicate: Callable[[PyTree], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> tuple[list[tuple[Any, ...]], list[Any], PyTreeSpec]:
    return tree_flatten_with_path(  # type: ignore[return-value]
        tree,
        is_leaf=leaf_predicate,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )


@substitute_in_graph(  # type: ignore[arg-type]
    optree.tree_structure,
    # We need to disable constant folding here because we want the function to reference the
    # PyTreeSpec class defined above, not the one in the C++ module.
    can_constant_fold_through=False,
)
def tree_structure(
    tree: PyTree,
    /,
    is_leaf: Callable[[PyTree], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> PyTreeSpec:
    return tree_flatten(  # type: ignore[return-value]
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )[1]


@substitute_in_graph(  # type: ignore[arg-type]
    optree.tree_unflatten,
    # We need to disable constant folding here because we want the function to reference the
    # PyTreeSpec class defined above, not the one in the C++ module.
    can_constant_fold_through=False,
)
def tree_unflatten(treespec: PyTreeSpec, leaves: Iterable[Any]) -> PyTree:
    if not _is_pytreespec_instance(treespec):
        raise TypeError(
            f"Expected `treespec` to be an instance of "
            f"PyTreeSpec but got item of type {type(treespec)}."
        )
    return treespec.unflatten(leaves)


_none_registration = optree.register_pytree_node.get(type(None))
if _none_registration is None:
    raise AssertionError("No pytree handler registered for NoneType")


@substitute_in_graph(  # type: ignore[arg-type]
    _none_registration.unflatten_func,
    can_constant_fold_through=True,
    skip_signature_check=True,
)
def none_unflatten(_: None, children: Iterable[_T], /) -> None:
    if len(list(children)) != 0:
        raise ValueError("Expected no children.")
    return None


with optree.dict_insertion_ordered(False, namespace="torch"):
    _dict_registration = optree.register_pytree_node.get(dict)
    if _dict_registration is None:
        raise AssertionError("No pytree handler registered for dict")


@substitute_in_graph(  # type: ignore[arg-type]
    _dict_registration.flatten_func,
    can_constant_fold_through=True,
    skip_signature_check=True,
)
def dict_flatten(
    dct: dict[_KT, _VT], /
) -> tuple[list[_VT], tuple[list[_KT], list[_KT]], tuple[_KT, ...]]:
    sorted_keys = optree.utils.total_order_sorted(dct)
    values = [dct[key] for key in sorted_keys]
    original_keys = list(dct)
    return values, (original_keys, sorted_keys), tuple(sorted_keys)


@substitute_in_graph(  # type: ignore[arg-type]
    _dict_registration.unflatten_func,
    can_constant_fold_through=True,
    skip_signature_check=True,
)
def dict_unflatten(
    metadata: tuple[list[_KT], list[_KT]],
    values: Iterable[_VT],
    /,
) -> dict[_KT, _VT]:
    original_keys, sorted_keys = metadata
    d = dict.fromkeys(original_keys)
    d.update(zip(sorted_keys, values, strict=True))
    return d  # type: ignore[return-value]
<|endoftext|>"""
Python polyfills for struct
"""

from __future__ import annotations

import struct
from typing import Any
from typing_extensions import Buffer

from ..decorators import substitute_in_graph


__all__ = [
    "pack",
    "unpack",
]


@substitute_in_graph(struct.pack, can_constant_fold_through=True)  # type: ignore[arg-type]
def pack(format: bytes | str, /, *v: Any) -> bytes:
    return struct.pack(format, *v)


@substitute_in_graph(struct.unpack, can_constant_fold_through=True)  # type: ignore[arg-type]
def unpack(format: bytes | str, buffer: Buffer, /) -> tuple[Any, ...]:
    return struct.unpack(format, buffer)
<|endoftext|>"""
Python polyfills for itertools
"""

from __future__ import annotations

import itertools
import operator
import sys
from collections.abc import Callable
from typing import overload, TYPE_CHECKING, TypeAlias, TypeVar

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


__all__ = [
    "accumulate",
    "combinations",
    "combinations_with_replacement",
    "compress",
    "cycle",
    "dropwhile",
    "filterfalse",
    "islice",
    "pairwise",
    "permutations",
    "starmap",
    "takewhile",
    "tee",
]

if sys.version_info >= (3, 12):
    __all__.append("batched")


_T = TypeVar("_T")
_U = TypeVar("_U")
_Predicate: TypeAlias = Callable[[_T], object]
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


# Reference: https://docs.python.org/3/library/itertools.html#itertools.accumulate
@substitute_in_graph(itertools.accumulate, is_embedded_type=True)  # type: ignore[arg-type]
def accumulate(
    iterable: Iterable[_T],
    func: Callable[[_T, _T], _T] | None = None,
    *,
    initial: _T | None = None,
) -> Iterator[_T]:
    # call iter outside of the generator to match cpython behavior
    iterator = iter(iterable)
    if func is None:
        func = operator.add

    def _accumulate(iterator: Iterator[_T]) -> Iterator[_T]:
        total = initial
        if total is None:
            try:
                total = next(iterator)
            except StopIteration:
                return

        yield total
        for element in iterator:
            total = func(total, element)
            yield total

    return _accumulate(iterator)


# Reference: https://docs.python.org/3/library/itertools.html#itertools.combinations
@substitute_in_graph(itertools.combinations, is_embedded_type=True)  # type: ignore[arg-type]
def combinations(iterable: Iterable[_T], r: int, /) -> Iterator[tuple[_T, ...]]:
    pool = tuple(iterable)
    n = len(pool)

    if r < 0:
        raise ValueError("r must be non-negative")

    def _combinations() -> Iterator[tuple[_T, ...]]:
        if r > n:
            return

        indices = list(range(r))
        yield tuple(pool[i] for i in indices)

        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return

            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1

            yield tuple(pool[i] for i in indices)

    return _combinations()


# Reference: https://docs.python.org/3/library/itertools.html#itertools.compress
@substitute_in_graph(itertools.compress, is_embedded_type=True)  # type: ignore[arg-type]
def compress(data: Iterable[_T], selectors: Iterable[_U], /) -> Iterator[_T]:
    return (datum for datum, selector in zip(data, selectors) if selector)


# Reference: https://docs.python.org/3/library/itertools.html#itertools.cycle
@substitute_in_graph(itertools.cycle, is_embedded_type=True)  # type: ignore[arg-type]
def cycle(iterable: Iterable[_T]) -> Iterator[_T]:
    iterator = iter(iterable)

    def _cycle(iterator: Iterator[_T]) -> Iterator[_T]:
        # pyrefly: ignore [implicit-any]
        saved = []
        for element in iterable:
            yield element
            saved.append(element)

        while saved:
            for element in saved:
                yield element

    return _cycle(iterator)


# Reference: https://docs.python.org/3/library/itertools.html#itertools.dropwhile
@substitute_in_graph(itertools.dropwhile, is_embedded_type=True)  # type: ignore[arg-type]
def dropwhile(predicate: _Predicate[_T], iterable: Iterable[_T], /) -> Iterator[_T]:
    # dropwhile(lambda x: x < 5, [1, 4, 6, 3, 8]) -> 6 3 8
    if not callable(predicate):
        raise TypeError(f"'{type(predicate).__name__}' object is not callable")

    iterator = iter(iterable)
    for x in iterator:
        if not predicate(x):
            yield x
            break

    yield from iterator


# Reference: https://docs.python.org/3/library/itertools.html#itertools.takewhile
@substitute_in_graph(itertools.takewhile, is_embedded_type=True)  # type: ignore[arg-type]
def takewhile(predicate: _Predicate[_T], iterable: Iterable[_T], /) -> Iterator[_T]:
    # takewhile(lambda x: x<5, [1,4,6,3,8]) → 1 4
    if not callable(predicate):
        raise TypeError(f"'{type(predicate).__name__}' object is not callable")

    for x in iterable:
        if not predicate(x):
            break
        yield x


@overload
def starmap(
    function: Callable[[], _U],
    iterable: Iterable[tuple[()]],
    /,
) -> itertools.starmap[_U]: ...


@overload
def starmap(
    function: Callable[[_T], _U],
    iterable: Iterable[tuple[_T]],
    /,
) -> itertools.starmap[_U]: ...


@overload
def starmap(
    function: Callable[[_T, _T1], _U],
    iterable: Iterable[tuple[_T, _T1]],
    /,
) -> itertools.starmap[_U]: ...


@overload
def starmap(
    function: Callable[[_T, _T1, _T2], _U],
    iterable: Iterable[tuple[_T, _T1, _T2]],
    /,
) -> itertools.starmap[_U]: ...


# Reference: https://docs.python.org/3/library/itertools.html#itertools.starmap
@substitute_in_graph(itertools.starmap, is_embedded_type=True)  # type: ignore[arg-type]
# pyrefly: ignore [implicit-any]
def starmap(function: Callable[..., _T], iterable: Iterable, /) -> Iterable[_T]:
    # starmap(pow, [(2,5), (3,2), (10,3)]) → 32 9 1000
    if not callable(function):
        raise TypeError(f"'{type(function).__name__}' object is not callable")

    for args in iterable:
        yield function(*args)


@substitute_in_graph(itertools.filterfalse, is_embedded_type=True)  # type: ignore[arg-type]
def filterfalse(function: _Predicate[_T], iterable: Iterable[_T], /) -> Iterator[_T]:
    it = iter(iterable)
    if function is None:
        return filter(operator.not_, it)
    else:
        return filter(lambda x: not function(x), it)


# Reference: https://docs.python.org/3/library/itertools.html#itertools.islice
@substitute_in_graph(itertools.islice, is_embedded_type=True)  # type: ignore[arg-type]
def islice(iterable: Iterable[_T], /, *args: int | None) -> Iterator[_T]:
    s = slice(*args)
    start = 0 if s.start is None else s.start
    stop = s.stop
    step = 1 if s.step is None else s.step
    if start < 0 or (stop is not None and stop < 0) or step <= 0:
        raise ValueError(
            "Indices for islice() must be None or an integer: 0 <= x <= sys.maxsize.",
        )

    if stop is None:
        # TODO: use indices = itertools.count() and merge implementation with the else branch
        #       when we support infinite iterators
        next_i = start
        for i, element in enumerate(iterable):
            if i == next_i:
                yield element
                next_i += step
    else:
        indices = range(max(start, stop))
        next_i = start
        for i, element in zip(indices, iterable):
            if i == next_i:
                yield element
                next_i += step


# Reference: https://docs.python.org/3/library/itertools.html#itertools.pairwise
@substitute_in_graph(itertools.pairwise, is_embedded_type=True)  # type: ignore[arg-type]
def pairwise(iterable: Iterable[_T], /) -> Iterator[tuple[_T, _T]]:
    a = None
    first = True
    for b in iterable:
        if first:
            first = False
        else:
            yield a, b  # type: ignore[misc]
        a = b


# Reference: https://docs.python.org/3/library/itertools.html#itertools.permutations
@substitute_in_graph(itertools.permutations, is_embedded_type=True)  # type: ignore[arg-type]
def permutations(
    iterable: Iterable[_T], r: int | None = None, /
) -> Iterator[tuple[_T, ...]]:
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r

    if r < 0:
        raise ValueError("r must be non-negative")

    def _permutations() -> Iterator[tuple[_T, ...]]:
        if r > n:
            return

        indices = list(range(n))
        cycles = list(range(n, n - r, -1))
        yield tuple(pool[i] for i in indices[:r])

        while n:
            for i in reversed(range(r)):
                cycles[i] -= 1
                if cycles[i] == 0:
                    indices[i:] = indices[i + 1 :] + indices[i : i + 1]
                    cycles[i] = n - i
                else:
                    j = cycles[i]
                    indices[i], indices[-j] = indices[-j], indices[i]
                    yield tuple(pool[i] for i in indices[:r])
                    break
            else:
                return

    return _permutations()


# Reference: https://docs.python.org/3/library/itertools.html#itertools.tee
@substitute_in_graph(itertools.tee)
def tee(iterable: Iterable[_T], n: int = 2, /) -> tuple[Iterator[_T], ...]:
    iterator = iter(iterable)
    shared_link = [None, None]

    def _tee(link) -> Iterator[_T]:  # type: ignore[no-untyped-def]
        try:
            while True:
                if link[1] is None:
                    link[0] = next(iterator)
                    link[1] = [None, None]
                value, link = link
                yield value
        except StopIteration:
            return

    return tuple(_tee(shared_link) for _ in range(n))


# Reference: https://docs.python.org/3/library/itertools.html#itertools.combinations_with_replacement
@substitute_in_graph(itertools.combinations_with_replacement, is_embedded_type=True)  # type: ignore[arg-type]
def combinations_with_replacement(
    iterable: Iterable[_T], r: int, /
) -> Iterator[tuple[_T, ...]]:
    if r < 0:
        raise ValueError("r must be non-negative")

    pool = tuple(iterable)
    n = len(pool)

    def _combinations_with_replacement() -> Iterator[tuple[_T, ...]]:
        if r == 0:
            yield ()
            return
        if n == 0:
            return

        indices = [0] * r
        yield tuple(pool[i] for i in indices)
        while True:
            for i in range(r - 1, -1, -1):
                if indices[i] != n - 1:
                    break
            else:
                return
            indices[i:] = [indices[i] + 1] * (r - i)
            yield tuple(pool[i] for i in indices)

    return _combinations_with_replacement()


if sys.version_info >= (3, 12):
    # Reference: https://docs.python.org/3/library/itertools.html#itertools.batched
    @substitute_in_graph(itertools.batched, is_embedded_type=True)  # type: ignore[arg-type]
    def batched(*args, **kwargs) -> Iterator[tuple[_T, ...]]:  # type: ignore[no-untyped-def]
        if len(args) != 2:
            raise TypeError(
                f"batched takes exactly 2 positional arguments({len(args)} given)"
            )
        if kwargs.keys() - {"strict"}:
            unexpected = next(iter(kwargs.keys() - {"strict"}))
            raise TypeError(
                f"batched() got an unexpected keyword argument '{unexpected}'"
            )

        iterable, n = args
        strict = kwargs.get("strict", False)
        n = operator.index(n)
        if n < 1:
            raise ValueError("n must be at least one")

        iterator = iter(iterable)

        def _batched(iterator: Iterator[_T]) -> Iterator[tuple[_T, ...]]:
            while batch := tuple(islice(iterator, n)):
                if strict and len(batch) != n:
                    raise ValueError("batched(): incomplete batch")
                yield batch

        return _batched(iterator)
<|endoftext|>"""
Python polyfills for heapq
"""

from __future__ import annotations

import heapq
import importlib
import sys
from typing import TYPE_CHECKING, TypeVar

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from types import ModuleType


_T = TypeVar("_T")


# Partially copied from CPython test/support/import_helper.py
# https://github.com/python/cpython/blob/bb8791c0b75b5970d109e5557bfcca8a578a02af/Lib/test/support/import_helper.py
def _save_and_remove_modules(names: set[str]) -> dict[str, ModuleType]:
    orig_modules = {}
    prefixes = tuple(name + "." for name in names)
    for modname in list(sys.modules):
        if modname in names or modname.startswith(prefixes):
            orig_modules[modname] = sys.modules.pop(modname)
    return orig_modules


def import_fresh_module(name: str, blocked: list[str]) -> ModuleType:
    # Keep track of modules saved for later restoration as well
    # as those which just need a blocking entry removed
    names = {name, *blocked}
    orig_modules = _save_and_remove_modules(names)
    for modname in blocked:
        sys.modules[modname] = None  # type: ignore[assignment]

    try:
        return importlib.import_module(name)
    finally:
        _save_and_remove_modules(names)
        sys.modules.update(orig_modules)


# Import the pure Python heapq module, blocking the C extension
py_heapq = import_fresh_module("heapq", blocked=["_heapq"])


__all__ = [
    "_heapify_max",
    "_heappop_max",
    "_heapreplace_max",
    "heapify",
    "heappop",
    "heappush",
    "heappushpop",
    "heapreplace",
    "merge",
    "nlargest",
    "nsmallest",
]


@substitute_in_graph(heapq._heapify_max)
def _heapify_max(heap: list[_T], /) -> None:
    return py_heapq._heapify_max(heap)


@substitute_in_graph(heapq._heappop_max)  # type: ignore[attr-defined]
def _heappop_max(heap: list[_T]) -> _T:
    return py_heapq._heappop_max(heap)


@substitute_in_graph(heapq._heapreplace_max)  # type: ignore[attr-defined]
def _heapreplace_max(heap: list[_T], item: _T) -> _T:
    return py_heapq._heapreplace_max(heap, item)


@substitute_in_graph(heapq.heapify)
def heapify(heap: list[_T], /) -> None:
    return py_heapq.heapify(heap)


@substitute_in_graph(heapq.heappop)
def heappop(heap: list[_T], /) -> _T:
    return py_heapq.heappop(heap)


@substitute_in_graph(heapq.heappush)
def heappush(heap: list[_T], item: _T) -> None:
    return py_heapq.heappush(heap, item)


@substitute_in_graph(heapq.heappushpop)
def heappushpop(heap: list[_T], item: _T) -> _T:
    return py_heapq.heappushpop(heap, item)


@substitute_in_graph(heapq.heapreplace)
def heapreplace(heap: list[_T], item: _T) -> _T:
    return py_heapq.heapreplace(heap, item)


@substitute_in_graph(heapq.merge)  # type: ignore[arg-type]
def merge(*iterables, key=None, reverse=False):  # type: ignore[no-untyped-def]
    return py_heapq.merge(*iterables, key=key, reverse=reverse)


@substitute_in_graph(heapq.nlargest)  # type: ignore[arg-type]
def nlargest(n, iterable, key=None):  # type: ignore[no-untyped-def]
    return py_heapq.nlargest(n, iterable, key=key)


@substitute_in_graph(heapq.nsmallest)  # type: ignore[arg-type]
def nsmallest(n, iterable, key=None):  # type: ignore[no-untyped-def]
    return py_heapq.nsmallest(n, iterable, key=key)
<|endoftext|>"""
Python polyfills for os
"""

from __future__ import annotations

import os
from typing import AnyStr

from ..decorators import substitute_in_graph


__all__ = ["fspath"]


# Copied from os.py in the standard library
# pyrefly: ignore [bad-argument-type]
@substitute_in_graph(os.fspath, can_constant_fold_through=True)
def fspath(path: AnyStr | os.PathLike[AnyStr]) -> AnyStr:
    if isinstance(path, (str, bytes)):
        return path

    path_type = type(path)
    try:
        path_repr = path_type.__fspath__(path)  # type: ignore[arg-type]
    except AttributeError:
        if hasattr(path_type, "__fspath__"):
            raise
        raise TypeError(
            f"expected str, bytes or os.PathLike object, not {path_type.__name__}",
        ) from None
    if isinstance(path_repr, (str, bytes)):
        return path_repr  # type: ignore[return-value]
    raise TypeError(
        f"expected {path_type.__name__}.__fspath__() to return str or bytes, "
        f"not {type(path_repr).__name__}",
    )
<|endoftext|>"""
Python polyfills for functools
"""

import functools
from collections.abc import Callable, Iterable
from functools import _initial_missing  # type: ignore[attr-defined]
from typing import Any, TypeVar

from ..decorators import substitute_in_graph


__all__ = ["cmp_to_key", "reduce"]


_T = TypeVar("_T")
_U = TypeVar("_U")


class _KeyWrapper:
    def __init__(self, mycmp: Callable[[Any, Any], int], obj: Any) -> None:
        self.mycmp = mycmp
        self.obj = obj

    def __lt__(self, other: "_KeyWrapper") -> bool:
        return self.mycmp(self.obj, other.obj) < 0

    def __gt__(self, other: "_KeyWrapper") -> bool:
        return self.mycmp(self.obj, other.obj) > 0

    def __eq__(self, other: object) -> bool:
        return self.mycmp(self.obj, other.obj) == 0  # type: ignore[attr-defined]

    def __le__(self, other: "_KeyWrapper") -> bool:
        return self.mycmp(self.obj, other.obj) <= 0

    def __ge__(self, other: "_KeyWrapper") -> bool:
        return self.mycmp(self.obj, other.obj) >= 0

    __hash__ = None  # type: ignore[assignment]


# Reference: https://docs.python.org/3/library/functools.html#functools.cmp_to_key
@substitute_in_graph(functools.cmp_to_key, skip_signature_check=True)
def cmp_to_key(mycmp: Callable[[Any, Any], int]) -> Callable[[Any], _KeyWrapper]:
    return functools.partial(_KeyWrapper, mycmp)


# Reference: https://docs.python.org/3/library/functools.html#functools.reduce
@substitute_in_graph(functools.reduce)
def reduce(
    function: Callable[[_U, _T], _U],
    iterable: Iterable[_T],
    initial: _U = _initial_missing,  # type: ignore[assignment]
    /,
) -> _U:
    it = iter(iterable)

    value: _U
    if initial is _initial_missing:
        try:
            value = next(it)  # type: ignore[assignment]
        except StopIteration:
            raise TypeError(
                "reduce() of empty iterable with no initial value",
            ) from None
    else:
        value = initial

    for element in it:
        value = function(value, element)

    return value
<|endoftext|>"""
Python polyfills for operator
"""

from __future__ import annotations

import operator
from typing import Any, overload, TYPE_CHECKING, TypeVar
from typing_extensions import TypeVarTuple, Unpack

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


# Most unary and binary operators are handled by BuiltinVariable (e.g., `pos`, `add`)
__all__ = ["attrgetter", "concat", "countOf", "iconcat", "itemgetter", "methodcaller"]


_T = TypeVar("_T")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_Ts = TypeVarTuple("_Ts")
_U = TypeVar("_U")
_U1 = TypeVar("_U1")
_U2 = TypeVar("_U2")
_Us = TypeVarTuple("_Us")


@overload
# pyrefly: ignore [inconsistent-overload]
def attrgetter(attr: str, /) -> Callable[[Any], _U]: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def attrgetter(
    attr1: str, attr2: str, /, *attrs: str
) -> Callable[[Any], tuple[_U1, _U2, Unpack[_Us]]]: ...


# Reference: https://docs.python.org/3/library/operator.html#operator.attrgetter
@substitute_in_graph(operator.attrgetter, is_embedded_type=True)  # type: ignore[arg-type,misc]
def attrgetter(*attrs: str) -> Callable[[Any], Any | tuple[Any, ...]]:
    if len(attrs) == 0:
        raise TypeError("attrgetter expected 1 argument, got 0")

    if any(not isinstance(attr, str) for attr in attrs):
        raise TypeError("attribute name must be a string")

    def resolve_attr(obj: Any, attr: str) -> Any:
        for name in attr.split("."):
            obj = getattr(obj, name)
        return obj

    if len(attrs) == 1:
        attr = attrs[0]

        def getter(obj: Any) -> Any:
            return resolve_attr(obj, attr)

    else:

        def getter(obj: Any) -> tuple[Any, ...]:  # type: ignore[misc]
            return tuple(resolve_attr(obj, attr) for attr in attrs)

    return getter


# Reference: https://docs.python.org/3/library/operator.html#operator.concat
@substitute_in_graph(operator.concat, can_constant_fold_through=True)  # type: ignore[arg-type]
def concat(a: Sequence[_T], b: Sequence[_T2], /) -> Sequence[_T | _T2]:
    return a + b  # type: ignore[operator]


# Reference: https://docs.python.org/3/library/operator.html#operator.countOf
@substitute_in_graph(operator.countOf, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def countOf(a: Iterable[_T], b: _T, /) -> int:
    return sum(it is b or it == b for it in a)


# Reference: https://docs.python.org/3/library/operator.html#operator.iconcat
@substitute_in_graph(operator.iconcat)  # type: ignore[arg-type]
def iconcat(a: Sequence[_T], b: Sequence[_T2], /) -> Sequence[_T | _T2]:
    a += b  # type: ignore[operator]
    return a  # type: ignore[return-value]


@overload
# pyrefly: ignore [inconsistent-overload]
def itemgetter(item: _T, /) -> Callable[[Any], _U]: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def itemgetter(
    item1: _T1, item2: _T2, /, *items: Unpack[_Ts]
) -> Callable[[Any], tuple[_U1, _U2, Unpack[_Us]]]: ...


# Reference: https://docs.python.org/3/library/operator.html#operator.itemgetter
@substitute_in_graph(operator.itemgetter, is_embedded_type=True)  # type: ignore[arg-type,misc]
def itemgetter(*items: Any) -> Callable[[Any], Any | tuple[Any, ...]]:
    if len(items) == 0:
        raise TypeError("itemgetter expected 1 argument, got 0")

    if len(items) == 1:
        item = items[0]

        def getter(obj: Any) -> Any:
            return obj[item]

    else:

        def getter(obj: Any) -> tuple[Any, ...]:  # type: ignore[misc]
            return tuple(obj[item] for item in items)

    return getter


# Reference: https://docs.python.org/3/library/operator.html#operator.methodcaller
@substitute_in_graph(operator.methodcaller, is_embedded_type=True)  # type: ignore[arg-type]
def methodcaller(name: str, /, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
    if not isinstance(name, str):
        raise TypeError("method name must be a string")

    def caller(obj: Any) -> Any:
        return getattr(obj, name)(*args, **kwargs)

    return caller
<|endoftext|>"""
Python polyfills for _io
"""

from __future__ import annotations

import _io

import sys
import warnings

from ..decorators import substitute_in_graph


__all__ = ["text_encoding"]


# Copied from Lib/_pyio.py in the standard library
# pyrefly: ignore [bad-argument-type]
@substitute_in_graph(_io.text_encoding, can_constant_fold_through=True)
def text_encoding(encoding: str | None, stacklevel: int = 2, /) -> str:
    if encoding is None:
        encoding = "utf-8" if sys.flags.utf8_mode else "locale"
        if sys.flags.warn_default_encoding:
            warnings.warn(
                "'encoding' argument not specified.",
                EncodingWarning,
                stacklevel + 1,
            )
    return encoding
<|endoftext|>from collections.abc import Callable
from typing import Any

from torch._C import _fx_map_aggregate, _fx_map_arg
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.node import Node

from ..decorators import substitute_in_graph


@substitute_in_graph(_fx_map_arg, can_constant_fold_through=True)
def map_arg(a: Any, fn: Callable[[Node], Any]) -> Any:
    return map_aggregate(a, lambda x: fn(x) if isinstance(x, Node) else x)


@substitute_in_graph(_fx_map_aggregate, can_constant_fold_through=True)
def map_aggregate(a: Any, fn: Callable[[Any], Any]) -> Any:
    result: Any
    if isinstance(a, tuple):
        it = (map_aggregate(elem, fn) for elem in a)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        result = type(a)(*it) if hasattr(a, "_fields") else tuple(it)
    elif isinstance(a, list):
        result = immutable_list([map_aggregate(elem, fn) for elem in a])
    elif isinstance(a, dict):
        result = immutable_dict([(k, map_aggregate(v, fn)) for k, v in a.items()])
    elif isinstance(a, slice):
        result = slice(
            map_aggregate(a.start, fn),
            map_aggregate(a.stop, fn),
            map_aggregate(a.step, fn),
        )
    else:
        result = fn(a)
    return result


__all__ = [
    "map_arg",
    "map_aggregate",
]
<|endoftext|>"""
This module provides debugging backends for TorchDynamo to help diagnose and troubleshoot
compilation and execution issues. It includes:

Key Debugging Backends:
- eager: Simple pass-through backend that runs models in eager mode
- eager_noexcept: Similar to eager but with additional exception handling
- eager_debug: Adds schema validation checks for custom operators
- aot_eager: Uses AOT Autograd with nop compiler for debugging
- aot_eager_decomp_partition: Uses TorchInductor decompositions for debugging
- torchscript: Compiles using TorchScript for debugging JIT-related issues

Testing and Development Tools:
- Backends for inducing specific errors (compile/runtime/accuracy)
- ExplainOutput class for detailed graph compilation analysis
- Utilities for cross-referencing and mode management
- Tools for graph detail inspection and break reason analysis

These backends are primarily used for:
1. Debugging graph breaks and compilation failures
2. Testing error handling and recovery mechanisms
3. Analyzing performance bottlenecks
4. Validating operator schemas and decompositions
"""

import dataclasses
import functools
import logging
from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any, TYPE_CHECKING

import torch
from functorch.compile import min_cut_rematerialization_partition
from torch import _guards
from torch._dynamo.output_graph import GraphCompileReason
from torch._functorch import config as functorch_config
from torch._functorch.compilers import ts_compile
from torch._inductor.output_code import OutputCode

from .common import aot_autograd
from .registry import CompiledFn, CompilerFn, register_debug_backend as register_backend


if TYPE_CHECKING:
    from torch.fx.node import Target


log = logging.getLogger(__name__)


@register_backend
def eager(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]:
    if kwargs:
        log.warning("eager backend ignoring extra kwargs %s", kwargs)

    if torch._functorch.config.force_autograd_cache:
        from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable

        return GraphModuleSerializableCallable(gm)
    return gm.forward


def make_eager_backend_with_torch_function_mode(
    mode: torch.overrides.TorchFunctionMode,
) -> Callable[..., Any]:
    return make_eager_backend_with_torch_function_modes([mode])


def make_eager_backend_with_torch_function_modes(
    modes: Iterable[torch.overrides.TorchFunctionMode],
) -> Callable[..., Any]:
    """Used to trace HOPs (cond and while) for eager execution, the metadata
    TF mode mutates vars outside of the scope of the HOP, and we can't have graph breaks
    in the HOP, so we need to externally run this mode and not trace it."""
    from contextlib import ExitStack

    def fn(
        gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
    ) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with ExitStack() as stack:
                for mode in modes:
                    stack.enter_context(mode)
                return gm.forward(*args, **kwargs)

        return wrapper

    return fn


@register_backend
def eager_noexcept(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]:
    if kwargs:
        log.warning("eager_noexcept backend ignoring extra kwargs %s", kwargs)

    # This backend is intended to check that dynamo-generated GraphModules
    # do not cause errors.
    def inner(*args: Any) -> Any:
        try:
            return gm(*args)
        except Exception as e:
            raise torch._dynamo.exc.TorchDynamoException(
                "Unexpected exception when running generated GraphModule"
            ) from e

    return inner


@register_backend
def pre_dispatch_eager(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> torch.fx.GraphModule:
    if kwargs:
        log.warning("pre_dispatch_eager backend ignoring extra kwargs %s", kwargs)

    from torch.fx.experimental.proxy_tensor import make_fx

    def runnable_gm(*args: Any) -> Any:
        return torch.fx.Interpreter(gm).run(*args)

    pre_dispatch_gm = make_fx(runnable_gm, pre_dispatch=True)(*fake_tensor_inputs)
    pre_dispatch_gm.print_readable()

    return pre_dispatch_gm


@register_backend
def eager_debug(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]:
    if kwargs:
        log.warning("eager_debug backend ignoring extra kwargs %s", kwargs)

    from torch._subclasses.schema_check_mode import SchemaCheckMode

    # We could add more debugging bits here.
    # Right now, this backend can be used to check for and error on
    # custom dispatcher ops that have incorrect schemas.
    def inner(*args: Any) -> Any:
        with SchemaCheckMode():
            return torch.fx.Interpreter(gm).run(*args)

    return inner


@register_backend(name="ts")  # type: ignore[misc]
def torchscript(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor]
) -> torch.jit.ScriptModule:
    from torch.fx._lazy_graph_module import _unwrap_lazy_graph_module

    return torch.jit.script(_unwrap_lazy_graph_module(gm))


def invoke_subgraph_inner_compiler(
    subgraph: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> Callable[..., Any]:
    """Inner compiler that wraps forward/backward graphs in invoke_subgraph HOP.

    This is used as the fw_compiler/bw_compiler for aot_autograd. When the resulting
    function is traced by make_fx, it emits an invoke_subgraph HOP instead of inlining.
    """
    from torch._dynamo import disable
    from torch._higher_order_ops.invoke_subgraph import invoke_subgraph_infer

    # pyrefly: ignore [deprecated]
    @torch._dynamo.allow_in_graph
    @disable
    def invoke_subgraph_wrapper_unboxed(*operands: Any) -> Any:
        return invoke_subgraph_infer(subgraph, *operands)

    # NB: The direct to unboxed path is broken, you MUST DO THIS

    def invoke_subgraph_wrapper(args: list[Any]) -> Any:
        # `allow_in_graph` is untyped and its `list` branch leaks a
        # `list | fn` union into the inferred return; as the outer decorator
        # here that makes `invoke_subgraph_wrapper_unboxed` look non-callable.
        # pyrefly: ignore [not-callable]
        return invoke_subgraph_wrapper_unboxed(*args)

    invoke_subgraph_wrapper._boxed_call = True  # type: ignore[attr-defined]

    return invoke_subgraph_wrapper


# I cannot say how many times I had to revert to this vibe coded version of
# the code, which worked, and the cleaner versions of the code did not work,
# so I'm leaving this here until we fix the rest of the bugs.
'''
# Counter for unique subgraph names in invoke_subgraph backend
_invoke_subgraph_counter = 0


def invoke_subgraph_inner_compiler_good(
    fx_g: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> Callable[..., Any]:
    """Inner compiler that wraps forward/backward graphs in invoke_subgraph HOP.

    This is used as the fw_compiler/bw_compiler for aot_autograd. When the resulting
    function is traced by make_fx, it emits an invoke_subgraph HOP instead of inlining.
    """
    from torch._higher_order_ops.invoke_subgraph import (
        invoke_subgraph as invoke_subgraph_hop,
    )
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    global _invoke_subgraph_counter
    _invoke_subgraph_counter += 1
    name = f"invoke_subgraph_{_invoke_subgraph_counter}"

    from torch._dynamo import disable

    # Check if fx_g uses boxed calling convention
    fx_g_is_boxed = getattr(fx_g, "_boxed_call", False)

    @disable
    @torch._dynamo.allow_in_graph
    def invoke_subgraph_wrapper_unboxed(*args: Any) -> Any:
        proxy_mode = get_proxy_mode()
        if proxy_mode is not None:
            # When being traced by make_fx, emit invoke_subgraph HOP
            return invoke_subgraph_hop(fx_g, name, *args)  # type: ignore[arg-type]
        else:
            # Normal execution path - call fx_g with proper calling convention
            if fx_g_is_boxed:
                return fx_g(list(args))
            else:
                return fx_g(*args)

    # Wrap to handle boxed arguments (list of args) as expected by AOTAutograd
    def invoke_subgraph_wrapper(args: list[Any]) -> Any:
        return invoke_subgraph_wrapper_unboxed(*args)

    invoke_subgraph_wrapper._boxed_call = True  # type: ignore[attr-defined]
    return invoke_subgraph_wrapper
'''


@register_backend
def invoke_subgraph(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]:
    """Backend that wraps forward/backward graphs in invoke_subgraph HOP when traced by make_fx.

    This backend uses AOTAutograd to partition into forward/backward graphs, then wraps
    each in an invoke_subgraph HOP. This is useful for recursive Dynamo tracing scenarios
    where you want the compiled subgraph to appear as invoke_subgraph HOPs in the outer
    trace rather than being inlined.

    Requires:
    - torch._dynamo.config.force_compile_during_fx_trace = True
      (this implicitly overrides error_on_nested_fx_trace)
    """
    if kwargs:
        log.warning("invoke_subgraph backend ignoring extra kwargs %s", kwargs)

    # Use AOTAutograd to partition into forward/backward
    return aot_autograd(
        fw_compiler=invoke_subgraph_inner_compiler,
        bw_compiler=invoke_subgraph_inner_compiler,
        partition_fn=min_cut_rematerialization_partition,
        keep_inference_input_mutations=True,
    )(gm, fake_tensor_inputs)


@dataclasses.dataclass
class AOTEagerOutputCode(OutputCode):
    """
    An OutputCode that wraps a GraphModule for eager-mode execution.

    This allows non-inductor backends (like aot_eager) to participate in
    the bundled autograd cache and aot_compile serialization flow.
    """

    gm: torch.fx.GraphModule | None = None
    _serialized_gm: bytes | None = dataclasses.field(default=None, init=False)

    def __call__(self, inputs: Any) -> Any:
        if self.gm is None:
            raise AssertionError("gm must not be None")
        return self.gm.forward(inputs)

    def prepare_for_serialization(self) -> None:
        from torch.fx._graph_pickler import GraphPickler, Options

        if self.gm is None:
            raise AssertionError("gm must not be None")
        for node in self.gm.graph.nodes:
            node.meta.pop("nn_module_stack", None)
            node.meta.pop("source_fn_stack", None)
            node.meta.pop("example_value", None)

        self._serialized_gm = GraphPickler.dumps(self.gm, Options(ops_filter=None))
        self.gm = None

    def post_compile(self, *args: Any, **kwargs: Any) -> None:
        if self.gm is None and self._serialized_gm is not None:
            from torch._subclasses import FakeTensorMode
            from torch.fx._graph_pickler import GraphPickler
            from torch.fx.experimental.symbolic_shapes import ShapeEnv
            from torch.fx.graph import _BoxedCodeGen

            fake_mode = FakeTensorMode(shape_env=ShapeEnv())
            gm = GraphPickler.loads(self._serialized_gm, fake_mode)
            if not isinstance(gm, torch.fx.GraphModule):
                raise AssertionError(f"Expected torch.fx.GraphModule, got {type(gm)}")
            self.gm = gm
            self.gm.graph.set_codegen(_BoxedCodeGen())
            self.gm.recompile()
            self._serialized_gm = None

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        pass


# used boxed call to discard inputs when they are no longer needed
def boxed_nop(
    fx_g: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> Callable[..., Any]:
    from torch.fx.graph import _BoxedCodeGen

    # Set the graph to use boxed codegen
    fx_g.graph.set_codegen(_BoxedCodeGen())
    fx_g.recompile()

    if functorch_config.force_autograd_cache or functorch_config.bundled_autograd_cache:
        result = AOTEagerOutputCode(gm=fx_g)
        result._boxed_call = True  # type: ignore[attr-defined]
        return result

    # Wrap the forward method in a function so we can set _boxed_call attribute
    forward_fn = fx_g.forward

    def run(args: Any) -> Any:
        from torch.utils._debug_mode import DebugInterpreter, get_active_debug_mode

        if (
            debug_mode := get_active_debug_mode()
        ) is not None and debug_mode.run_compile_with_interpreter:
            return DebugInterpreter(fx_g, backend="aot_eager").run(*args)
        return forward_fn(args)

    run._boxed_call = True  # type: ignore[attr-defined]
    return run


def boxed_nop_with_mode(
    fx_g: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    mode: torch.overrides.TorchFunctionMode,
) -> Callable[..., Any]:
    from torch.fx.graph import _BoxedCodeGen

    # Set the graph to use boxed codegen
    fx_g.graph.set_codegen(_BoxedCodeGen())
    fx_g.recompile()

    # Create a wrapper that runs with the mode
    forward_fn = fx_g.forward

    def run(args: Any) -> Any:
        with mode:
            return forward_fn(args)

    run._boxed_call = True  # type: ignore[attr-defined]
    return run


def fake_crossref_boxed_nop(
    fx_g: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    ignore_op_fn: Callable[[torch._ops.OpOverload], bool] | None = None,
) -> Callable[..., Any]:
    from torch.fx.graph import _BoxedCodeGen

    # Set the graph to use boxed codegen
    fx_g.graph.set_codegen(_BoxedCodeGen())
    fx_g.recompile()

    # Create a wrapper that runs with the mode
    forward_fn = fx_g.forward

    def run(args: Any) -> Any:
        with torch._subclasses.CrossRefFakeMode(ignore_op_fn):
            return forward_fn(args)

    run._boxed_call = True  # type: ignore[attr-defined]
    return run


def ignore_builtins(op: torch._ops.OpOverload) -> bool:
    return op.namespace in ("aten", "prims", "prim")


def get_nop_func() -> Callable[
    [torch.fx.GraphModule, list[torch.Tensor]], Callable[..., Any]
]:
    if not torch._functorch.config.fake_tensor_crossref:
        return boxed_nop
    elif torch._functorch.config.fake_tensor_crossref == "all":
        return fake_crossref_boxed_nop
    else:
        if torch._functorch.config.fake_tensor_crossref != "custom_ops":
            raise AssertionError(
                f"Expected fake_tensor_crossref to be 'custom_ops', "
                f"got {torch._functorch.config.fake_tensor_crossref!r}"
            )
        return functools.partial(fake_crossref_boxed_nop, ignore_op_fn=ignore_builtins)


# Useful for debugging purpose
# aot_eager uses AOT Autograd backend with nop compiler. It is helpful in debugging.
def aot_eager(
    gm: torch.fx.GraphModule,
    fake_tensor_inputs: list[torch.Tensor],
    fw_compiler: Callable[..., Any] | None = None,
    bw_compiler: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    return aot_autograd(
        fw_compiler=fw_compiler or boxed_nop,
        bw_compiler=bw_compiler or boxed_nop,
        partition_fn=min_cut_rematerialization_partition,
        keep_inference_input_mutations=True,
    )(gm, fake_tensor_inputs, **kwargs)


register_backend(name="aot_eager", compiler_fn=aot_eager)

aot_eager_default_partitioner = aot_autograd(
    fw_compiler=boxed_nop, keep_inference_input_mutations=True
)
register_backend(
    name="aot_eager_default_partitioner", compiler_fn=aot_eager_default_partitioner
)


# Uses TorchInductor AOT Autograd decomps and partitioner to isolate aot vs
# inductor problems.
# aot_eager_decomp_partition just replaces the inductor compiler with nop to help
# isolate inductor vs aot_eager errors
def aot_eager_decomp_partition(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]:
    if kwargs:
        log.warning(
            "aot_eager_decomp_partition backend ignoring extra kwargs %s", kwargs
        )

    from torch._inductor.compiler_bisector import CompilerBisector

    config_patches = {"unlift_effect_tokens": True}
    if bisect_changes := CompilerBisector.get_config_change(
        "aot_eager_decomp_partition"
    ):
        config_patches.update(bisect_changes)  # type: ignore[arg-type]

    with functorch_config.patch(config_patches):
        return aot_autograd(
            # these are taken from memory_efficient_fusion()
            fw_compiler=get_nop_func(),
            bw_compiler=get_nop_func(),
            # NB: lambda here is to delay import of inductor
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),
            partition_fn=functools.partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
        )(gm, fake_tensor_inputs)


register_backend(
    name="aot_eager_decomp_partition", compiler_fn=aot_eager_decomp_partition
)


# aot_eager_decomp_partition_with_mode is similar as aot_eager_decomp_partition,
# except that it takes a TorchDispatchMode mode and run the fw/bw in the mode
def aot_eager_decomp_partition_with_mode(
    gm: torch.fx.GraphModule,
    fake_tensor_inputs: list[torch.Tensor],
    mode: Any,
    **kwarg: Any,
) -> Callable[..., Any]:
    return aot_autograd(
        # these are taken from memory_efficient_fusion()
        fw_compiler=functools.partial(boxed_nop_with_mode, mode=mode),
        bw_compiler=functools.partial(boxed_nop_with_mode, mode=mode),
        # NB: lambda here is to delay import of inductor
        decompositions=lambda: import_module(
            "torch._inductor.compile_fx"
        ).select_decomp_table(),
        partition_fn=functools.partial(
            min_cut_rematerialization_partition, compiler="inductor"
        ),
    )(gm, fake_tensor_inputs)


register_backend(
    name="aot_eager_decomp_partition_with_mode",
    compiler_fn=aot_eager_decomp_partition_with_mode,  # type: ignore[arg-type]
)


def aot_eager_decomp_partition_crossref(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]:
    # if the config is set, respect it, otherwise only test custom_ops.
    # custom_op bad metas always manifest as an error whereas aten will only sometimes.
    # by default, use the less noisy option
    config_val = (
        "custom_ops"
        if not functorch_config.fake_tensor_crossref
        else functorch_config.fake_tensor_crossref
    )
    with functorch_config.patch(fake_tensor_crossref=config_val):
        return aot_eager_decomp_partition(gm, fake_tensor_inputs, **kwargs)


register_backend(
    name="aot_eager_decomp_partition_crossref",
    compiler_fn=aot_eager_decomp_partition_crossref,
)


# AOT Autograd with torchscript backend. Default partitioner.
# aot_ts uses torchscript backend. We can use this with both nnc and nvfuser
# by using the relevant fuser with torch.jit.fuser(...)
aot_ts = aot_autograd(fw_compiler=ts_compile)
register_backend(name="aot_ts", compiler_fn=aot_ts)

# These buggy backends are used for inducing bugs so that we can test
# our repro extraction / minifier scripts


class ReluCompileError(Exception):
    pass


class TestingOnlyCompileError(Exception):
    pass


@register_backend
def relu_compile_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target is torch.relu:
            raise ReluCompileError
    return gm


@register_backend
def relu_runtime_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target is torch.relu:
            node.target = torch._assert
            node.args = (False, "ReluRuntimeError")
    gm.recompile()
    return gm


@register_backend
def relu_accuracy_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target is torch.relu:
            node.target = torch.add
            node.args = (node.args[0], 1)
    gm.recompile()

    return gm


@register_backend
def non_leaf_compile_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule:
    # Require at least one non-trivial thing in the graph,
    # see https://github.com/pytorch/pytorch/issues/102898
    for node in gm.graph.nodes:
        if node.op == "call_function":
            break
    else:
        return gm
    for t in example_inputs:
        if not t.is_leaf:
            raise TestingOnlyCompileError
    return gm


@dataclasses.dataclass
class ExplainOutput:
    """
    This is the output of :func:`torch._dynamo.explain()`
    There is no reason to create this class directly.
    """

    graphs: list[torch.fx.GraphModule]
    graph_count: int
    graph_break_count: int
    break_reasons: list[GraphCompileReason]
    op_count: int
    ops_per_graph: list[list["Target"]] | None = None
    out_guards: list[_guards.Guard] | None = None
    compile_times: str | None = None

    def __str__(self) -> str:
        output = f"Graph Count: {self.graph_count}\n"
        output += f"Graph Break Count: {self.graph_break_count}\n"
        output += f"Op Count: {self.op_count}\n"

        output += "Break Reasons:\n"
        for idx, break_reason in enumerate(self.break_reasons):
            output += f"  Break Reason {idx + 1}:\n"
            output += f"    Reason: {break_reason.reason}\n"
            output += "    User Stack:\n"
            for frame_summary in break_reason.user_stack:
                output += f"      {frame_summary}\n"

        if self.ops_per_graph is not None:
            output += "Ops per Graph:\n"
            for idx, ops in enumerate(self.ops_per_graph):
                output += f"  Ops {idx + 1}:\n"
                for op in ops:
                    output += f"    {op}\n"

        if self.out_guards is not None:
            output += "Out Guards:\n"
            for i, guard in enumerate(self.out_guards):
                output += f"  Guard {i + 1}:\n"
                output += f"    {str(guard)}"

        if self.compile_times is not None:
            output += f"Compile Times: {self.compile_times}\n"
        return output


def _explain_graph_detail(
    gm: torch.fx.GraphModule,
    graphs: list[torch.fx.GraphModule],
    op_count: int,
    ops_per_graph: list[list["Target"]],
    break_reasons: list[GraphCompileReason],
) -> tuple[
    torch.fx.GraphModule,
    list[torch.fx.GraphModule],
    int,
    list[list["Target"]],
    list[GraphCompileReason],
]:
    """
    This function is a utility which processes a torch.fx.GraphModule and
    accumulates information about its ops, graph breaks, and other details. It
    is intended to be used by the ExplainWithBackend class and
    `torch._dynamo.explain()` to provide details from Dynamo's graph capture.

    Parameters:
        gm (torch.fx.GraphModule): The GraphModule to be processed.
        graphs (list): A list that accumulates all the GraphModules processed.
        op_count (int): The total count of operations in all GraphModules processed so far.
        ops_per_graph (list): A list that accumulates the operations of each GraphModule.
        break_reasons (list): A list that accumulates the reasons for breaks in each GraphModule.

    Returns:
        tuple: A tuple containing the processed GraphModule, the updated lists of graphs,
               operations per graph, and break reasons, and the updated operation count.
    """
    graphs.append(gm)
    ops = [node.target for node in gm.graph.nodes if node.op == "call_function"]
    op_count += len(ops)
    ops_per_graph.append(ops)
    if gm.compile_subgraph_reason.graph_break:  # type: ignore[union-attr]
        break_reasons.append(gm.compile_subgraph_reason)  # type: ignore[arg-type]

    return gm, graphs, op_count, ops_per_graph, break_reasons


class ExplainWithBackend:
    """
    This class is intended to be used as a backend for `torch.compile`. It is
    composable with other backends. When used in this way, it accumulates
    information about graph breaks, ops, and other info and provides a string
    representation summarizing this information.

    Attributes:
        backend (str): The name of the backend to use for optimization.
        graphs (list): A list of the graphs captured by TorchDynamo.
        op_count (int): The total number of operations in all optimized graphs.
        break_reasons (list): A list of graph break reasons with stack traces.

    Example Usage:
        def fn(x):
            x = torch.sigmoid(x)
            return x

        torch._dynamo.reset()
        eb = ExplainWithBackend("inductor")
        optimized_fn = torch.compile(fn, backend=eb)
        result = optimized_fn(torch.randn(5))
        print(eb.output())
    """

    def __init__(self, backend: CompilerFn | str) -> None:
        from .registry import lookup_backend

        self.backend = lookup_backend(backend)
        self.graphs: list[torch.fx.GraphModule] = []
        self.op_count = 0
        self.break_reasons: list[GraphCompileReason] = []

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> CompiledFn:
        ops_per_graph: list[list[Target]] = []
        gm, self.graphs, self.op_count, _, self.break_reasons = _explain_graph_detail(
            gm, self.graphs, self.op_count, ops_per_graph, self.break_reasons
        )
        return self.backend(gm, example_inputs)

    def output(self) -> ExplainOutput:
        graph_count = len(self.graphs)
        output = ExplainOutput(
            self.graphs,
            graph_count,
            graph_count - 1,
            self.break_reasons,
            self.op_count,
        )

        return output
<|endoftext|>"""
This module implements TorchDynamo's backend registry system for managing compiler backends.

The registry provides a centralized way to register, discover and manage different compiler
backends that can be used with torch.compile(). It handles:

- Backend registration and discovery through decorators and entry points
- Lazy loading of backend implementations
- Lookup and validation of backend names
- Categorization of backends using tags (debug, experimental, etc.)

Key components:
- CompilerFn: Type for backend compiler functions that transform FX graphs
- _BACKENDS: Registry mapping backend names to entry points
- _COMPILER_FNS: Registry mapping backend names to loaded compiler functions
- _BACKEND_TAGS: Registry mapping backend names to their tags

Example usage:
    @register_backend
    def my_compiler(fx_graph, example_inputs):
        # Transform FX graph into optimized implementation
        return compiled_fn

    # Use registered backend
    torch.compile(model, backend="my_compiler")

The registry also supports discovering backends through setuptools entry points
in the "torch_dynamo_backends" group. Example:
```
setup.py
---
from setuptools import setup

setup(
    name='my_torch_backend',
    version='0.1',
    packages=['my_torch_backend'],
    entry_points={
        'torch_dynamo_backends': [
            # name = path to entry point of backend implementation
            'my_compiler = my_torch_backend.compiler:my_compiler_function',
        ],
    },
)
```
```
my_torch_backend/compiler.py
---
def my_compiler_function(fx_graph, example_inputs):
    # Transform FX graph into optimized implementation
    return compiled_fn
```
Using `my_compiler` backend:
```
import torch

model = ...  # Your PyTorch model
optimized_model = torch.compile(model, backend="my_compiler")
```
"""

import functools
import logging
from collections.abc import Callable, Sequence
from importlib.metadata import EntryPoint
from typing import Any, Protocol

import torch
from torch import fx


log = logging.getLogger(__name__)


class CompiledFn(Protocol):
    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]: ...


CompilerFn = Callable[[fx.GraphModule, list[torch.Tensor]], CompiledFn]

_BACKENDS: dict[str, EntryPoint | None] = {}
_COMPILER_FNS: dict[str, CompilerFn] = {}
_BACKEND_TAGS: dict[str, tuple[str, ...]] = {}
_default_backend: str | CompilerFn = "inductor"


def register_backend(
    compiler_fn: CompilerFn | None = None,
    name: str | None = None,
    tags: Sequence[str] = (),
) -> Callable[..., Any]:
    """
    Decorator to add a given compiler to the registry to allow calling
    `torch.compile` with string shorthand.  Note: for projects not
    imported by default, it might be easier to pass a function directly
    as a backend and not use a string.

    Args:
        compiler_fn: Callable taking a FX graph and fake tensor inputs
        name: Optional name, defaults to `compiler_fn.__name__`
        tags: Optional set of string tags to categorize backend with
    """
    if compiler_fn is None:
        # @register_backend(name="") syntax
        return functools.partial(register_backend, name=name, tags=tags)  # type: ignore[return-value]
    if not callable(compiler_fn):
        raise AssertionError(f"compiler_fn must be callable, got {type(compiler_fn)}")
    name = name or compiler_fn.__name__
    if name in _COMPILER_FNS:
        raise AssertionError(f"duplicate name: {name}")
    if name not in _BACKENDS:
        _BACKENDS[name] = None
    _COMPILER_FNS[name] = compiler_fn
    _BACKEND_TAGS[name] = tuple(tags)
    return compiler_fn


register_debug_backend = functools.partial(register_backend, tags=("debug",))
register_experimental_backend = functools.partial(
    register_backend, tags=("experimental",)
)


def lookup_backend(compiler_fn: str | CompilerFn) -> CompilerFn:
    """Expand backend strings to functions"""
    if isinstance(compiler_fn, str):
        if compiler_fn not in _BACKENDS:
            _lazy_import()
        if compiler_fn not in _BACKENDS:
            import difflib

            from ..exc import InvalidBackend

            suggestions = difflib.get_close_matches(compiler_fn, list_backends(), n=2)
            raise InvalidBackend(name=compiler_fn, suggestions=suggestions)

        if compiler_fn not in _COMPILER_FNS:
            entry_point = _BACKENDS[compiler_fn]
            if entry_point is not None:
                register_backend(compiler_fn=entry_point.load(), name=compiler_fn)
        compiler_fn = _COMPILER_FNS[compiler_fn]
    return compiler_fn


# NOTE: can't type this due to public api mismatch; follow up with dev team
def list_backends(exclude_tags=("debug", "experimental")) -> list[str]:  # type: ignore[no-untyped-def]
    """
    Return valid strings that can be passed to:

        torch.compile(..., backend="name")
    """
    _lazy_import()
    exclude_tags_set = set(exclude_tags or ())

    backends = [
        name
        for name in _BACKENDS
        if not exclude_tags_set.intersection(_BACKEND_TAGS.get(name, ()))
    ]
    return sorted(backends)


@functools.cache
def _lazy_import() -> None:
    from .. import backends
    from ..utils import import_submodule

    import_submodule(backends)

    from ..repro.after_dynamo import dynamo_minifier_backend

    if dynamo_minifier_backend is None:
        raise AssertionError("dynamo_minifier_backend failed to load")

    _discover_entrypoint_backends()


@functools.cache
def _discover_entrypoint_backends() -> None:
    # importing here so it will pick up the mocked version in test_backends.py
    from importlib.metadata import entry_points

    group_name = "torch_dynamo_backends"
    eps = entry_points(group=group_name)
    # pyrefly: ignore [bad-index]
    eps_dict = {name: eps[name] for name in eps.names}
    for backend_name in eps_dict:
        _BACKENDS[backend_name] = eps_dict[backend_name]


def _is_registered_backend(compiler_fn: CompilerFn) -> bool:
    """
    Check if the given compiler function is a registered backend.
    Custom backends (user-provided callables not in the registry) return False.
    """
    # Ensure backends are loaded
    _lazy_import()

    # Check if it's directly a registered backend function
    if compiler_fn in _COMPILER_FNS.values():
        return True

    if isinstance(compiler_fn, torch._TorchCompileInductorWrapper):
        return True

    # _TorchCompileWrapper wraps either a registered backend or a custom callable
    if isinstance(compiler_fn, torch._TorchCompileWrapper):
        return compiler_fn.compiler_fn in _COMPILER_FNS.values()

    return False


def set_default_backend(backend: str | CompilerFn | None) -> None:
    """Set the default backend used by torch.compile when no backend is explicitly specified.

    Pass None to reset to the default ("inductor").
    """
    global _default_backend
    if backend is None:
        _default_backend = "inductor"
        return
    if not isinstance(backend, str) and not callable(backend):
        raise TypeError(f"backend must be a string or callable, got {type(backend)}")
    _default_backend = backend


def get_default_backend() -> str | CompilerFn:
    """Return the current default backend for torch.compile."""
    return _default_backend
<|endoftext|>"""
This module provides the TorchInductor backend integration for TorchDynamo.

TorchInductor is a compiler backend that generates optimized code for both CPU and GPU.
This module lazily imports and registers the TorchInductor compiler to avoid loading it
into memory when it is not being used. This helps reduce memory overhead when using
other backends.

The inductor backend can be used with torch.compile():
    model = torch.compile(model, backend="inductor")
"""

from typing import Any

from torch._dynamo import register_backend
from torch._dynamo.utils import dynamo_timed


@register_backend
def inductor(*args: Any, **kwargs: Any) -> Any:
    with dynamo_timed("inductor_import", log_pt2_compile_event=True):
        # do import here to avoid loading inductor into memory when it is not used
        # The AsyncCompile subproc pool can be slow to start, so warm it up as early
        # as possible.
        from torch._inductor.async_compile import maybe_warm_pool

        maybe_warm_pool()

        from torch._inductor.compile_fx import compile_fx

    return compile_fx(*args, **kwargs)
<|endoftext|>"""
This module provides common utilities and base classes for TorchDynamo backends.

Key components:
- AotAutograd: Base class for implementing AOT (Ahead-of-Time) autograd backends
- Backend utilities for handling:
  - Fake tensor conversion
  - Device/dtype detection from inputs
  - Memory efficient fusion
  - Graph flattening
  - Common compiler configurations

The utilities here are used by various backend implementations to handle
common operations and provide consistent behavior across different backends.
AOT autograd functionality is particularly important as it enables ahead-of-time
optimization of both forward and backward passes.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import Any, TYPE_CHECKING
from typing_extensions import ParamSpec, Required, TypedDict, TypeVar, Unpack
from unittest.mock import patch

import torch
from torch._dynamo import disable
from torch._dynamo.exc import TensorifyScalarRestartAnalysis
from torch._dynamo.utils import counters, defake, flatten_graph_inputs
from torch._functorch.aot_autograd import (
    aot_module_simplified,
    SerializableAOTDispatchCompiler,
)
from torch.utils._python_dispatch import _disable_current_modes


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch._inductor.compile_fx import CompilerConfigExtra
    from torch._ops import OpOverload


log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class AotAutogradKwargs(TypedDict, total=False):
    """Keyword arguments accepted by AotAutograd and forwarded verbatim to
    aot_module_simplified. decompositions may also be a zero-arg thunk returning
    the decomposition table (a workaround for circular imports); AotAutograd
    resolves it to the table before forwarding."""

    fw_compiler: Required[Callable[..., Any]]
    bw_compiler: Callable[..., Any] | None
    inference_compiler: Callable[..., Any] | None
    partition_fn: Callable[..., Any]
    decompositions: (
        dict[OpOverload, Callable[..., Any]]
        | Callable[[], dict[OpOverload, Callable[..., Any]]]
    )
    keep_inference_input_mutations: bool
    compiler_config_extra: CompilerConfigExtra | None
    ignore_shape_env: bool
    disable_functionalization: bool
    pre_grad_passes: Callable[..., Any] | None
    compile_region_name: str | None


class AotAutograd:
    def __init__(self, **kwargs: Unpack[AotAutogradKwargs]) -> None:
        self.__name__ = "compiler_fn"
        self.kwargs: AotAutogradKwargs = kwargs

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[Any], **kwargs: Any
    ) -> Callable[..., Any]:
        if kwargs:
            log.warning("aot_autograd-based backend ignoring extra kwargs %s", kwargs)

        if any(isinstance(x, (list, tuple, dict)) for x in example_inputs):
            return flatten_graph_inputs(
                gm,
                example_inputs,
                self,
            )

        # Hack to get around circular import problems with aot_eager_decomp_partition
        decompositions = self.kwargs.get("decompositions")
        if callable(decompositions):
            self.kwargs["decompositions"] = decompositions()

        # NB: don't delete counter increment
        counters["aot_autograd"]["total"] += 1

        def wrap_bw_compiler(bw_compiler_fn: Callable[P, R]) -> Callable[..., R]:
            def _wrapped_bw_compiler(*args: P.args, **kwargs: P.kwargs) -> R:
                # Note [Wrapping bw_compiler in disable]
                # The two disables here:
                # - stop TorchDynamo from trying to compile the bw_compiler function itself
                # - stop TorchDynamo from trying to compile our the generated backwards pass bw_compiler produces

                return disable(
                    disable(
                        bw_compiler_fn, reason="do not trace backward compiler function"
                    )(*args, **kwargs),  # type: ignore[misc]
                    reason="do not trace generated backwards pass",
                )

            _wrapped_bw_compiler._is_wrapped_bw_compiler = (  # pyrefly: ignore [missing-attribute]
                True
            )
            return _wrapped_bw_compiler

        bw_compiler = self.kwargs.get("bw_compiler") or self.kwargs["fw_compiler"]

        if isinstance(bw_compiler, SerializableAOTDispatchCompiler):
            if not getattr(bw_compiler.compiler_fn, "_is_wrapped_bw_compiler", False):
                bw_compiler.compiler_fn = wrap_bw_compiler(bw_compiler.compiler_fn)
        elif not getattr(bw_compiler, "_is_wrapped_bw_compiler", False):
            bw_compiler = wrap_bw_compiler(bw_compiler)

        self.kwargs["bw_compiler"] = bw_compiler
        self.kwargs["inference_compiler"] = (
            self.kwargs.get("inference_compiler") or self.kwargs["fw_compiler"]
        )

        from functorch.compile import nop
        from torch._inductor.debug import enable_aot_logging

        # debug asserts slow down compile time noticeably,
        # So only default them on when the aot_eager backend is used.
        if self.kwargs.get("fw_compiler", None) is nop:
            patch_config: contextlib.AbstractContextManager[Any] = patch(
                "functorch.compile.config.debug_assert", True
            )
        else:
            patch_config = contextlib.nullcontext()

        try:
            # NB: NOT cloned!
            with enable_aot_logging(), patch_config:
                # The decompositions thunk is resolved to a concrete table by the
                # callable check above, but pyrefly cannot narrow that through the
                # TypedDict subscript assignment, so the spread looks ill-typed.
                cg = aot_module_simplified(
                    gm,
                    example_inputs,
                    **self.kwargs,  # pyrefly: ignore [bad-argument-type]
                )
                counters["aot_autograd"]["ok"] += 1
                return disable(cg, reason="do not trace AOT-compiled graph")
        except TensorifyScalarRestartAnalysis:
            raise
        except Exception:
            counters["aot_autograd"]["not_ok"] += 1
            raise


def aot_autograd(**kwargs: Unpack[AotAutogradKwargs]) -> AotAutograd:
    return AotAutograd(**kwargs)


def mem_efficient_fusion_kwargs(use_decomps: bool) -> dict[str, Any]:
    from functorch.compile import (
        default_decompositions,
        min_cut_rematerialization_partition,
        ts_compile,
    )

    kwargs = {
        # these are taken from memory_efficient_fusion()
        "fw_compiler": ts_compile,
        "bw_compiler": ts_compile,
        "partition_fn": min_cut_rematerialization_partition,
    }

    if use_decomps:
        # pyrefly: ignore [bad-typed-dict-key]
        kwargs["decompositions"] = default_decompositions

    return kwargs


def fake_tensor_unsupported(fn: Callable[[Any, list[Any], Any], R]) -> Any:
    """
    Decorator for backends that need real inputs.  We swap out fake
    tensors for zero tensors.
    """

    @functools.wraps(fn)
    def wrapper(model: Any, inputs: Any, **kwargs: Any) -> Any:
        with _disable_current_modes():
            inputs = list(map(defake, inputs))
            return fn(model, inputs, **kwargs)  # type: ignore[call-arg]

    return wrapper


def device_from_inputs(example_inputs: Iterable[Any]) -> torch.device:
    for x in example_inputs:
        if isinstance(x, torch.Tensor):
            return x.device
    return torch.device("cpu")  # Default fallback


def dtype_from_inputs(example_inputs: Iterable[Any]) -> torch.dtype:
    for x in example_inputs:
        if isinstance(x, torch.Tensor):
            return x.dtype
    return torch.float32  # Default fallback
<|endoftext|>"""
This module provides TVM backend integration for TorchDynamo.

Apache TVM is a deep learning compiler framework that can optimize and execute
models on various hardware backends. The backend compiles graphs through TVM's
relax frontend and can be used with torch.compile():

    model = torch.compile(model, backend="tvm")

Tuning and target selection are configured through a TVM pipeline:

    pipeline = tvm.relax.get_pipeline("static_shape_tuning", target="llvm", total_trials=2000)
    model = torch.compile(model, backend="tvm", options={"pipeline": pipeline})
"""

import importlib.util
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import torch
from torch import fx

from .common import fake_tensor_unsupported
from .registry import register_backend


@register_backend
@fake_tensor_unsupported  # type: ignore[arg-type]
def tvm(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    options: MappingProxyType[str, Any] | None = None,
) -> Callable[..., Any]:
    try:
        from tvm.relax.frontend.torch import relax_dynamo  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "Please install apache-tvm to use the tvm backend. "
            "See https://tvm.apache.org/docs/install/index.html for instructions."
        ) from e

    pipeline = options.get("pipeline", None) if options else None
    return relax_dynamo(pipeline=pipeline)(gm, example_inputs)


def has_tvm() -> bool:
    # avoid the heavy tvm import just to check availability
    return importlib.util.find_spec("tvm") is not None
<|endoftext|>"""
This module implements CUDA graphs support for TorchDynamo backends.

CUDA graphs allow for capturing and replaying GPU operations, which can significantly
reduce CPU overhead in GPU-accelerated PyTorch models. This module provides:

- CUDA graph creation and management for both forward and backward passes
- Input mutation detection and handling
- Device compatibility checking
- Stack trace management for debugging
- Integration with TorchInductor's cudagraph trees

The backend supports two main modes:
1. cudagraphs: Full CUDA graph support with both forward and backward pass optimization
2. cudagraphs_inner: Lower-level CUDA graph implementation used for benchmarking

Key components:
- CudagraphsBackend: Main backend class for CUDA graph integration
- Mutation detection utilities to ensure graph safety
- Device mapping and compatibility checks
- Stack trace collection for debugging
"""

import functools
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.fx
from torch._dynamo import config
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.debugging import boxed_nop
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,
    check_multiple_devices_or_any_cpu_nodes,
    format_default_skip_message,
    get_mutation_stack_trace,
    get_placeholder_info,
    log_cudagraph_skip_and_bump_counter,
)
from torch._inductor.utils import (
    BoxedBool,
    count_tangents,
    get_first_incompatible_cudagraph_node,
    num_fw_fixed_arguments,
    output_node,
)
from torch.multiprocessing.reductions import StorageWeakRef

from .registry import register_backend


log = logging.getLogger(__name__)


def find_input_mutations(g: torch.fx.Graph) -> set[int]:
    def meta_fk(meta: dict[str, Any]) -> Any:
        return meta["val"] if "val" in meta else meta["fake_result"]

    inputs = defaultdict(set)
    input_idx = 0
    mutated_inputs = set()
    for n in g.nodes:
        if n.op == "placeholder":
            if isinstance(meta_fk(n.meta), torch.Tensor):
                inputs[StorageWeakRef(meta_fk(n.meta)._typed_storage())].add(input_idx)
            input_idx += 1
        elif n.op == "call_function":
            if not hasattr(n.target, "_schema"):
                continue

            schema = n.target._schema
            for i, arg in enumerate(schema.arguments):
                if i < len(n.args):
                    argument = n.args[i]
                else:
                    if arg.name not in n.kwargs:
                        continue
                    argument = n.kwargs[arg.name]
                mut_arg = False
                if arg.alias_info:
                    if arg.alias_info.is_write:
                        mut_arg = True
                if mut_arg:
                    # TODO: not correct for args that contain tensors in a struct
                    # like list
                    mutated_inputs |= inputs[
                        StorageWeakRef(meta_fk(argument.meta)._typed_storage())
                    ]

        # TODO: error on unrecognized nodes
    return mutated_inputs


def get_device_node_mapping(
    gm: torch.fx.GraphModule,
) -> dict[torch.device, torch.fx.Node]:
    device_node_mapping: dict[torch.device, torch.fx.Node] = {}
    for n in gm.graph.nodes:
        t = n.meta.get("val", None)
        if isinstance(t, torch.Tensor) and t.device not in device_node_mapping:
            device_node_mapping[t.device] = n
    return device_node_mapping


def check_for_mutation_ignore_cuda_graph_managed_tensor(
    aot_model: torch.fx.GraphModule, num_fixed: int
) -> str | None:
    mutation_indices = find_input_mutations(aot_model.graph) - set(range(num_fixed))
    if not mutation_indices:
        return None

    placeholders = get_placeholder_info(aot_model.graph)
    return get_mutation_stack_trace(placeholders, mutation_indices)


def check_for_skip(aot_model: torch.fx.GraphModule, num_fixed: int) -> str | None:
    if not config.cudagraph_backend_support_input_mutation:
        if mut_skip := check_for_mutation_ignore_cuda_graph_managed_tensor(
            aot_model, num_fixed
        ):
            return mut_skip

    if skip := check_multiple_devices_or_any_cpu_nodes(
        get_device_node_mapping(aot_model)
    ):
        return skip

    if node := get_first_incompatible_cudagraph_node(aot_model):
        return format_default_skip_message(f"incompatible op ({node.name})")

    return None


def get_device_index(gm: torch.fx.GraphModule) -> int:
    device = next(iter(get_device_node_mapping(gm)))
    if device.type != "cuda":
        raise AssertionError(f"Expected CUDA device, got {device.type}")
    return device.index


def get_stack_traces(gm: torch.fx.GraphModule) -> list[str | None]:
    output = output_node(gm)
    if len(output.args) != 1:
        raise AssertionError(
            f"Expected output node to have 1 arg, got {len(output.args)}"
        )
    args = output.args[0]
    if not hasattr(args, "__iter__"):
        return []
    return [
        (arg.stack_trace if isinstance(arg, torch.fx.node.Node) else None)
        for arg in args  # type: ignore[union-attr]
    ]


def cudagraphs(dynamo_model: torch.fx.GraphModule, dynamo_inputs: Sequence[Any]) -> Any:
    from torch._inductor.cudagraph_trees import cudagraphify_impl

    do_cudagraphs = BoxedBool(True)
    boxed_device_index = BoxedDeviceIndex(None)

    def forward_cudagraphs(
        aot_model: torch.fx.GraphModule,
        aot_inputs: list[Any],
        is_inference: bool = False,
    ) -> Any:
        interp = boxed_nop(aot_model, aot_inputs)
        fixed = num_fw_fixed_arguments(len(dynamo_inputs), len(aot_inputs))
        if skip_msg := check_for_skip(aot_model, fixed):
            BoxedBool.disable(do_cudagraphs)
            log_cudagraph_skip_and_bump_counter(
                f"skipping cudagraphs due to {skip_msg}"
            )
            return interp

        boxed_device_index.set(get_device_index(aot_model))
        out = cudagraphify_impl(
            interp,
            aot_inputs,
            range(fixed),
            device_index=boxed_device_index.value,
            is_backward=False,
            is_inference=is_inference,
            stack_traces=get_stack_traces(aot_model),
            placeholders=get_placeholder_info(aot_model.graph),
            mutated_input_idxs=find_input_mutations(aot_model.graph),
        )
        out._boxed_call = True  # type: ignore[attr-defined]
        return out

    def backward_cudagraphs(
        aot_model: torch.fx.GraphModule, aot_inputs: list[Any]
    ) -> Any:
        interp = boxed_nop(aot_model, aot_inputs)
        if not do_cudagraphs:
            return aot_model

        fixed = count_tangents(aot_model)
        if skip_msg := check_for_skip(aot_model, fixed):
            log_cudagraph_skip_and_bump_counter(
                f"skipping cudagraphs due to {skip_msg}"
            )

            # See [Backward Generation Handling]
            device_idx = boxed_device_index.value
            if device_idx is None:
                device_idx = 0  # Default to device 0 if not set
            manager = torch._inductor.cudagraph_trees.get_manager(
                device_idx, create_if_none_exists=False
            )
            if manager is None:
                raise AssertionError("cudagraph manager must exist for backward")

            def fn(inputs: list[Any]) -> Any:
                # pyrefly: ignore [missing-attribute]
                manager.set_to_running_backward()
                return aot_model(inputs)

            fn._boxed_call = True  # type: ignore[attr-defined]
            return fn

        out = cudagraphify_impl(
            interp,
            aot_inputs,
            range(fixed),
            device_index=get_device_index(aot_model),
            is_backward=True,
            is_inference=False,
            stack_traces=get_stack_traces(aot_model),
            placeholders=get_placeholder_info(aot_model.graph),
            mutated_input_idxs=find_input_mutations(aot_model.graph),
        )
        out._boxed_call = True  # type: ignore[attr-defined]
        return out

    aot_cudagraphs = aot_autograd(
        fw_compiler=forward_cudagraphs,
        bw_compiler=backward_cudagraphs,
        inference_compiler=functools.partial(forward_cudagraphs, is_inference=True),
        keep_inference_input_mutations=torch._dynamo.config.cudagraph_backend_keep_input_mutation,
    )
    return aot_cudagraphs(dynamo_model, dynamo_inputs)


class CudagraphsBackend:
    compiler_name = "cudagraphs"

    @staticmethod
    def reset() -> None:
        from torch._inductor.cudagraph_trees import reset_cudagraph_trees

        reset_cudagraph_trees()

    @staticmethod
    def __call__(
        model: torch.fx.GraphModule, inputs: Sequence[Any], **kwargs: Any
    ) -> Any:
        if kwargs:
            log.warning("cudagraphs backend ignoring extra kwargs %s", kwargs)
        return cudagraphs(model, inputs)


# aot_cudagraphs only applies CUDA graphs to the graph.  It is also helpful
# for debugging and can serve as a perf baseline.
register_backend(name="cudagraphs", compiler_fn=CudagraphsBackend())


def cudagraphs_inner(
    model: Callable[..., Any],
    inputs: Sequence[Any],
    copy_outputs: bool = True,
    copy_inputs: bool = True,
) -> Callable[..., Sequence[Any]]:
    """This isn't registered as a backend, but is used in some benchmarks"""
    if not isinstance(inputs, (list, tuple)):
        raise AssertionError(
            f"Expected inputs to be a list or tuple, got {type(inputs)}"
        )
    if copy_inputs:
        static_inputs = [torch.zeros_like(x) for x in inputs]
    else:
        static_inputs = list(inputs)

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    def run(*new_inputs: Any) -> Sequence[Any]:
        if len(static_inputs) != len(new_inputs):
            raise AssertionError(
                f"Expected {len(static_inputs)} inputs, got {len(new_inputs)}"
            )
        if copy_inputs:
            for dst, src in zip(static_inputs, new_inputs):
                dst.copy_(src)
        graph.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs

    return run
<|endoftext|>"""
This module implements distributed training optimizations for TorchDynamo backends.

It provides functionality to optimize models wrapped in DistributedDataParallel (DDP)
by intelligently splitting compiled graphs to align with DDP's gradient synchronization
boundaries. Key features include:

- Graph partitioning based on parameter bucket sizes
- Optimization of allreduce operations for distributed training
- Support for parameter ignoring and buffer handling
- Submodule compilation and management
- Debugging utilities for distributed training

The main component is the DDPOptimizer class, which handles graph splitting and
recompilation to enable efficient distributed training while maintaining the benefits
of compilation.
"""

import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from unittest import mock

import torch
from torch import fx
from torch._dynamo.backends.registry import CompiledFn, CompilerFn
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
from torch._logging import trace_structured
from torch.fx.node import Node


if TYPE_CHECKING:
    from torch._functorch._aot_autograd.schemas import ViewAndMutationMeta


# Regular log messages should go through 'log'.
# ddp_graph_log is a separate artifact logger reserved for dumping graphs.
# See docs/source/logging.rst for more info.
log = logging.getLogger(__name__)
ddp_graph_log = torch._logging.getArtifactLogger(__name__, "ddp_graphs")


def args_str(args: Any) -> str:
    # a debug helper
    if torch.is_tensor(args):
        return f"T[{args.shape}]"
    elif isinstance(args, tuple):
        return f"tuple({', '.join([args_str(x) for x in args])})"
    elif isinstance(args, list):
        return f"list({', '.join([args_str(x) for x in args])})"
    else:
        return str(args)


@dataclass
class Bucket:
    size: int = 0
    params: list[str] = field(default_factory=list)
    nodes: list[fx.Node] = field(default_factory=list)

    # param_ids is just used for unit testing
    param_ids: list[int] = field(default_factory=list)

    # keep track of any buckets that were extended for logging purposes
    opcount_increased_to_capture_external_output: int = 0
    paramsize_before_opcount_increase: int = 0


def bucket_has_external_output(bucket: Bucket) -> bool:
    nodes_in_bucket = set()
    # we want to iterate in reverse order, but clumsi-luckily the bucket.nodes list was already created backwards
    # so we don't reverse it here
    for node in bucket.nodes:
        # assume node.op != output, since those are filtered in the original iteration
        nodes_in_bucket.add(node)
        for user in node.users:
            if user not in nodes_in_bucket:
                return True
    return False


def pretty_print_buckets(buckets: list[Bucket], bucket_bytes_cap: int) -> None:
    headers = ("Index", "Size (b)", "Param Names")
    rows: list[tuple[int | None, int | None, str]] = []
    # pyrefly: ignore [implicit-any]
    extended_buckets = []
    for idx, bucket in enumerate(reversed(buckets)):
        if len(bucket.params) > 0:
            rows.append((idx, bucket.size, bucket.params[0]))
            rows.extend((None, None, param) for param in bucket.params[1:])
        if bucket.opcount_increased_to_capture_external_output > 0:
            extended_buckets.append(
                (
                    idx,
                    bucket.opcount_increased_to_capture_external_output,
                    bucket.size - bucket.paramsize_before_opcount_increase,
                )
            )

    if rows:
        log.info(
            "\nDDPOptimizer used bucket cap %s and created %d buckets. Enable debug logs for detailed bucket info.",
            bucket_bytes_cap,
            len(buckets),
        )

        if extended_buckets:
            log.warning(
                "Some buckets were extended beyond their requested parameter capacities"
                " in order to ensure each subgraph has an output node, required for fx graph partitioning."
                " This can be the case when a subgraph would have only contained nodes performing inplace mutation,"
                " and returning no logical outputs. This should not be a problem, unless it results in too few graph"
                " partitions for optimal DDP performance."
            )

        try:
            from tabulate import tabulate

            log.debug(
                "\nDDPOptimizer produced the following bucket assignments:\n%s",
                tabulate(rows, headers=headers, tablefmt="simple_grid"),
            )

            if extended_buckets:
                log.warning(
                    "DDPOptimizer extended these buckets to ensure per-subgraph output nodes:\n%s",
                    tabulate(
                        extended_buckets,
                        headers=("Index", "Extra Ops", "Extra Param Size (b)"),
                        tablefmt="simple_grid",
                    ),
                )
        except ImportError:
            log.debug(
                "Please `pip install tabulate` in order to display ddp bucket sizes and diagnostic information."
            )
    else:
        log.debug("DDPOptimizer captured no parameters and did not split this graph.")


def has_higher_order_op(gm: fx.GraphModule) -> bool:
    # Check if there is a higher order op in the graph
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            maybe_param = getattr(gm, node.target)
            if isinstance(maybe_param, torch.fx.GraphModule):
                return True
    return False


def propagate_metadata(orig_gm: fx.GraphModule, split_gm: fx.GraphModule) -> None:
    # Only propagate to partition submodules, not to HOP body subgraphs (e.g.
    # wrap_body_*) which are also hoisted as top-level children by split_module
    # and carry their own metadata (e.g. _checkpoint_context_fn for SAC).
    partition_names = split_gm.meta["partition_names"]
    for name, module in split_gm.named_modules():
        if name in partition_names:
            # TODO: add split id to CompileId: https://github.com/pytorch/tlparse/pull/83/files#r1880649384
            module.meta = orig_gm.meta
            module._param_name_to_source = orig_gm._param_name_to_source


def propagate_dynamo_source(orig_gm: fx.GraphModule, split_gm: fx.GraphModule) -> None:
    name_to_dynamo_source = {}
    for node in orig_gm.graph.find_nodes(op="placeholder"):
        name_to_dynamo_source[node.name] = node._dynamo_source

    partition_names = split_gm.meta["partition_names"]
    for name, module in split_gm.named_modules():
        if name in partition_names:
            for node in module.graph.find_nodes(op="placeholder"):
                # non-placeholder in original_gm may become placeholder in submodules
                node._dynamo_source = name_to_dynamo_source.get(node.name)


class DDPOptimizerContext:
    def __init__(self) -> None:
        self.curr_bucket: int = -1
        self.metadata_per_bucket: list[ViewAndMutationMeta] = []


# compile each of the partitioned submodules using the user-provided compiler
class SubmodCompiler(torch.fx.interpreter.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        compiler: CompilerFn,
        fake_mode: torch._subclasses.fake_tensor.FakeTensorMode,
        **compiler_configs: Any,
    ) -> None:
        super().__init__(module)
        self.compiler = compiler
        self.compiler_configs = compiler_configs
        self.fake_mode = fake_mode
        # See Note [DDPOptimizer and fw_metadata]
        ctx = torch._guards.TracingContext.try_get()
        if ctx is not None:
            ctx.ddp_optimizer_ctx = DDPOptimizerContext()

    def compile_submod(
        self, input_mod: fx.GraphModule, args: list[torch.Tensor], kwargs: Any
    ) -> Any:
        """
        Compile the submodule,
        using a wrapper to make sure its output is always a tuple,
        which is required by AotAutograd based compilers
        """
        if len(kwargs) != 0:
            raise AssertionError("We assume only args for these modules")

        class WrapperModule(torch.nn.Module):
            def __init__(
                self, submod: Callable[..., Any], unwrap_singleton_tuple: bool
            ) -> None:
                super().__init__()
                self.submod = submod
                self.unwrap_singleton_tuple = unwrap_singleton_tuple

            def forward(self, *args: Any) -> Any:
                x = self.submod(*args)
                # TODO(whc)
                # for some reason the isinstance check is necessary if I split one node per submod
                # - even though I supposedly wrapped the output in a tuple in those cases, the real
                # compiled module was still returning a tensor
                if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
                    return x[0]
                return x

        unwrap_singleton_tuple = False
        for sn in input_mod.graph.nodes:
            if sn.op == "output":
                if not isinstance(sn.args[0], tuple):
                    unwrap_singleton_tuple = True
                    sn.args = (sn.args,)

        input_mod.recompile()
        input_mod.compile_subgraph_reason = GraphCompileReason(  # type: ignore[assignment]
            "DDPOptimizer intentional graph-break (See Note [DDPOptimizer])."
            " Set `torch._dynamo.config.optimize_ddp = False` to disable.",
            [
                # it's close to useless to get a real stacktrace here, and quite verbose.
                traceback.FrameSummary(__file__, 0, "DDPOptimizer"),
            ],
        )

        wrapper = WrapperModule(
            self.compiler(input_mod, args, **self.compiler_configs),
            unwrap_singleton_tuple,
        )
        return wrapper

    # Note:
    #
    # The way distributed works today around fake tensors can be somewhat confusing.
    # Some of these codepaths are shared in both runtime, and compile time. The presence
    # of a fake_mode, read off of fake tensor inputs, dictates how we will operate.
    #
    # A few things to keep in mind:
    #
    # 1) We invoke `compile_submod` with a real module. The output of that gets stored
    # on the graph via `self.module.add_submodule(n.target, compiled_submod_real)`.
    #
    # 2) When running a call_module targeted node, if we have a fake_mode, we fakify the
    # module we got from self.fetch_attr(n.target). Regardless of fake_mode, we then execute it.
    #
    # 3) Fake tensors should always be around during compile time.
    #
    # 4) Fake tensors should never be around at runtime.
    #
    # 5) We end up with a compilation mode that takes a real submodule and fake tensors,
    # to match what aot_autograd expects. See Note: [Fake Modules and AOTAutograd]
    def run_node(self, n: Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        new_args = []
        if not self.fake_mode:
            raise AssertionError("fake_mode must be set")
        for arg in args:
            if isinstance(arg, torch.Tensor) and not isinstance(
                arg, torch._subclasses.FakeTensor
            ):
                new_args.append(torch._dynamo.utils.to_fake_tensor(arg, self.fake_mode))
            else:
                new_args.append(arg)

        log.debug("run_node %s, %s got args %s", n.op, n.target, args_str(args))
        if not isinstance(args, tuple):
            raise AssertionError(f"Expected args to be a tuple, got {type(args)}")
        if not isinstance(kwargs, dict):
            raise AssertionError(f"Expected kwargs to be a dict, got {type(kwargs)}")

        if n.op == "call_module":
            real_mod = self.fetch_attr(str(n.target))
            if self.fake_mode:
                curr_submod = deepcopy_to_fake_tensor(real_mod, self.fake_mode)
            else:
                curr_submod = real_mod

            ddp_graph_log.debug("\n---%s graph---\n%s", n.target, curr_submod.graph)

            # When calling the compiler on the submod, inputs (new_args) are expected to
            # be FakeTensors already since Dynamo would have made them FakeTensors in the
            # non-DDP flow.  However, the parameters are _not_ expected to be FakeTensors,
            # since this wrapping happens during compilation

            # Note: Returning Fake Tensors on First AOT Autograd Call
            #
            # Inductor will optimize strides of outputs when it deems it profitable.
            # For instance, converting to channels last. When we split the graph here
            # into multiple inductor compilations, we need to make sure that the
            # output strides of one compilation is appropriately passed to the subsequent
            # compilations. However, the mapping from inductor output to dynamo output
            # is non-trivial due to aot_autograd's deduping, de-aliasing, mutation, re-writing,
            # subclass handling, etc. In order to replay all this logic we set a flag such that
            # the first invocation of inductor in aot_autograd will return Fake Tensors with
            # appropriate strides. Then, all of aot autograd's runtime logic is replayed.
            # This gives us the appropriately strided outputs here which will reflect runtime strides.

            class FakeifyFirstAOTInvocationGuard:
                def __init__(self) -> None:
                    self.tc = torch._guards.TracingContext.try_get()
                    if not self.tc:
                        raise AssertionError("TracingContext must be set")
                    self.tc.fakify_first_call = True

                def __del__(self) -> None:
                    self.tc.fakify_first_call = False  # type: ignore[union-attr]

            # For aot_eager and other backends, tracing context is not set
            has_tracing_context = torch._guards.TracingContext.try_get() is not None
            if has_tracing_context:
                g = FakeifyFirstAOTInvocationGuard()  # noqa: F841

            from torch._dynamo.utils import counters

            init = counters["aot_autograd"]["total"]
            compiled_submod_real = self.compile_submod(real_mod, new_args, kwargs)

            # TODO - better way of doing this?
            # Only aot autograd handles fakifying first call
            invoked_aot_autograd = init != counters["aot_autograd"]["total"]

            # We update the original (outer) graph with a call into the compiled module
            # instead of the uncompiled one.
            self.module.delete_submodule(n.target)  # type: ignore[operator]
            n.target = "compiled_" + n.target  # type: ignore[operator]
            self.module.add_submodule(n.target, compiled_submod_real)  # type: ignore[operator]

            # Finally, we have to produce inputs for use compiling the next submodule,
            # and these need to be FakeTensors, so we execute the module under fake_mode
            # Because parameters are not fake we patch fake tensor mode to allow non fake inputs
            with (
                self.fake_mode,
                mock.patch.object(self.fake_mode, "allow_non_fake_inputs", True),
            ):
                if has_tracing_context and invoked_aot_autograd:
                    tracing_ctx = torch._guards.TracingContext.try_get()
                    if tracing_ctx is None:
                        raise AssertionError("TracingContext must not be None")
                    # DDPOptimizer maintains 1 dynamo graph -> N AOT graphs
                    # Dynamo only has 1 tracing context, so it needs to maintain all N AOT metadata instances
                    ddp_ctx = tracing_ctx.ddp_optimizer_ctx
                    if ddp_ctx is None:
                        raise AssertionError("ddp_optimizer_ctx must not be None")
                    if tracing_ctx.fw_metadata is None:
                        raise AssertionError("fw_metadata must not be None")
                    ddp_ctx.curr_bucket += 1
                    ddp_ctx.metadata_per_bucket.append(tracing_ctx.fw_metadata)

                    out = compiled_submod_real(*new_args, **kwargs)
                    # output should be fake or subclass
                    if not all(
                        (not isinstance(t, torch.Tensor) or type(t) is not torch.Tensor)
                        for t in (out if isinstance(out, (list, tuple)) else [out])
                    ):
                        raise AssertionError(
                            "Output should be fake or subclass, not plain torch.Tensor"
                        )
                    return out
                else:
                    return curr_submod(*new_args, **kwargs)
        else:
            # placeholder or output nodes don't need to get compiled, just executed
            return getattr(self, n.op)(n.target, new_args, kwargs)


class DDPOptimizer:
    """Note [DDPOptimizer]
    DDPOptimizer applies when dynamo compiles models wrapped in DistributedDataParallel (DDP),
    breaking the dynamo graph into chunks to compile separately, with the breaks aligning to
    the boundaries of gradient-allreduce buckets chosen by DDP.

    Background/Motivation
     - DDP uses allreduce collectives to synchronize partial gradients computed on different workers
     - DDP groups gradient allreduces into 'buckets' to optimize communication efficiency of all-reduce
     - Parameters grouped into buckets are assumed to be adjacent in time, so they become ready
       at around the same time during backward and thus can share the same allreduce efficiently
     - Allreduces must overlap with backward compute for optimal training performance
     - DDP schedules allreduces using 'hooks' fired from the c++ autograd engine in pytorch, which
       operates when individual grads become 'ready'
     - Dynamo+AOTAutograd produces a single fused graph that runs 'atomically' from the perspective of the
       autograd engine, such that all gradients become 'ready' at the same time.  Hooks fire after the whole
       fused backward function executes, preventing any overlap of compute and communication

    Algorithm
     - DDPOptimizer starts off with an FX graph traced by dynamo which represents forward.  It can traverse
       this graph in reverse order to determine the true order that gradients will become ready during backward.
     - Parameter sizes are counted in reverse order, up to a bucket size limit, at which point a new bucket is started
       and a graph break introduced
     - Each of the subgraphs is compiled by the compiler provided to dynamo by the user, and then fused back together
       into an outer module that is returned to the user

    Notes
     - It would be better to enforce (by adding an API to DDP) that the bucket splits chosen here are used by DDP,
       and that DDP does not need to detect or optimize bucket order by observing execution at runtime, as it does
       in eager.
     - If Dynamo can't capture a whole graph for the portion of the model wrapped by DDP, this algorithm will currently
       produce splits that do not necessarily align with the buckets used by DDP.  This should result in performance
       degradation approaching the baseline case where graph-splits are not used, but not worse.
     - If the backend compiler fails to compile a single subgraph, it will execute eagerly despite the rest of the
       subgraphs being compiled
     - DDP has a 'parameters_and_buffers_to_ignore' field, which DDPOptimizer attempts to honor by reading markers
       left by DDP on individual parameters.  In cases where other transformations, such as reparameterization, are
       also used, the ignore markers could be lost.  If DDPOptimizer fails to ignore a parameter ignored by DDP,
       it is not catastrophic but could impact performance by choosing sub-optimal bucket splits.
     - DDPOptimizer always ignores all buffers, regardless of their ignore flag, since buffers do not require gradients,
       and therefore aren't allreduced by DDP.  (They are broadcast during forward, but this is not covered by
       DDPOptimizer)

    Debugging
     - Generally, it is easiest to debug DDPOptimizer in a single process program, using pdb.
     - In many cases, the log messages are helpful (they show bucket size assignments)-
       just set TORCH_LOGS env to include any of 'dynamo', 'distributed', or 'dist_ddp'.
     - See `benchmarks/dynamo/distributed.py` for a simple harness that will run a toy model or a torchbench model
       in a single process (or with torchrun, in multiple processes)

    Args:
        bucket_bytes_cap (int): Controls the size of buckets, in bytes, used to determine graphbreaks.  Should be
            set to match the equivalent parameter on the original DDP module.

        backend_compile_fn (callable): A dynamo compiler function, to be invoked to compile each subgraph.

        first_bucket_cap (int): Controls the size of the first bucket.  Should match DDP's first bucket cap.  DDP
            special-cases the first bucket size since it is sometimes optimal to start a small allreduce early.

    """

    def __init__(
        self,
        bucket_bytes_cap: int,
        backend_compile_fn: CompilerFn,
        first_bucket_cap: int | None = None,
    ) -> None:
        if first_bucket_cap is not None:
            self.first_bucket_cap = first_bucket_cap
        elif torch.distributed.is_available():
            # this constant comes from C10D lib which is not always built
            self.first_bucket_cap = torch.distributed._DEFAULT_FIRST_BUCKET_BYTES
        else:
            self.first_bucket_cap = bucket_bytes_cap

        self.bucket_bytes_cap = bucket_bytes_cap
        if self.first_bucket_cap > self.bucket_bytes_cap:
            raise AssertionError(
                "First bucket should be smaller/equal to other buckets to get comms warmed up ASAP"
            )

        self.backend_compile_fn = backend_compile_fn

    def _ignore_parameter(self, parameter: torch.nn.Parameter) -> bool:
        return hasattr(parameter, "_ddp_ignored") and parameter._ddp_ignored

    def add_param(self, bucket: Bucket, param: torch.nn.Parameter, name: str) -> None:
        bucket.size += param.untyped_storage().nbytes()
        bucket.params.append(name)
        bucket.param_ids.append(id(param))

    def add_module_params_to_bucket(
        self,
        mod: torch.nn.Module,
        bucket: Bucket,
        processed_modules: set[torch.nn.Module],
        prefix: str,
    ) -> None:
        processed_modules.add(mod)
        for name, param in mod.named_parameters():
            if param.requires_grad and not self._ignore_parameter(param):
                self.add_param(bucket, param, f"{prefix}_{name}")

    def add_param_args(self, bucket: Bucket, node: fx.Node) -> None:
        for arg in node.args:
            if not isinstance(arg, torch.fx.node.Node):
                continue
            if arg.op != "placeholder":
                continue
            param = arg.meta["example_value"]
            if (
                isinstance(param, torch.nn.Parameter)
                and param.requires_grad
                and not self._ignore_parameter(param)
            ):
                self.add_param(bucket, param, str(arg.target))

    def compile_fn(
        self,
        gm: fx.GraphModule,
        example_inputs: list[torch.Tensor],
        **compiler_configs: Any,
    ) -> CompiledFn:
        """
        Implements graph splitting, first determining a set of buckets by counting
        parameter sizes in reverse graph order, then invoking the user/backend compiler
        to compile each subgraph. Finally, stitches compiled graphs into one graphmodule
        and returns its callable.
        """
        # 1: compute the partition map according to DDP bucket logic
        buckets = [Bucket()]  # (size, param_names)
        processed_modules: set[torch.nn.Module] = set()
        for node in reversed(gm.graph.nodes):
            if node.op in ("output", "placeholder"):
                continue

            if (
                buckets[0].size >= self.bucket_bytes_cap
                or len(buckets) == 1
                and buckets[0].size >= self.first_bucket_cap
            ):
                if bucket_has_external_output(buckets[0]):
                    buckets.insert(0, Bucket())
                else:
                    # continue building this bucket past the point of filling its parameter capacity,
                    # to increase chances it contains at least one node that is either a global output or
                    # passed as input to a subsequent graph

                    if buckets[0].opcount_increased_to_capture_external_output == 0:
                        buckets[0].paramsize_before_opcount_increase = buckets[0].size
                    buckets[0].opcount_increased_to_capture_external_output += 1

            if node.op == "call_function":
                self.add_param_args(buckets[0], node)

            elif node.op == "call_module":
                target_mod = gm.get_submodule(node.target)
                if target_mod not in processed_modules:
                    self.add_module_params_to_bucket(
                        target_mod, buckets[0], processed_modules, node.target
                    )
            elif node.op == "call_method":
                if isinstance(node.args[0].target, str):
                    target_mod = None
                    try:
                        target_mod = gm.get_submodule(node.args[0].target)
                    except AttributeError:
                        pass
                    if target_mod is not None and target_mod not in processed_modules:
                        self.add_module_params_to_bucket(
                            target_mod, buckets[0], processed_modules, node.target
                        )
                    # This handles situations like  tmp = torch.mm(x, self.weight.t())
                    # t: "f32[512, 512]" = l_self_seq_2_weight.t();  l_self_seq_2_weight = None
                    # tmp: "f32[512, 512]" = torch.mm(input_2, t);  input_2 = t = None
                    self.add_param_args(buckets[0], node)

            elif node.op == "get_attr":
                maybe_param = getattr(gm, node.target)
                if (
                    isinstance(maybe_param, torch.nn.Parameter)
                    and maybe_param.requires_grad
                    and not self._ignore_parameter(maybe_param)
                ):
                    self.add_param(buckets[0], maybe_param, node.target)

            # All nodes have to be mapped to a bucket, even if they don't have their own params
            # Ignored params still end up in buckets, we just don't count them towards the capacity
            buckets[0].nodes.append(node)

        if len(buckets) > 1 and buckets[0].size == 0:
            # we collected a small preamble graph with ops that don't include parameters, fuse it back
            buckets[1].nodes.extend(buckets[0].nodes)
            if len(buckets[0].params) != 0:
                raise AssertionError("Params should be empty if size is 0")
            del buckets[0]

        # stash buckets for testing/debugging purposes
        self.buckets = buckets
        pretty_print_buckets(buckets, self.bucket_bytes_cap)

        if len(buckets) == 1:
            # bypass split/fuse logic if there is only one bucket
            return self.backend_compile_fn(gm, example_inputs, **compiler_configs)

        # 2: partition the graphmodule according to bucket capacity
        partition_map = {}
        for idx, b in enumerate(buckets):
            for node in b.nodes:
                partition_map[node] = idx

        split_gm = fx.passes.split_module.split_module(
            gm,
            None,  # type: ignore[arg-type]
            lambda node: partition_map[node],
        )

        # See note [Assumption on Dynamo Metadata]
        propagate_dynamo_source(gm, split_gm)
        propagate_metadata(gm, split_gm)

        debug_str = (
            f"\n---orig graph---\n{gm.graph}\n"
            + f"\n---split graph---\n{split_gm.graph}\n"
        )
        for name, module in split_gm.named_modules():
            if "." not in name and len(name):
                # only print the submod graphs, not their children
                debug_str += f"\n---{name} graph---\n{module.graph}\n"
        debug_str += "\n---------------\n"
        ddp_graph_log.debug(debug_str)

        trace_structured(
            "optimize_ddp_split_graph",
            payload_fn=lambda: split_gm.print_readable(print_output=False),
        )
        for name, module in split_gm.named_modules():
            if "." not in name and len(name):
                trace_structured(
                    "optimize_ddp_split_child",
                    lambda: {"name": name},
                    payload_fn=lambda: module.print_readable(print_output=False),
                )

        fake_mode = detect_fake_mode(example_inputs)
        if fake_mode is None:
            fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        submod_compiler = SubmodCompiler(
            split_gm, self.backend_compile_fn, fake_mode, **compiler_configs
        )
        with torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():
            submod_compiler.run(*example_inputs)
        split_gm.recompile()

        ddp_graph_log.debug(
            "\n---final graph---\n%s\n---------------\n", split_gm.graph
        )
        return split_gm
<|endoftext|>import logging
from collections.abc import Callable
from typing import Any

import torch
from functorch.compile import make_boxed_func
from torch import fx

from ..backends.common import aot_autograd
from .registry import CompiledFn, register_backend, register_experimental_backend


log = logging.getLogger(__name__)


@register_experimental_backend
def openxla_eval(
    model: fx.GraphModule, fake_tensor_inputs: list[torch.Tensor]
) -> CompiledFn:
    return xla_backend_helper(model, fake_tensor_inputs, boxed=False)


def openxla_eval_boxed(
    model: fx.GraphModule, fake_tensor_inputs: list[torch.Tensor]
) -> Callable[..., Any]:
    return xla_backend_helper(model, fake_tensor_inputs, boxed=True)


def xla_backend_helper(
    model: fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], boxed: bool = False
) -> Callable[..., Any]:
    try:
        import torch_xla.core.dynamo_bridge as bridge
    except ImportError as e:
        raise ImportError(
            "Please follow the instruction in https://github.com/pytorch/xla#pytorchxla to install torch_xla"
        ) from e

    compiled_graph = None

    def fwd(*args: torch.Tensor) -> Any:
        nonlocal model
        nonlocal compiled_graph
        if compiled_graph is None:
            compiled_graph = bridge.extract_compiled_graph(model, args)
            del model
        return compiled_graph(*args)

    return make_boxed_func(fwd) if boxed else fwd


openxla = aot_autograd(
    fw_compiler=openxla_eval_boxed,
)
register_backend(name="openxla", compiler_fn=openxla)
<|endoftext|>"""
This provides an abstract class which parametrizes over an "output code" concept
for Inductor.  Intuitively, this represents the compiled callable which Inductor
produces which you can call to get optimized code.  However, this callable
has some other capabilities:

- It is serializable, so you can save/load this product from disk without
  having to do compilation again.

- (When using remote cache) it is addressable, so you can save just a key
  which you can use to load this product from remote cache later.

This class is abstract because we have several different implementations of
serialized format:

- Python wrapper (the default)

- AOTInductor (this produces ABI stable binaries which work across PyTorch
  versions)

"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
from functools import partial
from typing import Any, cast, Protocol, TYPE_CHECKING, TypeAlias

import torch
from torch._custom_class_base import CustomClassBase
from torch._dynamo.utils import counters, get_runtime_metrics_context
from torch._guards import compile_context, CompileContext
from torch._higher_order_ops.wrap import inductor_compiled_code
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,
    cudagraph_trees_clone_live_user_visible_outputs,
    CudagraphCachedInfo,
    CudagraphMetadata,
    get_input_storage_mutation_info,
    get_input_storage_mutation_reason,
    get_partition_cudagraph_metadata,
    get_placeholder_info,
    log_cudagraph_skip_and_bump_counter,
)
from torch._inductor.freezing_utils import has_frozen_params, is_frozen_param
from torch._inductor.utils import (
    _unstable_customized_partition_wrapper,
    align_inputs_from_check_idxs,
    BoxedBool,
    CUDAGraphWrapperMetadata,
    GraphPartitionMap,
    InputType,
    is_gpu,
    output_node,
    set_tracing_context_output_strides,
)
from torch.fx._graph_pickler import _node_metadata_key_filter_safe, _ops_filter_safe
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from . import config
from .runtime.autotune_cache import AutotuneCacheBundler


if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Callable, Sequence

    from torch._inductor import metrics
    from torch._inductor.graph import GraphLowering
    from torch._library.fake_class_registry import FakeScriptObject
    from torch.export.pt2_archive._package_weights import Weights

    from .compile_fx import _CompileFxKwargs
    from .triton_bundler import TritonBundle

    # Boxed calling convention: a single list of inputs in, graph outputs out.
    # Inductor's compiled-graph entry points and cudagraph wrappers all take
    # exactly this shape, so we name it rather than repeating Callable[..., Any].
    _BoxedCallable: TypeAlias = Callable[[Sequence[InputType]], object]

    class CompiledFnRunner(Protocol):
        """Runner emitted by the Python wrapper when graph_partition is on.

        codegen writes a Runner class (see codegen/wrapper.py) whose partitions
        hold the per-partition callables and whose recursively_apply_fns rewrites
        each partition in place (e.g. wrapping it in a cudagraph).
        """

        partitions: list[Callable[..., Any]]

        def recursively_apply_fns(self, fns: Sequence[Callable[..., Any]]) -> None: ...


log = logging.getLogger(__name__)


@dataclasses.dataclass
class OutputCode:
    # TODO: Remove underscores here

    # None if the output is not remote cacheable
    _fx_graph_cache_key: str | None = dataclasses.field(default=None, init=False)
    _fx_graph_cache_debug_lines: list[str] | None = dataclasses.field(
        default=None, init=False
    )

    # How long it took to compile this OutputCode, end to end
    _time_taken_ns: int | None = dataclasses.field(default=None, init=False)

    def __call__(self, inputs: Sequence[Any]) -> Any:
        raise NotImplementedError(type(self))

    def prepare_for_serialization(self) -> None:
        raise NotImplementedError(type(self))

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        raise NotImplementedError(type(self))

    # TODO: Get rid of this
    def set_triton_bundle(self, triton_bundle: Any) -> None:
        raise NotImplementedError(type(self))


_StrideExprStr: TypeAlias = str


# copy_ fails when trying to write to tensors with memory overlap,
# for expanded dimensions (a dimension which used to have size 1 -> ?)
# we can select one element from that dimension and write to it
# to achieve writing to all values of that dimension of the input tensor
def get_expanded_dims(t: torch.Tensor) -> list[int]:
    if not isinstance(t, torch.Tensor):
        # pyrefly: ignore [bad-return]
        return None
    return [i for i in range(t.ndim) if t.stride(i) == 0 and t.size(i) != 1]


def index_expanded_dims(t: torch.Tensor, expanded_dims: list[int]) -> torch.Tensor:
    for expanded_dim in expanded_dims:
        t = torch.ops.aten.slice(t, expanded_dim, 0, 1)
    return t


# Bytewise copy of src's reachable strided extent into dst, starting from each
# tensor's data_ptr. Requires matching sizes/strides/dtype. Use when dst
# self-overlaps and a regular strided copy_ would be ambiguous: the overlap
# pattern reproduces in dst because the underlying bytes are copied verbatim.
def copy_strided_storage_(dst: torch.Tensor, src: torch.Tensor) -> None:
    if dst.dtype != src.dtype:
        raise AssertionError(f"dtype mismatch: dst={dst.dtype} src={src.dtype}")
    if tuple(dst.size()) != tuple(src.size()):
        raise AssertionError(
            f"size mismatch: dst={tuple(dst.size())} src={tuple(src.size())}"
        )
    if tuple(dst.stride()) != tuple(src.stride()):
        raise AssertionError(
            f"stride mismatch: dst={tuple(dst.stride())} src={tuple(src.stride())}"
        )
    if dst.numel() == 0:
        return
    if not (all(st >= 0 for st in src.stride())):
        raise AssertionError("copy_strided_storage_ requires non-negative strides")
    elem = src.element_size()
    nbytes = (sum((s - 1) * st for s, st in zip(src.size(), src.stride())) + 1) * elem
    src_off = src.storage_offset() * elem
    dst_off = dst.storage_offset() * elem
    dst.untyped_storage()[dst_off : dst_off + nbytes].copy_(
        src.untyped_storage()[src_off : src_off + nbytes]
    )


def maybe_handle_backward_generation(
    compiled_graph: CompiledFxGraph,
    boxed_forward_device_index: BoxedDeviceIndex | None,
) -> None:
    if compiled_graph.current_callable is None:
        raise AssertionError("compiled_graph.current_callable must not be None")
    is_backward = compiled_graph.fx_kwargs["is_backward"]

    # See [Backward Generation Handling]
    # if cudagraph'd the forward and set the device, we need to let the cudagraph manager
    # know we are running the backward even if we will not run it in cudagraphs
    if is_backward and config.triton.cudagraph_trees:
        if boxed_forward_device_index is None:
            raise AssertionError("boxed_forward_device_index must not be None")
        if boxed_forward_device_index.value is None:
            raise AssertionError("boxed_forward_device_index.value must not be None")
        compiled_graph_callable = compiled_graph.current_callable

        manager = torch._inductor.cudagraph_trees.get_manager(
            boxed_forward_device_index.value, create_if_none_exists=False
        )
        # should already exist from forward
        if manager is None:
            raise AssertionError("CUDAGraph manager must not be None")

        def compiled_artifact(new_inputs: Sequence[InputType]) -> object:
            manager.set_to_running_backward()  # type: ignore[union-attr]
            return compiled_graph_callable(new_inputs)

        compiled_graph.current_callable = compiled_artifact


def prepare_cudagraph_post_compile(
    compiled_graph: CompiledFxGraph,
    example_inputs: Sequence[InputType],
    boxed_forward_device_index: BoxedDeviceIndex | None,
) -> None:
    if not config.triton.cudagraph_trees:
        # Force specialize all inputs so that CUDA graphs will work
        for t in example_inputs:
            if isinstance(t, torch.SymInt):
                int(t)  # guard

    is_inference = compiled_graph.fx_kwargs["is_inference"]
    is_backward = compiled_graph.fx_kwargs["is_backward"]
    if boxed_forward_device_index is not None and not is_inference and not is_backward:
        boxed_forward_device_index.set(next(iter(compiled_graph.device_idxs)))


def cudagraph_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    constants: dict[str, torch.Tensor | type],
    boxed_forward_device_index: BoxedDeviceIndex | None,
) -> None:
    """
    Checks for any reasons not to run cudagraphs and then
    runs it on compiled_graph.
    Mutates the `compiled_graph.current_callable` and `cudagraphs`
    """
    from torch._inductor.compiler_bisector import CompilerBisector

    if compiled_graph.current_callable is None:
        raise AssertionError("compiled_graph.current_callable must not be None")
    if compiled_graph.cudagraph_info is None:
        raise AssertionError("compiled_graph.cudagraph_info must not be None")
    cached_info = compiled_graph.cudagraph_info
    cudagraph_fail_reasons = cached_info.cudagraph_fail_reasons
    is_inference = compiled_graph.fx_kwargs["is_inference"]
    is_backward = compiled_graph.fx_kwargs["is_backward"]

    # Check if bisector wants to disable cudagraphs for this graph
    if CompilerBisector.disable_subsystem("inductor", "cudagraphs"):
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)
        log_cudagraph_skip_and_bump_counter("skipping cudagraphs due to bisector")
        return

    if not cudagraph_fail_reasons:
        fx_kwargs = compiled_graph.fx_kwargs
        static_input_idxs = fx_kwargs["static_input_idxs"]

        placeholders = cached_info.placeholders
        stack_traces = cached_info.stack_traces
        if stack_traces is None:
            raise AssertionError(
                "stack_traces should not be None in cudagraph_post_compile"
            )
        user_visible_output_idxs = getattr(cached_info, "user_visible_output_idxs", ())

        prepare_cudagraph_post_compile(
            compiled_graph, example_inputs, boxed_forward_device_index
        )

        current_callable = compiled_graph.current_callable
        if current_callable is None:
            raise AssertionError("current_callable must not be None")
        # Filter to only tensor constants (exclude opaque value type classes)
        tensor_constants = {
            k: v for k, v in constants.items() if isinstance(v, torch.Tensor)
        }

        device_index = next(iter(compiled_graph.device_idxs))
        cudagraphify_kwargs = dict(
            device_index=device_index,
            stack_traces=stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=tuple(tensor_constants.values()),
            placeholders=placeholders,
            mutated_input_idxs=tuple(compiled_graph.mutated_input_idxs),
            kernel_free_cudagraph=compiled_graph.kernel_free_cudagraph,
            user_visible_output_idxs=tuple(user_visible_output_idxs),
        )

        policy = config.cudagraph_policy
        if policy is not None:
            compiled_graph.current_callable = policy.cudagraphify(
                current_callable,
                example_inputs,
                static_input_idxs or (),
                **cudagraphify_kwargs,
            )
        else:
            from .compile_fx import cudagraphify

            compiled_graph.current_callable = cudagraphify(
                current_callable,
                static_input_idxs=static_input_idxs or (),
                **cudagraphify_kwargs,
            )

    else:
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)

        if any(is_gpu(device) for device in compiled_graph.device_types):
            # prefer better disable_cudagraphs_reason bc stack trace
            # TODO: migrate all disable reasons to stack trace, refactor
            if compiled_graph.disabled_cudagraphs_reason:
                log_cudagraph_skip_and_bump_counter(
                    compiled_graph.disabled_cudagraphs_reason
                )
            else:
                log_cudagraph_skip_and_bump_counter(
                    f"skipping cudagraphs due to {cudagraph_fail_reasons}"
                )


def cudagraph_partition_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    constants: dict[str, torch.Tensor | type],
    boxed_forward_device_index: BoxedDeviceIndex | None,
) -> None:
    """
    Cudagraphify each partition function, which first prepares the necessary
    metadata and then applies the cudagraphify function to each partition.

    Assuming all partition functions are cudagraphified and share the same order
    as `compiled_graph.partition_maps`. See [Note: Graph Partition Map for CUDAGraph].
    """
    from torch._inductor.compiler_bisector import CompilerBisector

    if CompilerBisector.disable_subsystem("inductor", "cudagraphs"):
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)
        log_cudagraph_skip_and_bump_counter("skipping cudagraphs due to bisector")
        return

    if compiled_graph.cudagraph_info is None:
        raise AssertionError("compiled_graph.cudagraph_info must not be None")
    cudagraph_fail_reasons = compiled_graph.cudagraph_info.cudagraph_fail_reasons

    if (
        cudagraph_fail_reasons
        or compiled_graph.partition_maps is None
        or len(compiled_graph.partition_maps) == 0
    ):
        # cudagraphify is not called if there are no partitions
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)
        return

    if compiled_graph.current_callable is None:
        raise AssertionError("compiled_graph.current_callable must not be None")
    if compiled_graph.recursively_apply_fns is None:
        raise AssertionError("compiled_graph.recursively_apply_fns must not be None")
    is_inference = compiled_graph.fx_kwargs["is_inference"]
    is_backward = compiled_graph.fx_kwargs["is_backward"]
    static_input_idxs = OrderedSet(compiled_graph.fx_kwargs["static_input_idxs"] or ())
    mutated_input_idxs = compiled_graph.mutated_input_idxs
    device_index = next(iter(compiled_graph.device_idxs))

    # Filter to only tensor constants (exclude opaque value type classes)
    tensor_constants = {
        k: v for k, v in constants.items() if isinstance(v, torch.Tensor)
    }

    if compiled_graph.cudagraph_info.stack_traces is None:
        raise AssertionError(
            "stack_traces should not be None in cudagraph_partition_post_compile"
        )
    graph_metadata = CudagraphMetadata(
        compiled_graph.cudagraph_info.placeholders,
        static_input_idxs,
        mutated_input_idxs,
        compiled_graph.cudagraph_info.stack_traces,
        OrderedSet(
            getattr(compiled_graph.cudagraph_info, "user_visible_output_idxs", ())
        ),
        tensor_constants,
    )

    prepare_cudagraph_post_compile(
        compiled_graph, example_inputs, boxed_forward_device_index
    )

    from .compile_fx import cudagraphify

    # cudagraphify each partition function, assuming every graph partition function
    # is cudagraphable. Non-cudagraphable ops (e.g., cpu ops) are inlined into
    # `call` function and not included in partition functions.
    cudagraphify_fns = []
    for partition_map in compiled_graph.partition_maps:
        partition_metadata = get_partition_cudagraph_metadata(
            partition_map,
            graph_metadata,
        )

        cudagraphify_fn = partial(
            cudagraphify,
            static_input_idxs=tuple(partition_metadata.static_input_idxs),
            device_index=device_index,
            stack_traces=partition_metadata.stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=tuple(partition_metadata.constants.values()),
            placeholders=partition_metadata.placeholders,
            mutated_input_idxs=tuple(partition_metadata.mutated_input_idxs),
            kernel_free_cudagraph=compiled_graph.kernel_free_cudagraph,
            user_visible_output_idxs=tuple(partition_metadata.user_visible_output_idxs),
        )
        cudagraphify_fns.append(cudagraphify_fn)

    compiled_graph.recursively_apply_fns(cudagraphify_fns)


def maybe_realign_inputs(
    ran_cudagraphs: BoxedBool,
    compiled_graph: CompiledFxGraph,
    inputs_to_check: Sequence[int],
    mutated_inputs_idxs: OrderedSet[int],
) -> None:
    """
    Realigns input strides from inputs_to_check if
    we didn't end up running cudagraphs. Mutates
    `compiled_graph.current_callable` if cudagraphs
    was run. Otherwise, does nothing.

    Non-mutated inputs are handled by deferred alignment copies
    in the generated code. Only mutated inputs need the wrapper
    for writeback.
    """
    if not ran_cudagraphs:
        check_idxs = inputs_to_check
        if compiled_graph._defers_input_alignment:
            # Non-mutated inputs are handled by deferred alignment copies
            # in the generated Python code. Only mutated inputs need the wrapper
            # for writeback. Backends that don't emit deferred copies (cpp_wrapper,
            # FXIR) need the full wrapper.
            check_idxs = [i for i in inputs_to_check if i in mutated_inputs_idxs]
        if check_idxs:
            if compiled_graph.current_callable is None:
                raise AssertionError("compiled_graph.current_callable must not be None")
            new_callable = align_inputs_from_check_idxs(
                compiled_graph.current_callable,
                check_idxs,
                mutated_inputs_idxs,
            )
            if new_callable is not compiled_graph.current_callable:
                # align_inputs_from_check_idxs preserves the boxed convention but
                # its wrapper is typed with a list[InputType] parameter; the field
                # uses the Sequence[InputType] boxed alias, so re-tag it here.
                compiled_graph.current_callable = cast("_BoxedCallable", new_callable)


class CompiledFxGraphConstants:
    """Wrapper class that unwraps constants from a compiled fx graph. This
    version of the class only supports directly grabbing the saved constants off of
    a CompiledFxGraph.

    With freezing, FxGraphCache doesn't store the constants of the input
    GraphModule it gets from AOTAutograd. Instead, it saves just the **names**
    of those constants, and grabs the constant values directly from the graph module
    passed in at runtime.

    Thing is, we don't always *have* the graph module available at runtime, hence
    the existence of this class and its CompiledFxGraphConstantsWithGm counterpart.

    To support freezing, FXGraphCache gets passed a CompiledFxGraphConstantsWithGm during
    post compile. Otherwise, CompiledFxGraphConstants supports the basic case of loading
    the value of constants directly off of the original saved object.
    """

    def unwrap(self, g: CompiledFxGraph) -> dict[str, torch.Tensor | type]:
        if g.constants is None:
            raise AssertionError("g.constants must not be None")
        return {**g.constants, **g.opaque_value_type_classes}


class CompiledFxGraphConstantsWithGm(CompiledFxGraphConstants):
    """
    This version of CompiledFxGraphConstants, instead of grabbing constants
    directly saved on CompiledFxGraphs, will just grab their names. Then, it takes
    a second GraphModule to grab the corresponding constant values out of.

    This is necessary for supporting freezing in FxGraphCache.
    """

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        self.gm = gm

    def unwrap(self, g: CompiledFxGraph) -> dict[str, torch.Tensor | type]:
        frozen_params = {
            name: getattr(self.gm, orig_name)
            for name, orig_name in g.frozen_param_names.items()
        }
        constants = g.constants or {}
        return {**constants, **frozen_params, **g.opaque_value_type_classes}


@dataclasses.dataclass
class CompiledFxGraph(OutputCode):
    """
    Class holding a compiled FX graph. This is the object serialized on disk
    to support FxGraph caching.
    """

    current_callable: _BoxedCallable | None
    recursively_apply_fns: Callable[[Sequence[Callable[..., Any]]], None] | None
    compiled_fn_runner: CompiledFnRunner | None
    cache_key: str
    source_code: str = dataclasses.field(repr=False)  # Do not display source_code
    runnable_graph_str: str = dataclasses.field(repr=False)  # Do not display graph
    inductor_post_grad_graph_str: str = dataclasses.field(
        repr=False
    )  # Do not display graph
    cache_linemap: list[tuple[int, str]] | None
    device_types: OrderedSet[str]
    device_idxs: OrderedSet[int]
    mutated_inputs: OrderedSet[str]
    mutated_input_idxs: OrderedSet[int]
    constants: dict[str, torch.Tensor] | None
    frozen_param_names: dict[str, str]
    torchbind_constants: dict[str, torch._C.ScriptObject | FakeScriptObject]
    opaque_value_type_classes: dict[str, type]
    output_strides: list[tuple[_StrideExprStr, ...] | None] | None
    disabled_cudagraphs_reason: str | None
    metrics_deltas: metrics.CachedMetricsDeltas
    counter_deltas: Counter[str]
    # This is a string representation of an expression we serialize
    # with the object so the guards can be evaluated in a different
    # context in order to verify the validity of serving a cached
    # fx graph. The expression must be generated by:
    # ShapeEnv.produce_guards_expression()
    guards_expr: str | None
    extern_libs_key: str | None
    inductor_provenance_mapping_str: str | None
    inductor_provenance_stack_traces_str: str | None
    kernel_free_cudagraph: bool

    cudagraph_info: CudagraphCachedInfo | None
    partition_maps: list[GraphPartitionMap] | None
    compile_region_name: str | None
    fx_kwargs: _CompileFxKwargs
    inputs_to_check: Sequence[int]

    _boxed_call: bool | None = None
    _triton_bundle: TritonBundle | None = None
    _wrap_compiled_regions: bool = False
    _defers_input_alignment: bool = False
    _compile_context: CompileContext | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )
    # Metadata-stripped copy of the FX graph for fake tensor propagation.
    # Running this graph under FakeTensorMode re-derives output shapes
    # (including aliasing) from the input shapes.
    _original_gm: torch.fx.GraphModule | None = None
    _serialized_original_gm: bytes | None = None

    def __init__(
        self,
        current_callable: _BoxedCallable | None,
        graph: GraphLowering,
        gm: torch.fx.GraphModule,
        output_strides: list[tuple[_StrideExprStr, ...] | None],
        disabled_cudagraphs_reason: str | None,
        metrics_deltas: metrics.CachedMetricsDeltas,
        counter_deltas: Counter[str],
        cudagraphs: BoxedBool,
        example_inputs: Sequence[InputType],
        static_input_idxs: Sequence[int],
        compile_region_name: str | None,
        fx_kwargs: _CompileFxKwargs,
        inputs_to_check: Sequence[int],
        runnable_graph_str: str,
        inductor_post_grad_graph_str: str,
        compiled_fn_runner: CompiledFnRunner | None = None,
        inductor_provenance_mapping_str: str | None = None,
        inductor_provenance_stack_traces_str: str | None = None,
    ) -> None:
        self.current_callable = current_callable
        self.compiled_fn_runner = compiled_fn_runner
        self.recursively_apply_fns = (
            compiled_fn_runner.recursively_apply_fns
            if compiled_fn_runner is not None
            else None
        )
        self.cache_key = graph.cache_key
        if graph.cache_path:
            with open(graph.cache_path) as f:
                self.source_code = f.read()
        self.runnable_graph_str = runnable_graph_str
        self.inductor_post_grad_graph_str = inductor_post_grad_graph_str
        self.inductor_provenance_mapping_str = inductor_provenance_mapping_str
        self.inductor_provenance_stack_traces_str = inductor_provenance_stack_traces_str
        self.cache_linemap = graph.cache_linemap
        # TODO - ordered set
        self.device_types = OrderedSet(graph.device_types)
        self.device_idxs = OrderedSet(graph.device_idxs)
        self.mutated_inputs = OrderedSet(graph.mutated_inputs)
        self.mutated_input_idxs = OrderedSet(graph.mutated_input_idxs)

        # We store the constant attributes in the cache entry and re-attach them
        # to the module created in PyCodeCache.load_by_key_path. In the case that
        # the graph has frozen parameters, we save the mapping from the attribute
        # names in the GraphLowering to the original name of the attribute in the
        # GraphModule. When we create the module from the cache entry, we then
        # look up the constants from the current GraphModule. This scheme allows
        # us to support caching with freezing.
        if not has_frozen_params(gm):
            self.constants = graph.constants
            self.frozen_param_names = {}
        else:
            self.constants = {}
            self.frozen_param_names = {}
            for k, v in graph.constants.items():
                if is_frozen_param(v):
                    self.frozen_param_names[k] = graph.allocated_constant_name[k]
                else:
                    self.constants[k] = v

        self.torchbind_constants = graph.torchbind_constants
        self.opaque_value_type_classes = graph.opaque_value_type_classes
        self.output_strides = output_strides
        self.disabled_cudagraphs_reason = disabled_cudagraphs_reason
        self.kernel_free_cudagraph = graph.kernel_free_cudagraph
        self.metrics_deltas = metrics_deltas
        self.counter_deltas = counter_deltas
        self.guards_expr = None
        self.extern_libs_key = None
        self.cudagraph_info = None
        self.partition_maps = graph.partition_maps
        self._defers_input_alignment = getattr(graph, "_defers_input_alignment", False)
        storage_mutation_info = get_input_storage_mutation_info(gm)
        self.fx_kwargs = {}
        self.inputs_to_check = ()

        cudagraph_info = None
        if cudagraphs:
            # check cudagraph disabling reasons from inductor lowering
            if self.disabled_cudagraphs_reason:
                if any(is_gpu(device) for device in self.device_types):
                    log_cudagraph_skip_and_bump_counter(
                        f"skipping cudagraphs due to {self.disabled_cudagraphs_reason}"
                    )
                else:
                    counters["inductor"]["cudagraph_skips"] += 1
                BoxedBool.disable(cudagraphs)
            else:
                if not config.triton.cudagraph_support_input_mutation:
                    # Skip supports for cudagraph-managed tensors
                    from torch._inductor.cudagraph_utils import (
                        check_for_mutation_ignore_cuda_graph_managed_tensor,
                    )

                    has_mutation_str = (
                        check_for_mutation_ignore_cuda_graph_managed_tensor(
                            gm,
                            self.mutated_inputs,
                            self.mutated_input_idxs,
                            static_input_idxs,
                        )
                    )
                    has_mutation = has_mutation_str is not None

                    if has_mutation:
                        self.disabled_cudagraphs_reason = has_mutation_str
                else:
                    # Check mutation later to support cudagraph-managed tensors
                    has_mutation = None

                storage_mutation_reason = get_input_storage_mutation_reason(
                    storage_mutation_info,
                )
                if storage_mutation_reason is not None:
                    self.disabled_cudagraphs_reason = storage_mutation_reason

                cudagraph_tests = [
                    (storage_mutation_reason is None, "input storage mutation"),
                    (not has_mutation, "mutated inputs"),
                    (
                        not config.force_disable_cudagraph_TESTING_ONLY,
                        "force_disable_cudagraph_TESTING_ONLY",
                    ),
                    (
                        all(
                            isinstance(
                                t,
                                (
                                    torch.Tensor,
                                    torch.SymInt,
                                    torch.Generator,
                                    CustomClassBase,
                                ),
                            )
                            for t in example_inputs
                        ),
                        "non-Tensor inputs",
                    ),
                ]
                output = output_node(gm)
                # output args are tuple of first argument
                if len(output.args) != 1:
                    raise AssertionError(
                        f"Expected output.args to have length 1, got {len(output.args)}"
                    )
                # Use stack traces captured on the output node before
                # post-grad passes, which may strip stack_trace from
                # individual arg nodes.
                stack_traces = output.meta.get("output_stack_traces") or [
                    (arg.stack_trace if isinstance(arg, torch.fx.node.Node) else None)
                    for arg in output.args[0]  # type: ignore[union-attr]
                ]
                user_visible_output_idxs = (
                    output.meta.get("user_visible_output_idxs", ())
                    if cudagraph_trees_clone_live_user_visible_outputs()
                    else ()
                )
                cudagraph_fail_reasons = [s for b, s in cudagraph_tests if not b]
                placeholders = tuple(get_placeholder_info(gm.graph))
                cudagraph_info = CudagraphCachedInfo(
                    placeholders,
                    stack_traces,
                    tuple(user_visible_output_idxs),
                    cudagraph_fail_reasons,
                )

        self.cudagraph_info = cudagraph_info
        self.compile_region_name = compile_region_name
        self.inputs_to_check = inputs_to_check
        self.fx_kwargs = fx_kwargs
        self._set_compile_context_for_autotune_cache()

        # aot autograd needs to know to pass in inputs as a list
        self._boxed_call = True

        # Store whether to wrap compiled regions in inductor_compiled_code HOP
        # This is set at compile time to avoid runtime overhead
        self._wrap_compiled_regions = config.wrap_inductor_compiled_regions

        if self._wrap_compiled_regions:
            # Store a metadata-stripped copy of the FX graph. Running this
            # under FakeTensorMode re-derives output shapes and aliasing
            # from the input fake tensors.
            import copy

            gm_copy = copy.deepcopy(gm)
            for node in gm_copy.graph.nodes:
                node.meta.clear()
            self._original_gm = gm_copy

    def __del__(self) -> None:
        if self.compiled_fn_runner is not None:
            # For torch._inductor.config.graph_partition = True,
            # self.compiled_fn_runner.partitions hold cudagraphified functions
            # which prevents deallocation. When CompiledFxGraph is deleted,
            # self.compiled_fn_runner will not be called in the future so we
            # should also delete these partitions.
            del self.compiled_fn_runner.partitions

    def _set_compile_context_for_autotune_cache(self) -> None:
        compile_context_for_autotune_cache = CompileContext.try_get()
        self._compile_context = (
            compile_context_for_autotune_cache
            if AutotuneCacheBundler.has_active_compile(
                compile_context_for_autotune_cache
            )
            else None
        )

    def __call__(self, inputs: Sequence[Any]) -> Any:
        if self.current_callable is None:
            raise AssertionError("self.current_callable must not be None")

        if (
            torch._inductor.debug.RECORD_GRAPH_EXECUTION
            and torch._inductor.debug.GRAPH_EXECUTION_ORDER is not None
        ):
            graph_id = self.fx_kwargs.get("graph_id")
            compile_id = (
                torch._inductor.debug.GRAPH_COMPILE_IDS.get(graph_id)
                if graph_id is not None
                and torch._inductor.debug.GRAPH_COMPILE_IDS is not None
                else None
            )
            torch._inductor.debug.GRAPH_EXECUTION_ORDER.append(
                {
                    "compile_id": compile_id,
                }
            )
        compile_context_for_autotune_cache = self._compile_context
        has_active_autotune_cache_bundler = AutotuneCacheBundler.has_active_compile(
            compile_context_for_autotune_cache
        )
        autotune_cache_context = (
            compile_context(compile_context_for_autotune_cache)
            if has_active_autotune_cache_bundler
            else contextlib.nullcontext()
        )
        # Autotune cache writes happen during the first call. End the bundler
        # while its saved compile context is restored, then clear the saved
        # context after leaving the compile_context manager.
        try:
            with autotune_cache_context:
                try:
                    # Checking the profiler directly is faster than nullcontext
                    if torch.autograd.profiler._is_profiler_enabled:
                        with torch._C._profiler._RecordFunctionFast(
                            f"## Call CompiledFxGraph {self._fx_graph_cache_key} ##",
                            keyword_values={"scope": "user_scope"},
                        ):
                            return self.current_callable(inputs)
                    else:
                        return self.current_callable(inputs)
                finally:
                    get_runtime_metrics_context().finish()
                    if has_active_autotune_cache_bundler:
                        AutotuneCacheBundler.end_compile()
        finally:
            if (
                has_active_autotune_cache_bundler
                and not AutotuneCacheBundler.has_active_compile(
                    compile_context_for_autotune_cache
                )
            ):
                self._compile_context = None

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        """
        Run a set of post processing steps after loading from the cache. These involve:
         - Setting the tracing context output strides
         - Running cudagraphs if enabled
         - Realigning inputs

        This runs whether or not we have a cache hit, and always runs directly after we get a CompiledFxGraph.
        The results of this function are *not* saved in the cache itself.
        """
        if config.graph_partition and _unstable_customized_partition_wrapper.wrapper:
            # Mechanically apply user-specified cudagraph wrappers without modification
            if self.recursively_apply_fns is None:
                raise AssertionError("self.recursively_apply_fns must not be None")
            if self.compiled_fn_runner is None:
                raise AssertionError("self.compiled_fn_runner must not be None")
            num_partitions = len(self.compiled_fn_runner.partitions)
            wrapper_metadatas = [
                CUDAGraphWrapperMetadata(num_partitions, i)
                for i in range(num_partitions)
            ]
            customized_wrapper = _unstable_customized_partition_wrapper.wrapper
            customized_wrappers_with_metadata = [
                lambda f, m=metadata: customized_wrapper(f, m)
                for metadata in wrapper_metadatas
            ]
            self.recursively_apply_fns(customized_wrappers_with_metadata)
            return

        set_tracing_context_output_strides(example_inputs, self)
        if graph_kwargs["cudagraphs"] is None:
            raise AssertionError("graph_kwargs['cudagraphs'] must not be None")
        if graph_kwargs["is_backward"] is None:
            raise AssertionError("graph_kwargs['is_backward'] must not be None")
        is_backward = graph_kwargs["is_backward"]
        cudagraphs: BoxedBool = graph_kwargs["cudagraphs"]

        # When a CUDAGraphPolicy is set and it says not to wrap this
        # inner CompiledFxGraph (e.g. because wrapping happens at the
        # outer level via policy.wrap_output), disable cudagraphs for
        # this graph so the rest of post_compile (input realignment,
        # _wrap_compiled_regions) still runs normally.
        policy = config.cudagraph_policy
        if policy is not None and not policy.should_wrap(self):
            counters["inductor"]["cudagraph_skips"] += 1
            BoxedBool.disable(cudagraphs)

        if cudagraphs:
            # It's possible that cudagraphs is enabled, but was disabled
            # during a previous compilation we're loading from the cache.
            # If so, we need to disable it on this new process too.
            if self.disabled_cudagraphs_reason:
                if any(is_gpu(device) for device in self.device_types):
                    log_cudagraph_skip_and_bump_counter(
                        f"skipping cudagraphs due to {self.disabled_cudagraphs_reason}"
                    )
                else:
                    counters["inductor"]["cudagraph_skips"] += 1
                BoxedBool.disable(cudagraphs)
            else:
                if is_backward:
                    if "boxed_forward_device_index" not in graph_kwargs:
                        raise AssertionError(
                            "Expected 'boxed_forward_device_index' in graph_kwargs"
                        )
                    boxed_forward_device_index = graph_kwargs[
                        "boxed_forward_device_index"
                    ]
                else:
                    # On the forward we don't know whether or not
                    # boxed_forward_device_index is set yet
                    boxed_forward_device_index = graph_kwargs.get(
                        "boxed_forward_device_index", None
                    )

                if config.graph_partition and policy is None:
                    # With graph_partition=True, we skip some cudagraph checks
                    # if it's supported with partition, so we use
                    # cudagraph_partition_post_compile.  When a CUDAGraphPolicy
                    # is active, we use cudagraph_post_compile instead so the
                    # policy controls wrapping via policy.cudagraphify().
                    cudagraph_partition_post_compile(
                        example_inputs,
                        self,
                        cudagraphs,
                        constants.unwrap(self),
                        boxed_forward_device_index,
                    )
                else:
                    cudagraph_post_compile(
                        example_inputs,
                        self,
                        cudagraphs,
                        constants.unwrap(self),
                        boxed_forward_device_index,
                    )
        inputs_to_check = self.inputs_to_check
        # cudagraphs could have been disabled from the earlier conditions
        # so we still need to realign inputs if that happens
        maybe_realign_inputs(
            cudagraphs,
            self,
            inputs_to_check,
            self.mutated_input_idxs,
        )

        if self._original_gm is None and self._serialized_original_gm is not None:
            from torch._subclasses import FakeTensorMode
            from torch.fx._graph_pickler import GraphPickler
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            fake_mode = FakeTensorMode(
                allow_non_fake_inputs=True,
                shape_env=ShapeEnv(),
            )
            original_gm = cast(
                torch.fx.GraphModule,
                GraphPickler.loads(self._serialized_original_gm, fake_mode),
            )
            original_gm.recompile()
            self._original_gm = original_gm
            self._serialized_original_gm = None

        # Apply inductor_compiled_code HOP wrapper if configured
        # This is done in post_compile to ensure it works with cached artifacts
        if self._wrap_compiled_regions and self.current_callable is not None:
            from torch._higher_order_ops.wrap import InductorCompiledCallable

            original_callable = self.current_callable

            inductor_callable = InductorCompiledCallable(
                original_callable,
                self._original_gm,
                compile_region_name=self.compile_region_name,
            )

            def wrapped_callable(inputs):
                if is_in_torch_dispatch_mode():
                    kwargs = (
                        {"name": self.compile_region_name}
                        if self.compile_region_name is not None
                        else {}
                    )
                    return inductor_compiled_code(inductor_callable, inputs, **kwargs)
                else:
                    return original_callable(inputs)

            self.current_callable = wrapped_callable

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        self._triton_bundle = triton_bundle

    def prepare_for_serialization(self) -> None:
        # We can't really serialize callables that may be C++/Triton/etc.,
        # so we serialize their PyCodeCache disk cache location instead.
        # TODO: This could be better if we're ever able to serialize compiled
        # models to disk.
        self._compile_context = None
        self.current_callable = None
        self.recursively_apply_fns = None
        self.compiled_fn_runner = None
        if self._original_gm is not None:
            from torch.fx._graph_pickler import GraphPickler, Options

            self._serialized_original_gm = GraphPickler.dumps(
                self._original_gm, Options(ops_filter=None)
            )
            self._original_gm = None
        # Note: _serialized_fx_graph is already in serializable form (SerializedGraphModule)
        # so it doesn't need to be cleared

    def write_to_disk(self) -> str:
        from torch._dynamo.utils import counters
        from torch._inductor.codecache import get_path, write_atomic

        # See _save_graph(); we don't store the callable in the cache entry so
        # recreate it here from the PyCodeCache disk cache.
        artifact_path = get_path(self.cache_key, "py")[2]
        code = self.source_code
        if not os.path.exists(artifact_path):
            counters["inductor"]["fxgraph_lookup_write_file"] += 1
            write_atomic(artifact_path, code, make_dirs=True)
        return artifact_path

    def after_deserialization(self, constants: CompiledFxGraphConstants) -> str:
        from torch._dynamo.utils import dynamo_timed
        from torch._inductor.codecache import PyCodeCache

        artifact_path = self.write_to_disk()

        try:
            with dynamo_timed(
                "PyCodeCache.load_by_key_path",
                log_pt2_compile_event=True,
            ):
                code_cache = PyCodeCache.load_by_key_path(
                    self.cache_key,
                    artifact_path,
                    self.cache_linemap,
                    constants.unwrap(self),
                )
                self.current_callable = code_cache.call
                self.recursively_apply_fns = getattr(
                    code_cache, "recursively_apply_fns", None
                )
                self.compiled_fn_runner = getattr(code_cache, "runner", None)
        except OSError:
            log.error("Failed to load artifact: %s", artifact_path)
            raise

        return artifact_path


@dataclasses.dataclass
class CompiledAOTI(OutputCode):
    """
    Class holding an AOTInductor compiled so.
    """

    filename: str | list[str | Weights] | torch.fx.GraphModule
    device_type: str
    current_callable: Callable[..., Any] | None = None
    _cached_files: dict[str, bytes] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not config.aot_inductor.link_libtorch:
            return

        if (
            torch._inductor.cpp_builder._IS_MACOS
            or torch._inductor.cpp_builder._IS_WINDOWS
        ):
            return

        if config.aot_inductor.cross_target_platform == "windows":
            return

        if config.aot_inductor.package_cpp_only:
            return

        if not config.enable_autograd_for_aot:
            return

        if isinstance(self.filename, list):
            current_callable = next(
                fn for fn in self.filename if isinstance(fn, str) and fn.endswith(".so")
            )
        else:
            current_callable = self.filename

        if isinstance(current_callable, torch.fx.GraphModule):
            # pyrefly: ignore [bad-assignment]
            self.current_callable = current_callable
            return

        if self.device_type.startswith("cuda"):
            current_callable = (
                torch._C._aoti.AOTIModelContainerRunnerCuda(  # type: ignore[call-arg]
                    current_callable,
                    1,
                    self.device_type,
                    "",
                    True,
                ).run  # type: ignore[attr-defined]
            )  # type: ignore[attr-defined]
        elif self.device_type.startswith("xpu"):
            current_callable = (
                torch._C._aoti.AOTIModelContainerRunnerXpu(  # type: ignore[call-arg]
                    current_callable,
                    1,
                    self.device_type,
                    "",
                ).run  # type: ignore[attr-defined]
            )  # type: ignore[attr-defined]
        elif self.device_type == "cpu":
            current_callable = (
                torch._C._aoti.AOTIModelContainerRunnerCpu(  # type: ignore[call-arg]
                    current_callable, 1
                ).run  # type: ignore[attr-defined]
            )  # type: ignore[attr-defined]
        else:
            raise RuntimeError(f"unsupported device type {self.device_type}")
        self.current_callable = current_callable
        self._boxed_call = True
        for file in self._cached_files:
            if not os.path.exists(file):
                with open(file, "wb") as f:
                    f.write(self._cached_files[file])

    def __call__(self, inputs: Sequence[Any]) -> Any:
        if self.current_callable is None:
            raise RuntimeError("AOTInductor compiled so is not loaded")
        return self.current_callable(inputs)

    def prepare_for_serialization(self) -> None:
        self.current_callable = None
        self._cached_files = {}
        filenames: list[str] = []
        if isinstance(self.filename, list):
            filenames = self.filename  # type: ignore[assignment]
        elif isinstance(self.filename, str):
            filenames = [self.filename]
        for name in filenames:
            with open(name, "rb") as f:
                self._cached_files[name] = f.read()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["current_callable"] = None
        return state

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        if self.current_callable is None:
            self.__post_init__()

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        pass


@dataclasses.dataclass
class MockFXGraphCacheOutput(OutputCode):
    gm: Any = None

    def __post_init__(self) -> None:
        self._boxed_call = True

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        pass

    def __call__(self, inputs: Sequence[Any]) -> Any:
        return self.gm(inputs)

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        pass


@dataclasses.dataclass
class RegionalOutputCode(OutputCode):
    """
    OutputCode for regional inductor compilation results.

    Regional inductor returns a torch.fx.GraphModule that contains both
    compiled regions (via standalone_compile) and eager regions. This needs
    special serialization using GraphPickler instead of standard pickle.

    The serialization strategy stores the GraphModule as bytes using
    GraphPickler.dumps(), which handles FakeTensors, AOTCompiledArtifacts,
    and other special objects that standard pickle cannot handle.
    """

    # The serialized graph module as bytes (using GraphPickler)
    _serialized_graph_module: bytes | None = None

    # The actual graph module (cleared during serialization)
    _graph_module: torch.nn.Module | None = None

    # Optional filter for ops during serialization
    _ops_filter: Callable[[str], bool] | None = None

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        ops_filter: Callable[[str], bool] = _ops_filter_safe,
        node_metadata_key_filter: Callable[[str], bool] | None = (
            _node_metadata_key_filter_safe
        ),
    ):
        """
        Args:
            graph_module: The torch.fx.GraphModule returned by regional_inductor
            ops_filter: Optional filter function for op names during serialization.
                If provided, only ops whose name passes the filter will be serialized.
        """
        super().__init__()
        self._graph_module = graph_module
        self._serialized_graph_module = None
        self._serialized_wrappers = []
        self._boxed_call = True
        _, module = self._unwrap_graph_module()
        self._inner_boxed_call = isinstance(
            module.graph._codegen, torch.fx.graph._BoxedCodeGen
        )
        self._ops_filter = ops_filter
        self._node_metadata_key_filter = node_metadata_key_filter

    def __call__(self, inputs: list[Any]) -> Any:
        """
        Execute the regional compiled graph.

        Args:
            inputs: A mutable list of inputs. Must be a list (not an arbitrary
                Sequence) because the boxed calling convention allows the callee
                to mutably clear the list when inputs become dead, enabling early
                memory deallocation.
        """
        if self._graph_module is None:
            raise RuntimeError(
                "RegionalOutputCode has no graph module loaded. "
                "Did you forget to call post_compile()?"
            )

        if self._inner_boxed_call:
            return self._graph_module(inputs)
        return self._graph_module(*inputs)

    @property
    def graph(self):
        _, module = self._unwrap_graph_module()
        return module.graph

    def _unwrap_graph_module(
        self,
    ) -> tuple[list[tuple[Callable, dict[str, Any]]], torch.fx.GraphModule]:
        module = self._graph_module
        serialized_wrappers = []
        if isinstance(module, torch._dynamo.OptimizedModule):
            dynamo_ctx = module.dynamo_ctx
            if not isinstance(dynamo_ctx, torch._dynamo.eval_frame.DisableContext):
                raise AssertionError(f"Expected DisableContext, got {type(dynamo_ctx)}")
            serialized_wrappers.append(
                (
                    torch._dynamo.disable,
                    {"reason": dynamo_ctx.msg, "wrapping": dynamo_ctx.wrapping},
                )
            )
            module = module._orig_mod
        if not isinstance(module, torch.fx.GraphModule):
            raise AssertionError(f"Expected torch.fx.GraphModule, got {type(module)}")
        return serialized_wrappers, module

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        """
        Post-compile processing for regional inductor.

        This deserializes the GraphModule from bytes using GraphPickler,
        extracting the fake_mode from example_inputs.
        """
        if self._graph_module is not None:
            return
        if self._serialized_graph_module is None:
            raise AssertionError("self._serialized_graph_module must not be None")
        # Get fake mode from example inputs
        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode(example_inputs)
        if fake_mode is None:
            raise RuntimeError(
                "Could not detect fake mode from example inputs. "
                "Regional inductor requires fake mode for deserialization."
            )

        # Deserialize the graph module
        from torch.fx._graph_pickler import GraphPickler

        gm = GraphPickler.loads(self._serialized_graph_module, fake_mode)
        if not isinstance(gm, torch.fx.GraphModule):
            raise AssertionError(f"Expected torch.fx.GraphModule, got {type(gm)}")
        gm.recompile()
        for fn, kwargs in reversed(self._serialized_wrappers):
            gm = fn(gm, **kwargs)
        self._graph_module = gm

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        """Regional inductor doesn't use triton bundles directly."""

    def prepare_for_serialization(self) -> None:
        """
        Prepare for serialization by converting the GraphModule to bytes.

        This uses GraphPickler to serialize the graph module since it contains
        special objects like FakeTensors and AOTCompiledArtifacts that need
        custom pickling.
        """
        if self._graph_module is not None:
            from torch.fx._graph_pickler import GraphPickler, Options

            self._serialized_graph_module = None
            self._serialized_wrappers, graph_module = self._unwrap_graph_module()

            self._serialized_graph_module = GraphPickler.dumps(
                graph_module,
                options=Options(
                    ops_filter=self._ops_filter,
                    node_metadata_key_filter=self._node_metadata_key_filter,
                ),
            )
            # Clear the graph module to avoid pickling it with standard pickle
            self._graph_module = None
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import inspect
import itertools
import re
import warnings
from io import StringIO
from typing import Any, Generic, Literal, NamedTuple, TYPE_CHECKING, TypeVar
from unittest.mock import patch

import sympy

import torch
import torch.utils._pytree as pytree

from ..utils._ordered_set import OrderedSet
from .utils import IndentedBuffer, reduction_num_outputs, sympy_index_symbol, sympy_str


if TYPE_CHECKING:
    from collections.abc import Callable


T = TypeVar("T")
AtomicMode = Literal[
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "atomic_and",
    "atomic_or",
    "atomic_xor",
    "atomic_cas",
    "atomic_xchg",
]
StoreMode = AtomicMode | Literal["tma"] | None
ReductionType = Literal[
    "argmax",
    "argmin",
    "argmax_value",
    "argmin_value",
    "argmax_with_value",
    "argmin_with_value",
    "welford_reduce",
    "welford_combine",
    "any",
    "fmax",
    "max",
    "min",
    "prod",
    "sum",
    "dot",
    "xor_sum",
    "online_softmax_reduce",
]


