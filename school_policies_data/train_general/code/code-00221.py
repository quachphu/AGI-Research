    "torch.utils._pytree._is_namedtuple_instance is private and will be removed in a future release. "
    "Please use torch.utils._pytree.is_namedtuple_instance instead.",
    category=FutureWarning,
)
def _is_namedtuple_instance(tree: Any) -> bool:
    return is_namedtuple_instance(tree)


def _get_node_type(tree: Any) -> Any:
    node_type = type(tree)
    # All namedtuple types are implicitly registered as pytree nodes.
    # XXX: Other parts of the codebase expect namedtuple types always return
    #      `namedtuple` instead of the actual namedtuple type. Even if the type
    #      is explicitly registered.
    if is_namedtuple_class(node_type):
        return namedtuple
    return node_type


# A leaf is defined as anything that is not a Node.
def tree_is_leaf(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    """Check if a pytree is a leaf.

    >>> tree_is_leaf(1)
    True
    >>> tree_is_leaf(None)
    True
    >>> tree_is_leaf([1, 2, 3])
    False
    >>> tree_is_leaf((1, 2, 3), is_leaf=lambda x: isinstance(x, tuple))
    True
    >>> tree_is_leaf({"a": 1, "b": 2, "c": 3})
    False
    >>> tree_is_leaf({"a": 1, "b": 2, "c": None})
    False
    """
    if is_leaf is not None and is_leaf(tree):
        return True
    return _get_node_type(tree) not in SUPPORTED_NODES


@deprecated(
    "torch.utils._pytree._is_leaf is private and will be removed in a future release. "
    "Please use torch.utils._pytree.tree_is_leaf instead.",
    category=FutureWarning,
)
def _is_leaf(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> bool:
    return tree_is_leaf(tree, is_leaf=is_leaf)


# A TreeSpec represents the structure of a pytree. It holds:
#   "type": the type of root Node of the pytree
#   context: some context that is useful in unflattening the pytree
#   children(): specs for each child of the root Node
#   num_nodes: the total number of nodes
#   num_leaves: the number of leaves
#   num_children: the number of children of the root Node (i.e., len(children()))
#   is_leaf(): whether the root Node is a leaf
@dataclasses.dataclass(init=False, frozen=True, eq=True, repr=False, slots=True)
class TreeSpec:
    """Representing the structure of the pytree."""

    type: Any
    _context: Context
    _children: list[Self]

    num_nodes: int = dataclasses.field(init=False)
    num_leaves: int = dataclasses.field(init=False)
    num_children: int = dataclasses.field(init=False)

    def __init__(
        self,
        type: Any,
        context: Context,  # keep for backward compatibility
        children_specs: list[Self],  # keep for backward compatibility
    ) -> None:
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_children", children_specs)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.type is None:
            if self._context is not None:
                raise AssertionError("leaf node should not have a context")
            if len(self._children) != 0:
                raise AssertionError("leaf node should not have children")
            num_nodes = 1
            num_leaves = 1
            num_children = 0
        else:
            num_nodes = 1
            num_leaves = 0
            for child in self._children:
                num_nodes += child.num_nodes
                num_leaves += child.num_leaves
            num_children = len(self._children)
        object.__setattr__(self, "num_nodes", num_nodes)
        object.__setattr__(self, "num_leaves", num_leaves)
        object.__setattr__(self, "num_children", num_children)

    def __repr__(self, indent: int = 0) -> str:
        repr_prefix: str = f"TreeSpec({self.type.__name__}, {self._context}, ["
        children_specs_str: str = ""
        if self.num_children > 0:
            indent += 2
            children_specs_str += self._children[0].__repr__(indent)
            children_specs_str += "," if self.num_children > 1 else ""
            children_specs_str += ",".join(
                [
                    "\n" + " " * indent + child.__repr__(indent)
                    for child in self._children[1:]
                ]
            )
        repr_suffix: str = f"{children_specs_str}])"
        return repr_prefix + repr_suffix

    def __eq__(self, other: PyTree) -> bool:
        if self is other:
            return True
        elif other.__class__ is self.__class__:
            if str(self.type) != str(other.type):
                return False
            if self._context != other._context:
                return False
            elif self._children != other._children:
                return False
            return True
        return NotImplemented

    @property
    def context(self) -> Context:
        return self._context

    @property
    @deprecated(
        "`treespec.children_specs` is deprecated. "
        "Use `treespec.child(index)` to access a single child, "
        "or `treespec.children()` to get all children.",
        category=FutureWarning,
    )
    def children_specs(self) -> list[Self]:
        return self._children

    def is_leaf(self) -> bool:
        """Test whether the treespec represents a leaf."""
        return self.num_nodes == 1 and self.num_leaves == 1

    def children(self) -> list[Self]:
        """Get all the child treespecs."""
        return self._children.copy()

    def child(self, index: int) -> Self:
        """Get the child treespec at the given index."""
        return self._children[index]

    def flatten_up_to(self, tree: PyTree) -> list[PyTree]:
        """Flatten the subtrees in ``tree`` up to the structure of this treespec and return a list of subtrees."""

        def helper(treespec: TreeSpec, node: PyTree, subtrees: list[PyTree]) -> None:
            if treespec.is_leaf():
                subtrees.append(node)
                return

            node_type = _get_node_type(node)
            if treespec.type not in BUILTIN_TYPES:
                # Always require custom node types to match exactly
                if node_type != treespec.type:
                    raise ValueError(
                        f"Type mismatch; "
                        f"expected {treespec.type!r}, but got {node_type!r}.",
                    )
                flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
                children, context = flatten_fn(node)
                if len(children) != treespec.num_children:
                    raise ValueError(
                        f"Node arity mismatch; "
                        f"expected {treespec.num_children}, but got {len(children)}.",
                    )
                if context != treespec._context:
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
                    dict_context = (
                        treespec._context
                        if treespec.type is not defaultdict
                        # ignore mismatch of `default_factory` for defaultdict
                        else treespec._context[1]
                    )
                    expected_keys = dict_context
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
                    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
                    children, context = flatten_fn(node)
                    if (
                        node_type is not deque  # ignore mismatch of `maxlen` for deque
                    ) and context != treespec._context:
                        raise ValueError(
                            f"Node context mismatch for node type {treespec.type!r}; "
                            f"expected {treespec._context!r}, but got {context!r}.",  # namedtuple type mismatch
                        )

            for subtree, subspec in zip(children, treespec._children, strict=True):
                helper(subspec, subtree, subtrees)

        subtrees: list[PyTree] = []
        helper(self, tree, subtrees)
        return subtrees

    def unflatten(self, leaves: Iterable[Any]) -> PyTree:
        """Reconstruct a pytree from the leaves."""
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

        unflatten_fn = SUPPORTED_NODES[self.type].unflatten_fn

        # Recursively unflatten the children
        start = 0
        end = 0
        child_pytrees = []
        for child_spec in self._children:
            end += child_spec.num_leaves
            child_pytrees.append(child_spec.unflatten(leaves[start:end]))
            start = end

        return unflatten_fn(child_pytrees, self._context)

    def __hash__(self) -> int:
        node_type = self.type
        if node_type is defaultdict:
            default_factory, dict_context = self._context
            hashable_context = (default_factory, tuple(dict_context))
        elif node_type in (dict, OrderedDict):
            hashable_context = tuple(self._context)
        elif node_type is None or node_type in BUILTIN_TYPES:
            hashable_context = self._context
        elif isinstance(self._context, ConstantNode):
            hashable_context = self._context.value
        else:
            # The context for user-defined node types might not be hashable.
            # Ignore it for hashing.
            # This does not break the correctness that equal objects imply the
            # same hash. This might increase the hash collision rate, but we
            # don't care about that.
            hashable_context = None
        return hash((node_type, hashable_context, tuple(self._children)))


PyTreeSpec: TypeAlias = TreeSpec


# NOTE: subclassing a dataclass is subtle. In order to enable reasoning about
# this class with `dataclasses.fields`, etc., while having a simplified
# constructor that takes no argument, we wrap with `dataclass(init=True, ...)`
# again, with fields that have `init=False`.
@deprecated(
    "`isinstance(treespec, LeafSpec)` is deprecated, "
    "use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.",
    category=FutureWarning,
)
@dataclasses.dataclass(init=False, frozen=True, eq=False, repr=False, slots=True)
class LeafSpec(TreeSpec):
    type: Any = dataclasses.field(default=None, init=False)
    _context: Context = dataclasses.field(default=None, init=False)
    _children: list[Self] = dataclasses.field(default_factory=list, init=False)

    def __init__(self) -> None:
        # On Python 3.10.0, the dataclass-generated __init__ for frozen+slots
        # dataclasses doesn't populate init=False fields with simple defaults,
        # causing copy.deepcopy to fail with AttributeError. Avoid the bug by
        # providing a custom __init__ (init=False above) instead.
        object.__setattr__(self, "type", None)
        object.__setattr__(self, "_context", None)
        object.__setattr__(self, "_children", [])
        object.__setattr__(self, "num_nodes", 1)
        object.__setattr__(self, "num_leaves", 1)
        object.__setattr__(self, "num_children", 0)

    def __repr__(self, indent: int = 0) -> str:
        return "*"

    def __reduce__(self) -> tuple[Callable[[], "LeafSpec"], tuple[()]]:
        # Reconstruct via the factory rather than the deprecated LeafSpec
        # constructor, so copy/pickle round-trips don't emit the FutureWarning
        # and reuse the shared singleton.
        return (treespec_leaf, ())


# All leaves are equivalent, so represent with a single object to save on
# object construction time
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=__name__, append=False
    )
    _LEAF_SPEC = LeafSpec()


def treespec_leaf() -> LeafSpec:
    """Make a treespec representing a leaf node."""
    return _LEAF_SPEC


def treespec_tuple(iterable: Iterable[TreeSpec] = (), /) -> TreeSpec:
    """Make a tuple treespec from an iterable of child treespecs."""
    children = list(iterable)
    if any(not isinstance(child, TreeSpec) for child in children):
        raise ValueError(f"Expected a tuple of TreeSpec values, got: {children!r}.")
    return TreeSpec(tuple, None, children)


def treespec_dict(
    mapping: Mapping[Any, TreeSpec] | Iterable[tuple[Any, TreeSpec]] = (),
    /,
    **kwargs: TreeSpec,
) -> TreeSpec:
    """Make a dict treespec from a dict of child treespecs."""
    dct = dict(mapping, **kwargs)
    if any(not isinstance(child, TreeSpec) for child in dct.values()):
        raise ValueError(f"Expected a dictionary of TreeSpec values, got: {dct!r}.")
    return TreeSpec(dict, list(dct.keys()), list(dct.values()))


def _is_pytreespec_instance(
    obj: Any,
) -> TypeIs["TreeSpec | cxx_pytree.PyTreeSpec"]:
    if isinstance(obj, TreeSpec):
        return True
    if "torch.utils._cxx_pytree" in sys.modules:
        # The C++ pytree module is not always available, so we check if it is loaded.
        # If the C++ pytree module is loaded, we can check if the treespec
        # is an instance of the C++ TreeSpec class.
        import torch.utils._cxx_pytree as cxx_pytree

        if isinstance(obj, cxx_pytree.PyTreeSpec):
            return True
    if "torch._dynamo.polyfills.pytree" in sys.modules:
        # The PyTorch Dynamo pytree module is not always available, so we check if it is loaded.
        # If the PyTorch Dynamo pytree module is loaded, we can check if the treespec
        # is an instance of the PyTorch Dynamo TreeSpec class.
        import torch._dynamo.polyfills.pytree as dynamo_pytree

        return isinstance(obj, dynamo_pytree.PyTreeSpec)
    return False


def _ensure_python_treespec_instance(
    treespec: "TreeSpec | cxx_pytree.PyTreeSpec",
) -> TreeSpec:
    if isinstance(treespec, TreeSpec):
        return treespec

    if not _is_pytreespec_instance(treespec):
        raise TypeError(
            f"Expected `treespec` to be an instance of "
            f"PyTreeSpec but got item of type {type(treespec)}."
        )
    dummy_tree = treespec.unflatten([0] * treespec.num_leaves)
    return tree_structure(dummy_tree)


def tree_flatten(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> tuple[list[Any], TreeSpec]:
    """Flattens a pytree into a list of values and a TreeSpec that can be used
    to reconstruct the pytree.
    """

    def helper(node: PyTree, leaves: list[Any]) -> TreeSpec:
        if tree_is_leaf(node, is_leaf=is_leaf):
            leaves.append(node)
            return _LEAF_SPEC

        node_type = _get_node_type(node)
        flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
        children, context = flatten_fn(node)

        # Recursively flatten the children
        subspecs = [helper(child, leaves) for child in children]
        return TreeSpec(node_type, context, subspecs)

    leaves: list[Any] = []
    treespec = helper(tree, leaves)
    return leaves, treespec


def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree:
    """Given a list of values and a TreeSpec, builds a pytree.
    This is the inverse operation of `tree_flatten`.
    """
    if not _is_pytreespec_instance(treespec):
        if not _is_pytreespec_instance(leaves):
            raise TypeError(
                f"Expected `treespec` to be an instance of "
                f"PyTreeSpec but got item of type {type(treespec)}."
            )
        # Allow passing the PyTreeSpec instance as the first argument
        leaves, treespec = treespec, leaves
    return treespec.unflatten(leaves)


def tree_iter(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> Iterable[Any]:
    """Get an iterator over the leaves of a pytree."""
    if tree_is_leaf(tree, is_leaf=is_leaf):
        yield tree
    else:
        node_type = _get_node_type(tree)
        flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
        child_pytrees, _ = flatten_fn(tree)

        # Recursively flatten the children
        for child in child_pytrees:
            yield from tree_iter(child, is_leaf=is_leaf)


def tree_leaves(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> list[Any]:
    """Get a list of leaves of a pytree."""
    return list(tree_iter(tree, is_leaf=is_leaf))


def tree_structure(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> TreeSpec:
    """Get the TreeSpec for a pytree."""
    return tree_flatten(tree, is_leaf=is_leaf)[1]


def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_map_`.

    >>> tree_map(lambda x: x + 1, {"x": 7, "y": (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_map(lambda x: x is None, {"x": 7, "y": (42, 64), "z": None})
    {'x': False, 'y': (False, False), 'z': True}

    If multiple inputs are given, the structure of the tree is taken from the first input;
    subsequent inputs need only have ``tree`` as a prefix:

    >>> tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and ``xs``
        is the tuple of values at corresponding nodes in ``rests``.
    """
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(map(func, *flat_args))


def tree_map_(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    """Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`.

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x, *xs)`` (not the return value) where ``x`` is the value at the corresponding leaf
        in ``tree`` and ``xs`` is the tuple of values at values at corresponding nodes in ``rests``.
    """
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    deque(map(func, *flat_args), maxlen=0)  # consume and exhaust the iterable
    return tree


Type2 = tuple[type[T], type[S]]
Type3 = tuple[type[T], type[S], type[U]]
TypeAny = type[Any] | tuple[type[Any], ...] | types.UnionType

Fn2 = Callable[[T | S], R]
Fn3 = Callable[[T | S | U], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]

MapOnlyFn = Callable[[T], Callable[[Any], Any]]


# These specializations help with type inference on the lambda passed to this
# function
@overload
def map_only(type_or_types_or_pred: type[T], /) -> MapOnlyFn[Fn[T, Any]]: ...


@overload
def map_only(type_or_types_or_pred: Type2[T, S], /) -> MapOnlyFn[Fn2[T, S, Any]]: ...


@overload
def map_only(
    type_or_types_or_pred: Type3[T, S, U], /
) -> MapOnlyFn[Fn3[T, S, U, Any]]: ...


# This specialization is needed for the implementations below that call
@overload
def map_only(type_or_types_or_pred: TypeAny, /) -> MapOnlyFn[FnAny[Any]]: ...


@overload
def map_only(
    type_or_types_or_pred: Callable[[Any], bool], /
) -> MapOnlyFn[FnAny[Any]]: ...


def map_only(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool], /
) -> MapOnlyFn[FnAny[Any]]:
    """
    Suppose you are writing a tree_map over tensors, leaving everything
    else unchanged.  Ordinarily you would have to write:

        def go(t):
            if isinstance(t, Tensor):
                return ...
            else:
                return t

    With this function, you only need to write:

        @map_only(Tensor)
        def go(t):
            return ...

    You can also directly use 'tree_map_only'
    """
    if isinstance(type_or_types_or_pred, (type, tuple, types.UnionType)):

        def pred(x: Any) -> bool:
            return isinstance(x, type_or_types_or_pred)  # type: ignore[arg-type]

    elif callable(type_or_types_or_pred):
        pred = type_or_types_or_pred  # type: ignore[assignment]
    else:
        raise TypeError("Argument must be a type, a tuple of types, or a callable.")

    def wrapper(func: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(func)
        def wrapped(x: T) -> Any:
            if pred(x):
                return func(x)
            return x

        return wrapped

    return wrapper


@overload
def tree_map_only(
    type_or_types_or_pred: type[T],
    /,
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: Type2[T, S],
    /,
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: Type3[T, S, U],
    /,
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: TypeAny,
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


def tree_map_only(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    return tree_map(map_only(type_or_types_or_pred)(func), tree, is_leaf=is_leaf)


@overload
def tree_map_only_(
    type_or_types_or_pred: type[T],
    /,
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: Type2[T, S],
    /,
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: Type3[T, S, U],
    /,
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: TypeAny,
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


def tree_map_only_(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    return tree_map_(map_only(type_or_types_or_pred)(func), tree, is_leaf=is_leaf)


def tree_all(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return all(map(pred, flat_args))


def tree_any(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return any(map(pred, flat_args))


@overload
def tree_all_only(
    type_or_types: type[T],
    /,
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_all_only(
    type_or_types: Type2[T, S],
    /,
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_all_only(
    type_or_types: Type3[T, S, U],
    /,
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


def tree_all_only(
    type_or_types: TypeAny,
    /,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return all(pred(x) for x in flat_args if isinstance(x, type_or_types))


@overload
def tree_any_only(
    type_or_types: type[T],
    /,
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_any_only(
    type_or_types: Type2[T, S],
    /,
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_any_only(
    type_or_types: Type3[T, S, U],
    /,
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


def tree_any_only(
    type_or_types: TypeAny,
    /,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return any(pred(x) for x in flat_args if isinstance(x, type_or_types))


# Broadcasts a pytree to the provided TreeSpec and returns the flattened
# values. If this is not possible, then this function returns None.
#
# For example, given pytree=0 and spec=TreeSpec(list, None, [LeafSpec(), LeafSpec()]),
# would return [0, 0]. This is useful for part of the vmap implementation:
# a user can pass in vmap(fn, in_dims)(*inputs). `in_dims` should be
# broadcastable to the tree structure of `inputs` and we use
# _broadcast_to_and_flatten to check this.
def _broadcast_to_and_flatten(
    tree: PyTree,
    treespec: TreeSpec,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> list[Any] | None:
    def broadcast_prefix(
        prefix_tree: PyTree,
        full_tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> list[Any]:
        result: list[Any] = []

        def add_leaves(x: Any, subtree: PyTree) -> None:
            subtreespec = tree_structure(subtree, is_leaf=is_leaf)
            result.extend([x] * subtreespec.num_leaves)

        tree_map_(
            add_leaves,
            prefix_tree,
            full_tree,
            is_leaf=is_leaf,
        )
        return result

    full_tree = tree_unflatten([0] * treespec.num_leaves, treespec)
    try:
        return broadcast_prefix(tree, full_tree, is_leaf=is_leaf)
    except ValueError:
        return None


@dataclasses.dataclass
class _TreeSpecSchema:
    """
    _TreeSpecSchema is the schema used to serialize the TreeSpec
    It contains the following fields:
    - type: A string name of the type. null for the case of a LeafSpec.
    - context: Any format which is json dumpable
    - children_spec: A list of children serialized specs.
    """

    type: str | None
    context: DumpableContext
    children_spec: list["_TreeSpecSchema"]


class _ProtocolFn(NamedTuple):
    treespec_to_json: Callable[[TreeSpec], DumpableContext]
    json_to_treespec: Callable[[DumpableContext], TreeSpec]


_SUPPORTED_PROTOCOLS: dict[int, _ProtocolFn] = {}


def _treespec_to_json(treespec: TreeSpec) -> _TreeSpecSchema:
    if treespec.is_leaf():
        return _TreeSpecSchema(None, None, [])

    if treespec.type not in SUPPORTED_SERIALIZED_TYPES:
        raise NotImplementedError(
            f"Serializing {treespec.type} in pytree is not registered.",
        )

    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[treespec.type]

    serialized_type_name = serialize_node_def.serialized_type_name

    if serialized_type_name == NO_SERIALIZED_TYPE_NAME_FOUND:
        raise NotImplementedError(
            f"No registered serialization name for {treespec.type} found. "
            "Please update your _register_pytree_node call with a `serialized_type_name` kwarg."
        )

    if serialize_node_def.to_dumpable_context is None:
        try:
            serialized_context = json.dumps(treespec._context, cls=EnumEncoder)
        except TypeError as e:
            raise TypeError(
                "Unable to serialize context. "
                "Please make the context json dump-able, or register a "
                "custom serializer using _register_pytree_node."
            ) from e
    else:
        serialized_context = serialize_node_def.to_dumpable_context(treespec._context)

    child_schemas = [_treespec_to_json(child) for child in treespec._children]

    return _TreeSpecSchema(serialized_type_name, serialized_context, child_schemas)


def enum_object_hook(obj: dict[str, Any]) -> Enum | dict[str, Any]:
    if "__enum__" in obj:
        modname, _, classname = obj["fqn"].partition(":")
        mod = importlib.import_module(modname)
        enum_cls = mod
        for attr in classname.split("."):
            enum_cls = getattr(enum_cls, attr)
        enum_cls = cast(type[Enum], enum_cls)

        return enum_cls[obj["name"]]
    return obj


def _json_to_treespec(json_schema: DumpableContext) -> TreeSpec:
    if (
        json_schema["type"] is None
        and json_schema["context"] is None
        and len(json_schema["children_spec"]) == 0
    ):
        return _LEAF_SPEC

    if json_schema["type"] not in SERIALIZED_TYPE_TO_PYTHON_TYPE:
        raise NotImplementedError(
            f"Deserializing {json_schema['type']} in pytree is not registered.",
        )

    typ = SERIALIZED_TYPE_TO_PYTHON_TYPE[json_schema["type"]]
    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[typ]

    if serialize_node_def.from_dumpable_context is None:
        try:
            context = json.loads(json_schema["context"], object_hook=enum_object_hook)
        except TypeError as ex:
            raise TypeError(
                "Unable to deserialize context. "
                "Please make the context json load-able, or register a "
                "custom serializer using _register_pytree_node.",
            ) from ex
    else:
        context = serialize_node_def.from_dumpable_context(json_schema["context"])

    children_specs = [
        _json_to_treespec(child_string) for child_string in json_schema["children_spec"]
    ]

    return TreeSpec(typ, context, children_specs)


_SUPPORTED_PROTOCOLS[1] = _ProtocolFn(_treespec_to_json, _json_to_treespec)


def treespec_dumps(treespec: TreeSpec, protocol: int | None = None) -> str:
    treespec = _ensure_python_treespec_instance(treespec)

    if protocol is None:
        protocol = DEFAULT_TREESPEC_SERIALIZATION_PROTOCOL

    if protocol in _SUPPORTED_PROTOCOLS:
        json_spec = _SUPPORTED_PROTOCOLS[protocol].treespec_to_json(treespec)
    else:
        raise ValueError(
            f"Unknown protocol {protocol}. "
            f"Available protocols: {list(_SUPPORTED_PROTOCOLS.keys())}",
        )

    str_spec = json.dumps((protocol, dataclasses.asdict(json_spec)), cls=EnumEncoder)
    return str_spec


@functools.lru_cache
def treespec_loads(serialized: str) -> TreeSpec:
    protocol, json_schema = json.loads(serialized)

    if protocol in _SUPPORTED_PROTOCOLS:
        return _SUPPORTED_PROTOCOLS[protocol].json_to_treespec(json_schema)
    raise ValueError(
        f"Unknown protocol {protocol}. "
        f"Available protocols: {list(_SUPPORTED_PROTOCOLS.keys())}",
    )


class _Asterisk(str):
    __slots__ = ()

    def __new__(cls) -> Self:
        return super().__new__(cls, "*")

    def __repr__(self) -> str:
        return "*"  # no quotes


_asterisk = _Asterisk()
del _Asterisk


def treespec_pprint(treespec: TreeSpec) -> str:
    dummy_tree = tree_unflatten([_asterisk] * treespec.num_leaves, treespec)
    return repr(dummy_tree)


# TODO(angelayi): remove this function after OSS/internal stabilize
@deprecated(
    "`pytree_to_str` is deprecated. Please use `treespec_dumps` instead.",
    category=FutureWarning,
)
def pytree_to_str(treespec: TreeSpec) -> str:
    return treespec_dumps(treespec)


# TODO(angelayi): remove this function after OSS/internal stabilize
@deprecated(
    "`str_to_pytree` is deprecated. Please use `treespec_loads` instead.",
    category=FutureWarning,
)
def str_to_pytree(json: str) -> TreeSpec:
    return treespec_loads(json)


def arg_tree_leaves(*args: PyTree, **kwargs: PyTree) -> list[Any]:
    """Get a flat list of arguments to this function

    A slightly faster version of tree_leaves((args, kwargs))
    """
    leaves: list[Any] = []
    for a in args:
        leaves.extend(tree_iter(a))
    for a in kwargs.values():
        leaves.extend(tree_iter(a))
    return leaves


def tree_flatten_with_path(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> tuple[list[tuple[KeyPath, Any]], TreeSpec]:
    """Flattens a pytree like :func:`tree_flatten`, but also returns each leaf's key path.

    Args:
        tree: a pytree to flatten. If it contains a custom type, that type must be
            registered with an appropriate `tree_flatten_with_path_fn` when registered
            with :func:`register_pytree_node`.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    Returns:
        A tuple where the first element is a list of (key path, leaf) pairs, and the
        second element is a :class:`TreeSpec` representing the structure of the flattened
        tree.
    """
    _, treespec = tree_flatten(tree, is_leaf)
    return list(_generate_key_paths((), tree, is_leaf)), treespec


def tree_leaves_with_path(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> list[tuple[KeyPath, Any]]:
    """Gets the leaves of a pytree like ``tree_leaves`` and returns each leaf's key path.

    Args:
        tree: a pytree. If it contains a custom type, that type must be
            registered with an appropriate `tree_flatten_with_path_fn` when registered
            with :func:`register_pytree_node`.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    Returns:
        A list of (key path, leaf) pairs.
    """
    return list(_generate_key_paths((), tree, is_leaf))


def _generate_key_paths(
    key_path: KeyPath,
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> Iterable[tuple[KeyPath, Any]]:
    if is_leaf and is_leaf(tree):
        yield key_path, tree
        return

    node_type = _get_node_type(tree)
    handler = SUPPORTED_NODES.get(node_type)
    if not handler:
        # This is a leaf
        yield key_path, tree
        return

    flatten_with_keys = handler.flatten_with_keys_fn
    if flatten_with_keys:
        key_children, _ = flatten_with_keys(tree)
        for k, c in key_children:
            yield from _generate_key_paths((*key_path, k), c, is_leaf)
    else:
        # We registered this pytree but didn't add a flatten_with_keys_fn, complain.
        raise ValueError(
            f"Did not find a flatten_with_keys_fn for type: {node_type}. "
            "Please pass a flatten_with_keys_fn argument to register_pytree_node."
        )


def tree_map_with_path(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    """Like :func:`tree_map`, but the provided callable takes an additional key path argument.

    Args:
        func: A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees. The first positional argument
            to ``func`` is the key path of the leaf in question. The second
            positional argument is the value of the leaf.
        tree: A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests: A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(keypath, x, *xs)`` where ``keypath`` is the key path at the
        corresponding leaf in ``tree``, ``x`` is the value at that leaf, and
        ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    keypath_leaves, treespec = tree_flatten_with_path(tree, is_leaf)
    keypath_leaves = list(zip(*keypath_leaves, strict=True))
    all_keypath_leaves = keypath_leaves + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(func(*xs) for xs in zip(*all_keypath_leaves, strict=True))


def keystr(kp: KeyPath) -> str:
    """Given a key path, return a pretty-printed representation."""
    return "".join([str(k) for k in kp])


def key_get(obj: Any, kp: KeyPath) -> Any:
    """Given an object and a key path, return the value at the key path."""
    for k in kp:
        obj = k.get(obj)
    return obj
<|endoftext|>import inspect
from typing import Any


def _signature_metadata(
    sig: inspect.Signature,
) -> tuple[tuple[inspect.Parameter, ...], bool, int]:
    """
    Returns tuple(sig.parameters.values()), if any has VAR_POSITIONAL or VAR_KEYWORD, and the max_positional
    """
    params = tuple(sig.parameters.values())
    has_var_args = False
    max_positional = 0

    for p in params:
        kind = p.kind
        if kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            has_var_args = True
        if kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            max_positional += 1

    return params, has_var_args, max_positional


def _fast_bind(
    sig: inspect.Signature, *args: Any, **kwargs: Any
) -> inspect.BoundArguments:
    """
    Fast path for inspect.Signature.bind() for signatures without
    VAR_POSITIONAL or VAR_KEYWORD parameters. Falls back to sig.bind()
    for signatures that contain *args or **kwargs.
    """
    params, has_var_args, max_positional = _signature_metadata(sig)

    # fallback for complex signatures
    if has_var_args:
        return sig.bind(*args, **kwargs)

    len_args = len(args)

    if len_args > max_positional:
        raise TypeError(
            f"Too many positional arguments: expected max {max_positional}, got {len_args}"
        )

    arguments: dict[str, Any] = {}
    arg_i = 0

    for p in params:
        name = p.name
        kind = p.kind

        if kind is inspect.Parameter.POSITIONAL_ONLY:
            if name in kwargs:
                raise TypeError(
                    f"Got some positional-only arguments passed as keyword arguments: '{name}'"
                )
            if arg_i < len_args:
                arguments[name] = args[arg_i]
                arg_i += 1
            elif p.default is inspect.Parameter.empty:
                raise TypeError(f"Missing required argument '{name}'")

        elif kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if arg_i < len_args:
                if name in kwargs:
                    raise TypeError(f"Multiple values for argument '{name}'")
                arguments[name] = args[arg_i]
                arg_i += 1
            elif name in kwargs:
                arguments[name] = kwargs[name]
            elif p.default is inspect.Parameter.empty:
                raise TypeError(f"Missing required argument '{name}'")

        elif kind is inspect.Parameter.KEYWORD_ONLY:
            if name in kwargs:
                arguments[name] = kwargs[name]
            elif p.default is inspect.Parameter.empty:
                raise TypeError(f"Missing required argument '{name}'")

    # disallow extra keyword arguments not in the signature
    # cause kwargs have been processed by sig.bind at the beginning
    for name in kwargs:
        if name not in sig.parameters:
            raise TypeError(f"Got an unexpected keyword argument '{name}'")

    return inspect.BoundArguments(sig, arguments)  # type: ignore[arg-type]
<|endoftext|># mypy: allow-untyped-defs
"""Utilities for counting theoretical floating point operations.

``FlopCounterMode`` is a context manager that intercepts PyTorch operators and
adds up their registered FLOP formulas. The result is a shape-based,
theoretical FLOP count for the operators that ran inside the context, not a
hardware performance measurement.

The counter is useful when comparing model graphs, activation checkpointing
plans, or shape changes with a stable FLOP accounting rule. It should not be
read as kernel instructions, wall-clock time, memory bandwidth, achieved
FLOP/s, Tensor Core utilization, or the exact work done by a fused kernel.

Counting semantics
------------------

* Counts are produced by formulas in ``flop_registry``. Operators without a
  formula may be decomposed into registered operators; otherwise they add zero
  FLOPs.
* Formula inputs have tensor arguments replaced by their shapes by default.
  Non-tensor arguments pass through unchanged. Use
  ``register_flop_formula(..., get_raw=True)`` only when the formula needs the
  original tensor arguments or metadata.
* Forward and backward operations are counted only if they execute while the
  context manager is active.
* The default formulas use dense, naive mathematical definitions. For example,
  matrix multiplication counts ``2 * m * n * k`` FLOPs.
* Counts do not automatically adjust for dtype-specific throughput, Tensor
  Cores, sparsity, quantization, masking, skipped elements, memory movement, or
  data-dependent early exits. A formula must explicitly model those semantics.
  For example, the built-in attention formulas are upper bounds for causal or
  otherwise masked attention unless the formula explicitly models the mask.
* Higher-order operators and ``torch.compile`` may expose decomposed or fused
  work differently from eager execution. Custom operators and custom Triton
  kernels need a formula or a decomposition if they should contribute FLOPs.
* Module attribution is tracked through ``ModuleTracker``. Totals are always
  available under ``"Global"``; submodule rows are best-effort attribution for
  module calls observed during the context.
* The workload still executes normally. Use this mode for a few representative
  iterations rather than long training runs if overhead or memory use matters.

Example
-------

.. code-block:: python

    import torch
    from torch.utils.flop_counter import FlopCounterMode

    model = torch.nn.Linear(16, 32)
    x = torch.randn(4, 16)

    with FlopCounterMode(display=False) as mode:
        model(x).sum().backward()

    print(mode.get_total_flops())
    print(mode.get_flop_counts()["Global"])

Registering a formula for a custom op
-------------------------------------

Register custom FLOP formulas before constructing ``FlopCounterMode``. The
mode snapshots the global registry during initialization.

.. code-block:: python

    from math import prod

    import torch
    from torch.utils.flop_counter import FlopCounterMode, register_flop_formula

    @torch.library.custom_op("example::scale", mutates_args=())
    def scale(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    @register_flop_formula(torch.ops.example.scale)
    def scale_flops(x_shape, *, out_shape=None) -> int:
        return prod(x_shape)

    x = torch.randn(8)
    with FlopCounterMode(display=False) as mode:
        scale(x)

    assert mode.get_total_flops() == 8
"""
from types import NoneType
import logging
import torch
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from .module_tracker import ModuleTracker
from typing import Any, TypeVar
from collections.abc import Callable
from collections.abc import Iterator
from typing_extensions import ParamSpec
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from math import prod
from functools import wraps
import warnings

__all__ = ["FlopCounterMode", "register_flop_formula"]

_T = TypeVar("_T")
_P = ParamSpec("_P")

log = logging.getLogger(__name__)


try:
    from triton.runtime.jit import JITFunction as _JITFunction
except ImportError:
    if any(getattr(torch.version, attr, None) is not None for attr in ["cuda", "hip", "xpu"]):
        log.warning("triton not found; flop counting will not work for triton kernels")
    _JITFunction = NoneType


aten = torch.ops.aten

def get_shape(i):
    if isinstance(i, torch.Tensor):
        return i.shape
    return i

flop_registry: dict[Any, Any] = {}

def shape_wrapper(f):
    @wraps(f)
    def nf(*args, out_val=None, **kwargs):
        args, kwargs, out_shape = tree_map(get_shape, (args, kwargs, out_val))
        return f(*args, out_shape=out_shape, **kwargs)
    return nf

def register_flop_formula(targets, get_raw=False) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:

    def register_fun(flop_formula: Callable[_P, _T]) -> Callable[_P, _T]:
        if not get_raw:
            flop_formula = shape_wrapper(flop_formula)

        def register(target) -> None:
            from torch._ops import HigherOrderOperator

            if not (isinstance(target, (torch._ops.OpOverloadPacket, _JITFunction, HigherOrderOperator))):
                raise ValueError(
                    f"register_flop_formula(targets): expected each target to be "
                    f"OpOverloadPacket (i.e. torch.ops.mylib.foo), JitFunction, "
                    f"or HigherOrderOperator"
                    f", got {target} which is of type {type(target)}")
            if target in flop_registry:
                raise RuntimeError(f"duplicate registrations for {target}")
            flop_registry[target] = flop_formula

        # To handle allowing multiple aten_ops at once
        torch.utils._pytree.tree_map_(register, targets)

        return flop_formula

    return register_fun

@register_flop_formula(aten.mm)
def mm_flop(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for matmul."""
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    m, k = a_shape
    k2, n = b_shape
    if k != k2:
        raise AssertionError(f"matmul: inner dimensions must match (k == k2), got {k} and {k2}")
    # NB(chilli): Should be 2 * k - 1 technically for FLOPs.
    return m * n * 2 * k

@register_flop_formula(aten.addmm)
def addmm_flop(self_shape, a_shape, b_shape, *args, **kwargs) -> int:
    """Count flops for addmm."""
    return mm_flop(a_shape, b_shape)

@register_flop_formula(aten.bmm)
def bmm_flop(a_shape, b_shape, *args, **kwargs) -> int:
    """Count flops for the bmm operation."""
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    b, m, k = a_shape
    b2, k2, n = b_shape
    if b != b2:
        raise AssertionError(f"bmm: batch dimensions must match (b == b2), got {b} and {b2}")
    if k != k2:
        raise AssertionError(f"bmm: inner dimensions must match (k == k2), got {k} and {k2}")
    # NB(chilli): Should be 2 * k - 1 technically for FLOPs.
    flop = b * m * n * 2 * k
    return flop

@register_flop_formula(aten.baddbmm)
def baddbmm_flop(self_shape, a_shape, b_shape, *args, **kwargs) -> int:
    """Count flops for the baddbmm operation."""
    # Inputs should be a list of length 3.
    # Inputs contains the shapes of three tensors.
    return bmm_flop(a_shape, b_shape)

@register_flop_formula(aten._scaled_mm)
def _scaled_mm_flop(
    a_shape,
    b_shape,
    scale_a_shape,
    scale_b_shape,
    bias_shape=None,
    scale_result_shape=None,
    out_dtype=None,
    use_fast_accum=False,
    out_shape=None,
    **kwargs,
) -> int:
    """Count flops for _scaled_mm."""
    return mm_flop(a_shape, b_shape)


def conv_flop_count(
    x_shape: list[int],
    w_shape: list[int],
    out_shape: list[int],
    transposed: bool = False,
) -> int:
    """Count flops for convolution.

    Note only multiplication is
    counted. Computation for bias are ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    c_out, c_in, *filter_size = w_shape

    """
    General idea here is that for a regular conv, for each point in the output
    spatial dimension we convolve the filter with something (hence
    `prod(conv_shape) * prod(filter_size)` ops). Then, this gets multiplied by
    1. batch_size, 2. the cross product of input and weight channels.

    For the transpose, it's not each point in the *output* spatial dimension but
    each point in the *input* spatial dimension.
    """
    # NB(chilli): I don't think this properly accounts for padding :think:
    # NB(chilli): Should be 2 * c_in - 1 technically for FLOPs.
    flop = prod(conv_shape) * prod(filter_size) * batch_size * c_out * c_in * 2
    return flop

@register_flop_formula([aten.convolution,
                        aten._convolution,
                        aten.cudnn_convolution,
                        aten._slow_conv2d_forward,
                        aten.convolution_overrideable])
def conv_flop(x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, *args, out_shape=None, **kwargs) -> int:
    """Count flops for convolution."""
    # pyrefly: ignore [bad-argument-type]
    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)


@register_flop_formula(aten.convolution_backward)
def conv_backward_flop(
        grad_out_shape,
        x_shape,
        w_shape,
        _bias,
        _stride,
        _padding,
        _dilation,
        transposed,
        _output_padding,
        _groups,
        output_mask,
        out_shape,
        **kwargs) -> int:

    def t(shape):
        return [shape[1], shape[0]] + list(shape[2:])
    flop_count = 0

    """
    Let's say we have a regular 1D conv
    {A, B, C} [inp]
    {i, j} [weight]
    => (conv)
    {Ai + Bj, Bi + Cj} [out]

    And as a reminder, the transposed conv of the above is
    => {Ai, Aj + Bi, Bj + Ci, Cj} [transposed conv out]

    For the backwards of conv, we now have
    {D, E} [grad_out]
    {A, B, C} [inp]
    {i, j} [weight]

    # grad_inp as conv_transpose(grad_out, weight)
    Let's first compute grad_inp. To do so, we can simply look at all the
    multiplications that each element of inp is involved in. For example, A is
    only involved in the first element of the output (and thus only depends upon
    D in grad_out), and C is only involved in the last element of the output
    (and thus only depends upon E in grad_out)

    {Di, Dj + Ei, Ej} [grad_inp]

    Note that this corresponds to the below conv_transpose. This gives us the
    output_mask[0] branch, which is grad_inp.

    {D, E} [inp (grad_out)]
    {i, j} [weight]
    => (conv_transpose)
    {Di, Dj + Ei, Ej} [out (grad_inp)]

    I leave the fact that grad_inp for a transposed conv is just conv(grad_out,
    weight) as an exercise for the reader.

    # grad_weight as conv(inp, grad_out)
    To compute grad_weight, we again look at the terms in the output, which as
    a reminder is:
    => {Ai + Bj, Bi + Cj} [out]
    => {D, E} [grad_out]
    If we manually compute the gradient for the weights, we see it's
    {AD + BE, BD + CE} [grad_weight]

    This corresponds to the below conv
    {A, B, C} [inp]
    {D, E} [weight (grad_out)]
    => (conv)
    {AD + BE, BD + CE} [out (grad_weight)]

    # grad_weight of transposed conv as conv(grad_out, inp)
    As a reminder, the terms of the output of a transposed conv are:
    => {Ai, Aj + Bi, Bj + Ci, Cj} [transposed conv out]
    => {D, E, F, G} [grad_out]

    Manually computing the gradient for the weights, we see it's
    {AD + BE + CF, AE + BF + CG} [grad_weight]

    This corresponds to the below conv
    {D, E, F, G} [inp (grad_out)]
    {A, B, C} [weight (inp)]
    => (conv)
    {AD + BE + CF, AE + BF + CG} [out (grad_weight)]

    For the full backwards formula, there are also some details involving
    transpose of the batch/channel dimensions and groups, but I skip those for
    the sake of brevity (and they're pretty similar to matmul backwards)

    Check [conv backwards decomposition as conv forwards]
    """
    # grad_inp as conv_transpose(grad_out, weight)
    if output_mask[0]:
        grad_input_shape = get_shape(out_shape[0])
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not transposed)

    if output_mask[1]:
        grad_weight_shape = get_shape(out_shape[1])
        if transposed:
            # grad_weight of transposed conv as conv(grad_out, inp)
            flop_count += conv_flop_count(t(grad_out_shape), t(x_shape), t(grad_weight_shape), transposed=False)
        else:
            # grad_weight as conv(inp, grad_out)
            flop_count += conv_flop_count(t(x_shape), t(grad_out_shape), t(grad_weight_shape), transposed=False)

    return flop_count

def sdpa_flop_count(query_shape, key_shape, value_shape):
    """
    Count flops for self-attention.

    Supports GQA (grouped-query attention) where key/value have fewer heads
    than the query. The kernel broadcasts KV heads to match query heads.
    """
    b, h_q, s_q, d_q = query_shape
    _b2, h_kv, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    if not (b == _b2 == _b3 and h_kv == _h3 and d_q == _d2 and s_k == _s3):
        raise AssertionError(
            f"sdpa_flop_count: query/key/value shapes are incompatible: "
            f"q={query_shape}, k={key_shape}, v={value_shape}"
        )
    if h_q < h_kv or h_q % h_kv != 0:
        raise AssertionError(
            f"sdpa_flop_count: query heads ({h_q}) must be a multiple of "
            f"key/value heads ({h_kv})"
        )
    total_flops = 0
    # q: [b, h_q, s_q, d_q] @ k: [b, h_q, d_q, s_k] -> scores: [b, h_q, s_q, s_k]
    total_flops += bmm_flop((b * h_q, s_q, d_q), (b * h_q, d_q, s_k))
    # scores: [b, h_q, s_q, s_k] @ v: [b, h_q, s_k, d_v] -> out: [b, h_q, s_q, d_v]
    total_flops += bmm_flop((b * h_q, s_q, s_k), (b * h_q, s_k, d_v))
    return total_flops


@register_flop_formula([aten._scaled_dot_product_efficient_attention,
                        aten._scaled_dot_product_flash_attention,
                        aten._scaled_dot_product_cudnn_attention])
def sdpa_flop(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for self-attention."""
    # NB: We aren't accounting for causal attention here
    return sdpa_flop_count(query_shape, key_shape, value_shape)


def _offsets_to_lengths(offsets, max_len):
    """
    If the offsets tensor is fake, then we don't know the actual lengths.
    In that case, we can just assume the worst case; each batch has max length.
    """
    from torch._subclasses.fake_tensor import is_fake_tensor
    from torch._subclasses.functional_tensor import FunctionalTensor
    if not (is_fake_tensor(offsets) or isinstance(offsets, FunctionalTensor)) and offsets.device.type != "meta":
        return offsets.diff().tolist()
    return [max_len] * (offsets.size(0) - 1)


def _unpack_flash_attention_nested_shapes(
    *,
    query,
    key,
    value,
    grad_out=None,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...] | None]]:
    """
    Given inputs to a flash_attention_(forward|backward) kernel, this will handle behavior for
    NestedTensor inputs by effectively unbinding the NestedTensor and yielding the shapes for
    each batch element.

    In the case that this isn't a NestedTensor kernel, then it just yields the original shapes.
    """
    if cum_seq_q is not None:
        # This means we should be dealing with a Nested Jagged Tensor query.
        # The inputs will have shape                  (sum(sequence len), heads, dimension)
        # In comparison, non-Nested inputs have shape (batch, heads, sequence len, dimension)
        # To deal with this, we convert to a shape of (batch, heads, max_seq_len, dimension)
        # So the flops calculation in this case is an overestimate of the actual flops.
        if len(key.shape) != 3:
            raise AssertionError("sdpa_flop_count: expected key.shape to be 3-dimensional")
        if len(value.shape) != 3:
            raise AssertionError("sdpa_flop_count: expected value.shape to be 3-dimensional")
        if grad_out is not None and grad_out.shape != query.shape:
            raise AssertionError("sdpa_flop_count: grad_out.shape must match query.shape when provided")
        _, h_q, d_q = query.shape
        _, h_k, d_k = key.shape
        _, h_v, d_v = value.shape
        if cum_seq_q is None:
            raise AssertionError("sdpa_flop_count: cum_seq_q must not be None")
        if cum_seq_k is None:
            raise AssertionError("sdpa_flop_count: cum_seq_k must not be None")
        if cum_seq_q.shape != cum_seq_k.shape:
            raise AssertionError("sdpa_flop_count: cum_seq_q and cum_seq_k must have the same shape")
        seq_q_lengths = _offsets_to_lengths(cum_seq_q, max_q)
        seq_k_lengths = _offsets_to_lengths(cum_seq_k, max_k)
        for (seq_q_len, seq_k_len) in zip(seq_q_lengths, seq_k_lengths, strict=True):
            new_query_shape = (1, h_q, seq_q_len, d_q)
            new_key_shape = (1, h_k, seq_k_len, d_k)
            new_value_shape = (1, h_v, seq_k_len, d_v)
            new_grad_out_shape = new_query_shape if grad_out is not None else None
            yield new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape
        return

    yield query.shape, key.shape, value.shape, grad_out.shape if grad_out is not None else None


def _unpack_efficient_attention_nested_shapes(
    *,
    query,
    key,
    value,
    grad_out=None,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...] | None]]:
    """
    Given inputs to a efficient_attention_(forward|backward) kernel, this will handle behavior for
    NestedTensor inputs by effectively unbinding the NestedTensor and yielding the shapes for
    each batch element.

    In the case that this isn't a NestedTensor kernel, then it just yields the original shapes.
    """
    if cu_seqlens_q is not None:
        # Unlike flash_attention_forward, we get a 4D tensor instead of a 3D tensor for efficient attention.
        #
        # This means we should be dealing with a Nested Jagged Tensor query.
        # The inputs will have shape                  (sum(sequence len), heads, dimension)
        # In comparison, non-Nested inputs have shape (batch, heads, sequence len, dimension)
        # To deal with this, we convert to a shape of (batch, heads, max_seq_len, dimension)
        # So the flops calculation in this case is an overestimate of the actual flops.
        if len(key.shape) != 4:
            raise AssertionError("_unpack_efficient_attention_nested_shapes: expected key.shape to be 4-dimensional")
        if len(value.shape) != 4:
            raise AssertionError("_unpack_efficient_attention_nested_shapes: expected value.shape to be 4-dimensional")
        if grad_out is not None and grad_out.shape != query.shape:
            raise AssertionError("_unpack_efficient_attention_nested_shapes: grad_out.shape must match query.shape when provided")
        _, _, h_q, d_q = query.shape
        _, _, h_k, d_k = key.shape
        _, _, h_v, d_v = value.shape
        if cu_seqlens_q is None:
            raise AssertionError("_unpack_efficient_attention_nested_shapes: cu_seqlens_q must not be None")
        if cu_seqlens_k is None:
            raise AssertionError("_unpack_efficient_attention_nested_shapes: cu_seqlens_k must not be None")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise AssertionError("_unpack_efficient_attention_nested_shapes: "
                                 "cu_seqlens_q and cu_seqlens_k must have the same shape")
        seqlens_q = _offsets_to_lengths(cu_seqlens_q, max_seqlen_q)
        seqlens_k = _offsets_to_lengths(cu_seqlens_k, max_seqlen_k)
        for len_q, len_k in zip(seqlens_q, seqlens_k, strict=True):
            new_query_shape = (1, h_q, len_q, d_q)
            new_key_shape = (1, h_k, len_k, d_k)
            new_value_shape = (1, h_v, len_k, d_v)
            new_grad_out_shape = new_query_shape if grad_out is not None else None
            yield new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape
        return

    yield query.shape, key.shape, value.shape, grad_out.shape if grad_out is not None else None


@register_flop_formula(aten._flash_attention_forward, get_raw=True)
def _flash_attention_forward_flop(
    query,
    key,
    value,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    *args,
    out_shape=None,
    **kwargs
) -> int:
    """Count flops for self-attention."""
    # NB: We aren't accounting for causal attention here
    if cum_seq_q is None and query.ndim == 4:
        query = query.transpose(-2, -3)
        key = key.transpose(-2, -3)
        value = value.transpose(-2, -3)
    # in case this is a nested tensor, we unpack the individual batch elements
    # and then sum the flops per batch element
    sizes = _unpack_flash_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
    )
    return sum(
        sdpa_flop_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )


@register_flop_formula(aten._efficient_attention_forward, get_raw=True)
def _efficient_attention_forward_flop(
    query,
    key,
    value,
    bias,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *args,
    **kwargs
) -> int:
    """Count flops for self-attention."""
    # NB: We aren't accounting for causal attention here
    # in case this is a nested tensor, we unpack the individual batch elements
    # and then sum the flops per batch element
    sizes = _unpack_efficient_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return sum(
        sdpa_flop_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )


def sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape):
    b, h_q, s_q, d_q = query_shape
    _b2, h_kv, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    _b4, _h4, _s4, _d4 = grad_out_shape
    if not (b == _b2 == _b3 == _b4 and h_kv == _h3 and h_q == _h4):
        raise AssertionError(
            "sdpa_backward_flop_count: batch/heads mismatch among tensors"
        )
    if h_q < h_kv or h_q % h_kv != 0:
        raise AssertionError(
            f"sdpa_backward_flop_count: query heads ({h_q}) must be a multiple of "
            f"key/value heads ({h_kv})"
        )
    if not (d_q == _d2 and d_v == _d4 and s_k == _s3 and s_q == _s4):
        raise AssertionError(
            "sdpa_backward_flop_count: grad_out/value/key/query shapes are incompatible"
        )
    total_flops = 0
    # Step 1: We recompute the scores matrix.
    # q: [b, h_q, s_q, d_q] @ k: [b, h_q, d_q, s_k] -> scores: [b, h_q, s_q, s_k]
    total_flops += bmm_flop((b * h_q, s_q, d_q), (b * h_q, d_q, s_k))

    # Step 2: We propagate the gradients through the score @ v operation.
    # gradOut: [b, h_q, s_q, d_v] @ v: [b, h_q, d_v, s_k] -> gradScores: [b, h_q, s_q, s_k]
    total_flops += bmm_flop((b * h_q, s_q, d_v), (b * h_q, d_v, s_k))
    # scores: [b, h_q, s_k, s_q] @ gradOut: [b, h_q, s_q, d_v] -> gradV: [b, h_q, s_k, d_v]
    total_flops += bmm_flop((b * h_q, s_k, s_q), (b * h_q, s_q, d_v))

    # Step 3: We propagate the gradients through the k @ v operation
    # gradScores: [b, h_q, s_q, s_k] @ k: [b, h_q, s_k, d_q] -> gradQ: [b, h_q, s_q, d_q]
    total_flops += bmm_flop((b * h_q, s_q, s_k), (b * h_q, s_k, d_q))
    # q: [b, h_q, d_q, s_q] @ gradScores: [b, h_q, s_q, s_k] -> gradK: [b, h_q, d_q, s_k]
    total_flops += bmm_flop((b * h_q, d_q, s_q), (b * h_q, s_q, s_k))
    return total_flops


@register_flop_formula([aten._scaled_dot_product_efficient_attention_backward,
                        aten._scaled_dot_product_flash_attention_backward,
                        aten._scaled_dot_product_cudnn_attention_backward])
def sdpa_backward_flop(grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for self-attention backward."""
    return sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)

@register_flop_formula(aten._flash_attention_backward, get_raw=True)
def _flash_attention_backward_flop(
    grad_out,
    query,
    key,
    value,
    out,  # named _out_shape to avoid kwarg collision with out_shape created in wrapper
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    *args,
    **kwargs,
) -> int:
    if cum_seq_q is None and query.ndim == 4:
        grad_out = grad_out.transpose(-2, -3)
        query = query.transpose(-2, -3)
        key = key.transpose(-2, -3)
        value = value.transpose(-2, -3)
    # in case this is a nested tensor, we unpack the individual batch elements
    # and then sum the flops per batch element
    shapes = _unpack_flash_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        grad_out=grad_out,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
    )
    return sum(
        sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in shapes
    )


@register_flop_formula(aten._efficient_attention_backward, get_raw=True)
def _efficient_attention_backward_flop(
    grad_out,
    query,
    key,
    value,
    bias,
    out,  # named _out to avoid kwarg collision with out created in wrapper
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *args,
    **kwargs,
) -> int:
    # in case this is a nested tensor, we unpack the individual batch elements
    # and then sum the flops per batch element
    shapes = _unpack_efficient_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        grad_out=grad_out,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return sum(
        sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in shapes
    )


def _register_flex_attention_flops() -> None:
    from torch._higher_order_ops.flex_attention import (
        flex_attention,
        flex_attention_backward,
    )

    def _get_sparsity_hint(kwargs: dict[str, Any]) -> float:
        node_meta = kwargs.get("_node_meta")
        if node_meta is None:
            return 0.0
        custom = node_meta.get("custom")
        if custom is None:
            return 0.0
        return max(0.0, min(1.0, custom.get("sparsity_hint", 0.0)))

    @register_flop_formula(flex_attention, get_raw=True)
    def flex_attention_forward_flop(
        query, key, value, *args, out_val=None, **kwargs
    ) -> int:
        flops = sdpa_flop_count(query.shape, key.shape, value.shape)
        sparsity = _get_sparsity_hint(kwargs)
        return int(flops * (1.0 - sparsity)) if sparsity > 0 else flops

    @register_flop_formula(flex_attention_backward, get_raw=True)
    def flex_attention_backward_flop(
        query, key, value, out, logsumexp, grad_out, *args, out_val=None, **kwargs
    ) -> int:
        grad_out_shape = grad_out.shape if grad_out is not None else out.shape
        flops = sdpa_backward_flop_count(
            grad_out_shape, query.shape, key.shape, value.shape
        )
        sparsity = _get_sparsity_hint(kwargs)
        return int(flops * (1.0 - sparsity)) if sparsity > 0 else flops


def _varlen_attn_forward_flop(
    query,
    key,
    value,
    cu_seq_q,
    cu_seq_k,
    max_q,
    max_k,
    *args,
    out_val=None,
    **kwargs,
) -> int:
    """Count flops for varlen_attn forward."""
    sizes = _unpack_flash_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        cum_seq_q=cu_seq_q,
        cum_seq_k=cu_seq_k if cu_seq_k is not None else cu_seq_q,
        max_q=max_q,
        max_k=max_k,
    )
    return sum(
        sdpa_flop_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )


def _varlen_attn_out_flop(
    out,
    query,
    key,
    value,
    cu_seq_q,
    cu_seq_k,
    max_q,
    max_k,
    *args,
    out_val=None,
    **kwargs,
) -> int:
    """Count flops for varlen_attn_out forward."""
    return _varlen_attn_forward_flop(
        query, key, value, cu_seq_q, cu_seq_k, max_q, max_k,
    )


def _varlen_attn_backward_flop(
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
    *args,
    out_val=None,
    **kwargs,
) -> int:
    """Count flops for varlen_attn backward."""
    sizes = _unpack_flash_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        grad_out=grad_out,
        cum_seq_q=cu_seq_q,
        cum_seq_k=cu_seq_k,
        max_q=max_q,
        max_k=max_k,
    )
    return sum(
        sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in sizes
    )


flop_registry = {
    aten.mm: mm_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.baddbmm: baddbmm_flop,
    aten._scaled_mm: _scaled_mm_flop,
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten.cudnn_convolution: conv_flop,
    aten.convolution_overrideable: conv_flop,
    aten._slow_conv2d_forward: conv_flop,
    aten.convolution_backward: conv_backward_flop,
    aten._scaled_dot_product_efficient_attention: sdpa_flop,
    aten._scaled_dot_product_flash_attention: sdpa_flop,
    aten._scaled_dot_product_cudnn_attention: sdpa_flop,
    aten._scaled_dot_product_efficient_attention_backward: sdpa_backward_flop,
    aten._scaled_dot_product_flash_attention_backward: sdpa_backward_flop,
    aten._scaled_dot_product_cudnn_attention_backward: sdpa_backward_flop,
    aten._flash_attention_forward: _flash_attention_forward_flop,
    aten._efficient_attention_forward: _efficient_attention_forward_flop,
    aten._flash_attention_backward: _flash_attention_backward_flop,
    aten._efficient_attention_backward: _efficient_attention_backward_flop,
}

_register_flex_attention_flops()


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


# Define the suffixes for different orders of magnitude
suffixes = ["", "K", "M", "B", "T"]
# Thanks BingChat!
def get_suffix_str(number):
    # Find the index of the appropriate suffix based on the number of digits
    # with some additional overflow.
    # i.e. 1.01B should be displayed as 1001M, not 1.001B
    index = max(0, min(len(suffixes) - 1, (len(str(number)) - 2) // 3))
    return suffixes[index]

def convert_num_with_suffix(number, suffix):
    index = suffixes.index(suffix)
    # Divide the number by 1000^index and format it to two decimal places
    value = f"{number / 1000 ** index:.3f}"
    # Return the value and the suffix as a string
    return value + suffixes[index]

def convert_to_percent_str(num, denom) -> str:
    if denom == 0:
        return "0%"
    return f"{num / denom:.2%}"

def _pytreeify_preserve_structure(f):
    @wraps(f)
    def nf(args):
        flat_args, spec = tree_flatten(args)
        out = f(*flat_args)
        return tree_unflatten(out, spec)

    return nf


class FlopCounterMode:
    """Count theoretical FLOPs for operators that run inside the context.

    ``FlopCounterMode`` uses ``TorchDispatchMode`` to intercept operations and
    apply registered FLOP formulas. Counts are shape-based and formula-defined;
    unsupported operations contribute zero FLOPs unless they decompose into
    supported operations.

    The optional ``mods`` argument is no longer needed for module attribution.
    When ``display`` is true, exiting the context prints a table. Use
    ``get_total_flops()`` or ``get_flop_counts()`` to read the same information
    programmatically.

    Example usage:

    .. code-block:: python

        mod = ...
        inp = ...

        with FlopCounterMode(display=False) as flop_counter:
            mod(inp).sum().backward()

        total = flop_counter.get_total_flops()

    """

    def __init__(
            self,
            mods: torch.nn.Module | list[torch.nn.Module] | None = None,
            depth: int = 2,
            display: bool = True,
            custom_mapping: dict[Any, Any] | None = None) -> None:
        super().__init__()
        self.flop_counts: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.depth = depth
        self.display = display
        self.mode: _FlopCounterMode | None = None
        if custom_mapping is None:
            custom_mapping = {}
        if mods is not None:
            warnings.warn("mods argument is not needed anymore, you can stop passing it", stacklevel=2)
        self.flop_registry = {
            **flop_registry,
            **{k: v if getattr(v, "_get_raw", False) else shape_wrapper(v) for k, v in custom_mapping.items()}
        }
        self.mod_tracker = ModuleTracker()

    def get_total_flops(self) -> int:
        return sum(self.flop_counts['Global'].values())

    def get_flop_counts(self) -> dict[str, dict[Any, int]]:
        """Return the flop counts as a dictionary of dictionaries.

        The outer
        dictionary is keyed by module name, and the inner dictionary is keyed by
        operation name.

        Returns:
            Dict[str, Dict[Any, int]]: The flop counts as a dictionary.
        """
        return {k: dict(v) for k, v in self.flop_counts.items()}

    def get_table(self, depth=None):
        if depth is None:
            depth = self.depth
        if depth is None:
            depth = 999999


        import tabulate

        tabulate.PRESERVE_WHITESPACE = True
        header = ["Module", "FLOP", "% Total"]
        values = []
        global_flops = self.get_total_flops()
        global_suffix = get_suffix_str(global_flops)
        is_global_subsumed = False

        def process_mod(mod_name, depth):
            nonlocal is_global_subsumed

            total_flops = sum(self.flop_counts[mod_name].values())

            is_global_subsumed |= total_flops >= global_flops

            padding = " " * depth
            values = []
            values.append([
                padding + mod_name,
                convert_num_with_suffix(total_flops, global_suffix),
                convert_to_percent_str(total_flops, global_flops)
            ])
            for k, v in self.flop_counts[mod_name].items():
                values.append([
                    padding + " - " + str(k),
                    convert_num_with_suffix(v, global_suffix),
                    convert_to_percent_str(v, global_flops)
                ])
            return values

        for mod in sorted(self.flop_counts.keys()):
            if mod == 'Global':
                continue
            mod_depth = mod.count(".") + 1
            if mod_depth > depth:
                continue

            cur_values = process_mod(mod, mod_depth - 1)
            values.extend(cur_values)

        # We do a bit of messing around here to only output the "Global" value
        # if there are any FLOPs in there that aren't already fully contained by
        # a module.
        if 'Global' in self.flop_counts and not is_global_subsumed:
            for value in values:
                value[0] = " " + value[0]

            values = process_mod('Global', 0) + values

        if len(values) == 0:
            values = [["Global", "0", "0%"]]

        return tabulate.tabulate(values, headers=header, colalign=("left", "right", "right"))

    # NB: This context manager is NOT reentrant
    def __enter__(self):
        self.flop_counts.clear()
        self.mod_tracker.__enter__()
        self.mode = _FlopCounterMode(self)
        self.mode.__enter__()
        return self

    def __exit__(self, *args):
        if self.mode is None:
            raise AssertionError("Internal error: FlopCounter.__exit__ called but mode is None")
        b = self.mode.__exit__(*args)
        self.mode = None  # break cycles
        self.mod_tracker.__exit__()
        if self.display:
            print(self.get_table(self.depth))
        return b

    def _count_flops(self, func_packet, out, args, kwargs):
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out_val=out)  # type: ignore[operator]
            for par in set(self.mod_tracker.parents):
                self.flop_counts[par][func_packet] += flop_count
        return out

class _FlopCounterMode(TorchDispatchMode):
    supports_higher_order_operators = True

    def __init__(self, counter: FlopCounterMode) -> None:
        self.counter = counter

    def _execute_with_isolated_flop_counting(self, branch_fn, operands):
        """Execute a branch function and capture its FLOP counts without
        affecting self.counter.flop_counts

        Args:
            branch_fn: The branch function to execute
            operands: Arguments to pass to the branch function

        Returns:
            Tuple of (result, flop_counts) where result is the branch output
            and flop_counts is a copy of the FLOP counts after execution
        """
        import copy
        checkpointed_flop_counts = copy.copy(self.counter.flop_counts)
        with self:
            result = branch_fn(*operands)
        flop_counts = copy.copy(self.counter.flop_counts)
        self.counter.flop_counts = checkpointed_flop_counts
        return result, flop_counts

    def _handle_higher_order_ops(self, func, types, args, kwargs):
        is_triton = func in {torch.ops.higher_order.triton_kernel_wrapper_mutation,
                             torch.ops.higher_order.triton_kernel_wrapper_functional}
        if is_triton:
            from torch._higher_order_ops.triton_kernel_wrap import get_kernel
            # Special case - look in the triton flop registry for the kernel
            from triton.runtime.jit import JITFunction
            kernel_name = get_kernel(kwargs["kernel_idx"])
            # Unwrap heuristics if they are present
            while not isinstance(kernel_name, JITFunction):
                if hasattr(kernel_name, "fn"):
                    kernel_name = kernel_name.fn
                else:
                    break
            return self.counter._count_flops(kernel_name, None, args, kwargs)
        elif func is torch.ops.higher_order.cond:
            # The flop counter for cond counts the upper bound of flops.
            # For example, if a matmul is executed 2 times in true branch
            # but only 1 time in the false branch, the flop counter will
            # record the larger number of flops, i.e. 2 times.
            pred, true_branch, false_branch, operands = args
            # Step 1: Count flops for true branch and false branch separately
            true_out, true_flop_counts = self._execute_with_isolated_flop_counting(
                true_branch, operands
            )
            if true_out is NotImplemented:
                return NotImplemented

            false_out, false_flop_counts = self._execute_with_isolated_flop_counting(
                false_branch, operands
            )
            if false_out is NotImplemented:
                return NotImplemented

            # Step 2: merge flop counts
            all_mod_keys = set(true_flop_counts.keys()) | set(false_flop_counts.keys())
            merged_flop_counts = {}
            for outer_key in all_mod_keys:
                true_func_counts = true_flop_counts[outer_key]
                false_func_counts = false_flop_counts[outer_key]

                merged_func_counts = {}
                all_func_keys = set(true_func_counts.keys()) | set(false_func_counts.keys())

                for func_key in all_func_keys:
                    true_val = true_func_counts.get(func_key, 0)
                    false_val = false_func_counts.get(func_key, 0)
                    merged_func_counts[func_key] = max(true_val, false_val)

                merged_flop_counts[outer_key] = merged_func_counts

            # Step 3: update the counter with merged counts
            for outer_key, inner_dict in merged_flop_counts.items():
                self.counter.flop_counts[outer_key].update(inner_dict)

            # It doesn't matter which one we return since true_fn and false_fn return
            # output with the same structure.
            return true_out
        else:
            return NotImplemented

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        # Skip ops from non-standard dispatch_sizes_strides_policy such as NJT
        if func in {torch.ops.aten.sym_is_contiguous.default,
                    torch.ops.aten.is_contiguous.default,
                    torch.ops.aten.is_contiguous.memory_format,
                    torch.ops.aten.is_strides_like_format.default,
                    torch.ops.aten.is_non_overlapping_and_dense.default,
                    torch.ops.aten.size.default,
                    torch.ops.aten.sym_size.default,
                    torch.ops.aten.stride.default,
                    torch.ops.aten.sym_stride.default,
                    torch.ops.aten.storage_offset.default,
                    torch.ops.aten.sym_storage_offset.default,
                    torch.ops.aten.numel.default,
                    torch.ops.aten.sym_numel.default,
                    torch.ops.aten.dim.default,
                    torch.ops.prim.layout.default}:

            return NotImplemented

        if isinstance(func, torch._ops.HigherOrderOperator):
            return self._handle_higher_order_ops(func, types, args, kwargs)

        # If we don't have func in flop_registry, see if it can decompose
        if func not in self.counter.flop_registry and func is not torch.ops.prim.device.default:
            with self:
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        # no further decomposition; execute & count flops
        out = func(*args, **kwargs)
        return self.counter._count_flops(func._overloadpacket, out, args, kwargs)
<|endoftext|>from __future__ import annotations

import textwrap
import traceback
import weakref
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_iter


if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE


class _TrackedTensorInfo:
    __slots__ = (
        "weakref",
        "creation_traceback_py",
        "creation_traceback_cpp",
        "deletion_traceback_py",
        "data_ptr",
        "device_index",
        "_storage_data_ptr",
    )

    def __init__(self, tensor: Tensor) -> None:
        from torch.utils import get_cpp_backtrace

        self.creation_traceback_py = "".join(traceback.format_stack()[:-3])
        self.creation_traceback_cpp = get_cpp_backtrace()
        self.deletion_traceback_py: str | None = None
        self.data_ptr = tensor.data_ptr()
        self.device_index = tensor.device.index

        def on_release(_: weakref.ref[torch.UntypedStorage]) -> None:
            try:
                self.deletion_traceback_py = "".join(traceback.format_stack()[:-1])
            except Exception:
                pass  # don't raise under GC

        storage = tensor.untyped_storage()
        self._storage_data_ptr = storage.data_ptr()
        self.weakref: weakref.ref[torch.UntypedStorage] = weakref.ref(
            storage, on_release
        )

    def is_alive(self) -> bool:
        storage = self.weakref()
        if storage is None:
            return False
        return storage.data_ptr() == self._storage_data_ptr


class _CUDAGraphInputLivenessTracker:
    def __init__(self) -> None:
        self._external_inputs: dict[int, _TrackedTensorInfo] = {}
        self._internal_outputs: set[int] = set()
        self._memory_snapshots: dict[_POOL_HANDLE, list[dict[str, Any]]] = {}
        self._dispatch_mode: _TensorTrackingMode | None = None

    def start(self) -> None:
        self._dispatch_mode = _TensorTrackingMode(self)
        self._dispatch_mode.__enter__()

    def stop(self) -> None:
        if self._dispatch_mode is not None:
            mode, self._dispatch_mode = self._dispatch_mode, None
            mode.__exit__(None, None, None)

    def track_external_input(self, tensor: Tensor) -> None:
        data_ptr = tensor.data_ptr()
        if (
            data_ptr not in self._internal_outputs
            and data_ptr not in self._external_inputs
        ):
            self._external_inputs[data_ptr] = _TrackedTensorInfo(tensor)

    def mark_internal_output(self, tensor: Tensor) -> None:
        data_ptr = tensor.data_ptr()
        if data_ptr not in self._external_inputs:
            self._internal_outputs.add(data_ptr)

    def check_alive(self, capture_pools: list[_POOL_HANDLE]) -> None:
        dead = [
            i
            for i in self._external_inputs.values()
            if not i.is_alive()
            and not any(
                self._is_tensor_from_capture_pool(i, pool) for pool in capture_pools
            )
        ]
        if not dead:
            return

        def fmt(label: str, tb: str | None) -> str:
            return f"  {label}:\n{textwrap.indent(tb.strip(), '    ')}\n" if tb else ""

        parts = [f"CUDA graph replay detected {len(dead)} dead tensor(s).\n"]
        for i, info in enumerate(dead[:5], 1):
            parts.append(f"Dead tensor #{i} (data_ptr={info.data_ptr:#x}):\n")
            parts.append(fmt("Creation Traceback (Python)", info.creation_traceback_py))
            parts.append(fmt("Creation Traceback (C++)", info.creation_traceback_cpp))
            parts.append(fmt("Deletion Traceback (Python)", info.deletion_traceback_py))
        if len(dead) > 5:
            parts.append(f"  ... and {len(dead) - 5} more\n")
        parts.append(
            fmt("Replay Traceback (Python)", "".join(traceback.format_stack()[:-2]))
        )
        raise RuntimeError("".join(parts))

    def _get_memory_snapshot(self, capture_pool: _POOL_HANDLE) -> list[dict[str, Any]]:
        if capture_pool not in self._memory_snapshots:
            self._memory_snapshots[capture_pool] = torch.cuda.memory.memory_snapshot(
                capture_pool, include_traces=False
            )
        return self._memory_snapshots[capture_pool]

    def _is_tensor_from_capture_pool(
        self, tensor: _TrackedTensorInfo, capture_pool: _POOL_HANDLE
    ) -> bool:
        device = tensor.device_index
        if device is None:
            return False
        tensor_ptr = tensor.data_ptr
        for meminfo in self._get_memory_snapshot(capture_pool):
            if meminfo["device"] != device:
                continue
            for block in meminfo["blocks"]:
                addr = block["address"]
                if (
                    addr <= tensor_ptr < addr + block["size"]
                    and "active" in block["state"]
                ):
                    return True
        return False


class _TensorTrackingMode(TorchDispatchMode):
    supports_higher_order_operators = True

    def __init__(self, tracker: _CUDAGraphInputLivenessTracker) -> None:
        self._tracker = tracker

    # Mirrors the CUDA dispatch key selection rule: if at least one tensor
    # input is a CUDA tensor, the CUDA dispatch key is selected and the op
    # runs (and is captured) on GPU.
    # See aten/src/ATen/core/dispatch/DispatchKeyExtractor.h for details.
    def _selects_cuda_dispatch_key(self, values: object) -> bool:
        for v in tree_iter(values):
            if isinstance(v, Tensor) and v.is_cuda:
                return True
        return False

    def __torch_dispatch__(
        self,
        func: object,
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        inputs = [args, kwargs]
        if torch.cuda.is_current_stream_capturing() and self._selects_cuda_dispatch_key(
            inputs
        ):
            out = func(*args, **kwargs)  # type: ignore[operator]
            self._track_inputs(inputs)
            self._mark_outputs(out)
        else:
            out = func(*args, **kwargs)  # type: ignore[operator]
        return out

    def _track_inputs(self, values: object) -> None:
        for v in tree_iter(values):
            if not isinstance(v, Tensor) or v.data_ptr() == 0:
                continue
            if v.is_cuda or v.is_pinned():
                self._tracker.track_external_input(v)

    def _mark_outputs(self, values: object) -> None:
        for v in tree_iter(values):
            if isinstance(v, Tensor) and v.is_cuda:
                self._tracker.mark_internal_output(v)
<|endoftext|># mypy: allow-untyped-defs
import torch
from collections import OrderedDict
import weakref
import warnings
from typing import Any

__all__ = ["RemovableHandle", "unserializable_hook", "warn_if_has_hooks", "BackwardHook"]

class RemovableHandle:
    r"""
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (Union[dict, List[dict]]): An additional dictionary or list of
            dictionaries whose keys will be deleted when the same keys are
            removed from ``hooks_dict``.
    """

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref: tuple = ()
        if isinstance(extra_dict, dict):
            self.extra_dict_ref = (weakref.ref(extra_dict),)
        elif isinstance(extra_dict, list):
            self.extra_dict_ref = tuple(weakref.ref(d) for d in extra_dict)

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        for ref in self.extra_dict_ref:
            extra_dict = ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __getstate__(self):
        if self.extra_dict_ref is None:
            return (self.hooks_dict_ref(), self.id)
        else:
            return (self.hooks_dict_ref(), self.id, tuple(ref() for ref in self.extra_dict_ref))

    def __setstate__(self, state) -> None:
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

        if len(state) < 3 or state[2] is None:
            self.extra_dict_ref = ()
        else:
            self.extra_dict_ref = tuple(weakref.ref(d) for d in state[2])

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()


def unserializable_hook(f):
    """
    Mark a function as an unserializable hook with this decorator.

    This suppresses warnings that would otherwise arise if you attempt
    to serialize a tensor that has a hook.
    """
    f.__torch_unserializable__ = True
    return f


def warn_if_has_hooks(tensor) -> None:
    if tensor._backward_hooks:
        for k in tensor._backward_hooks:
            hook = tensor._backward_hooks[k]
            if not hasattr(hook, "__torch_unserializable__"):
                warnings.warn(f"backward hook {repr(hook)} on tensor will not be "
                              "serialized.  If this is expected, you can "
                              "decorate the function with @torch.utils.hooks.unserializable_hook "
                              "to suppress this warning", stacklevel=2)

class BackwardHook:
    """
    A wrapper class to implement nn.Module backward hooks.

    It handles:
      - Ignoring non-Tensor inputs and replacing them by None before calling the user hook
      - Generating the proper Node to capture a set of Tensor's gradients
      - Linking the gradients captures for the outputs with the gradients captured for the input
      - Calling the user hook once both output and input gradients are available
    """

    def __init__(self, module, user_hooks, user_pre_hooks) -> None:
        self.user_hooks = user_hooks
        self.user_pre_hooks = user_pre_hooks
        self.module = module

        self.grad_outputs = None
        self.n_outputs = -1
        self.output_tensors_index = None
        self.n_inputs = -1
        self.input_tensors_index = None

    def _pack_with_none(self, indices, values, size):
        res = [None] * size
        for idx, val in zip(indices, values, strict=True):
            res[idx] = val

        return tuple(res)

    def _unpack_none(self, indices, values):
        res = [values[idx] for idx in indices]

        return tuple(res)

    def _set_user_hook(self, grad_fn) -> None:
        def hook(grad_input, _):
            if self.grad_outputs is None:
                # This happens because the gradient in your nn.Module flows to
                # the Module's input without " passing through the Module's
                # output, e.g. when you're doing double backward.
                return
            res = self._pack_with_none(self.input_tensors_index, grad_input, self.n_inputs)

            for hook in self.user_hooks:
                out = hook(self.module, res, self.grad_outputs)

                if out is None:
                    continue

                if len(out) != len(res):
                    raise RuntimeError("Backward hook returned an invalid number of grad_input, "
                                       f"got {len(out)}, but expected {len(res)}")

                res = out

            self.grad_outputs = None

            return self._unpack_none(self.input_tensors_index, res)

        grad_fn.register_hook(hook)

    def _apply_on_tensors(self, fn, args):
        # Can be used to apply the given function to the tensors contained in the
        # args. Will return updated args and the tensors indices
        tensors_idx = []
        tensors = []

        requires_grad = False
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tensors_idx.append(i)
                tensors.append(arg)
                requires_grad |= arg.requires_grad

        if not (requires_grad and torch.is_grad_enabled()):
            return args, None

        new_tensors = torch.nn.modules._functions.BackwardHookFunction.apply(*tensors)
        if len(new_tensors) == 0:
            raise RuntimeError("Cannot set Module backward hook for a Module with no input Tensors.")

        grad_fns = [t.grad_fn for t in new_tensors if t.grad_fn is not None and t.grad_fn.name() == "BackwardHookFunctionBackward"]
        if len(grad_fns) == 0:
            raise RuntimeError("Error while setting up backward hooks. Please open "
                               "an issue with a code sample to reproduce this.")

        fn(grad_fns[0])

        arg_list = list(args)
        for idx, val in zip(tensors_idx, new_tensors, strict=True):
            arg_list[idx] = val

        if type(args) is tuple:
            out = tuple(arg_list)
        else:
            out = type(args)(*arg_list)
        return out, tensors_idx

    def setup_input_hook(self, args):
        def fn(grad_fn) -> None:
            self._set_user_hook(grad_fn)

        res, input_idx = self._apply_on_tensors(fn, args)
        self.n_inputs = len(args)
        self.input_tensors_index = input_idx
        return res

    def setup_output_hook(self, args):
        def fn(grad_fn) -> None:
            def hook(_, grad_output):
                self.grad_outputs = self._pack_with_none(self.output_tensors_index,
                                                         grad_output,
                                                         self.n_outputs)

                if self.user_pre_hooks:
                    expected_len = len(self.grad_outputs)
                    for user_pre_hook in self.user_pre_hooks:
                        hook_grad_outputs = user_pre_hook(self.module, self.grad_outputs)
                        if hook_grad_outputs is None:
                            continue

                        actual_len = len(hook_grad_outputs)
                        if actual_len != expected_len:
                            raise RuntimeError("Backward pre hook returned an invalid number of grad_output, "
                                               f"got {actual_len}, but expected {expected_len}")
                        self.grad_outputs = hook_grad_outputs

                # We need to be able to clear self.grad_outputs but also return it
                local_grad_outputs = self.grad_outputs

                # Special case if no input required gradients, this hook should call the user
                # hook directly
                if self.input_tensors_index is None:
                    # Only full backward hooks (not pre-hooks) receive grad_input
                    # and are the subject of the warning; pre-hooks already ran
                    # above from grad_output. Skip both when none are registered.
                    if self.user_hooks:
                        warnings.warn("Full backward hook is firing when gradients are computed "
                                      "with respect to module outputs since no inputs require gradients. See "
                                      "https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook "
                                      "for more details.",
                                      stacklevel=5)
                        grad_inputs = self._pack_with_none([], [], self.n_inputs)
                        for user_hook in self.user_hooks:
                            res = user_hook(self.module, grad_inputs, self.grad_outputs)
                            if res is not None and not (isinstance(res, tuple) and all(el is None for el in res)):
                                raise RuntimeError("Backward hook for Modules where no input requires "
                                                   "gradient should always return None or None for all gradients.")
                    self.grad_outputs = None

                if local_grad_outputs is not None:
                    if self.output_tensors_index is None:
                        raise AssertionError("output_tensors_index should not be None when grad_outputs is not None")
                    return tuple(local_grad_outputs[i] for i in self.output_tensors_index)

            grad_fn.register_hook(hook)

        is_tuple = True
        if not isinstance(args, tuple):
            args = (args,)
            is_tuple = False

        res, output_idx = self._apply_on_tensors(fn, args)
        self.n_outputs = len(args)
        self.output_tensors_index = output_idx

        if not is_tuple:
            res = res[0]
        return res
<|endoftext|>import functools

import torch


@functools.cache
def has_jax_package() -> bool:
    """Check if JAX is installed."""
    try:
        import jax  # noqa: F401  # type: ignore[import-not-found]

        return True
    except ImportError:
        return False


@functools.cache
def has_pallas_package() -> bool:
    """Check if Pallas (JAX experimental) is available."""
    if not has_jax_package():
        return False
    try:
        from jax.experimental import (  # noqa: F401  # type: ignore[import-not-found]
            pallas as pl,
        )

        return True
    except ImportError:
        return False


@functools.cache
def get_jax_version(fallback: tuple[int, int, int] = (0, 0, 0)) -> tuple[int, int, int]:
    """Get JAX version as (major, minor, patch) tuple."""
    try:
        import jax  # type: ignore[import-not-found]

        version_parts = jax.__version__.split(".")
        major, minor, patch = (int(v) for v in version_parts[:3])
        return (major, minor, patch)
    except (ImportError, ValueError, AttributeError):
        return fallback


@functools.cache
def has_jax_cuda_backend() -> bool:
    """Check if JAX has CUDA backend support with SM90+ (required by Mosaic GPU)."""
    if not has_jax_package():
        return False
    try:
        import jax  # type: ignore[import-not-found]

        # Check if CUDA backend is available
        devices = jax.devices("gpu")
        if len(devices) == 0:
            return False

        # Mosaic GPU requires SM90+ (compute capability 9.0+)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major < 9:
                return False

        return True
    except Exception:
        return False


@functools.cache
def has_jax_tpu_backend() -> bool:
    """Check if JAX has TPU backend support."""
    if not has_jax_package():
        return False
    try:
        import jax  # type: ignore[import-not-found]

        # Check if TPU backend is available
        devices = jax.devices("tpu")
        return len(devices) > 0
    except Exception:
        return False


@functools.cache
def has_torch_tpu() -> bool:
    """Check if torch_tpu is installed and available."""
    # The TPU module is an autoloaded out-of-tree backend.
    tpu_module = getattr(torch, "tpu", None)
    return tpu_module is not None


@functools.cache
def has_cpu_pallas() -> bool:
    """Checks for a full Pallas-on-CPU environment."""
    return has_pallas_package()


@functools.cache
def has_cuda_pallas() -> bool:
    """Checks for a full Pallas-on-CUDA environment."""
    return has_pallas_package() and torch.cuda.is_available() and has_jax_cuda_backend()


@functools.cache
def has_tpu_pallas() -> bool:
    """Checks for a full Pallas-on-TPU environment."""
    return has_pallas_package() and has_jax_tpu_backend() and has_torch_tpu()


@functools.cache
def has_pallas() -> bool:
    """
    Check if Pallas backend is fully available for use.

    Requirements:
    - JAX package installed
    - Pallas (jax.experimental.pallas) available
    - A compatible backend (CUDA or TPU) is available in both PyTorch and JAX.
    """
    return has_cpu_pallas() or has_cuda_pallas() or has_tpu_pallas()
<|endoftext|># Owner(s): ["module: pytree"]

"""
Contains utility functions for working with nested python data structures.

A *pytree* is a Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `tree_map` to map a function over all Tensors inside some nested
collection of Tensors and `tree_leaves` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.
"""

import functools
import sys
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Any, overload, TypeAlias, TypeVar
from typing_extensions import deprecated, Self, TypeIs

import torch.utils._pytree as python_pytree
from torch.torch_version import TorchVersion as _TorchVersion
from torch.utils._pytree import (
    Context,
    DumpableContext,
    FlattenFn,
    FlattenWithKeysFn,
    FromDumpableContextFn,
    is_namedtuple,
    is_namedtuple_class,
    is_namedtuple_instance,
    is_structseq,
    is_structseq_class,
    is_structseq_instance,
    KeyPath,
    PyTree,
    ToDumpableContextFn,
    UnflattenFn,
)


# Do not try to import `optree` package if the static version check already fails.
if not python_pytree._cxx_pytree_dynamo_traceable:
    raise ImportError(
        f"{__name__} depends on `optree>={python_pytree._optree_minimum_version}`, "
        "which is an optional dependency of PyTorch. "
        "To use it, please upgrade your optree package via "
        "`python3 -m pip install --upgrade optree`"
    )


import optree
from optree import PyTreeSpec  # direct import for type annotations


__all__ = [
    "PyTree",
    "Context",
    "FlattenFn",
    "UnflattenFn",
    "DumpableContext",
    "ToDumpableContextFn",
    "FromDumpableContextFn",
    "PyTreeSpec",
    "TreeSpec",
    "LeafSpec",
    "keystr",
    "key_get",
    "register_pytree_node",
    "tree_is_leaf",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_unflatten",
    "tree_iter",
    "tree_leaves",
    "tree_leaves_with_path",
    "tree_structure",
    "tree_map",
    "tree_map_with_path",
    "tree_map_",
    "tree_map_only",
    "tree_map_only_",
    "tree_all",
    "tree_any",
    "tree_all_only",
    "tree_any_only",
    "treespec_dumps",
    "treespec_loads",
    "treespec_pprint",
    "is_namedtuple",
    "is_namedtuple_class",
    "is_namedtuple_instance",
    "is_structseq",
    "is_structseq_class",
    "is_structseq_instance",
]


# In-tree installation may have VCS-based versioning. Update the previous static version.
python_pytree._optree_version = _TorchVersion(optree.__version__)  # type: ignore[attr-defined]

__TORCH_DICT_SESSION = optree.dict_insertion_ordered(True, namespace="torch")
__TORCH_DICT_SESSION.__enter__()  # enable globally and permanently


T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
R = TypeVar("R")

TreeSpec: TypeAlias = PyTreeSpec
OpTreeUnflattenFn = Callable[[Context, Iterable[Any]], PyTree]

# Keep deprecated alias for backward compatibility
FlattenFunc = FlattenFn  # deprecated
UnflattenFunc = UnflattenFn  # deprecated
FlattenWithKeysFunc = FlattenWithKeysFn  # deprecated


def _reverse_args(func: UnflattenFn) -> OpTreeUnflattenFn:
    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return func(*reversed(args), **kwargs)

    return wrapped


def register_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFn,
    unflatten_fn: UnflattenFn,
    *,
    serialized_type_name: str | None = None,
    to_dumpable_context: ToDumpableContextFn | None = None,
    from_dumpable_context: FromDumpableContextFn | None = None,
    flatten_with_keys_fn: FlattenWithKeysFn | None = None,
) -> None:
    """Register a container-like type as pytree node.

    Args:
        cls (type): A Python type to treat as an internal pytree node.
        flatten_fn (callable): A function to be used during flattening, taking an instance of
            ``cls`` and returning a pair, with (1) an iterable for the children to be flattened
            recursively, and (2) some hashable auxiliary data to be stored in the treespec and to be
            passed to the ``unflatten_fn``.
        unflatten_fn (callable): A function taking two arguments: the auxiliary data that was
            returned by ``flatten_fn`` and stored in the treespec, and the unflattened children.
            The function should return an instance of ``cls``.
        serialized_type_name (str, optional): A keyword argument used to specify the fully
            qualified name used when serializing the tree spec.
        to_dumpable_context (callable, optional): An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable representation. This is
            used for json serialization, which is being used in :mod:`torch.export` right now.
        from_dumpable_context (callable, optional): An optional keyword argument to custom specify
            how to convert the custom json dumpable representation of the context back to the
            original context. This is used for json deserialization, which is being used in
            :mod:`torch.export` right now.

    Example::

        >>> # xdoctest: +SKIP
        >>> # Register a Python type with lambda functions
        >>> register_pytree_node(
        ...     set,
        ...     lambda s: (sorted(s), None, None),
        ...     lambda children, _: set(children),
        ... )
    """
    if flatten_with_keys_fn is not None:
        raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")

    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )

    python_pytree._private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )


@deprecated(
    "`torch.utils._cxx_pytree._register_pytree_node` is deprecated. "
    "Please use `torch.utils._cxx_pytree.register_pytree_node` instead.",
    category=FutureWarning,
)
def _register_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFn,
    unflatten_fn: UnflattenFn,
    *,
    serialized_type_name: str | None = None,
    to_dumpable_context: ToDumpableContextFn | None = None,
    from_dumpable_context: FromDumpableContextFn | None = None,
) -> None:
    """Register a container-like type as pytree node for the C++ pytree only.

    The ``namespace`` argument is used to avoid collisions that occur when different libraries
    register the same Python type with different behaviors. It is recommended to add a unique prefix
    to the namespace to avoid conflicts with other libraries. Namespaces can also be used to specify
    the same class in different namespaces for different use cases.

    .. warning::
        For safety reasons, a ``namespace`` must be specified while registering a custom type. It is
        used to isolate the behavior of flattening and unflattening a pytree node type. This is to
        prevent accidental collisions between different libraries that may register the same type.

    Args:
        cls (type): A Python type to treat as an internal pytree node.
        flatten_fn (callable): A function to be used during flattening, taking an instance of
            ``cls`` and returning a pair, with (1) an iterable for the children to be flattened
            recursively, and (2) some hashable auxiliary data to be stored in the treespec and to be
            passed to the ``unflatten_fn``.
        unflatten_fn (callable): A function taking two arguments: the auxiliary data that was
            returned by ``flatten_fn`` and stored in the treespec, and the unflattened children.
            The function should return an instance of ``cls``.
        serialized_type_name (str, optional): A keyword argument used to specify the fully
            qualified name used when serializing the tree spec.
        to_dumpable_context (callable, optional): An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable representation. This is
            used for json serialization, which is being used in :mod:`torch.export` right now.
        from_dumpable_context (callable, optional): An optional keyword argument to custom specify
            how to convert the custom json dumpable representation of the context back to the
            original context. This is used for json deserialization, which is being used in
            :mod:`torch.export` right now.
    """

    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )


def _private_register_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFn,
    unflatten_fn: UnflattenFn,
    *,
    serialized_type_name: str | None = None,
    to_dumpable_context: ToDumpableContextFn | None = None,
    from_dumpable_context: FromDumpableContextFn | None = None,
) -> None:
    """This is an internal function that is used to register a pytree node type
    for the C++ pytree only. End-users should use :func:`register_pytree_node`
    instead.
    """
    # TODO(XuehaiPan): remove this condition when we make Python pytree out-of-box support
    # PyStructSequence types
    if not optree.is_structseq_class(cls):
        optree.register_pytree_node(
            cls,
            flatten_fn,
            _reverse_args(unflatten_fn),
            namespace="torch",
        )


def _is_pytreespec_instance(
    obj: Any,
    /,
) -> TypeIs[TreeSpec | python_pytree.PyTreeSpec]:
    if isinstance(obj, (TreeSpec, python_pytree.PyTreeSpec)):
        return True
    if "torch._dynamo.polyfills.pytree" in sys.modules:
        # The PyTorch Dynamo pytree module is not always available, so we check if it is loaded.
        # If the PyTorch Dynamo pytree module is loaded, we can check if the treespec
        # is an instance of the PyTorch Dynamo TreeSpec class.
        import torch._dynamo.polyfills.pytree as dynamo_pytree

        return isinstance(obj, dynamo_pytree.PyTreeSpec)
    return False


def treespec_leaf() -> TreeSpec:
    """Make a treespec representing a leaf node."""
    return optree.treespec_leaf(none_is_leaf=True, namespace="torch")


def treespec_tuple(iterable: Iterable[TreeSpec] = (), /) -> TreeSpec:
    """Make a tuple treespec from an iterable of child treespecs."""
    return optree.treespec_tuple(iterable, none_is_leaf=True, namespace="torch")


def treespec_dict(
    mapping: Mapping[Any, TreeSpec] | Iterable[tuple[Any, TreeSpec]] = (),
    /,
    **kwargs: TreeSpec,
) -> TreeSpec:
    """Make a dict treespec from a dict of child treespecs."""
    return optree.treespec_dict(
        mapping,
        **kwargs,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_is_leaf(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    """Check if a pytree is a leaf.

    >>> tree_is_leaf(1)
    True
    >>> tree_is_leaf(None)
    True
    >>> tree_is_leaf([1, 2, 3])
    False
    >>> tree_is_leaf((1, 2, 3), is_leaf=lambda x: isinstance(x, tuple))
    True
    >>> tree_is_leaf({"a": 1, "b": 2, "c": 3})
    False
    >>> tree_is_leaf({"a": 1, "b": 2, "c": None})
    False

    Args:
        tree (pytree): A pytree to check if it is a leaf node.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A boolean indicating if the pytree is a leaf node.
    """
    return optree.tree_is_leaf(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_flatten(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> tuple[list[Any], TreeSpec]:
    """Flatten a pytree.

    See also :func:`tree_unflatten`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {"b": (2, [3, 4]), "a": 1, "c": None, "d": 5}
    >>> tree_flatten(tree)
    ([2, 3, 4, 1, None, 5], PyTreeSpec({'b': (*, [*, *]), 'a': *, 'c': *, 'd': *}, NoneIsLeaf, namespace='torch'))
    >>> tree_flatten(1)
    ([1], PyTreeSpec(*, NoneIsLeaf, namespace='torch'))
    >>> tree_flatten(None)
    ([None], PyTreeSpec(*, NoneIsLeaf, namespace='torch'))
    >>> from collections import OrderedDict
    >>> tree = OrderedDict([("b", (2, [3, 4])), ("a", 1), ("c", None), ("d", 5)])
    >>> tree_flatten(tree)
    ([2, 3, 4, 1, None, 5], PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': *, 'd': *}), NoneIsLeaf, namespace='torch'))

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A pair ``(leaves, treespec)`` where the first element is a list of leaf values and the
        second element is a treespec representing the structure of the pytree.
    """
    return optree.tree_flatten(  # type: ignore[return-value]
        tree,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree:
    """Reconstruct a pytree from the treespec and the leaves.

    The inverse of :func:`tree_flatten`.

    >>> tree = {"b": (2, [3, 4]), "a": 1, "c": None, "d": 5}
    >>> leaves, treespec = tree_flatten(tree)
    >>> tree == tree_unflatten(leaves, treespec)
    True

    Args:
        leaves (iterable): The list of leaves to use for reconstruction. The list must match the
            number of leaves of the treespec.
        treespec (TreeSpec): The treespec to reconstruct.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure described by
        ``treespec``.
    """
    if not _is_pytreespec_instance(treespec):
        if not _is_pytreespec_instance(leaves):
            raise TypeError(
                f"Expected `treespec` to be an instance of "
                f"PyTreeSpec but got item of type {type(treespec)}."
            )
        # Allow passing the PyTreeSpec instance as the first argument
        # pyrefly: ignore [bad-assignment]
        leaves, treespec = treespec, leaves
    return treespec.unflatten(leaves)


def tree_iter(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> Iterable[Any]:
    """Get an iterator over the leaves of a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {"b": (2, [3, 4]), "a": 1, "c": None, "d": 5}
    >>> list(tree_iter(tree))
    [2, 3, 4, 1, None, 5]
    >>> list(tree_iter(1))
    [1]
    >>> list(tree_iter(None))
    [None]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        An iterator over the leaf values.
    """
    return optree.tree_iter(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_leaves(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> list[Any]:
    """Get the leaves of a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {"b": (2, [3, 4]), "a": 1, "c": None, "d": 5}
    >>> tree_leaves(tree)
    [2, 3, 4, 1, None, 5]
    >>> tree_leaves(1)
    [1]
    >>> tree_leaves(None)
    [None]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A list of leaf values.
    """
    return optree.tree_leaves(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_structure(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> TreeSpec:
    """Get the treespec for a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {"b": (2, [3, 4]), "a": 1, "c": None, "d": 5}
    >>> tree_structure(tree)
    PyTreeSpec({'b': (*, [*, *]), 'a': *, 'c': *, 'd': *}, NoneIsLeaf, namespace='torch')
    >>> tree_structure(1)
    PyTreeSpec(*, NoneIsLeaf, namespace='torch')
    >>> tree_structure(None)
    PyTreeSpec(*, NoneIsLeaf, namespace='torch')

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A treespec object representing the structure of the pytree.
    """
    return optree.tree_structure(  # type: ignore[return-value]
        tree,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_map_`.

    >>> tree_map(lambda x: x + 1, {"x": 7, "y": (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_map(lambda x: x is None, {"x": 7, "y": (42, 64), "z": None})
    {'x': False, 'y': (False, False), 'z': True}

    If multiple inputs are given, the structure of the tree is taken from the first input;
    subsequent inputs need only have ``tree`` as a prefix:

    >>> tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and ``xs``
        is the tuple of values at corresponding nodes in ``rests``.
    """
    return optree.tree_map(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_map_(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    """Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`.

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x, *xs)`` (not the return value) where ``x`` is the value at the corresponding leaf
        in ``tree`` and ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    return optree.tree_map_(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )


Type2 = tuple[type[T], type[S]]
Type3 = tuple[type[T], type[S], type[U]]
TypeAny = type[Any] | tuple[type[Any], ...] | types.UnionType

Fn2 = Callable[[T | S], R]
Fn3 = Callable[[T | S | U], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]

MapOnlyFn = Callable[[T], Callable[[Any], Any]]


# These specializations help with type inference on the lambda passed to this
# function
@overload
def map_only(type_or_types_or_pred: type[T], /) -> MapOnlyFn[Fn[T, Any]]: ...


@overload
def map_only(type_or_types_or_pred: Type2[T, S], /) -> MapOnlyFn[Fn2[T, S, Any]]: ...


@overload
def map_only(
    type_or_types_or_pred: Type3[T, S, U], /
) -> MapOnlyFn[Fn3[T, S, U, Any]]: ...


# This specialization is needed for the implementations below that call
@overload
def map_only(type_or_types_or_pred: TypeAny, /) -> MapOnlyFn[FnAny[Any]]: ...


@overload
def map_only(
    type_or_types_or_pred: Callable[[Any], bool], /
) -> MapOnlyFn[FnAny[Any]]: ...


def map_only(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool], /
) -> MapOnlyFn[FnAny[Any]]:
    """
    Suppose you are writing a tree_map over tensors, leaving everything
    else unchanged.  Ordinarily you would have to write:

        def go(t):
            if isinstance(t, Tensor):
                return ...
            else:
                return t

    With this function, you only need to write:

        @map_only(Tensor)
        def go(t):
            return ...

    You can also directly use 'tree_map_only'
    """
    if isinstance(type_or_types_or_pred, (type, tuple, types.UnionType)):

        def pred(x: Any) -> bool:
            return isinstance(x, type_or_types_or_pred)  # type: ignore[arg-type]

    elif callable(type_or_types_or_pred):
        pred = type_or_types_or_pred  # type: ignore[assignment]
    else:
        raise TypeError("Argument must be a type, a tuple of types, or a callable.")

    def wrapper(func: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(func)
        def wrapped(x: T) -> Any:
            if pred(x):
                return func(x)
            return x

        return wrapped

    return wrapper


@overload
def tree_map_only(
    type_or_types_or_pred: type[T],
    /,
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: Type2[T, S],
    /,
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: Type3[T, S, U],
    /,
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: TypeAny,
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only(
    type_or_types_or_pred: Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


def tree_map_only(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    return tree_map(map_only(type_or_types_or_pred)(func), tree, is_leaf=is_leaf)


@overload
def tree_map_only_(
    type_or_types_or_pred: type[T],
    /,
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: Type2[T, S],
    /,
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: Type3[T, S, U],
    /,
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: TypeAny,
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


@overload
def tree_map_only_(
    type_or_types_or_pred: Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree: ...


def tree_map_only_(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    return tree_map_(map_only(type_or_types_or_pred)(func), tree, is_leaf=is_leaf)


def tree_all(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return all(map(pred, flat_args))


def tree_any(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return any(map(pred, flat_args))


@overload
def tree_all_only(
    type_or_types: type[T],
    /,
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_all_only(
    type_or_types: Type2[T, S],
    /,
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_all_only(
    type_or_types: Type3[T, S, U],
    /,
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


def tree_all_only(
    type_or_types: TypeAny,
    /,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return all(pred(x) for x in flat_args if isinstance(x, type_or_types))


@overload
def tree_any_only(
    type_or_types: type[T],
    /,
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_any_only(
    type_or_types: Type2[T, S],
    /,
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


@overload
def tree_any_only(
    type_or_types: Type3[T, S, U],
    /,
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool: ...


def tree_any_only(
    type_or_types: TypeAny,
    /,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> bool:
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return any(pred(x) for x in flat_args if isinstance(x, type_or_types))


def broadcast_prefix(
    prefix_tree: PyTree,
    full_tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> list[Any]:
    """Return a list of broadcasted leaves in ``prefix_tree`` to match the number of leaves in ``full_tree``.

    If a ``prefix_tree`` is a prefix of a ``full_tree``, this means the ``full_tree`` can be
    constructed by replacing the leaves of ``prefix_tree`` with appropriate **subtrees**.

    This function returns a list of leaves with the same size as ``full_tree``. The leaves are
    replicated from ``prefix_tree``. The number of replicas is determined by the corresponding
    subtree in ``full_tree``.

    >>> broadcast_prefix(1, [1, 2, 3])
    [1, 1, 1]
    >>> broadcast_prefix([1, 2, 3], [1, 2, 3])
    [1, 2, 3]
    >>> broadcast_prefix([1, 2, 3], [1, 2, 3, 4])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4; list: [1, 2, 3, 4].
    >>> broadcast_prefix([1, 2, 3], [1, 2, (3, 4)])
    [1, 2, 3, 3]
    >>> broadcast_prefix([1, 2, 3], [1, 2, {"a": 3, "b": 4, "c": (None, 5)}])
    [1, 2, 3, 3, 3, 3]

    Args:
        prefix_tree (pytree): A pytree with the same structure as a prefix of ``full_tree``.
        full_tree (pytree): A pytree with the same structure as a suffix of ``prefix_tree``.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A list of leaves in ``prefix_tree`` broadcasted to match the number of leaves in ``full_tree``.
    """
    result: list[Any] = []

    def add_leaves(x: Any, subtree: PyTree) -> None:
        subtreespec = tree_structure(subtree, is_leaf=is_leaf)
        result.extend([x] * subtreespec.num_leaves)

    tree_map_(
        add_leaves,
        prefix_tree,
        full_tree,
        is_leaf=is_leaf,
    )
    return result


# Broadcasts a pytree to the provided TreeSpec and returns the flattened
# values. If this is not possible, then this function returns None.
#
# For example, given pytree=0 and spec=TreeSpec(list, None, [LeafSpec(), LeafSpec()]),
# would return [0, 0]. This is useful for part of the vmap implementation:
# a user can pass in vmap(fn, in_dims)(*inputs). `in_dims` should be
# broadcastable to the tree structure of `inputs` and we use
# _broadcast_to_and_flatten to check this.
def _broadcast_to_and_flatten(
    tree: PyTree,
    treespec: TreeSpec,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> list[Any] | None:
    if not _is_pytreespec_instance(treespec):
        raise TypeError(
            f"Expected `treespec` to be an instance of "
            f"PyTreeSpec but got item of type {type(treespec)}."
        )
    full_tree = tree_unflatten([0] * treespec.num_leaves, treespec)
    try:
        return broadcast_prefix(tree, full_tree, is_leaf=is_leaf)
    except ValueError:
        return None


def treespec_dumps(treespec: TreeSpec, protocol: int | None = None) -> str:
    """Serialize a treespec to a JSON string."""
    if not _is_pytreespec_instance(treespec):
        raise TypeError(
            f"Expected `treespec` to be an instance of "
            f"PyTreeSpec but got item of type {type(treespec)}."
        )

    dummy_tree = tree_unflatten([0] * treespec.num_leaves, treespec)
    orig_treespec = python_pytree.tree_structure(dummy_tree)
    return python_pytree.treespec_dumps(orig_treespec, protocol=protocol)


@functools.lru_cache
def treespec_loads(serialized: str) -> TreeSpec:
    """Deserialize a treespec from a JSON string."""
    orig_treespec = python_pytree.treespec_loads(serialized)
    dummy_tree = python_pytree.tree_unflatten(
        [0] * orig_treespec.num_leaves,
        orig_treespec,
    )
    treespec = tree_structure(dummy_tree)
    return treespec


class _Asterisk(str):
    __slots__ = ()

    def __new__(cls) -> Self:
        return super().__new__(cls, "*")

    def __repr__(self) -> str:
        return "*"  # no quotes


_asterisk = _Asterisk()
del _Asterisk


def treespec_pprint(treespec: TreeSpec) -> str:
    dummy_tree = tree_unflatten([_asterisk] * treespec.num_leaves, treespec)
    return repr(dummy_tree)


class LeafSpecMeta(type(TreeSpec)):  # type: ignore[misc]
    def __instancecheck__(self, instance: object) -> bool:
        return _is_pytreespec_instance(instance) and instance.is_leaf()


@deprecated(
    "`isinstance(treespec, LeafSpec)` is deprecated, "
    "use `isinstance(treespec, TreeSpec)` and `treespec.is_leaf()` instead.",
    category=FutureWarning,
)
class LeafSpec(TreeSpec, metaclass=LeafSpecMeta):  # type: ignore[misc,final]
    def __new__(cls) -> Self:
        return treespec_leaf()  # type: ignore[return-value]


def tree_flatten_with_path(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> tuple[list[tuple[KeyPath, Any]], TreeSpec]:
    """Flattens a pytree like :func:`tree_flatten`, but also returns each leaf's key path.

    Args:
        tree: a pytree to flatten. If it contains a custom type, that type must be
            registered with an appropriate `tree_flatten_with_path_fn` when registered
            with :func:`register_pytree_node`.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    Returns:
        A tuple where the first element is a list of (key path, leaf) pairs, and the
        second element is a :class:`TreeSpec` representing the structure of the flattened
        tree.
    """
    raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")


def tree_leaves_with_path(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> list[tuple[KeyPath, Any]]:
    """Gets the leaves of a pytree like ``tree_leaves`` and returns each leaf's key path.

    Args:
        tree: a pytree. If it contains a custom type, that type must be
            registered with an appropriate `tree_flatten_with_path_fn` when registered
            with :func:`register_pytree_node`.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    Returns:
        A list of (key path, leaf) pairs.
    """
    raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")


def tree_map_with_path(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    """Like :func:`tree_map`, but the provided callable takes an additional key path argument.

    Args:
        func: A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees. The first positional argument
            to ``func`` is the key path of the leaf in question. The second
            positional argument is the value of the leaf.
        tree: A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests: A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(keypath, x, *xs)`` where ``keypath`` is the key path at the
        corresponding leaf in ``tree``, ``x`` is the value at that leaf, and
        ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")


def keystr(kp: KeyPath) -> str:
    """Given a key path, return a pretty-printed representation."""
    raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")


def key_get(obj: Any, kp: KeyPath) -> Any:
    """Given an object and a key path, return the value at the key path."""
    raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")


with python_pytree._NODE_REGISTRY_LOCK:
    # pyrefly: ignore [bad-assignment]
    python_pytree._cxx_pytree_imported = True
    args, kwargs = (), {}  # type: ignore[var-annotated]
    for args, kwargs in python_pytree._cxx_pytree_pending_imports:
        _private_register_pytree_node(*args, **kwargs)
    python_pytree._cxx_pytree_pending_imports.clear()
    del args, kwargs
<|endoftext|>import contextlib
import copy
import functools
import hashlib
import importlib
import inspect
import io
import keyword
import math
import os
import pickle
import tokenize
import unittest
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import Any, Generic, NoReturn, Optional, TYPE_CHECKING, TypeVar
from typing_extensions import deprecated

from torch._utils_internal import justknobs_check


# Types saved/loaded in configs
CONFIG_TYPES = (int, float, bool, type(None), str, list, set, tuple, dict)

# Immutable scalar types that don't need deepcopy when returned from configs.
# Everything else is defensively copied to prevent accidental mutation.
_IMMUTABLE_CONFIG_TYPES = (int, float, bool, type(None), str, tuple)


# Duplicated, because mypy needs these types statically
T = TypeVar("T", bound=int | float | bool | str | list | set | tuple | dict | None)


_UNSET_SENTINEL = object()


@dataclass(kw_only=True)
class _Config(Generic[T]):
    """Represents a config with richer behaviour than just a default value.
    ::
        i.e.
        foo = Config(justknob="//foo:bar", default=False)
        install_config_module(...)

    This configs must be installed with install_config_module to be used

    Precedence Order:
        alias: If set, the directly use the value of the alias.
        env_name_force: If set, this environment variable has precedence over
            everything after this.
            If multiple env variables are given, the precedence order is from
            left to right.
        user_override: If a user sets a value (i.e. foo.bar=True), that
            has precedence over everything after this.  User overrides are thread-local.
        env_name_default: If set, this environment variable will override everything
            after this.
            If multiple env variables are given, the precedence order is from
            left to right.
        justknob: If this pytorch installation supports justknobs, that will
            override defaults, but will not override the user_override precedence.
        default: This value is the lowest precedence, and will be used if nothing is
            set.

    Environment Variables:
        These are interpreted to be either "0" or "1" to represent true and false.

    Arguments:
        justknob: the name of the feature / JK. In OSS this is unused.
        default: is the value to default this knob to in OSS.
        alias: The alias config to read instead.
        env_name_force: The environment variable, or list of, to read that is a FORCE
            environment variable. I.e. it overrides everything except for alias.
        env_name_default: The environment variable, or list of, to read that changes the
            default behaviour. I.e. user overrides take preference.
    """

    default: T | object
    justknob: str | None = None
    env_name_default: list[str] | None = None
    env_name_force: list[str] | None = None
    value_type: type | None = None
    alias: str | None = None
    # Deprecation support
    deprecated: bool = False
    deprecation_message: str | None = None

    def __post_init__(self) -> None:
        self.env_name_default = _Config.string_or_list_of_string_to_list(
            self.env_name_default
        )
        self.env_name_force = _Config.string_or_list_of_string_to_list(
            self.env_name_force
        )

        if self.alias is not None:
            if (
                self.default is not _UNSET_SENTINEL
                or self.justknob is not None
                or self.env_name_default is not None
                or self.env_name_force is not None
            ):
                raise AssertionError(
                    "if alias is set, none of {default, justknob, \
                        env_name_default and env_name_force} can be set"
                )

    @staticmethod
    def string_or_list_of_string_to_list(
        val: str | list[str] | None,
    ) -> list[str] | None:
        if val is None:
            return None
        if isinstance(val, str):
            return [val]
        if not isinstance(val, list):
            raise AssertionError(f"val is not a list, got {type(val)}")
        return val


# In runtime, we unbox the Config[T] to a T, but typechecker cannot see this,
# so in order to allow for this dynamic behavior to work correctly with
# typechecking we are going to lie to the typechecker that Config[T] returns
# a T.
if TYPE_CHECKING:

    def Config(
        default: T | object = _UNSET_SENTINEL,
        justknob: str | None = None,
        env_name_default: str | list[str] | None = None,
        env_name_force: str | list[str] | None = None,
        value_type: type | None = None,
        alias: str | None = None,
        # Deprecation support
        deprecated: bool = False,
        deprecation_message: str | None = None,
    ) -> T: ...

else:

    def Config(
        default: T | object = _UNSET_SENTINEL,
        justknob: str | None = None,
        env_name_default: str | list[str] | None = None,
        env_name_force: str | list[str] | None = None,
        value_type: type | None = None,
        alias: str | None = None,
        # Deprecation support
        deprecated: bool = False,
        deprecation_message: str | None = None,
    ) -> _Config[T]:
        return _Config(
            default=default,
            justknob=justknob,
            env_name_default=env_name_default,
            env_name_force=env_name_force,
            value_type=value_type,
            alias=alias,
            # Deprecation support
            deprecated=deprecated,
            deprecation_message=deprecation_message,
        )


def _read_env_variable(name: str) -> bool | str | None:
    value = os.environ.get(name)
    if value == "1":
        return True
    if value == "0":
        return False
    return value


def install_config_module(module: ModuleType) -> None:
    """
    Converts a module-level config into a `ConfigModule()`.

    See _config_typing.pyi for instructions on how to get the converted module to typecheck.
    """

    class ConfigModuleInstance(ConfigModule):
        # __annotations__ is written to by Sphinx autodoc
        _bypass_keys = {
            "_hash_dirty_var",
            "_hash_cache_var",
            "_get_dict_dirty_keys_var",
            "_get_dict_cache_var",
            "__annotations__",
        }

    def visit(
        source: ModuleType | type,
        dest: ModuleType | SubConfigProxy,
        prefix: str,
    ) -> None:
        """Walk the module structure and move everything to module._config"""
        type_hints = inspect.get_annotations(source)
        for key, value in list(source.__dict__.items()):
            if (
                key.startswith("__")
                or isinstance(value, (ModuleType, FunctionType))
                or (
                    hasattr(value, "__module__")
                    and (
                        value.__module__ == "typing"
                        or value.__module__.startswith("collections.abc")
                    )
                )
                # Handle from torch.utils._config_module import Config
                or (isinstance(value, type) and issubclass(value, _Config))
            ):
                continue

            name = f"{prefix}{key}"
            annotated_type = type_hints.get(key, None)
            if isinstance(value, CONFIG_TYPES):
                config[name] = _ConfigEntry(
                    _Config(default=value, value_type=annotated_type), name
                )
                if dest is module:
                    delattr(module, key)
            elif isinstance(value, _Config):
                if annotated_type is not None and value.value_type is None:
                    value.value_type = annotated_type

                config[name] = _ConfigEntry(value, name)

                if dest is module:
                    delattr(module, key)
            elif isinstance(value, type):
                if value.__module__ != module.__name__:
                    raise AssertionError(
                        f"subconfig class {value} must be defined in module {module.__name__}"
                    )
                # a subconfig with `class Blah:` syntax
                proxy = SubConfigProxy(module, f"{name}.")
                visit(value, proxy, f"{name}.")
                if dest is module:
                    setattr(dest, key, proxy)
                else:
                    dest.__dict__[key] = proxy
            else:
                raise AssertionError(f"Unhandled config {key}={value} ({type(value)})")

    config: dict[str, _ConfigEntry] = {}

    compile_ignored_keys = get_assignments_with_compile_ignored_comments(module)

    visit(module, module, "")
    module._config = config  # type: ignore[attr-defined]
    module._compile_ignored_keys = compile_ignored_keys  # type: ignore[attr-defined]
    module.__class__ = ConfigModuleInstance
    module._hash_dirty_var = ContextVar(f"{module.__name__}._hash_dirty", default=True)  # type: ignore[attr-defined]  # pyrefly: ignore[missing-attribute]
    module._hash_cache_var = ContextVar(  # pyrefly: ignore[missing-attribute]
        f"{module.__name__}._hash_cache", default=None
    )  # type: ignore[attr-defined]
    module._get_dict_dirty_keys_var = ContextVar(  # pyrefly: ignore[missing-attribute]
        f"{module.__name__}._get_dict_dirty_keys", default=None
    )  # type: ignore[attr-defined]
    module._get_dict_cache_var = ContextVar(  # pyrefly: ignore[missing-attribute]
        f"{module.__name__}._get_dict_cache", default=None
    )  # type: ignore[attr-defined]


COMPILE_IGNORED_MARKER = "@compile_ignored"


# Gets all the keys (i.e. assignments) with a @compile_ignored comment
def get_assignments_with_compile_ignored_comments(module: ModuleType) -> set[str]:
    source_code = inspect.getsource(module)
    assignments = set()

    # Tokenize the source code to retrieve comments
    tokens = tokenize.tokenize(io.BytesIO(source_code.encode("utf-8")).readline)
    current_comment = "", -1
    prev_name = ""

    for token in tokens:
        if token.type == tokenize.COMMENT:
            prev_name = ""
            maybe_current = token.string.strip()
            if COMPILE_IGNORED_MARKER in maybe_current:
                if current_comment != ("", -1):
                    raise AssertionError(f"unconsumed {COMPILE_IGNORED_MARKER}")
                current_comment = maybe_current, token.start[0]
        elif token.type == tokenize.NAME:
            # Only accept the first name token, to handle if you have
            # something like foo: Bar = ...
            if not prev_name:
                prev_name = token.string
        elif token.type == tokenize.OP and token.string == "=":
            # Check if the current assignment follows a comment
            # with COMPILE_IGNORED_MARKER
            if (
                COMPILE_IGNORED_MARKER in current_comment[0]
                and current_comment[1] == token.start[0] - 1
            ):
                assignments.add(prev_name)
                current_comment = "", -1  # reset
            prev_name = ""
    if current_comment != ("", -1):
        raise AssertionError(f"unconsumed {COMPILE_IGNORED_MARKER}")
    return assignments


_GetDictCacheKey = tuple[tuple[str, ...], tuple[str, ...], bool]


@dataclass
class _ConfigEntry:
    # The default value specified in the configuration
    default: Any
    # The type of the configuration value
    value_type: type
    # The value specified by the user when they overrode the configuration
    # _UNSET_SENTINEL indicates the value is not set.
    user_override: ContextVar[object]
    # The justknob to check for this config
    justknob: str | None = None
    # environment variables are read at install time
    env_value_force: Any = _UNSET_SENTINEL
    env_value_default: Any = _UNSET_SENTINEL
    # Used to work around bad assumptions in unittest.mock.patch
    # The code to blame is
    # https://github.com/python/cpython/blob/94a7a4e22fb8f567090514785c69e65298acca42/Lib/unittest/mock.py#L1637
    # Essentially, mock.patch requires, that if __dict__ isn't accessible
    # (which it isn't), that after delattr is called on the object, the
    # object must throw when hasattr is called. Otherwise, it doesn't call
    # setattr again.
    # Technically we'll have an intermediate state of hiding the config while
    # mock.patch is unpatching itself, but it calls setattr after the delete
    # call so the final state is correct. It's just very unintuitive.
    # upstream bug - python/cpython#126886
    hide: bool = False
    alias: str | None = None
    # Deprecation support
    deprecated: bool = False
    deprecation_message: str | None = None
    _deprecation_warned: bool = False

    def __init__(self, config: _Config, name: str) -> None:
        self.default = config.default
        self.value_type = (
            config.value_type if config.value_type is not None else type(self.default)
        )
        self.justknob = config.justknob
        self.alias = config.alias
        # Deprecation fields
        self.deprecated = config.deprecated
        self.deprecation_message = config.deprecation_message
        self._deprecation_warned = False

        self.user_override = ContextVar(name, default=_UNSET_SENTINEL)
        if config.env_name_default is not None:
            for val in config.env_name_default:
                if (env_value := _read_env_variable(val)) is not None:
                    self.env_value_default = env_value
                    break
        if config.env_name_force is not None:
            for val in config.env_name_force:
                if (env_value := _read_env_variable(val)) is not None:
                    self.env_value_force = env_value
                    break

        # Ensure justknobs and envvars are allowlisted types
        if self.justknob is not None and self.default is not None:
            if not isinstance(self.default, bool):
                raise AssertionError(
                    f"justknobs only support booleans, {self.default} is not a boolean"
                )
        if self.value_type is not None and (
            config.env_name_default is not None or config.env_name_force is not None
        ):
            if self.value_type not in (
                bool,
                str,
                Optional[bool],  # noqa: UP045
                Optional[str],  # noqa: UP045
            ):
                raise AssertionError(
                    f"envvar configs only support (optional) booleans or strings, {self.value_type} is neither"
                )


class ConfigModule(ModuleType):
    # NOTE: This should be kept in sync with _config_typing.pyi.

    # The actual configuration settings.  E.g., torch._dynamo.config.debug
    # would live as "debug" in the key, and torch._inductor.config.triton.cudagraphs
    # maps as "triton.cudagraphs". See discussion on the class for meaning of various sub items
    _config: dict[str, _ConfigEntry]
    _bypass_keys: set[str]
    _compile_ignored_keys: set[str]
    _hash_dirty_var: ContextVar[bool]
    _hash_cache_var: ContextVar[bytes | None]
    # Per-thread cache state, backed by ContextVar so each thread/context gets
    # its own dirty set and cache (config values are per-thread via ContextVar).
    # None means fully dirty (initial state or >_GET_DICT_DIRTY_KEYS_CAP keys
    # changed); empty set means the last _get_dict result is up to date.
    _get_dict_dirty_keys_var: ContextVar[set[str] | None]
    _get_dict_cache_var: ContextVar[dict[_GetDictCacheKey, dict[str, Any]]]

    def __init__(self) -> None:
        raise NotImplementedError(
            f"use {__name__}.install_config_module(sys.modules[__name__])"
        )

    def _warn_if_deprecated(self, name: str, config: _ConfigEntry) -> None:
        """Issue deprecation warning for config if not already warned."""
        if config.deprecated and not config._deprecation_warned:
            import warnings

            msg = f"{self.__name__}.{name} is deprecated"
            if config.deprecation_message:
                msg += f" and {config.deprecation_message}"
            msg += ". It will be removed in a future version of PyTorch."
            warnings.warn(msg, FutureWarning, stacklevel=3)
            config._deprecation_warned = True

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._bypass_keys:
            super().__setattr__(name, value)
        elif name not in self._config:
            raise AttributeError(f"{self.__name__}.{name} does not exist")
        else:
            # Issue deprecation warning on write (once per config)
            config = self._config[name]
            self._warn_if_deprecated(name, config)

            if config.alias is not None:
                self._set_alias_val(config, value)
            else:
                config.user_override.set(value)
                self._hash_dirty_var.set(True)
                self._mark_get_dict_dirty(name)
                # Avoid a redundant instance-__dict__ write: hide defaults to False on
                # the class and is only ever set True by __delattr__ (the mock.patch
                # workaround), so only clear it when it is actually set.
                if config.hide:
                    config.hide = False

    def __getattr__(self, name: str) -> Any:
        try:
            config = self._config[name]

            if config.hide:
                raise AttributeError(f"{self.__name__}.{name} does not exist")

            # Issue deprecation warning on read (once per config)
            self._warn_if_deprecated(name, config)

            alias_val = self._get_alias_val(config)
            if alias_val is not _UNSET_SENTINEL:
                return alias_val

            if config.env_value_force is not _UNSET_SENTINEL:
                return config.env_value_force

            user_override = config.user_override.get()
            if user_override is not _UNSET_SENTINEL:
                return user_override

            if config.env_value_default is not _UNSET_SENTINEL:
                return config.env_value_default

            if config.justknob is not None:
                # JK only supports bools and ints
                return justknobs_check(name=config.justknob, default=config.default)

            # Reference types can still be modified, so copy them to
            # user_overrides to prevent accidental mutation of defaults.
            if not isinstance(config.default, _IMMUTABLE_CONFIG_TYPES):
                config.user_override.set(copy.deepcopy(config.default))
                return config.user_override.get()
            return config.default

        except KeyError as e:
            # make hasattr() work properly
            raise AttributeError(f"{self.__name__}.{name} does not exist") from e

    def __delattr__(self, name: str) -> None:
        self._hash_dirty_var.set(True)
        self._mark_get_dict_dirty(name)
        # must support delete because unittest.mock.patch deletes
        # then recreate things
        self._config[name].user_override.set(_UNSET_SENTINEL)
        self._config[name].hide = True

    def _get_alias_module_and_name(
        self, entry: _ConfigEntry
    ) -> tuple[ModuleType, str] | None:
        alias = entry.alias
        if alias is None:
            return None
        module_name, constant_name = alias.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise AttributeError(f"config alias {alias} does not exist") from e
        return module, constant_name

    def _get_alias_val(self, entry: _ConfigEntry) -> Any:
        data = self._get_alias_module_and_name(entry)
        if data is None:
            return _UNSET_SENTINEL
        module, constant_name = data
        constant_value = getattr(module, constant_name)
        return constant_value

    def _set_alias_val(self, entry: _ConfigEntry, val: Any) -> None:
        data = self._get_alias_module_and_name(entry)
        if data is None:
            raise AssertionError(
                "alias data should not be None when setting alias value"
            )
        module, constant_name = data
        setattr(module, constant_name, val)

    _GET_DICT_DIRTY_KEYS_CAP = 16

    def _mark_get_dict_dirty(self, name: str) -> None:
        dirty = self._get_dict_dirty_keys_var.get()
        if dirty is None:
            return
        dirty.add(name)
        if len(dirty) > self._GET_DICT_DIRTY_KEYS_CAP:
            self._get_dict_dirty_keys_var.set(None)

    def _is_default(self, name: str) -> bool:
        """
        Returns true if the config is at its default value.
        configs overridden by the env are not considered default.
        """
        config_val = self._config[name]
        # The config is not overridden by the user, and the env_value_default
        # is different from the default value (meaning user has set the env to
        # change the default value).
        not_set_env_default = (
            config_val.env_value_default is _UNSET_SENTINEL
            or config_val.env_value_default == config_val.default
        )
        not_set_env_force = (
            config_val.env_value_force is _UNSET_SENTINEL
            or config_val.env_value_force == config_val.default
        )

        unset = config_val.user_override.get() is _UNSET_SENTINEL
        # Handle reference types specially to avoid spammy warnings
        if not isinstance(config_val.default, _IMMUTABLE_CONFIG_TYPES):
            unset = unset or config_val.user_override.get() == config_val.default
        return unset and not_set_env_default and not_set_env_force

    def _get_dict(
        self,
        ignored_keys: list[str] | None = None,
        ignored_prefixes: list[str] | None = None,
        skip_default: bool = False,
        readonly_values: bool = False,
    ) -> dict[str, Any]:
        """Export a dictionary of current configuration keys and values.

        This function is designed to provide a single point which handles
        accessing config options and exporting them into a dictionary.
        This is used by a number of different user facing export methods
        which all have slightly different semantics re: how and what to
        skip.
        If a config is aliased, it skips this config.

        Arguments:
            ignored_keys are keys that should not be exported.
            ignored_prefixes are prefixes that if a key matches should
                not be exported
            skip_default does two things. One if a key has not been modified
                it skips it.
            readonly_values when True, enables caching of the result.
                The caller owns the returned dict but must not mutate
                its values. When False (default), no caching is used.
        """

        keys_to_update, config = None, None
        cache_key = (
            tuple(ignored_keys) if ignored_keys else (),
            tuple(ignored_prefixes) if ignored_prefixes else (),
            skip_default,
        )

        # Try to take a shortcut and only update dirty keys on top of a cached base.
        cache = self._get_dict_cache_var.get()
        if readonly_values:
            if cache is None:
                cache = {}
                self._get_dict_cache_var.set(cache)
            dirty_keys = self._get_dict_dirty_keys_var.get()
            cached = cache.get(cache_key)
            if cached is not None and dirty_keys is not None:
                # Shortcut: copy the cached base and update only dirty keys.
                # The cache entry itself is never mutated.
                keys_to_update = dirty_keys
                config = dict(cached)
            elif dirty_keys is None:
                # Fully dirty — clear entire cache and recompute
                cache = {}
                self._get_dict_cache_var.set(cache)
                self._get_dict_dirty_keys_var.set(set())

        # Recompute everything otherwise.
        if keys_to_update is None:
            keys_to_update = self._config.keys()
        if config is None:
            config = {}

        for key in keys_to_update:
            entry = self._config[key]
            if entry.alias is not None:
                config.pop(key, None)
                continue
            if ignored_keys and key in ignored_keys:
                config.pop(key, None)
                continue
            if ignored_prefixes:
                if any(key.startswith(prefix) for prefix in ignored_prefixes):
                    config.pop(key, None)
                    continue
            if skip_default and self._is_default(key):
                config.pop(key, None)
                continue

            # Read value directly, bypassing __getattr__ overhead
            # (deprecation warnings, alias resolution).
            user_override = entry.user_override.get()
            if entry.env_value_force is not _UNSET_SENTINEL:
                val = entry.env_value_force
            elif user_override is not _UNSET_SENTINEL:
                val = user_override
            elif entry.env_value_default is not _UNSET_SENTINEL:
                val = entry.env_value_default
            elif entry.justknob is not None:
                val = justknobs_check(name=entry.justknob, default=entry.default)
            else:
                val = entry.default
            if not isinstance(val, _IMMUTABLE_CONFIG_TYPES):
                val = copy.deepcopy(val)
            config[key] = val

        if readonly_values and cache_key not in cache:
            cache[cache_key] = dict(config)
        return config

    def get_type(self, config_name: str) -> type:
        return self._config[config_name].value_type

    def save_config(self) -> bytes:
        """Convert config to a pickled blob"""
        ignored_keys = getattr(self, "_save_config_ignore", [])
        return pickle.dumps(
            self._get_dict(ignored_keys=ignored_keys, readonly_values=True),
            protocol=2,
        )

    def save_config_portable(
        self, *, ignore_private_configs: bool = True, readonly_values: bool = False
    ) -> dict[str, Any]:
        """Convert config to portable format"""
        prefixes = []
        if ignore_private_configs:
            prefixes.append("_")
        prefixes.extend(getattr(self, "_cache_config_ignore_prefix", []))
        config = self._get_dict(
            ignored_prefixes=prefixes, readonly_values=readonly_values
        )
        factory_keys = getattr(self, "_cache_config_factory_keys", [])
        if factory_keys:
            for key in factory_keys:
                if key in config and config[key] is not None:
                    instance = config[key]()
                    if hasattr(instance, "uuid"):
                        config[key] = instance.uuid()
                    else:
                        raise RuntimeError(
                            f"Config '{key}' is set to {config[key]} which does not "
                            f"implement uuid(). Implement uuid() for cache key "
                            f"participation."
                        )
        return config

    def codegen_config(self) -> str:
        """Convert config to Python statements that replicate current config.
        This does NOT include config settings that are at default values.
        """

        # additional imports required
        imports = set()

        def get_module_name(func: Callable, add_dot: bool) -> str:
            module_name = func.__module__
            if module_name == "builtins":
                module_name = ""
            if add_dot and module_name != "":
                module_name += "."
            return module_name

        def add_import(func: Callable) -> None:
            module_name = get_module_name(func, False)
            if module_name:
                imports.add(module_name)

        def list_of_callables_to_string(v: list | set) -> list[str]:
            return [f"{get_module_name(item, True)}{item.__name__}" for item in v]

        def importable_callable(v: Any) -> bool:
            # functools.partial has no attributes below but is a callable
            return callable(v) and hasattr(v, "__module__") and hasattr(v, "__name__")

        def importable_ref(func: Any) -> tuple[str, str] | None:
            try:
                module_name = getattr(func, "__module__", None)
                qualname = getattr(func, "__qualname__", None)
                if (
                    not callable(func)
                    or not isinstance(module_name, str)
                    or not isinstance(qualname, str)
                    or not module_name
                    or not qualname
                    or module_name == "__main__"
                    or "<" in qualname
                ):
                    return None
                names = module_name.split(".") + qualname.split(".")
                if any(
                    not name.isidentifier() or keyword.iskeyword(name) for name in names
                ):
                    return None
                resolved = importlib.import_module(module_name)
                for name in qualname.split("."):
                    resolved = getattr(resolved, name)
                if resolved is not func:
                    return None
                import_name = "" if module_name == "builtins" else module_name
                prefix = f"{import_name}." if import_name else ""
                return f"{prefix}{qualname}", import_name
            except Exception:
                return None

        def serialize_partial_arg(val: Any) -> str:
            if type(val) in (type(None), bool, int, str, bytes) or (
                type(val) is float and math.isfinite(val)
            ):
                return repr(val)
            if type(val) not in (list, tuple, dict):
                raise ValueError(
                    f"unsupported functools.partial argument type {type(val).__name__}"
                )

            if type(val) in (list, tuple):
                for item in val:
                    serialize_partial_arg(item)
            else:
                for key, item in val.items():
                    serialize_partial_arg(key)
                    serialize_partial_arg(item)
            return repr(val)

        def get_partial_line(mod, k, v) -> str:  # type: ignore[no-untyped-def]
            if type(v) is not functools.partial:
                return f"# {mod}.{k} omitted: unsupported partial subclass"
            func_info = importable_ref(v.func)
            if func_info is None:
                return f"# {mod}.{k} omitted: partial callable cannot be re-imported"
            func_ref, import_name = func_info
            try:
                parts = [func_ref]
                parts.extend(serialize_partial_arg(arg) for arg in v.args)
                if v.keywords:
                    parts.append(f"**{serialize_partial_arg(v.keywords)}")
                expression = f"functools.partial({', '.join(parts)})"
                compile(expression, "<config>", "eval")
            except Exception:
                return f"# {mod}.{k} omitted: partial arguments cannot be serialized"
            if import_name:
                imports.add(import_name)
            imports.add("functools")
            return f"{mod}.{k} = {expression}"

        def get_config_line(mod, k, v) -> str:  # type: ignore[no-untyped-def]
            """
            Return a string version of the config line.
            Handle v when v is a callable, or a list/dict of callables. Add import statements for callables if necessary.
            We assume that the value of a single config won't be a mix of callables and non-callables.

            Example output:
                import logging
                import _warnings
                torch._dynamo.config.reorderable_logging_functions = { _warnings.warn, logging.warn, print }
            """
            if isinstance(v, functools.partial):
                return get_partial_line(mod, k, v)
            elif importable_callable(v):
                add_import(v)
                return f"{mod}.{k} = {get_module_name(v, True)}{v.__name__}"
            elif isinstance(v, (list, set)) and all(
                importable_callable(item) for item in v
            ):
                for item in v:
                    add_import(item)
                v_list = list_of_callables_to_string(v)
                if isinstance(v, list):
                    return f"{mod}.{k} = {v_list}"
                else:
                    return f"{mod}.{k} = {{ {', '.join(v_list)} }}"
            else:
                return f"{mod}.{k} = {v!r}"

        lines = []
        mod = self.__name__
        for k, v in self._get_dict(
            ignored_keys=getattr(self, "_save_config_ignore", []),
            skip_default=True,
            readonly_values=True,
        ).items():
            lines.append(get_config_line(mod, k, v))
        for import_name in imports:
            lines.insert(0, f"import {import_name}")
        return "\n".join(lines)

    def get_hash(self) -> bytes:
        """Hashes the configs that are not compile_ignored"""
        if self._hash_dirty_var.get() or self._hash_cache_var.get() is None:
            dict_to_hash = self._get_dict(
                ignored_keys=list(self._compile_ignored_keys), readonly_values=True
            )
            string_to_hash = repr(sorted(dict_to_hash.items()))
            self._hash_cache_var.set(
                hashlib.md5(
                    string_to_hash.encode("utf-8"), usedforsecurity=False
                ).digest()
            )
            self._hash_dirty_var.set(False)
        result = self._hash_cache_var.get()
        if result is None:
            raise AssertionError(
                "_hash_cache_var should not be None after recomputation"
            )
        return result

    @deprecated(
        "`config.to_dict()` has been deprecated. It no longer changes the underlying config."
        " use `config.get_config_copy()` instead if you just want a copy of the config, or "
        "config.load_config if you need mutable access",
        category=FutureWarning,
    )
    def to_dict(self) -> dict[str, Any]:
        return self.get_config_copy()

    @deprecated(
        "`config.shallow_copy_dict()` has been deprecated. It no longer changes the underlying config."
        " use `config.get_config_copy()` instead if you just want a copy of the config, or "
        "config.load_config if you need mutable access",
        category=FutureWarning,
    )
    def shallow_copy_dict(self) -> dict[str, Any]:
        return self.get_config_copy()

    def load_config(self, maybe_pickled_config: bytes | dict[str, Any]) -> None:
        """Restore from a prior call to save_config() or shallow_copy_dict()"""
        if not isinstance(maybe_pickled_config, dict):
            config = pickle.loads(maybe_pickled_config)
        else:
            config = maybe_pickled_config
        for k, v in config.items():
            if k in self._config:
                setattr(self, k, v)
            else:
                from torch._dynamo.utils import warn_once

                warn_once(f"key {k} with value {v} is not understood by this config")

    def get_config_copy(self) -> dict[str, Any]:
        return self._get_dict()

    def get_serializable_config_copy(self) -> dict[str, Any]:
        return self._get_dict(ignored_keys=getattr(self, "_save_config_ignore", []))

    def patch(
        self,
        arg1: str | dict[str, Any] | None = None,
        arg2: Any = None,
        **kwargs: dict[str, Any],
    ) -> "ContextDecorator":
        """
        Decorator and/or context manager to make temporary changes to a config.  Note that patched settings are thread-local.

        As a decorator:

            @config.patch("name", val)
            @config.patch(name1=val1, name2=val2)
            @config.patch({"name1": val1, "name2", val2})
            def foo(...):
                ...

        As a context manager:

            with config.patch("name", val):
                ...
        """
        changes: dict[str, Any]
        if arg1 is not None:
            if arg2 is not None:
                if not isinstance(arg1, str):
                    raise AssertionError(
                        "first argument must be a string when passing 2 positional args to patch"
                    )
                # patch("key", True) syntax
                changes = {arg1: arg2}
            else:
                if not isinstance(arg1, dict):
                    raise AssertionError(
                        "first argument must be a dict when passing a single positional arg to patch"
                    )
                # patch({"key": True}) syntax
                changes = arg1
            if kwargs:
                raise AssertionError(
                    "cannot pass both positional and keyword arguments to patch"
                )
        else:
            # patch(key=True) syntax
            changes = kwargs
            if arg2 is not None:
                raise AssertionError(
                    "second positional argument is only valid when first argument is a key string"
                )
        if not isinstance(changes, dict):
            raise AssertionError(f"expected `dict` got {type(changes)}")
        config = self

        class ConfigPatch(ContextDecorator):
            def __init__(self) -> None:
                self.changes = changes
                self._prior: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
                    f"{config.__name__}.ConfigPatch[{id(self)}]",
                    default=(),
                )

            def __enter__(self) -> None:
                prior: dict[str, Any] = {}
                for key in self.changes:
                    # KeyError on invalid entry
                    prior[key] = config.__getattr__(key)
                prior_stack = self._prior.get()
                self._prior.set((*prior_stack, prior))
                try:
                    for k, v in self.changes.items():
                        config.__setattr__(k, v)
                except Exception:
                    self._prior.set(prior_stack)
                    raise

            def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-untyped-def]
                prior_stack = self._prior.get()
                if not prior_stack:
                    raise AssertionError(
                        "prior should not be empty when exiting ConfigPatch"
                    )
                prior = prior_stack[-1]
                self._prior.set(prior_stack[:-1])
                for k, v in prior.items():
                    config.__setattr__(k, v)

        return ConfigPatch()

    def _make_closure_patcher(self, **changes: dict[str, Any]) -> Any:
        """
        A lower-overhead version of patch() for things on the critical path.

        Usage:

            # do this off the critical path
            change_fn = config.make_closure_patcher(foo=True)

            ...

            revert = change_fn()
            try:
              ...
            finally:
                revert()

        """
        config = self._config

        def change() -> Callable[[], None]:
            prior = {k: config[k].user_override.get() for k in changes}
            for k, v in changes.items():
                config[k].user_override.set(v)
                self._hash_dirty_var.set(True)
                self._mark_get_dict_dirty(k)

            def revert() -> None:
                for k, v in prior.items():
                    config[k].user_override.set(v)
                    self._hash_dirty_var.set(True)
                    self._mark_get_dict_dirty(k)

            return revert

        return change


class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """

    def __enter__(self) -> None:
        raise NotImplementedError("NYI")

    def __exit__(self, exc_type, exc_val, exc_tb) -> NoReturn:  # type: ignore[no-untyped-def]
        raise NotImplementedError("NYI")

    def __call__(self, func: Callable[[Any], Any]) -> Any:
        if isinstance(func, type) and issubclass(func, unittest.TestCase):

            class _TestCase(func):  # type: ignore[valid-type, misc]
                @classmethod
                def setUpClass(cls) -> None:
                    self.__enter__()
                    try:
                        super().setUpClass()
                    except Exception:
                        self.__exit__(None, None, None)
                        raise

                @classmethod
                def tearDownClass(cls) -> None:
                    try:
                        super().tearDownClass()
                    finally:
                        self.__exit__(None, None, None)

            _TestCase.__name__ = func.__name__
            _TestCase.__qualname__ = func.__qualname__
            _TestCase.__module__ = func.__module__

            return _TestCase

        return super().__call__(func)


class SubConfigProxy:
    """
    Shim to redirect to main config.
    `config.triton.cudagraphs` maps to _config["triton.cudagraphs"]
    """

    def __init__(self, config: object, prefix: str) -> None:
        # `super().__setattr__` to bypass custom `__setattr__`
        super().__setattr__("_config", config)
        super().__setattr__("_prefix", prefix)

    def __setattr__(self, name: str, value: object) -> None:
        return self._config.__setattr__(self._prefix + name, value)

    def __getattr__(self, name: str) -> Any:
        return self._config.__getattr__(self._prefix + name)

    def __delattr__(self, name: str) -> None:
        return self._config.__delattr__(self._prefix + name)


def get_tristate_env(name: str, default: Any = None) -> bool | None:
    value = os.environ.get(name)
    if value == "1":
        return True
    if value == "0":
        return False
    return default


def inherit_fields_from(parent_cls):
    def wrapper(child_cls):
        for k, v in parent_cls.__dict__.items():
            # copy fields that are not private and not overridden
            if not k.startswith("_") and k not in child_cls.__dict__:
                setattr(child_cls, k, v)
        return child_cls

    return wrapper
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import inspect
import os.path
import tempfile
import traceback
from types import TracebackType


# This file contains utilities for ensuring dynamically compile()'d
# code fragments display their line numbers in backtraces.
#
# The constraints:
#
# - We don't have control over the user exception printer (in particular,
#   we cannot assume the linecache trick will work, c.f.
#   https://stackoverflow.com/q/50515651/23845 )
#
# - We don't want to create temporary files every time we compile()
#   some code; file creation should happen lazily only at exception
#   time.  Arguably, you *should* be willing to write out your
#   generated Python code to file system, but in some situations
#   (esp. library code) it would violate user expectation to write
#   to the file system, so we try to avoid it.  In particular, we'd
#   like to keep the files around, so users can open up the files
#   mentioned in the trace; if the file is invisible, we want to
#   avoid clogging up the filesystem.
#
#   If this is not a constraint for you, there is a substantially simpler
#   way to implement the functionality in this PR: instead of using
#   eval/exec directly, just always write a Python file to filesystem
#   and compile that.
#
# - You have control over a context where the compiled code will get
#   executed, so that we can interpose while the stack is unwinding
#   (otherwise, we have no way to interpose on the exception printing
#   process.)
#
# There are two things you have to do to make use of the utilities here:
#
# - When you compile your source code, you must save its string source
#   in its f_globals under the magic name "__compile_source__"
#
# - Before running the compiled code, enter the
#   report_compile_source_on_error() context manager.


@contextlib.contextmanager
def report_compile_source_on_error():
    try:
        yield
    except Exception as exc:
        tb = exc.__traceback__

        # Walk the traceback, looking for frames that have
        # source attached
        stack = []
        while tb is not None:
            filename = tb.tb_frame.f_code.co_filename
            source = tb.tb_frame.f_globals.get("__compile_source__")

            if filename == "<string>" and source is not None:
                # What black magic are we doing here?  Intuitively, what
                # we would like to do is overwrite the co_filename on any
                # frames that were generated from exec/eval so that they
                # point to a temporary file that has the actual line
                # information, so Python's default error printer can print
                # useful line information on it.
                #
                # Writing out the temporary file is easy.  But overwriting
                # co_filename is not!  You can't modify the code object
                # associated with a frame.  You can, however, reconstruct
                # a traceback with entirely new frames from scratch, so that's
                # what we do.  But there's another problem, which is how to
                # make the frame?
                #
                # The black magic is we make a frankenstein frame and code
                # object which resembles the original frame/code enough so
                # that it will print properly under traceback and the default
                # error printer, but IT IS NOT THE ORIGINAL FRAME (you
                # couldn't, e.g., execute its code with different variables
                # and expect it to work.)

                # Don't delete the temporary file so the user can inspect it
                # TODO: This creates a temporary file for every frame, but we
                # technically only need one per distinct __compile_source__
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".py"
                ) as f:
                    f.write(source)
                # Create a frame.  Python doesn't let you construct
                # FrameType directly, so just make one with compile
                frame = tb.tb_frame
                code = compile("__inspect_currentframe()", f.name, "eval")
                code = code.replace(co_name=frame.f_code.co_name)
                # Python 3.11 only
                if hasattr(frame.f_code, "co_linetable"):
                    # We can't copy ALL of the metadata over, because you
                    # can cause Python to segfault this way.  What exactly
                    # do we need?  We need enough information for
                    # traceback to be able to print the exception
                    # correctly.  Code reading Lib/traceback.py reveals
                    # that traceback calls code.co_positions() in order to
                    # get the augmented line/col numbers.  Objects/codeobject.c,
                    # specifically _PyCode_InitAddressRange, reveals that
                    # this iterator is initialized from co_linetable and
                    # co_firstfileno.  So copy these we must!
                    code = code.replace(  # type: ignore[call-arg]
                        co_linetable=frame.f_code.co_linetable,  # type: ignore[attr-defined]
                        co_firstlineno=frame.f_code.co_firstlineno,  # type: ignore[attr-defined]
                    )
                fake_frame = eval(
                    code,
                    frame.f_globals,
                    {**frame.f_locals, "__inspect_currentframe": inspect.currentframe},
                )
                fake_tb = TracebackType(None, fake_frame, tb.tb_lasti, tb.tb_lineno)
                stack.append(fake_tb)
            else:
                stack.append(tb)

            tb = tb.tb_next

        # Reconstruct the linked list
        tb_next = None
        for tb in reversed(stack):
            tb.tb_next = tb_next
            tb_next = tb

        raise exc.with_traceback(tb_next)  # noqa: B904


def shorten_filename(fn, *, base=None):
    """Shorten a source filepath, with the assumption that torch/ subdirectories don't need to be shown to user."""
    if base is None:
        base = os.path.dirname(os.path.dirname(__file__))
    # Truncate torch/foo.py to foo.py
    try:
        prefix = os.path.commonpath([fn, base])
    except ValueError:
        return fn
    else:
        return fn[len(prefix) + 1 :]


def format_frame(frame, *, base=None, line=False) -> str:
    """
    Format a FrameSummary in a short way, without printing full absolute path or code.

    The idea is the result fits on a single line.
    """
    extra_line = ""
    if line:
        extra_line = f"{frame.line}  # "
    return f"{extra_line}{shorten_filename(frame.filename, base=base)}:{frame.lineno} in {frame.name}"


def format_traceback_short(tb):
    """Format a TracebackType in a short way, printing only the inner-most frame."""
    return format_frame(traceback.extract_tb(tb)[-1])


class CapturedTraceback:
    __slots__ = ["tb", "skip"]

    def __init__(self, tb, skip=0) -> None:
        self.tb = tb
        self.skip = skip

    def cleanup(self) -> None:
        self.tb = None

    def summary(self):
        import torch._C._profiler

        if self.tb is None:
            # TODO: Maybe indicate that the traceback was elided?
            return traceback.StackSummary()

        return _extract_symbolized_tb(
            torch._C._profiler.symbolize_tracebacks([self.tb])[0], self.skip
        )

    def __getstate__(self):
        return (
            None,
            {
                "tb": None,  # TB is not pickleable
                "skip": self.skip,
            },
        )

    @staticmethod
    def extract(*, script=False, cpp=False, skip=0):
        """
        Like traceback.extract_stack(), but faster (approximately 20x faster); it
        is fast enough that you can unconditionally log stacks this way as part of
        normal execution.  It returns a torch._C._profiler.CapturedTraceback
        object that must be formatted specially with format_captured_tb.

        By default, this only reports Python backtraces (like extract_stack).  You
        can set the script/cpp kwargs to also turn on TorchScript/C++ trace
        reporting.
        """
        import torch._C._profiler

        if script or cpp:
            if skip != 0:
                raise AssertionError("skip with script/cpp NYI")

        return CapturedTraceback(
            torch._C._profiler.gather_traceback(python=True, script=script, cpp=cpp),
            # Elide extract() frame if we don't have script/cpp frames.  If
            # we do have those frames, it doesn't work so force zero.
            0 if script or cpp else skip + 1,
        )

    def format(self):
