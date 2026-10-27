            "type": "ObjectSpec",
            "fields": {
                name: spec.to_jsonable() if hasattr(spec, "to_jsonable") else spec
                for name, spec in self._fields.items()
            },
        }


class DictSpec:
    """Spec for a Python ``dict``-typed value.

    Constructor::

        DictSpec({key: IntermediateSpec, ...})

    Keys may be ``str`` or ``int``.

    Example::

        DictSpec({"x": TensorSpec([ShapeVar("h"), STATIC])})
        DictSpec({"config": DictSpec({"batch": IntVar()})})
        DictSpec({0: TensorSpec([ShapeVar("h"), STATIC])})
    """

    def __init__(
        self, entries: dict[str | int, IntermediateSpec] | None = None
    ) -> None:
        self._entries: dict[str | int, IntermediateSpec] = (
            dict(entries) if entries else {}
        )
        for k in self._entries:
            if not isinstance(k, (str, int)):
                raise TypeError(
                    f"DictSpec entries must have str or int keys, got {type(k).__name__}: {k!r}"
                )

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[str | int]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self) -> Any:
        return self._entries.items()

    def __repr__(self) -> str:
        lines = ["dict_spec:"]
        for key, spec in self._entries.items():
            spec_repr = repr(spec)
            if "\n" in spec_repr:
                lines.append(f"{_INDENT}[{key!r}]:")
                for line in spec_repr.splitlines():
                    lines.append(_INDENT * 2 + line)
            else:
                lines.append(f"{_INDENT}[{key!r}]: {spec_repr}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "DictSpec",
            "entries": {
                str(key): spec.to_jsonable() if hasattr(spec, "to_jsonable") else spec
                for key, spec in self._entries.items()
            },
        }


class SeqSpec:
    """Spec for a Python ``list``- or ``tuple``-typed value.

    Per-position. The spec list may be shorter than the runtime sequence;
    any positions beyond ``len(self)`` are treated as fully static (i.e.
    equivalent to an unspecified slot).

    Constructor::

        SeqSpec([IntermediateSpec, ...])

    Example::

        SeqSpec([TensorSpec([ShapeVar("h"), 10]), 1])
        SeqSpec((TensorSpec([ShapeVar("a")]), TensorSpec([ShapeVar("b")])))
    """

    def __init__(self, entries: Sequence[IntermediateSpec]) -> None:
        self._entries: list[IntermediateSpec] = list(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        lines = ["seq_spec:"]
        for i, spec in enumerate(self._entries):
            spec_repr = repr(spec)
            if "\n" in spec_repr:
                lines.append(f"{_INDENT}[{i}]:")
                for line in spec_repr.splitlines():
                    lines.append(_INDENT * 2 + line)
            else:
                lines.append(f"{_INDENT}[{i}]: {spec_repr}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "SeqSpec",
            "entries": [
                spec.to_jsonable() if hasattr(spec, "to_jsonable") else spec
                for spec in self._entries
            ],
        }


# Per-int-leaf spec type: union of valid values that can appear at a slot
# where the runtime value is an int (tensor dim, scalar arg).
#
# Per-entry semantics:
# - IntVar (or its ShapeVar subclass): unbacked dynamic dimension.
# - SymInt (derived from spec IntVars via arithmetic, e.g. A * 2 + 1):
#   the dim must equal this expression at runtime.
# - int: static dimension pinned to that concrete value.
# - None: unspecified; treated as static. Value is inferred from the
#   example input if the consuming API supports it (e.g. torch.compile),
#   otherwise an error will be thrown.
LeafIntSpec: TypeAlias = IntVar | SymInt | int | None
# Leaf specs (individual argument specifications) — ints plus TensorSpec.
LeafSpec: TypeAlias = LeafIntSpec | TensorSpec
# Includes containers (``ObjectSpec`` / ``DictSpec`` / ``SeqSpec``)
# for nested specs reachable via dynamo source-chain walks.
IntermediateSpec: TypeAlias = LeafSpec | ObjectSpec | DictSpec | SeqSpec

# Value type for the dict passed to ``ParamsSpec``/``ShapesSpec``:
#   - named entries hold a single spec,
#   - ``"*args"`` holds a list of specs,
#   - ``"**kwargs"`` holds a dict of specs.
ParamsSpecValue: TypeAlias = (
    IntermediateSpec | list[IntermediateSpec] | dict[str, IntermediateSpec]
)


class ParamsSpec:
    """Specification for the arguments of a compiled function.

    Describes the dynamic shape behavior for named arguments, ``*args``,
    and ``**kwargs`` of a ``torch.compile``-wrapped function. Takes a
    single dict keyed by parameter name, with two reserved sentinel keys
    for the variadic slots::

        def func(x, n, *args, **kwargs): ...


        ParamsSpec(
            {
                "x": TensorSpec([ShapeVar("batch"), None]),
                "n": IntVar("seq"),
                "*args": [TensorSpec([ShapeVar("a")]), None],
                "**kwargs": {
                    "foo": TensorSpec([ShapeVar("b"), None]),
                    "bar": TensorSpec([ShapeVar("c"), None]),
                },
            }
        )

    Anything not expressed in ``ParamsSpec`` is STATIC.**
    """

    _VARARGS_KEY = "*args"
    _VARKW_KEY = "**kwargs"

    def __init__(
        self,
        params: dict[str, ParamsSpecValue] | None = None,
    ) -> None:
        self._named_args: dict[str, IntermediateSpec] = {}
        self._varargs: list[IntermediateSpec] | None = None
        self._varkw: dict[str, IntermediateSpec] | None = None
        if params is None:
            return
        for key, value in params.items():
            if key == self._VARARGS_KEY:
                if not isinstance(value, list):
                    raise ValueError(
                        f"ParamsSpec {self._VARARGS_KEY!r} value must be a list "
                        f"of leaf specs, got {type(value).__name__}"
                    )
                for i, v in enumerate(value):
                    _validate_spec_sym(v, where=f"ParamsSpec '{key}'[{i}]")
                self._varargs = list(value)
            elif key == self._VARKW_KEY:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"ParamsSpec {self._VARKW_KEY!r} value must be a dict "
                        f"of leaf specs, got {type(value).__name__}"
                    )
                for k, v in value.items():
                    _validate_spec_sym(v, where=f"ParamsSpec '{key}'[{k!r}]")
                self._varkw = dict(value)
            elif key.startswith("*"):
                raise ValueError(
                    f"Unknown sentinel key {key!r} in ParamsSpec; only "
                    f"{self._VARARGS_KEY!r} and {self._VARKW_KEY!r} are reserved"
                )
            else:
                _validate_spec_sym(value, where=f"ParamsSpec[{key!r}]")
                self._named_args[key] = cast(IntermediateSpec, value)

    def __repr__(self) -> str:
        def _indent_lines(text: str, prefix: str = _INDENT) -> str:
            return "\n".join(prefix + line for line in text.splitlines())

        def _entry_repr(key: str, value: Any) -> str:
            v_repr = repr(value)
            if "\n" in v_repr:
                return f"{key}:\n{_indent_lines(v_repr)}"
            return f"{key}: {v_repr}"

        lines = [_entry_repr(k, v) for k, v in self._named_args.items()]
        if self._varargs is not None:
            inner = "\n".join(
                _entry_repr(str(i), v) for i, v in enumerate(self._varargs)
            )
            lines.append(f"{self._VARARGS_KEY}:\n{_indent_lines(inner)}")
        if self._varkw is not None:
            inner = "\n".join(_entry_repr(k, v) for k, v in self._varkw.items())
            lines.append(f"{self._VARKW_KEY}:\n{_indent_lines(inner)}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            name: value.to_jsonable() if hasattr(value, "to_jsonable") else value
            for name, value in self._named_args.items()
        }
        if self._varargs is not None:
            params[self._VARARGS_KEY] = [
                v.to_jsonable() if hasattr(v, "to_jsonable") else v
                for v in self._varargs
            ]
        if self._varkw is not None:
            params[self._VARKW_KEY] = {
                name: value.to_jsonable() if hasattr(value, "to_jsonable") else value
                for name, value in self._varkw.items()
            }
        return {
            "type": "ParamsSpec",
            "params": params,
        }


class ShapesSpec:
    """Top-level shape specification for a ``torch.compile`` call.

    ``params`` describes the arguments of the compiled callable — for a raw
    function this is the function's parameters, for an ``nn.Module`` this
    is the parameters of ``forward`` (excluding ``self``).

    ``assumptions`` is an optional list of ``SymBool`` expressions built
    from spec ``IntVar`` / ``ShapeVar`` values. Each assumption is wired
    into the shape env at compile time and asserted at runtime via the
    deferred-runtime-assert mechanism::

        from torch.fx.experimental.dynamic_spec import (
            ParamsSpec as PARAMS,
            ShapesSpec,
            ShapeVar as VAR,
            TensorSpec as T,
        )

        batch = VAR("batch", min=2, max=128)
        ep = torch.export.export(
            mod,
            (torch.randn(8, 3), torch.randn(16, 3)),
            dynamic_shapes=ShapesSpec(
                params=PARAMS(
                    {
                        "x": T([batch, 3]),
                        "y": T([batch * 2, 3]),  # derived expression
                    }
                ),
                assumptions=[batch % 2 == 0],
            ),
            strict=True,
        )

    ``globals`` is reserved for future use and will raise
    ``NotImplementedError`` if set.
    """

    def __init__(
        self,
        params: ParamsSpec | dict[str, ParamsSpecValue] | None = None,
        *,
        globals: Any = None,
        assumptions: Sequence[SymBool] | None = None,
    ) -> None:
        # Normalize attributes up front so partially-constructed instances
        # (e.g. when a later check raises) still have a stable shape.
        self._params: ParamsSpec | None = None
        self._assumptions: list[SymBool] = []

        if globals is not None:
            raise NotImplementedError("ShapesSpec.globals is not supported yet")
        # Auto-wrap a bare dict so callers can write
        # ``ShapesSpec({"x": ...})`` instead of
        # ``ShapesSpec(params=ParamsSpec({"x": ...}))``.
        if isinstance(params, dict):
            params = ParamsSpec(params)
        elif params is not None and not isinstance(params, ParamsSpec):
            raise TypeError(
                f"shapes_spec must be ParamsSpec, dict, or None, "
                f"got {type(params).__name__}"
            )
        self._params = params

        if assumptions is not None:
            for i, a in enumerate(assumptions):
                if not isinstance(a, SymBool):
                    raise TypeError(
                        f"ShapesSpec.assumptions[{i}]: expected SymBool, "
                        f"got {type(a).__name__}: {a!r}"
                    )
                _validate_spec_sym(a, where=f"ShapesSpec.assumptions[{i}]")
                self._assumptions.append(a)

    @property
    def assumptions(self) -> list[SymBool]:
        return list(self._assumptions)

    def __repr__(self) -> str:
        lines = ["shapes_spec:"]
        if self._params is not None:
            lines.append(f"{_INDENT}params:")
            param_repr = repr(self._params)
            for line in param_repr.splitlines():
                lines.append(_INDENT * 2 + line)
        if self._assumptions:
            lines.append(f"{_INDENT}assumptions:")
            for a in self._assumptions:
                lines.append(_INDENT * 2 + repr(a))
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "ShapesSpec",
            "params": None if self._params is None else self._params.to_jsonable(),
            "assumptions": [repr(a) for a in self._assumptions],
        }


# Shared error message for rejecting `dynamic_shapes=ShapesSpec(...)`
# combined with `prefer_deferred_runtime_asserts_over_guards=True`. Used
# in multiple export entry points; lifted out so the wording can't drift.
_SHAPES_SPEC_VS_DEFERRED_RUNTIME_ASSERTS_MSG = (
    "`prefer_deferred_runtime_asserts_over_guards=True` cannot be "
    "combined with `dynamic_shapes=ShapesSpec(...)`. ShapesSpec "
    "currently uses unbacked symbols only, which already emit "
    "runtime assertions; the flag has no effect."
)


def _coerce_to_shapes_spec(
    x: Any,
) -> ShapesSpec | None:
    """Normalize a user-supplied dynamic-spec value to ``ShapesSpec | None``.

    Accepts ``None``, an existing ``ShapesSpec``, a ``ParamsSpec``, or a plain
    ``dict`` (the same shape ``ParamsSpec`` accepts). The dict and
    ``ParamsSpec`` forms are auto-wrapped by ``ShapesSpec.__init__``.
    """
    if x is None:
        return None
    if isinstance(x, ShapesSpec):
        return x
    if isinstance(x, (dict, ParamsSpec)):
        return ShapesSpec(x)
    raise TypeError(
        f"dynamic spec expects a dict, ShapesSpec, or ParamsSpec, "
        f"got {type(x).__name__}"
    )


# Attribute used to store the resolved ``ShapesSpec`` on a decorated function
# (or ``nn.Module.forward``). Read by ``_resolve_dynamic_shapes`` below.
_DYNAMIC_SPEC_ATTR = "_dynamic_spec"


def dynamic_spec(spec: Any) -> Any:
    """Attach a ``ShapesSpec`` to a function (or ``nn.Module.forward``).

    When :func:`torch.compile`, :func:`torch.export.export`, or
    :func:`torch.fx.experimental.proxy_tensor.make_fx` is invoked on a
    function (or module) that carries this metadata, the attached
    ``ShapesSpec`` is used as the dynamic-shape spec **unless** the caller
    also passes an explicit ``dynamic_shapes=`` kwarg, in which case the
    two-source ambiguity is rejected with ``ValueError``.

    ``spec`` accepts the same types as the call-site
    ``dynamic_shapes=`` a ``dict``, a ``ParamsSpec``, or a ``ShapesSpec``.
    To attach assumptions, build a ``ShapesSpec(params=..., assumptions=[...])``
    and pass it.

    Usage::

        # Simple per-param mapping (no assumptions):
        @dynamic_spec({"x": TensorSpec([ShapeVar("B"), STATIC])})
        def fn(x): ...


        # With assumptions: build a ShapesSpec.
        B = ShapeVar("B")


        @dynamic_spec(
            ShapesSpec(
                ParamsSpec({"x": TensorSpec([B, STATIC])}),
                assumptions=[B % 2 == 0],
            )
        )
        def fn(x): ...

    Equivalent for an ``nn.Module``::

        class M(nn.Module):
            @dynamic_spec({"x": TensorSpec([ShapeVar("B"), STATIC])})
            def forward(self, x): ...
    """
    resolved = _coerce_to_shapes_spec(spec)
    if resolved is None:
        raise TypeError(
            f"dynamic_spec(): `spec` must be a dict, ParamsSpec, or "
            f"ShapesSpec, got {type(spec).__name__}."
        )

    def _decorator(fn: Any) -> Any:
        if hasattr(fn, _DYNAMIC_SPEC_ATTR):
            raise ValueError(
                "@dynamic_spec(...) is already attached to "
                f"{getattr(fn, '__qualname__', repr(fn))}. Stacking multiple "
                "@dynamic_spec decorators on the same function is not "
                "allowed; merge them into a single spec."
            )
        setattr(fn, _DYNAMIC_SPEC_ATTR, resolved)
        return fn

    return _decorator


def _resolve_dynamic_shapes(
    fn_or_module: Callable[..., Any] | torch.nn.Module,
    dynamic_shapes_kwarg: Any,
) -> Any:
    """Resolve the effective dynamic-shapes spec for a call site.

    Reads ``@dynamic_spec(...)`` metadata from ``fn_or_module`` (for an
    ``nn.Module``, the metadata is read from its bound ``forward`` method).
    Raises ``ValueError`` if both the decorator and an explicit
    ``dynamic_shapes_kwarg`` are present, since the precedence is ambiguous.
    Returns ``dynamic_shapes_kwarg`` if the decorator is absent, else the
    attached ``ShapesSpec``.
    """
    fn = fn_or_module
    import torch.nn

    if isinstance(fn_or_module, torch.nn.Module):
        fn = fn_or_module.forward
    attached = getattr(fn, _DYNAMIC_SPEC_ATTR, None)
    if attached is None:
        return dynamic_shapes_kwarg
    if dynamic_shapes_kwarg is not None:
        raise ValueError(
            "Both `@dynamic_spec(...)` is attached to the function/forward "
            "AND a `dynamic_shapes=` argument was passed. Provide only one."
        )
    log.info(
        "Using @dynamic_spec attached to %s: %s",
        getattr(fn, "__qualname__", repr(fn)),
        attached,
    )
    return attached
<|endoftext|>import operator
from collections import deque
from typing import NamedTuple

import torch
from torch.fx.experimental.partitioner_utils import (
    Device,
    get_extra_size_of,
    get_latency_of_partitioned_graph,
    get_partition_to_latency_mapping,
    NodeLatency,
    Partition,
    PartitionerConfig,
    PartitionMode,
)
from torch.fx.graph_module import GraphModule
from torch.fx.node import map_arg, Node
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.passes.split_module import split_module


class DAGNode:
    """DAGNode class maintains useful information for a partition (submodule),
    and its input submodules and output submodules.
    """

    def __init__(
        self,
        submodule_node: Node,
        input_nodes: list[Node],
        output_nodes: list[Node],
        logical_device_ids: list[int],
        size_bytes: int,
    ) -> None:
        self.submodule_node: Node = submodule_node
        self.input_nodes: list[Node] = input_nodes
        self.output_nodes: list[Node] = output_nodes
        self.logical_device_ids: list[int] = logical_device_ids
        self.size_bytes = size_bytes

    def __str__(self) -> str:
        return str(self.submodule_node)


class DAG:
    """DAG class contains all the DAG nodes"""

    def __init__(self) -> None:
        self.nodes: list[DAGNode] = []

    def create_node(
        self,
        submodule_node: Node,
        input_nodes: list[Node],
        output_nodes: list[Node],
        logical_devices: list[int],
        size_bytes: int,
    ) -> None:
        node = DAGNode(
            submodule_node, input_nodes, output_nodes, logical_devices, size_bytes
        )
        self.nodes.append(node)


class PartitionResult(NamedTuple):
    """NamedTuple used for returning DAG and a new fx module"""

    dag: DAG
    module_with_submodules: GraphModule


"""Following are some helper functions for partition manipulation"""


def reset_partition_device(partitions: list[Partition]) -> None:
    for partition in partitions:
        partition.logical_device_ids = []


def combine_two_partitions(
    partition_0: Partition, partition_1: Partition, partitions: list[Partition]
) -> None:
    """Given a list of partitions and its two partitions,
    combine these two partitions into a new one appending to the partitions
    and remove the previous two partitions from the list of partitions
    """
    partition = Partition(len(partitions))
    partition.nodes = partition_0.nodes.union(partition_1.nodes)
    partition.recalculate_mem_size()
    partitions.append(partition)
    partitions.remove(partition_0)
    partitions.remove(partition_1)
    reorganize_partitions(partitions)
    return


def set_parents_and_children(partitions: list[Partition]) -> None:
    """Given a list of partitions, mark parents and children for each partition"""
    # Go through all nodes in a partition.
    # If a node's user is in other partition,
    # then the other partition is this partition's children.
    # This partition is the other partition's parent
    for partition in partitions:
        partition.children = set()
        partition.parents = set()
    for partition in partitions:
        for node in partition.nodes:
            # For each node in the current partition, find its users
            users = node.users
            for n in users:
                # Find which partition the user node belongs to.
                # Note that if the node itself also belongs to that partition,
                # that partition is not the child of the current partition
                for p in partitions:
                    if p != partition and n in p.nodes and node not in p.nodes:
                        partition.children.add(p)
                        p.parents.add(partition)
    return


def reorganize_partitions(partitions: list[Partition]) -> None:
    """Given a list of partitions, reorganize partition id,
    its parents and its children for each partition
    """
    # Rearrange partition ids
    for i, partition in enumerate(partitions):
        partition.partition_id = i
    set_parents_and_children(partitions)
    return


def get_bfs_level_partition(partitions: list[Partition]) -> None:
    """Given a list of partitions,
    mark the bfs level for each partition
    """
    current_level: set[Partition] = set()
    visited: set[Partition] = set()
    for partition in partitions:
        # If a partition has no parent, it should be in root level
        if len(partition.parents) == 0:
            current_level.add(partition)
    next_level: set[Partition] = set()
    level = 0
    # bfs
    while current_level:
        partition = current_level.pop()
        partition.bfs_level = level
        visited.add(partition)
        children = partition.children
        for child in children:
            if child not in next_level:
                next_level.add(child)
        if not current_level:
            current_level = next_level.copy()
            next_level = set()
            level += 1
    return


def get_node_to_partition_mapping(partitions: list[Partition]) -> dict[Node, int]:
    """Given a list of partitions,return node to partition mapping"""
    node_to_partition: dict[Node, int] = {}
    for partition in partitions:
        for node in partition.nodes:
            node_to_partition[node] = partition.partition_id
    return node_to_partition


def get_logical_id_to_device(devices: list[Device]) -> dict[int, Device]:
    """Get a mapping from device logical ID to Device object."""
    logical_id_to_device: dict[int, Device] = {}
    for d in devices:
        logical_id_to_device[d.logical_id] = d
    return logical_id_to_device


def get_device_partition_stats(
    partitions: list[Partition], devices: list[Device]
) -> tuple[dict[Device, list[Partition]], dict[Device, int], list[Partition]]:
    """Given a list of partitions and a list of devices, returns:
    1. A mapping from device to partitions on it;
    2. A mapping from device to its remaining memory size;
    3. A list of partitions that do not have a device.
    """
    # logical id to device
    logical_id_to_device = get_logical_id_to_device(devices)
    # Track partitions on device
    device_to_partitions: dict[Device, list[Partition]] = {}
    # Track device's left mem size
    device_to_left_mem_bytes: dict[Device, int] = {}
    for d in devices:
        device_to_partitions[d] = []
        device_to_left_mem_bytes[d] = d.available_mem_bytes

    # Deal with the partitions that already have a device
    # and also collect all partitions without a device (no_device_partitions)
    no_device_partitions = []
    for partition in partitions:
        if partition.logical_device_ids != []:
            for logical_id in partition.logical_device_ids:
                device = logical_id_to_device[logical_id]
                device_to_partitions[device].append(partition)
                device_to_left_mem_bytes[device] -= partition.used_mem_bytes
        else:
            no_device_partitions.append(partition)

    return (
        device_to_partitions,
        device_to_left_mem_bytes,
        no_device_partitions,
    )


def get_device_to_partitions_mapping(
    partitions: list[Partition], devices: list[Device]
) -> bool:
    """Given a list of partitions and a list of devices,
    map each partition into a device.
    """

    def calculate_extra_mem_bytes_needed_for(
        partition: Partition, partitions: list[Partition]
    ) -> int:
        all_nodes: set[Node] = set()
        for p in partitions:
            all_nodes = all_nodes.union(p.nodes)
        if len(all_nodes) == 0:
            return partition.used_mem_bytes
        all_nodes = all_nodes.union(partition.nodes)
        extra_size_needed = 0
        for node in partition.nodes:
            extra_size_needed += get_extra_size_of(node, all_nodes)
        return extra_size_needed

    def find_device_for(partition: Partition) -> bool:
        """Given a partition, find a logical device for the partition
        The algorithm is to put the partition on the device
        that has just enough mem left for that partition.
        device_to_left_mem_bytes is a dictionary between device and its left mem size
        sorted by its left mem size
        """
        for d in device_to_left_mem_bytes:
            extra_size_needed = calculate_extra_mem_bytes_needed_for(
                partition, device_to_partitions[d]
            )
            if extra_size_needed < device_to_left_mem_bytes[d]:
                device_to_partitions[d].append(partition)
                partition.logical_device_ids.append(d.logical_id)
                device_to_left_mem_bytes[d] -= extra_size_needed
                return True
        return False

    (
        device_to_partitions,
        device_to_left_mem_bytes,
        no_device_partitions,
    ) = get_device_partition_stats(partitions, devices)

    # Find devices for all the partitions without a device
    found_device = True
    for partition in no_device_partitions:
        device_to_left_mem_bytes = dict(
            sorted(device_to_left_mem_bytes.items(), key=operator.itemgetter(1))
        )
        found_device = find_device_for(partition)
        if not found_device:
            break
    return found_device


def check_dependency(partition: Partition) -> bool:
    """Given a partition,check if there is a circular dependency on
    this partition using bfs
    """
    visited: set[Partition] = {partition}
    queue: deque[Partition] = deque([partition])
    while queue:
        p = queue.popleft()
        for child in p.children:
            if child == partition:
                return True
            else:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
    return False


class Partitioner:
    """A fx module may not fit into one device.
    Partitioner class helps partition one fx module into submodules (partitions),
    so that the submodules can be executed crossing different accelerators.
    The main function of this class is self.partition_graph.
    It partitions the fx module based on the scheme specified in partition_config
    A DAG structure is returned
    along with a new fx module with submodule nodes.
    """

    def __init__(self) -> None:
        self.partitions: list[Partition] = []
        self.node_to_partition: dict[Node, int] = {}
        self.devices: list[Device] = []

    def partition_graph(
        self,
        fx_module: GraphModule,
        torch_module: torch.nn.Module,
        partitioner_config: PartitionerConfig,
    ) -> PartitionResult:
        """Given the fx module, torch module and partitioner_config,
        find the partitions, do the partitions,
        and then return a DAG and a new fx module with submodule nodes (partitions)
        """
        self.graph_module = fx_module
        self.torch_module = torch_module
        self.devices = partitioner_config.devices
        if len(self.devices) == 0:
            raise RuntimeError("No devices")
        # Tag the size in bytes to all nodes in the graph_module.
        get_size_of_all_nodes(self.graph_module)
        # Check if there are op nodes in the fx module
        nodes = self.graph_module.graph.nodes
        if all(node.op in {"placeholder", "get_attr", "output"} for node in nodes):
            raise RuntimeError("No Partition since no operations in the module")
        # Calculate total size of the fx module
        total_size_of_graph = 0
        for node in nodes:
            if node.op == "output":
                break
            total_size_of_graph += node.size_bytes.total_size
        # Find the device with the max mem size
        device_with_max_mem = max(self.devices, key=lambda d: d.available_mem_bytes)
        # AOT based partition
        if partitioner_config.mode == PartitionMode.aot_based:
            self.aot_based_partition(
                partitioner_config.node_to_partition_mapping,
                partitioner_config.partition_to_logical_device_mapping,
            )
        # Single partition if the whole module can be fit into one device
        elif total_size_of_graph <= device_with_max_mem.available_mem_bytes:
            self.find_single_partition(
                total_size_of_graph, logical_device_id=device_with_max_mem.logical_id
            )
        elif total_size_of_graph > sum(d.available_mem_bytes for d in self.devices):
            raise RuntimeError("Devices do not have enough memory for the module")
        else:
            # Sparse nn based partition
            if partitioner_config.mode == PartitionMode.sparse_nn:
                available_mem_bytes = self.devices[0].available_mem_bytes
                if not all(
                    device.available_mem_bytes == available_mem_bytes
                    for device in self.devices
                ):
                    raise RuntimeError("All devices must have same memory size!")
                # sparse_nn_partition only support same memory size
                # TODO: add different size support for sparse_nn_partition
                self.sparse_nn_partition(available_mem_bytes)
            # Cost aware partition
            elif partitioner_config.mode == PartitionMode.cost_aware:
                self.cost_aware_partition(
                    partitioner_config.transfer_rate_bytes_per_sec,
                    partitioner_config.node_to_latency_mapping,
                )
            # KL based partition
            elif partitioner_config.mode == PartitionMode.kl_based:
                self.kl_based_partition(
                    partitioner_config.transfer_rate_bytes_per_sec,
                    partitioner_config.node_to_latency_mapping,
                )
            else:
                self.size_based_partition()

        # Saturate host if possible.
        if partitioner_config.saturate_host:
            self.saturate_host()

        # Partition the graph module based on the partition assignment.
        module_with_submodules = self.do_partition()

        # The DAG contains DAGNodes with info of each partition's input nodes, output nodes
        # and how partitions are connected.
        dag = self.dump_dag(module_with_submodules)
        ret = PartitionResult(dag, module_with_submodules)
        return ret

    def find_single_partition(
        self, total_size_of_graph: int, logical_device_id: int = 0
    ) -> None:
        """Fit the whole fx module into one device"""
        partition_0 = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op == "output":
                # Skip the output node, but there can
                # be nodes after the output in certain cases.
                continue
            partition_0.nodes.add(node)
        partition_0.used_mem_bytes = total_size_of_graph
        partition_0.logical_device_ids = [logical_device_id]
        # Get the node to partition mapping
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        return

    def size_based_partition(self) -> None:
        """This method is to partition the fx module based on memory size.
        It uses greedy approach. The result may not be the best.
        The basic idea is:
        Step 1:
        Find a device which has enough memory to fit the current node, create an empty partition
        with the size of that device.
        Then keep adding the following nodes into the partition until the partition is full.
        Step 2:
        Repeat Step 1 until no device left
        Step 3:
        If some nodes are left, create a partition for each left node (single node partition).
        and then try to map those partitions into logical devices with enough mem left.
        """

        def find_device_based_on_size(node: Node) -> Device:
            """Given a node, this function is to find a logical device
            that could fit the node.
            """
            mem_size_needed = get_extra_size_of(node, set())
            device = Device("", -1, -1)
            for d in self.devices:
                if (
                    d not in occupied_devices
                    and d.available_mem_bytes >= mem_size_needed
                ):
                    device = d
                    break
            if device.available_mem_bytes < 0:
                raise RuntimeError(str(node) + " is too large to fit any device")
            occupied_devices.append(device)
            return device

        # Track partition and its left mem size
        partition_to_left_mem_bytes: dict[Partition, int] = {}
        # Track all the devices that have been used
        occupied_devices: list[Device] = []
        partition = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op in {"call_module", "call_method", "call_function"}:
                # Check if there are devices left
                if len(self.partitions) <= len(self.devices):
                    total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                    # Check if the current partition is the very first partition
                    if partition.used_mem_bytes == 0:
                        # Find a device to fit the first node, return available mem size
                        device = find_device_based_on_size(node)
                        occupied_devices.append(device)
                        # Update partition and its left mem size
                        partition_to_left_mem_bytes[partition] = (
                            device.available_mem_bytes
                        )
                        # Update available mem for the current partition
                        partition.logical_device_ids.append(device.logical_id)
                    else:
                        # The current partition is not the first partition
                        # Check if the current node can fit into current partition
                        if (
                            partition_to_left_mem_bytes[partition]
                            < total_size_of_input_nodes
                        ):
                            # Check if no device is left
                            if len(self.partitions) == len(self.devices):
                                # No device is left
                                # Create the first single node partition for the current node
                                self.create_single_node_partition(node)
                                continue
                            # Some devices are still left
                            # Create a new partition with a mem size that is enough for the current node
                            device = find_device_based_on_size(node)
                            partition = self.create_partition()
                            total_size_of_input_nodes = get_extra_size_of(
                                node, partition.nodes
                            )
                            partition_to_left_mem_bytes[partition] = (
                                device.available_mem_bytes
                            )
                            partition.logical_device_ids.append(device.logical_id)
                    partition.add_node(node)
                    partition_to_left_mem_bytes[partition] -= total_size_of_input_nodes
                # Create single node partitions if no device is left
                else:
                    self.create_single_node_partition(node)
        reorganize_partitions(self.partitions)
        # Get the node to partition mapping
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        # Mapping all partitions into device
        found_partition_to_device_mapping = get_device_to_partitions_mapping(
            self.partitions, self.devices
        )
        if not found_partition_to_device_mapping:
            raise RuntimeError("Cannot Get a Valid Partition to Logical Device Mapping")
        return

    def saturate_host(self) -> None:
        """Saturate host by assigning replicates to unused devices with enough memory.
        It uses a greedy approach to find a next available set of devices to place all split
        partitions: For each used device, it searches for an idle device with minimal memory
        size that can hold all the partition located on that device; If the search is successful
        for all used devices, it then assigns the new devices' logical ID to the corresponding
        partition.
        """
        (
            device_to_partitions,
            device_to_left_mem_bytes,
            no_device_partitions,
        ) = get_device_partition_stats(self.partitions, self.devices)

        if len(no_device_partitions) != 0:
            raise AssertionError(
                f"Expect no_device_partitions has 0 device, but get {len(no_device_partitions)}"
            )

        # Devices that hold partitions
        used_devices = [d for d in self.devices if len(device_to_partitions[d]) > 0]
        # Track replicates of the assigned devices
        replicated_device_to_used_device: dict[Device, Device] = {}

        while len(used_devices) * 2 + len(replicated_device_to_used_device) <= len(
            self.devices
        ):
            # Success flag for this round
            success = True
            # Devices that have not been assigned
            idle_devices = [
                d
                for d in self.devices
                if d not in used_devices and d not in replicated_device_to_used_device
            ]
            # Temporary mapping from replicated device to original device
            temp_replicate_mapping = {}

            # Find a new device to replicate all partitions on a used device
            for used_device in used_devices:
                # Idle devices that have enough memory
                available_devices = [
                    d
                    for d in idle_devices
                    if d.available_mem_bytes
                    >= used_device.available_mem_bytes
                    - device_to_left_mem_bytes[used_device]
                ]
                if len(available_devices) == 0:
                    success = False
                    break
                new_device = min(available_devices, key=lambda d: d.available_mem_bytes)
                idle_devices.remove(new_device)
                temp_replicate_mapping[new_device] = used_device

            if not success:
                break
            replicated_device_to_used_device.update(temp_replicate_mapping)

        # Update logical device IDs assigned to the partitions
        for (
            replicate_device,
            original_device,
        ) in replicated_device_to_used_device.items():
            logical_id = replicate_device.logical_id
            for partition in device_to_partitions[original_device]:
                partition.logical_device_ids.append(logical_id)
        for p in self.partitions:
            print(p.logical_device_ids)

    def do_partition(self) -> GraphModule:
        """Return a new fx module with submodule nodes (partitions)."""
        module_with_submodules = split_module(
            self.graph_module,
            self.torch_module,
            lambda node: self.node_to_partition[node],
        )
        return module_with_submodules

    def dump_dag(self, module_with_submodules: GraphModule) -> DAG:
        """Return the dag structure and the new fx module with submodules."""
        dag = DAG()
        for node in module_with_submodules.graph.nodes:
            if node.op == "output":
                break
            if node.op in {"placeholder", "get_attr"}:
                continue
            if node.target is operator.__getitem__:
                continue
            input_nodes: dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)
            map_arg(node.kwargs, input_nodes.setdefault)
            # When a node has two or more output nodes,
            # it outputs its result to 'getitem' nodes.
            # Those 'getitem' nodes are the output node for this node.
            # Otherwise, the output node is this node itself.
            if len(node.users) > 1:
                output_nodes = list(node.users)
            else:
                output_nodes = [node]
            partition_id = int(node.name.rsplit("_", 1)[-1])
            device_ids = self.partitions[partition_id].logical_device_ids
            size_bytes = self.partitions[partition_id].used_mem_bytes
            dag.create_node(
                node, list(input_nodes), output_nodes, device_ids, size_bytes
            )
        return dag

    def create_partition(self) -> Partition:
        """Create a partition and append it to self.partitions."""
        partition_id = len(self.partitions)
        partition = Partition(partition_id)
        self.partitions.append(partition)
        return partition

    def create_single_node_partition(self, node: Node) -> None:
        """Create a partition for a single node"""
        partition = self.create_partition()
        partition.add_node(node)
        return

    def sparse_nn_partition(self, available_mem_bytes: int) -> None:
        """This method partition a sparse nn module.
        It is size based partition but different from size_based_partition,
        it only works when all the devices have same memory size (available_mem_bytes).
        In the future, devices with different mem sizes will be supported like size_based_partition.
        It first traverse all the nodes and do the partitions based on the same memory size.
        If the current partition does not have enough memory left for a new op node
        (call_module, call_method, call_function), a new partition is created.
        When crossing the boundary between non-embedding nodes and embedding nodes,
        a new partition is created regardless.
        For example, if the current node is a non-embedding node but the next node is an
        embedding node, a new partition is created for the next node.
        After the partition, the partitions are combined as much as possible.
        The rule is that a non-embedding partition only
        combines with another non-embedding one.
        So as the embedding partitions.
        """

        def combine_partitions_based_on_size(
            partitions: list[Partition], available_mem_bytes: int
        ) -> None:
            """Combining small partitions together to keep as less partitions as possible.
            Here is an example of the algorithm to do this:
            Assume some partitions, we first sort them based on partition used memory size.
            [(partition_4, 1), (partition_3, 1), (partition_2, 2), (partition_1, 7), (partition_0, 9)]
            The available memory is 10.
            step 1: self.find_partition_to_combine_based_on_size()
            First, mark bfs level for each partition
            Second, look the smallest partition, partition_4: 10 - 1 = 9
            It means any partition has a used memory equal or less than 9 could combine this partition
            We go from the largest and selection partition_0.
            Check the bfs level for two partitions, if the level difference is less than 2,
            it can be combined.
            step 2: repeat step 1 until no partitions can be combined
            """
            find_combination = True
            while find_combination:
                # Sort partitions based on memory size
                sorted_partitions = sorted(partitions, key=lambda p: p.used_mem_bytes)
                # Mark bfs level
                get_bfs_level_partition(self.partitions)
                find_combination, partitions = find_partition_to_combine_based_on_size(
                    sorted_partitions,
                    available_mem_bytes,
                    partitions,
                )
            return

        def calculate_mem_bytes_needed(p1: Partition, p2: Partition) -> int:
            """Given two partitions, calculate how many mem bytes
            are needed if two partitions are combined
            """
            nodes = p1.nodes.union(p2.nodes)
            mem_bytes_needed = 0
            for node in nodes:
                mem_bytes_needed += get_extra_size_of(node, nodes)
            return mem_bytes_needed

        def find_partition_to_combine_based_on_size(
            sorted_partitions: list[Partition],
            available_mem_bytes: int,
            partitions: list[Partition],
        ) -> tuple[bool, list[Partition]]:
            """step 1 in combine_partition_based_on_size()"""
            find_combination = False
            smallest_partition = sorted_partitions.pop(0)
            for p in sorted_partitions[::-1]:
                if abs(smallest_partition.bfs_level - p.bfs_level) <= 1:
                    # Calculate how many bytes needed if combined
                    mem_bytes_needed = calculate_mem_bytes_needed(p, smallest_partition)
                    if mem_bytes_needed <= available_mem_bytes:
                        combine_two_partitions(p, smallest_partition, self.partitions)
                        partitions.remove(smallest_partition)
                        partitions.remove(p)
                        partitions.append(self.partitions[-1])
                        find_combination = True
                        break
            return find_combination, partitions

        def reset_partition_in_sparse_nn(partition: Partition) -> Partition:
            """Finalize current partition and create a new one."""
            if in_embedding_region:
                embedding_partitions.append(partition)
            else:
                non_embedding_partitions.append(partition)
            partition = self.create_partition()
            # pyrefly: ignore [missing-attribute]
            partition.left_mem_bytes = available_mem_bytes
            return partition

        def finalize_partition(partition: Partition) -> None:
            """Finalize current partition without creating a new one."""
            if in_embedding_region:
                embedding_partitions.append(partition)
            else:
                non_embedding_partitions.append(partition)

        def is_embedding_node(node: Node) -> bool:
            """Check if a node is an embedding node"""
            if node.op == "call_module":
                submodule = self.graph_module
                for atom in str(node.target).split("."):
                    if not hasattr(submodule, atom):
                        raise RuntimeError(
                            f"Module {submodule} has no attribute {atom}"
                        )
                    submodule = getattr(submodule, atom)
                    if "Embedding" in str(submodule):
                        return True
            return False

        # Track embedding partitions and non-embedding partitions separately
        embedding_partitions: list[Partition] = []
        non_embedding_partitions: list[Partition] = []
        # A Flag to check the boundary
        in_embedding_region: bool = False
        partition = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op in {"call_module", "call_method", "call_function"}:
                # Check if crossing the boundary between embedding nodes and non embedding nodes
                if is_embedding_node(node) != in_embedding_region:
                    # Crossing the boundary
                    # Check if the current partition is an empty partition
                    if partition.used_mem_bytes != 0:
                        # The current partition isn't an empty partition. Create a new one.
                        partition = reset_partition_in_sparse_nn(partition)
                    in_embedding_region = not in_embedding_region
                total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                if (
                    total_size_of_input_nodes + partition.used_mem_bytes
                    > available_mem_bytes
                ):
                    partition = reset_partition_in_sparse_nn(partition)
                    total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                    if total_size_of_input_nodes > available_mem_bytes:
                        raise RuntimeError(
                            node.target + " is too large to fit into a device"
                        )
                partition.add_node(node)
        finalize_partition(partition)
        # Set parents and children for partitions
        set_parents_and_children(self.partitions)
        # Combining non-embedding partitions
        combine_partitions_based_on_size(non_embedding_partitions, available_mem_bytes)
        # Combining embedding partitions
        combine_partitions_based_on_size(embedding_partitions, available_mem_bytes)
        total_size_of_non_embedding_partitions = 0
        for partition in non_embedding_partitions:
            total_size_of_non_embedding_partitions += partition.used_mem_bytes
        # Check if devices are enough for all partitions
        if len(embedding_partitions) > len(self.devices):
            msg = (
                "Need "
                + str(len(embedding_partitions))
                + " devices, but only "
                + str(len(self.devices))
                + " provided"
            )
            raise RuntimeError(msg)
        occupied_devices = []
        for i, partition in enumerate(embedding_partitions):
            # Check if all non-embedding partitions can fit into embedding partition devices
            if (
                total_size_of_non_embedding_partitions + partition.used_mem_bytes
                > available_mem_bytes
            ):
                raise RuntimeError(
                    "partition_"
                    + str(partition.partition_id)
                    + "(embedding partition) and non embedding partitions can not fit into one device"
                )
            else:
                # Add logical device to the partition
                partition.logical_device_ids = [self.devices[i].logical_id]
                occupied_devices.append(self.devices[i].logical_id)
        # Add logical devices to the non_embedding_partitions
        for partition in non_embedding_partitions:
            partition.logical_device_ids = occupied_devices
        # Get the node to partition mapping
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        return

    def cost_aware_partition(
        self,
        transfer_rate_bytes_per_sec: float,
        node_to_latency_mapping: dict[Node, NodeLatency],
    ) -> None:
        """This method is to partition the fx module based on the cost.
        The cost is the total latency of running the whole fx module.
        In partitioner_utils.py, the cost model is built.
        The cost aware partition algorithm is:
        #1. At every beginning, each node is a partition.
            Then we map all the partitions to the devices
            and calculate the cost
        #2. Then try to pre-combine any two of the partitions if the two
            partitions can be combined.
            (the bfs level is less than 2 or two partitions are connected and
            can find partition to device mapping)
            See if any partition pair could reduce the current cost.
            Choose the pair that shows the minimum cost and then combine them
        #3. Repeat #2 until the cost cannot be reduced.
        """

        def try_combining_partitions(
            p0_index: int, p1_index: int, partitions: list[Partition]
        ) -> float:
            """Given two partitions and a list of partitions, combine these two partitions
            and see what is the cost of the modified partition list
            """
            p0 = partitions[p0_index]
            p1 = partitions[p1_index]
            """If two partitions' bfs level are less than 2 or two partitions are connected to each other,
               then they can be combined
            """
            if (
                (abs(p0.bfs_level - p1.bfs_level) <= 1)
                or (p0 in p1.parents)
                or p0 in (p1.children)
            ):
                combine_two_partitions(p0, p1, partitions)
                # Check if a circular dependency exists after combining
                if check_dependency(partitions[-1]):
                    return float("inf")
                # Check if the modified partition list can be mapped to devices after combination
                reset_partition_device(partitions)
                found_device = get_device_to_partitions_mapping(
                    partitions, self.devices
                )
                if not found_device:
                    return float("inf")
                # Calculate the new cost
                partition_to_latency_mapping = get_partition_to_latency_mapping(
                    partitions, node_to_latency_mapping
                )
                cost = get_latency_of_partitioned_graph(
                    partitions,
                    partition_to_latency_mapping,
                    transfer_rate_bytes_per_sec,
                )
                return cost
            # If two partition can not be combined, the cost is inf
            return float("inf")

        def search_combination(
            transfer_rate_bytes_per_sec: float,
            node_to_latency_mapping: dict[Node, NodeLatency],
        ) -> bool:
            """Given transfer rate between partitions and each node's latency,
            find two partitions to combine so the cost of the partitions can
            be reduced.
            The algorithm is :
            1. Go through all the partition pairs and see
            if any pair of partitions can be combined.
            2. Calculate the cost after the combination.
            3. Select the minimum cost and combine its corresponding partition pair.
            """
            partition_to_latency_mapping = get_partition_to_latency_mapping(
                self.partitions, node_to_latency_mapping
            )
            cost = get_latency_of_partitioned_graph(
                self.partitions,
                partition_to_latency_mapping,
                transfer_rate_bytes_per_sec,
            )
            if len(self.partitions) == 1:
                return False
            partition_pair: list[int] = []
            for i in range(len(self.partitions) - 1):
                for j in range(i + 1, len(self.partitions)):
                    # Try to combine the partition pair
                    # and see the new cost after combination
                    new_cost = try_combining_partitions(i, j, self.partitions[:])
                    if new_cost <= cost:
                        partition_pair = [i, j]
                        cost = new_cost
                    reorganize_partitions(self.partitions)
            # If a partition pair is found, combine them
            if len(partition_pair) != 0:
                p0 = self.partitions[partition_pair[0]]
                p1 = self.partitions[partition_pair[1]]
                combine_two_partitions(p0, p1, self.partitions)
            get_bfs_level_partition(self.partitions)
            reset_partition_device(self.partitions)
            get_device_to_partitions_mapping(self.partitions, self.devices)
            return len(partition_pair) != 0

        for node in self.graph_module.graph.nodes:
            if node.op not in {"placeholder", "get_attr", "output"}:
                self.create_single_node_partition(node)
        # Set up parent partitions and children partitions for each partition
        set_parents_and_children(self.partitions)
        # Get bfs level for each partition
        get_bfs_level_partition(self.partitions)
        find_combination = True
        while find_combination:
            # Search for a pair partition to generate the minimum new cost,
            # then combine them
            find_combination = search_combination(
                transfer_rate_bytes_per_sec, node_to_latency_mapping
            )
        # Make sure all partitions are set up correctly
        reorganize_partitions(self.partitions)
        # Set up node to partition mapping
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        return

    def kl_based_partition(
        self,
        transfer_rate_bytes_per_sec: float,
        node_to_latency_mapping: dict[Node, NodeLatency],
    ) -> None:
        """This function is a cost aware partition based
        on Kernighan-Lin algorithm.
        First, the graph is partitioned using size_based_partition.
        Then, each node is swapped with any other node in a different
        partition, and at the same time, the cost is estimated after
        the swapping.
        For example, we have nodes n0, n1, n2, n3 and n4.
        Using size_based_partition, n0 and n1 are in Partition p0.
        n2, n3 and n4 in Partition p1. The current cost is estimated.
        We first tried using n0 to swap with n2 from the other partition.
        Then we see that swapping n0 and n2 shows a lower cost
        than the current cost and it is the minimum among other pairs like
        (n0, None)(This means moving n0 to Partition without swapping other nodes),
        (n0, n3) and (n0, n4). We swap n0 and n2 and set the new cost
        as the current cost.
        Then We repeat this process for all the other nodes until all swapping pairs
        are tried.
        """

        def swap_nodes(
            n0: Node | None, n1: Node | None, p0: Partition, p1: Partition
        ) -> None:
            # Either n0 or n1 could be None
            # That means we simply move the node
            # to another partition
            if n0 is not None:
                p0.remove_node(n0)
                p1.add_node(n0)
            if n1 is not None:
                p0.add_node(n1)
                p1.remove_node(n1)

        def try_swap_nodes(
            n0: Node | None,
            n1: Node | None,
            p0: Partition,
            p1: Partition,
            node_to_latency_mapping: dict[Node, NodeLatency],
            transfer_rate_per_sec: float,
        ) -> float:
            cost = float("inf")
            swap_nodes(n0, n1, p0, p1)
            # Reorganize partitions after swapping
            reorganize_partitions(self.partitions)
            # Check if there is a circular dependency after swapping
            if (not check_dependency(p0)) and (not check_dependency(p1)):
                reset_partition_device(self.partitions)
                partition_to_latency_mapping = get_partition_to_latency_mapping(
                    self.partitions, node_to_latency_mapping
                )
                # Check if all partitions can be mapped to logical devices after swapping
                found_device = get_device_to_partitions_mapping(
                    self.partitions, self.devices
                )
                if not found_device:
                    cost = float("inf")
                else:
                    cost = get_latency_of_partitioned_graph(
                        self.partitions,
                        partition_to_latency_mapping,
                        transfer_rate_bytes_per_sec,
                    )
            # Swap back and reset all partitions back to original
            swap_nodes(n1, n0, p0, p1)
            reorganize_partitions(self.partitions)
            reset_partition_device(self.partitions)
            get_device_to_partitions_mapping(self.partitions, self.devices)
            return cost

        def swap_node_to_partition(
            node: Node,
            p0: Partition,
            p1: Partition,
            node_to_latency_mapping: dict[Node, NodeLatency],
            transfer_rate_per_sec: float,
        ) -> tuple[float, list[Node]]:
            """This function helps to swap one node from partition p0
            with all the nodes in another partition p1
            """
            p1_nodes = list(p1.nodes) + [None]
            min_cost = float("inf")
            node_pair: list[Node] = []
            for n1 in p1_nodes:
                # Ignore the node if it is not a op node
                if n1 is not None and n1.op in {"placeholder", "get_attr"}:
                    continue
                # Try swapping node in p0 with n1 in p1
                cost = try_swap_nodes(
                    node, n1, p0, p1, node_to_latency_mapping, transfer_rate_per_sec
                )
                if cost < min_cost:
                    # pyrefly: ignore [bad-assignment]
                    node_pair = [node, n1]
                    min_cost = cost
            return cost, node_pair  # type: ignore[possibly-undefined]

        # First use size_base_partition
        self.size_based_partition()
        partition_to_latency_mapping = get_partition_to_latency_mapping(
            self.partitions, node_to_latency_mapping
        )
        # Calculate the cost of the partitions
        cost = get_latency_of_partitioned_graph(
            self.partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec
        )
        # Keep tracking the node pair that shows the better cost
        node_pair: list[Node] = []
        # Keep tracking the partition pair of node pair
        partition_pair: list[Partition] = []
        # Collect all the op nodes from the graph
        op_nodes = [
            n
            for n in self.graph_module.graph.nodes
            if n.op not in {"placeholder", "get_attr", "output"}
        ]
        for node in op_nodes:
            # Find which partition the current node belongs
            p0_index = self.node_to_partition[node]
            p0 = self.partitions[p0_index]
            # Go through all the other partitions to swap
            # with other nodes from those partitions
            for p1_index, _ in enumerate(self.partitions):
                if p0_index != p1_index:
                    p1 = self.partitions[p1_index]
                    new_cost, new_node_pair = swap_node_to_partition(
                        node,
                        p0,
                        p1,
                        node_to_latency_mapping,
                        transfer_rate_bytes_per_sec,
                    )
                    # Update the cost
                    # Track the swapped node pair and their partitions
                    if new_cost < cost:
                        cost = new_cost
                        node_pair = new_node_pair
                        partition_pair = [p0, p1]
            # Do the swapping after trying all the nodes from a partition
            if len(node_pair) != 0:
                swap_nodes(
                    node_pair[0], node_pair[1], partition_pair[0], partition_pair[1]
                )
                reorganize_partitions(self.partitions)
                get_device_to_partitions_mapping(self.partitions, self.devices)
        reorganize_partitions(self.partitions)
        # Mapping the device to the partition
        get_device_to_partitions_mapping(self.partitions, self.devices)
        return

    def aot_based_partition(
        self,
        node_to_partition_mapping: dict[Node, int],
        partition_to_logical_device_mapping: dict[int, list[int]],
    ) -> None:
        """This function helps to rebuild the partitions given the nodes and its
        corresponding partition id
        """
        partition_id_to_partition_mapping: dict[int, Partition] = {}
        self.node_to_partition = node_to_partition_mapping
        for node in self.node_to_partition:
            partition_id = self.node_to_partition[node]
            # If the requested partition has not been created, create the partition
            if partition_id not in partition_id_to_partition_mapping:
                partition = Partition(partition_id)
                self.partitions.append(partition)
                partition_id_to_partition_mapping[partition_id] = partition
                partition.logical_device_ids = partition_to_logical_device_mapping[
                    partition_id
                ]
            else:
                partition = partition_id_to_partition_mapping[
                    self.node_to_partition[node]
                ]
            # Add the current node into the partition
            partition.add_node(node)
<|endoftext|>from __future__ import annotations


"""
This file does three things:
- Contains the definition of SymNode
- Installs all the magic methods into SymBool, SymFloat, SymInt at import time
- Does not depend on sympy at import time

As this file is imported from within torch/__init__.py we do not want it to depend on SymPy
to avoid having to load SymPy at import time, as doing so is *very* slow.
"""


import builtins
import functools
import inspect
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache, update_wrapper
from typing import Any, overload, TYPE_CHECKING

import torch
import torch._logging.structured as structured

# NB: The sym_* functions are used via getattr() and must be imported here.
from torch import (
    sym_float,
    sym_ite,
    sym_max,
    sym_min,
    sym_not,
    SymBool,
    SymFloat,
    SymInt,
)
from torch._logging import dtrace_structured


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    import sympy

    from torch.fx.experimental.symbolic_shapes import ShapeEnv

log = logging.getLogger(__name__)
sym_node_log = torch._logging.getArtifactLogger(__name__, "sym_node")

# Sentinel value to indicate "don't compute hint" vs actual None
# When passed as hint to SymNode, it means we already know hint is unavailable
# and should not waste time calling compute_hint()
_NO_HINT: object = object()
# Type alias for hint values (including the sentinel)
HintType = bool | float | int | None


__all__ = ["SymNode", "method_to_operator", "magic_methods", "DynamicInt"]


from torch.types import py_sym_types as SymTypes


@overload
def _to_symtype(t: type[bool]) -> type[SymBool]: ...
@overload
def _to_symtype(t: type[int]) -> type[SymInt]: ...
@overload
def _to_symtype(t: type[float]) -> type[SymFloat]: ...
@overload
def _to_symtype(t: type) -> type: ...
def _to_symtype(t: type) -> type:
    if t is bool:
        return SymBool
    if t is int:
        return SymInt
    if t is float:
        return SymFloat
    return t


# TODO: An incomplete list
# 1. Set variables to be equal when we do equality
# 2. Specialize on 0/1 when we do subtraction
class SymNode:
    """
    This is a type erased SymInt/SymFloat which we use to do actual operations.
    End users don't touch this.  Magic methods are NOT defined on this object.
    """

    # Note [optimized_summation]: indicates that SymNode is an Add expression of the form
    # a + b + c + d... etc where all terms are unique symbols. This allows us to do some optimizations
    # for common patterns see _optimized_add.

    # The unfortunate reason we have this here is because sympy sets  __slots__ = () for add expression,
    # so we cannot add the attribute directly to the sympy expression. Furthermore, we cannot use it as
    # a weak dictionary key either! So instead, we attach the attribute here to the SymNode.
    _optimized_summation: bool = False

    def __init__(
        self,
        expr: object,
        shape_env: ShapeEnv | None,
        pytype: type,
        hint: HintType | object,
        constant: int | float | bool | None = None,
        fx_node: object = None,
        optimized_summation: bool = False,
    ) -> None:
        self._expr = expr
        self.shape_env = shape_env
        self.pytype = pytype
        self._optimized_summation = optimized_summation
        self._expr_ver = -1
        self._expr_cache: object | None = None

        # What's the difference between hint and constant?
        #
        # - A constant is known to be invariant across invocations of the model;
        #   it will always be this value.  We only really know this when we
        #   encounter an honest-to-goodness literal (when wrapping it into
        #   a SymNode, we set constant.)  Most of the time, constant is None
        #
        # - A hint is a *particular* value from the particular run we are
        #   tracing, but it may vary the next time around.  It's useful to
        #   keep this around, as if we need a concrete value from a SymNode,
        #   we will return the hint and guard on the expression that produced
        #   it giving the same hint next time around.  The hint is not
        #   guaranteed to be set either: if you have an unbacked SymNode,
        #   there won't be any hint; it was the result of some tensor-dependent
        #   computation, but we don't know what it actually is because we
        #   haven't actually run the tensor computation.
        #
        # If _hint is None, we will query maybe_evaluate_static(compute_hint=True)
        # in hopes that we've learned enough about the unbacked symints to
        # discharge the hint; otherwise, you're likely to just error out.
        #
        # (A previous version of this system had some optimizations to only
        # recompute when it was possible we had learned enough about the
        # unbacked symint that a hint was now possible, but as we added more
        # potential refinements to unbacked symints this got harder to keep
        # in sync, so we've deleted it for now.)

        def compute_hint() -> HintType | SymInt | SymFloat | SymBool:
            from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

            # This occasionally gets exercised by, e.g.,
            # convert_shape_to_symint.  It's just a nicety so you don't HAVE
            # to have a correct hint on hand when making a SymNode.
            # Don't attempt to compute for unbacked, this can be quite
            # expensive.
            if has_free_unbacked_symbols(self.expr):
                return None
            if self.shape_env is None:
                raise RuntimeError("shape_env is required to compute hint")
            hint = self.shape_env._maybe_evaluate_static(self.expr, compute_hint=True)
            if hint is not None:
                hint = self.pytype(hint) if not isinstance(hint, SymTypes) else hint
            return hint

        if hint is _NO_HINT:
            # Caller explicitly indicates hint is unavailable, don't compute
            hint = None
        elif hint is not None:
            if not (type(hint) is pytype or type(hint) is _to_symtype(pytype)):
                raise AssertionError(
                    "Cannot create SymNode of type "
                    f"{pytype} with incompatible hint of type {type(hint)}"
                )
            if self.shape_env and self.shape_env._translation_validation_enabled:
                # This is technically not TV, but this assert is expensive so
                # let's only do it when we're already doing expensive things
                computed_hint = compute_hint()
                if hint != computed_hint:
                    raise AssertionError(f"{hint} != {computed_hint} (for {self.expr})")
        else:
            hint = compute_hint()
        self._hint = hint
        self.constant: int | float | bool | None = constant

        # Record the FX node of the current node if we are doing translation
        # validation. They will be used for building the input assertions for
        # the translation validation problem.
        tx_validation_en = (
            self.shape_env and self.shape_env._translation_validation_enabled
        )
        self.fx_node = tx_validation_en and fx_node

    def with_shape_env(self, shape_env: ShapeEnv) -> SymNode:
        return SymNode(
            self._expr, shape_env, self.pytype, self._hint, self.constant, self.fx_node
        )

    def _value_eq(self, other: SymNode) -> bool:
        # Purposely don't include the shape_env in the eq.
        return (
            self._expr == other._expr  # type: ignore[bad-return]
            and self.pytype == other.pytype
            and self._hint == other._hint
            and self.constant == other.constant
            and self.fx_node == other.fx_node
        )

    def _value_hash(self) -> int:
        # Purposely don't include the shape_env in the hash.
        return hash((self._expr, self.pytype, self._hint, self.constant, self.fx_node))

    @property
    def expr(self) -> sympy.Basic:
        # is_Number is a class attribute, equivalent to isinstance of
        # sympy.Number without needing sympy imported here. Deliberately not
        # is_number, which walks the whole expression tree uncached: _expr is
        # not expected to be an unsimplified non-symbolic expression, and one
        # that slipped through would just take the general path below, where
        # replace() is a no-op on it anyway.
        if (
            isinstance(self._expr, int)
            or self._expr.is_Number  # pyrefly: ignore[missing-attribute]
        ):
            return self._expr
        if self.shape_env is None:
            raise AssertionError("shape_env is required to access expr")
        ver = self.shape_env._replacements_version_counter
        if ver == 0:
            return self._expr
        if self._expr_cache is not None and ver == self._expr_ver:
            return self._expr_cache
        result = self.shape_env.replace(self._expr)
        self._expr_ver = ver
        self._expr_cache = result
        return result

    @property
    def hint(self) -> HintType | SymInt | SymFloat | SymBool:
        return self._hint

    def has_hint(self) -> bool:
        return self._hint is not None

    def maybe_as_int(self) -> int | None:
        if self.expr.is_number:
            return int(self.expr)
        else:
            return None

    # NB: This does conversions, not sure if this is good or not
    def maybe_as_float(self) -> float | None:
        import sympy

        if isinstance(self.expr, sympy.Float):
            return float(self.expr)
        else:
            return None

    def maybe_as_bool(self) -> bool | None:
        import sympy

        if self.expr is sympy.true:
            return True
        elif self.expr is sympy.false:
            return False
        else:
            return None

    def is_int(self) -> bool:
        return self.pytype is int

    def is_float(self) -> bool:
        return self.pytype is float

    def is_bool(self) -> bool:
        return self.pytype is bool

    def is_nested_int(self) -> bool:
        # Unbacked SymInts cannot be nested int today
        return (
            self._hint is not None
            and isinstance(self._hint, SymInt)
            and self._hint.node.is_nested_int()
        )

    def wrap_int(self, num: int) -> SymNode:
        if type(num) is not int:
            raise AssertionError(f"Expected int, got {type(num)}")
        import sympy

        return SymNode(
            sympy.Integer(num), self.shape_env, int, num, constant=num, fx_node=num
        )

    def wrap_float(self, num: float) -> SymNode:
        if type(num) is not float:
            raise AssertionError(f"Expected float, got {type(num)}")
        import sympy

        return SymNode(
            sympy.Float(num), self.shape_env, float, num, constant=num, fx_node=num
        )

    def wrap_bool(self, num: bool) -> SymNode:
        if type(num) is not bool:
            raise AssertionError(f"Expected bool, got {type(num)}")
        import sympy

        return SymNode(
            sympy.true if num else sympy.false,
            self.shape_env,
            bool,
            num,
            constant=num,
            fx_node=num,
        )

    def clone(self) -> SymNode:
        return self

    def str(self) -> builtins.str:
        return f"{self.expr}"

    def __str__(self) -> builtins.str:
        return self.str()

    def __repr__(self) -> builtins.str:
        rep = [
            f"SymNode({self._expr}, shape_env={self.shape_env}, pytype={self.pytype}",
        ]
        if self._hint is not None:
            rep.append(f"hint={self._hint}")
        if self.constant is not None:
            rep.append(f"constant={self.constant}")
        if self.fx_node is not None:
            rep.append(f"fx_node={self.fx_node}")
        return ", ".join(rep) + ")"

    def _graph_repr(self) -> builtins.str:
        # Representation used by GraphModule to create a pythonic version of a graph
        return self.str()

    # These methods call the metaprogrammed methods, they're hand written
    # here so we get good stack traces
    def abs(self) -> SymNode:
        return self._abs()  # type: ignore[attr-defined]

    def pos(self) -> SymNode:
        return self._pos()  # type: ignore[attr-defined]

    def round(self, ndigits: int | None = None) -> SymNode:
        return self._round(ndigits)  # type: ignore[attr-defined]

    def trunc(self) -> SymNode:
        return self._trunc()  # type: ignore[attr-defined]

    def add(self, other: SymNode) -> SymNode:
        return self._add(other)  # type: ignore[attr-defined]

    def sub(self, other: SymNode) -> SymNode:
        return self._sub(other)  # type: ignore[attr-defined]

    def mul(self, other: SymNode) -> SymNode:
        return self._mul(other)  # type: ignore[attr-defined]

    def mod(self, other: SymNode) -> SymNode:
        return self._mod(other)  # type: ignore[attr-defined]

    def float_pow(self, other: SymNode) -> SymNode:
        return self._float_pow(other)  # type: ignore[attr-defined]

    def pow_by_natural(self, other: SymNode) -> SymNode:
        return self._pow_by_natural(other)  # type: ignore[attr-defined]

    def and_(self, other: SymNode) -> SymNode:
        return self._and_(other)  # type: ignore[attr-defined]

    def or_(self, other: SymNode) -> SymNode:
        return self._or_(other)  # type: ignore[attr-defined]

    def xor(self, other: SymNode) -> SymNode:
        return self._xor(other)  # type: ignore[attr-defined]

    def float_truediv(self, other: SymNode) -> SymNode:
        return self._float_truediv(other)  # type: ignore[attr-defined]

    def int_truediv(self, other: SymNode) -> SymNode:
        return self._int_truediv(other)  # type: ignore[attr-defined]

    def int_floordiv(self, other: SymNode) -> SymNode:
        return self._int_floordiv(other)  # type: ignore[attr-defined]

    def lshift(self, other: SymNode) -> SymNode:
        return self._lshift(other)  # type: ignore[attr-defined]

    def rshift(self, other: SymNode) -> SymNode:
        return self._rshift(other)  # type: ignore[attr-defined]

    def sym_not(self) -> SymNode:
        return self._sym_not()  # type: ignore[attr-defined]

    def eq(self, other: SymNode) -> SymNode:
        return self._eq(other)  # type: ignore[attr-defined]

    def ne(self, other: SymNode) -> SymNode:
        return self._ne(other)  # type: ignore[attr-defined]

    def gt(self, other: SymNode) -> SymNode:
        return self._gt(other)  # type: ignore[attr-defined]

    def lt(self, other: SymNode) -> SymNode:
        return self._lt(other)  # type: ignore[attr-defined]

    def le(self, other: SymNode) -> SymNode:
        return self._le(other)  # type: ignore[attr-defined]

    def ge(self, other: SymNode) -> SymNode:
        return self._ge(other)  # type: ignore[attr-defined]

    def floor(self) -> SymNode:
        return self._floor()  # type: ignore[attr-defined]

    def is_integer(self) -> SymNode:
        return self._is_integer()  # type: ignore[attr-defined]

    def sym_float(self) -> SymNode:
        return self._sym_float()  # type: ignore[attr-defined]

    def sym_int(self) -> SymNode:
        return self._sym_int()  # type: ignore[attr-defined]

    def ceil(self) -> SymNode:
        return self._ceil()  # type: ignore[attr-defined]

    def neg(self) -> SymNode:
        return self._neg()  # type: ignore[attr-defined]

    def sym_min(self, other: SymNode) -> SymNode:
        return self._sym_min(other)  # type: ignore[attr-defined]

    def sym_max(self, other: SymNode) -> SymNode:
        return self._sym_max(other)  # type: ignore[attr-defined]

    def sym_ite(self, then_val: SymNode, else_val: SymNode) -> SymNode:
        return self._sym_ite(then_val, else_val)  # type: ignore[attr-defined]

    def is_contiguous(self, sizes: list[SymNode], strides: list[SymNode]) -> SymNode:
        return self._is_contiguous(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_contiguous_2d(
        self, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymNode:
        return self._is_channels_last_contiguous_2d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_contiguous_3d(
        self, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymNode:
        return self._is_channels_last_contiguous_3d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_strides_2d(
        self, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymNode:
        return self._is_channels_last_strides_2d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_strides_3d(
        self, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymNode:
        return self._is_channels_last_strides_3d(sizes, strides)  # type: ignore[attr-defined]

    def is_non_overlapping_and_dense_indicator(
        self, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymNode:
        return self._is_non_overlapping_and_dense_indicator(sizes, strides)  # type: ignore[attr-defined]

    # Make C++ happy
    def sym_or(self, other: SymNode) -> SymNode:
        return self.or_(other)

    def sym_and(self, other: SymNode) -> SymNode:
        return self.and_(other)

    # Integer bitwise ops
    def bitwise_and(self, other: SymNode) -> SymNode:
        return self._bitwise_and(other)  # type: ignore[attr-defined]

    def bitwise_or(self, other: SymNode) -> SymNode:
        return self._bitwise_or(other)  # type: ignore[attr-defined]

    def bitwise_xor(self, other: SymNode) -> SymNode:
        return self._bitwise_xor(other)  # type: ignore[attr-defined]

    # There is no int_truediv available from C++
    def truediv(self, other: SymNode) -> SymNode:
        return self.float_truediv(other)

    def floordiv(self, other: SymNode) -> SymNode:
        return self.int_floordiv(other)

    # We didn't bind integer pow in C++
    def pow(self, other: SymNode) -> SymNode:
        return self.float_pow(other)

    def is_non_overlapping_and_dense(
        self, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymNode:
        return self.is_non_overlapping_and_dense_indicator(sizes, strides).eq(
            to_node(self, 1)
        )  # type: ignore[attr-defined]

    def int_(self) -> int:
        return self.guard_int("", 0)  # NB: uses Python backtrace

    # This one is currently done by hand, but if we add other variadic
    # functions consider factoring it out to be metaprogrammed too.  Note that
    # some load bearing logic is directly in torch.sym_sum

    def sym_sum(self, args: list[SymNode]) -> SymNode:
        import sympy

        # Inner impl
        from torch.fx.experimental.proxy_tensor import (
            get_proxy_mode,
            handle_sym_dispatch,
        )

        if get_proxy_mode():
            return to_node(
                self,
                handle_sym_dispatch(
                    torch.sym_sum,
                    (tuple(wrap_node(a) for a in args),),
                    {},
                ),
            )
        exprs = [a.expr for a in args]
        out = sympy.Add(*exprs)

        size_hints = []
        out_hint: object = _NO_HINT
        for a in args:
            if a.hint is None:
                break
            size_hints.append(a.hint)
        else:
            out_hint = sum(size_hints)  # pyrefly: ignore[no-matching-overload]

        if self.shape_env is None:
            raise RuntimeError("shape_env is required for sym_sum")
        fx_node, _ = self.shape_env._create_fx_call_function(
            torch.sym_sum, (tuple(a.fx_node for a in args),)
        )

        # NB: Only for integers!
        return SymNode(out, self.shape_env, int, out_hint, fx_node=fx_node)

    def evaluate(self, size_oblivious: bool = False) -> bool | int | float:
        if self.shape_env is None:
            raise RuntimeError("shape_env is required to evaluate")
        return self.shape_env.evaluate_sym_node(self, size_oblivious)

    # You can manually trigger a guard with this function
    def guard_int(self, file: builtins.str, line: int) -> int:
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        r = self.evaluate()
        try:
            return int(r)
        except Exception:
            log.warning("Failed to convert to int: %s", r)
            raise

    def guard_float(self, file: builtins.str, line: int) -> float:
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        r = self.evaluate()
        try:
            return float(r)
        except Exception:
            log.warning("Failed to convert to float: %s", r)
            raise

    def guard_bool(self, file: builtins.str, line: int) -> bool:
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        r = self.evaluate()
        try:
            return bool(r)
        except Exception:
            log.warning("Failed to convert to bool: %s", r)
            raise

    def expect_true(self, file: builtins.str, line: int) -> bool:
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        if self.shape_env is None:
            raise RuntimeError("shape_env is required for expect_true")
        if (
            self.has_hint()
            and not free_unbacked_symbols(self.expr)
            and not self.shape_env.prefer_deferred_runtime_asserts_over_guards
        ):
            # OK to generate guards
            return self.guard_bool(file, line)
        # Generate a deferred runtime assert (this might actually end up doing
        # a regular guard if we can!)
        # TODO: file/line here is very important, because the assert has been
        # deferred so you can't backtrace easily
        return self.shape_env.guard_or_defer_runtime_assert(
            self.expr, f"{file}:{line}", fx_node=self.fx_node
        )

    def statically_known_true(self, file: builtins.str, line: int) -> bool:
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        if not self.is_bool():
            raise AssertionError("Expected bool type")
        return statically_known_true(SymBool(self))

    def guard_size_oblivious(self, file: builtins.str, line: int) -> bool:
        """
        Like guard_bool, but if we encounter unbacked symbols, if those symbols
        are size-like, we will treat them as >= 2 for the purposes of the analysis.

        This CHANGES the runtime semantics, but all size-oblivious sites have been
        audited to ensure that the runtime semantics don't change in a material way.
        Acceptable runtime semantic changes are, e.g., squeeze() no longer dropping
        an unbacked one size, or a tensor reporting as non-contiguous even if it's
        contiguous if it would have been reported contiguous due to being empty.
        """
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        r = self.evaluate(size_oblivious=True)
        try:
            return bool(r)
        except Exception:
            log.warning("Failed to convert to bool: %s", r)
            raise

    def guard_or_false(self, file: builtins.str, line: int) -> bool:
        from torch.fx.experimental.symbolic_shapes import guard_or_false

        if not self.is_bool():
            raise AssertionError("Expected bool type")
        return guard_or_false(SymBool(self))

    def guard_or_true(self, file: builtins.str, line: int) -> bool:
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        if not self.is_bool():
            raise AssertionError("Expected bool type")
        return guard_or_true(SymBool(self))

    def bool_(self) -> bool:
        return self.guard_bool("", 0)

    def is_symbolic(self) -> bool:
        return True

    def nested_int(self) -> None:
        return None

    def is_constant(self) -> bool:
        return False


class _DynamicScalar:
    def __new__(cls, *args: object) -> Self:
        if cls is _DynamicScalar:
            raise TypeError("_DynamicScalar is an abstract base class, use DynamicInt.")
        return super().__new__(cls, *args)


class DynamicInt(_DynamicScalar, int):
    """
    User API for marking dynamic integers in `torch.compile`.
    Intended to be compatible with both compile and eager mode.

    Example usage::

        fn = torch.compile(f)
        x = DynamicInt(4)
        fn(x)  # compiles x as a dynamic integer input; returns f(4)
    """

    def __new__(cls, val: int) -> Self:
        if not isinstance(val, int):
            raise AssertionError(f"Expected int, got {type(val)}")
        obj = super().__new__(cls, int(val))
        return obj

    def __repr__(self) -> str:
        return f"DynamicInt({self.real})"

    def __floordiv__(
        self, other: int
    ) -> DynamicInt:  # // was casting to int without these overrides?
        return DynamicInt(self.real // other)

    def __rfloordiv__(self, other: int) -> DynamicInt:
        return DynamicInt(other // self.real)

    def __pow__(  # pyrefly: ignore[bad-override]
        self, other: int, modulo: int | None = None
    ) -> DynamicInt | float:
        if modulo is not None:
            result = pow(self.real, other, modulo)
        else:
            result = self.real**other
        # Only create DynamicInt if result is int, otherwise return plain value
        # (e.g., negative exponent produces float)
        if isinstance(result, int):
            return DynamicInt(result)
        return result

    def __rpow__(self, other: int, modulo: int | None = None) -> DynamicInt | float:
        if modulo is not None:
            result = pow(other, self.real, modulo)
        else:
            result = other**self.real
        # Only create DynamicInt if result is int, otherwise return plain value
        if isinstance(result, int):
            return DynamicInt(result)
        return result


# TODO: this probably needs the sizes-strides eval functions
METHOD_TO_OPERATOR = {
    "pos": operator.pos,
    "abs": operator.abs,
    "add": operator.add,
    "and": operator.and_,
    "bitwise_and": operator.and_,
    "ceil": math.ceil,
    "eq": operator.eq,
    "floor": math.floor,
    "trunc": math.trunc,
    "int_floordiv": operator.floordiv,
    "ge": operator.ge,
    "gt": operator.gt,
    "is_integer": lambda x: x.is_integer(),
    "le": operator.le,
    "lshift": operator.lshift,
    "lt": operator.lt,
    "mod": operator.mod,
    "mul": operator.mul,
    "ne": operator.ne,
    "neg": operator.neg,
    "or": operator.or_,
    "bitwise_or": operator.or_,
    "bitwise_xor": operator.xor,
    "float_pow": operator.pow,
    "pow_by_natural": operator.pow,
    "round": builtins.round,
    "rshift": operator.rshift,
    "sub": operator.sub,
    "sym_float": sym_float,
    "sym_ite": sym_ite,
    "sym_max": sym_max,
    "sym_min": sym_min,
    "sym_not": sym_not,
    "float_truediv": operator.truediv,
    "int_truediv": operator.truediv,
    "xor": operator.xor,
}

unary_magic_methods = {
    "abs",
    "sym_float",
    "sym_int",
    "ceil",
    "floor",
    "neg",
    "sym_not",
    "pos",
    "trunc",
}


# Adding math ops: sqrt, cos, sin, ...
def _get_sym_node_fn(name: str) -> Callable[[SymNode], SymNode]:
    def fn(self: SymNode) -> SymNode:
        return getattr(self, f"_sym_{name}")()

    return fn


math_op_names = (
    "sqrt",
    "cos",
    "cosh",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "asin",
    "acos",
    "atan",
    "log2",
)
for name in math_op_names:
    sym_name = f"sym_{name}"
    priv_sym_name = f"_{sym_name}"
    setattr(SymNode, sym_name, _get_sym_node_fn(name))
    METHOD_TO_OPERATOR[sym_name] = getattr(torch, priv_sym_name)
    unary_magic_methods.add(sym_name)
    # pyrefly: ignore [unresolvable-dunder-all]
    __all__.append(sym_name)


# Unary methods that are not magic methods
unary_nonmagic_methods = {
    "is_integer",
}

unary_methods = unary_magic_methods | unary_nonmagic_methods

# Most methods are only registered on SymInt and SymFloat
# Some methods are only registered on SymBool
only_bool_magic_methods = {"and", "or", "xor", "sym_not", "sym_ite"}
# Methods that implicitly convert SymBool into SymInt
bool_becomes_int_magic_methods = {"add", "sub", "mul"}
# Methods that are also on SymBool, in addition to on SymInt and SymFloat.
# Only equality (eq/ne) is included: ordering comparisons (ge/gt/le/lt) can't be
# evaluated on sympy Booleans, and SymBool ordering isn't supported.
also_bool_magic_methods = {"eq", "ne"}
bool_magic_methods = only_bool_magic_methods | also_bool_magic_methods

# Methods that are only for float
only_float_magic_methods = {"is_integer", "round", "sym_int", "sym_log2"}


magic_methods_on_operator_with_trailing_underscore = {"and", "or"}
# remap necessary because an op name can have a bitwise and boolean implementation
bitwise_ops = {"bitwise_and": "and", "bitwise_or": "or", "bitwise_xor": "xor"}


always_float_magic_methods = {"int_truediv", "float_truediv", "sym_float", "float_pow"}

for name in math_op_names:
    sym_name = f"sym_{name}"
    always_float_magic_methods.add(sym_name)


always_int_magic_methods = {"ceil", "floor", "trunc", "pow_by_natural"}
always_bool_magic_methods = {
    "eq",
    "ne",
    "gt",
    "lt",
    "le",
    "ge",
    "and",
    "or",
    "xor",
    "sym_not",
    "is_non_overlapping_and_dense",
    "is_integer",
}

# Methods that have a `__foo__` as well as `__rfoo__`


def _sympy_float_truediv(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import FloatTrueDiv

    return FloatTrueDiv(a, b)


def _sympy_int_truediv(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import IntTrueDiv

    return IntTrueDiv(a, b)


def _sympy_floordiv(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import FloorDiv

    return FloorDiv(a, b)


def _sympy_mod(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import Mod, PythonMod

    if a.is_nonnegative and b.is_nonnegative:
        return Mod(a, b)
    else:
        return PythonMod(a, b)


def _sympy_pow_by_natural(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import PowByNatural

    return PowByNatural(a, b)


def _sympy_float_pow(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import FloatPow

    return FloatPow(a, b)


def _sympy_and(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    import sympy

    return sympy.And(a, b)


def _sympy_or(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    import sympy

    return sympy.Or(a, b)


def _sympy_xor(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    import sympy

    return sympy.Xor(a, b)


def _sympy_lshift(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import LShift

    return LShift(a, b)


def _sympy_rshift(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import RShift

    return RShift(a, b)


def _binary_search_insert_arg(
    ordered_args: list[sympy.Basic], new_arg: sympy.Basic
) -> list[sympy.Basic] | None:
    """
    If new_arg is found in ordered_args None is returned, else the new
    ordered_args with new_arg inserted
    """
    if len(ordered_args) == 0:
        return [new_arg]

    from sympy.core.basic import _args_sortkey as sort_key, Basic

    # Fast path when new_arg > ordered_args[-1].
    if sort_key(ordered_args[-1]) < sort_key(new_arg):
        return ordered_args + [new_arg]

    # Fast path when new_arg < ordered_args[0].
    if sort_key(ordered_args[0]) > sort_key(new_arg):
        return [new_arg] + ordered_args

    low, high = 0, len(ordered_args) - 1

    while low <= high:
        mid = (low + high) // 2
        compare_result = Basic.compare(ordered_args[mid], new_arg)
        if compare_result == 0:
            return None
        elif compare_result < 0:
            low = mid + 1
        else:
            high = mid - 1

    ordered_args.insert(low, new_arg)
    return ordered_args


def _optimized_add(
    lhs: sympy.Basic,
    rhs: sympy.Basic,
    lhs_is_optimized_summation: bool = False,
    rhs_is_optimized_summation: bool = False,
) -> tuple[bool, sympy.Basic]:
    """
    Custom optimization for Add used to optimize incremental binary summations of certain properties. The idea
    is when we know the expression is a summation of unique symbols all we need to know is the correct order of symbols,
    and no other optimizations are needed. We pass evaluate=false, with the correct order of args and save the following.
    1. Avoid running other optimizations when the Add is constructed.
    2. Manually figure out the order of the args for the new expression in log(n) comparisons instead of nLog(n)
    (comparing terms is expensive and shows in the profiles).
    The function returns a tuple of (1) a boolean that indicates whether the output is a summation of unique symbols,
    (2) the result sympy expression.
    """
    import sympy
    from sympy.core.basic import _args_sortkey as sortkey

    def make_optimized(ordered_args: list[sympy.Basic]) -> tuple[bool, sympy.Basic]:
        if ordered_args is None:
            raise AssertionError("ordered_args is None")
        # Use _from_args directly to bypass _exec_constructor_postprocessors
        # which iterates over all args. This is safe because args are only
        # symbols or constants, which don't register postprocessors.
        # Pass is_commutative=True to avoid fuzzy_and check over all args.
        result = sympy.Add._from_args(ordered_args, is_commutative=True)
        return (True, result)

    from torch.utils._sympy.functions import _is_symbols_binary_summation

    lhs_is_optimized_summation |= _is_symbols_binary_summation(lhs)
    rhs_is_optimized_summation |= _is_symbols_binary_summation(rhs)

    if lhs_is_optimized_summation and rhs_is_optimized_summation:
        # (a0+a1..) + (a2+a3..) => (a0+a1+a2+a3)
        if sortkey(lhs._args[-1]) < sortkey(rhs._args[0]):
            return make_optimized(lhs._args + rhs._args)
        #  (a2+a3..) + (a0+a1..) => (a0+a1+a2+a3)
        if sortkey(lhs._args[0]) > sortkey(rhs._args[-1]):
            return make_optimized(rhs._args + lhs._args)

        #  (a1+a3) + (a0+a2) => (a0+a1+a2+a3)
        if len(lhs._args) <= 2 and len(rhs._args) <= 2:
            new_args = list(lhs._args)
            for a in rhs._args:
                new_args = _binary_search_insert_arg(new_args, a)
                if new_args is None:
                    break
            # None means an element already exists.
            if new_args is not None:
                return make_optimized(new_args)

    # (a0+a2) + a1 => (a0+a1+a2)
    if lhs_is_optimized_summation and rhs.is_symbol:
        new_args = _binary_search_insert_arg(list(lhs._args), rhs)
        # None means an element already exists.
        if new_args is not None:
            return make_optimized(new_args)

    # a1 + (a0+a2)=> (a0+a1+a2)
    if rhs_is_optimized_summation and lhs.is_symbol:
        new_args = _binary_search_insert_arg(list(rhs._args), lhs)
        # None means an element already exists.
        if new_args is not None:
            return make_optimized(new_args)

    result = sympy.Add(lhs, rhs)
    return (_is_symbols_binary_summation(result), result)


def _bitwise_and(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import BitwiseFn_bitwise_and

    return BitwiseFn_bitwise_and(a, b)


def _bitwise_or(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import BitwiseFn_bitwise_or

    return BitwiseFn_bitwise_or(a, b)


def _bitwise_xor(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import BitwiseFn_bitwise_xor

    return BitwiseFn_bitwise_xor(a, b)


reflectable_magic_methods = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "mod": _sympy_mod,
    "pow_by_natural": _sympy_pow_by_natural,
    "float_pow": _sympy_float_pow,
    "and": _sympy_and,
    "bitwise_and": _bitwise_and,
    "or": _sympy_or,
    "xor": _sympy_xor,
    "bitwise_or": _bitwise_or,
    "bitwise_xor": _bitwise_xor,
    "float_truediv": _sympy_float_truediv,
    "int_truediv": _sympy_int_truediv,
    "int_floordiv": _sympy_floordiv,
    "lshift": _sympy_lshift,
    "rshift": _sympy_rshift,
}


def _floor_ceil_helper(a: sympy.Basic, fn: Callable[..., sympy.Basic]) -> sympy.Basic:
    import sympy

    if isinstance(a, sympy.Mul):
        aa = a.args
        if len(aa) == 2 and isinstance(aa[0], sympy.Float) and aa[1].is_integer:
            coef = sympy.Integer(aa[0])
            if aa[0] == coef:  # structural equality test
                return coef * aa[1]
    if (
        isinstance(a, sympy.Float)
        and a == sympy.Integer(a)
        or isinstance(a, sympy.Integer)
    ):
        return sympy.Integer(a)
    return fn(a)


def _sympy_floor(a: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import FloorToInt

    return FloorToInt(a)


# NB: this is Python trunc semantics which returns an int.  Do NOT use this to
# represent torch.trunc (which is float to float)
def _sympy_trunc(a: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import TruncToInt

    return TruncToInt(a)


def _sympy_ceil(a: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import CeilToInt

    return CeilToInt(a)


def _sympy_eq(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    import sympy

    return sympy.Eq(a, b)


def _sympy_ne(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    import sympy

    return sympy.Ne(a, b)


def _sympy_gt(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    import sympy

    return sympy.Gt(a, b)


def _sympy_lt(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    import sympy

    return sympy.Lt(a, b)


def _sympy_le(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    import sympy

    return sympy.Le(a, b)


def _sympy_ge(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    import sympy

    return sympy.Ge(a, b)


def _sympy_min(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import Min

    return Min(a, b)


def _sympy_max(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import Max

    return Max(a, b)


def _sympy_ite(a: sympy.Basic, t: sympy.Basic, f: sympy.Basic) -> sympy.Basic:
    import sympy

    return sympy.Piecewise((t, a), (f, True))


current_module = sys.modules[__name__]


def _get_sym_math_fn(name: str) -> Callable[[sympy.Basic], sympy.Basic]:
    def fn(a: sympy.Basic) -> sympy.Basic:
        import torch.utils._sympy.functions

        return getattr(torch.utils._sympy.functions, f"OpaqueUnaryFn_{name}")(a)

    return fn


for name in math_op_names:
    priv_sympy_name = f"_sympy_{name}"
    fn = _get_sym_math_fn(name)
    fn.__qualname__ = fn.__name__ = priv_sympy_name
    setattr(current_module, priv_sympy_name, fn)

del fn, name, priv_sympy_name  # type: ignore[possibly-undefined]


def _sympy_abs(a: sympy.Basic) -> sympy.Basic:
    import sympy

    return sympy.Abs(a)


def _sympy_round(
    number: sympy.Basic, ndigits: sympy.Basic | None = None
) -> sympy.Basic:
    from torch.utils._sympy.functions import RoundDecimal, RoundToInt

    if ndigits is None:
        return RoundToInt(number)
    else:
        return RoundDecimal(number, ndigits)


def _sympy_sym_float(a: sympy.Basic) -> sympy.Basic:
    from torch.utils._sympy.functions import ToFloat

    # NB: Cannot use a * 1.0 here, because 0 * 1.0 is 0 which incorrectly
    # reports that it is an integer
    return ToFloat(a)


def _sympy_is_integer(a: sympy.Basic) -> sympy.Basic:
    import sympy

    from torch.utils._sympy.functions import ToFloat

    return sympy.Eq(ToFloat(sympy.floor(a)), a)


magic_methods = {
    **reflectable_magic_methods,
    "sym_not": operator.invert,
    "pos": operator.pos,
    "eq": _sympy_eq,
    "ne": _sympy_ne,
    "gt": _sympy_gt,
    "lt": _sympy_lt,
    "le": _sympy_le,
    "ge": _sympy_ge,
    "floor": _sympy_floor,
    "trunc": _sympy_trunc,
    "sym_float": _sympy_sym_float,
    "ceil": _sympy_ceil,
    "neg": operator.neg,
    "sym_min": _sympy_min,
    "sym_max": _sympy_max,
    "sym_ite": _sympy_ite,
    "abs": _sympy_abs,
    "round": _sympy_round,
    "is_integer": _sympy_is_integer,
}


for name in math_op_names:
    sym_name = f"sym_{name}"
    magic_methods[sym_name] = getattr(current_module, f"_sympy_{name}")

del name, sym_name, math_op_names, current_module  # type: ignore[possibly-undefined]


def sympy_is_contiguous(
    sizes: list[sympy.Basic], strides: list[sympy.Basic]
) -> sympy.Basic:
    dim = len(sizes)
    return sympy_is_contiguous_generic(sizes, strides, list(range(dim - 1, -1, -1)))


def sympy_is_contiguous_generic(
    sizes: list[sympy.Basic], strides: list[sympy.Basic], dim_order: list[int]
) -> sympy.Basic:
    import sympy

    dim = len(sizes)

    if len(dim_order) != dim:
        return sympy.false

    is_contiguous = sympy.true
    z = sympy.S.One
    # Contiguous if the strides make sense (or the dim is size 1)
    for d in dim_order:
        is_contiguous &= sympy.Eq(sizes[d], sympy.S.One) | sympy.Eq(strides[d], z)
        z *= sizes[d]
    # OR if any size is zero
    for d in range(dim):
        is_contiguous |= sympy.Eq(sizes[d], sympy.S.Zero)
    return is_contiguous


# NB: There is a TODO in C++ to allow omitting the batch dim.  If that
# happens you will need to refactor this


def sympy_is_channels_last_contiguous_2d(
    sizes: list[sympy.Basic], strides: list[sympy.Basic]
) -> sympy.Basic:
    return sympy_is_contiguous_generic(sizes, strides, [1, 3, 2, 0])


def sympy_is_channels_last_contiguous_3d(
    sizes: list[sympy.Basic], strides: list[sympy.Basic]
) -> sympy.Basic:
    return sympy_is_contiguous_generic(sizes, strides, [1, 4, 3, 2, 0])


def sympy_is_channels_last_strides_generic(
    sizes: list[sympy.Basic], strides: list[sympy.Basic], dim_order: list[int]
) -> sympy.Basic:
    import sympy

    from torch.utils._sympy.functions import Max

    dim = len(sizes)

    if dim != len(dim_order):
        return sympy.false

    m = sympy.S.Zero
    r = sympy.true

    # special case for trivial C dimension. default to NCHW
    r &= sympy.Ne(strides[1], 0)

    for d in dim_order:
        r &= sympy.Ne(sizes[d], 0) & (strides[d] >= m)
        # Fallback to NCHW as default layout for ambiguous cases
        # This is the flaw of implicit memory_format from strides.
        # N111 tensor with identical strides for size 1 dimension;
        # Two cases could lead us here:
        # a. N111 contiguous Tensor ([N,1,1,1]@[1,1,1,1])
        # b. N11W contiguous Tensor sliced on the W-dimension.
        # ([N,1,1,1]@[W,W,W,W])
        if d == 0:
            r &= sympy.Ne(m, strides[1])
        # This is necessary to:
        # 1. distinguish the memory_format of N1H1;
        #     [H, 1, 1, 1] channels_last stride
        #     [H, H, 1, 1] contiguous stride
        # 2. permutation of 1C1W:
        #     [1, C, 1, H]@[HC, H, H, 1] transpose(1, 3)
        #     [1, H, 1, C]@[HC, 1, H, H] shouldn't be identified as
        #     channels_last
        m = strides[d] * Max(sizes[d], 1)

    return r


def sympy_is_channels_last_strides_2d(
    sizes: list[sympy.Basic], strides: list[sympy.Basic]
) -> sympy.Basic:
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 3, 2, 0])


def sympy_is_channels_last_strides_3d(
    sizes: list[sympy.Basic], strides: list[sympy.Basic]
) -> sympy.Basic:
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 4, 3, 2, 0])


def _sympy_is_non_overlapping_and_dense_indicator(
    sizes: list[sympy.Basic], strides: list[sympy.Basic]
) -> sympy.Basic:
    from torch.utils._sympy.functions import IsNonOverlappingAndDenseIndicator

    return IsNonOverlappingAndDenseIndicator(*sizes, *strides)


sizes_strides_methods = {
    # TODO: These could also be done with indicators, maybe it is better
    # for reasoning to do it that way
    "is_contiguous": sympy_is_contiguous,
    "is_channels_last_contiguous_2d": sympy_is_channels_last_contiguous_2d,
    "is_channels_last_contiguous_3d": sympy_is_channels_last_contiguous_3d,
    "is_channels_last_strides_2d": sympy_is_channels_last_strides_2d,
    "is_channels_last_strides_3d": sympy_is_channels_last_strides_3d,
    "is_non_overlapping_and_dense_indicator": _sympy_is_non_overlapping_and_dense_indicator,
}


def to_node(self: SymNode, num: object) -> SymNode:
    if isinstance(num, SymTypes):
        return num.node
    elif type(num) is bool:
        return self.wrap_bool(num)
    elif type(num) is int:
        return self.wrap_int(num)
    elif type(num) is float:
        return self.wrap_float(num)
    else:
        # NotImplemented is important so that Python tries the
        # other magic method
        return NotImplemented


def wrap_node(x: SymNode) -> SymInt | SymFloat | SymBool | int | float | bool:
    # TODO: let C++ also take advantage of this
    if isinstance(x, SymNode) and x.constant is not None:
        return x.constant
    if x.is_int():
        return SymInt(x)
    elif x.is_float():
        return SymFloat(x)
    elif x.is_bool():
        return SymBool(x)
    else:
        raise AssertionError(f"unrecognized return type {x}")


def method_to_operator(method: str) -> Callable[..., object]:
    return METHOD_TO_OPERATOR[method]


def _make_node_magic(method: str, func: Callable[..., sympy.Basic]) -> None:
    func = lru_cache(256)(func)

    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f"{method}_"
    else:
        method_attr = method

    def uninteresting_files() -> set[str]:
        import torch

        mods = [
            torch._dynamo.eval_frame,
            torch._dynamo.utils,
            torch.fx.experimental.sym_node,
            torch,
        ]
        import torch._dynamo.guards

        return (
            {inspect.getfile(m) for m in mods}
            | torch._dynamo.guards.uninteresting_files()
            | {"<string>"}
        )

    def capture_provenance(fn: Callable[..., SymNode]) -> Callable[..., SymNode]:
        @functools.wraps(fn)
        def wrapper(self: SymNode, other: SymNode | None = None) -> SymNode:
            if other is None:
                result = fn(self)
            else:
                result = fn(self, other)
            if torch._logging._internal.GET_DTRACE_STRUCTURED:
                if other is not None:
                    arguments = [self, other]
                else:
                    arguments = [self]

                def get_id(sym_node: SymNode) -> int | None:
                    # We don't want to return an ID if the input is a constant
                    import sympy

                    if sym_node.constant is not None:
                        return None
                    elif id(sym_node) == id(result):
                        return None
                    elif isinstance(sym_node.expr, (sympy.Integer, sympy.Float)):
                        return None
                    elif sym_node.expr in (sympy.true, sympy.false):
                        return None
                    return id(sym_node)

                dtrace_structured(
                    "expression_created",
                    metadata_fn=lambda: {
                        "method": method,
                        "result": str(result),
                        "result_id": id(result),
                        "arguments": [str(a) for a in arguments],
                        "argument_ids": [
                            get_id(i) for i in arguments if get_id(i) is not None
                        ],
                        "user_stack": structured.get_user_stack(3),
                        "stack": structured.get_framework_stack(3),
                    },
                )

            return result

        return wrapper

    @capture_provenance
    def binary_magic_impl(self: SymNode, other: SymNode) -> SymNode:
        from torch.fx.experimental.proxy_tensor import (
            get_proxy_mode,
            handle_sym_dispatch,
        )

        op = method_to_operator(method)

        out_hint: object = _NO_HINT
        if self.hint is not None and other.hint is not None:
            out_hint = op(self.hint, other.hint)

        if get_proxy_mode():
            return to_node(
                self, handle_sym_dispatch(op, (wrap_node(self), wrap_node(other)), {})
            )
        if not isinstance(other, SymNode):
            raise AssertionError(f"Expected SymNode, got {type(other)}")
        optimized_summation = False
        try:
            if method == "mod":
                from torch.utils._sympy.functions import Mod, PythonMod

                # Special handling for mod that requires access to the value
                # ranges
                shape_env = self.shape_env
                if shape_env is None:
                    raise AssertionError("shape_env is required for mod")
                if (
                    self.expr.is_nonnegative
                    or shape_env.bound_sympy(self.expr).lower >= 0
                ) and (
                    other.expr.is_nonnegative
                    or shape_env.bound_sympy(other.expr).lower >= 0
                ):
                    out = Mod(self.expr, other.expr)
                else:
                    out = PythonMod(self.expr, other.expr)
            elif method == "add":
                # see Note [optimized_summation]
                (optimized_summation, out) = _optimized_add(
                    self.expr,
                    other.expr,
                    self._optimized_summation,
                    other._optimized_summation,
                )
            elif method in ("eq", "ne", "ge", "gt", "le", "lt"):
                import sympy

                from torch.utils._sympy.symbol import symbol_is_type, SymT

                # Optimization: when one side is a single unbacked symbol
                # and other is constant, use evaluate=False to skip expensive
                # relational evaluation. We only do this for unbacked symbols
                # because they have no assumptions (like positive=True) that
                # sympy would use during evaluation.
                lhs_is_unbacked = self.expr.is_symbol and symbol_is_type(
                    self.expr, SymT.UNBACKED_INT
                )
                rhs_is_unbacked = other.expr.is_symbol and symbol_is_type(
                    other.expr, SymT.UNBACKED_INT
                )
                if (lhs_is_unbacked and other.expr.is_number) or (
                    rhs_is_unbacked and self.expr.is_number
                ):
                    rel_class = {
                        "eq": sympy.Eq,
                        "ne": sympy.Ne,
                        "ge": sympy.Ge,
                        "gt": sympy.Gt,
                        "le": sympy.Le,
                        "lt": sympy.Lt,
                    }[method]
                    out = rel_class(self.expr, other.expr, evaluate=False)
                else:
                    out = func(self.expr, other.expr)

            else:
                # TODO: consider constant prop here
                out = func(self.expr, other.expr)
        except Exception:
            log.warning("failed to eval %s(%s, %s)", method, self.expr, other.expr)
            raise
        sym_node_log.debug("%s %s %s -> %s", method, self.expr, other.expr, out)
        pytype: type
        # This is not strictly correct. In Python, a**b may return complex when
        # a < 0 and b is a float: (-1)**2.1. Same for sympy.sqrt(-3.14). This
        # returns a float while both arguments are ints: 2**(-1). Also, max and
        # min do not type promote. To avoid having data-dependent control flow
        # here, we just set the type to float if one of the args is a float. In
        # case of a type mismatch, we assume that it will be detected during
        # evaluation.
        if method in always_float_magic_methods:
            pytype = float
        elif method in always_bool_magic_methods:
            pytype = bool
        elif self.pytype is float or other.pytype is float:
            pytype = float
        else:
            pytype = self.pytype

        if (
            pytype is not None
            and out_hint is not _NO_HINT
            and out_hint is not None
            and not isinstance(out_hint, SymTypes)
        ):
            out_hint = pytype(out_hint)  # type: ignore[arg-type]

        # Create a FX node that corresponds to the operation being applied to
        # this node.
        if self.shape_env is None:
            raise RuntimeError("shape_env is required for binary op")
        fx_node, _ = self.shape_env._create_fx_call_function(
            op, (self.fx_node, other.fx_node)
        )

        result = SymNode(
            out,
            self.shape_env,
            pytype,
            out_hint,  # type: ignore[arg-type]
            fx_node=fx_node,
            optimized_summation=optimized_summation,  # see Note [optimized_summation]
        )
        return result

    @capture_provenance
    def unary_magic_impl(self: SymNode) -> SymNode:
        from torch.fx.experimental.proxy_tensor import (
            get_proxy_mode,
            handle_sym_dispatch,
        )

        op = method_to_operator(method)
        if get_proxy_mode():
            return to_node(self, handle_sym_dispatch(op, (wrap_node(self),), {}))
        # TODO: consider constant prop here
        expr = self.expr
        if self.shape_env is None:
            raise RuntimeError("shape_env is required for unary op")
        if method == "floor" or method == "ceiling":
            expr = self.shape_env._simplify_floor_div(expr)

        try:
            out = func(expr)
        except Exception:
            log.warning("failed to eval %s(%s)", method, expr)
            raise
        sym_node_log.debug("%s %s -> %s", func, expr, out)
        out_hint: object = _NO_HINT
        if self.hint is not None:
            out_hint = op(self.hint)
        pytype: type
        if method in always_int_magic_methods:
            pytype = int
        elif method in always_bool_magic_methods:
            pytype = bool
        elif method in always_float_magic_methods:
            pytype = float
        else:
            pytype = self.pytype

        fx_node, _ = self.shape_env._create_fx_call_function(op, (self.fx_node,))
        return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)  # type: ignore[arg-type]

    if method in unary_methods:
        setattr(SymNode, f"_{method_attr}", unary_magic_impl)
    elif method == "sym_ite":

        def sym_ite_impl(
            pred_node: SymNode, then_node: SymNode, else_node: SymNode
        ) -> SymNode:
            from torch.fx.experimental.proxy_tensor import (
                get_proxy_mode,
                handle_sym_dispatch,
            )

            if pred_node.hint is None:
                out_hint = None
            elif pred_node.hint:
                out_hint = then_node.hint
            else:
                out_hint = else_node.hint
            if get_proxy_mode():
                return to_node(
                    pred_node,
                    handle_sym_dispatch(
                        sym_ite,
                        (
                            wrap_node(pred_node),
                            wrap_node(then_node),
                            wrap_node(else_node),
                        ),
                        {},
                    ),
                )

            try:
                out = func(pred_node.expr, then_node.expr, else_node.expr)
            except Exception:
                log.warning(
                    "failed to eval %s(%s, %s, %s)",
                    method,
                    pred_node.expr,
                    then_node.expr,
                    else_node.expr,
                )
                raise

            if pred_node.shape_env is None:
                raise RuntimeError("shape_env is required for sym_ite")
            fx_node, _ = pred_node.shape_env._create_fx_call_function(
                sym_ite, (pred_node.fx_node, then_node.fx_node, else_node.fx_node)
            )
            return SymNode(
                out, pred_node.shape_env, then_node.pytype, out_hint, fx_node=fx_node
            )

        setattr(SymNode, f"_{method_attr}", sym_ite_impl)
    elif method == "round":

        def round_impl(self: SymNode, ndigits: int | None = None) -> SymNode:
            from torch.fx.experimental.proxy_tensor import (
                get_proxy_mode,
                handle_sym_dispatch,
            )

            op = builtins.round
            if get_proxy_mode():
                return to_node(
                    self, handle_sym_dispatch(op, (wrap_node(self), ndigits), {})
                )

            expr = self.expr
            try:
                out = func(expr, ndigits)
            except Exception:
                log.warning("failed to eval %s(%s, ndigits=%s)", method, expr, ndigits)
                raise

            if ndigits is None:
                pytype = int
            else:
                pytype = self.pytype

            out_hint = None
            if self.hint is not None:
                out_hint = op(  # pyrefly: ignore[no-matching-overload]
                    self.hint, ndigits
                )

            # Internally, None is used as sentinel to indicate that a something is not a node on an FX graph. At the
            # same time, there is no way to wrap a plain None into an FX node. Thus, there is no way to pass None here
            # without triggering some asserts that check whether we are mixing FX nodes with untracked arguments. The
            # hack down below works, because all round function down the line all take ndigits=None as default in their
            # signature.
            # TODO: Remove the args construction below if a different sentinel is used by FX.
            # ezyang(May 2024): LOL
            args = [self.fx_node]
            if ndigits is not None:
                args.append(ndigits)
            if self.shape_env is None:
                raise RuntimeError("shape_env is required for round")
            fx_node, _ = self.shape_env._create_fx_call_function(op, tuple(args))
            return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)

        setattr(SymNode, f"_{method_attr}", round_impl)
    else:
        setattr(SymNode, f"_{method_attr}", binary_magic_impl)


def _make_node_sizes_strides(method: str, func: Callable[..., sympy.Basic]) -> None:
    # NB: don't LRU cache, lots of arguments

    def sizes_strides_impl(
        self: SymNode, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymNode:
        from torch.fx.experimental.proxy_tensor import (
            get_proxy_mode,
            handle_sym_dispatch,
        )

        op = getattr(sys.modules[__name__], method)
        if get_proxy_mode():
            return to_node(
                self,
                handle_sym_dispatch(
                    op,
                    ([wrap_node(s) for s in sizes], [wrap_node(s) for s in strides]),
                    {},
                ),
            )
        size_exprs = [s.expr for s in sizes]
        stride_exprs = [s.expr for s in strides]
        try:
            out = func(size_exprs, stride_exprs)
        except Exception:
            log.warning("failed to eval %s(%s, %s)", method, size_exprs, stride_exprs)
            raise
        # bool is never expandable

        size_hints = []
        out_hint = None
        for s in sizes:
            if s.hint is None:
                break
            size_hints.append(s.hint)
        else:
            stride_hints = []
            for s in strides:
                if s.hint is None:
                    break
                stride_hints.append(s.hint)
            else:
                out_hint = op(size_hints, stride_hints)

        # NB: This is the indicator function, not the actual bool!
        pytype: type
        if method.endswith("_indicator"):
            pytype = int
        else:
            pytype = bool
        return SymNode(out, self.shape_env, pytype, out_hint)

    setattr(SymNode, f"_{method}", sizes_strides_impl)

    # TODO: This is technically hotpath, but in the ideal end state
    # guards on this will resolve at a higher level so you never
    # spend time in this code
    def sizes_strides_user(
        sizes: list[object], strides: list[object]
    ) -> SymInt | SymFloat | SymBool | int | float | bool:
        import sympy

        from torch.fx.experimental.symbolic_shapes import (
            eval_is_non_overlapping_and_dense,
        )

        for a in itertools.chain(sizes, strides):
            if isinstance(a, SymInt):
                return wrap_node(
                    getattr(a.node, method)(
                        [to_node(a.node, b) for b in sizes],
                        [to_node(a.node, b) for b in strides],
                    )
                )
        if method == "is_non_overlapping_and_dense_indicator":
            return eval_is_non_overlapping_and_dense(
                sizes,  # pyrefly: ignore[bad-argument-type]
                strides,  # pyrefly: ignore[bad-argument-type]
            )
        else:
            # TODO: this is an awful implementation
            return bool(
                func(
                    [sympy.sympify(a) for a in sizes],
                    [sympy.sympify(a) for a in strides],
                )
            )

    # Skip for is_non_overlapping_and_dense_indicator
    if not hasattr(sys.modules[__name__], method):
        setattr(sys.modules[__name__], method, sizes_strides_user)


for method, func in magic_methods.items():
    _make_node_magic(method, func)

for method, func in sizes_strides_methods.items():
    _make_node_sizes_strides(method, func)


def _make_user_magic(method: str, user_type: type) -> None:
    # User magic takes care of wrapping the other operand into a node,
    # so that our internal logic can assume everything is nodes
    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f"sym_{method}"
    else:
        method_attr = method

    def get_constant(
        x: SymInt | int | SymFloat | float | SymBool | bool,
    ) -> int | float | bool:
        if isinstance(x, (int, float, bool)):
            return x
        if isinstance(x, SymInt):
            return x.node.guard_int("", 0)
        if isinstance(x, SymBool):
            return x.node.guard_bool("", 0)
        raise AssertionError("expect to be called with constant SymBools")

    def is_constant(x: SymInt | int | SymFloat | float | SymBool | bool) -> bool:
        if isinstance(x, (int, float, bool)):
            return True
        if isinstance(x, (SymInt, SymFloat, SymBool)):
            return x.node.is_constant()
        return False

    # Promotion rules for binary operations.  NB: we preserve PYTHON semantics
    #   - if args are same type, do nothing
    #   - if one arg is float, promote other arg to float
    #       - nb: this applies to floordiv, even though output is integral
    #       (it's still float)
    #   - pow is funny business
    #       - if both ints
    #       - trigger a guard on exponent >= 0
    #           - if non-negative, output is int
    #           - otherwise, output is float
    #   - otherwise, promote other arg to float
    #       - nb: complex is impossible to handle correctly lol, with
    #       negative base and integral float need to diverge semantics and
    #       just always return complex.  Neener neener pretend this problem
    #       doesn't exist
    #   - equality is pain: Python does the fancy thing where it unpacks the
    #     mantissa from the float and then compares that against the int.
    #     Which means it is able to tell that
    #     9007199254740993 != 9007199254740992. (rather than if the LHS was
    #     promoted to float, in which case it would have truncated to the RHS
    #     and subsequently been equal).  We'll model this exactly by having
    #     special mixed type equality operations.  Unfortunately, we need to
    #     do this for all comparison operations (maybe I'll only implement
    #     compare)
    #   - sym_ite mumble mumble really shouldn't allow mixed but whatever

    if method in bool_becomes_int_magic_methods:

        def promote(x: object) -> Any:
            """Implements True+True=2, which works in python but not sympy"""
            if isinstance(x, SymBool):
                return SymInt(x.node.wrap_int(int(x)))
            if type(x) is bool:
                return int(x)
            return x

    else:

        def promote(x: object) -> Any:
            return x

    def promote2(self: object, other: object) -> tuple[Any, Any]:
        # TODO: Remove eq and other relations from this list.
        # CPython has fancy implementations for these to get as much precision
        # as possible instead of just promoting to float64 and praying, so we
        # need to handle them specially too.
        # Also, note that int_truediv doesn't go through this path: both
        # arguments are "int" so there isn't any promotion
        if method not in [
            "add",
            "sub",
            "mul",
            "mod",
            "float_pow",
            "float_truediv",
            "int_floordiv",
            "sym_min",
            "sym_max",
            # TODO: remove these
            "eq",
            "ne",
            "gt",
            "lt",
            "le",
            "ge",
        ]:
            return self, other
        f_self = isinstance(self, (float, torch.SymFloat))
        f_other = isinstance(other, (float, torch.SymFloat))
        if f_self or f_other:
            if not f_self:
                self = torch.sym_float(self)
            if not f_other:
                other = torch.sym_float(other)
        return self, other

    # Before and after performing the operation, check if any operands are constant.
    # If so, extract out the constant values first. If `self` itself is a
    # constant, then "redispatch" by calling back into the operator. Sometimes
    # this means that operations involving SymBool return plain bools.
    # Alternatively, we could also rewrap into constant Symbool (i.e. by
    # implementing wrap_bool in ConstantSymNodeImpl), but we're not doing that
    # today for no particular reason.
    def unary_magic_impl(self: object) -> Any:
        self = promote(self)
        if is_constant(self):
            return (method_to_operator(method))(get_constant(self))
        return wrap_node(getattr(self.node, method_attr)())

    def binary_magic_impl(self: object, other: object) -> Any:
        if not isinstance(other, (int, float, bool, SymInt, SymFloat, SymBool)):
            return NotImplemented
        sym_node_log.debug("MAGIC %s %s %s", method, self, other)
        self = promote(self)
        other = promote(other)
        self, other = promote2(self, other)
        if is_constant(self):
            return (method_to_operator(method))(get_constant(self), other)
        if is_constant(other):
            other = get_constant(other)
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        ret = wrap_node(getattr(self.node, method_attr)(other_node))
        return get_constant(ret) if is_constant(ret) else ret

    def rbinary_magic_impl(self: object, other: object) -> Any:
        if not isinstance(other, (int, float, bool, SymInt, SymFloat, SymBool)):
            return NotImplemented
        self = promote(self)
        other = promote(other)
        self, other = promote2(self, other)
        if is_constant(self):
            return (method_to_operator(method))(other, get_constant(self))
        if is_constant(other):
            other = get_constant(other)
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        ret = wrap_node(getattr(other_node, method_attr)(self.node))
        return get_constant(ret) if is_constant(ret) else ret

    def setattrs(user_type: type, attr: str, symnode_impl: object) -> None:
        """
        Registers the SymNode magic method on SymInt/Float/Bool,
        and optionally registers a corresponding wrapped method on DynamicInt.
        """

        # SymInt/Float/Bool
        setattr(user_type, attr, symnode_impl)

        # DynamicInt impl
        def dynamic_int_impl(*args: object) -> Any:
            args = [  # pyrefly: ignore[bad-assignment]
                x.real if isinstance(x, DynamicInt) else x for x in args
            ]
            out = getattr(int, attr)(*args)
            if isinstance(out, int) and not isinstance(out, bool):
                return DynamicInt(out)
            return out

        if user_type is SymInt:
            setattr(DynamicInt, attr, dynamic_int_impl)

    if method in unary_magic_methods:
        setattrs(user_type, f"__{method}__", unary_magic_impl)
    elif method in unary_nonmagic_methods:
        orig = getattr(user_type, method)
        setattrs(user_type, method, update_wrapper(unary_magic_impl, orig))
    elif method == "sym_ite":

        def sym_ite_magic_impl(
            pred: SymBool, then_val: object, else_val: object
        ) -> Any:
            pred_node = pred.node
            then_node = to_node(pred_node, then_val)
            else_node = to_node(pred_node, else_val)
            if then_node is NotImplemented or else_node is NotImplemented:
                return NotImplemented
            if not (
                isinstance(then_node, SymNode)
                and isinstance(else_node, SymNode)
                and then_node.pytype == else_node.pytype
            ):
                raise AssertionError(
                    "then_node and else_node must be SymNodes with same pytype"
                )
            ret = wrap_node(getattr(pred.node, method_attr)(then_node, else_node))
            return (
                get_constant(ret)
                # pyrefly: ignore[missing-attribute]
                if ret.node.is_constant()
                else ret
            )

        setattrs(user_type, f"__{method}__", sym_ite_magic_impl)
    elif method == "round":

        def round_magic_impl(self: SymFloat, ndigits: int | None = None) -> Any:
            if is_constant(self):
                return builtins.round(get_constant(self), ndigits)

            return wrap_node(getattr(self.node, method)(ndigits))

        setattrs(user_type, f"__{method}__", round_magic_impl)
    else:
        method_name = method
        if method in bitwise_ops:
            method_name = bitwise_ops[method]
        setattrs(user_type, f"__{method_name}__", binary_magic_impl)
        if method in reflectable_magic_methods:
            setattrs(user_type, f"__r{method_name}__", rbinary_magic_impl)


for method in magic_methods:  # type: ignore[assignment]
    if method in only_bool_magic_methods:
        _make_user_magic(method, SymBool)
        continue
    if method in only_float_magic_methods:
        _make_user_magic(method, SymFloat)
        continue
    if method in also_bool_magic_methods or method in bool_becomes_int_magic_methods:
        _make_user_magic(method, SymBool)
    _make_user_magic(method, SymInt)
    if method not in bitwise_ops:
        _make_user_magic(method, SymFloat)

del method
del func
<|endoftext|>from __future__ import annotations

import functools
import inspect
import itertools
import logging
from dataclasses import dataclass
from typing import Any, ParamSpec, TYPE_CHECKING, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")

import torch
import torch.utils._pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.fx.experimental.symbolic_shapes import ShapeEnv, TrackedFake


log = logging.getLogger(__name__)
trace_shape_events_log = torch._logging.getArtifactLogger(
    __name__, "trace_shape_events"
)


__all__ = [
    "ShapeEnvEvent",
    "record_shapeenv_event",
    "replay_shape_env_events",
    "FakeTensorMeta",
    "shape_env_check_state_equal",
    "NotEqualError",
]

# [Note: Recording ShapeEnv Events]
# =================================
#
# What is a ShapeEnv event?
# -------------------------
# We consider a ShapeEnv event every function call (ShapeEnv method or
# independent function) that modifies the state of the ShapeEnv instance.
# Such calls are recorded alongside their positional and keyword arguments,
# so that it may be replayed over a different ShapeEnv instance.
#
# See [Note: ShapeEnv State Equality] for what is considered the state
# of a ShapeEnv instance.
#
# What is it for?
# ---------------
# ShapeEnv events recording is used for reconstructing the ShapeEnv in an
# arbitrary state in time.
#
# Being able to arbitrarily replay events like so is useful, mainly for
# translation validation bisection. i.e. if a ValidationException has been
# raised, find the earliest point in time where the translation validation
# fails.
#
# Besides that, it also allows us to inspect the given instance and,
# for example, check the guards that would actually be issued at that point.
#
# What kind of arguments can be stored in an event?
# -------------------------------------------------
# There's no specific rule for what cannot be used as an argument.
# That said, pay special attention to the following cases:
#
#   1. Tensor inputs: there are some tests that check whether the inputs
#      were garbage collected after execution. These will fail if there's
#      an event that is holding a reference to those inputs.
#
#   2. ShapeEnv arguments: if there is an argument of ShapeEnv type, that
#      will be automatically replaced by the new given ShapeEnv instance.
#
#   3. SymTypes arguments: they also hold references to ShapeEnv. So,
#      whenever we see them, we create a new instance, replacing the
#      ShapeEnv reference.
#
#   4. FX nodes: specifically, FX nodes from the FX graph for symbolic
#      shapes. That argument must be replaced when replaying the event at
#      ShapeEnvEvent.run, since it has to reference a node from the given
#      instance, and not from the recorded instance.


# Event class for reconstructing ShapeEnv at arbitrary time.
#
# Represents a method call that mutates ShapeEnv in a way that affects the
# issued guards, when ShapeEnv.produce_guards is called.
@dataclass
class ShapeEnvEvent:
    # ShapeEnv method.
    f: Callable[..., Any]

    # Arguments and keyword arguments called with.
    args: list[object] | None = None
    kwargs: dict[str, Any] | None = None

    # List of tracked_fakes at the time the method was called.
    tracked_fakes: list[TrackedFake] | None = None

    # Name of the captured event.
    # Used for special handling of particular methods.
    name: str | None = None

    # Replay itself, but using shape_env as self.
    def run(self, shape_env: ShapeEnv | None = None) -> Any:
        from torch.fx.experimental.symbolic_shapes import (
            is_symbolic,
            ShapeEnv,
            SymTypes,
        )

        # Special handling for the constructor event.
        if self.f is ShapeEnv:
            if not (
                shape_env is None and self.args is None and self.kwargs is not None
            ):
                raise AssertionError(
                    "ShapeEnv constructor requires shape_env=None, args=None, kwargs set"
                )
            return ShapeEnv(**self.kwargs)

        if shape_env is None:
            raise AssertionError("shape_env is required for non-constructor events")
        args = list(self.args or [])
        kwargs = dict(self.kwargs or {})

        # Replace any argument of type ShapeEnv by the given one.
        args, kwargs = pytree.tree_map_only(
            ShapeEnv, lambda _: shape_env, (args, kwargs)
        )

        # Replace any argument of type SymTypes by a new instance,
        # replacing its ShapeEnv reference.
        args, kwargs = pytree.tree_map_only(
            lambda x: isinstance(x, SymTypes) and is_symbolic(x),
            lambda a: type(a)(a.node.with_shape_env(shape_env)),
            (args, kwargs),
        )

        # Converts FX nodes using the mapping argument.
        def maybe_convert_node(x: Any) -> Any:
            if not isinstance(x, torch.fx.Node):
                # Don't do anything to x if it's not an FX node.
                return x

            # If, at some point, we created an FX node, it means that translation validation is on.
            # It also means we are building an FX graph for symbolic shapes at shape_env.graph, and
            # we are tracking node names at shape_env.name_to_node.
            if not hasattr(shape_env, "name_to_node"):
                raise AssertionError("shape_env missing name_to_node attribute")
            name_to_node = shape_env.name_to_node  # type: ignore[attr-defined]
            if x.name not in name_to_node:
                raise AssertionError(f"Node {x.name} not found in name_to_node")
            return name_to_node[x.name]

        # Replaces the value of a specific argument by the result of fn.
        def replacearg(index: int, key: str, fn: Callable[..., Any]) -> None:
            if index < len(args):
                args[index] = fn(args[index])
            if key in kwargs:
                kwargs[key] = fn(kwargs[key])

        if self.is_create_fx_call_function():
            # ShapeEnv.create_fx_call_function:
            # "args" parameter is a tuple of FX nodes from the FX graph of the old ShapeEnv.
            # They must be replaced, since a "call_function" FX node with this tuple as argument
            # will be added to the FX graph of the new shape_env.
            replacearg(
                index=2,
                key="args",
                fn=lambda args: tuple(maybe_convert_node(a) for a in args),
            )
        if self.is_evaluate_expr() or self.is_defer_runtime_assert():
            # ShapeEnv.evaluate_expr and ShapeEnv.guard_or_defer_runtime_assert:
            # "fx_node" parameter is an (optional) FX node that represents the evaluate expression.
            # They must be replaced, since it will be part of a "call_function" FX node for
            # torch._assert, which will be added to the FX graph of the new shape_env.
            replacearg(index=3, key="fx_node", fn=maybe_convert_node)

        # Actually call the method with the converted arguments.
        return self.f(*args, **kwargs)

    def __str__(self) -> str:
        name = self.name if self.name is not None else self.f.__name__
        return f"event: {name} ({self.args}, {self.kwargs})"

    def is_create_fx_call_function(self) -> bool:
        return self.name == "_create_fx_call_function"

    def is_evaluate_expr(self) -> bool:
        return self.name == "evaluate_expr"

    def is_defer_runtime_assert(self) -> bool:
        return self.name == "guard_or_defer_runtime_assert"


NEST = 0


# Extracts a ShapeEnv instance inside args and kwargs.
# Specifically, it looks for:
#   1. ShapeEnv arguments
#   2. SymInt, SymFloat, or SymBool arguments
# If we find more than one object of any of the above types, we
# also check that the ShapeEnv instance is the same for all of them.
def _extract_shape_env_and_assert_equal(
    args: tuple[object, ...] | list[object], kwargs: dict[str, object]
) -> ShapeEnv | None:
    from torch.fx.experimental.symbolic_shapes import is_symbolic, ShapeEnv, SymTypes

    def assert_equal(old: ShapeEnv | None, new: ShapeEnv) -> ShapeEnv:
        if old is not None:
            if old is not new:
                raise AssertionError("call with different ShapeEnv")
        return new

    shape_env = None
    for val in itertools.chain(args, kwargs.values()):
        if isinstance(val, ShapeEnv):
            shape_env = assert_equal(shape_env, val)
        if isinstance(val, SymTypes) and is_symbolic(val):
            # pyrefly: ignore [bad-argument-type]
            shape_env = assert_equal(shape_env, val.node.shape_env)

    return shape_env


# Decorator for recording the given function as a replayable event.
#
# This decorator should be used at every function that mutates the state of
# ShapeEnv in some way that affects the resulting issued guards (i.e. when
# ShapeEnv.produce_guards is called).
#
# save_tracked_fakes: saves a snapshot of the TrackedFake list.
# This is used when calling ShapeEnv.produce_guards at arbitrary points in time.
#
# name: the name of the function being recorded. Normally (and by default) this
# is taken from the decorated function but can be set if you need to override
# it.
#
# When to save the list of TrackedFake?
# =====================================
# We should save the list of TrackedFake whenever the translation validation
# bisection may actually stop and call the produce_guards method at the moment
# right after the recorded function was played. In other words, since the
# bisection bisects through torch._assert calls, we should save in all methods
# that adds a torch._assert call to the symbolic shapes FX graph.
#
# At the moment, there are 2 methods that save the list:
#   - ShapeEnv.evaluate_expr
#   - ShapeEnv.guard_or_defer_runtime_assert
def record_shapeenv_event(
    *, save_tracked_fakes: bool = False, name: str | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        if not callable(fn):
            raise AssertionError(f"Expected callable, got {type(fn)}")
        args = inspect.getfullargspec(fn).args
        if not (args and args[0] == "self"):
            raise AssertionError(
                "record_shapeenv_event should only wrap methods on ShapeEnv; refactor your "
                "code so that it calls into a method on ShapeEnv"
            )
        nonlocal name
        if name is None:
            name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            if not isinstance(args[0], ShapeEnv):
                raise AssertionError(f"Expected ShapeEnv, got {type(args[0])}")

            global NEST

            trace_shape_events_log.debug(
                "%scall %s(*%r, **%r)", " " * NEST, name, args[1:], kwargs
            )
            NEST += 1

            def retlog(r: _R) -> _R:
                trace_shape_events_log.debug("%s-> %s", " " * (NEST - 1), r)
                return r

            shape_env = args[0]

            try:
                if not shape_env.should_record_events or shape_env.is_recording:  # type: ignore[has-type]
                    # If ShapeEnv is already recording an event, call the wrapped
                    # function directly.
                    #
                    # NB: here, we skip the check of whether all ShapeEnv instances
                    # are equal, in favor of a faster dispatch.
                    return retlog(fn(*args, **kwargs))

                # Retrieve an instance of ShapeEnv.
                # Assumption: the collection of args and kwargs may not reference
                # different ShapeEnv instances.
                self = _extract_shape_env_and_assert_equal(args, kwargs)

                # If we are calling this function without any ShapeEnv instance
                # alive in its arguments, we don't record and call the original.
                if self is None:
                    return retlog(fn(*args, **kwargs))

                # Otherwise, start recording and call the function.
                with self._recording():
                    # Take a snapshot of the current tracked_fakes.
                    tracked_fakes = (
                        self._snapshot_tracked_fakes() if save_tracked_fakes else None
                    )
                    # Record the event for 'fn'.
                    event = ShapeEnvEvent(
                        fn,
                        list(args),
                        kwargs,
                        tracked_fakes,
                        name=name,
                    )
                    # Play the event on this ShapeEnv.
                    # NB: It's important to put the event first, because running
                    # the event can trigger internal events that must be ordered
                    # after this event.  However, if an exception happens, we do
                    # NOT want to have the event in the list, so pop it off from
                    # the record if an error happened
                    self.events.append(event)
                    try:
                        return retlog(event.run(self))
                    except Exception:
                        self.events.pop()
                        raise

            except Exception:
                if not shape_env.should_record_events or shape_env.is_recording:
                    # If ShapeEnv is disabled or already recording an event, re-raise the exception without logging.
                    raise
                log.error(
                    "failed while running %s(*%s, **%s)",
                    name,
                    args[1:],
                    kwargs,
                    exc_info=log.isEnabledFor(logging.INFO),
                )
                raise

            finally:
                NEST -= 1

        return wrapper

    return decorator


# Replays the ShapeEnvEvents list.
# It assumes the first event is the constructor call.
#
# fn: transforms an old FX node into one corresponding to the newly created ShapeEnv.
def replay_shape_env_events(events: list[ShapeEnvEvent]) -> ShapeEnv:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    constructor_event = events[0]
    if constructor_event.f != ShapeEnv:
        raise AssertionError(
            f"First event must be ShapeEnv constructor, got {constructor_event.f}"
        )

    # Constructs the new ShapeEnv.
    shape_env = constructor_event.run()

    for event in events[1:]:
        try:
            # Actually replays each event.
            # We need to call create_mapping_fn every time, since the node list might
            # change after each event is replayed.
            event.run(shape_env)
        except Exception:
            log.error("failed when running event: %s", event)
            raise

    return shape_env


# FakeTensor metadata.
# This is to be used in place of FakeTensor placeholders when calling
# ShapeEnv.produce_guards.
@dataclass
class FakeTensorMeta:
    tensor_size: tuple[int | torch.SymInt, ...]
    tensor_stride: tuple[int | torch.SymInt, ...]
    tensor_storage_offset: int | torch.SymInt
    is_nested: bool

    def size(self) -> tuple[int | torch.SymInt, ...]:
        return self.tensor_size

    def stride(self) -> tuple[int | torch.SymInt, ...]:
        return self.tensor_stride

    def storage_offset(self) -> int | torch.SymInt:
        return self.tensor_storage_offset

    def dim(self) -> int:
        return len(self.tensor_size)

    @staticmethod
    def from_fake(fake: torch.Tensor) -> FakeTensorMeta:
        return FakeTensorMeta(
            fake.size(), fake.stride(), fake.storage_offset(), fake.is_nested
        )


# [Note: ShapeEnv State Equality]
# ===============================
#
# What is considered ShapeEnv state?
# ----------------------------------
# We consider to be the state of a ShapeEnv instance everything that
# is not in the inline tuple inside remove_nonstate_variables function.
# That is: the fields within ShapeEnv that modify the flow of execution
# of the program.
#
# So, for example: the replacements field might influence on how an
# expression is simplified. That, in turn, may result in a guard being
# statically known (i.e. not added).
#
# On the other hand, var_to_stack serves only changes what is printed
# in the screen, i.e. used only for debugging purposes. Therefore, we
# should not consider it when comparing states.
#
# What to do on NotEqualError?
# ----------------------------
# Here are a few possible causes for getting a NotEqualError raised:
#
#   1. New field that does not belong in the ShapeEnv state.
#      For example: log field of type ShapeEnvLoggerAdapter. Different
#      ShapeEnv instances will always have different ShapeEnvLoggerAdapter
#      instances, i.e. equality comparison would fail.
#      Solution: add it to the inlined tuple inside remove_nonstate_variables
#      function inside check_equal method.
#
#   2. New field that is not directly comparable across instances.
#      For example: guards field of type List[ShapeGuard]. More specifically,
#      the ShapeGuard type holds an expression and a stack information
#      for debugging purposes. When replaying the even on a new ShapeEnv
#      instance, the stack would be different, which would trigger this error.
#      Solution: add a special case to the map_value function inside
#      check_equal function.
#
#   3. Mutation of ShapeEnv on some not recorded function.
#      If a mutation of the state of ShapeEnv happens inside a function
#      that is not recorded (or that no caller in the stack is recorded),
#      then, the replayed ShapeEnv won't catch that.
#      Solution: decorate the function with record_shape_env_event.


# Checks whether the state of two ShapeEnv are equal w.r.t. the guards
# returned by ShapeEnv.produce_guards.
def shape_env_check_state_equal(
    env1: ShapeEnv,
    env2: ShapeEnv,
    non_state_variable_names: tuple[str, ...],
    map_value: Callable[[str, object], object],
) -> None:
    # Collect and remove variables that don't necessarily represent the state
    # of a ShapeEnv. Note: we copy the dictionary so that we don't modify the
    # instance itself.
    env1_vars = vars(env1).copy()
    env2_vars = vars(env2).copy()

    for v in non_state_variable_names:
        if v in env1_vars:
            env1_vars.pop(v)
        if v in env2_vars:
            env2_vars.pop(v)

    # Function for transforming the mismatched values into string.
    # Needed, since dict and set entries order might not be the same every time.
    def value_to_str(value: Any) -> str:
        if isinstance(value, dict):
            return (
                "{"
                + ", ".join(f"{k}: {value[k]}" for k in sorted(value.keys(), key=str))
                + "}"
            )
        if isinstance(value, set):
            return "{" + ", ".join(f"{v}" for v in sorted(value)) + "}"
        return str(value)

    # Compares env1_vars with env2_vars.
    # Here, we allow the value of each field to be mapped, so that we appropriately
    # compare the two values.
    def compare_vars(
        map_value: Callable[[str, object], object],
    ) -> list[tuple[str, str, str]]:
        env1_set, env2_set = set(env1_vars), set(env2_vars)

        # First, compare the set of keys in each vars dictionary.
        if env1_set != env2_set:
            raise NotEqualError(
                "field set mismatch:",
                [
                    (
                        "found unique fields:",
                        str(sorted(env1_set - env2_set)),
                        str(sorted(env2_set - env1_set)),
                    ),
                ],
            )

        # Then, sort the keys, and compare the mapped values of each key.
        sorted_keys = list(env1_set)
        sorted_keys.sort()

        mapped_dict = [
            (k, map_value(k, env1_vars[k]), map_value(k, env2_vars[k]))
            for k in sorted_keys
        ]

        # Return a list of tuples representing the fields that did not match
        # alongside their respective mapped values.
        return [
            (f"{k}: values don't match.", value_to_str(val1), value_to_str(val2))
            for k, val1, val2 in mapped_dict
            if val1 != val2
        ]

    # Accumulate the mismatching fields.
    errors = compare_vars(map_value)

    if len(errors) > 0:
        raise NotEqualError("field values don't match:", errors)


class NotEqualError(Exception):
    def __init__(
        self,
        msg: str,
        mismatched: list[tuple[str, str, str]],
    ) -> None:
        details = "\n".join(
            [
                "\n".join(
                    [
                        f"==> {inner_msg}",
                        f"  >  Left: {str1}",
                        f"  > Right: {str2}",
                    ]
                )
                for inner_msg, str1, str2 in mismatched
            ]
        )

        super().__init__(
            f"""\
ShapeEnv not equal: {msg}

{details}
"""
        )
<|endoftext|>import ast
import copy
import functools
import inspect
import textwrap
from collections.abc import Callable
from types import FunctionType
from typing import Any, cast

import torch
from torch._sources import normalize_source_lines
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph import Graph


class AST_Rewriter(ast.NodeTransformer):
    """
    Take a FunctionType object representing a `forward` method, then
    perform an AST rewrite to swap out nodes that are not symbolically
    traceable with a callsite to the FX alternative.

    To support swapping out an AST node, define a new `visit` method on
    that node. For more details, see:
    https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    """

    # This function checks for new keys added in the globals dict. TorchDynamo
    # can insert new keys in the global dict and upset the check. Therefore, put
    # a disable here. This function is an optimization pass and not really
    # suitable for dynamo tracing anyways.
    @torch._dynamo.disable
    def rewrite(self, fn: FunctionType) -> FunctionType:
        # Normalize the source lines
        sourcelines, _ = inspect.getsourcelines(fn)
        sourcelines = normalize_source_lines(sourcelines)
        source = "".join(sourcelines)
        normalized_str = textwrap.dedent(source)

        # Rewrite the original AST
        source_ast = ast.parse(normalized_str)
        dest_ast = ast.fix_missing_locations(self.visit(source_ast))

        # Pull out the compiled function from the newly-created Module
        code = compile(dest_ast, "", "exec")
        globals_dict = copy.copy(fn.__globals__)
        keys_before = set(globals_dict.keys())
        exec(code, globals_dict)
        new_keys = list(set(globals_dict.keys()) - keys_before)
        if len(new_keys) != 1:
            raise AssertionError(f"Expected 1 new key, got {len(new_keys)}")
        fn_compiled = globals_dict[new_keys[0]]

        # return the compiled function with the original globals
        def change_func_globals(
            f: FunctionType, globals: dict[str, object]
        ) -> FunctionType:
            """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
            # __globals__ is a private member of the function class
            # so we have to copy the function, f, all of its member, except f.__globals__
            g = FunctionType(
                f.__code__,
                globals,
                name=f.__name__,
                argdefs=f.__defaults__,
                closure=f.__closure__,
            )
            g = functools.update_wrapper(g, f)
            g.__kwdefaults__ = copy.copy(f.__kwdefaults__)  # type:ignore[attr-defined]
            return g  # pyrefly: ignore [bad-return]

        # Return the correct FunctionType object
        return change_func_globals(fn_compiled, globals=fn.__globals__)

    def visit_Assert(self, node: ast.Assert) -> ast.Expr:
        """
        Swap out the Assert node (Python's `assert`) with a callsite to the
        symbolically-traceable torch._assert function
        """
        # Create the Call node
        n = ast.parse("torch._assert()", mode="eval")
        if not isinstance(n, ast.Expression):
            raise AssertionError(f"Expected ast.Expression, got {type(n)}")
        call_node = n.body
        if not isinstance(call_node, ast.Call):
            raise AssertionError(f"Expected ast.Call, got {type(call_node)}")
        msg = node.msg if node.msg else ast.Constant(value="", kind=None)
        call_node.args = [node.test, msg]

        # Ensure that the new node conforms to the Python AST grammar
        expr_wrapper = ast.Expr(value=call_node)

        # Return the new Call node to signify that we want to use it as
        # a replacement for the original _assert node
        return ast.copy_location(expr_wrapper, node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign:
        """
        Swap out Python's AnnAssign with an Assign node where the annotation function is called.
        Example:
             Original:
             y: Tensor_Type(1,2,3, Dyn) = f2(x)
            Output:
             y = annotate(f2(x),Tensor_Type((1,2,3,Dyn)))
        """
        return ast.Assign(
            targets=[node.target],
            value=ast.Call(
                func=ast.Name(id="annotate", ctx=ast.Load()),
                # pyrefly: ignore [bad-argument-type]
                args=[node.value, node.annotation],
                keywords=[],
            ),
        )


class RewritingTracer(Tracer):
    def trace(
        self,
        root: torch.nn.Module | Callable[..., Any],
        concrete_args: dict[str, Any] | None = None,
    ) -> Graph:
        return super().trace(_rewrite(root), concrete_args)


def _rewrite(
    fn: torch.nn.Module | Callable[..., Any],
) -> torch.nn.Module | Callable[..., Any]:
    if isinstance(fn, torch.nn.Module):
        # Rewrite this module's `forward` as well as the `forward`s of
        # all of this module's recursive descendants. Return the new,
        # rewritten module hierarchy.
        def rewrite_module(m: torch.nn.Module) -> torch.nn.Module:
            class RewrittenModule(torch.nn.Module):
                def __init__(self, orig: torch.nn.Module) -> None:
                    super().__init__()
                    for k, v in orig.__dict__.items():
                        if isinstance(v, torch.nn.Module):
                            self.__dict__[k] = copy.copy(rewrite_module(v))
                        else:
                            self.__dict__[k] = copy.copy(v)

            RewrittenModule.forward = AST_Rewriter().rewrite(
                cast(FunctionType, m.forward)
            )
            return RewrittenModule(m)

        return rewrite_module(fn)
    else:
        # Rewrite this single free function
        return AST_Rewriter().rewrite(cast(FunctionType, fn))
<|endoftext|>from enum import Enum
from typing import NamedTuple

from torch.fx.node import map_arg, Node


class Partition:
    """Partition class contains all the information about an individual partition.
    It also provides necessary methods for manipulating the partition.
    """

    def __init__(self, partition_id: int) -> None:
        self.nodes: set[Node] = set()
        self.partition_id = partition_id
        self.parents: set[Partition] = set()
        self.children: set[Partition] = set()
        self.bfs_level: int = -1
        self.used_mem_bytes: int = 0
        self.logical_device_ids: list[int] = []

    def __str__(self) -> str:
        return str(self.partition_id)

    def recalculate_mem_size(self) -> None:
        self.used_mem_bytes = 0
        for node in self.nodes:
            self.used_mem_bytes += get_extra_size_of(node, self.nodes)

    def add_node(self, node: Node) -> None:
        input_nodes: dict[Node, None] = {}
        map_arg(node.args, input_nodes.setdefault)
        map_arg(node.kwargs, input_nodes.setdefault)
        # Add current node's input nodes if they are placeholder or constants
        for n in input_nodes:
            if n.op in {"placeholder", "get_attr"}:
                self.nodes.add(n)
        self.nodes.add(node)
        self.recalculate_mem_size()

    def remove_node(self, node: Node) -> None:
        # Remove a node only if the node is in the partition
        if node in self.nodes:
            self.nodes.remove(node)
            # Collect the node's input nodes
            input_nodes: dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)
            map_arg(node.kwargs, input_nodes.setdefault)
            # Check if an input node is a placeholder or get_attr,
            # and this input node is not used by some other nodes in this partition,
            # then remove this input node
            for input_node in input_nodes:
                if all(
                    n not in self.nodes for n in input_node.users
                ) and input_node.op in {"placeholder", "get_attr"}:
                    self.nodes.remove(input_node)
            self.recalculate_mem_size()


class Device(NamedTuple):
    name: str
    available_mem_bytes: int
    logical_id: int


class NodeLatency(NamedTuple):
    # Latency due to the memory bandwidth
    mem_latency_sec: float
    # Latency due to the computation
    computer_latency_sec: float


class PartitionLatency(NamedTuple):
    # Sum of all nodes' memory latency on the critical path
    mem_latency_sec: float
    # Sum of all nodes' compute latency on the critical path
    computer_latency_sec: float
    # Latency of the critical path
    overall_latency_sec: float


class PartitionMode(Enum):
    size_based = 0
    sparse_nn = 1
    cost_aware = 2
    kl_based = 3
    aot_based = 4


class PartitionerConfig(NamedTuple):
    devices: list[Device]
    mode: PartitionMode = PartitionMode.size_based
    transfer_rate_bytes_per_sec: float = 0.0
    node_to_latency_mapping: dict[Node, NodeLatency] = {}
    node_to_partition_mapping: dict[Node, int] = {}
    partition_to_logical_device_mapping: dict[int, list[int]] = {}
    # Saturate host by replicating partitions to the remaining idle devices.
    saturate_host: bool = False


def get_extra_size_of(node: Node, nodes: set[Node]) -> int:
    """Given a node and a set of nodes,
    this function return the extra size that needed
    if this node is included in this set.
    """
    # Find all its input nodes
    input_nodes: dict[Node, None] = {}
    map_arg(node.args, input_nodes.setdefault)
    map_arg(node.kwargs, input_nodes.setdefault)
    # Calculate total size of related nodes
    total_size_of_input_nodes = 0
    for n in input_nodes:
        # Make sure this node hasn't been in this set yet
        if n not in nodes:
            size_bytes = getattr(n, "size_bytes", None)
            if size_bytes:
                total_size_of_input_nodes += size_bytes.output_size
            else:
                raise RuntimeError("node has no size_bytes attr")
    # Don't forget the op node itself
    size_bytes = getattr(node, "size_bytes", None)
    if size_bytes:
        total_size_of_input_nodes += size_bytes.total_size
    else:
        raise RuntimeError("node has no size_bytes attr")
    return total_size_of_input_nodes


def get_latency_of_one_partition(
    partition: Partition, node_to_latency_mapping: dict[Node, NodeLatency]
) -> PartitionLatency:
    """Given a partition and its nodes' latency, return a PartitionLatency for this partition"""

    def get_top_nodes(partition: Partition) -> list[Node]:
        """Given a partition, return a list of nodes on the top bfs level"""
        top_nodes: list[Node] = []
        for node in partition.nodes:
            # Skip placeholder and get_attr nodes
            if node.op in {"placeholder", "get_attr"}:
                continue
            input_nodes: dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)
            map_arg(node.kwargs, input_nodes.setdefault)
            # If a node has no input nodes in this partition,
            # or its input nodes in this partition are placeholders and get_attrs
            # this node is on the top bfs level in this partition
            if not any(
                n in partition.nodes and n.op not in {"placeholder", "get_attr"}
                for n in input_nodes
            ):
                top_nodes.append(node)
        return top_nodes

    def dfs_helper(node: Node, partition_latency: PartitionLatency) -> PartitionLatency:
        """Given a top node of a partition, this function returns
        the latency of the critical path in the partition
        """
        node_latency = node_to_latency_mapping[node]
        # Calculate the current overall latency of the partition
        overall_latency_sec = partition_latency.overall_latency_sec + max(
            node_latency.computer_latency_sec, node_latency.mem_latency_sec
        )
        # Update the mem latency of this path
        mem_latency_sec = (
            partition_latency.mem_latency_sec + node_latency.mem_latency_sec
        )
        # Update the compute latency of this path
        computer_latency_sec = (
            partition_latency.computer_latency_sec + node_latency.computer_latency_sec
        )
        # Get all users of this node that are in this partition
        users = set(node.users).intersection(partition.nodes)
        if users:
            max_latency = PartitionLatency(
                mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0
            )
            for n in users:
                # Get new partition latency recursively
                new_partition_latency = dfs_helper(
                    n,
                    PartitionLatency(
                        mem_latency_sec, computer_latency_sec, overall_latency_sec
                    ),
                )
                if (
                    new_partition_latency.overall_latency_sec
                    > max_latency.overall_latency_sec
                ):
                    max_latency = new_partition_latency
            return max_latency
        # If there is no user, the node is at bottom of the partition
        return PartitionLatency(
            mem_latency_sec, computer_latency_sec, overall_latency_sec
        )

    # Main part starts
    # Get all top level nodes of this partition
    top_nodes = get_top_nodes(partition)
    critical_path_latency = PartitionLatency(
        mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0
    )
    # Go through all top nodes and find the largest latency (critical pass latency)
    for node in top_nodes:
        partition_latency = dfs_helper(
            node,
            PartitionLatency(
                mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0
            ),
        )
        if (
            partition_latency.overall_latency_sec
            > critical_path_latency.overall_latency_sec
        ):
            critical_path_latency = partition_latency
    return critical_path_latency


def get_partition_to_latency_mapping(
    partitions: list[Partition], node_to_latency_mapping: dict[Node, NodeLatency]
) -> dict[Partition, PartitionLatency]:
    """Given all the partitions and node_to_latency_mapping dictionary,
    return a mapping dictionary of each partition to its overall latency
    """
    partition_to_latency_mapping: dict[Partition, PartitionLatency] = {}
    # Go through each partition and get its latency
    for partition in partitions:
        partition_latency = get_latency_of_one_partition(
            partition, node_to_latency_mapping
        )
        partition_to_latency_mapping[partition] = partition_latency
    return partition_to_latency_mapping


def get_comm_latency_between(
    parent_partition: Partition,
    child_partition: Partition,
    transfer_rate_bytes_per_sec: float,
) -> float:
    """Given two partitions (parent and child),
    calculate the communication latency between the two.
    """
    # If two partitions are on the same device, the comm latency is 0.
    if (
        parent_partition.logical_device_ids != []
        and child_partition.logical_device_ids != []
        and parent_partition.logical_device_ids == child_partition.logical_device_ids
    ):
        return 0.0
    # Keep tracking the communication size between parent and child
    comm_size = 0
    # Keep tracking all the counted node
    visited_nodes = set()
    # Go through all nodes in the child partition
    # If a node has input nodes from the parent partition,
    # the output size of those input nodes will be counted
    # and added to comm_size
    for node in child_partition.nodes:
        input_nodes: dict[Node, None] = {}
        map_arg(node.args, input_nodes.setdefault)
        map_arg(node.kwargs, input_nodes.setdefault)
        for n in input_nodes:
            if n in parent_partition.nodes and n not in visited_nodes:
                size_bytes = getattr(n, "size_bytes", None)
                if size_bytes is not None:
                    comm_size += size_bytes.output_size
                visited_nodes.add(n)
    return comm_size / transfer_rate_bytes_per_sec


def get_latency_of_partitioned_graph(
    partitions: list[Partition],
    partition_to_latency_mapping: dict[Partition, PartitionLatency],
    transfer_rate_bytes_per_sec: float,
) -> float:
    """Given all partitions in a graph, find the critical path among all partitions
    and return its latency as the latency of the whole graph
    """

    def dfs_helper(partition: Partition, latency_so_far_sec: float) -> float:
        """This function helps to recursively get the latency of a path of partitions"""
        # Update latency by adding current partition's latency
        latency_so_far_sec += partition_to_latency_mapping[
            partition
        ].overall_latency_sec

        if partition.children:
            max_latency_sec = 0.0
            for child in partition.children:
                # Calculate latency between
                comm_latency_sec = get_comm_latency_between(
                    partition, child, transfer_rate_bytes_per_sec
                )
                new_latency_sec = dfs_helper(
                    child, latency_so_far_sec + comm_latency_sec
                )
                if new_latency_sec > max_latency_sec:
                    max_latency_sec = new_latency_sec
            return max_latency_sec
        return latency_so_far_sec

    def get_top_partitions(partitions: list[Partition]) -> list[Partition]:
        """This function is to return all the partitions without parents
        as the starting points of all the paths
        """
        # If a partition has no parents, then it is a top partition
        top_partitions = [
            partition for partition in partitions if len(partition.parents) == 0
        ]
        return top_partitions

    top_partitions = get_top_partitions(partitions)
    critical_path_latency_sec = 0.0
    for partition in top_partitions:
        latency_sec = dfs_helper(partition, 0.0)
        if latency_sec > critical_path_latency_sec:
            critical_path_latency_sec = latency_sec
    return critical_path_latency_sec
<|endoftext|>import copy
import logging
import operator
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any, cast

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.mkldnn as th_mkldnn
from torch.fx.node import Argument, Target
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval


__all__ = [
    "matches_module_pattern",
    "replace_node_module",
    "fuse",
    "remove_dropout",
    "extract_subgraph",
    "modules_to_mkldnn",
    "reset_modules",
    "MklSubgraph",
    "gen_mkl_autotuner",
    "use_mkl_length",
    "UnionFind",
    "optimize_for_inference",
]


def _parent_name(target: str) -> tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


# Works for length 2 patterns with 2 modules
def matches_module_pattern(
    pattern: Iterable[type], node: fx.Node, modules: dict[str, torch.nn.Module]
) -> bool:
    if len(node.args) == 0:
        return False
    nodes: tuple[Argument, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != "call_module":
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def replace_node_module(
    node: fx.Node, modules: dict[str, torch.nn.Module], new_module: torch.nn.Module
) -> None:
    if not isinstance(node.target, str):
        raise AssertionError(f"Expected str target, got {type(node.target)}")
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


def fuse(
    model: torch.nn.Module, inplace: bool = False, no_trace: bool = False
) -> torch.nn.Module:
    """
    Fuses convolution/BN and linear/BN layers for inference purposes.
    Will deepcopy your model by default, but can modify the model inplace as well.
    """
    patterns = [
        (nn.Conv1d, nn.BatchNorm1d),
        (nn.Conv2d, nn.BatchNorm2d),
        (nn.Conv3d, nn.BatchNorm3d),
        (nn.Linear, nn.BatchNorm1d),
    ]
    if not inplace:
        model = copy.deepcopy(model)
    if not no_trace or not isinstance(model, torch.fx.GraphModule):
        fx_model = fx.symbolic_trace(model)
    else:
        fx_model = model
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:
                    # Output of conv/linear is used by other nodes
                    continue
                first_layer = modules[node.args[0].target]
                bn = modules[node.target]
                if not bn.track_running_stats:
                    continue
                if pattern[0] in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
                    fused_layer = fuse_conv_bn_eval(first_layer, bn)
                else:  # nn.Linear
                    fused_layer = fuse_linear_bn_eval(first_layer, bn)
                replace_node_module(node.args[0], modules, fused_layer)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)


def remove_dropout(model: nn.Module) -> nn.Module:
    """
    Removes all dropout layers from the module.
    """
    fx_model = fx.symbolic_trace(model)

    class DropoutRemover(torch.fx.Transformer):
        def call_module(
            self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
        ) -> Any:
            if isinstance(self.submodules[target], nn.Dropout):
                if len(args) != 1:
                    raise AssertionError(f"Expected 1 arg for Dropout, got {len(args)}")
                return args[0]
            else:
                return super().call_module(target, args, kwargs)

    return DropoutRemover(fx_model).transform()


def extract_subgraph(
    orig_module: nn.Module,
    nodes: list[fx.Node],
    inputs: list[fx.Node],
    outputs: list[fx.Node],
) -> fx.GraphModule:
    """
    Given lists of nodes from an existing graph that represent a subgraph, returns a submodule that executes that subgraph.
    """
    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    for input in inputs:
        new_node = new_graph.placeholder(input.name)
        env[input] = new_node
    for node in nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x])
        env[node] = new_node
    new_graph.output([env[output] for output in outputs])
    new_graph.lint()
    return fx.GraphModule(orig_module, new_graph)


mkldnn_supported = [
    nn.Conv2d,
    nn.Linear,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
    torch.relu,
    torch.transpose,
    torch.sigmoid,
    F.relu,
    F.avg_pool2d,
    F.adaptive_avg_pool2d,
]
# These are operators that may not be convertible into MKLDNN ops (e.g. the
# args are scalar values). Thus, we only include them in the subgraph if their
# arguments are already in MKLDNN.
# TODO: Determine whether this can be removed after type inference.
mkldnn_supported_unknown = [operator.add, operator.mul]


def modules_to_mkldnn(
    nodes: list[fx.Node], modules: dict[str, nn.Module]
) -> dict[nn.Module, nn.Module]:
    """
    For each node, if it's a module that can be preconverted into MKLDNN,
    then we do so and create a mapping to allow us to convert from the MKLDNN
    version of the module to the original.
    """
    # Built lazily so importing this module does not compile any TorchScript
    # ScriptModule via torch.utils.mkldnn.
    mkldnn_map = {
        nn.Conv2d: th_mkldnn.MkldnnConv2d,
        nn.Linear: th_mkldnn.MkldnnLinear,
        nn.BatchNorm2d: lambda a, _: th_mkldnn.MkldnnBatchNorm(a),
    }
    old_modules: dict[nn.Module, nn.Module] = {}
    for node in nodes:
        if node.op == "call_module":
            if not isinstance(node.target, str):
                raise AssertionError(f"Expected str target, got {type(node.target)}")
            cur_module = modules[node.target]
            if type(cur_module) in mkldnn_map:
                # pyrefly: ignore [bad-index, index-error]
                new_module = mkldnn_map[type(cur_module)](cur_module, torch.float)
                if not isinstance(new_module, nn.Module):
                    raise AssertionError(f"Expected nn.Module, got {type(new_module)}")
                old_modules[new_module] = copy.deepcopy(cur_module)
                replace_node_module(node, modules, new_module)
    return old_modules


def reset_modules(
    nodes: Iterable[fx.Node],
    modules: dict[str, nn.Module],
    old_modules: dict[nn.Module, nn.Module],
) -> None:
    """
    Maps each module that's been changed with `modules_to_mkldnn` back to its
    original.
    """
    for node in nodes:
        if node.op == "call_module":
            if not isinstance(node.target, str):
                raise AssertionError(f"Expected str target, got {type(node.target)}")
            cur_module = modules[node.target]
            if cur_module in old_modules:
                replace_node_module(node, modules, old_modules[cur_module])


class MklSubgraph:
    def __init__(self, fx_graph: fx.Graph) -> None:
        self.fx_graph = fx_graph
        self.nodes: list[fx.Node] = []
        self.start_nodes: list[fx.Node] = []
        self.end_nodes: list[fx.Node] = []


def gen_mkl_autotuner(
    example_inputs: list[torch.Tensor], iters: int = 10, warmup: int = 1
) -> Callable[[MklSubgraph], bool]:
    """
    This generates a heuristic that can be passed into `optimize_for_inference` that
    determines whether a subgraph should be run in MKL by running it with the example_inputs.

    Example usage:
        heuristic = gen_mkl_autotuner(example_inputs, iters=10)
        fast_model = optimization.optimize_for_inference(model, heuristic)
    """
    fx_model = None
    old_modules = None

    def use_mkl_heuristic(graph: MklSubgraph) -> bool:
        nonlocal fx_model, old_modules
        input_nodes = graph.start_nodes
        if fx_model is None:
            fx_model = graph.fx_graph.owning_module
            if fx_model is None:
                raise AssertionError("fx_graph.owning_module must not be None")
            old_modules = graph.fx_graph.old_modules  # type: ignore[attr-defined]
            ShapeProp(fx_model).propagate(example_inputs)
        sample_inputs = [torch.randn(node.shape) for node in input_nodes]  # type: ignore[attr-defined]
        output_args = cast(list[fx.Node], [node.args[0] for node in graph.end_nodes])
        if fx_model is None:
            raise AssertionError("fx_model must not be None")
        submodule = extract_subgraph(fx_model, graph.nodes, input_nodes, output_args)

        def benchmark(f: Callable[[], object]) -> float:
            for _ in range(warmup):
                f()
            begin = time.time()
            for _ in range(iters):
                f()
            return time.time() - begin

        mkl_time = benchmark(
            lambda: [
                i.to_dense() for i in submodule(*[i.to_mkldnn() for i in sample_inputs])
            ]
        )

        reset_modules(
            submodule.graph.nodes,
            dict(submodule.named_modules()),
            # pyrefly: ignore [bad-argument-type]  # old_modules is set before this point
            old_modules,
        )
        no_mkl_time = benchmark(lambda: submodule(*sample_inputs))
        return mkl_time < no_mkl_time

    return use_mkl_heuristic


def use_mkl_length(graph: MklSubgraph) -> bool:
    """
    This is a heuristic that can be passed into `optimize_for_inference` that
    determines whether a subgraph should be run in MKL by checking if there
    are more than 2 nodes in it
    """
    return len(graph.nodes) > 2


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent: list[int | None] = [None] * n
        self.size: list[int] = [0] * n

    def make_set(self, v: int) -> None:
        self.parent[v] = v
        self.size[v] = 1

    def find(self, v: int) -> int:
        par = self.parent[v]
        if v == par:
            return v
        if par is None:
            raise AssertionError("Parent is None")
        self.parent[v] = self.find(par)
        return cast(int, self.parent[v])

    def join(self, a: int, b: int) -> int | None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return a
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def optimize_for_inference(
    model: torch.nn.Module,
    pass_config: dict[str, Any] | None = None,
    tracer: type[fx.Tracer] = fx.Tracer,
) -> torch.nn.Module:
    """
    Performs a set of optimization passes to optimize a model for the
    purposes of inference. Specifically, the passes that are run are:
    1. Conv/BN fusion
    2. Dropout removal
    3. MKL layout optimizations

    The third optimization takes a function `use_mkl_heuristic` that's used
    to determine whether a subgraph should be explicitly run in MKL layout.

    Note: As FX does not currently handle aliasing, this pass currently
    assumes nothing aliases. If that isn't true, use at your own risk.
    """
    default_pass_config = {
        "conv_bn_fuse": True,
        "remove_dropout": True,
        "mkldnn_layout_optimize": {"heuristic": use_mkl_length},
    }
    if pass_config is None:
        pass_config = {}
    default_pass_config.update(pass_config)

    if default_pass_config["conv_bn_fuse"]:
        model = fuse(model)
    if default_pass_config["remove_dropout"]:
        model = remove_dropout(model)
    if default_pass_config["mkldnn_layout_optimize"] is False:
        return model
    if not isinstance(default_pass_config["mkldnn_layout_optimize"], dict):
        raise RuntimeError("mkldnn_layout_optimize config is not a dict")
    if "heuristic" not in default_pass_config["mkldnn_layout_optimize"]:
        raise RuntimeError("Heuristic not found in mkldnn_layout_optimize config")
    use_mkl_heuristic = default_pass_config["mkldnn_layout_optimize"]["heuristic"]

    cur_tracer = tracer()
    fx_graph = cur_tracer.trace(copy.deepcopy(model))
    fx.GraphModule(cur_tracer.root, fx_graph)
    modules: dict[str, nn.Module] = dict(model.named_modules())

    class MklSupport(Enum):
        NO = 1
        YES = 2
        UNKNOWN = 3

    # Inserts to_mkldnn and to_dense around every node we want to be a MKLDNN node.
    # If the op is in `mkldnn_supported` then we always treat it as a MKLDNN node.
    # However, if it's in `mkldnn_supported_unknown`, then we only treat it as
    # a MKLDNN node if its inputs are MKLDNN nodes.
    for node in list(fx_graph.nodes):
        supports_mkldnn = MklSupport.NO
        if node.op == "call_module":
            cur_module = modules[node.target]
            if type(cur_module) in mkldnn_supported:
                supports_mkldnn = MklSupport.YES
                sample_parameter = next(cur_module.parameters(), None)
                if sample_parameter is not None:
                    if sample_parameter.dtype != torch.float:
                        raise AssertionError(
                            "this pass is only for torch.float modules"
                        )
                    if sample_parameter.device != torch.device("cpu"):
                        raise AssertionError("this pass is only for CPU modules")
        elif node.op == "call_function":
            if node.target in mkldnn_supported:
                supports_mkldnn = MklSupport.YES
            elif node.target in mkldnn_supported_unknown:
                supports_mkldnn = MklSupport.UNKNOWN

        if supports_mkldnn != MklSupport.NO:
            if supports_mkldnn == MklSupport.UNKNOWN:
                if not any(arg.target == "to_dense" for arg in node.args):
                    continue
            with fx_graph.inserting_before(node):
                mkldnn_args = fx.map_arg(
                    node.args, lambda n: fx_graph.call_method("to_mkldnn", (n,))
                )

            node.args = cast(tuple[fx.node.Argument], mkldnn_args)

            with fx_graph.inserting_after(node):
                dense_x = fx_graph.create_node("call_method", "to_dense", (node,))
                node.replace_all_uses_with(dense_x)
                dense_x.args = (node,)

    # Does pre-conversion of all modules into MKLDNN (when possible)
    old_modules = modules_to_mkldnn(list(fx_graph.nodes), modules)
    fx_graph.old_modules = old_modules  # type: ignore[attr-defined]

    # optimizes all a -> to_dense -> to_mkldnn -> b patterns into a -> b
    for node in fx_graph.nodes:
        if node.op == "call_method" and node.target == "to_dense":
            prv_node = node.args[0]
            users = list(node.users)
            for user in users:
                if user.op == "call_method" and user.target == "to_mkldnn":
                    user.replace_all_uses_with(prv_node)
                    fx_graph.erase_node(user)
            if len(node.users) == 0:
                fx_graph.erase_node(node)

    num_nodes = len(fx_graph.nodes)
    uf = UnionFind(num_nodes)

    def get_color(n: fx.Node) -> int | None:
        if hasattr(n, "color"):  # Current node is part of a MKL subgraph
            return uf.find(n.color)
        if hasattr(n, "start_color"):  # Current node is input to MKL subgraph
            return uf.find(n.start_color)
        return None

    # This code is to find each MKLDNN subgraph. Each MKLDNN subgraph consists
    # of input nodes (which are only `to_mkldnn` calls), output nodes
    # (`to_dense` calls), and intermediate nodes, which are run entirely on
    # MKLDNN layout tensors.
    #
    # Specifically, this code does a flood fill on a directed acyclic graph
    # (DAG), starting from each possible "start node" (i.e: `to_mkldnn` nodes).
    # If every node only had one input, this would be sufficient. However, in
    # the case that a node has multiple inputs coming from different start
    # nodes (i.e. colors), we need to join these 2 colors into 1. That's done
    # using a Disjoint Set Union.
    for cur_idx, node in enumerate(fx_graph.nodes):
        if node.op == "call_method" and node.target == "to_mkldnn":
            node.start_color = cur_idx
            uf.make_set(cur_idx)
        elif node.op == "call_method" and node.target == "to_dense":
            if get_color(node.args[0]) is None:
                raise AssertionError("Expected color for to_dense input")
            node.end_color = get_color(node.args[0])
        else:
            cur_colors = [
                get_color(i)
                for i in node.all_input_nodes
                if isinstance(i, fx.Node)
                if get_color(i) is not None
            ]

            if len(cur_colors) == 0:
                continue
            if any(i is None for i in cur_colors):
                raise AssertionError("Found None in cur_colors")
            sorted_colors: list[int] = sorted(cur_colors)  # type: ignore[arg-type]
            node.color = sorted_colors[0]
            for other_color in sorted_colors[1:]:
                uf.join(sorted_colors[0], other_color)

    mkldnn_graphs: dict[int, MklSubgraph] = defaultdict(lambda: MklSubgraph(fx_graph))
    for node in fx_graph.nodes:
        if hasattr(node, "color"):
            mkldnn_graphs[uf.find(node.color)].nodes.append(node)
        if hasattr(node, "start_color"):
            mkldnn_graphs[uf.find(node.start_color)].start_nodes.append(node)
        if hasattr(node, "end_color"):
            mkldnn_graphs[uf.find(node.end_color)].end_nodes.append(node)

    # Now that we have all the subgraphs, we need to decide which MKLDNN
    # subgraphs we actually want to keep in MKLDNN.
    for graph in mkldnn_graphs.values():
        if not use_mkl_heuristic(graph):
            for node in graph.start_nodes + graph.end_nodes:
                prv = node.args[0]
                node.replace_all_uses_with(prv)  # type: ignore[arg-type]
                fx_graph.erase_node(node)
            reset_modules(graph.nodes, modules, old_modules)

    mkldnn_conversions = 0
    for node in fx_graph.nodes:
        if node.target == "to_mkldnn" or node.target == "to_dense":
            mkldnn_conversions += 1

    logging.getLogger(__name__).info("mkldnn conversions: %s", mkldnn_conversions)
    fx_graph.lint()
    result = fx.GraphModule(model, fx_graph)
    return result
<|endoftext|>"""
Size hinting utilities for symbolic shape expressions.

This module contains the core logic for resolving symbolic expressions to
concrete integer hints. Two strategies are provided:

- _guarding_hint_or_throw_base: strict, only uses backed symbol hints, throws on
  unbacked symbols. Use for correctness-critical guarding decisions.
- _optimization_hint_base: permissive, uses heuristics and fallbacks for unbacked
  symbols. Use for performance optimization decisions.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import Any, TYPE_CHECKING

import sympy

from torch.utils._sympy.numbers import int_oo


log = logging.getLogger(__name__)

# Maximum number of free symbols in an expression before we skip
# sympy.factor() in optimization_hint process for unbacked.
# Factoring polynomials with many variables is expensive.
SYMPY_FACTOR_MAX_FREE_SYMBOLS = 50

# Maximum product of expression free symbols and non-symbol replacement count
# before optimization_hint skips expensive SymPy subs() canonicalization.
SYMPY_SUBS_MAX_SYMBOL_REPLACEMENT_PRODUCT = 1024

if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv


def _sympy_subs(expr: sympy.Basic, replacements: dict[sympy.Expr, Any]) -> sympy.Basic:
    """
    When the passed replacement symbol v is a string, it is converted to a symbol with name v that
    have the same replaced expression integer and nonnegative properties.
    """

    def to_symbol(replaced: sympy.Expr, replacement: sympy.Expr | str) -> sympy.Symbol:
        if not isinstance(replaced, sympy.Expr):
            raise AssertionError(
                f"Expected sympy.Expr key, got {type(replaced)}: {replaced}"
            )
        if isinstance(replacement, str):
            return sympy.Symbol(
                replacement,
                integer=replaced.is_integer,  # type: ignore[attr-defined]
                nonnegative=replaced.is_nonnegative,  # type: ignore[attr-defined]
            )
        else:
            return replacement

    # xreplace is faster than subs, but is way more picky
    return sympy.sympify(expr).xreplace(
        {k: to_symbol(k, v) for k, v in replacements.items()}
    )


def _maybe_realize_expr(
    expr: sympy.Basic, nan_fallback: int | None
) -> int | bool | None:
    """
    Handle special sympy values in hinting APIs.

    Returns:
        - True/False for sympy.true/sympy.false (preserves bool type)
        - Raises ValueError for complex numbers
        - sys.maxsize for positive infinity
        - -sys.maxsize for negative infinity
        - fallback for NaN
        - None if no special handling needed
    """
    if expr is sympy.true:
        return True
    if expr is sympy.false:
        return False

    try:
        return int(expr)
    except (TypeError, ValueError):
        pass

    if isinstance(expr, sympy.Expr):
