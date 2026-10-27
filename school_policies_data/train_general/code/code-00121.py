                "Failed to convert graphs from %s to '%s' backend for call to `%s'"
                + ("" if extra_message is None else ": %s"),
                input_backend_names,
                backend_name,
                self.name,
                *(() if extra_message is None else (exc,)),
            )
            if extra_message is not None:
                raise NotImplementedError(extra_message) from exc
            raise
        if backend_name != "networkx":
            _logger.debug(
                "Using backend '%s' for call to `%s' with arguments: %s",
                backend_name,
                self.name,
                _LazyArgsRepr(self, converted_args, converted_kwargs),
            )
        try:
            return func(*converted_args, **converted_kwargs)
        except NotImplementedError as exc:
            if extra_message is not None:
                _logger.debug(
                    "Backend '%s' raised when calling `%s': %s",
                    backend_name,
                    self.name,
                    exc,
                )
                raise NotImplementedError(extra_message) from exc
            raise

    def _convert_and_call_for_tests(
        self, backend_name, args, kwargs, *, fallback_to_nx=False
    ):
        """Call this dispatchable function with a backend; for use with testing."""
        backend = _load_backend(backend_name)
        if not self._can_backend_run(backend_name, args, kwargs):
            if fallback_to_nx or not self.graphs:
                if fallback_to_nx:
                    _logger.debug(
                        "Falling back to use 'networkx' instead of '%s' backend "
                        "for call to `%s' with arguments: %s",
                        backend_name,
                        self.name,
                        _LazyArgsRepr(self, args, kwargs),
                    )
                return self.orig_func(*args, **kwargs)

            import pytest

            msg = f"'{self.name}' not implemented by {backend_name}"
            if hasattr(backend, self.name):
                msg += " with the given arguments"
            pytest.xfail(msg)

        from collections.abc import Iterable, Iterator, Mapping
        from copy import copy, deepcopy
        from io import BufferedReader, BytesIO, StringIO, TextIOWrapper
        from itertools import tee
        from random import Random

        import numpy as np
        from numpy.random import Generator, RandomState
        from scipy.sparse import sparray

        # We sometimes compare the backend result to the original result,
        # so we need two sets of arguments. We tee iterators and copy
        # random state so that they may be used twice.
        if not args:
            args1 = args2 = args
        else:
            args1, args2 = zip(
                *(
                    (arg, deepcopy(arg))
                    if isinstance(arg, RandomState)
                    else (arg, copy(arg))
                    if isinstance(arg, BytesIO | StringIO | Random | Generator)
                    else tee(arg)
                    if isinstance(arg, Iterator)
                    and not isinstance(arg, BufferedReader | TextIOWrapper)
                    else (arg, arg)
                    for arg in args
                )
            )
        if not kwargs:
            kwargs1 = kwargs2 = kwargs
        else:
            kwargs1, kwargs2 = zip(
                *(
                    ((k, v), (k, deepcopy(v)))
                    if isinstance(v, RandomState)
                    else ((k, v), (k, copy(v)))
                    if isinstance(v, BytesIO | StringIO | Random | Generator)
                    else ((k, (teed := tee(v))[0]), (k, teed[1]))
                    if isinstance(v, Iterator)
                    and not isinstance(v, BufferedReader | TextIOWrapper)
                    else ((k, v), (k, v))
                    for k, v in kwargs.items()
                )
            )
            kwargs1 = dict(kwargs1)
            kwargs2 = dict(kwargs2)
        try:
            converted_args, converted_kwargs = self._convert_arguments(
                backend_name, args1, kwargs1, use_cache=False, mutations=None
            )
            _logger.debug(
                "Using backend '%s' for call to `%s' with arguments: %s",
                backend_name,
                self.name,
                _LazyArgsRepr(self, converted_args, converted_kwargs),
            )
            result = getattr(backend, self.name)(*converted_args, **converted_kwargs)
        except NotImplementedError as exc:
            if fallback_to_nx:
                _logger.debug(
                    "Graph conversion failed; falling back to use 'networkx' instead "
                    "of '%s' backend for call to `%s'",
                    backend_name,
                    self.name,
                )
                return self.orig_func(*args2, **kwargs2)
            import pytest

            pytest.xfail(
                exc.args[0] if exc.args else f"{self.name} raised {type(exc).__name__}"
            )
        # Verify that `self._returns_graph` is correct. This compares the return type
        # to the type expected from `self._returns_graph`. This handles tuple and list
        # return types, but *does not* catch functions that yield graphs.
        if (
            self._returns_graph
            != (
                isinstance(result, nx.Graph)
                or hasattr(result, "__networkx_backend__")
                or isinstance(result, tuple | list)
                and any(
                    isinstance(x, nx.Graph) or hasattr(x, "__networkx_backend__")
                    for x in result
                )
            )
            and not (
                # May return Graph or None
                self.name in {"check_planarity", "check_planarity_recursive"}
                and any(x is None for x in result)
            )
            and not (
                # May return Graph or dict
                self.name in {"held_karp_ascent"}
                and any(isinstance(x, dict) for x in result)
            )
            and self.name
            not in {
                # yields graphs
                "all_triads",
                "general_k_edge_subgraphs",
                # yields graphs or arrays
                "nonisomorphic_trees",
            }
        ):
            raise RuntimeError(f"`returns_graph` is incorrect for {self.name}")

        def check_result(val, depth=0):
            if isinstance(val, np.number):
                raise RuntimeError(
                    f"{self.name} returned a numpy scalar {val} ({type(val)}, depth={depth})"
                )
            if isinstance(val, np.ndarray | sparray):
                return
            if isinstance(val, nx.Graph):
                check_result(val._node, depth=depth + 1)
                check_result(val._adj, depth=depth + 1)
                return
            if isinstance(val, Iterator):
                raise NotImplementedError
            if isinstance(val, Iterable) and not isinstance(val, str):
                for x in val:
                    check_result(x, depth=depth + 1)
            if isinstance(val, Mapping):
                for x in val.values():
                    check_result(x, depth=depth + 1)

        def check_iterator(it):
            for val in it:
                try:
                    check_result(val)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"{self.name} returned a numpy scalar {val} ({type(val)})"
                    ) from exc
                yield val

        if self.name in {"from_edgelist"}:
            # numpy scalars are explicitly given as values in some tests
            pass
        elif isinstance(result, Iterator):
            result = check_iterator(result)
        else:
            try:
                check_result(result)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{self.name} returned a numpy scalar {result} ({type(result)})"
                ) from exc
            check_result(result)

        if self.name in {
            "edmonds_karp",
            "barycenter",
            "contracted_edge",
            "contracted_nodes",
            "stochastic_graph",
            "relabel_nodes",
            "maximum_branching",
            "incremental_closeness_centrality",
            "minimal_branching",
            "minimum_spanning_arborescence",
            "recursive_simple_cycles",
            "connected_double_edge_swap",
        }:
            # Special-case algorithms that mutate input graphs
            bound = self.__signature__.bind(*converted_args, **converted_kwargs)
            bound.apply_defaults()
            bound2 = self.__signature__.bind(*args2, **kwargs2)
            bound2.apply_defaults()
            if self.name in {
                "minimal_branching",
                "minimum_spanning_arborescence",
                "recursive_simple_cycles",
                "connected_double_edge_swap",
            }:
                G1 = backend.convert_to_nx(bound.arguments["G"])
                G2 = bound2.arguments["G"]
                G2._adj = G1._adj
                if G2.is_directed():
                    G2._pred = G1._pred
                nx._clear_cache(G2)
            elif self.name == "edmonds_karp":
                R1 = backend.convert_to_nx(bound.arguments["residual"])
                R2 = bound2.arguments["residual"]
                if R1 is not None and R2 is not None:
                    for k, v in R1.edges.items():
                        R2.edges[k]["flow"] = v["flow"]
                    R2.graph.update(R1.graph)
                    nx._clear_cache(R2)
            elif self.name == "barycenter" and bound.arguments["attr"] is not None:
                G1 = backend.convert_to_nx(bound.arguments["G"])
                G2 = bound2.arguments["G"]
                attr = bound.arguments["attr"]
                for k, v in G1.nodes.items():
                    G2.nodes[k][attr] = v[attr]
                nx._clear_cache(G2)
            elif (
                self.name in {"contracted_nodes", "contracted_edge"}
                and not bound.arguments["copy"]
            ):
                # Edges and nodes changed; node "contraction" and edge "weight" attrs
                G1 = backend.convert_to_nx(bound.arguments["G"])
                G2 = bound2.arguments["G"]
                G2.__dict__.update(G1.__dict__)
                nx._clear_cache(G2)
            elif self.name == "stochastic_graph" and not bound.arguments["copy"]:
                G1 = backend.convert_to_nx(bound.arguments["G"])
                G2 = bound2.arguments["G"]
                for k, v in G1.edges.items():
                    G2.edges[k]["weight"] = v["weight"]
                nx._clear_cache(G2)
            elif (
                self.name == "relabel_nodes"
                and not bound.arguments["copy"]
                or self.name in {"incremental_closeness_centrality"}
            ):
                G1 = backend.convert_to_nx(bound.arguments["G"])
                G2 = bound2.arguments["G"]
                if G1 is G2:
                    return G2
                G2._node.clear()
                G2._node.update(G1._node)
                G2._adj.clear()
                G2._adj.update(G1._adj)
                if hasattr(G1, "_pred") and hasattr(G2, "_pred"):
                    G2._pred.clear()
                    G2._pred.update(G1._pred)
                if hasattr(G1, "_succ") and hasattr(G2, "_succ"):
                    G2._succ.clear()
                    G2._succ.update(G1._succ)
                nx._clear_cache(G2)
                if self.name == "relabel_nodes":
                    return G2
            return backend.convert_to_nx(result)

        converted_result = backend.convert_to_nx(result)
        if isinstance(converted_result, nx.Graph) and self.name not in {
            "boykov_kolmogorov",
            "preflow_push",
            "quotient_graph",
            "shortest_augmenting_path",
            "spectral_graph_forge",
            # We don't handle tempfile.NamedTemporaryFile arguments
            "read_gml",
            "read_graph6",
            "read_sparse6",
            # We don't handle io.BufferedReader or io.TextIOWrapper arguments
            "bipartite_read_edgelist",
            "read_adjlist",
            "read_edgelist",
            "read_graphml",
            "read_multiline_adjlist",
            "read_pajek",
            "from_pydot",
            "pydot_read_dot",
            "agraph_read_dot",
            # graph comparison fails b/c of nan values
            "read_gexf",
        }:
            # For graph return types (e.g. generators), we compare that results are
            # the same between the backend and networkx, then return the original
            # networkx result so the iteration order will be consistent in tests.
            G = self.orig_func(*args2, **kwargs2)
            if not nx.utils.graphs_equal(G, converted_result):
                assert G.number_of_nodes() == converted_result.number_of_nodes()
                assert G.number_of_edges() == converted_result.number_of_edges()
                assert G.graph == converted_result.graph
                assert G.nodes == converted_result.nodes
                assert G.adj == converted_result.adj
                assert type(G) is type(converted_result)
                raise AssertionError("Graphs are not equal")
            return G
        return converted_result

    def _make_doc(self):
        """Generate the backends section at the end for functions having an alternate
        backend implementation(s) using the `backend_info` entry-point."""

        if not self.backends:
            return self._orig_doc
        lines = [
            "Backends",
            "--------",
        ]
        for backend in sorted(self.backends):
            info = backend_info[backend]
            if "short_summary" in info:
                lines.append(f"{backend} : {info['short_summary']}")
            else:
                lines.append(backend)
            if "functions" not in info or self.name not in info["functions"]:
                lines.append("")
                continue

            func_info = info["functions"][self.name]

            # Renaming extra_docstring to additional_docs
            if func_docs := (
                func_info.get("additional_docs") or func_info.get("extra_docstring")
            ):
                lines.extend(
                    f"  {line}" if line else line for line in func_docs.split("\n")
                )
                add_gap = True
            else:
                add_gap = False

            # Renaming extra_parameters to additional_parameters
            if extra_parameters := (
                func_info.get("extra_parameters")
                or func_info.get("additional_parameters")
            ):
                if add_gap:
                    lines.append("")
                lines.append("  Additional parameters:")
                for param in sorted(extra_parameters):
                    lines.append(f"    {param}")
                    if desc := extra_parameters[param]:
                        lines.append(f"      {desc}")
                    lines.append("")
            else:
                lines.append("")

            if func_url := func_info.get("url"):
                lines.append(f"[`Source <{func_url}>`_]")
                lines.append("")

        lines.pop()  # Remove last empty line
        to_add = "\n    ".join(lines)
        if not self._orig_doc:
            return f"The original docstring for {self.name} was empty.\n\n    {to_add}"
        return f"{self._orig_doc.rstrip()}\n\n    {to_add}"

    def __reduce__(self):
        """Allow this object to be serialized with pickle.

        This uses the global registry `_registered_algorithms` to deserialize.
        """
        return _restore_dispatchable, (self.name,)


def _restore_dispatchable(name):
    return _registered_algorithms[name].__wrapped__


def _get_cache_key(
    *,
    edge_attrs,
    node_attrs,
    preserve_edge_attrs,
    preserve_node_attrs,
    preserve_graph_attrs,
):
    """Return key used by networkx caching given arguments for ``convert_from_nx``."""
    # edge_attrs: dict | None
    # node_attrs: dict | None
    # preserve_edge_attrs: bool (False if edge_attrs is not None)
    # preserve_node_attrs: bool (False if node_attrs is not None)
    return (
        frozenset(edge_attrs.items())
        if edge_attrs is not None
        else preserve_edge_attrs,
        frozenset(node_attrs.items())
        if node_attrs is not None
        else preserve_node_attrs,
    )


def _get_from_cache(cache, key, *, backend_name=None, mutations=None):
    """Search the networkx cache for a graph that is compatible with ``key``.

    Parameters
    ----------
    cache : dict
        If ``backend_name`` is given, then this is treated as ``G.__networkx_cache__``,
        but if ``backend_name`` is None, then this is treated as the resolved inner
        cache such as ``G.__networkx_cache__["backends"][backend_name]``.
    key : tuple
        Cache key from ``_get_cache_key``.
    backend_name : str, optional
        Name of the backend to control how ``cache`` is interpreted.
    mutations : list, optional
        Used internally to clear objects gotten from cache if inputs will be mutated.

    Returns
    -------
    tuple or None
        The key of the compatible graph found in the cache.
    graph or None
        A compatible graph or None.
    """
    if backend_name is not None:
        cache = cache.get("backends", {}).get(backend_name, {})
    if not cache:
        return None, None

    # Do a simple search for a cached graph with compatible data.
    # For example, if we need a single attribute, then it's okay
    # to use a cached graph that preserved all attributes.
    # This looks for an exact match first.
    edge_key, node_key = key
    for compat_key in itertools.product(
        (edge_key, True) if edge_key is not True else (True,),
        (node_key, True) if node_key is not True else (True,),
    ):
        if (rv := cache.get(compat_key)) is not None:
            if mutations is not None:
                # Remove this item from the cache (after all conversions) if
                # the call to this dispatchable function will mutate an input.
                mutations.append((cache, compat_key))
            return compat_key, rv
    if edge_key is not True and node_key is not True:
        # Iterate over the items in `cache` to see if any are compatible.
        # For example, if no edge attributes are needed, then a graph
        # with any edge attribute will suffice. We use the same logic
        # below (but switched) to clear unnecessary items from the cache.
        # Use `list(cache.items())` to be thread-safe.
        for (ekey, nkey), graph in list(cache.items()):
            if edge_key is False or ekey is True:
                pass  # Cache works for edge data!
            elif edge_key is True or ekey is False or not edge_key.issubset(ekey):
                continue  # Cache missing required edge data; does not work
            if node_key is False or nkey is True:
                pass  # Cache works for node data!
            elif node_key is True or nkey is False or not node_key.issubset(nkey):
                continue  # Cache missing required node data; does not work
            if mutations is not None:
                # Remove this item from the cache (after all conversions) if
                # the call to this dispatchable function will mutate an input.
                mutations.append((cache, (ekey, nkey)))
            return (ekey, nkey), graph
    return None, None


def _set_to_cache(cache, key, graph, *, backend_name=None):
    """Set a backend graph to the cache, and remove unnecessary cached items.

    Parameters
    ----------
    cache : dict
        If ``backend_name`` is given, then this is treated as ``G.__networkx_cache__``,
        but if ``backend_name`` is None, then this is treated as the resolved inner
        cache such as ``G.__networkx_cache__["backends"][backend_name]``.
    key : tuple
        Cache key from ``_get_cache_key``.
    graph : graph
    backend_name : str, optional
        Name of the backend to control how ``cache`` is interpreted.

    Returns
    -------
    dict
        The items that were removed from the cache.
    """
    if backend_name is not None:
        cache = cache.setdefault("backends", {}).setdefault(backend_name, {})
    # Remove old cached items that are no longer necessary since they
    # are dominated/subsumed/outdated by what was just calculated.
    # This uses the same logic as above, but with keys switched.
    # Also, don't update the cache here if the call will mutate an input.
    removed = {}
    edge_key, node_key = key
    cache[key] = graph  # Set at beginning to be thread-safe
    for cur_key in list(cache):
        if cur_key == key:
            continue
        ekey, nkey = cur_key
        if ekey is False or edge_key is True:
            pass
        elif ekey is True or edge_key is False or not ekey.issubset(edge_key):
            continue
        if nkey is False or node_key is True:
            pass
        elif nkey is True or node_key is False or not nkey.issubset(node_key):
            continue
        # Use pop instead of del to try to be thread-safe
        if (graph := cache.pop(cur_key, None)) is not None:
            removed[cur_key] = graph
    return removed


class _LazyArgsRepr:
    """Simple wrapper to display arguments of dispatchable functions in logging calls."""

    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.value = None

    def __repr__(self):
        if self.value is None:
            bound = self.func.__signature__.bind_partial(*self.args, **self.kwargs)
            inner = ", ".join(f"{key}={val!r}" for key, val in bound.arguments.items())
            self.value = f"({inner})"
        return self.value


if os.environ.get("_NETWORKX_BUILDING_DOCS_"):
    # When building docs with Sphinx, use the original function with the
    # dispatched __doc__, b/c Sphinx renders normal Python functions better.
    # This doesn't show e.g. `*, backend=None, **backend_kwargs` in the
    # signatures, which is probably okay. It does allow the docstring to be
    # updated based on the installed backends.
    _orig_dispatchable = _dispatchable

    def _dispatchable(func=None, **kwargs):  # type: ignore[no-redef]
        if func is None:
            return partial(_dispatchable, **kwargs)
        dispatched_func = _orig_dispatchable(func, **kwargs)
        func.__doc__ = dispatched_func.__doc__
        return func

    _dispatchable.__doc__ = _orig_dispatchable.__new__.__doc__  # type: ignore[method-assign,assignment]
    _sig = inspect.signature(_orig_dispatchable.__new__)
    _dispatchable.__signature__ = _sig.replace(  # type: ignore[method-assign,assignment]
        parameters=[v for k, v in _sig.parameters.items() if k != "cls"]
    )
<|endoftext|>"""Priority queue class with updatable priorities."""

import heapq

__all__ = ["MappedQueue"]


class _HeapElement:
    """This proxy class separates the heap element from its priority.

    The idea is that using a 2-tuple (priority, element) works
    for sorting, but not for dict lookup because priorities are
    often floating point values so round-off can mess up equality.

    So, we need inequalities to look at the priority (for sorting)
    and equality (and hash) to look at the element to enable
    updates to the priority.

    Unfortunately, this class can be tricky to work with if you forget that
    `__lt__` compares the priority while `__eq__` compares the element.
    In `greedy_modularity_communities()` the following code is
    used to check that two _HeapElements differ in either element or priority:

        if d_oldmax != row_max or d_oldmax.priority != row_max.priority:

    If the priorities are the same, this implementation uses the element
    as a tiebreaker. This provides compatibility with older systems that
    use tuples to combine priority and elements.
    """

    __slots__ = ["priority", "element", "_hash"]

    def __init__(self, priority, element):
        self.priority = priority
        self.element = element
        self._hash = hash(element)

    def __lt__(self, other):
        try:
            other_priority = other.priority
        except AttributeError:
            return self.priority < other
        # assume comparing to another _HeapElement
        if self.priority == other_priority:
            try:
                return self.element < other.element
            except TypeError as err:
                raise TypeError(
                    "Consider using a tuple, with a priority value that can be compared."
                )
        return self.priority < other_priority

    def __gt__(self, other):
        try:
            other_priority = other.priority
        except AttributeError:
            return self.priority > other
        # assume comparing to another _HeapElement
        if self.priority == other_priority:
            try:
                return self.element > other.element
            except TypeError as err:
                raise TypeError(
                    "Consider using a tuple, with a priority value that can be compared."
                )
        return self.priority > other_priority

    def __eq__(self, other):
        try:
            return self.element == other.element
        except AttributeError:
            return self.element == other

    def __hash__(self):
        return self._hash

    def __getitem__(self, indx):
        return self.priority if indx == 0 else self.element[indx - 1]

    def __iter__(self):
        yield self.priority
        try:
            yield from self.element
        except TypeError:
            yield self.element

    def __repr__(self):
        return f"_HeapElement({self.priority}, {self.element})"


class MappedQueue:
    """The MappedQueue class implements a min-heap with removal and update-priority.

    The min heap uses heapq as well as custom written _siftup and _siftdown
    methods to allow the heap positions to be tracked by an additional dict
    keyed by element to position. The smallest element can be popped in O(1) time,
    new elements can be pushed in O(log n) time, and any element can be removed
    or updated in O(log n) time. The queue cannot contain duplicate elements
    and an attempt to push an element already in the queue will have no effect.

    MappedQueue complements the heapq package from the python standard
    library. While MappedQueue is designed for maximum compatibility with
    heapq, it adds element removal, lookup, and priority update.

    Parameters
    ----------
    data : dict or iterable

    Examples
    --------

    A `MappedQueue` can be created empty, or optionally, given a dictionary
    of initial elements and priorities.  The methods `push`, `pop`,
    `remove`, and `update` operate on the queue.

    >>> colors_nm = {"red": 665, "blue": 470, "green": 550}
    >>> q = MappedQueue(colors_nm)
    >>> q.remove("red")
    >>> q.update("green", "violet", 400)
    >>> q.push("indigo", 425)
    True
    >>> [q.pop().element for i in range(len(q.heap))]
    ['violet', 'indigo', 'blue']

    A `MappedQueue` can also be initialized with a list or other iterable. The priority is assumed
    to be the sort order of the items in the list.

    >>> q = MappedQueue([916, 50, 4609, 493, 237])
    >>> q.remove(493)
    >>> q.update(237, 1117)
    >>> [q.pop() for i in range(len(q.heap))]
    [50, 916, 1117, 4609]

    An exception is raised if the elements are not comparable.

    >>> q = MappedQueue([100, "a"])
    Traceback (most recent call last):
    ...
    TypeError: '<' not supported between instances of 'int' and 'str'

    To avoid the exception, use a dictionary to assign priorities to the elements.

    >>> q = MappedQueue({100: 0, "a": 1})

    References
    ----------
    .. [1] Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2001).
       Introduction to algorithms second edition.
    .. [2] Knuth, D. E. (1997). The art of computer programming (Vol. 3).
       Pearson Education.
    """

    def __init__(self, data=None):
        """Priority queue class with updatable priorities."""
        if data is None:
            self.heap = []
        elif isinstance(data, dict):
            self.heap = [_HeapElement(v, k) for k, v in data.items()]
        else:
            self.heap = list(data)
        self.position = {}
        self._heapify()

    def _heapify(self):
        """Restore heap invariant and recalculate map."""
        heapq.heapify(self.heap)
        self.position = {elt: pos for pos, elt in enumerate(self.heap)}
        if len(self.heap) != len(self.position):
            raise AssertionError("Heap contains duplicate elements")

    def __len__(self):
        return len(self.heap)

    def push(self, elt, priority=None):
        """Add an element to the queue."""
        if priority is not None:
            elt = _HeapElement(priority, elt)
        # If element is already in queue, do nothing
        if elt in self.position:
            return False
        # Add element to heap and dict
        pos = len(self.heap)
        self.heap.append(elt)
        self.position[elt] = pos
        # Restore invariant by sifting down
        self._siftdown(0, pos)
        return True

    def pop(self):
        """Remove and return the smallest element in the queue."""
        # Remove smallest element
        elt = self.heap[0]
        del self.position[elt]
        # If elt is last item, remove and return
        if len(self.heap) == 1:
            self.heap.pop()
            return elt
        # Replace root with last element
        last = self.heap.pop()
        self.heap[0] = last
        self.position[last] = 0
        # Restore invariant by sifting up
        self._siftup(0)
        # Return smallest element
        return elt

    def update(self, elt, new, priority=None):
        """Replace an element in the queue with a new one."""
        if priority is not None:
            new = _HeapElement(priority, new)
        # Replace
        pos = self.position[elt]
        self.heap[pos] = new
        del self.position[elt]
        self.position[new] = pos
        # Restore invariant by sifting up
        self._siftup(pos)

    def remove(self, elt):
        """Remove an element from the queue."""
        # Find and remove element
        try:
            pos = self.position[elt]
            del self.position[elt]
        except KeyError:
            # Not in queue
            raise
        # If elt is last item, remove and return
        if pos == len(self.heap) - 1:
            self.heap.pop()
            return
        # Replace elt with last element
        last = self.heap.pop()
        self.heap[pos] = last
        self.position[last] = pos
        # Restore invariant by sifting up
        self._siftup(pos)

    def _siftup(self, pos):
        """Move smaller child up until hitting a leaf.

        Built to mimic code for heapq._siftup
        only updating position dict too.
        """
        heap, position = self.heap, self.position
        end_pos = len(heap)
        startpos = pos
        newitem = heap[pos]
        # Shift up the smaller child until hitting a leaf
        child_pos = (pos << 1) + 1  # start with leftmost child position
        while child_pos < end_pos:
            # Set child_pos to index of smaller child.
            child = heap[child_pos]
            right_pos = child_pos + 1
            if right_pos < end_pos:
                right = heap[right_pos]
                if not child < right:
                    child = right
                    child_pos = right_pos
            # Move the smaller child up.
            heap[pos] = child
            position[child] = pos
            pos = child_pos
            child_pos = (pos << 1) + 1
        # pos is a leaf position. Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        while pos > 0:
            parent_pos = (pos - 1) >> 1
            parent = heap[parent_pos]
            if not newitem < parent:
                break
            heap[pos] = parent
            position[parent] = pos
            pos = parent_pos
        heap[pos] = newitem
        position[newitem] = pos

    def _siftdown(self, start_pos, pos):
        """Restore invariant. keep swapping with parent until smaller.

        Built to mimic code for heapq._siftdown
        only updating position dict too.
        """
        heap, position = self.heap, self.position
        newitem = heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > start_pos:
            parent_pos = (pos - 1) >> 1
            parent = heap[parent_pos]
            if not newitem < parent:
                break
            heap[pos] = parent
            position[parent] = pos
            pos = parent_pos
        heap[pos] = newitem
        position[newitem] = pos
<|endoftext|>"""
Utilities for generating random numbers, random sequences, and
random selections.
"""

import networkx as nx
from networkx.utils import py_random_state

__all__ = [
    "powerlaw_sequence",
    "zipf_rv",
    "cumulative_distribution",
    "discrete_sequence",
    "random_weighted_sample",
    "weighted_choice",
]


# The same helpers for choosing random sequences from distributions
# uses Python's random module
# https://docs.python.org/3/library/random.html


@py_random_state(2)
def powerlaw_sequence(n, exponent=2.0, seed=None):
    """
    Return sample sequence of length n from a power law distribution.
    """
    return [seed.paretovariate(exponent - 1) for i in range(n)]


@py_random_state(2)
def zipf_rv(alpha, xmin=1, seed=None):
    r"""Returns a random value chosen from the Zipf distribution.

    The return value is an integer drawn from the probability distribution

    .. math::

        p(x)=\frac{x^{-\alpha}}{\zeta(\alpha, x_{\min})},

    where $\zeta(\alpha, x_{\min})$ is the Hurwitz zeta function.

    Parameters
    ----------
    alpha : float
      Exponent value of the distribution
    xmin : int
      Minimum value
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    x : int
      Random value from Zipf distribution

    Raises
    ------
    ValueError:
      If xmin < 1 or
      If alpha <= 1

    Notes
    -----
    The rejection algorithm generates random values for a the power-law
    distribution in uniformly bounded expected time dependent on
    parameters.  See [1]_ for details on its operation.

    Examples
    --------
    >>> nx.utils.zipf_rv(alpha=2, xmin=3, seed=42)
    8

    References
    ----------
    .. [1] Luc Devroye, Non-Uniform Random Variate Generation,
       Springer-Verlag, New York, 1986.
    """
    if xmin < 1:
        raise ValueError("xmin < 1")
    if alpha <= 1:
        raise ValueError("a <= 1.0")
    a1 = alpha - 1.0
    b = 2**a1
    while True:
        u = 1.0 - seed.random()  # u in (0,1]
        v = seed.random()  # v in [0,1)
        x = int(xmin * u ** -(1.0 / a1))
        t = (1.0 + (1.0 / x)) ** a1
        if v * x * (t - 1.0) / (b - 1.0) <= t / b:
            break
    return x


def cumulative_distribution(distribution):
    """Returns normalized cumulative distribution from discrete distribution."""

    cdf = [0.0]
    psum = sum(distribution)
    for i in range(len(distribution)):
        cdf.append(cdf[i] + distribution[i] / psum)
    return cdf


@py_random_state(3)
def discrete_sequence(n, distribution=None, cdistribution=None, seed=None):
    """
    Return sample sequence of length n from a given discrete distribution
    or discrete cumulative distribution.

    One of the following must be specified.

    distribution = histogram of values, will be normalized

    cdistribution = normalized discrete cumulative distribution

    """
    import bisect

    if cdistribution is not None:
        cdf = cdistribution
    elif distribution is not None:
        cdf = cumulative_distribution(distribution)
    else:
        raise nx.NetworkXError(
            "discrete_sequence: distribution or cdistribution missing"
        )

    # get a uniform random number
    inputseq = [seed.random() for i in range(n)]

    # choose from CDF
    seq = [bisect.bisect_left(cdf, s) - 1 for s in inputseq]
    return seq


@py_random_state(2)
def random_weighted_sample(mapping, k, seed=None):
    """Returns k items without replacement from a weighted sample.

    The input is a dictionary of items with weights as values.
    """
    if k > len(mapping):
        raise ValueError("sample larger than population")
    sample = set()
    while len(sample) < k:
        sample.add(weighted_choice(mapping, seed))
    return list(sample)


@py_random_state(1)
def weighted_choice(mapping, seed=None):
    """Returns a single element from a weighted sample.

    The input is a dictionary of items with weights as values.
    """
    # use roulette method
    rnd = seed.random() * sum(mapping.values())
    for k, w in mapping.items():
        rnd -= w
        if rnd < 0:
            return k
<|endoftext|>"""
Min-heaps.
"""

from heapq import heappop, heappush
from itertools import count

import networkx as nx

__all__ = ["MinHeap", "PairingHeap", "BinaryHeap"]


class MinHeap:
    """Base class for min-heaps.

    A MinHeap stores a collection of key-value pairs ordered by their values.
    It supports querying the minimum pair, inserting a new pair, decreasing the
    value in an existing pair and deleting the minimum pair.
    """

    class _Item:
        """Used by subclassess to represent a key-value pair."""

        __slots__ = ("key", "value")

        def __init__(self, key, value):
            self.key = key
            self.value = value

        def __repr__(self):
            return repr((self.key, self.value))

    def __init__(self):
        """Initialize a new min-heap."""
        self._dict = {}

    def min(self):
        """Query the minimum key-value pair.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
        raise NotImplementedError

    def pop(self):
        """Delete the minimum pair in the heap.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
        raise NotImplementedError

    def get(self, key, default=None):
        """Returns the value associated with a key.

        Parameters
        ----------
        key : hashable object
            The key to be looked up.

        default : object
            Default value to return if the key is not present in the heap.
            Default value: None.

        Returns
        -------
        value : object.
            The value associated with the key.
        """
        raise NotImplementedError

    def insert(self, key, value, allow_increase=False):
        """Insert a new key-value pair or modify the value in an existing
        pair.

        Parameters
        ----------
        key : hashable object
            The key.

        value : object comparable with existing values.
            The value.

        allow_increase : bool
            Whether the value is allowed to increase. If False, attempts to
            increase an existing value have no effect. Default value: False.

        Returns
        -------
        decreased : bool
            True if a pair is inserted or the existing value is decreased.
        """
        raise NotImplementedError

    def __nonzero__(self):
        """Returns whether the heap if empty."""
        return bool(self._dict)

    def __bool__(self):
        """Returns whether the heap if empty."""
        return bool(self._dict)

    def __len__(self):
        """Returns the number of key-value pairs in the heap."""
        return len(self._dict)

    def __contains__(self, key):
        """Returns whether a key exists in the heap.

        Parameters
        ----------
        key : any hashable object.
            The key to be looked up.
        """
        return key in self._dict


class PairingHeap(MinHeap):
    """A pairing heap."""

    class _Node(MinHeap._Item):
        """A node in a pairing heap.

        A tree in a pairing heap is stored using the left-child, right-sibling
        representation.
        """

        __slots__ = ("left", "next", "prev", "parent")

        def __init__(self, key, value):
            super().__init__(key, value)
            # The leftmost child.
            self.left = None
            # The next sibling.
            self.next = None
            # The previous sibling.
            self.prev = None
            # The parent.
            self.parent = None

    def __init__(self):
        """Initialize a pairing heap."""
        super().__init__()
        self._root = None

    def min(self):
        if self._root is None:
            raise nx.NetworkXError("heap is empty.")
        return (self._root.key, self._root.value)

    def pop(self):
        if self._root is None:
            raise nx.NetworkXError("heap is empty.")
        min_node = self._root
        self._root = self._merge_children(self._root)
        del self._dict[min_node.key]
        return (min_node.key, min_node.value)

    def get(self, key, default=None):
        node = self._dict.get(key)
        return node.value if node is not None else default

    def insert(self, key, value, allow_increase=False):
        node = self._dict.get(key)
        root = self._root
        if node is not None:
            if value < node.value:
                node.value = value
                if node is not root and value < node.parent.value:
                    self._cut(node)
                    self._root = self._link(root, node)
                return True
            elif allow_increase and value > node.value:
                node.value = value
                child = self._merge_children(node)
                # Nonstandard step: Link the merged subtree with the root. See
                # below for the standard step.
                if child is not None:
                    self._root = self._link(self._root, child)
                # Standard step: Perform a decrease followed by a pop as if the
                # value were the smallest in the heap. Then insert the new
                # value into the heap.
                # if node is not root:
                #     self._cut(node)
                #     if child is not None:
                #         root = self._link(root, child)
                #     self._root = self._link(root, node)
                # else:
                #     self._root = (self._link(node, child)
                #                   if child is not None else node)
            return False
        else:
            # Insert a new key.
            node = self._Node(key, value)
            self._dict[key] = node
            self._root = self._link(root, node) if root is not None else node
            return True

    def _link(self, root, other):
        """Link two nodes, making the one with the smaller value the parent of
        the other.
        """
        if other.value < root.value:
            root, other = other, root
        next = root.left
        other.next = next
        if next is not None:
            next.prev = other
        other.prev = None
        root.left = other
        other.parent = root
        return root

    def _merge_children(self, root):
        """Merge the subtrees of the root using the standard two-pass method.
        The resulting subtree is detached from the root.
        """
        node = root.left
        root.left = None
        if node is not None:
            link = self._link
            # Pass 1: Merge pairs of consecutive subtrees from left to right.
            # At the end of the pass, only the prev pointers of the resulting
            # subtrees have meaningful values. The other pointers will be fixed
            # in pass 2.
            prev = None
            while True:
                next = node.next
                if next is None:
                    node.prev = prev
                    break
                next_next = next.next
                node = link(node, next)
                node.prev = prev
                prev = node
                if next_next is None:
                    break
                node = next_next
            # Pass 2: Successively merge the subtrees produced by pass 1 from
            # right to left with the rightmost one.
            prev = node.prev
            while prev is not None:
                prev_prev = prev.prev
                node = link(prev, node)
                prev = prev_prev
            # Now node can become the new root. Its has no parent nor siblings.
            node.prev = None
            node.next = None
            node.parent = None
        return node

    def _cut(self, node):
        """Cut a node from its parent."""
        prev = node.prev
        next = node.next
        if prev is not None:
            prev.next = next
        else:
            node.parent.left = next
        node.prev = None
        if next is not None:
            next.prev = prev
            node.next = None
        node.parent = None


class BinaryHeap(MinHeap):
    """A binary heap."""

    def __init__(self):
        """Initialize a binary heap."""
        super().__init__()
        self._heap = []
        self._count = count()

    def min(self):
        dict = self._dict
        if not dict:
            raise nx.NetworkXError("heap is empty")
        heap = self._heap
        pop = heappop
        # Repeatedly remove stale key-value pairs until a up-to-date one is
        # met.
        while True:
            value, _, key = heap[0]
            if key in dict and value == dict[key]:
                break
            pop(heap)
        return (key, value)

    def pop(self):
        dict = self._dict
        if not dict:
            raise nx.NetworkXError("heap is empty")
        heap = self._heap
        pop = heappop
        # Repeatedly remove stale key-value pairs until a up-to-date one is
        # met.
        while True:
            value, _, key = heap[0]
            pop(heap)
            if key in dict and value == dict[key]:
                break
        del dict[key]
        return (key, value)

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def insert(self, key, value, allow_increase=False):
        dict = self._dict
        if key in dict:
            old_value = dict[key]
            if value < old_value or (allow_increase and value > old_value):
                # Since there is no way to efficiently obtain the location of a
                # key-value pair in the heap, insert a new pair even if ones
                # with the same key may already be present. Deem the old ones
                # as stale and skip them when the minimum pair is queried.
                dict[key] = value
                heappush(self._heap, (value, next(self._count), key))
                return value < old_value
            return False
        else:
            dict[key] = value
            heappush(self._heap, (value, next(self._count), key))
            return True
<|endoftext|>import collections
import os
import typing
import warnings
from dataclasses import dataclass

__all__ = ["Config"]


@dataclass(init=False, eq=False, slots=True, kw_only=True, match_args=False)
class Config:
    """The base class for NetworkX configuration.

    There are two ways to use this to create configurations. The recommended way
    is to subclass ``Config`` with docs and annotations.

    >>> class MyConfig(Config):
    ...     '''Breakfast!'''
    ...
    ...     eggs: int
    ...     spam: int
    ...
    ...     def _on_setattr(self, key, value):
    ...         assert isinstance(value, int) and value >= 0
    ...         return value
    >>> cfg = MyConfig(eggs=1, spam=5)

    Another way is to simply pass the initial configuration as keyword arguments to
    the ``Config`` instance:

    >>> cfg1 = Config(eggs=1, spam=5)
    >>> cfg1
    Config(eggs=1, spam=5)

    Once defined, config items may be modified, but can't be added or deleted by default.
    ``Config`` is a ``Mapping``, and can get and set configs via attributes or brackets:

    >>> cfg.eggs = 2
    >>> cfg.eggs
    2
    >>> cfg["spam"] = 42
    >>> cfg["spam"]
    42

    For convenience, it can also set configs within a context with the "with" statement:

    >>> with cfg(spam=3):
    ...     print("spam (in context):", cfg.spam)
    spam (in context): 3
    >>> print("spam (after context):", cfg.spam)
    spam (after context): 42

    Subclasses may also define ``_on_setattr`` (as done in the example above)
    to ensure the value being assigned is valid:

    >>> cfg.spam = -1
    Traceback (most recent call last):
        ...
    AssertionError

    If a more flexible configuration object is needed that allows adding and deleting
    configurations, then pass ``strict=False`` when defining the subclass:

    >>> class FlexibleConfig(Config, strict=False):
    ...     default_greeting: str = "Hello"
    >>> flexcfg = FlexibleConfig()
    >>> flexcfg.name = "Mr. Anderson"
    >>> flexcfg
    FlexibleConfig(default_greeting='Hello', name='Mr. Anderson')
    """

    def __init_subclass__(cls, strict=True):
        cls._strict = strict

    def __new__(cls, **kwargs):
        orig_class = cls
        if cls is Config:
            # Enable the "simple" case of accepting config definition as keywords
            cls = type(
                cls.__name__,
                (cls,),
                {"__annotations__": {key: typing.Any for key in kwargs}},
            )
        cls = dataclass(
            eq=False,
            repr=cls._strict,
            slots=cls._strict,
            kw_only=True,
            match_args=False,
        )(cls)
        if not cls._strict:
            cls.__repr__ = _flexible_repr
        cls._orig_class = orig_class  # Save original class so we can pickle
        cls._prev = None  # Stage previous configs to enable use as context manager
        cls._context_stack = []  # Stack of previous configs when used as context
        instance = object.__new__(cls)
        instance.__init__(**kwargs)
        return instance

    def _on_setattr(self, key, value):
        """Process config value and check whether it is valid. Useful for subclasses."""
        return value

    def _on_delattr(self, key):
        """Callback for when a config item is being deleted. Useful for subclasses."""

    # Control behavior of attributes
    def __dir__(self):
        return self.__dataclass_fields__.keys()

    def __setattr__(self, key, value):
        if self._strict and key not in self.__dataclass_fields__:
            raise AttributeError(f"Invalid config name: {key!r}")
        value = self._on_setattr(key, value)
        object.__setattr__(self, key, value)
        self.__class__._prev = None

    def __delattr__(self, key):
        if self._strict:
            raise TypeError(
                f"Configuration items can't be deleted (can't delete {key!r})."
            )
        self._on_delattr(key)
        object.__delattr__(self, key)
        self.__class__._prev = None

    # Be a `collection.abc.Collection`
    def __contains__(self, key):
        return (
            key in self.__dataclass_fields__ if self._strict else key in self.__dict__
        )

    def __iter__(self):
        return iter(self.__dataclass_fields__ if self._strict else self.__dict__)

    def __len__(self):
        return len(self.__dataclass_fields__ if self._strict else self.__dict__)

    def __reversed__(self):
        return reversed(self.__dataclass_fields__ if self._strict else self.__dict__)

    # Add dunder methods for `collections.abc.Mapping`
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as err:
            raise KeyError(*err.args) from None

    def __setitem__(self, key, value):
        try:
            self.__setattr__(key, value)
        except AttributeError as err:
            raise KeyError(*err.args) from None

    def __delitem__(self, key):
        try:
            self.__delattr__(key)
        except AttributeError as err:
            raise KeyError(*err.args) from None

    _ipython_key_completions_ = __dir__  # config["<TAB>

    # Go ahead and make it a `collections.abc.Mapping`
    def get(self, key, default=None):
        return getattr(self, key, default)

    def items(self):
        return collections.abc.ItemsView(self)

    def keys(self):
        return collections.abc.KeysView(self)

    def values(self):
        return collections.abc.ValuesView(self)

    # dataclass can define __eq__ for us, but do it here so it works after pickling
    def __eq__(self, other):
        if not isinstance(other, Config):
            return NotImplemented
        return self._orig_class == other._orig_class and self.items() == other.items()

    # Make pickle work
    def __reduce__(self):
        return self._deserialize, (self._orig_class, dict(self))

    @staticmethod
    def _deserialize(cls, kwargs):
        return cls(**kwargs)

    # Allow to be used as context manager
    def __call__(self, **kwargs):
        kwargs = {key: self._on_setattr(key, val) for key, val in kwargs.items()}
        prev = dict(self)
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.__class__._prev = prev
        return self

    def __enter__(self):
        if self.__class__._prev is None:
            raise RuntimeError(
                "Config being used as a context manager without config items being set. "
                "Set config items via keyword arguments when calling the config object. "
                "For example, using config as a context manager should be like:\n\n"
                '    >>> with cfg(breakfast="spam"):\n'
                "    ...     ...  # Do stuff\n"
            )
        self.__class__._context_stack.append(self.__class__._prev)
        self.__class__._prev = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        prev = self.__class__._context_stack.pop()
        for key, val in prev.items():
            setattr(self, key, val)


def _flexible_repr(self):
    return (
        f"{self.__class__.__qualname__}("
        + ", ".join(f"{key}={val!r}" for key, val in self.__dict__.items())
        + ")"
    )


# Register, b/c `Mapping.__subclasshook__` returns `NotImplemented`
collections.abc.Mapping.register(Config)


class BackendPriorities(Config, strict=False):
    """Configuration to control automatic conversion to and calling of backends.

    Priority is given to backends listed earlier.

    Parameters
    ----------
    algos : list of backend names
        This controls "algorithms" such as ``nx.pagerank`` that don't return a graph.
    generators : list of backend names
        This controls "generators" such as ``nx.from_pandas_edgelist`` that return a graph.
    kwargs : variadic keyword arguments of function name to list of backend names
        This allows each function to be configured separately and will override the config
        in ``algos`` or ``generators`` if present. The dispatchable function name may be
        gotten from the ``.name`` attribute such as ``nx.pagerank.name`` (it's typically
        the same as the name of the function).
    """

    algos: list[str]
    generators: list[str]

    def _on_setattr(self, key, value):
        from .backends import _registered_algorithms, backend_info

        if key in {"algos", "generators"}:
            pass
        elif key not in _registered_algorithms:
            raise AttributeError(
                f"Invalid config name: {key!r}. Expected 'algos', 'generators', or a name "
                "of a dispatchable function (e.g. `.name` attribute of the function)."
            )
        if not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
            raise TypeError(
                f"{key!r} config must be a list of backend names; got {value!r}"
            )
        if missing := {x for x in value if x not in backend_info}:
            missing = ", ".join(map(repr, sorted(missing)))
            raise ValueError(f"Unknown backend when setting {key!r}: {missing}")
        return value

    def _on_delattr(self, key):
        if key in {"algos", "generators"}:
            raise TypeError(f"{key!r} configuration item can't be deleted.")


class NetworkXConfig(Config):
    """Configuration for NetworkX that controls behaviors such as how to use backends.

    Attribute and bracket notation are supported for getting and setting configurations::

        >>> nx.config.backend_priority == nx.config["backend_priority"]
        True

    Parameters
    ----------
    backend_priority : list of backend names or dict or BackendPriorities
        Enable automatic conversion of graphs to backend graphs for functions
        implemented by the backend. Priority is given to backends listed earlier.
        This is a nested configuration with keys ``algos``, ``generators``, and,
        optionally, function names. Setting this value to a list of backend names
        will set ``nx.config.backend_priority.algos``. For more information, see
        ``help(nx.config.backend_priority)``. Default is empty list.

    backends : Config mapping of backend names to backend Config
        The keys of the Config mapping are names of all installed NetworkX backends,
        and the values are their configurations as Config mappings.

    cache_converted_graphs : bool
        If True, then save converted graphs to the cache of the input graph. Graph
        conversion may occur when automatically using a backend from `backend_priority`
        or when using the `backend=` keyword argument to a function call. Caching can
        improve performance by avoiding repeated conversions, but it uses more memory.
        Care should be taken to not manually mutate a graph that has cached graphs; for
        example, ``G[u][v][k] = val`` changes the graph, but does not clear the cache.
        Using methods such as ``G.add_edge(u, v, weight=val)`` will clear the cache to
        keep it consistent. ``G.__networkx_cache__.clear()`` manually clears the cache.
        Default is True.

    fallback_to_nx : bool
        If True, then "fall back" and run with the default "networkx" implementation
        for dispatchable functions not implemented by backends of input graphs. When a
        backend graph is passed to a dispatchable function, the default behavior is to
        use the implementation from that backend if possible and raise if not. Enabling
        ``fallback_to_nx`` makes the networkx implementation the fallback to use instead
        of raising, and will convert the backend graph to a networkx-compatible graph.
        Default is False.

    warnings_to_ignore : set of strings
        Control which warnings from NetworkX are not emitted. Valid elements:

        - `"cache"`: when a cached value is used from ``G.__networkx_cache__``.

    Notes
    -----
    Environment variables may be used to control some default configurations:

    - ``NETWORKX_BACKEND_PRIORITY``: set ``backend_priority.algos`` from comma-separated names.
    - ``NETWORKX_CACHE_CONVERTED_GRAPHS``: set ``cache_converted_graphs`` to True if nonempty.
    - ``NETWORKX_FALLBACK_TO_NX``: set ``fallback_to_nx`` to True if nonempty.
    - ``NETWORKX_WARNINGS_TO_IGNORE``: set `warnings_to_ignore` from comma-separated names.

    and can be used for finer control of ``backend_priority`` such as:

    - ``NETWORKX_BACKEND_PRIORITY_ALGOS``: same as ``NETWORKX_BACKEND_PRIORITY`` to set ``backend_priority.algos``.

    This is a global configuration. Use with caution when using from multiple threads.
    """

    backend_priority: BackendPriorities
    backends: Config
    cache_converted_graphs: bool
    fallback_to_nx: bool
    warnings_to_ignore: set[str]

    def _on_setattr(self, key, value):
        from .backends import backend_info

        if key == "backend_priority":
            if isinstance(value, list):
                getattr(self, key).algos = value
                value = getattr(self, key)
            elif isinstance(value, dict):
                kwargs = value
                value = BackendPriorities(algos=[], generators=[])
                for key, val in kwargs.items():
                    setattr(value, key, val)
            elif not isinstance(value, BackendPriorities):
                raise TypeError(
                    f"{key!r} config must be a dict of lists of backend names; got {value!r}"
                )
        elif key == "backends":
            if not (
                isinstance(value, Config)
                and all(isinstance(key, str) for key in value)
                and all(isinstance(val, Config) for val in value.values())
            ):
                raise TypeError(
                    f"{key!r} config must be a Config of backend configs; got {value!r}"
                )
            if missing := {x for x in value if x not in backend_info}:
                missing = ", ".join(map(repr, sorted(missing)))
                raise ValueError(f"Unknown backend when setting {key!r}: {missing}")
        elif key in {"cache_converted_graphs", "fallback_to_nx"}:
            if not isinstance(value, bool):
                raise TypeError(f"{key!r} config must be True or False; got {value!r}")
        elif key == "warnings_to_ignore":
            if not (isinstance(value, set) and all(isinstance(x, str) for x in value)):
                raise TypeError(
                    f"{key!r} config must be a set of warning names; got {value!r}"
                )
            known_warnings = {"cache"}
            if missing := {x for x in value if x not in known_warnings}:
                missing = ", ".join(map(repr, sorted(missing)))
                raise ValueError(
                    f"Unknown warning when setting {key!r}: {missing}. Valid entries: "
                    + ", ".join(sorted(known_warnings))
                )
        return value
<|endoftext|>r"""
*****
LaTeX
*****

Export NetworkX graphs in LaTeX format using the TikZ library within TeX/LaTeX.
Usually, you will want the drawing to appear in a figure environment so
you use ``to_latex(G, caption="A caption")``. If you want the raw
drawing commands without a figure environment use :func:`to_latex_raw`.
And if you want to write to a file instead of just returning the latex
code as a string, use ``write_latex(G, "filename.tex", caption="A caption")``.

To construct a figure with subfigures for each graph to be shown, provide
``to_latex`` or ``write_latex`` a list of graphs, a list of subcaptions,
and a number of rows of subfigures inside the figure.

To be able to refer to the figures or subfigures in latex using ``\\ref``,
the keyword ``latex_label`` is available for figures and `sub_labels` for
a list of labels, one for each subfigure.

We intend to eventually provide an interface to the TikZ Graph
features which include e.g. layout algorithms.

Let us know via github what you'd like to see available, or better yet
give us some code to do it, or even better make a github pull request
to add the feature.

The TikZ approach
=================
Drawing options can be stored on the graph as node/edge attributes, or
can be provided as dicts keyed by node/edge to a string of the options
for that node/edge. Similarly a label can be shown for each node/edge
by specifying the labels as graph node/edge attributes or by providing
a dict keyed by node/edge to the text to be written for that node/edge.

Options for the tikzpicture environment (e.g. "[scale=2]") can be provided
via a keyword argument. Similarly default node and edge options can be
provided through keywords arguments. The default node options are applied
to the single TikZ "path" that draws all nodes (and no edges). The default edge
options are applied to a TikZ "scope" which contains a path for each edge.

Examples
========
>>> G = nx.path_graph(3)
>>> nx.write_latex(G, "just_my_figure.tex", as_document=True)
>>> nx.write_latex(G, "my_figure.tex", caption="A path graph", latex_label="fig1")
>>> latex_code = nx.to_latex(G)  # a string rather than a file

You can change many features of the nodes and edges.

>>> G = nx.path_graph(4, create_using=nx.DiGraph)
>>> pos = {n: (n, n) for n in G}  # nodes set on a line

>>> G.nodes[0]["style"] = "blue"
>>> G.nodes[2]["style"] = "line width=3,draw"
>>> G.nodes[3]["label"] = "Stop"
>>> G.edges[(0, 1)]["label"] = "1st Step"
>>> G.edges[(0, 1)]["label_opts"] = "near start"
>>> G.edges[(1, 2)]["style"] = "line width=3"
>>> G.edges[(1, 2)]["label"] = "2nd Step"
>>> G.edges[(2, 3)]["style"] = "green"
>>> G.edges[(2, 3)]["label"] = "3rd Step"
>>> G.edges[(2, 3)]["label_opts"] = "near end"

>>> nx.write_latex(G, "latex_graph.tex", pos=pos, as_document=True)

Then compile the LaTeX using something like ``pdflatex latex_graph.tex``
and view the pdf file created: ``latex_graph.pdf``.

If you want **subfigures** each containing one graph, you can input a list of graphs.

>>> H1 = nx.path_graph(4)
>>> H2 = nx.complete_graph(4)
>>> H3 = nx.path_graph(8)
>>> H4 = nx.complete_graph(8)
>>> graphs = [H1, H2, H3, H4]
>>> caps = ["Path 4", "Complete graph 4", "Path 8", "Complete graph 8"]
>>> lbls = ["fig2a", "fig2b", "fig2c", "fig2d"]
>>> nx.write_latex(graphs, "subfigs.tex", n_rows=2, sub_captions=caps, sub_labels=lbls)
>>> latex_code = nx.to_latex(graphs, n_rows=2, sub_captions=caps, sub_labels=lbls)

>>> node_color = {0: "red", 1: "orange", 2: "blue", 3: "gray!90"}
>>> edge_width = {e: "line width=1.5" for e in H3.edges}
>>> pos = nx.circular_layout(H3)
>>> latex_code = nx.to_latex(H3, pos, node_options=node_color, edge_options=edge_width)
>>> print(latex_code)
\documentclass{report}
\usepackage{tikz}
\usepackage{subcaption}
<BLANKLINE>
\begin{document}
\begin{figure}
  \begin{tikzpicture}
      \draw
        (1.0, 0.0) node[red] (0){0}
        (0.707, 0.707) node[orange] (1){1}
        (-0.0, 1.0) node[blue] (2){2}
        (-0.707, 0.707) node[gray!90] (3){3}
        (-1.0, -0.0) node (4){4}
        (-0.707, -0.707) node (5){5}
        (0.0, -1.0) node (6){6}
        (0.707, -0.707) node (7){7};
      \begin{scope}[-]
        \draw[line width=1.5] (0) to (1);
        \draw[line width=1.5] (1) to (2);
        \draw[line width=1.5] (2) to (3);
        \draw[line width=1.5] (3) to (4);
        \draw[line width=1.5] (4) to (5);
        \draw[line width=1.5] (5) to (6);
        \draw[line width=1.5] (6) to (7);
      \end{scope}
    \end{tikzpicture}
\end{figure}
\end{document}

Notes
-----
If you want to change the preamble/postamble of the figure/document/subfigure
environment, use the keyword arguments: `figure_wrapper`, `document_wrapper`,
`subfigure_wrapper`. The default values are stored in private variables
e.g. ``nx.nx_layout._DOCUMENT_WRAPPER``

References
----------
TikZ:          https://tikz.dev/

TikZ options details:   https://tikz.dev/tikz-actions
"""

import numbers
import os

import networkx as nx

__all__ = [
    "to_latex_raw",
    "to_latex",
    "write_latex",
]


@nx.utils.not_implemented_for("multigraph")
def to_latex_raw(
    G,
    pos="pos",
    tikz_options="",
    default_node_options="",
    node_options="node_options",
    node_label="label",
    default_edge_options="",
    edge_options="edge_options",
    edge_label="label",
    edge_label_options="edge_label_options",
):
    """Return a string of the LaTeX/TikZ code to draw `G`

    This function produces just the code for the tikzpicture
    without any enclosing environment.

    Parameters
    ==========
    G : NetworkX graph
        The NetworkX graph to be drawn
    pos : string or dict (default "pos")
        The name of the node attribute on `G` that holds the position of each node.
        Positions can be sequences of length 2 with numbers for (x,y) coordinates.
        They can also be strings to denote positions in TikZ style, such as (x, y)
        or (angle:radius).
        If a dict, it should be keyed by node to a position.
        If an empty dict, a circular layout is computed by TikZ.
    tikz_options : string
        The tikzpicture options description defining the options for the picture.
        Often large scale options like `[scale=2]`.
    default_node_options : string
        The draw options for a path of nodes. Individual node options override these.
    node_options : string or dict
        The name of the node attribute on `G` that holds the options for each node.
        Or a dict keyed by node to a string holding the options for that node.
    node_label : string or dict
        The name of the node attribute on `G` that holds the node label (text)
        displayed for each node. If the attribute is "" or not present, the node
        itself is drawn as a string. LaTeX processing such as ``"$A_1$"`` is allowed.
        Or a dict keyed by node to a string holding the label for that node.
    default_edge_options : string
        The options for the scope drawing all edges. The default is "[-]" for
        undirected graphs and "[->]" for directed graphs.
    edge_options : string or dict
        The name of the edge attribute on `G` that holds the options for each edge.
        If the edge is a self-loop and ``"loop" not in edge_options`` the option
        "loop," is added to the options for the self-loop edge. Hence you can
        use "[loop above]" explicitly, but the default is "[loop]".
        Or a dict keyed by edge to a string holding the options for that edge.
    edge_label : string or dict
        The name of the edge attribute on `G` that holds the edge label (text)
        displayed for each edge. If the attribute is "" or not present, no edge
        label is drawn.
        Or a dict keyed by edge to a string holding the label for that edge.
    edge_label_options : string or dict
        The name of the edge attribute on `G` that holds the label options for
        each edge. For example, "[sloped,above,blue]". The default is no options.
        Or a dict keyed by edge to a string holding the label options for that edge.

    Returns
    =======
    latex_code : string
       The text string which draws the desired graph(s) when compiled by LaTeX.

    See Also
    ========
    to_latex
    write_latex
    """
    i4 = "\n    "
    i8 = "\n        "

    # set up position dict
    # TODO allow pos to be None and use a nice TikZ default
    if not isinstance(pos, dict):
        pos = nx.get_node_attributes(G, pos)
    if not pos:
        # circular layout with radius 2
        pos = {n: f"({round(360.0 * i / len(G), 3)}:2)" for i, n in enumerate(G)}
    for node in G:
        if node not in pos:
            raise nx.NetworkXError(f"node {node} has no specified pos {pos}")
        posnode = pos[node]
        if not isinstance(posnode, str):
            try:
                posx, posy = posnode
                pos[node] = f"({round(posx, 3)}, {round(posy, 3)})"
            except (TypeError, ValueError):
                msg = f"position pos[{node}] is not 2-tuple or a string: {posnode}"
                raise nx.NetworkXError(msg)

    # set up all the dicts
    if not isinstance(node_options, dict):
        node_options = nx.get_node_attributes(G, node_options)
    if not isinstance(node_label, dict):
        node_label = nx.get_node_attributes(G, node_label)
    if not isinstance(edge_options, dict):
        edge_options = nx.get_edge_attributes(G, edge_options)
    if not isinstance(edge_label, dict):
        edge_label = nx.get_edge_attributes(G, edge_label)
    if not isinstance(edge_label_options, dict):
        edge_label_options = nx.get_edge_attributes(G, edge_label_options)

    # process default options (add brackets or not)
    topts = "" if tikz_options == "" else f"[{tikz_options.strip('[]')}]"
    defn = "" if default_node_options == "" else f"[{default_node_options.strip('[]')}]"
    linestyle = f"{'->' if G.is_directed() else '-'}"
    if default_edge_options == "":
        defe = "[" + linestyle + "]"
    elif "-" in default_edge_options:
        defe = default_edge_options
    else:
        defe = f"[{linestyle},{default_edge_options.strip('[]')}]"

    # Construct the string line by line
    result = "  \\begin{tikzpicture}" + topts
    result += i4 + "  \\draw" + defn
    # load the nodes
    for n in G:
        # node options goes inside square brackets
        nopts = f"[{node_options[n].strip('[]')}]" if n in node_options else ""
        # node text goes inside curly brackets {}
        ntext = f"{{{node_label[n]}}}" if n in node_label else f"{{{n}}}"

        result += i8 + f"{pos[n]} node{nopts} ({n}){ntext}"
    result += ";\n"

    # load the edges
    result += "      \\begin{scope}" + defe
    for edge in G.edges:
        u, v = edge[:2]
        e_opts = f"{edge_options[edge]}".strip("[]") if edge in edge_options else ""
        # add loop options for selfloops if not present
        if u == v and "loop" not in e_opts:
            e_opts = "loop," + e_opts
        e_opts = f"[{e_opts}]" if e_opts != "" else ""
        # TODO -- handle bending of multiedges

        els = edge_label_options[edge] if edge in edge_label_options else ""
        # edge label options goes inside square brackets []
        els = f"[{els.strip('[]')}]"
        # edge text is drawn using the TikZ node command inside curly brackets {}
        e_label = f" node{els} {{{edge_label[edge]}}}" if edge in edge_label else ""

        result += i8 + f"\\draw{e_opts} ({u}) to{e_label} ({v});"

    result += "\n      \\end{scope}\n    \\end{tikzpicture}\n"
    return result


_DOC_WRAPPER_TIKZ = r"""\documentclass{{report}}
\usepackage{{tikz}}
\usepackage{{subcaption}}

\begin{{document}}
{content}
\end{{document}}"""


_FIG_WRAPPER = r"""\begin{{figure}}
{content}{caption}{label}
\end{{figure}}"""


_SUBFIG_WRAPPER = r"""  \begin{{subfigure}}{{{size}\textwidth}}
{content}{caption}{label}
  \end{{subfigure}}"""


def to_latex(
    Gbunch,
    pos="pos",
    tikz_options="",
    default_node_options="",
    node_options="node_options",
    node_label="node_label",
    default_edge_options="",
    edge_options="edge_options",
    edge_label="edge_label",
    edge_label_options="edge_label_options",
    caption="",
    latex_label="",
    sub_captions=None,
    sub_labels=None,
    n_rows=1,
    as_document=True,
    document_wrapper=_DOC_WRAPPER_TIKZ,
    figure_wrapper=_FIG_WRAPPER,
    subfigure_wrapper=_SUBFIG_WRAPPER,
):
    """Return latex code to draw the graph(s) in `Gbunch`

    The TikZ drawing utility in LaTeX is used to draw the graph(s).
    If `Gbunch` is a graph, it is drawn in a figure environment.
    If `Gbunch` is an iterable of graphs, each is drawn in a subfigure environment
    within a single figure environment.

    If `as_document` is True, the figure is wrapped inside a document environment
    so that the resulting string is ready to be compiled by LaTeX. Otherwise,
    the string is ready for inclusion in a larger tex document using ``\\include``
    or ``\\input`` statements.

    Parameters
    ==========
    Gbunch : NetworkX graph or iterable of NetworkX graphs
        The NetworkX graph to be drawn or an iterable of graphs
        to be drawn inside subfigures of a single figure.
    pos : string or list of strings
        The name of the node attribute on `G` that holds the position of each node.
        Positions can be sequences of length 2 with numbers for (x,y) coordinates.
        They can also be strings to denote positions in TikZ style, such as (x, y)
        or (angle:radius).
        If a dict, it should be keyed by node to a position.
        If an empty dict, a circular layout is computed by TikZ.
        If you are drawing many graphs in subfigures, use a list of position dicts.
    tikz_options : string
        The tikzpicture options description defining the options for the picture.
        Often large scale options like `[scale=2]`.
    default_node_options : string
        The draw options for a path of nodes. Individual node options override these.
    node_options : string or dict
        The name of the node attribute on `G` that holds the options for each node.
        Or a dict keyed by node to a string holding the options for that node.
    node_label : string or dict
        The name of the node attribute on `G` that holds the node label (text)
        displayed for each node. If the attribute is "" or not present, the node
        itself is drawn as a string. LaTeX processing such as ``"$A_1$"`` is allowed.
        Or a dict keyed by node to a string holding the label for that node.
    default_edge_options : string
        The options for the scope drawing all edges. The default is "[-]" for
        undirected graphs and "[->]" for directed graphs.
    edge_options : string or dict
        The name of the edge attribute on `G` that holds the options for each edge.
        If the edge is a self-loop and ``"loop" not in edge_options`` the option
        "loop," is added to the options for the self-loop edge. Hence you can
        use "[loop above]" explicitly, but the default is "[loop]".
        Or a dict keyed by edge to a string holding the options for that edge.
    edge_label : string or dict
        The name of the edge attribute on `G` that holds the edge label (text)
        displayed for each edge. If the attribute is "" or not present, no edge
        label is drawn.
        Or a dict keyed by edge to a string holding the label for that edge.
    edge_label_options : string or dict
        The name of the edge attribute on `G` that holds the label options for
        each edge. For example, "[sloped,above,blue]". The default is no options.
        Or a dict keyed by edge to a string holding the label options for that edge.
    caption : string
        The caption string for the figure environment
    latex_label : string
        The latex label used for the figure for easy referral from the main text
    sub_captions : list of strings
        The sub_caption string for each subfigure in the figure
    sub_latex_labels : list of strings
        The latex label for each subfigure in the figure
    n_rows : int
        The number of rows of subfigures to arrange for multiple graphs
    as_document : bool
        Whether to wrap the latex code in a document environment for compiling
    document_wrapper : formatted text string with variable ``content``.
        This text is called to evaluate the content embedded in a document
        environment with a preamble setting up TikZ.
    figure_wrapper : formatted text string
        This text is evaluated with variables ``content``, ``caption`` and ``label``.
        It wraps the content and if a caption is provided, adds the latex code for
        that caption, and if a label is provided, adds the latex code for a label.
    subfigure_wrapper : formatted text string
        This text evaluate variables ``size``, ``content``, ``caption`` and ``label``.
        It wraps the content and if a caption is provided, adds the latex code for
        that caption, and if a label is provided, adds the latex code for a label.
        The size is the vertical size of each row of subfigures as a fraction.

    Returns
    =======
    latex_code : string
        The text string which draws the desired graph(s) when compiled by LaTeX.

    See Also
    ========
    write_latex
    to_latex_raw
    """
    if hasattr(Gbunch, "adj"):
        raw = to_latex_raw(
            Gbunch,
            pos,
            tikz_options,
            default_node_options,
            node_options,
            node_label,
            default_edge_options,
            edge_options,
            edge_label,
            edge_label_options,
        )
    else:  # iterator of graphs
        sbf = subfigure_wrapper
        size = 1 / n_rows

        N = len(Gbunch)
        if isinstance(pos, str | dict):
            pos = [pos] * N
        if sub_captions is None:
            sub_captions = [""] * N
        if sub_labels is None:
            sub_labels = [""] * N
        if not (len(Gbunch) == len(pos) == len(sub_captions) == len(sub_labels)):
            raise nx.NetworkXError(
                "length of Gbunch, sub_captions and sub_figures must agree"
            )

        raw = ""
        for G, pos, subcap, sublbl in zip(Gbunch, pos, sub_captions, sub_labels):
            subraw = to_latex_raw(
                G,
                pos,
                tikz_options,
                default_node_options,
                node_options,
                node_label,
                default_edge_options,
                edge_options,
                edge_label,
                edge_label_options,
            )
            cap = f"    \\caption{{{subcap}}}" if subcap else ""
            lbl = f"\\label{{{sublbl}}}" if sublbl else ""
            raw += sbf.format(size=size, content=subraw, caption=cap, label=lbl)
            raw += "\n"

    # put raw latex code into a figure environment and optionally into a document
    raw = raw[:-1]
    cap = f"\n  \\caption{{{caption}}}" if caption else ""
    lbl = f"\\label{{{latex_label}}}" if latex_label else ""
    fig = figure_wrapper.format(content=raw, caption=cap, label=lbl)
    if as_document:
        return document_wrapper.format(content=fig)
    return fig


@nx.utils.open_file(1, mode="w")
def write_latex(Gbunch, path, **options):
    """Write the latex code to draw the graph(s) onto `path`.

    This convenience function creates the latex drawing code as a string
    and writes that to a file ready to be compiled when `as_document` is True
    or ready to be ``import`` ed or ``include`` ed into your main LaTeX document.

    The `path` argument can be a string filename or a file handle to write to.

    Parameters
    ----------
    Gbunch : NetworkX graph or iterable of NetworkX graphs
        If Gbunch is a graph, it is drawn in a figure environment.
        If Gbunch is an iterable of graphs, each is drawn in a subfigure
        environment within a single figure environment.
    path : filename
        Filename or file handle to write to
    options : dict
        By default, TikZ is used with options: (others are ignored)::

            pos : string or dict or list
                The name of the node attribute on `G` that holds the position of each node.
                Positions can be sequences of length 2 with numbers for (x,y) coordinates.
                They can also be strings to denote positions in TikZ style, such as (x, y)
                or (angle:radius).
                If a dict, it should be keyed by node to a position.
                If an empty dict, a circular layout is computed by TikZ.
                If you are drawing many graphs in subfigures, use a list of position dicts.
            tikz_options : string
                The tikzpicture options description defining the options for the picture.
                Often large scale options like `[scale=2]`.
            default_node_options : string
                The draw options for a path of nodes. Individual node options override these.
            node_options : string or dict
                The name of the node attribute on `G` that holds the options for each node.
                Or a dict keyed by node to a string holding the options for that node.
            node_label : string or dict
                The name of the node attribute on `G` that holds the node label (text)
                displayed for each node. If the attribute is "" or not present, the node
                itself is drawn as a string. LaTeX processing such as ``"$A_1$"`` is allowed.
                Or a dict keyed by node to a string holding the label for that node.
            default_edge_options : string
                The options for the scope drawing all edges. The default is "[-]" for
                undirected graphs and "[->]" for directed graphs.
            edge_options : string or dict
                The name of the edge attribute on `G` that holds the options for each edge.
                If the edge is a self-loop and ``"loop" not in edge_options`` the option
                "loop," is added to the options for the self-loop edge. Hence you can
                use "[loop above]" explicitly, but the default is "[loop]".
                Or a dict keyed by edge to a string holding the options for that edge.
            edge_label : string or dict
                The name of the edge attribute on `G` that holds the edge label (text)
                displayed for each edge. If the attribute is "" or not present, no edge
                label is drawn.
                Or a dict keyed by edge to a string holding the label for that edge.
            edge_label_options : string or dict
                The name of the edge attribute on `G` that holds the label options for
                each edge. For example, "[sloped,above,blue]". The default is no options.
                Or a dict keyed by edge to a string holding the label options for that edge.
            caption : string
                The caption string for the figure environment
            latex_label : string
                The latex label used for the figure for easy referral from the main text
            sub_captions : list of strings
                The sub_caption string for each subfigure in the figure
            sub_latex_labels : list of strings
                The latex label for each subfigure in the figure
            n_rows : int
                The number of rows of subfigures to arrange for multiple graphs
            as_document : bool
                Whether to wrap the latex code in a document environment for compiling
            document_wrapper : formatted text string with variable ``content``.
                This text is called to evaluate the content embedded in a document
                environment with a preamble setting up the TikZ syntax.
            figure_wrapper : formatted text string
                This text is evaluated with variables ``content``, ``caption`` and ``label``.
                It wraps the content and if a caption is provided, adds the latex code for
                that caption, and if a label is provided, adds the latex code for a label.
            subfigure_wrapper : formatted text string
                This text evaluate variables ``size``, ``content``, ``caption`` and ``label``.
                It wraps the content and if a caption is provided, adds the latex code for
                that caption, and if a label is provided, adds the latex code for a label.
                The size is the vertical size of each row of subfigures as a fraction.

    See Also
    ========
    to_latex
    """
    path.write(to_latex(Gbunch, **options))
<|endoftext|>"""
*****
Pydot
*****

Import and export NetworkX graphs in Graphviz dot format using pydot.

Either this module or nx_agraph can be used to interface with graphviz.

Examples
--------
>>> G = nx.complete_graph(5)
>>> PG = nx.nx_pydot.to_pydot(G)
>>> H = nx.nx_pydot.from_pydot(PG)

See Also
--------
 - pydot:         https://github.com/erocarrera/pydot
 - Graphviz:      https://www.graphviz.org
 - DOT Language:  http://www.graphviz.org/doc/info/lang.html
"""

from locale import getpreferredencoding

import networkx as nx
from networkx.utils import open_file

__all__ = [
    "write_dot",
    "read_dot",
    "graphviz_layout",
    "pydot_layout",
    "to_pydot",
    "from_pydot",
]


@open_file(1, mode="w")
def write_dot(G, path):
    """Write NetworkX graph G to Graphviz dot format on path.

    Path can be a string or a file handle.
    """
    P = to_pydot(G)
    path.write(P.to_string())
    return


@open_file(0, mode="r")
@nx._dispatchable(name="pydot_read_dot", graphs=None, returns_graph=True)
def read_dot(path):
    """Returns a NetworkX :class:`MultiGraph` or :class:`MultiDiGraph` from the
    dot file with the passed path.

    If this file contains multiple graphs, only the first such graph is
    returned. All graphs _except_ the first are silently ignored.

    Parameters
    ----------
    path : str or file
        Filename or file handle.

    Returns
    -------
    G : MultiGraph or MultiDiGraph
        A :class:`MultiGraph` or :class:`MultiDiGraph`.

    Notes
    -----
    Use `G = nx.Graph(nx.nx_pydot.read_dot(path))` to return a :class:`Graph` instead of a
    :class:`MultiGraph`.
    """
    import pydot

    data = path.read()

    # List of one or more "pydot.Dot" instances deserialized from this file.
    P_list = pydot.graph_from_dot_data(data)

    # Convert only the first such instance into a NetworkX graph.
    return from_pydot(P_list[0])


@nx._dispatchable(graphs=None, returns_graph=True)
def from_pydot(P):
    """Returns a NetworkX graph from a Pydot graph.

    Parameters
    ----------
    P : Pydot graph
      A graph created with Pydot

    Returns
    -------
    G : NetworkX multigraph
        A MultiGraph or MultiDiGraph.

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> A = nx.nx_pydot.to_pydot(K5)
    >>> G = nx.nx_pydot.from_pydot(A)  # return MultiGraph

    # make a Graph instead of MultiGraph
    >>> G = nx.Graph(nx.nx_pydot.from_pydot(A))

    """

    if P.get_strict(None):  # pydot bug: get_strict() shouldn't take argument
        multiedges = False
    else:
        multiedges = True

    if P.get_type() == "graph":  # undirected
        if multiedges:
            N = nx.MultiGraph()
        else:
            N = nx.Graph()
    else:
        if multiedges:
            N = nx.MultiDiGraph()
        else:
            N = nx.DiGraph()

    # assign defaults
    name = P.get_name().strip('"')
    if name != "":
        N.name = name

    # add nodes, attributes to N.node_attr
    for p in P.get_node_list():
        n = p.get_name().strip('"')
        if n in ("node", "graph", "edge"):
            continue
        N.add_node(n, **p.get_attributes())

    # add edges
    for e in P.get_edge_list():
        u = e.get_source()
        v = e.get_destination()
        attr = e.get_attributes()
        s = []
        d = []

        if isinstance(u, str):
            s.append(u.strip('"'))
        else:
            for unodes in u["nodes"]:
                s.append(unodes.strip('"'))

        if isinstance(v, str):
            d.append(v.strip('"'))
        else:
            for vnodes in v["nodes"]:
                d.append(vnodes.strip('"'))

        for source_node in s:
            for destination_node in d:
                N.add_edge(source_node, destination_node, **attr)

    # add default attributes for graph, nodes, edges
    pattr = P.get_attributes()
    if pattr:
        N.graph["graph"] = pattr
    try:
        N.graph["node"] = P.get_node_defaults()[0]
    except (IndexError, TypeError):
        pass  # N.graph['node']={}
    try:
        N.graph["edge"] = P.get_edge_defaults()[0]
    except (IndexError, TypeError):
        pass  # N.graph['edge']={}
    return N


def to_pydot(N):
    """Returns a pydot graph from a NetworkX graph N.

    Parameters
    ----------
    N : NetworkX graph
      A graph created with NetworkX

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> P = nx.nx_pydot.to_pydot(K5)

    Notes
    -----

    """
    import pydot

    # set Graphviz graph type
    if N.is_directed():
        graph_type = "digraph"
    else:
        graph_type = "graph"
    strict = nx.number_of_selfloops(N) == 0 and not N.is_multigraph()

    name = N.name
    graph_defaults = N.graph.get("graph", {})
    if name == "":
        P = pydot.Dot("", graph_type=graph_type, strict=strict, **graph_defaults)
    else:
        P = pydot.Dot(
            f'"{name}"', graph_type=graph_type, strict=strict, **graph_defaults
        )
    try:
        P.set_node_defaults(**N.graph["node"])
    except KeyError:
        pass
    try:
        P.set_edge_defaults(**N.graph["edge"])
    except KeyError:
        pass

    for n, nodedata in N.nodes(data=True):
        str_nodedata = {str(k): str(v) for k, v in nodedata.items()}
        n = str(n)
        p = pydot.Node(n, **str_nodedata)
        P.add_node(p)

    if N.is_multigraph():
        for u, v, key, edgedata in N.edges(data=True, keys=True):
            str_edgedata = {str(k): str(v) for k, v in edgedata.items() if k != "key"}
            u, v = str(u), str(v)
            edge = pydot.Edge(u, v, key=str(key), **str_edgedata)
            P.add_edge(edge)

    else:
        for u, v, edgedata in N.edges(data=True):
            str_edgedata = {str(k): str(v) for k, v in edgedata.items()}
            u, v = str(u), str(v)
            edge = pydot.Edge(u, v, **str_edgedata)
            P.add_edge(edge)
    return P


def graphviz_layout(G, prog="neato", root=None):
    """Create node positions using Pydot and Graphviz.

    Returns a dictionary of positions keyed by node.

    Parameters
    ----------
    G : NetworkX Graph
        The graph for which the layout is computed.
    prog : string (default: 'neato')
        The name of the GraphViz program to use for layout.
        Options depend on GraphViz version but may include:
        'dot', 'twopi', 'fdp', 'sfdp', 'circo'
    root : Node from G or None (default: None)
        The node of G from which to start some layout algorithms.

    Returns
    -------
      Dictionary of (x, y) positions keyed by node.

    Examples
    --------
    >>> G = nx.complete_graph(4)
    >>> pos = nx.nx_pydot.graphviz_layout(G)
    >>> pos = nx.nx_pydot.graphviz_layout(G, prog="dot")

    Notes
    -----
    This is a wrapper for pydot_layout.
    """
    return pydot_layout(G=G, prog=prog, root=root)


def pydot_layout(G, prog="neato", root=None):
    """Create node positions using :mod:`pydot` and Graphviz.

    Parameters
    ----------
    G : Graph
        NetworkX graph to be laid out.
    prog : string  (default: 'neato')
        Name of the GraphViz command to use for layout.
        Options depend on GraphViz version but may include:
        'dot', 'twopi', 'fdp', 'sfdp', 'circo'
    root : Node from G or None (default: None)
        The node of G from which to start some layout algorithms.

    Returns
    -------
    dict
        Dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.complete_graph(4)
    >>> pos = nx.nx_pydot.pydot_layout(G)
    >>> pos = nx.nx_pydot.pydot_layout(G, prog="dot")

    Notes
    -----
    If you use complex node objects, they may have the same string
    representation and GraphViz could treat them as the same node.
    The layout may assign both nodes a single location. See Issue #1568
    If this occurs in your case, consider relabeling the nodes just
    for the layout computation using something similar to::

        H = nx.convert_node_labels_to_integers(G, label_attribute="node_label")
        H_layout = nx.nx_pydot.pydot_layout(H, prog="dot")
        G_layout = {H.nodes[n]["node_label"]: p for n, p in H_layout.items()}

    """
    import pydot

    P = to_pydot(G)
    if root is not None:
        P.set("root", str(root))

    # List of low-level bytes comprising a string in the dot language converted
    # from the passed graph with the passed external GraphViz command.
    D_bytes = P.create_dot(prog=prog)

    # Unique string decoded from these bytes with the preferred locale encoding
    D = str(D_bytes, encoding=getpreferredencoding())

    if D == "":  # no data returned
        print(f"Graphviz layout with {prog} failed")
        print()
        print("To debug what happened try:")
        print("P = nx.nx_pydot.to_pydot(G)")
        print('P.write_dot("file.dot")')
        print(f"And then run {prog} on file.dot")
        return

    # List of one or more "pydot.Dot" instances deserialized from this string.
    Q_list = pydot.graph_from_dot_data(D)
    assert len(Q_list) == 1

    # The first and only such instance, as guaranteed by the above assertion.
    Q = Q_list[0]

    node_pos = {}
    for n in G.nodes():
        str_n = str(n)
        node = Q.get_node(pydot.quote_id_if_necessary(str_n))

        if isinstance(node, list):
            node = node[0]
        pos = node.get_pos()[1:-1]  # strip leading and trailing double quotes
        if pos is not None:
            xx, yy = pos.split(",")
            node_pos[n] = (float(xx), float(yy))
    return node_pos
<|endoftext|>"""
**********
Matplotlib
**********

Draw networks with matplotlib.

Examples
--------
>>> G = nx.complete_graph(5)
>>> nx.draw(G)

See Also
--------
 - :doc:`matplotlib <matplotlib:index>`
 - :func:`matplotlib.pyplot.scatter`
 - :obj:`matplotlib.patches.FancyArrowPatch`
"""

import collections
import itertools
from numbers import Number

import networkx as nx
from networkx.drawing.layout import (
    circular_layout,
    forceatlas2_layout,
    kamada_kawai_layout,
    planar_layout,
    random_layout,
    shell_layout,
    spectral_layout,
    spring_layout,
)

__all__ = [
    "draw",
    "draw_networkx",
    "draw_networkx_nodes",
    "draw_networkx_edges",
    "draw_networkx_labels",
    "draw_networkx_edge_labels",
    "draw_circular",
    "draw_kamada_kawai",
    "draw_random",
    "draw_spectral",
    "draw_spring",
    "draw_planar",
    "draw_shell",
    "draw_forceatlas2",
]


def draw(G, pos=None, ax=None, **kwds):
    """Draw the graph G with Matplotlib.

    Draw the graph as a simple representation with no node
    labels or edge labels and using the full Matplotlib figure area
    and no axis labels by default.  See draw_networkx() for more
    full-featured drawing that allows title, axis labels etc.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary, optional
        A dictionary with nodes as keys and positions as values.
        If not specified a spring layout positioning will be computed.
        See :py:mod:`networkx.drawing.layout` for functions that
        compute node positions.

    ax : Matplotlib Axes object, optional
        Draw the graph in specified Matplotlib axes.

    kwds : optional keywords
        See networkx.draw_networkx() for a description of optional keywords.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nx.draw(G)
    >>> nx.draw(G, pos=nx.spring_layout(G))  # use spring layout

    See Also
    --------
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    draw_networkx_edge_labels

    Notes
    -----
    This function has the same name as pylab.draw and pyplot.draw
    so beware when using `from networkx import *`

    since you might overwrite the pylab.draw function.

    With pyplot use

    >>> import matplotlib.pyplot as plt
    >>> G = nx.dodecahedral_graph()
    >>> nx.draw(G)  # networkx draw()
    >>> plt.draw()  # pyplot draw()

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html
    """
    import matplotlib.pyplot as plt

    if ax is None:
        cf = plt.gcf()
    else:
        cf = ax.get_figure()
    cf.set_facecolor("w")
    if ax is None:
        if cf.axes:
            ax = cf.gca()
        else:
            ax = cf.add_axes((0, 0, 1, 1))

    if "with_labels" not in kwds:
        kwds["with_labels"] = "labels" in kwds

    draw_networkx(G, pos=pos, ax=ax, **kwds)
    ax.set_axis_off()
    plt.draw_if_interactive()
    return


def draw_networkx(G, pos=None, arrows=None, with_labels=True, **kwds):
    r"""Draw the graph G using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.
    See draw() for simple drawing without labels or axes.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary, optional
        A dictionary with nodes as keys and positions as values.
        If not specified a spring layout positioning will be computed.
        See :py:mod:`networkx.drawing.layout` for functions that
        compute node positions.

    arrows : bool or None, optional (default=None)
        If `None`, directed graphs draw arrowheads with
        `~matplotlib.patches.FancyArrowPatch`, while undirected graphs draw edges
        via `~matplotlib.collections.LineCollection` for speed.
        If `True`, draw arrowheads with FancyArrowPatches (bendable and stylish).
        If `False`, draw edges using LineCollection (linear and fast).
        For directed graphs, if True draw arrowheads.
        Note: Arrows will be the same color as edges.

    arrowstyle : str (default='-\|>' for directed graphs)
        For directed graphs, choose the style of the arrowsheads.
        For undirected graphs default to '-'

        See `matplotlib.patches.ArrowStyle` for more options.

    arrowsize : int or list (default=10)
        For directed graphs, choose the size of the arrow head's length and
        width. A list of values can be passed in to assign a different size for arrow head's length and width.
        See `matplotlib.patches.FancyArrowPatch` for attribute `mutation_scale`
        for more info.

    with_labels :  bool (default=True)
        Set to True to draw labels on the nodes.

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    nodelist : list (default=list(G))
        Draw only specified nodes

    edgelist : list (default=list(G.edges()))
        Draw only specified edges

    node_size : scalar or array (default=300)
        Size of nodes.  If an array is specified it must be the
        same length as nodelist.

    node_color : color or array of colors (default='#1f78b4')
        Node color. Can be a single color or a sequence of colors with the same
        length as nodelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the cmap and vmin,vmax parameters. See
        matplotlib.scatter for more details.

    node_shape :  string (default='o')
        The shape of the node.  Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8'.

    alpha : float or None (default=None)
        The node and edge transparency

    cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of nodes

    vmin,vmax : float, optional
        Minimum and maximum for node colormap scaling

    linewidths : scalar or sequence (default=1.0)
        Line width of symbol border

    width : float or array of floats (default=1.0)
        Line width of edges

    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edgelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional
        Minimum and maximum for edge colormap scaling

    style : string (default=solid line)
        Edge line style e.g.: '-', '--', '-.', ':'
        or words like 'solid' or 'dashed'.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)

    labels : dictionary (default=None)
        Node labels in a dictionary of text labels keyed by node

    font_size : int (default=12 for nodes, 10 for edges)
        Font size for text labels

    font_color : color (default='k' black)
        Font color string. Color can be string or rgb (or rgba) tuple of
        floats from 0-1.

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    label : string, optional
        Label for graph legend

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    kwds : optional keywords
        See networkx.draw_networkx_nodes(), networkx.draw_networkx_edges(), and
        networkx.draw_networkx_labels() for a description of optional keywords.

    Notes
    -----
    For directed graphs, arrows  are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nx.draw(G)
    >>> nx.draw(G, pos=nx.spring_layout(G))  # use spring layout

    >>> import matplotlib.pyplot as plt
    >>> limits = plt.axis("off")  # turn off axis

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    draw_networkx_edge_labels
    """
    from inspect import signature

    import matplotlib.pyplot as plt

    # Get all valid keywords by inspecting the signatures of draw_networkx_nodes,
    # draw_networkx_edges, draw_networkx_labels

    valid_node_kwds = signature(draw_networkx_nodes).parameters.keys()
    valid_edge_kwds = signature(draw_networkx_edges).parameters.keys()
    valid_label_kwds = signature(draw_networkx_labels).parameters.keys()

    # Create a set with all valid keywords across the three functions and
    # remove the arguments of this function (draw_networkx)
    valid_kwds = (valid_node_kwds | valid_edge_kwds | valid_label_kwds) - {
        "G",
        "pos",
        "arrows",
        "with_labels",
    }

    if any(k not in valid_kwds for k in kwds):
        invalid_args = ", ".join([k for k in kwds if k not in valid_kwds])
        raise ValueError(f"Received invalid argument(s): {invalid_args}")

    node_kwds = {k: v for k, v in kwds.items() if k in valid_node_kwds}
    edge_kwds = {k: v for k, v in kwds.items() if k in valid_edge_kwds}
    label_kwds = {k: v for k, v in kwds.items() if k in valid_label_kwds}

    if pos is None:
        pos = nx.drawing.spring_layout(G)  # default to spring layout

    draw_networkx_nodes(G, pos, **node_kwds)
    draw_networkx_edges(G, pos, arrows=arrows, **edge_kwds)
    if with_labels:
        draw_networkx_labels(G, pos, **label_kwds)
    plt.draw_if_interactive()


def draw_networkx_nodes(
    G,
    pos,
    nodelist=None,
    node_size=300,
    node_color="#1f78b4",
    node_shape="o",
    alpha=None,
    cmap=None,
    vmin=None,
    vmax=None,
    ax=None,
    linewidths=None,
    edgecolors=None,
    label=None,
    margins=None,
    hide_ticks=True,
):
    """Draw the nodes of the graph G.

    This draws only the nodes of the graph G.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    nodelist : list (default list(G))
        Draw only specified nodes

    node_size : scalar or array (default=300)
        Size of nodes.  If an array it must be the same length as nodelist.

    node_color : color or array of colors (default='#1f78b4')
        Node color. Can be a single color or a sequence of colors with the same
        length as nodelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the cmap and vmin,vmax parameters. See
        matplotlib.scatter for more details.

    node_shape :  string (default='o')
        The shape of the node.  Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8'.

    alpha : float or array of floats (default=None)
        The node transparency.  This can be a single alpha value,
        in which case it will be applied to all the nodes of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    cmap : Matplotlib colormap (default=None)
        Colormap for mapping intensities of nodes

    vmin,vmax : floats or None (default=None)
        Minimum and maximum for node colormap scaling

    linewidths : [None | scalar | sequence] (default=1.0)
        Line width of symbol border

    edgecolors : [None | scalar | sequence] (default = node_color)
        Colors of node borders. Can be a single color or a sequence of colors with the
        same length as nodelist. Color can be string or rgb (or rgba) tuple of floats
        from 0-1. If numeric values are specified they will be mapped to colors
        using the cmap and vmin,vmax parameters. See `~matplotlib.pyplot.scatter` for more details.

    label : [None | string]
        Label for legend

    margins : float or 2-tuple, optional
        Sets the padding for axis autoscaling. Increase margin to prevent
        clipping for nodes that are near the edges of an image. Values should
        be in the range ``[0, 1]``. See :meth:`matplotlib.axes.Axes.margins`
        for details. The default is `None`, which uses the Matplotlib default.

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    Returns
    -------
    matplotlib.collections.PathCollection
        `PathCollection` of the nodes.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nodes = nx.draw_networkx_nodes(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_edges
    draw_networkx_labels
    draw_networkx_edge_labels
    """
    from collections.abc import Iterable

    import matplotlib as mpl
    import matplotlib.collections  # call as mpl.collections
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = list(G)

    if len(nodelist) == 0:  # empty nodelist, no drawing
        return mpl.collections.PathCollection(None)

    try:
        xy = np.asarray([pos[v] for v in nodelist])
    except KeyError as err:
        raise nx.NetworkXError(f"Node {err} has no position.") from err

    if isinstance(alpha, Iterable):
        node_color = apply_alpha(node_color, alpha, nodelist, cmap, vmin, vmax)
        alpha = None

    if not isinstance(node_shape, np.ndarray) and not isinstance(node_shape, list):
        node_shape = np.array([node_shape for _ in range(len(nodelist))])

    for shape in np.unique(node_shape):
        node_collection = ax.scatter(
            xy[node_shape == shape, 0],
            xy[node_shape == shape, 1],
            s=node_size,
            c=node_color,
            marker=shape,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            linewidths=linewidths,
            edgecolors=edgecolors,
            label=label,
        )
    if hide_ticks:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    if margins is not None:
        if isinstance(margins, Iterable):
            ax.margins(*margins)
        else:
            ax.margins(margins)

    node_collection.set_zorder(2)
    return node_collection


class FancyArrowFactory:
    """Draw arrows with `matplotlib.patches.FancyarrowPatch`"""

    class ConnectionStyleFactory:
        def __init__(self, connectionstyles, selfloop_height, ax=None):
            import matplotlib as mpl
            import matplotlib.path  # call as mpl.path
            import numpy as np

            self.ax = ax
            self.mpl = mpl
            self.np = np
            self.base_connection_styles = [
                mpl.patches.ConnectionStyle(cs) for cs in connectionstyles
            ]
            self.n = len(self.base_connection_styles)
            self.selfloop_height = selfloop_height

        def curved(self, edge_index):
            return self.base_connection_styles[edge_index % self.n]

        def self_loop(self, edge_index):
            def self_loop_connection(posA, posB, *args, **kwargs):
                if not self.np.all(posA == posB):
                    raise nx.NetworkXError(
                        "`self_loop` connection style method"
                        "is only to be used for self-loops"
                    )
                # this is called with _screen space_ values
                # so convert back to data space
                data_loc = self.ax.transData.inverted().transform(posA)
                v_shift = 0.1 * self.selfloop_height
                h_shift = v_shift * 0.5
                # put the top of the loop first so arrow is not hidden by node
                path = self.np.asarray(
                    [
                        # 1
                        [0, v_shift],
                        # 4 4 4
                        [h_shift, v_shift],
                        [h_shift, 0],
                        [0, 0],
                        # 4 4 4
                        [-h_shift, 0],
                        [-h_shift, v_shift],
                        [0, v_shift],
                    ]
                )
                # Rotate self loop 90 deg. if more than 1
                # This will allow for maximum of 4 visible self loops
                if edge_index % 4:
                    x, y = path.T
                    for _ in range(edge_index % 4):
                        x, y = y, -x
                    path = self.np.array([x, y]).T
                return self.mpl.path.Path(
                    self.ax.transData.transform(data_loc + path), [1, 4, 4, 4, 4, 4, 4]
                )

            return self_loop_connection

    def __init__(
        self,
        edge_pos,
        edgelist,
        nodelist,
        edge_indices,
        node_size,
        selfloop_height,
        connectionstyle="arc3",
        node_shape="o",
        arrowstyle="-",
        arrowsize=10,
        edge_color="k",
        alpha=None,
        linewidth=1.0,
        style="solid",
        min_source_margin=0,
        min_target_margin=0,
        ax=None,
    ):
        import matplotlib as mpl
        import matplotlib.patches  # call as mpl.patches
        import matplotlib.pyplot as plt
        import numpy as np

        if isinstance(connectionstyle, str):
            connectionstyle = [connectionstyle]
        elif np.iterable(connectionstyle):
            connectionstyle = list(connectionstyle)
        else:
            msg = "ConnectionStyleFactory arg `connectionstyle` must be str or iterable"
            raise nx.NetworkXError(msg)
        self.ax = ax
        self.mpl = mpl
        self.np = np
        self.edge_pos = edge_pos
        self.edgelist = edgelist
        self.nodelist = nodelist
        self.node_shape = node_shape
        self.min_source_margin = min_source_margin
        self.min_target_margin = min_target_margin
        self.edge_indices = edge_indices
        self.node_size = node_size
        self.connectionstyle_factory = self.ConnectionStyleFactory(
            connectionstyle, selfloop_height, ax
        )
        self.arrowstyle = arrowstyle
        self.arrowsize = arrowsize
        self.arrow_colors = mpl.colors.colorConverter.to_rgba_array(edge_color, alpha)
        self.linewidth = linewidth
        self.style = style
        if isinstance(arrowsize, list) and len(arrowsize) != len(edge_pos):
            raise ValueError("arrowsize should have the same length as edgelist")

    def __call__(self, i):
        (x1, y1), (x2, y2) = self.edge_pos[i]
        shrink_source = 0  # space from source to tail
        shrink_target = 0  # space from  head to target
        if (
            self.np.iterable(self.min_source_margin)
            and not isinstance(self.min_source_margin, str)
            and not isinstance(self.min_source_margin, tuple)
        ):
            min_source_margin = self.min_source_margin[i]
        else:
            min_source_margin = self.min_source_margin

        if (
            self.np.iterable(self.min_target_margin)
            and not isinstance(self.min_target_margin, str)
            and not isinstance(self.min_target_margin, tuple)
        ):
            min_target_margin = self.min_target_margin[i]
        else:
            min_target_margin = self.min_target_margin

        if self.np.iterable(self.node_size):  # many node sizes
            source, target = self.edgelist[i][:2]
            source_node_size = self.node_size[self.nodelist.index(source)]
            target_node_size = self.node_size[self.nodelist.index(target)]
            shrink_source = self.to_marker_edge(source_node_size, self.node_shape)
            shrink_target = self.to_marker_edge(target_node_size, self.node_shape)
        else:
            shrink_source = self.to_marker_edge(self.node_size, self.node_shape)
            shrink_target = shrink_source
        shrink_source = max(shrink_source, min_source_margin)
        shrink_target = max(shrink_target, min_target_margin)

        # scale factor of arrow head
        if isinstance(self.arrowsize, list):
            mutation_scale = self.arrowsize[i]
        else:
            mutation_scale = self.arrowsize

        if len(self.arrow_colors) > i:
            arrow_color = self.arrow_colors[i]
        elif len(self.arrow_colors) == 1:
            arrow_color = self.arrow_colors[0]
        else:  # Cycle through colors
            arrow_color = self.arrow_colors[i % len(self.arrow_colors)]

        if self.np.iterable(self.linewidth):
            if len(self.linewidth) > i:
                linewidth = self.linewidth[i]
            else:
                linewidth = self.linewidth[i % len(self.linewidth)]
        else:
            linewidth = self.linewidth

        if (
            self.np.iterable(self.style)
            and not isinstance(self.style, str)
            and not isinstance(self.style, tuple)
        ):
            if len(self.style) > i:
                linestyle = self.style[i]
            else:  # Cycle through styles
                linestyle = self.style[i % len(self.style)]
        else:
            linestyle = self.style

        if x1 == x2 and y1 == y2:
            connectionstyle = self.connectionstyle_factory.self_loop(
                self.edge_indices[i]
            )
        else:
            connectionstyle = self.connectionstyle_factory.curved(self.edge_indices[i])

        if (
            self.np.iterable(self.arrowstyle)
            and not isinstance(self.arrowstyle, str)
            and not isinstance(self.arrowstyle, tuple)
        ):
            arrowstyle = self.arrowstyle[i]
        else:
            arrowstyle = self.arrowstyle

        return self.mpl.patches.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=arrowstyle,
            shrinkA=shrink_source,
            shrinkB=shrink_target,
            mutation_scale=mutation_scale,
            color=arrow_color,
            linewidth=linewidth,
            connectionstyle=connectionstyle,
            linestyle=linestyle,
            zorder=1,  # arrows go behind nodes
        )

    def to_marker_edge(self, marker_size, marker):
        if marker in "s^>v<d":  # `large` markers need extra space
            return self.np.sqrt(2 * marker_size) / 2
        else:
            return self.np.sqrt(marker_size) / 2


def draw_networkx_edges(
    G,
    pos,
    edgelist=None,
    width=1.0,
    edge_color="k",
    style="solid",
    alpha=None,
    arrowstyle=None,
    arrowsize=10,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    arrows=None,
    label=None,
    node_size=300,
    nodelist=None,
    node_shape="o",
    connectionstyle="arc3",
    min_source_margin=0,
    min_target_margin=0,
    hide_ticks=True,
):
    r"""Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edgelist : collection of edge tuples (default=G.edges())
        Draw only specified edges

    width : float or array of floats (default=1.0)
        Line width of edges

    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edgelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    style : string or array of strings (default='solid')
        Edge line style e.g.: '-', '--', '-.', ':'
        or words like 'solid' or 'dashed'.
        Can be a single style or a sequence of styles with the same
        length as the edge list.
        If less styles than edges are given the styles will cycle.
        If more styles than edges are given the styles will be used sequentially
        and not be exhausted.
        Also, `(offset, onoffseq)` tuples can be used as style instead of a strings.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)

    alpha : float or array of floats (default=None)
        The edge transparency.  This can be a single alpha value,
        in which case it will be applied to all specified edges. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional
        Minimum and maximum for edge colormap scaling

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    arrows : bool or None, optional (default=None)
        If `None`, directed graphs draw arrowheads with
        `~matplotlib.patches.FancyArrowPatch`, while undirected graphs draw edges
        via `~matplotlib.collections.LineCollection` for speed.
        If `True`, draw arrowheads with FancyArrowPatches (bendable and stylish).
        If `False`, draw edges using LineCollection (linear and fast).

        Note: Arrowheads will be the same color as edges.

    arrowstyle : str or list of strs (default='-\|>' for directed graphs)
        For directed graphs and `arrows==True` defaults to '-\|>',
        For undirected graphs default to '-'.

        See `matplotlib.patches.ArrowStyle` for more options.

    arrowsize : int or list of ints(default=10)
        For directed graphs, choose the size of the arrow head's length and
        width. See `matplotlib.patches.FancyArrowPatch` for attribute
        `mutation_scale` for more info.

    connectionstyle : string or iterable of strings (default="arc3")
        Pass the connectionstyle parameter to create curved arc of rounding
        radius rad. For example, connectionstyle='arc3,rad=0.2'.
        See `matplotlib.patches.ConnectionStyle` and
        `matplotlib.patches.FancyArrowPatch` for more info.
        If Iterable, index indicates i'th edge key of MultiGraph

    node_size : scalar or array (default=300)
        Size of nodes. Though the nodes are not drawn with this function, the
        node size is used in determining edge positioning.

    nodelist : list, optional (default=G.nodes())
       This provides the node order for the `node_size` array (if it is an array).

    node_shape :  string (default='o')
        The marker used for nodes, used in determining edge positioning.
        Specification is as a `matplotlib.markers` marker, e.g. one of 'so^>v<dph8'.

    label : None or string
        Label for legend

    min_source_margin : int or list of ints (default=0)
        The minimum margin (gap) at the beginning of the edge at the source.

    min_target_margin : int or list of ints (default=0)
        The minimum margin (gap) at the end of the edge at the target.

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    Returns
    -------
     matplotlib.collections.LineCollection or a list of matplotlib.patches.FancyArrowPatch
        If ``arrows=True``, a list of FancyArrowPatches is returned.
        If ``arrows=False``, a LineCollection is returned.
        If ``arrows=None`` (the default), then a LineCollection is returned if
        `G` is undirected, otherwise returns a list of FancyArrowPatches.

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False or by passing an arrowstyle without
    an arrow on the end.

    Be sure to include `node_size` as a keyword argument; arrows are
    drawn considering the size of nodes.

    Self-loops are always drawn with `~matplotlib.patches.FancyArrowPatch`
    regardless of the value of `arrows` or whether `G` is directed.
    When ``arrows=False`` or ``arrows=None`` and `G` is undirected, the
    FancyArrowPatches corresponding to the self-loops are not explicitly
    returned. They should instead be accessed via the ``Axes.patches``
    attribute (see examples).

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> alphas = [0.3, 0.4, 0.5]
    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs
    ...     arc.set_alpha(alphas[i])

    The FancyArrowPatches corresponding to self-loops are not always
    returned, but can always be accessed via the ``patches`` attribute of the
    `matplotlib.Axes` object.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> G = nx.Graph([(0, 1), (0, 0)])  # Self-loop at node 0
    >>> edge_collection = nx.draw_networkx_edges(G, pos=nx.circular_layout(G), ax=ax)
    >>> self_loop_fap = ax.patches[0]

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_labels
    draw_networkx_edge_labels

    """
    import warnings

    import matplotlib as mpl
    import matplotlib.collections  # call as mpl.collections
    import matplotlib.colors  # call as mpl.colors
    import matplotlib.pyplot as plt
    import numpy as np

    # The default behavior is to use LineCollection to draw edges for
    # undirected graphs (for performance reasons) and use FancyArrowPatches
    # for directed graphs.
    # The `arrows` keyword can be used to override the default behavior
    if arrows is None:
        use_linecollection = not (G.is_directed() or G.is_multigraph())
    else:
        if not isinstance(arrows, bool):
            raise TypeError("Argument `arrows` must be of type bool or None")
        use_linecollection = not arrows

    if isinstance(connectionstyle, str):
        connectionstyle = [connectionstyle]
    elif np.iterable(connectionstyle):
        connectionstyle = list(connectionstyle)
    else:
        msg = "draw_networkx_edges arg `connectionstyle` must be str or iterable"
        raise nx.NetworkXError(msg)

    # Some kwargs only apply to FancyArrowPatches. Warn users when they use
    # non-default values for these kwargs when LineCollection is being used
    # instead of silently ignoring the specified option
    if use_linecollection:
        msg = (
            "\n\nThe {0} keyword argument is not applicable when drawing edges\n"
            "with LineCollection.\n\n"
            "To make this warning go away, either specify `arrows=True` to\n"
            "force FancyArrowPatches or use the default values.\n"
            "Note that using FancyArrowPatches may be slow for large graphs.\n"
        )
        if arrowstyle is not None:
            warnings.warn(msg.format("arrowstyle"), category=UserWarning, stacklevel=2)
        if arrowsize != 10:
            warnings.warn(msg.format("arrowsize"), category=UserWarning, stacklevel=2)
        if min_source_margin != 0:
            warnings.warn(
                msg.format("min_source_margin"), category=UserWarning, stacklevel=2
            )
        if min_target_margin != 0:
            warnings.warn(
                msg.format("min_target_margin"), category=UserWarning, stacklevel=2
            )
        if any(cs != "arc3" for cs in connectionstyle):
            warnings.warn(
                msg.format("connectionstyle"), category=UserWarning, stacklevel=2
            )

    # NOTE: Arrowstyle modification must occur after the warnings section
    if arrowstyle is None:
        arrowstyle = "-|>" if G.is_directed() else "-"

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges)  # (u, v, k) for multigraph (u, v) otherwise

    if len(edgelist):
        if G.is_multigraph():
            key_count = collections.defaultdict(lambda: itertools.count(0))
            edge_indices = [next(key_count[tuple(e[:2])]) for e in edgelist]
        else:
            edge_indices = [0] * len(edgelist)
    else:  # no edges!
        return []

    if nodelist is None:
        nodelist = list(G.nodes())

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
        np.iterable(edge_color)
        and (len(edge_color) == len(edge_pos))
        and np.all([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, mpl.colors.Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    # compute initial view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))
    w = maxx - minx
    h = maxy - miny

    # Self-loops are scaled by view extent, except in cases the extent
    # is 0, e.g. for a single node. In this case, fall back to scaling
    # by the maximum node size
    selfloop_height = h if h != 0 else 0.005 * np.array(node_size).max()
    fancy_arrow_factory = FancyArrowFactory(
        edge_pos,
        edgelist,
        nodelist,
        edge_indices,
        node_size,
        selfloop_height,
        connectionstyle,
        node_shape,
        arrowstyle,
        arrowsize,
        edge_color,
        alpha,
        width,
        style,
        min_source_margin,
        min_target_margin,
        ax=ax,
    )

    # Draw the edges
    if use_linecollection:
        edge_collection = mpl.collections.LineCollection(
            edge_pos,
            colors=edge_color,
            linewidths=width,
            antialiaseds=(1,),
            linestyle=style,
            alpha=alpha,
        )
        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)
        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)
        edge_viz_obj = edge_collection

        # Make sure selfloop edges are also drawn
        # ---------------------------------------
        selfloops_to_draw = [loop for loop in nx.selfloop_edges(G) if loop in edgelist]
        if selfloops_to_draw:
            edgelist_tuple = list(map(tuple, edgelist))
            arrow_collection = []
            for loop in selfloops_to_draw:
                i = edgelist_tuple.index(loop)
                arrow = fancy_arrow_factory(i)
                arrow_collection.append(arrow)
                ax.add_patch(arrow)
    else:
        edge_viz_obj = []
        for i in range(len(edgelist)):
            arrow = fancy_arrow_factory(i)
            ax.add_patch(arrow)
            edge_viz_obj.append(arrow)

    # update view after drawing
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    if hide_ticks:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    return edge_viz_obj


def draw_networkx_labels(
    G,
    pos,
    labels=None,
    font_size=12,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    clip_on=True,
    hide_ticks=True,
):
    """Draw node labels on the graph G.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    labels : dictionary (default={n: n for n in G})
        Node labels in a dictionary of text labels keyed by node.
        Node-keys in labels should appear as keys in `pos`.
        If needed use: `{n:lab for n,lab in labels.items() if n in pos}`

    font_size : int or dictionary of nodes to ints (default=12)
        Font size for text labels.

    font_color : color or dictionary of nodes to colors (default='k' black)
        Font color string. Color can be string or rgb (or rgba) tuple of
        floats from 0-1.

    font_weight : string or dictionary of nodes to strings (default='normal')
        Font weight.

    font_family : string or dictionary of nodes to strings (default='sans-serif')
        Font family.

    alpha : float or None or dictionary of nodes to floats (default=None)
        The text transparency.

    bbox : Matplotlib bbox, (default is Matplotlib's ax.text default)
        Specify text box properties (e.g. shape, color etc.) for node labels.

    horizontalalignment : string or array of strings (default='center')
        Horizontal alignment {'center', 'right', 'left'}. If an array is
        specified it must be the same length as `nodelist`.

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}.
        If an array is specified it must be the same length as `nodelist`.

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    clip_on : bool (default=True)
        Turn on clipping of node labels at axis boundaries

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    Returns
    -------
    dict
        `dict` of labels keyed on the nodes

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> labels = nx.draw_networkx_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_edge_labels
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = {n: n for n in G.nodes()}

    individual_params = set()

    def check_individual_params(p_value, p_name):
        if isinstance(p_value, dict):
            if len(p_value) != len(labels):
                raise ValueError(f"{p_name} must have the same length as labels.")
            individual_params.add(p_name)

    def get_param_value(node, p_value, p_name):
        if p_name in individual_params:
            return p_value[node]
        return p_value

    check_individual_params(font_size, "font_size")
    check_individual_params(font_color, "font_color")
    check_individual_params(font_weight, "font_weight")
    check_individual_params(font_family, "font_family")
    check_individual_params(alpha, "alpha")

    text_items = {}  # there is no text collection so we'll fake one
    for n, label in labels.items():
        (x, y) = pos[n]
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same
        t = ax.text(
            x,
            y,
            label,
            size=get_param_value(n, font_size, "font_size"),
            color=get_param_value(n, font_color, "font_color"),
            family=get_param_value(n, font_family, "font_family"),
            weight=get_param_value(n, font_weight, "font_weight"),
            alpha=get_param_value(n, alpha, "alpha"),
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            transform=ax.transData,
            bbox=bbox,
            clip_on=clip_on,
        )
        text_items[n] = t

    if hide_ticks:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    return text_items


def draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    node_size=300,
    nodelist=None,
    connectionstyle="arc3",
    hide_ticks=True,
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default=None)
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : color (default='k' black)
        Font color string. Color can be string or rgb (or rgba) tuple of
        floats from 0-1.

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (default=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    node_size : scalar or array (default=300)
        Size of nodes.  If an array it must be the same length as nodelist.

    nodelist : list, optional (default=G.nodes())
       This provides the node order for the `node_size` array (if it is an array).

    connectionstyle : string or iterable of strings (default="arc3")
        Pass the connectionstyle parameter to create curved arc of rounding
        radius rad. For example, connectionstyle='arc3,rad=0.2'.
        See `matplotlib.patches.ConnectionStyle` and
        `matplotlib.patches.FancyArrowPatch` for more info.
        If Iterable, index indicates i'th edge key of MultiGraph

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    class CurvedArrowText(mpl.text.Text):
        def __init__(
            self,
            arrow,
            *args,
            label_pos=0.5,
            labels_horizontal=False,
            ax=None,
            **kwargs,
        ):
            # Bind to FancyArrowPatch
            self.arrow = arrow
            # how far along the text should be on the curve,
            # 0 is at start, 1 is at end etc.
            self.label_pos = label_pos
            self.labels_horizontal = labels_horizontal
            if ax is None:
                ax = plt.gca()
            self.ax = ax
            self.x, self.y, self.angle = self._update_text_pos_angle(arrow)

            # Create text object
            super().__init__(self.x, self.y, *args, rotation=self.angle, **kwargs)
            # Bind to axis
            self.ax.add_artist(self)

        def _get_arrow_path_disp(self, arrow):
            """
            This is part of FancyArrowPatch._get_path_in_displaycoord
            It omits the second part of the method where path is converted
                to polygon based on width
            The transform is taken from ax, not the object, as the object
                has not been added yet, and doesn't have transform
            """
            dpi_cor = arrow._dpi_cor
            # trans_data = arrow.get_transform()
            trans_data = self.ax.transData
            if arrow._posA_posB is not None:
                posA = arrow._convert_xy_units(arrow._posA_posB[0])
                posB = arrow._convert_xy_units(arrow._posA_posB[1])
                (posA, posB) = trans_data.transform((posA, posB))
                _path = arrow.get_connectionstyle()(
                    posA,
                    posB,
                    patchA=arrow.patchA,
                    patchB=arrow.patchB,
                    shrinkA=arrow.shrinkA * dpi_cor,
                    shrinkB=arrow.shrinkB * dpi_cor,
                )
            else:
                _path = trans_data.transform_path(arrow._path_original)
            # Return is in display coordinates
            return _path

        def _update_text_pos_angle(self, arrow):
            # Fractional label position
            path_disp = self._get_arrow_path_disp(arrow)
            (x1, y1), (cx, cy), (x2, y2) = path_disp.vertices
            # Text position at a proportion t along the line in display coords
            # default is 0.5 so text appears at the halfway point
            t = self.label_pos
            tt = 1 - t
            x = tt**2 * x1 + 2 * t * tt * cx + t**2 * x2
            y = tt**2 * y1 + 2 * t * tt * cy + t**2 * y2
            if self.labels_horizontal:
                # Horizontal text labels
                angle = 0
            else:
                # Labels parallel to curve
                change_x = 2 * tt * (cx - x1) + 2 * t * (x2 - cx)
                change_y = 2 * tt * (cy - y1) + 2 * t * (y2 - cy)
                angle = (np.arctan2(change_y, change_x) / (2 * np.pi)) * 360
                # Text is "right way up"
                if angle > 90:
                    angle -= 180
                if angle < -90:
                    angle += 180
            (x, y) = self.ax.transData.inverted().transform((x, y))
            return x, y, angle

        def draw(self, renderer):
            # recalculate the text position and angle
            self.x, self.y, self.angle = self._update_text_pos_angle(self.arrow)
            self.set_position((self.x, self.y))
            self.set_rotation(self.angle)
            # redraw text
            super().draw(renderer)

    # use default box of white with white border
    if bbox is None:
        bbox = {"boxstyle": "round", "ec": (1.0, 1.0, 1.0), "fc": (1.0, 1.0, 1.0)}

    if isinstance(connectionstyle, str):
        connectionstyle = [connectionstyle]
    elif np.iterable(connectionstyle):
        connectionstyle = list(connectionstyle)
    else:
        raise nx.NetworkXError(
            "draw_networkx_edges arg `connectionstyle` must be"
            "string or iterable of strings"
        )

    if ax is None:
        ax = plt.gca()

    if edge_labels is None:
        kwds = {"keys": True} if G.is_multigraph() else {}
        edge_labels = {tuple(edge): d for *edge, d in G.edges(data=True, **kwds)}
    # NOTHING TO PLOT
    if not edge_labels:
        return {}
    edgelist, labels = zip(*edge_labels.items())

    if nodelist is None:
        nodelist = list(G.nodes())

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    if G.is_multigraph():
        key_count = collections.defaultdict(lambda: itertools.count(0))
        edge_indices = [next(key_count[tuple(e[:2])]) for e in edgelist]
    else:
        edge_indices = [0] * len(edgelist)

    # Used to determine self loop mid-point
    # Note, that this will not be accurate,
    #   if not drawing edge_labels for all edges drawn
    h = 0
    if edge_labels:
        miny = np.amin(np.ravel(edge_pos[:, :, 1]))
        maxy = np.amax(np.ravel(edge_pos[:, :, 1]))
        h = maxy - miny
    selfloop_height = h if h != 0 else 0.005 * np.array(node_size).max()
    fancy_arrow_factory = FancyArrowFactory(
        edge_pos,
        edgelist,
        nodelist,
        edge_indices,
        node_size,
        selfloop_height,
        connectionstyle,
        ax=ax,
    )

    individual_params = {}

    def check_individual_params(p_value, p_name):
        # TODO should this be list or array (as in a numpy array)?
        if isinstance(p_value, list):
            if len(p_value) != len(edgelist):
                raise ValueError(f"{p_name} must have the same length as edgelist.")
            individual_params[p_name] = p_value.iter()

    # Don't need to pass in an edge because these are lists, not dicts
    def get_param_value(p_value, p_name):
        if p_name in individual_params:
            return next(individual_params[p_name])
        return p_value

    check_individual_params(font_size, "font_size")
    check_individual_params(font_color, "font_color")
    check_individual_params(font_weight, "font_weight")
    check_individual_params(alpha, "alpha")
    check_individual_params(horizontalalignment, "horizontalalignment")
    check_individual_params(verticalalignment, "verticalalignment")
    check_individual_params(rotate, "rotate")
    check_individual_params(label_pos, "label_pos")

    text_items = {}
    for i, (edge, label) in enumerate(zip(edgelist, labels)):
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        n1, n2 = edge[:2]
        arrow = fancy_arrow_factory(i)
        if n1 == n2:
            connectionstyle_obj = arrow.get_connectionstyle()
            posA = ax.transData.transform(pos[n1])
            path_disp = connectionstyle_obj(posA, posA)
            path_data = ax.transData.inverted().transform_path(path_disp)
            x, y = path_data.vertices[0]
            text_items[edge] = ax.text(
                x,
                y,
                label,
                size=get_param_value(font_size, "font_size"),
                color=get_param_value(font_color, "font_color"),
                family=get_param_value(font_family, "font_family"),
                weight=get_param_value(font_weight, "font_weight"),
                alpha=get_param_value(alpha, "alpha"),
                horizontalalignment=get_param_value(
                    horizontalalignment, "horizontalalignment"
                ),
                verticalalignment=get_param_value(
                    verticalalignment, "verticalalignment"
                ),
                rotation=0,
                transform=ax.transData,
                bbox=bbox,
                zorder=1,
                clip_on=clip_on,
            )
        else:
            text_items[edge] = CurvedArrowText(
                arrow,
                label,
                size=get_param_value(font_size, "font_size"),
                color=get_param_value(font_color, "font_color"),
                family=get_param_value(font_family, "font_family"),
                weight=get_param_value(font_weight, "font_weight"),
                alpha=get_param_value(alpha, "alpha"),
                horizontalalignment=get_param_value(
                    horizontalalignment, "horizontalalignment"
                ),
                verticalalignment=get_param_value(
                    verticalalignment, "verticalalignment"
                ),
                transform=ax.transData,
                bbox=bbox,
                zorder=1,
                clip_on=clip_on,
                label_pos=get_param_value(label_pos, "label_pos"),
                labels_horizontal=not get_param_value(rotate, "rotate"),
                ax=ax,
            )

    if hide_ticks:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    return text_items


def draw_circular(G, **kwargs):
    """Draw the graph `G` with a circular layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.circular_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
        A networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    The layout is computed each time this function is called. For
    repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.circular_layout` directly and reuse the result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.circular_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.draw_circular(G)

    See Also
    --------
    :func:`~networkx.drawing.layout.circular_layout`
    """
    draw(G, circular_layout(G), **kwargs)


def draw_kamada_kawai(G, **kwargs):
    """Draw the graph `G` with a Kamada-Kawai force-directed layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.kamada_kawai_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
        A networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.kamada_kawai_layout` directly and reuse the
    result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.kamada_kawai_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.draw_kamada_kawai(G)

    See Also
    --------
    :func:`~networkx.drawing.layout.kamada_kawai_layout`
    """
    draw(G, kamada_kawai_layout(G), **kwargs)


def draw_random(G, **kwargs):
    """Draw the graph `G` with a random layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.random_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
        A networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.random_layout` directly and reuse the result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.random_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.lollipop_graph(4, 3)
    >>> nx.draw_random(G)

    See Also
    --------
    :func:`~networkx.drawing.layout.random_layout`
    """
    draw(G, random_layout(G), **kwargs)


def draw_spectral(G, **kwargs):
    """Draw the graph `G` with a spectral 2D layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.spectral_layout(G), **kwargs)

    For more information about how node positions are determined, see
    `~networkx.drawing.layout.spectral_layout`.

    Parameters
    ----------
    G : graph
        A networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.spectral_layout` directly and reuse the result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.spectral_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.draw_spectral(G)

    See Also
    --------
    :func:`~networkx.drawing.layout.spectral_layout`
    """
    draw(G, spectral_layout(G), **kwargs)


def draw_spring(G, **kwargs):
    """Draw the graph `G` with a spring layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.spring_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
        A networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    `~networkx.drawing.layout.spring_layout` is also the default layout for
    `draw`, so this function is equivalent to `draw`.

    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.spring_layout` directly and reuse the result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.spring_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(20)
    >>> nx.draw_spring(G)

    See Also
    --------
    draw
    :func:`~networkx.drawing.layout.spring_layout`
    """
    draw(G, spring_layout(G), **kwargs)


def draw_shell(G, nlist=None, **kwargs):
    """Draw networkx graph `G` with shell layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.shell_layout(G, nlist=nlist), **kwargs)

    Parameters
    ----------
    G : graph
        A networkx graph

    nlist : list of list of nodes, optional
        A list containing lists of nodes representing the shells.
        Default is `None`, meaning all nodes are in a single shell.
        See `~networkx.drawing.layout.shell_layout` for details.

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.shell_layout` directly and reuse the result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.shell_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> shells = [[0], [1, 2, 3]]
    >>> nx.draw_shell(G, nlist=shells)

    See Also
    --------
    :func:`~networkx.drawing.layout.shell_layout`
    """
    draw(G, shell_layout(G, nlist=nlist), **kwargs)


def draw_planar(G, **kwargs):
    """Draw a planar networkx graph `G` with planar layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.planar_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
        A planar networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Raises
    ------
    NetworkXException
        When `G` is not planar

    Notes
    -----
    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.planar_layout` directly and reuse the result::

        >>> G = nx.path_graph(5)
        >>> pos = nx.planar_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.draw_planar(G)

    See Also
    --------
    :func:`~networkx.drawing.layout.planar_layout`
    """
    draw(G, planar_layout(G), **kwargs)


def draw_forceatlas2(G, **kwargs):
    """Draw a networkx graph with forceatlas2 layout.

    This is a convenience function equivalent to::

       nx.draw(G, pos=nx.forceatlas2_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
       A networkx graph

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the pos parameter which is not used by this
       function.
    """
    draw(G, forceatlas2_layout(G), **kwargs)


def apply_alpha(colors, alpha, elem_list, cmap=None, vmin=None, vmax=None):
    """Apply an alpha (or list of alphas) to the colors provided.

    Parameters
    ----------

    colors : color string or array of floats (default='r')
        Color of element. Can be a single color format string,
        or a sequence of colors with the same length as nodelist.
        If numeric values are specified they will be mapped to
        colors using the cmap and vmin,vmax parameters.  See
        matplotlib.scatter for more details.

    alpha : float or array of floats
        Alpha values for elements. This can be a single alpha value, in
        which case it will be applied to all the elements of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    elem_list : array of networkx objects
        The list of elements which are being colored. These could be nodes,
        edges or labels.

    cmap : matplotlib colormap
        Color map for use if colors is a list of floats corresponding to points
        on a color mapping.

    vmin, vmax : float
        Minimum and maximum values for normalizing colors if a colormap is used

    Returns
    -------

    rgba_colors : numpy ndarray
        Array containing RGBA format values for each of the node colours.

    """
    from itertools import cycle, islice

    import matplotlib as mpl
    import matplotlib.cm  # call as mpl.cm
    import matplotlib.colors  # call as mpl.colors
    import numpy as np

    # If we have been provided with a list of numbers as long as elem_list,
    # apply the color mapping.
    if len(colors) == len(elem_list) and isinstance(colors[0], Number):
        mapper = mpl.cm.ScalarMappable(cmap=cmap)
        mapper.set_clim(vmin, vmax)
        rgba_colors = mapper.to_rgba(colors)
    # Otherwise, convert colors to matplotlib's RGB using the colorConverter
    # object.  These are converted to numpy ndarrays to be consistent with the
    # to_rgba method of ScalarMappable.
    else:
        try:
            rgba_colors = np.array([mpl.colors.colorConverter.to_rgba(colors)])
        except ValueError:
            rgba_colors = np.array(
                [mpl.colors.colorConverter.to_rgba(color) for color in colors]
            )
    # Set the final column of the rgba_colors to have the relevant alpha values
    try:
        # If alpha is longer than the number of colors, resize to the number of
        # elements.  Also, if rgba_colors.size (the number of elements of
        # rgba_colors) is the same as the number of elements, resize the array,
        # to avoid it being interpreted as a colormap by scatter()
        if len(alpha) > len(rgba_colors) or rgba_colors.size == len(elem_list):
            rgba_colors = np.resize(rgba_colors, (len(elem_list), 4))
            rgba_colors[1:, 0] = rgba_colors[0, 0]
            rgba_colors[1:, 1] = rgba_colors[0, 1]
            rgba_colors[1:, 2] = rgba_colors[0, 2]
        rgba_colors[:, 3] = list(islice(cycle(alpha), len(rgba_colors)))
    except TypeError:
        rgba_colors[:, -1] = alpha
    return rgba_colors
<|endoftext|>"""
***************
Graphviz AGraph
***************

Interface to pygraphviz AGraph class.

Examples
--------
>>> G = nx.complete_graph(5)
>>> A = nx.nx_agraph.to_agraph(G)
>>> H = nx.nx_agraph.from_agraph(A)

See Also
--------
 - Pygraphviz: http://pygraphviz.github.io/
 - Graphviz:      https://www.graphviz.org
 - DOT Language:  http://www.graphviz.org/doc/info/lang.html
"""

import os
import tempfile

import networkx as nx

__all__ = [
    "from_agraph",
    "to_agraph",
    "write_dot",
    "read_dot",
    "graphviz_layout",
    "pygraphviz_layout",
    "view_pygraphviz",
]


@nx._dispatchable(graphs=None, returns_graph=True)
def from_agraph(A, create_using=None):
    """Returns a NetworkX Graph or DiGraph from a PyGraphviz graph.

    Parameters
    ----------
    A : PyGraphviz AGraph
      A graph created with PyGraphviz

    create_using : NetworkX graph constructor, optional (default=None)
       Graph type to create. If graph instance, then cleared before populated.
       If `None`, then the appropriate Graph type is inferred from `A`.

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> A = nx.nx_agraph.to_agraph(K5)
    >>> G = nx.nx_agraph.from_agraph(A)

    Notes
    -----
    The Graph G will have a dictionary G.graph_attr containing
    the default graphviz attributes for graphs, nodes and edges.

    Default node attributes will be in the dictionary G.node_attr
    which is keyed by node.

    Edge attributes will be returned as edge data in G.  With
    edge_attr=False the edge data will be the Graphviz edge weight
    attribute or the value 1 if no edge weight attribute is found.

    """
    if create_using is None:
        if A.is_directed():
            if A.is_strict():
                create_using = nx.DiGraph
            else:
                create_using = nx.MultiDiGraph
        else:
            if A.is_strict():
                create_using = nx.Graph
            else:
                create_using = nx.MultiGraph

    # assign defaults
    N = nx.empty_graph(0, create_using)
    if A.name is not None:
        N.name = A.name

    # add graph attributes
    N.graph.update(A.graph_attr)

    # add nodes, attributes to N.node_attr
    for n in A.nodes():
        str_attr = {str(k): v for k, v in n.attr.items()}
        N.add_node(str(n), **str_attr)

    # add edges, assign edge data as dictionary of attributes
    for e in A.edges():
        u, v = str(e[0]), str(e[1])
        attr = dict(e.attr)
        str_attr = {str(k): v for k, v in attr.items()}
        if not N.is_multigraph():
            if e.name is not None:
                str_attr["key"] = e.name
            N.add_edge(u, v, **str_attr)
        else:
            N.add_edge(u, v, key=e.name, **str_attr)

    # add default attributes for graph, nodes, and edges
    # hang them on N.graph_attr
    N.graph["graph"] = dict(A.graph_attr)
    N.graph["node"] = dict(A.node_attr)
    N.graph["edge"] = dict(A.edge_attr)
    return N


def to_agraph(N):
    """Returns a pygraphviz graph from a NetworkX graph N.

    Parameters
    ----------
    N : NetworkX graph
      A graph created with NetworkX

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> A = nx.nx_agraph.to_agraph(K5)

    Notes
    -----
    If N has an dict N.graph_attr an attempt will be made first
    to copy properties attached to the graph (see from_agraph)
    and then updated with the calling arguments if any.

    """
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError("requires pygraphviz http://pygraphviz.github.io/") from err
    directed = N.is_directed()
    strict = nx.number_of_selfloops(N) == 0 and not N.is_multigraph()

    A = pygraphviz.AGraph(name=N.name, strict=strict, directed=directed)

    # default graph attributes
    A.graph_attr.update(N.graph.get("graph", {}))
    A.node_attr.update(N.graph.get("node", {}))
    A.edge_attr.update(N.graph.get("edge", {}))

    A.graph_attr.update(
        (k, v) for k, v in N.graph.items() if k not in ("graph", "node", "edge")
    )

    # add nodes
    for n, nodedata in N.nodes(data=True):
        A.add_node(n)
        # Add node data
        a = A.get_node(n)
        for key, val in nodedata.items():
            if key == "pos":
                a.attr["pos"] = f"{val[0]},{val[1]}!"
            else:
                a.attr[key] = str(val)

    # loop over edges
    if N.is_multigraph():
        for u, v, key, edgedata in N.edges(data=True, keys=True):
            str_edgedata = {k: str(v) for k, v in edgedata.items() if k != "key"}
            A.add_edge(u, v, key=str(key))
            # Add edge data
            a = A.get_edge(u, v)
            a.attr.update(str_edgedata)

    else:
        for u, v, edgedata in N.edges(data=True):
            str_edgedata = {k: str(v) for k, v in edgedata.items()}
            A.add_edge(u, v)
            # Add edge data
            a = A.get_edge(u, v)
            a.attr.update(str_edgedata)

    return A


def write_dot(G, path):
    """Write NetworkX graph G to Graphviz dot format on path.

    Parameters
    ----------
    G : graph
       A networkx graph
    path : filename
       Filename or file handle to write

    Notes
    -----
    To use a specific graph layout, call ``A.layout`` prior to `write_dot`.
    Note that some graphviz layouts are not guaranteed to be deterministic,
    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.
    """
    A = to_agraph(G)
    A.write(path)
    A.clear()
    return


@nx._dispatchable(name="agraph_read_dot", graphs=None, returns_graph=True)
def read_dot(path):
    """Returns a NetworkX graph from a dot file on path.

    Parameters
    ----------
    path : file or string
       File name or file handle to read.
    """
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError(
            "read_dot() requires pygraphviz http://pygraphviz.github.io/"
        ) from err
    A = pygraphviz.AGraph(file=path)
    gr = from_agraph(A)
    A.clear()
    return gr


def graphviz_layout(G, prog="neato", root=None, args=""):
    """Create node positions for G using Graphviz.

    Parameters
    ----------
    G : NetworkX graph
      A graph created with NetworkX
    prog : string
      Name of Graphviz layout program
    root : string, optional
      Root node for twopi layout
    args : string, optional
      Extra arguments to Graphviz layout program

    Returns
    -------
    Dictionary of x, y, positions keyed by node.

    Examples
    --------
    >>> G = nx.petersen_graph()
    >>> pos = nx.nx_agraph.graphviz_layout(G)
    >>> pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    Notes
    -----
    This is a wrapper for pygraphviz_layout.

    Note that some graphviz layouts are not guaranteed to be deterministic,
    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.
    """
    return pygraphviz_layout(G, prog=prog, root=root, args=args)


def pygraphviz_layout(G, prog="neato", root=None, args=""):
    """Create node positions for G using Graphviz.

    Parameters
    ----------
    G : NetworkX graph
      A graph created with NetworkX
    prog : string
      Name of Graphviz layout program
    root : string, optional
      Root node for twopi layout
    args : string, optional
      Extra arguments to Graphviz layout program

    Returns
    -------
    node_pos : dict
      Dictionary of x, y, positions keyed by node.

    Examples
    --------
    >>> G = nx.petersen_graph()
    >>> pos = nx.nx_agraph.graphviz_layout(G)
    >>> pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    Notes
    -----
    If you use complex node objects, they may have the same string
    representation and GraphViz could treat them as the same node.
    The layout may assign both nodes a single location. See Issue #1568
    If this occurs in your case, consider relabeling the nodes just
    for the layout computation using something similar to::

        >>> H = nx.convert_node_labels_to_integers(G, label_attribute="node_label")
        >>> H_layout = nx.nx_agraph.pygraphviz_layout(G, prog="dot")
        >>> G_layout = {H.nodes[n]["node_label"]: p for n, p in H_layout.items()}

    Note that some graphviz layouts are not guaranteed to be deterministic,
    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.
    """
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError("requires pygraphviz http://pygraphviz.github.io/") from err
    if root is not None:
        args += f"-Groot={root}"
    A = to_agraph(G)
    A.layout(prog=prog, args=args)
    node_pos = {}
    for n in G:
        node = pygraphviz.Node(A, n)
        try:
            xs = node.attr["pos"].split(",")
            node_pos[n] = tuple(float(x) for x in xs)
        except:
            print("no position for node", n)
            node_pos[n] = (0.0, 0.0)
    return node_pos


@nx.utils.open_file(5, "w+b")
def view_pygraphviz(
    G, edgelabel=None, prog="dot", args="", suffix="", path=None, show=True
):
    """Views the graph G using the specified layout algorithm.

    Parameters
    ----------
    G : NetworkX graph
        The machine to draw.
    edgelabel : str, callable, None
        If a string, then it specifies the edge attribute to be displayed
        on the edge labels. If a callable, then it is called for each
        edge and it should return the string to be displayed on the edges.
        The function signature of `edgelabel` should be edgelabel(data),
        where `data` is the edge attribute dictionary.
    prog : string
        Name of Graphviz layout program.
    args : str
        Additional arguments to pass to the Graphviz layout program.
    suffix : str
        If `filename` is None, we save to a temporary file.  The value of
        `suffix` will appear at the tail end of the temporary filename.
    path : str, None
        The filename used to save the image.  If None, save to a temporary
        file.  File formats are the same as those from pygraphviz.agraph.draw.
    show : bool, default = True
        Whether to display the graph with :mod:`PIL.Image.show`,
        default is `True`. If `False`, the rendered graph is still available
        at `path`.

    Returns
    -------
    path : str
        The filename of the generated image.
    A : PyGraphviz graph
        The PyGraphviz graph instance used to generate the image.

    Notes
    -----
    If this function is called in succession too quickly, sometimes the
    image is not displayed. So you might consider time.sleep(.5) between
    calls if you experience problems.

    Note that some graphviz layouts are not guaranteed to be deterministic,
    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.

    """
    if not len(G):
        raise nx.NetworkXException("An empty graph cannot be drawn.")

    # If we are providing default values for graphviz, these must be set
    # before any nodes or edges are added to the PyGraphviz graph object.
    # The reason for this is that default values only affect incoming objects.
    # If you change the default values after the objects have been added,
    # then they inherit no value and are set only if explicitly set.

    # to_agraph() uses these values.
    attrs = ["edge", "node", "graph"]
    for attr in attrs:
        if attr not in G.graph:
            G.graph[attr] = {}

    # These are the default values.
    edge_attrs = {"fontsize": "10"}
    node_attrs = {
        "style": "filled",
        "fillcolor": "#0000FF40",
        "height": "0.75",
        "width": "0.75",
        "shape": "circle",
    }
    graph_attrs = {}

    def update_attrs(which, attrs):
        # Update graph attributes. Return list of those which were added.
        added = []
        for k, v in attrs.items():
            if k not in G.graph[which]:
                G.graph[which][k] = v
                added.append(k)

    def clean_attrs(which, added):
        # Remove added attributes
        for attr in added:
            del G.graph[which][attr]
        if not G.graph[which]:
            del G.graph[which]

    # Update all default values
    update_attrs("edge", edge_attrs)
    update_attrs("node", node_attrs)
    update_attrs("graph", graph_attrs)

    # Convert to agraph, so we inherit default values
    A = to_agraph(G)

    # Remove the default values we added to the original graph.
    clean_attrs("edge", edge_attrs)
    clean_attrs("node", node_attrs)
    clean_attrs("graph", graph_attrs)

    # If the user passed in an edgelabel, we update the labels for all edges.
    if edgelabel is not None:
        if not callable(edgelabel):

            def func(data):
                return "".join(["  ", str(data[edgelabel]), "  "])

        else:
            func = edgelabel

        # update all the edge labels
        if G.is_multigraph():
            for u, v, key, data in G.edges(keys=True, data=True):
                # PyGraphviz doesn't convert the key to a string. See #339
                edge = A.get_edge(u, v, str(key))
                edge.attr["label"] = str(func(data))
        else:
            for u, v, data in G.edges(data=True):
                edge = A.get_edge(u, v)
                edge.attr["label"] = str(func(data))

    if path is None:
        ext = "png"
        if suffix:
            suffix = f"_{suffix}.{ext}"
        else:
            suffix = f".{ext}"
        path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    else:
        # Assume the decorator worked and it is a file-object.
        pass

    # Write graph to file
    A.draw(path=path, format=None, prog=prog, args=args)
    path.close()

    # Show graph in a new window (depends on platform configuration)
    if show:
        from PIL import Image

        Image.open(path.name).show()

    return path.name, A
<|endoftext|>"""
******
Layout
******

Node positioning algorithms for graph drawing.

For `random_layout()` the possible resulting shape
is a square of side [0, scale] (default: [0, 1])
Changing `center` shifts the layout by that amount.

For the other layout routines, the extent is
[center - scale, center + scale] (default: [-1, 1]).

Warning: Most layout routines have only been tested in 2-dimensions.

"""

import networkx as nx
from networkx.utils import np_random_state

__all__ = [
    "bipartite_layout",
    "circular_layout",
    "forceatlas2_layout",
    "kamada_kawai_layout",
    "random_layout",
    "rescale_layout",
    "rescale_layout_dict",
    "shell_layout",
    "spring_layout",
    "spectral_layout",
    "planar_layout",
    "fruchterman_reingold_layout",
    "spiral_layout",
    "multipartite_layout",
    "bfs_layout",
    "arf_layout",
]


def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center


@np_random_state(3)
def random_layout(G, center=None, dim=2, seed=None):
    """Position nodes uniformly at random in the unit square.

    For every node, a position is generated by choosing each of dim
    coordinates uniformly at random on the interval [0.0, 1.0).

    NumPy (http://scipy.org) is required for this function.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    seed : int, RandomState instance or None  optional (default=None)
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.lollipop_graph(4, 3)
    >>> pos = nx.random_layout(G)

    """
    import numpy as np

    G, center = _process_params(G, center, dim)
    pos = seed.rand(len(G), dim) + center
    pos = pos.astype(np.float32)
    pos = dict(zip(G, pos))

    return pos


def circular_layout(G, scale=1, center=None, dim=2):
    # dim=2 only
    """Position nodes on a circle.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.
        If dim>2, the remaining dimensions are set to zero
        in the returned positions.
        If dim<2, a ValueError is raised.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Raises
    ------
    ValueError
        If dim < 2

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.circular_layout(G)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    import numpy as np

    if dim < 2:
        raise ValueError("cannot handle dimensions < 2")

    G, center = _process_params(G, center, dim)

    paddims = max(0, (dim - 2))

    if len(G) == 0:
        pos = {}
    elif len(G) == 1:
        pos = {nx.utils.arbitrary_element(G): center}
    else:
        # Discard the extra angle since it matches 0 radians.
        theta = np.linspace(0, 1, len(G) + 1)[:-1] * 2 * np.pi
        theta = theta.astype(np.float32)
        pos = np.column_stack(
            [np.cos(theta), np.sin(theta), np.zeros((len(G), paddims))]
        )
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(G, pos))

    return pos


def shell_layout(G, nlist=None, rotate=None, scale=1, center=None, dim=2):
    """Position nodes in concentric circles.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    nlist : list of lists
       List of node lists for each shell.

    rotate : angle in radians (default=pi/len(nlist))
       Angle by which to rotate the starting position of each shell
       relative to the starting position of the previous shell.
       To recreate behavior before v2.5 use rotate=0.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout, currently only dim=2 is supported.
        Other dimension values result in a ValueError.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Raises
    ------
    ValueError
        If dim != 2

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> shells = [[0], [1, 2, 3]]
    >>> pos = nx.shell_layout(G, shells)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    import numpy as np

    if dim != 2:
        raise ValueError("can only handle 2 dimensions")

    G, center = _process_params(G, center, dim)

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G): center}

    if nlist is None:
        # draw the whole graph in one shell
        nlist = [list(G)]

    radius_bump = scale / len(nlist)

    if len(nlist[0]) == 1:
        # single node at center
        radius = 0.0
    else:
        # else start at r=1
        radius = radius_bump

    if rotate is None:
        rotate = np.pi / len(nlist)
    first_theta = rotate
    npos = {}
    for nodes in nlist:
        # Discard the last angle (endpoint=False) since 2*pi matches 0 radians
        theta = (
            np.linspace(0, 2 * np.pi, len(nodes), endpoint=False, dtype=np.float32)
            + first_theta
        )
        pos = radius * np.column_stack([np.cos(theta), np.sin(theta)]) + center
        npos.update(zip(nodes, pos))
        radius += radius_bump
        first_theta += rotate

    return npos


def bipartite_layout(
    G, nodes, align="vertical", scale=1, center=None, aspect_ratio=4 / 3
):
    """Position nodes in two straight lines.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    nodes : list or container
        Nodes in one node set of the bipartite graph.
        This set will be placed on left or top.

    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    aspect_ratio : number (default=4/3):
        The ratio of the width to the height of the layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.bipartite.gnmk_random_graph(3, 5, 10, seed=123)
    >>> top = nx.bipartite.sets(G)[0]
    >>> pos = nx.bipartite_layout(G, top)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """

    import numpy as np

    if align not in ("vertical", "horizontal"):
        msg = "align must be either vertical or horizontal."
        raise ValueError(msg)

    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}

    height = 1
    width = aspect_ratio * height
    offset = (width / 2, height / 2)

    top = dict.fromkeys(nodes)
    bottom = [v for v in G if v not in top]
    nodes = list(top) + bottom

    left_xs = np.repeat(0, len(top))
    right_xs = np.repeat(width, len(bottom))
    left_ys = np.linspace(0, height, len(top))
    right_ys = np.linspace(0, height, len(bottom))

    top_pos = np.column_stack([left_xs, left_ys]) - offset
    bottom_pos = np.column_stack([right_xs, right_ys]) - offset

    pos = np.concatenate([top_pos, bottom_pos])
    pos = rescale_layout(pos, scale=scale) + center
    if align == "horizontal":
        pos = pos[:, ::-1]  # swap x and y coords
    pos = dict(zip(nodes, pos))
    return pos


@np_random_state(10)
def spring_layout(
    G,
    k=None,
    pos=None,
    fixed=None,
    iterations=50,
    threshold=1e-4,
    weight="weight",
    scale=1,
    center=None,
    dim=2,
    seed=None,
):
    """Position nodes using Fruchterman-Reingold force-directed algorithm.

    The algorithm simulates a force-directed representation of the network
    treating edges as springs holding nodes close, while treating nodes
    as repelling objects, sometimes called an anti-gravity force.
    Simulation continues until the positions are close to an equilibrium.

    There are some hard-coded values: minimal distance between
    nodes (0.01) and "temperature" of 0.1 to ensure nodes don't fly away.
    During the simulation, `k` helps determine the distance between nodes,
    though `scale` and `center` determine the size and place after
    rescaling occurs at the end of the simulation.

    Fixing some nodes doesn't allow them to move in the simulation.
    It also turns off the rescaling feature at the simulation's end.
    In addition, setting `scale` to `None` turns off rescaling.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    k : float (default=None)
        Optimal distance between nodes.  If None the distance is set to
        1/sqrt(n) where n is the number of nodes.  Increase this value
        to move nodes farther apart.

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        random initial positions.

    fixed : list or None  optional (default=None)
        Nodes to keep fixed at initial position.
        Nodes not in ``G.nodes`` are ignored.
        ValueError raised if `fixed` specified and `pos` not.

    iterations : int  optional (default=50)
        Maximum number of iterations taken

    threshold: float optional (default = 1e-4)
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  Larger means a stronger attractive force.
        If None, then all edge weights are 1.

    scale : number or None (default: 1)
        Scale factor for positions. Not used unless `fixed is None`.
        If scale is None, no rescaling is performed.

    center : array-like or None
        Coordinate pair around which to center the layout.
        Not used unless `fixed is None`.

    dim : int
        Dimension of layout.

    seed : int, RandomState instance or None  optional (default=None)
        Used only for the initial positions in the algorithm.
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spring_layout(G)

    # The same using longer but equivalent function name
    >>> pos = nx.fruchterman_reingold_layout(G)
    """
    import numpy as np

    G, center = _process_params(G, center, dim)

    if fixed is not None:
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(G)}
        fixed = np.asarray([nfixed[node] for node in fixed if node in nfixed])

    if pos is not None:
        # Determine size of existing domain to adjust initial positions
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(len(G), dim) * dom_size + center

        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G.nodes()): center}

    try:
        # Sparse matrix
        if len(G) < 500:  # sparse solver for large graphs
            raise ValueError
        A = nx.to_scipy_sparse_array(G, weight=weight, dtype="f")
        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = _sparse_fruchterman_reingold(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed
        )
    except ValueError:
        A = nx.to_numpy_array(G, weight=weight)
        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = _fruchterman_reingold(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed
        )
    if fixed is None and scale is not None:
        pos = rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    return pos


fruchterman_reingold_layout = spring_layout


@np_random_state(7)
def _fruchterman_reingold(
    A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    import numpy as np

    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # matrix of difference between points
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # distance between points
        distance = np.linalg.norm(delta, axis=-1)
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        # displacement "force"
        displacement = np.einsum(
            "ijk,ij->ik", delta, (k * k / distance**2 - A * distance / k)
        )
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        pos += delta_pos
        # cool temperature
        t -= dt
        if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            break
    return pos


@np_random_state(7)
def _sparse_fruchterman_reingold(
    A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    # Sparse version
    import numpy as np
    import scipy as sp

    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err
    # make sure we have a LIst of Lists representation
    try:
        A = A.tolil()
    except AttributeError:
        A = (sp.sparse.coo_array(A)).tolil()

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # no fixed nodes
    if fixed is None:
        fixed = []

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (iterations + 1)

    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        # loop over rows
        for i in range(A.shape[0]):
            if i in fixed:
                continue
            # difference between this row's node position and all others
            delta = (pos[i] - pos).T
            # distance between points
            distance = np.sqrt((delta**2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            # the adjacency matrix row
            Ai = A.getrowview(i).toarray()  # TODO: revisit w/ sparse 1D container
            # displacement "force"
            displacement[:, i] += (
                delta * (k * k / distance**2 - Ai * distance / k)
            ).sum(axis=1)
        # update positions
        length = np.sqrt((displacement**2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos
        # cool temperature
        t -= dt
        if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            break
    return pos


def kamada_kawai_layout(
    G, dist=None, pos=None, weight="weight", scale=1, center=None, dim=2
):
    """Position nodes using Kamada-Kawai path-length cost-function.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    dist : dict (default=None)
        A two-level dictionary of optimal distances between nodes,
        indexed by source and destination node.
        If None, the distance is computed using shortest_path_length().

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        circular_layout() for dim >= 2 and a linear layout for dim == 1.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.kamada_kawai_layout(G)
    """
    import numpy as np

    G, center = _process_params(G, center, dim)
    nNodes = len(G)
    if nNodes == 0:
        return {}

    if dist is None:
        dist = dict(nx.shortest_path_length(G, weight=weight))
    dist_mtx = 1e6 * np.ones((nNodes, nNodes))
    for row, nr in enumerate(G):
        if nr not in dist:
            continue
        rdist = dist[nr]
        for col, nc in enumerate(G):
            if nc not in rdist:
                continue
            dist_mtx[row][col] = rdist[nc]

    if pos is None:
        if dim >= 3:
            pos = random_layout(G, dim=dim)
        elif dim == 2:
            pos = circular_layout(G, dim=dim)
        else:
            pos = dict(zip(G, np.linspace(0, 1, len(G))))
    pos_arr = np.array([pos[n] for n in G])

    pos = _kamada_kawai_solve(dist_mtx, pos_arr, dim)

    pos = rescale_layout(pos, scale=scale) + center
    return dict(zip(G, pos))


def _kamada_kawai_solve(dist_mtx, pos_arr, dim):
    # Anneal node locations based on the Kamada-Kawai cost-function,
    # using the supplied matrix of preferred inter-node distances,
    # and starting locations.

    import numpy as np
    import scipy as sp

