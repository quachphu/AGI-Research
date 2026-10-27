    chain_edges = set(chain.from_iterable(chains))
    if root is not None:
        H = H.subgraph(nx.node_connected_component(H, root)).copy()
    for u, v in H.edges():
        if (u, v) not in chain_edges and (v, u) not in chain_edges:
            if multigraph and len(G[u][v]) > 1:
                continue
            yield u, v


@not_implemented_for("directed")
@nx._dispatchable
def has_bridges(G, root=None):
    """Decide whether a graph has any bridges.

    A *bridge* in a graph is an edge whose removal causes the number of
    connected components of the graph to increase.

    Parameters
    ----------
    G : undirected graph

    root : node (optional)
       A node in the graph `G`. If specified, only the bridges in the
       connected component containing this node will be considered.

    Returns
    -------
    bool
       Whether the graph (or the connected component containing `root`)
       has any bridges.

    Raises
    ------
    NodeNotFound
       If `root` is not in the graph `G`.

    NetworkXNotImplemented
        If `G` is a directed graph.

    Examples
    --------
    The barbell graph with parameter zero has a single bridge::

        >>> G = nx.barbell_graph(10, 0)
        >>> nx.has_bridges(G)
        True

    On the other hand, the cycle graph has no bridges::

        >>> G = nx.cycle_graph(5)
        >>> nx.has_bridges(G)
        False

    Notes
    -----
    This implementation uses the :func:`networkx.bridges` function, so
    it shares its worst-case time complexity, $O(m + n)$, ignoring
    polylogarithmic factors, where $n$ is the number of nodes in the
    graph and $m$ is the number of edges.

    """
    try:
        next(bridges(G, root=root))
    except StopIteration:
        return False
    else:
        return True


@not_implemented_for("multigraph")
@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def local_bridges(G, with_span=True, weight=None):
    """Iterate over local bridges of `G` optionally computing the span

    A *local bridge* is an edge whose endpoints have no common neighbors.
    That is, the edge is not part of a triangle in the graph.

    The *span* of a *local bridge* is the shortest path length between
    the endpoints if the local bridge is removed.

    Parameters
    ----------
    G : undirected graph

    with_span : bool
        If True, yield a 3-tuple `(u, v, span)`

    weight : function, string or None (default: None)
        If function, used to compute edge weights for the span.
        If string, the edge data attribute used in calculating span.
        If None, all edges have weight 1.

    Yields
    ------
    e : edge
        The local bridges as an edge 2-tuple of nodes `(u, v)` or
        as a 3-tuple `(u, v, span)` when `with_span is True`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a directed graph or multigraph.

    Examples
    --------
    A cycle graph has every edge a local bridge with span N-1.

       >>> G = nx.cycle_graph(9)
       >>> (0, 8, 8) in set(nx.local_bridges(G))
       True
    """
    if with_span is not True:
        for u, v in G.edges:
            if not (set(G[u]) & set(G[v])):
                yield u, v
    else:
        wt = nx.weighted._weight_function(G, weight)
        for u, v in G.edges:
            if not (set(G[u]) & set(G[v])):
                enodes = {u, v}

                def hide_edge(n, nbr, d):
                    if n not in enodes or nbr not in enodes:
                        return wt(n, nbr, d)
                    return None

                try:
                    span = nx.shortest_path_length(G, u, v, weight=hide_edge)
                    yield u, v, span
                except nx.NetworkXNoPath:
                    yield u, v, float("inf")
<|endoftext|>"""Functions for finding and evaluating cuts in a graph."""

from itertools import chain

import networkx as nx

__all__ = [
    "boundary_expansion",
    "conductance",
    "cut_size",
    "edge_expansion",
    "mixing_expansion",
    "node_expansion",
    "normalized_cut_size",
    "volume",
]


# TODO STILL NEED TO UPDATE ALL THE DOCUMENTATION!


@nx._dispatchable(edge_attrs="weight")
def cut_size(G, S, T=None, weight=None):
    """Returns the size of the cut between two sets of nodes.

    A *cut* is a partition of the nodes of a graph into two sets. The
    *cut size* is the sum of the weights of the edges "between" the two
    sets of nodes.

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`. If not specified, this is taken to
        be the set complement of `S`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        Total weight of all edges from nodes in set `S` to nodes in
        set `T` (and, in the case of directed graphs, all edges from
        nodes in `T` to nodes in `S`).

    Examples
    --------
    In the graph with two cliques joined by a single edges, the natural
    bipartition of the graph into two blocks, one for each clique,
    yields a cut of weight one::

        >>> G = nx.barbell_graph(3, 0)
        >>> S = {0, 1, 2}
        >>> T = {3, 4, 5}
        >>> nx.cut_size(G, S, T)
        1

    Each parallel edge in a multigraph is counted when determining the
    cut size::

        >>> G = nx.MultiGraph(["ab", "ab"])
        >>> S = {"a"}
        >>> T = {"b"}
        >>> nx.cut_size(G, S, T)
        2

    Notes
    -----
    In a multigraph, the cut size is the total weight of edges including
    multiplicity.

    """
    edges = nx.edge_boundary(G, S, T, data=weight, default=1)
    if G.is_directed():
        edges = chain(edges, nx.edge_boundary(G, T, S, data=weight, default=1))
    return sum(weight for u, v, weight in edges)


@nx._dispatchable(edge_attrs="weight")
def volume(G, S, weight=None):
    """Returns the volume of a set of nodes.

    The *volume* of a set *S* is the sum of the (out-)degrees of nodes
    in *S* (taking into account parallel edges in multigraphs). [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The volume of the set of nodes represented by `S` in the graph
        `G`.

    See also
    --------
    conductance
    cut_size
    edge_expansion
    edge_boundary
    normalized_cut_size

    References
    ----------
    .. [1] David Gleich.
           *Hierarchical Directed Spectral Graph Partitioning*.
           <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf>

    """
    degree = G.out_degree if G.is_directed() else G.degree
    return sum(d for v, d in degree(S, weight=weight))


@nx._dispatchable(edge_attrs="weight")
def normalized_cut_size(G, S, T=None, weight=None):
    """Returns the normalized size of the cut between two sets of nodes.

    The *normalized cut size* is the cut size times the sum of the
    reciprocal sizes of the volumes of the two sets. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The normalized cut size between the two sets `S` and `T`.

    Notes
    -----
    In a multigraph, the cut size is the total weight of edges including
    multiplicity.

    See also
    --------
    conductance
    cut_size
    edge_expansion
    volume

    References
    ----------
    .. [1] David Gleich.
           *Hierarchical Directed Spectral Graph Partitioning*.
           <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf>

    """
    if T is None:
        T = set(G) - set(S)
    num_cut_edges = cut_size(G, S, T=T, weight=weight)
    volume_S = volume(G, S, weight=weight)
    volume_T = volume(G, T, weight=weight)
    return num_cut_edges * ((1 / volume_S) + (1 / volume_T))


@nx._dispatchable(edge_attrs="weight")
def conductance(G, S, T=None, weight=None):
    """Returns the conductance of two sets of nodes.

    The *conductance* is the quotient of the cut size and the smaller of
    the volumes of the two sets. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The conductance between the two sets `S` and `T`.

    See also
    --------
    cut_size
    edge_expansion
    normalized_cut_size
    volume

    References
    ----------
    .. [1] David Gleich.
           *Hierarchical Directed Spectral Graph Partitioning*.
           <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf>

    """
    if T is None:
        T = set(G) - set(S)
    num_cut_edges = cut_size(G, S, T, weight=weight)
    volume_S = volume(G, S, weight=weight)
    volume_T = volume(G, T, weight=weight)
    return num_cut_edges / min(volume_S, volume_T)


@nx._dispatchable(edge_attrs="weight")
def edge_expansion(G, S, T=None, weight=None):
    """Returns the edge expansion between two node sets.

    The *edge expansion* is the quotient of the cut size and the smaller
    of the cardinalities of the two sets. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The edge expansion between the two sets `S` and `T`.

    See also
    --------
    boundary_expansion
    mixing_expansion
    node_expansion

    References
    ----------
    .. [1] Fan Chung.
           *Spectral Graph Theory*.
           (CBMS Regional Conference Series in Mathematics, No. 92),
           American Mathematical Society, 1997, ISBN 0-8218-0315-8
           <http://www.math.ucsd.edu/~fan/research/revised.html>

    """
    if T is None:
        T = set(G) - set(S)
    num_cut_edges = cut_size(G, S, T=T, weight=weight)
    return num_cut_edges / min(len(S), len(T))


@nx._dispatchable(edge_attrs="weight")
def mixing_expansion(G, S, T=None, weight=None):
    """Returns the mixing expansion between two node sets.

    The *mixing expansion* is the quotient of the cut size and twice the
    number of edges in the graph. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The mixing expansion between the two sets `S` and `T`.

    See also
    --------
    boundary_expansion
    edge_expansion
    node_expansion

    References
    ----------
    .. [1] Vadhan, Salil P.
           "Pseudorandomness."
           *Foundations and Trends
           in Theoretical Computer Science* 7.1–3 (2011): 1–336.
           <https://doi.org/10.1561/0400000010>

    """
    num_cut_edges = cut_size(G, S, T=T, weight=weight)
    num_total_edges = G.number_of_edges()
    return num_cut_edges / (2 * num_total_edges)


# TODO What is the generalization to two arguments, S and T? Does the
# denominator become `min(len(S), len(T))`?
@nx._dispatchable
def node_expansion(G, S):
    """Returns the node expansion of the set `S`.

    The *node expansion* is the quotient of the size of the node
    boundary of *S* and the cardinality of *S*. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    Returns
    -------
    number
        The node expansion of the set `S`.

    See also
    --------
    boundary_expansion
    edge_expansion
    mixing_expansion

    References
    ----------
    .. [1] Vadhan, Salil P.
           "Pseudorandomness."
           *Foundations and Trends
           in Theoretical Computer Science* 7.1–3 (2011): 1–336.
           <https://doi.org/10.1561/0400000010>

    """
    neighborhood = set(chain.from_iterable(G.neighbors(v) for v in S))
    return len(neighborhood) / len(S)


# TODO What is the generalization to two arguments, S and T? Does the
# denominator become `min(len(S), len(T))`?
@nx._dispatchable
def boundary_expansion(G, S):
    """Returns the boundary expansion of the set `S`.

    The *boundary expansion* is the quotient of the size
    of the node boundary and the cardinality of *S*. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    Returns
    -------
    number
        The boundary expansion of the set `S`.

    See also
    --------
    edge_expansion
    mixing_expansion
    node_expansion

    References
    ----------
    .. [1] Vadhan, Salil P.
           "Pseudorandomness."
           *Foundations and Trends in Theoretical Computer Science*
           7.1–3 (2011): 1–336.
           <https://doi.org/10.1561/0400000010>

    """
    return len(nx.node_boundary(G, S)) / len(S)
<|endoftext|>from heapq import heappop, heappush
from itertools import count

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import not_implemented_for, pairwise

__all__ = [
    "all_simple_paths",
    "is_simple_path",
    "shortest_simple_paths",
    "all_simple_edge_paths",
]


@nx._dispatchable
def is_simple_path(G, nodes):
    """Returns True if and only if `nodes` form a simple path in `G`.

    A *simple path* in a graph is a nonempty sequence of nodes in which
    no node appears more than once in the sequence, and each adjacent
    pair of nodes in the sequence is adjacent in the graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    nodes : list
        A list of one or more nodes in the graph `G`.

    Returns
    -------
    bool
        Whether the given list of nodes represents a simple path in `G`.

    Notes
    -----
    An empty list of nodes is not a path but a list of one node is a
    path. Here's an explanation why.

    This function operates on *node paths*. One could also consider
    *edge paths*. There is a bijection between node paths and edge
    paths.

    The *length of a path* is the number of edges in the path, so a list
    of nodes of length *n* corresponds to a path of length *n* - 1.
    Thus the smallest edge path would be a list of zero edges, the empty
    path. This corresponds to a list of one node.

    To convert between a node path and an edge path, you can use code
    like the following::

        >>> from networkx.utils import pairwise
        >>> nodes = [0, 1, 2, 3]
        >>> edges = list(pairwise(nodes))
        >>> edges
        [(0, 1), (1, 2), (2, 3)]
        >>> nodes = [edges[0][0]] + [v for u, v in edges]
        >>> nodes
        [0, 1, 2, 3]

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> nx.is_simple_path(G, [2, 3, 0])
    True
    >>> nx.is_simple_path(G, [0, 2])
    False

    """
    # The empty list is not a valid path. Could also return
    # NetworkXPointlessConcept here.
    if len(nodes) == 0:
        return False

    # If the list is a single node, just check that the node is actually
    # in the graph.
    if len(nodes) == 1:
        return nodes[0] in G

    # check that all nodes in the list are in the graph, if at least one
    # is not in the graph, then this is not a simple path
    if not all(n in G for n in nodes):
        return False

    # If the list contains repeated nodes, then it's not a simple path
    if len(set(nodes)) != len(nodes):
        return False

    # Test that each adjacent pair of nodes is adjacent.
    return all(v in G[u] for u, v in pairwise(nodes))


@nx._dispatchable
def all_simple_paths(G, source, target, cutoff=None):
    """Generate all simple paths in the graph G from source to target.

    A simple path is a path with no repeated nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : nodes
       Single node or iterable of nodes at which to end path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    path_generator: generator
       A generator that produces lists of simple paths.  If there are no paths
       between the source and target within the given cutoff the generator
       produces no output. If it is possible to traverse the same sequence of
       nodes in multiple ways, namely through parallel edges, then it will be
       returned multiple times (once for each viable edge combination).

    Examples
    --------
    This iterator generates lists of nodes::

        >>> G = nx.complete_graph(4)
        >>> for path in nx.all_simple_paths(G, source=0, target=3):
        ...     print(path)
        ...
        [0, 1, 2, 3]
        [0, 1, 3]
        [0, 2, 1, 3]
        [0, 2, 3]
        [0, 3]

    You can generate only those paths that are shorter than a certain
    length by using the `cutoff` keyword argument::

        >>> paths = nx.all_simple_paths(G, source=0, target=3, cutoff=2)
        >>> print(list(paths))
        [[0, 1, 3], [0, 2, 3], [0, 3]]

    To get each path as the corresponding list of edges, you can use the
    :func:`networkx.utils.pairwise` helper function::

        >>> paths = nx.all_simple_paths(G, source=0, target=3)
        >>> for path in map(nx.utils.pairwise, paths):
        ...     print(list(path))
        [(0, 1), (1, 2), (2, 3)]
        [(0, 1), (1, 3)]
        [(0, 2), (2, 1), (1, 3)]
        [(0, 2), (2, 3)]
        [(0, 3)]

    Pass an iterable of nodes as target to generate all paths ending in any of several nodes::

        >>> G = nx.complete_graph(4)
        >>> for path in nx.all_simple_paths(G, source=0, target=[3, 2]):
        ...     print(path)
        ...
        [0, 1, 2]
        [0, 1, 2, 3]
        [0, 1, 3]
        [0, 1, 3, 2]
        [0, 2]
        [0, 2, 1, 3]
        [0, 2, 3]
        [0, 3]
        [0, 3, 1, 2]
        [0, 3, 2]

    The singleton path from ``source`` to itself is considered a simple path and is
    included in the results:

        >>> G = nx.empty_graph(5)
        >>> list(nx.all_simple_paths(G, source=0, target=0))
        [[0]]

        >>> G = nx.path_graph(3)
        >>> list(nx.all_simple_paths(G, source=0, target={0, 1, 2}))
        [[0], [0, 1], [0, 1, 2]]

    Iterate over each path from the root nodes to the leaf nodes in a
    directed acyclic graph using a functional programming approach::

        >>> from itertools import chain
        >>> from itertools import product
        >>> from itertools import starmap
        >>> from functools import partial
        >>>
        >>> chaini = chain.from_iterable
        >>>
        >>> G = nx.DiGraph([(0, 1), (1, 2), (0, 3), (3, 2)])
        >>> roots = (v for v, d in G.in_degree() if d == 0)
        >>> leaves = (v for v, d in G.out_degree() if d == 0)
        >>> all_paths = partial(nx.all_simple_paths, G)
        >>> list(chaini(starmap(all_paths, product(roots, leaves))))
        [[0, 1, 2], [0, 3, 2]]

    The same list computed using an iterative approach::

        >>> G = nx.DiGraph([(0, 1), (1, 2), (0, 3), (3, 2)])
        >>> roots = (v for v, d in G.in_degree() if d == 0)
        >>> leaves = (v for v, d in G.out_degree() if d == 0)
        >>> all_paths = []
        >>> for root in roots:
        ...     for leaf in leaves:
        ...         paths = nx.all_simple_paths(G, root, leaf)
        ...         all_paths.extend(paths)
        >>> all_paths
        [[0, 1, 2], [0, 3, 2]]

    Iterate over each path from the root nodes to the leaf nodes in a
    directed acyclic graph passing all leaves together to avoid unnecessary
    compute::

        >>> G = nx.DiGraph([(0, 1), (2, 1), (1, 3), (1, 4)])
        >>> roots = (v for v, d in G.in_degree() if d == 0)
        >>> leaves = [v for v, d in G.out_degree() if d == 0]
        >>> all_paths = []
        >>> for root in roots:
        ...     paths = nx.all_simple_paths(G, root, leaves)
        ...     all_paths.extend(paths)
        >>> all_paths
        [[0, 1, 3], [0, 1, 4], [2, 1, 3], [2, 1, 4]]

    If parallel edges offer multiple ways to traverse a given sequence of
    nodes, this sequence of nodes will be returned multiple times:

        >>> G = nx.MultiDiGraph([(0, 1), (0, 1), (1, 2)])
        >>> list(nx.all_simple_paths(G, 0, 2))
        [[0, 1, 2], [0, 1, 2]]

    Notes
    -----
    This algorithm uses a modified depth-first search to generate the
    paths [1]_.  A single path can be found in $O(V+E)$ time but the
    number of simple paths in a graph can be very large, e.g. $O(n!)$ in
    the complete graph of order $n$.

    This function does not check that a path exists between `source` and
    `target`. For large graphs, this may result in very long runtimes.
    Consider using `has_path` to check that a path exists between `source` and
    `target` before calling this function on large graphs.

    References
    ----------
    .. [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms",
       Addison Wesley Professional, 3rd ed., 2001.

    See Also
    --------
    all_shortest_paths, shortest_path, has_path

    """
    for edge_path in all_simple_edge_paths(G, source, target, cutoff):
        yield [source] + [edge[1] for edge in edge_path]


@nx._dispatchable
def all_simple_edge_paths(G, source, target, cutoff=None):
    """Generate lists of edges for all simple paths in G from source to target.

    A simple path is a path with no repeated nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : nodes
       Single node or iterable of nodes at which to end path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    path_generator: generator
       A generator that produces lists of simple paths.  If there are no paths
       between the source and target within the given cutoff the generator
       produces no output.
       For multigraphs, the list of edges have elements of the form `(u,v,k)`.
       Where `k` corresponds to the edge key.

    Examples
    --------

    Print the simple path edges of a Graph::

        >>> g = nx.Graph([(1, 2), (2, 4), (1, 3), (3, 4)])
        >>> for path in sorted(nx.all_simple_edge_paths(g, 1, 4)):
        ...     print(path)
        [(1, 2), (2, 4)]
        [(1, 3), (3, 4)]

    Print the simple path edges of a MultiGraph. Returned edges come with
    their associated keys::

        >>> mg = nx.MultiGraph()
        >>> mg.add_edge(1, 2, key="k0")
        'k0'
        >>> mg.add_edge(1, 2, key="k1")
        'k1'
        >>> mg.add_edge(2, 3, key="k0")
        'k0'
        >>> for path in sorted(nx.all_simple_edge_paths(mg, 1, 3)):
        ...     print(path)
        [(1, 2, 'k0'), (2, 3, 'k0')]
        [(1, 2, 'k1'), (2, 3, 'k0')]

    When ``source`` is one of the targets, the empty path starting and ending at
    ``source`` without traversing any edge is considered a valid simple edge path
    and is included in the results:

        >>> G = nx.Graph()
        >>> G.add_node(0)
        >>> paths = list(nx.all_simple_edge_paths(G, 0, 0))
        >>> for path in paths:
        ...     print(path)
        []
        >>> len(paths)
        1


    Notes
    -----
    This algorithm uses a modified depth-first search to generate the
    paths [1]_.  A single path can be found in $O(V+E)$ time but the
    number of simple paths in a graph can be very large, e.g. $O(n!)$ in
    the complete graph of order $n$.

    References
    ----------
    .. [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms",
       Addison Wesley Professional, 3rd ed., 2001.

    See Also
    --------
    all_shortest_paths, shortest_path, all_simple_paths

    """
    if source not in G:
        raise nx.NodeNotFound(f"source node {source} not in graph")

    if target in G:
        targets = {target}
    else:
        try:
            targets = set(target)
        except TypeError as err:
            raise nx.NodeNotFound(f"target node {target} not in graph") from err

    cutoff = cutoff if cutoff is not None else len(G) - 1

    if cutoff >= 0 and targets:
        yield from _all_simple_edge_paths(G, source, targets, cutoff)


def _all_simple_edge_paths(G, source, targets, cutoff):
    # We simulate recursion with a stack, keeping the current path being explored
    # and the outgoing edge iterators at each point in the stack.
    # To avoid unnecessary checks, the loop is structured in a way such that a path
    # is considered for yielding only after a new node/edge is added.
    # We bootstrap the search by adding a dummy iterator to the stack that only yields
    # a dummy edge to source (so that the trivial path has a chance of being included).

    get_edges = (
        (lambda node: G.edges(node, keys=True))
        if G.is_multigraph()
        else (lambda node: G.edges(node))
    )

    # The current_path is a dictionary that maps nodes in the path to the edge that was
    # used to enter that node (instead of a list of edges) because we want both a fast
    # membership test for nodes in the path and the preservation of insertion order.
    current_path = {None: None}
    stack = [iter([(None, source)])]

    while stack:
        # 1. Try to extend the current path.
        next_edge = next((e for e in stack[-1] if e[1] not in current_path), None)
        if next_edge is None:
            # All edges of the last node in the current path have been explored.
            stack.pop()
            current_path.popitem()
            continue
        previous_node, next_node, *_ = next_edge

        # 2. Check if we've reached a target.
        if next_node in targets:
            yield (list(current_path.values()) + [next_edge])[2:]  # remove dummy edge

        # 3. Only expand the search through the next node if it makes sense.
        if len(current_path) - 1 < cutoff and (
            targets - current_path.keys() - {next_node}
        ):
            current_path[next_node] = next_edge
            stack.append(iter(get_edges(next_node)))


@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def shortest_simple_paths(G, source, target, weight=None):
    """Generate all simple paths in the graph G from source to target,
       starting from shortest ones.

    A simple path is a path with no repeated nodes.

    If a weighted shortest path search is to be used, no negative weights
    are allowed.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    weight : string or function
        If it is a string, it is the name of the edge attribute to be
        used as a weight.

        If it is a function, the weight of an edge is the value returned
        by the function. The function must accept exactly three positional
        arguments: the two endpoints of an edge and the dictionary of edge
        attributes for that edge. The function must return a number or None.
        The weight function can be used to hide edges by returning None.
        So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
        will find the shortest red path.

        If None all edges are considered to have unit weight. Default
        value None.

    Returns
    -------
    path_generator: generator
       A generator that produces lists of simple paths, in order from
       shortest to longest.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    NetworkXError
       If source or target nodes are not in the input graph.

    NetworkXNotImplemented
       If the input graph is a Multi[Di]Graph.

    Examples
    --------

    >>> G = nx.cycle_graph(7)
    >>> paths = list(nx.shortest_simple_paths(G, 0, 3))
    >>> print(paths)
    [[0, 1, 2, 3], [0, 6, 5, 4, 3]]

    You can use this function to efficiently compute the k shortest/best
    paths between two nodes.

    >>> from itertools import islice
    >>> def k_shortest_paths(G, source, target, k, weight=None):
    ...     return list(
    ...         islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    ...     )
    >>> for path in k_shortest_paths(G, 0, 3, 2):
    ...     print(path)
    [0, 1, 2, 3]
    [0, 6, 5, 4, 3]

    Notes
    -----
    This procedure is based on algorithm by Jin Y. Yen [1]_.  Finding
    the first $K$ paths requires $O(KN^3)$ operations.

    See Also
    --------
    all_shortest_paths
    shortest_path
    all_simple_paths

    References
    ----------
    .. [1] Jin Y. Yen, "Finding the K Shortest Loopless Paths in a
       Network", Management Science, Vol. 17, No. 11, Theory Series
       (Jul., 1971), pp. 712-716.

    """
    if source not in G:
        raise nx.NodeNotFound(f"source node {source} not in graph")

    if target not in G:
        raise nx.NodeNotFound(f"target node {target} not in graph")

    if weight is None:
        length_func = len
        shortest_path_func = _bidirectional_shortest_path
    else:
        wt = _weight_function(G, weight)

        def length_func(path):
            return sum(
                wt(u, v, G.get_edge_data(u, v)) for (u, v) in zip(path, path[1:])
            )

        shortest_path_func = _bidirectional_dijkstra

    listA = []
    listB = PathBuffer()
    prev_path = None
    while True:
        if not prev_path:
            length, path = shortest_path_func(G, source, target, weight=weight)
            listB.push(length, path)
        else:
            ignore_nodes = set()
            ignore_edges = set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                try:
                    length, spur = shortest_path_func(
                        G,
                        root[-1],
                        target,
                        ignore_nodes=ignore_nodes,
                        ignore_edges=ignore_edges,
                        weight=weight,
                    )
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except nx.NetworkXNoPath:
                    pass
                ignore_nodes.add(root[-1])

        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
        else:
            break


class PathBuffer:
    def __init__(self):
        self.paths = set()
        self.sortedpaths = []
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return path


def _bidirectional_shortest_path(
    G, source, target, ignore_nodes=None, ignore_edges=None, weight=None
):
    """Returns the shortest path between source and target ignoring
       nodes and edges in the containers ignore_nodes and ignore_edges.

    This is a custom modification of the standard bidirectional shortest
    path implementation at networkx.algorithms.unweighted

    Parameters
    ----------
    G : NetworkX graph

    source : node
       starting node for path

    target : node
       ending node for path

    ignore_nodes : container of nodes
       nodes to ignore, optional

    ignore_edges : container of edges
       edges to ignore, optional

    weight : None
       This function accepts a weight argument for convenience of
       shortest_simple_paths function. It will be ignored.

    Returns
    -------
    path: list
       List of nodes in a path from source to target.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    See Also
    --------
    shortest_path

    """
    # call helper to do the real work
    results = _bidirectional_pred_succ(G, source, target, ignore_nodes, ignore_edges)
    pred, succ, w = results

    # build path from pred+w+succ
    path = []
    # from w to target
    while w is not None:
        path.append(w)
        w = succ[w]
    # from source to w
    w = pred[path[0]]
    while w is not None:
        path.insert(0, w)
        w = pred[w]

    return len(path), path


def _bidirectional_pred_succ(G, source, target, ignore_nodes=None, ignore_edges=None):
    """Bidirectional shortest path helper.
    Returns (pred,succ,w) where
    pred is a dictionary of predecessors from w to the source, and
    succ is a dictionary of successors from w to the target.
    """
    # does BFS from both source and target and meets in the middle
    if ignore_nodes and (source in ignore_nodes or target in ignore_nodes):
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
    if target == source:
        return ({target: None}, {source: None}, source)

    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.predecessors
        Gsucc = G.successors
    else:
        Gpred = G.neighbors
        Gsucc = G.neighbors

    # support optional nodes filter
    if ignore_nodes:

        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w

            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)

    # support optional edges filter
    if ignore_edges:
        if G.is_directed():

            def filter_pred_iter(pred_iter):
                def iterate(v):
                    for w in pred_iter(v):
                        if (w, v) not in ignore_edges:
                            yield w

                return iterate

            def filter_succ_iter(succ_iter):
                def iterate(v):
                    for w in succ_iter(v):
                        if (v, w) not in ignore_edges:
                            yield w

                return iterate

            Gpred = filter_pred_iter(Gpred)
            Gsucc = filter_succ_iter(Gsucc)

        else:

            def filter_iter(nodes):
                def iterate(v):
                    for w in nodes(v):
                        if (v, w) not in ignore_edges and (w, v) not in ignore_edges:
                            yield w

                return iterate

            Gpred = filter_iter(Gpred)
            Gsucc = filter_iter(Gsucc)

    # predecessor and successors in search
    pred = {source: None}
    succ = {target: None}

    # initialize fringes, start with forward
    forward_fringe = [source]
    reverse_fringe = [target]

    while forward_fringe and reverse_fringe:
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            for v in this_level:
                for w in Gsucc(v):
                    if w not in pred:
                        forward_fringe.append(w)
                        pred[w] = v
                    if w in succ:
                        # found path
                        return pred, succ, w
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in Gpred(v):
                    if w not in succ:
                        succ[w] = v
                        reverse_fringe.append(w)
                    if w in pred:
                        # found path
                        return pred, succ, w

    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")


def _bidirectional_dijkstra(
    G, source, target, weight="weight", ignore_nodes=None, ignore_edges=None
):
    """Dijkstra's algorithm for shortest paths using bidirectional search.

    This function returns the shortest path between source and target
    ignoring nodes and edges in the containers ignore_nodes and
    ignore_edges.

    This is a custom modification of the standard Dijkstra bidirectional
    shortest path implementation at networkx.algorithms.weighted

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node.

    target : node
        Ending node.

    weight: string, function, optional (default='weight')
        Edge data key or weight function corresponding to the edge weight
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    ignore_nodes : container of nodes
        nodes to ignore, optional

    ignore_edges : container of edges
        edges to ignore, optional

    Returns
    -------
    length : number
        Shortest path length.

    Returns a tuple of two dictionaries keyed by node.
    The first dictionary stores distance from the source.
    The second stores the path from the source to that node.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    In practice  bidirectional Dijkstra is much more than twice as fast as
    ordinary Dijkstra.

    Ordinary Dijkstra expands nodes in a sphere-like manner from the
    source. The radius of this sphere will eventually be the length
    of the shortest path. Bidirectional Dijkstra will expand nodes
    from both the source and the target, making two spheres of half
    this radius. Volume of the first sphere is pi*r*r while the
    others are 2*pi*r/2*r/2, making up half the volume.

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    shortest_path
    shortest_path_length
    """
    if ignore_nodes and (source in ignore_nodes or target in ignore_nodes):
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f"Node {source} not in graph")
        return (0, [source])

    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.predecessors
        Gsucc = G.successors
    else:
        Gpred = G.neighbors
        Gsucc = G.neighbors

    # support optional nodes filter
    if ignore_nodes:

        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w

            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)

    # support optional edges filter
    if ignore_edges:
        if G.is_directed():

            def filter_pred_iter(pred_iter):
                def iterate(v):
                    for w in pred_iter(v):
                        if (w, v) not in ignore_edges:
                            yield w

                return iterate

            def filter_succ_iter(succ_iter):
                def iterate(v):
                    for w in succ_iter(v):
                        if (v, w) not in ignore_edges:
                            yield w

                return iterate

            Gpred = filter_pred_iter(Gpred)
            Gsucc = filter_succ_iter(Gsucc)

        else:

            def filter_iter(nodes):
                def iterate(v):
                    for w in nodes(v):
                        if (v, w) not in ignore_edges and (w, v) not in ignore_edges:
                            yield w

                return iterate

            Gpred = filter_iter(Gpred)
            Gsucc = filter_iter(Gsucc)

    wt = _weight_function(G, weight)
    push = heappush
    pop = heappop
    # Init:   Forward             Backward
    dists = [{}, {}]  # dictionary of final distances
    paths = [{source: [source]}, {target: [target]}]  # dictionary of paths
    fringe = [[], []]  # heap of (distance, node) tuples for
    # extracting next node to expand
    seen = [{source: 0}, {target: 0}]  # dictionary of distances to
    # nodes seen
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), source))
    push(fringe[1], (0, next(c), target))
    # neighs for extracting correct neighbor information
    neighs = [Gsucc, Gpred]
    # variables to hold shortest discovered path
    # finaldist = 1e30000
    finalpath = []
    dir = 1
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        for w in neighs[dir](v):
            if dir == 0:  # forward
                minweight = wt(v, w, G.get_edge_data(v, w))
            else:  # back, must remember to change v,w->w,v
                minweight = wt(w, v, G.get_edge_data(w, v))
            if minweight is None:
                continue
            vwLength = dists[dir][v] + minweight

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
<|endoftext|>import networkx as nx

__all__ = ["s_metric"]


@nx._dispatchable
def s_metric(G):
    """Returns the s-metric [1]_ of graph.

    The s-metric is defined as the sum of the products ``deg(u) * deg(v)``
    for every edge ``(u, v)`` in `G`.

    Parameters
    ----------
    G : graph
        The graph used to compute the s-metric.

    Returns
    -------
    s : float
        The s-metric of the graph.

    References
    ----------
    .. [1] Lun Li, David Alderson, John C. Doyle, and Walter Willinger,
           Towards a Theory of Scale-Free Graphs:
           Definition, Properties, and  Implications (Extended Version), 2005.
           https://arxiv.org/abs/cond-mat/0501169
    """
    return float(sum(G.degree(u) * G.degree(v) for (u, v) in G.edges()))
<|endoftext|>"""This module provides the functions for node classification problem.

The functions in this module are not imported
into the top level `networkx` namespace.
You can access these functions by importing
the `networkx.algorithms.node_classification` modules,
then accessing the functions as attributes of `node_classification`.
For example:

  >>> from networkx.algorithms import node_classification
  >>> G = nx.path_graph(4)
  >>> G.edges()
  EdgeView([(0, 1), (1, 2), (2, 3)])
  >>> G.nodes[0]["label"] = "A"
  >>> G.nodes[3]["label"] = "B"
  >>> node_classification.harmonic_function(G)
  ['A', 'A', 'B', 'B']

References
----------
Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).
Semi-supervised learning using gaussian fields and harmonic functions.
In ICML (Vol. 3, pp. 912-919).
"""

import networkx as nx

__all__ = ["harmonic_function", "local_and_global_consistency"]


@nx.utils.not_implemented_for("directed")
@nx._dispatchable(node_attrs="label_name")
def harmonic_function(G, max_iter=30, label_name="label"):
    """Node classification by Harmonic function

    Function for computing Harmonic function algorithm by Zhu et al.

    Parameters
    ----------
    G : NetworkX Graph
    max_iter : int
        maximum number of iterations allowed
    label_name : string
        name of target labels to predict

    Returns
    -------
    predicted : list
        List of length ``len(G)`` with the predicted labels for each node.

    Raises
    ------
    NetworkXError
        If no nodes in `G` have attribute `label_name`.

    Examples
    --------
    >>> from networkx.algorithms import node_classification
    >>> G = nx.path_graph(4)
    >>> G.nodes[0]["label"] = "A"
    >>> G.nodes[3]["label"] = "B"
    >>> G.nodes(data=True)
    NodeDataView({0: {'label': 'A'}, 1: {}, 2: {}, 3: {'label': 'B'}})
    >>> G.edges()
    EdgeView([(0, 1), (1, 2), (2, 3)])
    >>> predicted = node_classification.harmonic_function(G)
    >>> predicted
    ['A', 'A', 'B', 'B']

    References
    ----------
    Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).
    Semi-supervised learning using gaussian fields and harmonic functions.
    In ICML (Vol. 3, pp. 912-919).
    """
    import numpy as np
    import scipy as sp

    X = nx.to_scipy_sparse_array(G)  # adjacency matrix
    labels, label_dict = _get_label_info(G, label_name)

    if labels.shape[0] == 0:
        raise nx.NetworkXError(
            f"No node on the input graph is labeled by '{label_name}'."
        )

    n_samples = X.shape[0]
    n_classes = label_dict.shape[0]
    F = np.zeros((n_samples, n_classes))

    # Build propagation matrix
    degrees = X.sum(axis=0)
    degrees[degrees == 0] = 1  # Avoid division by 0
    # TODO: csr_array
    D = sp.sparse.csr_array(sp.sparse.diags((1.0 / degrees), offsets=0))
    P = (D @ X).tolil()
    P[labels[:, 0]] = 0  # labels[:, 0] indicates IDs of labeled nodes
    # Build base matrix
    B = np.zeros((n_samples, n_classes))
    B[labels[:, 0], labels[:, 1]] = 1

    for _ in range(max_iter):
        F = (P @ F) + B

    return label_dict[np.argmax(F, axis=1)].tolist()


@nx.utils.not_implemented_for("directed")
@nx._dispatchable(node_attrs="label_name")
def local_and_global_consistency(G, alpha=0.99, max_iter=30, label_name="label"):
    """Node classification by Local and Global Consistency

    Function for computing Local and global consistency algorithm by Zhou et al.

    Parameters
    ----------
    G : NetworkX Graph
    alpha : float
        Clamping factor
    max_iter : int
        Maximum number of iterations allowed
    label_name : string
        Name of target labels to predict

    Returns
    -------
    predicted : list
        List of length ``len(G)`` with the predicted labels for each node.

    Raises
    ------
    NetworkXError
        If no nodes in `G` have attribute `label_name`.

    Examples
    --------
    >>> from networkx.algorithms import node_classification
    >>> G = nx.path_graph(4)
    >>> G.nodes[0]["label"] = "A"
    >>> G.nodes[3]["label"] = "B"
    >>> G.nodes(data=True)
    NodeDataView({0: {'label': 'A'}, 1: {}, 2: {}, 3: {'label': 'B'}})
    >>> G.edges()
    EdgeView([(0, 1), (1, 2), (2, 3)])
    >>> predicted = node_classification.local_and_global_consistency(G)
    >>> predicted
    ['A', 'A', 'B', 'B']

    References
    ----------
    Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
    Learning with local and global consistency.
    Advances in neural information processing systems, 16(16), 321-328.
    """
    import numpy as np
    import scipy as sp

    X = nx.to_scipy_sparse_array(G)  # adjacency matrix
    labels, label_dict = _get_label_info(G, label_name)

    if labels.shape[0] == 0:
        raise nx.NetworkXError(
            f"No node on the input graph is labeled by '{label_name}'."
        )

    n_samples = X.shape[0]
    n_classes = label_dict.shape[0]
    F = np.zeros((n_samples, n_classes))

    # Build propagation matrix
    degrees = X.sum(axis=0)
    degrees[degrees == 0] = 1  # Avoid division by 0
    # TODO: csr_array
    D2 = np.sqrt(sp.sparse.csr_array(sp.sparse.diags((1.0 / degrees), offsets=0)))
    P = alpha * ((D2 @ X) @ D2)
    # Build base matrix
    B = np.zeros((n_samples, n_classes))
    B[labels[:, 0], labels[:, 1]] = 1 - alpha

    for _ in range(max_iter):
        F = (P @ F) + B

    return label_dict[np.argmax(F, axis=1)].tolist()


def _get_label_info(G, label_name):
    """Get and return information of labels from the input graph

    Parameters
    ----------
    G : Network X graph
    label_name : string
        Name of the target label

    Returns
    -------
    labels : numpy array, shape = [n_labeled_samples, 2]
        Array of pairs of labeled node ID and label ID
    label_dict : numpy array, shape = [n_classes]
        Array of labels
        i-th element contains the label corresponding label ID `i`
    """
    import numpy as np

    labels = []
    label_to_id = {}
    lid = 0
    for i, n in enumerate(G.nodes(data=True)):
        if label_name in n[1]:
            label = n[1][label_name]
            if label not in label_to_id:
                label_to_id[label] = lid
                lid += 1
            labels.append([i, label_to_id[label]])
    labels = np.array(labels)
    label_dict = np.array(
        [label for label, _ in sorted(label_to_id.items(), key=lambda x: x[1])]
    )
    return (labels, label_dict)
<|endoftext|>"""
========================
Cycle finding algorithms
========================
"""

from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf

import networkx as nx
from networkx.utils import not_implemented_for, pairwise

__all__ = [
    "cycle_basis",
    "simple_cycles",
    "recursive_simple_cycles",
    "find_cycle",
    "minimum_cycle_basis",
    "chordless_cycles",
    "girth",
]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def cycle_basis(G, root=None):
    """Returns a list of cycles which form a basis for cycles of G.

    A basis for cycles of a network is a minimal collection of
    cycles such that any cycle in the network can be written
    as a sum of cycles in the basis.  Here summation of cycles
    is defined as "exclusive or" of the edges. Cycle bases are
    useful, e.g. when deriving equations for electric circuits
    using Kirchhoff's Laws.

    Parameters
    ----------
    G : NetworkX Graph
    root : node, optional
       Specify starting node for basis.

    Returns
    -------
    A list of cycle lists.  Each cycle list is a list of nodes
    which forms a cycle (loop) in G.

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_cycle(G, [0, 1, 2, 3])
    >>> nx.add_cycle(G, [0, 3, 4, 5])
    >>> nx.cycle_basis(G, 0)
    [[3, 4, 5, 0], [1, 2, 3, 0]]

    Notes
    -----
    This is adapted from algorithm CACM 491 [1]_.

    References
    ----------
    .. [1] Paton, K. An algorithm for finding a fundamental set of
       cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.

    See Also
    --------
    simple_cycles
    minimum_cycle_basis
    """
    gnodes = dict.fromkeys(G)  # set-like object that maintains node order
    cycles = []
    while gnodes:  # loop over connected components
        if root is None:
            root = gnodes.popitem()[0]
        stack = [root]
        pred = {root: root}
        used = {root: set()}
        while stack:  # walk the spanning tree finding cycles
            z = stack.pop()  # use last-in so cycles easier to find
            zused = used[z]
            for nbr in G[z]:
                if nbr not in used:  # new node
                    pred[nbr] = z
                    stack.append(nbr)
                    used[nbr] = {z}
                elif nbr == z:  # self loops
                    cycles.append([z])
                elif nbr not in zused:  # found a cycle
                    pn = used[nbr]
                    cycle = [nbr, z]
                    p = pred[z]
                    while p not in pn:
                        cycle.append(p)
                        p = pred[p]
                    cycle.append(p)
                    cycles.append(cycle)
                    used[nbr].add(z)
        for node in pred:
            gnodes.pop(node, None)
        root = None
    return cycles


@nx._dispatchable
def simple_cycles(G, length_bound=None):
    """Find simple cycles (elementary circuits) of a graph.

    A "simple cycle", or "elementary circuit", is a closed path where
    no node appears twice.  In a directed graph, two simple cycles are distinct
    if they are not cyclic permutations of each other.  In an undirected graph,
    two simple cycles are distinct if they are not cyclic permutations of each
    other nor of the other's reversal.

    Optionally, the cycles are bounded in length.  In the unbounded case, we use
    a nonrecursive, iterator/generator version of Johnson's algorithm [1]_.  In
    the bounded case, we use a version of the algorithm of Gupta and
    Suzumura [2]_. There may be better algorithms for some cases [3]_ [4]_ [5]_.

    The algorithms of Johnson, and Gupta and Suzumura, are enhanced by some
    well-known preprocessing techniques.  When `G` is directed, we restrict our
    attention to strongly connected components of `G`, generate all simple cycles
    containing a certain node, remove that node, and further decompose the
    remainder into strongly connected components.  When `G` is undirected, we
    restrict our attention to biconnected components, generate all simple cycles
    containing a particular edge, remove that edge, and further decompose the
    remainder into biconnected components.

    Note that multigraphs are supported by this function -- and in undirected
    multigraphs, a pair of parallel edges is considered a cycle of length 2.
    Likewise, self-loops are considered to be cycles of length 1.  We define
    cycles as sequences of nodes; so the presence of loops and parallel edges
    does not change the number of simple cycles in a graph.

    Parameters
    ----------
    G : NetworkX Graph
       A networkx graph. Undirected, directed, and multigraphs are all supported.

    length_bound : int or None, optional (default=None)
       If `length_bound` is an int, generate all simple cycles of `G` with length at
       most `length_bound`.  Otherwise, generate all simple cycles of `G`.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)])
    >>> sorted(nx.simple_cycles(G))
    [[0], [0, 1, 2], [0, 2], [1, 2], [2]]

    To filter the cycles so that they don't include certain nodes or edges,
    copy your graph and eliminate those nodes or edges before calling.
    For example, to exclude self-loops from the above example:

    >>> H = G.copy()
    >>> H.remove_edges_from(nx.selfloop_edges(G))
    >>> sorted(nx.simple_cycles(H))
    [[0, 1, 2], [0, 2], [1, 2]]

    Notes
    -----
    When `length_bound` is None, the time complexity is $O((n+e)(c+1))$ for $n$
    nodes, $e$ edges and $c$ simple circuits.  Otherwise, when ``length_bound > 1``,
    the time complexity is $O((c+n)(k-1)d^k)$ where $d$ is the average degree of
    the nodes of `G` and $k$ = `length_bound`.

    Raises
    ------
    ValueError
        when ``length_bound < 0``.

    References
    ----------
    .. [1] Finding all the elementary circuits of a directed graph.
       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.
       https://doi.org/10.1137/0204007
    .. [2] Finding All Bounded-Length Simple Cycles in a Directed Graph
       A. Gupta and T. Suzumura https://arxiv.org/abs/2105.10094
    .. [3] Enumerating the cycles of a digraph: a new preprocessing strategy.
       G. Loizou and P. Thanish, Information Sciences, v. 27, 163-182, 1982.
    .. [4] A search strategy for the elementary cycles of a directed graph.
       J.L. Szwarcfiter and P.E. Lauer, BIT NUMERICAL MATHEMATICS,
       v. 16, no. 2, 192-204, 1976.
    .. [5] Optimal Listing of Cycles and st-Paths in Undirected Graphs
        R. Ferreira and R. Grossi and A. Marino and N. Pisanti and R. Rizzi and
        G. Sacomoto https://arxiv.org/abs/1205.2766

    See Also
    --------
    cycle_basis
    chordless_cycles
    """

    if length_bound is not None:
        if length_bound == 0:
            return
        elif length_bound < 0:
            raise ValueError("length bound must be non-negative")

    directed = G.is_directed()
    yield from ([v] for v, Gv in G.adj.items() if v in Gv)

    if length_bound is not None and length_bound == 1:
        return

    if G.is_multigraph() and not directed:
        visited = set()
        for u, Gu in G.adj.items():
            multiplicity = ((v, len(Guv)) for v, Guv in Gu.items() if v in visited)
            yield from ([u, v] for v, m in multiplicity if m > 1)
            visited.add(u)

    # explicitly filter out loops; implicitly filter out parallel edges
    if directed:
        G = nx.DiGraph((u, v) for u, Gu in G.adj.items() for v in Gu if v != u)
    else:
        G = nx.Graph((u, v) for u, Gu in G.adj.items() for v in Gu if v != u)

    # this case is not strictly necessary but improves performance
    if length_bound is not None and length_bound == 2:
        if directed:
            visited = set()
            for u, Gu in G.adj.items():
                yield from (
                    [v, u] for v in visited.intersection(Gu) if G.has_edge(v, u)
                )
                visited.add(u)
        return

    if directed:
        yield from _directed_cycle_search(G, length_bound)
    else:
        yield from _undirected_cycle_search(G, length_bound)


def _directed_cycle_search(G, length_bound):
    """A dispatch function for `simple_cycles` for directed graphs.

    We generate all cycles of G through binary partition.

        1. Pick a node v in G which belongs to at least one cycle
            a. Generate all cycles of G which contain the node v.
            b. Recursively generate all cycles of G \\ v.

    This is accomplished through the following:

        1. Compute the strongly connected components SCC of G.
        2. Select and remove a biconnected component C from BCC.  Select a
           non-tree edge (u, v) of a depth-first search of G[C].
        3. For each simple cycle P containing v in G[C], yield P.
        4. Add the biconnected components of G[C \\ v] to BCC.

    If the parameter length_bound is not None, then step 3 will be limited to
    simple cycles of length at most length_bound.

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph

    length_bound : int or None
       If length_bound is an int, generate all simple cycles of G with length at most length_bound.
       Otherwise, generate all simple cycles of G.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.
    """

    scc = nx.strongly_connected_components
    components = [c for c in scc(G) if len(c) >= 2]
    while components:
        c = components.pop()
        Gc = G.subgraph(c)
        v = next(iter(c))
        if length_bound is None:
            yield from _johnson_cycle_search(Gc, [v])
        else:
            yield from _bounded_cycle_search(Gc, [v], length_bound)
        # delete v after searching G, to make sure we can find v
        G.remove_node(v)
        components.extend(c for c in scc(Gc) if len(c) >= 2)


def _undirected_cycle_search(G, length_bound):
    """A dispatch function for `simple_cycles` for undirected graphs.

    We generate all cycles of G through binary partition.

        1. Pick an edge (u, v) in G which belongs to at least one cycle
            a. Generate all cycles of G which contain the edge (u, v)
            b. Recursively generate all cycles of G \\ (u, v)

    This is accomplished through the following:

        1. Compute the biconnected components BCC of G.
        2. Select and remove a biconnected component C from BCC.  Select a
           non-tree edge (u, v) of a depth-first search of G[C].
        3. For each (v -> u) path P remaining in G[C] \\ (u, v), yield P.
        4. Add the biconnected components of G[C] \\ (u, v) to BCC.

    If the parameter length_bound is not None, then step 3 will be limited to simple paths
    of length at most length_bound.

    Parameters
    ----------
    G : NetworkX Graph
       An undirected graph

    length_bound : int or None
       If length_bound is an int, generate all simple cycles of G with length at most length_bound.
       Otherwise, generate all simple cycles of G.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.
    """

    bcc = nx.biconnected_components
    components = [c for c in bcc(G) if len(c) >= 3]
    while components:
        c = components.pop()
        Gc = G.subgraph(c)
        uv = list(next(iter(Gc.edges)))
        G.remove_edge(*uv)
        # delete (u, v) before searching G, to avoid fake 3-cycles [u, v, u]
        if length_bound is None:
            yield from _johnson_cycle_search(Gc, uv)
        else:
            yield from _bounded_cycle_search(Gc, uv, length_bound)
        components.extend(c for c in bcc(Gc) if len(c) >= 3)


class _NeighborhoodCache(dict):
    """Very lightweight graph wrapper which caches neighborhoods as list.

    This dict subclass uses the __missing__ functionality to query graphs for
    their neighborhoods, and store the result as a list.  This is used to avoid
    the performance penalty incurred by subgraph views.
    """

    def __init__(self, G):
        self.G = G

    def __missing__(self, v):
        Gv = self[v] = list(self.G[v])
        return Gv


def _johnson_cycle_search(G, path):
    """The main loop of the cycle-enumeration algorithm of Johnson.

    Parameters
    ----------
    G : NetworkX Graph or DiGraph
       A graph

    path : list
       A cycle prefix.  All cycles generated will begin with this prefix.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    References
    ----------
        .. [1] Finding all the elementary circuits of a directed graph.
       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.
       https://doi.org/10.1137/0204007

    """

    G = _NeighborhoodCache(G)
    blocked = set(path)
    B = defaultdict(set)  # graph portions that yield no elementary circuit
    start = path[0]
    stack = [iter(G[path[-1]])]
    closed = [False]
    while stack:
        nbrs = stack[-1]
        for w in nbrs:
            if w == start:
                yield path[:]
                closed[-1] = True
            elif w not in blocked:
                path.append(w)
                closed.append(False)
                stack.append(iter(G[w]))
                blocked.add(w)
                break
        else:  # no more nbrs
            stack.pop()
            v = path.pop()
            if closed.pop():
                if closed:
                    closed[-1] = True
                unblock_stack = {v}
                while unblock_stack:
                    u = unblock_stack.pop()
                    if u in blocked:
                        blocked.remove(u)
                        unblock_stack.update(B[u])
                        B[u].clear()
            else:
                for w in G[v]:
                    B[w].add(v)


def _bounded_cycle_search(G, path, length_bound):
    """The main loop of the cycle-enumeration algorithm of Gupta and Suzumura.

    Parameters
    ----------
    G : NetworkX Graph or DiGraph
       A graph

    path : list
       A cycle prefix.  All cycles generated will begin with this prefix.

    length_bound: int
        A length bound.  All cycles generated will have length at most length_bound.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    References
    ----------
    .. [1] Finding All Bounded-Length Simple Cycles in a Directed Graph
       A. Gupta and T. Suzumura https://arxiv.org/abs/2105.10094

    """
    G = _NeighborhoodCache(G)
    lock = {v: 0 for v in path}
    B = defaultdict(set)
    start = path[0]
    stack = [iter(G[path[-1]])]
    blen = [length_bound]
    while stack:
        nbrs = stack[-1]
        for w in nbrs:
            if w == start:
                yield path[:]
                blen[-1] = 1
            elif len(path) < lock.get(w, length_bound):
                path.append(w)
                blen.append(length_bound)
                lock[w] = len(path)
                stack.append(iter(G[w]))
                break
        else:
            stack.pop()
            v = path.pop()
            bl = blen.pop()
            if blen:
                blen[-1] = min(blen[-1], bl)
            if bl < length_bound:
                relax_stack = [(bl, v)]
                while relax_stack:
                    bl, u = relax_stack.pop()
                    if lock.get(u, length_bound) < length_bound - bl + 1:
                        lock[u] = length_bound - bl + 1
                        relax_stack.extend((bl + 1, w) for w in B[u].difference(path))
            else:
                for w in G[v]:
                    B[w].add(v)


@nx._dispatchable
def chordless_cycles(G, length_bound=None):
    """Find simple chordless cycles of a graph.

    A `simple cycle` is a closed path where no node appears twice.  In a simple
    cycle, a `chord` is an additional edge between two nodes in the cycle.  A
    `chordless cycle` is a simple cycle without chords.  Said differently, a
    chordless cycle is a cycle C in a graph G where the number of edges in the
    induced graph G[C] is equal to the length of `C`.

    Note that some care must be taken in the case that G is not a simple graph
    nor a simple digraph.  Some authors limit the definition of chordless cycles
    to have a prescribed minimum length; we do not.

        1. We interpret self-loops to be chordless cycles, except in multigraphs
           with multiple loops in parallel.  Likewise, in a chordless cycle of
           length greater than 1, there can be no nodes with self-loops.

        2. We interpret directed two-cycles to be chordless cycles, except in
           multi-digraphs when any edge in a two-cycle has a parallel copy.

        3. We interpret parallel pairs of undirected edges as two-cycles, except
           when a third (or more) parallel edge exists between the two nodes.

        4. Generalizing the above, edges with parallel clones may not occur in
           chordless cycles.

    In a directed graph, two chordless cycles are distinct if they are not
    cyclic permutations of each other.  In an undirected graph, two chordless
    cycles are distinct if they are not cyclic permutations of each other nor of
    the other's reversal.

    Optionally, the cycles are bounded in length.

    We use an algorithm strongly inspired by that of Dias et al [1]_.  It has
    been modified in the following ways:

        1. Recursion is avoided, per Python's limitations

        2. The labeling function is not necessary, because the starting paths
            are chosen (and deleted from the host graph) to prevent multiple
            occurrences of the same path

        3. The search is optionally bounded at a specified length

        4. Support for directed graphs is provided by extending cycles along
            forward edges, and blocking nodes along forward and reverse edges

        5. Support for multigraphs is provided by omitting digons from the set
            of forward edges

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph

    length_bound : int or None, optional (default=None)
       If length_bound is an int, generate all simple cycles of G with length at
       most length_bound.  Otherwise, generate all simple cycles of G.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    Examples
    --------
    >>> sorted(list(nx.chordless_cycles(nx.complete_graph(4))))
    [[1, 0, 2], [1, 0, 3], [2, 0, 3], [2, 1, 3]]

    Notes
    -----
    When length_bound is None, and the graph is simple, the time complexity is
    $O((n+e)(c+1))$ for $n$ nodes, $e$ edges and $c$ chordless cycles.

    Raises
    ------
    ValueError
        when length_bound < 0.

    References
    ----------
    .. [1] Efficient enumeration of chordless cycles
       E. Dias and D. Castonguay and H. Longo and W.A.R. Jradi
       https://arxiv.org/abs/1309.1051

    See Also
    --------
    simple_cycles
    """

    if length_bound is not None:
        if length_bound == 0:
            return
        elif length_bound < 0:
            raise ValueError("length bound must be non-negative")

    directed = G.is_directed()
    multigraph = G.is_multigraph()

    if multigraph:
        yield from ([v] for v, Gv in G.adj.items() if len(Gv.get(v, ())) == 1)
    else:
        yield from ([v] for v, Gv in G.adj.items() if v in Gv)

    if length_bound is not None and length_bound == 1:
        return

    # Nodes with loops cannot belong to longer cycles.  Let's delete them here.
    # also, we implicitly reduce the multiplicity of edges down to 1 in the case
    # of multiedges.
    if directed:
        F = nx.DiGraph((u, v) for u, Gu in G.adj.items() if u not in Gu for v in Gu)
        B = F.to_undirected(as_view=False)
    else:
        F = nx.Graph((u, v) for u, Gu in G.adj.items() if u not in Gu for v in Gu)
        B = None

    # If we're given a multigraph, we have a few cases to consider with parallel
    # edges.
    #
    # 1. If we have 2 or more edges in parallel between the nodes (u, v), we
    #    must not construct longer cycles along (u, v).
    # 2. If G is not directed, then a pair of parallel edges between (u, v) is a
    #    chordless cycle unless there exists a third (or more) parallel edge.
    # 3. If G is directed, then parallel edges do not form cycles, but do
    #    preclude back-edges from forming cycles (handled in the next section),
    #    Thus, if an edge (u, v) is duplicated and the reverse (v, u) is also
    #    present, then we remove both from F.
    #
    # In directed graphs, we need to consider both directions that edges can
    # take, so iterate over all edges (u, v) and possibly (v, u).  In undirected
    # graphs, we need to be a little careful to only consider every edge once,
    # so we use a "visited" set to emulate node-order comparisons.

    if multigraph:
        if not directed:
            B = F.copy()
            visited = set()
        for u, Gu in G.adj.items():
            if directed:
                multiplicity = ((v, len(Guv)) for v, Guv in Gu.items())
                for v, m in multiplicity:
                    if m > 1:
                        F.remove_edges_from(((u, v), (v, u)))
            else:
                multiplicity = ((v, len(Guv)) for v, Guv in Gu.items() if v in visited)
                for v, m in multiplicity:
                    if m == 2:
                        yield [u, v]
                    if m > 1:
                        F.remove_edge(u, v)
                visited.add(u)

    # If we're given a directed graphs, we need to think about digons.  If we
    # have two edges (u, v) and (v, u), then that's a two-cycle.  If either edge
    # was duplicated above, then we removed both from F.  So, any digons we find
    # here are chordless.  After finding digons, we remove their edges from F
    # to avoid traversing them in the search for chordless cycles.
    if directed:
        for u, Fu in F.adj.items():
            digons = [[u, v] for v in Fu if F.has_edge(v, u)]
            yield from digons
            F.remove_edges_from(digons)
            F.remove_edges_from(e[::-1] for e in digons)

    if length_bound is not None and length_bound == 2:
        return

    # Now, we prepare to search for cycles.  We have removed all cycles of
    # lengths 1 and 2, so F is a simple graph or simple digraph.  We repeatedly
    # separate digraphs into their strongly connected components, and undirected
    # graphs into their biconnected components.  For each component, we pick a
    # node v, search for chordless cycles based at each "stem" (u, v, w), and
    # then remove v from that component before separating the graph again.
    if directed:
        separate = nx.strongly_connected_components

        # Directed stems look like (u -> v -> w), so we use the product of
        # predecessors of v with successors of v.
        def stems(C, v):
            for u, w in product(C.pred[v], C.succ[v]):
                if not G.has_edge(u, w):  # omit stems with acyclic chords
                    yield [u, v, w], F.has_edge(w, u)

    else:
        separate = nx.biconnected_components

        # Undirected stems look like (u ~ v ~ w), but we must not also search
        # (w ~ v ~ u), so we use combinations of v's neighbors of length 2.
        def stems(C, v):
            yield from (([u, v, w], F.has_edge(w, u)) for u, w in combinations(C[v], 2))

    components = [c for c in separate(F) if len(c) > 2]
    while components:
        c = components.pop()
        v = next(iter(c))
        Fc = F.subgraph(c)
        Fcc = Bcc = None
        for S, is_triangle in stems(Fc, v):
            if is_triangle:
                yield S
            else:
                if Fcc is None:
                    Fcc = _NeighborhoodCache(Fc)
                    Bcc = Fcc if B is None else _NeighborhoodCache(B.subgraph(c))
                yield from _chordless_cycle_search(Fcc, Bcc, S, length_bound)

        components.extend(c for c in separate(F.subgraph(c - {v})) if len(c) > 2)


def _chordless_cycle_search(F, B, path, length_bound):
    """The main loop for chordless cycle enumeration.

    This algorithm is strongly inspired by that of Dias et al [1]_.  It has been
    modified in the following ways:

        1. Recursion is avoided, per Python's limitations

        2. The labeling function is not necessary, because the starting paths
            are chosen (and deleted from the host graph) to prevent multiple
            occurrences of the same path

        3. The search is optionally bounded at a specified length

        4. Support for directed graphs is provided by extending cycles along
            forward edges, and blocking nodes along forward and reverse edges

        5. Support for multigraphs is provided by omitting digons from the set
            of forward edges

    Parameters
    ----------
    F : _NeighborhoodCache
       A graph of forward edges to follow in constructing cycles

    B : _NeighborhoodCache
       A graph of blocking edges to prevent the production of chordless cycles

    path : list
       A cycle prefix.  All cycles generated will begin with this prefix.

    length_bound : int
       A length bound.  All cycles generated will have length at most length_bound.


    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    References
    ----------
    .. [1] Efficient enumeration of chordless cycles
       E. Dias and D. Castonguay and H. Longo and W.A.R. Jradi
       https://arxiv.org/abs/1309.1051

    """
    blocked = defaultdict(int)
    target = path[0]
    blocked[path[1]] = 1
    for w in path[1:]:
        for v in B[w]:
            blocked[v] += 1

    stack = [iter(F[path[2]])]
    while stack:
        nbrs = stack[-1]
        for w in nbrs:
            if blocked[w] == 1 and (length_bound is None or len(path) < length_bound):
                Fw = F[w]
                if target in Fw:
                    yield path + [w]
                else:
                    Bw = B[w]
                    if target in Bw:
                        continue
                    for v in Bw:
                        blocked[v] += 1
                    path.append(w)
                    stack.append(iter(Fw))
                    break
        else:
            stack.pop()
            for v in B[path.pop()]:
                blocked[v] -= 1


@not_implemented_for("undirected")
@nx._dispatchable(mutates_input=True)
def recursive_simple_cycles(G):
    """Find simple cycles (elementary circuits) of a directed graph.

    A `simple cycle`, or `elementary circuit`, is a closed path where
    no node appears twice. Two elementary circuits are distinct if they
    are not cyclic permutations of each other.

    This version uses a recursive algorithm to build a list of cycles.
    You should probably use the iterator version called simple_cycles().
    Warning: This recursive version uses lots of RAM!
    It appears in NetworkX for pedagogical value.

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph

    Returns
    -------
    A list of cycles, where each cycle is represented by a list of nodes
    along the cycle.

    Example:

    >>> edges = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    >>> G = nx.DiGraph(edges)
    >>> nx.recursive_simple_cycles(G)
    [[0], [2], [0, 1, 2], [0, 2], [1, 2]]

    Notes
    -----
    The implementation follows pp. 79-80 in [1]_.

    The time complexity is $O((n+e)(c+1))$ for $n$ nodes, $e$ edges and $c$
    elementary circuits.

    References
    ----------
    .. [1] Finding all the elementary circuits of a directed graph.
       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.
       https://doi.org/10.1137/0204007

    See Also
    --------
    simple_cycles, cycle_basis
    """

    # Jon Olav Vik, 2010-08-09
    def _unblock(thisnode):
        """Recursively unblock and remove nodes from B[thisnode]."""
        if blocked[thisnode]:
            blocked[thisnode] = False
            while B[thisnode]:
                _unblock(B[thisnode].pop())

    def circuit(thisnode, startnode, component):
        closed = False  # set to True if elementary path is closed
        path.append(thisnode)
        blocked[thisnode] = True
        for nextnode in component[thisnode]:  # direct successors of thisnode
            if nextnode == startnode:
                result.append(path[:])
                closed = True
            elif not blocked[nextnode]:
                if circuit(nextnode, startnode, component):
                    closed = True
        if closed:
            _unblock(thisnode)
        else:
            for nextnode in component[thisnode]:
                if thisnode not in B[nextnode]:  # TODO: use set for speedup?
                    B[nextnode].append(thisnode)
        path.pop()  # remove thisnode from path
        return closed

    path = []  # stack of nodes in current path
    blocked = defaultdict(bool)  # vertex: blocked from search?
    B = defaultdict(list)  # graph portions that yield no elementary circuit
    result = []  # list to accumulate the circuits found

    # Johnson's algorithm exclude self cycle edges like (v, v)
    # To be backward compatible, we record those cycles in advance
    # and then remove from subG
    for v in G:
        if G.has_edge(v, v):
            result.append([v])
            G.remove_edge(v, v)

    # Johnson's algorithm requires some ordering of the nodes.
    # They might not be sortable so we assign an arbitrary ordering.
    ordering = dict(zip(G, range(len(G))))
    for s in ordering:
        # Build the subgraph induced by s and following nodes in the ordering
        subgraph = G.subgraph(node for node in G if ordering[node] >= ordering[s])
        # Find the strongly connected component in the subgraph
        # that contains the least node according to the ordering
        strongcomp = nx.strongly_connected_components(subgraph)
        mincomp = min(strongcomp, key=lambda ns: min(ordering[n] for n in ns))
        component = G.subgraph(mincomp)
        if len(component) > 1:
            # smallest node in the component according to the ordering
            startnode = min(component, key=ordering.__getitem__)
            for node in component:
                blocked[node] = False
                B[node][:] = []
            dummy = circuit(startnode, startnode, component)
    return result


@nx._dispatchable
def find_cycle(G, source=None, orientation=None):
    """Returns a cycle found via depth-first traversal.

    The cycle is a list of edges indicating the cyclic path.
    Orientation of directed edges is controlled by `orientation`.

    Parameters
    ----------
    G : graph
        A directed/undirected graph/multigraph.

    source : node, list of nodes
        The node from which the traversal begins. If None, then a source
        is chosen arbitrarily and repeatedly until all edges from each node in
        the graph are searched.

    orientation : None | 'original' | 'reverse' | 'ignore' (default: None)
        For directed graphs and directed multigraphs, edge traversals need not
        respect the original orientation of the edges.
        When set to 'reverse' every edge is traversed in the reverse direction.
        When set to 'ignore', every edge is treated as undirected.
        When set to 'original', every edge is treated as directed.
        In all three cases, the yielded edge tuples add a last entry to
        indicate the direction in which that edge was traversed.
        If orientation is None, the yielded edge has no direction indicated.
        The direction is respected, but not reported.

    Returns
    -------
    edges : directed edges
        A list of directed edges indicating the path taken for the loop.
        If no cycle is found, then an exception is raised.
        For graphs, an edge is of the form `(u, v)` where `u` and `v`
        are the tail and head of the edge as determined by the traversal.
        For multigraphs, an edge is of the form `(u, v, key)`, where `key` is
        the key of the edge. When the graph is directed, then `u` and `v`
        are always in the order of the actual directed edge.
        If orientation is not None then the edge tuple is extended to include
        the direction of traversal ('forward' or 'reverse') on that edge.

    Raises
    ------
    NetworkXNoCycle
        If no cycle was found.

    Examples
    --------
    In this example, we construct a DAG and find, in the first call, that there
    are no directed cycles, and so an exception is raised. In the second call,
    we ignore edge orientations and find that there is an undirected cycle.
    Note that the second call finds a directed cycle while effectively
    traversing an undirected graph, and so, we found an "undirected cycle".
    This means that this DAG structure does not form a directed tree (which
    is also known as a polytree).

    >>> G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    >>> nx.find_cycle(G, orientation="original")
    Traceback (most recent call last):
        ...
    networkx.exception.NetworkXNoCycle: No cycle found.
    >>> list(nx.find_cycle(G, orientation="ignore"))
    [(0, 1, 'forward'), (1, 2, 'forward'), (0, 2, 'reverse')]

    See Also
    --------
    simple_cycles
    """
    if not G.is_directed() or orientation in (None, "original"):

        def tailhead(edge):
            return edge[:2]

    elif orientation == "reverse":

        def tailhead(edge):
            return edge[1], edge[0]

    elif orientation == "ignore":

        def tailhead(edge):
            if edge[-1] == "reverse":
                return edge[1], edge[0]
            return edge[:2]

    explored = set()
    cycle = []
    final_node = None
    for start_node in G.nbunch_iter(source):
        if start_node in explored:
            # No loop is possible.
            continue

        edges = []
        # All nodes seen in this iteration of edge_dfs
        seen = {start_node}
        # Nodes in active path.
        active_nodes = {start_node}
        previous_head = None

        for edge in nx.edge_dfs(G, start_node, orientation):
            # Determine if this edge is a continuation of the active path.
            tail, head = tailhead(edge)
            if head in explored:
                # Then we've already explored it. No loop is possible.
                continue
            if previous_head is not None and tail != previous_head:
                # This edge results from backtracking.
                # Pop until we get a node whose head equals the current tail.
                # So for example, we might have:
                #  (0, 1), (1, 2), (2, 3), (1, 4)
                # which must become:
                #  (0, 1), (1, 4)
                while True:
                    try:
                        popped_edge = edges.pop()
                    except IndexError:
                        edges = []
                        active_nodes = {tail}
                        break
                    else:
                        popped_head = tailhead(popped_edge)[1]
                        active_nodes.remove(popped_head)

                    if edges:
                        last_head = tailhead(edges[-1])[1]
                        if tail == last_head:
                            break
            edges.append(edge)

            if head in active_nodes:
                # We have a loop!
                cycle.extend(edges)
                final_node = head
                break
            else:
                seen.add(head)
                active_nodes.add(head)
                previous_head = head

        if cycle:
            break
        else:
            explored.update(seen)

    else:
        assert len(cycle) == 0
        raise nx.exception.NetworkXNoCycle("No cycle found.")

    # We now have a list of edges which ends on a cycle.
    # So we need to remove from the beginning edges that are not relevant.

    for i, edge in enumerate(cycle):
        tail, head = tailhead(edge)
        if tail == final_node:
            break

    return cycle[i:]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def minimum_cycle_basis(G, weight=None):
    """Returns a minimum weight cycle basis for G

    Minimum weight means a cycle basis for which the total weight
    (length for unweighted graphs) of all the cycles is minimum.

    Parameters
    ----------
    G : NetworkX Graph
    weight: string
        name of the edge attribute to use for edge weights

    Returns
    -------
    A list of cycle lists.  Each cycle list is a list of nodes
    which forms a cycle (loop) in G. Note that the nodes are not
    necessarily returned in a order by which they appear in the cycle

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_cycle(G, [0, 1, 2, 3])
    >>> nx.add_cycle(G, [0, 3, 4, 5])
    >>> nx.minimum_cycle_basis(G)
    [[5, 4, 3, 0], [3, 2, 1, 0]]

    References:
        [1] Kavitha, Telikepalli, et al. "An O(m^2n) Algorithm for
        Minimum Cycle Basis of Graphs."
        http://link.springer.com/article/10.1007/s00453-007-9064-z
        [2] de Pina, J. 1995. Applications of shortest path methods.
        Ph.D. thesis, University of Amsterdam, Netherlands

    See Also
    --------
    simple_cycles, cycle_basis
    """
    # We first split the graph in connected subgraphs
    return sum(
        (_min_cycle_basis(G.subgraph(c), weight) for c in nx.connected_components(G)),
        [],
    )


def _min_cycle_basis(G, weight):
    cb = []
    # We  extract the edges not in a spanning tree. We do not really need a
    # *minimum* spanning tree. That is why we call the next function with
    # weight=None. Depending on implementation, it may be faster as well
    tree_edges = list(nx.minimum_spanning_edges(G, weight=None, data=False))
    chords = G.edges - tree_edges - {(v, u) for u, v in tree_edges}

    # We maintain a set of vectors orthogonal to sofar found cycles
    set_orth = [{edge} for edge in chords]
    while set_orth:
        base = set_orth.pop()
        # kth cycle is "parallel" to kth vector in set_orth
        cycle_edges = _min_cycle(G, base, weight)
        cb.append([v for u, v in cycle_edges])

        # now update set_orth so that k+1,k+2... th elements are
        # orthogonal to the newly found cycle, as per [p. 336, 1]
        set_orth = [
            (
                {e for e in orth if e not in base if e[::-1] not in base}
                | {e for e in base if e not in orth if e[::-1] not in orth}
            )
            if sum((e in orth or e[::-1] in orth) for e in cycle_edges) % 2
            else orth
            for orth in set_orth
        ]
    return cb


def _min_cycle(G, orth, weight):
    """
    Computes the minimum weight cycle in G,
    orthogonal to the vector orth as per [p. 338, 1]
    Use (u, 1) to indicate the lifted copy of u (denoted u' in paper).
    """
    Gi = nx.Graph()

    # Add 2 copies of each edge in G to Gi.
    # If edge is in orth, add cross edge; otherwise in-plane edge
    for u, v, wt in G.edges(data=weight, default=1):
        if (u, v) in orth or (v, u) in orth:
            Gi.add_edges_from([(u, (v, 1)), ((u, 1), v)], Gi_weight=wt)
        else:
            Gi.add_edges_from([(u, v), ((u, 1), (v, 1))], Gi_weight=wt)

    # find the shortest length in Gi between n and (n, 1) for each n
    # Note: Use "Gi_weight" for name of weight attribute
    spl = nx.shortest_path_length
    lift = {n: spl(Gi, source=n, target=(n, 1), weight="Gi_weight") for n in G}

    # Now compute that short path in Gi, which translates to a cycle in G
    start = min(lift, key=lift.get)
    end = (start, 1)
    min_path_i = nx.shortest_path(Gi, source=start, target=end, weight="Gi_weight")

    # Now we obtain the actual path, re-map nodes in Gi to those in G
    min_path = [n if n in G else n[0] for n in min_path_i]

    # Now remove the edges that occur two times
    # two passes: flag which edges get kept, then build it
    edgelist = list(pairwise(min_path))
    edgeset = set()
    for e in edgelist:
        if e in edgeset:
            edgeset.remove(e)
        elif e[::-1] in edgeset:
            edgeset.remove(e[::-1])
        else:
            edgeset.add(e)

    min_edgelist = []
    for e in edgelist:
        if e in edgeset:
            min_edgelist.append(e)
            edgeset.remove(e)
        elif e[::-1] in edgeset:
            min_edgelist.append(e[::-1])
            edgeset.remove(e[::-1])

    return min_edgelist


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def girth(G):
    """Returns the girth of the graph.

    The girth of a graph is the length of its shortest cycle, or infinity if
    the graph is acyclic. The algorithm follows the description given on the
    Wikipedia page [1]_, and runs in time O(mn) on a graph with m edges and n
    nodes.

    Parameters
    ----------
    G : NetworkX Graph

    Returns
    -------
    int or math.inf

    Examples
    --------
    All examples below (except P_5) can easily be checked using Wikipedia,
    which has a page for each of these famous graphs.

    >>> nx.girth(nx.chvatal_graph())
    4
    >>> nx.girth(nx.tutte_graph())
    4
    >>> nx.girth(nx.petersen_graph())
    5
    >>> nx.girth(nx.heawood_graph())
    6
    >>> nx.girth(nx.pappus_graph())
    6
    >>> nx.girth(nx.path_graph(5))
    inf

    References
    ----------
    .. [1] `Wikipedia: Girth <https://en.wikipedia.org/wiki/Girth_(graph_theory)>`_

    """
    girth = depth_limit = inf
    tree_edge = nx.algorithms.traversal.breadth_first_search.TREE_EDGE
    level_edge = nx.algorithms.traversal.breadth_first_search.LEVEL_EDGE
    for n in G:
        # run a BFS from source n, keeping track of distances; since we want
        # the shortest cycle, no need to explore beyond the current minimum length
        depth = {n: 0}
        for u, v, label in nx.bfs_labeled_edges(G, n):
            du = depth[u]
            if du > depth_limit:
                break
            if label is tree_edge:
                depth[v] = du + 1
            else:
                # if (u, v) is a level edge, the length is du + du + 1 (odd)
                # otherwise, it's a forward edge; length is du + (du + 1) + 1 (even)
                delta = label is level_edge
                length = du + du + 2 - delta
                if length < girth:
                    girth = length
                    depth_limit = du - delta

    return girth
<|endoftext|>"""Functions for computing the Voronoi cells of a graph."""

import networkx as nx
from networkx.utils import groups

__all__ = ["voronoi_cells"]


@nx._dispatchable(edge_attrs="weight")
def voronoi_cells(G, center_nodes, weight="weight"):
    """Returns the Voronoi cells centered at `center_nodes` with respect
    to the shortest-path distance metric.

    If $C$ is a set of nodes in the graph and $c$ is an element of $C$,
    the *Voronoi cell* centered at a node $c$ is the set of all nodes
    $v$ that are closer to $c$ than to any other center node in $C$ with
    respect to the shortest-path distance metric. [1]_

    For directed graphs, this will compute the "outward" Voronoi cells,
    as defined in [1]_, in which distance is measured from the center
    nodes to the target node. For the "inward" Voronoi cells, use the
    :meth:`DiGraph.reverse` method to reverse the orientation of the
    edges before invoking this function on the directed graph.

    Parameters
    ----------
    G : NetworkX graph

    center_nodes : set
        A nonempty set of nodes in the graph `G` that represent the
        center of the Voronoi cells.

    weight : string or function
        The edge attribute (or an arbitrary function) representing the
        weight of an edge. This keyword argument is as described in the
        documentation for :func:`~networkx.multi_source_dijkstra_path`,
        for example.

    Returns
    -------
    dictionary
        A mapping from center node to set of all nodes in the graph
        closer to that center node than to any other center node. The
        keys of the dictionary are the element of `center_nodes`, and
        the values of the dictionary form a partition of the nodes of
        `G`.

    Examples
    --------
    To get only the partition of the graph induced by the Voronoi cells,
    take the collection of all values in the returned dictionary::

        >>> G = nx.path_graph(6)
        >>> center_nodes = {0, 3}
        >>> cells = nx.voronoi_cells(G, center_nodes)
        >>> partition = set(map(frozenset, cells.values()))
        >>> sorted(map(sorted, partition))
        [[0, 1], [2, 3, 4, 5]]

    Raises
    ------
    ValueError
        If `center_nodes` is empty.

    References
    ----------
    .. [1] Erwig, Martin. (2000),"The graph Voronoi diagram with applications."
        *Networks*, 36: 156--163.
        https://doi.org/10.1002/1097-0037(200010)36:3<156::AID-NET2>3.0.CO;2-L

    """
    # Determine the shortest paths from any one of the center nodes to
    # every node in the graph.
    #
    # This raises `ValueError` if `center_nodes` is an empty set.
    paths = nx.multi_source_dijkstra_path(G, center_nodes, weight=weight)
    # Determine the center node from which the shortest path originates.
    nearest = {v: p[0] for v, p in paths.items()}
    # Get the mapping from center node to all nodes closer to it than to
    # any other center node.
    cells = groups(nearest)
    # We collect all unreachable nodes under a special key, if there are any.
    unreachable = set(G) - set(nearest)
    if unreachable:
        cells["unreachable"] = unreachable
    return cells
<|endoftext|>"""
Eulerian circuits and graphs.
"""

from itertools import combinations

import networkx as nx

from ..utils import arbitrary_element, not_implemented_for

__all__ = [
    "is_eulerian",
    "eulerian_circuit",
    "eulerize",
    "is_semieulerian",
    "has_eulerian_path",
    "eulerian_path",
]


@nx._dispatchable
def is_eulerian(G):
    """Returns True if and only if `G` is Eulerian.

    A graph is *Eulerian* if it has an Eulerian circuit. An *Eulerian
    circuit* is a closed walk that includes each edge of a graph exactly
    once.

    Graphs with isolated vertices (i.e. vertices with zero degree) are not
    considered to have Eulerian circuits. Therefore, if the graph is not
    connected (or not strongly connected, for directed graphs), this function
    returns False.

    Parameters
    ----------
    G : NetworkX graph
       A graph, either directed or undirected.

    Examples
    --------
    >>> nx.is_eulerian(nx.DiGraph({0: [3], 1: [2], 2: [3], 3: [0, 1]}))
    True
    >>> nx.is_eulerian(nx.complete_graph(5))
    True
    >>> nx.is_eulerian(nx.petersen_graph())
    False

    If you prefer to allow graphs with isolated vertices to have Eulerian circuits,
    you can first remove such vertices and then call `is_eulerian` as below example shows.

    >>> G = nx.Graph([(0, 1), (1, 2), (0, 2)])
    >>> G.add_node(3)
    >>> nx.is_eulerian(G)
    False

    >>> G.remove_nodes_from(list(nx.isolates(G)))
    >>> nx.is_eulerian(G)
    True


    """
    if G.is_directed():
        # Every node must have equal in degree and out degree and the
        # graph must be strongly connected
        return all(
            G.in_degree(n) == G.out_degree(n) for n in G
        ) and nx.is_strongly_connected(G)
    # An undirected Eulerian graph has no vertices of odd degree and
    # must be connected.
    return all(d % 2 == 0 for v, d in G.degree()) and nx.is_connected(G)


@nx._dispatchable
def is_semieulerian(G):
    """Return True iff `G` is semi-Eulerian.

    G is semi-Eulerian if it has an Eulerian path but no Eulerian circuit.

    See Also
    --------
    has_eulerian_path
    is_eulerian
    """
    return has_eulerian_path(G) and not is_eulerian(G)


def _find_path_start(G):
    """Return a suitable starting vertex for an Eulerian path.

    If no path exists, return None.
    """
    if not has_eulerian_path(G):
        return None

    if is_eulerian(G):
        return arbitrary_element(G)

    if G.is_directed():
        v1, v2 = (v for v in G if G.in_degree(v) != G.out_degree(v))
        # Determines which is the 'start' node (as opposed to the 'end')
        if G.out_degree(v1) > G.in_degree(v1):
            return v1
        else:
            return v2

    else:
        # In an undirected graph randomly choose one of the possibilities
        start = [v for v in G if G.degree(v) % 2 != 0][0]
        return start


def _simplegraph_eulerian_circuit(G, source):
    if G.is_directed():
        degree = G.out_degree
        edges = G.out_edges
    else:
        degree = G.degree
        edges = G.edges
    vertex_stack = [source]
    last_vertex = None
    while vertex_stack:
        current_vertex = vertex_stack[-1]
        if degree(current_vertex) == 0:
            if last_vertex is not None:
                yield (last_vertex, current_vertex)
            last_vertex = current_vertex
            vertex_stack.pop()
        else:
            _, next_vertex = arbitrary_element(edges(current_vertex))
            vertex_stack.append(next_vertex)
            G.remove_edge(current_vertex, next_vertex)


def _multigraph_eulerian_circuit(G, source):
    if G.is_directed():
        degree = G.out_degree
        edges = G.out_edges
    else:
        degree = G.degree
        edges = G.edges
    vertex_stack = [(source, None)]
    last_vertex = None
    last_key = None
    while vertex_stack:
        current_vertex, current_key = vertex_stack[-1]
        if degree(current_vertex) == 0:
            if last_vertex is not None:
                yield (last_vertex, current_vertex, last_key)
            last_vertex, last_key = current_vertex, current_key
            vertex_stack.pop()
        else:
            triple = arbitrary_element(edges(current_vertex, keys=True))
            _, next_vertex, next_key = triple
            vertex_stack.append((next_vertex, next_key))
            G.remove_edge(current_vertex, next_vertex, next_key)


@nx._dispatchable
def eulerian_circuit(G, source=None, keys=False):
    """Returns an iterator over the edges of an Eulerian circuit in `G`.

    An *Eulerian circuit* is a closed walk that includes each edge of a
    graph exactly once.

    Parameters
    ----------
    G : NetworkX graph
       A graph, either directed or undirected.

    source : node, optional
       Starting node for circuit.

    keys : bool
       If False, edges generated by this function will be of the form
       ``(u, v)``. Otherwise, edges will be of the form ``(u, v, k)``.
       This option is ignored unless `G` is a multigraph.

    Returns
    -------
    edges : iterator
       An iterator over edges in the Eulerian circuit.

    Raises
    ------
    NetworkXError
       If the graph is not Eulerian.

    See Also
    --------
    is_eulerian

    Notes
    -----
    This is a linear time implementation of an algorithm adapted from [1]_.

    For general information about Euler tours, see [2]_.

    References
    ----------
    .. [1] J. Edmonds, E. L. Johnson.
       Matching, Euler tours and the Chinese postman.
       Mathematical programming, Volume 5, Issue 1 (1973), 111-114.
    .. [2] https://en.wikipedia.org/wiki/Eulerian_path

    Examples
    --------
    To get an Eulerian circuit in an undirected graph::

        >>> G = nx.complete_graph(3)
        >>> list(nx.eulerian_circuit(G))
        [(0, 2), (2, 1), (1, 0)]
        >>> list(nx.eulerian_circuit(G, source=1))
        [(1, 2), (2, 0), (0, 1)]

    To get the sequence of vertices in an Eulerian circuit::

        >>> [u for u, v in nx.eulerian_circuit(G)]
        [0, 2, 1]

    """
    if not is_eulerian(G):
        raise nx.NetworkXError("G is not Eulerian.")
    if G.is_directed():
        G = G.reverse()
    else:
        G = G.copy()
    if source is None:
        source = arbitrary_element(G)
    if G.is_multigraph():
        for u, v, k in _multigraph_eulerian_circuit(G, source):
            if keys:
                yield u, v, k
            else:
                yield u, v
    else:
        yield from _simplegraph_eulerian_circuit(G, source)


@nx._dispatchable
def has_eulerian_path(G, source=None):
    """Return True iff `G` has an Eulerian path.

    An Eulerian path is a path in a graph which uses each edge of a graph
    exactly once. If `source` is specified, then this function checks
    whether an Eulerian path that starts at node `source` exists.

    A directed graph has an Eulerian path iff:
        - at most one vertex has out_degree - in_degree = 1,
        - at most one vertex has in_degree - out_degree = 1,
        - every other vertex has equal in_degree and out_degree,
        - and all of its vertices belong to a single connected
          component of the underlying undirected graph.

    If `source` is not None, an Eulerian path starting at `source` exists if no
    other node has out_degree - in_degree = 1. This is equivalent to either
    there exists an Eulerian circuit or `source` has out_degree - in_degree = 1
    and the conditions above hold.

    An undirected graph has an Eulerian path iff:
        - exactly zero or two vertices have odd degree,
        - and all of its vertices belong to a single connected component.

    If `source` is not None, an Eulerian path starting at `source` exists if
    either there exists an Eulerian circuit or `source` has an odd degree and the
    conditions above hold.

    Graphs with isolated vertices (i.e. vertices with zero degree) are not considered
    to have an Eulerian path. Therefore, if the graph is not connected (or not strongly
    connected, for directed graphs), this function returns False.

    Parameters
    ----------
    G : NetworkX Graph
        The graph to find an euler path in.

    source : node, optional
        Starting node for path.

    Returns
    -------
    Bool : True if G has an Eulerian path.

    Examples
    --------
    If you prefer to allow graphs with isolated vertices to have Eulerian path,
    you can first remove such vertices and then call `has_eulerian_path` as below example shows.

    >>> G = nx.Graph([(0, 1), (1, 2), (0, 2)])
    >>> G.add_node(3)
    >>> nx.has_eulerian_path(G)
    False

    >>> G.remove_nodes_from(list(nx.isolates(G)))
    >>> nx.has_eulerian_path(G)
    True

    See Also
    --------
    is_eulerian
    eulerian_path
    """
    if nx.is_eulerian(G):
        return True

    if G.is_directed():
        ins = G.in_degree
        outs = G.out_degree
        # Since we know it is not eulerian, outs - ins must be 1 for source
        if source is not None and outs[source] - ins[source] != 1:
            return False

        unbalanced_ins = 0
        unbalanced_outs = 0
        for v in G:
            if ins[v] - outs[v] == 1:
                unbalanced_ins += 1
            elif outs[v] - ins[v] == 1:
                unbalanced_outs += 1
            elif ins[v] != outs[v]:
                return False

        return (
            unbalanced_ins <= 1 and unbalanced_outs <= 1 and nx.is_weakly_connected(G)
        )
    else:
        # We know it is not eulerian, so degree of source must be odd.
        if source is not None and G.degree[source] % 2 != 1:
            return False

        # Sum is 2 since we know it is not eulerian (which implies sum is 0)
        return sum(d % 2 == 1 for v, d in G.degree()) == 2 and nx.is_connected(G)


@nx._dispatchable
def eulerian_path(G, source=None, keys=False):
    """Return an iterator over the edges of an Eulerian path in `G`.

    Parameters
    ----------
    G : NetworkX Graph
        The graph in which to look for an eulerian path.
    source : node or None (default: None)
        The node at which to start the search. None means search over all
        starting nodes.
    keys : Bool (default: False)
        Indicates whether to yield edge 3-tuples (u, v, edge_key).
        The default yields edge 2-tuples

    Yields
    ------
    Edge tuples along the eulerian path.

    Warning: If `source` provided is not the start node of an Euler path
    will raise error even if an Euler Path exists.
    """
    if not has_eulerian_path(G, source):
        raise nx.NetworkXError("Graph has no Eulerian paths.")
    if G.is_directed():
        G = G.reverse()
        if source is None or nx.is_eulerian(G) is False:
            source = _find_path_start(G)
        if G.is_multigraph():
            for u, v, k in _multigraph_eulerian_circuit(G, source):
                if keys:
                    yield u, v, k
                else:
                    yield u, v
        else:
            yield from _simplegraph_eulerian_circuit(G, source)
    else:
        G = G.copy()
        if source is None:
            source = _find_path_start(G)
        if G.is_multigraph():
            if keys:
                yield from reversed(
                    [(v, u, k) for u, v, k in _multigraph_eulerian_circuit(G, source)]
                )
            else:
                yield from reversed(
                    [(v, u) for u, v, k in _multigraph_eulerian_circuit(G, source)]
                )
        else:
            yield from reversed(
                [(v, u) for u, v in _simplegraph_eulerian_circuit(G, source)]
            )


@not_implemented_for("directed")
@nx._dispatchable(returns_graph=True)
def eulerize(G):
    """Transforms a graph into an Eulerian graph.

    If `G` is Eulerian the result is `G` as a MultiGraph, otherwise the result is a smallest
    (in terms of the number of edges) multigraph whose underlying simple graph is `G`.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph

    Returns
    -------
    G : NetworkX multigraph

    Raises
    ------
    NetworkXError
       If the graph is not connected.

    See Also
    --------
    is_eulerian
    eulerian_circuit

    References
    ----------
    .. [1] J. Edmonds, E. L. Johnson.
       Matching, Euler tours and the Chinese postman.
       Mathematical programming, Volume 5, Issue 1 (1973), 111-114.
    .. [2] https://en.wikipedia.org/wiki/Eulerian_path
    .. [3] http://web.math.princeton.edu/math_alive/5/Notes1.pdf

    Examples
    --------
        >>> G = nx.complete_graph(10)
        >>> H = nx.eulerize(G)
        >>> nx.is_eulerian(H)
        True

    """
    if G.order() == 0:
        raise nx.NetworkXPointlessConcept("Cannot Eulerize null graph")
    if not nx.is_connected(G):
        raise nx.NetworkXError("G is not connected")
    odd_degree_nodes = [n for n, d in G.degree() if d % 2 == 1]
    G = nx.MultiGraph(G)
    if len(odd_degree_nodes) == 0:
        return G

    # get all shortest paths between vertices of odd degree
    odd_deg_pairs_paths = [
        (m, {n: nx.shortest_path(G, source=m, target=n)})
        for m, n in combinations(odd_degree_nodes, 2)
    ]

    # use the number of vertices in a graph + 1 as an upper bound on
    # the maximum length of a path in G
    upper_bound_on_max_path_length = len(G) + 1

    # use "len(G) + 1 - len(P)",
    # where P is a shortest path between vertices n and m,
    # as edge-weights in a new graph
    # store the paths in the graph for easy indexing later
    Gp = nx.Graph()
    for n, Ps in odd_deg_pairs_paths:
        for m, P in Ps.items():
            if n != m:
                Gp.add_edge(
                    m, n, weight=upper_bound_on_max_path_length - len(P), path=P
                )

    # find the minimum weight matching of edges in the weighted graph
    best_matching = nx.Graph(list(nx.max_weight_matching(Gp)))

    # duplicate each edge along each path in the set of paths in Gp
    for m, n in best_matching.edges():
        path = Gp[m][n]["path"]
        G.add_edges_from(nx.utils.pairwise(path))
    return G
<|endoftext|>"""Test sequences for graphiness."""

import heapq

import networkx as nx

__all__ = [
    "is_graphical",
    "is_multigraphical",
    "is_pseudographical",
    "is_digraphical",
    "is_valid_degree_sequence_erdos_gallai",
    "is_valid_degree_sequence_havel_hakimi",
]


@nx._dispatchable(graphs=None)
def is_graphical(sequence, method="eg"):
    """Returns True if sequence is a valid degree sequence.

    A degree sequence is valid if some graph can realize it.

    Parameters
    ----------
    sequence : list or iterable container
        A sequence of integer node degrees

    method : "eg" | "hh"  (default: 'eg')
        The method used to validate the degree sequence.
        "eg" corresponds to the Erdős-Gallai algorithm
        [EG1960]_, [choudum1986]_, and
        "hh" to the Havel-Hakimi algorithm
        [havel1955]_, [hakimi1962]_, [CL1996]_.

    Returns
    -------
    valid : bool
        True if the sequence is a valid degree sequence and False if not.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> sequence = (d for n, d in G.degree())
    >>> nx.is_graphical(sequence)
    True

    To test a non-graphical sequence:
    >>> sequence_list = [d for n, d in G.degree()]
    >>> sequence_list[-1] += 1
    >>> nx.is_graphical(sequence_list)
    False

    References
    ----------
    .. [EG1960] Erdős and Gallai, Mat. Lapok 11 264, 1960.
    .. [choudum1986] S.A. Choudum. "A simple proof of the Erdős-Gallai theorem on
       graph sequences." Bulletin of the Australian Mathematical Society, 33,
       pp 67-70, 1986. https://doi.org/10.1017/S0004972700002872
    .. [havel1955] Havel, V. "A Remark on the Existence of Finite Graphs"
       Casopis Pest. Mat. 80, 477-480, 1955.
    .. [hakimi1962] Hakimi, S. "On the Realizability of a Set of Integers as
       Degrees of the Vertices of a Graph." SIAM J. Appl. Math. 10, 496-506, 1962.
    .. [CL1996] G. Chartrand and L. Lesniak, "Graphs and Digraphs",
       Chapman and Hall/CRC, 1996.
    """
    if method == "eg":
        valid = is_valid_degree_sequence_erdos_gallai(list(sequence))
    elif method == "hh":
        valid = is_valid_degree_sequence_havel_hakimi(list(sequence))
    else:
        msg = "`method` must be 'eg' or 'hh'"
        raise nx.NetworkXException(msg)
    return valid


def _basic_graphical_tests(deg_sequence):
    # Sort and perform some simple tests on the sequence
    deg_sequence = nx.utils.make_list_of_ints(deg_sequence)
    p = len(deg_sequence)
    num_degs = [0] * p
    dmax, dmin, dsum, n = 0, p, 0, 0
    for d in deg_sequence:
        # Reject if degree is negative or larger than the sequence length
        if d < 0 or d >= p:
            raise nx.NetworkXUnfeasible
        # Process only the non-zero integers
        elif d > 0:
            dmax, dmin, dsum, n = max(dmax, d), min(dmin, d), dsum + d, n + 1
            num_degs[d] += 1
    # Reject sequence if it has odd sum or is oversaturated
    if dsum % 2 or dsum > n * (n - 1):
        raise nx.NetworkXUnfeasible
    return dmax, dmin, dsum, n, num_degs


@nx._dispatchable(graphs=None)
def is_valid_degree_sequence_havel_hakimi(deg_sequence):
    r"""Returns True if deg_sequence can be realized by a simple graph.

    The validation proceeds using the Havel-Hakimi theorem
    [havel1955]_, [hakimi1962]_, [CL1996]_.
    Worst-case run time is $O(s)$ where $s$ is the sum of the sequence.

    Parameters
    ----------
    deg_sequence : list
        A list of integers where each element specifies the degree of a node
        in a graph.

    Returns
    -------
    valid : bool
        True if deg_sequence is graphical and False if not.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])
    >>> sequence = (d for _, d in G.degree())
    >>> nx.is_valid_degree_sequence_havel_hakimi(sequence)
    True

    To test a non-valid sequence:
    >>> sequence_list = [d for _, d in G.degree()]
    >>> sequence_list[-1] += 1
    >>> nx.is_valid_degree_sequence_havel_hakimi(sequence_list)
    False

    Notes
    -----
    The ZZ condition says that for the sequence d if

    .. math::
        |d| >= \frac{(\max(d) + \min(d) + 1)^2}{4*\min(d)}

    then d is graphical.  This was shown in Theorem 6 in [1]_.

    References
    ----------
    .. [1] I.E. Zverovich and V.E. Zverovich. "Contributions to the theory
       of graphic sequences", Discrete Mathematics, 105, pp. 292-303 (1992).
    .. [havel1955] Havel, V. "A Remark on the Existence of Finite Graphs"
       Casopis Pest. Mat. 80, 477-480, 1955.
    .. [hakimi1962] Hakimi, S. "On the Realizability of a Set of Integers as
       Degrees of the Vertices of a Graph." SIAM J. Appl. Math. 10, 496-506, 1962.
    .. [CL1996] G. Chartrand and L. Lesniak, "Graphs and Digraphs",
       Chapman and Hall/CRC, 1996.
    """
    try:
        dmax, dmin, dsum, n, num_degs = _basic_graphical_tests(deg_sequence)
    except nx.NetworkXUnfeasible:
        return False
    # Accept if sequence has no non-zero degrees or passes the ZZ condition
    if n == 0 or 4 * dmin * n >= (dmax + dmin + 1) * (dmax + dmin + 1):
        return True

    modstubs = [0] * (dmax + 1)
    # Successively reduce degree sequence by removing the maximum degree
    while n > 0:
        # Retrieve the maximum degree in the sequence
        while num_degs[dmax] == 0:
            dmax -= 1
        # If there are not enough stubs to connect to, then the sequence is
        # not graphical
        if dmax > n - 1:
            return False

        # Remove largest stub in list
        num_degs[dmax], n = num_degs[dmax] - 1, n - 1
        # Reduce the next dmax largest stubs
        mslen = 0
        k = dmax
        for i in range(dmax):
            while num_degs[k] == 0:
                k -= 1
            num_degs[k], n = num_degs[k] - 1, n - 1
            if k > 1:
                modstubs[mslen] = k - 1
                mslen += 1
        # Add back to the list any non-zero stubs that were removed
        for i in range(mslen):
            stub = modstubs[i]
            num_degs[stub], n = num_degs[stub] + 1, n + 1
    return True


@nx._dispatchable(graphs=None)
def is_valid_degree_sequence_erdos_gallai(deg_sequence):
    r"""Returns True if deg_sequence can be realized by a simple graph.

    The validation is done using the Erdős-Gallai theorem [EG1960]_.

    Parameters
    ----------
    deg_sequence : list
        A list of integers

    Returns
    -------
    valid : bool
        True if deg_sequence is graphical and False if not.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])
    >>> sequence = (d for _, d in G.degree())
    >>> nx.is_valid_degree_sequence_erdos_gallai(sequence)
    True

    To test a non-valid sequence:
    >>> sequence_list = [d for _, d in G.degree()]
    >>> sequence_list[-1] += 1
    >>> nx.is_valid_degree_sequence_erdos_gallai(sequence_list)
    False

    Notes
    -----

    This implementation uses an equivalent form of the Erdős-Gallai criterion.
    Worst-case run time is $O(n)$ where $n$ is the length of the sequence.

    Specifically, a sequence d is graphical if and only if the
    sum of the sequence is even and for all strong indices k in the sequence,

     .. math::

       \sum_{i=1}^{k} d_i \leq k(k-1) + \sum_{j=k+1}^{n} \min(d_i,k)
             = k(n-1) - ( k \sum_{j=0}^{k-1} n_j - \sum_{j=0}^{k-1} j n_j )

    A strong index k is any index where d_k >= k and the value n_j is the
    number of occurrences of j in d.  The maximal strong index is called the
    Durfee index.

    This particular rearrangement comes from the proof of Theorem 3 in [2]_.

    The ZZ condition says that for the sequence d if

    .. math::
        |d| >= \frac{(\max(d) + \min(d) + 1)^2}{4*\min(d)}

    then d is graphical.  This was shown in Theorem 6 in [2]_.

    References
    ----------
    .. [1] A. Tripathi and S. Vijay. "A note on a theorem of Erdős & Gallai",
       Discrete Mathematics, 265, pp. 417-420 (2003).
    .. [2] I.E. Zverovich and V.E. Zverovich. "Contributions to the theory
       of graphic sequences", Discrete Mathematics, 105, pp. 292-303 (1992).
    .. [EG1960] Erdős and Gallai, Mat. Lapok 11 264, 1960.
    """
    try:
        dmax, dmin, dsum, n, num_degs = _basic_graphical_tests(deg_sequence)
    except nx.NetworkXUnfeasible:
        return False
    # Accept if sequence has no non-zero degrees or passes the ZZ condition
    if n == 0 or 4 * dmin * n >= (dmax + dmin + 1) * (dmax + dmin + 1):
        return True

    # Perform the EG checks using the reformulation of Zverovich and Zverovich
    k, sum_deg, sum_nj, sum_jnj = 0, 0, 0, 0
    for dk in range(dmax, dmin - 1, -1):
        if dk < k + 1:  # Check if already past Durfee index
            return True
        if num_degs[dk] > 0:
            run_size = num_degs[dk]  # Process a run of identical-valued degrees
            if dk < k + run_size:  # Check if end of run is past Durfee index
                run_size = dk - k  # Adjust back to Durfee index
            sum_deg += run_size * dk
            for v in range(run_size):
                sum_nj += num_degs[k + v]
                sum_jnj += (k + v) * num_degs[k + v]
            k += run_size
            if sum_deg > k * (n - 1) - k * sum_nj + sum_jnj:
                return False
    return True


@nx._dispatchable(graphs=None)
def is_multigraphical(sequence):
    """Returns True if some multigraph can realize the sequence.

    Parameters
    ----------
    sequence : list
        A list of integers

    Returns
    -------
    valid : bool
        True if deg_sequence is a multigraphic degree sequence and False if not.

    Examples
    --------
    >>> G = nx.MultiGraph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])
    >>> sequence = (d for _, d in G.degree())
    >>> nx.is_multigraphical(sequence)
    True

    To test a non-multigraphical sequence:
    >>> sequence_list = [d for _, d in G.degree()]
    >>> sequence_list[-1] += 1
    >>> nx.is_multigraphical(sequence_list)
    False

    Notes
    -----
    The worst-case run time is $O(n)$ where $n$ is the length of the sequence.

    References
    ----------
    .. [1] S. L. Hakimi. "On the realizability of a set of integers as
       degrees of the vertices of a linear graph", J. SIAM, 10, pp. 496-506
       (1962).
    """
    try:
        deg_sequence = nx.utils.make_list_of_ints(sequence)
    except nx.NetworkXError:
        return False
    dsum, dmax = 0, 0
    for d in deg_sequence:
        if d < 0:
            return False
        dsum, dmax = dsum + d, max(dmax, d)
    if dsum % 2 or dsum < 2 * dmax:
        return False
    return True


@nx._dispatchable(graphs=None)
def is_pseudographical(sequence):
    """Returns True if some pseudograph can realize the sequence.

    Every nonnegative integer sequence with an even sum is pseudographical
    (see [1]_).

    Parameters
    ----------
    sequence : list or iterable container
        A sequence of integer node degrees

    Returns
    -------
    valid : bool
      True if the sequence is a pseudographic degree sequence and False if not.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])
    >>> sequence = (d for _, d in G.degree())
    >>> nx.is_pseudographical(sequence)
    True

    To test a non-pseudographical sequence:
    >>> sequence_list = [d for _, d in G.degree()]
    >>> sequence_list[-1] += 1
    >>> nx.is_pseudographical(sequence_list)
    False

    Notes
    -----
    The worst-case run time is $O(n)$ where n is the length of the sequence.

    References
    ----------
    .. [1] F. Boesch and F. Harary. "Line removal algorithms for graphs
       and their degree lists", IEEE Trans. Circuits and Systems, CAS-23(12),
       pp. 778-782 (1976).
    """
    try:
        deg_sequence = nx.utils.make_list_of_ints(sequence)
    except nx.NetworkXError:
        return False
    return sum(deg_sequence) % 2 == 0 and min(deg_sequence) >= 0


@nx._dispatchable(graphs=None)
def is_digraphical(in_sequence, out_sequence):
    r"""Returns True if some directed graph can realize the in- and out-degree
    sequences.

    Parameters
    ----------
    in_sequence : list or iterable container
        A sequence of integer node in-degrees

    out_sequence : list or iterable container
        A sequence of integer node out-degrees

    Returns
    -------
    valid : bool
      True if in and out-sequences are digraphic False if not.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])
    >>> in_seq = (d for n, d in G.in_degree())
    >>> out_seq = (d for n, d in G.out_degree())
    >>> nx.is_digraphical(in_seq, out_seq)
    True

    To test a non-digraphical scenario:
    >>> in_seq_list = [d for n, d in G.in_degree()]
    >>> in_seq_list[-1] += 1
    >>> nx.is_digraphical(in_seq_list, out_seq)
    False

    Notes
    -----
    This algorithm is from Kleitman and Wang [1]_.
    The worst case runtime is $O(s \times \log n)$ where $s$ and $n$ are the
    sum and length of the sequences respectively.

    References
    ----------
    .. [1] D.J. Kleitman and D.L. Wang
       Algorithms for Constructing Graphs and Digraphs with Given Valences
       and Factors, Discrete Mathematics, 6(1), pp. 79-88 (1973)
    """
    try:
        in_deg_sequence = nx.utils.make_list_of_ints(in_sequence)
        out_deg_sequence = nx.utils.make_list_of_ints(out_sequence)
    except nx.NetworkXError:
        return False
    # Process the sequences and form two heaps to store degree pairs with
    # either zero or non-zero out degrees
    sumin, sumout, nin, nout = 0, 0, len(in_deg_sequence), len(out_deg_sequence)
    maxn = max(nin, nout)
    maxin = 0
    if maxn == 0:
        return True
    stubheap, zeroheap = [], []
    for n in range(maxn):
        in_deg, out_deg = 0, 0
        if n < nout:
            out_deg = out_deg_sequence[n]
        if n < nin:
            in_deg = in_deg_sequence[n]
        if in_deg < 0 or out_deg < 0:
            return False
        sumin, sumout, maxin = sumin + in_deg, sumout + out_deg, max(maxin, in_deg)
        if in_deg > 0:
            stubheap.append((-1 * out_deg, -1 * in_deg))
        elif out_deg > 0:
            zeroheap.append(-1 * out_deg)
    if sumin != sumout:
        return False
    heapq.heapify(stubheap)
    heapq.heapify(zeroheap)

    modstubs = [(0, 0)] * (maxin + 1)
    # Successively reduce degree sequence by removing the maximum out degree
    while stubheap:
        # Take the first value in the sequence with non-zero in degree
        (freeout, freein) = heapq.heappop(stubheap)
        freein *= -1
        if freein > len(stubheap) + len(zeroheap):
            return False

        # Attach out stubs to the nodes with the most in stubs
        mslen = 0
        for i in range(freein):
            if zeroheap and (not stubheap or stubheap[0][0] > zeroheap[0]):
                stubout = heapq.heappop(zeroheap)
                stubin = 0
            else:
                (stubout, stubin) = heapq.heappop(stubheap)
            if stubout == 0:
                return False
            # Check if target is now totally connected
            if stubout + 1 < 0 or stubin < 0:
                modstubs[mslen] = (stubout + 1, stubin)
                mslen += 1

        # Add back the nodes to the heap that still have available stubs
        for i in range(mslen):
            stub = modstubs[i]
            if stub[1] < 0:
                heapq.heappush(stubheap, stub)
            else:
                heapq.heappush(zeroheap, stub[0])
        if freeout < 0:
            heapq.heappush(zeroheap, freeout)
    return True
<|endoftext|>"""
Dominance algorithms.
"""

from functools import reduce

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["immediate_dominators", "dominance_frontiers"]


@not_implemented_for("undirected")
@nx._dispatchable
def immediate_dominators(G, start):
    """Returns the immediate dominators of all nodes of a directed graph.

    Parameters
    ----------
    G : a DiGraph or MultiDiGraph
        The graph where dominance is to be computed.

    start : node
        The start node of dominance computation.

    Returns
    -------
    idom : dict keyed by nodes
        A dict containing the immediate dominators of each node reachable from
        `start`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is undirected.

    NetworkXError
        If `start` is not in `G`.

    Notes
    -----
    Except for `start`, the immediate dominators are the parents of their
    corresponding nodes in the dominator tree.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)])
    >>> sorted(nx.immediate_dominators(G, 1).items())
    [(1, 1), (2, 1), (3, 1), (4, 3), (5, 1)]

    References
    ----------
    .. [1] Cooper, Keith D., Harvey, Timothy J. and Kennedy, Ken.
           "A simple, fast dominance algorithm." (2006).
           https://hdl.handle.net/1911/96345
    """
    if start not in G:
        raise nx.NetworkXError("start is not in G")

    idom = {start: start}

    order = list(nx.dfs_postorder_nodes(G, start))
    dfn = {u: i for i, u in enumerate(order)}
    order.pop()
    order.reverse()

    def intersect(u, v):
        while u != v:
            while dfn[u] < dfn[v]:
                u = idom[u]
            while dfn[u] > dfn[v]:
                v = idom[v]
        return u

    changed = True
    while changed:
        changed = False
        for u in order:
            new_idom = reduce(intersect, (v for v in G.pred[u] if v in idom))
            if u not in idom or idom[u] != new_idom:
                idom[u] = new_idom
                changed = True

    return idom


@nx._dispatchable
def dominance_frontiers(G, start):
    """Returns the dominance frontiers of all nodes of a directed graph.

    Parameters
    ----------
    G : a DiGraph or MultiDiGraph
        The graph where dominance is to be computed.

    start : node
        The start node of dominance computation.

    Returns
    -------
    df : dict keyed by nodes
        A dict containing the dominance frontiers of each node reachable from
        `start` as lists.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is undirected.

    NetworkXError
        If `start` is not in `G`.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)])
    >>> sorted((u, sorted(df)) for u, df in nx.dominance_frontiers(G, 1).items())
    [(1, []), (2, [5]), (3, [5]), (4, [5]), (5, [])]

    References
    ----------
    .. [1] Cooper, Keith D., Harvey, Timothy J. and Kennedy, Ken.
           "A simple, fast dominance algorithm." (2006).
           https://hdl.handle.net/1911/96345
    """
    idom = nx.immediate_dominators(G, start)

    df = {u: set() for u in idom}
    for u in idom:
        if len(G.pred[u]) >= 2:
            for v in G.pred[u]:
                if v in idom:
                    while v != idom[u]:
                        df[v].add(u)
                        v = idom[v]
    return df
<|endoftext|>"""Functions for computing measures of structural holes."""

import networkx as nx

__all__ = ["constraint", "local_constraint", "effective_size"]


@nx._dispatchable(edge_attrs="weight")
def mutual_weight(G, u, v, weight=None):
    """Returns the sum of the weights of the edge from `u` to `v` and
    the edge from `v` to `u` in `G`.

    `weight` is the edge data key that represents the edge weight. If
    the specified key is `None` or is not in the edge data for an edge,
    that edge is assumed to have weight 1.

    Pre-conditions: `u` and `v` must both be in `G`.

    """
    try:
        a_uv = G[u][v].get(weight, 1)
    except KeyError:
        a_uv = 0
    try:
        a_vu = G[v][u].get(weight, 1)
    except KeyError:
        a_vu = 0
    return a_uv + a_vu


@nx._dispatchable(edge_attrs="weight")
def normalized_mutual_weight(G, u, v, norm=sum, weight=None):
    """Returns normalized mutual weight of the edges from `u` to `v`
    with respect to the mutual weights of the neighbors of `u` in `G`.

    `norm` specifies how the normalization factor is computed. It must
    be a function that takes a single argument and returns a number.
    The argument will be an iterable of mutual weights
    of pairs ``(u, w)``, where ``w`` ranges over each (in- and
    out-)neighbor of ``u``. Commons values for `normalization` are
    ``sum`` and ``max``.

    `weight` can be ``None`` or a string, if None, all edge weights
    are considered equal. Otherwise holds the name of the edge
    attribute used as weight.

    """
    scale = norm(mutual_weight(G, u, w, weight) for w in set(nx.all_neighbors(G, u)))
    return 0 if scale == 0 else mutual_weight(G, u, v, weight) / scale


@nx._dispatchable(edge_attrs="weight")
def effective_size(G, nodes=None, weight=None):
    r"""Returns the effective size of all nodes in the graph ``G``.

    The *effective size* of a node's ego network is based on the concept
    of redundancy. A person's ego network has redundancy to the extent
    that her contacts are connected to each other as well. The
    nonredundant part of a person's relationships is the effective
    size of her ego network [1]_.  Formally, the effective size of a
    node $u$, denoted $e(u)$, is defined by

    .. math::

       e(u) = \sum_{v \in N(u) \setminus \{u\}}
       \left(1 - \sum_{w \in N(v)} p_{uw} m_{vw}\right)

    where $N(u)$ is the set of neighbors of $u$ and $p_{uw}$ is the
    normalized mutual weight of the (directed or undirected) edges
    joining $u$ and $v$, for each vertex $u$ and $v$ [1]_. And $m_{vw}$
    is the mutual weight of $v$ and $w$ divided by $v$ highest mutual
    weight with any of its neighbors. The *mutual weight* of $u$ and $v$
    is the sum of the weights of edges joining them (edge weights are
    assumed to be one if the graph is unweighted).

    For the case of unweighted and undirected graphs, Borgatti proposed
    a simplified formula to compute effective size [2]_

    .. math::

       e(u) = n - \frac{2t}{n}

    where `t` is the number of ties in the ego network (not including
    ties to ego) and `n` is the number of nodes (excluding ego).

    Parameters
    ----------
    G : NetworkX graph
        The graph containing ``v``. Directed graphs are treated like
        undirected graphs when computing neighbors of ``v``.

    nodes : container, optional
        Container of nodes in the graph ``G`` to compute the effective size.
        If None, the effective size of every node is computed.

    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    Returns
    -------
    dict
        Dictionary with nodes as keys and the effective size of the node as values.

    Notes
    -----
    Burt also defined the related concept of *efficiency* of a node's ego
    network, which is its effective size divided by the degree of that
    node [1]_. So you can easily compute efficiency:

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0, 1), (0, 2), (1, 0), (2, 1)])
    >>> esize = nx.effective_size(G)
    >>> efficiency = {n: v / G.degree(n) for n, v in esize.items()}

    See also
    --------
    constraint

    References
    ----------
    .. [1] Burt, Ronald S.
           *Structural Holes: The Social Structure of Competition.*
           Cambridge: Harvard University Press, 1995.

    .. [2] Borgatti, S.
           "Structural Holes: Unpacking Burt's Redundancy Measures"
           CONNECTIONS 20(1):35-38.
           http://www.analytictech.com/connections/v20(1)/holes.htm

    """

    def redundancy(G, u, v, weight=None):
        nmw = normalized_mutual_weight
        r = sum(
            nmw(G, u, w, weight=weight) * nmw(G, v, w, norm=max, weight=weight)
            for w in set(nx.all_neighbors(G, u))
        )
        return 1 - r

    effective_size = {}
    if nodes is None:
        nodes = G
    # Use Borgatti's simplified formula for unweighted and undirected graphs
    if not G.is_directed() and weight is None:
        for v in nodes:
            # Effective size is not defined for isolated nodes
            if len(G[v]) == 0:
                effective_size[v] = float("nan")
                continue
            E = nx.ego_graph(G, v, center=False, undirected=True)
            effective_size[v] = len(E) - (2 * E.size()) / len(E)
    else:
        for v in nodes:
            # Effective size is not defined for isolated nodes
            if len(G[v]) == 0:
                effective_size[v] = float("nan")
                continue
            effective_size[v] = sum(
                redundancy(G, v, u, weight) for u in set(nx.all_neighbors(G, v))
            )
    return effective_size


@nx._dispatchable(edge_attrs="weight")
def constraint(G, nodes=None, weight=None):
    r"""Returns the constraint on all nodes in the graph ``G``.

    The *constraint* is a measure of the extent to which a node *v* is
    invested in those nodes that are themselves invested in the
    neighbors of *v*. Formally, the *constraint on v*, denoted `c(v)`,
    is defined by

    .. math::

       c(v) = \sum_{w \in N(v) \setminus \{v\}} \ell(v, w)

    where $N(v)$ is the subset of the neighbors of `v` that are either
    predecessors or successors of `v` and $\ell(v, w)$ is the local
    constraint on `v` with respect to `w` [1]_. For the definition of local
    constraint, see :func:`local_constraint`.

    Parameters
    ----------
    G : NetworkX graph
        The graph containing ``v``. This can be either directed or undirected.

    nodes : container, optional
        Container of nodes in the graph ``G`` to compute the constraint. If
        None, the constraint of every node is computed.

    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    Returns
    -------
    dict
        Dictionary with nodes as keys and the constraint on the node as values.

    See also
    --------
    local_constraint

    References
    ----------
    .. [1] Burt, Ronald S.
           "Structural holes and good ideas".
           American Journal of Sociology (110): 349–399.

    """
    if nodes is None:
        nodes = G
    constraint = {}
    for v in nodes:
        # Constraint is not defined for isolated nodes
        if len(G[v]) == 0:
            constraint[v] = float("nan")
            continue
        constraint[v] = sum(
            local_constraint(G, v, n, weight) for n in set(nx.all_neighbors(G, v))
        )
    return constraint


@nx._dispatchable(edge_attrs="weight")
def local_constraint(G, u, v, weight=None):
    r"""Returns the local constraint on the node ``u`` with respect to
    the node ``v`` in the graph ``G``.

    Formally, the *local constraint on u with respect to v*, denoted
    $\ell(u, v)$, is defined by

    .. math::

       \ell(u, v) = \left(p_{uv} + \sum_{w \in N(v)} p_{uw} p_{wv}\right)^2,

    where $N(v)$ is the set of neighbors of $v$ and $p_{uv}$ is the
    normalized mutual weight of the (directed or undirected) edges
    joining $u$ and $v$, for each vertex $u$ and $v$ [1]_. The *mutual
    weight* of $u$ and $v$ is the sum of the weights of edges joining
    them (edge weights are assumed to be one if the graph is
    unweighted).

    Parameters
    ----------
    G : NetworkX graph
        The graph containing ``u`` and ``v``. This can be either
        directed or undirected.

    u : node
        A node in the graph ``G``.

    v : node
        A node in the graph ``G``.

    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    Returns
    -------
    float
        The constraint of the node ``v`` in the graph ``G``.

    See also
    --------
    constraint

    References
    ----------
    .. [1] Burt, Ronald S.
           "Structural holes and good ideas".
           American Journal of Sociology (110): 349–399.

    """
    nmw = normalized_mutual_weight
    direct = nmw(G, u, v, weight=weight)
    indirect = sum(
        nmw(G, u, w, weight=weight) * nmw(G, w, v, weight=weight)
        for w in set(nx.all_neighbors(G, u))
    )
    return (direct + indirect) ** 2
<|endoftext|>r"""Computation of graph non-randomness"""

import math

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["non_randomness"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def non_randomness(G, k=None, weight="weight"):
    """Compute the non-randomness of graph G.

    The first returned value nr is the sum of non-randomness values of all
    edges within the graph (where the non-randomness of an edge tends to be
    small when the two nodes linked by that edge are from two different
    communities).

    The second computed value nr_rd is a relative measure that indicates
    to what extent graph G is different from random graphs in terms
    of probability. When it is close to 0, the graph tends to be more
    likely generated by an Erdos Renyi model.

    Parameters
    ----------
    G : NetworkX graph
        Graph must be symmetric, connected, and without self-loops.

    k : int
        The number of communities in G.
        If k is not set, the function will use a default community
        detection algorithm to set it.

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight. If None, then each edge has weight 1, i.e., the graph is
        binary.

    Returns
    -------
    non-randomness : (float, float) tuple
        Non-randomness, Relative non-randomness w.r.t.
        Erdos Renyi random graphs.

    Raises
    ------
    NetworkXException
        if the input graph is not connected.
    NetworkXError
        if the input graph contains self-loops or if graph has no edges.

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> nr, nr_rd = nx.non_randomness(G, 2)
    >>> nr, nr_rd = nx.non_randomness(G, 2, "weight")

    Notes
    -----
    This computes Eq. (4.4) and (4.5) in Ref. [1]_.

    If a weight field is passed, this algorithm will use the eigenvalues
    of the weighted adjacency matrix to compute Eq. (4.4) and (4.5).

    References
    ----------
    .. [1] Xiaowei Ying and Xintao Wu,
           On Randomness Measures for Social Networks,
           SIAM International Conference on Data Mining. 2009
    """
    import numpy as np

    # corner case: graph has no edges
    if nx.is_empty(G):
        raise nx.NetworkXError("non_randomness not applicable to empty graphs")
    if not nx.is_connected(G):
        raise nx.NetworkXException("Non connected graph.")
    if len(list(nx.selfloop_edges(G))) > 0:
        raise nx.NetworkXError("Graph must not contain self-loops")

    if k is None:
        k = len(tuple(nx.community.label_propagation_communities(G)))

    # eq. 4.4
    eigenvalues = np.linalg.eigvals(nx.to_numpy_array(G, weight=weight))
    nr = float(np.real(np.sum(eigenvalues[:k])))

    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = (2 * k * m) / (n * (n - k))

    # eq. 4.5
    nr_rd = (nr - ((n - 2 * k) * p + k)) / math.sqrt(2 * k * p * (1 - p))

    return nr, nr_rd
<|endoftext|>"""
Link prediction algorithms.
"""

from math import log

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "resource_allocation_index",
    "jaccard_coefficient",
    "adamic_adar_index",
    "preferential_attachment",
    "cn_soundarajan_hopcroft",
    "ra_index_soundarajan_hopcroft",
    "within_inter_cluster",
    "common_neighbor_centrality",
]


def _apply_prediction(G, func, ebunch=None):
    """Applies the given function to each edge in the specified iterable
    of edges.

    `G` is an instance of :class:`networkx.Graph`.

    `func` is a function on two inputs, each of which is a node in the
    graph. The function can return anything, but it should return a
    value representing a prediction of the likelihood of a "link"
    joining the two nodes.

    `ebunch` is an iterable of pairs of nodes. If not specified, all
    non-edges in the graph `G` will be used.

    """
    if ebunch is None:
        ebunch = nx.non_edges(G)
    else:
        for u, v in ebunch:
            if u not in G:
                raise nx.NodeNotFound(f"Node {u} not in G.")
            if v not in G:
                raise nx.NodeNotFound(f"Node {v} not in G.")
    return ((u, v, func(u, v)) for u, v in ebunch)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def resource_allocation_index(G, ebunch=None):
    r"""Compute the resource allocation index of all node pairs in ebunch.

    Resource allocation index of `u` and `v` is defined as

    .. math::

        \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{|\Gamma(w)|}

    where $\Gamma(u)$ denotes the set of neighbors of $u$.

    Parameters
    ----------
    G : graph
        A NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Resource allocation index will be computed for each pair of
        nodes given in the iterable. The pairs must be given as
        2-tuples (u, v) where u and v are nodes in the graph. If ebunch
        is None then all nonexistent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their resource allocation index.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a `DiGraph`, a `Multigraph` or a `MultiDiGraph`.

    NodeNotFound
        If `ebunch` has a node that is not in `G`.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = nx.resource_allocation_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.75000000
    (2, 3) -> 0.75000000

    References
    ----------
    .. [1] T. Zhou, L. Lu, Y.-C. Zhang.
       Predicting missing links via local information.
       Eur. Phys. J. B 71 (2009) 623.
       https://arxiv.org/pdf/0901.0553.pdf
    """

    def predict(u, v):
        return sum(1 / G.degree(w) for w in nx.common_neighbors(G, u, v))

    return _apply_prediction(G, predict, ebunch)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def jaccard_coefficient(G, ebunch=None):
    r"""Compute the Jaccard coefficient of all node pairs in ebunch.

    Jaccard coefficient of nodes `u` and `v` is defined as

    .. math::

        \frac{|\Gamma(u) \cap \Gamma(v)|}{|\Gamma(u) \cup \Gamma(v)|}

    where $\Gamma(u)$ denotes the set of neighbors of $u$.

    Parameters
    ----------
    G : graph
        A NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Jaccard coefficient will be computed for each pair of nodes
        given in the iterable. The pairs must be given as 2-tuples
        (u, v) where u and v are nodes in the graph. If ebunch is None
        then all nonexistent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Jaccard coefficient.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a `DiGraph`, a `Multigraph` or a `MultiDiGraph`.

    NodeNotFound
        If `ebunch` has a node that is not in `G`.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = nx.jaccard_coefficient(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.60000000
    (2, 3) -> 0.60000000

    References
    ----------
    .. [1] D. Liben-Nowell, J. Kleinberg.
           The Link Prediction Problem for Social Networks (2004).
           http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    """

    def predict(u, v):
        union_size = len(set(G[u]) | set(G[v]))
        if union_size == 0:
            return 0
        return len(nx.common_neighbors(G, u, v)) / union_size

    return _apply_prediction(G, predict, ebunch)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def adamic_adar_index(G, ebunch=None):
    r"""Compute the Adamic-Adar index of all node pairs in ebunch.

    Adamic-Adar index of `u` and `v` is defined as

    .. math::

        \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\log |\Gamma(w)|}

    where $\Gamma(u)$ denotes the set of neighbors of $u$.
    This index leads to zero-division for nodes only connected via self-loops.
    It is intended to be used when no self-loops are present.

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Adamic-Adar index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        nonexistent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Adamic-Adar index.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a `DiGraph`, a `Multigraph` or a `MultiDiGraph`.

    NodeNotFound
        If `ebunch` has a node that is not in `G`.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = nx.adamic_adar_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 2.16404256
    (2, 3) -> 2.16404256

    References
    ----------
    .. [1] D. Liben-Nowell, J. Kleinberg.
           The Link Prediction Problem for Social Networks (2004).
           http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    """

    def predict(u, v):
        return sum(1 / log(G.degree(w)) for w in nx.common_neighbors(G, u, v))

    return _apply_prediction(G, predict, ebunch)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def common_neighbor_centrality(G, ebunch=None, alpha=0.8):
    r"""Return the CCPA score for each pair of nodes.

    Compute the Common Neighbor and Centrality based Parameterized Algorithm(CCPA)
    score of all node pairs in ebunch.

    CCPA score of `u` and `v` is defined as

    .. math::

        \alpha \cdot (|\Gamma (u){\cap }^{}\Gamma (v)|)+(1-\alpha )\cdot \frac{N}{{d}_{uv}}

    where $\Gamma(u)$ denotes the set of neighbors of $u$, $\Gamma(v)$ denotes the
    set of neighbors of $v$, $\alpha$ is  parameter varies between [0,1], $N$ denotes
    total number of nodes in the Graph and ${d}_{uv}$ denotes shortest distance
    between $u$ and $v$.

    This algorithm is based on two vital properties of nodes, namely the number
    of common neighbors and their centrality. Common neighbor refers to the common
    nodes between two nodes. Centrality refers to the prestige that a node enjoys
    in a network.

    .. seealso::

        :func:`common_neighbors`

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Preferential attachment score will be computed for each pair of
        nodes given in the iterable. The pairs must be given as
        2-tuples (u, v) where u and v are nodes in the graph. If ebunch
        is None then all nonexistent edges in the graph will be used.
        Default value: None.

    alpha : Parameter defined for participation of Common Neighbor
            and Centrality Algorithm share. Values for alpha should
            normally be between 0 and 1. Default value set to 0.8
            because author found better performance at 0.8 for all the
            dataset.
            Default value: 0.8


    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Common Neighbor and Centrality based
        Parameterized Algorithm(CCPA) score.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a `DiGraph`, a `Multigraph` or a `MultiDiGraph`.

    NetworkXAlgorithmError
        If self loops exist in `ebunch` or in `G` (if `ebunch` is `None`).

    NodeNotFound
        If `ebunch` has a node that is not in `G`.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = nx.common_neighbor_centrality(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p}")
    (0, 1) -> 3.4000000000000004
    (2, 3) -> 3.4000000000000004

    References
    ----------
    .. [1] Ahmad, I., Akhtar, M.U., Noor, S. et al.
           Missing Link Prediction using Common Neighbor and Centrality based Parameterized Algorithm.
           Sci Rep 10, 364 (2020).
           https://doi.org/10.1038/s41598-019-57304-y
    """

    # When alpha == 1, the CCPA score simplifies to the number of common neighbors.
    if alpha == 1:

        def predict(u, v):
            if u == v:
                raise nx.NetworkXAlgorithmError("Self loops are not supported")

            return len(nx.common_neighbors(G, u, v))

    else:
        spl = dict(nx.shortest_path_length(G))
        inf = float("inf")

        def predict(u, v):
            if u == v:
                raise nx.NetworkXAlgorithmError("Self loops are not supported")
            path_len = spl[u].get(v, inf)

            n_nbrs = len(nx.common_neighbors(G, u, v))
            return alpha * n_nbrs + (1 - alpha) * len(G) / path_len

    return _apply_prediction(G, predict, ebunch)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def preferential_attachment(G, ebunch=None):
    r"""Compute the preferential attachment score of all node pairs in ebunch.

    Preferential attachment score of `u` and `v` is defined as

    .. math::

        |\Gamma(u)| |\Gamma(v)|

    where $\Gamma(u)$ denotes the set of neighbors of $u$.

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Preferential attachment score will be computed for each pair of
        nodes given in the iterable. The pairs must be given as
        2-tuples (u, v) where u and v are nodes in the graph. If ebunch
        is None then all nonexistent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their preferential attachment score.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a `DiGraph`, a `Multigraph` or a `MultiDiGraph`.

    NodeNotFound
        If `ebunch` has a node that is not in `G`.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = nx.preferential_attachment(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p}")
    (0, 1) -> 16
    (2, 3) -> 16

    References
    ----------
    .. [1] D. Liben-Nowell, J. Kleinberg.
           The Link Prediction Problem for Social Networks (2004).
           http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    """

    def predict(u, v):
        return G.degree(u) * G.degree(v)

    return _apply_prediction(G, predict, ebunch)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(node_attrs="community")
def cn_soundarajan_hopcroft(G, ebunch=None, community="community"):
    r"""Count the number of common neighbors of all node pairs in ebunch
        using community information.

    For two nodes $u$ and $v$, this function computes the number of
    common neighbors and bonus one for each common neighbor belonging to
    the same community as $u$ and $v$. Mathematically,

    .. math::

        |\Gamma(u) \cap \Gamma(v)| + \sum_{w \in \Gamma(u) \cap \Gamma(v)} f(w)

    where $f(w)$ equals 1 if $w$ belongs to the same community as $u$
    and $v$ or 0 otherwise and $\Gamma(u)$ denotes the set of
    neighbors of $u$.

    Parameters
    ----------
    G : graph
        A NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        The score will be computed for each pair of nodes given in the
        iterable. The pairs must be given as 2-tuples (u, v) where u
        and v are nodes in the graph. If ebunch is None then all
        nonexistent edges in the graph will be used.
        Default value: None.

    community : string, optional (default = 'community')
        Nodes attribute name containing the community information.
        G[u][community] identifies which community u belongs to. Each
        node belongs to at most one community. Default value: 'community'.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their score.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a `DiGraph`, a `Multigraph` or a `MultiDiGraph`.

    NetworkXAlgorithmError
        If no community information is available for a node in `ebunch` or in `G` (if `ebunch` is `None`).

    NodeNotFound
        If `ebunch` has a node that is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> G.nodes[0]["community"] = 0
    >>> G.nodes[1]["community"] = 0
    >>> G.nodes[2]["community"] = 0
    >>> preds = nx.cn_soundarajan_hopcroft(G, [(0, 2)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p}")
    (0, 2) -> 2

    References
    ----------
    .. [1] Sucheta Soundarajan and John Hopcroft.
       Using community information to improve the precision of link
       prediction methods.
       In Proceedings of the 21st international conference companion on
       World Wide Web (WWW '12 Companion). ACM, New York, NY, USA, 607-608.
       http://doi.acm.org/10.1145/2187980.2188150
    """

    def predict(u, v):
        Cu = _community(G, u, community)
        Cv = _community(G, v, community)
        cnbors = nx.common_neighbors(G, u, v)
        neighbors = (
            sum(_community(G, w, community) == Cu for w in cnbors) if Cu == Cv else 0
        )
        return len(cnbors) + neighbors

    return _apply_prediction(G, predict, ebunch)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(node_attrs="community")
def ra_index_soundarajan_hopcroft(G, ebunch=None, community="community"):
    r"""Compute the resource allocation index of all node pairs in
    ebunch using community information.

    For two nodes $u$ and $v$, this function computes the resource
    allocation index considering only common neighbors belonging to the
    same community as $u$ and $v$. Mathematically,

    .. math::

        \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{f(w)}{|\Gamma(w)|}

    where $f(w)$ equals 1 if $w$ belongs to the same community as $u$
    and $v$ or 0 otherwise and $\Gamma(u)$ denotes the set of
    neighbors of $u$.

    Parameters
    ----------
    G : graph
        A NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        The score will be computed for each pair of nodes given in the
        iterable. The pairs must be given as 2-tuples (u, v) where u
        and v are nodes in the graph. If ebunch is None then all
        nonexistent edges in the graph will be used.
        Default value: None.

    community : string, optional (default = 'community')
        Nodes attribute name containing the community information.
        G[u][community] identifies which community u belongs to. Each
        node belongs to at most one community. Default value: 'community'.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their score.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a `DiGraph`, a `Multigraph` or a `MultiDiGraph`.

    NetworkXAlgorithmError
        If no community information is available for a node in `ebunch` or in `G` (if `ebunch` is `None`).

    NodeNotFound
        If `ebunch` has a node that is not in `G`.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    >>> G.nodes[0]["community"] = 0
    >>> G.nodes[1]["community"] = 0
    >>> G.nodes[2]["community"] = 1
    >>> G.nodes[3]["community"] = 0
    >>> preds = nx.ra_index_soundarajan_hopcroft(G, [(0, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 3) -> 0.50000000

    References
    ----------
    .. [1] Sucheta Soundarajan and John Hopcroft.
       Using community information to improve the precision of link
       prediction methods.
       In Proceedings of the 21st international conference companion on
       World Wide Web (WWW '12 Companion). ACM, New York, NY, USA, 607-608.
       http://doi.acm.org/10.1145/2187980.2188150
    """

    def predict(u, v):
        Cu = _community(G, u, community)
        Cv = _community(G, v, community)
        if Cu != Cv:
            return 0
        cnbors = nx.common_neighbors(G, u, v)
        return sum(1 / G.degree(w) for w in cnbors if _community(G, w, community) == Cu)

    return _apply_prediction(G, predict, ebunch)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(node_attrs="community")
def within_inter_cluster(G, ebunch=None, delta=0.001, community="community"):
    """Compute the ratio of within- and inter-cluster common neighbors
    of all node pairs in ebunch.

    For two nodes `u` and `v`, if a common neighbor `w` belongs to the
    same community as them, `w` is considered as within-cluster common
    neighbor of `u` and `v`. Otherwise, it is considered as
    inter-cluster common neighbor of `u` and `v`. The ratio between the
    size of the set of within- and inter-cluster common neighbors is
    defined as the WIC measure. [1]_

    Parameters
    ----------
    G : graph
        A NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        The WIC measure will be computed for each pair of nodes given in
        the iterable. The pairs must be given as 2-tuples (u, v) where
        u and v are nodes in the graph. If ebunch is None then all
        nonexistent edges in the graph will be used.
        Default value: None.

    delta : float, optional (default = 0.001)
        Value to prevent division by zero in case there is no
        inter-cluster common neighbor between two nodes. See [1]_ for
        details. Default value: 0.001.

    community : string, optional (default = 'community')
        Nodes attribute name containing the community information.
        G[u][community] identifies which community u belongs to. Each
        node belongs to at most one community. Default value: 'community'.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their WIC measure.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a `DiGraph`, a `Multigraph` or a `MultiDiGraph`.

    NetworkXAlgorithmError
        - If `delta` is less than or equal to zero.
        - If no community information is available for a node in `ebunch` or in `G` (if `ebunch` is `None`).

    NodeNotFound
        If `ebunch` has a node that is not in `G`.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)])
    >>> G.nodes[0]["community"] = 0
    >>> G.nodes[1]["community"] = 1
    >>> G.nodes[2]["community"] = 0
    >>> G.nodes[3]["community"] = 0
    >>> G.nodes[4]["community"] = 0
    >>> preds = nx.within_inter_cluster(G, [(0, 4)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 4) -> 1.99800200
    >>> preds = nx.within_inter_cluster(G, [(0, 4)], delta=0.5)
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 4) -> 1.33333333

    References
    ----------
    .. [1] Jorge Carlos Valverde-Rebaza and Alneu de Andrade Lopes.
       Link prediction in complex networks based on cluster information.
       In Proceedings of the 21st Brazilian conference on Advances in
       Artificial Intelligence (SBIA'12)
       https://doi.org/10.1007/978-3-642-34459-6_10
    """
    if delta <= 0:
        raise nx.NetworkXAlgorithmError("Delta must be greater than zero")

    def predict(u, v):
        Cu = _community(G, u, community)
        Cv = _community(G, v, community)
        if Cu != Cv:
            return 0
        cnbors = nx.common_neighbors(G, u, v)
        within = {w for w in cnbors if _community(G, w, community) == Cu}
        inter = cnbors - within
        return len(within) / (len(inter) + delta)

    return _apply_prediction(G, predict, ebunch)


def _community(G, u, community):
    """Get the community of the given node."""
    node_u = G.nodes[u]
    try:
        return node_u[community]
    except KeyError as err:
        raise nx.NetworkXAlgorithmError(
            f"No community information available for Node {u}"
        ) from err
<|endoftext|>r"""Function for computing the moral graph of a directed graph."""

import itertools

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["moral_graph"]


@not_implemented_for("undirected")
@nx._dispatchable(returns_graph=True)
def moral_graph(G):
    r"""Return the Moral Graph

    Returns the moralized graph of a given directed graph.

    Parameters
    ----------
    G : NetworkX graph
        Directed graph

    Returns
    -------
    H : NetworkX graph
        The undirected moralized graph of G

    Raises
    ------
    NetworkXNotImplemented
        If `G` is undirected.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (2, 5), (3, 4), (4, 3)])
    >>> G_moral = nx.moral_graph(G)
    >>> G_moral.edges()
    EdgeView([(1, 2), (2, 3), (2, 5), (2, 4), (3, 4)])

    Notes
    -----
    A moral graph is an undirected graph H = (V, E) generated from a
    directed Graph, where if a node has more than one parent node, edges
    between these parent nodes are inserted and all directed edges become
    undirected.

    https://en.wikipedia.org/wiki/Moral_graph

    References
    ----------
    .. [1] Wray L. Buntine. 1995. Chain graphs for learning.
           In Proceedings of the Eleventh conference on Uncertainty
           in artificial intelligence (UAI'95)
    """
    H = G.to_undirected()
    for preds in G.pred.values():
        predecessors_combinations = itertools.combinations(preds, r=2)
        H.add_edges_from(predecessors_combinations)
    return H
<|endoftext|>"""Routines to find the boundary of a set of nodes.

An edge boundary is a set of edges, each of which has exactly one
endpoint in a given set of nodes (or, in the case of directed graphs,
the set of edges whose source node is in the set).

A node boundary of a set *S* of nodes is the set of (out-)neighbors of
nodes in *S* that are outside *S*.

"""

from itertools import chain

import networkx as nx

__all__ = ["edge_boundary", "node_boundary"]


@nx._dispatchable(edge_attrs={"data": "default"}, preserve_edge_attrs="data")
def edge_boundary(G, nbunch1, nbunch2=None, data=False, keys=False, default=None):
    """Returns the edge boundary of `nbunch1`.

    The *edge boundary* of a set *S* with respect to a set *T* is the
    set of edges (*u*, *v*) such that *u* is in *S* and *v* is in *T*.
    If *T* is not specified, it is assumed to be the set of all nodes
    not in *S*.

    Parameters
    ----------
    G : NetworkX graph

    nbunch1 : iterable
        Iterable of nodes in the graph representing the set of nodes
        whose edge boundary will be returned. (This is the set *S* from
        the definition above.)

    nbunch2 : iterable
        Iterable of nodes representing the target (or "exterior") set of
        nodes. (This is the set *T* from the definition above.) If not
        specified, this is assumed to be the set of all nodes in `G`
        not in `nbunch1`.

    keys : bool
        This parameter has the same meaning as in
        :meth:`MultiGraph.edges`.

    data : bool or object
        This parameter has the same meaning as in
        :meth:`MultiGraph.edges`.

    default : object
        This parameter has the same meaning as in
        :meth:`MultiGraph.edges`.

    Returns
    -------
    iterator
        An iterator over the edges in the boundary of `nbunch1` with
        respect to `nbunch2`. If `keys`, `data`, or `default`
        are specified and `G` is a multigraph, then edges are returned
        with keys and/or data, as in :meth:`MultiGraph.edges`.

    Examples
    --------
    >>> G = nx.wheel_graph(6)

    When nbunch2=None:

    >>> list(nx.edge_boundary(G, (1, 3)))
    [(1, 0), (1, 2), (1, 5), (3, 0), (3, 2), (3, 4)]

    When nbunch2 is given:

    >>> list(nx.edge_boundary(G, (1, 3), (2, 0)))
    [(1, 0), (1, 2), (3, 0), (3, 2)]

    Notes
    -----
    Any element of `nbunch` that is not in the graph `G` will be
    ignored.

    `nbunch1` and `nbunch2` are usually meant to be disjoint, but in
    the interest of speed and generality, that is not required here.

    """
    nset1 = {n for n in nbunch1 if n in G}
    # Here we create an iterator over edges incident to nodes in the set
    # `nset1`. The `Graph.edges()` method does not provide a guarantee
    # on the orientation of the edges, so our algorithm below must
    # handle the case in which exactly one orientation, either (u, v) or
    # (v, u), appears in this iterable.
    if G.is_multigraph():
        edges = G.edges(nset1, data=data, keys=keys, default=default)
    else:
        edges = G.edges(nset1, data=data, default=default)
    # If `nbunch2` is not provided, then it is assumed to be the set
    # complement of `nbunch1`. For the sake of efficiency, this is
    # implemented by using the `not in` operator, instead of by creating
    # an additional set and using the `in` operator.
    if nbunch2 is None:
        return (e for e in edges if (e[0] in nset1) ^ (e[1] in nset1))
    nset2 = set(nbunch2)
    return (
        e
        for e in edges
        if (e[0] in nset1 and e[1] in nset2) or (e[1] in nset1 and e[0] in nset2)
    )


@nx._dispatchable
def node_boundary(G, nbunch1, nbunch2=None):
    """Returns the node boundary of `nbunch1`.

    The *node boundary* of a set *S* with respect to a set *T* is the
    set of nodes *v* in *T* such that for some *u* in *S*, there is an
    edge joining *u* to *v*. If *T* is not specified, it is assumed to
    be the set of all nodes not in *S*.

    Parameters
    ----------
    G : NetworkX graph

    nbunch1 : iterable
        Iterable of nodes in the graph representing the set of nodes
        whose node boundary will be returned. (This is the set *S* from
        the definition above.)

    nbunch2 : iterable
        Iterable of nodes representing the target (or "exterior") set of
        nodes. (This is the set *T* from the definition above.) If not
        specified, this is assumed to be the set of all nodes in `G`
        not in `nbunch1`.

    Returns
    -------
    set
        The node boundary of `nbunch1` with respect to `nbunch2`.

    Examples
    --------
    >>> G = nx.wheel_graph(6)

    When nbunch2=None:

    >>> list(nx.node_boundary(G, (3, 4)))
    [0, 2, 5]

    When nbunch2 is given:

    >>> list(nx.node_boundary(G, (3, 4), (0, 1, 5)))
    [0, 5]

    Notes
    -----
    Any element of `nbunch` that is not in the graph `G` will be
    ignored.

    `nbunch1` and `nbunch2` are usually meant to be disjoint, but in
    the interest of speed and generality, that is not required here.

    """
    nset1 = {n for n in nbunch1 if n in G}
    bdy = set(chain.from_iterable(G[v] for v in nset1)) - nset1
    # If `nbunch2` is not specified, it is assumed to be the set
    # complement of `nbunch1`.
    if nbunch2 is not None:
        bdy &= set(nbunch2)
    return bdy
<|endoftext|>"""
Algorithm to find a maximal (not maximum) independent set.

"""

import networkx as nx
from networkx.utils import not_implemented_for, py_random_state

__all__ = ["maximal_independent_set"]


@not_implemented_for("directed")
@py_random_state(2)
@nx._dispatchable
def maximal_independent_set(G, nodes=None, seed=None):
    """Returns a random maximal independent set guaranteed to contain
    a given set of nodes.

    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximal
    independent set is an independent set such that it is not possible
    to add a new node and still get an independent set.

    Parameters
    ----------
    G : NetworkX graph

    nodes : list or iterable
       Nodes that must be part of the independent set. This set of nodes
       must be independent.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    indep_nodes : list
       List of nodes that are part of a maximal independent set.

    Raises
    ------
    NetworkXUnfeasible
       If the nodes in the provided list are not part of the graph or
       do not form an independent set, an exception is raised.

    NetworkXNotImplemented
        If `G` is directed.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.maximal_independent_set(G)  # doctest: +SKIP
    [4, 0, 2]
    >>> nx.maximal_independent_set(G, [1])  # doctest: +SKIP
    [1, 3]

    Notes
    -----
    This algorithm does not solve the maximum independent set problem.

    """
    if not nodes:
        nodes = {seed.choice(list(G))}
    else:
        nodes = set(nodes)
    if not nodes.issubset(G):
        raise nx.NetworkXUnfeasible(f"{nodes} is not a subset of the nodes of G")
    neighbors = set.union(*[set(G.adj[v]) for v in nodes])
    if set.intersection(neighbors, nodes):
        raise nx.NetworkXUnfeasible(f"{nodes} is not an independent set of G")
    indep_nodes = list(nodes)
    available_nodes = set(G.nodes()).difference(neighbors.union(nodes))
    while available_nodes:
        node = seed.choice(list(available_nodes))
        indep_nodes.append(node)
        available_nodes.difference_update(list(G.adj[node]) + [node])
    return indep_nodes
<|endoftext|># See https://github.com/networkx/networkx/pull/1474
# Copyright 2011 Reya Group <http://www.reyagroup.com>
# Copyright 2011 Alex Levenson <alex@isnotinvain.com>
# Copyright 2011 Diederik van Liere <diederik.vanliere@rotman.utoronto.ca>
"""Functions for analyzing triads of a graph."""

from collections import defaultdict
from itertools import combinations, permutations

import networkx as nx
from networkx.utils import not_implemented_for, py_random_state

__all__ = [
    "triadic_census",
    "is_triad",
    "all_triplets",
    "all_triads",
    "triads_by_type",
    "triad_type",
    "random_triad",
]

#: The integer codes representing each type of triad.
#:
#: Triads that are the same up to symmetry have the same code.
TRICODES = (
    1,
    2,
    2,
    3,
    2,
    4,
    6,
    8,
    2,
    6,
    5,
    7,
    3,
    8,
    7,
    11,
    2,
    6,
    4,
    8,
    5,
    9,
    9,
    13,
    6,
    10,
    9,
    14,
    7,
    14,
    12,
    15,
    2,
    5,
    6,
    7,
    6,
    9,
    10,
    14,
    4,
    9,
    9,
    12,
    8,
    13,
    14,
    15,
    3,
    7,
    8,
    11,
    7,
    12,
    14,
    15,
    8,
    14,
    13,
    15,
    11,
    15,
    15,
    16,
)

#: The names of each type of triad. The order of the elements is
#: important: it corresponds to the tricodes given in :data:`TRICODES`.
TRIAD_NAMES = (
    "003",
    "012",
    "102",
    "021D",
    "021U",
    "021C",
    "111D",
    "111U",
    "030T",
    "030C",
    "201",
    "120D",
    "120U",
    "120C",
    "210",
    "300",
)


#: A dictionary mapping triad code to triad name.
TRICODE_TO_NAME = {i: TRIAD_NAMES[code - 1] for i, code in enumerate(TRICODES)}


def _tricode(G, v, u, w):
    """Returns the integer code of the given triad.

    This is some fancy magic that comes from Batagelj and Mrvar's paper. It
    treats each edge joining a pair of `v`, `u`, and `w` as a bit in
    the binary representation of an integer.

    """
    combos = ((v, u, 1), (u, v, 2), (v, w, 4), (w, v, 8), (u, w, 16), (w, u, 32))
    return sum(x for u, v, x in combos if v in G[u])


@not_implemented_for("undirected")
@nx._dispatchable
def triadic_census(G, nodelist=None):
    """Determines the triadic census of a directed graph.

    The triadic census is a count of how many of the 16 possible types of
    triads are present in a directed graph. If a list of nodes is passed, then
    only those triads are taken into account which have elements of nodelist in them.

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph
    nodelist : list
        List of nodes for which you want to calculate triadic census

    Returns
    -------
    census : dict
       Dictionary with triad type as keys and number of occurrences as values.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4), (4, 1), (4, 2)])
    >>> triadic_census = nx.triadic_census(G)
    >>> for key, value in triadic_census.items():
    ...     print(f"{key}: {value}")
    003: 0
    012: 0
    102: 0
    021D: 0
    021U: 0
    021C: 0
    111D: 0
    111U: 0
    030T: 2
    030C: 2
    201: 0
    120D: 0
    120U: 0
    120C: 0
    210: 0
    300: 0

    Notes
    -----
    This algorithm has complexity $O(m)$ where $m$ is the number of edges in
    the graph.

    For undirected graphs, the triadic census can be computed by first converting
    the graph into a directed graph using the ``G.to_directed()`` method.
    After this conversion, only the triad types 003, 102, 201 and 300 will be
    present in the undirected scenario.

    Raises
    ------
    ValueError
        If `nodelist` contains duplicate nodes or nodes not in `G`.
        If you want to ignore this you can preprocess with `set(nodelist) & G.nodes`

    See also
    --------
    triad_graph

    References
    ----------
    .. [1] Vladimir Batagelj and Andrej Mrvar, A subquadratic triad census
        algorithm for large sparse networks with small maximum degree,
        University of Ljubljana,
        http://vlado.fmf.uni-lj.si/pub/networks/doc/triads/triads.pdf

    """
    nodeset = set(G.nbunch_iter(nodelist))
    if nodelist is not None and len(nodelist) != len(nodeset):
        raise ValueError("nodelist includes duplicate nodes or nodes not in G")

    N = len(G)
    Nnot = N - len(nodeset)  # can signal special counting for subset of nodes

    # create an ordering of nodes with nodeset nodes first
    m = {n: i for i, n in enumerate(nodeset)}
    if Nnot:
        # add non-nodeset nodes later in the ordering
        not_nodeset = G.nodes - nodeset
        m.update((n, i + N) for i, n in enumerate(not_nodeset))

    # build all_neighbor dicts for easy counting
    # After Python 3.8 can leave off these keys(). Speedup also using G._pred
    # nbrs = {n: G._pred[n].keys() | G._succ[n].keys() for n in G}
    nbrs = {n: G.pred[n].keys() | G.succ[n].keys() for n in G}
    dbl_nbrs = {n: G.pred[n].keys() & G.succ[n].keys() for n in G}

    if Nnot:
        sgl_nbrs = {n: G.pred[n].keys() ^ G.succ[n].keys() for n in not_nodeset}
        # find number of edges not incident to nodes in nodeset
        sgl = sum(1 for n in not_nodeset for nbr in sgl_nbrs[n] if nbr not in nodeset)
        sgl_edges_outside = sgl // 2
        dbl = sum(1 for n in not_nodeset for nbr in dbl_nbrs[n] if nbr not in nodeset)
        dbl_edges_outside = dbl // 2

    # Initialize the count for each triad to be zero.
    census = {name: 0 for name in TRIAD_NAMES}
    # Main loop over nodes
    for v in nodeset:
        vnbrs = nbrs[v]
        dbl_vnbrs = dbl_nbrs[v]
        if Nnot:
            # set up counts of edges attached to v.
            sgl_unbrs_bdy = sgl_unbrs_out = dbl_unbrs_bdy = dbl_unbrs_out = 0
        for u in vnbrs:
            if m[u] <= m[v]:
                continue
            unbrs = nbrs[u]
            neighbors = (vnbrs | unbrs) - {u, v}
            # Count connected triads.
            for w in neighbors:
                if m[u] < m[w] or (m[v] < m[w] < m[u] and v not in nbrs[w]):
                    code = _tricode(G, v, u, w)
                    census[TRICODE_TO_NAME[code]] += 1

            # Use a formula for dyadic triads with edge incident to v
            if u in dbl_vnbrs:
                census["102"] += N - len(neighbors) - 2
            else:
                census["012"] += N - len(neighbors) - 2

            # Count edges attached to v. Subtract later to get triads with v isolated
            # _out are (u,unbr) for unbrs outside boundary of nodeset
            # _bdy are (u,unbr) for unbrs on boundary of nodeset (get double counted)
            if Nnot and u not in nodeset:
                sgl_unbrs = sgl_nbrs[u]
                sgl_unbrs_bdy += len(sgl_unbrs & vnbrs - nodeset)
                sgl_unbrs_out += len(sgl_unbrs - vnbrs - nodeset)
                dbl_unbrs = dbl_nbrs[u]
                dbl_unbrs_bdy += len(dbl_unbrs & vnbrs - nodeset)
                dbl_unbrs_out += len(dbl_unbrs - vnbrs - nodeset)
        # if nodeset == G.nodes, skip this b/c we will find the edge later.
        if Nnot:
            # Count edges outside nodeset not connected with v (v isolated triads)
            census["012"] += sgl_edges_outside - (sgl_unbrs_out + sgl_unbrs_bdy // 2)
            census["102"] += dbl_edges_outside - (dbl_unbrs_out + dbl_unbrs_bdy // 2)

    # calculate null triads: "003"
    # null triads = total number of possible triads - all found triads
    total_triangles = (N * (N - 1) * (N - 2)) // 6
    triangles_without_nodeset = (Nnot * (Nnot - 1) * (Nnot - 2)) // 6
    total_census = total_triangles - triangles_without_nodeset
    census["003"] = total_census - sum(census.values())

    return census


@nx._dispatchable
def is_triad(G):
    """Returns True if the graph G is a triad, else False.

    Parameters
    ----------
    G : graph
       A NetworkX Graph

    Returns
    -------
    istriad : boolean
       Whether G is a valid triad

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    >>> nx.is_triad(G)
    True
    >>> G.add_edge(0, 1)
    >>> nx.is_triad(G)
    False
    """
    if isinstance(G, nx.Graph):
        if G.order() == 3 and nx.is_directed(G):
            if not any((n, n) in G.edges() for n in G.nodes()):
                return True
    return False


@not_implemented_for("undirected")
@nx._dispatchable
def all_triplets(G):
    """Returns a generator of all possible sets of 3 nodes in a DiGraph.

    .. deprecated:: 3.3

       all_triplets is deprecated and will be removed in NetworkX version 3.5.
       Use `itertools.combinations` instead::

          all_triplets = itertools.combinations(G, 3)

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph

    Returns
    -------
    triplets : generator of 3-tuples
       Generator of tuples of 3 nodes

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    >>> list(nx.all_triplets(G))
    [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]

    """
    import warnings

    warnings.warn(
        (
            "\n\nall_triplets is deprecated and will be removed in v3.5.\n"
            "Use `itertools.combinations(G, 3)` instead."
        ),
        category=DeprecationWarning,
        stacklevel=4,
    )
    triplets = combinations(G.nodes(), 3)
    return triplets


@not_implemented_for("undirected")
@nx._dispatchable(returns_graph=True)
def all_triads(G):
    """A generator of all possible triads in G.

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph

    Returns
    -------
    all_triads : generator of DiGraphs
       Generator of triads (order-3 DiGraphs)

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4), (4, 1), (4, 2)])
    >>> for triad in nx.all_triads(G):
    ...     print(triad.edges)
    [(1, 2), (2, 3), (3, 1)]
    [(1, 2), (4, 1), (4, 2)]
    [(3, 1), (3, 4), (4, 1)]
    [(2, 3), (3, 4), (4, 2)]

    """
    triplets = combinations(G.nodes(), 3)
    for triplet in triplets:
        yield G.subgraph(triplet).copy()


@not_implemented_for("undirected")
@nx._dispatchable
def triads_by_type(G):
    """Returns a list of all triads for each triad type in a directed graph.
    There are exactly 16 different types of triads possible. Suppose 1, 2, 3 are three
    nodes, they will be classified as a particular triad type if their connections
    are as follows:

    - 003: 1, 2, 3
    - 012: 1 -> 2, 3
    - 102: 1 <-> 2, 3
    - 021D: 1 <- 2 -> 3
    - 021U: 1 -> 2 <- 3
    - 021C: 1 -> 2 -> 3
    - 111D: 1 <-> 2 <- 3
    - 111U: 1 <-> 2 -> 3
    - 030T: 1 -> 2 -> 3, 1 -> 3
    - 030C: 1 <- 2 <- 3, 1 -> 3
    - 201: 1 <-> 2 <-> 3
    - 120D: 1 <- 2 -> 3, 1 <-> 3
    - 120U: 1 -> 2 <- 3, 1 <-> 3
    - 120C: 1 -> 2 -> 3, 1 <-> 3
    - 210: 1 -> 2 <-> 3, 1 <-> 3
    - 300: 1 <-> 2 <-> 3, 1 <-> 3

    Refer to the :doc:`example gallery </auto_examples/graph/plot_triad_types>`
    for visual examples of the triad types.

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph

    Returns
    -------
    tri_by_type : dict
       Dictionary with triad types as keys and lists of triads as values.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 1), (5, 6), (5, 4), (6, 7)])
    >>> dict = nx.triads_by_type(G)
    >>> dict["120C"][0].edges()
    OutEdgeView([(1, 2), (1, 3), (2, 3), (3, 1)])
    >>> dict["012"][0].edges()
    OutEdgeView([(1, 2)])

    References
    ----------
    .. [1] Snijders, T. (2012). "Transitivity and triads." University of
        Oxford.
        https://web.archive.org/web/20170830032057/http://www.stats.ox.ac.uk/~snijders/Trans_Triads_ha.pdf
    """
    # num_triads = o * (o - 1) * (o - 2) // 6
    # if num_triads > TRIAD_LIMIT: print(WARNING)
    all_tri = all_triads(G)
    tri_by_type = defaultdict(list)
    for triad in all_tri:
        name = triad_type(triad)
        tri_by_type[name].append(triad)
    return tri_by_type


@not_implemented_for("undirected")
@nx._dispatchable
def triad_type(G):
    """Returns the sociological triad type for a triad.

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph with 3 nodes

    Returns
    -------
    triad_type : str
       A string identifying the triad type

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    >>> nx.triad_type(G)
    '030C'
    >>> G.add_edge(1, 3)
    >>> nx.triad_type(G)
    '120C'

    Notes
    -----
    There can be 6 unique edges in a triad (order-3 DiGraph) (so 2^^6=64 unique
    triads given 3 nodes). These 64 triads each display exactly 1 of 16
    topologies of triads (topologies can be permuted). These topologies are
    identified by the following notation:

    {m}{a}{n}{type} (for example: 111D, 210, 102)

    Here:

    {m}     = number of mutual ties (takes 0, 1, 2, 3); a mutual tie is (0,1)
              AND (1,0)
    {a}     = number of asymmetric ties (takes 0, 1, 2, 3); an asymmetric tie
              is (0,1) BUT NOT (1,0) or vice versa
    {n}     = number of null ties (takes 0, 1, 2, 3); a null tie is NEITHER
              (0,1) NOR (1,0)
    {type}  = a letter (takes U, D, C, T) corresponding to up, down, cyclical
              and transitive. This is only used for topologies that can have
              more than one form (eg: 021D and 021U).

    References
    ----------
    .. [1] Snijders, T. (2012). "Transitivity and triads." University of
        Oxford.
        https://web.archive.org/web/20170830032057/http://www.stats.ox.ac.uk/~snijders/Trans_Triads_ha.pdf
    """
    if not is_triad(G):
        raise nx.NetworkXAlgorithmError("G is not a triad (order-3 DiGraph)")
    num_edges = len(G.edges())
    if num_edges == 0:
        return "003"
    elif num_edges == 1:
        return "012"
    elif num_edges == 2:
        e1, e2 = G.edges()
        if set(e1) == set(e2):
            return "102"
        elif e1[0] == e2[0]:
            return "021D"
        elif e1[1] == e2[1]:
            return "021U"
        elif e1[1] == e2[0] or e2[1] == e1[0]:
            return "021C"
    elif num_edges == 3:
        for e1, e2, e3 in permutations(G.edges(), 3):
            if set(e1) == set(e2):
                if e3[0] in e1:
                    return "111U"
                # e3[1] in e1:
                return "111D"
            elif set(e1).symmetric_difference(set(e2)) == set(e3):
                if {e1[0], e2[0], e3[0]} == {e1[0], e2[0], e3[0]} == set(G.nodes()):
                    return "030C"
                # e3 == (e1[0], e2[1]) and e2 == (e1[1], e3[1]):
                return "030T"
    elif num_edges == 4:
        for e1, e2, e3, e4 in permutations(G.edges(), 4):
            if set(e1) == set(e2):
                # identify pair of symmetric edges (which necessarily exists)
                if set(e3) == set(e4):
                    return "201"
                if {e3[0]} == {e4[0]} == set(e3).intersection(set(e4)):
                    return "120D"
                if {e3[1]} == {e4[1]} == set(e3).intersection(set(e4)):
                    return "120U"
                if e3[1] == e4[0]:
                    return "120C"
    elif num_edges == 5:
        return "210"
    elif num_edges == 6:
        return "300"


@not_implemented_for("undirected")
@py_random_state(1)
@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def random_triad(G, seed=None):
    """Returns a random triad from a directed graph.

    .. deprecated:: 3.3

       random_triad is deprecated and will be removed in version 3.5.
       Use random sampling directly instead::

          G.subgraph(random.sample(list(G), 3))

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G2 : subgraph
       A randomly selected triad (order-3 NetworkX DiGraph)

    Raises
    ------
    NetworkXError
        If the input Graph has less than 3 nodes.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 1), (5, 6), (5, 4), (6, 7)])
    >>> triad = nx.random_triad(G, seed=1)
    >>> triad.edges
    OutEdgeView([(1, 2)])

    """
    import warnings

    warnings.warn(
        (
            "\n\nrandom_triad is deprecated and will be removed in NetworkX v3.5.\n"
            "Use random.sample instead, e.g.::\n\n"
            "\tG.subgraph(random.sample(list(G), 3))\n"
        ),
        category=DeprecationWarning,
        stacklevel=5,
    )
    if len(G) < 3:
        raise nx.NetworkXError(
            f"G needs at least 3 nodes to form a triad; (it has {len(G)} nodes)"
        )
    nodes = seed.sample(list(G.nodes()), 3)
    G2 = G.subgraph(nodes)
    return G2
<|endoftext|>"""Routines to calculate the broadcast time of certain graphs.

Broadcasting is an information dissemination problem in which a node in a graph,
called the originator, must distribute a message to all other nodes by placing
a series of calls along the edges of the graph. Once informed, other nodes aid
the originator in distributing the message.

The broadcasting must be completed as quickly as possible subject to the
following constraints:
- Each call requires one unit of time.
- A node can only participate in one call per unit of time.
- Each call only involves two adjacent nodes: a sender and a receiver.
"""

import networkx as nx
from networkx import NetworkXError
from networkx.utils import not_implemented_for

__all__ = [
    "tree_broadcast_center",
    "tree_broadcast_time",
]


def _get_max_broadcast_value(G, U, v, values):
    adj = sorted(set(G.neighbors(v)) & U, key=values.get, reverse=True)
    return max(values[u] + i for i, u in enumerate(adj, start=1))


def _get_broadcast_centers(G, v, values, target):
    adj = sorted(G.neighbors(v), key=values.get, reverse=True)
    j = next(i for i, u in enumerate(adj, start=1) if values[u] + i == target)
    return set([v] + adj[:j])


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def tree_broadcast_center(G):
    """Return the Broadcast Center of the tree `G`.

    The broadcast center of a graph G denotes the set of nodes having
    minimum broadcast time [1]_. This is a linear algorithm for determining
    the broadcast center of a tree with ``N`` nodes, as a by-product it also
    determines the broadcast time from the broadcast center.

    Parameters
    ----------
    G : undirected graph
        The graph should be an undirected tree

    Returns
    -------
    BC : (int, set) tuple
        minimum broadcast number of the tree, set of broadcast centers

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    References
    ----------
    .. [1] Slater, P.J., Cockayne, E.J., Hedetniemi, S.T,
       Information dissemination in trees. SIAM J.Comput. 10(4), 692–701 (1981)
    """
    # Assert that the graph G is a tree
    if not nx.is_tree(G):
        NetworkXError("Input graph is not a tree")
    # step 0
    if G.number_of_nodes() == 2:
        return 1, set(G.nodes())
    if G.number_of_nodes() == 1:
        return 0, set(G.nodes())

    # step 1
    U = {node for node, deg in G.degree if deg == 1}
    values = {n: 0 for n in U}
    T = G.copy()
    T.remove_nodes_from(U)

    # step 2
    W = {node for node, deg in T.degree if deg == 1}
    values.update((w, G.degree[w] - 1) for w in W)

    # step 3
    while T.number_of_nodes() >= 2:
        # step 4
        w = min(W, key=lambda n: values[n])
        v = next(T.neighbors(w))

        # step 5
        U.add(w)
        W.remove(w)
        T.remove_node(w)

        # step 6
        if T.degree(v) == 1:
            # update t(v)
            values.update({v: _get_max_broadcast_value(G, U, v, values)})
            W.add(v)

    # step 7
    v = nx.utils.arbitrary_element(T)
    b_T = _get_max_broadcast_value(G, U, v, values)
    return b_T, _get_broadcast_centers(G, v, values, b_T)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def tree_broadcast_time(G, node=None):
    """Return the Broadcast Time of the tree `G`.

    The minimum broadcast time of a node is defined as the minimum amount
    of time required to complete broadcasting starting from the
    originator. The broadcast time of a graph is the maximum over
    all nodes of the minimum broadcast time from that node [1]_.
    This function returns the minimum broadcast time of `node`.
    If `node` is None the broadcast time for the graph is returned.

    Parameters
    ----------
    G : undirected graph
        The graph should be an undirected tree
    node: int, optional
        index of starting node. If `None`, the algorithm returns the broadcast
        time of the tree.

    Returns
    -------
    BT : int
        Broadcast Time of a node in a tree

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    References
    ----------
    .. [1] Harutyunyan, H. A. and Li, Z.
        "A Simple Construction of Broadcast Graphs."
        In Computing and Combinatorics. COCOON 2019
        (Ed. D. Z. Du and C. Tian.) Springer, pp. 240-253, 2019.
    """
    b_T, b_C = tree_broadcast_center(G)
    if node is not None:
        return b_T + min(nx.shortest_path_length(G, node, u) for u in b_C)
    dist_from_center = dict.fromkeys(G, len(G))
    for u in b_C:
        for v, dist in nx.shortest_path_length(G, u).items():
            if dist < dist_from_center[v]:
                dist_from_center[v] = dist
    return b_T + max(dist_from_center.values())
<|endoftext|>"""
Algorithms for chordal graphs.

A graph is chordal if every cycle of length at least 4 has a chord
(an edge joining two nodes not adjacent in the cycle).
https://en.wikipedia.org/wiki/Chordal_graph
"""

import sys

import networkx as nx
from networkx.algorithms.components import connected_components
from networkx.utils import arbitrary_element, not_implemented_for

__all__ = [
    "is_chordal",
    "find_induced_nodes",
    "chordal_graph_cliques",
    "chordal_graph_treewidth",
    "NetworkXTreewidthBoundExceeded",
    "complete_to_chordal_graph",
]


class NetworkXTreewidthBoundExceeded(nx.NetworkXException):
    """Exception raised when a treewidth bound has been provided and it has
    been exceeded"""


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_chordal(G):
    """Checks whether G is a chordal graph.

    A graph is chordal if every cycle of length at least 4 has a chord
    (an edge joining two nodes not adjacent in the cycle).

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    Returns
    -------
    chordal : bool
      True if G is a chordal graph and False otherwise.

    Raises
    ------
    NetworkXNotImplemented
        The algorithm does not support DiGraph, MultiGraph and MultiDiGraph.

    Examples
    --------
    >>> e = [
    ...     (1, 2),
    ...     (1, 3),
    ...     (2, 3),
    ...     (2, 4),
    ...     (3, 4),
    ...     (3, 5),
    ...     (3, 6),
    ...     (4, 5),
    ...     (4, 6),
    ...     (5, 6),
    ... ]
    >>> G = nx.Graph(e)
    >>> nx.is_chordal(G)
    True

    Notes
    -----
    The routine tries to go through every node following maximum cardinality
    search. It returns False when it finds that the separator for any node
    is not a clique.  Based on the algorithms in [1]_.

    Self loops are ignored.

    References
    ----------
    .. [1] R. E. Tarjan and M. Yannakakis, Simple linear-time algorithms
       to test chordality of graphs, test acyclicity of hypergraphs, and
       selectively reduce acyclic hypergraphs, SIAM J. Comput., 13 (1984),
       pp. 566–579.
    """
    if len(G.nodes) <= 3:
        return True
    return len(_find_chordality_breaker(G)) == 0


@nx._dispatchable
def find_induced_nodes(G, s, t, treewidth_bound=sys.maxsize):
    """Returns the set of induced nodes in the path from s to t.

    Parameters
    ----------
    G : graph
      A chordal NetworkX graph
    s : node
        Source node to look for induced nodes
    t : node
        Destination node to look for induced nodes
    treewidth_bound: float
        Maximum treewidth acceptable for the graph H. The search
        for induced nodes will end as soon as the treewidth_bound is exceeded.

    Returns
    -------
    induced_nodes : Set of nodes
        The set of induced nodes in the path from s to t in G

