    >>> G = nx.cycle_graph(4, create_using=nx.DiGraph())
    >>> nx.add_cycle(G, [10, 11, 12])
    >>> [
    ...     len(c)
    ...     for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    ... ]
    [4, 3]

    If you only want the largest component, it's more efficient to
    use max instead of sort.

    >>> largest = max(nx.strongly_connected_components(G), key=len)

    See Also
    --------
    connected_components
    weakly_connected_components
    kosaraju_strongly_connected_components

    Notes
    -----
    Uses Tarjan's algorithm[1]_ with Nuutila's modifications[2]_.
    Nonrecursive version of algorithm.

    References
    ----------
    .. [1] Depth-first search and linear graph algorithms, R. Tarjan
       SIAM Journal of Computing 1(2):146-160, (1972).

    .. [2] On finding the strongly connected components in a directed graph.
       E. Nuutila and E. Soisalon-Soinen
       Information Processing Letters 49(1): 9-14, (1994)..

    """
    preorder = {}
    lowlink = {}
    scc_found = set()
    scc_queue = []
    i = 0  # Preorder counter
    neighbors = {v: iter(G[v]) for v in G}
    for source in G:
        if source not in scc_found:
            queue = [source]
            while queue:
                v = queue[-1]
                if v not in preorder:
                    i = i + 1
                    preorder[v] = i
                done = True
                for w in neighbors[v]:
                    if w not in preorder:
                        queue.append(w)
                        done = False
                        break
                if done:
                    lowlink[v] = preorder[v]
                    for w in G[v]:
                        if w not in scc_found:
                            if preorder[w] > preorder[v]:
                                lowlink[v] = min([lowlink[v], lowlink[w]])
                            else:
                                lowlink[v] = min([lowlink[v], preorder[w]])
                    queue.pop()
                    if lowlink[v] == preorder[v]:
                        scc = {v}
                        while scc_queue and preorder[scc_queue[-1]] > preorder[v]:
                            k = scc_queue.pop()
                            scc.add(k)
                        scc_found.update(scc)
                        yield scc
                    else:
                        scc_queue.append(v)


@not_implemented_for("undirected")
@nx._dispatchable
def kosaraju_strongly_connected_components(G, source=None):
    """Generate nodes in strongly connected components of graph.

    Parameters
    ----------
    G : NetworkX Graph
        A directed graph.

    Returns
    -------
    comp : generator of sets
        A generator of sets of nodes, one for each strongly connected
        component of G.

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    Generate a sorted list of strongly connected components, largest first.

    >>> G = nx.cycle_graph(4, create_using=nx.DiGraph())
    >>> nx.add_cycle(G, [10, 11, 12])
    >>> [
    ...     len(c)
    ...     for c in sorted(
    ...         nx.kosaraju_strongly_connected_components(G), key=len, reverse=True
    ...     )
    ... ]
    [4, 3]

    If you only want the largest component, it's more efficient to
    use max instead of sort.

    >>> largest = max(nx.kosaraju_strongly_connected_components(G), key=len)

    See Also
    --------
    strongly_connected_components

    Notes
    -----
    Uses Kosaraju's algorithm.

    """
    post = list(nx.dfs_postorder_nodes(G.reverse(copy=False), source=source))

    seen = set()
    while post:
        r = post.pop()
        if r in seen:
            continue
        c = nx.dfs_preorder_nodes(G, r)
        new = {v for v in c if v not in seen}
        seen.update(new)
        yield new


@not_implemented_for("undirected")
@nx._dispatchable
def number_strongly_connected_components(G):
    """Returns number of strongly connected components in graph.

    Parameters
    ----------
    G : NetworkX graph
       A directed graph.

    Returns
    -------
    n : integer
       Number of strongly connected components

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    >>> G = nx.DiGraph(
    ...     [(0, 1), (1, 2), (2, 0), (2, 3), (4, 5), (3, 4), (5, 6), (6, 3), (6, 7)]
    ... )
    >>> nx.number_strongly_connected_components(G)
    3

    See Also
    --------
    strongly_connected_components
    number_connected_components
    number_weakly_connected_components

    Notes
    -----
    For directed graphs only.
    """
    return sum(1 for scc in strongly_connected_components(G))


@not_implemented_for("undirected")
@nx._dispatchable
def is_strongly_connected(G):
    """Test directed graph for strong connectivity.

    A directed graph is strongly connected if and only if every vertex in
    the graph is reachable from every other vertex.

    Parameters
    ----------
    G : NetworkX Graph
       A directed graph.

    Returns
    -------
    connected : bool
      True if the graph is strongly connected, False otherwise.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 2)])
    >>> nx.is_strongly_connected(G)
    True
    >>> G.remove_edge(2, 3)
    >>> nx.is_strongly_connected(G)
    False

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    See Also
    --------
    is_weakly_connected
    is_semiconnected
    is_connected
    is_biconnected
    strongly_connected_components

    Notes
    -----
    For directed graphs only.
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            """Connectivity is undefined for the null graph."""
        )

    return len(next(strongly_connected_components(G))) == len(G)


@not_implemented_for("undirected")
@nx._dispatchable(returns_graph=True)
def condensation(G, scc=None):
    """Returns the condensation of G.

    The condensation of G is the graph with each of the strongly connected
    components contracted into a single node.

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph.

    scc:  list or generator (optional, default=None)
       Strongly connected components. If provided, the elements in
       `scc` must partition the nodes in `G`. If not provided, it will be
       calculated as scc=nx.strongly_connected_components(G).

    Returns
    -------
    C : NetworkX DiGraph
       The condensation graph C of G.  The node labels are integers
       corresponding to the index of the component in the list of
       strongly connected components of G.  C has a graph attribute named
       'mapping' with a dictionary mapping the original nodes to the
       nodes in C to which they belong.  Each node in C also has a node
       attribute 'members' with the set of original nodes in G that
       form the SCC that the node in C represents.

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    Contracting two sets of strongly connected nodes into two distinct SCC
    using the barbell graph.

    >>> G = nx.barbell_graph(4, 0)
    >>> G.remove_edge(3, 4)
    >>> G = nx.DiGraph(G)
    >>> H = nx.condensation(G)
    >>> H.nodes.data()
    NodeDataView({0: {'members': {0, 1, 2, 3}}, 1: {'members': {4, 5, 6, 7}}})
    >>> H.graph["mapping"]
    {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}

    Contracting a complete graph into one single SCC.

    >>> G = nx.complete_graph(7, create_using=nx.DiGraph)
    >>> H = nx.condensation(G)
    >>> H.nodes
    NodeView((0,))
    >>> H.nodes.data()
    NodeDataView({0: {'members': {0, 1, 2, 3, 4, 5, 6}}})

    Notes
    -----
    After contracting all strongly connected components to a single node,
    the resulting graph is a directed acyclic graph.

    """
    if scc is None:
        scc = nx.strongly_connected_components(G)
    mapping = {}
    members = {}
    C = nx.DiGraph()
    # Add mapping dict as graph attribute
    C.graph["mapping"] = mapping
    if len(G) == 0:
        return C
    for i, component in enumerate(scc):
        members[i] = component
        mapping.update((n, i) for n in component)
    number_of_components = i + 1
    C.add_nodes_from(range(number_of_components))
    C.add_edges_from(
        (mapping[u], mapping[v]) for u, v in G.edges() if mapping[u] != mapping[v]
    )
    # Add a list of members (ie original nodes) to each node (ie scc) in C.
    nx.set_node_attributes(C, members, "members")
    return C
<|endoftext|>"""Weakly connected components."""

import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = [
    "number_weakly_connected_components",
    "weakly_connected_components",
    "is_weakly_connected",
]


@not_implemented_for("undirected")
@nx._dispatchable
def weakly_connected_components(G):
    """Generate weakly connected components of G.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph

    Returns
    -------
    comp : generator of sets
        A generator of sets of nodes, one for each weakly connected
        component of G.

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    Generate a sorted list of weakly connected components, largest first.

    >>> G = nx.path_graph(4, create_using=nx.DiGraph())
    >>> nx.add_path(G, [10, 11, 12])
    >>> [
    ...     len(c)
    ...     for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    ... ]
    [4, 3]

    If you only want the largest component, it's more efficient to
    use max instead of sort:

    >>> largest_cc = max(nx.weakly_connected_components(G), key=len)

    See Also
    --------
    connected_components
    strongly_connected_components

    Notes
    -----
    For directed graphs only.

    """
    seen = set()
    n = len(G)  # must be outside the loop to avoid performance hit with graph views
    for v in G:
        if v not in seen:
            c = set(_plain_bfs(G, n, v))
            seen.update(c)
            yield c


@not_implemented_for("undirected")
@nx._dispatchable
def number_weakly_connected_components(G):
    """Returns the number of weakly connected components in G.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph.

    Returns
    -------
    n : integer
        Number of weakly connected components

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (2, 1), (3, 4)])
    >>> nx.number_weakly_connected_components(G)
    2

    See Also
    --------
    weakly_connected_components
    number_connected_components
    number_strongly_connected_components

    Notes
    -----
    For directed graphs only.

    """
    return sum(1 for wcc in weakly_connected_components(G))


@not_implemented_for("undirected")
@nx._dispatchable
def is_weakly_connected(G):
    """Test directed graph for weak connectivity.

    A directed graph is weakly connected if and only if the graph
    is connected when the direction of the edge between nodes is ignored.

    Note that if a graph is strongly connected (i.e. the graph is connected
    even when we account for directionality), it is by definition weakly
    connected as well.

    Parameters
    ----------
    G : NetworkX Graph
        A directed graph.

    Returns
    -------
    connected : bool
        True if the graph is weakly connected, False otherwise.

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (2, 1)])
    >>> G.add_node(3)
    >>> nx.is_weakly_connected(G)  # node 3 is not connected to the graph
    False
    >>> G.add_edge(2, 3)
    >>> nx.is_weakly_connected(G)
    True

    See Also
    --------
    is_strongly_connected
    is_semiconnected
    is_connected
    is_biconnected
    weakly_connected_components

    Notes
    -----
    For directed graphs only.

    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            """Connectivity is undefined for the null graph."""
        )

    return len(next(weakly_connected_components(G))) == len(G)


def _plain_bfs(G, n, source):
    """A fast BFS node generator

    The direction of the edge between nodes is ignored.

    For directed graphs only.

    """
    Gsucc = G._succ
    Gpred = G._pred
    seen = {source}
    nextlevel = [source]

    yield source
    while nextlevel:
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in Gsucc[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
                    yield w
            for w in Gpred[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
                    yield w
            if len(seen) == n:
                return
<|endoftext|>"""Attracting components."""

import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = [
    "number_attracting_components",
    "attracting_components",
    "is_attracting_component",
]


@not_implemented_for("undirected")
@nx._dispatchable
def attracting_components(G):
    """Generates the attracting components in `G`.

    An attracting component in a directed graph `G` is a strongly connected
    component with the property that a random walker on the graph will never
    leave the component, once it enters the component.

    The nodes in attracting components can also be thought of as recurrent
    nodes.  If a random walker enters the attractor containing the node, then
    the node will be visited infinitely often.

    To obtain induced subgraphs on each component use:
    ``(G.subgraph(c).copy() for c in attracting_components(G))``

    Parameters
    ----------
    G : DiGraph, MultiDiGraph
        The graph to be analyzed.

    Returns
    -------
    attractors : generator of sets
        A generator of sets of nodes, one for each attracting component of G.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is undirected.

    See Also
    --------
    number_attracting_components
    is_attracting_component

    """
    scc = list(nx.strongly_connected_components(G))
    cG = nx.condensation(G, scc)
    for n in cG:
        if cG.out_degree(n) == 0:
            yield scc[n]


@not_implemented_for("undirected")
@nx._dispatchable
def number_attracting_components(G):
    """Returns the number of attracting components in `G`.

    Parameters
    ----------
    G : DiGraph, MultiDiGraph
        The graph to be analyzed.

    Returns
    -------
    n : int
        The number of attracting components in G.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is undirected.

    See Also
    --------
    attracting_components
    is_attracting_component

    """
    return sum(1 for ac in attracting_components(G))


@not_implemented_for("undirected")
@nx._dispatchable
def is_attracting_component(G):
    """Returns True if `G` consists of a single attracting component.

    Parameters
    ----------
    G : DiGraph, MultiDiGraph
        The graph to be analyzed.

    Returns
    -------
    attracting : bool
        True if `G` has a single attracting component. Otherwise, False.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is undirected.

    See Also
    --------
    attracting_components
    number_attracting_components

    """
    ac = list(attracting_components(G))
    if len(ac) == 1:
        return len(ac[0]) == len(G)
    return False
<|endoftext|>"""Connected components."""

import networkx as nx
from networkx.utils.decorators import not_implemented_for

from ...utils import arbitrary_element

__all__ = [
    "number_connected_components",
    "connected_components",
    "is_connected",
    "node_connected_component",
]


@not_implemented_for("directed")
@nx._dispatchable
def connected_components(G):
    """Generate connected components.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph

    Returns
    -------
    comp : generator of sets
       A generator of sets of nodes, one for each component of G.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Examples
    --------
    Generate a sorted list of connected components, largest first.

    >>> G = nx.path_graph(4)
    >>> nx.add_path(G, [10, 11, 12])
    >>> [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    [4, 3]

    If you only want the largest connected component, it's more
    efficient to use max instead of sort.

    >>> largest_cc = max(nx.connected_components(G), key=len)

    To create the induced subgraph of each component use:

    >>> S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    See Also
    --------
    strongly_connected_components
    weakly_connected_components

    Notes
    -----
    For undirected graphs only.

    """
    seen = set()
    n = len(G)
    for v in G:
        if v not in seen:
            c = _plain_bfs(G, n, v)
            seen.update(c)
            yield c


@not_implemented_for("directed")
@nx._dispatchable
def number_connected_components(G):
    """Returns the number of connected components.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    Returns
    -------
    n : integer
       Number of connected components

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (5, 6), (3, 4)])
    >>> nx.number_connected_components(G)
    3

    See Also
    --------
    connected_components
    number_weakly_connected_components
    number_strongly_connected_components

    Notes
    -----
    For undirected graphs only.

    """
    return sum(1 for cc in connected_components(G))


@not_implemented_for("directed")
@nx._dispatchable
def is_connected(G):
    """Returns True if the graph is connected, False otherwise.

    Parameters
    ----------
    G : NetworkX Graph
       An undirected graph.

    Returns
    -------
    connected : bool
      True if the graph is connected, false otherwise.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> print(nx.is_connected(G))
    True

    See Also
    --------
    is_strongly_connected
    is_weakly_connected
    is_semiconnected
    is_biconnected
    connected_components

    Notes
    -----
    For undirected graphs only.

    """
    n = len(G)
    if n == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined for the null graph."
        )
    return sum(1 for node in _plain_bfs(G, n, arbitrary_element(G))) == len(G)


@not_implemented_for("directed")
@nx._dispatchable
def node_connected_component(G, n):
    """Returns the set of nodes in the component of graph containing node n.

    Parameters
    ----------
    G : NetworkX Graph
       An undirected graph.

    n : node label
       A node in G

    Returns
    -------
    comp : set
       A set of nodes in the component of G containing node n.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (5, 6), (3, 4)])
    >>> nx.node_connected_component(G, 0)  # nodes of component that contains node 0
    {0, 1, 2}

    See Also
    --------
    connected_components

    Notes
    -----
    For undirected graphs only.

    """
    return _plain_bfs(G, len(G), n)


def _plain_bfs(G, n, source):
    """A fast BFS node generator"""
    adj = G._adj
    seen = {source}
    nextlevel = [source]
    while nextlevel:
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
            if len(seen) == n:
                return seen
    return seen
<|endoftext|>"""Biconnected components and articulation points."""

from itertools import chain

import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = [
    "biconnected_components",
    "biconnected_component_edges",
    "is_biconnected",
    "articulation_points",
]


@not_implemented_for("directed")
@nx._dispatchable
def is_biconnected(G):
    """Returns True if the graph is biconnected, False otherwise.

    A graph is biconnected if, and only if, it cannot be disconnected by
    removing only one node (and all edges incident on that node).  If
    removing a node increases the number of disconnected components
    in the graph, that node is called an articulation point, or cut
    vertex.  A biconnected graph has no articulation points.

    Parameters
    ----------
    G : NetworkX Graph
        An undirected graph.

    Returns
    -------
    biconnected : bool
        True if the graph is biconnected, False otherwise.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is not undirected.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> print(nx.is_biconnected(G))
    False
    >>> G.add_edge(0, 3)
    >>> print(nx.is_biconnected(G))
    True

    See Also
    --------
    biconnected_components
    articulation_points
    biconnected_component_edges
    is_strongly_connected
    is_weakly_connected
    is_connected
    is_semiconnected

    Notes
    -----
    The algorithm to find articulation points and biconnected
    components is implemented using a non-recursive depth-first-search
    (DFS) that keeps track of the highest level that back edges reach
    in the DFS tree.  A node `n` is an articulation point if, and only
    if, there exists a subtree rooted at `n` such that there is no
    back edge from any successor of `n` that links to a predecessor of
    `n` in the DFS tree.  By keeping track of all the edges traversed
    by the DFS we can obtain the biconnected components because all
    edges of a bicomponent will be traversed consecutively between
    articulation points.

    References
    ----------
    .. [1] Hopcroft, J.; Tarjan, R. (1973).
       "Efficient algorithms for graph manipulation".
       Communications of the ACM 16: 372–378. doi:10.1145/362248.362272

    """
    bccs = biconnected_components(G)
    try:
        bcc = next(bccs)
    except StopIteration:
        # No bicomponents (empty graph?)
        return False
    try:
        next(bccs)
    except StopIteration:
        # Only one bicomponent
        return len(bcc) == len(G)
    else:
        # Multiple bicomponents
        return False


@not_implemented_for("directed")
@nx._dispatchable
def biconnected_component_edges(G):
    """Returns a generator of lists of edges, one list for each biconnected
    component of the input graph.

    Biconnected components are maximal subgraphs such that the removal of a
    node (and all edges incident on that node) will not disconnect the
    subgraph.  Note that nodes may be part of more than one biconnected
    component.  Those nodes are articulation points, or cut vertices.
    However, each edge belongs to one, and only one, biconnected component.

    Notice that by convention a dyad is considered a biconnected component.

    Parameters
    ----------
    G : NetworkX Graph
        An undirected graph.

    Returns
    -------
    edges : generator of lists
        Generator of lists of edges, one list for each bicomponent.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is not undirected.

    Examples
    --------
    >>> G = nx.barbell_graph(4, 2)
    >>> print(nx.is_biconnected(G))
    False
    >>> bicomponents_edges = list(nx.biconnected_component_edges(G))
    >>> len(bicomponents_edges)
    5
    >>> G.add_edge(2, 8)
    >>> print(nx.is_biconnected(G))
    True
    >>> bicomponents_edges = list(nx.biconnected_component_edges(G))
    >>> len(bicomponents_edges)
    1

    See Also
    --------
    is_biconnected,
    biconnected_components,
    articulation_points,

    Notes
    -----
    The algorithm to find articulation points and biconnected
    components is implemented using a non-recursive depth-first-search
    (DFS) that keeps track of the highest level that back edges reach
    in the DFS tree.  A node `n` is an articulation point if, and only
    if, there exists a subtree rooted at `n` such that there is no
    back edge from any successor of `n` that links to a predecessor of
    `n` in the DFS tree.  By keeping track of all the edges traversed
    by the DFS we can obtain the biconnected components because all
    edges of a bicomponent will be traversed consecutively between
    articulation points.

    References
    ----------
    .. [1] Hopcroft, J.; Tarjan, R. (1973).
           "Efficient algorithms for graph manipulation".
           Communications of the ACM 16: 372–378. doi:10.1145/362248.362272

    """
    yield from _biconnected_dfs(G, components=True)


@not_implemented_for("directed")
@nx._dispatchable
def biconnected_components(G):
    """Returns a generator of sets of nodes, one set for each biconnected
    component of the graph

    Biconnected components are maximal subgraphs such that the removal of a
    node (and all edges incident on that node) will not disconnect the
    subgraph. Note that nodes may be part of more than one biconnected
    component.  Those nodes are articulation points, or cut vertices.  The
    removal of articulation points will increase the number of connected
    components of the graph.

    Notice that by convention a dyad is considered a biconnected component.

    Parameters
    ----------
    G : NetworkX Graph
        An undirected graph.

    Returns
    -------
    nodes : generator
        Generator of sets of nodes, one set for each biconnected component.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is not undirected.

    Examples
    --------
    >>> G = nx.lollipop_graph(5, 1)
    >>> print(nx.is_biconnected(G))
    False
    >>> bicomponents = list(nx.biconnected_components(G))
    >>> len(bicomponents)
    2
    >>> G.add_edge(0, 5)
    >>> print(nx.is_biconnected(G))
    True
    >>> bicomponents = list(nx.biconnected_components(G))
    >>> len(bicomponents)
    1

    You can generate a sorted list of biconnected components, largest
    first, using sort.

    >>> G.remove_edge(0, 5)
    >>> [len(c) for c in sorted(nx.biconnected_components(G), key=len, reverse=True)]
    [5, 2]

    If you only want the largest connected component, it's more
    efficient to use max instead of sort.

    >>> Gc = max(nx.biconnected_components(G), key=len)

    To create the components as subgraphs use:
    ``(G.subgraph(c).copy() for c in biconnected_components(G))``

    See Also
    --------
    is_biconnected
    articulation_points
    biconnected_component_edges
    k_components : this function is a special case where k=2
    bridge_components : similar to this function, but is defined using
        2-edge-connectivity instead of 2-node-connectivity.

    Notes
    -----
    The algorithm to find articulation points and biconnected
    components is implemented using a non-recursive depth-first-search
    (DFS) that keeps track of the highest level that back edges reach
    in the DFS tree.  A node `n` is an articulation point if, and only
    if, there exists a subtree rooted at `n` such that there is no
    back edge from any successor of `n` that links to a predecessor of
    `n` in the DFS tree.  By keeping track of all the edges traversed
    by the DFS we can obtain the biconnected components because all
    edges of a bicomponent will be traversed consecutively between
    articulation points.

    References
    ----------
    .. [1] Hopcroft, J.; Tarjan, R. (1973).
           "Efficient algorithms for graph manipulation".
           Communications of the ACM 16: 372–378. doi:10.1145/362248.362272

    """
    for comp in _biconnected_dfs(G, components=True):
        yield set(chain.from_iterable(comp))


@not_implemented_for("directed")
@nx._dispatchable
def articulation_points(G):
    """Yield the articulation points, or cut vertices, of a graph.

    An articulation point or cut vertex is any node whose removal (along with
    all its incident edges) increases the number of connected components of
    a graph.  An undirected connected graph without articulation points is
    biconnected. Articulation points belong to more than one biconnected
    component of a graph.

    Notice that by convention a dyad is considered a biconnected component.

    Parameters
    ----------
    G : NetworkX Graph
        An undirected graph.

    Yields
    ------
    node
        An articulation point in the graph.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is not undirected.

    Examples
    --------

    >>> G = nx.barbell_graph(4, 2)
    >>> print(nx.is_biconnected(G))
    False
    >>> len(list(nx.articulation_points(G)))
    4
    >>> G.add_edge(2, 8)
    >>> print(nx.is_biconnected(G))
    True
    >>> len(list(nx.articulation_points(G)))
    0

    See Also
    --------
    is_biconnected
    biconnected_components
    biconnected_component_edges

    Notes
    -----
    The algorithm to find articulation points and biconnected
    components is implemented using a non-recursive depth-first-search
    (DFS) that keeps track of the highest level that back edges reach
    in the DFS tree.  A node `n` is an articulation point if, and only
    if, there exists a subtree rooted at `n` such that there is no
    back edge from any successor of `n` that links to a predecessor of
    `n` in the DFS tree.  By keeping track of all the edges traversed
    by the DFS we can obtain the biconnected components because all
    edges of a bicomponent will be traversed consecutively between
    articulation points.

    References
    ----------
    .. [1] Hopcroft, J.; Tarjan, R. (1973).
           "Efficient algorithms for graph manipulation".
           Communications of the ACM 16: 372–378. doi:10.1145/362248.362272

    """
    seen = set()
    for articulation in _biconnected_dfs(G, components=False):
        if articulation not in seen:
            seen.add(articulation)
            yield articulation


@not_implemented_for("directed")
def _biconnected_dfs(G, components=True):
    # depth-first search algorithm to generate articulation points
    # and biconnected components
    visited = set()
    for start in G:
        if start in visited:
            continue
        discovery = {start: 0}  # time of first discovery of node during search
        low = {start: 0}
        root_children = 0
        visited.add(start)
        edge_stack = []
        stack = [(start, start, iter(G[start]))]
        edge_index = {}
        while stack:
            grandparent, parent, children = stack[-1]
            try:
                child = next(children)
                if grandparent == child:
                    continue
                if child in visited:
                    if discovery[child] <= discovery[parent]:  # back edge
                        low[parent] = min(low[parent], discovery[child])
                        if components:
                            edge_index[parent, child] = len(edge_stack)
                            edge_stack.append((parent, child))
                else:
                    low[child] = discovery[child] = len(discovery)
                    visited.add(child)
                    stack.append((parent, child, iter(G[child])))
                    if components:
                        edge_index[parent, child] = len(edge_stack)
                        edge_stack.append((parent, child))

            except StopIteration:
                stack.pop()
                if len(stack) > 1:
                    if low[parent] >= discovery[grandparent]:
                        if components:
                            ind = edge_index[grandparent, parent]
                            yield edge_stack[ind:]
                            del edge_stack[ind:]

                        else:
                            yield grandparent
                    low[grandparent] = min(low[parent], low[grandparent])
                elif stack:  # length 1 so grandparent is root
                    root_children += 1
                    if components:
                        ind = edge_index[grandparent, parent]
                        yield edge_stack[ind:]
                        del edge_stack[ind:]
        if not components:
            # root node is articulation point if it has more than 1 child
            if root_children > 1:
                yield start
<|endoftext|>"""Node assortativity coefficients and correlation measures."""

import networkx as nx
from networkx.algorithms.assortativity.mixing import (
    attribute_mixing_matrix,
    degree_mixing_matrix,
)
from networkx.algorithms.assortativity.pairs import node_degree_xy

__all__ = [
    "degree_pearson_correlation_coefficient",
    "degree_assortativity_coefficient",
    "attribute_assortativity_coefficient",
    "numeric_assortativity_coefficient",
]


@nx._dispatchable(edge_attrs="weight")
def degree_assortativity_coefficient(G, x="out", y="in", weight=None, nodes=None):
    """Compute degree assortativity of graph.

    Assortativity measures the similarity of connections
    in the graph with respect to the node degree.

    Parameters
    ----------
    G : NetworkX graph

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    nodes: list or iterable (optional)
        Compute degree assortativity only for nodes in container.
        The default is all nodes.

    Returns
    -------
    r : float
       Assortativity of graph by degree.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> r = nx.degree_assortativity_coefficient(G)
    >>> print(f"{r:3.1f}")
    -0.5

    See Also
    --------
    attribute_assortativity_coefficient
    numeric_assortativity_coefficient
    degree_mixing_dict
    degree_mixing_matrix

    Notes
    -----
    This computes Eq. (21) in Ref. [1]_ , where e is the joint
    probability distribution (mixing matrix) of the degrees.  If G is
    directed than the matrix e is the joint probability of the
    user-specified degree type for the source and target.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    .. [2] Foster, J.G., Foster, D.V., Grassberger, P. & Paczuski, M.
       Edge direction and the structure of networks, PNAS 107, 10815-20 (2010).
    """
    if nodes is None:
        nodes = G.nodes

    degrees = None

    if G.is_directed():
        indeg = (
            {d for _, d in G.in_degree(nodes, weight=weight)}
            if "in" in (x, y)
            else set()
        )
        outdeg = (
            {d for _, d in G.out_degree(nodes, weight=weight)}
            if "out" in (x, y)
            else set()
        )
        degrees = set.union(indeg, outdeg)
    else:
        degrees = {d for _, d in G.degree(nodes, weight=weight)}

    mapping = {d: i for i, d in enumerate(degrees)}
    M = degree_mixing_matrix(G, x=x, y=y, nodes=nodes, weight=weight, mapping=mapping)

    return _numeric_ac(M, mapping=mapping)


@nx._dispatchable(edge_attrs="weight")
def degree_pearson_correlation_coefficient(G, x="out", y="in", weight=None, nodes=None):
    """Compute degree assortativity of graph.

    Assortativity measures the similarity of connections
    in the graph with respect to the node degree.

    This is the same as degree_assortativity_coefficient but uses the
    potentially faster scipy.stats.pearsonr function.

    Parameters
    ----------
    G : NetworkX graph

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    nodes: list or iterable (optional)
        Compute pearson correlation of degrees only for specified nodes.
        The default is all nodes.

    Returns
    -------
    r : float
       Assortativity of graph by degree.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> r = nx.degree_pearson_correlation_coefficient(G)
    >>> print(f"{r:3.1f}")
    -0.5

    Notes
    -----
    This calls scipy.stats.pearsonr.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks
           Physical Review E, 67 026126, 2003
    .. [2] Foster, J.G., Foster, D.V., Grassberger, P. & Paczuski, M.
       Edge direction and the structure of networks, PNAS 107, 10815-20 (2010).
    """
    import scipy as sp

    xy = node_degree_xy(G, x=x, y=y, nodes=nodes, weight=weight)
    x, y = zip(*xy)
    return float(sp.stats.pearsonr(x, y)[0])


@nx._dispatchable(node_attrs="attribute")
def attribute_assortativity_coefficient(G, attribute, nodes=None):
    """Compute assortativity for node attributes.

    Assortativity measures the similarity of connections
    in the graph with respect to the given attribute.

    Parameters
    ----------
    G : NetworkX graph

    attribute : string
        Node attribute key

    nodes: list or iterable (optional)
        Compute attribute assortativity for nodes in container.
        The default is all nodes.

    Returns
    -------
    r: float
       Assortativity of graph for given attribute

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_nodes_from([0, 1], color="red")
    >>> G.add_nodes_from([2, 3], color="blue")
    >>> G.add_edges_from([(0, 1), (2, 3)])
    >>> print(nx.attribute_assortativity_coefficient(G, "color"))
    1.0

    Notes
    -----
    This computes Eq. (2) in Ref. [1]_ , (trace(M)-sum(M^2))/(1-sum(M^2)),
    where M is the joint probability distribution (mixing matrix)
    of the specified attribute.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    """
    M = attribute_mixing_matrix(G, attribute, nodes)
    return attribute_ac(M)


@nx._dispatchable(node_attrs="attribute")
def numeric_assortativity_coefficient(G, attribute, nodes=None):
    """Compute assortativity for numerical node attributes.

    Assortativity measures the similarity of connections
    in the graph with respect to the given numeric attribute.

    Parameters
    ----------
    G : NetworkX graph

    attribute : string
        Node attribute key.

    nodes: list or iterable (optional)
        Compute numeric assortativity only for attributes of nodes in
        container. The default is all nodes.

    Returns
    -------
    r: float
       Assortativity of graph for given attribute

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_nodes_from([0, 1], size=2)
    >>> G.add_nodes_from([2, 3], size=3)
    >>> G.add_edges_from([(0, 1), (2, 3)])
    >>> print(nx.numeric_assortativity_coefficient(G, "size"))
    1.0

    Notes
    -----
    This computes Eq. (21) in Ref. [1]_ , which is the Pearson correlation
    coefficient of the specified (scalar valued) attribute across edges.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks
           Physical Review E, 67 026126, 2003
    """
    if nodes is None:
        nodes = G.nodes
    vals = {G.nodes[n][attribute] for n in nodes}
    mapping = {d: i for i, d in enumerate(vals)}
    M = attribute_mixing_matrix(G, attribute, nodes, mapping)
    return _numeric_ac(M, mapping)


def attribute_ac(M):
    """Compute assortativity for attribute matrix M.

    Parameters
    ----------
    M : numpy.ndarray
        2D ndarray representing the attribute mixing matrix.

    Notes
    -----
    This computes Eq. (2) in Ref. [1]_ , (trace(e)-sum(e^2))/(1-sum(e^2)),
    where e is the joint probability distribution (mixing matrix)
    of the specified attribute.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    """
    if M.sum() != 1.0:
        M = M / M.sum()
    s = (M @ M).sum()
    t = M.trace()
    r = (t - s) / (1 - s)
    return float(r)


def _numeric_ac(M, mapping):
    # M is a 2D numpy array
    # numeric assortativity coefficient, pearsonr
    import numpy as np

    if M.sum() != 1.0:
        M = M / M.sum()
    x = np.array(list(mapping.keys()))
    y = x  # x and y have the same support
    idx = list(mapping.values())
    a = M.sum(axis=0)
    b = M.sum(axis=1)
    vara = (a[idx] * x**2).sum() - ((a[idx] * x).sum()) ** 2
    varb = (b[idx] * y**2).sum() - ((b[idx] * y).sum()) ** 2
    xy = np.outer(x, y)
    ab = np.outer(a[idx], b[idx])
    return float((xy * (M - ab)).sum() / np.sqrt(vara * varb))
<|endoftext|>from collections import defaultdict

import networkx as nx

__all__ = ["average_degree_connectivity"]


@nx._dispatchable(edge_attrs="weight")
def average_degree_connectivity(
    G, source="in+out", target="in+out", nodes=None, weight=None
):
    r"""Compute the average degree connectivity of graph.

    The average degree connectivity is the average nearest neighbor degree of
    nodes with degree k. For weighted graphs, an analogous measure can
    be computed using the weighted average neighbors degree defined in
    [1]_, for a node `i`, as

    .. math::

        k_{nn,i}^{w} = \frac{1}{s_i} \sum_{j \in N(i)} w_{ij} k_j

    where `s_i` is the weighted degree of node `i`,
    `w_{ij}` is the weight of the edge that links `i` and `j`,
    and `N(i)` are the neighbors of node `i`.

    Parameters
    ----------
    G : NetworkX graph

    source :  "in"|"out"|"in+out" (default:"in+out")
       Directed graphs only. Use "in"- or "out"-degree for source node.

    target : "in"|"out"|"in+out" (default:"in+out"
       Directed graphs only. Use "in"- or "out"-degree for target node.

    nodes : list or iterable (optional)
        Compute neighbor connectivity for these nodes. The default is all
        nodes.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    d : dict
       A dictionary keyed by degree k with the value of average connectivity.

    Raises
    ------
    NetworkXError
        If either `source` or `target` are not one of 'in',
        'out', or 'in+out'.
        If either `source` or `target` is passed for an undirected graph.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> G.edges[1, 2]["weight"] = 3
    >>> nx.average_degree_connectivity(G)
    {1: 2.0, 2: 1.5}
    >>> nx.average_degree_connectivity(G, weight="weight")
    {1: 2.0, 2: 1.75}

    See Also
    --------
    average_neighbor_degree

    References
    ----------
    .. [1] A. Barrat, M. Barthélemy, R. Pastor-Satorras, and A. Vespignani,
       "The architecture of complex weighted networks".
       PNAS 101 (11): 3747–3752 (2004).
    """
    # First, determine the type of neighbors and the type of degree to use.
    if G.is_directed():
        if source not in ("in", "out", "in+out"):
            raise nx.NetworkXError('source must be one of "in", "out", or "in+out"')
        if target not in ("in", "out", "in+out"):
            raise nx.NetworkXError('target must be one of "in", "out", or "in+out"')
        direction = {"out": G.out_degree, "in": G.in_degree, "in+out": G.degree}
        neighbor_funcs = {
            "out": G.successors,
            "in": G.predecessors,
            "in+out": G.neighbors,
        }
        source_degree = direction[source]
        target_degree = direction[target]
        neighbors = neighbor_funcs[source]
        # `reverse` indicates whether to look at the in-edge when
        # computing the weight of an edge.
        reverse = source == "in"
    else:
        if source != "in+out" or target != "in+out":
            raise nx.NetworkXError(
                f"source and target arguments are only supported for directed graphs"
            )
        source_degree = G.degree
        target_degree = G.degree
        neighbors = G.neighbors
        reverse = False
    dsum = defaultdict(int)
    dnorm = defaultdict(int)
    # Check if `source_nodes` is actually a single node in the graph.
    source_nodes = source_degree(nodes)
    if nodes in G:
        source_nodes = [(nodes, source_degree(nodes))]
    for n, k in source_nodes:
        nbrdeg = target_degree(neighbors(n))
        if weight is None:
            s = sum(d for n, d in nbrdeg)
        else:  # weight nbr degree by weight of (n,nbr) edge
            if reverse:
                s = sum(G[nbr][n].get(weight, 1) * d for nbr, d in nbrdeg)
            else:
                s = sum(G[n][nbr].get(weight, 1) * d for nbr, d in nbrdeg)
        dnorm[k] += source_degree(n, weight=weight)
        dsum[k] += s

    # normalize
    return {k: avg if dnorm[k] == 0 else avg / dnorm[k] for k, avg in dsum.items()}
<|endoftext|>"""Generators of x-y pairs of node data."""

import networkx as nx

__all__ = ["node_attribute_xy", "node_degree_xy"]


@nx._dispatchable(node_attrs="attribute")
def node_attribute_xy(G, attribute, nodes=None):
    """Yields 2-tuples of node attribute values for all edges in `G`.

    This generator yields, for each edge in `G` incident to a node in `nodes`,
    a 2-tuple of form ``(attribute value,  attribute value)`` for the parameter
    specified node-attribute.

    Parameters
    ----------
    G: NetworkX graph

    attribute: key
        The node attribute key.

    nodes: list or iterable (optional)
        Use only edges that are incident to specified nodes.
        The default is all nodes.

    Yields
    ------
    (x, y): 2-tuple
        Generates 2-tuple of (attribute, attribute) values.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_node(1, color="red")
    >>> G.add_node(2, color="blue")
    >>> G.add_node(3, color="green")
    >>> G.add_edge(1, 2)
    >>> list(nx.node_attribute_xy(G, "color"))
    [('red', 'blue')]

    Notes
    -----
    For undirected graphs, each edge is produced twice, once for each edge
    representation (u, v) and (v, u), with the exception of self-loop edges
    which only appear once.
    """
    if nodes is None:
        nodes = set(G)
    else:
        nodes = set(nodes)
    Gnodes = G.nodes
    for u, nbrsdict in G.adjacency():
        if u not in nodes:
            continue
        uattr = Gnodes[u].get(attribute, None)
        if G.is_multigraph():
            for v, keys in nbrsdict.items():
                vattr = Gnodes[v].get(attribute, None)
                for _ in keys:
                    yield (uattr, vattr)
        else:
            for v in nbrsdict:
                vattr = Gnodes[v].get(attribute, None)
                yield (uattr, vattr)


@nx._dispatchable(edge_attrs="weight")
def node_degree_xy(G, x="out", y="in", weight=None, nodes=None):
    """Yields 2-tuples of ``(degree, degree)`` values for edges in `G`.

    This generator yields, for each edge in `G` incident to a node in `nodes`,
    a 2-tuple of form ``(degree, degree)``. The node degrees are weighted
    when a `weight` attribute is specified.

    Parameters
    ----------
    G: NetworkX graph

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    nodes: list or iterable (optional)
        Use only edges that are adjacency to specified nodes.
        The default is all nodes.

    Yields
    ------
    (x, y): 2-tuple
        Generates 2-tuple of (degree, degree) values.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2)
    >>> list(nx.node_degree_xy(G, x="out", y="in"))
    [(1, 1)]
    >>> list(nx.node_degree_xy(G, x="in", y="out"))
    [(0, 0)]

    Notes
    -----
    For undirected graphs, each edge is produced twice, once for each edge
    representation (u, v) and (v, u), with the exception of self-loop edges
    which only appear once.
    """
    nodes = set(G) if nodes is None else set(nodes)
    if G.is_directed():
        direction = {"out": G.out_degree, "in": G.in_degree}
        xdeg = direction[x]
        ydeg = direction[y]
    else:
        xdeg = ydeg = G.degree

    for u, degu in xdeg(nodes, weight=weight):
        # use G.edges to treat multigraphs correctly
        neighbors = (nbr for _, nbr in G.edges(u) if nbr in nodes)
        for _, degv in ydeg(neighbors, weight=weight):
            yield degu, degv
<|endoftext|>import networkx as nx

__all__ = ["average_neighbor_degree"]


@nx._dispatchable(edge_attrs="weight")
def average_neighbor_degree(G, source="out", target="out", nodes=None, weight=None):
    r"""Returns the average degree of the neighborhood of each node.

    In an undirected graph, the neighborhood `N(i)` of node `i` contains the
    nodes that are connected to `i` by an edge.

    For directed graphs, `N(i)` is defined according to the parameter `source`:

        - if source is 'in', then `N(i)` consists of predecessors of node `i`.
        - if source is 'out', then `N(i)` consists of successors of node `i`.
        - if source is 'in+out', then `N(i)` is both predecessors and successors.

    The average neighborhood degree of a node `i` is

    .. math::

        k_{nn,i} = \frac{1}{|N(i)|} \sum_{j \in N(i)} k_j

    where `N(i)` are the neighbors of node `i` and `k_j` is
    the degree of node `j` which belongs to `N(i)`. For weighted
    graphs, an analogous measure can be defined [1]_,

    .. math::

        k_{nn,i}^{w} = \frac{1}{s_i} \sum_{j \in N(i)} w_{ij} k_j

    where `s_i` is the weighted degree of node `i`, `w_{ij}`
    is the weight of the edge that links `i` and `j` and
    `N(i)` are the neighbors of node `i`.


    Parameters
    ----------
    G : NetworkX graph

    source : string ("in"|"out"|"in+out"), optional (default="out")
       Directed graphs only.
       Use "in"- or "out"-neighbors of source node.

    target : string ("in"|"out"|"in+out"), optional (default="out")
       Directed graphs only.
       Use "in"- or "out"-degree for target node.

    nodes : list or iterable, optional (default=G.nodes)
        Compute neighbor degree only for specified nodes.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    d: dict
       A dictionary keyed by node to the average degree of its neighbors.

    Raises
    ------
    NetworkXError
        If either `source` or `target` are not one of 'in', 'out', or 'in+out'.
        If either `source` or `target` is passed for an undirected graph.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> G.edges[0, 1]["weight"] = 5
    >>> G.edges[2, 3]["weight"] = 3

    >>> nx.average_neighbor_degree(G)
    {0: 2.0, 1: 1.5, 2: 1.5, 3: 2.0}
    >>> nx.average_neighbor_degree(G, weight="weight")
    {0: 2.0, 1: 1.1666666666666667, 2: 1.25, 3: 2.0}

    >>> G = nx.DiGraph()
    >>> nx.add_path(G, [0, 1, 2, 3])
    >>> nx.average_neighbor_degree(G, source="in", target="in")
    {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}

    >>> nx.average_neighbor_degree(G, source="out", target="out")
    {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}

    See Also
    --------
    average_degree_connectivity

    References
    ----------
    .. [1] A. Barrat, M. Barthélemy, R. Pastor-Satorras, and A. Vespignani,
       "The architecture of complex weighted networks".
       PNAS 101 (11): 3747–3752 (2004).
    """
    if G.is_directed():
        if source == "in":
            source_degree = G.in_degree
        elif source == "out":
            source_degree = G.out_degree
        elif source == "in+out":
            source_degree = G.degree
        else:
            raise nx.NetworkXError(
                f"source argument {source} must be 'in', 'out' or 'in+out'"
            )

        if target == "in":
            target_degree = G.in_degree
        elif target == "out":
            target_degree = G.out_degree
        elif target == "in+out":
            target_degree = G.degree
        else:
            raise nx.NetworkXError(
                f"target argument {target} must be 'in', 'out' or 'in+out'"
            )
    else:
        if source != "out" or target != "out":
            raise nx.NetworkXError(
                f"source and target arguments are only supported for directed graphs"
            )
        source_degree = target_degree = G.degree

    # precompute target degrees -- should *not* be weighted degree
    t_deg = dict(target_degree())

    # Set up both predecessor and successor neighbor dicts leaving empty if not needed
    G_P = G_S = {n: {} for n in G}
    if G.is_directed():
        # "in" or "in+out" cases: G_P contains predecessors
        if "in" in source:
            G_P = G.pred
        # "out" or "in+out" cases: G_S contains successors
        if "out" in source:
            G_S = G.succ
    else:
        # undirected leave G_P empty but G_S is the adjacency
        G_S = G.adj

    # Main loop: Compute average degree of neighbors
    avg = {}
    for n, deg in source_degree(nodes, weight=weight):
        # handle degree zero average
        if deg == 0:
            avg[n] = 0.0
            continue

        # we sum over both G_P and G_S, but one of the two is usually empty.
        if weight is None:
            avg[n] = (
                sum(t_deg[nbr] for nbr in G_S[n]) + sum(t_deg[nbr] for nbr in G_P[n])
            ) / deg
        else:
            avg[n] = (
                sum(dd.get(weight, 1) * t_deg[nbr] for nbr, dd in G_S[n].items())
                + sum(dd.get(weight, 1) * t_deg[nbr] for nbr, dd in G_P[n].items())
            ) / deg
    return avg
<|endoftext|>"""
Mixing matrices for node attributes and degree.
"""

import networkx as nx
from networkx.algorithms.assortativity.pairs import node_attribute_xy, node_degree_xy
from networkx.utils import dict_to_numpy_array

__all__ = [
    "attribute_mixing_matrix",
    "attribute_mixing_dict",
    "degree_mixing_matrix",
    "degree_mixing_dict",
    "mixing_dict",
]


@nx._dispatchable(node_attrs="attribute")
def attribute_mixing_dict(G, attribute, nodes=None, normalized=False):
    """Returns dictionary representation of mixing matrix for attribute.

    Parameters
    ----------
    G : graph
       NetworkX graph object.

    attribute : string
       Node attribute key.

    nodes: list or iterable (optional)
        Unse nodes in container to build the dict. The default is all nodes.

    normalized : bool (default=False)
       Return counts if False or probabilities if True.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_nodes_from([0, 1], color="red")
    >>> G.add_nodes_from([2, 3], color="blue")
    >>> G.add_edge(1, 3)
    >>> d = nx.attribute_mixing_dict(G, "color")
    >>> print(d["red"]["blue"])
    1
    >>> print(d["blue"]["red"])  # d symmetric for undirected graphs
    1

    Returns
    -------
    d : dictionary
       Counts or joint probability of occurrence of attribute pairs.
    """
    xy_iter = node_attribute_xy(G, attribute, nodes)
    return mixing_dict(xy_iter, normalized=normalized)


@nx._dispatchable(node_attrs="attribute")
def attribute_mixing_matrix(G, attribute, nodes=None, mapping=None, normalized=True):
    """Returns mixing matrix for attribute.

    Parameters
    ----------
    G : graph
       NetworkX graph object.

    attribute : string
       Node attribute key.

    nodes: list or iterable (optional)
        Use only nodes in container to build the matrix. The default is
        all nodes.

    mapping : dictionary, optional
       Mapping from node attribute to integer index in matrix.
       If not specified, an arbitrary ordering will be used.

    normalized : bool (default=True)
       Return counts if False or probabilities if True.

    Returns
    -------
    m: numpy array
       Counts or joint probability of occurrence of attribute pairs.

    Notes
    -----
    If each node has a unique attribute value, the unnormalized mixing matrix
    will be equal to the adjacency matrix. To get a denser mixing matrix,
    the rounding can be performed to form groups of nodes with equal values.
    For example, the exact height of persons in cm (180.79155222, 163.9080892,
    163.30095355, 167.99016217, 168.21590163, ...) can be rounded to (180, 163,
    163, 168, 168, ...).

    Definitions of attribute mixing matrix vary on whether the matrix
    should include rows for attribute values that don't arise. Here we
    do not include such empty-rows. But you can force them to appear
    by inputting a `mapping` that includes those values.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> gender = {0: "male", 1: "female", 2: "female"}
    >>> nx.set_node_attributes(G, gender, "gender")
    >>> mapping = {"male": 0, "female": 1}
    >>> mix_mat = nx.attribute_mixing_matrix(G, "gender", mapping=mapping)
    >>> mix_mat
    array([[0.  , 0.25],
           [0.25, 0.5 ]])
    """
    d = attribute_mixing_dict(G, attribute, nodes)
    a = dict_to_numpy_array(d, mapping=mapping)
    if normalized:
        a = a / a.sum()
    return a


@nx._dispatchable(edge_attrs="weight")
def degree_mixing_dict(G, x="out", y="in", weight=None, nodes=None, normalized=False):
    """Returns dictionary representation of mixing matrix for degree.

    Parameters
    ----------
    G : graph
        NetworkX graph object.

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    normalized : bool (default=False)
        Return counts if False or probabilities if True.

    Returns
    -------
    d: dictionary
       Counts or joint probability of occurrence of degree pairs.
    """
    xy_iter = node_degree_xy(G, x=x, y=y, nodes=nodes, weight=weight)
    return mixing_dict(xy_iter, normalized=normalized)


@nx._dispatchable(edge_attrs="weight")
def degree_mixing_matrix(
    G, x="out", y="in", weight=None, nodes=None, normalized=True, mapping=None
):
    """Returns mixing matrix for attribute.

    Parameters
    ----------
    G : graph
       NetworkX graph object.

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    nodes: list or iterable (optional)
        Build the matrix using only nodes in container.
        The default is all nodes.

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    normalized : bool (default=True)
       Return counts if False or probabilities if True.

    mapping : dictionary, optional
       Mapping from node degree to integer index in matrix.
       If not specified, an arbitrary ordering will be used.

    Returns
    -------
    m: numpy array
       Counts, or joint probability, of occurrence of node degree.

    Notes
    -----
    Definitions of degree mixing matrix vary on whether the matrix
    should include rows for degree values that don't arise. Here we
    do not include such empty-rows. But you can force them to appear
    by inputting a `mapping` that includes those values. See examples.

    Examples
    --------
    >>> G = nx.star_graph(3)
    >>> mix_mat = nx.degree_mixing_matrix(G)
    >>> mix_mat
    array([[0. , 0.5],
           [0.5, 0. ]])

    If you want every possible degree to appear as a row, even if no nodes
    have that degree, use `mapping` as follows,

    >>> max_degree = max(deg for n, deg in G.degree)
    >>> mapping = {x: x for x in range(max_degree + 1)}  # identity mapping
    >>> mix_mat = nx.degree_mixing_matrix(G, mapping=mapping)
    >>> mix_mat
    array([[0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0.5],
           [0. , 0. , 0. , 0. ],
           [0. , 0.5, 0. , 0. ]])
    """
    d = degree_mixing_dict(G, x=x, y=y, nodes=nodes, weight=weight)
    a = dict_to_numpy_array(d, mapping=mapping)
    if normalized:
        a = a / a.sum()
    return a


def mixing_dict(xy, normalized=False):
    """Returns a dictionary representation of mixing matrix.

    Parameters
    ----------
    xy : list or container of two-tuples
       Pairs of (x,y) items.

    attribute : string
       Node attribute key

    normalized : bool (default=False)
       Return counts if False or probabilities if True.

    Returns
    -------
    d: dictionary
       Counts or Joint probability of occurrence of values in xy.
    """
    d = {}
    psum = 0.0
    for x, y in xy:
        if x not in d:
            d[x] = {}
        if y not in d:
            d[y] = {}
        v = d[x].get(y, 0)
        d[x][y] = v + 1
        psum += 1

    if normalized:
        for _, jdict in d.items():
            for j in jdict:
                jdict[j] /= psum
    return d
<|endoftext|>from networkx.algorithms.assortativity.connectivity import *
from networkx.algorithms.assortativity.correlation import *
from networkx.algorithms.assortativity.mixing import *
from networkx.algorithms.assortativity.neighbor_degree import *
from networkx.algorithms.assortativity.pairs import *
<|endoftext|>"""Functions for encoding and decoding trees.

Since a tree is a highly restricted form of graph, it can be represented
concisely in several ways. This module includes functions for encoding
and decoding trees in the form of nested tuples and Prüfer
sequences. The former requires a rooted tree, whereas the latter can be
applied to unrooted trees. Furthermore, there is a bijection from Prüfer
sequences to labeled trees.

"""

from collections import Counter
from itertools import chain

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "from_nested_tuple",
    "from_prufer_sequence",
    "NotATree",
    "to_nested_tuple",
    "to_prufer_sequence",
]


class NotATree(nx.NetworkXException):
    """Raised when a function expects a tree (that is, a connected
    undirected graph with no cycles) but gets a non-tree graph as input
    instead.

    """


@not_implemented_for("directed")
@nx._dispatchable(graphs="T")
def to_nested_tuple(T, root, canonical_form=False):
    """Returns a nested tuple representation of the given tree.

    The nested tuple representation of a tree is defined
    recursively. The tree with one node and no edges is represented by
    the empty tuple, ``()``. A tree with ``k`` subtrees is represented
    by a tuple of length ``k`` in which each element is the nested tuple
    representation of a subtree.

    Parameters
    ----------
    T : NetworkX graph
        An undirected graph object representing a tree.

    root : node
        The node in ``T`` to interpret as the root of the tree.

    canonical_form : bool
        If ``True``, each tuple is sorted so that the function returns
        a canonical form for rooted trees. This means "lighter" subtrees
        will appear as nested tuples before "heavier" subtrees. In this
        way, each isomorphic rooted tree has the same nested tuple
        representation.

    Returns
    -------
    tuple
        A nested tuple representation of the tree.

    Notes
    -----
    This function is *not* the inverse of :func:`from_nested_tuple`; the
    only guarantee is that the rooted trees are isomorphic.

    See also
    --------
    from_nested_tuple
    to_prufer_sequence

    Examples
    --------
    The tree need not be a balanced binary tree::

        >>> T = nx.Graph()
        >>> T.add_edges_from([(0, 1), (0, 2), (0, 3)])
        >>> T.add_edges_from([(1, 4), (1, 5)])
        >>> T.add_edges_from([(3, 6), (3, 7)])
        >>> root = 0
        >>> nx.to_nested_tuple(T, root)
        (((), ()), (), ((), ()))

    Continuing the above example, if ``canonical_form`` is ``True``, the
    nested tuples will be sorted::

        >>> nx.to_nested_tuple(T, root, canonical_form=True)
        ((), ((), ()), ((), ()))

    Even the path graph can be interpreted as a tree::

        >>> T = nx.path_graph(4)
        >>> root = 0
        >>> nx.to_nested_tuple(T, root)
        ((((),),),)

    """

    def _make_tuple(T, root, _parent):
        """Recursively compute the nested tuple representation of the
        given rooted tree.

        ``_parent`` is the parent node of ``root`` in the supertree in
        which ``T`` is a subtree, or ``None`` if ``root`` is the root of
        the supertree. This argument is used to determine which
        neighbors of ``root`` are children and which is the parent.

        """
        # Get the neighbors of `root` that are not the parent node. We
        # are guaranteed that `root` is always in `T` by construction.
        children = set(T[root]) - {_parent}
        if len(children) == 0:
            return ()
        nested = (_make_tuple(T, v, root) for v in children)
        if canonical_form:
            nested = sorted(nested)
        return tuple(nested)

    # Do some sanity checks on the input.
    if not nx.is_tree(T):
        raise nx.NotATree("provided graph is not a tree")
    if root not in T:
        raise nx.NodeNotFound(f"Graph {T} contains no node {root}")

    return _make_tuple(T, root, None)


@nx._dispatchable(graphs=None, returns_graph=True)
def from_nested_tuple(sequence, sensible_relabeling=False):
    """Returns the rooted tree corresponding to the given nested tuple.

    The nested tuple representation of a tree is defined
    recursively. The tree with one node and no edges is represented by
    the empty tuple, ``()``. A tree with ``k`` subtrees is represented
    by a tuple of length ``k`` in which each element is the nested tuple
    representation of a subtree.

    Parameters
    ----------
    sequence : tuple
        A nested tuple representing a rooted tree.

    sensible_relabeling : bool
        Whether to relabel the nodes of the tree so that nodes are
        labeled in increasing order according to their breadth-first
        search order from the root node.

    Returns
    -------
    NetworkX graph
        The tree corresponding to the given nested tuple, whose root
        node is node 0. If ``sensible_labeling`` is ``True``, nodes will
        be labeled in breadth-first search order starting from the root
        node.

    Notes
    -----
    This function is *not* the inverse of :func:`to_nested_tuple`; the
    only guarantee is that the rooted trees are isomorphic.

    See also
    --------
    to_nested_tuple
    from_prufer_sequence

    Examples
    --------
    Sensible relabeling ensures that the nodes are labeled from the root
    starting at 0::

        >>> balanced = (((), ()), ((), ()))
        >>> T = nx.from_nested_tuple(balanced, sensible_relabeling=True)
        >>> edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        >>> all((u, v) in T.edges() or (v, u) in T.edges() for (u, v) in edges)
        True

    """

    def _make_tree(sequence):
        """Recursively creates a tree from the given sequence of nested
        tuples.

        This function employs the :func:`~networkx.tree.join` function
        to recursively join subtrees into a larger tree.

        """
        # The empty sequence represents the empty tree, which is the
        # (unique) graph with a single node. We mark the single node
        # with an attribute that indicates that it is the root of the
        # graph.
        if len(sequence) == 0:
            return nx.empty_graph(1)
        # For a nonempty sequence, get the subtrees for each child
        # sequence and join all the subtrees at their roots. After
        # joining the subtrees, the root is node 0.
        return nx.tree.join_trees([(_make_tree(child), 0) for child in sequence])

    # Make the tree and remove the `is_root` node attribute added by the
    # helper function.
    T = _make_tree(sequence)
    if sensible_relabeling:
        # Relabel the nodes according to their breadth-first search
        # order, starting from the root node (that is, the node 0).
        bfs_nodes = chain([0], (v for u, v in nx.bfs_edges(T, 0)))
        labels = {v: i for i, v in enumerate(bfs_nodes)}
        # We would like to use `copy=False`, but `relabel_nodes` doesn't
        # allow a relabel mapping that can't be topologically sorted.
        T = nx.relabel_nodes(T, labels)
    return T


@not_implemented_for("directed")
@nx._dispatchable(graphs="T")
def to_prufer_sequence(T):
    r"""Returns the Prüfer sequence of the given tree.

    A *Prüfer sequence* is a list of *n* - 2 numbers between 0 and
    *n* - 1, inclusive. The tree corresponding to a given Prüfer
    sequence can be recovered by repeatedly joining a node in the
    sequence with a node with the smallest potential degree according to
    the sequence.

    Parameters
    ----------
    T : NetworkX graph
        An undirected graph object representing a tree.

    Returns
    -------
    list
        The Prüfer sequence of the given tree.

    Raises
    ------
    NetworkXPointlessConcept
        If the number of nodes in `T` is less than two.

    NotATree
        If `T` is not a tree.

    KeyError
        If the set of nodes in `T` is not {0, …, *n* - 1}.

    Notes
    -----
    There is a bijection from labeled trees to Prüfer sequences. This
    function is the inverse of the :func:`from_prufer_sequence`
    function.

    Sometimes Prüfer sequences use nodes labeled from 1 to *n* instead
    of from 0 to *n* - 1. This function requires nodes to be labeled in
    the latter form. You can use :func:`~networkx.relabel_nodes` to
    relabel the nodes of your tree to the appropriate format.

    This implementation is from [1]_ and has a running time of
    $O(n)$.

    See also
    --------
    to_nested_tuple
    from_prufer_sequence

    References
    ----------
    .. [1] Wang, Xiaodong, Lei Wang, and Yingjie Wu.
           "An optimal algorithm for Prufer codes."
           *Journal of Software Engineering and Applications* 2.02 (2009): 111.
           <https://doi.org/10.4236/jsea.2009.22016>

    Examples
    --------
    There is a bijection between Prüfer sequences and labeled trees, so
    this function is the inverse of the :func:`from_prufer_sequence`
    function:

    >>> edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)]
    >>> tree = nx.Graph(edges)
    >>> sequence = nx.to_prufer_sequence(tree)
    >>> sequence
    [3, 3, 3, 4]
    >>> tree2 = nx.from_prufer_sequence(sequence)
    >>> list(tree2.edges()) == edges
    True

    """
    # Perform some sanity checks on the input.
    n = len(T)
    if n < 2:
        msg = "Prüfer sequence undefined for trees with fewer than two nodes"
        raise nx.NetworkXPointlessConcept(msg)
    if not nx.is_tree(T):
        raise nx.NotATree("provided graph is not a tree")
    if set(T) != set(range(n)):
        raise KeyError("tree must have node labels {0, ..., n - 1}")

    degree = dict(T.degree())

    def parents(u):
        return next(v for v in T[u] if degree[v] > 1)

    index = u = next(k for k in range(n) if degree[k] == 1)
    result = []
    for i in range(n - 2):
        v = parents(u)
        result.append(v)
        degree[v] -= 1
        if v < index and degree[v] == 1:
            u = v
        else:
            index = u = next(k for k in range(index + 1, n) if degree[k] == 1)
    return result


@nx._dispatchable(graphs=None, returns_graph=True)
def from_prufer_sequence(sequence):
    r"""Returns the tree corresponding to the given Prüfer sequence.

    A *Prüfer sequence* is a list of *n* - 2 numbers between 0 and
    *n* - 1, inclusive. The tree corresponding to a given Prüfer
    sequence can be recovered by repeatedly joining a node in the
    sequence with a node with the smallest potential degree according to
    the sequence.

    Parameters
    ----------
    sequence : list
        A Prüfer sequence, which is a list of *n* - 2 integers between
        zero and *n* - 1, inclusive.

    Returns
    -------
    NetworkX graph
        The tree corresponding to the given Prüfer sequence.

    Raises
    ------
    NetworkXError
        If the Prüfer sequence is not valid.

    Notes
    -----
    There is a bijection from labeled trees to Prüfer sequences. This
    function is the inverse of the :func:`from_prufer_sequence` function.

    Sometimes Prüfer sequences use nodes labeled from 1 to *n* instead
    of from 0 to *n* - 1. This function requires nodes to be labeled in
    the latter form. You can use :func:`networkx.relabel_nodes` to
    relabel the nodes of your tree to the appropriate format.

    This implementation is from [1]_ and has a running time of
    $O(n)$.

    References
    ----------
    .. [1] Wang, Xiaodong, Lei Wang, and Yingjie Wu.
           "An optimal algorithm for Prufer codes."
           *Journal of Software Engineering and Applications* 2.02 (2009): 111.
           <https://doi.org/10.4236/jsea.2009.22016>

    See also
    --------
    from_nested_tuple
    to_prufer_sequence

    Examples
    --------
    There is a bijection between Prüfer sequences and labeled trees, so
    this function is the inverse of the :func:`to_prufer_sequence`
    function:

    >>> edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)]
    >>> tree = nx.Graph(edges)
    >>> sequence = nx.to_prufer_sequence(tree)
    >>> sequence
    [3, 3, 3, 4]
    >>> tree2 = nx.from_prufer_sequence(sequence)
    >>> list(tree2.edges()) == edges
    True

    """
    n = len(sequence) + 2
    # `degree` stores the remaining degree (plus one) for each node. The
    # degree of a node in the decoded tree is one more than the number
    # of times it appears in the code.
    degree = Counter(chain(sequence, range(n)))
    T = nx.empty_graph(n)
    # `not_orphaned` is the set of nodes that have a parent in the
    # tree. After the loop, there should be exactly two nodes that are
    # not in this set.
    not_orphaned = set()
    index = u = next(k for k in range(n) if degree[k] == 1)
    for v in sequence:
        # check the validity of the prufer sequence
        if v < 0 or v > n - 1:
            raise nx.NetworkXError(
                f"Invalid Prufer sequence: Values must be between 0 and {n-1}, got {v}"
            )
        T.add_edge(u, v)
        not_orphaned.add(u)
        degree[v] -= 1
        if v < index and degree[v] == 1:
            u = v
        else:
            index = u = next(k for k in range(index + 1, n) if degree[k] == 1)
    # At this point, there must be exactly two orphaned nodes; join them.
    orphans = set(T) - not_orphaned
    u, v = orphans
    T.add_edge(u, v)
    return T
<|endoftext|>"""Operations on trees."""

from functools import partial
from itertools import accumulate, chain

import networkx as nx

__all__ = ["join_trees"]


# Argument types don't match dispatching, but allow manual selection of backend
@nx._dispatchable(graphs=None, returns_graph=True)
def join_trees(rooted_trees, *, label_attribute=None, first_label=0):
    """Returns a new rooted tree made by joining `rooted_trees`

    Constructs a new tree by joining each tree in `rooted_trees`.
    A new root node is added and connected to each of the roots
    of the input trees. While copying the nodes from the trees,
    relabeling to integers occurs. If the `label_attribute` is provided,
    the old node labels will be stored in the new tree under this attribute.

    Parameters
    ----------
    rooted_trees : list
        A list of pairs in which each left element is a NetworkX graph
        object representing a tree and each right element is the root
        node of that tree. The nodes of these trees will be relabeled to
        integers.

    label_attribute : str
        If provided, the old node labels will be stored in the new tree
        under this node attribute. If not provided, the original labels
        of the nodes in the input trees are not stored.

    first_label : int, optional (default=0)
        Specifies the label for the new root node. If provided, the root node of the joined tree
        will have this label. If not provided, the root node will default to a label of 0.

    Returns
    -------
    NetworkX graph
        The rooted tree resulting from joining the provided `rooted_trees`. The new tree has a root node
        labeled as specified by `first_label` (defaulting to 0 if not provided). Subtrees from the input
        `rooted_trees` are attached to this new root node. Each non-root node, if the `label_attribute`
        is provided, has an attribute that indicates the original label of the node in the input tree.

    Notes
    -----
    Trees are stored in NetworkX as NetworkX Graphs. There is no specific
    enforcement of the fact that these are trees. Testing for each tree
    can be done using :func:`networkx.is_tree`.

    Graph, edge, and node attributes are propagated from the given
    rooted trees to the created tree. If there are any overlapping graph
    attributes, those from later trees will overwrite those from earlier
    trees in the tuple of positional arguments.

    Examples
    --------
    Join two full balanced binary trees of height *h* to get a full
    balanced binary tree of depth *h* + 1::

        >>> h = 4
        >>> left = nx.balanced_tree(2, h)
        >>> right = nx.balanced_tree(2, h)
        >>> joined_tree = nx.join_trees([(left, 0), (right, 0)])
        >>> nx.is_isomorphic(joined_tree, nx.balanced_tree(2, h + 1))
        True

    """
    if not rooted_trees:
        return nx.empty_graph(1)

    # Unzip the zipped list of (tree, root) pairs.
    trees, roots = zip(*rooted_trees)

    # The join of the trees has the same type as the type of the first tree.
    R = type(trees[0])()

    lengths = (len(tree) for tree in trees[:-1])
    first_labels = list(accumulate(lengths, initial=first_label + 1))

    new_roots = []
    for tree, root, first_node in zip(trees, roots, first_labels):
        new_root = first_node + list(tree.nodes()).index(root)
        new_roots.append(new_root)

    # Relabel the nodes so that their union is the integers starting at first_label.
    relabel = partial(
        nx.convert_node_labels_to_integers, label_attribute=label_attribute
    )
    new_trees = [
        relabel(tree, first_label=first_label)
        for tree, first_label in zip(trees, first_labels)
    ]

    # Add all sets of nodes and edges, attributes
    for tree in new_trees:
        R.update(tree)

    # Finally, join the subtrees at the root. We know first_label is unused by the way we relabeled the subtrees.
    R.add_node(first_label)
    R.add_edges_from((first_label, root) for root in new_roots)

    return R
<|endoftext|>"""
Recognition Tests
=================

A *forest* is an acyclic, undirected graph, and a *tree* is a connected forest.
Depending on the subfield, there are various conventions for generalizing these
definitions to directed graphs.

In one convention, directed variants of forest and tree are defined in an
identical manner, except that the direction of the edges is ignored. In effect,
each directed edge is treated as a single undirected edge. Then, additional
restrictions are imposed to define *branchings* and *arborescences*.

In another convention, directed variants of forest and tree correspond to
the previous convention's branchings and arborescences, respectively. Then two
new terms, *polyforest* and *polytree*, are defined to correspond to the other
convention's forest and tree.

Summarizing::

   +-----------------------------+
   | Convention A | Convention B |
   +=============================+
   | forest       | polyforest   |
   | tree         | polytree     |
   | branching    | forest       |
   | arborescence | tree         |
   +-----------------------------+

Each convention has its reasons. The first convention emphasizes definitional
similarity in that directed forests and trees are only concerned with
acyclicity and do not have an in-degree constraint, just as their undirected
counterparts do not. The second convention emphasizes functional similarity
in the sense that the directed analog of a spanning tree is a spanning
arborescence. That is, take any spanning tree and choose one node as the root.
Then every edge is assigned a direction such there is a directed path from the
root to every other node. The result is a spanning arborescence.

NetworkX follows convention "A". Explicitly, these are:

undirected forest
   An undirected graph with no undirected cycles.

undirected tree
   A connected, undirected forest.

directed forest
   A directed graph with no undirected cycles. Equivalently, the underlying
   graph structure (which ignores edge orientations) is an undirected forest.
   In convention B, this is known as a polyforest.

directed tree
   A weakly connected, directed forest. Equivalently, the underlying graph
   structure (which ignores edge orientations) is an undirected tree. In
   convention B, this is known as a polytree.

branching
   A directed forest with each node having, at most, one parent. So the maximum
   in-degree is equal to 1. In convention B, this is known as a forest.

arborescence
   A directed tree with each node having, at most, one parent. So the maximum
   in-degree is equal to 1. In convention B, this is known as a tree.

For trees and arborescences, the adjective "spanning" may be added to designate
that the graph, when considered as a forest/branching, consists of a single
tree/arborescence that includes all nodes in the graph. It is true, by
definition, that every tree/arborescence is spanning with respect to the nodes
that define the tree/arborescence and so, it might seem redundant to introduce
the notion of "spanning". However, the nodes may represent a subset of
nodes from a larger graph, and it is in this context that the term "spanning"
becomes a useful notion.

"""

import networkx as nx

__all__ = ["is_arborescence", "is_branching", "is_forest", "is_tree"]


@nx.utils.not_implemented_for("undirected")
@nx._dispatchable
def is_arborescence(G):
    """
    Returns True if `G` is an arborescence.

    An arborescence is a directed tree with maximum in-degree equal to 1.

    Parameters
    ----------
    G : graph
        The graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is an arborescence.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (0, 2), (2, 3), (3, 4)])
    >>> nx.is_arborescence(G)
    True
    >>> G.remove_edge(0, 1)
    >>> G.add_edge(1, 2)  # maximum in-degree is 2
    >>> nx.is_arborescence(G)
    False

    Notes
    -----
    In another convention, an arborescence is known as a *tree*.

    See Also
    --------
    is_tree

    """
    return is_tree(G) and max(d for n, d in G.in_degree()) <= 1


@nx.utils.not_implemented_for("undirected")
@nx._dispatchable
def is_branching(G):
    """
    Returns True if `G` is a branching.

    A branching is a directed forest with maximum in-degree equal to 1.

    Parameters
    ----------
    G : directed graph
        The directed graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is a branching.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])
    >>> nx.is_branching(G)
    True
    >>> G.remove_edge(2, 3)
    >>> G.add_edge(3, 1)  # maximum in-degree is 2
    >>> nx.is_branching(G)
    False

    Notes
    -----
    In another convention, a branching is also known as a *forest*.

    See Also
    --------
    is_forest

    """
    return is_forest(G) and max(d for n, d in G.in_degree()) <= 1


@nx._dispatchable
def is_forest(G):
    """
    Returns True if `G` is a forest.

    A forest is a graph with no undirected cycles.

    For directed graphs, `G` is a forest if the underlying graph is a forest.
    The underlying graph is obtained by treating each directed edge as a single
    undirected edge in a multigraph.

    Parameters
    ----------
    G : graph
        The graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is a forest.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is empty.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])
    >>> nx.is_forest(G)
    True
    >>> G.add_edge(4, 1)
    >>> nx.is_forest(G)
    False

    Notes
    -----
    In another convention, a directed forest is known as a *polyforest* and
    then *forest* corresponds to a *branching*.

    See Also
    --------
    is_branching

    """
    if len(G) == 0:
        raise nx.exception.NetworkXPointlessConcept("G has no nodes.")

    if G.is_directed():
        components = (G.subgraph(c) for c in nx.weakly_connected_components(G))
    else:
        components = (G.subgraph(c) for c in nx.connected_components(G))

    return all(len(c) - 1 == c.number_of_edges() for c in components)


@nx._dispatchable
def is_tree(G):
    """
    Returns True if `G` is a tree.

    A tree is a connected graph with no undirected cycles.

    For directed graphs, `G` is a tree if the underlying graph is a tree. The
    underlying graph is obtained by treating each directed edge as a single
    undirected edge in a multigraph.

    Parameters
    ----------
    G : graph
        The graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is a tree.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is empty.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])
    >>> nx.is_tree(G)  # n-1 edges
    True
    >>> G.add_edge(3, 4)
    >>> nx.is_tree(G)  # n edges
    False

    Notes
    -----
    In another convention, a directed tree is known as a *polytree* and then
    *tree* corresponds to an *arborescence*.

    See Also
    --------
    is_arborescence

    """
    if len(G) == 0:
        raise nx.exception.NetworkXPointlessConcept("G has no nodes.")

    if G.is_directed():
        is_connected = nx.is_weakly_connected
    else:
        is_connected = nx.is_connected

    # A connected graph with no cycles has n-1 edges.
    return len(G) - 1 == G.number_of_edges() and is_connected(G)
<|endoftext|>"""
Algorithms for calculating min/max spanning trees/forests.

"""

from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from math import isnan
from operator import itemgetter
from queue import PriorityQueue

import networkx as nx
from networkx.utils import UnionFind, not_implemented_for, py_random_state

__all__ = [
    "minimum_spanning_edges",
    "maximum_spanning_edges",
    "minimum_spanning_tree",
    "maximum_spanning_tree",
    "number_of_spanning_trees",
    "random_spanning_tree",
    "partition_spanning_tree",
    "EdgePartition",
    "SpanningTreeIterator",
]


class EdgePartition(Enum):
    """
    An enum to store the state of an edge partition. The enum is written to the
    edges of a graph before being pasted to `kruskal_mst_edges`. Options are:

    - EdgePartition.OPEN
    - EdgePartition.INCLUDED
    - EdgePartition.EXCLUDED
    """

    OPEN = 0
    INCLUDED = 1
    EXCLUDED = 2


@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight", preserve_edge_attrs="data")
def boruvka_mst_edges(
    G, minimum=True, weight="weight", keys=False, data=True, ignore_nan=False
):
    """Iterate over edges of a Borůvka's algorithm min/max spanning tree.

    Parameters
    ----------
    G : NetworkX Graph
        The edges of `G` must have distinct weights,
        otherwise the edges may not form a tree.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.

    weight : string (default: 'weight')
        The name of the edge attribute holding the edge weights.

    keys : bool (default: True)
        This argument is ignored since this function is not
        implemented for multigraphs; it exists only for consistency
        with the other minimum spanning tree functions.

    data : bool (default: True)
        Flag for whether to yield edge attribute dicts.
        If True, yield edges `(u, v, d)`, where `d` is the attribute dict.
        If False, yield edges `(u, v)`.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    """
    # Initialize a forest, assuming initially that it is the discrete
    # partition of the nodes of the graph.
    forest = UnionFind(G)

    def best_edge(component):
        """Returns the optimum (minimum or maximum) edge on the edge
        boundary of the given set of nodes.

        A return value of ``None`` indicates an empty boundary.

        """
        sign = 1 if minimum else -1
        minwt = float("inf")
        boundary = None
        for e in nx.edge_boundary(G, component, data=True):
            wt = e[-1].get(weight, 1) * sign
            if isnan(wt):
                if ignore_nan:
                    continue
                msg = f"NaN found as an edge weight. Edge {e}"
                raise ValueError(msg)
            if wt < minwt:
                minwt = wt
                boundary = e
        return boundary

    # Determine the optimum edge in the edge boundary of each component
    # in the forest.
    best_edges = (best_edge(component) for component in forest.to_sets())
    best_edges = [edge for edge in best_edges if edge is not None]
    # If each entry was ``None``, that means the graph was disconnected,
    # so we are done generating the forest.
    while best_edges:
        # Determine the optimum edge in the edge boundary of each
        # component in the forest.
        #
        # This must be a sequence, not an iterator. In this list, the
        # same edge may appear twice, in different orientations (but
        # that's okay, since a union operation will be called on the
        # endpoints the first time it is seen, but not the second time).
        #
        # Any ``None`` indicates that the edge boundary for that
        # component was empty, so that part of the forest has been
        # completed.
        #
        # TODO This can be parallelized, both in the outer loop over
        # each component in the forest and in the computation of the
        # minimum. (Same goes for the identical lines outside the loop.)
        best_edges = (best_edge(component) for component in forest.to_sets())
        best_edges = [edge for edge in best_edges if edge is not None]
        # Join trees in the forest using the best edges, and yield that
        # edge, since it is part of the spanning tree.
        #
        # TODO This loop can be parallelized, to an extent (the union
        # operation must be atomic).
        for u, v, d in best_edges:
            if forest[u] != forest[v]:
                if data:
                    yield u, v, d
                else:
                    yield u, v
                forest.union(u, v)


@nx._dispatchable(
    edge_attrs={"weight": None, "partition": None}, preserve_edge_attrs="data"
)
def kruskal_mst_edges(
    G, minimum, weight="weight", keys=True, data=True, ignore_nan=False, partition=None
):
    """
    Iterate over edge of a Kruskal's algorithm min/max spanning tree.

    Parameters
    ----------
    G : NetworkX Graph
        The graph holding the tree of interest.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.

    weight : string (default: 'weight')
        The name of the edge attribute holding the edge weights.

    keys : bool (default: True)
        If `G` is a multigraph, `keys` controls whether edge keys ar yielded.
        Otherwise `keys` is ignored.

    data : bool (default: True)
        Flag for whether to yield edge attribute dicts.
        If True, yield edges `(u, v, d)`, where `d` is the attribute dict.
        If False, yield edges `(u, v)`.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    partition : string (default: None)
        The name of the edge attribute holding the partition data, if it exists.
        Partition data is written to the edges using the `EdgePartition` enum.
        If a partition exists, all included edges and none of the excluded edges
        will appear in the final tree. Open edges may or may not be used.

    Yields
    ------
    edge tuple
        The edges as discovered by Kruskal's method. Each edge can
        take the following forms: `(u, v)`, `(u, v, d)` or `(u, v, k, d)`
        depending on the `key` and `data` parameters
    """
    subtrees = UnionFind()
    if G.is_multigraph():
        edges = G.edges(keys=True, data=True)
    else:
        edges = G.edges(data=True)

    """
    Sort the edges of the graph with respect to the partition data. 
    Edges are returned in the following order:

    * Included edges
    * Open edges from smallest to largest weight
    * Excluded edges
    """
    included_edges = []
    open_edges = []
    for e in edges:
        d = e[-1]
        wt = d.get(weight, 1)
        if isnan(wt):
            if ignore_nan:
                continue
            raise ValueError(f"NaN found as an edge weight. Edge {e}")

        edge = (wt,) + e
        if d.get(partition) == EdgePartition.INCLUDED:
            included_edges.append(edge)
        elif d.get(partition) == EdgePartition.EXCLUDED:
            continue
        else:
            open_edges.append(edge)

    if minimum:
        sorted_open_edges = sorted(open_edges, key=itemgetter(0))
    else:
        sorted_open_edges = sorted(open_edges, key=itemgetter(0), reverse=True)

    # Condense the lists into one
    included_edges.extend(sorted_open_edges)
    sorted_edges = included_edges
    del open_edges, sorted_open_edges, included_edges

    # Multigraphs need to handle edge keys in addition to edge data.
    if G.is_multigraph():
        for wt, u, v, k, d in sorted_edges:
            if subtrees[u] != subtrees[v]:
                if keys:
                    if data:
                        yield u, v, k, d
                    else:
                        yield u, v, k
                else:
                    if data:
                        yield u, v, d
                    else:
                        yield u, v
                subtrees.union(u, v)
    else:
        for wt, u, v, d in sorted_edges:
            if subtrees[u] != subtrees[v]:
                if data:
                    yield u, v, d
                else:
                    yield u, v
                subtrees.union(u, v)


@nx._dispatchable(edge_attrs="weight", preserve_edge_attrs="data")
def prim_mst_edges(G, minimum, weight="weight", keys=True, data=True, ignore_nan=False):
    """Iterate over edges of Prim's algorithm min/max spanning tree.

    Parameters
    ----------
    G : NetworkX Graph
        The graph holding the tree of interest.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.

    weight : string (default: 'weight')
        The name of the edge attribute holding the edge weights.

    keys : bool (default: True)
        If `G` is a multigraph, `keys` controls whether edge keys ar yielded.
        Otherwise `keys` is ignored.

    data : bool (default: True)
        Flag for whether to yield edge attribute dicts.
        If True, yield edges `(u, v, d)`, where `d` is the attribute dict.
        If False, yield edges `(u, v)`.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    """
    is_multigraph = G.is_multigraph()
    push = heappush
    pop = heappop

    nodes = set(G)
    c = count()

    sign = 1 if minimum else -1

    while nodes:
        u = nodes.pop()
        frontier = []
        visited = {u}
        if is_multigraph:
            for v, keydict in G.adj[u].items():
                for k, d in keydict.items():
                    wt = d.get(weight, 1) * sign
                    if isnan(wt):
                        if ignore_nan:
                            continue
                        msg = f"NaN found as an edge weight. Edge {(u, v, k, d)}"
                        raise ValueError(msg)
                    push(frontier, (wt, next(c), u, v, k, d))
        else:
            for v, d in G.adj[u].items():
                wt = d.get(weight, 1) * sign
                if isnan(wt):
                    if ignore_nan:
                        continue
                    msg = f"NaN found as an edge weight. Edge {(u, v, d)}"
                    raise ValueError(msg)
                push(frontier, (wt, next(c), u, v, d))
        while nodes and frontier:
            if is_multigraph:
                W, _, u, v, k, d = pop(frontier)
            else:
                W, _, u, v, d = pop(frontier)
            if v in visited or v not in nodes:
                continue
            # Multigraphs need to handle edge keys in addition to edge data.
            if is_multigraph and keys:
                if data:
                    yield u, v, k, d
                else:
                    yield u, v, k
            else:
                if data:
                    yield u, v, d
                else:
                    yield u, v
            # update frontier
            visited.add(v)
            nodes.discard(v)
            if is_multigraph:
                for w, keydict in G.adj[v].items():
                    if w in visited:
                        continue
                    for k2, d2 in keydict.items():
                        new_weight = d2.get(weight, 1) * sign
                        if isnan(new_weight):
                            if ignore_nan:
                                continue
                            msg = f"NaN found as an edge weight. Edge {(v, w, k2, d2)}"
                            raise ValueError(msg)
                        push(frontier, (new_weight, next(c), v, w, k2, d2))
            else:
                for w, d2 in G.adj[v].items():
                    if w in visited:
                        continue
                    new_weight = d2.get(weight, 1) * sign
                    if isnan(new_weight):
                        if ignore_nan:
                            continue
                        msg = f"NaN found as an edge weight. Edge {(v, w, d2)}"
                        raise ValueError(msg)
                    push(frontier, (new_weight, next(c), v, w, d2))


ALGORITHMS = {
    "boruvka": boruvka_mst_edges,
    "borůvka": boruvka_mst_edges,
    "kruskal": kruskal_mst_edges,
    "prim": prim_mst_edges,
}


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight", preserve_edge_attrs="data")
def minimum_spanning_edges(
    G, algorithm="kruskal", weight="weight", keys=True, data=True, ignore_nan=False
):
    """Generate edges in a minimum spanning forest of an undirected
    weighted graph.

    A minimum spanning tree is a subgraph of the graph (a tree)
    with the minimum sum of edge weights.  A spanning forest is a
    union of the spanning trees for each connected component of the graph.

    Parameters
    ----------
    G : undirected Graph
       An undirected graph. If `G` is connected, then the algorithm finds a
       spanning tree. Otherwise, a spanning forest is found.

    algorithm : string
       The algorithm to use when finding a minimum spanning tree. Valid
       choices are 'kruskal', 'prim', or 'boruvka'. The default is 'kruskal'.

    weight : string
       Edge data key to use for weight (default 'weight').

    keys : bool
       Whether to yield edge key in multigraphs in addition to the edge.
       If `G` is not a multigraph, this is ignored.

    data : bool, optional
       If True yield the edge data along with the edge.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    Returns
    -------
    edges : iterator
       An iterator over edges in a maximum spanning tree of `G`.
       Edges connecting nodes `u` and `v` are represented as tuples:
       `(u, v, k, d)` or `(u, v, k)` or `(u, v, d)` or `(u, v)`

       If `G` is a multigraph, `keys` indicates whether the edge key `k` will
       be reported in the third position in the edge tuple. `data` indicates
       whether the edge datadict `d` will appear at the end of the edge tuple.

       If `G` is not a multigraph, the tuples are `(u, v, d)` if `data` is True
       or `(u, v)` if `data` is False.

    Examples
    --------
    >>> from networkx.algorithms import tree

    Find minimum spanning edges by Kruskal's algorithm

    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)
    >>> mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
    >>> edgelist = list(mst)
    >>> sorted(sorted(e) for e in edgelist)
    [[0, 1], [1, 2], [2, 3]]

    Find minimum spanning edges by Prim's algorithm

    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)
    >>> mst = tree.minimum_spanning_edges(G, algorithm="prim", data=False)
    >>> edgelist = list(mst)
    >>> sorted(sorted(e) for e in edgelist)
    [[0, 1], [1, 2], [2, 3]]

    Notes
    -----
    For Borůvka's algorithm, each edge must have a weight attribute, and
    each edge weight must be distinct.

    For the other algorithms, if the graph edges do not have a weight
    attribute a default weight of 1 will be used.

    Modified code from David Eppstein, April 2006
    http://www.ics.uci.edu/~eppstein/PADS/

    """
    try:
        algo = ALGORITHMS[algorithm]
    except KeyError as err:
        msg = f"{algorithm} is not a valid choice for an algorithm."
        raise ValueError(msg) from err

    return algo(
        G, minimum=True, weight=weight, keys=keys, data=data, ignore_nan=ignore_nan
    )


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight", preserve_edge_attrs="data")
def maximum_spanning_edges(
    G, algorithm="kruskal", weight="weight", keys=True, data=True, ignore_nan=False
):
    """Generate edges in a maximum spanning forest of an undirected
    weighted graph.

    A maximum spanning tree is a subgraph of the graph (a tree)
    with the maximum possible sum of edge weights.  A spanning forest is a
    union of the spanning trees for each connected component of the graph.

    Parameters
    ----------
    G : undirected Graph
       An undirected graph. If `G` is connected, then the algorithm finds a
       spanning tree. Otherwise, a spanning forest is found.

    algorithm : string
       The algorithm to use when finding a maximum spanning tree. Valid
       choices are 'kruskal', 'prim', or 'boruvka'. The default is 'kruskal'.

    weight : string
       Edge data key to use for weight (default 'weight').

    keys : bool
       Whether to yield edge key in multigraphs in addition to the edge.
       If `G` is not a multigraph, this is ignored.

    data : bool, optional
       If True yield the edge data along with the edge.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    Returns
    -------
    edges : iterator
       An iterator over edges in a maximum spanning tree of `G`.
       Edges connecting nodes `u` and `v` are represented as tuples:
       `(u, v, k, d)` or `(u, v, k)` or `(u, v, d)` or `(u, v)`

       If `G` is a multigraph, `keys` indicates whether the edge key `k` will
       be reported in the third position in the edge tuple. `data` indicates
       whether the edge datadict `d` will appear at the end of the edge tuple.

       If `G` is not a multigraph, the tuples are `(u, v, d)` if `data` is True
       or `(u, v)` if `data` is False.

    Examples
    --------
    >>> from networkx.algorithms import tree

    Find maximum spanning edges by Kruskal's algorithm

    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)
    >>> mst = tree.maximum_spanning_edges(G, algorithm="kruskal", data=False)
    >>> edgelist = list(mst)
    >>> sorted(sorted(e) for e in edgelist)
    [[0, 1], [0, 3], [1, 2]]

    Find maximum spanning edges by Prim's algorithm

    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)  # assign weight 2 to edge 0-3
    >>> mst = tree.maximum_spanning_edges(G, algorithm="prim", data=False)
    >>> edgelist = list(mst)
    >>> sorted(sorted(e) for e in edgelist)
    [[0, 1], [0, 3], [2, 3]]

    Notes
    -----
    For Borůvka's algorithm, each edge must have a weight attribute, and
    each edge weight must be distinct.

    For the other algorithms, if the graph edges do not have a weight
    attribute a default weight of 1 will be used.

    Modified code from David Eppstein, April 2006
    http://www.ics.uci.edu/~eppstein/PADS/
    """
    try:
        algo = ALGORITHMS[algorithm]
    except KeyError as err:
        msg = f"{algorithm} is not a valid choice for an algorithm."
        raise ValueError(msg) from err

    return algo(
        G, minimum=False, weight=weight, keys=keys, data=data, ignore_nan=ignore_nan
    )


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def minimum_spanning_tree(G, weight="weight", algorithm="kruskal", ignore_nan=False):
    """Returns a minimum spanning tree or forest on an undirected graph `G`.

    Parameters
    ----------
    G : undirected graph
        An undirected graph. If `G` is connected, then the algorithm finds a
        spanning tree. Otherwise, a spanning forest is found.

    weight : str
       Data key to use for edge weights.

    algorithm : string
       The algorithm to use when finding a minimum spanning tree. Valid
       choices are 'kruskal', 'prim', or 'boruvka'. The default is
       'kruskal'.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    Returns
    -------
    G : NetworkX Graph
       A minimum spanning tree or forest.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)
    >>> T = nx.minimum_spanning_tree(G)
    >>> sorted(T.edges(data=True))
    [(0, 1, {}), (1, 2, {}), (2, 3, {})]


    Notes
    -----
    For Borůvka's algorithm, each edge must have a weight attribute, and
    each edge weight must be distinct.

    For the other algorithms, if the graph edges do not have a weight
    attribute a default weight of 1 will be used.

    There may be more than one tree with the same minimum or maximum weight.
    See :mod:`networkx.tree.recognition` for more detailed definitions.

    Isolated nodes with self-loops are in the tree as edgeless isolated nodes.

    """
    edges = minimum_spanning_edges(
        G, algorithm, weight, keys=True, data=True, ignore_nan=ignore_nan
    )
    T = G.__class__()  # Same graph class as G
    T.graph.update(G.graph)
    T.add_nodes_from(G.nodes.items())
    T.add_edges_from(edges)
    return T


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def partition_spanning_tree(
    G, minimum=True, weight="weight", partition="partition", ignore_nan=False
):
    """
    Find a spanning tree while respecting a partition of edges.

    Edges can be flagged as either `INCLUDED` which are required to be in the
    returned tree, `EXCLUDED`, which cannot be in the returned tree and `OPEN`.

    This is used in the SpanningTreeIterator to create new partitions following
    the algorithm of Sörensen and Janssens [1]_.

    Parameters
    ----------
    G : undirected graph
        An undirected graph.

    minimum : bool (default: True)
        Determines whether the returned tree is the minimum spanning tree of
        the partition of the maximum one.

    weight : str
        Data key to use for edge weights.

    partition : str
        The key for the edge attribute containing the partition
        data on the graph. Edges can be included, excluded or open using the
        `EdgePartition` enum.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.


    Returns
    -------
    G : NetworkX Graph
        A minimum spanning tree using all of the included edges in the graph and
        none of the excluded edges.

    References
    ----------
    .. [1] G.K. Janssens, K. Sörensen, An algorithm to generate all spanning
           trees in order of increasing cost, Pesquisa Operacional, 2005-08,
           Vol. 25 (2), p. 219-229,
           https://www.scielo.br/j/pope/a/XHswBwRwJyrfL88dmMwYNWp/?lang=en
    """
    edges = kruskal_mst_edges(
        G,
        minimum,
        weight,
        keys=True,
        data=True,
        ignore_nan=ignore_nan,
        partition=partition,
    )
    T = G.__class__()  # Same graph class as G
    T.graph.update(G.graph)
    T.add_nodes_from(G.nodes.items())
    T.add_edges_from(edges)
    return T


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def maximum_spanning_tree(G, weight="weight", algorithm="kruskal", ignore_nan=False):
    """Returns a maximum spanning tree or forest on an undirected graph `G`.

    Parameters
    ----------
    G : undirected graph
        An undirected graph. If `G` is connected, then the algorithm finds a
        spanning tree. Otherwise, a spanning forest is found.

    weight : str
       Data key to use for edge weights.

    algorithm : string
       The algorithm to use when finding a maximum spanning tree. Valid
       choices are 'kruskal', 'prim', or 'boruvka'. The default is
       'kruskal'.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.


    Returns
    -------
    G : NetworkX Graph
       A maximum spanning tree or forest.


    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)
    >>> T = nx.maximum_spanning_tree(G)
    >>> sorted(T.edges(data=True))
    [(0, 1, {}), (0, 3, {'weight': 2}), (1, 2, {})]


    Notes
    -----
    For Borůvka's algorithm, each edge must have a weight attribute, and
    each edge weight must be distinct.

    For the other algorithms, if the graph edges do not have a weight
    attribute a default weight of 1 will be used.

    There may be more than one tree with the same minimum or maximum weight.
    See :mod:`networkx.tree.recognition` for more detailed definitions.

    Isolated nodes with self-loops are in the tree as edgeless isolated nodes.

    """
    edges = maximum_spanning_edges(
        G, algorithm, weight, keys=True, data=True, ignore_nan=ignore_nan
    )
    edges = list(edges)
    T = G.__class__()  # Same graph class as G
    T.graph.update(G.graph)
    T.add_nodes_from(G.nodes.items())
    T.add_edges_from(edges)
    return T


@py_random_state(3)
@nx._dispatchable(preserve_edge_attrs=True, returns_graph=True)
def random_spanning_tree(G, weight=None, *, multiplicative=True, seed=None):
    """
    Sample a random spanning tree using the edges weights of `G`.

    This function supports two different methods for determining the
    probability of the graph. If ``multiplicative=True``, the probability
    is based on the product of edge weights, and if ``multiplicative=False``
    it is based on the sum of the edge weight. However, since it is
    easier to determine the total weight of all spanning trees for the
    multiplicative version, that is significantly faster and should be used if
    possible. Additionally, setting `weight` to `None` will cause a spanning tree
    to be selected with uniform probability.

    The function uses algorithm A8 in [1]_ .

    Parameters
    ----------
    G : nx.Graph
        An undirected version of the original graph.

    weight : string
        The edge key for the edge attribute holding edge weight.

    multiplicative : bool, default=True
        If `True`, the probability of each tree is the product of its edge weight
        over the sum of the product of all the spanning trees in the graph. If
        `False`, the probability is the sum of its edge weight over the sum of
        the sum of weights for all spanning trees in the graph.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    nx.Graph
        A spanning tree using the distribution defined by the weight of the tree.

    References
    ----------
    .. [1] V. Kulkarni, Generating random combinatorial objects, Journal of
       Algorithms, 11 (1990), pp. 185–207
    """

    def find_node(merged_nodes, node):
        """
        We can think of clusters of contracted nodes as having one
        representative in the graph. Each node which is not in merged_nodes
        is still its own representative. Since a representative can be later
        contracted, we need to recursively search though the dict to find
        the final representative, but once we know it we can use path
        compression to speed up the access of the representative for next time.

        This cannot be replaced by the standard NetworkX union_find since that
        data structure will merge nodes with less representing nodes into the
        one with more representing nodes but this function requires we merge
        them using the order that contract_edges contracts using.

        Parameters
        ----------
        merged_nodes : dict
            The dict storing the mapping from node to representative
        node
            The node whose representative we seek

        Returns
        -------
        The representative of the `node`
        """
        if node not in merged_nodes:
            return node
        else:
            rep = find_node(merged_nodes, merged_nodes[node])
            merged_nodes[node] = rep
            return rep

    def prepare_graph():
        """
        For the graph `G`, remove all edges not in the set `V` and then
        contract all edges in the set `U`.

        Returns
        -------
        A copy of `G` which has had all edges not in `V` removed and all edges
        in `U` contracted.
        """

        # The result is a MultiGraph version of G so that parallel edges are
        # allowed during edge contraction
        result = nx.MultiGraph(incoming_graph_data=G)

        # Remove all edges not in V
        edges_to_remove = set(result.edges()).difference(V)
        result.remove_edges_from(edges_to_remove)

        # Contract all edges in U
        #
        # Imagine that you have two edges to contract and they share an
        # endpoint like this:
        #                        [0] ----- [1] ----- [2]
        # If we contract (0, 1) first, the contraction function will always
        # delete the second node it is passed so the resulting graph would be
        #                             [0] ----- [2]
        # and edge (1, 2) no longer exists but (0, 2) would need to be contracted
        # in its place now. That is why I use the below dict as a merge-find
        # data structure with path compression to track how the nodes are merged.
        merged_nodes = {}

        for u, v in U:
            u_rep = find_node(merged_nodes, u)
            v_rep = find_node(merged_nodes, v)
            # We cannot contract a node with itself
            if u_rep == v_rep:
                continue
            nx.contracted_nodes(result, u_rep, v_rep, self_loops=False, copy=False)
            merged_nodes[v_rep] = u_rep

        return merged_nodes, result

    def spanning_tree_total_weight(G, weight):
        """
        Find the sum of weights of the spanning trees of `G` using the
        appropriate `method`.

        This is easy if the chosen method is 'multiplicative', since we can
        use Kirchhoff's Tree Matrix Theorem directly. However, with the
        'additive' method, this process is slightly more complex and less
        computationally efficient as we have to find the number of spanning
        trees which contain each possible edge in the graph.

        Parameters
        ----------
        G : NetworkX Graph
            The graph to find the total weight of all spanning trees on.

        weight : string
            The key for the weight edge attribute of the graph.

        Returns
        -------
        float
            The sum of either the multiplicative or additive weight for all
            spanning trees in the graph.
        """
        if multiplicative:
            return nx.total_spanning_tree_weight(G, weight)
        else:
            # There are two cases for the total spanning tree additive weight.
            # 1. There is one edge in the graph. Then the only spanning tree is
            #    that edge itself, which will have a total weight of that edge
            #    itself.
            if G.number_of_edges() == 1:
                return G.edges(data=weight).__iter__().__next__()[2]
            # 2. There are no edges or two or more edges in the graph. Then, we find the
            #    total weight of the spanning trees using the formula in the
            #    reference paper: take the weight of each edge and multiply it by
            #    the number of spanning trees which include that edge. This
            #    can be accomplished by contracting the edge and finding the
            #    multiplicative total spanning tree weight if the weight of each edge
            #    is assumed to be 1, which is conveniently built into networkx already,
            #    by calling total_spanning_tree_weight with weight=None.
            #    Note that with no edges the returned value is just zero.
            else:
                total = 0
                for u, v, w in G.edges(data=weight):
                    total += w * nx.total_spanning_tree_weight(
                        nx.contracted_edge(G, edge=(u, v), self_loops=False), None
                    )
                return total

    if G.number_of_nodes() < 2:
        # no edges in the spanning tree
        return nx.empty_graph(G.nodes)

    U = set()
    st_cached_value = 0
    V = set(G.edges())
    shuffled_edges = list(G.edges())
    seed.shuffle(shuffled_edges)

    for u, v in shuffled_edges:
        e_weight = G[u][v][weight] if weight is not None else 1
        node_map, prepared_G = prepare_graph()
        G_total_tree_weight = spanning_tree_total_weight(prepared_G, weight)
        # Add the edge to U so that we can compute the total tree weight
        # assuming we include that edge
        # Now, if (u, v) cannot exist in G because it is fully contracted out
        # of existence, then it by definition cannot influence G_e's Kirchhoff
        # value. But, we also cannot pick it.
        rep_edge = (find_node(node_map, u), find_node(node_map, v))
        # Check to see if the 'representative edge' for the current edge is
        # in prepared_G. If so, then we can pick it.
        if rep_edge in prepared_G.edges:
            prepared_G_e = nx.contracted_edge(
                prepared_G, edge=rep_edge, self_loops=False
            )
            G_e_total_tree_weight = spanning_tree_total_weight(prepared_G_e, weight)
            if multiplicative:
                threshold = e_weight * G_e_total_tree_weight / G_total_tree_weight
            else:
                numerator = (
                    st_cached_value + e_weight
                ) * nx.total_spanning_tree_weight(prepared_G_e) + G_e_total_tree_weight
                denominator = (
                    st_cached_value * nx.total_spanning_tree_weight(prepared_G)
                    + G_total_tree_weight
                )
                threshold = numerator / denominator
        else:
            threshold = 0.0
        z = seed.uniform(0.0, 1.0)
        if z > threshold:
            # Remove the edge from V since we did not pick it.
            V.remove((u, v))
        else:
            # Add the edge to U since we picked it.
            st_cached_value += e_weight
            U.add((u, v))
        # If we decide to keep an edge, it may complete the spanning tree.
        if len(U) == G.number_of_nodes() - 1:
            spanning_tree = nx.Graph()
            spanning_tree.add_edges_from(U)
            return spanning_tree
    raise Exception(f"Something went wrong! Only {len(U)} edges in the spanning tree!")


class SpanningTreeIterator:
    """
    Iterate over all spanning trees of a graph in either increasing or
    decreasing cost.

    Notes
    -----
    This iterator uses the partition scheme from [1]_ (included edges,
    excluded edges and open edges) as well as a modified Kruskal's Algorithm
    to generate minimum spanning trees which respect the partition of edges.
    For spanning trees with the same weight, ties are broken arbitrarily.

    References
    ----------
    .. [1] G.K. Janssens, K. Sörensen, An algorithm to generate all spanning
           trees in order of increasing cost, Pesquisa Operacional, 2005-08,
           Vol. 25 (2), p. 219-229,
           https://www.scielo.br/j/pope/a/XHswBwRwJyrfL88dmMwYNWp/?lang=en
    """

    @dataclass(order=True)
    class Partition:
        """
        This dataclass represents a partition and stores a dict with the edge
        data and the weight of the minimum spanning tree of the partition dict.
        """

        mst_weight: float
        partition_dict: dict = field(compare=False)

        def __copy__(self):
            return SpanningTreeIterator.Partition(
                self.mst_weight, self.partition_dict.copy()
            )

    def __init__(self, G, weight="weight", minimum=True, ignore_nan=False):
        """
        Initialize the iterator

        Parameters
        ----------
        G : nx.Graph
            The directed graph which we need to iterate trees over

        weight : String, default = "weight"
            The edge attribute used to store the weight of the edge

        minimum : bool, default = True
            Return the trees in increasing order while true and decreasing order
            while false.

        ignore_nan : bool, default = False
            If a NaN is found as an edge weight normally an exception is raised.
            If `ignore_nan is True` then that edge is ignored instead.
        """
        self.G = G.copy()
        self.G.__networkx_cache__ = None  # Disable caching
        self.weight = weight
        self.minimum = minimum
        self.ignore_nan = ignore_nan
        # Randomly create a key for an edge attribute to hold the partition data
        self.partition_key = (
            "SpanningTreeIterators super secret partition attribute name"
        )

    def __iter__(self):
        """
        Returns
        -------
        SpanningTreeIterator
            The iterator object for this graph
        """
        self.partition_queue = PriorityQueue()
        self._clear_partition(self.G)
        mst_weight = partition_spanning_tree(
            self.G, self.minimum, self.weight, self.partition_key, self.ignore_nan
        ).size(weight=self.weight)

        self.partition_queue.put(
            self.Partition(mst_weight if self.minimum else -mst_weight, {})
        )

        return self

    def __next__(self):
        """
        Returns
        -------
        (multi)Graph
            The spanning tree of next greatest weight, which ties broken
            arbitrarily.
        """
        if self.partition_queue.empty():
            del self.G, self.partition_queue
            raise StopIteration

        partition = self.partition_queue.get()
        self._write_partition(partition)
        next_tree = partition_spanning_tree(
            self.G, self.minimum, self.weight, self.partition_key, self.ignore_nan
        )
        self._partition(partition, next_tree)

        self._clear_partition(next_tree)
        return next_tree

    def _partition(self, partition, partition_tree):
        """
        Create new partitions based of the minimum spanning tree of the
        current minimum partition.

        Parameters
        ----------
        partition : Partition
            The Partition instance used to generate the current minimum spanning
            tree.
        partition_tree : nx.Graph
            The minimum spanning tree of the input partition.
        """
        # create two new partitions with the data from the input partition dict
        p1 = self.Partition(0, partition.partition_dict.copy())
        p2 = self.Partition(0, partition.partition_dict.copy())
        for e in partition_tree.edges:
            # determine if the edge was open or included
            if e not in partition.partition_dict:
                # This is an open edge
                p1.partition_dict[e] = EdgePartition.EXCLUDED
                p2.partition_dict[e] = EdgePartition.INCLUDED

                self._write_partition(p1)
                p1_mst = partition_spanning_tree(
                    self.G,
                    self.minimum,
                    self.weight,
                    self.partition_key,
                    self.ignore_nan,
                )
                p1_mst_weight = p1_mst.size(weight=self.weight)
                if nx.is_connected(p1_mst):
                    p1.mst_weight = p1_mst_weight if self.minimum else -p1_mst_weight
                    self.partition_queue.put(p1.__copy__())
                p1.partition_dict = p2.partition_dict.copy()

    def _write_partition(self, partition):
        """
        Writes the desired partition into the graph to calculate the minimum
        spanning tree.

        Parameters
        ----------
        partition : Partition
            A Partition dataclass describing a partition on the edges of the
            graph.
        """

        partition_dict = partition.partition_dict
        partition_key = self.partition_key
        G = self.G

        edges = (
            G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
        )
        for *e, d in edges:
            d[partition_key] = partition_dict.get(tuple(e), EdgePartition.OPEN)

    def _clear_partition(self, G):
        """
        Removes partition data from the graph
        """
        partition_key = self.partition_key
        edges = (
            G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
        )
        for *e, d in edges:
            if partition_key in d:
                del d[partition_key]


@nx._dispatchable(edge_attrs="weight")
def number_of_spanning_trees(G, *, root=None, weight=None):
    """Returns the number of spanning trees in `G`.

    A spanning tree for an undirected graph is a tree that connects
    all nodes in the graph. For a directed graph, the analog of a
    spanning tree is called a (spanning) arborescence. The arborescence
    includes a unique directed path from the `root` node to each other node.
    The graph must be weakly connected, and the root must be a node
    that includes all nodes as successors [3]_. Note that to avoid
    discussing sink-roots and reverse-arborescences, we have reversed
    the edge orientation from [3]_ and use the in-degree laplacian.

    This function (when `weight` is `None`) returns the number of
    spanning trees for an undirected graph and the number of
    arborescences from a single root node for a directed graph.
    When `weight` is the name of an edge attribute which holds the
    weight value of each edge, the function returns the sum over
    all trees of the multiplicative weight of each tree. That is,
    the weight of the tree is the product of its edge weights.

    Kirchoff's Tree Matrix Theorem states that any cofactor of the
    Laplacian matrix of a graph is the number of spanning trees in the
    graph. (Here we use cofactors for a diagonal entry so that the
    cofactor becomes the determinant of the matrix with one row
    and its matching column removed.) For a weighted Laplacian matrix,
    the cofactor is the sum across all spanning trees of the
    multiplicative weight of each tree. That is, the weight of each
    tree is the product of its edge weights. The theorem is also
    known as Kirchhoff's theorem [1]_ and the Matrix-Tree theorem [2]_.

    For directed graphs, a similar theorem (Tutte's Theorem) holds with
    the cofactor chosen to be the one with row and column removed that
    correspond to the root. The cofactor is the number of arborescences
    with the specified node as root. And the weighted version gives the
    sum of the arborescence weights with root `root`. The arborescence
    weight is the product of its edge weights.

    Parameters
    ----------
    G : NetworkX graph

    root : node
       A node in the directed graph `G` that has all nodes as descendants.
       (This is ignored for undirected graphs.)

    weight : string or None, optional (default=None)
        The name of the edge attribute holding the edge weight.
        If `None`, then each edge is assumed to have a weight of 1.

    Returns
    -------
    Number
        Undirected graphs:
            The number of spanning trees of the graph `G`.
            Or the sum of all spanning tree weights of the graph `G`
            where the weight of a tree is the product of its edge weights.
        Directed graphs:
            The number of arborescences of `G` rooted at node `root`.
            Or the sum of all arborescence weights of the graph `G` with
            specified root where the weight of an arborescence is the product
            of its edge weights.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` does not contain any nodes.

    NetworkXError
        If the graph `G` is directed and the root node
        is not specified or is not in G.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> round(nx.number_of_spanning_trees(G))
    125

    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=2)
    >>> G.add_edge(1, 3, weight=1)
    >>> G.add_edge(2, 3, weight=1)
    >>> round(nx.number_of_spanning_trees(G, weight="weight"))
    5

    Notes
    -----
    Self-loops are excluded. Multi-edges are contracted in one edge
    equal to the sum of the weights.

    References
    ----------
    .. [1] Wikipedia
       "Kirchhoff's theorem."
       https://en.wikipedia.org/wiki/Kirchhoff%27s_theorem
    .. [2] Kirchhoff, G. R.
        Über die Auflösung der Gleichungen, auf welche man
        bei der Untersuchung der linearen Vertheilung
        Galvanischer Ströme geführt wird
        Annalen der Physik und Chemie, vol. 72, pp. 497-508, 1847.
    .. [3] Margoliash, J.
        "Matrix-Tree Theorem for Directed Graphs"
        https://www.math.uchicago.edu/~may/VIGRE/VIGRE2010/REUPapers/Margoliash.pdf
    """
    import numpy as np

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Graph G must contain at least one node.")

    # undirected G
    if not nx.is_directed(G):
        if not nx.is_connected(G):
            return 0
        G_laplacian = nx.laplacian_matrix(G, weight=weight).toarray()
        return float(np.linalg.det(G_laplacian[1:, 1:]))

    # directed G
    if root is None:
        raise nx.NetworkXError("Input `root` must be provided when G is directed")
    if root not in G:
        raise nx.NetworkXError("The node root is not in the graph G.")
    if not nx.is_weakly_connected(G):
        return 0

    # Compute directed Laplacian matrix
    nodelist = [root] + [n for n in G if n != root]
    A = nx.adjacency_matrix(G, nodelist=nodelist, weight=weight)
    D = np.diag(A.sum(axis=0))
    G_laplacian = D - A

    # Compute number of spanning trees
    return float(np.linalg.det(G_laplacian[1:, 1:]))
<|endoftext|>r"""Function for computing a junction tree of a graph."""

from itertools import combinations

import networkx as nx
from networkx.algorithms import chordal_graph_cliques, complete_to_chordal_graph, moral
from networkx.utils import not_implemented_for

__all__ = ["junction_tree"]


@not_implemented_for("multigraph")
@nx._dispatchable(returns_graph=True)
def junction_tree(G):
    r"""Returns a junction tree of a given graph.

    A junction tree (or clique tree) is constructed from a (un)directed graph G.
    The tree is constructed based on a moralized and triangulated version of G.
    The tree's nodes consist of maximal cliques and sepsets of the revised graph.
    The sepset of two cliques is the intersection of the nodes of these cliques,
    e.g. the sepset of (A,B,C) and (A,C,E,F) is (A,C). These nodes are often called
    "variables" in this literature. The tree is bipartite with each sepset
    connected to its two cliques.

    Junction Trees are not unique as the order of clique consideration determines
    which sepsets are included.

    The junction tree algorithm consists of five steps [1]_:

    1. Moralize the graph
    2. Triangulate the graph
    3. Find maximal cliques
    4. Build the tree from cliques, connecting cliques with shared
       nodes, set edge-weight to number of shared variables
    5. Find maximum spanning tree


    Parameters
    ----------
    G : networkx.Graph
        Directed or undirected graph.

    Returns
    -------
    junction_tree : networkx.Graph
        The corresponding junction tree of `G`.

    Raises
    ------
    NetworkXNotImplemented
        Raised if `G` is an instance of `MultiGraph` or `MultiDiGraph`.

    References
    ----------
    .. [1] Junction tree algorithm:
       https://en.wikipedia.org/wiki/Junction_tree_algorithm

    .. [2] Finn V. Jensen and Frank Jensen. 1994. Optimal
       junction trees. In Proceedings of the Tenth international
       conference on Uncertainty in artificial intelligence (UAI’94).
       Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 360–366.
    """

    clique_graph = nx.Graph()

    if G.is_directed():
        G = moral.moral_graph(G)
    chordal_graph, _ = complete_to_chordal_graph(G)

    cliques = [tuple(sorted(i)) for i in chordal_graph_cliques(chordal_graph)]
    clique_graph.add_nodes_from(cliques, type="clique")

    for edge in combinations(cliques, 2):
        set_edge_0 = set(edge[0])
        set_edge_1 = set(edge[1])
        if not set_edge_0.isdisjoint(set_edge_1):
            sepset = tuple(sorted(set_edge_0.intersection(set_edge_1)))
            clique_graph.add_edge(edge[0], edge[1], weight=len(sepset), sepset=sepset)

    junction_tree = nx.maximum_spanning_tree(clique_graph)

    for edge in list(junction_tree.edges(data=True)):
        junction_tree.add_node(edge[2]["sepset"], type="sepset")
        junction_tree.add_edge(edge[0], edge[2]["sepset"])
        junction_tree.add_edge(edge[1], edge[2]["sepset"])
        junction_tree.remove_edge(edge[0], edge[1])

    return junction_tree
<|endoftext|>"""
Algorithms for finding optimum branchings and spanning arborescences.

This implementation is based on:

    J. Edmonds, Optimum branchings, J. Res. Natl. Bur. Standards 71B (1967),
    233–240. URL: http://archive.org/details/jresv71Bn4p233

"""

# TODO: Implement method from Gabow, Galil, Spence and Tarjan:
#
# @article{
#    year={1986},
#    issn={0209-9683},
#    journal={Combinatorica},
#    volume={6},
#    number={2},
#    doi={10.1007/BF02579168},
#    title={Efficient algorithms for finding minimum spanning trees in
#        undirected and directed graphs},
#    url={https://doi.org/10.1007/BF02579168},
#    publisher={Springer-Verlag},
#    keywords={68 B 15; 68 C 05},
#    author={Gabow, Harold N. and Galil, Zvi and Spencer, Thomas and Tarjan,
#        Robert E.},
#    pages={109-122},
#    language={English}
# }
import string
from dataclasses import dataclass, field
from operator import itemgetter
from queue import PriorityQueue

import networkx as nx
from networkx.utils import py_random_state

from .recognition import is_arborescence, is_branching

__all__ = [
    "branching_weight",
    "greedy_branching",
    "maximum_branching",
    "minimum_branching",
    "minimal_branching",
    "maximum_spanning_arborescence",
    "minimum_spanning_arborescence",
    "ArborescenceIterator",
]

KINDS = {"max", "min"}

STYLES = {
    "branching": "branching",
    "arborescence": "arborescence",
    "spanning arborescence": "arborescence",
}

INF = float("inf")


@py_random_state(1)
def random_string(L=15, seed=None):
    return "".join([seed.choice(string.ascii_letters) for n in range(L)])


def _min_weight(weight):
    return -weight


def _max_weight(weight):
    return weight


@nx._dispatchable(edge_attrs={"attr": "default"})
def branching_weight(G, attr="weight", default=1):
    """
    Returns the total weight of a branching.

    You must access this function through the networkx.algorithms.tree module.

    Parameters
    ----------
    G : DiGraph
        The directed graph.
    attr : str
        The attribute to use as weights. If None, then each edge will be
        treated equally with a weight of 1.
    default : float
        When `attr` is not None, then if an edge does not have that attribute,
        `default` specifies what value it should take.

    Returns
    -------
    weight: int or float
        The total weight of the branching.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from([(0, 1, 2), (1, 2, 4), (2, 3, 3), (3, 4, 2)])
    >>> nx.tree.branching_weight(G)
    11

    """
    return sum(edge[2].get(attr, default) for edge in G.edges(data=True))


@py_random_state(4)
@nx._dispatchable(edge_attrs={"attr": "default"}, returns_graph=True)
def greedy_branching(G, attr="weight", default=1, kind="max", seed=None):
    """
    Returns a branching obtained through a greedy algorithm.

    This algorithm is wrong, and cannot give a proper optimal branching.
    However, we include it for pedagogical reasons, as it can be helpful to
    see what its outputs are.

    The output is a branching, and possibly, a spanning arborescence. However,
    it is not guaranteed to be optimal in either case.

    Parameters
    ----------
    G : DiGraph
        The directed graph to scan.
    attr : str
        The attribute to use as weights. If None, then each edge will be
        treated equally with a weight of 1.
    default : float
        When `attr` is not None, then if an edge does not have that attribute,
        `default` specifies what value it should take.
    kind : str
        The type of optimum to search for: 'min' or 'max' greedy branching.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    B : directed graph
        The greedily obtained branching.

    """
    if kind not in KINDS:
        raise nx.NetworkXException("Unknown value for `kind`.")

    if kind == "min":
        reverse = False
    else:
        reverse = True

    if attr is None:
        # Generate a random string the graph probably won't have.
        attr = random_string(seed=seed)

    edges = [(u, v, data.get(attr, default)) for (u, v, data) in G.edges(data=True)]

    # We sort by weight, but also by nodes to normalize behavior across runs.
    try:
        edges.sort(key=itemgetter(2, 0, 1), reverse=reverse)
    except TypeError:
        # This will fail in Python 3.x if the nodes are of varying types.
        # In that case, we use the arbitrary order.
        edges.sort(key=itemgetter(2), reverse=reverse)

    # The branching begins with a forest of no edges.
    B = nx.DiGraph()
    B.add_nodes_from(G)

    # Now we add edges greedily so long we maintain the branching.
    uf = nx.utils.UnionFind()
    for i, (u, v, w) in enumerate(edges):
        if uf[u] == uf[v]:
            # Adding this edge would form a directed cycle.
            continue
        elif B.in_degree(v) == 1:
            # The edge would increase the degree to be greater than one.
            continue
        else:
            # If attr was None, then don't insert weights...
            data = {}
            if attr is not None:
                data[attr] = w
            B.add_edge(u, v, **data)
            uf.union(u, v)

    return B


@nx._dispatchable(preserve_edge_attrs=True, returns_graph=True)
def maximum_branching(
    G,
    attr="weight",
    default=1,
    preserve_attrs=False,
    partition=None,
):
    #######################################
    ### Data Structure Helper Functions ###
    #######################################

    def edmonds_add_edge(G, edge_index, u, v, key, **d):
        """
        Adds an edge to `G` while also updating the edge index.

        This algorithm requires the use of an external dictionary to track
        the edge keys since it is possible that the source or destination
        node of an edge will be changed and the default key-handling
        capabilities of the MultiDiGraph class do not account for this.

        Parameters
        ----------
        G : MultiDiGraph
            The graph to insert an edge into.
        edge_index : dict
            A mapping from integers to the edges of the graph.
        u : node
            The source node of the new edge.
        v : node
            The destination node of the new edge.
        key : int
            The key to use from `edge_index`.
        d : keyword arguments, optional
            Other attributes to store on the new edge.
        """

        if key in edge_index:
            uu, vv, _ = edge_index[key]
            if (u != uu) or (v != vv):
                raise Exception(f"Key {key!r} is already in use.")

        G.add_edge(u, v, key, **d)
        edge_index[key] = (u, v, G.succ[u][v][key])

    def edmonds_remove_node(G, edge_index, n):
        """
        Remove a node from the graph, updating the edge index to match.

        Parameters
        ----------
        G : MultiDiGraph
            The graph to remove an edge from.
        edge_index : dict
            A mapping from integers to the edges of the graph.
        n : node
            The node to remove from `G`.
        """
        keys = set()
        for keydict in G.pred[n].values():
            keys.update(keydict)
        for keydict in G.succ[n].values():
            keys.update(keydict)

        for key in keys:
            del edge_index[key]

        G.remove_node(n)

    #######################
    ### Algorithm Setup ###
    #######################

    # Pick an attribute name that the original graph is unlikly to have
    candidate_attr = "edmonds' secret candidate attribute"
    new_node_base_name = "edmonds new node base name "

    G_original = G
    G = nx.MultiDiGraph()
    G.__networkx_cache__ = None  # Disable caching

    # A dict to reliably track mutations to the edges using the key of the edge.
    G_edge_index = {}
    # Each edge is given an arbitrary numerical key
    for key, (u, v, data) in enumerate(G_original.edges(data=True)):
        d = {attr: data.get(attr, default)}

        if data.get(partition) is not None:
            d[partition] = data.get(partition)

        if preserve_attrs:
            for d_k, d_v in data.items():
                if d_k != attr:
                    d[d_k] = d_v

        edmonds_add_edge(G, G_edge_index, u, v, key, **d)

    level = 0  # Stores the number of contracted nodes

    # These are the buckets from the paper.
    #
    # In the paper, G^i are modified versions of the original graph.
    # D^i and E^i are the nodes and edges of the maximal edges that are
    # consistent with G^i. In this implementation, D^i and E^i are stored
    # together as the graph B^i. We will have strictly more B^i then the
    # paper will have.
    #
    # Note that the data in graphs and branchings are tuples with the graph as
    # the first element and the edge index as the second.
    B = nx.MultiDiGraph()
    B_edge_index = {}
    graphs = []  # G^i list
    branchings = []  # B^i list
    selected_nodes = set()  # D^i bucket
    uf = nx.utils.UnionFind()

    # A list of lists of edge indices. Each list is a circuit for graph G^i.
    # Note the edge list is not required to be a circuit in G^0.
    circuits = []

    # Stores the index of the minimum edge in the circuit found in G^i and B^i.
    # The ordering of the edges seems to preserver the weight ordering from
    # G^0. So even if the circuit does not form a circuit in G^0, it is still
    # true that the minimum edges in circuit G^0 (despite their weights being
    # different)
    minedge_circuit = []

    ###########################
    ### Algorithm Structure ###
    ###########################

    # Each step listed in the algorithm is an inner function. Thus, the overall
    # loop structure is:
    #
    # while True:
    #     step_I1()
    #     if cycle detected:
    #         step_I2()
    #     elif every node of G is in D and E is a branching:
    #         break

    ##################################
    ### Algorithm Helper Functions ###
    ##################################

    def edmonds_find_desired_edge(v):
        """
        Find the edge directed towards v with maximal weight.

        If an edge partition exists in this graph, return the included
        edge if it exists and never return any excluded edge.

        Note: There can only be one included edge for each vertex otherwise
        the edge partition is empty.

        Parameters
        ----------
        v : node
            The node to search for the maximal weight incoming edge.
        """
        edge = None
        max_weight = -INF
        for u, _, key, data in G.in_edges(v, data=True, keys=True):
            # Skip excluded edges
            if data.get(partition) == nx.EdgePartition.EXCLUDED:
                continue

            new_weight = data[attr]

            # Return the included edge
            if data.get(partition) == nx.EdgePartition.INCLUDED:
                max_weight = new_weight
                edge = (u, v, key, new_weight, data)
                break

            # Find the best open edge
            if new_weight > max_weight:
                max_weight = new_weight
                edge = (u, v, key, new_weight, data)

        return edge, max_weight

    def edmonds_step_I2(v, desired_edge, level):
        """
        Perform step I2 from Edmonds' paper

        First, check if the last step I1 created a cycle. If it did not, do nothing.
        If it did, store the cycle for later reference and contract it.

        Parameters
        ----------
        v : node
            The current node to consider
        desired_edge : edge
            The minimum desired edge to remove from the cycle.
        level : int
            The current level, i.e. the number of cycles that have already been removed.
        """
        u = desired_edge[0]

        Q_nodes = nx.shortest_path(B, v, u)
        Q_edges = [
            list(B[Q_nodes[i]][vv].keys())[0] for i, vv in enumerate(Q_nodes[1:])
        ]
        Q_edges.append(desired_edge[2])  # Add the new edge key to complete the circuit

        # Get the edge in the circuit with the minimum weight.
        # Also, save the incoming weights for each node.
        minweight = INF
        minedge = None
        Q_incoming_weight = {}
        for edge_key in Q_edges:
            u, v, data = B_edge_index[edge_key]
            w = data[attr]
            # We cannot remove an included edge, even if it is the
            # minimum edge in the circuit
            Q_incoming_weight[v] = w
            if data.get(partition) == nx.EdgePartition.INCLUDED:
                continue
            if w < minweight:
                minweight = w
                minedge = edge_key

        circuits.append(Q_edges)
        minedge_circuit.append(minedge)
        graphs.append((G.copy(), G_edge_index.copy()))
        branchings.append((B.copy(), B_edge_index.copy()))

        # Mutate the graph to contract the circuit
        new_node = new_node_base_name + str(level)
        G.add_node(new_node)
        new_edges = []
        for u, v, key, data in G.edges(data=True, keys=True):
            if u in Q_incoming_weight:
                if v in Q_incoming_weight:
                    # Circuit edge. For the moment do nothing,
                    # eventually it will be removed.
                    continue
                else:
                    # Outgoing edge from a node in the circuit.
                    # Make it come from the new node instead
                    dd = data.copy()
                    new_edges.append((new_node, v, key, dd))
            else:
                if v in Q_incoming_weight:
                    # Incoming edge to the circuit.
                    # Update it's weight
                    w = data[attr]
                    w += minweight - Q_incoming_weight[v]
                    dd = data.copy()
                    dd[attr] = w
                    new_edges.append((u, new_node, key, dd))
                else:
                    # Outside edge. No modification needed
                    continue

        for node in Q_nodes:
            edmonds_remove_node(G, G_edge_index, node)
            edmonds_remove_node(B, B_edge_index, node)

        selected_nodes.difference_update(set(Q_nodes))

        for u, v, key, data in new_edges:
            edmonds_add_edge(G, G_edge_index, u, v, key, **data)
            if candidate_attr in data:
                del data[candidate_attr]
                edmonds_add_edge(B, B_edge_index, u, v, key, **data)
                uf.union(u, v)

    def is_root(G, u, edgekeys):
        """
        Returns True if `u` is a root node in G.

        Node `u` is a root node if its in-degree over the specified edges is zero.

        Parameters
        ----------
        G : Graph
            The current graph.
        u : node
            The node in `G` to check if it is a root.
        edgekeys : iterable of edges
            The edges for which to check if `u` is a root of.
        """
        if u not in G:
            raise Exception(f"{u!r} not in G")

        for v in G.pred[u]:
            for edgekey in G.pred[u][v]:
                if edgekey in edgekeys:
                    return False, edgekey
        else:
            return True, None

    nodes = iter(list(G.nodes))
    while True:
        try:
            v = next(nodes)
        except StopIteration:
            # If there are no more new nodes to consider, then we should
            # meet stopping condition (b) from the paper:
            #   (b) every node of G^i is in D^i and E^i is a branching
            assert len(G) == len(B)
            if len(B):
                assert is_branching(B)

            graphs.append((G.copy(), G_edge_index.copy()))
            branchings.append((B.copy(), B_edge_index.copy()))
            circuits.append([])
            minedge_circuit.append(None)

            break
        else:
            #####################
            ### BEGIN STEP I1 ###
            #####################

            # This is a very simple step, so I don't think it needs a method of it's own
            if v in selected_nodes:
                continue

        selected_nodes.add(v)
        B.add_node(v)
        desired_edge, desired_edge_weight = edmonds_find_desired_edge(v)

        # There might be no desired edge if all edges are excluded or
        # v is the last node to be added to B, the ultimate root of the branching
        if desired_edge is not None and desired_edge_weight > 0:
            u = desired_edge[0]
            # Flag adding the edge will create a circuit before merging the two
            # connected components of u and v in B
            circuit = uf[u] == uf[v]
            dd = {attr: desired_edge_weight}
            if desired_edge[4].get(partition) is not None:
                dd[partition] = desired_edge[4].get(partition)

            edmonds_add_edge(B, B_edge_index, u, v, desired_edge[2], **dd)
            G[u][v][desired_edge[2]][candidate_attr] = True
            uf.union(u, v)

            ###################
            ### END STEP I1 ###
            ###################

            #####################
            ### BEGIN STEP I2 ###
            #####################

            if circuit:
                edmonds_step_I2(v, desired_edge, level)
                nodes = iter(list(G.nodes()))
                level += 1

            ###################
            ### END STEP I2 ###
            ###################

    #####################
    ### BEGIN STEP I3 ###
    #####################

    # Create a new graph of the same class as the input graph
    H = G_original.__class__()

    # Start with the branching edges in the last level.
    edges = set(branchings[level][1])
    while level > 0:
        level -= 1

        # The current level is i, and we start counting from 0.
        #
        # We need the node at level i+1 that results from merging a circuit
        # at level i. basename_0 is the first merged node and this happens
        # at level 1. That is basename_0 is a node at level 1 that results
        # from merging a circuit at level 0.

        merged_node = new_node_base_name + str(level)
        circuit = circuits[level]
        isroot, edgekey = is_root(graphs[level + 1][0], merged_node, edges)
        edges.update(circuit)

        if isroot:
            minedge = minedge_circuit[level]
            if minedge is None:
                raise Exception

            # Remove the edge in the cycle with minimum weight
            edges.remove(minedge)
        else:
            # We have identified an edge at the next higher level that
            # transitions into the merged node at this level. That edge
            # transitions to some corresponding node at the current level.
            #
            # We want to remove an edge from the cycle that transitions
            # into the corresponding node, otherwise the result would not
            # be a branching.

            G, G_edge_index = graphs[level]
            target = G_edge_index[edgekey][1]
            for edgekey in circuit:
                u, v, data = G_edge_index[edgekey]
                if v == target:
                    break
            else:
                raise Exception("Couldn't find edge incoming to merged node.")

            edges.remove(edgekey)

    H.add_nodes_from(G_original)
    for edgekey in edges:
        u, v, d = graphs[0][1][edgekey]
        dd = {attr: d[attr]}

        if preserve_attrs:
            for key, value in d.items():
                if key not in [attr, candidate_attr]:
                    dd[key] = value

        H.add_edge(u, v, **dd)

    ###################
    ### END STEP I3 ###
    ###################

    return H


@nx._dispatchable(preserve_edge_attrs=True, mutates_input=True, returns_graph=True)
def minimum_branching(
    G, attr="weight", default=1, preserve_attrs=False, partition=None
):
    for _, _, d in G.edges(data=True):
        d[attr] = -d.get(attr, default)
    nx._clear_cache(G)

    B = maximum_branching(G, attr, default, preserve_attrs, partition)

    for _, _, d in G.edges(data=True):
        d[attr] = -d.get(attr, default)
    nx._clear_cache(G)

    for _, _, d in B.edges(data=True):
        d[attr] = -d.get(attr, default)
    nx._clear_cache(B)

    return B


@nx._dispatchable(preserve_edge_attrs=True, mutates_input=True, returns_graph=True)
def minimal_branching(
    G, /, *, attr="weight", default=1, preserve_attrs=False, partition=None
):
    """
    Returns a minimal branching from `G`.

    A minimal branching is a branching similar to a minimal arborescence but
    without the requirement that the result is actually a spanning arborescence.
    This allows minimal branchinges to be computed over graphs which may not
    have arborescence (such as multiple components).

    Parameters
    ----------
    G : (multi)digraph-like
        The graph to be searched.
    attr : str
        The edge attribute used in determining optimality.
    default : float
        The value of the edge attribute used if an edge does not have
        the attribute `attr`.
    preserve_attrs : bool
        If True, preserve the other attributes of the original graph (that are not
        passed to `attr`)
    partition : str
        The key for the edge attribute containing the partition
        data on the graph. Edges can be included, excluded or open using the
        `EdgePartition` enum.

    Returns
    -------
    B : (multi)digraph-like
        A minimal branching.
    """
    max_weight = -INF
    min_weight = INF
    for _, _, w in G.edges(data=attr, default=default):
        if w > max_weight:
            max_weight = w
        if w < min_weight:
            min_weight = w

    for _, _, d in G.edges(data=True):
        # Transform the weights so that the minimum weight is larger than
        # the difference between the max and min weights. This is important
        # in order to prevent the edge weights from becoming negative during
        # computation
        d[attr] = max_weight + 1 + (max_weight - min_weight) - d.get(attr, default)
    nx._clear_cache(G)

    B = maximum_branching(G, attr, default, preserve_attrs, partition)

    # Reverse the weight transformations
    for _, _, d in G.edges(data=True):
        d[attr] = max_weight + 1 + (max_weight - min_weight) - d.get(attr, default)
    nx._clear_cache(G)

    for _, _, d in B.edges(data=True):
        d[attr] = max_weight + 1 + (max_weight - min_weight) - d.get(attr, default)
    nx._clear_cache(B)

    return B


@nx._dispatchable(preserve_edge_attrs=True, mutates_input=True, returns_graph=True)
def maximum_spanning_arborescence(
    G, attr="weight", default=1, preserve_attrs=False, partition=None
):
    # In order to use the same algorithm is the maximum branching, we need to adjust
    # the weights of the graph. The branching algorithm can choose to not include an
    # edge if it doesn't help find a branching, mainly triggered by edges with negative
    # weights.
    #
    # To prevent this from happening while trying to find a spanning arborescence, we
    # just have to tweak the edge weights so that they are all positive and cannot
    # become negative during the branching algorithm, find the maximum branching and
    # then return them to their original values.

    min_weight = INF
    max_weight = -INF
    for _, _, w in G.edges(data=attr, default=default):
        if w < min_weight:
            min_weight = w
        if w > max_weight:
            max_weight = w

    for _, _, d in G.edges(data=True):
        d[attr] = d.get(attr, default) - min_weight + 1 - (min_weight - max_weight)
    nx._clear_cache(G)

    B = maximum_branching(G, attr, default, preserve_attrs, partition)

    for _, _, d in G.edges(data=True):
        d[attr] = d.get(attr, default) + min_weight - 1 + (min_weight - max_weight)
    nx._clear_cache(G)

    for _, _, d in B.edges(data=True):
        d[attr] = d.get(attr, default) + min_weight - 1 + (min_weight - max_weight)
    nx._clear_cache(B)

    if not is_arborescence(B):
        raise nx.exception.NetworkXException("No maximum spanning arborescence in G.")

    return B


@nx._dispatchable(preserve_edge_attrs=True, mutates_input=True, returns_graph=True)
def minimum_spanning_arborescence(
    G, attr="weight", default=1, preserve_attrs=False, partition=None
):
    B = minimal_branching(
        G,
        attr=attr,
        default=default,
        preserve_attrs=preserve_attrs,
        partition=partition,
    )

    if not is_arborescence(B):
        raise nx.exception.NetworkXException("No minimum spanning arborescence in G.")

    return B


docstring_branching = """
Returns a {kind} {style} from G.

Parameters
----------
G : (multi)digraph-like
    The graph to be searched.
attr : str
    The edge attribute used to in determining optimality.
default : float
    The value of the edge attribute used if an edge does not have
    the attribute `attr`.
preserve_attrs : bool
    If True, preserve the other attributes of the original graph (that are not
    passed to `attr`)
partition : str
    The key for the edge attribute containing the partition
    data on the graph. Edges can be included, excluded or open using the
    `EdgePartition` enum.

Returns
-------
B : (multi)digraph-like
    A {kind} {style}.
"""

docstring_arborescence = (
    docstring_branching
    + """
Raises
------
NetworkXException
    If the graph does not contain a {kind} {style}.

"""
)

maximum_branching.__doc__ = docstring_branching.format(
    kind="maximum", style="branching"
)

minimum_branching.__doc__ = (
    docstring_branching.format(kind="minimum", style="branching")
    + """
See Also
--------
    minimal_branching
"""
)

maximum_spanning_arborescence.__doc__ = docstring_arborescence.format(
    kind="maximum", style="spanning arborescence"
)

minimum_spanning_arborescence.__doc__ = docstring_arborescence.format(
    kind="minimum", style="spanning arborescence"
)


class ArborescenceIterator:
    """
    Iterate over all spanning arborescences of a graph in either increasing or
    decreasing cost.

    Notes
    -----
    This iterator uses the partition scheme from [1]_ (included edges,
    excluded edges and open edges). It generates minimum spanning
    arborescences using a modified Edmonds' Algorithm which respects the
    partition of edges. For arborescences with the same weight, ties are
    broken arbitrarily.

    References
    ----------
    .. [1] G.K. Janssens, K. Sörensen, An algorithm to generate all spanning
           trees in order of increasing cost, Pesquisa Operacional, 2005-08,
           Vol. 25 (2), p. 219-229,
           https://www.scielo.br/j/pope/a/XHswBwRwJyrfL88dmMwYNWp/?lang=en
    """

    @dataclass(order=True)
    class Partition:
        """
        This dataclass represents a partition and stores a dict with the edge
        data and the weight of the minimum spanning arborescence of the
        partition dict.
        """

        mst_weight: float
        partition_dict: dict = field(compare=False)

        def __copy__(self):
            return ArborescenceIterator.Partition(
                self.mst_weight, self.partition_dict.copy()
            )

    def __init__(self, G, weight="weight", minimum=True, init_partition=None):
        """
        Initialize the iterator

        Parameters
        ----------
        G : nx.DiGraph
            The directed graph which we need to iterate trees over

        weight : String, default = "weight"
            The edge attribute used to store the weight of the edge

        minimum : bool, default = True
            Return the trees in increasing order while true and decreasing order
            while false.

        init_partition : tuple, default = None
            In the case that certain edges have to be included or excluded from
            the arborescences, `init_partition` should be in the form
            `(included_edges, excluded_edges)` where each edges is a
            `(u, v)`-tuple inside an iterable such as a list or set.

        """
        self.G = G.copy()
        self.weight = weight
        self.minimum = minimum
        self.method = (
            minimum_spanning_arborescence if minimum else maximum_spanning_arborescence
        )
        # Randomly create a key for an edge attribute to hold the partition data
        self.partition_key = (
            "ArborescenceIterators super secret partition attribute name"
        )
        if init_partition is not None:
            partition_dict = {}
            for e in init_partition[0]:
                partition_dict[e] = nx.EdgePartition.INCLUDED
            for e in init_partition[1]:
                partition_dict[e] = nx.EdgePartition.EXCLUDED
            self.init_partition = ArborescenceIterator.Partition(0, partition_dict)
        else:
            self.init_partition = None

    def __iter__(self):
        """
        Returns
        -------
        ArborescenceIterator
            The iterator object for this graph
        """
        self.partition_queue = PriorityQueue()
        self._clear_partition(self.G)

        # Write the initial partition if it exists.
        if self.init_partition is not None:
            self._write_partition(self.init_partition)

        mst_weight = self.method(
            self.G,
            self.weight,
            partition=self.partition_key,
            preserve_attrs=True,
        ).size(weight=self.weight)

        self.partition_queue.put(
            self.Partition(
                mst_weight if self.minimum else -mst_weight,
                (
                    {}
                    if self.init_partition is None
                    else self.init_partition.partition_dict
                ),
            )
        )

        return self

    def __next__(self):
        """
        Returns
        -------
        (multi)Graph
            The spanning tree of next greatest weight, which ties broken
            arbitrarily.
        """
        if self.partition_queue.empty():
            del self.G, self.partition_queue
            raise StopIteration

        partition = self.partition_queue.get()
        self._write_partition(partition)
        next_arborescence = self.method(
            self.G,
            self.weight,
            partition=self.partition_key,
            preserve_attrs=True,
        )
        self._partition(partition, next_arborescence)

        self._clear_partition(next_arborescence)
        return next_arborescence

    def _partition(self, partition, partition_arborescence):
        """
        Create new partitions based of the minimum spanning tree of the
        current minimum partition.

        Parameters
        ----------
        partition : Partition
            The Partition instance used to generate the current minimum spanning
            tree.
        partition_arborescence : nx.Graph
            The minimum spanning arborescence of the input partition.
        """
        # create two new partitions with the data from the input partition dict
        p1 = self.Partition(0, partition.partition_dict.copy())
        p2 = self.Partition(0, partition.partition_dict.copy())
        for e in partition_arborescence.edges:
            # determine if the edge was open or included
            if e not in partition.partition_dict:
                # This is an open edge
                p1.partition_dict[e] = nx.EdgePartition.EXCLUDED
                p2.partition_dict[e] = nx.EdgePartition.INCLUDED

                self._write_partition(p1)
                try:
                    p1_mst = self.method(
                        self.G,
                        self.weight,
                        partition=self.partition_key,
                        preserve_attrs=True,
                    )

                    p1_mst_weight = p1_mst.size(weight=self.weight)
                    p1.mst_weight = p1_mst_weight if self.minimum else -p1_mst_weight
                    self.partition_queue.put(p1.__copy__())
                except nx.NetworkXException:
                    pass

                p1.partition_dict = p2.partition_dict.copy()

    def _write_partition(self, partition):
        """
        Writes the desired partition into the graph to calculate the minimum
        spanning tree. Also, if one incoming edge is included, mark all others
        as excluded so that if that vertex is merged during Edmonds' algorithm
        we cannot still pick another of that vertex's included edges.

        Parameters
        ----------
        partition : Partition
            A Partition dataclass describing a partition on the edges of the
            graph.
        """
        for u, v, d in self.G.edges(data=True):
            if (u, v) in partition.partition_dict:
                d[self.partition_key] = partition.partition_dict[(u, v)]
            else:
                d[self.partition_key] = nx.EdgePartition.OPEN
        nx._clear_cache(self.G)

        for n in self.G:
            included_count = 0
            excluded_count = 0
            for u, v, d in self.G.in_edges(nbunch=n, data=True):
                if d.get(self.partition_key) == nx.EdgePartition.INCLUDED:
                    included_count += 1
                elif d.get(self.partition_key) == nx.EdgePartition.EXCLUDED:
                    excluded_count += 1
            # Check that if there is an included edges, all other incoming ones
            # are excluded. If not fix it!
            if included_count == 1 and excluded_count != self.G.in_degree(n) - 1:
                for u, v, d in self.G.in_edges(nbunch=n, data=True):
                    if d.get(self.partition_key) != nx.EdgePartition.INCLUDED:
                        d[self.partition_key] = nx.EdgePartition.EXCLUDED

    def _clear_partition(self, G):
        """
        Removes partition data from the graph
        """
        for u, v, d in G.edges(data=True):
            if self.partition_key in d:
                del d[self.partition_key]
        nx._clear_cache(self.G)
<|endoftext|>"""Function for detecting communities based on Louvain Community Detection
Algorithm"""

import itertools
from collections import defaultdict, deque

import networkx as nx
from networkx.algorithms.community import modularity
from networkx.utils import py_random_state

__all__ = ["louvain_communities", "louvain_partitions"]


@py_random_state("seed")
@nx._dispatchable(edge_attrs="weight")
def louvain_communities(
    G, weight="weight", resolution=1, threshold=0.0000001, max_level=None, seed=None
):
    r"""Find the best partition of a graph using the Louvain Community Detection
    Algorithm.

    Louvain Community Detection Algorithm is a simple method to extract the community
    structure of a network. This is a heuristic method based on modularity optimization. [1]_

    The algorithm works in 2 steps. On the first step it assigns every node to be
    in its own community and then for each node it tries to find the maximum positive
    modularity gain by moving each node to all of its neighbor communities. If no positive
    gain is achieved the node remains in its original community.

    The modularity gain obtained by moving an isolated node $i$ into a community $C$ can
    easily be calculated by the following formula (combining [1]_ [2]_ and some algebra):

    .. math::
        \Delta Q = \frac{k_{i,in}}{2m} - \gamma\frac{ \Sigma_{tot} \cdot k_i}{2m^2}

    where $m$ is the size of the graph, $k_{i,in}$ is the sum of the weights of the links
    from $i$ to nodes in $C$, $k_i$ is the sum of the weights of the links incident to node $i$,
    $\Sigma_{tot}$ is the sum of the weights of the links incident to nodes in $C$ and $\gamma$
    is the resolution parameter.

    For the directed case the modularity gain can be computed using this formula according to [3]_

    .. math::
        \Delta Q = \frac{k_{i,in}}{m}
        - \gamma\frac{k_i^{out} \cdot\Sigma_{tot}^{in} + k_i^{in} \cdot \Sigma_{tot}^{out}}{m^2}

    where $k_i^{out}$, $k_i^{in}$ are the outer and inner weighted degrees of node $i$ and
    $\Sigma_{tot}^{in}$, $\Sigma_{tot}^{out}$ are the sum of in-going and out-going links incident
    to nodes in $C$.

    The first phase continues until no individual move can improve the modularity.

    The second phase consists in building a new network whose nodes are now the communities
    found in the first phase. To do so, the weights of the links between the new nodes are given by
    the sum of the weight of the links between nodes in the corresponding two communities. Once this
    phase is complete it is possible to reapply the first phase creating bigger communities with
    increased modularity.

    The above two phases are executed until no modularity gain is achieved (or is less than
    the `threshold`, or until `max_levels` is reached).

    Be careful with self-loops in the input graph. These are treated as
    previously reduced communities -- as if the process had been started
    in the middle of the algorithm. Large self-loop edge weights thus
    represent strong communities and in practice may be hard to add
    other nodes to.  If your input graph edge weights for self-loops
    do not represent already reduced communities you may want to remove
    the self-loops before inputting that graph.

    Parameters
    ----------
    G : NetworkX graph
    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.
    resolution : float, optional (default=1)
        If resolution is less than 1, the algorithm favors larger communities.
        Greater than 1 favors smaller communities
    threshold : float, optional (default=0.0000001)
        Modularity gain threshold for each level. If the gain of modularity
        between 2 levels of the algorithm is less than the given threshold
        then the algorithm stops and returns the resulting communities.
    max_level : int or None, optional (default=None)
        The maximum number of levels (steps of the algorithm) to compute.
        Must be a positive integer or None. If None, then there is no max
        level and the threshold parameter determines the stopping condition.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    list
        A list of sets (partition of `G`). Each set represents one community and contains
        all the nodes that constitute it.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.petersen_graph()
    >>> nx.community.louvain_communities(G, seed=123)
    [{0, 4, 5, 7, 9}, {1, 2, 3, 6, 8}]

    Notes
    -----
    The order in which the nodes are considered can affect the final output. In the algorithm
    the ordering happens using a random shuffle.

    References
    ----------
    .. [1] Blondel, V.D. et al. Fast unfolding of communities in
       large networks. J. Stat. Mech 10008, 1-12(2008). https://doi.org/10.1088/1742-5468/2008/10/P10008
    .. [2] Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing
       well-connected communities. Sci Rep 9, 5233 (2019). https://doi.org/10.1038/s41598-019-41695-z
    .. [3] Nicolas Dugué, Anthony Perez. Directed Louvain : maximizing modularity in directed networks.
        [Research Report] Université d’Orléans. 2015. hal-01231784. https://hal.archives-ouvertes.fr/hal-01231784

    See Also
    --------
    louvain_partitions
    """

    partitions = louvain_partitions(G, weight, resolution, threshold, seed)
    if max_level is not None:
        if max_level <= 0:
            raise ValueError("max_level argument must be a positive integer or None")
        partitions = itertools.islice(partitions, max_level)
    final_partition = deque(partitions, maxlen=1)
    return final_partition.pop()


@py_random_state("seed")
@nx._dispatchable(edge_attrs="weight")
def louvain_partitions(
    G, weight="weight", resolution=1, threshold=0.0000001, seed=None
):
    """Yields partitions for each level of the Louvain Community Detection Algorithm

    Louvain Community Detection Algorithm is a simple method to extract the community
    structure of a network. This is a heuristic method based on modularity optimization. [1]_

    The partitions at each level (step of the algorithm) form a dendrogram of communities.
    A dendrogram is a diagram representing a tree and each level represents
    a partition of the G graph. The top level contains the smallest communities
    and as you traverse to the bottom of the tree the communities get bigger
    and the overall modularity increases making the partition better.

    Each level is generated by executing the two phases of the Louvain Community
    Detection Algorithm.

    Be careful with self-loops in the input graph. These are treated as
    previously reduced communities -- as if the process had been started
    in the middle of the algorithm. Large self-loop edge weights thus
    represent strong communities and in practice may be hard to add
    other nodes to.  If your input graph edge weights for self-loops
    do not represent already reduced communities you may want to remove
    the self-loops before inputting that graph.

    Parameters
    ----------
    G : NetworkX graph
    weight : string or None, optional (default="weight")
     The name of an edge attribute that holds the numerical value
     used as a weight. If None then each edge has weight 1.
    resolution : float, optional (default=1)
        If resolution is less than 1, the algorithm favors larger communities.
        Greater than 1 favors smaller communities
    threshold : float, optional (default=0.0000001)
     Modularity gain threshold for each level. If the gain of modularity
     between 2 levels of the algorithm is less than the given threshold
     then the algorithm stops and returns the resulting communities.
    seed : integer, random_state, or None (default)
     Indicator of random number generation state.
     See :ref:`Randomness<randomness>`.

    Yields
    ------
    list
        A list of sets (partition of `G`). Each set represents one community and contains
        all the nodes that constitute it.

    References
    ----------
    .. [1] Blondel, V.D. et al. Fast unfolding of communities in
       large networks. J. Stat. Mech 10008, 1-12(2008)

    See Also
    --------
    louvain_communities
    """

    partition = [{u} for u in G.nodes()]
    if nx.is_empty(G):
        yield partition
        return
    mod = modularity(G, partition, resolution=resolution, weight=weight)
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from(G)
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))

    m = graph.size(weight="weight")
    partition, inner_partition, improvement = _one_level(
        graph, m, partition, resolution, is_directed, seed
    )
    improvement = True
    while improvement:
        # gh-5901 protect the sets in the yielded list from further manipulation here
        yield [s.copy() for s in partition]
        new_mod = modularity(
            graph, inner_partition, resolution=resolution, weight="weight"
        )
        if new_mod - mod <= threshold:
            return
        mod = new_mod
        graph = _gen_graph(graph, inner_partition)
        partition, inner_partition, improvement = _one_level(
            graph, m, partition, resolution, is_directed, seed
        )


def _one_level(G, m, partition, resolution=1, is_directed=False, seed=None):
    """Calculate one level of the Louvain partitions tree

    Parameters
    ----------
    G : NetworkX Graph/DiGraph
        The graph from which to detect communities
    m : number
        The size of the graph `G`.
    partition : list of sets of nodes
        A valid partition of the graph `G`
    resolution : positive number
        The resolution parameter for computing the modularity of a partition
    is_directed : bool
        True if `G` is a directed graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    """
    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
    if is_directed:
        in_degrees = dict(G.in_degree(weight="weight"))
        out_degrees = dict(G.out_degree(weight="weight"))
        Stot_in = list(in_degrees.values())
        Stot_out = list(out_degrees.values())
        # Calculate weights for both in and out neighbors without considering self-loops
        nbrs = {}
        for u in G:
            nbrs[u] = defaultdict(float)
            for _, n, wt in G.out_edges(u, data="weight"):
                if u != n:
                    nbrs[u][n] += wt
            for n, _, wt in G.in_edges(u, data="weight"):
                if u != n:
                    nbrs[u][n] += wt
    else:
        degrees = dict(G.degree(weight="weight"))
        Stot = list(degrees.values())
        nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
    rand_nodes = list(G.nodes)
    seed.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            best_mod = 0
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com)
            if is_directed:
                in_degree = in_degrees[u]
                out_degree = out_degrees[u]
                Stot_in[best_com] -= in_degree
                Stot_out[best_com] -= out_degree
                remove_cost = (
                    -weights2com[best_com] / m
                    + resolution
                    * (out_degree * Stot_in[best_com] + in_degree * Stot_out[best_com])
                    / m**2
                )
            else:
                degree = degrees[u]
                Stot[best_com] -= degree
                remove_cost = -weights2com[best_com] / m + resolution * (
                    Stot[best_com] * degree
                ) / (2 * m**2)
            for nbr_com, wt in weights2com.items():
                if is_directed:
                    gain = (
                        remove_cost
                        + wt / m
                        - resolution
                        * (
                            out_degree * Stot_in[nbr_com]
                            + in_degree * Stot_out[nbr_com]
                        )
                        / m**2
                    )
                else:
                    gain = (
                        remove_cost
                        + wt / m
                        - resolution * (Stot[nbr_com] * degree) / (2 * m**2)
                    )
                if gain > best_mod:
                    best_mod = gain
                    best_com = nbr_com
            if is_directed:
                Stot_in[best_com] += in_degree
                Stot_out[best_com] += out_degree
            else:
                Stot[best_com] += degree
            if best_com != node2com[u]:
                com = G.nodes[u].get("nodes", {u})
                partition[node2com[u]].difference_update(com)
                inner_partition[node2com[u]].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    return partition, inner_partition, improvement


def _neighbor_weights(nbrs, node2com):
    """Calculate weights between node and its neighbor communities.

    Parameters
    ----------
    nbrs : dictionary
           Dictionary with nodes' neighbors as keys and their edge weight as value.
    node2com : dictionary
           Dictionary with all graph's nodes as keys and their community index as value.

    """
    weights = defaultdict(float)
    for nbr, wt in nbrs.items():
        weights[node2com[nbr]] += wt
    return weights


def _gen_graph(G, partition):
    """Generate a new graph based on the partitions of a given graph"""
    H = G.__class__()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for node in part:
            node2com[node] = i
            nodes.update(G.nodes[node].get("nodes", {node}))
        H.add_node(i, nodes=nodes)

    for node1, node2, wt in G.edges(data=True):
        wt = wt["weight"]
        com1 = node2com[node1]
        com2 = node2com[node2]
        temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
        H.add_edge(com1, com2, weight=wt + temp)
    return H


def _convert_multigraph(G, weight, is_directed):
    """Convert a Multigraph to normal Graph"""
    if is_directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()
    H.add_nodes_from(G)
    for u, v, wt in G.edges(data=weight, default=1):
        if H.has_edge(u, v):
            H[u][v]["weight"] += wt
        else:
            H.add_edge(u, v, weight=wt)
    return H
<|endoftext|>"""
Label propagation community detection algorithms.
"""

from collections import Counter, defaultdict, deque

import networkx as nx
from networkx.utils import groups, not_implemented_for, py_random_state

__all__ = [
    "label_propagation_communities",
    "asyn_lpa_communities",
    "fast_label_propagation_communities",
]


@py_random_state("seed")
@nx._dispatchable(edge_attrs="weight")
def fast_label_propagation_communities(G, *, weight=None, seed=None):
    """Returns communities in `G` as detected by fast label propagation.

    The fast label propagation algorithm is described in [1]_. The algorithm is
    probabilistic and the found communities may vary in different executions.

    The algorithm operates as follows. First, the community label of each node is
    set to a unique label. The algorithm then repeatedly updates the labels of
    the nodes to the most frequent label in their neighborhood. In case of ties,
    a random label is chosen from the most frequent labels.

    The algorithm maintains a queue of nodes that still need to be processed.
    Initially, all nodes are added to the queue in a random order. Then the nodes
    are removed from the queue one by one and processed. If a node updates its label,
    all its neighbors that have a different label are added to the queue (if not
    already in the queue). The algorithm stops when the queue is empty.

    Parameters
    ----------
    G : Graph, DiGraph, MultiGraph, or MultiDiGraph
        Any NetworkX graph.

    weight : string, or None (default)
        The edge attribute representing a non-negative weight of an edge. If None,
        each edge is assumed to have weight one. The weight of an edge is used in
        determining the frequency with which a label appears among the neighbors of
        a node (edge with weight `w` is equivalent to `w` unweighted edges).

    seed : integer, random_state, or None (default)
        Indicator of random number generation state. See :ref:`Randomness<randomness>`.

    Returns
    -------
    communities : iterable
        Iterable of communities given as sets of nodes.

    Notes
    -----
    Edge directions are ignored for directed graphs.
    Edge weights must be non-negative numbers.

    References
    ----------
    .. [1] Vincent A. Traag & Lovro Šubelj. "Large network community detection by
       fast label propagation." Scientific Reports 13 (2023): 2701.
       https://doi.org/10.1038/s41598-023-29610-z
    """

    # Queue of nodes to be processed.
    nodes_queue = deque(G)
    seed.shuffle(nodes_queue)

    # Set of nodes in the queue.
    nodes_set = set(G)

    # Assign unique label to each node.
    comms = {node: i for i, node in enumerate(G)}

    while nodes_queue:
        # Remove next node from the queue to process.
        node = nodes_queue.popleft()
        nodes_set.remove(node)

        # Isolated nodes retain their initial label.
        if G.degree(node) > 0:
            # Compute frequency of labels in node's neighborhood.
            label_freqs = _fast_label_count(G, comms, node, weight)
            max_freq = max(label_freqs.values())

            # Always sample new label from most frequent labels.
            comm = seed.choice(
                [comm for comm in label_freqs if label_freqs[comm] == max_freq]
            )

            if comms[node] != comm:
                comms[node] = comm

                # Add neighbors that have different label to the queue.
                for nbr in nx.all_neighbors(G, node):
                    if comms[nbr] != comm and nbr not in nodes_set:
                        nodes_queue.append(nbr)
                        nodes_set.add(nbr)

    yield from groups(comms).values()


def _fast_label_count(G, comms, node, weight=None):
    """Computes the frequency of labels in the neighborhood of a node.

    Returns a dictionary keyed by label to the frequency of that label.
    """

    if weight is None:
        # Unweighted (un)directed simple graph.
        if not G.is_multigraph():
            label_freqs = Counter(map(comms.get, nx.all_neighbors(G, node)))

        # Unweighted (un)directed multigraph.
        else:
            label_freqs = defaultdict(int)
            for nbr in G[node]:
                label_freqs[comms[nbr]] += len(G[node][nbr])

            if G.is_directed():
                for nbr in G.pred[node]:
                    label_freqs[comms[nbr]] += len(G.pred[node][nbr])

    else:
        # Weighted undirected simple/multigraph.
        label_freqs = defaultdict(float)
        for _, nbr, w in G.edges(node, data=weight, default=1):
            label_freqs[comms[nbr]] += w

        # Weighted directed simple/multigraph.
        if G.is_directed():
            for nbr, _, w in G.in_edges(node, data=weight, default=1):
                label_freqs[comms[nbr]] += w

    return label_freqs


@py_random_state(2)
@nx._dispatchable(edge_attrs="weight")
def asyn_lpa_communities(G, weight=None, seed=None):
    """Returns communities in `G` as detected by asynchronous label
    propagation.

    The asynchronous label propagation algorithm is described in
    [1]_. The algorithm is probabilistic and the found communities may
    vary on different executions.

    The algorithm proceeds as follows. After initializing each node with
    a unique label, the algorithm repeatedly sets the label of a node to
    be the label that appears most frequently among that nodes
    neighbors. The algorithm halts when each node has the label that
    appears most frequently among its neighbors. The algorithm is
    asynchronous because each node is updated without waiting for
    updates on the remaining nodes.

    This generalized version of the algorithm in [1]_ accepts edge
    weights.

    Parameters
    ----------
    G : Graph

    weight : string
        The edge attribute representing the weight of an edge.
        If None, each edge is assumed to have weight one. In this
        algorithm, the weight of an edge is used in determining the
        frequency with which a label appears among the neighbors of a
        node: a higher weight means the label appears more often.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    communities : iterable
        Iterable of communities given as sets of nodes.

    Notes
    -----
    Edge weight attributes must be numerical.

    References
    ----------
    .. [1] Raghavan, Usha Nandini, Réka Albert, and Soundar Kumara. "Near
           linear time algorithm to detect community structures in large-scale
           networks." Physical Review E 76.3 (2007): 036106.
    """

    labels = {n: i for i, n in enumerate(G)}
    cont = True

    while cont:
        cont = False
        nodes = list(G)
        seed.shuffle(nodes)

        for node in nodes:
            if not G[node]:
                continue

            # Get label frequencies among adjacent nodes.
            # Depending on the order they are processed in,
            # some nodes will be in iteration t and others in t-1,
            # making the algorithm asynchronous.
            if weight is None:
                # initialising a Counter from an iterator of labels is
                # faster for getting unweighted label frequencies
                label_freq = Counter(map(labels.get, G[node]))
            else:
                # updating a defaultdict is substantially faster
                # for getting weighted label frequencies
                label_freq = defaultdict(float)
                for _, v, wt in G.edges(node, data=weight, default=1):
                    label_freq[labels[v]] += wt

            # Get the labels that appear with maximum frequency.
            max_freq = max(label_freq.values())
            best_labels = [
                label for label, freq in label_freq.items() if freq == max_freq
            ]

            # If the node does not have one of the maximum frequency labels,
            # randomly choose one of them and update the node's label.
            # Continue the iteration as long as at least one node
            # doesn't have a maximum frequency label.
            if labels[node] not in best_labels:
                labels[node] = seed.choice(best_labels)
                cont = True

    yield from groups(labels).values()


@not_implemented_for("directed")
@nx._dispatchable
def label_propagation_communities(G):
    """Generates community sets determined by label propagation

    Finds communities in `G` using a semi-synchronous label propagation
    method [1]_. This method combines the advantages of both the synchronous
    and asynchronous models. Not implemented for directed graphs.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.

    Returns
    -------
    communities : iterable
        A dict_values object that contains a set of nodes for each community.

    Raises
    ------
    NetworkXNotImplemented
       If the graph is directed

    References
    ----------
    .. [1] Cordasco, G., & Gargano, L. (2010, December). Community detection
       via semi-synchronous label propagation algorithms. In Business
       Applications of Social Network Analysis (BASNA), 2010 IEEE International
       Workshop on (pp. 1-8). IEEE.
    """
    coloring = _color_network(G)
    # Create a unique label for each node in the graph
    labeling = {v: k for k, v in enumerate(G)}
    while not _labeling_complete(labeling, G):
        # Update the labels of every node with the same color.
        for color, nodes in coloring.items():
            for n in nodes:
                _update_label(n, labeling, G)

    clusters = defaultdict(set)
    for node, label in labeling.items():
        clusters[label].add(node)
    return clusters.values()


def _color_network(G):
    """Colors the network so that neighboring nodes all have distinct colors.

    Returns a dict keyed by color to a set of nodes with that color.
    """
    coloring = {}  # color => set(node)
    colors = nx.coloring.greedy_color(G)
    for node, color in colors.items():
        if color in coloring:
            coloring[color].add(node)
        else:
            coloring[color] = {node}
    return coloring


def _labeling_complete(labeling, G):
    """Determines whether or not LPA is done.

    Label propagation is complete when all nodes have a label that is
    in the set of highest frequency labels amongst its neighbors.

    Nodes with no neighbors are considered complete.
    """
    return all(
        labeling[v] in _most_frequent_labels(v, labeling, G) for v in G if len(G[v]) > 0
    )


def _most_frequent_labels(node, labeling, G):
    """Returns a set of all labels with maximum frequency in `labeling`.

    Input `labeling` should be a dict keyed by node to labels.
    """
    if not G[node]:
        # Nodes with no neighbors are themselves a community and are labeled
        # accordingly, hence the immediate if statement.
        return {labeling[node]}

    # Compute the frequencies of all neighbors of node
    freqs = Counter(labeling[q] for q in G[node])
    max_freq = max(freqs.values())
    return {label for label, freq in freqs.items() if freq == max_freq}


def _update_label(node, labeling, G):
    """Updates the label of a node using the Prec-Max tie breaking algorithm

    The algorithm is explained in: 'Community Detection via Semi-Synchronous
    Label Propagation Algorithms' Cordasco and Gargano, 2011
    """
    high_labels = _most_frequent_labels(node, labeling, G)
    if len(high_labels) == 1:
        labeling[node] = high_labels.pop()
    elif len(high_labels) > 1:
        # Prec-Max
        if labeling[node] not in high_labels:
            labeling[node] = max(high_labels)
<|endoftext|>"""Asynchronous Fluid Communities algorithm for community detection."""

from collections import Counter

import networkx as nx
from networkx.algorithms.components import is_connected
from networkx.exception import NetworkXError
from networkx.utils import groups, not_implemented_for, py_random_state

__all__ = ["asyn_fluidc"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state(3)
@nx._dispatchable
def asyn_fluidc(G, k, max_iter=100, seed=None):
    """Returns communities in `G` as detected by Fluid Communities algorithm.

    The asynchronous fluid communities algorithm is described in
    [1]_. The algorithm is based on the simple idea of fluids interacting
    in an environment, expanding and pushing each other. Its initialization is
    random, so found communities may vary on different executions.

