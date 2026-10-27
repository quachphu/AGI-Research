    >>> G = nx.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> round(nx.effective_graph_resistance(G), 10)
    10.25

    Notes
    -----
    The implementation is based on Theorem 2.2 in [2]_. Self-loops are ignored.
    Multi-edges are contracted in one edge with weight equal to the harmonic sum of the weights.

    References
    ----------
    .. [1] Wolfram
       "Kirchhoff Index."
       https://mathworld.wolfram.com/KirchhoffIndex.html
    .. [2] W. Ellens, F. M. Spieksma, P. Van Mieghem, A. Jamakovic, R. E. Kooij.
        Effective graph resistance.
        Lin. Alg. Appl. 435:2491-2506, 2011.
    """
    import numpy as np

    if len(G) == 0:
        raise nx.NetworkXError("Graph G must contain at least one node.")

    # Disconnected graphs have infinite Effective graph resistance
    if not nx.is_connected(G):
        return float("inf")

    # Invert weights
    G = G.copy()
    if invert_weight and weight is not None:
        if G.is_multigraph():
            for u, v, k, d in G.edges(keys=True, data=True):
                d[weight] = 1 / d[weight]
        else:
            for u, v, d in G.edges(data=True):
                d[weight] = 1 / d[weight]

    # Get Laplacian eigenvalues
    mu = np.sort(nx.laplacian_spectrum(G, weight=weight))

    # Compute Effective graph resistance based on spectrum of the Laplacian
    # Self-loops are ignored
    return float(np.sum(1 / mu[1:]) * G.number_of_nodes())


@nx.utils.not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def kemeny_constant(G, *, weight=None):
    """Returns the Kemeny constant of the given graph.

    The *Kemeny constant* (or Kemeny's constant) of a graph `G`
    can be computed by regarding the graph as a Markov chain.
    The Kemeny constant is then the expected number of time steps
    to transition from a starting state i to a random destination state
    sampled from the Markov chain's stationary distribution.
    The Kemeny constant is independent of the chosen initial state [1]_.

    The Kemeny constant measures the time needed for spreading
    across a graph. Low values indicate a closely connected graph
    whereas high values indicate a spread-out graph.

    If weight is not provided, then a weight of 1 is used for all edges.

    Since `G` represents a Markov chain, the weights must be positive.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default=None)
       The edge data key used to compute the Kemeny constant.
       If None, then each edge has weight 1.

    Returns
    -------
    float
        The Kemeny constant of the graph `G`.

    Raises
    ------
    NetworkXNotImplemented
        If the graph `G` is directed.

    NetworkXError
        If the graph `G` is not connected, or contains no nodes,
        or has edges with negative weights.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> round(nx.kemeny_constant(G), 10)
    3.2

    Notes
    -----
    The implementation is based on equation (3.3) in [2]_.
    Self-loops are allowed and indicate a Markov chain where
    the state can remain the same. Multi-edges are contracted
    in one edge with weight equal to the sum of the weights.

    References
    ----------
    .. [1] Wikipedia
       "Kemeny's constant."
       https://en.wikipedia.org/wiki/Kemeny%27s_constant
    .. [2] Lovász L.
        Random walks on graphs: A survey.
        Paul Erdös is Eighty, vol. 2, Bolyai Society,
        Mathematical Studies, Keszthely, Hungary (1993), pp. 1-46
    """
    import numpy as np
    import scipy as sp

    if len(G) == 0:
        raise nx.NetworkXError("Graph G must contain at least one node.")
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph G must be connected.")
    if nx.is_negatively_weighted(G, weight=weight):
        raise nx.NetworkXError("The weights of graph G must be nonnegative.")

    # Compute matrix H = D^-1/2 A D^-1/2
    A = nx.adjacency_matrix(G, weight=weight)
    n, m = A.shape
    diags = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sp.sparse.csr_array(sp.sparse.spdiags(diags_sqrt, 0, m, n, format="csr"))
    H = DH @ (A @ DH)

    # Compute eigenvalues of H
    eig = np.sort(sp.linalg.eigvalsh(H.todense()))

    # Compute the Kemeny constant
    return float(np.sum(1 / (1 - eig[:-1])))
<|endoftext|>"""
Functions for identifying isolate (degree zero) nodes.
"""

import networkx as nx

__all__ = ["is_isolate", "isolates", "number_of_isolates"]


@nx._dispatchable
def is_isolate(G, n):
    """Determines whether a node is an isolate.

    An *isolate* is a node with no neighbors (that is, with degree
    zero). For directed graphs, this means no in-neighbors and no
    out-neighbors.

    Parameters
    ----------
    G : NetworkX graph

    n : node
        A node in `G`.

    Returns
    -------
    is_isolate : bool
       True if and only if `n` has no neighbors.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2)
    >>> G.add_node(3)
    >>> nx.is_isolate(G, 2)
    False
    >>> nx.is_isolate(G, 3)
    True
    """
    return G.degree(n) == 0


@nx._dispatchable
def isolates(G):
    """Iterator over isolates in the graph.

    An *isolate* is a node with no neighbors (that is, with degree
    zero). For directed graphs, this means no in-neighbors and no
    out-neighbors.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    iterator
        An iterator over the isolates of `G`.

    Examples
    --------
    To get a list of all isolates of a graph, use the :class:`list`
    constructor::

        >>> G = nx.Graph()
        >>> G.add_edge(1, 2)
        >>> G.add_node(3)
        >>> list(nx.isolates(G))
        [3]

    To remove all isolates in the graph, first create a list of the
    isolates, then use :meth:`Graph.remove_nodes_from`::

        >>> G.remove_nodes_from(list(nx.isolates(G)))
        >>> list(G)
        [1, 2]

    For digraphs, isolates have zero in-degree and zero out_degre::

        >>> G = nx.DiGraph([(0, 1), (1, 2)])
        >>> G.add_node(3)
        >>> list(nx.isolates(G))
        [3]

    """
    return (n for n, d in G.degree() if d == 0)


@nx._dispatchable
def number_of_isolates(G):
    """Returns the number of isolates in the graph.

    An *isolate* is a node with no neighbors (that is, with degree
    zero). For directed graphs, this means no in-neighbors and no
    out-neighbors.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    int
        The number of degree zero nodes in the graph `G`.

    """
    return sum(1 for v in isolates(G))
<|endoftext|>"""
Flow based cut algorithms
"""

import itertools

import networkx as nx

# Define the default maximum flow function to use in all flow based
# cut algorithms.
from networkx.algorithms.flow import build_residual_network, edmonds_karp

default_flow_func = edmonds_karp

from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity

__all__ = [
    "minimum_st_node_cut",
    "minimum_node_cut",
    "minimum_st_edge_cut",
    "minimum_edge_cut",
]


@nx._dispatchable(
    graphs={"G": 0, "auxiliary?": 4},
    preserve_edge_attrs={"auxiliary": {"capacity": float("inf")}},
    preserve_graph_attrs={"auxiliary"},
)
def minimum_st_edge_cut(G, s, t, flow_func=None, auxiliary=None, residual=None):
    """Returns the edges of the cut-set of a minimum (s, t)-cut.

    This function returns the set of edges of minimum cardinality that,
    if removed, would destroy all paths among source and target in G.
    Edge weights are not considered. See :meth:`minimum_cut` for
    computing minimum cuts considering edge weights.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node for the flow.

    t : node
        Sink node for the flow.

    auxiliary : NetworkX DiGraph
        Auxiliary digraph to compute flow based node connectivity. It has
        to have a graph attribute called mapping with a dictionary mapping
        node names in G and in the auxiliary digraph. If provided
        it will be reused instead of recreated. Default value: None.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See :meth:`node_connectivity` for
        details. The choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    residual : NetworkX DiGraph
        Residual network to compute maximum flow. If provided it will be
        reused instead of recreated. Default value: None.

    Returns
    -------
    cutset : set
        Set of edges that, if removed from the graph, will disconnect it.

    See also
    --------
    :meth:`minimum_cut`
    :meth:`minimum_node_cut`
    :meth:`minimum_edge_cut`
    :meth:`stoer_wagner`
    :meth:`node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Examples
    --------
    This function is not imported in the base NetworkX namespace, so you
    have to explicitly import it from the connectivity package:

    >>> from networkx.algorithms.connectivity import minimum_st_edge_cut

    We use in this example the platonic icosahedral graph, which has edge
    connectivity 5.

    >>> G = nx.icosahedral_graph()
    >>> len(minimum_st_edge_cut(G, 0, 6))
    5

    If you need to compute local edge cuts on several pairs of
    nodes in the same graph, it is recommended that you reuse the
    data structures that NetworkX uses in the computation: the
    auxiliary digraph for edge connectivity, and the residual
    network for the underlying maximum flow computation.

    Example of how to compute local edge cuts among all pairs of
    nodes of the platonic icosahedral graph reusing the data
    structures.

    >>> import itertools
    >>> # You also have to explicitly import the function for
    >>> # building the auxiliary digraph from the connectivity package
    >>> from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity
    >>> H = build_auxiliary_edge_connectivity(G)
    >>> # And the function for building the residual network from the
    >>> # flow package
    >>> from networkx.algorithms.flow import build_residual_network
    >>> # Note that the auxiliary digraph has an edge attribute named capacity
    >>> R = build_residual_network(H, "capacity")
    >>> result = dict.fromkeys(G, dict())
    >>> # Reuse the auxiliary digraph and the residual network by passing them
    >>> # as parameters
    >>> for u, v in itertools.combinations(G, 2):
    ...     k = len(minimum_st_edge_cut(G, u, v, auxiliary=H, residual=R))
    ...     result[u][v] = k
    >>> all(result[u][v] == 5 for u, v in itertools.combinations(G, 2))
    True

    You can also use alternative flow algorithms for computing edge
    cuts. For instance, in dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better than
    the default :meth:`edmonds_karp` which is faster for sparse
    networks with highly skewed degree distributions. Alternative flow
    functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> len(minimum_st_edge_cut(G, 0, 6, flow_func=shortest_augmenting_path))
    5

    """
    if flow_func is None:
        flow_func = default_flow_func

    if auxiliary is None:
        H = build_auxiliary_edge_connectivity(G)
    else:
        H = auxiliary

    kwargs = {"capacity": "capacity", "flow_func": flow_func, "residual": residual}

    cut_value, partition = nx.minimum_cut(H, s, t, **kwargs)
    reachable, non_reachable = partition
    # Any edge in the original graph linking the two sets in the
    # partition is part of the edge cutset
    cutset = set()
    for u, nbrs in ((n, G[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    return cutset


@nx._dispatchable(
    graphs={"G": 0, "auxiliary?": 4},
    preserve_node_attrs={"auxiliary": {"id": None}},
    preserve_graph_attrs={"auxiliary"},
)
def minimum_st_node_cut(G, s, t, flow_func=None, auxiliary=None, residual=None):
    r"""Returns a set of nodes of minimum cardinality that disconnect source
    from target in G.

    This function returns the set of nodes of minimum cardinality that,
    if removed, would destroy all paths among source and target in G.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node.

    t : node
        Target node.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The choice
        of the default function may change from version to version and
        should not be relied on. Default value: None.

    auxiliary : NetworkX DiGraph
        Auxiliary digraph to compute flow based node connectivity. It has
        to have a graph attribute called mapping with a dictionary mapping
        node names in G and in the auxiliary digraph. If provided
        it will be reused instead of recreated. Default value: None.

    residual : NetworkX DiGraph
        Residual network to compute maximum flow. If provided it will be
        reused instead of recreated. Default value: None.

    Returns
    -------
    cutset : set
        Set of nodes that, if removed, would destroy all paths between
        source and target in G.

    Examples
    --------
    This function is not imported in the base NetworkX namespace, so you
    have to explicitly import it from the connectivity package:

    >>> from networkx.algorithms.connectivity import minimum_st_node_cut

    We use in this example the platonic icosahedral graph, which has node
    connectivity 5.

    >>> G = nx.icosahedral_graph()
    >>> len(minimum_st_node_cut(G, 0, 6))
    5

    If you need to compute local st cuts between several pairs of
    nodes in the same graph, it is recommended that you reuse the
    data structures that NetworkX uses in the computation: the
    auxiliary digraph for node connectivity and node cuts, and the
    residual network for the underlying maximum flow computation.

    Example of how to compute local st node cuts reusing the data
    structures:

    >>> # You also have to explicitly import the function for
    >>> # building the auxiliary digraph from the connectivity package
    >>> from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
    >>> H = build_auxiliary_node_connectivity(G)
    >>> # And the function for building the residual network from the
    >>> # flow package
    >>> from networkx.algorithms.flow import build_residual_network
    >>> # Note that the auxiliary digraph has an edge attribute named capacity
    >>> R = build_residual_network(H, "capacity")
    >>> # Reuse the auxiliary digraph and the residual network by passing them
    >>> # as parameters
    >>> len(minimum_st_node_cut(G, 0, 6, auxiliary=H, residual=R))
    5

    You can also use alternative flow algorithms for computing minimum st
    node cuts. For instance, in dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better than
    the default :meth:`edmonds_karp` which is faster for sparse
    networks with highly skewed degree distributions. Alternative flow
    functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> len(minimum_st_node_cut(G, 0, 6, flow_func=shortest_augmenting_path))
    5

    Notes
    -----
    This is a flow based implementation of minimum node cut. The algorithm
    is based in solving a number of maximum flow computations to determine
    the capacity of the minimum cut on an auxiliary directed network that
    corresponds to the minimum node cut of G. It handles both directed
    and undirected graphs. This implementation is based on algorithm 11
    in [1]_.

    See also
    --------
    :meth:`minimum_node_cut`
    :meth:`minimum_edge_cut`
    :meth:`stoer_wagner`
    :meth:`node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    if auxiliary is None:
        H = build_auxiliary_node_connectivity(G)
    else:
        H = auxiliary

    mapping = H.graph.get("mapping", None)
    if mapping is None:
        raise nx.NetworkXError("Invalid auxiliary digraph.")
    if G.has_edge(s, t) or G.has_edge(t, s):
        return {}
    kwargs = {"flow_func": flow_func, "residual": residual, "auxiliary": H}

    # The edge cut in the auxiliary digraph corresponds to the node cut in the
    # original graph.
    edge_cut = minimum_st_edge_cut(H, f"{mapping[s]}B", f"{mapping[t]}A", **kwargs)
    # Each node in the original graph maps to two nodes of the auxiliary graph
    node_cut = {H.nodes[node]["id"] for edge in edge_cut for node in edge}
    return node_cut - {s, t}


@nx._dispatchable
def minimum_node_cut(G, s=None, t=None, flow_func=None):
    r"""Returns a set of nodes of minimum cardinality that disconnects G.

    If source and target nodes are provided, this function returns the
    set of nodes of minimum cardinality that, if removed, would destroy
    all paths among source and target in G. If not, it returns a set
    of nodes of minimum cardinality that disconnects G.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node. Optional. Default value: None.

    t : node
        Target node. Optional. Default value: None.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The
        choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    Returns
    -------
    cutset : set
        Set of nodes that, if removed, would disconnect G. If source
        and target nodes are provided, the set contains the nodes that
        if removed, would destroy all paths between source and target.

    Examples
    --------
    >>> # Platonic icosahedral graph has node connectivity 5
    >>> G = nx.icosahedral_graph()
    >>> node_cut = nx.minimum_node_cut(G)
    >>> len(node_cut)
    5

    You can use alternative flow algorithms for the underlying maximum
    flow computation. In dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better
    than the default :meth:`edmonds_karp`, which is faster for
    sparse networks with highly skewed degree distributions. Alternative
    flow functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> node_cut == nx.minimum_node_cut(G, flow_func=shortest_augmenting_path)
    True

    If you specify a pair of nodes (source and target) as parameters,
    this function returns a local st node cut.

    >>> len(nx.minimum_node_cut(G, 3, 7))
    5

    If you need to perform several local st cuts among different
    pairs of nodes on the same graph, it is recommended that you reuse
    the data structures used in the maximum flow computations. See
    :meth:`minimum_st_node_cut` for details.

    Notes
    -----
    This is a flow based implementation of minimum node cut. The algorithm
    is based in solving a number of maximum flow computations to determine
    the capacity of the minimum cut on an auxiliary directed network that
    corresponds to the minimum node cut of G. It handles both directed
    and undirected graphs. This implementation is based on algorithm 11
    in [1]_.

    See also
    --------
    :meth:`minimum_st_node_cut`
    :meth:`minimum_cut`
    :meth:`minimum_edge_cut`
    :meth:`stoer_wagner`
    :meth:`node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    if (s is not None and t is None) or (s is None and t is not None):
        raise nx.NetworkXError("Both source and target must be specified.")

    # Local minimum node cut.
    if s is not None and t is not None:
        if s not in G:
            raise nx.NetworkXError(f"node {s} not in graph")
        if t not in G:
            raise nx.NetworkXError(f"node {t} not in graph")
        return minimum_st_node_cut(G, s, t, flow_func=flow_func)

    # Global minimum node cut.
    # Analog to the algorithm 11 for global node connectivity in [1].
    if G.is_directed():
        if not nx.is_weakly_connected(G):
            raise nx.NetworkXError("Input graph is not connected")
        iter_func = itertools.permutations

        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors(v), G.successors(v)])

    else:
        if not nx.is_connected(G):
            raise nx.NetworkXError("Input graph is not connected")
        iter_func = itertools.combinations
        neighbors = G.neighbors

    # Reuse the auxiliary digraph and the residual network.
    H = build_auxiliary_node_connectivity(G)
    R = build_residual_network(H, "capacity")
    kwargs = {"flow_func": flow_func, "auxiliary": H, "residual": R}

    # Choose a node with minimum degree.
    v = min(G, key=G.degree)
    # Initial node cutset is all neighbors of the node with minimum degree.
    min_cut = set(G[v])
    # Compute st node cuts between v and all its non-neighbors nodes in G.
    for w in set(G) - set(neighbors(v)) - {v}:
        this_cut = minimum_st_node_cut(G, v, w, **kwargs)
        if len(min_cut) >= len(this_cut):
            min_cut = this_cut
    # Also for non adjacent pairs of neighbors of v.
    for x, y in iter_func(neighbors(v), 2):
        if y in G[x]:
            continue
        this_cut = minimum_st_node_cut(G, x, y, **kwargs)
        if len(min_cut) >= len(this_cut):
            min_cut = this_cut

    return min_cut


@nx._dispatchable
def minimum_edge_cut(G, s=None, t=None, flow_func=None):
    r"""Returns a set of edges of minimum cardinality that disconnects G.

    If source and target nodes are provided, this function returns the
    set of edges of minimum cardinality that, if removed, would break
    all paths among source and target in G. If not, it returns a set of
    edges of minimum cardinality that disconnects G.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node. Optional. Default value: None.

    t : node
        Target node. Optional. Default value: None.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The
        choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    Returns
    -------
    cutset : set
        Set of edges that, if removed, would disconnect G. If source
        and target nodes are provided, the set contains the edges that
        if removed, would destroy all paths between source and target.

    Examples
    --------
    >>> # Platonic icosahedral graph has edge connectivity 5
    >>> G = nx.icosahedral_graph()
    >>> len(nx.minimum_edge_cut(G))
    5

    You can use alternative flow algorithms for the underlying
    maximum flow computation. In dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better
    than the default :meth:`edmonds_karp`, which is faster for
    sparse networks with highly skewed degree distributions.
    Alternative flow functions have to be explicitly imported
    from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> len(nx.minimum_edge_cut(G, flow_func=shortest_augmenting_path))
    5

    If you specify a pair of nodes (source and target) as parameters,
    this function returns the value of local edge connectivity.

    >>> nx.edge_connectivity(G, 3, 7)
    5

    If you need to perform several local computations among different
    pairs of nodes on the same graph, it is recommended that you reuse
    the data structures used in the maximum flow computations. See
    :meth:`local_edge_connectivity` for details.

    Notes
    -----
    This is a flow based implementation of minimum edge cut. For
    undirected graphs the algorithm works by finding a 'small' dominating
    set of nodes of G (see algorithm 7 in [1]_) and computing the maximum
    flow between an arbitrary node in the dominating set and the rest of
    nodes in it. This is an implementation of algorithm 6 in [1]_. For
    directed graphs, the algorithm does n calls to the max flow function.
    The function raises an error if the directed graph is not weakly
    connected and returns an empty set if it is weakly connected.
    It is an implementation of algorithm 8 in [1]_.

    See also
    --------
    :meth:`minimum_st_edge_cut`
    :meth:`minimum_node_cut`
    :meth:`stoer_wagner`
    :meth:`node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    if (s is not None and t is None) or (s is None and t is not None):
        raise nx.NetworkXError("Both source and target must be specified.")

    # reuse auxiliary digraph and residual network
    H = build_auxiliary_edge_connectivity(G)
    R = build_residual_network(H, "capacity")
    kwargs = {"flow_func": flow_func, "residual": R, "auxiliary": H}

    # Local minimum edge cut if s and t are not None
    if s is not None and t is not None:
        if s not in G:
            raise nx.NetworkXError(f"node {s} not in graph")
        if t not in G:
            raise nx.NetworkXError(f"node {t} not in graph")
        return minimum_st_edge_cut(H, s, t, **kwargs)

    # Global minimum edge cut
    # Analog to the algorithm for global edge connectivity
    if G.is_directed():
        # Based on algorithm 8 in [1]
        if not nx.is_weakly_connected(G):
            raise nx.NetworkXError("Input graph is not connected")

        # Initial cutset is all edges of a node with minimum degree
        node = min(G, key=G.degree)
        min_cut = set(G.edges(node))
        nodes = list(G)
        n = len(nodes)
        for i in range(n):
            try:
                this_cut = minimum_st_edge_cut(H, nodes[i], nodes[i + 1], **kwargs)
                if len(this_cut) <= len(min_cut):
                    min_cut = this_cut
            except IndexError:  # Last node!
                this_cut = minimum_st_edge_cut(H, nodes[i], nodes[0], **kwargs)
                if len(this_cut) <= len(min_cut):
                    min_cut = this_cut

        return min_cut

    else:  # undirected
        # Based on algorithm 6 in [1]
        if not nx.is_connected(G):
            raise nx.NetworkXError("Input graph is not connected")

        # Initial cutset is all edges of a node with minimum degree
        node = min(G, key=G.degree)
        min_cut = set(G.edges(node))
        # A dominating set is \lambda-covering
        # We need a dominating set with at least two nodes
        for node in G:
            D = nx.dominating_set(G, start_with=node)
            v = D.pop()
            if D:
                break
        else:
            # in complete graphs the dominating set will always be of one node
            # thus we return min_cut, which now contains the edges of a node
            # with minimum degree
            return min_cut
        for w in D:
            this_cut = minimum_st_edge_cut(H, v, w, **kwargs)
            if len(this_cut) <= len(min_cut):
                min_cut = this_cut

        return min_cut
<|endoftext|>"""
Flow based connectivity algorithms
"""

import itertools
from operator import itemgetter

import networkx as nx

# Define the default maximum flow function to use in all flow based
# connectivity algorithms.
from networkx.algorithms.flow import (
    boykov_kolmogorov,
    build_residual_network,
    dinitz,
    edmonds_karp,
    preflow_push,
    shortest_augmenting_path,
)

default_flow_func = edmonds_karp

from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity

__all__ = [
    "average_node_connectivity",
    "local_node_connectivity",
    "node_connectivity",
    "local_edge_connectivity",
    "edge_connectivity",
    "all_pairs_node_connectivity",
]


@nx._dispatchable(graphs={"G": 0, "auxiliary?": 4}, preserve_graph_attrs={"auxiliary"})
def local_node_connectivity(
    G, s, t, flow_func=None, auxiliary=None, residual=None, cutoff=None
):
    r"""Computes local node connectivity for nodes s and t.

    Local node connectivity for two non adjacent nodes s and t is the
    minimum number of nodes that must be removed (along with their incident
    edges) to disconnect them.

    This is a flow based implementation of node connectivity. We compute the
    maximum flow on an auxiliary digraph build from the original input
    graph (see below for details).

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    s : node
        Source node

    t : node
        Target node

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The choice
        of the default function may change from version to version and
        should not be relied on. Default value: None.

    auxiliary : NetworkX DiGraph
        Auxiliary digraph to compute flow based node connectivity. It has
        to have a graph attribute called mapping with a dictionary mapping
        node names in G and in the auxiliary digraph. If provided
        it will be reused instead of recreated. Default value: None.

    residual : NetworkX DiGraph
        Residual network to compute maximum flow. If provided it will be
        reused instead of recreated. Default value: None.

    cutoff : integer, float, or None (default: None)
        If specified, the maximum flow algorithm will terminate when the
        flow value reaches or exceeds the cutoff. This only works for flows
        that support the cutoff parameter (most do) and is ignored otherwise.

    Returns
    -------
    K : integer
        local node connectivity for nodes s and t

    Examples
    --------
    This function is not imported in the base NetworkX namespace, so you
    have to explicitly import it from the connectivity package:

    >>> from networkx.algorithms.connectivity import local_node_connectivity

    We use in this example the platonic icosahedral graph, which has node
    connectivity 5.

    >>> G = nx.icosahedral_graph()
    >>> local_node_connectivity(G, 0, 6)
    5

    If you need to compute local connectivity on several pairs of
    nodes in the same graph, it is recommended that you reuse the
    data structures that NetworkX uses in the computation: the
    auxiliary digraph for node connectivity, and the residual
    network for the underlying maximum flow computation.

    Example of how to compute local node connectivity among
    all pairs of nodes of the platonic icosahedral graph reusing
    the data structures.

    >>> import itertools
    >>> # You also have to explicitly import the function for
    >>> # building the auxiliary digraph from the connectivity package
    >>> from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
    >>> H = build_auxiliary_node_connectivity(G)
    >>> # And the function for building the residual network from the
    >>> # flow package
    >>> from networkx.algorithms.flow import build_residual_network
    >>> # Note that the auxiliary digraph has an edge attribute named capacity
    >>> R = build_residual_network(H, "capacity")
    >>> result = dict.fromkeys(G, dict())
    >>> # Reuse the auxiliary digraph and the residual network by passing them
    >>> # as parameters
    >>> for u, v in itertools.combinations(G, 2):
    ...     k = local_node_connectivity(G, u, v, auxiliary=H, residual=R)
    ...     result[u][v] = k
    >>> all(result[u][v] == 5 for u, v in itertools.combinations(G, 2))
    True

    You can also use alternative flow algorithms for computing node
    connectivity. For instance, in dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better than
    the default :meth:`edmonds_karp` which is faster for sparse
    networks with highly skewed degree distributions. Alternative flow
    functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> local_node_connectivity(G, 0, 6, flow_func=shortest_augmenting_path)
    5

    Notes
    -----
    This is a flow based implementation of node connectivity. We compute the
    maximum flow using, by default, the :meth:`edmonds_karp` algorithm (see:
    :meth:`maximum_flow`) on an auxiliary digraph build from the original
    input graph:

    For an undirected graph G having `n` nodes and `m` edges we derive a
    directed graph H with `2n` nodes and `2m+n` arcs by replacing each
    original node `v` with two nodes `v_A`, `v_B` linked by an (internal)
    arc in H. Then for each edge (`u`, `v`) in G we add two arcs
    (`u_B`, `v_A`) and (`v_B`, `u_A`) in H. Finally we set the attribute
    capacity = 1 for each arc in H [1]_ .

    For a directed graph G having `n` nodes and `m` arcs we derive a
    directed graph H with `2n` nodes and `m+n` arcs by replacing each
    original node `v` with two nodes `v_A`, `v_B` linked by an (internal)
    arc (`v_A`, `v_B`) in H. Then for each arc (`u`, `v`) in G we add one arc
    (`u_B`, `v_A`) in H. Finally we set the attribute capacity = 1 for
    each arc in H.

    This is equal to the local node connectivity because the value of
    a maximum s-t-flow is equal to the capacity of a minimum s-t-cut.

    See also
    --------
    :meth:`local_edge_connectivity`
    :meth:`node_connectivity`
    :meth:`minimum_node_cut`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1] Kammer, Frank and Hanjo Taubig. Graph Connectivity. in Brandes and
        Erlebach, 'Network Analysis: Methodological Foundations', Lecture
        Notes in Computer Science, Volume 3418, Springer-Verlag, 2005.
        http://www.informatik.uni-augsburg.de/thi/personen/kammer/Graph_Connectivity.pdf

    """
    if flow_func is None:
        flow_func = default_flow_func

    if auxiliary is None:
        H = build_auxiliary_node_connectivity(G)
    else:
        H = auxiliary

    mapping = H.graph.get("mapping", None)
    if mapping is None:
        raise nx.NetworkXError("Invalid auxiliary digraph.")

    kwargs = {"flow_func": flow_func, "residual": residual}

    if flow_func is not preflow_push:
        kwargs["cutoff"] = cutoff

    if flow_func is shortest_augmenting_path:
        kwargs["two_phase"] = True

    return nx.maximum_flow_value(H, f"{mapping[s]}B", f"{mapping[t]}A", **kwargs)


@nx._dispatchable
def node_connectivity(G, s=None, t=None, flow_func=None):
    r"""Returns node connectivity for a graph or digraph G.

    Node connectivity is equal to the minimum number of nodes that
    must be removed to disconnect G or render it trivial. If source
    and target nodes are provided, this function returns the local node
    connectivity: the minimum number of nodes that must be removed to break
    all paths from source to target in G.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    s : node
        Source node. Optional. Default value: None.

    t : node
        Target node. Optional. Default value: None.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The
        choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    Returns
    -------
    K : integer
        Node connectivity of G, or local node connectivity if source
        and target are provided.

    Examples
    --------
    >>> # Platonic icosahedral graph is 5-node-connected
    >>> G = nx.icosahedral_graph()
    >>> nx.node_connectivity(G)
    5

    You can use alternative flow algorithms for the underlying maximum
    flow computation. In dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better
    than the default :meth:`edmonds_karp`, which is faster for
    sparse networks with highly skewed degree distributions. Alternative
    flow functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> nx.node_connectivity(G, flow_func=shortest_augmenting_path)
    5

    If you specify a pair of nodes (source and target) as parameters,
    this function returns the value of local node connectivity.

    >>> nx.node_connectivity(G, 3, 7)
    5

    If you need to perform several local computations among different
    pairs of nodes on the same graph, it is recommended that you reuse
    the data structures used in the maximum flow computations. See
    :meth:`local_node_connectivity` for details.

    Notes
    -----
    This is a flow based implementation of node connectivity. The
    algorithm works by solving $O((n-\delta-1+\delta(\delta-1)/2))$
    maximum flow problems on an auxiliary digraph. Where $\delta$
    is the minimum degree of G. For details about the auxiliary
    digraph and the computation of local node connectivity see
    :meth:`local_node_connectivity`. This implementation is based
    on algorithm 11 in [1]_.

    See also
    --------
    :meth:`local_node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    if (s is not None and t is None) or (s is None and t is not None):
        raise nx.NetworkXError("Both source and target must be specified.")

    # Local node connectivity
    if s is not None and t is not None:
        if s not in G:
            raise nx.NetworkXError(f"node {s} not in graph")
        if t not in G:
            raise nx.NetworkXError(f"node {t} not in graph")
        return local_node_connectivity(G, s, t, flow_func=flow_func)

    # Global node connectivity
    if G.is_directed():
        if not nx.is_weakly_connected(G):
            return 0
        iter_func = itertools.permutations
        # It is necessary to consider both predecessors
        # and successors for directed graphs

        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors(v), G.successors(v)])

    else:
        if not nx.is_connected(G):
            return 0
        iter_func = itertools.combinations
        neighbors = G.neighbors

    # Reuse the auxiliary digraph and the residual network
    H = build_auxiliary_node_connectivity(G)
    R = build_residual_network(H, "capacity")
    kwargs = {"flow_func": flow_func, "auxiliary": H, "residual": R}

    # Pick a node with minimum degree
    # Node connectivity is bounded by degree.
    v, K = min(G.degree(), key=itemgetter(1))
    # compute local node connectivity with all its non-neighbors nodes
    for w in set(G) - set(neighbors(v)) - {v}:
        kwargs["cutoff"] = K
        K = min(K, local_node_connectivity(G, v, w, **kwargs))
    # Also for non adjacent pairs of neighbors of v
    for x, y in iter_func(neighbors(v), 2):
        if y in G[x]:
            continue
        kwargs["cutoff"] = K
        K = min(K, local_node_connectivity(G, x, y, **kwargs))

    return K


@nx._dispatchable
def average_node_connectivity(G, flow_func=None):
    r"""Returns the average connectivity of a graph G.

    The average connectivity `\bar{\kappa}` of a graph G is the average
    of local node connectivity over all pairs of nodes of G [1]_ .

    .. math::

        \bar{\kappa}(G) = \frac{\sum_{u,v} \kappa_{G}(u,v)}{{n \choose 2}}

    Parameters
    ----------

    G : NetworkX graph
        Undirected graph

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See :meth:`local_node_connectivity`
        for details. The choice of the default function may change from
        version to version and should not be relied on. Default value: None.

    Returns
    -------
    K : float
        Average node connectivity

    See also
    --------
    :meth:`local_node_connectivity`
    :meth:`node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1]  Beineke, L., O. Oellermann, and R. Pippert (2002). The average
            connectivity of a graph. Discrete mathematics 252(1-3), 31-45.
            http://www.sciencedirect.com/science/article/pii/S0012365X01001807

    """
    if G.is_directed():
        iter_func = itertools.permutations
    else:
        iter_func = itertools.combinations

    # Reuse the auxiliary digraph and the residual network
    H = build_auxiliary_node_connectivity(G)
    R = build_residual_network(H, "capacity")
    kwargs = {"flow_func": flow_func, "auxiliary": H, "residual": R}

    num, den = 0, 0
    for u, v in iter_func(G, 2):
        num += local_node_connectivity(G, u, v, **kwargs)
        den += 1

    if den == 0:  # Null Graph
        return 0
    return num / den


@nx._dispatchable
def all_pairs_node_connectivity(G, nbunch=None, flow_func=None):
    """Compute node connectivity between all pairs of nodes of G.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    nbunch: container
        Container of nodes. If provided node connectivity will be computed
        only over pairs of nodes in nbunch.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The
        choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    Returns
    -------
    all_pairs : dict
        A dictionary with node connectivity between all pairs of nodes
        in G, or in nbunch if provided.

    See also
    --------
    :meth:`local_node_connectivity`
    :meth:`edge_connectivity`
    :meth:`local_edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    """
    if nbunch is None:
        nbunch = G
    else:
        nbunch = set(nbunch)

    directed = G.is_directed()
    if directed:
        iter_func = itertools.permutations
    else:
        iter_func = itertools.combinations

    all_pairs = {n: {} for n in nbunch}

    # Reuse auxiliary digraph and residual network
    H = build_auxiliary_node_connectivity(G)
    mapping = H.graph["mapping"]
    R = build_residual_network(H, "capacity")
    kwargs = {"flow_func": flow_func, "auxiliary": H, "residual": R}

    for u, v in iter_func(nbunch, 2):
        K = local_node_connectivity(G, u, v, **kwargs)
        all_pairs[u][v] = K
        if not directed:
            all_pairs[v][u] = K

    return all_pairs


@nx._dispatchable(graphs={"G": 0, "auxiliary?": 4})
def local_edge_connectivity(
    G, s, t, flow_func=None, auxiliary=None, residual=None, cutoff=None
):
    r"""Returns local edge connectivity for nodes s and t in G.

    Local edge connectivity for two nodes s and t is the minimum number
    of edges that must be removed to disconnect them.

    This is a flow based implementation of edge connectivity. We compute the
    maximum flow on an auxiliary digraph build from the original
    network (see below for details). This is equal to the local edge
    connectivity because the value of a maximum s-t-flow is equal to the
    capacity of a minimum s-t-cut (Ford and Fulkerson theorem) [1]_ .

    Parameters
    ----------
    G : NetworkX graph
        Undirected or directed graph

    s : node
        Source node

    t : node
        Target node

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The
        choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    auxiliary : NetworkX DiGraph
        Auxiliary digraph for computing flow based edge connectivity. If
        provided it will be reused instead of recreated. Default value: None.

    residual : NetworkX DiGraph
        Residual network to compute maximum flow. If provided it will be
        reused instead of recreated. Default value: None.

    cutoff : integer, float, or None (default: None)
        If specified, the maximum flow algorithm will terminate when the
        flow value reaches or exceeds the cutoff. This only works for flows
        that support the cutoff parameter (most do) and is ignored otherwise.

    Returns
    -------
    K : integer
        local edge connectivity for nodes s and t.

    Examples
    --------
    This function is not imported in the base NetworkX namespace, so you
    have to explicitly import it from the connectivity package:

    >>> from networkx.algorithms.connectivity import local_edge_connectivity

    We use in this example the platonic icosahedral graph, which has edge
    connectivity 5.

    >>> G = nx.icosahedral_graph()
    >>> local_edge_connectivity(G, 0, 6)
    5

    If you need to compute local connectivity on several pairs of
    nodes in the same graph, it is recommended that you reuse the
    data structures that NetworkX uses in the computation: the
    auxiliary digraph for edge connectivity, and the residual
    network for the underlying maximum flow computation.

    Example of how to compute local edge connectivity among
    all pairs of nodes of the platonic icosahedral graph reusing
    the data structures.

    >>> import itertools
    >>> # You also have to explicitly import the function for
    >>> # building the auxiliary digraph from the connectivity package
    >>> from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity
    >>> H = build_auxiliary_edge_connectivity(G)
    >>> # And the function for building the residual network from the
    >>> # flow package
    >>> from networkx.algorithms.flow import build_residual_network
    >>> # Note that the auxiliary digraph has an edge attribute named capacity
    >>> R = build_residual_network(H, "capacity")
    >>> result = dict.fromkeys(G, dict())
    >>> # Reuse the auxiliary digraph and the residual network by passing them
    >>> # as parameters
    >>> for u, v in itertools.combinations(G, 2):
    ...     k = local_edge_connectivity(G, u, v, auxiliary=H, residual=R)
    ...     result[u][v] = k
    >>> all(result[u][v] == 5 for u, v in itertools.combinations(G, 2))
    True

    You can also use alternative flow algorithms for computing edge
    connectivity. For instance, in dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better than
    the default :meth:`edmonds_karp` which is faster for sparse
    networks with highly skewed degree distributions. Alternative flow
    functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> local_edge_connectivity(G, 0, 6, flow_func=shortest_augmenting_path)
    5

    Notes
    -----
    This is a flow based implementation of edge connectivity. We compute the
    maximum flow using, by default, the :meth:`edmonds_karp` algorithm on an
    auxiliary digraph build from the original input graph:

    If the input graph is undirected, we replace each edge (`u`,`v`) with
    two reciprocal arcs (`u`, `v`) and (`v`, `u`) and then we set the attribute
    'capacity' for each arc to 1. If the input graph is directed we simply
    add the 'capacity' attribute. This is an implementation of algorithm 1
    in [1]_.

    The maximum flow in the auxiliary network is equal to the local edge
    connectivity because the value of a maximum s-t-flow is equal to the
    capacity of a minimum s-t-cut (Ford and Fulkerson theorem).

    See also
    --------
    :meth:`edge_connectivity`
    :meth:`local_node_connectivity`
    :meth:`node_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    if flow_func is None:
        flow_func = default_flow_func

    if auxiliary is None:
        H = build_auxiliary_edge_connectivity(G)
    else:
        H = auxiliary

    kwargs = {"flow_func": flow_func, "residual": residual}

    if flow_func is not preflow_push:
        kwargs["cutoff"] = cutoff

    if flow_func is shortest_augmenting_path:
        kwargs["two_phase"] = True

    return nx.maximum_flow_value(H, s, t, **kwargs)


@nx._dispatchable
def edge_connectivity(G, s=None, t=None, flow_func=None, cutoff=None):
    r"""Returns the edge connectivity of the graph or digraph G.

    The edge connectivity is equal to the minimum number of edges that
    must be removed to disconnect G or render it trivial. If source
    and target nodes are provided, this function returns the local edge
    connectivity: the minimum number of edges that must be removed to
    break all paths from source to target in G.

    Parameters
    ----------
    G : NetworkX graph
        Undirected or directed graph

    s : node
        Source node. Optional. Default value: None.

    t : node
        Target node. Optional. Default value: None.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The
        choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    cutoff : integer, float, or None (default: None)
        If specified, the maximum flow algorithm will terminate when the
        flow value reaches or exceeds the cutoff. This only works for flows
        that support the cutoff parameter (most do) and is ignored otherwise.

    Returns
    -------
    K : integer
        Edge connectivity for G, or local edge connectivity if source
        and target were provided

    Examples
    --------
    >>> # Platonic icosahedral graph is 5-edge-connected
    >>> G = nx.icosahedral_graph()
    >>> nx.edge_connectivity(G)
    5

    You can use alternative flow algorithms for the underlying
    maximum flow computation. In dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better
    than the default :meth:`edmonds_karp`, which is faster for
    sparse networks with highly skewed degree distributions.
    Alternative flow functions have to be explicitly imported
    from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> nx.edge_connectivity(G, flow_func=shortest_augmenting_path)
    5

    If you specify a pair of nodes (source and target) as parameters,
    this function returns the value of local edge connectivity.

    >>> nx.edge_connectivity(G, 3, 7)
    5

    If you need to perform several local computations among different
    pairs of nodes on the same graph, it is recommended that you reuse
    the data structures used in the maximum flow computations. See
    :meth:`local_edge_connectivity` for details.

    Notes
    -----
    This is a flow based implementation of global edge connectivity.
    For undirected graphs the algorithm works by finding a 'small'
    dominating set of nodes of G (see algorithm 7 in [1]_ ) and
    computing local maximum flow (see :meth:`local_edge_connectivity`)
    between an arbitrary node in the dominating set and the rest of
    nodes in it. This is an implementation of algorithm 6 in [1]_ .
    For directed graphs, the algorithm does n calls to the maximum
    flow function. This is an implementation of algorithm 8 in [1]_ .

    See also
    --------
    :meth:`local_edge_connectivity`
    :meth:`local_node_connectivity`
    :meth:`node_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`
    :meth:`k_edge_components`
    :meth:`k_edge_subgraphs`

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    if (s is not None and t is None) or (s is None and t is not None):
        raise nx.NetworkXError("Both source and target must be specified.")

    # Local edge connectivity
    if s is not None and t is not None:
        if s not in G:
            raise nx.NetworkXError(f"node {s} not in graph")
        if t not in G:
            raise nx.NetworkXError(f"node {t} not in graph")
        return local_edge_connectivity(G, s, t, flow_func=flow_func, cutoff=cutoff)

    # Global edge connectivity
    # reuse auxiliary digraph and residual network
    H = build_auxiliary_edge_connectivity(G)
    R = build_residual_network(H, "capacity")
    kwargs = {"flow_func": flow_func, "auxiliary": H, "residual": R}

    if G.is_directed():
        # Algorithm 8 in [1]
        if not nx.is_weakly_connected(G):
            return 0

        # initial value for \lambda is minimum degree
        L = min(d for n, d in G.degree())
        nodes = list(G)
        n = len(nodes)

        if cutoff is not None:
            L = min(cutoff, L)

        for i in range(n):
            kwargs["cutoff"] = L
            try:
                L = min(L, local_edge_connectivity(G, nodes[i], nodes[i + 1], **kwargs))
            except IndexError:  # last node!
                L = min(L, local_edge_connectivity(G, nodes[i], nodes[0], **kwargs))
        return L
    else:  # undirected
        # Algorithm 6 in [1]
        if not nx.is_connected(G):
            return 0

        # initial value for \lambda is minimum degree
        L = min(d for n, d in G.degree())

        if cutoff is not None:
            L = min(cutoff, L)

        # A dominating set is \lambda-covering
        # We need a dominating set with at least two nodes
        for node in G:
            D = nx.dominating_set(G, start_with=node)
            v = D.pop()
            if D:
                break
        else:
            # in complete graphs the dominating sets will always be of one node
            # thus we return min degree
            return L

        for w in D:
            kwargs["cutoff"] = L
            L = min(L, local_edge_connectivity(G, v, w, **kwargs))

        return L
<|endoftext|>"""
Algorithms for finding k-edge-augmentations

A k-edge-augmentation is a set of edges, that once added to a graph, ensures
that the graph is k-edge-connected; i.e. the graph cannot be disconnected
unless k or more edges are removed.  Typically, the goal is to find the
augmentation with minimum weight.  In general, it is not guaranteed that a
k-edge-augmentation exists.

See Also
--------
:mod:`edge_kcomponents` : algorithms for finding k-edge-connected components
:mod:`connectivity` : algorithms for determining edge connectivity.
"""

import itertools as it
import math
from collections import defaultdict, namedtuple

import networkx as nx
from networkx.utils import not_implemented_for, py_random_state

__all__ = ["k_edge_augmentation", "is_k_edge_connected", "is_locally_k_edge_connected"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_k_edge_connected(G, k):
    """Tests to see if a graph is k-edge-connected.

    Is it impossible to disconnect the graph by removing fewer than k edges?
    If so, then G is k-edge-connected.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    k : integer
        edge connectivity to test for

    Returns
    -------
    boolean
        True if G is k-edge-connected.

    See Also
    --------
    :func:`is_locally_k_edge_connected`

    Examples
    --------
    >>> G = nx.barbell_graph(10, 0)
    >>> nx.is_k_edge_connected(G, k=1)
    True
    >>> nx.is_k_edge_connected(G, k=2)
    False
    """
    if k < 1:
        raise ValueError(f"k must be positive, not {k}")
    # First try to quickly determine if G is not k-edge-connected
    if G.number_of_nodes() < k + 1:
        return False
    elif any(d < k for n, d in G.degree()):
        return False
    else:
        # Otherwise perform the full check
        if k == 1:
            return nx.is_connected(G)
        elif k == 2:
            return nx.is_connected(G) and not nx.has_bridges(G)
        else:
            return nx.edge_connectivity(G, cutoff=k) >= k


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_locally_k_edge_connected(G, s, t, k):
    """Tests to see if an edge in a graph is locally k-edge-connected.

    Is it impossible to disconnect s and t by removing fewer than k edges?
    If so, then s and t are locally k-edge-connected in G.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    s : node
        Source node

    t : node
        Target node

    k : integer
        local edge connectivity for nodes s and t

    Returns
    -------
    boolean
        True if s and t are locally k-edge-connected in G.

    See Also
    --------
    :func:`is_k_edge_connected`

    Examples
    --------
    >>> from networkx.algorithms.connectivity import is_locally_k_edge_connected
    >>> G = nx.barbell_graph(10, 0)
    >>> is_locally_k_edge_connected(G, 5, 15, k=1)
    True
    >>> is_locally_k_edge_connected(G, 5, 15, k=2)
    False
    >>> is_locally_k_edge_connected(G, 1, 5, k=2)
    True
    """
    if k < 1:
        raise ValueError(f"k must be positive, not {k}")

    # First try to quickly determine s, t is not k-locally-edge-connected in G
    if G.degree(s) < k or G.degree(t) < k:
        return False
    else:
        # Otherwise perform the full check
        if k == 1:
            return nx.has_path(G, s, t)
        else:
            localk = nx.connectivity.local_edge_connectivity(G, s, t, cutoff=k)
            return localk >= k


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def k_edge_augmentation(G, k, avail=None, weight=None, partial=False):
    """Finds set of edges to k-edge-connect G.

    Adding edges from the augmentation to G make it impossible to disconnect G
    unless k or more edges are removed. This function uses the most efficient
    function available (depending on the value of k and if the problem is
    weighted or unweighted) to search for a minimum weight subset of available
    edges that k-edge-connects G. In general, finding a k-edge-augmentation is
    NP-hard, so solutions are not guaranteed to be minimal. Furthermore, a
    k-edge-augmentation may not exist.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    k : integer
        Desired edge connectivity

    avail : dict or a set of 2 or 3 tuples
        The available edges that can be used in the augmentation.

        If unspecified, then all edges in the complement of G are available.
        Otherwise, each item is an available edge (with an optional weight).

        In the unweighted case, each item is an edge ``(u, v)``.

        In the weighted case, each item is a 3-tuple ``(u, v, d)`` or a dict
        with items ``(u, v): d``.  The third item, ``d``, can be a dictionary
        or a real number.  If ``d`` is a dictionary ``d[weight]``
        correspondings to the weight.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples where the
        third item in each tuple is a dictionary.

    partial : boolean
        If partial is True and no feasible k-edge-augmentation exists, then all
        a partial k-edge-augmentation is generated. Adding the edges in a
        partial augmentation to G, minimizes the number of k-edge-connected
        components and maximizes the edge connectivity between those
        components. For details, see :func:`partial_k_edge_augmentation`.

    Yields
    ------
    edge : tuple
        Edges that, once added to G, would cause G to become k-edge-connected.
        If partial is False, an error is raised if this is not possible.
        Otherwise, generated edges form a partial augmentation, which
        k-edge-connects any part of G where it is possible, and maximally
        connects the remaining parts.

    Raises
    ------
    NetworkXUnfeasible
        If partial is False and no k-edge-augmentation exists.

    NetworkXNotImplemented
        If the input graph is directed or a multigraph.

    ValueError:
        If k is less than 1

    Notes
    -----
    When k=1 this returns an optimal solution.

    When k=2 and ``avail`` is None, this returns an optimal solution.
    Otherwise when k=2, this returns a 2-approximation of the optimal solution.

    For k>3, this problem is NP-hard and this uses a randomized algorithm that
        produces a feasible solution, but provides no guarantees on the
        solution weight.

    Examples
    --------
    >>> # Unweighted cases
    >>> G = nx.path_graph((1, 2, 3, 4))
    >>> G.add_node(5)
    >>> sorted(nx.k_edge_augmentation(G, k=1))
    [(1, 5)]
    >>> sorted(nx.k_edge_augmentation(G, k=2))
    [(1, 5), (5, 4)]
    >>> sorted(nx.k_edge_augmentation(G, k=3))
    [(1, 4), (1, 5), (2, 5), (3, 5), (4, 5)]
    >>> complement = list(nx.k_edge_augmentation(G, k=5, partial=True))
    >>> G.add_edges_from(complement)
    >>> nx.edge_connectivity(G)
    4

    >>> # Weighted cases
    >>> G = nx.path_graph((1, 2, 3, 4))
    >>> G.add_node(5)
    >>> # avail can be a tuple with a dict
    >>> avail = [(1, 5, {"weight": 11}), (2, 5, {"weight": 10})]
    >>> sorted(nx.k_edge_augmentation(G, k=1, avail=avail, weight="weight"))
    [(2, 5)]
    >>> # or avail can be a 3-tuple with a real number
    >>> avail = [(1, 5, 11), (2, 5, 10), (4, 3, 1), (4, 5, 51)]
    >>> sorted(nx.k_edge_augmentation(G, k=2, avail=avail))
    [(1, 5), (2, 5), (4, 5)]
    >>> # or avail can be a dict
    >>> avail = {(1, 5): 11, (2, 5): 10, (4, 3): 1, (4, 5): 51}
    >>> sorted(nx.k_edge_augmentation(G, k=2, avail=avail))
    [(1, 5), (2, 5), (4, 5)]
    >>> # If augmentation is infeasible, then a partial solution can be found
    >>> avail = {(1, 5): 11}
    >>> sorted(nx.k_edge_augmentation(G, k=2, avail=avail, partial=True))
    [(1, 5)]
    """
    try:
        if k <= 0:
            raise ValueError(f"k must be a positive integer, not {k}")
        elif G.number_of_nodes() < k + 1:
            msg = f"impossible to {k} connect in graph with less than {k + 1} nodes"
            raise nx.NetworkXUnfeasible(msg)
        elif avail is not None and len(avail) == 0:
            if not nx.is_k_edge_connected(G, k):
                raise nx.NetworkXUnfeasible("no available edges")
            aug_edges = []
        elif k == 1:
            aug_edges = one_edge_augmentation(
                G, avail=avail, weight=weight, partial=partial
            )
        elif k == 2:
            aug_edges = bridge_augmentation(G, avail=avail, weight=weight)
        else:
            # raise NotImplementedError(f'not implemented for k>2. k={k}')
            aug_edges = greedy_k_edge_augmentation(
                G, k=k, avail=avail, weight=weight, seed=0
            )
        # Do eager evaluation so we can catch any exceptions
        # Before executing partial code.
        yield from list(aug_edges)
    except nx.NetworkXUnfeasible:
        if partial:
            # Return all available edges
            if avail is None:
                aug_edges = complement_edges(G)
            else:
                # If we can't k-edge-connect the entire graph, try to
                # k-edge-connect as much as possible
                aug_edges = partial_k_edge_augmentation(
                    G, k=k, avail=avail, weight=weight
                )
            yield from aug_edges
        else:
            raise


@nx._dispatchable
def partial_k_edge_augmentation(G, k, avail, weight=None):
    """Finds augmentation that k-edge-connects as much of the graph as possible.

    When a k-edge-augmentation is not possible, we can still try to find a
    small set of edges that partially k-edge-connects as much of the graph as
    possible. All possible edges are generated between remaining parts.
    This minimizes the number of k-edge-connected subgraphs in the resulting
    graph and maximizes the edge connectivity between those subgraphs.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    k : integer
        Desired edge connectivity

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    Yields
    ------
    edge : tuple
        Edges in the partial augmentation of G. These edges k-edge-connect any
        part of G where it is possible, and maximally connects the remaining
        parts. In other words, all edges from avail are generated except for
        those within subgraphs that have already become k-edge-connected.

    Notes
    -----
    Construct H that augments G with all edges in avail.
    Find the k-edge-subgraphs of H.
    For each k-edge-subgraph, if the number of nodes is more than k, then find
    the k-edge-augmentation of that graph and add it to the solution. Then add
    all edges in avail between k-edge subgraphs to the solution.

    See Also
    --------
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> G.add_node(8)
    >>> avail = [(1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5), (1, 8)]
    >>> sorted(partial_k_edge_augmentation(G, k=2, avail=avail))
    [(1, 5), (1, 8)]
    """

    def _edges_between_disjoint(H, only1, only2):
        """finds edges between disjoint nodes"""
        only1_adj = {u: set(H.adj[u]) for u in only1}
        for u, neighbs in only1_adj.items():
            # Find the neighbors of u in only1 that are also in only2
            neighbs12 = neighbs.intersection(only2)
            for v in neighbs12:
                yield (u, v)

    avail_uv, avail_w = _unpack_available_edges(avail, weight=weight, G=G)

    # Find which parts of the graph can be k-edge-connected
    H = G.copy()
    H.add_edges_from(
        (
            (u, v, {"weight": w, "generator": (u, v)})
            for (u, v), w in zip(avail, avail_w)
        )
    )
    k_edge_subgraphs = list(nx.k_edge_subgraphs(H, k=k))

    # Generate edges to k-edge-connect internal subgraphs
    for nodes in k_edge_subgraphs:
        if len(nodes) > 1:
            # Get the k-edge-connected subgraph
            C = H.subgraph(nodes).copy()
            # Find the internal edges that were available
            sub_avail = {
                d["generator"]: d["weight"]
                for (u, v, d) in C.edges(data=True)
                if "generator" in d
            }
            # Remove potential augmenting edges
            C.remove_edges_from(sub_avail.keys())
            # Find a subset of these edges that makes the component
            # k-edge-connected and ignore the rest
            yield from nx.k_edge_augmentation(C, k=k, avail=sub_avail)

    # Generate all edges between CCs that could not be k-edge-connected
    for cc1, cc2 in it.combinations(k_edge_subgraphs, 2):
        for u, v in _edges_between_disjoint(H, cc1, cc2):
            d = H.get_edge_data(u, v)
            edge = d.get("generator", None)
            if edge is not None:
                yield edge


@not_implemented_for("multigraph")
@not_implemented_for("directed")
@nx._dispatchable
def one_edge_augmentation(G, avail=None, weight=None, partial=False):
    """Finds minimum weight set of edges to connect G.

    Equivalent to :func:`k_edge_augmentation` when k=1. Adding the resulting
    edges to G will make it 1-edge-connected. The solution is optimal for both
    weighted and non-weighted variants.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    partial : boolean
        If partial is True and no feasible k-edge-augmentation exists, then the
        augmenting edges minimize the number of connected components.

    Yields
    ------
    edge : tuple
        Edges in the one-augmentation of G

    Raises
    ------
    NetworkXUnfeasible
        If partial is False and no one-edge-augmentation exists.

    Notes
    -----
    Uses either :func:`unconstrained_one_edge_augmentation` or
    :func:`weighted_one_edge_augmentation` depending on whether ``avail`` is
    specified. Both algorithms are based on finding a minimum spanning tree.
    As such both algorithms find optimal solutions and run in linear time.

    See Also
    --------
    :func:`k_edge_augmentation`
    """
    if avail is None:
        return unconstrained_one_edge_augmentation(G)
    else:
        return weighted_one_edge_augmentation(
            G, avail=avail, weight=weight, partial=partial
        )


@not_implemented_for("multigraph")
@not_implemented_for("directed")
@nx._dispatchable
def bridge_augmentation(G, avail=None, weight=None):
    """Finds the a set of edges that bridge connects G.

    Equivalent to :func:`k_edge_augmentation` when k=2, and partial=False.
    Adding the resulting edges to G will make it 2-edge-connected.  If no
    constraints are specified the returned set of edges is minimum an optimal,
    otherwise the solution is approximated.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    Yields
    ------
    edge : tuple
        Edges in the bridge-augmentation of G

    Raises
    ------
    NetworkXUnfeasible
        If no bridge-augmentation exists.

    Notes
    -----
    If there are no constraints the solution can be computed in linear time
    using :func:`unconstrained_bridge_augmentation`. Otherwise, the problem
    becomes NP-hard and is the solution is approximated by
    :func:`weighted_bridge_augmentation`.

    See Also
    --------
    :func:`k_edge_augmentation`
    """
    if G.number_of_nodes() < 3:
        raise nx.NetworkXUnfeasible("impossible to bridge connect less than 3 nodes")
    if avail is None:
        return unconstrained_bridge_augmentation(G)
    else:
        return weighted_bridge_augmentation(G, avail, weight=weight)


# --- Algorithms and Helpers ---


def _ordered(u, v):
    """Returns the nodes in an undirected edge in lower-triangular order"""
    return (u, v) if u < v else (v, u)


def _unpack_available_edges(avail, weight=None, G=None):
    """Helper to separate avail into edges and corresponding weights"""
    if weight is None:
        weight = "weight"
    if isinstance(avail, dict):
        avail_uv = list(avail.keys())
        avail_w = list(avail.values())
    else:

        def _try_getitem(d):
            try:
                return d[weight]
            except TypeError:
                return d

        avail_uv = [tup[0:2] for tup in avail]
        avail_w = [1 if len(tup) == 2 else _try_getitem(tup[-1]) for tup in avail]

    if G is not None:
        # Edges already in the graph are filtered
        flags = [not G.has_edge(u, v) for u, v in avail_uv]
        avail_uv = list(it.compress(avail_uv, flags))
        avail_w = list(it.compress(avail_w, flags))
    return avail_uv, avail_w


MetaEdge = namedtuple("MetaEdge", ("meta_uv", "uv", "w"))


def _lightest_meta_edges(mapping, avail_uv, avail_w):
    """Maps available edges in the original graph to edges in the metagraph.

    Parameters
    ----------
    mapping : dict
        mapping produced by :func:`collapse`, that maps each node in the
        original graph to a node in the meta graph

    avail_uv : list
        list of edges

    avail_w : list
        list of edge weights

    Notes
    -----
    Each node in the metagraph is a k-edge-connected component in the original
    graph.  We don't care about any edge within the same k-edge-connected
    component, so we ignore self edges.  We also are only interested in the
    minimum weight edge bridging each k-edge-connected component so, we group
    the edges by meta-edge and take the lightest in each group.

    Examples
    --------
    >>> # Each group represents a meta-node
    >>> groups = ([1, 2, 3], [4, 5], [6])
    >>> mapping = {n: meta_n for meta_n, ns in enumerate(groups) for n in ns}
    >>> avail_uv = [(1, 2), (3, 6), (1, 4), (5, 2), (6, 1), (2, 6), (3, 1)]
    >>> avail_w = [20, 99, 20, 15, 50, 99, 20]
    >>> sorted(_lightest_meta_edges(mapping, avail_uv, avail_w))
    [MetaEdge(meta_uv=(0, 1), uv=(5, 2), w=15), MetaEdge(meta_uv=(0, 2), uv=(6, 1), w=50)]
    """
    grouped_wuv = defaultdict(list)
    for w, (u, v) in zip(avail_w, avail_uv):
        # Order the meta-edge so it can be used as a dict key
        meta_uv = _ordered(mapping[u], mapping[v])
        # Group each available edge using the meta-edge as a key
        grouped_wuv[meta_uv].append((w, u, v))

    # Now that all available edges are grouped, choose one per group
    for (mu, mv), choices_wuv in grouped_wuv.items():
        # Ignore available edges within the same meta-node
        if mu != mv:
            # Choose the lightest available edge belonging to each meta-edge
            w, u, v = min(choices_wuv)
            yield MetaEdge((mu, mv), (u, v), w)


@nx._dispatchable
def unconstrained_one_edge_augmentation(G):
    """Finds the smallest set of edges to connect G.

    This is a variant of the unweighted MST problem.
    If G is not empty, a feasible solution always exists.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    Yields
    ------
    edge : tuple
        Edges in the one-edge-augmentation of G

    See Also
    --------
    :func:`one_edge_augmentation`
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3), (4, 5)])
    >>> G.add_nodes_from([6, 7, 8])
    >>> sorted(unconstrained_one_edge_augmentation(G))
    [(1, 4), (4, 6), (6, 7), (7, 8)]
    """
    ccs1 = list(nx.connected_components(G))
    C = collapse(G, ccs1)
    # When we are not constrained, we can just make a meta graph tree.
    meta_nodes = list(C.nodes())
    # build a path in the metagraph
    meta_aug = list(zip(meta_nodes, meta_nodes[1:]))
    # map that path to the original graph
    inverse = defaultdict(list)
    for k, v in C.graph["mapping"].items():
        inverse[v].append(k)
    for mu, mv in meta_aug:
        yield (inverse[mu][0], inverse[mv][0])


@nx._dispatchable
def weighted_one_edge_augmentation(G, avail, weight=None, partial=False):
    """Finds the minimum weight set of edges to connect G if one exists.

    This is a variant of the weighted MST problem.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    partial : boolean
        If partial is True and no feasible k-edge-augmentation exists, then the
        augmenting edges minimize the number of connected components.

    Yields
    ------
    edge : tuple
        Edges in the subset of avail chosen to connect G.

    See Also
    --------
    :func:`one_edge_augmentation`
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3), (4, 5)])
    >>> G.add_nodes_from([6, 7, 8])
    >>> # any edge not in avail has an implicit weight of infinity
    >>> avail = [(1, 3), (1, 5), (4, 7), (4, 8), (6, 1), (8, 1), (8, 2)]
    >>> sorted(weighted_one_edge_augmentation(G, avail))
    [(1, 5), (4, 7), (6, 1), (8, 1)]
    >>> # find another solution by giving large weights to edges in the
    >>> # previous solution (note some of the old edges must be used)
    >>> avail = [(1, 3), (1, 5, 99), (4, 7, 9), (6, 1, 99), (8, 1, 99), (8, 2)]
    >>> sorted(weighted_one_edge_augmentation(G, avail))
    [(1, 5), (4, 7), (6, 1), (8, 2)]
    """
    avail_uv, avail_w = _unpack_available_edges(avail, weight=weight, G=G)
    # Collapse CCs in the original graph into nodes in a metagraph
    # Then find an MST of the metagraph instead of the original graph
    C = collapse(G, nx.connected_components(G))
    mapping = C.graph["mapping"]
    # Assign each available edge to an edge in the metagraph
    candidate_mapping = _lightest_meta_edges(mapping, avail_uv, avail_w)
    # nx.set_edge_attributes(C, name='weight', values=0)
    C.add_edges_from(
        (mu, mv, {"weight": w, "generator": uv})
        for (mu, mv), uv, w in candidate_mapping
    )
    # Find MST of the meta graph
    meta_mst = nx.minimum_spanning_tree(C)
    if not partial and not nx.is_connected(meta_mst):
        raise nx.NetworkXUnfeasible("Not possible to connect G with available edges")
    # Yield the edge that generated the meta-edge
    for mu, mv, d in meta_mst.edges(data=True):
        if "generator" in d:
            edge = d["generator"]
            yield edge


@nx._dispatchable
def unconstrained_bridge_augmentation(G):
    """Finds an optimal 2-edge-augmentation of G using the fewest edges.

    This is an implementation of the algorithm detailed in [1]_.
    The basic idea is to construct a meta-graph of bridge-ccs, connect leaf
    nodes of the trees to connect the entire graph, and finally connect the
    leafs of the tree in dfs-preorder to bridge connect the entire graph.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    Yields
    ------
    edge : tuple
        Edges in the bridge augmentation of G

    Notes
    -----
    Input: a graph G.
    First find the bridge components of G and collapse each bridge-cc into a
    node of a metagraph graph C, which is guaranteed to be a forest of trees.

    C contains p "leafs" --- nodes with exactly one incident edge.
    C contains q "isolated nodes" --- nodes with no incident edges.

    Theorem: If p + q > 1, then at least :math:`ceil(p / 2) + q` edges are
        needed to bridge connect C. This algorithm achieves this min number.

    The method first adds enough edges to make G into a tree and then pairs
    leafs in a simple fashion.

    Let n be the number of trees in C. Let v(i) be an isolated vertex in the
    i-th tree if one exists, otherwise it is a pair of distinct leafs nodes
    in the i-th tree. Alternating edges from these sets (i.e.  adding edges
    A1 = [(v(i)[0], v(i + 1)[1]), v(i + 1)[0], v(i + 2)[1])...]) connects C
    into a tree T. This tree has p' = p + 2q - 2(n -1) leafs and no isolated
    vertices. A1 has n - 1 edges. The next step finds ceil(p' / 2) edges to
    biconnect any tree with p' leafs.

    Convert T into an arborescence T' by picking an arbitrary root node with
    degree >= 2 and directing all edges away from the root. Note the
    implementation implicitly constructs T'.

    The leafs of T are the nodes with no existing edges in T'.
    Order the leafs of T' by DFS preorder. Then break this list in half
    and add the zipped pairs to A2.

    The set A = A1 + A2 is the minimum augmentation in the metagraph.

    To convert this to edges in the original graph

    References
    ----------
    .. [1] Eswaran, Kapali P., and R. Endre Tarjan. (1975) Augmentation problems.
        http://epubs.siam.org/doi/abs/10.1137/0205044

    See Also
    --------
    :func:`bridge_augmentation`
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> sorted(unconstrained_bridge_augmentation(G))
    [(1, 7)]
    >>> G = nx.path_graph((1, 2, 3, 2, 4, 5, 6, 7))
    >>> sorted(unconstrained_bridge_augmentation(G))
    [(1, 3), (3, 7)]
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2)])
    >>> G.add_node(4)
    >>> sorted(unconstrained_bridge_augmentation(G))
    [(1, 4), (4, 0)]
    """
    # -----
    # Mapping of terms from (Eswaran and Tarjan):
    #     G = G_0 - the input graph
    #     C = G_0' - the bridge condensation of G. (This is a forest of trees)
    #     A1 = A_1 - the edges to connect the forest into a tree
    #         leaf = pendant - a node with degree of 1

    #     alpha(v) = maps the node v in G to its meta-node in C
    #     beta(x) = maps the meta-node x in C to any node in the bridge
    #         component of G corresponding to x.

    # find the 2-edge-connected components of G
    bridge_ccs = list(nx.connectivity.bridge_components(G))
    # condense G into an forest C
    C = collapse(G, bridge_ccs)

    # Choose pairs of distinct leaf nodes in each tree. If this is not
    # possible then make a pair using the single isolated node in the tree.
    vset1 = [
        tuple(cc) * 2  # case1: an isolated node
        if len(cc) == 1
        else sorted(cc, key=C.degree)[0:2]  # case2: pair of leaf nodes
        for cc in nx.connected_components(C)
    ]
    if len(vset1) > 1:
        # Use this set to construct edges that connect C into a tree.
        nodes1 = [vs[0] for vs in vset1]
        nodes2 = [vs[1] for vs in vset1]
        A1 = list(zip(nodes1[1:], nodes2))
    else:
        A1 = []
    # Connect each tree in the forest to construct an arborescence
    T = C.copy()
    T.add_edges_from(A1)

    # If there are only two leaf nodes, we simply connect them.
    leafs = [n for n, d in T.degree() if d == 1]
    if len(leafs) == 1:
        A2 = []
    if len(leafs) == 2:
        A2 = [tuple(leafs)]
    else:
        # Choose an arbitrary non-leaf root
        try:
            root = next(n for n, d in T.degree() if d > 1)
        except StopIteration:  # no nodes found with degree > 1
            return
        # order the leaves of C by (induced directed) preorder
        v2 = [n for n in nx.dfs_preorder_nodes(T, root) if T.degree(n) == 1]
        # connecting first half of the leafs in pre-order to the second
        # half will bridge connect the tree with the fewest edges.
        half = math.ceil(len(v2) / 2)
        A2 = list(zip(v2[:half], v2[-half:]))

    # collect the edges used to augment the original forest
    aug_tree_edges = A1 + A2

    # Construct the mapping (beta) from meta-nodes to regular nodes
    inverse = defaultdict(list)
    for k, v in C.graph["mapping"].items():
        inverse[v].append(k)
    # sort so we choose minimum degree nodes first
    inverse = {
        mu: sorted(mapped, key=lambda u: (G.degree(u), u))
        for mu, mapped in inverse.items()
    }

    # For each meta-edge, map back to an arbitrary pair in the original graph
    G2 = G.copy()
    for mu, mv in aug_tree_edges:
        # Find the first available edge that doesn't exist and return it
        for u, v in it.product(inverse[mu], inverse[mv]):
            if not G2.has_edge(u, v):
                G2.add_edge(u, v)
                yield u, v
                break


@nx._dispatchable
def weighted_bridge_augmentation(G, avail, weight=None):
    """Finds an approximate min-weight 2-edge-augmentation of G.

    This is an implementation of the approximation algorithm detailed in [1]_.
    It chooses a set of edges from avail to add to G that renders it
    2-edge-connected if such a subset exists.  This is done by finding a
    minimum spanning arborescence of a specially constructed metagraph.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    avail : set of 2 or 3 tuples.
        candidate edges (with optional weights) to choose from

    weight : string
        key to use to find weights if avail is a set of 3-tuples where the
        third item in each tuple is a dictionary.

    Yields
    ------
    edge : tuple
        Edges in the subset of avail chosen to bridge augment G.

    Notes
    -----
    Finding a weighted 2-edge-augmentation is NP-hard.
    Any edge not in ``avail`` is considered to have a weight of infinity.
    The approximation factor is 2 if ``G`` is connected and 3 if it is not.
    Runs in :math:`O(m + n log(n))` time

    References
    ----------
    .. [1] Khuller, Samir, and Ramakrishna Thurimella. (1993) Approximation
        algorithms for graph augmentation.
        http://www.sciencedirect.com/science/article/pii/S0196677483710102

    See Also
    --------
    :func:`bridge_augmentation`
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4))
    >>> # When the weights are equal, (1, 4) is the best
    >>> avail = [(1, 4, 1), (1, 3, 1), (2, 4, 1)]
    >>> sorted(weighted_bridge_augmentation(G, avail))
    [(1, 4)]
    >>> # Giving (1, 4) a high weight makes the two edge solution the best.
    >>> avail = [(1, 4, 1000), (1, 3, 1), (2, 4, 1)]
    >>> sorted(weighted_bridge_augmentation(G, avail))
    [(1, 3), (2, 4)]
    >>> # ------
    >>> G = nx.path_graph((1, 2, 3, 4))
    >>> G.add_node(5)
    >>> avail = [(1, 5, 11), (2, 5, 10), (4, 3, 1), (4, 5, 1)]
    >>> sorted(weighted_bridge_augmentation(G, avail=avail))
    [(1, 5), (4, 5)]
    >>> avail = [(1, 5, 11), (2, 5, 10), (4, 3, 1), (4, 5, 51)]
    >>> sorted(weighted_bridge_augmentation(G, avail=avail))
    [(1, 5), (2, 5), (4, 5)]
    """

    if weight is None:
        weight = "weight"

    # If input G is not connected the approximation factor increases to 3
    if not nx.is_connected(G):
        H = G.copy()
        connectors = list(one_edge_augmentation(H, avail=avail, weight=weight))
        H.add_edges_from(connectors)

        yield from connectors
    else:
        connectors = []
        H = G

    if len(avail) == 0:
        if nx.has_bridges(H):
            raise nx.NetworkXUnfeasible("no augmentation possible")

    avail_uv, avail_w = _unpack_available_edges(avail, weight=weight, G=H)

    # Collapse input into a metagraph. Meta nodes are bridge-ccs
    bridge_ccs = nx.connectivity.bridge_components(H)
    C = collapse(H, bridge_ccs)

    # Use the meta graph to shrink avail to a small feasible subset
    mapping = C.graph["mapping"]
    # Choose the minimum weight feasible edge in each group
    meta_to_wuv = {
        (mu, mv): (w, uv)
        for (mu, mv), uv, w in _lightest_meta_edges(mapping, avail_uv, avail_w)
    }

    # Mapping of terms from (Khuller and Thurimella):
    #     C         : G_0 = (V, E^0)
    #        This is the metagraph where each node is a 2-edge-cc in G.
    #        The edges in C represent bridges in the original graph.
    #     (mu, mv)  : E - E^0  # they group both avail and given edges in E
    #     T         : \Gamma
    #     D         : G^D = (V, E_D)

    #     The paper uses ancestor because children point to parents, which is
    #     contrary to networkx standards.  So, we actually need to run
    #     nx.least_common_ancestor on the reversed Tree.

    # Pick an arbitrary leaf from C as the root
    try:
        root = next(n for n, d in C.degree() if d == 1)
    except StopIteration:  # no nodes found with degree == 1
        return
    # Root C into a tree TR by directing all edges away from the root
    # Note in their paper T directs edges towards the root
    TR = nx.dfs_tree(C, root)

    # Add to D the directed edges of T and set their weight to zero
    # This indicates that it costs nothing to use edges that were given.
    D = nx.reverse(TR).copy()

    nx.set_edge_attributes(D, name="weight", values=0)

    # The LCA of mu and mv in T is the shared ancestor of mu and mv that is
    # located farthest from the root.
    lca_gen = nx.tree_all_pairs_lowest_common_ancestor(
        TR, root=root, pairs=meta_to_wuv.keys()
    )

    for (mu, mv), lca in lca_gen:
        w, uv = meta_to_wuv[(mu, mv)]
        if lca == mu:
            # If u is an ancestor of v in TR, then add edge u->v to D
            D.add_edge(lca, mv, weight=w, generator=uv)
        elif lca == mv:
            # If v is an ancestor of u in TR, then add edge v->u to D
            D.add_edge(lca, mu, weight=w, generator=uv)
        else:
            # If neither u nor v is a ancestor of the other in TR
            # let t = lca(TR, u, v) and add edges t->u and t->v
            # Track the original edge that GENERATED these edges.
            D.add_edge(lca, mu, weight=w, generator=uv)
            D.add_edge(lca, mv, weight=w, generator=uv)

    # Then compute a minimum rooted branching
    try:
        # Note the original edges must be directed towards to root for the
        # branching to give us a bridge-augmentation.
        A = _minimum_rooted_branching(D, root)
    except nx.NetworkXException as err:
        # If there is no branching then augmentation is not possible
        raise nx.NetworkXUnfeasible("no 2-edge-augmentation possible") from err

    # For each edge e, in the branching that did not belong to the directed
    # tree T, add the corresponding edge that **GENERATED** it (this is not
    # necessarily e itself!)

    # ensure the third case does not generate edges twice
    bridge_connectors = set()
    for mu, mv in A.edges():
        data = D.get_edge_data(mu, mv)
        if "generator" in data:
            # Add the avail edge that generated the branching edge.
            edge = data["generator"]
            bridge_connectors.add(edge)

    yield from bridge_connectors


def _minimum_rooted_branching(D, root):
    """Helper function to compute a minimum rooted branching (aka rooted
    arborescence)

    Before the branching can be computed, the directed graph must be rooted by
    removing the predecessors of root.

    A branching / arborescence of rooted graph G is a subgraph that contains a
    directed path from the root to every other vertex. It is the directed
    analog of the minimum spanning tree problem.

    References
    ----------
    [1] Khuller, Samir (2002) Advanced Algorithms Lecture 24 Notes.
    https://web.archive.org/web/20121030033722/https://www.cs.umd.edu/class/spring2011/cmsc651/lec07.pdf
    """
    rooted = D.copy()
    # root the graph by removing all predecessors to `root`.
    rooted.remove_edges_from([(u, root) for u in D.predecessors(root)])
    # Then compute the branching / arborescence.
    A = nx.minimum_spanning_arborescence(rooted)
    return A


@nx._dispatchable(returns_graph=True)
def collapse(G, grouped_nodes):
    """Collapses each group of nodes into a single node.

    This is similar to condensation, but works on undirected graphs.

    Parameters
    ----------
    G : NetworkX Graph

    grouped_nodes:  list or generator
       Grouping of nodes to collapse. The grouping must be disjoint.
       If grouped_nodes are strongly_connected_components then this is
       equivalent to :func:`condensation`.

    Returns
    -------
    C : NetworkX Graph
       The collapsed graph C of G with respect to the node grouping.  The node
       labels are integers corresponding to the index of the component in the
       list of grouped_nodes.  C has a graph attribute named 'mapping' with a
       dictionary mapping the original nodes to the nodes in C to which they
       belong.  Each node in C also has a node attribute 'members' with the set
       of original nodes in G that form the group that the node in C
       represents.

    Examples
    --------
    >>> # Collapses a graph using disjoint groups, but not necessarily connected
    >>> G = nx.Graph([(1, 0), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (5, 7)])
    >>> G.add_node("A")
    >>> grouped_nodes = [{0, 1, 2, 3}, {5, 6, 7}]
    >>> C = collapse(G, grouped_nodes)
    >>> members = nx.get_node_attributes(C, "members")
    >>> sorted(members.keys())
    [0, 1, 2, 3]
    >>> member_values = set(map(frozenset, members.values()))
    >>> assert {0, 1, 2, 3} in member_values
    >>> assert {4} in member_values
    >>> assert {5, 6, 7} in member_values
    >>> assert {"A"} in member_values
    """
    mapping = {}
    members = {}
    C = G.__class__()
    i = 0  # required if G is empty
    remaining = set(G.nodes())
    for i, group in enumerate(grouped_nodes):
        group = set(group)
        assert remaining.issuperset(
            group
        ), "grouped nodes must exist in G and be disjoint"
        remaining.difference_update(group)
        members[i] = group
        mapping.update((n, i) for n in group)
    # remaining nodes are in their own group
    for i, node in enumerate(remaining, start=i + 1):
        group = {node}
        members[i] = group
        mapping.update((n, i) for n in group)
    number_of_groups = i + 1
    C.add_nodes_from(range(number_of_groups))
    C.add_edges_from(
        (mapping[u], mapping[v]) for u, v in G.edges() if mapping[u] != mapping[v]
    )
    # Add a list of members (ie original nodes) to each node (ie scc) in C.
    nx.set_node_attributes(C, name="members", values=members)
    # Add mapping dict as graph attribute
    C.graph["mapping"] = mapping
    return C


@nx._dispatchable
def complement_edges(G):
    """Returns only the edges in the complement of G

    Parameters
    ----------
    G : NetworkX Graph

    Yields
    ------
    edge : tuple
        Edges in the complement of G

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4))
    >>> sorted(complement_edges(G))
    [(1, 3), (1, 4), (2, 4)]
    >>> G = nx.path_graph((1, 2, 3, 4), nx.DiGraph())
    >>> sorted(complement_edges(G))
    [(1, 3), (1, 4), (2, 1), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)]
    >>> G = nx.complete_graph(1000)
    >>> sorted(complement_edges(G))
    []
    """
    G_adj = G._adj  # Store as a variable to eliminate attribute lookup
    if G.is_directed():
        for u, v in it.combinations(G.nodes(), 2):
            if v not in G_adj[u]:
                yield (u, v)
            if u not in G_adj[v]:
                yield (v, u)
    else:
        for u, v in it.combinations(G.nodes(), 2):
            if v not in G_adj[u]:
                yield (u, v)


def _compat_shuffle(rng, input):
    """wrapper around rng.shuffle for python 2 compatibility reasons"""
    rng.shuffle(input)


@not_implemented_for("multigraph")
@not_implemented_for("directed")
@py_random_state(4)
@nx._dispatchable
def greedy_k_edge_augmentation(G, k, avail=None, weight=None, seed=None):
    """Greedy algorithm for finding a k-edge-augmentation

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    k : integer
        Desired edge connectivity

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Yields
    ------
    edge : tuple
        Edges in the greedy augmentation of G

    Notes
    -----
    The algorithm is simple. Edges are incrementally added between parts of the
    graph that are not yet locally k-edge-connected. Then edges are from the
    augmenting set are pruned as long as local-edge-connectivity is not broken.

    This algorithm is greedy and does not provide optimality guarantees. It
    exists only to provide :func:`k_edge_augmentation` with the ability to
    generate a feasible solution for arbitrary k.

    See Also
    --------
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> sorted(greedy_k_edge_augmentation(G, k=2))
    [(1, 7)]
    >>> sorted(greedy_k_edge_augmentation(G, k=1, avail=[]))
    []
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> avail = {(u, v): 1 for (u, v) in complement_edges(G)}
    >>> # randomized pruning process can produce different solutions
    >>> sorted(greedy_k_edge_augmentation(G, k=4, avail=avail, seed=2))
    [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 6), (3, 7), (5, 7)]
    >>> sorted(greedy_k_edge_augmentation(G, k=4, avail=avail, seed=3))
    [(1, 3), (1, 5), (1, 6), (2, 4), (2, 6), (3, 7), (4, 7), (5, 7)]
    """
    # Result set
    aug_edges = []

    done = is_k_edge_connected(G, k)
    if done:
        return
    if avail is None:
        # all edges are available
        avail_uv = list(complement_edges(G))
        avail_w = [1] * len(avail_uv)
    else:
        # Get the unique set of unweighted edges
        avail_uv, avail_w = _unpack_available_edges(avail, weight=weight, G=G)

    # Greedy: order lightest edges. Use degree sum to tie-break
    tiebreaker = [sum(map(G.degree, uv)) for uv in avail_uv]
    avail_wduv = sorted(zip(avail_w, tiebreaker, avail_uv))
    avail_uv = [uv for w, d, uv in avail_wduv]

    # Incrementally add edges in until we are k-connected
    H = G.copy()
    for u, v in avail_uv:
        done = False
        if not is_locally_k_edge_connected(H, u, v, k=k):
            # Only add edges in parts that are not yet locally k-edge-connected
            aug_edges.append((u, v))
            H.add_edge(u, v)
            # Did adding this edge help?
            if H.degree(u) >= k and H.degree(v) >= k:
                done = is_k_edge_connected(H, k)
        if done:
            break

    # Check for feasibility
    if not done:
        raise nx.NetworkXUnfeasible("not able to k-edge-connect with available edges")

    # Randomized attempt to reduce the size of the solution
    _compat_shuffle(seed, aug_edges)
    for u, v in list(aug_edges):
        # Don't remove if we know it would break connectivity
        if H.degree(u) <= k or H.degree(v) <= k:
            continue
        H.remove_edge(u, v)
        aug_edges.remove((u, v))
        if not is_k_edge_connected(H, k=k):
            # If removing this edge breaks feasibility, undo
            H.add_edge(u, v)
            aug_edges.append((u, v))

    # Generate results
    yield from aug_edges
<|endoftext|>"""
Utilities for connectivity package
"""

import networkx as nx

__all__ = ["build_auxiliary_node_connectivity", "build_auxiliary_edge_connectivity"]


@nx._dispatchable(returns_graph=True)
def build_auxiliary_node_connectivity(G):
    r"""Creates a directed graph D from an undirected graph G to compute flow
    based node connectivity.

    For an undirected graph G having `n` nodes and `m` edges we derive a
    directed graph D with `2n` nodes and `2m+n` arcs by replacing each
    original node `v` with two nodes `vA`, `vB` linked by an (internal)
    arc in D. Then for each edge (`u`, `v`) in G we add two arcs (`uB`, `vA`)
    and (`vB`, `uA`) in D. Finally we set the attribute capacity = 1 for each
    arc in D [1]_.

    For a directed graph having `n` nodes and `m` arcs we derive a
    directed graph D with `2n` nodes and `m+n` arcs by replacing each
    original node `v` with two nodes `vA`, `vB` linked by an (internal)
    arc (`vA`, `vB`) in D. Then for each arc (`u`, `v`) in G we add one
    arc (`uB`, `vA`) in D. Finally we set the attribute capacity = 1 for
    each arc in D.

    A dictionary with a mapping between nodes in the original graph and the
    auxiliary digraph is stored as a graph attribute: D.graph['mapping'].

    References
    ----------
    .. [1] Kammer, Frank and Hanjo Taubig. Graph Connectivity. in Brandes and
        Erlebach, 'Network Analysis: Methodological Foundations', Lecture
        Notes in Computer Science, Volume 3418, Springer-Verlag, 2005.
        https://doi.org/10.1007/978-3-540-31955-9_7

    """
    directed = G.is_directed()

    mapping = {}
    H = nx.DiGraph()

    for i, node in enumerate(G):
        mapping[node] = i
        H.add_node(f"{i}A", id=node)
        H.add_node(f"{i}B", id=node)
        H.add_edge(f"{i}A", f"{i}B", capacity=1)

    edges = []
    for source, target in G.edges():
        edges.append((f"{mapping[source]}B", f"{mapping[target]}A"))
        if not directed:
            edges.append((f"{mapping[target]}B", f"{mapping[source]}A"))
    H.add_edges_from(edges, capacity=1)

    # Store mapping as graph attribute
    H.graph["mapping"] = mapping
    return H


@nx._dispatchable(returns_graph=True)
def build_auxiliary_edge_connectivity(G):
    """Auxiliary digraph for computing flow based edge connectivity

    If the input graph is undirected, we replace each edge (`u`,`v`) with
    two reciprocal arcs (`u`, `v`) and (`v`, `u`) and then we set the attribute
    'capacity' for each arc to 1. If the input graph is directed we simply
    add the 'capacity' attribute. Part of algorithm 1 in [1]_ .

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms. (this is a
        chapter, look for the reference of the book).
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf
    """
    if G.is_directed():
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges(), capacity=1)
        return H
    else:
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        for source, target in G.edges():
            H.add_edges_from([(source, target), (target, source)], capacity=1)
        return H
<|endoftext|>"""Connectivity and cut algorithms"""

from .connectivity import *
from .cuts import *
from .edge_augmentation import *
from .edge_kcomponents import *
from .disjoint_paths import *
from .kcomponents import *
from .kcutsets import *
from .stoerwagner import *
from .utils import *
<|endoftext|>"""
Stoer-Wagner minimum cut algorithm.
"""

from itertools import islice

import networkx as nx

from ...utils import BinaryHeap, arbitrary_element, not_implemented_for

__all__ = ["stoer_wagner"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def stoer_wagner(G, weight="weight", heap=BinaryHeap):
    r"""Returns the weighted minimum edge cut using the Stoer-Wagner algorithm.

    Determine the minimum edge cut of a connected graph using the
    Stoer-Wagner algorithm. In weighted cases, all weights must be
    nonnegative.

    The running time of the algorithm depends on the type of heaps used:

    ============== =============================================
    Type of heap   Running time
    ============== =============================================
    Binary heap    $O(n (m + n) \log n)$
    Fibonacci heap $O(nm + n^2 \log n)$
    Pairing heap   $O(2^{2 \sqrt{\log \log n}} nm + n^2 \log n)$
    ============== =============================================

    Parameters
    ----------
    G : NetworkX graph
        Edges of the graph are expected to have an attribute named by the
        weight parameter below. If this attribute is not present, the edge is
        considered to have unit weight.

    weight : string
        Name of the weight attribute of the edges. If the attribute is not
        present, unit weight is assumed. Default value: 'weight'.

    heap : class
        Type of heap to be used in the algorithm. It should be a subclass of
        :class:`MinHeap` or implement a compatible interface.

        If a stock heap implementation is to be used, :class:`BinaryHeap` is
        recommended over :class:`PairingHeap` for Python implementations without
        optimized attribute accesses (e.g., CPython) despite a slower
        asymptotic running time. For Python implementations with optimized
        attribute accesses (e.g., PyPy), :class:`PairingHeap` provides better
        performance. Default value: :class:`BinaryHeap`.

    Returns
    -------
    cut_value : integer or float
        The sum of weights of edges in a minimum cut.

    partition : pair of node lists
        A partitioning of the nodes that defines a minimum cut.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or a multigraph.

    NetworkXError
        If the graph has less than two nodes, is not connected or has a
        negative-weighted edge.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge("x", "a", weight=3)
    >>> G.add_edge("x", "b", weight=1)
    >>> G.add_edge("a", "c", weight=3)
    >>> G.add_edge("b", "c", weight=5)
    >>> G.add_edge("b", "d", weight=4)
    >>> G.add_edge("d", "e", weight=2)
    >>> G.add_edge("c", "y", weight=2)
    >>> G.add_edge("e", "y", weight=3)
    >>> cut_value, partition = nx.stoer_wagner(G)
    >>> cut_value
    4
    """
    n = len(G)
    if n < 2:
        raise nx.NetworkXError("graph has less than two nodes.")
    if not nx.is_connected(G):
        raise nx.NetworkXError("graph is not connected.")

    # Make a copy of the graph for internal use.
    G = nx.Graph(
        (u, v, {"weight": e.get(weight, 1)}) for u, v, e in G.edges(data=True) if u != v
    )
    G.__networkx_cache__ = None  # Disable caching

    for u, v, e in G.edges(data=True):
        if e["weight"] < 0:
            raise nx.NetworkXError("graph has a negative-weighted edge.")

    cut_value = float("inf")
    nodes = set(G)
    contractions = []  # contracted node pairs

    # Repeatedly pick a pair of nodes to contract until only one node is left.
    for i in range(n - 1):
        # Pick an arbitrary node u and create a set A = {u}.
        u = arbitrary_element(G)
        A = {u}
        # Repeatedly pick the node "most tightly connected" to A and add it to
        # A. The tightness of connectivity of a node not in A is defined by the
        # of edges connecting it to nodes in A.
        h = heap()  # min-heap emulating a max-heap
        for v, e in G[u].items():
            h.insert(v, -e["weight"])
        # Repeat until all but one node has been added to A.
        for j in range(n - i - 2):
            u = h.pop()[0]
            A.add(u)
            for v, e in G[u].items():
                if v not in A:
                    h.insert(v, h.get(v, 0) - e["weight"])
        # A and the remaining node v define a "cut of the phase". There is a
        # minimum cut of the original graph that is also a cut of the phase.
        # Due to contractions in earlier phases, v may in fact represent
        # multiple nodes in the original graph.
        v, w = h.min()
        w = -w
        if w < cut_value:
            cut_value = w
            best_phase = i
        # Contract v and the last node added to A.
        contractions.append((u, v))
        for w, e in G[v].items():
            if w != u:
                if w not in G[u]:
                    G.add_edge(u, w, weight=e["weight"])
                else:
                    G[u][w]["weight"] += e["weight"]
        G.remove_node(v)

    # Recover the optimal partitioning from the contractions.
    G = nx.Graph(islice(contractions, best_phase))
    v = contractions[best_phase][1]
    G.add_node(v)
    reachable = set(nx.single_source_shortest_path_length(G, v))
    partition = (list(reachable), list(nodes - reachable))

    return cut_value, partition
<|endoftext|>"""
Algorithms for finding k-edge-connected components and subgraphs.

A k-edge-connected component (k-edge-cc) is a maximal set of nodes in G, such
that all pairs of node have an edge-connectivity of at least k.

A k-edge-connected subgraph (k-edge-subgraph) is a maximal set of nodes in G,
such that the subgraph of G defined by the nodes has an edge-connectivity at
least k.
"""

import itertools as it
from functools import partial

import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for

__all__ = [
    "k_edge_components",
    "k_edge_subgraphs",
    "bridge_components",
    "EdgeComponentAuxGraph",
]


@not_implemented_for("multigraph")
@nx._dispatchable
def k_edge_components(G, k):
    """Generates nodes in each maximal k-edge-connected component in G.

    Parameters
    ----------
    G : NetworkX graph

    k : Integer
        Desired edge connectivity

    Returns
    -------
    k_edge_components : a generator of k-edge-ccs. Each set of returned nodes
       will have k-edge-connectivity in the graph G.

    See Also
    --------
    :func:`local_edge_connectivity`
    :func:`k_edge_subgraphs` : similar to this function, but the subgraph
        defined by the nodes must also have k-edge-connectivity.
    :func:`k_components` : similar to this function, but uses node-connectivity
        instead of edge-connectivity

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is a multigraph.

    ValueError:
        If k is less than 1

    Notes
    -----
    Attempts to use the most efficient implementation available based on k.
    If k=1, this is simply connected components for directed graphs and
    connected components for undirected graphs.
    If k=2 on an efficient bridge connected component algorithm from _[1] is
    run based on the chain decomposition.
    Otherwise, the algorithm from _[2] is used.

    Examples
    --------
    >>> import itertools as it
    >>> from networkx.utils import pairwise
    >>> paths = [
    ...     (1, 2, 4, 3, 1, 4),
    ...     (5, 6, 7, 8, 5, 7, 8, 6),
    ... ]
    >>> G = nx.Graph()
    >>> G.add_nodes_from(it.chain(*paths))
    >>> G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))
    >>> # note this returns {1, 4} unlike k_edge_subgraphs
    >>> sorted(map(sorted, nx.k_edge_components(G, k=3)))
    [[1, 4], [2], [3], [5, 6, 7, 8]]

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bridge_%28graph_theory%29
    .. [2] Wang, Tianhao, et al. (2015) A simple algorithm for finding all
        k-edge-connected components.
        http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136264
    """
    # Compute k-edge-ccs using the most efficient algorithms available.
    if k < 1:
        raise ValueError("k cannot be less than 1")
    if G.is_directed():
        if k == 1:
            return nx.strongly_connected_components(G)
        else:
            # TODO: investigate https://arxiv.org/abs/1412.6466 for k=2
            aux_graph = EdgeComponentAuxGraph.construct(G)
            return aux_graph.k_edge_components(k)
    else:
        if k == 1:
            return nx.connected_components(G)
        elif k == 2:
            return bridge_components(G)
        else:
            aux_graph = EdgeComponentAuxGraph.construct(G)
            return aux_graph.k_edge_components(k)


@not_implemented_for("multigraph")
@nx._dispatchable
def k_edge_subgraphs(G, k):
    """Generates nodes in each maximal k-edge-connected subgraph in G.

    Parameters
    ----------
    G : NetworkX graph

    k : Integer
        Desired edge connectivity

    Returns
    -------
    k_edge_subgraphs : a generator of k-edge-subgraphs
        Each k-edge-subgraph is a maximal set of nodes that defines a subgraph
        of G that is k-edge-connected.

    See Also
    --------
    :func:`edge_connectivity`
    :func:`k_edge_components` : similar to this function, but nodes only
        need to have k-edge-connectivity within the graph G and the subgraphs
        might not be k-edge-connected.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is a multigraph.

    ValueError:
        If k is less than 1

    Notes
    -----
    Attempts to use the most efficient implementation available based on k.
    If k=1, or k=2 and the graph is undirected, then this simply calls
    `k_edge_components`.  Otherwise the algorithm from _[1] is used.

    Examples
    --------
    >>> import itertools as it
    >>> from networkx.utils import pairwise
    >>> paths = [
    ...     (1, 2, 4, 3, 1, 4),
    ...     (5, 6, 7, 8, 5, 7, 8, 6),
    ... ]
    >>> G = nx.Graph()
    >>> G.add_nodes_from(it.chain(*paths))
    >>> G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))
    >>> # note this does not return {1, 4} unlike k_edge_components
    >>> sorted(map(sorted, nx.k_edge_subgraphs(G, k=3)))
    [[1], [2], [3], [4], [5, 6, 7, 8]]

    References
    ----------
    .. [1] Zhou, Liu, et al. (2012) Finding maximal k-edge-connected subgraphs
        from a large graph.  ACM International Conference on Extending Database
        Technology 2012 480-–491.
        https://openproceedings.org/2012/conf/edbt/ZhouLYLCL12.pdf
    """
    if k < 1:
        raise ValueError("k cannot be less than 1")
    if G.is_directed():
        if k <= 1:
            # For directed graphs ,
            # When k == 1, k-edge-ccs and k-edge-subgraphs are the same
            return k_edge_components(G, k)
        else:
            return _k_edge_subgraphs_nodes(G, k)
    else:
        if k <= 2:
            # For undirected graphs,
            # when k <= 2, k-edge-ccs and k-edge-subgraphs are the same
            return k_edge_components(G, k)
        else:
            return _k_edge_subgraphs_nodes(G, k)


def _k_edge_subgraphs_nodes(G, k):
    """Helper to get the nodes from the subgraphs.

    This allows k_edge_subgraphs to return a generator.
    """
    for C in general_k_edge_subgraphs(G, k):
        yield set(C.nodes())


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def bridge_components(G):
    """Finds all bridge-connected components G.

    Parameters
    ----------
    G : NetworkX undirected graph

    Returns
    -------
    bridge_components : a generator of 2-edge-connected components


    See Also
    --------
    :func:`k_edge_subgraphs` : this function is a special case for an
        undirected graph where k=2.
    :func:`biconnected_components` : similar to this function, but is defined
        using 2-node-connectivity instead of 2-edge-connectivity.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is directed or a multigraph.

    Notes
    -----
    Bridge-connected components are also known as 2-edge-connected components.

    Examples
    --------
    >>> # The barbell graph with parameter zero has a single bridge
    >>> G = nx.barbell_graph(5, 0)
    >>> from networkx.algorithms.connectivity.edge_kcomponents import bridge_components
    >>> sorted(map(sorted, bridge_components(G)))
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    """
    H = G.copy()
    H.remove_edges_from(nx.bridges(G))
    yield from nx.connected_components(H)


class EdgeComponentAuxGraph:
    r"""A simple algorithm to find all k-edge-connected components in a graph.

    Constructing the auxiliary graph (which may take some time) allows for the
    k-edge-ccs to be found in linear time for arbitrary k.

    Notes
    -----
    This implementation is based on [1]_. The idea is to construct an auxiliary
    graph from which the k-edge-ccs can be extracted in linear time. The
    auxiliary graph is constructed in $O(|V|\cdot F)$ operations, where F is the
    complexity of max flow. Querying the components takes an additional $O(|V|)$
    operations. This algorithm can be slow for large graphs, but it handles an
    arbitrary k and works for both directed and undirected inputs.

    The undirected case for k=1 is exactly connected components.
    The undirected case for k=2 is exactly bridge connected components.
    The directed case for k=1 is exactly strongly connected components.

    References
    ----------
    .. [1] Wang, Tianhao, et al. (2015) A simple algorithm for finding all
        k-edge-connected components.
        http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136264

    Examples
    --------
    >>> import itertools as it
    >>> from networkx.utils import pairwise
    >>> from networkx.algorithms.connectivity import EdgeComponentAuxGraph
    >>> # Build an interesting graph with multiple levels of k-edge-ccs
    >>> paths = [
    ...     (1, 2, 3, 4, 1, 3, 4, 2),  # a 3-edge-cc (a 4 clique)
    ...     (5, 6, 7, 5),  # a 2-edge-cc (a 3 clique)
    ...     (1, 5),  # combine first two ccs into a 1-edge-cc
    ...     (0,),  # add an additional disconnected 1-edge-cc
    ... ]
    >>> G = nx.Graph()
    >>> G.add_nodes_from(it.chain(*paths))
    >>> G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))
    >>> # Constructing the AuxGraph takes about O(n ** 4)
    >>> aux_graph = EdgeComponentAuxGraph.construct(G)
    >>> # Once constructed, querying takes O(n)
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=1)))
    [[0], [1, 2, 3, 4, 5, 6, 7]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=2)))
    [[0], [1, 2, 3, 4], [5, 6, 7]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=3)))
    [[0], [1, 2, 3, 4], [5], [6], [7]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=4)))
    [[0], [1], [2], [3], [4], [5], [6], [7]]

    The auxiliary graph is primarily used for k-edge-ccs but it
    can also speed up the queries of k-edge-subgraphs by refining the
    search space.

    >>> import itertools as it
    >>> from networkx.utils import pairwise
    >>> from networkx.algorithms.connectivity import EdgeComponentAuxGraph
    >>> paths = [
    ...     (1, 2, 4, 3, 1, 4),
    ... ]
    >>> G = nx.Graph()
    >>> G.add_nodes_from(it.chain(*paths))
    >>> G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))
    >>> aux_graph = EdgeComponentAuxGraph.construct(G)
    >>> sorted(map(sorted, aux_graph.k_edge_subgraphs(k=3)))
    [[1], [2], [3], [4]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=3)))
    [[1, 4], [2], [3]]
    """

    # @not_implemented_for('multigraph')  # TODO: fix decor for classmethods
    @classmethod
    def construct(EdgeComponentAuxGraph, G):
        """Builds an auxiliary graph encoding edge-connectivity between nodes.

        Notes
        -----
        Given G=(V, E), initialize an empty auxiliary graph A.
        Choose an arbitrary source node s.  Initialize a set N of available
        nodes (that can be used as the sink). The algorithm picks an
        arbitrary node t from N - {s}, and then computes the minimum st-cut
        (S, T) with value w. If G is directed the minimum of the st-cut or
        the ts-cut is used instead. Then, the edge (s, t) is added to the
        auxiliary graph with weight w. The algorithm is called recursively
        first using S as the available nodes and s as the source, and then
        using T and t. Recursion stops when the source is the only available
        node.

        Parameters
        ----------
        G : NetworkX graph
        """
        # workaround for classmethod decorator
        not_implemented_for("multigraph")(lambda G: G)(G)

        def _recursive_build(H, A, source, avail):
            # Terminate once the flow has been compute to every node.
            if {source} == avail:
                return
            # pick an arbitrary node as the sink
            sink = arbitrary_element(avail - {source})
            # find the minimum cut and its weight
            value, (S, T) = nx.minimum_cut(H, source, sink)
            if H.is_directed():
                # check if the reverse direction has a smaller cut
                value_, (T_, S_) = nx.minimum_cut(H, sink, source)
                if value_ < value:
                    value, S, T = value_, S_, T_
            # add edge with weight of cut to the aux graph
            A.add_edge(source, sink, weight=value)
            # recursively call until all but one node is used
            _recursive_build(H, A, source, avail.intersection(S))
            _recursive_build(H, A, sink, avail.intersection(T))

        # Copy input to ensure all edges have unit capacity
        H = G.__class__()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges(), capacity=1)

        # A is the auxiliary graph to be constructed
        # It is a weighted undirected tree
        A = nx.Graph()

        # Pick an arbitrary node as the source
        if H.number_of_nodes() > 0:
            source = arbitrary_element(H.nodes())
            # Initialize a set of elements that can be chosen as the sink
            avail = set(H.nodes())

            # This constructs A
            _recursive_build(H, A, source, avail)

        # This class is a container the holds the auxiliary graph A and
        # provides access the k_edge_components function.
        self = EdgeComponentAuxGraph()
        self.A = A
        self.H = H
        return self

    def k_edge_components(self, k):
        """Queries the auxiliary graph for k-edge-connected components.

        Parameters
        ----------
        k : Integer
            Desired edge connectivity

        Returns
        -------
        k_edge_components : a generator of k-edge-ccs

        Notes
        -----
        Given the auxiliary graph, the k-edge-connected components can be
        determined in linear time by removing all edges with weights less than
        k from the auxiliary graph.  The resulting connected components are the
        k-edge-ccs in the original graph.
        """
        if k < 1:
            raise ValueError("k cannot be less than 1")
        A = self.A
        # "traverse the auxiliary graph A and delete all edges with weights less
        # than k"
        aux_weights = nx.get_edge_attributes(A, "weight")
        # Create a relevant graph with the auxiliary edges with weights >= k
        R = nx.Graph()
        R.add_nodes_from(A.nodes())
        R.add_edges_from(e for e, w in aux_weights.items() if w >= k)

        # Return the nodes that are k-edge-connected in the original graph
        yield from nx.connected_components(R)

    def k_edge_subgraphs(self, k):
        """Queries the auxiliary graph for k-edge-connected subgraphs.

        Parameters
        ----------
        k : Integer
            Desired edge connectivity

        Returns
        -------
        k_edge_subgraphs : a generator of k-edge-subgraphs

        Notes
        -----
        Refines the k-edge-ccs into k-edge-subgraphs. The running time is more
        than $O(|V|)$.

        For single values of k it is faster to use `nx.k_edge_subgraphs`.
        But for multiple values of k, it can be faster to build AuxGraph and
        then use this method.
        """
        if k < 1:
            raise ValueError("k cannot be less than 1")
        H = self.H
        A = self.A
        # "traverse the auxiliary graph A and delete all edges with weights less
        # than k"
        aux_weights = nx.get_edge_attributes(A, "weight")
        # Create a relevant graph with the auxiliary edges with weights >= k
        R = nx.Graph()
        R.add_nodes_from(A.nodes())
        R.add_edges_from(e for e, w in aux_weights.items() if w >= k)

        # Return the components whose subgraphs are k-edge-connected
        for cc in nx.connected_components(R):
            if len(cc) < k:
                # Early return optimization
                for node in cc:
                    yield {node}
            else:
                # Call subgraph solution to refine the results
                C = H.subgraph(cc)
                yield from k_edge_subgraphs(C, k)


def _low_degree_nodes(G, k, nbunch=None):
    """Helper for finding nodes with degree less than k."""
    # Nodes with degree less than k cannot be k-edge-connected.
    if G.is_directed():
        # Consider both in and out degree in the directed case
        seen = set()
        for node, degree in G.out_degree(nbunch):
            if degree < k:
                seen.add(node)
                yield node
        for node, degree in G.in_degree(nbunch):
            if node not in seen and degree < k:
                seen.add(node)
                yield node
    else:
        # Only the degree matters in the undirected case
        for node, degree in G.degree(nbunch):
            if degree < k:
                yield node


def _high_degree_components(G, k):
    """Helper for filtering components that can't be k-edge-connected.

    Removes and generates each node with degree less than k.  Then generates
    remaining components where all nodes have degree at least k.
    """
    # Iteratively remove parts of the graph that are not k-edge-connected
    H = G.copy()
    singletons = set(_low_degree_nodes(H, k))
    while singletons:
        # Only search neighbors of removed nodes
        nbunch = set(it.chain.from_iterable(map(H.neighbors, singletons)))
        nbunch.difference_update(singletons)
        H.remove_nodes_from(singletons)
        for node in singletons:
            yield {node}
        singletons = set(_low_degree_nodes(H, k, nbunch))

    # Note: remaining connected components may not be k-edge-connected
    if G.is_directed():
        yield from nx.strongly_connected_components(H)
    else:
        yield from nx.connected_components(H)


@nx._dispatchable(returns_graph=True)
def general_k_edge_subgraphs(G, k):
    """General algorithm to find all maximal k-edge-connected subgraphs in `G`.

    Parameters
    ----------
    G : nx.Graph
       Graph in which all maximal k-edge-connected subgraphs will be found.

    k : int

    Yields
    ------
    k_edge_subgraphs : Graph instances that are k-edge-subgraphs
        Each k-edge-subgraph contains a maximal set of nodes that defines a
        subgraph of `G` that is k-edge-connected.

    Notes
    -----
    Implementation of the basic algorithm from [1]_.  The basic idea is to find
    a global minimum cut of the graph. If the cut value is at least k, then the
    graph is a k-edge-connected subgraph and can be added to the results.
    Otherwise, the cut is used to split the graph in two and the procedure is
    applied recursively. If the graph is just a single node, then it is also
    added to the results. At the end, each result is either guaranteed to be
    a single node or a subgraph of G that is k-edge-connected.

    This implementation contains optimizations for reducing the number of calls
    to max-flow, but there are other optimizations in [1]_ that could be
    implemented.

    References
    ----------
    .. [1] Zhou, Liu, et al. (2012) Finding maximal k-edge-connected subgraphs
        from a large graph.  ACM International Conference on Extending Database
        Technology 2012 480-–491.
        https://openproceedings.org/2012/conf/edbt/ZhouLYLCL12.pdf

    Examples
    --------
    >>> from networkx.utils import pairwise
    >>> paths = [
    ...     (11, 12, 13, 14, 11, 13, 14, 12),  # a 4-clique
    ...     (21, 22, 23, 24, 21, 23, 24, 22),  # another 4-clique
    ...     # connect the cliques with high degree but low connectivity
    ...     (50, 13),
    ...     (12, 50, 22),
    ...     (13, 102, 23),
    ...     (14, 101, 24),
    ... ]
    >>> G = nx.Graph(it.chain(*[pairwise(path) for path in paths]))
    >>> sorted(len(k_sg) for k_sg in k_edge_subgraphs(G, k=3))
    [1, 1, 1, 4, 4]
    """
    if k < 1:
        raise ValueError("k cannot be less than 1")

    # Node pruning optimization (incorporates early return)
    # find_ccs is either connected_components/strongly_connected_components
    find_ccs = partial(_high_degree_components, k=k)

    # Quick return optimization
    if G.number_of_nodes() < k:
        for node in G.nodes():
            yield G.subgraph([node]).copy()
        return

    # Intermediate results
    R0 = {G.subgraph(cc).copy() for cc in find_ccs(G)}
    # Subdivide CCs in the intermediate results until they are k-conn
    while R0:
        G1 = R0.pop()
        if G1.number_of_nodes() == 1:
            yield G1
        else:
            # Find a global minimum cut
            cut_edges = nx.minimum_edge_cut(G1)
            cut_value = len(cut_edges)
            if cut_value < k:
                # G1 is not k-edge-connected, so subdivide it
                G1.remove_edges_from(cut_edges)
                for cc in find_ccs(G1):
                    R0.add(G1.subgraph(cc).copy())
            else:
                # Otherwise we found a k-edge-connected subgraph
                yield G1
<|endoftext|>"""
Moody and White algorithm for k-components
"""

from collections import defaultdict
from itertools import combinations
from operator import itemgetter

import networkx as nx

# Define the default maximum flow function.
from networkx.algorithms.flow import edmonds_karp
from networkx.utils import not_implemented_for

default_flow_func = edmonds_karp

__all__ = ["k_components"]


@not_implemented_for("directed")
@nx._dispatchable
def k_components(G, flow_func=None):
    r"""Returns the k-component structure of a graph G.

    A `k`-component is a maximal subgraph of a graph G that has, at least,
    node connectivity `k`: we need to remove at least `k` nodes to break it
    into more components. `k`-components have an inherent hierarchical
    structure because they are nested in terms of connectivity: a connected
    graph can contain several 2-components, each of which can contain
    one or more 3-components, and so forth.

    Parameters
    ----------
    G : NetworkX graph

    flow_func : function
        Function to perform the underlying flow computations. Default value
        :meth:`edmonds_karp`. This function performs better in sparse graphs with
        right tailed degree distributions. :meth:`shortest_augmenting_path` will
        perform better in denser graphs.

    Returns
    -------
    k_components : dict
        Dictionary with all connectivity levels `k` in the input Graph as keys
        and a list of sets of nodes that form a k-component of level `k` as
        values.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is directed.

    Examples
    --------
    >>> # Petersen graph has 10 nodes and it is triconnected, thus all
    >>> # nodes are in a single component on all three connectivity levels
    >>> G = nx.petersen_graph()
    >>> k_components = nx.k_components(G)

    Notes
    -----
    Moody and White [1]_ (appendix A) provide an algorithm for identifying
    k-components in a graph, which is based on Kanevsky's algorithm [2]_
    for finding all minimum-size node cut-sets of a graph (implemented in
    :meth:`all_node_cuts` function):

        1. Compute node connectivity, k, of the input graph G.

        2. Identify all k-cutsets at the current level of connectivity using
           Kanevsky's algorithm.

        3. Generate new graph components based on the removal of
           these cutsets. Nodes in a cutset belong to both sides
           of the induced cut.

        4. If the graph is neither complete nor trivial, return to 1;
           else end.

    This implementation also uses some heuristics (see [3]_ for details)
    to speed up the computation.

    See also
    --------
    node_connectivity
    all_node_cuts
    biconnected_components : special case of this function when k=2
    k_edge_components : similar to this function, but uses edge-connectivity
        instead of node-connectivity

    References
    ----------
    .. [1]  Moody, J. and D. White (2003). Social cohesion and embeddedness:
            A hierarchical conception of social groups.
            American Sociological Review 68(1), 103--28.
            http://www2.asanet.org/journals/ASRFeb03MoodyWhite.pdf

    .. [2]  Kanevsky, A. (1993). Finding all minimum-size separating vertex
            sets in a graph. Networks 23(6), 533--541.
            http://onlinelibrary.wiley.com/doi/10.1002/net.3230230604/abstract

    .. [3]  Torrents, J. and F. Ferraro (2015). Structural Cohesion:
            Visualization and Heuristics for Fast Computation.
            https://arxiv.org/pdf/1503.04476v1

    """
    # Dictionary with connectivity level (k) as keys and a list of
    # sets of nodes that form a k-component as values. Note that
    # k-components can overlap (but only k - 1 nodes).
    k_components = defaultdict(list)
    # Define default flow function
    if flow_func is None:
        flow_func = default_flow_func
    # Bicomponents as a base to check for higher order k-components
    for component in nx.connected_components(G):
        # isolated nodes have connectivity 0
        comp = set(component)
        if len(comp) > 1:
            k_components[1].append(comp)
    bicomponents = [G.subgraph(c) for c in nx.biconnected_components(G)]
    for bicomponent in bicomponents:
        bicomp = set(bicomponent)
        # avoid considering dyads as bicomponents
        if len(bicomp) > 2:
            k_components[2].append(bicomp)
    for B in bicomponents:
        if len(B) <= 2:
            continue
        k = nx.node_connectivity(B, flow_func=flow_func)
        if k > 2:
            k_components[k].append(set(B))
        # Perform cuts in a DFS like order.
        cuts = list(nx.all_node_cuts(B, k=k, flow_func=flow_func))
        stack = [(k, _generate_partition(B, cuts, k))]
        while stack:
            (parent_k, partition) = stack[-1]
            try:
                nodes = next(partition)
                C = B.subgraph(nodes)
                this_k = nx.node_connectivity(C, flow_func=flow_func)
                if this_k > parent_k and this_k > 2:
                    k_components[this_k].append(set(C))
                cuts = list(nx.all_node_cuts(C, k=this_k, flow_func=flow_func))
                if cuts:
                    stack.append((this_k, _generate_partition(C, cuts, this_k)))
            except StopIteration:
                stack.pop()

    # This is necessary because k-components may only be reported at their
    # maximum k level. But we want to return a dictionary in which keys are
    # connectivity levels and values list of sets of components, without
    # skipping any connectivity level. Also, it's possible that subsets of
    # an already detected k-component appear at a level k. Checking for this
    # in the while loop above penalizes the common case. Thus we also have to
    # _consolidate all connectivity levels in _reconstruct_k_components.
    return _reconstruct_k_components(k_components)


def _consolidate(sets, k):
    """Merge sets that share k or more elements.

    See: http://rosettacode.org/wiki/Set_consolidation

    The iterative python implementation posted there is
    faster than this because of the overhead of building a
    Graph and calling nx.connected_components, but it's not
    clear for us if we can use it in NetworkX because there
    is no licence for the code.

    """
    G = nx.Graph()
    nodes = dict(enumerate(sets))
    G.add_nodes_from(nodes)
    G.add_edges_from(
        (u, v) for u, v in combinations(nodes, 2) if len(nodes[u] & nodes[v]) >= k
    )
    for component in nx.connected_components(G):
        yield set.union(*[nodes[n] for n in component])


def _generate_partition(G, cuts, k):
    def has_nbrs_in_partition(G, node, partition):
        return any(n in partition for n in G[node])

    components = []
    nodes = {n for n, d in G.degree() if d > k} - {n for cut in cuts for n in cut}
    H = G.subgraph(nodes)
    for cc in nx.connected_components(H):
        component = set(cc)
        for cut in cuts:
            for node in cut:
                if has_nbrs_in_partition(G, node, cc):
                    component.add(node)
        if len(component) < G.order():
            components.append(component)
    yield from _consolidate(components, k + 1)


def _reconstruct_k_components(k_comps):
    result = {}
    max_k = max(k_comps)
    for k in reversed(range(1, max_k + 1)):
        if k == max_k:
            result[k] = list(_consolidate(k_comps[k], k))
        elif k not in k_comps:
            result[k] = list(_consolidate(result[k + 1], k))
        else:
            nodes_at_k = set.union(*k_comps[k])
            to_add = [c for c in result[k + 1] if any(n not in nodes_at_k for n in c)]
            if to_add:
                result[k] = list(_consolidate(k_comps[k] + to_add, k))
            else:
                result[k] = list(_consolidate(k_comps[k], k))
    return result


def build_k_number_dict(kcomps):
    result = {}
    for k, comps in sorted(kcomps.items(), key=itemgetter(0)):
        for comp in comps:
            for node in comp:
                result[node] = k
    return result
<|endoftext|># This module uses material from the Wikipedia article Hopcroft--Karp algorithm
# <https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm>, accessed on
# January 3, 2015, which is released under the Creative Commons
# Attribution-Share-Alike License 3.0
# <http://creativecommons.org/licenses/by-sa/3.0/>. That article includes
# pseudocode, which has been translated into the corresponding Python code.
#
# Portions of this module use code from David Eppstein's Python Algorithms and
# Data Structures (PADS) library, which is dedicated to the public domain (for
# proof, see <http://www.ics.uci.edu/~eppstein/PADS/ABOUT-PADS.txt>).
"""Provides functions for computing maximum cardinality matchings and minimum
weight full matchings in a bipartite graph.

If you don't care about the particular implementation of the maximum matching
algorithm, simply use the :func:`maximum_matching`. If you do care, you can
import one of the named maximum matching algorithms directly.

For example, to find a maximum matching in the complete bipartite graph with
two vertices on the left and three vertices on the right:

>>> G = nx.complete_bipartite_graph(2, 3)
>>> left, right = nx.bipartite.sets(G)
>>> list(left)
[0, 1]
>>> list(right)
[2, 3, 4]
>>> nx.bipartite.maximum_matching(G)
{0: 2, 1: 3, 2: 0, 3: 1}

The dictionary returned by :func:`maximum_matching` includes a mapping for
vertices in both the left and right vertex sets.

Similarly, :func:`minimum_weight_full_matching` produces, for a complete
weighted bipartite graph, a matching whose cardinality is the cardinality of
the smaller of the two partitions, and for which the sum of the weights of the
edges included in the matching is minimal.

"""

import collections
import itertools

import networkx as nx
from networkx.algorithms.bipartite import sets as bipartite_sets
from networkx.algorithms.bipartite.matrix import biadjacency_matrix

__all__ = [
    "maximum_matching",
    "hopcroft_karp_matching",
    "eppstein_matching",
    "to_vertex_cover",
    "minimum_weight_full_matching",
]

INFINITY = float("inf")


@nx._dispatchable
def hopcroft_karp_matching(G, top_nodes=None):
    """Returns the maximum cardinality matching of the bipartite graph `G`.

    A matching is a set of edges that do not share any nodes. A maximum
    cardinality matching is a matching with the most edges possible. It
    is not always unique. Finding a matching in a bipartite graph can be
    treated as a networkx flow problem.

    The functions ``hopcroft_karp_matching`` and ``maximum_matching``
    are aliases of the same function.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container of nodes

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matches`, such that
      ``matches[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matches`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented with the `Hopcroft--Karp matching algorithm
    <https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm>`_ for
    bipartite graphs.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    maximum_matching
    hopcroft_karp_matching
    eppstein_matching

    References
    ----------
    .. [1] John E. Hopcroft and Richard M. Karp. "An n^{5 / 2} Algorithm for
       Maximum Matchings in Bipartite Graphs" In: **SIAM Journal of Computing**
       2.4 (1973), pp. 225--231. <https://doi.org/10.1137/0202019>.

    """

    # First we define some auxiliary search functions.
    #
    # If you are a human reading these auxiliary search functions, the "global"
    # variables `leftmatches`, `rightmatches`, `distances`, etc. are defined
    # below the functions, so that they are initialized close to the initial
    # invocation of the search functions.
    def breadth_first_search():
        for v in left:
            if leftmatches[v] is None:
                distances[v] = 0
                queue.append(v)
            else:
                distances[v] = INFINITY
        distances[None] = INFINITY
        while queue:
            v = queue.popleft()
            if distances[v] < distances[None]:
                for u in G[v]:
                    if distances[rightmatches[u]] is INFINITY:
                        distances[rightmatches[u]] = distances[v] + 1
                        queue.append(rightmatches[u])
        return distances[None] is not INFINITY

    def depth_first_search(v):
        if v is not None:
            for u in G[v]:
                if distances[rightmatches[u]] == distances[v] + 1:
                    if depth_first_search(rightmatches[u]):
                        rightmatches[u] = v
                        leftmatches[v] = u
                        return True
            distances[v] = INFINITY
            return False
        return True

    # Initialize the "global" variables that maintain state during the search.
    left, right = bipartite_sets(G, top_nodes)
    leftmatches = {v: None for v in left}
    rightmatches = {v: None for v in right}
    distances = {}
    queue = collections.deque()

    # Implementation note: this counter is incremented as pairs are matched but
    # it is currently not used elsewhere in the computation.
    num_matched_pairs = 0
    while breadth_first_search():
        for v in left:
            if leftmatches[v] is None:
                if depth_first_search(v):
                    num_matched_pairs += 1

    # Strip the entries matched to `None`.
    leftmatches = {k: v for k, v in leftmatches.items() if v is not None}
    rightmatches = {k: v for k, v in rightmatches.items() if v is not None}

    # At this point, the left matches and the right matches are inverses of one
    # another. In other words,
    #
    #     leftmatches == {v, k for k, v in rightmatches.items()}
    #
    # Finally, we combine both the left matches and right matches.
    return dict(itertools.chain(leftmatches.items(), rightmatches.items()))


@nx._dispatchable
def eppstein_matching(G, top_nodes=None):
    """Returns the maximum cardinality matching of the bipartite graph `G`.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matching`, such that
      ``matching[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matching`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented with David Eppstein's version of the algorithm
    Hopcroft--Karp algorithm (see :func:`hopcroft_karp_matching`), which
    originally appeared in the `Python Algorithms and Data Structures library
    (PADS) <http://www.ics.uci.edu/~eppstein/PADS/ABOUT-PADS.txt>`_.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------

    hopcroft_karp_matching

    """
    # Due to its original implementation, a directed graph is needed
    # so that the two sets of bipartite nodes can be distinguished
    left, right = bipartite_sets(G, top_nodes)
    G = nx.DiGraph(G.edges(left))
    # initialize greedy matching (redundant, but faster than full search)
    matching = {}
    for u in G:
        for v in G[u]:
            if v not in matching:
                matching[v] = u
                break
    while True:
        # structure residual graph into layers
        # pred[u] gives the neighbor in the previous layer for u in U
        # preds[v] gives a list of neighbors in the previous layer for v in V
        # unmatched gives a list of unmatched vertices in final layer of V,
        # and is also used as a flag value for pred[u] when u is in the first
        # layer
        preds = {}
        unmatched = []
        pred = {u: unmatched for u in G}
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)

        # repeatedly extend layering structure by another pair of layers
        while layer and not unmatched:
            newLayer = {}
            for u in layer:
                for v in G[u]:
                    if v not in preds:
                        newLayer.setdefault(v, []).append(u)
            layer = []
            for v in newLayer:
                preds[v] = newLayer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        # did we finish layering without finding any alternating paths?
        if not unmatched:
            # TODO - The lines between --- were unused and were thus commented
            # out. This whole commented chunk should be reviewed to determine
            # whether it should be built upon or completely removed.
            # ---
            # unlayered = {}
            # for u in G:
            #     # TODO Why is extra inner loop necessary?
            #     for v in G[u]:
            #         if v not in preds:
            #             unlayered[v] = None
            # ---
            # TODO Originally, this function returned a three-tuple:
            #
            #     return (matching, list(pred), list(unlayered))
            #
            # For some reason, the documentation for this function
            # indicated that the second and third elements of the returned
            # three-tuple would be the vertices in the left and right vertex
            # sets, respectively, that are also in the maximum independent set.
            # However, what I think the author meant was that the second
            # element is the list of vertices that were unmatched and the third
            # element was the list of vertices that were matched. Since that
            # seems to be the case, they don't really need to be returned,
            # since that information can be inferred from the matching
            # dictionary.

            # All the matched nodes must be a key in the dictionary
            for key in matching.copy():
                matching[matching[key]] = key
            return matching

        # recursively search backward through layers to find alternating paths
        # recursion returns true if found path, false otherwise
        def recurse(v):
            if v in preds:
                L = preds.pop(v)
                for u in L:
                    if u in pred:
                        pu = pred.pop(u)
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return True
            return False

        for v in unmatched:
            recurse(v)


def _is_connected_by_alternating_path(G, v, matched_edges, unmatched_edges, targets):
    """Returns True if and only if the vertex `v` is connected to one of
    the target vertices by an alternating path in `G`.

    An *alternating path* is a path in which every other edge is in the
    specified maximum matching (and the remaining edges in the path are not in
    the matching). An alternating path may have matched edges in the even
    positions or in the odd positions, as long as the edges alternate between
    'matched' and 'unmatched'.

    `G` is an undirected bipartite NetworkX graph.

    `v` is a vertex in `G`.

    `matched_edges` is a set of edges present in a maximum matching in `G`.

    `unmatched_edges` is a set of edges not present in a maximum
    matching in `G`.

    `targets` is a set of vertices.

    """

    def _alternating_dfs(u, along_matched=True):
        """Returns True if and only if `u` is connected to one of the
        targets by an alternating path.

        `u` is a vertex in the graph `G`.

        If `along_matched` is True, this step of the depth-first search
        will continue only through edges in the given matching. Otherwise, it
        will continue only through edges *not* in the given matching.

        """
        visited = set()
        # Follow matched edges when depth is even,
        # and follow unmatched edges when depth is odd.
        initial_depth = 0 if along_matched else 1
        stack = [(u, iter(G[u]), initial_depth)]
        while stack:
            parent, children, depth = stack[-1]
            valid_edges = matched_edges if depth % 2 else unmatched_edges
            try:
                child = next(children)
                if child not in visited:
                    if (parent, child) in valid_edges or (child, parent) in valid_edges:
                        if child in targets:
                            return True
                        visited.add(child)
                        stack.append((child, iter(G[child]), depth + 1))
            except StopIteration:
                stack.pop()
        return False

    # Check for alternating paths starting with edges in the matching, then
    # check for alternating paths starting with edges not in the
    # matching.
    return _alternating_dfs(v, along_matched=True) or _alternating_dfs(
        v, along_matched=False
    )


def _connected_by_alternating_paths(G, matching, targets):
    """Returns the set of vertices that are connected to one of the target
    vertices by an alternating path in `G` or are themselves a target.

    An *alternating path* is a path in which every other edge is in the
    specified maximum matching (and the remaining edges in the path are not in
    the matching). An alternating path may have matched edges in the even
    positions or in the odd positions, as long as the edges alternate between
    'matched' and 'unmatched'.

    `G` is an undirected bipartite NetworkX graph.

    `matching` is a dictionary representing a maximum matching in `G`, as
    returned by, for example, :func:`maximum_matching`.

    `targets` is a set of vertices.

    """
    # Get the set of matched edges and the set of unmatched edges. Only include
    # one version of each undirected edge (for example, include edge (1, 2) but
    # not edge (2, 1)). Using frozensets as an intermediary step we do not
    # require nodes to be orderable.
    edge_sets = {frozenset((u, v)) for u, v in matching.items()}
    matched_edges = {tuple(edge) for edge in edge_sets}
    unmatched_edges = {
        (u, v) for (u, v) in G.edges() if frozenset((u, v)) not in edge_sets
    }

    return {
        v
        for v in G
        if v in targets
        or _is_connected_by_alternating_path(
            G, v, matched_edges, unmatched_edges, targets
        )
    }


@nx._dispatchable
def to_vertex_cover(G, matching, top_nodes=None):
    """Returns the minimum vertex cover corresponding to the given maximum
    matching of the bipartite graph `G`.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    matching : dictionary

      A dictionary whose keys are vertices in `G` and whose values are the
      distinct neighbors comprising the maximum matching for `G`, as returned
      by, for example, :func:`maximum_matching`. The dictionary *must*
      represent the maximum matching.

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    vertex_cover : :class:`set`

      The minimum vertex cover in `G`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented using the procedure guaranteed by `Konig's
    theorem
    <https://en.wikipedia.org/wiki/K%C3%B6nig%27s_theorem_%28graph_theory%29>`_,
    which proves an equivalence between a maximum matching and a minimum vertex
    cover in bipartite graphs.

    Since a minimum vertex cover is the complement of a maximum independent set
    for any graph, one can compute the maximum independent set of a bipartite
    graph this way:

    >>> G = nx.complete_bipartite_graph(2, 3)
    >>> matching = nx.bipartite.maximum_matching(G)
    >>> vertex_cover = nx.bipartite.to_vertex_cover(G, matching)
    >>> independent_set = set(G) - vertex_cover
    >>> print(list(independent_set))
    [2, 3, 4]

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    """
    # This is a Python implementation of the algorithm described at
    # <https://en.wikipedia.org/wiki/K%C3%B6nig%27s_theorem_%28graph_theory%29#Proof>.
    L, R = bipartite_sets(G, top_nodes)
    # Let U be the set of unmatched vertices in the left vertex set.
    unmatched_vertices = set(G) - set(matching)
    U = unmatched_vertices & L
    # Let Z be the set of vertices that are either in U or are connected to U
    # by alternating paths.
    Z = _connected_by_alternating_paths(G, matching, U)
    # At this point, every edge either has a right endpoint in Z or a left
    # endpoint not in Z. This gives us the vertex cover.
    return (L - Z) | (R & Z)


#: Returns the maximum cardinality matching in the given bipartite graph.
#:
#: This function is simply an alias for :func:`hopcroft_karp_matching`.
maximum_matching = hopcroft_karp_matching


@nx._dispatchable(edge_attrs="weight")
def minimum_weight_full_matching(G, top_nodes=None, weight="weight"):
    r"""Returns a minimum weight full matching of the bipartite graph `G`.

    Let :math:`G = ((U, V), E)` be a weighted bipartite graph with real weights
    :math:`w : E \to \mathbb{R}`. This function then produces a matching
    :math:`M \subseteq E` with cardinality

    .. math::
       \lvert M \rvert = \min(\lvert U \rvert, \lvert V \rvert),

    which minimizes the sum of the weights of the edges included in the
    matching, :math:`\sum_{e \in M} w(e)`, or raises an error if no such
    matching exists.

    When :math:`\lvert U \rvert = \lvert V \rvert`, this is commonly
    referred to as a perfect matching; here, since we allow
    :math:`\lvert U \rvert` and :math:`\lvert V \rvert` to differ, we
    follow Karp [1]_ and refer to the matching as *full*.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed.

    weight : string, optional (default='weight')

       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matches`, such that
      ``matches[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matches`.

    Raises
    ------
    ValueError
      Raised if no full matching exists.

    ImportError
      Raised if SciPy is not available.

    Notes
    -----
    The problem of determining a minimum weight full matching is also known as
    the rectangular linear assignment problem. This implementation defers the
    calculation of the assignment to SciPy.

    References
    ----------
    .. [1] Richard Manning Karp:
       An algorithm to Solve the m x n Assignment Problem in Expected Time
       O(mn log n).
       Networks, 10(2):143–152, 1980.

    """
    import numpy as np
    import scipy as sp

    left, right = nx.bipartite.sets(G, top_nodes)
    U = list(left)
    V = list(right)
    # We explicitly create the biadjacency matrix having infinities
    # where edges are missing (as opposed to zeros, which is what one would
    # get by using toarray on the sparse matrix).
    weights_sparse = biadjacency_matrix(
        G, row_order=U, column_order=V, weight=weight, format="coo"
    )
    weights = np.full(weights_sparse.shape, np.inf)
    weights[weights_sparse.row, weights_sparse.col] = weights_sparse.data
    left_matches = sp.optimize.linear_sum_assignment(weights)
    d = {U[u]: V[v] for u, v in zip(*left_matches)}
    # d will contain the matching from edges in left to right; we need to
    # add the ones from right to left as well.
    d.update({v: u for u, v in d.items()})
    return d
<|endoftext|>"""Functions for computing clustering of pairs"""

import itertools

import networkx as nx

__all__ = [
    "clustering",
    "average_clustering",
    "latapy_clustering",
    "robins_alexander_clustering",
]


def cc_dot(nu, nv):
    return len(nu & nv) / len(nu | nv)


def cc_max(nu, nv):
    return len(nu & nv) / max(len(nu), len(nv))


def cc_min(nu, nv):
    return len(nu & nv) / min(len(nu), len(nv))


modes = {"dot": cc_dot, "min": cc_min, "max": cc_max}


@nx._dispatchable
def latapy_clustering(G, nodes=None, mode="dot"):
    r"""Compute a bipartite clustering coefficient for nodes.

    The bipartite clustering coefficient is a measure of local density
    of connections defined as [1]_:

    .. math::

       c_u = \frac{\sum_{v \in N(N(u))} c_{uv} }{|N(N(u))|}

    where `N(N(u))` are the second order neighbors of `u` in `G` excluding `u`,
    and `c_{uv}` is the pairwise clustering coefficient between nodes
    `u` and `v`.

    The mode selects the function for `c_{uv}` which can be:

    `dot`:

    .. math::

       c_{uv}=\frac{|N(u)\cap N(v)|}{|N(u) \cup N(v)|}

    `min`:

    .. math::

       c_{uv}=\frac{|N(u)\cap N(v)|}{min(|N(u)|,|N(v)|)}

    `max`:

    .. math::

       c_{uv}=\frac{|N(u)\cap N(v)|}{max(|N(u)|,|N(v)|)}


    Parameters
    ----------
    G : graph
        A bipartite graph

    nodes : list or iterable (optional)
        Compute bipartite clustering for these nodes. The default
        is all nodes in G.

    mode : string
        The pairwise bipartite clustering method to be used in the computation.
        It must be "dot", "max", or "min".

    Returns
    -------
    clustering : dictionary
        A dictionary keyed by node with the clustering coefficient value.


    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)  # path graphs are bipartite
    >>> c = bipartite.clustering(G)
    >>> c[0]
    0.5
    >>> c = bipartite.clustering(G, mode="min")
    >>> c[0]
    1.0

    See Also
    --------
    robins_alexander_clustering
    average_clustering
    networkx.algorithms.cluster.square_clustering

    References
    ----------
    .. [1] Latapy, Matthieu, Clémence Magnien, and Nathalie Del Vecchio (2008).
       Basic notions for the analysis of large two-mode networks.
       Social Networks 30(1), 31--48.
    """
    if not nx.algorithms.bipartite.is_bipartite(G):
        raise nx.NetworkXError("Graph is not bipartite")

    try:
        cc_func = modes[mode]
    except KeyError as err:
        raise nx.NetworkXError(
            "Mode for bipartite clustering must be: dot, min or max"
        ) from err

    if nodes is None:
        nodes = G
    ccs = {}
    for v in nodes:
        cc = 0.0
        nbrs2 = {u for nbr in G[v] for u in G[nbr]} - {v}
        for u in nbrs2:
            cc += cc_func(set(G[u]), set(G[v]))
        if cc > 0.0:  # len(nbrs2)>0
            cc /= len(nbrs2)
        ccs[v] = cc
    return ccs


clustering = latapy_clustering


@nx._dispatchable(name="bipartite_average_clustering")
def average_clustering(G, nodes=None, mode="dot"):
    r"""Compute the average bipartite clustering coefficient.

    A clustering coefficient for the whole graph is the average,

    .. math::

       C = \frac{1}{n}\sum_{v \in G} c_v,

    where `n` is the number of nodes in `G`.

    Similar measures for the two bipartite sets can be defined [1]_

    .. math::

       C_X = \frac{1}{|X|}\sum_{v \in X} c_v,

    where `X` is a bipartite set of `G`.

    Parameters
    ----------
    G : graph
        a bipartite graph

    nodes : list or iterable, optional
        A container of nodes to use in computing the average.
        The nodes should be either the entire graph (the default) or one of the
        bipartite sets.

    mode : string
        The pairwise bipartite clustering method.
        It must be "dot", "max", or "min"

    Returns
    -------
    clustering : float
       The average bipartite clustering for the given set of nodes or the
       entire graph if no nodes are specified.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.star_graph(3)  # star graphs are bipartite
    >>> bipartite.average_clustering(G)
    0.75
    >>> X, Y = bipartite.sets(G)
    >>> bipartite.average_clustering(G, X)
    0.0
    >>> bipartite.average_clustering(G, Y)
    1.0

    See Also
    --------
    clustering

    Notes
    -----
    The container of nodes passed to this function must contain all of the nodes
    in one of the bipartite sets ("top" or "bottom") in order to compute
    the correct average bipartite clustering coefficients.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.


    References
    ----------
    .. [1] Latapy, Matthieu, Clémence Magnien, and Nathalie Del Vecchio (2008).
        Basic notions for the analysis of large two-mode networks.
        Social Networks 30(1), 31--48.
    """
    if nodes is None:
        nodes = G
    ccs = latapy_clustering(G, nodes=nodes, mode=mode)
    return sum(ccs[v] for v in nodes) / len(nodes)


@nx._dispatchable
def robins_alexander_clustering(G):
    r"""Compute the bipartite clustering of G.

    Robins and Alexander [1]_ defined bipartite clustering coefficient as
    four times the number of four cycles `C_4` divided by the number of
    three paths `L_3` in a bipartite graph:

    .. math::

       CC_4 = \frac{4 * C_4}{L_3}

    Parameters
    ----------
    G : graph
        a bipartite graph

    Returns
    -------
    clustering : float
       The Robins and Alexander bipartite clustering for the input graph.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.davis_southern_women_graph()
    >>> print(round(bipartite.robins_alexander_clustering(G), 3))
    0.468

    See Also
    --------
    latapy_clustering
    networkx.algorithms.cluster.square_clustering

    References
    ----------
    .. [1] Robins, G. and M. Alexander (2004). Small worlds among interlocking
           directors: Network structure and distance in bipartite graphs.
           Computational & Mathematical Organization Theory 10(1), 69–94.

    """
    if G.order() < 4 or G.size() < 3:
        return 0
    L_3 = _threepaths(G)
    if L_3 == 0:
        return 0
    C_4 = _four_cycles(G)
    return (4.0 * C_4) / L_3


def _four_cycles(G):
    cycles = 0
    for v in G:
        for u, w in itertools.combinations(G[v], 2):
            cycles += len((set(G[u]) & set(G[w])) - {v})
    return cycles / 4


def _threepaths(G):
    paths = 0
    for v in G:
        for u in G[v]:
            for w in set(G[u]) - {v}:
                paths += len(set(G[w]) - {v, u})
    # Divide by two because we count each three path twice
    # one for each possible starting point
    return paths / 2
<|endoftext|>"""
====================
Biadjacency matrices
====================
"""

import itertools

import networkx as nx
from networkx.convert_matrix import _generate_weighted_edges

__all__ = ["biadjacency_matrix", "from_biadjacency_matrix"]


@nx._dispatchable(edge_attrs="weight")
def biadjacency_matrix(
    G, row_order, column_order=None, dtype=None, weight="weight", format="csr"
):
    r"""Returns the biadjacency matrix of the bipartite graph G.

    Let `G = (U, V, E)` be a bipartite graph with node sets
    `U = u_{1},...,u_{r}` and `V = v_{1},...,v_{s}`. The biadjacency
    matrix [1]_ is the `r` x `s` matrix `B` in which `b_{i,j} = 1`
    if, and only if, `(u_i, v_j) \in E`. If the parameter `weight` is
    not `None` and matches the name of an edge attribute, its value is
    used instead of 1.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    row_order : list of nodes
       The rows of the matrix are ordered according to the list of nodes.

    column_order : list, optional
       The columns of the matrix are ordered according to the list of nodes.
       If column_order is None, then the ordering of columns is arbitrary.

    dtype : NumPy data-type, optional
        A valid NumPy dtype used to initialize the array. If None, then the
        NumPy default is used.

    weight : string or None, optional (default='weight')
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.

    format : str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}
        The type of the matrix to be returned (default 'csr').  For
        some algorithms different implementations of sparse matrices
        can perform better.  See [2]_ for details.

    Returns
    -------
    M : SciPy sparse array
        Biadjacency matrix representation of the bipartite graph G.

    Notes
    -----
    No attempt is made to check that the input graph is bipartite.

    For directed bipartite graphs only successors are considered as neighbors.
    To obtain an adjacency matrix with ones (or weight values) for both
    predecessors and successors you have to generate two biadjacency matrices
    where the rows of one of them are the columns of the other, and then add
    one to the transpose of the other.

    See Also
    --------
    adjacency_matrix
    from_biadjacency_matrix

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Adjacency_matrix#Adjacency_matrix_of_a_bipartite_graph
    .. [2] Scipy Dev. References, "Sparse Matrices",
       https://docs.scipy.org/doc/scipy/reference/sparse.html
    """
    import scipy as sp

    nlen = len(row_order)
    if nlen == 0:
        raise nx.NetworkXError("row_order is empty list")
    if len(row_order) != len(set(row_order)):
        msg = "Ambiguous ordering: `row_order` contained duplicates."
        raise nx.NetworkXError(msg)
    if column_order is None:
        column_order = list(set(G) - set(row_order))
    mlen = len(column_order)
    if len(column_order) != len(set(column_order)):
        msg = "Ambiguous ordering: `column_order` contained duplicates."
        raise nx.NetworkXError(msg)

    row_index = dict(zip(row_order, itertools.count()))
    col_index = dict(zip(column_order, itertools.count()))

    if G.number_of_edges() == 0:
        row, col, data = [], [], []
    else:
        row, col, data = zip(
            *(
                (row_index[u], col_index[v], d.get(weight, 1))
                for u, v, d in G.edges(row_order, data=True)
                if u in row_index and v in col_index
            )
        )
    A = sp.sparse.coo_array((data, (row, col)), shape=(nlen, mlen), dtype=dtype)
    try:
        return A.asformat(format)
    except ValueError as err:
        raise nx.NetworkXError(f"Unknown sparse array format: {format}") from err


@nx._dispatchable(graphs=None, returns_graph=True)
def from_biadjacency_matrix(A, create_using=None, edge_attribute="weight"):
    r"""Creates a new bipartite graph from a biadjacency matrix given as a
    SciPy sparse array.

    Parameters
    ----------
    A: scipy sparse array
      A biadjacency matrix representation of a graph

    create_using: NetworkX graph
       Use specified graph for result.  The default is Graph()

    edge_attribute: string
       Name of edge attribute to store matrix numeric value. The data will
       have the same type as the matrix entry (int, float, (real,imag)).

    Notes
    -----
    The nodes are labeled with the attribute `bipartite` set to an integer
    0 or 1 representing membership in part 0 or part 1 of the bipartite graph.

    If `create_using` is an instance of :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph` and the entries of `A` are of
    type :class:`int`, then this function returns a multigraph (of the same
    type as `create_using`) with parallel edges. In this case, `edge_attribute`
    will be ignored.

    See Also
    --------
    biadjacency_matrix
    from_numpy_array

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Adjacency_matrix#Adjacency_matrix_of_a_bipartite_graph
    """
    G = nx.empty_graph(0, create_using)
    n, m = A.shape
    # Make sure we get even the isolated nodes of the graph.
    G.add_nodes_from(range(n), bipartite=0)
    G.add_nodes_from(range(n, n + m), bipartite=1)
    # Create an iterable over (u, v, w) triples and for each triple, add an
    # edge from u to v with weight w.
    triples = ((u, n + v, d) for (u, v, d) in _generate_weighted_edges(A))
    # If the entries in the adjacency matrix are integers and the graph is a
    # multigraph, then create parallel edges, each with weight 1, for each
    # entry in the adjacency matrix. Otherwise, create one edge for each
    # positive entry in the adjacency matrix and set the weight of that edge to
    # be the entry in the matrix.
    if A.dtype.kind in ("i", "u") and G.is_multigraph():
        chain = itertools.chain.from_iterable
        triples = chain(((u, v, 1) for d in range(w)) for (u, v, w) in triples)
    G.add_weighted_edges_from(triples, weight=edge_attribute)
    return G
<|endoftext|>"""
Spectral bipartivity measure.
"""

import networkx as nx

__all__ = ["spectral_bipartivity"]


@nx._dispatchable(edge_attrs="weight")
def spectral_bipartivity(G, nodes=None, weight="weight"):
    """Returns the spectral bipartivity.

    Parameters
    ----------
    G : NetworkX graph

    nodes : list or container  optional(default is all nodes)
      Nodes to return value of spectral bipartivity contribution.

    weight : string or None  optional (default = 'weight')
      Edge data key to use for edge weights. If None, weights set to 1.

    Returns
    -------
    sb : float or dict
       A single number if the keyword nodes is not specified, or
       a dictionary keyed by node with the spectral bipartivity contribution
       of that node as the value.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> bipartite.spectral_bipartivity(G)
    1.0

    Notes
    -----
    This implementation uses Numpy (dense) matrices which are not efficient
    for storing large sparse graphs.

    See Also
    --------
    color

    References
    ----------
    .. [1] E. Estrada and J. A. Rodríguez-Velázquez, "Spectral measures of
       bipartivity in complex networks", PhysRev E 72, 046105 (2005)
    """
    import scipy as sp

    nodelist = list(G)  # ordering of nodes in matrix
    A = nx.to_numpy_array(G, nodelist, weight=weight)
    expA = sp.linalg.expm(A)
    expmA = sp.linalg.expm(-A)
    coshA = 0.5 * (expA + expmA)
    if nodes is None:
        # return single number for entire graph
        return float(coshA.diagonal().sum() / expA.diagonal().sum())
    else:
        # contribution for individual nodes
        index = dict(zip(nodelist, range(len(nodelist))))
        sb = {}
        for n in nodes:
            i = index[n]
            sb[n] = coshA.item(i, i) / expA.item(i, i)
        return sb
<|endoftext|>"""
********************
Bipartite Edge Lists
********************
Read and write NetworkX graphs as bipartite edge lists.

Format
------
You can read or write three formats of edge lists with these functions.

Node pairs with no data::

 1 2

Python dictionary as data::

 1 2 {'weight':7, 'color':'green'}

Arbitrary data::

 1 2 7 green

For each edge (u, v) the node u is assigned to part 0 and the node v to part 1.
"""

__all__ = ["generate_edgelist", "write_edgelist", "parse_edgelist", "read_edgelist"]

import networkx as nx
from networkx.utils import not_implemented_for, open_file


@open_file(1, mode="wb")
def write_edgelist(G, path, comments="#", delimiter=" ", data=True, encoding="utf-8"):
    """Write a bipartite graph as a list of edges.

    Parameters
    ----------
    G : Graph
       A NetworkX bipartite graph
    path : file or string
       File or filename to write. If a file is provided, it must be
       opened in 'wb' mode. Filenames ending in .gz or .bz2 will be compressed.
    comments : string, optional
       The character used to indicate the start of a comment
    delimiter : string, optional
       The string used to separate values.  The default is whitespace.
    data : bool or list, optional
       If False write no edge data.
       If True write a string representation of the edge data dictionary..
       If a list (or other iterable) is provided, write the  keys specified
       in the list.
    encoding: string, optional
       Specify which encoding to use when writing file.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> G.add_nodes_from([0, 2], bipartite=0)
    >>> G.add_nodes_from([1, 3], bipartite=1)
    >>> nx.write_edgelist(G, "test.edgelist")
    >>> fh = open("test.edgelist", "wb")
    >>> nx.write_edgelist(G, fh)
    >>> nx.write_edgelist(G, "test.edgelist.gz")
    >>> nx.write_edgelist(G, "test.edgelist.gz", data=False)

    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=7, color="red")
    >>> nx.write_edgelist(G, "test.edgelist", data=False)
    >>> nx.write_edgelist(G, "test.edgelist", data=["color"])
    >>> nx.write_edgelist(G, "test.edgelist", data=["color", "weight"])

    See Also
    --------
    write_edgelist
    generate_edgelist
    """
    for line in generate_edgelist(G, delimiter, data):
        line += "\n"
        path.write(line.encode(encoding))


@not_implemented_for("directed")
def generate_edgelist(G, delimiter=" ", data=True):
    """Generate a single line of the bipartite graph G in edge list format.

    Parameters
    ----------
    G : NetworkX graph
       The graph is assumed to have node attribute `part` set to 0,1 representing
       the two graph parts

    delimiter : string, optional
       Separator for node labels

    data : bool or list of keys
       If False generate no edge data.  If True use a dictionary
       representation of edge data.  If a list of keys use a list of data
       values corresponding to the keys.

    Returns
    -------
    lines : string
        Lines of data in adjlist format.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> G.add_nodes_from([0, 2], bipartite=0)
    >>> G.add_nodes_from([1, 3], bipartite=1)
    >>> G[1][2]["weight"] = 3
    >>> G[2][3]["capacity"] = 12
    >>> for line in bipartite.generate_edgelist(G, data=False):
    ...     print(line)
    0 1
    2 1
    2 3

    >>> for line in bipartite.generate_edgelist(G):
    ...     print(line)
    0 1 {}
    2 1 {'weight': 3}
    2 3 {'capacity': 12}

    >>> for line in bipartite.generate_edgelist(G, data=["weight"]):
    ...     print(line)
    0 1
    2 1 3
    2 3
    """
    try:
        part0 = [n for n, d in G.nodes.items() if d["bipartite"] == 0]
    except BaseException as err:
        raise AttributeError("Missing node attribute `bipartite`") from err
    if data is True or data is False:
        for n in part0:
            for edge in G.edges(n, data=data):
                yield delimiter.join(map(str, edge))
    else:
        for n in part0:
            for u, v, d in G.edges(n, data=True):
                edge = [u, v]
                try:
                    edge.extend(d[k] for k in data)
                except KeyError:
                    pass  # missing data for this edge, should warn?
                yield delimiter.join(map(str, edge))


@nx._dispatchable(name="bipartite_parse_edgelist", graphs=None, returns_graph=True)
def parse_edgelist(
    lines, comments="#", delimiter=None, create_using=None, nodetype=None, data=True
):
    """Parse lines of an edge list representation of a bipartite graph.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in edgelist format
    comments : string, optional
       Marker for comment lines
    delimiter : string, optional
       Separator for node labels
    create_using: NetworkX graph container, optional
       Use given NetworkX graph for holding nodes or edges.
    nodetype : Python type, optional
       Convert nodes to this type.
    data : bool or list of (label,type) tuples
       If False generate no edge data or if True use a dictionary
       representation of edge data or a list tuples specifying dictionary
       key names and types for edge data.

    Returns
    -------
    G: NetworkX Graph
        The bipartite graph corresponding to lines

    Examples
    --------
    Edgelist with no data:

    >>> from networkx.algorithms import bipartite
    >>> lines = ["1 2", "2 3", "3 4"]
    >>> G = bipartite.parse_edgelist(lines, nodetype=int)
    >>> sorted(G.nodes())
    [1, 2, 3, 4]
    >>> sorted(G.nodes(data=True))
    [(1, {'bipartite': 0}), (2, {'bipartite': 0}), (3, {'bipartite': 0}), (4, {'bipartite': 1})]
    >>> sorted(G.edges())
    [(1, 2), (2, 3), (3, 4)]

    Edgelist with data in Python dictionary representation:

    >>> lines = ["1 2 {'weight':3}", "2 3 {'weight':27}", "3 4 {'weight':3.0}"]
    >>> G = bipartite.parse_edgelist(lines, nodetype=int)
    >>> sorted(G.nodes())
    [1, 2, 3, 4]
    >>> sorted(G.edges(data=True))
    [(1, 2, {'weight': 3}), (2, 3, {'weight': 27}), (3, 4, {'weight': 3.0})]

    Edgelist with data in a list:

    >>> lines = ["1 2 3", "2 3 27", "3 4 3.0"]
    >>> G = bipartite.parse_edgelist(lines, nodetype=int, data=(("weight", float),))
    >>> sorted(G.nodes())
    [1, 2, 3, 4]
    >>> sorted(G.edges(data=True))
    [(1, 2, {'weight': 3.0}), (2, 3, {'weight': 27.0}), (3, 4, {'weight': 3.0})]

    See Also
    --------
    """
    from ast import literal_eval

    G = nx.empty_graph(0, create_using)
    for line in lines:
        p = line.find(comments)
        if p >= 0:
            line = line[:p]
        if not len(line):
            continue
        # split line, should have 2 or more
        s = line.rstrip("\n").split(delimiter)
        if len(s) < 2:
            continue
        u = s.pop(0)
        v = s.pop(0)
        d = s
        if nodetype is not None:
            try:
                u = nodetype(u)
                v = nodetype(v)
            except BaseException as err:
                raise TypeError(
                    f"Failed to convert nodes {u},{v} to type {nodetype}."
                ) from err

        if len(d) == 0 or data is False:
            # no data or data type specified
            edgedata = {}
        elif data is True:
            # no edge types specified
            try:  # try to evaluate as dictionary
                edgedata = dict(literal_eval(" ".join(d)))
            except BaseException as err:
                raise TypeError(
                    f"Failed to convert edge data ({d}) to dictionary."
                ) from err
        else:
            # convert edge data to dictionary with specified keys and type
            if len(d) != len(data):
                raise IndexError(
                    f"Edge data {d} and data_keys {data} are not the same length"
                )
            edgedata = {}
            for (edge_key, edge_type), edge_value in zip(data, d):
                try:
                    edge_value = edge_type(edge_value)
                except BaseException as err:
                    raise TypeError(
                        f"Failed to convert {edge_key} data "
                        f"{edge_value} to type {edge_type}."
                    ) from err
                edgedata.update({edge_key: edge_value})
        G.add_node(u, bipartite=0)
        G.add_node(v, bipartite=1)
        G.add_edge(u, v, **edgedata)
    return G


@open_file(0, mode="rb")
@nx._dispatchable(name="bipartite_read_edgelist", graphs=None, returns_graph=True)
def read_edgelist(
    path,
    comments="#",
    delimiter=None,
    create_using=None,
    nodetype=None,
    data=True,
    edgetype=None,
    encoding="utf-8",
):
    """Read a bipartite graph from a list of edges.

    Parameters
    ----------
    path : file or string
       File or filename to read. If a file is provided, it must be
       opened in 'rb' mode.
       Filenames ending in .gz or .bz2 will be uncompressed.
    comments : string, optional
       The character used to indicate the start of a comment.
    delimiter : string, optional
       The string used to separate values.  The default is whitespace.
    create_using : Graph container, optional,
       Use specified container to build graph.  The default is networkx.Graph,
       an undirected graph.
    nodetype : int, float, str, Python type, optional
       Convert node data from strings to specified type
    data : bool or list of (label,type) tuples
       Tuples specifying dictionary key names and types for edge data
    edgetype : int, float, str, Python type, optional OBSOLETE
       Convert edge data from strings to specified type and use as 'weight'
    encoding: string, optional
       Specify which encoding to use when reading file.

    Returns
    -------
    G : graph
       A networkx Graph or other type specified with create_using

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> G.add_nodes_from([0, 2], bipartite=0)
    >>> G.add_nodes_from([1, 3], bipartite=1)
    >>> bipartite.write_edgelist(G, "test.edgelist")
    >>> G = bipartite.read_edgelist("test.edgelist")

    >>> fh = open("test.edgelist", "rb")
    >>> G = bipartite.read_edgelist(fh)
    >>> fh.close()

    >>> G = bipartite.read_edgelist("test.edgelist", nodetype=int)

    Edgelist with data in a list:

    >>> textline = "1 2 3"
    >>> fh = open("test.edgelist", "w")
    >>> d = fh.write(textline)
    >>> fh.close()
    >>> G = bipartite.read_edgelist(
    ...     "test.edgelist", nodetype=int, data=(("weight", float),)
    ... )
    >>> list(G)
    [1, 2]
    >>> list(G.edges(data=True))
    [(1, 2, {'weight': 3.0})]

    See parse_edgelist() for more examples of formatting.

    See Also
    --------
    parse_edgelist

    Notes
    -----
    Since nodes must be hashable, the function nodetype must return hashable
    types (e.g. int, float, str, frozenset - or tuples of those, etc.)
    """
    lines = (line.decode(encoding) for line in path)
    return parse_edgelist(
        lines,
        comments=comments,
        delimiter=delimiter,
        create_using=create_using,
        nodetype=nodetype,
        data=data,
    )
<|endoftext|>"""Functions related to graph covers."""

import networkx as nx
from networkx.algorithms.bipartite.matching import hopcroft_karp_matching
from networkx.algorithms.covering import min_edge_cover as _min_edge_cover
from networkx.utils import not_implemented_for

__all__ = ["min_edge_cover"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(name="bipartite_min_edge_cover")
def min_edge_cover(G, matching_algorithm=None):
    """Returns a set of edges which constitutes
    the minimum edge cover of the graph.

    The smallest edge cover can be found in polynomial time by finding
    a maximum matching and extending it greedily so that all nodes
    are covered.

    Parameters
    ----------
    G : NetworkX graph
        An undirected bipartite graph.

    matching_algorithm : function
        A function that returns a maximum cardinality matching in a
        given bipartite graph. The function must take one input, the
        graph ``G``, and return a dictionary mapping each node to its
        mate. If not specified,
        :func:`~networkx.algorithms.bipartite.matching.hopcroft_karp_matching`
        will be used. Other possibilities include
        :func:`~networkx.algorithms.bipartite.matching.eppstein_matching`,

    Returns
    -------
    set
        A set of the edges in a minimum edge cover of the graph, given as
        pairs of nodes. It contains both the edges `(u, v)` and `(v, u)`
        for given nodes `u` and `v` among the edges of minimum edge cover.

    Notes
    -----
    An edge cover of a graph is a set of edges such that every node of
    the graph is incident to at least one edge of the set.
    A minimum edge cover is an edge covering of smallest cardinality.

    Due to its implementation, the worst-case running time of this algorithm
    is bounded by the worst-case running time of the function
    ``matching_algorithm``.
    """
    if G.order() == 0:  # Special case for the empty graph
        return set()
    if matching_algorithm is None:
        matching_algorithm = hopcroft_karp_matching
    return _min_edge_cover(G, matching_algorithm=matching_algorithm)
<|endoftext|>"""
==========================
Bipartite Graph Algorithms
==========================
"""

import networkx as nx
from networkx.algorithms.components import connected_components
from networkx.exception import AmbiguousSolution

__all__ = [
    "is_bipartite",
    "is_bipartite_node_set",
    "color",
    "sets",
    "density",
    "degrees",
]


@nx._dispatchable
def color(G):
    """Returns a two-coloring of the graph.

    Raises an exception if the graph is not bipartite.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    color : dictionary
        A dictionary keyed by node with a 1 or 0 as data for each node color.

    Raises
    ------
    NetworkXError
        If the graph is not two-colorable.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> c = bipartite.color(G)
    >>> print(c)
    {0: 1, 1: 0, 2: 1, 3: 0}

    You can use this to set a node attribute indicating the bipartite set:

    >>> nx.set_node_attributes(G, c, "bipartite")
    >>> print(G.nodes[0]["bipartite"])
    1
    >>> print(G.nodes[1]["bipartite"])
    0
    """
    if G.is_directed():
        import itertools

        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors(v), G.successors(v)])

    else:
        neighbors = G.neighbors

    color = {}
    for n in G:  # handle disconnected graphs
        if n in color or len(G[n]) == 0:  # skip isolates
            continue
        queue = [n]
        color[n] = 1  # nodes seen with color (1 or 0)
        while queue:
            v = queue.pop()
            c = 1 - color[v]  # opposite color of node v
            for w in neighbors(v):
                if w in color:
                    if color[w] == color[v]:
                        raise nx.NetworkXError("Graph is not bipartite.")
                else:
                    color[w] = c
                    queue.append(w)
    # color isolates with 0
    color.update(dict.fromkeys(nx.isolates(G), 0))
    return color


@nx._dispatchable
def is_bipartite(G):
    """Returns True if graph G is bipartite, False if not.

    Parameters
    ----------
    G : NetworkX graph

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> print(bipartite.is_bipartite(G))
    True

    See Also
    --------
    color, is_bipartite_node_set
    """
    try:
        color(G)
        return True
    except nx.NetworkXError:
        return False


@nx._dispatchable
def is_bipartite_node_set(G, nodes):
    """Returns True if nodes and G/nodes are a bipartition of G.

    Parameters
    ----------
    G : NetworkX graph

    nodes: list or container
      Check if nodes are a one of a bipartite set.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> X = set([1, 3])
    >>> bipartite.is_bipartite_node_set(G, X)
    True

    Notes
    -----
    An exception is raised if the input nodes are not distinct, because in this
    case some bipartite algorithms will yield incorrect results.
    For connected graphs the bipartite sets are unique.  This function handles
    disconnected graphs.
    """
    S = set(nodes)

    if len(S) < len(nodes):
        # this should maybe just return False?
        raise AmbiguousSolution(
            "The input node set contains duplicates.\n"
            "This may lead to incorrect results when using it in bipartite algorithms.\n"
            "Consider using set(nodes) as the input"
        )

    for CC in (G.subgraph(c).copy() for c in connected_components(G)):
        X, Y = sets(CC)
        if not (
            (X.issubset(S) and Y.isdisjoint(S)) or (Y.issubset(S) and X.isdisjoint(S))
        ):
            return False
    return True


@nx._dispatchable
def sets(G, top_nodes=None):
    """Returns bipartite node sets of graph G.

    Raises an exception if the graph is not bipartite or if the input
    graph is disconnected and thus more than one valid solution exists.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    Parameters
    ----------
    G : NetworkX graph

    top_nodes : container, optional
      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    X : set
      Nodes from one side of the bipartite graph.
    Y : set
      Nodes from the other side.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.
    NetworkXError
      Raised if the input graph is not bipartite.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> X, Y = bipartite.sets(G)
    >>> list(X)
    [0, 2]
    >>> list(Y)
    [1, 3]

    See Also
    --------
    color

    """
    if G.is_directed():
        is_connected = nx.is_weakly_connected
    else:
        is_connected = nx.is_connected
    if top_nodes is not None:
        X = set(top_nodes)
        Y = set(G) - X
    else:
        if not is_connected(G):
            msg = "Disconnected graph: Ambiguous solution for bipartite sets."
            raise nx.AmbiguousSolution(msg)
        c = color(G)
        X = {n for n, is_top in c.items() if is_top}
        Y = {n for n, is_top in c.items() if not is_top}
    return (X, Y)


@nx._dispatchable(graphs="B")
def density(B, nodes):
    """Returns density of bipartite graph B.

    Parameters
    ----------
    B : NetworkX graph

    nodes: list or container
      Nodes in one node set of the bipartite graph.

    Returns
    -------
    d : float
       The bipartite density

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.complete_bipartite_graph(3, 2)
    >>> X = set([0, 1, 2])
    >>> bipartite.density(G, X)
    1.0
    >>> Y = set([3, 4])
    >>> bipartite.density(G, Y)
    1.0

    Notes
    -----
    The container of nodes passed as argument must contain all nodes
    in one of the two bipartite node sets to avoid ambiguity in the
    case of disconnected graphs.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    color
    """
    n = len(B)
    m = nx.number_of_edges(B)
    nb = len(nodes)
    nt = n - nb
    if m == 0:  # includes cases n==0 and n==1
        d = 0.0
    else:
        if B.is_directed():
            d = m / (2 * nb * nt)
        else:
            d = m / (nb * nt)
    return d


@nx._dispatchable(graphs="B", edge_attrs="weight")
def degrees(B, nodes, weight=None):
    """Returns the degrees of the two node sets in the bipartite graph B.

    Parameters
    ----------
    B : NetworkX graph

    nodes: list or container
      Nodes in one node set of the bipartite graph.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    Returns
    -------
    (degX,degY) : tuple of dictionaries
       The degrees of the two bipartite sets as dictionaries keyed by node.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.complete_bipartite_graph(3, 2)
    >>> Y = set([3, 4])
    >>> degX, degY = bipartite.degrees(G, Y)
    >>> dict(degX)
    {0: 2, 1: 2, 2: 2}

    Notes
    -----
    The container of nodes passed as argument must contain all nodes
    in one of the two bipartite node sets to avoid ambiguity in the
    case of disconnected graphs.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    color, density
    """
    bottom = set(nodes)
    top = set(B) - bottom
    return (B.degree(top, weight), B.degree(bottom, weight))
