      If None, edge weights are ignored.
      Otherwise holds the name of the edge attribute used as weight.
      The weight of an edge is treated as the length or distance between the two sides.

    cutoff : bool, optional (default=None)
      If specified, only consider paths of length <= cutoff.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with centrality as the value.

    See Also
    --------
    betweenness_centrality

    Notes
    -----
    Load centrality is slightly different than betweenness. It was originally
    introduced by [2]_. For this load algorithm see [1]_.

    References
    ----------
    .. [1] Mark E. J. Newman:
       Scientific collaboration networks. II.
       Shortest paths, weighted networks, and centrality.
       Physical Review E 64, 016132, 2001.
       http://journals.aps.org/pre/abstract/10.1103/PhysRevE.64.016132
    .. [2] Kwang-Il Goh, Byungnam Kahng and Doochul Kim
       Universal behavior of Load Distribution in Scale-Free Networks.
       Physical Review Letters 87(27):1–4, 2001.
       https://doi.org/10.1103/PhysRevLett.87.278701
    """
    if v is not None:  # only one node
        betweenness = 0.0
        for source in G:
            ubetween = _node_betweenness(G, source, cutoff, False, weight)
            betweenness += ubetween[v] if v in ubetween else 0
        if normalized:
            order = G.order()
            if order <= 2:
                return betweenness  # no normalization b=0 for all nodes
            betweenness *= 1.0 / ((order - 1) * (order - 2))
    else:
        betweenness = {}.fromkeys(G, 0.0)
        for source in betweenness:
            ubetween = _node_betweenness(G, source, cutoff, False, weight)
            for vk in ubetween:
                betweenness[vk] += ubetween[vk]
        if normalized:
            order = G.order()
            if order <= 2:
                return betweenness  # no normalization b=0 for all nodes
            scale = 1.0 / ((order - 1) * (order - 2))
            for v in betweenness:
                betweenness[v] *= scale
    return betweenness  # all nodes


def _node_betweenness(G, source, cutoff=False, normalized=True, weight=None):
    """Node betweenness_centrality helper:

    See betweenness_centrality for what you probably want.
    This actually computes "load" and not betweenness.
    See https://networkx.lanl.gov/ticket/103

    This calculates the load of each node for paths from a single source.
    (The fraction of number of shortests paths from source that go
    through each node.)

    To get the load for a node you need to do all-pairs shortest paths.

    If weight is not None then use Dijkstra for finding shortest paths.
    """
    # get the predecessor and path length data
    if weight is None:
        (pred, length) = nx.predecessor(G, source, cutoff=cutoff, return_seen=True)
    else:
        (pred, length) = nx.dijkstra_predecessor_and_distance(G, source, cutoff, weight)

    # order the nodes by path length
    onodes = [(l, vert) for (vert, l) in length.items()]
    onodes.sort()
    onodes[:] = [vert for (l, vert) in onodes if l > 0]

    # initialize betweenness
    between = {}.fromkeys(length, 1.0)

    while onodes:
        v = onodes.pop()
        if v in pred:
            num_paths = len(pred[v])  # Discount betweenness if more than
            for x in pred[v]:  # one shortest path.
                if x == source:  # stop if hit source because all remaining v
                    break  # also have pred[v]==[source]
                between[x] += between[v] / num_paths
    #  remove source
    for v in between:
        between[v] -= 1
    # rescale to be between 0 and 1
    if normalized:
        l = len(between)
        if l > 2:
            # scale by 1/the number of possible paths
            scale = 1 / ((l - 1) * (l - 2))
            for v in between:
                between[v] *= scale
    return between


load_centrality = newman_betweenness_centrality


@nx._dispatchable
def edge_load_centrality(G, cutoff=False):
    """Compute edge load.

    WARNING: This concept of edge load has not been analysed
    or discussed outside of NetworkX that we know of.
    It is based loosely on load_centrality in the sense that
    it counts the number of shortest paths which cross each edge.
    This function is for demonstration and testing purposes.

    Parameters
    ----------
    G : graph
        A networkx graph

    cutoff : bool, optional (default=False)
        If specified, only consider paths of length <= cutoff.

    Returns
    -------
    A dict keyed by edge 2-tuple to the number of shortest paths
    which use that edge. Where more than one path is shortest
    the count is divided equally among paths.
    """
    betweenness = {}
    for u, v in G.edges():
        betweenness[(u, v)] = 0.0
        betweenness[(v, u)] = 0.0

    for source in G:
        ubetween = _edge_betweenness(G, source, cutoff=cutoff)
        for e, ubetweenv in ubetween.items():
            betweenness[e] += ubetweenv  # cumulative total
    return betweenness


def _edge_betweenness(G, source, nodes=None, cutoff=False):
    """Edge betweenness helper."""
    # get the predecessor data
    (pred, length) = nx.predecessor(G, source, cutoff=cutoff, return_seen=True)
    # order the nodes by path length
    onodes = [n for n, d in sorted(length.items(), key=itemgetter(1))]
    # initialize betweenness, doesn't account for any edge weights
    between = {}
    for u, v in G.edges(nodes):
        between[(u, v)] = 1.0
        between[(v, u)] = 1.0

    while onodes:  # work through all paths
        v = onodes.pop()
        if v in pred:
            # Discount betweenness if more than one shortest path.
            num_paths = len(pred[v])
            for w in pred[v]:
                if w in pred:
                    # Discount betweenness, mult path
                    num_paths = len(pred[w])
                    for x in pred[w]:
                        between[(w, x)] += between[(v, w)] / num_paths
                        between[(x, w)] += between[(w, v)] / num_paths
    return between
<|endoftext|>"""Current-flow betweenness centrality measures for subsets of nodes."""

import networkx as nx
from networkx.algorithms.centrality.flow_matrix import flow_matrix_row
from networkx.utils import not_implemented_for, reverse_cuthill_mckee_ordering

__all__ = [
    "current_flow_betweenness_centrality_subset",
    "edge_current_flow_betweenness_centrality_subset",
]


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def current_flow_betweenness_centrality_subset(
    G, sources, targets, normalized=True, weight=None, dtype=float, solver="lu"
):
    r"""Compute current-flow betweenness centrality for subsets of nodes.

    Current-flow betweenness centrality uses an electrical current
    model for information spreading in contrast to betweenness
    centrality which uses shortest paths.

    Current-flow betweenness centrality is also known as
    random-walk betweenness centrality [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    sources: list of nodes
      Nodes to use as sources for current

    targets: list of nodes
      Nodes to use as sinks for current

    normalized : bool, optional (default=True)
      If True the betweenness values are normalized by b=b/(n-1)(n-2) where
      n is the number of nodes in G.

    weight : string or None, optional (default=None)
      Key for edge data used as the edge weight.
      If None, then use 1 as each edge weight.
      The weight reflects the capacity or the strength of the
      edge.

    dtype: data type (float)
      Default data type for internal matrices.
      Set to np.float32 for lower memory consumption.

    solver: string (default='lu')
       Type of linear solver to use for computing the flow matrix.
       Options are "full" (uses most memory), "lu" (recommended), and
       "cg" (uses least memory).

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.

    See Also
    --------
    approximate_current_flow_betweenness_centrality
    betweenness_centrality
    edge_betweenness_centrality
    edge_current_flow_betweenness_centrality

    Notes
    -----
    Current-flow betweenness can be computed in $O(I(n-1)+mn \log n)$
    time [1]_, where $I(n-1)$ is the time needed to compute the
    inverse Laplacian.  For a full matrix this is $O(n^3)$ but using
    sparse methods you can achieve $O(nm{\sqrt k})$ where $k$ is the
    Laplacian matrix condition number.

    The space required is $O(nw)$ where $w$ is the width of the sparse
    Laplacian matrix.  Worse case is $w=n$ for $O(n^2)$.

    If the edges have a 'weight' attribute they will be used as
    weights in this algorithm.  Unspecified weights are set to 1.

    References
    ----------
    .. [1] Centrality Measures Based on Current Flow.
       Ulrik Brandes and Daniel Fleischer,
       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS '05).
       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.
       https://doi.org/10.1007/978-3-540-31856-9_44

    .. [2] A measure of betweenness centrality based on random walks,
       M. E. J. Newman, Social Networks 27, 39-54 (2005).
    """
    import numpy as np

    from networkx.utils import reverse_cuthill_mckee_ordering

    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph not connected.")
    N = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    # make a copy with integer labels according to rcm ordering
    # this could be done without a copy if we really wanted to
    mapping = dict(zip(ordering, range(N)))
    H = nx.relabel_nodes(G, mapping)
    betweenness = dict.fromkeys(H, 0.0)  # b[n]=0 for n in H
    for row, (s, t) in flow_matrix_row(H, weight=weight, dtype=dtype, solver=solver):
        for ss in sources:
            i = mapping[ss]
            for tt in targets:
                j = mapping[tt]
                betweenness[s] += 0.5 * abs(row.item(i) - row.item(j))
                betweenness[t] += 0.5 * abs(row.item(i) - row.item(j))
    if normalized:
        nb = (N - 1.0) * (N - 2.0)  # normalization factor
    else:
        nb = 2.0
    for node in H:
        betweenness[node] = betweenness[node] / nb + 1.0 / (2 - N)
    return {ordering[node]: value for node, value in betweenness.items()}


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def edge_current_flow_betweenness_centrality_subset(
    G, sources, targets, normalized=True, weight=None, dtype=float, solver="lu"
):
    r"""Compute current-flow betweenness centrality for edges using subsets
    of nodes.

    Current-flow betweenness centrality uses an electrical current
    model for information spreading in contrast to betweenness
    centrality which uses shortest paths.

    Current-flow betweenness centrality is also known as
    random-walk betweenness centrality [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    sources: list of nodes
      Nodes to use as sources for current

    targets: list of nodes
      Nodes to use as sinks for current

    normalized : bool, optional (default=True)
      If True the betweenness values are normalized by b=b/(n-1)(n-2) where
      n is the number of nodes in G.

    weight : string or None, optional (default=None)
      Key for edge data used as the edge weight.
      If None, then use 1 as each edge weight.
      The weight reflects the capacity or the strength of the
      edge.

    dtype: data type (float)
      Default data type for internal matrices.
      Set to np.float32 for lower memory consumption.

    solver: string (default='lu')
       Type of linear solver to use for computing the flow matrix.
       Options are "full" (uses most memory), "lu" (recommended), and
       "cg" (uses least memory).

    Returns
    -------
    nodes : dict
       Dictionary of edge tuples with betweenness centrality as the value.

    See Also
    --------
    betweenness_centrality
    edge_betweenness_centrality
    current_flow_betweenness_centrality

    Notes
    -----
    Current-flow betweenness can be computed in $O(I(n-1)+mn \log n)$
    time [1]_, where $I(n-1)$ is the time needed to compute the
    inverse Laplacian.  For a full matrix this is $O(n^3)$ but using
    sparse methods you can achieve $O(nm{\sqrt k})$ where $k$ is the
    Laplacian matrix condition number.

    The space required is $O(nw)$ where $w$ is the width of the sparse
    Laplacian matrix.  Worse case is $w=n$ for $O(n^2)$.

    If the edges have a 'weight' attribute they will be used as
    weights in this algorithm.  Unspecified weights are set to 1.

    References
    ----------
    .. [1] Centrality Measures Based on Current Flow.
       Ulrik Brandes and Daniel Fleischer,
       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS '05).
       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.
       https://doi.org/10.1007/978-3-540-31856-9_44

    .. [2] A measure of betweenness centrality based on random walks,
       M. E. J. Newman, Social Networks 27, 39-54 (2005).
    """
    import numpy as np

    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph not connected.")
    N = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    # make a copy with integer labels according to rcm ordering
    # this could be done without a copy if we really wanted to
    mapping = dict(zip(ordering, range(N)))
    H = nx.relabel_nodes(G, mapping)
    edges = (tuple(sorted((u, v))) for u, v in H.edges())
    betweenness = dict.fromkeys(edges, 0.0)
    if normalized:
        nb = (N - 1.0) * (N - 2.0)  # normalization factor
    else:
        nb = 2.0
    for row, (e) in flow_matrix_row(H, weight=weight, dtype=dtype, solver=solver):
        for ss in sources:
            i = mapping[ss]
            for tt in targets:
                j = mapping[tt]
                betweenness[e] += 0.5 * abs(row.item(i) - row.item(j))
        betweenness[e] /= nb
    return {(ordering[s], ordering[t]): value for (s, t), value in betweenness.items()}
<|endoftext|>"""Percolation centrality measures."""

import networkx as nx
from networkx.algorithms.centrality.betweenness import (
    _single_source_dijkstra_path_basic as dijkstra,
)
from networkx.algorithms.centrality.betweenness import (
    _single_source_shortest_path_basic as shortest_path,
)

__all__ = ["percolation_centrality"]


@nx._dispatchable(node_attrs="attribute", edge_attrs="weight")
def percolation_centrality(G, attribute="percolation", states=None, weight=None):
    r"""Compute the percolation centrality for nodes.

    Percolation centrality of a node $v$, at a given time, is defined
    as the proportion of ‘percolated paths’ that go through that node.

    This measure quantifies relative impact of nodes based on their
    topological connectivity, as well as their percolation states.

    Percolation states of nodes are used to depict network percolation
    scenarios (such as during infection transmission in a social network
    of individuals, spreading of computer viruses on computer networks, or
    transmission of disease over a network of towns) over time. In this
    measure usually the percolation state is expressed as a decimal
    between 0.0 and 1.0.

    When all nodes are in the same percolated state this measure is
    equivalent to betweenness centrality.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    attribute : None or string, optional (default='percolation')
      Name of the node attribute to use for percolation state, used
      if `states` is None. If a node does not set the attribute the
      state of that node will be set to the default value of 1.
      If all nodes do not have the attribute all nodes will be set to
      1 and the centrality measure will be equivalent to betweenness centrality.

    states : None or dict, optional (default=None)
      Specify percolation states for the nodes, nodes as keys states
      as values.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      The weight of an edge is treated as the length or distance between the two sides.


    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with percolation centrality as the value.

    See Also
    --------
    betweenness_centrality

    Notes
    -----
    The algorithm is from Mahendra Piraveenan, Mikhail Prokopenko, and
    Liaquat Hossain [1]_
    Pair dependencies are calculated and accumulated using [2]_

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    References
    ----------
    .. [1] Mahendra Piraveenan, Mikhail Prokopenko, Liaquat Hossain
       Percolation Centrality: Quantifying Graph-Theoretic Impact of Nodes
       during Percolation in Networks
       http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0053095
    .. [2] Ulrik Brandes:
       A Faster Algorithm for Betweenness Centrality.
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    """
    percolation = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G

    nodes = G

    if states is None:
        states = nx.get_node_attributes(nodes, attribute, default=1)

    # sum of all percolation states
    p_sigma_x_t = 0.0
    for v in states.values():
        p_sigma_x_t += v

    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma, _ = shortest_path(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma, _ = dijkstra(G, s, weight)
        # accumulation
        percolation = _accumulate_percolation(
            percolation, S, P, sigma, s, states, p_sigma_x_t
        )

    n = len(G)

    for v in percolation:
        percolation[v] *= 1 / (n - 2)

    return percolation


def _accumulate_percolation(percolation, S, P, sigma, s, states, p_sigma_x_t):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            # percolation weight
            pw_s_w = states[s] / (p_sigma_x_t - states[w])
            percolation[w] += delta[w] * pw_s_w
    return percolation
<|endoftext|>"""
Subraph centrality and communicability betweenness.
"""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "subgraph_centrality_exp",
    "subgraph_centrality",
    "communicability_betweenness_centrality",
    "estrada_index",
]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def subgraph_centrality_exp(G):
    r"""Returns the subgraph centrality for each node of G.

    Subgraph centrality  of a node `n` is the sum of weighted closed
    walks of all lengths starting and ending at node `n`. The weights
    decrease with path length. Each closed walk is associated with a
    connected subgraph ([1]_).

    Parameters
    ----------
    G: graph

    Returns
    -------
    nodes:dictionary
        Dictionary of nodes with subgraph centrality as the value.

    Raises
    ------
    NetworkXError
        If the graph is not undirected and simple.

    See Also
    --------
    subgraph_centrality:
        Alternative algorithm of the subgraph centrality for each node of G.

    Notes
    -----
    This version of the algorithm exponentiates the adjacency matrix.

    The subgraph centrality of a node `u` in G can be found using
    the matrix exponential of the adjacency matrix of G [1]_,

    .. math::

        SC(u)=(e^A)_{uu} .

    References
    ----------
    .. [1] Ernesto Estrada, Juan A. Rodriguez-Velazquez,
       "Subgraph centrality in complex networks",
       Physical Review E 71, 056103 (2005).
       https://arxiv.org/abs/cond-mat/0504730

    Examples
    --------
    (Example from [1]_)
    >>> G = nx.Graph(
    ...     [
    ...         (1, 2),
    ...         (1, 5),
    ...         (1, 8),
    ...         (2, 3),
    ...         (2, 8),
    ...         (3, 4),
    ...         (3, 6),
    ...         (4, 5),
    ...         (4, 7),
    ...         (5, 6),
    ...         (6, 7),
    ...         (7, 8),
    ...     ]
    ... )
    >>> sc = nx.subgraph_centrality_exp(G)
    >>> print([f"{node} {sc[node]:0.2f}" for node in sorted(sc)])
    ['1 3.90', '2 3.90', '3 3.64', '4 3.71', '5 3.64', '6 3.71', '7 3.64', '8 3.90']
    """
    # alternative implementation that calculates the matrix exponential
    import scipy as sp

    nodelist = list(G)  # ordering of nodes in matrix
    A = nx.to_numpy_array(G, nodelist)
    # convert to 0-1 matrix
    A[A != 0.0] = 1
    expA = sp.linalg.expm(A)
    # convert diagonal to dictionary keyed by node
    sc = dict(zip(nodelist, map(float, expA.diagonal())))
    return sc


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def subgraph_centrality(G):
    r"""Returns subgraph centrality for each node in G.

    Subgraph centrality  of a node `n` is the sum of weighted closed
    walks of all lengths starting and ending at node `n`. The weights
    decrease with path length. Each closed walk is associated with a
    connected subgraph ([1]_).

    Parameters
    ----------
    G: graph

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with subgraph centrality as the value.

    Raises
    ------
    NetworkXError
       If the graph is not undirected and simple.

    See Also
    --------
    subgraph_centrality_exp:
        Alternative algorithm of the subgraph centrality for each node of G.

    Notes
    -----
    This version of the algorithm computes eigenvalues and eigenvectors
    of the adjacency matrix.

    Subgraph centrality of a node `u` in G can be found using
    a spectral decomposition of the adjacency matrix [1]_,

    .. math::

       SC(u)=\sum_{j=1}^{N}(v_{j}^{u})^2 e^{\lambda_{j}},

    where `v_j` is an eigenvector of the adjacency matrix `A` of G
    corresponding to the eigenvalue `\lambda_j`.

    Examples
    --------
    (Example from [1]_)
    >>> G = nx.Graph(
    ...     [
    ...         (1, 2),
    ...         (1, 5),
    ...         (1, 8),
    ...         (2, 3),
    ...         (2, 8),
    ...         (3, 4),
    ...         (3, 6),
    ...         (4, 5),
    ...         (4, 7),
    ...         (5, 6),
    ...         (6, 7),
    ...         (7, 8),
    ...     ]
    ... )
    >>> sc = nx.subgraph_centrality(G)
    >>> print([f"{node} {sc[node]:0.2f}" for node in sorted(sc)])
    ['1 3.90', '2 3.90', '3 3.64', '4 3.71', '5 3.64', '6 3.71', '7 3.64', '8 3.90']

    References
    ----------
    .. [1] Ernesto Estrada, Juan A. Rodriguez-Velazquez,
       "Subgraph centrality in complex networks",
       Physical Review E 71, 056103 (2005).
       https://arxiv.org/abs/cond-mat/0504730

    """
    import numpy as np

    nodelist = list(G)  # ordering of nodes in matrix
    A = nx.to_numpy_array(G, nodelist)
    # convert to 0-1 matrix
    A[np.nonzero(A)] = 1
    w, v = np.linalg.eigh(A)
    vsquare = np.array(v) ** 2
    expw = np.exp(w)
    xg = vsquare @ expw
    # convert vector dictionary keyed by node
    sc = dict(zip(nodelist, map(float, xg)))
    return sc


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def communicability_betweenness_centrality(G):
    r"""Returns subgraph communicability for all pairs of nodes in G.

    Communicability betweenness measure makes use of the number of walks
    connecting every pair of nodes as the basis of a betweenness centrality
    measure.

    Parameters
    ----------
    G: graph

    Returns
    -------
    nodes : dictionary
        Dictionary of nodes with communicability betweenness as the value.

    Raises
    ------
    NetworkXError
        If the graph is not undirected and simple.

    Notes
    -----
    Let `G=(V,E)` be a simple undirected graph with `n` nodes and `m` edges,
    and `A` denote the adjacency matrix of `G`.

    Let `G(r)=(V,E(r))` be the graph resulting from
    removing all edges connected to node `r` but not the node itself.

    The adjacency matrix for `G(r)` is `A+E(r)`,  where `E(r)` has nonzeros
    only in row and column `r`.

    The subraph betweenness of a node `r`  is [1]_

    .. math::

         \omega_{r} = \frac{1}{C}\sum_{p}\sum_{q}\frac{G_{prq}}{G_{pq}},
         p\neq q, q\neq r,

    where
    `G_{prq}=(e^{A}_{pq} - (e^{A+E(r)})_{pq}`  is the number of walks
    involving node r,
    `G_{pq}=(e^{A})_{pq}` is the number of closed walks starting
    at node `p` and ending at node `q`,
    and `C=(n-1)^{2}-(n-1)` is a normalization factor equal to the
    number of terms in the sum.

    The resulting `\omega_{r}` takes values between zero and one.
    The lower bound cannot be attained for a connected
    graph, and the upper bound is attained in the star graph.

    References
    ----------
    .. [1] Ernesto Estrada, Desmond J. Higham, Naomichi Hatano,
       "Communicability Betweenness in Complex Networks"
       Physica A 388 (2009) 764-774.
       https://arxiv.org/abs/0905.4102

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])
    >>> cbc = nx.communicability_betweenness_centrality(G)
    >>> print([f"{node} {cbc[node]:0.2f}" for node in sorted(cbc)])
    ['0 0.03', '1 0.45', '2 0.51', '3 0.45', '4 0.40', '5 0.19', '6 0.03']
    """
    import numpy as np
    import scipy as sp

    nodelist = list(G)  # ordering of nodes in matrix
    n = len(nodelist)
    A = nx.to_numpy_array(G, nodelist)
    # convert to 0-1 matrix
    A[np.nonzero(A)] = 1
    expA = sp.linalg.expm(A)
    mapping = dict(zip(nodelist, range(n)))
    cbc = {}
    for v in G:
        # remove row and col of node v
        i = mapping[v]
        row = A[i, :].copy()
        col = A[:, i].copy()
        A[i, :] = 0
        A[:, i] = 0
        B = (expA - sp.linalg.expm(A)) / expA
        # sum with row/col of node v and diag set to zero
        B[i, :] = 0
        B[:, i] = 0
        B -= np.diag(np.diag(B))
        cbc[v] = float(B.sum())
        # put row and col back
        A[i, :] = row
        A[:, i] = col
    # rescale when more than two nodes
    order = len(cbc)
    if order > 2:
        scale = 1.0 / ((order - 1.0) ** 2 - (order - 1.0))
        cbc = {node: value * scale for node, value in cbc.items()}
    return cbc


@nx._dispatchable
def estrada_index(G):
    r"""Returns the Estrada index of a the graph G.

    The Estrada Index is a topological index of folding or 3D "compactness" ([1]_).

    Parameters
    ----------
    G: graph

    Returns
    -------
    estrada index: float

    Raises
    ------
    NetworkXError
        If the graph is not undirected and simple.

    Notes
    -----
    Let `G=(V,E)` be a simple undirected graph with `n` nodes  and let
    `\lambda_{1}\leq\lambda_{2}\leq\cdots\lambda_{n}`
    be a non-increasing ordering of the eigenvalues of its adjacency
    matrix `A`. The Estrada index is ([1]_, [2]_)

    .. math::
        EE(G)=\sum_{j=1}^n e^{\lambda _j}.

    References
    ----------
    .. [1] E. Estrada, "Characterization of 3D molecular structure",
       Chem. Phys. Lett. 319, 713 (2000).
       https://doi.org/10.1016/S0009-2614(00)00158-5
    .. [2] José Antonio de la Peñaa, Ivan Gutman, Juan Rada,
       "Estimating the Estrada index",
       Linear Algebra and its Applications. 427, 1 (2007).
       https://doi.org/10.1016/j.laa.2007.06.020

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])
    >>> ei = nx.estrada_index(G)
    >>> print(f"{ei:0.5}")
    20.55
    """
    return sum(subgraph_centrality(G).values())
<|endoftext|>"""Group centrality measures."""

from copy import deepcopy

import networkx as nx
from networkx.algorithms.centrality.betweenness import (
    _accumulate_endpoints,
    _single_source_dijkstra_path_basic,
    _single_source_shortest_path_basic,
)
from networkx.utils.decorators import not_implemented_for

__all__ = [
    "group_betweenness_centrality",
    "group_closeness_centrality",
    "group_degree_centrality",
    "group_in_degree_centrality",
    "group_out_degree_centrality",
    "prominent_group",
]


@nx._dispatchable(edge_attrs="weight")
def group_betweenness_centrality(G, C, normalized=True, weight=None, endpoints=False):
    r"""Compute the group betweenness centrality for a group of nodes.

    Group betweenness centrality of a group of nodes $C$ is the sum of the
    fraction of all-pairs shortest paths that pass through any vertex in $C$

    .. math::

       c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\sigma(s, t|C)$ is the number of
    those paths passing through some node in group $C$. Note that
    $(s, t)$ are not members of the group ($V-C$ is the set of nodes
    in $V$ that are not in $C$).

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    C : list or set or list of lists or list of sets
      A group or a list of groups containing nodes which belong to G, for which group betweenness
      centrality is to be calculated.

    normalized : bool, optional (default=True)
      If True, group betweenness is normalized by `1/((|V|-|C|)(|V|-|C|-1))`
      where `|V|` is the number of nodes in G and `|C|` is the number of nodes in C.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      The weight of an edge is treated as the length or distance between the two sides.

    endpoints : bool, optional (default=False)
      If True include the endpoints in the shortest path counts.

    Raises
    ------
    NodeNotFound
       If node(s) in C are not present in G.

    Returns
    -------
    betweenness : list of floats or float
       If C is a single group then return a float. If C is a list with
       several groups then return a list of group betweenness centralities.

    See Also
    --------
    betweenness_centrality

    Notes
    -----
    Group betweenness centrality is described in [1]_ and its importance discussed in [3]_.
    The initial implementation of the algorithm is mentioned in [2]_. This function uses
    an improved algorithm presented in [4]_.

    The number of nodes in the group must be a maximum of n - 2 where `n`
    is the total number of nodes in the graph.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    The total number of paths between source and target is counted
    differently for directed and undirected graphs. Directed paths
    between "u" and "v" are counted as two possible paths (one each
    direction) while undirected paths between "u" and "v" are counted
    as one path. Said another way, the sum in the expression above is
    over all ``s != t`` for directed graphs and for ``s < t`` for undirected graphs.


    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    .. [2] Ulrik Brandes:
       On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.72.9610&rep=rep1&type=pdf
    .. [3] Sourav Medya et. al.:
       Group Centrality Maximization via Network Design.
       SIAM International Conference on Data Mining, SDM 2018, 126–134.
       https://sites.cs.ucsb.edu/~arlei/pubs/sdm18.pdf
    .. [4] Rami Puzis, Yuval Elovici, and Shlomi Dolev.
       "Fast algorithm for successive computation of group betweenness centrality."
       https://journals.aps.org/pre/pdf/10.1103/PhysRevE.76.056709

    """
    GBC = []  # initialize betweenness
    list_of_groups = True
    #  check weather C contains one or many groups
    if any(el in G for el in C):
        C = [C]
        list_of_groups = False
    set_v = {node for group in C for node in group}
    if set_v - G.nodes:  # element(s) of C not in G
        raise nx.NodeNotFound(f"The node(s) {set_v - G.nodes} are in C but not in G.")

    # pre-processing
    PB, sigma, D = _group_preprocessing(G, set_v, weight)

    # the algorithm for each group
    for group in C:
        group = set(group)  # set of nodes in group
        # initialize the matrices of the sigma and the PB
        GBC_group = 0
        sigma_m = deepcopy(sigma)
        PB_m = deepcopy(PB)
        sigma_m_v = deepcopy(sigma_m)
        PB_m_v = deepcopy(PB_m)
        for v in group:
            GBC_group += PB_m[v][v]
            for x in group:
                for y in group:
                    dxvy = 0
                    dxyv = 0
                    dvxy = 0
                    if not (
                        sigma_m[x][y] == 0 or sigma_m[x][v] == 0 or sigma_m[v][y] == 0
                    ):
                        if D[x][v] == D[x][y] + D[y][v]:
                            dxyv = sigma_m[x][y] * sigma_m[y][v] / sigma_m[x][v]
                        if D[x][y] == D[x][v] + D[v][y]:
                            dxvy = sigma_m[x][v] * sigma_m[v][y] / sigma_m[x][y]
                        if D[v][y] == D[v][x] + D[x][y]:
                            dvxy = sigma_m[v][x] * sigma[x][y] / sigma[v][y]
                    sigma_m_v[x][y] = sigma_m[x][y] * (1 - dxvy)
                    PB_m_v[x][y] = PB_m[x][y] - PB_m[x][y] * dxvy
                    if y != v:
                        PB_m_v[x][y] -= PB_m[x][v] * dxyv
                    if x != v:
                        PB_m_v[x][y] -= PB_m[v][y] * dvxy
            sigma_m, sigma_m_v = sigma_m_v, sigma_m
            PB_m, PB_m_v = PB_m_v, PB_m

        # endpoints
        v, c = len(G), len(group)
        if not endpoints:
            scale = 0
            # if the graph is connected then subtract the endpoints from
            # the count for all the nodes in the graph. else count how many
            # nodes are connected to the group's nodes and subtract that.
            if nx.is_directed(G):
                if nx.is_strongly_connected(G):
                    scale = c * (2 * v - c - 1)
            elif nx.is_connected(G):
                scale = c * (2 * v - c - 1)
            if scale == 0:
                for group_node1 in group:
                    for node in D[group_node1]:
                        if node != group_node1:
                            if node in group:
                                scale += 1
                            else:
                                scale += 2
            GBC_group -= scale

        # normalized
        if normalized:
            scale = 1 / ((v - c) * (v - c - 1))
            GBC_group *= scale

        # If undirected than count only the undirected edges
        elif not G.is_directed():
            GBC_group /= 2

        GBC.append(GBC_group)
    if list_of_groups:
        return GBC
    return GBC[0]


def _group_preprocessing(G, set_v, weight):
    sigma = {}
    delta = {}
    D = {}
    betweenness = dict.fromkeys(G, 0)
    for s in G:
        if weight is None:  # use BFS
            S, P, sigma[s], D[s] = _single_source_shortest_path_basic(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma[s], D[s] = _single_source_dijkstra_path_basic(G, s, weight)
        betweenness, delta[s] = _accumulate_endpoints(betweenness, S, P, sigma[s], s)
        for i in delta[s]:  # add the paths from s to i and rescale sigma
            if s != i:
                delta[s][i] += 1
            if weight is not None:
                sigma[s][i] = sigma[s][i] / 2
    # building the path betweenness matrix only for nodes that appear in the group
    PB = dict.fromkeys(G)
    for group_node1 in set_v:
        PB[group_node1] = dict.fromkeys(G, 0.0)
        for group_node2 in set_v:
            if group_node2 not in D[group_node1]:
                continue
            for node in G:
                # if node is connected to the two group nodes than continue
                if group_node2 in D[node] and group_node1 in D[node]:
                    if (
                        D[node][group_node2]
                        == D[node][group_node1] + D[group_node1][group_node2]
                    ):
                        PB[group_node1][group_node2] += (
                            delta[node][group_node2]
                            * sigma[node][group_node1]
                            * sigma[group_node1][group_node2]
                            / sigma[node][group_node2]
                        )
    return PB, sigma, D


@nx._dispatchable(edge_attrs="weight")
def prominent_group(
    G, k, weight=None, C=None, endpoints=False, normalized=True, greedy=False
):
    r"""Find the prominent group of size $k$ in graph $G$. The prominence of the
    group is evaluated by the group betweenness centrality.

    Group betweenness centrality of a group of nodes $C$ is the sum of the
    fraction of all-pairs shortest paths that pass through any vertex in $C$

    .. math::

       c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\sigma(s, t|C)$ is the number of
    those paths passing through some node in group $C$. Note that
    $(s, t)$ are not members of the group ($V-C$ is the set of nodes
    in $V$ that are not in $C$).

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    k : int
       The number of nodes in the group.

    normalized : bool, optional (default=True)
       If True, group betweenness is normalized by ``1/((|V|-|C|)(|V|-|C|-1))``
       where ``|V|`` is the number of nodes in G and ``|C|`` is the number of
       nodes in C.

    weight : None or string, optional (default=None)
       If None, all edge weights are considered equal.
       Otherwise holds the name of the edge attribute used as weight.
       The weight of an edge is treated as the length or distance between the two sides.

    endpoints : bool, optional (default=False)
       If True include the endpoints in the shortest path counts.

    C : list or set, optional (default=None)
       list of nodes which won't be candidates of the prominent group.

    greedy : bool, optional (default=False)
       Using a naive greedy algorithm in order to find non-optimal prominent
       group. For scale free networks the results are negligibly below the optimal
       results.

    Raises
    ------
    NodeNotFound
       If node(s) in C are not present in G.

    Returns
    -------
    max_GBC : float
       The group betweenness centrality of the prominent group.

    max_group : list
        The list of nodes in the prominent group.

    See Also
    --------
    betweenness_centrality, group_betweenness_centrality

    Notes
    -----
    Group betweenness centrality is described in [1]_ and its importance discussed in [3]_.
    The algorithm is described in [2]_ and is based on techniques mentioned in [4]_.

    The number of nodes in the group must be a maximum of ``n - 2`` where ``n``
    is the total number of nodes in the graph.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    The total number of paths between source and target is counted
    differently for directed and undirected graphs. Directed paths
    between "u" and "v" are counted as two possible paths (one each
    direction) while undirected paths between "u" and "v" are counted
    as one path. Said another way, the sum in the expression above is
    over all ``s != t`` for directed graphs and for ``s < t`` for undirected graphs.

    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    .. [2] Rami Puzis, Yuval Elovici, and Shlomi Dolev:
       "Finding the Most Prominent Group in Complex Networks"
       AI communications 20(4): 287-296, 2007.
       https://www.researchgate.net/profile/Rami_Puzis2/publication/220308855
    .. [3] Sourav Medya et. al.:
       Group Centrality Maximization via Network Design.
       SIAM International Conference on Data Mining, SDM 2018, 126–134.
       https://sites.cs.ucsb.edu/~arlei/pubs/sdm18.pdf
    .. [4] Rami Puzis, Yuval Elovici, and Shlomi Dolev.
       "Fast algorithm for successive computation of group betweenness centrality."
       https://journals.aps.org/pre/pdf/10.1103/PhysRevE.76.056709
    """
    import numpy as np
    import pandas as pd

    if C is not None:
        C = set(C)
        if C - G.nodes:  # element(s) of C not in G
            raise nx.NodeNotFound(f"The node(s) {C - G.nodes} are in C but not in G.")
        nodes = list(G.nodes - C)
    else:
        nodes = list(G.nodes)
    DF_tree = nx.Graph()
    DF_tree.__networkx_cache__ = None  # Disable caching
    PB, sigma, D = _group_preprocessing(G, nodes, weight)
    betweenness = pd.DataFrame.from_dict(PB)
    if C is not None:
        for node in C:
            # remove from the betweenness all the nodes not part of the group
            betweenness.drop(index=node, inplace=True)
            betweenness.drop(columns=node, inplace=True)
    CL = [node for _, node in sorted(zip(np.diag(betweenness), nodes), reverse=True)]
    max_GBC = 0
    max_group = []
    DF_tree.add_node(
        1,
        CL=CL,
        betweenness=betweenness,
        GBC=0,
        GM=[],
        sigma=sigma,
        cont=dict(zip(nodes, np.diag(betweenness))),
    )

    # the algorithm
    DF_tree.nodes[1]["heu"] = 0
    for i in range(k):
        DF_tree.nodes[1]["heu"] += DF_tree.nodes[1]["cont"][DF_tree.nodes[1]["CL"][i]]
    max_GBC, DF_tree, max_group = _dfbnb(
        G, k, DF_tree, max_GBC, 1, D, max_group, nodes, greedy
    )

    v = len(G)
    if not endpoints:
        scale = 0
        # if the graph is connected then subtract the endpoints from
        # the count for all the nodes in the graph. else count how many
        # nodes are connected to the group's nodes and subtract that.
        if nx.is_directed(G):
            if nx.is_strongly_connected(G):
                scale = k * (2 * v - k - 1)
        elif nx.is_connected(G):
            scale = k * (2 * v - k - 1)
        if scale == 0:
            for group_node1 in max_group:
                for node in D[group_node1]:
                    if node != group_node1:
                        if node in max_group:
                            scale += 1
                        else:
                            scale += 2
        max_GBC -= scale

    # normalized
    if normalized:
        scale = 1 / ((v - k) * (v - k - 1))
        max_GBC *= scale

    # If undirected then count only the undirected edges
    elif not G.is_directed():
        max_GBC /= 2
    max_GBC = float(f"{max_GBC:.2f}")
    return max_GBC, max_group


def _dfbnb(G, k, DF_tree, max_GBC, root, D, max_group, nodes, greedy):
    # stopping condition - if we found a group of size k and with higher GBC then prune
    if len(DF_tree.nodes[root]["GM"]) == k and DF_tree.nodes[root]["GBC"] > max_GBC:
        return DF_tree.nodes[root]["GBC"], DF_tree, DF_tree.nodes[root]["GM"]
    # stopping condition - if the size of group members equal to k or there are less than
    # k - |GM| in the candidate list or the heuristic function plus the GBC is below the
    # maximal GBC found then prune
    if (
        len(DF_tree.nodes[root]["GM"]) == k
        or len(DF_tree.nodes[root]["CL"]) <= k - len(DF_tree.nodes[root]["GM"])
        or DF_tree.nodes[root]["GBC"] + DF_tree.nodes[root]["heu"] <= max_GBC
    ):
        return max_GBC, DF_tree, max_group

    # finding the heuristic of both children
    node_p, node_m, DF_tree = _heuristic(k, root, DF_tree, D, nodes, greedy)

    # finding the child with the bigger heuristic + GBC and expand
    # that node first if greedy then only expand the plus node
    if greedy:
        max_GBC, DF_tree, max_group = _dfbnb(
            G, k, DF_tree, max_GBC, node_p, D, max_group, nodes, greedy
        )

    elif (
        DF_tree.nodes[node_p]["GBC"] + DF_tree.nodes[node_p]["heu"]
        > DF_tree.nodes[node_m]["GBC"] + DF_tree.nodes[node_m]["heu"]
    ):
        max_GBC, DF_tree, max_group = _dfbnb(
            G, k, DF_tree, max_GBC, node_p, D, max_group, nodes, greedy
        )
        max_GBC, DF_tree, max_group = _dfbnb(
            G, k, DF_tree, max_GBC, node_m, D, max_group, nodes, greedy
        )
    else:
        max_GBC, DF_tree, max_group = _dfbnb(
            G, k, DF_tree, max_GBC, node_m, D, max_group, nodes, greedy
        )
        max_GBC, DF_tree, max_group = _dfbnb(
            G, k, DF_tree, max_GBC, node_p, D, max_group, nodes, greedy
        )
    return max_GBC, DF_tree, max_group


def _heuristic(k, root, DF_tree, D, nodes, greedy):
    import numpy as np

    # This helper function add two nodes to DF_tree - one left son and the
    # other right son, finds their heuristic, CL, GBC, and GM
    node_p = DF_tree.number_of_nodes() + 1
    node_m = DF_tree.number_of_nodes() + 2
    added_node = DF_tree.nodes[root]["CL"][0]

    # adding the plus node
    DF_tree.add_nodes_from([(node_p, deepcopy(DF_tree.nodes[root]))])
    DF_tree.nodes[node_p]["GM"].append(added_node)
    DF_tree.nodes[node_p]["GBC"] += DF_tree.nodes[node_p]["cont"][added_node]
    root_node = DF_tree.nodes[root]
    for x in nodes:
        for y in nodes:
            dxvy = 0
            dxyv = 0
            dvxy = 0
            if not (
                root_node["sigma"][x][y] == 0
                or root_node["sigma"][x][added_node] == 0
                or root_node["sigma"][added_node][y] == 0
            ):
                if D[x][added_node] == D[x][y] + D[y][added_node]:
                    dxyv = (
                        root_node["sigma"][x][y]
                        * root_node["sigma"][y][added_node]
                        / root_node["sigma"][x][added_node]
                    )
                if D[x][y] == D[x][added_node] + D[added_node][y]:
                    dxvy = (
                        root_node["sigma"][x][added_node]
                        * root_node["sigma"][added_node][y]
                        / root_node["sigma"][x][y]
                    )
                if D[added_node][y] == D[added_node][x] + D[x][y]:
                    dvxy = (
                        root_node["sigma"][added_node][x]
                        * root_node["sigma"][x][y]
                        / root_node["sigma"][added_node][y]
                    )
            DF_tree.nodes[node_p]["sigma"][x][y] = root_node["sigma"][x][y] * (1 - dxvy)
            DF_tree.nodes[node_p]["betweenness"].loc[y, x] = (
                root_node["betweenness"][x][y] - root_node["betweenness"][x][y] * dxvy
            )
            if y != added_node:
                DF_tree.nodes[node_p]["betweenness"].loc[y, x] -= (
                    root_node["betweenness"][x][added_node] * dxyv
                )
            if x != added_node:
                DF_tree.nodes[node_p]["betweenness"].loc[y, x] -= (
                    root_node["betweenness"][added_node][y] * dvxy
                )

    DF_tree.nodes[node_p]["CL"] = [
        node
        for _, node in sorted(
            zip(np.diag(DF_tree.nodes[node_p]["betweenness"]), nodes), reverse=True
        )
        if node not in DF_tree.nodes[node_p]["GM"]
    ]
    DF_tree.nodes[node_p]["cont"] = dict(
        zip(nodes, np.diag(DF_tree.nodes[node_p]["betweenness"]))
    )
    DF_tree.nodes[node_p]["heu"] = 0
    for i in range(k - len(DF_tree.nodes[node_p]["GM"])):
        DF_tree.nodes[node_p]["heu"] += DF_tree.nodes[node_p]["cont"][
            DF_tree.nodes[node_p]["CL"][i]
        ]

    # adding the minus node - don't insert the first node in the CL to GM
    # Insert minus node only if isn't greedy type algorithm
    if not greedy:
        DF_tree.add_nodes_from([(node_m, deepcopy(DF_tree.nodes[root]))])
        DF_tree.nodes[node_m]["CL"].pop(0)
        DF_tree.nodes[node_m]["cont"].pop(added_node)
        DF_tree.nodes[node_m]["heu"] = 0
        for i in range(k - len(DF_tree.nodes[node_m]["GM"])):
            DF_tree.nodes[node_m]["heu"] += DF_tree.nodes[node_m]["cont"][
                DF_tree.nodes[node_m]["CL"][i]
            ]
    else:
        node_m = None

    return node_p, node_m, DF_tree


@nx._dispatchable(edge_attrs="weight")
def group_closeness_centrality(G, S, weight=None):
    r"""Compute the group closeness centrality for a group of nodes.

    Group closeness centrality of a group of nodes $S$ is a measure
    of how close the group is to the other nodes in the graph.

    .. math::

       c_{close}(S) = \frac{|V-S|}{\sum_{v \in V-S} d_{S, v}}

       d_{S, v} = min_{u \in S} (d_{u, v})

    where $V$ is the set of nodes, $d_{S, v}$ is the distance of
    the group $S$ from $v$ defined as above. ($V-S$ is the set of nodes
    in $V$ that are not in $S$).

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    S : list or set
       S is a group of nodes which belong to G, for which group closeness
       centrality is to be calculated.

    weight : None or string, optional (default=None)
       If None, all edge weights are considered equal.
       Otherwise holds the name of the edge attribute used as weight.
       The weight of an edge is treated as the length or distance between the two sides.

    Raises
    ------
    NodeNotFound
       If node(s) in S are not present in G.

    Returns
    -------
    closeness : float
       Group closeness centrality of the group S.

    See Also
    --------
    closeness_centrality

    Notes
    -----
    The measure was introduced in [1]_.
    The formula implemented here is described in [2]_.

    Higher values of closeness indicate greater centrality.

    It is assumed that 1 / 0 is 0 (required in the case of directed graphs,
    or when a shortest path length is 0).

    The number of nodes in the group must be a maximum of n - 1 where `n`
    is the total number of nodes in the graph.

    For directed graphs, the incoming distance is utilized here. To use the
    outward distance, act on `G.reverse()`.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    .. [2] J. Zhao et. al.:
       Measuring and Maximizing Group Closeness Centrality over
       Disk Resident Graphs.
       WWWConference Proceedings, 2014. 689-694.
       https://doi.org/10.1145/2567948.2579356
    """
    if G.is_directed():
        G = G.reverse()  # reverse view
    closeness = 0  # initialize to 0
    V = set(G)  # set of nodes in G
    S = set(S)  # set of nodes in group S
    V_S = V - S  # set of nodes in V but not S
    shortest_path_lengths = nx.multi_source_dijkstra_path_length(G, S, weight=weight)
    # accumulation
    for v in V_S:
        try:
            closeness += shortest_path_lengths[v]
        except KeyError:  # no path exists
            closeness += 0
    try:
        closeness = len(V_S) / closeness
    except ZeroDivisionError:  # 1 / 0 assumed as 0
        closeness = 0
    return closeness


@nx._dispatchable
def group_degree_centrality(G, S):
    """Compute the group degree centrality for a group of nodes.

    Group degree centrality of a group of nodes $S$ is the fraction
    of non-group members connected to group members.

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    S : list or set
       S is a group of nodes which belong to G, for which group degree
       centrality is to be calculated.

    Raises
    ------
    NetworkXError
       If node(s) in S are not in G.

    Returns
    -------
    centrality : float
       Group degree centrality of the group S.

    See Also
    --------
    degree_centrality
    group_in_degree_centrality
    group_out_degree_centrality

    Notes
    -----
    The measure was introduced in [1]_.

    The number of nodes in the group must be a maximum of n - 1 where `n`
    is the total number of nodes in the graph.

    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    """
    centrality = len(set().union(*[set(G.neighbors(i)) for i in S]) - set(S))
    centrality /= len(G.nodes()) - len(S)
    return centrality


@not_implemented_for("undirected")
@nx._dispatchable
def group_in_degree_centrality(G, S):
    """Compute the group in-degree centrality for a group of nodes.

    Group in-degree centrality of a group of nodes $S$ is the fraction
    of non-group members connected to group members by incoming edges.

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    S : list or set
       S is a group of nodes which belong to G, for which group in-degree
       centrality is to be calculated.

    Returns
    -------
    centrality : float
       Group in-degree centrality of the group S.

    Raises
    ------
    NetworkXNotImplemented
       If G is undirected.

    NodeNotFound
       If node(s) in S are not in G.

    See Also
    --------
    degree_centrality
    group_degree_centrality
    group_out_degree_centrality

    Notes
    -----
    The number of nodes in the group must be a maximum of n - 1 where `n`
    is the total number of nodes in the graph.

    `G.neighbors(i)` gives nodes with an outward edge from i, in a DiGraph,
    so for group in-degree centrality, the reverse graph is used.
    """
    return group_degree_centrality(G.reverse(), S)


@not_implemented_for("undirected")
@nx._dispatchable
def group_out_degree_centrality(G, S):
    """Compute the group out-degree centrality for a group of nodes.

    Group out-degree centrality of a group of nodes $S$ is the fraction
    of non-group members connected to group members by outgoing edges.

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    S : list or set
       S is a group of nodes which belong to G, for which group in-degree
       centrality is to be calculated.

    Returns
    -------
    centrality : float
       Group out-degree centrality of the group S.

    Raises
    ------
    NetworkXNotImplemented
       If G is undirected.

    NodeNotFound
       If node(s) in S are not in G.

    See Also
    --------
    degree_centrality
    group_degree_centrality
    group_in_degree_centrality

    Notes
    -----
    The number of nodes in the group must be a maximum of n - 1 where `n`
    is the total number of nodes in the graph.

    `G.neighbors(i)` gives nodes with an outward edge from i, in a DiGraph,
    so for group out-degree centrality, the graph itself is used.
    """
    return group_degree_centrality(G, S)
<|endoftext|>"""Functions for computing the harmonic centrality of a graph."""

from functools import partial

import networkx as nx

__all__ = ["harmonic_centrality"]


@nx._dispatchable(edge_attrs="distance")
def harmonic_centrality(G, nbunch=None, distance=None, sources=None):
    r"""Compute harmonic centrality for nodes.

    Harmonic centrality [1]_ of a node `u` is the sum of the reciprocal
    of the shortest path distances from all other nodes to `u`

    .. math::

        C(u) = \sum_{v \neq u} \frac{1}{d(v, u)}

    where `d(v, u)` is the shortest-path distance between `v` and `u`.

    If `sources` is given as an argument, the returned harmonic centrality
    values are calculated as the sum of the reciprocals of the shortest
    path distances from the nodes specified in `sources` to `u` instead
    of from all nodes to `u`.

    Notice that higher values indicate higher centrality.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    nbunch : container (default: all nodes in G)
      Container of nodes for which harmonic centrality values are calculated.

    sources : container (default: all nodes in G)
      Container of nodes `v` over which reciprocal distances are computed.
      Nodes not in `G` are silently ignored.

    distance : edge attribute key, optional (default=None)
      Use the specified edge attribute as the edge distance in shortest
      path calculations.  If `None`, then each edge will have distance equal to 1.

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with harmonic centrality as the value.

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality,
    degree_centrality, closeness_centrality

    Notes
    -----
    If the 'distance' keyword is set to an edge attribute key then the
    shortest-path length will be computed using Dijkstra's algorithm with
    that edge attribute as the edge weight.

    References
    ----------
    .. [1] Boldi, Paolo, and Sebastiano Vigna. "Axioms for centrality."
           Internet Mathematics 10.3-4 (2014): 222-262.
    """

    nbunch = set(G.nbunch_iter(nbunch) if nbunch is not None else G.nodes)
    sources = set(G.nbunch_iter(sources) if sources is not None else G.nodes)

    centrality = {u: 0 for u in nbunch}

    transposed = False
    if len(nbunch) < len(sources):
        transposed = True
        nbunch, sources = sources, nbunch
        if nx.is_directed(G):
            G = nx.reverse(G, copy=False)

    spl = partial(nx.shortest_path_length, G, weight=distance)
    for v in sources:
        dist = spl(v)
        for u in nbunch.intersection(dist):
            d = dist[u]
            if d == 0:  # handle u == v and edges with 0 weight
                continue
            centrality[v if transposed else u] += 1 / d

    return centrality
<|endoftext|>"""
Laplacian centrality measures.
"""

import networkx as nx

__all__ = ["laplacian_centrality"]


@nx._dispatchable(edge_attrs="weight")
def laplacian_centrality(
    G, normalized=True, nodelist=None, weight="weight", walk_type=None, alpha=0.95
):
    r"""Compute the Laplacian centrality for nodes in the graph `G`.

    The Laplacian Centrality of a node ``i`` is measured by the drop in the
    Laplacian Energy after deleting node ``i`` from the graph. The Laplacian Energy
    is the sum of the squared eigenvalues of a graph's Laplacian matrix.

    .. math::

        C_L(u_i,G) = \frac{(\Delta E)_i}{E_L (G)} = \frac{E_L (G)-E_L (G_i)}{E_L (G)}

        E_L (G) = \sum_{i=0}^n \lambda_i^2

    Where $E_L (G)$ is the Laplacian energy of graph `G`,
    E_L (G_i) is the Laplacian energy of graph `G` after deleting node ``i``
    and $\lambda_i$ are the eigenvalues of `G`'s Laplacian matrix.
    This formula shows the normalized value. Without normalization,
    the numerator on the right side is returned.

    Parameters
    ----------
    G : graph
        A networkx graph

    normalized : bool (default = True)
        If True the centrality score is scaled so the sum over all nodes is 1.
        If False the centrality score for each node is the drop in Laplacian
        energy when that node is removed.

    nodelist : list, optional (default = None)
        The rows and columns are ordered according to the nodes in nodelist.
        If nodelist is None, then the ordering is produced by G.nodes().

    weight: string or None, optional (default=`weight`)
        Optional parameter `weight` to compute the Laplacian matrix.
        The edge data key used to compute each value in the matrix.
        If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
        Optional parameter `walk_type` used when calling
        :func:`directed_laplacian_matrix <networkx.directed_laplacian_matrix>`.
        One of ``"random"``, ``"lazy"``, or ``"pagerank"``. If ``walk_type=None``
        (the default), then a value is selected according to the properties of `G`:
        - ``walk_type="random"`` if `G` is strongly connected and aperiodic
        - ``walk_type="lazy"`` if `G` is strongly connected but not aperiodic
        - ``walk_type="pagerank"`` for all other cases.

    alpha : real (default = 0.95)
        Optional parameter `alpha` used when calling
        :func:`directed_laplacian_matrix <networkx.directed_laplacian_matrix>`.
        (1 - alpha) is the teleportation probability used with pagerank.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with Laplacian centrality as the value.

    Examples
    --------
    >>> G = nx.Graph()
    >>> edges = [(0, 1, 4), (0, 2, 2), (2, 1, 1), (1, 3, 2), (1, 4, 2), (4, 5, 1)]
    >>> G.add_weighted_edges_from(edges)
    >>> sorted((v, f"{c:0.2f}") for v, c in laplacian_centrality(G).items())
    [(0, '0.70'), (1, '0.90'), (2, '0.28'), (3, '0.22'), (4, '0.26'), (5, '0.04')]

    Notes
    -----
    The algorithm is implemented based on [1]_ with an extension to directed graphs
    using the ``directed_laplacian_matrix`` function.

    Raises
    ------
    NetworkXPointlessConcept
        If the graph `G` is the null graph.
    ZeroDivisionError
        If the graph `G` has no edges (is empty) and normalization is requested.

    References
    ----------
    .. [1] Qi, X., Fuller, E., Wu, Q., Wu, Y., and Zhang, C.-Q. (2012).
       Laplacian centrality: A new centrality measure for weighted networks.
       Information Sciences, 194:240-253.
       https://math.wvu.edu/~cqzhang/Publication-files/my-paper/INS-2012-Laplacian-W.pdf

    See Also
    --------
    :func:`~networkx.linalg.laplacianmatrix.directed_laplacian_matrix`
    :func:`~networkx.linalg.laplacianmatrix.laplacian_matrix`
    """
    import numpy as np
    import scipy as sp

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("null graph has no centrality defined")
    if G.size(weight=weight) == 0:
        if normalized:
            raise ZeroDivisionError("graph with no edges has zero full energy")
        return {n: 0 for n in G}

    if nodelist is not None:
        nodeset = set(G.nbunch_iter(nodelist))
        if len(nodeset) != len(nodelist):
            raise nx.NetworkXError("nodelist has duplicate nodes or nodes not in G")
        nodes = nodelist + [n for n in G if n not in nodeset]
    else:
        nodelist = nodes = list(G)

    if G.is_directed():
        lap_matrix = nx.directed_laplacian_matrix(G, nodes, weight, walk_type, alpha)
    else:
        lap_matrix = nx.laplacian_matrix(G, nodes, weight).toarray()

    full_energy = np.power(sp.linalg.eigh(lap_matrix, eigvals_only=True), 2).sum()

    # calculate laplacian centrality
    laplace_centralities_dict = {}
    for i, node in enumerate(nodelist):
        # remove row and col i from lap_matrix
        all_but_i = list(np.arange(lap_matrix.shape[0]))
        all_but_i.remove(i)
        A_2 = lap_matrix[all_but_i, :][:, all_but_i]

        # Adjust diagonal for removed row
        new_diag = lap_matrix.diagonal() - abs(lap_matrix[:, i])
        np.fill_diagonal(A_2, new_diag[all_but_i])

        if len(all_but_i) > 0:  # catches degenerate case of single node
            new_energy = np.power(sp.linalg.eigh(A_2, eigvals_only=True), 2).sum()
        else:
            new_energy = 0.0

        lapl_cent = full_energy - new_energy
        if normalized:
            lapl_cent = lapl_cent / full_energy

        laplace_centralities_dict[node] = float(lapl_cent)

    return laplace_centralities_dict
<|endoftext|>"""Provides functions for computing minors of a graph."""

from itertools import chain, combinations, permutations, product

import networkx as nx
from networkx import density
from networkx.exception import NetworkXException
from networkx.utils import arbitrary_element

__all__ = [
    "contracted_edge",
    "contracted_nodes",
    "equivalence_classes",
    "identified_nodes",
    "quotient_graph",
]

chaini = chain.from_iterable


def equivalence_classes(iterable, relation):
    """Returns equivalence classes of `relation` when applied to `iterable`.

    The equivalence classes, or blocks, consist of objects from `iterable`
    which are all equivalent. They are defined to be equivalent if the
    `relation` function returns `True` when passed any two objects from that
    class, and `False` otherwise. To define an equivalence relation the
    function must be reflexive, symmetric and transitive.

    Parameters
    ----------
    iterable : list, tuple, or set
        An iterable of elements/nodes.

    relation : function
        A Boolean-valued function that implements an equivalence relation
        (reflexive, symmetric, transitive binary relation) on the elements
        of `iterable` - it must take two elements and return `True` if
        they are related, or `False` if not.

    Returns
    -------
    set of frozensets
        A set of frozensets representing the partition induced by the equivalence
        relation function `relation` on the elements of `iterable`. Each
        member set in the return set represents an equivalence class, or
        block, of the partition.

        Duplicate elements will be ignored so it makes the most sense for
        `iterable` to be a :class:`set`.

    Notes
    -----
    This function does not check that `relation` represents an equivalence
    relation. You can check that your equivalence classes provide a partition
    using `is_partition`.

    Examples
    --------
    Let `X` be the set of integers from `0` to `9`, and consider an equivalence
    relation `R` on `X` of congruence modulo `3`: this means that two integers
    `x` and `y` in `X` are equivalent under `R` if they leave the same
    remainder when divided by `3`, i.e. `(x - y) mod 3 = 0`.

    The equivalence classes of this relation are `{0, 3, 6, 9}`, `{1, 4, 7}`,
    `{2, 5, 8}`: `0`, `3`, `6`, `9` are all divisible by `3` and leave zero
    remainder; `1`, `4`, `7` leave remainder `1`; while `2`, `5` and `8` leave
    remainder `2`. We can see this by calling `equivalence_classes` with
    `X` and a function implementation of `R`.

    >>> X = set(range(10))
    >>> def mod3(x, y):
    ...     return (x - y) % 3 == 0
    >>> equivalence_classes(X, mod3)  # doctest: +SKIP
    {frozenset({1, 4, 7}), frozenset({8, 2, 5}), frozenset({0, 9, 3, 6})}
    """
    # For simplicity of implementation, we initialize the return value as a
    # list of lists, then convert it to a set of sets at the end of the
    # function.
    blocks = []
    # Determine the equivalence class for each element of the iterable.
    for y in iterable:
        # Each element y must be in *exactly one* equivalence class.
        #
        # Each block is guaranteed to be non-empty
        for block in blocks:
            x = arbitrary_element(block)
            if relation(x, y):
                block.append(y)
                break
        else:
            # If the element y is not part of any known equivalence class, it
            # must be in its own, so we create a new singleton equivalence
            # class for it.
            blocks.append([y])
    return {frozenset(block) for block in blocks}


@nx._dispatchable(edge_attrs="weight", returns_graph=True)
def quotient_graph(
    G,
    partition,
    edge_relation=None,
    node_data=None,
    edge_data=None,
    weight="weight",
    relabel=False,
    create_using=None,
):
    """Returns the quotient graph of `G` under the specified equivalence
    relation on nodes.

    Parameters
    ----------
    G : NetworkX graph
        The graph for which to return the quotient graph with the
        specified node relation.

    partition : function, or dict or list of lists, tuples or sets
        If a function, this function must represent an equivalence
        relation on the nodes of `G`. It must take two arguments *u*
        and *v* and return True exactly when *u* and *v* are in the
        same equivalence class. The equivalence classes form the nodes
        in the returned graph.

        If a dict of lists/tuples/sets, the keys can be any meaningful
        block labels, but the values must be the block lists/tuples/sets
        (one list/tuple/set per block), and the blocks must form a valid
        partition of the nodes of the graph. That is, each node must be
        in exactly one block of the partition.

        If a list of sets, the list must form a valid partition of
        the nodes of the graph. That is, each node must be in exactly
        one block of the partition.

    edge_relation : Boolean function with two arguments
        This function must represent an edge relation on the *blocks* of
        the `partition` of `G`. It must take two arguments, *B* and *C*,
        each one a set of nodes, and return True exactly when there should be
        an edge joining block *B* to block *C* in the returned graph.

        If `edge_relation` is not specified, it is assumed to be the
        following relation. Block *B* is related to block *C* if and
        only if some node in *B* is adjacent to some node in *C*,
        according to the edge set of `G`.

    node_data : function
        This function takes one argument, *B*, a set of nodes in `G`,
        and must return a dictionary representing the node data
        attributes to set on the node representing *B* in the quotient graph.
        If None, the following node attributes will be set:

        * 'graph', the subgraph of the graph `G` that this block
          represents,
        * 'nnodes', the number of nodes in this block,
        * 'nedges', the number of edges within this block,
        * 'density', the density of the subgraph of `G` that this
          block represents.

    edge_data : function
        This function takes two arguments, *B* and *C*, each one a set
        of nodes, and must return a dictionary representing the edge
        data attributes to set on the edge joining *B* and *C*, should
        there be an edge joining *B* and *C* in the quotient graph (if
        no such edge occurs in the quotient graph as determined by
        `edge_relation`, then the output of this function is ignored).

        If the quotient graph would be a multigraph, this function is
        not applied, since the edge data from each edge in the graph
        `G` appears in the edges of the quotient graph.

    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.

    relabel : bool
        If True, relabel the nodes of the quotient graph to be
        nonnegative integers. Otherwise, the nodes are identified with
        :class:`frozenset` instances representing the blocks given in
        `partition`.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        The quotient graph of `G` under the equivalence relation
        specified by `partition`. If the partition were given as a
        list of :class:`set` instances and `relabel` is False,
        each node will be a :class:`frozenset` corresponding to the same
        :class:`set`.

    Raises
    ------
    NetworkXException
        If the given partition is not a valid partition of the nodes of
        `G`.

    Examples
    --------
    The quotient graph of the complete bipartite graph under the "same
    neighbors" equivalence relation is `K_2`. Under this relation, two nodes
    are equivalent if they are not adjacent but have the same neighbor set.

    >>> G = nx.complete_bipartite_graph(2, 3)
    >>> same_neighbors = lambda u, v: (u not in G[v] and v not in G[u] and G[u] == G[v])
    >>> Q = nx.quotient_graph(G, same_neighbors)
    >>> K2 = nx.complete_graph(2)
    >>> nx.is_isomorphic(Q, K2)
    True

    The quotient graph of a directed graph under the "same strongly connected
    component" equivalence relation is the condensation of the graph (see
    :func:`condensation`). This example comes from the Wikipedia article
    *`Strongly connected component`_*.

    >>> G = nx.DiGraph()
    >>> edges = [
    ...     "ab",
    ...     "be",
    ...     "bf",
    ...     "bc",
    ...     "cg",
    ...     "cd",
    ...     "dc",
    ...     "dh",
    ...     "ea",
    ...     "ef",
    ...     "fg",
    ...     "gf",
    ...     "hd",
    ...     "hf",
    ... ]
    >>> G.add_edges_from(tuple(x) for x in edges)
    >>> components = list(nx.strongly_connected_components(G))
    >>> sorted(sorted(component) for component in components)
    [['a', 'b', 'e'], ['c', 'd', 'h'], ['f', 'g']]
    >>>
    >>> C = nx.condensation(G, components)
    >>> component_of = C.graph["mapping"]
    >>> same_component = lambda u, v: component_of[u] == component_of[v]
    >>> Q = nx.quotient_graph(G, same_component)
    >>> nx.is_isomorphic(C, Q)
    True

    Node identification can be represented as the quotient of a graph under the
    equivalence relation that places the two nodes in one block and each other
    node in its own singleton block.

    >>> K24 = nx.complete_bipartite_graph(2, 4)
    >>> K34 = nx.complete_bipartite_graph(3, 4)
    >>> C = nx.contracted_nodes(K34, 1, 2)
    >>> nodes = {1, 2}
    >>> is_contracted = lambda u, v: u in nodes and v in nodes
    >>> Q = nx.quotient_graph(K34, is_contracted)
    >>> nx.is_isomorphic(Q, C)
    True
    >>> nx.is_isomorphic(Q, K24)
    True

    The blockmodeling technique described in [1]_ can be implemented as a
    quotient graph.

    >>> G = nx.path_graph(6)
    >>> partition = [{0, 1}, {2, 3}, {4, 5}]
    >>> M = nx.quotient_graph(G, partition, relabel=True)
    >>> list(M.edges())
    [(0, 1), (1, 2)]

    Here is the sample example but using partition as a dict of block sets.

    >>> G = nx.path_graph(6)
    >>> partition = {0: {0, 1}, 2: {2, 3}, 4: {4, 5}}
    >>> M = nx.quotient_graph(G, partition, relabel=True)
    >>> list(M.edges())
    [(0, 1), (1, 2)]

    Partitions can be represented in various ways:

    0. a list/tuple/set of block lists/tuples/sets
    1. a dict with block labels as keys and blocks lists/tuples/sets as values
    2. a dict with block lists/tuples/sets as keys and block labels as values
    3. a function from nodes in the original iterable to block labels
    4. an equivalence relation function on the target iterable

    As `quotient_graph` is designed to accept partitions represented as (0), (1) or
    (4) only, the `equivalence_classes` function can be used to get the partitions
    in the right form, in order to call `quotient_graph`.

    .. _Strongly connected component: https://en.wikipedia.org/wiki/Strongly_connected_component

    References
    ----------
    .. [1] Patrick Doreian, Vladimir Batagelj, and Anuska Ferligoj.
           *Generalized Blockmodeling*.
           Cambridge University Press, 2004.

    """
    # If the user provided an equivalence relation as a function to compute
    # the blocks of the partition on the nodes of G induced by the
    # equivalence relation.
    if callable(partition):
        # equivalence_classes always return partition of whole G.
        partition = equivalence_classes(G, partition)
        if not nx.community.is_partition(G, partition):
            raise nx.NetworkXException(
                "Input `partition` is not an equivalence relation for nodes of G"
            )
        return _quotient_graph(
            G,
            partition,
            edge_relation,
            node_data,
            edge_data,
            weight,
            relabel,
            create_using,
        )

    # If the partition is a dict, it is assumed to be one where the keys are
    # user-defined block labels, and values are block lists, tuples or sets.
    if isinstance(partition, dict):
        partition = list(partition.values())

    # If the user provided partition as a collection of sets. Then we
    # need to check if partition covers all of G nodes. If the answer
    # is 'No' then we need to prepare suitable subgraph view.
    partition_nodes = set().union(*partition)
    if len(partition_nodes) != len(G):
        G = G.subgraph(partition_nodes)
    # Each node in the graph/subgraph must be in exactly one block.
    if not nx.community.is_partition(G, partition):
        raise NetworkXException("each node must be in exactly one part of `partition`")
    return _quotient_graph(
        G,
        partition,
        edge_relation,
        node_data,
        edge_data,
        weight,
        relabel,
        create_using,
    )


def _quotient_graph(
    G, partition, edge_relation, node_data, edge_data, weight, relabel, create_using
):
    """Construct the quotient graph assuming input has been checked"""
    if create_using is None:
        H = G.__class__()
    else:
        H = nx.empty_graph(0, create_using)
    # By default set some basic information about the subgraph that each block
    # represents on the nodes in the quotient graph.
    if node_data is None:

        def node_data(b):
            S = G.subgraph(b)
            return {
                "graph": S,
                "nnodes": len(S),
                "nedges": S.number_of_edges(),
                "density": density(S),
            }

    # Each block of the partition becomes a node in the quotient graph.
    partition = [frozenset(b) for b in partition]
    H.add_nodes_from((b, node_data(b)) for b in partition)
    # By default, the edge relation is the relation defined as follows. B is
    # adjacent to C if a node in B is adjacent to a node in C, according to the
    # edge set of G.
    #
    # This is not a particularly efficient implementation of this relation:
    # there are O(n^2) pairs to check and each check may require O(log n) time
    # (to check set membership). This can certainly be parallelized.
    if edge_relation is None:

        def edge_relation(b, c):
            return any(v in G[u] for u, v in product(b, c))

    # By default, sum the weights of the edges joining pairs of nodes across
    # blocks to get the weight of the edge joining those two blocks.
    if edge_data is None:

        def edge_data(b, c):
            edgedata = (
                d
                for u, v, d in G.edges(b | c, data=True)
                if (u in b and v in c) or (u in c and v in b)
            )
            return {"weight": sum(d.get(weight, 1) for d in edgedata)}

    block_pairs = permutations(H, 2) if H.is_directed() else combinations(H, 2)
    # In a multigraph, add one edge in the quotient graph for each edge
    # in the original graph.
    if H.is_multigraph():
        edges = chaini(
            (
                (b, c, G.get_edge_data(u, v, default={}))
                for u, v in product(b, c)
                if v in G[u]
            )
            for b, c in block_pairs
            if edge_relation(b, c)
        )
    # In a simple graph, apply the edge data function to each pair of
    # blocks to determine the edge data attributes to apply to each edge
    # in the quotient graph.
    else:
        edges = (
            (b, c, edge_data(b, c)) for (b, c) in block_pairs if edge_relation(b, c)
        )
    H.add_edges_from(edges)
    # If requested by the user, relabel the nodes to be integers,
    # numbered in increasing order from zero in the same order as the
    # iteration order of `partition`.
    if relabel:
        # Can't use nx.convert_node_labels_to_integers() here since we
        # want the order of iteration to be the same for backward
        # compatibility with the nx.blockmodel() function.
        labels = {b: i for i, b in enumerate(partition)}
        H = nx.relabel_nodes(H, labels)
    return H


@nx._dispatchable(
    preserve_all_attrs=True, mutates_input={"not copy": 4}, returns_graph=True
)
def contracted_nodes(G, u, v, self_loops=True, copy=True):
    """Returns the graph that results from contracting `u` and `v`.

    Node contraction identifies the two nodes as a single node incident to any
    edge that was incident to the original two nodes.

    Parameters
    ----------
    G : NetworkX graph
        The graph whose nodes will be contracted.

    u, v : nodes
        Must be nodes in `G`.

    self_loops : Boolean
        If this is True, any edges joining `u` and `v` in `G` become
        self-loops on the new node in the returned graph.

    copy : Boolean
        If this is True (default True), make a copy of
        `G` and return that instead of directly changing `G`.


    Returns
    -------
    Networkx graph
        If Copy is True,
        A new graph object of the same type as `G` (leaving `G` unmodified)
        with `u` and `v` identified in a single node. The right node `v`
        will be merged into the node `u`, so only `u` will appear in the
        returned graph.
        If copy is False,
        Modifies `G` with `u` and `v` identified in a single node.
        The right node `v` will be merged into the node `u`, so
        only `u` will appear in the returned graph.

    Notes
    -----
    For multigraphs, the edge keys for the realigned edges may
    not be the same as the edge keys for the old edges. This is
    natural because edge keys are unique only within each pair of nodes.

    For non-multigraphs where `u` and `v` are adjacent to a third node
    `w`, the edge (`v`, `w`) will be contracted into the edge (`u`,
    `w`) with its attributes stored into a "contraction" attribute.

    This function is also available as `identified_nodes`.

    Examples
    --------
    Contracting two nonadjacent nodes of the cycle graph on four nodes `C_4`
    yields the path graph (ignoring parallel edges):

    >>> G = nx.cycle_graph(4)
    >>> M = nx.contracted_nodes(G, 1, 3)
    >>> P3 = nx.path_graph(3)
    >>> nx.is_isomorphic(M, P3)
    True

    >>> G = nx.MultiGraph(P3)
    >>> M = nx.contracted_nodes(G, 0, 2)
    >>> M.edges
    MultiEdgeView([(0, 1, 0), (0, 1, 1)])

    >>> G = nx.Graph([(1, 2), (2, 2)])
    >>> H = nx.contracted_nodes(G, 1, 2, self_loops=False)
    >>> list(H.nodes())
    [1]
    >>> list(H.edges())
    [(1, 1)]

    In a ``MultiDiGraph`` with a self loop, the in and out edges will
    be treated separately as edges, so while contracting a node which
    has a self loop the contraction will add multiple edges:

    >>> G = nx.MultiDiGraph([(1, 2), (2, 2)])
    >>> H = nx.contracted_nodes(G, 1, 2)
    >>> list(H.edges())  # edge 1->2, 2->2, 2<-2 from the original Graph G
    [(1, 1), (1, 1), (1, 1)]
    >>> H = nx.contracted_nodes(G, 1, 2, self_loops=False)
    >>> list(H.edges())  # edge 2->2, 2<-2 from the original Graph G
    [(1, 1), (1, 1)]

    See Also
    --------
    contracted_edge
    quotient_graph

    """
    # Copying has significant overhead and can be disabled if needed
    if copy:
        H = G.copy()
    else:
        H = G

    # edge code uses G.edges(v) instead of G.adj[v] to handle multiedges
    if H.is_directed():
        edges_to_remap = chain(G.in_edges(v, data=True), G.out_edges(v, data=True))
    else:
        edges_to_remap = G.edges(v, data=True)

    # If the H=G, the generators change as H changes
    # This makes the edges_to_remap independent of H
    if not copy:
        edges_to_remap = list(edges_to_remap)

    v_data = H.nodes[v]
    H.remove_node(v)

    for prev_w, prev_x, d in edges_to_remap:
        w = prev_w if prev_w != v else u
        x = prev_x if prev_x != v else u

        if ({prev_w, prev_x} == {u, v}) and not self_loops:
            continue

        if not H.has_edge(w, x) or G.is_multigraph():
            H.add_edge(w, x, **d)
        else:
            if "contraction" in H.edges[(w, x)]:
                H.edges[(w, x)]["contraction"][(prev_w, prev_x)] = d
            else:
                H.edges[(w, x)]["contraction"] = {(prev_w, prev_x): d}

    if "contraction" in H.nodes[u]:
        H.nodes[u]["contraction"][v] = v_data
    else:
        H.nodes[u]["contraction"] = {v: v_data}
    return H


identified_nodes = contracted_nodes


@nx._dispatchable(
    preserve_edge_attrs=True, mutates_input={"not copy": 3}, returns_graph=True
)
def contracted_edge(G, edge, self_loops=True, copy=True):
    """Returns the graph that results from contracting the specified edge.

    Edge contraction identifies the two endpoints of the edge as a single node
    incident to any edge that was incident to the original two nodes. A graph
    that results from edge contraction is called a *minor* of the original
    graph.

    Parameters
    ----------
    G : NetworkX graph
       The graph whose edge will be contracted.

    edge : tuple
       Must be a pair of nodes in `G`.

    self_loops : Boolean
       If this is True, any edges (including `edge`) joining the
       endpoints of `edge` in `G` become self-loops on the new node in the
       returned graph.

    copy : Boolean (default True)
        If this is True, a the contraction will be performed on a copy of `G`,
        otherwise the contraction will happen in place.

    Returns
    -------
    Networkx graph
       A new graph object of the same type as `G` (leaving `G` unmodified)
       with endpoints of `edge` identified in a single node. The right node
       of `edge` will be merged into the left one, so only the left one will
       appear in the returned graph.

    Raises
    ------
    ValueError
       If `edge` is not an edge in `G`.

    Examples
    --------
    Attempting to contract two nonadjacent nodes yields an error:

    >>> G = nx.cycle_graph(4)
    >>> nx.contracted_edge(G, (1, 3))
    Traceback (most recent call last):
      ...
    ValueError: Edge (1, 3) does not exist in graph G; cannot contract it

    Contracting two adjacent nodes in the cycle graph on *n* nodes yields the
    cycle graph on *n - 1* nodes:

    >>> C5 = nx.cycle_graph(5)
    >>> C4 = nx.cycle_graph(4)
    >>> M = nx.contracted_edge(C5, (0, 1), self_loops=False)
    >>> nx.is_isomorphic(M, C4)
    True

    See also
    --------
    contracted_nodes
    quotient_graph

    """
    u, v = edge[:2]
    if not G.has_edge(u, v):
        raise ValueError(f"Edge {edge} does not exist in graph G; cannot contract it")
    return contracted_nodes(G, u, v, self_loops=self_loops, copy=copy)
<|endoftext|>"""
**************
Graph Matching
**************

Given a graph G = (V,E), a matching M in G is a set of pairwise non-adjacent
edges; that is, no two edges share a common vertex.

`Wikipedia: Matching <https://en.wikipedia.org/wiki/Matching_(graph_theory)>`_
"""

import networkx as nx

__all__ = ["min_maximal_matching"]


@nx._dispatchable
def min_maximal_matching(G):
    r"""Returns the minimum maximal matching of G. That is, out of all maximal
    matchings of the graph G, the smallest is returned.

    Parameters
    ----------
    G : NetworkX graph
      Undirected graph

    Returns
    -------
    min_maximal_matching : set
      Returns a set of edges such that no two edges share a common endpoint
      and every edge not in the set shares some common endpoint in the set.
      Cardinality will be 2*OPT in the worst case.

    Notes
    -----
    The algorithm computes an approximate solution for the minimum maximal
    cardinality matching problem. The solution is no more than 2 * OPT in size.
    Runtime is $O(|E|)$.

    References
    ----------
    .. [1] Vazirani, Vijay Approximation Algorithms (2001)
    """
    return nx.maximal_matching(G)
<|endoftext|>"""Functions for computing large cliques and maximum independent sets."""

import networkx as nx
from networkx.algorithms.approximation import ramsey
from networkx.utils import not_implemented_for

__all__ = [
    "clique_removal",
    "max_clique",
    "large_clique_size",
    "maximum_independent_set",
]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def maximum_independent_set(G):
    """Returns an approximate maximum independent set.

    Independent set or stable set is a set of vertices in a graph, no two of
    which are adjacent. That is, it is a set I of vertices such that for every
    two vertices in I, there is no edge connecting the two. Equivalently, each
    edge in the graph has at most one endpoint in I. The size of an independent
    set is the number of vertices it contains [1]_.

    A maximum independent set is a largest independent set for a given graph G
    and its size is denoted $\\alpha(G)$. The problem of finding such a set is called
    the maximum independent set problem and is an NP-hard optimization problem.
    As such, it is unlikely that there exists an efficient algorithm for finding
    a maximum independent set of a graph.

    The Independent Set algorithm is based on [2]_.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    iset : Set
        The apx-maximum independent set

    Examples
    --------
    >>> G = nx.path_graph(10)
    >>> nx.approximation.maximum_independent_set(G)
    {0, 2, 4, 6, 9}

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    Notes
    -----
    Finds the $O(|V|/(log|V|)^2)$ apx of independent set in the worst case.

    References
    ----------
    .. [1] `Wikipedia: Independent set
        <https://en.wikipedia.org/wiki/Independent_set_(graph_theory)>`_
    .. [2] Boppana, R., & Halldórsson, M. M. (1992).
       Approximating maximum independent sets by excluding subgraphs.
       BIT Numerical Mathematics, 32(2), 180–196. Springer.
    """
    iset, _ = clique_removal(G)
    return iset


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def max_clique(G):
    r"""Find the Maximum Clique

    Finds the $O(|V|/(log|V|)^2)$ apx of maximum clique/independent set
    in the worst case.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    clique : set
        The apx-maximum clique of the graph

    Examples
    --------
    >>> G = nx.path_graph(10)
    >>> nx.approximation.max_clique(G)
    {8, 9}

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    Notes
    -----
    A clique in an undirected graph G = (V, E) is a subset of the vertex set
    `C \subseteq V` such that for every two vertices in C there exists an edge
    connecting the two. This is equivalent to saying that the subgraph
    induced by C is complete (in some cases, the term clique may also refer
    to the subgraph).

    A maximum clique is a clique of the largest possible size in a given graph.
    The clique number `\omega(G)` of a graph G is the number of
    vertices in a maximum clique in G. The intersection number of
    G is the smallest number of cliques that together cover all edges of G.

    https://en.wikipedia.org/wiki/Maximum_clique

    References
    ----------
    .. [1] Boppana, R., & Halldórsson, M. M. (1992).
        Approximating maximum independent sets by excluding subgraphs.
        BIT Numerical Mathematics, 32(2), 180–196. Springer.
        doi:10.1007/BF01994876
    """
    # finding the maximum clique in a graph is equivalent to finding
    # the independent set in the complementary graph
    cgraph = nx.complement(G)
    iset, _ = clique_removal(cgraph)
    return iset


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def clique_removal(G):
    r"""Repeatedly remove cliques from the graph.

    Results in a $O(|V|/(\log |V|)^2)$ approximation of maximum clique
    and independent set. Returns the largest independent set found, along
    with found maximal cliques.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    max_ind_cliques : (set, list) tuple
        2-tuple of Maximal Independent Set and list of maximal cliques (sets).

    Examples
    --------
    >>> G = nx.path_graph(10)
    >>> nx.approximation.clique_removal(G)
    ({0, 2, 4, 6, 9}, [{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}])

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    References
    ----------
    .. [1] Boppana, R., & Halldórsson, M. M. (1992).
        Approximating maximum independent sets by excluding subgraphs.
        BIT Numerical Mathematics, 32(2), 180–196. Springer.
    """
    graph = G.copy()
    c_i, i_i = ramsey.ramsey_R2(graph)
    cliques = [c_i]
    isets = [i_i]
    while graph:
        graph.remove_nodes_from(c_i)
        c_i, i_i = ramsey.ramsey_R2(graph)
        if c_i:
            cliques.append(c_i)
        if i_i:
            isets.append(i_i)
    # Determine the largest independent set as measured by cardinality.
    maxiset = max(isets, key=len)
    return maxiset, cliques


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def large_clique_size(G):
    """Find the size of a large clique in a graph.

    A *clique* is a subset of nodes in which each pair of nodes is
    adjacent. This function is a heuristic for finding the size of a
    large clique in the graph.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    k: integer
       The size of a large clique in the graph.

    Examples
    --------
    >>> G = nx.path_graph(10)
    >>> nx.approximation.large_clique_size(G)
    2

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    Notes
    -----
    This implementation is from [1]_. Its worst case time complexity is
    :math:`O(n d^2)`, where *n* is the number of nodes in the graph and
    *d* is the maximum degree.

    This function is a heuristic, which means it may work well in
    practice, but there is no rigorous mathematical guarantee on the
    ratio between the returned number and the actual largest clique size
    in the graph.

    References
    ----------
    .. [1] Pattabiraman, Bharath, et al.
       "Fast Algorithms for the Maximum Clique Problem on Massive Graphs
       with Applications to Overlapping Community Detection."
       *Internet Mathematics* 11.4-5 (2015): 421--448.
       <https://doi.org/10.1080/15427951.2014.986778>

    See also
    --------

    :func:`networkx.algorithms.approximation.clique.max_clique`
        A function that returns an approximate maximum clique with a
        guarantee on the approximation ratio.

    :mod:`networkx.algorithms.clique`
        Functions for finding the exact maximum clique in a graph.

    """
    degrees = G.degree

    def _clique_heuristic(G, U, size, best_size):
        if not U:
            return max(best_size, size)
        u = max(U, key=degrees)
        U.remove(u)
        N_prime = {v for v in G[u] if degrees[v] >= best_size}
        return _clique_heuristic(G, U & N_prime, size + 1, best_size)

    best_size = 0
    nodes = (u for u in G if degrees[u] >= best_size)
    for u in nodes:
        neighbors = {v for v in G[u] if degrees[v] >= best_size}
        best_size = _clique_heuristic(G, neighbors, 1, best_size)
    return best_size
<|endoftext|>from itertools import chain

import networkx as nx
from networkx.utils import not_implemented_for, pairwise

__all__ = ["metric_closure", "steiner_tree"]


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight", returns_graph=True)
def metric_closure(G, weight="weight"):
    """Return the metric closure of a graph.

    The metric closure of a graph *G* is the complete graph in which each edge
    is weighted by the shortest path distance between the nodes in *G* .

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    NetworkX graph
        Metric closure of the graph `G`.

    """
    M = nx.Graph()

    Gnodes = set(G)

    # check for connected graph while processing first node
    all_paths_iter = nx.all_pairs_dijkstra(G, weight=weight)
    u, (distance, path) = next(all_paths_iter)
    if Gnodes - set(distance):
        msg = "G is not a connected graph. metric_closure is not defined."
        raise nx.NetworkXError(msg)
    Gnodes.remove(u)
    for v in Gnodes:
        M.add_edge(u, v, distance=distance[v], path=path[v])

    # first node done -- now process the rest
    for u, (distance, path) in all_paths_iter:
        Gnodes.remove(u)
        for v in Gnodes:
            M.add_edge(u, v, distance=distance[v], path=path[v])

    return M


def _mehlhorn_steiner_tree(G, terminal_nodes, weight):
    paths = nx.multi_source_dijkstra_path(G, terminal_nodes)

    d_1 = {}
    s = {}
    for v in G.nodes():
        s[v] = paths[v][0]
        d_1[(v, s[v])] = len(paths[v]) - 1

    # G1-G4 names match those from the Mehlhorn 1988 paper.
    G_1_prime = nx.Graph()
    for u, v, data in G.edges(data=True):
        su, sv = s[u], s[v]
        weight_here = d_1[(u, su)] + data.get(weight, 1) + d_1[(v, sv)]
        if not G_1_prime.has_edge(su, sv):
            G_1_prime.add_edge(su, sv, weight=weight_here)
        else:
            new_weight = min(weight_here, G_1_prime[su][sv]["weight"])
            G_1_prime.add_edge(su, sv, weight=new_weight)

    G_2 = nx.minimum_spanning_edges(G_1_prime, data=True)

    G_3 = nx.Graph()
    for u, v, d in G_2:
        path = nx.shortest_path(G, u, v, weight)
        for n1, n2 in pairwise(path):
            G_3.add_edge(n1, n2)

    G_3_mst = list(nx.minimum_spanning_edges(G_3, data=False))
    if G.is_multigraph():
        G_3_mst = (
            (u, v, min(G[u][v], key=lambda k: G[u][v][k][weight])) for u, v in G_3_mst
        )
    G_4 = G.edge_subgraph(G_3_mst).copy()
    _remove_nonterminal_leaves(G_4, terminal_nodes)
    return G_4.edges()


def _kou_steiner_tree(G, terminal_nodes, weight):
    # H is the subgraph induced by terminal_nodes in the metric closure M of G.
    M = metric_closure(G, weight=weight)
    H = M.subgraph(terminal_nodes)

    # Use the 'distance' attribute of each edge provided by M.
    mst_edges = nx.minimum_spanning_edges(H, weight="distance", data=True)

    # Create an iterator over each edge in each shortest path; repeats are okay
    mst_all_edges = chain.from_iterable(pairwise(d["path"]) for u, v, d in mst_edges)
    if G.is_multigraph():
        mst_all_edges = (
            (u, v, min(G[u][v], key=lambda k: G[u][v][k][weight]))
            for u, v in mst_all_edges
        )

    # Find the MST again, over this new set of edges
    G_S = G.edge_subgraph(mst_all_edges)
    T_S = nx.minimum_spanning_edges(G_S, weight="weight", data=False)

    # Leaf nodes that are not terminal might still remain; remove them here
    T_H = G.edge_subgraph(T_S).copy()
    _remove_nonterminal_leaves(T_H, terminal_nodes)

    return T_H.edges()


def _remove_nonterminal_leaves(G, terminals):
    terminal_set = set(terminals)
    leaves = {n for n in G if len(set(G[n]) - {n}) == 1}
    nonterminal_leaves = leaves - terminal_set

    while nonterminal_leaves:
        # Removing a node may create new non-terminal leaves, so we limit
        # search for candidate non-terminal nodes to neighbors of current
        # non-terminal nodes
        candidate_leaves = set.union(*(set(G[n]) for n in nonterminal_leaves))
        candidate_leaves -= nonterminal_leaves | terminal_set
        # Remove current set of non-terminal nodes
        G.remove_nodes_from(nonterminal_leaves)
        # Find any new non-terminal nodes from the set of candidates
        leaves = {n for n in candidate_leaves if len(set(G[n]) - {n}) == 1}
        nonterminal_leaves = leaves - terminal_set


ALGORITHMS = {
    "kou": _kou_steiner_tree,
    "mehlhorn": _mehlhorn_steiner_tree,
}


@not_implemented_for("directed")
@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def steiner_tree(G, terminal_nodes, weight="weight", method=None):
    r"""Return an approximation to the minimum Steiner tree of a graph.

    The minimum Steiner tree of `G` w.r.t a set of `terminal_nodes` (also *S*)
    is a tree within `G` that spans those nodes and has minimum size (sum of
    edge weights) among all such trees.

    The approximation algorithm is specified with the `method` keyword
    argument. All three available algorithms produce a tree whose weight is
    within a ``(2 - (2 / l))`` factor of the weight of the optimal Steiner tree,
    where ``l`` is the minimum number of leaf nodes across all possible Steiner
    trees.

    * ``"kou"`` [2]_ (runtime $O(|S| |V|^2)$) computes the minimum spanning tree of
      the subgraph of the metric closure of *G* induced by the terminal nodes,
      where the metric closure of *G* is the complete graph in which each edge is
      weighted by the shortest path distance between the nodes in *G*.

    * ``"mehlhorn"`` [3]_ (runtime $O(|E|+|V|\log|V|)$) modifies Kou et al.'s
      algorithm, beginning by finding the closest terminal node for each
      non-terminal. This data is used to create a complete graph containing only
      the terminal nodes, in which edge is weighted with the shortest path
      distance between them. The algorithm then proceeds in the same way as Kou
      et al..

    Parameters
    ----------
    G : NetworkX graph

    terminal_nodes : list
         A list of terminal nodes for which minimum steiner tree is
         to be found.

    weight : string (default = 'weight')
        Use the edge attribute specified by this string as the edge weight.
        Any edge attribute not present defaults to 1.

    method : string, optional (default = 'mehlhorn')
        The algorithm to use to approximate the Steiner tree.
        Supported options: 'kou', 'mehlhorn'.
        Other inputs produce a ValueError.

    Returns
    -------
    NetworkX graph
        Approximation to the minimum steiner tree of `G` induced by
        `terminal_nodes` .

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed.

    ValueError
        If the specified `method` is not supported.

    Notes
    -----
    For multigraphs, the edge between two nodes with minimum weight is the
    edge put into the Steiner tree.


    References
    ----------
    .. [1] Steiner_tree_problem on Wikipedia.
           https://en.wikipedia.org/wiki/Steiner_tree_problem
    .. [2] Kou, L., G. Markowsky, and L. Berman. 1981.
           ‘A Fast Algorithm for Steiner Trees’.
           Acta Informatica 15 (2): 141–45.
           https://doi.org/10.1007/BF00288961.
    .. [3] Mehlhorn, Kurt. 1988.
           ‘A Faster Approximation Algorithm for the Steiner Problem in Graphs’.
           Information Processing Letters 27 (3): 125–28.
           https://doi.org/10.1016/0020-0190(88)90066-X.
    """
    if method is None:
        method = "mehlhorn"

    try:
        algo = ALGORITHMS[method]
    except KeyError as e:
        raise ValueError(f"{method} is not a valid choice for an algorithm.") from e

    edges = algo(G, terminal_nodes, weight)
    # For multigraph we should add the minimal weight edge keys
    if G.is_multigraph():
        edges = (
            (u, v, min(G[u][v], key=lambda k: G[u][v][k][weight])) for u, v in edges
        )
    T = G.edge_subgraph(edges)
    return T
<|endoftext|>"""
Ramsey numbers.
"""

import networkx as nx
from networkx.utils import not_implemented_for

from ...utils import arbitrary_element

__all__ = ["ramsey_R2"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def ramsey_R2(G):
    r"""Compute the largest clique and largest independent set in `G`.

    This can be used to estimate bounds for the 2-color
    Ramsey number `R(2;s,t)` for `G`.

    This is a recursive implementation which could run into trouble
    for large recursions. Note that self-loop edges are ignored.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    max_pair : (set, set) tuple
        Maximum clique, Maximum independent set.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.
    """
    if not G:
        return set(), set()

    node = arbitrary_element(G)
    nbrs = (nbr for nbr in nx.all_neighbors(G, node) if nbr != node)
    nnbrs = nx.non_neighbors(G, node)
    c_1, i_1 = ramsey_R2(G.subgraph(nbrs).copy())
    c_2, i_2 = ramsey_R2(G.subgraph(nnbrs).copy())

    c_1.add(node)
    i_2.add(node)
    # Choose the larger of the two cliques and the larger of the two
    # independent sets, according to cardinality.
    return max(c_1, c_2, key=len), max(i_1, i_2, key=len)
<|endoftext|>"""Functions for computing treewidth decomposition.

Treewidth of an undirected graph is a number associated with the graph.
It can be defined as the size of the largest vertex set (bag) in a tree
decomposition of the graph minus one.

`Wikipedia: Treewidth <https://en.wikipedia.org/wiki/Treewidth>`_

The notions of treewidth and tree decomposition have gained their
attractiveness partly because many graph and network problems that are
intractable (e.g., NP-hard) on arbitrary graphs become efficiently
solvable (e.g., with a linear time algorithm) when the treewidth of the
input graphs is bounded by a constant [1]_ [2]_.

There are two different functions for computing a tree decomposition:
:func:`treewidth_min_degree` and :func:`treewidth_min_fill_in`.

.. [1] Hans L. Bodlaender and Arie M. C. A. Koster. 2010. "Treewidth
      computations I.Upper bounds". Inf. Comput. 208, 3 (March 2010),259-275.
      http://dx.doi.org/10.1016/j.ic.2009.03.008

.. [2] Hans L. Bodlaender. "Discovering Treewidth". Institute of Information
      and Computing Sciences, Utrecht University.
      Technical Report UU-CS-2005-018.
      http://www.cs.uu.nl

.. [3] K. Wang, Z. Lu, and J. Hicks *Treewidth*.
      https://web.archive.org/web/20210507025929/http://web.eecs.utk.edu/~cphill25/cs594_spring2015_projects/treewidth.pdf

"""

import itertools
import sys
from heapq import heapify, heappop, heappush

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["treewidth_min_degree", "treewidth_min_fill_in"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(returns_graph=True)
def treewidth_min_degree(G):
    """Returns a treewidth decomposition using the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree, i.e., first
    the node with the lowest degree is chosen, then the graph is updated
    and the corresponding node is removed. Next, a new node with the lowest
    degree is chosen, and so on.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
          2-tuple with treewidth and the corresponding decomposed tree.
    """
    deg_heuristic = MinDegreeHeuristic(G)
    return treewidth_decomp(G, lambda graph: deg_heuristic.best_node(graph))


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(returns_graph=True)
def treewidth_min_fill_in(G):
    """Returns a treewidth decomposition using the Minimum Fill-in heuristic.

    The heuristic chooses a node from the graph, where the number of edges
    added turning the neighborhood of the chosen node into clique is as
    small as possible.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
        2-tuple with treewidth and the corresponding decomposed tree.
    """
    return treewidth_decomp(G, min_fill_in_heuristic)


class MinDegreeHeuristic:
    """Implements the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree
    (number of neighbors), i.e., first the node with the lowest degree is
    chosen, then the graph is updated and the corresponding node is
    removed. Next, a new node with the lowest degree is chosen, and so on.
    """

    def __init__(self, graph):
        self._graph = graph

        # nodes that have to be updated in the heap before each iteration
        self._update_nodes = []

        self._degreeq = []  # a heapq with 3-tuples (degree,unique_id,node)
        self.count = itertools.count()

        # build heap with initial degrees
        for n in graph:
            self._degreeq.append((len(graph[n]), next(self.count), n))
        heapify(self._degreeq)

    def best_node(self, graph):
        # update nodes in self._update_nodes
        for n in self._update_nodes:
            # insert changed degrees into degreeq
            heappush(self._degreeq, (len(graph[n]), next(self.count), n))

        # get the next valid (minimum degree) node
        while self._degreeq:
            (min_degree, _, elim_node) = heappop(self._degreeq)
            if elim_node not in graph or len(graph[elim_node]) != min_degree:
                # outdated entry in degreeq
                continue
            elif min_degree == len(graph) - 1:
                # fully connected: abort condition
                return None

            # remember to update nodes in the heap before getting the next node
            self._update_nodes = graph[elim_node]
            return elim_node

        # the heap is empty: abort
        return None


def min_fill_in_heuristic(graph):
    """Implements the Minimum Degree heuristic.

    Returns the node from the graph, where the number of edges added when
    turning the neighborhood of the chosen node into clique is as small as
    possible. This algorithm chooses the nodes using the Minimum Fill-In
    heuristic. The running time of the algorithm is :math:`O(V^3)` and it uses
    additional constant memory."""

    if len(graph) == 0:
        return None

    min_fill_in_node = None

    min_fill_in = sys.maxsize

    # sort nodes by degree
    nodes_by_degree = sorted(graph, key=lambda x: len(graph[x]))
    min_degree = len(graph[nodes_by_degree[0]])

    # abort condition (handle complete graph)
    if min_degree == len(graph) - 1:
        return None

    for node in nodes_by_degree:
        num_fill_in = 0
        nbrs = graph[node]
        for nbr in nbrs:
            # count how many nodes in nbrs current nbr is not connected to
            # subtract 1 for the node itself
            num_fill_in += len(nbrs - graph[nbr]) - 1
            if num_fill_in >= 2 * min_fill_in:
                break

        num_fill_in /= 2  # divide by 2 because of double counting

        if num_fill_in < min_fill_in:  # update min-fill-in node
            if num_fill_in == 0:
                return node
            min_fill_in = num_fill_in
            min_fill_in_node = node

    return min_fill_in_node


@nx._dispatchable(returns_graph=True)
def treewidth_decomp(G, heuristic=min_fill_in_heuristic):
    """Returns a treewidth decomposition using the passed heuristic.

    Parameters
    ----------
    G : NetworkX graph
    heuristic : heuristic function

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
        2-tuple with treewidth and the corresponding decomposed tree.
    """

    # make dict-of-sets structure
    graph = {n: set(G[n]) - {n} for n in G}

    # stack containing nodes and neighbors in the order from the heuristic
    node_stack = []

    # get first node from heuristic
    elim_node = heuristic(graph)
    while elim_node is not None:
        # connect all neighbors with each other
        nbrs = graph[elim_node]
        for u, v in itertools.permutations(nbrs, 2):
            if v not in graph[u]:
                graph[u].add(v)

        # push node and its current neighbors on stack
        node_stack.append((elim_node, nbrs))

        # remove node from graph
        for u in graph[elim_node]:
            graph[u].remove(elim_node)

        del graph[elim_node]
        elim_node = heuristic(graph)

    # the abort condition is met; put all remaining nodes into one bag
    decomp = nx.Graph()
    first_bag = frozenset(graph.keys())
    decomp.add_node(first_bag)

    treewidth = len(first_bag) - 1

    while node_stack:
        # get node and its neighbors from the stack
        (curr_node, nbrs) = node_stack.pop()

        # find a bag all neighbors are in
        old_bag = None
        for bag in decomp.nodes:
            if nbrs <= bag:
                old_bag = bag
                break

        if old_bag is None:
            # no old_bag was found: just connect to the first_bag
            old_bag = first_bag

        # create new node for decomposition
        nbrs.add(curr_node)
        new_bag = frozenset(nbrs)

        # update treewidth
        treewidth = max(treewidth, len(new_bag) - 1)

        # add edge to decomposition (implicitly also adds the new node)
        decomp.add_edge(old_bag, new_bag)

    return treewidth, decomp
<|endoftext|>"""Fast approximation for node connectivity"""

import itertools
from operator import itemgetter

import networkx as nx

__all__ = [
    "local_node_connectivity",
    "node_connectivity",
    "all_pairs_node_connectivity",
]


@nx._dispatchable(name="approximate_local_node_connectivity")
def local_node_connectivity(G, source, target, cutoff=None):
    """Compute node connectivity between source and target.

    Pairwise or local node connectivity between two distinct and nonadjacent
    nodes is the minimum number of nodes that must be removed (minimum
    separating cutset) to disconnect them. By Menger's theorem, this is equal
    to the number of node independent paths (paths that share no nodes other
    than source and target). Which is what we compute in this function.

    This algorithm is a fast approximation that gives an strict lower
    bound on the actual number of node independent paths between two nodes [1]_.
    It works for both directed and undirected graphs.

    Parameters
    ----------

    G : NetworkX graph

    source : node
        Starting node for node connectivity

    target : node
        Ending node for node connectivity

    cutoff : integer
        Maximum node connectivity to consider. If None, the minimum degree
        of source or target is used as a cutoff. Default value None.

    Returns
    -------
    k: integer
       pairwise node connectivity

    Examples
    --------
    >>> # Platonic octahedral graph has node connectivity 4
    >>> # for each non adjacent node pair
    >>> from networkx.algorithms import approximation as approx
    >>> G = nx.octahedral_graph()
    >>> approx.local_node_connectivity(G, 0, 5)
    4

    Notes
    -----
    This algorithm [1]_ finds node independents paths between two nodes by
    computing their shortest path using BFS, marking the nodes of the path
    found as 'used' and then searching other shortest paths excluding the
    nodes marked as used until no more paths exist. It is not exact because
    a shortest path could use nodes that, if the path were longer, may belong
    to two different node independent paths. Thus it only guarantees an
    strict lower bound on node connectivity.

    Note that the authors propose a further refinement, losing accuracy and
    gaining speed, which is not implemented yet.

    See also
    --------
    all_pairs_node_connectivity
    node_connectivity

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf

    """
    if target == source:
        raise nx.NetworkXError("source and target have to be different nodes.")

    # Maximum possible node independent paths
    if G.is_directed():
        possible = min(G.out_degree(source), G.in_degree(target))
    else:
        possible = min(G.degree(source), G.degree(target))

    K = 0
    if not possible:
        return K

    if cutoff is None:
        cutoff = float("inf")

    exclude = set()
    for i in range(min(possible, cutoff)):
        try:
            path = _bidirectional_shortest_path(G, source, target, exclude)
            exclude.update(set(path))
            K += 1
        except nx.NetworkXNoPath:
            break

    return K


@nx._dispatchable(name="approximate_node_connectivity")
def node_connectivity(G, s=None, t=None):
    r"""Returns an approximation for node connectivity for a graph or digraph G.

    Node connectivity is equal to the minimum number of nodes that
    must be removed to disconnect G or render it trivial. By Menger's theorem,
    this is equal to the number of node independent paths (paths that
    share no nodes other than source and target).

    If source and target nodes are provided, this function returns the
    local node connectivity: the minimum number of nodes that must be
    removed to break all paths from source to target in G.

    This algorithm is based on a fast approximation that gives an strict lower
    bound on the actual number of node independent paths between two nodes [1]_.
    It works for both directed and undirected graphs.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    s : node
        Source node. Optional. Default value: None.

    t : node
        Target node. Optional. Default value: None.

    Returns
    -------
    K : integer
        Node connectivity of G, or local node connectivity if source
        and target are provided.

    Examples
    --------
    >>> # Platonic octahedral graph is 4-node-connected
    >>> from networkx.algorithms import approximation as approx
    >>> G = nx.octahedral_graph()
    >>> approx.node_connectivity(G)
    4

    Notes
    -----
    This algorithm [1]_ finds node independents paths between two nodes by
    computing their shortest path using BFS, marking the nodes of the path
    found as 'used' and then searching other shortest paths excluding the
    nodes marked as used until no more paths exist. It is not exact because
    a shortest path could use nodes that, if the path were longer, may belong
    to two different node independent paths. Thus it only guarantees an
    strict lower bound on node connectivity.

    See also
    --------
    all_pairs_node_connectivity
    local_node_connectivity

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf

    """
    if (s is not None and t is None) or (s is None and t is not None):
        raise nx.NetworkXError("Both source and target must be specified.")

    # Local node connectivity
    if s is not None and t is not None:
        if s not in G:
            raise nx.NetworkXError(f"node {s} not in graph")
        if t not in G:
            raise nx.NetworkXError(f"node {t} not in graph")
        return local_node_connectivity(G, s, t)

    # Global node connectivity
    if G.is_directed():
        connected_func = nx.is_weakly_connected
        iter_func = itertools.permutations

        def neighbors(v):
            return itertools.chain(G.predecessors(v), G.successors(v))

    else:
        connected_func = nx.is_connected
        iter_func = itertools.combinations
        neighbors = G.neighbors

    if not connected_func(G):
        return 0

    # Choose a node with minimum degree
    v, minimum_degree = min(G.degree(), key=itemgetter(1))
    # Node connectivity is bounded by minimum degree
    K = minimum_degree
    # compute local node connectivity with all non-neighbors nodes
    # and store the minimum
    for w in set(G) - set(neighbors(v)) - {v}:
        K = min(K, local_node_connectivity(G, v, w, cutoff=K))
    # Same for non adjacent pairs of neighbors of v
    for x, y in iter_func(neighbors(v), 2):
        if y not in G[x] and x != y:
            K = min(K, local_node_connectivity(G, x, y, cutoff=K))
    return K


@nx._dispatchable(name="approximate_all_pairs_node_connectivity")
def all_pairs_node_connectivity(G, nbunch=None, cutoff=None):
    """Compute node connectivity between all pairs of nodes.

    Pairwise or local node connectivity between two distinct and nonadjacent
    nodes is the minimum number of nodes that must be removed (minimum
    separating cutset) to disconnect them. By Menger's theorem, this is equal
    to the number of node independent paths (paths that share no nodes other
    than source and target). Which is what we compute in this function.

    This algorithm is a fast approximation that gives an strict lower
    bound on the actual number of node independent paths between two nodes [1]_.
    It works for both directed and undirected graphs.


    Parameters
    ----------
    G : NetworkX graph

    nbunch: container
        Container of nodes. If provided node connectivity will be computed
        only over pairs of nodes in nbunch.

    cutoff : integer
        Maximum node connectivity to consider. If None, the minimum degree
        of source or target is used as a cutoff in each pair of nodes.
        Default value None.

    Returns
    -------
    K : dictionary
        Dictionary, keyed by source and target, of pairwise node connectivity

    Examples
    --------
    A 3 node cycle with one extra node attached has connectivity 2 between all
    nodes in the cycle and connectivity 1 between the extra node and the rest:

    >>> G = nx.cycle_graph(3)
    >>> G.add_edge(2, 3)
    >>> import pprint  # for nice dictionary formatting
    >>> pprint.pprint(nx.all_pairs_node_connectivity(G))
    {0: {1: 2, 2: 2, 3: 1},
     1: {0: 2, 2: 2, 3: 1},
     2: {0: 2, 1: 2, 3: 1},
     3: {0: 1, 1: 1, 2: 1}}

    See Also
    --------
    local_node_connectivity
    node_connectivity

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf
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

    for u, v in iter_func(nbunch, 2):
        k = local_node_connectivity(G, u, v, cutoff=cutoff)
        all_pairs[u][v] = k
        if not directed:
            all_pairs[v][u] = k

    return all_pairs


def _bidirectional_shortest_path(G, source, target, exclude):
    """Returns shortest path between source and target ignoring nodes in the
    container 'exclude'.

    Parameters
    ----------

    G : NetworkX graph

    source : node
        Starting node for path

    target : node
        Ending node for path

    exclude: container
        Container for nodes to exclude from the search for shortest paths

    Returns
    -------
    path: list
        Shortest path between source and target ignoring nodes in 'exclude'

    Raises
    ------
    NetworkXNoPath
        If there is no path or if nodes are adjacent and have only one path
        between them

    Notes
    -----
    This function and its helper are originally from
    networkx.algorithms.shortest_paths.unweighted and are modified to
    accept the extra parameter 'exclude', which is a container for nodes
    already used in other paths that should be ignored.

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf

    """
    # call helper to do the real work
    results = _bidirectional_pred_succ(G, source, target, exclude)
    pred, succ, w = results

    # build path from pred+w+succ
    path = []
    # from source to w
    while w is not None:
        path.append(w)
        w = pred[w]
    path.reverse()
    # from w to target
    w = succ[path[-1]]
    while w is not None:
        path.append(w)
        w = succ[w]

    return path


def _bidirectional_pred_succ(G, source, target, exclude):
    # does BFS from both source and target and meets in the middle
    # excludes nodes in the container "exclude" from the search

    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.predecessors
        Gsucc = G.successors
    else:
        Gpred = G.neighbors
        Gsucc = G.neighbors

    # predecessor and successors in search
    pred = {source: None}
    succ = {target: None}

    # initialize fringes, start with forward
    forward_fringe = [source]
    reverse_fringe = [target]

    level = 0

    while forward_fringe and reverse_fringe:
        # Make sure that we iterate one step forward and one step backwards
        # thus source and target will only trigger "found path" when they are
        # adjacent and then they can be safely included in the container 'exclude'
        level += 1
        if level % 2 != 0:
            this_level = forward_fringe
            forward_fringe = []
            for v in this_level:
                for w in Gsucc(v):
                    if w in exclude:
                        continue
                    if w not in pred:
                        forward_fringe.append(w)
                        pred[w] = v
                    if w in succ:
                        return pred, succ, w  # found path
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in Gpred(v):
                    if w in exclude:
                        continue
                    if w not in succ:
                        succ[w] = v
                        reverse_fringe.append(w)
                    if w in pred:
                        return pred, succ, w  # found path

    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
<|endoftext|>"""
=================================
Travelling Salesman Problem (TSP)
=================================

Implementation of approximate algorithms
for solving and approximating the TSP problem.

Categories of algorithms which are implemented:

- Christofides (provides a 3/2-approximation of TSP)
- Greedy
- Simulated Annealing (SA)
- Threshold Accepting (TA)
- Asadpour Asymmetric Traveling Salesman Algorithm

The Travelling Salesman Problem tries to find, given the weight
(distance) between all points where a salesman has to visit, the
route so that:

- The total distance (cost) which the salesman travels is minimized.
- The salesman returns to the starting point.
- Note that for a complete graph, the salesman visits each point once.

The function `travelling_salesman_problem` allows for incomplete
graphs by finding all-pairs shortest paths, effectively converting
the problem to a complete graph problem. It calls one of the
approximate methods on that problem and then converts the result
back to the original graph using the previously found shortest paths.

TSP is an NP-hard problem in combinatorial optimization,
important in operations research and theoretical computer science.

http://en.wikipedia.org/wiki/Travelling_salesman_problem
"""

import math

import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.utils import not_implemented_for, pairwise, py_random_state

__all__ = [
    "traveling_salesman_problem",
    "christofides",
    "asadpour_atsp",
    "greedy_tsp",
    "simulated_annealing_tsp",
    "threshold_accepting_tsp",
]


def swap_two_nodes(soln, seed):
    """Swap two nodes in `soln` to give a neighbor solution.

    Parameters
    ----------
    soln : list of nodes
        Current cycle of nodes

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    list
        The solution after move is applied. (A neighbor solution.)

    Notes
    -----
        This function assumes that the incoming list `soln` is a cycle
        (that the first and last element are the same) and also that
        we don't want any move to change the first node in the list
        (and thus not the last node either).

        The input list is changed as well as returned. Make a copy if needed.

    See Also
    --------
        move_one_node
    """
    a, b = seed.sample(range(1, len(soln) - 1), k=2)
    soln[a], soln[b] = soln[b], soln[a]
    return soln


def move_one_node(soln, seed):
    """Move one node to another position to give a neighbor solution.

    The node to move and the position to move to are chosen randomly.
    The first and last nodes are left untouched as soln must be a cycle
    starting at that node.

    Parameters
    ----------
    soln : list of nodes
        Current cycle of nodes

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    list
        The solution after move is applied. (A neighbor solution.)

    Notes
    -----
        This function assumes that the incoming list `soln` is a cycle
        (that the first and last element are the same) and also that
        we don't want any move to change the first node in the list
        (and thus not the last node either).

        The input list is changed as well as returned. Make a copy if needed.

    See Also
    --------
        swap_two_nodes
    """
    a, b = seed.sample(range(1, len(soln) - 1), k=2)
    soln.insert(b, soln.pop(a))
    return soln


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def christofides(G, weight="weight", tree=None):
    """Approximate a solution of the traveling salesman problem

    Compute a 3/2-approximation of the traveling salesman problem
    in a complete undirected graph using Christofides [1]_ algorithm.

    Parameters
    ----------
    G : Graph
        `G` should be a complete weighted undirected graph.
        The distance between all pairs of nodes should be included.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    tree : NetworkX graph or None (default: None)
        A minimum spanning tree of G. Or, if None, the minimum spanning
        tree is computed using :func:`networkx.minimum_spanning_tree`

    Returns
    -------
    list
        List of nodes in `G` along a cycle with a 3/2-approximation of
        the minimal Hamiltonian cycle.

    References
    ----------
    .. [1] Christofides, Nicos. "Worst-case analysis of a new heuristic for
       the travelling salesman problem." No. RR-388. Carnegie-Mellon Univ
       Pittsburgh Pa Management Sciences Research Group, 1976.
    """
    # Remove selfloops if necessary
    loop_nodes = nx.nodes_with_selfloops(G)
    try:
        node = next(loop_nodes)
    except StopIteration:
        pass
    else:
        G = G.copy()
        G.remove_edge(node, node)
        G.remove_edges_from((n, n) for n in loop_nodes)
    # Check that G is a complete graph
    N = len(G) - 1
    # This check ignores selfloops which is what we want here.
    if any(len(nbrdict) != N for n, nbrdict in G.adj.items()):
        raise nx.NetworkXError("G must be a complete graph.")

    if tree is None:
        tree = nx.minimum_spanning_tree(G, weight=weight)
    L = G.copy()
    L.remove_nodes_from([v for v, degree in tree.degree if not (degree % 2)])
    MG = nx.MultiGraph()
    MG.add_edges_from(tree.edges)
    edges = nx.min_weight_matching(L, weight=weight)
    MG.add_edges_from(edges)
    return _shortcutting(nx.eulerian_circuit(MG))


def _shortcutting(circuit):
    """Remove duplicate nodes in the path"""
    nodes = []
    for u, v in circuit:
        if v in nodes:
            continue
        if not nodes:
            nodes.append(u)
        nodes.append(v)
    nodes.append(nodes[0])
    return nodes


@nx._dispatchable(edge_attrs="weight")
def traveling_salesman_problem(
    G, weight="weight", nodes=None, cycle=True, method=None, **kwargs
):
    """Find the shortest path in `G` connecting specified nodes

    This function allows approximate solution to the traveling salesman
    problem on networks that are not complete graphs and/or where the
    salesman does not need to visit all nodes.

    This function proceeds in two steps. First, it creates a complete
    graph using the all-pairs shortest_paths between nodes in `nodes`.
    Edge weights in the new graph are the lengths of the paths
    between each pair of nodes in the original graph.
    Second, an algorithm (default: `christofides` for undirected and
    `asadpour_atsp` for directed) is used to approximate the minimal Hamiltonian
    cycle on this new graph. The available algorithms are:

     - christofides
     - greedy_tsp
     - simulated_annealing_tsp
     - threshold_accepting_tsp
     - asadpour_atsp

    Once the Hamiltonian Cycle is found, this function post-processes to
    accommodate the structure of the original graph. If `cycle` is ``False``,
    the biggest weight edge is removed to make a Hamiltonian path.
    Then each edge on the new complete graph used for that analysis is
    replaced by the shortest_path between those nodes on the original graph.
    If the input graph `G` includes edges with weights that do not adhere to
    the triangle inequality, such as when `G` is not a complete graph (i.e
    length of non-existent edges is infinity), then the returned path may
    contain some repeating nodes (other than the starting node).

    Parameters
    ----------
    G : NetworkX graph
        A possibly weighted graph

    nodes : collection of nodes (default=G.nodes)
        collection (list, set, etc.) of nodes to visit

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    cycle : bool (default: True)
        Indicates whether a cycle should be returned, or a path.
        Note: the cycle is the approximate minimal cycle.
        The path simply removes the biggest edge in that cycle.

    method : function (default: None)
        A function that returns a cycle on all nodes and approximates
        the solution to the traveling salesman problem on a complete
        graph. The returned cycle is then used to find a corresponding
        solution on `G`. `method` should be callable; take inputs
        `G`, and `weight`; and return a list of nodes along the cycle.

        Provided options include :func:`christofides`, :func:`greedy_tsp`,
        :func:`simulated_annealing_tsp` and :func:`threshold_accepting_tsp`.

        If `method is None`: use :func:`christofides` for undirected `G` and
        :func:`asadpour_atsp` for directed `G`.

    **kwargs : dict
        Other keyword arguments to be passed to the `method` function passed in.

    Returns
    -------
    list
        List of nodes in `G` along a path with an approximation of the minimal
        path through `nodes`.

    Raises
    ------
    NetworkXError
        If `G` is a directed graph it has to be strongly connected or the
        complete version cannot be generated.

    Examples
    --------
    >>> tsp = nx.approximation.traveling_salesman_problem
    >>> G = nx.cycle_graph(9)
    >>> G[4][5]["weight"] = 5  # all other weights are 1
    >>> tsp(G, nodes=[3, 6])
    [3, 2, 1, 0, 8, 7, 6, 7, 8, 0, 1, 2, 3]
    >>> path = tsp(G, cycle=False)
    >>> path in ([4, 3, 2, 1, 0, 8, 7, 6, 5], [5, 6, 7, 8, 0, 1, 2, 3, 4])
    True

    While no longer required, you can still build (curry) your own function
    to provide parameter values to the methods.

    >>> SA_tsp = nx.approximation.simulated_annealing_tsp
    >>> method = lambda G, weight: SA_tsp(G, "greedy", weight=weight, temp=500)
    >>> path = tsp(G, cycle=False, method=method)
    >>> path in ([4, 3, 2, 1, 0, 8, 7, 6, 5], [5, 6, 7, 8, 0, 1, 2, 3, 4])
    True

    Otherwise, pass other keyword arguments directly into the tsp function.

    >>> path = tsp(
    ...     G,
    ...     cycle=False,
    ...     method=nx.approximation.simulated_annealing_tsp,
    ...     init_cycle="greedy",
    ...     temp=500,
    ... )
    >>> path in ([4, 3, 2, 1, 0, 8, 7, 6, 5], [5, 6, 7, 8, 0, 1, 2, 3, 4])
    True
    """
    if method is None:
        if G.is_directed():
            method = asadpour_atsp
        else:
            method = christofides
    if nodes is None:
        nodes = list(G.nodes)

    dist = {}
    path = {}
    for n, (d, p) in nx.all_pairs_dijkstra(G, weight=weight):
        dist[n] = d
        path[n] = p

    if G.is_directed():
        # If the graph is not strongly connected, raise an exception
        if not nx.is_strongly_connected(G):
            raise nx.NetworkXError("G is not strongly connected")
        GG = nx.DiGraph()
    else:
        GG = nx.Graph()
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            GG.add_edge(u, v, weight=dist[u][v])

    best_GG = method(GG, weight=weight, **kwargs)

    if not cycle:
        # find and remove the biggest edge
        (u, v) = max(pairwise(best_GG), key=lambda x: dist[x[0]][x[1]])
        pos = best_GG.index(u) + 1
        while best_GG[pos] != v:
            pos = best_GG[pos:].index(u) + 1
        best_GG = best_GG[pos:-1] + best_GG[:pos]

    best_path = []
    for u, v in pairwise(best_GG):
        best_path.extend(path[u][v][:-1])
    best_path.append(v)
    return best_path


@not_implemented_for("undirected")
@py_random_state(2)
@nx._dispatchable(edge_attrs="weight", mutates_input=True)
def asadpour_atsp(G, weight="weight", seed=None, source=None):
    """
    Returns an approximate solution to the traveling salesman problem.

    This approximate solution is one of the best known approximations for the
    asymmetric traveling salesman problem developed by Asadpour et al,
    [1]_. The algorithm first solves the Held-Karp relaxation to find a lower
    bound for the weight of the cycle. Next, it constructs an exponential
    distribution of undirected spanning trees where the probability of an
    edge being in the tree corresponds to the weight of that edge using a
    maximum entropy rounding scheme. Next we sample that distribution
    $2 \\lceil \\ln n \\rceil$ times and save the minimum sampled tree once the
    direction of the arcs is added back to the edges. Finally, we augment
    then short circuit that graph to find the approximate tour for the
    salesman.

    Parameters
    ----------
    G : nx.DiGraph
        The graph should be a complete weighted directed graph. The
        distance between all paris of nodes should be included and the triangle
        inequality should hold. That is, the direct edge between any two nodes
        should be the path of least cost.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    source : node label (default=`None`)
        If given, return the cycle starting and ending at the given node.

    Returns
    -------
    cycle : list of nodes
        Returns the cycle (list of nodes) that a salesman can follow to minimize
        the total weight of the trip.

    Raises
    ------
    NetworkXError
        If `G` is not complete or has less than two nodes, the algorithm raises
        an exception.

    NetworkXError
        If `source` is not `None` and is not a node in `G`, the algorithm raises
        an exception.

    NetworkXNotImplemented
        If `G` is an undirected graph.

    References
    ----------
    .. [1] A. Asadpour, M. X. Goemans, A. Madry, S. O. Gharan, and A. Saberi,
       An o(log n/log log n)-approximation algorithm for the asymmetric
       traveling salesman problem, Operations research, 65 (2017),
       pp. 1043–1061

    Examples
    --------
    >>> import networkx as nx
    >>> import networkx.algorithms.approximation as approx
    >>> G = nx.complete_graph(3, create_using=nx.DiGraph)
    >>> nx.set_edge_attributes(
    ...     G,
    ...     {(0, 1): 2, (1, 2): 2, (2, 0): 2, (0, 2): 1, (2, 1): 1, (1, 0): 1},
    ...     "weight",
    ... )
    >>> tour = approx.asadpour_atsp(G, source=0)
    >>> tour
    [0, 2, 1, 0]
    """
    from math import ceil, exp
    from math import log as ln

    # Check that G is a complete graph
    N = len(G) - 1
    if N < 2:
        raise nx.NetworkXError("G must have at least two nodes")
    # This check ignores selfloops which is what we want here.
    if any(len(nbrdict) - (n in nbrdict) != N for n, nbrdict in G.adj.items()):
        raise nx.NetworkXError("G is not a complete DiGraph")
    # Check that the source vertex, if given, is in the graph
    if source is not None and source not in G.nodes:
        raise nx.NetworkXError("Given source node not in G.")

    opt_hk, z_star = held_karp_ascent(G, weight)

    # Test to see if the ascent method found an integer solution or a fractional
    # solution. If it is integral then z_star is a nx.Graph, otherwise it is
    # a dict
    if not isinstance(z_star, dict):
        # Here we are using the shortcutting method to go from the list of edges
        # returned from eulerian_circuit to a list of nodes
        return _shortcutting(nx.eulerian_circuit(z_star, source=source))

    # Create the undirected support of z_star
    z_support = nx.MultiGraph()
    for u, v in z_star:
        if (u, v) not in z_support.edges:
            edge_weight = min(G[u][v][weight], G[v][u][weight])
            z_support.add_edge(u, v, **{weight: edge_weight})

    # Create the exponential distribution of spanning trees
    gamma = spanning_tree_distribution(z_support, z_star)

    # Write the lambda values to the edges of z_support
    z_support = nx.Graph(z_support)
    lambda_dict = {(u, v): exp(gamma[(u, v)]) for u, v in z_support.edges()}
    nx.set_edge_attributes(z_support, lambda_dict, "weight")
    del gamma, lambda_dict

    # Sample 2 * ceil( ln(n) ) spanning trees and record the minimum one
    minimum_sampled_tree = None
    minimum_sampled_tree_weight = math.inf
    for _ in range(2 * ceil(ln(G.number_of_nodes()))):
        sampled_tree = random_spanning_tree(z_support, "weight", seed=seed)
        sampled_tree_weight = sampled_tree.size(weight)
        if sampled_tree_weight < minimum_sampled_tree_weight:
            minimum_sampled_tree = sampled_tree.copy()
            minimum_sampled_tree_weight = sampled_tree_weight

    # Orient the edges in that tree to keep the cost of the tree the same.
    t_star = nx.MultiDiGraph()
    for u, v, d in minimum_sampled_tree.edges(data=weight):
        if d == G[u][v][weight]:
            t_star.add_edge(u, v, **{weight: d})
        else:
            t_star.add_edge(v, u, **{weight: d})

    # Find the node demands needed to neutralize the flow of t_star in G
    node_demands = {n: t_star.out_degree(n) - t_star.in_degree(n) for n in t_star}
    nx.set_node_attributes(G, node_demands, "demand")

    # Find the min_cost_flow
    flow_dict = nx.min_cost_flow(G, "demand")

    # Build the flow into t_star
    for source, values in flow_dict.items():
        for target in values:
            if (source, target) not in t_star.edges and values[target] > 0:
                # IF values[target] > 0 we have to add that many edges
                for _ in range(values[target]):
                    t_star.add_edge(source, target)

    # Return the shortcut eulerian circuit
    circuit = nx.eulerian_circuit(t_star, source=source)
    return _shortcutting(circuit)


@nx._dispatchable(edge_attrs="weight", mutates_input=True, returns_graph=True)
def held_karp_ascent(G, weight="weight"):
    """
    Minimizes the Held-Karp relaxation of the TSP for `G`

    Solves the Held-Karp relaxation of the input complete digraph and scales
    the output solution for use in the Asadpour [1]_ ASTP algorithm.

    The Held-Karp relaxation defines the lower bound for solutions to the
    ATSP, although it does return a fractional solution. This is used in the
    Asadpour algorithm as an initial solution which is later rounded to a
    integral tree within the spanning tree polytopes. This function solves
    the relaxation with the branch and bound method in [2]_.

    Parameters
    ----------
    G : nx.DiGraph
        The graph should be a complete weighted directed graph.
        The distance between all paris of nodes should be included.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    Returns
    -------
    OPT : float
        The cost for the optimal solution to the Held-Karp relaxation
    z : dict or nx.Graph
        A symmetrized and scaled version of the optimal solution to the
        Held-Karp relaxation for use in the Asadpour algorithm.

        If an integral solution is found, then that is an optimal solution for
        the ATSP problem and that is returned instead.

    References
    ----------
    .. [1] A. Asadpour, M. X. Goemans, A. Madry, S. O. Gharan, and A. Saberi,
       An o(log n/log log n)-approximation algorithm for the asymmetric
       traveling salesman problem, Operations research, 65 (2017),
       pp. 1043–1061

    .. [2] M. Held, R. M. Karp, The traveling-salesman problem and minimum
           spanning trees, Operations Research, 1970-11-01, Vol. 18 (6),
           pp.1138-1162
    """
    import numpy as np
    from scipy import optimize

    def k_pi():
        """
        Find the set of minimum 1-Arborescences for G at point pi.

        Returns
        -------
        Set
            The set of minimum 1-Arborescences
        """
        # Create a copy of G without vertex 1.
        G_1 = G.copy()
        minimum_1_arborescences = set()
        minimum_1_arborescence_weight = math.inf

        # node is node '1' in the Held and Karp paper
        n = next(G.__iter__())
        G_1.remove_node(n)

        # Iterate over the spanning arborescences of the graph until we know
        # that we have found the minimum 1-arborescences. My proposed strategy
        # is to find the most extensive root to connect to from 'node 1' and
        # the least expensive one. We then iterate over arborescences until
        # the cost of the basic arborescence is the cost of the minimum one
        # plus the difference between the most and least expensive roots,
        # that way the cost of connecting 'node 1' will by definition not by
        # minimum
        min_root = {"node": None, weight: math.inf}
        max_root = {"node": None, weight: -math.inf}
        for u, v, d in G.edges(n, data=True):
            if d[weight] < min_root[weight]:
                min_root = {"node": v, weight: d[weight]}
            if d[weight] > max_root[weight]:
                max_root = {"node": v, weight: d[weight]}

        min_in_edge = min(G.in_edges(n, data=True), key=lambda x: x[2][weight])
        min_root[weight] = min_root[weight] + min_in_edge[2][weight]
        max_root[weight] = max_root[weight] + min_in_edge[2][weight]

        min_arb_weight = math.inf
        for arb in nx.ArborescenceIterator(G_1):
            arb_weight = arb.size(weight)
            if min_arb_weight == math.inf:
                min_arb_weight = arb_weight
            elif arb_weight > min_arb_weight + max_root[weight] - min_root[weight]:
                break
            # We have to pick the root node of the arborescence for the out
            # edge of the first vertex as that is the only node without an
            # edge directed into it.
            for N, deg in arb.in_degree:
                if deg == 0:
                    # root found
                    arb.add_edge(n, N, **{weight: G[n][N][weight]})
                    arb_weight += G[n][N][weight]
                    break

            # We can pick the minimum weight in-edge for the vertex with
            # a cycle. If there are multiple edges with the same, minimum
            # weight, We need to add all of them.
            #
            # Delete the edge (N, v) so that we cannot pick it.
            edge_data = G[N][n]
            G.remove_edge(N, n)
            min_weight = min(G.in_edges(n, data=weight), key=lambda x: x[2])[2]
            min_edges = [
                (u, v, d) for u, v, d in G.in_edges(n, data=weight) if d == min_weight
            ]
            for u, v, d in min_edges:
                new_arb = arb.copy()
                new_arb.add_edge(u, v, **{weight: d})
                new_arb_weight = arb_weight + d
                # Check to see the weight of the arborescence, if it is a
                # new minimum, clear all of the old potential minimum
                # 1-arborescences and add this is the only one. If its
                # weight is above the known minimum, do not add it.
                if new_arb_weight < minimum_1_arborescence_weight:
                    minimum_1_arborescences.clear()
                    minimum_1_arborescence_weight = new_arb_weight
                # We have a 1-arborescence, add it to the set
                if new_arb_weight == minimum_1_arborescence_weight:
                    minimum_1_arborescences.add(new_arb)
            G.add_edge(N, n, **edge_data)

        return minimum_1_arborescences

    def direction_of_ascent():
        """
        Find the direction of ascent at point pi.

        See [1]_ for more information.

        Returns
        -------
        dict
            A mapping from the nodes of the graph which represents the direction
            of ascent.

        References
        ----------
        .. [1] M. Held, R. M. Karp, The traveling-salesman problem and minimum
           spanning trees, Operations Research, 1970-11-01, Vol. 18 (6),
           pp.1138-1162
        """
        # 1. Set d equal to the zero n-vector.
        d = {}
        for n in G:
            d[n] = 0
        del n
        # 2. Find a 1-Arborescence T^k such that k is in K(pi, d).
        minimum_1_arborescences = k_pi()
        while True:
            # Reduce K(pi) to K(pi, d)
            # Find the arborescence in K(pi) which increases the lest in
            # direction d
            min_k_d_weight = math.inf
            min_k_d = None
            for arborescence in minimum_1_arborescences:
                weighted_cost = 0
                for n, deg in arborescence.degree:
                    weighted_cost += d[n] * (deg - 2)
                if weighted_cost < min_k_d_weight:
                    min_k_d_weight = weighted_cost
                    min_k_d = arborescence

            # 3. If sum of d_i * v_{i, k} is greater than zero, terminate
            if min_k_d_weight > 0:
                return d, min_k_d
            # 4. d_i = d_i + v_{i, k}
            for n, deg in min_k_d.degree:
                d[n] += deg - 2
            # Check that we do not need to terminate because the direction
            # of ascent does not exist. This is done with linear
            # programming.
            c = np.full(len(minimum_1_arborescences), -1, dtype=int)
            a_eq = np.empty((len(G) + 1, len(minimum_1_arborescences)), dtype=int)
            b_eq = np.zeros(len(G) + 1, dtype=int)
            b_eq[len(G)] = 1
            for arb_count, arborescence in enumerate(minimum_1_arborescences):
                n_count = len(G) - 1
                for n, deg in arborescence.degree:
                    a_eq[n_count][arb_count] = deg - 2
                    n_count -= 1
                a_eq[len(G)][arb_count] = 1
            program_result = optimize.linprog(
                c, A_eq=a_eq, b_eq=b_eq, method="highs-ipm"
            )
            # If the constants exist, then the direction of ascent doesn't
            if program_result.success:
                # There is no direction of ascent
                return None, minimum_1_arborescences

            # 5. GO TO 2

    def find_epsilon(k, d):
        """
        Given the direction of ascent at pi, find the maximum distance we can go
        in that direction.

        Parameters
        ----------
        k_xy : set
            The set of 1-arborescences which have the minimum rate of increase
            in the direction of ascent

        d : dict
            The direction of ascent

        Returns
        -------
        float
            The distance we can travel in direction `d`
        """
        min_epsilon = math.inf
        for e_u, e_v, e_w in G.edges(data=weight):
            if (e_u, e_v) in k.edges:
                continue
            # Now, I have found a condition which MUST be true for the edges to
            # be a valid substitute. The edge in the graph which is the
            # substitute is the one with the same terminated end. This can be
            # checked rather simply.
            #
            # Find the edge within k which is the substitute. Because k is a
            # 1-arborescence, we know that they is only one such edges
            # leading into every vertex.
            if len(k.in_edges(e_v, data=weight)) > 1:
                raise Exception
            sub_u, sub_v, sub_w = next(k.in_edges(e_v, data=weight).__iter__())
            k.add_edge(e_u, e_v, **{weight: e_w})
            k.remove_edge(sub_u, sub_v)
            if (
                max(d for n, d in k.in_degree()) <= 1
                and len(G) == k.number_of_edges()
                and nx.is_weakly_connected(k)
            ):
                # Ascent method calculation
                if d[sub_u] == d[e_u] or sub_w == e_w:
                    # Revert to the original graph
                    k.remove_edge(e_u, e_v)
                    k.add_edge(sub_u, sub_v, **{weight: sub_w})
                    continue
                epsilon = (sub_w - e_w) / (d[e_u] - d[sub_u])
                if 0 < epsilon < min_epsilon:
                    min_epsilon = epsilon
            # Revert to the original graph
            k.remove_edge(e_u, e_v)
            k.add_edge(sub_u, sub_v, **{weight: sub_w})

        return min_epsilon

    # I have to know that the elements in pi correspond to the correct elements
    # in the direction of ascent, even if the node labels are not integers.
    # Thus, I will use dictionaries to made that mapping.
    pi_dict = {}
    for n in G:
        pi_dict[n] = 0
    del n
    original_edge_weights = {}
    for u, v, d in G.edges(data=True):
        original_edge_weights[(u, v)] = d[weight]
    dir_ascent, k_d = direction_of_ascent()
    while dir_ascent is not None:
        max_distance = find_epsilon(k_d, dir_ascent)
        for n, v in dir_ascent.items():
            pi_dict[n] += max_distance * v
        for u, v, d in G.edges(data=True):
            d[weight] = original_edge_weights[(u, v)] + pi_dict[u]
        dir_ascent, k_d = direction_of_ascent()
    nx._clear_cache(G)
    # k_d is no longer an individual 1-arborescence but rather a set of
    # minimal 1-arborescences at the maximum point of the polytope and should
    # be reflected as such
    k_max = k_d

    # Search for a cycle within k_max. If a cycle exists, return it as the
    # solution
    for k in k_max:
        if len([n for n in k if k.degree(n) == 2]) == G.order():
            # Tour found
            # TODO: this branch does not restore original_edge_weights of G!
            return k.size(weight), k

    # Write the original edge weights back to G and every member of k_max at
    # the maximum point. Also average the number of times that edge appears in
    # the set of minimal 1-arborescences.
    x_star = {}
    size_k_max = len(k_max)
    for u, v, d in G.edges(data=True):
        edge_count = 0
        d[weight] = original_edge_weights[(u, v)]
        for k in k_max:
            if (u, v) in k.edges():
                edge_count += 1
                k[u][v][weight] = original_edge_weights[(u, v)]
        x_star[(u, v)] = edge_count / size_k_max
    # Now symmetrize the edges in x_star and scale them according to (5) in
    # reference [1]
    z_star = {}
    scale_factor = (G.order() - 1) / G.order()
    for u, v in x_star:
        frequency = x_star[(u, v)] + x_star[(v, u)]
        if frequency > 0:
            z_star[(u, v)] = scale_factor * frequency
    del x_star
    # Return the optimal weight and the z dict
    return next(k_max.__iter__()).size(weight), z_star


@nx._dispatchable
def spanning_tree_distribution(G, z):
    """
    Find the asadpour exponential distribution of spanning trees.

    Solves the Maximum Entropy Convex Program in the Asadpour algorithm [1]_
    using the approach in section 7 to build an exponential distribution of
    undirected spanning trees.

    This algorithm ensures that the probability of any edge in a spanning
    tree is proportional to the sum of the probabilities of the tress
    containing that edge over the sum of the probabilities of all spanning
    trees of the graph.

    Parameters
    ----------
    G : nx.MultiGraph
        The undirected support graph for the Held Karp relaxation

    z : dict
        The output of `held_karp_ascent()`, a scaled version of the Held-Karp
        solution.

    Returns
    -------
    gamma : dict
        The probability distribution which approximately preserves the marginal
        probabilities of `z`.
    """
    from math import exp
    from math import log as ln

    def q(e):
        """
        The value of q(e) is described in the Asadpour paper is "the
        probability that edge e will be included in a spanning tree T that is
        chosen with probability proportional to exp(gamma(T))" which
        basically means that it is the total probability of the edge appearing
        across the whole distribution.

        Parameters
        ----------
        e : tuple
            The `(u, v)` tuple describing the edge we are interested in

        Returns
        -------
        float
            The probability that a spanning tree chosen according to the
            current values of gamma will include edge `e`.
        """
        # Create the laplacian matrices
        for u, v, d in G.edges(data=True):
            d[lambda_key] = exp(gamma[(u, v)])
        G_Kirchhoff = nx.total_spanning_tree_weight(G, lambda_key)
        G_e = nx.contracted_edge(G, e, self_loops=False)
        G_e_Kirchhoff = nx.total_spanning_tree_weight(G_e, lambda_key)

        # Multiply by the weight of the contracted edge since it is not included
        # in the total weight of the contracted graph.
        return exp(gamma[(e[0], e[1])]) * G_e_Kirchhoff / G_Kirchhoff

    # initialize gamma to the zero dict
    gamma = {}
    for u, v, _ in G.edges:
        gamma[(u, v)] = 0

    # set epsilon
    EPSILON = 0.2

    # pick an edge attribute name that is unlikely to be in the graph
    lambda_key = "spanning_tree_distribution's secret attribute name for lambda"

    while True:
        # We need to know that know that no values of q_e are greater than
        # (1 + epsilon) * z_e, however changing one gamma value can increase the
        # value of a different q_e, so we have to complete the for loop without
        # changing anything for the condition to be meet
        in_range_count = 0
        # Search for an edge with q_e > (1 + epsilon) * z_e
        for u, v in gamma:
            e = (u, v)
            q_e = q(e)
            z_e = z[e]
            if q_e > (1 + EPSILON) * z_e:
                delta = ln(
                    (q_e * (1 - (1 + EPSILON / 2) * z_e))
                    / ((1 - q_e) * (1 + EPSILON / 2) * z_e)
                )
                gamma[e] -= delta
                # Check that delta had the desired effect
                new_q_e = q(e)
                desired_q_e = (1 + EPSILON / 2) * z_e
                if round(new_q_e, 8) != round(desired_q_e, 8):
                    raise nx.NetworkXError(
                        f"Unable to modify probability for edge ({u}, {v})"
                    )
            else:
                in_range_count += 1
        # Check if the for loop terminated without changing any gamma
        if in_range_count == len(gamma):
            break

    # Remove the new edge attributes
    for _, _, d in G.edges(data=True):
        if lambda_key in d:
            del d[lambda_key]

    return gamma


@nx._dispatchable(edge_attrs="weight")
def greedy_tsp(G, weight="weight", source=None):
    """Return a low cost cycle starting at `source` and its cost.

    This approximates a solution to the traveling salesman problem.
    It finds a cycle of all the nodes that a salesman can visit in order
    to visit many nodes while minimizing total distance.
    It uses a simple greedy algorithm.
    In essence, this function returns a large cycle given a source point
    for which the total cost of the cycle is minimized.

    Parameters
    ----------
    G : Graph
        The Graph should be a complete weighted undirected graph.
        The distance between all pairs of nodes should be included.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    source : node, optional (default: first node in list(G))
        Starting node.  If None, defaults to ``next(iter(G))``

    Returns
    -------
    cycle : list of nodes
        Returns the cycle (list of nodes) that a salesman
        can follow to minimize total weight of the trip.

    Raises
    ------
    NetworkXError
        If `G` is not complete, the algorithm raises an exception.

    Examples
    --------
    >>> from networkx.algorithms import approximation as approx
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     {
    ...         ("A", "B", 3),
    ...         ("A", "C", 17),
    ...         ("A", "D", 14),
    ...         ("B", "A", 3),
    ...         ("B", "C", 12),
    ...         ("B", "D", 16),
    ...         ("C", "A", 13),
    ...         ("C", "B", 12),
    ...         ("C", "D", 4),
    ...         ("D", "A", 14),
    ...         ("D", "B", 15),
    ...         ("D", "C", 2),
    ...     }
    ... )
    >>> cycle = approx.greedy_tsp(G, source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31

    Notes
    -----
    This implementation of a greedy algorithm is based on the following:

    - The algorithm adds a node to the solution at every iteration.
    - The algorithm selects a node not already in the cycle whose connection
      to the previous node adds the least cost to the cycle.

    A greedy algorithm does not always give the best solution.
    However, it can construct a first feasible solution which can
    be passed as a parameter to an iterative improvement algorithm such
    as Simulated Annealing, or Threshold Accepting.

    Time complexity: It has a running time $O(|V|^2)$
    """
    # Check that G is a complete graph
    N = len(G) - 1
    # This check ignores selfloops which is what we want here.
    if any(len(nbrdict) - (n in nbrdict) != N for n, nbrdict in G.adj.items()):
        raise nx.NetworkXError("G must be a complete graph.")

    if source is None:
        source = nx.utils.arbitrary_element(G)

    if G.number_of_nodes() == 2:
        neighbor = next(G.neighbors(source))
        return [source, neighbor, source]

    nodeset = set(G)
    nodeset.remove(source)
    cycle = [source]
    next_node = source
    while nodeset:
        nbrdict = G[next_node]
        next_node = min(nodeset, key=lambda n: nbrdict[n].get(weight, 1))
        cycle.append(next_node)
        nodeset.remove(next_node)
    cycle.append(cycle[0])
    return cycle


@py_random_state(9)
@nx._dispatchable(edge_attrs="weight")
def simulated_annealing_tsp(
    G,
    init_cycle,
    weight="weight",
    source=None,
    temp=100,
    move="1-1",
    max_iterations=10,
    N_inner=100,
    alpha=0.01,
    seed=None,
):
    """Returns an approximate solution to the traveling salesman problem.

    This function uses simulated annealing to approximate the minimal cost
    cycle through the nodes. Starting from a suboptimal solution, simulated
    annealing perturbs that solution, occasionally accepting changes that make
    the solution worse to escape from a locally optimal solution. The chance
    of accepting such changes decreases over the iterations to encourage
    an optimal result.  In summary, the function returns a cycle starting
    at `source` for which the total cost is minimized. It also returns the cost.

    The chance of accepting a proposed change is related to a parameter called
    the temperature (annealing has a physical analogue of steel hardening
    as it cools). As the temperature is reduced, the chance of moves that
    increase cost goes down.

    Parameters
    ----------
    G : Graph
        `G` should be a complete weighted graph.
        The distance between all pairs of nodes should be included.

    init_cycle : list of all nodes or "greedy"
        The initial solution (a cycle through all nodes returning to the start).
        This argument has no default to make you think about it.
        If "greedy", use `greedy_tsp(G, weight)`.
        Other common starting cycles are `list(G) + [next(iter(G))]` or the final
        result of `simulated_annealing_tsp` when doing `threshold_accepting_tsp`.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    source : node, optional (default: first node in list(G))
        Starting node.  If None, defaults to ``next(iter(G))``

    temp : int, optional (default=100)
        The algorithm's temperature parameter. It represents the initial
        value of temperature

    move : "1-1" or "1-0" or function, optional (default="1-1")
        Indicator of what move to use when finding new trial solutions.
        Strings indicate two special built-in moves:

        - "1-1": 1-1 exchange which transposes the position
          of two elements of the current solution.
          The function called is :func:`swap_two_nodes`.
          For example if we apply 1-1 exchange in the solution
          ``A = [3, 2, 1, 4, 3]``
          we can get the following by the transposition of 1 and 4 elements:
          ``A' = [3, 2, 4, 1, 3]``
        - "1-0": 1-0 exchange which moves an node in the solution
          to a new position.
          The function called is :func:`move_one_node`.
          For example if we apply 1-0 exchange in the solution
          ``A = [3, 2, 1, 4, 3]``
          we can transfer the fourth element to the second position:
          ``A' = [3, 4, 2, 1, 3]``

        You may provide your own functions to enact a move from
        one solution to a neighbor solution. The function must take
        the solution as input along with a `seed` input to control
        random number generation (see the `seed` input here).
        Your function should maintain the solution as a cycle with
        equal first and last node and all others appearing once.
        Your function should return the new solution.

    max_iterations : int, optional (default=10)
        Declared done when this number of consecutive iterations of
        the outer loop occurs without any change in the best cost solution.

    N_inner : int, optional (default=100)
        The number of iterations of the inner loop.

    alpha : float between (0, 1), optional (default=0.01)
        Percentage of temperature decrease in each iteration
        of outer loop

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    cycle : list of nodes
        Returns the cycle (list of nodes) that a salesman
        can follow to minimize total weight of the trip.

    Raises
    ------
    NetworkXError
        If `G` is not complete the algorithm raises an exception.

    Examples
    --------
    >>> from networkx.algorithms import approximation as approx
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     {
    ...         ("A", "B", 3),
    ...         ("A", "C", 17),
    ...         ("A", "D", 14),
    ...         ("B", "A", 3),
    ...         ("B", "C", 12),
    ...         ("B", "D", 16),
    ...         ("C", "A", 13),
    ...         ("C", "B", 12),
    ...         ("C", "D", 4),
    ...         ("D", "A", 14),
    ...         ("D", "B", 15),
    ...         ("D", "C", 2),
    ...     }
    ... )
    >>> cycle = approx.simulated_annealing_tsp(G, "greedy", source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31
    >>> incycle = ["D", "B", "A", "C", "D"]
    >>> cycle = approx.simulated_annealing_tsp(G, incycle, source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31

    Notes
    -----
    Simulated Annealing is a metaheuristic local search algorithm.
    The main characteristic of this algorithm is that it accepts
    even solutions which lead to the increase of the cost in order
    to escape from low quality local optimal solutions.

    This algorithm needs an initial solution. If not provided, it is
    constructed by a simple greedy algorithm. At every iteration, the
    algorithm selects thoughtfully a neighbor solution.
    Consider $c(x)$ cost of current solution and $c(x')$ cost of a
    neighbor solution.
    If $c(x') - c(x) <= 0$ then the neighbor solution becomes the current
    solution for the next iteration. Otherwise, the algorithm accepts
    the neighbor solution with probability $p = exp - ([c(x') - c(x)] / temp)$.
    Otherwise the current solution is retained.

    `temp` is a parameter of the algorithm and represents temperature.

    Time complexity:
    For $N_i$ iterations of the inner loop and $N_o$ iterations of the
    outer loop, this algorithm has running time $O(N_i * N_o * |V|)$.

    For more information and how the algorithm is inspired see:
    http://en.wikipedia.org/wiki/Simulated_annealing
    """
    if move == "1-1":
        move = swap_two_nodes
    elif move == "1-0":
        move = move_one_node
    if init_cycle == "greedy":
        # Construct an initial solution using a greedy algorithm.
        cycle = greedy_tsp(G, weight=weight, source=source)
        if G.number_of_nodes() == 2:
            return cycle

    else:
        cycle = list(init_cycle)
        if source is None:
            source = cycle[0]
        elif source != cycle[0]:
            raise nx.NetworkXError("source must be first node in init_cycle")
        if cycle[0] != cycle[-1]:
            raise nx.NetworkXError("init_cycle must be a cycle. (return to start)")

        if len(cycle) - 1 != len(G) or len(set(G.nbunch_iter(cycle))) != len(G):
            raise nx.NetworkXError("init_cycle should be a cycle over all nodes in G.")

        # Check that G is a complete graph
        N = len(G) - 1
        # This check ignores selfloops which is what we want here.
        if any(len(nbrdict) - (n in nbrdict) != N for n, nbrdict in G.adj.items()):
            raise nx.NetworkXError("G must be a complete graph.")

        if G.number_of_nodes() == 2:
            neighbor = next(G.neighbors(source))
            return [source, neighbor, source]

    # Find the cost of initial solution
    cost = sum(G[u][v].get(weight, 1) for u, v in pairwise(cycle))

    count = 0
    best_cycle = cycle.copy()
    best_cost = cost
    while count <= max_iterations and temp > 0:
        count += 1
        for i in range(N_inner):
            adj_sol = move(cycle, seed)
            adj_cost = sum(G[u][v].get(weight, 1) for u, v in pairwise(adj_sol))
            delta = adj_cost - cost
            if delta <= 0:
                # Set current solution the adjacent solution.
                cycle = adj_sol
                cost = adj_cost

                if cost < best_cost:
                    count = 0
                    best_cycle = cycle.copy()
                    best_cost = cost
            else:
                # Accept even a worse solution with probability p.
                p = math.exp(-delta / temp)
                if p >= seed.random():
                    cycle = adj_sol
                    cost = adj_cost
        temp -= temp * alpha

    return best_cycle


@py_random_state(9)
@nx._dispatchable(edge_attrs="weight")
def threshold_accepting_tsp(
    G,
    init_cycle,
    weight="weight",
    source=None,
    threshold=1,
    move="1-1",
    max_iterations=10,
    N_inner=100,
    alpha=0.1,
    seed=None,
):
    """Returns an approximate solution to the traveling salesman problem.

    This function uses threshold accepting methods to approximate the minimal cost
    cycle through the nodes. Starting from a suboptimal solution, threshold
    accepting methods perturb that solution, accepting any changes that make
    the solution no worse than increasing by a threshold amount. Improvements
    in cost are accepted, but so are changes leading to small increases in cost.
    This allows the solution to leave suboptimal local minima in solution space.
    The threshold is decreased slowly as iterations proceed helping to ensure
    an optimum. In summary, the function returns a cycle starting at `source`
    for which the total cost is minimized.

    Parameters
    ----------
    G : Graph
        `G` should be a complete weighted graph.
        The distance between all pairs of nodes should be included.

    init_cycle : list or "greedy"
        The initial solution (a cycle through all nodes returning to the start).
        This argument has no default to make you think about it.
        If "greedy", use `greedy_tsp(G, weight)`.
        Other common starting cycles are `list(G) + [next(iter(G))]` or the final
        result of `simulated_annealing_tsp` when doing `threshold_accepting_tsp`.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    source : node, optional (default: first node in list(G))
        Starting node.  If None, defaults to ``next(iter(G))``

    threshold : int, optional (default=1)
        The algorithm's threshold parameter. It represents the initial
        threshold's value

    move : "1-1" or "1-0" or function, optional (default="1-1")
        Indicator of what move to use when finding new trial solutions.
        Strings indicate two special built-in moves:

        - "1-1": 1-1 exchange which transposes the position
          of two elements of the current solution.
          The function called is :func:`swap_two_nodes`.
          For example if we apply 1-1 exchange in the solution
          ``A = [3, 2, 1, 4, 3]``
          we can get the following by the transposition of 1 and 4 elements:
          ``A' = [3, 2, 4, 1, 3]``
        - "1-0": 1-0 exchange which moves an node in the solution
          to a new position.
          The function called is :func:`move_one_node`.
          For example if we apply 1-0 exchange in the solution
          ``A = [3, 2, 1, 4, 3]``
          we can transfer the fourth element to the second position:
          ``A' = [3, 4, 2, 1, 3]``

        You may provide your own functions to enact a move from
        one solution to a neighbor solution. The function must take
        the solution as input along with a `seed` input to control
        random number generation (see the `seed` input here).
        Your function should maintain the solution as a cycle with
        equal first and last node and all others appearing once.
        Your function should return the new solution.

    max_iterations : int, optional (default=10)
        Declared done when this number of consecutive iterations of
        the outer loop occurs without any change in the best cost solution.

    N_inner : int, optional (default=100)
        The number of iterations of the inner loop.

    alpha : float between (0, 1), optional (default=0.1)
        Percentage of threshold decrease when there is at
        least one acceptance of a neighbor solution.
        If no inner loop moves are accepted the threshold remains unchanged.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    cycle : list of nodes
        Returns the cycle (list of nodes) that a salesman
        can follow to minimize total weight of the trip.

    Raises
    ------
    NetworkXError
        If `G` is not complete the algorithm raises an exception.

    Examples
    --------
    >>> from networkx.algorithms import approximation as approx
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     {
    ...         ("A", "B", 3),
    ...         ("A", "C", 17),
    ...         ("A", "D", 14),
    ...         ("B", "A", 3),
    ...         ("B", "C", 12),
    ...         ("B", "D", 16),
    ...         ("C", "A", 13),
    ...         ("C", "B", 12),
    ...         ("C", "D", 4),
    ...         ("D", "A", 14),
    ...         ("D", "B", 15),
    ...         ("D", "C", 2),
    ...     }
    ... )
    >>> cycle = approx.threshold_accepting_tsp(G, "greedy", source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31
    >>> incycle = ["D", "B", "A", "C", "D"]
    >>> cycle = approx.threshold_accepting_tsp(G, incycle, source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31

    Notes
    -----
    Threshold Accepting is a metaheuristic local search algorithm.
    The main characteristic of this algorithm is that it accepts
    even solutions which lead to the increase of the cost in order
    to escape from low quality local optimal solutions.

    This algorithm needs an initial solution. This solution can be
    constructed by a simple greedy algorithm. At every iteration, it
    selects thoughtfully a neighbor solution.
    Consider $c(x)$ cost of current solution and $c(x')$ cost of
    neighbor solution.
    If $c(x') - c(x) <= threshold$ then the neighbor solution becomes the current
    solution for the next iteration, where the threshold is named threshold.

    In comparison to the Simulated Annealing algorithm, the Threshold
    Accepting algorithm does not accept very low quality solutions
    (due to the presence of the threshold value). In the case of
    Simulated Annealing, even a very low quality solution can
    be accepted with probability $p$.

    Time complexity:
    It has a running time $O(m * n * |V|)$ where $m$ and $n$ are the number
    of times the outer and inner loop run respectively.

    For more information and how algorithm is inspired see:
    https://doi.org/10.1016/0021-9991(90)90201-B

    See Also
    --------
    simulated_annealing_tsp

    """
    if move == "1-1":
        move = swap_two_nodes
    elif move == "1-0":
        move = move_one_node
    if init_cycle == "greedy":
        # Construct an initial solution using a greedy algorithm.
        cycle = greedy_tsp(G, weight=weight, source=source)
        if G.number_of_nodes() == 2:
            return cycle

    else:
        cycle = list(init_cycle)
        if source is None:
            source = cycle[0]
        elif source != cycle[0]:
            raise nx.NetworkXError("source must be first node in init_cycle")
        if cycle[0] != cycle[-1]:
            raise nx.NetworkXError("init_cycle must be a cycle. (return to start)")

        if len(cycle) - 1 != len(G) or len(set(G.nbunch_iter(cycle))) != len(G):
            raise nx.NetworkXError("init_cycle is not all and only nodes.")

        # Check that G is a complete graph
        N = len(G) - 1
        # This check ignores selfloops which is what we want here.
        if any(len(nbrdict) - (n in nbrdict) != N for n, nbrdict in G.adj.items()):
            raise nx.NetworkXError("G must be a complete graph.")

        if G.number_of_nodes() == 2:
            neighbor = list(G.neighbors(source))[0]
            return [source, neighbor, source]

    # Find the cost of initial solution
    cost = sum(G[u][v].get(weight, 1) for u, v in pairwise(cycle))

    count = 0
    best_cycle = cycle.copy()
    best_cost = cost
    while count <= max_iterations:
        count += 1
        accepted = False
        for i in range(N_inner):
            adj_sol = move(cycle, seed)
            adj_cost = sum(G[u][v].get(weight, 1) for u, v in pairwise(adj_sol))
            delta = adj_cost - cost
            if delta <= threshold:
                accepted = True

                # Set current solution the adjacent solution.
                cycle = adj_sol
                cost = adj_cost

                if cost < best_cost:
                    count = 0
                    best_cycle = cycle.copy()
                    best_cost = cost
        if accepted:
            threshold -= threshold * alpha

    return best_cycle
<|endoftext|>"""Functions for computing an approximate minimum weight vertex cover.

A |vertex cover|_ is a subset of nodes such that each edge in the graph
is incident to at least one node in the subset.

.. _vertex cover: https://en.wikipedia.org/wiki/Vertex_cover
.. |vertex cover| replace:: *vertex cover*

"""

import networkx as nx

__all__ = ["min_weighted_vertex_cover"]


@nx._dispatchable(node_attrs="weight")
def min_weighted_vertex_cover(G, weight=None):
    r"""Returns an approximate minimum weighted vertex cover.

    The set of nodes returned by this function is guaranteed to be a
    vertex cover, and the total weight of the set is guaranteed to be at
    most twice the total weight of the minimum weight vertex cover. In
    other words,

    .. math::

       w(S) \leq 2 * w(S^*),

    where $S$ is the vertex cover returned by this function,
    $S^*$ is the vertex cover of minimum weight out of all vertex
    covers of the graph, and $w$ is the function that computes the
    sum of the weights of each node in that given set.

    Parameters
    ----------
    G : NetworkX graph

    weight : string, optional (default = None)
        If None, every node has weight 1. If a string, use this node
        attribute as the node weight. A node without this attribute is
        assumed to have weight 1.

    Returns
    -------
    min_weighted_cover : set
        Returns a set of nodes whose weight sum is no more than twice
        the weight sum of the minimum weight vertex cover.

    Notes
    -----
    For a directed graph, a vertex cover has the same definition: a set
    of nodes such that each edge in the graph is incident to at least
    one node in the set. Whether the node is the head or tail of the
    directed edge is ignored.

    This is the local-ratio algorithm for computing an approximate
    vertex cover. The algorithm greedily reduces the costs over edges,
    iteratively building a cover. The worst-case runtime of this
    implementation is $O(m \log n)$, where $n$ is the number
    of nodes and $m$ the number of edges in the graph.

    References
    ----------
    .. [1] Bar-Yehuda, R., and Even, S. (1985). "A local-ratio theorem for
       approximating the weighted vertex cover problem."
       *Annals of Discrete Mathematics*, 25, 27–46
       <http://www.cs.technion.ac.il/~reuven/PDF/vc_lr.pdf>

    """
    cost = dict(G.nodes(data=weight, default=1))
    # While there are uncovered edges, choose an uncovered and update
    # the cost of the remaining edges.
    cover = set()
    for u, v in G.edges():
        if u in cover or v in cover:
            continue
        if cost[u] <= cost[v]:
            cover.add(u)
            cost[v] -= cost[u]
        else:
            cover.add(v)
            cost[u] -= cost[v]
    return cover
<|endoftext|>import networkx as nx
from networkx.utils.decorators import not_implemented_for, py_random_state

__all__ = ["randomized_partitioning", "one_exchange"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state(1)
@nx._dispatchable(edge_attrs="weight")
def randomized_partitioning(G, seed=None, p=0.5, weight=None):
    """Compute a random partitioning of the graph nodes and its cut value.

    A partitioning is calculated by observing each node
    and deciding to add it to the partition with probability `p`,
    returning a random cut and its corresponding value (the
    sum of weights of edges connecting different partitions).

    Parameters
    ----------
    G : NetworkX graph

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    p : scalar
        Probability for each node to be part of the first partition.
        Should be in [0,1]

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    cut_size : scalar
        Value of the minimum cut.

    partition : pair of node sets
        A partitioning of the nodes that defines a minimum cut.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> cut_size, partition = nx.approximation.randomized_partitioning(G, seed=1)
    >>> cut_size
    6
    >>> partition
    ({0, 3, 4}, {1, 2})

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.
    """
    cut = {node for node in G.nodes() if seed.random() < p}
    cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    partition = (cut, G.nodes - cut)
    return cut_size, partition


def _swap_node_partition(cut, node):
    return cut - {node} if node in cut else cut.union({node})


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state(2)
@nx._dispatchable(edge_attrs="weight")
def one_exchange(G, initial_cut=None, seed=None, weight=None):
    """Compute a partitioning of the graphs nodes and the corresponding cut value.

    Use a greedy one exchange strategy to find a locally maximal cut
    and its value, it works by finding the best node (one that gives
    the highest gain to the cut value) to add to the current cut
    and repeats this process until no improvement can be made.

    Parameters
    ----------
    G : networkx Graph
        Graph to find a maximum cut for.

    initial_cut : set
        Cut to use as a starting point. If not supplied the algorithm
        starts with an empty cut.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    cut_value : scalar
        Value of the maximum cut.

    partition : pair of node sets
        A partitioning of the nodes that defines a maximum cut.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> curr_cut_size, partition = nx.approximation.one_exchange(G, seed=1)
    >>> curr_cut_size
    6
    >>> partition
    ({0, 2}, {1, 3, 4})

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.
    """
    if initial_cut is None:
        initial_cut = set()
    cut = set(initial_cut)
    current_cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    while True:
        nodes = list(G.nodes())
        # Shuffling the nodes ensures random tie-breaks in the following call to max
        seed.shuffle(nodes)
        best_node_to_swap = max(
            nodes,
            key=lambda v: nx.algorithms.cut_size(
                G, _swap_node_partition(cut, v), weight=weight
            ),
            default=None,
        )
        potential_cut = _swap_node_partition(cut, best_node_to_swap)
        potential_cut_size = nx.algorithms.cut_size(G, potential_cut, weight=weight)

        if potential_cut_size > current_cut_size:
            cut = potential_cut
            current_cut_size = potential_cut_size
        else:
            break

    partition = (cut, G.nodes - cut)
    return current_cut_size, partition
<|endoftext|>import networkx as nx
from networkx.utils import not_implemented_for, py_random_state

__all__ = ["average_clustering"]


@not_implemented_for("directed")
@py_random_state(2)
@nx._dispatchable(name="approximate_average_clustering")
def average_clustering(G, trials=1000, seed=None):
    r"""Estimates the average clustering coefficient of G.

    The local clustering of each node in `G` is the fraction of triangles
    that actually exist over all possible triangles in its neighborhood.
    The average clustering coefficient of a graph `G` is the mean of
    local clusterings.

    This function finds an approximate average clustering coefficient
    for G by repeating `n` times (defined in `trials`) the following
    experiment: choose a node at random, choose two of its neighbors
    at random, and check if they are connected. The approximate
    coefficient is the fraction of triangles found over the number
    of trials [1]_.

    Parameters
    ----------
    G : NetworkX graph

    trials : integer
        Number of trials to perform (default 1000).

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    c : float
        Approximated average clustering coefficient.

    Examples
    --------
    >>> from networkx.algorithms import approximation
    >>> G = nx.erdos_renyi_graph(10, 0.2, seed=10)
    >>> approximation.average_clustering(G, trials=1000, seed=10)
    0.214

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    References
    ----------
    .. [1] Schank, Thomas, and Dorothea Wagner. Approximating clustering
       coefficient and transitivity. Universität Karlsruhe, Fakultät für
       Informatik, 2004.
       https://doi.org/10.5445/IR/1000001239

    """
    n = len(G)
    triangles = 0
    nodes = list(G)
    for i in [int(seed.random() * n) for i in range(trials)]:
        nbrs = list(G[nodes[i]])
        if len(nbrs) < 2:
            continue
        u, v = seed.sample(nbrs, 2)
        if u in G[v]:
            triangles += 1
    return triangles / trials
<|endoftext|>"""Approximations of graph properties and Heuristic methods for optimization.

The functions in this class are not imported into the top-level ``networkx``
namespace so the easiest way to use them is with::

    >>> from networkx.algorithms import approximation

Another option is to import the specific function with
``from networkx.algorithms.approximation import function_name``.

"""

from networkx.algorithms.approximation.clustering_coefficient import *
from networkx.algorithms.approximation.clique import *
from networkx.algorithms.approximation.connectivity import *
from networkx.algorithms.approximation.distance_measures import *
from networkx.algorithms.approximation.dominating_set import *
from networkx.algorithms.approximation.kcomponents import *
from networkx.algorithms.approximation.matching import *
from networkx.algorithms.approximation.ramsey import *
from networkx.algorithms.approximation.steinertree import *
from networkx.algorithms.approximation.traveling_salesman import *
from networkx.algorithms.approximation.treewidth import *
from networkx.algorithms.approximation.vertex_cover import *
from networkx.algorithms.approximation.maxcut import *
<|endoftext|>"""Functions for finding node and edge dominating sets.

A `dominating set`_ for an undirected graph *G* with vertex set *V*
and edge set *E* is a subset *D* of *V* such that every vertex not in
*D* is adjacent to at least one member of *D*. An `edge dominating set`_
is a subset *F* of *E* such that every edge not in *F* is
incident to an endpoint of at least one edge in *F*.

.. _dominating set: https://en.wikipedia.org/wiki/Dominating_set
.. _edge dominating set: https://en.wikipedia.org/wiki/Edge_dominating_set

"""

import networkx as nx

from ...utils import not_implemented_for
from ..matching import maximal_matching

__all__ = ["min_weighted_dominating_set", "min_edge_dominating_set"]


# TODO Why doesn't this algorithm work for directed graphs?
@not_implemented_for("directed")
@nx._dispatchable(node_attrs="weight")
def min_weighted_dominating_set(G, weight=None):
    r"""Returns a dominating set that approximates the minimum weight node
    dominating set.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph.

    weight : string
        The node attribute storing the weight of an node. If provided,
        the node attribute with this key must be a number for each
        node. If not provided, each node is assumed to have weight one.

    Returns
    -------
    min_weight_dominating_set : set
        A set of nodes, the sum of whose weights is no more than `(\log
        w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of
        each node in the graph and `w(V^*)` denotes the sum of the
        weights of each node in the minimum weight dominating set.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 4), (1, 4), (1, 2), (2, 3), (3, 4), (2, 5)])
    >>> nx.approximation.min_weighted_dominating_set(G)
    {1, 2, 4}

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Notes
    -----
    This algorithm computes an approximate minimum weighted dominating
    set for the graph `G`. The returned solution has weight `(\log
    w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of each
    node in the graph and `w(V^*)` denotes the sum of the weights of
    each node in the minimum weight dominating set for the graph.

    This implementation of the algorithm runs in $O(m)$ time, where $m$
    is the number of edges in the graph.

    References
    ----------
    .. [1] Vazirani, Vijay V.
           *Approximation Algorithms*.
           Springer Science & Business Media, 2001.

    """
    # The unique dominating set for the null graph is the empty set.
    if len(G) == 0:
        return set()

    # This is the dominating set that will eventually be returned.
    dom_set = set()

    def _cost(node_and_neighborhood):
        """Returns the cost-effectiveness of greedily choosing the given
        node.

        `node_and_neighborhood` is a two-tuple comprising a node and its
        closed neighborhood.

        """
        v, neighborhood = node_and_neighborhood
        return G.nodes[v].get(weight, 1) / len(neighborhood - dom_set)

    # This is a set of all vertices not already covered by the
    # dominating set.
    vertices = set(G)
    # This is a dictionary mapping each node to the closed neighborhood
    # of that node.
    neighborhoods = {v: {v} | set(G[v]) for v in G}

    # Continue until all vertices are adjacent to some node in the
    # dominating set.
    while vertices:
        # Find the most cost-effective node to add, along with its
        # closed neighborhood.
        dom_node, min_set = min(neighborhoods.items(), key=_cost)
        # Add the node to the dominating set and reduce the remaining
        # set of nodes to cover.
        dom_set.add(dom_node)
        del neighborhoods[dom_node]
        vertices -= min_set

    return dom_set


@nx._dispatchable
def min_edge_dominating_set(G):
    r"""Returns minimum cardinality edge dominating set.

    Parameters
    ----------
    G : NetworkX graph
      Undirected graph

    Returns
    -------
    min_edge_dominating_set : set
      Returns a set of dominating edges whose size is no more than 2 * OPT.

    Examples
    --------
    >>> G = nx.petersen_graph()
    >>> nx.approximation.min_edge_dominating_set(G)
    {(0, 1), (4, 9), (6, 8), (5, 7), (2, 3)}

    Raises
    ------
    ValueError
        If the input graph `G` is empty.

    Notes
    -----
    The algorithm computes an approximate solution to the edge dominating set
    problem. The result is no more than 2 * OPT in terms of size of the set.
    Runtime of the algorithm is $O(|E|)$.
    """
    if not G:
        raise ValueError("Expected non-empty NetworkX graph!")
    return maximal_matching(G)
<|endoftext|>"""Fast approximation for k-component structure"""

import itertools
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property

import networkx as nx
from networkx.algorithms.approximation import local_node_connectivity
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for

__all__ = ["k_components"]


@not_implemented_for("directed")
@nx._dispatchable(name="approximate_k_components")
def k_components(G, min_density=0.95):
    r"""Returns the approximate k-component structure of a graph G.

    A `k`-component is a maximal subgraph of a graph G that has, at least,
    node connectivity `k`: we need to remove at least `k` nodes to break it
    into more components. `k`-components have an inherent hierarchical
    structure because they are nested in terms of connectivity: a connected
    graph can contain several 2-components, each of which can contain
    one or more 3-components, and so forth.

    This implementation is based on the fast heuristics to approximate
    the `k`-component structure of a graph [1]_. Which, in turn, it is based on
    a fast approximation algorithm for finding good lower bounds of the number
    of node independent paths between two nodes [2]_.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    min_density : Float
        Density relaxation threshold. Default value 0.95

    Returns
    -------
