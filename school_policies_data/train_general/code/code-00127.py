<|endoftext|>r"""This module provides functions and operations for bipartite
graphs.  Bipartite graphs `B = (U, V, E)` have two node sets `U,V` and edges in
`E` that only connect nodes from opposite sets. It is common in the literature
to use an spatial analogy referring to the two node sets as top and bottom nodes.

The bipartite algorithms are not imported into the networkx namespace
at the top level so the easiest way to use them is with:

>>> from networkx.algorithms import bipartite

NetworkX does not have a custom bipartite graph class but the Graph()
or DiGraph() classes can be used to represent bipartite graphs. However,
you have to keep track of which set each node belongs to, and make
sure that there is no edge between nodes of the same set. The convention used
in NetworkX is to use a node attribute named `bipartite` with values 0 or 1 to
identify the sets each node belongs to. This convention is not enforced in
the source code of bipartite functions, it's only a recommendation.

For example:

>>> B = nx.Graph()
>>> # Add nodes with the node attribute "bipartite"
>>> B.add_nodes_from([1, 2, 3, 4], bipartite=0)
>>> B.add_nodes_from(["a", "b", "c"], bipartite=1)
>>> # Add edges only between nodes of opposite node sets
>>> B.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), (4, "a")])

Many algorithms of the bipartite module of NetworkX require, as an argument, a
container with all the nodes that belong to one set, in addition to the bipartite
graph `B`. The functions in the bipartite package do not check that the node set
is actually correct nor that the input graph is actually bipartite.
If `B` is connected, you can find the two node sets using a two-coloring
algorithm:

>>> nx.is_connected(B)
True
>>> bottom_nodes, top_nodes = bipartite.sets(B)

However, if the input graph is not connected, there are more than one possible
colorations. This is the reason why we require the user to pass a container
with all nodes of one bipartite node set as an argument to most bipartite
functions. In the face of ambiguity, we refuse the temptation to guess and
raise an :exc:`AmbiguousSolution <networkx.AmbiguousSolution>`
Exception if the input graph for
:func:`bipartite.sets <networkx.algorithms.bipartite.basic.sets>`
is disconnected.

Using the `bipartite` node attribute, you can easily get the two node sets:

>>> top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
>>> bottom_nodes = set(B) - top_nodes

So you can easily use the bipartite algorithms that require, as an argument, a
container with all nodes that belong to one node set:

>>> print(round(bipartite.density(B, bottom_nodes), 2))
0.5
>>> G = bipartite.projected_graph(B, top_nodes)

All bipartite graph generators in NetworkX build bipartite graphs with the
`bipartite` node attribute. Thus, you can use the same approach:

>>> RB = bipartite.random_graph(5, 7, 0.2)
>>> RB_top = {n for n, d in RB.nodes(data=True) if d["bipartite"] == 0}
>>> RB_bottom = set(RB) - RB_top
>>> list(RB_top)
[0, 1, 2, 3, 4]
>>> list(RB_bottom)
[5, 6, 7, 8, 9, 10, 11]

For other bipartite graph generators see
:mod:`Generators <networkx.algorithms.bipartite.generators>`.

"""

from networkx.algorithms.bipartite.basic import *
from networkx.algorithms.bipartite.centrality import *
from networkx.algorithms.bipartite.cluster import *
from networkx.algorithms.bipartite.covering import *
from networkx.algorithms.bipartite.edgelist import *
from networkx.algorithms.bipartite.matching import *
from networkx.algorithms.bipartite.matrix import *
from networkx.algorithms.bipartite.projection import *
from networkx.algorithms.bipartite.redundancy import *
from networkx.algorithms.bipartite.spectral import *
from networkx.algorithms.bipartite.generators import *
from networkx.algorithms.bipartite.extendability import *
<|endoftext|>"""Node redundancy for bipartite graphs."""

from itertools import combinations

import networkx as nx
from networkx import NetworkXError

__all__ = ["node_redundancy"]


@nx._dispatchable
def node_redundancy(G, nodes=None):
    r"""Computes the node redundancy coefficients for the nodes in the bipartite
    graph `G`.

    The redundancy coefficient of a node `v` is the fraction of pairs of
    neighbors of `v` that are both linked to other nodes. In a one-mode
    projection these nodes would be linked together even if `v` were
    not there.

    More formally, for any vertex `v`, the *redundancy coefficient of `v`* is
    defined by

    .. math::

        rc(v) = \frac{|\{\{u, w\} \subseteq N(v),
        \: \exists v' \neq  v,\: (v',u) \in E\:
        \mathrm{and}\: (v',w) \in E\}|}{ \frac{|N(v)|(|N(v)|-1)}{2}},

    where `N(v)` is the set of neighbors of `v` in `G`.

    Parameters
    ----------
    G : graph
        A bipartite graph

    nodes : list or iterable (optional)
        Compute redundancy for these nodes. The default is all nodes in G.

    Returns
    -------
    redundancy : dictionary
        A dictionary keyed by node with the node redundancy value.

    Examples
    --------
    Compute the redundancy coefficient of each node in a graph::

        >>> from networkx.algorithms import bipartite
        >>> G = nx.cycle_graph(4)
        >>> rc = bipartite.node_redundancy(G)
        >>> rc[0]
        1.0

    Compute the average redundancy for the graph::

        >>> from networkx.algorithms import bipartite
        >>> G = nx.cycle_graph(4)
        >>> rc = bipartite.node_redundancy(G)
        >>> sum(rc.values()) / len(G)
        1.0

    Compute the average redundancy for a set of nodes::

        >>> from networkx.algorithms import bipartite
        >>> G = nx.cycle_graph(4)
        >>> rc = bipartite.node_redundancy(G)
        >>> nodes = [0, 2]
        >>> sum(rc[n] for n in nodes) / len(nodes)
        1.0

    Raises
    ------
    NetworkXError
        If any of the nodes in the graph (or in `nodes`, if specified) has
        (out-)degree less than two (which would result in division by zero,
        according to the definition of the redundancy coefficient).

    References
    ----------
    .. [1] Latapy, Matthieu, Clémence Magnien, and Nathalie Del Vecchio (2008).
       Basic notions for the analysis of large two-mode networks.
       Social Networks 30(1), 31--48.

    """
    if nodes is None:
        nodes = G
    if any(len(G[v]) < 2 for v in nodes):
        raise NetworkXError(
            "Cannot compute redundancy coefficient for a node"
            " that has fewer than two neighbors."
        )
    # TODO This can be trivially parallelized.
    return {v: _node_redundancy(G, v) for v in nodes}


def _node_redundancy(G, v):
    """Returns the redundancy of the node `v` in the bipartite graph `G`.

    If `G` is a graph with `n` nodes, the redundancy of a node is the ratio
    of the "overlap" of `v` to the maximum possible overlap of `v`
    according to its degree. The overlap of `v` is the number of pairs of
    neighbors that have mutual neighbors themselves, other than `v`.

    `v` must have at least two neighbors in `G`.

    """
    n = len(G[v])
    overlap = sum(
        1 for (u, w) in combinations(G[v], 2) if (set(G[u]) & set(G[w])) - {v}
    )
    return (2 * overlap) / (n * (n - 1))
<|endoftext|>import networkx as nx

__all__ = ["degree_centrality", "betweenness_centrality", "closeness_centrality"]


@nx._dispatchable(name="bipartite_degree_centrality")
def degree_centrality(G, nodes):
    r"""Compute the degree centrality for nodes in a bipartite network.

    The degree centrality for a node `v` is the fraction of nodes
    connected to it.

    Parameters
    ----------
    G : graph
       A bipartite network

    nodes : list or container
      Container with all nodes in one bipartite node set.

    Returns
    -------
    centrality : dictionary
       Dictionary keyed by node with bipartite degree centrality as the value.

    Examples
    --------
    >>> G = nx.wheel_graph(5)
    >>> top_nodes = {0, 1, 2}
    >>> nx.bipartite.degree_centrality(G, nodes=top_nodes)
    {0: 2.0, 1: 1.5, 2: 1.5, 3: 1.0, 4: 1.0}

    See Also
    --------
    betweenness_centrality
    closeness_centrality
    :func:`~networkx.algorithms.bipartite.basic.sets`
    :func:`~networkx.algorithms.bipartite.basic.is_bipartite`

    Notes
    -----
    The nodes input parameter must contain all nodes in one bipartite node set,
    but the dictionary returned contains all nodes from both bipartite node
    sets. See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    For unipartite networks, the degree centrality values are
    normalized by dividing by the maximum possible degree (which is
    `n-1` where `n` is the number of nodes in G).

    In the bipartite case, the maximum possible degree of a node in a
    bipartite node set is the number of nodes in the opposite node set
    [1]_.  The degree centrality for a node `v` in the bipartite
    sets `U` with `n` nodes and `V` with `m` nodes is

    .. math::

        d_{v} = \frac{deg(v)}{m}, \mbox{for} v \in U ,

        d_{v} = \frac{deg(v)}{n}, \mbox{for} v \in V ,


    where `deg(v)` is the degree of node `v`.

    References
    ----------
    .. [1] Borgatti, S.P. and Halgin, D. In press. "Analyzing Affiliation
        Networks". In Carrington, P. and Scott, J. (eds) The Sage Handbook
        of Social Network Analysis. Sage Publications.
        https://dx.doi.org/10.4135/9781446294413.n28
    """
    top = set(nodes)
    bottom = set(G) - top
    s = 1.0 / len(bottom)
    centrality = {n: d * s for n, d in G.degree(top)}
    s = 1.0 / len(top)
    centrality.update({n: d * s for n, d in G.degree(bottom)})
    return centrality


@nx._dispatchable(name="bipartite_betweenness_centrality")
def betweenness_centrality(G, nodes):
    r"""Compute betweenness centrality for nodes in a bipartite network.

    Betweenness centrality of a node `v` is the sum of the
    fraction of all-pairs shortest paths that pass through `v`.

    Values of betweenness are normalized by the maximum possible
    value which for bipartite graphs is limited by the relative size
    of the two node sets [1]_.

    Let `n` be the number of nodes in the node set `U` and
    `m` be the number of nodes in the node set `V`, then
    nodes in `U` are normalized by dividing by

    .. math::

       \frac{1}{2} [m^2 (s + 1)^2 + m (s + 1)(2t - s - 1) - t (2s - t + 3)] ,

    where

    .. math::

        s = (n - 1) \div m , t = (n - 1) \mod m ,

    and nodes in `V` are normalized by dividing by

    .. math::

        \frac{1}{2} [n^2 (p + 1)^2 + n (p + 1)(2r - p - 1) - r (2p - r + 3)] ,

    where,

    .. math::

        p = (m - 1) \div n , r = (m - 1) \mod n .

    Parameters
    ----------
    G : graph
        A bipartite graph

    nodes : list or container
        Container with all nodes in one bipartite node set.

    Returns
    -------
    betweenness : dictionary
        Dictionary keyed by node with bipartite betweenness centrality
        as the value.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> top_nodes = {1, 2}
    >>> nx.bipartite.betweenness_centrality(G, nodes=top_nodes)
    {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}

    See Also
    --------
    degree_centrality
    closeness_centrality
    :func:`~networkx.algorithms.bipartite.basic.sets`
    :func:`~networkx.algorithms.bipartite.basic.is_bipartite`

    Notes
    -----
    The nodes input parameter must contain all nodes in one bipartite node set,
    but the dictionary returned contains all nodes from both node sets.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.


    References
    ----------
    .. [1] Borgatti, S.P. and Halgin, D. In press. "Analyzing Affiliation
        Networks". In Carrington, P. and Scott, J. (eds) The Sage Handbook
        of Social Network Analysis. Sage Publications.
        https://dx.doi.org/10.4135/9781446294413.n28
    """
    top = set(nodes)
    bottom = set(G) - top
    n = len(top)
    m = len(bottom)
    s, t = divmod(n - 1, m)
    bet_max_top = (
        ((m**2) * ((s + 1) ** 2))
        + (m * (s + 1) * (2 * t - s - 1))
        - (t * ((2 * s) - t + 3))
    ) / 2.0
    p, r = divmod(m - 1, n)
    bet_max_bot = (
        ((n**2) * ((p + 1) ** 2))
        + (n * (p + 1) * (2 * r - p - 1))
        - (r * ((2 * p) - r + 3))
    ) / 2.0
    betweenness = nx.betweenness_centrality(G, normalized=False, weight=None)
    for node in top:
        betweenness[node] /= bet_max_top
    for node in bottom:
        betweenness[node] /= bet_max_bot
    return betweenness


@nx._dispatchable(name="bipartite_closeness_centrality")
def closeness_centrality(G, nodes, normalized=True):
    r"""Compute the closeness centrality for nodes in a bipartite network.

    The closeness of a node is the distance to all other nodes in the
    graph or in the case that the graph is not connected to all other nodes
    in the connected component containing that node.

    Parameters
    ----------
    G : graph
        A bipartite network

    nodes : list or container
        Container with all nodes in one bipartite node set.

    normalized : bool, optional
      If True (default) normalize by connected component size.

    Returns
    -------
    closeness : dictionary
        Dictionary keyed by node with bipartite closeness centrality
        as the value.

    Examples
    --------
    >>> G = nx.wheel_graph(5)
    >>> top_nodes = {0, 1, 2}
    >>> nx.bipartite.closeness_centrality(G, nodes=top_nodes)
    {0: 1.5, 1: 1.2, 2: 1.2, 3: 1.0, 4: 1.0}

    See Also
    --------
    betweenness_centrality
    degree_centrality
    :func:`~networkx.algorithms.bipartite.basic.sets`
    :func:`~networkx.algorithms.bipartite.basic.is_bipartite`

    Notes
    -----
    The nodes input parameter must contain all nodes in one bipartite node set,
    but the dictionary returned contains all nodes from both node sets.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.


    Closeness centrality is normalized by the minimum distance possible.
    In the bipartite case the minimum distance for a node in one bipartite
    node set is 1 from all nodes in the other node set and 2 from all
    other nodes in its own set [1]_. Thus the closeness centrality
    for node `v`  in the two bipartite sets `U` with
    `n` nodes and `V` with `m` nodes is

    .. math::

        c_{v} = \frac{m + 2(n - 1)}{d}, \mbox{for} v \in U,

        c_{v} = \frac{n + 2(m - 1)}{d}, \mbox{for} v \in V,

    where `d` is the sum of the distances from `v` to all
    other nodes.

    Higher values of closeness  indicate higher centrality.

    As in the unipartite case, setting normalized=True causes the
    values to normalized further to n-1 / size(G)-1 where n is the
    number of nodes in the connected part of graph containing the
    node.  If the graph is not completely connected, this algorithm
    computes the closeness centrality for each connected part
    separately.

    References
    ----------
    .. [1] Borgatti, S.P. and Halgin, D. In press. "Analyzing Affiliation
        Networks". In Carrington, P. and Scott, J. (eds) The Sage Handbook
        of Social Network Analysis. Sage Publications.
        https://dx.doi.org/10.4135/9781446294413.n28
    """
    closeness = {}
    path_length = nx.single_source_shortest_path_length
    top = set(nodes)
    bottom = set(G) - top
    n = len(top)
    m = len(bottom)
    for node in top:
        sp = dict(path_length(G, node))
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1:
            closeness[node] = (m + 2 * (n - 1)) / totsp
            if normalized:
                s = (len(sp) - 1) / (len(G) - 1)
                closeness[node] *= s
        else:
            closeness[node] = 0.0
    for node in bottom:
        sp = dict(path_length(G, node))
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1:
            closeness[node] = (n + 2 * (m - 1)) / totsp
            if normalized:
                s = (len(sp) - 1) / (len(G) - 1)
                closeness[node] *= s
        else:
            closeness[node] = 0.0
    return closeness
<|endoftext|>"""
Generators and functions for bipartite graphs.
"""

import math
import numbers
from functools import reduce

import networkx as nx
from networkx.utils import nodes_or_number, py_random_state

__all__ = [
    "configuration_model",
    "havel_hakimi_graph",
    "reverse_havel_hakimi_graph",
    "alternating_havel_hakimi_graph",
    "preferential_attachment_graph",
    "random_graph",
    "gnmk_random_graph",
    "complete_bipartite_graph",
]


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number([0, 1])
def complete_bipartite_graph(n1, n2, create_using=None):
    """Returns the complete bipartite graph `K_{n_1,n_2}`.

    The graph is composed of two partitions with nodes 0 to (n1 - 1)
    in the first and nodes n1 to (n1 + n2 - 1) in the second.
    Each node in the first is connected to each node in the second.

    Parameters
    ----------
    n1, n2 : integer or iterable container of nodes
        If integers, nodes are from `range(n1)` and `range(n1, n1 + n2)`.
        If a container, the elements are the nodes.
    create_using : NetworkX graph instance, (default: nx.Graph)
       Return graph of this type.

    Notes
    -----
    Nodes are the integers 0 to `n1 + n2 - 1` unless either n1 or n2 are
    containers of nodes. If only one of n1 or n2 are integers, that
    integer is replaced by `range` of that integer.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.complete_bipartite_graph
    """
    G = nx.empty_graph(0, create_using)
    if G.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    n1, top = n1
    n2, bottom = n2
    if isinstance(n1, numbers.Integral) and isinstance(n2, numbers.Integral):
        bottom = [n1 + i for i in bottom]
    G.add_nodes_from(top, bipartite=0)
    G.add_nodes_from(bottom, bipartite=1)
    if len(G) != len(top) + len(bottom):
        raise nx.NetworkXError("Inputs n1 and n2 must contain distinct nodes")
    G.add_edges_from((u, v) for u in top for v in bottom)
    G.graph["name"] = f"complete_bipartite_graph({len(top)}, {len(bottom)})"
    return G


@py_random_state(3)
@nx._dispatchable(name="bipartite_configuration_model", graphs=None, returns_graph=True)
def configuration_model(aseq, bseq, create_using=None, seed=None):
    """Returns a random bipartite graph from two given degree sequences.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from set A are connected to nodes in set B by choosing
    randomly from the possible free stubs, one in A and one in B.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.configuration_model
    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    # length and sum of each sequence
    lena = len(aseq)
    lenb = len(bseq)
    suma = sum(aseq)
    sumb = sum(bseq)

    if not suma == sumb:
        raise nx.NetworkXError(
            f"invalid degree sequences, sum(aseq)!=sum(bseq),{suma},{sumb}"
        )

    G = _add_nodes_with_bipartite_label(G, lena, lenb)

    if len(aseq) == 0 or max(aseq) == 0:
        return G  # done if no edges

    # build lists of degree-repeated vertex numbers
    stubs = [[v] * aseq[v] for v in range(lena)]
    astubs = [x for subseq in stubs for x in subseq]

    stubs = [[v] * bseq[v - lena] for v in range(lena, lena + lenb)]
    bstubs = [x for subseq in stubs for x in subseq]

    # shuffle lists
    seed.shuffle(astubs)
    seed.shuffle(bstubs)

    G.add_edges_from([astubs[i], bstubs[i]] for i in range(suma))

    G.name = "bipartite_configuration_model"
    return G


@nx._dispatchable(name="bipartite_havel_hakimi_graph", graphs=None, returns_graph=True)
def havel_hakimi_graph(aseq, bseq, create_using=None):
    """Returns a bipartite graph from two given degree sequences using a
    Havel-Hakimi style construction.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from the set A are connected to nodes in the set B by
    connecting the highest degree nodes in set A to the highest degree
    nodes in set B until all stubs are connected.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.havel_hakimi_graph
    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    # length of the each sequence
    naseq = len(aseq)
    nbseq = len(bseq)

    suma = sum(aseq)
    sumb = sum(bseq)

    if not suma == sumb:
        raise nx.NetworkXError(
            f"invalid degree sequences, sum(aseq)!=sum(bseq),{suma},{sumb}"
        )

    G = _add_nodes_with_bipartite_label(G, naseq, nbseq)

    if len(aseq) == 0 or max(aseq) == 0:
        return G  # done if no edges

    # build list of degree-repeated vertex numbers
    astubs = [[aseq[v], v] for v in range(naseq)]
    bstubs = [[bseq[v - naseq], v] for v in range(naseq, naseq + nbseq)]
    astubs.sort()
    while astubs:
        (degree, u) = astubs.pop()  # take of largest degree node in the a set
        if degree == 0:
            break  # done, all are zero
        # connect the source to largest degree nodes in the b set
        bstubs.sort()
        for target in bstubs[-degree:]:
            v = target[1]
            G.add_edge(u, v)
            target[0] -= 1  # note this updates bstubs too.
            if target[0] == 0:
                bstubs.remove(target)

    G.name = "bipartite_havel_hakimi_graph"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def reverse_havel_hakimi_graph(aseq, bseq, create_using=None):
    """Returns a bipartite graph from two given degree sequences using a
    Havel-Hakimi style construction.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from set A are connected to nodes in the set B by connecting
    the highest degree nodes in set A to the lowest degree nodes in
    set B until all stubs are connected.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.reverse_havel_hakimi_graph
    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    # length of the each sequence
    lena = len(aseq)
    lenb = len(bseq)
    suma = sum(aseq)
    sumb = sum(bseq)

    if not suma == sumb:
        raise nx.NetworkXError(
            f"invalid degree sequences, sum(aseq)!=sum(bseq),{suma},{sumb}"
        )

    G = _add_nodes_with_bipartite_label(G, lena, lenb)

    if len(aseq) == 0 or max(aseq) == 0:
        return G  # done if no edges

    # build list of degree-repeated vertex numbers
    astubs = [[aseq[v], v] for v in range(lena)]
    bstubs = [[bseq[v - lena], v] for v in range(lena, lena + lenb)]
    astubs.sort()
    bstubs.sort()
    while astubs:
        (degree, u) = astubs.pop()  # take of largest degree node in the a set
        if degree == 0:
            break  # done, all are zero
        # connect the source to the smallest degree nodes in the b set
        for target in bstubs[0:degree]:
            v = target[1]
            G.add_edge(u, v)
            target[0] -= 1  # note this updates bstubs too.
            if target[0] == 0:
                bstubs.remove(target)

    G.name = "bipartite_reverse_havel_hakimi_graph"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def alternating_havel_hakimi_graph(aseq, bseq, create_using=None):
    """Returns a bipartite graph from two given degree sequences using
    an alternating Havel-Hakimi style construction.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from the set A are connected to nodes in the set B by
    connecting the highest degree nodes in set A to alternatively the
    highest and the lowest degree nodes in set B until all stubs are
    connected.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.alternating_havel_hakimi_graph
    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    # length of the each sequence
    naseq = len(aseq)
    nbseq = len(bseq)
    suma = sum(aseq)
    sumb = sum(bseq)

    if not suma == sumb:
        raise nx.NetworkXError(
            f"invalid degree sequences, sum(aseq)!=sum(bseq),{suma},{sumb}"
        )

    G = _add_nodes_with_bipartite_label(G, naseq, nbseq)

    if len(aseq) == 0 or max(aseq) == 0:
        return G  # done if no edges
    # build list of degree-repeated vertex numbers
    astubs = [[aseq[v], v] for v in range(naseq)]
    bstubs = [[bseq[v - naseq], v] for v in range(naseq, naseq + nbseq)]
    while astubs:
        astubs.sort()
        (degree, u) = astubs.pop()  # take of largest degree node in the a set
        if degree == 0:
            break  # done, all are zero
        bstubs.sort()
        small = bstubs[0 : degree // 2]  # add these low degree targets
        large = bstubs[(-degree + degree // 2) :]  # now high degree targets
        stubs = [x for z in zip(large, small) for x in z]  # combine, sorry
        if len(stubs) < len(small) + len(large):  # check for zip truncation
            stubs.append(large.pop())
        for target in stubs:
            v = target[1]
            G.add_edge(u, v)
            target[0] -= 1  # note this updates bstubs too.
            if target[0] == 0:
                bstubs.remove(target)

    G.name = "bipartite_alternating_havel_hakimi_graph"
    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def preferential_attachment_graph(aseq, p, create_using=None, seed=None):
    """Create a bipartite graph with a preferential attachment model from
    a given single degree sequence.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes starting with node len(aseq).
    The number of nodes in set B is random.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    p :  float
       Probability that a new bottom node is added.
    create_using : NetworkX graph instance, optional
       Return graph of this type.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    References
    ----------
    .. [1] Guillaume, J.L. and Latapy, M.,
       Bipartite graphs as models of complex networks.
       Physica A: Statistical Mechanics and its Applications,
       2006, 371(2), pp.795-813.
    .. [2] Jean-Loup Guillaume and Matthieu Latapy,
       Bipartite structure of all complex networks,
       Inf. Process. Lett. 90, 2004, pg. 215-221
       https://doi.org/10.1016/j.ipl.2004.03.007

    Notes
    -----
    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.preferential_attachment_graph
    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    if p > 1:
        raise nx.NetworkXError(f"probability {p} > 1")

    naseq = len(aseq)
    G = _add_nodes_with_bipartite_label(G, naseq, 0)
    vv = [[v] * aseq[v] for v in range(naseq)]
    while vv:
        while vv[0]:
            source = vv[0][0]
            vv[0].remove(source)
            if seed.random() < p or len(G) == naseq:
                target = len(G)
                G.add_node(target, bipartite=1)
                G.add_edge(source, target)
            else:
                bb = [[b] * G.degree(b) for b in range(naseq, len(G))]
                # flatten the list of lists into a list.
                bbstubs = reduce(lambda x, y: x + y, bb)
                # choose preferentially a bottom node.
                target = seed.choice(bbstubs)
                G.add_node(target, bipartite=1)
                G.add_edge(source, target)
        vv.remove(vv[0])
    G.name = "bipartite_preferential_attachment_model"
    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_graph(n, m, p, seed=None, directed=False):
    """Returns a bipartite random graph.

    This is a bipartite version of the binomial (Erdős-Rényi) graph.
    The graph is composed of two partitions. Set A has nodes 0 to
    (n - 1) and set B has nodes n to (n + m - 1).

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set.
    m : int
        The number of nodes in the second bipartite set.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True return a directed graph

    Notes
    -----
    The bipartite random graph algorithm chooses each of the n*m (undirected)
    or 2*nm (directed) possible edges with probability p.

    This algorithm is $O(n+m)$ where $m$ is the expected number of edges.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.random_graph

    See Also
    --------
    gnp_random_graph, configuration_model

    References
    ----------
    .. [1] Vladimir Batagelj and Ulrik Brandes,
       "Efficient generation of large random networks",
       Phys. Rev. E, 71, 036113, 2005.
    """
    G = nx.Graph()
    G = _add_nodes_with_bipartite_label(G, n, m)
    if directed:
        G = nx.DiGraph(G)
    G.name = f"fast_gnp_random_graph({n},{m},{p})"

    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_bipartite_graph(n, m)

    lp = math.log(1.0 - p)

    v = 0
    w = -1
    while v < n:
        lr = math.log(1.0 - seed.random())
        w = w + 1 + int(lr / lp)
        while w >= m and v < n:
            w = w - m
            v = v + 1
        if v < n:
            G.add_edge(v, n + w)

    if directed:
        # use the same algorithm to
        # add edges from the "m" to "n" set
        v = 0
        w = -1
        while v < n:
            lr = math.log(1.0 - seed.random())
            w = w + 1 + int(lr / lp)
            while w >= m and v < n:
                w = w - m
                v = v + 1
            if v < n:
                G.add_edge(n + w, v)

    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def gnmk_random_graph(n, m, k, seed=None, directed=False):
    """Returns a random bipartite graph G_{n,m,k}.

    Produces a bipartite graph chosen randomly out of the set of all graphs
    with n top nodes, m bottom nodes, and k edges.
    The graph is composed of two sets of nodes.
    Set A has nodes 0 to (n - 1) and set B has nodes n to (n + m - 1).

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set.
    m : int
        The number of nodes in the second bipartite set.
    k : int
        The number of edges
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True return a directed graph

    Examples
    --------
    from nx.algorithms import bipartite
    G = bipartite.gnmk_random_graph(10,20,50)

    See Also
    --------
    gnm_random_graph

    Notes
    -----
    If k > m * n then a complete bipartite graph is returned.

    This graph is a bipartite version of the `G_{nm}` random graph model.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.gnmk_random_graph
    """
    G = nx.Graph()
    G = _add_nodes_with_bipartite_label(G, n, m)
    if directed:
        G = nx.DiGraph(G)
    G.name = f"bipartite_gnm_random_graph({n},{m},{k})"
    if n == 1 or m == 1:
        return G
    max_edges = n * m  # max_edges for bipartite networks
    if k >= max_edges:  # Maybe we should raise an exception here
        return nx.complete_bipartite_graph(n, m, create_using=G)

    top = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    bottom = list(set(G) - set(top))
    edge_count = 0
    while edge_count < k:
        # generate random edge,u,v
        u = seed.choice(top)
        v = seed.choice(bottom)
        if v in G[u]:
            continue
        else:
            G.add_edge(u, v)
            edge_count += 1
    return G


def _add_nodes_with_bipartite_label(G, lena, lenb):
    G.add_nodes_from(range(lena + lenb))
    b = dict(zip(range(lena), [0] * lena))
    b.update(dict(zip(range(lena, lena + lenb), [1] * lenb)))
    nx.set_node_attributes(G, b, "bipartite")
    return G
<|endoftext|>"""One-mode (unipartite) projections of bipartite graphs."""

import networkx as nx
from networkx.exception import NetworkXAlgorithmError
from networkx.utils import not_implemented_for

__all__ = [
    "projected_graph",
    "weighted_projected_graph",
    "collaboration_weighted_projected_graph",
    "overlap_weighted_projected_graph",
    "generic_weighted_projected_graph",
]


@nx._dispatchable(
    graphs="B", preserve_node_attrs=True, preserve_graph_attrs=True, returns_graph=True
)
def projected_graph(B, nodes, multigraph=False):
    r"""Returns the projection of B onto one of its node sets.

    Returns the graph G that is the projection of the bipartite graph B
    onto the specified nodes. They retain their attributes and are connected
    in G if they have a common neighbor in B.

    Parameters
    ----------
    B : NetworkX graph
      The input graph should be bipartite.

    nodes : list or iterable
      Nodes to project onto (the "bottom" nodes).

    multigraph: bool (default=False)
       If True return a multigraph where the multiple edges represent multiple
       shared neighbors.  They edge key in the multigraph is assigned to the
       label of the neighbor.

    Returns
    -------
    Graph : NetworkX graph or multigraph
       A graph that is the projection onto the given nodes.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> B = nx.path_graph(4)
    >>> G = bipartite.projected_graph(B, [1, 3])
    >>> list(G)
    [1, 3]
    >>> list(G.edges())
    [(1, 3)]

    If nodes `a`, and `b` are connected through both nodes 1 and 2 then
    building a multigraph results in two edges in the projection onto
    [`a`, `b`]:

    >>> B = nx.Graph()
    >>> B.add_edges_from([("a", 1), ("b", 1), ("a", 2), ("b", 2)])
    >>> G = bipartite.projected_graph(B, ["a", "b"], multigraph=True)
    >>> print([sorted((u, v)) for u, v in G.edges()])
    [['a', 'b'], ['a', 'b']]

    Notes
    -----
    No attempt is made to verify that the input graph B is bipartite.
    Returns a simple graph that is the projection of the bipartite graph B
    onto the set of nodes given in list nodes.  If multigraph=True then
    a multigraph is returned with an edge for every shared neighbor.

    Directed graphs are allowed as input.  The output will also then
    be a directed graph with edges if there is a directed path between
    the nodes.

    The graph and node properties are (shallow) copied to the projected graph.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    is_bipartite,
    is_bipartite_node_set,
    sets,
    weighted_projected_graph,
    collaboration_weighted_projected_graph,
    overlap_weighted_projected_graph,
    generic_weighted_projected_graph
    """
    if B.is_multigraph():
        raise nx.NetworkXError("not defined for multigraphs")
    if B.is_directed():
        directed = True
        if multigraph:
            G = nx.MultiDiGraph()
        else:
            G = nx.DiGraph()
    else:
        directed = False
        if multigraph:
            G = nx.MultiGraph()
        else:
            G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n, B.nodes[n]) for n in nodes)
    for u in nodes:
        nbrs2 = {v for nbr in B[u] for v in B[nbr] if v != u}
        if multigraph:
            for n in nbrs2:
                if directed:
                    links = set(B[u]) & set(B.pred[n])
                else:
                    links = set(B[u]) & set(B[n])
                for l in links:
                    if not G.has_edge(u, n, l):
                        G.add_edge(u, n, key=l)
        else:
            G.add_edges_from((u, n) for n in nbrs2)
    return G


@not_implemented_for("multigraph")
@nx._dispatchable(graphs="B", returns_graph=True)
def weighted_projected_graph(B, nodes, ratio=False):
    r"""Returns a weighted projection of B onto one of its node sets.

    The weighted projected graph is the projection of the bipartite
    network B onto the specified nodes with weights representing the
    number of shared neighbors or the ratio between actual shared
    neighbors and possible shared neighbors if ``ratio is True`` [1]_.
    The nodes retain their attributes and are connected in the resulting
    graph if they have an edge to a common node in the original graph.

    Parameters
    ----------
    B : NetworkX graph
        The input graph should be bipartite.

    nodes : list or iterable
        Distinct nodes to project onto (the "bottom" nodes).

    ratio: Bool (default=False)
        If True, edge weight is the ratio between actual shared neighbors
        and maximum possible shared neighbors (i.e., the size of the other
        node set). If False, edges weight is the number of shared neighbors.

    Returns
    -------
    Graph : NetworkX graph
       A graph that is the projection onto the given nodes.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> B = nx.path_graph(4)
    >>> G = bipartite.weighted_projected_graph(B, [1, 3])
    >>> list(G)
    [1, 3]
    >>> list(G.edges(data=True))
    [(1, 3, {'weight': 1})]
    >>> G = bipartite.weighted_projected_graph(B, [1, 3], ratio=True)
    >>> list(G.edges(data=True))
    [(1, 3, {'weight': 0.5})]

    Notes
    -----
    No attempt is made to verify that the input graph B is bipartite, or that
    the input nodes are distinct. However, if the length of the input nodes is
    greater than or equal to the nodes in the graph B, an exception is raised.
    If the nodes are not distinct but don't raise this error, the output weights
    will be incorrect.
    The graph and node properties are (shallow) copied to the projected graph.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    is_bipartite,
    is_bipartite_node_set,
    sets,
    collaboration_weighted_projected_graph,
    overlap_weighted_projected_graph,
    generic_weighted_projected_graph
    projected_graph

    References
    ----------
    .. [1] Borgatti, S.P. and Halgin, D. In press. "Analyzing Affiliation
        Networks". In Carrington, P. and Scott, J. (eds) The Sage Handbook
        of Social Network Analysis. Sage Publications.
    """
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n, B.nodes[n]) for n in nodes)
    n_top = len(B) - len(nodes)

    if n_top < 1:
        raise NetworkXAlgorithmError(
            f"the size of the nodes to project onto ({len(nodes)}) is >= the graph size ({len(B)}).\n"
            "They are either not a valid bipartite partition or contain duplicates"
        )

    for u in nodes:
        unbrs = set(B[u])
        nbrs2 = {n for nbr in unbrs for n in B[nbr]} - {u}
        for v in nbrs2:
            vnbrs = set(pred[v])
            common = unbrs & vnbrs
            if not ratio:
                weight = len(common)
            else:
                weight = len(common) / n_top
            G.add_edge(u, v, weight=weight)
    return G


@not_implemented_for("multigraph")
@nx._dispatchable(graphs="B", returns_graph=True)
def collaboration_weighted_projected_graph(B, nodes):
    r"""Newman's weighted projection of B onto one of its node sets.

    The collaboration weighted projection is the projection of the
    bipartite network B onto the specified nodes with weights assigned
    using Newman's collaboration model [1]_:

    .. math::

        w_{u, v} = \sum_k \frac{\delta_{u}^{k} \delta_{v}^{k}}{d_k - 1}

    where `u` and `v` are nodes from the bottom bipartite node set,
    and `k` is a node of the top node set.
    The value `d_k` is the degree of node `k` in the bipartite
    network and `\delta_{u}^{k}` is 1 if node `u` is
    linked to node `k` in the original bipartite graph or 0 otherwise.

    The nodes retain their attributes and are connected in the resulting
    graph if have an edge to a common node in the original bipartite
    graph.

    Parameters
    ----------
    B : NetworkX graph
      The input graph should be bipartite.

    nodes : list or iterable
      Nodes to project onto (the "bottom" nodes).

    Returns
    -------
    Graph : NetworkX graph
       A graph that is the projection onto the given nodes.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> B = nx.path_graph(5)
    >>> B.add_edge(1, 5)
    >>> G = bipartite.collaboration_weighted_projected_graph(B, [0, 2, 4, 5])
    >>> list(G)
    [0, 2, 4, 5]
    >>> for edge in sorted(G.edges(data=True)):
    ...     print(edge)
    (0, 2, {'weight': 0.5})
    (0, 5, {'weight': 0.5})
    (2, 4, {'weight': 1.0})
    (2, 5, {'weight': 0.5})

    Notes
    -----
    No attempt is made to verify that the input graph B is bipartite.
    The graph and node properties are (shallow) copied to the projected graph.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    is_bipartite,
    is_bipartite_node_set,
    sets,
    weighted_projected_graph,
    overlap_weighted_projected_graph,
    generic_weighted_projected_graph,
    projected_graph

    References
    ----------
    .. [1] Scientific collaboration networks: II.
        Shortest paths, weighted networks, and centrality,
        M. E. J. Newman, Phys. Rev. E 64, 016132 (2001).
    """
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n, B.nodes[n]) for n in nodes)
    for u in nodes:
        unbrs = set(B[u])
        nbrs2 = {n for nbr in unbrs for n in B[nbr] if n != u}
        for v in nbrs2:
            vnbrs = set(pred[v])
            common_degree = (len(B[n]) for n in unbrs & vnbrs)
            weight = sum(1.0 / (deg - 1) for deg in common_degree if deg > 1)
            G.add_edge(u, v, weight=weight)
    return G


@not_implemented_for("multigraph")
@nx._dispatchable(graphs="B", returns_graph=True)
def overlap_weighted_projected_graph(B, nodes, jaccard=True):
    r"""Overlap weighted projection of B onto one of its node sets.

    The overlap weighted projection is the projection of the bipartite
    network B onto the specified nodes with weights representing
    the Jaccard index between the neighborhoods of the two nodes in the
    original bipartite network [1]_:

    .. math::

        w_{v, u} = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}

    or if the parameter 'jaccard' is False, the fraction of common
    neighbors by minimum of both nodes degree in the original
    bipartite graph [1]_:

    .. math::

        w_{v, u} = \frac{|N(u) \cap N(v)|}{min(|N(u)|, |N(v)|)}

    The nodes retain their attributes and are connected in the resulting
    graph if have an edge to a common node in the original bipartite graph.

    Parameters
    ----------
    B : NetworkX graph
        The input graph should be bipartite.

    nodes : list or iterable
        Nodes to project onto (the "bottom" nodes).

    jaccard: Bool (default=True)

    Returns
    -------
    Graph : NetworkX graph
       A graph that is the projection onto the given nodes.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> B = nx.path_graph(5)
    >>> nodes = [0, 2, 4]
    >>> G = bipartite.overlap_weighted_projected_graph(B, nodes)
    >>> list(G)
    [0, 2, 4]
    >>> list(G.edges(data=True))
    [(0, 2, {'weight': 0.5}), (2, 4, {'weight': 0.5})]
    >>> G = bipartite.overlap_weighted_projected_graph(B, nodes, jaccard=False)
    >>> list(G.edges(data=True))
    [(0, 2, {'weight': 1.0}), (2, 4, {'weight': 1.0})]

    Notes
    -----
    No attempt is made to verify that the input graph B is bipartite.
    The graph and node properties are (shallow) copied to the projected graph.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    is_bipartite,
    is_bipartite_node_set,
    sets,
    weighted_projected_graph,
    collaboration_weighted_projected_graph,
    generic_weighted_projected_graph,
    projected_graph

    References
    ----------
    .. [1] Borgatti, S.P. and Halgin, D. In press. Analyzing Affiliation
        Networks. In Carrington, P. and Scott, J. (eds) The Sage Handbook
        of Social Network Analysis. Sage Publications.

    """
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n, B.nodes[n]) for n in nodes)
    for u in nodes:
        unbrs = set(B[u])
        nbrs2 = {n for nbr in unbrs for n in B[nbr]} - {u}
        for v in nbrs2:
            vnbrs = set(pred[v])
            if jaccard:
                wt = len(unbrs & vnbrs) / len(unbrs | vnbrs)
            else:
                wt = len(unbrs & vnbrs) / min(len(unbrs), len(vnbrs))
            G.add_edge(u, v, weight=wt)
    return G


@not_implemented_for("multigraph")
@nx._dispatchable(graphs="B", preserve_all_attrs=True, returns_graph=True)
def generic_weighted_projected_graph(B, nodes, weight_function=None):
    r"""Weighted projection of B with a user-specified weight function.

    The bipartite network B is projected on to the specified nodes
    with weights computed by a user-specified function.  This function
    must accept as a parameter the neighborhood sets of two nodes and
    return an integer or a float.

    The nodes retain their attributes and are connected in the resulting graph
    if they have an edge to a common node in the original graph.

    Parameters
    ----------
    B : NetworkX graph
        The input graph should be bipartite.

    nodes : list or iterable
        Nodes to project onto (the "bottom" nodes).

    weight_function : function
        This function must accept as parameters the same input graph
        that this function, and two nodes; and return an integer or a float.
        The default function computes the number of shared neighbors.

    Returns
    -------
    Graph : NetworkX graph
       A graph that is the projection onto the given nodes.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> # Define some custom weight functions
    >>> def jaccard(G, u, v):
    ...     unbrs = set(G[u])
    ...     vnbrs = set(G[v])
    ...     return float(len(unbrs & vnbrs)) / len(unbrs | vnbrs)
    >>> def my_weight(G, u, v, weight="weight"):
    ...     w = 0
    ...     for nbr in set(G[u]) & set(G[v]):
    ...         w += G[u][nbr].get(weight, 1) + G[v][nbr].get(weight, 1)
    ...     return w
    >>> # A complete bipartite graph with 4 nodes and 4 edges
    >>> B = nx.complete_bipartite_graph(2, 2)
    >>> # Add some arbitrary weight to the edges
    >>> for i, (u, v) in enumerate(B.edges()):
    ...     B.edges[u, v]["weight"] = i + 1
    >>> for edge in B.edges(data=True):
    ...     print(edge)
    (0, 2, {'weight': 1})
    (0, 3, {'weight': 2})
    (1, 2, {'weight': 3})
    (1, 3, {'weight': 4})
    >>> # By default, the weight is the number of shared neighbors
    >>> G = bipartite.generic_weighted_projected_graph(B, [0, 1])
    >>> print(list(G.edges(data=True)))
    [(0, 1, {'weight': 2})]
    >>> # To specify a custom weight function use the weight_function parameter
    >>> G = bipartite.generic_weighted_projected_graph(
    ...     B, [0, 1], weight_function=jaccard
    ... )
    >>> print(list(G.edges(data=True)))
    [(0, 1, {'weight': 1.0})]
    >>> G = bipartite.generic_weighted_projected_graph(
    ...     B, [0, 1], weight_function=my_weight
    ... )
    >>> print(list(G.edges(data=True)))
    [(0, 1, {'weight': 10})]

    Notes
    -----
    No attempt is made to verify that the input graph B is bipartite.
    The graph and node properties are (shallow) copied to the projected graph.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    is_bipartite,
    is_bipartite_node_set,
    sets,
    weighted_projected_graph,
    collaboration_weighted_projected_graph,
    overlap_weighted_projected_graph,
    projected_graph

    """
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    if weight_function is None:

        def weight_function(G, u, v):
            # Notice that we use set(pred[v]) for handling the directed case.
            return len(set(G[u]) & set(pred[v]))

    G.graph.update(B.graph)
    G.add_nodes_from((n, B.nodes[n]) for n in nodes)
    for u in nodes:
        nbrs2 = {n for nbr in set(B[u]) for n in B[nbr]} - {u}
        for v in nbrs2:
            weight = weight_function(B, u, v)
            G.add_edge(u, v, weight=weight)
    return G
<|endoftext|>"""Provides a function for computing the extendability of a graph which is
undirected, simple, connected and bipartite and contains at least one perfect matching."""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["maximal_extendability"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def maximal_extendability(G):
    """Computes the extendability of a graph.

    The extendability of a graph is defined as the maximum $k$ for which `G`
    is $k$-extendable. Graph `G` is $k$-extendable if and only if `G` has a
    perfect matching and every set of $k$ independent edges can be extended
    to a perfect matching in `G`.

    Parameters
    ----------
    G : NetworkX Graph
        A fully-connected bipartite graph without self-loops

    Returns
    -------
    extendability : int

    Raises
    ------
    NetworkXError
       If the graph `G` is disconnected.
       If the graph `G` is not bipartite.
       If the graph `G` does not contain a perfect matching.
       If the residual graph of `G` is not strongly connected.

    Notes
    -----
    Definition:
    Let `G` be a simple, connected, undirected and bipartite graph with a perfect
    matching M and bipartition (U,V). The residual graph of `G`, denoted by $G_M$,
    is the graph obtained from G by directing the edges of M from V to U and the
    edges that do not belong to M from U to V.

    Lemma [1]_ :
    Let M be a perfect matching of `G`. `G` is $k$-extendable if and only if its residual
    graph $G_M$ is strongly connected and there are $k$ vertex-disjoint directed
    paths between every vertex of U and every vertex of V.

    Assuming that input graph `G` is undirected, simple, connected, bipartite and contains
    a perfect matching M, this function constructs the residual graph $G_M$ of G and
    returns the minimum value among the maximum vertex-disjoint directed paths between
    every vertex of U and every vertex of V in $G_M$. By combining the definitions
    and the lemma, this value represents the extendability of the graph `G`.

    Time complexity O($n^3$ $m^2$)) where $n$ is the number of vertices
    and $m$ is the number of edges.

    References
    ----------
    .. [1] "A polynomial algorithm for the extendability problem in bipartite graphs",
          J. Lakhal, L. Litzler, Information Processing Letters, 1998.
    .. [2] "On n-extendible graphs", M. D. Plummer, Discrete Mathematics, 31:201–210, 1980
          https://doi.org/10.1016/0012-365X(80)90037-0

    """
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph G is not connected")

    if not nx.bipartite.is_bipartite(G):
        raise nx.NetworkXError("Graph G is not bipartite")

    U, V = nx.bipartite.sets(G)

    maximum_matching = nx.bipartite.hopcroft_karp_matching(G)

    if not nx.is_perfect_matching(G, maximum_matching):
        raise nx.NetworkXError("Graph G does not contain a perfect matching")

    # list of edges in perfect matching, directed from V to U
    pm = [(node, maximum_matching[node]) for node in V & maximum_matching.keys()]

    # Direct all the edges of G, from V to U if in matching, else from U to V
    directed_edges = [
        (x, y) if (x in V and (x, y) in pm) or (x in U and (y, x) not in pm) else (y, x)
        for x, y in G.edges
    ]

    # Construct the residual graph of G
    residual_G = nx.DiGraph()
    residual_G.add_nodes_from(G)
    residual_G.add_edges_from(directed_edges)

    if not nx.is_strongly_connected(residual_G):
        raise nx.NetworkXError("The residual graph of G is not strongly connected")

    # For node-pairs between V & U, keep min of max number of node-disjoint paths
    # Variable $k$ stands for the extendability of graph G
    k = float("inf")
    for u in U:
        for v in V:
            num_paths = sum(1 for _ in nx.node_disjoint_paths(residual_G, u, v))
            k = k if k < num_paths else num_paths
    return k
<|endoftext|>"""
Greedy graph coloring using various strategies.
"""

import itertools
from collections import defaultdict, deque

import networkx as nx
from networkx.utils import arbitrary_element, py_random_state

__all__ = [
    "greedy_color",
    "strategy_connected_sequential",
    "strategy_connected_sequential_bfs",
    "strategy_connected_sequential_dfs",
    "strategy_independent_set",
    "strategy_largest_first",
    "strategy_random_sequential",
    "strategy_saturation_largest_first",
    "strategy_smallest_last",
]


def strategy_largest_first(G, colors):
    """Returns a list of the nodes of ``G`` in decreasing order by
    degree.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    """
    return sorted(G, key=G.degree, reverse=True)


@py_random_state(2)
def strategy_random_sequential(G, colors, seed=None):
    """Returns a random permutation of the nodes of ``G`` as a list.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
    nodes = list(G)
    seed.shuffle(nodes)
    return nodes


def strategy_smallest_last(G, colors):
    """Returns a deque of the nodes of ``G``, "smallest" last.

    Specifically, the degrees of each node are tracked in a bucket queue.
    From this, the node of minimum degree is repeatedly popped from the
    graph, updating its neighbors' degrees.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    This implementation of the strategy runs in $O(n + m)$ time
    (ignoring polylogarithmic factors), where $n$ is the number of nodes
    and $m$ is the number of edges.

    This strategy is related to :func:`strategy_independent_set`: if we
    interpret each node removed as an independent set of size one, then
    this strategy chooses an independent set of size one instead of a
    maximal independent set.

    """
    H = G.copy()
    result = deque()

    # Build initial degree list (i.e. the bucket queue data structure)
    degrees = defaultdict(set)  # set(), for fast random-access removals
    lbound = float("inf")
    for node, d in H.degree():
        degrees[d].add(node)
        lbound = min(lbound, d)  # Lower bound on min-degree.

    def find_min_degree():
        # Save time by starting the iterator at `lbound`, not 0.
        # The value that we find will be our new `lbound`, which we set later.
        return next(d for d in itertools.count(lbound) if d in degrees)

    for _ in G:
        # Pop a min-degree node and add it to the list.
        min_degree = find_min_degree()
        u = degrees[min_degree].pop()
        if not degrees[min_degree]:  # Clean up the degree list.
            del degrees[min_degree]
        result.appendleft(u)

        # Update degrees of removed node's neighbors.
        for v in H[u]:
            degree = H.degree(v)
            degrees[degree].remove(v)
            if not degrees[degree]:  # Clean up the degree list.
                del degrees[degree]
            degrees[degree - 1].add(v)

        # Finally, remove the node.
        H.remove_node(u)
        lbound = min_degree - 1  # Subtract 1 in case of tied neighbors.

    return result


def _maximal_independent_set(G):
    """Returns a maximal independent set of nodes in ``G`` by repeatedly
    choosing an independent node of minimum degree (with respect to the
    subgraph of unchosen nodes).

    """
    result = set()
    remaining = set(G)
    while remaining:
        G = G.subgraph(remaining)
        v = min(remaining, key=G.degree)
        result.add(v)
        remaining -= set(G[v]) | {v}
    return result


def strategy_independent_set(G, colors):
    """Uses a greedy independent set removal strategy to determine the
    colors.

    This function updates ``colors`` **in-place** and return ``None``,
    unlike the other strategy functions in this module.

    This algorithm repeatedly finds and removes a maximal independent
    set, assigning each node in the set an unused color.

    ``G`` is a NetworkX graph.

    This strategy is related to :func:`strategy_smallest_last`: in that
    strategy, an independent set of size one is chosen at each step
    instead of a maximal independent set.

    """
    remaining_nodes = set(G)
    while len(remaining_nodes) > 0:
        nodes = _maximal_independent_set(G.subgraph(remaining_nodes))
        remaining_nodes -= nodes
        yield from nodes


def strategy_connected_sequential_bfs(G, colors):
    """Returns an iterable over nodes in ``G`` in the order given by a
    breadth-first traversal.

    The generated sequence has the property that for each node except
    the first, at least one neighbor appeared earlier in the sequence.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    """
    return strategy_connected_sequential(G, colors, "bfs")


def strategy_connected_sequential_dfs(G, colors):
    """Returns an iterable over nodes in ``G`` in the order given by a
    depth-first traversal.

    The generated sequence has the property that for each node except
    the first, at least one neighbor appeared earlier in the sequence.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    """
    return strategy_connected_sequential(G, colors, "dfs")


def strategy_connected_sequential(G, colors, traversal="bfs"):
    """Returns an iterable over nodes in ``G`` in the order given by a
    breadth-first or depth-first traversal.

    ``traversal`` must be one of the strings ``'dfs'`` or ``'bfs'``,
    representing depth-first traversal or breadth-first traversal,
    respectively.

    The generated sequence has the property that for each node except
    the first, at least one neighbor appeared earlier in the sequence.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    """
    if traversal == "bfs":
        traverse = nx.bfs_edges
    elif traversal == "dfs":
        traverse = nx.dfs_edges
    else:
        raise nx.NetworkXError(
            "Please specify one of the strings 'bfs' or"
            " 'dfs' for connected sequential ordering"
        )
    for component in nx.connected_components(G):
        source = arbitrary_element(component)
        # Yield the source node, then all the nodes in the specified
        # traversal order.
        yield source
        for _, end in traverse(G.subgraph(component), source):
            yield end


def strategy_saturation_largest_first(G, colors):
    """Iterates over all the nodes of ``G`` in "saturation order" (also
    known as "DSATUR").

    ``G`` is a NetworkX graph. ``colors`` is a dictionary mapping nodes of
    ``G`` to colors, for those nodes that have already been colored.

    """
    distinct_colors = {v: set() for v in G}

    # Add the node color assignments given in colors to the
    # distinct colors set for each neighbor of that node
    for node, color in colors.items():
        for neighbor in G[node]:
            distinct_colors[neighbor].add(color)

    # Check that the color assignments in colors are valid
    # i.e. no neighboring nodes have the same color
    if len(colors) >= 2:
        for node, color in colors.items():
            if color in distinct_colors[node]:
                raise nx.NetworkXError("Neighboring nodes must have different colors")

    # If 0 nodes have been colored, simply choose the node of highest degree.
    if not colors:
        node = max(G, key=G.degree)
        yield node
        # Add the color 0 to the distinct colors set for each
        # neighbor of that node.
        for v in G[node]:
            distinct_colors[v].add(0)

    while len(G) != len(colors):
        # Update the distinct color sets for the neighbors.
        for node, color in colors.items():
            for neighbor in G[node]:
                distinct_colors[neighbor].add(color)

        # Compute the maximum saturation and the set of nodes that
        # achieve that saturation.
        saturation = {v: len(c) for v, c in distinct_colors.items() if v not in colors}
        # Yield the node with the highest saturation, and break ties by
        # degree.
        node = max(saturation, key=lambda v: (saturation[v], G.degree(v)))
        yield node


#: Dictionary mapping name of a strategy as a string to the strategy function.
STRATEGIES = {
    "largest_first": strategy_largest_first,
    "random_sequential": strategy_random_sequential,
    "smallest_last": strategy_smallest_last,
    "independent_set": strategy_independent_set,
    "connected_sequential_bfs": strategy_connected_sequential_bfs,
    "connected_sequential_dfs": strategy_connected_sequential_dfs,
    "connected_sequential": strategy_connected_sequential,
    "saturation_largest_first": strategy_saturation_largest_first,
    "DSATUR": strategy_saturation_largest_first,
}


@nx._dispatchable
def greedy_color(G, strategy="largest_first", interchange=False):
    """Color a graph using various strategies of greedy graph coloring.

    Attempts to color a graph using as few colors as possible, where no
    neighbors of a node can have same color as the node itself. The
    given strategy determines the order in which nodes are colored.

    The strategies are described in [1]_, and smallest-last is based on
    [2]_.

    Parameters
    ----------
    G : NetworkX graph

    strategy : string or function(G, colors)
       A function (or a string representing a function) that provides
       the coloring strategy, by returning nodes in the ordering they
       should be colored. ``G`` is the graph, and ``colors`` is a
       dictionary of the currently assigned colors, keyed by nodes. The
       function must return an iterable over all the nodes in ``G``.

       If the strategy function is an iterator generator (that is, a
       function with ``yield`` statements), keep in mind that the
       ``colors`` dictionary will be updated after each ``yield``, since
       this function chooses colors greedily.

       If ``strategy`` is a string, it must be one of the following,
       each of which represents one of the built-in strategy functions.

       * ``'largest_first'``
       * ``'random_sequential'``
       * ``'smallest_last'``
       * ``'independent_set'``
       * ``'connected_sequential_bfs'``
       * ``'connected_sequential_dfs'``
       * ``'connected_sequential'`` (alias for the previous strategy)
       * ``'saturation_largest_first'``
       * ``'DSATUR'`` (alias for the previous strategy)

    interchange: bool
       Will use the color interchange algorithm described by [3]_ if set
       to ``True``.

       Note that ``saturation_largest_first`` and ``independent_set``
       do not work with interchange. Furthermore, if you use
       interchange with your own strategy function, you cannot rely
       on the values in the ``colors`` argument.

    Returns
    -------
    A dictionary with keys representing nodes and values representing
    corresponding coloring.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> d = nx.coloring.greedy_color(G, strategy="largest_first")
    >>> d in [{0: 0, 1: 1, 2: 0, 3: 1}, {0: 1, 1: 0, 2: 1, 3: 0}]
    True

    Raises
    ------
    NetworkXPointlessConcept
        If ``strategy`` is ``saturation_largest_first`` or
        ``independent_set`` and ``interchange`` is ``True``.

    References
    ----------
    .. [1] Adrian Kosowski, and Krzysztof Manuszewski,
       Classical Coloring of Graphs, Graph Colorings, 2-19, 2004.
       ISBN 0-8218-3458-4.
    .. [2] David W. Matula, and Leland L. Beck, "Smallest-last
       ordering and clustering and graph coloring algorithms." *J. ACM* 30,
       3 (July 1983), 417–427. <https://doi.org/10.1145/2402.322385>
    .. [3] Maciej M. Sysło, Narsingh Deo, Janusz S. Kowalik,
       Discrete Optimization Algorithms with Pascal Programs, 415-424, 1983.
       ISBN 0-486-45353-7.

    """
    if len(G) == 0:
        return {}
    # Determine the strategy provided by the caller.
    strategy = STRATEGIES.get(strategy, strategy)
    if not callable(strategy):
        raise nx.NetworkXError(
            f"strategy must be callable or a valid string. {strategy} not valid."
        )
    # Perform some validation on the arguments before executing any
    # strategy functions.
    if interchange:
        if strategy is strategy_independent_set:
            msg = "interchange cannot be used with independent_set"
            raise nx.NetworkXPointlessConcept(msg)
        if strategy is strategy_saturation_largest_first:
            msg = "interchange cannot be used with" " saturation_largest_first"
            raise nx.NetworkXPointlessConcept(msg)
    colors = {}
    nodes = strategy(G, colors)
    if interchange:
        return _greedy_coloring_with_interchange(G, nodes)
    for u in nodes:
        # Set to keep track of colors of neighbors
        nbr_colors = {colors[v] for v in G[u] if v in colors}
        # Find the first unused color.
        for color in itertools.count():
            if color not in nbr_colors:
                break
        # Assign the new color to the current node.
        colors[u] = color
    return colors


# Tools for coloring with interchanges
class _Node:
    __slots__ = ["node_id", "color", "adj_list", "adj_color"]

    def __init__(self, node_id, n):
        self.node_id = node_id
        self.color = -1
        self.adj_list = None
        self.adj_color = [None for _ in range(n)]

    def __repr__(self):
        return (
            f"Node_id: {self.node_id}, Color: {self.color}, "
            f"Adj_list: ({self.adj_list}), adj_color: ({self.adj_color})"
        )

    def assign_color(self, adj_entry, color):
        adj_entry.col_prev = None
        adj_entry.col_next = self.adj_color[color]
        self.adj_color[color] = adj_entry
        if adj_entry.col_next is not None:
            adj_entry.col_next.col_prev = adj_entry

    def clear_color(self, adj_entry, color):
        if adj_entry.col_prev is None:
            self.adj_color[color] = adj_entry.col_next
        else:
            adj_entry.col_prev.col_next = adj_entry.col_next
        if adj_entry.col_next is not None:
            adj_entry.col_next.col_prev = adj_entry.col_prev

    def iter_neighbors(self):
        adj_node = self.adj_list
        while adj_node is not None:
            yield adj_node
            adj_node = adj_node.next

    def iter_neighbors_color(self, color):
        adj_color_node = self.adj_color[color]
        while adj_color_node is not None:
            yield adj_color_node.node_id
            adj_color_node = adj_color_node.col_next


class _AdjEntry:
    __slots__ = ["node_id", "next", "mate", "col_next", "col_prev"]

    def __init__(self, node_id):
        self.node_id = node_id
        self.next = None
        self.mate = None
        self.col_next = None
        self.col_prev = None

    def __repr__(self):
        col_next = None if self.col_next is None else self.col_next.node_id
        col_prev = None if self.col_prev is None else self.col_prev.node_id
        return (
            f"Node_id: {self.node_id}, Next: ({self.next}), "
            f"Mate: ({self.mate.node_id}), "
            f"col_next: ({col_next}), col_prev: ({col_prev})"
        )


def _greedy_coloring_with_interchange(G, nodes):
    """Return a coloring for `original_graph` using interchange approach

    This procedure is an adaption of the algorithm described by [1]_,
    and is an implementation of coloring with interchange. Please be
    advised, that the datastructures used are rather complex because
    they are optimized to minimize the time spent identifying
    subcomponents of the graph, which are possible candidates for color
    interchange.

    Parameters
    ----------
    G : NetworkX graph
        The graph to be colored

    nodes : list
        nodes ordered using the strategy of choice

    Returns
    -------
    dict :
        A dictionary keyed by node to a color value

    References
    ----------
    .. [1] Maciej M. Syslo, Narsingh Deo, Janusz S. Kowalik,
       Discrete Optimization Algorithms with Pascal Programs, 415-424, 1983.
       ISBN 0-486-45353-7.
    """
    n = len(G)

    graph = {node: _Node(node, n) for node in G}

    for node1, node2 in G.edges():
        adj_entry1 = _AdjEntry(node2)
        adj_entry2 = _AdjEntry(node1)
        adj_entry1.mate = adj_entry2
        adj_entry2.mate = adj_entry1
        node1_head = graph[node1].adj_list
        adj_entry1.next = node1_head
        graph[node1].adj_list = adj_entry1
        node2_head = graph[node2].adj_list
        adj_entry2.next = node2_head
        graph[node2].adj_list = adj_entry2

    k = 0
    for node in nodes:
        # Find the smallest possible, unused color
        neighbors = graph[node].iter_neighbors()
        col_used = {graph[adj_node.node_id].color for adj_node in neighbors}
        col_used.discard(-1)
        k1 = next(itertools.dropwhile(lambda x: x in col_used, itertools.count()))

        # k1 is now the lowest available color
        if k1 > k:
            connected = True
            visited = set()
            col1 = -1
            col2 = -1
            while connected and col1 < k:
                col1 += 1
                neighbor_cols = graph[node].iter_neighbors_color(col1)
                col1_adj = list(neighbor_cols)

                col2 = col1
                while connected and col2 < k:
                    col2 += 1
                    visited = set(col1_adj)
                    frontier = list(col1_adj)
                    i = 0
                    while i < len(frontier):
                        search_node = frontier[i]
                        i += 1
                        col_opp = col2 if graph[search_node].color == col1 else col1
                        neighbor_cols = graph[search_node].iter_neighbors_color(col_opp)

                        for neighbor in neighbor_cols:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                frontier.append(neighbor)

                    # Search if node is not adj to any col2 vertex
                    connected = (
                        len(
                            visited.intersection(graph[node].iter_neighbors_color(col2))
                        )
                        > 0
                    )

            # If connected is false then we can swap !!!
            if not connected:
                # Update all the nodes in the component
                for search_node in visited:
                    graph[search_node].color = (
                        col2 if graph[search_node].color == col1 else col1
                    )
                    col2_adj = graph[search_node].adj_color[col2]
                    graph[search_node].adj_color[col2] = graph[search_node].adj_color[
                        col1
                    ]
                    graph[search_node].adj_color[col1] = col2_adj

                # Update all the neighboring nodes
                for search_node in visited:
                    col = graph[search_node].color
                    col_opp = col1 if col == col2 else col2
                    for adj_node in graph[search_node].iter_neighbors():
                        if graph[adj_node.node_id].color != col_opp:
                            # Direct reference to entry
                            adj_mate = adj_node.mate
                            graph[adj_node.node_id].clear_color(adj_mate, col_opp)
                            graph[adj_node.node_id].assign_color(adj_mate, col)
                k1 = col1

        # We can color this node color k1
        graph[node].color = k1
        k = max(k1, k)

        # Update the neighbors of this node
        for adj_node in graph[node].iter_neighbors():
            adj_mate = adj_node.mate
            graph[adj_node.node_id].assign_color(adj_mate, k1)

    return {node.node_id: node.color for node in graph.values()}
<|endoftext|>"""
Equitable coloring of graphs with bounded degree.
"""

from collections import defaultdict

import networkx as nx

__all__ = ["equitable_color"]


@nx._dispatchable
def is_coloring(G, coloring):
    """Determine if the coloring is a valid coloring for the graph G."""
    # Verify that the coloring is valid.
    return all(coloring[s] != coloring[d] for s, d in G.edges)


@nx._dispatchable
def is_equitable(G, coloring, num_colors=None):
    """Determines if the coloring is valid and equitable for the graph G."""

    if not is_coloring(G, coloring):
        return False

    # Verify whether it is equitable.
    color_set_size = defaultdict(int)
    for color in coloring.values():
        color_set_size[color] += 1

    if num_colors is not None:
        for color in range(num_colors):
            if color not in color_set_size:
                # These colors do not have any vertices attached to them.
                color_set_size[color] = 0

    # If there are more than 2 distinct values, the coloring cannot be equitable
    all_set_sizes = set(color_set_size.values())
    if len(all_set_sizes) == 0 and num_colors is None:  # Was an empty graph
        return True
    elif len(all_set_sizes) == 1:
        return True
    elif len(all_set_sizes) == 2:
        a, b = list(all_set_sizes)
        return abs(a - b) <= 1
    else:  # len(all_set_sizes) > 2:
        return False


def make_C_from_F(F):
    C = defaultdict(list)
    for node, color in F.items():
        C[color].append(node)

    return C


def make_N_from_L_C(L, C):
    nodes = L.keys()
    colors = C.keys()
    return {
        (node, color): sum(1 for v in L[node] if v in C[color])
        for node in nodes
        for color in colors
    }


def make_H_from_C_N(C, N):
    return {
        (c1, c2): sum(1 for node in C[c1] if N[(node, c2)] == 0) for c1 in C for c2 in C
    }


def change_color(u, X, Y, N, H, F, C, L):
    """Change the color of 'u' from X to Y and update N, H, F, C."""
    assert F[u] == X and X != Y

    # Change the class of 'u' from X to Y
    F[u] = Y

    for k in C:
        # 'u' witnesses an edge from k -> Y instead of from k -> X now.
        if N[u, k] == 0:
            H[(X, k)] -= 1
            H[(Y, k)] += 1

    for v in L[u]:
        # 'v' has lost a neighbor in X and gained one in Y
        N[(v, X)] -= 1
        N[(v, Y)] += 1

        if N[(v, X)] == 0:
            # 'v' witnesses F[v] -> X
            H[(F[v], X)] += 1

        if N[(v, Y)] == 1:
            # 'v' no longer witnesses F[v] -> Y
            H[(F[v], Y)] -= 1

    C[X].remove(u)
    C[Y].append(u)


def move_witnesses(src_color, dst_color, N, H, F, C, T_cal, L):
    """Move witness along a path from src_color to dst_color."""
    X = src_color
    while X != dst_color:
        Y = T_cal[X]
        # Move _any_ witness from X to Y = T_cal[X]
        w = next(x for x in C[X] if N[(x, Y)] == 0)
        change_color(w, X, Y, N=N, H=H, F=F, C=C, L=L)
        X = Y


@nx._dispatchable(mutates_input=True)
def pad_graph(G, num_colors):
    """Add a disconnected complete clique K_p such that the number of nodes in
    the graph becomes a multiple of `num_colors`.

    Assumes that the graph's nodes are labelled using integers.

    Returns the number of nodes with each color.
    """

    n_ = len(G)
    r = num_colors - 1

    # Ensure that the number of nodes in G is a multiple of (r + 1)
    s = n_ // (r + 1)
    if n_ != s * (r + 1):
        p = (r + 1) - n_ % (r + 1)
        s += 1

        # Complete graph K_p between (imaginary) nodes [n_, ... , n_ + p]
        K = nx.relabel_nodes(nx.complete_graph(p), {idx: idx + n_ for idx in range(p)})
        G.add_edges_from(K.edges)

    return s


def procedure_P(V_minus, V_plus, N, H, F, C, L, excluded_colors=None):
    """Procedure P as described in the paper."""

    if excluded_colors is None:
        excluded_colors = set()

    A_cal = set()
    T_cal = {}
    R_cal = []

    # BFS to determine A_cal, i.e. colors reachable from V-
    reachable = [V_minus]
    marked = set(reachable)
    idx = 0

    while idx < len(reachable):
        pop = reachable[idx]
        idx += 1

        A_cal.add(pop)
        R_cal.append(pop)

        # TODO: Checking whether a color has been visited can be made faster by
        # using a look-up table instead of testing for membership in a set by a
        # logarithmic factor.
        next_layer = []
        for k in C:
            if (
                H[(k, pop)] > 0
                and k not in A_cal
                and k not in excluded_colors
                and k not in marked
            ):
                next_layer.append(k)

        for dst in next_layer:
            # Record that `dst` can reach `pop`
            T_cal[dst] = pop

        marked.update(next_layer)
        reachable.extend(next_layer)

    # Variables for the algorithm
    b = len(C) - len(A_cal)

    if V_plus in A_cal:
        # Easy case: V+ is in A_cal
        # Move one node from V+ to V- using T_cal to find the parents.
        move_witnesses(V_plus, V_minus, N=N, H=H, F=F, C=C, T_cal=T_cal, L=L)
    else:
        # If there is a solo edge, we can resolve the situation by
        # moving witnesses from B to A, making G[A] equitable and then
        # recursively balancing G[B - w] with a different V_minus and
        # but the same V_plus.

        A_0 = set()
        A_cal_0 = set()
        num_terminal_sets_found = 0
        made_equitable = False

        for W_1 in R_cal[::-1]:
            for v in C[W_1]:
                X = None

                for U in C:
                    if N[(v, U)] == 0 and U in A_cal and U != W_1:
                        X = U

                # v does not witness an edge in H[A_cal]
                if X is None:
                    continue

                for U in C:
                    # Note: Departing from the paper here.
                    if N[(v, U)] >= 1 and U not in A_cal:
                        X_prime = U
                        w = v

                        try:
                            # Finding the solo neighbor of w in X_prime
                            y = next(
                                node
                                for node in L[w]
                                if F[node] == X_prime and N[(node, W_1)] == 1
                            )
                        except StopIteration:
                            pass
                        else:
                            W = W_1

                            # Move w from W to X, now X has one extra node.
                            change_color(w, W, X, N=N, H=H, F=F, C=C, L=L)

                            # Move witness from X to V_minus, making the coloring
                            # equitable.
                            move_witnesses(
                                src_color=X,
                                dst_color=V_minus,
                                N=N,
                                H=H,
                                F=F,
                                C=C,
                                T_cal=T_cal,
                                L=L,
                            )

                            # Move y from X_prime to W, making W the correct size.
                            change_color(y, X_prime, W, N=N, H=H, F=F, C=C, L=L)

                            # Then call the procedure on G[B - y]
                            procedure_P(
                                V_minus=X_prime,
                                V_plus=V_plus,
                                N=N,
                                H=H,
                                C=C,
                                F=F,
                                L=L,
                                excluded_colors=excluded_colors.union(A_cal),
                            )
                            made_equitable = True
                            break

                if made_equitable:
                    break
            else:
                # No node in W_1 was found such that
                # it had a solo-neighbor.
                A_cal_0.add(W_1)
                A_0.update(C[W_1])
                num_terminal_sets_found += 1

            if num_terminal_sets_found == b:
                # Otherwise, construct the maximal independent set and find
                # a pair of z_1, z_2 as in Case II.

                # BFS to determine B_cal': the set of colors reachable from V+
                B_cal_prime = set()
                T_cal_prime = {}

                reachable = [V_plus]
                marked = set(reachable)
                idx = 0
                while idx < len(reachable):
                    pop = reachable[idx]
                    idx += 1

                    B_cal_prime.add(pop)

                    # No need to check for excluded_colors here because
                    # they only exclude colors from A_cal
                    next_layer = [
                        k
                        for k in C
                        if H[(pop, k)] > 0 and k not in B_cal_prime and k not in marked
                    ]

                    for dst in next_layer:
                        T_cal_prime[pop] = dst

                    marked.update(next_layer)
                    reachable.extend(next_layer)

                # Construct the independent set of G[B']
                I_set = set()
                I_covered = set()
                W_covering = {}

                B_prime = [node for k in B_cal_prime for node in C[k]]

                # Add the nodes in V_plus to I first.
                for z in C[V_plus] + B_prime:
                    if z in I_covered or F[z] not in B_cal_prime:
                        continue

                    I_set.add(z)
                    I_covered.add(z)
                    I_covered.update(list(L[z]))

                    for w in L[z]:
                        if F[w] in A_cal_0 and N[(z, F[w])] == 1:
                            if w not in W_covering:
                                W_covering[w] = z
                            else:
                                # Found z1, z2 which have the same solo
                                # neighbor in some W
                                z_1 = W_covering[w]
                                # z_2 = z

                                Z = F[z_1]
                                W = F[w]

                                # shift nodes along W, V-
                                move_witnesses(
                                    W, V_minus, N=N, H=H, F=F, C=C, T_cal=T_cal, L=L
                                )

                                # shift nodes along V+ to Z
                                move_witnesses(
                                    V_plus,
                                    Z,
                                    N=N,
                                    H=H,
                                    F=F,
                                    C=C,
                                    T_cal=T_cal_prime,
                                    L=L,
                                )

                                # change color of z_1 to W
                                change_color(z_1, Z, W, N=N, H=H, F=F, C=C, L=L)

                                # change color of w to some color in B_cal
                                W_plus = next(
                                    k for k in C if N[(w, k)] == 0 and k not in A_cal
                                )
                                change_color(w, W, W_plus, N=N, H=H, F=F, C=C, L=L)

                                # recurse with G[B \cup W*]
                                excluded_colors.update(
                                    [k for k in C if k != W and k not in B_cal_prime]
                                )
                                procedure_P(
                                    V_minus=W,
                                    V_plus=W_plus,
                                    N=N,
                                    H=H,
                                    C=C,
                                    F=F,
                                    L=L,
                                    excluded_colors=excluded_colors,
                                )

                                made_equitable = True
                                break

                    if made_equitable:
                        break
                else:
                    assert False, (
                        "Must find a w which is the solo neighbor "
                        "of two vertices in B_cal_prime."
                    )

            if made_equitable:
                break


@nx._dispatchable
def equitable_color(G, num_colors):
    """Provides an equitable coloring for nodes of `G`.

    Attempts to color a graph using `num_colors` colors, where no neighbors of
    a node can have same color as the node itself and the number of nodes with
    each color differ by at most 1. `num_colors` must be greater than the
    maximum degree of `G`. The algorithm is described in [1]_ and has
    complexity O(num_colors * n**2).

    Parameters
    ----------
    G : networkX graph
       The nodes of this graph will be colored.

    num_colors : number of colors to use
       This number must be at least one more than the maximum degree of nodes
       in the graph.

    Returns
    -------
    A dictionary with keys representing nodes and values representing
    corresponding coloring.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> nx.coloring.equitable_color(G, num_colors=3)  # doctest: +SKIP
    {0: 2, 1: 1, 2: 2, 3: 0}

    Raises
    ------
    NetworkXAlgorithmError
        If `num_colors` is not at least the maximum degree of the graph `G`

    References
    ----------
    .. [1] Kierstead, H. A., Kostochka, A. V., Mydlarz, M., & Szemerédi, E.
        (2010). A fast algorithm for equitable coloring. Combinatorica, 30(2),
        217-224.
    """

    # Map nodes to integers for simplicity later.
    nodes_to_int = {}
    int_to_nodes = {}

    for idx, node in enumerate(G.nodes):
        nodes_to_int[node] = idx
        int_to_nodes[idx] = node

    G = nx.relabel_nodes(G, nodes_to_int, copy=True)

    # Basic graph statistics and sanity check.
    if len(G.nodes) > 0:
        r_ = max(G.degree(node) for node in G.nodes)
    else:
        r_ = 0

    if r_ >= num_colors:
        raise nx.NetworkXAlgorithmError(
            f"Graph has maximum degree {r_}, needs "
            f"{r_ + 1} (> {num_colors}) colors for guaranteed coloring."
        )

    # Ensure that the number of nodes in G is a multiple of (r + 1)
    pad_graph(G, num_colors)

    # Starting the algorithm.
    # L = {node: list(G.neighbors(node)) for node in G.nodes}
    L_ = {node: [] for node in G.nodes}

    # Arbitrary equitable allocation of colors to nodes.
    F = {node: idx % num_colors for idx, node in enumerate(G.nodes)}

    C = make_C_from_F(F)

    # The neighborhood is empty initially.
    N = make_N_from_L_C(L_, C)

    # Currently all nodes witness all edges.
    H = make_H_from_C_N(C, N)

    # Start of algorithm.
    edges_seen = set()

    for u in sorted(G.nodes):
        for v in sorted(G.neighbors(u)):
            # Do not double count edges if (v, u) has already been seen.
            if (v, u) in edges_seen:
                continue

            edges_seen.add((u, v))

            L_[u].append(v)
            L_[v].append(u)

            N[(u, F[v])] += 1
            N[(v, F[u])] += 1

            if F[u] != F[v]:
                # Were 'u' and 'v' witnesses for F[u] -> F[v] or F[v] -> F[u]?
                if N[(u, F[v])] == 1:
                    H[F[u], F[v]] -= 1  # u cannot witness an edge between F[u], F[v]

                if N[(v, F[u])] == 1:
                    H[F[v], F[u]] -= 1  # v cannot witness an edge between F[v], F[u]

        if N[(u, F[u])] != 0:
            # Find the first color where 'u' does not have any neighbors.
            Y = next(k for k in C if N[(u, k)] == 0)
            X = F[u]
            change_color(u, X, Y, N=N, H=H, F=F, C=C, L=L_)

            # Procedure P
            procedure_P(V_minus=X, V_plus=Y, N=N, H=H, F=F, C=C, L=L_)

    return {int_to_nodes[x]: F[x] for x in int_to_nodes}
<|endoftext|>"""
Compute the shortest paths and path lengths between nodes in the graph.

These algorithms work with undirected and directed graphs.

"""

import warnings

import networkx as nx

__all__ = [
    "shortest_path",
    "all_shortest_paths",
    "single_source_all_shortest_paths",
    "all_pairs_all_shortest_paths",
    "shortest_path_length",
    "average_shortest_path_length",
    "has_path",
]


@nx._dispatchable
def has_path(G, source, target):
    """Returns *True* if *G* has a path from *source* to *target*.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path
    """
    try:
        nx.shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return False
    return True


@nx._dispatchable(edge_attrs="weight")
def shortest_path(G, source=None, target=None, weight=None, method="dijkstra"):
    """Compute shortest paths in the graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
        Starting node for path. If not specified, compute shortest
        paths for each possible starting node.

    target : node, optional
        Ending node for path. If not specified, compute shortest
        paths to all possible nodes.

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'dijkstra')
        The algorithm to use to compute the path.
        Supported options: 'dijkstra', 'bellman-ford'.
        Other inputs produce a ValueError.
        If `weight` is None, unweighted graph methods are used, and this
        suggestion is ignored.

    Returns
    -------
    path: list or dictionary
        All returned paths include both the source and target in the path.

        If the source and target are both specified, return a single list
        of nodes in a shortest path from the source to the target.

        If only the source is specified, return a dictionary keyed by
        targets with a list of nodes in a shortest path from the source
        to one of the targets.

        If only the target is specified, return a dictionary keyed by
        sources with a list of nodes in a shortest path from one of the
        sources to the target.

        If neither the source nor target are specified return a dictionary
        of dictionaries with path[source][target]=[list of nodes in path].

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    ValueError
        If `method` is not among the supported options.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> print(nx.shortest_path(G, source=0, target=4))
    [0, 1, 2, 3, 4]
    >>> p = nx.shortest_path(G, source=0)  # target not specified
    >>> p[3]  # shortest path from source=0 to target=3
    [0, 1, 2, 3]
    >>> p = nx.shortest_path(G, target=4)  # source not specified
    >>> p[1]  # shortest path from source=1 to target=4
    [1, 2, 3, 4]
    >>> p = dict(nx.shortest_path(G))  # source, target not specified
    >>> p[2][4]  # shortest path from source=2 to target=4
    [2, 3, 4]

    Notes
    -----
    There may be more than one shortest path between a source and target.
    This returns only one of them.

    See Also
    --------
    all_pairs_shortest_path
    all_pairs_dijkstra_path
    all_pairs_bellman_ford_path
    single_source_shortest_path
    single_source_dijkstra_path
    single_source_bellman_ford_path
    """
    if method not in ("dijkstra", "bellman-ford"):
        # so we don't need to check in each branch later
        raise ValueError(f"method not supported: {method}")
    method = "unweighted" if weight is None else method
    if source is None:
        if target is None:
            warnings.warn(
                (
                    "\n\nshortest_path will return an iterator that yields\n"
                    "(node, path) pairs instead of a dictionary when source\n"
                    "and target are unspecified beginning in version 3.5\n\n"
                    "To keep the current behavior, use:\n\n"
                    "\tdict(nx.shortest_path(G))"
                ),
                FutureWarning,
                stacklevel=3,
            )

            # Find paths between all pairs.
            if method == "unweighted":
                paths = dict(nx.all_pairs_shortest_path(G))
            elif method == "dijkstra":
                paths = dict(nx.all_pairs_dijkstra_path(G, weight=weight))
            else:  # method == 'bellman-ford':
                paths = dict(nx.all_pairs_bellman_ford_path(G, weight=weight))
        else:
            # Find paths from all nodes co-accessible to the target.
            if G.is_directed():
                G = G.reverse(copy=False)
            if method == "unweighted":
                paths = nx.single_source_shortest_path(G, target)
            elif method == "dijkstra":
                paths = nx.single_source_dijkstra_path(G, target, weight=weight)
            else:  # method == 'bellman-ford':
                paths = nx.single_source_bellman_ford_path(G, target, weight=weight)
            # Now flip the paths so they go from a source to the target.
            for target in paths:
                paths[target] = list(reversed(paths[target]))
    else:
        if target is None:
            # Find paths to all nodes accessible from the source.
            if method == "unweighted":
                paths = nx.single_source_shortest_path(G, source)
            elif method == "dijkstra":
                paths = nx.single_source_dijkstra_path(G, source, weight=weight)
            else:  # method == 'bellman-ford':
                paths = nx.single_source_bellman_ford_path(G, source, weight=weight)
        else:
            # Find shortest source-target path.
            if method == "unweighted":
                paths = nx.bidirectional_shortest_path(G, source, target)
            elif method == "dijkstra":
                _, paths = nx.bidirectional_dijkstra(G, source, target, weight)
            else:  # method == 'bellman-ford':
                paths = nx.bellman_ford_path(G, source, target, weight)
    return paths


@nx._dispatchable(edge_attrs="weight")
def shortest_path_length(G, source=None, target=None, weight=None, method="dijkstra"):
    """Compute shortest path lengths in the graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
        Starting node for path.
        If not specified, compute shortest path lengths using all nodes as
        source nodes.

    target : node, optional
        Ending node for path.
        If not specified, compute shortest path lengths using all nodes as
        target nodes.

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'dijkstra')
        The algorithm to use to compute the path length.
        Supported options: 'dijkstra', 'bellman-ford'.
        Other inputs produce a ValueError.
        If `weight` is None, unweighted graph methods are used, and this
        suggestion is ignored.

    Returns
    -------
    length: number or iterator
        If the source and target are both specified, return the length of
        the shortest path from the source to the target.

        If only the source is specified, return a dict keyed by target
        to the shortest path length from the source to that target.

        If only the target is specified, return a dict keyed by source
        to the shortest path length from that source to the target.

        If neither the source nor target are specified, return an iterator
        over (source, dictionary) where dictionary is keyed by target to
        shortest path length from source to that target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    ValueError
        If `method` is not among the supported options.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.shortest_path_length(G, source=0, target=4)
    4
    >>> p = nx.shortest_path_length(G, source=0)  # target not specified
    >>> p[4]
    4
    >>> p = nx.shortest_path_length(G, target=4)  # source not specified
    >>> p[0]
    4
    >>> p = dict(nx.shortest_path_length(G))  # source,target not specified
    >>> p[0][4]
    4

    Notes
    -----
    The length of the path is always 1 less than the number of nodes involved
    in the path since the length measures the number of edges followed.

    For digraphs this returns the shortest directed path length. To find path
    lengths in the reverse direction use G.reverse(copy=False) first to flip
    the edge orientation.

    See Also
    --------
    all_pairs_shortest_path_length
    all_pairs_dijkstra_path_length
    all_pairs_bellman_ford_path_length
    single_source_shortest_path_length
    single_source_dijkstra_path_length
    single_source_bellman_ford_path_length
    """
    if method not in ("dijkstra", "bellman-ford"):
        # so we don't need to check in each branch later
        raise ValueError(f"method not supported: {method}")
    method = "unweighted" if weight is None else method
    if source is None:
        if target is None:
            # Find paths between all pairs.
            if method == "unweighted":
                paths = nx.all_pairs_shortest_path_length(G)
            elif method == "dijkstra":
                paths = nx.all_pairs_dijkstra_path_length(G, weight=weight)
            else:  # method == 'bellman-ford':
                paths = nx.all_pairs_bellman_ford_path_length(G, weight=weight)
        else:
            # Find paths from all nodes co-accessible to the target.
            if G.is_directed():
                G = G.reverse(copy=False)
            if method == "unweighted":
                path_length = nx.single_source_shortest_path_length
                paths = path_length(G, target)
            elif method == "dijkstra":
                path_length = nx.single_source_dijkstra_path_length
                paths = path_length(G, target, weight=weight)
            else:  # method == 'bellman-ford':
                path_length = nx.single_source_bellman_ford_path_length
                paths = path_length(G, target, weight=weight)
    else:
        if target is None:
            # Find paths to all nodes accessible from the source.
            if method == "unweighted":
                paths = nx.single_source_shortest_path_length(G, source)
            elif method == "dijkstra":
                path_length = nx.single_source_dijkstra_path_length
                paths = path_length(G, source, weight=weight)
            else:  # method == 'bellman-ford':
                path_length = nx.single_source_bellman_ford_path_length
                paths = path_length(G, source, weight=weight)
        else:
            # Find shortest source-target path.
            if method == "unweighted":
                p = nx.bidirectional_shortest_path(G, source, target)
                paths = len(p) - 1
            elif method == "dijkstra":
                paths = nx.dijkstra_path_length(G, source, target, weight)
            else:  # method == 'bellman-ford':
                paths = nx.bellman_ford_path_length(G, source, target, weight)
    return paths


@nx._dispatchable(edge_attrs="weight")
def average_shortest_path_length(G, weight=None, method=None):
    r"""Returns the average shortest path length.

    The average shortest path length is

    .. math::

       a =\sum_{\substack{s,t \in V \\ s\neq t}} \frac{d(s, t)}{n(n-1)}

    where `V` is the set of nodes in `G`,
    `d(s, t)` is the shortest path from `s` to `t`,
    and `n` is the number of nodes in `G`.

    .. versionchanged:: 3.0
       An exception is raised for directed graphs that are not strongly
       connected.

    Parameters
    ----------
    G : NetworkX graph

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'unweighted' or 'dijkstra')
        The algorithm to use to compute the path lengths.
        Supported options are 'unweighted', 'dijkstra', 'bellman-ford',
        'floyd-warshall' and 'floyd-warshall-numpy'.
        Other method values produce a ValueError.
        The default method is 'unweighted' if `weight` is None,
        otherwise the default method is 'dijkstra'.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is the null graph (that is, the graph on zero nodes).

    NetworkXError
        If `G` is not connected (or not strongly connected, in the case
        of a directed graph).

    ValueError
        If `method` is not among the supported options.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.average_shortest_path_length(G)
    2.0

    For disconnected graphs, you can compute the average shortest path
    length for each component

    >>> G = nx.Graph([(1, 2), (3, 4)])
    >>> for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
    ...     print(nx.average_shortest_path_length(C))
    1.0
    1.0

    """
    single_source_methods = ["unweighted", "dijkstra", "bellman-ford"]
    all_pairs_methods = ["floyd-warshall", "floyd-warshall-numpy"]
    supported_methods = single_source_methods + all_pairs_methods

    if method is None:
        method = "unweighted" if weight is None else "dijkstra"
    if method not in supported_methods:
        raise ValueError(f"method not supported: {method}")

    n = len(G)
    # For the special case of the null graph, raise an exception, since
    # there are no paths in the null graph.
    if n == 0:
        msg = (
            "the null graph has no paths, thus there is no average "
            "shortest path length"
        )
        raise nx.NetworkXPointlessConcept(msg)
    # For the special case of the trivial graph, return zero immediately.
    if n == 1:
        return 0
    # Shortest path length is undefined if the graph is not strongly connected.
    if G.is_directed() and not nx.is_strongly_connected(G):
        raise nx.NetworkXError("Graph is not strongly connected.")
    # Shortest path length is undefined if the graph is not connected.
    if not G.is_directed() and not nx.is_connected(G):
        raise nx.NetworkXError("Graph is not connected.")

    # Compute all-pairs shortest paths.
    def path_length(v):
        if method == "unweighted":
            return nx.single_source_shortest_path_length(G, v)
        elif method == "dijkstra":
            return nx.single_source_dijkstra_path_length(G, v, weight=weight)
        elif method == "bellman-ford":
            return nx.single_source_bellman_ford_path_length(G, v, weight=weight)

    if method in single_source_methods:
        # Sum the distances for each (ordered) pair of source and target node.
        s = sum(l for u in G for l in path_length(u).values())
    else:
        if method == "floyd-warshall":
            all_pairs = nx.floyd_warshall(G, weight=weight)
            s = sum(sum(t.values()) for t in all_pairs.values())
        elif method == "floyd-warshall-numpy":
            s = float(nx.floyd_warshall_numpy(G, weight=weight).sum())
    return s / (n * (n - 1))


@nx._dispatchable(edge_attrs="weight")
def all_shortest_paths(G, source, target, weight=None, method="dijkstra"):
    """Compute all shortest simple paths in the graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path.

    target : node
       Ending node for path.

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'dijkstra')
       The algorithm to use to compute the path lengths.
       Supported options: 'dijkstra', 'bellman-ford'.
       Other inputs produce a ValueError.
       If `weight` is None, unweighted graph methods are used, and this
       suggestion is ignored.

    Returns
    -------
    paths : generator of lists
        A generator of all paths between source and target.

    Raises
    ------
    ValueError
        If `method` is not among the supported options.

    NetworkXNoPath
        If `target` cannot be reached from `source`.

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_path(G, [0, 1, 2])
    >>> nx.add_path(G, [0, 10, 2])
    >>> print([p for p in nx.all_shortest_paths(G, source=0, target=2)])
    [[0, 1, 2], [0, 10, 2]]

    Notes
    -----
    There may be many shortest paths between the source and target.  If G
    contains zero-weight cycles, this function will not produce all shortest
    paths because doing so would produce infinitely many paths of unbounded
    length -- instead, we only produce the shortest simple paths.

    See Also
    --------
    shortest_path
    single_source_shortest_path
    all_pairs_shortest_path
    """
    method = "unweighted" if weight is None else method
    if method == "unweighted":
        pred = nx.predecessor(G, source)
    elif method == "dijkstra":
        pred, dist = nx.dijkstra_predecessor_and_distance(G, source, weight=weight)
    elif method == "bellman-ford":
        pred, dist = nx.bellman_ford_predecessor_and_distance(G, source, weight=weight)
    else:
        raise ValueError(f"method not supported: {method}")

    return _build_paths_from_predecessors({source}, target, pred)


@nx._dispatchable(edge_attrs="weight")
def single_source_all_shortest_paths(G, source, weight=None, method="dijkstra"):
    """Compute all shortest simple paths from the given source in the graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path.

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'dijkstra')
       The algorithm to use to compute the path lengths.
       Supported options: 'dijkstra', 'bellman-ford'.
       Other inputs produce a ValueError.
       If `weight` is None, unweighted graph methods are used, and this
       suggestion is ignored.

    Returns
    -------
    paths : generator of dictionary
        A generator of all paths between source and all nodes in the graph.

    Raises
    ------
    ValueError
        If `method` is not among the supported options.

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_path(G, [0, 1, 2, 3, 0])
    >>> dict(nx.single_source_all_shortest_paths(G, source=0))
    {0: [[0]], 1: [[0, 1]], 2: [[0, 1, 2], [0, 3, 2]], 3: [[0, 3]]}

    Notes
    -----
    There may be many shortest paths between the source and target.  If G
    contains zero-weight cycles, this function will not produce all shortest
    paths because doing so would produce infinitely many paths of unbounded
    length -- instead, we only produce the shortest simple paths.

    See Also
    --------
    shortest_path
    all_shortest_paths
    single_source_shortest_path
    all_pairs_shortest_path
    all_pairs_all_shortest_paths
    """
    method = "unweighted" if weight is None else method
    if method == "unweighted":
        pred = nx.predecessor(G, source)
    elif method == "dijkstra":
        pred, dist = nx.dijkstra_predecessor_and_distance(G, source, weight=weight)
    elif method == "bellman-ford":
        pred, dist = nx.bellman_ford_predecessor_and_distance(G, source, weight=weight)
    else:
        raise ValueError(f"method not supported: {method}")
    for n in G:
        try:
            yield n, list(_build_paths_from_predecessors({source}, n, pred))
        except nx.NetworkXNoPath:
            pass


@nx._dispatchable(edge_attrs="weight")
def all_pairs_all_shortest_paths(G, weight=None, method="dijkstra"):
    """Compute all shortest paths between all nodes.

    Parameters
    ----------
    G : NetworkX graph

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'dijkstra')
       The algorithm to use to compute the path lengths.
       Supported options: 'dijkstra', 'bellman-ford'.
       Other inputs produce a ValueError.
       If `weight` is None, unweighted graph methods are used, and this
       suggestion is ignored.

    Returns
    -------
    paths : generator of dictionary
        Dictionary of arrays, keyed by source and target, of all shortest paths.

    Raises
    ------
    ValueError
        If `method` is not among the supported options.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> dict(nx.all_pairs_all_shortest_paths(G))[0][2]
    [[0, 1, 2], [0, 3, 2]]
    >>> dict(nx.all_pairs_all_shortest_paths(G))[0][3]
    [[0, 3]]

    Notes
    -----
    There may be multiple shortest paths with equal lengths. Unlike
    all_pairs_shortest_path, this method returns all shortest paths.

    See Also
    --------
    all_pairs_shortest_path
    single_source_all_shortest_paths
    """
    for n in G:
        yield (
            n,
            dict(single_source_all_shortest_paths(G, n, weight=weight, method=method)),
        )


def _build_paths_from_predecessors(sources, target, pred):
    """Compute all simple paths to target, given the predecessors found in
    pred, terminating when any source in sources is found.

    Parameters
    ----------
    sources : set
       Starting nodes for path.

    target : node
       Ending node for path.

    pred : dict
       A dictionary of predecessor lists, keyed by node

    Returns
    -------
    paths : generator of lists
        A generator of all paths between source and target.

    Raises
    ------
    NetworkXNoPath
        If `target` cannot be reached from `source`.

    Notes
    -----
    There may be many paths between the sources and target.  If there are
    cycles among the predecessors, this function will not produce all
    possible paths because doing so would produce infinitely many paths
    of unbounded length -- instead, we only produce simple paths.

    See Also
    --------
    shortest_path
    single_source_shortest_path
    all_pairs_shortest_path
    all_shortest_paths
    bellman_ford_path
    """
    if target not in pred:
        raise nx.NetworkXNoPath(f"Target {target} cannot be reached from given sources")

    seen = {target}
    stack = [[target, 0]]
    top = 0
    while top >= 0:
        node, i = stack[top]
        if node in sources:
            yield [p for p, n in reversed(stack[: top + 1])]
        if len(pred[node]) > i:
            stack[top][1] = i + 1
            next = pred[node][i]
            if next in seen:
                continue
            else:
                seen.add(next)
            top += 1
            if top == len(stack):
                stack.append([next, 0])
            else:
                stack[top][:] = [next, 0]
        else:
            seen.discard(node)
            top -= 1
<|endoftext|>"""
Shortest path algorithms for unweighted graphs.
"""

import warnings

import networkx as nx

__all__ = [
    "bidirectional_shortest_path",
    "single_source_shortest_path",
    "single_source_shortest_path_length",
    "single_target_shortest_path",
    "single_target_shortest_path_length",
    "all_pairs_shortest_path",
    "all_pairs_shortest_path_length",
    "predecessor",
]


@nx._dispatchable
def single_source_shortest_path_length(G, source, cutoff=None):
    """Compute the shortest path lengths from source to all reachable nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    lengths : dict
        Dict keyed by node to shortest path length to source.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = nx.single_source_shortest_path_length(G, 0)
    >>> length[4]
    4
    >>> for node in length:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 3
    4: 4

    See Also
    --------
    shortest_path_length
    """
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} is not in G")
    if cutoff is None:
        cutoff = float("inf")
    nextlevel = [source]
    return dict(_single_shortest_path_length(G._adj, nextlevel, cutoff))


def _single_shortest_path_length(adj, firstlevel, cutoff):
    """Yields (node, level) in a breadth first search

    Shortest Path Length helper function
    Parameters
    ----------
        adj : dict
            Adjacency dict or view
        firstlevel : list
            starting nodes, e.g. [source] or [target]
        cutoff : int or float
            level at which we stop the process
    """
    seen = set(firstlevel)
    nextlevel = firstlevel
    level = 0
    n = len(adj)
    for v in nextlevel:
        yield (v, level)
    while nextlevel and cutoff > level:
        level += 1
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
                    yield (w, level)
            if len(seen) == n:
                return


@nx._dispatchable
def single_target_shortest_path_length(G, target, cutoff=None):
    """Compute the shortest path lengths to target from all reachable nodes.

    Parameters
    ----------
    G : NetworkX graph

    target : node
       Target node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    lengths : iterator
        (source, shortest path length) iterator

    Examples
    --------
    >>> G = nx.path_graph(5, create_using=nx.DiGraph())
    >>> length = dict(nx.single_target_shortest_path_length(G, 4))
    >>> length[0]
    4
    >>> for node in range(5):
    ...     print(f"{node}: {length[node]}")
    0: 4
    1: 3
    2: 2
    3: 1
    4: 0

    See Also
    --------
    single_source_shortest_path_length, shortest_path_length
    """
    if target not in G:
        raise nx.NodeNotFound(f"Target {target} is not in G")

    warnings.warn(
        (
            "\n\nsingle_target_shortest_path_length will return a dict instead of"
            "\nan iterator in version 3.5"
        ),
        FutureWarning,
        stacklevel=3,
    )

    if cutoff is None:
        cutoff = float("inf")
    # handle either directed or undirected
    adj = G._pred if G.is_directed() else G._adj
    nextlevel = [target]
    # for version 3.3 we will return a dict like this:
    # return dict(_single_shortest_path_length(adj, nextlevel, cutoff))
    return _single_shortest_path_length(adj, nextlevel, cutoff)


@nx._dispatchable
def all_pairs_shortest_path_length(G, cutoff=None):
    """Computes the shortest path lengths between all nodes in `G`.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer, optional
        Depth at which to stop the search. Only paths of length at most
        `cutoff` are returned.

    Returns
    -------
    lengths : iterator
        (source, dictionary) iterator with dictionary keyed by target and
        shortest path length as the key value.

    Notes
    -----
    The iterator returned only has reachable node pairs.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = dict(nx.all_pairs_shortest_path_length(G))
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"1 - {node}: {length[1][node]}")
    1 - 0: 1
    1 - 1: 0
    1 - 2: 1
    1 - 3: 2
    1 - 4: 3
    >>> length[3][2]
    1
    >>> length[2][2]
    0

    """
    length = single_source_shortest_path_length
    # TODO This can be trivially parallelized.
    for n in G:
        yield (n, length(G, n, cutoff=cutoff))


@nx._dispatchable
def bidirectional_shortest_path(G, source, target):
    """Returns a list of nodes in a shortest path between source and target.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       starting node for path

    target : node label
       ending node for path

    Returns
    -------
    path: list
       List of nodes in a path from source to target.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_path(G, [0, 1, 2, 3, 0, 4, 5, 6, 7, 4])
    >>> nx.bidirectional_shortest_path(G, 2, 6)
    [2, 1, 0, 4, 5, 6]

    See Also
    --------
    shortest_path

    Notes
    -----
    This algorithm is used by shortest_path(G, source, target).
    """

    if source not in G:
        raise nx.NodeNotFound(f"Source {source} is not in G")

    if target not in G:
        raise nx.NodeNotFound(f"Target {target} is not in G")

    # call helper to do the real work
    results = _bidirectional_pred_succ(G, source, target)
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


def _bidirectional_pred_succ(G, source, target):
    """Bidirectional shortest path helper.

    Returns (pred, succ, w) where
    pred is a dictionary of predecessors from w to the source, and
    succ is a dictionary of successors from w to the target.
    """
    # does BFS from both source and target and meets in the middle
    if target == source:
        return ({target: None}, {source: None}, source)

    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.pred
        Gsucc = G.succ
    else:
        Gpred = G.adj
        Gsucc = G.adj

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
                for w in Gsucc[v]:
                    if w not in pred:
                        forward_fringe.append(w)
                        pred[w] = v
                    if w in succ:  # path found
                        return pred, succ, w
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in Gpred[v]:
                    if w not in succ:
                        succ[w] = v
                        reverse_fringe.append(w)
                    if w in pred:  # found path
                        return pred, succ, w

    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")


@nx._dispatchable
def single_source_shortest_path(G, source, cutoff=None):
    """Compute shortest path between source
    and all other nodes reachable from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    paths : dictionary
        Dictionary, keyed by target, of shortest paths.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = nx.single_source_shortest_path(G, 0)
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    The shortest path is not necessarily unique. So there can be multiple
    paths between the source and each target node, all of which have the
    same 'shortest' length. For each target node, this function returns
    only one of those paths.

    See Also
    --------
    shortest_path
    """
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not in G")

    def join(p1, p2):
        return p1 + p2

    if cutoff is None:
        cutoff = float("inf")
    nextlevel = {source: 1}  # list of nodes to check at next level
    paths = {source: [source]}  # paths dictionary  (paths to key from source)
    return dict(_single_shortest_path(G.adj, nextlevel, paths, cutoff, join))


def _single_shortest_path(adj, firstlevel, paths, cutoff, join):
    """Returns shortest paths

    Shortest Path helper function
    Parameters
    ----------
        adj : dict
            Adjacency dict or view
        firstlevel : dict
            starting nodes, e.g. {source: 1} or {target: 1}
        paths : dict
            paths for starting nodes, e.g. {source: [source]}
        cutoff : int or float
            level at which we stop the process
        join : function
            function to construct a path from two partial paths. Requires two
            list inputs `p1` and `p2`, and returns a list. Usually returns
            `p1 + p2` (forward from source) or `p2 + p1` (backward from target)
    """
    level = 0  # the current level
    nextlevel = firstlevel
    while nextlevel and cutoff > level:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in adj[v]:
                if w not in paths:
                    paths[w] = join(paths[v], [w])
                    nextlevel[w] = 1
        level += 1
    return paths


@nx._dispatchable
def single_target_shortest_path(G, target, cutoff=None):
    """Compute shortest path to target from all nodes that reach target.

    Parameters
    ----------
    G : NetworkX graph

    target : node label
       Target node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    paths : dictionary
        Dictionary, keyed by target, of shortest paths.

    Examples
    --------
    >>> G = nx.path_graph(5, create_using=nx.DiGraph())
    >>> path = nx.single_target_shortest_path(G, 4)
    >>> path[0]
    [0, 1, 2, 3, 4]

    Notes
    -----
    The shortest path is not necessarily unique. So there can be multiple
    paths between the source and each target node, all of which have the
    same 'shortest' length. For each target node, this function returns
    only one of those paths.

    See Also
    --------
    shortest_path, single_source_shortest_path
    """
    if target not in G:
        raise nx.NodeNotFound(f"Target {target} not in G")

    def join(p1, p2):
        return p2 + p1

    # handle undirected graphs
    adj = G.pred if G.is_directed() else G.adj
    if cutoff is None:
        cutoff = float("inf")
    nextlevel = {target: 1}  # list of nodes to check at next level
    paths = {target: [target]}  # paths dictionary  (paths to key from source)
    return dict(_single_shortest_path(adj, nextlevel, paths, cutoff, join))


@nx._dispatchable
def all_pairs_shortest_path(G, cutoff=None):
    """Compute shortest paths between all nodes.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer, optional
        Depth at which to stop the search. Only paths of length at most
        `cutoff` are returned.

    Returns
    -------
    paths : iterator
        Dictionary, keyed by source and target, of shortest paths.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = dict(nx.all_pairs_shortest_path(G))
    >>> print(path[0][4])
    [0, 1, 2, 3, 4]

    Notes
    -----
    There may be multiple shortest paths with the same length between
    two nodes. For each pair, this function returns only one of those paths.

    See Also
    --------
    floyd_warshall
    all_pairs_all_shortest_paths

    """
    # TODO This can be trivially parallelized.
    for n in G:
        yield (n, single_source_shortest_path(G, n, cutoff=cutoff))


@nx._dispatchable
def predecessor(G, source, target=None, cutoff=None, return_seen=None):
    """Returns dict of predecessors for the path from source to all nodes in G.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    target : node label, optional
       Ending node for path. If provided only predecessors between
       source and target are returned

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    return_seen : bool, optional (default=None)
        Whether to return a dictionary, keyed by node, of the level (number of
        hops) to reach the node (as seen during breadth-first-search).

    Returns
    -------
    pred : dictionary
        Dictionary, keyed by node, of predecessors in the shortest path.


    (pred, seen): tuple of dictionaries
        If `return_seen` argument is set to `True`, then a tuple of dictionaries
        is returned. The first element is the dictionary, keyed by node, of
        predecessors in the shortest path. The second element is the dictionary,
        keyed by node, of the level (number of hops) to reach the node (as seen
        during breadth-first-search).

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> list(G)
    [0, 1, 2, 3]
    >>> nx.predecessor(G, 0)
    {0: [], 1: [0], 2: [1], 3: [2]}
    >>> nx.predecessor(G, 0, return_seen=True)
    ({0: [], 1: [0], 2: [1], 3: [2]}, {0: 0, 1: 1, 2: 2, 3: 3})


    """
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not in G")

    level = 0  # the current level
    nextlevel = [source]  # list of nodes to check at next level
    seen = {source: level}  # level (number of hops) when seen in BFS
    pred = {source: []}  # predecessor dictionary
    while nextlevel:
        level = level + 1
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in G[v]:
                if w not in seen:
                    pred[w] = [v]
                    seen[w] = level
                    nextlevel.append(w)
                elif seen[w] == level:  # add v to predecessor list if it
                    pred[w].append(v)  # is at the correct level
        if cutoff and cutoff <= level:
            break

    if target is not None:
        if return_seen:
            if target not in pred:
                return ([], -1)  # No predecessor
            return (pred[target], seen[target])
        else:
            if target not in pred:
                return []  # No predecessor
            return pred[target]
    else:
        if return_seen:
            return (pred, seen)
        else:
            return pred
<|endoftext|>"""Floyd-Warshall algorithm for shortest paths."""

import networkx as nx

__all__ = [
    "floyd_warshall",
    "floyd_warshall_predecessor_and_distance",
    "reconstruct_path",
    "floyd_warshall_numpy",
]


@nx._dispatchable(edge_attrs="weight")
def floyd_warshall_numpy(G, nodelist=None, weight="weight"):
    """Find all-pairs shortest path lengths using Floyd's algorithm.

    This algorithm for finding shortest paths takes advantage of
    matrix representations of a graph and works well for dense
    graphs where all-pairs shortest path lengths are desired.
    The results are returned as a NumPy array, distance[i, j],
    where i and j are the indexes of two nodes in nodelist.
    The entry distance[i, j] is the distance along a shortest
    path from i to j. If no path exists the distance is Inf.

    Parameters
    ----------
    G : NetworkX graph

    nodelist : list, optional (default=G.nodes)
       The rows and columns are ordered by the nodes in nodelist.
       If nodelist is None then the ordering is produced by G.nodes.
       Nodelist should include all nodes in G.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.

    Returns
    -------
    distance : 2D numpy.ndarray
        A numpy array of shortest path distances between nodes.
        If there is no path between two nodes the value is Inf.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     [(0, 1, 5), (1, 2, 2), (2, 3, -3), (1, 3, 10), (3, 2, 8)]
    ... )
    >>> nx.floyd_warshall_numpy(G)
    array([[ 0.,  5.,  7.,  4.],
           [inf,  0.,  2., -1.],
           [inf, inf,  0., -3.],
           [inf, inf,  8.,  0.]])

    Notes
    -----
    Floyd's algorithm is appropriate for finding shortest paths in
    dense graphs or graphs with negative weights when Dijkstra's
    algorithm fails. This algorithm can still fail if there are negative
    cycles. It has running time $O(n^3)$ with running space of $O(n^2)$.

    Raises
    ------
    NetworkXError
        If nodelist is not a list of the nodes in G.
    """
    import numpy as np

    if nodelist is not None:
        if not (len(nodelist) == len(G) == len(set(nodelist))):
            raise nx.NetworkXError(
                "nodelist must contain every node in G with no repeats."
                "If you wanted a subgraph of G use G.subgraph(nodelist)"
            )

    # To handle cases when an edge has weight=0, we must make sure that
    # nonedges are not given the value 0 as well.
    A = nx.to_numpy_array(
        G, nodelist, multigraph_weight=min, weight=weight, nonedge=np.inf
    )
    n, m = A.shape
    np.fill_diagonal(A, 0)  # diagonal elements should be zero
    for i in range(n):
        # The second term has the same shape as A due to broadcasting
        A = np.minimum(A, A[i, :][np.newaxis, :] + A[:, i][:, np.newaxis])
    return A


@nx._dispatchable(edge_attrs="weight")
def floyd_warshall_predecessor_and_distance(G, weight="weight"):
    """Find all-pairs shortest path lengths using Floyd's algorithm.

    Parameters
    ----------
    G : NetworkX graph

    weight: string, optional (default= 'weight')
       Edge data key corresponding to the edge weight.

    Returns
    -------
    predecessor,distance : dictionaries
       Dictionaries, keyed by source and target, of predecessors and distances
       in the shortest path.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     [
    ...         ("s", "u", 10),
    ...         ("s", "x", 5),
    ...         ("u", "v", 1),
    ...         ("u", "x", 2),
    ...         ("v", "y", 1),
    ...         ("x", "u", 3),
    ...         ("x", "v", 5),
    ...         ("x", "y", 2),
    ...         ("y", "s", 7),
    ...         ("y", "v", 6),
    ...     ]
    ... )
    >>> predecessors, _ = nx.floyd_warshall_predecessor_and_distance(G)
    >>> print(nx.reconstruct_path("s", "v", predecessors))
    ['s', 'x', 'u', 'v']

    Notes
    -----
    Floyd's algorithm is appropriate for finding shortest paths
    in dense graphs or graphs with negative weights when Dijkstra's algorithm
    fails.  This algorithm can still fail if there are negative cycles.
    It has running time $O(n^3)$ with running space of $O(n^2)$.

    See Also
    --------
    floyd_warshall
    floyd_warshall_numpy
    all_pairs_shortest_path
    all_pairs_shortest_path_length
    """
    from collections import defaultdict

    # dictionary-of-dictionaries representation for dist and pred
    # use some defaultdict magick here
    # for dist the default is the floating point inf value
    dist = defaultdict(lambda: defaultdict(lambda: float("inf")))
    for u in G:
        dist[u][u] = 0
    pred = defaultdict(dict)
    # initialize path distance dictionary to be the adjacency matrix
    # also set the distance to self to 0 (zero diagonal)
    undirected = not G.is_directed()
    for u, v, d in G.edges(data=True):
        e_weight = d.get(weight, 1.0)
        dist[u][v] = min(e_weight, dist[u][v])
        pred[u][v] = u
        if undirected:
            dist[v][u] = min(e_weight, dist[v][u])
            pred[v][u] = v
    for w in G:
        dist_w = dist[w]  # save recomputation
        for u in G:
            dist_u = dist[u]  # save recomputation
            for v in G:
                d = dist_u[w] + dist_w[v]
                if dist_u[v] > d:
                    dist_u[v] = d
                    pred[u][v] = pred[w][v]
    return dict(pred), dict(dist)


@nx._dispatchable(graphs=None)
def reconstruct_path(source, target, predecessors):
    """Reconstruct a path from source to target using the predecessors
    dict as returned by floyd_warshall_predecessor_and_distance

    Parameters
    ----------
    source : node
       Starting node for path

    target : node
       Ending node for path

    predecessors: dictionary
       Dictionary, keyed by source and target, of predecessors in the
       shortest path, as returned by floyd_warshall_predecessor_and_distance

    Returns
    -------
    path : list
       A list of nodes containing the shortest path from source to target

       If source and target are the same, an empty list is returned

    Notes
    -----
    This function is meant to give more applicability to the
    floyd_warshall_predecessor_and_distance function

    See Also
    --------
    floyd_warshall_predecessor_and_distance
    """
    if source == target:
        return []
    prev = predecessors[source]
    curr = prev[target]
    path = [target, curr]
    while curr != source:
        curr = prev[curr]
        path.append(curr)
    return list(reversed(path))


@nx._dispatchable(edge_attrs="weight")
def floyd_warshall(G, weight="weight"):
    """Find all-pairs shortest path lengths using Floyd's algorithm.

    Parameters
    ----------
    G : NetworkX graph

    weight: string, optional (default= 'weight')
       Edge data key corresponding to the edge weight.


    Returns
    -------
    distance : dict
       A dictionary,  keyed by source and target, of shortest paths distances
       between nodes.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     [(0, 1, 5), (1, 2, 2), (2, 3, -3), (1, 3, 10), (3, 2, 8)]
    ... )
    >>> fw = nx.floyd_warshall(G, weight="weight")
    >>> results = {a: dict(b) for a, b in fw.items()}
    >>> print(results)
    {0: {0: 0, 1: 5, 2: 7, 3: 4}, 1: {1: 0, 2: 2, 3: -1, 0: inf}, 2: {2: 0, 3: -3, 0: inf, 1: inf}, 3: {3: 0, 2: 8, 0: inf, 1: inf}}

    Notes
    -----
    Floyd's algorithm is appropriate for finding shortest paths
    in dense graphs or graphs with negative weights when Dijkstra's algorithm
    fails.  This algorithm can still fail if there are negative cycles.
    It has running time $O(n^3)$ with running space of $O(n^2)$.

    See Also
    --------
    floyd_warshall_predecessor_and_distance
    floyd_warshall_numpy
    all_pairs_shortest_path
    all_pairs_shortest_path_length
    """
    # could make this its own function to reduce memory costs
    return floyd_warshall_predecessor_and_distance(G, weight=weight)[1]
<|endoftext|>"""Shortest paths and path lengths using the A* ("A star") algorithm."""

from heapq import heappop, heappush
from itertools import count

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function

__all__ = ["astar_path", "astar_path_length"]


@nx._dispatchable(edge_attrs="weight", preserve_node_attrs="heuristic")
def astar_path(G, source, target, heuristic=None, weight="weight", *, cutoff=None):
    """Returns a list of nodes in a shortest path between source and target
    using the A* ("A-star") algorithm.

    There may be more than one shortest path.  This returns only one.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.
       If the heuristic is inadmissible (if it might
       overestimate the cost of reaching the goal from a node),
       the result may not be a shortest path.
       The algorithm does not support updating heuristic
       values for the same node due to caching the first
       heuristic calculation per node.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.
       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number or None to indicate a hidden edge.

    cutoff : float, optional
       If this is provided, the search will be bounded to this value. I.e. if
       the evaluation function surpasses this value for a node n, the node will not
       be expanded further and will be ignored. More formally, let h'(n) be the
       heuristic function, and g(n) be the cost of reaching n from the source node. Then,
       if g(n) + h'(n) > cutoff, the node will not be explored further.
       Note that if the heuristic is inadmissible, it is possible that paths
       are ignored even though they satisfy the cutoff.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> print(nx.astar_path(G, 0, 4))
    [0, 1, 2, 3, 4]
    >>> G = nx.grid_graph(dim=[3, 3])  # nodes are two-tuples (x,y)
    >>> nx.set_edge_attributes(G, {e: e[1][0] * 2 for e in G.edges()}, "cost")
    >>> def dist(a, b):
    ...     (x1, y1) = a
    ...     (x2, y2) = b
    ...     return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    >>> print(nx.astar_path(G, (0, 0), (2, 2), heuristic=dist, weight="cost"))
    [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    shortest_path, dijkstra_path

    """
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} is not in G")

    if target not in G:
        raise nx.NodeNotFound(f"Target {target} is not in G")

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in G_succ[curnode].items():
            cost = weight(curnode, neighbor, w)
            if cost is None:
                continue
            ncost = dist + cost
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)

            if cutoff and ncost + h > cutoff:
                continue

            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")


@nx._dispatchable(edge_attrs="weight", preserve_node_attrs="heuristic")
def astar_path_length(
    G, source, target, heuristic=None, weight="weight", *, cutoff=None
):
    """Returns the length of the shortest path between source and target using
    the A* ("A-star") algorithm.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.
       If the heuristic is inadmissible (if it might
       overestimate the cost of reaching the goal from a node),
       the result may not be a shortest path.
       The algorithm does not support updating heuristic
       values for the same node due to caching the first
       heuristic calculation per node.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.
       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number or None to indicate a hidden edge.

    cutoff : float, optional
       If this is provided, the search will be bounded to this value. I.e. if
       the evaluation function surpasses this value for a node n, the node will not
       be expanded further and will be ignored. More formally, let h'(n) be the
       heuristic function, and g(n) be the cost of reaching n from the source node. Then,
       if g(n) + h'(n) > cutoff, the node will not be explored further.
       Note that if the heuristic is inadmissible, it is possible that paths
       are ignored even though they satisfy the cutoff.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    See Also
    --------
    astar_path

    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    weight = _weight_function(G, weight)
    path = astar_path(G, source, target, heuristic, weight, cutoff=cutoff)
    return sum(weight(u, v, G[u][v]) for u, v in zip(path[:-1], path[1:]))
<|endoftext|>from networkx.algorithms.shortest_paths.generic import *
from networkx.algorithms.shortest_paths.unweighted import *
from networkx.algorithms.shortest_paths.weighted import *
from networkx.algorithms.shortest_paths.astar import *
from networkx.algorithms.shortest_paths.dense import *
<|endoftext|>"""
Shortest path algorithms for weighted graphs.
"""

from collections import deque
from heapq import heappop, heappush
from itertools import count

import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors

__all__ = [
    "dijkstra_path",
    "dijkstra_path_length",
    "bidirectional_dijkstra",
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
    "multi_source_dijkstra",
    "multi_source_dijkstra_path",
    "multi_source_dijkstra_path_length",
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "dijkstra_predecessor_and_distance",
    "bellman_ford_path",
    "bellman_ford_path_length",
    "single_source_bellman_ford",
    "single_source_bellman_ford_path",
    "single_source_bellman_ford_path_length",
    "all_pairs_bellman_ford_path",
    "all_pairs_bellman_ford_path_length",
    "bellman_ford_predecessor_and_distance",
    "negative_edge_cycle",
    "find_negative_cycle",
    "goldberg_radzik",
    "johnson",
]


def _weight_function(G, weight):
    """Returns a function that returns the weight of an edge.

    The returned function is specifically suitable for input to
    functions :func:`_dijkstra` and :func:`_bellman_ford_relaxation`.

    Parameters
    ----------
    G : NetworkX graph.

    weight : string or function
        If it is callable, `weight` itself is returned. If it is a string,
        it is assumed to be the name of the edge attribute that represents
        the weight of an edge. In that case, a function is returned that
        gets the edge weight according to the specified edge attribute.

    Returns
    -------
    function
        This function returns a callable that accepts exactly three inputs:
        a node, an node adjacent to the first one, and the edge attribute
        dictionary for the eedge joining those nodes. That function returns
        a number representing the weight of an edge.

    If `G` is a multigraph, and `weight` is not callable, the
    minimum edge weight over all parallel edges is returned. If any edge
    does not have an attribute with key `weight`, it is assumed to
    have weight one.

    """
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


@nx._dispatchable(edge_attrs="weight")
def dijkstra_path(G, source, target, weight="weight"):
    """Returns the shortest weighted path from source to target in G.

    Uses Dijkstra's Method to compute the shortest weighted path
    between two nodes in a graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node

    target : node
        Ending node

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    path : list
        List of nodes in a shortest path.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> print(nx.dijkstra_path(G, 0, 4))
    [0, 1, 2, 3, 4]

    Find edges of shortest path in Multigraph

    >>> G = nx.MultiDiGraph()
    >>> G.add_weighted_edges_from([(1, 2, 0.75), (1, 2, 0.5), (2, 3, 0.5), (1, 3, 1.5)])
    >>> nodes = nx.dijkstra_path(G, 1, 3)
    >>> edges = nx.utils.pairwise(nodes)
    >>> list(
    ...     (u, v, min(G[u][v], key=lambda k: G[u][v][k].get("weight", 1)))
    ...     for u, v in edges
    ... )
    [(1, 2, 1), (2, 3, 0)]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    The weight function can be used to include node weights.

    >>> def func(u, v, d):
    ...     node_u_wt = G.nodes[u].get("node_weight", 1)
    ...     node_v_wt = G.nodes[v].get("node_weight", 1)
    ...     edge_wt = d.get("weight", 1)
    ...     return node_u_wt / 2 + node_v_wt / 2 + edge_wt

    In this example we take the average of start and end node
    weights of an edge and add it to the weight of the edge.

    The function :func:`single_source_dijkstra` computes both
    path and length-of-path if you need both, use that.

    See Also
    --------
    bidirectional_dijkstra
    bellman_ford_path
    single_source_dijkstra
    """
    (length, path) = single_source_dijkstra(G, source, target=target, weight=weight)
    return path


@nx._dispatchable(edge_attrs="weight")
def dijkstra_path_length(G, source, target, weight="weight"):
    """Returns the shortest weighted path length in G from source to target.

    Uses Dijkstra's Method to compute the shortest weighted path length
    between two nodes in a graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        starting node for path

    target : node label
        ending node for path

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    length : number
        Shortest path length.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.dijkstra_path_length(G, 0, 4)
    4

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    The function :func:`single_source_dijkstra` computes both
    path and length-of-path if you need both, use that.

    See Also
    --------
    bidirectional_dijkstra
    bellman_ford_path_length
    single_source_dijkstra

    """
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} not found in graph")
    if source == target:
        return 0
    weight = _weight_function(G, weight)
    length = _dijkstra(G, source, weight, target=target)
    try:
        return length[target]
    except KeyError as err:
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}") from err


@nx._dispatchable(edge_attrs="weight")
def single_source_dijkstra_path(G, source, cutoff=None, weight="weight"):
    """Find shortest weighted paths in G from a source node.

    Compute shortest path between source and all other reachable
    nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for path.

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    paths : dictionary
        Dictionary of shortest path lengths keyed by target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = nx.single_source_dijkstra_path(G, 0)
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    single_source_dijkstra, single_source_bellman_ford

    """
    return multi_source_dijkstra_path(G, {source}, cutoff=cutoff, weight=weight)


@nx._dispatchable(edge_attrs="weight")
def single_source_dijkstra_path_length(G, source, cutoff=None, weight="weight"):
    """Find shortest weighted path lengths in G from a source node.

    Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        Starting node for path

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    length : dict
        Dict keyed by node to shortest path length from source.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = nx.single_source_dijkstra_path_length(G, 0)
    >>> length[4]
    4
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 3
    4: 4

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    single_source_dijkstra, single_source_bellman_ford_path_length

    """
    return multi_source_dijkstra_path_length(G, {source}, cutoff=cutoff, weight=weight)


@nx._dispatchable(edge_attrs="weight")
def single_source_dijkstra(G, source, target=None, cutoff=None, weight="weight"):
    """Find shortest weighted paths and lengths from a source node.

    Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Uses Dijkstra's algorithm to compute shortest paths and lengths
    between a source and all other reachable nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        Starting node for path

    target : node label, optional
        Ending node for path

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.


    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    distance, path : pair of dictionaries, or numeric and list.
        If target is None, paths and lengths to all nodes are computed.
        The return value is a tuple of two dictionaries keyed by target nodes.
        The first dictionary stores distance to each target node.
        The second stores the path to each target node.
        If target is not None, returns a tuple (distance, path), where
        distance is the distance from source to target and path is a list
        representing the path from source to target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length, path = nx.single_source_dijkstra(G, 0)
    >>> length[4]
    4
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 3
    4: 4
    >>> path[4]
    [0, 1, 2, 3, 4]
    >>> length, path = nx.single_source_dijkstra(G, 0, 1)
    >>> length
    1
    >>> path
    [0, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Based on the Python cookbook recipe (119466) at
    https://code.activestate.com/recipes/119466/

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    single_source_dijkstra_path
    single_source_dijkstra_path_length
    single_source_bellman_ford
    """
    return multi_source_dijkstra(
        G, {source}, cutoff=cutoff, target=target, weight=weight
    )


@nx._dispatchable(edge_attrs="weight")
def multi_source_dijkstra_path(G, sources, cutoff=None, weight="weight"):
    """Find shortest weighted paths in G from a given set of source
    nodes.

    Compute shortest path between any of the source nodes and all other
    reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    paths : dictionary
        Dictionary of shortest paths keyed by target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = nx.multi_source_dijkstra_path(G, {0, 4})
    >>> path[1]
    [0, 1]
    >>> path[3]
    [4, 3]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Raises
    ------
    ValueError
        If `sources` is empty.
    NodeNotFound
        If any of `sources` is not in `G`.

    See Also
    --------
    multi_source_dijkstra, multi_source_bellman_ford

    """
    length, path = multi_source_dijkstra(G, sources, cutoff=cutoff, weight=weight)
    return path


@nx._dispatchable(edge_attrs="weight")
def multi_source_dijkstra_path_length(G, sources, cutoff=None, weight="weight"):
    """Find shortest weighted path lengths in G from a given set of
    source nodes.

    Compute the shortest path length between any of the source nodes and
    all other reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    length : dict
        Dict keyed by node to shortest path length to nearest source.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = nx.multi_source_dijkstra_path_length(G, {0, 4})
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 1
    4: 0

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Raises
    ------
    ValueError
        If `sources` is empty.
    NodeNotFound
        If any of `sources` is not in `G`.

    See Also
    --------
    multi_source_dijkstra

    """
    if not sources:
        raise ValueError("sources must not be empty")
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f"Node {s} not found in graph")
    weight = _weight_function(G, weight)
    return _dijkstra_multisource(G, sources, weight, cutoff=cutoff)


@nx._dispatchable(edge_attrs="weight")
def multi_source_dijkstra(G, sources, target=None, cutoff=None, weight="weight"):
    """Find shortest weighted paths and lengths from a given set of
    source nodes.

    Uses Dijkstra's algorithm to compute the shortest paths and lengths
    between one of the source nodes and the given `target`, or all other
    reachable nodes if not specified, for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    target : node label, optional
        Ending node for path

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    distance, path : pair of dictionaries, or numeric and list
        If target is None, returns a tuple of two dictionaries keyed by node.
        The first dictionary stores distance from one of the source nodes.
        The second stores the path from one of the sources to that node.
        If target is not None, returns a tuple of (distance, path) where
        distance is the distance from source to target and path is a list
        representing the path from source to target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length, path = nx.multi_source_dijkstra(G, {0, 4})
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 1
    4: 0
    >>> path[1]
    [0, 1]
    >>> path[3]
    [4, 3]

    >>> length, path = nx.multi_source_dijkstra(G, {0, 4}, 1)
    >>> length
    1
    >>> path
    [0, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Based on the Python cookbook recipe (119466) at
    https://code.activestate.com/recipes/119466/

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    Raises
    ------
    ValueError
        If `sources` is empty.
    NodeNotFound
        If any of `sources` is not in `G`.

    See Also
    --------
    multi_source_dijkstra_path
    multi_source_dijkstra_path_length

    """
    if not sources:
        raise ValueError("sources must not be empty")
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f"Node {s} not found in graph")
    if target in sources:
        return (0, [target])
    weight = _weight_function(G, weight)
    paths = {source: [source] for source in sources}  # dictionary of paths
    dist = _dijkstra_multisource(
        G, sources, weight, paths=paths, cutoff=cutoff, target=target
    )
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError as err:
        raise nx.NetworkXNoPath(f"No path to {target}.") from err


def _dijkstra(G, source, weight, pred=None, paths=None, cutoff=None, target=None):
    """Uses Dijkstra's algorithm to find shortest weighted paths from a
    single source.

    This is a convenience function for :func:`_dijkstra_multisource`
    with all the arguments the same, except the keyword argument
    `sources` set to ``[source]``.

    """
    return _dijkstra_multisource(
        G, [source], weight, pred=pred, paths=paths, cutoff=cutoff, target=target
    )


def _dijkstra_multisource(
    G, sources, weight, pred=None, paths=None, cutoff=None, target=None
):
    """Uses Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty iterable of nodes
        Starting nodes for paths. If this is just an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in this
        iterable, the computed paths may begin from any one of the start
        nodes.

    weight: function
        Function with (u, v, data) input that returns that edge's weight
        or None to indicate a hidden edge

    pred: dict of lists, optional(default=None)
        dict to store a list of predecessors keyed by that node
        If None, predecessors are not stored.

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node.
        If None, paths are not stored.

    target : node label, optional
        Ending node for path. Search is halted when target is found.

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    Returns
    -------
    distance : dictionary
        A mapping from node to shortest distance to that node from one
        of the source nodes.

    Raises
    ------
    NodeNotFound
        If any of `sources` is not in `G`.

    Notes
    -----
    The optional predecessor and path dictionaries can be accessed by
    the caller through the original pred and paths objects passed
    as arguments. No need to explicitly return pred or paths.

    """
    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    for source in sources:
        seen[source] = 0
        push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():
            cost = weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] + cost
            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist:
                u_dist = dist[u]
                if vu_dist < u_dist:
                    raise ValueError("Contradictory paths found:", "negative weights?")
                elif pred is not None and vu_dist == u_dist:
                    pred[u].append(v)
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    return dist


@nx._dispatchable(edge_attrs="weight")
def dijkstra_predecessor_and_distance(G, source, cutoff=None, weight="weight"):
    """Compute weighted shortest path length and predecessors.

    Uses Dijkstra's Method to obtain the shortest weighted paths
    and return dictionaries of predecessors for each node and
    distance for each node from the `source`.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        Starting node for path

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    pred, distance : dictionaries
        Returns two dictionaries representing a list of predecessors
        of a node and the distance to each node.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The list of predecessors contains more than one element only when
    there are more than one shortest paths to the key node.

    Examples
    --------
    >>> G = nx.path_graph(5, create_using=nx.DiGraph())
    >>> pred, dist = nx.dijkstra_predecessor_and_distance(G, 0)
    >>> sorted(pred.items())
    [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]
    >>> sorted(dist.items())
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    >>> pred, dist = nx.dijkstra_predecessor_and_distance(G, 0, 1)
    >>> sorted(pred.items())
    [(0, []), (1, [0])]
    >>> sorted(dist.items())
    [(0, 0), (1, 1)]
    """
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} is not found in the graph")
    weight = _weight_function(G, weight)
    pred = {source: []}  # dictionary of predecessors
    return (pred, _dijkstra(G, source, weight, pred=pred, cutoff=cutoff))


@nx._dispatchable(edge_attrs="weight")
def all_pairs_dijkstra(G, cutoff=None, weight="weight"):
    """Find shortest weighted paths and lengths between all nodes.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edge[u][v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Yields
    ------
    (node, (distance, path)) : (node obj, (dict, dict))
        Each source node has two associated dicts. The first holds distance
        keyed by target and the second holds paths keyed by target.
        (See single_source_dijkstra for the source/target node terminology.)
        If desired you can apply `dict()` to this function to create a dict
        keyed by source node to the two dicts.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> len_path = dict(nx.all_pairs_dijkstra(G))
    >>> len_path[3][0][1]
    2
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"3 - {node}: {len_path[3][0][node]}")
    3 - 0: 3
    3 - 1: 2
    3 - 2: 1
    3 - 3: 0
    3 - 4: 1
    >>> len_path[3][1][1]
    [3, 2, 1]
    >>> for n, (dist, path) in nx.all_pairs_dijkstra(G):
    ...     print(path[1])
    [0, 1]
    [1]
    [2, 1]
    [3, 2, 1]
    [4, 3, 2, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The yielded dicts only have keys for reachable nodes.
    """
    for n in G:
        dist, path = single_source_dijkstra(G, n, cutoff=cutoff, weight=weight)
        yield (n, (dist, path))


@nx._dispatchable(edge_attrs="weight")
def all_pairs_dijkstra_path_length(G, cutoff=None, weight="weight"):
    """Compute shortest path lengths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    distance : iterator
        (source, dictionary) iterator with dictionary keyed by target and
        shortest path length as the key value.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = dict(nx.all_pairs_dijkstra_path_length(G))
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"1 - {node}: {length[1][node]}")
    1 - 0: 1
    1 - 1: 0
    1 - 2: 1
    1 - 3: 2
    1 - 4: 3
    >>> length[3][2]
    1
    >>> length[2][2]
    0

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The dictionary returned only has keys for reachable node pairs.
    """
    length = single_source_dijkstra_path_length
    for n in G:
        yield (n, length(G, n, cutoff=cutoff, weight=weight))


@nx._dispatchable(edge_attrs="weight")
def all_pairs_dijkstra_path(G, cutoff=None, weight="weight"):
    """Compute shortest paths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    paths : iterator
        (source, dictionary) iterator with dictionary keyed by target and
        shortest path as the key value.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = dict(nx.all_pairs_dijkstra_path(G))
    >>> path[0][4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    floyd_warshall, all_pairs_bellman_ford_path

    """
    path = single_source_dijkstra_path
    # TODO This can be trivially parallelized.
    for n in G:
        yield (n, path(G, n, cutoff=cutoff, weight=weight))


@nx._dispatchable(edge_attrs="weight")
def bellman_ford_predecessor_and_distance(
    G, source, target=None, weight="weight", heuristic=False
):
    """Compute shortest path lengths and predecessors on shortest paths
    in weighted graphs.

    The algorithm has a running time of $O(mn)$ where $n$ is the number of
    nodes and $m$ is the number of edges.  It is slower than Dijkstra but
    can handle negative edge weights.

    If a negative cycle is detected, you can use :func:`find_negative_cycle`
    to return the cycle and examine it. Shortest paths are not defined when
    a negative cycle exists because once reached, the path can cycle forever
    to build up arbitrarily low weights.

    Parameters
    ----------
    G : NetworkX graph
        The algorithm works for all types of graphs, including directed
        graphs and multigraphs.

    source: node label
        Starting node for path

    target : node label, optional
        Ending node for path

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    heuristic : bool
        Determines whether to use a heuristic to early detect negative
        cycles at a hopefully negligible cost.

    Returns
    -------
    pred, dist : dictionaries
        Returns two dictionaries keyed by node to predecessor in the
        path and to the distance from the source respectively.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXUnbounded
        If the (di)graph contains a negative (di)cycle, the
        algorithm raises an exception to indicate the presence of the
        negative (di)cycle.  Note: any negative weight edge in an
        undirected graph is a negative cycle.

    Examples
    --------
    >>> G = nx.path_graph(5, create_using=nx.DiGraph())
    >>> pred, dist = nx.bellman_ford_predecessor_and_distance(G, 0)
    >>> sorted(pred.items())
    [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]
    >>> sorted(dist.items())
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    >>> pred, dist = nx.bellman_ford_predecessor_and_distance(G, 0, 1)
    >>> sorted(pred.items())
    [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]
    >>> sorted(dist.items())
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    >>> G = nx.cycle_graph(5, create_using=nx.DiGraph())
    >>> G[1][2]["weight"] = -7
    >>> nx.bellman_ford_predecessor_and_distance(G, 0)
    Traceback (most recent call last):
        ...
    networkx.exception.NetworkXUnbounded: Negative cycle detected.

    See Also
    --------
    find_negative_cycle

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The dictionaries returned only have keys for nodes reachable from
    the source.

    In the case where the (di)graph is not connected, if a component
    not containing the source contains a negative (di)cycle, it
    will not be detected.

    In NetworkX v2.1 and prior, the source node had predecessor `[None]`.
    In NetworkX v2.2 this changed to the source node having predecessor `[]`
    """
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} is not found in the graph")
    weight = _weight_function(G, weight)
    if G.is_multigraph():
        if any(
            weight(u, v, {k: d}) < 0
            for u, v, k, d in nx.selfloop_edges(G, keys=True, data=True)
        ):
            raise nx.NetworkXUnbounded("Negative cycle detected.")
    else:
        if any(weight(u, v, d) < 0 for u, v, d in nx.selfloop_edges(G, data=True)):
            raise nx.NetworkXUnbounded("Negative cycle detected.")

    dist = {source: 0}
    pred = {source: []}

    if len(G) == 1:
        return pred, dist

    weight = _weight_function(G, weight)

    dist = _bellman_ford(
        G, [source], weight, pred=pred, dist=dist, target=target, heuristic=heuristic
    )
    return (pred, dist)


def _bellman_ford(
    G,
    source,
    weight,
    pred=None,
    paths=None,
    dist=None,
    target=None,
    heuristic=True,
):
    """Calls relaxation loop for Bellman–Ford algorithm and builds paths

    This is an implementation of the SPFA variant.
    See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

    Parameters
    ----------
    G : NetworkX graph

    source: list
        List of source nodes. The shortest path from any of the source
        nodes will be found if multiple sources are provided.

    weight : function
        The weight of an edge is the value returned by the function. The
        function must accept exactly three positional arguments: the two
        endpoints of an edge and the dictionary of edge attributes for
        that edge. The function must return a number.

    pred: dict of lists, optional (default=None)
        dict to store a list of predecessors keyed by that node
        If None, predecessors are not stored

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node
        If None, paths are not stored

    dist: dict, optional (default=None)
        dict to store distance from source to the keyed node
        If None, returned dist dict contents default to 0 for every node in the
        source list

    target: node label, optional
        Ending node for path. Path lengths to other destinations may (and
        probably will) be incorrect.

    heuristic : bool
        Determines whether to use a heuristic to early detect negative
        cycles at a hopefully negligible cost.

    Returns
    -------
    dist : dict
        Returns a dict keyed by node to the distance from the source.
        Dicts for paths and pred are in the mutated input dicts by those names.

    Raises
    ------
    NodeNotFound
        If any of `source` is not in `G`.

    NetworkXUnbounded
        If the (di)graph contains a negative (di)cycle, the
        algorithm raises an exception to indicate the presence of the
        negative (di)cycle.  Note: any negative weight edge in an
        undirected graph is a negative cycle
    """
    if pred is None:
        pred = {v: [] for v in source}

    if dist is None:
        dist = {v: 0 for v in source}

    negative_cycle_found = _inner_bellman_ford(
        G,
        source,
        weight,
        pred,
        dist,
        heuristic,
    )
    if negative_cycle_found is not None:
        raise nx.NetworkXUnbounded("Negative cycle detected.")

    if paths is not None:
        sources = set(source)
        dsts = [target] if target is not None else pred
        for dst in dsts:
            gen = _build_paths_from_predecessors(sources, dst, pred)
            paths[dst] = next(gen)

    return dist


def _inner_bellman_ford(
    G,
    sources,
    weight,
    pred,
    dist=None,
    heuristic=True,
):
    """Inner Relaxation loop for Bellman–Ford algorithm.

    This is an implementation of the SPFA variant.
    See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

    Parameters
    ----------
    G : NetworkX graph

    source: list
        List of source nodes. The shortest path from any of the source
        nodes will be found if multiple sources are provided.

    weight : function
        The weight of an edge is the value returned by the function. The
        function must accept exactly three positional arguments: the two
        endpoints of an edge and the dictionary of edge attributes for
        that edge. The function must return a number.

    pred: dict of lists
        dict to store a list of predecessors keyed by that node

    dist: dict, optional (default=None)
        dict to store distance from source to the keyed node
        If None, returned dist dict contents default to 0 for every node in the
        source list

    heuristic : bool
        Determines whether to use a heuristic to early detect negative
        cycles at a hopefully negligible cost.

    Returns
    -------
    node or None
        Return a node `v` where processing discovered a negative cycle.
        If no negative cycle found, return None.

    Raises
    ------
    NodeNotFound
        If any of `source` is not in `G`.
    """
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f"Source {s} not in G")

    if pred is None:
        pred = {v: [] for v in sources}

    if dist is None:
        dist = {v: 0 for v in sources}

    # Heuristic Storage setup. Note: use None because nodes cannot be None
    nonexistent_edge = (None, None)
    pred_edge = {v: None for v in sources}
    recent_update = {v: nonexistent_edge for v in sources}

