    The algorithm proceeds as follows. First each of the initial k communities
    is initialized in a random vertex in the graph. Then the algorithm iterates
    over all vertices in a random order, updating the community of each vertex
    based on its own community and the communities of its neighbors. This
    process is performed several times until convergence.
    At all times, each community has a total density of 1, which is equally
    distributed among the vertices it contains. If a vertex changes of
    community, vertex densities of affected communities are adjusted
    immediately. When a complete iteration over all vertices is done, such that
    no vertex changes the community it belongs to, the algorithm has converged
    and returns.

    This is the original version of the algorithm described in [1]_.
    Unfortunately, it does not support weighted graphs yet.

    Parameters
    ----------
    G : NetworkX graph
        Graph must be simple and undirected.

    k : integer
        The number of communities to be found.

    max_iter : integer
        The number of maximum iterations allowed. By default 100.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    communities : iterable
        Iterable of communities given as sets of nodes.

    Notes
    -----
    k variable is not an optional argument.

    References
    ----------
    .. [1] Parés F., Garcia-Gasulla D. et al. "Fluid Communities: A
       Competitive and Highly Scalable Community Detection Algorithm".
       [https://arxiv.org/pdf/1703.09307.pdf].
    """
    # Initial checks
    if not isinstance(k, int):
        raise NetworkXError("k must be an integer.")
    if not k > 0:
        raise NetworkXError("k must be greater than 0.")
    if not is_connected(G):
        raise NetworkXError("Fluid Communities require connected Graphs.")
    if len(G) < k:
        raise NetworkXError("k cannot be bigger than the number of nodes.")
    # Initialization
    max_density = 1.0
    vertices = list(G)
    seed.shuffle(vertices)
    communities = {n: i for i, n in enumerate(vertices[:k])}
    density = {}
    com_to_numvertices = {}
    for vertex in communities:
        com_to_numvertices[communities[vertex]] = 1
        density[communities[vertex]] = max_density
    # Set up control variables and start iterating
    iter_count = 0
    cont = True
    while cont:
        cont = False
        iter_count += 1
        # Loop over all vertices in graph in a random order
        vertices = list(G)
        seed.shuffle(vertices)
        for vertex in vertices:
            # Updating rule
            com_counter = Counter()
            # Take into account self vertex community
            try:
                com_counter.update({communities[vertex]: density[communities[vertex]]})
            except KeyError:
                pass
            # Gather neighbor vertex communities
            for v in G[vertex]:
                try:
                    com_counter.update({communities[v]: density[communities[v]]})
                except KeyError:
                    continue
            # Check which is the community with highest density
            new_com = -1
            if len(com_counter.keys()) > 0:
                max_freq = max(com_counter.values())
                best_communities = [
                    com
                    for com, freq in com_counter.items()
                    if (max_freq - freq) < 0.0001
                ]
                # If actual vertex com in best communities, it is preserved
                try:
                    if communities[vertex] in best_communities:
                        new_com = communities[vertex]
                except KeyError:
                    pass
                # If vertex community changes...
                if new_com == -1:
                    # Set flag of non-convergence
                    cont = True
                    # Randomly chose a new community from candidates
                    new_com = seed.choice(best_communities)
                    # Update previous community status
                    try:
                        com_to_numvertices[communities[vertex]] -= 1
                        density[communities[vertex]] = (
                            max_density / com_to_numvertices[communities[vertex]]
                        )
                    except KeyError:
                        pass
                    # Update new community status
                    communities[vertex] = new_com
                    com_to_numvertices[communities[vertex]] += 1
                    density[communities[vertex]] = (
                        max_density / com_to_numvertices[communities[vertex]]
                    )
        # If maximum iterations reached --> output actual results
        if iter_count > max_iter:
            break
    # Return results by grouping communities as list of vertices
    return iter(groups(communities).values())
<|endoftext|>"""Helper functions for community-finding algorithms."""

import networkx as nx

__all__ = ["is_partition"]


@nx._dispatchable
def is_partition(G, communities):
    """Returns *True* if `communities` is a partition of the nodes of `G`.

    A partition of a universe set is a family of pairwise disjoint sets
    whose union is the entire universe set.

    Parameters
    ----------
    G : NetworkX graph.

    communities : list or iterable of sets of nodes
        If not a list, the iterable is converted internally to a list.
        If it is an iterator it is exhausted.

    """
    # Alternate implementation:
    # return all(sum(1 if v in c else 0 for c in communities) == 1 for v in G)
    if not isinstance(communities, list):
        communities = list(communities)
    nodes = {n for c in communities for n in c if n in G}

    return len(G) == len(nodes) == sum(len(c) for c in communities)
<|endoftext|>"""Functions for computing the Kernighan–Lin bipartition algorithm."""

from itertools import count

import networkx as nx
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils import BinaryHeap, not_implemented_for, py_random_state

__all__ = ["kernighan_lin_bisection"]


def _kernighan_lin_sweep(edges, side):
    """
    This is a modified form of Kernighan-Lin, which moves single nodes at a
    time, alternating between sides to keep the bisection balanced.  We keep
    two min-heaps of swap costs to make optimal-next-move selection fast.
    """
    costs0, costs1 = costs = BinaryHeap(), BinaryHeap()
    for u, side_u, edges_u in zip(count(), side, edges):
        cost_u = sum(w if side[v] else -w for v, w in edges_u)
        costs[side_u].insert(u, cost_u if side_u else -cost_u)

    def _update_costs(costs_x, x):
        for y, w in edges[x]:
            costs_y = costs[side[y]]
            cost_y = costs_y.get(y)
            if cost_y is not None:
                cost_y += 2 * (-w if costs_x is costs_y else w)
                costs_y.insert(y, cost_y, True)

    i = 0
    totcost = 0
    while costs0 and costs1:
        u, cost_u = costs0.pop()
        _update_costs(costs0, u)
        v, cost_v = costs1.pop()
        _update_costs(costs1, v)
        totcost += cost_u + cost_v
        i += 1
        yield totcost, i, (u, v)


@not_implemented_for("directed")
@py_random_state(4)
@nx._dispatchable(edge_attrs="weight")
def kernighan_lin_bisection(G, partition=None, max_iter=10, weight="weight", seed=None):
    """Partition a graph into two blocks using the Kernighan–Lin
    algorithm.

    This algorithm partitions a network into two sets by iteratively
    swapping pairs of nodes to reduce the edge cut between the two sets.  The
    pairs are chosen according to a modified form of Kernighan-Lin [1]_, which
    moves node individually, alternating between sides to keep the bisection
    balanced.

    Parameters
    ----------
    G : NetworkX graph
        Graph must be undirected.

    partition : tuple
        Pair of iterables containing an initial partition. If not
        specified, a random balanced partition is used.

    max_iter : int
        Maximum number of times to attempt swaps to find an
        improvement before giving up.

    weight : key
        Edge data key to use as weight. If None, the weights are all
        set to one.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Only used if partition is None

    Returns
    -------
    partition : tuple
        A pair of sets of nodes representing the bipartition.

    Raises
    ------
    NetworkXError
        If partition is not a valid partition of the nodes of the graph.

    References
    ----------
    .. [1] Kernighan, B. W.; Lin, Shen (1970).
       "An efficient heuristic procedure for partitioning graphs."
       *Bell Systems Technical Journal* 49: 291--307.
       Oxford University Press 2011.

    """
    n = len(G)
    labels = list(G)
    seed.shuffle(labels)
    index = {v: i for i, v in enumerate(labels)}

    if partition is None:
        side = [0] * (n // 2) + [1] * ((n + 1) // 2)
    else:
        try:
            A, B = partition
        except (TypeError, ValueError) as err:
            raise nx.NetworkXError("partition must be two sets") from err
        if not is_partition(G, (A, B)):
            raise nx.NetworkXError("partition invalid")
        side = [0] * n
        for a in A:
            side[index[a]] = 1

    if G.is_multigraph():
        edges = [
            [
                (index[u], sum(e.get(weight, 1) for e in d.values()))
                for u, d in G[v].items()
            ]
            for v in labels
        ]
    else:
        edges = [
            [(index[u], e.get(weight, 1)) for u, e in G[v].items()] for v in labels
        ]

    for i in range(max_iter):
        costs = list(_kernighan_lin_sweep(edges, side))
        min_cost, min_i, _ = min(costs)
        if min_cost >= 0:
            break

        for _, _, (u, v) in costs[:min_i]:
            side[u] = 1
            side[v] = 0

    A = {u for u, s in zip(labels, side) if s == 0}
    B = {u for u, s in zip(labels, side) if s == 1}
    return A, B
<|endoftext|>from collections import defaultdict

import networkx as nx

__all__ = ["k_clique_communities"]


@nx._dispatchable
def k_clique_communities(G, k, cliques=None):
    """Find k-clique communities in graph using the percolation method.

    A k-clique community is the union of all cliques of size k that
    can be reached through adjacent (sharing k-1 nodes) k-cliques.

    Parameters
    ----------
    G : NetworkX graph

    k : int
       Size of smallest clique

    cliques: list or generator
       Precomputed cliques (use networkx.find_cliques(G))

    Returns
    -------
    Yields sets of nodes, one for each k-clique community.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> K5 = nx.convert_node_labels_to_integers(G, first_label=2)
    >>> G.add_edges_from(K5.edges())
    >>> c = list(nx.community.k_clique_communities(G, 4))
    >>> sorted(list(c[0]))
    [0, 1, 2, 3, 4, 5, 6]
    >>> list(nx.community.k_clique_communities(G, 6))
    []

    References
    ----------
    .. [1] Gergely Palla, Imre Derényi, Illés Farkas1, and Tamás Vicsek,
       Uncovering the overlapping community structure of complex networks
       in nature and society Nature 435, 814-818, 2005,
       doi:10.1038/nature03607
    """
    if k < 2:
        raise nx.NetworkXError(f"k={k}, k must be greater than 1.")
    if cliques is None:
        cliques = nx.find_cliques(G)
    cliques = [frozenset(c) for c in cliques if len(c) >= k]

    # First index which nodes are in which cliques
    membership_dict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            membership_dict[node].append(clique)

    # For each clique, see which adjacent cliques percolate
    perc_graph = nx.Graph()
    perc_graph.add_nodes_from(cliques)
    for clique in cliques:
        for adj_clique in _get_adjacent_cliques(clique, membership_dict):
            if len(clique.intersection(adj_clique)) >= (k - 1):
                perc_graph.add_edge(clique, adj_clique)

    # Connected components of clique graph with perc edges
    # are the percolated cliques
    for component in nx.connected_components(perc_graph):
        yield (frozenset.union(*component))


def _get_adjacent_cliques(clique, membership_dict):
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques
<|endoftext|>"""Functions for computing and measuring community structure.

The ``community`` subpackage can be accessed by using :mod:`networkx.community`, then accessing the
functions as attributes of ``community``. For example::

    >>> import networkx as nx
    >>> G = nx.barbell_graph(5, 1)
    >>> communities_generator = nx.community.girvan_newman(G)
    >>> top_level_communities = next(communities_generator)
    >>> next_level_communities = next(communities_generator)
    >>> sorted(map(sorted, next_level_communities))
    [[0, 1, 2, 3, 4], [5], [6, 7, 8, 9, 10]]

"""

from networkx.algorithms.community.asyn_fluid import *
from networkx.algorithms.community.centrality import *
from networkx.algorithms.community.divisive import *
from networkx.algorithms.community.kclique import *
from networkx.algorithms.community.kernighan_lin import *
from networkx.algorithms.community.label_propagation import *
from networkx.algorithms.community.lukes import *
from networkx.algorithms.community.modularity_max import *
from networkx.algorithms.community.quality import *
from networkx.algorithms.community.community_utils import *
from networkx.algorithms.community.louvain import *
<|endoftext|>"""Functions for computing communities based on centrality notions."""

import networkx as nx

__all__ = ["girvan_newman"]


@nx._dispatchable(preserve_edge_attrs="most_valuable_edge")
def girvan_newman(G, most_valuable_edge=None):
    """Finds communities in a graph using the Girvan–Newman method.

    Parameters
    ----------
    G : NetworkX graph

    most_valuable_edge : function
        Function that takes a graph as input and outputs an edge. The
        edge returned by this function will be recomputed and removed at
        each iteration of the algorithm.

        If not specified, the edge with the highest
        :func:`networkx.edge_betweenness_centrality` will be used.

    Returns
    -------
    iterator
        Iterator over tuples of sets of nodes in `G`. Each set of node
        is a community, each tuple is a sequence of communities at a
        particular level of the algorithm.

    Examples
    --------
    To get the first pair of communities::

        >>> G = nx.path_graph(10)
        >>> comp = nx.community.girvan_newman(G)
        >>> tuple(sorted(c) for c in next(comp))
        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])

    To get only the first *k* tuples of communities, use
    :func:`itertools.islice`::

        >>> import itertools
        >>> G = nx.path_graph(8)
        >>> k = 2
        >>> comp = nx.community.girvan_newman(G)
        >>> for communities in itertools.islice(comp, k):
        ...     print(tuple(sorted(c) for c in communities))
        ...
        ([0, 1, 2, 3], [4, 5, 6, 7])
        ([0, 1], [2, 3], [4, 5, 6, 7])

    To stop getting tuples of communities once the number of communities
    is greater than *k*, use :func:`itertools.takewhile`::

        >>> import itertools
        >>> G = nx.path_graph(8)
        >>> k = 4
        >>> comp = nx.community.girvan_newman(G)
        >>> limited = itertools.takewhile(lambda c: len(c) <= k, comp)
        >>> for communities in limited:
        ...     print(tuple(sorted(c) for c in communities))
        ...
        ([0, 1, 2, 3], [4, 5, 6, 7])
        ([0, 1], [2, 3], [4, 5, 6, 7])
        ([0, 1], [2, 3], [4, 5], [6, 7])

    To just choose an edge to remove based on the weight::

        >>> from operator import itemgetter
        >>> G = nx.path_graph(10)
        >>> edges = G.edges()
        >>> nx.set_edge_attributes(G, {(u, v): v for u, v in edges}, "weight")
        >>> def heaviest(G):
        ...     u, v, w = max(G.edges(data="weight"), key=itemgetter(2))
        ...     return (u, v)
        ...
        >>> comp = nx.community.girvan_newman(G, most_valuable_edge=heaviest)
        >>> tuple(sorted(c) for c in next(comp))
        ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9])

    To utilize edge weights when choosing an edge with, for example, the
    highest betweenness centrality::

        >>> from networkx import edge_betweenness_centrality as betweenness
        >>> def most_central_edge(G):
        ...     centrality = betweenness(G, weight="weight")
        ...     return max(centrality, key=centrality.get)
        ...
        >>> G = nx.path_graph(10)
        >>> comp = nx.community.girvan_newman(G, most_valuable_edge=most_central_edge)
        >>> tuple(sorted(c) for c in next(comp))
        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])

    To specify a different ranking algorithm for edges, use the
    `most_valuable_edge` keyword argument::

        >>> from networkx import edge_betweenness_centrality
        >>> from random import random
        >>> def most_central_edge(G):
        ...     centrality = edge_betweenness_centrality(G)
        ...     max_cent = max(centrality.values())
        ...     # Scale the centrality values so they are between 0 and 1,
        ...     # and add some random noise.
        ...     centrality = {e: c / max_cent for e, c in centrality.items()}
        ...     # Add some random noise.
        ...     centrality = {e: c + random() for e, c in centrality.items()}
        ...     return max(centrality, key=centrality.get)
        ...
        >>> G = nx.path_graph(10)
        >>> comp = nx.community.girvan_newman(G, most_valuable_edge=most_central_edge)

    Notes
    -----
    The Girvan–Newman algorithm detects communities by progressively
    removing edges from the original graph. The algorithm removes the
    "most valuable" edge, traditionally the edge with the highest
    betweenness centrality, at each step. As the graph breaks down into
    pieces, the tightly knit community structure is exposed and the
    result can be depicted as a dendrogram.

    """
    # If the graph is already empty, simply return its connected
    # components.
    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return
    # If no function is provided for computing the most valuable edge,
    # use the edge betweenness centrality.
    if most_valuable_edge is None:

        def most_valuable_edge(G):
            """Returns the edge with the highest betweenness centrality
            in the graph `G`.

            """
            # We have guaranteed that the graph is non-empty, so this
            # dictionary will never be empty.
            betweenness = nx.edge_betweenness_centrality(G)
            return max(betweenness, key=betweenness.get)

    # The copy of G here must include the edge weight data.
    g = G.copy().to_undirected()
    # Self-loops must be removed because their removal has no effect on
    # the connected components of the graph.
    g.remove_edges_from(nx.selfloop_edges(g))
    while g.number_of_edges() > 0:
        yield _without_most_central_edges(g, most_valuable_edge)


def _without_most_central_edges(G, most_valuable_edge):
    """Returns the connected components of the graph that results from
    repeatedly removing the most "valuable" edge in the graph.

    `G` must be a non-empty graph. This function modifies the graph `G`
    in-place; that is, it removes edges on the graph `G`.

    `most_valuable_edge` is a function that takes the graph `G` as input
    (or a subgraph with one or more edges of `G` removed) and returns an
    edge. That edge will be removed and this process will be repeated
    until the number of connected components in the graph increases.

    """
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)
    return new_components
<|endoftext|>import functools

import networkx as nx

__all__ = [
    "edge_betweenness_partition",
    "edge_current_flow_betweenness_partition",
]


@nx._dispatchable(edge_attrs="weight")
def edge_betweenness_partition(G, number_of_sets, *, weight=None):
    """Partition created by iteratively removing the highest edge betweenness edge.

    This algorithm works by calculating the edge betweenness for all
    edges and removing the edge with the highest value. It is then
    determined whether the graph has been broken into at least
    `number_of_sets` connected components.
    If not the process is repeated.

    Parameters
    ----------
    G : NetworkX Graph, DiGraph or MultiGraph
      Graph to be partitioned

    number_of_sets : int
      Number of sets in the desired partition of the graph

    weight : key, optional, default=None
      The key to use if using weights for edge betweenness calculation

    Returns
    -------
    C : list of sets
      Partition of the nodes of G

    Raises
    ------
    NetworkXError
      If number_of_sets is <= 0 or if number_of_sets > len(G)

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> part = nx.community.edge_betweenness_partition(G, 2)
    >>> {0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21} in part
    True
    >>> {
    ...     2,
    ...     8,
    ...     9,
    ...     14,
    ...     15,
    ...     18,
    ...     20,
    ...     22,
    ...     23,
    ...     24,
    ...     25,
    ...     26,
    ...     27,
    ...     28,
    ...     29,
    ...     30,
    ...     31,
    ...     32,
    ...     33,
    ... } in part
    True

    See Also
    --------
    edge_current_flow_betweenness_partition

    Notes
    -----
    This algorithm is fairly slow, as both the calculation of connected
    components and edge betweenness relies on all pairs shortest
    path algorithms. They could potentially be combined to cut down
    on overall computation time.

    References
    ----------
    .. [1] Santo Fortunato 'Community Detection in Graphs' Physical Reports
       Volume 486, Issue 3-5 p. 75-174
       http://arxiv.org/abs/0906.0612
    """
    if number_of_sets <= 0:
        raise nx.NetworkXError("number_of_sets must be >0")
    if number_of_sets == 1:
        return [set(G)]
    if number_of_sets == len(G):
        return [{n} for n in G]
    if number_of_sets > len(G):
        raise nx.NetworkXError("number_of_sets must be <= len(G)")

    H = G.copy()
    partition = list(nx.connected_components(H))
    while len(partition) < number_of_sets:
        ranking = nx.edge_betweenness_centrality(H, weight=weight)
        edge = max(ranking, key=ranking.get)
        H.remove_edge(*edge)
        partition = list(nx.connected_components(H))
    return partition


@nx._dispatchable(edge_attrs="weight")
def edge_current_flow_betweenness_partition(G, number_of_sets, *, weight=None):
    """Partition created by removing the highest edge current flow betweenness edge.

    This algorithm works by calculating the edge current flow
    betweenness for all edges and removing the edge with the
    highest value. It is then determined whether the graph has
    been broken into at least `number_of_sets` connected
    components. If not the process is repeated.

    Parameters
    ----------
    G : NetworkX Graph, DiGraph or MultiGraph
      Graph to be partitioned

    number_of_sets : int
      Number of sets in the desired partition of the graph

    weight : key, optional (default=None)
      The edge attribute key to use as weights for
      edge current flow betweenness calculations

    Returns
    -------
    C : list of sets
      Partition of G

    Raises
    ------
    NetworkXError
      If number_of_sets is <= 0 or number_of_sets > len(G)

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> part = nx.community.edge_current_flow_betweenness_partition(G, 2)
    >>> {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 19, 21} in part
    True
    >>> {8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33} in part
    True


    See Also
    --------
    edge_betweenness_partition

    Notes
    -----
    This algorithm is extremely slow, as the recalculation of the edge
    current flow betweenness is extremely slow.

    References
    ----------
    .. [1] Santo Fortunato 'Community Detection in Graphs' Physical Reports
       Volume 486, Issue 3-5 p. 75-174
       http://arxiv.org/abs/0906.0612
    """
    if number_of_sets <= 0:
        raise nx.NetworkXError("number_of_sets must be >0")
    elif number_of_sets == 1:
        return [set(G)]
    elif number_of_sets == len(G):
        return [{n} for n in G]
    elif number_of_sets > len(G):
        raise nx.NetworkXError("number_of_sets must be <= len(G)")

    rank = functools.partial(
        nx.edge_current_flow_betweenness_centrality, normalized=False, weight=weight
    )

    # current flow requires a connected network so we track the components explicitly
    H = G.copy()
    partition = list(nx.connected_components(H))
    if len(partition) > 1:
        Hcc_subgraphs = [H.subgraph(cc).copy() for cc in partition]
    else:
        Hcc_subgraphs = [H]

    ranking = {}
    for Hcc in Hcc_subgraphs:
        ranking.update(rank(Hcc))

    while len(partition) < number_of_sets:
        edge = max(ranking, key=ranking.get)
        for cc, Hcc in zip(partition, Hcc_subgraphs):
            if edge[0] in cc:
                Hcc.remove_edge(*edge)
                del ranking[edge]
                splitcc_list = list(nx.connected_components(Hcc))
                if len(splitcc_list) > 1:
                    # there are 2 connected components. split off smaller one
                    cc_new = min(splitcc_list, key=len)
                    Hcc_new = Hcc.subgraph(cc_new).copy()
                    # update edge rankings for Hcc_new
                    newranks = rank(Hcc_new)
                    for e, r in newranks.items():
                        ranking[e if e in ranking else e[::-1]] = r
                    # append new cc and Hcc to their lists.
                    partition.append(cc_new)
                    Hcc_subgraphs.append(Hcc_new)

                    # leave existing cc and Hcc in their lists, but shrink them
                    Hcc.remove_nodes_from(cc_new)
                    cc.difference_update(cc_new)
                # update edge rankings for Hcc whether it was split or not
                newranks = rank(Hcc)
                for e, r in newranks.items():
                    ranking[e if e in ranking else e[::-1]] = r
                break
    return partition
<|endoftext|>"""Functions for measuring the quality of a partition (into
communities).

"""

from itertools import combinations

import networkx as nx
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils.decorators import argmap

__all__ = ["modularity", "partition_quality"]


class NotAPartition(NetworkXError):
    """Raised if a given collection is not a partition."""

    def __init__(self, G, collection):
        msg = f"{collection} is not a valid partition of the graph {G}"
        super().__init__(msg)


def _require_partition(G, partition):
    """Decorator to check that a valid partition is input to a function

    Raises :exc:`networkx.NetworkXError` if the partition is not valid.

    This decorator should be used on functions whose first two arguments
    are a graph and a partition of the nodes of that graph (in that
    order)::

        >>> @require_partition
        ... def foo(G, partition):
        ...     print("partition is valid!")
        ...
        >>> G = nx.complete_graph(5)
        >>> partition = [{0, 1}, {2, 3}, {4}]
        >>> foo(G, partition)
        partition is valid!
        >>> partition = [{0}, {2, 3}, {4}]
        >>> foo(G, partition)
        Traceback (most recent call last):
          ...
        networkx.exception.NetworkXError: `partition` is not a valid partition of the nodes of G
        >>> partition = [{0, 1}, {1, 2, 3}, {4}]
        >>> foo(G, partition)
        Traceback (most recent call last):
          ...
        networkx.exception.NetworkXError: `partition` is not a valid partition of the nodes of G

    """
    if is_partition(G, partition):
        return G, partition
    raise nx.NetworkXError("`partition` is not a valid partition of the nodes of G")


require_partition = argmap(_require_partition, (0, 1))


@nx._dispatchable
def intra_community_edges(G, partition):
    """Returns the number of intra-community edges for a partition of `G`.

    Parameters
    ----------
    G : NetworkX graph.

    partition : iterable of sets of nodes
        This must be a partition of the nodes of `G`.

    The "intra-community edges" are those edges joining a pair of nodes
    in the same block of the partition.

    """
    return sum(G.subgraph(block).size() for block in partition)


@nx._dispatchable
def inter_community_edges(G, partition):
    """Returns the number of inter-community edges for a partition of `G`.
    according to the given
    partition of the nodes of `G`.

    Parameters
    ----------
    G : NetworkX graph.

    partition : iterable of sets of nodes
        This must be a partition of the nodes of `G`.

    The *inter-community edges* are those edges joining a pair of nodes
    in different blocks of the partition.

    Implementation note: this function creates an intermediate graph
    that may require the same amount of memory as that of `G`.

    """
    # Alternate implementation that does not require constructing a new
    # graph object (but does require constructing an affiliation
    # dictionary):
    #
    #     aff = dict(chain.from_iterable(((v, block) for v in block)
    #                                    for block in partition))
    #     return sum(1 for u, v in G.edges() if aff[u] != aff[v])
    #
    MG = nx.MultiDiGraph if G.is_directed() else nx.MultiGraph
    return nx.quotient_graph(G, partition, create_using=MG).size()


@nx._dispatchable
def inter_community_non_edges(G, partition):
    """Returns the number of inter-community non-edges according to the
    given partition of the nodes of `G`.

    Parameters
    ----------
    G : NetworkX graph.

    partition : iterable of sets of nodes
        This must be a partition of the nodes of `G`.

    A *non-edge* is a pair of nodes (undirected if `G` is undirected)
    that are not adjacent in `G`. The *inter-community non-edges* are
    those non-edges on a pair of nodes in different blocks of the
    partition.

    Implementation note: this function creates two intermediate graphs,
    which may require up to twice the amount of memory as required to
    store `G`.

    """
    # Alternate implementation that does not require constructing two
    # new graph objects (but does require constructing an affiliation
    # dictionary):
    #
    #     aff = dict(chain.from_iterable(((v, block) for v in block)
    #                                    for block in partition))
    #     return sum(1 for u, v in nx.non_edges(G) if aff[u] != aff[v])
    #
    return inter_community_edges(nx.complement(G), partition)


@nx._dispatchable(edge_attrs="weight")
def modularity(G, communities, weight="weight", resolution=1):
    r"""Returns the modularity of the given partition of the graph.

    Modularity is defined in [1]_ as

    .. math::
        Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \gamma\frac{k_ik_j}{2m}\right)
            \delta(c_i,c_j)

    where $m$ is the number of edges (or sum of all edge weights as in [5]_),
    $A$ is the adjacency matrix of `G`, $k_i$ is the (weighted) degree of $i$,
    $\gamma$ is the resolution parameter, and $\delta(c_i, c_j)$ is 1 if $i$ and
    $j$ are in the same community else 0.

    According to [2]_ (and verified by some algebra) this can be reduced to

    .. math::
       Q = \sum_{c=1}^{n}
       \left[ \frac{L_c}{m} - \gamma\left( \frac{k_c}{2m} \right) ^2 \right]

    where the sum iterates over all communities $c$, $m$ is the number of edges,
    $L_c$ is the number of intra-community links for community $c$,
    $k_c$ is the sum of degrees of the nodes in community $c$,
    and $\gamma$ is the resolution parameter.

    The resolution parameter sets an arbitrary tradeoff between intra-group
    edges and inter-group edges. More complex grouping patterns can be
    discovered by analyzing the same network with multiple values of gamma
    and then combining the results [3]_. That said, it is very common to
    simply use gamma=1. More on the choice of gamma is in [4]_.

    The second formula is the one actually used in calculation of the modularity.
    For directed graphs the second formula replaces $k_c$ with $k^{in}_c k^{out}_c$.

    Parameters
    ----------
    G : NetworkX Graph

    communities : list or iterable of set of nodes
        These node sets must represent a partition of G's nodes.

    weight : string or None, optional (default="weight")
        The edge attribute that holds the numerical value used
        as a weight. If None or an edge does not have that attribute,
        then that edge has weight 1.

    resolution : float (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    Returns
    -------
    Q : float
        The modularity of the partition.

    Raises
    ------
    NotAPartition
        If `communities` is not a partition of the nodes of `G`.

    Examples
    --------
    >>> G = nx.barbell_graph(3, 0)
    >>> nx.community.modularity(G, [{0, 1, 2}, {3, 4, 5}])
    0.35714285714285715
    >>> nx.community.modularity(G, nx.community.label_propagation_communities(G))
    0.35714285714285715

    References
    ----------
    .. [1] M. E. J. Newman "Networks: An Introduction", page 224.
       Oxford University Press, 2011.
    .. [2] Clauset, Aaron, Mark EJ Newman, and Cristopher Moore.
       "Finding community structure in very large networks."
       Phys. Rev. E 70.6 (2004). <https://arxiv.org/abs/cond-mat/0408187>
    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community Detection"
       Phys. Rev. E 74, 016110, 2006. https://doi.org/10.1103/PhysRevE.74.016110
    .. [4] M. E. J. Newman, "Equivalence between modularity optimization and
       maximum likelihood methods for community detection"
       Phys. Rev. E 94, 052315, 2016. https://doi.org/10.1103/PhysRevE.94.052315
    .. [5] Blondel, V.D. et al. "Fast unfolding of communities in large
       networks" J. Stat. Mech 10008, 1-12 (2008).
       https://doi.org/10.1088/1742-5468/2008/10/P10008
    """
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_partition(G, communities):
        raise NotAPartition(G, communities)

    directed = G.is_directed()
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        m = sum(out_degree.values())
        norm = 1 / m**2
    else:
        out_degree = in_degree = dict(G.degree(weight=weight))
        deg_sum = sum(out_degree.values())
        m = deg_sum / 2
        norm = 1 / deg_sum**2

    def community_contribution(community):
        comm = set(community)
        L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)

        out_degree_sum = sum(out_degree[u] for u in comm)
        in_degree_sum = sum(in_degree[u] for u in comm) if directed else out_degree_sum

        return L_c / m - resolution * out_degree_sum * in_degree_sum * norm

    return sum(map(community_contribution, communities))


@require_partition
@nx._dispatchable
def partition_quality(G, partition):
    """Returns the coverage and performance of a partition of G.

    The *coverage* of a partition is the ratio of the number of
    intra-community edges to the total number of edges in the graph.

    The *performance* of a partition is the number of
    intra-community edges plus inter-community non-edges divided by the total
    number of potential edges.

    This algorithm has complexity $O(C^2 + L)$ where C is the number of communities and L is the number of links.

    Parameters
    ----------
    G : NetworkX graph

    partition : sequence
        Partition of the nodes of `G`, represented as a sequence of
        sets of nodes (blocks). Each block of the partition represents a
        community.

    Returns
    -------
    (float, float)
        The (coverage, performance) tuple of the partition, as defined above.

    Raises
    ------
    NetworkXError
        If `partition` is not a valid partition of the nodes of `G`.

    Notes
    -----
    If `G` is a multigraph;
        - for coverage, the multiplicity of edges is counted
        - for performance, the result is -1 (total number of possible edges is not defined)

    References
    ----------
    .. [1] Santo Fortunato.
           "Community Detection in Graphs".
           *Physical Reports*, Volume 486, Issue 3--5 pp. 75--174
           <https://arxiv.org/abs/0906.0612>
    """

    node_community = {}
    for i, community in enumerate(partition):
        for node in community:
            node_community[node] = i

    # `performance` is not defined for multigraphs
    if not G.is_multigraph():
        # Iterate over the communities, quadratic, to calculate `possible_inter_community_edges`
        possible_inter_community_edges = sum(
            len(p1) * len(p2) for p1, p2 in combinations(partition, 2)
        )

        if G.is_directed():
            possible_inter_community_edges *= 2
    else:
        possible_inter_community_edges = 0

    # Compute the number of edges in the complete graph -- `n` nodes,
    # directed or undirected, depending on `G`
    n = len(G)
    total_pairs = n * (n - 1)
    if not G.is_directed():
        total_pairs //= 2

    intra_community_edges = 0
    inter_community_non_edges = possible_inter_community_edges

    # Iterate over the links to count `intra_community_edges` and `inter_community_non_edges`
    for e in G.edges():
        if node_community[e[0]] == node_community[e[1]]:
            intra_community_edges += 1
        else:
            inter_community_non_edges -= 1

    coverage = intra_community_edges / len(G.edges)

    if G.is_multigraph():
        performance = -1.0
    else:
        performance = (intra_community_edges + inter_community_non_edges) / total_pairs

    return coverage, performance
<|endoftext|>"""Functions for detecting communities based on modularity."""

from collections import defaultdict

import networkx as nx
from networkx.algorithms.community.quality import modularity
from networkx.utils import not_implemented_for
from networkx.utils.mapped_queue import MappedQueue

__all__ = [
    "greedy_modularity_communities",
    "naive_greedy_modularity_communities",
]


def _greedy_modularity_communities_generator(G, weight=None, resolution=1):
    r"""Yield community partitions of G and the modularity change at each step.

    This function performs Clauset-Newman-Moore greedy modularity maximization [2]_
    At each step of the process it yields the change in modularity that will occur in
    the next step followed by yielding the new community partition after that step.

    Greedy modularity maximization begins with each node in its own community
    and repeatedly joins the pair of communities that lead to the largest
    modularity until one community contains all nodes (the partition has one set).

    This function maximizes the generalized modularity, where `resolution`
    is the resolution parameter, often expressed as $\gamma$.
    See :func:`~networkx.algorithms.community.quality.modularity`.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight.  If None, then each edge has weight 1.
        The degree is the sum of the edge weights adjacent to the node.

    resolution : float (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    Yields
    ------
    Alternating yield statements produce the following two objects:

    communities: dict_values
        A dict_values of frozensets of nodes, one for each community.
        This represents a partition of the nodes of the graph into communities.
        The first yield is the partition with each node in its own community.

    dq: float
        The change in modularity when merging the next two communities
        that leads to the largest modularity.

    See Also
    --------
    modularity

    References
    ----------
    .. [1] Newman, M. E. J. "Networks: An Introduction", page 224
       Oxford University Press 2011.
    .. [2] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.
    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community
       Detection" Phys. Rev. E74, 2006.
    .. [4] Newman, M. E. J."Analysis of weighted networks"
       Physical Review E 70(5 Pt 2):056131, 2004.
    """
    directed = G.is_directed()
    N = G.number_of_nodes()

    # Count edges (or the sum of edge-weights for weighted graphs)
    m = G.size(weight)
    q0 = 1 / m

    # Calculate degrees (notation from the papers)
    # a : the fraction of (weighted) out-degree for each node
    # b : the fraction of (weighted) in-degree for each node
    if directed:
        a = {node: deg_out * q0 for node, deg_out in G.out_degree(weight=weight)}
        b = {node: deg_in * q0 for node, deg_in in G.in_degree(weight=weight)}
    else:
        a = b = {node: deg * q0 * 0.5 for node, deg in G.degree(weight=weight)}

    # this preliminary step collects the edge weights for each node pair
    # It handles multigraph and digraph and works fine for graph.
    dq_dict = defaultdict(lambda: defaultdict(float))
    for u, v, wt in G.edges(data=weight, default=1):
        if u == v:
            continue
        dq_dict[u][v] += wt
        dq_dict[v][u] += wt

    # now scale and subtract the expected edge-weights term
    for u, nbrdict in dq_dict.items():
        for v, wt in nbrdict.items():
            dq_dict[u][v] = q0 * wt - resolution * (a[u] * b[v] + b[u] * a[v])

    # Use -dq to get a max_heap instead of a min_heap
    # dq_heap holds a heap for each node's neighbors
    dq_heap = {u: MappedQueue({(u, v): -dq for v, dq in dq_dict[u].items()}) for u in G}
    # H -> all_dq_heap holds a heap with the best items for each node
    H = MappedQueue([dq_heap[n].heap[0] for n in G if len(dq_heap[n]) > 0])

    # Initialize single-node communities
    communities = {n: frozenset([n]) for n in G}
    yield communities.values()

    # Merge the two communities that lead to the largest modularity
    while len(H) > 1:
        # Find best merge
        # Remove from heap of row maxes
        # Ties will be broken by choosing the pair with lowest min community id
        try:
            negdq, u, v = H.pop()
        except IndexError:
            break
        dq = -negdq
        yield dq
        # Remove best merge from row u heap
        dq_heap[u].pop()
        # Push new row max onto H
        if len(dq_heap[u]) > 0:
            H.push(dq_heap[u].heap[0])
        # If this element was also at the root of row v, we need to remove the
        # duplicate entry from H
        if dq_heap[v].heap[0] == (v, u):
            H.remove((v, u))
            # Remove best merge from row v heap
            dq_heap[v].remove((v, u))
            # Push new row max onto H
            if len(dq_heap[v]) > 0:
                H.push(dq_heap[v].heap[0])
        else:
            # Duplicate wasn't in H, just remove from row v heap
            dq_heap[v].remove((v, u))

        # Perform merge
        communities[v] = frozenset(communities[u] | communities[v])
        del communities[u]

        # Get neighbor communities connected to the merged communities
        u_nbrs = set(dq_dict[u])
        v_nbrs = set(dq_dict[v])
        all_nbrs = (u_nbrs | v_nbrs) - {u, v}
        both_nbrs = u_nbrs & v_nbrs
        # Update dq for merge of u into v
        for w in all_nbrs:
            # Calculate new dq value
            if w in both_nbrs:
                dq_vw = dq_dict[v][w] + dq_dict[u][w]
            elif w in v_nbrs:
                dq_vw = dq_dict[v][w] - resolution * (a[u] * b[w] + a[w] * b[u])
            else:  # w in u_nbrs
                dq_vw = dq_dict[u][w] - resolution * (a[v] * b[w] + a[w] * b[v])
            # Update rows v and w
            for row, col in [(v, w), (w, v)]:
                dq_heap_row = dq_heap[row]
                # Update dict for v,w only (u is removed below)
                dq_dict[row][col] = dq_vw
                # Save old max of per-row heap
                if len(dq_heap_row) > 0:
                    d_oldmax = dq_heap_row.heap[0]
                else:
                    d_oldmax = None
                # Add/update heaps
                d = (row, col)
                d_negdq = -dq_vw
                # Save old value for finding heap index
                if w in v_nbrs:
                    # Update existing element in per-row heap
                    dq_heap_row.update(d, d, priority=d_negdq)
                else:
                    # We're creating a new nonzero element, add to heap
                    dq_heap_row.push(d, priority=d_negdq)
                # Update heap of row maxes if necessary
                if d_oldmax is None:
                    # No entries previously in this row, push new max
                    H.push(d, priority=d_negdq)
                else:
                    # We've updated an entry in this row, has the max changed?
                    row_max = dq_heap_row.heap[0]
                    if d_oldmax != row_max or d_oldmax.priority != row_max.priority:
                        H.update(d_oldmax, row_max)

        # Remove row/col u from dq_dict matrix
        for w in dq_dict[u]:
            # Remove from dict
            dq_old = dq_dict[w][u]
            del dq_dict[w][u]
            # Remove from heaps if we haven't already
            if w != v:
                # Remove both row and column
                for row, col in [(w, u), (u, w)]:
                    dq_heap_row = dq_heap[row]
                    # Check if replaced dq is row max
                    d_old = (row, col)
                    if dq_heap_row.heap[0] == d_old:
                        # Update per-row heap and heap of row maxes
                        dq_heap_row.remove(d_old)
                        H.remove(d_old)
                        # Update row max
                        if len(dq_heap_row) > 0:
                            H.push(dq_heap_row.heap[0])
                    else:
                        # Only update per-row heap
                        dq_heap_row.remove(d_old)

        del dq_dict[u]
        # Mark row u as deleted, but keep placeholder
        dq_heap[u] = MappedQueue()
        # Merge u into v and update a
        a[v] += a[u]
        a[u] = 0
        if directed:
            b[v] += b[u]
            b[u] = 0

        yield communities.values()


@nx._dispatchable(edge_attrs="weight")
def greedy_modularity_communities(
    G,
    weight=None,
    resolution=1,
    cutoff=1,
    best_n=None,
):
    r"""Find communities in G using greedy modularity maximization.

    This function uses Clauset-Newman-Moore greedy modularity maximization [2]_
    to find the community partition with the largest modularity.

    Greedy modularity maximization begins with each node in its own community
    and repeatedly joins the pair of communities that lead to the largest
    modularity until no further increase in modularity is possible (a maximum).
    Two keyword arguments adjust the stopping condition. `cutoff` is a lower
    limit on the number of communities so you can stop the process before
    reaching a maximum (used to save computation time). `best_n` is an upper
    limit on the number of communities so you can make the process continue
    until at most n communities remain even if the maximum modularity occurs
    for more. To obtain exactly n communities, set both `cutoff` and `best_n` to n.

    This function maximizes the generalized modularity, where `resolution`
    is the resolution parameter, often expressed as $\gamma$.
    See :func:`~networkx.algorithms.community.quality.modularity`.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight.  If None, then each edge has weight 1.
        The degree is the sum of the edge weights adjacent to the node.

    resolution : float, optional (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    cutoff : int, optional (default=1)
        A minimum number of communities below which the merging process stops.
        The process stops at this number of communities even if modularity
        is not maximized. The goal is to let the user stop the process early.
        The process stops before the cutoff if it finds a maximum of modularity.

    best_n : int or None, optional (default=None)
        A maximum number of communities above which the merging process will
        not stop. This forces community merging to continue after modularity
        starts to decrease until `best_n` communities remain.
        If ``None``, don't force it to continue beyond a maximum.

    Raises
    ------
    ValueError : If the `cutoff` or `best_n`  value is not in the range
        ``[1, G.number_of_nodes()]``, or if `best_n` < `cutoff`.

    Returns
    -------
    communities: list
        A list of frozensets of nodes, one for each community.
        Sorted by length with largest communities first.

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> c = nx.community.greedy_modularity_communities(G)
    >>> sorted(c[0])
    [8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    See Also
    --------
    modularity

    References
    ----------
    .. [1] Newman, M. E. J. "Networks: An Introduction", page 224
       Oxford University Press 2011.
    .. [2] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.
    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community
       Detection" Phys. Rev. E74, 2006.
    .. [4] Newman, M. E. J."Analysis of weighted networks"
       Physical Review E 70(5 Pt 2):056131, 2004.
    """
    if not G.size():
        return [{n} for n in G]

    if (cutoff < 1) or (cutoff > G.number_of_nodes()):
        raise ValueError(f"cutoff must be between 1 and {len(G)}. Got {cutoff}.")
    if best_n is not None:
        if (best_n < 1) or (best_n > G.number_of_nodes()):
            raise ValueError(f"best_n must be between 1 and {len(G)}. Got {best_n}.")
        if best_n < cutoff:
            raise ValueError(f"Must have best_n >= cutoff. Got {best_n} < {cutoff}")
        if best_n == 1:
            return [set(G)]
    else:
        best_n = G.number_of_nodes()

    # retrieve generator object to construct output
    community_gen = _greedy_modularity_communities_generator(
        G, weight=weight, resolution=resolution
    )

    # construct the first best community
    communities = next(community_gen)

    # continue merging communities until one of the breaking criteria is satisfied
    while len(communities) > cutoff:
        try:
            dq = next(community_gen)
        # StopIteration occurs when communities are the connected components
        except StopIteration:
            communities = sorted(communities, key=len, reverse=True)
            # if best_n requires more merging, merge big sets for highest modularity
            while len(communities) > best_n:
                comm1, comm2, *rest = communities
                communities = [comm1 ^ comm2]
                communities.extend(rest)
            return communities

        # keep going unless max_mod is reached or best_n says to merge more
        if dq < 0 and len(communities) <= best_n:
            break
        communities = next(community_gen)

    return sorted(communities, key=len, reverse=True)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def naive_greedy_modularity_communities(G, resolution=1, weight=None):
    r"""Find communities in G using greedy modularity maximization.

    This implementation is O(n^4), much slower than alternatives, but it is
    provided as an easy-to-understand reference implementation.

    Greedy modularity maximization begins with each node in its own community
    and joins the pair of communities that most increases modularity until no
    such pair exists.

    This function maximizes the generalized modularity, where `resolution`
    is the resolution parameter, often expressed as $\gamma$.
    See :func:`~networkx.algorithms.community.quality.modularity`.

    Parameters
    ----------
    G : NetworkX graph
        Graph must be simple and undirected.

    resolution : float (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight.  If None, then each edge has weight 1.
        The degree is the sum of the edge weights adjacent to the node.

    Returns
    -------
    list
        A list of sets of nodes, one for each community.
        Sorted by length with largest communities first.

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> c = nx.community.naive_greedy_modularity_communities(G)
    >>> sorted(c[0])
    [8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    See Also
    --------
    greedy_modularity_communities
    modularity
    """
    # First create one community for each node
    communities = [frozenset([u]) for u in G.nodes()]
    # Track merges
    merges = []
    # Greedily merge communities until no improvement is possible
    old_modularity = None
    new_modularity = modularity(G, communities, resolution=resolution, weight=weight)
    while old_modularity is None or new_modularity > old_modularity:
        # Save modularity for comparison
        old_modularity = new_modularity
        # Find best pair to merge
        trial_communities = list(communities)
        to_merge = None
        for i, u in enumerate(communities):
            for j, v in enumerate(communities):
                # Skip i==j and empty communities
                if j <= i or len(u) == 0 or len(v) == 0:
                    continue
                # Merge communities u and v
                trial_communities[j] = u | v
                trial_communities[i] = frozenset([])
                trial_modularity = modularity(
                    G, trial_communities, resolution=resolution, weight=weight
                )
                if trial_modularity >= new_modularity:
                    # Check if strictly better or tie
                    if trial_modularity > new_modularity:
                        # Found new best, save modularity and group indexes
                        new_modularity = trial_modularity
                        to_merge = (i, j, new_modularity - old_modularity)
                    elif to_merge and min(i, j) < min(to_merge[0], to_merge[1]):
                        # Break ties by choosing pair with lowest min id
                        new_modularity = trial_modularity
                        to_merge = (i, j, new_modularity - old_modularity)
                # Un-merge
                trial_communities[i] = u
                trial_communities[j] = v
        if to_merge is not None:
            # If the best merge improves modularity, use it
            merges.append(to_merge)
            i, j, dq = to_merge
            u, v = communities[i], communities[j]
            communities[j] = u | v
            communities[i] = frozenset([])
    # Remove empty communities and sort
    return sorted((c for c in communities if len(c) > 0), key=len, reverse=True)
<|endoftext|>"""Lukes Algorithm for exact optimal weighted tree partitioning."""

from copy import deepcopy
from functools import lru_cache
from random import choice

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["lukes_partitioning"]

D_EDGE_W = "weight"
D_EDGE_VALUE = 1.0
D_NODE_W = "weight"
D_NODE_VALUE = 1
PKEY = "partitions"
CLUSTER_EVAL_CACHE_SIZE = 2048


def _split_n_from(n, min_size_of_first_part):
    # splits j in two parts of which the first is at least
    # the second argument
    assert n >= min_size_of_first_part
    for p1 in range(min_size_of_first_part, n + 1):
        yield p1, n - p1


@nx._dispatchable(node_attrs="node_weight", edge_attrs="edge_weight")
def lukes_partitioning(G, max_size, node_weight=None, edge_weight=None):
    """Optimal partitioning of a weighted tree using the Lukes algorithm.

    This algorithm partitions a connected, acyclic graph featuring integer
    node weights and float edge weights. The resulting clusters are such
    that the total weight of the nodes in each cluster does not exceed
    max_size and that the weight of the edges that are cut by the partition
    is minimum. The algorithm is based on [1]_.

    Parameters
    ----------
    G : NetworkX graph

    max_size : int
        Maximum weight a partition can have in terms of sum of
        node_weight for all nodes in the partition

    edge_weight : key
        Edge data key to use as weight. If None, the weights are all
        set to one.

    node_weight : key
        Node data key to use as weight. If None, the weights are all
        set to one. The data must be int.

    Returns
    -------
    partition : list
        A list of sets of nodes representing the clusters of the
        partition.

    Raises
    ------
    NotATree
        If G is not a tree.
    TypeError
        If any of the values of node_weight is not int.

    References
    ----------
    .. [1] Lukes, J. A. (1974).
       "Efficient Algorithm for the Partitioning of Trees."
       IBM Journal of Research and Development, 18(3), 217–224.

    """
    # First sanity check and tree preparation
    if not nx.is_tree(G):
        raise nx.NotATree("lukes_partitioning works only on trees")
    else:
        if nx.is_directed(G):
            root = [n for n, d in G.in_degree() if d == 0]
            assert len(root) == 1
            root = root[0]
            t_G = deepcopy(G)
        else:
            root = choice(list(G.nodes))
            # this has the desirable side effect of not inheriting attributes
            t_G = nx.dfs_tree(G, root)

    # Since we do not want to screw up the original graph,
    # if we have a blank attribute, we make a deepcopy
    if edge_weight is None or node_weight is None:
        safe_G = deepcopy(G)
        if edge_weight is None:
            nx.set_edge_attributes(safe_G, D_EDGE_VALUE, D_EDGE_W)
            edge_weight = D_EDGE_W
        if node_weight is None:
            nx.set_node_attributes(safe_G, D_NODE_VALUE, D_NODE_W)
            node_weight = D_NODE_W
    else:
        safe_G = G

    # Second sanity check
    # The values of node_weight MUST BE int.
    # I cannot see any room for duck typing without incurring serious
    # danger of subtle bugs.
    all_n_attr = nx.get_node_attributes(safe_G, node_weight).values()
    for x in all_n_attr:
        if not isinstance(x, int):
            raise TypeError(
                "lukes_partitioning needs integer "
                f"values for node_weight ({node_weight})"
            )

    # SUBROUTINES -----------------------
    # these functions are defined here for two reasons:
    # - brevity: we can leverage global "safe_G"
    # - caching: signatures are hashable

    @not_implemented_for("undirected")
    # this is intended to be called only on t_G
    def _leaves(gr):
        for x in gr.nodes:
            if not nx.descendants(gr, x):
                yield x

    @not_implemented_for("undirected")
    def _a_parent_of_leaves_only(gr):
        tleaves = set(_leaves(gr))
        for n in set(gr.nodes) - tleaves:
            if all(x in tleaves for x in nx.descendants(gr, n)):
                return n

    @lru_cache(CLUSTER_EVAL_CACHE_SIZE)
    def _value_of_cluster(cluster):
        valid_edges = [e for e in safe_G.edges if e[0] in cluster and e[1] in cluster]
        return sum(safe_G.edges[e][edge_weight] for e in valid_edges)

    def _value_of_partition(partition):
        return sum(_value_of_cluster(frozenset(c)) for c in partition)

    @lru_cache(CLUSTER_EVAL_CACHE_SIZE)
    def _weight_of_cluster(cluster):
        return sum(safe_G.nodes[n][node_weight] for n in cluster)

    def _pivot(partition, node):
        ccx = [c for c in partition if node in c]
        assert len(ccx) == 1
        return ccx[0]

    def _concatenate_or_merge(partition_1, partition_2, x, i, ref_weight):
        ccx = _pivot(partition_1, x)
        cci = _pivot(partition_2, i)
        merged_xi = ccx.union(cci)

        # We first check if we can do the merge.
        # If so, we do the actual calculations, otherwise we concatenate
        if _weight_of_cluster(frozenset(merged_xi)) <= ref_weight:
            cp1 = list(filter(lambda x: x != ccx, partition_1))
            cp2 = list(filter(lambda x: x != cci, partition_2))

            option_2 = [merged_xi] + cp1 + cp2
            return option_2, _value_of_partition(option_2)
        else:
            option_1 = partition_1 + partition_2
            return option_1, _value_of_partition(option_1)

    # INITIALIZATION -----------------------
    leaves = set(_leaves(t_G))
    for lv in leaves:
        t_G.nodes[lv][PKEY] = {}
        slot = safe_G.nodes[lv][node_weight]
        t_G.nodes[lv][PKEY][slot] = [{lv}]
        t_G.nodes[lv][PKEY][0] = [{lv}]

    for inner in [x for x in t_G.nodes if x not in leaves]:
        t_G.nodes[inner][PKEY] = {}
        slot = safe_G.nodes[inner][node_weight]
        t_G.nodes[inner][PKEY][slot] = [{inner}]
    nx._clear_cache(t_G)

    # CORE ALGORITHM -----------------------
    while True:
        x_node = _a_parent_of_leaves_only(t_G)
        weight_of_x = safe_G.nodes[x_node][node_weight]
        best_value = 0
        best_partition = None
        bp_buffer = {}
        x_descendants = nx.descendants(t_G, x_node)
        for i_node in x_descendants:
            for j in range(weight_of_x, max_size + 1):
                for a, b in _split_n_from(j, weight_of_x):
                    if (
                        a not in t_G.nodes[x_node][PKEY]
                        or b not in t_G.nodes[i_node][PKEY]
                    ):
                        # it's not possible to form this particular weight sum
                        continue

                    part1 = t_G.nodes[x_node][PKEY][a]
                    part2 = t_G.nodes[i_node][PKEY][b]
                    part, value = _concatenate_or_merge(part1, part2, x_node, i_node, j)

                    if j not in bp_buffer or bp_buffer[j][1] < value:
                        # we annotate in the buffer the best partition for j
                        bp_buffer[j] = part, value

                    # we also keep track of the overall best partition
                    if best_value <= value:
                        best_value = value
                        best_partition = part

            # as illustrated in Lukes, once we finished a child, we can
            # discharge the partitions we found into the graph
            # (the key phrase is make all x == x')
            # so that they are used by the subsequent children
            for w, (best_part_for_vl, vl) in bp_buffer.items():
                t_G.nodes[x_node][PKEY][w] = best_part_for_vl
            bp_buffer.clear()

        # the absolute best partition for this node
        # across all weights has to be stored at 0
        t_G.nodes[x_node][PKEY][0] = best_partition
        t_G.remove_nodes_from(x_descendants)

        if x_node == root:
            # the 0-labeled partition of root
            # is the optimal one for the whole tree
            return t_G.nodes[root][PKEY][0]
<|endoftext|>"""Hubs and authorities analysis of graph structure."""

import networkx as nx

__all__ = ["hits"]


@nx._dispatchable(preserve_edge_attrs={"G": {"weight": 1}})
def hits(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True):
    """Returns HITS hubs and authorities values for nodes.

    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    max_iter : integer, optional
      Maximum number of iterations in power method.

    tol : float, optional
      Error tolerance used to check convergence in power method iteration.

    nstart : dictionary, optional
      Starting value of each node for power method iteration.

    normalized : bool (default=True)
       Normalize results by the sum of all of the values.

    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> h, a = nx.hits(G)

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-32, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    import numpy as np
    import scipy as sp

    if len(G) == 0:
        return {}, {}
    A = nx.adjacency_matrix(G, nodelist=list(G), dtype=float)

    if nstart is not None:
        nstart = np.array(list(nstart.values()))
    if max_iter <= 0:
        raise nx.PowerIterationFailedConvergence(max_iter)
    try:
        _, _, vt = sp.sparse.linalg.svds(A, k=1, v0=nstart, maxiter=max_iter, tol=tol)
    except sp.sparse.linalg.ArpackNoConvergence as exc:
        raise nx.PowerIterationFailedConvergence(max_iter) from exc

    a = vt.flatten().real
    h = A @ a
    if normalized:
        h /= h.sum()
        a /= a.sum()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return hubs, authorities


def _hits_python(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True):
    if isinstance(G, nx.MultiGraph | nx.MultiDiGraph):
        raise Exception("hits() not defined for graphs with multiedges.")
    if len(G) == 0:
        return {}, {}
    # choose fixed starting vector if not given
    if nstart is None:
        h = dict.fromkeys(G, 1.0 / G.number_of_nodes())
    else:
        h = nstart
        # normalize starting vector
        s = 1.0 / sum(h.values())
        for k in h:
            h[k] *= s
    for _ in range(max_iter):  # power iteration: make up to max_iter iterations
        hlast = h
        h = dict.fromkeys(hlast.keys(), 0)
        a = dict.fromkeys(hlast.keys(), 0)
        # this "matrix multiply" looks odd because it is
        # doing a left multiply a^T=hlast^T*G
        for n in h:
            for nbr in G[n]:
                a[nbr] += hlast[n] * G[n][nbr].get("weight", 1)
        # now multiply h=Ga
        for n in h:
            for nbr in G[n]:
                h[n] += a[nbr] * G[n][nbr].get("weight", 1)
        # normalize vector
        s = 1.0 / max(h.values())
        for n in h:
            h[n] *= s
        # normalize vector
        s = 1.0 / max(a.values())
        for n in a:
            a[n] *= s
        # check convergence, l1 norm
        err = sum(abs(h[n] - hlast[n]) for n in h)
        if err < tol:
            break
    else:
        raise nx.PowerIterationFailedConvergence(max_iter)
    if normalized:
        s = 1.0 / sum(a.values())
        for n in a:
            a[n] *= s
        s = 1.0 / sum(h.values())
        for n in h:
            h[n] *= s
    return h, a


def _hits_numpy(G, normalized=True):
    """Returns HITS hubs and authorities values for nodes.

    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    normalized : bool (default=True)
       Normalize results by the sum of all of the values.

    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.

    Examples
    --------
    >>> G = nx.path_graph(4)

    The `hubs` and `authorities` are given by the eigenvectors corresponding to the
    maximum eigenvalues of the hubs_matrix and the authority_matrix, respectively.

    The ``hubs`` and ``authority`` matrices are computed from the adjacency
    matrix:

    >>> adj_ary = nx.to_numpy_array(G)
    >>> hubs_matrix = adj_ary @ adj_ary.T
    >>> authority_matrix = adj_ary.T @ adj_ary

    `_hits_numpy` maps the eigenvector corresponding to the maximum eigenvalue
    of the respective matrices to the nodes in `G`:

    >>> from networkx.algorithms.link_analysis.hits_alg import _hits_numpy
    >>> hubs, authority = _hits_numpy(G)

    Notes
    -----
    The eigenvector calculation uses NumPy's interface to LAPACK.

    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-32, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    import numpy as np

    if len(G) == 0:
        return {}, {}
    adj_ary = nx.to_numpy_array(G)
    # Hub matrix
    H = adj_ary @ adj_ary.T
    e, ev = np.linalg.eig(H)
    h = ev[:, np.argmax(e)]  # eigenvector corresponding to the maximum eigenvalue
    # Authority matrix
    A = adj_ary.T @ adj_ary
    e, ev = np.linalg.eig(A)
    a = ev[:, np.argmax(e)]  # eigenvector corresponding to the maximum eigenvalue
    if normalized:
        h /= h.sum()
        a /= a.sum()
    else:
        h /= h.max()
        a /= a.max()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return hubs, authorities


def _hits_scipy(G, max_iter=100, tol=1.0e-6, nstart=None, normalized=True):
    """Returns HITS hubs and authorities values for nodes.


    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    max_iter : integer, optional
      Maximum number of iterations in power method.

    tol : float, optional
      Error tolerance used to check convergence in power method iteration.

    nstart : dictionary, optional
      Starting value of each node for power method iteration.

    normalized : bool (default=True)
       Normalize results by the sum of all of the values.

    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.

    Examples
    --------
    >>> from networkx.algorithms.link_analysis.hits_alg import _hits_scipy
    >>> G = nx.path_graph(4)
    >>> h, a = _hits_scipy(G)

    Notes
    -----
    This implementation uses SciPy sparse matrices.

    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-632, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    import numpy as np

    if len(G) == 0:
        return {}, {}
    A = nx.to_scipy_sparse_array(G, nodelist=list(G))
    (n, _) = A.shape  # should be square
    ATA = A.T @ A  # authority matrix
    # choose fixed starting vector if not given
    if nstart is None:
        x = np.ones((n, 1)) / n
    else:
        x = np.array([nstart.get(n, 0) for n in list(G)], dtype=float)
        x /= x.sum()

    # power iteration on authority matrix
    i = 0
    while True:
        xlast = x
        x = ATA @ x
        x /= x.max()
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < tol:
            break
        if i > max_iter:
            raise nx.PowerIterationFailedConvergence(max_iter)
        i += 1

    a = x.flatten()
    h = A @ a
    if normalized:
        h /= h.sum()
        a /= a.sum()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return hubs, authorities
<|endoftext|>"""PageRank analysis of graph structure."""

from warnings import warn

import networkx as nx

__all__ = ["pagerank", "google_matrix"]


@nx._dispatchable(edge_attrs="weight")
def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    """Returns the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specified, a nodes personalization value will be zero.
      By default, a uniform distribution is used.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.
      The iteration will stop after a tolerance of ``len(G) * tol`` is reached.

    nstart : dictionary, optional
      Starting value of PageRank iteration for each node.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified). This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.


    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value

    Examples
    --------
    >>> G = nx.DiGraph(nx.path_graph(4))
    >>> pr = nx.pagerank(G, alpha=0.9)

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop after
    an error tolerance of ``len(G) * tol`` has been reached. If the
    number of iterations exceed `max_iter`, a
    :exc:`networkx.exception.PowerIterationFailedConvergence` exception
    is raised.

    The PageRank algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs by converting each edge in the
    directed graph to two edges.

    See Also
    --------
    google_matrix

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
       The PageRank citation ranking: Bringing order to the Web. 1999
       http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf

    """
    return _pagerank_scipy(
        G, alpha, personalization, max_iter, tol, nstart, weight, dangling
    )


def _pagerank_python(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    if len(G) == 0:
        return {}

    D = G.to_directed()

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = sum(nstart.values())
        x = {k: v / s for k, v in nstart.items()}

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        s = sum(personalization.values())
        p = {k: v / s for k, v in personalization.items()}

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        s = sum(dangling.values())
        dangling_weights = {k: v / s for k, v in dangling.items()}
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for _, nbr, wt in W.edges(n, data=weight):
                x[nbr] += alpha * xlast[n] * wt
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        # check convergence, l1 norm
        err = sum(abs(x[n] - xlast[n]) for n in x)
        if err < N * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)


@nx._dispatchable(edge_attrs="weight")
def google_matrix(
    G, alpha=0.85, personalization=None, nodelist=None, weight="weight", dangling=None
):
    """Returns the Google matrix of the graph.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float
      The damping factor.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specified, a nodes personalization value will be zero.
      By default, a uniform distribution is used.

    nodelist : list, optional
      The rows and columns are ordered according to the nodes in nodelist.
      If nodelist is None, then the ordering is produced by G.nodes().

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes below). It may be common to have the dangling dict to
      be the same as the personalization dict.

    Returns
    -------
    A : 2D NumPy ndarray
       Google matrix of the graph

    Notes
    -----
    The array returned represents the transition matrix that describes the
    Markov chain used in PageRank. For PageRank to converge to a unique
    solution (i.e., a unique stationary distribution in a Markov chain), the
    transition matrix must be irreducible. In other words, it must be that
    there exists a path between every pair of nodes in the graph, or else there
    is the potential of "rank sinks."

    This implementation works with Multi(Di)Graphs. For multigraphs the
    weight between two nodes is set to be the sum of all edge weights
    between those nodes.

    See Also
    --------
    pagerank
    """
    import numpy as np

    if nodelist is None:
        nodelist = list(G)

    A = nx.to_numpy_array(G, nodelist=nodelist, weight=weight)
    N = len(G)
    if N == 0:
        return A

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    dangling_nodes = np.where(A.sum(axis=1) == 0)[0]

    # Assign dangling_weights to any dangling nodes (nodes with no out links)
    A[dangling_nodes] = dangling_weights

    A /= A.sum(axis=1)[:, np.newaxis]  # Normalize rows to sum to 1

    return alpha * A + (1 - alpha) * p


def _pagerank_numpy(
    G, alpha=0.85, personalization=None, weight="weight", dangling=None
):
    """Returns the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specified, a nodes personalization value will be zero.
      By default, a uniform distribution is used.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value.

    Examples
    --------
    >>> from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_numpy
    >>> G = nx.DiGraph(nx.path_graph(4))
    >>> pr = _pagerank_numpy(G, alpha=0.9)

    Notes
    -----
    The eigenvector calculation uses NumPy's interface to the LAPACK
    eigenvalue solvers.  This will be the fastest and most accurate
    for small graphs.

    This implementation works with Multi(Di)Graphs. For multigraphs the
    weight between two nodes is set to be the sum of all edge weights
    between those nodes.

    See Also
    --------
    pagerank, google_matrix

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
       The PageRank citation ranking: Bringing order to the Web. 1999
       http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf
    """
    import numpy as np

    if len(G) == 0:
        return {}
    M = google_matrix(
        G, alpha, personalization=personalization, weight=weight, dangling=dangling
    )
    # use numpy LAPACK solver
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    ind = np.argmax(eigenvalues)
    # eigenvector of largest eigenvalue is at ind, normalized
    largest = np.array(eigenvectors[:, ind]).flatten().real
    norm = largest.sum()
    return dict(zip(G, map(float, largest / norm)))


def _pagerank_scipy(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    """Returns the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specified, a nodes personalization value will be zero.
      By default, a uniform distribution is used.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.
      The iteration will stop after a tolerance of ``len(G) * tol`` is reached.

    nstart : dictionary, optional
      Starting value of PageRank iteration for each node.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value

    Examples
    --------
    >>> from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_scipy
    >>> G = nx.DiGraph(nx.path_graph(4))
    >>> pr = _pagerank_scipy(G, alpha=0.9)

    Notes
    -----
    The eigenvector calculation uses power iteration with a SciPy
    sparse matrix representation.

    This implementation works with Multi(Di)Graphs. For multigraphs the
    weight between two nodes is set to be the sum of all edge weights
    between those nodes.

    See Also
    --------
    pagerank

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
       The PageRank citation ranking: Bringing order to the Web. 1999
       http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf
    """
    import numpy as np
    import scipy as sp

    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    # TODO: csr_array
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    # initial vector
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x /= x.sum()

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x @ A + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)
<|endoftext|>"""Copyright (c) 2015 – Thomson Licensing, SAS

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the
disclaimer below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of Thomson Licensing, or Technicolor, nor the names
of its contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
GRANTED BY THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import networkx as nx
from networkx.utils import not_implemented_for

# Authors: Erwan Le Merrer (erwan.lemerrer@technicolor.com)

__all__ = ["second_order_centrality"]


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def second_order_centrality(G, weight="weight"):
    """Compute the second order centrality for nodes of G.

    The second order centrality of a given node is the standard deviation of
    the return times to that node of a perpetual random walk on G:

    Parameters
    ----------
    G : graph
      A NetworkX connected and undirected graph.

    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.

    Returns
    -------
    nodes : dictionary
       Dictionary keyed by node with second order centrality as the value.

    Examples
    --------
    >>> G = nx.star_graph(10)
    >>> soc = nx.second_order_centrality(G)
    >>> print(sorted(soc.items(), key=lambda x: x[1])[0][0])  # pick first id
    0

    Raises
    ------
    NetworkXException
        If the graph G is empty, non connected or has negative weights.

    See Also
    --------
    betweenness_centrality

    Notes
    -----
    Lower values of second order centrality indicate higher centrality.

    The algorithm is from Kermarrec, Le Merrer, Sericola and Trédan [1]_.

    This code implements the analytical version of the algorithm, i.e.,
    there is no simulation of a random walk process involved. The random walk
    is here unbiased (corresponding to eq 6 of the paper [1]_), thus the
    centrality values are the standard deviations for random walk return times
    on the transformed input graph G (equal in-degree at each nodes by adding
    self-loops).

    Complexity of this implementation, made to run locally on a single machine,
    is O(n^3), with n the size of G, which makes it viable only for small
    graphs.

    References
    ----------
    .. [1] Anne-Marie Kermarrec, Erwan Le Merrer, Bruno Sericola, Gilles Trédan
       "Second order centrality: Distributed assessment of nodes criticity in
       complex networks", Elsevier Computer Communications 34(5):619-628, 2011.
    """
    import numpy as np

    n = len(G)

    if n == 0:
        raise nx.NetworkXException("Empty graph.")
    if not nx.is_connected(G):
        raise nx.NetworkXException("Non connected graph.")
    if any(d.get(weight, 0) < 0 for u, v, d in G.edges(data=True)):
        raise nx.NetworkXException("Graph has negative edge weights.")

    # balancing G for Metropolis-Hastings random walks
    G = nx.DiGraph(G)
    in_deg = dict(G.in_degree(weight=weight))
    d_max = max(in_deg.values())
    for i, deg in in_deg.items():
        if deg < d_max:
            G.add_edge(i, i, weight=d_max - deg)

    P = nx.to_numpy_array(G)
    P /= P.sum(axis=1)[:, np.newaxis]  # to transition probability matrix

    def _Qj(P, j):
        P = P.copy()
        P[:, j] = 0
        return P

    M = np.empty([n, n])

    for i in range(n):
        M[:, i] = np.linalg.solve(
            np.identity(n) - _Qj(P, i), np.ones([n, 1])[:, 0]
        )  # eq 3

    return dict(
        zip(
            G.nodes,
            (float(np.sqrt(2 * np.sum(M[:, i]) - n * (n + 1))) for i in range(n)),
        )
    )  # eq 6
<|endoftext|>"""Algorithm to select influential nodes in a graph using VoteRank."""

import networkx as nx

__all__ = ["voterank"]


@nx._dispatchable
def voterank(G, number_of_nodes=None):
    """Select a list of influential nodes in a graph using VoteRank algorithm

    VoteRank [1]_ computes a ranking of the nodes in a graph G based on a
    voting scheme. With VoteRank, all nodes vote for each of its in-neighbors
    and the node with the highest votes is elected iteratively. The voting
    ability of out-neighbors of elected nodes is decreased in subsequent turns.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    number_of_nodes : integer, optional
        Number of ranked nodes to extract (default all nodes).

    Returns
    -------
    voterank : list
        Ordered list of computed seeds.
        Only nodes with positive number of votes are returned.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 4)])
    >>> nx.voterank(G)
    [0, 1]

    The algorithm can be used both for undirected and directed graphs.
    However, the directed version is different in two ways:
    (i) nodes only vote for their in-neighbors and
    (ii) only the voting ability of elected node and its out-neighbors are updated:

    >>> G = nx.DiGraph([(0, 1), (2, 1), (2, 3), (3, 4)])
    >>> nx.voterank(G)
    [2, 3]

    Notes
    -----
    Each edge is treated independently in case of multigraphs.

    References
    ----------
    .. [1] Zhang, J.-X. et al. (2016).
        Identifying a set of influential spreaders in complex networks.
        Sci. Rep. 6, 27823; doi: 10.1038/srep27823.
    """
    influential_nodes = []
    vote_rank = {}
    if len(G) == 0:
        return influential_nodes
    if number_of_nodes is None or number_of_nodes > len(G):
        number_of_nodes = len(G)
    if G.is_directed():
        # For directed graphs compute average out-degree
        avgDegree = sum(deg for _, deg in G.out_degree()) / len(G)
    else:
        # For undirected graphs compute average degree
        avgDegree = sum(deg for _, deg in G.degree()) / len(G)
    # step 1 - initiate all nodes to (0,1) (score, voting ability)
    for n in G.nodes():
        vote_rank[n] = [0, 1]
    # Repeat steps 1b to 4 until num_seeds are elected.
    for _ in range(number_of_nodes):
        # step 1b - reset rank
        for n in G.nodes():
            vote_rank[n][0] = 0
        # step 2 - vote
        for n, nbr in G.edges():
            # In directed graphs nodes only vote for their in-neighbors
            vote_rank[n][0] += vote_rank[nbr][1]
            if not G.is_directed():
                vote_rank[nbr][0] += vote_rank[n][1]
        for n in influential_nodes:
            vote_rank[n][0] = 0
        # step 3 - select top node
        n = max(G.nodes, key=lambda x: vote_rank[x][0])
        if vote_rank[n][0] == 0:
            return influential_nodes
        influential_nodes.append(n)
        # weaken the selected node
        vote_rank[n] = [0, 0]
        # step 4 - update voterank properties
        for _, nbr in G.edges(n):
            vote_rank[nbr][1] -= 1 / avgDegree
            vote_rank[nbr][1] = max(vote_rank[nbr][1], 0)
    return influential_nodes
<|endoftext|>"""Trophic levels"""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["trophic_levels", "trophic_differences", "trophic_incoherence_parameter"]


@not_implemented_for("undirected")
@nx._dispatchable(edge_attrs="weight")
def trophic_levels(G, weight="weight"):
    r"""Compute the trophic levels of nodes.

    The trophic level of a node $i$ is

    .. math::

        s_i = 1 + \frac{1}{k^{in}_i} \sum_{j} a_{ij} s_j

    where $k^{in}_i$ is the in-degree of i

    .. math::

        k^{in}_i = \sum_{j} a_{ij}

    and nodes with $k^{in}_i = 0$ have $s_i = 1$ by convention.

    These are calculated using the method outlined in Levine [1]_.

    Parameters
    ----------
    G : DiGraph
        A directed networkx graph

    Returns
    -------
    nodes : dict
        Dictionary of nodes with trophic level as the value.

    References
    ----------
    .. [1] Stephen Levine (1980) J. theor. Biol. 83, 195-207
    """
    import numpy as np

    # find adjacency matrix
    a = nx.adjacency_matrix(G, weight=weight).T.toarray()

    # drop rows/columns where in-degree is zero
    rowsum = np.sum(a, axis=1)
    p = a[rowsum != 0][:, rowsum != 0]
    # normalise so sum of in-degree weights is 1 along each row
    p = p / rowsum[rowsum != 0][:, np.newaxis]

    # calculate trophic levels
    nn = p.shape[0]
    i = np.eye(nn)
    try:
        n = np.linalg.inv(i - p)
    except np.linalg.LinAlgError as err:
        # LinAlgError is raised when there is a non-basal node
        msg = (
            "Trophic levels are only defined for graphs where every "
            + "node has a path from a basal node (basal nodes are nodes "
            + "with no incoming edges)."
        )
        raise nx.NetworkXError(msg) from err
    y = n.sum(axis=1) + 1

    levels = {}

    # all nodes with in-degree zero have trophic level == 1
    zero_node_ids = (node_id for node_id, degree in G.in_degree if degree == 0)
    for node_id in zero_node_ids:
        levels[node_id] = 1

    # all other nodes have levels as calculated
    nonzero_node_ids = (node_id for node_id, degree in G.in_degree if degree != 0)
    for i, node_id in enumerate(nonzero_node_ids):
        levels[node_id] = y.item(i)

    return levels


@not_implemented_for("undirected")
@nx._dispatchable(edge_attrs="weight")
def trophic_differences(G, weight="weight"):
    r"""Compute the trophic differences of the edges of a directed graph.

    The trophic difference $x_ij$ for each edge is defined in Johnson et al.
    [1]_ as:

    .. math::
        x_ij = s_j - s_i

    Where $s_i$ is the trophic level of node $i$.

    Parameters
    ----------
    G : DiGraph
        A directed networkx graph

    Returns
    -------
    diffs : dict
        Dictionary of edges with trophic differences as the value.

    References
    ----------
    .. [1] Samuel Johnson, Virginia Dominguez-Garcia, Luca Donetti, Miguel A.
        Munoz (2014) PNAS "Trophic coherence determines food-web stability"
    """
    levels = trophic_levels(G, weight=weight)
    diffs = {}
    for u, v in G.edges:
        diffs[(u, v)] = levels[v] - levels[u]
    return diffs


@not_implemented_for("undirected")
@nx._dispatchable(edge_attrs="weight")
def trophic_incoherence_parameter(G, weight="weight", cannibalism=False):
    r"""Compute the trophic incoherence parameter of a graph.

    Trophic coherence is defined as the homogeneity of the distribution of
    trophic distances: the more similar, the more coherent. This is measured by
    the standard deviation of the trophic differences and referred to as the
    trophic incoherence parameter $q$ by [1].

    Parameters
    ----------
    G : DiGraph
        A directed networkx graph

    cannibalism: Boolean
        If set to False, self edges are not considered in the calculation

    Returns
    -------
    trophic_incoherence_parameter : float
        The trophic coherence of a graph

    References
    ----------
    .. [1] Samuel Johnson, Virginia Dominguez-Garcia, Luca Donetti, Miguel A.
        Munoz (2014) PNAS "Trophic coherence determines food-web stability"
    """
    import numpy as np

    if cannibalism:
        diffs = trophic_differences(G, weight=weight)
    else:
        # If no cannibalism, remove self-edges
        self_loops = list(nx.selfloop_edges(G))
        if self_loops:
            # Make a copy so we do not change G's edges in memory
            G_2 = G.copy()
            G_2.remove_edges_from(self_loops)
        else:
            # Avoid copy otherwise
            G_2 = G
        diffs = trophic_differences(G_2, weight=weight)
    return float(np.std(list(diffs.values())))
<|endoftext|>"""Betweenness centrality measures for subsets of nodes."""

import networkx as nx
from networkx.algorithms.centrality.betweenness import (
    _add_edge_keys,
)
from networkx.algorithms.centrality.betweenness import (
    _single_source_dijkstra_path_basic as dijkstra,
)
from networkx.algorithms.centrality.betweenness import (
    _single_source_shortest_path_basic as shortest_path,
)

__all__ = [
    "betweenness_centrality_subset",
    "edge_betweenness_centrality_subset",
]


@nx._dispatchable(edge_attrs="weight")
def betweenness_centrality_subset(G, sources, targets, normalized=False, weight=None):
    r"""Compute betweenness centrality for a subset of nodes.

    .. math::

       c_B(v) =\sum_{s\in S, t \in T} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where $S$ is the set of sources, $T$ is the set of targets,
    $\sigma(s, t)$ is the number of shortest $(s, t)$-paths,
    and $\sigma(s, t|v)$ is the number of those paths
    passing through some  node $v$ other than $s, t$.
    If $s = t$, $\sigma(s, t) = 1$,
    and if $v \in {s, t}$, $\sigma(s, t|v) = 0$ [2]_.


    Parameters
    ----------
    G : graph
      A NetworkX graph.

    sources: list of nodes
      Nodes to use as sources for shortest paths in betweenness

    targets: list of nodes
      Nodes to use as targets for shortest paths in betweenness

    normalized : bool, optional
      If True the betweenness values are normalized by $2/((n-1)(n-2))$
      for graphs, and $1/((n-1)(n-2))$ for directed graphs where $n$
      is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      Weights are used to calculate weighted shortest paths, so they are
      interpreted as distances.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.

    See Also
    --------
    edge_betweenness_centrality
    load_centrality

    Notes
    -----
    The basic algorithm is from [1]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    The normalization might seem a little strange but it is
    designed to make betweenness_centrality(G) be the same as
    betweenness_centrality_subset(G,sources=G.nodes(),targets=G.nodes()).

    The total number of paths between source and target is counted
    differently for directed and undirected graphs. Directed paths
    are easy to count. Undirected paths are tricky: should a path
    from "u" to "v" count as 1 undirected path or as 2 directed paths?

    For betweenness_centrality we report the number of undirected
    paths when G is undirected.

    For betweenness_centrality_subset the reporting is different.
    If the source and target subsets are the same, then we want
    to count undirected paths. But if the source and target subsets
    differ -- for example, if sources is {0} and targets is {1},
    then we are only counting the paths in one direction. They are
    undirected paths but we are counting them in a directed way.
    To count them as undirected paths, each should count as half a path.

    References
    ----------
    .. [1] Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    """
    b = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    for s in sources:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma, _ = shortest_path(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma, _ = dijkstra(G, s, weight)
        b = _accumulate_subset(b, S, P, sigma, s, targets)
    b = _rescale(b, len(G), normalized=normalized, directed=G.is_directed())
    return b


@nx._dispatchable(edge_attrs="weight")
def edge_betweenness_centrality_subset(
    G, sources, targets, normalized=False, weight=None
):
    r"""Compute betweenness centrality for edges for a subset of nodes.

    .. math::

       c_B(v) =\sum_{s\in S,t \in T} \frac{\sigma(s, t|e)}{\sigma(s, t)}

    where $S$ is the set of sources, $T$ is the set of targets,
    $\sigma(s, t)$ is the number of shortest $(s, t)$-paths,
    and $\sigma(s, t|e)$ is the number of those paths
    passing through edge $e$ [2]_.

    Parameters
    ----------
    G : graph
      A networkx graph.

    sources: list of nodes
      Nodes to use as sources for shortest paths in betweenness

    targets: list of nodes
      Nodes to use as targets for shortest paths in betweenness

    normalized : bool, optional
      If True the betweenness values are normalized by `2/(n(n-1))`
      for graphs, and `1/(n(n-1))` for directed graphs where `n`
      is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      Weights are used to calculate weighted shortest paths, so they are
      interpreted as distances.

    Returns
    -------
    edges : dictionary
       Dictionary of edges with Betweenness centrality as the value.

    See Also
    --------
    betweenness_centrality
    edge_load

    Notes
    -----
    The basic algorithm is from [1]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    The normalization might seem a little strange but it is the same
    as in edge_betweenness_centrality() and is designed to make
    edge_betweenness_centrality(G) be the same as
    edge_betweenness_centrality_subset(G,sources=G.nodes(),targets=G.nodes()).

    References
    ----------
    .. [1] Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    """
    b = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    b.update(dict.fromkeys(G.edges(), 0.0))  # b[e] for e in G.edges()
    for s in sources:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma, _ = shortest_path(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma, _ = dijkstra(G, s, weight)
        b = _accumulate_edges_subset(b, S, P, sigma, s, targets)
    for n in G:  # remove nodes to only return edges
        del b[n]
    b = _rescale_e(b, len(G), normalized=normalized, directed=G.is_directed())
    if G.is_multigraph():
        b = _add_edge_keys(G, b, weight=weight)
    return b


def _accumulate_subset(betweenness, S, P, sigma, s, targets):
    delta = dict.fromkeys(S, 0.0)
    target_set = set(targets) - {s}
    while S:
        w = S.pop()
        if w in target_set:
            coeff = (delta[w] + 1.0) / sigma[w]
        else:
            coeff = delta[w] / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def _accumulate_edges_subset(betweenness, S, P, sigma, s, targets):
    """edge_betweenness_centrality_subset helper."""
    delta = dict.fromkeys(S, 0)
    target_set = set(targets)
    while S:
        w = S.pop()
        for v in P[w]:
            if w in target_set:
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
            else:
                c = delta[w] / len(P[w])
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def _rescale(betweenness, n, normalized, directed=False):
    """betweenness_centrality_subset helper."""
    if normalized:
        if n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1.0 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


def _rescale_e(betweenness, n, normalized, directed=False):
    """edge_betweenness_centrality_subset helper."""
    if normalized:
        if n <= 1:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1.0 / (n * (n - 1))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness
<|endoftext|>"""Degree centrality measures."""

import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = ["degree_centrality", "in_degree_centrality", "out_degree_centrality"]


@nx._dispatchable
def degree_centrality(G):
    """Compute the degree centrality for nodes.

    The degree centrality for a node v is the fraction of nodes it
    is connected to.

    Parameters
    ----------
    G : graph
      A networkx graph

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with degree centrality as the value.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    >>> nx.degree_centrality(G)
    {0: 1.0, 1: 1.0, 2: 0.6666666666666666, 3: 0.6666666666666666}

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality

    Notes
    -----
    The degree centrality values are normalized by dividing by the maximum
    possible degree in a simple graph n-1 where n is the number of nodes in G.

    For multigraphs or graphs with self loops the maximum degree might
    be higher than n-1 and values of degree centrality greater than 1
    are possible.
    """
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for n, d in G.degree()}
    return centrality


@not_implemented_for("undirected")
@nx._dispatchable
def in_degree_centrality(G):
    """Compute the in-degree centrality for nodes.

    The in-degree centrality for a node v is the fraction of nodes its
    incoming edges are connected to.

    Parameters
    ----------
    G : graph
        A NetworkX graph

    Returns
    -------
    nodes : dictionary
        Dictionary of nodes with in-degree centrality as values.

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    >>> nx.in_degree_centrality(G)
    {0: 0.0, 1: 0.3333333333333333, 2: 0.6666666666666666, 3: 0.6666666666666666}

    See Also
    --------
    degree_centrality, out_degree_centrality

    Notes
    -----
    The degree centrality values are normalized by dividing by the maximum
    possible degree in a simple graph n-1 where n is the number of nodes in G.

    For multigraphs or graphs with self loops the maximum degree might
    be higher than n-1 and values of degree centrality greater than 1
    are possible.
    """
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for n, d in G.in_degree()}
    return centrality


@not_implemented_for("undirected")
@nx._dispatchable
def out_degree_centrality(G):
    """Compute the out-degree centrality for nodes.

    The out-degree centrality for a node v is the fraction of nodes its
    outgoing edges are connected to.

    Parameters
    ----------
    G : graph
        A NetworkX graph

    Returns
    -------
    nodes : dictionary
        Dictionary of nodes with out-degree centrality as values.

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    >>> nx.out_degree_centrality(G)
    {0: 1.0, 1: 0.6666666666666666, 2: 0.0, 3: 0.0}

    See Also
    --------
    degree_centrality, in_degree_centrality

    Notes
    -----
    The degree centrality values are normalized by dividing by the maximum
    possible degree in a simple graph n-1 where n is the number of nodes in G.

    For multigraphs or graphs with self loops the maximum degree might
    be higher than n-1 and values of degree centrality greater than 1
    are possible.
    """
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for n, d in G.out_degree()}
    return centrality
<|endoftext|>"""Current-flow closeness centrality measures."""

import networkx as nx
from networkx.algorithms.centrality.flow_matrix import (
    CGInverseLaplacian,
    FullInverseLaplacian,
    SuperLUInverseLaplacian,
)
from networkx.utils import not_implemented_for, reverse_cuthill_mckee_ordering

__all__ = ["current_flow_closeness_centrality", "information_centrality"]


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def current_flow_closeness_centrality(G, weight=None, dtype=float, solver="lu"):
    """Compute current-flow closeness centrality for nodes.

    Current-flow closeness centrality is variant of closeness
    centrality based on effective resistance between nodes in
    a network. This metric is also known as information centrality.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      The weight reflects the capacity or the strength of the
      edge.

    dtype: data type (default=float)
      Default data type for internal matrices.
      Set to np.float32 for lower memory consumption.

    solver: string (default='lu')
       Type of linear solver to use for computing the flow matrix.
       Options are "full" (uses most memory), "lu" (recommended), and
       "cg" (uses least memory).

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with current flow closeness centrality as the value.

    See Also
    --------
    closeness_centrality

    Notes
    -----
    The algorithm is from Brandes [1]_.

    See also [2]_ for the original definition of information centrality.

    References
    ----------
    .. [1] Ulrik Brandes and Daniel Fleischer,
       Centrality Measures Based on Current Flow.
       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS '05).
       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.
       https://doi.org/10.1007/978-3-540-31856-9_44

    .. [2] Karen Stephenson and Marvin Zelen:
       Rethinking centrality: Methods and examples.
       Social Networks 11(1):1-37, 1989.
       https://doi.org/10.1016/0378-8733(89)90016-6
    """
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph not connected.")
    solvername = {
        "full": FullInverseLaplacian,
        "lu": SuperLUInverseLaplacian,
        "cg": CGInverseLaplacian,
    }
    N = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    # make a copy with integer labels according to rcm ordering
    # this could be done without a copy if we really wanted to
    H = nx.relabel_nodes(G, dict(zip(ordering, range(N))))
    betweenness = dict.fromkeys(H, 0.0)  # b[n]=0 for n in H
    N = H.number_of_nodes()
    L = nx.laplacian_matrix(H, nodelist=range(N), weight=weight).asformat("csc")
    L = L.astype(dtype)
    C2 = solvername[solver](L, width=1, dtype=dtype)  # initialize solver
    for v in H:
        col = C2.get_row(v)
        for w in H:
            betweenness[v] += col.item(v) - 2 * col.item(w)
            betweenness[w] += col.item(v)
    return {ordering[node]: 1 / value for node, value in betweenness.items()}


information_centrality = current_flow_closeness_centrality
<|endoftext|>"""Katz centrality."""

import math

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["katz_centrality", "katz_centrality_numpy"]


@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def katz_centrality(
    G,
    alpha=0.1,
    beta=1.0,
    max_iter=1000,
    tol=1.0e-6,
    nstart=None,
    normalized=True,
    weight=None,
):
    r"""Compute the Katz centrality for the nodes of the graph G.

    Katz centrality computes the centrality for a node based on the centrality
    of its neighbors. It is a generalization of the eigenvector centrality. The
    Katz centrality for node $i$ is

    .. math::

        x_i = \alpha \sum_{j} A_{ij} x_j + \beta,

    where $A$ is the adjacency matrix of graph G with eigenvalues $\lambda$.

    The parameter $\beta$ controls the initial centrality and

    .. math::

        \alpha < \frac{1}{\lambda_{\max}}.

    Katz centrality computes the relative influence of a node within a
    network by measuring the number of the immediate neighbors (first
    degree nodes) and also all other nodes in the network that connect
    to the node under consideration through these immediate neighbors.

    Extra weight can be provided to immediate neighbors through the
    parameter $\beta$.  Connections made with distant neighbors
    are, however, penalized by an attenuation factor $\alpha$ which
    should be strictly less than the inverse largest eigenvalue of the
    adjacency matrix in order for the Katz centrality to be computed
    correctly. More information is provided in [1]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    alpha : float, optional (default=0.1)
      Attenuation factor

    beta : scalar or dictionary, optional (default=1.0)
      Weight attributed to the immediate neighborhood. If not a scalar, the
      dictionary must have a value for every node.

    max_iter : integer, optional (default=1000)
      Maximum number of iterations in power method.

    tol : float, optional (default=1.0e-6)
      Error tolerance used to check convergence in power method iteration.

    nstart : dictionary, optional
      Starting value of Katz iteration for each node.

    normalized : bool, optional (default=True)
      If True normalize the resulting values.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      In this measure the weight is interpreted as the connection strength.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with Katz centrality as the value.

    Raises
    ------
    NetworkXError
       If the parameter `beta` is not a scalar but lacks a value for at least
       one node

    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    Examples
    --------
    >>> import math
    >>> G = nx.path_graph(4)
    >>> phi = (1 + math.sqrt(5)) / 2.0  # largest eigenvalue of adj matrix
    >>> centrality = nx.katz_centrality(G, 1 / phi - 0.01)
    >>> for n, c in sorted(centrality.items()):
    ...     print(f"{n} {c:.2f}")
    0 0.37
    1 0.60
    2 0.60
    3 0.37

    See Also
    --------
    katz_centrality_numpy
    eigenvector_centrality
    eigenvector_centrality_numpy
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Katz centrality was introduced by [2]_.

    This algorithm it uses the power method to find the eigenvector
    corresponding to the largest eigenvalue of the adjacency matrix of ``G``.
    The parameter ``alpha`` should be strictly less than the inverse of largest
    eigenvalue of the adjacency matrix for the algorithm to converge.
    You can use ``max(nx.adjacency_spectrum(G))`` to get $\lambda_{\max}$ the largest
    eigenvalue of the adjacency matrix.
    The iteration will stop after ``max_iter`` iterations or an error tolerance of
    ``number_of_nodes(G) * tol`` has been reached.

    For strongly connected graphs, as $\alpha \to 1/\lambda_{\max}$, and $\beta > 0$,
    Katz centrality approaches the results for eigenvector centrality.

    For directed graphs this finds "left" eigenvectors which corresponds
    to the in-edges in the graph. For out-edges Katz centrality,
    first reverse the graph with ``G.reverse()``.

    References
    ----------
    .. [1] Mark E. J. Newman:
       Networks: An Introduction.
       Oxford University Press, USA, 2010, p. 720.
    .. [2] Leo Katz:
       A New Status Index Derived from Sociometric Index.
       Psychometrika 18(1):39–43, 1953
       https://link.springer.com/content/pdf/10.1007/BF02289026.pdf
    """
    if len(G) == 0:
        return {}

    nnodes = G.number_of_nodes()

    if nstart is None:
        # choose starting vector with entries of 0
        x = {n: 0 for n in G}
    else:
        x = nstart

    try:
        b = dict.fromkeys(G, float(beta))
    except (TypeError, ValueError, AttributeError) as err:
        b = beta
        if set(beta) != set(G):
            raise nx.NetworkXError(
                "beta dictionary must have a value for every node"
            ) from err

    # make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast, 0)
        # do the multiplication y^T = Alpha * x^T A + Beta
        for n in x:
            for nbr in G[n]:
                x[nbr] += xlast[n] * G[n][nbr].get(weight, 1)
        for n in x:
            x[n] = alpha * x[n] + b[n]

        # check convergence
        error = sum(abs(x[n] - xlast[n]) for n in x)
        if error < nnodes * tol:
            if normalized:
                # normalize vector
                try:
                    s = 1.0 / math.hypot(*x.values())
                except ZeroDivisionError:
                    s = 1.0
            else:
                s = 1
            for n in x:
                x[n] *= s
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)


@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def katz_centrality_numpy(G, alpha=0.1, beta=1.0, normalized=True, weight=None):
    r"""Compute the Katz centrality for the graph G.

    Katz centrality computes the centrality for a node based on the centrality
    of its neighbors. It is a generalization of the eigenvector centrality. The
    Katz centrality for node $i$ is

    .. math::

        x_i = \alpha \sum_{j} A_{ij} x_j + \beta,

    where $A$ is the adjacency matrix of graph G with eigenvalues $\lambda$.

    The parameter $\beta$ controls the initial centrality and

    .. math::

        \alpha < \frac{1}{\lambda_{\max}}.

    Katz centrality computes the relative influence of a node within a
    network by measuring the number of the immediate neighbors (first
    degree nodes) and also all other nodes in the network that connect
    to the node under consideration through these immediate neighbors.

    Extra weight can be provided to immediate neighbors through the
    parameter $\beta$.  Connections made with distant neighbors
    are, however, penalized by an attenuation factor $\alpha$ which
    should be strictly less than the inverse largest eigenvalue of the
    adjacency matrix in order for the Katz centrality to be computed
    correctly. More information is provided in [1]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    alpha : float
      Attenuation factor

    beta : scalar or dictionary, optional (default=1.0)
      Weight attributed to the immediate neighborhood. If not a scalar the
      dictionary must have an value for every node.

    normalized : bool
      If True normalize the resulting values.

    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      In this measure the weight is interpreted as the connection strength.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with Katz centrality as the value.

    Raises
    ------
    NetworkXError
       If the parameter `beta` is not a scalar but lacks a value for at least
       one node

    Examples
    --------
    >>> import math
    >>> G = nx.path_graph(4)
    >>> phi = (1 + math.sqrt(5)) / 2.0  # largest eigenvalue of adj matrix
    >>> centrality = nx.katz_centrality_numpy(G, 1 / phi)
    >>> for n, c in sorted(centrality.items()):
    ...     print(f"{n} {c:.2f}")
    0 0.37
    1 0.60
    2 0.60
    3 0.37

    See Also
    --------
    katz_centrality
    eigenvector_centrality_numpy
    eigenvector_centrality
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Katz centrality was introduced by [2]_.

    This algorithm uses a direct linear solver to solve the above equation.
    The parameter ``alpha`` should be strictly less than the inverse of largest
    eigenvalue of the adjacency matrix for there to be a solution.
    You can use ``max(nx.adjacency_spectrum(G))`` to get $\lambda_{\max}$ the largest
    eigenvalue of the adjacency matrix.

    For strongly connected graphs, as $\alpha \to 1/\lambda_{\max}$, and $\beta > 0$,
    Katz centrality approaches the results for eigenvector centrality.

    For directed graphs this finds "left" eigenvectors which corresponds
    to the in-edges in the graph. For out-edges Katz centrality,
    first reverse the graph with ``G.reverse()``.

    References
    ----------
    .. [1] Mark E. J. Newman:
       Networks: An Introduction.
       Oxford University Press, USA, 2010, p. 173.
    .. [2] Leo Katz:
       A New Status Index Derived from Sociometric Index.
       Psychometrika 18(1):39–43, 1953
       https://link.springer.com/content/pdf/10.1007/BF02289026.pdf
    """
    import numpy as np

    if len(G) == 0:
        return {}
    try:
        nodelist = beta.keys()
        if set(nodelist) != set(G):
            raise nx.NetworkXError("beta dictionary must have a value for every node")
        b = np.array(list(beta.values()), dtype=float)
    except AttributeError:
        nodelist = list(G)
        try:
            b = np.ones((len(nodelist), 1)) * beta
        except (TypeError, ValueError, AttributeError) as err:
            raise nx.NetworkXError("beta must be a number") from err

    A = nx.adjacency_matrix(G, nodelist=nodelist, weight=weight).todense().T
    n = A.shape[0]
    centrality = np.linalg.solve(np.eye(n, n) - (alpha * A), b).squeeze()

    # Normalize: rely on truediv to cast to float, then tolist to make Python numbers
    norm = np.sign(sum(centrality)) * np.linalg.norm(centrality) if normalized else 1
    return dict(zip(nodelist, (centrality / norm).tolist()))
<|endoftext|>"""Functions for computing reaching centrality of a node or a graph."""

import networkx as nx
from networkx.utils import pairwise

__all__ = ["global_reaching_centrality", "local_reaching_centrality"]


def _average_weight(G, path, weight=None):
    """Returns the average weight of an edge in a weighted path.

    Parameters
    ----------
    G : graph
      A networkx graph.

    path: list
      A list of vertices that define the path.

    weight : None or string, optional (default=None)
      If None, edge weights are ignored.  Then the average weight of an edge
      is assumed to be the multiplicative inverse of the length of the path.
      Otherwise holds the name of the edge attribute used as weight.
    """
    path_length = len(path) - 1
    if path_length <= 0:
        return 0
    if weight is None:
        return 1 / path_length
    total_weight = sum(G.edges[i, j][weight] for i, j in pairwise(path))
    return total_weight / path_length


@nx._dispatchable(edge_attrs="weight")
def global_reaching_centrality(G, weight=None, normalized=True):
    """Returns the global reaching centrality of a directed graph.

    The *global reaching centrality* of a weighted directed graph is the
    average over all nodes of the difference between the local reaching
    centrality of the node and the greatest local reaching centrality of
    any node in the graph [1]_. For more information on the local
    reaching centrality, see :func:`local_reaching_centrality`.
    Informally, the local reaching centrality is the proportion of the
    graph that is reachable from the neighbors of the node.

    Parameters
    ----------
    G : DiGraph
        A networkx DiGraph.

    weight : None or string, optional (default=None)
        Attribute to use for edge weights. If ``None``, each edge weight
        is assumed to be one. A higher weight implies a stronger
        connection between nodes and a *shorter* path length.

    normalized : bool, optional (default=True)
        Whether to normalize the edge weights by the total sum of edge
        weights.

    Returns
    -------
    h : float
        The global reaching centrality of the graph.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2)
    >>> G.add_edge(1, 3)
    >>> nx.global_reaching_centrality(G)
    1.0
    >>> G.add_edge(3, 2)
    >>> nx.global_reaching_centrality(G)
    0.75

    See also
    --------
    local_reaching_centrality

    References
    ----------
    .. [1] Mones, Enys, Lilla Vicsek, and Tamás Vicsek.
           "Hierarchy Measure for Complex Networks."
           *PLoS ONE* 7.3 (2012): e33799.
           https://doi.org/10.1371/journal.pone.0033799
    """
    if nx.is_negatively_weighted(G, weight=weight):
        raise nx.NetworkXError("edge weights must be positive")
    total_weight = G.size(weight=weight)
    if total_weight <= 0:
        raise nx.NetworkXError("Size of G must be positive")
    # If provided, weights must be interpreted as connection strength
    # (so higher weights are more likely to be chosen). However, the
    # shortest path algorithms in NetworkX assume the provided "weight"
    # is actually a distance (so edges with higher weight are less
    # likely to be chosen). Therefore we need to invert the weights when
    # computing shortest paths.
    #
    # If weight is None, we leave it as-is so that the shortest path
    # algorithm can use a faster, unweighted algorithm.
    if weight is not None:

        def as_distance(u, v, d):
            return total_weight / d.get(weight, 1)

        shortest_paths = nx.shortest_path(G, weight=as_distance)
    else:
        shortest_paths = nx.shortest_path(G)

    centrality = local_reaching_centrality
    # TODO This can be trivially parallelized.
    lrc = [
        centrality(G, node, paths=paths, weight=weight, normalized=normalized)
        for node, paths in shortest_paths.items()
    ]

    max_lrc = max(lrc)
    return sum(max_lrc - c for c in lrc) / (len(G) - 1)


@nx._dispatchable(edge_attrs="weight")
def local_reaching_centrality(G, v, paths=None, weight=None, normalized=True):
    """Returns the local reaching centrality of a node in a directed
    graph.

    The *local reaching centrality* of a node in a directed graph is the
    proportion of other nodes reachable from that node [1]_.

    Parameters
    ----------
    G : DiGraph
        A NetworkX DiGraph.

    v : node
        A node in the directed graph `G`.

    paths : dictionary (default=None)
        If this is not `None` it must be a dictionary representation
        of single-source shortest paths, as computed by, for example,
        :func:`networkx.shortest_path` with source node `v`. Use this
        keyword argument if you intend to invoke this function many
        times but don't want the paths to be recomputed each time.

    weight : None or string, optional (default=None)
        Attribute to use for edge weights.  If `None`, each edge weight
        is assumed to be one. A higher weight implies a stronger
        connection between nodes and a *shorter* path length.

    normalized : bool, optional (default=True)
        Whether to normalize the edge weights by the total sum of edge
        weights.

    Returns
    -------
    h : float
        The local reaching centrality of the node ``v`` in the graph
        ``G``.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3)])
    >>> nx.local_reaching_centrality(G, 3)
    0.0
    >>> G.add_edge(3, 2)
    >>> nx.local_reaching_centrality(G, 3)
    0.5

    See also
    --------
    global_reaching_centrality

    References
    ----------
    .. [1] Mones, Enys, Lilla Vicsek, and Tamás Vicsek.
           "Hierarchy Measure for Complex Networks."
           *PLoS ONE* 7.3 (2012): e33799.
           https://doi.org/10.1371/journal.pone.0033799
    """
    # Corner case: graph with single node containing a self-loop
    if (total_weight := G.size(weight=weight)) > 0 and len(G) == 1:
        raise nx.NetworkXError(
            "local_reaching_centrality of a single node with self-loop not well-defined"
        )
    if paths is None:
        if nx.is_negatively_weighted(G, weight=weight):
            raise nx.NetworkXError("edge weights must be positive")
        if total_weight <= 0:
            raise nx.NetworkXError("Size of G must be positive")
        if weight is not None:
            # Interpret weights as lengths.
            def as_distance(u, v, d):
                return total_weight / d.get(weight, 1)

            paths = nx.shortest_path(G, source=v, weight=as_distance)
        else:
            paths = nx.shortest_path(G, source=v)
    # If the graph is unweighted, simply return the proportion of nodes
    # reachable from the source node ``v``.
    if weight is None and G.is_directed():
        return (len(paths) - 1) / (len(G) - 1)
    if normalized and weight is not None:
        norm = G.size(weight=weight) / G.size()
    else:
        norm = 1
    # TODO This can be trivially parallelized.
    avgw = (_average_weight(G, path, weight=weight) for path in paths.values())
    sum_avg_weight = sum(avgw) / norm
    return sum_avg_weight / (len(G) - 1)
<|endoftext|>from itertools import combinations

import networkx as nx

__all__ = ["dispersion"]


@nx._dispatchable
def dispersion(G, u=None, v=None, normalized=True, alpha=1.0, b=0.0, c=0.0):
    r"""Calculate dispersion between `u` and `v` in `G`.

    A link between two actors (`u` and `v`) has a high dispersion when their
    mutual ties (`s` and `t`) are not well connected with each other.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    u : node, optional
        The source for the dispersion score (e.g. ego node of the network).
    v : node, optional
        The target of the dispersion score if specified.
    normalized : bool
        If True (default) normalize by the embeddedness of the nodes (u and v).
    alpha, b, c : float
        Parameters for the normalization procedure. When `normalized` is True,
        the dispersion value is normalized by::

            result = ((dispersion + b) ** alpha) / (embeddedness + c)

        as long as the denominator is nonzero.

    Returns
    -------
    nodes : dictionary
        If u (v) is specified, returns a dictionary of nodes with dispersion
        score for all "target" ("source") nodes. If neither u nor v is
        specified, returns a dictionary of dictionaries for all nodes 'u' in the
        graph with a dispersion score for each node 'v'.

    Notes
    -----
    This implementation follows Lars Backstrom and Jon Kleinberg [1]_. Typical
    usage would be to run dispersion on the ego network $G_u$ if $u$ were
    specified.  Running :func:`dispersion` with neither $u$ nor $v$ specified
    can take some time to complete.

    References
    ----------
    .. [1] Romantic Partnerships and the Dispersion of Social Ties:
        A Network Analysis of Relationship Status on Facebook.
        Lars Backstrom, Jon Kleinberg.
        https://arxiv.org/pdf/1310.6753v1.pdf

    """

    def _dispersion(G_u, u, v):
        """dispersion for all nodes 'v' in a ego network G_u of node 'u'"""
        u_nbrs = set(G_u[u])
        ST = {n for n in G_u[v] if n in u_nbrs}
        set_uv = {u, v}
        # all possible ties of connections that u and b share
        possib = combinations(ST, 2)
        total = 0
        for s, t in possib:
            # neighbors of s that are in G_u, not including u and v
            nbrs_s = u_nbrs.intersection(G_u[s]) - set_uv
            # s and t are not directly connected
            if t not in nbrs_s:
                # s and t do not share a connection
                if nbrs_s.isdisjoint(G_u[t]):
                    # tick for disp(u, v)
                    total += 1
        # neighbors that u and v share
        embeddedness = len(ST)

        dispersion_val = total
        if normalized:
            dispersion_val = (total + b) ** alpha
            if embeddedness + c != 0:
                dispersion_val /= embeddedness + c

        return dispersion_val

    if u is None:
        # v and u are not specified
        if v is None:
            results = {n: {} for n in G}
            for u in G:
                for v in G[u]:
                    results[u][v] = _dispersion(G, u, v)
        # u is not specified, but v is
        else:
            results = dict.fromkeys(G[v], {})
            for u in G[v]:
                results[u] = _dispersion(G, v, u)
    else:
        # u is specified with no target v
        if v is None:
            results = dict.fromkeys(G[u], {})
            for v in G[u]:
                results[v] = _dispersion(G, u, v)
        # both u and v are specified
        else:
            results = _dispersion(G, u, v)

    return results
<|endoftext|>"""Betweenness centrality measures."""

from collections import deque
from heapq import heappop, heappush
from itertools import count

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import py_random_state
from networkx.utils.decorators import not_implemented_for

__all__ = ["betweenness_centrality", "edge_betweenness_centrality"]


@py_random_state(5)
@nx._dispatchable(edge_attrs="weight")
def betweenness_centrality(
    G, k=None, normalized=True, weight=None, endpoints=False, seed=None
):
    r"""Compute the shortest-path betweenness centrality for nodes.

    Betweenness centrality of a node $v$ is the sum of the
    fraction of all-pairs shortest paths that pass through $v$

    .. math::

       c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths,  and $\sigma(s, t|v)$ is the number of
    those paths  passing through some  node $v$ other than $s, t$.
    If $s = t$, $\sigma(s, t) = 1$, and if $v \in {s, t}$,
    $\sigma(s, t|v) = 0$ [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    k : int, optional (default=None)
      If k is not None use k node samples to estimate betweenness.
      The value of k <= n where n is the number of nodes in the graph.
      Higher values give better approximation.

    normalized : bool, optional
      If True the betweenness values are normalized by `2/((n-1)(n-2))`
      for graphs, and `1/((n-1)(n-2))` for directed graphs where `n`
      is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      Weights are used to calculate weighted shortest paths, so they are
      interpreted as distances.

    endpoints : bool, optional
      If True include the endpoints in the shortest path counts.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Note that this is only used if k is not None.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.

    See Also
    --------
    edge_betweenness_centrality
    load_centrality

    Notes
    -----
    The algorithm is from Ulrik Brandes [1]_.
    See [4]_ for the original first published version and [2]_ for details on
    algorithms for variations and related metrics.

    For approximate betweenness calculations set k=#samples to use
    k nodes ("pivots") to estimate the betweenness values. For an estimate
    of the number of pivots needed see [3]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    The total number of paths between source and target is counted
    differently for directed and undirected graphs. Directed paths
    are easy to count. Undirected paths are tricky: should a path
    from "u" to "v" count as 1 undirected path or as 2 directed paths?

    For betweenness_centrality we report the number of undirected
    paths when G is undirected.

    For betweenness_centrality_subset the reporting is different.
    If the source and target subsets are the same, then we want
    to count undirected paths. But if the source and target subsets
    differ -- for example, if sources is {0} and targets is {1},
    then we are only counting the paths in one direction. They are
    undirected paths but we are counting them in a directed way.
    To count them as undirected paths, each should count as half a path.

    This algorithm is not guaranteed to be correct if edge weights
    are floating point numbers. As a workaround you can use integer
    numbers by multiplying the relevant edge attributes by a convenient
    constant factor (eg 100) and converting to integers.

    References
    ----------
    .. [1] Ulrik Brandes:
       A Faster Algorithm for Betweenness Centrality.
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes:
       On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    .. [3] Ulrik Brandes and Christian Pich:
       Centrality Estimation in Large Networks.
       International Journal of Bifurcation and Chaos 17(7):2303-2318, 2007.
       https://dx.doi.org/10.1142/S0218127407018403
    .. [4] Linton C. Freeman:
       A set of measures of centrality based on betweenness.
       Sociometry 40: 35–41, 1977
       https://doi.org/10.2307/3033543
    """
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    if k is None:
        nodes = G
    else:
        nodes = seed.sample(list(G.nodes()), k)
    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma, _ = _single_source_shortest_path_basic(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma, _ = _single_source_dijkstra_path_basic(G, s, weight)
        # accumulation
        if endpoints:
            betweenness, _ = _accumulate_endpoints(betweenness, S, P, sigma, s)
        else:
            betweenness, _ = _accumulate_basic(betweenness, S, P, sigma, s)
    # rescaling
    betweenness = _rescale(
        betweenness,
        len(G),
        normalized=normalized,
        directed=G.is_directed(),
        k=k,
        endpoints=endpoints,
    )
    return betweenness


@py_random_state(4)
@nx._dispatchable(edge_attrs="weight")
def edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None):
    r"""Compute betweenness centrality for edges.

    Betweenness centrality of an edge $e$ is the sum of the
    fraction of all-pairs shortest paths that pass through $e$

    .. math::

       c_B(e) =\sum_{s,t \in V} \frac{\sigma(s, t|e)}{\sigma(s, t)}

    where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\sigma(s, t|e)$ is the number of
    those paths passing through edge $e$ [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    k : int, optional (default=None)
      If k is not None use k node samples to estimate betweenness.
      The value of k <= n where n is the number of nodes in the graph.
      Higher values give better approximation.

    normalized : bool, optional
      If True the betweenness values are normalized by $2/(n(n-1))$
      for graphs, and $1/(n(n-1))$ for directed graphs where $n$
      is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      Weights are used to calculate weighted shortest paths, so they are
      interpreted as distances.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Note that this is only used if k is not None.

    Returns
    -------
    edges : dictionary
       Dictionary of edges with betweenness centrality as the value.

    See Also
    --------
    betweenness_centrality
    edge_load

    Notes
    -----
    The algorithm is from Ulrik Brandes [1]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    References
    ----------
    .. [1]  A Faster Algorithm for Betweenness Centrality. Ulrik Brandes,
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    """
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    # b[e]=0 for e in G.edges()
    betweenness.update(dict.fromkeys(G.edges(), 0.0))
    if k is None:
        nodes = G
    else:
        nodes = seed.sample(list(G.nodes()), k)
    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma, _ = _single_source_shortest_path_basic(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma, _ = _single_source_dijkstra_path_basic(G, s, weight)
        # accumulation
        betweenness = _accumulate_edges(betweenness, S, P, sigma, s)
    # rescaling
    for n in G:  # remove nodes to only return edges
        del betweenness[n]
    betweenness = _rescale_e(
        betweenness, len(G), normalized=normalized, directed=G.is_directed()
    )
    if G.is_multigraph():
        betweenness = _add_edge_keys(G, betweenness, weight=weight)
    return betweenness


# helpers for betweenness centrality


def _single_source_shortest_path_basic(G, s):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = deque([s])
    while Q:  # use BFS to find shortest paths
        v = Q.popleft()
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:  # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma, D


def _single_source_dijkstra_path_basic(G, s, weight):
    weight = _weight_function(G, weight)
    # modified from Eppstein
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []  # use Q as heap with (distance,node id) tuples
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue  # already searched this node.
        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + weight(v, w, edgedata)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
    return S, P, sigma, D


def _accumulate_basic(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness, delta


def _accumulate_endpoints(betweenness, S, P, sigma, s):
    betweenness[s] += len(S) - 1
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w] + 1
    return betweenness, delta


def _accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def _rescale(betweenness, n, normalized, directed=False, k=None, endpoints=False):
    if normalized:
        if endpoints:
            if n < 2:
                scale = None  # no normalization
            else:
                # Scale factor should include endpoint nodes
                scale = 1 / (n * (n - 1))
        elif n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


def _rescale_e(betweenness, n, normalized, directed=False, k=None):
    if normalized:
        if n <= 1:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / (n * (n - 1))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


@not_implemented_for("graph")
def _add_edge_keys(G, betweenness, weight=None):
    r"""Adds the corrected betweenness centrality (BC) values for multigraphs.

    Parameters
    ----------
    G : NetworkX graph.

    betweenness : dictionary
        Dictionary mapping adjacent node tuples to betweenness centrality values.

    weight : string or function
        See `_weight_function` for details. Defaults to `None`.

    Returns
    -------
    edges : dictionary
        The parameter `betweenness` including edges with keys and their
        betweenness centrality values.

    The BC value is divided among edges of equal weight.
    """
    _weight = _weight_function(G, weight)

    edge_bc = dict.fromkeys(G.edges, 0.0)
    for u, v in betweenness:
        d = G[u][v]
        wt = _weight(u, v, d)
        keys = [k for k in d if _weight(u, v, {k: d[k]}) == wt]
        bc = betweenness[(u, v)] / len(keys)
        for k in keys:
            edge_bc[(u, v, k)] = bc

    return edge_bc
<|endoftext|>"""Functions for computing eigenvector centrality."""

import math

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["eigenvector_centrality", "eigenvector_centrality_numpy"]


@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None, weight=None):
    r"""Compute the eigenvector centrality for the graph G.

    Eigenvector centrality computes the centrality for a node by adding
    the centrality of its predecessors. The centrality for node $i$ is the
    $i$-th element of a left eigenvector associated with the eigenvalue $\lambda$
    of maximum modulus that is positive. Such an eigenvector $x$ is
    defined up to a multiplicative constant by the equation

    .. math::

         \lambda x^T = x^T A,

    where $A$ is the adjacency matrix of the graph G. By definition of
    row-column product, the equation above is equivalent to

    .. math::

        \lambda x_i = \sum_{j\to i}x_j.

    That is, adding the eigenvector centralities of the predecessors of
    $i$ one obtains the eigenvector centrality of $i$ multiplied by
    $\lambda$. In the case of undirected graphs, $x$ also solves the familiar
    right-eigenvector equation $Ax = \lambda x$.

    By virtue of the Perron–Frobenius theorem [1]_, if G is strongly
    connected there is a unique eigenvector $x$, and all its entries
    are strictly positive.

    If G is not strongly connected there might be several left
    eigenvectors associated with $\lambda$, and some of their elements
    might be zero.

    Parameters
    ----------
    G : graph
      A networkx graph.

    max_iter : integer, optional (default=100)
      Maximum number of power iterations.

    tol : float, optional (default=1.0e-6)
      Error tolerance (in Euclidean norm) used to check convergence in
      power iteration.

    nstart : dictionary, optional (default=None)
      Starting value of power iteration for each node. Must have a nonzero
      projection on the desired eigenvector for the power method to converge.
      If None, this implementation uses an all-ones vector, which is a safe
      choice.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal. Otherwise holds the
      name of the edge attribute used as weight. In this measure the
      weight is interpreted as the connection strength.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with eigenvector centrality as the value. The
       associated vector has unit Euclidean norm and the values are
       nonegative.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> centrality = nx.eigenvector_centrality(G)
    >>> sorted((v, f"{c:0.2f}") for v, c in centrality.items())
    [(0, '0.37'), (1, '0.60'), (2, '0.60'), (3, '0.37')]

    Raises
    ------
    NetworkXPointlessConcept
        If the graph G is the null graph.

    NetworkXError
        If each value in `nstart` is zero.

    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    See Also
    --------
    eigenvector_centrality_numpy
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Eigenvector centrality was introduced by Landau [2]_ for chess
    tournaments. It was later rediscovered by Wei [3]_ and then
    popularized by Kendall [4]_ in the context of sport ranking. Berge
    introduced a general definition for graphs based on social connections
    [5]_. Bonacich [6]_ reintroduced again eigenvector centrality and made
    it popular in link analysis.

    This function computes the left dominant eigenvector, which corresponds
    to adding the centrality of predecessors: this is the usual approach.
    To add the centrality of successors first reverse the graph with
    ``G.reverse()``.

    The implementation uses power iteration [7]_ to compute a dominant
    eigenvector starting from the provided vector `nstart`. Convergence is
    guaranteed as long as `nstart` has a nonzero projection on a dominant
    eigenvector, which certainly happens using the default value.

    The method stops when the change in the computed vector between two
    iterations is smaller than an error tolerance of ``G.number_of_nodes()
    * tol`` or after ``max_iter`` iterations, but in the second case it
    raises an exception.

    This implementation uses $(A + I)$ rather than the adjacency matrix
    $A$ because the change preserves eigenvectors, but it shifts the
    spectrum, thus guaranteeing convergence even for networks with
    negative eigenvalues of maximum modulus.

    References
    ----------
    .. [1] Abraham Berman and Robert J. Plemmons.
       "Nonnegative Matrices in the Mathematical Sciences."
       Classics in Applied Mathematics. SIAM, 1994.

    .. [2] Edmund Landau.
       "Zur relativen Wertbemessung der Turnierresultate."
       Deutsches Wochenschach, 11:366–369, 1895.

    .. [3] Teh-Hsing Wei.
       "The Algebraic Foundations of Ranking Theory."
       PhD thesis, University of Cambridge, 1952.

    .. [4] Maurice G. Kendall.
       "Further contributions to the theory of paired comparisons."
       Biometrics, 11(1):43–62, 1955.
       https://www.jstor.org/stable/3001479

    .. [5] Claude Berge
       "Théorie des graphes et ses applications."
       Dunod, Paris, France, 1958.

    .. [6] Phillip Bonacich.
       "Technique for analyzing overlapping memberships."
       Sociological Methodology, 4:176–185, 1972.
       https://www.jstor.org/stable/270732

    .. [7] Power iteration:: https://en.wikipedia.org/wiki/Power_iteration

    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "cannot compute centrality for the null graph"
        )
    # If no initial vector is provided, start with the all-ones vector.
    if nstart is None:
        nstart = {v: 1 for v in G}
    if all(v == 0 for v in nstart.values()):
        raise nx.NetworkXError("initial vector cannot have all zero values")
    # Normalize the initial vector so that each entry is in [0, 1]. This is
    # guaranteed to never have a divide-by-zero error by the previous line.
    nstart_sum = sum(nstart.values())
    x = {k: v / nstart_sum for k, v in nstart.items()}
    nnodes = G.number_of_nodes()
    # make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = xlast.copy()  # Start with xlast times I to iterate with (A+I)
        # do the multiplication y^T = x^T A (left eigenvector)
        for n in x:
            for nbr in G[n]:
                w = G[n][nbr].get(weight, 1) if weight else 1
                x[nbr] += xlast[n] * w
        # Normalize the vector. The normalization denominator `norm`
        # should never be zero by the Perron--Frobenius
        # theorem. However, in case it is due to numerical error, we
        # assume the norm to be one instead.
        norm = math.hypot(*x.values()) or 1
        x = {k: v / norm for k, v in x.items()}
        # Check for convergence (in the L_1 norm).
        if sum(abs(x[n] - xlast[n]) for n in x) < nnodes * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)


@nx._dispatchable(edge_attrs="weight")
def eigenvector_centrality_numpy(G, weight=None, max_iter=50, tol=0):
    r"""Compute the eigenvector centrality for the graph `G`.

    Eigenvector centrality computes the centrality for a node by adding
    the centrality of its predecessors. The centrality for node $i$ is the
    $i$-th element of a left eigenvector associated with the eigenvalue $\lambda$
    of maximum modulus that is positive. Such an eigenvector $x$ is
    defined up to a multiplicative constant by the equation

    .. math::

         \lambda x^T = x^T A,

    where $A$ is the adjacency matrix of the graph `G`. By definition of
    row-column product, the equation above is equivalent to

    .. math::

        \lambda x_i = \sum_{j\to i}x_j.

    That is, adding the eigenvector centralities of the predecessors of
    $i$ one obtains the eigenvector centrality of $i$ multiplied by
    $\lambda$. In the case of undirected graphs, $x$ also solves the familiar
    right-eigenvector equation $Ax = \lambda x$.

    By virtue of the Perron--Frobenius theorem [1]_, if `G` is (strongly)
    connected, there is a unique eigenvector $x$, and all its entries
    are strictly positive.

    However, if `G` is not (strongly) connected, there might be several left
    eigenvectors associated with $\lambda$, and some of their elements
    might be zero.
    Depending on the method used to choose eigenvectors, round-off error can affect
    which of the infinitely many eigenvectors is reported.
    This can lead to inconsistent results for the same graph,
    which the underlying implementation is not robust to.
    For this reason, only (strongly) connected graphs are accepted.

    Parameters
    ----------
    G : graph
        A connected NetworkX graph.

    weight : None or string, optional (default=None)
        If ``None``, all edge weights are considered equal. Otherwise holds the
        name of the edge attribute used as weight. In this measure the
        weight is interpreted as the connection strength.

    max_iter : integer, optional (default=50)
        Maximum number of Arnoldi update iterations allowed.

    tol : float, optional (default=0)
        Relative accuracy for eigenvalues (stopping criterion).
        The default value of 0 implies machine precision.

    Returns
    -------
    nodes : dict of nodes
        Dictionary of nodes with eigenvector centrality as the value. The
        associated vector has unit Euclidean norm and the values are
        nonnegative.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> centrality = nx.eigenvector_centrality_numpy(G)
    >>> print([f"{node} {centrality[node]:0.2f}" for node in centrality])
    ['0 0.37', '1 0.60', '2 0.60', '3 0.37']

    Raises
    ------
    NetworkXPointlessConcept
        If the graph `G` is the null graph.

    ArpackNoConvergence
        When the requested convergence is not obtained. The currently
        converged eigenvalues and eigenvectors can be found as
        eigenvalues and eigenvectors attributes of the exception object.

    AmbiguousSolution
        If `G` is not connected.

    See Also
    --------
    :func:`scipy.sparse.linalg.eigs`
    eigenvector_centrality
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Eigenvector centrality was introduced by Landau [2]_ for chess
    tournaments. It was later rediscovered by Wei [3]_ and then
    popularized by Kendall [4]_ in the context of sport ranking. Berge
    introduced a general definition for graphs based on social connections
    [5]_. Bonacich [6]_ reintroduced again eigenvector centrality and made
    it popular in link analysis.

    This function computes the left dominant eigenvector, which corresponds
    to adding the centrality of predecessors: this is the usual approach.
    To add the centrality of successors first reverse the graph with
    ``G.reverse()``.

    This implementation uses the
    :func:`SciPy sparse eigenvalue solver<scipy.sparse.linalg.eigs>` (ARPACK)
    to find the largest eigenvalue/eigenvector pair using Arnoldi iterations
    [7]_.

    References
    ----------
    .. [1] Abraham Berman and Robert J. Plemmons.
       "Nonnegative Matrices in the Mathematical Sciences".
       Classics in Applied Mathematics. SIAM, 1994.

    .. [2] Edmund Landau.
       "Zur relativen Wertbemessung der Turnierresultate".
       Deutsches Wochenschach, 11:366--369, 1895.

    .. [3] Teh-Hsing Wei.
       "The Algebraic Foundations of Ranking Theory".
       PhD thesis, University of Cambridge, 1952.

    .. [4] Maurice G. Kendall.
       "Further contributions to the theory of paired comparisons".
       Biometrics, 11(1):43--62, 1955.
       https://www.jstor.org/stable/3001479

    .. [5] Claude Berge.
       "Théorie des graphes et ses applications".
       Dunod, Paris, France, 1958.

    .. [6] Phillip Bonacich.
       "Technique for analyzing overlapping memberships".
       Sociological Methodology, 4:176--185, 1972.
       https://www.jstor.org/stable/270732

    .. [7] Arnoldi, W. E. (1951).
       "The principle of minimized iterations in the solution of the matrix eigenvalue problem".
       Quarterly of Applied Mathematics. 9 (1): 17--29.
       https://doi.org/10.1090/qam/42792
    """
    import numpy as np
    import scipy as sp

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "cannot compute centrality for the null graph"
        )
    connected = nx.is_strongly_connected(G) if G.is_directed() else nx.is_connected(G)
    if not connected:  # See gh-6888.
        raise nx.AmbiguousSolution(
            "`eigenvector_centrality_numpy` does not give consistent results for disconnected graphs"
        )
    M = nx.to_scipy_sparse_array(G, nodelist=list(G), weight=weight, dtype=float)
    _, eigenvector = sp.sparse.linalg.eigs(
        M.T, k=1, which="LR", maxiter=max_iter, tol=tol
    )
    largest = eigenvector.flatten().real
    norm = np.sign(largest.sum()) * sp.linalg.norm(largest)
    return dict(zip(G, (largest / norm).tolist()))
<|endoftext|>"""
Closeness centrality measures.
"""

import functools

import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils.decorators import not_implemented_for

__all__ = ["closeness_centrality", "incremental_closeness_centrality"]


@nx._dispatchable(edge_attrs="distance")
def closeness_centrality(G, u=None, distance=None, wf_improved=True):
    r"""Compute closeness centrality for nodes.

    Closeness centrality [1]_ of a node `u` is the reciprocal of the
    average shortest path distance to `u` over all `n-1` reachable nodes.

    .. math::

        C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    where `d(v, u)` is the shortest-path distance between `v` and `u`,
    and `n-1` is the number of nodes reachable from `u`. Notice that the
    closeness distance function computes the incoming distance to `u`
    for directed graphs. To use outward distance, act on `G.reverse()`.

    Notice that higher values of closeness indicate higher centrality.

    Wasserman and Faust propose an improved formula for graphs with
    more than one connected component. The result is "a ratio of the
    fraction of actors in the group who are reachable, to the average
    distance" from the reachable actors [2]_. You might think this
    scale factor is inverted but it is not. As is, nodes from small
    components receive a smaller closeness value. Letting `N` denote
    the number of nodes in the graph,

    .. math::

        C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    Parameters
    ----------
    G : graph
      A NetworkX graph

    u : node, optional
      Return only the value for node u

    distance : edge attribute key, optional (default=None)
      Use the specified edge attribute as the edge distance in shortest
      path calculations.  If `None` (the default) all edges have a distance of 1.
      Absent edge attributes are assigned a distance of 1. Note that no check
      is performed to ensure that edges have the provided attribute.

    wf_improved : bool, optional (default=True)
      If True, scale by the fraction of nodes reachable. This gives the
      Wasserman and Faust improved formula. For single component graphs
      it is the same as the original formula.

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with closeness centrality as the value.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    >>> nx.closeness_centrality(G)
    {0: 1.0, 1: 1.0, 2: 0.75, 3: 0.75}

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality,
    degree_centrality, incremental_closeness_centrality

    Notes
    -----
    The closeness centrality is normalized to `(n-1)/(|G|-1)` where
    `n` is the number of nodes in the connected part of graph
    containing the node.  If the graph is not completely connected,
    this algorithm computes the closeness centrality for each
    connected part separately scaled by that parts size.

    If the 'distance' keyword is set to an edge attribute key then the
    shortest-path length will be computed using Dijkstra's algorithm with
    that edge attribute as the edge weight.

    The closeness centrality uses *inward* distance to a node, not outward.
    If you want to use outword distances apply the function to `G.reverse()`

    In NetworkX 2.2 and earlier a bug caused Dijkstra's algorithm to use the
    outward distance rather than the inward distance. If you use a 'distance'
    keyword and a DiGraph, your results will change between v2.2 and v2.3.

    References
    ----------
    .. [1] Linton C. Freeman: Centrality in networks: I.
       Conceptual clarification. Social Networks 1:215-239, 1979.
       https://doi.org/10.1016/0378-8733(78)90021-7
    .. [2] pg. 201 of Wasserman, S. and Faust, K.,
       Social Network Analysis: Methods and Applications, 1994,
       Cambridge University Press.
    """
    if G.is_directed():
        G = G.reverse()  # create a reversed graph view

    if distance is not None:
        # use Dijkstra's algorithm with specified attribute as edge weight
        path_length = functools.partial(
            nx.single_source_dijkstra_path_length, weight=distance
        )
    else:
        path_length = nx.single_source_shortest_path_length

    if u is None:
        nodes = G.nodes
    else:
        nodes = [u]
    closeness_dict = {}
    for n in nodes:
        sp = path_length(G, n)
        totsp = sum(sp.values())
        len_G = len(G)
        _closeness_centrality = 0.0
        if totsp > 0.0 and len_G > 1:
            _closeness_centrality = (len(sp) - 1.0) / totsp
            # normalize to number of nodes-1 in connected part
            if wf_improved:
                s = (len(sp) - 1.0) / (len_G - 1)
                _closeness_centrality *= s
        closeness_dict[n] = _closeness_centrality
    if u is not None:
        return closeness_dict[u]
    return closeness_dict


@not_implemented_for("directed")
@nx._dispatchable(mutates_input=True)
def incremental_closeness_centrality(
    G, edge, prev_cc=None, insertion=True, wf_improved=True
):
    r"""Incremental closeness centrality for nodes.

    Compute closeness centrality for nodes using level-based work filtering
    as described in Incremental Algorithms for Closeness Centrality by Sariyuce et al.

    Level-based work filtering detects unnecessary updates to the closeness
    centrality and filters them out.

    ---
    From "Incremental Algorithms for Closeness Centrality":

    Theorem 1: Let :math:`G = (V, E)` be a graph and u and v be two vertices in V
    such that there is no edge (u, v) in E. Let :math:`G' = (V, E \cup uv)`
    Then :math:`cc[s] = cc'[s]` if and only if :math:`\left|dG(s, u) - dG(s, v)\right| \leq 1`.

    Where :math:`dG(u, v)` denotes the length of the shortest path between
    two vertices u, v in a graph G, cc[s] is the closeness centrality for a
    vertex s in V, and cc'[s] is the closeness centrality for a
    vertex s in V, with the (u, v) edge added.
    ---

    We use Theorem 1 to filter out updates when adding or removing an edge.
    When adding an edge (u, v), we compute the shortest path lengths from all
    other nodes to u and to v before the node is added. When removing an edge,
    we compute the shortest path lengths after the edge is removed. Then we
    apply Theorem 1 to use previously computed closeness centrality for nodes
    where :math:`\left|dG(s, u) - dG(s, v)\right| \leq 1`. This works only for
    undirected, unweighted graphs; the distance argument is not supported.

    Closeness centrality [1]_ of a node `u` is the reciprocal of the
    sum of the shortest path distances from `u` to all `n-1` other nodes.
    Since the sum of distances depends on the number of nodes in the
    graph, closeness is normalized by the sum of minimum possible
    distances `n-1`.

    .. math::

        C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    where `d(v, u)` is the shortest-path distance between `v` and `u`,
    and `n` is the number of nodes in the graph.

    Notice that higher values of closeness indicate higher centrality.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    edge : tuple
      The modified edge (u, v) in the graph.

    prev_cc : dictionary
      The previous closeness centrality for all nodes in the graph.

    insertion : bool, optional
      If True (default) the edge was inserted, otherwise it was deleted from the graph.

    wf_improved : bool, optional (default=True)
      If True, scale by the fraction of nodes reachable. This gives the
      Wasserman and Faust improved formula. For single component graphs
      it is the same as the original formula.

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with closeness centrality as the value.

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality,
    degree_centrality, closeness_centrality

    Notes
    -----
    The closeness centrality is normalized to `(n-1)/(|G|-1)` where
    `n` is the number of nodes in the connected part of graph
    containing the node.  If the graph is not completely connected,
    this algorithm computes the closeness centrality for each
    connected part separately.

    References
    ----------
    .. [1] Freeman, L.C., 1979. Centrality in networks: I.
       Conceptual clarification.  Social Networks 1, 215--239.
       https://doi.org/10.1016/0378-8733(78)90021-7
    .. [2] Sariyuce, A.E. ; Kaya, K. ; Saule, E. ; Catalyiirek, U.V. Incremental
       Algorithms for Closeness Centrality. 2013 IEEE International Conference on Big Data
       http://sariyuce.com/papers/bigdata13.pdf
    """
    if prev_cc is not None and set(prev_cc.keys()) != set(G.nodes()):
        raise NetworkXError("prev_cc and G do not have the same nodes")

    # Unpack edge
    (u, v) = edge
    path_length = nx.single_source_shortest_path_length

    if insertion:
        # For edge insertion, we want shortest paths before the edge is inserted
        du = path_length(G, u)
        dv = path_length(G, v)

        G.add_edge(u, v)
    else:
        G.remove_edge(u, v)

        # For edge removal, we want shortest paths after the edge is removed
        du = path_length(G, u)
        dv = path_length(G, v)

    if prev_cc is None:
        return nx.closeness_centrality(G)

    nodes = G.nodes()
    closeness_dict = {}
    for n in nodes:
        if n in du and n in dv and abs(du[n] - dv[n]) <= 1:
            closeness_dict[n] = prev_cc[n]
        else:
            sp = path_length(G, n)
            totsp = sum(sp.values())
            len_G = len(G)
            _closeness_centrality = 0.0
            if totsp > 0.0 and len_G > 1:
                _closeness_centrality = (len(sp) - 1.0) / totsp
                # normalize to number of nodes-1 in connected part
                if wf_improved:
                    s = (len(sp) - 1.0) / (len_G - 1)
                    _closeness_centrality *= s
            closeness_dict[n] = _closeness_centrality

    # Leave the graph as we found it
    if insertion:
        G.remove_edge(u, v)
    else:
        G.add_edge(u, v)

    return closeness_dict
<|endoftext|># Helpers for current-flow betweenness and current-flow closeness
# Lazy computations for inverse Laplacian and flow-matrix rows.
import networkx as nx


@nx._dispatchable(edge_attrs="weight")
def flow_matrix_row(G, weight=None, dtype=float, solver="lu"):
    # Generate a row of the current-flow matrix
    import numpy as np

    solvername = {
        "full": FullInverseLaplacian,
        "lu": SuperLUInverseLaplacian,
        "cg": CGInverseLaplacian,
    }
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G, nodelist=range(n), weight=weight).asformat("csc")
    L = L.astype(dtype)
    C = solvername[solver](L, dtype=dtype)  # initialize solver
    w = C.w  # w is the Laplacian matrix width
    # row-by-row flow matrix
    for u, v in sorted(sorted((u, v)) for u, v in G.edges()):
        B = np.zeros(w, dtype=dtype)
        c = G[u][v].get(weight, 1.0)
        B[u % w] = c
        B[v % w] = -c
        # get only the rows needed in the inverse laplacian
        # and multiply to get the flow matrix row
        row = B @ C.get_rows(u, v)
        yield row, (u, v)


# Class to compute the inverse laplacian only for specified rows
# Allows computation of the current-flow matrix without storing entire
# inverse laplacian matrix
class InverseLaplacian:
    def __init__(self, L, width=None, dtype=None):
        global np
        import numpy as np

        (n, n) = L.shape
        self.dtype = dtype
        self.n = n
        if width is None:
            self.w = self.width(L)
        else:
            self.w = width
        self.C = np.zeros((self.w, n), dtype=dtype)
        self.L1 = L[1:, 1:]
        self.init_solver(L)

    def init_solver(self, L):
        pass

    def solve(self, r):
        raise nx.NetworkXError("Implement solver")

    def solve_inverse(self, r):
        raise nx.NetworkXError("Implement solver")

    def get_rows(self, r1, r2):
        for r in range(r1, r2 + 1):
            self.C[r % self.w, 1:] = self.solve_inverse(r)
        return self.C

    def get_row(self, r):
        self.C[r % self.w, 1:] = self.solve_inverse(r)
        return self.C[r % self.w]

    def width(self, L):
        m = 0
        for i, row in enumerate(L):
            w = 0
            y = np.nonzero(row)[-1]
            if len(y) > 0:
                v = y - i
                w = v.max() - v.min() + 1
                m = max(w, m)
        return m


class FullInverseLaplacian(InverseLaplacian):
    def init_solver(self, L):
        self.IL = np.zeros(L.shape, dtype=self.dtype)
        self.IL[1:, 1:] = np.linalg.inv(self.L1.todense())

    def solve(self, rhs):
        s = np.zeros(rhs.shape, dtype=self.dtype)
        s = self.IL @ rhs
        return s

    def solve_inverse(self, r):
        return self.IL[r, 1:]


class SuperLUInverseLaplacian(InverseLaplacian):
    def init_solver(self, L):
        import scipy as sp

        self.lusolve = sp.sparse.linalg.factorized(self.L1.tocsc())

    def solve_inverse(self, r):
        rhs = np.zeros(self.n, dtype=self.dtype)
        rhs[r] = 1
        return self.lusolve(rhs[1:])

    def solve(self, rhs):
        s = np.zeros(rhs.shape, dtype=self.dtype)
        s[1:] = self.lusolve(rhs[1:])
        return s


class CGInverseLaplacian(InverseLaplacian):
    def init_solver(self, L):
        global sp
        import scipy as sp

        ilu = sp.sparse.linalg.spilu(self.L1.tocsc())
        n = self.n - 1
        self.M = sp.sparse.linalg.LinearOperator(shape=(n, n), matvec=ilu.solve)

    def solve(self, rhs):
        s = np.zeros(rhs.shape, dtype=self.dtype)
        s[1:] = sp.sparse.linalg.cg(self.L1, rhs[1:], M=self.M, atol=0)[0]
        return s

    def solve_inverse(self, r):
        rhs = np.zeros(self.n, self.dtype)
        rhs[r] = 1
        return sp.sparse.linalg.cg(self.L1, rhs[1:], M=self.M, atol=0)[0]
<|endoftext|>from .betweenness import *
from .betweenness_subset import *
from .closeness import *
from .current_flow_betweenness import *
from .current_flow_betweenness_subset import *
from .current_flow_closeness import *
from .degree_alg import *
from .dispersion import *
from .eigenvector import *
from .group import *
from .harmonic import *
from .katz import *
from .load import *
from .percolation import *
from .reaching import *
from .second_order import *
from .subgraph_alg import *
from .trophic import *
from .voterank_alg import *
from .laplacian import *
<|endoftext|>"""Current-flow betweenness centrality measures."""

import networkx as nx
from networkx.algorithms.centrality.flow_matrix import (
    CGInverseLaplacian,
    FullInverseLaplacian,
    SuperLUInverseLaplacian,
    flow_matrix_row,
)
from networkx.utils import (
    not_implemented_for,
    py_random_state,
    reverse_cuthill_mckee_ordering,
)

__all__ = [
    "current_flow_betweenness_centrality",
    "approximate_current_flow_betweenness_centrality",
    "edge_current_flow_betweenness_centrality",
]


@not_implemented_for("directed")
@py_random_state(7)
@nx._dispatchable(edge_attrs="weight")
def approximate_current_flow_betweenness_centrality(
    G,
    normalized=True,
    weight=None,
    dtype=float,
    solver="full",
    epsilon=0.5,
    kmax=10000,
    seed=None,
):
    r"""Compute the approximate current-flow betweenness centrality for nodes.

    Approximates the current-flow betweenness centrality within absolute
    error of epsilon with high probability [1]_.


    Parameters
    ----------
    G : graph
      A NetworkX graph

    normalized : bool, optional (default=True)
      If True the betweenness values are normalized by 2/[(n-1)(n-2)] where
      n is the number of nodes in G.

    weight : string or None, optional (default=None)
      Key for edge data used as the edge weight.
      If None, then use 1 as each edge weight.
      The weight reflects the capacity or the strength of the
      edge.

    dtype : data type (float)
      Default data type for internal matrices.
      Set to np.float32 for lower memory consumption.

    solver : string (default='full')
       Type of linear solver to use for computing the flow matrix.
       Options are "full" (uses most memory), "lu" (recommended), and
       "cg" (uses least memory).

    epsilon: float
        Absolute error tolerance.

    kmax: int
       Maximum number of sample node pairs to use for approximation.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.

    See Also
    --------
    current_flow_betweenness_centrality

    Notes
    -----
    The running time is $O((1/\epsilon^2)m{\sqrt k} \log n)$
    and the space required is $O(m)$ for $n$ nodes and $m$ edges.

    If the edges have a 'weight' attribute they will be used as
    weights in this algorithm.  Unspecified weights are set to 1.

    References
    ----------
    .. [1] Ulrik Brandes and Daniel Fleischer:
       Centrality Measures Based on Current Flow.
       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS '05).
       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.
       https://doi.org/10.1007/978-3-540-31856-9_44
    """
    import numpy as np

    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph not connected.")
    solvername = {
        "full": FullInverseLaplacian,
        "lu": SuperLUInverseLaplacian,
        "cg": CGInverseLaplacian,
    }
    n = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    # make a copy with integer labels according to rcm ordering
    # this could be done without a copy if we really wanted to
    H = nx.relabel_nodes(G, dict(zip(ordering, range(n))))
    L = nx.laplacian_matrix(H, nodelist=range(n), weight=weight).asformat("csc")
    L = L.astype(dtype)
    C = solvername[solver](L, dtype=dtype)  # initialize solver
    betweenness = dict.fromkeys(H, 0.0)
    nb = (n - 1.0) * (n - 2.0)  # normalization factor
    cstar = n * (n - 1) / nb
    l = 1  # parameter in approximation, adjustable
    k = l * int(np.ceil((cstar / epsilon) ** 2 * np.log(n)))
    if k > kmax:
        msg = f"Number random pairs k>kmax ({k}>{kmax}) "
        raise nx.NetworkXError(msg, "Increase kmax or epsilon")
    cstar2k = cstar / (2 * k)
    for _ in range(k):
        s, t = pair = seed.sample(range(n), 2)
        b = np.zeros(n, dtype=dtype)
        b[s] = 1
        b[t] = -1
        p = C.solve(b)
        for v in H:
            if v in pair:
                continue
            for nbr in H[v]:
                w = H[v][nbr].get(weight, 1.0)
                betweenness[v] += float(w * np.abs(p[v] - p[nbr]) * cstar2k)
    if normalized:
        factor = 1.0
    else:
        factor = nb / 2.0
    # remap to original node names and "unnormalize" if required
    return {ordering[k]: v * factor for k, v in betweenness.items()}


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def current_flow_betweenness_centrality(
    G, normalized=True, weight=None, dtype=float, solver="full"
):
    r"""Compute current-flow betweenness centrality for nodes.

    Current-flow betweenness centrality uses an electrical current
    model for information spreading in contrast to betweenness
    centrality which uses shortest paths.

    Current-flow betweenness centrality is also known as
    random-walk betweenness centrality [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    normalized : bool, optional (default=True)
      If True the betweenness values are normalized by 2/[(n-1)(n-2)] where
      n is the number of nodes in G.

    weight : string or None, optional (default=None)
      Key for edge data used as the edge weight.
      If None, then use 1 as each edge weight.
      The weight reflects the capacity or the strength of the
      edge.

    dtype : data type (float)
      Default data type for internal matrices.
      Set to np.float32 for lower memory consumption.

    solver : string (default='full')
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
    Current-flow betweenness can be computed in  $O(I(n-1)+mn \log n)$
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
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph not connected.")
    N = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    # make a copy with integer labels according to rcm ordering
    # this could be done without a copy if we really wanted to
    H = nx.relabel_nodes(G, dict(zip(ordering, range(N))))
    betweenness = dict.fromkeys(H, 0.0)  # b[n]=0 for n in H
    for row, (s, t) in flow_matrix_row(H, weight=weight, dtype=dtype, solver=solver):
        pos = dict(zip(row.argsort()[::-1], range(N)))
        for i in range(N):
            betweenness[s] += (i - pos[i]) * row.item(i)
            betweenness[t] += (N - i - 1 - pos[i]) * row.item(i)
    if normalized:
        nb = (N - 1.0) * (N - 2.0)  # normalization factor
    else:
        nb = 2.0
    return {ordering[n]: (b - n) * 2.0 / nb for n, b in betweenness.items()}


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def edge_current_flow_betweenness_centrality(
    G, normalized=True, weight=None, dtype=float, solver="full"
):
    r"""Compute current-flow betweenness centrality for edges.

    Current-flow betweenness centrality uses an electrical current
    model for information spreading in contrast to betweenness
    centrality which uses shortest paths.

    Current-flow betweenness centrality is also known as
    random-walk betweenness centrality [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    normalized : bool, optional (default=True)
      If True the betweenness values are normalized by 2/[(n-1)(n-2)] where
      n is the number of nodes in G.

    weight : string or None, optional (default=None)
      Key for edge data used as the edge weight.
      If None, then use 1 as each edge weight.
      The weight reflects the capacity or the strength of the
      edge.

    dtype : data type (default=float)
      Default data type for internal matrices.
      Set to np.float32 for lower memory consumption.

    solver : string (default='full')
       Type of linear solver to use for computing the flow matrix.
       Options are "full" (uses most memory), "lu" (recommended), and
       "cg" (uses least memory).

    Returns
    -------
    nodes : dictionary
       Dictionary of edge tuples with betweenness centrality as the value.

    Raises
    ------
    NetworkXError
        The algorithm does not support DiGraphs.
        If the input graph is an instance of DiGraph class, NetworkXError
        is raised.

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
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph not connected.")
    N = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    # make a copy with integer labels according to rcm ordering
    # this could be done without a copy if we really wanted to
    H = nx.relabel_nodes(G, dict(zip(ordering, range(N))))
    edges = (tuple(sorted((u, v))) for u, v in H.edges())
    betweenness = dict.fromkeys(edges, 0.0)
    if normalized:
        nb = (N - 1.0) * (N - 2.0)  # normalization factor
    else:
        nb = 2.0
    for row, (e) in flow_matrix_row(H, weight=weight, dtype=dtype, solver=solver):
        pos = dict(zip(row.argsort()[::-1], range(1, N + 1)))
        for i in range(N):
            betweenness[e] += (i + 1 - pos[i]) * row.item(i)
            betweenness[e] += (N - i - pos[i]) * row.item(i)
        betweenness[e] /= nb
    return {(ordering[s], ordering[t]): b for (s, t), b in betweenness.items()}
<|endoftext|>"""Load centrality."""

from operator import itemgetter

import networkx as nx

__all__ = ["load_centrality", "edge_load_centrality"]


@nx._dispatchable(edge_attrs="weight")
def newman_betweenness_centrality(G, v=None, cutoff=None, normalized=True, weight=None):
    """Compute load centrality for nodes.

    The load centrality of a node is the fraction of all shortest
    paths that pass through that node.

    Parameters
    ----------
    G : graph
      A networkx graph.

    normalized : bool, optional (default=True)
      If True the betweenness values are normalized by b=b/(n-1)(n-2) where
      n is the number of nodes in G.

    weight : None or string, optional (default=None)
