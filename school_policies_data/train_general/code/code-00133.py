    k_components : dict
        Dictionary with connectivity level `k` as key and a list of
        sets of nodes that form a k-component of level `k` as values.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Examples
    --------
    >>> # Petersen graph has 10 nodes and it is triconnected, thus all
    >>> # nodes are in a single component on all three connectivity levels
    >>> from networkx.algorithms import approximation as apxa
    >>> G = nx.petersen_graph()
    >>> k_components = apxa.k_components(G)

    Notes
    -----
    The logic of the approximation algorithm for computing the `k`-component
    structure [1]_ is based on repeatedly applying simple and fast algorithms
    for `k`-cores and biconnected components in order to narrow down the
    number of pairs of nodes over which we have to compute White and Newman's
    approximation algorithm for finding node independent paths [2]_. More
    formally, this algorithm is based on Whitney's theorem, which states
    an inclusion relation among node connectivity, edge connectivity, and
    minimum degree for any graph G. This theorem implies that every
    `k`-component is nested inside a `k`-edge-component, which in turn,
    is contained in a `k`-core. Thus, this algorithm computes node independent
    paths among pairs of nodes in each biconnected part of each `k`-core,
    and repeats this procedure for each `k` from 3 to the maximal core number
    of a node in the input graph.

    Because, in practice, many nodes of the core of level `k` inside a
    bicomponent actually are part of a component of level k, the auxiliary
    graph needed for the algorithm is likely to be very dense. Thus, we use
    a complement graph data structure (see `AntiGraph`) to save memory.
    AntiGraph only stores information of the edges that are *not* present
    in the actual auxiliary graph. When applying algorithms to this
    complement graph data structure, it behaves as if it were the dense
    version.

    See also
    --------
    k_components

    References
    ----------
    .. [1]  Torrents, J. and F. Ferraro (2015) Structural Cohesion:
            Visualization and Heuristics for Fast Computation.
            https://arxiv.org/pdf/1503.04476v1

    .. [2]  White, Douglas R., and Mark Newman (2001) A Fast Algorithm for
            Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
            https://www.santafe.edu/research/results/working-papers/fast-approximation-algorithms-for-finding-node-ind

    .. [3]  Moody, J. and D. White (2003). Social cohesion and embeddedness:
            A hierarchical conception of social groups.
            American Sociological Review 68(1), 103--28.
            https://doi.org/10.2307/3088904

    """
    # Dictionary with connectivity level (k) as keys and a list of
    # sets of nodes that form a k-component as values
    k_components = defaultdict(list)
    # make a few functions local for speed
    node_connectivity = local_node_connectivity
    k_core = nx.k_core
    core_number = nx.core_number
    biconnected_components = nx.biconnected_components
    combinations = itertools.combinations
    # Exact solution for k = {1,2}
    # There is a linear time algorithm for triconnectivity, if we had an
    # implementation available we could start from k = 4.
    for component in nx.connected_components(G):
        # isolated nodes have connectivity 0
        comp = set(component)
        if len(comp) > 1:
            k_components[1].append(comp)
    for bicomponent in nx.biconnected_components(G):
        # avoid considering dyads as bicomponents
        bicomp = set(bicomponent)
        if len(bicomp) > 2:
            k_components[2].append(bicomp)
    # There is no k-component of k > maximum core number
    # \kappa(G) <= \lambda(G) <= \delta(G)
    g_cnumber = core_number(G)
    max_core = max(g_cnumber.values())
    for k in range(3, max_core + 1):
        C = k_core(G, k, core_number=g_cnumber)
        for nodes in biconnected_components(C):
            # Build a subgraph SG induced by the nodes that are part of
            # each biconnected component of the k-core subgraph C.
            if len(nodes) < k:
                continue
            SG = G.subgraph(nodes)
            # Build auxiliary graph
            H = _AntiGraph()
            H.add_nodes_from(SG.nodes())
            for u, v in combinations(SG, 2):
                K = node_connectivity(SG, u, v, cutoff=k)
                if k > K:
                    H.add_edge(u, v)
            for h_nodes in biconnected_components(H):
                if len(h_nodes) <= k:
                    continue
                SH = H.subgraph(h_nodes)
                for Gc in _cliques_heuristic(SG, SH, k, min_density):
                    for k_nodes in biconnected_components(Gc):
                        Gk = nx.k_core(SG.subgraph(k_nodes), k)
                        if len(Gk) <= k:
                            continue
                        k_components[k].append(set(Gk))
    return k_components


def _cliques_heuristic(G, H, k, min_density):
    h_cnumber = nx.core_number(H)
    for i, c_value in enumerate(sorted(set(h_cnumber.values()), reverse=True)):
        cands = {n for n, c in h_cnumber.items() if c == c_value}
        # Skip checking for overlap for the highest core value
        if i == 0:
            overlap = False
        else:
            overlap = set.intersection(
                *[{x for x in H[n] if x not in cands} for n in cands]
            )
        if overlap and len(overlap) < k:
            SH = H.subgraph(cands | overlap)
        else:
            SH = H.subgraph(cands)
        sh_cnumber = nx.core_number(SH)
        SG = nx.k_core(G.subgraph(SH), k)
        while not (_same(sh_cnumber) and nx.density(SH) >= min_density):
            # This subgraph must be writable => .copy()
            SH = H.subgraph(SG).copy()
            if len(SH) <= k:
                break
            sh_cnumber = nx.core_number(SH)
            sh_deg = dict(SH.degree())
            min_deg = min(sh_deg.values())
            SH.remove_nodes_from(n for n, d in sh_deg.items() if d == min_deg)
            SG = nx.k_core(G.subgraph(SH), k)
        else:
            yield SG


def _same(measure, tol=0):
    vals = set(measure.values())
    if (max(vals) - min(vals)) <= tol:
        return True
    return False


class _AntiGraph(nx.Graph):
    """
    Class for complement graphs.

    The main goal is to be able to work with big and dense graphs with
    a low memory footprint.

    In this class you add the edges that *do not exist* in the dense graph,
    the report methods of the class return the neighbors, the edges and
    the degree as if it was the dense graph. Thus it's possible to use
    an instance of this class with some of NetworkX functions. In this
    case we only use k-core, connected_components, and biconnected_components.
    """

    all_edge_dict = {"weight": 1}

    def single_edge_dict(self):
        return self.all_edge_dict

    edge_attr_dict_factory = single_edge_dict  # type: ignore[assignment]

    def __getitem__(self, n):
        """Returns a dict of neighbors of node n in the dense graph.

        Parameters
        ----------
        n : node
           A node in the graph.

        Returns
        -------
        adj_dict : dictionary
           The adjacency dictionary for nodes connected to n.

        """
        all_edge_dict = self.all_edge_dict
        return {
            node: all_edge_dict for node in set(self._adj) - set(self._adj[n]) - {n}
        }

    def neighbors(self, n):
        """Returns an iterator over all neighbors of node n in the
        dense graph.
        """
        try:
            return iter(set(self._adj) - set(self._adj[n]) - {n})
        except KeyError as err:
            raise NetworkXError(f"The node {n} is not in the graph.") from err

    class AntiAtlasView(Mapping):
        """An adjacency inner dict for AntiGraph"""

        def __init__(self, graph, node):
            self._graph = graph
            self._atlas = graph._adj[node]
            self._node = node

        def __len__(self):
            return len(self._graph) - len(self._atlas) - 1

        def __iter__(self):
            return (n for n in self._graph if n not in self._atlas and n != self._node)

        def __getitem__(self, nbr):
            nbrs = set(self._graph._adj) - set(self._atlas) - {self._node}
            if nbr in nbrs:
                return self._graph.all_edge_dict
            raise KeyError(nbr)

    class AntiAdjacencyView(AntiAtlasView):
        """An adjacency outer dict for AntiGraph"""

        def __init__(self, graph):
            self._graph = graph
            self._atlas = graph._adj

        def __len__(self):
            return len(self._atlas)

        def __iter__(self):
            return iter(self._graph)

        def __getitem__(self, node):
            if node not in self._graph:
                raise KeyError(node)
            return self._graph.AntiAtlasView(self._graph, node)

    @cached_property
    def adj(self):
        return self.AntiAdjacencyView(self)

    def subgraph(self, nodes):
        """This subgraph method returns a full AntiGraph. Not a View"""
        nodes = set(nodes)
        G = _AntiGraph()
        G.add_nodes_from(nodes)
        for n in G:
            Gnbrs = G.adjlist_inner_dict_factory()
            G._adj[n] = Gnbrs
            for nbr, d in self._adj[n].items():
                if nbr in G._adj:
                    Gnbrs[nbr] = d
                    G._adj[nbr][n] = d
        G.graph = self.graph
        return G

    class AntiDegreeView(nx.reportviews.DegreeView):
        def __iter__(self):
            all_nodes = set(self._succ)
            for n in self._nodes:
                nbrs = all_nodes - set(self._succ[n]) - {n}
                yield (n, len(nbrs))

        def __getitem__(self, n):
            nbrs = set(self._succ) - set(self._succ[n]) - {n}
            # AntiGraph is a ThinGraph so all edges have weight 1
            return len(nbrs) + (n in nbrs)

    @cached_property
    def degree(self):
        """Returns an iterator for (node, degree) and degree for single node.

        The node degree is the number of edges adjacent to the node.

        Parameters
        ----------
        nbunch : iterable container, optional (default=all nodes)
            A container of nodes.  The container will be iterated
            through once.

        weight : string or None, optional (default=None)
           The edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights adjacent to the node.

        Returns
        -------
        deg:
            Degree of the node, if a single node is passed as argument.
        nd_iter : an iterator
            The iterator returns two-tuples of (node, degree).

        See Also
        --------
        degree

        Examples
        --------
        >>> G = nx.path_graph(4)
        >>> G.degree(0)  # node 0 with degree 1
        1
        >>> list(G.degree([0, 1]))
        [(0, 1), (1, 2)]

        """
        return self.AntiDegreeView(self)

    def adjacency(self):
        """Returns an iterator of (node, adjacency set) tuples for all nodes
           in the dense graph.

        This is the fastest way to look at every edge.
        For directed graphs, only outgoing adjacencies are included.

        Returns
        -------
        adj_iter : iterator
           An iterator of (node, adjacency set) for all nodes in
           the graph.

        """
        for n in self._adj:
            yield (n, set(self._adj) - set(self._adj[n]) - {n})
<|endoftext|>"""Distance measures approximated metrics."""

import networkx as nx
from networkx.utils.decorators import py_random_state

__all__ = ["diameter"]


@py_random_state(1)
@nx._dispatchable(name="approximate_diameter")
def diameter(G, seed=None):
    """Returns a lower bound on the diameter of the graph G.

    The function computes a lower bound on the diameter (i.e., the maximum eccentricity)
    of a directed or undirected graph G. The procedure used varies depending on the graph
    being directed or not.

    If G is an `undirected` graph, then the function uses the `2-sweep` algorithm [1]_.
    The main idea is to pick the farthest node from a random node and return its eccentricity.

    Otherwise, if G is a `directed` graph, the function uses the `2-dSweep` algorithm [2]_,
    The procedure starts by selecting a random source node $s$ from which it performs a
    forward and a backward BFS. Let $a_1$ and $a_2$ be the farthest nodes in the forward and
    backward cases, respectively. Then, it computes the backward eccentricity of $a_1$ using
    a backward BFS and the forward eccentricity of $a_2$ using a forward BFS.
    Finally, it returns the best lower bound between the two.

    In both cases, the time complexity is linear with respect to the size of G.

    Parameters
    ----------
    G : NetworkX graph

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    d : integer
       Lower Bound on the Diameter of G

    Examples
    --------
    >>> G = nx.path_graph(10)  # undirected graph
    >>> nx.diameter(G)
    9
    >>> G = nx.cycle_graph(3, create_using=nx.DiGraph)  # directed graph
    >>> nx.diameter(G)
    2

    Raises
    ------
    NetworkXError
        If the graph is empty or
        If the graph is undirected and not connected or
        If the graph is directed and not strongly connected.

    See Also
    --------
    networkx.algorithms.distance_measures.diameter

    References
    ----------
    .. [1] Magnien, Clémence, Matthieu Latapy, and Michel Habib.
       *Fast computation of empirically tight bounds for the diameter of massive graphs.*
       Journal of Experimental Algorithmics (JEA), 2009.
       https://arxiv.org/pdf/0904.2728.pdf
    .. [2] Crescenzi, Pierluigi, Roberto Grossi, Leonardo Lanzi, and Andrea Marino.
       *On computing the diameter of real-world directed (weighted) graphs.*
       International Symposium on Experimental Algorithms. Springer, Berlin, Heidelberg, 2012.
       https://courses.cs.ut.ee/MTAT.03.238/2014_fall/uploads/Main/diameter.pdf
    """
    # if G is empty
    if not G:
        raise nx.NetworkXError("Expected non-empty NetworkX graph!")
    # if there's only a node
    if G.number_of_nodes() == 1:
        return 0
    # if G is directed
    if G.is_directed():
        return _two_sweep_directed(G, seed)
    # else if G is undirected
    return _two_sweep_undirected(G, seed)


def _two_sweep_undirected(G, seed):
    """Helper function for finding a lower bound on the diameter
        for undirected Graphs.

        The idea is to pick the farthest node from a random node
        and return its eccentricity.

        ``G`` is a NetworkX undirected graph.

    .. note::

        ``seed`` is a random.Random or numpy.random.RandomState instance
    """
    # select a random source node
    source = seed.choice(list(G))
    # get the distances to the other nodes
    distances = nx.shortest_path_length(G, source)
    # if some nodes have not been visited, then the graph is not connected
    if len(distances) != len(G):
        raise nx.NetworkXError("Graph not connected.")
    # take a node that is (one of) the farthest nodes from the source
    *_, node = distances
    # return the eccentricity of the node
    return nx.eccentricity(G, node)


def _two_sweep_directed(G, seed):
    """Helper function for finding a lower bound on the diameter
        for directed Graphs.

        It implements 2-dSweep, the directed version of the 2-sweep algorithm.
        The algorithm follows the following steps.
        1. Select a source node $s$ at random.
        2. Perform a forward BFS from $s$ to select a node $a_1$ at the maximum
        distance from the source, and compute $LB_1$, the backward eccentricity of $a_1$.
        3. Perform a backward BFS from $s$ to select a node $a_2$ at the maximum
        distance from the source, and compute $LB_2$, the forward eccentricity of $a_2$.
        4. Return the maximum between $LB_1$ and $LB_2$.

        ``G`` is a NetworkX directed graph.

    .. note::

        ``seed`` is a random.Random or numpy.random.RandomState instance
    """
    # get a new digraph G' with the edges reversed in the opposite direction
    G_reversed = G.reverse()
    # select a random source node
    source = seed.choice(list(G))
    # compute forward distances from source
    forward_distances = nx.shortest_path_length(G, source)
    # compute backward distances  from source
    backward_distances = nx.shortest_path_length(G_reversed, source)
    # if either the source can't reach every node or not every node
    # can reach the source, then the graph is not strongly connected
    n = len(G)
    if len(forward_distances) != n or len(backward_distances) != n:
        raise nx.NetworkXError("DiGraph not strongly connected.")
    # take a node a_1 at the maximum distance from the source in G
    *_, a_1 = forward_distances
    # take a node a_2 at the maximum distance from the source in G_reversed
    *_, a_2 = backward_distances
    # return the max between the backward eccentricity of a_1 and the forward eccentricity of a_2
    return max(nx.eccentricity(G_reversed, a_1), nx.eccentricity(G, a_2))
<|endoftext|>"""
===========================
Depth First Search on Edges
===========================

Algorithms for a depth-first traversal of edges in a graph.

"""

import networkx as nx

FORWARD = "forward"
REVERSE = "reverse"

__all__ = ["edge_dfs"]


@nx._dispatchable
def edge_dfs(G, source=None, orientation=None):
    """A directed, depth-first-search of edges in `G`, beginning at `source`.

    Yield the edges of G in a depth-first-search order continuing until
    all edges are generated.

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

    Yields
    ------
    edge : directed edge
        A directed edge indicating the path taken by the depth-first traversal.
        For graphs, `edge` is of the form `(u, v)` where `u` and `v`
        are the tail and head of the edge as determined by the traversal.
        For multigraphs, `edge` is of the form `(u, v, key)`, where `key` is
        the key of the edge. When the graph is directed, then `u` and `v`
        are always in the order of the actual directed edge.
        If orientation is not None then the edge tuple is extended to include
        the direction of traversal ('forward' or 'reverse') on that edge.

    Examples
    --------
    >>> nodes = [0, 1, 2, 3]
    >>> edges = [(0, 1), (1, 0), (1, 0), (2, 1), (3, 1)]

    >>> list(nx.edge_dfs(nx.Graph(edges), nodes))
    [(0, 1), (1, 2), (1, 3)]

    >>> list(nx.edge_dfs(nx.DiGraph(edges), nodes))
    [(0, 1), (1, 0), (2, 1), (3, 1)]

    >>> list(nx.edge_dfs(nx.MultiGraph(edges), nodes))
    [(0, 1, 0), (1, 0, 1), (0, 1, 2), (1, 2, 0), (1, 3, 0)]

    >>> list(nx.edge_dfs(nx.MultiDiGraph(edges), nodes))
    [(0, 1, 0), (1, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 0)]

    >>> list(nx.edge_dfs(nx.DiGraph(edges), nodes, orientation="ignore"))
    [(0, 1, 'forward'), (1, 0, 'forward'), (2, 1, 'reverse'), (3, 1, 'reverse')]

    >>> list(nx.edge_dfs(nx.MultiDiGraph(edges), nodes, orientation="ignore"))
    [(0, 1, 0, 'forward'), (1, 0, 0, 'forward'), (1, 0, 1, 'reverse'), (2, 1, 0, 'reverse'), (3, 1, 0, 'reverse')]

    Notes
    -----
    The goal of this function is to visit edges. It differs from the more
    familiar depth-first traversal of nodes, as provided by
    :func:`~networkx.algorithms.traversal.depth_first_search.dfs_edges`, in
    that it does not stop once every node has been visited. In a directed graph
    with edges [(0, 1), (1, 2), (2, 1)], the edge (2, 1) would not be visited
    if not for the functionality provided by this function.

    See Also
    --------
    :func:`~networkx.algorithms.traversal.depth_first_search.dfs_edges`

    """
    nodes = list(G.nbunch_iter(source))
    if not nodes:
        return

    directed = G.is_directed()
    kwds = {"data": False}
    if G.is_multigraph() is True:
        kwds["keys"] = True

    # set up edge lookup
    if orientation is None:

        def edges_from(node):
            return iter(G.edges(node, **kwds))

    elif not directed or orientation == "original":

        def edges_from(node):
            for e in G.edges(node, **kwds):
                yield e + (FORWARD,)

    elif orientation == "reverse":

        def edges_from(node):
            for e in G.in_edges(node, **kwds):
                yield e + (REVERSE,)

    elif orientation == "ignore":

        def edges_from(node):
            for e in G.edges(node, **kwds):
                yield e + (FORWARD,)
            for e in G.in_edges(node, **kwds):
                yield e + (REVERSE,)

    else:
        raise nx.NetworkXError("invalid orientation argument.")

    # set up formation of edge_id to easily look up if edge already returned
    if directed:

        def edge_id(edge):
            # remove direction indicator
            return edge[:-1] if orientation is not None else edge

    else:

        def edge_id(edge):
            # single id for undirected requires frozenset on nodes
            return (frozenset(edge[:2]),) + edge[2:]

    # Basic setup
    check_reverse = directed and orientation in ("reverse", "ignore")

    visited_edges = set()
    visited_nodes = set()
    edges = {}

    # start DFS
    for start_node in nodes:
        stack = [start_node]
        while stack:
            current_node = stack[-1]
            if current_node not in visited_nodes:
                edges[current_node] = edges_from(current_node)
                visited_nodes.add(current_node)

            try:
                edge = next(edges[current_node])
            except StopIteration:
                # No more edges from the current node.
                stack.pop()
            else:
                edgeid = edge_id(edge)
                if edgeid not in visited_edges:
                    visited_edges.add(edgeid)
                    # Mark the traversed "to" node as to-be-explored.
                    if check_reverse and edge[-1] == REVERSE:
                        stack.append(edge[0])
                    else:
                        stack.append(edge[1])
                    yield edge
<|endoftext|>"""Basic algorithms for breadth-first searching the nodes of a graph."""

import networkx as nx

__all__ = ["bfs_beam_edges"]


@nx._dispatchable
def bfs_beam_edges(G, source, value, width=None):
    """Iterates over edges in a beam search.

    The beam search is a generalized breadth-first search in which only
    the "best" *w* neighbors of the current node are enqueued, where *w*
    is the beam width and "best" is an application-specific
    heuristic. In general, a beam search with a small beam width might
    not visit each node in the graph.

    .. note::

       With the default value of ``width=None`` or `width` greater than the
       maximum degree of the graph, this function equates to a slower
       version of `~networkx.algorithms.traversal.breadth_first_search.bfs_edges`.
       All nodes will be visited, though the order of the reported edges may
       vary. In such cases, `value` has no effect - consider using `bfs_edges`
       directly instead.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for the breadth-first search; this function
        iterates over only those edges in the component reachable from
        this node.

    value : function
        A function that takes a node of the graph as input and returns a
        real number indicating how "good" it is. A higher value means it
        is more likely to be visited sooner during the search. When
        visiting a new node, only the `width` neighbors with the highest
        `value` are enqueued (in decreasing order of `value`).

    width : int (default = None)
        The beam width for the search. This is the number of neighbors
        (ordered by `value`) to enqueue when visiting each new node.

    Yields
    ------
    edge
        Edges in the beam search starting from `source`, given as a pair
        of nodes.

    Examples
    --------
    To give nodes with, for example, a higher centrality precedence
    during the search, set the `value` function to return the centrality
    value of the node:

    >>> G = nx.karate_club_graph()
    >>> centrality = nx.eigenvector_centrality(G)
    >>> list(nx.bfs_beam_edges(G, source=0, value=centrality.get, width=3))
    [(0, 2), (0, 1), (0, 8), (2, 32), (1, 13), (8, 33)]
    """

    if width is None:
        width = len(G)

    def successors(v):
        """Returns a list of the best neighbors of a node.

        `v` is a node in the graph `G`.

        The "best" neighbors are chosen according to the `value`
        function (higher is better). Only the `width` best neighbors of
        `v` are returned.
        """
        # TODO The Python documentation states that for small values, it
        # is better to use `heapq.nlargest`. We should determine the
        # threshold at which its better to use `heapq.nlargest()`
        # instead of `sorted()[:]` and apply that optimization here.
        #
        # If `width` is greater than the number of neighbors of `v`, all
        # neighbors are returned by the semantics of slicing in
        # Python. This occurs in the special case that the user did not
        # specify a `width`: in this case all neighbors are always
        # returned, so this is just a (slower) implementation of
        # `bfs_edges(G, source)` but with a sorted enqueue step.
        return iter(sorted(G.neighbors(v), key=value, reverse=True)[:width])

    yield from nx.generic_bfs_edges(G, source, successors)
<|endoftext|>"""
=============================
Breadth First Search on Edges
=============================

Algorithms for a breadth-first traversal of edges in a graph.

"""

from collections import deque

import networkx as nx

FORWARD = "forward"
REVERSE = "reverse"

__all__ = ["edge_bfs"]


@nx._dispatchable
def edge_bfs(G, source=None, orientation=None):
    """A directed, breadth-first-search of edges in `G`, beginning at `source`.

    Yield the edges of G in a breadth-first-search order continuing until
    all edges are generated.

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

    Yields
    ------
    edge : directed edge
        A directed edge indicating the path taken by the breadth-first-search.
        For graphs, `edge` is of the form `(u, v)` where `u` and `v`
        are the tail and head of the edge as determined by the traversal.
        For multigraphs, `edge` is of the form `(u, v, key)`, where `key` is
        the key of the edge. When the graph is directed, then `u` and `v`
        are always in the order of the actual directed edge.
        If orientation is not None then the edge tuple is extended to include
        the direction of traversal ('forward' or 'reverse') on that edge.

    Examples
    --------
    >>> nodes = [0, 1, 2, 3]
    >>> edges = [(0, 1), (1, 0), (1, 0), (2, 0), (2, 1), (3, 1)]

    >>> list(nx.edge_bfs(nx.Graph(edges), nodes))
    [(0, 1), (0, 2), (1, 2), (1, 3)]

    >>> list(nx.edge_bfs(nx.DiGraph(edges), nodes))
    [(0, 1), (1, 0), (2, 0), (2, 1), (3, 1)]

    >>> list(nx.edge_bfs(nx.MultiGraph(edges), nodes))
    [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (1, 2, 0), (1, 3, 0)]

    >>> list(nx.edge_bfs(nx.MultiDiGraph(edges), nodes))
    [(0, 1, 0), (1, 0, 0), (1, 0, 1), (2, 0, 0), (2, 1, 0), (3, 1, 0)]

    >>> list(nx.edge_bfs(nx.DiGraph(edges), nodes, orientation="ignore"))
    [(0, 1, 'forward'), (1, 0, 'reverse'), (2, 0, 'reverse'), (2, 1, 'reverse'), (3, 1, 'reverse')]

    >>> list(nx.edge_bfs(nx.MultiDiGraph(edges), nodes, orientation="ignore"))
    [(0, 1, 0, 'forward'), (1, 0, 0, 'reverse'), (1, 0, 1, 'reverse'), (2, 0, 0, 'reverse'), (2, 1, 0, 'reverse'), (3, 1, 0, 'reverse')]

    Notes
    -----
    The goal of this function is to visit edges. It differs from the more
    familiar breadth-first-search of nodes, as provided by
    :func:`networkx.algorithms.traversal.breadth_first_search.bfs_edges`, in
    that it does not stop once every node has been visited. In a directed graph
    with edges [(0, 1), (1, 2), (2, 1)], the edge (2, 1) would not be visited
    if not for the functionality provided by this function.

    The naming of this function is very similar to bfs_edges. The difference
    is that 'edge_bfs' yields edges even if they extend back to an already
    explored node while 'bfs_edges' yields the edges of the tree that results
    from a breadth-first-search (BFS) so no edges are reported if they extend
    to already explored nodes. That means 'edge_bfs' reports all edges while
    'bfs_edges' only report those traversed by a node-based BFS. Yet another
    description is that 'bfs_edges' reports the edges traversed during BFS
    while 'edge_bfs' reports all edges in the order they are explored.

    See Also
    --------
    bfs_edges
    bfs_tree
    edge_dfs

    """
    nodes = list(G.nbunch_iter(source))
    if not nodes:
        return

    directed = G.is_directed()
    kwds = {"data": False}
    if G.is_multigraph() is True:
        kwds["keys"] = True

    # set up edge lookup
    if orientation is None:

        def edges_from(node):
            return iter(G.edges(node, **kwds))

    elif not directed or orientation == "original":

        def edges_from(node):
            for e in G.edges(node, **kwds):
                yield e + (FORWARD,)

    elif orientation == "reverse":

        def edges_from(node):
            for e in G.in_edges(node, **kwds):
                yield e + (REVERSE,)

    elif orientation == "ignore":

        def edges_from(node):
            for e in G.edges(node, **kwds):
                yield e + (FORWARD,)
            for e in G.in_edges(node, **kwds):
                yield e + (REVERSE,)

    else:
        raise nx.NetworkXError("invalid orientation argument.")

    if directed:
        neighbors = G.successors

        def edge_id(edge):
            # remove direction indicator
            return edge[:-1] if orientation is not None else edge

    else:
        neighbors = G.neighbors

        def edge_id(edge):
            return (frozenset(edge[:2]),) + edge[2:]

    check_reverse = directed and orientation in ("reverse", "ignore")

    # start BFS
    visited_nodes = set(nodes)
    visited_edges = set()
    queue = deque([(n, edges_from(n)) for n in nodes])
    while queue:
        parent, children_edges = queue.popleft()
        for edge in children_edges:
            if check_reverse and edge[-1] == REVERSE:
                child = edge[0]
            else:
                child = edge[1]
            if child not in visited_nodes:
                visited_nodes.add(child)
                queue.append((child, edges_from(child)))
            edgeid = edge_id(edge)
            if edgeid not in visited_edges:
                visited_edges.add(edgeid)
                yield edge
<|endoftext|>"""Basic algorithms for breadth-first searching the nodes of a graph."""

from collections import deque

import networkx as nx

__all__ = [
    "bfs_edges",
    "bfs_tree",
    "bfs_predecessors",
    "bfs_successors",
    "descendants_at_distance",
    "bfs_layers",
    "bfs_labeled_edges",
    "generic_bfs_edges",
]


@nx._dispatchable
def generic_bfs_edges(G, source, neighbors=None, depth_limit=None):
    """Iterate over edges in a breadth-first search.

    The breadth-first search begins at `source` and enqueues the
    neighbors of newly visited nodes specified by the `neighbors`
    function.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for the breadth-first search; this function
        iterates over only those edges in the component reachable from
        this node.

    neighbors : function
        A function that takes a newly visited node of the graph as input
        and returns an *iterator* (not just a list) of nodes that are
        neighbors of that node with custom ordering. If not specified, this is
        just the ``G.neighbors`` method, but in general it can be any function
        that returns an iterator over some or all of the neighbors of a
        given node, in any order.

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth.

    Yields
    ------
    edge
        Edges in the breadth-first search starting from `source`.

    Examples
    --------
    >>> G = nx.path_graph(7)
    >>> list(nx.generic_bfs_edges(G, source=0))
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    >>> list(nx.generic_bfs_edges(G, source=2))
    [(2, 1), (2, 3), (1, 0), (3, 4), (4, 5), (5, 6)]
    >>> list(nx.generic_bfs_edges(G, source=2, depth_limit=2))
    [(2, 1), (2, 3), (1, 0), (3, 4)]

    The `neighbors` param can be used to specify the visitation order of each
    node's neighbors generically. In the following example, we modify the default
    neighbor to return *odd* nodes first:

    >>> def odd_first(n):
    ...     return sorted(G.neighbors(n), key=lambda x: x % 2, reverse=True)

    >>> G = nx.star_graph(5)
    >>> list(nx.generic_bfs_edges(G, source=0))  # Default neighbor ordering
    [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    >>> list(nx.generic_bfs_edges(G, source=0, neighbors=odd_first))
    [(0, 1), (0, 3), (0, 5), (0, 2), (0, 4)]

    Notes
    -----
    This implementation is from `PADS`_, which was in the public domain
    when it was first accessed in July, 2004.  The modifications
    to allow depth limits are based on the Wikipedia article
    "`Depth-limited-search`_".

    .. _PADS: http://www.ics.uci.edu/~eppstein/PADS/BFS.py
    .. _Depth-limited-search: https://en.wikipedia.org/wiki/Depth-limited_search
    """
    if neighbors is None:
        neighbors = G.neighbors
    if depth_limit is None:
        depth_limit = len(G)

    seen = {source}
    n = len(G)
    depth = 0
    next_parents_children = [(source, neighbors(source))]
    while next_parents_children and depth < depth_limit:
        this_parents_children = next_parents_children
        next_parents_children = []
        for parent, children in this_parents_children:
            for child in children:
                if child not in seen:
                    seen.add(child)
                    next_parents_children.append((child, neighbors(child)))
                    yield parent, child
            if len(seen) == n:
                return
        depth += 1


@nx._dispatchable
def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """Iterate over edges in a breadth-first-search starting at source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search; this function
       iterates over only those edges in the component reachable from
       this node.

    reverse : bool, optional
       If True traverse a directed graph in the reverse direction

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    sort_neighbors : function (default=None)
        A function that takes an iterator over nodes as the input, and
        returns an iterable of the same nodes with a custom ordering.
        For example, `sorted` will sort the nodes in increasing order.

    Yields
    ------
    edge: 2-tuple of nodes
       Yields edges resulting from the breadth-first search.

    Examples
    --------
    To get the edges in a breadth-first search::

        >>> G = nx.path_graph(3)
        >>> list(nx.bfs_edges(G, 0))
        [(0, 1), (1, 2)]
        >>> list(nx.bfs_edges(G, source=0, depth_limit=1))
        [(0, 1)]

    To get the nodes in a breadth-first search order::

        >>> G = nx.path_graph(3)
        >>> root = 2
        >>> edges = nx.bfs_edges(G, root)
        >>> nodes = [root] + [v for u, v in edges]
        >>> nodes
        [2, 1, 0]

    Notes
    -----
    The naming of this function is very similar to
    :func:`~networkx.algorithms.traversal.edgebfs.edge_bfs`. The difference
    is that ``edge_bfs`` yields edges even if they extend back to an already
    explored node while this generator yields the edges of the tree that results
    from a breadth-first-search (BFS) so no edges are reported if they extend
    to already explored nodes. That means ``edge_bfs`` reports all edges while
    ``bfs_edges`` only reports those traversed by a node-based BFS. Yet another
    description is that ``bfs_edges`` reports the edges traversed during BFS
    while ``edge_bfs`` reports all edges in the order they are explored.

    Based on the breadth-first search implementation in PADS [1]_
    by D. Eppstein, July 2004; with modifications to allow depth limits
    as described in [2]_.

    References
    ----------
    .. [1] http://www.ics.uci.edu/~eppstein/PADS/BFS.py.
    .. [2] https://en.wikipedia.org/wiki/Depth-limited_search

    See Also
    --------
    bfs_tree
    :func:`~networkx.algorithms.traversal.depth_first_search.dfs_edges`
    :func:`~networkx.algorithms.traversal.edgebfs.edge_bfs`

    """
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors

    if sort_neighbors is not None:
        yield from generic_bfs_edges(
            G, source, lambda node: iter(sort_neighbors(successors(node))), depth_limit
        )
    else:
        yield from generic_bfs_edges(G, source, successors, depth_limit)


@nx._dispatchable(returns_graph=True)
def bfs_tree(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """Returns an oriented tree constructed from of a breadth-first-search
    starting at source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search

    reverse : bool, optional
       If True traverse a directed graph in the reverse direction

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    sort_neighbors : function (default=None)
        A function that takes an iterator over nodes as the input, and
        returns an iterable of the same nodes with a custom ordering.
        For example, `sorted` will sort the nodes in increasing order.

    Returns
    -------
    T: NetworkX DiGraph
       An oriented tree

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> list(nx.bfs_tree(G, 1).edges())
    [(1, 0), (1, 2)]
    >>> H = nx.Graph()
    >>> nx.add_path(H, [0, 1, 2, 3, 4, 5, 6])
    >>> nx.add_path(H, [2, 7, 8, 9, 10])
    >>> sorted(list(nx.bfs_tree(H, source=3, depth_limit=3).edges()))
    [(1, 0), (2, 1), (2, 7), (3, 2), (3, 4), (4, 5), (5, 6), (7, 8)]


    Notes
    -----
    Based on http://www.ics.uci.edu/~eppstein/PADS/BFS.py
    by D. Eppstein, July 2004. The modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited-search`_".

    .. _Depth-limited-search: https://en.wikipedia.org/wiki/Depth-limited_search

    See Also
    --------
    dfs_tree
    bfs_edges
    edge_bfs
    """
    T = nx.DiGraph()
    T.add_node(source)
    edges_gen = bfs_edges(
        G,
        source,
        reverse=reverse,
        depth_limit=depth_limit,
        sort_neighbors=sort_neighbors,
    )
    T.add_edges_from(edges_gen)
    return T


@nx._dispatchable
def bfs_predecessors(G, source, depth_limit=None, sort_neighbors=None):
    """Returns an iterator of predecessors in breadth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    sort_neighbors : function (default=None)
        A function that takes an iterator over nodes as the input, and
        returns an iterable of the same nodes with a custom ordering.
        For example, `sorted` will sort the nodes in increasing order.

    Returns
    -------
    pred: iterator
        (node, predecessor) iterator where `predecessor` is the predecessor of
        `node` in a breadth first search starting from `source`.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> dict(nx.bfs_predecessors(G, 0))
    {1: 0, 2: 1}
    >>> H = nx.Graph()
    >>> H.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    >>> dict(nx.bfs_predecessors(H, 0))
    {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    >>> M = nx.Graph()
    >>> nx.add_path(M, [0, 1, 2, 3, 4, 5, 6])
    >>> nx.add_path(M, [2, 7, 8, 9, 10])
    >>> sorted(nx.bfs_predecessors(M, source=1, depth_limit=3))
    [(0, 1), (2, 1), (3, 2), (4, 3), (7, 2), (8, 7)]
    >>> N = nx.DiGraph()
    >>> nx.add_path(N, [0, 1, 2, 3, 4, 7])
    >>> nx.add_path(N, [3, 5, 6, 7])
    >>> sorted(nx.bfs_predecessors(N, source=2))
    [(3, 2), (4, 3), (5, 3), (6, 5), (7, 4)]

    Notes
    -----
    Based on http://www.ics.uci.edu/~eppstein/PADS/BFS.py
    by D. Eppstein, July 2004. The modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited-search`_".

    .. _Depth-limited-search: https://en.wikipedia.org/wiki/Depth-limited_search

    See Also
    --------
    bfs_tree
    bfs_edges
    edge_bfs
    """
    for s, t in bfs_edges(
        G, source, depth_limit=depth_limit, sort_neighbors=sort_neighbors
    ):
        yield (t, s)


@nx._dispatchable
def bfs_successors(G, source, depth_limit=None, sort_neighbors=None):
    """Returns an iterator of successors in breadth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Specify starting node for breadth-first search

    depth_limit : int, optional(default=len(G))
        Specify the maximum search depth

    sort_neighbors : function (default=None)
        A function that takes an iterator over nodes as the input, and
        returns an iterable of the same nodes with a custom ordering.
        For example, `sorted` will sort the nodes in increasing order.

    Returns
    -------
    succ: iterator
       (node, successors) iterator where `successors` is the non-empty list of
       successors of `node` in a breadth first search from `source`.
       To appear in the iterator, `node` must have successors.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> dict(nx.bfs_successors(G, 0))
    {0: [1], 1: [2]}
    >>> H = nx.Graph()
    >>> H.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    >>> dict(nx.bfs_successors(H, 0))
    {0: [1, 2], 1: [3, 4], 2: [5, 6]}
    >>> G = nx.Graph()
    >>> nx.add_path(G, [0, 1, 2, 3, 4, 5, 6])
    >>> nx.add_path(G, [2, 7, 8, 9, 10])
    >>> dict(nx.bfs_successors(G, source=1, depth_limit=3))
    {1: [0, 2], 2: [3, 7], 3: [4], 7: [8]}
    >>> G = nx.DiGraph()
    >>> nx.add_path(G, [0, 1, 2, 3, 4, 5])
    >>> dict(nx.bfs_successors(G, source=3))
    {3: [4], 4: [5]}

    Notes
    -----
    Based on http://www.ics.uci.edu/~eppstein/PADS/BFS.py
    by D. Eppstein, July 2004.The modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited-search`_".

    .. _Depth-limited-search: https://en.wikipedia.org/wiki/Depth-limited_search

    See Also
    --------
    bfs_tree
    bfs_edges
    edge_bfs
    """
    parent = source
    children = []
    for p, c in bfs_edges(
        G, source, depth_limit=depth_limit, sort_neighbors=sort_neighbors
    ):
        if p == parent:
            children.append(c)
            continue
        yield (parent, children)
        children = [c]
        parent = p
    yield (parent, children)


@nx._dispatchable
def bfs_layers(G, sources):
    """Returns an iterator of all the layers in breadth-first search traversal.

    Parameters
    ----------
    G : NetworkX graph
        A graph over which to find the layers using breadth-first search.

    sources : node in `G` or list of nodes in `G`
        Specify starting nodes for single source or multiple sources breadth-first search

    Yields
    ------
    layer: list of nodes
        Yields list of nodes at the same distance from sources

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> dict(enumerate(nx.bfs_layers(G, [0, 4])))
    {0: [0, 4], 1: [1, 3], 2: [2]}
    >>> H = nx.Graph()
    >>> H.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    >>> dict(enumerate(nx.bfs_layers(H, [1])))
    {0: [1], 1: [0, 3, 4], 2: [2], 3: [5, 6]}
    >>> dict(enumerate(nx.bfs_layers(H, [1, 6])))
    {0: [1, 6], 1: [0, 3, 4, 2], 2: [5]}
    """
    if sources in G:
        sources = [sources]

    current_layer = list(sources)
    visited = set(sources)

    for source in current_layer:
        if source not in G:
            raise nx.NetworkXError(f"The node {source} is not in the graph.")

    # this is basically BFS, except that the current layer only stores the nodes at
    # same distance from sources at each iteration
    while current_layer:
        yield current_layer
        next_layer = []
        for node in current_layer:
            for child in G[node]:
                if child not in visited:
                    visited.add(child)
                    next_layer.append(child)
        current_layer = next_layer


REVERSE_EDGE = "reverse"
TREE_EDGE = "tree"
FORWARD_EDGE = "forward"
LEVEL_EDGE = "level"


@nx._dispatchable
def bfs_labeled_edges(G, sources):
    """Iterate over edges in a breadth-first search (BFS) labeled by type.

    We generate triple of the form (*u*, *v*, *d*), where (*u*, *v*) is the
    edge being explored in the breadth-first search and *d* is one of the
    strings 'tree', 'forward', 'level', or 'reverse'.  A 'tree' edge is one in
    which *v* is first discovered and placed into the layer below *u*.  A
    'forward' edge is one in which *u* is on the layer above *v* and *v* has
    already been discovered.  A 'level' edge is one in which both *u* and *v*
    occur on the same layer.  A 'reverse' edge is one in which *u* is on a layer
    below *v*.

    We emit each edge exactly once.  In an undirected graph, 'reverse' edges do
    not occur, because each is discovered either as a 'tree' or 'forward' edge.

    Parameters
    ----------
    G : NetworkX graph
        A graph over which to find the layers using breadth-first search.

    sources : node in `G` or list of nodes in `G`
        Starting nodes for single source or multiple sources breadth-first search

    Yields
    ------
    edges: generator
       A generator of triples (*u*, *v*, *d*) where (*u*, *v*) is the edge being
       explored and *d* is described above.

    Examples
    --------
    >>> G = nx.cycle_graph(4, create_using=nx.DiGraph)
    >>> list(nx.bfs_labeled_edges(G, 0))
    [(0, 1, 'tree'), (1, 2, 'tree'), (2, 3, 'tree'), (3, 0, 'reverse')]
    >>> G = nx.complete_graph(3)
    >>> list(nx.bfs_labeled_edges(G, 0))
    [(0, 1, 'tree'), (0, 2, 'tree'), (1, 2, 'level')]
    >>> list(nx.bfs_labeled_edges(G, [0, 1]))
    [(0, 1, 'level'), (0, 2, 'tree'), (1, 2, 'forward')]
    """
    if sources in G:
        sources = [sources]

    neighbors = G._adj
    directed = G.is_directed()
    visited = set()
    visit = visited.discard if directed else visited.add
    # We use visited in a negative sense, so the visited set stays empty for the
    # directed case and level edges are reported on their first occurrence in
    # the undirected case.  Note our use of visited.discard -- this is built-in
    # thus somewhat faster than a python-defined def nop(x): pass
    depth = {s: 0 for s in sources}
    queue = deque(depth.items())
    push = queue.append
    pop = queue.popleft
    while queue:
        u, du = pop()
        for v in neighbors[u]:
            if v not in depth:
                depth[v] = dv = du + 1
                push((v, dv))
                yield u, v, TREE_EDGE
            else:
                dv = depth[v]
                if du == dv:
                    if v not in visited:
                        yield u, v, LEVEL_EDGE
                elif du < dv:
                    yield u, v, FORWARD_EDGE
                elif directed:
                    yield u, v, REVERSE_EDGE
        visit(u)


@nx._dispatchable
def descendants_at_distance(G, source, distance):
    """Returns all nodes at a fixed `distance` from `source` in `G`.

    Parameters
    ----------
    G : NetworkX graph
        A graph
    source : node in `G`
    distance : the distance of the wanted nodes from `source`

    Returns
    -------
    set()
        The descendants of `source` in `G` at the given `distance` from `source`

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.descendants_at_distance(G, 2, 2)
    {0, 4}
    >>> H = nx.DiGraph()
    >>> H.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    >>> nx.descendants_at_distance(H, 0, 2)
    {3, 4, 5, 6}
    >>> nx.descendants_at_distance(H, 5, 0)
    {5}
    >>> nx.descendants_at_distance(H, 5, 1)
    set()
    """
    if source not in G:
        raise nx.NetworkXError(f"The node {source} is not in the graph.")

    bfs_generator = nx.bfs_layers(G, source)
    for i, layer in enumerate(bfs_generator):
        if i == distance:
            return set(layer)
    return set()
<|endoftext|>"""Basic algorithms for depth-first searching the nodes of a graph."""

from collections import defaultdict

import networkx as nx

__all__ = [
    "dfs_edges",
    "dfs_tree",
    "dfs_predecessors",
    "dfs_successors",
    "dfs_preorder_nodes",
    "dfs_postorder_nodes",
    "dfs_labeled_edges",
]


@nx._dispatchable
def dfs_edges(G, source=None, depth_limit=None, *, sort_neighbors=None):
    """Iterate over edges in a depth-first-search (DFS).

    Perform a depth-first-search over the nodes of `G` and yield
    the edges in order. This may not generate all edges in `G`
    (see `~networkx.algorithms.traversal.edgedfs.edge_dfs`).

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
       Specify starting node for depth-first search and yield edges in
       the component reachable from source.

    depth_limit : int, optional (default=len(G))
       Specify the maximum search depth.

    sort_neighbors : function (default=None)
        A function that takes an iterator over nodes as the input, and
        returns an iterable of the same nodes with a custom ordering.
        For example, `sorted` will sort the nodes in increasing order.

    Yields
    ------
    edge: 2-tuple of nodes
       Yields edges resulting from the depth-first-search.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> list(nx.dfs_edges(G, source=0))
    [(0, 1), (1, 2), (2, 3), (3, 4)]
    >>> list(nx.dfs_edges(G, source=0, depth_limit=2))
    [(0, 1), (1, 2)]

    Notes
    -----
    If a source is not specified then a source is chosen arbitrarily and
    repeatedly until all components in the graph are searched.

    The implementation of this function is adapted from David Eppstein's
    depth-first search function in PADS [1]_, with modifications
    to allow depth limits based on the Wikipedia article
    "Depth-limited search" [2]_.

    See Also
    --------
    dfs_preorder_nodes
    dfs_postorder_nodes
    dfs_labeled_edges
    :func:`~networkx.algorithms.traversal.edgedfs.edge_dfs`
    :func:`~networkx.algorithms.traversal.breadth_first_search.bfs_edges`

    References
    ----------
    .. [1] http://www.ics.uci.edu/~eppstein/PADS
    .. [2] https://en.wikipedia.org/wiki/Depth-limited_search
    """
    if source is None:
        # edges for all components
        nodes = G
    else:
        # edges for components with source
        nodes = [source]
    if depth_limit is None:
        depth_limit = len(G)

    get_children = (
        G.neighbors
        if sort_neighbors is None
        else lambda n: iter(sort_neighbors(G.neighbors(n)))
    )

    visited = set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, get_children(start))]
        depth_now = 1
        while stack:
            parent, children = stack[-1]
            for child in children:
                if child not in visited:
                    yield parent, child
                    visited.add(child)
                    if depth_now < depth_limit:
                        stack.append((child, get_children(child)))
                        depth_now += 1
                        break
            else:
                stack.pop()
                depth_now -= 1


@nx._dispatchable(returns_graph=True)
def dfs_tree(G, source=None, depth_limit=None, *, sort_neighbors=None):
    """Returns oriented tree constructed from a depth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
       Specify starting node for depth-first search.

    depth_limit : int, optional (default=len(G))
       Specify the maximum search depth.

    sort_neighbors : function (default=None)
        A function that takes an iterator over nodes as the input, and
        returns an iterable of the same nodes with a custom ordering.
        For example, `sorted` will sort the nodes in increasing order.

    Returns
    -------
    T : NetworkX DiGraph
       An oriented tree

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> T = nx.dfs_tree(G, source=0, depth_limit=2)
    >>> list(T.edges())
    [(0, 1), (1, 2)]
    >>> T = nx.dfs_tree(G, source=0)
    >>> list(T.edges())
    [(0, 1), (1, 2), (2, 3), (3, 4)]

    See Also
    --------
    dfs_preorder_nodes
    dfs_postorder_nodes
    dfs_labeled_edges
    :func:`~networkx.algorithms.traversal.edgedfs.edge_dfs`
    :func:`~networkx.algorithms.traversal.breadth_first_search.bfs_tree`
    """
    T = nx.DiGraph()
    if source is None:
        T.add_nodes_from(G)
    else:
        T.add_node(source)
    T.add_edges_from(dfs_edges(G, source, depth_limit, sort_neighbors=sort_neighbors))
    return T


@nx._dispatchable
def dfs_predecessors(G, source=None, depth_limit=None, *, sort_neighbors=None):
    """Returns dictionary of predecessors in depth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
       Specify starting node for depth-first search.
       Note that you will get predecessors for all nodes in the
       component containing `source`. This input only specifies
       where the DFS starts.

    depth_limit : int, optional (default=len(G))
       Specify the maximum search depth.

    sort_neighbors : function (default=None)
        A function that takes an iterator over nodes as the input, and
        returns an iterable of the same nodes with a custom ordering.
        For example, `sorted` will sort the nodes in increasing order.

    Returns
    -------
    pred: dict
       A dictionary with nodes as keys and predecessor nodes as values.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.dfs_predecessors(G, source=0)
    {1: 0, 2: 1, 3: 2}
    >>> nx.dfs_predecessors(G, source=0, depth_limit=2)
    {1: 0, 2: 1}

    Notes
    -----
    If a source is not specified then a source is chosen arbitrarily and
    repeatedly until all components in the graph are searched.

    The implementation of this function is adapted from David Eppstein's
    depth-first search function in `PADS`_, with modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited search`_".

    .. _PADS: http://www.ics.uci.edu/~eppstein/PADS
    .. _Depth-limited search: https://en.wikipedia.org/wiki/Depth-limited_search

    See Also
    --------
    dfs_preorder_nodes
    dfs_postorder_nodes
    dfs_labeled_edges
    :func:`~networkx.algorithms.traversal.edgedfs.edge_dfs`
    :func:`~networkx.algorithms.traversal.breadth_first_search.bfs_tree`
    """
    return {
        t: s
        for s, t in dfs_edges(G, source, depth_limit, sort_neighbors=sort_neighbors)
    }


@nx._dispatchable
def dfs_successors(G, source=None, depth_limit=None, *, sort_neighbors=None):
    """Returns dictionary of successors in depth-first-search from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
       Specify starting node for depth-first search.
       Note that you will get successors for all nodes in the
       component containing `source`. This input only specifies
       where the DFS starts.

    depth_limit : int, optional (default=len(G))
       Specify the maximum search depth.

    sort_neighbors : function (default=None)
        A function that takes an iterator over nodes as the input, and
        returns an iterable of the same nodes with a custom ordering.
        For example, `sorted` will sort the nodes in increasing order.

    Returns
    -------
    succ: dict
       A dictionary with nodes as keys and list of successor nodes as values.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.dfs_successors(G, source=0)
    {0: [1], 1: [2], 2: [3], 3: [4]}
    >>> nx.dfs_successors(G, source=0, depth_limit=2)
    {0: [1], 1: [2]}

    Notes
    -----
    If a source is not specified then a source is chosen arbitrarily and
    repeatedly until all components in the graph are searched.

    The implementation of this function is adapted from David Eppstein's
    depth-first search function in `PADS`_, with modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited search`_".

    .. _PADS: http://www.ics.uci.edu/~eppstein/PADS
    .. _Depth-limited search: https://en.wikipedia.org/wiki/Depth-limited_search

    See Also
    --------
    dfs_preorder_nodes
    dfs_postorder_nodes
    dfs_labeled_edges
    :func:`~networkx.algorithms.traversal.edgedfs.edge_dfs`
    :func:`~networkx.algorithms.traversal.breadth_first_search.bfs_tree`
    """
    d = defaultdict(list)
    for s, t in dfs_edges(
        G,
        source=source,
        depth_limit=depth_limit,
        sort_neighbors=sort_neighbors,
    ):
        d[s].append(t)
    return dict(d)


@nx._dispatchable
def dfs_postorder_nodes(G, source=None, depth_limit=None, *, sort_neighbors=None):
    """Generate nodes in a depth-first-search post-ordering starting at source.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
       Specify starting node for depth-first search.

    depth_limit : int, optional (default=len(G))
       Specify the maximum search depth.

    sort_neighbors : function (default=None)
        A function that takes an iterator over nodes as the input, and
        returns an iterable of the same nodes with a custom ordering.
        For example, `sorted` will sort the nodes in increasing order.

    Returns
    -------
    nodes: generator
       A generator of nodes in a depth-first-search post-ordering.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> list(nx.dfs_postorder_nodes(G, source=0))
    [4, 3, 2, 1, 0]
    >>> list(nx.dfs_postorder_nodes(G, source=0, depth_limit=2))
    [1, 0]

    Notes
    -----
    If a source is not specified then a source is chosen arbitrarily and
    repeatedly until all components in the graph are searched.

    The implementation of this function is adapted from David Eppstein's
    depth-first search function in `PADS`_, with modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited search`_".

    .. _PADS: http://www.ics.uci.edu/~eppstein/PADS
    .. _Depth-limited search: https://en.wikipedia.org/wiki/Depth-limited_search

    See Also
    --------
    dfs_edges
    dfs_preorder_nodes
    dfs_labeled_edges
    :func:`~networkx.algorithms.traversal.edgedfs.edge_dfs`
    :func:`~networkx.algorithms.traversal.breadth_first_search.bfs_tree`
    """
    edges = nx.dfs_labeled_edges(
        G, source=source, depth_limit=depth_limit, sort_neighbors=sort_neighbors
    )
    return (v for u, v, d in edges if d == "reverse")


@nx._dispatchable
def dfs_preorder_nodes(G, source=None, depth_limit=None, *, sort_neighbors=None):
    """Generate nodes in a depth-first-search pre-ordering starting at source.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
       Specify starting node for depth-first search and return nodes in
       the component reachable from source.

    depth_limit : int, optional (default=len(G))
       Specify the maximum search depth.

    sort_neighbors : function (default=None)
        A function that takes an iterator over nodes as the input, and
        returns an iterable of the same nodes with a custom ordering.
        For example, `sorted` will sort the nodes in increasing order.

    Returns
    -------
    nodes: generator
       A generator of nodes in a depth-first-search pre-ordering.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> list(nx.dfs_preorder_nodes(G, source=0))
    [0, 1, 2, 3, 4]
    >>> list(nx.dfs_preorder_nodes(G, source=0, depth_limit=2))
    [0, 1, 2]

    Notes
    -----
    If a source is not specified then a source is chosen arbitrarily and
    repeatedly until all components in the graph are searched.

    The implementation of this function is adapted from David Eppstein's
    depth-first search function in `PADS`_, with modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited search`_".

    .. _PADS: http://www.ics.uci.edu/~eppstein/PADS
    .. _Depth-limited search: https://en.wikipedia.org/wiki/Depth-limited_search

    See Also
    --------
    dfs_edges
    dfs_postorder_nodes
    dfs_labeled_edges
    :func:`~networkx.algorithms.traversal.breadth_first_search.bfs_edges`
    """
    edges = nx.dfs_labeled_edges(
        G, source=source, depth_limit=depth_limit, sort_neighbors=sort_neighbors
    )
    return (v for u, v, d in edges if d == "forward")


@nx._dispatchable
def dfs_labeled_edges(G, source=None, depth_limit=None, *, sort_neighbors=None):
    """Iterate over edges in a depth-first-search (DFS) labeled by type.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
       Specify starting node for depth-first search and return edges in
       the component reachable from source.

    depth_limit : int, optional (default=len(G))
       Specify the maximum search depth.

    sort_neighbors : function (default=None)
        A function that takes an iterator over nodes as the input, and
        returns an iterable of the same nodes with a custom ordering.
        For example, `sorted` will sort the nodes in increasing order.

    Returns
    -------
    edges: generator
       A generator of triples of the form (*u*, *v*, *d*), where (*u*,
       *v*) is the edge being explored in the depth-first search and *d*
       is one of the strings 'forward', 'nontree', 'reverse', or 'reverse-depth_limit'.
       A 'forward' edge is one in which *u* has been visited but *v* has
       not. A 'nontree' edge is one in which both *u* and *v* have been
       visited but the edge is not in the DFS tree. A 'reverse' edge is
       one in which both *u* and *v* have been visited and the edge is in
       the DFS tree. When the `depth_limit` is reached via a 'forward' edge,
       a 'reverse' edge is immediately generated rather than the subtree
       being explored. To indicate this flavor of 'reverse' edge, the string
       yielded is 'reverse-depth_limit'.

    Examples
    --------

    The labels reveal the complete transcript of the depth-first search
    algorithm in more detail than, for example, :func:`dfs_edges`::

        >>> from pprint import pprint
        >>>
        >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 1)])
        >>> pprint(list(nx.dfs_labeled_edges(G, source=0)))
        [(0, 0, 'forward'),
         (0, 1, 'forward'),
         (1, 2, 'forward'),
         (2, 1, 'nontree'),
         (1, 2, 'reverse'),
         (0, 1, 'reverse'),
         (0, 0, 'reverse')]

    Notes
    -----
    If a source is not specified then a source is chosen arbitrarily and
    repeatedly until all components in the graph are searched.

    The implementation of this function is adapted from David Eppstein's
    depth-first search function in `PADS`_, with modifications
    to allow depth limits based on the Wikipedia article
    "`Depth-limited search`_".

    .. _PADS: http://www.ics.uci.edu/~eppstein/PADS
    .. _Depth-limited search: https://en.wikipedia.org/wiki/Depth-limited_search

    See Also
    --------
    dfs_edges
    dfs_preorder_nodes
    dfs_postorder_nodes
    """
    # Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py
    # by D. Eppstein, July 2004.
    if source is None:
        # edges for all components
        nodes = G
    else:
        # edges for components with source
        nodes = [source]
    if depth_limit is None:
        depth_limit = len(G)

    get_children = (
        G.neighbors
        if sort_neighbors is None
        else lambda n: iter(sort_neighbors(G.neighbors(n)))
    )

    visited = set()
    for start in nodes:
        if start in visited:
            continue
        yield start, start, "forward"
        visited.add(start)
        stack = [(start, get_children(start))]
        depth_now = 1
        while stack:
            parent, children = stack[-1]
            for child in children:
                if child in visited:
                    yield parent, child, "nontree"
                else:
                    yield parent, child, "forward"
                    visited.add(child)
                    if depth_now < depth_limit:
                        stack.append((child, iter(get_children(child))))
                        depth_now += 1
                        break
                    else:
                        yield parent, child, "reverse-depth_limit"
            else:
                stack.pop()
                depth_now -= 1
                if stack:
                    yield stack[-1][0], parent, "reverse"
        yield start, start, "reverse"
<|endoftext|>"""
*************************
Multi-line Adjacency List
*************************
Read and write NetworkX graphs as multi-line adjacency lists.

The multi-line adjacency list format is useful for graphs with
nodes that can be meaningfully represented as strings.  With this format
simple edge data can be stored but node or graph data is not.

Format
------
The first label in a line is the source node label followed by the node degree
d.  The next d lines are target node labels and optional edge data.
That pattern repeats for all nodes in the graph.

The graph with edges a-b, a-c, d-e can be represented as the following
adjacency list (anything following the # in a line is a comment)::

     # example.multiline-adjlist
     a 2
     b
     c
     d 1
     e
"""

__all__ = [
    "generate_multiline_adjlist",
    "write_multiline_adjlist",
    "parse_multiline_adjlist",
    "read_multiline_adjlist",
]

import networkx as nx
from networkx.utils import open_file


def generate_multiline_adjlist(G, delimiter=" "):
    """Generate a single line of the graph G in multiline adjacency list format.

    Parameters
    ----------
    G : NetworkX graph

    delimiter : string, optional
       Separator for node labels

    Returns
    -------
    lines : string
        Lines of data in multiline adjlist format.

    Examples
    --------
    >>> G = nx.lollipop_graph(4, 3)
    >>> for line in nx.generate_multiline_adjlist(G):
    ...     print(line)
    0 3
    1 {}
    2 {}
    3 {}
    1 2
    2 {}
    3 {}
    2 1
    3 {}
    3 1
    4 {}
    4 1
    5 {}
    5 1
    6 {}
    6 0

    See Also
    --------
    write_multiline_adjlist, read_multiline_adjlist
    """
    if G.is_directed():
        if G.is_multigraph():
            for s, nbrs in G.adjacency():
                nbr_edges = [
                    (u, data)
                    for u, datadict in nbrs.items()
                    for key, data in datadict.items()
                ]
                deg = len(nbr_edges)
                yield str(s) + delimiter + str(deg)
                for u, d in nbr_edges:
                    if d is None:
                        yield str(u)
                    else:
                        yield str(u) + delimiter + str(d)
        else:  # directed single edges
            for s, nbrs in G.adjacency():
                deg = len(nbrs)
                yield str(s) + delimiter + str(deg)
                for u, d in nbrs.items():
                    if d is None:
                        yield str(u)
                    else:
                        yield str(u) + delimiter + str(d)
    else:  # undirected
        if G.is_multigraph():
            seen = set()  # helper dict used to avoid duplicate edges
            for s, nbrs in G.adjacency():
                nbr_edges = [
                    (u, data)
                    for u, datadict in nbrs.items()
                    if u not in seen
                    for key, data in datadict.items()
                ]
                deg = len(nbr_edges)
                yield str(s) + delimiter + str(deg)
                for u, d in nbr_edges:
                    if d is None:
                        yield str(u)
                    else:
                        yield str(u) + delimiter + str(d)
                seen.add(s)
        else:  # undirected single edges
            seen = set()  # helper dict used to avoid duplicate edges
            for s, nbrs in G.adjacency():
                nbr_edges = [(u, d) for u, d in nbrs.items() if u not in seen]
                deg = len(nbr_edges)
                yield str(s) + delimiter + str(deg)
                for u, d in nbr_edges:
                    if d is None:
                        yield str(u)
                    else:
                        yield str(u) + delimiter + str(d)
                seen.add(s)


@open_file(1, mode="wb")
def write_multiline_adjlist(G, path, delimiter=" ", comments="#", encoding="utf-8"):
    """Write the graph G in multiline adjacency list format to path

    Parameters
    ----------
    G : NetworkX graph

    path : string or file
       Filename or file handle to write to.
       Filenames ending in .gz or .bz2 will be compressed.

    comments : string, optional
       Marker for comment lines

    delimiter : string, optional
       Separator for node labels

    encoding : string, optional
       Text encoding.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_multiline_adjlist(G, "test.adjlist")

    The path can be a file handle or a string with the name of the file. If a
    file handle is provided, it has to be opened in 'wb' mode.

    >>> fh = open("test.adjlist", "wb")
    >>> nx.write_multiline_adjlist(G, fh)

    Filenames ending in .gz or .bz2 will be compressed.

    >>> nx.write_multiline_adjlist(G, "test.adjlist.gz")

    See Also
    --------
    read_multiline_adjlist
    """
    import sys
    import time

    pargs = comments + " ".join(sys.argv)
    header = (
        f"{pargs}\n"
        + comments
        + f" GMT {time.asctime(time.gmtime())}\n"
        + comments
        + f" {G.name}\n"
    )
    path.write(header.encode(encoding))

    for multiline in generate_multiline_adjlist(G, delimiter):
        multiline += "\n"
        path.write(multiline.encode(encoding))


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_multiline_adjlist(
    lines, comments="#", delimiter=None, create_using=None, nodetype=None, edgetype=None
):
    """Parse lines of a multiline adjacency list representation of a graph.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in multiline adjlist format

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    nodetype : Python type, optional
       Convert nodes to this type.

    edgetype : Python type, optional
       Convert edges to this type.

    comments : string, optional
       Marker for comment lines

    delimiter : string, optional
       Separator for node labels.  The default is whitespace.

    Returns
    -------
    G: NetworkX graph
        The graph corresponding to the lines in multiline adjacency list format.

    Examples
    --------
    >>> lines = [
    ...     "1 2",
    ...     "2 {'weight':3, 'name': 'Frodo'}",
    ...     "3 {}",
    ...     "2 1",
    ...     "5 {'weight':6, 'name': 'Saruman'}",
    ... ]
    >>> G = nx.parse_multiline_adjlist(iter(lines), nodetype=int)
    >>> list(G)
    [1, 2, 3, 5]

    """
    from ast import literal_eval

    G = nx.empty_graph(0, create_using)
    for line in lines:
        p = line.find(comments)
        if p >= 0:
            line = line[:p]
        if not line:
            continue
        try:
            (u, deg) = line.rstrip("\n").split(delimiter)
            deg = int(deg)
        except BaseException as err:
            raise TypeError(f"Failed to read node and degree on line ({line})") from err
        if nodetype is not None:
            try:
                u = nodetype(u)
            except BaseException as err:
                raise TypeError(
                    f"Failed to convert node ({u}) to type {nodetype}"
                ) from err
        G.add_node(u)
        for i in range(deg):
            while True:
                try:
                    line = next(lines)
                except StopIteration as err:
                    msg = f"Failed to find neighbor for node ({u})"
                    raise TypeError(msg) from err
                p = line.find(comments)
                if p >= 0:
                    line = line[:p]
                if line:
                    break
            vlist = line.rstrip("\n").split(delimiter)
            numb = len(vlist)
            if numb < 1:
                continue  # isolated node
            v = vlist.pop(0)
            data = "".join(vlist)
            if nodetype is not None:
                try:
                    v = nodetype(v)
                except BaseException as err:
                    raise TypeError(
                        f"Failed to convert node ({v}) to type {nodetype}"
                    ) from err
            if edgetype is not None:
                try:
                    edgedata = {"weight": edgetype(data)}
                except BaseException as err:
                    raise TypeError(
                        f"Failed to convert edge data ({data}) to type {edgetype}"
                    ) from err
            else:
                try:  # try to evaluate
                    edgedata = literal_eval(data)
                except:
                    edgedata = {}
            G.add_edge(u, v, **edgedata)

    return G


@open_file(0, mode="rb")
@nx._dispatchable(graphs=None, returns_graph=True)
def read_multiline_adjlist(
    path,
    comments="#",
    delimiter=None,
    create_using=None,
    nodetype=None,
    edgetype=None,
    encoding="utf-8",
):
    """Read graph in multi-line adjacency list format from path.

    Parameters
    ----------
    path : string or file
       Filename or file handle to read.
       Filenames ending in .gz or .bz2 will be uncompressed.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    nodetype : Python type, optional
       Convert nodes to this type.

    edgetype : Python type, optional
       Convert edge data to this type.

    comments : string, optional
       Marker for comment lines

    delimiter : string, optional
       Separator for node labels.  The default is whitespace.

    Returns
    -------
    G: NetworkX graph

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_multiline_adjlist(G, "test.adjlist")
    >>> G = nx.read_multiline_adjlist("test.adjlist")

    The path can be a file or a string with the name of the file. If a
    file s provided, it has to be opened in 'rb' mode.

    >>> fh = open("test.adjlist", "rb")
    >>> G = nx.read_multiline_adjlist(fh)

    Filenames ending in .gz or .bz2 will be compressed.

    >>> nx.write_multiline_adjlist(G, "test.adjlist.gz")
    >>> G = nx.read_multiline_adjlist("test.adjlist.gz")

    The optional nodetype is a function to convert node strings to nodetype.

    For example

    >>> G = nx.read_multiline_adjlist("test.adjlist", nodetype=int)

    will attempt to convert all nodes to integer type.

    The optional edgetype is a function to convert edge data strings to
    edgetype.

    >>> G = nx.read_multiline_adjlist("test.adjlist")

    The optional create_using parameter is a NetworkX graph container.
    The default is Graph(), an undirected graph.  To read the data as
    a directed graph use

    >>> G = nx.read_multiline_adjlist("test.adjlist", create_using=nx.DiGraph)

    Notes
    -----
    This format does not store graph, node, or edge data.

    See Also
    --------
    write_multiline_adjlist
    """
    lines = (line.decode(encoding) for line in path)
    return parse_multiline_adjlist(
        lines,
        comments=comments,
        delimiter=delimiter,
        create_using=create_using,
        nodetype=nodetype,
        edgetype=edgetype,
    )
<|endoftext|>"""
Read graphs in LEDA format.

LEDA is a C++ class library for efficient data types and algorithms.

Format
------
See http://www.algorithmic-solutions.info/leda_guide/graphs/leda_native_graph_fileformat.html

"""
# Original author: D. Eppstein, UC Irvine, August 12, 2003.
# The original code at http://www.ics.uci.edu/~eppstein/PADS/ is public domain.

__all__ = ["read_leda", "parse_leda"]

import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file


@open_file(0, mode="rb")
@nx._dispatchable(graphs=None, returns_graph=True)
def read_leda(path, encoding="UTF-8"):
    """Read graph in LEDA format from path.

    Parameters
    ----------
    path : file or string
       File or filename to read.  Filenames ending in .gz or .bz2  will be
       uncompressed.

    Returns
    -------
    G : NetworkX graph

    Examples
    --------
    G=nx.read_leda('file.leda')

    References
    ----------
    .. [1] http://www.algorithmic-solutions.info/leda_guide/graphs/leda_native_graph_fileformat.html
    """
    lines = (line.decode(encoding) for line in path)
    G = parse_leda(lines)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_leda(lines):
    """Read graph in LEDA format from string or iterable.

    Parameters
    ----------
    lines : string or iterable
       Data in LEDA format.

    Returns
    -------
    G : NetworkX graph

    Examples
    --------
    G=nx.parse_leda(string)

    References
    ----------
    .. [1] http://www.algorithmic-solutions.info/leda_guide/graphs/leda_native_graph_fileformat.html
    """
    if isinstance(lines, str):
        lines = iter(lines.split("\n"))
    lines = iter(
        [
            line.rstrip("\n")
            for line in lines
            if not (line.startswith(("#", "\n")) or line == "")
        ]
    )
    for i in range(3):
        next(lines)
    # Graph
    du = int(next(lines))  # -1=directed, -2=undirected
    if du == -1:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Nodes
    n = int(next(lines))  # number of nodes
    node = {}
    for i in range(1, n + 1):  # LEDA counts from 1 to n
        symbol = next(lines).rstrip().strip("|{}|  ")
        if symbol == "":
            symbol = str(i)  # use int if no label - could be trouble
        node[i] = symbol

    G.add_nodes_from([s for i, s in node.items()])

    # Edges
    m = int(next(lines))  # number of edges
    for i in range(m):
        try:
            s, t, reversal, label = next(lines).split()
        except BaseException as err:
            raise NetworkXError(f"Too few fields in LEDA.GRAPH edge {i+1}") from err
        # BEWARE: no handling of reversal edges
        G.add_edge(node[int(s)], node[int(t)], label=label[2:-2])
    return G
<|endoftext|>"""
Text-based visual representations of graphs
"""

import sys
import warnings
from collections import defaultdict

import networkx as nx
from networkx.utils import open_file

__all__ = ["generate_network_text", "write_network_text"]


class BaseGlyphs:
    @classmethod
    def as_dict(cls):
        return {
            a: getattr(cls, a)
            for a in dir(cls)
            if not a.startswith("_") and a != "as_dict"
        }


class AsciiBaseGlyphs(BaseGlyphs):
    empty: str = "+"
    newtree_last: str = "+-- "
    newtree_mid: str = "+-- "
    endof_forest: str = "    "
    within_forest: str = ":   "
    within_tree: str = "|   "


class AsciiDirectedGlyphs(AsciiBaseGlyphs):
    last: str = "L-> "
    mid: str = "|-> "
    backedge: str = "<-"
    vertical_edge: str = "!"


class AsciiUndirectedGlyphs(AsciiBaseGlyphs):
    last: str = "L-- "
    mid: str = "|-- "
    backedge: str = "-"
    vertical_edge: str = "|"


class UtfBaseGlyphs(BaseGlyphs):
    # Notes on available box and arrow characters
    # https://en.wikipedia.org/wiki/Box-drawing_character
    # https://stackoverflow.com/questions/2701192/triangle-arrow
    empty: str = "╙"
    newtree_last: str = "╙── "
    newtree_mid: str = "╟── "
    endof_forest: str = "    "
    within_forest: str = "╎   "
    within_tree: str = "│   "


class UtfDirectedGlyphs(UtfBaseGlyphs):
    last: str = "└─╼ "
    mid: str = "├─╼ "
    backedge: str = "╾"
    vertical_edge: str = "╽"


class UtfUndirectedGlyphs(UtfBaseGlyphs):
    last: str = "└── "
    mid: str = "├── "
    backedge: str = "─"
    vertical_edge: str = "│"


def generate_network_text(
    graph,
    with_labels=True,
    sources=None,
    max_depth=None,
    ascii_only=False,
    vertical_chains=False,
):
    """Generate lines in the "network text" format

    This works via a depth-first traversal of the graph and writing a line for
    each unique node encountered. Non-tree edges are written to the right of
    each node, and connection to a non-tree edge is indicated with an ellipsis.
    This representation works best when the input graph is a forest, but any
    graph can be represented.

    This notation is original to networkx, although it is simple enough that it
    may be known in existing literature. See #5602 for details. The procedure
    is summarized as follows:

    1. Given a set of source nodes (which can be specified, or automatically
    discovered via finding the (strongly) connected components and choosing one
    node with minimum degree from each), we traverse the graph in depth first
    order.

    2. Each reachable node will be printed exactly once on it's own line.

    3. Edges are indicated in one of four ways:

        a. a parent "L-style" connection on the upper left. This corresponds to
        a traversal in the directed DFS tree.

        b. a backref "<-style" connection shown directly on the right. For
        directed graphs, these are drawn for any incoming edges to a node that
        is not a parent edge. For undirected graphs, these are drawn for only
        the non-parent edges that have already been represented (The edges that
        have not been represented will be handled in the recursive case).

        c. a child "L-style" connection on the lower right. Drawing of the
        children are handled recursively.

        d. if ``vertical_chains`` is true, and a parent node only has one child
        a "vertical-style" edge is drawn between them.

    4. The children of each node (wrt the directed DFS tree) are drawn
    underneath and to the right of it. In the case that a child node has already
    been drawn the connection is replaced with an ellipsis ("...") to indicate
    that there is one or more connections represented elsewhere.

    5. If a maximum depth is specified, an edge to nodes past this maximum
    depth will be represented by an ellipsis.

    6. If a node has a truthy "collapse" value, then we do not traverse past
    that node.

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent

    with_labels : bool | str
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. If given as a
        string, then that attribute name will be used instead of "label".
        Defaults to True.

    sources : List
        Specifies which nodes to start traversal from. Note: nodes that are not
        reachable from one of these sources may not be shown. If unspecified,
        the minimal set of nodes needed to reach all others will be used.

    max_depth : int | None
        The maximum depth to traverse before stopping. Defaults to None.

    ascii_only : Boolean
        If True only ASCII characters are used to construct the visualization

    vertical_chains : Boolean
        If True, chains of nodes will be drawn vertically when possible.

    Yields
    ------
    str : a line of generated text

    Examples
    --------
    >>> graph = nx.path_graph(10)
    >>> graph.add_node("A")
    >>> graph.add_node("B")
    >>> graph.add_node("C")
    >>> graph.add_node("D")
    >>> graph.add_edge(9, "A")
    >>> graph.add_edge(9, "B")
    >>> graph.add_edge(9, "C")
    >>> graph.add_edge("C", "D")
    >>> graph.add_edge("C", "E")
    >>> graph.add_edge("C", "F")
    >>> nx.write_network_text(graph)
    ╙── 0
        └── 1
            └── 2
                └── 3
                    └── 4
                        └── 5
                            └── 6
                                └── 7
                                    └── 8
                                        └── 9
                                            ├── A
                                            ├── B
                                            └── C
                                                ├── D
                                                ├── E
                                                └── F
    >>> nx.write_network_text(graph, vertical_chains=True)
    ╙── 0
        │
        1
        │
        2
        │
        3
        │
        4
        │
        5
        │
        6
        │
        7
        │
        8
        │
        9
        ├── A
        ├── B
        └── C
            ├── D
            ├── E
            └── F
    """
    from typing import Any, NamedTuple

    class StackFrame(NamedTuple):
        parent: Any
        node: Any
        indents: list
        this_islast: bool
        this_vertical: bool

    collapse_attr = "collapse"

    is_directed = graph.is_directed()

    if is_directed:
        glyphs = AsciiDirectedGlyphs if ascii_only else UtfDirectedGlyphs
        succ = graph.succ
        pred = graph.pred
    else:
        glyphs = AsciiUndirectedGlyphs if ascii_only else UtfUndirectedGlyphs
        succ = graph.adj
        pred = graph.adj

    if isinstance(with_labels, str):
        label_attr = with_labels
    elif with_labels:
        label_attr = "label"
    else:
        label_attr = None

    if max_depth == 0:
        yield glyphs.empty + " ..."
    elif len(graph.nodes) == 0:
        yield glyphs.empty
    else:
        # If the nodes to traverse are unspecified, find the minimal set of
        # nodes that will reach the entire graph
        if sources is None:
            sources = _find_sources(graph)

        # Populate the stack with each:
        # 1. parent node in the DFS tree (or None for root nodes),
        # 2. the current node in the DFS tree
        # 2. a list of indentations indicating depth
        # 3. a flag indicating if the node is the final one to be written.
        # Reverse the stack so sources are popped in the correct order.
        last_idx = len(sources) - 1
        stack = [
            StackFrame(None, node, [], (idx == last_idx), False)
            for idx, node in enumerate(sources)
        ][::-1]

        num_skipped_children = defaultdict(lambda: 0)
        seen_nodes = set()
        while stack:
            parent, node, indents, this_islast, this_vertical = stack.pop()

            if node is not Ellipsis:
                skip = node in seen_nodes
                if skip:
                    # Mark that we skipped a parent's child
                    num_skipped_children[parent] += 1

                if this_islast:
                    # If we reached the last child of a parent, and we skipped
                    # any of that parents children, then we should emit an
                    # ellipsis at the end after this.
                    if num_skipped_children[parent] and parent is not None:
                        # Append the ellipsis to be emitted last
                        next_islast = True
                        try_frame = StackFrame(
                            node, Ellipsis, indents, next_islast, False
                        )
                        stack.append(try_frame)

                        # Redo this frame, but not as a last object
                        next_islast = False
                        try_frame = StackFrame(
                            parent, node, indents, next_islast, this_vertical
                        )
                        stack.append(try_frame)
                        continue

                if skip:
                    continue
                seen_nodes.add(node)

            if not indents:
                # Top level items (i.e. trees in the forest) get different
                # glyphs to indicate they are not actually connected
                if this_islast:
                    this_vertical = False
                    this_prefix = indents + [glyphs.newtree_last]
                    next_prefix = indents + [glyphs.endof_forest]
                else:
                    this_prefix = indents + [glyphs.newtree_mid]
                    next_prefix = indents + [glyphs.within_forest]

            else:
                # Non-top-level items
                if this_vertical:
                    this_prefix = indents
                    next_prefix = indents
                else:
                    if this_islast:
                        this_prefix = indents + [glyphs.last]
                        next_prefix = indents + [glyphs.endof_forest]
                    else:
                        this_prefix = indents + [glyphs.mid]
                        next_prefix = indents + [glyphs.within_tree]

            if node is Ellipsis:
                label = " ..."
                suffix = ""
                children = []
            else:
                if label_attr is not None:
                    label = str(graph.nodes[node].get(label_attr, node))
                else:
                    label = str(node)

                # Determine if we want to show the children of this node.
                if collapse_attr is not None:
                    collapse = graph.nodes[node].get(collapse_attr, False)
                else:
                    collapse = False

                # Determine:
                # (1) children to traverse into after showing this node.
                # (2) parents to immediately show to the right of this node.
                if is_directed:
                    # In the directed case we must show every successor node
                    # note: it may be skipped later, but we don't have that
                    # information here.
                    children = list(succ[node])
                    # In the directed case we must show every predecessor
                    # except for parent we directly traversed from.
                    handled_parents = {parent}
                else:
                    # Showing only the unseen children results in a more
                    # concise representation for the undirected case.
                    children = [
                        child for child in succ[node] if child not in seen_nodes
                    ]

                    # In the undirected case, parents are also children, so we
                    # only need to immediately show the ones we can no longer
                    # traverse
                    handled_parents = {*children, parent}

                if max_depth is not None and len(indents) == max_depth - 1:
                    # Use ellipsis to indicate we have reached maximum depth
                    if children:
                        children = [Ellipsis]
                    handled_parents = {parent}

                if collapse:
                    # Collapsing a node is the same as reaching maximum depth
                    if children:
                        children = [Ellipsis]
                    handled_parents = {parent}

                # The other parents are other predecessors of this node that
                # are not handled elsewhere.
                other_parents = [p for p in pred[node] if p not in handled_parents]
                if other_parents:
                    if label_attr is not None:
                        other_parents_labels = ", ".join(
                            [
                                str(graph.nodes[p].get(label_attr, p))
                                for p in other_parents
                            ]
                        )
                    else:
                        other_parents_labels = ", ".join(
                            [str(p) for p in other_parents]
                        )
                    suffix = " ".join(["", glyphs.backedge, other_parents_labels])
                else:
                    suffix = ""

            # Emit the line for this node, this will be called for each node
            # exactly once.
            if this_vertical:
                yield "".join(this_prefix + [glyphs.vertical_edge])

            yield "".join(this_prefix + [label, suffix])

            if vertical_chains:
                if is_directed:
                    num_children = len(set(children))
                else:
                    num_children = len(set(children) - {parent})
                # The next node can be drawn vertically if it is the only
                # remaining child of this node.
                next_is_vertical = num_children == 1
            else:
                next_is_vertical = False

            # Push children on the stack in reverse order so they are popped in
            # the original order.
            for idx, child in enumerate(children[::-1]):
                next_islast = idx == 0
                try_frame = StackFrame(
                    node, child, next_prefix, next_islast, next_is_vertical
                )
                stack.append(try_frame)


@open_file(1, "w")
def write_network_text(
    graph,
    path=None,
    with_labels=True,
    sources=None,
    max_depth=None,
    ascii_only=False,
    end="\n",
    vertical_chains=False,
):
    """Creates a nice text representation of a graph

    This works via a depth-first traversal of the graph and writing a line for
    each unique node encountered. Non-tree edges are written to the right of
    each node, and connection to a non-tree edge is indicated with an ellipsis.
    This representation works best when the input graph is a forest, but any
    graph can be represented.

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent

    path : string or file or callable or None
       Filename or file handle for data output.
       if a function, then it will be called for each generated line.
       if None, this will default to "sys.stdout.write"

    with_labels : bool | str
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. If given as a
        string, then that attribute name will be used instead of "label".
        Defaults to True.

    sources : List
        Specifies which nodes to start traversal from. Note: nodes that are not
        reachable from one of these sources may not be shown. If unspecified,
        the minimal set of nodes needed to reach all others will be used.

    max_depth : int | None
        The maximum depth to traverse before stopping. Defaults to None.

    ascii_only : Boolean
        If True only ASCII characters are used to construct the visualization

    end : string
        The line ending character

    vertical_chains : Boolean
        If True, chains of nodes will be drawn vertically when possible.

    Examples
    --------
    >>> graph = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    >>> nx.write_network_text(graph)
    ╙── 0
        ├─╼ 1
        │   ├─╼ 3
        │   └─╼ 4
        └─╼ 2
            ├─╼ 5
            └─╼ 6

    >>> # A near tree with one non-tree edge
    >>> graph.add_edge(5, 1)
    >>> nx.write_network_text(graph)
    ╙── 0
        ├─╼ 1 ╾ 5
        │   ├─╼ 3
        │   └─╼ 4
        └─╼ 2
            ├─╼ 5
            │   └─╼  ...
            └─╼ 6

    >>> graph = nx.cycle_graph(5)
    >>> nx.write_network_text(graph)
    ╙── 0
        ├── 1
        │   └── 2
        │       └── 3
        │           └── 4 ─ 0
        └──  ...

    >>> graph = nx.cycle_graph(5, nx.DiGraph)
    >>> nx.write_network_text(graph, vertical_chains=True)
    ╙── 0 ╾ 4
        ╽
        1
        ╽
        2
        ╽
        3
        ╽
        4
        └─╼  ...

    >>> nx.write_network_text(graph, vertical_chains=True, ascii_only=True)
    +-- 0 <- 4
        !
        1
        !
        2
        !
        3
        !
        4
        L->  ...

    >>> graph = nx.generators.barbell_graph(4, 2)
    >>> nx.write_network_text(graph, vertical_chains=False)
    ╙── 4
        ├── 5
        │   └── 6
        │       ├── 7
        │       │   ├── 8 ─ 6
        │       │   │   └── 9 ─ 6, 7
        │       │   └──  ...
        │       └──  ...
        └── 3
            ├── 0
            │   ├── 1 ─ 3
            │   │   └── 2 ─ 0, 3
            │   └──  ...
            └──  ...
    >>> nx.write_network_text(graph, vertical_chains=True)
    ╙── 4
        ├── 5
        │   │
        │   6
        │   ├── 7
        │   │   ├── 8 ─ 6
        │   │   │   │
        │   │   │   9 ─ 6, 7
        │   │   └──  ...
        │   └──  ...
        └── 3
            ├── 0
            │   ├── 1 ─ 3
            │   │   │
            │   │   2 ─ 0, 3
            │   └──  ...
            └──  ...

    >>> graph = nx.complete_graph(5, create_using=nx.Graph)
    >>> nx.write_network_text(graph)
    ╙── 0
        ├── 1
        │   ├── 2 ─ 0
        │   │   ├── 3 ─ 0, 1
        │   │   │   └── 4 ─ 0, 1, 2
        │   │   └──  ...
        │   └──  ...
        └──  ...

    >>> graph = nx.complete_graph(3, create_using=nx.DiGraph)
    >>> nx.write_network_text(graph)
    ╙── 0 ╾ 1, 2
        ├─╼ 1 ╾ 2
        │   ├─╼ 2 ╾ 0
        │   │   └─╼  ...
        │   └─╼  ...
        └─╼  ...
    """
    if path is None:
        # The path is unspecified, write to stdout
        _write = sys.stdout.write
    elif hasattr(path, "write"):
        # The path is already an open file
        _write = path.write
    elif callable(path):
        # The path is a custom callable
        _write = path
    else:
        raise TypeError(type(path))

    for line in generate_network_text(
        graph,
        with_labels=with_labels,
        sources=sources,
        max_depth=max_depth,
        ascii_only=ascii_only,
        vertical_chains=vertical_chains,
    ):
        _write(line + end)


def _find_sources(graph):
    """
    Determine a minimal set of nodes such that the entire graph is reachable
    """
    # For each connected part of the graph, choose at least
    # one node as a starting point, preferably without a parent
    if graph.is_directed():
        # Choose one node from each SCC with minimum in_degree
        sccs = list(nx.strongly_connected_components(graph))
        # condensing the SCCs forms a dag, the nodes in this graph with
        # 0 in-degree correspond to the SCCs from which the minimum set
        # of nodes from which all other nodes can be reached.
        scc_graph = nx.condensation(graph, sccs)
        supernode_to_nodes = {sn: [] for sn in scc_graph.nodes()}
        # Note: the order of mapping differs between pypy and cpython
        # so we have to loop over graph nodes for consistency
        mapping = scc_graph.graph["mapping"]
        for n in graph.nodes:
            sn = mapping[n]
            supernode_to_nodes[sn].append(n)
        sources = []
        for sn in scc_graph.nodes():
            if scc_graph.in_degree[sn] == 0:
                scc = supernode_to_nodes[sn]
                node = min(scc, key=lambda n: graph.in_degree[n])
                sources.append(node)
    else:
        # For undirected graph, the entire graph will be reachable as
        # long as we consider one node from every connected component
        sources = [
            min(cc, key=lambda n: graph.degree[n])
            for cc in nx.connected_components(graph)
        ]
        sources = sorted(sources, key=lambda n: graph.degree[n])
    return sources


def _parse_network_text(lines):
    """Reconstructs a graph from a network text representation.

    This is mainly used for testing.  Network text is for display, not
    serialization, as such this cannot parse all network text representations
    because node labels can be ambiguous with the glyphs and indentation used
    to represent edge structure. Additionally, there is no way to determine if
    disconnected graphs were originally directed or undirected.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in network text format

    Returns
    -------
    G: NetworkX graph
        The graph corresponding to the lines in network text format.
    """
    from itertools import chain
    from typing import Any, NamedTuple, Union

    class ParseStackFrame(NamedTuple):
        node: Any
        indent: int
        has_vertical_child: int | None

    initial_line_iter = iter(lines)

    is_ascii = None
    is_directed = None

    ##############
    # Initial Pass
    ##############

    # Do an initial pass over the lines to determine what type of graph it is.
    # Remember what these lines were, so we can reiterate over them in the
    # parsing pass.
    initial_lines = []
    try:
        first_line = next(initial_line_iter)
    except StopIteration:
        ...
    else:
        initial_lines.append(first_line)
        # The first character indicates if it is an ASCII or UTF graph
        first_char = first_line[0]
        if first_char in {
            UtfBaseGlyphs.empty,
            UtfBaseGlyphs.newtree_mid[0],
            UtfBaseGlyphs.newtree_last[0],
        }:
            is_ascii = False
        elif first_char in {
            AsciiBaseGlyphs.empty,
            AsciiBaseGlyphs.newtree_mid[0],
            AsciiBaseGlyphs.newtree_last[0],
        }:
            is_ascii = True
        else:
            raise AssertionError(f"Unexpected first character: {first_char}")

    if is_ascii:
        directed_glyphs = AsciiDirectedGlyphs.as_dict()
        undirected_glyphs = AsciiUndirectedGlyphs.as_dict()
    else:
        directed_glyphs = UtfDirectedGlyphs.as_dict()
        undirected_glyphs = UtfUndirectedGlyphs.as_dict()

    # For both directed / undirected glyphs, determine which glyphs never
    # appear as substrings in the other undirected / directed glyphs.  Glyphs
    # with this property unambiguously indicates if a graph is directed /
    # undirected.
    directed_items = set(directed_glyphs.values())
    undirected_items = set(undirected_glyphs.values())
    unambiguous_directed_items = []
    for item in directed_items:
        other_items = undirected_items
        other_supersets = [other for other in other_items if item in other]
        if not other_supersets:
            unambiguous_directed_items.append(item)
    unambiguous_undirected_items = []
    for item in undirected_items:
        other_items = directed_items
        other_supersets = [other for other in other_items if item in other]
        if not other_supersets:
            unambiguous_undirected_items.append(item)

    for line in initial_line_iter:
        initial_lines.append(line)
        if any(item in line for item in unambiguous_undirected_items):
            is_directed = False
            break
        elif any(item in line for item in unambiguous_directed_items):
            is_directed = True
            break

    if is_directed is None:
        # Not enough information to determine, choose undirected by default
        is_directed = False

    glyphs = directed_glyphs if is_directed else undirected_glyphs

    # the backedge symbol by itself can be ambiguous, but with spaces around it
    # becomes unambiguous.
    backedge_symbol = " " + glyphs["backedge"] + " "

    # Reconstruct an iterator over all of the lines.
    parsing_line_iter = chain(initial_lines, initial_line_iter)

    ##############
    # Parsing Pass
    ##############

    edges = []
    nodes = []
    is_empty = None

    noparent = object()  # sentinel value

    # keep a stack of previous nodes that could be parents of subsequent nodes
    stack = [ParseStackFrame(noparent, -1, None)]

    for line in parsing_line_iter:
        if line == glyphs["empty"]:
            # If the line is the empty glyph, we are done.
            # There shouldn't be anything else after this.
            is_empty = True
            continue

        if backedge_symbol in line:
            # This line has one or more backedges, separate those out
            node_part, backedge_part = line.split(backedge_symbol)
            backedge_nodes = [u.strip() for u in backedge_part.split(", ")]
            # Now the node can be parsed
            node_part = node_part.rstrip()
            prefix, node = node_part.rsplit(" ", 1)
            node = node.strip()
            # Add the backedges to the edge list
            edges.extend([(u, node) for u in backedge_nodes])
        else:
            # No backedge, the tail of this line is the node
            prefix, node = line.rsplit(" ", 1)
            node = node.strip()

        prev = stack.pop()

        if node in glyphs["vertical_edge"]:
            # Previous node is still the previous node, but we know it will
            # have exactly one child, which will need to have its nesting level
            # adjusted.
            modified_prev = ParseStackFrame(
                prev.node,
                prev.indent,
                True,
            )
            stack.append(modified_prev)
            continue

        # The length of the string before the node characters give us a hint
        # about our nesting level. The only case where this doesn't work is
        # when there are vertical chains, which is handled explicitly.
        indent = len(prefix)
        curr = ParseStackFrame(node, indent, None)

        if prev.has_vertical_child:
            # In this case we know prev must be the parent of our current line,
            # so we don't have to search the stack. (which is good because the
            # indentation check wouldn't work in this case).
            ...
        else:
            # If the previous node nesting-level is greater than the current
            # nodes nesting-level than the previous node was the end of a path,
            # and is not our parent. We can safely pop nodes off the stack
            # until we find one with a comparable nesting-level, which is our
            # parent.
            while curr.indent <= prev.indent:
                prev = stack.pop()

        if node == "...":
            # The current previous node is no longer a valid parent,
            # keep it popped from the stack.
            stack.append(prev)
        else:
            # The previous and current nodes may still be parents, so add them
            # back onto the stack.
            stack.append(prev)
            stack.append(curr)

            # Add the node and the edge to its parent to the node / edge lists.
            nodes.append(curr.node)
            if prev.node is not noparent:
                edges.append((prev.node, curr.node))

    if is_empty:
        # Sanity check
        assert len(nodes) == 0

    # Reconstruct the graph
    cls = nx.DiGraph if is_directed else nx.Graph
    new = cls()
    new.add_nodes_from(nodes)
    new.add_edges_from(edges)
    return new
<|endoftext|>"""
*****
Pajek
*****
Read graphs in Pajek format.

This implementation handles directed and undirected graphs including
those with self loops and parallel edges.

Format
------
See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
for format information.

"""

import warnings

import networkx as nx
from networkx.utils import open_file

__all__ = ["read_pajek", "parse_pajek", "generate_pajek", "write_pajek"]


def generate_pajek(G):
    """Generate lines in Pajek graph format.

    Parameters
    ----------
    G : graph
       A Networkx graph

    References
    ----------
    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    """
    if G.name == "":
        name = "NetworkX"
    else:
        name = G.name
    # Apparently many Pajek format readers can't process this line
    # So we'll leave it out for now.
    # yield '*network %s'%name

    # write nodes with attributes
    yield f"*vertices {G.order()}"
    nodes = list(G)
    # make dictionary mapping nodes to integers
    nodenumber = dict(zip(nodes, range(1, len(nodes) + 1)))
    for n in nodes:
        # copy node attributes and pop mandatory attributes
        # to avoid duplication.
        na = G.nodes.get(n, {}).copy()
        x = na.pop("x", 0.0)
        y = na.pop("y", 0.0)
        try:
            id = int(na.pop("id", nodenumber[n]))
        except ValueError as err:
            err.args += (
                (
                    "Pajek format requires 'id' to be an int()."
                    " Refer to the 'Relabeling nodes' section."
                ),
            )
            raise
        nodenumber[n] = id
        shape = na.pop("shape", "ellipse")
        s = " ".join(map(make_qstr, (id, n, x, y, shape)))
        # only optional attributes are left in na.
        for k, v in na.items():
            if isinstance(v, str) and v.strip() != "":
                s += f" {make_qstr(k)} {make_qstr(v)}"
            else:
                warnings.warn(
                    f"Node attribute {k} is not processed. {('Empty attribute' if isinstance(v, str) else 'Non-string attribute')}."
                )
        yield s

    # write edges with attributes
    if G.is_directed():
        yield "*arcs"
    else:
        yield "*edges"
    for u, v, edgedata in G.edges(data=True):
        d = edgedata.copy()
        value = d.pop("weight", 1.0)  # use 1 as default edge value
        s = " ".join(map(make_qstr, (nodenumber[u], nodenumber[v], value)))
        for k, v in d.items():
            if isinstance(v, str) and v.strip() != "":
                s += f" {make_qstr(k)} {make_qstr(v)}"
            else:
                warnings.warn(
                    f"Edge attribute {k} is not processed. {('Empty attribute' if isinstance(v, str) else 'Non-string attribute')}."
                )
        yield s


@open_file(1, mode="wb")
def write_pajek(G, path, encoding="UTF-8"):
    """Write graph in Pajek format to path.

    Parameters
    ----------
    G : graph
       A Networkx graph
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_pajek(G, "test.net")

    Warnings
    --------
    Optional node attributes and edge attributes must be non-empty strings.
    Otherwise it will not be written into the file. You will need to
    convert those attributes to strings if you want to keep them.

    References
    ----------
    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    """
    for line in generate_pajek(G):
        line += "\n"
        path.write(line.encode(encoding))


@open_file(0, mode="rb")
@nx._dispatchable(graphs=None, returns_graph=True)
def read_pajek(path, encoding="UTF-8"):
    """Read graph in Pajek format from path.

    Parameters
    ----------
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be uncompressed.

    Returns
    -------
    G : NetworkX MultiGraph or MultiDiGraph.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_pajek(G, "test.net")
    >>> G = nx.read_pajek("test.net")

    To create a Graph instead of a MultiGraph use

    >>> G1 = nx.Graph(G)

    References
    ----------
    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    """
    lines = (line.decode(encoding) for line in path)
    return parse_pajek(lines)


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_pajek(lines):
    """Parse Pajek format graph from string or iterable.

    Parameters
    ----------
    lines : string or iterable
       Data in Pajek format.

    Returns
    -------
    G : NetworkX graph

    See Also
    --------
    read_pajek

    """
    import shlex

    # multigraph=False
    if isinstance(lines, str):
        lines = iter(lines.split("\n"))
    lines = iter([line.rstrip("\n") for line in lines])
    G = nx.MultiDiGraph()  # are multiedges allowed in Pajek? assume yes
    labels = []  # in the order of the file, needed for matrix
    while lines:
        try:
            l = next(lines)
        except:  # EOF
            break
        if l.lower().startswith("*network"):
            try:
                label, name = l.split(None, 1)
            except ValueError:
                # Line was not of the form:  *network NAME
                pass
            else:
                G.graph["name"] = name
        elif l.lower().startswith("*vertices"):
            nodelabels = {}
            l, nnodes = l.split()
            for i in range(int(nnodes)):
                l = next(lines)
                try:
                    splitline = [
                        x.decode("utf-8") for x in shlex.split(str(l).encode("utf-8"))
                    ]
                except AttributeError:
                    splitline = shlex.split(str(l))
                id, label = splitline[0:2]
                labels.append(label)
                G.add_node(label)
                nodelabels[id] = label
                G.nodes[label]["id"] = id
                try:
                    x, y, shape = splitline[2:5]
                    G.nodes[label].update(
                        {"x": float(x), "y": float(y), "shape": shape}
                    )
                except:
                    pass
                extra_attr = zip(splitline[5::2], splitline[6::2])
                G.nodes[label].update(extra_attr)
        elif l.lower().startswith("*edges") or l.lower().startswith("*arcs"):
            if l.lower().startswith("*edge"):
                # switch from multidigraph to multigraph
                G = nx.MultiGraph(G)
            if l.lower().startswith("*arcs"):
                # switch to directed with multiple arcs for each existing edge
                G = G.to_directed()
            for l in lines:
                try:
                    splitline = [
                        x.decode("utf-8") for x in shlex.split(str(l).encode("utf-8"))
                    ]
                except AttributeError:
                    splitline = shlex.split(str(l))

                if len(splitline) < 2:
                    continue
                ui, vi = splitline[0:2]
                u = nodelabels.get(ui, ui)
                v = nodelabels.get(vi, vi)
                # parse the data attached to this edge and put in a dictionary
                edge_data = {}
                try:
                    # there should always be a single value on the edge?
                    w = splitline[2:3]
                    edge_data.update({"weight": float(w[0])})
                except:
                    pass
                    # if there isn't, just assign a 1
                #                    edge_data.update({'value':1})
                extra_attr = zip(splitline[3::2], splitline[4::2])
                edge_data.update(extra_attr)
                # if G.has_edge(u,v):
                #     multigraph=True
                G.add_edge(u, v, **edge_data)
        elif l.lower().startswith("*matrix"):
            G = nx.DiGraph(G)
            adj_list = (
                (labels[row], labels[col], {"weight": int(data)})
                for (row, line) in enumerate(lines)
                for (col, data) in enumerate(line.split())
                if int(data) != 0
            )
            G.add_edges_from(adj_list)

    return G


def make_qstr(t):
    """Returns the string representation of t.
    Add outer double-quotes if the string has a space.
    """
    if not isinstance(t, str):
        t = str(t)
    if " " in t:
        t = f'"{t}"'
    return t
<|endoftext|>"""
*******
GraphML
*******
Read and write graphs in GraphML format.

.. warning::

    This parser uses the standard xml library present in Python, which is
    insecure - see :external+python:mod:`xml` for additional information.
    Only parse GraphML files you trust.

This implementation does not support mixed graphs (directed and unidirected
edges together), hyperedges, nested graphs, or ports.

"GraphML is a comprehensive and easy-to-use file format for graphs. It
consists of a language core to describe the structural properties of a
graph and a flexible extension mechanism to add application-specific
data. Its main features include support of

    * directed, undirected, and mixed graphs,
    * hypergraphs,
    * hierarchical graphs,
    * graphical representations,
    * references to external data,
    * application-specific attribute data, and
    * light-weight parsers.

Unlike many other file formats for graphs, GraphML does not use a
custom syntax. Instead, it is based on XML and hence ideally suited as
a common denominator for all kinds of services generating, archiving,
or processing graphs."

http://graphml.graphdrawing.org/

Format
------
GraphML is an XML format.  See
http://graphml.graphdrawing.org/specification.html for the specification and
http://graphml.graphdrawing.org/primer/graphml-primer.html
for examples.
"""

import warnings
from collections import defaultdict

import networkx as nx
from networkx.utils import open_file

__all__ = [
    "write_graphml",
    "read_graphml",
    "generate_graphml",
    "write_graphml_xml",
    "write_graphml_lxml",
    "parse_graphml",
    "GraphMLWriter",
    "GraphMLReader",
]


@open_file(1, mode="wb")
def write_graphml_xml(
    G,
    path,
    encoding="utf-8",
    prettyprint=True,
    infer_numeric_types=False,
    named_key_ids=False,
    edge_id_from_attribute=None,
):
    """Write G in GraphML XML format to path

    Parameters
    ----------
    G : graph
       A networkx graph
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.
    encoding : string (optional)
       Encoding for text data.
    prettyprint : bool (optional)
       If True use line breaks and indenting in output XML.
    infer_numeric_types : boolean
       Determine if numeric types should be generalized.
       For example, if edges have both int and float 'weight' attributes,
       we infer in GraphML that both are floats.
    named_key_ids : bool (optional)
       If True use attr.name as value for key elements' id attribute.
    edge_id_from_attribute : dict key (optional)
        If provided, the graphml edge id is set by looking up the corresponding
        edge data attribute keyed by this parameter. If `None` or the key does not exist in edge data,
        the edge id is set by the edge key if `G` is a MultiGraph, else the edge id is left unset.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_graphml(G, "test.graphml")

    Notes
    -----
    This implementation does not support mixed graphs (directed
    and unidirected edges together) hyperedges, nested graphs, or ports.
    """
    writer = GraphMLWriter(
        encoding=encoding,
        prettyprint=prettyprint,
        infer_numeric_types=infer_numeric_types,
        named_key_ids=named_key_ids,
        edge_id_from_attribute=edge_id_from_attribute,
    )
    writer.add_graph_element(G)
    writer.dump(path)


@open_file(1, mode="wb")
def write_graphml_lxml(
    G,
    path,
    encoding="utf-8",
    prettyprint=True,
    infer_numeric_types=False,
    named_key_ids=False,
    edge_id_from_attribute=None,
):
    """Write G in GraphML XML format to path

    This function uses the LXML framework and should be faster than
    the version using the xml library.

    Parameters
    ----------
    G : graph
       A networkx graph
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.
    encoding : string (optional)
       Encoding for text data.
    prettyprint : bool (optional)
       If True use line breaks and indenting in output XML.
    infer_numeric_types : boolean
       Determine if numeric types should be generalized.
       For example, if edges have both int and float 'weight' attributes,
       we infer in GraphML that both are floats.
    named_key_ids : bool (optional)
       If True use attr.name as value for key elements' id attribute.
    edge_id_from_attribute : dict key (optional)
        If provided, the graphml edge id is set by looking up the corresponding
        edge data attribute keyed by this parameter. If `None` or the key does not exist in edge data,
        the edge id is set by the edge key if `G` is a MultiGraph, else the edge id is left unset.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_graphml_lxml(G, "fourpath.graphml")

    Notes
    -----
    This implementation does not support mixed graphs (directed
    and unidirected edges together) hyperedges, nested graphs, or ports.
    """
    try:
        import lxml.etree as lxmletree
    except ImportError:
        return write_graphml_xml(
            G,
            path,
            encoding,
            prettyprint,
            infer_numeric_types,
            named_key_ids,
            edge_id_from_attribute,
        )

    writer = GraphMLWriterLxml(
        path,
        graph=G,
        encoding=encoding,
        prettyprint=prettyprint,
        infer_numeric_types=infer_numeric_types,
        named_key_ids=named_key_ids,
        edge_id_from_attribute=edge_id_from_attribute,
    )
    writer.dump()


def generate_graphml(
    G,
    encoding="utf-8",
    prettyprint=True,
    named_key_ids=False,
    edge_id_from_attribute=None,
):
    """Generate GraphML lines for G

    Parameters
    ----------
    G : graph
       A networkx graph
    encoding : string (optional)
       Encoding for text data.
    prettyprint : bool (optional)
       If True use line breaks and indenting in output XML.
    named_key_ids : bool (optional)
       If True use attr.name as value for key elements' id attribute.
    edge_id_from_attribute : dict key (optional)
        If provided, the graphml edge id is set by looking up the corresponding
        edge data attribute keyed by this parameter. If `None` or the key does not exist in edge data,
        the edge id is set by the edge key if `G` is a MultiGraph, else the edge id is left unset.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> linefeed = chr(10)  # linefeed = \n
    >>> s = linefeed.join(nx.generate_graphml(G))
    >>> for line in nx.generate_graphml(G):  # doctest: +SKIP
    ...     print(line)

    Notes
    -----
    This implementation does not support mixed graphs (directed and unidirected
    edges together) hyperedges, nested graphs, or ports.
    """
    writer = GraphMLWriter(
        encoding=encoding,
        prettyprint=prettyprint,
        named_key_ids=named_key_ids,
        edge_id_from_attribute=edge_id_from_attribute,
    )
    writer.add_graph_element(G)
    yield from str(writer).splitlines()


@open_file(0, mode="rb")
@nx._dispatchable(graphs=None, returns_graph=True)
def read_graphml(path, node_type=str, edge_key_type=int, force_multigraph=False):
    """Read graph in GraphML format from path.

    Parameters
    ----------
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.

    node_type: Python type (default: str)
       Convert node ids to this type

    edge_key_type: Python type (default: int)
       Convert graphml edge ids to this type. Multigraphs use id as edge key.
       Non-multigraphs add to edge attribute dict with name "id".

    force_multigraph : bool (default: False)
       If True, return a multigraph with edge keys. If False (the default)
       return a multigraph when multiedges are in the graph.

    Returns
    -------
    graph: NetworkX graph
        If parallel edges are present or `force_multigraph=True` then
        a MultiGraph or MultiDiGraph is returned. Otherwise a Graph/DiGraph.
        The returned graph is directed if the file indicates it should be.

    Notes
    -----
    Default node and edge attributes are not propagated to each node and edge.
    They can be obtained from `G.graph` and applied to node and edge attributes
    if desired using something like this:

    >>> default_color = G.graph["node_default"]["color"]  # doctest: +SKIP
    >>> for node, data in G.nodes(data=True):  # doctest: +SKIP
    ...     if "color" not in data:
    ...         data["color"] = default_color
    >>> default_color = G.graph["edge_default"]["color"]  # doctest: +SKIP
    >>> for u, v, data in G.edges(data=True):  # doctest: +SKIP
    ...     if "color" not in data:
    ...         data["color"] = default_color

    This implementation does not support mixed graphs (directed and unidirected
    edges together), hypergraphs, nested graphs, or ports.

    For multigraphs the GraphML edge "id" will be used as the edge
    key.  If not specified then they "key" attribute will be used.  If
    there is no "key" attribute a default NetworkX multigraph edge key
    will be provided.

    Files with the yEd "yfiles" extension can be read. The type of the node's
    shape is preserved in the `shape_type` node attribute.

    yEd compressed files ("file.graphmlz" extension) can be read by renaming
    the file to "file.graphml.gz".

    """
    reader = GraphMLReader(node_type, edge_key_type, force_multigraph)
    # need to check for multiple graphs
    glist = list(reader(path=path))
    if len(glist) == 0:
        # If no graph comes back, try looking for an incomplete header
        header = b'<graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
        path.seek(0)
        old_bytes = path.read()
        new_bytes = old_bytes.replace(b"<graphml>", header)
        glist = list(reader(string=new_bytes))
        if len(glist) == 0:
            raise nx.NetworkXError("file not successfully read as graphml")
    return glist[0]


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_graphml(
    graphml_string, node_type=str, edge_key_type=int, force_multigraph=False
):
    """Read graph in GraphML format from string.

    Parameters
    ----------
    graphml_string : string
       String containing graphml information
       (e.g., contents of a graphml file).

    node_type: Python type (default: str)
       Convert node ids to this type

    edge_key_type: Python type (default: int)
       Convert graphml edge ids to this type. Multigraphs use id as edge key.
       Non-multigraphs add to edge attribute dict with name "id".

    force_multigraph : bool (default: False)
       If True, return a multigraph with edge keys. If False (the default)
       return a multigraph when multiedges are in the graph.


    Returns
    -------
    graph: NetworkX graph
        If no parallel edges are found a Graph or DiGraph is returned.
        Otherwise a MultiGraph or MultiDiGraph is returned.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> linefeed = chr(10)  # linefeed = \n
    >>> s = linefeed.join(nx.generate_graphml(G))
    >>> H = nx.parse_graphml(s)

    Notes
    -----
    Default node and edge attributes are not propagated to each node and edge.
    They can be obtained from `G.graph` and applied to node and edge attributes
    if desired using something like this:

    >>> default_color = G.graph["node_default"]["color"]  # doctest: +SKIP
    >>> for node, data in G.nodes(data=True):  # doctest: +SKIP
    ...     if "color" not in data:
    ...         data["color"] = default_color
    >>> default_color = G.graph["edge_default"]["color"]  # doctest: +SKIP
    >>> for u, v, data in G.edges(data=True):  # doctest: +SKIP
    ...     if "color" not in data:
    ...         data["color"] = default_color

    This implementation does not support mixed graphs (directed and unidirected
    edges together), hypergraphs, nested graphs, or ports.

    For multigraphs the GraphML edge "id" will be used as the edge
    key.  If not specified then they "key" attribute will be used.  If
    there is no "key" attribute a default NetworkX multigraph edge key
    will be provided.

    """
    reader = GraphMLReader(node_type, edge_key_type, force_multigraph)
    # need to check for multiple graphs
    glist = list(reader(string=graphml_string))
    if len(glist) == 0:
        # If no graph comes back, try looking for an incomplete header
        header = '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
        new_string = graphml_string.replace("<graphml>", header)
        glist = list(reader(string=new_string))
        if len(glist) == 0:
            raise nx.NetworkXError("file not successfully read as graphml")
    return glist[0]


class GraphML:
    NS_GRAPHML = "http://graphml.graphdrawing.org/xmlns"
    NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"
    # xmlns:y="http://www.yworks.com/xml/graphml"
    NS_Y = "http://www.yworks.com/xml/graphml"
    SCHEMALOCATION = " ".join(
        [
            "http://graphml.graphdrawing.org/xmlns",
            "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
        ]
    )

    def construct_types(self):
        types = [
            (int, "integer"),  # for Gephi GraphML bug
            (str, "yfiles"),
            (str, "string"),
            (int, "int"),
            (int, "long"),
            (float, "float"),
            (float, "double"),
            (bool, "boolean"),
        ]

        # These additions to types allow writing numpy types
        try:
            import numpy as np
        except:
            pass
        else:
            # prepend so that python types are created upon read (last entry wins)
            types = [
                (np.float64, "float"),
                (np.float32, "float"),
                (np.float16, "float"),
                (np.int_, "int"),
                (np.int8, "int"),
                (np.int16, "int"),
                (np.int32, "int"),
                (np.int64, "int"),
                (np.uint8, "int"),
                (np.uint16, "int"),
                (np.uint32, "int"),
                (np.uint64, "int"),
                (np.int_, "int"),
                (np.intc, "int"),
                (np.intp, "int"),
            ] + types

        self.xml_type = dict(types)
        self.python_type = dict(reversed(a) for a in types)

    # This page says that data types in GraphML follow Java(TM).
    #  http://graphml.graphdrawing.org/primer/graphml-primer.html#AttributesDefinition
    # true and false are the only boolean literals:
    #  http://en.wikibooks.org/wiki/Java_Programming/Literals#Boolean_Literals
    convert_bool = {
        # We use data.lower() in actual use.
        "true": True,
        "false": False,
        # Include integer strings for convenience.
        "0": False,
        0: False,
        "1": True,
        1: True,
    }

    def get_xml_type(self, key):
        """Wrapper around the xml_type dict that raises a more informative
        exception message when a user attempts to use data of a type not
        supported by GraphML."""
        try:
            return self.xml_type[key]
        except KeyError as err:
            raise TypeError(
                f"GraphML does not support type {key} as data values."
            ) from err


class GraphMLWriter(GraphML):
    def __init__(
        self,
        graph=None,
        encoding="utf-8",
        prettyprint=True,
        infer_numeric_types=False,
        named_key_ids=False,
        edge_id_from_attribute=None,
    ):
        self.construct_types()
        from xml.etree.ElementTree import Element

        self.myElement = Element

        self.infer_numeric_types = infer_numeric_types
        self.prettyprint = prettyprint
        self.named_key_ids = named_key_ids
        self.edge_id_from_attribute = edge_id_from_attribute
        self.encoding = encoding
        self.xml = self.myElement(
            "graphml",
            {
                "xmlns": self.NS_GRAPHML,
                "xmlns:xsi": self.NS_XSI,
                "xsi:schemaLocation": self.SCHEMALOCATION,
            },
        )
        self.keys = {}
        self.attributes = defaultdict(list)
        self.attribute_types = defaultdict(set)

        if graph is not None:
            self.add_graph_element(graph)

    def __str__(self):
        from xml.etree.ElementTree import tostring

        if self.prettyprint:
            self.indent(self.xml)
        s = tostring(self.xml).decode(self.encoding)
        return s

    def attr_type(self, name, scope, value):
        """Infer the attribute type of data named name. Currently this only
        supports inference of numeric types.

        If self.infer_numeric_types is false, type is used. Otherwise, pick the
        most general of types found across all values with name and scope. This
        means edges with data named 'weight' are treated separately from nodes
        with data named 'weight'.
        """
        if self.infer_numeric_types:
            types = self.attribute_types[(name, scope)]

            if len(types) > 1:
                types = {self.get_xml_type(t) for t in types}
                if "string" in types:
                    return str
                elif "float" in types or "double" in types:
                    return float
                else:
                    return int
            else:
                return list(types)[0]
        else:
            return type(value)

    def get_key(self, name, attr_type, scope, default):
        keys_key = (name, attr_type, scope)
        try:
            return self.keys[keys_key]
        except KeyError:
            if self.named_key_ids:
                new_id = name
            else:
                new_id = f"d{len(list(self.keys))}"

            self.keys[keys_key] = new_id
            key_kwargs = {
                "id": new_id,
                "for": scope,
                "attr.name": name,
                "attr.type": attr_type,
            }
            key_element = self.myElement("key", **key_kwargs)
            # add subelement for data default value if present
            if default is not None:
                default_element = self.myElement("default")
                default_element.text = str(default)
                key_element.append(default_element)
            self.xml.insert(0, key_element)
        return new_id

    def add_data(self, name, element_type, value, scope="all", default=None):
        """
        Make a data element for an edge or a node. Keep a log of the
        type in the keys table.
        """
        if element_type not in self.xml_type:
            raise nx.NetworkXError(
                f"GraphML writer does not support {element_type} as data values."
            )
        keyid = self.get_key(name, self.get_xml_type(element_type), scope, default)
        data_element = self.myElement("data", key=keyid)
        data_element.text = str(value)
        return data_element

    def add_attributes(self, scope, xml_obj, data, default):
        """Appends attribute data to edges or nodes, and stores type information
        to be added later. See add_graph_element.
        """
        for k, v in data.items():
            self.attribute_types[(str(k), scope)].add(type(v))
            self.attributes[xml_obj].append([k, v, scope, default.get(k)])

    def add_nodes(self, G, graph_element):
        default = G.graph.get("node_default", {})
        for node, data in G.nodes(data=True):
            node_element = self.myElement("node", id=str(node))
            self.add_attributes("node", node_element, data, default)
            graph_element.append(node_element)

    def add_edges(self, G, graph_element):
        if G.is_multigraph():
            for u, v, key, data in G.edges(data=True, keys=True):
                edge_element = self.myElement(
                    "edge",
                    source=str(u),
                    target=str(v),
                    id=str(data.get(self.edge_id_from_attribute))
                    if self.edge_id_from_attribute
                    and self.edge_id_from_attribute in data
                    else str(key),
                )
                default = G.graph.get("edge_default", {})
                self.add_attributes("edge", edge_element, data, default)
                graph_element.append(edge_element)
        else:
            for u, v, data in G.edges(data=True):
                if self.edge_id_from_attribute and self.edge_id_from_attribute in data:
                    # select attribute to be edge id
                    edge_element = self.myElement(
                        "edge",
                        source=str(u),
                        target=str(v),
                        id=str(data.get(self.edge_id_from_attribute)),
                    )
                else:
                    # default: no edge id
                    edge_element = self.myElement("edge", source=str(u), target=str(v))
                default = G.graph.get("edge_default", {})
                self.add_attributes("edge", edge_element, data, default)
                graph_element.append(edge_element)

    def add_graph_element(self, G):
        """
        Serialize graph G in GraphML to the stream.
        """
        if G.is_directed():
            default_edge_type = "directed"
        else:
            default_edge_type = "undirected"

        graphid = G.graph.pop("id", None)
        if graphid is None:
            graph_element = self.myElement("graph", edgedefault=default_edge_type)
        else:
            graph_element = self.myElement(
                "graph", edgedefault=default_edge_type, id=graphid
            )
        default = {}
        data = {
            k: v
            for (k, v) in G.graph.items()
            if k not in ["node_default", "edge_default"]
        }
        self.add_attributes("graph", graph_element, data, default)
        self.add_nodes(G, graph_element)
        self.add_edges(G, graph_element)

        # self.attributes contains a mapping from XML Objects to a list of
        # data that needs to be added to them.
        # We postpone processing in order to do type inference/generalization.
        # See self.attr_type
        for xml_obj, data in self.attributes.items():
            for k, v, scope, default in data:
                xml_obj.append(
                    self.add_data(
                        str(k), self.attr_type(k, scope, v), str(v), scope, default
                    )
                )
        self.xml.append(graph_element)

    def add_graphs(self, graph_list):
        """Add many graphs to this GraphML document."""
        for G in graph_list:
            self.add_graph_element(G)

    def dump(self, stream):
        from xml.etree.ElementTree import ElementTree

        if self.prettyprint:
            self.indent(self.xml)
        document = ElementTree(self.xml)
        document.write(stream, encoding=self.encoding, xml_declaration=True)

    def indent(self, elem, level=0):
        # in-place prettyprint formatter
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


class IncrementalElement:
    """Wrapper for _IncrementalWriter providing an Element like interface.

    This wrapper does not intend to be a complete implementation but rather to
    deal with those calls used in GraphMLWriter.
    """

    def __init__(self, xml, prettyprint):
        self.xml = xml
        self.prettyprint = prettyprint

    def append(self, element):
        self.xml.write(element, pretty_print=self.prettyprint)


class GraphMLWriterLxml(GraphMLWriter):
    def __init__(
        self,
        path,
        graph=None,
        encoding="utf-8",
        prettyprint=True,
        infer_numeric_types=False,
        named_key_ids=False,
        edge_id_from_attribute=None,
    ):
        self.construct_types()
        import lxml.etree as lxmletree

        self.myElement = lxmletree.Element

        self._encoding = encoding
        self._prettyprint = prettyprint
        self.named_key_ids = named_key_ids
        self.edge_id_from_attribute = edge_id_from_attribute
        self.infer_numeric_types = infer_numeric_types

        self._xml_base = lxmletree.xmlfile(path, encoding=encoding)
        self._xml = self._xml_base.__enter__()
        self._xml.write_declaration()

        # We need to have a xml variable that support insertion. This call is
        # used for adding the keys to the document.
        # We will store those keys in a plain list, and then after the graph
        # element is closed we will add them to the main graphml element.
        self.xml = []
        self._keys = self.xml
        self._graphml = self._xml.element(
            "graphml",
            {
                "xmlns": self.NS_GRAPHML,
                "xmlns:xsi": self.NS_XSI,
                "xsi:schemaLocation": self.SCHEMALOCATION,
            },
        )
        self._graphml.__enter__()
        self.keys = {}
        self.attribute_types = defaultdict(set)

        if graph is not None:
            self.add_graph_element(graph)

    def add_graph_element(self, G):
        """
        Serialize graph G in GraphML to the stream.
        """
        if G.is_directed():
            default_edge_type = "directed"
        else:
            default_edge_type = "undirected"

        graphid = G.graph.pop("id", None)
        if graphid is None:
            graph_element = self._xml.element("graph", edgedefault=default_edge_type)
        else:
            graph_element = self._xml.element(
                "graph", edgedefault=default_edge_type, id=graphid
            )

        # gather attributes types for the whole graph
        # to find the most general numeric format needed.
        # Then pass through attributes to create key_id for each.
        graphdata = {
            k: v
            for k, v in G.graph.items()
            if k not in ("node_default", "edge_default")
        }
        node_default = G.graph.get("node_default", {})
        edge_default = G.graph.get("edge_default", {})
        # Graph attributes
        for k, v in graphdata.items():
            self.attribute_types[(str(k), "graph")].add(type(v))
        for k, v in graphdata.items():
            element_type = self.get_xml_type(self.attr_type(k, "graph", v))
            self.get_key(str(k), element_type, "graph", None)
        # Nodes and data
        for node, d in G.nodes(data=True):
            for k, v in d.items():
                self.attribute_types[(str(k), "node")].add(type(v))
        for node, d in G.nodes(data=True):
            for k, v in d.items():
                T = self.get_xml_type(self.attr_type(k, "node", v))
                self.get_key(str(k), T, "node", node_default.get(k))
        # Edges and data
        if G.is_multigraph():
            for u, v, ekey, d in G.edges(keys=True, data=True):
                for k, v in d.items():
                    self.attribute_types[(str(k), "edge")].add(type(v))
            for u, v, ekey, d in G.edges(keys=True, data=True):
                for k, v in d.items():
                    T = self.get_xml_type(self.attr_type(k, "edge", v))
                    self.get_key(str(k), T, "edge", edge_default.get(k))
        else:
            for u, v, d in G.edges(data=True):
                for k, v in d.items():
                    self.attribute_types[(str(k), "edge")].add(type(v))
            for u, v, d in G.edges(data=True):
                for k, v in d.items():
                    T = self.get_xml_type(self.attr_type(k, "edge", v))
                    self.get_key(str(k), T, "edge", edge_default.get(k))

        # Now add attribute keys to the xml file
        for key in self.xml:
            self._xml.write(key, pretty_print=self._prettyprint)

        # The incremental_writer writes each node/edge as it is created
        incremental_writer = IncrementalElement(self._xml, self._prettyprint)
        with graph_element:
            self.add_attributes("graph", incremental_writer, graphdata, {})
            self.add_nodes(G, incremental_writer)  # adds attributes too
            self.add_edges(G, incremental_writer)  # adds attributes too

    def add_attributes(self, scope, xml_obj, data, default):
        """Appends attribute data."""
        for k, v in data.items():
            data_element = self.add_data(
                str(k), self.attr_type(str(k), scope, v), str(v), scope, default.get(k)
            )
            xml_obj.append(data_element)

    def __str__(self):
        return object.__str__(self)

    def dump(self, stream=None):
        self._graphml.__exit__(None, None, None)
        self._xml_base.__exit__(None, None, None)


# default is lxml is present.
write_graphml = write_graphml_lxml


class GraphMLReader(GraphML):
    """Read a GraphML document.  Produces NetworkX graph objects."""

    def __init__(self, node_type=str, edge_key_type=int, force_multigraph=False):
        self.construct_types()
        self.node_type = node_type
        self.edge_key_type = edge_key_type
        self.multigraph = force_multigraph  # If False, test for multiedges
        self.edge_ids = {}  # dict mapping (u,v) tuples to edge id attributes

    def __call__(self, path=None, string=None):
        from xml.etree.ElementTree import ElementTree, fromstring

        if path is not None:
            self.xml = ElementTree(file=path)
        elif string is not None:
            self.xml = fromstring(string)
        else:
            raise ValueError("Must specify either 'path' or 'string' as kwarg")
        (keys, defaults) = self.find_graphml_keys(self.xml)
        for g in self.xml.findall(f"{{{self.NS_GRAPHML}}}graph"):
            yield self.make_graph(g, keys, defaults)

    def make_graph(self, graph_xml, graphml_keys, defaults, G=None):
        # set default graph type
        edgedefault = graph_xml.get("edgedefault", None)
        if G is None:
            if edgedefault == "directed":
                G = nx.MultiDiGraph()
            else:
                G = nx.MultiGraph()
        # set defaults for graph attributes
        G.graph["node_default"] = {}
        G.graph["edge_default"] = {}
        for key_id, value in defaults.items():
            key_for = graphml_keys[key_id]["for"]
            name = graphml_keys[key_id]["name"]
            python_type = graphml_keys[key_id]["type"]
            if key_for == "node":
                G.graph["node_default"].update({name: python_type(value)})
            if key_for == "edge":
                G.graph["edge_default"].update({name: python_type(value)})
        # hyperedges are not supported
        hyperedge = graph_xml.find(f"{{{self.NS_GRAPHML}}}hyperedge")
        if hyperedge is not None:
            raise nx.NetworkXError("GraphML reader doesn't support hyperedges")
        # add nodes
        for node_xml in graph_xml.findall(f"{{{self.NS_GRAPHML}}}node"):
            self.add_node(G, node_xml, graphml_keys, defaults)
        # add edges
        for edge_xml in graph_xml.findall(f"{{{self.NS_GRAPHML}}}edge"):
            self.add_edge(G, edge_xml, graphml_keys)
        # add graph data
        data = self.decode_data_elements(graphml_keys, graph_xml)
        G.graph.update(data)

        # switch to Graph or DiGraph if no parallel edges were found
        if self.multigraph:
            return G

        G = nx.DiGraph(G) if G.is_directed() else nx.Graph(G)
        # add explicit edge "id" from file as attribute in NX graph.
        nx.set_edge_attributes(G, values=self.edge_ids, name="id")
        return G

    def add_node(self, G, node_xml, graphml_keys, defaults):
        """Add a node to the graph."""
        # warn on finding unsupported ports tag
        ports = node_xml.find(f"{{{self.NS_GRAPHML}}}port")
        if ports is not None:
            warnings.warn("GraphML port tag not supported.")
        # find the node by id and cast it to the appropriate type
        node_id = self.node_type(node_xml.get("id"))
        # get data/attributes for node
        data = self.decode_data_elements(graphml_keys, node_xml)
        G.add_node(node_id, **data)
        # get child nodes
        if node_xml.attrib.get("yfiles.foldertype") == "group":
            graph_xml = node_xml.find(f"{{{self.NS_GRAPHML}}}graph")
            self.make_graph(graph_xml, graphml_keys, defaults, G)

    def add_edge(self, G, edge_element, graphml_keys):
        """Add an edge to the graph."""
        # warn on finding unsupported ports tag
        ports = edge_element.find(f"{{{self.NS_GRAPHML}}}port")
        if ports is not None:
            warnings.warn("GraphML port tag not supported.")

        # raise error if we find mixed directed and undirected edges
        directed = edge_element.get("directed")
        if G.is_directed() and directed == "false":
            msg = "directed=false edge found in directed graph."
            raise nx.NetworkXError(msg)
        if (not G.is_directed()) and directed == "true":
            msg = "directed=true edge found in undirected graph."
            raise nx.NetworkXError(msg)

        source = self.node_type(edge_element.get("source"))
        target = self.node_type(edge_element.get("target"))
        data = self.decode_data_elements(graphml_keys, edge_element)
        # GraphML stores edge ids as an attribute
        # NetworkX uses them as keys in multigraphs too if no key
        # attribute is specified
        edge_id = edge_element.get("id")
        if edge_id:
            # self.edge_ids is used by `make_graph` method for non-multigraphs
            self.edge_ids[source, target] = edge_id
            try:
                edge_id = self.edge_key_type(edge_id)
            except ValueError:  # Could not convert.
                pass
        else:
            edge_id = data.get("key")

        if G.has_edge(source, target):
            # mark this as a multigraph
            self.multigraph = True

        # Use add_edges_from to avoid error with add_edge when `'key' in data`
        # Note there is only one edge here...
        G.add_edges_from([(source, target, edge_id, data)])

    def decode_data_elements(self, graphml_keys, obj_xml):
        """Use the key information to decode the data XML if present."""
        data = {}
        for data_element in obj_xml.findall(f"{{{self.NS_GRAPHML}}}data"):
            key = data_element.get("key")
            try:
                data_name = graphml_keys[key]["name"]
                data_type = graphml_keys[key]["type"]
            except KeyError as err:
                raise nx.NetworkXError(f"Bad GraphML data: no key {key}") from err
            text = data_element.text
            # assume anything with subelements is a yfiles extension
            if text is not None and len(list(data_element)) == 0:
                if data_type == bool:
                    # Ignore cases.
                    # http://docs.oracle.com/javase/6/docs/api/java/lang/
                    # Boolean.html#parseBoolean%28java.lang.String%29
                    data[data_name] = self.convert_bool[text.lower()]
                else:
                    data[data_name] = data_type(text)
            elif len(list(data_element)) > 0:
                # Assume yfiles as subelements, try to extract node_label
                node_label = None
                # set GenericNode's configuration as shape type
                gn = data_element.find(f"{{{self.NS_Y}}}GenericNode")
                if gn is not None:
                    data["shape_type"] = gn.get("configuration")
                for node_type in ["GenericNode", "ShapeNode", "SVGNode", "ImageNode"]:
                    pref = f"{{{self.NS_Y}}}{node_type}/{{{self.NS_Y}}}"
                    geometry = data_element.find(f"{pref}Geometry")
                    if geometry is not None:
                        data["x"] = geometry.get("x")
                        data["y"] = geometry.get("y")
                    if node_label is None:
                        node_label = data_element.find(f"{pref}NodeLabel")
                    shape = data_element.find(f"{pref}Shape")
                    if shape is not None:
                        data["shape_type"] = shape.get("type")
                if node_label is not None:
                    data["label"] = node_label.text

                # check all the different types of edges available in yEd.
                for edge_type in [
                    "PolyLineEdge",
                    "SplineEdge",
                    "QuadCurveEdge",
                    "BezierEdge",
                    "ArcEdge",
                ]:
                    pref = f"{{{self.NS_Y}}}{edge_type}/{{{self.NS_Y}}}"
                    edge_label = data_element.find(f"{pref}EdgeLabel")
                    if edge_label is not None:
                        break
                if edge_label is not None:
                    data["label"] = edge_label.text
            elif text is None:
                data[data_name] = ""
        return data

    def find_graphml_keys(self, graph_element):
        """Extracts all the keys and key defaults from the xml."""
        graphml_keys = {}
        graphml_key_defaults = {}
        for k in graph_element.findall(f"{{{self.NS_GRAPHML}}}key"):
            attr_id = k.get("id")
            attr_type = k.get("attr.type")
            attr_name = k.get("attr.name")
            yfiles_type = k.get("yfiles.type")
            if yfiles_type is not None:
                attr_name = yfiles_type
                attr_type = "yfiles"
            if attr_type is None:
                attr_type = "string"
                warnings.warn(f"No key type for id {attr_id}. Using string")
            if attr_name is None:
                raise nx.NetworkXError(f"Unknown key for id {attr_id}.")
            graphml_keys[attr_id] = {
                "name": attr_name,
                "type": self.python_type[attr_type],
                "for": k.get("for"),
            }
            # check for "default" sub-element of key element
            default = k.find(f"{{{self.NS_GRAPHML}}}default")
            if default is not None:
                # Handle default values identically to data element values
                python_type = graphml_keys[attr_id]["type"]
                if python_type == bool:
                    graphml_key_defaults[attr_id] = self.convert_bool[
                        default.text.lower()
                    ]
                else:
                    graphml_key_defaults[attr_id] = python_type(default.text)
        return graphml_keys, graphml_key_defaults
<|endoftext|># Original author: D. Eppstein, UC Irvine, August 12, 2003.
# The original code at https://www.ics.uci.edu/~eppstein/PADS/ is public domain.
"""Functions for reading and writing graphs in the *sparse6* format.

The *sparse6* file format is a space-efficient format for large sparse
graphs. For small graphs or large dense graphs, use the *graph6* file
format.

For more information, see the `sparse6`_ homepage.

.. _sparse6: https://users.cecs.anu.edu.au/~bdm/data/formats.html

"""

import networkx as nx
from networkx.exception import NetworkXError
from networkx.readwrite.graph6 import data_to_n, n_to_data
from networkx.utils import not_implemented_for, open_file

__all__ = ["from_sparse6_bytes", "read_sparse6", "to_sparse6_bytes", "write_sparse6"]


def _generate_sparse6_bytes(G, nodes, header):
    """Yield bytes in the sparse6 encoding of a graph.

    `G` is an undirected simple graph. `nodes` is the list of nodes for
    which the node-induced subgraph will be encoded; if `nodes` is the
    list of all nodes in the graph, the entire graph will be
    encoded. `header` is a Boolean that specifies whether to generate
    the header ``b'>>sparse6<<'`` before the remaining data.

    This function generates `bytes` objects in the following order:

    1. the header (if requested),
    2. the encoding of the number of nodes,
    3. each character, one-at-a-time, in the encoding of the requested
       node-induced subgraph,
    4. a newline character.

    This function raises :exc:`ValueError` if the graph is too large for
    the graph6 format (that is, greater than ``2 ** 36`` nodes).

    """
    n = len(G)
    if n >= 2**36:
        raise ValueError(
            "sparse6 is only defined if number of nodes is less than 2 ** 36"
        )
    if header:
        yield b">>sparse6<<"
    yield b":"
    for d in n_to_data(n):
        yield str.encode(chr(d + 63))

    k = 1
    while 1 << k < n:
        k += 1

    def enc(x):
        """Big endian k-bit encoding of x"""
        return [1 if (x & 1 << (k - 1 - i)) else 0 for i in range(k)]

    edges = sorted((max(u, v), min(u, v)) for u, v in G.edges())
    bits = []
    curv = 0
    for v, u in edges:
        if v == curv:  # current vertex edge
            bits.append(0)
            bits.extend(enc(u))
        elif v == curv + 1:  # next vertex edge
            curv += 1
            bits.append(1)
            bits.extend(enc(u))
        else:  # skip to vertex v and then add edge to u
            curv = v
            bits.append(1)
            bits.extend(enc(v))
            bits.append(0)
            bits.extend(enc(u))
    if k < 6 and n == (1 << k) and ((-len(bits)) % 6) >= k and curv < (n - 1):
        # Padding special case: small k, n=2^k,
        # more than k bits of padding needed,
        # current vertex is not (n-1) --
        # appending 1111... would add a loop on (n-1)
        bits.append(0)
        bits.extend([1] * ((-len(bits)) % 6))
    else:
        bits.extend([1] * ((-len(bits)) % 6))

    data = [
        (bits[i + 0] << 5)
        + (bits[i + 1] << 4)
        + (bits[i + 2] << 3)
        + (bits[i + 3] << 2)
        + (bits[i + 4] << 1)
        + (bits[i + 5] << 0)
        for i in range(0, len(bits), 6)
    ]

    for d in data:
        yield str.encode(chr(d + 63))
    yield b"\n"


@nx._dispatchable(graphs=None, returns_graph=True)
def from_sparse6_bytes(string):
    """Read an undirected graph in sparse6 format from string.

    Parameters
    ----------
    string : string
       Data in sparse6 format

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If the string is unable to be parsed in sparse6 format

    Examples
    --------
    >>> G = nx.from_sparse6_bytes(b":A_")
    >>> sorted(G.edges())
    [(0, 1), (0, 1), (0, 1)]

    See Also
    --------
    read_sparse6, write_sparse6

    References
    ----------
    .. [1] Sparse6 specification
           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    if string.startswith(b">>sparse6<<"):
        string = string[11:]
    if not string.startswith(b":"):
        raise NetworkXError("Expected leading colon in sparse6")

    chars = [c - 63 for c in string[1:]]
    n, data = data_to_n(chars)
    k = 1
    while 1 << k < n:
        k += 1

    def parseData():
        """Returns stream of pairs b[i], x[i] for sparse6 format."""
        chunks = iter(data)
        d = None  # partial data word
        dLen = 0  # how many unparsed bits are left in d

        while 1:
            if dLen < 1:
                try:
                    d = next(chunks)
                except StopIteration:
                    return
                dLen = 6
            dLen -= 1
            b = (d >> dLen) & 1  # grab top remaining bit

            x = d & ((1 << dLen) - 1)  # partially built up value of x
            xLen = dLen  # how many bits included so far in x
            while xLen < k:  # now grab full chunks until we have enough
                try:
                    d = next(chunks)
                except StopIteration:
                    return
                dLen = 6
                x = (x << 6) + d
                xLen += 6
            x = x >> (xLen - k)  # shift back the extra bits
            dLen = xLen - k
            yield b, x

    v = 0

    G = nx.MultiGraph()
    G.add_nodes_from(range(n))

    multigraph = False
    for b, x in parseData():
        if b == 1:
            v += 1
        # padding with ones can cause overlarge number here
        if x >= n or v >= n:
            break
        elif x > v:
            v = x
        else:
            if G.has_edge(x, v):
                multigraph = True
            G.add_edge(x, v)
    if not multigraph:
        G = nx.Graph(G)
    return G


def to_sparse6_bytes(G, nodes=None, header=True):
    """Convert an undirected graph to bytes in sparse6 format.

    Parameters
    ----------
    G : Graph (undirected)

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by ``G.nodes()`` is used.

    header: bool
       If True add '>>sparse6<<' bytes to head of data.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed.

    ValueError
        If the graph has at least ``2 ** 36`` nodes; the sparse6 format
        is only defined for graphs of order less than ``2 ** 36``.

    Examples
    --------
    >>> nx.to_sparse6_bytes(nx.path_graph(2))
    b'>>sparse6<<:An\\n'

    See Also
    --------
    to_sparse6_bytes, read_sparse6, write_sparse6_bytes

    Notes
    -----
    The returned bytes end with a newline character.

    The format does not support edge or node labels.

    References
    ----------
    .. [1] Graph6 specification
           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    if nodes is not None:
        G = G.subgraph(nodes)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return b"".join(_generate_sparse6_bytes(G, nodes, header))


@open_file(0, mode="rb")
@nx._dispatchable(graphs=None, returns_graph=True)
def read_sparse6(path):
    """Read an undirected graph in sparse6 format from path.

    Parameters
    ----------
    path : file or string
       File or filename to write.

    Returns
    -------
    G : Graph/Multigraph or list of Graphs/MultiGraphs
       If the file contains multiple lines then a list of graphs is returned

    Raises
    ------
    NetworkXError
        If the string is unable to be parsed in sparse6 format

    Examples
    --------
    You can read a sparse6 file by giving the path to the file::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False) as f:
        ...     _ = f.write(b">>sparse6<<:An\\n")
        ...     _ = f.seek(0)
        ...     G = nx.read_sparse6(f.name)
        >>> list(G.edges())
        [(0, 1)]

    You can also read a sparse6 file by giving an open file-like object::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     _ = f.write(b">>sparse6<<:An\\n")
        ...     _ = f.seek(0)
        ...     G = nx.read_sparse6(f)
        >>> list(G.edges())
        [(0, 1)]

    See Also
    --------
    read_sparse6, from_sparse6_bytes

    References
    ----------
    .. [1] Sparse6 specification
           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    glist = []
    for line in path:
        line = line.strip()
        if not len(line):
            continue
        glist.append(from_sparse6_bytes(line))
    if len(glist) == 1:
        return glist[0]
    else:
        return glist


@not_implemented_for("directed")
@open_file(1, mode="wb")
def write_sparse6(G, path, nodes=None, header=True):
    """Write graph G to given path in sparse6 format.

    Parameters
    ----------
    G : Graph (undirected)

    path : file or string
       File or filename to write

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by G.nodes() is used.

    header: bool
       If True add '>>sparse6<<' string to head of data

    Raises
    ------
    NetworkXError
        If the graph is directed

    Examples
    --------
    You can write a sparse6 file by giving the path to the file::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False) as f:
        ...     nx.write_sparse6(nx.path_graph(2), f.name)
        ...     print(f.read())
        b'>>sparse6<<:An\\n'

    You can also write a sparse6 file by giving an open file-like object::

        >>> with tempfile.NamedTemporaryFile() as f:
        ...     nx.write_sparse6(nx.path_graph(2), f)
        ...     _ = f.seek(0)
        ...     print(f.read())
        b'>>sparse6<<:An\\n'

    See Also
    --------
    read_sparse6, from_sparse6_bytes

    Notes
    -----
    The format does not support edge or node labels.

    References
    ----------
    .. [1] Sparse6 specification
           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    if nodes is not None:
        G = G.subgraph(nodes)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    for b in _generate_sparse6_bytes(G, nodes, header):
        path.write(b)
<|endoftext|>"""
**************
Adjacency List
**************
Read and write NetworkX graphs as adjacency lists.

Adjacency list format is useful for graphs without data associated
with nodes or edges and for nodes that can be meaningfully represented
as strings.

Format
------
The adjacency list format consists of lines with node labels.  The
first label in a line is the source node.  Further labels in the line
are considered target nodes and are added to the graph along with an edge
between the source node and target node.

The graph with edges a-b, a-c, d-e can be represented as the following
adjacency list (anything following the # in a line is a comment)::

     a b c # source target target
     d e
"""

__all__ = ["generate_adjlist", "write_adjlist", "parse_adjlist", "read_adjlist"]

import networkx as nx
from networkx.utils import open_file


def generate_adjlist(G, delimiter=" "):
    """Generate a single line of the graph G in adjacency list format.

    Parameters
    ----------
    G : NetworkX graph

    delimiter : string, optional
       Separator for node labels

    Returns
    -------
    lines : string
        Lines of data in adjlist format.

    Examples
    --------
    >>> G = nx.lollipop_graph(4, 3)
    >>> for line in nx.generate_adjlist(G):
    ...     print(line)
    0 1 2 3
    1 2 3
    2 3
    3 4
    4 5
    5 6
    6

    See Also
    --------
    write_adjlist, read_adjlist

    Notes
    -----
    The default `delimiter=" "` will result in unexpected results if node names contain
    whitespace characters. To avoid this problem, specify an alternate delimiter when spaces are
    valid in node names.

    NB: This option is not available for data that isn't user-generated.

    """
    directed = G.is_directed()
    seen = set()
    for s, nbrs in G.adjacency():
        line = str(s) + delimiter
        for t, data in nbrs.items():
            if not directed and t in seen:
                continue
            if G.is_multigraph():
                for d in data.values():
                    line += str(t) + delimiter
            else:
                line += str(t) + delimiter
        if not directed:
            seen.add(s)
        yield line[: -len(delimiter)]


@open_file(1, mode="wb")
def write_adjlist(G, path, comments="#", delimiter=" ", encoding="utf-8"):
    """Write graph G in single-line adjacency-list format to path.


    Parameters
    ----------
    G : NetworkX graph

    path : string or file
       Filename or file handle for data output.
       Filenames ending in .gz or .bz2 will be compressed.

    comments : string, optional
       Marker for comment lines

    delimiter : string, optional
       Separator for node labels

    encoding : string, optional
       Text encoding.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_adjlist(G, "test.adjlist")

    The path can be a filehandle or a string with the name of the file. If a
    filehandle is provided, it has to be opened in 'wb' mode.

    >>> fh = open("test.adjlist", "wb")
    >>> nx.write_adjlist(G, fh)

    Notes
    -----
    The default `delimiter=" "` will result in unexpected results if node names contain
    whitespace characters. To avoid this problem, specify an alternate delimiter when spaces are
    valid in node names.
    NB: This option is not available for data that isn't user-generated.

    This format does not store graph, node, or edge data.

    See Also
    --------
    read_adjlist, generate_adjlist
    """
    import sys
    import time

    pargs = comments + " ".join(sys.argv) + "\n"
    header = (
        pargs
        + comments
        + f" GMT {time.asctime(time.gmtime())}\n"
        + comments
        + f" {G.name}\n"
    )
    path.write(header.encode(encoding))

    for line in generate_adjlist(G, delimiter):
        line += "\n"
        path.write(line.encode(encoding))


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_adjlist(
    lines, comments="#", delimiter=None, create_using=None, nodetype=None
):
    """Parse lines of a graph adjacency list representation.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in adjlist format

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    nodetype : Python type, optional
       Convert nodes to this type.

    comments : string, optional
       Marker for comment lines

    delimiter : string, optional
       Separator for node labels.  The default is whitespace.

    Returns
    -------
    G: NetworkX graph
        The graph corresponding to the lines in adjacency list format.

    Examples
    --------
    >>> lines = ["1 2 5", "2 3 4", "3 5", "4", "5"]
    >>> G = nx.parse_adjlist(lines, nodetype=int)
    >>> nodes = [1, 2, 3, 4, 5]
    >>> all(node in G for node in nodes)
    True
    >>> edges = [(1, 2), (1, 5), (2, 3), (2, 4), (3, 5)]
    >>> all((u, v) in G.edges() or (v, u) in G.edges() for (u, v) in edges)
    True

    See Also
    --------
    read_adjlist

    """
    G = nx.empty_graph(0, create_using)
    for line in lines:
        p = line.find(comments)
        if p >= 0:
            line = line[:p]
        if not len(line):
            continue
        vlist = line.rstrip("\n").split(delimiter)
        u = vlist.pop(0)
        # convert types
        if nodetype is not None:
            try:
                u = nodetype(u)
            except BaseException as err:
                raise TypeError(
                    f"Failed to convert node ({u}) to type {nodetype}"
                ) from err
        G.add_node(u)
        if nodetype is not None:
            try:
                vlist = list(map(nodetype, vlist))
            except BaseException as err:
                raise TypeError(
                    f"Failed to convert nodes ({','.join(vlist)}) to type {nodetype}"
                ) from err
        G.add_edges_from([(u, v) for v in vlist])
    return G


@open_file(0, mode="rb")
@nx._dispatchable(graphs=None, returns_graph=True)
def read_adjlist(
    path,
    comments="#",
    delimiter=None,
    create_using=None,
    nodetype=None,
    encoding="utf-8",
):
    """Read graph in adjacency list format from path.

    Parameters
    ----------
    path : string or file
       Filename or file handle to read.
       Filenames ending in .gz or .bz2 will be uncompressed.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    nodetype : Python type, optional
       Convert nodes to this type.

    comments : string, optional
       Marker for comment lines

    delimiter : string, optional
       Separator for node labels.  The default is whitespace.

    Returns
    -------
    G: NetworkX graph
        The graph corresponding to the lines in adjacency list format.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_adjlist(G, "test.adjlist")
    >>> G = nx.read_adjlist("test.adjlist")

    The path can be a filehandle or a string with the name of the file. If a
    filehandle is provided, it has to be opened in 'rb' mode.

    >>> fh = open("test.adjlist", "rb")
    >>> G = nx.read_adjlist(fh)

    Filenames ending in .gz or .bz2 will be compressed.

    >>> nx.write_adjlist(G, "test.adjlist.gz")
    >>> G = nx.read_adjlist("test.adjlist.gz")

    The optional nodetype is a function to convert node strings to nodetype.

    For example

    >>> G = nx.read_adjlist("test.adjlist", nodetype=int)

    will attempt to convert all nodes to integer type.

    Since nodes must be hashable, the function nodetype must return hashable
    types (e.g. int, float, str, frozenset - or tuples of those, etc.)

    The optional create_using parameter indicates the type of NetworkX graph
    created.  The default is `nx.Graph`, an undirected graph.
    To read the data as a directed graph use

    >>> G = nx.read_adjlist("test.adjlist", create_using=nx.DiGraph)

    Notes
    -----
    This format does not store graph or node data.

    See Also
    --------
    write_adjlist
    """
    lines = (line.decode(encoding) for line in path)
    return parse_adjlist(
        lines,
        comments=comments,
        delimiter=delimiter,
        create_using=create_using,
        nodetype=nodetype,
    )
<|endoftext|>"""Read and write graphs in GEXF format.

.. warning::
    This parser uses the standard xml library present in Python, which is
    insecure - see :external+python:mod:`xml` for additional information.
    Only parse GEFX files you trust.

GEXF (Graph Exchange XML Format) is a language for describing complex
network structures, their associated data and dynamics.

This implementation does not support mixed graphs (directed and
undirected edges together).

Format
------
GEXF is an XML format.  See http://gexf.net/schema.html for the
specification and http://gexf.net/basic.html for examples.
"""

import itertools
import time
from xml.etree.ElementTree import (
    Element,
    ElementTree,
    SubElement,
    register_namespace,
    tostring,
)

import networkx as nx
from networkx.utils import open_file

__all__ = ["write_gexf", "read_gexf", "relabel_gexf_graph", "generate_gexf"]


@open_file(1, mode="wb")
def write_gexf(G, path, encoding="utf-8", prettyprint=True, version="1.2draft"):
    """Write G in GEXF format to path.

    "GEXF (Graph Exchange XML Format) is a language for describing
    complex networks structures, their associated data and dynamics" [1]_.

    Node attributes are checked according to the version of the GEXF
    schemas used for parameters which are not user defined,
    e.g. visualization 'viz' [2]_. See example for usage.

    Parameters
    ----------
    G : graph
       A NetworkX graph
    path : file or string
       File or file name to write.
       File names ending in .gz or .bz2 will be compressed.
    encoding : string (optional, default: 'utf-8')
       Encoding for text data.
    prettyprint : bool (optional, default: True)
       If True use line breaks and indenting in output XML.
    version: string (optional, default: '1.2draft')
       The version of GEXF to be used for nodes attributes checking

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_gexf(G, "test.gexf")

    # visualization data
    >>> G.nodes[0]["viz"] = {"size": 54}
    >>> G.nodes[0]["viz"]["position"] = {"x": 0, "y": 1}
    >>> G.nodes[0]["viz"]["color"] = {"r": 0, "g": 0, "b": 256}


    Notes
    -----
    This implementation does not support mixed graphs (directed and undirected
    edges together).

    The node id attribute is set to be the string of the node label.
    If you want to specify an id use set it as node data, e.g.
    node['a']['id']=1 to set the id of node 'a' to 1.

    References
    ----------
    .. [1] GEXF File Format, http://gexf.net/
    .. [2] GEXF schema, http://gexf.net/schema.html
    """
    writer = GEXFWriter(encoding=encoding, prettyprint=prettyprint, version=version)
    writer.add_graph(G)
    writer.write(path)


def generate_gexf(G, encoding="utf-8", prettyprint=True, version="1.2draft"):
    """Generate lines of GEXF format representation of G.

    "GEXF (Graph Exchange XML Format) is a language for describing
    complex networks structures, their associated data and dynamics" [1]_.

    Parameters
    ----------
    G : graph
    A NetworkX graph
    encoding : string (optional, default: 'utf-8')
    Encoding for text data.
    prettyprint : bool (optional, default: True)
    If True use line breaks and indenting in output XML.
    version : string (default: 1.2draft)
    Version of GEFX File Format (see http://gexf.net/schema.html)
    Supported values: "1.1draft", "1.2draft"


    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> linefeed = chr(10)  # linefeed=\n
    >>> s = linefeed.join(nx.generate_gexf(G))
    >>> for line in nx.generate_gexf(G):  # doctest: +SKIP
    ...     print(line)

    Notes
    -----
    This implementation does not support mixed graphs (directed and undirected
    edges together).

    The node id attribute is set to be the string of the node label.
    If you want to specify an id use set it as node data, e.g.
    node['a']['id']=1 to set the id of node 'a' to 1.

    References
    ----------
    .. [1] GEXF File Format, https://gephi.org/gexf/format/
    """
    writer = GEXFWriter(encoding=encoding, prettyprint=prettyprint, version=version)
    writer.add_graph(G)
    yield from str(writer).splitlines()


@open_file(0, mode="rb")
@nx._dispatchable(graphs=None, returns_graph=True)
def read_gexf(path, node_type=None, relabel=False, version="1.2draft"):
    """Read graph in GEXF format from path.

    "GEXF (Graph Exchange XML Format) is a language for describing
    complex networks structures, their associated data and dynamics" [1]_.

    Parameters
    ----------
    path : file or string
       File or file name to read.
       File names ending in .gz or .bz2 will be decompressed.
    node_type: Python type (default: None)
       Convert node ids to this type if not None.
    relabel : bool (default: False)
       If True relabel the nodes to use the GEXF node "label" attribute
       instead of the node "id" attribute as the NetworkX node label.
    version : string (default: 1.2draft)
    Version of GEFX File Format (see http://gexf.net/schema.html)
       Supported values: "1.1draft", "1.2draft"

    Returns
    -------
    graph: NetworkX graph
        If no parallel edges are found a Graph or DiGraph is returned.
        Otherwise a MultiGraph or MultiDiGraph is returned.

    Notes
    -----
    This implementation does not support mixed graphs (directed and undirected
    edges together).

    References
    ----------
    .. [1] GEXF File Format, http://gexf.net/
    """
    reader = GEXFReader(node_type=node_type, version=version)
    if relabel:
        G = relabel_gexf_graph(reader(path))
    else:
        G = reader(path)
    return G


class GEXF:
    versions = {
        "1.1draft": {
            "NS_GEXF": "http://www.gexf.net/1.1draft",
            "NS_VIZ": "http://www.gexf.net/1.1draft/viz",
            "NS_XSI": "http://www.w3.org/2001/XMLSchema-instance",
            "SCHEMALOCATION": " ".join(
                [
                    "http://www.gexf.net/1.1draft",
                    "http://www.gexf.net/1.1draft/gexf.xsd",
                ]
            ),
            "VERSION": "1.1",
        },
        "1.2draft": {
            "NS_GEXF": "http://www.gexf.net/1.2draft",
            "NS_VIZ": "http://www.gexf.net/1.2draft/viz",
            "NS_XSI": "http://www.w3.org/2001/XMLSchema-instance",
            "SCHEMALOCATION": " ".join(
                [
                    "http://www.gexf.net/1.2draft",
                    "http://www.gexf.net/1.2draft/gexf.xsd",
                ]
            ),
            "VERSION": "1.2",
        },
    }

    def construct_types(self):
        types = [
            (int, "integer"),
            (float, "float"),
            (float, "double"),
            (bool, "boolean"),
            (list, "string"),
            (dict, "string"),
            (int, "long"),
            (str, "liststring"),
            (str, "anyURI"),
            (str, "string"),
        ]

        # These additions to types allow writing numpy types
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            # prepend so that python types are created upon read (last entry wins)
            types = [
                (np.float64, "float"),
                (np.float32, "float"),
                (np.float16, "float"),
                (np.int_, "int"),
                (np.int8, "int"),
                (np.int16, "int"),
                (np.int32, "int"),
                (np.int64, "int"),
                (np.uint8, "int"),
                (np.uint16, "int"),
                (np.uint32, "int"),
                (np.uint64, "int"),
                (np.int_, "int"),
                (np.intc, "int"),
                (np.intp, "int"),
            ] + types

        self.xml_type = dict(types)
        self.python_type = dict(reversed(a) for a in types)

    # http://www.w3.org/TR/xmlschema-2/#boolean
    convert_bool = {
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "0": False,
        0: False,
        "1": True,
        1: True,
    }

    def set_version(self, version):
        d = self.versions.get(version)
        if d is None:
            raise nx.NetworkXError(f"Unknown GEXF version {version}.")
        self.NS_GEXF = d["NS_GEXF"]
        self.NS_VIZ = d["NS_VIZ"]
        self.NS_XSI = d["NS_XSI"]
        self.SCHEMALOCATION = d["SCHEMALOCATION"]
        self.VERSION = d["VERSION"]
        self.version = version


class GEXFWriter(GEXF):
    # class for writing GEXF format files
    # use write_gexf() function
    def __init__(
        self, graph=None, encoding="utf-8", prettyprint=True, version="1.2draft"
    ):
        self.construct_types()
        self.prettyprint = prettyprint
        self.encoding = encoding
        self.set_version(version)
        self.xml = Element(
            "gexf",
            {
                "xmlns": self.NS_GEXF,
                "xmlns:xsi": self.NS_XSI,
                "xsi:schemaLocation": self.SCHEMALOCATION,
                "version": self.VERSION,
            },
        )

        # Make meta element a non-graph element
        # Also add lastmodifieddate as attribute, not tag
        meta_element = Element("meta")
        subelement_text = f"NetworkX {nx.__version__}"
        SubElement(meta_element, "creator").text = subelement_text
        meta_element.set("lastmodifieddate", time.strftime("%Y-%m-%d"))
        self.xml.append(meta_element)

        register_namespace("viz", self.NS_VIZ)

        # counters for edge and attribute identifiers
        self.edge_id = itertools.count()
        self.attr_id = itertools.count()
        self.all_edge_ids = set()
        # default attributes are stored in dictionaries
        self.attr = {}
        self.attr["node"] = {}
        self.attr["edge"] = {}
        self.attr["node"]["dynamic"] = {}
        self.attr["node"]["static"] = {}
        self.attr["edge"]["dynamic"] = {}
        self.attr["edge"]["static"] = {}

        if graph is not None:
            self.add_graph(graph)

    def __str__(self):
        if self.prettyprint:
            self.indent(self.xml)
        s = tostring(self.xml).decode(self.encoding)
        return s

    def add_graph(self, G):
        # first pass through G collecting edge ids
        for u, v, dd in G.edges(data=True):
            eid = dd.get("id")
            if eid is not None:
                self.all_edge_ids.add(str(eid))
        # set graph attributes
        if G.graph.get("mode") == "dynamic":
            mode = "dynamic"
        else:
            mode = "static"
        # Add a graph element to the XML
        if G.is_directed():
            default = "directed"
        else:
            default = "undirected"
        name = G.graph.get("name", "")
        graph_element = Element("graph", defaultedgetype=default, mode=mode, name=name)
        self.graph_element = graph_element
        self.add_nodes(G, graph_element)
        self.add_edges(G, graph_element)
        self.xml.append(graph_element)

    def add_nodes(self, G, graph_element):
        nodes_element = Element("nodes")
        for node, data in G.nodes(data=True):
            node_data = data.copy()
            node_id = str(node_data.pop("id", node))
            kw = {"id": node_id}
            label = str(node_data.pop("label", node))
            kw["label"] = label
            try:
                pid = node_data.pop("pid")
                kw["pid"] = str(pid)
            except KeyError:
                pass
            try:
                start = node_data.pop("start")
                kw["start"] = str(start)
                self.alter_graph_mode_timeformat(start)
            except KeyError:
                pass
            try:
                end = node_data.pop("end")
                kw["end"] = str(end)
                self.alter_graph_mode_timeformat(end)
            except KeyError:
                pass
            # add node element with attributes
            node_element = Element("node", **kw)
            # add node element and attr subelements
            default = G.graph.get("node_default", {})
            node_data = self.add_parents(node_element, node_data)
            if self.VERSION == "1.1":
                node_data = self.add_slices(node_element, node_data)
            else:
                node_data = self.add_spells(node_element, node_data)
            node_data = self.add_viz(node_element, node_data)
            node_data = self.add_attributes("node", node_element, node_data, default)
            nodes_element.append(node_element)
        graph_element.append(nodes_element)

    def add_edges(self, G, graph_element):
        def edge_key_data(G):
            # helper function to unify multigraph and graph edge iterator
            if G.is_multigraph():
                for u, v, key, data in G.edges(data=True, keys=True):
                    edge_data = data.copy()
                    edge_data.update(key=key)
                    edge_id = edge_data.pop("id", None)
                    if edge_id is None:
                        edge_id = next(self.edge_id)
                        while str(edge_id) in self.all_edge_ids:
                            edge_id = next(self.edge_id)
                        self.all_edge_ids.add(str(edge_id))
                    yield u, v, edge_id, edge_data
            else:
                for u, v, data in G.edges(data=True):
                    edge_data = data.copy()
                    edge_id = edge_data.pop("id", None)
                    if edge_id is None:
                        edge_id = next(self.edge_id)
                        while str(edge_id) in self.all_edge_ids:
                            edge_id = next(self.edge_id)
                        self.all_edge_ids.add(str(edge_id))
                    yield u, v, edge_id, edge_data

        edges_element = Element("edges")
        for u, v, key, edge_data in edge_key_data(G):
            kw = {"id": str(key)}
            try:
                edge_label = edge_data.pop("label")
                kw["label"] = str(edge_label)
            except KeyError:
                pass
            try:
                edge_weight = edge_data.pop("weight")
                kw["weight"] = str(edge_weight)
            except KeyError:
                pass
            try:
                edge_type = edge_data.pop("type")
                kw["type"] = str(edge_type)
            except KeyError:
                pass
            try:
                start = edge_data.pop("start")
                kw["start"] = str(start)
                self.alter_graph_mode_timeformat(start)
            except KeyError:
                pass
            try:
                end = edge_data.pop("end")
                kw["end"] = str(end)
                self.alter_graph_mode_timeformat(end)
            except KeyError:
                pass
            source_id = str(G.nodes[u].get("id", u))
            target_id = str(G.nodes[v].get("id", v))
            edge_element = Element("edge", source=source_id, target=target_id, **kw)
            default = G.graph.get("edge_default", {})
            if self.VERSION == "1.1":
                edge_data = self.add_slices(edge_element, edge_data)
            else:
                edge_data = self.add_spells(edge_element, edge_data)
            edge_data = self.add_viz(edge_element, edge_data)
            edge_data = self.add_attributes("edge", edge_element, edge_data, default)
            edges_element.append(edge_element)
        graph_element.append(edges_element)

    def add_attributes(self, node_or_edge, xml_obj, data, default):
        # Add attrvalues to node or edge
        attvalues = Element("attvalues")
        if len(data) == 0:
            return data
        mode = "static"
        for k, v in data.items():
            # rename generic multigraph key to avoid any name conflict
            if k == "key":
                k = "networkx_key"
            val_type = type(v)
            if val_type not in self.xml_type:
                raise TypeError(f"attribute value type is not allowed: {val_type}")
            if isinstance(v, list):
                # dynamic data
                for val, start, end in v:
                    val_type = type(val)
                    if start is not None or end is not None:
                        mode = "dynamic"
                        self.alter_graph_mode_timeformat(start)
                        self.alter_graph_mode_timeformat(end)
                        break
                attr_id = self.get_attr_id(
                    str(k), self.xml_type[val_type], node_or_edge, default, mode
                )
                for val, start, end in v:
                    e = Element("attvalue")
                    e.attrib["for"] = attr_id
                    e.attrib["value"] = str(val)
                    # Handle nan, inf, -inf differently
                    if val_type == float:
                        if e.attrib["value"] == "inf":
                            e.attrib["value"] = "INF"
                        elif e.attrib["value"] == "nan":
                            e.attrib["value"] = "NaN"
                        elif e.attrib["value"] == "-inf":
                            e.attrib["value"] = "-INF"
                    if start is not None:
                        e.attrib["start"] = str(start)
                    if end is not None:
                        e.attrib["end"] = str(end)
                    attvalues.append(e)
            else:
                # static data
                mode = "static"
                attr_id = self.get_attr_id(
                    str(k), self.xml_type[val_type], node_or_edge, default, mode
                )
                e = Element("attvalue")
                e.attrib["for"] = attr_id
                if isinstance(v, bool):
                    e.attrib["value"] = str(v).lower()
                else:
                    e.attrib["value"] = str(v)
                    # Handle float nan, inf, -inf differently
                    if val_type == float:
                        if e.attrib["value"] == "inf":
                            e.attrib["value"] = "INF"
                        elif e.attrib["value"] == "nan":
                            e.attrib["value"] = "NaN"
                        elif e.attrib["value"] == "-inf":
                            e.attrib["value"] = "-INF"
                attvalues.append(e)
        xml_obj.append(attvalues)
        return data

    def get_attr_id(self, title, attr_type, edge_or_node, default, mode):
        # find the id of the attribute or generate a new id
        try:
            return self.attr[edge_or_node][mode][title]
        except KeyError:
            # generate new id
            new_id = str(next(self.attr_id))
            self.attr[edge_or_node][mode][title] = new_id
            attr_kwargs = {"id": new_id, "title": title, "type": attr_type}
            attribute = Element("attribute", **attr_kwargs)
            # add subelement for data default value if present
            default_title = default.get(title)
            if default_title is not None:
                default_element = Element("default")
                default_element.text = str(default_title)
                attribute.append(default_element)
            # new insert it into the XML
            attributes_element = None
            for a in self.graph_element.findall("attributes"):
                # find existing attributes element by class and mode
                a_class = a.get("class")
                a_mode = a.get("mode", "static")
                if a_class == edge_or_node and a_mode == mode:
                    attributes_element = a
            if attributes_element is None:
                # create new attributes element
                attr_kwargs = {"mode": mode, "class": edge_or_node}
                attributes_element = Element("attributes", **attr_kwargs)
                self.graph_element.insert(0, attributes_element)
            attributes_element.append(attribute)
        return new_id

    def add_viz(self, element, node_data):
        viz = node_data.pop("viz", False)
        if viz:
            color = viz.get("color")
            if color is not None:
                if self.VERSION == "1.1":
                    e = Element(
                        f"{{{self.NS_VIZ}}}color",
                        r=str(color.get("r")),
                        g=str(color.get("g")),
                        b=str(color.get("b")),
                    )
                else:
                    e = Element(
                        f"{{{self.NS_VIZ}}}color",
                        r=str(color.get("r")),
                        g=str(color.get("g")),
                        b=str(color.get("b")),
                        a=str(color.get("a", 1.0)),
                    )
                element.append(e)

            size = viz.get("size")
            if size is not None:
                e = Element(f"{{{self.NS_VIZ}}}size", value=str(size))
                element.append(e)

            thickness = viz.get("thickness")
            if thickness is not None:
                e = Element(f"{{{self.NS_VIZ}}}thickness", value=str(thickness))
                element.append(e)

