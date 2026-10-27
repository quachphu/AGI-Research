    Raises
    ------
    NetworkXError
        The algorithm does not support DiGraph, MultiGraph and MultiDiGraph.
        If the input graph is an instance of one of these classes, a
        :exc:`NetworkXError` is raised.
        The algorithm can only be applied to chordal graphs. If the input
        graph is found to be non-chordal, a :exc:`NetworkXError` is raised.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G = nx.generators.classic.path_graph(10)
    >>> induced_nodes = nx.find_induced_nodes(G, 1, 9, 2)
    >>> sorted(induced_nodes)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    Notes
    -----
    G must be a chordal graph and (s,t) an edge that is not in G.

    If a treewidth_bound is provided, the search for induced nodes will end
    as soon as the treewidth_bound is exceeded.

    The algorithm is inspired by Algorithm 4 in [1]_.
    A formal definition of induced node can also be found on that reference.

    Self Loops are ignored

    References
    ----------
    .. [1] Learning Bounded Treewidth Bayesian Networks.
       Gal Elidan, Stephen Gould; JMLR, 9(Dec):2699--2731, 2008.
       http://jmlr.csail.mit.edu/papers/volume9/elidan08a/elidan08a.pdf
    """
    if not is_chordal(G):
        raise nx.NetworkXError("Input graph is not chordal.")

    H = nx.Graph(G)
    H.add_edge(s, t)
    induced_nodes = set()
    triplet = _find_chordality_breaker(H, s, treewidth_bound)
    while triplet:
        (u, v, w) = triplet
        induced_nodes.update(triplet)
        for n in triplet:
            if n != s:
                H.add_edge(s, n)
        triplet = _find_chordality_breaker(H, s, treewidth_bound)
    if induced_nodes:
        # Add t and the second node in the induced path from s to t.
        induced_nodes.add(t)
        for u in G[s]:
            if len(induced_nodes & set(G[u])) == 2:
                induced_nodes.add(u)
                break
    return induced_nodes


@nx._dispatchable
def chordal_graph_cliques(G):
    """Returns all maximal cliques of a chordal graph.

    The algorithm breaks the graph in connected components and performs a
    maximum cardinality search in each component to get the cliques.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    Yields
    ------
    frozenset of nodes
        Maximal cliques, each of which is a frozenset of
        nodes in `G`. The order of cliques is arbitrary.

    Raises
    ------
    NetworkXError
        The algorithm does not support DiGraph, MultiGraph and MultiDiGraph.
        The algorithm can only be applied to chordal graphs. If the input
        graph is found to be non-chordal, a :exc:`NetworkXError` is raised.

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
    ...     (7, 8),
    ... ]
    >>> G = nx.Graph(e)
    >>> G.add_node(9)
    >>> cliques = [c for c in chordal_graph_cliques(G)]
    >>> cliques[0]
    frozenset({1, 2, 3})
    """
    for C in (G.subgraph(c).copy() for c in connected_components(G)):
        if C.number_of_nodes() == 1:
            if nx.number_of_selfloops(C) > 0:
                raise nx.NetworkXError("Input graph is not chordal.")
            yield frozenset(C.nodes())
        else:
            unnumbered = set(C.nodes())
            v = arbitrary_element(C)
            unnumbered.remove(v)
            numbered = {v}
            clique_wanna_be = {v}
            while unnumbered:
                v = _max_cardinality_node(C, unnumbered, numbered)
                unnumbered.remove(v)
                numbered.add(v)
                new_clique_wanna_be = set(C.neighbors(v)) & numbered
                sg = C.subgraph(clique_wanna_be)
                if _is_complete_graph(sg):
                    new_clique_wanna_be.add(v)
                    if not new_clique_wanna_be >= clique_wanna_be:
                        yield frozenset(clique_wanna_be)
                    clique_wanna_be = new_clique_wanna_be
                else:
                    raise nx.NetworkXError("Input graph is not chordal.")
            yield frozenset(clique_wanna_be)


@nx._dispatchable
def chordal_graph_treewidth(G):
    """Returns the treewidth of the chordal graph G.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    Returns
    -------
    treewidth : int
        The size of the largest clique in the graph minus one.

    Raises
    ------
    NetworkXError
        The algorithm does not support DiGraph, MultiGraph and MultiDiGraph.
        The algorithm can only be applied to chordal graphs. If the input
        graph is found to be non-chordal, a :exc:`NetworkXError` is raised.

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
    ...     (7, 8),
    ... ]
    >>> G = nx.Graph(e)
    >>> G.add_node(9)
    >>> nx.chordal_graph_treewidth(G)
    3

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Tree_decomposition#Treewidth
    """
    if not is_chordal(G):
        raise nx.NetworkXError("Input graph is not chordal.")

    max_clique = -1
    for clique in nx.chordal_graph_cliques(G):
        max_clique = max(max_clique, len(clique))
    return max_clique - 1


def _is_complete_graph(G):
    """Returns True if G is a complete graph."""
    if nx.number_of_selfloops(G) > 0:
        raise nx.NetworkXError("Self loop found in _is_complete_graph()")
    n = G.number_of_nodes()
    if n < 2:
        return True
    e = G.number_of_edges()
    max_edges = (n * (n - 1)) / 2
    return e == max_edges


def _find_missing_edge(G):
    """Given a non-complete graph G, returns a missing edge."""
    nodes = set(G)
    for u in G:
        missing = nodes - set(list(G[u].keys()) + [u])
        if missing:
            return (u, missing.pop())


def _max_cardinality_node(G, choices, wanna_connect):
    """Returns a the node in choices that has more connections in G
    to nodes in wanna_connect.
    """
    max_number = -1
    for x in choices:
        number = len([y for y in G[x] if y in wanna_connect])
        if number > max_number:
            max_number = number
            max_cardinality_node = x
    return max_cardinality_node


def _find_chordality_breaker(G, s=None, treewidth_bound=sys.maxsize):
    """Given a graph G, starts a max cardinality search
    (starting from s if s is given and from an arbitrary node otherwise)
    trying to find a non-chordal cycle.

    If it does find one, it returns (u,v,w) where u,v,w are the three
    nodes that together with s are involved in the cycle.

    It ignores any self loops.
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Graph has no nodes.")
    unnumbered = set(G)
    if s is None:
        s = arbitrary_element(G)
    unnumbered.remove(s)
    numbered = {s}
    current_treewidth = -1
    while unnumbered:  # and current_treewidth <= treewidth_bound:
        v = _max_cardinality_node(G, unnumbered, numbered)
        unnumbered.remove(v)
        numbered.add(v)
        clique_wanna_be = set(G[v]) & numbered
        sg = G.subgraph(clique_wanna_be)
        if _is_complete_graph(sg):
            # The graph seems to be chordal by now. We update the treewidth
            current_treewidth = max(current_treewidth, len(clique_wanna_be))
            if current_treewidth > treewidth_bound:
                raise nx.NetworkXTreewidthBoundExceeded(
                    f"treewidth_bound exceeded: {current_treewidth}"
                )
        else:
            # sg is not a clique,
            # look for an edge that is not included in sg
            (u, w) = _find_missing_edge(sg)
            return (u, v, w)
    return ()


@not_implemented_for("directed")
@nx._dispatchable(returns_graph=True)
def complete_to_chordal_graph(G):
    """Return a copy of G completed to a chordal graph

    Adds edges to a copy of G to create a chordal graph. A graph G=(V,E) is
    called chordal if for each cycle with length bigger than 3, there exist
    two non-adjacent nodes connected by an edge (called a chord).

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    H : NetworkX graph
        The chordal enhancement of G
    alpha : Dictionary
            The elimination ordering of nodes of G

    Notes
    -----
    There are different approaches to calculate the chordal
    enhancement of a graph. The algorithm used here is called
    MCS-M and gives at least minimal (local) triangulation of graph. Note
    that this triangulation is not necessarily a global minimum.

    https://en.wikipedia.org/wiki/Chordal_graph

    References
    ----------
    .. [1] Berry, Anne & Blair, Jean & Heggernes, Pinar & Peyton, Barry. (2004)
           Maximum Cardinality Search for Computing Minimal Triangulations of
           Graphs.  Algorithmica. 39. 287-298. 10.1007/s00453-004-1084-3.

    Examples
    --------
    >>> from networkx.algorithms.chordal import complete_to_chordal_graph
    >>> G = nx.wheel_graph(10)
    >>> H, alpha = complete_to_chordal_graph(G)
    """
    H = G.copy()
    alpha = {node: 0 for node in H}
    if nx.is_chordal(H):
        return H, alpha
    chords = set()
    weight = {node: 0 for node in H.nodes()}
    unnumbered_nodes = list(H.nodes())
    for i in range(len(H.nodes()), 0, -1):
        # get the node in unnumbered_nodes with the maximum weight
        z = max(unnumbered_nodes, key=lambda node: weight[node])
        unnumbered_nodes.remove(z)
        alpha[z] = i
        update_nodes = []
        for y in unnumbered_nodes:
            if G.has_edge(y, z):
                update_nodes.append(y)
            else:
                # y_weight will be bigger than node weights between y and z
                y_weight = weight[y]
                lower_nodes = [
                    node for node in unnumbered_nodes if weight[node] < y_weight
                ]
                if nx.has_path(H.subgraph(lower_nodes + [z, y]), y, z):
                    update_nodes.append(y)
                    chords.add((z, y))
        # during calculation of paths the weights should not be updated
        for node in update_nodes:
            weight[node] += 1
    H.add_edges_from(chords)
    return H, alpha
<|endoftext|>"""Functions for computing and verifying regular graphs."""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["is_regular", "is_k_regular", "k_factor"]


@nx._dispatchable
def is_regular(G):
    """Determines whether the graph ``G`` is a regular graph.

    A regular graph is a graph where each vertex has the same degree. A
    regular digraph is a graph where the indegree and outdegree of each
    vertex are equal.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    bool
        Whether the given graph or digraph is regular.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 1)])
    >>> nx.is_regular(G)
    True

    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Graph has no nodes.")
    n1 = nx.utils.arbitrary_element(G)
    if not G.is_directed():
        d1 = G.degree(n1)
        return all(d1 == d for _, d in G.degree)
    else:
        d_in = G.in_degree(n1)
        in_regular = all(d_in == d for _, d in G.in_degree)
        d_out = G.out_degree(n1)
        out_regular = all(d_out == d for _, d in G.out_degree)
        return in_regular and out_regular


@not_implemented_for("directed")
@nx._dispatchable
def is_k_regular(G, k):
    """Determines whether the graph ``G`` is a k-regular graph.

    A k-regular graph is a graph where each vertex has degree k.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    bool
        Whether the given graph is k-regular.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)])
    >>> nx.is_k_regular(G, k=3)
    False

    """
    return all(d == k for n, d in G.degree)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(preserve_edge_attrs=True, returns_graph=True)
def k_factor(G, k, matching_weight="weight"):
    """Compute a k-factor of G

    A k-factor of a graph is a spanning k-regular subgraph.
    A spanning k-regular subgraph of G is a subgraph that contains
    each vertex of G and a subset of the edges of G such that each
    vertex has degree k.

    Parameters
    ----------
    G : NetworkX graph
      Undirected graph

    matching_weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.
       Used for finding the max-weighted perfect matching.
       If key not found, uses 1 as weight.

    Returns
    -------
    G2 : NetworkX graph
        A k-factor of G

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)])
    >>> G2 = nx.k_factor(G, k=1)
    >>> G2.edges()
    EdgeView([(1, 2), (3, 4)])

    References
    ----------
    .. [1] "An algorithm for computing simple k-factors.",
       Meijer, Henk, Yurai Núñez-Rodríguez, and David Rappaport,
       Information processing letters, 2009.
    """

    from networkx.algorithms.matching import is_perfect_matching, max_weight_matching

    class LargeKGadget:
        def __init__(self, k, degree, node, g):
            self.original = node
            self.g = g
            self.k = k
            self.degree = degree

            self.outer_vertices = [(node, x) for x in range(degree)]
            self.core_vertices = [(node, x + degree) for x in range(degree - k)]

        def replace_node(self):
            adj_view = self.g[self.original]
            neighbors = list(adj_view.keys())
            edge_attrs = list(adj_view.values())
            for outer, neighbor, edge_attrs in zip(
                self.outer_vertices, neighbors, edge_attrs
            ):
                self.g.add_edge(outer, neighbor, **edge_attrs)
            for core in self.core_vertices:
                for outer in self.outer_vertices:
                    self.g.add_edge(core, outer)
            self.g.remove_node(self.original)

        def restore_node(self):
            self.g.add_node(self.original)
            for outer in self.outer_vertices:
                adj_view = self.g[outer]
                for neighbor, edge_attrs in list(adj_view.items()):
                    if neighbor not in self.core_vertices:
                        self.g.add_edge(self.original, neighbor, **edge_attrs)
                        break
            g.remove_nodes_from(self.outer_vertices)
            g.remove_nodes_from(self.core_vertices)

    class SmallKGadget:
        def __init__(self, k, degree, node, g):
            self.original = node
            self.k = k
            self.degree = degree
            self.g = g

            self.outer_vertices = [(node, x) for x in range(degree)]
            self.inner_vertices = [(node, x + degree) for x in range(degree)]
            self.core_vertices = [(node, x + 2 * degree) for x in range(k)]

        def replace_node(self):
            adj_view = self.g[self.original]
            for outer, inner, (neighbor, edge_attrs) in zip(
                self.outer_vertices, self.inner_vertices, list(adj_view.items())
            ):
                self.g.add_edge(outer, inner)
                self.g.add_edge(outer, neighbor, **edge_attrs)
            for core in self.core_vertices:
                for inner in self.inner_vertices:
                    self.g.add_edge(core, inner)
            self.g.remove_node(self.original)

        def restore_node(self):
            self.g.add_node(self.original)
            for outer in self.outer_vertices:
                adj_view = self.g[outer]
                for neighbor, edge_attrs in adj_view.items():
                    if neighbor not in self.core_vertices:
                        self.g.add_edge(self.original, neighbor, **edge_attrs)
                        break
            self.g.remove_nodes_from(self.outer_vertices)
            self.g.remove_nodes_from(self.inner_vertices)
            self.g.remove_nodes_from(self.core_vertices)

    # Step 1
    if any(d < k for _, d in G.degree):
        raise nx.NetworkXUnfeasible("Graph contains a vertex with degree less than k")
    g = G.copy()

    # Step 2
    gadgets = []
    for node, degree in list(g.degree):
        if k < degree / 2.0:
            gadget = SmallKGadget(k, degree, node, g)
        else:
            gadget = LargeKGadget(k, degree, node, g)
        gadget.replace_node()
        gadgets.append(gadget)

    # Step 3
    matching = max_weight_matching(g, maxcardinality=True, weight=matching_weight)

    # Step 4
    if not is_perfect_matching(g, matching):
        raise nx.NetworkXUnfeasible(
            "Cannot find k-factor because no perfect matching exists"
        )

    for edge in g.edges():
        if edge not in matching and (edge[1], edge[0]) not in matching:
            g.remove_edge(edge[0], edge[1])

    for gadget in gadgets:
        gadget.restore_node()

    return g
<|endoftext|>from collections import defaultdict

import networkx as nx

__all__ = ["check_planarity", "is_planar", "PlanarEmbedding"]


@nx._dispatchable
def is_planar(G):
    """Returns True if and only if `G` is planar.

    A graph is *planar* iff it can be drawn in a plane without
    any edge intersections.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    bool
       Whether the graph is planar.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2)])
    >>> nx.is_planar(G)
    True
    >>> nx.is_planar(nx.complete_graph(5))
    False

    See Also
    --------
    check_planarity :
        Check if graph is planar *and* return a `PlanarEmbedding` instance if True.
    """

    return check_planarity(G, counterexample=False)[0]


@nx._dispatchable(returns_graph=True)
def check_planarity(G, counterexample=False):
    """Check if a graph is planar and return a counterexample or an embedding.

    A graph is planar iff it can be drawn in a plane without
    any edge intersections.

    Parameters
    ----------
    G : NetworkX graph
    counterexample : bool
        A Kuratowski subgraph (to proof non planarity) is only returned if set
        to true.

    Returns
    -------
    (is_planar, certificate) : (bool, NetworkX graph) tuple
        is_planar is true if the graph is planar.
        If the graph is planar `certificate` is a PlanarEmbedding
        otherwise it is a Kuratowski subgraph.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2)])
    >>> is_planar, P = nx.check_planarity(G)
    >>> print(is_planar)
    True

    When `G` is planar, a `PlanarEmbedding` instance is returned:

    >>> P.get_data()
    {0: [1, 2], 1: [0], 2: [0]}

    Notes
    -----
    A (combinatorial) embedding consists of cyclic orderings of the incident
    edges at each vertex. Given such an embedding there are multiple approaches
    discussed in literature to drawing the graph (subject to various
    constraints, e.g. integer coordinates), see e.g. [2].

    The planarity check algorithm and extraction of the combinatorial embedding
    is based on the Left-Right Planarity Test [1].

    A counterexample is only generated if the corresponding parameter is set,
    because the complexity of the counterexample generation is higher.

    See also
    --------
    is_planar :
        Check for planarity without creating a `PlanarEmbedding` or counterexample.

    References
    ----------
    .. [1] Ulrik Brandes:
        The Left-Right Planarity Test
        2009
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.217.9208
    .. [2] Takao Nishizeki, Md Saidur Rahman:
        Planar graph drawing
        Lecture Notes Series on Computing: Volume 12
        2004
    """

    planarity_state = LRPlanarity(G)
    embedding = planarity_state.lr_planarity()
    if embedding is None:
        # graph is not planar
        if counterexample:
            return False, get_counterexample(G)
        else:
            return False, None
    else:
        # graph is planar
        return True, embedding


@nx._dispatchable(returns_graph=True)
def check_planarity_recursive(G, counterexample=False):
    """Recursive version of :meth:`check_planarity`."""
    planarity_state = LRPlanarity(G)
    embedding = planarity_state.lr_planarity_recursive()
    if embedding is None:
        # graph is not planar
        if counterexample:
            return False, get_counterexample_recursive(G)
        else:
            return False, None
    else:
        # graph is planar
        return True, embedding


@nx._dispatchable(returns_graph=True)
def get_counterexample(G):
    """Obtains a Kuratowski subgraph.

    Raises nx.NetworkXException if G is planar.

    The function removes edges such that the graph is still not planar.
    At some point the removal of any edge would make the graph planar.
    This subgraph must be a Kuratowski subgraph.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    subgraph : NetworkX graph
        A Kuratowski subgraph that proves that G is not planar.

    """
    # copy graph
    G = nx.Graph(G)

    if check_planarity(G)[0]:
        raise nx.NetworkXException("G is planar - no counter example.")

    # find Kuratowski subgraph
    subgraph = nx.Graph()
    for u in G:
        nbrs = list(G[u])
        for v in nbrs:
            G.remove_edge(u, v)
            if check_planarity(G)[0]:
                G.add_edge(u, v)
                subgraph.add_edge(u, v)

    return subgraph


@nx._dispatchable(returns_graph=True)
def get_counterexample_recursive(G):
    """Recursive version of :meth:`get_counterexample`."""

    # copy graph
    G = nx.Graph(G)

    if check_planarity_recursive(G)[0]:
        raise nx.NetworkXException("G is planar - no counter example.")

    # find Kuratowski subgraph
    subgraph = nx.Graph()
    for u in G:
        nbrs = list(G[u])
        for v in nbrs:
            G.remove_edge(u, v)
            if check_planarity_recursive(G)[0]:
                G.add_edge(u, v)
                subgraph.add_edge(u, v)

    return subgraph


class Interval:
    """Represents a set of return edges.

    All return edges in an interval induce a same constraint on the contained
    edges, which means that all edges must either have a left orientation or
    all edges must have a right orientation.
    """

    def __init__(self, low=None, high=None):
        self.low = low
        self.high = high

    def empty(self):
        """Check if the interval is empty"""
        return self.low is None and self.high is None

    def copy(self):
        """Returns a copy of this interval"""
        return Interval(self.low, self.high)

    def conflicting(self, b, planarity_state):
        """Returns True if interval I conflicts with edge b"""
        return (
            not self.empty()
            and planarity_state.lowpt[self.high] > planarity_state.lowpt[b]
        )


class ConflictPair:
    """Represents a different constraint between two intervals.

    The edges in the left interval must have a different orientation than
    the one in the right interval.
    """

    def __init__(self, left=Interval(), right=Interval()):
        self.left = left
        self.right = right

    def swap(self):
        """Swap left and right intervals"""
        temp = self.left
        self.left = self.right
        self.right = temp

    def lowest(self, planarity_state):
        """Returns the lowest lowpoint of a conflict pair"""
        if self.left.empty():
            return planarity_state.lowpt[self.right.low]
        if self.right.empty():
            return planarity_state.lowpt[self.left.low]
        return min(
            planarity_state.lowpt[self.left.low], planarity_state.lowpt[self.right.low]
        )


def top_of_stack(l):
    """Returns the element on top of the stack."""
    if not l:
        return None
    return l[-1]


class LRPlanarity:
    """A class to maintain the state during planarity check."""

    __slots__ = [
        "G",
        "roots",
        "height",
        "lowpt",
        "lowpt2",
        "nesting_depth",
        "parent_edge",
        "DG",
        "adjs",
        "ordered_adjs",
        "ref",
        "side",
        "S",
        "stack_bottom",
        "lowpt_edge",
        "left_ref",
        "right_ref",
        "embedding",
    ]

    def __init__(self, G):
        # copy G without adding self-loops
        self.G = nx.Graph()
        self.G.add_nodes_from(G.nodes)
        for e in G.edges:
            if e[0] != e[1]:
                self.G.add_edge(e[0], e[1])

        self.roots = []

        # distance from tree root
        self.height = defaultdict(lambda: None)

        self.lowpt = {}  # height of lowest return point of an edge
        self.lowpt2 = {}  # height of second lowest return point
        self.nesting_depth = {}  # for nesting order

        # None -> missing edge
        self.parent_edge = defaultdict(lambda: None)

        # oriented DFS graph
        self.DG = nx.DiGraph()
        self.DG.add_nodes_from(G.nodes)

        self.adjs = {}
        self.ordered_adjs = {}

        self.ref = defaultdict(lambda: None)
        self.side = defaultdict(lambda: 1)

        # stack of conflict pairs
        self.S = []
        self.stack_bottom = {}
        self.lowpt_edge = {}

        self.left_ref = {}
        self.right_ref = {}

        self.embedding = PlanarEmbedding()

    def lr_planarity(self):
        """Execute the LR planarity test.

        Returns
        -------
        embedding : dict
            If the graph is planar an embedding is returned. Otherwise None.
        """
        if self.G.order() > 2 and self.G.size() > 3 * self.G.order() - 6:
            # graph is not planar
            return None

        # make adjacency lists for dfs
        for v in self.G:
            self.adjs[v] = list(self.G[v])

        # orientation of the graph by depth first search traversal
        for v in self.G:
            if self.height[v] is None:
                self.height[v] = 0
                self.roots.append(v)
                self.dfs_orientation(v)

        # Free no longer used variables
        self.G = None
        self.lowpt2 = None
        self.adjs = None

        # testing
        for v in self.DG:  # sort the adjacency lists by nesting depth
            # note: this sorting leads to non linear time
            self.ordered_adjs[v] = sorted(
                self.DG[v], key=lambda x: self.nesting_depth[(v, x)]
            )
        for v in self.roots:
            if not self.dfs_testing(v):
                return None

        # Free no longer used variables
        self.height = None
        self.lowpt = None
        self.S = None
        self.stack_bottom = None
        self.lowpt_edge = None

        for e in self.DG.edges:
            self.nesting_depth[e] = self.sign(e) * self.nesting_depth[e]

        self.embedding.add_nodes_from(self.DG.nodes)
        for v in self.DG:
            # sort the adjacency lists again
            self.ordered_adjs[v] = sorted(
                self.DG[v], key=lambda x: self.nesting_depth[(v, x)]
            )
            # initialize the embedding
            previous_node = None
            for w in self.ordered_adjs[v]:
                self.embedding.add_half_edge(v, w, ccw=previous_node)
                previous_node = w

        # Free no longer used variables
        self.DG = None
        self.nesting_depth = None
        self.ref = None

        # compute the complete embedding
        for v in self.roots:
            self.dfs_embedding(v)

        # Free no longer used variables
        self.roots = None
        self.parent_edge = None
        self.ordered_adjs = None
        self.left_ref = None
        self.right_ref = None
        self.side = None

        return self.embedding

    def lr_planarity_recursive(self):
        """Recursive version of :meth:`lr_planarity`."""
        if self.G.order() > 2 and self.G.size() > 3 * self.G.order() - 6:
            # graph is not planar
            return None

        # orientation of the graph by depth first search traversal
        for v in self.G:
            if self.height[v] is None:
                self.height[v] = 0
                self.roots.append(v)
                self.dfs_orientation_recursive(v)

        # Free no longer used variable
        self.G = None

        # testing
        for v in self.DG:  # sort the adjacency lists by nesting depth
            # note: this sorting leads to non linear time
            self.ordered_adjs[v] = sorted(
                self.DG[v], key=lambda x: self.nesting_depth[(v, x)]
            )
        for v in self.roots:
            if not self.dfs_testing_recursive(v):
                return None

        for e in self.DG.edges:
            self.nesting_depth[e] = self.sign_recursive(e) * self.nesting_depth[e]

        self.embedding.add_nodes_from(self.DG.nodes)
        for v in self.DG:
            # sort the adjacency lists again
            self.ordered_adjs[v] = sorted(
                self.DG[v], key=lambda x: self.nesting_depth[(v, x)]
            )
            # initialize the embedding
            previous_node = None
            for w in self.ordered_adjs[v]:
                self.embedding.add_half_edge(v, w, ccw=previous_node)
                previous_node = w

        # compute the complete embedding
        for v in self.roots:
            self.dfs_embedding_recursive(v)

        return self.embedding

    def dfs_orientation(self, v):
        """Orient the graph by DFS, compute lowpoints and nesting order."""
        # the recursion stack
        dfs_stack = [v]
        # index of next edge to handle in adjacency list of each node
        ind = defaultdict(lambda: 0)
        # boolean to indicate whether to skip the initial work for an edge
        skip_init = defaultdict(lambda: False)

        while dfs_stack:
            v = dfs_stack.pop()
            e = self.parent_edge[v]

            for w in self.adjs[v][ind[v] :]:
                vw = (v, w)

                if not skip_init[vw]:
                    if (v, w) in self.DG.edges or (w, v) in self.DG.edges:
                        ind[v] += 1
                        continue  # the edge was already oriented

                    self.DG.add_edge(v, w)  # orient the edge

                    self.lowpt[vw] = self.height[v]
                    self.lowpt2[vw] = self.height[v]
                    if self.height[w] is None:  # (v, w) is a tree edge
                        self.parent_edge[w] = vw
                        self.height[w] = self.height[v] + 1

                        dfs_stack.append(v)  # revisit v after finishing w
                        dfs_stack.append(w)  # visit w next
                        skip_init[vw] = True  # don't redo this block
                        break  # handle next node in dfs_stack (i.e. w)
                    else:  # (v, w) is a back edge
                        self.lowpt[vw] = self.height[w]

                # determine nesting graph
                self.nesting_depth[vw] = 2 * self.lowpt[vw]
                if self.lowpt2[vw] < self.height[v]:  # chordal
                    self.nesting_depth[vw] += 1

                # update lowpoints of parent edge e
                if e is not None:
                    if self.lowpt[vw] < self.lowpt[e]:
                        self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[vw])
                        self.lowpt[e] = self.lowpt[vw]
                    elif self.lowpt[vw] > self.lowpt[e]:
                        self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[vw])
                    else:
                        self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[vw])

                ind[v] += 1

    def dfs_orientation_recursive(self, v):
        """Recursive version of :meth:`dfs_orientation`."""
        e = self.parent_edge[v]
        for w in self.G[v]:
            if (v, w) in self.DG.edges or (w, v) in self.DG.edges:
                continue  # the edge was already oriented
            vw = (v, w)
            self.DG.add_edge(v, w)  # orient the edge

            self.lowpt[vw] = self.height[v]
            self.lowpt2[vw] = self.height[v]
            if self.height[w] is None:  # (v, w) is a tree edge
                self.parent_edge[w] = vw
                self.height[w] = self.height[v] + 1
                self.dfs_orientation_recursive(w)
            else:  # (v, w) is a back edge
                self.lowpt[vw] = self.height[w]

            # determine nesting graph
            self.nesting_depth[vw] = 2 * self.lowpt[vw]
            if self.lowpt2[vw] < self.height[v]:  # chordal
                self.nesting_depth[vw] += 1

            # update lowpoints of parent edge e
            if e is not None:
                if self.lowpt[vw] < self.lowpt[e]:
                    self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[vw])
                    self.lowpt[e] = self.lowpt[vw]
                elif self.lowpt[vw] > self.lowpt[e]:
                    self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[vw])
                else:
                    self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[vw])

    def dfs_testing(self, v):
        """Test for LR partition."""
        # the recursion stack
        dfs_stack = [v]
        # index of next edge to handle in adjacency list of each node
        ind = defaultdict(lambda: 0)
        # boolean to indicate whether to skip the initial work for an edge
        skip_init = defaultdict(lambda: False)

        while dfs_stack:
            v = dfs_stack.pop()
            e = self.parent_edge[v]
            # to indicate whether to skip the final block after the for loop
            skip_final = False

            for w in self.ordered_adjs[v][ind[v] :]:
                ei = (v, w)

                if not skip_init[ei]:
                    self.stack_bottom[ei] = top_of_stack(self.S)

                    if ei == self.parent_edge[w]:  # tree edge
                        dfs_stack.append(v)  # revisit v after finishing w
                        dfs_stack.append(w)  # visit w next
                        skip_init[ei] = True  # don't redo this block
                        skip_final = True  # skip final work after breaking
                        break  # handle next node in dfs_stack (i.e. w)
                    else:  # back edge
                        self.lowpt_edge[ei] = ei
                        self.S.append(ConflictPair(right=Interval(ei, ei)))

                # integrate new return edges
                if self.lowpt[ei] < self.height[v]:
                    if w == self.ordered_adjs[v][0]:  # e_i has return edge
                        self.lowpt_edge[e] = self.lowpt_edge[ei]
                    else:  # add constraints of e_i
                        if not self.add_constraints(ei, e):
                            # graph is not planar
                            return False

                ind[v] += 1

            if not skip_final:
                # remove back edges returning to parent
                if e is not None:  # v isn't root
                    self.remove_back_edges(e)

        return True

    def dfs_testing_recursive(self, v):
        """Recursive version of :meth:`dfs_testing`."""
        e = self.parent_edge[v]
        for w in self.ordered_adjs[v]:
            ei = (v, w)
            self.stack_bottom[ei] = top_of_stack(self.S)
            if ei == self.parent_edge[w]:  # tree edge
                if not self.dfs_testing_recursive(w):
                    return False
            else:  # back edge
                self.lowpt_edge[ei] = ei
                self.S.append(ConflictPair(right=Interval(ei, ei)))

            # integrate new return edges
            if self.lowpt[ei] < self.height[v]:
                if w == self.ordered_adjs[v][0]:  # e_i has return edge
                    self.lowpt_edge[e] = self.lowpt_edge[ei]
                else:  # add constraints of e_i
                    if not self.add_constraints(ei, e):
                        # graph is not planar
                        return False

        # remove back edges returning to parent
        if e is not None:  # v isn't root
            self.remove_back_edges(e)
        return True

    def add_constraints(self, ei, e):
        P = ConflictPair()
        # merge return edges of e_i into P.right
        while True:
            Q = self.S.pop()
            if not Q.left.empty():
                Q.swap()
            if not Q.left.empty():  # not planar
                return False
            if self.lowpt[Q.right.low] > self.lowpt[e]:
                # merge intervals
                if P.right.empty():  # topmost interval
                    P.right = Q.right.copy()
                else:
                    self.ref[P.right.low] = Q.right.high
                P.right.low = Q.right.low
            else:  # align
                self.ref[Q.right.low] = self.lowpt_edge[e]
            if top_of_stack(self.S) == self.stack_bottom[ei]:
                break
        # merge conflicting return edges of e_1,...,e_i-1 into P.L
        while top_of_stack(self.S).left.conflicting(ei, self) or top_of_stack(
            self.S
        ).right.conflicting(ei, self):
            Q = self.S.pop()
            if Q.right.conflicting(ei, self):
                Q.swap()
            if Q.right.conflicting(ei, self):  # not planar
                return False
            # merge interval below lowpt(e_i) into P.R
            self.ref[P.right.low] = Q.right.high
            if Q.right.low is not None:
                P.right.low = Q.right.low

            if P.left.empty():  # topmost interval
                P.left = Q.left.copy()
            else:
                self.ref[P.left.low] = Q.left.high
            P.left.low = Q.left.low

        if not (P.left.empty() and P.right.empty()):
            self.S.append(P)
        return True

    def remove_back_edges(self, e):
        u = e[0]
        # trim back edges ending at parent u
        # drop entire conflict pairs
        while self.S and top_of_stack(self.S).lowest(self) == self.height[u]:
            P = self.S.pop()
            if P.left.low is not None:
                self.side[P.left.low] = -1

        if self.S:  # one more conflict pair to consider
            P = self.S.pop()
            # trim left interval
            while P.left.high is not None and P.left.high[1] == u:
                P.left.high = self.ref[P.left.high]
            if P.left.high is None and P.left.low is not None:
                # just emptied
                self.ref[P.left.low] = P.right.low
                self.side[P.left.low] = -1
                P.left.low = None
            # trim right interval
            while P.right.high is not None and P.right.high[1] == u:
                P.right.high = self.ref[P.right.high]
            if P.right.high is None and P.right.low is not None:
                # just emptied
                self.ref[P.right.low] = P.left.low
                self.side[P.right.low] = -1
                P.right.low = None
            self.S.append(P)

        # side of e is side of a highest return edge
        if self.lowpt[e] < self.height[u]:  # e has return edge
            hl = top_of_stack(self.S).left.high
            hr = top_of_stack(self.S).right.high

            if hl is not None and (hr is None or self.lowpt[hl] > self.lowpt[hr]):
                self.ref[e] = hl
            else:
                self.ref[e] = hr

    def dfs_embedding(self, v):
        """Completes the embedding."""
        # the recursion stack
        dfs_stack = [v]
        # index of next edge to handle in adjacency list of each node
        ind = defaultdict(lambda: 0)

        while dfs_stack:
            v = dfs_stack.pop()

            for w in self.ordered_adjs[v][ind[v] :]:
                ind[v] += 1
                ei = (v, w)

                if ei == self.parent_edge[w]:  # tree edge
                    self.embedding.add_half_edge_first(w, v)
                    self.left_ref[v] = w
                    self.right_ref[v] = w

                    dfs_stack.append(v)  # revisit v after finishing w
                    dfs_stack.append(w)  # visit w next
                    break  # handle next node in dfs_stack (i.e. w)
                else:  # back edge
                    if self.side[ei] == 1:
                        self.embedding.add_half_edge(w, v, ccw=self.right_ref[w])
                    else:
                        self.embedding.add_half_edge(w, v, cw=self.left_ref[w])
                        self.left_ref[w] = v

    def dfs_embedding_recursive(self, v):
        """Recursive version of :meth:`dfs_embedding`."""
        for w in self.ordered_adjs[v]:
            ei = (v, w)
            if ei == self.parent_edge[w]:  # tree edge
                self.embedding.add_half_edge_first(w, v)
                self.left_ref[v] = w
                self.right_ref[v] = w
                self.dfs_embedding_recursive(w)
            else:  # back edge
                if self.side[ei] == 1:
                    # place v directly after right_ref[w] in embed. list of w
                    self.embedding.add_half_edge(w, v, ccw=self.right_ref[w])
                else:
                    # place v directly before left_ref[w] in embed. list of w
                    self.embedding.add_half_edge(w, v, cw=self.left_ref[w])
                    self.left_ref[w] = v

    def sign(self, e):
        """Resolve the relative side of an edge to the absolute side."""
        # the recursion stack
        dfs_stack = [e]
        # dict to remember reference edges
        old_ref = defaultdict(lambda: None)

        while dfs_stack:
            e = dfs_stack.pop()

            if self.ref[e] is not None:
                dfs_stack.append(e)  # revisit e after finishing self.ref[e]
                dfs_stack.append(self.ref[e])  # visit self.ref[e] next
                old_ref[e] = self.ref[e]  # remember value of self.ref[e]
                self.ref[e] = None
            else:
                self.side[e] *= self.side[old_ref[e]]

        return self.side[e]

    def sign_recursive(self, e):
        """Recursive version of :meth:`sign`."""
        if self.ref[e] is not None:
            self.side[e] = self.side[e] * self.sign_recursive(self.ref[e])
            self.ref[e] = None
        return self.side[e]


class PlanarEmbedding(nx.DiGraph):
    """Represents a planar graph with its planar embedding.

    The planar embedding is given by a `combinatorial embedding
    <https://en.wikipedia.org/wiki/Graph_embedding#Combinatorial_embedding>`_.

    .. note:: `check_planarity` is the preferred way to check if a graph is planar.

    **Neighbor ordering:**

    In comparison to a usual graph structure, the embedding also stores the
    order of all neighbors for every vertex.
    The order of the neighbors can be given in clockwise (cw) direction or
    counterclockwise (ccw) direction. This order is stored as edge attributes
    in the underlying directed graph. For the edge (u, v) the edge attribute
    'cw' is set to the neighbor of u that follows immediately after v in
    clockwise direction.

    In order for a PlanarEmbedding to be valid it must fulfill multiple
    conditions. It is possible to check if these conditions are fulfilled with
    the method :meth:`check_structure`.
    The conditions are:

    * Edges must go in both directions (because the edge attributes differ)
    * Every edge must have a 'cw' and 'ccw' attribute which corresponds to a
      correct planar embedding.

    As long as a PlanarEmbedding is invalid only the following methods should
    be called:

    * :meth:`add_half_edge`
    * :meth:`connect_components`

    Even though the graph is a subclass of nx.DiGraph, it can still be used
    for algorithms that require undirected graphs, because the method
    :meth:`is_directed` is overridden. This is possible, because a valid
    PlanarGraph must have edges in both directions.

    **Half edges:**

    In methods like `add_half_edge` the term "half-edge" is used, which is
    a term that is used in `doubly connected edge lists
    <https://en.wikipedia.org/wiki/Doubly_connected_edge_list>`_. It is used
    to emphasize that the edge is only in one direction and there exists
    another half-edge in the opposite direction.
    While conventional edges always have two faces (including outer face) next
    to them, it is possible to assign each half-edge *exactly one* face.
    For a half-edge (u, v) that is oriented such that u is below v then the
    face that belongs to (u, v) is to the right of this half-edge.

    See Also
    --------
    is_planar :
        Preferred way to check if an existing graph is planar.

    check_planarity :
        A convenient way to create a `PlanarEmbedding`. If not planar,
        it returns a subgraph that shows this.

    Examples
    --------

    Create an embedding of a star graph (compare `nx.star_graph(3)`):

    >>> G = nx.PlanarEmbedding()
    >>> G.add_half_edge(0, 1)
    >>> G.add_half_edge(0, 2, ccw=1)
    >>> G.add_half_edge(0, 3, ccw=2)
    >>> G.add_half_edge(1, 0)
    >>> G.add_half_edge(2, 0)
    >>> G.add_half_edge(3, 0)

    Alternatively the same embedding can also be defined in counterclockwise
    orientation. The following results in exactly the same PlanarEmbedding:

    >>> G = nx.PlanarEmbedding()
    >>> G.add_half_edge(0, 1)
    >>> G.add_half_edge(0, 3, cw=1)
    >>> G.add_half_edge(0, 2, cw=3)
    >>> G.add_half_edge(1, 0)
    >>> G.add_half_edge(2, 0)
    >>> G.add_half_edge(3, 0)

    After creating a graph, it is possible to validate that the PlanarEmbedding
    object is correct:

    >>> G.check_structure()

    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        self.add_edge = self.__forbidden
        self.add_edges_from = self.__forbidden
        self.add_weighted_edges_from = self.__forbidden

    def __forbidden(self, *args, **kwargs):
        """Forbidden operation

        Any edge additions to a PlanarEmbedding should be done using
        method `add_half_edge`.
        """
        raise NotImplementedError(
            "Use `add_half_edge` method to add edges to a PlanarEmbedding."
        )

    def get_data(self):
        """Converts the adjacency structure into a better readable structure.

        Returns
        -------
        embedding : dict
            A dict mapping all nodes to a list of neighbors sorted in
            clockwise order.

        See Also
        --------
        set_data

        """
        embedding = {}
        for v in self:
            embedding[v] = list(self.neighbors_cw_order(v))
        return embedding

    def set_data(self, data):
        """Inserts edges according to given sorted neighbor list.

        The input format is the same as the output format of get_data().

        Parameters
        ----------
        data : dict
            A dict mapping all nodes to a list of neighbors sorted in
            clockwise order.

        See Also
        --------
        get_data

        """
        for v in data:
            ref = None
            for w in reversed(data[v]):
                self.add_half_edge(v, w, cw=ref)
                ref = w

    def remove_node(self, n):
        """Remove node n.

        Removes the node n and all adjacent edges, updating the
        PlanarEmbedding to account for any resulting edge removal.
        Attempting to remove a non-existent node will raise an exception.

        Parameters
        ----------
        n : node
           A node in the graph

        Raises
        ------
        NetworkXError
           If n is not in the graph.

        See Also
        --------
        remove_nodes_from

        """
        try:
            for u in self._pred[n]:
                succs_u = self._succ[u]
                un_cw = succs_u[n]["cw"]
                un_ccw = succs_u[n]["ccw"]
                del succs_u[n]
                del self._pred[u][n]
                if n != un_cw:
                    succs_u[un_cw]["ccw"] = un_ccw
                    succs_u[un_ccw]["cw"] = un_cw
            del self._node[n]
            del self._succ[n]
            del self._pred[n]
        except KeyError as err:  # NetworkXError if n not in self
            raise nx.NetworkXError(
                f"The node {n} is not in the planar embedding."
            ) from err
        nx._clear_cache(self)

    def remove_nodes_from(self, nodes):
        """Remove multiple nodes.

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, set, etc.).  If a node
            in the container is not in the graph it is silently ignored.

        See Also
        --------
        remove_node

        Notes
        -----
        When removing nodes from an iterator over the graph you are changing,
        a `RuntimeError` will be raised with message:
        `RuntimeError: dictionary changed size during iteration`. This
        happens when the graph's underlying dictionary is modified during
        iteration. To avoid this error, evaluate the iterator into a separate
        object, e.g. by using `list(iterator_of_nodes)`, and pass this
        object to `G.remove_nodes_from`.

        """
        for n in nodes:
            if n in self._node:
                self.remove_node(n)
            # silently skip non-existing nodes

    def neighbors_cw_order(self, v):
        """Generator for the neighbors of v in clockwise order.

        Parameters
        ----------
        v : node

        Yields
        ------
        node

        """
        succs = self._succ[v]
        if not succs:
            # v has no neighbors
            return
        start_node = next(reversed(succs))
        yield start_node
        current_node = succs[start_node]["cw"]
        while start_node != current_node:
            yield current_node
            current_node = succs[current_node]["cw"]

    def add_half_edge(self, start_node, end_node, *, cw=None, ccw=None):
        """Adds a half-edge from `start_node` to `end_node`.

        If the half-edge is not the first one out of `start_node`, a reference
        node must be provided either in the clockwise (parameter `cw`) or in
        the counterclockwise (parameter `ccw`) direction. Only one of `cw`/`ccw`
        can be specified (or neither in the case of the first edge).
        Note that specifying a reference in the clockwise (`cw`) direction means
        inserting the new edge in the first counterclockwise position with
        respect to the reference (and vice-versa).

        Parameters
        ----------
        start_node : node
            Start node of inserted edge.
        end_node : node
            End node of inserted edge.
        cw, ccw: node
            End node of reference edge.
            Omit or pass `None` if adding the first out-half-edge of `start_node`.


        Raises
        ------
        NetworkXException
            If the `cw` or `ccw` node is not a successor of `start_node`.
            If `start_node` has successors, but neither `cw` or `ccw` is provided.
            If both `cw` and `ccw` are specified.

        See Also
        --------
        connect_components
        """

        succs = self._succ.get(start_node)
        if succs:
            # there is already some edge out of start_node
            leftmost_nbr = next(reversed(self._succ[start_node]))
            if cw is not None:
                if cw not in succs:
                    raise nx.NetworkXError("Invalid clockwise reference node.")
                if ccw is not None:
                    raise nx.NetworkXError("Only one of cw/ccw can be specified.")
                ref_ccw = succs[cw]["ccw"]
                super().add_edge(start_node, end_node, cw=cw, ccw=ref_ccw)
                succs[ref_ccw]["cw"] = end_node
                succs[cw]["ccw"] = end_node
                # when (cw == leftmost_nbr), the newly added neighbor is
                # already at the end of dict self._succ[start_node] and
                # takes the place of the former leftmost_nbr
                move_leftmost_nbr_to_end = cw != leftmost_nbr
            elif ccw is not None:
                if ccw not in succs:
                    raise nx.NetworkXError("Invalid counterclockwise reference node.")
                ref_cw = succs[ccw]["cw"]
                super().add_edge(start_node, end_node, cw=ref_cw, ccw=ccw)
                succs[ref_cw]["ccw"] = end_node
                succs[ccw]["cw"] = end_node
                move_leftmost_nbr_to_end = True
            else:
                raise nx.NetworkXError(
                    "Node already has out-half-edge(s), either cw or ccw reference node required."
                )
            if move_leftmost_nbr_to_end:
                # LRPlanarity (via self.add_half_edge_first()) requires that
                # we keep track of the leftmost neighbor, which we accomplish
                # by keeping it as the last key in dict self._succ[start_node]
                succs[leftmost_nbr] = succs.pop(leftmost_nbr)

        else:
            if cw is not None or ccw is not None:
                raise nx.NetworkXError("Invalid reference node.")
            # adding the first edge out of start_node
            super().add_edge(start_node, end_node, ccw=end_node, cw=end_node)

    def check_structure(self):
        """Runs without exceptions if this object is valid.

        Checks that the following properties are fulfilled:

        * Edges go in both directions (because the edge attributes differ).
        * Every edge has a 'cw' and 'ccw' attribute which corresponds to a
          correct planar embedding.

        Running this method verifies that the underlying Graph must be planar.

        Raises
        ------
        NetworkXException
            This exception is raised with a short explanation if the
            PlanarEmbedding is invalid.
        """
        # Check fundamental structure
        for v in self:
            try:
                sorted_nbrs = set(self.neighbors_cw_order(v))
            except KeyError as err:
                msg = f"Bad embedding. Missing orientation for a neighbor of {v}"
                raise nx.NetworkXException(msg) from err

            unsorted_nbrs = set(self[v])
            if sorted_nbrs != unsorted_nbrs:
                msg = "Bad embedding. Edge orientations not set correctly."
                raise nx.NetworkXException(msg)
            for w in self[v]:
                # Check if opposite half-edge exists
                if not self.has_edge(w, v):
                    msg = "Bad embedding. Opposite half-edge is missing."
                    raise nx.NetworkXException(msg)

        # Check planarity
        counted_half_edges = set()
        for component in nx.connected_components(self):
            if len(component) == 1:
                # Don't need to check single node component
                continue
            num_nodes = len(component)
            num_half_edges = 0
            num_faces = 0
            for v in component:
                for w in self.neighbors_cw_order(v):
                    num_half_edges += 1
                    if (v, w) not in counted_half_edges:
                        # We encountered a new face
                        num_faces += 1
                        # Mark all half-edges belonging to this face
                        self.traverse_face(v, w, counted_half_edges)
            num_edges = num_half_edges // 2  # num_half_edges is even
            if num_nodes - num_edges + num_faces != 2:
                # The result does not match Euler's formula
                msg = "Bad embedding. The graph does not match Euler's formula"
                raise nx.NetworkXException(msg)

    def add_half_edge_ccw(self, start_node, end_node, reference_neighbor):
        """Adds a half-edge from start_node to end_node.

        The half-edge is added counter clockwise next to the existing half-edge
        (start_node, reference_neighbor).

        Parameters
        ----------
        start_node : node
            Start node of inserted edge.
        end_node : node
            End node of inserted edge.
        reference_neighbor: node
            End node of reference edge.

        Raises
        ------
        NetworkXException
            If the reference_neighbor does not exist.

        See Also
        --------
        add_half_edge
        add_half_edge_cw
        connect_components

        """
        self.add_half_edge(start_node, end_node, cw=reference_neighbor)

    def add_half_edge_cw(self, start_node, end_node, reference_neighbor):
        """Adds a half-edge from start_node to end_node.

        The half-edge is added clockwise next to the existing half-edge
        (start_node, reference_neighbor).

        Parameters
        ----------
        start_node : node
            Start node of inserted edge.
        end_node : node
            End node of inserted edge.
        reference_neighbor: node
            End node of reference edge.

        Raises
        ------
        NetworkXException
            If the reference_neighbor does not exist.

        See Also
        --------
        add_half_edge
        add_half_edge_ccw
        connect_components
        """
        self.add_half_edge(start_node, end_node, ccw=reference_neighbor)

    def remove_edge(self, u, v):
        """Remove the edge between u and v.

        Parameters
        ----------
        u, v : nodes
        Remove the half-edges (u, v) and (v, u) and update the
        edge ordering around the removed edge.

        Raises
        ------
        NetworkXError
        If there is not an edge between u and v.

        See Also
        --------
        remove_edges_from : remove a collection of edges
        """
        try:
            succs_u = self._succ[u]
            succs_v = self._succ[v]
            uv_cw = succs_u[v]["cw"]
            uv_ccw = succs_u[v]["ccw"]
            vu_cw = succs_v[u]["cw"]
            vu_ccw = succs_v[u]["ccw"]
            del succs_u[v]
            del self._pred[v][u]
            del succs_v[u]
            del self._pred[u][v]
            if v != uv_cw:
                succs_u[uv_cw]["ccw"] = uv_ccw
                succs_u[uv_ccw]["cw"] = uv_cw
            if u != vu_cw:
                succs_v[vu_cw]["ccw"] = vu_ccw
                succs_v[vu_ccw]["cw"] = vu_cw
        except KeyError as err:
            raise nx.NetworkXError(
                f"The edge {u}-{v} is not in the planar embedding."
            ) from err
        nx._clear_cache(self)

    def remove_edges_from(self, ebunch):
        """Remove all edges specified in ebunch.

        Parameters
        ----------
        ebunch: list or container of edge tuples
            Each pair of half-edges between the nodes given in the tuples
            will be removed from the graph. The nodes can be passed as:

                - 2-tuples (u, v) half-edges (u, v) and (v, u).
                - 3-tuples (u, v, k) where k is ignored.

        See Also
        --------
        remove_edge : remove a single edge

        Notes
        -----
        Will fail silently if an edge in ebunch is not in the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> ebunch = [(1, 2), (2, 3)]
        >>> G.remove_edges_from(ebunch)
        """
        for e in ebunch:
            u, v = e[:2]  # ignore edge data
            # assuming that the PlanarEmbedding is valid, if the half_edge
            # (u, v) is in the graph, then so is half_edge (v, u)
            if u in self._succ and v in self._succ[u]:
                self.remove_edge(u, v)

    def connect_components(self, v, w):
        """Adds half-edges for (v, w) and (w, v) at some position.

        This method should only be called if v and w are in different
        components, or it might break the embedding.
        This especially means that if `connect_components(v, w)`
        is called it is not allowed to call `connect_components(w, v)`
        afterwards. The neighbor orientations in both directions are
        all set correctly after the first call.

        Parameters
        ----------
        v : node
        w : node

        See Also
        --------
        add_half_edge
        """
        if v in self._succ and self._succ[v]:
            ref = next(reversed(self._succ[v]))
        else:
            ref = None
        self.add_half_edge(v, w, cw=ref)
        if w in self._succ and self._succ[w]:
            ref = next(reversed(self._succ[w]))
        else:
            ref = None
        self.add_half_edge(w, v, cw=ref)

    def add_half_edge_first(self, start_node, end_node):
        """Add a half-edge and set end_node as start_node's leftmost neighbor.

        The new edge is inserted counterclockwise with respect to the current
        leftmost neighbor, if there is one.

        Parameters
        ----------
        start_node : node
        end_node : node

        See Also
        --------
        add_half_edge
        connect_components
        """
        succs = self._succ.get(start_node)
        # the leftmost neighbor is the last entry in the
        # self._succ[start_node] dict
        leftmost_nbr = next(reversed(succs)) if succs else None
        self.add_half_edge(start_node, end_node, cw=leftmost_nbr)

    def next_face_half_edge(self, v, w):
        """Returns the following half-edge left of a face.

        Parameters
        ----------
        v : node
        w : node

        Returns
        -------
        half-edge : tuple
        """
        new_node = self[w][v]["ccw"]
        return w, new_node

    def traverse_face(self, v, w, mark_half_edges=None):
        """Returns nodes on the face that belong to the half-edge (v, w).

        The face that is traversed lies to the right of the half-edge (in an
        orientation where v is below w).

        Optionally it is possible to pass a set to which all encountered half
        edges are added. Before calling this method, this set must not include
        any half-edges that belong to the face.

        Parameters
        ----------
        v : node
            Start node of half-edge.
        w : node
            End node of half-edge.
        mark_half_edges: set, optional
            Set to which all encountered half-edges are added.

        Returns
        -------
        face : list
            A list of nodes that lie on this face.
        """
        if mark_half_edges is None:
            mark_half_edges = set()

        face_nodes = [v]
        mark_half_edges.add((v, w))
        prev_node = v
        cur_node = w
        # Last half-edge is (incoming_node, v)
        incoming_node = self[v][w]["cw"]

        while cur_node != v or prev_node != incoming_node:
            face_nodes.append(cur_node)
            prev_node, cur_node = self.next_face_half_edge(prev_node, cur_node)
            if (prev_node, cur_node) in mark_half_edges:
                raise nx.NetworkXException("Bad planar embedding. Impossible face.")
            mark_half_edges.add((prev_node, cur_node))

        return face_nodes

    def is_directed(self):
        """A valid PlanarEmbedding is undirected.

        All reverse edges are contained, i.e. for every existing
        half-edge (v, w) the half-edge in the opposite direction (w, v) is also
        contained.
        """
        return False

    def copy(self, as_view=False):
        if as_view is True:
            return nx.graphviews.generic_graph_view(self)
        G = self.__class__()
        G.graph.update(self.graph)
        G.add_nodes_from((n, d.copy()) for n, d in self._node.items())
        super(self.__class__, G).add_edges_from(
            (u, v, datadict.copy())
            for u, nbrs in self._adj.items()
            for v, datadict in nbrs.items()
        )
        return G
<|endoftext|>"""Functions related to graph covers."""

from functools import partial
from itertools import chain

import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for

__all__ = ["min_edge_cover", "is_edge_cover"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def min_edge_cover(G, matching_algorithm=None):
    """Returns the min cardinality edge cover of the graph as a set of edges.

    A smallest edge cover can be found in polynomial time by finding
    a maximum matching and extending it greedily so that all nodes
    are covered. This function follows that process. A maximum matching
    algorithm can be specified for the first step of the algorithm.
    The resulting set may return a set with one 2-tuple for each edge,
    (the usual case) or with both 2-tuples `(u, v)` and `(v, u)` for
    each edge. The latter is only done when a bipartite matching algorithm
    is specified as `matching_algorithm`.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    matching_algorithm : function
        A function that returns a maximum cardinality matching for `G`.
        The function must take one input, the graph `G`, and return
        either a set of edges (with only one direction for the pair of nodes)
        or a dictionary mapping each node to its mate. If not specified,
        :func:`~networkx.algorithms.matching.max_weight_matching` is used.
        Common bipartite matching functions include
        :func:`~networkx.algorithms.bipartite.matching.hopcroft_karp_matching`
        or
        :func:`~networkx.algorithms.bipartite.matching.eppstein_matching`.

    Returns
    -------
    min_cover : set

        A set of the edges in a minimum edge cover in the form of tuples.
        It contains only one of the equivalent 2-tuples `(u, v)` and `(v, u)`
        for each edge. If a bipartite method is used to compute the matching,
        the returned set contains both the 2-tuples `(u, v)` and `(v, u)`
        for each edge of a minimum edge cover.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    >>> sorted(nx.min_edge_cover(G))
    [(2, 1), (3, 0)]

    Notes
    -----
    An edge cover of a graph is a set of edges such that every node of
    the graph is incident to at least one edge of the set.
    The minimum edge cover is an edge covering of smallest cardinality.

    Due to its implementation, the worst-case running time of this algorithm
    is bounded by the worst-case running time of the function
    ``matching_algorithm``.

    Minimum edge cover for `G` can also be found using the `min_edge_covering`
    function in :mod:`networkx.algorithms.bipartite.covering` which is
    simply this function with a default matching algorithm of
    :func:`~networkx.algorithms.bipartite.matching.hopcraft_karp_matching`
    """
    if len(G) == 0:
        return set()
    if nx.number_of_isolates(G) > 0:
        # ``min_cover`` does not exist as there is an isolated node
        raise nx.NetworkXException(
            "Graph has a node with no edge incident on it, so no edge cover exists."
        )
    if matching_algorithm is None:
        matching_algorithm = partial(nx.max_weight_matching, maxcardinality=True)
    maximum_matching = matching_algorithm(G)
    # ``min_cover`` is superset of ``maximum_matching``
    try:
        # bipartite matching algs return dict so convert if needed
        min_cover = set(maximum_matching.items())
        bipartite_cover = True
    except AttributeError:
        min_cover = maximum_matching
        bipartite_cover = False
    # iterate for uncovered nodes
    uncovered_nodes = set(G) - {v for u, v in min_cover} - {u for u, v in min_cover}
    for v in uncovered_nodes:
        # Since `v` is uncovered, each edge incident to `v` will join it
        # with a covered node (otherwise, if there were an edge joining
        # uncovered nodes `u` and `v`, the maximum matching algorithm
        # would have found it), so we can choose an arbitrary edge
        # incident to `v`. (This applies only in a simple graph, not a
        # multigraph.)
        u = arbitrary_element(G[v])
        min_cover.add((u, v))
        if bipartite_cover:
            min_cover.add((v, u))
    return min_cover


@not_implemented_for("directed")
@nx._dispatchable
def is_edge_cover(G, cover):
    """Decides whether a set of edges is a valid edge cover of the graph.

    Given a set of edges, whether it is an edge covering can
    be decided if we just check whether all nodes of the graph
    has an edge from the set, incident on it.

    Parameters
    ----------
    G : NetworkX graph
        An undirected bipartite graph.

    cover : set
        Set of edges to be checked.

    Returns
    -------
    bool
        Whether the set of edges is a valid edge cover of the graph.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    >>> cover = {(2, 1), (3, 0)}
    >>> nx.is_edge_cover(G, cover)
    True

    Notes
    -----
    An edge cover of a graph is a set of edges such that every node of
    the graph is incident to at least one edge of the set.
    """
    return set(G) <= set(chain.from_iterable(cover))
<|endoftext|>"""Functions for computing rich-club coefficients."""

from itertools import accumulate

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["rich_club_coefficient"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def rich_club_coefficient(G, normalized=True, Q=100, seed=None):
    r"""Returns the rich-club coefficient of the graph `G`.

    For each degree *k*, the *rich-club coefficient* is the ratio of the
    number of actual to the number of potential edges for nodes with
    degree greater than *k*:

    .. math::

        \phi(k) = \frac{2 E_k}{N_k (N_k - 1)}

    where `N_k` is the number of nodes with degree larger than *k*, and
    `E_k` is the number of edges among those nodes.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph with neither parallel edges nor self-loops.
    normalized : bool (optional)
        Normalize using randomized network as in [1]_
    Q : float (optional, default=100)
        If `normalized` is True, perform `Q * m` double-edge
        swaps, where `m` is the number of edges in `G`, to use as a
        null-model for normalization.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    rc : dictionary
       A dictionary, keyed by degree, with rich-club coefficient values.

    Raises
    ------
    NetworkXError
        If `G` has fewer than four nodes and ``normalized=True``.
        A randomly sampled graph for normalization cannot be generated in this case.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (4, 5)])
    >>> rc = nx.rich_club_coefficient(G, normalized=False, seed=42)
    >>> rc[0]
    0.4

    Notes
    -----
    The rich club definition and algorithm are found in [1]_.  This
    algorithm ignores any edge weights and is not defined for directed
    graphs or graphs with parallel edges or self loops.

    Normalization is done by computing the rich club coefficient for a randomly
    sampled graph with the same degree distribution as `G` by
    repeatedly swapping the endpoints of existing edges. For graphs with fewer than 4
    nodes, it is not possible to generate a random graph with a prescribed
    degree distribution, as the degree distribution fully determines the graph
    (hence making the coefficients trivially normalized to 1).
    This function raises an exception in this case.

    Estimates for appropriate values of `Q` are found in [2]_.

    References
    ----------
    .. [1] Julian J. McAuley, Luciano da Fontoura Costa,
       and Tibério S. Caetano,
       "The rich-club phenomenon across complex network hierarchies",
       Applied Physics Letters Vol 91 Issue 8, August 2007.
       https://arxiv.org/abs/physics/0701290
    .. [2] R. Milo, N. Kashtan, S. Itzkovitz, M. E. J. Newman, U. Alon,
       "Uniform generation of random graphs with arbitrary degree
       sequences", 2006. https://arxiv.org/abs/cond-mat/0312028
    """
    if nx.number_of_selfloops(G) > 0:
        raise Exception(
            "rich_club_coefficient is not implemented for graphs with self loops."
        )
    rc = _compute_rc(G)
    if normalized:
        # make R a copy of G, randomize with Q*|E| double edge swaps
        # and use rich_club coefficient of R to normalize
        R = G.copy()
        E = R.number_of_edges()
        nx.double_edge_swap(R, Q * E, max_tries=Q * E * 10, seed=seed)
        rcran = _compute_rc(R)
        rc = {k: v / rcran[k] for k, v in rc.items()}
    return rc


def _compute_rc(G):
    """Returns the rich-club coefficient for each degree in the graph
    `G`.

    `G` is an undirected graph without multiedges.

    Returns a dictionary mapping degree to rich-club coefficient for
    that degree.

    """
    deghist = nx.degree_histogram(G)
    total = sum(deghist)
    # Compute the number of nodes with degree greater than `k`, for each
    # degree `k` (omitting the last entry, which is zero).
    nks = (total - cs for cs in accumulate(deghist) if total - cs > 1)
    # Create a sorted list of pairs of edge endpoint degrees.
    #
    # The list is sorted in reverse order so that we can pop from the
    # right side of the list later, instead of popping from the left
    # side of the list, which would have a linear time cost.
    edge_degrees = sorted((sorted(map(G.degree, e)) for e in G.edges()), reverse=True)
    ek = G.number_of_edges()
    if ek == 0:
        return {}

    k1, k2 = edge_degrees.pop()
    rc = {}
    for d, nk in enumerate(nks):
        while k1 <= d:
            if len(edge_degrees) == 0:
                ek = 0
                break
            k1, k2 = edge_degrees.pop()
            ek -= 1
        rc[d] = 2 * ek / (nk * (nk - 1))
    return rc
<|endoftext|>from networkx.algorithms.assortativity import *
from networkx.algorithms.asteroidal import *
from networkx.algorithms.boundary import *
from networkx.algorithms.broadcasting import *
from networkx.algorithms.bridges import *
from networkx.algorithms.chains import *
from networkx.algorithms.centrality import *
from networkx.algorithms.chordal import *
from networkx.algorithms.cluster import *
from networkx.algorithms.clique import *
from networkx.algorithms.communicability_alg import *
from networkx.algorithms.components import *
from networkx.algorithms.coloring import *
from networkx.algorithms.core import *
from networkx.algorithms.covering import *
from networkx.algorithms.cycles import *
from networkx.algorithms.cuts import *
from networkx.algorithms.d_separation import *
from networkx.algorithms.dag import *
from networkx.algorithms.distance_measures import *
from networkx.algorithms.distance_regular import *
from networkx.algorithms.dominance import *
from networkx.algorithms.dominating import *
from networkx.algorithms.efficiency_measures import *
from networkx.algorithms.euler import *
from networkx.algorithms.graphical import *
from networkx.algorithms.hierarchy import *
from networkx.algorithms.hybrid import *
from networkx.algorithms.link_analysis import *
from networkx.algorithms.link_prediction import *
from networkx.algorithms.lowest_common_ancestors import *
from networkx.algorithms.isolate import *
from networkx.algorithms.matching import *
from networkx.algorithms.minors import *
from networkx.algorithms.mis import *
from networkx.algorithms.moral import *
from networkx.algorithms.non_randomness import *
from networkx.algorithms.operators import *
from networkx.algorithms.planarity import *
from networkx.algorithms.planar_drawing import *
from networkx.algorithms.polynomials import *
from networkx.algorithms.reciprocity import *
from networkx.algorithms.regular import *
from networkx.algorithms.richclub import *
from networkx.algorithms.shortest_paths import *
from networkx.algorithms.similarity import *
from networkx.algorithms.graph_hashing import *
from networkx.algorithms.simple_paths import *
from networkx.algorithms.smallworld import *
from networkx.algorithms.smetric import *
from networkx.algorithms.structuralholes import *
from networkx.algorithms.sparsifiers import *
from networkx.algorithms.summarization import *
from networkx.algorithms.swap import *
from networkx.algorithms.time_dependent import *
from networkx.algorithms.traversal import *
from networkx.algorithms.triads import *
from networkx.algorithms.vitality import *
from networkx.algorithms.voronoi import *
from networkx.algorithms.walks import *
from networkx.algorithms.wiener import *

# Make certain subpackages available to the user as direct imports from
# the `networkx` namespace.
from networkx.algorithms import approximation
from networkx.algorithms import assortativity
from networkx.algorithms import bipartite
from networkx.algorithms import node_classification
from networkx.algorithms import centrality
from networkx.algorithms import chordal
from networkx.algorithms import cluster
from networkx.algorithms import clique
from networkx.algorithms import components
from networkx.algorithms import connectivity
from networkx.algorithms import community
from networkx.algorithms import coloring
from networkx.algorithms import flow
from networkx.algorithms import isomorphism
from networkx.algorithms import link_analysis
from networkx.algorithms import lowest_common_ancestors
from networkx.algorithms import operators
from networkx.algorithms import shortest_paths
from networkx.algorithms import tournament
from networkx.algorithms import traversal
from networkx.algorithms import tree

# Make certain functions from some of the previous subpackages available
# to the user as direct imports from the `networkx` namespace.
from networkx.algorithms.bipartite import complete_bipartite_graph
from networkx.algorithms.bipartite import is_bipartite
from networkx.algorithms.bipartite import projected_graph
from networkx.algorithms.connectivity import all_pairs_node_connectivity
from networkx.algorithms.connectivity import all_node_cuts
from networkx.algorithms.connectivity import average_node_connectivity
from networkx.algorithms.connectivity import edge_connectivity
from networkx.algorithms.connectivity import edge_disjoint_paths
from networkx.algorithms.connectivity import k_components
from networkx.algorithms.connectivity import k_edge_components
from networkx.algorithms.connectivity import k_edge_subgraphs
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity import is_k_edge_connected
from networkx.algorithms.connectivity import minimum_edge_cut
from networkx.algorithms.connectivity import minimum_node_cut
from networkx.algorithms.connectivity import node_connectivity
from networkx.algorithms.connectivity import node_disjoint_paths
from networkx.algorithms.connectivity import stoer_wagner
from networkx.algorithms.flow import capacity_scaling
from networkx.algorithms.flow import cost_of_flow
from networkx.algorithms.flow import gomory_hu_tree
from networkx.algorithms.flow import max_flow_min_cost
from networkx.algorithms.flow import maximum_flow
from networkx.algorithms.flow import maximum_flow_value
from networkx.algorithms.flow import min_cost_flow
from networkx.algorithms.flow import min_cost_flow_cost
from networkx.algorithms.flow import minimum_cut
from networkx.algorithms.flow import minimum_cut_value
from networkx.algorithms.flow import network_simplex
from networkx.algorithms.isomorphism import could_be_isomorphic
from networkx.algorithms.isomorphism import fast_could_be_isomorphic
from networkx.algorithms.isomorphism import faster_could_be_isomorphic
from networkx.algorithms.isomorphism import is_isomorphic
from networkx.algorithms.isomorphism.vf2pp import *
from networkx.algorithms.tree.branchings import maximum_branching
from networkx.algorithms.tree.branchings import maximum_spanning_arborescence
from networkx.algorithms.tree.branchings import minimum_branching
from networkx.algorithms.tree.branchings import minimum_spanning_arborescence
from networkx.algorithms.tree.branchings import ArborescenceIterator
from networkx.algorithms.tree.coding import *
from networkx.algorithms.tree.decomposition import *
from networkx.algorithms.tree.mst import *
from networkx.algorithms.tree.operations import *
from networkx.algorithms.tree.recognition import *
from networkx.algorithms.tournament import is_tournament
<|endoftext|>"""
Graph summarization finds smaller representations of graphs resulting in faster
runtime of algorithms, reduced storage needs, and noise reduction.
Summarization has applications in areas such as visualization, pattern mining,
clustering and community detection, and more.  Core graph summarization
techniques are grouping/aggregation, bit-compression,
simplification/sparsification, and influence based. Graph summarization
algorithms often produce either summary graphs in the form of supergraphs or
sparsified graphs, or a list of independent structures. Supergraphs are the
most common product, which consist of supernodes and original nodes and are
connected by edges and superedges, which represent aggregate edges between
nodes and supernodes.

Grouping/aggregation based techniques compress graphs by representing
close/connected nodes and edges in a graph by a single node/edge in a
supergraph. Nodes can be grouped together into supernodes based on their
structural similarities or proximity within a graph to reduce the total number
of nodes in a graph. Edge-grouping techniques group edges into lossy/lossless
nodes called compressor or virtual nodes to reduce the total number of edges in
a graph. Edge-grouping techniques can be lossless, meaning that they can be
used to re-create the original graph, or techniques can be lossy, requiring
less space to store the summary graph, but at the expense of lower
reconstruction accuracy of the original graph.

Bit-compression techniques minimize the amount of information needed to
describe the original graph, while revealing structural patterns in the
original graph.  The two-part minimum description length (MDL) is often used to
represent the model and the original graph in terms of the model.  A key
difference between graph compression and graph summarization is that graph
summarization focuses on finding structural patterns within the original graph,
whereas graph compression focuses on compressions the original graph to be as
small as possible.  **NOTE**: Some bit-compression methods exist solely to
compress a graph without creating a summary graph or finding comprehensible
structural patterns.

Simplification/Sparsification techniques attempt to create a sparse
representation of a graph by removing unimportant nodes and edges from the
graph.  Sparsified graphs differ from supergraphs created by
grouping/aggregation by only containing a subset of the original nodes and
edges of the original graph.

Influence based techniques aim to find a high-level description of influence
propagation in a large graph.  These methods are scarce and have been mostly
applied to social graphs.

*dedensification* is a grouping/aggregation based technique to compress the
neighborhoods around high-degree nodes in unweighted graphs by adding
compressor nodes that summarize multiple edges of the same type to
high-degree nodes (nodes with a degree greater than a given threshold).
Dedensification was developed for the purpose of increasing performance of
query processing around high-degree nodes in graph databases and enables direct
operations on the compressed graph.  The structural patterns surrounding
high-degree nodes in the original is preserved while using fewer edges and
adding a small number of compressor nodes.  The degree of nodes present in the
original graph is also preserved. The current implementation of dedensification
supports graphs with one edge type.

For more information on graph summarization, see `Graph Summarization Methods
and Applications: A Survey <https://dl.acm.org/doi/abs/10.1145/3186727>`_
"""

from collections import Counter, defaultdict

import networkx as nx

__all__ = ["dedensify", "snap_aggregation"]


@nx._dispatchable(mutates_input={"not copy": 3}, returns_graph=True)
def dedensify(G, threshold, prefix=None, copy=True):
    """Compresses neighborhoods around high-degree nodes

    Reduces the number of edges to high-degree nodes by adding compressor nodes
    that summarize multiple edges of the same type to high-degree nodes (nodes
    with a degree greater than a given threshold).  Dedensification also has
    the added benefit of reducing the number of edges around high-degree nodes.
    The implementation currently supports graphs with a single edge type.

    Parameters
    ----------
    G: graph
       A networkx graph
    threshold: int
       Minimum degree threshold of a node to be considered a high degree node.
       The threshold must be greater than or equal to 2.
    prefix: str or None, optional (default: None)
       An optional prefix for denoting compressor nodes
    copy: bool, optional (default: True)
       Indicates if dedensification should be done inplace

    Returns
    -------
    dedensified networkx graph : (graph, set)
        2-tuple of the dedensified graph and set of compressor nodes

    Notes
    -----
    According to the algorithm in [1]_, removes edges in a graph by
    compressing/decompressing the neighborhoods around high degree nodes by
    adding compressor nodes that summarize multiple edges of the same type
    to high-degree nodes.  Dedensification will only add a compressor node when
    doing so will reduce the total number of edges in the given graph. This
    implementation currently supports graphs with a single edge type.

    Examples
    --------
    Dedensification will only add compressor nodes when doing so would result
    in fewer edges::

        >>> original_graph = nx.DiGraph()
        >>> original_graph.add_nodes_from(
        ...     ["1", "2", "3", "4", "5", "6", "A", "B", "C"]
        ... )
        >>> original_graph.add_edges_from(
        ...     [
        ...         ("1", "C"), ("1", "B"),
        ...         ("2", "C"), ("2", "B"), ("2", "A"),
        ...         ("3", "B"), ("3", "A"), ("3", "6"),
        ...         ("4", "C"), ("4", "B"), ("4", "A"),
        ...         ("5", "B"), ("5", "A"),
        ...         ("6", "5"),
        ...         ("A", "6")
        ...     ]
        ... )
        >>> c_graph, c_nodes = nx.dedensify(original_graph, threshold=2)
        >>> original_graph.number_of_edges()
        15
        >>> c_graph.number_of_edges()
        14

    A dedensified, directed graph can be "densified" to reconstruct the
    original graph::

        >>> original_graph = nx.DiGraph()
        >>> original_graph.add_nodes_from(
        ...     ["1", "2", "3", "4", "5", "6", "A", "B", "C"]
        ... )
        >>> original_graph.add_edges_from(
        ...     [
        ...         ("1", "C"), ("1", "B"),
        ...         ("2", "C"), ("2", "B"), ("2", "A"),
        ...         ("3", "B"), ("3", "A"), ("3", "6"),
        ...         ("4", "C"), ("4", "B"), ("4", "A"),
        ...         ("5", "B"), ("5", "A"),
        ...         ("6", "5"),
        ...         ("A", "6")
        ...     ]
        ... )
        >>> c_graph, c_nodes = nx.dedensify(original_graph, threshold=2)
        >>> # re-densifies the compressed graph into the original graph
        >>> for c_node in c_nodes:
        ...     all_neighbors = set(nx.all_neighbors(c_graph, c_node))
        ...     out_neighbors = set(c_graph.neighbors(c_node))
        ...     for out_neighbor in out_neighbors:
        ...         c_graph.remove_edge(c_node, out_neighbor)
        ...     in_neighbors = all_neighbors - out_neighbors
        ...     for in_neighbor in in_neighbors:
        ...         c_graph.remove_edge(in_neighbor, c_node)
        ...         for out_neighbor in out_neighbors:
        ...             c_graph.add_edge(in_neighbor, out_neighbor)
        ...     c_graph.remove_node(c_node)
        ...
        >>> nx.is_isomorphic(original_graph, c_graph)
        True

    References
    ----------
    .. [1] Maccioni, A., & Abadi, D. J. (2016, August).
       Scalable pattern matching over compressed graphs via dedensification.
       In Proceedings of the 22nd ACM SIGKDD International Conference on
       Knowledge Discovery and Data Mining (pp. 1755-1764).
       http://www.cs.umd.edu/~abadi/papers/graph-dedense.pdf
    """
    if threshold < 2:
        raise nx.NetworkXError("The degree threshold must be >= 2")

    degrees = G.in_degree if G.is_directed() else G.degree
    # Group nodes based on degree threshold
    high_degree_nodes = {n for n, d in degrees if d > threshold}
    low_degree_nodes = G.nodes() - high_degree_nodes

    auxiliary = {}
    for node in G:
        high_degree_nbrs = frozenset(high_degree_nodes & set(G[node]))
        if high_degree_nbrs:
            if high_degree_nbrs in auxiliary:
                auxiliary[high_degree_nbrs].add(node)
            else:
                auxiliary[high_degree_nbrs] = {node}

    if copy:
        G = G.copy()

    compressor_nodes = set()
    for index, (high_degree_nodes, low_degree_nodes) in enumerate(auxiliary.items()):
        low_degree_node_count = len(low_degree_nodes)
        high_degree_node_count = len(high_degree_nodes)
        old_edges = high_degree_node_count * low_degree_node_count
        new_edges = high_degree_node_count + low_degree_node_count
        if old_edges <= new_edges:
            continue
        compression_node = "".join(str(node) for node in high_degree_nodes)
        if prefix:
            compression_node = str(prefix) + compression_node
        for node in low_degree_nodes:
            for high_node in high_degree_nodes:
                if G.has_edge(node, high_node):
                    G.remove_edge(node, high_node)

            G.add_edge(node, compression_node)
        for node in high_degree_nodes:
            G.add_edge(compression_node, node)
        compressor_nodes.add(compression_node)
    return G, compressor_nodes


def _snap_build_graph(
    G,
    groups,
    node_attributes,
    edge_attributes,
    neighbor_info,
    edge_types,
    prefix,
    supernode_attribute,
    superedge_attribute,
):
    """
    Build the summary graph from the data structures produced in the SNAP aggregation algorithm

    Used in the SNAP aggregation algorithm to build the output summary graph and supernode
    lookup dictionary.  This process uses the original graph and the data structures to
    create the supernodes with the correct node attributes, and the superedges with the correct
    edge attributes

    Parameters
    ----------
    G: networkx.Graph
        the original graph to be summarized
    groups: dict
        A dictionary of unique group IDs and their corresponding node groups
    node_attributes: iterable
        An iterable of the node attributes considered in the summarization process
    edge_attributes: iterable
        An iterable of the edge attributes considered in the summarization process
    neighbor_info: dict
        A data structure indicating the number of edges a node has with the
        groups in the current summarization of each edge type
    edge_types: dict
        dictionary of edges in the graph and their corresponding attributes recognized
        in the summarization
    prefix: string
        The prefix to be added to all supernodes
    supernode_attribute: str
        The node attribute for recording the supernode groupings of nodes
    superedge_attribute: str
        The edge attribute for recording the edge types represented by superedges

    Returns
    -------
    summary graph: Networkx graph
    """
    output = G.__class__()
    node_label_lookup = {}
    for index, group_id in enumerate(groups):
        group_set = groups[group_id]
        supernode = f"{prefix}{index}"
        node_label_lookup[group_id] = supernode
        supernode_attributes = {
            attr: G.nodes[next(iter(group_set))][attr] for attr in node_attributes
        }
        supernode_attributes[supernode_attribute] = group_set
        output.add_node(supernode, **supernode_attributes)

    for group_id in groups:
        group_set = groups[group_id]
        source_supernode = node_label_lookup[group_id]
        for other_group, group_edge_types in neighbor_info[
            next(iter(group_set))
        ].items():
            if group_edge_types:
                target_supernode = node_label_lookup[other_group]
                summary_graph_edge = (source_supernode, target_supernode)

                edge_types = [
                    dict(zip(edge_attributes, edge_type))
                    for edge_type in group_edge_types
                ]

                has_edge = output.has_edge(*summary_graph_edge)
                if output.is_multigraph():
                    if not has_edge:
                        for edge_type in edge_types:
                            output.add_edge(*summary_graph_edge, **edge_type)
                    elif not output.is_directed():
                        existing_edge_data = output.get_edge_data(*summary_graph_edge)
                        for edge_type in edge_types:
                            if edge_type not in existing_edge_data.values():
                                output.add_edge(*summary_graph_edge, **edge_type)
                else:
                    superedge_attributes = {superedge_attribute: edge_types}
                    output.add_edge(*summary_graph_edge, **superedge_attributes)

    return output


def _snap_eligible_group(G, groups, group_lookup, edge_types):
    """
    Determines if a group is eligible to be split.

    A group is eligible to be split if all nodes in the group have edges of the same type(s)
    with the same other groups.

    Parameters
    ----------
    G: graph
        graph to be summarized
    groups: dict
        A dictionary of unique group IDs and their corresponding node groups
    group_lookup: dict
        dictionary of nodes and their current corresponding group ID
    edge_types: dict
        dictionary of edges in the graph and their corresponding attributes recognized
        in the summarization

    Returns
    -------
    tuple: group ID to split, and neighbor-groups participation_counts data structure
    """
    nbr_info = {node: {gid: Counter() for gid in groups} for node in group_lookup}
    for group_id in groups:
        current_group = groups[group_id]

        # build nbr_info for nodes in group
        for node in current_group:
            nbr_info[node] = {group_id: Counter() for group_id in groups}
            edges = G.edges(node, keys=True) if G.is_multigraph() else G.edges(node)
            for edge in edges:
                neighbor = edge[1]
                edge_type = edge_types[edge]
                neighbor_group_id = group_lookup[neighbor]
                nbr_info[node][neighbor_group_id][edge_type] += 1

        # check if group_id is eligible to be split
        group_size = len(current_group)
        for other_group_id in groups:
            edge_counts = Counter()
            for node in current_group:
                edge_counts.update(nbr_info[node][other_group_id].keys())

            if not all(count == group_size for count in edge_counts.values()):
                # only the nbr_info of the returned group_id is required for handling group splits
                return group_id, nbr_info

    # if no eligible groups, complete nbr_info is calculated
    return None, nbr_info


def _snap_split(groups, neighbor_info, group_lookup, group_id):
    """
    Splits a group based on edge types and updates the groups accordingly

    Splits the group with the given group_id based on the edge types
    of the nodes so that each new grouping will all have the same
    edges with other nodes.

    Parameters
    ----------
    groups: dict
        A dictionary of unique group IDs and their corresponding node groups
    neighbor_info: dict
        A data structure indicating the number of edges a node has with the
        groups in the current summarization of each edge type
    edge_types: dict
        dictionary of edges in the graph and their corresponding attributes recognized
        in the summarization
    group_lookup: dict
        dictionary of nodes and their current corresponding group ID
    group_id: object
        ID of group to be split

    Returns
    -------
    dict
        The updated groups based on the split
    """
    new_group_mappings = defaultdict(set)
    for node in groups[group_id]:
        signature = tuple(
            frozenset(edge_types) for edge_types in neighbor_info[node].values()
        )
        new_group_mappings[signature].add(node)

    # leave the biggest new_group as the original group
    new_groups = sorted(new_group_mappings.values(), key=len)
    for new_group in new_groups[:-1]:
        # Assign unused integer as the new_group_id
        # ids are tuples, so will not interact with the original group_ids
        new_group_id = len(groups)
        groups[new_group_id] = new_group
        groups[group_id] -= new_group
        for node in new_group:
            group_lookup[node] = new_group_id

    return groups


@nx._dispatchable(
    node_attrs="[node_attributes]", edge_attrs="[edge_attributes]", returns_graph=True
)
def snap_aggregation(
    G,
    node_attributes,
    edge_attributes=(),
    prefix="Supernode-",
    supernode_attribute="group",
    superedge_attribute="types",
):
    """Creates a summary graph based on attributes and connectivity.

    This function uses the Summarization by Grouping Nodes on Attributes
    and Pairwise edges (SNAP) algorithm for summarizing a given
    graph by grouping nodes by node attributes and their edge attributes
    into supernodes in a summary graph.  This name SNAP should not be
    confused with the Stanford Network Analysis Project (SNAP).

    Here is a high-level view of how this algorithm works:

    1) Group nodes by node attribute values.

    2) Iteratively split groups until all nodes in each group have edges
    to nodes in the same groups. That is, until all the groups are homogeneous
    in their member nodes' edges to other groups.  For example,
    if all the nodes in group A only have edge to nodes in group B, then the
    group is homogeneous and does not need to be split. If all nodes in group B
    have edges with nodes in groups {A, C}, but some also have edges with other
    nodes in B, then group B is not homogeneous and needs to be split into
    groups have edges with {A, C} and a group of nodes having
    edges with {A, B, C}.  This way, viewers of the summary graph can
    assume that all nodes in the group have the exact same node attributes and
    the exact same edges.

    3) Build the output summary graph, where the groups are represented by
    super-nodes. Edges represent the edges shared between all the nodes in each
    respective groups.

    A SNAP summary graph can be used to visualize graphs that are too large to display
    or visually analyze, or to efficiently identify sets of similar nodes with similar connectivity
    patterns to other sets of similar nodes based on specified node and/or edge attributes in a graph.

    Parameters
    ----------
    G: graph
        Networkx Graph to be summarized
    node_attributes: iterable, required
        An iterable of the node attributes used to group nodes in the summarization process. Nodes
        with the same values for these attributes will be grouped together in the summary graph.
    edge_attributes: iterable, optional
        An iterable of the edge attributes considered in the summarization process.  If provided, unique
        combinations of the attribute values found in the graph are used to
        determine the edge types in the graph.  If not provided, all edges
        are considered to be of the same type.
    prefix: str
        The prefix used to denote supernodes in the summary graph. Defaults to 'Supernode-'.
    supernode_attribute: str
        The node attribute for recording the supernode groupings of nodes. Defaults to 'group'.
    superedge_attribute: str
        The edge attribute for recording the edge types of multiple edges. Defaults to 'types'.

    Returns
    -------
    networkx.Graph: summary graph

    Examples
    --------
    SNAP aggregation takes a graph and summarizes it in the context of user-provided
    node and edge attributes such that a viewer can more easily extract and
    analyze the information represented by the graph

    >>> nodes = {
    ...     "A": dict(color="Red"),
    ...     "B": dict(color="Red"),
    ...     "C": dict(color="Red"),
    ...     "D": dict(color="Red"),
    ...     "E": dict(color="Blue"),
    ...     "F": dict(color="Blue"),
    ... }
    >>> edges = [
    ...     ("A", "E", "Strong"),
    ...     ("B", "F", "Strong"),
    ...     ("C", "E", "Weak"),
    ...     ("D", "F", "Weak"),
    ... ]
    >>> G = nx.Graph()
    >>> for node in nodes:
    ...     attributes = nodes[node]
    ...     G.add_node(node, **attributes)
    >>> for source, target, type in edges:
    ...     G.add_edge(source, target, type=type)
    >>> node_attributes = ("color",)
    >>> edge_attributes = ("type",)
    >>> summary_graph = nx.snap_aggregation(
    ...     G, node_attributes=node_attributes, edge_attributes=edge_attributes
    ... )

    Notes
    -----
    The summary graph produced is called a maximum Attribute-edge
    compatible (AR-compatible) grouping.  According to [1]_, an
    AR-compatible grouping means that all nodes in each group have the same
    exact node attribute values and the same exact edges and
    edge types to one or more nodes in the same groups.  The maximal
    AR-compatible grouping is the grouping with the minimal cardinality.

    The AR-compatible grouping is the most detailed grouping provided by
    any of the SNAP algorithms.

    References
    ----------
    .. [1] Y. Tian, R. A. Hankins, and J. M. Patel. Efficient aggregation
       for graph summarization. In Proc. 2008 ACM-SIGMOD Int. Conf.
       Management of Data (SIGMOD’08), pages 567–580, Vancouver, Canada,
       June 2008.
    """
    edge_types = {
        edge: tuple(attrs.get(attr) for attr in edge_attributes)
        for edge, attrs in G.edges.items()
    }
    if not G.is_directed():
        if G.is_multigraph():
            # list is needed to avoid mutating while iterating
            edges = [((v, u, k), etype) for (u, v, k), etype in edge_types.items()]
        else:
            # list is needed to avoid mutating while iterating
            edges = [((v, u), etype) for (u, v), etype in edge_types.items()]
        edge_types.update(edges)

    group_lookup = {
        node: tuple(attrs[attr] for attr in node_attributes)
        for node, attrs in G.nodes.items()
    }
    groups = defaultdict(set)
    for node, node_type in group_lookup.items():
        groups[node_type].add(node)

    eligible_group_id, nbr_info = _snap_eligible_group(
        G, groups, group_lookup, edge_types
    )
    while eligible_group_id:
        groups = _snap_split(groups, nbr_info, group_lookup, eligible_group_id)
        eligible_group_id, nbr_info = _snap_eligible_group(
            G, groups, group_lookup, edge_types
        )
    return _snap_build_graph(
        G,
        groups,
        node_attributes,
        edge_attributes,
        nbr_info,
        edge_types,
        prefix,
        supernode_attribute,
        superedge_attribute,
    )
<|endoftext|>"""
Find the k-cores of a graph.

The k-core is found by recursively pruning nodes with degrees less than k.

See the following references for details:

An O(m) Algorithm for Cores Decomposition of Networks
Vladimir Batagelj and Matjaz Zaversnik, 2003.
https://arxiv.org/abs/cs.DS/0310049

Generalized Cores
Vladimir Batagelj and Matjaz Zaversnik, 2002.
https://arxiv.org/pdf/cs/0202039

For directed graphs a more general notion is that of D-cores which
looks at (k, l) restrictions on (in, out) degree. The (k, k) D-core
is the k-core.

D-cores: Measuring Collaboration of Directed Graphs Based on Degeneracy
Christos Giatsidis, Dimitrios M. Thilikos, Michalis Vazirgiannis, ICDM 2011.
http://www.graphdegeneracy.org/dcores_ICDM_2011.pdf

Multi-scale structure and topological anomaly detection via a new network \
statistic: The onion decomposition
L. Hébert-Dufresne, J. A. Grochow, and A. Allard
Scientific Reports 6, 31708 (2016)
http://doi.org/10.1038/srep31708

"""

import networkx as nx

__all__ = [
    "core_number",
    "k_core",
    "k_shell",
    "k_crust",
    "k_corona",
    "k_truss",
    "onion_layers",
]


@nx.utils.not_implemented_for("multigraph")
@nx._dispatchable
def core_number(G):
    """Returns the core number for each node.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    The core number of a node is the largest value k of a k-core containing
    that node.

    Parameters
    ----------
    G : NetworkX graph
       An undirected or directed graph

    Returns
    -------
    core_number : dictionary
       A dictionary keyed by node to the core number.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a multigraph or contains self loops.

    Notes
    -----
    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Examples
    --------
    >>> degrees = [0, 1, 2, 2, 2, 2, 3]
    >>> H = nx.havel_hakimi_graph(degrees)
    >>> nx.core_number(H)
    {0: 1, 1: 2, 2: 2, 3: 2, 4: 1, 5: 2, 6: 0}
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (2, 1), (2, 3), (2, 4), (3, 4), (4, 3)])
    >>> nx.core_number(G)
    {1: 2, 2: 2, 3: 2, 4: 2}

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik, 2003.
       https://arxiv.org/abs/cs.DS/0310049
    """
    if nx.number_of_selfloops(G) > 0:
        msg = (
            "Input graph has self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
        raise nx.NetworkXNotImplemented(msg)
    degrees = dict(G.degree())
    # Sort nodes by degree.
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    # The initial guess for the core number of a node is its degree.
    core = degrees
    nbrs = {v: list(nx.all_neighbors(G, v)) for v in G}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return core


def _core_subgraph(G, k_filter, k=None, core=None):
    """Returns the subgraph induced by nodes passing filter `k_filter`.

    Parameters
    ----------
    G : NetworkX graph
       The graph or directed graph to process
    k_filter : filter function
       This function filters the nodes chosen. It takes three inputs:
       A node of G, the filter's cutoff, and the core dict of the graph.
       The function should return a Boolean value.
    k : int, optional
      The order of the core. If not specified use the max core number.
      This value is used as the cutoff for the filter.
    core : dict, optional
      Precomputed core numbers keyed by node for the graph `G`.
      If not specified, the core numbers will be computed from `G`.

    """
    if core is None:
        core = core_number(G)
    if k is None:
        k = max(core.values())
    nodes = (v for v in core if k_filter(v, k, core))
    return G.subgraph(nodes).copy()


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def k_core(G, k=None, core_number=None):
    """Returns the k-core of G.

    A k-core is a maximal subgraph that contains nodes of degree `k` or more.

    .. deprecated:: 3.3
       `k_core` will not accept `MultiGraph` objects in version 3.5.

    Parameters
    ----------
    G : NetworkX graph
      A graph or directed graph
    k : int, optional
      The order of the core. If not specified return the main core.
    core_number : dictionary, optional
      Precomputed core numbers for the graph G.

    Returns
    -------
    G : NetworkX graph
      The k-core subgraph

    Raises
    ------
    NetworkXNotImplemented
      The k-core is not defined for multigraphs or graphs with self loops.

    Notes
    -----
    The main core is the core with `k` as the largest core_number.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Graph, node, and edge attributes are copied to the subgraph.

    Examples
    --------
    >>> degrees = [0, 1, 2, 2, 2, 2, 3]
    >>> H = nx.havel_hakimi_graph(degrees)
    >>> H.degree
    DegreeView({0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 0})
    >>> nx.k_core(H).nodes
    NodeView((1, 2, 3, 5))

    See Also
    --------
    core_number

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik,  2003.
       https://arxiv.org/abs/cs.DS/0310049
    """

    import warnings

    if G.is_multigraph():
        warnings.warn(
            (
                "\n\n`k_core` will not accept `MultiGraph` objects in version 3.5.\n"
                "Convert it to an undirected graph instead, using::\n\n"
                "\tG = nx.Graph(G)\n"
            ),
            category=DeprecationWarning,
            stacklevel=5,
        )

    def k_filter(v, k, c):
        return c[v] >= k

    return _core_subgraph(G, k_filter, k, core_number)


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def k_shell(G, k=None, core_number=None):
    """Returns the k-shell of G.

    The k-shell is the subgraph induced by nodes with core number k.
    That is, nodes in the k-core that are not in the (k+1)-core.

    .. deprecated:: 3.3
       `k_shell` will not accept `MultiGraph` objects in version 3.5.

    Parameters
    ----------
    G : NetworkX graph
      A graph or directed graph.
    k : int, optional
      The order of the shell. If not specified return the outer shell.
    core_number : dictionary, optional
      Precomputed core numbers for the graph G.


    Returns
    -------
    G : NetworkX graph
       The k-shell subgraph

    Raises
    ------
    NetworkXNotImplemented
        The k-shell is not implemented for multigraphs or graphs with self loops.

    Notes
    -----
    This is similar to k_corona but in that case only neighbors in the
    k-core are considered.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Graph, node, and edge attributes are copied to the subgraph.

    Examples
    --------
    >>> degrees = [0, 1, 2, 2, 2, 2, 3]
    >>> H = nx.havel_hakimi_graph(degrees)
    >>> H.degree
    DegreeView({0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 0})
    >>> nx.k_shell(H, k=1).nodes
    NodeView((0, 4))

    See Also
    --------
    core_number
    k_corona


    References
    ----------
    .. [1] A model of Internet topology using k-shell decomposition
       Shai Carmi, Shlomo Havlin, Scott Kirkpatrick, Yuval Shavitt,
       and Eran Shir, PNAS  July 3, 2007   vol. 104  no. 27  11150-11154
       http://www.pnas.org/content/104/27/11150.full
    """

    import warnings

    if G.is_multigraph():
        warnings.warn(
            (
                "\n\n`k_shell` will not accept `MultiGraph` objects in version 3.5.\n"
                "Convert it to an undirected graph instead, using::\n\n"
                "\tG = nx.Graph(G)\n"
            ),
            category=DeprecationWarning,
            stacklevel=5,
        )

    def k_filter(v, k, c):
        return c[v] == k

    return _core_subgraph(G, k_filter, k, core_number)


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def k_crust(G, k=None, core_number=None):
    """Returns the k-crust of G.

    The k-crust is the graph G with the edges of the k-core removed
    and isolated nodes found after the removal of edges are also removed.

    .. deprecated:: 3.3
       `k_crust` will not accept `MultiGraph` objects in version 3.5.

    Parameters
    ----------
    G : NetworkX graph
       A graph or directed graph.
    k : int, optional
      The order of the shell. If not specified return the main crust.
    core_number : dictionary, optional
      Precomputed core numbers for the graph G.

    Returns
    -------
    G : NetworkX graph
       The k-crust subgraph

    Raises
    ------
    NetworkXNotImplemented
        The k-crust is not implemented for multigraphs or graphs with self loops.

    Notes
    -----
    This definition of k-crust is different than the definition in [1]_.
    The k-crust in [1]_ is equivalent to the k+1 crust of this algorithm.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Graph, node, and edge attributes are copied to the subgraph.

    Examples
    --------
    >>> degrees = [0, 1, 2, 2, 2, 2, 3]
    >>> H = nx.havel_hakimi_graph(degrees)
    >>> H.degree
    DegreeView({0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 0})
    >>> nx.k_crust(H, k=1).nodes
    NodeView((0, 4, 6))

    See Also
    --------
    core_number

    References
    ----------
    .. [1] A model of Internet topology using k-shell decomposition
       Shai Carmi, Shlomo Havlin, Scott Kirkpatrick, Yuval Shavitt,
       and Eran Shir, PNAS  July 3, 2007   vol. 104  no. 27  11150-11154
       http://www.pnas.org/content/104/27/11150.full
    """

    import warnings

    if G.is_multigraph():
        warnings.warn(
            (
                "\n\n`k_crust` will not accept `MultiGraph` objects in version 3.5.\n"
                "Convert it to an undirected graph instead, using::\n\n"
                "\tG = nx.Graph(G)\n"
            ),
            category=DeprecationWarning,
            stacklevel=5,
        )

    # Default for k is one less than in _core_subgraph, so just inline.
    #    Filter is c[v] <= k
    if core_number is None:
        core_number = nx.core_number(G)
    if k is None:
        k = max(core_number.values()) - 1
    nodes = (v for v in core_number if core_number[v] <= k)
    return G.subgraph(nodes).copy()


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def k_corona(G, k, core_number=None):
    """Returns the k-corona of G.

    The k-corona is the subgraph of nodes in the k-core which have
    exactly k neighbors in the k-core.

    .. deprecated:: 3.3
       `k_corona` will not accept `MultiGraph` objects in version 3.5.

    Parameters
    ----------
    G : NetworkX graph
       A graph or directed graph
    k : int
       The order of the corona.
    core_number : dictionary, optional
       Precomputed core numbers for the graph G.

    Returns
    -------
    G : NetworkX graph
       The k-corona subgraph

    Raises
    ------
    NetworkXNotImplemented
        The k-corona is not defined for multigraphs or graphs with self loops.

    Notes
    -----
    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Graph, node, and edge attributes are copied to the subgraph.

    Examples
    --------
    >>> degrees = [0, 1, 2, 2, 2, 2, 3]
    >>> H = nx.havel_hakimi_graph(degrees)
    >>> H.degree
    DegreeView({0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 0})
    >>> nx.k_corona(H, k=2).nodes
    NodeView((1, 2, 3, 5))

    See Also
    --------
    core_number

    References
    ----------
    .. [1]  k -core (bootstrap) percolation on complex networks:
       Critical phenomena and nonlocal effects,
       A. V. Goltsev, S. N. Dorogovtsev, and J. F. F. Mendes,
       Phys. Rev. E 73, 056101 (2006)
       http://link.aps.org/doi/10.1103/PhysRevE.73.056101
    """

    import warnings

    if G.is_multigraph():
        warnings.warn(
            (
                "\n\n`k_corona` will not accept `MultiGraph` objects in version 3.5.\n"
                "Convert it to an undirected graph instead, using::\n\n"
                "\tG = nx.Graph(G)\n"
            ),
            category=DeprecationWarning,
            stacklevel=5,
        )

    def func(v, k, c):
        return c[v] == k and k == sum(1 for w in G[v] if c[w] >= k)

    return _core_subgraph(G, func, k, core_number)


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def k_truss(G, k):
    """Returns the k-truss of `G`.

    The k-truss is the maximal induced subgraph of `G` which contains at least
    three vertices where every edge is incident to at least `k-2` triangles.

    Parameters
    ----------
    G : NetworkX graph
      An undirected graph
    k : int
      The order of the truss

    Returns
    -------
    H : NetworkX graph
      The k-truss subgraph

    Raises
    ------
    NetworkXNotImplemented
      If `G` is a multigraph or directed graph or if it contains self loops.

    Notes
    -----
    A k-clique is a (k-2)-truss and a k-truss is a (k+1)-core.

    Graph, node, and edge attributes are copied to the subgraph.

    K-trusses were originally defined in [2] which states that the k-truss
    is the maximal induced subgraph where each edge belongs to at least
    `k-2` triangles. A more recent paper, [1], uses a slightly different
    definition requiring that each edge belong to at least `k` triangles.
    This implementation uses the original definition of `k-2` triangles.

    Examples
    --------
    >>> degrees = [0, 1, 2, 2, 2, 2, 3]
    >>> H = nx.havel_hakimi_graph(degrees)
    >>> H.degree
    DegreeView({0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 0})
    >>> nx.k_truss(H, k=2).nodes
    NodeView((0, 1, 2, 3, 4, 5))

    References
    ----------
    .. [1] Bounds and Algorithms for k-truss. Paul Burkhardt, Vance Faber,
       David G. Harris, 2018. https://arxiv.org/abs/1806.05523v2
    .. [2] Trusses: Cohesive Subgraphs for Social Network Analysis. Jonathan
       Cohen, 2005.
    """
    if nx.number_of_selfloops(G) > 0:
        msg = (
            "Input graph has self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
        raise nx.NetworkXNotImplemented(msg)

    H = G.copy()

    n_dropped = 1
    while n_dropped > 0:
        n_dropped = 0
        to_drop = []
        seen = set()
        for u in H:
            nbrs_u = set(H[u])
            seen.add(u)
            new_nbrs = [v for v in nbrs_u if v not in seen]
            for v in new_nbrs:
                if len(nbrs_u & set(H[v])) < (k - 2):
                    to_drop.append((u, v))
        H.remove_edges_from(to_drop)
        n_dropped = len(to_drop)
        H.remove_nodes_from(list(nx.isolates(H)))

    return H


@nx.utils.not_implemented_for("multigraph")
@nx.utils.not_implemented_for("directed")
@nx._dispatchable
def onion_layers(G):
    """Returns the layer of each vertex in an onion decomposition of the graph.

    The onion decomposition refines the k-core decomposition by providing
    information on the internal organization of each k-shell. It is usually
    used alongside the `core numbers`.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph without self loops.

    Returns
    -------
    od_layers : dictionary
        A dictionary keyed by node to the onion layer. The layers are
        contiguous integers starting at 1.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a multigraph or directed graph or if it contains self loops.

    Examples
    --------
    >>> degrees = [0, 1, 2, 2, 2, 2, 3]
    >>> H = nx.havel_hakimi_graph(degrees)
    >>> H.degree
    DegreeView({0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 0})
    >>> nx.onion_layers(H)
    {6: 1, 0: 2, 4: 3, 1: 4, 2: 4, 3: 4, 5: 4}

    See Also
    --------
    core_number

    References
    ----------
    .. [1] Multi-scale structure and topological anomaly detection via a new
       network statistic: The onion decomposition
       L. Hébert-Dufresne, J. A. Grochow, and A. Allard
       Scientific Reports 6, 31708 (2016)
       http://doi.org/10.1038/srep31708
    .. [2] Percolation and the effective structure of complex networks
       A. Allard and L. Hébert-Dufresne
       Physical Review X 9, 011023 (2019)
       http://doi.org/10.1103/PhysRevX.9.011023
    """
    if nx.number_of_selfloops(G) > 0:
        msg = (
            "Input graph contains self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
        raise nx.NetworkXNotImplemented(msg)
    # Dictionaries to register the k-core/onion decompositions.
    od_layers = {}
    # Adjacency list
    neighbors = {v: list(nx.all_neighbors(G, v)) for v in G}
    # Effective degree of nodes.
    degrees = dict(G.degree())
    # Performs the onion decomposition.
    current_core = 1
    current_layer = 1
    # Sets vertices of degree 0 to layer 1, if any.
    isolated_nodes = list(nx.isolates(G))
    if len(isolated_nodes) > 0:
        for v in isolated_nodes:
            od_layers[v] = current_layer
            degrees.pop(v)
        current_layer = 2
    # Finds the layer for the remaining nodes.
    while len(degrees) > 0:
        # Sets the order for looking at nodes.
        nodes = sorted(degrees, key=degrees.get)
        # Sets properly the current core.
        min_degree = degrees[nodes[0]]
        if min_degree > current_core:
            current_core = min_degree
        # Identifies vertices in the current layer.
        this_layer = []
        for n in nodes:
            if degrees[n] > current_core:
                break
            this_layer.append(n)
        # Identifies the core/layer of the vertices in the current layer.
        for v in this_layer:
            od_layers[v] = current_layer
            for n in neighbors[v]:
                neighbors[n].remove(v)
                degrees[n] = degrees[n] - 1
            degrees.pop(v)
        # Updates the layer count.
        current_layer = current_layer + 1
    # Returns the dictionaries containing the onion layer of each vertices.
    return od_layers
<|endoftext|>"""Provides algorithms supporting the computation of graph polynomials.

Graph polynomials are polynomial-valued graph invariants that encode a wide
variety of structural information. Examples include the Tutte polynomial,
chromatic polynomial, characteristic polynomial, and matching polynomial. An
extensive treatment is provided in [1]_.

For a simple example, the `~sympy.matrices.matrices.MatrixDeterminant.charpoly`
method can be used to compute the characteristic polynomial from the adjacency
matrix of a graph. Consider the complete graph ``K_4``:

>>> import sympy
>>> x = sympy.Symbol("x")
>>> G = nx.complete_graph(4)
>>> A = nx.to_numpy_array(G, dtype=int)
>>> M = sympy.SparseMatrix(A)
>>> M.charpoly(x).as_expr()
x**4 - 6*x**2 - 8*x - 3


.. [1] Y. Shi, M. Dehmer, X. Li, I. Gutman,
   "Graph Polynomials"
"""

from collections import deque

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["tutte_polynomial", "chromatic_polynomial"]


@not_implemented_for("directed")
@nx._dispatchable
def tutte_polynomial(G):
    r"""Returns the Tutte polynomial of `G`

    This function computes the Tutte polynomial via an iterative version of
    the deletion-contraction algorithm.

    The Tutte polynomial `T_G(x, y)` is a fundamental graph polynomial invariant in
    two variables. It encodes a wide array of information related to the
    edge-connectivity of a graph; "Many problems about graphs can be reduced to
    problems of finding and evaluating the Tutte polynomial at certain values" [1]_.
    In fact, every deletion-contraction-expressible feature of a graph is a
    specialization of the Tutte polynomial [2]_ (see Notes for examples).

    There are several equivalent definitions; here are three:

    Def 1 (rank-nullity expansion): For `G` an undirected graph, `n(G)` the
    number of vertices of `G`, `E` the edge set of `G`, `V` the vertex set of
    `G`, and `c(A)` the number of connected components of the graph with vertex
    set `V` and edge set `A` [3]_:

    .. math::

        T_G(x, y) = \sum_{A \in E} (x-1)^{c(A) - c(E)} (y-1)^{c(A) + |A| - n(G)}

    Def 2 (spanning tree expansion): Let `G` be an undirected graph, `T` a spanning
    tree of `G`, and `E` the edge set of `G`. Let `E` have an arbitrary strict
    linear order `L`. Let `B_e` be the unique minimal nonempty edge cut of
    $E \setminus T \cup {e}$. An edge `e` is internally active with respect to
    `T` and `L` if `e` is the least edge in `B_e` according to the linear order
    `L`. The internal activity of `T` (denoted `i(T)`) is the number of edges
    in $E \setminus T$ that are internally active with respect to `T` and `L`.
    Let `P_e` be the unique path in $T \cup {e}$ whose source and target vertex
    are the same. An edge `e` is externally active with respect to `T` and `L`
    if `e` is the least edge in `P_e` according to the linear order `L`. The
    external activity of `T` (denoted `e(T)`) is the number of edges in
    $E \setminus T$ that are externally active with respect to `T` and `L`.
    Then [4]_ [5]_:

    .. math::

        T_G(x, y) = \sum_{T \text{ a spanning tree of } G} x^{i(T)} y^{e(T)}

    Def 3 (deletion-contraction recurrence): For `G` an undirected graph, `G-e`
    the graph obtained from `G` by deleting edge `e`, `G/e` the graph obtained
    from `G` by contracting edge `e`, `k(G)` the number of cut-edges of `G`,
    and `l(G)` the number of self-loops of `G`:

    .. math::
        T_G(x, y) = \begin{cases}
    	   x^{k(G)} y^{l(G)}, & \text{if all edges are cut-edges or self-loops} \\
           T_{G-e}(x, y) + T_{G/e}(x, y), & \text{otherwise, for an arbitrary edge $e$ not a cut-edge or loop}
        \end{cases}

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    instance of `sympy.core.add.Add`
        A Sympy expression representing the Tutte polynomial for `G`.

    Examples
    --------
    >>> C = nx.cycle_graph(5)
    >>> nx.tutte_polynomial(C)
    x**4 + x**3 + x**2 + x + y

    >>> D = nx.diamond_graph()
    >>> nx.tutte_polynomial(D)
    x**3 + 2*x**2 + 2*x*y + x + y**2 + y

    Notes
    -----
    Some specializations of the Tutte polynomial:

    - `T_G(1, 1)` counts the number of spanning trees of `G`
    - `T_G(1, 2)` counts the number of connected spanning subgraphs of `G`
    - `T_G(2, 1)` counts the number of spanning forests in `G`
    - `T_G(0, 2)` counts the number of strong orientations of `G`
    - `T_G(2, 0)` counts the number of acyclic orientations of `G`

    Edge contraction is defined and deletion-contraction is introduced in [6]_.
    Combinatorial meaning of the coefficients is introduced in [7]_.
    Universality, properties, and applications are discussed in [8]_.

    Practically, up-front computation of the Tutte polynomial may be useful when
    users wish to repeatedly calculate edge-connectivity-related information
    about one or more graphs.

    References
    ----------
    .. [1] M. Brandt,
       "The Tutte Polynomial."
       Talking About Combinatorial Objects Seminar, 2015
       https://math.berkeley.edu/~brandtm/talks/tutte.pdf
    .. [2] A. Björklund, T. Husfeldt, P. Kaski, M. Koivisto,
       "Computing the Tutte polynomial in vertex-exponential time"
       49th Annual IEEE Symposium on Foundations of Computer Science, 2008
       https://ieeexplore.ieee.org/abstract/document/4691000
    .. [3] Y. Shi, M. Dehmer, X. Li, I. Gutman,
       "Graph Polynomials," p. 14
    .. [4] Y. Shi, M. Dehmer, X. Li, I. Gutman,
       "Graph Polynomials," p. 46
    .. [5] A. Nešetril, J. Goodall,
       "Graph invariants, homomorphisms, and the Tutte polynomial"
       https://iuuk.mff.cuni.cz/~andrew/Tutte.pdf
    .. [6] D. B. West,
       "Introduction to Graph Theory," p. 84
    .. [7] G. Coutinho,
       "A brief introduction to the Tutte polynomial"
       Structural Analysis of Complex Networks, 2011
       https://homepages.dcc.ufmg.br/~gabriel/seminars/coutinho_tuttepolynomial_seminar.pdf
    .. [8] J. A. Ellis-Monaghan, C. Merino,
       "Graph polynomials and their applications I: The Tutte polynomial"
       Structural Analysis of Complex Networks, 2011
       https://arxiv.org/pdf/0803.3079.pdf
    """
    import sympy

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    stack = deque()
    stack.append(nx.MultiGraph(G))

    polynomial = 0
    while stack:
        G = stack.pop()
        bridges = set(nx.bridges(G))

        e = None
        for i in G.edges:
            if (i[0], i[1]) not in bridges and i[0] != i[1]:
                e = i
                break
        if not e:
            loops = list(nx.selfloop_edges(G, keys=True))
            polynomial += x ** len(bridges) * y ** len(loops)
        else:
            # deletion-contraction
            C = nx.contracted_edge(G, e, self_loops=True)
            C.remove_edge(e[0], e[0])
            G.remove_edge(*e)
            stack.append(G)
            stack.append(C)
    return sympy.simplify(polynomial)


@not_implemented_for("directed")
@nx._dispatchable
def chromatic_polynomial(G):
    r"""Returns the chromatic polynomial of `G`

    This function computes the chromatic polynomial via an iterative version of
    the deletion-contraction algorithm.

    The chromatic polynomial `X_G(x)` is a fundamental graph polynomial
    invariant in one variable. Evaluating `X_G(k)` for an natural number `k`
    enumerates the proper k-colorings of `G`.

    There are several equivalent definitions; here are three:

    Def 1 (explicit formula):
    For `G` an undirected graph, `c(G)` the number of connected components of
    `G`, `E` the edge set of `G`, and `G(S)` the spanning subgraph of `G` with
    edge set `S` [1]_:

    .. math::

        X_G(x) = \sum_{S \subseteq E} (-1)^{|S|} x^{c(G(S))}


    Def 2 (interpolating polynomial):
    For `G` an undirected graph, `n(G)` the number of vertices of `G`, `k_0 = 0`,
    and `k_i` the number of distinct ways to color the vertices of `G` with `i`
    unique colors (for `i` a natural number at most `n(G)`), `X_G(x)` is the
    unique Lagrange interpolating polynomial of degree `n(G)` through the points
    `(0, k_0), (1, k_1), \dots, (n(G), k_{n(G)})` [2]_.


    Def 3 (chromatic recurrence):
    For `G` an undirected graph, `G-e` the graph obtained from `G` by deleting
    edge `e`, `G/e` the graph obtained from `G` by contracting edge `e`, `n(G)`
    the number of vertices of `G`, and `e(G)` the number of edges of `G` [3]_:

    .. math::
        X_G(x) = \begin{cases}
    	   x^{n(G)}, & \text{if $e(G)=0$} \\
           X_{G-e}(x) - X_{G/e}(x), & \text{otherwise, for an arbitrary edge $e$}
        \end{cases}

    This formulation is also known as the Fundamental Reduction Theorem [4]_.


    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    instance of `sympy.core.add.Add`
        A Sympy expression representing the chromatic polynomial for `G`.

    Examples
    --------
    >>> C = nx.cycle_graph(5)
    >>> nx.chromatic_polynomial(C)
    x**5 - 5*x**4 + 10*x**3 - 10*x**2 + 4*x

    >>> G = nx.complete_graph(4)
    >>> nx.chromatic_polynomial(G)
    x**4 - 6*x**3 + 11*x**2 - 6*x

    Notes
    -----
    Interpretation of the coefficients is discussed in [5]_. Several special
    cases are listed in [2]_.

    The chromatic polynomial is a specialization of the Tutte polynomial; in
    particular, ``X_G(x) = T_G(x, 0)`` [6]_.

    The chromatic polynomial may take negative arguments, though evaluations
    may not have chromatic interpretations. For instance, ``X_G(-1)`` enumerates
    the acyclic orientations of `G` [7]_.

    References
    ----------
    .. [1] D. B. West,
       "Introduction to Graph Theory," p. 222
    .. [2] E. W. Weisstein
       "Chromatic Polynomial"
       MathWorld--A Wolfram Web Resource
       https://mathworld.wolfram.com/ChromaticPolynomial.html
    .. [3] D. B. West,
       "Introduction to Graph Theory," p. 221
    .. [4] J. Zhang, J. Goodall,
       "An Introduction to Chromatic Polynomials"
       https://math.mit.edu/~apost/courses/18.204_2018/Julie_Zhang_paper.pdf
    .. [5] R. C. Read,
       "An Introduction to Chromatic Polynomials"
       Journal of Combinatorial Theory, 1968
       https://math.berkeley.edu/~mrklug/ReadChromatic.pdf
    .. [6] W. T. Tutte,
       "Graph-polynomials"
       Advances in Applied Mathematics, 2004
       https://www.sciencedirect.com/science/article/pii/S0196885803000411
    .. [7] R. P. Stanley,
       "Acyclic orientations of graphs"
       Discrete Mathematics, 2006
       https://math.mit.edu/~rstan/pubs/pubfiles/18.pdf
    """
    import sympy

    x = sympy.Symbol("x")
    stack = deque()
    stack.append(nx.MultiGraph(G, contraction_idx=0))

    polynomial = 0
    while stack:
        G = stack.pop()
        edges = list(G.edges)
        if not edges:
            polynomial += (-1) ** G.graph["contraction_idx"] * x ** len(G)
        else:
            e = edges[0]
            C = nx.contracted_edge(G, e, self_loops=True)
            C.graph["contraction_idx"] = G.graph["contraction_idx"] + 1
            C.remove_edge(e[0], e[0])
            G.remove_edge(*e)
            stack.append(G)
            stack.append(C)
    return polynomial
<|endoftext|>"""Swap edges in a graph."""

import math

import networkx as nx
from networkx.utils import py_random_state

__all__ = ["double_edge_swap", "connected_double_edge_swap", "directed_edge_swap"]


@nx.utils.not_implemented_for("undirected")
@py_random_state(3)
@nx._dispatchable(mutates_input=True, returns_graph=True)
def directed_edge_swap(G, *, nswap=1, max_tries=100, seed=None):
    """Swap three edges in a directed graph while keeping the node degrees fixed.

    A directed edge swap swaps three edges such that a -> b -> c -> d becomes
    a -> c -> b -> d. This pattern of swapping allows all possible states with the
    same in- and out-degree distribution in a directed graph to be reached.

    If the swap would create parallel edges (e.g. if a -> c already existed in the
    previous example), another attempt is made to find a suitable trio of edges.

    Parameters
    ----------
    G : DiGraph
       A directed graph

    nswap : integer (optional, default=1)
       Number of three-edge (directed) swaps to perform

    max_tries : integer (optional, default=100)
       Maximum number of attempts to swap edges

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : DiGraph
       The graph after the edges are swapped.

    Raises
    ------
    NetworkXError
        If `G` is not directed, or
        If nswap > max_tries, or
        If there are fewer than 4 nodes or 3 edges in `G`.
    NetworkXAlgorithmError
        If the number of swap attempts exceeds `max_tries` before `nswap` swaps are made

    Notes
    -----
    Does not enforce any connectivity constraints.

    The graph G is modified in place.

    A later swap is allowed to undo a previous swap.

    References
    ----------
    .. [1] Erdős, Péter L., et al. “A Simple Havel-Hakimi Type Algorithm to Realize
           Graphical Degree Sequences of Directed Graphs.” ArXiv:0905.4913 [Math],
           Jan. 2010. https://doi.org/10.48550/arXiv.0905.4913.
           Published  2010 in Elec. J. Combinatorics (17(1)). R66.
           http://www.combinatorics.org/Volume_17/PDF/v17i1r66.pdf
    .. [2] “Combinatorics - Reaching All Possible Simple Directed Graphs with a given
           Degree Sequence with 2-Edge Swaps.” Mathematics Stack Exchange,
           https://math.stackexchange.com/questions/22272/. Accessed 30 May 2022.
    """
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        raise nx.NetworkXError("DiGraph has fewer than four nodes.")
    if len(G.edges) < 3:
        raise nx.NetworkXError("DiGraph has fewer than 3 edges")

    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    tries = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence

    while swapcount < nswap:
        # choose source node index from discrete distribution
        start_index = discrete_sequence(1, cdistribution=cdf, seed=seed)[0]
        start = keys[start_index]
        tries += 1

        if tries > max_tries:
            msg = f"Maximum number of swap attempts ({tries}) exceeded before desired swaps achieved ({nswap})."
            raise nx.NetworkXAlgorithmError(msg)

        # If the given node doesn't have any out edges, then there isn't anything to swap
        if G.out_degree(start) == 0:
            continue
        second = seed.choice(list(G.succ[start]))
        if start == second:
            continue

        if G.out_degree(second) == 0:
            continue
        third = seed.choice(list(G.succ[second]))
        if second == third:
            continue

        if G.out_degree(third) == 0:
            continue
        fourth = seed.choice(list(G.succ[third]))
        if third == fourth:
            continue

        if (
            third not in G.succ[start]
            and fourth not in G.succ[second]
            and second not in G.succ[third]
        ):
            # Swap nodes
            G.add_edge(start, third)
            G.add_edge(third, second)
            G.add_edge(second, fourth)
            G.remove_edge(start, second)
            G.remove_edge(second, third)
            G.remove_edge(third, fourth)
            swapcount += 1

    return G


@py_random_state(3)
@nx._dispatchable(mutates_input=True, returns_graph=True)
def double_edge_swap(G, nswap=1, max_tries=100, seed=None):
    """Swap two edges in the graph while keeping the node degrees fixed.

    A double-edge swap removes two randomly chosen edges u-v and x-y
    and creates the new edges u-x and v-y::

     u--v            u  v
            becomes  |  |
     x--y            x  y

    If either the edge u-x or v-y already exist no swap is performed
    and another attempt is made to find a suitable edge pair.

    Parameters
    ----------
    G : graph
       An undirected graph

    nswap : integer (optional, default=1)
       Number of double-edge swaps to perform

    max_tries : integer (optional)
       Maximum number of attempts to swap edges

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
       The graph after double edge swaps.

    Raises
    ------
    NetworkXError
        If `G` is directed, or
        If `nswap` > `max_tries`, or
        If there are fewer than 4 nodes or 2 edges in `G`.
    NetworkXAlgorithmError
        If the number of swap attempts exceeds `max_tries` before `nswap` swaps are made

    Notes
    -----
    Does not enforce any connectivity constraints.

    The graph G is modified in place.
    """
    if G.is_directed():
        raise nx.NetworkXError(
            "double_edge_swap() not defined for directed graphs. Use directed_edge_swap instead."
        )
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        raise nx.NetworkXError("Graph has fewer than four nodes.")
    if len(G.edges) < 2:
        raise nx.NetworkXError("Graph has fewer than 2 edges")
    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    n = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence
    while swapcount < nswap:
        #        if random.random() < 0.5: continue # trick to avoid periodicities?
        # pick two random edges without creating edge list
        # choose source node indices from discrete distribution
        (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=seed)
        if ui == xi:
            continue  # same source, skip
        u = keys[ui]  # convert index to label
        x = keys[xi]
        # choose target uniformly from neighbors
        v = seed.choice(list(G[u]))
        y = seed.choice(list(G[x]))
        if v == y:
            continue  # same target, skip
        if (x not in G[u]) and (y not in G[v]):  # don't create parallel edges
            G.add_edge(u, x)
            G.add_edge(v, y)
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            swapcount += 1
        if n >= max_tries:
            e = (
                f"Maximum number of swap attempts ({n}) exceeded "
                f"before desired swaps achieved ({nswap})."
            )
            raise nx.NetworkXAlgorithmError(e)
        n += 1
    return G


@py_random_state(3)
@nx._dispatchable(mutates_input=True)
def connected_double_edge_swap(G, nswap=1, _window_threshold=3, seed=None):
    """Attempts the specified number of double-edge swaps in the graph `G`.

    A double-edge swap removes two randomly chosen edges `(u, v)` and `(x,
    y)` and creates the new edges `(u, x)` and `(v, y)`::

     u--v            u  v
            becomes  |  |
     x--y            x  y

    If either `(u, x)` or `(v, y)` already exist, then no swap is performed
    so the actual number of swapped edges is always *at most* `nswap`.

    Parameters
    ----------
    G : graph
       An undirected graph

    nswap : integer (optional, default=1)
       Number of double-edge swaps to perform

    _window_threshold : integer

       The window size below which connectedness of the graph will be checked
       after each swap.

       The "window" in this function is a dynamically updated integer that
       represents the number of swap attempts to make before checking if the
       graph remains connected. It is an optimization used to decrease the
       running time of the algorithm in exchange for increased complexity of
       implementation.

       If the window size is below this threshold, then the algorithm checks
       after each swap if the graph remains connected by checking if there is a
       path joining the two nodes whose edge was just removed. If the window
       size is above this threshold, then the algorithm performs do all the
       swaps in the window and only then check if the graph is still connected.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    int
       The number of successful swaps

    Raises
    ------

    NetworkXError

       If the input graph is not connected, or if the graph has fewer than four
       nodes.

    Notes
    -----

    The initial graph `G` must be connected, and the resulting graph is
    connected. The graph `G` is modified in place.

    References
    ----------
    .. [1] C. Gkantsidis and M. Mihail and E. Zegura,
           The Markov chain simulation method for generating connected
           power law random graphs, 2003.
           http://citeseer.ist.psu.edu/gkantsidis03markov.html
    """
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph not connected")
    if len(G) < 4:
        raise nx.NetworkXError("Graph has fewer than four nodes.")
    n = 0
    swapcount = 0
    deg = G.degree()
    # Label key for nodes
    dk = [n for n, d in G.degree()]
    cdf = nx.utils.cumulative_distribution([d for n, d in G.degree()])
    discrete_sequence = nx.utils.discrete_sequence
    window = 1
    while n < nswap:
        wcount = 0
        swapped = []
        # If the window is small, we just check each time whether the graph is
        # connected by checking if the nodes that were just separated are still
        # connected.
        if window < _window_threshold:
            # This Boolean keeps track of whether there was a failure or not.
            fail = False
            while wcount < window and n < nswap:
                # Pick two random edges without creating the edge list. Choose
                # source nodes from the discrete degree distribution.
                (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=seed)
                # If the source nodes are the same, skip this pair.
                if ui == xi:
                    continue
                # Convert an index to a node label.
                u = dk[ui]
                x = dk[xi]
                # Choose targets uniformly from neighbors.
                v = seed.choice(list(G.neighbors(u)))
                y = seed.choice(list(G.neighbors(x)))
                # If the target nodes are the same, skip this pair.
                if v == y:
                    continue
                if x not in G[u] and y not in G[v]:
                    G.remove_edge(u, v)
                    G.remove_edge(x, y)
                    G.add_edge(u, x)
                    G.add_edge(v, y)
                    swapped.append((u, v, x, y))
                    swapcount += 1
                n += 1
                # If G remains connected...
                if nx.has_path(G, u, v):
                    wcount += 1
                # Otherwise, undo the changes.
                else:
                    G.add_edge(u, v)
                    G.add_edge(x, y)
                    G.remove_edge(u, x)
                    G.remove_edge(v, y)
                    swapcount -= 1
                    fail = True
            # If one of the swaps failed, reduce the window size.
            if fail:
                window = math.ceil(window / 2)
            else:
                window += 1
        # If the window is large, then there is a good chance that a bunch of
        # swaps will work. It's quicker to do all those swaps first and then
        # check if the graph remains connected.
        else:
            while wcount < window and n < nswap:
                # Pick two random edges without creating the edge list. Choose
                # source nodes from the discrete degree distribution.
                (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=seed)
                # If the source nodes are the same, skip this pair.
                if ui == xi:
                    continue
                # Convert an index to a node label.
                u = dk[ui]
                x = dk[xi]
                # Choose targets uniformly from neighbors.
                v = seed.choice(list(G.neighbors(u)))
                y = seed.choice(list(G.neighbors(x)))
                # If the target nodes are the same, skip this pair.
                if v == y:
                    continue
                if x not in G[u] and y not in G[v]:
                    G.remove_edge(u, v)
                    G.remove_edge(x, y)
                    G.add_edge(u, x)
                    G.add_edge(v, y)
                    swapped.append((u, v, x, y))
                    swapcount += 1
                n += 1
                wcount += 1
            # If the graph remains connected, increase the window size.
            if nx.is_connected(G):
                window += 1
            # Otherwise, undo the changes from the previous window and decrease
            # the window size.
            else:
                while swapped:
                    (u, v, x, y) = swapped.pop()
                    G.add_edge(u, v)
                    G.add_edge(x, y)
                    G.remove_edge(u, x)
                    G.remove_edge(v, y)
                    swapcount -= 1
                window = math.ceil(window / 2)
    return swapcount
<|endoftext|>"""
Functions for hashing graphs to strings.
Isomorphic graphs should be assigned identical hashes.
For now, only Weisfeiler-Lehman hashing is implemented.
"""

from collections import Counter, defaultdict
from hashlib import blake2b

import networkx as nx

__all__ = ["weisfeiler_lehman_graph_hash", "weisfeiler_lehman_subgraph_hashes"]


def _hash_label(label, digest_size):
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()


def _init_node_labels(G, edge_attr, node_attr):
    if node_attr:
        return {u: str(dd[node_attr]) for u, dd in G.nodes(data=True)}
    elif edge_attr:
        return {u: "" for u in G}
    else:
        return {u: str(deg) for u, deg in G.degree()}


def _neighborhood_aggregate(G, node, node_labels, edge_attr=None):
    """
    Compute new labels for given node by aggregating
    the labels of each node's neighbors.
    """
    label_list = []
    for nbr in G.neighbors(node):
        prefix = "" if edge_attr is None else str(G[node][nbr][edge_attr])
        label_list.append(prefix + node_labels[nbr])
    return node_labels[node] + "".join(sorted(label_list))


@nx.utils.not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs={"edge_attr": None}, node_attrs="node_attr")
def weisfeiler_lehman_graph_hash(
    G, edge_attr=None, node_attr=None, iterations=3, digest_size=16
):
    """Return Weisfeiler Lehman (WL) graph hash.

    The function iteratively aggregates and hashes neighborhoods of each node.
    After each node's neighbors are hashed to obtain updated node labels,
    a hashed histogram of resulting labels is returned as the final hash.

    Hashes are identical for isomorphic graphs and strong guarantees that
    non-isomorphic graphs will get different hashes. See [1]_ for details.

    If no node or edge attributes are provided, the degree of each node
    is used as its initial label.
    Otherwise, node and/or edge labels are used to compute the hash.

    Parameters
    ----------
    G : graph
        The graph to be hashed.
        Can have node and/or edge attributes. Can also have no attributes.
    edge_attr : string, optional (default=None)
        The key in edge attribute dictionary to be used for hashing.
        If None, edge labels are ignored.
    node_attr: string, optional (default=None)
        The key in node attribute dictionary to be used for hashing.
        If None, and no edge_attr given, use the degrees of the nodes as labels.
    iterations: int, optional (default=3)
        Number of neighbor aggregations to perform.
        Should be larger for larger graphs.
    digest_size: int, optional (default=16)
        Size (in bits) of blake2b hash digest to use for hashing node labels.

    Returns
    -------
    h : string
        Hexadecimal string corresponding to hash of the input graph.

    Examples
    --------
    Two graphs with edge attributes that are isomorphic, except for
    differences in the edge labels.

    >>> G1 = nx.Graph()
    >>> G1.add_edges_from(
    ...     [
    ...         (1, 2, {"label": "A"}),
    ...         (2, 3, {"label": "A"}),
    ...         (3, 1, {"label": "A"}),
    ...         (1, 4, {"label": "B"}),
    ...     ]
    ... )
    >>> G2 = nx.Graph()
    >>> G2.add_edges_from(
    ...     [
    ...         (5, 6, {"label": "B"}),
    ...         (6, 7, {"label": "A"}),
    ...         (7, 5, {"label": "A"}),
    ...         (7, 8, {"label": "A"}),
    ...     ]
    ... )

    Omitting the `edge_attr` option, results in identical hashes.

    >>> nx.weisfeiler_lehman_graph_hash(G1)
    '7bc4dde9a09d0b94c5097b219891d81a'
    >>> nx.weisfeiler_lehman_graph_hash(G2)
    '7bc4dde9a09d0b94c5097b219891d81a'

    With edge labels, the graphs are no longer assigned
    the same hash digest.

    >>> nx.weisfeiler_lehman_graph_hash(G1, edge_attr="label")
    'c653d85538bcf041d88c011f4f905f10'
    >>> nx.weisfeiler_lehman_graph_hash(G2, edge_attr="label")
    '3dcd84af1ca855d0eff3c978d88e7ec7'

    Notes
    -----
    To return the WL hashes of each subgraph of a graph, use
    `weisfeiler_lehman_subgraph_hashes`

    Similarity between hashes does not imply similarity between graphs.

    References
    ----------
    .. [1] Shervashidze, Nino, Pascal Schweitzer, Erik Jan Van Leeuwen,
       Kurt Mehlhorn, and Karsten M. Borgwardt. Weisfeiler Lehman
       Graph Kernels. Journal of Machine Learning Research. 2011.
       http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf

    See also
    --------
    weisfeiler_lehman_subgraph_hashes
    """

    def weisfeiler_lehman_step(G, labels, edge_attr=None):
        """
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        """
        new_labels = {}
        for node in G.nodes():
            label = _neighborhood_aggregate(G, node, labels, edge_attr=edge_attr)
            new_labels[node] = _hash_label(label, digest_size)
        return new_labels

    # set initial node labels
    node_labels = _init_node_labels(G, edge_attr, node_attr)

    subgraph_hash_counts = []
    for _ in range(iterations):
        node_labels = weisfeiler_lehman_step(G, node_labels, edge_attr=edge_attr)
        counter = Counter(node_labels.values())
        # sort the counter, extend total counts
        subgraph_hash_counts.extend(sorted(counter.items(), key=lambda x: x[0]))

    # hash the final counter
    return _hash_label(str(tuple(subgraph_hash_counts)), digest_size)


@nx.utils.not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs={"edge_attr": None}, node_attrs="node_attr")
def weisfeiler_lehman_subgraph_hashes(
    G,
    edge_attr=None,
    node_attr=None,
    iterations=3,
    digest_size=16,
    include_initial_labels=False,
):
    """
    Return a dictionary of subgraph hashes by node.

    Dictionary keys are nodes in `G`, and values are a list of hashes.
    Each hash corresponds to a subgraph rooted at a given node u in `G`.
    Lists of subgraph hashes are sorted in increasing order of depth from
    their root node, with the hash at index i corresponding to a subgraph
    of nodes at most i edges distance from u. Thus, each list will contain
    `iterations` elements - a hash for a subgraph at each depth. If
    `include_initial_labels` is set to `True`, each list will additionally
    have contain a hash of the initial node label (or equivalently a
    subgraph of depth 0) prepended, totalling ``iterations + 1`` elements.

    The function iteratively aggregates and hashes neighborhoods of each node.
    This is achieved for each step by replacing for each node its label from
    the previous iteration with its hashed 1-hop neighborhood aggregate.
    The new node label is then appended to a list of node labels for each
    node.

    To aggregate neighborhoods for a node $u$ at each step, all labels of
    nodes adjacent to $u$ are concatenated. If the `edge_attr` parameter is set,
    labels for each neighboring node are prefixed with the value of this attribute
    along the connecting edge from this neighbor to node $u$. The resulting string
    is then hashed to compress this information into a fixed digest size.

    Thus, at the $i$-th iteration, nodes within $i$ hops influence any given
    hashed node label. We can therefore say that at depth $i$ for node $u$
    we have a hash for a subgraph induced by the $i$-hop neighborhood of $u$.

    The output can be used to create general Weisfeiler-Lehman graph kernels,
    or generate features for graphs or nodes - for example to generate 'words' in
    a graph as seen in the 'graph2vec' algorithm.
    See [1]_ & [2]_ respectively for details.

    Hashes are identical for isomorphic subgraphs and there exist strong
    guarantees that non-isomorphic graphs will get different hashes.
    See [1]_ for details.

    If no node or edge attributes are provided, the degree of each node
    is used as its initial label.
    Otherwise, node and/or edge labels are used to compute the hash.

    Parameters
    ----------
    G : graph
        The graph to be hashed.
        Can have node and/or edge attributes. Can also have no attributes.
    edge_attr : string, optional (default=None)
        The key in edge attribute dictionary to be used for hashing.
        If None, edge labels are ignored.
    node_attr : string, optional (default=None)
        The key in node attribute dictionary to be used for hashing.
        If None, and no edge_attr given, use the degrees of the nodes as labels.
        If None, and edge_attr is given, each node starts with an identical label.
    iterations : int, optional (default=3)
        Number of neighbor aggregations to perform.
        Should be larger for larger graphs.
    digest_size : int, optional (default=16)
        Size (in bits) of blake2b hash digest to use for hashing node labels.
        The default size is 16 bits.
    include_initial_labels : bool, optional (default=False)
        If True, include the hashed initial node label as the first subgraph
        hash for each node.

    Returns
    -------
    node_subgraph_hashes : dict
        A dictionary with each key given by a node in G, and each value given
        by the subgraph hashes in order of depth from the key node.

    Examples
    --------
    Finding similar nodes in different graphs:

    >>> G1 = nx.Graph()
    >>> G1.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)])
    >>> G2 = nx.Graph()
    >>> G2.add_edges_from([(1, 3), (2, 3), (1, 6), (1, 5), (4, 6)])
    >>> g1_hashes = nx.weisfeiler_lehman_subgraph_hashes(
    ...     G1, iterations=3, digest_size=8
    ... )
    >>> g2_hashes = nx.weisfeiler_lehman_subgraph_hashes(
    ...     G2, iterations=3, digest_size=8
    ... )

    Even though G1 and G2 are not isomorphic (they have different numbers of edges),
    the hash sequence of depth 3 for node 1 in G1 and node 5 in G2 are similar:

    >>> g1_hashes[1]
    ['a93b64973cfc8897', 'db1b43ae35a1878f', '57872a7d2059c1c0']
    >>> g2_hashes[5]
    ['a93b64973cfc8897', 'db1b43ae35a1878f', '1716d2a4012fa4bc']

    The first 2 WL subgraph hashes match. From this we can conclude that it's very
    likely the neighborhood of 2 hops around these nodes are isomorphic.

    However the 3-hop neighborhoods of ``G1`` and ``G2`` are not isomorphic since the
    3rd hashes in the lists above are not equal.

    These nodes may be candidates to be classified together since their local topology
    is similar.

    Notes
    -----
    To hash the full graph when subgraph hashes are not needed, use
    `weisfeiler_lehman_graph_hash` for efficiency.

    Similarity between hashes does not imply similarity between graphs.

    References
    ----------
    .. [1] Shervashidze, Nino, Pascal Schweitzer, Erik Jan Van Leeuwen,
       Kurt Mehlhorn, and Karsten M. Borgwardt. Weisfeiler Lehman
       Graph Kernels. Journal of Machine Learning Research. 2011.
       http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf
    .. [2] Annamalai Narayanan, Mahinthan Chandramohan, Rajasekar Venkatesan,
       Lihui Chen, Yang Liu and Shantanu Jaiswa. graph2vec: Learning
       Distributed Representations of Graphs. arXiv. 2017
       https://arxiv.org/pdf/1707.05005.pdf

    See also
    --------
    weisfeiler_lehman_graph_hash
    """

    def weisfeiler_lehman_step(G, labels, node_subgraph_hashes, edge_attr=None):
        """
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        Appends the new hashed label to the dictionary of subgraph hashes
        originating from and indexed by each node in G
        """
        new_labels = {}
        for node in G.nodes():
            label = _neighborhood_aggregate(G, node, labels, edge_attr=edge_attr)
            hashed_label = _hash_label(label, digest_size)
            new_labels[node] = hashed_label
            node_subgraph_hashes[node].append(hashed_label)
        return new_labels

    node_labels = _init_node_labels(G, edge_attr, node_attr)
    if include_initial_labels:
        node_subgraph_hashes = {
            k: [_hash_label(v, digest_size)] for k, v in node_labels.items()
        }
    else:
        node_subgraph_hashes = defaultdict(list)

    for _ in range(iterations):
        node_labels = weisfeiler_lehman_step(
            G, node_labels, node_subgraph_hashes, edge_attr
        )

    return dict(node_subgraph_hashes)
<|endoftext|>"""
Flow Hierarchy.
"""

import networkx as nx

__all__ = ["flow_hierarchy"]


@nx._dispatchable(edge_attrs="weight")
def flow_hierarchy(G, weight=None):
    """Returns the flow hierarchy of a directed network.

    Flow hierarchy is defined as the fraction of edges not participating
    in cycles in a directed graph [1]_.

    Parameters
    ----------
    G : DiGraph or MultiDiGraph
       A directed graph

    weight : string, optional (default=None)
       Attribute to use for edge weights. If None the weight defaults to 1.

    Returns
    -------
    h : float
       Flow hierarchy value

    Raises
    ------
    NetworkXError
       If `G` is not a directed graph or if `G` has no edges.

    Notes
    -----
    The algorithm described in [1]_ computes the flow hierarchy through
    exponentiation of the adjacency matrix.  This function implements an
    alternative approach that finds strongly connected components.
    An edge is in a cycle if and only if it is in a strongly connected
    component, which can be found in $O(m)$ time using Tarjan's algorithm.

    References
    ----------
    .. [1] Luo, J.; Magee, C.L. (2011),
       Detecting evolving patterns of self-organizing networks by flow
       hierarchy measurement, Complexity, Volume 16 Issue 6 53-61.
       DOI: 10.1002/cplx.20368
       http://web.mit.edu/~cmagee/www/documents/28-DetectingEvolvingPatterns_FlowHierarchy.pdf
    """
    # corner case: G has no edges
    if nx.is_empty(G):
        raise nx.NetworkXError("flow_hierarchy not applicable to empty graphs")
    if not G.is_directed():
        raise nx.NetworkXError("G must be a digraph in flow_hierarchy")
    scc = nx.strongly_connected_components(G)
    return 1 - sum(G.subgraph(c).size(weight) for c in scc) / G.size(weight)
<|endoftext|>"""Time dependent algorithms."""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["cd_index"]


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
@nx._dispatchable(node_attrs={"time": None, "weight": 1})
def cd_index(G, node, time_delta, *, time="time", weight=None):
    r"""Compute the CD index for `node` within the graph `G`.

    Calculates the CD index for the given node of the graph,
    considering only its predecessors who have the `time` attribute
    smaller than or equal to the `time` attribute of the `node`
    plus `time_delta`.

    Parameters
    ----------
    G : graph
       A directed networkx graph whose nodes have `time` attributes and optionally
       `weight` attributes (if a weight is not given, it is considered 1).
    node : node
       The node for which the CD index is calculated.
    time_delta : numeric or timedelta
       Amount of time after the `time` attribute of the `node`. The value of
       `time_delta` must support comparison with the `time` node attribute. For
       example, if the `time` attribute of the nodes are `datetime.datetime`
       objects, then `time_delta` should be a `datetime.timedelta` object.
    time : string (Optional, default is "time")
        The name of the node attribute that will be used for the calculations.
    weight : string (Optional, default is None)
        The name of the node attribute used as weight.

    Returns
    -------
    float
       The CD index calculated for the node `node` within the graph `G`.

    Raises
    ------
    NetworkXError
       If not all nodes have a `time` attribute or
       `time_delta` and `time` attribute types are not compatible or
       `n` equals 0.

    NetworkXNotImplemented
        If `G` is a non-directed graph or a multigraph.

    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> G = nx.DiGraph()
    >>> nodes = {
    ...     1: {"time": datetime(2015, 1, 1)},
    ...     2: {"time": datetime(2012, 1, 1), "weight": 4},
    ...     3: {"time": datetime(2010, 1, 1)},
    ...     4: {"time": datetime(2008, 1, 1)},
    ...     5: {"time": datetime(2014, 1, 1)},
    ... }
    >>> G.add_nodes_from([(n, nodes[n]) for n in nodes])
    >>> edges = [(1, 3), (1, 4), (2, 3), (3, 4), (3, 5)]
    >>> G.add_edges_from(edges)
    >>> delta = timedelta(days=5 * 365)
    >>> nx.cd_index(G, 3, time_delta=delta, time="time")
    0.5
    >>> nx.cd_index(G, 3, time_delta=delta, time="time", weight="weight")
    0.12

    Integers can also be used for the time values:
    >>> node_times = {1: 2015, 2: 2012, 3: 2010, 4: 2008, 5: 2014}
    >>> nx.set_node_attributes(G, node_times, "new_time")
    >>> nx.cd_index(G, 3, time_delta=4, time="new_time")
    0.5
    >>> nx.cd_index(G, 3, time_delta=4, time="new_time", weight="weight")
    0.12

    Notes
    -----
    This method implements the algorithm for calculating the CD index,
    as described in the paper by Funk and Owen-Smith [1]_. The CD index
    is used in order to check how consolidating or destabilizing a patent
    is, hence the nodes of the graph represent patents and the edges show
    the citations between these patents. The mathematical model is given
    below:

    .. math::
        CD_{t}=\frac{1}{n_{t}}\sum_{i=1}^{n}\frac{-2f_{it}b_{it}+f_{it}}{w_{it}},

    where `f_{it}` equals 1 if `i` cites the focal patent else 0, `b_{it}` equals
    1 if `i` cites any of the focal patents successors else 0, `n_{t}` is the number
    of forward citations in `i` and `w_{it}` is a matrix of weight for patent `i`
    at time `t`.

    The `datetime.timedelta` package can lead to off-by-one issues when converting
    from years to days. In the example above `timedelta(days=5 * 365)` looks like
    5 years, but it isn't because of leap year days. So it gives the same result
    as `timedelta(days=4 * 365)`. But using `timedelta(days=5 * 365 + 1)` gives
    a 5 year delta **for this choice of years** but may not if the 5 year gap has
    more than 1 leap year. To avoid these issues, use integers to represent years,
    or be very careful when you convert units of time.

    References
    ----------
    .. [1] Funk, Russell J., and Jason Owen-Smith.
           "A dynamic network measure of technological change."
           Management science 63, no. 3 (2017): 791-817.
           http://russellfunk.org/cdindex/static/papers/funk_ms_2017.pdf

    """
    if not all(time in G.nodes[n] for n in G):
        raise nx.NetworkXError("Not all nodes have a 'time' attribute.")

    try:
        # get target_date
        target_date = G.nodes[node][time] + time_delta
        # keep the predecessors that existed before the target date
        pred = {i for i in G.pred[node] if G.nodes[i][time] <= target_date}
    except:
        raise nx.NetworkXError(
            "Addition and comparison are not supported between 'time_delta' "
            "and 'time' types."
        )

    # -1 if any edge between node's predecessors and node's successors, else 1
    b = [-1 if any(j in G[i] for j in G[node]) else 1 for i in pred]

    # n is size of the union of the focal node's predecessors and its successors' predecessors
    n = len(pred.union(*(G.pred[s].keys() - {node} for s in G[node])))
    if n == 0:
        raise nx.NetworkXError("The cd index cannot be defined.")

    # calculate cd index
    if weight is None:
        return round(sum(bi for bi in b) / n, 2)
    else:
        # If a node has the specified weight attribute, its weight is used in the calculation
        # otherwise, a weight of 1 is assumed for that node
        weights = [G.nodes[i].get(weight, 1) for i in pred]
        return round(sum(bi / wt for bi, wt in zip(b, weights)) / n, 2)
<|endoftext|>"""
Vitality measures.
"""

from functools import partial

import networkx as nx

__all__ = ["closeness_vitality"]


@nx._dispatchable(edge_attrs="weight")
def closeness_vitality(G, node=None, weight=None, wiener_index=None):
    """Returns the closeness vitality for nodes in the graph.

    The *closeness vitality* of a node, defined in Section 3.6.2 of [1],
    is the change in the sum of distances between all node pairs when
    excluding that node.

    Parameters
    ----------
    G : NetworkX graph
        A strongly-connected graph.

    weight : string
        The name of the edge attribute used as weight. This is passed
        directly to the :func:`~networkx.wiener_index` function.

    node : object
        If specified, only the closeness vitality for this node will be
        returned. Otherwise, a dictionary mapping each node to its
        closeness vitality will be returned.

    Other parameters
    ----------------
    wiener_index : number
        If you have already computed the Wiener index of the graph
        `G`, you can provide that value here. Otherwise, it will be
        computed for you.

    Returns
    -------
    dictionary or float
        If `node` is None, this function returns a dictionary
        with nodes as keys and closeness vitality as the
        value. Otherwise, it returns only the closeness vitality for the
        specified `node`.

        The closeness vitality of a node may be negative infinity if
        removing that node would disconnect the graph.

    Examples
    --------
    >>> G = nx.cycle_graph(3)
    >>> nx.closeness_vitality(G)
    {0: 2.0, 1: 2.0, 2: 2.0}

    See Also
    --------
    closeness_centrality

    References
    ----------
    .. [1] Ulrik Brandes, Thomas Erlebach (eds.).
           *Network Analysis: Methodological Foundations*.
           Springer, 2005.
           <http://books.google.com/books?id=TTNhSm7HYrIC>

    """
    if wiener_index is None:
        wiener_index = nx.wiener_index(G, weight=weight)
    if node is not None:
        after = nx.wiener_index(G.subgraph(set(G) - {node}), weight=weight)
        return wiener_index - after
    vitality = partial(closeness_vitality, G, weight=weight, wiener_index=wiener_index)
    return {v: vitality(node=v) for v in G}
<|endoftext|>"""Functions measuring similarity using graph edit distance.

The graph edit distance is the number of edge/node changes needed
to make two graphs isomorphic.

The default algorithm/implementation is sub-optimal for some graphs.
The problem of finding the exact Graph Edit Distance (GED) is NP-hard
so it is often slow. If the simple interface `graph_edit_distance`
takes too long for your graph, try `optimize_graph_edit_distance`
and/or `optimize_edit_paths`.

At the same time, I encourage capable people to investigate
alternative GED algorithms, in order to improve the choices available.
"""

import math
import time
import warnings
from dataclasses import dataclass
from itertools import product

import networkx as nx
from networkx.utils import np_random_state

__all__ = [
    "graph_edit_distance",
    "optimal_edit_paths",
    "optimize_graph_edit_distance",
    "optimize_edit_paths",
    "simrank_similarity",
    "panther_similarity",
    "generate_random_paths",
]


def debug_print(*args, **kwargs):
    print(*args, **kwargs)


@nx._dispatchable(
    graphs={"G1": 0, "G2": 1}, preserve_edge_attrs=True, preserve_node_attrs=True
)
def graph_edit_distance(
    G1,
    G2,
    node_match=None,
    edge_match=None,
    node_subst_cost=None,
    node_del_cost=None,
    node_ins_cost=None,
    edge_subst_cost=None,
    edge_del_cost=None,
    edge_ins_cost=None,
    roots=None,
    upper_bound=None,
    timeout=None,
):
    """Returns GED (graph edit distance) between graphs G1 and G2.

    Graph edit distance is a graph similarity measure analogous to
    Levenshtein distance for strings.  It is defined as minimum cost
    of edit path (sequence of node and edge edit operations)
    transforming graph G1 to graph isomorphic to G2.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be of the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2
        should be considered equal during matching.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute
        dictionaries for n1 and n2 as inputs.

        Ignored if node_subst_cost is specified.  If neither
        node_match nor node_subst_cost are specified then node
        attributes are not considered.

    edge_match : callable
        A function that returns True if the edge attribute dictionaries
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during matching.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute
        dictionaries of the edges under consideration.

        Ignored if edge_subst_cost is specified.  If neither
        edge_match nor edge_subst_cost are specified then edge
        attributes are not considered.

    node_subst_cost, node_del_cost, node_ins_cost : callable
        Functions that return the costs of node substitution, node
        deletion, and node insertion, respectively.

        The functions will be called like

           node_subst_cost(G1.nodes[n1], G2.nodes[n2]),
           node_del_cost(G1.nodes[n1]),
           node_ins_cost(G2.nodes[n2]).

        That is, the functions will receive the node attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function node_subst_cost overrides node_match if specified.
        If neither node_match nor node_subst_cost are specified then
        default node substitution cost of 0 is used (node attributes
        are not considered during matching).

        If node_del_cost is not specified then default node deletion
        cost of 1 is used.  If node_ins_cost is not specified then
        default node insertion cost of 1 is used.

    edge_subst_cost, edge_del_cost, edge_ins_cost : callable
        Functions that return the costs of edge substitution, edge
        deletion, and edge insertion, respectively.

        The functions will be called like

           edge_subst_cost(G1[u1][v1], G2[u2][v2]),
           edge_del_cost(G1[u1][v1]),
           edge_ins_cost(G2[u2][v2]).

        That is, the functions will receive the edge attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function edge_subst_cost overrides edge_match if specified.
        If neither edge_match nor edge_subst_cost are specified then
        default edge substitution cost of 0 is used (edge attributes
        are not considered during matching).

        If edge_del_cost is not specified then default edge deletion
        cost of 1 is used.  If edge_ins_cost is not specified then
        default edge insertion cost of 1 is used.

    roots : 2-tuple
        Tuple where first element is a node in G1 and the second
        is a node in G2.
        These nodes are forced to be matched in the comparison to
        allow comparison between rooted graphs.

    upper_bound : numeric
        Maximum edit distance to consider.  Return None if no edit
        distance under or equal to upper_bound exists.

    timeout : numeric
        Maximum number of seconds to execute.
        After timeout is met, the current best GED is returned.

    Examples
    --------
    >>> G1 = nx.cycle_graph(6)
    >>> G2 = nx.wheel_graph(7)
    >>> nx.graph_edit_distance(G1, G2)
    7.0

    >>> G1 = nx.star_graph(5)
    >>> G2 = nx.star_graph(5)
    >>> nx.graph_edit_distance(G1, G2, roots=(0, 0))
    0.0
    >>> nx.graph_edit_distance(G1, G2, roots=(1, 0))
    8.0

    See Also
    --------
    optimal_edit_paths, optimize_graph_edit_distance,

    is_isomorphic: test for graph edit distance of 0

    References
    ----------
    .. [1] Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, Patrick
       Martineau. An Exact Graph Edit Distance Algorithm for Solving
       Pattern Recognition Problems. 4th International Conference on
       Pattern Recognition Applications and Methods 2015, Jan 2015,
       Lisbon, Portugal. 2015,
       <10.5220/0005209202710278>. <hal-01168816>
       https://hal.archives-ouvertes.fr/hal-01168816

    """
    bestcost = None
    for _, _, cost in optimize_edit_paths(
        G1,
        G2,
        node_match,
        edge_match,
        node_subst_cost,
        node_del_cost,
        node_ins_cost,
        edge_subst_cost,
        edge_del_cost,
        edge_ins_cost,
        upper_bound,
        True,
        roots,
        timeout,
    ):
        # assert bestcost is None or cost < bestcost
        bestcost = cost
    return bestcost


@nx._dispatchable(graphs={"G1": 0, "G2": 1})
def optimal_edit_paths(
    G1,
    G2,
    node_match=None,
    edge_match=None,
    node_subst_cost=None,
    node_del_cost=None,
    node_ins_cost=None,
    edge_subst_cost=None,
    edge_del_cost=None,
    edge_ins_cost=None,
    upper_bound=None,
):
    """Returns all minimum-cost edit paths transforming G1 to G2.

    Graph edit path is a sequence of node and edge edit operations
    transforming graph G1 to graph isomorphic to G2.  Edit operations
    include substitutions, deletions, and insertions.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be of the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2
        should be considered equal during matching.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute
        dictionaries for n1 and n2 as inputs.

        Ignored if node_subst_cost is specified.  If neither
        node_match nor node_subst_cost are specified then node
        attributes are not considered.

    edge_match : callable
        A function that returns True if the edge attribute dictionaries
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during matching.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute
        dictionaries of the edges under consideration.

        Ignored if edge_subst_cost is specified.  If neither
        edge_match nor edge_subst_cost are specified then edge
        attributes are not considered.

    node_subst_cost, node_del_cost, node_ins_cost : callable
        Functions that return the costs of node substitution, node
        deletion, and node insertion, respectively.

        The functions will be called like

           node_subst_cost(G1.nodes[n1], G2.nodes[n2]),
           node_del_cost(G1.nodes[n1]),
           node_ins_cost(G2.nodes[n2]).

        That is, the functions will receive the node attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function node_subst_cost overrides node_match if specified.
        If neither node_match nor node_subst_cost are specified then
        default node substitution cost of 0 is used (node attributes
        are not considered during matching).

        If node_del_cost is not specified then default node deletion
        cost of 1 is used.  If node_ins_cost is not specified then
        default node insertion cost of 1 is used.

    edge_subst_cost, edge_del_cost, edge_ins_cost : callable
        Functions that return the costs of edge substitution, edge
        deletion, and edge insertion, respectively.

        The functions will be called like

           edge_subst_cost(G1[u1][v1], G2[u2][v2]),
           edge_del_cost(G1[u1][v1]),
           edge_ins_cost(G2[u2][v2]).

        That is, the functions will receive the edge attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function edge_subst_cost overrides edge_match if specified.
        If neither edge_match nor edge_subst_cost are specified then
        default edge substitution cost of 0 is used (edge attributes
        are not considered during matching).

        If edge_del_cost is not specified then default edge deletion
        cost of 1 is used.  If edge_ins_cost is not specified then
        default edge insertion cost of 1 is used.

    upper_bound : numeric
        Maximum edit distance to consider.

    Returns
    -------
    edit_paths : list of tuples (node_edit_path, edge_edit_path)
       - node_edit_path : list of tuples ``(u, v)`` indicating node transformations
         between `G1` and `G2`. ``u`` is `None` for insertion, ``v`` is `None`
         for deletion.
       - edge_edit_path : list of tuples ``((u1, v1), (u2, v2))`` indicating edge
         transformations between `G1` and `G2`. ``(None, (u2,v2))`` for insertion
         and ``((u1,v1), None)`` for deletion.

    cost : numeric
        Optimal edit path cost (graph edit distance). When the cost
        is zero, it indicates that `G1` and `G2` are isomorphic.

    Examples
    --------
    >>> G1 = nx.cycle_graph(4)
    >>> G2 = nx.wheel_graph(5)
    >>> paths, cost = nx.optimal_edit_paths(G1, G2)
    >>> len(paths)
    40
    >>> cost
    5.0

    Notes
    -----
    To transform `G1` into a graph isomorphic to `G2`, apply the node
    and edge edits in the returned ``edit_paths``.
    In the case of isomorphic graphs, the cost is zero, and the paths
    represent different isomorphic mappings (isomorphisms). That is, the
    edits involve renaming nodes and edges to match the structure of `G2`.

    See Also
    --------
    graph_edit_distance, optimize_edit_paths

    References
    ----------
    .. [1] Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, Patrick
       Martineau. An Exact Graph Edit Distance Algorithm for Solving
       Pattern Recognition Problems. 4th International Conference on
       Pattern Recognition Applications and Methods 2015, Jan 2015,
       Lisbon, Portugal. 2015,
       <10.5220/0005209202710278>. <hal-01168816>
       https://hal.archives-ouvertes.fr/hal-01168816

    """
    paths = []
    bestcost = None
    for vertex_path, edge_path, cost in optimize_edit_paths(
        G1,
        G2,
        node_match,
        edge_match,
        node_subst_cost,
        node_del_cost,
        node_ins_cost,
        edge_subst_cost,
        edge_del_cost,
        edge_ins_cost,
        upper_bound,
        False,
    ):
        # assert bestcost is None or cost <= bestcost
        if bestcost is not None and cost < bestcost:
            paths = []
        paths.append((vertex_path, edge_path))
        bestcost = cost
    return paths, bestcost


@nx._dispatchable(graphs={"G1": 0, "G2": 1})
def optimize_graph_edit_distance(
    G1,
    G2,
    node_match=None,
    edge_match=None,
    node_subst_cost=None,
    node_del_cost=None,
    node_ins_cost=None,
    edge_subst_cost=None,
    edge_del_cost=None,
    edge_ins_cost=None,
    upper_bound=None,
):
    """Returns consecutive approximations of GED (graph edit distance)
    between graphs G1 and G2.

    Graph edit distance is a graph similarity measure analogous to
    Levenshtein distance for strings.  It is defined as minimum cost
    of edit path (sequence of node and edge edit operations)
    transforming graph G1 to graph isomorphic to G2.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be of the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2
        should be considered equal during matching.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute
        dictionaries for n1 and n2 as inputs.

        Ignored if node_subst_cost is specified.  If neither
        node_match nor node_subst_cost are specified then node
        attributes are not considered.

    edge_match : callable
        A function that returns True if the edge attribute dictionaries
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during matching.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute
        dictionaries of the edges under consideration.

        Ignored if edge_subst_cost is specified.  If neither
        edge_match nor edge_subst_cost are specified then edge
        attributes are not considered.

    node_subst_cost, node_del_cost, node_ins_cost : callable
        Functions that return the costs of node substitution, node
        deletion, and node insertion, respectively.

        The functions will be called like

           node_subst_cost(G1.nodes[n1], G2.nodes[n2]),
           node_del_cost(G1.nodes[n1]),
           node_ins_cost(G2.nodes[n2]).

        That is, the functions will receive the node attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function node_subst_cost overrides node_match if specified.
        If neither node_match nor node_subst_cost are specified then
        default node substitution cost of 0 is used (node attributes
        are not considered during matching).

        If node_del_cost is not specified then default node deletion
        cost of 1 is used.  If node_ins_cost is not specified then
        default node insertion cost of 1 is used.

    edge_subst_cost, edge_del_cost, edge_ins_cost : callable
        Functions that return the costs of edge substitution, edge
        deletion, and edge insertion, respectively.

        The functions will be called like

           edge_subst_cost(G1[u1][v1], G2[u2][v2]),
           edge_del_cost(G1[u1][v1]),
           edge_ins_cost(G2[u2][v2]).

        That is, the functions will receive the edge attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function edge_subst_cost overrides edge_match if specified.
        If neither edge_match nor edge_subst_cost are specified then
        default edge substitution cost of 0 is used (edge attributes
        are not considered during matching).

        If edge_del_cost is not specified then default edge deletion
        cost of 1 is used.  If edge_ins_cost is not specified then
        default edge insertion cost of 1 is used.

    upper_bound : numeric
        Maximum edit distance to consider.

    Returns
    -------
    Generator of consecutive approximations of graph edit distance.

    Examples
    --------
    >>> G1 = nx.cycle_graph(6)
    >>> G2 = nx.wheel_graph(7)
    >>> for v in nx.optimize_graph_edit_distance(G1, G2):
    ...     minv = v
    >>> minv
    7.0

    See Also
    --------
    graph_edit_distance, optimize_edit_paths

    References
    ----------
    .. [1] Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, Patrick
       Martineau. An Exact Graph Edit Distance Algorithm for Solving
       Pattern Recognition Problems. 4th International Conference on
       Pattern Recognition Applications and Methods 2015, Jan 2015,
       Lisbon, Portugal. 2015,
       <10.5220/0005209202710278>. <hal-01168816>
       https://hal.archives-ouvertes.fr/hal-01168816
    """
    for _, _, cost in optimize_edit_paths(
        G1,
        G2,
        node_match,
        edge_match,
        node_subst_cost,
        node_del_cost,
        node_ins_cost,
        edge_subst_cost,
        edge_del_cost,
        edge_ins_cost,
        upper_bound,
        True,
    ):
        yield cost


@nx._dispatchable(
    graphs={"G1": 0, "G2": 1}, preserve_edge_attrs=True, preserve_node_attrs=True
)
def optimize_edit_paths(
    G1,
    G2,
    node_match=None,
    edge_match=None,
    node_subst_cost=None,
    node_del_cost=None,
    node_ins_cost=None,
    edge_subst_cost=None,
    edge_del_cost=None,
    edge_ins_cost=None,
    upper_bound=None,
    strictly_decreasing=True,
    roots=None,
    timeout=None,
):
    """GED (graph edit distance) calculation: advanced interface.

    Graph edit path is a sequence of node and edge edit operations
    transforming graph G1 to graph isomorphic to G2.  Edit operations
    include substitutions, deletions, and insertions.

    Graph edit distance is defined as minimum cost of edit path.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be of the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2
        should be considered equal during matching.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute
        dictionaries for n1 and n2 as inputs.

        Ignored if node_subst_cost is specified.  If neither
        node_match nor node_subst_cost are specified then node
        attributes are not considered.

    edge_match : callable
        A function that returns True if the edge attribute dictionaries
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during matching.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute
        dictionaries of the edges under consideration.

        Ignored if edge_subst_cost is specified.  If neither
        edge_match nor edge_subst_cost are specified then edge
        attributes are not considered.

    node_subst_cost, node_del_cost, node_ins_cost : callable
        Functions that return the costs of node substitution, node
        deletion, and node insertion, respectively.

        The functions will be called like

           node_subst_cost(G1.nodes[n1], G2.nodes[n2]),
           node_del_cost(G1.nodes[n1]),
           node_ins_cost(G2.nodes[n2]).

        That is, the functions will receive the node attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function node_subst_cost overrides node_match if specified.
        If neither node_match nor node_subst_cost are specified then
        default node substitution cost of 0 is used (node attributes
        are not considered during matching).

        If node_del_cost is not specified then default node deletion
        cost of 1 is used.  If node_ins_cost is not specified then
        default node insertion cost of 1 is used.

    edge_subst_cost, edge_del_cost, edge_ins_cost : callable
        Functions that return the costs of edge substitution, edge
        deletion, and edge insertion, respectively.

        The functions will be called like

           edge_subst_cost(G1[u1][v1], G2[u2][v2]),
           edge_del_cost(G1[u1][v1]),
           edge_ins_cost(G2[u2][v2]).

        That is, the functions will receive the edge attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function edge_subst_cost overrides edge_match if specified.
        If neither edge_match nor edge_subst_cost are specified then
        default edge substitution cost of 0 is used (edge attributes
        are not considered during matching).

        If edge_del_cost is not specified then default edge deletion
        cost of 1 is used.  If edge_ins_cost is not specified then
        default edge insertion cost of 1 is used.

    upper_bound : numeric
        Maximum edit distance to consider.

    strictly_decreasing : bool
        If True, return consecutive approximations of strictly
        decreasing cost.  Otherwise, return all edit paths of cost
        less than or equal to the previous minimum cost.

    roots : 2-tuple
        Tuple where first element is a node in G1 and the second
        is a node in G2.
        These nodes are forced to be matched in the comparison to
        allow comparison between rooted graphs.

    timeout : numeric
        Maximum number of seconds to execute.
        After timeout is met, the current best GED is returned.

    Returns
    -------
    Generator of tuples (node_edit_path, edge_edit_path, cost)
        node_edit_path : list of tuples (u, v)
        edge_edit_path : list of tuples ((u1, v1), (u2, v2))
        cost : numeric

    See Also
    --------
    graph_edit_distance, optimize_graph_edit_distance, optimal_edit_paths

    References
    ----------
    .. [1] Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, Patrick
       Martineau. An Exact Graph Edit Distance Algorithm for Solving
       Pattern Recognition Problems. 4th International Conference on
       Pattern Recognition Applications and Methods 2015, Jan 2015,
       Lisbon, Portugal. 2015,
       <10.5220/0005209202710278>. <hal-01168816>
       https://hal.archives-ouvertes.fr/hal-01168816

    """
    # TODO: support DiGraph

    import numpy as np
    import scipy as sp

    @dataclass
    class CostMatrix:
        C: ...
        lsa_row_ind: ...
        lsa_col_ind: ...
        ls: ...

    def make_CostMatrix(C, m, n):
        # assert(C.shape == (m + n, m + n))
        lsa_row_ind, lsa_col_ind = sp.optimize.linear_sum_assignment(C)

        # Fixup dummy assignments:
        # each substitution i<->j should have dummy assignment m+j<->n+i
        # NOTE: fast reduce of Cv relies on it
        # Create masks for substitution and dummy indices
        is_subst = (lsa_row_ind < m) & (lsa_col_ind < n)
        is_dummy = (lsa_row_ind >= m) & (lsa_col_ind >= n)

        # Map dummy assignments to the correct indices
        lsa_row_ind[is_dummy] = lsa_col_ind[is_subst] + m
        lsa_col_ind[is_dummy] = lsa_row_ind[is_subst] + n

        return CostMatrix(
            C, lsa_row_ind, lsa_col_ind, C[lsa_row_ind, lsa_col_ind].sum()
        )

    def extract_C(C, i, j, m, n):
        # assert(C.shape == (m + n, m + n))
        row_ind = [k in i or k - m in j for k in range(m + n)]
        col_ind = [k in j or k - n in i for k in range(m + n)]
        return C[row_ind, :][:, col_ind]

    def reduce_C(C, i, j, m, n):
        # assert(C.shape == (m + n, m + n))
        row_ind = [k not in i and k - m not in j for k in range(m + n)]
        col_ind = [k not in j and k - n not in i for k in range(m + n)]
        return C[row_ind, :][:, col_ind]

    def reduce_ind(ind, i):
        # assert set(ind) == set(range(len(ind)))
        rind = ind[[k not in i for k in ind]]
        for k in set(i):
            rind[rind >= k] -= 1
        return rind

    def match_edges(u, v, pending_g, pending_h, Ce, matched_uv=None):
        """
        Parameters:
            u, v: matched vertices, u=None or v=None for
               deletion/insertion
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_uv: partial vertex edit path
                list of tuples (u, v) of previously matched vertex
                    mappings u<->v, u=None or v=None for
                    deletion/insertion

        Returns:
            list of (i, j): indices of edge mappings g<->h
            localCe: local CostMatrix of edge mappings
                (basically submatrix of Ce at cross of rows i, cols j)
        """
        M = len(pending_g)
        N = len(pending_h)
        # assert Ce.C.shape == (M + N, M + N)

        # only attempt to match edges after one node match has been made
        # this will stop self-edges on the first node being automatically deleted
        # even when a substitution is the better option
        if matched_uv is None or len(matched_uv) == 0:
            g_ind = []
            h_ind = []
        else:
            g_ind = [
                i
                for i in range(M)
                if pending_g[i][:2] == (u, u)
                or any(
                    pending_g[i][:2] in ((p, u), (u, p), (p, p)) for p, q in matched_uv
                )
            ]
            h_ind = [
                j
                for j in range(N)
                if pending_h[j][:2] == (v, v)
                or any(
                    pending_h[j][:2] in ((q, v), (v, q), (q, q)) for p, q in matched_uv
                )
            ]

        m = len(g_ind)
        n = len(h_ind)

        if m or n:
            C = extract_C(Ce.C, g_ind, h_ind, M, N)
            # assert C.shape == (m + n, m + n)

            # Forbid structurally invalid matches
            # NOTE: inf remembered from Ce construction
            for k, i in enumerate(g_ind):
                g = pending_g[i][:2]
                for l, j in enumerate(h_ind):
                    h = pending_h[j][:2]
                    if nx.is_directed(G1) or nx.is_directed(G2):
                        if any(
                            g == (p, u) and h == (q, v) or g == (u, p) and h == (v, q)
                            for p, q in matched_uv
                        ):
                            continue
                    else:
                        if any(
                            g in ((p, u), (u, p)) and h in ((q, v), (v, q))
                            for p, q in matched_uv
                        ):
                            continue
                    if g == (u, u) or any(g == (p, p) for p, q in matched_uv):
                        continue
                    if h == (v, v) or any(h == (q, q) for p, q in matched_uv):
