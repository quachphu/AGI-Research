    .. plot::

        >>> nx.draw(nx.circular_ladder_graph(5))

    """
    G = ladder_graph(n, create_using)
    G.add_edge(0, n - 1)
    G.add_edge(n, 2 * n - 1)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def circulant_graph(n, offsets, create_using=None):
    r"""Returns the circulant graph $Ci_n(x_1, x_2, ..., x_m)$ with $n$ nodes.

    The circulant graph $Ci_n(x_1, ..., x_m)$ consists of $n$ nodes $0, ..., n-1$
    such that node $i$ is connected to nodes $(i + x) \mod n$ and $(i - x) \mod n$
    for all $x$ in $x_1, ..., x_m$. Thus $Ci_n(1)$ is a cycle graph.

    .. plot::

        >>> nx.draw(nx.circulant_graph(10, [1]))

    Parameters
    ----------
    n : integer
        The number of nodes in the graph.
    offsets : list of integers
        A list of node offsets, $x_1$ up to $x_m$, as described above.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX Graph of type create_using

    Examples
    --------
    Many well-known graph families are subfamilies of the circulant graphs;
    for example, to create the cycle graph on n points, we connect every
    node to nodes on either side (with offset plus or minus one). For n = 10,

    >>> G = nx.circulant_graph(10, [1])
    >>> edges = [
    ...     (0, 9),
    ...     (0, 1),
    ...     (1, 2),
    ...     (2, 3),
    ...     (3, 4),
    ...     (4, 5),
    ...     (5, 6),
    ...     (6, 7),
    ...     (7, 8),
    ...     (8, 9),
    ... ]
    >>> sorted(edges) == sorted(G.edges())
    True

    Similarly, we can create the complete graph
    on 5 points with the set of offsets [1, 2]:

    >>> G = nx.circulant_graph(5, [1, 2])
    >>> edges = [
    ...     (0, 1),
    ...     (0, 2),
    ...     (0, 3),
    ...     (0, 4),
    ...     (1, 2),
    ...     (1, 3),
    ...     (1, 4),
    ...     (2, 3),
    ...     (2, 4),
    ...     (3, 4),
    ... ]
    >>> sorted(edges) == sorted(G.edges())
    True

    """
    G = empty_graph(n, create_using)
    for i in range(n):
        for j in offsets:
            G.add_edge(i, (i - j) % n)
            G.add_edge(i, (i + j) % n)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number(0)
def cycle_graph(n, create_using=None):
    """Returns the cycle graph $C_n$ of cyclically connected nodes.

    $C_n$ is a path with its two end-nodes connected.

    .. plot::

        >>> nx.draw(nx.cycle_graph(5))

    Parameters
    ----------
    n : int or iterable container of nodes
        If n is an integer, nodes are from `range(n)`.
        If n is a container of nodes, those nodes appear in the graph.
        Warning: n is not checked for duplicates and if present the
        resulting graph may not be as desired. Make sure you have no duplicates.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    If create_using is directed, the direction is in increasing order.

    """
    _, nodes = n
    G = empty_graph(nodes, create_using)
    G.add_edges_from(pairwise(nodes, cyclic=True))
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def dorogovtsev_goltsev_mendes_graph(n, create_using=None):
    """Returns the hierarchically constructed Dorogovtsev--Goltsev--Mendes graph.

    The Dorogovtsev--Goltsev--Mendes [1]_ procedure deterministically produces a
    scale-free graph with ``3/2 * (3**(n-1) + 1)`` nodes
    and ``3**n`` edges for a given `n`.

    Note that `n` denotes the number of times the state transition is applied,
    starting from the base graph with ``n = 0`` (no transitions), as in [2]_.
    This is different from the parameter ``t = n - 1`` in [1]_.

    .. plot::

        >>> nx.draw(nx.dorogovtsev_goltsev_mendes_graph(3))

    Parameters
    ----------
    n : integer
        The generation number.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. Directed graphs and multigraphs are not supported.

    Returns
    -------
    G : NetworkX `Graph`

    Raises
    ------
    NetworkXError
        If `n` is less than zero.

        If `create_using` is a directed graph or multigraph.

    Examples
    --------
    >>> G = nx.dorogovtsev_goltsev_mendes_graph(3)
    >>> G.number_of_nodes()
    15
    >>> G.number_of_edges()
    27
    >>> nx.is_planar(G)
    True

    References
    ----------
    .. [1] S. N. Dorogovtsev, A. V. Goltsev and J. F. F. Mendes,
        "Pseudofractal scale-free web", Physical Review E 65, 066122, 2002.
        https://arxiv.org/pdf/cond-mat/0112143.pdf
    .. [2] Weisstein, Eric W. "Dorogovtsev--Goltsev--Mendes Graph".
        From MathWorld--A Wolfram Web Resource.
        https://mathworld.wolfram.com/Dorogovtsev-Goltsev-MendesGraph.html
    """
    if n < 0:
        raise NetworkXError("n must be greater than or equal to 0")

    G = empty_graph(0, create_using)
    if G.is_directed():
        raise NetworkXError("directed graph not supported")
    if G.is_multigraph():
        raise NetworkXError("multigraph not supported")

    G.add_edge(0, 1)
    new_node = 2  # next node to be added
    for _ in range(n):  # iterate over number of generations.
        new_edges = []
        for u, v in G.edges():
            new_edges.append((u, new_node))
            new_edges.append((v, new_node))
            new_node += 1

        G.add_edges_from(new_edges)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number(0)
def empty_graph(n=0, create_using=None, default=Graph):
    """Returns the empty graph with n nodes and zero edges.

    .. plot::

        >>> nx.draw(nx.empty_graph(5))

    Parameters
    ----------
    n : int or iterable container of nodes (default = 0)
        If n is an integer, nodes are from `range(n)`.
        If n is a container of nodes, those nodes appear in the graph.
    create_using : Graph Instance, Constructor or None
        Indicator of type of graph to return.
        If a Graph-type instance, then clear and use it.
        If None, use the `default` constructor.
        If a constructor, call it to create an empty graph.
    default : Graph constructor (optional, default = nx.Graph)
        The constructor to use if create_using is None.
        If None, then nx.Graph is used.
        This is used when passing an unknown `create_using` value
        through your home-grown function to `empty_graph` and
        you want a default constructor other than nx.Graph.

    Examples
    --------
    >>> G = nx.empty_graph(10)
    >>> G.number_of_nodes()
    10
    >>> G.number_of_edges()
    0
    >>> G = nx.empty_graph("ABC")
    >>> G.number_of_nodes()
    3
    >>> sorted(G)
    ['A', 'B', 'C']

    Notes
    -----
    The variable create_using should be a Graph Constructor or a
    "graph"-like object. Constructors, e.g. `nx.Graph` or `nx.MultiGraph`
    will be used to create the returned graph. "graph"-like objects
    will be cleared (nodes and edges will be removed) and refitted as
    an empty "graph" with nodes specified in n. This capability
    is useful for specifying the class-nature of the resulting empty
    "graph" (i.e. Graph, DiGraph, MyWeirdGraphClass, etc.).

    The variable create_using has three main uses:
    Firstly, the variable create_using can be used to create an
    empty digraph, multigraph, etc.  For example,

    >>> n = 10
    >>> G = nx.empty_graph(n, create_using=nx.DiGraph)

    will create an empty digraph on n nodes.

    Secondly, one can pass an existing graph (digraph, multigraph,
    etc.) via create_using. For example, if G is an existing graph
    (resp. digraph, multigraph, etc.), then empty_graph(n, create_using=G)
    will empty G (i.e. delete all nodes and edges using G.clear())
    and then add n nodes and zero edges, and return the modified graph.

    Thirdly, when constructing your home-grown graph creation function
    you can use empty_graph to construct the graph by passing a user
    defined create_using to empty_graph. In this case, if you want the
    default constructor to be other than nx.Graph, specify `default`.

    >>> def mygraph(n, create_using=None):
    ...     G = nx.empty_graph(n, create_using, nx.MultiGraph)
    ...     G.add_edges_from([(0, 1), (0, 1)])
    ...     return G
    >>> G = mygraph(3)
    >>> G.is_multigraph()
    True
    >>> G = mygraph(3, nx.Graph)
    >>> G.is_multigraph()
    False

    See also create_empty_copy(G).

    """
    if create_using is None:
        G = default()
    elif isinstance(create_using, type):
        G = create_using()
    elif not hasattr(create_using, "adj"):
        raise TypeError("create_using is not a valid NetworkX graph type or instance")
    else:
        # create_using is a NetworkX style Graph
        create_using.clear()
        G = create_using

    _, nodes = n
    G.add_nodes_from(nodes)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def ladder_graph(n, create_using=None):
    """Returns the Ladder graph of length n.

    This is two paths of n nodes, with
    each pair connected by a single edge.

    Node labels are the integers 0 to 2*n - 1.

    .. plot::

        >>> nx.draw(nx.ladder_graph(5))

    """
    G = empty_graph(2 * n, create_using)
    if G.is_directed():
        raise NetworkXError("Directed Graph not supported")
    G.add_edges_from(pairwise(range(n)))
    G.add_edges_from(pairwise(range(n, 2 * n)))
    G.add_edges_from((v, v + n) for v in range(n))
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number([0, 1])
def lollipop_graph(m, n, create_using=None):
    """Returns the Lollipop Graph; ``K_m`` connected to ``P_n``.

    This is the Barbell Graph without the right barbell.

    .. plot::

        >>> nx.draw(nx.lollipop_graph(3, 4))

    Parameters
    ----------
    m, n : int or iterable container of nodes
        If an integer, nodes are from ``range(m)`` and ``range(m, m+n)``.
        If a container of nodes, those nodes appear in the graph.
        Warning: `m` and `n` are not checked for duplicates and if present the
        resulting graph may not be as desired. Make sure you have no duplicates.

        The nodes for `m` appear in the complete graph $K_m$ and the nodes
        for `n` appear in the path $P_n$
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    Networkx graph
       A complete graph with `m` nodes connected to a path of length `n`.

    Notes
    -----
    The 2 subgraphs are joined via an edge ``(m-1, m)``.
    If ``n=0``, this is merely a complete graph.

    (This graph is an extremal example in David Aldous and Jim
    Fill's etext on Random Walks on Graphs.)

    """
    m, m_nodes = m
    M = len(m_nodes)
    if M < 2:
        raise NetworkXError("Invalid description: m should indicate at least 2 nodes")

    n, n_nodes = n
    if isinstance(m, numbers.Integral) and isinstance(n, numbers.Integral):
        n_nodes = list(range(M, M + n))
    N = len(n_nodes)

    # the ball
    G = complete_graph(m_nodes, create_using)
    if G.is_directed():
        raise NetworkXError("Directed Graph not supported")

    # the stick
    G.add_nodes_from(n_nodes)
    if N > 1:
        G.add_edges_from(pairwise(n_nodes))

    if len(G) != M + N:
        raise NetworkXError("Nodes must be distinct in containers m and n")

    # connect ball to stick
    if M > 0 and N > 0:
        G.add_edge(m_nodes[-1], n_nodes[0])
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def null_graph(create_using=None):
    """Returns the Null graph with no nodes or edges.

    See empty_graph for the use of create_using.

    """
    G = empty_graph(0, create_using)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number(0)
def path_graph(n, create_using=None):
    """Returns the Path graph `P_n` of linearly connected nodes.

    .. plot::

        >>> nx.draw(nx.path_graph(5))

    Parameters
    ----------
    n : int or iterable
        If an integer, nodes are 0 to n - 1.
        If an iterable of nodes, in the order they appear in the path.
        Warning: n is not checked for duplicates and if present the
        resulting graph may not be as desired. Make sure you have no duplicates.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    """
    _, nodes = n
    G = empty_graph(nodes, create_using)
    G.add_edges_from(pairwise(nodes))
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number(0)
def star_graph(n, create_using=None):
    """Return the star graph

    The star graph consists of one center node connected to n outer nodes.

    .. plot::

        >>> nx.draw(nx.star_graph(6))

    Parameters
    ----------
    n : int or iterable
        If an integer, node labels are 0 to n with center 0.
        If an iterable of nodes, the center is the first.
        Warning: n is not checked for duplicates and if present the
        resulting graph may not be as desired. Make sure you have no duplicates.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    The graph has n+1 nodes for integer n.
    So star_graph(3) is the same as star_graph(range(4)).
    """
    n, nodes = n
    if isinstance(n, numbers.Integral):
        nodes.append(int(n))  # there should be n+1 nodes
    G = empty_graph(nodes, create_using)
    if G.is_directed():
        raise NetworkXError("Directed Graph not supported")

    if len(nodes) > 1:
        hub, *spokes = nodes
        G.add_edges_from((hub, node) for node in spokes)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number([0, 1])
def tadpole_graph(m, n, create_using=None):
    """Returns the (m,n)-tadpole graph; ``C_m`` connected to ``P_n``.

    This graph on m+n nodes connects a cycle of size `m` to a path of length `n`.
    It looks like a tadpole. It is also called a kite graph or a dragon graph.

    .. plot::

        >>> nx.draw(nx.tadpole_graph(3, 5))

    Parameters
    ----------
    m, n : int or iterable container of nodes
        If an integer, nodes are from ``range(m)`` and ``range(m,m+n)``.
        If a container of nodes, those nodes appear in the graph.
        Warning: `m` and `n` are not checked for duplicates and if present the
        resulting graph may not be as desired.

        The nodes for `m` appear in the cycle graph $C_m$ and the nodes
        for `n` appear in the path $P_n$.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    Networkx graph
       A cycle of size `m` connected to a path of length `n`.

    Raises
    ------
    NetworkXError
        If ``m < 2``. The tadpole graph is undefined for ``m<2``.

    Notes
    -----
    The 2 subgraphs are joined via an edge ``(m-1, m)``.
    If ``n=0``, this is a cycle graph.
    `m` and/or `n` can be a container of nodes instead of an integer.

    """
    m, m_nodes = m
    M = len(m_nodes)
    if M < 2:
        raise NetworkXError("Invalid description: m should indicate at least 2 nodes")

    n, n_nodes = n
    if isinstance(m, numbers.Integral) and isinstance(n, numbers.Integral):
        n_nodes = list(range(M, M + n))

    # the circle
    G = cycle_graph(m_nodes, create_using)
    if G.is_directed():
        raise NetworkXError("Directed Graph not supported")

    # the stick
    nx.add_path(G, [m_nodes[-1]] + list(n_nodes))

    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def trivial_graph(create_using=None):
    """Return the Trivial graph with one node (with label 0) and no edges.

    .. plot::

        >>> nx.draw(nx.trivial_graph(), with_labels=True)

    """
    G = empty_graph(1, create_using)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def turan_graph(n, r):
    r"""Return the Turan Graph

    The Turan Graph is a complete multipartite graph on $n$ nodes
    with $r$ disjoint subsets. That is, edges connect each node to
    every node not in its subset.

    Given $n$ and $r$, we create a complete multipartite graph with
    $r-(n \mod r)$ partitions of size $n/r$, rounded down, and
    $n \mod r$ partitions of size $n/r+1$, rounded down.

    .. plot::

        >>> nx.draw(nx.turan_graph(6, 2))

    Parameters
    ----------
    n : int
        The number of nodes.
    r : int
        The number of partitions.
        Must be less than or equal to n.

    Notes
    -----
    Must satisfy $1 <= r <= n$.
    The graph has $(r-1)(n^2)/(2r)$ edges, rounded down.
    """

    if not 1 <= r <= n:
        raise NetworkXError("Must satisfy 1 <= r <= n")

    partitions = [n // r] * (r - (n % r)) + [n // r + 1] * (n % r)
    G = complete_multipartite_graph(*partitions)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number(0)
def wheel_graph(n, create_using=None):
    """Return the wheel graph

    The wheel graph consists of a hub node connected to a cycle of (n-1) nodes.

    .. plot::

        >>> nx.draw(nx.wheel_graph(5))

    Parameters
    ----------
    n : int or iterable
        If an integer, node labels are 0 to n with center 0.
        If an iterable of nodes, the center is the first.
        Warning: n is not checked for duplicates and if present the
        resulting graph may not be as desired. Make sure you have no duplicates.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Node labels are the integers 0 to n - 1.
    """
    _, nodes = n
    G = empty_graph(nodes, create_using)
    if G.is_directed():
        raise NetworkXError("Directed Graph not supported")

    if len(nodes) > 1:
        hub, *rim = nodes
        G.add_edges_from((hub, node) for node in rim)
        if len(rim) > 1:
            G.add_edges_from(pairwise(rim, cyclic=True))
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def complete_multipartite_graph(*subset_sizes):
    """Returns the complete multipartite graph with the specified subset sizes.

    .. plot::

        >>> nx.draw(nx.complete_multipartite_graph(1, 2, 3))

    Parameters
    ----------
    subset_sizes : tuple of integers or tuple of node iterables
       The arguments can either all be integer number of nodes or they
       can all be iterables of nodes. If integers, they represent the
       number of nodes in each subset of the multipartite graph.
       If iterables, each is used to create the nodes for that subset.
       The length of subset_sizes is the number of subsets.

    Returns
    -------
    G : NetworkX Graph
       Returns the complete multipartite graph with the specified subsets.

       For each node, the node attribute 'subset' is an integer
       indicating which subset contains the node.

    Examples
    --------
    Creating a complete tripartite graph, with subsets of one, two, and three
    nodes, respectively.

        >>> G = nx.complete_multipartite_graph(1, 2, 3)
        >>> [G.nodes[u]["subset"] for u in G]
        [0, 1, 1, 2, 2, 2]
        >>> list(G.edges(0))
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        >>> list(G.edges(2))
        [(2, 0), (2, 3), (2, 4), (2, 5)]
        >>> list(G.edges(4))
        [(4, 0), (4, 1), (4, 2)]

        >>> G = nx.complete_multipartite_graph("a", "bc", "def")
        >>> [G.nodes[u]["subset"] for u in sorted(G)]
        [0, 1, 1, 2, 2, 2]

    Notes
    -----
    This function generalizes several other graph builder functions.

    - If no subset sizes are given, this returns the null graph.
    - If a single subset size `n` is given, this returns the empty graph on
      `n` nodes.
    - If two subset sizes `m` and `n` are given, this returns the complete
      bipartite graph on `m + n` nodes.
    - If subset sizes `1` and `n` are given, this returns the star graph on
      `n + 1` nodes.

    See also
    --------
    complete_bipartite_graph
    """
    # The complete multipartite graph is an undirected simple graph.
    G = Graph()

    if len(subset_sizes) == 0:
        return G

    # set up subsets of nodes
    try:
        extents = pairwise(itertools.accumulate((0,) + subset_sizes))
        subsets = [range(start, end) for start, end in extents]
    except TypeError:
        subsets = subset_sizes
    else:
        if any(size < 0 for size in subset_sizes):
            raise NetworkXError(f"Negative number of nodes not valid: {subset_sizes}")

    # add nodes with subset attribute
    # while checking that ints are not mixed with iterables
    try:
        for i, subset in enumerate(subsets):
            G.add_nodes_from(subset, subset=i)
    except TypeError as err:
        raise NetworkXError("Arguments must be all ints or all iterables") from err

    # Across subsets, all nodes should be adjacent.
    # We can use itertools.combinations() because undirected.
    for subset1, subset2 in itertools.combinations(subsets, 2):
        G.add_edges_from(itertools.product(subset1, subset2))
    return G
<|endoftext|>"""
Generators for the small graph atlas.
"""

import gzip
import importlib.resources
import os
import os.path
from itertools import islice

import networkx as nx

__all__ = ["graph_atlas", "graph_atlas_g"]

#: The total number of graphs in the atlas.
#:
#: The graphs are labeled starting from 0 and extending to (but not
#: including) this number.
NUM_GRAPHS = 1253

#: The path to the data file containing the graph edge lists.
#:
#: This is the absolute path of the gzipped text file containing the
#: edge list for each graph in the atlas. The file contains one entry
#: per graph in the atlas, in sequential order, starting from graph
#: number 0 and extending through graph number 1252 (see
#: :data:`NUM_GRAPHS`). Each entry looks like
#:
#: .. sourcecode:: text
#:
#:    GRAPH 6
#:    NODES 3
#:    0 1
#:    0 2
#:
#: where the first two lines are the graph's index in the atlas and the
#: number of nodes in the graph, and the remaining lines are the edge
#: list.
#:
#: This file was generated from a Python list of graphs via code like
#: the following::
#:
#:     import gzip
#:     from networkx.generators.atlas import graph_atlas_g
#:     from networkx.readwrite.edgelist import write_edgelist
#:
#:     with gzip.open('atlas.dat.gz', 'wb') as f:
#:         for i, G in enumerate(graph_atlas_g()):
#:             f.write(bytes(f'GRAPH {i}\n', encoding='utf-8'))
#:             f.write(bytes(f'NODES {len(G)}\n', encoding='utf-8'))
#:             write_edgelist(G, f, data=False)
#:

# Path to the atlas file
ATLAS_FILE = importlib.resources.files("networkx.generators") / "atlas.dat.gz"


def _generate_graphs():
    """Sequentially read the file containing the edge list data for the
    graphs in the atlas and generate the graphs one at a time.

    This function reads the file given in :data:`.ATLAS_FILE`.

    """
    with gzip.open(ATLAS_FILE, "rb") as f:
        line = f.readline()
        while line and line.startswith(b"GRAPH"):
            # The first two lines of each entry tell us the index of the
            # graph in the list and the number of nodes in the graph.
            # They look like this:
            #
            #     GRAPH 3
            #     NODES 2
            #
            graph_index = int(line[6:].rstrip())
            line = f.readline()
            num_nodes = int(line[6:].rstrip())
            # The remaining lines contain the edge list, until the next
            # GRAPH line (or until the end of the file).
            edgelist = []
            line = f.readline()
            while line and not line.startswith(b"GRAPH"):
                edgelist.append(line.rstrip())
                line = f.readline()
            G = nx.Graph()
            G.name = f"G{graph_index}"
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(tuple(map(int, e.split())) for e in edgelist)
            yield G


@nx._dispatchable(graphs=None, returns_graph=True)
def graph_atlas(i):
    """Returns graph number `i` from the Graph Atlas.

    For more information, see :func:`.graph_atlas_g`.

    Parameters
    ----------
    i : int
        The index of the graph from the atlas to get. The graph at index
        0 is assumed to be the null graph.

    Returns
    -------
    list
        A list of :class:`~networkx.Graph` objects, the one at index *i*
        corresponding to the graph *i* in the Graph Atlas.

    See also
    --------
    graph_atlas_g

    Notes
    -----
    The time required by this function increases linearly with the
    argument `i`, since it reads a large file sequentially in order to
    generate the graph [1]_.

    References
    ----------
    .. [1] Ronald C. Read and Robin J. Wilson, *An Atlas of Graphs*.
           Oxford University Press, 1998.

    """
    if not (0 <= i < NUM_GRAPHS):
        raise ValueError(f"index must be between 0 and {NUM_GRAPHS}")
    return next(islice(_generate_graphs(), i, None))


@nx._dispatchable(graphs=None, returns_graph=True)
def graph_atlas_g():
    """Returns the list of all graphs with up to seven nodes named in the
    Graph Atlas.

    The graphs are listed in increasing order by

    1. number of nodes,
    2. number of edges,
    3. degree sequence (for example 111223 < 112222),
    4. number of automorphisms,

    in that order, with three exceptions as described in the *Notes*
    section below. This causes the list to correspond with the index of
    the graphs in the Graph Atlas [atlas]_, with the first graph,
    ``G[0]``, being the null graph.

    Returns
    -------
    list
        A list of :class:`~networkx.Graph` objects, the one at index *i*
        corresponding to the graph *i* in the Graph Atlas.

    See also
    --------
    graph_atlas

    Notes
    -----
    This function may be expensive in both time and space, since it
    reads a large file sequentially in order to populate the list.

    Although the NetworkX atlas functions match the order of graphs
    given in the "Atlas of Graphs" book, there are (at least) three
    errors in the ordering described in the book. The following three
    pairs of nodes violate the lexicographically nondecreasing sorted
    degree sequence rule:

    - graphs 55 and 56 with degree sequences 001111 and 000112,
    - graphs 1007 and 1008 with degree sequences 3333444 and 3333336,
    - graphs 1012 and 1213 with degree sequences 1244555 and 1244456.

    References
    ----------
    .. [atlas] Ronald C. Read and Robin J. Wilson,
               *An Atlas of Graphs*.
               Oxford University Press, 1998.

    """
    return list(_generate_graphs())
<|endoftext|>"""Generate graphs with a given degree sequence or expected degree sequence."""

import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter

import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample

__all__ = [
    "configuration_model",
    "directed_configuration_model",
    "expected_degree_graph",
    "havel_hakimi_graph",
    "directed_havel_hakimi_graph",
    "degree_sequence_tree",
    "random_degree_sequence_graph",
]

chaini = chain.from_iterable


def _to_stublist(degree_sequence):
    """Returns a list of degree-repeated node numbers.

    ``degree_sequence`` is a list of nonnegative integers representing
    the degrees of nodes in a graph.

    This function returns a list of node numbers with multiplicities
    according to the given degree sequence. For example, if the first
    element of ``degree_sequence`` is ``3``, then the first node number,
    ``0``, will appear at the head of the returned list three times. The
    node numbers are assumed to be the numbers zero through
    ``len(degree_sequence) - 1``.

    Examples
    --------

    >>> degree_sequence = [1, 2, 3]
    >>> _to_stublist(degree_sequence)
    [0, 1, 1, 2, 2, 2]

    If a zero appears in the sequence, that means the node exists but
    has degree zero, so that number will be skipped in the returned
    list::

    >>> degree_sequence = [2, 0, 1]
    >>> _to_stublist(degree_sequence)
    [0, 0, 2]

    """
    return list(chaini([n] * d for n, d in enumerate(degree_sequence)))


def _configuration_model(
    deg_sequence, create_using, directed=False, in_deg_sequence=None, seed=None
):
    """Helper function for generating either undirected or directed
    configuration model graphs.

    ``deg_sequence`` is a list of nonnegative integers representing the
    degree of the node whose label is the index of the list element.

    ``create_using`` see :func:`~networkx.empty_graph`.

    ``directed`` and ``in_deg_sequence`` are required if you want the
    returned graph to be generated using the directed configuration
    model algorithm. If ``directed`` is ``False``, then ``deg_sequence``
    is interpreted as the degree sequence of an undirected graph and
    ``in_deg_sequence`` is ignored. Otherwise, if ``directed`` is
    ``True``, then ``deg_sequence`` is interpreted as the out-degree
    sequence and ``in_deg_sequence`` as the in-degree sequence of a
    directed graph.

    .. note::

       ``deg_sequence`` and ``in_deg_sequence`` need not be the same
       length.

    ``seed`` is a random.Random or numpy.random.RandomState instance

    This function returns a graph, directed if and only if ``directed``
    is ``True``, generated according to the configuration model
    algorithm. For more information on the algorithm, see the
    :func:`configuration_model` or :func:`directed_configuration_model`
    functions.

    """
    n = len(deg_sequence)
    G = nx.empty_graph(n, create_using)
    # If empty, return the null graph immediately.
    if n == 0:
        return G
    # Build a list of available degree-repeated nodes.  For example,
    # for degree sequence [3, 2, 1, 1, 1], the "stub list" is
    # initially [0, 0, 0, 1, 1, 2, 3, 4], that is, node 0 has degree
    # 3 and thus is repeated 3 times, etc.
    #
    # Also, shuffle the stub list in order to get a random sequence of
    # node pairs.
    if directed:
        pairs = zip_longest(deg_sequence, in_deg_sequence, fillvalue=0)
        # Unzip the list of pairs into a pair of lists.
        out_deg, in_deg = zip(*pairs)

        out_stublist = _to_stublist(out_deg)
        in_stublist = _to_stublist(in_deg)

        seed.shuffle(out_stublist)
        seed.shuffle(in_stublist)
    else:
        stublist = _to_stublist(deg_sequence)
        # Choose a random balanced bipartition of the stublist, which
        # gives a random pairing of nodes. In this implementation, we
        # shuffle the list and then split it in half.
        n = len(stublist)
        half = n // 2
        seed.shuffle(stublist)
        out_stublist, in_stublist = stublist[:half], stublist[half:]
    G.add_edges_from(zip(out_stublist, in_stublist))
    return G


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def configuration_model(deg_sequence, create_using=None, seed=None):
    """Returns a random graph with the given degree sequence.

    The configuration model generates a random pseudograph (graph with
    parallel edges and self loops) by randomly assigning edges to
    match the given degree sequence.

    Parameters
    ----------
    deg_sequence :  list of nonnegative integers
        Each list entry corresponds to the degree of a node.
    create_using : NetworkX graph constructor, optional (default MultiGraph)
        Graph type to create. If graph instance, then cleared before populated.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : MultiGraph
        A graph with the specified degree sequence.
        Nodes are labeled starting at 0 with an index
        corresponding to the position in deg_sequence.

    Raises
    ------
    NetworkXError
        If the degree sequence does not have an even sum.

    See Also
    --------
    is_graphical

    Notes
    -----
    As described by Newman [1]_.

    A non-graphical degree sequence (not realizable by some simple
    graph) is allowed since this function returns graphs with self
    loops and parallel edges.  An exception is raised if the degree
    sequence does not have an even sum.

    This configuration model construction process can lead to
    duplicate edges and loops.  You can remove the self-loops and
    parallel edges (see below) which will likely result in a graph
    that doesn't have the exact degree sequence specified.

    The density of self-loops and parallel edges tends to decrease as
    the number of nodes increases. However, typically the number of
    self-loops will approach a Poisson distribution with a nonzero mean,
    and similarly for the number of parallel edges.  Consider a node
    with *k* stubs. The probability of being joined to another stub of
    the same node is basically (*k* - *1*) / *N*, where *k* is the
    degree and *N* is the number of nodes. So the probability of a
    self-loop scales like *c* / *N* for some constant *c*. As *N* grows,
    this means we expect *c* self-loops. Similarly for parallel edges.

    References
    ----------
    .. [1] M.E.J. Newman, "The structure and function of complex networks",
       SIAM REVIEW 45-2, pp 167-256, 2003.

    Examples
    --------
    You can create a degree sequence following a particular distribution
    by using the one of the distribution functions in
    :mod:`~networkx.utils.random_sequence` (or one of your own). For
    example, to create an undirected multigraph on one hundred nodes
    with degree sequence chosen from the power law distribution:

    >>> sequence = nx.random_powerlaw_tree_sequence(100, tries=5000)
    >>> G = nx.configuration_model(sequence)
    >>> len(G)
    100
    >>> actual_degrees = [d for v, d in G.degree()]
    >>> actual_degrees == sequence
    True

    The returned graph is a multigraph, which may have parallel
    edges. To remove any parallel edges from the returned graph:

    >>> G = nx.Graph(G)

    Similarly, to remove self-loops:

    >>> G.remove_edges_from(nx.selfloop_edges(G))

    """
    if sum(deg_sequence) % 2 != 0:
        msg = "Invalid degree sequence: sum of degrees must be even, not odd"
        raise nx.NetworkXError(msg)

    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for directed graphs")

    G = _configuration_model(deg_sequence, G, seed=seed)

    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def directed_configuration_model(
    in_degree_sequence, out_degree_sequence, create_using=None, seed=None
):
    """Returns a directed_random graph with the given degree sequences.

    The configuration model generates a random directed pseudograph
    (graph with parallel edges and self loops) by randomly assigning
    edges to match the given degree sequences.

    Parameters
    ----------
    in_degree_sequence :  list of nonnegative integers
       Each list entry corresponds to the in-degree of a node.
    out_degree_sequence :  list of nonnegative integers
       Each list entry corresponds to the out-degree of a node.
    create_using : NetworkX graph constructor, optional (default MultiDiGraph)
        Graph type to create. If graph instance, then cleared before populated.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : MultiDiGraph
        A graph with the specified degree sequences.
        Nodes are labeled starting at 0 with an index
        corresponding to the position in deg_sequence.

    Raises
    ------
    NetworkXError
        If the degree sequences do not have the same sum.

    See Also
    --------
    configuration_model

    Notes
    -----
    Algorithm as described by Newman [1]_.

    A non-graphical degree sequence (not realizable by some simple
    graph) is allowed since this function returns graphs with self
    loops and parallel edges.  An exception is raised if the degree
    sequences does not have the same sum.

    This configuration model construction process can lead to
    duplicate edges and loops.  You can remove the self-loops and
    parallel edges (see below) which will likely result in a graph
    that doesn't have the exact degree sequence specified.  This
    "finite-size effect" decreases as the size of the graph increases.

    References
    ----------
    .. [1] Newman, M. E. J. and Strogatz, S. H. and Watts, D. J.
       Random graphs with arbitrary degree distributions and their applications
       Phys. Rev. E, 64, 026118 (2001)

    Examples
    --------
    One can modify the in- and out-degree sequences from an existing
    directed graph in order to create a new directed graph. For example,
    here we modify the directed path graph:

    >>> D = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    >>> din = list(d for n, d in D.in_degree())
    >>> dout = list(d for n, d in D.out_degree())
    >>> din.append(1)
    >>> dout[0] = 2
    >>> # We now expect an edge from node 0 to a new node, node 3.
    ... D = nx.directed_configuration_model(din, dout)

    The returned graph is a directed multigraph, which may have parallel
    edges. To remove any parallel edges from the returned graph:

    >>> D = nx.DiGraph(D)

    Similarly, to remove self-loops:

    >>> D.remove_edges_from(nx.selfloop_edges(D))

    """
    if sum(in_degree_sequence) != sum(out_degree_sequence):
        msg = "Invalid degree sequences: sequences must have equal sums"
        raise nx.NetworkXError(msg)

    if create_using is None:
        create_using = nx.MultiDiGraph

    G = _configuration_model(
        out_degree_sequence,
        create_using,
        directed=True,
        in_deg_sequence=in_degree_sequence,
        seed=seed,
    )

    name = "directed configuration_model {} nodes {} edges"
    return G


@py_random_state(1)
@nx._dispatchable(graphs=None, returns_graph=True)
def expected_degree_graph(w, seed=None, selfloops=True):
    r"""Returns a random graph with given expected degrees.

    Given a sequence of expected degrees $W=(w_0,w_1,\ldots,w_{n-1})$
    of length $n$ this algorithm assigns an edge between node $u$ and
    node $v$ with probability

    .. math::

       p_{uv} = \frac{w_u w_v}{\sum_k w_k} .

    Parameters
    ----------
    w : list
        The list of expected degrees.
    selfloops: bool (default=True)
        Set to False to remove the possibility of self-loop edges.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    Graph

    Examples
    --------
    >>> z = [10 for i in range(100)]
    >>> G = nx.expected_degree_graph(z)

    Notes
    -----
    The nodes have integer labels corresponding to index of expected degrees
    input sequence.

    The complexity of this algorithm is $\mathcal{O}(n+m)$ where $n$ is the
    number of nodes and $m$ is the expected number of edges.

    The model in [1]_ includes the possibility of self-loop edges.
    Set selfloops=False to produce a graph without self loops.

    For finite graphs this model doesn't produce exactly the given
    expected degree sequence.  Instead the expected degrees are as
    follows.

    For the case without self loops (selfloops=False),

    .. math::

       E[deg(u)] = \sum_{v \ne u} p_{uv}
                = w_u \left( 1 - \frac{w_u}{\sum_k w_k} \right) .


    NetworkX uses the standard convention that a self-loop edge counts 2
    in the degree of a node, so with self loops (selfloops=True),

    .. math::

       E[deg(u)] =  \sum_{v \ne u} p_{uv}  + 2 p_{uu}
                = w_u \left( 1 + \frac{w_u}{\sum_k w_k} \right) .

    References
    ----------
    .. [1] Fan Chung and L. Lu, Connected components in random graphs with
       given expected degree sequences, Ann. Combinatorics, 6,
       pp. 125-145, 2002.
    .. [2] Joel Miller and Aric Hagberg,
       Efficient generation of networks with given expected degrees,
       in Algorithms and Models for the Web-Graph (WAW 2011),
       Alan Frieze, Paul Horn, and Paweł Prałat (Eds), LNCS 6732,
       pp. 115-126, 2011.
    """
    n = len(w)
    G = nx.empty_graph(n)

    # If there are no nodes are no edges in the graph, return the empty graph.
    if n == 0 or max(w) == 0:
        return G

    rho = 1 / sum(w)
    # Sort the weights in decreasing order. The original order of the
    # weights dictates the order of the (integer) node labels, so we
    # need to remember the permutation applied in the sorting.
    order = sorted(enumerate(w), key=itemgetter(1), reverse=True)
    mapping = {c: u for c, (u, v) in enumerate(order)}
    seq = [v for u, v in order]
    last = n
    if not selfloops:
        last -= 1
    for u in range(last):
        v = u
        if not selfloops:
            v += 1
        factor = seq[u] * rho
        p = min(seq[v] * factor, 1)
        while v < n and p > 0:
            if p != 1:
                r = seed.random()
                v += math.floor(math.log(r, 1 - p))
            if v < n:
                q = min(seq[v] * factor, 1)
                if seed.random() < q / p:
                    G.add_edge(mapping[u], mapping[v])
                v += 1
                p = q
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def havel_hakimi_graph(deg_sequence, create_using=None):
    """Returns a simple graph with given degree sequence constructed
    using the Havel-Hakimi algorithm.

    Parameters
    ----------
    deg_sequence: list of integers
        Each integer corresponds to the degree of a node (need not be sorted).
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Directed graphs are not allowed.

    Raises
    ------
    NetworkXException
        For a non-graphical degree sequence (i.e. one
        not realizable by some simple graph).

    Notes
    -----
    The Havel-Hakimi algorithm constructs a simple graph by
    successively connecting the node of highest degree to other nodes
    of highest degree, resorting remaining nodes by degree, and
    repeating the process. The resulting graph has a high
    degree-associativity.  Nodes are labeled 1,.., len(deg_sequence),
    corresponding to their position in deg_sequence.

    The basic algorithm is from Hakimi [1]_ and was generalized by
    Kleitman and Wang [2]_.

    References
    ----------
    .. [1] Hakimi S., On Realizability of a Set of Integers as
       Degrees of the Vertices of a Linear Graph. I,
       Journal of SIAM, 10(3), pp. 496-506 (1962)
    .. [2] Kleitman D.J. and Wang D.L.
       Algorithms for Constructing Graphs and Digraphs with Given Valences
       and Factors  Discrete Mathematics, 6(1), pp. 79-88 (1973)
    """
    if not nx.is_graphical(deg_sequence):
        raise nx.NetworkXError("Invalid degree sequence")

    p = len(deg_sequence)
    G = nx.empty_graph(p, create_using)
    if G.is_directed():
        raise nx.NetworkXError("Directed graphs are not supported")
    num_degs = [[] for i in range(p)]
    dmax, dsum, n = 0, 0, 0
    for d in deg_sequence:
        # Process only the non-zero integers
        if d > 0:
            num_degs[d].append(n)
            dmax, dsum, n = max(dmax, d), dsum + d, n + 1
    # Return graph if no edges
    if n == 0:
        return G

    modstubs = [(0, 0)] * (dmax + 1)
    # Successively reduce degree sequence by removing the maximum degree
    while n > 0:
        # Retrieve the maximum degree in the sequence
        while len(num_degs[dmax]) == 0:
            dmax -= 1
        # If there are not enough stubs to connect to, then the sequence is
        # not graphical
        if dmax > n - 1:
            raise nx.NetworkXError("Non-graphical integer sequence")

        # Remove largest stub in list
        source = num_degs[dmax].pop()
        n -= 1
        # Reduce the next dmax largest stubs
        mslen = 0
        k = dmax
        for i in range(dmax):
            while len(num_degs[k]) == 0:
                k -= 1
            target = num_degs[k].pop()
            G.add_edge(source, target)
            n -= 1
            if k > 1:
                modstubs[mslen] = (k - 1, target)
                mslen += 1
        # Add back to the list any nonzero stubs that were removed
        for i in range(mslen):
            (stubval, stubtarget) = modstubs[i]
            num_degs[stubval].append(stubtarget)
            n += 1

    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def directed_havel_hakimi_graph(in_deg_sequence, out_deg_sequence, create_using=None):
    """Returns a directed graph with the given degree sequences.

    Parameters
    ----------
    in_deg_sequence :  list of integers
        Each list entry corresponds to the in-degree of a node.
    out_deg_sequence : list of integers
        Each list entry corresponds to the out-degree of a node.
    create_using : NetworkX graph constructor, optional (default DiGraph)
        Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : DiGraph
        A graph with the specified degree sequences.
        Nodes are labeled starting at 0 with an index
        corresponding to the position in deg_sequence

    Raises
    ------
    NetworkXError
        If the degree sequences are not digraphical.

    See Also
    --------
    configuration_model

    Notes
    -----
    Algorithm as described by Kleitman and Wang [1]_.

    References
    ----------
    .. [1] D.J. Kleitman and D.L. Wang
       Algorithms for Constructing Graphs and Digraphs with Given Valences
       and Factors Discrete Mathematics, 6(1), pp. 79-88 (1973)
    """
    in_deg_sequence = nx.utils.make_list_of_ints(in_deg_sequence)
    out_deg_sequence = nx.utils.make_list_of_ints(out_deg_sequence)

    # Process the sequences and form two heaps to store degree pairs with
    # either zero or nonzero out degrees
    sumin, sumout = 0, 0
    nin, nout = len(in_deg_sequence), len(out_deg_sequence)
    maxn = max(nin, nout)
    G = nx.empty_graph(maxn, create_using, default=nx.DiGraph)
    if maxn == 0:
        return G
    maxin = 0
    stubheap, zeroheap = [], []
    for n in range(maxn):
        in_deg, out_deg = 0, 0
        if n < nout:
            out_deg = out_deg_sequence[n]
        if n < nin:
            in_deg = in_deg_sequence[n]
        if in_deg < 0 or out_deg < 0:
            raise nx.NetworkXError(
                "Invalid degree sequences. Sequence values must be positive."
            )
        sumin, sumout, maxin = sumin + in_deg, sumout + out_deg, max(maxin, in_deg)
        if in_deg > 0:
            stubheap.append((-1 * out_deg, -1 * in_deg, n))
        elif out_deg > 0:
            zeroheap.append((-1 * out_deg, n))
    if sumin != sumout:
        raise nx.NetworkXError(
            "Invalid degree sequences. Sequences must have equal sums."
        )
    heapq.heapify(stubheap)
    heapq.heapify(zeroheap)

    modstubs = [(0, 0, 0)] * (maxin + 1)
    # Successively reduce degree sequence by removing the maximum
    while stubheap:
        # Remove first value in the sequence with a non-zero in degree
        (freeout, freein, target) = heapq.heappop(stubheap)
        freein *= -1
        if freein > len(stubheap) + len(zeroheap):
            raise nx.NetworkXError("Non-digraphical integer sequence")

        # Attach arcs from the nodes with the most stubs
        mslen = 0
        for i in range(freein):
            if zeroheap and (not stubheap or stubheap[0][0] > zeroheap[0][0]):
                (stubout, stubsource) = heapq.heappop(zeroheap)
                stubin = 0
            else:
                (stubout, stubin, stubsource) = heapq.heappop(stubheap)
            if stubout == 0:
                raise nx.NetworkXError("Non-digraphical integer sequence")
            G.add_edge(stubsource, target)
            # Check if source is now totally connected
            if stubout + 1 < 0 or stubin < 0:
                modstubs[mslen] = (stubout + 1, stubin, stubsource)
                mslen += 1

        # Add the nodes back to the heaps that still have available stubs
        for i in range(mslen):
            stub = modstubs[i]
            if stub[1] < 0:
                heapq.heappush(stubheap, stub)
            else:
                heapq.heappush(zeroheap, (stub[0], stub[2]))
        if freeout < 0:
            heapq.heappush(zeroheap, (freeout, target))

    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def degree_sequence_tree(deg_sequence, create_using=None):
    """Make a tree for the given degree sequence.

    A tree has #nodes-#edges=1 so
    the degree sequence must have
    len(deg_sequence)-sum(deg_sequence)/2=1
    """
    # The sum of the degree sequence must be even (for any undirected graph).
    degree_sum = sum(deg_sequence)
    if degree_sum % 2 != 0:
        msg = "Invalid degree sequence: sum of degrees must be even, not odd"
        raise nx.NetworkXError(msg)
    if len(deg_sequence) - degree_sum // 2 != 1:
        msg = (
            "Invalid degree sequence: tree must have number of nodes equal"
            " to one less than the number of edges"
        )
        raise nx.NetworkXError(msg)
    G = nx.empty_graph(0, create_using)
    if G.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    # Sort all degrees greater than 1 in decreasing order.
    #
    # TODO Does this need to be sorted in reverse order?
    deg = sorted((s for s in deg_sequence if s > 1), reverse=True)

    # make path graph as backbone
    n = len(deg) + 2
    nx.add_path(G, range(n))
    last = n

    # add the leaves
    for source in range(1, n - 1):
        nedges = deg.pop() - 2
        for target in range(last, last + nedges):
            G.add_edge(source, target)
        last += nedges

    # in case we added one too many
    if len(G) > len(deg_sequence):
        G.remove_node(0)
    return G


@py_random_state(1)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_degree_sequence_graph(sequence, seed=None, tries=10):
    r"""Returns a simple random graph with the given degree sequence.

    If the maximum degree $d_m$ in the sequence is $O(m^{1/4})$ then the
    algorithm produces almost uniform random graphs in $O(m d_m)$ time
    where $m$ is the number of edges.

    Parameters
    ----------
    sequence :  list of integers
        Sequence of degrees
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    tries : int, optional
        Maximum number of tries to create a graph

    Returns
    -------
    G : Graph
        A graph with the specified degree sequence.
        Nodes are labeled starting at 0 with an index
        corresponding to the position in the sequence.

    Raises
    ------
    NetworkXUnfeasible
        If the degree sequence is not graphical.
    NetworkXError
        If a graph is not produced in specified number of tries

    See Also
    --------
    is_graphical, configuration_model

    Notes
    -----
    The generator algorithm [1]_ is not guaranteed to produce a graph.

    References
    ----------
    .. [1] Moshen Bayati, Jeong Han Kim, and Amin Saberi,
       A sequential algorithm for generating random graphs.
       Algorithmica, Volume 58, Number 4, 860-910,
       DOI: 10.1007/s00453-009-9340-1

    Examples
    --------
    >>> sequence = [1, 2, 2, 3]
    >>> G = nx.random_degree_sequence_graph(sequence, seed=42)
    >>> sorted(d for n, d in G.degree())
    [1, 2, 2, 3]
    """
    DSRG = DegreeSequenceRandomGraph(sequence, seed)
    for try_n in range(tries):
        try:
            return DSRG.generate()
        except nx.NetworkXUnfeasible:
            pass
    raise nx.NetworkXError(f"failed to generate graph in {tries} tries")


class DegreeSequenceRandomGraph:
    # class to generate random graphs with a given degree sequence
    # use random_degree_sequence_graph()
    def __init__(self, degree, rng):
        if not nx.is_graphical(degree):
            raise nx.NetworkXUnfeasible("degree sequence is not graphical")
        self.rng = rng
        self.degree = list(degree)
        # node labels are integers 0,...,n-1
        self.m = sum(self.degree) / 2.0  # number of edges
        try:
            self.dmax = max(self.degree)  # maximum degree
        except ValueError:
            self.dmax = 0

    def generate(self):
        # remaining_degree is mapping from int->remaining degree
        self.remaining_degree = dict(enumerate(self.degree))
        # add all nodes to make sure we get isolated nodes
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.remaining_degree)
        # remove zero degree nodes
        for n, d in list(self.remaining_degree.items()):
            if d == 0:
                del self.remaining_degree[n]
        if len(self.remaining_degree) > 0:
            # build graph in three phases according to how many unmatched edges
            self.phase1()
            self.phase2()
            self.phase3()
        return self.graph

    def update_remaining(self, u, v, aux_graph=None):
        # decrement remaining nodes, modify auxiliary graph if in phase3
        if aux_graph is not None:
            # remove edges from auxiliary graph
            aux_graph.remove_edge(u, v)
        if self.remaining_degree[u] == 1:
            del self.remaining_degree[u]
            if aux_graph is not None:
                aux_graph.remove_node(u)
        else:
            self.remaining_degree[u] -= 1
        if self.remaining_degree[v] == 1:
            del self.remaining_degree[v]
            if aux_graph is not None:
                aux_graph.remove_node(v)
        else:
            self.remaining_degree[v] -= 1

    def p(self, u, v):
        # degree probability
        return 1 - self.degree[u] * self.degree[v] / (4.0 * self.m)

    def q(self, u, v):
        # remaining degree probability
        norm = max(self.remaining_degree.values()) ** 2
        return self.remaining_degree[u] * self.remaining_degree[v] / norm

    def suitable_edge(self):
        """Returns True if and only if an arbitrary remaining node can
        potentially be joined with some other remaining node.

        """
        nodes = iter(self.remaining_degree)
        u = next(nodes)
        return any(v not in self.graph[u] for v in nodes)

    def phase1(self):
        # choose node pairs from (degree) weighted distribution
        rem_deg = self.remaining_degree
        while sum(rem_deg.values()) >= 2 * self.dmax**2:
            u, v = sorted(random_weighted_sample(rem_deg, 2, self.rng))
            if self.graph.has_edge(u, v):
                continue
            if self.rng.random() < self.p(u, v):  # accept edge
                self.graph.add_edge(u, v)
                self.update_remaining(u, v)

    def phase2(self):
        # choose remaining nodes uniformly at random and use rejection sampling
        remaining_deg = self.remaining_degree
        rng = self.rng
        while len(remaining_deg) >= 2 * self.dmax:
            while True:
                u, v = sorted(rng.sample(list(remaining_deg.keys()), 2))
                if self.graph.has_edge(u, v):
                    continue
                if rng.random() < self.q(u, v):
                    break
            if rng.random() < self.p(u, v):  # accept edge
                self.graph.add_edge(u, v)
                self.update_remaining(u, v)

    def phase3(self):
        # build potential remaining edges and choose with rejection sampling
        potential_edges = combinations(self.remaining_degree, 2)
        # build auxiliary graph of potential edges not already in graph
        H = nx.Graph(
            [(u, v) for (u, v) in potential_edges if not self.graph.has_edge(u, v)]
        )
        rng = self.rng
        while self.remaining_degree:
            if not self.suitable_edge():
                raise nx.NetworkXUnfeasible("no suitable edges left")
            while True:
                u, v = sorted(rng.choice(list(H.edges())))
                if rng.random() < self.q(u, v):
                    break
            if rng.random() < self.p(u, v):  # accept edge
                self.graph.add_edge(u, v)
                self.update_remaining(u, v, aux_graph=H)
<|endoftext|>"""Functions for generating trees.

The functions sampling trees at random in this module come
in two variants: labeled and unlabeled. The labeled variants
sample from every possible tree with the given number of nodes
uniformly at random. The unlabeled variants sample from every
possible *isomorphism class* of trees with the given number
of nodes uniformly at random.

To understand the difference, consider the following example.
There are two isomorphism classes of trees with four nodes.
One is that of the path graph, the other is that of the
star graph. The unlabeled variant will return a line graph or
a star graph with probability 1/2.

The labeled variant will return the line graph
with probability 3/4 and the star graph with probability 1/4,
because there are more labeled variants of the line graph
than of the star graph. More precisely, the line graph has
an automorphism group of order 2, whereas the star graph has
an automorphism group of order 6, so the line graph has three
times as many labeled variants as the star graph, and thus
three more chances to be drawn.

Additionally, some functions in this module can sample rooted
trees and forests uniformly at random. A rooted tree is a tree
with a designated root node. A rooted forest is a disjoint union
of rooted trees.
"""

import warnings
from collections import Counter, defaultdict
from math import comb, factorial

import networkx as nx
from networkx.utils import py_random_state

__all__ = [
    "prefix_tree",
    "prefix_tree_recursive",
    "random_labeled_tree",
    "random_labeled_rooted_tree",
    "random_labeled_rooted_forest",
    "random_unlabeled_tree",
    "random_unlabeled_rooted_tree",
    "random_unlabeled_rooted_forest",
]


@nx._dispatchable(graphs=None, returns_graph=True)
def prefix_tree(paths):
    """Creates a directed prefix tree from a list of paths.

    Usually the paths are described as strings or lists of integers.

    A "prefix tree" represents the prefix structure of the strings.
    Each node represents a prefix of some string. The root represents
    the empty prefix with children for the single letter prefixes which
    in turn have children for each double letter prefix starting with
    the single letter corresponding to the parent node, and so on.

    More generally the prefixes do not need to be strings. A prefix refers
    to the start of a sequence. The root has children for each one element
    prefix and they have children for each two element prefix that starts
    with the one element sequence of the parent, and so on.

    Note that this implementation uses integer nodes with an attribute.
    Each node has an attribute "source" whose value is the original element
    of the path to which this node corresponds. For example, suppose `paths`
    consists of one path: "can". Then the nodes `[1, 2, 3]` which represent
    this path have "source" values "c", "a" and "n".

    All the descendants of a node have a common prefix in the sequence/path
    associated with that node. From the returned tree, the prefix for each
    node can be constructed by traversing the tree up to the root and
    accumulating the "source" values along the way.

    The root node is always `0` and has "source" attribute `None`.
    The root is the only node with in-degree zero.
    The nil node is always `-1` and has "source" attribute `"NIL"`.
    The nil node is the only node with out-degree zero.


    Parameters
    ----------
    paths: iterable of paths
        An iterable of paths which are themselves sequences.
        Matching prefixes among these sequences are identified with
        nodes of the prefix tree. One leaf of the tree is associated
        with each path. (Identical paths are associated with the same
        leaf of the tree.)


    Returns
    -------
    tree: DiGraph
        A directed graph representing an arborescence consisting of the
        prefix tree generated by `paths`. Nodes are directed "downward",
        from parent to child. A special "synthetic" root node is added
        to be the parent of the first node in each path. A special
        "synthetic" leaf node, the "nil" node `-1`, is added to be the child
        of all nodes representing the last element in a path. (The
        addition of this nil node technically makes this not an
        arborescence but a directed acyclic graph; removing the nil node
        makes it an arborescence.)


    Notes
    -----
    The prefix tree is also known as a *trie*.


    Examples
    --------
    Create a prefix tree from a list of strings with common prefixes::

        >>> paths = ["ab", "abs", "ad"]
        >>> T = nx.prefix_tree(paths)
        >>> list(T.edges)
        [(0, 1), (1, 2), (1, 4), (2, -1), (2, 3), (3, -1), (4, -1)]

    The leaf nodes can be obtained as predecessors of the nil node::

        >>> root, NIL = 0, -1
        >>> list(T.predecessors(NIL))
        [2, 3, 4]

    To recover the original paths that generated the prefix tree,
    traverse up the tree from the node `-1` to the node `0`::

        >>> recovered = []
        >>> for v in T.predecessors(NIL):
        ...     prefix = ""
        ...     while v != root:
        ...         prefix = str(T.nodes[v]["source"]) + prefix
        ...         v = next(T.predecessors(v))  # only one predecessor
        ...     recovered.append(prefix)
        >>> sorted(recovered)
        ['ab', 'abs', 'ad']
    """

    def get_children(parent, paths):
        children = defaultdict(list)
        # Populate dictionary with key(s) as the child/children of the root and
        # value(s) as the remaining paths of the corresponding child/children
        for path in paths:
            # If path is empty, we add an edge to the NIL node.
            if not path:
                tree.add_edge(parent, NIL)
                continue
            child, *rest = path
            # `child` may exist as the head of more than one path in `paths`.
            children[child].append(rest)
        return children

    # Initialize the prefix tree with a root node and a nil node.
    tree = nx.DiGraph()
    root = 0
    tree.add_node(root, source=None)
    NIL = -1
    tree.add_node(NIL, source="NIL")
    children = get_children(root, paths)
    stack = [(root, iter(children.items()))]
    while stack:
        parent, remaining_children = stack[-1]
        try:
            child, remaining_paths = next(remaining_children)
        # Pop item off stack if there are no remaining children
        except StopIteration:
            stack.pop()
            continue
        # We relabel each child with an unused name.
        new_name = len(tree) - 1
        # The "source" node attribute stores the original node name.
        tree.add_node(new_name, source=child)
        tree.add_edge(parent, new_name)
        children = get_children(new_name, remaining_paths)
        stack.append((new_name, iter(children.items())))

    return tree


@nx._dispatchable(graphs=None, returns_graph=True)
def prefix_tree_recursive(paths):
    """Recursively creates a directed prefix tree from a list of paths.

    The original recursive version of prefix_tree for comparison. It is
    the same algorithm but the recursion is unrolled onto a stack.

    Usually the paths are described as strings or lists of integers.

    A "prefix tree" represents the prefix structure of the strings.
    Each node represents a prefix of some string. The root represents
    the empty prefix with children for the single letter prefixes which
    in turn have children for each double letter prefix starting with
    the single letter corresponding to the parent node, and so on.

    More generally the prefixes do not need to be strings. A prefix refers
    to the start of a sequence. The root has children for each one element
    prefix and they have children for each two element prefix that starts
    with the one element sequence of the parent, and so on.

    Note that this implementation uses integer nodes with an attribute.
    Each node has an attribute "source" whose value is the original element
    of the path to which this node corresponds. For example, suppose `paths`
    consists of one path: "can". Then the nodes `[1, 2, 3]` which represent
    this path have "source" values "c", "a" and "n".

    All the descendants of a node have a common prefix in the sequence/path
    associated with that node. From the returned tree, ehe prefix for each
    node can be constructed by traversing the tree up to the root and
    accumulating the "source" values along the way.

    The root node is always `0` and has "source" attribute `None`.
    The root is the only node with in-degree zero.
    The nil node is always `-1` and has "source" attribute `"NIL"`.
    The nil node is the only node with out-degree zero.


    Parameters
    ----------
    paths: iterable of paths
        An iterable of paths which are themselves sequences.
        Matching prefixes among these sequences are identified with
        nodes of the prefix tree. One leaf of the tree is associated
        with each path. (Identical paths are associated with the same
        leaf of the tree.)


    Returns
    -------
    tree: DiGraph
        A directed graph representing an arborescence consisting of the
        prefix tree generated by `paths`. Nodes are directed "downward",
        from parent to child. A special "synthetic" root node is added
        to be the parent of the first node in each path. A special
        "synthetic" leaf node, the "nil" node `-1`, is added to be the child
        of all nodes representing the last element in a path. (The
        addition of this nil node technically makes this not an
        arborescence but a directed acyclic graph; removing the nil node
        makes it an arborescence.)


    Notes
    -----
    The prefix tree is also known as a *trie*.


    Examples
    --------
    Create a prefix tree from a list of strings with common prefixes::

        >>> paths = ["ab", "abs", "ad"]
        >>> T = nx.prefix_tree(paths)
        >>> list(T.edges)
        [(0, 1), (1, 2), (1, 4), (2, -1), (2, 3), (3, -1), (4, -1)]

    The leaf nodes can be obtained as predecessors of the nil node.

        >>> root, NIL = 0, -1
        >>> list(T.predecessors(NIL))
        [2, 3, 4]

    To recover the original paths that generated the prefix tree,
    traverse up the tree from the node `-1` to the node `0`::

        >>> recovered = []
        >>> for v in T.predecessors(NIL):
        ...     prefix = ""
        ...     while v != root:
        ...         prefix = str(T.nodes[v]["source"]) + prefix
        ...         v = next(T.predecessors(v))  # only one predecessor
        ...     recovered.append(prefix)
        >>> sorted(recovered)
        ['ab', 'abs', 'ad']
    """

    def _helper(paths, root, tree):
        """Recursively create a trie from the given list of paths.

        `paths` is a list of paths, each of which is itself a list of
        nodes, relative to the given `root` (but not including it). This
        list of paths will be interpreted as a tree-like structure, in
        which two paths that share a prefix represent two branches of
        the tree with the same initial segment.

        `root` is the parent of the node at index 0 in each path.

        `tree` is the "accumulator", the :class:`networkx.DiGraph`
        representing the branching to which the new nodes and edges will
        be added.

        """
        # For each path, remove the first node and make it a child of root.
        # Any remaining paths then get processed recursively.
        children = defaultdict(list)
        for path in paths:
            # If path is empty, we add an edge to the NIL node.
            if not path:
                tree.add_edge(root, NIL)
                continue
            child, *rest = path
            # `child` may exist as the head of more than one path in `paths`.
            children[child].append(rest)
        # Add a node for each child, connect root, recurse to remaining paths
        for child, remaining_paths in children.items():
            # We relabel each child with an unused name.
            new_name = len(tree) - 1
            # The "source" node attribute stores the original node name.
            tree.add_node(new_name, source=child)
            tree.add_edge(root, new_name)
            _helper(remaining_paths, new_name, tree)

    # Initialize the prefix tree with a root node and a nil node.
    tree = nx.DiGraph()
    root = 0
    tree.add_node(root, source=None)
    NIL = -1
    tree.add_node(NIL, source="NIL")
    # Populate the tree.
    _helper(paths, root, tree)
    return tree


@py_random_state("seed")
@nx._dispatchable(graphs=None, returns_graph=True)
def random_labeled_tree(n, *, seed=None):
    """Returns a labeled tree on `n` nodes chosen uniformly at random.

    Generating uniformly distributed random Prüfer sequences and
    converting them into the corresponding trees is a straightforward
    method of generating uniformly distributed random labeled trees.
    This function implements this method.

    Parameters
    ----------
    n : int
        The number of nodes, greater than zero.
    seed : random_state
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`

    Returns
    -------
     :class:`networkx.Graph`
        A `networkx.Graph` with nodes in the set {0, …, *n* - 1}.

    Raises
    ------
    NetworkXPointlessConcept
        If `n` is zero (because the null graph is not a tree).

    Examples
    --------
    >>> G = nx.random_labeled_tree(5, seed=42)
    >>> nx.is_tree(G)
    True
    >>> G.edges
    EdgeView([(0, 1), (0, 3), (0, 2), (2, 4)])

    A tree with *arbitrarily directed* edges can be created by assigning
    generated edges to a ``DiGraph``:

    >>> DG = nx.DiGraph()
    >>> DG.add_edges_from(G.edges)
    >>> nx.is_tree(DG)
    True
    >>> DG.edges
    OutEdgeView([(0, 1), (0, 3), (0, 2), (2, 4)])
    """
    # Cannot create a Prüfer sequence unless `n` is at least two.
    if n == 0:
        raise nx.NetworkXPointlessConcept("the null graph is not a tree")
    if n == 1:
        return nx.empty_graph(1)
    return nx.from_prufer_sequence([seed.choice(range(n)) for i in range(n - 2)])


@py_random_state("seed")
@nx._dispatchable(graphs=None, returns_graph=True)
def random_labeled_rooted_tree(n, *, seed=None):
    """Returns a labeled rooted tree with `n` nodes.

    The returned tree is chosen uniformly at random from all labeled rooted trees.

    Parameters
    ----------
    n : int
        The number of nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    :class:`networkx.Graph`
        A `networkx.Graph` with integer nodes 0 <= node <= `n` - 1.
        The root of the tree is selected uniformly from the nodes.
        The "root" graph attribute identifies the root of the tree.

    Notes
    -----
    This function returns the result of :func:`random_labeled_tree`
    with a randomly selected root.

    Raises
    ------
    NetworkXPointlessConcept
        If `n` is zero (because the null graph is not a tree).
    """
    t = random_labeled_tree(n, seed=seed)
    t.graph["root"] = seed.randint(0, n - 1)
    return t


@py_random_state("seed")
@nx._dispatchable(graphs=None, returns_graph=True)
def random_labeled_rooted_forest(n, *, seed=None):
    """Returns a labeled rooted forest with `n` nodes.

    The returned forest is chosen uniformly at random using a
    generalization of Prüfer sequences [1]_ in the form described in [2]_.

    Parameters
    ----------
    n : int
        The number of nodes.
    seed : random_state
       See :ref:`Randomness<randomness>`.

    Returns
    -------
    :class:`networkx.Graph`
        A `networkx.Graph` with integer nodes 0 <= node <= `n` - 1.
        The "roots" graph attribute is a set of integers containing the roots.

    References
    ----------
    .. [1] Knuth, Donald E. "Another Enumeration of Trees."
        Canadian Journal of Mathematics, 20 (1968): 1077-1086.
        https://doi.org/10.4153/CJM-1968-104-8
    .. [2] Rubey, Martin. "Counting Spanning Trees". Diplomarbeit
        zur Erlangung des akademischen Grades Magister der
        Naturwissenschaften an der Formal- und Naturwissenschaftlichen
        Fakultät der Universität Wien. Wien, May 2000.
    """

    # Select the number of roots by iterating over the cumulative count of trees
    # with at most k roots
    def _select_k(n, seed):
        r = seed.randint(0, (n + 1) ** (n - 1) - 1)
        cum_sum = 0
        for k in range(1, n):
            cum_sum += (factorial(n - 1) * n ** (n - k)) // (
                factorial(k - 1) * factorial(n - k)
            )
            if r < cum_sum:
                return k

        return n

    F = nx.empty_graph(n)
    if n == 0:
        F.graph["roots"] = {}
        return F
    # Select the number of roots k
    k = _select_k(n, seed)
    if k == n:
        F.graph["roots"] = set(range(n))
        return F  # Nothing to do
    # Select the roots
    roots = seed.sample(range(n), k)
    # Nonroots
    p = set(range(n)).difference(roots)
    # Coding sequence
    N = [seed.randint(0, n - 1) for i in range(n - k - 1)]
    # Multiset of elements in N also in p
    degree = Counter([x for x in N if x in p])
    # Iterator over the elements of p with degree zero
    iterator = iter(x for x in p if degree[x] == 0)
    u = last = next(iterator)
    # This loop is identical to that for Prüfer sequences,
    # except that we can draw nodes only from p
    for v in N:
        F.add_edge(u, v)
        degree[v] -= 1
        if v < last and degree[v] == 0:
            u = v
        else:
            last = u = next(iterator)

    F.add_edge(u, roots[0])
    F.graph["roots"] = set(roots)
    return F


# The following functions support generation of unlabeled trees and forests.


def _to_nx(edges, n_nodes, root=None, roots=None):
    """
    Converts the (edges, n_nodes) input to a :class:`networkx.Graph`.
    The (edges, n_nodes) input is a list of even length, where each pair
    of consecutive integers represents an edge, and an integer `n_nodes`.
    Integers in the list are elements of `range(n_nodes)`.

    Parameters
    ----------
    edges : list of ints
        The flattened list of edges of the graph.
    n_nodes : int
        The number of nodes of the graph.
    root: int (default=None)
        If not None, the "root" attribute of the graph will be set to this value.
    roots: collection of ints (default=None)
        If not None, he "roots" attribute of the graph will be set to this value.

    Returns
    -------
    :class:`networkx.Graph`
        The graph with `n_nodes` nodes and edges given by `edges`.
    """
    G = nx.empty_graph(n_nodes)
    G.add_edges_from(edges)
    if root is not None:
        G.graph["root"] = root
    if roots is not None:
        G.graph["roots"] = roots
    return G


def _num_rooted_trees(n, cache_trees):
    """Returns the number of unlabeled rooted trees with `n` nodes.

    See also https://oeis.org/A000081.

    Parameters
    ----------
    n : int
        The number of nodes
    cache_trees : list of ints
        The $i$-th element is the number of unlabeled rooted trees with $i$ nodes,
        which is used as a cache (and is extended to length $n+1$ if needed)

    Returns
    -------
    int
        The number of unlabeled rooted trees with `n` nodes.
    """
    for n_i in range(len(cache_trees), n + 1):
        cache_trees.append(
            sum(
                [
                    d * cache_trees[n_i - j * d] * cache_trees[d]
                    for d in range(1, n_i)
                    for j in range(1, (n_i - 1) // d + 1)
                ]
            )
            // (n_i - 1)
        )
    return cache_trees[n]


def _select_jd_trees(n, cache_trees, seed):
    """Returns a pair $(j,d)$ with a specific probability

    Given $n$, returns a pair of positive integers $(j,d)$ with the probability
    specified in formula (5) of Chapter 29 of [1]_.

    Parameters
    ----------
    n : int
        The number of nodes
    cache_trees : list of ints
        Cache for :func:`_num_rooted_trees`.
    seed : random_state
       See :ref:`Randomness<randomness>`.

    Returns
    -------
    (int, int)
        A pair of positive integers $(j,d)$ satisfying formula (5) of
        Chapter 29 of [1]_.

    References
    ----------
    .. [1] Nijenhuis, Albert, and Wilf, Herbert S.
        "Combinatorial algorithms: for computers and calculators."
        Academic Press, 1978.
        https://doi.org/10.1016/C2013-0-11243-3
    """
    p = seed.randint(0, _num_rooted_trees(n, cache_trees) * (n - 1) - 1)
    cumsum = 0
    for d in range(n - 1, 0, -1):
        for j in range(1, (n - 1) // d + 1):
            cumsum += (
                d
                * _num_rooted_trees(n - j * d, cache_trees)
                * _num_rooted_trees(d, cache_trees)
            )
            if p < cumsum:
                return (j, d)


def _random_unlabeled_rooted_tree(n, cache_trees, seed):
    """Returns an unlabeled rooted tree with `n` nodes.

    Returns an unlabeled rooted tree with `n` nodes chosen uniformly
    at random using the "RANRUT" algorithm from [1]_.
    The tree is returned in the form: (list_of_edges, number_of_nodes)

    Parameters
    ----------
    n : int
        The number of nodes, greater than zero.
    cache_trees : list ints
        Cache for :func:`_num_rooted_trees`.
    seed : random_state
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    (list_of_edges, number_of_nodes) : list, int
        A random unlabeled rooted tree with `n` nodes as a 2-tuple
        ``(list_of_edges, number_of_nodes)``.
        The root is node 0.

    References
    ----------
    .. [1] Nijenhuis, Albert, and Wilf, Herbert S.
        "Combinatorial algorithms: for computers and calculators."
        Academic Press, 1978.
        https://doi.org/10.1016/C2013-0-11243-3
    """
    if n == 1:
        edges, n_nodes = [], 1
        return edges, n_nodes
    if n == 2:
        edges, n_nodes = [(0, 1)], 2
        return edges, n_nodes

    j, d = _select_jd_trees(n, cache_trees, seed)
    t1, t1_nodes = _random_unlabeled_rooted_tree(n - j * d, cache_trees, seed)
    t2, t2_nodes = _random_unlabeled_rooted_tree(d, cache_trees, seed)
    t12 = [(0, t2_nodes * i + t1_nodes) for i in range(j)]
    t1.extend(t12)
    for _ in range(j):
        t1.extend((n1 + t1_nodes, n2 + t1_nodes) for n1, n2 in t2)
        t1_nodes += t2_nodes

    return t1, t1_nodes


@py_random_state("seed")
@nx._dispatchable(graphs=None, returns_graph=True)
def random_unlabeled_rooted_tree(n, *, number_of_trees=None, seed=None):
    """Returns a number of unlabeled rooted trees uniformly at random

    Returns one or more (depending on `number_of_trees`)
    unlabeled rooted trees with `n` nodes drawn uniformly
    at random.

    Parameters
    ----------
    n : int
        The number of nodes
    number_of_trees : int or None (default)
        If not None, this number of trees is generated and returned.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    :class:`networkx.Graph` or list of :class:`networkx.Graph`
        A single `networkx.Graph` (or a list thereof, if `number_of_trees`
        is specified) with nodes in the set {0, …, *n* - 1}.
        The "root" graph attribute identifies the root of the tree.

    Notes
    -----
    The trees are generated using the "RANRUT" algorithm from [1]_.
    The algorithm needs to compute some counting functions
    that are relatively expensive: in case several trees are needed,
    it is advisable to use the `number_of_trees` optional argument
    to reuse the counting functions.

    Raises
    ------
    NetworkXPointlessConcept
        If `n` is zero (because the null graph is not a tree).

    References
    ----------
    .. [1] Nijenhuis, Albert, and Wilf, Herbert S.
        "Combinatorial algorithms: for computers and calculators."
        Academic Press, 1978.
        https://doi.org/10.1016/C2013-0-11243-3
    """
    if n == 0:
        raise nx.NetworkXPointlessConcept("the null graph is not a tree")
    cache_trees = [0, 1]  # initial cache of number of rooted trees
    if number_of_trees is None:
        return _to_nx(*_random_unlabeled_rooted_tree(n, cache_trees, seed), root=0)
    return [
        _to_nx(*_random_unlabeled_rooted_tree(n, cache_trees, seed), root=0)
        for i in range(number_of_trees)
    ]


def _num_rooted_forests(n, q, cache_forests):
    """Returns the number of unlabeled rooted forests with `n` nodes, and with
    no more than `q` nodes per tree. A recursive formula for this is (2) in
    [1]_. This function is implemented using dynamic programming instead of
    recursion.

    Parameters
    ----------
    n : int
        The number of nodes.
    q : int
        The maximum number of nodes for each tree of the forest.
    cache_forests : list of ints
        The $i$-th element is the number of unlabeled rooted forests with
        $i$ nodes, and with no more than `q` nodes per tree; this is used
        as a cache (and is extended to length `n` + 1 if needed).

    Returns
    -------
    int
        The number of unlabeled rooted forests with `n` nodes with no more than
        `q` nodes per tree.

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    for n_i in range(len(cache_forests), n + 1):
        q_i = min(n_i, q)
        cache_forests.append(
            sum(
                [
                    d * cache_forests[n_i - j * d] * cache_forests[d - 1]
                    for d in range(1, q_i + 1)
                    for j in range(1, n_i // d + 1)
                ]
            )
            // n_i
        )

    return cache_forests[n]


def _select_jd_forests(n, q, cache_forests, seed):
    """Given `n` and `q`, returns a pair of positive integers $(j,d)$
    such that $j\\leq d$, with probability satisfying (F1) of [1]_.

    Parameters
    ----------
    n : int
        The number of nodes.
    q : int
        The maximum number of nodes for each tree of the forest.
    cache_forests : list of ints
        Cache for :func:`_num_rooted_forests`.
    seed : random_state
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    (int, int)
        A pair of positive integers $(j,d)$

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    p = seed.randint(0, _num_rooted_forests(n, q, cache_forests) * n - 1)
    cumsum = 0
    for d in range(q, 0, -1):
        for j in range(1, n // d + 1):
            cumsum += (
                d
                * _num_rooted_forests(n - j * d, q, cache_forests)
                * _num_rooted_forests(d - 1, q, cache_forests)
            )
            if p < cumsum:
                return (j, d)


def _random_unlabeled_rooted_forest(n, q, cache_trees, cache_forests, seed):
    """Returns an unlabeled rooted forest with `n` nodes, and with no more
    than `q` nodes per tree, drawn uniformly at random. It is an implementation
    of the algorithm "Forest" of [1]_.

    Parameters
    ----------
    n : int
        The number of nodes.
    q : int
        The maximum number of nodes per tree.
    cache_trees :
        Cache for :func:`_num_rooted_trees`.
    cache_forests :
        Cache for :func:`_num_rooted_forests`.
    seed : random_state
       See :ref:`Randomness<randomness>`.

    Returns
    -------
    (edges, n, r) : (list, int, list)
        The forest (edges, n) and a list r of root nodes.

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    if n == 0:
        return ([], 0, [])

    j, d = _select_jd_forests(n, q, cache_forests, seed)
    t1, t1_nodes, r1 = _random_unlabeled_rooted_forest(
        n - j * d, q, cache_trees, cache_forests, seed
    )
    t2, t2_nodes = _random_unlabeled_rooted_tree(d, cache_trees, seed)
    for _ in range(j):
        r1.append(t1_nodes)
        t1.extend((n1 + t1_nodes, n2 + t1_nodes) for n1, n2 in t2)
        t1_nodes += t2_nodes
    return t1, t1_nodes, r1


@py_random_state("seed")
@nx._dispatchable(graphs=None, returns_graph=True)
def random_unlabeled_rooted_forest(n, *, q=None, number_of_forests=None, seed=None):
    """Returns a forest or list of forests selected at random.

    Returns one or more (depending on `number_of_forests`)
    unlabeled rooted forests with `n` nodes, and with no more than
    `q` nodes per tree, drawn uniformly at random.
    The "roots" graph attribute identifies the roots of the forest.

    Parameters
    ----------
    n : int
        The number of nodes
    q : int or None (default)
        The maximum number of nodes per tree.
    number_of_forests : int or None (default)
        If not None, this number of forests is generated and returned.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    :class:`networkx.Graph` or list of :class:`networkx.Graph`
        A single `networkx.Graph` (or a list thereof, if `number_of_forests`
        is specified) with nodes in the set {0, …, *n* - 1}.
        The "roots" graph attribute is a set containing the roots
        of the trees in the forest.

    Notes
    -----
    This function implements the algorithm "Forest" of [1]_.
    The algorithm needs to compute some counting functions
    that are relatively expensive: in case several trees are needed,
    it is advisable to use the `number_of_forests` optional argument
    to reuse the counting functions.

    Raises
    ------
    ValueError
        If `n` is non-zero but `q` is zero.

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    if q is None:
        q = n
    if q == 0 and n != 0:
        raise ValueError("q must be a positive integer if n is positive.")

    cache_trees = [0, 1]  # initial cache of number of rooted trees
    cache_forests = [1]  # initial cache of number of rooted forests

    if number_of_forests is None:
        g, nodes, rs = _random_unlabeled_rooted_forest(
            n, q, cache_trees, cache_forests, seed
        )
        return _to_nx(g, nodes, roots=set(rs))

    res = []
    for i in range(number_of_forests):
        g, nodes, rs = _random_unlabeled_rooted_forest(
            n, q, cache_trees, cache_forests, seed
        )
        res.append(_to_nx(g, nodes, roots=set(rs)))
    return res


def _num_trees(n, cache_trees):
    """Returns the number of unlabeled trees with `n` nodes.

    See also https://oeis.org/A000055.

    Parameters
    ----------
    n : int
        The number of nodes.
    cache_trees : list of ints
        Cache for :func:`_num_rooted_trees`.

    Returns
    -------
    int
        The number of unlabeled trees with `n` nodes.
    """
    r = _num_rooted_trees(n, cache_trees) - sum(
        [
            _num_rooted_trees(j, cache_trees) * _num_rooted_trees(n - j, cache_trees)
            for j in range(1, n // 2 + 1)
        ]
    )
    if n % 2 == 0:
        r += comb(_num_rooted_trees(n // 2, cache_trees) + 1, 2)
    return r


def _bicenter(n, cache, seed):
    """Returns a bi-centroidal tree on `n` nodes drawn uniformly at random.

    This function implements the algorithm Bicenter of [1]_.

    Parameters
    ----------
    n : int
        The number of nodes (must be even).
    cache : list of ints.
        Cache for :func:`_num_rooted_trees`.
    seed : random_state
        See :ref:`Randomness<randomness>`

    Returns
    -------
    (edges, n)
        The tree as a list of edges and number of nodes.

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    t, t_nodes = _random_unlabeled_rooted_tree(n // 2, cache, seed)
    if seed.randint(0, _num_rooted_trees(n // 2, cache)) == 0:
        t2, t2_nodes = t, t_nodes
    else:
        t2, t2_nodes = _random_unlabeled_rooted_tree(n // 2, cache, seed)
    t.extend([(n1 + (n // 2), n2 + (n // 2)) for n1, n2 in t2])
    t.append((0, n // 2))
    return t, t_nodes + t2_nodes


def _random_unlabeled_tree(n, cache_trees, cache_forests, seed):
    """Returns a tree on `n` nodes drawn uniformly at random.
    It implements the Wilf's algorithm "Free" of [1]_.

    Parameters
    ----------
    n : int
        The number of nodes, greater than zero.
    cache_trees : list of ints
        Cache for :func:`_num_rooted_trees`.
    cache_forests : list of ints
        Cache for :func:`_num_rooted_forests`.
    seed : random_state
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`

    Returns
    -------
    (edges, n)
        The tree as a list of edges and number of nodes.

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    if n % 2 == 1:
        p = 0
    else:
        p = comb(_num_rooted_trees(n // 2, cache_trees) + 1, 2)
    if seed.randint(0, _num_trees(n, cache_trees) - 1) < p:
        return _bicenter(n, cache_trees, seed)
    else:
        f, n_f, r = _random_unlabeled_rooted_forest(
            n - 1, (n - 1) // 2, cache_trees, cache_forests, seed
        )
        for i in r:
            f.append((i, n_f))
        return f, n_f + 1


@py_random_state("seed")
@nx._dispatchable(graphs=None, returns_graph=True)
def random_unlabeled_tree(n, *, number_of_trees=None, seed=None):
    """Returns a tree or list of trees chosen randomly.

    Returns one or more (depending on `number_of_trees`)
    unlabeled trees with `n` nodes drawn uniformly at random.

    Parameters
    ----------
    n : int
        The number of nodes
    number_of_trees : int or None (default)
        If not None, this number of trees is generated and returned.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    :class:`networkx.Graph` or list of :class:`networkx.Graph`
        A single `networkx.Graph` (or a list thereof, if
        `number_of_trees` is specified) with nodes in the set {0, …, *n* - 1}.

    Raises
    ------
    NetworkXPointlessConcept
        If `n` is zero (because the null graph is not a tree).

    Notes
    -----
    This function generates an unlabeled tree uniformly at random using
    Wilf's algorithm "Free" of [1]_. The algorithm needs to
    compute some counting functions that are relatively expensive:
    in case several trees are needed, it is advisable to use the
    `number_of_trees` optional argument to reuse the counting
    functions.

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    if n == 0:
        raise nx.NetworkXPointlessConcept("the null graph is not a tree")

    cache_trees = [0, 1]  # initial cache of number of rooted trees
    cache_forests = [1]  # initial cache of number of rooted forests
    if number_of_trees is None:
        return _to_nx(*_random_unlabeled_tree(n, cache_trees, cache_forests, seed))
    else:
        return [
            _to_nx(*_random_unlabeled_tree(n, cache_trees, cache_forests, seed))
            for i in range(number_of_trees)
        ]
<|endoftext|>"""
Time Series Graphs
"""

import itertools

import networkx as nx

__all__ = ["visibility_graph"]


@nx._dispatchable(graphs=None, returns_graph=True)
def visibility_graph(series):
    """
    Return a Visibility Graph of an input Time Series.

    A visibility graph converts a time series into a graph. The constructed graph
    uses integer nodes to indicate which event in the series the node represents.
    Edges are formed as follows: consider a bar plot of the series and view that
    as a side view of a landscape with a node at the top of each bar. An edge
    means that the nodes can be connected by a straight "line-of-sight" without
    being obscured by any bars between the nodes.

    The resulting graph inherits several properties of the series in its structure.
    Thereby, periodic series convert into regular graphs, random series convert
    into random graphs, and fractal series convert into scale-free networks [1]_.

    Parameters
    ----------
    series : Sequence[Number]
       A Time Series sequence (iterable and sliceable) of numeric values
       representing times.

    Returns
    -------
    NetworkX Graph
        The Visibility Graph of the input series

    Examples
    --------
    >>> series_list = [range(10), [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3]]
    >>> for s in series_list:
    ...     g = nx.visibility_graph(s)
    ...     print(g)
    Graph with 10 nodes and 9 edges
    Graph with 12 nodes and 18 edges

    References
    ----------
    .. [1] Lacasa, Lucas, Bartolo Luque, Fernando Ballesteros, Jordi Luque, and Juan Carlos Nuno.
           "From time series to complex networks: The visibility graph." Proceedings of the
           National Academy of Sciences 105, no. 13 (2008): 4972-4975.
           https://www.pnas.org/doi/10.1073/pnas.0709247105
    """

    # Sequential values are always connected
    G = nx.path_graph(len(series))
    nx.set_node_attributes(G, dict(enumerate(series)), "value")

    # Check all combinations of nodes n series
    for (n1, t1), (n2, t2) in itertools.combinations(enumerate(series), 2):
        # check if any value between obstructs line of sight
        slope = (t2 - t1) / (n2 - n1)
        offset = t2 - slope * n2

        obstructed = any(
            t >= slope * n + offset
            for n, t in enumerate(series[n1 + 1 : n2], start=n1 + 1)
        )

        if not obstructed:
            G.add_edge(n1, n2)

    return G
<|endoftext|>"""
Famous social networks.
"""

import networkx as nx

__all__ = [
    "karate_club_graph",
    "davis_southern_women_graph",
    "florentine_families_graph",
    "les_miserables_graph",
]


@nx._dispatchable(graphs=None, returns_graph=True)
def karate_club_graph():
    """Returns Zachary's Karate Club graph.

    Each node in the returned graph has a node attribute 'club' that
    indicates the name of the club to which the member represented by that node
    belongs, either 'Mr. Hi' or 'Officer'. Each edge has a weight based on the
    number of contexts in which that edge's incident node members interacted.

    The dataset is derived from the 'Club After Split From Data' column of Table 3 in [1]_.
    This was in turn derived from the 'Club After Fission' column of Table 1 in the
    same paper. Note that the nodes are 0-indexed in NetworkX, but 1-indexed in the
    paper (the 'Individual Number in Matrix C' column of Table 3 starts at 1). This
    means, for example, that ``G.nodes[9]["club"]`` returns 'Officer', which
    corresponds to row 10 of Table 3 in the paper.

    Examples
    --------
    To get the name of the club to which a node belongs::

        >>> G = nx.karate_club_graph()
        >>> G.nodes[5]["club"]
        'Mr. Hi'
        >>> G.nodes[9]["club"]
        'Officer'

    References
    ----------
    .. [1] Zachary, Wayne W.
       "An Information Flow Model for Conflict and Fission in Small Groups."
       *Journal of Anthropological Research*, 33, 452--473, (1977).
    """
    # Create the set of all members, and the members of each club.
    all_members = set(range(34))
    club1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21}
    # club2 = all_members - club1

    G = nx.Graph()
    G.add_nodes_from(all_members)
    G.name = "Zachary's Karate Club"

    zacharydat = """\
0 4 5 3 3 3 3 2 2 0 2 3 2 3 0 0 0 2 0 2 0 2 0 0 0 0 0 0 0 0 0 2 0 0
4 0 6 3 0 0 0 4 0 0 0 0 0 5 0 0 0 1 0 2 0 2 0 0 0 0 0 0 0 0 2 0 0 0
5 6 0 3 0 0 0 4 5 1 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 0 0 0 3 0
3 3 3 0 0 0 0 3 0 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 0 0 0 0 0 2 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 0 0 0 0 0 5 0 0 0 3 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 0 0 0 2 5 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
2 4 4 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
2 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 4 3
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
2 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 5 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 2
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 4
0 0 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2
2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1
2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 4 0 2 0 0 5 4
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 3 0 0 0 2 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 2 0 0 0 0 0 0 7 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 2
0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 3 0 0 0 0 0 0 0 0 4
0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 2
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 4 0 0 0 0 0 3 2
0 2 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 3
2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 7 0 0 2 0 0 0 4 4
0 0 2 0 0 0 0 0 3 0 0 0 0 0 3 3 0 0 1 0 3 0 2 5 0 0 0 0 0 4 3 4 0 5
0 0 0 0 0 0 0 0 4 2 0 0 0 3 2 4 0 0 2 1 1 0 3 4 0 0 2 4 2 2 3 4 5 0"""

    for row, line in enumerate(zacharydat.split("\n")):
        thisrow = [int(b) for b in line.split()]
        for col, entry in enumerate(thisrow):
            if entry >= 1:
                G.add_edge(row, col, weight=entry)

    # Add the name of each member's club as a node attribute.
    for v in G:
        G.nodes[v]["club"] = "Mr. Hi" if v in club1 else "Officer"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def davis_southern_women_graph():
    """Returns Davis Southern women social network.

    This is a bipartite graph.

    References
    ----------
    .. [1] A. Davis, Gardner, B. B., Gardner, M. R., 1941. Deep South.
        University of Chicago Press, Chicago, IL.
    """
    G = nx.Graph()
    # Top nodes
    women = [
        "Evelyn Jefferson",
        "Laura Mandeville",
        "Theresa Anderson",
        "Brenda Rogers",
        "Charlotte McDowd",
        "Frances Anderson",
        "Eleanor Nye",
        "Pearl Oglethorpe",
        "Ruth DeSand",
        "Verne Sanderson",
        "Myra Liddel",
        "Katherina Rogers",
        "Sylvia Avondale",
        "Nora Fayette",
        "Helen Lloyd",
        "Dorothy Murchison",
        "Olivia Carleton",
        "Flora Price",
    ]
    G.add_nodes_from(women, bipartite=0)
    # Bottom nodes
    events = [
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "E8",
        "E9",
        "E10",
        "E11",
        "E12",
        "E13",
        "E14",
    ]
    G.add_nodes_from(events, bipartite=1)

    G.add_edges_from(
        [
            ("Evelyn Jefferson", "E1"),
            ("Evelyn Jefferson", "E2"),
            ("Evelyn Jefferson", "E3"),
            ("Evelyn Jefferson", "E4"),
            ("Evelyn Jefferson", "E5"),
            ("Evelyn Jefferson", "E6"),
            ("Evelyn Jefferson", "E8"),
            ("Evelyn Jefferson", "E9"),
            ("Laura Mandeville", "E1"),
            ("Laura Mandeville", "E2"),
            ("Laura Mandeville", "E3"),
            ("Laura Mandeville", "E5"),
            ("Laura Mandeville", "E6"),
            ("Laura Mandeville", "E7"),
            ("Laura Mandeville", "E8"),
            ("Theresa Anderson", "E2"),
            ("Theresa Anderson", "E3"),
            ("Theresa Anderson", "E4"),
            ("Theresa Anderson", "E5"),
            ("Theresa Anderson", "E6"),
            ("Theresa Anderson", "E7"),
            ("Theresa Anderson", "E8"),
            ("Theresa Anderson", "E9"),
            ("Brenda Rogers", "E1"),
            ("Brenda Rogers", "E3"),
            ("Brenda Rogers", "E4"),
            ("Brenda Rogers", "E5"),
            ("Brenda Rogers", "E6"),
            ("Brenda Rogers", "E7"),
            ("Brenda Rogers", "E8"),
            ("Charlotte McDowd", "E3"),
            ("Charlotte McDowd", "E4"),
            ("Charlotte McDowd", "E5"),
            ("Charlotte McDowd", "E7"),
            ("Frances Anderson", "E3"),
            ("Frances Anderson", "E5"),
            ("Frances Anderson", "E6"),
            ("Frances Anderson", "E8"),
            ("Eleanor Nye", "E5"),
            ("Eleanor Nye", "E6"),
            ("Eleanor Nye", "E7"),
            ("Eleanor Nye", "E8"),
            ("Pearl Oglethorpe", "E6"),
            ("Pearl Oglethorpe", "E8"),
            ("Pearl Oglethorpe", "E9"),
            ("Ruth DeSand", "E5"),
            ("Ruth DeSand", "E7"),
            ("Ruth DeSand", "E8"),
            ("Ruth DeSand", "E9"),
            ("Verne Sanderson", "E7"),
            ("Verne Sanderson", "E8"),
            ("Verne Sanderson", "E9"),
            ("Verne Sanderson", "E12"),
            ("Myra Liddel", "E8"),
            ("Myra Liddel", "E9"),
            ("Myra Liddel", "E10"),
            ("Myra Liddel", "E12"),
            ("Katherina Rogers", "E8"),
            ("Katherina Rogers", "E9"),
            ("Katherina Rogers", "E10"),
            ("Katherina Rogers", "E12"),
            ("Katherina Rogers", "E13"),
            ("Katherina Rogers", "E14"),
            ("Sylvia Avondale", "E7"),
            ("Sylvia Avondale", "E8"),
            ("Sylvia Avondale", "E9"),
            ("Sylvia Avondale", "E10"),
            ("Sylvia Avondale", "E12"),
            ("Sylvia Avondale", "E13"),
            ("Sylvia Avondale", "E14"),
            ("Nora Fayette", "E6"),
            ("Nora Fayette", "E7"),
            ("Nora Fayette", "E9"),
            ("Nora Fayette", "E10"),
            ("Nora Fayette", "E11"),
            ("Nora Fayette", "E12"),
            ("Nora Fayette", "E13"),
            ("Nora Fayette", "E14"),
            ("Helen Lloyd", "E7"),
            ("Helen Lloyd", "E8"),
            ("Helen Lloyd", "E10"),
            ("Helen Lloyd", "E11"),
            ("Helen Lloyd", "E12"),
            ("Dorothy Murchison", "E8"),
            ("Dorothy Murchison", "E9"),
            ("Olivia Carleton", "E9"),
            ("Olivia Carleton", "E11"),
            ("Flora Price", "E9"),
            ("Flora Price", "E11"),
        ]
    )
    G.graph["top"] = women
    G.graph["bottom"] = events
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def florentine_families_graph():
    """Returns Florentine families graph.

    References
    ----------
    .. [1] Ronald L. Breiger and Philippa E. Pattison
       Cumulated social roles: The duality of persons and their algebras,1
       Social Networks, Volume 8, Issue 3, September 1986, Pages 215-256
    """
    G = nx.Graph()
    G.add_edge("Acciaiuoli", "Medici")
    G.add_edge("Castellani", "Peruzzi")
    G.add_edge("Castellani", "Strozzi")
    G.add_edge("Castellani", "Barbadori")
    G.add_edge("Medici", "Barbadori")
    G.add_edge("Medici", "Ridolfi")
    G.add_edge("Medici", "Tornabuoni")
    G.add_edge("Medici", "Albizzi")
    G.add_edge("Medici", "Salviati")
    G.add_edge("Salviati", "Pazzi")
    G.add_edge("Peruzzi", "Strozzi")
    G.add_edge("Peruzzi", "Bischeri")
    G.add_edge("Strozzi", "Ridolfi")
    G.add_edge("Strozzi", "Bischeri")
    G.add_edge("Ridolfi", "Tornabuoni")
    G.add_edge("Tornabuoni", "Guadagni")
    G.add_edge("Albizzi", "Ginori")
    G.add_edge("Albizzi", "Guadagni")
    G.add_edge("Bischeri", "Guadagni")
    G.add_edge("Guadagni", "Lamberteschi")
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def les_miserables_graph():
    """Returns coappearance network of characters in the novel Les Miserables.

    References
    ----------
    .. [1] D. E. Knuth, 1993.
       The Stanford GraphBase: a platform for combinatorial computing,
       pp. 74-87. New York: AcM Press.
    """
    G = nx.Graph()
    G.add_edge("Napoleon", "Myriel", weight=1)
    G.add_edge("MlleBaptistine", "Myriel", weight=8)
    G.add_edge("MmeMagloire", "Myriel", weight=10)
    G.add_edge("MmeMagloire", "MlleBaptistine", weight=6)
    G.add_edge("CountessDeLo", "Myriel", weight=1)
    G.add_edge("Geborand", "Myriel", weight=1)
    G.add_edge("Champtercier", "Myriel", weight=1)
    G.add_edge("Cravatte", "Myriel", weight=1)
    G.add_edge("Count", "Myriel", weight=2)
    G.add_edge("OldMan", "Myriel", weight=1)
    G.add_edge("Valjean", "Labarre", weight=1)
    G.add_edge("Valjean", "MmeMagloire", weight=3)
    G.add_edge("Valjean", "MlleBaptistine", weight=3)
    G.add_edge("Valjean", "Myriel", weight=5)
    G.add_edge("Marguerite", "Valjean", weight=1)
    G.add_edge("MmeDeR", "Valjean", weight=1)
    G.add_edge("Isabeau", "Valjean", weight=1)
    G.add_edge("Gervais", "Valjean", weight=1)
    G.add_edge("Listolier", "Tholomyes", weight=4)
    G.add_edge("Fameuil", "Tholomyes", weight=4)
    G.add_edge("Fameuil", "Listolier", weight=4)
    G.add_edge("Blacheville", "Tholomyes", weight=4)
    G.add_edge("Blacheville", "Listolier", weight=4)
    G.add_edge("Blacheville", "Fameuil", weight=4)
    G.add_edge("Favourite", "Tholomyes", weight=3)
    G.add_edge("Favourite", "Listolier", weight=3)
    G.add_edge("Favourite", "Fameuil", weight=3)
    G.add_edge("Favourite", "Blacheville", weight=4)
    G.add_edge("Dahlia", "Tholomyes", weight=3)
    G.add_edge("Dahlia", "Listolier", weight=3)
    G.add_edge("Dahlia", "Fameuil", weight=3)
    G.add_edge("Dahlia", "Blacheville", weight=3)
    G.add_edge("Dahlia", "Favourite", weight=5)
    G.add_edge("Zephine", "Tholomyes", weight=3)
    G.add_edge("Zephine", "Listolier", weight=3)
    G.add_edge("Zephine", "Fameuil", weight=3)
    G.add_edge("Zephine", "Blacheville", weight=3)
    G.add_edge("Zephine", "Favourite", weight=4)
    G.add_edge("Zephine", "Dahlia", weight=4)
    G.add_edge("Fantine", "Tholomyes", weight=3)
    G.add_edge("Fantine", "Listolier", weight=3)
    G.add_edge("Fantine", "Fameuil", weight=3)
    G.add_edge("Fantine", "Blacheville", weight=3)
    G.add_edge("Fantine", "Favourite", weight=4)
    G.add_edge("Fantine", "Dahlia", weight=4)
    G.add_edge("Fantine", "Zephine", weight=4)
    G.add_edge("Fantine", "Marguerite", weight=2)
    G.add_edge("Fantine", "Valjean", weight=9)
    G.add_edge("MmeThenardier", "Fantine", weight=2)
    G.add_edge("MmeThenardier", "Valjean", weight=7)
    G.add_edge("Thenardier", "MmeThenardier", weight=13)
    G.add_edge("Thenardier", "Fantine", weight=1)
    G.add_edge("Thenardier", "Valjean", weight=12)
    G.add_edge("Cosette", "MmeThenardier", weight=4)
    G.add_edge("Cosette", "Valjean", weight=31)
    G.add_edge("Cosette", "Tholomyes", weight=1)
    G.add_edge("Cosette", "Thenardier", weight=1)
    G.add_edge("Javert", "Valjean", weight=17)
    G.add_edge("Javert", "Fantine", weight=5)
    G.add_edge("Javert", "Thenardier", weight=5)
    G.add_edge("Javert", "MmeThenardier", weight=1)
    G.add_edge("Javert", "Cosette", weight=1)
    G.add_edge("Fauchelevent", "Valjean", weight=8)
    G.add_edge("Fauchelevent", "Javert", weight=1)
    G.add_edge("Bamatabois", "Fantine", weight=1)
    G.add_edge("Bamatabois", "Javert", weight=1)
    G.add_edge("Bamatabois", "Valjean", weight=2)
    G.add_edge("Perpetue", "Fantine", weight=1)
    G.add_edge("Simplice", "Perpetue", weight=2)
    G.add_edge("Simplice", "Valjean", weight=3)
    G.add_edge("Simplice", "Fantine", weight=2)
    G.add_edge("Simplice", "Javert", weight=1)
    G.add_edge("Scaufflaire", "Valjean", weight=1)
    G.add_edge("Woman1", "Valjean", weight=2)
    G.add_edge("Woman1", "Javert", weight=1)
    G.add_edge("Judge", "Valjean", weight=3)
    G.add_edge("Judge", "Bamatabois", weight=2)
    G.add_edge("Champmathieu", "Valjean", weight=3)
    G.add_edge("Champmathieu", "Judge", weight=3)
    G.add_edge("Champmathieu", "Bamatabois", weight=2)
    G.add_edge("Brevet", "Judge", weight=2)
    G.add_edge("Brevet", "Champmathieu", weight=2)
    G.add_edge("Brevet", "Valjean", weight=2)
    G.add_edge("Brevet", "Bamatabois", weight=1)
    G.add_edge("Chenildieu", "Judge", weight=2)
    G.add_edge("Chenildieu", "Champmathieu", weight=2)
    G.add_edge("Chenildieu", "Brevet", weight=2)
    G.add_edge("Chenildieu", "Valjean", weight=2)
    G.add_edge("Chenildieu", "Bamatabois", weight=1)
    G.add_edge("Cochepaille", "Judge", weight=2)
    G.add_edge("Cochepaille", "Champmathieu", weight=2)
    G.add_edge("Cochepaille", "Brevet", weight=2)
    G.add_edge("Cochepaille", "Chenildieu", weight=2)
    G.add_edge("Cochepaille", "Valjean", weight=2)
    G.add_edge("Cochepaille", "Bamatabois", weight=1)
    G.add_edge("Pontmercy", "Thenardier", weight=1)
    G.add_edge("Boulatruelle", "Thenardier", weight=1)
    G.add_edge("Eponine", "MmeThenardier", weight=2)
    G.add_edge("Eponine", "Thenardier", weight=3)
    G.add_edge("Anzelma", "Eponine", weight=2)
    G.add_edge("Anzelma", "Thenardier", weight=2)
    G.add_edge("Anzelma", "MmeThenardier", weight=1)
    G.add_edge("Woman2", "Valjean", weight=3)
    G.add_edge("Woman2", "Cosette", weight=1)
    G.add_edge("Woman2", "Javert", weight=1)
    G.add_edge("MotherInnocent", "Fauchelevent", weight=3)
    G.add_edge("MotherInnocent", "Valjean", weight=1)
    G.add_edge("Gribier", "Fauchelevent", weight=2)
    G.add_edge("MmeBurgon", "Jondrette", weight=1)
    G.add_edge("Gavroche", "MmeBurgon", weight=2)
    G.add_edge("Gavroche", "Thenardier", weight=1)
    G.add_edge("Gavroche", "Javert", weight=1)
    G.add_edge("Gavroche", "Valjean", weight=1)
    G.add_edge("Gillenormand", "Cosette", weight=3)
    G.add_edge("Gillenormand", "Valjean", weight=2)
    G.add_edge("Magnon", "Gillenormand", weight=1)
    G.add_edge("Magnon", "MmeThenardier", weight=1)
    G.add_edge("MlleGillenormand", "Gillenormand", weight=9)
    G.add_edge("MlleGillenormand", "Cosette", weight=2)
    G.add_edge("MlleGillenormand", "Valjean", weight=2)
    G.add_edge("MmePontmercy", "MlleGillenormand", weight=1)
    G.add_edge("MmePontmercy", "Pontmercy", weight=1)
    G.add_edge("MlleVaubois", "MlleGillenormand", weight=1)
    G.add_edge("LtGillenormand", "MlleGillenormand", weight=2)
    G.add_edge("LtGillenormand", "Gillenormand", weight=1)
    G.add_edge("LtGillenormand", "Cosette", weight=1)
    G.add_edge("Marius", "MlleGillenormand", weight=6)
    G.add_edge("Marius", "Gillenormand", weight=12)
    G.add_edge("Marius", "Pontmercy", weight=1)
    G.add_edge("Marius", "LtGillenormand", weight=1)
    G.add_edge("Marius", "Cosette", weight=21)
    G.add_edge("Marius", "Valjean", weight=19)
    G.add_edge("Marius", "Tholomyes", weight=1)
    G.add_edge("Marius", "Thenardier", weight=2)
    G.add_edge("Marius", "Eponine", weight=5)
    G.add_edge("Marius", "Gavroche", weight=4)
    G.add_edge("BaronessT", "Gillenormand", weight=1)
    G.add_edge("BaronessT", "Marius", weight=1)
    G.add_edge("Mabeuf", "Marius", weight=1)
    G.add_edge("Mabeuf", "Eponine", weight=1)
    G.add_edge("Mabeuf", "Gavroche", weight=1)
    G.add_edge("Enjolras", "Marius", weight=7)
    G.add_edge("Enjolras", "Gavroche", weight=7)
    G.add_edge("Enjolras", "Javert", weight=6)
    G.add_edge("Enjolras", "Mabeuf", weight=1)
    G.add_edge("Enjolras", "Valjean", weight=4)
    G.add_edge("Combeferre", "Enjolras", weight=15)
    G.add_edge("Combeferre", "Marius", weight=5)
    G.add_edge("Combeferre", "Gavroche", weight=6)
    G.add_edge("Combeferre", "Mabeuf", weight=2)
    G.add_edge("Prouvaire", "Gavroche", weight=1)
    G.add_edge("Prouvaire", "Enjolras", weight=4)
    G.add_edge("Prouvaire", "Combeferre", weight=2)
    G.add_edge("Feuilly", "Gavroche", weight=2)
    G.add_edge("Feuilly", "Enjolras", weight=6)
    G.add_edge("Feuilly", "Prouvaire", weight=2)
    G.add_edge("Feuilly", "Combeferre", weight=5)
    G.add_edge("Feuilly", "Mabeuf", weight=1)
    G.add_edge("Feuilly", "Marius", weight=1)
    G.add_edge("Courfeyrac", "Marius", weight=9)
    G.add_edge("Courfeyrac", "Enjolras", weight=17)
    G.add_edge("Courfeyrac", "Combeferre", weight=13)
    G.add_edge("Courfeyrac", "Gavroche", weight=7)
    G.add_edge("Courfeyrac", "Mabeuf", weight=2)
    G.add_edge("Courfeyrac", "Eponine", weight=1)
    G.add_edge("Courfeyrac", "Feuilly", weight=6)
    G.add_edge("Courfeyrac", "Prouvaire", weight=3)
    G.add_edge("Bahorel", "Combeferre", weight=5)
    G.add_edge("Bahorel", "Gavroche", weight=5)
    G.add_edge("Bahorel", "Courfeyrac", weight=6)
    G.add_edge("Bahorel", "Mabeuf", weight=2)
    G.add_edge("Bahorel", "Enjolras", weight=4)
    G.add_edge("Bahorel", "Feuilly", weight=3)
    G.add_edge("Bahorel", "Prouvaire", weight=2)
    G.add_edge("Bahorel", "Marius", weight=1)
    G.add_edge("Bossuet", "Marius", weight=5)
    G.add_edge("Bossuet", "Courfeyrac", weight=12)
    G.add_edge("Bossuet", "Gavroche", weight=5)
    G.add_edge("Bossuet", "Bahorel", weight=4)
    G.add_edge("Bossuet", "Enjolras", weight=10)
    G.add_edge("Bossuet", "Feuilly", weight=6)
    G.add_edge("Bossuet", "Prouvaire", weight=2)
    G.add_edge("Bossuet", "Combeferre", weight=9)
    G.add_edge("Bossuet", "Mabeuf", weight=1)
    G.add_edge("Bossuet", "Valjean", weight=1)
    G.add_edge("Joly", "Bahorel", weight=5)
    G.add_edge("Joly", "Bossuet", weight=7)
    G.add_edge("Joly", "Gavroche", weight=3)
    G.add_edge("Joly", "Courfeyrac", weight=5)
    G.add_edge("Joly", "Enjolras", weight=5)
    G.add_edge("Joly", "Feuilly", weight=5)
    G.add_edge("Joly", "Prouvaire", weight=2)
    G.add_edge("Joly", "Combeferre", weight=5)
    G.add_edge("Joly", "Mabeuf", weight=1)
    G.add_edge("Joly", "Marius", weight=2)
    G.add_edge("Grantaire", "Bossuet", weight=3)
    G.add_edge("Grantaire", "Enjolras", weight=3)
    G.add_edge("Grantaire", "Combeferre", weight=1)
    G.add_edge("Grantaire", "Courfeyrac", weight=2)
    G.add_edge("Grantaire", "Joly", weight=2)
    G.add_edge("Grantaire", "Gavroche", weight=1)
    G.add_edge("Grantaire", "Bahorel", weight=1)
    G.add_edge("Grantaire", "Feuilly", weight=1)
    G.add_edge("Grantaire", "Prouvaire", weight=1)
    G.add_edge("MotherPlutarch", "Mabeuf", weight=3)
    G.add_edge("Gueulemer", "Thenardier", weight=5)
    G.add_edge("Gueulemer", "Valjean", weight=1)
    G.add_edge("Gueulemer", "MmeThenardier", weight=1)
    G.add_edge("Gueulemer", "Javert", weight=1)
    G.add_edge("Gueulemer", "Gavroche", weight=1)
    G.add_edge("Gueulemer", "Eponine", weight=1)
    G.add_edge("Babet", "Thenardier", weight=6)
    G.add_edge("Babet", "Gueulemer", weight=6)
    G.add_edge("Babet", "Valjean", weight=1)
    G.add_edge("Babet", "MmeThenardier", weight=1)
    G.add_edge("Babet", "Javert", weight=2)
    G.add_edge("Babet", "Gavroche", weight=1)
    G.add_edge("Babet", "Eponine", weight=1)
    G.add_edge("Claquesous", "Thenardier", weight=4)
    G.add_edge("Claquesous", "Babet", weight=4)
    G.add_edge("Claquesous", "Gueulemer", weight=4)
    G.add_edge("Claquesous", "Valjean", weight=1)
    G.add_edge("Claquesous", "MmeThenardier", weight=1)
    G.add_edge("Claquesous", "Javert", weight=1)
    G.add_edge("Claquesous", "Eponine", weight=1)
    G.add_edge("Claquesous", "Enjolras", weight=1)
    G.add_edge("Montparnasse", "Javert", weight=1)
    G.add_edge("Montparnasse", "Babet", weight=2)
    G.add_edge("Montparnasse", "Gueulemer", weight=2)
    G.add_edge("Montparnasse", "Claquesous", weight=2)
    G.add_edge("Montparnasse", "Valjean", weight=1)
    G.add_edge("Montparnasse", "Gavroche", weight=1)
    G.add_edge("Montparnasse", "Eponine", weight=1)
    G.add_edge("Montparnasse", "Thenardier", weight=1)
    G.add_edge("Toussaint", "Cosette", weight=2)
    G.add_edge("Toussaint", "Javert", weight=1)
    G.add_edge("Toussaint", "Valjean", weight=1)
    G.add_edge("Child1", "Gavroche", weight=2)
    G.add_edge("Child2", "Gavroche", weight=2)
    G.add_edge("Child2", "Child1", weight=3)
    G.add_edge("Brujon", "Babet", weight=3)
    G.add_edge("Brujon", "Gueulemer", weight=3)
    G.add_edge("Brujon", "Thenardier", weight=3)
    G.add_edge("Brujon", "Gavroche", weight=1)
    G.add_edge("Brujon", "Eponine", weight=1)
    G.add_edge("Brujon", "Claquesous", weight=1)
    G.add_edge("Brujon", "Montparnasse", weight=1)
    G.add_edge("MmeHucheloup", "Bossuet", weight=1)
    G.add_edge("MmeHucheloup", "Joly", weight=1)
    G.add_edge("MmeHucheloup", "Grantaire", weight=1)
    G.add_edge("MmeHucheloup", "Bahorel", weight=1)
    G.add_edge("MmeHucheloup", "Courfeyrac", weight=1)
    G.add_edge("MmeHucheloup", "Gavroche", weight=1)
    G.add_edge("MmeHucheloup", "Enjolras", weight=1)
    return G
<|endoftext|>"""
Generators for random graphs.

"""

import itertools
import math
from collections import defaultdict

import networkx as nx
from networkx.utils import py_random_state

from ..utils.misc import check_create_using
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree

__all__ = [
    "fast_gnp_random_graph",
    "gnp_random_graph",
    "dense_gnm_random_graph",
    "gnm_random_graph",
    "erdos_renyi_graph",
    "binomial_graph",
    "newman_watts_strogatz_graph",
    "watts_strogatz_graph",
    "connected_watts_strogatz_graph",
    "random_regular_graph",
    "barabasi_albert_graph",
    "dual_barabasi_albert_graph",
    "extended_barabasi_albert_graph",
    "powerlaw_cluster_graph",
    "random_lobster",
    "random_shell_graph",
    "random_powerlaw_tree",
    "random_powerlaw_tree_sequence",
    "random_kernel_graph",
]


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def fast_gnp_random_graph(n, p, seed=None, directed=False, *, create_using=None):
    """Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph or
    a binomial graph.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True, this function returns a directed graph.
    create_using : Graph constructor, optional (default=nx.Graph or nx.DiGraph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph types are not supported and raise a ``NetworkXError``.
        By default NetworkX Graph or DiGraph are used depending on `directed`.

    Notes
    -----
    The $G_{n,p}$ graph algorithm chooses each of the $[n (n - 1)] / 2$
    (undirected) or $n (n - 1)$ (directed) possible edges with probability $p$.

    This algorithm [1]_ runs in $O(n + m)$ time, where `m` is the expected number of
    edges, which equals $p n (n - 1) / 2$. This should be faster than
    :func:`gnp_random_graph` when $p$ is small and the expected number of edges
    is small (that is, the graph is sparse).

    See Also
    --------
    gnp_random_graph

    References
    ----------
    .. [1] Vladimir Batagelj and Ulrik Brandes,
       "Efficient generation of large random networks",
       Phys. Rev. E, 71, 036113, 2005.
    """
    default = nx.DiGraph if directed else nx.Graph
    create_using = check_create_using(
        create_using, directed=directed, multigraph=False, default=default
    )
    if p <= 0 or p >= 1:
        return nx.gnp_random_graph(
            n, p, seed=seed, directed=directed, create_using=create_using
        )

    G = empty_graph(n, create_using=create_using)

    lp = math.log(1.0 - p)

    if directed:
        v = 1
        w = -1
        while v < n:
            lr = math.log(1.0 - seed.random())
            w = w + 1 + int(lr / lp)
            while w >= v and v < n:
                w = w - v
                v = v + 1
            if v < n:
                G.add_edge(w, v)

    # Nodes in graph are from 0,n-1 (start with v as the second node index).
    v = 1
    w = -1
    while v < n:
        lr = math.log(1.0 - seed.random())
        w = w + 1 + int(lr / lp)
        while w >= v and v < n:
            w = w - v
            v = v + 1
        if v < n:
            G.add_edge(v, w)
    return G


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def gnp_random_graph(n, p, seed=None, directed=False, *, create_using=None):
    """Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
    or a binomial graph.

    The $G_{n,p}$ model chooses each of the possible edges with probability $p$.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True, this function returns a directed graph.
    create_using : Graph constructor, optional (default=nx.Graph or nx.DiGraph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph types are not supported and raise a ``NetworkXError``.
        By default NetworkX Graph or DiGraph are used depending on `directed`.

    See Also
    --------
    fast_gnp_random_graph

    Notes
    -----
    This algorithm [2]_ runs in $O(n^2)$ time.  For sparse graphs (that is, for
    small values of $p$), :func:`fast_gnp_random_graph` is a faster algorithm.

    :func:`binomial_graph` and :func:`erdos_renyi_graph` are
    aliases for :func:`gnp_random_graph`.

    >>> nx.binomial_graph is nx.gnp_random_graph
    True
    >>> nx.erdos_renyi_graph is nx.gnp_random_graph
    True

    References
    ----------
    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
    """
    default = nx.DiGraph if directed else nx.Graph
    create_using = check_create_using(
        create_using, directed=directed, multigraph=False, default=default
    )
    if p >= 1:
        return complete_graph(n, create_using=create_using)

    G = nx.empty_graph(n, create_using=create_using)
    if p <= 0:
        return G

    edgetool = itertools.permutations if directed else itertools.combinations
    for e in edgetool(range(n), 2):
        if seed.random() < p:
            G.add_edge(*e)
    return G


# add some aliases to common names
binomial_graph = gnp_random_graph
erdos_renyi_graph = gnp_random_graph


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def dense_gnm_random_graph(n, m, seed=None, *, create_using=None):
    """Returns a $G_{n,m}$ random graph.

    In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set
    of all graphs with $n$ nodes and $m$ edges.

    This algorithm should be faster than :func:`gnm_random_graph` for dense
    graphs.

    Parameters
    ----------
    n : int
        The number of nodes.
    m : int
        The number of edges.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    See Also
    --------
    gnm_random_graph

    Notes
    -----
    Algorithm by Keith M. Briggs Mar 31, 2006.
    Inspired by Knuth's Algorithm S (Selection sampling technique),
    in section 3.4.2 of [1]_.

    References
    ----------
    .. [1] Donald E. Knuth, The Art of Computer Programming,
        Volume 2/Seminumerical algorithms, Third Edition, Addison-Wesley, 1997.
    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    mmax = n * (n - 1) // 2
    if m >= mmax:
        return complete_graph(n, create_using)
    G = empty_graph(n, create_using)

    if n == 1:
        return G

    u = 0
    v = 1
    t = 0
    k = 0
    while True:
        if seed.randrange(mmax - t) < m - k:
            G.add_edge(u, v)
            k += 1
            if k == m:
                return G
        t += 1
        v += 1
        if v == n:  # go to next row of adjacency matrix
            u += 1
            v = u + 1


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def gnm_random_graph(n, m, seed=None, directed=False, *, create_using=None):
    """Returns a $G_{n,m}$ random graph.

    In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set
    of all graphs with $n$ nodes and $m$ edges.

    This algorithm should be faster than :func:`dense_gnm_random_graph` for
    sparse graphs.

    Parameters
    ----------
    n : int
        The number of nodes.
    m : int
        The number of edges.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True return a directed graph
    create_using : Graph constructor, optional (default=nx.Graph or nx.DiGraph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph types are not supported and raise a ``NetworkXError``.
        By default NetworkX Graph or DiGraph are used depending on `directed`.

    See also
    --------
    dense_gnm_random_graph

    """
    default = nx.DiGraph if directed else nx.Graph
    create_using = check_create_using(
        create_using, directed=directed, multigraph=False, default=default
    )
    if n == 1:
        return nx.empty_graph(n, create_using=create_using)
    max_edges = n * (n - 1) if directed else n * (n - 1) / 2.0
    if m >= max_edges:
        return complete_graph(n, create_using=create_using)

    G = nx.empty_graph(n, create_using=create_using)
    nlist = list(G)
    edge_count = 0
    while edge_count < m:
        # generate random edge,u,v
        u = seed.choice(nlist)
        v = seed.choice(nlist)
        if u == v or G.has_edge(u, v):
            continue
        else:
            G.add_edge(u, v)
            edge_count = edge_count + 1
    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def newman_watts_strogatz_graph(n, k, p, seed=None, *, create_using=None):
    """Returns a Newman–Watts–Strogatz small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes.
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of adding a new edge for each edge.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is
    connected with its $k$ nearest neighbors (or $k - 1$ neighbors if $k$
    is odd).  Then shortcuts are created by adding new edges as follows: for
    each edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest
    neighbors" with probability $p$ add a new edge $(u, w)$ with
    randomly-chosen existing node $w$.  In contrast with
    :func:`watts_strogatz_graph`, no edges are removed.

    See Also
    --------
    watts_strogatz_graph

    References
    ----------
    .. [1] M. E. J. Newman and D. J. Watts,
       Renormalization group analysis of the small-world network model,
       Physics Letters A, 263, 341, 1999.
       https://doi.org/10.1016/S0375-9601(99)00757-4
    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    if k > n:
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")

    # If k == n the graph return is a complete graph
    if k == n:
        return nx.complete_graph(n, create_using)

    G = empty_graph(n, create_using)
    nlist = list(G.nodes())
    fromv = nlist
    # connect the k/2 neighbors
    for j in range(1, k // 2 + 1):
        tov = fromv[j:] + fromv[0:j]  # the first j are now last
        for i in range(len(fromv)):
            G.add_edge(fromv[i], tov[i])
    # for each edge u-v, with probability p, randomly select existing
    # node w and add new edge u-w
    e = list(G.edges())
    for u, v in e:
        if seed.random() < p:
            w = seed.choice(nlist)
            # no self-loops and reject if edge u-w exists
            # is that the correct NWS model?
            while w == u or G.has_edge(u, w):
                w = seed.choice(nlist)
                if G.degree(u) >= n - 1:
                    break  # skip this rewiring
            else:
                G.add_edge(u, w)
    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def watts_strogatz_graph(n, k, p, seed=None, *, create_using=None):
    """Returns a Watts–Strogatz small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    See Also
    --------
    newman_watts_strogatz_graph
    connected_watts_strogatz_graph

    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
    to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
    Then shortcuts are created by replacing some edges as follows: for each
    edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
    with probability $p$ replace it with a new edge $(u, w)$ with uniformly
    random choice of existing node $w$.

    In contrast with :func:`newman_watts_strogatz_graph`, the random rewiring
    does not increase the number of edges. The rewired graph is not guaranteed
    to be connected as in :func:`connected_watts_strogatz_graph`.

    References
    ----------
    .. [1] Duncan J. Watts and Steven H. Strogatz,
       Collective dynamics of small-world networks,
       Nature, 393, pp. 440--442, 1998.
    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    if k > n:
        raise nx.NetworkXError("k>n, choose smaller k or larger n")

    # If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        G = nx.complete_graph(n, create_using)
        return G

    G = nx.empty_graph(n, create_using=create_using)
    nodes = list(range(n))  # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, k // 2 + 1):  # outer loop is neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        # inner loop in node order
        for u, v in zip(nodes, targets):
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    return G


@py_random_state(4)
@nx._dispatchable(graphs=None, returns_graph=True)
def connected_watts_strogatz_graph(n, k, p, tries=100, seed=None, *, create_using=None):
    """Returns a connected Watts–Strogatz small-world graph.

    Attempts to generate a connected graph by repeated generation of
    Watts–Strogatz small-world graphs.  An exception is raised if the maximum
    number of tries is exceeded.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    tries : int
        Number of attempts to generate a connected graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
    to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
    Then shortcuts are created by replacing some edges as follows: for each
    edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
    with probability $p$ replace it with a new edge $(u, w)$ with uniformly
    random choice of existing node $w$.
    The entire process is repeated until a connected graph results.

    See Also
    --------
    newman_watts_strogatz_graph
    watts_strogatz_graph

    References
    ----------
    .. [1] Duncan J. Watts and Steven H. Strogatz,
       Collective dynamics of small-world networks,
       Nature, 393, pp. 440--442, 1998.
    """
    for i in range(tries):
        # seed is an RNG so should change sequence each call
        G = watts_strogatz_graph(n, k, p, seed, create_using=create_using)
        if nx.is_connected(G):
            return G
    raise nx.NetworkXError("Maximum number of tries exceeded")


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_regular_graph(d, n, seed=None, *, create_using=None):
    r"""Returns a random $d$-regular graph on $n$ nodes.

    A regular graph is a graph where each node has the same number of neighbors.

    The resulting graph has no self-loops or parallel edges.

    Parameters
    ----------
    d : int
      The degree of each node.
    n : integer
      The number of nodes. The value of $n \times d$ must be even.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Notes
    -----
    The nodes are numbered from $0$ to $n - 1$.

    Kim and Vu's paper [2]_ shows that this algorithm samples in an
    asymptotically uniform way from the space of random graphs when
    $d = O(n^{1 / 3 - \epsilon})$.

    Raises
    ------

    NetworkXError
        If $n \times d$ is odd or $d$ is greater than or equal to $n$.

    References
    ----------
    .. [1] A. Steger and N. Wormald,
       Generating random regular graphs quickly,
       Probability and Computing 8 (1999), 377-396, 1999.
       https://doi.org/10.1017/S0963548399003867

    .. [2] Jeong Han Kim and Van H. Vu,
       Generating random regular graphs,
       Proceedings of the thirty-fifth ACM symposium on Theory of computing,
       San Diego, CA, USA, pp 213--222, 2003.
       http://portal.acm.org/citation.cfm?id=780542.780576
    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    if (n * d) % 2 != 0:
        raise nx.NetworkXError("n * d must be even")

    if not 0 <= d < n:
        raise nx.NetworkXError("the 0 <= d < n inequality must be satisfied")

    G = nx.empty_graph(n, create_using=create_using)

    if d == 0:
        return G

    def _suitable(edges, potential_edges):
        # Helper subroutine to check if there are suitable edges remaining
        # If False, the generation of the graph has failed
        if not potential_edges:
            return True
        for s1 in potential_edges:
            for s2 in potential_edges:
                # Two iterators on the same dictionary are guaranteed
                # to visit it in the same order if there are no
                # intervening modifications.
                if s1 == s2:
                    # Only need to consider s1-s2 pair one time
                    break
                if s1 > s2:
                    s1, s2 = s2, s1
                if (s1, s2) not in edges:
                    return True
        return False

    def _try_creation():
        # Attempt to create an edge set

        edges = set()
        stubs = list(range(n)) * d

        while stubs:
            potential_edges = defaultdict(lambda: 0)
            seed.shuffle(stubs)
            stubiter = iter(stubs)
            for s1, s2 in zip(stubiter, stubiter):
                if s1 > s2:
                    s1, s2 = s2, s1
                if s1 != s2 and ((s1, s2) not in edges):
                    edges.add((s1, s2))
                else:
                    potential_edges[s1] += 1
                    potential_edges[s2] += 1

            if not _suitable(edges, potential_edges):
                return None  # failed to find suitable edge set

            stubs = [
                node
                for node, potential in potential_edges.items()
                for _ in range(potential)
            ]
        return edges

    # Even though a suitable edge set exists,
    # the generation of such a set is not guaranteed.
    # Try repeatedly to find one.
    edges = _try_creation()
    while edges is None:
        edges = _try_creation()
    G.add_edges_from(edges)

    return G


def _random_subset(seq, m, rng):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def barabasi_albert_graph(n, m, seed=None, initial_graph=None, *, create_using=None):
    """Returns a random graph using Barabási–Albert preferential attachment

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    initial_graph : Graph or None (default)
        Initial network for Barabási–Albert algorithm.
        It should be a connected graph for most use cases.
        A copy of `initial_graph` is used.
        If None, starts from a star graph on (m+1) nodes.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n``, or
        the initial graph number of nodes m0 does not satisfy ``m <= m0 <= n``.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
        )

    if initial_graph is None:
        # Default initial graph : star graph on (m + 1) nodes
        G = star_graph(m, create_using)
    else:
        if len(initial_graph) < m or len(initial_graph) > n:
            raise nx.NetworkXError(
                f"Barabási–Albert initial graph needs between m={m} and n={n} nodes"
            )
        G = initial_graph.copy()

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    # Start adding the other n - m0 nodes.
    source = len(G)
    while source < n:
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)

        source += 1
    return G


@py_random_state(4)
@nx._dispatchable(graphs=None, returns_graph=True)
def dual_barabasi_albert_graph(
    n, m1, m2, p, seed=None, initial_graph=None, *, create_using=None
):
    """Returns a random graph using dual Barabási–Albert preferential attachment

    A graph of $n$ nodes is grown by attaching new nodes each with either $m_1$
    edges (with probability $p$) or $m_2$ edges (with probability $1-p$) that
    are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m1 : int
        Number of edges to link each new node to existing nodes with probability $p$
    m2 : int
        Number of edges to link each new node to existing nodes with probability $1-p$
    p : float
        The probability of attaching $m_1$ edges (as opposed to $m_2$ edges)
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    initial_graph : Graph or None (default)
        Initial network for Barabási–Albert algorithm.
        A copy of `initial_graph` is used.
        It should be connected for most use cases.
        If None, starts from an star graph on max(m1, m2) + 1 nodes.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m1` and `m2` do not satisfy ``1 <= m1,m2 < n``, or
        `p` does not satisfy ``0 <= p <= 1``, or
        the initial graph number of nodes m0 does not satisfy m1, m2 <= m0 <= n.

    References
    ----------
    .. [1] N. Moshiri "The dual-Barabasi-Albert model", arXiv:1810.10538.
    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    if m1 < 1 or m1 >= n:
        raise nx.NetworkXError(
            f"Dual Barabási–Albert must have m1 >= 1 and m1 < n, m1 = {m1}, n = {n}"
        )
    if m2 < 1 or m2 >= n:
        raise nx.NetworkXError(
            f"Dual Barabási–Albert must have m2 >= 1 and m2 < n, m2 = {m2}, n = {n}"
        )
    if p < 0 or p > 1:
        raise nx.NetworkXError(
            f"Dual Barabási–Albert network must have 0 <= p <= 1, p = {p}"
        )

    # For simplicity, if p == 0 or 1, just return BA
    if p == 1:
        return barabasi_albert_graph(n, m1, seed, create_using=create_using)
    elif p == 0:
        return barabasi_albert_graph(n, m2, seed, create_using=create_using)

    if initial_graph is None:
        # Default initial graph : star graph on max(m1, m2) nodes
        G = star_graph(max(m1, m2), create_using)
    else:
        if len(initial_graph) < max(m1, m2) or len(initial_graph) > n:
            raise nx.NetworkXError(
                f"Barabási–Albert initial graph must have between "
                f"max(m1, m2) = {max(m1, m2)} and n = {n} nodes"
            )
        G = initial_graph.copy()

    # Target nodes for new edges
    targets = list(G)
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    # Start adding the remaining nodes.
    source = len(G)
    while source < n:
        # Pick which m to use (m1 or m2)
        if seed.random() < p:
            m = m1
        else:
            m = m2
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)

        source += 1
    return G


@py_random_state(4)
@nx._dispatchable(graphs=None, returns_graph=True)
def extended_barabasi_albert_graph(n, m, p, q, seed=None, *, create_using=None):
    """Returns an extended Barabási–Albert model graph.

    An extended Barabási–Albert model graph is a random graph constructed
    using preferential attachment. The extended model allows new edges,
    rewired edges or new nodes. Based on the probabilities $p$ and $q$
    with $p + q < 1$, the growing behavior of the graph is determined as:

    1) With $p$ probability, $m$ new edges are added to the graph,
    starting from randomly chosen existing nodes and attached preferentially at the
    other end.

    2) With $q$ probability, $m$ existing edges are rewired
    by randomly choosing an edge and rewiring one end to a preferentially chosen node.

    3) With $(1 - p - q)$ probability, $m$ new nodes are added to the graph
    with edges attached preferentially.

    When $p = q = 0$, the model behaves just like the Barabási–Alber model.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges with which a new node attaches to existing nodes
    p : float
        Probability value for adding an edge between existing nodes. p + q < 1
    q : float
        Probability value of rewiring of existing edges. p + q < 1
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n`` or ``1 >= p + q``

    References
    ----------
    .. [1] Albert, R., & Barabási, A. L. (2000)
       Topology of evolving networks: local events and universality
       Physical review letters, 85(24), 5234.
    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    if m < 1 or m >= n:
        msg = f"Extended Barabasi-Albert network needs m>=1 and m<n, m={m}, n={n}"
        raise nx.NetworkXError(msg)
    if p + q >= 1:
        msg = f"Extended Barabasi-Albert network needs p + q <= 1, p={p}, q={q}"
        raise nx.NetworkXError(msg)

    # Add m initial nodes (m0 in barabasi-speak)
    G = empty_graph(m, create_using)

    # List of nodes to represent the preferential attachment random selection.
    # At the creation of the graph, all nodes are added to the list
    # so that even nodes that are not connected have a chance to get selected,
    # for rewiring and adding of edges.
    # With each new edge, nodes at the ends of the edge are added to the list.
    attachment_preference = []
    attachment_preference.extend(range(m))

    # Start adding the other n-m nodes. The first node is m.
    new_node = m
    while new_node < n:
        a_probability = seed.random()

        # Total number of edges of a Clique of all the nodes
        clique_degree = len(G) - 1
        clique_size = (len(G) * clique_degree) / 2

        # Adding m new edges, if there is room to add them
        if a_probability < p and G.size() <= clique_size - m:
            # Select the nodes where an edge can be added
            eligible_nodes = [nd for nd, deg in G.degree() if deg < clique_degree]
            for i in range(m):
                # Choosing a random source node from eligible_nodes
                src_node = seed.choice(eligible_nodes)

                # Picking a possible node that is not 'src_node' or
                # neighbor with 'src_node', with preferential attachment
                prohibited_nodes = list(G[src_node])
                prohibited_nodes.append(src_node)
                # This will raise an exception if the sequence is empty
                dest_node = seed.choice(
                    [nd for nd in attachment_preference if nd not in prohibited_nodes]
                )
                # Adding the new edge
                G.add_edge(src_node, dest_node)

                # Appending both nodes to add to their preferential attachment
                attachment_preference.append(src_node)
                attachment_preference.append(dest_node)

                # Adjusting the eligible nodes. Degree may be saturated.
                if G.degree(src_node) == clique_degree:
                    eligible_nodes.remove(src_node)
                if G.degree(dest_node) == clique_degree and dest_node in eligible_nodes:
                    eligible_nodes.remove(dest_node)

        # Rewiring m edges, if there are enough edges
        elif p <= a_probability < (p + q) and m <= G.size() < clique_size:
            # Selecting nodes that have at least 1 edge but that are not
            # fully connected to ALL other nodes (center of star).
            # These nodes are the pivot nodes of the edges to rewire
            eligible_nodes = [nd for nd, deg in G.degree() if 0 < deg < clique_degree]
            for i in range(m):
                # Choosing a random source node
                node = seed.choice(eligible_nodes)

                # The available nodes do have a neighbor at least.
                nbr_nodes = list(G[node])

                # Choosing the other end that will get detached
                src_node = seed.choice(nbr_nodes)

                # Picking a target node that is not 'node' or
                # neighbor with 'node', with preferential attachment
                nbr_nodes.append(node)
                dest_node = seed.choice(
                    [nd for nd in attachment_preference if nd not in nbr_nodes]
                )
                # Rewire
                G.remove_edge(node, src_node)
                G.add_edge(node, dest_node)

                # Adjusting the preferential attachment list
                attachment_preference.remove(src_node)
                attachment_preference.append(dest_node)

                # Adjusting the eligible nodes.
                # nodes may be saturated or isolated.
                if G.degree(src_node) == 0 and src_node in eligible_nodes:
                    eligible_nodes.remove(src_node)
                if dest_node in eligible_nodes:
                    if G.degree(dest_node) == clique_degree:
                        eligible_nodes.remove(dest_node)
                else:
                    if G.degree(dest_node) == 1:
                        eligible_nodes.append(dest_node)

        # Adding new node with m edges
        else:
            # Select the edges' nodes by preferential attachment
            targets = _random_subset(attachment_preference, m, seed)
            G.add_edges_from(zip([new_node] * m, targets))

            # Add one node to the list for each new edge just created.
            attachment_preference.extend(targets)
            # The new node has m edges to it, plus itself: m + 1
            attachment_preference.extend([new_node] * (m + 1))
            new_node += 1
    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def powerlaw_cluster_graph(n, m, p, seed=None, *, create_using=None):
    """Holme and Kim algorithm for growing graphs with powerlaw
    degree distribution and approximate average clustering.

    Parameters
    ----------
    n : int
        the number of nodes
    m : int
        the number of random edges to add for each new node
    p : float,
        Probability of adding a triangle after adding a random edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Notes
    -----
    The average clustering has a hard time getting above a certain
    cutoff that depends on `m`.  This cutoff is often quite low.  The
    transitivity (fraction of triangles to possible triangles) seems to
    decrease with network size.

    It is essentially the Barabási–Albert (BA) growth model with an
    extra step that each random edge is followed by a chance of
    making an edge to one of its neighbors too (and thus a triangle).

    This algorithm improves on BA in the sense that it enables a
    higher average clustering to be attained if desired.

    It seems possible to have a disconnected graph with this algorithm
    since the initial `m` nodes may not be all linked to a new node
    on the first iteration like the BA model.

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m <= n`` or `p` does not
        satisfy ``0 <= p <= 1``.

    References
    ----------
    .. [1] P. Holme and B. J. Kim,
       "Growing scale-free networks with tunable clustering",
       Phys. Rev. E, 65, 026107, 2002.
    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    if m < 1 or n < m:
        raise nx.NetworkXError(f"NetworkXError must have m>1 and m<n, m={m},n={n}")

    if p > 1 or p < 0:
        raise nx.NetworkXError(f"NetworkXError p must be in [0,1], p={p}")

    G = empty_graph(m, create_using)  # add m initial nodes (m0 in barabasi-speak)
    repeated_nodes = list(G)  # list of existing nodes to sample from
    # with nodes repeated once for each adjacent edge
    source = m  # next node is m
    while source < n:  # Now add the other n-1 nodes
        possible_targets = _random_subset(repeated_nodes, m, seed)
        # do one preferential attachment for new node
        target = possible_targets.pop()
        G.add_edge(source, target)
        repeated_nodes.append(target)  # add one node to list for each new link
        count = 1
        while count < m:  # add m-1 more new links
            if seed.random() < p:  # clustering step: add triangle
                neighborhood = [
                    nbr
                    for nbr in G.neighbors(target)
                    if not G.has_edge(source, nbr) and nbr != source
                ]
                if neighborhood:  # if there is a neighbor without a link
                    nbr = seed.choice(neighborhood)
                    G.add_edge(source, nbr)  # add triangle
                    repeated_nodes.append(nbr)
                    count = count + 1
                    continue  # go to top of while loop
            # else do preferential attachment step if above fails
            target = possible_targets.pop()
            G.add_edge(source, target)
            repeated_nodes.append(target)
            count = count + 1

        repeated_nodes.extend([source] * m)  # add source node to list m times
        source += 1
    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_lobster(n, p1, p2, seed=None, *, create_using=None):
    """Returns a random lobster graph.

    A lobster is a tree that reduces to a caterpillar when pruning all
    leaf nodes. A caterpillar is a tree that reduces to a path graph
    when pruning all leaf nodes; setting `p2` to zero produces a caterpillar.

    This implementation iterates on the probabilities `p1` and `p2` to add
    edges at levels 1 and 2, respectively. Graphs are therefore constructed
    iteratively with uniform randomness at each level rather than being selected
    uniformly at random from the set of all possible lobsters.

    Parameters
    ----------
    n : int
        The expected number of nodes in the backbone
    p1 : float
        Probability of adding an edge to the backbone
    p2 : float
        Probability of adding an edge one level beyond backbone
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : Graph constructor, optional (default=nx.Grap)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Raises
    ------
    NetworkXError
        If `p1` or `p2` parameters are >= 1 because the while loops would never finish.
    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    p1, p2 = abs(p1), abs(p2)
    if any(p >= 1 for p in [p1, p2]):
        raise nx.NetworkXError("Probability values for `p1` and `p2` must both be < 1.")

    # a necessary ingredient in any self-respecting graph library
    llen = int(2 * seed.random() * n + 0.5)
    L = path_graph(llen, create_using)
    # build caterpillar: add edges to path graph with probability p1
    current_node = llen - 1
    for n in range(llen):
        while seed.random() < p1:  # add fuzzy caterpillar parts
            current_node += 1
            L.add_edge(n, current_node)
            cat_node = current_node
            while seed.random() < p2:  # add crunchy lobster bits
                current_node += 1
                L.add_edge(cat_node, current_node)
    return L  # voila, un lobster!


@py_random_state(1)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_shell_graph(constructor, seed=None, *, create_using=None):
    """Returns a random shell graph for the constructor given.

    Parameters
    ----------
    constructor : list of three-tuples
        Represents the parameters for a shell, starting at the center
        shell.  Each element of the list must be of the form `(n, m,
        d)`, where `n` is the number of nodes in the shell, `m` is
        the number of edges in the shell, and `d` is the ratio of
        inter-shell (next) edges to intra-shell edges. If `d` is zero,
        there will be no intra-shell edges, and if `d` is one there
        will be all possible intra-shell edges.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. Graph instances are not supported.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Examples
    --------
    >>> constructor = [(10, 20, 0.8), (20, 40, 0.8)]
    >>> G = nx.random_shell_graph(constructor)

    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    G = empty_graph(0, create_using)

    glist = []
    intra_edges = []
    nnodes = 0
    # create gnm graphs for each shell
    for n, m, d in constructor:
        inter_edges = int(m * d)
        intra_edges.append(m - inter_edges)
        g = nx.convert_node_labels_to_integers(
            gnm_random_graph(n, inter_edges, seed=seed, create_using=G.__class__),
            first_label=nnodes,
        )
        glist.append(g)
        nnodes += n
        G = nx.operators.union(G, g)

    # connect the shells randomly
    for gi in range(len(glist) - 1):
        nlist1 = list(glist[gi])
        nlist2 = list(glist[gi + 1])
        total_edges = intra_edges[gi]
        edge_count = 0
        while edge_count < total_edges:
            u = seed.choice(nlist1)
            v = seed.choice(nlist2)
            if u == v or G.has_edge(u, v):
                continue
            else:
                G.add_edge(u, v)
                edge_count = edge_count + 1
    return G


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_powerlaw_tree(n, gamma=3, seed=None, tries=100, *, create_using=None):
    """Returns a tree with a power law degree distribution.

    Parameters
    ----------
    n : int
        The number of nodes.
    gamma : float
        Exponent of the power law.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    tries : int
        Number of attempts to adjust the sequence to make it a tree.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Raises
    ------
    NetworkXError
        If no valid sequence is found within the maximum number of
        attempts.

    Notes
    -----
    A trial power law degree sequence is chosen and then elements are
    swapped with new elements from a powerlaw distribution until the
    sequence makes a tree (by checking, for example, that the number of
    edges is one smaller than the number of nodes).

    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    # This call may raise a NetworkXError if the number of tries is succeeded.
    seq = random_powerlaw_tree_sequence(n, gamma=gamma, seed=seed, tries=tries)
    G = degree_sequence_tree(seq, create_using)
    return G


@py_random_state(2)
@nx._dispatchable(graphs=None)
def random_powerlaw_tree_sequence(n, gamma=3, seed=None, tries=100):
    """Returns a degree sequence for a tree with a power law distribution.

    Parameters
    ----------
    n : int,
        The number of nodes.
    gamma : float
        Exponent of the power law.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    tries : int
        Number of attempts to adjust the sequence to make it a tree.

    Raises
    ------
    NetworkXError
        If no valid sequence is found within the maximum number of
        attempts.

    Notes
    -----
    A trial power law degree sequence is chosen and then elements are
    swapped with new elements from a power law distribution until
    the sequence makes a tree (by checking, for example, that the number of
    edges is one smaller than the number of nodes).

    """
    # get trial sequence
    z = nx.utils.powerlaw_sequence(n, exponent=gamma, seed=seed)
    # round to integer values in the range [0,n]
    zseq = [min(n, max(round(s), 0)) for s in z]

    # another sequence to swap values from
    z = nx.utils.powerlaw_sequence(tries, exponent=gamma, seed=seed)
    # round to integer values in the range [0,n]
    swap = [min(n, max(round(s), 0)) for s in z]

    for deg in swap:
        # If this degree sequence can be the degree sequence of a tree, return
        # it. It can be a tree if the number of edges is one fewer than the
        # number of nodes, or in other words, `n - sum(zseq) / 2 == 1`. We
        # use an equivalent condition below that avoids floating point
        # operations.
        if 2 * n - sum(zseq) == 2:
            return zseq
        index = seed.randint(0, n - 1)
        zseq[index] = swap.pop()

    raise nx.NetworkXError(
        f"Exceeded max ({tries}) attempts for a valid tree sequence."
    )


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_kernel_graph(
    n, kernel_integral, kernel_root=None, seed=None, *, create_using=None
):
    r"""Returns an random graph based on the specified kernel.

    The algorithm chooses each of the $[n(n-1)]/2$ possible edges with
    probability specified by a kernel $\kappa(x,y)$ [1]_.  The kernel
    $\kappa(x,y)$ must be a symmetric (in $x,y$), non-negative,
    bounded function.

    Parameters
    ----------
    n : int
        The number of nodes
    kernel_integral : function
        Function that returns the definite integral of the kernel $\kappa(x,y)$,
        $F(y,a,b) := \int_a^b \kappa(x,y)dx$
    kernel_root: function (optional)
        Function that returns the root $b$ of the equation $F(y,a,b) = r$.
        If None, the root is found using :func:`scipy.optimize.brentq`
        (this requires SciPy).
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Notes
    -----
    The kernel is specified through its definite integral which must be
    provided as one of the arguments. If the integral and root of the
    kernel integral can be found in $O(1)$ time then this algorithm runs in
    time $O(n+m)$ where m is the expected number of edges [2]_.

    The nodes are set to integers from $0$ to $n-1$.

    Examples
    --------
    Generate an Erdős–Rényi random graph $G(n,c/n)$, with kernel
    $\kappa(x,y)=c$ where $c$ is the mean expected degree.

    >>> def integral(u, w, z):
    ...     return c * (z - w)
    >>> def root(u, w, r):
    ...     return r / c + w
    >>> c = 1
    >>> graph = nx.random_kernel_graph(1000, integral, root)

    See Also
    --------
    gnp_random_graph
    expected_degree_graph

    References
    ----------
    .. [1] Bollobás, Béla,  Janson, S. and Riordan, O.
       "The phase transition in inhomogeneous random graphs",
       *Random Structures Algorithms*, 31, 3--122, 2007.

    .. [2] Hagberg A, Lemons N (2015),
       "Fast Generation of Sparse Random Kernel Graphs".
       PLoS ONE 10(9): e0135177, 2015. doi:10.1371/journal.pone.0135177
    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    if kernel_root is None:
        import scipy as sp

        def kernel_root(y, a, r):
            def my_function(b):
                return kernel_integral(y, a, b) - r

            return sp.optimize.brentq(my_function, a, 1)

    graph = nx.empty_graph(create_using=create_using)
    graph.add_nodes_from(range(n))
    (i, j) = (1, 1)
    while i < n:
        r = -math.log(1 - seed.random())  # (1-seed.random()) in (0, 1]
        if kernel_integral(i / n, j / n, 1) <= r:
            i, j = i + 1, i + 1
        else:
            j = math.ceil(n * kernel_root(i / n, j / n, r))
            graph.add_edge(i - 1, j - 1)
    return graph
<|endoftext|>"""Functions for generating graphs based on the "duplication" method.

These graph generators start with a small initial graph then duplicate
nodes and (partially) duplicate their edges. These functions are
generally inspired by biological networks.

"""

import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import py_random_state
from networkx.utils.misc import check_create_using

__all__ = ["partial_duplication_graph", "duplication_divergence_graph"]


@py_random_state(4)
@nx._dispatchable(graphs=None, returns_graph=True)
def partial_duplication_graph(N, n, p, q, seed=None, *, create_using=None):
    """Returns a random graph using the partial duplication model.

    Parameters
    ----------
    N : int
        The total number of nodes in the final graph.

    n : int
        The number of nodes in the initial clique.

    p : float
        The probability of joining each neighbor of a node to the
        duplicate node. Must be a number in the between zero and one,
        inclusive.

    q : float
        The probability of joining the source node to the duplicate
        node. Must be a number in the between zero and one, inclusive.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Notes
    -----
    A graph of nodes is grown by creating a fully connected graph
    of size `n`. The following procedure is then repeated until
    a total of `N` nodes have been reached.

    1. A random node, *u*, is picked and a new node, *v*, is created.
    2. For each neighbor of *u* an edge from the neighbor to *v* is created
       with probability `p`.
    3. An edge from *u* to *v* is created with probability `q`.

    This algorithm appears in [1].

    This implementation allows the possibility of generating
    disconnected graphs.

    References
    ----------
    .. [1] Knudsen Michael, and Carsten Wiuf. "A Markov chain approach to
           randomly grown graphs." Journal of Applied Mathematics 2008.
           <https://doi.org/10.1155/2008/190836>

    """
    create_using = check_create_using(create_using, directed=False, multigraph=False)
    if p < 0 or p > 1 or q < 0 or q > 1:
        msg = "partial duplication graph must have 0 <= p, q <= 1."
        raise NetworkXError(msg)
    if n > N:
        raise NetworkXError("partial duplication graph must have n <= N.")

    G = nx.complete_graph(n, create_using)
    for new_node in range(n, N):
        # Pick a random vertex, u, already in the graph.
        src_node = seed.randint(0, new_node - 1)

        # Add a new vertex, v, to the graph.
        G.add_node(new_node)

        # For each neighbor of u...
        for nbr_node in list(nx.all_neighbors(G, src_node)):
            # Add the neighbor to v with probability p.
            if seed.random() < p:
                G.add_edge(new_node, nbr_node)

        # Join v and u with probability q.
        if seed.random() < q:
            G.add_edge(new_node, src_node)
    return G


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def duplication_divergence_graph(n, p, seed=None, *, create_using=None):
    """Returns an undirected graph using the duplication-divergence model.

    A graph of `n` nodes is created by duplicating the initial nodes
    and retaining edges incident to the original nodes with a retention
    probability `p`.

    Parameters
    ----------
    n : int
        The desired number of nodes in the graph.
    p : float
        The probability for retaining the edge of the replicated node.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `p` is not a valid probability.
        If `n` is less than 2.

    Notes
    -----
    This algorithm appears in [1].

    This implementation disallows the possibility of generating
    disconnected graphs.

    References
    ----------
    .. [1] I. Ispolatov, P. L. Krapivsky, A. Yuryev,
       "Duplication-divergence model of protein interaction network",
       Phys. Rev. E, 71, 061911, 2005.

    """
    if p > 1 or p < 0:
        msg = f"NetworkXError p={p} is not in [0,1]."
        raise nx.NetworkXError(msg)
    if n < 2:
        msg = "n must be greater than or equal to 2"
        raise nx.NetworkXError(msg)

    create_using = check_create_using(create_using, directed=False, multigraph=False)
    G = nx.empty_graph(create_using=create_using)

    # Initialize the graph with two connected nodes.
    G.add_edge(0, 1)
    i = 2
    while i < n:
        # Choose a random node from current graph to duplicate.
        random_node = seed.choice(list(G))
        # Make the replica.
        G.add_node(i)
        # flag indicates whether at least one edge is connected on the replica.
        flag = False
        for nbr in G.neighbors(random_node):
            if seed.random() < p:
                # Link retention step.
                G.add_edge(i, nbr)
                flag = True
        if not flag:
            # Delete replica if no edges retained.
            G.remove_node(i)
        else:
            # Successful duplication.
            i += 1
    return G
<|endoftext|># See https://github.com/networkx/networkx/pull/1474
# Copyright 2011 Reya Group <http://www.reyagroup.com>
# Copyright 2011 Alex Levenson <alex@isnotinvain.com>
# Copyright 2011 Diederik van Liere <diederik.vanliere@rotman.utoronto.ca>
"""Functions that generate the triad graphs, that is, the possible
digraphs on three nodes.

"""

import networkx as nx
from networkx.classes import DiGraph

__all__ = ["triad_graph"]

#: Dictionary mapping triad name to list of directed edges in the
#: digraph representation of that triad (with nodes 'a', 'b', and 'c').
TRIAD_EDGES = {
    "003": [],
    "012": ["ab"],
    "102": ["ab", "ba"],
    "021D": ["ba", "bc"],
    "021U": ["ab", "cb"],
    "021C": ["ab", "bc"],
    "111D": ["ac", "ca", "bc"],
    "111U": ["ac", "ca", "cb"],
    "030T": ["ab", "cb", "ac"],
    "030C": ["ba", "cb", "ac"],
    "201": ["ab", "ba", "ac", "ca"],
    "120D": ["bc", "ba", "ac", "ca"],
    "120U": ["ab", "cb", "ac", "ca"],
    "120C": ["ab", "bc", "ac", "ca"],
    "210": ["ab", "bc", "cb", "ac", "ca"],
    "300": ["ab", "ba", "bc", "cb", "ac", "ca"],
}


@nx._dispatchable(graphs=None, returns_graph=True)
def triad_graph(triad_name):
    """Returns the triad graph with the given name.

    Each string in the following tuple is a valid triad name::

        (
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

    Each triad name corresponds to one of the possible valid digraph on
    three nodes.

    Parameters
    ----------
    triad_name : string
        The name of a triad, as described above.

    Returns
    -------
    :class:`~networkx.DiGraph`
        The digraph on three nodes with the given name. The nodes of the
        graph are the single-character strings 'a', 'b', and 'c'.

    Raises
    ------
    ValueError
        If `triad_name` is not the name of a triad.

    See also
    --------
    triadic_census

    """
    if triad_name not in TRIAD_EDGES:
        raise ValueError(
            f'unknown triad name "{triad_name}"; use one of the triad names'
            " in the TRIAD_NAMES constant"
        )
    G = DiGraph()
    G.add_nodes_from("abc")
    G.add_edges_from(TRIAD_EDGES[triad_name])
    return G
<|endoftext|>"""Generators for classes of graphs used in studying social networks."""

import itertools
import math

import networkx as nx
from networkx.utils import py_random_state

__all__ = [
    "caveman_graph",
    "connected_caveman_graph",
    "relaxed_caveman_graph",
    "random_partition_graph",
    "planted_partition_graph",
    "gaussian_random_partition_graph",
    "ring_of_cliques",
    "windmill_graph",
    "stochastic_block_model",
    "LFR_benchmark_graph",
]


@nx._dispatchable(graphs=None, returns_graph=True)
def caveman_graph(l, k):
    """Returns a caveman graph of `l` cliques of size `k`.

    Parameters
    ----------
    l : int
      Number of cliques
    k : int
      Size of cliques

    Returns
    -------
    G : NetworkX Graph
      caveman graph

    Notes
    -----
    This returns an undirected graph, it can be converted to a directed
    graph using :func:`nx.to_directed`, or a multigraph using
    ``nx.MultiGraph(nx.caveman_graph(l, k))``. Only the undirected version is
    described in [1]_ and it is unclear which of the directed
    generalizations is most useful.

    Examples
    --------
    >>> G = nx.caveman_graph(3, 3)

    See also
    --------

    connected_caveman_graph

    References
    ----------
    .. [1] Watts, D. J. 'Networks, Dynamics, and the Small-World Phenomenon.'
       Amer. J. Soc. 105, 493-527, 1999.
    """
    # l disjoint cliques of size k
    G = nx.empty_graph(l * k)
    if k > 1:
        for start in range(0, l * k, k):
            edges = itertools.combinations(range(start, start + k), 2)
            G.add_edges_from(edges)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def connected_caveman_graph(l, k):
    """Returns a connected caveman graph of `l` cliques of size `k`.

    The connected caveman graph is formed by creating `n` cliques of size
    `k`, then a single edge in each clique is rewired to a node in an
    adjacent clique.

    Parameters
    ----------
    l : int
      number of cliques
    k : int
      size of cliques (k at least 2 or NetworkXError is raised)

    Returns
    -------
    G : NetworkX Graph
      connected caveman graph

    Raises
    ------
    NetworkXError
        If the size of cliques `k` is smaller than 2.

    Notes
    -----
    This returns an undirected graph, it can be converted to a directed
    graph using :func:`nx.to_directed`, or a multigraph using
    ``nx.MultiGraph(nx.caveman_graph(l, k))``. Only the undirected version is
    described in [1]_ and it is unclear which of the directed
    generalizations is most useful.

    Examples
    --------
    >>> G = nx.connected_caveman_graph(3, 3)

    References
    ----------
    .. [1] Watts, D. J. 'Networks, Dynamics, and the Small-World Phenomenon.'
       Amer. J. Soc. 105, 493-527, 1999.
    """
    if k < 2:
        raise nx.NetworkXError(
            "The size of cliques in a connected caveman graph must be at least 2."
        )

    G = nx.caveman_graph(l, k)
    for start in range(0, l * k, k):
        G.remove_edge(start, start + 1)
        G.add_edge(start, (start - 1) % (l * k))
    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def relaxed_caveman_graph(l, k, p, seed=None):
    """Returns a relaxed caveman graph.

    A relaxed caveman graph starts with `l` cliques of size `k`.  Edges are
    then randomly rewired with probability `p` to link different cliques.

    Parameters
    ----------
    l : int
      Number of groups
    k : int
      Size of cliques
    p : float
      Probability of rewiring each edge.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : NetworkX Graph
      Relaxed Caveman Graph

    Raises
    ------
    NetworkXError
     If p is not in [0,1]

    Examples
    --------
    >>> G = nx.relaxed_caveman_graph(2, 3, 0.1, seed=42)

    References
    ----------
    .. [1] Santo Fortunato, Community Detection in Graphs,
       Physics Reports Volume 486, Issues 3-5, February 2010, Pages 75-174.
       https://arxiv.org/abs/0906.0612
    """
    G = nx.caveman_graph(l, k)
    nodes = list(G)
    for u, v in G.edges():
        if seed.random() < p:  # rewire the edge
            x = seed.choice(nodes)
            if G.has_edge(u, x):
                continue
            G.remove_edge(u, v)
            G.add_edge(u, x)
    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_partition_graph(sizes, p_in, p_out, seed=None, directed=False):
    """Returns the random partition graph with a partition of sizes.

    A partition graph is a graph of communities with sizes defined by
    s in sizes. Nodes in the same group are connected with probability
    p_in and nodes of different groups are connected with probability
    p_out.

    Parameters
    ----------
    sizes : list of ints
      Sizes of groups
    p_in : float
      probability of edges with in groups
    p_out : float
      probability of edges between groups
    directed : boolean optional, default=False
      Whether to create a directed graph
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : NetworkX Graph or DiGraph
      random partition graph of size sum(gs)

    Raises
    ------
    NetworkXError
      If p_in or p_out is not in [0,1]

    Examples
    --------
    >>> G = nx.random_partition_graph([10, 10, 10], 0.25, 0.01)
    >>> len(G)
    30
    >>> partition = G.graph["partition"]
    >>> len(partition)
    3

    Notes
    -----
    This is a generalization of the planted-l-partition described in
    [1]_.  It allows for the creation of groups of any size.

    The partition is store as a graph attribute 'partition'.

    References
    ----------
    .. [1] Santo Fortunato 'Community Detection in Graphs' Physical Reports
       Volume 486, Issue 3-5 p. 75-174. https://arxiv.org/abs/0906.0612
    """
    # Use geometric method for O(n+m) complexity algorithm
    # partition = nx.community_sets(nx.get_node_attributes(G, 'affiliation'))
    if not 0.0 <= p_in <= 1.0:
        raise nx.NetworkXError("p_in must be in [0,1]")
    if not 0.0 <= p_out <= 1.0:
        raise nx.NetworkXError("p_out must be in [0,1]")

    # create connection matrix
    num_blocks = len(sizes)
    p = [[p_out for s in range(num_blocks)] for r in range(num_blocks)]
    for r in range(num_blocks):
        p[r][r] = p_in

    return stochastic_block_model(
        sizes,
        p,
        nodelist=None,
        seed=seed,
        directed=directed,
        selfloops=False,
        sparse=True,
    )


@py_random_state(4)
@nx._dispatchable(graphs=None, returns_graph=True)
def planted_partition_graph(l, k, p_in, p_out, seed=None, directed=False):
    """Returns the planted l-partition graph.

    This model partitions a graph with n=l*k vertices in
    l groups with k vertices each. Vertices of the same
    group are linked with a probability p_in, and vertices
    of different groups are linked with probability p_out.

    Parameters
    ----------
    l : int
      Number of groups
    k : int
      Number of vertices in each group
    p_in : float
      probability of connecting vertices within a group
    p_out : float
      probability of connected vertices between groups
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool,optional (default=False)
      If True return a directed graph

    Returns
    -------
    G : NetworkX Graph or DiGraph
      planted l-partition graph

    Raises
    ------
    NetworkXError
      If `p_in`, `p_out` are not in `[0, 1]`

    Examples
    --------
    >>> G = nx.planted_partition_graph(4, 3, 0.5, 0.1, seed=42)

    See Also
    --------
    random_partition_model

    References
    ----------
    .. [1] A. Condon, R.M. Karp, Algorithms for graph partitioning
        on the planted partition model,
        Random Struct. Algor. 18 (2001) 116-140.

    .. [2] Santo Fortunato 'Community Detection in Graphs' Physical Reports
       Volume 486, Issue 3-5 p. 75-174. https://arxiv.org/abs/0906.0612
    """
    return random_partition_graph([k] * l, p_in, p_out, seed=seed, directed=directed)


@py_random_state(6)
@nx._dispatchable(graphs=None, returns_graph=True)
def gaussian_random_partition_graph(n, s, v, p_in, p_out, directed=False, seed=None):
    """Generate a Gaussian random partition graph.

    A Gaussian random partition graph is created by creating k partitions
    each with a size drawn from a normal distribution with mean s and variance
    s/v. Nodes are connected within clusters with probability p_in and
    between clusters with probability p_out[1]

    Parameters
    ----------
    n : int
      Number of nodes in the graph
    s : float
      Mean cluster size
    v : float
      Shape parameter. The variance of cluster size distribution is s/v.
    p_in : float
      Probability of intra cluster connection.
    p_out : float
      Probability of inter cluster connection.
    directed : boolean, optional default=False
      Whether to create a directed graph or not
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : NetworkX Graph or DiGraph
      gaussian random partition graph

    Raises
    ------
    NetworkXError
      If s is > n
      If p_in or p_out is not in [0,1]

    Notes
    -----
    Note the number of partitions is dependent on s,v and n, and that the
    last partition may be considerably smaller, as it is sized to simply
    fill out the nodes [1]

    See Also
    --------
    random_partition_graph

    Examples
    --------
    >>> G = nx.gaussian_random_partition_graph(100, 10, 10, 0.25, 0.1)
    >>> len(G)
    100

    References
    ----------
    .. [1] Ulrik Brandes, Marco Gaertler, Dorothea Wagner,
       Experiments on Graph Clustering Algorithms,
       In the proceedings of the 11th Europ. Symp. Algorithms, 2003.
    """
    if s > n:
        raise nx.NetworkXError("s must be <= n")
    assigned = 0
    sizes = []
    while True:
        size = int(seed.gauss(s, s / v + 0.5))
        if size < 1:  # how to handle 0 or negative sizes?
            continue
        if assigned + size >= n:
            sizes.append(n - assigned)
            break
        assigned += size
        sizes.append(size)
    return random_partition_graph(sizes, p_in, p_out, seed=seed, directed=directed)


@nx._dispatchable(graphs=None, returns_graph=True)
def ring_of_cliques(num_cliques, clique_size):
    """Defines a "ring of cliques" graph.

    A ring of cliques graph is consisting of cliques, connected through single
    links. Each clique is a complete graph.

    Parameters
    ----------
    num_cliques : int
        Number of cliques
    clique_size : int
        Size of cliques

    Returns
    -------
    G : NetworkX Graph
        ring of cliques graph

    Raises
    ------
    NetworkXError
        If the number of cliques is lower than 2 or
        if the size of cliques is smaller than 2.

    Examples
    --------
    >>> G = nx.ring_of_cliques(8, 4)

    See Also
    --------
    connected_caveman_graph

    Notes
    -----
    The `connected_caveman_graph` graph removes a link from each clique to
    connect it with the next clique. Instead, the `ring_of_cliques` graph
    simply adds the link without removing any link from the cliques.
    """
    if num_cliques < 2:
        raise nx.NetworkXError("A ring of cliques must have at least two cliques")
    if clique_size < 2:
        raise nx.NetworkXError("The cliques must have at least two nodes")

    G = nx.Graph()
    for i in range(num_cliques):
        edges = itertools.combinations(
            range(i * clique_size, i * clique_size + clique_size), 2
        )
        G.add_edges_from(edges)
        G.add_edge(
            i * clique_size + 1, (i + 1) * clique_size % (num_cliques * clique_size)
        )
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def windmill_graph(n, k):
    """Generate a windmill graph.
    A windmill graph is a graph of `n` cliques each of size `k` that are all
    joined at one node.
    It can be thought of as taking a disjoint union of `n` cliques of size `k`,
    selecting one point from each, and contracting all of the selected points.
    Alternatively, one could generate `n` cliques of size `k-1` and one node
    that is connected to all other nodes in the graph.

    Parameters
    ----------
    n : int
        Number of cliques
    k : int
        Size of cliques

    Returns
    -------
    G : NetworkX Graph
        windmill graph with n cliques of size k

    Raises
    ------
    NetworkXError
        If the number of cliques is less than two
        If the size of the cliques are less than two

    Examples
    --------
    >>> G = nx.windmill_graph(4, 5)

    Notes
    -----
    The node labeled `0` will be the node connected to all other nodes.
    Note that windmill graphs are usually denoted `Wd(k,n)`, so the parameters
    are in the opposite order as the parameters of this method.
    """
    if n < 2:
        msg = "A windmill graph must have at least two cliques"
        raise nx.NetworkXError(msg)
    if k < 2:
        raise nx.NetworkXError("The cliques must have at least two nodes")

    G = nx.disjoint_union_all(
        itertools.chain(
            [nx.complete_graph(k)], (nx.complete_graph(k - 1) for _ in range(n - 1))
        )
    )
    G.add_edges_from((0, i) for i in range(k, G.number_of_nodes()))
    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def stochastic_block_model(
    sizes, p, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True
):
    """Returns a stochastic block model graph.

    This model partitions the nodes in blocks of arbitrary sizes, and places
    edges between pairs of nodes independently, with a probability that depends
    on the blocks.

    Parameters
    ----------
    sizes : list of ints
        Sizes of blocks
    p : list of list of floats
        Element (r,s) gives the density of edges going from the nodes
        of group r to nodes of group s.
        p must match the number of groups (len(sizes) == len(p)),
        and it must be symmetric if the graph is undirected.
    nodelist : list, optional
        The block tags are assigned according to the node identifiers
        in nodelist. If nodelist is None, then the ordering is the
        range [0,sum(sizes)-1].
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : boolean optional, default=False
        Whether to create a directed graph or not.
    selfloops : boolean optional, default=False
        Whether to include self-loops or not.
    sparse: boolean optional, default=True
        Use the sparse heuristic to speed up the generator.

    Returns
    -------
    g : NetworkX Graph or DiGraph
        Stochastic block model graph of size sum(sizes)

    Raises
    ------
    NetworkXError
      If probabilities are not in [0,1].
      If the probability matrix is not square (directed case).
      If the probability matrix is not symmetric (undirected case).
      If the sizes list does not match nodelist or the probability matrix.
      If nodelist contains duplicate.

    Examples
    --------
    >>> sizes = [75, 75, 300]
    >>> probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
    >>> g = nx.stochastic_block_model(sizes, probs, seed=0)
    >>> len(g)
    450
    >>> H = nx.quotient_graph(g, g.graph["partition"], relabel=True)
    >>> for v in H.nodes(data=True):
    ...     print(round(v[1]["density"], 3))
    0.245
    0.348
    0.405
    >>> for v in H.edges(data=True):
    ...     print(round(1.0 * v[2]["weight"] / (sizes[v[0]] * sizes[v[1]]), 3))
    0.051
    0.022
    0.07

    See Also
    --------
    random_partition_graph
    planted_partition_graph
    gaussian_random_partition_graph
    gnp_random_graph

    References
    ----------
    .. [1] Holland, P. W., Laskey, K. B., & Leinhardt, S.,
           "Stochastic blockmodels: First steps",
           Social networks, 5(2), 109-137, 1983.
    """
    # Check if dimensions match
    if len(sizes) != len(p):
        raise nx.NetworkXException("'sizes' and 'p' do not match.")
    # Check for probability symmetry (undirected) and shape (directed)
    for row in p:
        if len(p) != len(row):
            raise nx.NetworkXException("'p' must be a square matrix.")
    if not directed:
        p_transpose = [list(i) for i in zip(*p)]
        for i in zip(p, p_transpose):
            for j in zip(i[0], i[1]):
                if abs(j[0] - j[1]) > 1e-08:
                    raise nx.NetworkXException("'p' must be symmetric.")
    # Check for probability range
    for row in p:
        for prob in row:
            if prob < 0 or prob > 1:
                raise nx.NetworkXException("Entries of 'p' not in [0,1].")
    # Check for nodelist consistency
    if nodelist is not None:
        if len(nodelist) != sum(sizes):
            raise nx.NetworkXException("'nodelist' and 'sizes' do not match.")
        if len(nodelist) != len(set(nodelist)):
            raise nx.NetworkXException("nodelist contains duplicate.")
    else:
        nodelist = range(sum(sizes))

    # Setup the graph conditionally to the directed switch.
    block_range = range(len(sizes))
    if directed:
        g = nx.DiGraph()
        block_iter = itertools.product(block_range, block_range)
    else:
        g = nx.Graph()
        block_iter = itertools.combinations_with_replacement(block_range, 2)
    # Split nodelist in a partition (list of sets).
    size_cumsum = [sum(sizes[0:x]) for x in range(len(sizes) + 1)]
    g.graph["partition"] = [
        set(nodelist[size_cumsum[x] : size_cumsum[x + 1]])
        for x in range(len(size_cumsum) - 1)
    ]
    # Setup nodes and graph name
    for block_id, nodes in enumerate(g.graph["partition"]):
        for node in nodes:
            g.add_node(node, block=block_id)

    g.name = "stochastic_block_model"

    # Test for edge existence
    parts = g.graph["partition"]
    for i, j in block_iter:
        if i == j:
            if directed:
                if selfloops:
                    edges = itertools.product(parts[i], parts[i])
                else:
                    edges = itertools.permutations(parts[i], 2)
            else:
                edges = itertools.combinations(parts[i], 2)
                if selfloops:
                    edges = itertools.chain(edges, zip(parts[i], parts[i]))
            for e in edges:
                if seed.random() < p[i][j]:
                    g.add_edge(*e)
        else:
            edges = itertools.product(parts[i], parts[j])
        if sparse:
            if p[i][j] == 1:  # Test edges cases p_ij = 0 or 1
                for e in edges:
                    g.add_edge(*e)
            elif p[i][j] > 0:
                while True:
                    try:
                        logrand = math.log(seed.random())
                        skip = math.floor(logrand / math.log(1 - p[i][j]))
                        # consume "skip" edges
                        next(itertools.islice(edges, skip, skip), None)
                        e = next(edges)
                        g.add_edge(*e)  # __safe
                    except StopIteration:
                        break
        else:
            for e in edges:
                if seed.random() < p[i][j]:
                    g.add_edge(*e)  # __safe
    return g


def _zipf_rv_below(gamma, xmin, threshold, seed):
    """Returns a random value chosen from the bounded Zipf distribution.

    Repeatedly draws values from the Zipf distribution until the
    threshold is met, then returns that value.
    """
    result = nx.utils.zipf_rv(gamma, xmin, seed)
    while result > threshold:
        result = nx.utils.zipf_rv(gamma, xmin, seed)
    return result


