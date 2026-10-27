def _powerlaw_sequence(gamma, low, high, condition, length, max_iters, seed):
    """Returns a list of numbers obeying a constrained power law distribution.

    ``gamma`` and ``low`` are the parameters for the Zipf distribution.

    ``high`` is the maximum allowed value for values draw from the Zipf
    distribution. For more information, see :func:`_zipf_rv_below`.

    ``condition`` and ``length`` are Boolean-valued functions on
    lists. While generating the list, random values are drawn and
    appended to the list until ``length`` is satisfied by the created
    list. Once ``condition`` is satisfied, the sequence generated in
    this way is returned.

    ``max_iters`` indicates the number of times to generate a list
    satisfying ``length``. If the number of iterations exceeds this
    value, :exc:`~networkx.exception.ExceededMaxIterations` is raised.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
    for i in range(max_iters):
        seq = []
        while not length(seq):
            seq.append(_zipf_rv_below(gamma, low, high, seed))
        if condition(seq):
            return seq
    raise nx.ExceededMaxIterations("Could not create power law sequence")


def _hurwitz_zeta(x, q, tolerance):
    """The Hurwitz zeta function, or the Riemann zeta function of two arguments.

    ``x`` must be greater than one and ``q`` must be positive.

    This function repeatedly computes subsequent partial sums until
    convergence, as decided by ``tolerance``.
    """
    z = 0
    z_prev = -float("inf")
    k = 0
    while abs(z - z_prev) > tolerance:
        z_prev = z
        z += 1 / ((k + q) ** x)
        k += 1
    return z


def _generate_min_degree(gamma, average_degree, max_degree, tolerance, max_iters):
    """Returns a minimum degree from the given average degree."""
    # Defines zeta function whether or not Scipy is available
    try:
        from scipy.special import zeta
    except ImportError:

        def zeta(x, q):
            return _hurwitz_zeta(x, q, tolerance)

    min_deg_top = max_degree
    min_deg_bot = 1
    min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
    itrs = 0
    mid_avg_deg = 0
    while abs(mid_avg_deg - average_degree) > tolerance:
        if itrs > max_iters:
            raise nx.ExceededMaxIterations("Could not match average_degree")
        mid_avg_deg = 0
        for x in range(int(min_deg_mid), max_degree + 1):
            mid_avg_deg += (x ** (-gamma + 1)) / zeta(gamma, min_deg_mid)
        if mid_avg_deg > average_degree:
            min_deg_top = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        else:
            min_deg_bot = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        itrs += 1
    # return int(min_deg_mid + 0.5)
    return round(min_deg_mid)


def _generate_communities(degree_seq, community_sizes, mu, max_iters, seed):
    """Returns a list of sets, each of which represents a community.

    ``degree_seq`` is the degree sequence that must be met by the
    graph.

    ``community_sizes`` is the community size distribution that must be
    met by the generated list of sets.

    ``mu`` is a float in the interval [0, 1] indicating the fraction of
    intra-community edges incident to each node.

    ``max_iters`` is the number of times to try to add a node to a
    community. This must be greater than the length of
    ``degree_seq``, otherwise this function will always fail. If
    the number of iterations exceeds this value,
    :exc:`~networkx.exception.ExceededMaxIterations` is raised.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    The communities returned by this are sets of integers in the set {0,
    ..., *n* - 1}, where *n* is the length of ``degree_seq``.

    """
    # This assumes the nodes in the graph will be natural numbers.
    result = [set() for _ in community_sizes]
    n = len(degree_seq)
    free = list(range(n))
    for i in range(max_iters):
        v = free.pop()
        c = seed.choice(range(len(community_sizes)))
        # s = int(degree_seq[v] * (1 - mu) + 0.5)
        s = round(degree_seq[v] * (1 - mu))
        # If the community is large enough, add the node to the chosen
        # community. Otherwise, return it to the list of unaffiliated
        # nodes.
        if s < community_sizes[c]:
            result[c].add(v)
        else:
            free.append(v)
        # If the community is too big, remove a node from it.
        if len(result[c]) > community_sizes[c]:
            free.append(result[c].pop())
        if not free:
            return result
    msg = "Could not assign communities; try increasing min_community"
    raise nx.ExceededMaxIterations(msg)


@py_random_state(11)
@nx._dispatchable(graphs=None, returns_graph=True)
def LFR_benchmark_graph(
    n,
    tau1,
    tau2,
    mu,
    average_degree=None,
    min_degree=None,
    max_degree=None,
    min_community=None,
    max_community=None,
    tol=1.0e-7,
    max_iters=500,
    seed=None,
):
    r"""Returns the LFR benchmark graph.

    This algorithm proceeds as follows:

    1) Find a degree sequence with a power law distribution, and minimum
       value ``min_degree``, which has approximate average degree
       ``average_degree``. This is accomplished by either

       a) specifying ``min_degree`` and not ``average_degree``,
       b) specifying ``average_degree`` and not ``min_degree``, in which
          case a suitable minimum degree will be found.

       ``max_degree`` can also be specified, otherwise it will be set to
       ``n``. Each node *u* will have $\mu \mathrm{deg}(u)$ edges
       joining it to nodes in communities other than its own and $(1 -
       \mu) \mathrm{deg}(u)$ edges joining it to nodes in its own
       community.
    2) Generate community sizes according to a power law distribution
       with exponent ``tau2``. If ``min_community`` and
       ``max_community`` are not specified they will be selected to be
       ``min_degree`` and ``max_degree``, respectively.  Community sizes
       are generated until the sum of their sizes equals ``n``.
    3) Each node will be randomly assigned a community with the
       condition that the community is large enough for the node's
       intra-community degree, $(1 - \mu) \mathrm{deg}(u)$ as
       described in step 2. If a community grows too large, a random node
       will be selected for reassignment to a new community, until all
       nodes have been assigned a community.
    4) Each node *u* then adds $(1 - \mu) \mathrm{deg}(u)$
       intra-community edges and $\mu \mathrm{deg}(u)$ inter-community
       edges.

    Parameters
    ----------
    n : int
        Number of nodes in the created graph.

    tau1 : float
        Power law exponent for the degree distribution of the created
        graph. This value must be strictly greater than one.

    tau2 : float
        Power law exponent for the community size distribution in the
        created graph. This value must be strictly greater than one.

    mu : float
        Fraction of inter-community edges incident to each node. This
        value must be in the interval [0, 1].

    average_degree : float
        Desired average degree of nodes in the created graph. This value
        must be in the interval [0, *n*]. Exactly one of this and
        ``min_degree`` must be specified, otherwise a
        :exc:`NetworkXError` is raised.

    min_degree : int
        Minimum degree of nodes in the created graph. This value must be
        in the interval [0, *n*]. Exactly one of this and
        ``average_degree`` must be specified, otherwise a
        :exc:`NetworkXError` is raised.

    max_degree : int
        Maximum degree of nodes in the created graph. If not specified,
        this is set to ``n``, the total number of nodes in the graph.

    min_community : int
        Minimum size of communities in the graph. If not specified, this
        is set to ``min_degree``.

    max_community : int
        Maximum size of communities in the graph. If not specified, this
        is set to ``n``, the total number of nodes in the graph.

    tol : float
        Tolerance when comparing floats, specifically when comparing
        average degree values.

    max_iters : int
        Maximum number of iterations to try to create the community sizes,
        degree distribution, and community affiliations.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : NetworkX graph
        The LFR benchmark graph generated according to the specified
        parameters.

        Each node in the graph has a node attribute ``'community'`` that
        stores the community (that is, the set of nodes) that includes
        it.

    Raises
    ------
    NetworkXError
        If any of the parameters do not meet their upper and lower bounds:

        - ``tau1`` and ``tau2`` must be strictly greater than 1.
        - ``mu`` must be in [0, 1].
        - ``max_degree`` must be in {1, ..., *n*}.
        - ``min_community`` and ``max_community`` must be in {0, ...,
          *n*}.

        If not exactly one of ``average_degree`` and ``min_degree`` is
        specified.

        If ``min_degree`` is not specified and a suitable ``min_degree``
        cannot be found.

    ExceededMaxIterations
        If a valid degree sequence cannot be created within
        ``max_iters`` number of iterations.

        If a valid set of community sizes cannot be created within
        ``max_iters`` number of iterations.

        If a valid community assignment cannot be created within ``10 *
        n * max_iters`` number of iterations.

    Examples
    --------
    Basic usage::

        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> n = 250
        >>> tau1 = 3
        >>> tau2 = 1.5
        >>> mu = 0.1
        >>> G = LFR_benchmark_graph(
        ...     n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10
        ... )

    Continuing the example above, you can get the communities from the
    node attributes of the graph::

        >>> communities = {frozenset(G.nodes[v]["community"]) for v in G}

    Notes
    -----
    This algorithm differs slightly from the original way it was
    presented in [1].

    1) Rather than connecting the graph via a configuration model then
       rewiring to match the intra-community and inter-community
       degrees, we do this wiring explicitly at the end, which should be
       equivalent.
    2) The code posted on the author's website [2] calculates the random
       power law distributed variables and their average using
       continuous approximations, whereas we use the discrete
       distributions here as both degree and community size are
       discrete.

    Though the authors describe the algorithm as quite robust, testing
    during development indicates that a somewhat narrower parameter set
    is likely to successfully produce a graph. Some suggestions have
    been provided in the event of exceptions.

    References
    ----------
    .. [1] "Benchmark graphs for testing community detection algorithms",
           Andrea Lancichinetti, Santo Fortunato, and Filippo Radicchi,
           Phys. Rev. E 78, 046110 2008
    .. [2] https://www.santofortunato.net/resources

    """
    # Perform some basic parameter validation.
    if not tau1 > 1:
        raise nx.NetworkXError("tau1 must be greater than one")
    if not tau2 > 1:
        raise nx.NetworkXError("tau2 must be greater than one")
    if not 0 <= mu <= 1:
        raise nx.NetworkXError("mu must be in the interval [0, 1]")

    # Validate parameters for generating the degree sequence.
    if max_degree is None:
        max_degree = n
    elif not 0 < max_degree <= n:
        raise nx.NetworkXError("max_degree must be in the interval (0, n]")
    if not ((min_degree is None) ^ (average_degree is None)):
        raise nx.NetworkXError(
            "Must assign exactly one of min_degree and average_degree"
        )
    if min_degree is None:
        min_degree = _generate_min_degree(
            tau1, average_degree, max_degree, tol, max_iters
        )

    # Generate a degree sequence with a power law distribution.
    low, high = min_degree, max_degree

    def condition(seq):
        return sum(seq) % 2 == 0

    def length(seq):
        return len(seq) >= n

    deg_seq = _powerlaw_sequence(tau1, low, high, condition, length, max_iters, seed)

    # Validate parameters for generating the community size sequence.
    if min_community is None:
        min_community = min(deg_seq)
    if max_community is None:
        max_community = max(deg_seq)

    # Generate a community size sequence with a power law distribution.
    #
    # TODO The original code incremented the number of iterations each
    # time a new Zipf random value was drawn from the distribution. This
    # differed from the way the number of iterations was incremented in
    # `_powerlaw_degree_sequence`, so this code was changed to match
    # that one. As a result, this code is allowed many more chances to
    # generate a valid community size sequence.
    low, high = min_community, max_community

    def condition(seq):
        return sum(seq) == n

    def length(seq):
        return sum(seq) >= n

    comms = _powerlaw_sequence(tau2, low, high, condition, length, max_iters, seed)

    # Generate the communities based on the given degree sequence and
    # community sizes.
    max_iters *= 10 * n
    communities = _generate_communities(deg_seq, comms, mu, max_iters, seed)

    # Finally, generate the benchmark graph based on the given
    # communities, joining nodes according to the intra- and
    # inter-community degrees.
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for c in communities:
        for u in c:
            while G.degree(u) < round(deg_seq[u] * (1 - mu)):
                v = seed.choice(list(c))
                G.add_edge(u, v)
            while G.degree(u) < deg_seq[u]:
                v = seed.choice(range(n))
                if v not in c:
                    G.add_edge(u, v)
            G.nodes[u]["community"] = c
    return G
<|endoftext|>"""Generate graphs with given degree and triangle sequence."""

import networkx as nx
from networkx.utils import py_random_state

__all__ = ["random_clustered_graph"]


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_clustered_graph(joint_degree_sequence, create_using=None, seed=None):
    r"""Generate a random graph with the given joint independent edge degree and
    triangle degree sequence.

    This uses a configuration model-like approach to generate a random graph
    (with parallel edges and self-loops) by randomly assigning edges to match
    the given joint degree sequence.

    The joint degree sequence is a list of pairs of integers of the form
    $[(d_{1,i}, d_{1,t}), \dotsc, (d_{n,i}, d_{n,t})]$. According to this list,
    vertex $u$ is a member of $d_{u,t}$ triangles and has $d_{u, i}$ other
    edges. The number $d_{u,t}$ is the *triangle degree* of $u$ and the number
    $d_{u,i}$ is the *independent edge degree*.

    Parameters
    ----------
    joint_degree_sequence : list of integer pairs
        Each list entry corresponds to the independent edge degree and
        triangle degree of a node.
    create_using : NetworkX graph constructor, optional (default MultiGraph)
       Graph type to create. If graph instance, then cleared before populated.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : MultiGraph
        A graph with the specified degree sequence. Nodes are labeled
        starting at 0 with an index corresponding to the position in
        deg_sequence.

    Raises
    ------
    NetworkXError
        If the independent edge degree sequence sum is not even
        or the triangle degree sequence sum is not divisible by 3.

    Notes
    -----
    As described by Miller [1]_ (see also Newman [2]_ for an equivalent
    description).

    A non-graphical degree sequence (not realizable by some simple
    graph) is allowed since this function returns graphs with self
    loops and parallel edges.  An exception is raised if the
    independent degree sequence does not have an even sum or the
    triangle degree sequence sum is not divisible by 3.

    This configuration model-like construction process can lead to
    duplicate edges and loops.  You can remove the self-loops and
    parallel edges (see below) which will likely result in a graph
    that doesn't have the exact degree sequence specified.  This
    "finite-size effect" decreases as the size of the graph increases.

    References
    ----------
    .. [1] Joel C. Miller. "Percolation and epidemics in random clustered
           networks". In: Physical review. E, Statistical, nonlinear, and soft
           matter physics 80 (2 Part 1 August 2009).
    .. [2] M. E. J. Newman. "Random Graphs with Clustering".
           In: Physical Review Letters 103 (5 July 2009)

    Examples
    --------
    >>> deg = [(1, 0), (1, 0), (1, 0), (2, 0), (1, 0), (2, 1), (0, 1), (0, 1)]
    >>> G = nx.random_clustered_graph(deg)

    To remove parallel edges:

    >>> G = nx.Graph(G)

    To remove self loops:

    >>> G.remove_edges_from(nx.selfloop_edges(G))

    """
    # In Python 3, zip() returns an iterator. Make this into a list.
    joint_degree_sequence = list(joint_degree_sequence)

    N = len(joint_degree_sequence)
    G = nx.empty_graph(N, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    ilist = []
    tlist = []
    for n in G:
        degrees = joint_degree_sequence[n]
        for icount in range(degrees[0]):
            ilist.append(n)
        for tcount in range(degrees[1]):
            tlist.append(n)

    if len(ilist) % 2 != 0 or len(tlist) % 3 != 0:
        raise nx.NetworkXError("Invalid degree sequence")

    seed.shuffle(ilist)
    seed.shuffle(tlist)
    while ilist:
        G.add_edge(ilist.pop(), ilist.pop())
    while tlist:
        n1 = tlist.pop()
        n2 = tlist.pop()
        n3 = tlist.pop()
        G.add_edges_from([(n1, n2), (n1, n3), (n2, n3)])
    return G
<|endoftext|>"""
A package for generating various graphs in networkx.

"""

from networkx.generators.atlas import *
from networkx.generators.classic import *
from networkx.generators.cographs import *
from networkx.generators.community import *
from networkx.generators.degree_seq import *
from networkx.generators.directed import *
from networkx.generators.duplication import *
from networkx.generators.ego import *
from networkx.generators.expanders import *
from networkx.generators.geometric import *
from networkx.generators.harary_graph import *
from networkx.generators.internet_as_graphs import *
from networkx.generators.intersection import *
from networkx.generators.interval_graph import *
from networkx.generators.joint_degree_seq import *
from networkx.generators.lattice import *
from networkx.generators.line import *
from networkx.generators.mycielski import *
from networkx.generators.nonisomorphic_trees import *
from networkx.generators.random_clustered import *
from networkx.generators.random_graphs import *
from networkx.generators.small import *
from networkx.generators.social import *
from networkx.generators.spectral_graph_forge import *
from networkx.generators.stochastic import *
from networkx.generators.sudoku import *
from networkx.generators.time_series import *
from networkx.generators.trees import *
from networkx.generators.triads import *
<|endoftext|>"""Generates graphs with a given eigenvector structure"""

import networkx as nx
from networkx.utils import np_random_state

__all__ = ["spectral_graph_forge"]


@np_random_state(3)
@nx._dispatchable(returns_graph=True)
def spectral_graph_forge(G, alpha, transformation="identity", seed=None):
    """Returns a random simple graph with spectrum resembling that of `G`

    This algorithm, called Spectral Graph Forge (SGF), computes the
    eigenvectors of a given graph adjacency matrix, filters them and
    builds a random graph with a similar eigenstructure.
    SGF has been proved to be particularly useful for synthesizing
    realistic social networks and it can also be used to anonymize
    graph sensitive data.

    Parameters
    ----------
    G : Graph
    alpha :  float
        Ratio representing the percentage of eigenvectors of G to consider,
        values in [0,1].
    transformation : string, optional
        Represents the intended matrix linear transformation, possible values
        are 'identity' and 'modularity'
    seed : integer, random_state, or None (default)
        Indicator of numpy random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    H : Graph
        A graph with a similar eigenvector structure of the input one.

    Raises
    ------
    NetworkXError
        If transformation has a value different from 'identity' or 'modularity'

    Notes
    -----
    Spectral Graph Forge (SGF) generates a random simple graph resembling the
    global properties of the given one.
    It leverages the low-rank approximation of the associated adjacency matrix
    driven by the *alpha* precision parameter.
    SGF preserves the number of nodes of the input graph and their ordering.
    This way, nodes of output graphs resemble the properties of the input one
    and attributes can be directly mapped.

    It considers the graph adjacency matrices which can optionally be
    transformed to other symmetric real matrices (currently transformation
    options include *identity* and *modularity*).
    The *modularity* transformation, in the sense of Newman's modularity matrix
    allows the focusing on community structure related properties of the graph.

    SGF applies a low-rank approximation whose fixed rank is computed from the
    ratio *alpha* of the input graph adjacency matrix dimension.
    This step performs a filtering on the input eigenvectors similar to the low
    pass filtering common in telecommunications.

    The filtered values (after truncation) are used as input to a Bernoulli
    sampling for constructing a random adjacency matrix.

    References
    ----------
    ..  [1] L. Baldesi, C. T. Butts, A. Markopoulou, "Spectral Graph Forge:
        Graph Generation Targeting Modularity", IEEE Infocom, '18.
        https://arxiv.org/abs/1801.01715
    ..  [2] M. Newman, "Networks: an introduction", Oxford university press,
        2010

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> H = nx.spectral_graph_forge(G, 0.3)
    >>>
    """
    import numpy as np
    import scipy as sp

    available_transformations = ["identity", "modularity"]
    alpha = np.clip(alpha, 0, 1)
    A = nx.to_numpy_array(G)
    n = A.shape[1]
    level = round(n * alpha)

    if transformation not in available_transformations:
        msg = f"{transformation!r} is not a valid transformation. "
        msg += f"Transformations: {available_transformations}"
        raise nx.NetworkXError(msg)

    K = np.ones((1, n)) @ A

    B = A
    if transformation == "modularity":
        B -= K.T @ K / K.sum()

    # Compute low-rank approximation of B
    evals, evecs = np.linalg.eigh(B)
    k = np.argsort(np.abs(evals))[::-1]  # indices of evals in descending order
    evecs[:, k[np.arange(level, n)]] = 0  # set smallest eigenvectors to 0
    B = evecs @ np.diag(evals) @ evecs.T

    if transformation == "modularity":
        B += K.T @ K / K.sum()

    B = np.clip(B, 0, 1)
    np.fill_diagonal(B, 0)

    for i in range(n - 1):
        B[i, i + 1 :] = sp.stats.bernoulli.rvs(B[i, i + 1 :], random_state=seed)
        B[i + 1 :, i] = np.transpose(B[i, i + 1 :])

    H = nx.from_numpy_array(B)

    return H
<|endoftext|>r"""Generators for cographs

A cograph is a graph containing no path on four vertices.
Cographs or $P_4$-free graphs can be obtained from a single vertex
by disjoint union and complementation operations.

References
----------
.. [0] D.G. Corneil, H. Lerchs, L.Stewart Burlingham,
    "Complement reducible graphs",
    Discrete Applied Mathematics, Volume 3, Issue 3, 1981, Pages 163-174,
    ISSN 0166-218X.
"""

import networkx as nx
from networkx.utils import py_random_state

__all__ = ["random_cograph"]


@py_random_state(1)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_cograph(n, seed=None):
    r"""Returns a random cograph with $2 ^ n$ nodes.

    A cograph is a graph containing no path on four vertices.
    Cographs or $P_4$-free graphs can be obtained from a single vertex
    by disjoint union and complementation operations.

    This generator starts off from a single vertex and performs disjoint
    union and full join operations on itself.
    The decision on which operation will take place is random.

    Parameters
    ----------
    n : int
        The order of the cograph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : A random graph containing no path on four vertices.

    See Also
    --------
    full_join
    union

    References
    ----------
    .. [1] D.G. Corneil, H. Lerchs, L.Stewart Burlingham,
       "Complement reducible graphs",
       Discrete Applied Mathematics, Volume 3, Issue 3, 1981, Pages 163-174,
       ISSN 0166-218X.
    """
    R = nx.empty_graph(1)

    for i in range(n):
        RR = nx.relabel_nodes(R.copy(), lambda x: x + len(R))

        if seed.randint(0, 1) == 0:
            R = nx.full_join(R, RR)
        else:
            R = nx.disjoint_union(R, RR)

    return R
<|endoftext|>"""Functions for generating grid graphs and lattices

The :func:`grid_2d_graph`, :func:`triangular_lattice_graph`, and
:func:`hexagonal_lattice_graph` functions correspond to the three
`regular tilings of the plane`_, the square, triangular, and hexagonal
tilings, respectively. :func:`grid_graph` and :func:`hypercube_graph`
are similar for arbitrary dimensions. Useful relevant discussion can
be found about `Triangular Tiling`_, and `Square, Hex and Triangle Grids`_

.. _regular tilings of the plane: https://en.wikipedia.org/wiki/List_of_regular_polytopes_and_compounds#Euclidean_tilings
.. _Square, Hex and Triangle Grids: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/
.. _Triangular Tiling: https://en.wikipedia.org/wiki/Triangular_tiling

"""

from itertools import repeat
from math import sqrt

import networkx as nx
from networkx.classes import set_node_attributes
from networkx.exception import NetworkXError
from networkx.generators.classic import cycle_graph, empty_graph, path_graph
from networkx.relabel import relabel_nodes
from networkx.utils import flatten, nodes_or_number, pairwise

__all__ = [
    "grid_2d_graph",
    "grid_graph",
    "hypercube_graph",
    "triangular_lattice_graph",
    "hexagonal_lattice_graph",
]


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number([0, 1])
def grid_2d_graph(m, n, periodic=False, create_using=None):
    """Returns the two-dimensional grid graph.

    The grid graph has each node connected to its four nearest neighbors.

    Parameters
    ----------
    m, n : int or iterable container of nodes
        If an integer, nodes are from `range(n)`.
        If a container, elements become the coordinate of the nodes.

    periodic : bool or iterable
        If `periodic` is True, both dimensions are periodic. If False, none
        are periodic.  If `periodic` is iterable, it should yield 2 bool
        values indicating whether the 1st and 2nd axes, respectively, are
        periodic.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        The (possibly periodic) grid graph of the specified dimensions.

    """
    G = empty_graph(0, create_using)
    row_name, rows = m
    col_name, cols = n
    G.add_nodes_from((i, j) for i in rows for j in cols)
    G.add_edges_from(((i, j), (pi, j)) for pi, i in pairwise(rows) for j in cols)
    G.add_edges_from(((i, j), (i, pj)) for i in rows for pj, j in pairwise(cols))

    try:
        periodic_r, periodic_c = periodic
    except TypeError:
        periodic_r = periodic_c = periodic

    if periodic_r and len(rows) > 2:
        first = rows[0]
        last = rows[-1]
        G.add_edges_from(((first, j), (last, j)) for j in cols)
    if periodic_c and len(cols) > 2:
        first = cols[0]
        last = cols[-1]
        G.add_edges_from(((i, first), (i, last)) for i in rows)
    # both directions for directed
    if G.is_directed():
        G.add_edges_from((v, u) for u, v in G.edges())
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def grid_graph(dim, periodic=False):
    """Returns the *n*-dimensional grid graph.

    The dimension *n* is the length of the list `dim` and the size in
    each dimension is the value of the corresponding list element.

    Parameters
    ----------
    dim : list or tuple of numbers or iterables of nodes
        'dim' is a tuple or list with, for each dimension, either a number
        that is the size of that dimension or an iterable of nodes for
        that dimension. The dimension of the grid_graph is the length
        of `dim`.

    periodic : bool or iterable
        If `periodic` is True, all dimensions are periodic. If False all
        dimensions are not periodic. If `periodic` is iterable, it should
        yield `dim` bool values each of which indicates whether the
        corresponding axis is periodic.

    Returns
    -------
    NetworkX graph
        The (possibly periodic) grid graph of the specified dimensions.

    Examples
    --------
    To produce a 2 by 3 by 4 grid graph, a graph on 24 nodes:

    >>> from networkx import grid_graph
    >>> G = grid_graph(dim=(2, 3, 4))
    >>> len(G)
    24
    >>> G = grid_graph(dim=(range(7, 9), range(3, 6)))
    >>> len(G)
    6
    """
    from networkx.algorithms.operators.product import cartesian_product

    if not dim:
        return empty_graph(0)

    try:
        func = (cycle_graph if p else path_graph for p in periodic)
    except TypeError:
        func = repeat(cycle_graph if periodic else path_graph)

    G = next(func)(dim[0])
    for current_dim in dim[1:]:
        Gnew = next(func)(current_dim)
        G = cartesian_product(Gnew, G)
    # graph G is done but has labels of the form (1, (2, (3, 1))) so relabel
    H = relabel_nodes(G, flatten)
    return H


@nx._dispatchable(graphs=None, returns_graph=True)
def hypercube_graph(n):
    """Returns the *n*-dimensional hypercube graph.

    The nodes are the integers between 0 and ``2 ** n - 1``, inclusive.

    For more information on the hypercube graph, see the Wikipedia
    article `Hypercube graph`_.

    .. _Hypercube graph: https://en.wikipedia.org/wiki/Hypercube_graph

    Parameters
    ----------
    n : int
        The dimension of the hypercube.
        The number of nodes in the graph will be ``2 ** n``.

    Returns
    -------
    NetworkX graph
        The hypercube graph of dimension *n*.
    """
    dim = n * [2]
    G = grid_graph(dim)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def triangular_lattice_graph(
    m, n, periodic=False, with_positions=True, create_using=None
):
    r"""Returns the $m$ by $n$ triangular lattice graph.

    The `triangular lattice graph`_ is a two-dimensional `grid graph`_ in
    which each square unit has a diagonal edge (each grid unit has a chord).

    The returned graph has $m$ rows and $n$ columns of triangles. Rows and
    columns include both triangles pointing up and down. Rows form a strip
    of constant height. Columns form a series of diamond shapes, staggered
    with the columns on either side. Another way to state the size is that
    the nodes form a grid of `m+1` rows and `(n + 1) // 2` columns.
    The odd row nodes are shifted horizontally relative to the even rows.

    Directed graph types have edges pointed up or right.

    Positions of nodes are computed by default or `with_positions is True`.
    The position of each node (embedded in a euclidean plane) is stored in
    the graph using equilateral triangles with sidelength 1.
    The height between rows of nodes is thus $\sqrt(3)/2$.
    Nodes lie in the first quadrant with the node $(0, 0)$ at the origin.

    .. _triangular lattice graph: http://mathworld.wolfram.com/TriangularGrid.html
    .. _grid graph: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/
    .. _Triangular Tiling: https://en.wikipedia.org/wiki/Triangular_tiling

    Parameters
    ----------
    m : int
        The number of rows in the lattice.

    n : int
        The number of columns in the lattice.

    periodic : bool (default: False)
        If True, join the boundary vertices of the grid using periodic
        boundary conditions. The join between boundaries is the final row
        and column of triangles. This means there is one row and one column
        fewer nodes for the periodic lattice. Periodic lattices require
        `m >= 3`, `n >= 5` and are allowed but misaligned if `m` or `n` are odd

    with_positions : bool (default: True)
        Store the coordinates of each node in the graph node attribute 'pos'.
        The coordinates provide a lattice with equilateral triangles.
        Periodic positions shift the nodes vertically in a nonlinear way so
        the edges don't overlap so much.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        The *m* by *n* triangular lattice graph.
    """
    H = empty_graph(0, create_using)
    if n == 0 or m == 0:
        return H
    if periodic:
        if n < 5 or m < 3:
            msg = f"m > 2 and n > 4 required for periodic. m={m}, n={n}"
            raise NetworkXError(msg)

    N = (n + 1) // 2  # number of nodes in row
    rows = range(m + 1)
    cols = range(N + 1)
    # Make grid
    H.add_edges_from(((i, j), (i + 1, j)) for j in rows for i in cols[:N])
    H.add_edges_from(((i, j), (i, j + 1)) for j in rows[:m] for i in cols)
    # add diagonals
    H.add_edges_from(((i, j), (i + 1, j + 1)) for j in rows[1:m:2] for i in cols[:N])
    H.add_edges_from(((i + 1, j), (i, j + 1)) for j in rows[:m:2] for i in cols[:N])
    # identify boundary nodes if periodic
    from networkx.algorithms.minors import contracted_nodes

    if periodic is True:
        for i in cols:
            H = contracted_nodes(H, (i, 0), (i, m))
        for j in rows[:m]:
            H = contracted_nodes(H, (0, j), (N, j))
    elif n % 2:
        # remove extra nodes
        H.remove_nodes_from((N, j) for j in rows[1::2])

    # Add position node attributes
    if with_positions:
        ii = (i for i in cols for j in rows)
        jj = (j for i in cols for j in rows)
        xx = (0.5 * (j % 2) + i for i in cols for j in rows)
        h = sqrt(3) / 2
        if periodic:
            yy = (h * j + 0.01 * i * i for i in cols for j in rows)
        else:
            yy = (h * j for i in cols for j in rows)
        pos = {(i, j): (x, y) for i, j, x, y in zip(ii, jj, xx, yy) if (i, j) in H}
        set_node_attributes(H, pos, "pos")
    return H


@nx._dispatchable(graphs=None, returns_graph=True)
def hexagonal_lattice_graph(
    m, n, periodic=False, with_positions=True, create_using=None
):
    """Returns an `m` by `n` hexagonal lattice graph.

    The *hexagonal lattice graph* is a graph whose nodes and edges are
    the `hexagonal tiling`_ of the plane.

    The returned graph will have `m` rows and `n` columns of hexagons.
    `Odd numbered columns`_ are shifted up relative to even numbered columns.

    Positions of nodes are computed by default or `with_positions is True`.
    Node positions creating the standard embedding in the plane
    with sidelength 1 and are stored in the node attribute 'pos'.
    `pos = nx.get_node_attributes(G, 'pos')` creates a dict ready for drawing.

    .. _hexagonal tiling: https://en.wikipedia.org/wiki/Hexagonal_tiling
    .. _Odd numbered columns: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/

    Parameters
    ----------
    m : int
        The number of rows of hexagons in the lattice.

    n : int
        The number of columns of hexagons in the lattice.

    periodic : bool
        Whether to make a periodic grid by joining the boundary vertices.
        For this to work `n` must be even and both `n > 1` and `m > 1`.
        The periodic connections create another row and column of hexagons
        so these graphs have fewer nodes as boundary nodes are identified.

    with_positions : bool (default: True)
        Store the coordinates of each node in the graph node attribute 'pos'.
        The coordinates provide a lattice with vertical columns of hexagons
        offset to interleave and cover the plane.
        Periodic positions shift the nodes vertically in a nonlinear way so
        the edges don't overlap so much.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        If graph is directed, edges will point up or right.

    Returns
    -------
    NetworkX graph
        The *m* by *n* hexagonal lattice graph.
    """
    G = empty_graph(0, create_using)
    if m == 0 or n == 0:
        return G
    if periodic and (n % 2 == 1 or m < 2 or n < 2):
        msg = "periodic hexagonal lattice needs m > 1, n > 1 and even n"
        raise NetworkXError(msg)

    M = 2 * m  # twice as many nodes as hexagons vertically
    rows = range(M + 2)
    cols = range(n + 1)
    # make lattice
    col_edges = (((i, j), (i, j + 1)) for i in cols for j in rows[: M + 1])
    row_edges = (((i, j), (i + 1, j)) for i in cols[:n] for j in rows if i % 2 == j % 2)
    G.add_edges_from(col_edges)
    G.add_edges_from(row_edges)
    # Remove corner nodes with one edge
    G.remove_node((0, M + 1))
    G.remove_node((n, (M + 1) * (n % 2)))

    # identify boundary nodes if periodic
    from networkx.algorithms.minors import contracted_nodes

    if periodic:
        for i in cols[:n]:
            G = contracted_nodes(G, (i, 0), (i, M))
        for i in cols[1:]:
            G = contracted_nodes(G, (i, 1), (i, M + 1))
        for j in rows[1:M]:
            G = contracted_nodes(G, (0, j), (n, j))
        G.remove_node((n, M))

    # calc position in embedded space
    ii = (i for i in cols for j in rows)
    jj = (j for i in cols for j in rows)
    xx = (0.5 + i + i // 2 + (j % 2) * ((i % 2) - 0.5) for i in cols for j in rows)
    h = sqrt(3) / 2
    if periodic:
        yy = (h * j + 0.01 * i * i for i in cols for j in rows)
    else:
        yy = (h * j for i in cols for j in rows)
    # exclude nodes not in G
    pos = {(i, j): (x, y) for i, j, x, y in zip(ii, jj, xx, yy) if (i, j) in G}
    set_node_attributes(G, pos, "pos")
    return G
<|endoftext|>"""Generator for Sudoku graphs

This module gives a generator for n-Sudoku graphs. It can be used to develop
algorithms for solving or generating Sudoku puzzles.

A completed Sudoku grid is a 9x9 array of integers between 1 and 9, with no
number appearing twice in the same row, column, or 3x3 box.

+---------+---------+---------+
| | 8 6 4 | | 3 7 1 | | 2 5 9 |
| | 3 2 5 | | 8 4 9 | | 7 6 1 |
| | 9 7 1 | | 2 6 5 | | 8 4 3 |
+---------+---------+---------+
| | 4 3 6 | | 1 9 2 | | 5 8 7 |
| | 1 9 8 | | 6 5 7 | | 4 3 2 |
| | 2 5 7 | | 4 8 3 | | 9 1 6 |
+---------+---------+---------+
| | 6 8 9 | | 7 3 4 | | 1 2 5 |
| | 7 1 3 | | 5 2 8 | | 6 9 4 |
| | 5 4 2 | | 9 1 6 | | 3 7 8 |
+---------+---------+---------+


The Sudoku graph is an undirected graph with 81 vertices, corresponding to
the cells of a Sudoku grid. It is a regular graph of degree 20. Two distinct
vertices are adjacent if and only if the corresponding cells belong to the
same row, column, or box. A completed Sudoku grid corresponds to a vertex
coloring of the Sudoku graph with nine colors.

More generally, the n-Sudoku graph is a graph with n^4 vertices, corresponding
to the cells of an n^2 by n^2 grid. Two distinct vertices are adjacent if and
only if they belong to the same row, column, or n by n box.

References
----------
.. [1] Herzberg, A. M., & Murty, M. R. (2007). Sudoku squares and chromatic
    polynomials. Notices of the AMS, 54(6), 708-717.
.. [2] Sander, Torsten (2009), "Sudoku graphs are integral",
    Electronic Journal of Combinatorics, 16 (1): Note 25, 7pp, MR 2529816
.. [3] Wikipedia contributors. "Glossary of Sudoku." Wikipedia, The Free
    Encyclopedia, 3 Dec. 2019. Web. 22 Dec. 2019.
"""

import networkx as nx
from networkx.exception import NetworkXError

__all__ = ["sudoku_graph"]


@nx._dispatchable(graphs=None, returns_graph=True)
def sudoku_graph(n=3):
    """Returns the n-Sudoku graph. The default value of n is 3.

    The n-Sudoku graph is a graph with n^4 vertices, corresponding to the
    cells of an n^2 by n^2 grid. Two distinct vertices are adjacent if and
    only if they belong to the same row, column, or n-by-n box.

    Parameters
    ----------
    n: integer
       The order of the Sudoku graph, equal to the square root of the
       number of rows. The default is 3.

    Returns
    -------
    NetworkX graph
        The n-Sudoku graph Sud(n).

    Examples
    --------
    >>> G = nx.sudoku_graph()
    >>> G.number_of_nodes()
    81
    >>> G.number_of_edges()
    810
    >>> sorted(G.neighbors(42))
    [6, 15, 24, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 51, 52, 53, 60, 69, 78]
    >>> G = nx.sudoku_graph(2)
    >>> G.number_of_nodes()
    16
    >>> G.number_of_edges()
    56

    References
    ----------
    .. [1] Herzberg, A. M., & Murty, M. R. (2007). Sudoku squares and chromatic
       polynomials. Notices of the AMS, 54(6), 708-717.
    .. [2] Sander, Torsten (2009), "Sudoku graphs are integral",
       Electronic Journal of Combinatorics, 16 (1): Note 25, 7pp, MR 2529816
    .. [3] Wikipedia contributors. "Glossary of Sudoku." Wikipedia, The Free
       Encyclopedia, 3 Dec. 2019. Web. 22 Dec. 2019.
    """

    if n < 0:
        raise NetworkXError("The order must be greater than or equal to zero.")

    n2 = n * n
    n3 = n2 * n
    n4 = n3 * n

    # Construct an empty graph with n^4 nodes
    G = nx.empty_graph(n4)

    # A Sudoku graph of order 0 or 1 has no edges
    if n < 2:
        return G

    # Add edges for cells in the same row
    for row_no in range(n2):
        row_start = row_no * n2
        for j in range(1, n2):
            for i in range(j):
                G.add_edge(row_start + i, row_start + j)

    # Add edges for cells in the same column
    for col_no in range(n2):
        for j in range(col_no, n4, n2):
            for i in range(col_no, j, n2):
                G.add_edge(i, j)

    # Add edges for cells in the same box
    for band_no in range(n):
        for stack_no in range(n):
            box_start = n3 * band_no + n * stack_no
            for j in range(1, n2):
                for i in range(j):
                    u = box_start + (i % n) + n2 * (i // n)
                    v = box_start + (j % n) + n2 * (j // n)
                    G.add_edge(u, v)

    return G
<|endoftext|>"""Generators for Harary graphs

This module gives two generators for the Harary graph, which was
introduced by the famous mathematician Frank Harary in his 1962 work [H]_.
The first generator gives the Harary graph that maximizes the node
connectivity with given number of nodes and given number of edges.
The second generator gives the Harary graph that minimizes
the number of edges in the graph with given node connectivity and
number of nodes.

References
----------
.. [H] Harary, F. "The Maximum Connectivity of a Graph."
       Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.

"""

import networkx as nx
from networkx.exception import NetworkXError

__all__ = ["hnm_harary_graph", "hkn_harary_graph"]


@nx._dispatchable(graphs=None, returns_graph=True)
def hnm_harary_graph(n, m, create_using=None):
    """Returns the Harary graph with given numbers of nodes and edges.

    The Harary graph $H_{n,m}$ is the graph that maximizes node connectivity
    with $n$ nodes and $m$ edges.

    This maximum node connectivity is known to be floor($2m/n$). [1]_

    Parameters
    ----------
    n: integer
       The number of nodes the generated graph is to contain

    m: integer
       The number of edges the generated graph is to contain

    create_using : NetworkX graph constructor, optional Graph type
     to create (default=nx.Graph). If graph instance, then cleared
     before populated.

    Returns
    -------
    NetworkX graph
        The Harary graph $H_{n,m}$.

    See Also
    --------
    hkn_harary_graph

    Notes
    -----
    This algorithm runs in $O(m)$ time.
    It is implemented by following the Reference [2]_.

    References
    ----------
    .. [1] F. T. Boesch, A. Satyanarayana, and C. L. Suffel,
       "A Survey of Some Network Reliability Analysis and Synthesis Results,"
       Networks, pp. 99-107, 2009.

    .. [2] Harary, F. "The Maximum Connectivity of a Graph."
       Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.
    """

    if n < 1:
        raise NetworkXError("The number of nodes must be >= 1!")
    if m < n - 1:
        raise NetworkXError("The number of edges must be >= n - 1 !")
    if m > n * (n - 1) // 2:
        raise NetworkXError("The number of edges must be <= n(n-1)/2")

    # Construct an empty graph with n nodes first
    H = nx.empty_graph(n, create_using)
    # Get the floor of average node degree
    d = 2 * m // n

    # Test the parity of n and d
    if (n % 2 == 0) or (d % 2 == 0):
        # Start with a regular graph of d degrees
        offset = d // 2
        for i in range(n):
            for j in range(1, offset + 1):
                H.add_edge(i, (i - j) % n)
                H.add_edge(i, (i + j) % n)
        if d & 1:
            # in case d is odd; n must be even in this case
            half = n // 2
            for i in range(half):
                # add edges diagonally
                H.add_edge(i, i + half)
        # Get the remainder of 2*m modulo n
        r = 2 * m % n
        if r > 0:
            # add remaining edges at offset+1
            for i in range(r // 2):
                H.add_edge(i, i + offset + 1)
    else:
        # Start with a regular graph of (d - 1) degrees
        offset = (d - 1) // 2
        for i in range(n):
            for j in range(1, offset + 1):
                H.add_edge(i, (i - j) % n)
                H.add_edge(i, (i + j) % n)
        half = n // 2
        for i in range(m - n * offset):
            # add the remaining m - n*offset edges between i and i+half
            H.add_edge(i, (i + half) % n)

    return H


@nx._dispatchable(graphs=None, returns_graph=True)
def hkn_harary_graph(k, n, create_using=None):
    """Returns the Harary graph with given node connectivity and node number.

    The Harary graph $H_{k,n}$ is the graph that minimizes the number of
    edges needed with given node connectivity $k$ and node number $n$.

    This smallest number of edges is known to be ceil($kn/2$) [1]_.

    Parameters
    ----------
    k: integer
       The node connectivity of the generated graph

    n: integer
       The number of nodes the generated graph is to contain

    create_using : NetworkX graph constructor, optional Graph type
     to create (default=nx.Graph). If graph instance, then cleared
     before populated.

    Returns
    -------
    NetworkX graph
        The Harary graph $H_{k,n}$.

    See Also
    --------
    hnm_harary_graph

    Notes
    -----
    This algorithm runs in $O(kn)$ time.
    It is implemented by following the Reference [2]_.

    References
    ----------
    .. [1] Weisstein, Eric W. "Harary Graph." From MathWorld--A Wolfram Web
     Resource. http://mathworld.wolfram.com/HararyGraph.html.

    .. [2] Harary, F. "The Maximum Connectivity of a Graph."
      Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.
    """

    if k < 1:
        raise NetworkXError("The node connectivity must be >= 1!")
    if n < k + 1:
        raise NetworkXError("The number of nodes must be >= k+1 !")

    # in case of connectivity 1, simply return the path graph
    if k == 1:
        H = nx.path_graph(n, create_using)
        return H

    # Construct an empty graph with n nodes first
    H = nx.empty_graph(n, create_using)

    # Test the parity of k and n
    if (k % 2 == 0) or (n % 2 == 0):
        # Construct a regular graph with k degrees
        offset = k // 2
        for i in range(n):
            for j in range(1, offset + 1):
                H.add_edge(i, (i - j) % n)
                H.add_edge(i, (i + j) % n)
        if k & 1:
            # odd degree; n must be even in this case
            half = n // 2
            for i in range(half):
                # add edges diagonally
                H.add_edge(i, i + half)
    else:
        # Construct a regular graph with (k - 1) degrees
        offset = (k - 1) // 2
        for i in range(n):
            for j in range(1, offset + 1):
                H.add_edge(i, (i - j) % n)
                H.add_edge(i, (i + j) % n)
        half = n // 2
        for i in range(half + 1):
            # add half+1 edges between i and i+half
            H.add_edge(i, (i + half) % n)

    return H
<|endoftext|>"""Generate graphs with a given joint degree and directed joint degree"""

import networkx as nx
from networkx.utils import py_random_state

__all__ = [
    "is_valid_joint_degree",
    "is_valid_directed_joint_degree",
    "joint_degree_graph",
    "directed_joint_degree_graph",
]


@nx._dispatchable(graphs=None)
def is_valid_joint_degree(joint_degrees):
    """Checks whether the given joint degree dictionary is realizable.

    A *joint degree dictionary* is a dictionary of dictionaries, in
    which entry ``joint_degrees[k][l]`` is an integer representing the
    number of edges joining nodes of degree *k* with nodes of degree
    *l*. Such a dictionary is realizable as a simple graph if and only
    if the following conditions are satisfied.

    - each entry must be an integer,
    - the total number of nodes of degree *k*, computed by
      ``sum(joint_degrees[k].values()) / k``, must be an integer,
    - the total number of edges joining nodes of degree *k* with
      nodes of degree *l* cannot exceed the total number of possible edges,
    - each diagonal entry ``joint_degrees[k][k]`` must be even (this is
      a convention assumed by the :func:`joint_degree_graph` function).


    Parameters
    ----------
    joint_degrees :  dictionary of dictionary of integers
        A joint degree dictionary in which entry ``joint_degrees[k][l]``
        is the number of edges joining nodes of degree *k* with nodes of
        degree *l*.

    Returns
    -------
    bool
        Whether the given joint degree dictionary is realizable as a
        simple graph.

    References
    ----------
    .. [1] M. Gjoka, M. Kurant, A. Markopoulou, "2.5K Graphs: from Sampling
       to Generation", IEEE Infocom, 2013.
    .. [2] I. Stanton, A. Pinar, "Constructing and sampling graphs with a
       prescribed joint degree distribution", Journal of Experimental
       Algorithmics, 2012.
    """

    degree_count = {}
    for k in joint_degrees:
        if k > 0:
            k_size = sum(joint_degrees[k].values()) / k
            if not k_size.is_integer():
                return False
            degree_count[k] = k_size

    for k in joint_degrees:
        for l in joint_degrees[k]:
            if not float(joint_degrees[k][l]).is_integer():
                return False

            if (k != l) and (joint_degrees[k][l] > degree_count[k] * degree_count[l]):
                return False
            elif k == l:
                if joint_degrees[k][k] > degree_count[k] * (degree_count[k] - 1):
                    return False
                if joint_degrees[k][k] % 2 != 0:
                    return False

    # if all above conditions have been satisfied then the input
    # joint degree is realizable as a simple graph.
    return True


def _neighbor_switch(G, w, unsat, h_node_residual, avoid_node_id=None):
    """Releases one free stub for ``w``, while preserving joint degree in G.

    Parameters
    ----------
    G : NetworkX graph
        Graph in which the neighbor switch will take place.
    w : integer
        Node id for which we will execute this neighbor switch.
    unsat : set of integers
        Set of unsaturated node ids that have the same degree as w.
    h_node_residual: dictionary of integers
        Keeps track of the remaining stubs  for a given node.
    avoid_node_id: integer
        Node id to avoid when selecting w_prime.

    Notes
    -----
    First, it selects *w_prime*, an  unsaturated node that has the same degree
    as ``w``. Second, it selects *switch_node*, a neighbor node of ``w`` that
    is not  connected to *w_prime*. Then it executes an edge swap i.e. removes
    (``w``,*switch_node*) and adds (*w_prime*,*switch_node*). Gjoka et. al. [1]
    prove that such an edge swap is always possible.

    References
    ----------
    .. [1] M. Gjoka, B. Tillman, A. Markopoulou, "Construction of Simple
       Graphs with a Target Joint Degree Matrix and Beyond", IEEE Infocom, '15
    """

    if (avoid_node_id is None) or (h_node_residual[avoid_node_id] > 1):
        # select unsaturated node w_prime that has the same degree as w
        w_prime = next(iter(unsat))
    else:
        # assume that the node pair (v,w) has been selected for connection. if
        # - neighbor_switch is called for node w,
        # - nodes v and w have the same degree,
        # - node v=avoid_node_id has only one stub left,
        # then prevent v=avoid_node_id from being selected as w_prime.

        iter_var = iter(unsat)
        while True:
            w_prime = next(iter_var)
            if w_prime != avoid_node_id:
                break

    # select switch_node, a neighbor of w, that is not connected to w_prime
    w_prime_neighbs = G[w_prime]  # slightly faster declaring this variable
    for v in G[w]:
        if (v not in w_prime_neighbs) and (v != w_prime):
            switch_node = v
            break

    # remove edge (w,switch_node), add edge (w_prime,switch_node) and update
    # data structures
    G.remove_edge(w, switch_node)
    G.add_edge(w_prime, switch_node)
    h_node_residual[w] += 1
    h_node_residual[w_prime] -= 1
    if h_node_residual[w_prime] == 0:
        unsat.remove(w_prime)


@py_random_state(1)
@nx._dispatchable(graphs=None, returns_graph=True)
def joint_degree_graph(joint_degrees, seed=None):
    """Generates a random simple graph with the given joint degree dictionary.

    Parameters
    ----------
    joint_degrees :  dictionary of dictionary of integers
        A joint degree dictionary in which entry ``joint_degrees[k][l]`` is the
        number of edges joining nodes of degree *k* with nodes of degree *l*.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph
        A graph with the specified joint degree dictionary.

    Raises
    ------
    NetworkXError
        If *joint_degrees* dictionary is not realizable.

    Notes
    -----
    In each iteration of the "while loop" the algorithm picks two disconnected
    nodes *v* and *w*, of degree *k* and *l* correspondingly,  for which
    ``joint_degrees[k][l]`` has not reached its target yet. It then adds
    edge (*v*, *w*) and increases the number of edges in graph G by one.

    The intelligence of the algorithm lies in the fact that  it is always
    possible to add an edge between such disconnected nodes *v* and *w*,
    even if one or both nodes do not have free stubs. That is made possible by
    executing a "neighbor switch", an edge rewiring move that releases
    a free stub while keeping the joint degree of G the same.

    The algorithm continues for E (number of edges) iterations of
    the "while loop", at the which point all entries of the given
    ``joint_degrees[k][l]`` have reached their target values and the
    construction is complete.

    References
    ----------
    ..  [1] M. Gjoka, B. Tillman, A. Markopoulou, "Construction of Simple
        Graphs with a Target Joint Degree Matrix and Beyond", IEEE Infocom, '15

    Examples
    --------
    >>> joint_degrees = {
    ...     1: {4: 1},
    ...     2: {2: 2, 3: 2, 4: 2},
    ...     3: {2: 2, 4: 1},
    ...     4: {1: 1, 2: 2, 3: 1},
    ... }
    >>> G = nx.joint_degree_graph(joint_degrees)
    >>>
    """

    if not is_valid_joint_degree(joint_degrees):
        msg = "Input joint degree dict not realizable as a simple graph"
        raise nx.NetworkXError(msg)

    # compute degree count from joint_degrees
    degree_count = {k: sum(l.values()) // k for k, l in joint_degrees.items() if k > 0}

    # start with empty N-node graph
    N = sum(degree_count.values())
    G = nx.empty_graph(N)

    # for a given degree group, keep the list of all node ids
    h_degree_nodelist = {}

    # for a given node, keep track of the remaining stubs
    h_node_residual = {}

    # populate h_degree_nodelist and h_node_residual
    nodeid = 0
    for degree, num_nodes in degree_count.items():
        h_degree_nodelist[degree] = range(nodeid, nodeid + num_nodes)
        for v in h_degree_nodelist[degree]:
            h_node_residual[v] = degree
        nodeid += int(num_nodes)

    # iterate over every degree pair (k,l) and add the number of edges given
    # for each pair
    for k in joint_degrees:
        for l in joint_degrees[k]:
            # n_edges_add is the number of edges to add for the
            # degree pair (k,l)
            n_edges_add = joint_degrees[k][l]

            if (n_edges_add > 0) and (k >= l):
                # number of nodes with degree k and l
                k_size = degree_count[k]
                l_size = degree_count[l]

                # k_nodes and l_nodes consist of all nodes of degree k and l
                k_nodes = h_degree_nodelist[k]
                l_nodes = h_degree_nodelist[l]

                # k_unsat and l_unsat consist of nodes of degree k and l that
                # are unsaturated (nodes that have at least 1 available stub)
                k_unsat = {v for v in k_nodes if h_node_residual[v] > 0}

                if k != l:
                    l_unsat = {w for w in l_nodes if h_node_residual[w] > 0}
                else:
                    l_unsat = k_unsat
                    n_edges_add = joint_degrees[k][l] // 2

                while n_edges_add > 0:
                    # randomly pick nodes v and w that have degrees k and l
                    v = k_nodes[seed.randrange(k_size)]
                    w = l_nodes[seed.randrange(l_size)]

                    # if nodes v and w are disconnected then attempt to connect
                    if not G.has_edge(v, w) and (v != w):
                        # if node v has no free stubs then do neighbor switch
                        if h_node_residual[v] == 0:
                            _neighbor_switch(G, v, k_unsat, h_node_residual)

                        # if node w has no free stubs then do neighbor switch
                        if h_node_residual[w] == 0:
                            if k != l:
                                _neighbor_switch(G, w, l_unsat, h_node_residual)
                            else:
                                _neighbor_switch(
                                    G, w, l_unsat, h_node_residual, avoid_node_id=v
                                )

                        # add edge (v, w) and update data structures
                        G.add_edge(v, w)
                        h_node_residual[v] -= 1
                        h_node_residual[w] -= 1
                        n_edges_add -= 1

                        if h_node_residual[v] == 0:
                            k_unsat.discard(v)
                        if h_node_residual[w] == 0:
                            l_unsat.discard(w)
    return G


@nx._dispatchable(graphs=None)
def is_valid_directed_joint_degree(in_degrees, out_degrees, nkk):
    """Checks whether the given directed joint degree input is realizable

    Parameters
    ----------
    in_degrees :  list of integers
        in degree sequence contains the in degrees of nodes.
    out_degrees : list of integers
        out degree sequence contains the out degrees of nodes.
    nkk  :  dictionary of dictionary of integers
        directed joint degree dictionary. for nodes of out degree k (first
        level of dict) and nodes of in degree l (second level of dict)
        describes the number of edges.

    Returns
    -------
    boolean
        returns true if given input is realizable, else returns false.

    Notes
    -----
    Here is the list of conditions that the inputs (in/out degree sequences,
    nkk) need to satisfy for simple directed graph realizability:

    - Condition 0: in_degrees and out_degrees have the same length
    - Condition 1: nkk[k][l]  is integer for all k,l
    - Condition 2: sum(nkk[k])/k = number of nodes with partition id k, is an
                   integer and matching degree sequence
    - Condition 3: number of edges and non-chords between k and l cannot exceed
                   maximum possible number of edges


    References
    ----------
    [1] B. Tillman, A. Markopoulou, C. T. Butts & M. Gjoka,
        "Construction of Directed 2K Graphs". In Proc. of KDD 2017.
    """
    V = {}  # number of nodes with in/out degree.
    forbidden = {}
    if len(in_degrees) != len(out_degrees):
        return False

    for idx in range(len(in_degrees)):
        i = in_degrees[idx]
        o = out_degrees[idx]
        V[(i, 0)] = V.get((i, 0), 0) + 1
        V[(o, 1)] = V.get((o, 1), 0) + 1

        forbidden[(o, i)] = forbidden.get((o, i), 0) + 1

    S = {}  # number of edges going from in/out degree nodes.
    for k in nkk:
        for l in nkk[k]:
            val = nkk[k][l]
            if not float(val).is_integer():  # condition 1
                return False

            if val > 0:
                S[(k, 1)] = S.get((k, 1), 0) + val
                S[(l, 0)] = S.get((l, 0), 0) + val
                # condition 3
                if val + forbidden.get((k, l), 0) > V[(k, 1)] * V[(l, 0)]:
                    return False

    return all(S[s] / s[0] == V[s] for s in S)


def _directed_neighbor_switch(
    G, w, unsat, h_node_residual_out, chords, h_partition_in, partition
):
    """Releases one free stub for node w, while preserving joint degree in G.

    Parameters
    ----------
    G : networkx directed graph
        graph within which the edge swap will take place.
    w : integer
        node id for which we need to perform a neighbor switch.
    unsat: set of integers
        set of node ids that have the same degree as w and are unsaturated.
    h_node_residual_out: dict of integers
        for a given node, keeps track of the remaining stubs to be added.
    chords: set of tuples
        keeps track of available positions to add edges.
    h_partition_in: dict of integers
        for a given node, keeps track of its partition id (in degree).
    partition: integer
        partition id to check if chords have to be updated.

    Notes
    -----
    First, it selects node w_prime that (1) has the same degree as w and
    (2) is unsaturated. Then, it selects node v, a neighbor of w, that is
    not connected to w_prime and does an edge swap i.e. removes (w,v) and
    adds (w_prime,v). If neighbor switch is not possible for w using
    w_prime and v, then return w_prime; in [1] it's proven that
    such unsaturated nodes can be used.

    References
    ----------
    [1] B. Tillman, A. Markopoulou, C. T. Butts & M. Gjoka,
        "Construction of Directed 2K Graphs". In Proc. of KDD 2017.
    """
    w_prime = unsat.pop()
    unsat.add(w_prime)
    # select node t, a neighbor of w, that is not connected to w_prime
    w_neighbs = list(G.successors(w))
    # slightly faster declaring this variable
    w_prime_neighbs = list(G.successors(w_prime))

    for v in w_neighbs:
        if (v not in w_prime_neighbs) and w_prime != v:
            # removes (w,v), add (w_prime,v)  and update data structures
            G.remove_edge(w, v)
            G.add_edge(w_prime, v)

            if h_partition_in[v] == partition:
                chords.add((w, v))
                chords.discard((w_prime, v))

            h_node_residual_out[w] += 1
            h_node_residual_out[w_prime] -= 1
            if h_node_residual_out[w_prime] == 0:
                unsat.remove(w_prime)
            return None

    # If neighbor switch didn't work, use unsaturated node
    return w_prime


def _directed_neighbor_switch_rev(
    G, w, unsat, h_node_residual_in, chords, h_partition_out, partition
):
    """The reverse of directed_neighbor_switch.

    Parameters
    ----------
    G : networkx directed graph
        graph within which the edge swap will take place.
    w : integer
        node id for which we need to perform a neighbor switch.
    unsat: set of integers
        set of node ids that have the same degree as w and are unsaturated.
    h_node_residual_in: dict of integers
        for a given node, keeps track of the remaining stubs to be added.
    chords: set of tuples
        keeps track of available positions to add edges.
    h_partition_out: dict of integers
        for a given node, keeps track of its partition id (out degree).
    partition: integer
        partition id to check if chords have to be updated.

    Notes
    -----
    Same operation as directed_neighbor_switch except it handles this operation
    for incoming edges instead of outgoing.
    """
    w_prime = unsat.pop()
    unsat.add(w_prime)
    # slightly faster declaring these as variables.
    w_neighbs = list(G.predecessors(w))
    w_prime_neighbs = list(G.predecessors(w_prime))
    # select node v, a neighbor of w, that is not connected to w_prime.
    for v in w_neighbs:
        if (v not in w_prime_neighbs) and w_prime != v:
            # removes (v,w), add (v,w_prime) and update data structures.
            G.remove_edge(v, w)
            G.add_edge(v, w_prime)
            if h_partition_out[v] == partition:
                chords.add((v, w))
                chords.discard((v, w_prime))

            h_node_residual_in[w] += 1
            h_node_residual_in[w_prime] -= 1
            if h_node_residual_in[w_prime] == 0:
                unsat.remove(w_prime)
            return None

    # If neighbor switch didn't work, use the unsaturated node.
    return w_prime


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def directed_joint_degree_graph(in_degrees, out_degrees, nkk, seed=None):
    """Generates a random simple directed graph with the joint degree.

    Parameters
    ----------
    degree_seq :  list of tuples (of size 3)
        degree sequence contains tuples of nodes with node id, in degree and
        out degree.
    nkk  :  dictionary of dictionary of integers
        directed joint degree dictionary, for nodes of out degree k (first
        level of dict) and nodes of in degree l (second level of dict)
        describes the number of edges.
    seed : hashable object, optional
        Seed for random number generator.

    Returns
    -------
    G : Graph
        A directed graph with the specified inputs.

    Raises
    ------
    NetworkXError
        If degree_seq and nkk are not realizable as a simple directed graph.


    Notes
    -----
    Similarly to the undirected version:
    In each iteration of the "while loop" the algorithm picks two disconnected
    nodes v and w, of degree k and l correspondingly,  for which nkk[k][l] has
    not reached its target yet i.e. (for given k,l): n_edges_add < nkk[k][l].
    It then adds edge (v,w) and always increases the number of edges in graph G
    by one.

    The intelligence of the algorithm lies in the fact that  it is always
    possible to add an edge between disconnected nodes v and w, for which
    nkk[degree(v)][degree(w)] has not reached its target, even if one or both
    nodes do not have free stubs. If either node v or w does not have a free
    stub, we perform a "neighbor switch", an edge rewiring move that releases a
    free stub while keeping nkk the same.

    The difference for the directed version lies in the fact that neighbor
    switches might not be able to rewire, but in these cases unsaturated nodes
    can be reassigned to use instead, see [1] for detailed description and
    proofs.

    The algorithm continues for E (number of edges in the graph) iterations of
    the "while loop", at which point all entries of the given nkk[k][l] have
    reached their target values and the construction is complete.

    References
    ----------
    [1] B. Tillman, A. Markopoulou, C. T. Butts & M. Gjoka,
        "Construction of Directed 2K Graphs". In Proc. of KDD 2017.

    Examples
    --------
    >>> in_degrees = [0, 1, 1, 2]
    >>> out_degrees = [1, 1, 1, 1]
    >>> nkk = {1: {1: 2, 2: 2}}
    >>> G = nx.directed_joint_degree_graph(in_degrees, out_degrees, nkk)
    >>>
    """
    if not is_valid_directed_joint_degree(in_degrees, out_degrees, nkk):
        msg = "Input is not realizable as a simple graph"
        raise nx.NetworkXError(msg)

    # start with an empty directed graph.
    G = nx.DiGraph()

    # for a given group, keep the list of all node ids.
    h_degree_nodelist_in = {}
    h_degree_nodelist_out = {}
    # for a given group, keep the list of all unsaturated node ids.
    h_degree_nodelist_in_unsat = {}
    h_degree_nodelist_out_unsat = {}
    # for a given node, keep track of the remaining stubs to be added.
    h_node_residual_out = {}
    h_node_residual_in = {}
    # for a given node, keep track of the partition id.
    h_partition_out = {}
    h_partition_in = {}
    # keep track of non-chords between pairs of partition ids.
    non_chords = {}

    # populate data structures
    for idx, i in enumerate(in_degrees):
        idx = int(idx)
        if i > 0:
            h_degree_nodelist_in.setdefault(i, [])
            h_degree_nodelist_in_unsat.setdefault(i, set())
            h_degree_nodelist_in[i].append(idx)
            h_degree_nodelist_in_unsat[i].add(idx)
            h_node_residual_in[idx] = i
            h_partition_in[idx] = i

    for idx, o in enumerate(out_degrees):
        o = out_degrees[idx]
        non_chords[(o, in_degrees[idx])] = non_chords.get((o, in_degrees[idx]), 0) + 1
        idx = int(idx)
        if o > 0:
            h_degree_nodelist_out.setdefault(o, [])
            h_degree_nodelist_out_unsat.setdefault(o, set())
            h_degree_nodelist_out[o].append(idx)
            h_degree_nodelist_out_unsat[o].add(idx)
            h_node_residual_out[idx] = o
            h_partition_out[idx] = o

        G.add_node(idx)

    nk_in = {}
    nk_out = {}
    for p in h_degree_nodelist_in:
        nk_in[p] = len(h_degree_nodelist_in[p])
    for p in h_degree_nodelist_out:
        nk_out[p] = len(h_degree_nodelist_out[p])

    # iterate over every degree pair (k,l) and add the number of edges given
    # for each pair.
    for k in nkk:
        for l in nkk[k]:
            n_edges_add = nkk[k][l]

            if n_edges_add > 0:
                # chords contains a random set of potential edges.
                chords = set()

                k_len = nk_out[k]
                l_len = nk_in[l]
                chords_sample = seed.sample(
                    range(k_len * l_len), n_edges_add + non_chords.get((k, l), 0)
                )

                num = 0
                while len(chords) < n_edges_add:
                    i = h_degree_nodelist_out[k][chords_sample[num] % k_len]
                    j = h_degree_nodelist_in[l][chords_sample[num] // k_len]
                    num += 1
                    if i != j:
                        chords.add((i, j))

                # k_unsat and l_unsat consist of nodes of in/out degree k and l
                # that are unsaturated i.e. those nodes that have at least one
                # available stub
                k_unsat = h_degree_nodelist_out_unsat[k]
                l_unsat = h_degree_nodelist_in_unsat[l]

                while n_edges_add > 0:
                    v, w = chords.pop()
                    chords.add((v, w))

                    # if node v has no free stubs then do neighbor switch.
                    if h_node_residual_out[v] == 0:
                        _v = _directed_neighbor_switch(
                            G,
                            v,
                            k_unsat,
                            h_node_residual_out,
                            chords,
                            h_partition_in,
                            l,
                        )
                        if _v is not None:
                            v = _v

                    # if node w has no free stubs then do neighbor switch.
                    if h_node_residual_in[w] == 0:
                        _w = _directed_neighbor_switch_rev(
                            G,
                            w,
                            l_unsat,
                            h_node_residual_in,
                            chords,
                            h_partition_out,
                            k,
                        )
                        if _w is not None:
                            w = _w

                    # add edge (v,w) and update data structures.
                    G.add_edge(v, w)
                    h_node_residual_out[v] -= 1
                    h_node_residual_in[w] -= 1
                    n_edges_add -= 1
                    chords.discard((v, w))

                    if h_node_residual_out[v] == 0:
                        k_unsat.discard(v)
                    if h_node_residual_in[w] == 0:
                        l_unsat.discard(w)
    return G
<|endoftext|>"""
Generators for some directed graphs, including growing network (GN) graphs and
scale-free graphs.

"""

import numbers
from collections import Counter

import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import discrete_sequence, py_random_state, weighted_choice

__all__ = [
    "gn_graph",
    "gnc_graph",
    "gnr_graph",
    "random_k_out_graph",
    "scale_free_graph",
]


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def gn_graph(n, kernel=None, create_using=None, seed=None):
    """Returns the growing network (GN) digraph with `n` nodes.

    The GN graph is built by adding nodes one at a time with a link to one
    previously added node.  The target node for the link is chosen with
    probability based on degree.  The default attachment kernel is a linear
    function of the degree of a node.

    The graph is always a (directed) tree.

    Parameters
    ----------
    n : int
        The number of nodes for the generated graph.
    kernel : function
        The attachment kernel.
    create_using : NetworkX graph constructor, optional (default DiGraph)
        Graph type to create. If graph instance, then cleared before populated.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Examples
    --------
    To create the undirected GN graph, use the :meth:`~DiGraph.to_directed`
    method::

    >>> D = nx.gn_graph(10)  # the GN graph
    >>> G = D.to_undirected()  # the undirected version

    To specify an attachment kernel, use the `kernel` keyword argument::

    >>> D = nx.gn_graph(10, kernel=lambda x: x**1.5)  # A_k = k^1.5

    References
    ----------
    .. [1] P. L. Krapivsky and S. Redner,
           Organization of Growing Random Networks,
           Phys. Rev. E, 63, 066123, 2001.
    """
    G = empty_graph(1, create_using, default=nx.DiGraph)
    if not G.is_directed():
        raise nx.NetworkXError("create_using must indicate a Directed Graph")

    if kernel is None:

        def kernel(x):
            return x

    if n == 1:
        return G

    G.add_edge(1, 0)  # get started
    ds = [1, 1]  # degree sequence

    for source in range(2, n):
        # compute distribution from kernel and degree
        dist = [kernel(d) for d in ds]
        # choose target from discrete distribution
        target = discrete_sequence(1, distribution=dist, seed=seed)[0]
        G.add_edge(source, target)
        ds.append(1)  # the source has only one link (degree one)
        ds[target] += 1  # add one to the target link degree
    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def gnr_graph(n, p, create_using=None, seed=None):
    """Returns the growing network with redirection (GNR) digraph with `n`
    nodes and redirection probability `p`.

    The GNR graph is built by adding nodes one at a time with a link to one
    previously added node.  The previous target node is chosen uniformly at
    random.  With probability `p` the link is instead "redirected" to the
    successor node of the target.

    The graph is always a (directed) tree.

    Parameters
    ----------
    n : int
        The number of nodes for the generated graph.
    p : float
        The redirection probability.
    create_using : NetworkX graph constructor, optional (default DiGraph)
        Graph type to create. If graph instance, then cleared before populated.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Examples
    --------
    To create the undirected GNR graph, use the :meth:`~DiGraph.to_directed`
    method::

    >>> D = nx.gnr_graph(10, 0.5)  # the GNR graph
    >>> G = D.to_undirected()  # the undirected version

    References
    ----------
    .. [1] P. L. Krapivsky and S. Redner,
           Organization of Growing Random Networks,
           Phys. Rev. E, 63, 066123, 2001.
    """
    G = empty_graph(1, create_using, default=nx.DiGraph)
    if not G.is_directed():
        raise nx.NetworkXError("create_using must indicate a Directed Graph")

    if n == 1:
        return G

    for source in range(1, n):
        target = seed.randrange(0, source)
        if seed.random() < p and target != 0:
            target = next(G.successors(target))
        G.add_edge(source, target)
    return G


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def gnc_graph(n, create_using=None, seed=None):
    """Returns the growing network with copying (GNC) digraph with `n` nodes.

    The GNC graph is built by adding nodes one at a time with a link to one
    previously added node (chosen uniformly at random) and to all of that
    node's successors.

    Parameters
    ----------
    n : int
        The number of nodes for the generated graph.
    create_using : NetworkX graph constructor, optional (default DiGraph)
        Graph type to create. If graph instance, then cleared before populated.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    References
    ----------
    .. [1] P. L. Krapivsky and S. Redner,
           Network Growth by Copying,
           Phys. Rev. E, 71, 036118, 2005k.},
    """
    G = empty_graph(1, create_using, default=nx.DiGraph)
    if not G.is_directed():
        raise nx.NetworkXError("create_using must indicate a Directed Graph")

    if n == 1:
        return G

    for source in range(1, n):
        target = seed.randrange(0, source)
        for succ in G.successors(target):
            G.add_edge(source, succ)
        G.add_edge(source, target)
    return G


@py_random_state(6)
@nx._dispatchable(graphs=None, returns_graph=True)
def scale_free_graph(
    n,
    alpha=0.41,
    beta=0.54,
    gamma=0.05,
    delta_in=0.2,
    delta_out=0,
    seed=None,
    initial_graph=None,
):
    """Returns a scale-free directed graph.

    Parameters
    ----------
    n : integer
        Number of nodes in graph
    alpha : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the in-degree distribution.
    beta : float
        Probability for adding an edge between two existing nodes.
        One existing node is chosen randomly according the in-degree
        distribution and the other chosen randomly according to the out-degree
        distribution.
    gamma : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the out-degree distribution.
    delta_in : float
        Bias for choosing nodes from in-degree distribution.
    delta_out : float
        Bias for choosing nodes from out-degree distribution.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    initial_graph : MultiDiGraph instance, optional
        Build the scale-free graph starting from this initial MultiDiGraph,
        if provided.

    Returns
    -------
    MultiDiGraph

    Examples
    --------
    Create a scale-free graph on one hundred nodes::

    >>> G = nx.scale_free_graph(100)

    Notes
    -----
    The sum of `alpha`, `beta`, and `gamma` must be 1.

    References
    ----------
    .. [1] B. Bollobás, C. Borgs, J. Chayes, and O. Riordan,
           Directed scale-free graphs,
           Proceedings of the fourteenth annual ACM-SIAM Symposium on
           Discrete Algorithms, 132--139, 2003.
    """

    def _choose_node(candidates, node_list, delta):
        if delta > 0:
            bias_sum = len(node_list) * delta
            p_delta = bias_sum / (bias_sum + len(candidates))
            if seed.random() < p_delta:
                return seed.choice(node_list)
        return seed.choice(candidates)

    if initial_graph is not None and hasattr(initial_graph, "_adj"):
        if not isinstance(initial_graph, nx.MultiDiGraph):
            raise nx.NetworkXError("initial_graph must be a MultiDiGraph.")
        G = initial_graph
    else:
        # Start with 3-cycle
        G = nx.MultiDiGraph([(0, 1), (1, 2), (2, 0)])

    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if beta <= 0:
        raise ValueError("beta must be > 0.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    if abs(alpha + beta + gamma - 1.0) >= 1e-9:
        raise ValueError("alpha+beta+gamma must equal 1.")

    if delta_in < 0:
        raise ValueError("delta_in must be >= 0.")

    if delta_out < 0:
        raise ValueError("delta_out must be >= 0.")

    # pre-populate degree states
    vs = sum((count * [idx] for idx, count in G.out_degree()), [])
    ws = sum((count * [idx] for idx, count in G.in_degree()), [])

    # pre-populate node state
    node_list = list(G.nodes())

    # see if there already are number-based nodes
    numeric_nodes = [n for n in node_list if isinstance(n, numbers.Number)]
    if len(numeric_nodes) > 0:
        # set cursor for new nodes appropriately
        cursor = max(int(n.real) for n in numeric_nodes) + 1
    else:
        # or start at zero
        cursor = 0

    while len(G) < n:
        r = seed.random()

        # random choice in alpha,beta,gamma ranges
        if r < alpha:
            # alpha
            # add new node v
            v = cursor
            cursor += 1
            # also add to node state
            node_list.append(v)
            # choose w according to in-degree and delta_in
            w = _choose_node(ws, node_list, delta_in)

        elif r < alpha + beta:
            # beta
            # choose v according to out-degree and delta_out
            v = _choose_node(vs, node_list, delta_out)
            # choose w according to in-degree and delta_in
            w = _choose_node(ws, node_list, delta_in)

        else:
            # gamma
            # choose v according to out-degree and delta_out
            v = _choose_node(vs, node_list, delta_out)
            # add new node w
            w = cursor
            cursor += 1
            # also add to node state
            node_list.append(w)

        # add edge to graph
        G.add_edge(v, w)

        # update degree states
        vs.append(v)
        ws.append(w)

    return G


@py_random_state(4)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_uniform_k_out_graph(n, k, self_loops=True, with_replacement=True, seed=None):
    """Returns a random `k`-out graph with uniform attachment.

    A random `k`-out graph with uniform attachment is a multidigraph
    generated by the following algorithm. For each node *u*, choose
    `k` nodes *v* uniformly at random (with replacement). Add a
    directed edge joining *u* to *v*.

    Parameters
    ----------
    n : int
        The number of nodes in the returned graph.

    k : int
        The out-degree of each node in the returned graph.

    self_loops : bool
        If True, self-loops are allowed when generating the graph.

    with_replacement : bool
        If True, neighbors are chosen with replacement and the
        returned graph will be a directed multigraph. Otherwise,
        neighbors are chosen without replacement and the returned graph
        will be a directed graph.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    NetworkX graph
        A `k`-out-regular directed graph generated according to the
        above algorithm. It will be a multigraph if and only if
        `with_replacement` is True.

    Raises
    ------
    ValueError
        If `with_replacement` is False and `k` is greater than
        `n`.

    See also
    --------
    random_k_out_graph

    Notes
    -----
    The return digraph or multidigraph may not be strongly connected, or
    even weakly connected.

    If `with_replacement` is True, this function is similar to
    :func:`random_k_out_graph`, if that function had parameter `alpha`
    set to positive infinity.

    """
    if with_replacement:
        create_using = nx.MultiDiGraph()

        def sample(v, nodes):
            if not self_loops:
                nodes = nodes - {v}
            return (seed.choice(list(nodes)) for i in range(k))

    else:
        create_using = nx.DiGraph()

        def sample(v, nodes):
            if not self_loops:
                nodes = nodes - {v}
            return seed.sample(list(nodes), k)

    G = nx.empty_graph(n, create_using)
    nodes = set(G)
    for u in G:
        G.add_edges_from((u, v) for v in sample(u, nodes))
    return G


@py_random_state(4)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_k_out_graph(n, k, alpha, self_loops=True, seed=None):
    """Returns a random `k`-out graph with preferential attachment.

    A random `k`-out graph with preferential attachment is a
    multidigraph generated by the following algorithm.

    1. Begin with an empty digraph, and initially set each node to have
       weight `alpha`.
    2. Choose a node `u` with out-degree less than `k` uniformly at
       random.
    3. Choose a node `v` from with probability proportional to its
       weight.
    4. Add a directed edge from `u` to `v`, and increase the weight
       of `v` by one.
    5. If each node has out-degree `k`, halt, otherwise repeat from
       step 2.

    For more information on this model of random graph, see [1].

    Parameters
    ----------
    n : int
        The number of nodes in the returned graph.

    k : int
        The out-degree of each node in the returned graph.

    alpha : float
        A positive :class:`float` representing the initial weight of
        each vertex. A higher number means that in step 3 above, nodes
        will be chosen more like a true uniformly random sample, and a
        lower number means that nodes are more likely to be chosen as
        their in-degree increases. If this parameter is not positive, a
        :exc:`ValueError` is raised.

    self_loops : bool
        If True, self-loops are allowed when generating the graph.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    :class:`~networkx.classes.MultiDiGraph`
        A `k`-out-regular multidigraph generated according to the above
        algorithm.

    Raises
    ------
    ValueError
        If `alpha` is not positive.

    Notes
    -----
    The returned multidigraph may not be strongly connected, or even
    weakly connected.

    References
    ----------
    [1]: Peterson, Nicholas R., and Boris Pittel.
         "Distance between two random `k`-out digraphs, with and without
         preferential attachment."
         arXiv preprint arXiv:1311.5961 (2013).
         <https://arxiv.org/abs/1311.5961>

    """
    if alpha < 0:
        raise ValueError("alpha must be positive")
    G = nx.empty_graph(n, create_using=nx.MultiDiGraph)
    weights = Counter({v: alpha for v in G})
    for i in range(k * n):
        u = seed.choice([v for v, d in G.out_degree() if d < k])
        # If self-loops are not allowed, make the source node `u` have
        # weight zero.
        if not self_loops:
            adjustment = Counter({u: weights[u]})
        else:
            adjustment = Counter()
        v = weighted_choice(weights - adjustment, seed=seed)
        G.add_edge(u, v)
        weights[v] += 1
    return G
<|endoftext|>"""Functions for generating stochastic graphs from a given weighted directed
graph.

"""

import networkx as nx
from networkx.classes import DiGraph, MultiDiGraph
from networkx.utils import not_implemented_for

__all__ = ["stochastic_graph"]


@not_implemented_for("undirected")
@nx._dispatchable(
    edge_attrs="weight", mutates_input={"not copy": 1}, returns_graph=True
)
def stochastic_graph(G, copy=True, weight="weight"):
    """Returns a right-stochastic representation of directed graph `G`.

    A right-stochastic graph is a weighted digraph in which for each
    node, the sum of the weights of all the out-edges of that node is
    1. If the graph is already weighted (for example, via a 'weight'
    edge attribute), the reweighting takes that into account.

    Parameters
    ----------
    G : directed graph
        A :class:`~networkx.DiGraph` or :class:`~networkx.MultiDiGraph`.

    copy : boolean, optional
        If this is True, then this function returns a new graph with
        the stochastic reweighting. Otherwise, the original graph is
        modified in-place (and also returned, for convenience).

    weight : edge attribute key (optional, default='weight')
        Edge attribute key used for reading the existing weight and
        setting the new weight.  If no attribute with this key is found
        for an edge, then the edge weight is assumed to be 1. If an edge
        has a weight, it must be a positive number.

    """
    if copy:
        G = MultiDiGraph(G) if G.is_multigraph() else DiGraph(G)
    # There is a tradeoff here: the dictionary of node degrees may
    # require a lot of memory, whereas making a call to `G.out_degree`
    # inside the loop may be costly in computation time.
    degree = dict(G.out_degree(weight=weight))
    for u, v, d in G.edges(data=True):
        if degree[u] == 0:
            d[weight] = 0
        else:
            d[weight] = d.get(weight, 1) / degree[u]
    nx._clear_cache(G)
    return G
<|endoftext|>"""
Various small and named graphs, together with some compact generators.

"""

__all__ = [
    "LCF_graph",
    "bull_graph",
    "chvatal_graph",
    "cubical_graph",
    "desargues_graph",
    "diamond_graph",
    "dodecahedral_graph",
    "frucht_graph",
    "heawood_graph",
    "hoffman_singleton_graph",
    "house_graph",
    "house_x_graph",
    "icosahedral_graph",
    "krackhardt_kite_graph",
    "moebius_kantor_graph",
    "octahedral_graph",
    "pappus_graph",
    "petersen_graph",
    "sedgewick_maze_graph",
    "tetrahedral_graph",
    "truncated_cube_graph",
    "truncated_tetrahedron_graph",
    "tutte_graph",
]

from functools import wraps

import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
    complete_graph,
    cycle_graph,
    empty_graph,
    path_graph,
)


def _raise_on_directed(func):
    """
    A decorator which inspects the `create_using` argument and raises a
    NetworkX exception when `create_using` is a DiGraph (class or instance) for
    graph generators that do not support directed outputs.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get("create_using") is not None:
            G = nx.empty_graph(create_using=kwargs["create_using"])
            if G.is_directed():
                raise NetworkXError("Directed Graph not supported")
        return func(*args, **kwargs)

    return wrapper


@nx._dispatchable(graphs=None, returns_graph=True)
def LCF_graph(n, shift_list, repeats, create_using=None):
    """
    Return the cubic graph specified in LCF notation.

    LCF (Lederberg-Coxeter-Fruchte) notation[1]_ is a compressed
    notation used in the generation of various cubic Hamiltonian
    graphs of high symmetry. See, for example, `dodecahedral_graph`,
    `desargues_graph`, `heawood_graph` and `pappus_graph`.

    Nodes are drawn from ``range(n)``. Each node ``n_i`` is connected with
    node ``n_i + shift % n`` where ``shift`` is given by cycling through
    the input `shift_list` `repeat` s times.

    Parameters
    ----------
    n : int
       The starting graph is the `n`-cycle with nodes ``0, ..., n-1``.
       The null graph is returned if `n` < 1.

    shift_list : list
       A list of integer shifts mod `n`, ``[s1, s2, .., sk]``

    repeats : int
       Integer specifying the number of times that shifts in `shift_list`
       are successively applied to each current node in the n-cycle
       to generate an edge between ``n_current`` and ``n_current + shift mod n``.

    Returns
    -------
    G : Graph
       A graph instance created from the specified LCF notation.

    Examples
    --------
    The utility graph $K_{3,3}$

    >>> G = nx.LCF_graph(6, [3, -3], 3)
    >>> G.edges()
    EdgeView([(0, 1), (0, 5), (0, 3), (1, 2), (1, 4), (2, 3), (2, 5), (3, 4), (4, 5)])

    The Heawood graph:

    >>> G = nx.LCF_graph(14, [5, -5], 7)
    >>> nx.is_isomorphic(G, nx.heawood_graph())
    True

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/LCF_notation

    """
    if n <= 0:
        return empty_graph(0, create_using)

    # start with the n-cycle
    G = cycle_graph(n, create_using)
    if G.is_directed():
        raise NetworkXError("Directed Graph not supported")
    G.name = "LCF_graph"
    nodes = sorted(G)

    n_extra_edges = repeats * len(shift_list)
    # edges are added n_extra_edges times
    # (not all of these need be new)
    if n_extra_edges < 1:
        return G

    for i in range(n_extra_edges):
        shift = shift_list[i % len(shift_list)]  # cycle through shift_list
        v1 = nodes[i % n]  # cycle repeatedly through nodes
        v2 = nodes[(i + shift) % n]
        G.add_edge(v1, v2)
    return G


# -------------------------------------------------------------------------------
#   Various small and named graphs
# -------------------------------------------------------------------------------


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def bull_graph(create_using=None):
    """
    Returns the Bull Graph

    The Bull Graph has 5 nodes and 5 edges. It is a planar undirected
    graph in the form of a triangle with two disjoint pendant edges [1]_
    The name comes from the triangle and pendant edges representing
    respectively the body and legs of a bull.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        A bull graph with 5 nodes

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bull_graph.

    """
    G = nx.from_dict_of_lists(
        {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 4], 3: [1], 4: [2]},
        create_using=create_using,
    )
    G.name = "Bull Graph"
    return G


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def chvatal_graph(create_using=None):
    """
    Returns the Chvátal Graph

    The Chvátal Graph is an undirected graph with 12 nodes and 24 edges [1]_.
    It has 370 distinct (directed) Hamiltonian cycles, giving a unique generalized
    LCF notation of order 4, two of order 6 , and 43 of order 1 [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        The Chvátal graph with 12 nodes and 24 edges

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chv%C3%A1tal_graph
    .. [2] https://mathworld.wolfram.com/ChvatalGraph.html

    """
    G = nx.from_dict_of_lists(
        {
            0: [1, 4, 6, 9],
            1: [2, 5, 7],
            2: [3, 6, 8],
            3: [4, 7, 9],
            4: [5, 8],
            5: [10, 11],
            6: [10, 11],
            7: [8, 11],
            8: [10],
            9: [10, 11],
        },
        create_using=create_using,
    )
    G.name = "Chvatal Graph"
    return G


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def cubical_graph(create_using=None):
    """
    Returns the 3-regular Platonic Cubical Graph

    The skeleton of the cube (the nodes and edges) form a graph, with 8
    nodes, and 12 edges. It is a special case of the hypercube graph.
    It is one of 5 Platonic graphs, each a skeleton of its
    Platonic solid [1]_.
    Such graphs arise in parallel processing in computers.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        A cubical graph with 8 nodes and 12 edges

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cube#Cubical_graph

    """
    G = nx.from_dict_of_lists(
        {
            0: [1, 3, 4],
            1: [0, 2, 7],
            2: [1, 3, 6],
            3: [0, 2, 5],
            4: [0, 5, 7],
            5: [3, 4, 6],
            6: [2, 5, 7],
            7: [1, 4, 6],
        },
        create_using=create_using,
    )
    G.name = "Platonic Cubical Graph"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def desargues_graph(create_using=None):
    """
    Returns the Desargues Graph

    The Desargues Graph is a non-planar, distance-transitive cubic graph
    with 20 nodes and 30 edges [1]_.
    It is a symmetric graph. It can be represented in LCF notation
    as [5,-5,9,-9]^5 [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Desargues Graph with 20 nodes and 30 edges

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Desargues_graph
    .. [2] https://mathworld.wolfram.com/DesarguesGraph.html
    """
    G = LCF_graph(20, [5, -5, 9, -9], 5, create_using)
    G.name = "Desargues Graph"
    return G


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def diamond_graph(create_using=None):
    """
    Returns the Diamond graph

    The Diamond Graph is  planar undirected graph with 4 nodes and 5 edges.
    It is also sometimes known as the double triangle graph or kite graph [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Diamond Graph with 4 nodes and 5 edges

    References
    ----------
    .. [1] https://mathworld.wolfram.com/DiamondGraph.html
    """
    G = nx.from_dict_of_lists(
        {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2]}, create_using=create_using
    )
    G.name = "Diamond Graph"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def dodecahedral_graph(create_using=None):
    """
    Returns the Platonic Dodecahedral graph.

    The dodecahedral graph has 20 nodes and 30 edges. The skeleton of the
    dodecahedron forms a graph. It is one of 5 Platonic graphs [1]_.
    It can be described in LCF notation as:
    ``[10, 7, 4, -4, -7, 10, -4, 7, -7, 4]^2`` [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Dodecahedral Graph with 20 nodes and 30 edges

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Regular_dodecahedron#Dodecahedral_graph
    .. [2] https://mathworld.wolfram.com/DodecahedralGraph.html

    """
    G = LCF_graph(20, [10, 7, 4, -4, -7, 10, -4, 7, -7, 4], 2, create_using)
    G.name = "Dodecahedral Graph"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def frucht_graph(create_using=None):
    """
    Returns the Frucht Graph.

    The Frucht Graph is the smallest cubical graph whose
    automorphism group consists only of the identity element [1]_.
    It has 12 nodes and 18 edges and no nontrivial symmetries.
    It is planar and Hamiltonian [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Frucht Graph with 12 nodes and 18 edges

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Frucht_graph
    .. [2] https://mathworld.wolfram.com/FruchtGraph.html

    """
    G = cycle_graph(7, create_using)
    G.add_edges_from(
        [
            [0, 7],
            [1, 7],
            [2, 8],
            [3, 9],
            [4, 9],
            [5, 10],
            [6, 10],
            [7, 11],
            [8, 11],
            [8, 9],
            [10, 11],
        ]
    )

    G.name = "Frucht Graph"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def heawood_graph(create_using=None):
    """
    Returns the Heawood Graph, a (3,6) cage.

    The Heawood Graph is an undirected graph with 14 nodes and 21 edges,
    named after Percy John Heawood [1]_.
    It is cubic symmetric, nonplanar, Hamiltonian, and can be represented
    in LCF notation as ``[5,-5]^7`` [2]_.
    It is the unique (3,6)-cage: the regular cubic graph of girth 6 with
    minimal number of vertices [3]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Heawood Graph with 14 nodes and 21 edges

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Heawood_graph
    .. [2] https://mathworld.wolfram.com/HeawoodGraph.html
    .. [3] https://www.win.tue.nl/~aeb/graphs/Heawood.html

    """
    G = LCF_graph(14, [5, -5], 7, create_using)
    G.name = "Heawood Graph"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def hoffman_singleton_graph():
    """
    Returns the Hoffman-Singleton Graph.

    The Hoffman–Singleton graph is a symmetrical undirected graph
    with 50 nodes and 175 edges.
    All indices lie in ``Z % 5``: that is, the integers mod 5 [1]_.
    It is the only regular graph of vertex degree 7, diameter 2, and girth 5.
    It is the unique (7,5)-cage graph and Moore graph, and contains many
    copies of the Petersen graph [2]_.

    Returns
    -------
    G : networkx Graph
        Hoffman–Singleton Graph with 50 nodes and 175 edges

    Notes
    -----
    Constructed from pentagon and pentagram as follows: Take five pentagons $P_h$
    and five pentagrams $Q_i$ . Join vertex $j$ of $P_h$ to vertex $h·i+j$ of $Q_i$ [3]_.

    References
    ----------
    .. [1] https://blogs.ams.org/visualinsight/2016/02/01/hoffman-singleton-graph/
    .. [2] https://mathworld.wolfram.com/Hoffman-SingletonGraph.html
    .. [3] https://en.wikipedia.org/wiki/Hoffman%E2%80%93Singleton_graph

    """
    G = nx.Graph()
    for i in range(5):
        for j in range(5):
            G.add_edge(("pentagon", i, j), ("pentagon", i, (j - 1) % 5))
            G.add_edge(("pentagon", i, j), ("pentagon", i, (j + 1) % 5))
            G.add_edge(("pentagram", i, j), ("pentagram", i, (j - 2) % 5))
            G.add_edge(("pentagram", i, j), ("pentagram", i, (j + 2) % 5))
            for k in range(5):
                G.add_edge(("pentagon", i, j), ("pentagram", k, (i * k + j) % 5))
    G = nx.convert_node_labels_to_integers(G)
    G.name = "Hoffman-Singleton Graph"
    return G


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def house_graph(create_using=None):
    """
    Returns the House graph (square with triangle on top)

    The house graph is a simple undirected graph with
    5 nodes and 6 edges [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        House graph in the form of a square with a triangle on top

    References
    ----------
    .. [1] https://mathworld.wolfram.com/HouseGraph.html
    """
    G = nx.from_dict_of_lists(
        {0: [1, 2], 1: [0, 3], 2: [0, 3, 4], 3: [1, 2, 4], 4: [2, 3]},
        create_using=create_using,
    )
    G.name = "House Graph"
    return G


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def house_x_graph(create_using=None):
    """
    Returns the House graph with a cross inside the house square.

    The House X-graph is the House graph plus the two edges connecting diagonally
    opposite vertices of the square base. It is also one of the two graphs
    obtained by removing two edges from the pentatope graph [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        House graph with diagonal vertices connected

    References
    ----------
    .. [1] https://mathworld.wolfram.com/HouseGraph.html
    """
    G = house_graph(create_using)
    G.add_edges_from([(0, 3), (1, 2)])
    G.name = "House-with-X-inside Graph"
    return G


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def icosahedral_graph(create_using=None):
    """
    Returns the Platonic Icosahedral graph.

    The icosahedral graph has 12 nodes and 30 edges. It is a Platonic graph
    whose nodes have the connectivity of the icosahedron. It is undirected,
    regular and Hamiltonian [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Icosahedral graph with 12 nodes and 30 edges.

    References
    ----------
    .. [1] https://mathworld.wolfram.com/IcosahedralGraph.html
    """
    G = nx.from_dict_of_lists(
        {
            0: [1, 5, 7, 8, 11],
            1: [2, 5, 6, 8],
            2: [3, 6, 8, 9],
            3: [4, 6, 9, 10],
            4: [5, 6, 10, 11],
            5: [6, 11],
            7: [8, 9, 10, 11],
            8: [9],
            9: [10],
            10: [11],
        },
        create_using=create_using,
    )
    G.name = "Platonic Icosahedral Graph"
    return G


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def krackhardt_kite_graph(create_using=None):
    """
    Returns the Krackhardt Kite Social Network.

    A 10 actor social network introduced by David Krackhardt
    to illustrate different centrality measures [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Krackhardt Kite graph with 10 nodes and 18 edges

    Notes
    -----
    The traditional labeling is:
    Andre=1, Beverley=2, Carol=3, Diane=4,
    Ed=5, Fernando=6, Garth=7, Heather=8, Ike=9, Jane=10.

    References
    ----------
    .. [1] Krackhardt, David. "Assessing the Political Landscape: Structure,
       Cognition, and Power in Organizations". Administrative Science Quarterly.
       35 (2): 342–369. doi:10.2307/2393394. JSTOR 2393394. June 1990.

    """
    G = nx.from_dict_of_lists(
        {
            0: [1, 2, 3, 5],
            1: [0, 3, 4, 6],
            2: [0, 3, 5],
            3: [0, 1, 2, 4, 5, 6],
            4: [1, 3, 6],
            5: [0, 2, 3, 6, 7],
            6: [1, 3, 4, 5, 7],
            7: [5, 6, 8],
            8: [7, 9],
            9: [8],
        },
        create_using=create_using,
    )
    G.name = "Krackhardt Kite Social Network"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def moebius_kantor_graph(create_using=None):
    """
    Returns the Moebius-Kantor graph.

    The Möbius-Kantor graph is the cubic symmetric graph on 16 nodes.
    Its LCF notation is [5,-5]^8, and it is isomorphic to the generalized
    Petersen graph [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Moebius-Kantor graph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius%E2%80%93Kantor_graph

    """
    G = LCF_graph(16, [5, -5], 8, create_using)
    G.name = "Moebius-Kantor Graph"
    return G


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def octahedral_graph(create_using=None):
    """
    Returns the Platonic Octahedral graph.

    The octahedral graph is the 6-node 12-edge Platonic graph having the
    connectivity of the octahedron [1]_. If 6 couples go to a party,
    and each person shakes hands with every person except his or her partner,
    then this graph describes the set of handshakes that take place;
    for this reason it is also called the cocktail party graph [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Octahedral graph

    References
    ----------
    .. [1] https://mathworld.wolfram.com/OctahedralGraph.html
    .. [2] https://en.wikipedia.org/wiki/Tur%C3%A1n_graph#Special_cases

    """
    G = nx.from_dict_of_lists(
        {0: [1, 2, 3, 4], 1: [2, 3, 5], 2: [4, 5], 3: [4, 5], 4: [5]},
        create_using=create_using,
    )
    G.name = "Platonic Octahedral Graph"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def pappus_graph():
    """
    Returns the Pappus graph.

    The Pappus graph is a cubic symmetric distance-regular graph with 18 nodes
    and 27 edges. It is Hamiltonian and can be represented in LCF notation as
    [5,7,-7,7,-7,-5]^3 [1]_.

    Returns
    -------
    G : networkx Graph
        Pappus graph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Pappus_graph
    """
    G = LCF_graph(18, [5, 7, -7, 7, -7, -5], 3)
    G.name = "Pappus Graph"
    return G


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def petersen_graph(create_using=None):
    """
    Returns the Petersen graph.

    The Peterson graph is a cubic, undirected graph with 10 nodes and 15 edges [1]_.
    Julius Petersen constructed the graph as the smallest counterexample
    against the claim that a connected bridgeless cubic graph
    has an edge colouring with three colours [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Petersen graph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Petersen_graph
    .. [2] https://www.win.tue.nl/~aeb/drg/graphs/Petersen.html
    """
    G = nx.from_dict_of_lists(
        {
            0: [1, 4, 5],
            1: [0, 2, 6],
            2: [1, 3, 7],
            3: [2, 4, 8],
            4: [3, 0, 9],
            5: [0, 7, 8],
            6: [1, 8, 9],
            7: [2, 5, 9],
            8: [3, 5, 6],
            9: [4, 6, 7],
        },
        create_using=create_using,
    )
    G.name = "Petersen Graph"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def sedgewick_maze_graph(create_using=None):
    """
    Return a small maze with a cycle.

    This is the maze used in Sedgewick, 3rd Edition, Part 5, Graph
    Algorithms, Chapter 18, e.g. Figure 18.2 and following [1]_.
    Nodes are numbered 0,..,7

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Small maze with a cycle

    References
    ----------
    .. [1] Figure 18.2, Chapter 18, Graph Algorithms (3rd Ed), Sedgewick
    """
    G = empty_graph(0, create_using)
    G.add_nodes_from(range(8))
    G.add_edges_from([[0, 2], [0, 7], [0, 5]])
    G.add_edges_from([[1, 7], [2, 6]])
    G.add_edges_from([[3, 4], [3, 5]])
    G.add_edges_from([[4, 5], [4, 7], [4, 6]])
    G.name = "Sedgewick Maze"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def tetrahedral_graph(create_using=None):
    """
    Returns the 3-regular Platonic Tetrahedral graph.

    Tetrahedral graph has 4 nodes and 6 edges. It is a
    special case of the complete graph, K4, and wheel graph, W4.
    It is one of the 5 platonic graphs [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Tetrahedral Graph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Tetrahedron#Tetrahedral_graph

    """
    G = complete_graph(4, create_using)
    G.name = "Platonic Tetrahedral Graph"
    return G


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def truncated_cube_graph(create_using=None):
    """
    Returns the skeleton of the truncated cube.

    The truncated cube is an Archimedean solid with 14 regular
    faces (6 octagonal and 8 triangular), 36 edges and 24 nodes [1]_.
    The truncated cube is created by truncating (cutting off) the tips
    of the cube one third of the way into each edge [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Skeleton of the truncated cube

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Truncated_cube
    .. [2] https://www.coolmath.com/reference/polyhedra-truncated-cube

    """
    G = nx.from_dict_of_lists(
        {
            0: [1, 2, 4],
            1: [11, 14],
            2: [3, 4],
            3: [6, 8],
            4: [5],
            5: [16, 18],
            6: [7, 8],
            7: [10, 12],
            8: [9],
            9: [17, 20],
            10: [11, 12],
            11: [14],
            12: [13],
            13: [21, 22],
            14: [15],
            15: [19, 23],
            16: [17, 18],
            17: [20],
            18: [19],
            19: [23],
            20: [21],
            21: [22],
            22: [23],
        },
        create_using=create_using,
    )
    G.name = "Truncated Cube Graph"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def truncated_tetrahedron_graph(create_using=None):
    """
    Returns the skeleton of the truncated Platonic tetrahedron.

    The truncated tetrahedron is an Archimedean solid with 4 regular hexagonal faces,
    4 equilateral triangle faces, 12 nodes and 18 edges. It can be constructed by truncating
    all 4 vertices of a regular tetrahedron at one third of the original edge length [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Skeleton of the truncated tetrahedron

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Truncated_tetrahedron

    """
    G = path_graph(12, create_using)
    G.add_edges_from([(0, 2), (0, 9), (1, 6), (3, 11), (4, 11), (5, 7), (8, 10)])
    G.name = "Truncated Tetrahedron Graph"
    return G


@_raise_on_directed
@nx._dispatchable(graphs=None, returns_graph=True)
def tutte_graph(create_using=None):
    """
    Returns the Tutte graph.

    The Tutte graph is a cubic polyhedral, non-Hamiltonian graph. It has
    46 nodes and 69 edges.
    It is a counterexample to Tait's conjecture that every 3-regular polyhedron
    has a Hamiltonian cycle.
    It can be realized geometrically from a tetrahedron by multiply truncating
    three of its vertices [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Tutte graph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Tutte_graph
    """
    G = nx.from_dict_of_lists(
        {
            0: [1, 2, 3],
            1: [4, 26],
            2: [10, 11],
            3: [18, 19],
            4: [5, 33],
            5: [6, 29],
            6: [7, 27],
            7: [8, 14],
            8: [9, 38],
            9: [10, 37],
            10: [39],
            11: [12, 39],
            12: [13, 35],
            13: [14, 15],
            14: [34],
            15: [16, 22],
            16: [17, 44],
            17: [18, 43],
            18: [45],
            19: [20, 45],
            20: [21, 41],
            21: [22, 23],
            22: [40],
            23: [24, 27],
            24: [25, 32],
            25: [26, 31],
            26: [33],
            27: [28],
            28: [29, 32],
            29: [30],
            30: [31, 33],
            31: [32],
            34: [35, 38],
            35: [36],
            36: [37, 39],
            37: [38],
            40: [41, 44],
            41: [42],
            42: [43, 45],
            43: [44],
        },
        create_using=create_using,
    )
    G.name = "Tutte's Graph"
    return G
<|endoftext|>"""
Ego graph.
"""

__all__ = ["ego_graph"]

import networkx as nx


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def ego_graph(G, n, radius=1, center=True, undirected=False, distance=None):
    """Returns induced subgraph of neighbors centered at node n within
    a given radius.

    Parameters
    ----------
    G : graph
      A NetworkX Graph or DiGraph

    n : node
      A single node

    radius : number, optional
      Include all neighbors of distance<=radius from n.

    center : bool, optional
      If False, do not include center node in graph

    undirected : bool, optional
      If True use both in- and out-neighbors of directed graphs.

    distance : key, optional
      Use specified edge data key as distance.  For example, setting
      distance='weight' will use the edge weight to measure the
      distance from the node n.

    Notes
    -----
    For directed graphs D this produces the "out" neighborhood
    or successors.  If you want the neighborhood of predecessors
    first reverse the graph with D.reverse().  If you want both
    directions use the keyword argument undirected=True.

    Node, edge, and graph attributes are copied to the returned subgraph.
    """
    if undirected:
        if distance is not None:
            sp, _ = nx.single_source_dijkstra(
                G.to_undirected(), n, cutoff=radius, weight=distance
            )
        else:
            sp = dict(
                nx.single_source_shortest_path_length(
                    G.to_undirected(), n, cutoff=radius
                )
            )
    else:
        if distance is not None:
            sp, _ = nx.single_source_dijkstra(G, n, cutoff=radius, weight=distance)
        else:
            sp = dict(nx.single_source_shortest_path_length(G, n, cutoff=radius))

    H = G.subgraph(sp).copy()
    if not center:
        H.remove_node(n)
    return H
<|endoftext|>"""View of Graphs as SubGraph, Reverse, Directed, Undirected.

In some algorithms it is convenient to temporarily morph
a graph to exclude some nodes or edges. It should be better
to do that via a view than to remove and then re-add.
In other algorithms it is convenient to temporarily morph
a graph to reverse directed edges, or treat a directed graph
as undirected, etc. This module provides those graph views.

The resulting views are essentially read-only graphs that
report data from the original graph object. We provide an
attribute G._graph which points to the underlying graph object.

Note: Since graphviews look like graphs, one can end up with
view-of-view-of-view chains. Be careful with chains because
they become very slow with about 15 nested views.
For the common simple case of node induced subgraphs created
from the graph class, we short-cut the chain by returning a
subgraph of the original graph directly rather than a subgraph
of a subgraph. We are careful not to disrupt any edge filter in
the middle subgraph. In general, determining how to short-cut
the chain is tricky and much harder with restricted_views than
with induced subgraphs.
Often it is easiest to use .copy() to avoid chains.
"""

import networkx as nx
from networkx.classes.coreviews import (
    FilterAdjacency,
    FilterAtlas,
    FilterMultiAdjacency,
    UnionAdjacency,
    UnionMultiAdjacency,
)
from networkx.classes.filters import no_filter
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for

__all__ = ["generic_graph_view", "subgraph_view", "reverse_view"]


def generic_graph_view(G, create_using=None):
    """Returns a read-only view of `G`.

    The graph `G` and its attributes are not copied but viewed through the new graph object
    of the same class as `G` (or of the class specified in `create_using`).

    Parameters
    ----------
    G : graph
        A directed/undirected graph/multigraph.

    create_using : NetworkX graph constructor, optional (default=None)
       Graph type to create. If graph instance, then cleared before populated.
       If `None`, then the appropriate Graph type is inferred from `G`.

    Returns
    -------
    newG : graph
        A view of the input graph `G` and its attributes as viewed through
        the `create_using` class.

    Raises
    ------
    NetworkXError
        If `G` is a multigraph (or multidigraph) but `create_using` is not, or vice versa.

    Notes
    -----
    The returned graph view is read-only (cannot modify the graph).
    Yet the view reflects any changes in `G`. The intent is to mimic dict views.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=0.3)
    >>> G.add_edge(2, 3, weight=0.5)
    >>> G.edges(data=True)
    EdgeDataView([(1, 2, {'weight': 0.3}), (2, 3, {'weight': 0.5})])

    The view exposes the attributes from the original graph.

    >>> viewG = nx.graphviews.generic_graph_view(G)
    >>> viewG.edges(data=True)
    EdgeDataView([(1, 2, {'weight': 0.3}), (2, 3, {'weight': 0.5})])

    Changes to `G` are reflected in `viewG`.

    >>> G.remove_edge(2, 3)
    >>> G.edges(data=True)
    EdgeDataView([(1, 2, {'weight': 0.3})])

    >>> viewG.edges(data=True)
    EdgeDataView([(1, 2, {'weight': 0.3})])

    We can change the graph type with the `create_using` parameter.

    >>> type(G)
    <class 'networkx.classes.graph.Graph'>
    >>> viewDG = nx.graphviews.generic_graph_view(G, create_using=nx.DiGraph)
    >>> type(viewDG)
    <class 'networkx.classes.digraph.DiGraph'>
    """
    if create_using is None:
        newG = G.__class__()
    else:
        newG = nx.empty_graph(0, create_using)
    if G.is_multigraph() != newG.is_multigraph():
        raise NetworkXError("Multigraph for G must agree with create_using")
    newG = nx.freeze(newG)

    # create view by assigning attributes from G
    newG._graph = G
    newG.graph = G.graph

    newG._node = G._node
    if newG.is_directed():
        if G.is_directed():
            newG._succ = G._succ
            newG._pred = G._pred
            # newG._adj is synced with _succ
        else:
            newG._succ = G._adj
            newG._pred = G._adj
            # newG._adj is synced with _succ
    elif G.is_directed():
        if G.is_multigraph():
            newG._adj = UnionMultiAdjacency(G._succ, G._pred)
        else:
            newG._adj = UnionAdjacency(G._succ, G._pred)
    else:
        newG._adj = G._adj
    return newG


def subgraph_view(G, *, filter_node=no_filter, filter_edge=no_filter):
    """View of `G` applying a filter on nodes and edges.

    `subgraph_view` provides a read-only view of the input graph that excludes
    nodes and edges based on the outcome of two filter functions `filter_node`
    and `filter_edge`.

    The `filter_node` function takes one argument --- the node --- and returns
    `True` if the node should be included in the subgraph, and `False` if it
    should not be included.

    The `filter_edge` function takes two (or three arguments if `G` is a
    multi-graph) --- the nodes describing an edge, plus the edge-key if
    parallel edges are possible --- and returns `True` if the edge should be
    included in the subgraph, and `False` if it should not be included.

    Both node and edge filter functions are called on graph elements as they
    are queried, meaning there is no up-front cost to creating the view.

    Parameters
    ----------
    G : networkx.Graph
        A directed/undirected graph/multigraph

    filter_node : callable, optional
        A function taking a node as input, which returns `True` if the node
        should appear in the view.

    filter_edge : callable, optional
        A function taking as input the two nodes describing an edge (plus the
        edge-key if `G` is a multi-graph), which returns `True` if the edge
        should appear in the view.

    Returns
    -------
    graph : networkx.Graph
        A read-only graph view of the input graph.

    Examples
    --------
    >>> G = nx.path_graph(6)

    Filter functions operate on the node, and return `True` if the node should
    appear in the view:

    >>> def filter_node(n1):
    ...     return n1 != 5
    >>> view = nx.subgraph_view(G, filter_node=filter_node)
    >>> view.nodes()
    NodeView((0, 1, 2, 3, 4))

    We can use a closure pattern to filter graph elements based on additional
    data --- for example, filtering on edge data attached to the graph:

    >>> G[3][4]["cross_me"] = False
    >>> def filter_edge(n1, n2):
    ...     return G[n1][n2].get("cross_me", True)
    >>> view = nx.subgraph_view(G, filter_edge=filter_edge)
    >>> view.edges()
    EdgeView([(0, 1), (1, 2), (2, 3), (4, 5)])

    >>> view = nx.subgraph_view(
    ...     G,
    ...     filter_node=filter_node,
    ...     filter_edge=filter_edge,
    ... )
    >>> view.nodes()
    NodeView((0, 1, 2, 3, 4))
    >>> view.edges()
    EdgeView([(0, 1), (1, 2), (2, 3)])
    """
    newG = nx.freeze(G.__class__())
    newG._NODE_OK = filter_node
    newG._EDGE_OK = filter_edge

    # create view by assigning attributes from G
    newG._graph = G
    newG.graph = G.graph

    newG._node = FilterAtlas(G._node, filter_node)
    if G.is_multigraph():
        Adj = FilterMultiAdjacency

        def reverse_edge(u, v, k=None):
            return filter_edge(v, u, k)

    else:
        Adj = FilterAdjacency

        def reverse_edge(u, v, k=None):
            return filter_edge(v, u)

    if G.is_directed():
        newG._succ = Adj(G._succ, filter_node, filter_edge)
        newG._pred = Adj(G._pred, filter_node, reverse_edge)
        # newG._adj is synced with _succ
    else:
        newG._adj = Adj(G._adj, filter_node, filter_edge)
    return newG


@not_implemented_for("undirected")
def reverse_view(G):
    """View of `G` with edge directions reversed

    `reverse_view` returns a read-only view of the input graph where
    edge directions are reversed.

    Identical to digraph.reverse(copy=False)

    Parameters
    ----------
    G : networkx.DiGraph

    Returns
    -------
    graph : networkx.DiGraph

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2)
    >>> G.add_edge(2, 3)
    >>> G.edges()
    OutEdgeView([(1, 2), (2, 3)])

    >>> view = nx.reverse_view(G)
    >>> view.edges()
    OutEdgeView([(2, 1), (3, 2)])
    """
    newG = generic_graph_view(G)
    newG._succ, newG._pred = G._pred, G._succ
    # newG._adj is synced with _succ
    return newG
<|endoftext|>"""Base class for directed graphs."""

from copy import deepcopy
from functools import cached_property

import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.graph import Graph
from networkx.classes.reportviews import (
    DiDegreeView,
    InDegreeView,
    InEdgeView,
    OutDegreeView,
    OutEdgeView,
)
from networkx.exception import NetworkXError

__all__ = ["DiGraph"]


class _CachedPropertyResetterAdjAndSucc:
    """Data Descriptor class that syncs and resets cached properties adj and succ

    The cached properties `adj` and `succ` are reset whenever `_adj` or `_succ`
    are set to new objects. In addition, the attributes `_succ` and `_adj`
    are synced so these two names point to the same object.

    Warning: most of the time, when ``G._adj`` is set, ``G._pred`` should also
    be set to maintain a valid data structure. They share datadicts.

    This object sits on a class and ensures that any instance of that
    class clears its cached properties "succ" and "adj" whenever the
    underlying instance attributes "_succ" or "_adj" are set to a new object.
    It only affects the set process of the obj._adj and obj._succ attribute.
    All get/del operations act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """

    def __set__(self, obj, value):
        od = obj.__dict__
        od["_adj"] = value
        od["_succ"] = value
        # reset cached properties
        props = [
            "adj",
            "succ",
            "edges",
            "out_edges",
            "degree",
            "out_degree",
            "in_degree",
        ]
        for prop in props:
            if prop in od:
                del od[prop]


class _CachedPropertyResetterPred:
    """Data Descriptor class for _pred that resets ``pred`` cached_property when needed

    This assumes that the ``cached_property`` ``G.pred`` should be reset whenever
    ``G._pred`` is set to a new value.

    Warning: most of the time, when ``G._pred`` is set, ``G._adj`` should also
    be set to maintain a valid data structure. They share datadicts.

    This object sits on a class and ensures that any instance of that
    class clears its cached property "pred" whenever the underlying
    instance attribute "_pred" is set to a new object. It only affects
    the set process of the obj._pred attribute. All get/del operations
    act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """

    def __set__(self, obj, value):
        od = obj.__dict__
        od["_pred"] = value
        # reset cached properties
        props = ["pred", "in_edges", "degree", "out_degree", "in_degree"]
        for prop in props:
            if prop in od:
                del od[prop]


class DiGraph(Graph):
    """
    Base class for directed graphs.

    A DiGraph stores nodes and edges with optional data, or attributes.

    DiGraphs hold directed edges.  Self loops are allowed but multiple
    (parallel) edges are not.

    Nodes can be arbitrary (hashable) Python objects with optional
    key/value attributes. By convention `None` is not used as a node.

    Edges are represented as links between nodes with optional
    key/value attributes.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    Graph
    MultiGraph
    MultiDiGraph

    Examples
    --------
    Create an empty graph structure (a "null graph") with no nodes and
    no edges.

    >>> G = nx.DiGraph()

    G can be grown in several ways.

    **Nodes:**

    Add one node at a time:

    >>> G.add_node(1)

    Add the nodes from any container (a list, dict, set or
    even the lines from a file or the nodes from another graph).

    >>> G.add_nodes_from([2, 3])
    >>> G.add_nodes_from(range(100, 110))
    >>> H = nx.path_graph(10)
    >>> G.add_nodes_from(H)

    In addition to strings and integers any hashable Python object
    (except None) can represent a node, e.g. a customized node object,
    or even another Graph.

    >>> G.add_node(H)

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge(1, 2)

    a list of edges,

    >>> G.add_edges_from([(1, 2), (1, 3)])

    or a collection of edges,

    >>> G.add_edges_from(H.edges)

    If some edges connect nodes not yet in the graph, the nodes
    are added automatically.  There are no errors when adding
    nodes or edges that already exist.

    **Attributes:**

    Each graph, node, and edge can hold key/value attribute pairs
    in an associated attribute dictionary (the keys must be hashable).
    By default these are empty, but can be added or changed using
    add_edge, add_node or direct manipulation of the attribute
    dictionaries named graph, node and edge respectively.

    >>> G = nx.DiGraph(day="Friday")
    >>> G.graph
    {'day': 'Friday'}

    Add node attributes using add_node(), add_nodes_from() or G.nodes

    >>> G.add_node(1, time="5pm")
    >>> G.add_nodes_from([3], time="2pm")
    >>> G.nodes[1]
    {'time': '5pm'}
    >>> G.nodes[1]["room"] = 714
    >>> del G.nodes[1]["room"]  # remove attribute
    >>> list(G.nodes(data=True))
    [(1, {'time': '5pm'}), (3, {'time': '2pm'})]

    Add edge attributes using add_edge(), add_edges_from(), subscript
    notation, or G.edges.

    >>> G.add_edge(1, 2, weight=4.7)
    >>> G.add_edges_from([(3, 4), (4, 5)], color="red")
    >>> G.add_edges_from([(1, 2, {"color": "blue"}), (2, 3, {"weight": 8})])
    >>> G[1][2]["weight"] = 4.7
    >>> G.edges[1, 2]["weight"] = 4

    Warning: we protect the graph data structure by making `G.edges[1, 2]` a
    read-only dict-like structure. However, you can assign to attributes
    in e.g. `G.edges[1, 2]`. Thus, use 2 sets of brackets to add/change
    data attributes: `G.edges[1, 2]['weight'] = 4`
    (For multigraphs: `MG.edges[u, v, key][name] = value`).

    **Shortcuts:**

    Many common graph features allow python syntax to speed reporting.

    >>> 1 in G  # check if node in graph
    True
    >>> [n for n in G if n < 3]  # iterate through nodes
    [1, 2]
    >>> len(G)  # number of nodes in graph
    5

    Often the best way to traverse all edges of a graph is via the neighbors.
    The neighbors are reported as an adjacency-dict `G.adj` or `G.adjacency()`

    >>> for n, nbrsdict in G.adjacency():
    ...     for nbr, eattr in nbrsdict.items():
    ...         if "weight" in eattr:
    ...             # Do something useful with the edges
    ...             pass

    But the edges reporting object is often more convenient:

    >>> for u, v, weight in G.edges(data="weight"):
    ...     if weight is not None:
    ...         # Do something useful with the edges
    ...         pass

    **Reporting:**

    Simple graph information is obtained using object-attributes and methods.
    Reporting usually provides views instead of containers to reduce memory
    usage. The views update as the graph is updated similarly to dict-views.
    The objects `nodes`, `edges` and `adj` provide access to data attributes
    via lookup (e.g. `nodes[n]`, `edges[u, v]`, `adj[u][v]`) and iteration
    (e.g. `nodes.items()`, `nodes.data('color')`,
    `nodes.data('color', default='blue')` and similarly for `edges`)
    Views exist for `nodes`, `edges`, `neighbors()`/`adj` and `degree`.

    For details on these and other miscellaneous methods, see below.

    **Subclasses (Advanced):**

    The Graph class uses a dict-of-dict-of-dict data structure.
    The outer dict (node_dict) holds adjacency information keyed by node.
    The next dict (adjlist_dict) represents the adjacency information and holds
    edge data keyed by neighbor.  The inner dict (edge_attr_dict) represents
    the edge data and holds edge attribute values keyed by attribute names.

    Each of these three dicts can be replaced in a subclass by a user defined
    dict-like object. In general, the dict-like features should be
    maintained but extra features can be added. To replace one of the
    dicts create a new graph class by changing the class(!) variable
    holding the factory for that dict-like structure. The variable names are
    node_dict_factory, node_attr_dict_factory, adjlist_inner_dict_factory,
    adjlist_outer_dict_factory, edge_attr_dict_factory and graph_attr_dict_factory.

    node_dict_factory : function, (default: dict)
        Factory function to be used to create the dict containing node
        attributes, keyed by node id.
        It should require no arguments and return a dict-like object

    node_attr_dict_factory: function, (default: dict)
        Factory function to be used to create the node attribute
        dict which holds attribute values keyed by attribute name.
        It should require no arguments and return a dict-like object

    adjlist_outer_dict_factory : function, (default: dict)
        Factory function to be used to create the outer-most dict
        in the data structure that holds adjacency info keyed by node.
        It should require no arguments and return a dict-like object.

    adjlist_inner_dict_factory : function, optional (default: dict)
        Factory function to be used to create the adjacency list
        dict which holds edge data keyed by neighbor.
        It should require no arguments and return a dict-like object

    edge_attr_dict_factory : function, optional (default: dict)
        Factory function to be used to create the edge attribute
        dict which holds attribute values keyed by attribute name.
        It should require no arguments and return a dict-like object.

    graph_attr_dict_factory : function, (default: dict)
        Factory function to be used to create the graph attribute
        dict which holds attribute values keyed by attribute name.
        It should require no arguments and return a dict-like object.

    Typically, if your extension doesn't impact the data structure all
    methods will inherited without issue except: `to_directed/to_undirected`.
    By default these methods create a DiGraph/Graph class and you probably
    want them to create your extension of a DiGraph/Graph. To facilitate
    this we define two class variables that you can set in your subclass.

    to_directed_class : callable, (default: DiGraph or MultiDiGraph)
        Class to create a new graph structure in the `to_directed` method.
        If `None`, a NetworkX class (DiGraph or MultiDiGraph) is used.

    to_undirected_class : callable, (default: Graph or MultiGraph)
        Class to create a new graph structure in the `to_undirected` method.
        If `None`, a NetworkX class (Graph or MultiGraph) is used.

    **Subclassing Example**

    Create a low memory graph class that effectively disallows edge
    attributes by using a single attribute dict for all edges.
    This reduces the memory used, but you lose edge attributes.

    >>> class ThinGraph(nx.Graph):
    ...     all_edge_dict = {"weight": 1}
    ...
    ...     def single_edge_dict(self):
    ...         return self.all_edge_dict
    ...
    ...     edge_attr_dict_factory = single_edge_dict
    >>> G = ThinGraph()
    >>> G.add_edge(2, 1)
    >>> G[2][1]
    {'weight': 1}
    >>> G.add_edge(2, 2)
    >>> G[2][1] is G[2][2]
    True
    """

    _adj = _CachedPropertyResetterAdjAndSucc()  # type: ignore[assignment]
    _succ = _adj  # type: ignore[has-type]
    _pred = _CachedPropertyResetterPred()

    def __init__(self, incoming_graph_data=None, **attr):
        """Initialize a graph with edges, name, or graph attributes.

        Parameters
        ----------
        incoming_graph_data : input graph (optional, default: None)
            Data to initialize graph.  If None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object.  If the corresponding optional Python
            packages are installed the data can also be a 2D NumPy array, a
            SciPy sparse array, or a PyGraphviz graph.

        attr : keyword arguments, optional (default= no attributes)
            Attributes to add to graph as key=value pairs.

        See Also
        --------
        convert

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G = nx.Graph(name="my graph")
        >>> e = [(1, 2), (2, 3), (3, 4)]  # list of edges
        >>> G = nx.Graph(e)

        Arbitrary graph attribute pairs (key=value) may be assigned

        >>> G = nx.Graph(e, day="Friday")
        >>> G.graph
        {'day': 'Friday'}

        """
        self.graph = self.graph_attr_dict_factory()  # dictionary for graph attributes
        self._node = self.node_dict_factory()  # dictionary for node attr
        # We store two adjacency lists:
        # the predecessors of node n are stored in the dict self._pred
        # the successors of node n are stored in the dict self._succ=self._adj
        self._adj = self.adjlist_outer_dict_factory()  # empty adjacency dict successor
        self._pred = self.adjlist_outer_dict_factory()  # predecessor
        # Note: self._succ = self._adj  # successor

        self.__networkx_cache__ = {}
        # attempt to load graph with data
        if incoming_graph_data is not None:
            convert.to_networkx_graph(incoming_graph_data, create_using=self)
        # load graph attributes (must be after convert)
        self.graph.update(attr)

    @cached_property
    def adj(self):
        """Graph adjacency object holding the neighbors of each node.

        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edge-data-dict.  So `G.adj[3][2]['color'] = 'blue'` sets
        the color of the edge `(3, 2)` to `"blue"`.

        Iterating over G.adj behaves like a dict. Useful idioms include
        `for nbr, datadict in G.adj[n].items():`.

        The neighbor information is also provided by subscripting the graph.
        So `for nbr, foovalue in G[node].data('foo', default=1):` works.

        For directed graphs, `G.adj` holds outgoing (successor) info.
        """
        return AdjacencyView(self._succ)

    @cached_property
    def succ(self):
        """Graph adjacency object holding the successors of each node.

        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edge-data-dict.  So `G.succ[3][2]['color'] = 'blue'` sets
        the color of the edge `(3, 2)` to `"blue"`.

        Iterating over G.succ behaves like a dict. Useful idioms include
        `for nbr, datadict in G.succ[n].items():`.  A data-view not provided
        by dicts also exists: `for nbr, foovalue in G.succ[node].data('foo'):`
        and a default can be set via a `default` argument to the `data` method.

        The neighbor information is also provided by subscripting the graph.
        So `for nbr, foovalue in G[node].data('foo', default=1):` works.

        For directed graphs, `G.adj` is identical to `G.succ`.
        """
        return AdjacencyView(self._succ)

    @cached_property
    def pred(self):
        """Graph adjacency object holding the predecessors of each node.

        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edge-data-dict.  So `G.pred[2][3]['color'] = 'blue'` sets
        the color of the edge `(3, 2)` to `"blue"`.

        Iterating over G.pred behaves like a dict. Useful idioms include
        `for nbr, datadict in G.pred[n].items():`.  A data-view not provided
        by dicts also exists: `for nbr, foovalue in G.pred[node].data('foo'):`
        A default can be set via a `default` argument to the `data` method.
        """
        return AdjacencyView(self._pred)

    def add_node(self, node_for_adding, **attr):
        """Add a single node `node_for_adding` and update node attributes.

        Parameters
        ----------
        node_for_adding : node
            A node can be any hashable Python object except None.
        attr : keyword arguments, optional
            Set or change node attributes using key=value.

        See Also
        --------
        add_nodes_from

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_node(1)
        >>> G.add_node("Hello")
        >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> G.add_node(K3)
        >>> G.number_of_nodes()
        3

        Use keywords set/change node attributes:

        >>> G.add_node(1, size=10)
        >>> G.add_node(3, weight=0.4, UTM=("13S", 382871, 3972649))

        Notes
        -----
        A hashable object is one that can be used as a key in a Python
        dictionary. This includes strings, numbers, tuples of strings
        and numbers, etc.

        On many platforms hashable items also include mutables such as
        NetworkX Graphs, though one should be careful that the hash
        doesn't change on mutables.
        """
        if node_for_adding not in self._succ:
            if node_for_adding is None:
                raise ValueError("None cannot be a node")
            self._succ[node_for_adding] = self.adjlist_inner_dict_factory()
            self._pred[node_for_adding] = self.adjlist_inner_dict_factory()
            attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
            attr_dict.update(attr)
        else:  # update attr even if node already exists
            self._node[node_for_adding].update(attr)
        nx._clear_cache(self)

    def add_nodes_from(self, nodes_for_adding, **attr):
        """Add multiple nodes.

        Parameters
        ----------
        nodes_for_adding : iterable container
            A container of nodes (list, dict, set, etc.).
            OR
            A container of (node, attribute dict) tuples.
            Node attributes are updated using the attribute dict.
        attr : keyword arguments, optional (default= no attributes)
            Update attributes for all nodes in nodes.
            Node attributes specified in nodes as a tuple take
            precedence over attributes specified via keyword arguments.

        See Also
        --------
        add_node

        Notes
        -----
        When adding nodes from an iterator over the graph you are changing,
        a `RuntimeError` can be raised with message:
        `RuntimeError: dictionary changed size during iteration`. This
        happens when the graph's underlying dictionary is modified during
        iteration. To avoid this error, evaluate the iterator into a separate
        object, e.g. by using `list(iterator_of_nodes)`, and pass this
        object to `G.add_nodes_from`.

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_nodes_from("Hello")
        >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> G.add_nodes_from(K3)
        >>> sorted(G.nodes(), key=str)
        [0, 1, 2, 'H', 'e', 'l', 'o']

        Use keywords to update specific node attributes for every node.

        >>> G.add_nodes_from([1, 2], size=10)
        >>> G.add_nodes_from([3, 4], weight=0.4)

        Use (node, attrdict) tuples to update attributes for specific nodes.

        >>> G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])
        >>> G.nodes[1]["size"]
        11
        >>> H = nx.Graph()
        >>> H.add_nodes_from(G.nodes(data=True))
        >>> H.nodes[1]["size"]
        11

        Evaluate an iterator over a graph if using it to modify the same graph

        >>> G = nx.DiGraph([(0, 1), (1, 2), (3, 4)])
        >>> # wrong way - will raise RuntimeError
        >>> # G.add_nodes_from(n + 1 for n in G.nodes)
        >>> # correct way
        >>> G.add_nodes_from(list(n + 1 for n in G.nodes))
        """
        for n in nodes_for_adding:
            try:
                newnode = n not in self._node
                newdict = attr
            except TypeError:
                n, ndict = n
                newnode = n not in self._node
                newdict = attr.copy()
                newdict.update(ndict)
            if newnode:
                if n is None:
                    raise ValueError("None cannot be a node")
                self._succ[n] = self.adjlist_inner_dict_factory()
                self._pred[n] = self.adjlist_inner_dict_factory()
                self._node[n] = self.node_attr_dict_factory()
            self._node[n].update(newdict)
        nx._clear_cache(self)

    def remove_node(self, n):
        """Remove node n.

        Removes the node n and all adjacent edges.
        Attempting to remove a nonexistent node will raise an exception.

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

        Examples
        --------
        >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> list(G.edges)
        [(0, 1), (1, 2)]
        >>> G.remove_node(1)
        >>> list(G.edges)
        []

        """
        try:
            nbrs = self._succ[n]
            del self._node[n]
        except KeyError as err:  # NetworkXError if n not in self
            raise NetworkXError(f"The node {n} is not in the digraph.") from err
        for u in nbrs:
            del self._pred[u][n]  # remove all edges n-u in digraph
        del self._succ[n]  # remove node from succ
        for u in self._pred[n]:
            del self._succ[u][n]  # remove all edges n-u in digraph
        del self._pred[n]  # remove node from pred
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

        Examples
        --------
        >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> e = list(G.nodes)
        >>> e
        [0, 1, 2]
        >>> G.remove_nodes_from(e)
        >>> list(G.nodes)
        []

        Evaluate an iterator over a graph if using it to modify the same graph

        >>> G = nx.DiGraph([(0, 1), (1, 2), (3, 4)])
        >>> # this command will fail, as the graph's dict is modified during iteration
        >>> # G.remove_nodes_from(n for n in G.nodes if n < 2)
        >>> # this command will work, since the dictionary underlying graph is not modified
        >>> G.remove_nodes_from(list(n for n in G.nodes if n < 2))
        """
        for n in nodes:
            try:
                succs = self._succ[n]
                del self._node[n]
                for u in succs:
                    del self._pred[u][n]  # remove all edges n-u in digraph
                del self._succ[n]  # now remove node
                for u in self._pred[n]:
                    del self._succ[u][n]  # remove all edges n-u in digraph
                del self._pred[n]  # now remove node
            except KeyError:
                pass  # silent failure on remove
        nx._clear_cache(self)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Edge attributes can be specified with keywords or by directly
        accessing the edge's attribute dictionary. See examples below.

        Parameters
        ----------
        u_of_edge, v_of_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        See Also
        --------
        add_edges_from : add a collection of edges

        Notes
        -----
        Adding an edge that already exists updates the edge data.

        Many NetworkX algorithms designed for weighted graphs use
        an edge attribute (by default `weight`) to hold a numerical value.

        Examples
        --------
        The following all add the edge e=(1, 2) to graph G:

        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> e = (1, 2)
        >>> G.add_edge(1, 2)  # explicit two-node form
        >>> G.add_edge(*e)  # single edge as tuple of two nodes
        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container

        Associate data to edges using keywords:

        >>> G.add_edge(1, 2, weight=3)
        >>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)

        For non-string attribute keys, use subscript notation.

        >>> G.add_edge(1, 2)
        >>> G[1][2].update({0: 5})
        >>> G.edges[1, 2].update({0: 5})
        """
        u, v = u_of_edge, v_of_edge
        # add nodes
        if u not in self._succ:
            if u is None:
                raise ValueError("None cannot be a node")
            self._succ[u] = self.adjlist_inner_dict_factory()
            self._pred[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._succ:
            if v is None:
                raise ValueError("None cannot be a node")
            self._succ[v] = self.adjlist_inner_dict_factory()
            self._pred[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        # add the edge
        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
        datadict.update(attr)
        self._succ[u][v] = datadict
        self._pred[v][u] = datadict
        nx._clear_cache(self)

    def add_edges_from(self, ebunch_to_add, **attr):
        """Add all the edges in ebunch_to_add.

        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the container will be added to the
            graph. The edges must be given as 2-tuples (u, v) or
            3-tuples (u, v, d) where d is a dictionary containing edge data.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        See Also
        --------
        add_edge : add a single edge
        add_weighted_edges_from : convenient way to add weighted edges

        Notes
        -----
        Adding the same edge twice has no effect but any edge data
        will be updated when each duplicate edge is added.

        Edge attributes specified in an ebunch take precedence over
        attributes specified via keyword arguments.

        When adding edges from an iterator over the graph you are changing,
        a `RuntimeError` can be raised with message:
        `RuntimeError: dictionary changed size during iteration`. This
        happens when the graph's underlying dictionary is modified during
        iteration. To avoid this error, evaluate the iterator into a separate
        object, e.g. by using `list(iterator_of_edges)`, and pass this
        object to `G.add_edges_from`.

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edges_from([(0, 1), (1, 2)])  # using a list of edge tuples
        >>> e = zip(range(0, 3), range(1, 4))
        >>> G.add_edges_from(e)  # Add the path graph 0-1-2-3

        Associate data to edges

        >>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
        >>> G.add_edges_from([(3, 4), (1, 4)], label="WN2898")

        Evaluate an iterator over a graph if using it to modify the same graph

        >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
        >>> # Grow graph by one new node, adding edges to all existing nodes.
        >>> # wrong way - will raise RuntimeError
        >>> # G.add_edges_from(((5, n) for n in G.nodes))
        >>> # right way - note that there will be no self-edge for node 5
        >>> G.add_edges_from(list((5, n) for n in G.nodes))
        """
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}
            else:
                raise NetworkXError(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
            if u not in self._succ:
                if u is None:
                    raise ValueError("None cannot be a node")
                self._succ[u] = self.adjlist_inner_dict_factory()
                self._pred[u] = self.adjlist_inner_dict_factory()
                self._node[u] = self.node_attr_dict_factory()
            if v not in self._succ:
                if v is None:
                    raise ValueError("None cannot be a node")
                self._succ[v] = self.adjlist_inner_dict_factory()
                self._pred[v] = self.adjlist_inner_dict_factory()
                self._node[v] = self.node_attr_dict_factory()
            datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
            datadict.update(attr)
            datadict.update(dd)
            self._succ[u][v] = datadict
            self._pred[v][u] = datadict
        nx._clear_cache(self)

    def remove_edge(self, u, v):
        """Remove the edge between u and v.

        Parameters
        ----------
        u, v : nodes
            Remove the edge between nodes u and v.

        Raises
        ------
        NetworkXError
            If there is not an edge between u and v.

        See Also
        --------
        remove_edges_from : remove a collection of edges

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, etc
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.remove_edge(0, 1)
        >>> e = (1, 2)
        >>> G.remove_edge(*e)  # unpacks e from an edge tuple
        >>> e = (2, 3, {"weight": 7})  # an edge with attribute data
        >>> G.remove_edge(*e[:2])  # select first part of edge tuple
        """
        try:
            del self._succ[u][v]
            del self._pred[v][u]
        except KeyError as err:
            raise NetworkXError(f"The edge {u}-{v} not in graph.") from err
        nx._clear_cache(self)

    def remove_edges_from(self, ebunch):
        """Remove all edges specified in ebunch.

        Parameters
        ----------
        ebunch: list or container of edge tuples
            Each edge given in the list or container will be removed
            from the graph. The edges can be:

                - 2-tuples (u, v) edge between u and v.
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
            if u in self._succ and v in self._succ[u]:
                del self._succ[u][v]
                del self._pred[v][u]
        nx._clear_cache(self)

    def has_successor(self, u, v):
        """Returns True if node u has successor v.

        This is true if graph has the edge u->v.
        """
        return u in self._succ and v in self._succ[u]

    def has_predecessor(self, u, v):
        """Returns True if node u has predecessor v.

        This is true if graph has the edge u<-v.
        """
        return u in self._pred and v in self._pred[u]

    def successors(self, n):
        """Returns an iterator over successor nodes of n.

        A successor of n is a node m such that there exists a directed
        edge from n to m.

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
        predecessors

        Notes
        -----
        neighbors() and successors() are the same.
        """
        try:
            return iter(self._succ[n])
        except KeyError as err:
            raise NetworkXError(f"The node {n} is not in the digraph.") from err

    # digraph definitions
    neighbors = successors

    def predecessors(self, n):
        """Returns an iterator over predecessor nodes of n.

        A predecessor of n is a node m such that there exists a directed
        edge from m to n.

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
        successors
        """
        try:
            return iter(self._pred[n])
        except KeyError as err:
            raise NetworkXError(f"The node {n} is not in the digraph.") from err

    @cached_property
    def edges(self):
        """An OutEdgeView of the DiGraph as G.edges or G.edges().

        edges(self, nbunch=None, data=False, default=None)

        The OutEdgeView provides set-like operations on the edge-tuples
        as well as edge attribute lookup. When called, it also provides
        an EdgeDataView object which allows control of access to edge
        attributes (but does not provide set-like operations).
        Hence, `G.edges[u, v]['color']` provides the value of the color
        attribute for edge `(u, v)` while
        `for (u, v, c) in G.edges.data('color', default='red'):`
        iterates through all the edges yielding the color attribute
        with default `'red'` if no color attribute exists.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges from these nodes.
        data : string or bool, optional (default=False)
            The edge attribute returned in 3-tuple (u, v, ddict[data]).
            If True, return edge attribute dict in 3-tuple (u, v, ddict).
            If False, return 2-tuple (u, v).
        default : value, optional (default=None)
            Value used for edges that don't have the requested attribute.
            Only relevant if data is not True or False.

        Returns
        -------
        edges : OutEdgeView
            A view of edge attributes, usually it iterates over (u, v)
            or (u, v, d) tuples of edges, but can also be used for
            attribute lookup as `edges[u, v]['foo']`.

        See Also
        --------
        in_edges, out_edges

        Notes
        -----
        Nodes in nbunch that are not in the graph will be (quietly) ignored.
        For directed graphs this returns the out-edges.

        Examples
        --------
        >>> G = nx.DiGraph()  # or MultiDiGraph, etc
        >>> nx.add_path(G, [0, 1, 2])
        >>> G.add_edge(2, 3, weight=5)
        >>> [e for e in G.edges]
        [(0, 1), (1, 2), (2, 3)]
        >>> G.edges.data()  # default data is {} (empty dict)
        OutEdgeDataView([(0, 1, {}), (1, 2, {}), (2, 3, {'weight': 5})])
        >>> G.edges.data("weight", default=1)
        OutEdgeDataView([(0, 1, 1), (1, 2, 1), (2, 3, 5)])
        >>> G.edges([0, 2])  # only edges originating from these nodes
        OutEdgeDataView([(0, 1), (2, 3)])
        >>> G.edges(0)  # only edges from node 0
        OutEdgeDataView([(0, 1)])

        """
        return OutEdgeView(self)

    # alias out_edges to edges
    @cached_property
    def out_edges(self):
        return OutEdgeView(self)

    out_edges.__doc__ = edges.__doc__

    @cached_property
    def in_edges(self):
        """A view of the in edges of the graph as G.in_edges or G.in_edges().

        in_edges(self, nbunch=None, data=False, default=None):

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.
        data : string or bool, optional (default=False)
            The edge attribute returned in 3-tuple (u, v, ddict[data]).
            If True, return edge attribute dict in 3-tuple (u, v, ddict).
            If False, return 2-tuple (u, v).
        default : value, optional (default=None)
            Value used for edges that don't have the requested attribute.
            Only relevant if data is not True or False.

        Returns
        -------
        in_edges : InEdgeView or InEdgeDataView
            A view of edge attributes, usually it iterates over (u, v)
            or (u, v, d) tuples of edges, but can also be used for
            attribute lookup as `edges[u, v]['foo']`.

        Examples
        --------
        >>> G = nx.DiGraph()
        >>> G.add_edge(1, 2, color="blue")
        >>> G.in_edges()
        InEdgeView([(1, 2)])
        >>> G.in_edges(nbunch=2)
        InEdgeDataView([(1, 2)])

        See Also
        --------
        edges
        """
        return InEdgeView(self)

    @cached_property
    def degree(self):
        """A DegreeView for the Graph as G.degree or G.degree().

        The node degree is the number of edges adjacent to the node.
        The weighted node degree is the sum of the edge weights for
        edges incident to that node.

        This object provides an iterator for (node, degree) as well as
        lookup for the degree for a single node.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.

        weight : string or None, optional (default=None)
           The name of an edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights adjacent to the node.

        Returns
        -------
        DiDegreeView or int
            If multiple nodes are requested (the default), returns a `DiDegreeView`
            mapping nodes to their degree.
            If a single node is requested, returns the degree of the node as an integer.

        See Also
        --------
        in_degree, out_degree

        Examples
        --------
        >>> G = nx.DiGraph()  # or MultiDiGraph
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.degree(0)  # node 0 with degree 1
        1
        >>> list(G.degree([0, 1, 2]))
        [(0, 1), (1, 2), (2, 2)]

        """
        return DiDegreeView(self)

    @cached_property
    def in_degree(self):
        """An InDegreeView for (node, in_degree) or in_degree for single node.

        The node in_degree is the number of edges pointing to the node.
        The weighted node degree is the sum of the edge weights for
        edges incident to that node.

        This object provides an iteration over (node, in_degree) as well as
        lookup for the degree for a single node.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.

        weight : string or None, optional (default=None)
           The name of an edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights adjacent to the node.

        Returns
        -------
        If a single node is requested
        deg : int
            In-degree of the node

        OR if multiple nodes are requested
        nd_iter : iterator
            The iterator returns two-tuples of (node, in-degree).

        See Also
        --------
        degree, out_degree

        Examples
        --------
        >>> G = nx.DiGraph()
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.in_degree(0)  # node 0 with degree 0
        0
        >>> list(G.in_degree([0, 1, 2]))
        [(0, 0), (1, 1), (2, 1)]

        """
        return InDegreeView(self)

    @cached_property
    def out_degree(self):
        """An OutDegreeView for (node, out_degree)

        The node out_degree is the number of edges pointing out of the node.
        The weighted node degree is the sum of the edge weights for
        edges incident to that node.

        This object provides an iterator over (node, out_degree) as well as
        lookup for the degree for a single node.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.

        weight : string or None, optional (default=None)
           The name of an edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights adjacent to the node.

        Returns
        -------
        If a single node is requested
        deg : int
            Out-degree of the node

        OR if multiple nodes are requested
        nd_iter : iterator
            The iterator returns two-tuples of (node, out-degree).

        See Also
        --------
        degree, in_degree

        Examples
        --------
        >>> G = nx.DiGraph()
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.out_degree(0)  # node 0 with degree 1
        1
        >>> list(G.out_degree([0, 1, 2]))
        [(0, 1), (1, 1), (2, 1)]

        """
        return OutDegreeView(self)

    def clear(self):
        """Remove all nodes and edges from the graph.

        This also removes the name, and all graph, node, and edge attributes.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.clear()
        >>> list(G.nodes)
        []
        >>> list(G.edges)
        []

        """
        self._succ.clear()
        self._pred.clear()
        self._node.clear()
        self.graph.clear()
        nx._clear_cache(self)

    def clear_edges(self):
        """Remove all edges from the graph without altering nodes.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.clear_edges()
        >>> list(G.nodes)
        [0, 1, 2, 3]
        >>> list(G.edges)
        []

        """
        for predecessor_dict in self._pred.values():
            predecessor_dict.clear()
        for successor_dict in self._succ.values():
            successor_dict.clear()
        nx._clear_cache(self)

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return False

    def is_directed(self):
        """Returns True if graph is directed, False otherwise."""
        return True

    def to_undirected(self, reciprocal=False, as_view=False):
        """Returns an undirected representation of the digraph.

        Parameters
        ----------
        reciprocal : bool (optional)
          If True only keep edges that appear in both directions
          in the original digraph.
        as_view : bool (optional, default=False)
          If True return an undirected view of the original directed graph.

        Returns
        -------
        G : Graph
            An undirected graph with the same name and nodes and
            with edge (u, v, data) if either (u, v, data) or (v, u, data)
            is in the digraph.  If both edges exist in digraph and
            their edge data is different, only one edge is created
            with an arbitrary choice of which edge data to use.
            You must check and correct for this manually if desired.

        See Also
        --------
        Graph, copy, add_edge, add_edges_from

        Notes
        -----
        If edges in both directions (u, v) and (v, u) exist in the
        graph, attributes for the new undirected edge will be a combination of
        the attributes of the directed edges.  The edge data is updated
        in the (arbitrary) order that the edges are encountered.  For
        more customized control of the edge attributes use add_edge().

        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.

        This is in contrast to the similar G=DiGraph(D) which returns a
        shallow copy of the data.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Warning: If you have subclassed DiGraph to use dict-like objects
        in the data structure, those changes do not transfer to the
        Graph created by this method.

        Examples
        --------
        >>> G = nx.path_graph(2)  # or MultiGraph, etc
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1), (1, 0)]
        >>> G2 = H.to_undirected()
        >>> list(G2.edges)
        [(0, 1)]
        """
        graph_class = self.to_undirected_class()
        if as_view is True:
            return nx.graphviews.generic_graph_view(self, graph_class)
        # deepcopy when not a view
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
        if reciprocal is True:
            G.add_edges_from(
                (u, v, deepcopy(d))
                for u, nbrs in self._adj.items()
                for v, d in nbrs.items()
                if v in self._pred[u]
            )
        else:
            G.add_edges_from(
                (u, v, deepcopy(d))
                for u, nbrs in self._adj.items()
                for v, d in nbrs.items()
            )
        return G

    def reverse(self, copy=True):
        """Returns the reverse of the graph.

        The reverse is a graph with the same nodes and edges
        but with the directions of the edges reversed.

        Parameters
        ----------
        copy : bool optional (default=True)
            If True, return a new DiGraph holding the reversed edges.
            If False, the reverse graph is created using a view of
            the original graph.
        """
        if copy:
            H = self.__class__()
            H.graph.update(deepcopy(self.graph))
            H.add_nodes_from((n, deepcopy(d)) for n, d in self.nodes.items())
            H.add_edges_from((v, u, deepcopy(d)) for u, v, d in self.edges(data=True))
            return H
        return nx.reverse_view(self)
<|endoftext|>"""Base class for undirected graphs.

The Graph class allows any hashable object as a node
and can associate key/value attribute pairs with each undirected edge.

Self-loops are allowed but multiple edges are not (see MultiGraph).

For directed graphs see DiGraph and MultiDiGraph.
"""

from copy import deepcopy
from functools import cached_property

import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from networkx.exception import NetworkXError

__all__ = ["Graph"]


class _CachedPropertyResetterAdj:
    """Data Descriptor class for _adj that resets ``adj`` cached_property when needed

    This assumes that the ``cached_property`` ``G.adj`` should be reset whenever
    ``G._adj`` is set to a new value.

    This object sits on a class and ensures that any instance of that
    class clears its cached property "adj" whenever the underlying
    instance attribute "_adj" is set to a new object. It only affects
    the set process of the obj._adj attribute. All get/del operations
    act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """

    def __set__(self, obj, value):
        od = obj.__dict__
        od["_adj"] = value
        # reset cached properties
        props = ["adj", "edges", "degree"]
        for prop in props:
            if prop in od:
                del od[prop]


class _CachedPropertyResetterNode:
    """Data Descriptor class for _node that resets ``nodes`` cached_property when needed

    This assumes that the ``cached_property`` ``G.node`` should be reset whenever
    ``G._node`` is set to a new value.

    This object sits on a class and ensures that any instance of that
    class clears its cached property "nodes" whenever the underlying
    instance attribute "_node" is set to a new object. It only affects
    the set process of the obj._adj attribute. All get/del operations
    act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """

    def __set__(self, obj, value):
        od = obj.__dict__
        od["_node"] = value
        # reset cached properties
        if "nodes" in od:
            del od["nodes"]


class Graph:
    """
    Base class for undirected graphs.

    A Graph stores nodes and edges with optional data, or attributes.

    Graphs hold undirected edges.  Self loops are allowed but multiple
    (parallel) edges are not.

    Nodes can be arbitrary (hashable) Python objects with optional
    key/value attributes, except that `None` is not allowed as a node.

    Edges are represented as links between nodes with optional
    key/value attributes.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    DiGraph
    MultiGraph
    MultiDiGraph

    Examples
    --------
    Create an empty graph structure (a "null graph") with no nodes and
    no edges.

    >>> G = nx.Graph()

    G can be grown in several ways.

    **Nodes:**

    Add one node at a time:

    >>> G.add_node(1)

    Add the nodes from any container (a list, dict, set or
    even the lines from a file or the nodes from another graph).

    >>> G.add_nodes_from([2, 3])
    >>> G.add_nodes_from(range(100, 110))
    >>> H = nx.path_graph(10)
    >>> G.add_nodes_from(H)

    In addition to strings and integers any hashable Python object
    (except None) can represent a node, e.g. a customized node object,
    or even another Graph.

    >>> G.add_node(H)

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge(1, 2)

    a list of edges,

    >>> G.add_edges_from([(1, 2), (1, 3)])

    or a collection of edges,

    >>> G.add_edges_from(H.edges)

    If some edges connect nodes not yet in the graph, the nodes
    are added automatically.  There are no errors when adding
    nodes or edges that already exist.

    **Attributes:**

    Each graph, node, and edge can hold key/value attribute pairs
    in an associated attribute dictionary (the keys must be hashable).
    By default these are empty, but can be added or changed using
    add_edge, add_node or direct manipulation of the attribute
    dictionaries named graph, node and edge respectively.

    >>> G = nx.Graph(day="Friday")
    >>> G.graph
    {'day': 'Friday'}

    Add node attributes using add_node(), add_nodes_from() or G.nodes

    >>> G.add_node(1, time="5pm")
    >>> G.add_nodes_from([3], time="2pm")
    >>> G.nodes[1]
    {'time': '5pm'}
    >>> G.nodes[1]["room"] = 714  # node must exist already to use G.nodes
    >>> del G.nodes[1]["room"]  # remove attribute
    >>> list(G.nodes(data=True))
    [(1, {'time': '5pm'}), (3, {'time': '2pm'})]

    Add edge attributes using add_edge(), add_edges_from(), subscript
    notation, or G.edges.

    >>> G.add_edge(1, 2, weight=4.7)
    >>> G.add_edges_from([(3, 4), (4, 5)], color="red")
    >>> G.add_edges_from([(1, 2, {"color": "blue"}), (2, 3, {"weight": 8})])
    >>> G[1][2]["weight"] = 4.7
    >>> G.edges[1, 2]["weight"] = 4

    Warning: we protect the graph data structure by making `G.edges` a
    read-only dict-like structure. However, you can assign to attributes
    in e.g. `G.edges[1, 2]`. Thus, use 2 sets of brackets to add/change
    data attributes: `G.edges[1, 2]['weight'] = 4`
    (For multigraphs: `MG.edges[u, v, key][name] = value`).

    **Shortcuts:**

    Many common graph features allow python syntax to speed reporting.

    >>> 1 in G  # check if node in graph
    True
    >>> [n for n in G if n < 3]  # iterate through nodes
    [1, 2]
    >>> len(G)  # number of nodes in graph
    5

    Often the best way to traverse all edges of a graph is via the neighbors.
    The neighbors are reported as an adjacency-dict `G.adj` or `G.adjacency()`

    >>> for n, nbrsdict in G.adjacency():
    ...     for nbr, eattr in nbrsdict.items():
    ...         if "weight" in eattr:
    ...             # Do something useful with the edges
    ...             pass

    But the edges() method is often more convenient:

    >>> for u, v, weight in G.edges.data("weight"):
    ...     if weight is not None:
    ...         # Do something useful with the edges
    ...         pass

    **Reporting:**

    Simple graph information is obtained using object-attributes and methods.
    Reporting typically provides views instead of containers to reduce memory
    usage. The views update as the graph is updated similarly to dict-views.
    The objects `nodes`, `edges` and `adj` provide access to data attributes
    via lookup (e.g. `nodes[n]`, `edges[u, v]`, `adj[u][v]`) and iteration
    (e.g. `nodes.items()`, `nodes.data('color')`,
    `nodes.data('color', default='blue')` and similarly for `edges`)
    Views exist for `nodes`, `edges`, `neighbors()`/`adj` and `degree`.

    For details on these and other miscellaneous methods, see below.

    **Subclasses (Advanced):**

    The Graph class uses a dict-of-dict-of-dict data structure.
    The outer dict (node_dict) holds adjacency information keyed by node.
    The next dict (adjlist_dict) represents the adjacency information and holds
    edge data keyed by neighbor.  The inner dict (edge_attr_dict) represents
    the edge data and holds edge attribute values keyed by attribute names.

    Each of these three dicts can be replaced in a subclass by a user defined
    dict-like object. In general, the dict-like features should be
    maintained but extra features can be added. To replace one of the
    dicts create a new graph class by changing the class(!) variable
    holding the factory for that dict-like structure.

    node_dict_factory : function, (default: dict)
        Factory function to be used to create the dict containing node
        attributes, keyed by node id.
        It should require no arguments and return a dict-like object

    node_attr_dict_factory: function, (default: dict)
        Factory function to be used to create the node attribute
        dict which holds attribute values keyed by attribute name.
        It should require no arguments and return a dict-like object

    adjlist_outer_dict_factory : function, (default: dict)
        Factory function to be used to create the outer-most dict
        in the data structure that holds adjacency info keyed by node.
        It should require no arguments and return a dict-like object.

    adjlist_inner_dict_factory : function, (default: dict)
        Factory function to be used to create the adjacency list
        dict which holds edge data keyed by neighbor.
        It should require no arguments and return a dict-like object

    edge_attr_dict_factory : function, (default: dict)
        Factory function to be used to create the edge attribute
        dict which holds attribute values keyed by attribute name.
        It should require no arguments and return a dict-like object.

    graph_attr_dict_factory : function, (default: dict)
        Factory function to be used to create the graph attribute
        dict which holds attribute values keyed by attribute name.
        It should require no arguments and return a dict-like object.

    Typically, if your extension doesn't impact the data structure all
    methods will inherit without issue except: `to_directed/to_undirected`.
    By default these methods create a DiGraph/Graph class and you probably
    want them to create your extension of a DiGraph/Graph. To facilitate
    this we define two class variables that you can set in your subclass.

    to_directed_class : callable, (default: DiGraph or MultiDiGraph)
        Class to create a new graph structure in the `to_directed` method.
        If `None`, a NetworkX class (DiGraph or MultiDiGraph) is used.

    to_undirected_class : callable, (default: Graph or MultiGraph)
        Class to create a new graph structure in the `to_undirected` method.
        If `None`, a NetworkX class (Graph or MultiGraph) is used.

    **Subclassing Example**

    Create a low memory graph class that effectively disallows edge
    attributes by using a single attribute dict for all edges.
    This reduces the memory used, but you lose edge attributes.

    >>> class ThinGraph(nx.Graph):
    ...     all_edge_dict = {"weight": 1}
    ...
    ...     def single_edge_dict(self):
    ...         return self.all_edge_dict
    ...
    ...     edge_attr_dict_factory = single_edge_dict
    >>> G = ThinGraph()
    >>> G.add_edge(2, 1)
    >>> G[2][1]
    {'weight': 1}
    >>> G.add_edge(2, 2)
    >>> G[2][1] is G[2][2]
    True
    """

    __networkx_backend__ = "networkx"

    _adj = _CachedPropertyResetterAdj()
    _node = _CachedPropertyResetterNode()

    node_dict_factory = dict
    node_attr_dict_factory = dict
    adjlist_outer_dict_factory = dict
    adjlist_inner_dict_factory = dict
    edge_attr_dict_factory = dict
    graph_attr_dict_factory = dict

    def to_directed_class(self):
        """Returns the class to use for empty directed copies.

        If you subclass the base classes, use this to designate
        what directed class to use for `to_directed()` copies.
        """
        return nx.DiGraph

    def to_undirected_class(self):
        """Returns the class to use for empty undirected copies.

        If you subclass the base classes, use this to designate
        what directed class to use for `to_directed()` copies.
        """
        return Graph

    def __init__(self, incoming_graph_data=None, **attr):
        """Initialize a graph with edges, name, or graph attributes.

        Parameters
        ----------
        incoming_graph_data : input graph (optional, default: None)
            Data to initialize graph. If None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object.  If the corresponding optional Python
            packages are installed the data can also be a 2D NumPy array, a
            SciPy sparse array, or a PyGraphviz graph.

        attr : keyword arguments, optional (default= no attributes)
            Attributes to add to graph as key=value pairs.

        See Also
        --------
        convert

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G = nx.Graph(name="my graph")
        >>> e = [(1, 2), (2, 3), (3, 4)]  # list of edges
        >>> G = nx.Graph(e)

        Arbitrary graph attribute pairs (key=value) may be assigned

        >>> G = nx.Graph(e, day="Friday")
        >>> G.graph
        {'day': 'Friday'}

        """
        self.graph = self.graph_attr_dict_factory()  # dictionary for graph attributes
        self._node = self.node_dict_factory()  # empty node attribute dict
        self._adj = self.adjlist_outer_dict_factory()  # empty adjacency dict
        self.__networkx_cache__ = {}
        # attempt to load graph with data
        if incoming_graph_data is not None:
            convert.to_networkx_graph(incoming_graph_data, create_using=self)
        # load graph attributes (must be after convert)
        self.graph.update(attr)

    @cached_property
    def adj(self):
        """Graph adjacency object holding the neighbors of each node.

        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edge-data-dict.  So `G.adj[3][2]['color'] = 'blue'` sets
        the color of the edge `(3, 2)` to `"blue"`.

        Iterating over G.adj behaves like a dict. Useful idioms include
        `for nbr, datadict in G.adj[n].items():`.

        The neighbor information is also provided by subscripting the graph.
        So `for nbr, foovalue in G[node].data('foo', default=1):` works.

        For directed graphs, `G.adj` holds outgoing (successor) info.
        """
        return AdjacencyView(self._adj)

    @property
    def name(self):
        """String identifier of the graph.

        This graph attribute appears in the attribute dict G.graph
        keyed by the string `"name"`. as well as an attribute (technically
        a property) `G.name`. This is entirely user controlled.
        """
        return self.graph.get("name", "")

    @name.setter
    def name(self, s):
        self.graph["name"] = s
        nx._clear_cache(self)

    def __str__(self):
        """Returns a short summary of the graph.

        Returns
        -------
        info : string
            Graph information including the graph name (if any), graph type, and the
            number of nodes and edges.

        Examples
        --------
        >>> G = nx.Graph(name="foo")
        >>> str(G)
        "Graph named 'foo' with 0 nodes and 0 edges"

        >>> G = nx.path_graph(3)
        >>> str(G)
        'Graph with 3 nodes and 2 edges'

        """
        return "".join(
            [
                type(self).__name__,
                f" named {self.name!r}" if self.name else "",
                f" with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges",
            ]
        )

    def __iter__(self):
        """Iterate over the nodes. Use: 'for n in G'.

        Returns
        -------
        niter : iterator
            An iterator over all nodes in the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> [n for n in G]
        [0, 1, 2, 3]
        >>> list(G)
        [0, 1, 2, 3]
        """
        return iter(self._node)

    def __contains__(self, n):
        """Returns True if n is a node, False otherwise. Use: 'n in G'.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> 1 in G
        True
        """
        try:
            return n in self._node
        except TypeError:
            return False

    def __len__(self):
        """Returns the number of nodes in the graph. Use: 'len(G)'.

        Returns
        -------
        nnodes : int
            The number of nodes in the graph.

        See Also
        --------
        number_of_nodes: identical method
        order: identical method

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> len(G)
        4

        """
        return len(self._node)

    def __getitem__(self, n):
        """Returns a dict of neighbors of node n.  Use: 'G[n]'.

        Parameters
        ----------
        n : node
           A node in the graph.

        Returns
        -------
        adj_dict : dictionary
           The adjacency dictionary for nodes connected to n.

        Notes
        -----
        G[n] is the same as G.adj[n] and similar to G.neighbors(n)
        (which is an iterator over G.adj[n])

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G[0]
        AtlasView({1: {}})
        """
        return self.adj[n]

    def add_node(self, node_for_adding, **attr):
        """Add a single node `node_for_adding` and update node attributes.

        Parameters
        ----------
        node_for_adding : node
            A node can be any hashable Python object except None.
        attr : keyword arguments, optional
            Set or change node attributes using key=value.

        See Also
        --------
        add_nodes_from

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_node(1)
        >>> G.add_node("Hello")
        >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> G.add_node(K3)
        >>> G.number_of_nodes()
        3

        Use keywords set/change node attributes:

        >>> G.add_node(1, size=10)
        >>> G.add_node(3, weight=0.4, UTM=("13S", 382871, 3972649))

        Notes
        -----
        A hashable object is one that can be used as a key in a Python
        dictionary. This includes strings, numbers, tuples of strings
        and numbers, etc.

        On many platforms hashable items also include mutables such as
        NetworkX Graphs, though one should be careful that the hash
        doesn't change on mutables.
        """
        if node_for_adding not in self._node:
            if node_for_adding is None:
                raise ValueError("None cannot be a node")
            self._adj[node_for_adding] = self.adjlist_inner_dict_factory()
            attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
            attr_dict.update(attr)
        else:  # update attr even if node already exists
            self._node[node_for_adding].update(attr)
        nx._clear_cache(self)

    def add_nodes_from(self, nodes_for_adding, **attr):
        """Add multiple nodes.

        Parameters
        ----------
        nodes_for_adding : iterable container
            A container of nodes (list, dict, set, etc.).
            OR
            A container of (node, attribute dict) tuples.
            Node attributes are updated using the attribute dict.
        attr : keyword arguments, optional (default= no attributes)
            Update attributes for all nodes in nodes.
            Node attributes specified in nodes as a tuple take
            precedence over attributes specified via keyword arguments.

        See Also
        --------
        add_node

        Notes
        -----
        When adding nodes from an iterator over the graph you are changing,
        a `RuntimeError` can be raised with message:
        `RuntimeError: dictionary changed size during iteration`. This
        happens when the graph's underlying dictionary is modified during
        iteration. To avoid this error, evaluate the iterator into a separate
        object, e.g. by using `list(iterator_of_nodes)`, and pass this
        object to `G.add_nodes_from`.

        Examples
