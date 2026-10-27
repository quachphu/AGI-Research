- Node keys for both provided graphs need to be fully orderable as well as
  hashable.
- Node and edge equality is assumed to be transitive: if A is equal to B, and
  B is equal to C, then A is equal to C.

References
----------
.. [1] M. Houbraken, S. Demeyer, T. Michoel, P. Audenaert, D. Colle,
   M. Pickavet, "The Index-Based Subgraph Matching Algorithm with General
   Symmetries (ISMAGS): Exploiting Symmetry for Faster Subgraph
   Enumeration", PLoS One 9(5): e97896, 2014.
   https://doi.org/10.1371/journal.pone.0097896
.. [2] https://en.wikipedia.org/wiki/Maximum_common_induced_subgraph
"""

__all__ = ["ISMAGS"]

import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps


def are_all_equal(iterable):
    """
    Returns ``True`` if and only if all elements in `iterable` are equal; and
    ``False`` otherwise.

    Parameters
    ----------
    iterable: collections.abc.Iterable
        The container whose elements will be checked.

    Returns
    -------
    bool
        ``True`` iff all elements in `iterable` compare equal, ``False``
        otherwise.
    """
    try:
        shape = iterable.shape
    except AttributeError:
        pass
    else:
        if len(shape) > 1:
            message = "The function does not works on multidimensional arrays."
            raise NotImplementedError(message) from None

    iterator = iter(iterable)
    first = next(iterator, None)
    return all(item == first for item in iterator)


def make_partitions(items, test):
    """
    Partitions items into sets based on the outcome of ``test(item1, item2)``.
    Pairs of items for which `test` returns `True` end up in the same set.

    Parameters
    ----------
    items : collections.abc.Iterable[collections.abc.Hashable]
        Items to partition
    test : collections.abc.Callable[collections.abc.Hashable, collections.abc.Hashable]
        A function that will be called with 2 arguments, taken from items.
        Should return `True` if those 2 items need to end up in the same
        partition, and `False` otherwise.

    Returns
    -------
    list[set]
        A list of sets, with each set containing part of the items in `items`,
        such that ``all(test(*pair) for pair in  itertools.combinations(set, 2))
        == True``

    Notes
    -----
    The function `test` is assumed to be transitive: if ``test(a, b)`` and
    ``test(b, c)`` return ``True``, then ``test(a, c)`` must also be ``True``.
    """
    partitions = []
    for item in items:
        for partition in partitions:
            p_item = next(iter(partition))
            if test(item, p_item):
                partition.add(item)
                break
        else:  # No break
            partitions.append({item})
    return partitions


def partition_to_color(partitions):
    """
    Creates a dictionary that maps each item in each partition to the index of
    the partition to which it belongs.

    Parameters
    ----------
    partitions: collections.abc.Sequence[collections.abc.Iterable]
        As returned by :func:`make_partitions`.

    Returns
    -------
    dict
    """
    colors = {}
    for color, keys in enumerate(partitions):
        for key in keys:
            colors[key] = color
    return colors


def intersect(collection_of_sets):
    """
    Given an collection of sets, returns the intersection of those sets.

    Parameters
    ----------
    collection_of_sets: collections.abc.Collection[set]
        A collection of sets.

    Returns
    -------
    set
        An intersection of all sets in `collection_of_sets`. Will have the same
        type as the item initially taken from `collection_of_sets`.
    """
    collection_of_sets = list(collection_of_sets)
    first = collection_of_sets.pop()
    out = reduce(set.intersection, collection_of_sets, set(first))
    return type(first)(out)


class ISMAGS:
    """
    Implements the ISMAGS subgraph matching algorithm. [1]_ ISMAGS stands for
    "Index-based Subgraph Matching Algorithm with General Symmetries". As the
    name implies, it is symmetry aware and will only generate non-symmetric
    isomorphisms.

    Notes
    -----
    The implementation imposes additional conditions compared to the VF2
    algorithm on the graphs provided and the comparison functions
    (:attr:`node_equality` and :attr:`edge_equality`):

     - Node keys in both graphs must be orderable as well as hashable.
     - Equality must be transitive: if A is equal to B, and B is equal to C,
       then A must be equal to C.

    Attributes
    ----------
    graph: networkx.Graph
    subgraph: networkx.Graph
    node_equality: collections.abc.Callable
        The function called to see if two nodes should be considered equal.
        It's signature looks like this:
        ``f(graph1: networkx.Graph, node1, graph2: networkx.Graph, node2) -> bool``.
        `node1` is a node in `graph1`, and `node2` a node in `graph2`.
        Constructed from the argument `node_match`.
    edge_equality: collections.abc.Callable
        The function called to see if two edges should be considered equal.
        It's signature looks like this:
        ``f(graph1: networkx.Graph, edge1, graph2: networkx.Graph, edge2) -> bool``.
        `edge1` is an edge in `graph1`, and `edge2` an edge in `graph2`.
        Constructed from the argument `edge_match`.

    References
    ----------
    .. [1] M. Houbraken, S. Demeyer, T. Michoel, P. Audenaert, D. Colle,
       M. Pickavet, "The Index-Based Subgraph Matching Algorithm with General
       Symmetries (ISMAGS): Exploiting Symmetry for Faster Subgraph
       Enumeration", PLoS One 9(5): e97896, 2014.
       https://doi.org/10.1371/journal.pone.0097896
    """

    def __init__(self, graph, subgraph, node_match=None, edge_match=None, cache=None):
        """
        Parameters
        ----------
        graph: networkx.Graph
        subgraph: networkx.Graph
        node_match: collections.abc.Callable or None
            Function used to determine whether two nodes are equivalent. Its
            signature should look like ``f(n1: dict, n2: dict) -> bool``, with
            `n1` and `n2` node property dicts. See also
            :func:`~networkx.algorithms.isomorphism.categorical_node_match` and
            friends.
            If `None`, all nodes are considered equal.
        edge_match: collections.abc.Callable or None
            Function used to determine whether two edges are equivalent. Its
            signature should look like ``f(e1: dict, e2: dict) -> bool``, with
            `e1` and `e2` edge property dicts. See also
            :func:`~networkx.algorithms.isomorphism.categorical_edge_match` and
            friends.
            If `None`, all edges are considered equal.
        cache: collections.abc.Mapping
            A cache used for caching graph symmetries.
        """
        # TODO: graph and subgraph setter methods that invalidate the caches.
        # TODO: allow for precomputed partitions and colors
        self.graph = graph
        self.subgraph = subgraph
        self._symmetry_cache = cache
        # Naming conventions are taken from the original paper. For your
        # sanity:
        #   sg: subgraph
        #   g: graph
        #   e: edge(s)
        #   n: node(s)
        # So: sgn means "subgraph nodes".
        self._sgn_partitions_ = None
        self._sge_partitions_ = None

        self._sgn_colors_ = None
        self._sge_colors_ = None

        self._gn_partitions_ = None
        self._ge_partitions_ = None

        self._gn_colors_ = None
        self._ge_colors_ = None

        self._node_compat_ = None
        self._edge_compat_ = None

        if node_match is None:
            self.node_equality = self._node_match_maker(lambda n1, n2: True)
            self._sgn_partitions_ = [set(self.subgraph.nodes)]
            self._gn_partitions_ = [set(self.graph.nodes)]
            self._node_compat_ = {0: 0}
        else:
            self.node_equality = self._node_match_maker(node_match)
        if edge_match is None:
            self.edge_equality = self._edge_match_maker(lambda e1, e2: True)
            self._sge_partitions_ = [set(self.subgraph.edges)]
            self._ge_partitions_ = [set(self.graph.edges)]
            self._edge_compat_ = {0: 0}
        else:
            self.edge_equality = self._edge_match_maker(edge_match)

    @property
    def _sgn_partitions(self):
        if self._sgn_partitions_ is None:

            def nodematch(node1, node2):
                return self.node_equality(self.subgraph, node1, self.subgraph, node2)

            self._sgn_partitions_ = make_partitions(self.subgraph.nodes, nodematch)
        return self._sgn_partitions_

    @property
    def _sge_partitions(self):
        if self._sge_partitions_ is None:

            def edgematch(edge1, edge2):
                return self.edge_equality(self.subgraph, edge1, self.subgraph, edge2)

            self._sge_partitions_ = make_partitions(self.subgraph.edges, edgematch)
        return self._sge_partitions_

    @property
    def _gn_partitions(self):
        if self._gn_partitions_ is None:

            def nodematch(node1, node2):
                return self.node_equality(self.graph, node1, self.graph, node2)

            self._gn_partitions_ = make_partitions(self.graph.nodes, nodematch)
        return self._gn_partitions_

    @property
    def _ge_partitions(self):
        if self._ge_partitions_ is None:

            def edgematch(edge1, edge2):
                return self.edge_equality(self.graph, edge1, self.graph, edge2)

            self._ge_partitions_ = make_partitions(self.graph.edges, edgematch)
        return self._ge_partitions_

    @property
    def _sgn_colors(self):
        if self._sgn_colors_ is None:
            self._sgn_colors_ = partition_to_color(self._sgn_partitions)
        return self._sgn_colors_

    @property
    def _sge_colors(self):
        if self._sge_colors_ is None:
            self._sge_colors_ = partition_to_color(self._sge_partitions)
        return self._sge_colors_

    @property
    def _gn_colors(self):
        if self._gn_colors_ is None:
            self._gn_colors_ = partition_to_color(self._gn_partitions)
        return self._gn_colors_

    @property
    def _ge_colors(self):
        if self._ge_colors_ is None:
            self._ge_colors_ = partition_to_color(self._ge_partitions)
        return self._ge_colors_

    @property
    def _node_compatibility(self):
        if self._node_compat_ is not None:
            return self._node_compat_
        self._node_compat_ = {}
        for sgn_part_color, gn_part_color in itertools.product(
            range(len(self._sgn_partitions)), range(len(self._gn_partitions))
        ):
            sgn = next(iter(self._sgn_partitions[sgn_part_color]))
            gn = next(iter(self._gn_partitions[gn_part_color]))
            if self.node_equality(self.subgraph, sgn, self.graph, gn):
                self._node_compat_[sgn_part_color] = gn_part_color
        return self._node_compat_

    @property
    def _edge_compatibility(self):
        if self._edge_compat_ is not None:
            return self._edge_compat_
        self._edge_compat_ = {}
        for sge_part_color, ge_part_color in itertools.product(
            range(len(self._sge_partitions)), range(len(self._ge_partitions))
        ):
            sge = next(iter(self._sge_partitions[sge_part_color]))
            ge = next(iter(self._ge_partitions[ge_part_color]))
            if self.edge_equality(self.subgraph, sge, self.graph, ge):
                self._edge_compat_[sge_part_color] = ge_part_color
        return self._edge_compat_

    @staticmethod
    def _node_match_maker(cmp):
        @wraps(cmp)
        def comparer(graph1, node1, graph2, node2):
            return cmp(graph1.nodes[node1], graph2.nodes[node2])

        return comparer

    @staticmethod
    def _edge_match_maker(cmp):
        @wraps(cmp)
        def comparer(graph1, edge1, graph2, edge2):
            return cmp(graph1.edges[edge1], graph2.edges[edge2])

        return comparer

    def find_isomorphisms(self, symmetry=True):
        """Find all subgraph isomorphisms between subgraph and graph

        Finds isomorphisms where :attr:`subgraph` <= :attr:`graph`.

        Parameters
        ----------
        symmetry: bool
            Whether symmetry should be taken into account. If False, found
            isomorphisms may be symmetrically equivalent.

        Yields
        ------
        dict
            The found isomorphism mappings of {graph_node: subgraph_node}.
        """
        # The networkx VF2 algorithm is slightly funny in when it yields an
        # empty dict and when not.
        if not self.subgraph:
            yield {}
            return
        elif not self.graph:
            return
        elif len(self.graph) < len(self.subgraph):
            return

        if symmetry:
            _, cosets = self.analyze_symmetry(
                self.subgraph, self._sgn_partitions, self._sge_colors
            )
            constraints = self._make_constraints(cosets)
        else:
            constraints = []

        candidates = self._find_nodecolor_candidates()
        la_candidates = self._get_lookahead_candidates()
        for sgn in self.subgraph:
            extra_candidates = la_candidates[sgn]
            if extra_candidates:
                candidates[sgn] = candidates[sgn] | {frozenset(extra_candidates)}

        if any(candidates.values()):
            start_sgn = min(candidates, key=lambda n: min(candidates[n], key=len))
            candidates[start_sgn] = (intersect(candidates[start_sgn]),)
            yield from self._map_nodes(start_sgn, candidates, constraints)
        else:
            return

    @staticmethod
    def _find_neighbor_color_count(graph, node, node_color, edge_color):
        """
        For `node` in `graph`, count the number of edges of a specific color
        it has to nodes of a specific color.
        """
        counts = Counter()
        neighbors = graph[node]
        for neighbor in neighbors:
            n_color = node_color[neighbor]
            if (node, neighbor) in edge_color:
                e_color = edge_color[node, neighbor]
            else:
                e_color = edge_color[neighbor, node]
            counts[e_color, n_color] += 1
        return counts

    def _get_lookahead_candidates(self):
        """
        Returns a mapping of {subgraph node: collection of graph nodes} for
        which the graph nodes are feasible candidates for the subgraph node, as
        determined by looking ahead one edge.
        """
        g_counts = {}
        for gn in self.graph:
            g_counts[gn] = self._find_neighbor_color_count(
                self.graph, gn, self._gn_colors, self._ge_colors
            )
        candidates = defaultdict(set)
        for sgn in self.subgraph:
            sg_count = self._find_neighbor_color_count(
                self.subgraph, sgn, self._sgn_colors, self._sge_colors
            )
            new_sg_count = Counter()
            for (sge_color, sgn_color), count in sg_count.items():
                try:
                    ge_color = self._edge_compatibility[sge_color]
                    gn_color = self._node_compatibility[sgn_color]
                except KeyError:
                    pass
                else:
                    new_sg_count[ge_color, gn_color] = count

            for gn, g_count in g_counts.items():
                if all(new_sg_count[x] <= g_count[x] for x in new_sg_count):
                    # Valid candidate
                    candidates[sgn].add(gn)
        return candidates

    def largest_common_subgraph(self, symmetry=True):
        """
        Find the largest common induced subgraphs between :attr:`subgraph` and
        :attr:`graph`.

        Parameters
        ----------
        symmetry: bool
            Whether symmetry should be taken into account. If False, found
            largest common subgraphs may be symmetrically equivalent.

        Yields
        ------
        dict
            The found isomorphism mappings of {graph_node: subgraph_node}.
        """
        # The networkx VF2 algorithm is slightly funny in when it yields an
        # empty dict and when not.
        if not self.subgraph:
            yield {}
            return
        elif not self.graph:
            return

        if symmetry:
            _, cosets = self.analyze_symmetry(
                self.subgraph, self._sgn_partitions, self._sge_colors
            )
            constraints = self._make_constraints(cosets)
        else:
            constraints = []

        candidates = self._find_nodecolor_candidates()

        if any(candidates.values()):
            yield from self._largest_common_subgraph(candidates, constraints)
        else:
            return

    def analyze_symmetry(self, graph, node_partitions, edge_colors):
        """
        Find a minimal set of permutations and corresponding co-sets that
        describe the symmetry of `graph`, given the node and edge equalities
        given by `node_partitions` and `edge_colors`, respectively.

        Parameters
        ----------
        graph : networkx.Graph
            The graph whose symmetry should be analyzed.
        node_partitions : list of sets
            A list of sets containing node keys. Node keys in the same set
            are considered equivalent. Every node key in `graph` should be in
            exactly one of the sets. If all nodes are equivalent, this should
            be ``[set(graph.nodes)]``.
        edge_colors : dict mapping edges to their colors
            A dict mapping every edge in `graph` to its corresponding color.
            Edges with the same color are considered equivalent. If all edges
            are equivalent, this should be ``{e: 0 for e in graph.edges}``.


        Returns
        -------
        set[frozenset]
            The found permutations. This is a set of frozensets of pairs of node
            keys which can be exchanged without changing :attr:`subgraph`.
        dict[collections.abc.Hashable, set[collections.abc.Hashable]]
            The found co-sets. The co-sets is a dictionary of
            ``{node key: set of node keys}``.
            Every key-value pair describes which ``values`` can be interchanged
            without changing nodes less than ``key``.
        """
        if self._symmetry_cache is not None:
            key = hash(
                (
                    tuple(graph.nodes),
                    tuple(graph.edges),
                    tuple(map(tuple, node_partitions)),
                    tuple(edge_colors.items()),
                )
            )
            if key in self._symmetry_cache:
                return self._symmetry_cache[key]
        node_partitions = list(
            self._refine_node_partitions(graph, node_partitions, edge_colors)
        )
        assert len(node_partitions) == 1
        node_partitions = node_partitions[0]
        permutations, cosets = self._process_ordered_pair_partitions(
            graph, node_partitions, node_partitions, edge_colors
        )
        if self._symmetry_cache is not None:
            self._symmetry_cache[key] = permutations, cosets
        return permutations, cosets

    def is_isomorphic(self, symmetry=False):
        """
        Returns True if :attr:`graph` is isomorphic to :attr:`subgraph` and
        False otherwise.

        Returns
        -------
        bool
        """
        return len(self.subgraph) == len(self.graph) and self.subgraph_is_isomorphic(
            symmetry
        )

    def subgraph_is_isomorphic(self, symmetry=False):
        """
        Returns True if a subgraph of :attr:`graph` is isomorphic to
        :attr:`subgraph` and False otherwise.

        Returns
        -------
        bool
        """
        # symmetry=False, since we only need to know whether there is any
        # example; figuring out all symmetry elements probably costs more time
        # than it gains.
        isom = next(self.subgraph_isomorphisms_iter(symmetry=symmetry), None)
        return isom is not None

    def isomorphisms_iter(self, symmetry=True):
        """
        Does the same as :meth:`find_isomorphisms` if :attr:`graph` and
        :attr:`subgraph` have the same number of nodes.
        """
        if len(self.graph) == len(self.subgraph):
            yield from self.subgraph_isomorphisms_iter(symmetry=symmetry)

    def subgraph_isomorphisms_iter(self, symmetry=True):
        """Alternative name for :meth:`find_isomorphisms`."""
        return self.find_isomorphisms(symmetry)

    def _find_nodecolor_candidates(self):
        """
        Per node in subgraph find all nodes in graph that have the same color.
        """
        candidates = defaultdict(set)
        for sgn in self.subgraph.nodes:
            sgn_color = self._sgn_colors[sgn]
            if sgn_color in self._node_compatibility:
                gn_color = self._node_compatibility[sgn_color]
                candidates[sgn].add(frozenset(self._gn_partitions[gn_color]))
            else:
                candidates[sgn].add(frozenset())
        candidates = dict(candidates)
        for sgn, options in candidates.items():
            candidates[sgn] = frozenset(options)
        return candidates

    @staticmethod
    def _make_constraints(cosets):
        """
        Turn cosets into constraints.
        """
        constraints = []
        for node_i, node_ts in cosets.items():
            for node_t in node_ts:
                if node_i != node_t:
                    # Node i must be smaller than node t.
                    constraints.append((node_i, node_t))
        return constraints

    @staticmethod
    def _find_node_edge_color(graph, node_colors, edge_colors):
        """
        For every node in graph, come up with a color that combines 1) the
        color of the node, and 2) the number of edges of a color to each type
        of node.
        """
        counts = defaultdict(lambda: defaultdict(int))
        for node1, node2 in graph.edges:
            if (node1, node2) in edge_colors:
                # FIXME directed graphs
                ecolor = edge_colors[node1, node2]
            else:
                ecolor = edge_colors[node2, node1]
            # Count per node how many edges it has of what color to nodes of
            # what color
            counts[node1][ecolor, node_colors[node2]] += 1
            counts[node2][ecolor, node_colors[node1]] += 1

        node_edge_colors = {}
        for node in graph.nodes:
            node_edge_colors[node] = node_colors[node], set(counts[node].items())

        return node_edge_colors

    @staticmethod
    def _get_permutations_by_length(items):
        """
        Get all permutations of items, but only permute items with the same
        length.

        >>> found = list(ISMAGS._get_permutations_by_length([[1], [2], [3, 4], [4, 5]]))
        >>> answer = [
        ...     (([1], [2]), ([3, 4], [4, 5])),
        ...     (([1], [2]), ([4, 5], [3, 4])),
        ...     (([2], [1]), ([3, 4], [4, 5])),
        ...     (([2], [1]), ([4, 5], [3, 4])),
        ... ]
        >>> found == answer
        True
        """
        by_len = defaultdict(list)
        for item in items:
            by_len[len(item)].append(item)

        yield from itertools.product(
            *(itertools.permutations(by_len[l]) for l in sorted(by_len))
        )

    @classmethod
    def _refine_node_partitions(cls, graph, node_partitions, edge_colors, branch=False):
        """
        Given a partition of nodes in graph, make the partitions smaller such
        that all nodes in a partition have 1) the same color, and 2) the same
        number of edges to specific other partitions.
        """

        def equal_color(node1, node2):
            return node_edge_colors[node1] == node_edge_colors[node2]

        node_partitions = list(node_partitions)
        node_colors = partition_to_color(node_partitions)
        node_edge_colors = cls._find_node_edge_color(graph, node_colors, edge_colors)
        if all(
            are_all_equal(node_edge_colors[node] for node in partition)
            for partition in node_partitions
        ):
            yield node_partitions
            return

        new_partitions = []
        output = [new_partitions]
        for partition in node_partitions:
            if not are_all_equal(node_edge_colors[node] for node in partition):
                refined = make_partitions(partition, equal_color)
                if (
                    branch
                    and len(refined) != 1
                    and len({len(r) for r in refined}) != len([len(r) for r in refined])
                ):
                    # This is where it breaks. There are multiple new cells
                    # in refined with the same length, and their order
                    # matters.
                    # So option 1) Hit it with a big hammer and simply make all
                    # orderings.
                    permutations = cls._get_permutations_by_length(refined)
                    new_output = []
                    for n_p in output:
                        for permutation in permutations:
                            new_output.append(n_p + list(permutation[0]))
                    output = new_output
                else:
                    for n_p in output:
                        n_p.extend(sorted(refined, key=len))
            else:
                for n_p in output:
                    n_p.append(partition)
        for n_p in output:
            yield from cls._refine_node_partitions(graph, n_p, edge_colors, branch)

    def _edges_of_same_color(self, sgn1, sgn2):
        """
        Returns all edges in :attr:`graph` that have the same colour as the
        edge between sgn1 and sgn2 in :attr:`subgraph`.
        """
        if (sgn1, sgn2) in self._sge_colors:
            # FIXME directed graphs
            sge_color = self._sge_colors[sgn1, sgn2]
        else:
            sge_color = self._sge_colors[sgn2, sgn1]
        if sge_color in self._edge_compatibility:
            ge_color = self._edge_compatibility[sge_color]
            g_edges = self._ge_partitions[ge_color]
        else:
            g_edges = []
        return g_edges

    def _map_nodes(self, sgn, candidates, constraints, mapping=None, to_be_mapped=None):
        """
        Find all subgraph isomorphisms honoring constraints.
        """
        if mapping is None:
            mapping = {}
        else:
            mapping = mapping.copy()
        if to_be_mapped is None:
            to_be_mapped = set(self.subgraph.nodes)

        # Note, we modify candidates here. Doesn't seem to affect results, but
        # remember this.
        # candidates = candidates.copy()
        sgn_candidates = intersect(candidates[sgn])
        candidates[sgn] = frozenset([sgn_candidates])
        for gn in sgn_candidates:
            # We're going to try to map sgn to gn.
            if gn in mapping.values() or sgn not in to_be_mapped:
                # gn is already mapped to something
                continue  # pragma: no cover

            # REDUCTION and COMBINATION
            mapping[sgn] = gn
            # BASECASE
            if to_be_mapped == set(mapping.keys()):
                yield {v: k for k, v in mapping.items()}
                continue
            left_to_map = to_be_mapped - set(mapping.keys())

            new_candidates = candidates.copy()
            sgn_nbrs = set(self.subgraph[sgn])
            not_gn_nbrs = set(self.graph.nodes) - set(self.graph[gn])
            for sgn2 in left_to_map:
                if sgn2 not in sgn_nbrs:
                    gn2_options = not_gn_nbrs
                else:
                    # Get all edges to gn of the right color:
                    g_edges = self._edges_of_same_color(sgn, sgn2)
                    # FIXME directed graphs
                    # And all nodes involved in those which are connected to gn
                    gn2_options = {n for e in g_edges for n in e if gn in e}
                # Node color compatibility should be taken care of by the
                # initial candidate lists made by find_subgraphs

                # Add gn2_options to the right collection. Since new_candidates
                # is a dict of frozensets of frozensets of node indices it's
                # a bit clunky. We can't do .add, and + also doesn't work. We
                # could do |, but I deem union to be clearer.
                new_candidates[sgn2] = new_candidates[sgn2].union(
                    [frozenset(gn2_options)]
                )

                if (sgn, sgn2) in constraints:
                    gn2_options = {gn2 for gn2 in self.graph if gn2 > gn}
                elif (sgn2, sgn) in constraints:
                    gn2_options = {gn2 for gn2 in self.graph if gn2 < gn}
                else:
                    continue  # pragma: no cover
                new_candidates[sgn2] = new_candidates[sgn2].union(
                    [frozenset(gn2_options)]
                )

            # The next node is the one that is unmapped and has fewest
            # candidates
            next_sgn = min(left_to_map, key=lambda n: min(new_candidates[n], key=len))
            yield from self._map_nodes(
                next_sgn,
                new_candidates,
                constraints,
                mapping=mapping,
                to_be_mapped=to_be_mapped,
            )
            # Unmap sgn-gn. Strictly not necessary since it'd get overwritten
            # when making a new mapping for sgn.
            # del mapping[sgn]

    def _largest_common_subgraph(self, candidates, constraints, to_be_mapped=None):
        """
        Find all largest common subgraphs honoring constraints.
        """
        if to_be_mapped is None:
            to_be_mapped = {frozenset(self.subgraph.nodes)}

        # The LCS problem is basically a repeated subgraph isomorphism problem
        # with smaller and smaller subgraphs. We store the nodes that are
        # "part of" the subgraph in to_be_mapped, and we make it a little
        # smaller every iteration.

        current_size = len(next(iter(to_be_mapped), []))

        found_iso = False
        if current_size <= len(self.graph):
            # There's no point in trying to find isomorphisms of
            # graph >= subgraph if subgraph has more nodes than graph.

            # Try the isomorphism first with the nodes with lowest ID. So sort
            # them. Those are more likely to be part of the final
            # correspondence. This makes finding the first answer(s) faster. In
            # theory.
            for nodes in sorted(to_be_mapped, key=sorted):
                # Find the isomorphism between subgraph[to_be_mapped] <= graph
                next_sgn = min(nodes, key=lambda n: min(candidates[n], key=len))
                isomorphs = self._map_nodes(
                    next_sgn, candidates, constraints, to_be_mapped=nodes
                )

                # This is effectively `yield from isomorphs`, except that we look
                # whether an item was yielded.
                try:
                    item = next(isomorphs)
                except StopIteration:
                    pass
                else:
                    yield item
                    yield from isomorphs
                    found_iso = True

        # BASECASE
        if found_iso or current_size == 1:
            # Shrinking has no point because either 1) we end up with a smaller
            # common subgraph (and we want the largest), or 2) there'll be no
            # more subgraph.
            return

        left_to_be_mapped = set()
        for nodes in to_be_mapped:
            for sgn in nodes:
                # We're going to remove sgn from to_be_mapped, but subject to
                # symmetry constraints. We know that for every constraint we
                # have those subgraph nodes are equal. So whenever we would
                # remove the lower part of a constraint, remove the higher
                # instead. This is all dealth with by _remove_node. And because
                # left_to_be_mapped is a set, we don't do double work.

                # And finally, make the subgraph one node smaller.
                # REDUCTION
                new_nodes = self._remove_node(sgn, nodes, constraints)
                left_to_be_mapped.add(new_nodes)
        # COMBINATION
        yield from self._largest_common_subgraph(
            candidates, constraints, to_be_mapped=left_to_be_mapped
        )

    @staticmethod
    def _remove_node(node, nodes, constraints):
        """
        Returns a new set where node has been removed from nodes, subject to
        symmetry constraints. We know, that for every constraint we have
        those subgraph nodes are equal. So whenever we would remove the
        lower part of a constraint, remove the higher instead.
        """
        while True:
            for low, high in constraints:
                if low == node and high in nodes:
                    node = high
                    break
            else:  # no break, couldn't find node in constraints
                break
        return frozenset(nodes - {node})

    @staticmethod
    def _find_permutations(top_partitions, bottom_partitions):
        """
        Return the pairs of top/bottom partitions where the partitions are
        different. Ensures that all partitions in both top and bottom
        partitions have size 1.
        """
        # Find permutations
        permutations = set()
        for top, bot in zip(top_partitions, bottom_partitions):
            # top and bot have only one element
            if len(top) != 1 or len(bot) != 1:
                raise IndexError(
                    "Not all nodes are coupled. This is"
                    f" impossible: {top_partitions}, {bottom_partitions}"
                )
            if top != bot:
                permutations.add(frozenset((next(iter(top)), next(iter(bot)))))
        return permutations

    @staticmethod
    def _update_orbits(orbits, permutations):
        """
        Update orbits based on permutations. Orbits is modified in place.
        For every pair of items in permutations their respective orbits are
        merged.
        """
        for permutation in permutations:
            node, node2 = permutation
            # Find the orbits that contain node and node2, and replace the
            # orbit containing node with the union
            first = second = None
            for idx, orbit in enumerate(orbits):
                if first is not None and second is not None:
                    break
                if node in orbit:
                    first = idx
                if node2 in orbit:
                    second = idx
            if first != second:
                orbits[first].update(orbits[second])
                del orbits[second]

    def _couple_nodes(
        self,
        top_partitions,
        bottom_partitions,
        pair_idx,
        t_node,
        b_node,
        graph,
        edge_colors,
    ):
        """
        Generate new partitions from top and bottom_partitions where t_node is
        coupled to b_node. pair_idx is the index of the partitions where t_ and
        b_node can be found.
        """
        t_partition = top_partitions[pair_idx]
        b_partition = bottom_partitions[pair_idx]
        assert t_node in t_partition and b_node in b_partition
        # Couple node to node2. This means they get their own partition
        new_top_partitions = [top.copy() for top in top_partitions]
        new_bottom_partitions = [bot.copy() for bot in bottom_partitions]
        new_t_groups = {t_node}, t_partition - {t_node}
        new_b_groups = {b_node}, b_partition - {b_node}
        # Replace the old partitions with the coupled ones
        del new_top_partitions[pair_idx]
        del new_bottom_partitions[pair_idx]
        new_top_partitions[pair_idx:pair_idx] = new_t_groups
        new_bottom_partitions[pair_idx:pair_idx] = new_b_groups

        new_top_partitions = self._refine_node_partitions(
            graph, new_top_partitions, edge_colors
        )
        new_bottom_partitions = self._refine_node_partitions(
            graph, new_bottom_partitions, edge_colors, branch=True
        )
        new_top_partitions = list(new_top_partitions)
        assert len(new_top_partitions) == 1
        new_top_partitions = new_top_partitions[0]
        for bot in new_bottom_partitions:
            yield list(new_top_partitions), bot

    def _process_ordered_pair_partitions(
        self,
        graph,
        top_partitions,
        bottom_partitions,
        edge_colors,
        orbits=None,
        cosets=None,
    ):
        """
        Processes ordered pair partitions as per the reference paper. Finds and
        returns all permutations and cosets that leave the graph unchanged.
        """
        if orbits is None:
            orbits = [{node} for node in graph.nodes]
        else:
            # Note that we don't copy orbits when we are given one. This means
            # we leak information between the recursive branches. This is
            # intentional!
            orbits = orbits
        if cosets is None:
            cosets = {}
        else:
            cosets = cosets.copy()

        assert all(
            len(t_p) == len(b_p) for t_p, b_p in zip(top_partitions, bottom_partitions)
        )

        # BASECASE
        if all(len(top) == 1 for top in top_partitions):
            # All nodes are mapped
            permutations = self._find_permutations(top_partitions, bottom_partitions)
            self._update_orbits(orbits, permutations)
            if permutations:
                return [permutations], cosets
            else:
                return [], cosets

        permutations = []
        unmapped_nodes = {
            (node, idx)
            for idx, t_partition in enumerate(top_partitions)
            for node in t_partition
            if len(t_partition) > 1
        }
        node, pair_idx = min(unmapped_nodes)
        b_partition = bottom_partitions[pair_idx]

        for node2 in sorted(b_partition):
            if len(b_partition) == 1:
                # Can never result in symmetry
                continue
            if node != node2 and any(
                node in orbit and node2 in orbit for orbit in orbits
            ):
                # Orbit prune branch
                continue
            # REDUCTION
            # Couple node to node2
            partitions = self._couple_nodes(
                top_partitions,
                bottom_partitions,
                pair_idx,
                node,
                node2,
                graph,
                edge_colors,
            )
            for opp in partitions:
                new_top_partitions, new_bottom_partitions = opp

                new_perms, new_cosets = self._process_ordered_pair_partitions(
                    graph,
                    new_top_partitions,
                    new_bottom_partitions,
                    edge_colors,
                    orbits,
                    cosets,
                )
                # COMBINATION
                permutations += new_perms
                cosets.update(new_cosets)

        mapped = {
            k
            for top, bottom in zip(top_partitions, bottom_partitions)
            for k in top
            if len(top) == 1 and top == bottom
        }
        ks = {k for k in graph.nodes if k < node}
        # Have all nodes with ID < node been mapped?
        find_coset = ks <= mapped and node not in cosets
        if find_coset:
            # Find the orbit that contains node
            for orbit in orbits:
                if node in orbit:
                    cosets[node] = orbit.copy()
        return permutations, cosets
<|endoftext|>"""
Graph isomorphism functions.
"""

import networkx as nx
from networkx.exception import NetworkXError

__all__ = [
    "could_be_isomorphic",
    "fast_could_be_isomorphic",
    "faster_could_be_isomorphic",
    "is_isomorphic",
]


@nx._dispatchable(graphs={"G1": 0, "G2": 1})
def could_be_isomorphic(G1, G2):
    """Returns False if graphs are definitely not isomorphic.
    True does NOT guarantee isomorphism.

    Parameters
    ----------
    G1, G2 : graphs
       The two graphs G1 and G2 must be the same type.

    Notes
    -----
    Checks for matching degree, triangle, and number of cliques sequences.
    The triangle sequence contains the number of triangles each node is part of.
    The clique sequence contains for each node the number of maximal cliques
    involving that node.

    """

    # Check global properties
    if G1.order() != G2.order():
        return False

    # Check local properties
    d1 = G1.degree()
    t1 = nx.triangles(G1)
    clqs_1 = list(nx.find_cliques(G1))
    c1 = {n: sum(1 for c in clqs_1 if n in c) for n in G1}  # number of cliques
    props1 = [[d, t1[v], c1[v]] for v, d in d1]
    props1.sort()

    d2 = G2.degree()
    t2 = nx.triangles(G2)
    clqs_2 = list(nx.find_cliques(G2))
    c2 = {n: sum(1 for c in clqs_2 if n in c) for n in G2}  # number of cliques
    props2 = [[d, t2[v], c2[v]] for v, d in d2]
    props2.sort()

    if props1 != props2:
        return False

    # OK...
    return True


graph_could_be_isomorphic = could_be_isomorphic


@nx._dispatchable(graphs={"G1": 0, "G2": 1})
def fast_could_be_isomorphic(G1, G2):
    """Returns False if graphs are definitely not isomorphic.

    True does NOT guarantee isomorphism.

    Parameters
    ----------
    G1, G2 : graphs
       The two graphs G1 and G2 must be the same type.

    Notes
    -----
    Checks for matching degree and triangle sequences. The triangle
    sequence contains the number of triangles each node is part of.
    """
    # Check global properties
    if G1.order() != G2.order():
        return False

    # Check local properties
    d1 = G1.degree()
    t1 = nx.triangles(G1)
    props1 = [[d, t1[v]] for v, d in d1]
    props1.sort()

    d2 = G2.degree()
    t2 = nx.triangles(G2)
    props2 = [[d, t2[v]] for v, d in d2]
    props2.sort()

    if props1 != props2:
        return False

    # OK...
    return True


fast_graph_could_be_isomorphic = fast_could_be_isomorphic


@nx._dispatchable(graphs={"G1": 0, "G2": 1})
def faster_could_be_isomorphic(G1, G2):
    """Returns False if graphs are definitely not isomorphic.

    True does NOT guarantee isomorphism.

    Parameters
    ----------
    G1, G2 : graphs
       The two graphs G1 and G2 must be the same type.

    Notes
    -----
    Checks for matching degree sequences.
    """
    # Check global properties
    if G1.order() != G2.order():
        return False

    # Check local properties
    d1 = sorted(d for n, d in G1.degree())
    d2 = sorted(d for n, d in G2.degree())

    if d1 != d2:
        return False

    # OK...
    return True


faster_graph_could_be_isomorphic = faster_could_be_isomorphic


@nx._dispatchable(
    graphs={"G1": 0, "G2": 1},
    preserve_edge_attrs="edge_match",
    preserve_node_attrs="node_match",
)
def is_isomorphic(G1, G2, node_match=None, edge_match=None):
    """Returns True if the graphs G1 and G2 are isomorphic and False otherwise.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2 should
        be considered equal during the isomorphism test.
        If node_match is not specified then node attributes are not considered.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute dictionaries
        for n1 and n2 as inputs.

    edge_match : callable
        A function that returns True if the edge attribute dictionary
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during the isomorphism test.  If edge_match is
        not specified then edge attributes are not considered.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute dictionaries
        of the edges under consideration.

    Notes
    -----
    Uses the vf2 algorithm [1]_.

    Examples
    --------
    >>> import networkx.algorithms.isomorphism as iso

    For digraphs G1 and G2, using 'weight' edge attribute (default: 1)

    >>> G1 = nx.DiGraph()
    >>> G2 = nx.DiGraph()
    >>> nx.add_path(G1, [1, 2, 3, 4], weight=1)
    >>> nx.add_path(G2, [10, 20, 30, 40], weight=2)
    >>> em = iso.numerical_edge_match("weight", 1)
    >>> nx.is_isomorphic(G1, G2)  # no weights considered
    True
    >>> nx.is_isomorphic(G1, G2, edge_match=em)  # match weights
    False

    For multidigraphs G1 and G2, using 'fill' node attribute (default: '')

    >>> G1 = nx.MultiDiGraph()
    >>> G2 = nx.MultiDiGraph()
    >>> G1.add_nodes_from([1, 2, 3], fill="red")
    >>> G2.add_nodes_from([10, 20, 30, 40], fill="red")
    >>> nx.add_path(G1, [1, 2, 3, 4], weight=3, linewidth=2.5)
    >>> nx.add_path(G2, [10, 20, 30, 40], weight=3)
    >>> nm = iso.categorical_node_match("fill", "red")
    >>> nx.is_isomorphic(G1, G2, node_match=nm)
    True

    For multidigraphs G1 and G2, using 'weight' edge attribute (default: 7)

    >>> G1.add_edge(1, 2, weight=7)
    1
    >>> G2.add_edge(10, 20)
    1
    >>> em = iso.numerical_multiedge_match("weight", 7, rtol=1e-6)
    >>> nx.is_isomorphic(G1, G2, edge_match=em)
    True

    For multigraphs G1 and G2, using 'weight' and 'linewidth' edge attributes
    with default values 7 and 2.5. Also using 'fill' node attribute with
    default value 'red'.

    >>> em = iso.numerical_multiedge_match(["weight", "linewidth"], [7, 2.5])
    >>> nm = iso.categorical_node_match("fill", "red")
    >>> nx.is_isomorphic(G1, G2, edge_match=em, node_match=nm)
    True

    See Also
    --------
    numerical_node_match, numerical_edge_match, numerical_multiedge_match
    categorical_node_match, categorical_edge_match, categorical_multiedge_match

    References
    ----------
    .. [1]  L. P. Cordella, P. Foggia, C. Sansone, M. Vento,
       "An Improved Algorithm for Matching Large Graphs",
       3rd IAPR-TC15 Workshop  on Graph-based Representations in
       Pattern Recognition, Cuen, pp. 149-159, 2001.
       https://www.researchgate.net/publication/200034365_An_Improved_Algorithm_for_Matching_Large_Graphs
    """
    if G1.is_directed() and G2.is_directed():
        GM = nx.algorithms.isomorphism.DiGraphMatcher
    elif (not G1.is_directed()) and (not G2.is_directed()):
        GM = nx.algorithms.isomorphism.GraphMatcher
    else:
        raise NetworkXError("Graphs G1 and G2 are not of the same type.")

    gm = GM(G1, G2, node_match=node_match, edge_match=edge_match)

    return gm.is_isomorphic()
<|endoftext|>"""Functions which help end users define customize node_match and
edge_match functions to use during isomorphism checks.
"""

import math
import types
from itertools import permutations

__all__ = [
    "categorical_node_match",
    "categorical_edge_match",
    "categorical_multiedge_match",
    "numerical_node_match",
    "numerical_edge_match",
    "numerical_multiedge_match",
    "generic_node_match",
    "generic_edge_match",
    "generic_multiedge_match",
]


def copyfunc(f, name=None):
    """Returns a deepcopy of a function."""
    return types.FunctionType(
        f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__
    )


def allclose(x, y, rtol=1.0000000000000001e-05, atol=1e-08):
    """Returns True if x and y are sufficiently close, elementwise.

    Parameters
    ----------
    rtol : float
        The relative error tolerance.
    atol : float
        The absolute error tolerance.

    """
    # assume finite weights, see numpy.allclose() for reference
    return all(math.isclose(xi, yi, rel_tol=rtol, abs_tol=atol) for xi, yi in zip(x, y))


categorical_doc = """
Returns a comparison function for a categorical node attribute.

The value(s) of the attr(s) must be hashable and comparable via the ==
operator since they are placed into a set([]) object.  If the sets from
G1 and G2 are the same, then the constructed function returns True.

Parameters
----------
attr : string | list
    The categorical node attribute to compare, or a list of categorical
    node attributes to compare.
default : value | list
    The default value for the categorical node attribute, or a list of
    default values for the categorical node attributes.

Returns
-------
match : function
    The customized, categorical `node_match` function.

Examples
--------
>>> import networkx.algorithms.isomorphism as iso
>>> nm = iso.categorical_node_match("size", 1)
>>> nm = iso.categorical_node_match(["color", "size"], ["red", 2])

"""


def categorical_node_match(attr, default):
    if isinstance(attr, str):

        def match(data1, data2):
            return data1.get(attr, default) == data2.get(attr, default)

    else:
        attrs = list(zip(attr, default))  # Python 3

        def match(data1, data2):
            return all(data1.get(attr, d) == data2.get(attr, d) for attr, d in attrs)

    return match


categorical_edge_match = copyfunc(categorical_node_match, "categorical_edge_match")


def categorical_multiedge_match(attr, default):
    if isinstance(attr, str):

        def match(datasets1, datasets2):
            values1 = {data.get(attr, default) for data in datasets1.values()}
            values2 = {data.get(attr, default) for data in datasets2.values()}
            return values1 == values2

    else:
        attrs = list(zip(attr, default))  # Python 3

        def match(datasets1, datasets2):
            values1 = set()
            for data1 in datasets1.values():
                x = tuple(data1.get(attr, d) for attr, d in attrs)
                values1.add(x)
            values2 = set()
            for data2 in datasets2.values():
                x = tuple(data2.get(attr, d) for attr, d in attrs)
                values2.add(x)
            return values1 == values2

    return match


# Docstrings for categorical functions.
categorical_node_match.__doc__ = categorical_doc
categorical_edge_match.__doc__ = categorical_doc.replace("node", "edge")
tmpdoc = categorical_doc.replace("node", "edge")
tmpdoc = tmpdoc.replace("categorical_edge_match", "categorical_multiedge_match")
categorical_multiedge_match.__doc__ = tmpdoc


numerical_doc = """
Returns a comparison function for a numerical node attribute.

The value(s) of the attr(s) must be numerical and sortable.  If the
sorted list of values from G1 and G2 are the same within some
tolerance, then the constructed function returns True.

Parameters
----------
attr : string | list
    The numerical node attribute to compare, or a list of numerical
    node attributes to compare.
default : value | list
    The default value for the numerical node attribute, or a list of
    default values for the numerical node attributes.
rtol : float
    The relative error tolerance.
atol : float
    The absolute error tolerance.

Returns
-------
match : function
    The customized, numerical `node_match` function.

Examples
--------
>>> import networkx.algorithms.isomorphism as iso
>>> nm = iso.numerical_node_match("weight", 1.0)
>>> nm = iso.numerical_node_match(["weight", "linewidth"], [0.25, 0.5])

"""


def numerical_node_match(attr, default, rtol=1.0000000000000001e-05, atol=1e-08):
    if isinstance(attr, str):

        def match(data1, data2):
            return math.isclose(
                data1.get(attr, default),
                data2.get(attr, default),
                rel_tol=rtol,
                abs_tol=atol,
            )

    else:
        attrs = list(zip(attr, default))  # Python 3

        def match(data1, data2):
            values1 = [data1.get(attr, d) for attr, d in attrs]
            values2 = [data2.get(attr, d) for attr, d in attrs]
            return allclose(values1, values2, rtol=rtol, atol=atol)

    return match


numerical_edge_match = copyfunc(numerical_node_match, "numerical_edge_match")


def numerical_multiedge_match(attr, default, rtol=1.0000000000000001e-05, atol=1e-08):
    if isinstance(attr, str):

        def match(datasets1, datasets2):
            values1 = sorted(data.get(attr, default) for data in datasets1.values())
            values2 = sorted(data.get(attr, default) for data in datasets2.values())
            return allclose(values1, values2, rtol=rtol, atol=atol)

    else:
        attrs = list(zip(attr, default))  # Python 3

        def match(datasets1, datasets2):
            values1 = []
            for data1 in datasets1.values():
                x = tuple(data1.get(attr, d) for attr, d in attrs)
                values1.append(x)
            values2 = []
            for data2 in datasets2.values():
                x = tuple(data2.get(attr, d) for attr, d in attrs)
                values2.append(x)
            values1.sort()
            values2.sort()
            for xi, yi in zip(values1, values2):
                if not allclose(xi, yi, rtol=rtol, atol=atol):
                    return False
            else:
                return True

    return match


# Docstrings for numerical functions.
numerical_node_match.__doc__ = numerical_doc
numerical_edge_match.__doc__ = numerical_doc.replace("node", "edge")
tmpdoc = numerical_doc.replace("node", "edge")
tmpdoc = tmpdoc.replace("numerical_edge_match", "numerical_multiedge_match")
numerical_multiedge_match.__doc__ = tmpdoc


generic_doc = """
Returns a comparison function for a generic attribute.

The value(s) of the attr(s) are compared using the specified
operators. If all the attributes are equal, then the constructed
function returns True.

Parameters
----------
attr : string | list
    The node attribute to compare, or a list of node attributes
    to compare.
default : value | list
    The default value for the node attribute, or a list of
    default values for the node attributes.
op : callable | list
    The operator to use when comparing attribute values, or a list
    of operators to use when comparing values for each attribute.

Returns
-------
match : function
    The customized, generic `node_match` function.

Examples
--------
>>> from operator import eq
>>> from math import isclose
>>> from networkx.algorithms.isomorphism import generic_node_match
>>> nm = generic_node_match("weight", 1.0, isclose)
>>> nm = generic_node_match("color", "red", eq)
>>> nm = generic_node_match(["weight", "color"], [1.0, "red"], [isclose, eq])

"""


def generic_node_match(attr, default, op):
    if isinstance(attr, str):

        def match(data1, data2):
            return op(data1.get(attr, default), data2.get(attr, default))

    else:
        attrs = list(zip(attr, default, op))  # Python 3

        def match(data1, data2):
            for attr, d, operator in attrs:
                if not operator(data1.get(attr, d), data2.get(attr, d)):
                    return False
            else:
                return True

    return match


generic_edge_match = copyfunc(generic_node_match, "generic_edge_match")


def generic_multiedge_match(attr, default, op):
    """Returns a comparison function for a generic attribute.

    The value(s) of the attr(s) are compared using the specified
    operators. If all the attributes are equal, then the constructed
    function returns True. Potentially, the constructed edge_match
    function can be slow since it must verify that no isomorphism
    exists between the multiedges before it returns False.

    Parameters
    ----------
    attr : string | list
        The edge attribute to compare, or a list of node attributes
        to compare.
    default : value | list
        The default value for the edge attribute, or a list of
        default values for the edgeattributes.
    op : callable | list
        The operator to use when comparing attribute values, or a list
        of operators to use when comparing values for each attribute.

    Returns
    -------
    match : function
        The customized, generic `edge_match` function.

    Examples
    --------
    >>> from operator import eq
    >>> from math import isclose
    >>> from networkx.algorithms.isomorphism import generic_node_match
    >>> nm = generic_node_match("weight", 1.0, isclose)
    >>> nm = generic_node_match("color", "red", eq)
    >>> nm = generic_node_match(["weight", "color"], [1.0, "red"], [isclose, eq])

    """

    # This is slow, but generic.
    # We must test every possible isomorphism between the edges.
    if isinstance(attr, str):
        attr = [attr]
        default = [default]
        op = [op]
    attrs = list(zip(attr, default))  # Python 3

    def match(datasets1, datasets2):
        values1 = []
        for data1 in datasets1.values():
            x = tuple(data1.get(attr, d) for attr, d in attrs)
            values1.append(x)
        values2 = []
        for data2 in datasets2.values():
            x = tuple(data2.get(attr, d) for attr, d in attrs)
            values2.append(x)
        for vals2 in permutations(values2):
            for xi, yi in zip(values1, vals2):
                if not all(map(lambda x, y, z: z(x, y), xi, yi, op)):
                    # This is not an isomorphism, go to next permutation.
                    break
            else:
                # Then we found an isomorphism.
                return True
        else:
            # Then there are no isomorphisms between the multiedges.
            return False

    return match


# Docstrings for numerical functions.
generic_node_match.__doc__ = generic_doc
generic_edge_match.__doc__ = generic_doc.replace("node", "edge")
<|endoftext|>from networkx.algorithms.isomorphism.isomorph import *
from networkx.algorithms.isomorphism.vf2userfunc import *
from networkx.algorithms.isomorphism.matchhelpers import *
from networkx.algorithms.isomorphism.temporalisomorphvf2 import *
from networkx.algorithms.isomorphism.ismags import *
from networkx.algorithms.isomorphism.tree_isomorphism import *
from networkx.algorithms.isomorphism.vf2pp import *
<|endoftext|>"""
***************
VF2++ Algorithm
***************

An implementation of the VF2++ algorithm [1]_ for Graph Isomorphism testing.

The simplest interface to use this module is to call:

`vf2pp_is_isomorphic`: to check whether two graphs are isomorphic.
`vf2pp_isomorphism`: to obtain the node mapping between two graphs,
in case they are isomorphic.
`vf2pp_all_isomorphisms`: to generate all possible mappings between two graphs,
if isomorphic.

Introduction
------------
The VF2++ algorithm, follows a similar logic to that of VF2, while also
introducing new easy-to-check cutting rules and determining the optimal access
order of nodes. It is also implemented in a non-recursive manner, which saves
both time and space, when compared to its previous counterpart.

The optimal node ordering is obtained after taking into consideration both the
degree but also the label rarity of each node.
This way we place the nodes that are more likely to match, first in the order,
thus examining the most promising branches in the beginning.
The rules also consider node labels, making it easier to prune unfruitful
branches early in the process.

Examples
--------

Suppose G1 and G2 are Isomorphic Graphs. Verification is as follows:

Without node labels:

>>> import networkx as nx
>>> G1 = nx.path_graph(4)
>>> G2 = nx.path_graph(4)
>>> nx.vf2pp_is_isomorphic(G1, G2, node_label=None)
True
>>> nx.vf2pp_isomorphism(G1, G2, node_label=None)
{1: 1, 2: 2, 0: 0, 3: 3}

With node labels:

>>> G1 = nx.path_graph(4)
>>> G2 = nx.path_graph(4)
>>> mapped = {1: 1, 2: 2, 3: 3, 0: 0}
>>> nx.set_node_attributes(
...     G1, dict(zip(G1, ["blue", "red", "green", "yellow"])), "label"
... )
>>> nx.set_node_attributes(
...     G2,
...     dict(zip([mapped[u] for u in G1], ["blue", "red", "green", "yellow"])),
...     "label",
... )
>>> nx.vf2pp_is_isomorphic(G1, G2, node_label="label")
True
>>> nx.vf2pp_isomorphism(G1, G2, node_label="label")
{1: 1, 2: 2, 0: 0, 3: 3}

References
----------
.. [1] Jüttner, Alpár & Madarasi, Péter. (2018). "VF2++—An improved subgraph
   isomorphism algorithm". Discrete Applied Mathematics. 242.
   https://doi.org/10.1016/j.dam.2018.02.018

"""

import collections

import networkx as nx

__all__ = ["vf2pp_isomorphism", "vf2pp_is_isomorphic", "vf2pp_all_isomorphisms"]

_GraphParameters = collections.namedtuple(
    "_GraphParameters",
    [
        "G1",
        "G2",
        "G1_labels",
        "G2_labels",
        "nodes_of_G1Labels",
        "nodes_of_G2Labels",
        "G2_nodes_of_degree",
    ],
)

_StateParameters = collections.namedtuple(
    "_StateParameters",
    [
        "mapping",
        "reverse_mapping",
        "T1",
        "T1_in",
        "T1_tilde",
        "T1_tilde_in",
        "T2",
        "T2_in",
        "T2_tilde",
        "T2_tilde_in",
    ],
)


@nx._dispatchable(graphs={"G1": 0, "G2": 1}, node_attrs={"node_label": "default_label"})
def vf2pp_isomorphism(G1, G2, node_label=None, default_label=None):
    """Return an isomorphic mapping between `G1` and `G2` if it exists.

    Parameters
    ----------
    G1, G2 : NetworkX Graph or MultiGraph instances.
        The two graphs to check for isomorphism.

    node_label : str, optional
        The name of the node attribute to be used when comparing nodes.
        The default is `None`, meaning node attributes are not considered
        in the comparison. Any node that doesn't have the `node_label`
        attribute uses `default_label` instead.

    default_label : scalar
        Default value to use when a node doesn't have an attribute
        named `node_label`. Default is `None`.

    Returns
    -------
    dict or None
        Node mapping if the two graphs are isomorphic. None otherwise.
    """
    try:
        mapping = next(vf2pp_all_isomorphisms(G1, G2, node_label, default_label))
        return mapping
    except StopIteration:
        return None


@nx._dispatchable(graphs={"G1": 0, "G2": 1}, node_attrs={"node_label": "default_label"})
def vf2pp_is_isomorphic(G1, G2, node_label=None, default_label=None):
    """Examines whether G1 and G2 are isomorphic.

    Parameters
    ----------
    G1, G2 : NetworkX Graph or MultiGraph instances.
        The two graphs to check for isomorphism.

    node_label : str, optional
        The name of the node attribute to be used when comparing nodes.
        The default is `None`, meaning node attributes are not considered
        in the comparison. Any node that doesn't have the `node_label`
        attribute uses `default_label` instead.

    default_label : scalar
        Default value to use when a node doesn't have an attribute
        named `node_label`. Default is `None`.

    Returns
    -------
    bool
        True if the two graphs are isomorphic, False otherwise.
    """
    if vf2pp_isomorphism(G1, G2, node_label, default_label) is not None:
        return True
    return False


@nx._dispatchable(graphs={"G1": 0, "G2": 1}, node_attrs={"node_label": "default_label"})
def vf2pp_all_isomorphisms(G1, G2, node_label=None, default_label=None):
    """Yields all the possible mappings between G1 and G2.

    Parameters
    ----------
    G1, G2 : NetworkX Graph or MultiGraph instances.
        The two graphs to check for isomorphism.

    node_label : str, optional
        The name of the node attribute to be used when comparing nodes.
        The default is `None`, meaning node attributes are not considered
        in the comparison. Any node that doesn't have the `node_label`
        attribute uses `default_label` instead.

    default_label : scalar
        Default value to use when a node doesn't have an attribute
        named `node_label`. Default is `None`.

    Yields
    ------
    dict
        Isomorphic mapping between the nodes in `G1` and `G2`.
    """
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return False

    # Create the degree dicts based on graph type
    if G1.is_directed():
        G1_degree = {
            n: (in_degree, out_degree)
            for (n, in_degree), (_, out_degree) in zip(G1.in_degree, G1.out_degree)
        }
        G2_degree = {
            n: (in_degree, out_degree)
            for (n, in_degree), (_, out_degree) in zip(G2.in_degree, G2.out_degree)
        }
    else:
        G1_degree = dict(G1.degree)
        G2_degree = dict(G2.degree)

    if not G1.is_directed():
        find_candidates = _find_candidates
        restore_Tinout = _restore_Tinout
    else:
        find_candidates = _find_candidates_Di
        restore_Tinout = _restore_Tinout_Di

    # Check that both graphs have the same number of nodes and degree sequence
    if G1.order() != G2.order():
        return False
    if sorted(G1_degree.values()) != sorted(G2_degree.values()):
        return False

    # Initialize parameters and cache necessary information about degree and labels
    graph_params, state_params = _initialize_parameters(
        G1, G2, G2_degree, node_label, default_label
    )

    # Check if G1 and G2 have the same labels, and that number of nodes per label is equal between the two graphs
    if not _precheck_label_properties(graph_params):
        return False

    # Calculate the optimal node ordering
    node_order = _matching_order(graph_params)

    # Initialize the stack
    stack = []
    candidates = iter(
        find_candidates(node_order[0], graph_params, state_params, G1_degree)
    )
    stack.append((node_order[0], candidates))

    mapping = state_params.mapping
    reverse_mapping = state_params.reverse_mapping

    # Index of the node from the order, currently being examined
    matching_node = 1

    while stack:
        current_node, candidate_nodes = stack[-1]

        try:
            candidate = next(candidate_nodes)
        except StopIteration:
            # If no remaining candidates, return to a previous state, and follow another branch
            stack.pop()
            matching_node -= 1
            if stack:
                # Pop the previously added u-v pair, and look for a different candidate _v for u
                popped_node1, _ = stack[-1]
                popped_node2 = mapping[popped_node1]
                mapping.pop(popped_node1)
                reverse_mapping.pop(popped_node2)
                restore_Tinout(popped_node1, popped_node2, graph_params, state_params)
            continue

        if _feasibility(current_node, candidate, graph_params, state_params):
            # Terminate if mapping is extended to its full
            if len(mapping) == G2.number_of_nodes() - 1:
                cp_mapping = mapping.copy()
                cp_mapping[current_node] = candidate
                yield cp_mapping
                continue

            # Feasibility rules pass, so extend the mapping and update the parameters
            mapping[current_node] = candidate
            reverse_mapping[candidate] = current_node
            _update_Tinout(current_node, candidate, graph_params, state_params)
            # Append the next node and its candidates to the stack
            candidates = iter(
                find_candidates(
                    node_order[matching_node], graph_params, state_params, G1_degree
                )
            )
            stack.append((node_order[matching_node], candidates))
            matching_node += 1


def _precheck_label_properties(graph_params):
    G1, G2, G1_labels, G2_labels, nodes_of_G1Labels, nodes_of_G2Labels, _ = graph_params
    if any(
        label not in nodes_of_G1Labels or len(nodes_of_G1Labels[label]) != len(nodes)
        for label, nodes in nodes_of_G2Labels.items()
    ):
        return False
    return True


def _initialize_parameters(G1, G2, G2_degree, node_label=None, default_label=-1):
    """Initializes all the necessary parameters for VF2++

    Parameters
    ----------
    G1,G2: NetworkX Graph or MultiGraph instances.
        The two graphs to check for isomorphism or monomorphism

    G1_labels,G2_labels: dict
        The label of every node in G1 and G2 respectively

    Returns
    -------
    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2
        G1_labels,G2_labels: dict

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_out, T2_out: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti
    """
    G1_labels = dict(G1.nodes(data=node_label, default=default_label))
    G2_labels = dict(G2.nodes(data=node_label, default=default_label))

    graph_params = _GraphParameters(
        G1,
        G2,
        G1_labels,
        G2_labels,
        nx.utils.groups(G1_labels),
        nx.utils.groups(G2_labels),
        nx.utils.groups(G2_degree),
    )

    T1, T1_in = set(), set()
    T2, T2_in = set(), set()
    if G1.is_directed():
        T1_tilde, T1_tilde_in = (
            set(G1.nodes()),
            set(),
        )  # todo: do we need Ti_tilde_in? What nodes does it have?
        T2_tilde, T2_tilde_in = set(G2.nodes()), set()
    else:
        T1_tilde, T1_tilde_in = set(G1.nodes()), set()
        T2_tilde, T2_tilde_in = set(G2.nodes()), set()

    state_params = _StateParameters(
        {},
        {},
        T1,
        T1_in,
        T1_tilde,
        T1_tilde_in,
        T2,
        T2_in,
        T2_tilde,
        T2_tilde_in,
    )

    return graph_params, state_params


def _matching_order(graph_params):
    """The node ordering as introduced in VF2++.

    Notes
    -----
    Taking into account the structure of the Graph and the node labeling, the nodes are placed in an order such that,
    most of the unfruitful/infeasible branches of the search space can be pruned on high levels, significantly
    decreasing the number of visited states. The premise is that, the algorithm will be able to recognize
    inconsistencies early, proceeding to go deep into the search tree only if it's needed.

    Parameters
    ----------
    graph_params: namedtuple
        Contains:

            G1,G2: NetworkX Graph or MultiGraph instances.
                The two graphs to check for isomorphism or monomorphism.

            G1_labels,G2_labels: dict
                The label of every node in G1 and G2 respectively.

    Returns
    -------
    node_order: list
        The ordering of the nodes.
    """
    G1, G2, G1_labels, _, _, nodes_of_G2Labels, _ = graph_params
    if not G1 and not G2:
        return {}

    if G1.is_directed():
        G1 = G1.to_undirected(as_view=True)

    V1_unordered = set(G1.nodes())
    label_rarity = {label: len(nodes) for label, nodes in nodes_of_G2Labels.items()}
    used_degrees = {node: 0 for node in G1}
    node_order = []

    while V1_unordered:
        max_rarity = min(label_rarity[G1_labels[x]] for x in V1_unordered)
        rarest_nodes = [
            n for n in V1_unordered if label_rarity[G1_labels[n]] == max_rarity
        ]
        max_node = max(rarest_nodes, key=G1.degree)

        for dlevel_nodes in nx.bfs_layers(G1, max_node):
            nodes_to_add = dlevel_nodes.copy()
            while nodes_to_add:
                max_used_degree = max(used_degrees[n] for n in nodes_to_add)
                max_used_degree_nodes = [
                    n for n in nodes_to_add if used_degrees[n] == max_used_degree
                ]
                max_degree = max(G1.degree[n] for n in max_used_degree_nodes)
                max_degree_nodes = [
                    n for n in max_used_degree_nodes if G1.degree[n] == max_degree
                ]
                next_node = min(
                    max_degree_nodes, key=lambda x: label_rarity[G1_labels[x]]
                )

                node_order.append(next_node)
                for node in G1.neighbors(next_node):
                    used_degrees[node] += 1

                nodes_to_add.remove(next_node)
                label_rarity[G1_labels[next_node]] -= 1
                V1_unordered.discard(next_node)

    return node_order


def _find_candidates(
    u, graph_params, state_params, G1_degree
):  # todo: make the 4th argument the degree of u
    """Given node u of G1, finds the candidates of u from G2.

    Parameters
    ----------
    u: Graph node
        The node from G1 for which to find the candidates from G2.

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_tilde, T2_tilde: set
            Ti_tilde contains all the nodes from Gi, that are neither in the mapping nor in Ti

    Returns
    -------
    candidates: set
        The nodes from G2 which are candidates for u.
    """
    G1, G2, G1_labels, _, _, nodes_of_G2Labels, G2_nodes_of_degree = graph_params
    mapping, reverse_mapping, _, _, _, _, _, _, T2_tilde, _ = state_params

    covered_nbrs = [nbr for nbr in G1[u] if nbr in mapping]
    if not covered_nbrs:
        candidates = set(nodes_of_G2Labels[G1_labels[u]])
        candidates.intersection_update(G2_nodes_of_degree[G1_degree[u]])
        candidates.intersection_update(T2_tilde)
        candidates.difference_update(reverse_mapping)
        if G1.is_multigraph():
            candidates.difference_update(
                {
                    node
                    for node in candidates
                    if G1.number_of_edges(u, u) != G2.number_of_edges(node, node)
                }
            )
        return candidates

    nbr1 = covered_nbrs[0]
    common_nodes = set(G2[mapping[nbr1]])

    for nbr1 in covered_nbrs[1:]:
        common_nodes.intersection_update(G2[mapping[nbr1]])

    common_nodes.difference_update(reverse_mapping)
    common_nodes.intersection_update(G2_nodes_of_degree[G1_degree[u]])
    common_nodes.intersection_update(nodes_of_G2Labels[G1_labels[u]])
    if G1.is_multigraph():
        common_nodes.difference_update(
            {
                node
                for node in common_nodes
                if G1.number_of_edges(u, u) != G2.number_of_edges(node, node)
            }
        )
    return common_nodes


def _find_candidates_Di(u, graph_params, state_params, G1_degree):
    G1, G2, G1_labels, _, _, nodes_of_G2Labels, G2_nodes_of_degree = graph_params
    mapping, reverse_mapping, _, _, _, _, _, _, T2_tilde, _ = state_params

    covered_successors = [succ for succ in G1[u] if succ in mapping]
    covered_predecessors = [pred for pred in G1.pred[u] if pred in mapping]

    if not (covered_successors or covered_predecessors):
        candidates = set(nodes_of_G2Labels[G1_labels[u]])
        candidates.intersection_update(G2_nodes_of_degree[G1_degree[u]])
        candidates.intersection_update(T2_tilde)
        candidates.difference_update(reverse_mapping)
        if G1.is_multigraph():
            candidates.difference_update(
                {
                    node
                    for node in candidates
                    if G1.number_of_edges(u, u) != G2.number_of_edges(node, node)
                }
            )
        return candidates

    if covered_successors:
        succ1 = covered_successors[0]
        common_nodes = set(G2.pred[mapping[succ1]])

        for succ1 in covered_successors[1:]:
            common_nodes.intersection_update(G2.pred[mapping[succ1]])
    else:
        pred1 = covered_predecessors.pop()
        common_nodes = set(G2[mapping[pred1]])

    for pred1 in covered_predecessors:
        common_nodes.intersection_update(G2[mapping[pred1]])

    common_nodes.difference_update(reverse_mapping)
    common_nodes.intersection_update(G2_nodes_of_degree[G1_degree[u]])
    common_nodes.intersection_update(nodes_of_G2Labels[G1_labels[u]])
    if G1.is_multigraph():
        common_nodes.difference_update(
            {
                node
                for node in common_nodes
                if G1.number_of_edges(u, u) != G2.number_of_edges(node, node)
            }
        )
    return common_nodes


def _feasibility(node1, node2, graph_params, state_params):
    """Given a candidate pair of nodes u and v from G1 and G2 respectively, checks if it's feasible to extend the
    mapping, i.e. if u and v can be matched.

    Notes
    -----
    This function performs all the necessary checking by applying both consistency and cutting rules.

    Parameters
    ----------
    node1, node2: Graph node
        The candidate pair of nodes being checked for matching

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_out, T2_out: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti

    Returns
    -------
    True if all checks are successful, False otherwise.
    """
    G1 = graph_params.G1

    if _cut_PT(node1, node2, graph_params, state_params):
        return False

    if G1.is_multigraph():
        if not _consistent_PT(node1, node2, graph_params, state_params):
            return False

    return True


def _cut_PT(u, v, graph_params, state_params):
    """Implements the cutting rules for the ISO problem.

    Parameters
    ----------
    u, v: Graph node
        The two candidate nodes being examined.

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_tilde, T2_tilde: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti

    Returns
    -------
    True if we should prune this branch, i.e. the node pair failed the cutting checks. False otherwise.
    """
    G1, G2, G1_labels, G2_labels, _, _, _ = graph_params
    (
        _,
        _,
        T1,
        T1_in,
        T1_tilde,
        _,
        T2,
        T2_in,
        T2_tilde,
        _,
    ) = state_params

    u_labels_predecessors, v_labels_predecessors = {}, {}
    if G1.is_directed():
        u_labels_predecessors = nx.utils.groups(
            {n1: G1_labels[n1] for n1 in G1.pred[u]}
        )
        v_labels_predecessors = nx.utils.groups(
            {n2: G2_labels[n2] for n2 in G2.pred[v]}
        )

        if set(u_labels_predecessors.keys()) != set(v_labels_predecessors.keys()):
            return True

    u_labels_successors = nx.utils.groups({n1: G1_labels[n1] for n1 in G1[u]})
    v_labels_successors = nx.utils.groups({n2: G2_labels[n2] for n2 in G2[v]})

    # if the neighbors of u, do not have the same labels as those of v, NOT feasible.
    if set(u_labels_successors.keys()) != set(v_labels_successors.keys()):
        return True

    for label, G1_nbh in u_labels_successors.items():
        G2_nbh = v_labels_successors[label]

        if G1.is_multigraph():
            # Check for every neighbor in the neighborhood, if u-nbr1 has same edges as v-nbr2
            u_nbrs_edges = sorted(G1.number_of_edges(u, x) for x in G1_nbh)
            v_nbrs_edges = sorted(G2.number_of_edges(v, x) for x in G2_nbh)
            if any(
                u_nbr_edges != v_nbr_edges
                for u_nbr_edges, v_nbr_edges in zip(u_nbrs_edges, v_nbrs_edges)
            ):
                return True

        if len(T1.intersection(G1_nbh)) != len(T2.intersection(G2_nbh)):
            return True
        if len(T1_tilde.intersection(G1_nbh)) != len(T2_tilde.intersection(G2_nbh)):
            return True
        if G1.is_directed() and len(T1_in.intersection(G1_nbh)) != len(
            T2_in.intersection(G2_nbh)
        ):
            return True

    if not G1.is_directed():
        return False

    for label, G1_pred in u_labels_predecessors.items():
        G2_pred = v_labels_predecessors[label]

        if G1.is_multigraph():
            # Check for every neighbor in the neighborhood, if u-nbr1 has same edges as v-nbr2
            u_pred_edges = sorted(G1.number_of_edges(u, x) for x in G1_pred)
            v_pred_edges = sorted(G2.number_of_edges(v, x) for x in G2_pred)
            if any(
                u_nbr_edges != v_nbr_edges
                for u_nbr_edges, v_nbr_edges in zip(u_pred_edges, v_pred_edges)
            ):
                return True

        if len(T1.intersection(G1_pred)) != len(T2.intersection(G2_pred)):
            return True
        if len(T1_tilde.intersection(G1_pred)) != len(T2_tilde.intersection(G2_pred)):
            return True
        if len(T1_in.intersection(G1_pred)) != len(T2_in.intersection(G2_pred)):
            return True

    return False


def _consistent_PT(u, v, graph_params, state_params):
    """Checks the consistency of extending the mapping using the current node pair.

    Parameters
    ----------
    u, v: Graph node
        The two candidate nodes being examined.

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_out, T2_out: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti

    Returns
    -------
    True if the pair passes all the consistency checks successfully. False otherwise.
    """
    G1, G2 = graph_params.G1, graph_params.G2
    mapping, reverse_mapping = state_params.mapping, state_params.reverse_mapping

    for neighbor in G1[u]:
        if neighbor in mapping:
            if G1.number_of_edges(u, neighbor) != G2.number_of_edges(
                v, mapping[neighbor]
            ):
                return False

    for neighbor in G2[v]:
        if neighbor in reverse_mapping:
            if G1.number_of_edges(u, reverse_mapping[neighbor]) != G2.number_of_edges(
                v, neighbor
            ):
                return False

    if not G1.is_directed():
        return True

    for predecessor in G1.pred[u]:
        if predecessor in mapping:
            if G1.number_of_edges(predecessor, u) != G2.number_of_edges(
                mapping[predecessor], v
            ):
                return False

    for predecessor in G2.pred[v]:
        if predecessor in reverse_mapping:
            if G1.number_of_edges(
                reverse_mapping[predecessor], u
            ) != G2.number_of_edges(predecessor, v):
                return False

    return True


def _update_Tinout(new_node1, new_node2, graph_params, state_params):
    """Updates the Ti/Ti_out (i=1,2) when a new node pair u-v is added to the mapping.

    Notes
    -----
    This function should be called right after the feasibility checks are passed, and node1 is mapped to node2. The
    purpose of this function is to avoid brute force computing of Ti/Ti_out by iterating over all nodes of the graph
    and checking which nodes satisfy the necessary conditions. Instead, in every step of the algorithm we focus
    exclusively on the two nodes that are being added to the mapping, incrementally updating Ti/Ti_out.

    Parameters
    ----------
    new_node1, new_node2: Graph node
        The two new nodes, added to the mapping.

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_tilde, T2_tilde: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti
    """
    G1, G2, _, _, _, _, _ = graph_params
    (
        mapping,
        reverse_mapping,
        T1,
        T1_in,
        T1_tilde,
        T1_tilde_in,
        T2,
        T2_in,
        T2_tilde,
        T2_tilde_in,
    ) = state_params

    uncovered_successors_G1 = {succ for succ in G1[new_node1] if succ not in mapping}
    uncovered_successors_G2 = {
        succ for succ in G2[new_node2] if succ not in reverse_mapping
    }

    # Add the uncovered neighbors of node1 and node2 in T1 and T2 respectively
    T1.update(uncovered_successors_G1)
    T2.update(uncovered_successors_G2)
    T1.discard(new_node1)
    T2.discard(new_node2)

    T1_tilde.difference_update(uncovered_successors_G1)
    T2_tilde.difference_update(uncovered_successors_G2)
    T1_tilde.discard(new_node1)
    T2_tilde.discard(new_node2)

    if not G1.is_directed():
        return

    uncovered_predecessors_G1 = {
        pred for pred in G1.pred[new_node1] if pred not in mapping
    }
    uncovered_predecessors_G2 = {
        pred for pred in G2.pred[new_node2] if pred not in reverse_mapping
    }

    T1_in.update(uncovered_predecessors_G1)
    T2_in.update(uncovered_predecessors_G2)
    T1_in.discard(new_node1)
    T2_in.discard(new_node2)

    T1_tilde.difference_update(uncovered_predecessors_G1)
    T2_tilde.difference_update(uncovered_predecessors_G2)
    T1_tilde.discard(new_node1)
    T2_tilde.discard(new_node2)


def _restore_Tinout(popped_node1, popped_node2, graph_params, state_params):
    """Restores the previous version of Ti/Ti_out when a node pair is deleted from the mapping.

    Parameters
    ----------
    popped_node1, popped_node2: Graph node
        The two nodes deleted from the mapping.

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_tilde, T2_tilde: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti
    """
    # If the node we want to remove from the mapping, has at least one covered neighbor, add it to T1.
    G1, G2, _, _, _, _, _ = graph_params
    (
        mapping,
        reverse_mapping,
        T1,
        T1_in,
        T1_tilde,
        T1_tilde_in,
        T2,
        T2_in,
        T2_tilde,
        T2_tilde_in,
    ) = state_params

    is_added = False
    for neighbor in G1[popped_node1]:
        if neighbor in mapping:
            # if a neighbor of the excluded node1 is in the mapping, keep node1 in T1
            is_added = True
            T1.add(popped_node1)
        else:
            # check if its neighbor has another connection with a covered node. If not, only then exclude it from T1
            if any(nbr in mapping for nbr in G1[neighbor]):
                continue
            T1.discard(neighbor)
            T1_tilde.add(neighbor)

    # Case where the node is not present in neither the mapping nor T1. By definition, it should belong to T1_tilde
    if not is_added:
        T1_tilde.add(popped_node1)

    is_added = False
    for neighbor in G2[popped_node2]:
        if neighbor in reverse_mapping:
            is_added = True
            T2.add(popped_node2)
        else:
            if any(nbr in reverse_mapping for nbr in G2[neighbor]):
                continue
            T2.discard(neighbor)
            T2_tilde.add(neighbor)

    if not is_added:
        T2_tilde.add(popped_node2)


def _restore_Tinout_Di(popped_node1, popped_node2, graph_params, state_params):
    # If the node we want to remove from the mapping, has at least one covered neighbor, add it to T1.
    G1, G2, _, _, _, _, _ = graph_params
    (
        mapping,
        reverse_mapping,
        T1,
        T1_in,
        T1_tilde,
        T1_tilde_in,
        T2,
        T2_in,
        T2_tilde,
        T2_tilde_in,
    ) = state_params

    is_added = False
    for successor in G1[popped_node1]:
        if successor in mapping:
            # if a neighbor of the excluded node1 is in the mapping, keep node1 in T1
            is_added = True
            T1_in.add(popped_node1)
        else:
            # check if its neighbor has another connection with a covered node. If not, only then exclude it from T1
            if not any(pred in mapping for pred in G1.pred[successor]):
                T1.discard(successor)

            if not any(succ in mapping for succ in G1[successor]):
                T1_in.discard(successor)

            if successor not in T1:
                if successor not in T1_in:
                    T1_tilde.add(successor)

    for predecessor in G1.pred[popped_node1]:
        if predecessor in mapping:
            # if a neighbor of the excluded node1 is in the mapping, keep node1 in T1
            is_added = True
            T1.add(popped_node1)
        else:
            # check if its neighbor has another connection with a covered node. If not, only then exclude it from T1
            if not any(pred in mapping for pred in G1.pred[predecessor]):
                T1.discard(predecessor)

            if not any(succ in mapping for succ in G1[predecessor]):
                T1_in.discard(predecessor)

            if not (predecessor in T1 or predecessor in T1_in):
                T1_tilde.add(predecessor)

    # Case where the node is not present in neither the mapping nor T1. By definition it should belong to T1_tilde
    if not is_added:
        T1_tilde.add(popped_node1)

    is_added = False
    for successor in G2[popped_node2]:
        if successor in reverse_mapping:
            is_added = True
            T2_in.add(popped_node2)
        else:
            if not any(pred in reverse_mapping for pred in G2.pred[successor]):
                T2.discard(successor)

            if not any(succ in reverse_mapping for succ in G2[successor]):
                T2_in.discard(successor)

            if successor not in T2:
                if successor not in T2_in:
                    T2_tilde.add(successor)

    for predecessor in G2.pred[popped_node2]:
        if predecessor in reverse_mapping:
            # if a neighbor of the excluded node1 is in the mapping, keep node1 in T1
            is_added = True
            T2.add(popped_node2)
        else:
            # check if its neighbor has another connection with a covered node. If not, only then exclude it from T1
            if not any(pred in reverse_mapping for pred in G2.pred[predecessor]):
                T2.discard(predecessor)

            if not any(succ in reverse_mapping for succ in G2[predecessor]):
                T2_in.discard(predecessor)

            if not (predecessor in T2 or predecessor in T2_in):
                T2_tilde.add(predecessor)

    if not is_added:
        T2_tilde.add(popped_node2)
<|endoftext|>"""
*****************************
Time-respecting VF2 Algorithm
*****************************

An extension of the VF2 algorithm for time-respecting graph isomorphism
testing in temporal graphs.

A temporal graph is one in which edges contain a datetime attribute,
denoting when interaction occurred between the incident nodes. A
time-respecting subgraph of a temporal graph is a subgraph such that
all interactions incident to a node occurred within a time threshold,
delta, of each other. A directed time-respecting subgraph has the
added constraint that incoming interactions to a node must precede
outgoing interactions from the same node - this enforces a sense of
directed flow.

Introduction
------------

The TimeRespectingGraphMatcher and TimeRespectingDiGraphMatcher
extend the GraphMatcher and DiGraphMatcher classes, respectively,
to include temporal constraints on matches. This is achieved through
a semantic check, via the semantic_feasibility() function.

As well as including G1 (the graph in which to seek embeddings) and
G2 (the subgraph structure of interest), the name of the temporal
attribute on the edges and the time threshold, delta, must be supplied
as arguments to the matching constructors.

A delta of zero is the strictest temporal constraint on the match -
only embeddings in which all interactions occur at the same time will
be returned. A delta of one day will allow embeddings in which
adjacent interactions occur up to a day apart.

Examples
--------

Examples will be provided when the datetime type has been incorporated.


Temporal Subgraph Isomorphism
-----------------------------

A brief discussion of the somewhat diverse current literature will be
included here.

References
----------

[1] Redmond, U. and Cunningham, P. Temporal subgraph isomorphism. In:
The 2013 IEEE/ACM International Conference on Advances in Social
Networks Analysis and Mining (ASONAM). Niagara Falls, Canada; 2013:
pages 1451 - 1452. [65]

For a discussion of the literature on temporal networks:

[3] P. Holme and J. Saramaki. Temporal networks. Physics Reports,
519(3):97–125, 2012.

Notes
-----

Handles directed and undirected graphs and graphs with parallel edges.

"""

import networkx as nx

from .isomorphvf2 import DiGraphMatcher, GraphMatcher

__all__ = ["TimeRespectingGraphMatcher", "TimeRespectingDiGraphMatcher"]


class TimeRespectingGraphMatcher(GraphMatcher):
    def __init__(self, G1, G2, temporal_attribute_name, delta):
        """Initialize TimeRespectingGraphMatcher.

        G1 and G2 should be nx.Graph or nx.MultiGraph instances.

        Examples
        --------
        To create a TimeRespectingGraphMatcher which checks for
        syntactic and semantic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> from datetime import timedelta
        >>> G1 = nx.Graph(nx.path_graph(4, create_using=nx.Graph()))

        >>> G2 = nx.Graph(nx.path_graph(4, create_using=nx.Graph()))

        >>> GM = isomorphism.TimeRespectingGraphMatcher(
        ...     G1, G2, "date", timedelta(days=1)
        ... )
        """
        self.temporal_attribute_name = temporal_attribute_name
        self.delta = delta
        super().__init__(G1, G2)

    def one_hop(self, Gx, Gx_node, neighbors):
        """
        Edges one hop out from a node in the mapping should be
        time-respecting with respect to each other.
        """
        dates = []
        for n in neighbors:
            if isinstance(Gx, nx.Graph):  # Graph G[u][v] returns the data dictionary.
                dates.append(Gx[Gx_node][n][self.temporal_attribute_name])
            else:  # MultiGraph G[u][v] returns a dictionary of key -> data dictionary.
                for edge in Gx[Gx_node][
                    n
                ].values():  # Iterates all edges between node pair.
                    dates.append(edge[self.temporal_attribute_name])
        if any(x is None for x in dates):
            raise ValueError("Datetime not supplied for at least one edge.")
        return not dates or max(dates) - min(dates) <= self.delta

    def two_hop(self, Gx, core_x, Gx_node, neighbors):
        """
        Paths of length 2 from Gx_node should be time-respecting.
        """
        return all(
            self.one_hop(Gx, v, [n for n in Gx[v] if n in core_x] + [Gx_node])
            for v in neighbors
        )

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is semantically
        feasible.

        Any subclass which redefines semantic_feasibility() must
        maintain the self.tests if needed, to keep the match() method
        functional. Implementations should consider multigraphs.
        """
        neighbors = [n for n in self.G1[G1_node] if n in self.core_1]
        if not self.one_hop(self.G1, G1_node, neighbors):  # Fail fast on first node.
            return False
        if not self.two_hop(self.G1, self.core_1, G1_node, neighbors):
            return False
        # Otherwise, this node is semantically feasible!
        return True


class TimeRespectingDiGraphMatcher(DiGraphMatcher):
    def __init__(self, G1, G2, temporal_attribute_name, delta):
        """Initialize TimeRespectingDiGraphMatcher.

        G1 and G2 should be nx.DiGraph or nx.MultiDiGraph instances.

        Examples
        --------
        To create a TimeRespectingDiGraphMatcher which checks for
        syntactic and semantic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> from datetime import timedelta
        >>> G1 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))

        >>> G2 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))

        >>> GM = isomorphism.TimeRespectingDiGraphMatcher(
        ...     G1, G2, "date", timedelta(days=1)
        ... )
        """
        self.temporal_attribute_name = temporal_attribute_name
        self.delta = delta
        super().__init__(G1, G2)

    def get_pred_dates(self, Gx, Gx_node, core_x, pred):
        """
        Get the dates of edges from predecessors.
        """
        pred_dates = []
        if isinstance(Gx, nx.DiGraph):  # Graph G[u][v] returns the data dictionary.
            for n in pred:
                pred_dates.append(Gx[n][Gx_node][self.temporal_attribute_name])
        else:  # MultiGraph G[u][v] returns a dictionary of key -> data dictionary.
            for n in pred:
                for edge in Gx[n][
                    Gx_node
                ].values():  # Iterates all edge data between node pair.
                    pred_dates.append(edge[self.temporal_attribute_name])
        return pred_dates

    def get_succ_dates(self, Gx, Gx_node, core_x, succ):
        """
        Get the dates of edges to successors.
        """
        succ_dates = []
        if isinstance(Gx, nx.DiGraph):  # Graph G[u][v] returns the data dictionary.
            for n in succ:
                succ_dates.append(Gx[Gx_node][n][self.temporal_attribute_name])
        else:  # MultiGraph G[u][v] returns a dictionary of key -> data dictionary.
            for n in succ:
                for edge in Gx[Gx_node][
                    n
                ].values():  # Iterates all edge data between node pair.
                    succ_dates.append(edge[self.temporal_attribute_name])
        return succ_dates

    def one_hop(self, Gx, Gx_node, core_x, pred, succ):
        """
        The ego node.
        """
        pred_dates = self.get_pred_dates(Gx, Gx_node, core_x, pred)
        succ_dates = self.get_succ_dates(Gx, Gx_node, core_x, succ)
        return self.test_one(pred_dates, succ_dates) and self.test_two(
            pred_dates, succ_dates
        )

    def two_hop_pred(self, Gx, Gx_node, core_x, pred):
        """
        The predecessors of the ego node.
        """
        return all(
            self.one_hop(
                Gx,
                p,
                core_x,
                self.preds(Gx, core_x, p),
                self.succs(Gx, core_x, p, Gx_node),
            )
            for p in pred
        )

    def two_hop_succ(self, Gx, Gx_node, core_x, succ):
        """
        The successors of the ego node.
        """
        return all(
            self.one_hop(
                Gx,
                s,
                core_x,
                self.preds(Gx, core_x, s, Gx_node),
                self.succs(Gx, core_x, s),
            )
            for s in succ
        )

    def preds(self, Gx, core_x, v, Gx_node=None):
        pred = [n for n in Gx.predecessors(v) if n in core_x]
        if Gx_node:
            pred.append(Gx_node)
        return pred

    def succs(self, Gx, core_x, v, Gx_node=None):
        succ = [n for n in Gx.successors(v) if n in core_x]
        if Gx_node:
            succ.append(Gx_node)
        return succ

    def test_one(self, pred_dates, succ_dates):
        """
        Edges one hop out from Gx_node in the mapping should be
        time-respecting with respect to each other, regardless of
        direction.
        """
        time_respecting = True
        dates = pred_dates + succ_dates

        if any(x is None for x in dates):
            raise ValueError("Date or datetime not supplied for at least one edge.")

        dates.sort()  # Small to large.
        if 0 < len(dates) and not (dates[-1] - dates[0] <= self.delta):
            time_respecting = False
        return time_respecting

    def test_two(self, pred_dates, succ_dates):
        """
        Edges from a dual Gx_node in the mapping should be ordered in
        a time-respecting manner.
        """
        time_respecting = True
        pred_dates.sort()
        succ_dates.sort()
        # First out before last in; negative of the necessary condition for time-respect.
        if (
            0 < len(succ_dates)
            and 0 < len(pred_dates)
            and succ_dates[0] < pred_dates[-1]
        ):
            time_respecting = False
        return time_respecting

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is semantically
        feasible.

        Any subclass which redefines semantic_feasibility() must
        maintain the self.tests if needed, to keep the match() method
        functional. Implementations should consider multigraphs.
        """
        pred, succ = (
            [n for n in self.G1.predecessors(G1_node) if n in self.core_1],
            [n for n in self.G1.successors(G1_node) if n in self.core_1],
        )
        if not self.one_hop(
            self.G1, G1_node, self.core_1, pred, succ
        ):  # Fail fast on first node.
            return False
        if not self.two_hop_pred(self.G1, G1_node, self.core_1, pred):
            return False
        if not self.two_hop_succ(self.G1, G1_node, self.core_1, succ):
            return False
        # Otherwise, this node is semantically feasible!
        return True
<|endoftext|>"""
*************
VF2 Algorithm
*************

An implementation of VF2 algorithm for graph isomorphism testing.

The simplest interface to use this module is to call the
:func:`is_isomorphic <networkx.algorithms.isomorphism.is_isomorphic>`
function.

Introduction
------------

The GraphMatcher and DiGraphMatcher are responsible for matching
graphs or directed graphs in a predetermined manner.  This
usually means a check for an isomorphism, though other checks
are also possible.  For example, a subgraph of one graph
can be checked for isomorphism to a second graph.

Matching is done via syntactic feasibility. It is also possible
to check for semantic feasibility. Feasibility, then, is defined
as the logical AND of the two functions.

To include a semantic check, the (Di)GraphMatcher class should be
subclassed, and the
:meth:`semantic_feasibility <networkx.algorithms.isomorphism.GraphMatcher.semantic_feasibility>`
function should be redefined.  By default, the semantic feasibility function always
returns ``True``.  The effect of this is that semantics are not
considered in the matching of G1 and G2.

Examples
--------

Suppose G1 and G2 are isomorphic graphs. Verification is as follows:

>>> from networkx.algorithms import isomorphism
>>> G1 = nx.path_graph(4)
>>> G2 = nx.path_graph(4)
>>> GM = isomorphism.GraphMatcher(G1, G2)
>>> GM.is_isomorphic()
True

GM.mapping stores the isomorphism mapping from G1 to G2.

>>> GM.mapping
{0: 0, 1: 1, 2: 2, 3: 3}


Suppose G1 and G2 are isomorphic directed graphs.
Verification is as follows:

>>> G1 = nx.path_graph(4, create_using=nx.DiGraph)
>>> G2 = nx.path_graph(4, create_using=nx.DiGraph)
>>> DiGM = isomorphism.DiGraphMatcher(G1, G2)
>>> DiGM.is_isomorphic()
True

DiGM.mapping stores the isomorphism mapping from G1 to G2.

>>> DiGM.mapping
{0: 0, 1: 1, 2: 2, 3: 3}



Subgraph Isomorphism
--------------------
Graph theory literature can be ambiguous about the meaning of the
above statement, and we seek to clarify it now.

In the VF2 literature, a mapping ``M`` is said to be a graph-subgraph
isomorphism iff ``M`` is an isomorphism between ``G2`` and a subgraph of ``G1``.
Thus, to say that ``G1`` and ``G2`` are graph-subgraph isomorphic is to say
that a subgraph of ``G1`` is isomorphic to ``G2``.

Other literature uses the phrase 'subgraph isomorphic' as in '``G1`` does
not have a subgraph isomorphic to ``G2``'.  Another use is as an in adverb
for isomorphic.  Thus, to say that ``G1`` and ``G2`` are subgraph isomorphic
is to say that a subgraph of ``G1`` is isomorphic to ``G2``.

Finally, the term 'subgraph' can have multiple meanings. In this
context, 'subgraph' always means a 'node-induced subgraph'. Edge-induced
subgraph isomorphisms are not directly supported, but one should be
able to perform the check by making use of
:func:`line_graph <networkx.generators.line.line_graph>`. For
subgraphs which are not induced, the term 'monomorphism' is preferred
over 'isomorphism'.

Let ``G = (N, E)`` be a graph with a set of nodes ``N`` and set of edges ``E``.

If ``G' = (N', E')`` is a subgraph, then:
    ``N'`` is a subset of ``N`` and
    ``E'`` is a subset of ``E``.

If ``G' = (N', E')`` is a node-induced subgraph, then:
    ``N'`` is a subset of ``N`` and
    ``E'`` is the subset of edges in ``E`` relating nodes in ``N'``.

If ``G' = (N', E')`` is an edge-induced subgraph, then:
    ``N'`` is the subset of nodes in ``N`` related by edges in ``E'`` and
    ``E'`` is a subset of ``E``.

If ``G' = (N', E')`` is a monomorphism, then:
    ``N'`` is a subset of ``N`` and
    ``E'`` is a subset of the set of edges in ``E`` relating nodes in ``N'``.

Note that if ``G'`` is a node-induced subgraph of ``G``, then it is always a
subgraph monomorphism of ``G``, but the opposite is not always true, as a
monomorphism can have fewer edges.

References
----------
[1]   Luigi P. Cordella, Pasquale Foggia, Carlo Sansone, Mario Vento,
      "A (Sub)Graph Isomorphism Algorithm for Matching Large Graphs",
      IEEE Transactions on Pattern Analysis and Machine Intelligence,
      vol. 26,  no. 10,  pp. 1367-1372,  Oct.,  2004.
      http://ieeexplore.ieee.org/iel5/34/29305/01323804.pdf

[2]   L. P. Cordella, P. Foggia, C. Sansone, M. Vento, "An Improved
      Algorithm for Matching Large Graphs", 3rd IAPR-TC15 Workshop
      on Graph-based Representations in Pattern Recognition, Cuen,
      pp. 149-159, 2001.
      https://www.researchgate.net/publication/200034365_An_Improved_Algorithm_for_Matching_Large_Graphs

See Also
--------
:meth:`semantic_feasibility <networkx.algorithms.isomorphism.GraphMatcher.semantic_feasibility>`
:meth:`syntactic_feasibility <networkx.algorithms.isomorphism.GraphMatcher.syntactic_feasibility>`

Notes
-----

The implementation handles both directed and undirected graphs as well
as multigraphs.

In general, the subgraph isomorphism problem is NP-complete whereas the
graph isomorphism problem is most likely not NP-complete (although no
polynomial-time algorithm is known to exist).

"""

# This work was originally coded by Christopher Ellison
# as part of the Computational Mechanics Python (CMPy) project.
# James P. Crutchfield, principal investigator.
# Complexity Sciences Center and Physics Department, UC Davis.

import sys

__all__ = ["GraphMatcher", "DiGraphMatcher"]


class GraphMatcher:
    """Implementation of VF2 algorithm for matching undirected graphs.

    Suitable for Graph and MultiGraph instances.
    """

    def __init__(self, G1, G2):
        """Initialize GraphMatcher.

        Parameters
        ----------
        G1,G2: NetworkX Graph or MultiGraph instances.
           The two graphs to check for isomorphism or monomorphism.

        Examples
        --------
        To create a GraphMatcher which checks for syntactic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> G1 = nx.path_graph(4)
        >>> G2 = nx.path_graph(4)
        >>> GM = isomorphism.GraphMatcher(G1, G2)
        """
        self.G1 = G1
        self.G2 = G2
        self.G1_nodes = set(G1.nodes())
        self.G2_nodes = set(G2.nodes())
        self.G2_node_order = {n: i for i, n in enumerate(G2)}

        # Set recursion limit.
        self.old_recursion_limit = sys.getrecursionlimit()
        expected_max_recursion_level = len(self.G2)
        if self.old_recursion_limit < 1.5 * expected_max_recursion_level:
            # Give some breathing room.
            sys.setrecursionlimit(int(1.5 * expected_max_recursion_level))

        # Declare that we will be searching for a graph-graph isomorphism.
        self.test = "graph"

        # Initialize state
        self.initialize()

    def reset_recursion_limit(self):
        """Restores the recursion limit."""
        # TODO:
        # Currently, we use recursion and set the recursion level higher.
        # It would be nice to restore the level, but because the
        # (Di)GraphMatcher classes make use of cyclic references, garbage
        # collection will never happen when we define __del__() to
        # restore the recursion level. The result is a memory leak.
        # So for now, we do not automatically restore the recursion level,
        # and instead provide a method to do this manually. Eventually,
        # we should turn this into a non-recursive implementation.
        sys.setrecursionlimit(self.old_recursion_limit)

    def candidate_pairs_iter(self):
        """Iterator over candidate pairs of nodes in G1 and G2."""

        # All computations are done using the current state!

        G1_nodes = self.G1_nodes
        G2_nodes = self.G2_nodes
        min_key = self.G2_node_order.__getitem__

        # First we compute the inout-terminal sets.
        T1_inout = [node for node in self.inout_1 if node not in self.core_1]
        T2_inout = [node for node in self.inout_2 if node not in self.core_2]

        # If T1_inout and T2_inout are both nonempty.
        # P(s) = T1_inout x {min T2_inout}
        if T1_inout and T2_inout:
            node_2 = min(T2_inout, key=min_key)
            for node_1 in T1_inout:
                yield node_1, node_2

        else:
            # If T1_inout and T2_inout were both empty....
            # P(s) = (N_1 - M_1) x {min (N_2 - M_2)}
            # if not (T1_inout or T2_inout):  # as suggested by  [2], incorrect
            if 1:  # as inferred from [1], correct
                # First we determine the candidate node for G2
                other_node = min(G2_nodes - set(self.core_2), key=min_key)
                for node in self.G1:
                    if node not in self.core_1:
                        yield node, other_node

        # For all other cases, we don't have any candidate pairs.

    def initialize(self):
        """Reinitializes the state of the algorithm.

        This method should be redefined if using something other than GMState.
        If only subclassing GraphMatcher, a redefinition is not necessary.

        """

        # core_1[n] contains the index of the node paired with n, which is m,
        #           provided n is in the mapping.
        # core_2[m] contains the index of the node paired with m, which is n,
        #           provided m is in the mapping.
        self.core_1 = {}
        self.core_2 = {}

        # See the paper for definitions of M_x and T_x^{y}

        # inout_1[n]  is non-zero if n is in M_1 or in T_1^{inout}
        # inout_2[m]  is non-zero if m is in M_2 or in T_2^{inout}
        #
        # The value stored is the depth of the SSR tree when the node became
        # part of the corresponding set.
        self.inout_1 = {}
        self.inout_2 = {}
        # Practically, these sets simply store the nodes in the subgraph.

        self.state = GMState(self)

        # Provide a convenient way to access the isomorphism mapping.
        self.mapping = self.core_1.copy()

    def is_isomorphic(self):
        """Returns True if G1 and G2 are isomorphic graphs."""

        # Let's do two very quick checks!
        # QUESTION: Should we call faster_graph_could_be_isomorphic(G1,G2)?
        # For now, I just copy the code.

        # Check global properties
        if self.G1.order() != self.G2.order():
            return False

        # Check local properties
        d1 = sorted(d for n, d in self.G1.degree())
        d2 = sorted(d for n, d in self.G2.degree())
        if d1 != d2:
            return False

        try:
            x = next(self.isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def isomorphisms_iter(self):
        """Generator over isomorphisms between G1 and G2."""
        # Declare that we are looking for a graph-graph isomorphism.
        self.test = "graph"
        self.initialize()
        yield from self.match()

    def match(self):
        """Extends the isomorphism mapping.

        This function is called recursively to determine if a complete
        isomorphism can be found between G1 and G2.  It cleans up the class
        variables after each recursive call. If an isomorphism is found,
        we yield the mapping.

        """
        if len(self.core_1) == len(self.G2):
            # Save the final mapping, otherwise garbage collection deletes it.
            self.mapping = self.core_1.copy()
            # The mapping is complete.
            yield self.mapping
        else:
            for G1_node, G2_node in self.candidate_pairs_iter():
                if self.syntactic_feasibility(G1_node, G2_node):
                    if self.semantic_feasibility(G1_node, G2_node):
                        # Recursive call, adding the feasible state.
                        newstate = self.state.__class__(self, G1_node, G2_node)
                        yield from self.match()

                        # restore data structures
                        newstate.restore()

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is semantically feasible.

        The semantic feasibility function should return True if it is
        acceptable to add the candidate pair (G1_node, G2_node) to the current
        partial isomorphism mapping.   The logic should focus on semantic
        information contained in the edge data or a formalized node class.

        By acceptable, we mean that the subsequent mapping can still become a
        complete isomorphism mapping.  Thus, if adding the candidate pair
        definitely makes it so that the subsequent mapping cannot become a
        complete isomorphism mapping, then this function must return False.

        The default semantic feasibility function always returns True. The
        effect is that semantics are not considered in the matching of G1
        and G2.

        The semantic checks might differ based on the what type of test is
        being performed.  A keyword description of the test is stored in
        self.test.  Here is a quick description of the currently implemented
        tests::

          test='graph'
            Indicates that the graph matcher is looking for a graph-graph
            isomorphism.

          test='subgraph'
            Indicates that the graph matcher is looking for a subgraph-graph
            isomorphism such that a subgraph of G1 is isomorphic to G2.

          test='mono'
            Indicates that the graph matcher is looking for a subgraph-graph
            monomorphism such that a subgraph of G1 is monomorphic to G2.

        Any subclass which redefines semantic_feasibility() must maintain
        the above form to keep the match() method functional. Implementations
        should consider multigraphs.
        """
        return True

    def subgraph_is_isomorphic(self):
        """Returns `True` if a subgraph of ``G1`` is isomorphic to ``G2``.

        Examples
        --------
        When creating the `GraphMatcher`, the order of the arguments is important

        >>> G = nx.Graph([("A", "B"), ("B", "C"), ("A", "C")])
        >>> H = nx.Graph([(0, 1), (1, 2), (0, 2), (1, 3), (0, 4)])

        Check whether a subgraph of G is isomorphic to H:

        >>> isomatcher = nx.isomorphism.GraphMatcher(G, H)
        >>> isomatcher.subgraph_is_isomorphic()
        False

        Check whether a subgraph of H is isomorphic to G:

        >>> isomatcher = nx.isomorphism.GraphMatcher(H, G)
        >>> isomatcher.subgraph_is_isomorphic()
        True
        """
        try:
            x = next(self.subgraph_isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def subgraph_is_monomorphic(self):
        """Returns `True` if a subgraph of ``G1`` is monomorphic to ``G2``.

        Examples
        --------
        When creating the `GraphMatcher`, the order of the arguments is important.

        >>> G = nx.Graph([("A", "B"), ("B", "C")])
        >>> H = nx.Graph([(0, 1), (1, 2), (0, 2)])

        Check whether a subgraph of G is monomorphic to H:

        >>> isomatcher = nx.isomorphism.GraphMatcher(G, H)
        >>> isomatcher.subgraph_is_monomorphic()
        False

        Check whether a subgraph of H is isomorphic to G:

        >>> isomatcher = nx.isomorphism.GraphMatcher(H, G)
        >>> isomatcher.subgraph_is_monomorphic()
        True
        """
        try:
            x = next(self.subgraph_monomorphisms_iter())
            return True
        except StopIteration:
            return False

    def subgraph_isomorphisms_iter(self):
        """Generator over isomorphisms between a subgraph of ``G1`` and ``G2``.

        Examples
        --------
        When creating the `GraphMatcher`, the order of the arguments is important

        >>> G = nx.Graph([("A", "B"), ("B", "C"), ("A", "C")])
        >>> H = nx.Graph([(0, 1), (1, 2), (0, 2), (1, 3), (0, 4)])

        Yield isomorphic mappings between ``H`` and subgraphs of ``G``:

        >>> isomatcher = nx.isomorphism.GraphMatcher(G, H)
        >>> list(isomatcher.subgraph_isomorphisms_iter())
        []

        Yield isomorphic mappings  between ``G`` and subgraphs of ``H``:

        >>> isomatcher = nx.isomorphism.GraphMatcher(H, G)
        >>> next(isomatcher.subgraph_isomorphisms_iter())
        {0: 'A', 1: 'B', 2: 'C'}

        """
        # Declare that we are looking for graph-subgraph isomorphism.
        self.test = "subgraph"
        self.initialize()
        yield from self.match()

    def subgraph_monomorphisms_iter(self):
        """Generator over monomorphisms between a subgraph of ``G1`` and ``G2``.

        Examples
        --------
        When creating the `GraphMatcher`, the order of the arguments is important.

        >>> G = nx.Graph([("A", "B"), ("B", "C")])
        >>> H = nx.Graph([(0, 1), (1, 2), (0, 2)])

        Yield monomorphic mappings between ``H`` and subgraphs of ``G``:

        >>> isomatcher = nx.isomorphism.GraphMatcher(G, H)
        >>> list(isomatcher.subgraph_monomorphisms_iter())
        []

        Yield monomorphic mappings  between ``G`` and subgraphs of ``H``:

        >>> isomatcher = nx.isomorphism.GraphMatcher(H, G)
        >>> next(isomatcher.subgraph_monomorphisms_iter())
        {0: 'A', 1: 'B', 2: 'C'}
        """
        # Declare that we are looking for graph-subgraph monomorphism.
        self.test = "mono"
        self.initialize()
        yield from self.match()

    def syntactic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is syntactically feasible.

        This function returns True if it is adding the candidate pair
        to the current partial isomorphism/monomorphism mapping is allowable.
        The addition is allowable if the inclusion of the candidate pair does
        not make it impossible for an isomorphism/monomorphism to be found.
        """

        # The VF2 algorithm was designed to work with graphs having, at most,
        # one edge connecting any two nodes.  This is not the case when
        # dealing with an MultiGraphs.
        #
        # Basically, when we test the look-ahead rules R_neighbor, we will
        # make sure that the number of edges are checked. We also add
        # a R_self check to verify that the number of selfloops is acceptable.
        #
        # Users might be comparing Graph instances with MultiGraph instances.
        # So the generic GraphMatcher class must work with MultiGraphs.
        # Care must be taken since the value in the innermost dictionary is a
        # singlet for Graph instances.  For MultiGraphs, the value in the
        # innermost dictionary is a list.

        ###
        # Test at each step to get a return value as soon as possible.
        ###

        # Look ahead 0

        # R_self

        # The number of selfloops for G1_node must equal the number of
        # self-loops for G2_node. Without this check, we would fail on
        # R_neighbor at the next recursion level. But it is good to prune the
        # search tree now.

        if self.test == "mono":
            if self.G1.number_of_edges(G1_node, G1_node) < self.G2.number_of_edges(
                G2_node, G2_node
            ):
                return False
        else:
            if self.G1.number_of_edges(G1_node, G1_node) != self.G2.number_of_edges(
                G2_node, G2_node
            ):
                return False

        # R_neighbor

        # For each neighbor n' of n in the partial mapping, the corresponding
        # node m' is a neighbor of m, and vice versa. Also, the number of
        # edges must be equal.
        if self.test != "mono":
            for neighbor in self.G1[G1_node]:
                if neighbor in self.core_1:
                    if self.core_1[neighbor] not in self.G2[G2_node]:
                        return False
                    elif self.G1.number_of_edges(
                        neighbor, G1_node
                    ) != self.G2.number_of_edges(self.core_1[neighbor], G2_node):
                        return False

        for neighbor in self.G2[G2_node]:
            if neighbor in self.core_2:
                if self.core_2[neighbor] not in self.G1[G1_node]:
                    return False
                elif self.test == "mono":
                    if self.G1.number_of_edges(
                        self.core_2[neighbor], G1_node
                    ) < self.G2.number_of_edges(neighbor, G2_node):
                        return False
                else:
                    if self.G1.number_of_edges(
                        self.core_2[neighbor], G1_node
                    ) != self.G2.number_of_edges(neighbor, G2_node):
                        return False

        if self.test != "mono":
            # Look ahead 1

            # R_terminout
            # The number of neighbors of n in T_1^{inout} is equal to the
            # number of neighbors of m that are in T_2^{inout}, and vice versa.
            num1 = 0
            for neighbor in self.G1[G1_node]:
                if (neighbor in self.inout_1) and (neighbor not in self.core_1):
                    num1 += 1
            num2 = 0
            for neighbor in self.G2[G2_node]:
                if (neighbor in self.inout_2) and (neighbor not in self.core_2):
                    num2 += 1
            if self.test == "graph":
                if num1 != num2:
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # Look ahead 2

            # R_new

            # The number of neighbors of n that are neither in the core_1 nor
            # T_1^{inout} is equal to the number of neighbors of m
            # that are neither in core_2 nor T_2^{inout}.
            num1 = 0
            for neighbor in self.G1[G1_node]:
                if neighbor not in self.inout_1:
                    num1 += 1
            num2 = 0
            for neighbor in self.G2[G2_node]:
                if neighbor not in self.inout_2:
                    num2 += 1
            if self.test == "graph":
                if num1 != num2:
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

        # Otherwise, this node pair is syntactically feasible!
        return True


class DiGraphMatcher(GraphMatcher):
    """Implementation of VF2 algorithm for matching directed graphs.

    Suitable for DiGraph and MultiDiGraph instances.
    """

    def __init__(self, G1, G2):
        """Initialize DiGraphMatcher.

        G1 and G2 should be nx.Graph or nx.MultiGraph instances.

        Examples
        --------
        To create a GraphMatcher which checks for syntactic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> G1 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))
        >>> G2 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))
        >>> DiGM = isomorphism.DiGraphMatcher(G1, G2)
        """
        super().__init__(G1, G2)

    def candidate_pairs_iter(self):
        """Iterator over candidate pairs of nodes in G1 and G2."""

        # All computations are done using the current state!

        G1_nodes = self.G1_nodes
        G2_nodes = self.G2_nodes
        min_key = self.G2_node_order.__getitem__

        # First we compute the out-terminal sets.
        T1_out = [node for node in self.out_1 if node not in self.core_1]
        T2_out = [node for node in self.out_2 if node not in self.core_2]

        # If T1_out and T2_out are both nonempty.
        # P(s) = T1_out x {min T2_out}
        if T1_out and T2_out:
            node_2 = min(T2_out, key=min_key)
            for node_1 in T1_out:
                yield node_1, node_2

        # If T1_out and T2_out were both empty....
        # We compute the in-terminal sets.

        # elif not (T1_out or T2_out):   # as suggested by [2], incorrect
        else:  # as suggested by [1], correct
            T1_in = [node for node in self.in_1 if node not in self.core_1]
            T2_in = [node for node in self.in_2 if node not in self.core_2]

            # If T1_in and T2_in are both nonempty.
            # P(s) = T1_out x {min T2_out}
            if T1_in and T2_in:
                node_2 = min(T2_in, key=min_key)
                for node_1 in T1_in:
                    yield node_1, node_2

            # If all terminal sets are empty...
            # P(s) = (N_1 - M_1) x {min (N_2 - M_2)}

            # elif not (T1_in or T2_in):   # as suggested by  [2], incorrect
            else:  # as inferred from [1], correct
                node_2 = min(G2_nodes - set(self.core_2), key=min_key)
                for node_1 in G1_nodes:
                    if node_1 not in self.core_1:
                        yield node_1, node_2

        # For all other cases, we don't have any candidate pairs.

    def initialize(self):
        """Reinitializes the state of the algorithm.

        This method should be redefined if using something other than DiGMState.
        If only subclassing GraphMatcher, a redefinition is not necessary.
        """

        # core_1[n] contains the index of the node paired with n, which is m,
        #           provided n is in the mapping.
        # core_2[m] contains the index of the node paired with m, which is n,
        #           provided m is in the mapping.
        self.core_1 = {}
        self.core_2 = {}

        # See the paper for definitions of M_x and T_x^{y}

        # in_1[n]  is non-zero if n is in M_1 or in T_1^{in}
        # out_1[n] is non-zero if n is in M_1 or in T_1^{out}
        #
        # in_2[m]  is non-zero if m is in M_2 or in T_2^{in}
        # out_2[m] is non-zero if m is in M_2 or in T_2^{out}
        #
        # The value stored is the depth of the search tree when the node became
        # part of the corresponding set.
        self.in_1 = {}
        self.in_2 = {}
        self.out_1 = {}
        self.out_2 = {}

        self.state = DiGMState(self)

        # Provide a convenient way to access the isomorphism mapping.
        self.mapping = self.core_1.copy()

    def syntactic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is syntactically feasible.

        This function returns True if it is adding the candidate pair
        to the current partial isomorphism/monomorphism mapping is allowable.
        The addition is allowable if the inclusion of the candidate pair does
        not make it impossible for an isomorphism/monomorphism to be found.
        """

        # The VF2 algorithm was designed to work with graphs having, at most,
        # one edge connecting any two nodes.  This is not the case when
        # dealing with an MultiGraphs.
        #
        # Basically, when we test the look-ahead rules R_pred and R_succ, we
        # will make sure that the number of edges are checked.  We also add
        # a R_self check to verify that the number of selfloops is acceptable.

        # Users might be comparing DiGraph instances with MultiDiGraph
        # instances. So the generic DiGraphMatcher class must work with
        # MultiDiGraphs. Care must be taken since the value in the innermost
        # dictionary is a singlet for DiGraph instances.  For MultiDiGraphs,
        # the value in the innermost dictionary is a list.

        ###
        # Test at each step to get a return value as soon as possible.
        ###

        # Look ahead 0

        # R_self

        # The number of selfloops for G1_node must equal the number of
        # self-loops for G2_node. Without this check, we would fail on R_pred
        # at the next recursion level. This should prune the tree even further.
        if self.test == "mono":
            if self.G1.number_of_edges(G1_node, G1_node) < self.G2.number_of_edges(
                G2_node, G2_node
            ):
                return False
        else:
            if self.G1.number_of_edges(G1_node, G1_node) != self.G2.number_of_edges(
                G2_node, G2_node
            ):
                return False

        # R_pred

        # For each predecessor n' of n in the partial mapping, the
        # corresponding node m' is a predecessor of m, and vice versa. Also,
        # the number of edges must be equal
        if self.test != "mono":
            for predecessor in self.G1.pred[G1_node]:
                if predecessor in self.core_1:
                    if self.core_1[predecessor] not in self.G2.pred[G2_node]:
                        return False
                    elif self.G1.number_of_edges(
                        predecessor, G1_node
                    ) != self.G2.number_of_edges(self.core_1[predecessor], G2_node):
                        return False

        for predecessor in self.G2.pred[G2_node]:
            if predecessor in self.core_2:
                if self.core_2[predecessor] not in self.G1.pred[G1_node]:
                    return False
                elif self.test == "mono":
                    if self.G1.number_of_edges(
                        self.core_2[predecessor], G1_node
                    ) < self.G2.number_of_edges(predecessor, G2_node):
                        return False
                else:
                    if self.G1.number_of_edges(
                        self.core_2[predecessor], G1_node
                    ) != self.G2.number_of_edges(predecessor, G2_node):
                        return False

        # R_succ

        # For each successor n' of n in the partial mapping, the corresponding
        # node m' is a successor of m, and vice versa. Also, the number of
        # edges must be equal.
        if self.test != "mono":
            for successor in self.G1[G1_node]:
                if successor in self.core_1:
                    if self.core_1[successor] not in self.G2[G2_node]:
                        return False
                    elif self.G1.number_of_edges(
                        G1_node, successor
                    ) != self.G2.number_of_edges(G2_node, self.core_1[successor]):
                        return False

        for successor in self.G2[G2_node]:
            if successor in self.core_2:
                if self.core_2[successor] not in self.G1[G1_node]:
                    return False
                elif self.test == "mono":
                    if self.G1.number_of_edges(
                        G1_node, self.core_2[successor]
                    ) < self.G2.number_of_edges(G2_node, successor):
                        return False
                else:
                    if self.G1.number_of_edges(
                        G1_node, self.core_2[successor]
                    ) != self.G2.number_of_edges(G2_node, successor):
                        return False

        if self.test != "mono":
            # Look ahead 1

            # R_termin
            # The number of predecessors of n that are in T_1^{in} is equal to the
            # number of predecessors of m that are in T_2^{in}.
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if (predecessor in self.in_1) and (predecessor not in self.core_1):
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if (predecessor in self.in_2) and (predecessor not in self.core_2):
                    num2 += 1
            if self.test == "graph":
                if num1 != num2:
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # The number of successors of n that are in T_1^{in} is equal to the
            # number of successors of m that are in T_2^{in}.
            num1 = 0
            for successor in self.G1[G1_node]:
                if (successor in self.in_1) and (successor not in self.core_1):
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if (successor in self.in_2) and (successor not in self.core_2):
                    num2 += 1
            if self.test == "graph":
                if num1 != num2:
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # R_termout

            # The number of predecessors of n that are in T_1^{out} is equal to the
            # number of predecessors of m that are in T_2^{out}.
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if (predecessor in self.out_1) and (predecessor not in self.core_1):
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if (predecessor in self.out_2) and (predecessor not in self.core_2):
                    num2 += 1
            if self.test == "graph":
                if num1 != num2:
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # The number of successors of n that are in T_1^{out} is equal to the
            # number of successors of m that are in T_2^{out}.
            num1 = 0
            for successor in self.G1[G1_node]:
                if (successor in self.out_1) and (successor not in self.core_1):
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if (successor in self.out_2) and (successor not in self.core_2):
                    num2 += 1
            if self.test == "graph":
                if num1 != num2:
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # Look ahead 2

            # R_new

            # The number of predecessors of n that are neither in the core_1 nor
            # T_1^{in} nor T_1^{out} is equal to the number of predecessors of m
            # that are neither in core_2 nor T_2^{in} nor T_2^{out}.
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if (predecessor not in self.in_1) and (predecessor not in self.out_1):
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if (predecessor not in self.in_2) and (predecessor not in self.out_2):
                    num2 += 1
            if self.test == "graph":
                if num1 != num2:
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # The number of successors of n that are neither in the core_1 nor
            # T_1^{in} nor T_1^{out} is equal to the number of successors of m
            # that are neither in core_2 nor T_2^{in} nor T_2^{out}.
            num1 = 0
            for successor in self.G1[G1_node]:
                if (successor not in self.in_1) and (successor not in self.out_1):
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if (successor not in self.in_2) and (successor not in self.out_2):
                    num2 += 1
            if self.test == "graph":
                if num1 != num2:
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

        # Otherwise, this node pair is syntactically feasible!
        return True

    def subgraph_is_isomorphic(self):
        """Returns `True` if a subgraph of ``G1`` is isomorphic to ``G2``.

        Examples
        --------
        When creating the `DiGraphMatcher`, the order of the arguments is important

        >>> G = nx.DiGraph([("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")])
        >>> H = nx.DiGraph(nx.path_graph(5))

        Check whether a subgraph of G is isomorphic to H:

        >>> isomatcher = nx.isomorphism.DiGraphMatcher(G, H)
        >>> isomatcher.subgraph_is_isomorphic()
        False

        Check whether a subgraph of H is isomorphic to G:

        >>> isomatcher = nx.isomorphism.DiGraphMatcher(H, G)
        >>> isomatcher.subgraph_is_isomorphic()
        True
        """
        return super().subgraph_is_isomorphic()

    def subgraph_is_monomorphic(self):
        """Returns `True` if a subgraph of ``G1`` is monomorphic to ``G2``.

        Examples
        --------
        When creating the `DiGraphMatcher`, the order of the arguments is important.

        >>> G = nx.DiGraph([("A", "B"), ("C", "B"), ("D", "C")])
        >>> H = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 2)])

        Check whether a subgraph of G is monomorphic to H:

        >>> isomatcher = nx.isomorphism.DiGraphMatcher(G, H)
        >>> isomatcher.subgraph_is_monomorphic()
        False

        Check whether a subgraph of H is isomorphic to G:

        >>> isomatcher = nx.isomorphism.DiGraphMatcher(H, G)
        >>> isomatcher.subgraph_is_monomorphic()
        True
        """
        return super().subgraph_is_monomorphic()

    def subgraph_isomorphisms_iter(self):
        """Generator over isomorphisms between a subgraph of ``G1`` and ``G2``.

        Examples
        --------
        When creating the `DiGraphMatcher`, the order of the arguments is important

        >>> G = nx.DiGraph([("B", "C"), ("C", "B"), ("C", "D"), ("D", "C")])
        >>> H = nx.DiGraph(nx.path_graph(5))

        Yield isomorphic mappings between ``H`` and subgraphs of ``G``:

        >>> isomatcher = nx.isomorphism.DiGraphMatcher(G, H)
        >>> list(isomatcher.subgraph_isomorphisms_iter())
        []

        Yield isomorphic mappings between ``G`` and subgraphs of ``H``:

        >>> isomatcher = nx.isomorphism.DiGraphMatcher(H, G)
        >>> next(isomatcher.subgraph_isomorphisms_iter())
        {0: 'B', 1: 'C', 2: 'D'}
        """
        return super().subgraph_isomorphisms_iter()

    def subgraph_monomorphisms_iter(self):
        """Generator over monomorphisms between a subgraph of ``G1`` and ``G2``.

        Examples
        --------
        When creating the `DiGraphMatcher`, the order of the arguments is important.

        >>> G = nx.DiGraph([("A", "B"), ("C", "B"), ("D", "C")])
        >>> H = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 2)])

        Yield monomorphic mappings between ``H`` and subgraphs of ``G``:

        >>> isomatcher = nx.isomorphism.DiGraphMatcher(G, H)
        >>> list(isomatcher.subgraph_monomorphisms_iter())
        []

        Yield monomorphic mappings between ``G`` and subgraphs of ``H``:

        >>> isomatcher = nx.isomorphism.DiGraphMatcher(H, G)
        >>> next(isomatcher.subgraph_monomorphisms_iter())
        {3: 'A', 2: 'B', 1: 'C', 0: 'D'}
        """
        return super().subgraph_monomorphisms_iter()


class GMState:
    """Internal representation of state for the GraphMatcher class.

    This class is used internally by the GraphMatcher class.  It is used
    only to store state specific data. There will be at most G2.order() of
    these objects in memory at a time, due to the depth-first search
    strategy employed by the VF2 algorithm.
    """

    def __init__(self, GM, G1_node=None, G2_node=None):
        """Initializes GMState object.

        Pass in the GraphMatcher to which this GMState belongs and the
        new node pair that will be added to the GraphMatcher's current
        isomorphism mapping.
        """
        self.GM = GM

        # Initialize the last stored node pair.
        self.G1_node = None
        self.G2_node = None
        self.depth = len(GM.core_1)

        if G1_node is None or G2_node is None:
            # Then we reset the class variables
            GM.core_1 = {}
            GM.core_2 = {}
            GM.inout_1 = {}
            GM.inout_2 = {}

        # Watch out! G1_node == 0 should evaluate to True.
        if G1_node is not None and G2_node is not None:
            # Add the node pair to the isomorphism mapping.
            GM.core_1[G1_node] = G2_node
            GM.core_2[G2_node] = G1_node

            # Store the node that was added last.
            self.G1_node = G1_node
            self.G2_node = G2_node

            # Now we must update the other two vectors.
            # We will add only if it is not in there already!
            self.depth = len(GM.core_1)

            # First we add the new nodes...
            if G1_node not in GM.inout_1:
                GM.inout_1[G1_node] = self.depth
            if G2_node not in GM.inout_2:
                GM.inout_2[G2_node] = self.depth

            # Now we add every other node...

            # Updates for T_1^{inout}
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update(
                    [neighbor for neighbor in GM.G1[node] if neighbor not in GM.core_1]
                )
            for node in new_nodes:
                if node not in GM.inout_1:
                    GM.inout_1[node] = self.depth

            # Updates for T_2^{inout}
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update(
                    [neighbor for neighbor in GM.G2[node] if neighbor not in GM.core_2]
                )
            for node in new_nodes:
                if node not in GM.inout_2:
                    GM.inout_2[node] = self.depth

    def restore(self):
        """Deletes the GMState object and restores the class variables."""
        # First we remove the node that was added from the core vectors.
        # Watch out! G1_node == 0 should evaluate to True.
        if self.G1_node is not None and self.G2_node is not None:
            del self.GM.core_1[self.G1_node]
            del self.GM.core_2[self.G2_node]

        # Now we revert the other two vectors.
        # Thus, we delete all entries which have this depth level.
        for vector in (self.GM.inout_1, self.GM.inout_2):
            for node in list(vector.keys()):
                if vector[node] == self.depth:
                    del vector[node]


class DiGMState:
    """Internal representation of state for the DiGraphMatcher class.

    This class is used internally by the DiGraphMatcher class.  It is used
    only to store state specific data. There will be at most G2.order() of
    these objects in memory at a time, due to the depth-first search
    strategy employed by the VF2 algorithm.

    """

    def __init__(self, GM, G1_node=None, G2_node=None):
        """Initializes DiGMState object.

        Pass in the DiGraphMatcher to which this DiGMState belongs and the
        new node pair that will be added to the GraphMatcher's current
        isomorphism mapping.
        """
        self.GM = GM

        # Initialize the last stored node pair.
        self.G1_node = None
        self.G2_node = None
        self.depth = len(GM.core_1)

        if G1_node is None or G2_node is None:
            # Then we reset the class variables
            GM.core_1 = {}
            GM.core_2 = {}
            GM.in_1 = {}
            GM.in_2 = {}
            GM.out_1 = {}
            GM.out_2 = {}

        # Watch out! G1_node == 0 should evaluate to True.
        if G1_node is not None and G2_node is not None:
            # Add the node pair to the isomorphism mapping.
            GM.core_1[G1_node] = G2_node
            GM.core_2[G2_node] = G1_node

            # Store the node that was added last.
            self.G1_node = G1_node
            self.G2_node = G2_node

            # Now we must update the other four vectors.
            # We will add only if it is not in there already!
            self.depth = len(GM.core_1)

            # First we add the new nodes...
            for vector in (GM.in_1, GM.out_1):
                if G1_node not in vector:
                    vector[G1_node] = self.depth
            for vector in (GM.in_2, GM.out_2):
                if G2_node not in vector:
                    vector[G2_node] = self.depth

            # Now we add every other node...

            # Updates for T_1^{in}
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update(
                    [
                        predecessor
                        for predecessor in GM.G1.predecessors(node)
                        if predecessor not in GM.core_1
                    ]
                )
            for node in new_nodes:
                if node not in GM.in_1:
                    GM.in_1[node] = self.depth

            # Updates for T_2^{in}
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update(
                    [
                        predecessor
                        for predecessor in GM.G2.predecessors(node)
                        if predecessor not in GM.core_2
                    ]
                )
            for node in new_nodes:
                if node not in GM.in_2:
                    GM.in_2[node] = self.depth

            # Updates for T_1^{out}
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update(
                    [
                        successor
                        for successor in GM.G1.successors(node)
                        if successor not in GM.core_1
                    ]
                )
            for node in new_nodes:
                if node not in GM.out_1:
                    GM.out_1[node] = self.depth

            # Updates for T_2^{out}
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update(
                    [
                        successor
                        for successor in GM.G2.successors(node)
                        if successor not in GM.core_2
                    ]
                )
            for node in new_nodes:
                if node not in GM.out_2:
                    GM.out_2[node] = self.depth

    def restore(self):
        """Deletes the DiGMState object and restores the class variables."""

        # First we remove the node that was added from the core vectors.
        # Watch out! G1_node == 0 should evaluate to True.
        if self.G1_node is not None and self.G2_node is not None:
            del self.GM.core_1[self.G1_node]
            del self.GM.core_2[self.G2_node]

        # Now we revert the other four vectors.
        # Thus, we delete all entries which have this depth level.
        for vector in (self.GM.in_1, self.GM.in_2, self.GM.out_1, self.GM.out_2):
            for node in list(vector.keys()):
                if vector[node] == self.depth:
                    del vector[node]
<|endoftext|>"""
Operations on graphs including union, intersection, difference.
"""

import networkx as nx

__all__ = [
    "union",
    "compose",
    "disjoint_union",
    "intersection",
    "difference",
    "symmetric_difference",
    "full_join",
]
_G_H = {"G": 0, "H": 1}


@nx._dispatchable(graphs=_G_H, preserve_all_attrs=True, returns_graph=True)
def union(G, H, rename=()):
    """Combine graphs G and H. The names of nodes must be unique.

    A name collision between the graphs will raise an exception.

    A renaming facility is provided to avoid name collisions.


    Parameters
    ----------
    G, H : graph
       A NetworkX graph

    rename : iterable , optional
       Node names of G and H can be changed by specifying the tuple
       rename=('G-','H-') (for example).  Node "u" in G is then renamed
       "G-u" and "v" in H is renamed "H-v".

    Returns
    -------
    U : A union graph with the same type as G.

    See Also
    --------
    compose
    :func:`~networkx.Graph.update`
    disjoint_union

    Notes
    -----
    To combine graphs that have common nodes, consider compose(G, H)
    or the method, Graph.update().

    disjoint_union() is similar to union() except that it avoids name clashes
    by relabeling the nodes with sequential integers.

    Edge and node attributes are propagated from G and H to the union graph.
    Graph attributes are also propagated, but if they are present in both G and H,
    then the value from H is used.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2)])
    >>> H = nx.Graph([(0, 1), (0, 3), (1, 3), (1, 2)])
    >>> U = nx.union(G, H, rename=("G", "H"))
    >>> U.nodes
    NodeView(('G0', 'G1', 'G2', 'H0', 'H1', 'H3', 'H2'))
    >>> U.edges
    EdgeView([('G0', 'G1'), ('G0', 'G2'), ('G1', 'G2'), ('H0', 'H1'), ('H0', 'H3'), ('H1', 'H3'), ('H1', 'H2')])


    """
    return nx.union_all([G, H], rename)


@nx._dispatchable(graphs=_G_H, preserve_all_attrs=True, returns_graph=True)
def disjoint_union(G, H):
    """Combine graphs G and H. The nodes are assumed to be unique (disjoint).

    This algorithm automatically relabels nodes to avoid name collisions.

    Parameters
    ----------
    G,H : graph
       A NetworkX graph

    Returns
    -------
    U : A union graph with the same type as G.

    See Also
    --------
    union
    compose
    :func:`~networkx.Graph.update`

    Notes
    -----
    A new graph is created, of the same class as G.  It is recommended
    that G and H be either both directed or both undirected.

    The nodes of G are relabeled 0 to len(G)-1, and the nodes of H are
    relabeled len(G) to len(G)+len(H)-1.

    Renumbering forces G and H to be disjoint, so no exception is ever raised for a name collision.
    To preserve the check for common nodes, use union().

    Edge and node attributes are propagated from G and H to the union graph.
    Graph attributes are also propagated, but if they are present in both G and H,
    then the value from H is used.

    To combine graphs that have common nodes, consider compose(G, H)
    or the method, Graph.update().

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2)])
    >>> H = nx.Graph([(0, 3), (1, 2), (2, 3)])
    >>> G.nodes[0]["key1"] = 5
    >>> H.nodes[0]["key2"] = 10
    >>> U = nx.disjoint_union(G, H)
    >>> U.nodes(data=True)
    NodeDataView({0: {'key1': 5}, 1: {}, 2: {}, 3: {'key2': 10}, 4: {}, 5: {}, 6: {}})
    >>> U.edges
    EdgeView([(0, 1), (0, 2), (1, 2), (3, 4), (4, 6), (5, 6)])
    """
    return nx.disjoint_union_all([G, H])


@nx._dispatchable(graphs=_G_H, returns_graph=True)
def intersection(G, H):
    """Returns a new graph that contains only the nodes and the edges that exist in
    both G and H.

    Parameters
    ----------
    G,H : graph
       A NetworkX graph. G and H can have different node sets but must be both graphs or both multigraphs.

    Raises
    ------
    NetworkXError
        If one is a MultiGraph and the other one is a graph.

    Returns
    -------
    GH : A new graph with the same type as G.

    Notes
    -----
    Attributes from the graph, nodes, and edges are not copied to the new
    graph.  If you want a new graph of the intersection of G and H
    with the attributes (including edge data) from G use remove_nodes_from()
    as follows

    >>> G = nx.path_graph(3)
    >>> H = nx.path_graph(5)
    >>> R = G.copy()
    >>> R.remove_nodes_from(n for n in G if n not in H)
    >>> R.remove_edges_from(e for e in G.edges if e not in H.edges)

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2)])
    >>> H = nx.Graph([(0, 3), (1, 2), (2, 3)])
    >>> R = nx.intersection(G, H)
    >>> R.nodes
    NodeView((0, 1, 2))
    >>> R.edges
    EdgeView([(1, 2)])
    """
    return nx.intersection_all([G, H])


@nx._dispatchable(graphs=_G_H, returns_graph=True)
def difference(G, H):
    """Returns a new graph that contains the edges that exist in G but not in H.

    The node sets of H and G must be the same.

    Parameters
    ----------
    G,H : graph
       A NetworkX graph. G and H must have the same node sets.

    Returns
    -------
    D : A new graph with the same type as G.

    Notes
    -----
    Attributes from the graph, nodes, and edges are not copied to the new
    graph.  If you want a new graph of the difference of G and H with
    the attributes (including edge data) from G use remove_nodes_from()
    as follows:

    >>> G = nx.path_graph(3)
    >>> H = nx.path_graph(5)
    >>> R = G.copy()
    >>> R.remove_nodes_from(n for n in G if n in H)

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3)])
    >>> H = nx.Graph([(0, 1), (1, 2), (0, 3)])
    >>> R = nx.difference(G, H)
    >>> R.nodes
    NodeView((0, 1, 2, 3))
    >>> R.edges
    EdgeView([(0, 2), (1, 3)])
    """
    # create new graph
    if not G.is_multigraph() == H.is_multigraph():
        raise nx.NetworkXError("G and H must both be graphs or multigraphs.")
    R = nx.create_empty_copy(G, with_data=False)

    if set(G) != set(H):
        raise nx.NetworkXError("Node sets of graphs not equal")

    if G.is_multigraph():
        edges = G.edges(keys=True)
    else:
        edges = G.edges()
    for e in edges:
        if not H.has_edge(*e):
            R.add_edge(*e)
    return R


@nx._dispatchable(graphs=_G_H, returns_graph=True)
def symmetric_difference(G, H):
    """Returns new graph with edges that exist in either G or H but not both.

    The node sets of H and G must be the same.

    Parameters
    ----------
    G,H : graph
       A NetworkX graph.  G and H must have the same node sets.

    Returns
    -------
    D : A new graph with the same type as G.

    Notes
    -----
    Attributes from the graph, nodes, and edges are not copied to the new
    graph.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3)])
    >>> H = nx.Graph([(0, 1), (1, 2), (0, 3)])
    >>> R = nx.symmetric_difference(G, H)
    >>> R.nodes
    NodeView((0, 1, 2, 3))
    >>> R.edges
    EdgeView([(0, 2), (0, 3), (1, 3)])
    """
    # create new graph
    if not G.is_multigraph() == H.is_multigraph():
        raise nx.NetworkXError("G and H must both be graphs or multigraphs.")
    R = nx.create_empty_copy(G, with_data=False)

    if set(G) != set(H):
        raise nx.NetworkXError("Node sets of graphs not equal")

    gnodes = set(G)  # set of nodes in G
    hnodes = set(H)  # set of nodes in H
    nodes = gnodes.symmetric_difference(hnodes)
    R.add_nodes_from(nodes)

    if G.is_multigraph():
        edges = G.edges(keys=True)
    else:
        edges = G.edges()
    # we could copy the data here but then this function doesn't
    # match intersection and difference
    for e in edges:
        if not H.has_edge(*e):
            R.add_edge(*e)

    if H.is_multigraph():
        edges = H.edges(keys=True)
    else:
        edges = H.edges()
    for e in edges:
        if not G.has_edge(*e):
            R.add_edge(*e)
    return R


@nx._dispatchable(graphs=_G_H, preserve_all_attrs=True, returns_graph=True)
def compose(G, H):
    """Compose graph G with H by combining nodes and edges into a single graph.

    The node sets and edges sets do not need to be disjoint.

    Composing preserves the attributes of nodes and edges.
    Attribute values from H take precedent over attribute values from G.

    Parameters
    ----------
    G, H : graph
       A NetworkX graph

    Returns
    -------
    C: A new graph with the same type as G

    See Also
    --------
    :func:`~networkx.Graph.update`
    union
    disjoint_union

    Notes
    -----
    It is recommended that G and H be either both directed or both undirected.

    For MultiGraphs, the edges are identified by incident nodes AND edge-key.
    This can cause surprises (i.e., edge `(1, 2)` may or may not be the same
    in two graphs) if you use MultiGraph without keeping track of edge keys.

    If combining the attributes of common nodes is not desired, consider union(),
    which raises an exception for name collisions.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2)])
    >>> H = nx.Graph([(0, 1), (1, 2)])
    >>> R = nx.compose(G, H)
    >>> R.nodes
    NodeView((0, 1, 2))
    >>> R.edges
    EdgeView([(0, 1), (0, 2), (1, 2)])

    By default, the attributes from `H` take precedent over attributes from `G`.
    If you prefer another way of combining attributes, you can update them after the compose operation:

    >>> G = nx.Graph([(0, 1, {"weight": 2.0}), (3, 0, {"weight": 100.0})])
    >>> H = nx.Graph([(0, 1, {"weight": 10.0}), (1, 2, {"weight": -1.0})])
    >>> nx.set_node_attributes(G, {0: "dark", 1: "light", 3: "black"}, name="color")
    >>> nx.set_node_attributes(H, {0: "green", 1: "orange", 2: "yellow"}, name="color")
    >>> GcomposeH = nx.compose(G, H)

    Normally, color attribute values of nodes of GcomposeH come from H. We can workaround this as follows:

    >>> node_data = {
    ...     n: G.nodes[n]["color"] + " " + H.nodes[n]["color"]
    ...     for n in G.nodes & H.nodes
    ... }
    >>> nx.set_node_attributes(GcomposeH, node_data, "color")
    >>> print(GcomposeH.nodes[0]["color"])
    dark green

    >>> print(GcomposeH.nodes[3]["color"])
    black

    Similarly, we can update edge attributes after the compose operation in a way we prefer:

    >>> edge_data = {
    ...     e: G.edges[e]["weight"] * H.edges[e]["weight"] for e in G.edges & H.edges
    ... }
    >>> nx.set_edge_attributes(GcomposeH, edge_data, "weight")
    >>> print(GcomposeH.edges[(0, 1)]["weight"])
    20.0

    >>> print(GcomposeH.edges[(3, 0)]["weight"])
    100.0
    """
    return nx.compose_all([G, H])


@nx._dispatchable(graphs=_G_H, preserve_all_attrs=True, returns_graph=True)
def full_join(G, H, rename=(None, None)):
    """Returns the full join of graphs G and H.

    Full join is the union of G and H in which all edges between
    G and H are added.
    The node sets of G and H must be disjoint,
    otherwise an exception is raised.

    Parameters
    ----------
    G, H : graph
       A NetworkX graph

    rename : tuple , default=(None, None)
       Node names of G and H can be changed by specifying the tuple
       rename=('G-','H-') (for example).  Node "u" in G is then renamed
       "G-u" and "v" in H is renamed "H-v".

    Returns
    -------
    U : The full join graph with the same type as G.

    Notes
    -----
    It is recommended that G and H be either both directed or both undirected.

    If G is directed, then edges from G to H are added as well as from H to G.

    Note that full_join() does not produce parallel edges for MultiGraphs.

    The full join operation of graphs G and H is the same as getting
    their complement, performing a disjoint union, and finally getting
    the complement of the resulting graph.

    Graph, edge, and node attributes are propagated from G and H
    to the union graph.  If a graph attribute is present in both
    G and H the value from H is used.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2)])
    >>> H = nx.Graph([(3, 4)])
    >>> R = nx.full_join(G, H, rename=("G", "H"))
    >>> R.nodes
    NodeView(('G0', 'G1', 'G2', 'H3', 'H4'))
    >>> R.edges
    EdgeView([('G0', 'G1'), ('G0', 'G2'), ('G0', 'H3'), ('G0', 'H4'), ('G1', 'H3'), ('G1', 'H4'), ('G2', 'H3'), ('G2', 'H4'), ('H3', 'H4')])

    See Also
    --------
    union
    disjoint_union
    """
    R = union(G, H, rename)

    def add_prefix(graph, prefix):
        if prefix is None:
            return graph

        def label(x):
            return f"{prefix}{x}"

        return nx.relabel_nodes(graph, label)

    G = add_prefix(G, rename[0])
    H = add_prefix(H, rename[1])

    for i in G:
        for j in H:
            R.add_edge(i, j)
    if R.is_directed():
        for i in H:
            for j in G:
                R.add_edge(i, j)

    return R
<|endoftext|>"""Unary operations on graphs"""

import networkx as nx

__all__ = ["complement", "reverse"]


@nx._dispatchable(returns_graph=True)
def complement(G):
    """Returns the graph complement of G.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    Returns
    -------
    GC : A new graph.

    Notes
    -----
    Note that `complement` does not create self-loops and also
    does not produce parallel edges for MultiGraphs.

    Graph, node, and edge data are not propagated to the new graph.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (3, 5)])
    >>> G_complement = nx.complement(G)
    >>> G_complement.edges()  # This shows the edges of the complemented graph
    EdgeView([(1, 4), (1, 5), (2, 4), (2, 5), (4, 5)])

    """
    R = G.__class__()
    R.add_nodes_from(G)
    R.add_edges_from(
        ((n, n2) for n, nbrs in G.adjacency() for n2 in G if n2 not in nbrs if n != n2)
    )
    return R


@nx._dispatchable(returns_graph=True)
def reverse(G, copy=True):
    """Returns the reverse directed graph of G.

    Parameters
    ----------
    G : directed graph
        A NetworkX directed graph
    copy : bool
        If True, then a new graph is returned. If False, then the graph is
        reversed in place.

    Returns
    -------
    H : directed graph
        The reversed G.

    Raises
    ------
    NetworkXError
        If graph is undirected.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 4), (3, 5)])
    >>> G_reversed = nx.reverse(G)
    >>> G_reversed.edges()
    OutEdgeView([(2, 1), (3, 1), (3, 2), (4, 3), (5, 3)])

    """
    if not G.is_directed():
        raise nx.NetworkXError("Cannot reverse an undirected graph.")
    else:
        return G.reverse(copy=copy)
<|endoftext|>from networkx.algorithms.operators.all import *
from networkx.algorithms.operators.binary import *
from networkx.algorithms.operators.product import *
from networkx.algorithms.operators.unary import *
<|endoftext|>"""
Graph products.
"""

from itertools import product

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "tensor_product",
    "cartesian_product",
    "lexicographic_product",
    "strong_product",
    "power",
    "rooted_product",
    "corona_product",
    "modular_product",
]
_G_H = {"G": 0, "H": 1}


def _dict_product(d1, d2):
    return {k: (d1.get(k), d2.get(k)) for k in set(d1) | set(d2)}


# Generators for producing graph products
def _node_product(G, H):
    for u, v in product(G, H):
        yield ((u, v), _dict_product(G.nodes[u], H.nodes[v]))


def _directed_edges_cross_edges(G, H):
    if not G.is_multigraph() and not H.is_multigraph():
        for u, v, c in G.edges(data=True):
            for x, y, d in H.edges(data=True):
                yield (u, x), (v, y), _dict_product(c, d)
    if not G.is_multigraph() and H.is_multigraph():
        for u, v, c in G.edges(data=True):
            for x, y, k, d in H.edges(data=True, keys=True):
                yield (u, x), (v, y), k, _dict_product(c, d)
    if G.is_multigraph() and not H.is_multigraph():
        for u, v, k, c in G.edges(data=True, keys=True):
            for x, y, d in H.edges(data=True):
                yield (u, x), (v, y), k, _dict_product(c, d)
    if G.is_multigraph() and H.is_multigraph():
        for u, v, j, c in G.edges(data=True, keys=True):
            for x, y, k, d in H.edges(data=True, keys=True):
                yield (u, x), (v, y), (j, k), _dict_product(c, d)


def _undirected_edges_cross_edges(G, H):
    if not G.is_multigraph() and not H.is_multigraph():
        for u, v, c in G.edges(data=True):
            for x, y, d in H.edges(data=True):
                yield (v, x), (u, y), _dict_product(c, d)
    if not G.is_multigraph() and H.is_multigraph():
        for u, v, c in G.edges(data=True):
            for x, y, k, d in H.edges(data=True, keys=True):
                yield (v, x), (u, y), k, _dict_product(c, d)
    if G.is_multigraph() and not H.is_multigraph():
        for u, v, k, c in G.edges(data=True, keys=True):
            for x, y, d in H.edges(data=True):
                yield (v, x), (u, y), k, _dict_product(c, d)
    if G.is_multigraph() and H.is_multigraph():
        for u, v, j, c in G.edges(data=True, keys=True):
            for x, y, k, d in H.edges(data=True, keys=True):
                yield (v, x), (u, y), (j, k), _dict_product(c, d)


def _edges_cross_nodes(G, H):
    if G.is_multigraph():
        for u, v, k, d in G.edges(data=True, keys=True):
            for x in H:
                yield (u, x), (v, x), k, d
    else:
        for u, v, d in G.edges(data=True):
            for x in H:
                if H.is_multigraph():
                    yield (u, x), (v, x), None, d
                else:
                    yield (u, x), (v, x), d


def _nodes_cross_edges(G, H):
    if H.is_multigraph():
        for x in G:
            for u, v, k, d in H.edges(data=True, keys=True):
                yield (x, u), (x, v), k, d
    else:
        for x in G:
            for u, v, d in H.edges(data=True):
                if G.is_multigraph():
                    yield (x, u), (x, v), None, d
                else:
                    yield (x, u), (x, v), d


def _edges_cross_nodes_and_nodes(G, H):
    if G.is_multigraph():
        for u, v, k, d in G.edges(data=True, keys=True):
            for x in H:
                for y in H:
                    yield (u, x), (v, y), k, d
    else:
        for u, v, d in G.edges(data=True):
            for x in H:
                for y in H:
                    if H.is_multigraph():
                        yield (u, x), (v, y), None, d
                    else:
                        yield (u, x), (v, y), d


def _init_product_graph(G, H):
    if G.is_directed() != H.is_directed():
        msg = "G and H must be both directed or both undirected"
        raise nx.NetworkXError(msg)
    if G.is_multigraph() or H.is_multigraph():
        GH = nx.MultiGraph()
    else:
        GH = nx.Graph()
    if G.is_directed():
        GH = GH.to_directed()
    return GH


@nx._dispatchable(graphs=_G_H, preserve_node_attrs=True, returns_graph=True)
def tensor_product(G, H):
    r"""Returns the tensor product of G and H.

    The tensor product $P$ of the graphs $G$ and $H$ has a node set that
    is the Cartesian product of the node sets, $V(P)=V(G) \times V(H)$.
    $P$ has an edge $((u,v), (x,y))$ if and only if $(u,x)$ is an edge in $G$
    and $(v,y)$ is an edge in $H$.

    Tensor product is sometimes also referred to as the categorical product,
    direct product, cardinal product or conjunction.


    Parameters
    ----------
    G, H: graphs
     Networkx graphs.

    Returns
    -------
    P: NetworkX graph
     The tensor product of G and H. P will be a multi-graph if either G
     or H is a multi-graph, will be a directed if G and H are directed,
     and undirected if G and H are undirected.

    Raises
    ------
    NetworkXError
     If G and H are not both directed or both undirected.

    Notes
    -----
    Node attributes in P are two-tuple of the G and H node attributes.
    Missing attributes are assigned None.

    Examples
    --------
    >>> G = nx.Graph()
    >>> H = nx.Graph()
    >>> G.add_node(0, a1=True)
    >>> H.add_node("a", a2="Spam")
    >>> P = nx.tensor_product(G, H)
    >>> list(P)
    [(0, 'a')]

    Edge attributes and edge keys (for multigraphs) are also copied to the
    new product graph
    """
    GH = _init_product_graph(G, H)
    GH.add_nodes_from(_node_product(G, H))
    GH.add_edges_from(_directed_edges_cross_edges(G, H))
    if not GH.is_directed():
        GH.add_edges_from(_undirected_edges_cross_edges(G, H))
    return GH


@nx._dispatchable(graphs=_G_H, preserve_node_attrs=True, returns_graph=True)
def cartesian_product(G, H):
    r"""Returns the Cartesian product of G and H.

    The Cartesian product $P$ of the graphs $G$ and $H$ has a node set that
    is the Cartesian product of the node sets, $V(P)=V(G) \times V(H)$.
    $P$ has an edge $((u,v),(x,y))$ if and only if either $u$ is equal to $x$
    and both $v$ and $y$ are adjacent in $H$ or if $v$ is equal to $y$ and
    both $u$ and $x$ are adjacent in $G$.

    Parameters
    ----------
    G, H: graphs
     Networkx graphs.

    Returns
    -------
    P: NetworkX graph
     The Cartesian product of G and H. P will be a multi-graph if either G
     or H is a multi-graph. Will be a directed if G and H are directed,
     and undirected if G and H are undirected.

    Raises
    ------
    NetworkXError
     If G and H are not both directed or both undirected.

    Notes
    -----
    Node attributes in P are two-tuple of the G and H node attributes.
    Missing attributes are assigned None.

    Examples
    --------
    >>> G = nx.Graph()
    >>> H = nx.Graph()
    >>> G.add_node(0, a1=True)
    >>> H.add_node("a", a2="Spam")
    >>> P = nx.cartesian_product(G, H)
    >>> list(P)
    [(0, 'a')]

    Edge attributes and edge keys (for multigraphs) are also copied to the
    new product graph
    """
    GH = _init_product_graph(G, H)
    GH.add_nodes_from(_node_product(G, H))
    GH.add_edges_from(_edges_cross_nodes(G, H))
    GH.add_edges_from(_nodes_cross_edges(G, H))
    return GH


@nx._dispatchable(graphs=_G_H, preserve_node_attrs=True, returns_graph=True)
def lexicographic_product(G, H):
    r"""Returns the lexicographic product of G and H.

    The lexicographical product $P$ of the graphs $G$ and $H$ has a node set
    that is the Cartesian product of the node sets, $V(P)=V(G) \times V(H)$.
    $P$ has an edge $((u,v), (x,y))$ if and only if $(u,v)$ is an edge in $G$
    or $u==v$ and $(x,y)$ is an edge in $H$.

    Parameters
    ----------
    G, H: graphs
     Networkx graphs.

    Returns
    -------
    P: NetworkX graph
     The Cartesian product of G and H. P will be a multi-graph if either G
     or H is a multi-graph. Will be a directed if G and H are directed,
     and undirected if G and H are undirected.

    Raises
    ------
    NetworkXError
     If G and H are not both directed or both undirected.

    Notes
    -----
    Node attributes in P are two-tuple of the G and H node attributes.
    Missing attributes are assigned None.

    Examples
    --------
    >>> G = nx.Graph()
    >>> H = nx.Graph()
    >>> G.add_node(0, a1=True)
    >>> H.add_node("a", a2="Spam")
    >>> P = nx.lexicographic_product(G, H)
    >>> list(P)
    [(0, 'a')]

    Edge attributes and edge keys (for multigraphs) are also copied to the
    new product graph
    """
    GH = _init_product_graph(G, H)
    GH.add_nodes_from(_node_product(G, H))
    # Edges in G regardless of H designation
    GH.add_edges_from(_edges_cross_nodes_and_nodes(G, H))
    # For each x in G, only if there is an edge in H
    GH.add_edges_from(_nodes_cross_edges(G, H))
    return GH


@nx._dispatchable(graphs=_G_H, preserve_node_attrs=True, returns_graph=True)
def strong_product(G, H):
    r"""Returns the strong product of G and H.

    The strong product $P$ of the graphs $G$ and $H$ has a node set that
    is the Cartesian product of the node sets, $V(P)=V(G) \times V(H)$.
    $P$ has an edge $((u,x), (v,y))$ if any of the following conditions
    are met:

    - $u=v$ and $(x,y)$ is an edge in $H$
    - $x=y$ and $(u,v)$ is an edge in $G$
    - $(u,v)$ is an edge in $G$ and $(x,y)$ is an edge in $H$

    Parameters
    ----------
    G, H: graphs
     Networkx graphs.

    Returns
    -------
    P: NetworkX graph
     The Cartesian product of G and H. P will be a multi-graph if either G
     or H is a multi-graph. Will be a directed if G and H are directed,
     and undirected if G and H are undirected.

    Raises
    ------
    NetworkXError
     If G and H are not both directed or both undirected.

    Notes
    -----
    Node attributes in P are two-tuple of the G and H node attributes.
    Missing attributes are assigned None.

    Examples
    --------
    >>> G = nx.Graph()
    >>> H = nx.Graph()
    >>> G.add_node(0, a1=True)
    >>> H.add_node("a", a2="Spam")
    >>> P = nx.strong_product(G, H)
    >>> list(P)
    [(0, 'a')]

    Edge attributes and edge keys (for multigraphs) are also copied to the
    new product graph
    """
    GH = _init_product_graph(G, H)
    GH.add_nodes_from(_node_product(G, H))
    GH.add_edges_from(_nodes_cross_edges(G, H))
    GH.add_edges_from(_edges_cross_nodes(G, H))
    GH.add_edges_from(_directed_edges_cross_edges(G, H))
    if not GH.is_directed():
        GH.add_edges_from(_undirected_edges_cross_edges(G, H))
    return GH


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(returns_graph=True)
def power(G, k):
    """Returns the specified power of a graph.

    The $k$th power of a simple graph $G$, denoted $G^k$, is a
    graph on the same set of nodes in which two distinct nodes $u$ and
    $v$ are adjacent in $G^k$ if and only if the shortest path
    distance between $u$ and $v$ in $G$ is at most $k$.

    Parameters
    ----------
    G : graph
        A NetworkX simple graph object.

    k : positive integer
        The power to which to raise the graph `G`.

    Returns
    -------
    NetworkX simple graph
        `G` to the power `k`.

    Raises
    ------
    ValueError
        If the exponent `k` is not positive.

    NetworkXNotImplemented
        If `G` is not a simple graph.

    Examples
    --------
    The number of edges will never decrease when taking successive
    powers:

    >>> G = nx.path_graph(4)
    >>> list(nx.power(G, 2).edges)
    [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    >>> list(nx.power(G, 3).edges)
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    The `k` th power of a cycle graph on *n* nodes is the complete graph
    on *n* nodes, if `k` is at least ``n // 2``:

    >>> G = nx.cycle_graph(5)
    >>> H = nx.complete_graph(5)
    >>> nx.is_isomorphic(nx.power(G, 2), H)
    True
    >>> G = nx.cycle_graph(8)
    >>> H = nx.complete_graph(8)
    >>> nx.is_isomorphic(nx.power(G, 4), H)
    True

    References
    ----------
    .. [1] J. A. Bondy, U. S. R. Murty, *Graph Theory*. Springer, 2008.

    Notes
    -----
    This definition of "power graph" comes from Exercise 3.1.6 of
    *Graph Theory* by Bondy and Murty [1]_.

    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    H = nx.Graph()
    H.add_nodes_from(G)
    # update BFS code to ignore self loops.
    for n in G:
        seen = {}  # level (number of hops) when seen in BFS
        level = 1  # the current level
        nextlevel = G[n]
        while nextlevel:
            thislevel = nextlevel  # advance to next level
            nextlevel = {}  # and start a new list (fringe)
            for v in thislevel:
                if v == n:  # avoid self loop
                    continue
                if v not in seen:
                    seen[v] = level  # set the level of vertex v
                    nextlevel.update(G[v])  # add neighbors of v
            if k <= level:
                break
            level += 1
        H.add_edges_from((n, nbr) for nbr in seen)
    return H


@not_implemented_for("multigraph")
@nx._dispatchable(graphs=_G_H, returns_graph=True)
def rooted_product(G, H, root):
    """Return the rooted product of graphs G and H rooted at root in H.

    A new graph is constructed representing the rooted product of
    the inputted graphs, G and H, with a root in H.
    A rooted product duplicates H for each nodes in G with the root
    of H corresponding to the node in G. Nodes are renamed as the direct
    product of G and H. The result is a subgraph of the cartesian product.

    Parameters
    ----------
    G,H : graph
       A NetworkX graph
    root : node
       A node in H

    Returns
    -------
    R : The rooted product of G and H with a specified root in H

    Notes
    -----
    The nodes of R are the Cartesian Product of the nodes of G and H.
    The nodes of G and H are not relabeled.
    """
    if root not in H:
        raise nx.NodeNotFound("root must be a vertex in H")

    R = nx.Graph()
    R.add_nodes_from(product(G, H))

    R.add_edges_from(((e[0], root), (e[1], root)) for e in G.edges())
    R.add_edges_from(((g, e[0]), (g, e[1])) for g in G for e in H.edges())

    return R


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(graphs=_G_H, returns_graph=True)
def corona_product(G, H):
    r"""Returns the Corona product of G and H.

    The corona product of $G$ and $H$ is the graph $C = G \circ H$ obtained by
    taking one copy of $G$, called the center graph, $|V(G)|$ copies of $H$,
    called the outer graph, and making the $i$-th vertex of $G$ adjacent to
    every vertex of the $i$-th copy of $H$, where $1 ≤ i ≤ |V(G)|$.

    Parameters
    ----------
    G, H: NetworkX graphs
        The graphs to take the carona product of.
        `G` is the center graph and `H` is the outer graph

    Returns
    -------
    C: NetworkX graph
        The Corona product of G and H.

    Raises
    ------
    NetworkXError
        If G and H are not both directed or both undirected.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> H = nx.path_graph(2)
    >>> C = nx.corona_product(G, H)
    >>> list(C)
    [0, 1, 2, 3, (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    >>> print(C)
    Graph with 12 nodes and 16 edges

    References
    ----------
    [1] M. Tavakoli, F. Rahbarnia, and A. R. Ashrafi,
        "Studying the corona product of graphs under some graph invariants,"
        Transactions on Combinatorics, vol. 3, no. 3, pp. 43–49, Sep. 2014,
        doi: 10.22108/toc.2014.5542.
    [2] A. Faraji, "Corona Product in Graph Theory," Ali Faraji, May 11, 2021.
        https://blog.alifaraji.ir/math/graph-theory/corona-product.html (accessed Dec. 07, 2021).
    """
    GH = _init_product_graph(G, H)
    GH.add_nodes_from(G)
    GH.add_edges_from(G.edges)

    for G_node in G:
        # copy nodes of H in GH, call it H_i
        GH.add_nodes_from((G_node, v) for v in H)

        # copy edges of H_i based on H
        GH.add_edges_from(
            ((G_node, e0), (G_node, e1), d) for e0, e1, d in H.edges.data()
        )

        # creating new edges between H_i and a G's node
        GH.add_edges_from((G_node, (G_node, H_node)) for H_node in H)

    return GH


@nx._dispatchable(
    graphs=_G_H, preserve_edge_attrs=True, preserve_node_attrs=True, returns_graph=True
)
def modular_product(G, H):
    r"""Returns the Modular product of G and H.

    The modular product of `G` and `H` is the graph $M = G \nabla H$,
    consisting of the node set $V(M) = V(G) \times V(H)$ that is the Cartesian
    product of the node sets of `G` and `H`. Further, M contains an edge ((u, v), (x, y)):

    - if u is adjacent to x in `G` and v is adjacent to y in `H`, or
    - if u is not adjacent to x in `G` and v is not adjacent to y in `H`.

    More formally::

        E(M) = {((u, v), (x, y)) | ((u, x) in E(G) and (v, y) in E(H)) or
                                   ((u, x) not in E(G) and (v, y) not in E(H))}

    Parameters
    ----------
    G, H: NetworkX graphs
        The graphs to take the modular product of.

    Returns
    -------
    M: NetworkX graph
        The Modular product of `G` and `H`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not a simple graph.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> H = nx.path_graph(2)
    >>> M = nx.modular_product(G, H)
    >>> list(M)
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    >>> print(M)
    Graph with 8 nodes and 8 edges

    Notes
    -----
    The *modular product* is defined in [1]_ and was first
    introduced as the *weak modular product*.

    The modular product reduces the problem of counting isomorphic subgraphs
    in `G` and `H` to the problem of counting cliques in M. The subgraphs of
    `G` and `H` that are induced by the nodes of a clique in M are
    isomorphic [2]_ [3]_.

    References
    ----------
    .. [1] R. Hammack, W. Imrich, and S. Klavžar,
        "Handbook of Product Graphs", CRC Press, 2011.

    .. [2] H. G. Barrow and R. M. Burstall,
        "Subgraph isomorphism, matching relational structures and maximal
        cliques", Information Processing Letters, vol. 4, issue 4, pp. 83-84,
        1976, https://doi.org/10.1016/0020-0190(76)90049-1.

    .. [3] V. G. Vizing, "Reduction of the problem of isomorphism and isomorphic
        entrance to the task of finding the nondensity of a graph." Proc. Third
        All-Union Conference on Problems of Theoretical Cybernetics. 1974.
    """
    if G.is_directed() or H.is_directed():
        raise nx.NetworkXNotImplemented(
            "Modular product not implemented for directed graphs"
        )
    if G.is_multigraph() or H.is_multigraph():
        raise nx.NetworkXNotImplemented(
            "Modular product not implemented for multigraphs"
        )

    GH = _init_product_graph(G, H)
    GH.add_nodes_from(_node_product(G, H))

    for u, v, c in G.edges(data=True):
        for x, y, d in H.edges(data=True):
            GH.add_edge((u, x), (v, y), **_dict_product(c, d))
            GH.add_edge((v, x), (u, y), **_dict_product(c, d))

    G = nx.complement(G)
    H = nx.complement(H)

    for u, v, c in G.edges(data=True):
        for x, y, d in H.edges(data=True):
            GH.add_edge((u, x), (v, y), **_dict_product(c, d))
            GH.add_edge((v, x), (u, y), **_dict_product(c, d))

    return GH
<|endoftext|>"""Operations on many graphs."""

from itertools import chain, repeat

import networkx as nx

__all__ = ["union_all", "compose_all", "disjoint_union_all", "intersection_all"]


@nx._dispatchable(graphs="[graphs]", preserve_all_attrs=True, returns_graph=True)
def union_all(graphs, rename=()):
    """Returns the union of all graphs.

    The graphs must be disjoint, otherwise an exception is raised.

    Parameters
    ----------
    graphs : iterable
       Iterable of NetworkX graphs

    rename : iterable , optional
       Node names of graphs can be changed by specifying the tuple
       rename=('G-','H-') (for example).  Node "u" in G is then renamed
       "G-u" and "v" in H is renamed "H-v". Infinite generators (like itertools.count)
       are also supported.

    Returns
    -------
    U : a graph with the same type as the first graph in list

    Raises
    ------
    ValueError
       If `graphs` is an empty list.

    NetworkXError
        In case of mixed type graphs, like MultiGraph and Graph, or directed and undirected graphs.

    Notes
    -----
    For operating on mixed type graphs, they should be converted to the same type.
    >>> G = nx.Graph()
    >>> H = nx.DiGraph()
    >>> GH = union_all([nx.DiGraph(G), H])

    To force a disjoint union with node relabeling, use
    disjoint_union_all(G,H) or convert_node_labels_to integers().

    Graph, edge, and node attributes are propagated to the union graph.
    If a graph attribute is present in multiple graphs, then the value
    from the last graph in the list with that attribute is used.

    Examples
    --------
    >>> G1 = nx.Graph([(1, 2), (2, 3)])
    >>> G2 = nx.Graph([(4, 5), (5, 6)])
    >>> result_graph = nx.union_all([G1, G2])
    >>> result_graph.nodes()
    NodeView((1, 2, 3, 4, 5, 6))
    >>> result_graph.edges()
    EdgeView([(1, 2), (2, 3), (4, 5), (5, 6)])

    See Also
    --------
    union
    disjoint_union_all
    """
    R = None
    seen_nodes = set()

    # rename graph to obtain disjoint node labels
    def add_prefix(graph, prefix):
        if prefix is None:
            return graph

        def label(x):
            return f"{prefix}{x}"

        return nx.relabel_nodes(graph, label)

    rename = chain(rename, repeat(None))
    graphs = (add_prefix(G, name) for G, name in zip(graphs, rename))

    for i, G in enumerate(graphs):
        G_nodes_set = set(G.nodes)
        if i == 0:
            # Union is the same type as first graph
            R = G.__class__()
        elif G.is_directed() != R.is_directed():
            raise nx.NetworkXError("All graphs must be directed or undirected.")
        elif G.is_multigraph() != R.is_multigraph():
            raise nx.NetworkXError("All graphs must be graphs or multigraphs.")
        elif not seen_nodes.isdisjoint(G_nodes_set):
            raise nx.NetworkXError(
                "The node sets of the graphs are not disjoint.\n"
                "Use `rename` to specify prefixes for the graphs or use\n"
                "disjoint_union(G1, G2, ..., GN)."
            )

        seen_nodes |= G_nodes_set
        R.graph.update(G.graph)
        R.add_nodes_from(G.nodes(data=True))
        R.add_edges_from(
            G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
        )

    if R is None:
        raise ValueError("cannot apply union_all to an empty list")

    return R


@nx._dispatchable(graphs="[graphs]", preserve_all_attrs=True, returns_graph=True)
def disjoint_union_all(graphs):
    """Returns the disjoint union of all graphs.

    This operation forces distinct integer node labels starting with 0
    for the first graph in the list and numbering consecutively.

    Parameters
    ----------
    graphs : iterable
       Iterable of NetworkX graphs

    Returns
    -------
    U : A graph with the same type as the first graph in list

    Raises
    ------
    ValueError
       If `graphs` is an empty list.

    NetworkXError
        In case of mixed type graphs, like MultiGraph and Graph, or directed and undirected graphs.

    Examples
    --------
    >>> G1 = nx.Graph([(1, 2), (2, 3)])
    >>> G2 = nx.Graph([(4, 5), (5, 6)])
    >>> U = nx.disjoint_union_all([G1, G2])
    >>> list(U.nodes())
    [0, 1, 2, 3, 4, 5]
    >>> list(U.edges())
    [(0, 1), (1, 2), (3, 4), (4, 5)]

    Notes
    -----
    For operating on mixed type graphs, they should be converted to the same type.

    Graph, edge, and node attributes are propagated to the union graph.
    If a graph attribute is present in multiple graphs, then the value
    from the last graph in the list with that attribute is used.
    """

    def yield_relabeled(graphs):
        first_label = 0
        for G in graphs:
            yield nx.convert_node_labels_to_integers(G, first_label=first_label)
            first_label += len(G)

    R = union_all(yield_relabeled(graphs))

    return R


@nx._dispatchable(graphs="[graphs]", preserve_all_attrs=True, returns_graph=True)
def compose_all(graphs):
    """Returns the composition of all graphs.

    Composition is the simple union of the node sets and edge sets.
    The node sets of the supplied graphs need not be disjoint.

    Parameters
    ----------
    graphs : iterable
       Iterable of NetworkX graphs

    Returns
    -------
    C : A graph with the same type as the first graph in list

    Raises
    ------
    ValueError
       If `graphs` is an empty list.

    NetworkXError
        In case of mixed type graphs, like MultiGraph and Graph, or directed and undirected graphs.

    Examples
    --------
    >>> G1 = nx.Graph([(1, 2), (2, 3)])
    >>> G2 = nx.Graph([(3, 4), (5, 6)])
    >>> C = nx.compose_all([G1, G2])
    >>> list(C.nodes())
    [1, 2, 3, 4, 5, 6]
    >>> list(C.edges())
    [(1, 2), (2, 3), (3, 4), (5, 6)]

    Notes
    -----
    For operating on mixed type graphs, they should be converted to the same type.

    Graph, edge, and node attributes are propagated to the union graph.
    If a graph attribute is present in multiple graphs, then the value
    from the last graph in the list with that attribute is used.
    """
    R = None

    # add graph attributes, H attributes take precedent over G attributes
    for i, G in enumerate(graphs):
        if i == 0:
            # create new graph
            R = G.__class__()
        elif G.is_directed() != R.is_directed():
            raise nx.NetworkXError("All graphs must be directed or undirected.")
        elif G.is_multigraph() != R.is_multigraph():
            raise nx.NetworkXError("All graphs must be graphs or multigraphs.")

        R.graph.update(G.graph)
        R.add_nodes_from(G.nodes(data=True))
        R.add_edges_from(
            G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
        )

    if R is None:
        raise ValueError("cannot apply compose_all to an empty list")

    return R


@nx._dispatchable(graphs="[graphs]", returns_graph=True)
def intersection_all(graphs):
    """Returns a new graph that contains only the nodes and the edges that exist in
    all graphs.

    Parameters
    ----------
    graphs : iterable
       Iterable of NetworkX graphs

    Returns
    -------
    R : A new graph with the same type as the first graph in list

    Raises
    ------
    ValueError
       If `graphs` is an empty list.

    NetworkXError
        In case of mixed type graphs, like MultiGraph and Graph, or directed and undirected graphs.

    Notes
    -----
    For operating on mixed type graphs, they should be converted to the same type.

    Attributes from the graph, nodes, and edges are not copied to the new
    graph.

    The resulting graph can be updated with attributes if desired. For example, code which adds the minimum attribute for each node across all graphs could work.
    >>> g = nx.Graph()
    >>> g.add_node(0, capacity=4)
    >>> g.add_node(1, capacity=3)
    >>> g.add_edge(0, 1)

    >>> h = g.copy()
    >>> h.nodes[0]["capacity"] = 2

    >>> gh = nx.intersection_all([g, h])

    >>> new_node_attr = {
    ...     n: min(*(anyG.nodes[n].get("capacity", float("inf")) for anyG in [g, h]))
    ...     for n in gh
    ... }
    >>> nx.set_node_attributes(gh, new_node_attr, "new_capacity")
    >>> gh.nodes(data=True)
    NodeDataView({0: {'new_capacity': 2}, 1: {'new_capacity': 3}})

    Examples
    --------
    >>> G1 = nx.Graph([(1, 2), (2, 3)])
    >>> G2 = nx.Graph([(2, 3), (3, 4)])
    >>> R = nx.intersection_all([G1, G2])
    >>> list(R.nodes())
    [2, 3]
    >>> list(R.edges())
    [(2, 3)]

    """
    R = None

    for i, G in enumerate(graphs):
        G_nodes_set = set(G.nodes)
        G_edges_set = set(G.edges)
        if not G.is_directed():
            if G.is_multigraph():
                G_edges_set.update((v, u, k) for u, v, k in list(G_edges_set))
            else:
                G_edges_set.update((v, u) for u, v in list(G_edges_set))
        if i == 0:
            # create new graph
            R = G.__class__()
            node_intersection = G_nodes_set
            edge_intersection = G_edges_set
        elif G.is_directed() != R.is_directed():
            raise nx.NetworkXError("All graphs must be directed or undirected.")
        elif G.is_multigraph() != R.is_multigraph():
            raise nx.NetworkXError("All graphs must be graphs or multigraphs.")
        else:
            node_intersection &= G_nodes_set
            edge_intersection &= G_edges_set

    if R is None:
        raise ValueError("cannot apply intersection_all to an empty list")

    R.add_nodes_from(node_intersection)
    R.add_edges_from(edge_intersection)

    return R
<|endoftext|>"""Semiconnectedness."""

import networkx as nx
from networkx.utils import not_implemented_for, pairwise

__all__ = ["is_semiconnected"]


@not_implemented_for("undirected")
@nx._dispatchable
def is_semiconnected(G):
    r"""Returns True if the graph is semiconnected, False otherwise.

    A graph is semiconnected if and only if for any pair of nodes, either one
    is reachable from the other, or they are mutually reachable.

    This function uses a theorem that states that a DAG is semiconnected
    if for any topological sort, for node $v_n$ in that sort, there is an
    edge $(v_i, v_{i+1})$. That allows us to check if a non-DAG `G` is
    semiconnected by condensing the graph: i.e. constructing a new graph `H`
    with nodes being the strongly connected components of `G`, and edges
    (scc_1, scc_2) if there is a edge $(v_1, v_2)$ in `G` for some
    $v_1 \in scc_1$ and $v_2 \in scc_2$. That results in a DAG, so we compute
    the topological sort of `H` and check if for every $n$ there is an edge
    $(scc_n, scc_{n+1})$.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph.

    Returns
    -------
    semiconnected : bool
        True if the graph is semiconnected, False otherwise.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is undirected.

    NetworkXPointlessConcept
        If the graph is empty.

    Examples
    --------
    >>> G = nx.path_graph(4, create_using=nx.DiGraph())
    >>> print(nx.is_semiconnected(G))
    True
    >>> G = nx.DiGraph([(1, 2), (3, 2)])
    >>> print(nx.is_semiconnected(G))
    False

    See Also
    --------
    is_strongly_connected
    is_weakly_connected
    is_connected
    is_biconnected
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined for the null graph."
        )

    if not nx.is_weakly_connected(G):
        return False

    H = nx.condensation(G)

    return all(H.has_edge(u, v) for u, v in pairwise(nx.topological_sort(H)))
<|endoftext|>"""Strongly connected components."""

import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = [
    "number_strongly_connected_components",
    "strongly_connected_components",
    "is_strongly_connected",
    "kosaraju_strongly_connected_components",
    "condensation",
]


@not_implemented_for("undirected")
@nx._dispatchable
def strongly_connected_components(G):
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

