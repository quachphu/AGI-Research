    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)
    inf = float("inf")
    n = len(G)

    count = {}
    q = deque(sources)
    in_q = set(sources)
    while q:
        u = q.popleft()
        in_q.remove(u)

        # Skip relaxations if any of the predecessors of u is in the queue.
        if all(pred_u not in in_q for pred_u in pred[u]):
            dist_u = dist[u]
            for v, e in G_succ[u].items():
                dist_v = dist_u + weight(u, v, e)

                if dist_v < dist.get(v, inf):
                    # In this conditional branch we are updating the path with v.
                    # If it happens that some earlier update also added node v
                    # that implies the existence of a negative cycle since
                    # after the update node v would lie on the update path twice.
                    # The update path is stored up to one of the source nodes,
                    # therefore u is always in the dict recent_update
                    if heuristic:
                        if v in recent_update[u]:
                            # Negative cycle found!
                            pred[v].append(u)
                            return v

                        # Transfer the recent update info from u to v if the
                        # same source node is the head of the update path.
                        # If the source node is responsible for the cost update,
                        # then clear the history and use it instead.
                        if v in pred_edge and pred_edge[v] == u:
                            recent_update[v] = recent_update[u]
                        else:
                            recent_update[v] = (u, v)

                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
                        count_v = count.get(v, 0) + 1
                        if count_v == n:
                            # Negative cycle found!
                            return v

                        count[v] = count_v
                    dist[v] = dist_v
                    pred[v] = [u]
                    pred_edge[v] = u

                elif dist.get(v) is not None and dist_v == dist.get(v):
                    pred[v].append(u)

    # successfully found shortest_path. No negative cycles found.
    return None


@nx._dispatchable(edge_attrs="weight")
def bellman_ford_path(G, source, target, weight="weight"):
    """Returns the shortest path from source to target in a weighted graph G.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node

    target : node
        Ending node

    weight : string or function (default="weight")
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
    >>> nx.bellman_ford_path(G, 0, 4)
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    dijkstra_path, bellman_ford_path_length
    """
    length, path = single_source_bellman_ford(G, source, target=target, weight=weight)
    return path


@nx._dispatchable(edge_attrs="weight")
def bellman_ford_path_length(G, source, target, weight="weight"):
    """Returns the shortest path length from source to target
    in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        starting node for path

    target : node label
        ending node for path

    weight : string or function (default="weight")
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
    >>> nx.bellman_ford_path_length(G, 0, 4)
    4

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    dijkstra_path_length, bellman_ford_path
    """
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f"Node {source} not found in graph")
        return 0

    weight = _weight_function(G, weight)

    length = _bellman_ford(G, [source], weight, target=target)

    try:
        return length[target]
    except KeyError as err:
        raise nx.NetworkXNoPath(f"node {target} not reachable from {source}") from err


@nx._dispatchable(edge_attrs="weight")
def single_source_bellman_ford_path(G, source, weight="weight"):
    """Compute shortest path between source and all other reachable
    nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for path.

    weight : string or function (default="weight")
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
    >>> path = nx.single_source_bellman_ford_path(G, 0)
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    single_source_dijkstra, single_source_bellman_ford

    """
    (length, path) = single_source_bellman_ford(G, source, weight=weight)
    return path


@nx._dispatchable(edge_attrs="weight")
def single_source_bellman_ford_path_length(G, source, weight="weight"):
    """Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        Starting node for path

    weight : string or function (default="weight")
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

    Returns
    -------
    length : dictionary
        Dictionary of shortest path length keyed by target

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = nx.single_source_bellman_ford_path_length(G, 0)
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

    See Also
    --------
    single_source_dijkstra, single_source_bellman_ford

    """
    weight = _weight_function(G, weight)
    return _bellman_ford(G, [source], weight)


@nx._dispatchable(edge_attrs="weight")
def single_source_bellman_ford(G, source, target=None, weight="weight"):
    """Compute shortest paths and lengths in a weighted graph G.

    Uses Bellman-Ford algorithm for shortest paths.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
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

    Returns
    -------
    distance, path : pair of dictionaries, or numeric and list
        If target is None, returns a tuple of two dictionaries keyed by node.
        The first dictionary stores distance from one of the source nodes.
        The second stores the path from one of the sources to that node.
        If target is not None, returns a tuple of (distance, path) where
        distance is the distance from source to target and path is a list
        representing the path from source to target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length, path = nx.single_source_bellman_ford(G, 0)
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
    >>> length, path = nx.single_source_bellman_ford(G, 0, 1)
    >>> length
    1
    >>> path
    [0, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    single_source_dijkstra
    single_source_bellman_ford_path
    single_source_bellman_ford_path_length
    """
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f"Node {source} is not found in the graph")
        return (0, [source])

    weight = _weight_function(G, weight)

    paths = {source: [source]}  # dictionary of paths
    dist = _bellman_ford(G, [source], weight, paths=paths, target=target)
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError as err:
        msg = f"Node {target} not reachable from {source}"
        raise nx.NetworkXNoPath(msg) from err


@nx._dispatchable(edge_attrs="weight")
def all_pairs_bellman_ford_path_length(G, weight="weight"):
    """Compute shortest path lengths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or function (default="weight")
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

    Returns
    -------
    distance : iterator
        (source, dictionary) iterator with dictionary keyed by target and
        shortest path length as the key value.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = dict(nx.all_pairs_bellman_ford_path_length(G))
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
    length = single_source_bellman_ford_path_length
    for n in G:
        yield (n, dict(length(G, n, weight=weight)))


@nx._dispatchable(edge_attrs="weight")
def all_pairs_bellman_ford_path(G, weight="weight"):
    """Compute shortest paths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or function (default="weight")
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

    Returns
    -------
    paths : iterator
        (source, dictionary) iterator with dictionary keyed by target and
        shortest path as the key value.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = dict(nx.all_pairs_bellman_ford_path(G))
    >>> path[0][4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    floyd_warshall, all_pairs_dijkstra_path

    """
    path = single_source_bellman_ford_path
    for n in G:
        yield (n, path(G, n, weight=weight))


@nx._dispatchable(edge_attrs="weight")
def goldberg_radzik(G, source, weight="weight"):
    """Compute shortest path lengths and predecessors on shortest paths
    in weighted graphs.

    The algorithm has a running time of $O(mn)$ where $n$ is the number of
    nodes and $m$ is the number of edges.  It is slower than Dijkstra but
    can handle negative edge weights.

    Parameters
    ----------
    G : NetworkX graph
        The algorithm works for all types of graphs, including directed
        graphs and multigraphs.

    source: node label
        Starting node for path

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

        As of NetworkX v3.2, a zero weight cycle is no longer
        incorrectly reported as a negative weight cycle.


    Examples
    --------
    >>> G = nx.path_graph(5, create_using=nx.DiGraph())
    >>> pred, dist = nx.goldberg_radzik(G, 0)
    >>> sorted(pred.items())
    [(0, None), (1, 0), (2, 1), (3, 2), (4, 3)]
    >>> sorted(dist.items())
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    >>> G = nx.cycle_graph(5, create_using=nx.DiGraph())
    >>> G[1][2]["weight"] = -7
    >>> nx.goldberg_radzik(G, 0)
    Traceback (most recent call last):
        ...
    networkx.exception.NetworkXUnbounded: Negative cycle detected.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The dictionaries returned only have keys for nodes reachable from
    the source.

    In the case where the (di)graph is not connected, if a component
    not containing the source contains a negative (di)cycle, it
    will not be detected.

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

    if len(G) == 1:
        return {source: None}, {source: 0}

    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

    inf = float("inf")
    d = {u: inf for u in G}
    d[source] = 0
    pred = {source: None}

    def topo_sort(relabeled):
        """Topologically sort nodes relabeled in the previous round and detect
        negative cycles.
        """
        # List of nodes to scan in this round. Denoted by A in Goldberg and
        # Radzik's paper.
        to_scan = []
        # In the DFS in the loop below, neg_count records for each node the
        # number of edges of negative reduced costs on the path from a DFS root
        # to the node in the DFS forest. The reduced cost of an edge (u, v) is
        # defined as d[u] + weight[u][v] - d[v].
        #
        # neg_count also doubles as the DFS visit marker array.
        neg_count = {}
        for u in relabeled:
            # Skip visited nodes.
            if u in neg_count:
                continue
            d_u = d[u]
            # Skip nodes without out-edges of negative reduced costs.
            if all(d_u + weight(u, v, e) >= d[v] for v, e in G_succ[u].items()):
                continue
            # Nonrecursive DFS that inserts nodes reachable from u via edges of
            # nonpositive reduced costs into to_scan in (reverse) topological
            # order.
            stack = [(u, iter(G_succ[u].items()))]
            in_stack = {u}
            neg_count[u] = 0
            while stack:
                u, it = stack[-1]
                try:
                    v, e = next(it)
                except StopIteration:
                    to_scan.append(u)
                    stack.pop()
                    in_stack.remove(u)
                    continue
                t = d[u] + weight(u, v, e)
                d_v = d[v]
                if t < d_v:
                    is_neg = t < d_v
                    d[v] = t
                    pred[v] = u
                    if v not in neg_count:
                        neg_count[v] = neg_count[u] + int(is_neg)
                        stack.append((v, iter(G_succ[v].items())))
                        in_stack.add(v)
                    elif v in in_stack and neg_count[u] + int(is_neg) > neg_count[v]:
                        # (u, v) is a back edge, and the cycle formed by the
                        # path v to u and (u, v) contains at least one edge of
                        # negative reduced cost. The cycle must be of negative
                        # cost.
                        raise nx.NetworkXUnbounded("Negative cycle detected.")
        to_scan.reverse()
        return to_scan

    def relax(to_scan):
        """Relax out-edges of relabeled nodes."""
        relabeled = set()
        # Scan nodes in to_scan in topological order and relax incident
        # out-edges. Add the relabled nodes to labeled.
        for u in to_scan:
            d_u = d[u]
            for v, e in G_succ[u].items():
                w_e = weight(u, v, e)
                if d_u + w_e < d[v]:
                    d[v] = d_u + w_e
                    pred[v] = u
                    relabeled.add(v)
        return relabeled

    # Set of nodes relabled in the last round of scan operations. Denoted by B
    # in Goldberg and Radzik's paper.
    relabeled = {source}

    while relabeled:
        to_scan = topo_sort(relabeled)
        relabeled = relax(to_scan)

    d = {u: d[u] for u in pred}
    return pred, d


@nx._dispatchable(edge_attrs="weight")
def negative_edge_cycle(G, weight="weight", heuristic=True):
    """Returns True if there exists a negative edge cycle anywhere in G.

    Parameters
    ----------
    G : NetworkX graph

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
        cycles at a negligible cost. In case of graphs with a negative cycle,
        the performance of detection increases by at least an order of magnitude.

    Returns
    -------
    negative_cycle : bool
        True if a negative edge cycle exists, otherwise False.

    Examples
    --------
    >>> G = nx.cycle_graph(5, create_using=nx.DiGraph())
    >>> print(nx.negative_edge_cycle(G))
    False
    >>> G[1][2]["weight"] = -7
    >>> print(nx.negative_edge_cycle(G))
    True

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    This algorithm uses bellman_ford_predecessor_and_distance() but finds
    negative cycles on any component by first adding a new node connected to
    every node, and starting bellman_ford_predecessor_and_distance on that
    node.  It then removes that extra node.
    """
    if G.size() == 0:
        return False

    # find unused node to use temporarily
    newnode = -1
    while newnode in G:
        newnode -= 1
    # connect it to all nodes
    G.add_edges_from([(newnode, n) for n in G])

    try:
        bellman_ford_predecessor_and_distance(
            G, newnode, weight=weight, heuristic=heuristic
        )
    except nx.NetworkXUnbounded:
        return True
    finally:
        G.remove_node(newnode)
    return False


@nx._dispatchable(edge_attrs="weight")
def find_negative_cycle(G, source, weight="weight"):
    """Returns a cycle with negative total weight if it exists.

    Bellman-Ford is used to find shortest_paths. That algorithm
    stops if there exists a negative cycle. This algorithm
    picks up from there and returns the found negative cycle.

    The cycle consists of a list of nodes in the cycle order. The last
    node equals the first to make it a cycle.
    You can look up the edge weights in the original graph. In the case
    of multigraphs the relevant edge is the minimal weight edge between
    the nodes in the 2-tuple.

    If the graph has no negative cycle, a NetworkXError is raised.

    Parameters
    ----------
    G : NetworkX graph

    source: node label
        The search for the negative cycle will start from this node.

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

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     [(0, 1, 2), (1, 2, 2), (2, 0, 1), (1, 4, 2), (4, 0, -5)]
    ... )
    >>> nx.find_negative_cycle(G, 0)
    [4, 0, 1, 4]

    Returns
    -------
    cycle : list
        A list of nodes in the order of the cycle found. The last node
        equals the first to indicate a cycle.

    Raises
    ------
    NetworkXError
        If no negative cycle is found.
    """
    weight = _weight_function(G, weight)
    pred = {source: []}

    v = _inner_bellman_ford(G, [source], weight, pred=pred)
    if v is None:
        raise nx.NetworkXError("No negative cycles detected.")

    # negative cycle detected... find it
    neg_cycle = []
    stack = [(v, list(pred[v]))]
    seen = {v}
    while stack:
        node, preds = stack[-1]
        if v in preds:
            # found the cycle
            neg_cycle.extend([node, v])
            neg_cycle = list(reversed(neg_cycle))
            return neg_cycle

        if preds:
            nbr = preds.pop()
            if nbr not in seen:
                stack.append((nbr, list(pred[nbr])))
                neg_cycle.append(node)
                seen.add(nbr)
        else:
            stack.pop()
            if neg_cycle:
                neg_cycle.pop()
            else:
                if v in G[v] and weight(G, v, v) < 0:
                    return [v, v]
                # should not reach here
                raise nx.NetworkXError("Negative cycle is detected but not found")
    # should not get here...
    msg = "negative cycle detected but not identified"
    raise nx.NetworkXUnbounded(msg)


@nx._dispatchable(edge_attrs="weight")
def bidirectional_dijkstra(G, source, target, weight="weight"):
    r"""Dijkstra's algorithm for shortest paths using bidirectional search.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node.

    target : node
        Ending node.

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
    length, path : number and list
        length is the distance from source to target.
        path is a list of nodes on a path from source to target.

    Raises
    ------
    NodeNotFound
        If `source` or `target` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length, path = nx.bidirectional_dijkstra(G, 0, 4)
    >>> print(length)
    4
    >>> print(path)
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    In practice  bidirectional Dijkstra is much more than twice as fast as
    ordinary Dijkstra.

    Ordinary Dijkstra expands nodes in a sphere-like manner from the
    source. The radius of this sphere will eventually be the length
    of the shortest path. Bidirectional Dijkstra will expand nodes
    from both the source and the target, making two spheres of half
    this radius. Volume of the first sphere is `\pi*r*r` while the
    others are `2*\pi*r/2*r/2`, making up half the volume.

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    shortest_path
    shortest_path_length
    """
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} is not in G")

    if target not in G:
        raise nx.NodeNotFound(f"Target {target} is not in G")

    if source == target:
        return (0, [source])

    weight = _weight_function(G, weight)
    push = heappush
    pop = heappop
    # Init:  [Forward, Backward]
    dists = [{}, {}]  # dictionary of final distances
    paths = [{source: [source]}, {target: [target]}]  # dictionary of paths
    fringe = [[], []]  # heap of (distance, node) for choosing node to expand
    seen = [{source: 0}, {target: 0}]  # dict of distances to seen nodes
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), source))
    push(fringe[1], (0, next(c), target))
    # neighs for extracting correct neighbor information
    if G.is_directed():
        neighs = [G._succ, G._pred]
    else:
        neighs = [G._adj, G._adj]
    # variables to hold shortest discovered path
    # finaldist = 1e30000
    finalpath = []
    dir = 1
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        for w, d in neighs[dir][v].items():
            # weight(v, w, d) for forward and weight(w, v, d) for back direction
            cost = weight(v, w, d) if dir == 0 else weight(w, v, d)
            if cost is None:
                continue
            vwLength = dists[dir][v] + cost
            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")


@nx._dispatchable(edge_attrs="weight")
def johnson(G, weight="weight"):
    r"""Uses Johnson's Algorithm to compute shortest paths.

    Johnson's Algorithm finds a shortest path between each pair of
    nodes in a weighted graph even if negative weights are present.

    Parameters
    ----------
    G : NetworkX graph

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

    Returns
    -------
    distance : dictionary
        Dictionary, keyed by source and target, of shortest paths.

    Examples
    --------
    >>> graph = nx.DiGraph()
    >>> graph.add_weighted_edges_from(
    ...     [("0", "3", 3), ("0", "1", -5), ("0", "2", 2), ("1", "2", 4), ("2", "3", 1)]
    ... )
    >>> paths = nx.johnson(graph, weight="weight")
    >>> paths["0"]["2"]
    ['0', '1', '2']

    Notes
    -----
    Johnson's algorithm is suitable even for graphs with negative weights. It
    works by using the Bellman–Ford algorithm to compute a transformation of
    the input graph that removes all negative weights, allowing Dijkstra's
    algorithm to be used on the transformed graph.

    The time complexity of this algorithm is $O(n^2 \log n + n m)$,
    where $n$ is the number of nodes and $m$ the number of edges in the
    graph. For dense graphs, this may be faster than the Floyd–Warshall
    algorithm.

    See Also
    --------
    floyd_warshall_predecessor_and_distance
    floyd_warshall_numpy
    all_pairs_shortest_path
    all_pairs_shortest_path_length
    all_pairs_dijkstra_path
    bellman_ford_predecessor_and_distance
    all_pairs_bellman_ford_path
    all_pairs_bellman_ford_path_length

    """
    dist = {v: 0 for v in G}
    pred = {v: [] for v in G}
    weight = _weight_function(G, weight)

    # Calculate distance of shortest paths
    dist_bellman = _bellman_ford(G, list(G), weight, pred=pred, dist=dist)

    # Update the weight function to take into account the Bellman--Ford
    # relaxation distances.
    def new_weight(u, v, d):
        return weight(u, v, d) + dist_bellman[u] - dist_bellman[v]

    def dist_path(v):
        paths = {v: [v]}
        _dijkstra(G, v, new_weight, paths=paths)
        return paths

    return {v: dist_path(v) for v in G}
<|endoftext|>"""
Edmonds-Karp algorithm for maximum flow problems.
"""

import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network

__all__ = ["edmonds_karp"]


def edmonds_karp_core(R, s, t, cutoff):
    """Implementation of the Edmonds-Karp algorithm."""
    R_nodes = R.nodes
    R_pred = R.pred
    R_succ = R.succ

    inf = R.graph["inf"]

    def augment(path):
        """Augment flow along a path from s to t."""
        # Determine the path residual capacity.
        flow = inf
        it = iter(path)
        u = next(it)
        for v in it:
            attr = R_succ[u][v]
            flow = min(flow, attr["capacity"] - attr["flow"])
            u = v
        if flow * 2 > inf:
            raise nx.NetworkXUnbounded("Infinite capacity path, flow unbounded above.")
        # Augment flow along the path.
        it = iter(path)
        u = next(it)
        for v in it:
            R_succ[u][v]["flow"] += flow
            R_succ[v][u]["flow"] -= flow
            u = v
        return flow

    def bidirectional_bfs():
        """Bidirectional breadth-first search for an augmenting path."""
        pred = {s: None}
        q_s = [s]
        succ = {t: None}
        q_t = [t]
        while True:
            q = []
            if len(q_s) <= len(q_t):
                for u in q_s:
                    for v, attr in R_succ[u].items():
                        if v not in pred and attr["flow"] < attr["capacity"]:
                            pred[v] = u
                            if v in succ:
                                return v, pred, succ
                            q.append(v)
                if not q:
                    return None, None, None
                q_s = q
            else:
                for u in q_t:
                    for v, attr in R_pred[u].items():
                        if v not in succ and attr["flow"] < attr["capacity"]:
                            succ[v] = u
                            if v in pred:
                                return v, pred, succ
                            q.append(v)
                if not q:
                    return None, None, None
                q_t = q

    # Look for shortest augmenting paths using breadth-first search.
    flow_value = 0
    while flow_value < cutoff:
        v, pred, succ = bidirectional_bfs()
        if pred is None:
            break
        path = [v]
        # Trace a path from s to v.
        u = v
        while u != s:
            u = pred[u]
            path.append(u)
        path.reverse()
        # Trace a path from v to t.
        u = v
        while u != t:
            u = succ[u]
            path.append(u)
        flow_value += augment(path)

    return flow_value


def edmonds_karp_impl(G, s, t, capacity, residual, cutoff):
    """Implementation of the Edmonds-Karp algorithm."""
    if s not in G:
        raise nx.NetworkXError(f"node {str(s)} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"node {str(t)} not in graph")
    if s == t:
        raise nx.NetworkXError("source and sink are the same node")

    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual

    # Initialize/reset the residual network.
    for u in R:
        for e in R[u].values():
            e["flow"] = 0

    if cutoff is None:
        cutoff = float("inf")
    R.graph["flow_value"] = edmonds_karp_core(R, s, t, cutoff)

    return R


@nx._dispatchable(edge_attrs={"capacity": float("inf")}, returns_graph=True)
def edmonds_karp(
    G, s, t, capacity="capacity", residual=None, value_only=False, cutoff=None
):
    """Find a maximum single-commodity flow using the Edmonds-Karp algorithm.

    This function returns the residual network resulting after computing
    the maximum flow. See below for details about the conventions
    NetworkX uses for defining residual networks.

    This algorithm has a running time of $O(n m^2)$ for $n$ nodes and $m$
    edges.


    Parameters
    ----------
    G : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    s : node
        Source node for the flow.

    t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    residual : NetworkX graph
        Residual network on which the algorithm is to be executed. If None, a
        new residual network is created. Default value: None.

    value_only : bool
        If True compute only the value of the maximum flow. This parameter
        will be ignored by this algorithm because it is not applicable.

    cutoff : integer, float
        If specified, the algorithm will terminate when the flow value reaches
        or exceeds the cutoff. In this case, it may be unable to immediately
        determine a minimum cut. Default value: None.

    Returns
    -------
    R : NetworkX DiGraph
        Residual network after computing the maximum flow.

    Raises
    ------
    NetworkXError
        The algorithm does not support MultiGraph and MultiDiGraph. If
        the input graph is an instance of one of these two classes, a
        NetworkXError is raised.

    NetworkXUnbounded
        If the graph has a path of infinite capacity, the value of a
        feasible flow on the graph is unbounded above and the function
        raises a NetworkXUnbounded.

    See also
    --------
    :meth:`maximum_flow`
    :meth:`minimum_cut`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Notes
    -----
    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. If :samp:`cutoff` is not
    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such
    that :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Examples
    --------
    >>> from networkx.algorithms.flow import edmonds_karp

    The functions that implement flow algorithms and output a residual
    network, such as this one, are not imported to the base NetworkX
    namespace, so you have to explicitly import them from the flow package.

    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)
    >>> R = edmonds_karp(G, "x", "y")
    >>> flow_value = nx.maximum_flow_value(G, "x", "y")
    >>> flow_value
    3.0
    >>> flow_value == R.graph["flow_value"]
    True

    """
    R = edmonds_karp_impl(G, s, t, capacity, residual, cutoff)
    R.graph["algorithm"] = "edmonds_karp"
    nx._clear_cache(R)
    return R
<|endoftext|>"""
Highest-label preflow-push algorithm for maximum flow problems.
"""

from collections import deque
from itertools import islice

import networkx as nx

from ...utils import arbitrary_element
from .utils import (
    CurrentEdge,
    GlobalRelabelThreshold,
    Level,
    build_residual_network,
    detect_unboundedness,
)

__all__ = ["preflow_push"]


def preflow_push_impl(G, s, t, capacity, residual, global_relabel_freq, value_only):
    """Implementation of the highest-label preflow-push algorithm."""
    if s not in G:
        raise nx.NetworkXError(f"node {str(s)} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"node {str(t)} not in graph")
    if s == t:
        raise nx.NetworkXError("source and sink are the same node")

    if global_relabel_freq is None:
        global_relabel_freq = 0
    if global_relabel_freq < 0:
        raise nx.NetworkXError("global_relabel_freq must be nonnegative.")

    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual

    detect_unboundedness(R, s, t)

    R_nodes = R.nodes
    R_pred = R.pred
    R_succ = R.succ

    # Initialize/reset the residual network.
    for u in R:
        R_nodes[u]["excess"] = 0
        for e in R_succ[u].values():
            e["flow"] = 0

    def reverse_bfs(src):
        """Perform a reverse breadth-first search from src in the residual
        network.
        """
        heights = {src: 0}
        q = deque([(src, 0)])
        while q:
            u, height = q.popleft()
            height += 1
            for v, attr in R_pred[u].items():
                if v not in heights and attr["flow"] < attr["capacity"]:
                    heights[v] = height
                    q.append((v, height))
        return heights

    # Initialize heights of the nodes.
    heights = reverse_bfs(t)

    if s not in heights:
        # t is not reachable from s in the residual network. The maximum flow
        # must be zero.
        R.graph["flow_value"] = 0
        return R

    n = len(R)
    # max_height represents the height of the highest level below level n with
    # at least one active node.
    max_height = max(heights[u] for u in heights if u != s)
    heights[s] = n

    grt = GlobalRelabelThreshold(n, R.size(), global_relabel_freq)

    # Initialize heights and 'current edge' data structures of the nodes.
    for u in R:
        R_nodes[u]["height"] = heights[u] if u in heights else n + 1
        R_nodes[u]["curr_edge"] = CurrentEdge(R_succ[u])

    def push(u, v, flow):
        """Push flow units of flow from u to v."""
        R_succ[u][v]["flow"] += flow
        R_succ[v][u]["flow"] -= flow
        R_nodes[u]["excess"] -= flow
        R_nodes[v]["excess"] += flow

    # The maximum flow must be nonzero now. Initialize the preflow by
    # saturating all edges emanating from s.
    for u, attr in R_succ[s].items():
        flow = attr["capacity"]
        if flow > 0:
            push(s, u, flow)

    # Partition nodes into levels.
    levels = [Level() for i in range(2 * n)]
    for u in R:
        if u != s and u != t:
            level = levels[R_nodes[u]["height"]]
            if R_nodes[u]["excess"] > 0:
                level.active.add(u)
            else:
                level.inactive.add(u)

    def activate(v):
        """Move a node from the inactive set to the active set of its level."""
        if v != s and v != t:
            level = levels[R_nodes[v]["height"]]
            if v in level.inactive:
                level.inactive.remove(v)
                level.active.add(v)

    def relabel(u):
        """Relabel a node to create an admissible edge."""
        grt.add_work(len(R_succ[u]))
        return (
            min(
                R_nodes[v]["height"]
                for v, attr in R_succ[u].items()
                if attr["flow"] < attr["capacity"]
            )
            + 1
        )

    def discharge(u, is_phase1):
        """Discharge a node until it becomes inactive or, during phase 1 (see
        below), its height reaches at least n. The node is known to have the
        largest height among active nodes.
        """
        height = R_nodes[u]["height"]
        curr_edge = R_nodes[u]["curr_edge"]
        # next_height represents the next height to examine after discharging
        # the current node. During phase 1, it is capped to below n.
        next_height = height
        levels[height].active.remove(u)
        while True:
            v, attr = curr_edge.get()
            if height == R_nodes[v]["height"] + 1 and attr["flow"] < attr["capacity"]:
                flow = min(R_nodes[u]["excess"], attr["capacity"] - attr["flow"])
                push(u, v, flow)
                activate(v)
                if R_nodes[u]["excess"] == 0:
                    # The node has become inactive.
                    levels[height].inactive.add(u)
                    break
            try:
                curr_edge.move_to_next()
            except StopIteration:
                # We have run off the end of the adjacency list, and there can
                # be no more admissible edges. Relabel the node to create one.
                height = relabel(u)
                if is_phase1 and height >= n - 1:
                    # Although the node is still active, with a height at least
                    # n - 1, it is now known to be on the s side of the minimum
                    # s-t cut. Stop processing it until phase 2.
                    levels[height].active.add(u)
                    break
                # The first relabel operation after global relabeling may not
                # increase the height of the node since the 'current edge' data
                # structure is not rewound. Use height instead of (height - 1)
                # in case other active nodes at the same level are missed.
                next_height = height
        R_nodes[u]["height"] = height
        return next_height

    def gap_heuristic(height):
        """Apply the gap heuristic."""
        # Move all nodes at levels (height + 1) to max_height to level n + 1.
        for level in islice(levels, height + 1, max_height + 1):
            for u in level.active:
                R_nodes[u]["height"] = n + 1
            for u in level.inactive:
                R_nodes[u]["height"] = n + 1
            levels[n + 1].active.update(level.active)
            level.active.clear()
            levels[n + 1].inactive.update(level.inactive)
            level.inactive.clear()

    def global_relabel(from_sink):
        """Apply the global relabeling heuristic."""
        src = t if from_sink else s
        heights = reverse_bfs(src)
        if not from_sink:
            # s must be reachable from t. Remove t explicitly.
            del heights[t]
        max_height = max(heights.values())
        if from_sink:
            # Also mark nodes from which t is unreachable for relabeling. This
            # serves the same purpose as the gap heuristic.
            for u in R:
                if u not in heights and R_nodes[u]["height"] < n:
                    heights[u] = n + 1
        else:
            # Shift the computed heights because the height of s is n.
            for u in heights:
                heights[u] += n
            max_height += n
        del heights[src]
        for u, new_height in heights.items():
            old_height = R_nodes[u]["height"]
            if new_height != old_height:
                if u in levels[old_height].active:
                    levels[old_height].active.remove(u)
                    levels[new_height].active.add(u)
                else:
                    levels[old_height].inactive.remove(u)
                    levels[new_height].inactive.add(u)
                R_nodes[u]["height"] = new_height
        return max_height

    # Phase 1: Find the maximum preflow by pushing as much flow as possible to
    # t.

    height = max_height
    while height > 0:
        # Discharge active nodes in the current level.
        while True:
            level = levels[height]
            if not level.active:
                # All active nodes in the current level have been discharged.
                # Move to the next lower level.
                height -= 1
                break
            # Record the old height and level for the gap heuristic.
            old_height = height
            old_level = level
            u = arbitrary_element(level.active)
            height = discharge(u, True)
            if grt.is_reached():
                # Global relabeling heuristic: Recompute the exact heights of
                # all nodes.
                height = global_relabel(True)
                max_height = height
                grt.clear_work()
            elif not old_level.active and not old_level.inactive:
                # Gap heuristic: If the level at old_height is empty (a 'gap'),
                # a minimum cut has been identified. All nodes with heights
                # above old_height can have their heights set to n + 1 and not
                # be further processed before a maximum preflow is found.
                gap_heuristic(old_height)
                height = old_height - 1
                max_height = height
            else:
                # Update the height of the highest level with at least one
                # active node.
                max_height = max(max_height, height)

    # A maximum preflow has been found. The excess at t is the maximum flow
    # value.
    if value_only:
        R.graph["flow_value"] = R_nodes[t]["excess"]
        return R

    # Phase 2: Convert the maximum preflow into a maximum flow by returning the
    # excess to s.

    # Relabel all nodes so that they have accurate heights.
    height = global_relabel(False)
    grt.clear_work()

    # Continue to discharge the active nodes.
    while height > n:
        # Discharge active nodes in the current level.
        while True:
            level = levels[height]
            if not level.active:
                # All active nodes in the current level have been discharged.
                # Move to the next lower level.
                height -= 1
                break
            u = arbitrary_element(level.active)
            height = discharge(u, False)
            if grt.is_reached():
                # Global relabeling heuristic.
                height = global_relabel(False)
                grt.clear_work()

    R.graph["flow_value"] = R_nodes[t]["excess"]
    return R


@nx._dispatchable(edge_attrs={"capacity": float("inf")}, returns_graph=True)
def preflow_push(
    G, s, t, capacity="capacity", residual=None, global_relabel_freq=1, value_only=False
):
    r"""Find a maximum single-commodity flow using the highest-label
    preflow-push algorithm.

    This function returns the residual network resulting after computing
    the maximum flow. See below for details about the conventions
    NetworkX uses for defining residual networks.

    This algorithm has a running time of $O(n^2 \sqrt{m})$ for $n$ nodes and
    $m$ edges.


    Parameters
    ----------
    G : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    s : node
        Source node for the flow.

    t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    residual : NetworkX graph
        Residual network on which the algorithm is to be executed. If None, a
        new residual network is created. Default value: None.

    global_relabel_freq : integer, float
        Relative frequency of applying the global relabeling heuristic to speed
        up the algorithm. If it is None, the heuristic is disabled. Default
        value: 1.

    value_only : bool
        If False, compute a maximum flow; otherwise, compute a maximum preflow
        which is enough for computing the maximum flow value. Default value:
        False.

    Returns
    -------
    R : NetworkX DiGraph
        Residual network after computing the maximum flow.

    Raises
    ------
    NetworkXError
        The algorithm does not support MultiGraph and MultiDiGraph. If
        the input graph is an instance of one of these two classes, a
        NetworkXError is raised.

    NetworkXUnbounded
        If the graph has a path of infinite capacity, the value of a
        feasible flow on the graph is unbounded above and the function
        raises a NetworkXUnbounded.

    See also
    --------
    :meth:`maximum_flow`
    :meth:`minimum_cut`
    :meth:`edmonds_karp`
    :meth:`shortest_augmenting_path`

    Notes
    -----
    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`. For each node :samp:`u` in :samp:`R`,
    :samp:`R.nodes[u]['excess']` represents the difference between flow into
    :samp:`u` and flow out of :samp:`u`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. Reachability to :samp:`t` using
    only edges :samp:`(u, v)` such that
    :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Examples
    --------
    >>> from networkx.algorithms.flow import preflow_push

    The functions that implement flow algorithms and output a residual
    network, such as this one, are not imported to the base NetworkX
    namespace, so you have to explicitly import them from the flow package.

    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)
    >>> R = preflow_push(G, "x", "y")
    >>> flow_value = nx.maximum_flow_value(G, "x", "y")
    >>> flow_value == R.graph["flow_value"]
    True
    >>> # preflow_push also stores the maximum flow value
    >>> # in the excess attribute of the sink node t
    >>> flow_value == R.nodes["y"]["excess"]
    True
    >>> # For some problems, you might only want to compute a
    >>> # maximum preflow.
    >>> R = preflow_push(G, "x", "y", value_only=True)
    >>> flow_value == R.graph["flow_value"]
    True
    >>> flow_value == R.nodes["y"]["excess"]
    True

    """
    R = preflow_push_impl(G, s, t, capacity, residual, global_relabel_freq, value_only)
    R.graph["algorithm"] = "preflow_push"
    nx._clear_cache(R)
    return R
<|endoftext|>"""
Maximum flow (and minimum cut) algorithms on capacitated graphs.
"""

import networkx as nx

from .boykovkolmogorov import boykov_kolmogorov
from .dinitz_alg import dinitz
from .edmondskarp import edmonds_karp
from .preflowpush import preflow_push
from .shortestaugmentingpath import shortest_augmenting_path
from .utils import build_flow_dict

# Define the default flow function for computing maximum flow.
default_flow_func = preflow_push

__all__ = ["maximum_flow", "maximum_flow_value", "minimum_cut", "minimum_cut_value"]


@nx._dispatchable(graphs="flowG", edge_attrs={"capacity": float("inf")})
def maximum_flow(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    """Find a maximum single-commodity flow.

    Parameters
    ----------
    flowG : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    _s : node
        Source node for the flow.

    _t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes
        in a capacitated graph. The function has to accept at least three
        parameters: a Graph or Digraph, a source node, and a target node.
        And return a residual network that follows NetworkX conventions
        (see Notes). If flow_func is None, the default maximum
        flow function (:meth:`preflow_push`) is used. See below for
        alternative algorithms. The choice of the default function may change
        from version to version and should not be relied on. Default value:
        None.

    kwargs : Any other keyword parameter is passed to the function that
        computes the maximum flow.

    Returns
    -------
    flow_value : integer, float
        Value of the maximum flow, i.e., net outflow from the source.

    flow_dict : dict
        A dictionary containing the value of the flow that went through
        each edge.

    Raises
    ------
    NetworkXError
        The algorithm does not support MultiGraph and MultiDiGraph. If
        the input graph is an instance of one of these two classes, a
        NetworkXError is raised.

    NetworkXUnbounded
        If the graph has a path of infinite capacity, the value of a
        feasible flow on the graph is unbounded above and the function
        raises a NetworkXUnbounded.

    See also
    --------
    :meth:`maximum_flow_value`
    :meth:`minimum_cut`
    :meth:`minimum_cut_value`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Notes
    -----
    The function used in the flow_func parameter has to return a residual
    network that follows NetworkX conventions:

    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. Reachability to :samp:`t` using
    only edges :samp:`(u, v)` such that
    :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Specific algorithms may store extra data in :samp:`R`.

    The function should supports an optional boolean parameter value_only. When
    True, it can optionally terminate the algorithm as soon as the maximum flow
    value and the minimum cut can be determined.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)

    maximum_flow returns both the value of the maximum flow and a
    dictionary with all flows.

    >>> flow_value, flow_dict = nx.maximum_flow(G, "x", "y")
    >>> flow_value
    3.0
    >>> print(flow_dict["x"]["b"])
    1.0

    You can also use alternative algorithms for computing the
    maximum flow by using the flow_func parameter.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> flow_value == nx.maximum_flow(G, "x", "y", flow_func=shortest_augmenting_path)[
    ...     0
    ... ]
    True

    """
    if flow_func is None:
        if kwargs:
            raise nx.NetworkXError(
                "You have to explicitly set a flow_func if"
                " you need to pass parameters via kwargs."
            )
        flow_func = default_flow_func

    if not callable(flow_func):
        raise nx.NetworkXError("flow_func has to be callable.")

    R = flow_func(flowG, _s, _t, capacity=capacity, value_only=False, **kwargs)
    flow_dict = build_flow_dict(flowG, R)

    return (R.graph["flow_value"], flow_dict)


@nx._dispatchable(graphs="flowG", edge_attrs={"capacity": float("inf")})
def maximum_flow_value(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    """Find the value of maximum single-commodity flow.

    Parameters
    ----------
    flowG : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    _s : node
        Source node for the flow.

    _t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes
        in a capacitated graph. The function has to accept at least three
        parameters: a Graph or Digraph, a source node, and a target node.
        And return a residual network that follows NetworkX conventions
        (see Notes). If flow_func is None, the default maximum
        flow function (:meth:`preflow_push`) is used. See below for
        alternative algorithms. The choice of the default function may change
        from version to version and should not be relied on. Default value:
        None.

    kwargs : Any other keyword parameter is passed to the function that
        computes the maximum flow.

    Returns
    -------
    flow_value : integer, float
        Value of the maximum flow, i.e., net outflow from the source.

    Raises
    ------
    NetworkXError
        The algorithm does not support MultiGraph and MultiDiGraph. If
        the input graph is an instance of one of these two classes, a
        NetworkXError is raised.

    NetworkXUnbounded
        If the graph has a path of infinite capacity, the value of a
        feasible flow on the graph is unbounded above and the function
        raises a NetworkXUnbounded.

    See also
    --------
    :meth:`maximum_flow`
    :meth:`minimum_cut`
    :meth:`minimum_cut_value`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Notes
    -----
    The function used in the flow_func parameter has to return a residual
    network that follows NetworkX conventions:

    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. Reachability to :samp:`t` using
    only edges :samp:`(u, v)` such that
    :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Specific algorithms may store extra data in :samp:`R`.

    The function should supports an optional boolean parameter value_only. When
    True, it can optionally terminate the algorithm as soon as the maximum flow
    value and the minimum cut can be determined.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)

    maximum_flow_value computes only the value of the
    maximum flow:

    >>> flow_value = nx.maximum_flow_value(G, "x", "y")
    >>> flow_value
    3.0

    You can also use alternative algorithms for computing the
    maximum flow by using the flow_func parameter.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> flow_value == nx.maximum_flow_value(
    ...     G, "x", "y", flow_func=shortest_augmenting_path
    ... )
    True

    """
    if flow_func is None:
        if kwargs:
            raise nx.NetworkXError(
                "You have to explicitly set a flow_func if"
                " you need to pass parameters via kwargs."
            )
        flow_func = default_flow_func

    if not callable(flow_func):
        raise nx.NetworkXError("flow_func has to be callable.")

    R = flow_func(flowG, _s, _t, capacity=capacity, value_only=True, **kwargs)

    return R.graph["flow_value"]


@nx._dispatchable(graphs="flowG", edge_attrs={"capacity": float("inf")})
def minimum_cut(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    """Compute the value and the node partition of a minimum (s, t)-cut.

    Use the max-flow min-cut theorem, i.e., the capacity of a minimum
    capacity cut is equal to the flow value of a maximum flow.

    Parameters
    ----------
    flowG : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    _s : node
        Source node for the flow.

    _t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes
        in a capacitated graph. The function has to accept at least three
        parameters: a Graph or Digraph, a source node, and a target node.
        And return a residual network that follows NetworkX conventions
        (see Notes). If flow_func is None, the default maximum
        flow function (:meth:`preflow_push`) is used. See below for
        alternative algorithms. The choice of the default function may change
        from version to version and should not be relied on. Default value:
        None.

    kwargs : Any other keyword parameter is passed to the function that
        computes the maximum flow.

    Returns
    -------
    cut_value : integer, float
        Value of the minimum cut.

    partition : pair of node sets
        A partitioning of the nodes that defines a minimum cut.

    Raises
    ------
    NetworkXUnbounded
        If the graph has a path of infinite capacity, all cuts have
        infinite capacity and the function raises a NetworkXError.

    See also
    --------
    :meth:`maximum_flow`
    :meth:`maximum_flow_value`
    :meth:`minimum_cut_value`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Notes
    -----
    The function used in the flow_func parameter has to return a residual
    network that follows NetworkX conventions:

    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. Reachability to :samp:`t` using
    only edges :samp:`(u, v)` such that
    :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Specific algorithms may store extra data in :samp:`R`.

    The function should supports an optional boolean parameter value_only. When
    True, it can optionally terminate the algorithm as soon as the maximum flow
    value and the minimum cut can be determined.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)

    minimum_cut computes both the value of the
    minimum cut and the node partition:

    >>> cut_value, partition = nx.minimum_cut(G, "x", "y")
    >>> reachable, non_reachable = partition

    'partition' here is a tuple with the two sets of nodes that define
    the minimum cut. You can compute the cut set of edges that induce
    the minimum cut as follows:

    >>> cutset = set()
    >>> for u, nbrs in ((n, G[n]) for n in reachable):
    ...     cutset.update((u, v) for v in nbrs if v in non_reachable)
    >>> print(sorted(cutset))
    [('c', 'y'), ('x', 'b')]
    >>> cut_value == sum(G.edges[u, v]["capacity"] for (u, v) in cutset)
    True

    You can also use alternative algorithms for computing the
    minimum cut by using the flow_func parameter.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> cut_value == nx.minimum_cut(G, "x", "y", flow_func=shortest_augmenting_path)[0]
    True

    """
    if flow_func is None:
        if kwargs:
            raise nx.NetworkXError(
                "You have to explicitly set a flow_func if"
                " you need to pass parameters via kwargs."
            )
        flow_func = default_flow_func

    if not callable(flow_func):
        raise nx.NetworkXError("flow_func has to be callable.")

    if kwargs.get("cutoff") is not None and flow_func is preflow_push:
        raise nx.NetworkXError("cutoff should not be specified.")

    R = flow_func(flowG, _s, _t, capacity=capacity, value_only=True, **kwargs)
    # Remove saturated edges from the residual network
    cutset = [(u, v, d) for u, v, d in R.edges(data=True) if d["flow"] == d["capacity"]]
    R.remove_edges_from(cutset)

    # Then, reachable and non reachable nodes from source in the
    # residual network form the node partition that defines
    # the minimum cut.
    non_reachable = set(dict(nx.shortest_path_length(R, target=_t)))
    partition = (set(flowG) - non_reachable, non_reachable)
    # Finally add again cutset edges to the residual network to make
    # sure that it is reusable.
    R.add_edges_from(cutset)
    return (R.graph["flow_value"], partition)


@nx._dispatchable(graphs="flowG", edge_attrs={"capacity": float("inf")})
def minimum_cut_value(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    """Compute the value of a minimum (s, t)-cut.

    Use the max-flow min-cut theorem, i.e., the capacity of a minimum
    capacity cut is equal to the flow value of a maximum flow.

    Parameters
    ----------
    flowG : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    _s : node
        Source node for the flow.

    _t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes
        in a capacitated graph. The function has to accept at least three
        parameters: a Graph or Digraph, a source node, and a target node.
        And return a residual network that follows NetworkX conventions
        (see Notes). If flow_func is None, the default maximum
        flow function (:meth:`preflow_push`) is used. See below for
        alternative algorithms. The choice of the default function may change
        from version to version and should not be relied on. Default value:
        None.

    kwargs : Any other keyword parameter is passed to the function that
        computes the maximum flow.

    Returns
    -------
    cut_value : integer, float
        Value of the minimum cut.

    Raises
    ------
    NetworkXUnbounded
        If the graph has a path of infinite capacity, all cuts have
        infinite capacity and the function raises a NetworkXError.

    See also
    --------
    :meth:`maximum_flow`
    :meth:`maximum_flow_value`
    :meth:`minimum_cut`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Notes
    -----
    The function used in the flow_func parameter has to return a residual
    network that follows NetworkX conventions:

    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. Reachability to :samp:`t` using
    only edges :samp:`(u, v)` such that
    :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Specific algorithms may store extra data in :samp:`R`.

    The function should supports an optional boolean parameter value_only. When
    True, it can optionally terminate the algorithm as soon as the maximum flow
    value and the minimum cut can be determined.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)

    minimum_cut_value computes only the value of the
    minimum cut:

    >>> cut_value = nx.minimum_cut_value(G, "x", "y")
    >>> cut_value
    3.0

    You can also use alternative algorithms for computing the
    minimum cut by using the flow_func parameter.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> cut_value == nx.minimum_cut_value(
    ...     G, "x", "y", flow_func=shortest_augmenting_path
    ... )
    True

    """
    if flow_func is None:
        if kwargs:
            raise nx.NetworkXError(
                "You have to explicitly set a flow_func if"
                " you need to pass parameters via kwargs."
            )
        flow_func = default_flow_func

    if not callable(flow_func):
        raise nx.NetworkXError("flow_func has to be callable.")

    if kwargs.get("cutoff") is not None and flow_func is preflow_push:
        raise nx.NetworkXError("cutoff should not be specified.")

    R = flow_func(flowG, _s, _t, capacity=capacity, value_only=True, **kwargs)

    return R.graph["flow_value"]
<|endoftext|>"""
Gomory-Hu tree of undirected Graphs.
"""

import networkx as nx
from networkx.utils import not_implemented_for

from .edmondskarp import edmonds_karp
from .utils import build_residual_network

default_flow_func = edmonds_karp

__all__ = ["gomory_hu_tree"]


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs={"capacity": float("inf")}, returns_graph=True)
def gomory_hu_tree(G, capacity="capacity", flow_func=None):
    r"""Returns the Gomory-Hu tree of an undirected graph G.

    A Gomory-Hu tree of an undirected graph with capacities is a
    weighted tree that represents the minimum s-t cuts for all s-t
    pairs in the graph.

    It only requires `n-1` minimum cut computations instead of the
    obvious `n(n-1)/2`. The tree represents all s-t cuts as the
    minimum cut value among any pair of nodes is the minimum edge
    weight in the shortest path between the two nodes in the
    Gomory-Hu tree.

    The Gomory-Hu tree also has the property that removing the
    edge with the minimum weight in the shortest path between
    any two nodes leaves two connected components that form
    a partition of the nodes in G that defines the minimum s-t
    cut.

    See Examples section below for details.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    flow_func : function
        Function to perform the underlying flow computations. Default value
        :func:`edmonds_karp`. This function performs better in sparse graphs
        with right tailed degree distributions.
        :func:`shortest_augmenting_path` will perform better in denser
        graphs.

    Returns
    -------
    Tree : NetworkX graph
        A NetworkX graph representing the Gomory-Hu tree of the input graph.

    Raises
    ------
    NetworkXNotImplemented
        Raised if the input graph is directed.

    NetworkXError
        Raised if the input graph is an empty Graph.

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> nx.set_edge_attributes(G, 1, "capacity")
    >>> T = nx.gomory_hu_tree(G)
    >>> # The value of the minimum cut between any pair
    ... # of nodes in G is the minimum edge weight in the
    ... # shortest path between the two nodes in the
    ... # Gomory-Hu tree.
    ... def minimum_edge_weight_in_shortest_path(T, u, v):
    ...     path = nx.shortest_path(T, u, v, weight="weight")
    ...     return min((T[u][v]["weight"], (u, v)) for (u, v) in zip(path, path[1:]))
    >>> u, v = 0, 33
    >>> cut_value, edge = minimum_edge_weight_in_shortest_path(T, u, v)
    >>> cut_value
    10
    >>> nx.minimum_cut_value(G, u, v)
    10
    >>> # The Gomory-Hu tree also has the property that removing the
    ... # edge with the minimum weight in the shortest path between
    ... # any two nodes leaves two connected components that form
    ... # a partition of the nodes in G that defines the minimum s-t
    ... # cut.
    ... cut_value, edge = minimum_edge_weight_in_shortest_path(T, u, v)
    >>> T.remove_edge(*edge)
    >>> U, V = list(nx.connected_components(T))
    >>> # Thus U and V form a partition that defines a minimum cut
    ... # between u and v in G. You can compute the edge cut set,
    ... # that is, the set of edges that if removed from G will
    ... # disconnect u from v in G, with this information:
    ... cutset = set()
    >>> for x, nbrs in ((n, G[n]) for n in U):
    ...     cutset.update((x, y) for y in nbrs if y in V)
    >>> # Because we have set the capacities of all edges to 1
    ... # the cutset contains ten edges
    ... len(cutset)
    10
    >>> # You can use any maximum flow algorithm for the underlying
    ... # flow computations using the argument flow_func
    ... from networkx.algorithms import flow
    >>> T = nx.gomory_hu_tree(G, flow_func=flow.boykov_kolmogorov)
    >>> cut_value, edge = minimum_edge_weight_in_shortest_path(T, u, v)
    >>> cut_value
    10
    >>> nx.minimum_cut_value(G, u, v, flow_func=flow.boykov_kolmogorov)
    10

    Notes
    -----
    This implementation is based on Gusfield approach [1]_ to compute
    Gomory-Hu trees, which does not require node contractions and has
    the same computational complexity than the original method.

    See also
    --------
    :func:`minimum_cut`
    :func:`maximum_flow`

    References
    ----------
    .. [1] Gusfield D: Very simple methods for all pairs network flow analysis.
           SIAM J Comput 19(1):143-155, 1990.

    """
    if flow_func is None:
        flow_func = default_flow_func

    if len(G) == 0:  # empty graph
        msg = "Empty Graph does not have a Gomory-Hu tree representation"
        raise nx.NetworkXError(msg)

    # Start the tree as a star graph with an arbitrary node at the center
    tree = {}
    labels = {}
    iter_nodes = iter(G)
    root = next(iter_nodes)
    for n in iter_nodes:
        tree[n] = root

    # Reuse residual network
    R = build_residual_network(G, capacity)

    # For all the leaves in the star graph tree (that is n-1 nodes).
    for source in tree:
        # Find neighbor in the tree
        target = tree[source]
        # compute minimum cut
        cut_value, partition = nx.minimum_cut(
            G, source, target, capacity=capacity, flow_func=flow_func, residual=R
        )
        labels[(source, target)] = cut_value
        # Update the tree
        # Source will always be in partition[0] and target in partition[1]
        for node in partition[0]:
            if node != source and node in tree and tree[node] == target:
                tree[node] = source
                labels[node, source] = labels.get((node, target), cut_value)
        #
        if target != root and tree[target] in partition[0]:
            labels[source, tree[target]] = labels[target, tree[target]]
            labels[target, source] = cut_value
            tree[source] = tree[target]
            tree[target] = source

    # Build the tree
    T = nx.Graph()
    T.add_nodes_from(G)
    T.add_weighted_edges_from(((u, v, labels[u, v]) for u, v in tree.items()))
    return T
<|endoftext|>"""
Utility classes and functions for network flow algorithms.
"""

from collections import deque

import networkx as nx

__all__ = [
    "CurrentEdge",
    "Level",
    "GlobalRelabelThreshold",
    "build_residual_network",
    "detect_unboundedness",
    "build_flow_dict",
]


class CurrentEdge:
    """Mechanism for iterating over out-edges incident to a node in a circular
    manner. StopIteration exception is raised when wraparound occurs.
    """

    __slots__ = ("_edges", "_it", "_curr")

    def __init__(self, edges):
        self._edges = edges
        if self._edges:
            self._rewind()

    def get(self):
        return self._curr

    def move_to_next(self):
        try:
            self._curr = next(self._it)
        except StopIteration:
            self._rewind()
            raise

    def _rewind(self):
        self._it = iter(self._edges.items())
        self._curr = next(self._it)


class Level:
    """Active and inactive nodes in a level."""

    __slots__ = ("active", "inactive")

    def __init__(self):
        self.active = set()
        self.inactive = set()


class GlobalRelabelThreshold:
    """Measurement of work before the global relabeling heuristic should be
    applied.
    """

    def __init__(self, n, m, freq):
        self._threshold = (n + m) / freq if freq else float("inf")
        self._work = 0

    def add_work(self, work):
        self._work += work

    def is_reached(self):
        return self._work >= self._threshold

    def clear_work(self):
        self._work = 0


@nx._dispatchable(edge_attrs={"capacity": float("inf")}, returns_graph=True)
def build_residual_network(G, capacity):
    """Build a residual network and initialize a zero flow.

    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. If :samp:`cutoff` is not
    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such
    that :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    """
    if G.is_multigraph():
        raise nx.NetworkXError("MultiGraph and MultiDiGraph not supported (yet).")

    R = nx.DiGraph()
    R.__networkx_cache__ = None  # Disable caching
    R.add_nodes_from(G)

    inf = float("inf")
    # Extract edges with positive capacities. Self loops excluded.
    edge_list = [
        (u, v, attr)
        for u, v, attr in G.edges(data=True)
        if u != v and attr.get(capacity, inf) > 0
    ]
    # Simulate infinity with three times the sum of the finite edge capacities
    # or any positive value if the sum is zero. This allows the
    # infinite-capacity edges to be distinguished for unboundedness detection
    # and directly participate in residual capacity calculation. If the maximum
    # flow is finite, these edges cannot appear in the minimum cut and thus
    # guarantee correctness. Since the residual capacity of an
    # infinite-capacity edge is always at least 2/3 of inf, while that of an
    # finite-capacity edge is at most 1/3 of inf, if an operation moves more
    # than 1/3 of inf units of flow to t, there must be an infinite-capacity
    # s-t path in G.
    inf = (
        3
        * sum(
            attr[capacity]
            for u, v, attr in edge_list
            if capacity in attr and attr[capacity] != inf
        )
        or 1
    )
    if G.is_directed():
        for u, v, attr in edge_list:
            r = min(attr.get(capacity, inf), inf)
            if not R.has_edge(u, v):
                # Both (u, v) and (v, u) must be present in the residual
                # network.
                R.add_edge(u, v, capacity=r)
                R.add_edge(v, u, capacity=0)
            else:
                # The edge (u, v) was added when (v, u) was visited.
                R[u][v]["capacity"] = r
    else:
        for u, v, attr in edge_list:
            # Add a pair of edges with equal residual capacities.
            r = min(attr.get(capacity, inf), inf)
            R.add_edge(u, v, capacity=r)
            R.add_edge(v, u, capacity=r)

    # Record the value simulating infinity.
    R.graph["inf"] = inf

    return R


@nx._dispatchable(
    graphs="R",
    preserve_edge_attrs={"R": {"capacity": float("inf")}},
    preserve_graph_attrs=True,
)
def detect_unboundedness(R, s, t):
    """Detect an infinite-capacity s-t path in R."""
    q = deque([s])
    seen = {s}
    inf = R.graph["inf"]
    while q:
        u = q.popleft()
        for v, attr in R[u].items():
            if attr["capacity"] == inf and v not in seen:
                if v == t:
                    raise nx.NetworkXUnbounded(
                        "Infinite capacity path, flow unbounded above."
                    )
                seen.add(v)
                q.append(v)


@nx._dispatchable(graphs={"G": 0, "R": 1}, preserve_edge_attrs={"R": {"flow": None}})
def build_flow_dict(G, R):
    """Build a flow dictionary from a residual network."""
    flow_dict = {}
    for u in G:
        flow_dict[u] = {v: 0 for v in G[u]}
        flow_dict[u].update(
            (v, attr["flow"]) for v, attr in R[u].items() if attr["flow"] > 0
        )
    return flow_dict
<|endoftext|>"""
Minimum cost flow algorithms on directed connected graphs.
"""

__all__ = ["min_cost_flow_cost", "min_cost_flow", "cost_of_flow", "max_flow_min_cost"]

import networkx as nx


@nx._dispatchable(
    node_attrs="demand", edge_attrs={"capacity": float("inf"), "weight": 0}
)
def min_cost_flow_cost(G, demand="demand", capacity="capacity", weight="weight"):
    r"""Find the cost of a minimum cost flow satisfying all demands in digraph G.

    G is a digraph with edge costs and capacities and in which nodes
    have demand, i.e., they want to send or receive some amount of
    flow. A negative demand means that the node wants to send flow, a
    positive demand means that the node want to receive flow. A flow on
    the digraph G satisfies all demand if the net flow into each node
    is equal to the demand of that node.

    Parameters
    ----------
    G : NetworkX graph
        DiGraph on which a minimum cost flow satisfying all demands is
        to be found.

    demand : string
        Nodes of the graph G are expected to have an attribute demand
        that indicates how much flow a node wants to send (negative
        demand) or receive (positive demand). Note that the sum of the
        demands should be 0 otherwise the problem in not feasible. If
        this attribute is not present, a node is considered to have 0
        demand. Default value: 'demand'.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    weight : string
        Edges of the graph G are expected to have an attribute weight
        that indicates the cost incurred by sending one unit of flow on
        that edge. If not present, the weight is considered to be 0.
        Default value: 'weight'.

    Returns
    -------
    flowCost : integer, float
        Cost of a minimum cost flow satisfying all demands.

    Raises
    ------
    NetworkXError
        This exception is raised if the input graph is not directed or
        not connected.

    NetworkXUnfeasible
        This exception is raised in the following situations:

            * The sum of the demands is not zero. Then, there is no
              flow satisfying all demands.
            * There is no flow satisfying all demand.

    NetworkXUnbounded
        This exception is raised if the digraph G has a cycle of
        negative cost and infinite capacity. Then, the cost of a flow
        satisfying all demands is unbounded below.

    See also
    --------
    cost_of_flow, max_flow_min_cost, min_cost_flow, network_simplex

    Notes
    -----
    This algorithm is not guaranteed to work if edge weights or demands
    are floating point numbers (overflows and roundoff errors can
    cause problems). As a workaround you can use integer numbers by
    multiplying the relevant edge attributes by a convenient
    constant factor (eg 100).

    Examples
    --------
    A simple example of a min cost flow problem.

    >>> G = nx.DiGraph()
    >>> G.add_node("a", demand=-5)
    >>> G.add_node("d", demand=5)
    >>> G.add_edge("a", "b", weight=3, capacity=4)
    >>> G.add_edge("a", "c", weight=6, capacity=10)
    >>> G.add_edge("b", "d", weight=1, capacity=9)
    >>> G.add_edge("c", "d", weight=2, capacity=5)
    >>> flowCost = nx.min_cost_flow_cost(G)
    >>> flowCost
    24
    """
    return nx.network_simplex(G, demand=demand, capacity=capacity, weight=weight)[0]


@nx._dispatchable(
    node_attrs="demand", edge_attrs={"capacity": float("inf"), "weight": 0}
)
def min_cost_flow(G, demand="demand", capacity="capacity", weight="weight"):
    r"""Returns a minimum cost flow satisfying all demands in digraph G.

    G is a digraph with edge costs and capacities and in which nodes
    have demand, i.e., they want to send or receive some amount of
    flow. A negative demand means that the node wants to send flow, a
    positive demand means that the node want to receive flow. A flow on
    the digraph G satisfies all demand if the net flow into each node
    is equal to the demand of that node.

    Parameters
    ----------
    G : NetworkX graph
        DiGraph on which a minimum cost flow satisfying all demands is
        to be found.

    demand : string
        Nodes of the graph G are expected to have an attribute demand
        that indicates how much flow a node wants to send (negative
        demand) or receive (positive demand). Note that the sum of the
        demands should be 0 otherwise the problem in not feasible. If
        this attribute is not present, a node is considered to have 0
        demand. Default value: 'demand'.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    weight : string
        Edges of the graph G are expected to have an attribute weight
        that indicates the cost incurred by sending one unit of flow on
        that edge. If not present, the weight is considered to be 0.
        Default value: 'weight'.

    Returns
    -------
    flowDict : dictionary
        Dictionary of dictionaries keyed by nodes such that
        flowDict[u][v] is the flow edge (u, v).

    Raises
    ------
    NetworkXError
        This exception is raised if the input graph is not directed or
        not connected.

    NetworkXUnfeasible
        This exception is raised in the following situations:

            * The sum of the demands is not zero. Then, there is no
              flow satisfying all demands.
            * There is no flow satisfying all demand.

    NetworkXUnbounded
        This exception is raised if the digraph G has a cycle of
        negative cost and infinite capacity. Then, the cost of a flow
        satisfying all demands is unbounded below.

    See also
    --------
    cost_of_flow, max_flow_min_cost, min_cost_flow_cost, network_simplex

    Notes
    -----
    This algorithm is not guaranteed to work if edge weights or demands
    are floating point numbers (overflows and roundoff errors can
    cause problems). As a workaround you can use integer numbers by
    multiplying the relevant edge attributes by a convenient
    constant factor (eg 100).

    Examples
    --------
    A simple example of a min cost flow problem.

    >>> G = nx.DiGraph()
    >>> G.add_node("a", demand=-5)
    >>> G.add_node("d", demand=5)
    >>> G.add_edge("a", "b", weight=3, capacity=4)
    >>> G.add_edge("a", "c", weight=6, capacity=10)
    >>> G.add_edge("b", "d", weight=1, capacity=9)
    >>> G.add_edge("c", "d", weight=2, capacity=5)
    >>> flowDict = nx.min_cost_flow(G)
    >>> flowDict
    {'a': {'b': 4, 'c': 1}, 'd': {}, 'b': {'d': 4}, 'c': {'d': 1}}
    """
    return nx.network_simplex(G, demand=demand, capacity=capacity, weight=weight)[1]


@nx._dispatchable(edge_attrs={"weight": 0})
def cost_of_flow(G, flowDict, weight="weight"):
    """Compute the cost of the flow given by flowDict on graph G.

    Note that this function does not check for the validity of the
    flow flowDict. This function will fail if the graph G and the
    flow don't have the same edge set.

    Parameters
    ----------
    G : NetworkX graph
        DiGraph on which a minimum cost flow satisfying all demands is
        to be found.

    weight : string
        Edges of the graph G are expected to have an attribute weight
        that indicates the cost incurred by sending one unit of flow on
        that edge. If not present, the weight is considered to be 0.
        Default value: 'weight'.

    flowDict : dictionary
        Dictionary of dictionaries keyed by nodes such that
        flowDict[u][v] is the flow edge (u, v).

    Returns
    -------
    cost : Integer, float
        The total cost of the flow. This is given by the sum over all
        edges of the product of the edge's flow and the edge's weight.

    See also
    --------
    max_flow_min_cost, min_cost_flow, min_cost_flow_cost, network_simplex

    Notes
    -----
    This algorithm is not guaranteed to work if edge weights or demands
    are floating point numbers (overflows and roundoff errors can
    cause problems). As a workaround you can use integer numbers by
    multiplying the relevant edge attributes by a convenient
    constant factor (eg 100).

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_node("a", demand=-5)
    >>> G.add_node("d", demand=5)
    >>> G.add_edge("a", "b", weight=3, capacity=4)
    >>> G.add_edge("a", "c", weight=6, capacity=10)
    >>> G.add_edge("b", "d", weight=1, capacity=9)
    >>> G.add_edge("c", "d", weight=2, capacity=5)
    >>> flowDict = nx.min_cost_flow(G)
    >>> flowDict
    {'a': {'b': 4, 'c': 1}, 'd': {}, 'b': {'d': 4}, 'c': {'d': 1}}
    >>> nx.cost_of_flow(G, flowDict)
    24
    """
    return sum((flowDict[u][v] * d.get(weight, 0) for u, v, d in G.edges(data=True)))


@nx._dispatchable(edge_attrs={"capacity": float("inf"), "weight": 0})
def max_flow_min_cost(G, s, t, capacity="capacity", weight="weight"):
    """Returns a maximum (s, t)-flow of minimum cost.

    G is a digraph with edge costs and capacities. There is a source
    node s and a sink node t. This function finds a maximum flow from
    s to t whose total cost is minimized.

    Parameters
    ----------
    G : NetworkX graph
        DiGraph on which a minimum cost flow satisfying all demands is
        to be found.

    s: node label
        Source of the flow.

    t: node label
        Destination of the flow.

    capacity: string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    weight: string
        Edges of the graph G are expected to have an attribute weight
        that indicates the cost incurred by sending one unit of flow on
        that edge. If not present, the weight is considered to be 0.
        Default value: 'weight'.

    Returns
    -------
    flowDict: dictionary
        Dictionary of dictionaries keyed by nodes such that
        flowDict[u][v] is the flow edge (u, v).

    Raises
    ------
    NetworkXError
        This exception is raised if the input graph is not directed or
        not connected.

    NetworkXUnbounded
        This exception is raised if there is an infinite capacity path
        from s to t in G. In this case there is no maximum flow. This
        exception is also raised if the digraph G has a cycle of
        negative cost and infinite capacity. Then, the cost of a flow
        is unbounded below.

    See also
    --------
    cost_of_flow, min_cost_flow, min_cost_flow_cost, network_simplex

    Notes
    -----
    This algorithm is not guaranteed to work if edge weights or demands
    are floating point numbers (overflows and roundoff errors can
    cause problems). As a workaround you can use integer numbers by
    multiplying the relevant edge attributes by a convenient
    constant factor (eg 100).

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from(
    ...     [
    ...         (1, 2, {"capacity": 12, "weight": 4}),
    ...         (1, 3, {"capacity": 20, "weight": 6}),
    ...         (2, 3, {"capacity": 6, "weight": -3}),
    ...         (2, 6, {"capacity": 14, "weight": 1}),
    ...         (3, 4, {"weight": 9}),
    ...         (3, 5, {"capacity": 10, "weight": 5}),
    ...         (4, 2, {"capacity": 19, "weight": 13}),
    ...         (4, 5, {"capacity": 4, "weight": 0}),
    ...         (5, 7, {"capacity": 28, "weight": 2}),
    ...         (6, 5, {"capacity": 11, "weight": 1}),
    ...         (6, 7, {"weight": 8}),
    ...         (7, 4, {"capacity": 6, "weight": 6}),
    ...     ]
    ... )
    >>> mincostFlow = nx.max_flow_min_cost(G, 1, 7)
    >>> mincost = nx.cost_of_flow(G, mincostFlow)
    >>> mincost
    373
    >>> from networkx.algorithms.flow import maximum_flow
    >>> maxFlow = maximum_flow(G, 1, 7)[1]
    >>> nx.cost_of_flow(G, maxFlow) >= mincost
    True
    >>> mincostFlowValue = sum((mincostFlow[u][7] for u in G.predecessors(7))) - sum(
    ...     (mincostFlow[7][v] for v in G.successors(7))
    ... )
    >>> mincostFlowValue == nx.maximum_flow_value(G, 1, 7)
    True

    """
    maxFlow = nx.maximum_flow_value(G, s, t, capacity=capacity)
    H = nx.DiGraph(G)
    H.add_node(s, demand=-maxFlow)
    H.add_node(t, demand=maxFlow)
    return min_cost_flow(H, capacity=capacity, weight=weight)
<|endoftext|>"""
Capacity scaling minimum cost flow algorithm.
"""

__all__ = ["capacity_scaling"]

from itertools import chain
from math import log

import networkx as nx

from ...utils import BinaryHeap, arbitrary_element, not_implemented_for


def _detect_unboundedness(R):
    """Detect infinite-capacity negative cycles."""
    G = nx.DiGraph()
    G.add_nodes_from(R)

    # Value simulating infinity.
    inf = R.graph["inf"]
    # True infinity.
    f_inf = float("inf")
    for u in R:
        for v, e in R[u].items():
            # Compute the minimum weight of infinite-capacity (u, v) edges.
            w = f_inf
            for k, e in e.items():
                if e["capacity"] == inf:
                    w = min(w, e["weight"])
            if w != f_inf:
                G.add_edge(u, v, weight=w)

    if nx.negative_edge_cycle(G):
        raise nx.NetworkXUnbounded(
            "Negative cost cycle of infinite capacity found. "
            "Min cost flow may be unbounded below."
        )


@not_implemented_for("undirected")
def _build_residual_network(G, demand, capacity, weight):
    """Build a residual network and initialize a zero flow."""
    if sum(G.nodes[u].get(demand, 0) for u in G) != 0:
        raise nx.NetworkXUnfeasible("Sum of the demands should be 0.")

    R = nx.MultiDiGraph()
    R.add_nodes_from(
        (u, {"excess": -G.nodes[u].get(demand, 0), "potential": 0}) for u in G
    )

    inf = float("inf")
    # Detect selfloops with infinite capacities and negative weights.
    for u, v, e in nx.selfloop_edges(G, data=True):
        if e.get(weight, 0) < 0 and e.get(capacity, inf) == inf:
            raise nx.NetworkXUnbounded(
                "Negative cost cycle of infinite capacity found. "
                "Min cost flow may be unbounded below."
            )

    # Extract edges with positive capacities. Self loops excluded.
    if G.is_multigraph():
        edge_list = [
            (u, v, k, e)
            for u, v, k, e in G.edges(data=True, keys=True)
            if u != v and e.get(capacity, inf) > 0
        ]
    else:
        edge_list = [
            (u, v, 0, e)
            for u, v, e in G.edges(data=True)
            if u != v and e.get(capacity, inf) > 0
        ]
    # Simulate infinity with the larger of the sum of absolute node imbalances
    # the sum of finite edge capacities or any positive value if both sums are
    # zero. This allows the infinite-capacity edges to be distinguished for
    # unboundedness detection and directly participate in residual capacity
    # calculation.
    inf = (
        max(
            sum(abs(R.nodes[u]["excess"]) for u in R),
            2
            * sum(
                e[capacity]
                for u, v, k, e in edge_list
                if capacity in e and e[capacity] != inf
            ),
        )
        or 1
    )
    for u, v, k, e in edge_list:
        r = min(e.get(capacity, inf), inf)
        w = e.get(weight, 0)
        # Add both (u, v) and (v, u) into the residual network marked with the
        # original key. (key[1] == True) indicates the (u, v) is in the
        # original network.
        R.add_edge(u, v, key=(k, True), capacity=r, weight=w, flow=0)
        R.add_edge(v, u, key=(k, False), capacity=0, weight=-w, flow=0)

    # Record the value simulating infinity.
    R.graph["inf"] = inf

    _detect_unboundedness(R)

    return R


def _build_flow_dict(G, R, capacity, weight):
    """Build a flow dictionary from a residual network."""
    inf = float("inf")
    flow_dict = {}
    if G.is_multigraph():
        for u in G:
            flow_dict[u] = {}
            for v, es in G[u].items():
                flow_dict[u][v] = {
                    # Always saturate negative selfloops.
                    k: (
                        0
                        if (
                            u != v or e.get(capacity, inf) <= 0 or e.get(weight, 0) >= 0
                        )
                        else e[capacity]
                    )
                    for k, e in es.items()
                }
            for v, es in R[u].items():
                if v in flow_dict[u]:
                    flow_dict[u][v].update(
                        (k[0], e["flow"]) for k, e in es.items() if e["flow"] > 0
                    )
    else:
        for u in G:
            flow_dict[u] = {
                # Always saturate negative selfloops.
                v: (
                    0
                    if (u != v or e.get(capacity, inf) <= 0 or e.get(weight, 0) >= 0)
                    else e[capacity]
                )
                for v, e in G[u].items()
            }
            flow_dict[u].update(
                (v, e["flow"])
                for v, es in R[u].items()
                for e in es.values()
                if e["flow"] > 0
            )
    return flow_dict


@nx._dispatchable(
    node_attrs="demand", edge_attrs={"capacity": float("inf"), "weight": 0}
)
def capacity_scaling(
    G, demand="demand", capacity="capacity", weight="weight", heap=BinaryHeap
):
    r"""Find a minimum cost flow satisfying all demands in digraph G.

    This is a capacity scaling successive shortest augmenting path algorithm.

    G is a digraph with edge costs and capacities and in which nodes
    have demand, i.e., they want to send or receive some amount of
    flow. A negative demand means that the node wants to send flow, a
    positive demand means that the node want to receive flow. A flow on
    the digraph G satisfies all demand if the net flow into each node
    is equal to the demand of that node.

    Parameters
    ----------
    G : NetworkX graph
        DiGraph or MultiDiGraph on which a minimum cost flow satisfying all
        demands is to be found.

    demand : string
        Nodes of the graph G are expected to have an attribute demand
        that indicates how much flow a node wants to send (negative
        demand) or receive (positive demand). Note that the sum of the
        demands should be 0 otherwise the problem in not feasible. If
        this attribute is not present, a node is considered to have 0
        demand. Default value: 'demand'.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    weight : string
        Edges of the graph G are expected to have an attribute weight
        that indicates the cost incurred by sending one unit of flow on
        that edge. If not present, the weight is considered to be 0.
        Default value: 'weight'.

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
    flowCost : integer
        Cost of a minimum cost flow satisfying all demands.

    flowDict : dictionary
        If G is a digraph, a dict-of-dicts keyed by nodes such that
        flowDict[u][v] is the flow on edge (u, v).
        If G is a MultiDiGraph, a dict-of-dicts-of-dicts keyed by nodes
        so that flowDict[u][v][key] is the flow on edge (u, v, key).

    Raises
    ------
    NetworkXError
        This exception is raised if the input graph is not directed,
        not connected.

    NetworkXUnfeasible
        This exception is raised in the following situations:

            * The sum of the demands is not zero. Then, there is no
              flow satisfying all demands.
            * There is no flow satisfying all demand.

    NetworkXUnbounded
        This exception is raised if the digraph G has a cycle of
        negative cost and infinite capacity. Then, the cost of a flow
        satisfying all demands is unbounded below.

    Notes
    -----
    This algorithm does not work if edge weights are floating-point numbers.

    See also
    --------
    :meth:`network_simplex`

    Examples
    --------
    A simple example of a min cost flow problem.

    >>> G = nx.DiGraph()
    >>> G.add_node("a", demand=-5)
    >>> G.add_node("d", demand=5)
    >>> G.add_edge("a", "b", weight=3, capacity=4)
    >>> G.add_edge("a", "c", weight=6, capacity=10)
    >>> G.add_edge("b", "d", weight=1, capacity=9)
    >>> G.add_edge("c", "d", weight=2, capacity=5)
    >>> flowCost, flowDict = nx.capacity_scaling(G)
    >>> flowCost
    24
    >>> flowDict
    {'a': {'b': 4, 'c': 1}, 'd': {}, 'b': {'d': 4}, 'c': {'d': 1}}

    It is possible to change the name of the attributes used for the
    algorithm.

    >>> G = nx.DiGraph()
    >>> G.add_node("p", spam=-4)
    >>> G.add_node("q", spam=2)
    >>> G.add_node("a", spam=-2)
    >>> G.add_node("d", spam=-1)
    >>> G.add_node("t", spam=2)
    >>> G.add_node("w", spam=3)
    >>> G.add_edge("p", "q", cost=7, vacancies=5)
    >>> G.add_edge("p", "a", cost=1, vacancies=4)
    >>> G.add_edge("q", "d", cost=2, vacancies=3)
    >>> G.add_edge("t", "q", cost=1, vacancies=2)
    >>> G.add_edge("a", "t", cost=2, vacancies=4)
    >>> G.add_edge("d", "w", cost=3, vacancies=4)
    >>> G.add_edge("t", "w", cost=4, vacancies=1)
    >>> flowCost, flowDict = nx.capacity_scaling(
    ...     G, demand="spam", capacity="vacancies", weight="cost"
    ... )
    >>> flowCost
    37
    >>> flowDict
    {'p': {'q': 2, 'a': 2}, 'q': {'d': 1}, 'a': {'t': 4}, 'd': {'w': 2}, 't': {'q': 1, 'w': 1}, 'w': {}}
    """
    R = _build_residual_network(G, demand, capacity, weight)

    inf = float("inf")
    # Account cost of negative selfloops.
    flow_cost = sum(
        0
        if e.get(capacity, inf) <= 0 or e.get(weight, 0) >= 0
        else e[capacity] * e[weight]
        for u, v, e in nx.selfloop_edges(G, data=True)
    )

    # Determine the maximum edge capacity.
    wmax = max(chain([-inf], (e["capacity"] for u, v, e in R.edges(data=True))))
    if wmax == -inf:
        # Residual network has no edges.
        return flow_cost, _build_flow_dict(G, R, capacity, weight)

    R_nodes = R.nodes
    R_succ = R.succ

    delta = 2 ** int(log(wmax, 2))
    while delta >= 1:
        # Saturate Δ-residual edges with negative reduced costs to achieve
        # Δ-optimality.
        for u in R:
            p_u = R_nodes[u]["potential"]
            for v, es in R_succ[u].items():
                for k, e in es.items():
                    flow = e["capacity"] - e["flow"]
                    if e["weight"] - p_u + R_nodes[v]["potential"] < 0:
                        flow = e["capacity"] - e["flow"]
                        if flow >= delta:
                            e["flow"] += flow
                            R_succ[v][u][(k[0], not k[1])]["flow"] -= flow
                            R_nodes[u]["excess"] -= flow
                            R_nodes[v]["excess"] += flow
        # Determine the Δ-active nodes.
        S = set()
        T = set()
        S_add = S.add
        S_remove = S.remove
        T_add = T.add
        T_remove = T.remove
        for u in R:
            excess = R_nodes[u]["excess"]
            if excess >= delta:
                S_add(u)
            elif excess <= -delta:
                T_add(u)
        # Repeatedly augment flow from S to T along shortest paths until
        # Δ-feasibility is achieved.
        while S and T:
            s = arbitrary_element(S)
            t = None
            # Search for a shortest path in terms of reduce costs from s to
            # any t in T in the Δ-residual network.
            d = {}
            pred = {s: None}
            h = heap()
            h_insert = h.insert
            h_get = h.get
            h_insert(s, 0)
            while h:
                u, d_u = h.pop()
                d[u] = d_u
                if u in T:
                    # Path found.
                    t = u
                    break
                p_u = R_nodes[u]["potential"]
                for v, es in R_succ[u].items():
                    if v in d:
                        continue
                    wmin = inf
                    # Find the minimum-weighted (u, v) Δ-residual edge.
                    for k, e in es.items():
                        if e["capacity"] - e["flow"] >= delta:
                            w = e["weight"]
                            if w < wmin:
                                wmin = w
                                kmin = k
                                emin = e
                    if wmin == inf:
                        continue
                    # Update the distance label of v.
                    d_v = d_u + wmin - p_u + R_nodes[v]["potential"]
                    if h_insert(v, d_v):
                        pred[v] = (u, kmin, emin)
            if t is not None:
                # Augment Δ units of flow from s to t.
                while u != s:
                    v = u
                    u, k, e = pred[v]
                    e["flow"] += delta
                    R_succ[v][u][(k[0], not k[1])]["flow"] -= delta
                # Account node excess and deficit.
                R_nodes[s]["excess"] -= delta
                R_nodes[t]["excess"] += delta
                if R_nodes[s]["excess"] < delta:
                    S_remove(s)
                if R_nodes[t]["excess"] > -delta:
                    T_remove(t)
                # Update node potentials.
                d_t = d[t]
                for u, d_u in d.items():
                    R_nodes[u]["potential"] -= d_u - d_t
            else:
                # Path not found.
                S_remove(s)
        delta //= 2

    if any(R.nodes[u]["excess"] != 0 for u in R):
        raise nx.NetworkXUnfeasible("No flow satisfying all demands.")

    # Calculate the flow cost.
    for u in R:
        for v, es in R_succ[u].items():
            for e in es.values():
                flow = e["flow"]
                if flow > 0:
                    flow_cost += flow * e["weight"]

    return flow_cost, _build_flow_dict(G, R, capacity, weight)
<|endoftext|>"""
Dinitz' algorithm for maximum flow problems.
"""

from collections import deque

import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
from networkx.utils import pairwise

__all__ = ["dinitz"]


@nx._dispatchable(edge_attrs={"capacity": float("inf")}, returns_graph=True)
def dinitz(G, s, t, capacity="capacity", residual=None, value_only=False, cutoff=None):
    """Find a maximum single-commodity flow using Dinitz' algorithm.

    This function returns the residual network resulting after computing
    the maximum flow. See below for details about the conventions
    NetworkX uses for defining residual networks.

    This algorithm has a running time of $O(n^2 m)$ for $n$ nodes and $m$
    edges [1]_.


    Parameters
    ----------
    G : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    s : node
        Source node for the flow.

    t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    residual : NetworkX graph
        Residual network on which the algorithm is to be executed. If None, a
        new residual network is created. Default value: None.

    value_only : bool
        If True compute only the value of the maximum flow. This parameter
        will be ignored by this algorithm because it is not applicable.

    cutoff : integer, float
        If specified, the algorithm will terminate when the flow value reaches
        or exceeds the cutoff. In this case, it may be unable to immediately
        determine a minimum cut. Default value: None.

    Returns
    -------
    R : NetworkX DiGraph
        Residual network after computing the maximum flow.

    Raises
    ------
    NetworkXError
        The algorithm does not support MultiGraph and MultiDiGraph. If
        the input graph is an instance of one of these two classes, a
        NetworkXError is raised.

    NetworkXUnbounded
        If the graph has a path of infinite capacity, the value of a
        feasible flow on the graph is unbounded above and the function
        raises a NetworkXUnbounded.

    See also
    --------
    :meth:`maximum_flow`
    :meth:`minimum_cut`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Notes
    -----
    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. If :samp:`cutoff` is not
    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such
    that :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Examples
    --------
    >>> from networkx.algorithms.flow import dinitz

    The functions that implement flow algorithms and output a residual
    network, such as this one, are not imported to the base NetworkX
    namespace, so you have to explicitly import them from the flow package.

    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)
    >>> R = dinitz(G, "x", "y")
    >>> flow_value = nx.maximum_flow_value(G, "x", "y")
    >>> flow_value
    3.0
    >>> flow_value == R.graph["flow_value"]
    True

    References
    ----------
    .. [1] Dinitz' Algorithm: The Original Version and Even's Version.
           2006. Yefim Dinitz. In Theoretical Computer Science. Lecture
           Notes in Computer Science. Volume 3895. pp 218-240.
           https://doi.org/10.1007/11685654_10

    """
    R = dinitz_impl(G, s, t, capacity, residual, cutoff)
    R.graph["algorithm"] = "dinitz"
    nx._clear_cache(R)
    return R


def dinitz_impl(G, s, t, capacity, residual, cutoff):
    if s not in G:
        raise nx.NetworkXError(f"node {str(s)} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"node {str(t)} not in graph")
    if s == t:
        raise nx.NetworkXError("source and sink are the same node")

    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual

    # Initialize/reset the residual network.
    for u in R:
        for e in R[u].values():
            e["flow"] = 0

    # Use an arbitrary high value as infinite. It is computed
    # when building the residual network.
    INF = R.graph["inf"]

    if cutoff is None:
        cutoff = INF

    R_succ = R.succ
    R_pred = R.pred

    def breath_first_search():
        parents = {}
        vertex_dist = {s: 0}
        queue = deque([(s, 0)])
        # Record all the potential edges of shortest augmenting paths
        while queue:
            if t in parents:
                break
            u, dist = queue.popleft()
            for v, attr in R_succ[u].items():
                if attr["capacity"] - attr["flow"] > 0:
                    if v in parents:
                        if vertex_dist[v] == dist + 1:
                            parents[v].append(u)
                    else:
                        parents[v] = deque([u])
                        vertex_dist[v] = dist + 1
                        queue.append((v, dist + 1))
        return parents

    def depth_first_search(parents):
        # DFS to find all the shortest augmenting paths
        """Build a path using DFS starting from the sink"""
        total_flow = 0
        u = t
        # path also functions as a stack
        path = [u]
        # The loop ends with no augmenting path left in the layered graph
        while True:
            if len(parents[u]) > 0:
                v = parents[u][0]
                path.append(v)
            else:
                path.pop()
                if len(path) == 0:
                    break
                v = path[-1]
                parents[v].popleft()
            # Augment the flow along the path found
            if v == s:
                flow = INF
                for u, v in pairwise(path):
                    flow = min(flow, R_pred[u][v]["capacity"] - R_pred[u][v]["flow"])
                for u, v in pairwise(reversed(path)):
                    R_pred[v][u]["flow"] += flow
                    R_pred[u][v]["flow"] -= flow
                    # Find the proper node to continue the search
                    if R_pred[v][u]["capacity"] - R_pred[v][u]["flow"] == 0:
                        parents[v].popleft()
                        while path[-1] != v:
                            path.pop()
                total_flow += flow
                v = path[-1]
            u = v
        return total_flow

    flow_value = 0
    while flow_value < cutoff:
        parents = breath_first_search()
        if t not in parents:
            break
        this_flow = depth_first_search(parents)
        if this_flow * 2 > INF:
            raise nx.NetworkXUnbounded("Infinite capacity path, flow unbounded above.")
        flow_value += this_flow

    R.graph["flow_value"] = flow_value
    return R
<|endoftext|>from .maxflow import *
from .mincost import *
from .boykovkolmogorov import *
from .dinitz_alg import *
from .edmondskarp import *
from .gomory_hu import *
from .preflowpush import *
from .shortestaugmentingpath import *
from .capacityscaling import *
from .networksimplex import *
from .utils import build_flow_dict, build_residual_network
<|endoftext|>"""
Boykov-Kolmogorov algorithm for maximum flow problems.
"""

from collections import deque
from operator import itemgetter

import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network

__all__ = ["boykov_kolmogorov"]


@nx._dispatchable(edge_attrs={"capacity": float("inf")}, returns_graph=True)
def boykov_kolmogorov(
    G, s, t, capacity="capacity", residual=None, value_only=False, cutoff=None
):
    r"""Find a maximum single-commodity flow using Boykov-Kolmogorov algorithm.

    This function returns the residual network resulting after computing
    the maximum flow. See below for details about the conventions
    NetworkX uses for defining residual networks.

    This algorithm has worse case complexity $O(n^2 m |C|)$ for $n$ nodes, $m$
    edges, and $|C|$ the cost of the minimum cut [1]_. This implementation
    uses the marking heuristic defined in [2]_ which improves its running
    time in many practical problems.

    Parameters
    ----------
    G : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    s : node
        Source node for the flow.

    t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    residual : NetworkX graph
        Residual network on which the algorithm is to be executed. If None, a
        new residual network is created. Default value: None.

    value_only : bool
        If True compute only the value of the maximum flow. This parameter
        will be ignored by this algorithm because it is not applicable.

    cutoff : integer, float
        If specified, the algorithm will terminate when the flow value reaches
        or exceeds the cutoff. In this case, it may be unable to immediately
        determine a minimum cut. Default value: None.

    Returns
    -------
    R : NetworkX DiGraph
        Residual network after computing the maximum flow.

    Raises
    ------
    NetworkXError
        The algorithm does not support MultiGraph and MultiDiGraph. If
        the input graph is an instance of one of these two classes, a
        NetworkXError is raised.

    NetworkXUnbounded
        If the graph has a path of infinite capacity, the value of a
        feasible flow on the graph is unbounded above and the function
        raises a NetworkXUnbounded.

    See also
    --------
    :meth:`maximum_flow`
    :meth:`minimum_cut`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Notes
    -----
    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. If :samp:`cutoff` is not
    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such
    that :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Examples
    --------
    >>> from networkx.algorithms.flow import boykov_kolmogorov

    The functions that implement flow algorithms and output a residual
    network, such as this one, are not imported to the base NetworkX
    namespace, so you have to explicitly import them from the flow package.

    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)
    >>> R = boykov_kolmogorov(G, "x", "y")
    >>> flow_value = nx.maximum_flow_value(G, "x", "y")
    >>> flow_value
    3.0
    >>> flow_value == R.graph["flow_value"]
    True

    A nice feature of the Boykov-Kolmogorov algorithm is that a partition
    of the nodes that defines a minimum cut can be easily computed based
    on the search trees used during the algorithm. These trees are stored
    in the graph attribute `trees` of the residual network.

    >>> source_tree, target_tree = R.graph["trees"]
    >>> partition = (set(source_tree), set(G) - set(source_tree))

    Or equivalently:

    >>> partition = (set(G) - set(target_tree), set(target_tree))

    References
    ----------
    .. [1] Boykov, Y., & Kolmogorov, V. (2004). An experimental comparison
           of min-cut/max-flow algorithms for energy minimization in vision.
           Pattern Analysis and Machine Intelligence, IEEE Transactions on,
           26(9), 1124-1137.
           https://doi.org/10.1109/TPAMI.2004.60

    .. [2] Vladimir Kolmogorov. Graph-based Algorithms for Multi-camera
           Reconstruction Problem. PhD thesis, Cornell University, CS Department,
           2003. pp. 109-114.
           https://web.archive.org/web/20170809091249/https://pub.ist.ac.at/~vnk/papers/thesis.pdf

    """
    R = boykov_kolmogorov_impl(G, s, t, capacity, residual, cutoff)
    R.graph["algorithm"] = "boykov_kolmogorov"
    nx._clear_cache(R)
    return R


def boykov_kolmogorov_impl(G, s, t, capacity, residual, cutoff):
    if s not in G:
        raise nx.NetworkXError(f"node {str(s)} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"node {str(t)} not in graph")
    if s == t:
        raise nx.NetworkXError("source and sink are the same node")

    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual

    # Initialize/reset the residual network.
    # This is way too slow
    # nx.set_edge_attributes(R, 0, 'flow')
    for u in R:
        for e in R[u].values():
            e["flow"] = 0

    # Use an arbitrary high value as infinite. It is computed
    # when building the residual network.
    INF = R.graph["inf"]

    if cutoff is None:
        cutoff = INF

    R_succ = R.succ
    R_pred = R.pred

    def grow():
        """Bidirectional breadth-first search for the growth stage.

        Returns a connecting edge, that is and edge that connects
        a node from the source search tree with a node from the
        target search tree.
        The first node in the connecting edge is always from the
        source tree and the last node from the target tree.
        """
        while active:
            u = active[0]
            if u in source_tree:
                this_tree = source_tree
                other_tree = target_tree
                neighbors = R_succ
            else:
                this_tree = target_tree
                other_tree = source_tree
                neighbors = R_pred
            for v, attr in neighbors[u].items():
                if attr["capacity"] - attr["flow"] > 0:
                    if v not in this_tree:
                        if v in other_tree:
                            return (u, v) if this_tree is source_tree else (v, u)
                        this_tree[v] = u
                        dist[v] = dist[u] + 1
                        timestamp[v] = timestamp[u]
                        active.append(v)
                    elif v in this_tree and _is_closer(u, v):
                        this_tree[v] = u
                        dist[v] = dist[u] + 1
                        timestamp[v] = timestamp[u]
            _ = active.popleft()
        return None, None

    def augment(u, v):
        """Augmentation stage.

        Reconstruct path and determine its residual capacity.
        We start from a connecting edge, which links a node
        from the source tree to a node from the target tree.
        The connecting edge is the output of the grow function
        and the input of this function.
        """
        attr = R_succ[u][v]
        flow = min(INF, attr["capacity"] - attr["flow"])
        path = [u]
        # Trace a path from u to s in source_tree.
        w = u
        while w != s:
            n = w
            w = source_tree[n]
            attr = R_pred[n][w]
            flow = min(flow, attr["capacity"] - attr["flow"])
            path.append(w)
        path.reverse()
        # Trace a path from v to t in target_tree.
        path.append(v)
        w = v
        while w != t:
            n = w
            w = target_tree[n]
            attr = R_succ[n][w]
            flow = min(flow, attr["capacity"] - attr["flow"])
            path.append(w)
        # Augment flow along the path and check for saturated edges.
        it = iter(path)
        u = next(it)
        these_orphans = []
        for v in it:
            R_succ[u][v]["flow"] += flow
            R_succ[v][u]["flow"] -= flow
            if R_succ[u][v]["flow"] == R_succ[u][v]["capacity"]:
                if v in source_tree:
                    source_tree[v] = None
                    these_orphans.append(v)
                if u in target_tree:
                    target_tree[u] = None
                    these_orphans.append(u)
            u = v
        orphans.extend(sorted(these_orphans, key=dist.get))
        return flow

    def adopt():
        """Adoption stage.

        Reconstruct search trees by adopting or discarding orphans.
        During augmentation stage some edges got saturated and thus
        the source and target search trees broke down to forests, with
        orphans as roots of some of its trees. We have to reconstruct
        the search trees rooted to source and target before we can grow
        them again.
        """
        while orphans:
            u = orphans.popleft()
            if u in source_tree:
                tree = source_tree
                neighbors = R_pred
            else:
                tree = target_tree
                neighbors = R_succ
            nbrs = ((n, attr, dist[n]) for n, attr in neighbors[u].items() if n in tree)
            for v, attr, d in sorted(nbrs, key=itemgetter(2)):
                if attr["capacity"] - attr["flow"] > 0:
                    if _has_valid_root(v, tree):
                        tree[u] = v
                        dist[u] = dist[v] + 1
                        timestamp[u] = time
                        break
            else:
                nbrs = (
                    (n, attr, dist[n]) for n, attr in neighbors[u].items() if n in tree
                )
                for v, attr, d in sorted(nbrs, key=itemgetter(2)):
                    if attr["capacity"] - attr["flow"] > 0:
                        if v not in active:
                            active.append(v)
                    if tree[v] == u:
                        tree[v] = None
                        orphans.appendleft(v)
                if u in active:
                    active.remove(u)
                del tree[u]

    def _has_valid_root(n, tree):
        path = []
        v = n
        while v is not None:
            path.append(v)
            if v in (s, t):
                base_dist = 0
                break
            elif timestamp[v] == time:
                base_dist = dist[v]
                break
            v = tree[v]
        else:
            return False
        length = len(path)
        for i, u in enumerate(path, 1):
            dist[u] = base_dist + length - i
            timestamp[u] = time
        return True

    def _is_closer(u, v):
        return timestamp[v] <= timestamp[u] and dist[v] > dist[u] + 1

    source_tree = {s: None}
    target_tree = {t: None}
    active = deque([s, t])
    orphans = deque()
    flow_value = 0
    # data structures for the marking heuristic
    time = 1
    timestamp = {s: time, t: time}
    dist = {s: 0, t: 0}
    while flow_value < cutoff:
        # Growth stage
        u, v = grow()
        if u is None:
            break
        time += 1
        # Augmentation stage
        flow_value += augment(u, v)
        # Adoption stage
        adopt()

    if flow_value * 2 > INF:
        raise nx.NetworkXUnbounded("Infinite capacity path, flow unbounded above.")

    # Add source and target tree in a graph attribute.
    # A partition that defines a minimum cut can be directly
    # computed from the search trees as explained in the docstrings.
    R.graph["trees"] = (source_tree, target_tree)
    # Add the standard flow_value graph attribute.
    R.graph["flow_value"] = flow_value
    return R
<|endoftext|>"""
Shortest augmenting path algorithm for maximum flow problems.
"""

from collections import deque

import networkx as nx

from .edmondskarp import edmonds_karp_core
from .utils import CurrentEdge, build_residual_network

__all__ = ["shortest_augmenting_path"]


def shortest_augmenting_path_impl(G, s, t, capacity, residual, two_phase, cutoff):
    """Implementation of the shortest augmenting path algorithm."""
    if s not in G:
        raise nx.NetworkXError(f"node {str(s)} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"node {str(t)} not in graph")
    if s == t:
        raise nx.NetworkXError("source and sink are the same node")

    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual

    R_nodes = R.nodes
    R_pred = R.pred
    R_succ = R.succ

    # Initialize/reset the residual network.
    for u in R:
        for e in R_succ[u].values():
            e["flow"] = 0

    # Initialize heights of the nodes.
    heights = {t: 0}
    q = deque([(t, 0)])
    while q:
        u, height = q.popleft()
        height += 1
        for v, attr in R_pred[u].items():
            if v not in heights and attr["flow"] < attr["capacity"]:
                heights[v] = height
                q.append((v, height))

    if s not in heights:
        # t is not reachable from s in the residual network. The maximum flow
        # must be zero.
        R.graph["flow_value"] = 0
        return R

    n = len(G)
    m = R.size() / 2

    # Initialize heights and 'current edge' data structures of the nodes.
    for u in R:
        R_nodes[u]["height"] = heights[u] if u in heights else n
        R_nodes[u]["curr_edge"] = CurrentEdge(R_succ[u])

    # Initialize counts of nodes in each level.
    counts = [0] * (2 * n - 1)
    for u in R:
        counts[R_nodes[u]["height"]] += 1

    inf = R.graph["inf"]

    def augment(path):
        """Augment flow along a path from s to t."""
        # Determine the path residual capacity.
        flow = inf
        it = iter(path)
        u = next(it)
        for v in it:
            attr = R_succ[u][v]
            flow = min(flow, attr["capacity"] - attr["flow"])
            u = v
        if flow * 2 > inf:
            raise nx.NetworkXUnbounded("Infinite capacity path, flow unbounded above.")
        # Augment flow along the path.
        it = iter(path)
        u = next(it)
        for v in it:
            R_succ[u][v]["flow"] += flow
            R_succ[v][u]["flow"] -= flow
            u = v
        return flow

    def relabel(u):
        """Relabel a node to create an admissible edge."""
        height = n - 1
        for v, attr in R_succ[u].items():
            if attr["flow"] < attr["capacity"]:
                height = min(height, R_nodes[v]["height"])
        return height + 1

    if cutoff is None:
        cutoff = float("inf")

    # Phase 1: Look for shortest augmenting paths using depth-first search.

    flow_value = 0
    path = [s]
    u = s
    d = n if not two_phase else int(min(m**0.5, 2 * n ** (2.0 / 3)))
    done = R_nodes[s]["height"] >= d
    while not done:
        height = R_nodes[u]["height"]
        curr_edge = R_nodes[u]["curr_edge"]
        # Depth-first search for the next node on the path to t.
        while True:
            v, attr = curr_edge.get()
            if height == R_nodes[v]["height"] + 1 and attr["flow"] < attr["capacity"]:
                # Advance to the next node following an admissible edge.
                path.append(v)
                u = v
                break
            try:
                curr_edge.move_to_next()
            except StopIteration:
                counts[height] -= 1
                if counts[height] == 0:
                    # Gap heuristic: If relabeling causes a level to become
                    # empty, a minimum cut has been identified. The algorithm
                    # can now be terminated.
                    R.graph["flow_value"] = flow_value
                    return R
                height = relabel(u)
                if u == s and height >= d:
                    if not two_phase:
                        # t is disconnected from s in the residual network. No
                        # more augmenting paths exist.
                        R.graph["flow_value"] = flow_value
                        return R
                    else:
                        # t is at least d steps away from s. End of phase 1.
                        done = True
                        break
                counts[height] += 1
                R_nodes[u]["height"] = height
                if u != s:
                    # After relabeling, the last edge on the path is no longer
                    # admissible. Retreat one step to look for an alternative.
                    path.pop()
                    u = path[-1]
                    break
        if u == t:
            # t is reached. Augment flow along the path and reset it for a new
            # depth-first search.
            flow_value += augment(path)
            if flow_value >= cutoff:
                R.graph["flow_value"] = flow_value
                return R
            path = [s]
            u = s

    # Phase 2: Look for shortest augmenting paths using breadth-first search.
    flow_value += edmonds_karp_core(R, s, t, cutoff - flow_value)

    R.graph["flow_value"] = flow_value
    return R


@nx._dispatchable(edge_attrs={"capacity": float("inf")}, returns_graph=True)
def shortest_augmenting_path(
    G,
    s,
    t,
    capacity="capacity",
    residual=None,
    value_only=False,
    two_phase=False,
    cutoff=None,
):
    r"""Find a maximum single-commodity flow using the shortest augmenting path
    algorithm.

    This function returns the residual network resulting after computing
    the maximum flow. See below for details about the conventions
    NetworkX uses for defining residual networks.

    This algorithm has a running time of $O(n^2 m)$ for $n$ nodes and $m$
    edges.


    Parameters
    ----------
    G : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    s : node
        Source node for the flow.

    t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    residual : NetworkX graph
        Residual network on which the algorithm is to be executed. If None, a
        new residual network is created. Default value: None.

    value_only : bool
        If True compute only the value of the maximum flow. This parameter
        will be ignored by this algorithm because it is not applicable.

    two_phase : bool
        If True, a two-phase variant is used. The two-phase variant improves
        the running time on unit-capacity networks from $O(nm)$ to
        $O(\min(n^{2/3}, m^{1/2}) m)$. Default value: False.

    cutoff : integer, float
        If specified, the algorithm will terminate when the flow value reaches
        or exceeds the cutoff. In this case, it may be unable to immediately
        determine a minimum cut. Default value: None.

    Returns
    -------
    R : NetworkX DiGraph
        Residual network after computing the maximum flow.

    Raises
    ------
    NetworkXError
        The algorithm does not support MultiGraph and MultiDiGraph. If
        the input graph is an instance of one of these two classes, a
        NetworkXError is raised.

    NetworkXUnbounded
        If the graph has a path of infinite capacity, the value of a
        feasible flow on the graph is unbounded above and the function
        raises a NetworkXUnbounded.

    See also
    --------
    :meth:`maximum_flow`
    :meth:`minimum_cut`
    :meth:`edmonds_karp`
    :meth:`preflow_push`

    Notes
    -----
    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. If :samp:`cutoff` is not
    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such
    that :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Examples
    --------
    >>> from networkx.algorithms.flow import shortest_augmenting_path

    The functions that implement flow algorithms and output a residual
    network, such as this one, are not imported to the base NetworkX
    namespace, so you have to explicitly import them from the flow package.

    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)
    >>> R = shortest_augmenting_path(G, "x", "y")
    >>> flow_value = nx.maximum_flow_value(G, "x", "y")
    >>> flow_value
    3.0
    >>> flow_value == R.graph["flow_value"]
    True

    """
    R = shortest_augmenting_path_impl(G, s, t, capacity, residual, two_phase, cutoff)
    R.graph["algorithm"] = "shortest_augmenting_path"
    nx._clear_cache(R)
    return R
<|endoftext|>"""
Minimum cost flow algorithms on directed connected graphs.
"""

__all__ = ["network_simplex"]

from itertools import chain, islice, repeat
from math import ceil, sqrt

import networkx as nx
from networkx.utils import not_implemented_for


class _DataEssentialsAndFunctions:
    def __init__(
        self, G, multigraph, demand="demand", capacity="capacity", weight="weight"
    ):
        # Number all nodes and edges and hereafter reference them using ONLY their numbers
        self.node_list = list(G)  # nodes
        self.node_indices = {u: i for i, u in enumerate(self.node_list)}  # node indices
        self.node_demands = [
            G.nodes[u].get(demand, 0) for u in self.node_list
        ]  # node demands

        self.edge_sources = []  # edge sources
        self.edge_targets = []  # edge targets
        if multigraph:
            self.edge_keys = []  # edge keys
        self.edge_indices = {}  # edge indices
        self.edge_capacities = []  # edge capacities
        self.edge_weights = []  # edge weights

        if not multigraph:
            edges = G.edges(data=True)
        else:
            edges = G.edges(data=True, keys=True)

        inf = float("inf")
        edges = (e for e in edges if e[0] != e[1] and e[-1].get(capacity, inf) != 0)
        for i, e in enumerate(edges):
            self.edge_sources.append(self.node_indices[e[0]])
            self.edge_targets.append(self.node_indices[e[1]])
            if multigraph:
                self.edge_keys.append(e[2])
            self.edge_indices[e[:-1]] = i
            self.edge_capacities.append(e[-1].get(capacity, inf))
            self.edge_weights.append(e[-1].get(weight, 0))

        # spanning tree specific data to be initialized

        self.edge_count = None  # number of edges
        self.edge_flow = None  # edge flows
        self.node_potentials = None  # node potentials
        self.parent = None  # parent nodes
        self.parent_edge = None  # edges to parents
        self.subtree_size = None  # subtree sizes
        self.next_node_dft = None  # next nodes in depth-first thread
        self.prev_node_dft = None  # previous nodes in depth-first thread
        self.last_descendent_dft = None  # last descendants in depth-first thread
        self._spanning_tree_initialized = (
            False  # False until initialize_spanning_tree() is called
        )

    def initialize_spanning_tree(self, n, faux_inf):
        self.edge_count = len(self.edge_indices)  # number of edges
        self.edge_flow = list(
            chain(repeat(0, self.edge_count), (abs(d) for d in self.node_demands))
        )  # edge flows
        self.node_potentials = [
            faux_inf if d <= 0 else -faux_inf for d in self.node_demands
        ]  # node potentials
        self.parent = list(chain(repeat(-1, n), [None]))  # parent nodes
        self.parent_edge = list(
            range(self.edge_count, self.edge_count + n)
        )  # edges to parents
        self.subtree_size = list(chain(repeat(1, n), [n + 1]))  # subtree sizes
        self.next_node_dft = list(
            chain(range(1, n), [-1, 0])
        )  # next nodes in depth-first thread
        self.prev_node_dft = list(range(-1, n))  # previous nodes in depth-first thread
        self.last_descendent_dft = list(
            chain(range(n), [n - 1])
        )  # last descendants in depth-first thread
        self._spanning_tree_initialized = True  # True only if all the assignments pass

    def find_apex(self, p, q):
        """
        Find the lowest common ancestor of nodes p and q in the spanning tree.
        """
        size_p = self.subtree_size[p]
        size_q = self.subtree_size[q]
        while True:
            while size_p < size_q:
                p = self.parent[p]
                size_p = self.subtree_size[p]
            while size_p > size_q:
                q = self.parent[q]
                size_q = self.subtree_size[q]
            if size_p == size_q:
                if p != q:
                    p = self.parent[p]
                    size_p = self.subtree_size[p]
                    q = self.parent[q]
                    size_q = self.subtree_size[q]
                else:
                    return p

    def trace_path(self, p, w):
        """
        Returns the nodes and edges on the path from node p to its ancestor w.
        """
        Wn = [p]
        We = []
        while p != w:
            We.append(self.parent_edge[p])
            p = self.parent[p]
            Wn.append(p)
        return Wn, We

    def find_cycle(self, i, p, q):
        """
        Returns the nodes and edges on the cycle containing edge i == (p, q)
        when the latter is added to the spanning tree.

        The cycle is oriented in the direction from p to q.
        """
        w = self.find_apex(p, q)
        Wn, We = self.trace_path(p, w)
        Wn.reverse()
        We.reverse()
        if We != [i]:
            We.append(i)
        WnR, WeR = self.trace_path(q, w)
        del WnR[-1]
        Wn += WnR
        We += WeR
        return Wn, We

    def augment_flow(self, Wn, We, f):
        """
        Augment f units of flow along a cycle represented by Wn and We.
        """
        for i, p in zip(We, Wn):
            if self.edge_sources[i] == p:
                self.edge_flow[i] += f
            else:
                self.edge_flow[i] -= f

    def trace_subtree(self, p):
        """
        Yield the nodes in the subtree rooted at a node p.
        """
        yield p
        l = self.last_descendent_dft[p]
        while p != l:
            p = self.next_node_dft[p]
            yield p

    def remove_edge(self, s, t):
        """
        Remove an edge (s, t) where parent[t] == s from the spanning tree.
        """
        size_t = self.subtree_size[t]
        prev_t = self.prev_node_dft[t]
        last_t = self.last_descendent_dft[t]
        next_last_t = self.next_node_dft[last_t]
        # Remove (s, t).
        self.parent[t] = None
        self.parent_edge[t] = None
        # Remove the subtree rooted at t from the depth-first thread.
        self.next_node_dft[prev_t] = next_last_t
        self.prev_node_dft[next_last_t] = prev_t
        self.next_node_dft[last_t] = t
        self.prev_node_dft[t] = last_t
        # Update the subtree sizes and last descendants of the (old) ancestors
        # of t.
        while s is not None:
            self.subtree_size[s] -= size_t
            if self.last_descendent_dft[s] == last_t:
                self.last_descendent_dft[s] = prev_t
            s = self.parent[s]

    def make_root(self, q):
        """
        Make a node q the root of its containing subtree.
        """
        ancestors = []
        while q is not None:
            ancestors.append(q)
            q = self.parent[q]
        ancestors.reverse()
        for p, q in zip(ancestors, islice(ancestors, 1, None)):
            size_p = self.subtree_size[p]
            last_p = self.last_descendent_dft[p]
            prev_q = self.prev_node_dft[q]
            last_q = self.last_descendent_dft[q]
            next_last_q = self.next_node_dft[last_q]
            # Make p a child of q.
            self.parent[p] = q
            self.parent[q] = None
            self.parent_edge[p] = self.parent_edge[q]
            self.parent_edge[q] = None
            self.subtree_size[p] = size_p - self.subtree_size[q]
            self.subtree_size[q] = size_p
            # Remove the subtree rooted at q from the depth-first thread.
            self.next_node_dft[prev_q] = next_last_q
            self.prev_node_dft[next_last_q] = prev_q
            self.next_node_dft[last_q] = q
            self.prev_node_dft[q] = last_q
            if last_p == last_q:
                self.last_descendent_dft[p] = prev_q
                last_p = prev_q
            # Add the remaining parts of the subtree rooted at p as a subtree
            # of q in the depth-first thread.
            self.prev_node_dft[p] = last_q
            self.next_node_dft[last_q] = p
            self.next_node_dft[last_p] = q
            self.prev_node_dft[q] = last_p
            self.last_descendent_dft[q] = last_p

    def add_edge(self, i, p, q):
        """
        Add an edge (p, q) to the spanning tree where q is the root of a subtree.
        """
        last_p = self.last_descendent_dft[p]
        next_last_p = self.next_node_dft[last_p]
        size_q = self.subtree_size[q]
        last_q = self.last_descendent_dft[q]
        # Make q a child of p.
        self.parent[q] = p
        self.parent_edge[q] = i
        # Insert the subtree rooted at q into the depth-first thread.
        self.next_node_dft[last_p] = q
        self.prev_node_dft[q] = last_p
        self.prev_node_dft[next_last_p] = last_q
        self.next_node_dft[last_q] = next_last_p
        # Update the subtree sizes and last descendants of the (new) ancestors
        # of q.
        while p is not None:
            self.subtree_size[p] += size_q
            if self.last_descendent_dft[p] == last_p:
                self.last_descendent_dft[p] = last_q
            p = self.parent[p]

    def update_potentials(self, i, p, q):
        """
        Update the potentials of the nodes in the subtree rooted at a node
        q connected to its parent p by an edge i.
        """
        if q == self.edge_targets[i]:
            d = self.node_potentials[p] - self.edge_weights[i] - self.node_potentials[q]
        else:
            d = self.node_potentials[p] + self.edge_weights[i] - self.node_potentials[q]
        for q in self.trace_subtree(q):
            self.node_potentials[q] += d

    def reduced_cost(self, i):
        """Returns the reduced cost of an edge i."""
        c = (
            self.edge_weights[i]
            - self.node_potentials[self.edge_sources[i]]
            + self.node_potentials[self.edge_targets[i]]
        )
        return c if self.edge_flow[i] == 0 else -c

    def find_entering_edges(self):
        """Yield entering edges until none can be found."""
        if self.edge_count == 0:
            return

        # Entering edges are found by combining Dantzig's rule and Bland's
        # rule. The edges are cyclically grouped into blocks of size B. Within
        # each block, Dantzig's rule is applied to find an entering edge. The
        # blocks to search is determined following Bland's rule.
        B = int(ceil(sqrt(self.edge_count)))  # pivot block size
        M = (self.edge_count + B - 1) // B  # number of blocks needed to cover all edges
        m = 0  # number of consecutive blocks without eligible
        # entering edges
        f = 0  # first edge in block
        while m < M:
            # Determine the next block of edges.
            l = f + B
            if l <= self.edge_count:
                edges = range(f, l)
            else:
                l -= self.edge_count
                edges = chain(range(f, self.edge_count), range(l))
            f = l
            # Find the first edge with the lowest reduced cost.
            i = min(edges, key=self.reduced_cost)
            c = self.reduced_cost(i)
            if c >= 0:
                # No entering edge found in the current block.
                m += 1
            else:
                # Entering edge found.
                if self.edge_flow[i] == 0:
                    p = self.edge_sources[i]
                    q = self.edge_targets[i]
                else:
                    p = self.edge_targets[i]
                    q = self.edge_sources[i]
                yield i, p, q
                m = 0
        # All edges have nonnegative reduced costs. The current flow is
        # optimal.

    def residual_capacity(self, i, p):
        """Returns the residual capacity of an edge i in the direction away
        from its endpoint p.
        """
        return (
            self.edge_capacities[i] - self.edge_flow[i]
            if self.edge_sources[i] == p
            else self.edge_flow[i]
        )

    def find_leaving_edge(self, Wn, We):
        """Returns the leaving edge in a cycle represented by Wn and We."""
        j, s = min(
            zip(reversed(We), reversed(Wn)),
            key=lambda i_p: self.residual_capacity(*i_p),
        )
        t = self.edge_targets[j] if self.edge_sources[j] == s else self.edge_sources[j]
        return j, s, t


@not_implemented_for("undirected")
@nx._dispatchable(
    node_attrs="demand", edge_attrs={"capacity": float("inf"), "weight": 0}
)
def network_simplex(G, demand="demand", capacity="capacity", weight="weight"):
    r"""Find a minimum cost flow satisfying all demands in digraph G.

    This is a primal network simplex algorithm that uses the leaving
    arc rule to prevent cycling.

    G is a digraph with edge costs and capacities and in which nodes
    have demand, i.e., they want to send or receive some amount of
    flow. A negative demand means that the node wants to send flow, a
    positive demand means that the node want to receive flow. A flow on
    the digraph G satisfies all demand if the net flow into each node
    is equal to the demand of that node.

    Parameters
    ----------
    G : NetworkX graph
        DiGraph on which a minimum cost flow satisfying all demands is
        to be found.

    demand : string
        Nodes of the graph G are expected to have an attribute demand
        that indicates how much flow a node wants to send (negative
        demand) or receive (positive demand). Note that the sum of the
        demands should be 0 otherwise the problem in not feasible. If
        this attribute is not present, a node is considered to have 0
        demand. Default value: 'demand'.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    weight : string
        Edges of the graph G are expected to have an attribute weight
        that indicates the cost incurred by sending one unit of flow on
        that edge. If not present, the weight is considered to be 0.
        Default value: 'weight'.

    Returns
    -------
    flowCost : integer, float
        Cost of a minimum cost flow satisfying all demands.

    flowDict : dictionary
        Dictionary of dictionaries keyed by nodes such that
        flowDict[u][v] is the flow edge (u, v).

    Raises
    ------
    NetworkXError
        This exception is raised if the input graph is not directed or
        not connected.

    NetworkXUnfeasible
        This exception is raised in the following situations:

            * The sum of the demands is not zero. Then, there is no
              flow satisfying all demands.
            * There is no flow satisfying all demand.

    NetworkXUnbounded
        This exception is raised if the digraph G has a cycle of
        negative cost and infinite capacity. Then, the cost of a flow
        satisfying all demands is unbounded below.

    Notes
    -----
    This algorithm is not guaranteed to work if edge weights or demands
    are floating point numbers (overflows and roundoff errors can
    cause problems). As a workaround you can use integer numbers by
    multiplying the relevant edge attributes by a convenient
    constant factor (eg 100).

    See also
    --------
    cost_of_flow, max_flow_min_cost, min_cost_flow, min_cost_flow_cost

    Examples
    --------
    A simple example of a min cost flow problem.

    >>> G = nx.DiGraph()
    >>> G.add_node("a", demand=-5)
    >>> G.add_node("d", demand=5)
    >>> G.add_edge("a", "b", weight=3, capacity=4)
    >>> G.add_edge("a", "c", weight=6, capacity=10)
    >>> G.add_edge("b", "d", weight=1, capacity=9)
    >>> G.add_edge("c", "d", weight=2, capacity=5)
    >>> flowCost, flowDict = nx.network_simplex(G)
    >>> flowCost
    24
    >>> flowDict
    {'a': {'b': 4, 'c': 1}, 'd': {}, 'b': {'d': 4}, 'c': {'d': 1}}

    The mincost flow algorithm can also be used to solve shortest path
    problems. To find the shortest path between two nodes u and v,
    give all edges an infinite capacity, give node u a demand of -1 and
    node v a demand a 1. Then run the network simplex. The value of a
    min cost flow will be the distance between u and v and edges
    carrying positive flow will indicate the path.

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
    >>> G.add_node("s", demand=-1)
    >>> G.add_node("v", demand=1)
    >>> flowCost, flowDict = nx.network_simplex(G)
    >>> flowCost == nx.shortest_path_length(G, "s", "v", weight="weight")
    True
    >>> sorted([(u, v) for u in flowDict for v in flowDict[u] if flowDict[u][v] > 0])
    [('s', 'x'), ('u', 'v'), ('x', 'u')]
    >>> nx.shortest_path(G, "s", "v", weight="weight")
    ['s', 'x', 'u', 'v']

    It is possible to change the name of the attributes used for the
    algorithm.

    >>> G = nx.DiGraph()
    >>> G.add_node("p", spam=-4)
    >>> G.add_node("q", spam=2)
    >>> G.add_node("a", spam=-2)
    >>> G.add_node("d", spam=-1)
    >>> G.add_node("t", spam=2)
    >>> G.add_node("w", spam=3)
    >>> G.add_edge("p", "q", cost=7, vacancies=5)
    >>> G.add_edge("p", "a", cost=1, vacancies=4)
    >>> G.add_edge("q", "d", cost=2, vacancies=3)
    >>> G.add_edge("t", "q", cost=1, vacancies=2)
    >>> G.add_edge("a", "t", cost=2, vacancies=4)
    >>> G.add_edge("d", "w", cost=3, vacancies=4)
    >>> G.add_edge("t", "w", cost=4, vacancies=1)
    >>> flowCost, flowDict = nx.network_simplex(
    ...     G, demand="spam", capacity="vacancies", weight="cost"
    ... )
    >>> flowCost
    37
    >>> flowDict
    {'p': {'q': 2, 'a': 2}, 'q': {'d': 1}, 'a': {'t': 4}, 'd': {'w': 2}, 't': {'q': 1, 'w': 1}, 'w': {}}

    References
    ----------
    .. [1] Z. Kiraly, P. Kovacs.
           Efficient implementation of minimum-cost flow algorithms.
           Acta Universitatis Sapientiae, Informatica 4(1):67--118. 2012.
    .. [2] R. Barr, F. Glover, D. Klingman.
           Enhancement of spanning tree labeling procedures for network
           optimization.
           INFOR 17(1):16--34. 1979.
    """
    ###########################################################################
    # Problem essentials extraction and sanity check
    ###########################################################################

    if len(G) == 0:
        raise nx.NetworkXError("graph has no nodes")

    multigraph = G.is_multigraph()

    # extracting data essential to problem
    DEAF = _DataEssentialsAndFunctions(
        G, multigraph, demand=demand, capacity=capacity, weight=weight
    )

    ###########################################################################
    # Quick Error Detection
    ###########################################################################

    inf = float("inf")
    for u, d in zip(DEAF.node_list, DEAF.node_demands):
        if abs(d) == inf:
            raise nx.NetworkXError(f"node {u!r} has infinite demand")
    for e, w in zip(DEAF.edge_indices, DEAF.edge_weights):
        if abs(w) == inf:
            raise nx.NetworkXError(f"edge {e!r} has infinite weight")
    if not multigraph:
        edges = nx.selfloop_edges(G, data=True)
    else:
        edges = nx.selfloop_edges(G, data=True, keys=True)
    for e in edges:
        if abs(e[-1].get(weight, 0)) == inf:
            raise nx.NetworkXError(f"edge {e[:-1]!r} has infinite weight")

    ###########################################################################
    # Quick Infeasibility Detection
    ###########################################################################

    if sum(DEAF.node_demands) != 0:
        raise nx.NetworkXUnfeasible("total node demand is not zero")
    for e, c in zip(DEAF.edge_indices, DEAF.edge_capacities):
        if c < 0:
            raise nx.NetworkXUnfeasible(f"edge {e!r} has negative capacity")
    if not multigraph:
        edges = nx.selfloop_edges(G, data=True)
    else:
        edges = nx.selfloop_edges(G, data=True, keys=True)
    for e in edges:
        if e[-1].get(capacity, inf) < 0:
            raise nx.NetworkXUnfeasible(f"edge {e[:-1]!r} has negative capacity")

    ###########################################################################
    # Initialization
    ###########################################################################

    # Add a dummy node -1 and connect all existing nodes to it with infinite-
    # capacity dummy edges. Node -1 will serve as the root of the
    # spanning tree of the network simplex method. The new edges will used to
    # trivially satisfy the node demands and create an initial strongly
    # feasible spanning tree.
    for i, d in enumerate(DEAF.node_demands):
        # Must be greater-than here. Zero-demand nodes must have
        # edges pointing towards the root to ensure strong feasibility.
        if d > 0:
            DEAF.edge_sources.append(-1)
            DEAF.edge_targets.append(i)
        else:
            DEAF.edge_sources.append(i)
            DEAF.edge_targets.append(-1)
    faux_inf = (
        3
        * max(
            chain(
                [
                    sum(c for c in DEAF.edge_capacities if c < inf),
                    sum(abs(w) for w in DEAF.edge_weights),
                ],
                (abs(d) for d in DEAF.node_demands),
            )
        )
        or 1
    )

    n = len(DEAF.node_list)  # number of nodes
    DEAF.edge_weights.extend(repeat(faux_inf, n))
    DEAF.edge_capacities.extend(repeat(faux_inf, n))

    # Construct the initial spanning tree.
    DEAF.initialize_spanning_tree(n, faux_inf)

    ###########################################################################
    # Pivot loop
    ###########################################################################

    for i, p, q in DEAF.find_entering_edges():
        Wn, We = DEAF.find_cycle(i, p, q)
        j, s, t = DEAF.find_leaving_edge(Wn, We)
        DEAF.augment_flow(Wn, We, DEAF.residual_capacity(j, s))
        # Do nothing more if the entering edge is the same as the leaving edge.
        if i != j:
            if DEAF.parent[t] != s:
                # Ensure that s is the parent of t.
                s, t = t, s
            if We.index(i) > We.index(j):
                # Ensure that q is in the subtree rooted at t.
                p, q = q, p
            DEAF.remove_edge(s, t)
            DEAF.make_root(q)
            DEAF.add_edge(i, p, q)
            DEAF.update_potentials(i, p, q)

    ###########################################################################
    # Infeasibility and unboundedness detection
    ###########################################################################

    if any(DEAF.edge_flow[i] != 0 for i in range(-n, 0)):
        raise nx.NetworkXUnfeasible("no flow satisfies all node demands")

    if any(DEAF.edge_flow[i] * 2 >= faux_inf for i in range(DEAF.edge_count)) or any(
        e[-1].get(capacity, inf) == inf and e[-1].get(weight, 0) < 0
        for e in nx.selfloop_edges(G, data=True)
    ):
        raise nx.NetworkXUnbounded("negative cycle with infinite capacity found")

    ###########################################################################
    # Flow cost calculation and flow dict construction
    ###########################################################################

    del DEAF.edge_flow[DEAF.edge_count :]
    flow_cost = sum(w * x for w, x in zip(DEAF.edge_weights, DEAF.edge_flow))
    flow_dict = {n: {} for n in DEAF.node_list}

    def add_entry(e):
        """Add a flow dict entry."""
        d = flow_dict[e[0]]
        for k in e[1:-2]:
            try:
                d = d[k]
            except KeyError:
                t = {}
                d[k] = t
                d = t
        d[e[-2]] = e[-1]

    DEAF.edge_sources = (
        DEAF.node_list[s] for s in DEAF.edge_sources
    )  # Use original nodes.
    DEAF.edge_targets = (
        DEAF.node_list[t] for t in DEAF.edge_targets
    )  # Use original nodes.
    if not multigraph:
        for e in zip(DEAF.edge_sources, DEAF.edge_targets, DEAF.edge_flow):
            add_entry(e)
        edges = G.edges(data=True)
    else:
        for e in zip(
            DEAF.edge_sources, DEAF.edge_targets, DEAF.edge_keys, DEAF.edge_flow
        ):
            add_entry(e)
        edges = G.edges(data=True, keys=True)
    for e in edges:
        if e[0] != e[1]:
            if e[-1].get(capacity, inf) == 0:
                add_entry(e[:-1] + (0,))
        else:
            w = e[-1].get(weight, 0)
            if w >= 0:
                add_entry(e[:-1] + (0,))
            else:
                c = e[-1][capacity]
                flow_cost += w * c
                add_entry(e[:-1] + (c,))

    return flow_cost, flow_dict
<|endoftext|>"""
Module to simplify the specification of user-defined equality functions for
node and edge attributes during isomorphism checks.

During the construction of an isomorphism, the algorithm considers two
candidate nodes n1 in G1 and n2 in G2.  The graphs G1 and G2 are then
compared with respect to properties involving n1 and n2, and if the outcome
is good, then the candidate nodes are considered isomorphic. NetworkX
provides a simple mechanism for users to extend the comparisons to include
node and edge attributes.

Node attributes are handled by the node_match keyword. When considering
n1 and n2, the algorithm passes their node attribute dictionaries to
node_match, and if it returns False, then n1 and n2 cannot be
considered to be isomorphic.

Edge attributes are handled by the edge_match keyword. When considering
n1 and n2, the algorithm must verify that outgoing edges from n1 are
commensurate with the outgoing edges for n2. If the graph is directed,
then a similar check is also performed for incoming edges.

Focusing only on outgoing edges, we consider pairs of nodes (n1, v1) from
G1 and (n2, v2) from G2. For graphs and digraphs, there is only one edge
between (n1, v1) and only one edge between (n2, v2). Those edge attribute
dictionaries are passed to edge_match, and if it returns False, then
n1 and n2 cannot be considered isomorphic. For multigraphs and
multidigraphs, there can be multiple edges between (n1, v1) and also
multiple edges between (n2, v2).  Now, there must exist an isomorphism
from "all the edges between (n1, v1)" to "all the edges between (n2, v2)".
So, all of the edge attribute dictionaries are passed to edge_match, and
it must determine if there is an isomorphism between the two sets of edges.
"""

from . import isomorphvf2 as vf2

__all__ = ["GraphMatcher", "DiGraphMatcher", "MultiGraphMatcher", "MultiDiGraphMatcher"]


def _semantic_feasibility(self, G1_node, G2_node):
    """Returns True if mapping G1_node to G2_node is semantically feasible."""
    # Make sure the nodes match
    if self.node_match is not None:
        nm = self.node_match(self.G1.nodes[G1_node], self.G2.nodes[G2_node])
        if not nm:
            return False

    # Make sure the edges match
    if self.edge_match is not None:
        # Cached lookups
        G1nbrs = self.G1_adj[G1_node]
        G2nbrs = self.G2_adj[G2_node]
        core_1 = self.core_1
        edge_match = self.edge_match

        for neighbor in G1nbrs:
            # G1_node is not in core_1, so we must handle R_self separately
            if neighbor == G1_node:
                if G2_node in G2nbrs and not edge_match(
                    G1nbrs[G1_node], G2nbrs[G2_node]
                ):
                    return False
            elif neighbor in core_1:
                G2_nbr = core_1[neighbor]
                if G2_nbr in G2nbrs and not edge_match(
                    G1nbrs[neighbor], G2nbrs[G2_nbr]
                ):
                    return False
        # syntactic check has already verified that neighbors are symmetric

    return True


class GraphMatcher(vf2.GraphMatcher):
    """VF2 isomorphism checker for undirected graphs."""

    def __init__(self, G1, G2, node_match=None, edge_match=None):
        """Initialize graph matcher.

        Parameters
        ----------
        G1, G2: graph
            The graphs to be tested.

        node_match: callable
            A function that returns True iff node n1 in G1 and n2 in G2
            should be considered equal during the isomorphism test. The
            function will be called like::

               node_match(G1.nodes[n1], G2.nodes[n2])

            That is, the function will receive the node attribute dictionaries
            of the nodes under consideration. If None, then no attributes are
            considered when testing for an isomorphism.

        edge_match: callable
            A function that returns True iff the edge attribute dictionary for
            the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should be
            considered equal during the isomorphism test. The function will be
            called like::

               edge_match(G1[u1][v1], G2[u2][v2])

            That is, the function will receive the edge attribute dictionaries
            of the edges under consideration. If None, then no attributes are
            considered when testing for an isomorphism.

        """
        vf2.GraphMatcher.__init__(self, G1, G2)

        self.node_match = node_match
        self.edge_match = edge_match

        # These will be modified during checks to minimize code repeat.
        self.G1_adj = self.G1.adj
        self.G2_adj = self.G2.adj

    semantic_feasibility = _semantic_feasibility


class DiGraphMatcher(vf2.DiGraphMatcher):
    """VF2 isomorphism checker for directed graphs."""

    def __init__(self, G1, G2, node_match=None, edge_match=None):
        """Initialize graph matcher.

        Parameters
        ----------
        G1, G2 : graph
            The graphs to be tested.

        node_match : callable
            A function that returns True iff node n1 in G1 and n2 in G2
            should be considered equal during the isomorphism test. The
            function will be called like::

               node_match(G1.nodes[n1], G2.nodes[n2])

            That is, the function will receive the node attribute dictionaries
            of the nodes under consideration. If None, then no attributes are
            considered when testing for an isomorphism.

        edge_match : callable
            A function that returns True iff the edge attribute dictionary for
            the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should be
            considered equal during the isomorphism test. The function will be
            called like::

               edge_match(G1[u1][v1], G2[u2][v2])

            That is, the function will receive the edge attribute dictionaries
            of the edges under consideration. If None, then no attributes are
            considered when testing for an isomorphism.

        """
        vf2.DiGraphMatcher.__init__(self, G1, G2)

        self.node_match = node_match
        self.edge_match = edge_match

        # These will be modified during checks to minimize code repeat.
        self.G1_adj = self.G1.adj
        self.G2_adj = self.G2.adj

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if mapping G1_node to G2_node is semantically feasible."""

        # Test node_match and also test edge_match on successors
        feasible = _semantic_feasibility(self, G1_node, G2_node)
        if not feasible:
            return False

        # Test edge_match on predecessors
        self.G1_adj = self.G1.pred
        self.G2_adj = self.G2.pred
        feasible = _semantic_feasibility(self, G1_node, G2_node)
        self.G1_adj = self.G1.adj
        self.G2_adj = self.G2.adj

        return feasible


# The "semantics" of edge_match are different for multi(di)graphs, but
# the implementation is the same.  So, technically we do not need to
# provide "multi" versions, but we do so to match NetworkX's base classes.


class MultiGraphMatcher(GraphMatcher):
    """VF2 isomorphism checker for undirected multigraphs."""


class MultiDiGraphMatcher(DiGraphMatcher):
    """VF2 isomorphism checker for directed multigraphs."""
<|endoftext|>"""
An algorithm for finding if two undirected trees are isomorphic,
and if so returns an isomorphism between the two sets of nodes.

This algorithm uses a routine to tell if two rooted trees (trees with a
specified root node) are isomorphic, which may be independently useful.

This implements an algorithm from:
The Design and Analysis of Computer Algorithms
by Aho, Hopcroft, and Ullman
Addison-Wesley Publishing 1974
Example 3.2 pp. 84-86.

A more understandable version of this algorithm is described in:
Homework Assignment 5
McGill University SOCS 308-250B, Winter 2002
by Matthew Suderman
http://crypto.cs.mcgill.ca/~crepeau/CS250/2004/HW5+.pdf
"""

import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = ["rooted_tree_isomorphism", "tree_isomorphism"]


@nx._dispatchable(graphs={"t1": 0, "t2": 2}, returns_graph=True)
def root_trees(t1, root1, t2, root2):
    """Create a single digraph dT of free trees t1 and t2
    #   with roots root1 and root2 respectively
    # rename the nodes with consecutive integers
    # so that all nodes get a unique name between both trees

    # our new "fake" root node is 0
    # t1 is numbers from 1 ... n
    # t2 is numbered from n+1 to 2n
    """

    dT = nx.DiGraph()

    newroot1 = 1  # left root will be 1
    newroot2 = nx.number_of_nodes(t1) + 1  # right will be n+1

    # may be overlap in node names here so need separate maps
    # given the old name, what is the new
    namemap1 = {root1: newroot1}
    namemap2 = {root2: newroot2}

    # add an edge from our new root to root1 and root2
    dT.add_edge(0, namemap1[root1])
    dT.add_edge(0, namemap2[root2])

    for i, (v1, v2) in enumerate(nx.bfs_edges(t1, root1)):
        namemap1[v2] = i + namemap1[root1] + 1
        dT.add_edge(namemap1[v1], namemap1[v2])

    for i, (v1, v2) in enumerate(nx.bfs_edges(t2, root2)):
        namemap2[v2] = i + namemap2[root2] + 1
        dT.add_edge(namemap2[v1], namemap2[v2])

    # now we really want the inverse of namemap1 and namemap2
    # giving the old name given the new
    # since the values of namemap1 and namemap2 are unique
    # there won't be collisions
    namemap = {}
    for old, new in namemap1.items():
        namemap[new] = old
    for old, new in namemap2.items():
        namemap[new] = old

    return (dT, namemap, newroot1, newroot2)


# figure out the level of each node, with 0 at root
@nx._dispatchable
def assign_levels(G, root):
    level = {}
    level[root] = 0
    for v1, v2 in nx.bfs_edges(G, root):
        level[v2] = level[v1] + 1

    return level


# now group the nodes at each level
def group_by_levels(levels):
    L = {}
    for n, lev in levels.items():
        if lev not in L:
            L[lev] = []
        L[lev].append(n)

    return L


# now lets get the isomorphism by walking the ordered_children
def generate_isomorphism(v, w, M, ordered_children):
    # make sure tree1 comes first
    assert v < w
    M.append((v, w))
    for i, (x, y) in enumerate(zip(ordered_children[v], ordered_children[w])):
        generate_isomorphism(x, y, M, ordered_children)


@nx._dispatchable(graphs={"t1": 0, "t2": 2})
def rooted_tree_isomorphism(t1, root1, t2, root2):
    """
    Given two rooted trees `t1` and `t2`,
    with roots `root1` and `root2` respectively
    this routine will determine if they are isomorphic.

    These trees may be either directed or undirected,
    but if they are directed, all edges should flow from the root.

    It returns the isomorphism, a mapping of the nodes of `t1` onto the nodes
    of `t2`, such that two trees are then identical.

    Note that two trees may have more than one isomorphism, and this
    routine just returns one valid mapping.

    Parameters
    ----------
    `t1` :  NetworkX graph
        One of the trees being compared

    `root1` : a node of `t1` which is the root of the tree

    `t2` : undirected NetworkX graph
        The other tree being compared

    `root2` : a node of `t2` which is the root of the tree

    This is a subroutine used to implement `tree_isomorphism`, but will
    be somewhat faster if you already have rooted trees.

    Returns
    -------
    isomorphism : list
        A list of pairs in which the left element is a node in `t1`
        and the right element is a node in `t2`.  The pairs are in
        arbitrary order.  If the nodes in one tree is mapped to the names in
        the other, then trees will be identical. Note that an isomorphism
        will not necessarily be unique.

        If `t1` and `t2` are not isomorphic, then it returns the empty list.
    """

    assert nx.is_tree(t1)
    assert nx.is_tree(t2)

    # get the rooted tree formed by combining them
    # with unique names
    (dT, namemap, newroot1, newroot2) = root_trees(t1, root1, t2, root2)

    # compute the distance from the root, with 0 for our
    levels = assign_levels(dT, 0)

    # height
    h = max(levels.values())

    # collect nodes into a dict by level
    L = group_by_levels(levels)

    # each node has a label, initially set to 0
    label = {v: 0 for v in dT}
    # and also ordered_labels and ordered_children
    # which will store ordered tuples
    ordered_labels = {v: () for v in dT}
    ordered_children = {v: () for v in dT}

    # nothing to do on last level so start on h-1
    # also nothing to do for our fake level 0, so skip that
    for i in range(h - 1, 0, -1):
        # update the ordered_labels and ordered_children
        # for any children
        for v in L[i]:
            # nothing to do if no children
            if dT.out_degree(v) > 0:
                # get all the pairs of labels and nodes of children
                # and sort by labels
                s = sorted((label[u], u) for u in dT.successors(v))

                # invert to give a list of two tuples
                # the sorted labels, and the corresponding children
                ordered_labels[v], ordered_children[v] = list(zip(*s))

        # now collect and sort the sorted ordered_labels
        # for all nodes in L[i], carrying along the node
        forlabel = sorted((ordered_labels[v], v) for v in L[i])

        # now assign labels to these nodes, according to the sorted order
        # starting from 0, where identical ordered_labels get the same label
        current = 0
        for i, (ol, v) in enumerate(forlabel):
            # advance to next label if not 0, and different from previous
            if (i != 0) and (ol != forlabel[i - 1][0]):
                current += 1
            label[v] = current

    # they are isomorphic if the labels of newroot1 and newroot2 are 0
    isomorphism = []
    if label[newroot1] == 0 and label[newroot2] == 0:
        generate_isomorphism(newroot1, newroot2, isomorphism, ordered_children)

        # get the mapping back in terms of the old names
        # return in sorted order for neatness
        isomorphism = [(namemap[u], namemap[v]) for (u, v) in isomorphism]

    return isomorphism


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(graphs={"t1": 0, "t2": 1})
def tree_isomorphism(t1, t2):
    """
    Given two undirected (or free) trees `t1` and `t2`,
    this routine will determine if they are isomorphic.
    It returns the isomorphism, a mapping of the nodes of `t1` onto the nodes
    of `t2`, such that two trees are then identical.

    Note that two trees may have more than one isomorphism, and this
    routine just returns one valid mapping.

    Parameters
    ----------
    t1 : undirected NetworkX graph
        One of the trees being compared

    t2 : undirected NetworkX graph
        The other tree being compared

    Returns
    -------
    isomorphism : list
        A list of pairs in which the left element is a node in `t1`
        and the right element is a node in `t2`.  The pairs are in
        arbitrary order.  If the nodes in one tree is mapped to the names in
        the other, then trees will be identical. Note that an isomorphism
        will not necessarily be unique.

        If `t1` and `t2` are not isomorphic, then it returns the empty list.

    Notes
    -----
    This runs in O(n*log(n)) time for trees with n nodes.
    """

    assert nx.is_tree(t1)
    assert nx.is_tree(t2)

    # To be isomorphic, t1 and t2 must have the same number of nodes.
    if nx.number_of_nodes(t1) != nx.number_of_nodes(t2):
        return []

    # Another shortcut is that the sorted degree sequences need to be the same.
    degree_sequence1 = sorted(d for (n, d) in t1.degree())
    degree_sequence2 = sorted(d for (n, d) in t2.degree())

    if degree_sequence1 != degree_sequence2:
        return []

    # A tree can have either 1 or 2 centers.
    # If the number doesn't match then t1 and t2 are not isomorphic.
    center1 = nx.center(t1)
    center2 = nx.center(t2)

    if len(center1) != len(center2):
        return []

    # If there is only 1 center in each, then use it.
    if len(center1) == 1:
        return rooted_tree_isomorphism(t1, center1[0], t2, center2[0])

    # If there both have 2 centers,  then try the first for t1
    # with the first for t2.
    attempts = rooted_tree_isomorphism(t1, center1[0], t2, center2[0])

    # If that worked we're done.
    if len(attempts) > 0:
        return attempts

    # Otherwise, try center1[0] with the center2[1], and see if that works
    return rooted_tree_isomorphism(t1, center1[0], t2, center2[1])
<|endoftext|>"""
ISMAGS Algorithm
================

Provides a Python implementation of the ISMAGS algorithm. [1]_

It is capable of finding (subgraph) isomorphisms between two graphs, taking the
symmetry of the subgraph into account. In most cases the VF2 algorithm is
faster (at least on small graphs) than this implementation, but in some cases
there is an exponential number of isomorphisms that are symmetrically
equivalent. In that case, the ISMAGS algorithm will provide only one solution
per symmetry group.

>>> petersen = nx.petersen_graph()
>>> ismags = nx.isomorphism.ISMAGS(petersen, petersen)
>>> isomorphisms = list(ismags.isomorphisms_iter(symmetry=False))
>>> len(isomorphisms)
120
>>> isomorphisms = list(ismags.isomorphisms_iter(symmetry=True))
>>> answer = [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}]
>>> answer == isomorphisms
True

In addition, this implementation also provides an interface to find the
largest common induced subgraph [2]_ between any two graphs, again taking
symmetry into account. Given `graph` and `subgraph` the algorithm will remove
nodes from the `subgraph` until `subgraph` is isomorphic to a subgraph of
`graph`. Since only the symmetry of `subgraph` is taken into account it is
worth thinking about how you provide your graphs:

>>> graph1 = nx.path_graph(4)
>>> graph2 = nx.star_graph(3)
>>> ismags = nx.isomorphism.ISMAGS(graph1, graph2)
>>> ismags.is_isomorphic()
False
>>> largest_common_subgraph = list(ismags.largest_common_subgraph())
>>> answer = [{1: 0, 0: 1, 2: 2}, {2: 0, 1: 1, 3: 2}]
>>> answer == largest_common_subgraph
True
>>> ismags2 = nx.isomorphism.ISMAGS(graph2, graph1)
>>> largest_common_subgraph = list(ismags2.largest_common_subgraph())
>>> answer = [
...     {1: 0, 0: 1, 2: 2},
...     {1: 0, 0: 1, 3: 2},
...     {2: 0, 0: 1, 1: 2},
...     {2: 0, 0: 1, 3: 2},
...     {3: 0, 0: 1, 1: 2},
...     {3: 0, 0: 1, 2: 2},
... ]
>>> answer == largest_common_subgraph
True

However, when not taking symmetry into account, it doesn't matter:

>>> largest_common_subgraph = list(ismags.largest_common_subgraph(symmetry=False))
>>> answer = [
...     {1: 0, 0: 1, 2: 2},
...     {1: 0, 2: 1, 0: 2},
...     {2: 0, 1: 1, 3: 2},
...     {2: 0, 3: 1, 1: 2},
...     {1: 0, 0: 1, 2: 3},
...     {1: 0, 2: 1, 0: 3},
...     {2: 0, 1: 1, 3: 3},
...     {2: 0, 3: 1, 1: 3},
...     {1: 0, 0: 2, 2: 3},
...     {1: 0, 2: 2, 0: 3},
...     {2: 0, 1: 2, 3: 3},
...     {2: 0, 3: 2, 1: 3},
... ]
>>> answer == largest_common_subgraph
True
>>> largest_common_subgraph = list(ismags2.largest_common_subgraph(symmetry=False))
>>> answer = [
...     {1: 0, 0: 1, 2: 2},
...     {1: 0, 0: 1, 3: 2},
...     {2: 0, 0: 1, 1: 2},
...     {2: 0, 0: 1, 3: 2},
...     {3: 0, 0: 1, 1: 2},
...     {3: 0, 0: 1, 2: 2},
...     {1: 1, 0: 2, 2: 3},
...     {1: 1, 0: 2, 3: 3},
...     {2: 1, 0: 2, 1: 3},
...     {2: 1, 0: 2, 3: 3},
...     {3: 1, 0: 2, 1: 3},
...     {3: 1, 0: 2, 2: 3},
... ]
>>> answer == largest_common_subgraph
True

Notes
-----
- The current implementation works for undirected graphs only. The algorithm
  in general should work for directed graphs as well though.
