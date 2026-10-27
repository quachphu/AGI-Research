                        continue
                    C[k, l] = inf

            localCe = make_CostMatrix(C, m, n)
            ij = [
                (
                    g_ind[k] if k < m else M + h_ind[l],
                    h_ind[l] if l < n else N + g_ind[k],
                )
                for k, l in zip(localCe.lsa_row_ind, localCe.lsa_col_ind)
                if k < m or l < n
            ]

        else:
            ij = []
            localCe = CostMatrix(np.empty((0, 0)), [], [], 0)

        return ij, localCe

    def reduce_Ce(Ce, ij, m, n):
        if len(ij):
            i, j = zip(*ij)
            m_i = m - sum(1 for t in i if t < m)
            n_j = n - sum(1 for t in j if t < n)
            return make_CostMatrix(reduce_C(Ce.C, i, j, m, n), m_i, n_j)
        return Ce

    def get_edit_ops(
        matched_uv, pending_u, pending_v, Cv, pending_g, pending_h, Ce, matched_cost
    ):
        """
        Parameters:
            matched_uv: partial vertex edit path
                list of tuples (u, v) of vertex mappings u<->v,
                u=None or v=None for deletion/insertion
            pending_u, pending_v: lists of vertices not yet mapped
            Cv: CostMatrix of pending vertex mappings
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_cost: cost of partial edit path

        Returns:
            sequence of
                (i, j): indices of vertex mapping u<->v
                Cv_ij: reduced CostMatrix of pending vertex mappings
                    (basically Cv with row i, col j removed)
                list of (x, y): indices of edge mappings g<->h
                Ce_xy: reduced CostMatrix of pending edge mappings
                    (basically Ce with rows x, cols y removed)
                cost: total cost of edit operation
            NOTE: most promising ops first
        """
        m = len(pending_u)
        n = len(pending_v)
        # assert Cv.C.shape == (m + n, m + n)

        # 1) a vertex mapping from optimal linear sum assignment
        i, j = min(
            (k, l) for k, l in zip(Cv.lsa_row_ind, Cv.lsa_col_ind) if k < m or l < n
        )
        xy, localCe = match_edges(
            pending_u[i] if i < m else None,
            pending_v[j] if j < n else None,
            pending_g,
            pending_h,
            Ce,
            matched_uv,
        )
        Ce_xy = reduce_Ce(Ce, xy, len(pending_g), len(pending_h))
        # assert Ce.ls <= localCe.ls + Ce_xy.ls
        if prune(matched_cost + Cv.ls + localCe.ls + Ce_xy.ls):
            pass
        else:
            # get reduced Cv efficiently
            Cv_ij = CostMatrix(
                reduce_C(Cv.C, (i,), (j,), m, n),
                reduce_ind(Cv.lsa_row_ind, (i, m + j)),
                reduce_ind(Cv.lsa_col_ind, (j, n + i)),
                Cv.ls - Cv.C[i, j],
            )
            yield (i, j), Cv_ij, xy, Ce_xy, Cv.C[i, j] + localCe.ls

        # 2) other candidates, sorted by lower-bound cost estimate
        other = []
        fixed_i, fixed_j = i, j
        if m <= n:
            candidates = (
                (t, fixed_j)
                for t in range(m + n)
                if t != fixed_i and (t < m or t == m + fixed_j)
            )
        else:
            candidates = (
                (fixed_i, t)
                for t in range(m + n)
                if t != fixed_j and (t < n or t == n + fixed_i)
            )
        for i, j in candidates:
            if prune(matched_cost + Cv.C[i, j] + Ce.ls):
                continue
            Cv_ij = make_CostMatrix(
                reduce_C(Cv.C, (i,), (j,), m, n),
                m - 1 if i < m else m,
                n - 1 if j < n else n,
            )
            # assert Cv.ls <= Cv.C[i, j] + Cv_ij.ls
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + Ce.ls):
                continue
            xy, localCe = match_edges(
                pending_u[i] if i < m else None,
                pending_v[j] if j < n else None,
                pending_g,
                pending_h,
                Ce,
                matched_uv,
            )
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + localCe.ls):
                continue
            Ce_xy = reduce_Ce(Ce, xy, len(pending_g), len(pending_h))
            # assert Ce.ls <= localCe.ls + Ce_xy.ls
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + localCe.ls + Ce_xy.ls):
                continue
            other.append(((i, j), Cv_ij, xy, Ce_xy, Cv.C[i, j] + localCe.ls))

        yield from sorted(other, key=lambda t: t[4] + t[1].ls + t[3].ls)

    def get_edit_paths(
        matched_uv,
        pending_u,
        pending_v,
        Cv,
        matched_gh,
        pending_g,
        pending_h,
        Ce,
        matched_cost,
    ):
        """
        Parameters:
            matched_uv: partial vertex edit path
                list of tuples (u, v) of vertex mappings u<->v,
                u=None or v=None for deletion/insertion
            pending_u, pending_v: lists of vertices not yet mapped
            Cv: CostMatrix of pending vertex mappings
            matched_gh: partial edge edit path
                list of tuples (g, h) of edge mappings g<->h,
                g=None or h=None for deletion/insertion
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_cost: cost of partial edit path

        Returns:
            sequence of (vertex_path, edge_path, cost)
                vertex_path: complete vertex edit path
                    list of tuples (u, v) of vertex mappings u<->v,
                    u=None or v=None for deletion/insertion
                edge_path: complete edge edit path
                    list of tuples (g, h) of edge mappings g<->h,
                    g=None or h=None for deletion/insertion
                cost: total cost of edit path
            NOTE: path costs are non-increasing
        """
        # debug_print('matched-uv:', matched_uv)
        # debug_print('matched-gh:', matched_gh)
        # debug_print('matched-cost:', matched_cost)
        # debug_print('pending-u:', pending_u)
        # debug_print('pending-v:', pending_v)
        # debug_print(Cv.C)
        # assert list(sorted(G1.nodes)) == list(sorted(list(u for u, v in matched_uv if u is not None) + pending_u))
        # assert list(sorted(G2.nodes)) == list(sorted(list(v for u, v in matched_uv if v is not None) + pending_v))
        # debug_print('pending-g:', pending_g)
        # debug_print('pending-h:', pending_h)
        # debug_print(Ce.C)
        # assert list(sorted(G1.edges)) == list(sorted(list(g for g, h in matched_gh if g is not None) + pending_g))
        # assert list(sorted(G2.edges)) == list(sorted(list(h for g, h in matched_gh if h is not None) + pending_h))
        # debug_print()

        if prune(matched_cost + Cv.ls + Ce.ls):
            return

        if not max(len(pending_u), len(pending_v)):
            # assert not len(pending_g)
            # assert not len(pending_h)
            # path completed!
            # assert matched_cost <= maxcost_value
            nonlocal maxcost_value
            maxcost_value = min(maxcost_value, matched_cost)
            yield matched_uv, matched_gh, matched_cost

        else:
            edit_ops = get_edit_ops(
                matched_uv,
                pending_u,
                pending_v,
                Cv,
                pending_g,
                pending_h,
                Ce,
                matched_cost,
            )
            for ij, Cv_ij, xy, Ce_xy, edit_cost in edit_ops:
                i, j = ij
                # assert Cv.C[i, j] + sum(Ce.C[t] for t in xy) == edit_cost
                if prune(matched_cost + edit_cost + Cv_ij.ls + Ce_xy.ls):
                    continue

                # dive deeper
                u = pending_u.pop(i) if i < len(pending_u) else None
                v = pending_v.pop(j) if j < len(pending_v) else None
                matched_uv.append((u, v))
                for x, y in xy:
                    len_g = len(pending_g)
                    len_h = len(pending_h)
                    matched_gh.append(
                        (
                            pending_g[x] if x < len_g else None,
                            pending_h[y] if y < len_h else None,
                        )
                    )
                sortedx = sorted(x for x, y in xy)
                sortedy = sorted(y for x, y in xy)
                G = [
                    (pending_g.pop(x) if x < len(pending_g) else None)
                    for x in reversed(sortedx)
                ]
                H = [
                    (pending_h.pop(y) if y < len(pending_h) else None)
                    for y in reversed(sortedy)
                ]

                yield from get_edit_paths(
                    matched_uv,
                    pending_u,
                    pending_v,
                    Cv_ij,
                    matched_gh,
                    pending_g,
                    pending_h,
                    Ce_xy,
                    matched_cost + edit_cost,
                )

                # backtrack
                if u is not None:
                    pending_u.insert(i, u)
                if v is not None:
                    pending_v.insert(j, v)
                matched_uv.pop()
                for x, g in zip(sortedx, reversed(G)):
                    if g is not None:
                        pending_g.insert(x, g)
                for y, h in zip(sortedy, reversed(H)):
                    if h is not None:
                        pending_h.insert(y, h)
                for _ in xy:
                    matched_gh.pop()

    # Initialization

    pending_u = list(G1.nodes)
    pending_v = list(G2.nodes)

    initial_cost = 0
    if roots:
        root_u, root_v = roots
        if root_u not in pending_u or root_v not in pending_v:
            raise nx.NodeNotFound("Root node not in graph.")

        # remove roots from pending
        pending_u.remove(root_u)
        pending_v.remove(root_v)

    # cost matrix of vertex mappings
    m = len(pending_u)
    n = len(pending_v)
    C = np.zeros((m + n, m + n))
    if node_subst_cost:
        C[0:m, 0:n] = np.array(
            [
                node_subst_cost(G1.nodes[u], G2.nodes[v])
                for u in pending_u
                for v in pending_v
            ]
        ).reshape(m, n)
        if roots:
            initial_cost = node_subst_cost(G1.nodes[root_u], G2.nodes[root_v])
    elif node_match:
        C[0:m, 0:n] = np.array(
            [
                1 - int(node_match(G1.nodes[u], G2.nodes[v]))
                for u in pending_u
                for v in pending_v
            ]
        ).reshape(m, n)
        if roots:
            initial_cost = 1 - node_match(G1.nodes[root_u], G2.nodes[root_v])
    else:
        # all zeroes
        pass
    # assert not min(m, n) or C[0:m, 0:n].min() >= 0
    if node_del_cost:
        del_costs = [node_del_cost(G1.nodes[u]) for u in pending_u]
    else:
        del_costs = [1] * len(pending_u)
    # assert not m or min(del_costs) >= 0
    if node_ins_cost:
        ins_costs = [node_ins_cost(G2.nodes[v]) for v in pending_v]
    else:
        ins_costs = [1] * len(pending_v)
    # assert not n or min(ins_costs) >= 0
    inf = C[0:m, 0:n].sum() + sum(del_costs) + sum(ins_costs) + 1
    C[0:m, n : n + m] = np.array(
        [del_costs[i] if i == j else inf for i in range(m) for j in range(m)]
    ).reshape(m, m)
    C[m : m + n, 0:n] = np.array(
        [ins_costs[i] if i == j else inf for i in range(n) for j in range(n)]
    ).reshape(n, n)
    Cv = make_CostMatrix(C, m, n)
    # debug_print(f"Cv: {m} x {n}")
    # debug_print(Cv.C)

    pending_g = list(G1.edges)
    pending_h = list(G2.edges)

    # cost matrix of edge mappings
    m = len(pending_g)
    n = len(pending_h)
    C = np.zeros((m + n, m + n))
    if edge_subst_cost:
        C[0:m, 0:n] = np.array(
            [
                edge_subst_cost(G1.edges[g], G2.edges[h])
                for g in pending_g
                for h in pending_h
            ]
        ).reshape(m, n)
    elif edge_match:
        C[0:m, 0:n] = np.array(
            [
                1 - int(edge_match(G1.edges[g], G2.edges[h]))
                for g in pending_g
                for h in pending_h
            ]
        ).reshape(m, n)
    else:
        # all zeroes
        pass
    # assert not min(m, n) or C[0:m, 0:n].min() >= 0
    if edge_del_cost:
        del_costs = [edge_del_cost(G1.edges[g]) for g in pending_g]
    else:
        del_costs = [1] * len(pending_g)
    # assert not m or min(del_costs) >= 0
    if edge_ins_cost:
        ins_costs = [edge_ins_cost(G2.edges[h]) for h in pending_h]
    else:
        ins_costs = [1] * len(pending_h)
    # assert not n or min(ins_costs) >= 0
    inf = C[0:m, 0:n].sum() + sum(del_costs) + sum(ins_costs) + 1
    C[0:m, n : n + m] = np.array(
        [del_costs[i] if i == j else inf for i in range(m) for j in range(m)]
    ).reshape(m, m)
    C[m : m + n, 0:n] = np.array(
        [ins_costs[i] if i == j else inf for i in range(n) for j in range(n)]
    ).reshape(n, n)
    Ce = make_CostMatrix(C, m, n)
    # debug_print(f'Ce: {m} x {n}')
    # debug_print(Ce.C)
    # debug_print()

    maxcost_value = Cv.C.sum() + Ce.C.sum() + 1

    if timeout is not None:
        if timeout <= 0:
            raise nx.NetworkXError("Timeout value must be greater than 0")
        start = time.perf_counter()

    def prune(cost):
        if timeout is not None:
            if time.perf_counter() - start > timeout:
                return True
        if upper_bound is not None:
            if cost > upper_bound:
                return True
        if cost > maxcost_value:
            return True
        if strictly_decreasing and cost >= maxcost_value:
            return True
        return False

    # Now go!

    done_uv = [] if roots is None else [roots]

    for vertex_path, edge_path, cost in get_edit_paths(
        done_uv, pending_u, pending_v, Cv, [], pending_g, pending_h, Ce, initial_cost
    ):
        # assert sorted(G1.nodes) == sorted(u for u, v in vertex_path if u is not None)
        # assert sorted(G2.nodes) == sorted(v for u, v in vertex_path if v is not None)
        # assert sorted(G1.edges) == sorted(g for g, h in edge_path if g is not None)
        # assert sorted(G2.edges) == sorted(h for g, h in edge_path if h is not None)
        # print(vertex_path, edge_path, cost, file = sys.stderr)
        # assert cost == maxcost_value
        yield list(vertex_path), list(edge_path), float(cost)


@nx._dispatchable
def simrank_similarity(
    G,
    source=None,
    target=None,
    importance_factor=0.9,
    max_iterations=1000,
    tolerance=1e-4,
):
    """Returns the SimRank similarity of nodes in the graph ``G``.

    SimRank is a similarity metric that says "two objects are considered
    to be similar if they are referenced by similar objects." [1]_.

    The pseudo-code definition from the paper is::

        def simrank(G, u, v):
            in_neighbors_u = G.predecessors(u)
            in_neighbors_v = G.predecessors(v)
            scale = C / (len(in_neighbors_u) * len(in_neighbors_v))
            return scale * sum(
                simrank(G, w, x) for w, x in product(in_neighbors_u, in_neighbors_v)
            )

    where ``G`` is the graph, ``u`` is the source, ``v`` is the target,
    and ``C`` is a float decay or importance factor between 0 and 1.

    The SimRank algorithm for determining node similarity is defined in
    [2]_.

    Parameters
    ----------
    G : NetworkX graph
        A NetworkX graph

    source : node
        If this is specified, the returned dictionary maps each node
        ``v`` in the graph to the similarity between ``source`` and
        ``v``.

    target : node
        If both ``source`` and ``target`` are specified, the similarity
        value between ``source`` and ``target`` is returned. If
        ``target`` is specified but ``source`` is not, this argument is
        ignored.

    importance_factor : float
        The relative importance of indirect neighbors with respect to
        direct neighbors.

    max_iterations : integer
        Maximum number of iterations.

    tolerance : float
        Error tolerance used to check convergence. When an iteration of
        the algorithm finds that no similarity value changes more than
        this amount, the algorithm halts.

    Returns
    -------
    similarity : dictionary or float
        If ``source`` and ``target`` are both ``None``, this returns a
        dictionary of dictionaries, where keys are node pairs and value
        are similarity of the pair of nodes.

        If ``source`` is not ``None`` but ``target`` is, this returns a
        dictionary mapping node to the similarity of ``source`` and that
        node.

        If neither ``source`` nor ``target`` is ``None``, this returns
        the similarity value for the given pair of nodes.

    Raises
    ------
    ExceededMaxIterations
        If the algorithm does not converge within ``max_iterations``.

    NodeNotFound
        If either ``source`` or ``target`` is not in `G`.

    Examples
    --------
    >>> G = nx.cycle_graph(2)
    >>> nx.simrank_similarity(G)
    {0: {0: 1.0, 1: 0.0}, 1: {0: 0.0, 1: 1.0}}
    >>> nx.simrank_similarity(G, source=0)
    {0: 1.0, 1: 0.0}
    >>> nx.simrank_similarity(G, source=0, target=0)
    1.0

    The result of this function can be converted to a numpy array
    representing the SimRank matrix by using the node order of the
    graph to determine which row and column represent each node.
    Other ordering of nodes is also possible.

    >>> import numpy as np
    >>> sim = nx.simrank_similarity(G)
    >>> np.array([[sim[u][v] for v in G] for u in G])
    array([[1., 0.],
           [0., 1.]])
    >>> sim_1d = nx.simrank_similarity(G, source=0)
    >>> np.array([sim[0][v] for v in G])
    array([1., 0.])

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/SimRank
    .. [2] G. Jeh and J. Widom.
           "SimRank: a measure of structural-context similarity",
           In KDD'02: Proceedings of the Eighth ACM SIGKDD
           International Conference on Knowledge Discovery and Data Mining,
           pp. 538--543. ACM Press, 2002.
    """
    import numpy as np

    nodelist = list(G)
    if source is not None:
        if source not in nodelist:
            raise nx.NodeNotFound(f"Source node {source} not in G")
        else:
            s_indx = nodelist.index(source)
    else:
        s_indx = None

    if target is not None:
        if target not in nodelist:
            raise nx.NodeNotFound(f"Target node {target} not in G")
        else:
            t_indx = nodelist.index(target)
    else:
        t_indx = None

    x = _simrank_similarity_numpy(
        G, s_indx, t_indx, importance_factor, max_iterations, tolerance
    )

    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return dict(zip(G, x.tolist()))
        # else x.ndim == 2
        return {u: dict(zip(G, row)) for u, row in zip(G, x.tolist())}
    return float(x)


def _simrank_similarity_python(
    G,
    source=None,
    target=None,
    importance_factor=0.9,
    max_iterations=1000,
    tolerance=1e-4,
):
    """Returns the SimRank similarity of nodes in the graph ``G``.

    This pure Python version is provided for pedagogical purposes.

    Examples
    --------
    >>> G = nx.cycle_graph(2)
    >>> nx.similarity._simrank_similarity_python(G)
    {0: {0: 1, 1: 0.0}, 1: {0: 0.0, 1: 1}}
    >>> nx.similarity._simrank_similarity_python(G, source=0)
    {0: 1, 1: 0.0}
    >>> nx.similarity._simrank_similarity_python(G, source=0, target=0)
    1
    """
    # build up our similarity adjacency dictionary output
    newsim = {u: {v: 1 if u == v else 0 for v in G} for u in G}

    # These functions compute the update to the similarity value of the nodes
    # `u` and `v` with respect to the previous similarity values.
    def avg_sim(s):
        return sum(newsim[w][x] for (w, x) in s) / len(s) if s else 0.0

    Gadj = G.pred if G.is_directed() else G.adj

    def sim(u, v):
        return importance_factor * avg_sim(list(product(Gadj[u], Gadj[v])))

    for its in range(max_iterations):
        oldsim = newsim
        newsim = {u: {v: sim(u, v) if u != v else 1 for v in G} for u in G}
        is_close = all(
            all(
                abs(newsim[u][v] - old) <= tolerance * (1 + abs(old))
                for v, old in nbrs.items()
            )
            for u, nbrs in oldsim.items()
        )
        if is_close:
            break

    if its + 1 == max_iterations:
        raise nx.ExceededMaxIterations(
            f"simrank did not converge after {max_iterations} iterations."
        )

    if source is not None and target is not None:
        return newsim[source][target]
    if source is not None:
        return newsim[source]
    return newsim


def _simrank_similarity_numpy(
    G,
    source=None,
    target=None,
    importance_factor=0.9,
    max_iterations=1000,
    tolerance=1e-4,
):
    """Calculate SimRank of nodes in ``G`` using matrices with ``numpy``.

    The SimRank algorithm for determining node similarity is defined in
    [1]_.

    Parameters
    ----------
    G : NetworkX graph
        A NetworkX graph

    source : node
        If this is specified, the returned dictionary maps each node
        ``v`` in the graph to the similarity between ``source`` and
        ``v``.

    target : node
        If both ``source`` and ``target`` are specified, the similarity
        value between ``source`` and ``target`` is returned. If
        ``target`` is specified but ``source`` is not, this argument is
        ignored.

    importance_factor : float
        The relative importance of indirect neighbors with respect to
        direct neighbors.

    max_iterations : integer
        Maximum number of iterations.

    tolerance : float
        Error tolerance used to check convergence. When an iteration of
        the algorithm finds that no similarity value changes more than
        this amount, the algorithm halts.

    Returns
    -------
    similarity : numpy array or float
        If ``source`` and ``target`` are both ``None``, this returns a
        2D array containing SimRank scores of the nodes.

        If ``source`` is not ``None`` but ``target`` is, this returns an
        1D array containing SimRank scores of ``source`` and that
        node.

        If neither ``source`` nor ``target`` is ``None``, this returns
        the similarity value for the given pair of nodes.

    Examples
    --------
    >>> G = nx.cycle_graph(2)
    >>> nx.similarity._simrank_similarity_numpy(G)
    array([[1., 0.],
           [0., 1.]])
    >>> nx.similarity._simrank_similarity_numpy(G, source=0)
    array([1., 0.])
    >>> nx.similarity._simrank_similarity_numpy(G, source=0, target=0)
    1.0

    References
    ----------
    .. [1] G. Jeh and J. Widom.
           "SimRank: a measure of structural-context similarity",
           In KDD'02: Proceedings of the Eighth ACM SIGKDD
           International Conference on Knowledge Discovery and Data Mining,
           pp. 538--543. ACM Press, 2002.
    """
    # This algorithm follows roughly
    #
    #     S = max{C * (A.T * S * A), I}
    #
    # where C is the importance factor, A is the column normalized
    # adjacency matrix, and I is the identity matrix.
    import numpy as np

    adjacency_matrix = nx.to_numpy_array(G)

    # column-normalize the ``adjacency_matrix``
    s = np.array(adjacency_matrix.sum(axis=0))
    s[s == 0] = 1
    adjacency_matrix /= s  # adjacency_matrix.sum(axis=0)

    newsim = np.eye(len(G), dtype=np.float64)
    for its in range(max_iterations):
        prevsim = newsim.copy()
        newsim = importance_factor * ((adjacency_matrix.T @ prevsim) @ adjacency_matrix)
        np.fill_diagonal(newsim, 1.0)

        if np.allclose(prevsim, newsim, atol=tolerance):
            break

    if its + 1 == max_iterations:
        raise nx.ExceededMaxIterations(
            f"simrank did not converge after {max_iterations} iterations."
        )

    if source is not None and target is not None:
        return float(newsim[source, target])
    if source is not None:
        return newsim[source]
    return newsim


@nx._dispatchable(edge_attrs="weight")
def panther_similarity(
    G, source, k=5, path_length=5, c=0.5, delta=0.1, eps=None, weight="weight"
):
    r"""Returns the Panther similarity of nodes in the graph `G` to node ``v``.

    Panther is a similarity metric that says "two objects are considered
    to be similar if they frequently appear on the same paths." [1]_.

    Parameters
    ----------
    G : NetworkX graph
        A NetworkX graph
    source : node
        Source node for which to find the top `k` similar other nodes
    k : int (default = 5)
        The number of most similar nodes to return.
    path_length : int (default = 5)
        How long the randomly generated paths should be (``T`` in [1]_)
    c : float (default = 0.5)
        A universal positive constant used to scale the number
        of sample random paths to generate.
    delta : float (default = 0.1)
        The probability that the similarity $S$ is not an epsilon-approximation to (R, phi),
        where $R$ is the number of random paths and $\phi$ is the probability
        that an element sampled from a set $A \subseteq D$, where $D$ is the domain.
    eps : float or None (default = None)
        The error bound. Per [1]_, a good value is ``sqrt(1/|E|)``. Therefore,
        if no value is provided, the recommended computed value will be used.
    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.

    Returns
    -------
    similarity : dictionary
        Dictionary of nodes to similarity scores (as floats). Note:
        the self-similarity (i.e., ``v``) will not be included in
        the returned dictionary. So, for ``k = 5``, a dictionary of
        top 4 nodes and their similarity scores will be returned.

    Raises
    ------
    NetworkXUnfeasible
        If `source` is an isolated node.

    NodeNotFound
        If `source` is not in `G`.

    Notes
    -----
        The isolated nodes in `G` are ignored.

    Examples
    --------
    >>> G = nx.star_graph(10)
    >>> sim = nx.panther_similarity(G, 0)

    References
    ----------
    .. [1] Zhang, J., Tang, J., Ma, C., Tong, H., Jing, Y., & Li, J.
           Panther: Fast top-k similarity search on large networks.
           In Proceedings of the ACM SIGKDD International Conference
           on Knowledge Discovery and Data Mining (Vol. 2015-August, pp. 1445–1454).
           Association for Computing Machinery. https://doi.org/10.1145/2783258.2783267.
    """
    import numpy as np

    if source not in G:
        raise nx.NodeNotFound(f"Source node {source} not in G")

    isolates = set(nx.isolates(G))

    if source in isolates:
        raise nx.NetworkXUnfeasible(
            f"Panther similarity is not defined for the isolated source node {source}."
        )

    G = G.subgraph([node for node in G.nodes if node not in isolates]).copy()

    num_nodes = G.number_of_nodes()
    if num_nodes < k:
        warnings.warn(
            f"Number of nodes is {num_nodes}, but requested k is {k}. "
            "Setting k to number of nodes."
        )
        k = num_nodes
    # According to [1], they empirically determined
    # a good value for ``eps`` to be sqrt( 1 / |E| )
    if eps is None:
        eps = np.sqrt(1.0 / G.number_of_edges())

    inv_node_map = {name: index for index, name in enumerate(G.nodes)}
    node_map = np.array(G)

    # Calculate the sample size ``R`` for how many paths
    # to randomly generate
    t_choose_2 = math.comb(path_length, 2)
    sample_size = int((c / eps**2) * (np.log2(t_choose_2) + 1 + np.log(1 / delta)))
    index_map = {}
    _ = list(
        generate_random_paths(
            G, sample_size, path_length=path_length, index_map=index_map, weight=weight
        )
    )
    S = np.zeros(num_nodes)

    inv_sample_size = 1 / sample_size

    source_paths = set(index_map[source])

    # Calculate the path similarities
    # between ``source`` (v) and ``node`` (v_j)
    # using our inverted index mapping of
    # vertices to paths
    for node, paths in index_map.items():
        # Only consider paths where both
        # ``node`` and ``source`` are present
        common_paths = source_paths.intersection(paths)
        S[inv_node_map[node]] = len(common_paths) * inv_sample_size

    # Retrieve top ``k`` similar
    # Note: the below performed anywhere from 4-10x faster
    # (depending on input sizes) vs the equivalent ``np.argsort(S)[::-1]``
    top_k_unsorted = np.argpartition(S, -k)[-k:]
    top_k_sorted = top_k_unsorted[np.argsort(S[top_k_unsorted])][::-1]

    # Add back the similarity scores
    top_k_with_val = dict(
        zip(node_map[top_k_sorted].tolist(), S[top_k_sorted].tolist())
    )

    # Remove the self-similarity
    top_k_with_val.pop(source, None)
    return top_k_with_val


@np_random_state(5)
@nx._dispatchable(edge_attrs="weight")
def generate_random_paths(
    G, sample_size, path_length=5, index_map=None, weight="weight", seed=None
):
    """Randomly generate `sample_size` paths of length `path_length`.

    Parameters
    ----------
    G : NetworkX graph
        A NetworkX graph
    sample_size : integer
        The number of paths to generate. This is ``R`` in [1]_.
    path_length : integer (default = 5)
        The maximum size of the path to randomly generate.
        This is ``T`` in [1]_. According to the paper, ``T >= 5`` is
        recommended.
    index_map : dictionary, optional
        If provided, this will be populated with the inverted
        index of nodes mapped to the set of generated random path
        indices within ``paths``.
    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    paths : generator of lists
        Generator of `sample_size` paths each with length `path_length`.

    Examples
    --------
    Note that the return value is the list of paths:

    >>> G = nx.star_graph(3)
    >>> random_path = nx.generate_random_paths(G, 2)

    By passing a dictionary into `index_map`, it will build an
    inverted index mapping of nodes to the paths in which that node is present:

    >>> G = nx.star_graph(3)
    >>> index_map = {}
    >>> random_path = nx.generate_random_paths(G, 3, index_map=index_map)
    >>> paths_containing_node_0 = [
    ...     random_path[path_idx] for path_idx in index_map.get(0, [])
    ... ]

    References
    ----------
    .. [1] Zhang, J., Tang, J., Ma, C., Tong, H., Jing, Y., & Li, J.
           Panther: Fast top-k similarity search on large networks.
           In Proceedings of the ACM SIGKDD International Conference
           on Knowledge Discovery and Data Mining (Vol. 2015-August, pp. 1445–1454).
           Association for Computing Machinery. https://doi.org/10.1145/2783258.2783267.
    """
    import numpy as np

    randint_fn = (
        seed.integers if isinstance(seed, np.random.Generator) else seed.randint
    )

    # Calculate transition probabilities between
    # every pair of vertices according to Eq. (3)
    adj_mat = nx.to_numpy_array(G, weight=weight)
    inv_row_sums = np.reciprocal(adj_mat.sum(axis=1)).reshape(-1, 1)
    transition_probabilities = adj_mat * inv_row_sums

    node_map = list(G)
    num_nodes = G.number_of_nodes()

    for path_index in range(sample_size):
        # Sample current vertex v = v_i uniformly at random
        node_index = randint_fn(num_nodes)
        node = node_map[node_index]

        # Add v into p_r and add p_r into the path set
        # of v, i.e., P_v
        path = [node]

        # Build the inverted index (P_v) of vertices to paths
        if index_map is not None:
            if node in index_map:
                index_map[node].add(path_index)
            else:
                index_map[node] = {path_index}

        starting_index = node_index
        for _ in range(path_length):
            # Randomly sample a neighbor (v_j) according
            # to transition probabilities from ``node`` (v) to its neighbors
            nbr_index = seed.choice(
                num_nodes, p=transition_probabilities[starting_index]
            )

            # Set current vertex (v = v_j)
            starting_index = nbr_index

            # Add v into p_r
            nbr_node = node_map[nbr_index]
            path.append(nbr_node)

            # Add p_r into P_v
            if index_map is not None:
                if nbr_node in index_map:
                    index_map[nbr_node].add(path_index)
                else:
                    index_map[nbr_node] = {path_index}

        yield path
<|endoftext|>"""
Provides functions for finding and testing for locally `(k, l)`-connected
graphs.

"""

import copy

import networkx as nx

__all__ = ["kl_connected_subgraph", "is_kl_connected"]


@nx._dispatchable(returns_graph=True)
def kl_connected_subgraph(G, k, l, low_memory=False, same_as_graph=False):
    """Returns the maximum locally `(k, l)`-connected subgraph of `G`.

    A graph is locally `(k, l)`-connected if for each edge `(u, v)` in the
    graph there are at least `l` edge-disjoint paths of length at most `k`
    joining `u` to `v`.

    Parameters
    ----------
    G : NetworkX graph
        The graph in which to find a maximum locally `(k, l)`-connected
        subgraph.

    k : integer
        The maximum length of paths to consider. A higher number means a looser
        connectivity requirement.

    l : integer
        The number of edge-disjoint paths. A higher number means a stricter
        connectivity requirement.

    low_memory : bool
        If this is True, this function uses an algorithm that uses slightly
        more time but less memory.

    same_as_graph : bool
        If True then return a tuple of the form `(H, is_same)`,
        where `H` is the maximum locally `(k, l)`-connected subgraph and
        `is_same` is a Boolean representing whether `G` is locally `(k,
        l)`-connected (and hence, whether `H` is simply a copy of the input
        graph `G`).

    Returns
    -------
    NetworkX graph or two-tuple
        If `same_as_graph` is True, then this function returns a
        two-tuple as described above. Otherwise, it returns only the maximum
        locally `(k, l)`-connected subgraph.

    See also
    --------
    is_kl_connected

    References
    ----------
    .. [1] Chung, Fan and Linyuan Lu. "The Small World Phenomenon in Hybrid
           Power Law Graphs." *Complex Networks*. Springer Berlin Heidelberg,
           2004. 89--104.

    """
    H = copy.deepcopy(G)  # subgraph we construct by removing from G

    graphOK = True
    deleted_some = True  # hack to start off the while loop
    while deleted_some:
        deleted_some = False
        # We use `for edge in list(H.edges()):` instead of
        # `for edge in H.edges():` because we edit the graph `H` in
        # the loop. Hence using an iterator will result in
        # `RuntimeError: dictionary changed size during iteration`
        for edge in list(H.edges()):
            (u, v) = edge
            # Get copy of graph needed for this search
            if low_memory:
                verts = {u, v}
                for i in range(k):
                    for w in verts.copy():
                        verts.update(G[w])
                G2 = G.subgraph(verts).copy()
            else:
                G2 = copy.deepcopy(G)
            ###
            path = [u, v]
            cnt = 0
            accept = 0
            while path:
                cnt += 1  # Found a path
                if cnt >= l:
                    accept = 1
                    break
                # record edges along this graph
                prev = u
                for w in path:
                    if prev != w:
                        G2.remove_edge(prev, w)
                        prev = w
                #                path = shortest_path(G2, u, v, k) # ??? should "Cutoff" be k+1?
                try:
                    path = nx.shortest_path(G2, u, v)  # ??? should "Cutoff" be k+1?
                except nx.NetworkXNoPath:
                    path = False
            # No Other Paths
            if accept == 0:
                H.remove_edge(u, v)
                deleted_some = True
                if graphOK:
                    graphOK = False
    # We looked through all edges and removed none of them.
    # So, H is the maximal (k,l)-connected subgraph of G
    if same_as_graph:
        return (H, graphOK)
    return H


@nx._dispatchable
def is_kl_connected(G, k, l, low_memory=False):
    """Returns True if and only if `G` is locally `(k, l)`-connected.

    A graph is locally `(k, l)`-connected if for each edge `(u, v)` in the
    graph there are at least `l` edge-disjoint paths of length at most `k`
    joining `u` to `v`.

    Parameters
    ----------
    G : NetworkX graph
        The graph to test for local `(k, l)`-connectedness.

    k : integer
        The maximum length of paths to consider. A higher number means a looser
        connectivity requirement.

    l : integer
        The number of edge-disjoint paths. A higher number means a stricter
        connectivity requirement.

    low_memory : bool
        If this is True, this function uses an algorithm that uses slightly
        more time but less memory.

    Returns
    -------
    bool
        Whether the graph is locally `(k, l)`-connected subgraph.

    See also
    --------
    kl_connected_subgraph

    References
    ----------
    .. [1] Chung, Fan and Linyuan Lu. "The Small World Phenomenon in Hybrid
           Power Law Graphs." *Complex Networks*. Springer Berlin Heidelberg,
           2004. 89--104.

    """
    graphOK = True
    for edge in G.edges():
        (u, v) = edge
        # Get copy of graph needed for this search
        if low_memory:
            verts = {u, v}
            for i in range(k):
                [verts.update(G.neighbors(w)) for w in verts.copy()]
            G2 = G.subgraph(verts)
        else:
            G2 = copy.deepcopy(G)
        ###
        path = [u, v]
        cnt = 0
        accept = 0
        while path:
            cnt += 1  # Found a path
            if cnt >= l:
                accept = 1
                break
            # record edges along this graph
            prev = u
            for w in path:
                if w != prev:
                    G2.remove_edge(prev, w)
                    prev = w
            #            path = shortest_path(G2, u, v, k) # ??? should "Cutoff" be k+1?
            try:
                path = nx.shortest_path(G2, u, v)  # ??? should "Cutoff" be k+1?
            except nx.NetworkXNoPath:
                path = False
        # No Other Paths
        if accept == 0:
            graphOK = False
            break
    # return status
    return graphOK
<|endoftext|>"""Functions related to the Wiener Index of a graph.

The Wiener Index is a topological measure of a graph
related to the distance between nodes and their degree.
The Schultz Index and Gutman Index are similar measures.
They are used categorize molecules via the network of
atoms connected by chemical bonds. The indices are
correlated with functional aspects of the molecules.

References
----------
.. [1] `Wikipedia: Wiener Index <https://en.wikipedia.org/wiki/Wiener_index>`_
.. [2] M.V. Diudeaa and I. Gutman, Wiener-Type Topological Indices,
       Croatica Chemica Acta, 71 (1998), 21-51.
       https://hrcak.srce.hr/132323
"""

import itertools as it

import networkx as nx

__all__ = ["wiener_index", "schultz_index", "gutman_index"]


@nx._dispatchable(edge_attrs="weight")
def wiener_index(G, weight=None):
    """Returns the Wiener index of the given graph.

    The *Wiener index* of a graph is the sum of the shortest-path
    (weighted) distances between each pair of reachable nodes.
    For pairs of nodes in undirected graphs, only one orientation
    of the pair is counted.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default: None)
        If None, every edge has weight 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        The edge weights are used to computing shortest-path distances.

    Returns
    -------
    number
        The Wiener index of the graph `G`.

    Raises
    ------
    NetworkXError
        If the graph `G` is not connected.

    Notes
    -----
    If a pair of nodes is not reachable, the distance is assumed to be
    infinity. This means that for graphs that are not
    strongly-connected, this function returns ``inf``.

    The Wiener index is not usually defined for directed graphs, however
    this function uses the natural generalization of the Wiener index to
    directed graphs.

    Examples
    --------
    The Wiener index of the (unweighted) complete graph on *n* nodes
    equals the number of pairs of the *n* nodes, since each pair of
    nodes is at distance one::

        >>> n = 10
        >>> G = nx.complete_graph(n)
        >>> nx.wiener_index(G) == n * (n - 1) / 2
        True

    Graphs that are not strongly-connected have infinite Wiener index::

        >>> G = nx.empty_graph(2)
        >>> nx.wiener_index(G)
        inf

    References
    ----------
    .. [1] `Wikipedia: Wiener Index <https://en.wikipedia.org/wiki/Wiener_index>`_
    """
    connected = nx.is_strongly_connected(G) if G.is_directed() else nx.is_connected(G)
    if not connected:
        return float("inf")

    spl = nx.shortest_path_length(G, weight=weight)
    total = sum(it.chain.from_iterable(nbrs.values() for node, nbrs in spl))
    # Need to account for double counting pairs of nodes in undirected graphs.
    return total if G.is_directed() else total / 2


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def schultz_index(G, weight=None):
    r"""Returns the Schultz Index (of the first kind) of `G`

    The *Schultz Index* [3]_ of a graph is the sum over all node pairs of
    distances times the sum of degrees. Consider an undirected graph `G`.
    For each node pair ``(u, v)`` compute ``dist(u, v) * (deg(u) + deg(v)``
    where ``dist`` is the shortest path length between two nodes and ``deg``
    is the degree of a node.

    The Schultz Index is the sum of these quantities over all (unordered)
    pairs of nodes.

    Parameters
    ----------
    G : NetworkX graph
        The undirected graph of interest.
    weight : string or None, optional (default: None)
        If None, every edge has weight 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        The edge weights are used to computing shortest-path distances.

    Returns
    -------
    number
        The first kind of Schultz Index of the graph `G`.

    Examples
    --------
    The Schultz Index of the (unweighted) complete graph on *n* nodes
    equals the number of pairs of the *n* nodes times ``2 * (n - 1)``,
    since each pair of nodes is at distance one and the sum of degree
    of two nodes is ``2 * (n - 1)``.

    >>> n = 10
    >>> G = nx.complete_graph(n)
    >>> nx.schultz_index(G) == (n * (n - 1) / 2) * (2 * (n - 1))
    True

    Graph that is disconnected

    >>> nx.schultz_index(nx.empty_graph(2))
    inf

    References
    ----------
    .. [1] I. Gutman, Selected properties of the Schultz molecular topological index,
           J. Chem. Inf. Comput. Sci. 34 (1994), 1087–1089.
           https://doi.org/10.1021/ci00021a009
    .. [2] M.V. Diudeaa and I. Gutman, Wiener-Type Topological Indices,
           Croatica Chemica Acta, 71 (1998), 21-51.
           https://hrcak.srce.hr/132323
    .. [3] H. P. Schultz, Topological organic chemistry. 1.
           Graph theory and topological indices of alkanes,i
           J. Chem. Inf. Comput. Sci. 29 (1989), 239–257.

    """
    if not nx.is_connected(G):
        return float("inf")

    spl = nx.shortest_path_length(G, weight=weight)
    d = dict(G.degree, weight=weight)
    return sum(dist * (d[u] + d[v]) for u, info in spl for v, dist in info.items()) / 2


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def gutman_index(G, weight=None):
    r"""Returns the Gutman Index for the graph `G`.

    The *Gutman Index* measures the topology of networks, especially for molecule
    networks of atoms connected by bonds [1]_. It is also called the Schultz Index
    of the second kind [2]_.

    Consider an undirected graph `G` with node set ``V``.
    The Gutman Index of a graph is the sum over all (unordered) pairs of nodes
    of nodes ``(u, v)``, with distance ``dist(u, v)`` and degrees ``deg(u)``
    and ``deg(v)``, of ``dist(u, v) * deg(u) * deg(v)``

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default: None)
        If None, every edge has weight 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        The edge weights are used to computing shortest-path distances.

    Returns
    -------
    number
        The Gutman Index of the graph `G`.

    Examples
    --------
    The Gutman Index of the (unweighted) complete graph on *n* nodes
    equals the number of pairs of the *n* nodes times ``(n - 1) * (n - 1)``,
    since each pair of nodes is at distance one and the product of degree of two
    vertices is ``(n - 1) * (n - 1)``.

    >>> n = 10
    >>> G = nx.complete_graph(n)
    >>> nx.gutman_index(G) == (n * (n - 1) / 2) * ((n - 1) * (n - 1))
    True

    Graphs that are disconnected

    >>> G = nx.empty_graph(2)
    >>> nx.gutman_index(G)
    inf

    References
    ----------
    .. [1] M.V. Diudeaa and I. Gutman, Wiener-Type Topological Indices,
           Croatica Chemica Acta, 71 (1998), 21-51.
           https://hrcak.srce.hr/132323
    .. [2] I. Gutman, Selected properties of the Schultz molecular topological index,
           J. Chem. Inf. Comput. Sci. 34 (1994), 1087–1089.
           https://doi.org/10.1021/ci00021a009

    """
    if not nx.is_connected(G):
        return float("inf")

    spl = nx.shortest_path_length(G, weight=weight)
    d = dict(G.degree, weight=weight)
    return sum(dist * d[u] * d[v] for u, vinfo in spl for v, dist in vinfo.items()) / 2
<|endoftext|>"""
Algorithms for asteroidal triples and asteroidal numbers in graphs.

An asteroidal triple in a graph G is a set of three non-adjacent vertices
u, v and w such that there exist a path between any two of them that avoids
closed neighborhood of the third. More formally, v_j, v_k belongs to the same
connected component of G - N[v_i], where N[v_i] denotes the closed neighborhood
of v_i. A graph which does not contain any asteroidal triples is called
an AT-free graph. The class of AT-free graphs is a graph class for which
many NP-complete problems are solvable in polynomial time. Amongst them,
independent set and coloring.
"""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["is_at_free", "find_asteroidal_triple"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def find_asteroidal_triple(G):
    r"""Find an asteroidal triple in the given graph.

    An asteroidal triple is a triple of non-adjacent vertices such that
    there exists a path between any two of them which avoids the closed
    neighborhood of the third. It checks all independent triples of vertices
    and whether they are an asteroidal triple or not. This is done with the
    help of a data structure called a component structure.
    A component structure encodes information about which vertices belongs to
    the same connected component when the closed neighborhood of a given vertex
    is removed from the graph. The algorithm used to check is the trivial
    one, outlined in [1]_, which has a runtime of
    :math:`O(|V||\overline{E} + |V||E|)`, where the second term is the
    creation of the component structure.

    Parameters
    ----------
    G : NetworkX Graph
        The graph to check whether is AT-free or not

    Returns
    -------
    list or None
        An asteroidal triple is returned as a list of nodes. If no asteroidal
        triple exists, i.e. the graph is AT-free, then None is returned.
        The returned value depends on the certificate parameter. The default
        option is a bool which is True if the graph is AT-free, i.e. the
        given graph contains no asteroidal triples, and False otherwise, i.e.
        if the graph contains at least one asteroidal triple.

    Notes
    -----
    The component structure and the algorithm is described in [1]_. The current
    implementation implements the trivial algorithm for simple graphs.

    References
    ----------
    .. [1] Ekkehard Köhler,
       "Recognizing Graphs without asteroidal triples",
       Journal of Discrete Algorithms 2, pages 439-452, 2004.
       https://www.sciencedirect.com/science/article/pii/S157086670400019X
    """
    V = set(G.nodes)

    if len(V) < 6:
        # An asteroidal triple cannot exist in a graph with 5 or less vertices.
        return None

    component_structure = create_component_structure(G)
    E_complement = set(nx.complement(G).edges)

    for e in E_complement:
        u = e[0]
        v = e[1]
        u_neighborhood = set(G[u]).union([u])
        v_neighborhood = set(G[v]).union([v])
        union_of_neighborhoods = u_neighborhood.union(v_neighborhood)
        for w in V - union_of_neighborhoods:
            # Check for each pair of vertices whether they belong to the
            # same connected component when the closed neighborhood of the
            # third is removed.
            if (
                component_structure[u][v] == component_structure[u][w]
                and component_structure[v][u] == component_structure[v][w]
                and component_structure[w][u] == component_structure[w][v]
            ):
                return [u, v, w]
    return None


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_at_free(G):
    """Check if a graph is AT-free.

    The method uses the `find_asteroidal_triple` method to recognize
    an AT-free graph. If no asteroidal triple is found the graph is
    AT-free and True is returned. If at least one asteroidal triple is
    found the graph is not AT-free and False is returned.

    Parameters
    ----------
    G : NetworkX Graph
        The graph to check whether is AT-free or not.

    Returns
    -------
    bool
        True if G is AT-free and False otherwise.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (4, 5)])
    >>> nx.is_at_free(G)
    True

    >>> G = nx.cycle_graph(6)
    >>> nx.is_at_free(G)
    False
    """
    return find_asteroidal_triple(G) is None


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def create_component_structure(G):
    r"""Create component structure for G.

    A *component structure* is an `nxn` array, denoted `c`, where `n` is
    the number of vertices,  where each row and column corresponds to a vertex.

    .. math::
        c_{uv} = \begin{cases} 0, if v \in N[u] \\
            k, if v \in component k of G \setminus N[u] \end{cases}

    Where `k` is an arbitrary label for each component. The structure is used
    to simplify the detection of asteroidal triples.

    Parameters
    ----------
    G : NetworkX Graph
        Undirected, simple graph.

    Returns
    -------
    component_structure : dictionary
        A dictionary of dictionaries, keyed by pairs of vertices.

    """
    V = set(G.nodes)
    component_structure = {}
    for v in V:
        label = 0
        closed_neighborhood = set(G[v]).union({v})
        row_dict = {}
        for u in closed_neighborhood:
            row_dict[u] = 0

        G_reduced = G.subgraph(set(G.nodes) - closed_neighborhood)
        for cc in nx.connected_components(G_reduced):
            label += 1
            for u in cc:
                row_dict[u] = label

        component_structure[v] = row_dict

    return component_structure
<|endoftext|>"""Functions concerning tournament graphs.

A `tournament graph`_ is a complete oriented graph. In other words, it
is a directed graph in which there is exactly one directed edge joining
each pair of distinct nodes. For each function in this module that
accepts a graph as input, you must provide a tournament graph. The
responsibility is on the caller to ensure that the graph is a tournament
graph:

    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    >>> nx.is_tournament(G)
    True

To access the functions in this module, you must access them through the
:mod:`networkx.tournament` module::

    >>> nx.tournament.is_reachable(G, 0, 1)
    True

.. _tournament graph: https://en.wikipedia.org/wiki/Tournament_%28graph_theory%29

"""

from itertools import combinations

import networkx as nx
from networkx.algorithms.simple_paths import is_simple_path as is_path
from networkx.utils import arbitrary_element, not_implemented_for, py_random_state

__all__ = [
    "hamiltonian_path",
    "is_reachable",
    "is_strongly_connected",
    "is_tournament",
    "random_tournament",
    "score_sequence",
]


def index_satisfying(iterable, condition):
    """Returns the index of the first element in `iterable` that
    satisfies the given condition.

    If no such element is found (that is, when the iterable is
    exhausted), this returns the length of the iterable (that is, one
    greater than the last index of the iterable).

    `iterable` must not be empty. If `iterable` is empty, this
    function raises :exc:`ValueError`.

    """
    # Pre-condition: iterable must not be empty.
    for i, x in enumerate(iterable):
        if condition(x):
            return i
    # If we reach the end of the iterable without finding an element
    # that satisfies the condition, return the length of the iterable,
    # which is one greater than the index of its last element. If the
    # iterable was empty, `i` will not be defined, so we raise an
    # exception.
    try:
        return i + 1
    except NameError as err:
        raise ValueError("iterable must be non-empty") from err


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_tournament(G):
    """Returns True if and only if `G` is a tournament.

    A tournament is a directed graph, with neither self-loops nor
    multi-edges, in which there is exactly one directed edge joining
    each pair of distinct nodes.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph representing a tournament.

    Returns
    -------
    bool
        Whether the given graph is a tournament graph.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    >>> nx.is_tournament(G)
    True

    Notes
    -----
    Some definitions require a self-loop on each node, but that is not
    the convention used here.

    """
    # In a tournament, there is exactly one directed edge joining each pair.
    return (
        all((v in G[u]) ^ (u in G[v]) for u, v in combinations(G, 2))
        and nx.number_of_selfloops(G) == 0
    )


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
@nx._dispatchable
def hamiltonian_path(G):
    """Returns a Hamiltonian path in the given tournament graph.

    Each tournament has a Hamiltonian path. If furthermore, the
    tournament is strongly connected, then the returned Hamiltonian path
    is a Hamiltonian cycle (by joining the endpoints of the path).

    Parameters
    ----------
    G : NetworkX graph
        A directed graph representing a tournament.

    Returns
    -------
    path : list
        A list of nodes which form a Hamiltonian path in `G`.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    >>> nx.is_tournament(G)
    True
    >>> nx.tournament.hamiltonian_path(G)
    [0, 1, 2, 3]

    Notes
    -----
    This is a recursive implementation with an asymptotic running time
    of $O(n^2)$, ignoring multiplicative polylogarithmic factors, where
    $n$ is the number of nodes in the graph.

    """
    if len(G) == 0:
        return []
    if len(G) == 1:
        return [arbitrary_element(G)]
    v = arbitrary_element(G)
    hampath = hamiltonian_path(G.subgraph(set(G) - {v}))
    # Get the index of the first node in the path that does *not* have
    # an edge to `v`, then insert `v` before that node.
    index = index_satisfying(hampath, lambda u: v not in G[u])
    hampath.insert(index, v)
    return hampath


@py_random_state(1)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_tournament(n, seed=None):
    r"""Returns a random tournament graph on `n` nodes.

    Parameters
    ----------
    n : int
        The number of nodes in the returned graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : DiGraph
        A tournament on `n` nodes, with exactly one directed edge joining
        each pair of distinct nodes.

    Notes
    -----
    This algorithm adds, for each pair of distinct nodes, an edge with
    uniformly random orientation. In other words, `\binom{n}{2}` flips
    of an unbiased coin decide the orientations of the edges in the
    graph.

    """
    # Flip an unbiased coin for each pair of distinct nodes.
    coins = (seed.random() for i in range((n * (n - 1)) // 2))
    pairs = combinations(range(n), 2)
    edges = ((u, v) if r < 0.5 else (v, u) for (u, v), r in zip(pairs, coins))
    return nx.DiGraph(edges)


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
@nx._dispatchable
def score_sequence(G):
    """Returns the score sequence for the given tournament graph.

    The score sequence is the sorted list of the out-degrees of the
    nodes of the graph.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph representing a tournament.

    Returns
    -------
    list
        A sorted list of the out-degrees of the nodes of `G`.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 0), (1, 3), (0, 2), (0, 3), (2, 1), (3, 2)])
    >>> nx.is_tournament(G)
    True
    >>> nx.tournament.score_sequence(G)
    [1, 1, 2, 2]

    """
    return sorted(d for v, d in G.out_degree())


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
@nx._dispatchable(preserve_edge_attrs={"G": {"weight": 1}})
def tournament_matrix(G):
    r"""Returns the tournament matrix for the given tournament graph.

    This function requires SciPy.

    The *tournament matrix* of a tournament graph with edge set *E* is
    the matrix *T* defined by

    .. math::

       T_{i j} =
       \begin{cases}
       +1 & \text{if } (i, j) \in E \\
       -1 & \text{if } (j, i) \in E \\
       0 & \text{if } i == j.
       \end{cases}

    An equivalent definition is `T = A - A^T`, where *A* is the
    adjacency matrix of the graph `G`.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph representing a tournament.

    Returns
    -------
    SciPy sparse array
        The tournament matrix of the tournament graph `G`.

    Raises
    ------
    ImportError
        If SciPy is not available.

    """
    A = nx.adjacency_matrix(G)
    return A - A.T


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_reachable(G, s, t):
    """Decides whether there is a path from `s` to `t` in the
    tournament.

    This function is more theoretically efficient than the reachability
    checks than the shortest path algorithms in
    :mod:`networkx.algorithms.shortest_paths`.

    The given graph **must** be a tournament, otherwise this function's
    behavior is undefined.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph representing a tournament.

    s : node
        A node in the graph.

    t : node
        A node in the graph.

    Returns
    -------
    bool
        Whether there is a path from `s` to `t` in `G`.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 0), (1, 3), (1, 2), (2, 3), (2, 0), (3, 0)])
    >>> nx.is_tournament(G)
    True
    >>> nx.tournament.is_reachable(G, 1, 3)
    True
    >>> nx.tournament.is_reachable(G, 3, 2)
    False

    Notes
    -----
    Although this function is more theoretically efficient than the
    generic shortest path functions, a speedup requires the use of
    parallelism. Though it may in the future, the current implementation
    does not use parallelism, thus you may not see much of a speedup.

    This algorithm comes from [1].

    References
    ----------
    .. [1] Tantau, Till.
           "A note on the complexity of the reachability problem for
           tournaments."
           *Electronic Colloquium on Computational Complexity*. 2001.
           <http://eccc.hpi-web.de/report/2001/092/>
    """

    def two_neighborhood(G, v):
        """Returns the set of nodes at distance at most two from `v`.

        `G` must be a graph and `v` a node in that graph.

        The returned set includes the nodes at distance zero (that is,
        the node `v` itself), the nodes at distance one (that is, the
        out-neighbors of `v`), and the nodes at distance two.

        """
        return {
            x for x in G if x == v or x in G[v] or any(is_path(G, [v, z, x]) for z in G)
        }

    def is_closed(G, nodes):
        """Decides whether the given set of nodes is closed.

        A set *S* of nodes is *closed* if for each node *u* in the graph
        not in *S* and for each node *v* in *S*, there is an edge from
        *u* to *v*.

        """
        return all(v in G[u] for u in set(G) - nodes for v in nodes)

    neighborhoods = [two_neighborhood(G, v) for v in G]
    return all(not (is_closed(G, S) and s in S and t not in S) for S in neighborhoods)


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
@nx._dispatchable(name="tournament_is_strongly_connected")
def is_strongly_connected(G):
    """Decides whether the given tournament is strongly connected.

    This function is more theoretically efficient than the
    :func:`~networkx.algorithms.components.is_strongly_connected`
    function.

    The given graph **must** be a tournament, otherwise this function's
    behavior is undefined.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph representing a tournament.

    Returns
    -------
    bool
        Whether the tournament is strongly connected.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 0)])
    >>> nx.is_tournament(G)
    True
    >>> nx.tournament.is_strongly_connected(G)
    True
    >>> G.remove_edge(3, 0)
    >>> G.add_edge(0, 3)
    >>> nx.is_tournament(G)
    True
    >>> nx.tournament.is_strongly_connected(G)
    False

    Notes
    -----
    Although this function is more theoretically efficient than the
    generic strong connectivity function, a speedup requires the use of
    parallelism. Though it may in the future, the current implementation
    does not use parallelism, thus you may not see much of a speedup.

    This algorithm comes from [1].

    References
    ----------
    .. [1] Tantau, Till.
           "A note on the complexity of the reachability problem for
           tournaments."
           *Electronic Colloquium on Computational Complexity*. 2001.
           <http://eccc.hpi-web.de/report/2001/092/>

    """
    return all(is_reachable(G, u, v) for u in G for v in G)
<|endoftext|>"""Functions for computing sparsifiers of graphs."""

import math

import networkx as nx
from networkx.utils import not_implemented_for, py_random_state

__all__ = ["spanner"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state(3)
@nx._dispatchable(edge_attrs="weight", returns_graph=True)
def spanner(G, stretch, weight=None, seed=None):
    """Returns a spanner of the given graph with the given stretch.

    A spanner of a graph G = (V, E) with stretch t is a subgraph
    H = (V, E_S) such that E_S is a subset of E and the distance between
    any pair of nodes in H is at most t times the distance between the
    nodes in G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    stretch : float
        The stretch of the spanner.

    weight : object
        The edge attribute to use as distance.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    NetworkX graph
        A spanner of the given graph with the given stretch.

    Raises
    ------
    ValueError
        If a stretch less than 1 is given.

    Notes
    -----
    This function implements the spanner algorithm by Baswana and Sen,
    see [1].

    This algorithm is a randomized las vegas algorithm: The expected
    running time is O(km) where k = (stretch + 1) // 2 and m is the
    number of edges in G. The returned graph is always a spanner of the
    given graph with the specified stretch. For weighted graphs the
    number of edges in the spanner is O(k * n^(1 + 1 / k)) where k is
    defined as above and n is the number of nodes in G. For unweighted
    graphs the number of edges is O(n^(1 + 1 / k) + kn).

    References
    ----------
    [1] S. Baswana, S. Sen. A Simple and Linear Time Randomized
    Algorithm for Computing Sparse Spanners in Weighted Graphs.
    Random Struct. Algorithms 30(4): 532-563 (2007).
    """
    if stretch < 1:
        raise ValueError("stretch must be at least 1")

    k = (stretch + 1) // 2

    # initialize spanner H with empty edge set
    H = nx.empty_graph()
    H.add_nodes_from(G.nodes)

    # phase 1: forming the clusters
    # the residual graph has V' from the paper as its node set
    # and E' from the paper as its edge set
    residual_graph = _setup_residual_graph(G, weight)
    # clustering is a dictionary that maps nodes in a cluster to the
    # cluster center
    clustering = {v: v for v in G.nodes}
    sample_prob = math.pow(G.number_of_nodes(), -1 / k)
    size_limit = 2 * math.pow(G.number_of_nodes(), 1 + 1 / k)

    i = 0
    while i < k - 1:
        # step 1: sample centers
        sampled_centers = set()
        for center in set(clustering.values()):
            if seed.random() < sample_prob:
                sampled_centers.add(center)

        # combined loop for steps 2 and 3
        edges_to_add = set()
        edges_to_remove = set()
        new_clustering = {}
        for v in residual_graph.nodes:
            if clustering[v] in sampled_centers:
                continue

            # step 2: find neighboring (sampled) clusters and
            # lightest edges to them
            lightest_edge_neighbor, lightest_edge_weight = _lightest_edge_dicts(
                residual_graph, clustering, v
            )
            neighboring_sampled_centers = (
                set(lightest_edge_weight.keys()) & sampled_centers
            )

            # step 3: add edges to spanner
            if not neighboring_sampled_centers:
                # connect to each neighboring center via lightest edge
                for neighbor in lightest_edge_neighbor.values():
                    edges_to_add.add((v, neighbor))
                # remove all incident edges
                for neighbor in residual_graph.adj[v]:
                    edges_to_remove.add((v, neighbor))

            else:  # there is a neighboring sampled center
                closest_center = min(
                    neighboring_sampled_centers, key=lightest_edge_weight.get
                )
                closest_center_weight = lightest_edge_weight[closest_center]
                closest_center_neighbor = lightest_edge_neighbor[closest_center]

                edges_to_add.add((v, closest_center_neighbor))
                new_clustering[v] = closest_center

                # connect to centers with edge weight less than
                # closest_center_weight
                for center, edge_weight in lightest_edge_weight.items():
                    if edge_weight < closest_center_weight:
                        neighbor = lightest_edge_neighbor[center]
                        edges_to_add.add((v, neighbor))

                # remove edges to centers with edge weight less than
                # closest_center_weight
                for neighbor in residual_graph.adj[v]:
                    nbr_cluster = clustering[neighbor]
                    nbr_weight = lightest_edge_weight[nbr_cluster]
                    if (
                        nbr_cluster == closest_center
                        or nbr_weight < closest_center_weight
                    ):
                        edges_to_remove.add((v, neighbor))

        # check whether iteration added too many edges to spanner,
        # if so repeat
        if len(edges_to_add) > size_limit:
            # an iteration is repeated O(1) times on expectation
            continue

        # iteration succeeded
        i = i + 1

        # actually add edges to spanner
        for u, v in edges_to_add:
            _add_edge_to_spanner(H, residual_graph, u, v, weight)

        # actually delete edges from residual graph
        residual_graph.remove_edges_from(edges_to_remove)

        # copy old clustering data to new_clustering
        for node, center in clustering.items():
            if center in sampled_centers:
                new_clustering[node] = center
        clustering = new_clustering

        # step 4: remove intra-cluster edges
        for u in residual_graph.nodes:
            for v in list(residual_graph.adj[u]):
                if clustering[u] == clustering[v]:
                    residual_graph.remove_edge(u, v)

        # update residual graph node set
        for v in list(residual_graph.nodes):
            if v not in clustering:
                residual_graph.remove_node(v)

    # phase 2: vertex-cluster joining
    for v in residual_graph.nodes:
        lightest_edge_neighbor, _ = _lightest_edge_dicts(residual_graph, clustering, v)
        for neighbor in lightest_edge_neighbor.values():
            _add_edge_to_spanner(H, residual_graph, v, neighbor, weight)

    return H


def _setup_residual_graph(G, weight):
    """Setup residual graph as a copy of G with unique edges weights.

    The node set of the residual graph corresponds to the set V' from
    the Baswana-Sen paper and the edge set corresponds to the set E'
    from the paper.

    This function associates distinct weights to the edges of the
    residual graph (even for unweighted input graphs), as required by
    the algorithm.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    weight : object
        The edge attribute to use as distance.

    Returns
    -------
    NetworkX graph
        The residual graph used for the Baswana-Sen algorithm.
    """
    residual_graph = G.copy()

    # establish unique edge weights, even for unweighted graphs
    for u, v in G.edges():
        if not weight:
            residual_graph[u][v]["weight"] = (id(u), id(v))
        else:
            residual_graph[u][v]["weight"] = (G[u][v][weight], id(u), id(v))

    return residual_graph


def _lightest_edge_dicts(residual_graph, clustering, node):
    """Find the lightest edge to each cluster.

    Searches for the minimum-weight edge to each cluster adjacent to
    the given node.

    Parameters
    ----------
    residual_graph : NetworkX graph
        The residual graph used by the Baswana-Sen algorithm.

    clustering : dictionary
        The current clustering of the nodes.

    node : node
        The node from which the search originates.

    Returns
    -------
    lightest_edge_neighbor, lightest_edge_weight : dictionary, dictionary
        lightest_edge_neighbor is a dictionary that maps a center C to
        a node v in the corresponding cluster such that the edge from
        the given node to v is the lightest edge from the given node to
        any node in cluster. lightest_edge_weight maps a center C to the
        weight of the aforementioned edge.

    Notes
    -----
    If a cluster has no node that is adjacent to the given node in the
    residual graph then the center of the cluster is not a key in the
    returned dictionaries.
    """
    lightest_edge_neighbor = {}
    lightest_edge_weight = {}
    for neighbor in residual_graph.adj[node]:
        nbr_center = clustering[neighbor]
        weight = residual_graph[node][neighbor]["weight"]
        if (
            nbr_center not in lightest_edge_weight
            or weight < lightest_edge_weight[nbr_center]
        ):
            lightest_edge_neighbor[nbr_center] = neighbor
            lightest_edge_weight[nbr_center] = weight
    return lightest_edge_neighbor, lightest_edge_weight


def _add_edge_to_spanner(H, residual_graph, u, v, weight):
    """Add the edge {u, v} to the spanner H and take weight from
    the residual graph.

    Parameters
    ----------
    H : NetworkX graph
        The spanner under construction.

    residual_graph : NetworkX graph
        The residual graph used by the Baswana-Sen algorithm. The weight
        for the edge is taken from this graph.

    u : node
        One endpoint of the edge.

    v : node
        The other endpoint of the edge.

    weight : object
        The edge attribute to use as distance.
    """
    H.add_edge(u, v)
    if weight:
        H[u][v][weight] = residual_graph[u][v]["weight"][0]
<|endoftext|>"""
Threshold Graphs - Creation, manipulation and identification.
"""

from math import sqrt

import networkx as nx
from networkx.utils import py_random_state

__all__ = ["is_threshold_graph", "find_threshold_graph"]


@nx._dispatchable
def is_threshold_graph(G):
    """
    Returns `True` if `G` is a threshold graph.

    Parameters
    ----------
    G : NetworkX graph instance
        An instance of `Graph`, `DiGraph`, `MultiGraph` or `MultiDiGraph`

    Returns
    -------
    bool
        `True` if `G` is a threshold graph, `False` otherwise.

    Examples
    --------
    >>> from networkx.algorithms.threshold import is_threshold_graph
    >>> G = nx.path_graph(3)
    >>> is_threshold_graph(G)
    True
    >>> G = nx.barbell_graph(3, 3)
    >>> is_threshold_graph(G)
    False

    References
    ----------
    .. [1] Threshold graphs: https://en.wikipedia.org/wiki/Threshold_graph
    """
    return is_threshold_sequence([d for n, d in G.degree()])


def is_threshold_sequence(degree_sequence):
    """
    Returns True if the sequence is a threshold degree sequence.

    Uses the property that a threshold graph must be constructed by
    adding either dominating or isolated nodes. Thus, it can be
    deconstructed iteratively by removing a node of degree zero or a
    node that connects to the remaining nodes.  If this deconstruction
    fails then the sequence is not a threshold sequence.
    """
    ds = degree_sequence[:]  # get a copy so we don't destroy original
    ds.sort()
    while ds:
        if ds[0] == 0:  # if isolated node
            ds.pop(0)  # remove it
            continue
        if ds[-1] != len(ds) - 1:  # is the largest degree node dominating?
            return False  # no, not a threshold degree sequence
        ds.pop()  # yes, largest is the dominating node
        ds = [d - 1 for d in ds]  # remove it and decrement all degrees
    return True


def creation_sequence(degree_sequence, with_labels=False, compact=False):
    """
    Determines the creation sequence for the given threshold degree sequence.

    The creation sequence is a list of single characters 'd'
    or 'i': 'd' for dominating or 'i' for isolated vertices.
    Dominating vertices are connected to all vertices present when it
    is added.  The first node added is by convention 'd'.
    This list can be converted to a string if desired using "".join(cs)

    If with_labels==True:
    Returns a list of 2-tuples containing the vertex number
    and a character 'd' or 'i' which describes the type of vertex.

    If compact==True:
    Returns the creation sequence in a compact form that is the number
    of 'i's and 'd's alternating.
    Examples:
    [1,2,2,3] represents d,i,i,d,d,i,i,i
    [3,1,2] represents d,d,d,i,d,d

    Notice that the first number is the first vertex to be used for
    construction and so is always 'd'.

    with_labels and compact cannot both be True.

    Returns None if the sequence is not a threshold sequence
    """
    if with_labels and compact:
        raise ValueError("compact sequences cannot be labeled")

    # make an indexed copy
    if isinstance(degree_sequence, dict):  # labeled degree sequence
        ds = [[degree, label] for (label, degree) in degree_sequence.items()]
    else:
        ds = [[d, i] for i, d in enumerate(degree_sequence)]
    ds.sort()
    cs = []  # creation sequence
    while ds:
        if ds[0][0] == 0:  # isolated node
            (d, v) = ds.pop(0)
            if len(ds) > 0:  # make sure we start with a d
                cs.insert(0, (v, "i"))
            else:
                cs.insert(0, (v, "d"))
            continue
        if ds[-1][0] != len(ds) - 1:  # Not dominating node
            return None  # not a threshold degree sequence
        (d, v) = ds.pop()
        cs.insert(0, (v, "d"))
        ds = [[d[0] - 1, d[1]] for d in ds]  # decrement due to removing node

    if with_labels:
        return cs
    if compact:
        return make_compact(cs)
    return [v[1] for v in cs]  # not labeled


def make_compact(creation_sequence):
    """
    Returns the creation sequence in a compact form
    that is the number of 'i's and 'd's alternating.

    Examples
    --------
    >>> from networkx.algorithms.threshold import make_compact
    >>> make_compact(["d", "i", "i", "d", "d", "i", "i", "i"])
    [1, 2, 2, 3]
    >>> make_compact(["d", "d", "d", "i", "d", "d"])
    [3, 1, 2]

    Notice that the first number is the first vertex
    to be used for construction and so is always 'd'.

    Labeled creation sequences lose their labels in the
    compact representation.

    >>> make_compact([3, 1, 2])
    [3, 1, 2]
    """
    first = creation_sequence[0]
    if isinstance(first, str):  # creation sequence
        cs = creation_sequence[:]
    elif isinstance(first, tuple):  # labeled creation sequence
        cs = [s[1] for s in creation_sequence]
    elif isinstance(first, int):  # compact creation sequence
        return creation_sequence
    else:
        raise TypeError("Not a valid creation sequence type")

    ccs = []
    count = 1  # count the run lengths of d's or i's.
    for i in range(1, len(cs)):
        if cs[i] == cs[i - 1]:
            count += 1
        else:
            ccs.append(count)
            count = 1
    ccs.append(count)  # don't forget the last one
    return ccs


def uncompact(creation_sequence):
    """
    Converts a compact creation sequence for a threshold
    graph to a standard creation sequence (unlabeled).
    If the creation_sequence is already standard, return it.
    See creation_sequence.
    """
    first = creation_sequence[0]
    if isinstance(first, str):  # creation sequence
        return creation_sequence
    elif isinstance(first, tuple):  # labeled creation sequence
        return creation_sequence
    elif isinstance(first, int):  # compact creation sequence
        ccscopy = creation_sequence[:]
    else:
        raise TypeError("Not a valid creation sequence type")
    cs = []
    while ccscopy:
        cs.extend(ccscopy.pop(0) * ["d"])
        if ccscopy:
            cs.extend(ccscopy.pop(0) * ["i"])
    return cs


def creation_sequence_to_weights(creation_sequence):
    """
    Returns a list of node weights which create the threshold
    graph designated by the creation sequence.  The weights
    are scaled so that the threshold is 1.0.  The order of the
    nodes is the same as that in the creation sequence.
    """
    # Turn input sequence into a labeled creation sequence
    first = creation_sequence[0]
    if isinstance(first, str):  # creation sequence
        if isinstance(creation_sequence, list):
            wseq = creation_sequence[:]
        else:
            wseq = list(creation_sequence)  # string like 'ddidid'
    elif isinstance(first, tuple):  # labeled creation sequence
        wseq = [v[1] for v in creation_sequence]
    elif isinstance(first, int):  # compact creation sequence
        wseq = uncompact(creation_sequence)
    else:
        raise TypeError("Not a valid creation sequence type")
    # pass through twice--first backwards
    wseq.reverse()
    w = 0
    prev = "i"
    for j, s in enumerate(wseq):
        if s == "i":
            wseq[j] = w
            prev = s
        elif prev == "i":
            prev = s
            w += 1
    wseq.reverse()  # now pass through forwards
    for j, s in enumerate(wseq):
        if s == "d":
            wseq[j] = w
            prev = s
        elif prev == "d":
            prev = s
            w += 1
    # Now scale weights
    if prev == "d":
        w += 1
    wscale = 1 / w
    return [ww * wscale for ww in wseq]
    # return wseq


def weights_to_creation_sequence(
    weights, threshold=1, with_labels=False, compact=False
):
    """
    Returns a creation sequence for a threshold graph
    determined by the weights and threshold given as input.
    If the sum of two node weights is greater than the
    threshold value, an edge is created between these nodes.

    The creation sequence is a list of single characters 'd'
    or 'i': 'd' for dominating or 'i' for isolated vertices.
    Dominating vertices are connected to all vertices present
    when it is added.  The first node added is by convention 'd'.

    If with_labels==True:
    Returns a list of 2-tuples containing the vertex number
    and a character 'd' or 'i' which describes the type of vertex.

    If compact==True:
    Returns the creation sequence in a compact form that is the number
    of 'i's and 'd's alternating.
    Examples:
    [1,2,2,3] represents d,i,i,d,d,i,i,i
    [3,1,2] represents d,d,d,i,d,d

    Notice that the first number is the first vertex to be used for
    construction and so is always 'd'.

    with_labels and compact cannot both be True.
    """
    if with_labels and compact:
        raise ValueError("compact sequences cannot be labeled")

    # make an indexed copy
    if isinstance(weights, dict):  # labeled weights
        wseq = [[w, label] for (label, w) in weights.items()]
    else:
        wseq = [[w, i] for i, w in enumerate(weights)]
    wseq.sort()
    cs = []  # creation sequence
    cutoff = threshold - wseq[-1][0]
    while wseq:
        if wseq[0][0] < cutoff:  # isolated node
            (w, label) = wseq.pop(0)
            cs.append((label, "i"))
        else:
            (w, label) = wseq.pop()
            cs.append((label, "d"))
            cutoff = threshold - wseq[-1][0]
        if len(wseq) == 1:  # make sure we start with a d
            (w, label) = wseq.pop()
            cs.append((label, "d"))
    # put in correct order
    cs.reverse()

    if with_labels:
        return cs
    if compact:
        return make_compact(cs)
    return [v[1] for v in cs]  # not labeled


# Manipulating NetworkX.Graphs in context of threshold graphs
@nx._dispatchable(graphs=None, returns_graph=True)
def threshold_graph(creation_sequence, create_using=None):
    """
    Create a threshold graph from the creation sequence or compact
    creation_sequence.

    The input sequence can be a

    creation sequence (e.g. ['d','i','d','d','d','i'])
    labeled creation sequence (e.g. [(0,'d'),(2,'d'),(1,'i')])
    compact creation sequence (e.g. [2,1,1,2,0])

    Use cs=creation_sequence(degree_sequence,labeled=True)
    to convert a degree sequence to a creation sequence.

    Returns None if the sequence is not valid
    """
    # Turn input sequence into a labeled creation sequence
    first = creation_sequence[0]
    if isinstance(first, str):  # creation sequence
        ci = list(enumerate(creation_sequence))
    elif isinstance(first, tuple):  # labeled creation sequence
        ci = creation_sequence[:]
    elif isinstance(first, int):  # compact creation sequence
        cs = uncompact(creation_sequence)
        ci = list(enumerate(cs))
    else:
        print("not a valid creation sequence type")
        return None

    G = nx.empty_graph(0, create_using)
    if G.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    G.name = "Threshold Graph"

    # add nodes and edges
    # if type is 'i' just add nodea
    # if type is a d connect to everything previous
    while ci:
        (v, node_type) = ci.pop(0)
        if node_type == "d":  # dominating type, connect to all existing nodes
            # We use `for u in list(G):` instead of
            # `for u in G:` because we edit the graph `G` in
            # the loop. Hence using an iterator will result in
            # `RuntimeError: dictionary changed size during iteration`
            for u in list(G):
                G.add_edge(v, u)
        G.add_node(v)
    return G


@nx._dispatchable
def find_alternating_4_cycle(G):
    """
    Returns False if there aren't any alternating 4 cycles.
    Otherwise returns the cycle as [a,b,c,d] where (a,b)
    and (c,d) are edges and (a,c) and (b,d) are not.
    """
    for u, v in G.edges():
        for w in G.nodes():
            if not G.has_edge(u, w) and u != w:
                for x in G.neighbors(w):
                    if not G.has_edge(v, x) and v != x:
                        return [u, v, w, x]
    return False


@nx._dispatchable(returns_graph=True)
def find_threshold_graph(G, create_using=None):
    """
    Returns a threshold subgraph that is close to largest in `G`.

    The threshold graph will contain the largest degree node in G.

    Parameters
    ----------
    G : NetworkX graph instance
        An instance of `Graph`, or `MultiDiGraph`
    create_using : NetworkX graph class or `None` (default), optional
        Type of graph to use when constructing the threshold graph.
        If `None`, infer the appropriate graph type from the input.

    Returns
    -------
    graph :
        A graph instance representing the threshold graph

    Examples
    --------
    >>> from networkx.algorithms.threshold import find_threshold_graph
    >>> G = nx.barbell_graph(3, 3)
    >>> T = find_threshold_graph(G)
    >>> T.nodes  # may vary
    NodeView((7, 8, 5, 6))

    References
    ----------
    .. [1] Threshold graphs: https://en.wikipedia.org/wiki/Threshold_graph
    """
    return threshold_graph(find_creation_sequence(G), create_using)


@nx._dispatchable
def find_creation_sequence(G):
    """
    Find a threshold subgraph that is close to largest in G.
    Returns the labeled creation sequence of that threshold graph.
    """
    cs = []
    # get a local pointer to the working part of the graph
    H = G
    while H.order() > 0:
        # get new degree sequence on subgraph
        dsdict = dict(H.degree())
        ds = [(d, v) for v, d in dsdict.items()]
        ds.sort()
        # Update threshold graph nodes
        if ds[-1][0] == 0:  # all are isolated
            cs.extend(zip(dsdict, ["i"] * (len(ds) - 1) + ["d"]))
            break  # Done!
        # pull off isolated nodes
        while ds[0][0] == 0:
            (d, iso) = ds.pop(0)
            cs.append((iso, "i"))
        # find new biggest node
        (d, bigv) = ds.pop()
        # add edges of star to t_g
        cs.append((bigv, "d"))
        # form subgraph of neighbors of big node
        H = H.subgraph(H.neighbors(bigv))
    cs.reverse()
    return cs


# Properties of Threshold Graphs
def triangles(creation_sequence):
    """
    Compute number of triangles in the threshold graph with the
    given creation sequence.
    """
    # shortcut algorithm that doesn't require computing number
    # of triangles at each node.
    cs = creation_sequence  # alias
    dr = cs.count("d")  # number of d's in sequence
    ntri = dr * (dr - 1) * (dr - 2) / 6  # number of triangles in clique of nd d's
    # now add dr choose 2 triangles for every 'i' in sequence where
    # dr is the number of d's to the right of the current i
    for i, typ in enumerate(cs):
        if typ == "i":
            ntri += dr * (dr - 1) / 2
        else:
            dr -= 1
    return ntri


def triangle_sequence(creation_sequence):
    """
    Return triangle sequence for the given threshold graph creation sequence.

    """
    cs = creation_sequence
    seq = []
    dr = cs.count("d")  # number of d's to the right of the current pos
    dcur = (dr - 1) * (dr - 2) // 2  # number of triangles through a node of clique dr
    irun = 0  # number of i's in the last run
    drun = 0  # number of d's in the last run
    for i, sym in enumerate(cs):
        if sym == "d":
            drun += 1
            tri = dcur + (dr - 1) * irun  # new triangles at this d
        else:  # cs[i]="i":
            if prevsym == "d":  # new string of i's
                dcur += (dr - 1) * irun  # accumulate shared shortest paths
                irun = 0  # reset i run counter
                dr -= drun  # reduce number of d's to right
                drun = 0  # reset d run counter
            irun += 1
            tri = dr * (dr - 1) // 2  # new triangles at this i
        seq.append(tri)
        prevsym = sym
    return seq


def cluster_sequence(creation_sequence):
    """
    Return cluster sequence for the given threshold graph creation sequence.
    """
    triseq = triangle_sequence(creation_sequence)
    degseq = degree_sequence(creation_sequence)
    cseq = []
    for i, deg in enumerate(degseq):
        tri = triseq[i]
        if deg <= 1:  # isolated vertex or single pair gets cc 0
            cseq.append(0)
            continue
        max_size = (deg * (deg - 1)) // 2
        cseq.append(tri / max_size)
    return cseq


def degree_sequence(creation_sequence):
    """
    Return degree sequence for the threshold graph with the given
    creation sequence
    """
    cs = creation_sequence  # alias
    seq = []
    rd = cs.count("d")  # number of d to the right
    for i, sym in enumerate(cs):
        if sym == "d":
            rd -= 1
            seq.append(rd + i)
        else:
            seq.append(rd)
    return seq


def density(creation_sequence):
    """
    Return the density of the graph with this creation_sequence.
    The density is the fraction of possible edges present.
    """
    N = len(creation_sequence)
    two_size = sum(degree_sequence(creation_sequence))
    two_possible = N * (N - 1)
    den = two_size / two_possible
    return den


def degree_correlation(creation_sequence):
    """
    Return the degree-degree correlation over all edges.
    """
    cs = creation_sequence
    s1 = 0  # deg_i*deg_j
    s2 = 0  # deg_i^2+deg_j^2
    s3 = 0  # deg_i+deg_j
    m = 0  # number of edges
    rd = cs.count("d")  # number of d nodes to the right
    rdi = [i for i, sym in enumerate(cs) if sym == "d"]  # index of "d"s
    ds = degree_sequence(cs)
    for i, sym in enumerate(cs):
        if sym == "d":
            if i != rdi[0]:
                print("Logic error in degree_correlation", i, rdi)
                raise ValueError
            rdi.pop(0)
        degi = ds[i]
        for dj in rdi:
            degj = ds[dj]
            s1 += degj * degi
            s2 += degi**2 + degj**2
            s3 += degi + degj
            m += 1
    denom = 2 * m * s2 - s3 * s3
    numer = 4 * m * s1 - s3 * s3
    if denom == 0:
        if numer == 0:
            return 1
        raise ValueError(f"Zero Denominator but Numerator is {numer}")
    return numer / denom


def shortest_path(creation_sequence, u, v):
    """
    Find the shortest path between u and v in a
    threshold graph G with the given creation_sequence.

    For an unlabeled creation_sequence, the vertices
    u and v must be integers in (0,len(sequence)) referring
    to the position of the desired vertices in the sequence.

    For a labeled creation_sequence, u and v are labels of vertices.

    Use cs=creation_sequence(degree_sequence,with_labels=True)
    to convert a degree sequence to a creation sequence.

    Returns a list of vertices from u to v.
    Example: if they are neighbors, it returns [u,v]
    """
    # Turn input sequence into a labeled creation sequence
    first = creation_sequence[0]
    if isinstance(first, str):  # creation sequence
        cs = [(i, creation_sequence[i]) for i in range(len(creation_sequence))]
    elif isinstance(first, tuple):  # labeled creation sequence
        cs = creation_sequence[:]
    elif isinstance(first, int):  # compact creation sequence
        ci = uncompact(creation_sequence)
        cs = [(i, ci[i]) for i in range(len(ci))]
    else:
        raise TypeError("Not a valid creation sequence type")

    verts = [s[0] for s in cs]
    if v not in verts:
        raise ValueError(f"Vertex {v} not in graph from creation_sequence")
    if u not in verts:
        raise ValueError(f"Vertex {u} not in graph from creation_sequence")
    # Done checking
    if u == v:
        return [u]

    uindex = verts.index(u)
    vindex = verts.index(v)
    bigind = max(uindex, vindex)
    if cs[bigind][1] == "d":
        return [u, v]
    # must be that cs[bigind][1]=='i'
    cs = cs[bigind:]
    while cs:
        vert = cs.pop()
        if vert[1] == "d":
            return [u, vert[0], v]
    # All after u are type 'i' so no connection
    return -1


def shortest_path_length(creation_sequence, i):
    """
    Return the shortest path length from indicated node to
    every other node for the threshold graph with the given
    creation sequence.
    Node is indicated by index i in creation_sequence unless
    creation_sequence is labeled in which case, i is taken to
    be the label of the node.

    Paths lengths in threshold graphs are at most 2.
    Length to unreachable nodes is set to -1.
    """
    # Turn input sequence into a labeled creation sequence
    first = creation_sequence[0]
    if isinstance(first, str):  # creation sequence
        if isinstance(creation_sequence, list):
            cs = creation_sequence[:]
        else:
            cs = list(creation_sequence)
    elif isinstance(first, tuple):  # labeled creation sequence
        cs = [v[1] for v in creation_sequence]
        i = [v[0] for v in creation_sequence].index(i)
    elif isinstance(first, int):  # compact creation sequence
        cs = uncompact(creation_sequence)
    else:
        raise TypeError("Not a valid creation sequence type")

    # Compute
    N = len(cs)
    spl = [2] * N  # length 2 to every node
    spl[i] = 0  # except self which is 0
    # 1 for all d's to the right
    for j in range(i + 1, N):
        if cs[j] == "d":
            spl[j] = 1
    if cs[i] == "d":  # 1 for all nodes to the left
        for j in range(i):
            spl[j] = 1
    # and -1 for any trailing i to indicate unreachable
    for j in range(N - 1, 0, -1):
        if cs[j] == "d":
            break
        spl[j] = -1
    return spl


def betweenness_sequence(creation_sequence, normalized=True):
    """
    Return betweenness for the threshold graph with the given creation
    sequence.  The result is unscaled.  To scale the values
    to the interval [0,1] divide by (n-1)*(n-2).
    """
    cs = creation_sequence
    seq = []  # betweenness
    lastchar = "d"  # first node is always a 'd'
    dr = float(cs.count("d"))  # number of d's to the right of current pos
    irun = 0  # number of i's in the last run
    drun = 0  # number of d's in the last run
    dlast = 0.0  # betweenness of last d
    for i, c in enumerate(cs):
        if c == "d":  # cs[i]=="d":
            # betweenness = amt shared with earlier d's and i's
            #             + new isolated nodes covered
            #             + new paths to all previous nodes
            b = dlast + (irun - 1) * irun / dr + 2 * irun * (i - drun - irun) / dr
            drun += 1  # update counter
        else:  # cs[i]="i":
            if lastchar == "d":  # if this is a new run of i's
                dlast = b  # accumulate betweenness
                dr -= drun  # update number of d's to the right
                drun = 0  # reset d counter
                irun = 0  # reset i counter
            b = 0  # isolated nodes have zero betweenness
            irun += 1  # add another i to the run
        seq.append(float(b))
        lastchar = c

    # normalize by the number of possible shortest paths
    if normalized:
        order = len(cs)
        scale = 1.0 / ((order - 1) * (order - 2))
        seq = [s * scale for s in seq]

    return seq


def eigenvectors(creation_sequence):
    """
    Return a 2-tuple of Laplacian eigenvalues and eigenvectors
    for the threshold network with creation_sequence.
    The first value is a list of eigenvalues.
    The second value is a list of eigenvectors.
    The lists are in the same order so corresponding eigenvectors
    and eigenvalues are in the same position in the two lists.

    Notice that the order of the eigenvalues returned by eigenvalues(cs)
    may not correspond to the order of these eigenvectors.
    """
    ccs = make_compact(creation_sequence)
    N = sum(ccs)
    vec = [0] * N
    val = vec[:]
    # get number of type d nodes to the right (all for first node)
    dr = sum(ccs[::2])

    nn = ccs[0]
    vec[0] = [1.0 / sqrt(N)] * N
    val[0] = 0
    e = dr
    dr -= nn
    type_d = True
    i = 1
    dd = 1
    while dd < nn:
        scale = 1.0 / sqrt(dd * dd + i)
        vec[i] = i * [-scale] + [dd * scale] + [0] * (N - i - 1)
        val[i] = e
        i += 1
        dd += 1
    if len(ccs) == 1:
        return (val, vec)
    for nn in ccs[1:]:
        scale = 1.0 / sqrt(nn * i * (i + nn))
        vec[i] = i * [-nn * scale] + nn * [i * scale] + [0] * (N - i - nn)
        # find eigenvalue
        type_d = not type_d
        if type_d:
            e = i + dr
            dr -= nn
        else:
            e = dr
        val[i] = e
        st = i
        i += 1
        dd = 1
        while dd < nn:
            scale = 1.0 / sqrt(i - st + dd * dd)
            vec[i] = [0] * st + (i - st) * [-scale] + [dd * scale] + [0] * (N - i - 1)
            val[i] = e
            i += 1
            dd += 1
    return (val, vec)


def spectral_projection(u, eigenpairs):
    """
    Returns the coefficients of each eigenvector
    in a projection of the vector u onto the normalized
    eigenvectors which are contained in eigenpairs.

    eigenpairs should be a list of two objects.  The
    first is a list of eigenvalues and the second a list
    of eigenvectors.  The eigenvectors should be lists.

    There's not a lot of error checking on lengths of
    arrays, etc. so be careful.
    """
    coeff = []
    evect = eigenpairs[1]
    for ev in evect:
        c = sum(evv * uv for (evv, uv) in zip(ev, u))
        coeff.append(c)
    return coeff


def eigenvalues(creation_sequence):
    """
    Return sequence of eigenvalues of the Laplacian of the threshold
    graph for the given creation_sequence.

    Based on the Ferrer's diagram method.  The spectrum is integral
    and is the conjugate of the degree sequence.

    See::

      @Article{degree-merris-1994,
       author = {Russel Merris},
       title = {Degree maximal graphs are Laplacian integral},
       journal = {Linear Algebra Appl.},
       year = {1994},
       volume = {199},
       pages = {381--389},
      }

    """
    degseq = degree_sequence(creation_sequence)
    degseq.sort()
    eiglist = []  # zero is always one eigenvalue
    eig = 0
    row = len(degseq)
    bigdeg = degseq.pop()
    while row:
        if bigdeg < row:
            eiglist.append(eig)
            row -= 1
        else:
            eig += 1
            if degseq:
                bigdeg = degseq.pop()
            else:
                bigdeg = 0
    return eiglist


# Threshold graph creation routines


@py_random_state(2)
def random_threshold_sequence(n, p, seed=None):
    """
    Create a random threshold sequence of size n.
    A creation sequence is built by randomly choosing d's with
    probability p and i's with probability 1-p.

    s=nx.random_threshold_sequence(10,0.5)

    returns a threshold sequence of length 10 with equal
    probably of an i or a d at each position.

    A "random" threshold graph can be built with

    G=nx.threshold_graph(s)

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0,1]")

    cs = ["d"]  # threshold sequences always start with a d
    for i in range(1, n):
        if seed.random() < p:
            cs.append("d")
        else:
            cs.append("i")
    return cs


# maybe *_d_threshold_sequence routines should
# be (or be called from) a single routine with a more descriptive name
# and a keyword parameter?
def right_d_threshold_sequence(n, m):
    """
    Create a skewed threshold graph with a given number
    of vertices (n) and a given number of edges (m).

    The routine returns an unlabeled creation sequence
    for the threshold graph.

    FIXME: describe algorithm

    """
    cs = ["d"] + ["i"] * (n - 1)  # create sequence with n insolated nodes

    #  m <n : not enough edges, make disconnected
    if m < n:
        cs[m] = "d"
        return cs

    # too many edges
    if m > n * (n - 1) / 2:
        raise ValueError("Too many edges for this many nodes.")

    # connected case m >n-1
    ind = n - 1
    sum = n - 1
    while sum < m:
        cs[ind] = "d"
        ind -= 1
        sum += ind
    ind = m - (sum - ind)
    cs[ind] = "d"
    return cs


def left_d_threshold_sequence(n, m):
    """
    Create a skewed threshold graph with a given number
    of vertices (n) and a given number of edges (m).

    The routine returns an unlabeled creation sequence
    for the threshold graph.

    FIXME: describe algorithm

    """
    cs = ["d"] + ["i"] * (n - 1)  # create sequence with n insolated nodes

    #  m <n : not enough edges, make disconnected
    if m < n:
        cs[m] = "d"
        return cs

    # too many edges
    if m > n * (n - 1) / 2:
        raise ValueError("Too many edges for this many nodes.")

    # Connected case when M>N-1
    cs[n - 1] = "d"
    sum = n - 1
    ind = 1
    while sum < m:
        cs[ind] = "d"
        sum += ind
        ind += 1
    if sum > m:  # be sure not to change the first vertex
        cs[sum - m] = "i"
    return cs


@py_random_state(3)
def swap_d(cs, p_split=1.0, p_combine=1.0, seed=None):
    """
    Perform a "swap" operation on a threshold sequence.

    The swap preserves the number of nodes and edges
    in the graph for the given sequence.
    The resulting sequence is still a threshold sequence.

    Perform one split and one combine operation on the
    'd's of a creation sequence for a threshold graph.
    This operation maintains the number of nodes and edges
    in the graph, but shifts the edges from node to node
    maintaining the threshold quality of the graph.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
    # preprocess the creation sequence
    dlist = [i for (i, node_type) in enumerate(cs[1:-1]) if node_type == "d"]
    # split
    if seed.random() < p_split:
        choice = seed.choice(dlist)
        split_to = seed.choice(range(choice))
        flip_side = choice - split_to
        if split_to != flip_side and cs[split_to] == "i" and cs[flip_side] == "i":
            cs[choice] = "i"
            cs[split_to] = "d"
            cs[flip_side] = "d"
            dlist.remove(choice)
            # don't add or combine may reverse this action
            # dlist.extend([split_to,flip_side])
    #            print >>sys.stderr,"split at %s to %s and %s"%(choice,split_to,flip_side)
    # combine
    if seed.random() < p_combine and dlist:
        first_choice = seed.choice(dlist)
        second_choice = seed.choice(dlist)
        target = first_choice + second_choice
        if target >= len(cs) or cs[target] == "d" or first_choice == second_choice:
            return cs
        # OK to combine
        cs[first_choice] = "i"
        cs[second_choice] = "i"
        cs[target] = "d"
    #        print >>sys.stderr,"combine %s and %s to make %s."%(first_choice,second_choice,target)

    return cs
<|endoftext|>"""Algorithms for finding the lowest common ancestor of trees and DAGs."""

from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import combinations_with_replacement

import networkx as nx
from networkx.utils import UnionFind, arbitrary_element, not_implemented_for

__all__ = [
    "all_pairs_lowest_common_ancestor",
    "tree_all_pairs_lowest_common_ancestor",
    "lowest_common_ancestor",
]


@not_implemented_for("undirected")
@nx._dispatchable
def all_pairs_lowest_common_ancestor(G, pairs=None):
    """Return the lowest common ancestor of all pairs or the provided pairs

    Parameters
    ----------
    G : NetworkX directed graph

    pairs : iterable of pairs of nodes, optional (default: all pairs)
        The pairs of nodes of interest.
        If None, will find the LCA of all pairs of nodes.

    Yields
    ------
    ((node1, node2), lca) : 2-tuple
        Where lca is least common ancestor of node1 and node2.
        Note that for the default case, the order of the node pair is not considered,
        e.g. you will not get both ``(a, b)`` and ``(b, a)``

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is null.
    NetworkXError
        If `G` is not a DAG.

    Examples
    --------
    The default behavior is to yield the lowest common ancestor for all
    possible combinations of nodes in `G`, including self-pairings:

    >>> G = nx.DiGraph([(0, 1), (0, 3), (1, 2)])
    >>> dict(nx.all_pairs_lowest_common_ancestor(G))
    {(0, 0): 0, (0, 1): 0, (0, 3): 0, (0, 2): 0, (1, 1): 1, (1, 3): 0, (1, 2): 1, (3, 3): 3, (3, 2): 0, (2, 2): 2}

    The pairs argument can be used to limit the output to only the
    specified node pairings:

    >>> dict(nx.all_pairs_lowest_common_ancestor(G, pairs=[(1, 2), (2, 3)]))
    {(1, 2): 1, (2, 3): 0}

    Notes
    -----
    Only defined on non-null directed acyclic graphs.

    See Also
    --------
    lowest_common_ancestor
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError("LCA only defined on directed acyclic graphs.")
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")

    if pairs is None:
        pairs = combinations_with_replacement(G, 2)
    else:
        # Convert iterator to iterable, if necessary. Trim duplicates.
        pairs = dict.fromkeys(pairs)
        # Verify that each of the nodes in the provided pairs is in G
        nodeset = set(G)
        for pair in pairs:
            if set(pair) - nodeset:
                raise nx.NodeNotFound(
                    f"Node(s) {set(pair) - nodeset} from pair {pair} not in G."
                )

    # Once input validation is done, construct the generator
    def generate_lca_from_pairs(G, pairs):
        ancestor_cache = {}

        for v, w in pairs:
            if v not in ancestor_cache:
                ancestor_cache[v] = nx.ancestors(G, v)
                ancestor_cache[v].add(v)
            if w not in ancestor_cache:
                ancestor_cache[w] = nx.ancestors(G, w)
                ancestor_cache[w].add(w)

            common_ancestors = ancestor_cache[v] & ancestor_cache[w]

            if common_ancestors:
                common_ancestor = next(iter(common_ancestors))
                while True:
                    successor = None
                    for lower_ancestor in G.successors(common_ancestor):
                        if lower_ancestor in common_ancestors:
                            successor = lower_ancestor
                            break
                    if successor is None:
                        break
                    common_ancestor = successor
                yield ((v, w), common_ancestor)

    return generate_lca_from_pairs(G, pairs)


@not_implemented_for("undirected")
@nx._dispatchable
def lowest_common_ancestor(G, node1, node2, default=None):
    """Compute the lowest common ancestor of the given pair of nodes.

    Parameters
    ----------
    G : NetworkX directed graph

    node1, node2 : nodes in the graph.

    default : object
        Returned if no common ancestor between `node1` and `node2`

    Returns
    -------
    The lowest common ancestor of node1 and node2,
    or default if they have no common ancestors.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> nx.add_path(G, (0, 1, 2, 3))
    >>> nx.add_path(G, (0, 4, 3))
    >>> nx.lowest_common_ancestor(G, 2, 4)
    0

    See Also
    --------
    all_pairs_lowest_common_ancestor"""

    ans = list(all_pairs_lowest_common_ancestor(G, pairs=[(node1, node2)]))
    if ans:
        assert len(ans) == 1
        return ans[0][1]
    return default


@not_implemented_for("undirected")
@nx._dispatchable
def tree_all_pairs_lowest_common_ancestor(G, root=None, pairs=None):
    r"""Yield the lowest common ancestor for sets of pairs in a tree.

    Parameters
    ----------
    G : NetworkX directed graph (must be a tree)

    root : node, optional (default: None)
        The root of the subtree to operate on.
        If None, assume the entire graph has exactly one source and use that.

    pairs : iterable or iterator of pairs of nodes, optional (default: None)
        The pairs of interest. If None, Defaults to all pairs of nodes
        under `root` that have a lowest common ancestor.

    Returns
    -------
    lcas : generator of tuples `((u, v), lca)` where `u` and `v` are nodes
        in `pairs` and `lca` is their lowest common ancestor.

    Examples
    --------
    >>> import pprint
    >>> G = nx.DiGraph([(1, 3), (2, 4), (1, 2)])
    >>> pprint.pprint(dict(nx.tree_all_pairs_lowest_common_ancestor(G)))
    {(1, 1): 1,
     (2, 1): 1,
     (2, 2): 2,
     (3, 1): 1,
     (3, 2): 1,
     (3, 3): 3,
     (3, 4): 1,
     (4, 1): 1,
     (4, 2): 2,
     (4, 4): 4}

    We can also use `pairs` argument to specify the pairs of nodes for which we
    want to compute lowest common ancestors. Here is an example:

    >>> dict(nx.tree_all_pairs_lowest_common_ancestor(G, pairs=[(1, 4), (2, 3)]))
    {(2, 3): 1, (1, 4): 1}

    Notes
    -----
    Only defined on non-null trees represented with directed edges from
    parents to children. Uses Tarjan's off-line lowest-common-ancestors
    algorithm. Runs in time $O(4 \times (V + E + P))$ time, where 4 is the largest
    value of the inverse Ackermann function likely to ever come up in actual
    use, and $P$ is the number of pairs requested (or $V^2$ if all are needed).

    Tarjan, R. E. (1979), "Applications of path compression on balanced trees",
    Journal of the ACM 26 (4): 690-715, doi:10.1145/322154.322161.

    See Also
    --------
    all_pairs_lowest_common_ancestor: similar routine for general DAGs
    lowest_common_ancestor: just a single pair for general DAGs
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")

    # Index pairs of interest for efficient lookup from either side.
    if pairs is not None:
        pair_dict = defaultdict(set)
        # See note on all_pairs_lowest_common_ancestor.
        if not isinstance(pairs, Mapping | Set):
            pairs = set(pairs)
        for u, v in pairs:
            for n in (u, v):
                if n not in G:
                    msg = f"The node {str(n)} is not in the digraph."
                    raise nx.NodeNotFound(msg)
            pair_dict[u].add(v)
            pair_dict[v].add(u)

    # If root is not specified, find the exactly one node with in degree 0 and
    # use it. Raise an error if none are found, or more than one is. Also check
    # for any nodes with in degree larger than 1, which would imply G is not a
    # tree.
    if root is None:
        for n, deg in G.in_degree:
            if deg == 0:
                if root is not None:
                    msg = "No root specified and tree has multiple sources."
                    raise nx.NetworkXError(msg)
                root = n
            # checking deg>1 is not sufficient for MultiDiGraphs
            elif deg > 1 and len(G.pred[n]) > 1:
                msg = "Tree LCA only defined on trees; use DAG routine."
                raise nx.NetworkXError(msg)
    if root is None:
        raise nx.NetworkXError("Graph contains a cycle.")

    # Iterative implementation of Tarjan's offline lca algorithm
    # as described in CLRS on page 521 (2nd edition)/page 584 (3rd edition)
    uf = UnionFind()
    ancestors = {}
    for node in G:
        ancestors[node] = uf[node]

    colors = defaultdict(bool)
    for node in nx.dfs_postorder_nodes(G, root):
        colors[node] = True
        for v in pair_dict[node] if pairs is not None else G:
            if colors[v]:
                # If the user requested both directions of a pair, give it.
                # Otherwise, just give one.
                if pairs is not None and (node, v) in pairs:
                    yield (node, v), ancestors[uf[v]]
                if pairs is None or (v, node) in pairs:
                    yield (v, node), ancestors[uf[v]]
        if node != root:
            parent = arbitrary_element(G.pred[node])
            uf.union(parent, node)
            ancestors[uf[parent]] = parent
<|endoftext|>"""Functions for computing dominating sets in a graph."""

from itertools import chain

import networkx as nx
from networkx.utils import arbitrary_element

__all__ = ["dominating_set", "is_dominating_set"]


@nx._dispatchable
def dominating_set(G, start_with=None):
    r"""Finds a dominating set for the graph G.

    A *dominating set* for a graph with node set *V* is a subset *D* of
    *V* such that every node not in *D* is adjacent to at least one
    member of *D* [1]_.

    Parameters
    ----------
    G : NetworkX graph

    start_with : node (default=None)
        Node to use as a starting point for the algorithm.

    Returns
    -------
    D : set
        A dominating set for G.

    Notes
    -----
    This function is an implementation of algorithm 7 in [2]_ which
    finds some dominating set, not necessarily the smallest one.

    See also
    --------
    is_dominating_set

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dominating_set

    .. [2] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    all_nodes = set(G)
    if start_with is None:
        start_with = arbitrary_element(all_nodes)
    if start_with not in G:
        raise nx.NetworkXError(f"node {start_with} is not in G")
    dominating_set = {start_with}
    dominated_nodes = set(G[start_with])
    remaining_nodes = all_nodes - dominated_nodes - dominating_set
    while remaining_nodes:
        # Choose an arbitrary node and determine its undominated neighbors.
        v = remaining_nodes.pop()
        undominated_nbrs = set(G[v]) - dominating_set
        # Add the node to the dominating set and the neighbors to the
        # dominated set. Finally, remove all of those nodes from the set
        # of remaining nodes.
        dominating_set.add(v)
        dominated_nodes |= undominated_nbrs
        remaining_nodes -= undominated_nbrs
    return dominating_set


@nx._dispatchable
def is_dominating_set(G, nbunch):
    """Checks if `nbunch` is a dominating set for `G`.

    A *dominating set* for a graph with node set *V* is a subset *D* of
    *V* such that every node not in *D* is adjacent to at least one
    member of *D* [1]_.

    Parameters
    ----------
    G : NetworkX graph

    nbunch : iterable
        An iterable of nodes in the graph `G`.

    See also
    --------
    dominating_set

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dominating_set

    """
    testset = {n for n in nbunch if n in G}
    nbrs = set(chain.from_iterable(G[n] for n in testset))
    return len(set(G) - testset - nbrs) == 0
<|endoftext|>"""Provides functions for computing the efficiency of nodes and graphs."""

import networkx as nx
from networkx.exception import NetworkXNoPath

from ..utils import not_implemented_for

__all__ = ["efficiency", "local_efficiency", "global_efficiency"]


@not_implemented_for("directed")
@nx._dispatchable
def efficiency(G, u, v):
    """Returns the efficiency of a pair of nodes in a graph.

    The *efficiency* of a pair of nodes is the multiplicative inverse of the
    shortest path distance between the nodes [1]_. Returns 0 if no path
    between nodes.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        An undirected graph for which to compute the average local efficiency.
    u, v : node
        Nodes in the graph ``G``.

    Returns
    -------
    float
        Multiplicative inverse of the shortest path distance between the nodes.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    >>> nx.efficiency(G, 2, 3)  # this gives efficiency for node 2 and 3
    0.5

    Notes
    -----
    Edge weights are ignored when computing the shortest path distances.

    See also
    --------
    local_efficiency
    global_efficiency

    References
    ----------
    .. [1] Latora, Vito, and Massimo Marchiori.
           "Efficient behavior of small-world networks."
           *Physical Review Letters* 87.19 (2001): 198701.
           <https://doi.org/10.1103/PhysRevLett.87.198701>

    """
    try:
        eff = 1 / nx.shortest_path_length(G, u, v)
    except NetworkXNoPath:
        eff = 0
    return eff


@not_implemented_for("directed")
@nx._dispatchable
def global_efficiency(G):
    """Returns the average global efficiency of the graph.

    The *efficiency* of a pair of nodes in a graph is the multiplicative
    inverse of the shortest path distance between the nodes. The *average
    global efficiency* of a graph is the average efficiency of all pairs of
    nodes [1]_.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        An undirected graph for which to compute the average global efficiency.

    Returns
    -------
    float
        The average global efficiency of the graph.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    >>> round(nx.global_efficiency(G), 12)
    0.916666666667

    Notes
    -----
    Edge weights are ignored when computing the shortest path distances.

    See also
    --------
    local_efficiency

    References
    ----------
    .. [1] Latora, Vito, and Massimo Marchiori.
           "Efficient behavior of small-world networks."
           *Physical Review Letters* 87.19 (2001): 198701.
           <https://doi.org/10.1103/PhysRevLett.87.198701>

    """
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        lengths = nx.all_pairs_shortest_path_length(G)
        g_eff = 0
        for source, targets in lengths:
            for target, distance in targets.items():
                if distance > 0:
                    g_eff += 1 / distance
        g_eff /= denom
        # g_eff = sum(1 / d for s, tgts in lengths
        #                   for t, d in tgts.items() if d > 0) / denom
    else:
        g_eff = 0
    # TODO This can be made more efficient by computing all pairs shortest
    # path lengths in parallel.
    return g_eff


@not_implemented_for("directed")
@nx._dispatchable
def local_efficiency(G):
    """Returns the average local efficiency of the graph.

    The *efficiency* of a pair of nodes in a graph is the multiplicative
    inverse of the shortest path distance between the nodes. The *local
    efficiency* of a node in the graph is the average global efficiency of the
    subgraph induced by the neighbors of the node. The *average local
    efficiency* is the average of the local efficiencies of each node [1]_.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        An undirected graph for which to compute the average local efficiency.

    Returns
    -------
    float
        The average local efficiency of the graph.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    >>> nx.local_efficiency(G)
    0.9166666666666667

    Notes
    -----
    Edge weights are ignored when computing the shortest path distances.

    See also
    --------
    global_efficiency

    References
    ----------
    .. [1] Latora, Vito, and Massimo Marchiori.
           "Efficient behavior of small-world networks."
           *Physical Review Letters* 87.19 (2001): 198701.
           <https://doi.org/10.1103/PhysRevLett.87.198701>

    """
    efficiency_list = (global_efficiency(G.subgraph(G[v])) for v in G)
    return sum(efficiency_list) / len(G)
<|endoftext|>"""Algorithms for directed acyclic graphs (DAGs).

Note that most of these functions are only guaranteed to work for DAGs.
In general, these functions do not check for acyclic-ness, so it is up
to the user to check for that.
"""

import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd

import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise

__all__ = [
    "descendants",
    "ancestors",
    "topological_sort",
    "lexicographical_topological_sort",
    "all_topological_sorts",
    "topological_generations",
    "is_directed_acyclic_graph",
    "is_aperiodic",
    "transitive_closure",
    "transitive_closure_dag",
    "transitive_reduction",
    "antichains",
    "dag_longest_path",
    "dag_longest_path_length",
    "dag_to_branching",
    "compute_v_structures",
]

chaini = chain.from_iterable


@nx._dispatchable
def descendants(G, source):
    """Returns all nodes reachable from `source` in `G`.

    Parameters
    ----------
    G : NetworkX Graph
    source : node in `G`

    Returns
    -------
    set()
        The descendants of `source` in `G`

    Raises
    ------
    NetworkXError
        If node `source` is not in `G`.

    Examples
    --------
    >>> DG = nx.path_graph(5, create_using=nx.DiGraph)
    >>> sorted(nx.descendants(DG, 2))
    [3, 4]

    The `source` node is not a descendant of itself, but can be included manually:

    >>> sorted(nx.descendants(DG, 2) | {2})
    [2, 3, 4]

    See also
    --------
    ancestors
    """
    return {child for parent, child in nx.bfs_edges(G, source)}


@nx._dispatchable
def ancestors(G, source):
    """Returns all nodes having a path to `source` in `G`.

    Parameters
    ----------
    G : NetworkX Graph
    source : node in `G`

    Returns
    -------
    set()
        The ancestors of `source` in `G`

    Raises
    ------
    NetworkXError
        If node `source` is not in `G`.

    Examples
    --------
    >>> DG = nx.path_graph(5, create_using=nx.DiGraph)
    >>> sorted(nx.ancestors(DG, 2))
    [0, 1]

    The `source` node is not an ancestor of itself, but can be included manually:

    >>> sorted(nx.ancestors(DG, 2) | {2})
    [0, 1, 2]

    See also
    --------
    descendants
    """
    return {child for parent, child in nx.bfs_edges(G, source, reverse=True)}


@nx._dispatchable
def has_cycle(G):
    """Decides whether the directed graph has a cycle."""
    try:
        # Feed the entire iterator into a zero-length deque.
        deque(topological_sort(G), maxlen=0)
    except nx.NetworkXUnfeasible:
        return True
    else:
        return False


@nx._dispatchable
def is_directed_acyclic_graph(G):
    """Returns True if the graph `G` is a directed acyclic graph (DAG) or
    False if not.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    bool
        True if `G` is a DAG, False otherwise

    Examples
    --------
    Undirected graph::

        >>> G = nx.Graph([(1, 2), (2, 3)])
        >>> nx.is_directed_acyclic_graph(G)
        False

    Directed graph with cycle::

        >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        >>> nx.is_directed_acyclic_graph(G)
        False

    Directed acyclic graph::

        >>> G = nx.DiGraph([(1, 2), (2, 3)])
        >>> nx.is_directed_acyclic_graph(G)
        True

    See also
    --------
    topological_sort
    """
    return G.is_directed() and not has_cycle(G)


@nx._dispatchable
def topological_generations(G):
    """Stratifies a DAG into generations.

    A topological generation is node collection in which ancestors of a node in each
    generation are guaranteed to be in a previous generation, and any descendants of
    a node are guaranteed to be in a following generation. Nodes are guaranteed to
    be in the earliest possible generation that they can belong to.

    Parameters
    ----------
    G : NetworkX digraph
        A directed acyclic graph (DAG)

    Yields
    ------
    sets of nodes
        Yields sets of nodes representing each generation.

    Raises
    ------
    NetworkXError
        Generations are defined for directed graphs only. If the graph
        `G` is undirected, a :exc:`NetworkXError` is raised.

    NetworkXUnfeasible
        If `G` is not a directed acyclic graph (DAG) no topological generations
        exist and a :exc:`NetworkXUnfeasible` exception is raised.  This can also
        be raised if `G` is changed while the returned iterator is being processed

    RuntimeError
        If `G` is changed while the returned iterator is being processed.

    Examples
    --------
    >>> DG = nx.DiGraph([(2, 1), (3, 1)])
    >>> [sorted(generation) for generation in nx.topological_generations(DG)]
    [[2, 3], [1]]

    Notes
    -----
    The generation in which a node resides can also be determined by taking the
    max-path-distance from the node to the farthest leaf node. That value can
    be obtained with this function using `enumerate(topological_generations(G))`.

    See also
    --------
    topological_sort
    """
    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined on undirected graphs.")

    multigraph = G.is_multigraph()
    indegree_map = {v: d for v, d in G.in_degree() if d > 0}
    zero_indegree = [v for v, d in G.in_degree() if d == 0]

    while zero_indegree:
        this_generation = zero_indegree
        zero_indegree = []
        for node in this_generation:
            if node not in G:
                raise RuntimeError("Graph changed during iteration")
            for child in G.neighbors(node):
                try:
                    indegree_map[child] -= len(G[node][child]) if multigraph else 1
                except KeyError as err:
                    raise RuntimeError("Graph changed during iteration") from err
                if indegree_map[child] == 0:
                    zero_indegree.append(child)
                    del indegree_map[child]
        yield this_generation

    if indegree_map:
        raise nx.NetworkXUnfeasible(
            "Graph contains a cycle or graph changed during iteration"
        )


@nx._dispatchable
def topological_sort(G):
    """Returns a generator of nodes in topologically sorted order.

    A topological sort is a nonunique permutation of the nodes of a
    directed graph such that an edge from u to v implies that u
    appears before v in the topological sort order. This ordering is
    valid only if the graph has no directed cycles.

    Parameters
    ----------
    G : NetworkX digraph
        A directed acyclic graph (DAG)

    Yields
    ------
    nodes
        Yields the nodes in topological sorted order.

    Raises
    ------
    NetworkXError
        Topological sort is defined for directed graphs only. If the graph `G`
        is undirected, a :exc:`NetworkXError` is raised.

    NetworkXUnfeasible
        If `G` is not a directed acyclic graph (DAG) no topological sort exists
        and a :exc:`NetworkXUnfeasible` exception is raised.  This can also be
        raised if `G` is changed while the returned iterator is being processed

    RuntimeError
        If `G` is changed while the returned iterator is being processed.

    Examples
    --------
    To get the reverse order of the topological sort:

    >>> DG = nx.DiGraph([(1, 2), (2, 3)])
    >>> list(reversed(list(nx.topological_sort(DG))))
    [3, 2, 1]

    If your DiGraph naturally has the edges representing tasks/inputs
    and nodes representing people/processes that initiate tasks, then
    topological_sort is not quite what you need. You will have to change
    the tasks to nodes with dependence reflected by edges. The result is
    a kind of topological sort of the edges. This can be done
    with :func:`networkx.line_graph` as follows:

    >>> list(nx.topological_sort(nx.line_graph(DG)))
    [(1, 2), (2, 3)]

    Notes
    -----
    This algorithm is based on a description and proof in
    "Introduction to Algorithms: A Creative Approach" [1]_ .

    See also
    --------
    is_directed_acyclic_graph, lexicographical_topological_sort

    References
    ----------
    .. [1] Manber, U. (1989).
       *Introduction to Algorithms - A Creative Approach.* Addison-Wesley.
    """
    for generation in nx.topological_generations(G):
        yield from generation


@nx._dispatchable
def lexicographical_topological_sort(G, key=None):
    """Generate the nodes in the unique lexicographical topological sort order.

    Generates a unique ordering of nodes by first sorting topologically (for which there are often
    multiple valid orderings) and then additionally by sorting lexicographically.

    A topological sort arranges the nodes of a directed graph so that the
    upstream node of each directed edge precedes the downstream node.
    It is always possible to find a solution for directed graphs that have no cycles.
    There may be more than one valid solution.

    Lexicographical sorting is just sorting alphabetically. It is used here to break ties in the
    topological sort and to determine a single, unique ordering.  This can be useful in comparing
    sort results.

    The lexicographical order can be customized by providing a function to the `key=` parameter.
    The definition of the key function is the same as used in python's built-in `sort()`.
    The function takes a single argument and returns a key to use for sorting purposes.

    Lexicographical sorting can fail if the node names are un-sortable. See the example below.
    The solution is to provide a function to the `key=` argument that returns sortable keys.


    Parameters
    ----------
    G : NetworkX digraph
        A directed acyclic graph (DAG)

    key : function, optional
        A function of one argument that converts a node name to a comparison key.
        It defines and resolves ambiguities in the sort order.  Defaults to the identity function.

    Yields
    ------
    nodes
        Yields the nodes of G in lexicographical topological sort order.

    Raises
    ------
    NetworkXError
        Topological sort is defined for directed graphs only. If the graph `G`
        is undirected, a :exc:`NetworkXError` is raised.

    NetworkXUnfeasible
        If `G` is not a directed acyclic graph (DAG) no topological sort exists
        and a :exc:`NetworkXUnfeasible` exception is raised.  This can also be
        raised if `G` is changed while the returned iterator is being processed

    RuntimeError
        If `G` is changed while the returned iterator is being processed.

    TypeError
        Results from un-sortable node names.
        Consider using `key=` parameter to resolve ambiguities in the sort order.

    Examples
    --------
    >>> DG = nx.DiGraph([(2, 1), (2, 5), (1, 3), (1, 4), (5, 4)])
    >>> list(nx.lexicographical_topological_sort(DG))
    [2, 1, 3, 5, 4]
    >>> list(nx.lexicographical_topological_sort(DG, key=lambda x: -x))
    [2, 5, 1, 4, 3]

    The sort will fail for any graph with integer and string nodes. Comparison of integer to strings
    is not defined in python.  Is 3 greater or less than 'red'?

    >>> DG = nx.DiGraph([(1, "red"), (3, "red"), (1, "green"), (2, "blue")])
    >>> list(nx.lexicographical_topological_sort(DG))
    Traceback (most recent call last):
    ...
    TypeError: '<' not supported between instances of 'str' and 'int'
    ...

    Incomparable nodes can be resolved using a `key` function. This example function
    allows comparison of integers and strings by returning a tuple where the first
    element is True for `str`, False otherwise. The second element is the node name.
    This groups the strings and integers separately so they can be compared only among themselves.

    >>> key = lambda node: (isinstance(node, str), node)
    >>> list(nx.lexicographical_topological_sort(DG, key=key))
    [1, 2, 3, 'blue', 'green', 'red']

    Notes
    -----
    This algorithm is based on a description and proof in
    "Introduction to Algorithms: A Creative Approach" [1]_ .

    See also
    --------
    topological_sort

    References
    ----------
    .. [1] Manber, U. (1989).
       *Introduction to Algorithms - A Creative Approach.* Addison-Wesley.
    """
    if not G.is_directed():
        msg = "Topological sort not defined on undirected graphs."
        raise nx.NetworkXError(msg)

    if key is None:

        def key(node):
            return node

    nodeid_map = {n: i for i, n in enumerate(G)}

    def create_tuple(node):
        return key(node), nodeid_map[node], node

    indegree_map = {v: d for v, d in G.in_degree() if d > 0}
    # These nodes have zero indegree and ready to be returned.
    zero_indegree = [create_tuple(v) for v, d in G.in_degree() if d == 0]
    heapq.heapify(zero_indegree)

    while zero_indegree:
        _, _, node = heapq.heappop(zero_indegree)

        if node not in G:
            raise RuntimeError("Graph changed during iteration")
        for _, child in G.edges(node):
            try:
                indegree_map[child] -= 1
            except KeyError as err:
                raise RuntimeError("Graph changed during iteration") from err
            if indegree_map[child] == 0:
                try:
                    heapq.heappush(zero_indegree, create_tuple(child))
                except TypeError as err:
                    raise TypeError(
                        f"{err}\nConsider using `key=` parameter to resolve ambiguities in the sort order."
                    )
                del indegree_map[child]

        yield node

    if indegree_map:
        msg = "Graph contains a cycle or graph changed during iteration"
        raise nx.NetworkXUnfeasible(msg)


@not_implemented_for("undirected")
@nx._dispatchable
def all_topological_sorts(G):
    """Returns a generator of _all_ topological sorts of the directed graph G.

    A topological sort is a nonunique permutation of the nodes such that an
    edge from u to v implies that u appears before v in the topological sort
    order.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed graph

    Yields
    ------
    topological_sort_order : list
        a list of nodes in `G`, representing one of the topological sort orders

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not directed
    NetworkXUnfeasible
        If `G` is not acyclic

    Examples
    --------
    To enumerate all topological sorts of directed graph:

    >>> DG = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
    >>> list(nx.all_topological_sorts(DG))
    [[1, 2, 4, 3], [1, 2, 3, 4]]

    Notes
    -----
    Implements an iterative version of the algorithm given in [1].

    References
    ----------
    .. [1] Knuth, Donald E., Szwarcfiter, Jayme L. (1974).
       "A Structured Program to Generate All Topological Sorting Arrangements"
       Information Processing Letters, Volume 2, Issue 6, 1974, Pages 153-157,
       ISSN 0020-0190,
       https://doi.org/10.1016/0020-0190(74)90001-5.
       Elsevier (North-Holland), Amsterdam
    """
    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined on undirected graphs.")

    # the names of count and D are chosen to match the global variables in [1]
    # number of edges originating in a vertex v
    count = dict(G.in_degree())
    # vertices with indegree 0
    D = deque([v for v, d in G.in_degree() if d == 0])
    # stack of first value chosen at a position k in the topological sort
    bases = []
    current_sort = []

    # do-while construct
    while True:
        assert all(count[v] == 0 for v in D)

        if len(current_sort) == len(G):
            yield list(current_sort)

            # clean-up stack
            while len(current_sort) > 0:
                assert len(bases) == len(current_sort)
                q = current_sort.pop()

                # "restores" all edges (q, x)
                # NOTE: it is important to iterate over edges instead
                # of successors, so count is updated correctly in multigraphs
                for _, j in G.out_edges(q):
                    count[j] += 1
                    assert count[j] >= 0
                # remove entries from D
                while len(D) > 0 and count[D[-1]] > 0:
                    D.pop()

                # corresponds to a circular shift of the values in D
                # if the first value chosen (the base) is in the first
                # position of D again, we are done and need to consider the
                # previous condition
                D.appendleft(q)
                if D[-1] == bases[-1]:
                    # all possible values have been chosen at current position
                    # remove corresponding marker
                    bases.pop()
                else:
                    # there are still elements that have not been fixed
                    # at the current position in the topological sort
                    # stop removing elements, escape inner loop
                    break

        else:
            if len(D) == 0:
                raise nx.NetworkXUnfeasible("Graph contains a cycle.")

            # choose next node
            q = D.pop()
            # "erase" all edges (q, x)
            # NOTE: it is important to iterate over edges instead
            # of successors, so count is updated correctly in multigraphs
            for _, j in G.out_edges(q):
                count[j] -= 1
                assert count[j] >= 0
                if count[j] == 0:
                    D.append(j)
            current_sort.append(q)

            # base for current position might _not_ be fixed yet
            if len(bases) < len(current_sort):
                bases.append(q)

        if len(bases) == 0:
            break


@nx._dispatchable
def is_aperiodic(G):
    """Returns True if `G` is aperiodic.

    A directed graph is aperiodic if there is no integer k > 1 that
    divides the length of every cycle in the graph.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed graph

    Returns
    -------
    bool
        True if the graph is aperiodic False otherwise

    Raises
    ------
    NetworkXError
        If `G` is not directed

    Examples
    --------
    A graph consisting of one cycle, the length of which is 2. Therefore ``k = 2``
    divides the length of every cycle in the graph and thus the graph
    is *not aperiodic*::

        >>> DG = nx.DiGraph([(1, 2), (2, 1)])
        >>> nx.is_aperiodic(DG)
        False

    A graph consisting of two cycles: one of length 2 and the other of length 3.
    The cycle lengths are coprime, so there is no single value of k where ``k > 1``
    that divides each cycle length and therefore the graph is *aperiodic*::

        >>> DG = nx.DiGraph([(1, 2), (2, 3), (3, 1), (1, 4), (4, 1)])
        >>> nx.is_aperiodic(DG)
        True

    A graph consisting of two cycles: one of length 2 and the other of length 4.
    The lengths of the cycles share a common factor ``k = 2``, and therefore
    the graph is *not aperiodic*::

        >>> DG = nx.DiGraph([(1, 2), (2, 1), (3, 4), (4, 5), (5, 6), (6, 3)])
        >>> nx.is_aperiodic(DG)
        False

    An acyclic graph, therefore the graph is *not aperiodic*::

        >>> DG = nx.DiGraph([(1, 2), (2, 3)])
        >>> nx.is_aperiodic(DG)
        False

    Notes
    -----
    This uses the method outlined in [1]_, which runs in $O(m)$ time
    given $m$ edges in `G`. Note that a graph is not aperiodic if it is
    acyclic as every integer trivial divides length 0 cycles.

    References
    ----------
    .. [1] Jarvis, J. P.; Shier, D. R. (1996),
       "Graph-theoretic analysis of finite Markov chains,"
       in Shier, D. R.; Wallenius, K. T., Applied Mathematical Modeling:
       A Multidisciplinary Approach, CRC Press.
    """
    if not G.is_directed():
        raise nx.NetworkXError("is_aperiodic not defined for undirected graphs")
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Graph has no nodes.")
    s = arbitrary_element(G)
    levels = {s: 0}
    this_level = [s]
    g = 0
    lev = 1
    while this_level:
        next_level = []
        for u in this_level:
            for v in G[u]:
                if v in levels:  # Non-Tree Edge
                    g = gcd(g, levels[u] - levels[v] + 1)
                else:  # Tree Edge
                    next_level.append(v)
                    levels[v] = lev
        this_level = next_level
        lev += 1
    if len(levels) == len(G):  # All nodes in tree
        return g == 1
    else:
        return g == 1 and nx.is_aperiodic(G.subgraph(set(G) - set(levels)))


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def transitive_closure(G, reflexive=False):
    """Returns transitive closure of a graph

    The transitive closure of G = (V,E) is a graph G+ = (V,E+) such that
    for all v, w in V there is an edge (v, w) in E+ if and only if there
    is a path from v to w in G.

    Handling of paths from v to v has some flexibility within this definition.
    A reflexive transitive closure creates a self-loop for the path
    from v to v of length 0. The usual transitive closure creates a
    self-loop only if a cycle exists (a path from v to v with length > 0).
    We also allow an option for no self-loops.

    Parameters
    ----------
    G : NetworkX Graph
        A directed/undirected graph/multigraph.
    reflexive : Bool or None, optional (default: False)
        Determines when cycles create self-loops in the Transitive Closure.
        If True, trivial cycles (length 0) create self-loops. The result
        is a reflexive transitive closure of G.
        If False (the default) non-trivial cycles create self-loops.
        If None, self-loops are not created.

    Returns
    -------
    NetworkX graph
        The transitive closure of `G`

    Raises
    ------
    NetworkXError
        If `reflexive` not in `{None, True, False}`

    Examples
    --------
    The treatment of trivial (i.e. length 0) cycles is controlled by the
    `reflexive` parameter.

    Trivial (i.e. length 0) cycles do not create self-loops when
    ``reflexive=False`` (the default)::

        >>> DG = nx.DiGraph([(1, 2), (2, 3)])
        >>> TC = nx.transitive_closure(DG, reflexive=False)
        >>> TC.edges()
        OutEdgeView([(1, 2), (1, 3), (2, 3)])

    However, nontrivial (i.e. length greater than 0) cycles create self-loops
    when ``reflexive=False`` (the default)::

        >>> DG = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        >>> TC = nx.transitive_closure(DG, reflexive=False)
        >>> TC.edges()
        OutEdgeView([(1, 2), (1, 3), (1, 1), (2, 3), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)])

    Trivial cycles (length 0) create self-loops when ``reflexive=True``::

        >>> DG = nx.DiGraph([(1, 2), (2, 3)])
        >>> TC = nx.transitive_closure(DG, reflexive=True)
        >>> TC.edges()
        OutEdgeView([(1, 2), (1, 1), (1, 3), (2, 3), (2, 2), (3, 3)])

    And the third option is not to create self-loops at all when ``reflexive=None``::

        >>> DG = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        >>> TC = nx.transitive_closure(DG, reflexive=None)
        >>> TC.edges()
        OutEdgeView([(1, 2), (1, 3), (2, 3), (2, 1), (3, 1), (3, 2)])

    References
    ----------
    .. [1] https://www.ics.uci.edu/~eppstein/PADS/PartialOrder.py
    """
    TC = G.copy()

    if reflexive not in {None, True, False}:
        raise nx.NetworkXError("Incorrect value for the parameter `reflexive`")

    for v in G:
        if reflexive is None:
            TC.add_edges_from((v, u) for u in nx.descendants(G, v) if u not in TC[v])
        elif reflexive is True:
            TC.add_edges_from(
                (v, u) for u in nx.descendants(G, v) | {v} if u not in TC[v]
            )
        elif reflexive is False:
            TC.add_edges_from((v, e[1]) for e in nx.edge_bfs(G, v) if e[1] not in TC[v])

    return TC


@not_implemented_for("undirected")
@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def transitive_closure_dag(G, topo_order=None):
    """Returns the transitive closure of a directed acyclic graph.

    This function is faster than the function `transitive_closure`, but fails
    if the graph has a cycle.

    The transitive closure of G = (V,E) is a graph G+ = (V,E+) such that
    for all v, w in V there is an edge (v, w) in E+ if and only if there
    is a non-null path from v to w in G.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed acyclic graph (DAG)

    topo_order: list or tuple, optional
        A topological order for G (if None, the function will compute one)

    Returns
    -------
    NetworkX DiGraph
        The transitive closure of `G`

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not directed
    NetworkXUnfeasible
        If `G` has a cycle

    Examples
    --------
    >>> DG = nx.DiGraph([(1, 2), (2, 3)])
    >>> TC = nx.transitive_closure_dag(DG)
    >>> TC.edges()
    OutEdgeView([(1, 2), (1, 3), (2, 3)])

    Notes
    -----
    This algorithm is probably simple enough to be well-known but I didn't find
    a mention in the literature.
    """
    if topo_order is None:
        topo_order = list(topological_sort(G))

    TC = G.copy()

    # idea: traverse vertices following a reverse topological order, connecting
    # each vertex to its descendants at distance 2 as we go
    for v in reversed(topo_order):
        TC.add_edges_from((v, u) for u in nx.descendants_at_distance(TC, v, 2))

    return TC


@not_implemented_for("undirected")
@nx._dispatchable(returns_graph=True)
def transitive_reduction(G):
    """Returns transitive reduction of a directed graph

    The transitive reduction of G = (V,E) is a graph G- = (V,E-) such that
    for all v,w in V there is an edge (v,w) in E- if and only if (v,w) is
    in E and there is no path from v to w in G with length greater than 1.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed acyclic graph (DAG)

    Returns
    -------
    NetworkX DiGraph
        The transitive reduction of `G`

    Raises
    ------
    NetworkXError
        If `G` is not a directed acyclic graph (DAG) transitive reduction is
        not uniquely defined and a :exc:`NetworkXError` exception is raised.

    Examples
    --------
    To perform transitive reduction on a DiGraph:

    >>> DG = nx.DiGraph([(1, 2), (2, 3), (1, 3)])
    >>> TR = nx.transitive_reduction(DG)
    >>> list(TR.edges)
    [(1, 2), (2, 3)]

    To avoid unnecessary data copies, this implementation does not return a
    DiGraph with node/edge data.
    To perform transitive reduction on a DiGraph and transfer node/edge data:

    >>> DG = nx.DiGraph()
    >>> DG.add_edges_from([(1, 2), (2, 3), (1, 3)], color="red")
    >>> TR = nx.transitive_reduction(DG)
    >>> TR.add_nodes_from(DG.nodes(data=True))
    >>> TR.add_edges_from((u, v, DG.edges[u, v]) for u, v in TR.edges)
    >>> list(TR.edges(data=True))
    [(1, 2, {'color': 'red'}), (2, 3, {'color': 'red'})]

    References
    ----------
    https://en.wikipedia.org/wiki/Transitive_reduction

    """
    if not is_directed_acyclic_graph(G):
        msg = "Directed Acyclic Graph required for transitive_reduction"
        raise nx.NetworkXError(msg)
    TR = nx.DiGraph()
    TR.add_nodes_from(G.nodes())
    descendants = {}
    # count before removing set stored in descendants
    check_count = dict(G.in_degree)
    for u in G:
        u_nbrs = set(G[u])
        for v in G[u]:
            if v in u_nbrs:
                if v not in descendants:
                    descendants[v] = {y for x, y in nx.dfs_edges(G, v)}
                u_nbrs -= descendants[v]
            check_count[v] -= 1
            if check_count[v] == 0:
                del descendants[v]
        TR.add_edges_from((u, v) for v in u_nbrs)
    return TR


@not_implemented_for("undirected")
@nx._dispatchable
def antichains(G, topo_order=None):
    """Generates antichains from a directed acyclic graph (DAG).

    An antichain is a subset of a partially ordered set such that any
    two elements in the subset are incomparable.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed acyclic graph (DAG)

    topo_order: list or tuple, optional
        A topological order for G (if None, the function will compute one)

    Yields
    ------
    antichain : list
        a list of nodes in `G` representing an antichain

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not directed

    NetworkXUnfeasible
        If `G` contains a cycle

    Examples
    --------
    >>> DG = nx.DiGraph([(1, 2), (1, 3)])
    >>> list(nx.antichains(DG))
    [[], [3], [2], [2, 3], [1]]

    Notes
    -----
    This function was originally developed by Peter Jipsen and Franco Saliola
    for the SAGE project. It's included in NetworkX with permission from the
    authors. Original SAGE code at:

    https://github.com/sagemath/sage/blob/master/src/sage/combinat/posets/hasse_diagram.py

    References
    ----------
    .. [1] Free Lattices, by R. Freese, J. Jezek and J. B. Nation,
       AMS, Vol 42, 1995, p. 226.
    """
    if topo_order is None:
        topo_order = list(nx.topological_sort(G))

    TC = nx.transitive_closure_dag(G, topo_order)
    antichains_stacks = [([], list(reversed(topo_order)))]

    while antichains_stacks:
        (antichain, stack) = antichains_stacks.pop()
        # Invariant:
        #  - the elements of antichain are independent
        #  - the elements of stack are independent from those of antichain
        yield antichain
        while stack:
            x = stack.pop()
            new_antichain = antichain + [x]
            new_stack = [t for t in stack if not ((t in TC[x]) or (x in TC[t]))]
            antichains_stacks.append((new_antichain, new_stack))


@not_implemented_for("undirected")
@nx._dispatchable(edge_attrs={"weight": "default_weight"})
def dag_longest_path(G, weight="weight", default_weight=1, topo_order=None):
    """Returns the longest path in a directed acyclic graph (DAG).

    If `G` has edges with `weight` attribute the edge data are used as
    weight values.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed acyclic graph (DAG)

    weight : str, optional
        Edge data key to use for weight

    default_weight : int, optional
        The weight of edges that do not have a weight attribute

    topo_order: list or tuple, optional
        A topological order for `G` (if None, the function will compute one)

    Returns
    -------
    list
        Longest path

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not directed

    Examples
    --------
    >>> DG = nx.DiGraph(
    ...     [(0, 1, {"cost": 1}), (1, 2, {"cost": 1}), (0, 2, {"cost": 42})]
    ... )
    >>> list(nx.all_simple_paths(DG, 0, 2))
    [[0, 1, 2], [0, 2]]
    >>> nx.dag_longest_path(DG)
    [0, 1, 2]
    >>> nx.dag_longest_path(DG, weight="cost")
    [0, 2]

    In the case where multiple valid topological orderings exist, `topo_order`
    can be used to specify a specific ordering:

    >>> DG = nx.DiGraph([(0, 1), (0, 2)])
    >>> sorted(nx.all_topological_sorts(DG))  # Valid topological orderings
    [[0, 1, 2], [0, 2, 1]]
    >>> nx.dag_longest_path(DG, topo_order=[0, 1, 2])
    [0, 1]
    >>> nx.dag_longest_path(DG, topo_order=[0, 2, 1])
    [0, 2]

    See also
    --------
    dag_longest_path_length

    """
    if not G:
        return []

    if topo_order is None:
        topo_order = nx.topological_sort(G)

    dist = {}  # stores {v : (length, u)}
    for v in topo_order:
        us = [
            (
                dist[u][0]
                + (
                    max(data.values(), key=lambda x: x.get(weight, default_weight))
                    if G.is_multigraph()
                    else data
                ).get(weight, default_weight),
                u,
            )
            for u, data in G.pred[v].items()
        ]

        # Use the best predecessor if there is one and its distance is
        # non-negative, otherwise terminate.
        maxu = max(us, key=lambda x: x[0]) if us else (0, v)
        dist[v] = maxu if maxu[0] >= 0 else (0, v)

    u = None
    v = max(dist, key=lambda x: dist[x][0])
    path = []
    while u != v:
        path.append(v)
        u = v
        v = dist[v][1]

    path.reverse()
    return path


@not_implemented_for("undirected")
@nx._dispatchable(edge_attrs={"weight": "default_weight"})
def dag_longest_path_length(G, weight="weight", default_weight=1):
    """Returns the longest path length in a DAG

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed acyclic graph (DAG)

    weight : string, optional
        Edge data key to use for weight

    default_weight : int, optional
        The weight of edges that do not have a weight attribute

    Returns
    -------
    int
        Longest path length

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not directed

    Examples
    --------
    >>> DG = nx.DiGraph(
    ...     [(0, 1, {"cost": 1}), (1, 2, {"cost": 1}), (0, 2, {"cost": 42})]
    ... )
    >>> list(nx.all_simple_paths(DG, 0, 2))
    [[0, 1, 2], [0, 2]]
    >>> nx.dag_longest_path_length(DG)
    2
    >>> nx.dag_longest_path_length(DG, weight="cost")
    42

    See also
    --------
    dag_longest_path
    """
    path = nx.dag_longest_path(G, weight, default_weight)
    path_length = 0
    if G.is_multigraph():
        for u, v in pairwise(path):
            i = max(G[u][v], key=lambda x: G[u][v][x].get(weight, default_weight))
            path_length += G[u][v][i].get(weight, default_weight)
    else:
        for u, v in pairwise(path):
            path_length += G[u][v].get(weight, default_weight)

    return path_length


@nx._dispatchable
def root_to_leaf_paths(G):
    """Yields root-to-leaf paths in a directed acyclic graph.

    `G` must be a directed acyclic graph. If not, the behavior of this
    function is undefined. A "root" in this graph is a node of in-degree
    zero and a "leaf" a node of out-degree zero.

    When invoked, this function iterates over each path from any root to
    any leaf. A path is a list of nodes.

    """
    roots = (v for v, d in G.in_degree() if d == 0)
    leaves = (v for v, d in G.out_degree() if d == 0)
    all_paths = partial(nx.all_simple_paths, G)
    # TODO In Python 3, this would be better as `yield from ...`.
    return chaini(starmap(all_paths, product(roots, leaves)))


@not_implemented_for("multigraph")
@not_implemented_for("undirected")
@nx._dispatchable(returns_graph=True)
def dag_to_branching(G):
    """Returns a branching representing all (overlapping) paths from
    root nodes to leaf nodes in the given directed acyclic graph.

    As described in :mod:`networkx.algorithms.tree.recognition`, a
    *branching* is a directed forest in which each node has at most one
    parent. In other words, a branching is a disjoint union of
    *arborescences*. For this function, each node of in-degree zero in
    `G` becomes a root of one of the arborescences, and there will be
    one leaf node for each distinct path from that root to a leaf node
    in `G`.

    Each node `v` in `G` with *k* parents becomes *k* distinct nodes in
    the returned branching, one for each parent, and the sub-DAG rooted
    at `v` is duplicated for each copy. The algorithm then recurses on
    the children of each copy of `v`.

    Parameters
    ----------
    G : NetworkX graph
        A directed acyclic graph.

    Returns
    -------
    DiGraph
        The branching in which there is a bijection between root-to-leaf
        paths in `G` (in which multiple paths may share the same leaf)
        and root-to-leaf paths in the branching (in which there is a
        unique path from a root to a leaf).

        Each node has an attribute 'source' whose value is the original
        node to which this node corresponds. No other graph, node, or
        edge attributes are copied into this new graph.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not directed, or if `G` is a multigraph.

    HasACycle
        If `G` is not acyclic.

    Examples
    --------
    To examine which nodes in the returned branching were produced by
    which original node in the directed acyclic graph, we can collect
    the mapping from source node to new nodes into a dictionary. For
    example, consider the directed diamond graph::

        >>> from collections import defaultdict
        >>> from operator import itemgetter
        >>>
        >>> G = nx.DiGraph(nx.utils.pairwise("abd"))
        >>> G.add_edges_from(nx.utils.pairwise("acd"))
        >>> B = nx.dag_to_branching(G)
        >>>
        >>> sources = defaultdict(set)
        >>> for v, source in B.nodes(data="source"):
        ...     sources[source].add(v)
        >>> len(sources["a"])
        1
        >>> len(sources["d"])
        2

    To copy node attributes from the original graph to the new graph,
    you can use a dictionary like the one constructed in the above
    example::

        >>> for source, nodes in sources.items():
        ...     for v in nodes:
        ...         B.nodes[v].update(G.nodes[source])

    Notes
    -----
    This function is not idempotent in the sense that the node labels in
    the returned branching may be uniquely generated each time the
    function is invoked. In fact, the node labels may not be integers;
    in order to relabel the nodes to be more readable, you can use the
    :func:`networkx.convert_node_labels_to_integers` function.

    The current implementation of this function uses
    :func:`networkx.prefix_tree`, so it is subject to the limitations of
    that function.

    """
    if has_cycle(G):
        msg = "dag_to_branching is only defined for acyclic graphs"
        raise nx.HasACycle(msg)
    paths = root_to_leaf_paths(G)
    B = nx.prefix_tree(paths)
    # Remove the synthetic `root`(0) and `NIL`(-1) nodes from the tree
    B.remove_node(0)
    B.remove_node(-1)
    return B


@not_implemented_for("undirected")
@nx._dispatchable
def compute_v_structures(G):
    """Yields 3-node tuples that represent the v-structures in `G`.

    .. deprecated:: 3.4

       `compute_v_structures` actually yields colliders. It will be removed in
       version 3.6. Use `nx.dag.v_structures` or `nx.dag.colliders` instead.

    Colliders are triples in the directed acyclic graph (DAG) where two parent nodes
    point to the same child node. V-structures are colliders where the two parent
    nodes are not adjacent. In a causal graph setting, the parents do not directly
    depend on each other, but conditioning on the child node provides an association.

    Parameters
    ----------
    G : graph
        A networkx `~networkx.DiGraph`.

    Yields
    ------
    A 3-tuple representation of a v-structure
        Each v-structure is a 3-tuple with the parent, collider, and other parent.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is an undirected graph.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (0, 4), (3, 1), (2, 4), (0, 5), (4, 5), (1, 5)])
    >>> nx.is_directed_acyclic_graph(G)
    True
    >>> list(nx.compute_v_structures(G))
    [(0, 4, 2), (0, 5, 4), (0, 5, 1), (4, 5, 1)]

    See Also
    --------
    v_structures
    colliders

    Notes
    -----
    This function was written to be used on DAGs, however it works on cyclic graphs
    too. Since colliders are referred to in the cyclic causal graph literature
    [2]_ we allow cyclic graphs in this function. It is suggested that you test if
    your input graph is acyclic as in the example if you want that property.

    References
    ----------
    .. [1]  `Pearl's PRIMER <https://bayes.cs.ucla.edu/PRIMER/primer-ch2.pdf>`_
            Ch-2 page 50: v-structures def.
    .. [2] A Hyttinen, P.O. Hoyer, F. Eberhardt, M J ̈arvisalo, (2013)
           "Discovering cyclic causal models with latent variables:
           a general SAT-based procedure", UAI'13: Proceedings of the Twenty-Ninth
           Conference on Uncertainty in Artificial Intelligence, pg 301–310,
           `doi:10.5555/3023638.3023669 <https://dl.acm.org/doi/10.5555/3023638.3023669>`_
    """
    import warnings

    warnings.warn(
        (
            "\n\n`compute_v_structures` actually yields colliders. It will be\n"
            "removed in version 3.6. Use `nx.dag.v_structures` or `nx.dag.colliders`\n"
            "instead.\n"
        ),
        category=DeprecationWarning,
        stacklevel=5,
    )

    return colliders(G)


@not_implemented_for("undirected")
@nx._dispatchable
def v_structures(G):
    """Yields 3-node tuples that represent the v-structures in `G`.

    Colliders are triples in the directed acyclic graph (DAG) where two parent nodes
    point to the same child node. V-structures are colliders where the two parent
    nodes are not adjacent. In a causal graph setting, the parents do not directly
    depend on each other, but conditioning on the child node provides an association.

    Parameters
    ----------
    G : graph
        A networkx `~networkx.DiGraph`.

    Yields
    ------
    A 3-tuple representation of a v-structure
        Each v-structure is a 3-tuple with the parent, collider, and other parent.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is an undirected graph.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (0, 4), (3, 1), (2, 4), (0, 5), (4, 5), (1, 5)])
    >>> nx.is_directed_acyclic_graph(G)
    True
    >>> list(nx.dag.v_structures(G))
    [(0, 4, 2), (0, 5, 1), (4, 5, 1)]

    See Also
    --------
    colliders

    Notes
    -----
    This function was written to be used on DAGs, however it works on cyclic graphs
    too. Since colliders are referred to in the cyclic causal graph literature
    [2]_ we allow cyclic graphs in this function. It is suggested that you test if
    your input graph is acyclic as in the example if you want that property.

    References
    ----------
    .. [1]  `Pearl's PRIMER <https://bayes.cs.ucla.edu/PRIMER/primer-ch2.pdf>`_
            Ch-2 page 50: v-structures def.
    .. [2] A Hyttinen, P.O. Hoyer, F. Eberhardt, M J ̈arvisalo, (2013)
           "Discovering cyclic causal models with latent variables:
           a general SAT-based procedure", UAI'13: Proceedings of the Twenty-Ninth
           Conference on Uncertainty in Artificial Intelligence, pg 301–310,
           `doi:10.5555/3023638.3023669 <https://dl.acm.org/doi/10.5555/3023638.3023669>`_
    """
    for p1, c, p2 in colliders(G):
        if not (G.has_edge(p1, p2) or G.has_edge(p2, p1)):
            yield (p1, c, p2)


@not_implemented_for("undirected")
@nx._dispatchable
def colliders(G):
    """Yields 3-node tuples that represent the colliders in `G`.

    In a Directed Acyclic Graph (DAG), if you have three nodes A, B, and C, and
    there are edges from A to C and from B to C, then C is a collider [1]_ . In
    a causal graph setting, this means that both events A and B are "causing" C,
    and conditioning on C provide an association between A and B even if
    no direct causal relationship exists between A and B.

    Parameters
    ----------
    G : graph
        A networkx `~networkx.DiGraph`.

    Yields
    ------
    A 3-tuple representation of a collider
        Each collider is a 3-tuple with the parent, collider, and other parent.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is an undirected graph.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (0, 4), (3, 1), (2, 4), (0, 5), (4, 5), (1, 5)])
    >>> nx.is_directed_acyclic_graph(G)
    True
    >>> list(nx.dag.colliders(G))
    [(0, 4, 2), (0, 5, 4), (0, 5, 1), (4, 5, 1)]

    See Also
    --------
    v_structures

    Notes
    -----
    This function was written to be used on DAGs, however it works on cyclic graphs
    too. Since colliders are referred to in the cyclic causal graph literature
    [2]_ we allow cyclic graphs in this function. It is suggested that you test if
    your input graph is acyclic as in the example if you want that property.

    References
    ----------
    .. [1] `Wikipedia: Collider in causal graphs <https://en.wikipedia.org/wiki/Collider_(statistics)>`_
    .. [2] A Hyttinen, P.O. Hoyer, F. Eberhardt, M J ̈arvisalo, (2013)
           "Discovering cyclic causal models with latent variables:
           a general SAT-based procedure", UAI'13: Proceedings of the Twenty-Ninth
           Conference on Uncertainty in Artificial Intelligence, pg 301–310,
           `doi:10.5555/3023638.3023669 <https://dl.acm.org/doi/10.5555/3023638.3023669>`_
    """
    for node in G.nodes:
        for p1, p2 in combinations(G.predecessors(node), 2):
            yield (p1, node, p2)
<|endoftext|>"""Function for computing walks in a graph."""

import networkx as nx

__all__ = ["number_of_walks"]


@nx._dispatchable
def number_of_walks(G, walk_length):
    """Returns the number of walks connecting each pair of nodes in `G`

    A *walk* is a sequence of nodes in which each adjacent pair of nodes
    in the sequence is adjacent in the graph. A walk can repeat the same
    edge and go in the opposite direction just as people can walk on a
    set of paths, but standing still is not counted as part of the walk.

    This function only counts the walks with `walk_length` edges. Note that
    the number of nodes in the walk sequence is one more than `walk_length`.
    The number of walks can grow very quickly on a larger graph
    and with a larger walk length.

    Parameters
    ----------
    G : NetworkX graph

    walk_length : int
        A nonnegative integer representing the length of a walk.

    Returns
    -------
    dict
        A dictionary of dictionaries in which outer keys are source
        nodes, inner keys are target nodes, and inner values are the
        number of walks of length `walk_length` connecting those nodes.

    Raises
    ------
    ValueError
        If `walk_length` is negative

    Examples
    --------

    >>> G = nx.Graph([(0, 1), (1, 2)])
    >>> walks = nx.number_of_walks(G, 2)
    >>> walks
    {0: {0: 1, 1: 0, 2: 1}, 1: {0: 0, 1: 2, 2: 0}, 2: {0: 1, 1: 0, 2: 1}}
    >>> total_walks = sum(sum(tgts.values()) for _, tgts in walks.items())

    You can also get the number of walks from a specific source node using the
    returned dictionary. For example, number of walks of length 1 from node 0
    can be found as follows:

    >>> walks = nx.number_of_walks(G, 1)
    >>> walks[0]
    {0: 0, 1: 1, 2: 0}
    >>> sum(walks[0].values())  # walks from 0 of length 1
    1

    Similarly, a target node can also be specified:

    >>> walks[0][1]
    1

    """
    import numpy as np

    if walk_length < 0:
        raise ValueError(f"`walk_length` cannot be negative: {walk_length}")

    A = nx.adjacency_matrix(G, weight=None)
    # TODO: Use matrix_power from scipy.sparse when available
    # power = sp.sparse.linalg.matrix_power(A, walk_length)
    power = np.linalg.matrix_power(A.toarray(), walk_length)
    result = {
        u: {v: power.item(u_idx, v_idx) for v_idx, v in enumerate(G)}
        for u_idx, u in enumerate(G)
    }
    return result
<|endoftext|>"""Graph diameter, radius, eccentricity and other properties."""

import math

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "eccentricity",
    "diameter",
    "harmonic_diameter",
    "radius",
    "periphery",
    "center",
    "barycenter",
    "resistance_distance",
    "kemeny_constant",
    "effective_graph_resistance",
]


def _extrema_bounding(G, compute="diameter", weight=None):
    """Compute requested extreme distance metric of undirected graph G

    Computation is based on smart lower and upper bounds, and in practice
    linear in the number of nodes, rather than quadratic (except for some
    border cases such as complete graphs or circle shaped graphs).

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph

    compute : string denoting the requesting metric
       "diameter" for the maximal eccentricity value,
       "radius" for the minimal eccentricity value,
       "periphery" for the set of nodes with eccentricity equal to the diameter,
       "center" for the set of nodes with eccentricity equal to the radius,
       "eccentricities" for the maximum distance from each node to all other nodes in G

    weight : string, function, or None
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

        If this is None, every edge has weight/distance/cost 1.

        Weights stored as floating point values can lead to small round-off
        errors in distances. Use integer weights to avoid this.

        Weights should be positive, since they are distances.

    Returns
    -------
    value : value of the requested metric
       int for "diameter" and "radius" or
       list of nodes for "center" and "periphery" or
       dictionary of eccentricity values keyed by node for "eccentricities"

    Raises
    ------
    NetworkXError
        If the graph consists of multiple components
    ValueError
        If `compute` is not one of "diameter", "radius", "periphery", "center", or "eccentricities".

    Notes
    -----
    This algorithm was proposed in [1]_ and discussed further in [2]_ and [3]_.

    References
    ----------
    .. [1] F. W. Takes, W. A. Kosters,
       "Determining the diameter of small world networks."
       Proceedings of the 20th ACM international conference on Information and knowledge management, 2011
       https://dl.acm.org/doi/abs/10.1145/2063576.2063748
    .. [2] F. W. Takes, W. A. Kosters,
       "Computing the Eccentricity Distribution of Large Graphs."
       Algorithms, 2013
       https://www.mdpi.com/1999-4893/6/1/100
    .. [3] M. Borassi, P. Crescenzi, M. Habib, W. A. Kosters, A. Marino, F. W. Takes,
       "Fast diameter and radius BFS-based computation in (weakly connected) real-world graphs: With an application to the six degrees of separation games. "
       Theoretical Computer Science, 2015
       https://www.sciencedirect.com/science/article/pii/S0304397515001644
    """
    # init variables
    degrees = dict(G.degree())  # start with the highest degree node
    minlowernode = max(degrees, key=degrees.get)
    N = len(degrees)  # number of nodes
    # alternate between smallest lower and largest upper bound
    high = False
    # status variables
    ecc_lower = dict.fromkeys(G, 0)
    ecc_upper = dict.fromkeys(G, N)
    candidates = set(G)

    # (re)set bound extremes
    minlower = N
    maxlower = 0
    minupper = N
    maxupper = 0

    # repeat the following until there are no more candidates
    while candidates:
        if high:
            current = maxuppernode  # select node with largest upper bound
        else:
            current = minlowernode  # select node with smallest lower bound
        high = not high

        # get distances from/to current node and derive eccentricity
        dist = nx.shortest_path_length(G, source=current, weight=weight)

        if len(dist) != N:
            msg = "Cannot compute metric because graph is not connected."
            raise nx.NetworkXError(msg)
        current_ecc = max(dist.values())

        # print status update
        #        print ("ecc of " + str(current) + " (" + str(ecc_lower[current]) + "/"
        #        + str(ecc_upper[current]) + ", deg: " + str(dist[current]) + ") is "
        #        + str(current_ecc))
        #        print(ecc_upper)

        # (re)set bound extremes
        maxuppernode = None
        minlowernode = None

        # update node bounds
        for i in candidates:
            # update eccentricity bounds
            d = dist[i]
            ecc_lower[i] = low = max(ecc_lower[i], max(d, (current_ecc - d)))
            ecc_upper[i] = upp = min(ecc_upper[i], current_ecc + d)

            # update min/max values of lower and upper bounds
            minlower = min(ecc_lower[i], minlower)
            maxlower = max(ecc_lower[i], maxlower)
            minupper = min(ecc_upper[i], minupper)
            maxupper = max(ecc_upper[i], maxupper)

        # update candidate set
        if compute == "diameter":
            ruled_out = {
                i
                for i in candidates
                if ecc_upper[i] <= maxlower and 2 * ecc_lower[i] >= maxupper
            }
        elif compute == "radius":
            ruled_out = {
                i
                for i in candidates
                if ecc_lower[i] >= minupper and ecc_upper[i] + 1 <= 2 * minlower
            }
        elif compute == "periphery":
            ruled_out = {
                i
                for i in candidates
                if ecc_upper[i] < maxlower
                and (maxlower == maxupper or ecc_lower[i] > maxupper)
            }
        elif compute == "center":
            ruled_out = {
                i
                for i in candidates
                if ecc_lower[i] > minupper
                and (minlower == minupper or ecc_upper[i] + 1 < 2 * minlower)
            }
        elif compute == "eccentricities":
            ruled_out = set()
        else:
            msg = "compute must be one of 'diameter', 'radius', 'periphery', 'center', 'eccentricities'"
            raise ValueError(msg)

        ruled_out.update(i for i in candidates if ecc_lower[i] == ecc_upper[i])
        candidates -= ruled_out

        #        for i in ruled_out:
        #            print("removing %g: ecc_u: %g maxl: %g ecc_l: %g maxu: %g"%
        #                    (i,ecc_upper[i],maxlower,ecc_lower[i],maxupper))
        #        print("node %g: ecc_u: %g maxl: %g ecc_l: %g maxu: %g"%
        #                    (4,ecc_upper[4],maxlower,ecc_lower[4],maxupper))
        #        print("NODE 4: %g"%(ecc_upper[4] <= maxlower))
        #        print("NODE 4: %g"%(2 * ecc_lower[4] >= maxupper))
        #        print("NODE 4: %g"%(ecc_upper[4] <= maxlower
        #                            and 2 * ecc_lower[4] >= maxupper))

        # updating maxuppernode and minlowernode for selection in next round
        for i in candidates:
            if (
                minlowernode is None
                or (
                    ecc_lower[i] == ecc_lower[minlowernode]
                    and degrees[i] > degrees[minlowernode]
                )
                or (ecc_lower[i] < ecc_lower[minlowernode])
            ):
                minlowernode = i

            if (
                maxuppernode is None
                or (
                    ecc_upper[i] == ecc_upper[maxuppernode]
                    and degrees[i] > degrees[maxuppernode]
                )
                or (ecc_upper[i] > ecc_upper[maxuppernode])
            ):
                maxuppernode = i

        # print status update
    #        print (" min=" + str(minlower) + "/" + str(minupper) +
    #        " max=" + str(maxlower) + "/" + str(maxupper) +
    #        " candidates: " + str(len(candidates)))
    #        print("cand:",candidates)
    #        print("ecc_l",ecc_lower)
    #        print("ecc_u",ecc_upper)
    #        wait = input("press Enter to continue")

    # return the correct value of the requested metric
    if compute == "diameter":
        return maxlower
    if compute == "radius":
        return minupper
    if compute == "periphery":
        p = [v for v in G if ecc_lower[v] == maxlower]
        return p
    if compute == "center":
        c = [v for v in G if ecc_upper[v] == minupper]
        return c
    if compute == "eccentricities":
        return ecc_lower
    return None


@nx._dispatchable(edge_attrs="weight")
def eccentricity(G, v=None, sp=None, weight=None):
    """Returns the eccentricity of nodes in G.

    The eccentricity of a node v is the maximum distance from v to
    all other nodes in G.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    v : node, optional
       Return value of specified node

    sp : dict of dicts, optional
       All pairs shortest path lengths as a dictionary of dictionaries

    weight : string, function, or None (default=None)
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

        If this is None, every edge has weight/distance/cost 1.

        Weights stored as floating point values can lead to small round-off
        errors in distances. Use integer weights to avoid this.

        Weights should be positive, since they are distances.

    Returns
    -------
    ecc : dictionary
       A dictionary of eccentricity values keyed by node.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> dict(nx.eccentricity(G))
    {1: 2, 2: 3, 3: 2, 4: 2, 5: 3}

    >>> dict(
    ...     nx.eccentricity(G, v=[1, 5])
    ... )  # This returns the eccentricity of node 1 & 5
    {1: 2, 5: 3}

    """
    #    if v is None:                # none, use entire graph
    #        nodes=G.nodes()
    #    elif v in G:               # is v a single node
    #        nodes=[v]
    #    else:                      # assume v is a container of nodes
    #        nodes=v
    order = G.order()
    e = {}
    for n in G.nbunch_iter(v):
        if sp is None:
            length = nx.shortest_path_length(G, source=n, weight=weight)

            L = len(length)
        else:
            try:
                length = sp[n]
                L = len(length)
            except TypeError as err:
                raise nx.NetworkXError('Format of "sp" is invalid.') from err
        if L != order:
            if G.is_directed():
                msg = (
                    "Found infinite path length because the digraph is not"
                    " strongly connected"
                )
            else:
                msg = "Found infinite path length because the graph is not" " connected"
            raise nx.NetworkXError(msg)

        e[n] = max(length.values())

    if v in G:
        return e[v]  # return single value
    return e


@nx._dispatchable(edge_attrs="weight")
def diameter(G, e=None, usebounds=False, weight=None):
    """Returns the diameter of the graph G.

    The diameter is the maximum eccentricity.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    e : eccentricity dictionary, optional
      A precomputed dictionary of eccentricities.

    weight : string, function, or None
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

        If this is None, every edge has weight/distance/cost 1.

        Weights stored as floating point values can lead to small round-off
        errors in distances. Use integer weights to avoid this.

        Weights should be positive, since they are distances.

    Returns
    -------
    d : integer
       Diameter of graph

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> nx.diameter(G)
    3

    See Also
    --------
    eccentricity
    """
    if usebounds is True and e is None and not G.is_directed():
        return _extrema_bounding(G, compute="diameter", weight=weight)
    if e is None:
        e = eccentricity(G, weight=weight)
    return max(e.values())


@nx._dispatchable
def harmonic_diameter(G, sp=None):
    """Returns the harmonic diameter of the graph G.

    The harmonic diameter of a graph is the harmonic mean of the distances
    between all pairs of distinct vertices. Graphs that are not strongly
    connected have infinite diameter and mean distance, making such
    measures not useful. Restricting the diameter or mean distance to
    finite distances yields paradoxical values (e.g., a perfect match
    would have diameter one). The harmonic mean handles gracefully
    infinite distances (e.g., a perfect match has harmonic diameter equal
    to the number of vertices minus one), making it possible to assign a
    meaningful value to all graphs.

    Note that in [1] the harmonic diameter is called "connectivity length":
    however, "harmonic diameter" is a more standard name from the
    theory of metric spaces. The name "harmonic mean distance" is perhaps
    a more descriptive name, but is not used in the literature, so we use the
    name "harmonic diameter" here.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    sp : dict of dicts, optional
       All-pairs shortest path lengths as a dictionary of dictionaries

    Returns
    -------
    hd : float
       Harmonic diameter of graph

    References
    ----------
    .. [1] Massimo Marchiori and Vito Latora, "Harmony in the small-world".
           *Physica A: Statistical Mechanics and Its Applications*
           285(3-4), pages 539-546, 2000.
           <https://doi.org/10.1016/S0378-4371(00)00311-3>
    """
    order = G.order()

    sum_invd = 0
    for n in G:
        if sp is None:
            length = nx.single_source_shortest_path_length(G, n)
        else:
            try:
                length = sp[n]
                L = len(length)
            except TypeError as err:
                raise nx.NetworkXError('Format of "sp" is invalid.') from err

        for d in length.values():
            # Note that this will skip the zero distance from n to itself,
            # as it should be, but also zero-weight paths in weighted graphs.
            if d != 0:
                sum_invd += 1 / d

    if sum_invd != 0:
        return order * (order - 1) / sum_invd
    if order > 1:
        return math.inf
    return math.nan


@nx._dispatchable(edge_attrs="weight")
def periphery(G, e=None, usebounds=False, weight=None):
    """Returns the periphery of the graph G.

    The periphery is the set of nodes with eccentricity equal to the diameter.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    e : eccentricity dictionary, optional
      A precomputed dictionary of eccentricities.

    weight : string, function, or None
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

        If this is None, every edge has weight/distance/cost 1.

        Weights stored as floating point values can lead to small round-off
        errors in distances. Use integer weights to avoid this.

        Weights should be positive, since they are distances.

    Returns
    -------
    p : list
       List of nodes in periphery

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> nx.periphery(G)
    [2, 5]

    See Also
    --------
    barycenter
    center
    """
    if usebounds is True and e is None and not G.is_directed():
        return _extrema_bounding(G, compute="periphery", weight=weight)
    if e is None:
        e = eccentricity(G, weight=weight)
    diameter = max(e.values())
    p = [v for v in e if e[v] == diameter]
    return p


@nx._dispatchable(edge_attrs="weight")
def radius(G, e=None, usebounds=False, weight=None):
    """Returns the radius of the graph G.

    The radius is the minimum eccentricity.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    e : eccentricity dictionary, optional
      A precomputed dictionary of eccentricities.

    weight : string, function, or None
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

        If this is None, every edge has weight/distance/cost 1.

        Weights stored as floating point values can lead to small round-off
        errors in distances. Use integer weights to avoid this.

        Weights should be positive, since they are distances.

    Returns
    -------
    r : integer
       Radius of graph

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> nx.radius(G)
    2

    """
    if usebounds is True and e is None and not G.is_directed():
        return _extrema_bounding(G, compute="radius", weight=weight)
    if e is None:
        e = eccentricity(G, weight=weight)
    return min(e.values())


@nx._dispatchable(edge_attrs="weight")
def center(G, e=None, usebounds=False, weight=None):
    """Returns the center of the graph G.

    The center is the set of nodes with eccentricity equal to radius.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    e : eccentricity dictionary, optional
      A precomputed dictionary of eccentricities.

    weight : string, function, or None
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

        If this is None, every edge has weight/distance/cost 1.

        Weights stored as floating point values can lead to small round-off
        errors in distances. Use integer weights to avoid this.

        Weights should be positive, since they are distances.

    Returns
    -------
    c : list
       List of nodes in center

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> list(nx.center(G))
    [1, 3, 4]

    See Also
    --------
    barycenter
    periphery
    """
    if usebounds is True and e is None and not G.is_directed():
        return _extrema_bounding(G, compute="center", weight=weight)
    if e is None:
        e = eccentricity(G, weight=weight)
    radius = min(e.values())
    p = [v for v in e if e[v] == radius]
    return p


@nx._dispatchable(edge_attrs="weight", mutates_input={"attr": 2})
def barycenter(G, weight=None, attr=None, sp=None):
    r"""Calculate barycenter of a connected graph, optionally with edge weights.

    The :dfn:`barycenter` a
    :func:`connected <networkx.algorithms.components.is_connected>` graph
    :math:`G` is the subgraph induced by the set of its nodes :math:`v`
    minimizing the objective function

    .. math::

        \sum_{u \in V(G)} d_G(u, v),

    where :math:`d_G` is the (possibly weighted) :func:`path length
    <networkx.algorithms.shortest_paths.generic.shortest_path_length>`.
    The barycenter is also called the :dfn:`median`. See [West01]_, p. 78.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        The connected graph :math:`G`.
    weight : :class:`str`, optional
        Passed through to
        :func:`~networkx.algorithms.shortest_paths.generic.shortest_path_length`.
    attr : :class:`str`, optional
        If given, write the value of the objective function to each node's
        `attr` attribute. Otherwise do not store the value.
    sp : dict of dicts, optional
       All pairs shortest path lengths as a dictionary of dictionaries

    Returns
    -------
    list
        Nodes of `G` that induce the barycenter of `G`.

    Raises
    ------
    NetworkXNoPath
        If `G` is disconnected. `G` may appear disconnected to
        :func:`barycenter` if `sp` is given but is missing shortest path
        lengths for any pairs.
    ValueError
        If `sp` and `weight` are both given.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> nx.barycenter(G)
    [1, 3, 4]

    See Also
    --------
    center
    periphery
    """
    if sp is None:
        sp = nx.shortest_path_length(G, weight=weight)
    else:
        sp = sp.items()
        if weight is not None:
            raise ValueError("Cannot use both sp, weight arguments together")
    smallest, barycenter_vertices, n = float("inf"), [], len(G)
    for v, dists in sp:
        if len(dists) < n:
            raise nx.NetworkXNoPath(
                f"Input graph {G} is disconnected, so every induced subgraph "
                "has infinite barycentricity."
            )
        barycentricity = sum(dists.values())
        if attr is not None:
            G.nodes[v][attr] = barycentricity
        if barycentricity < smallest:
            smallest = barycentricity
            barycenter_vertices = [v]
        elif barycentricity == smallest:
            barycenter_vertices.append(v)
    if attr is not None:
        nx._clear_cache(G)
    return barycenter_vertices


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def resistance_distance(G, nodeA=None, nodeB=None, weight=None, invert_weight=True):
    """Returns the resistance distance between pairs of nodes in graph G.

    The resistance distance between two nodes of a graph is akin to treating
    the graph as a grid of resistors with a resistance equal to the provided
    weight [1]_, [2]_.

    If weight is not provided, then a weight of 1 is used for all edges.

    If two nodes are the same, the resistance distance is zero.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    nodeA : node or None, optional (default=None)
      A node within graph G.
      If None, compute resistance distance using all nodes as source nodes.

    nodeB : node or None, optional (default=None)
      A node within graph G.
      If None, compute resistance distance using all nodes as target nodes.

    weight : string or None, optional (default=None)
       The edge data key used to compute the resistance distance.
       If None, then each edge has weight 1.

    invert_weight : boolean (default=True)
        Proper calculation of resistance distance requires building the
        Laplacian matrix with the reciprocal of the weight. Not required
        if the weight is already inverted. Weight cannot be zero.

    Returns
    -------
    rd : dict or float
       If `nodeA` and `nodeB` are given, resistance distance between `nodeA`
       and `nodeB`. If `nodeA` or `nodeB` is unspecified (the default), a
       dictionary of nodes with resistance distances as the value.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a directed graph.

    NetworkXError
        If `G` is not connected, or contains no nodes,
        or `nodeA` is not in `G` or `nodeB` is not in `G`.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> round(nx.resistance_distance(G, 1, 3), 10)
    0.625

    Notes
    -----
    The implementation is based on Theorem A in [2]_. Self-loops are ignored.
    Multi-edges are contracted in one edge with weight equal to the harmonic sum of the weights.

    References
    ----------
    .. [1] Wikipedia
       "Resistance distance."
       https://en.wikipedia.org/wiki/Resistance_distance
    .. [2] D. J. Klein and M. Randic.
        Resistance distance.
        J. of Math. Chem. 12:81-95, 1993.
    """
    import numpy as np

    if len(G) == 0:
        raise nx.NetworkXError("Graph G must contain at least one node.")
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph G must be strongly connected.")
    if nodeA is not None and nodeA not in G:
        raise nx.NetworkXError("Node A is not in graph G.")
    if nodeB is not None and nodeB not in G:
        raise nx.NetworkXError("Node B is not in graph G.")

    G = G.copy()
    node_list = list(G)

    # Invert weights
    if invert_weight and weight is not None:
        if G.is_multigraph():
            for u, v, k, d in G.edges(keys=True, data=True):
                d[weight] = 1 / d[weight]
        else:
            for u, v, d in G.edges(data=True):
                d[weight] = 1 / d[weight]

    # Compute resistance distance using the Pseudo-inverse of the Laplacian
    # Self-loops are ignored
    L = nx.laplacian_matrix(G, weight=weight).todense()
    Linv = np.linalg.pinv(L, hermitian=True)

    # Return relevant distances
    if nodeA is not None and nodeB is not None:
        i = node_list.index(nodeA)
        j = node_list.index(nodeB)
        return Linv.item(i, i) + Linv.item(j, j) - Linv.item(i, j) - Linv.item(j, i)

    elif nodeA is not None:
        i = node_list.index(nodeA)
        d = {}
        for n in G:
            j = node_list.index(n)
            d[n] = Linv.item(i, i) + Linv.item(j, j) - Linv.item(i, j) - Linv.item(j, i)
        return d

    elif nodeB is not None:
        j = node_list.index(nodeB)
        d = {}
        for n in G:
            i = node_list.index(n)
            d[n] = Linv.item(i, i) + Linv.item(j, j) - Linv.item(i, j) - Linv.item(j, i)
        return d

    else:
        d = {}
        for n in G:
            i = node_list.index(n)
            d[n] = {}
            for n2 in G:
                j = node_list.index(n2)
                d[n][n2] = (
                    Linv.item(i, i)
                    + Linv.item(j, j)
                    - Linv.item(i, j)
                    - Linv.item(j, i)
                )
        return d


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def effective_graph_resistance(G, weight=None, invert_weight=True):
    """Returns the Effective graph resistance of G.

    Also known as the Kirchhoff index.

    The effective graph resistance is defined as the sum
    of the resistance distance of every node pair in G [1]_.

    If weight is not provided, then a weight of 1 is used for all edges.

    The effective graph resistance of a disconnected graph is infinite.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    weight : string or None, optional (default=None)
       The edge data key used to compute the effective graph resistance.
       If None, then each edge has weight 1.

    invert_weight : boolean (default=True)
        Proper calculation of resistance distance requires building the
        Laplacian matrix with the reciprocal of the weight. Not required
        if the weight is already inverted. Weight cannot be zero.

    Returns
    -------
    RG : float
        The effective graph resistance of `G`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a directed graph.

    NetworkXError
        If `G` does not contain any nodes.

    Examples
    --------
