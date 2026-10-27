    meanwt = 1e-3
    costargs = (np, 1 / (dist_mtx + np.eye(dist_mtx.shape[0]) * 1e-3), meanwt, dim)

    optresult = sp.optimize.minimize(
        _kamada_kawai_costfn,
        pos_arr.ravel(),
        method="L-BFGS-B",
        args=costargs,
        jac=True,
    )

    return optresult.x.reshape((-1, dim))


def _kamada_kawai_costfn(pos_vec, np, invdist, meanweight, dim):
    # Cost-function and gradient for Kamada-Kawai layout algorithm
    nNodes = invdist.shape[0]
    pos_arr = pos_vec.reshape((nNodes, dim))

    delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    nodesep = np.linalg.norm(delta, axis=-1)
    direction = np.einsum("ijk,ij->ijk", delta, 1 / (nodesep + np.eye(nNodes) * 1e-3))

    offset = nodesep * invdist - 1.0
    offset[np.diag_indices(nNodes)] = 0

    cost = 0.5 * np.sum(offset**2)
    grad = np.einsum("ij,ij,ijk->ik", invdist, offset, direction) - np.einsum(
        "ij,ij,ijk->jk", invdist, offset, direction
    )

    # Additional parabolic term to encourage mean position to be near origin:
    sumpos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * np.sum(sumpos**2)
    grad += meanweight * sumpos

    return (cost, grad.ravel())


def spectral_layout(G, weight="weight", scale=1, center=None, dim=2):
    """Position nodes using the eigenvectors of the graph Laplacian.

    Using the unnormalized Laplacian, the layout shows possible clusters of
    nodes which are an approximation of the ratio cut. If dim is the number of
    dimensions then the positions are the entries of the dim eigenvectors
    corresponding to the ascending eigenvalues starting from the second one.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spectral_layout(G)

    Notes
    -----
    Directed graphs will be considered as undirected graphs when
    positioning the nodes.

    For larger graphs (>500 nodes) this will use the SciPy sparse
    eigenvalue solver (ARPACK).
    """
    # handle some special cases that break the eigensolvers
    import numpy as np

    G, center = _process_params(G, center, dim)

    if len(G) <= 2:
        if len(G) == 0:
            pos = np.array([])
        elif len(G) == 1:
            pos = np.array([center])
        else:
            pos = np.array([np.zeros(dim), np.array(center) * 2.0])
        return dict(zip(G, pos))
    try:
        # Sparse matrix
        if len(G) < 500:  # dense solver is faster for small graphs
            raise ValueError
        A = nx.to_scipy_sparse_array(G, weight=weight, dtype="d")
        # Symmetrize directed graphs
        if G.is_directed():
            A = A + np.transpose(A)
        pos = _sparse_spectral(A, dim)
    except (ImportError, ValueError):
        # Dense matrix
        A = nx.to_numpy_array(G, weight=weight)
        # Symmetrize directed graphs
        if G.is_directed():
            A += A.T
        pos = _spectral(A, dim)

    pos = rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    return pos


def _spectral(A, dim=2):
    # Input adjacency matrix A
    # Uses dense eigenvalue solver from numpy
    import numpy as np

    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "spectral() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    # form Laplacian matrix where D is diagonal of degrees
    D = np.identity(nnodes, dtype=A.dtype) * np.sum(A, axis=1)
    L = D - A

    eigenvalues, eigenvectors = np.linalg.eig(L)
    # sort and keep smallest nonzero
    index = np.argsort(eigenvalues)[1 : dim + 1]  # 0 index is zero eigenvalue
    return np.real(eigenvectors[:, index])


def _sparse_spectral(A, dim=2):
    # Input adjacency matrix A
    # Uses sparse eigenvalue solver from scipy
    # Could use multilevel methods here, see Koren "On spectral graph drawing"
    import numpy as np
    import scipy as sp

    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "sparse_spectral() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    # form Laplacian matrix
    # TODO: Rm csr_array wrapper in favor of spdiags array constructor when available
    D = sp.sparse.csr_array(sp.sparse.spdiags(A.sum(axis=1), 0, nnodes, nnodes))
    L = D - A

    k = dim + 1
    # number of Lanczos vectors for ARPACK solver.What is the right scaling?
    ncv = max(2 * k + 1, int(np.sqrt(nnodes)))
    # return smallest k eigenvalues and eigenvectors
    eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(L, k, which="SM", ncv=ncv)
    index = np.argsort(eigenvalues)[1:k]  # 0 index is zero eigenvalue
    return np.real(eigenvectors[:, index])


def planar_layout(G, scale=1, center=None, dim=2):
    """Position nodes without edge intersections.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G. If G is of type
        nx.PlanarEmbedding, the positions are selected accordingly.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Raises
    ------
    NetworkXException
        If G is not planar

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.planar_layout(G)
    """
    import numpy as np

    if dim != 2:
        raise ValueError("can only handle 2 dimensions")

    G, center = _process_params(G, center, dim)

    if len(G) == 0:
        return {}

    if isinstance(G, nx.PlanarEmbedding):
        embedding = G
    else:
        is_planar, embedding = nx.check_planarity(G)
        if not is_planar:
            raise nx.NetworkXException("G is not planar.")
    pos = nx.combinatorial_embedding_to_pos(embedding)
    node_list = list(embedding)
    pos = np.vstack([pos[x] for x in node_list])
    pos = pos.astype(np.float64)
    pos = rescale_layout(pos, scale=scale) + center
    return dict(zip(node_list, pos))


def spiral_layout(G, scale=1, center=None, dim=2, resolution=0.35, equidistant=False):
    """Position nodes in a spiral layout.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.
    scale : number (default: 1)
        Scale factor for positions.
    center : array-like or None
        Coordinate pair around which to center the layout.
    dim : int, default=2
        Dimension of layout, currently only dim=2 is supported.
        Other dimension values result in a ValueError.
    resolution : float, default=0.35
        The compactness of the spiral layout returned.
        Lower values result in more compressed spiral layouts.
    equidistant : bool, default=False
        If True, nodes will be positioned equidistant from each other
        by decreasing angle further from center.
        If False, nodes will be positioned at equal angles
        from each other by increasing separation further from center.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Raises
    ------
    ValueError
        If dim != 2

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spiral_layout(G)
    >>> nx.draw(G, pos=pos)

    Notes
    -----
    This algorithm currently only works in two dimensions.

    """
    import numpy as np

    if dim != 2:
        raise ValueError("can only handle 2 dimensions")

    G, center = _process_params(G, center, dim)

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G): center}

    pos = []
    if equidistant:
        chord = 1
        step = 0.5
        theta = resolution
        theta += chord / (step * theta)
        for _ in range(len(G)):
            r = step * theta
            theta += chord / r
            pos.append([np.cos(theta) * r, np.sin(theta) * r])

    else:
        dist = np.arange(len(G), dtype=float)
        angle = resolution * dist
        pos = np.transpose(dist * np.array([np.cos(angle), np.sin(angle)]))

    pos = rescale_layout(np.array(pos), scale=scale) + center

    pos = dict(zip(G, pos))

    return pos


def multipartite_layout(G, subset_key="subset", align="vertical", scale=1, center=None):
    """Position nodes in layers of straight lines.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    subset_key : string or dict (default='subset')
        If a string, the key of node data in G that holds the node subset.
        If a dict, keyed by layer number to the nodes in that layer/subset.

    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.complete_multipartite_graph(28, 16, 10)
    >>> pos = nx.multipartite_layout(G)

    or use a dict to provide the layers of the layout

    >>> G = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
    >>> layers = {"a": [0], "b": [1], "c": [2, 3], "d": [4]}
    >>> pos = nx.multipartite_layout(G, subset_key=layers)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    Network does not need to be a complete multipartite graph. As long as nodes
    have subset_key data, they will be placed in the corresponding layers.

    """
    import numpy as np

    if align not in ("vertical", "horizontal"):
        msg = "align must be either vertical or horizontal."
        raise ValueError(msg)

    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}

    try:
        # check if subset_key is dict-like
        if len(G) != sum(len(nodes) for nodes in subset_key.values()):
            raise nx.NetworkXError(
                "all nodes must be in one subset of `subset_key` dict"
            )
    except AttributeError:
        # subset_key is not a dict, hence a string
        node_to_subset = nx.get_node_attributes(G, subset_key)
        if len(node_to_subset) != len(G):
            raise nx.NetworkXError(
                f"all nodes need a subset_key attribute: {subset_key}"
            )
        subset_key = nx.utils.groups(node_to_subset)

    # Sort by layer, if possible
    try:
        layers = dict(sorted(subset_key.items()))
    except TypeError:
        layers = subset_key

    pos = None
    nodes = []
    width = len(layers)
    for i, layer in enumerate(layers.values()):
        height = len(layer)
        xs = np.repeat(i, height)
        ys = np.arange(0, height, dtype=float)
        offset = ((width - 1) / 2, (height - 1) / 2)
        layer_pos = np.column_stack([xs, ys]) - offset
        if pos is None:
            pos = layer_pos
        else:
            pos = np.concatenate([pos, layer_pos])
        nodes.extend(layer)
    pos = rescale_layout(pos, scale=scale) + center
    if align == "horizontal":
        pos = pos[:, ::-1]  # swap x and y coords
    pos = dict(zip(nodes, pos))
    return pos


@np_random_state("seed")
def arf_layout(
    G,
    pos=None,
    scaling=1,
    a=1.1,
    etol=1e-6,
    dt=1e-3,
    max_iter=1000,
    *,
    seed=None,
):
    """Arf layout for networkx

    The attractive and repulsive forces (arf) layout [1]
    improves the spring layout in three ways. First, it
    prevents congestion of highly connected nodes due to
    strong forcing between nodes. Second, it utilizes the
    layout space more effectively by preventing large gaps
    that spring layout tends to create. Lastly, the arf
    layout represents symmetries in the layout better than
    the default spring layout.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Networkx graph.
    pos : dict
        Initial  position of  the nodes.  If set  to None  a
        random layout will be used.
    scaling : float
        Scales the radius of the circular layout space.
    a : float
        Strength of springs between connected nodes. Should be larger than 1. The greater a, the clearer the separation ofunconnected sub clusters.
    etol : float
        Gradient sum of spring forces must be larger than `etol` before successful termination.
    dt : float
        Time step for force differential equation simulations.
    max_iter : int
        Max iterations before termination of the algorithm.
    seed : int, RandomState instance or None  optional (default=None)
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.

    References
    .. [1] "Self-Organization Applied to Dynamic Network Layout", M. Geipel,
            International Journal of Modern Physics C, 2007, Vol 18, No 10, pp. 1537-1549.
            https://doi.org/10.1142/S0129183107011558 https://arxiv.org/abs/0704.1748

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.grid_graph((5, 5))
    >>> pos = nx.arf_layout(G)

    """
    import warnings

    import numpy as np

    if a <= 1:
        msg = "The parameter a should be larger than 1"
        raise ValueError(msg)

    pos_tmp = nx.random_layout(G, seed=seed)
    if pos is None:
        pos = pos_tmp
    else:
        for node in G.nodes():
            if node not in pos:
                pos[node] = pos_tmp[node].copy()

    # Initialize spring constant matrix
    N = len(G)
    # No nodes no computation
    if N == 0:
        return pos

    # init force of springs
    K = np.ones((N, N)) - np.eye(N)
    node_order = {node: i for i, node in enumerate(G)}
    for x, y in G.edges():
        if x != y:
            idx, jdx = (node_order[i] for i in (x, y))
            K[idx, jdx] = a

    # vectorize values
    p = np.asarray(list(pos.values()))

    # equation 10 in [1]
    rho = scaling * np.sqrt(N)

    # looping variables
    error = etol + 1
    n_iter = 0
    while error > etol:
        diff = p[:, np.newaxis] - p[np.newaxis]
        A = np.linalg.norm(diff, axis=-1)[..., np.newaxis]
        # attraction_force - repulsions force
        # suppress nans due to division; caused by diagonal set to zero.
        # Does not affect the computation due to nansum
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            change = K[..., np.newaxis] * diff - rho / A * diff
        change = np.nansum(change, axis=0)
        p += change * dt

        error = np.linalg.norm(change, axis=-1).sum()
        if n_iter > max_iter:
            break
        n_iter += 1
    return dict(zip(G.nodes(), p))


@np_random_state("seed")
def forceatlas2_layout(
    G,
    pos=None,
    *,
    max_iter=100,
    jitter_tolerance=1.0,
    scaling_ratio=2.0,
    gravity=1.0,
    distributed_action=False,
    strong_gravity=False,
    node_mass=None,
    node_size=None,
    weight=None,
    dissuade_hubs=False,
    linlog=False,
    seed=None,
    dim=2,
):
    """Position nodes using the ForceAtlas2 force-directed layout algorithm.

    This function applies the ForceAtlas2 layout algorithm [1]_ to a NetworkX graph,
    positioning the nodes in a way that visually represents the structure of the graph.
    The algorithm uses physical simulation to minimize the energy of the system,
    resulting in a more readable layout.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph to be laid out.
    pos : dict or None, optional
        Initial positions of the nodes. If None, random initial positions are used.
    max_iter : int (default: 100)
        Number of iterations for the layout optimization.
    jitter_tolerance : float (default: 1.0)
        Controls the tolerance for adjusting the speed of layout generation.
    scaling_ratio : float (default: 2.0)
        Determines the scaling of attraction and repulsion forces.
    distributed_attraction : bool (default: False)
        Distributes the attraction force evenly among nodes.
    strong_gravity : bool (default: False)
        Applies a strong gravitational pull towards the center.
    node_mass : dict or None, optional
        Maps nodes to their masses, influencing the attraction to other nodes.
    node_size : dict or None, optional
        Maps nodes to their sizes, preventing crowding by creating a halo effect.
    dissuade_hubs : bool (default: False)
        Prevents the clustering of hub nodes.
    linlog : bool (default: False)
        Uses logarithmic attraction instead of linear.
    seed : int, RandomState instance or None  optional (default=None)
        Used only for the initial positions in the algorithm.
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.
    dim : int (default: 2)
        Sets the dimensions for the layout. Ignored if `pos` is provided.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.florentine_families_graph()
    >>> pos = nx.forceatlas2_layout(G)
    >>> nx.draw(G, pos=pos)

    References
    ----------
    .. [1] Jacomy, M., Venturini, T., Heymann, S., & Bastian, M. (2014).
           ForceAtlas2, a continuous graph layout algorithm for handy network
           visualization designed for the Gephi software. PloS one, 9(6), e98679.
           https://doi.org/10.1371/journal.pone.0098679
    """
    import numpy as np

    if len(G) == 0:
        return {}
    # parse optional pos positions
    if pos is None:
        pos = nx.random_layout(G, dim=dim, seed=seed)
        pos_arr = np.array(list(pos.values()))
    else:
        # set default node interval within the initial pos values
        pos_init = np.array(list(pos.values()))
        max_pos = pos_init.max(axis=0)
        min_pos = pos_init.min(axis=0)
        dim = max_pos.size
        pos_arr = min_pos + seed.rand(len(G), dim) * (max_pos - min_pos)
        for idx, node in enumerate(G):
            if node in pos:
                pos_arr[idx] = pos[node].copy()

    mass = np.zeros(len(G))
    size = np.zeros(len(G))

    # Only adjust for size when the users specifies size other than default (1)
    adjust_sizes = False
    if node_size is None:
        node_size = {}
    else:
        adjust_sizes = True

    if node_mass is None:
        node_mass = {}

    for idx, node in enumerate(G):
        mass[idx] = node_mass.get(node, G.degree(node) + 1)
        size[idx] = node_size.get(node, 1)

    n = len(G)
    gravities = np.zeros((n, dim))
    attraction = np.zeros((n, dim))
    repulsion = np.zeros((n, dim))
    A = nx.to_numpy_array(G, weight=weight)

    def estimate_factor(n, swing, traction, speed, speed_efficiency, jitter_tolerance):
        """Computes the scaling factor for the force in the ForceAtlas2 layout algorithm.

        This   helper  function   adjusts   the  speed   and
        efficiency  of the  layout generation  based on  the
        current state of  the system, such as  the number of
        nodes, current swing, and traction forces.

        Parameters
        ----------
        n : int
            Number of nodes in the graph.
        swing : float
            The current swing, representing the oscillation of the nodes.
        traction : float
            The current traction force, representing the attraction between nodes.
        speed : float
            The current speed of the layout generation.
        speed_efficiency : float
            The efficiency of the current speed, influencing how fast the layout converges.
        jitter_tolerance : float
            The tolerance for jitter, affecting how much speed adjustment is allowed.

        Returns
        -------
        tuple
            A tuple containing the updated speed and speed efficiency.

        Notes
        -----
        This function is a part of the ForceAtlas2 layout algorithm and is used to dynamically adjust the
        layout parameters to achieve an optimal and stable visualization.

        """
        import numpy as np

        # estimate jitter
        opt_jitter = 0.05 * np.sqrt(n)
        min_jitter = np.sqrt(opt_jitter)
        max_jitter = 10
        min_speed_efficiency = 0.05

        other = min(max_jitter, opt_jitter * traction / n**2)
        jitter = jitter_tolerance * max(min_jitter, other)

        if swing / traction > 2.0:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.5
            jitter = max(jitter, jitter_tolerance)
        if swing == 0:
            target_speed = np.inf
        else:
            target_speed = jitter * speed_efficiency * traction / swing

        if swing > jitter * traction:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.7
        elif speed < 1000:
            speed_efficiency *= 1.3

        max_rise = 0.5
        speed = speed + min(target_speed - speed, max_rise * speed)
        return speed, speed_efficiency

    speed = 1
    speed_efficiency = 1
    swing = 1
    traction = 1
    for _ in range(max_iter):
        # compute pairwise difference
        diff = pos_arr[:, None] - pos_arr[None]
        # compute pairwise distance
        distance = np.linalg.norm(diff, axis=-1)

        # linear attraction
        if linlog:
            attraction = -np.log(1 + distance) / distance
            np.fill_diagonal(attraction, 0)
            attraction = np.einsum("ij, ij -> ij", attraction, A)
            attraction = np.einsum("ijk, ij -> ik", diff, attraction)

        else:
            attraction = -np.einsum("ijk, ij -> ik", diff, A)

        if distributed_action:
            attraction /= mass[:, None]

        # repulsion
        tmp = mass[:, None] @ mass[None]
        if adjust_sizes:
            distance += -size[:, None] - size[None]

        d2 = distance**2
        # remove self-interaction
        np.fill_diagonal(tmp, 0)
        np.fill_diagonal(d2, 1)
        factor = (tmp / d2) * scaling_ratio
        repulsion = np.einsum("ijk, ij -> ik", diff, factor)

        # gravity
        gravities = (
            -gravity
            * mass[:, None]
            * pos_arr
            / np.linalg.norm(pos_arr, axis=-1)[:, None]
        )

        if strong_gravity:
            gravities *= np.linalg.norm(pos_arr, axis=-1)[:, None]
        # total forces
        update = attraction + repulsion + gravities

        # compute total swing and traction
        swing += (mass * np.linalg.norm(pos_arr - update, axis=-1)).sum()
        traction += (0.5 * mass * np.linalg.norm(pos_arr + update, axis=-1)).sum()

        speed, speed_efficiency = estimate_factor(
            n,
            swing,
            traction,
            speed,
            speed_efficiency,
            jitter_tolerance,
        )

        # update pos
        if adjust_sizes:
            swinging = mass * np.linalg.norm(update, axis=-1)
            factor = 0.1 * speed / (1 + np.sqrt(speed * swinging))
            df = np.linalg.norm(update, axis=-1)
            factor = np.minimum(factor * df, 10.0 * np.ones(df.shape)) / df
        else:
            swinging = mass * np.linalg.norm(update, axis=-1)
            factor = speed / (1 + np.sqrt(speed * swinging))

        pos_arr += update * factor[:, None]
        if abs((update * factor[:, None]).sum()) < 1e-10:
            break

    return dict(zip(G, pos_arr))


def rescale_layout(pos, scale=1):
    """Returns scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.

    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.

    See Also
    --------
    rescale_layout_dict
    """
    import numpy as np

    # Find max length over all dimensions
    pos -= pos.mean(axis=0)
    lim = np.abs(pos).max()  # max coordinate for all axes
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        pos *= scale / lim
    return pos


def rescale_layout_dict(pos, scale=1):
    """Return a dictionary of scaled positions keyed by node

    Parameters
    ----------
    pos : A dictionary of positions keyed by node

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : A dictionary of positions keyed by node

    Examples
    --------
    >>> import numpy as np
    >>> pos = {0: np.array((0, 0)), 1: np.array((1, 1)), 2: np.array((0.5, 0.5))}
    >>> nx.rescale_layout_dict(pos)
    {0: array([-1., -1.]), 1: array([1., 1.]), 2: array([0., 0.])}

    >>> pos = {0: np.array((0, 0)), 1: np.array((-1, 1)), 2: np.array((-0.5, 0.5))}
    >>> nx.rescale_layout_dict(pos, scale=2)
    {0: array([ 2., -2.]), 1: array([-2.,  2.]), 2: array([0., 0.])}

    See Also
    --------
    rescale_layout
    """
    import numpy as np

    if not pos:  # empty_graph
        return {}
    pos_v = np.array(list(pos.values()))
    pos_v = rescale_layout(pos_v, scale=scale)
    return dict(zip(pos, pos_v))


def bfs_layout(G, start, *, align="vertical", scale=1, center=None):
    """Position nodes according to breadth-first search algorithm.

    Parameters
    ----------
    G : NetworkX graph
        A position will be assigned to every node in G.

    start : node in `G`
        Starting node for bfs

    center : array-like or None
        Coordinate pair around which to center the layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.bfs_layout(G, 0)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    G, center = _process_params(G, center, 2)

    # Compute layers with BFS
    layers = dict(enumerate(nx.bfs_layers(G, start)))

    if len(G) != sum(len(nodes) for nodes in layers.values()):
        raise nx.NetworkXError(
            "bfs_layout didn't include all nodes. Perhaps use input graph:\n"
            "        G.subgraph(nx.node_connected_component(G, start))"
        )

    # Compute node positions with multipartite_layout
    return multipartite_layout(
        G, subset_key=layers, align=align, scale=scale, center=center
    )
<|endoftext|>"""Functions for computing and verifying matchings in a graph."""

from collections import Counter
from itertools import combinations, repeat

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "is_matching",
    "is_maximal_matching",
    "is_perfect_matching",
    "max_weight_matching",
    "min_weight_matching",
    "maximal_matching",
]


@not_implemented_for("multigraph")
@not_implemented_for("directed")
@nx._dispatchable
def maximal_matching(G):
    r"""Find a maximal matching in the graph.

    A matching is a subset of edges in which no node occurs more than once.
    A maximal matching cannot add more edges and still be a matching.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    matching : set
        A maximal matching of the graph.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])
    >>> sorted(nx.maximal_matching(G))
    [(1, 2), (3, 5)]

    Notes
    -----
    The algorithm greedily selects a maximal matching M of the graph G
    (i.e. no superset of M exists). It runs in $O(|E|)$ time.
    """
    matching = set()
    nodes = set()
    for edge in G.edges():
        # If the edge isn't covered, add it to the matching
        # then remove neighborhood of u and v from consideration.
        u, v = edge
        if u not in nodes and v not in nodes and u != v:
            matching.add(edge)
            nodes.update(edge)
    return matching


def matching_dict_to_set(matching):
    """Converts matching dict format to matching set format

    Converts a dictionary representing a matching (as returned by
    :func:`max_weight_matching`) to a set representing a matching (as
    returned by :func:`maximal_matching`).

    In the definition of maximal matching adopted by NetworkX,
    self-loops are not allowed, so the provided dictionary is expected
    to never have any mapping from a key to itself. However, the
    dictionary is expected to have mirrored key/value pairs, for
    example, key ``u`` with value ``v`` and key ``v`` with value ``u``.

    """
    edges = set()
    for edge in matching.items():
        u, v = edge
        if (v, u) in edges or edge in edges:
            continue
        if u == v:
            raise nx.NetworkXError(f"Selfloops cannot appear in matchings {edge}")
        edges.add(edge)
    return edges


@nx._dispatchable
def is_matching(G, matching):
    """Return True if ``matching`` is a valid matching of ``G``

    A *matching* in a graph is a set of edges in which no two distinct
    edges share a common endpoint. Each node is incident to at most one
    edge in the matching. The edges are said to be independent.

    Parameters
    ----------
    G : NetworkX graph

    matching : dict or set
        A dictionary or set representing a matching. If a dictionary, it
        must have ``matching[u] == v`` and ``matching[v] == u`` for each
        edge ``(u, v)`` in the matching. If a set, it must have elements
        of the form ``(u, v)``, where ``(u, v)`` is an edge in the
        matching.

    Returns
    -------
    bool
        Whether the given set or dictionary represents a valid matching
        in the graph.

    Raises
    ------
    NetworkXError
        If the proposed matching has an edge to a node not in G.
        Or if the matching is not a collection of 2-tuple edges.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])
    >>> nx.is_maximal_matching(G, {1: 3, 2: 4})  # using dict to represent matching
    True

    >>> nx.is_matching(G, {(1, 3), (2, 4)})  # using set to represent matching
    True

    """
    if isinstance(matching, dict):
        matching = matching_dict_to_set(matching)

    nodes = set()
    for edge in matching:
        if len(edge) != 2:
            raise nx.NetworkXError(f"matching has non-2-tuple edge {edge}")
        u, v = edge
        if u not in G or v not in G:
            raise nx.NetworkXError(f"matching contains edge {edge} with node not in G")
        if u == v:
            return False
        if not G.has_edge(u, v):
            return False
        if u in nodes or v in nodes:
            return False
        nodes.update(edge)
    return True


@nx._dispatchable
def is_maximal_matching(G, matching):
    """Return True if ``matching`` is a maximal matching of ``G``

    A *maximal matching* in a graph is a matching in which adding any
    edge would cause the set to no longer be a valid matching.

    Parameters
    ----------
    G : NetworkX graph

    matching : dict or set
        A dictionary or set representing a matching. If a dictionary, it
        must have ``matching[u] == v`` and ``matching[v] == u`` for each
        edge ``(u, v)`` in the matching. If a set, it must have elements
        of the form ``(u, v)``, where ``(u, v)`` is an edge in the
        matching.

    Returns
    -------
    bool
        Whether the given set or dictionary represents a valid maximal
        matching in the graph.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (3, 5)])
    >>> nx.is_maximal_matching(G, {(1, 2), (3, 4)})
    True

    """
    if isinstance(matching, dict):
        matching = matching_dict_to_set(matching)
    # If the given set is not a matching, then it is not a maximal matching.
    edges = set()
    nodes = set()
    for edge in matching:
        if len(edge) != 2:
            raise nx.NetworkXError(f"matching has non-2-tuple edge {edge}")
        u, v = edge
        if u not in G or v not in G:
            raise nx.NetworkXError(f"matching contains edge {edge} with node not in G")
        if u == v:
            return False
        if not G.has_edge(u, v):
            return False
        if u in nodes or v in nodes:
            return False
        nodes.update(edge)
        edges.add(edge)
        edges.add((v, u))
    # A matching is maximal if adding any new edge from G to it
    # causes the resulting set to match some node twice.
    # Be careful to check for adding selfloops
    for u, v in G.edges:
        if (u, v) not in edges:
            # could add edge (u, v) to edges and have a bigger matching
            if u not in nodes and v not in nodes and u != v:
                return False
    return True


@nx._dispatchable
def is_perfect_matching(G, matching):
    """Return True if ``matching`` is a perfect matching for ``G``

    A *perfect matching* in a graph is a matching in which exactly one edge
    is incident upon each vertex.

    Parameters
    ----------
    G : NetworkX graph

    matching : dict or set
        A dictionary or set representing a matching. If a dictionary, it
        must have ``matching[u] == v`` and ``matching[v] == u`` for each
        edge ``(u, v)`` in the matching. If a set, it must have elements
        of the form ``(u, v)``, where ``(u, v)`` is an edge in the
        matching.

    Returns
    -------
    bool
        Whether the given set or dictionary represents a valid perfect
        matching in the graph.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6)])
    >>> my_match = {1: 2, 3: 5, 4: 6}
    >>> nx.is_perfect_matching(G, my_match)
    True

    """
    if isinstance(matching, dict):
        matching = matching_dict_to_set(matching)

    nodes = set()
    for edge in matching:
        if len(edge) != 2:
            raise nx.NetworkXError(f"matching has non-2-tuple edge {edge}")
        u, v = edge
        if u not in G or v not in G:
            raise nx.NetworkXError(f"matching contains edge {edge} with node not in G")
        if u == v:
            return False
        if not G.has_edge(u, v):
            return False
        if u in nodes or v in nodes:
            return False
        nodes.update(edge)
    return len(nodes) == len(G)


@not_implemented_for("multigraph")
@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def min_weight_matching(G, weight="weight"):
    """Computing a minimum-weight maximal matching of G.

    Use the maximum-weight algorithm with edge weights subtracted
    from the maximum weight of all edges.

    A matching is a subset of edges in which no node occurs more than once.
    The weight of a matching is the sum of the weights of its edges.
    A maximal matching cannot add more edges and still be a matching.
    The cardinality of a matching is the number of matched edges.

    This method replaces the edge weights with 1 plus the maximum edge weight
    minus the original edge weight.

    new_weight = (max_weight + 1) - edge_weight

    then runs :func:`max_weight_matching` with the new weights.
    The max weight matching with these new weights corresponds
    to the min weight matching using the original weights.
    Adding 1 to the max edge weight keeps all edge weights positive
    and as integers if they started as integers.

    You might worry that adding 1 to each weight would make the algorithm
    favor matchings with more edges. But we use the parameter
    `maxcardinality=True` in `max_weight_matching` to ensure that the
    number of edges in the competing matchings are the same and thus
    the optimum does not change due to changes in the number of edges.

    Read the documentation of `max_weight_matching` for more information.

    Parameters
    ----------
    G : NetworkX graph
      Undirected graph

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.
       If key not found, uses 1 as weight.

    Returns
    -------
    matching : set
        A minimal weight matching of the graph.

    See Also
    --------
    max_weight_matching
    """
    if len(G.edges) == 0:
        return max_weight_matching(G, maxcardinality=True, weight=weight)
    G_edges = G.edges(data=weight, default=1)
    max_weight = 1 + max(w for _, _, w in G_edges)
    InvG = nx.Graph()
    edges = ((u, v, max_weight - w) for u, v, w in G_edges)
    InvG.add_weighted_edges_from(edges, weight=weight)
    return max_weight_matching(InvG, maxcardinality=True, weight=weight)


@not_implemented_for("multigraph")
@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def max_weight_matching(G, maxcardinality=False, weight="weight"):
    """Compute a maximum-weighted matching of G.

    A matching is a subset of edges in which no node occurs more than once.
    The weight of a matching is the sum of the weights of its edges.
    A maximal matching cannot add more edges and still be a matching.
    The cardinality of a matching is the number of matched edges.

    Parameters
    ----------
    G : NetworkX graph
      Undirected graph

    maxcardinality: bool, optional (default=False)
       If maxcardinality is True, compute the maximum-cardinality matching
       with maximum weight among all maximum-cardinality matchings.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.
       If key not found, uses 1 as weight.


    Returns
    -------
    matching : set
        A maximal matching of the graph.

     Examples
    --------
    >>> G = nx.Graph()
    >>> edges = [(1, 2, 6), (1, 3, 2), (2, 3, 1), (2, 4, 7), (3, 5, 9), (4, 5, 3)]
    >>> G.add_weighted_edges_from(edges)
    >>> sorted(nx.max_weight_matching(G))
    [(2, 4), (5, 3)]

    Notes
    -----
    If G has edges with weight attributes the edge data are used as
    weight values else the weights are assumed to be 1.

    This function takes time O(number_of_nodes ** 3).

    If all edge weights are integers, the algorithm uses only integer
    computations.  If floating point weights are used, the algorithm
    could return a slightly suboptimal matching due to numeric
    precision errors.

    This method is based on the "blossom" method for finding augmenting
    paths and the "primal-dual" method for finding a matching of maximum
    weight, both methods invented by Jack Edmonds [1]_.

    Bipartite graphs can also be matched using the functions present in
    :mod:`networkx.algorithms.bipartite.matching`.

    References
    ----------
    .. [1] "Efficient Algorithms for Finding Maximum Matching in Graphs",
       Zvi Galil, ACM Computing Surveys, 1986.
    """
    #
    # The algorithm is taken from "Efficient Algorithms for Finding Maximum
    # Matching in Graphs" by Zvi Galil, ACM Computing Surveys, 1986.
    # It is based on the "blossom" method for finding augmenting paths and
    # the "primal-dual" method for finding a matching of maximum weight, both
    # methods invented by Jack Edmonds.
    #
    # A C program for maximum weight matching by Ed Rothberg was used
    # extensively to validate this new code.
    #
    # Many terms used in the code comments are explained in the paper
    # by Galil. You will probably need the paper to make sense of this code.
    #

    class NoNode:
        """Dummy value which is different from any node."""

    class Blossom:
        """Representation of a non-trivial blossom or sub-blossom."""

        __slots__ = ["childs", "edges", "mybestedges"]

        # b.childs is an ordered list of b's sub-blossoms, starting with
        # the base and going round the blossom.

        # b.edges is the list of b's connecting edges, such that
        # b.edges[i] = (v, w) where v is a vertex in b.childs[i]
        # and w is a vertex in b.childs[wrap(i+1)].

        # If b is a top-level S-blossom,
        # b.mybestedges is a list of least-slack edges to neighboring
        # S-blossoms, or None if no such list has been computed yet.
        # This is used for efficient computation of delta3.

        # Generate the blossom's leaf vertices.
        def leaves(self):
            stack = [*self.childs]
            while stack:
                t = stack.pop()
                if isinstance(t, Blossom):
                    stack.extend(t.childs)
                else:
                    yield t

    # Get a list of vertices.
    gnodes = list(G)
    if not gnodes:
        return set()  # don't bother with empty graphs

    # Find the maximum edge weight.
    maxweight = 0
    allinteger = True
    for i, j, d in G.edges(data=True):
        wt = d.get(weight, 1)
        if i != j and wt > maxweight:
            maxweight = wt
        allinteger = allinteger and (str(type(wt)).split("'")[1] in ("int", "long"))

    # If v is a matched vertex, mate[v] is its partner vertex.
    # If v is a single vertex, v does not occur as a key in mate.
    # Initially all vertices are single; updated during augmentation.
    mate = {}

    # If b is a top-level blossom,
    # label.get(b) is None if b is unlabeled (free),
    #                 1 if b is an S-blossom,
    #                 2 if b is a T-blossom.
    # The label of a vertex is found by looking at the label of its top-level
    # containing blossom.
    # If v is a vertex inside a T-blossom, label[v] is 2 iff v is reachable
    # from an S-vertex outside the blossom.
    # Labels are assigned during a stage and reset after each augmentation.
    label = {}

    # If b is a labeled top-level blossom,
    # labeledge[b] = (v, w) is the edge through which b obtained its label
    # such that w is a vertex in b, or None if b's base vertex is single.
    # If w is a vertex inside a T-blossom and label[w] == 2,
    # labeledge[w] = (v, w) is an edge through which w is reachable from
    # outside the blossom.
    labeledge = {}

    # If v is a vertex, inblossom[v] is the top-level blossom to which v
    # belongs.
    # If v is a top-level vertex, inblossom[v] == v since v is itself
    # a (trivial) top-level blossom.
    # Initially all vertices are top-level trivial blossoms.
    inblossom = dict(zip(gnodes, gnodes))

    # If b is a sub-blossom,
    # blossomparent[b] is its immediate parent (sub-)blossom.
    # If b is a top-level blossom, blossomparent[b] is None.
    blossomparent = dict(zip(gnodes, repeat(None)))

    # If b is a (sub-)blossom,
    # blossombase[b] is its base VERTEX (i.e. recursive sub-blossom).
    blossombase = dict(zip(gnodes, gnodes))

    # If w is a free vertex (or an unreached vertex inside a T-blossom),
    # bestedge[w] = (v, w) is the least-slack edge from an S-vertex,
    # or None if there is no such edge.
    # If b is a (possibly trivial) top-level S-blossom,
    # bestedge[b] = (v, w) is the least-slack edge to a different S-blossom
    # (v inside b), or None if there is no such edge.
    # This is used for efficient computation of delta2 and delta3.
    bestedge = {}

    # If v is a vertex,
    # dualvar[v] = 2 * u(v) where u(v) is the v's variable in the dual
    # optimization problem (if all edge weights are integers, multiplication
    # by two ensures that all values remain integers throughout the algorithm).
    # Initially, u(v) = maxweight / 2.
    dualvar = dict(zip(gnodes, repeat(maxweight)))

    # If b is a non-trivial blossom,
    # blossomdual[b] = z(b) where z(b) is b's variable in the dual
    # optimization problem.
    blossomdual = {}

    # If (v, w) in allowedge or (w, v) in allowedg, then the edge
    # (v, w) is known to have zero slack in the optimization problem;
    # otherwise the edge may or may not have zero slack.
    allowedge = {}

    # Queue of newly discovered S-vertices.
    queue = []

    # Return 2 * slack of edge (v, w) (does not work inside blossoms).
    def slack(v, w):
        return dualvar[v] + dualvar[w] - 2 * G[v][w].get(weight, 1)

    # Assign label t to the top-level blossom containing vertex w,
    # coming through an edge from vertex v.
    def assignLabel(w, t, v):
        b = inblossom[w]
        assert label.get(w) is None and label.get(b) is None
        label[w] = label[b] = t
        if v is not None:
            labeledge[w] = labeledge[b] = (v, w)
        else:
            labeledge[w] = labeledge[b] = None
        bestedge[w] = bestedge[b] = None
        if t == 1:
            # b became an S-vertex/blossom; add it(s vertices) to the queue.
            if isinstance(b, Blossom):
                queue.extend(b.leaves())
            else:
                queue.append(b)
        elif t == 2:
            # b became a T-vertex/blossom; assign label S to its mate.
            # (If b is a non-trivial blossom, its base is the only vertex
            # with an external mate.)
            base = blossombase[b]
            assignLabel(mate[base], 1, base)

    # Trace back from vertices v and w to discover either a new blossom
    # or an augmenting path. Return the base vertex of the new blossom,
    # or NoNode if an augmenting path was found.
    def scanBlossom(v, w):
        # Trace back from v and w, placing breadcrumbs as we go.
        path = []
        base = NoNode
        while v is not NoNode:
            # Look for a breadcrumb in v's blossom or put a new breadcrumb.
            b = inblossom[v]
            if label[b] & 4:
                base = blossombase[b]
                break
            assert label[b] == 1
            path.append(b)
            label[b] = 5
            # Trace one step back.
            if labeledge[b] is None:
                # The base of blossom b is single; stop tracing this path.
                assert blossombase[b] not in mate
                v = NoNode
            else:
                assert labeledge[b][0] == mate[blossombase[b]]
                v = labeledge[b][0]
                b = inblossom[v]
                assert label[b] == 2
                # b is a T-blossom; trace one more step back.
                v = labeledge[b][0]
            # Swap v and w so that we alternate between both paths.
            if w is not NoNode:
                v, w = w, v
        # Remove breadcrumbs.
        for b in path:
            label[b] = 1
        # Return base vertex, if we found one.
        return base

    # Construct a new blossom with given base, through S-vertices v and w.
    # Label the new blossom as S; set its dual variable to zero;
    # relabel its T-vertices to S and add them to the queue.
    def addBlossom(base, v, w):
        bb = inblossom[base]
        bv = inblossom[v]
        bw = inblossom[w]
        # Create blossom.
        b = Blossom()
        blossombase[b] = base
        blossomparent[b] = None
        blossomparent[bb] = b
        # Make list of sub-blossoms and their interconnecting edge endpoints.
        b.childs = path = []
        b.edges = edgs = [(v, w)]
        # Trace back from v to base.
        while bv != bb:
            # Add bv to the new blossom.
            blossomparent[bv] = b
            path.append(bv)
            edgs.append(labeledge[bv])
            assert label[bv] == 2 or (
                label[bv] == 1 and labeledge[bv][0] == mate[blossombase[bv]]
            )
            # Trace one step back.
            v = labeledge[bv][0]
            bv = inblossom[v]
        # Add base sub-blossom; reverse lists.
        path.append(bb)
        path.reverse()
        edgs.reverse()
        # Trace back from w to base.
        while bw != bb:
            # Add bw to the new blossom.
            blossomparent[bw] = b
            path.append(bw)
            edgs.append((labeledge[bw][1], labeledge[bw][0]))
            assert label[bw] == 2 or (
                label[bw] == 1 and labeledge[bw][0] == mate[blossombase[bw]]
            )
            # Trace one step back.
            w = labeledge[bw][0]
            bw = inblossom[w]
        # Set label to S.
        assert label[bb] == 1
        label[b] = 1
        labeledge[b] = labeledge[bb]
        # Set dual variable to zero.
        blossomdual[b] = 0
        # Relabel vertices.
        for v in b.leaves():
            if label[inblossom[v]] == 2:
                # This T-vertex now turns into an S-vertex because it becomes
                # part of an S-blossom; add it to the queue.
                queue.append(v)
            inblossom[v] = b
        # Compute b.mybestedges.
        bestedgeto = {}
        for bv in path:
            if isinstance(bv, Blossom):
                if bv.mybestedges is not None:
                    # Walk this subblossom's least-slack edges.
                    nblist = bv.mybestedges
                    # The sub-blossom won't need this data again.
                    bv.mybestedges = None
                else:
                    # This subblossom does not have a list of least-slack
                    # edges; get the information from the vertices.
                    nblist = [
                        (v, w) for v in bv.leaves() for w in G.neighbors(v) if v != w
                    ]
            else:
                nblist = [(bv, w) for w in G.neighbors(bv) if bv != w]
            for k in nblist:
                (i, j) = k
                if inblossom[j] == b:
                    i, j = j, i
                bj = inblossom[j]
                if (
                    bj != b
                    and label.get(bj) == 1
                    and ((bj not in bestedgeto) or slack(i, j) < slack(*bestedgeto[bj]))
                ):
                    bestedgeto[bj] = k
            # Forget about least-slack edge of the subblossom.
            bestedge[bv] = None
        b.mybestedges = list(bestedgeto.values())
        # Select bestedge[b].
        mybestedge = None
        bestedge[b] = None
        for k in b.mybestedges:
            kslack = slack(*k)
            if mybestedge is None or kslack < mybestslack:
                mybestedge = k
                mybestslack = kslack
        bestedge[b] = mybestedge

    # Expand the given top-level blossom.
    def expandBlossom(b, endstage):
        # This is an obnoxiously complicated recursive function for the sake of
        # a stack-transformation.  So, we hack around the complexity by using
        # a trampoline pattern.  By yielding the arguments to each recursive
        # call, we keep the actual callstack flat.

        def _recurse(b, endstage):
            # Convert sub-blossoms into top-level blossoms.
            for s in b.childs:
                blossomparent[s] = None
                if isinstance(s, Blossom):
                    if endstage and blossomdual[s] == 0:
                        # Recursively expand this sub-blossom.
                        yield s
                    else:
                        for v in s.leaves():
                            inblossom[v] = s
                else:
                    inblossom[s] = s
            # If we expand a T-blossom during a stage, its sub-blossoms must be
            # relabeled.
            if (not endstage) and label.get(b) == 2:
                # Start at the sub-blossom through which the expanding
                # blossom obtained its label, and relabel sub-blossoms untili
                # we reach the base.
                # Figure out through which sub-blossom the expanding blossom
                # obtained its label initially.
                entrychild = inblossom[labeledge[b][1]]
                # Decide in which direction we will go round the blossom.
                j = b.childs.index(entrychild)
                if j & 1:
                    # Start index is odd; go forward and wrap.
                    j -= len(b.childs)
                    jstep = 1
                else:
                    # Start index is even; go backward.
                    jstep = -1
                # Move along the blossom until we get to the base.
                v, w = labeledge[b]
                while j != 0:
                    # Relabel the T-sub-blossom.
                    if jstep == 1:
                        p, q = b.edges[j]
                    else:
                        q, p = b.edges[j - 1]
                    label[w] = None
                    label[q] = None
                    assignLabel(w, 2, v)
                    # Step to the next S-sub-blossom and note its forward edge.
                    allowedge[(p, q)] = allowedge[(q, p)] = True
                    j += jstep
                    if jstep == 1:
                        v, w = b.edges[j]
                    else:
                        w, v = b.edges[j - 1]
                    # Step to the next T-sub-blossom.
                    allowedge[(v, w)] = allowedge[(w, v)] = True
                    j += jstep
                # Relabel the base T-sub-blossom WITHOUT stepping through to
                # its mate (so don't call assignLabel).
                bw = b.childs[j]
                label[w] = label[bw] = 2
                labeledge[w] = labeledge[bw] = (v, w)
                bestedge[bw] = None
                # Continue along the blossom until we get back to entrychild.
                j += jstep
                while b.childs[j] != entrychild:
                    # Examine the vertices of the sub-blossom to see whether
                    # it is reachable from a neighboring S-vertex outside the
                    # expanding blossom.
                    bv = b.childs[j]
                    if label.get(bv) == 1:
                        # This sub-blossom just got label S through one of its
                        # neighbors; leave it be.
                        j += jstep
                        continue
                    if isinstance(bv, Blossom):
                        for v in bv.leaves():
                            if label.get(v):
                                break
                    else:
                        v = bv
                    # If the sub-blossom contains a reachable vertex, assign
                    # label T to the sub-blossom.
                    if label.get(v):
                        assert label[v] == 2
                        assert inblossom[v] == bv
                        label[v] = None
                        label[mate[blossombase[bv]]] = None
                        assignLabel(v, 2, labeledge[v][0])
                    j += jstep
            # Remove the expanded blossom entirely.
            label.pop(b, None)
            labeledge.pop(b, None)
            bestedge.pop(b, None)
            del blossomparent[b]
            del blossombase[b]
            del blossomdual[b]

        # Now, we apply the trampoline pattern.  We simulate a recursive
        # callstack by maintaining a stack of generators, each yielding a
        # sequence of function arguments.  We grow the stack by appending a call
        # to _recurse on each argument tuple, and shrink the stack whenever a
        # generator is exhausted.
        stack = [_recurse(b, endstage)]
        while stack:
            top = stack[-1]
            for s in top:
                stack.append(_recurse(s, endstage))
                break
            else:
                stack.pop()

    # Swap matched/unmatched edges over an alternating path through blossom b
    # between vertex v and the base vertex. Keep blossom bookkeeping
    # consistent.
    def augmentBlossom(b, v):
        # This is an obnoxiously complicated recursive function for the sake of
        # a stack-transformation.  So, we hack around the complexity by using
        # a trampoline pattern.  By yielding the arguments to each recursive
        # call, we keep the actual callstack flat.

        def _recurse(b, v):
            # Bubble up through the blossom tree from vertex v to an immediate
            # sub-blossom of b.
            t = v
            while blossomparent[t] != b:
                t = blossomparent[t]
            # Recursively deal with the first sub-blossom.
            if isinstance(t, Blossom):
                yield (t, v)
            # Decide in which direction we will go round the blossom.
            i = j = b.childs.index(t)
            if i & 1:
                # Start index is odd; go forward and wrap.
                j -= len(b.childs)
                jstep = 1
            else:
                # Start index is even; go backward.
                jstep = -1
            # Move along the blossom until we get to the base.
            while j != 0:
                # Step to the next sub-blossom and augment it recursively.
                j += jstep
                t = b.childs[j]
                if jstep == 1:
                    w, x = b.edges[j]
                else:
                    x, w = b.edges[j - 1]
                if isinstance(t, Blossom):
                    yield (t, w)
                # Step to the next sub-blossom and augment it recursively.
                j += jstep
                t = b.childs[j]
                if isinstance(t, Blossom):
                    yield (t, x)
                # Match the edge connecting those sub-blossoms.
                mate[w] = x
                mate[x] = w
            # Rotate the list of sub-blossoms to put the new base at the front.
            b.childs = b.childs[i:] + b.childs[:i]
            b.edges = b.edges[i:] + b.edges[:i]
            blossombase[b] = blossombase[b.childs[0]]
            assert blossombase[b] == v

        # Now, we apply the trampoline pattern.  We simulate a recursive
        # callstack by maintaining a stack of generators, each yielding a
        # sequence of function arguments.  We grow the stack by appending a call
        # to _recurse on each argument tuple, and shrink the stack whenever a
        # generator is exhausted.
        stack = [_recurse(b, v)]
        while stack:
            top = stack[-1]
            for args in top:
                stack.append(_recurse(*args))
                break
            else:
                stack.pop()

    # Swap matched/unmatched edges over an alternating path between two
    # single vertices. The augmenting path runs through S-vertices v and w.
    def augmentMatching(v, w):
        for s, j in ((v, w), (w, v)):
            # Match vertex s to vertex j. Then trace back from s
            # until we find a single vertex, swapping matched and unmatched
            # edges as we go.
            while 1:
                bs = inblossom[s]
                assert label[bs] == 1
                assert (labeledge[bs] is None and blossombase[bs] not in mate) or (
                    labeledge[bs][0] == mate[blossombase[bs]]
                )
                # Augment through the S-blossom from s to base.
                if isinstance(bs, Blossom):
                    augmentBlossom(bs, s)
                # Update mate[s]
                mate[s] = j
                # Trace one step back.
                if labeledge[bs] is None:
                    # Reached single vertex; stop.
                    break
                t = labeledge[bs][0]
                bt = inblossom[t]
                assert label[bt] == 2
                # Trace one more step back.
                s, j = labeledge[bt]
                # Augment through the T-blossom from j to base.
                assert blossombase[bt] == t
                if isinstance(bt, Blossom):
                    augmentBlossom(bt, j)
                # Update mate[j]
                mate[j] = s

    # Verify that the optimum solution has been reached.
    def verifyOptimum():
        if maxcardinality:
            # Vertices may have negative dual;
            # find a constant non-negative number to add to all vertex duals.
            vdualoffset = max(0, -min(dualvar.values()))
        else:
            vdualoffset = 0
        # 0. all dual variables are non-negative
        assert min(dualvar.values()) + vdualoffset >= 0
        assert len(blossomdual) == 0 or min(blossomdual.values()) >= 0
        # 0. all edges have non-negative slack and
        # 1. all matched edges have zero slack;
        for i, j, d in G.edges(data=True):
            wt = d.get(weight, 1)
            if i == j:
                continue  # ignore self-loops
            s = dualvar[i] + dualvar[j] - 2 * wt
            iblossoms = [i]
            jblossoms = [j]
            while blossomparent[iblossoms[-1]] is not None:
                iblossoms.append(blossomparent[iblossoms[-1]])
            while blossomparent[jblossoms[-1]] is not None:
                jblossoms.append(blossomparent[jblossoms[-1]])
            iblossoms.reverse()
            jblossoms.reverse()
            for bi, bj in zip(iblossoms, jblossoms):
                if bi != bj:
                    break
                s += 2 * blossomdual[bi]
            assert s >= 0
            if mate.get(i) == j or mate.get(j) == i:
                assert mate[i] == j and mate[j] == i
                assert s == 0
        # 2. all single vertices have zero dual value;
        for v in gnodes:
            assert (v in mate) or dualvar[v] + vdualoffset == 0
        # 3. all blossoms with positive dual value are full.
        for b in blossomdual:
            if blossomdual[b] > 0:
                assert len(b.edges) % 2 == 1
                for i, j in b.edges[1::2]:
                    assert mate[i] == j and mate[j] == i
        # Ok.

    # Main loop: continue until no further improvement is possible.
    while 1:
        # Each iteration of this loop is a "stage".
        # A stage finds an augmenting path and uses that to improve
        # the matching.

        # Remove labels from top-level blossoms/vertices.
        label.clear()
        labeledge.clear()

        # Forget all about least-slack edges.
        bestedge.clear()
        for b in blossomdual:
            b.mybestedges = None

        # Loss of labeling means that we can not be sure that currently
        # allowable edges remain allowable throughout this stage.
        allowedge.clear()

        # Make queue empty.
        queue[:] = []

        # Label single blossoms/vertices with S and put them in the queue.
        for v in gnodes:
            if (v not in mate) and label.get(inblossom[v]) is None:
                assignLabel(v, 1, None)

        # Loop until we succeed in augmenting the matching.
        augmented = 0
        while 1:
            # Each iteration of this loop is a "substage".
            # A substage tries to find an augmenting path;
            # if found, the path is used to improve the matching and
            # the stage ends. If there is no augmenting path, the
            # primal-dual method is used to pump some slack out of
            # the dual variables.

            # Continue labeling until all vertices which are reachable
            # through an alternating path have got a label.
            while queue and not augmented:
                # Take an S vertex from the queue.
                v = queue.pop()
                assert label[inblossom[v]] == 1

                # Scan its neighbors:
                for w in G.neighbors(v):
                    if w == v:
                        continue  # ignore self-loops
                    # w is a neighbor to v
                    bv = inblossom[v]
                    bw = inblossom[w]
                    if bv == bw:
                        # this edge is internal to a blossom; ignore it
                        continue
                    if (v, w) not in allowedge:
                        kslack = slack(v, w)
                        if kslack <= 0:
                            # edge k has zero slack => it is allowable
                            allowedge[(v, w)] = allowedge[(w, v)] = True
                    if (v, w) in allowedge:
                        if label.get(bw) is None:
                            # (C1) w is a free vertex;
                            # label w with T and label its mate with S (R12).
                            assignLabel(w, 2, v)
                        elif label.get(bw) == 1:
                            # (C2) w is an S-vertex (not in the same blossom);
                            # follow back-links to discover either an
                            # augmenting path or a new blossom.
                            base = scanBlossom(v, w)
                            if base is not NoNode:
                                # Found a new blossom; add it to the blossom
                                # bookkeeping and turn it into an S-blossom.
                                addBlossom(base, v, w)
                            else:
                                # Found an augmenting path; augment the
                                # matching and end this stage.
                                augmentMatching(v, w)
                                augmented = 1
                                break
                        elif label.get(w) is None:
                            # w is inside a T-blossom, but w itself has not
                            # yet been reached from outside the blossom;
                            # mark it as reached (we need this to relabel
                            # during T-blossom expansion).
                            assert label[bw] == 2
                            label[w] = 2
                            labeledge[w] = (v, w)
                    elif label.get(bw) == 1:
                        # keep track of the least-slack non-allowable edge to
                        # a different S-blossom.
                        if bestedge.get(bv) is None or kslack < slack(*bestedge[bv]):
                            bestedge[bv] = (v, w)
                    elif label.get(w) is None:
                        # w is a free vertex (or an unreached vertex inside
                        # a T-blossom) but we can not reach it yet;
                        # keep track of the least-slack edge that reaches w.
                        if bestedge.get(w) is None or kslack < slack(*bestedge[w]):
                            bestedge[w] = (v, w)

            if augmented:
                break

            # There is no augmenting path under these constraints;
            # compute delta and reduce slack in the optimization problem.
            # (Note that our vertex dual variables, edge slacks and delta's
            # are pre-multiplied by two.)
            deltatype = -1
            delta = deltaedge = deltablossom = None

            # Compute delta1: the minimum value of any vertex dual.
            if not maxcardinality:
                deltatype = 1
                delta = min(dualvar.values())

            # Compute delta2: the minimum slack on any edge between
            # an S-vertex and a free vertex.
            for v in G.nodes():
                if label.get(inblossom[v]) is None and bestedge.get(v) is not None:
                    d = slack(*bestedge[v])
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 2
                        deltaedge = bestedge[v]

            # Compute delta3: half the minimum slack on any edge between
            # a pair of S-blossoms.
            for b in blossomparent:
                if (
                    blossomparent[b] is None
                    and label.get(b) == 1
                    and bestedge.get(b) is not None
                ):
                    kslack = slack(*bestedge[b])
                    if allinteger:
                        assert (kslack % 2) == 0
                        d = kslack // 2
                    else:
                        d = kslack / 2.0
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 3
                        deltaedge = bestedge[b]

            # Compute delta4: minimum z variable of any T-blossom.
            for b in blossomdual:
                if (
                    blossomparent[b] is None
                    and label.get(b) == 2
                    and (deltatype == -1 or blossomdual[b] < delta)
                ):
                    delta = blossomdual[b]
                    deltatype = 4
                    deltablossom = b

            if deltatype == -1:
                # No further improvement possible; max-cardinality optimum
                # reached. Do a final delta update to make the optimum
                # verifiable.
                assert maxcardinality
                deltatype = 1
                delta = max(0, min(dualvar.values()))

            # Update dual variables according to delta.
            for v in gnodes:
                if label.get(inblossom[v]) == 1:
                    # S-vertex: 2*u = 2*u - 2*delta
                    dualvar[v] -= delta
                elif label.get(inblossom[v]) == 2:
                    # T-vertex: 2*u = 2*u + 2*delta
                    dualvar[v] += delta
            for b in blossomdual:
                if blossomparent[b] is None:
                    if label.get(b) == 1:
                        # top-level S-blossom: z = z + 2*delta
                        blossomdual[b] += delta
                    elif label.get(b) == 2:
                        # top-level T-blossom: z = z - 2*delta
                        blossomdual[b] -= delta

            # Take action at the point where minimum delta occurred.
            if deltatype == 1:
                # No further improvement possible; optimum reached.
                break
            elif deltatype == 2:
                # Use the least-slack edge to continue the search.
                (v, w) = deltaedge
                assert label[inblossom[v]] == 1
                allowedge[(v, w)] = allowedge[(w, v)] = True
                queue.append(v)
            elif deltatype == 3:
                # Use the least-slack edge to continue the search.
                (v, w) = deltaedge
                allowedge[(v, w)] = allowedge[(w, v)] = True
                assert label[inblossom[v]] == 1
                queue.append(v)
            elif deltatype == 4:
                # Expand the least-z blossom.
                expandBlossom(deltablossom, False)

            # End of a this substage.

        # Paranoia check that the matching is symmetric.
        for v in mate:
            assert mate[mate[v]] == v

        # Stop when no more augmenting path can be found.
        if not augmented:
            break

        # End of a stage; expand all S-blossoms which have zero dual.
        for b in list(blossomdual.keys()):
            if b not in blossomdual:
                continue  # already expanded
            if blossomparent[b] is None and label.get(b) == 1 and blossomdual[b] == 0:
                expandBlossom(b, True)

    # Verify that we reached the optimum solution (only for integer weights).
    if allinteger:
        verifyOptimum()

    return matching_dict_to_set(mate)
<|endoftext|>"""Functions for finding and manipulating cliques.

Finding the largest clique in a graph is NP-complete problem, so most of
these algorithms have an exponential running time; for more information,
see the Wikipedia article on the clique problem [1]_.

.. [1] clique problem:: https://en.wikipedia.org/wiki/Clique_problem

"""

from collections import defaultdict, deque
from itertools import chain, combinations, islice

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "find_cliques",
    "find_cliques_recursive",
    "make_max_clique_graph",
    "make_clique_bipartite",
    "node_clique_number",
    "number_of_cliques",
    "enumerate_all_cliques",
    "max_weight_clique",
]


@not_implemented_for("directed")
@nx._dispatchable
def enumerate_all_cliques(G):
    """Returns all cliques in an undirected graph.

    This function returns an iterator over cliques, each of which is a
    list of nodes. The iteration is ordered by cardinality of the
    cliques: first all cliques of size one, then all cliques of size
    two, etc.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    iterator
        An iterator over cliques, each of which is a list of nodes in
        `G`. The cliques are ordered according to size.

    Notes
    -----
    To obtain a list of all cliques, use
    `list(enumerate_all_cliques(G))`. However, be aware that in the
    worst-case, the length of this list can be exponential in the number
    of nodes in the graph (for example, when the graph is the complete
    graph). This function avoids storing all cliques in memory by only
    keeping current candidate node lists in memory during its search.

    The implementation is adapted from the algorithm by Zhang, et
    al. (2005) [1]_ to output all cliques discovered.

    This algorithm ignores self-loops and parallel edges, since cliques
    are not conventionally defined with such edges.

    References
    ----------
    .. [1] Yun Zhang, Abu-Khzam, F.N., Baldwin, N.E., Chesler, E.J.,
           Langston, M.A., Samatova, N.F.,
           "Genome-Scale Computational Approaches to Memory-Intensive
           Applications in Systems Biology".
           *Supercomputing*, 2005. Proceedings of the ACM/IEEE SC 2005
           Conference, pp. 12, 12--18 Nov. 2005.
           <https://doi.org/10.1109/SC.2005.29>.

    """
    index = {}
    nbrs = {}
    for u in G:
        index[u] = len(index)
        # Neighbors of u that appear after u in the iteration order of G.
        nbrs[u] = {v for v in G[u] if v not in index}

    queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in G)
    # Loop invariants:
    # 1. len(base) is nondecreasing.
    # 2. (base + cnbrs) is sorted with respect to the iteration order of G.
    # 3. cnbrs is a set of common neighbors of nodes in base.
    while queue:
        base, cnbrs = map(list, queue.popleft())
        yield base
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append(
                (
                    chain(base, [u]),
                    filter(nbrs[u].__contains__, islice(cnbrs, i + 1, None)),
                )
            )


@not_implemented_for("directed")
@nx._dispatchable
def find_cliques(G, nodes=None):
    """Returns all maximal cliques in an undirected graph.

    For each node *n*, a *maximal clique for n* is a largest complete
    subgraph containing *n*. The largest maximal clique is sometimes
    called the *maximum clique*.

    This function returns an iterator over cliques, each of which is a
    list of nodes. It is an iterative implementation, so should not
    suffer from recursion depth issues.

    This function accepts a list of `nodes` and only the maximal cliques
    containing all of these `nodes` are returned. It can considerably speed up
    the running time if some specific cliques are desired.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    nodes : list, optional (default=None)
        If provided, only yield *maximal cliques* containing all nodes in `nodes`.
        If `nodes` isn't a clique itself, a ValueError is raised.

    Returns
    -------
    iterator
        An iterator over maximal cliques, each of which is a list of
        nodes in `G`. If `nodes` is provided, only the maximal cliques
        containing all the nodes in `nodes` are returned. The order of
        cliques is arbitrary.

    Raises
    ------
    ValueError
        If `nodes` is not a clique.

    Examples
    --------
    >>> from pprint import pprint  # For nice dict formatting
    >>> G = nx.karate_club_graph()
    >>> sum(1 for c in nx.find_cliques(G))  # The number of maximal cliques in G
    36
    >>> max(nx.find_cliques(G), key=len)  # The largest maximal clique in G
    [0, 1, 2, 3, 13]

    The size of the largest maximal clique is known as the *clique number* of
    the graph, which can be found directly with:

    >>> max(len(c) for c in nx.find_cliques(G))
    5

    One can also compute the number of maximal cliques in `G` that contain a given
    node. The following produces a dictionary keyed by node whose
    values are the number of maximal cliques in `G` that contain the node:

    >>> pprint({n: sum(1 for c in nx.find_cliques(G) if n in c) for n in G})
    {0: 13,
     1: 6,
     2: 7,
     3: 3,
     4: 2,
     5: 3,
     6: 3,
     7: 1,
     8: 3,
     9: 2,
     10: 2,
     11: 1,
     12: 1,
     13: 2,
     14: 1,
     15: 1,
     16: 1,
     17: 1,
     18: 1,
     19: 2,
     20: 1,
     21: 1,
     22: 1,
     23: 3,
     24: 2,
     25: 2,
     26: 1,
     27: 3,
     28: 2,
     29: 2,
     30: 2,
     31: 4,
     32: 9,
     33: 14}

    Or, similarly, the maximal cliques in `G` that contain a given node.
    For example, the 4 maximal cliques that contain node 31:

    >>> [c for c in nx.find_cliques(G) if 31 in c]
    [[0, 31], [33, 32, 31], [33, 28, 31], [24, 25, 31]]

    See Also
    --------
    find_cliques_recursive
        A recursive version of the same algorithm.

    Notes
    -----
    To obtain a list of all maximal cliques, use
    `list(find_cliques(G))`. However, be aware that in the worst-case,
    the length of this list can be exponential in the number of nodes in
    the graph. This function avoids storing all cliques in memory by
    only keeping current candidate node lists in memory during its search.

    This implementation is based on the algorithm published by Bron and
    Kerbosch (1973) [1]_, as adapted by Tomita, Tanaka and Takahashi
    (2006) [2]_ and discussed in Cazals and Karande (2008) [3]_. It
    essentially unrolls the recursion used in the references to avoid
    issues of recursion stack depth (for a recursive implementation, see
    :func:`find_cliques_recursive`).

    This algorithm ignores self-loops and parallel edges, since cliques
    are not conventionally defined with such edges.

    References
    ----------
    .. [1] Bron, C. and Kerbosch, J.
       "Algorithm 457: finding all cliques of an undirected graph".
       *Communications of the ACM* 16, 9 (Sep. 1973), 575--577.
       <http://portal.acm.org/citation.cfm?doid=362342.362367>

    .. [2] Etsuji Tomita, Akira Tanaka, Haruhisa Takahashi,
       "The worst-case time complexity for generating all maximal
       cliques and computational experiments",
       *Theoretical Computer Science*, Volume 363, Issue 1,
       Computing and Combinatorics,
       10th Annual International Conference on
       Computing and Combinatorics (COCOON 2004), 25 October 2006, Pages 28--42
       <https://doi.org/10.1016/j.tcs.2006.06.015>

    .. [3] F. Cazals, C. Karande,
       "A note on the problem of reporting maximal cliques",
       *Theoretical Computer Science*,
       Volume 407, Issues 1--3, 6 November 2008, Pages 564--568,
       <https://doi.org/10.1016/j.tcs.2008.05.010>

    """
    if len(G) == 0:
        return

    adj = {u: {v for v in G[u] if v != u} for u in G}

    # Initialize Q with the given nodes and subg, cand with their nbrs
    Q = nodes[:] if nodes is not None else []
    cand = set(G)
    for node in Q:
        if node not in cand:
            raise ValueError(f"The given `nodes` {nodes} do not form a clique")
        cand &= adj[node]

    if not cand:
        yield Q[:]
        return

    subg = cand.copy()
    stack = []
    Q.append(None)

    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]

    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass


# TODO Should this also be not implemented for directed graphs?
@nx._dispatchable
def find_cliques_recursive(G, nodes=None):
    """Returns all maximal cliques in a graph.

    For each node *v*, a *maximal clique for v* is a largest complete
    subgraph containing *v*. The largest maximal clique is sometimes
    called the *maximum clique*.

    This function returns an iterator over cliques, each of which is a
    list of nodes. It is a recursive implementation, so may suffer from
    recursion depth issues, but is included for pedagogical reasons.
    For a non-recursive implementation, see :func:`find_cliques`.

    This function accepts a list of `nodes` and only the maximal cliques
    containing all of these `nodes` are returned. It can considerably speed up
    the running time if some specific cliques are desired.

    Parameters
    ----------
    G : NetworkX graph

    nodes : list, optional (default=None)
        If provided, only yield *maximal cliques* containing all nodes in `nodes`.
        If `nodes` isn't a clique itself, a ValueError is raised.

    Returns
    -------
    iterator
        An iterator over maximal cliques, each of which is a list of
        nodes in `G`. If `nodes` is provided, only the maximal cliques
        containing all the nodes in `nodes` are yielded. The order of
        cliques is arbitrary.

    Raises
    ------
    ValueError
        If `nodes` is not a clique.

    See Also
    --------
    find_cliques
        An iterative version of the same algorithm. See docstring for examples.

    Notes
    -----
    To obtain a list of all maximal cliques, use
    `list(find_cliques_recursive(G))`. However, be aware that in the
    worst-case, the length of this list can be exponential in the number
    of nodes in the graph. This function avoids storing all cliques in memory
    by only keeping current candidate node lists in memory during its search.

    This implementation is based on the algorithm published by Bron and
    Kerbosch (1973) [1]_, as adapted by Tomita, Tanaka and Takahashi
    (2006) [2]_ and discussed in Cazals and Karande (2008) [3]_. For a
    non-recursive implementation, see :func:`find_cliques`.

    This algorithm ignores self-loops and parallel edges, since cliques
    are not conventionally defined with such edges.

    References
    ----------
    .. [1] Bron, C. and Kerbosch, J.
       "Algorithm 457: finding all cliques of an undirected graph".
       *Communications of the ACM* 16, 9 (Sep. 1973), 575--577.
       <http://portal.acm.org/citation.cfm?doid=362342.362367>

    .. [2] Etsuji Tomita, Akira Tanaka, Haruhisa Takahashi,
       "The worst-case time complexity for generating all maximal
       cliques and computational experiments",
       *Theoretical Computer Science*, Volume 363, Issue 1,
       Computing and Combinatorics,
       10th Annual International Conference on
       Computing and Combinatorics (COCOON 2004), 25 October 2006, Pages 28--42
       <https://doi.org/10.1016/j.tcs.2006.06.015>

    .. [3] F. Cazals, C. Karande,
       "A note on the problem of reporting maximal cliques",
       *Theoretical Computer Science*,
       Volume 407, Issues 1--3, 6 November 2008, Pages 564--568,
       <https://doi.org/10.1016/j.tcs.2008.05.010>

    """
    if len(G) == 0:
        return iter([])

    adj = {u: {v for v in G[u] if v != u} for u in G}

    # Initialize Q with the given nodes and subg, cand with their nbrs
    Q = nodes[:] if nodes is not None else []
    cand_init = set(G)
    for node in Q:
        if node not in cand_init:
            raise ValueError(f"The given `nodes` {nodes} do not form a clique")
        cand_init &= adj[node]

    if not cand_init:
        return iter([Q])

    subg_init = cand_init.copy()

    def expand(subg, cand):
        u = max(subg, key=lambda u: len(cand & adj[u]))
        for q in cand - adj[u]:
            cand.remove(q)
            Q.append(q)
            adj_q = adj[q]
            subg_q = subg & adj_q
            if not subg_q:
                yield Q[:]
            else:
                cand_q = cand & adj_q
                if cand_q:
                    yield from expand(subg_q, cand_q)
            Q.pop()

    return expand(subg_init, cand_init)


@nx._dispatchable(returns_graph=True)
def make_max_clique_graph(G, create_using=None):
    """Returns the maximal clique graph of the given graph.

    The nodes of the maximal clique graph of `G` are the cliques of
    `G` and an edge joins two cliques if the cliques are not disjoint.

    Parameters
    ----------
    G : NetworkX graph

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        A graph whose nodes are the cliques of `G` and whose edges
        join two cliques if they are not disjoint.

    Notes
    -----
    This function behaves like the following code::

        import networkx as nx

        G = nx.make_clique_bipartite(G)
        cliques = [v for v in G.nodes() if G.nodes[v]["bipartite"] == 0]
        G = nx.bipartite.projected_graph(G, cliques)
        G = nx.relabel_nodes(G, {-v: v - 1 for v in G})

    It should be faster, though, since it skips all the intermediate
    steps.

    """
    if create_using is None:
        B = G.__class__()
    else:
        B = nx.empty_graph(0, create_using)
    cliques = list(enumerate(set(c) for c in find_cliques(G)))
    # Add a numbered node for each clique.
    B.add_nodes_from(i for i, c in cliques)
    # Join cliques by an edge if they share a node.
    clique_pairs = combinations(cliques, 2)
    B.add_edges_from((i, j) for (i, c1), (j, c2) in clique_pairs if c1 & c2)
    return B


@nx._dispatchable(returns_graph=True)
def make_clique_bipartite(G, fpos=None, create_using=None, name=None):
    """Returns the bipartite clique graph corresponding to `G`.

    In the returned bipartite graph, the "bottom" nodes are the nodes of
    `G` and the "top" nodes represent the maximal cliques of `G`.
    There is an edge from node *v* to clique *C* in the returned graph
    if and only if *v* is an element of *C*.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    fpos : bool
        If True or not None, the returned graph will have an
        additional attribute, `pos`, a dictionary mapping node to
        position in the Euclidean plane.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        A bipartite graph whose "bottom" set is the nodes of the graph
        `G`, whose "top" set is the cliques of `G`, and whose edges
        join nodes of `G` to the cliques that contain them.

        The nodes of the graph `G` have the node attribute
        'bipartite' set to 1 and the nodes representing cliques
        have the node attribute 'bipartite' set to 0, as is the
        convention for bipartite graphs in NetworkX.

    """
    B = nx.empty_graph(0, create_using)
    B.clear()
    # The "bottom" nodes in the bipartite graph are the nodes of the
    # original graph, G.
    B.add_nodes_from(G, bipartite=1)
    for i, cl in enumerate(find_cliques(G)):
        # The "top" nodes in the bipartite graph are the cliques. These
        # nodes get negative numbers as labels.
        name = -i - 1
        B.add_node(name, bipartite=0)
        B.add_edges_from((v, name) for v in cl)
    return B


@nx._dispatchable
def node_clique_number(G, nodes=None, cliques=None, separate_nodes=False):
    """Returns the size of the largest maximal clique containing each given node.

    Returns a single or list depending on input nodes.
    An optional list of cliques can be input if already computed.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    cliques : list, optional (default=None)
        A list of cliques, each of which is itself a list of nodes.
        If not specified, the list of all cliques will be computed
        using :func:`find_cliques`.

    Returns
    -------
    int or dict
        If `nodes` is a single node, returns the size of the
        largest maximal clique in `G` containing that node.
        Otherwise return a dict keyed by node to the size
        of the largest maximal clique containing that node.

    See Also
    --------
    find_cliques
        find_cliques yields the maximal cliques of G.
        It accepts a `nodes` argument which restricts consideration to
        maximal cliques containing all the given `nodes`.
        The search for the cliques is optimized for `nodes`.
    """
    if cliques is None:
        if nodes is not None:
            # Use ego_graph to decrease size of graph
            # check for single node
            if nodes in G:
                return max(len(c) for c in find_cliques(nx.ego_graph(G, nodes)))
            # handle multiple nodes
            return {
                n: max(len(c) for c in find_cliques(nx.ego_graph(G, n))) for n in nodes
            }

        # nodes is None--find all cliques
        cliques = list(find_cliques(G))

    # single node requested
    if nodes in G:
        return max(len(c) for c in cliques if nodes in c)

    # multiple nodes requested
    # preprocess all nodes (faster than one at a time for even 2 nodes)
    size_for_n = defaultdict(int)
    for c in cliques:
        size_of_c = len(c)
        for n in c:
            if size_for_n[n] < size_of_c:
                size_for_n[n] = size_of_c
    if nodes is None:
        return size_for_n
    return {n: size_for_n[n] for n in nodes}


def number_of_cliques(G, nodes=None, cliques=None):
    """Returns the number of maximal cliques for each node.

    Returns a single or list depending on input nodes.
    Optional list of cliques can be input if already computed.
    """
    if cliques is None:
        cliques = list(find_cliques(G))

    if nodes is None:
        nodes = list(G.nodes())  # none, get entire graph

    if not isinstance(nodes, list):  # check for a list
        v = nodes
        # assume it is a single value
        numcliq = len([1 for c in cliques if v in c])
    else:
        numcliq = {}
        for v in nodes:
            numcliq[v] = len([1 for c in cliques if v in c])
    return numcliq


class MaxWeightClique:
    """A class for the maximum weight clique algorithm.

    This class is a helper for the `max_weight_clique` function.  The class
    should not normally be used directly.

    Parameters
    ----------
    G : NetworkX graph
        The undirected graph for which a maximum weight clique is sought
    weight : string or None, optional (default='weight')
        The node attribute that holds the integer value used as a weight.
        If None, then each node has weight 1.

    Attributes
    ----------
    G : NetworkX graph
        The undirected graph for which a maximum weight clique is sought
    node_weights: dict
        The weight of each node
    incumbent_nodes : list
        The nodes of the incumbent clique (the best clique found so far)
    incumbent_weight: int
        The weight of the incumbent clique
    """

    def __init__(self, G, weight):
        self.G = G
        self.incumbent_nodes = []
        self.incumbent_weight = 0

        if weight is None:
            self.node_weights = {v: 1 for v in G.nodes()}
        else:
            for v in G.nodes():
                if weight not in G.nodes[v]:
                    errmsg = f"Node {v!r} does not have the requested weight field."
                    raise KeyError(errmsg)
                if not isinstance(G.nodes[v][weight], int):
                    errmsg = f"The {weight!r} field of node {v!r} is not an integer."
                    raise ValueError(errmsg)
            self.node_weights = {v: G.nodes[v][weight] for v in G.nodes()}

    def update_incumbent_if_improved(self, C, C_weight):
        """Update the incumbent if the node set C has greater weight.

        C is assumed to be a clique.
        """
        if C_weight > self.incumbent_weight:
            self.incumbent_nodes = C[:]
            self.incumbent_weight = C_weight

    def greedily_find_independent_set(self, P):
        """Greedily find an independent set of nodes from a set of
        nodes P."""
        independent_set = []
        P = P[:]
        while P:
            v = P[0]
            independent_set.append(v)
            P = [w for w in P if v != w and not self.G.has_edge(v, w)]
        return independent_set

    def find_branching_nodes(self, P, target):
        """Find a set of nodes to branch on."""
        residual_wt = {v: self.node_weights[v] for v in P}
        total_wt = 0
        P = P[:]
        while P:
            independent_set = self.greedily_find_independent_set(P)
            min_wt_in_class = min(residual_wt[v] for v in independent_set)
            total_wt += min_wt_in_class
            if total_wt > target:
                break
            for v in independent_set:
                residual_wt[v] -= min_wt_in_class
            P = [v for v in P if residual_wt[v] != 0]
        return P

    def expand(self, C, C_weight, P):
        """Look for the best clique that contains all the nodes in C and zero or
        more of the nodes in P, backtracking if it can be shown that no such
        clique has greater weight than the incumbent.
        """
        self.update_incumbent_if_improved(C, C_weight)
        branching_nodes = self.find_branching_nodes(P, self.incumbent_weight - C_weight)
        while branching_nodes:
            v = branching_nodes.pop()
            P.remove(v)
            new_C = C + [v]
            new_C_weight = C_weight + self.node_weights[v]
            new_P = [w for w in P if self.G.has_edge(v, w)]
            self.expand(new_C, new_C_weight, new_P)

    def find_max_weight_clique(self):
        """Find a maximum weight clique."""
        # Sort nodes in reverse order of degree for speed
        nodes = sorted(self.G.nodes(), key=lambda v: self.G.degree(v), reverse=True)
        nodes = [v for v in nodes if self.node_weights[v] > 0]
        self.expand([], 0, nodes)


@not_implemented_for("directed")
@nx._dispatchable(node_attrs="weight")
def max_weight_clique(G, weight="weight"):
    """Find a maximum weight clique in G.

    A *clique* in a graph is a set of nodes such that every two distinct nodes
    are adjacent.  The *weight* of a clique is the sum of the weights of its
    nodes.  A *maximum weight clique* of graph G is a clique C in G such that
    no clique in G has weight greater than the weight of C.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph
    weight : string or None, optional (default='weight')
        The node attribute that holds the integer value used as a weight.
        If None, then each node has weight 1.

    Returns
    -------
    clique : list
        the nodes of a maximum weight clique
    weight : int
        the weight of a maximum weight clique

    Notes
    -----
    The implementation is recursive, and therefore it may run into recursion
    depth issues if G contains a clique whose number of nodes is close to the
    recursion depth limit.

    At each search node, the algorithm greedily constructs a weighted
    independent set cover of part of the graph in order to find a small set of
    nodes on which to branch.  The algorithm is very similar to the algorithm
    of Tavares et al. [1]_, other than the fact that the NetworkX version does
    not use bitsets.  This style of algorithm for maximum weight clique (and
    maximum weight independent set, which is the same problem but on the
    complement graph) has a decades-long history.  See Algorithm B of Warren
    and Hicks [2]_ and the references in that paper.

    References
    ----------
    .. [1] Tavares, W.A., Neto, M.B.C., Rodrigues, C.D., Michelon, P.: Um
           algoritmo de branch and bound para o problema da clique máxima
           ponderada.  Proceedings of XLVII SBPO 1 (2015).

    .. [2] Warren, Jeffrey S, Hicks, Illya V.: Combinatorial Branch-and-Bound
           for the Maximum Weight Independent Set Problem.  Technical Report,
           Texas A&M University (2016).
    """

    mwc = MaxWeightClique(G, weight)
    mwc.find_max_weight_clique()
    return mwc.incumbent_nodes, mwc.incumbent_weight
<|endoftext|>"""Algorithms to characterize the number of triangles in a graph."""

from collections import Counter
from itertools import chain, combinations

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "triangles",
    "average_clustering",
    "clustering",
    "transitivity",
    "square_clustering",
    "generalized_degree",
]


@not_implemented_for("directed")
@nx._dispatchable
def triangles(G, nodes=None):
    """Compute the number of triangles.

    Finds the number of triangles that include a node as one vertex.

    Parameters
    ----------
    G : graph
       A networkx graph

    nodes : node, iterable of nodes, or None (default=None)
        If a singleton node, return the number of triangles for that node.
        If an iterable, compute the number of triangles for each of those nodes.
        If `None` (the default) compute the number of triangles for all nodes in `G`.

    Returns
    -------
    out : dict or int
       If `nodes` is a container of nodes, returns number of triangles keyed by node (dict).
       If `nodes` is a specific node, returns number of triangles for the node (int).

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.triangles(G, 0))
    6
    >>> print(nx.triangles(G))
    {0: 6, 1: 6, 2: 6, 3: 6, 4: 6}
    >>> print(list(nx.triangles(G, [0, 1]).values()))
    [6, 6]

    Notes
    -----
    Self loops are ignored.

    """
    if nodes is not None:
        # If `nodes` represents a single node, return only its number of triangles
        if nodes in G:
            return next(_triangles_and_degree_iter(G, nodes))[2] // 2

        # if `nodes` is a container of nodes, then return a
        # dictionary mapping node to number of triangles.
        return {v: t // 2 for v, d, t, _ in _triangles_and_degree_iter(G, nodes)}

    # if nodes is None, then compute triangles for the complete graph

    # dict used to avoid visiting the same nodes twice
    # this allows calculating/counting each triangle only once
    later_nbrs = {}

    # iterate over the nodes in a graph
    for node, neighbors in G.adjacency():
        later_nbrs[node] = {n for n in neighbors if n not in later_nbrs and n != node}

    # instantiate Counter for each node to include isolated nodes
    # add 1 to the count if a nodes neighbor's neighbor is also a neighbor
    triangle_counts = Counter(dict.fromkeys(G, 0))
    for node1, neighbors in later_nbrs.items():
        for node2 in neighbors:
            third_nodes = neighbors & later_nbrs[node2]
            m = len(third_nodes)
            triangle_counts[node1] += m
            triangle_counts[node2] += m
            triangle_counts.update(third_nodes)

    return dict(triangle_counts)


@not_implemented_for("multigraph")
def _triangles_and_degree_iter(G, nodes=None):
    """Return an iterator of (node, degree, triangles, generalized degree).

    This double counts triangles so you may want to divide by 2.
    See degree(), triangles() and generalized_degree() for definitions
    and details.

    """
    if nodes is None:
        nodes_nbrs = G.adj.items()
    else:
        nodes_nbrs = ((n, G[n]) for n in G.nbunch_iter(nodes))

    for v, v_nbrs in nodes_nbrs:
        vs = set(v_nbrs) - {v}
        gen_degree = Counter(len(vs & (set(G[w]) - {w})) for w in vs)
        ntriangles = sum(k * val for k, val in gen_degree.items())
        yield (v, len(vs), ntriangles, gen_degree)


@not_implemented_for("multigraph")
def _weighted_triangles_and_degree_iter(G, nodes=None, weight="weight"):
    """Return an iterator of (node, degree, weighted_triangles).

    Used for weighted clustering.
    Note: this returns the geometric average weight of edges in the triangle.
    Also, each triangle is counted twice (each direction).
    So you may want to divide by 2.

    """
    import numpy as np

    if weight is None or G.number_of_edges() == 0:
        max_weight = 1
    else:
        max_weight = max(d.get(weight, 1) for u, v, d in G.edges(data=True))
    if nodes is None:
        nodes_nbrs = G.adj.items()
    else:
        nodes_nbrs = ((n, G[n]) for n in G.nbunch_iter(nodes))

    def wt(u, v):
        return G[u][v].get(weight, 1) / max_weight

    for i, nbrs in nodes_nbrs:
        inbrs = set(nbrs) - {i}
        weighted_triangles = 0
        seen = set()
        for j in inbrs:
            seen.add(j)
            # This avoids counting twice -- we double at the end.
            jnbrs = set(G[j]) - seen
            # Only compute the edge weight once, before the inner inner
            # loop.
            wij = wt(i, j)
            weighted_triangles += np.cbrt(
                [(wij * wt(j, k) * wt(k, i)) for k in inbrs & jnbrs]
            ).sum()
        yield (i, len(inbrs), 2 * float(weighted_triangles))


@not_implemented_for("multigraph")
def _directed_triangles_and_degree_iter(G, nodes=None):
    """Return an iterator of
    (node, total_degree, reciprocal_degree, directed_triangles).

    Used for directed clustering.
    Note that unlike `_triangles_and_degree_iter()`, this function counts
    directed triangles so does not count triangles twice.

    """
    nodes_nbrs = ((n, G._pred[n], G._succ[n]) for n in G.nbunch_iter(nodes))

    for i, preds, succs in nodes_nbrs:
        ipreds = set(preds) - {i}
        isuccs = set(succs) - {i}

        directed_triangles = 0
        for j in chain(ipreds, isuccs):
            jpreds = set(G._pred[j]) - {j}
            jsuccs = set(G._succ[j]) - {j}
            directed_triangles += sum(
                1
                for k in chain(
                    (ipreds & jpreds),
                    (ipreds & jsuccs),
                    (isuccs & jpreds),
                    (isuccs & jsuccs),
                )
            )
        dtotal = len(ipreds) + len(isuccs)
        dbidirectional = len(ipreds & isuccs)
        yield (i, dtotal, dbidirectional, directed_triangles)


@not_implemented_for("multigraph")
def _directed_weighted_triangles_and_degree_iter(G, nodes=None, weight="weight"):
    """Return an iterator of
    (node, total_degree, reciprocal_degree, directed_weighted_triangles).

    Used for directed weighted clustering.
    Note that unlike `_weighted_triangles_and_degree_iter()`, this function counts
    directed triangles so does not count triangles twice.

    """
    import numpy as np

    if weight is None or G.number_of_edges() == 0:
        max_weight = 1
    else:
        max_weight = max(d.get(weight, 1) for u, v, d in G.edges(data=True))

    nodes_nbrs = ((n, G._pred[n], G._succ[n]) for n in G.nbunch_iter(nodes))

    def wt(u, v):
        return G[u][v].get(weight, 1) / max_weight

    for i, preds, succs in nodes_nbrs:
        ipreds = set(preds) - {i}
        isuccs = set(succs) - {i}

        directed_triangles = 0
        for j in ipreds:
            jpreds = set(G._pred[j]) - {j}
            jsuccs = set(G._succ[j]) - {j}
            directed_triangles += np.cbrt(
                [(wt(j, i) * wt(k, i) * wt(k, j)) for k in ipreds & jpreds]
            ).sum()
            directed_triangles += np.cbrt(
                [(wt(j, i) * wt(k, i) * wt(j, k)) for k in ipreds & jsuccs]
            ).sum()
            directed_triangles += np.cbrt(
                [(wt(j, i) * wt(i, k) * wt(k, j)) for k in isuccs & jpreds]
            ).sum()
            directed_triangles += np.cbrt(
                [(wt(j, i) * wt(i, k) * wt(j, k)) for k in isuccs & jsuccs]
            ).sum()

        for j in isuccs:
            jpreds = set(G._pred[j]) - {j}
            jsuccs = set(G._succ[j]) - {j}
            directed_triangles += np.cbrt(
                [(wt(i, j) * wt(k, i) * wt(k, j)) for k in ipreds & jpreds]
            ).sum()
            directed_triangles += np.cbrt(
                [(wt(i, j) * wt(k, i) * wt(j, k)) for k in ipreds & jsuccs]
            ).sum()
            directed_triangles += np.cbrt(
                [(wt(i, j) * wt(i, k) * wt(k, j)) for k in isuccs & jpreds]
            ).sum()
            directed_triangles += np.cbrt(
                [(wt(i, j) * wt(i, k) * wt(j, k)) for k in isuccs & jsuccs]
            ).sum()

        dtotal = len(ipreds) + len(isuccs)
        dbidirectional = len(ipreds & isuccs)
        yield (i, dtotal, dbidirectional, float(directed_triangles))


@nx._dispatchable(edge_attrs="weight")
def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    r"""Compute the average clustering coefficient for the graph G.

    The clustering coefficient for the graph is the average,

    .. math::

       C = \frac{1}{n}\sum_{v \in G} c_v,

    where :math:`n` is the number of nodes in `G`.

    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute average clustering for nodes in this container.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    count_zeros : bool
       If False include only the nodes with nonzero clustering in the average.

    Returns
    -------
    avg : float
       Average clustering

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.average_clustering(G))
    1.0

    Notes
    -----
    This is a space saving routine; it might be faster
    to use the clustering function to get a list and then take the average.

    Self loops are ignored.

    References
    ----------
    .. [1] Generalizations of the clustering coefficient to weighted
       complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
       K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
       http://jponnela.com/web_documents/a9.pdf
    .. [2] Marcus Kaiser,  Mean clustering coefficients: the role of isolated
       nodes and leafs on clustering measures for small-world networks.
       https://arxiv.org/abs/0802.2512
    """
    c = clustering(G, nodes, weight=weight).values()
    if not count_zeros:
        c = [v for v in c if abs(v) > 0]
    return sum(c) / len(c)


@nx._dispatchable(edge_attrs="weight")
def clustering(G, nodes=None, weight=None):
    r"""Compute the clustering coefficient for nodes.

    For unweighted graphs, the clustering of a node :math:`u`
    is the fraction of possible triangles through that node that exist,

    .. math::

      c_u = \frac{2 T(u)}{deg(u)(deg(u)-1)},

    where :math:`T(u)` is the number of triangles through node :math:`u` and
    :math:`deg(u)` is the degree of :math:`u`.

    For weighted graphs, there are several ways to define clustering [1]_.
    the one used here is defined
    as the geometric average of the subgraph edge weights [2]_,

    .. math::

       c_u = \frac{1}{deg(u)(deg(u)-1))}
             \sum_{vw} (\hat{w}_{uv} \hat{w}_{uw} \hat{w}_{vw})^{1/3}.

    The edge weights :math:`\hat{w}_{uv}` are normalized by the maximum weight
    in the network :math:`\hat{w}_{uv} = w_{uv}/\max(w)`.

    The value of :math:`c_u` is assigned to 0 if :math:`deg(u) < 2`.

    Additionally, this weighted definition has been generalized to support negative edge weights [3]_.

    For directed graphs, the clustering is similarly defined as the fraction
    of all possible directed triangles or geometric average of the subgraph
    edge weights for unweighted and weighted directed graph respectively [4]_.

    .. math::

       c_u = \frac{T(u)}{2(deg^{tot}(u)(deg^{tot}(u)-1) - 2deg^{\leftrightarrow}(u))},

    where :math:`T(u)` is the number of directed triangles through node
    :math:`u`, :math:`deg^{tot}(u)` is the sum of in degree and out degree of
    :math:`u` and :math:`deg^{\leftrightarrow}(u)` is the reciprocal degree of
    :math:`u`.


    Parameters
    ----------
    G : graph

    nodes : node, iterable of nodes, or None (default=None)
        If a singleton node, return the number of triangles for that node.
        If an iterable, compute the number of triangles for each of those nodes.
        If `None` (the default) compute the number of triangles for all nodes in `G`.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    out : float, or dictionary
       Clustering coefficient at specified nodes

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.clustering(G, 0))
    1.0
    >>> print(nx.clustering(G))
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    Notes
    -----
    Self loops are ignored.

    References
    ----------
    .. [1] Generalizations of the clustering coefficient to weighted
       complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
       K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
       http://jponnela.com/web_documents/a9.pdf
    .. [2] Intensity and coherence of motifs in weighted complex
       networks by J. P. Onnela, J. Saramäki, J. Kertész, and K. Kaski,
       Physical Review E, 71(6), 065103 (2005).
    .. [3] Generalization of Clustering Coefficients to Signed Correlation Networks
       by G. Costantini and M. Perugini, PloS one, 9(2), e88669 (2014).
    .. [4] Clustering in complex directed networks by G. Fagiolo,
       Physical Review E, 76(2), 026107 (2007).
    """
    if G.is_directed():
        if weight is not None:
            td_iter = _directed_weighted_triangles_and_degree_iter(G, nodes, weight)
            clusterc = {
                v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2)
                for v, dt, db, t in td_iter
            }
        else:
            td_iter = _directed_triangles_and_degree_iter(G, nodes)
            clusterc = {
                v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2)
                for v, dt, db, t in td_iter
            }
    else:
        # The formula 2*T/(d*(d-1)) from docs is t/(d*(d-1)) here b/c t==2*T
        if weight is not None:
            td_iter = _weighted_triangles_and_degree_iter(G, nodes, weight)
            clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for v, d, t in td_iter}
        else:
            td_iter = _triangles_and_degree_iter(G, nodes)
            clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for v, d, t, _ in td_iter}
    if nodes in G:
        # Return the value of the sole entry in the dictionary.
        return clusterc[nodes]
    return clusterc


@nx._dispatchable
def transitivity(G):
    r"""Compute graph transitivity, the fraction of all possible triangles
    present in G.

    Possible triangles are identified by the number of "triads"
    (two edges with a shared vertex).

    The transitivity is

    .. math::

        T = 3\frac{\#triangles}{\#triads}.

    Parameters
    ----------
    G : graph

    Returns
    -------
    out : float
       Transitivity

    Notes
    -----
    Self loops are ignored.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.transitivity(G))
    1.0
    """
    triangles_contri = [
        (t, d * (d - 1)) for v, d, t, _ in _triangles_and_degree_iter(G)
    ]
    # If the graph is empty
    if len(triangles_contri) == 0:
        return 0
    triangles, contri = map(sum, zip(*triangles_contri))
    return 0 if triangles == 0 else triangles / contri


@nx._dispatchable
def square_clustering(G, nodes=None):
    r"""Compute the squares clustering coefficient for nodes.

    For each node return the fraction of possible squares that exist at
    the node [1]_

    .. math::
       C_4(v) = \frac{ \sum_{u=1}^{k_v}
       \sum_{w=u+1}^{k_v} q_v(u,w) }{ \sum_{u=1}^{k_v}
       \sum_{w=u+1}^{k_v} [a_v(u,w) + q_v(u,w)]},

    where :math:`q_v(u,w)` are the number of common neighbors of :math:`u` and
    :math:`w` other than :math:`v` (ie squares), and :math:`a_v(u,w) = (k_u -
    (1+q_v(u,w)+\theta_{uv})) + (k_w - (1+q_v(u,w)+\theta_{uw}))`, where
    :math:`\theta_{uw} = 1` if :math:`u` and :math:`w` are connected and 0
    otherwise. [2]_

    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute clustering for nodes in this container.

    Returns
    -------
    c4 : dictionary
       A dictionary keyed by node with the square clustering coefficient value.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.square_clustering(G, 0))
    1.0
    >>> print(nx.square_clustering(G))
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    Notes
    -----
    While :math:`C_3(v)` (triangle clustering) gives the probability that
    two neighbors of node v are connected with each other, :math:`C_4(v)` is
    the probability that two neighbors of node v share a common
    neighbor different from v. This algorithm can be applied to both
    bipartite and unipartite networks.

    References
    ----------
    .. [1] Pedro G. Lind, Marta C. González, and Hans J. Herrmann. 2005
        Cycles and clustering in bipartite networks.
        Physical Review E (72) 056127.
    .. [2] Zhang, Peng et al. Clustering Coefficient and Community Structure of
        Bipartite Networks. Physica A: Statistical Mechanics and its Applications 387.27 (2008): 6869–6875.
        https://arxiv.org/abs/0710.0117v1
    """
    if nodes is None:
        node_iter = G
    else:
        node_iter = G.nbunch_iter(nodes)
    clustering = {}
    for v in node_iter:
        clustering[v] = 0
        potential = 0
        for u, w in combinations(G[v], 2):
            squares = len((set(G[u]) & set(G[w])) - {v})
            clustering[v] += squares
            degm = squares + 1
            if w in G[u]:
                degm += 1
            potential += (len(G[u]) - degm) + (len(G[w]) - degm) + squares
        if potential > 0:
            clustering[v] /= potential
    if nodes in G:
        # Return the value of the sole entry in the dictionary.
        return clustering[nodes]
    return clustering


@not_implemented_for("directed")
@nx._dispatchable
def generalized_degree(G, nodes=None):
    r"""Compute the generalized degree for nodes.

    For each node, the generalized degree shows how many edges of given
    triangle multiplicity the node is connected to. The triangle multiplicity
    of an edge is the number of triangles an edge participates in. The
    generalized degree of node :math:`i` can be written as a vector
    :math:`\mathbf{k}_i=(k_i^{(0)}, \dotsc, k_i^{(N-2)})` where
    :math:`k_i^{(j)}` is the number of edges attached to node :math:`i` that
    participate in :math:`j` triangles.

    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute the generalized degree for nodes in this container.

    Returns
    -------
    out : Counter, or dictionary of Counters
       Generalized degree of specified nodes. The Counter is keyed by edge
       triangle multiplicity.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.generalized_degree(G, 0))
    Counter({3: 4})
    >>> print(nx.generalized_degree(G))
    {0: Counter({3: 4}), 1: Counter({3: 4}), 2: Counter({3: 4}), 3: Counter({3: 4}), 4: Counter({3: 4})}

    To recover the number of triangles attached to a node:

    >>> k1 = nx.generalized_degree(G, 0)
    >>> sum([k * v for k, v in k1.items()]) / 2 == nx.triangles(G, 0)
    True

    Notes
    -----
    Self loops are ignored.

    In a network of N nodes, the highest triangle multiplicity an edge can have
    is N-2.

    The return value does not include a `zero` entry if no edges of a
    particular triangle multiplicity are present.

    The number of triangles node :math:`i` is attached to can be recovered from
    the generalized degree :math:`\mathbf{k}_i=(k_i^{(0)}, \dotsc,
    k_i^{(N-2)})` by :math:`(k_i^{(1)}+2k_i^{(2)}+\dotsc +(N-2)k_i^{(N-2)})/2`.

    References
    ----------
    .. [1] Networks with arbitrary edge multiplicities by V. Zlatić,
        D. Garlaschelli and G. Caldarelli, EPL (Europhysics Letters),
        Volume 97, Number 2 (2012).
        https://iopscience.iop.org/article/10.1209/0295-5075/97/28005
    """
    if nodes in G:
        return next(_triangles_and_degree_iter(G, nodes))[3]
    return {v: gd for v, d, t, gd in _triangles_and_degree_iter(G, nodes)}
<|endoftext|>"""
=======================
Distance-regular graphs
=======================
"""

import networkx as nx
from networkx.utils import not_implemented_for

from .distance_measures import diameter

__all__ = [
    "is_distance_regular",
    "is_strongly_regular",
    "intersection_array",
    "global_parameters",
]


@nx._dispatchable
def is_distance_regular(G):
    """Returns True if the graph is distance regular, False otherwise.

    A connected graph G is distance-regular if for any nodes x,y
    and any integers i,j=0,1,...,d (where d is the graph
    diameter), the number of vertices at distance i from x and
    distance j from y depends only on i,j and the graph distance
    between x and y, independently of the choice of x and y.

    Parameters
    ----------
    G: Networkx graph (undirected)

    Returns
    -------
    bool
      True if the graph is Distance Regular, False otherwise

    Examples
    --------
    >>> G = nx.hypercube_graph(6)
    >>> nx.is_distance_regular(G)
    True

    See Also
    --------
    intersection_array, global_parameters

    Notes
    -----
    For undirected and simple graphs only

    References
    ----------
    .. [1] Brouwer, A. E.; Cohen, A. M.; and Neumaier, A.
        Distance-Regular Graphs. New York: Springer-Verlag, 1989.
    .. [2] Weisstein, Eric W. "Distance-Regular Graph."
        http://mathworld.wolfram.com/Distance-RegularGraph.html

    """
    try:
        intersection_array(G)
        return True
    except nx.NetworkXError:
        return False


def global_parameters(b, c):
    """Returns global parameters for a given intersection array.

    Given a distance-regular graph G with integers b_i, c_i,i = 0,....,d
    such that for any 2 vertices x,y in G at a distance i=d(x,y), there
    are exactly c_i neighbors of y at a distance of i-1 from x and b_i
    neighbors of y at a distance of i+1 from x.

    Thus, a distance regular graph has the global parameters,
    [[c_0,a_0,b_0],[c_1,a_1,b_1],......,[c_d,a_d,b_d]] for the
    intersection array  [b_0,b_1,.....b_{d-1};c_1,c_2,.....c_d]
    where a_i+b_i+c_i=k , k= degree of every vertex.

    Parameters
    ----------
    b : list

    c : list

    Returns
    -------
    iterable
       An iterable over three tuples.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> b, c = nx.intersection_array(G)
    >>> list(nx.global_parameters(b, c))
    [(0, 0, 3), (1, 0, 2), (1, 1, 1), (1, 1, 1), (2, 0, 1), (3, 0, 0)]

    References
    ----------
    .. [1] Weisstein, Eric W. "Global Parameters."
       From MathWorld--A Wolfram Web Resource.
       http://mathworld.wolfram.com/GlobalParameters.html

    See Also
    --------
    intersection_array
    """
    return ((y, b[0] - x - y, x) for x, y in zip(b + [0], [0] + c))


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def intersection_array(G):
    """Returns the intersection array of a distance-regular graph.

    Given a distance-regular graph G with integers b_i, c_i,i = 0,....,d
    such that for any 2 vertices x,y in G at a distance i=d(x,y), there
    are exactly c_i neighbors of y at a distance of i-1 from x and b_i
    neighbors of y at a distance of i+1 from x.

    A distance regular graph's intersection array is given by,
    [b_0,b_1,.....b_{d-1};c_1,c_2,.....c_d]

    Parameters
    ----------
    G: Networkx graph (undirected)

    Returns
    -------
    b,c: tuple of lists

    Examples
    --------
    >>> G = nx.icosahedral_graph()
    >>> nx.intersection_array(G)
    ([5, 2, 1], [1, 2, 5])

    References
    ----------
    .. [1] Weisstein, Eric W. "Intersection Array."
       From MathWorld--A Wolfram Web Resource.
       http://mathworld.wolfram.com/IntersectionArray.html

    See Also
    --------
    global_parameters
    """
    # test for regular graph (all degrees must be equal)
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Graph has no nodes.")
    degree = iter(G.degree())
    (_, k) = next(degree)
    for _, knext in degree:
        if knext != k:
            raise nx.NetworkXError("Graph is not distance regular.")
        k = knext
    path_length = dict(nx.all_pairs_shortest_path_length(G))
    diameter = max(max(path_length[n].values()) for n in path_length)
    bint = {}  # 'b' intersection array
    cint = {}  # 'c' intersection array
    for u in G:
        for v in G:
            try:
                i = path_length[u][v]
            except KeyError as err:  # graph must be connected
                raise nx.NetworkXError("Graph is not distance regular.") from err
            # number of neighbors of v at a distance of i-1 from u
            c = len([n for n in G[v] if path_length[n][u] == i - 1])
            # number of neighbors of v at a distance of i+1 from u
            b = len([n for n in G[v] if path_length[n][u] == i + 1])
            # b,c are independent of u and v
            if cint.get(i, c) != c or bint.get(i, b) != b:
                raise nx.NetworkXError("Graph is not distance regular")
            bint[i] = b
            cint[i] = c
    return (
        [bint.get(j, 0) for j in range(diameter)],
        [cint.get(j + 1, 0) for j in range(diameter)],
    )


# TODO There is a definition for directed strongly regular graphs.
@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_strongly_regular(G):
    """Returns True if and only if the given graph is strongly
    regular.

    An undirected graph is *strongly regular* if

    * it is regular,
    * each pair of adjacent vertices has the same number of neighbors in
      common,
    * each pair of nonadjacent vertices has the same number of neighbors
      in common.

    Each strongly regular graph is a distance-regular graph.
    Conversely, if a distance-regular graph has diameter two, then it is
    a strongly regular graph. For more information on distance-regular
    graphs, see :func:`is_distance_regular`.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    bool
        Whether `G` is strongly regular.

    Examples
    --------

    The cycle graph on five vertices is strongly regular. It is
    two-regular, each pair of adjacent vertices has no shared neighbors,
    and each pair of nonadjacent vertices has one shared neighbor::

        >>> G = nx.cycle_graph(5)
        >>> nx.is_strongly_regular(G)
        True

    """
    # Here is an alternate implementation based directly on the
    # definition of strongly regular graphs:
    #
    #     return (all_equal(G.degree().values())
    #             and all_equal(len(common_neighbors(G, u, v))
    #                           for u, v in G.edges())
    #             and all_equal(len(common_neighbors(G, u, v))
    #                           for u, v in non_edges(G)))
    #
    # We instead use the fact that a distance-regular graph of diameter
    # two is strongly regular.
    return is_distance_regular(G) and diameter(G) == 2
<|endoftext|>"""Functions for estimating the small-world-ness of graphs.

A small world network is characterized by a small average shortest path length,
and a large clustering coefficient.

Small-worldness is commonly measured with the coefficient sigma or omega.

Both coefficients compare the average clustering coefficient and shortest path
length of a given graph against the same quantities for an equivalent random
or lattice graph.

For more information, see the Wikipedia article on small-world network [1]_.

.. [1] Small-world network:: https://en.wikipedia.org/wiki/Small-world_network

"""

import networkx as nx
from networkx.utils import not_implemented_for, py_random_state

__all__ = ["random_reference", "lattice_reference", "sigma", "omega"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state(3)
@nx._dispatchable(returns_graph=True)
def random_reference(G, niter=1, connectivity=True, seed=None):
    """Compute a random graph by swapping edges of a given graph.

    Parameters
    ----------
    G : graph
        An undirected graph with 4 or more nodes.

    niter : integer (optional, default=1)
        An edge is rewired approximately `niter` times.

    connectivity : boolean (optional, default=True)
        When True, ensure connectivity for the randomized graph.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
        The randomized graph.

    Raises
    ------
    NetworkXError
        If there are fewer than 4 nodes or 2 edges in `G`

    Notes
    -----
    The implementation is adapted from the algorithm by Maslov and Sneppen
    (2002) [1]_.

    References
    ----------
    .. [1] Maslov, Sergei, and Kim Sneppen.
           "Specificity and stability in topology of protein networks."
           Science 296.5569 (2002): 910-913.
    """
    if len(G) < 4:
        raise nx.NetworkXError("Graph has fewer than four nodes.")
    if len(G.edges) < 2:
        raise nx.NetworkXError("Graph has fewer that 2 edges")

    from networkx.utils import cumulative_distribution, discrete_sequence

    local_conn = nx.connectivity.local_edge_connectivity

    G = G.copy()
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = cumulative_distribution(degrees)  # cdf of degree
    nnodes = len(G)
    nedges = nx.number_of_edges(G)
    niter = niter * nedges
    ntries = int(nnodes * nedges / (nnodes * (nnodes - 1) / 2))
    swapcount = 0

    for i in range(niter):
        n = 0
        while n < ntries:
            # pick two random edges without creating edge list
            # choose source node indices from discrete distribution
            (ai, ci) = discrete_sequence(2, cdistribution=cdf, seed=seed)
            if ai == ci:
                continue  # same source, skip
            a = keys[ai]  # convert index to label
            c = keys[ci]
            # choose target uniformly from neighbors
            b = seed.choice(list(G.neighbors(a)))
            d = seed.choice(list(G.neighbors(c)))
            if b in [a, c, d] or d in [a, b, c]:
                continue  # all vertices should be different

            # don't create parallel edges
            if (d not in G[a]) and (b not in G[c]):
                G.add_edge(a, d)
                G.add_edge(c, b)
                G.remove_edge(a, b)
                G.remove_edge(c, d)

                # Check if the graph is still connected
                if connectivity and local_conn(G, a, b) == 0:
                    # Not connected, revert the swap
                    G.remove_edge(a, d)
                    G.remove_edge(c, b)
                    G.add_edge(a, b)
                    G.add_edge(c, d)
                else:
                    swapcount += 1
                    break
            n += 1
    return G


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state(4)
@nx._dispatchable(returns_graph=True)
def lattice_reference(G, niter=5, D=None, connectivity=True, seed=None):
    """Latticize the given graph by swapping edges.

    Parameters
    ----------
    G : graph
        An undirected graph.

    niter : integer (optional, default=1)
        An edge is rewired approximately niter times.

    D : numpy.array (optional, default=None)
        Distance to the diagonal matrix.

    connectivity : boolean (optional, default=True)
        Ensure connectivity for the latticized graph when set to True.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
        The latticized graph.

    Raises
    ------
    NetworkXError
        If there are fewer than 4 nodes or 2 edges in `G`

    Notes
    -----
    The implementation is adapted from the algorithm by Sporns et al. [1]_.
    which is inspired from the original work by Maslov and Sneppen(2002) [2]_.

    References
    ----------
    .. [1] Sporns, Olaf, and Jonathan D. Zwi.
       "The small world of the cerebral cortex."
       Neuroinformatics 2.2 (2004): 145-162.
    .. [2] Maslov, Sergei, and Kim Sneppen.
       "Specificity and stability in topology of protein networks."
       Science 296.5569 (2002): 910-913.
    """
    import numpy as np

    from networkx.utils import cumulative_distribution, discrete_sequence

    local_conn = nx.connectivity.local_edge_connectivity

    if len(G) < 4:
        raise nx.NetworkXError("Graph has fewer than four nodes.")
    if len(G.edges) < 2:
        raise nx.NetworkXError("Graph has fewer that 2 edges")
    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    G = G.copy()
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = cumulative_distribution(degrees)  # cdf of degree

    nnodes = len(G)
    nedges = nx.number_of_edges(G)
    if D is None:
        D = np.zeros((nnodes, nnodes))
        un = np.arange(1, nnodes)
        um = np.arange(nnodes - 1, 0, -1)
        u = np.append((0,), np.where(un < um, un, um))

        for v in range(int(np.ceil(nnodes / 2))):
            D[nnodes - v - 1, :] = np.append(u[v + 1 :], u[: v + 1])
            D[v, :] = D[nnodes - v - 1, :][::-1]

    niter = niter * nedges
    # maximal number of rewiring attempts per 'niter'
    max_attempts = int(nnodes * nedges / (nnodes * (nnodes - 1) / 2))

    for _ in range(niter):
        n = 0
        while n < max_attempts:
            # pick two random edges without creating edge list
            # choose source node indices from discrete distribution
            (ai, ci) = discrete_sequence(2, cdistribution=cdf, seed=seed)
            if ai == ci:
                continue  # same source, skip
            a = keys[ai]  # convert index to label
            c = keys[ci]
            # choose target uniformly from neighbors
            b = seed.choice(list(G.neighbors(a)))
            d = seed.choice(list(G.neighbors(c)))
            bi = keys.index(b)
            di = keys.index(d)

            if b in [a, c, d] or d in [a, b, c]:
                continue  # all vertices should be different

            # don't create parallel edges
            if (d not in G[a]) and (b not in G[c]):
                if D[ai, bi] + D[ci, di] >= D[ai, ci] + D[bi, di]:
                    # only swap if we get closer to the diagonal
                    G.add_edge(a, d)
                    G.add_edge(c, b)
                    G.remove_edge(a, b)
                    G.remove_edge(c, d)

                    # Check if the graph is still connected
                    if connectivity and local_conn(G, a, b) == 0:
                        # Not connected, revert the swap
                        G.remove_edge(a, d)
                        G.remove_edge(c, b)
                        G.add_edge(a, b)
                        G.add_edge(c, d)
                    else:
                        break
            n += 1

    return G


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state(3)
@nx._dispatchable
def sigma(G, niter=100, nrand=10, seed=None):
    """Returns the small-world coefficient (sigma) of the given graph.

    The small-world coefficient is defined as:
    sigma = C/Cr / L/Lr
    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Cr and Lr are respectively the average
    clustering coefficient and average shortest path length of an equivalent
    random graph.

    A graph is commonly classified as small-world if sigma>1.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    niter : integer (optional, default=100)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.
    nrand : integer (optional, default=10)
        Number of random graphs generated to compute the average clustering
        coefficient (Cr) and average shortest path length (Lr).
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    sigma : float
        The small-world coefficient of G.

    Notes
    -----
    The implementation is adapted from Humphries et al. [1]_ [2]_.

    References
    ----------
    .. [1] The brainstem reticular formation is a small-world, not scale-free,
           network M. D. Humphries, K. Gurney and T. J. Prescott,
           Proc. Roy. Soc. B 2006 273, 503-511, doi:10.1098/rspb.2005.3354.
    .. [2] Humphries and Gurney (2008).
           "Network 'Small-World-Ness': A Quantitative Method for Determining
           Canonical Network Equivalence".
           PLoS One. 3 (4). PMID 18446219. doi:10.1371/journal.pone.0002051.
    """
    import numpy as np

    # Compute the mean clustering coefficient and average shortest path length
    # for an equivalent random graph
    randMetrics = {"C": [], "L": []}
    for i in range(nrand):
        Gr = random_reference(G, niter=niter, seed=seed)
        randMetrics["C"].append(nx.transitivity(Gr))
        randMetrics["L"].append(nx.average_shortest_path_length(Gr))

    C = nx.transitivity(G)
    L = nx.average_shortest_path_length(G)
    Cr = np.mean(randMetrics["C"])
    Lr = np.mean(randMetrics["L"])

    sigma = (C / Cr) / (L / Lr)

    return float(sigma)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state(3)
@nx._dispatchable
def omega(G, niter=5, nrand=10, seed=None):
    """Returns the small-world coefficient (omega) of a graph

    The small-world coefficient of a graph G is:

    omega = Lr/L - C/Cl

    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Lr is the average shortest path length
    of an equivalent random graph and Cl is the average clustering coefficient
    of an equivalent lattice graph.

    The small-world coefficient (omega) measures how much G is like a lattice
    or a random graph. Negative values mean G is similar to a lattice whereas
    positive values mean G is a random graph.
    Values close to 0 mean that G has small-world characteristics.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    niter: integer (optional, default=5)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.

    nrand: integer (optional, default=10)
        Number of random graphs generated to compute the maximal clustering
        coefficient (Cr) and average shortest path length (Lr).

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.


    Returns
    -------
    omega : float
        The small-world coefficient (omega)

    Notes
    -----
    The implementation is adapted from the algorithm by Telesford et al. [1]_.

    References
    ----------
    .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
           "The Ubiquity of Small-World Networks".
           Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.
           doi:10.1089/brain.2011.0038.
    """
    import numpy as np

    # Compute the mean clustering coefficient and average shortest path length
    # for an equivalent random graph
    randMetrics = {"C": [], "L": []}

    # Calculate initial average clustering coefficient which potentially will
    # get replaced by higher clustering coefficients from generated lattice
    # reference graphs
    Cl = nx.average_clustering(G)

    niter_lattice_reference = niter
    niter_random_reference = niter * 2

    for _ in range(nrand):
        # Generate random graph
        Gr = random_reference(G, niter=niter_random_reference, seed=seed)
        randMetrics["L"].append(nx.average_shortest_path_length(Gr))

        # Generate lattice graph
        Gl = lattice_reference(G, niter=niter_lattice_reference, seed=seed)

        # Replace old clustering coefficient, if clustering is higher in
        # generated lattice reference
        Cl_temp = nx.average_clustering(Gl)
        if Cl_temp > Cl:
            Cl = Cl_temp

    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)
    Lr = np.mean(randMetrics["L"])

    omega = (Lr / L) - (C / Cl)

    return float(omega)
<|endoftext|>"""Functions for finding chains in a graph."""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["chain_decomposition"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def chain_decomposition(G, root=None):
    """Returns the chain decomposition of a graph.

    The *chain decomposition* of a graph with respect a depth-first
    search tree is a set of cycles or paths derived from the set of
    fundamental cycles of the tree in the following manner. Consider
    each fundamental cycle with respect to the given tree, represented
    as a list of edges beginning with the nontree edge oriented away
    from the root of the tree. For each fundamental cycle, if it
    overlaps with any previous fundamental cycle, just take the initial
    non-overlapping segment, which is a path instead of a cycle. Each
    cycle or path is called a *chain*. For more information, see [1]_.

    Parameters
    ----------
    G : undirected graph

    root : node (optional)
       A node in the graph `G`. If specified, only the chain
       decomposition for the connected component containing this node
       will be returned. This node indicates the root of the depth-first
       search tree.

    Yields
    ------
    chain : list
       A list of edges representing a chain. There is no guarantee on
       the orientation of the edges in each chain (for example, if a
       chain includes the edge joining nodes 1 and 2, the chain may
       include either (1, 2) or (2, 1)).

    Raises
    ------
    NodeNotFound
       If `root` is not in the graph `G`.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> list(nx.chain_decomposition(G))
    [[(4, 5), (5, 3), (3, 4)]]

    Notes
    -----
    The worst-case running time of this implementation is linear in the
    number of nodes and number of edges [1]_.

    References
    ----------
    .. [1] Jens M. Schmidt (2013). "A simple test on 2-vertex-
       and 2-edge-connectivity." *Information Processing Letters*,
       113, 241–244. Elsevier. <https://doi.org/10.1016/j.ipl.2013.01.016>

    """

    def _dfs_cycle_forest(G, root=None):
        """Builds a directed graph composed of cycles from the given graph.

        `G` is an undirected simple graph. `root` is a node in the graph
        from which the depth-first search is started.

        This function returns both the depth-first search cycle graph
        (as a :class:`~networkx.DiGraph`) and the list of nodes in
        depth-first preorder. The depth-first search cycle graph is a
        directed graph whose edges are the edges of `G` oriented toward
        the root if the edge is a tree edge and away from the root if
        the edge is a non-tree edge. If `root` is not specified, this
        performs a depth-first search on each connected component of `G`
        and returns a directed forest instead.

        If `root` is not in the graph, this raises :exc:`KeyError`.

        """
        # Create a directed graph from the depth-first search tree with
        # root node `root` in which tree edges are directed toward the
        # root and nontree edges are directed away from the root. For
        # each node with an incident nontree edge, this creates a
        # directed cycle starting with the nontree edge and returning to
        # that node.
        #
        # The `parent` node attribute stores the parent of each node in
        # the DFS tree. The `nontree` edge attribute indicates whether
        # the edge is a tree edge or a nontree edge.
        #
        # We also store the order of the nodes found in the depth-first
        # search in the `nodes` list.
        H = nx.DiGraph()
        nodes = []
        for u, v, d in nx.dfs_labeled_edges(G, source=root):
            if d == "forward":
                # `dfs_labeled_edges()` yields (root, root, 'forward')
                # if it is beginning the search on a new connected
                # component.
                if u == v:
                    H.add_node(v, parent=None)
                    nodes.append(v)
                else:
                    H.add_node(v, parent=u)
                    H.add_edge(v, u, nontree=False)
                    nodes.append(v)
            # `dfs_labeled_edges` considers nontree edges in both
            # orientations, so we need to not add the edge if it its
            # other orientation has been added.
            elif d == "nontree" and v not in H[u]:
                H.add_edge(v, u, nontree=True)
            else:
                # Do nothing on 'reverse' edges; we only care about
                # forward and nontree edges.
                pass
        return H, nodes

    def _build_chain(G, u, v, visited):
        """Generate the chain starting from the given nontree edge.

        `G` is a DFS cycle graph as constructed by
        :func:`_dfs_cycle_graph`. The edge (`u`, `v`) is a nontree edge
        that begins a chain. `visited` is a set representing the nodes
        in `G` that have already been visited.

        This function yields the edges in an initial segment of the
        fundamental cycle of `G` starting with the nontree edge (`u`,
        `v`) that includes all the edges up until the first node that
        appears in `visited`. The tree edges are given by the 'parent'
        node attribute. The `visited` set is updated to add each node in
        an edge yielded by this function.

        """
        while v not in visited:
            yield u, v
            visited.add(v)
            u, v = v, G.nodes[v]["parent"]
        yield u, v

    # Check if the root is in the graph G. If not, raise NodeNotFound
    if root is not None and root not in G:
        raise nx.NodeNotFound(f"Root node {root} is not in graph")

    # Create a directed version of H that has the DFS edges directed
    # toward the root and the nontree edges directed away from the root
    # (in each connected component).
    H, nodes = _dfs_cycle_forest(G, root)

    # Visit the nodes again in DFS order. For each node, and for each
    # nontree edge leaving that node, compute the fundamental cycle for
    # that nontree edge starting with that edge. If the fundamental
    # cycle overlaps with any visited nodes, just take the prefix of the
    # cycle up to the point of visited nodes.
    #
    # We repeat this process for each connected component (implicitly,
    # since `nodes` already has a list of the nodes grouped by connected
    # component).
    visited = set()
    for u in nodes:
        visited.add(u)
        # For each nontree edge going out of node u...
        edges = ((u, v) for u, v, d in H.out_edges(u, data="nontree") if d)
        for u, v in edges:
            # Create the cycle or cycle prefix starting with the
            # nontree edge.
            chain = list(_build_chain(H, u, v, visited))
            yield chain
<|endoftext|>"""Algorithms to calculate reciprocity in a directed graph."""

import networkx as nx
from networkx import NetworkXError

from ..utils import not_implemented_for

__all__ = ["reciprocity", "overall_reciprocity"]


@not_implemented_for("undirected", "multigraph")
@nx._dispatchable
def reciprocity(G, nodes=None):
    r"""Compute the reciprocity in a directed graph.

    The reciprocity of a directed graph is defined as the ratio
    of the number of edges pointing in both directions to the total
    number of edges in the graph.
    Formally, $r = |{(u,v) \in G|(v,u) \in G}| / |{(u,v) \in G}|$.

    The reciprocity of a single node u is defined similarly,
    it is the ratio of the number of edges in both directions to
    the total number of edges attached to node u.

    Parameters
    ----------
    G : graph
       A networkx directed graph
    nodes : container of nodes, optional (default=whole graph)
       Compute reciprocity for nodes in this container.

    Returns
    -------
    out : dictionary
       Reciprocity keyed by node label.

    Notes
    -----
    The reciprocity is not defined for isolated nodes.
    In such cases this function will return None.

    """
    # If `nodes` is not specified, calculate the reciprocity of the graph.
    if nodes is None:
        return overall_reciprocity(G)

    # If `nodes` represents a single node in the graph, return only its
    # reciprocity.
    if nodes in G:
        reciprocity = next(_reciprocity_iter(G, nodes))[1]
        if reciprocity is None:
            raise NetworkXError("Not defined for isolated nodes.")
        else:
            return reciprocity

    # Otherwise, `nodes` represents an iterable of nodes, so return a
    # dictionary mapping node to its reciprocity.
    return dict(_reciprocity_iter(G, nodes))


def _reciprocity_iter(G, nodes):
    """Return an iterator of (node, reciprocity)."""
    n = G.nbunch_iter(nodes)
    for node in n:
        pred = set(G.predecessors(node))
        succ = set(G.successors(node))
        overlap = pred & succ
        n_total = len(pred) + len(succ)

        # Reciprocity is not defined for isolated nodes.
        # Return None.
        if n_total == 0:
            yield (node, None)
        else:
            reciprocity = 2 * len(overlap) / n_total
            yield (node, reciprocity)


@not_implemented_for("undirected", "multigraph")
@nx._dispatchable
def overall_reciprocity(G):
    """Compute the reciprocity for the whole graph.

    See the doc of reciprocity for the definition.

    Parameters
    ----------
    G : graph
       A networkx graph

    """
    n_all_edge = G.number_of_edges()
    n_overlap_edge = (n_all_edge - G.to_undirected().number_of_edges()) * 2

    if n_all_edge == 0:
        raise NetworkXError("Not defined for empty graphs")

    return n_overlap_edge / n_all_edge
<|endoftext|>"""
Communicability.
"""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["communicability", "communicability_exp"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def communicability(G):
    r"""Returns communicability between all pairs of nodes in G.

    The communicability between pairs of nodes in G is the sum of
    walks of different lengths starting at node u and ending at node v.

    Parameters
    ----------
    G: graph

    Returns
    -------
    comm: dictionary of dictionaries
        Dictionary of dictionaries keyed by nodes with communicability
        as the value.

    Raises
    ------
    NetworkXError
       If the graph is not undirected and simple.

    See Also
    --------
    communicability_exp:
       Communicability between all pairs of nodes in G  using spectral
       decomposition.
    communicability_betweenness_centrality:
       Communicability betweenness centrality for each node in G.

    Notes
    -----
    This algorithm uses a spectral decomposition of the adjacency matrix.
    Let G=(V,E) be a simple undirected graph.  Using the connection between
    the powers  of the adjacency matrix and the number of walks in the graph,
    the communicability  between nodes `u` and `v` based on the graph spectrum
    is [1]_

    .. math::
        C(u,v)=\sum_{j=1}^{n}\phi_{j}(u)\phi_{j}(v)e^{\lambda_{j}},

    where `\phi_{j}(u)` is the `u\rm{th}` element of the `j\rm{th}` orthonormal
    eigenvector of the adjacency matrix associated with the eigenvalue
    `\lambda_{j}`.

    References
    ----------
    .. [1] Ernesto Estrada, Naomichi Hatano,
       "Communicability in complex networks",
       Phys. Rev. E 77, 036111 (2008).
       https://arxiv.org/abs/0707.0756

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])
    >>> c = nx.communicability(G)
    """
    import numpy as np

    nodelist = list(G)  # ordering of nodes in matrix
    A = nx.to_numpy_array(G, nodelist)
    # convert to 0-1 matrix
    A[A != 0.0] = 1
    w, vec = np.linalg.eigh(A)
    expw = np.exp(w)
    mapping = dict(zip(nodelist, range(len(nodelist))))
    c = {}
    # computing communicabilities
    for u in G:
        c[u] = {}
        for v in G:
            s = 0
            p = mapping[u]
            q = mapping[v]
            for j in range(len(nodelist)):
                s += vec[:, j][p] * vec[:, j][q] * expw[j]
            c[u][v] = float(s)
    return c


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def communicability_exp(G):
    r"""Returns communicability between all pairs of nodes in G.

    Communicability between pair of node (u,v) of node in G is the sum of
    walks of different lengths starting at node u and ending at node v.

    Parameters
    ----------
    G: graph

    Returns
    -------
    comm: dictionary of dictionaries
        Dictionary of dictionaries keyed by nodes with communicability
        as the value.

    Raises
    ------
    NetworkXError
        If the graph is not undirected and simple.

    See Also
    --------
    communicability:
       Communicability between pairs of nodes in G.
    communicability_betweenness_centrality:
       Communicability betweenness centrality for each node in G.

    Notes
    -----
    This algorithm uses matrix exponentiation of the adjacency matrix.

    Let G=(V,E) be a simple undirected graph.  Using the connection between
    the powers  of the adjacency matrix and the number of walks in the graph,
    the communicability between nodes u and v is [1]_,

    .. math::
        C(u,v) = (e^A)_{uv},

    where `A` is the adjacency matrix of G.

    References
    ----------
    .. [1] Ernesto Estrada, Naomichi Hatano,
       "Communicability in complex networks",
       Phys. Rev. E 77, 036111 (2008).
       https://arxiv.org/abs/0707.0756

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])
    >>> c = nx.communicability_exp(G)
    """
    import scipy as sp

    nodelist = list(G)  # ordering of nodes in matrix
    A = nx.to_numpy_array(G, nodelist)
    # convert to 0-1 matrix
    A[A != 0.0] = 1
    # communicability matrix
    expA = sp.linalg.expm(A)
    mapping = dict(zip(nodelist, range(len(nodelist))))
    c = {}
    for u in G:
        c[u] = {}
        for v in G:
            c[u][v] = float(expA[mapping[u], mapping[v]])
    return c
<|endoftext|>from collections import defaultdict

import networkx as nx

__all__ = ["combinatorial_embedding_to_pos"]


def combinatorial_embedding_to_pos(embedding, fully_triangulate=False):
    """Assigns every node a (x, y) position based on the given embedding

    The algorithm iteratively inserts nodes of the input graph in a certain
    order and rearranges previously inserted nodes so that the planar drawing
    stays valid. This is done efficiently by only maintaining relative
    positions during the node placements and calculating the absolute positions
    at the end. For more information see [1]_.

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
        This defines the order of the edges

    fully_triangulate : bool
        If set to True the algorithm adds edges to a copy of the input
        embedding and makes it chordal.

    Returns
    -------
    pos : dict
        Maps each node to a tuple that defines the (x, y) position

    References
    ----------
    .. [1] M. Chrobak and T.H. Payne:
        A Linear-time Algorithm for Drawing a Planar Graph on a Grid 1989
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.6677

    """
    if len(embedding.nodes()) < 4:
        # Position the node in any triangle
        default_positions = [(0, 0), (2, 0), (1, 1)]
        pos = {}
        for i, v in enumerate(embedding.nodes()):
            pos[v] = default_positions[i]
        return pos

    embedding, outer_face = triangulate_embedding(embedding, fully_triangulate)

    # The following dicts map a node to another node
    # If a node is not in the key set it means that the node is not yet in G_k
    # If a node maps to None then the corresponding subtree does not exist
    left_t_child = {}
    right_t_child = {}

    # The following dicts map a node to an integer
    delta_x = {}
    y_coordinate = {}

    node_list = get_canonical_ordering(embedding, outer_face)

    # 1. Phase: Compute relative positions

    # Initialization
    v1, v2, v3 = node_list[0][0], node_list[1][0], node_list[2][0]

    delta_x[v1] = 0
    y_coordinate[v1] = 0
    right_t_child[v1] = v3
    left_t_child[v1] = None

    delta_x[v2] = 1
    y_coordinate[v2] = 0
    right_t_child[v2] = None
    left_t_child[v2] = None

    delta_x[v3] = 1
    y_coordinate[v3] = 1
    right_t_child[v3] = v2
    left_t_child[v3] = None

    for k in range(3, len(node_list)):
        vk, contour_nbrs = node_list[k]
        wp = contour_nbrs[0]
        wp1 = contour_nbrs[1]
        wq = contour_nbrs[-1]
        wq1 = contour_nbrs[-2]
        adds_mult_tri = len(contour_nbrs) > 2

        # Stretch gaps:
        delta_x[wp1] += 1
        delta_x[wq] += 1

        delta_x_wp_wq = sum(delta_x[x] for x in contour_nbrs[1:])

        # Adjust offsets
        delta_x[vk] = (-y_coordinate[wp] + delta_x_wp_wq + y_coordinate[wq]) // 2
        y_coordinate[vk] = (y_coordinate[wp] + delta_x_wp_wq + y_coordinate[wq]) // 2
        delta_x[wq] = delta_x_wp_wq - delta_x[vk]
        if adds_mult_tri:
            delta_x[wp1] -= delta_x[vk]

        # Install v_k:
        right_t_child[wp] = vk
        right_t_child[vk] = wq
        if adds_mult_tri:
            left_t_child[vk] = wp1
            right_t_child[wq1] = None
        else:
            left_t_child[vk] = None

    # 2. Phase: Set absolute positions
    pos = {}
    pos[v1] = (0, y_coordinate[v1])
    remaining_nodes = [v1]
    while remaining_nodes:
        parent_node = remaining_nodes.pop()

        # Calculate position for left child
        set_position(
            parent_node, left_t_child, remaining_nodes, delta_x, y_coordinate, pos
        )
        # Calculate position for right child
        set_position(
            parent_node, right_t_child, remaining_nodes, delta_x, y_coordinate, pos
        )
    return pos


def set_position(parent, tree, remaining_nodes, delta_x, y_coordinate, pos):
    """Helper method to calculate the absolute position of nodes."""
    child = tree[parent]
    parent_node_x = pos[parent][0]
    if child is not None:
        # Calculate pos of child
        child_x = parent_node_x + delta_x[child]
        pos[child] = (child_x, y_coordinate[child])
        # Remember to calculate pos of its children
        remaining_nodes.append(child)


def get_canonical_ordering(embedding, outer_face):
    """Returns a canonical ordering of the nodes

    The canonical ordering of nodes (v1, ..., vn) must fulfill the following
    conditions:
    (See Lemma 1 in [2]_)

    - For the subgraph G_k of the input graph induced by v1, ..., vk it holds:
        - 2-connected
        - internally triangulated
        - the edge (v1, v2) is part of the outer face
    - For a node v(k+1) the following holds:
        - The node v(k+1) is part of the outer face of G_k
        - It has at least two neighbors in G_k
        - All neighbors of v(k+1) in G_k lie consecutively on the outer face of
          G_k (excluding the edge (v1, v2)).

    The algorithm used here starts with G_n (containing all nodes). It first
    selects the nodes v1 and v2. And then tries to find the order of the other
    nodes by checking which node can be removed in order to fulfill the
    conditions mentioned above. This is done by calculating the number of
    chords of nodes on the outer face. For more information see [1]_.

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
        The embedding must be triangulated
    outer_face : list
        The nodes on the outer face of the graph

    Returns
    -------
    ordering : list
        A list of tuples `(vk, wp_wq)`. Here `vk` is the node at this position
        in the canonical ordering. The element `wp_wq` is a list of nodes that
        make up the outer face of G_k.

    References
    ----------
    .. [1] Steven Chaplick.
        Canonical Orders of Planar Graphs and (some of) Their Applications 2015
        https://wuecampus2.uni-wuerzburg.de/moodle/pluginfile.php/545727/mod_resource/content/0/vg-ss15-vl03-canonical-orders-druckversion.pdf
    .. [2] M. Chrobak and T.H. Payne:
        A Linear-time Algorithm for Drawing a Planar Graph on a Grid 1989
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.6677

    """
    v1 = outer_face[0]
    v2 = outer_face[1]
    chords = defaultdict(int)  # Maps nodes to the number of their chords
    marked_nodes = set()
    ready_to_pick = set(outer_face)

    # Initialize outer_face_ccw_nbr (do not include v1 -> v2)
    outer_face_ccw_nbr = {}
    prev_nbr = v2
    for idx in range(2, len(outer_face)):
        outer_face_ccw_nbr[prev_nbr] = outer_face[idx]
        prev_nbr = outer_face[idx]
    outer_face_ccw_nbr[prev_nbr] = v1

    # Initialize outer_face_cw_nbr (do not include v2 -> v1)
    outer_face_cw_nbr = {}
    prev_nbr = v1
    for idx in range(len(outer_face) - 1, 0, -1):
        outer_face_cw_nbr[prev_nbr] = outer_face[idx]
        prev_nbr = outer_face[idx]

    def is_outer_face_nbr(x, y):
        if x not in outer_face_ccw_nbr:
            return outer_face_cw_nbr[x] == y
        if x not in outer_face_cw_nbr:
            return outer_face_ccw_nbr[x] == y
        return outer_face_ccw_nbr[x] == y or outer_face_cw_nbr[x] == y

    def is_on_outer_face(x):
        return x not in marked_nodes and (x in outer_face_ccw_nbr or x == v1)

    # Initialize number of chords
    for v in outer_face:
        for nbr in embedding.neighbors_cw_order(v):
            if is_on_outer_face(nbr) and not is_outer_face_nbr(v, nbr):
                chords[v] += 1
                ready_to_pick.discard(v)

    # Initialize canonical_ordering
    canonical_ordering = [None] * len(embedding.nodes())
    canonical_ordering[0] = (v1, [])
    canonical_ordering[1] = (v2, [])
    ready_to_pick.discard(v1)
    ready_to_pick.discard(v2)

    for k in range(len(embedding.nodes()) - 1, 1, -1):
        # 1. Pick v from ready_to_pick
        v = ready_to_pick.pop()
        marked_nodes.add(v)

        # v has exactly two neighbors on the outer face (wp and wq)
        wp = None
        wq = None
        # Iterate over neighbors of v to find wp and wq
        nbr_iterator = iter(embedding.neighbors_cw_order(v))
        while True:
            nbr = next(nbr_iterator)
            if nbr in marked_nodes:
                # Only consider nodes that are not yet removed
                continue
            if is_on_outer_face(nbr):
                # nbr is either wp or wq
                if nbr == v1:
                    wp = v1
                elif nbr == v2:
                    wq = v2
                else:
                    if outer_face_cw_nbr[nbr] == v:
                        # nbr is wp
                        wp = nbr
                    else:
                        # nbr is wq
                        wq = nbr
            if wp is not None and wq is not None:
                # We don't need to iterate any further
                break

        # Obtain new nodes on outer face (neighbors of v from wp to wq)
        wp_wq = [wp]
        nbr = wp
        while nbr != wq:
            # Get next neighbor (clockwise on the outer face)
            next_nbr = embedding[v][nbr]["ccw"]
            wp_wq.append(next_nbr)
            # Update outer face
            outer_face_cw_nbr[nbr] = next_nbr
            outer_face_ccw_nbr[next_nbr] = nbr
            # Move to next neighbor of v
            nbr = next_nbr

        if len(wp_wq) == 2:
            # There was a chord between wp and wq, decrease number of chords
            chords[wp] -= 1
            if chords[wp] == 0:
                ready_to_pick.add(wp)
            chords[wq] -= 1
            if chords[wq] == 0:
                ready_to_pick.add(wq)
        else:
            # Update all chords involving w_(p+1) to w_(q-1)
            new_face_nodes = set(wp_wq[1:-1])
            for w in new_face_nodes:
                # If we do not find a chord for w later we can pick it next
                ready_to_pick.add(w)
                for nbr in embedding.neighbors_cw_order(w):
                    if is_on_outer_face(nbr) and not is_outer_face_nbr(w, nbr):
                        # There is a chord involving w
                        chords[w] += 1
                        ready_to_pick.discard(w)
                        if nbr not in new_face_nodes:
                            # Also increase chord for the neighbor
                            # We only iterator over new_face_nodes
                            chords[nbr] += 1
                            ready_to_pick.discard(nbr)
        # Set the canonical ordering node and the list of contour neighbors
        canonical_ordering[k] = (v, wp_wq)

    return canonical_ordering


def triangulate_face(embedding, v1, v2):
    """Triangulates the face given by half edge (v, w)

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
    v1 : node
        The half-edge (v1, v2) belongs to the face that gets triangulated
    v2 : node
    """
    _, v3 = embedding.next_face_half_edge(v1, v2)
    _, v4 = embedding.next_face_half_edge(v2, v3)
    if v1 in (v2, v3):
        # The component has less than 3 nodes
        return
    while v1 != v4:
        # Add edge if not already present on other side
        if embedding.has_edge(v1, v3):
            # Cannot triangulate at this position
            v1, v2, v3 = v2, v3, v4
        else:
            # Add edge for triangulation
            embedding.add_half_edge(v1, v3, ccw=v2)
            embedding.add_half_edge(v3, v1, cw=v2)
            v1, v2, v3 = v1, v3, v4
        # Get next node
        _, v4 = embedding.next_face_half_edge(v2, v3)


def triangulate_embedding(embedding, fully_triangulate=True):
    """Triangulates the embedding.

    Traverses faces of the embedding and adds edges to a copy of the
    embedding to triangulate it.
    The method also ensures that the resulting graph is 2-connected by adding
    edges if the same vertex is contained twice on a path around a face.

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
        The input graph must contain at least 3 nodes.

    fully_triangulate : bool
        If set to False the face with the most nodes is chooses as outer face.
        This outer face does not get triangulated.

    Returns
    -------
    (embedding, outer_face) : (nx.PlanarEmbedding, list) tuple
        The element `embedding` is a new embedding containing all edges from
        the input embedding and the additional edges to triangulate the graph.
        The element `outer_face` is a list of nodes that lie on the outer face.
        If the graph is fully triangulated these are three arbitrary connected
        nodes.

    """
    if len(embedding.nodes) <= 1:
        return embedding, list(embedding.nodes)
    embedding = nx.PlanarEmbedding(embedding)

    # Get a list with a node for each connected component
    component_nodes = [next(iter(x)) for x in nx.connected_components(embedding)]

    # 1. Make graph a single component (add edge between components)
    for i in range(len(component_nodes) - 1):
        v1 = component_nodes[i]
        v2 = component_nodes[i + 1]
        embedding.connect_components(v1, v2)

    # 2. Calculate faces, ensure 2-connectedness and determine outer face
    outer_face = []  # A face with the most number of nodes
    face_list = []
    edges_visited = set()  # Used to keep track of already visited faces
    for v in embedding.nodes():
        for w in embedding.neighbors_cw_order(v):
            new_face = make_bi_connected(embedding, v, w, edges_visited)
            if new_face:
                # Found a new face
                face_list.append(new_face)
                if len(new_face) > len(outer_face):
                    # The face is a candidate to be the outer face
                    outer_face = new_face

    # 3. Triangulate (internal) faces
    for face in face_list:
        if face is not outer_face or fully_triangulate:
            # Triangulate this face
            triangulate_face(embedding, face[0], face[1])

    if fully_triangulate:
        v1 = outer_face[0]
        v2 = outer_face[1]
        v3 = embedding[v2][v1]["ccw"]
        outer_face = [v1, v2, v3]

    return embedding, outer_face


def make_bi_connected(embedding, starting_node, outgoing_node, edges_counted):
    """Triangulate a face and make it 2-connected

    This method also adds all edges on the face to `edges_counted`.

    Parameters
    ----------
    embedding: nx.PlanarEmbedding
        The embedding that defines the faces
    starting_node : node
        A node on the face
    outgoing_node : node
        A node such that the half edge (starting_node, outgoing_node) belongs
        to the face
    edges_counted: set
        Set of all half-edges that belong to a face that have been visited

    Returns
    -------
    face_nodes: list
        A list of all nodes at the border of this face
    """

    # Check if the face has already been calculated
    if (starting_node, outgoing_node) in edges_counted:
        # This face was already counted
        return []
    edges_counted.add((starting_node, outgoing_node))

    # Add all edges to edges_counted which have this face to their left
    v1 = starting_node
    v2 = outgoing_node
    face_list = [starting_node]  # List of nodes around the face
    face_set = set(face_list)  # Set for faster queries
    _, v3 = embedding.next_face_half_edge(v1, v2)

    # Move the nodes v1, v2, v3 around the face:
    while v2 != starting_node or v3 != outgoing_node:
        if v1 == v2:
            raise nx.NetworkXException("Invalid half-edge")
        # cycle is not completed yet
        if v2 in face_set:
            # v2 encountered twice: Add edge to ensure 2-connectedness
            embedding.add_half_edge(v1, v3, ccw=v2)
            embedding.add_half_edge(v3, v1, cw=v2)
            edges_counted.add((v2, v3))
            edges_counted.add((v3, v1))
            v2 = v1
        else:
            face_set.add(v2)
            face_list.append(v2)

        # set next edge
        v1 = v2
        v2, v3 = embedding.next_face_half_edge(v2, v3)

        # remember that this edge has been counted
        edges_counted.add((v1, v2))

    return face_list
<|endoftext|>"""
Algorithm for testing d-separation in DAGs.

*d-separation* is a test for conditional independence in probability
distributions that can be factorized using DAGs.  It is a purely
graphical test that uses the underlying graph and makes no reference
to the actual distribution parameters.  See [1]_ for a formal
definition.

The implementation is based on the conceptually simple linear time
algorithm presented in [2]_.  Refer to [3]_, [4]_ for a couple of
alternative algorithms.

The functional interface in NetworkX consists of three functions:

- `find_minimal_d_separator` returns a minimal d-separator set ``z``.
  That is, removing any node or nodes from it makes it no longer a d-separator.
- `is_d_separator` checks if a given set is a d-separator.
- `is_minimal_d_separator` checks if a given set is a minimal d-separator.

D-separators
------------

Here, we provide a brief overview of d-separation and related concepts that
are relevant for understanding it:

The ideas of d-separation and d-connection relate to paths being open or blocked.

- A "path" is a sequence of nodes connected in order by edges. Unlike for most
  graph theory analysis, the direction of the edges is ignored. Thus the path
  can be thought of as a traditional path on the undirected version of the graph.
- A "candidate d-separator" ``z`` is a set of nodes being considered as
  possibly blocking all paths between two prescribed sets ``x`` and ``y`` of nodes.
  We refer to each node in the candidate d-separator as "known".
- A "collider" node on a path is a node that is a successor of its two neighbor
  nodes on the path. That is, ``c`` is a collider if the edge directions
  along the path look like ``... u -> c <- v ...``.
- If a collider node or any of its descendants are "known", the collider
  is called an "open collider". Otherwise it is a "blocking collider".
- Any path can be "blocked" in two ways. If the path contains a "known" node
  that is not a collider, the path is blocked. Also, if the path contains a
  collider that is not a "known" node, the path is blocked.
- A path is "open" if it is not blocked. That is, it is open if every node is
  either an open collider or not a "known". Said another way, every
  "known" in the path is a collider and every collider is open (has a
  "known" as a inclusive descendant). The concept of "open path" is meant to
  demonstrate a probabilistic conditional dependence between two nodes given
  prescribed knowledge ("known" nodes).
- Two sets ``x`` and ``y`` of nodes are "d-separated" by a set of nodes ``z``
  if all paths between nodes in ``x`` and nodes in ``y`` are blocked. That is,
  if there are no open paths from any node in ``x`` to any node in ``y``.
  Such a set ``z`` is a "d-separator" of ``x`` and ``y``.
- A "minimal d-separator" is a d-separator ``z`` for which no node or subset
  of nodes can be removed with it still being a d-separator.

The d-separator blocks some paths between ``x`` and ``y`` but opens others.
Nodes in the d-separator block paths if the nodes are not colliders.
But if a collider or its descendant nodes are in the d-separation set, the
colliders are open, allowing a path through that collider.

Illustration of D-separation with examples
------------------------------------------

A pair of two nodes, ``u`` and ``v``, are d-connected if there is a path
from ``u`` to ``v`` that is not blocked. That means, there is an open
path from ``u`` to ``v``.

For example, if the d-separating set is the empty set, then the following paths are
open between ``u`` and ``v``:

- u <- n -> v
- u -> w -> ... -> n -> v

If  on the other hand, ``n`` is in the d-separating set, then ``n`` blocks
those paths between ``u`` and ``v``.

Colliders block a path if they and their descendants are not included
in the d-separating set. An example of a path that is blocked when the
d-separating set is empty is:

- u -> w -> ... -> n <- v

The node ``n`` is a collider in this path and is not in the d-separating set.
So ``n`` blocks this path. However, if ``n`` or a descendant of ``n`` is
included in the d-separating set, then the path through the collider
at ``n`` (... -> n <- ...) is "open".

D-separation is concerned with blocking all paths between nodes from ``x`` to ``y``.
A d-separating set between ``x`` and ``y`` is one where all paths are blocked.

D-separation and its applications in probability
------------------------------------------------

D-separation is commonly used in probabilistic causal-graph models. D-separation
connects the idea of probabilistic "dependence" with separation in a graph. If
one assumes the causal Markov condition [5]_, (every node is conditionally
independent of its non-descendants, given its parents) then d-separation implies
conditional independence in probability distributions.
Symmetrically, d-connection implies dependence.

The intuition is as follows. The edges on a causal graph indicate which nodes
influence the outcome of other nodes directly. An edge from u to v
implies that the outcome of event ``u`` influences the probabilities for
the outcome of event ``v``. Certainly knowing ``u`` changes predictions for ``v``.
But also knowing ``v`` changes predictions for ``u``. The outcomes are dependent.
Furthermore, an edge from ``v`` to ``w`` would mean that ``w`` and ``v`` are dependent
and thus that ``u`` could indirectly influence ``w``.

Without any knowledge about the system (candidate d-separating set is empty)
a causal graph ``u -> v -> w`` allows all three nodes to be dependent. But
if we know the outcome of ``v``, the conditional probabilities of outcomes for
``u`` and ``w`` are independent of each other. That is, once we know the outcome
for ```v`, the probabilities for ``w`` do not depend on the outcome for ``u``.
This is the idea behind ``v`` blocking the path if it is "known" (in the candidate
d-separating set).

The same argument works whether the direction of the edges are both
left-going and when both arrows head out from the middle. Having a "known"
node on a path blocks the collider-free path because those relationships
make the conditional probabilities independent.

The direction of the causal edges does impact dependence precisely in the
case of a collider e.g. ``u -> v <- w``. In that situation, both ``u`` and ``w``
influence ``v```. But they do not directly influence each other. So without any
knowledge of any outcomes, ``u`` and ``w`` are independent. That is the idea behind
colliders blocking the path. But, if ``v`` is known, the conditional probabilities
of ``u`` and ``w`` can be dependent. This is the heart of Berkson's Paradox [6]_.
For example, suppose ``u`` and ``w`` are boolean events (they either happen or do not)
and ``v`` represents the outcome "at least one of ``u`` and ``w`` occur". Then knowing
``v`` is true makes the conditional probabilities of ``u`` and ``w`` dependent.
Essentially, knowing that at least one of them is true raises the probability of
each. But further knowledge that ``w`` is true (or false) change the conditional
probability of ``u`` to either the original value or 1. So the conditional
probability of ``u`` depends on the outcome of ``w`` even though there is no
causal relationship between them. When a collider is known, dependence can
occur across paths through that collider. This is the reason open colliders
do not block paths.

Furthermore, even if ``v`` is not "known", if one of its descendants is "known"
we can use that information to know more about ``v`` which again makes
``u`` and ``w`` potentially dependent. Suppose the chance of ``n`` occurring
is much higher when ``v`` occurs ("at least one of ``u`` and ``w`` occur").
Then if we know ``n`` occurred, it is more likely that ``v`` occurred and that
makes the chance of ``u`` and ``w`` dependent. This is the idea behind why
a collider does no block a path if any descendant of the collider is "known".

When two sets of nodes ``x`` and ``y`` are d-separated by a set ``z``,
it means that given the outcomes of the nodes in ``z``, the probabilities
of outcomes of the nodes in ``x`` are independent of the outcomes of the
nodes in ``y`` and vice versa.

Examples
--------
A Hidden Markov Model with 5 observed states and 5 hidden states
where the hidden states have causal relationships resulting in
a path results in the following causal network. We check that
early states along the path are separated from late state in
the path by the d-separator of the middle hidden state.
Thus if we condition on the middle hidden state, the early
state probabilities are independent of the late state outcomes.

>>> G = nx.DiGraph()
>>> G.add_edges_from(
...     [
...         ("H1", "H2"),
...         ("H2", "H3"),
...         ("H3", "H4"),
...         ("H4", "H5"),
...         ("H1", "O1"),
...         ("H2", "O2"),
...         ("H3", "O3"),
...         ("H4", "O4"),
...         ("H5", "O5"),
...     ]
... )
>>> x, y, z = ({"H1", "O1"}, {"H5", "O5"}, {"H3"})
>>> nx.is_d_separator(G, x, y, z)
True
>>> nx.is_minimal_d_separator(G, x, y, z)
True
>>> nx.is_minimal_d_separator(G, x, y, z | {"O3"})
False
>>> z = nx.find_minimal_d_separator(G, x | y, {"O2", "O3", "O4"})
>>> z == {"H2", "H4"}
True

If no minimal_d_separator exists, `None` is returned

>>> other_z = nx.find_minimal_d_separator(G, x | y, {"H2", "H3"})
>>> other_z is None
True


References
----------

.. [1] Pearl, J.  (2009).  Causality.  Cambridge: Cambridge University Press.

.. [2] Darwiche, A.  (2009).  Modeling and reasoning with Bayesian networks.
   Cambridge: Cambridge University Press.

.. [3] Shachter, Ross D. "Bayes-ball: The rational pastime (for
   determining irrelevance and requisite information in belief networks
   and influence diagrams)." In Proceedings of the Fourteenth Conference
   on Uncertainty in Artificial Intelligence (UAI), (pp. 480–487). 1998.

.. [4] Koller, D., & Friedman, N. (2009).
   Probabilistic graphical models: principles and techniques. The MIT Press.

.. [5] https://en.wikipedia.org/wiki/Causal_Markov_condition

.. [6] https://en.wikipedia.org/wiki/Berkson%27s_paradox

"""

from collections import deque
from itertools import chain

import networkx as nx
from networkx.utils import UnionFind, not_implemented_for

__all__ = [
    "is_d_separator",
    "is_minimal_d_separator",
    "find_minimal_d_separator",
    "d_separated",
    "minimal_d_separator",
]


@not_implemented_for("undirected")
@nx._dispatchable
def is_d_separator(G, x, y, z):
    """Return whether node sets `x` and `y` are d-separated by `z`.

    Parameters
    ----------
    G : nx.DiGraph
        A NetworkX DAG.

    x : node or set of nodes
        First node or set of nodes in `G`.

    y : node or set of nodes
        Second node or set of nodes in `G`.

    z : node or set of nodes
        Potential separator (set of conditioning nodes in `G`). Can be empty set.

    Returns
    -------
    b : bool
        A boolean that is true if `x` is d-separated from `y` given `z` in `G`.

    Raises
    ------
    NetworkXError
        The *d-separation* test is commonly used on disjoint sets of
        nodes in acyclic directed graphs.  Accordingly, the algorithm
        raises a :exc:`NetworkXError` if the node sets are not
        disjoint or if the input graph is not a DAG.

    NodeNotFound
        If any of the input nodes are not found in the graph,
        a :exc:`NodeNotFound` exception is raised

    Notes
    -----
    A d-separating set in a DAG is a set of nodes that
    blocks all paths between the two sets. Nodes in `z`
    block a path if they are part of the path and are not a collider,
    or a descendant of a collider. Also colliders that are not in `z`
    block a path. A collider structure along a path
    is ``... -> c <- ...`` where ``c`` is the collider node.

    https://en.wikipedia.org/wiki/Bayesian_network#d-separation
    """
    try:
        x = {x} if x in G else x
        y = {y} if y in G else y
        z = {z} if z in G else z

        intersection = x & y or x & z or y & z
        if intersection:
            raise nx.NetworkXError(
                f"The sets are not disjoint, with intersection {intersection}"
            )

        set_v = x | y | z
        if set_v - G.nodes:
            raise nx.NodeNotFound(f"The node(s) {set_v - G.nodes} are not found in G")
    except TypeError:
        raise nx.NodeNotFound("One of x, y, or z is not a node or a set of nodes in G")

    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError("graph should be directed acyclic")

    # contains -> and <-> edges from starting node T
    forward_deque = deque([])
    forward_visited = set()

    # contains <- and - edges from starting node T
    backward_deque = deque(x)
    backward_visited = set()

    ancestors_or_z = set().union(*[nx.ancestors(G, node) for node in x]) | z | x

    while forward_deque or backward_deque:
        if backward_deque:
            node = backward_deque.popleft()
            backward_visited.add(node)
            if node in y:
                return False
            if node in z:
                continue

            # add <- edges to backward deque
            backward_deque.extend(G.pred[node].keys() - backward_visited)
            # add -> edges to forward deque
            forward_deque.extend(G.succ[node].keys() - forward_visited)

        if forward_deque:
            node = forward_deque.popleft()
            forward_visited.add(node)
            if node in y:
                return False

            # Consider if -> node <- is opened due to ancestor of node in z
            if node in ancestors_or_z:
                # add <- edges to backward deque
                backward_deque.extend(G.pred[node].keys() - backward_visited)
            if node not in z:
                # add -> edges to forward deque
                forward_deque.extend(G.succ[node].keys() - forward_visited)

    return True


@not_implemented_for("undirected")
@nx._dispatchable
def find_minimal_d_separator(G, x, y, *, included=None, restricted=None):
    """Returns a minimal d-separating set between `x` and `y` if possible

    A d-separating set in a DAG is a set of nodes that blocks all
    paths between the two sets of nodes, `x` and `y`. This function
    constructs a d-separating set that is "minimal", meaning no nodes can
    be removed without it losing the d-separating property for `x` and `y`.
    If no d-separating sets exist for `x` and `y`, this returns `None`.

    In a DAG there may be more than one minimal d-separator between two
    sets of nodes. Minimal d-separators are not always unique. This function
    returns one minimal d-separator, or `None` if no d-separator exists.

    Uses the algorithm presented in [1]_. The complexity of the algorithm
    is :math:`O(m)`, where :math:`m` stands for the number of edges in
    the subgraph of G consisting of only the ancestors of `x` and `y`.
    For full details, see [1]_.

    Parameters
    ----------
    G : graph
        A networkx DAG.
    x : set | node
        A node or set of nodes in the graph.
    y : set | node
        A node or set of nodes in the graph.
    included : set | node | None
        A node or set of nodes which must be included in the found separating set,
        default is None, which means the empty set.
    restricted : set | node | None
        Restricted node or set of nodes to consider. Only these nodes can be in
        the found separating set, default is None meaning all nodes in ``G``.

    Returns
    -------
    z : set | None
        The minimal d-separating set, if at least one d-separating set exists,
        otherwise None.

    Raises
    ------
    NetworkXError
        Raises a :exc:`NetworkXError` if the input graph is not a DAG
        or if node sets `x`, `y`, and `included` are not disjoint.

    NodeNotFound
        If any of the input nodes are not found in the graph,
        a :exc:`NodeNotFound` exception is raised.

    References
    ----------
    .. [1] van der Zander, Benito, and Maciej Liśkiewicz. "Finding
        minimal d-separators in linear time and applications." In
        Uncertainty in Artificial Intelligence, pp. 637-647. PMLR, 2020.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError("graph should be directed acyclic")

    try:
        x = {x} if x in G else x
        y = {y} if y in G else y

        if included is None:
            included = set()
        elif included in G:
            included = {included}

        if restricted is None:
            restricted = set(G)
        elif restricted in G:
            restricted = {restricted}

        set_y = x | y | included | restricted
        if set_y - G.nodes:
            raise nx.NodeNotFound(f"The node(s) {set_y - G.nodes} are not found in G")
    except TypeError:
        raise nx.NodeNotFound(
            "One of x, y, included or restricted is not a node or set of nodes in G"
        )

    if not included <= restricted:
        raise nx.NetworkXError(
            f"Included nodes {included} must be in restricted nodes {restricted}"
        )

    intersection = x & y or x & included or y & included
    if intersection:
        raise nx.NetworkXError(
            f"The sets x, y, included are not disjoint. Overlap: {intersection}"
        )

    nodeset = x | y | included
    ancestors_x_y_included = nodeset.union(*[nx.ancestors(G, node) for node in nodeset])

    z_init = restricted & (ancestors_x_y_included - (x | y))

    x_closure = _reachable(G, x, ancestors_x_y_included, z_init)
    if x_closure & y:
        return None

    z_updated = z_init & (x_closure | included)
    y_closure = _reachable(G, y, ancestors_x_y_included, z_updated)
    return z_updated & (y_closure | included)


@not_implemented_for("undirected")
@nx._dispatchable
def is_minimal_d_separator(G, x, y, z, *, included=None, restricted=None):
    """Determine if `z` is a minimal d-separator for `x` and `y`.

    A d-separator, `z`, in a DAG is a set of nodes that blocks
    all paths from nodes in set `x` to nodes in set `y`.
    A minimal d-separator is a d-separator `z` such that removing
    any subset of nodes makes it no longer a d-separator.

    Note: This function checks whether `z` is a d-separator AND is
    minimal. One can use the function `is_d_separator` to only check if
    `z` is a d-separator. See examples below.

    Parameters
    ----------
    G : nx.DiGraph
        A NetworkX DAG.
    x : node | set
        A node or set of nodes in the graph.
    y : node | set
        A node or set of nodes in the graph.
    z : node | set
        The node or set of nodes to check if it is a minimal d-separating set.
        The function :func:`is_d_separator` is called inside this function
        to verify that `z` is in fact a d-separator.
    included : set | node | None
        A node or set of nodes which must be included in the found separating set,
        default is ``None``, which means the empty set.
    restricted : set | node | None
        Restricted node or set of nodes to consider. Only these nodes can be in
        the found separating set, default is ``None`` meaning all nodes in ``G``.

    Returns
    -------
    bool
        Whether or not the set `z` is a minimal d-separator subject to
        `restricted` nodes and `included` node constraints.

    Examples
    --------
    >>> G = nx.path_graph([0, 1, 2, 3], create_using=nx.DiGraph)
    >>> G.add_node(4)
    >>> nx.is_minimal_d_separator(G, 0, 2, {1})
    True
    >>> # since {1} is the minimal d-separator, {1, 3, 4} is not minimal
    >>> nx.is_minimal_d_separator(G, 0, 2, {1, 3, 4})
    False
    >>> # alternatively, if we only want to check that {1, 3, 4} is a d-separator
    >>> nx.is_d_separator(G, 0, 2, {1, 3, 4})
    True

    Raises
    ------
    NetworkXError
        Raises a :exc:`NetworkXError` if the input graph is not a DAG.

    NodeNotFound
        If any of the input nodes are not found in the graph,
        a :exc:`NodeNotFound` exception is raised.

    References
    ----------
    .. [1] van der Zander, Benito, and Maciej Liśkiewicz. "Finding
        minimal d-separators in linear time and applications." In
        Uncertainty in Artificial Intelligence, pp. 637-647. PMLR, 2020.

    Notes
    -----
    This function works on verifying that a set is minimal and
    d-separating between two nodes. Uses criterion (a), (b), (c) on
    page 4 of [1]_. a) closure(`x`) and `y` are disjoint. b) `z` contains
    all nodes from `included` and is contained in the `restricted`
    nodes and in the union of ancestors of `x`, `y`, and `included`.
    c) the nodes in `z` not in `included` are contained in both
    closure(x) and closure(y). The closure of a set is the set of nodes
    connected to the set by a directed path in G.

    The complexity is :math:`O(m)`, where :math:`m` stands for the
    number of edges in the subgraph of G consisting of only the
    ancestors of `x` and `y`.

    For full details, see [1]_.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError("graph should be directed acyclic")

    try:
        x = {x} if x in G else x
        y = {y} if y in G else y
        z = {z} if z in G else z

        if included is None:
            included = set()
        elif included in G:
            included = {included}

        if restricted is None:
            restricted = set(G)
        elif restricted in G:
            restricted = {restricted}

        set_y = x | y | included | restricted
        if set_y - G.nodes:
            raise nx.NodeNotFound(f"The node(s) {set_y - G.nodes} are not found in G")
    except TypeError:
        raise nx.NodeNotFound(
            "One of x, y, z, included or restricted is not a node or set of nodes in G"
        )

    if not included <= z:
        raise nx.NetworkXError(
            f"Included nodes {included} must be in proposed separating set z {x}"
        )
    if not z <= restricted:
        raise nx.NetworkXError(
            f"Separating set {z} must be contained in restricted set {restricted}"
        )

    intersection = x.intersection(y) or x.intersection(z) or y.intersection(z)
    if intersection:
        raise nx.NetworkXError(
            f"The sets are not disjoint, with intersection {intersection}"
        )

    nodeset = x | y | included
    ancestors_x_y_included = nodeset.union(*[nx.ancestors(G, n) for n in nodeset])

    # criterion (a) -- check that z is actually a separator
    x_closure = _reachable(G, x, ancestors_x_y_included, z)
    if x_closure & y:
        return False

    # criterion (b) -- basic constraint; included and restricted already checked above
    if not (z <= ancestors_x_y_included):
        return False

    # criterion (c) -- check that z is minimal
    y_closure = _reachable(G, y, ancestors_x_y_included, z)
    if not ((z - included) <= (x_closure & y_closure)):
        return False
    return True


@not_implemented_for("undirected")
def _reachable(G, x, a, z):
    """Modified Bayes-Ball algorithm for finding d-connected nodes.

    Find all nodes in `a` that are d-connected to those in `x` by
    those in `z`. This is an implementation of the function
    `REACHABLE` in [1]_ (which is itself a modification of the
    Bayes-Ball algorithm [2]_) when restricted to DAGs.

    Parameters
    ----------
    G : nx.DiGraph
        A NetworkX DAG.
    x : node | set
        A node in the DAG, or a set of nodes.
    a : node | set
        A (set of) node(s) in the DAG containing the ancestors of `x`.
    z : node | set
        The node or set of nodes conditioned on when checking d-connectedness.

    Returns
    -------
    w : set
        The closure of `x` in `a` with respect to d-connectedness
        given `z`.

    References
    ----------
    .. [1] van der Zander, Benito, and Maciej Liśkiewicz. "Finding
        minimal d-separators in linear time and applications." In
        Uncertainty in Artificial Intelligence, pp. 637-647. PMLR, 2020.

    .. [2] Shachter, Ross D. "Bayes-ball: The rational pastime
       (for determining irrelevance and requisite information in
       belief networks and influence diagrams)." In Proceedings of the
       Fourteenth Conference on Uncertainty in Artificial Intelligence
       (UAI), (pp. 480–487). 1998.
    """

    def _pass(e, v, f, n):
        """Whether a ball entering node `v` along edge `e` passes to `n` along `f`.

        Boolean function defined on page 6 of [1]_.

        Parameters
        ----------
        e : bool
            Directed edge by which the ball got to node `v`; `True` iff directed into `v`.
        v : node
            Node where the ball is.
        f : bool
            Directed edge connecting nodes `v` and `n`; `True` iff directed `n`.
        n : node
            Checking whether the ball passes to this node.

        Returns
        -------
        b : bool
            Whether the ball passes or not.

        References
        ----------
        .. [1] van der Zander, Benito, and Maciej Liśkiewicz. "Finding
           minimal d-separators in linear time and applications." In
           Uncertainty in Artificial Intelligence, pp. 637-647. PMLR, 2020.
        """
        is_element_of_A = n in a
        # almost_definite_status = True  # always true for DAGs; not so for RCGs
        collider_if_in_Z = v not in z or (e and not f)
        return is_element_of_A and collider_if_in_Z  # and almost_definite_status

    queue = deque([])
    for node in x:
        if bool(G.pred[node]):
            queue.append((True, node))
        if bool(G.succ[node]):
            queue.append((False, node))
    processed = queue.copy()

    while any(queue):
        e, v = queue.popleft()
        preds = ((False, n) for n in G.pred[v])
        succs = ((True, n) for n in G.succ[v])
        f_n_pairs = chain(preds, succs)
        for f, n in f_n_pairs:
            if (f, n) not in processed and _pass(e, v, f, n):
                queue.append((f, n))
                processed.append((f, n))

    return {w for (_, w) in processed}


# Deprecated functions:
def d_separated(G, x, y, z):
    """Return whether nodes sets ``x`` and ``y`` are d-separated by ``z``.

    .. deprecated:: 3.3

        This function is deprecated and will be removed in NetworkX v3.5.
        Please use `is_d_separator(G, x, y, z)`.

    """
    import warnings

    warnings.warn(
        "d_separated is deprecated and will be removed in NetworkX v3.5."
        "Please use `is_d_separator(G, x, y, z)`.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return nx.is_d_separator(G, x, y, z)


def minimal_d_separator(G, u, v):
    """Returns a minimal_d-separating set between `x` and `y` if possible

    .. deprecated:: 3.3

        minimal_d_separator is deprecated and will be removed in NetworkX v3.5.
        Please use `find_minimal_d_separator(G, x, y)`.

    """
    import warnings

    warnings.warn(
        (
            "This function is deprecated and will be removed in NetworkX v3.5."
            "Please use `is_d_separator(G, x, y)`."
        ),
        category=DeprecationWarning,
        stacklevel=2,
    )
    return nx.find_minimal_d_separator(G, u, v)
<|endoftext|>"""Bridge-finding algorithms."""

from itertools import chain

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["bridges", "has_bridges", "local_bridges"]


@not_implemented_for("directed")
@nx._dispatchable
def bridges(G, root=None):
    """Generate all bridges in a graph.

    A *bridge* in a graph is an edge whose removal causes the number of
    connected components of the graph to increase.  Equivalently, a bridge is an
    edge that does not belong to any cycle. Bridges are also known as cut-edges,
    isthmuses, or cut arcs.

    Parameters
    ----------
    G : undirected graph

    root : node (optional)
       A node in the graph `G`. If specified, only the bridges in the
       connected component containing this node will be returned.

    Yields
    ------
    e : edge
       An edge in the graph whose removal disconnects the graph (or
       causes the number of connected components to increase).

    Raises
    ------
    NodeNotFound
       If `root` is not in the graph `G`.

    NetworkXNotImplemented
        If `G` is a directed graph.

    Examples
    --------
    The barbell graph with parameter zero has a single bridge:

    >>> G = nx.barbell_graph(10, 0)
    >>> list(nx.bridges(G))
    [(9, 10)]

    Notes
    -----
    This is an implementation of the algorithm described in [1]_.  An edge is a
    bridge if and only if it is not contained in any chain. Chains are found
    using the :func:`networkx.chain_decomposition` function.

    The algorithm described in [1]_ requires a simple graph. If the provided
    graph is a multigraph, we convert it to a simple graph and verify that any
    bridges discovered by the chain decomposition algorithm are not multi-edges.

    Ignoring polylogarithmic factors, the worst-case time complexity is the
    same as the :func:`networkx.chain_decomposition` function,
    $O(m + n)$, where $n$ is the number of nodes in the graph and $m$ is
    the number of edges.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bridge_%28graph_theory%29#Bridge-Finding_with_Chain_Decompositions
    """
    multigraph = G.is_multigraph()
    H = nx.Graph(G) if multigraph else G
    chains = nx.chain_decomposition(H, root=root)
