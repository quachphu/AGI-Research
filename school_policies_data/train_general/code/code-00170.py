        colors the direct links below each untruncated non-singleton node
        ``k`` using ``colors[k]``.
    ax : matplotlib Axes instance, optional
        If None and `no_plot` is not True, the dendrogram will be plotted
        on the current axes.  Otherwise if `no_plot` is not True the
        dendrogram will be plotted on the given ``Axes`` instance. This can be
        useful if the dendrogram is part of a more complex figure.
    above_threshold_color : str, optional
        This matplotlib color string sets the color of the links above the
        color_threshold. The default is ``'C0'``.

    Returns
    -------
    R : dict
        A dictionary of data structures computed to render the
        dendrogram. Its has the following keys:

        ``'color_list'``
          A list of color names. The k'th element represents the color of the
          k'th link.

        ``'icoord'`` and ``'dcoord'``
          Each of them is a list of lists. Let ``icoord = [I1, I2, ..., Ip]``
          where ``Ik = [xk1, xk2, xk3, xk4]`` and ``dcoord = [D1, D2, ..., Dp]``
          where ``Dk = [yk1, yk2, yk3, yk4]``, then the k'th link painted is
          ``(xk1, yk1)`` - ``(xk2, yk2)`` - ``(xk3, yk3)`` - ``(xk4, yk4)``.

        ``'ivl'``
          A list of labels corresponding to the leaf nodes.

        ``'leaves'``
          For each i, ``H[i] == j``, cluster node ``j`` appears in position
          ``i`` in the left-to-right traversal of the leaves, where
          :math:`j < 2n-1` and :math:`i < n`. If ``j`` is less than ``n``, the
          ``i``-th leaf node corresponds to an original observation.
          Otherwise, it corresponds to a non-singleton cluster.

        ``'leaves_color_list'``
          A list of color names. The k'th element represents the color of the
          k'th leaf.

    See Also
    --------
    linkage, set_link_color_palette

    Notes
    -----
    It is expected that the distances in ``Z[:,2]`` be monotonic, otherwise
    crossings appear in the dendrogram.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster import hierarchy
    >>> import matplotlib.pyplot as plt

    A very basic example:

    >>> ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
    ...                    400., 754., 564., 138., 219., 869., 669.])
    >>> Z = hierarchy.linkage(ytdist, 'single')
    >>> plt.figure()
    >>> dn = hierarchy.dendrogram(Z)

    Now, plot in given axes, improve the color scheme and use both vertical and
    horizontal orientations:

    >>> hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
    >>> fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    >>> dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',
    ...                            orientation='top')
    >>> dn2 = hierarchy.dendrogram(Z, ax=axes[1],
    ...                            above_threshold_color='#bcbddc',
    ...                            orientation='right')
    >>> hierarchy.set_link_color_palette(None)  # reset to default after use
    >>> plt.show()

    """
    # This feature was thought about but never implemented (still useful?):
    #
    #         ... = dendrogram(..., leaves_order=None)
    #
    #         Plots the leaves in the order specified by a vector of
    #         original observation indices. If the vector contains duplicates
    #         or results in a crossing, an exception will be thrown. Passing
    #         None orders leaf nodes based on the order they appear in the
    #         pre-order traversal.
    xp = array_namespace(Z)
    Z = _asarray(Z, order='c', xp=xp)

    if orientation not in ["top", "left", "bottom", "right"]:
        raise ValueError("orientation must be one of 'top', 'left', "
                         "'bottom', or 'right'")

    if labels is not None:
        try:
            len_labels = len(labels)
        except (TypeError, AttributeError):
            len_labels = labels.shape[0]
        if Z.shape[0] + 1 != len_labels:
            raise ValueError("Dimensions of Z and labels must be consistent.")

    is_valid_linkage(Z, throw=True, name='Z')
    Zs = Z.shape
    n = Zs[0] + 1
    if isinstance(p, (int, float)):
        p = int(p)
    else:
        raise TypeError('The second argument must be a number')

    if truncate_mode not in ('lastp', 'mtica', 'level', 'none', None):
        # 'mtica' is kept working for backwards compat.
        raise ValueError('Invalid truncation mode.')

    if truncate_mode == 'lastp':
        if p > n or p == 0:
            p = n

    if truncate_mode == 'mtica':
        # 'mtica' is an alias
        truncate_mode = 'level'

    if truncate_mode == 'level':
        if p <= 0:
            p = np.inf

    if get_leaves:
        lvs = []
    else:
        lvs = None

    icoord_list = []
    dcoord_list = []
    color_list = []
    current_color = [0]
    currently_below_threshold = [False]
    ivl = []  # list of leaves

    if color_threshold is None or (isinstance(color_threshold, str) and
                                   color_threshold == 'default'):
        color_threshold = xp.max(Z[:, 2]) * 0.7

    R = {'icoord': icoord_list, 'dcoord': dcoord_list, 'ivl': ivl,
         'leaves': lvs, 'color_list': color_list}

    # Empty list will be filled in _dendrogram_calculate_info
    contraction_marks = [] if show_contracted else None

    _dendrogram_calculate_info(
        Z=Z, p=p,
        truncate_mode=truncate_mode,
        color_threshold=color_threshold,
        get_leaves=get_leaves,
        orientation=orientation,
        labels=labels,
        count_sort=count_sort,
        distance_sort=distance_sort,
        show_leaf_counts=show_leaf_counts,
        i=2*n - 2,
        iv=0.0,
        ivl=ivl,
        n=n,
        icoord_list=icoord_list,
        dcoord_list=dcoord_list,
        lvs=lvs,
        current_color=current_color,
        color_list=color_list,
        currently_below_threshold=currently_below_threshold,
        leaf_label_func=leaf_label_func,
        contraction_marks=contraction_marks,
        link_color_func=link_color_func,
        above_threshold_color=above_threshold_color)

    if not no_plot:
        mh = xp.max(Z[:, 2])
        _plot_dendrogram(icoord_list, dcoord_list, ivl, p, n, mh, orientation,
                         no_labels, color_list,
                         leaf_font_size=leaf_font_size,
                         leaf_rotation=leaf_rotation,
                         contraction_marks=contraction_marks,
                         ax=ax,
                         above_threshold_color=above_threshold_color)

    R["leaves_color_list"] = _get_leaves_color_list(R)

    return R


def _get_leaves_color_list(R):
    leaves_color_list = [None] * len(R['leaves'])
    for link_x, link_y, link_color in zip(R['icoord'],
                                          R['dcoord'],
                                          R['color_list']):
        for (xi, yi) in zip(link_x, link_y):
            if yi == 0.0 and (xi % 5 == 0 and xi % 2 == 1):
                # if yi is 0.0 and xi is divisible by 5 and odd,
                # the point is a leaf
                # xi of leaves are      5, 15, 25, 35, ... (see `iv_ticks`)
                # index of leaves are   0,  1,  2,  3, ... as below
                leaf_index = (int(xi) - 5) // 10
                # each leaf has a same color of its link.
                leaves_color_list[leaf_index] = link_color
    return leaves_color_list


def _append_singleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func,
                                i, labels):
    # If the leaf id structure is not None and is a list then the caller
    # to dendrogram has indicated that cluster id's corresponding to the
    # leaf nodes should be recorded.

    if lvs is not None:
        lvs.append(int(i))

    # If leaf node labels are to be displayed...
    if ivl is not None:
        # If a leaf_label_func has been provided, the label comes from the
        # string returned from the leaf_label_func, which is a function
        # passed to dendrogram.
        if leaf_label_func:
            ivl.append(leaf_label_func(int(i)))
        else:
            # Otherwise, if the dendrogram caller has passed a labels list
            # for the leaf nodes, use it.
            if labels is not None:
                ivl.append(labels[int(i - n)])
            else:
                # Otherwise, use the id as the label for the leaf.x
                ivl.append(str(int(i)))


def _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func,
                                   i, labels, show_leaf_counts):
    # If the leaf id structure is not None and is a list then the caller
    # to dendrogram has indicated that cluster id's corresponding to the
    # leaf nodes should be recorded.

    if lvs is not None:
        lvs.append(int(i))
    if ivl is not None:
        if leaf_label_func:
            ivl.append(leaf_label_func(int(i)))
        else:
            if show_leaf_counts:
                ivl.append("(" + str(np.asarray(Z[i - n, 3], dtype=np.int64)) + ")")
            else:
                ivl.append("")


def _append_contraction_marks(Z, iv, i, n, contraction_marks, xp):
    _append_contraction_marks_sub(Z, iv, int_floor(Z[i - n, 0], xp),
                                  n, contraction_marks, xp)
    _append_contraction_marks_sub(Z, iv, int_floor(Z[i - n, 1], xp),
                                  n, contraction_marks, xp)


def _append_contraction_marks_sub(Z, iv, i, n, contraction_marks, xp):
    if i >= n:
        contraction_marks.append((iv, Z[i - n, 2]))
        _append_contraction_marks_sub(Z, iv, int_floor(Z[i - n, 0], xp),
                                      n, contraction_marks, xp)
        _append_contraction_marks_sub(Z, iv, int_floor(Z[i - n, 1], xp),
                                      n, contraction_marks, xp)


def _dendrogram_calculate_info(Z, p, truncate_mode,
                               color_threshold=np.inf, get_leaves=True,
                               orientation='top', labels=None,
                               count_sort=False, distance_sort=False,
                               show_leaf_counts=False, i=-1, iv=0.0,
                               ivl=None, n=0, icoord_list=None, dcoord_list=None,
                               lvs=None, mhr=False,
                               current_color=None, color_list=None,
                               currently_below_threshold=None,
                               leaf_label_func=None, level=0,
                               contraction_marks=None,
                               link_color_func=None,
                               above_threshold_color='C0'):
    """
    Calculate the endpoints of the links as well as the labels for the
    the dendrogram rooted at the node with index i. iv is the independent
    variable value to plot the left-most leaf node below the root node i
    (if orientation='top', this would be the left-most x value where the
    plotting of this root node i and its descendents should begin).

    ivl is a list to store the labels of the leaf nodes. The leaf_label_func
    is called whenever ivl != None, labels == None, and
    leaf_label_func != None. When ivl != None and labels != None, the
    labels list is used only for labeling the leaf nodes. When
    ivl == None, no labels are generated for leaf nodes.

    When get_leaves==True, a list of leaves is built as they are visited
    in the dendrogram.

    Returns a tuple with l being the independent variable coordinate that
    corresponds to the midpoint of cluster to the left of cluster i if
    i is non-singleton, otherwise the independent coordinate of the leaf
    node if i is a leaf node.

    Returns
    -------
    A tuple (left, w, h, md), where:
        * left is the independent variable coordinate of the center of the
          the U of the subtree

        * w is the amount of space used for the subtree (in independent
          variable units)

        * h is the height of the subtree in dependent variable units

        * md is the ``max(Z[*,2]``) for all nodes ``*`` below and including
          the target node.

    """
    xp = array_namespace(Z)
    if n == 0:
        raise ValueError("Invalid singleton cluster count n.")

    if i == -1:
        raise ValueError("Invalid root cluster index i.")

    if truncate_mode == 'lastp':
        # If the node is a leaf node but corresponds to a non-singleton
        # cluster, its label is either the empty string or the number of
        # original observations belonging to cluster i.
        if 2*n - p > i >= n:
            d = Z[i - n, 2]
            _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl,
                                           leaf_label_func, i, labels,
                                           show_leaf_counts)
            if contraction_marks is not None:
                _append_contraction_marks(Z, iv + 5.0, i, n, contraction_marks, xp)
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                        leaf_label_func, i, labels)
            return (iv + 5.0, 10.0, 0.0, 0.0)
    elif truncate_mode == 'level':
        if i > n and level > p:
            d = Z[i - n, 2]
            _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl,
                                           leaf_label_func, i, labels,
                                           show_leaf_counts)
            if contraction_marks is not None:
                _append_contraction_marks(Z, iv + 5.0, i, n, contraction_marks, xp)
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                        leaf_label_func, i, labels)
            return (iv + 5.0, 10.0, 0.0, 0.0)

    # Otherwise, only truncate if we have a leaf node.
    #
    # Only place leaves if they correspond to original observations.
    if i < n:
        _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                    leaf_label_func, i, labels)
        return (iv + 5.0, 10.0, 0.0, 0.0)

    # !!! Otherwise, we don't have a leaf node, so work on plotting a
    # non-leaf node.
    # Actual indices of a and b
    aa = int_floor(Z[i - n, 0], xp)
    ab = int_floor(Z[i - n, 1], xp)
    if aa >= n:
        # The number of singletons below cluster a
        na = Z[aa - n, 3]
        # The distance between a's two direct children.
        da = Z[aa - n, 2]
    else:
        na = 1
        da = 0.0
    if ab >= n:
        nb = Z[ab - n, 3]
        db = Z[ab - n, 2]
    else:
        nb = 1
        db = 0.0

    if count_sort == 'ascending' or count_sort is True:
        # If a has a count greater than b, it and its descendents should
        # be drawn to the right. Otherwise, to the left.
        if na > nb:
            # The cluster index to draw to the left (ua) will be ab
            # and the one to draw to the right (ub) will be aa
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif count_sort == 'descending':
        # If a has a count less than or equal to b, it and its
        # descendents should be drawn to the left. Otherwise, to
        # the right.
        if na > nb:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    elif distance_sort == 'ascending' or distance_sort is True:
        # If a has a distance greater than b, it and its descendents should
        # be drawn to the right. Otherwise, to the left.
        if da > db:
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif distance_sort == 'descending':
        # If a has a distance less than or equal to b, it and its
        # descendents should be drawn to the left. Otherwise, to
        # the right.
        if da > db:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    else:
        ua = aa
        ub = ab

    # Updated iv variable and the amount of space used.
    (uiva, uwa, uah, uamd) = \
        _dendrogram_calculate_info(
            Z=Z, p=p,
            truncate_mode=truncate_mode,
            color_threshold=color_threshold,
            get_leaves=get_leaves,
            orientation=orientation,
            labels=labels,
            count_sort=count_sort,
            distance_sort=distance_sort,
            show_leaf_counts=show_leaf_counts,
            i=ua, iv=iv, ivl=ivl, n=n,
            icoord_list=icoord_list,
            dcoord_list=dcoord_list, lvs=lvs,
            current_color=current_color,
            color_list=color_list,
            currently_below_threshold=currently_below_threshold,
            leaf_label_func=leaf_label_func,
            level=level + 1, contraction_marks=contraction_marks,
            link_color_func=link_color_func,
            above_threshold_color=above_threshold_color)

    h = Z[i - n, 2]
    if h >= color_threshold or color_threshold <= 0:
        c = above_threshold_color

        if currently_below_threshold[0]:
            current_color[0] = (current_color[0] + 1) % len(_link_line_colors)
        currently_below_threshold[0] = False
    else:
        currently_below_threshold[0] = True
        c = _link_line_colors[current_color[0]]

    (uivb, uwb, ubh, ubmd) = \
        _dendrogram_calculate_info(
            Z=Z, p=p,
            truncate_mode=truncate_mode,
            color_threshold=color_threshold,
            get_leaves=get_leaves,
            orientation=orientation,
            labels=labels,
            count_sort=count_sort,
            distance_sort=distance_sort,
            show_leaf_counts=show_leaf_counts,
            i=ub, iv=iv + uwa, ivl=ivl, n=n,
            icoord_list=icoord_list,
            dcoord_list=dcoord_list, lvs=lvs,
            current_color=current_color,
            color_list=color_list,
            currently_below_threshold=currently_below_threshold,
            leaf_label_func=leaf_label_func,
            level=level + 1, contraction_marks=contraction_marks,
            link_color_func=link_color_func,
            above_threshold_color=above_threshold_color)

    max_dist = max(uamd, ubmd, h)

    icoord_list.append([uiva, uiva, uivb, uivb])
    dcoord_list.append([uah, h, h, ubh])
    if link_color_func is not None:
        v = link_color_func(int(i))
        if not isinstance(v, str):
            raise TypeError("link_color_func must return a matplotlib "
                            "color string!")
        color_list.append(v)
    else:
        color_list.append(c)

    return (((uiva + uivb) / 2), uwa + uwb, h, max_dist)


def is_isomorphic(T1, T2):
    """
    Determine if two different cluster assignments are equivalent.

    Parameters
    ----------
    T1 : array_like
        An assignment of singleton cluster ids to flat cluster ids.
    T2 : array_like
        An assignment of singleton cluster ids to flat cluster ids.

    Returns
    -------
    b : bool
        Whether the flat cluster assignments `T1` and `T2` are
        equivalent.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    fcluster : for the creation of flat cluster assignments.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import fcluster, is_isomorphic
    >>> from scipy.cluster.hierarchy import single, complete
    >>> from scipy.spatial.distance import pdist

    Two flat cluster assignments can be isomorphic if they represent the same
    cluster assignment, with different labels.

    For example, we can use the `scipy.cluster.hierarchy.single`: method
    and flatten the output to four clusters:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = single(pdist(X))
    >>> T = fcluster(Z, 1, criterion='distance')
    >>> T
    array([3, 3, 3, 4, 4, 4, 2, 2, 2, 1, 1, 1], dtype=int32)

    We can then do the same using the
    `scipy.cluster.hierarchy.complete`: method:

    >>> Z = complete(pdist(X))
    >>> T_ = fcluster(Z, 1.5, criterion='distance')
    >>> T_
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)

    As we can see, in both cases we obtain four clusters and all the data
    points are distributed in the same way - the only thing that changes
    are the flat cluster labels (3 => 1, 4 =>2, 2 =>3 and 4 =>1), so both
    cluster assignments are isomorphic:

    >>> is_isomorphic(T, T_)
    True

    """
    T1 = np.asarray(T1, order='c')
    T2 = np.asarray(T2, order='c')

    T1S = T1.shape
    T2S = T2.shape

    if len(T1S) != 1:
        raise ValueError('T1 must be one-dimensional.')
    if len(T2S) != 1:
        raise ValueError('T2 must be one-dimensional.')
    if T1S[0] != T2S[0]:
        raise ValueError('T1 and T2 must have the same number of elements.')
    n = T1S[0]
    d1 = {}
    d2 = {}
    for i in range(0, n):
        if T1[i] in d1:
            if T2[i] not in d2:
                return False
            if d1[T1[i]] != T2[i] or d2[T2[i]] != T1[i]:
                return False
        elif T2[i] in d2:
            return False
        else:
            d1[T1[i]] = T2[i]
            d2[T2[i]] = T1[i]
    return True


def maxdists(Z):
    """
    Return the maximum distance between any non-singleton cluster.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a matrix. See
        ``linkage`` for more information.

    Returns
    -------
    maxdists : ndarray
        A ``(n-1)`` sized numpy array of doubles; ``MD[i]`` represents
        the maximum distance between any cluster (including
        singletons) below and including the node with index i. More
        specifically, ``MD[i] = Z[Q(i)-n, 2].max()`` where ``Q(i)`` is the
        set of all node indices below and including node i.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    is_monotonic : for testing for monotonicity of a linkage matrix.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import median, maxdists
    >>> from scipy.spatial.distance import pdist

    Given a linkage matrix ``Z``, `scipy.cluster.hierarchy.maxdists`
    computes for each new cluster generated (i.e., for each row of the linkage
    matrix) what is the maximum distance between any two child clusters.

    Due to the nature of hierarchical clustering, in many cases this is going
    to be just the distance between the two child clusters that were merged
    to form the current one - that is, Z[:,2].

    However, for non-monotonic cluster assignments such as
    `scipy.cluster.hierarchy.median` clustering this is not always the
    case: There may be cluster formations were the distance between the two
    clusters merged is smaller than the distance between their children.

    We can see this in an example:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = median(pdist(X))
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.11803399,  3.        ],
           [ 5.        , 13.        ,  1.11803399,  3.        ],
           [ 8.        , 15.        ,  1.11803399,  3.        ],
           [11.        , 14.        ,  1.11803399,  3.        ],
           [18.        , 19.        ,  3.        ,  6.        ],
           [16.        , 17.        ,  3.5       ,  6.        ],
           [20.        , 21.        ,  3.25      , 12.        ]])
    >>> maxdists(Z)
    array([1.        , 1.        , 1.        , 1.        , 1.11803399,
           1.11803399, 1.11803399, 1.11803399, 3.        , 3.5       ,
           3.5       ])

    Note that while the distance between the two clusters merged when creating the
    last cluster is 3.25, there are two children (clusters 16 and 17) whose distance
    is larger (3.5). Thus, `scipy.cluster.hierarchy.maxdists` returns 3.5 in
    this case.

    """
    xp = array_namespace(Z)
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')

    n = Z.shape[0] + 1
    MD = np.zeros((n - 1,))
    Z = np.asarray(Z)
    _hierarchy.get_max_dist_for_each_cluster(Z, MD, int(n))
    MD = xp.asarray(MD)
    return MD


def maxinconsts(Z, R):
    """
    Return the maximum inconsistency coefficient for each
    non-singleton cluster and its children.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a matrix. See
        `linkage` for more information.
    R : ndarray
        The inconsistency matrix.

    Returns
    -------
    MI : ndarray
        A monotonic ``(n-1)``-sized numpy array of doubles.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    inconsistent : for the creation of a inconsistency matrix.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import median, inconsistent, maxinconsts
    >>> from scipy.spatial.distance import pdist

    Given a data set ``X``, we can apply a clustering method to obtain a
    linkage matrix ``Z``. `scipy.cluster.hierarchy.inconsistent` can
    be also used to obtain the inconsistency matrix ``R`` associated to
    this clustering process:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = median(pdist(X))
    >>> R = inconsistent(Z)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.11803399,  3.        ],
           [ 5.        , 13.        ,  1.11803399,  3.        ],
           [ 8.        , 15.        ,  1.11803399,  3.        ],
           [11.        , 14.        ,  1.11803399,  3.        ],
           [18.        , 19.        ,  3.        ,  6.        ],
           [16.        , 17.        ,  3.5       ,  6.        ],
           [20.        , 21.        ,  3.25      , 12.        ]])
    >>> R
    array([[1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.74535599, 1.08655358, 3.        , 1.15470054],
           [1.91202266, 1.37522872, 3.        , 1.15470054],
           [3.25      , 0.25      , 3.        , 0.        ]])

    Here, `scipy.cluster.hierarchy.maxinconsts` can be used to compute
    the maximum value of the inconsistency statistic (the last column of
    ``R``) for each non-singleton cluster and its children:

    >>> maxinconsts(Z, R)
    array([0.        , 0.        , 0.        , 0.        , 0.70710678,
           0.70710678, 0.70710678, 0.70710678, 1.15470054, 1.15470054,
           1.15470054])

    """
    xp = array_namespace(Z, R)
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    R = _asarray(R, order='C', dtype=xp.float64, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    is_valid_im(R, throw=True, name='R')

    n = Z.shape[0] + 1
    if Z.shape[0] != R.shape[0]:
        raise ValueError("The inconsistency matrix and linkage matrix each "
                         "have a different number of rows.")
    MI = np.zeros((n - 1,))
    Z = np.asarray(Z)
    R = np.asarray(R)
    _hierarchy.get_max_Rfield_for_each_cluster(Z, R, MI, int(n), 3)
    MI = xp.asarray(MI)
    return MI


def maxRstat(Z, R, i):
    """
    Return the maximum statistic for each non-singleton cluster and its
    children.

    Parameters
    ----------
    Z : array_like
        The hierarchical clustering encoded as a matrix. See `linkage` for more
        information.
    R : array_like
        The inconsistency matrix.
    i : int
        The column of `R` to use as the statistic.

    Returns
    -------
    MR : ndarray
        Calculates the maximum statistic for the i'th column of the
        inconsistency matrix `R` for each non-singleton cluster
        node. ``MR[j]`` is the maximum over ``R[Q(j)-n, i]``, where
        ``Q(j)`` the set of all node ids corresponding to nodes below
        and including ``j``.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    inconsistent : for the creation of a inconsistency matrix.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import median, inconsistent, maxRstat
    >>> from scipy.spatial.distance import pdist

    Given a data set ``X``, we can apply a clustering method to obtain a
    linkage matrix ``Z``. `scipy.cluster.hierarchy.inconsistent` can
    be also used to obtain the inconsistency matrix ``R`` associated to
    this clustering process:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = median(pdist(X))
    >>> R = inconsistent(Z)
    >>> R
    array([[1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.74535599, 1.08655358, 3.        , 1.15470054],
           [1.91202266, 1.37522872, 3.        , 1.15470054],
           [3.25      , 0.25      , 3.        , 0.        ]])

    `scipy.cluster.hierarchy.maxRstat` can be used to compute
    the maximum value of each column of ``R``, for each non-singleton
    cluster and its children:

    >>> maxRstat(Z, R, 0)
    array([1.        , 1.        , 1.        , 1.        , 1.05901699,
           1.05901699, 1.05901699, 1.05901699, 1.74535599, 1.91202266,
           3.25      ])
    >>> maxRstat(Z, R, 1)
    array([0.        , 0.        , 0.        , 0.        , 0.08346263,
           0.08346263, 0.08346263, 0.08346263, 1.08655358, 1.37522872,
           1.37522872])
    >>> maxRstat(Z, R, 3)
    array([0.        , 0.        , 0.        , 0.        , 0.70710678,
           0.70710678, 0.70710678, 0.70710678, 1.15470054, 1.15470054,
           1.15470054])

    """
    xp = array_namespace(Z, R)
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    R = _asarray(R, order='C', dtype=xp.float64, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    is_valid_im(R, throw=True, name='R')

    if not isinstance(i, int):
        raise TypeError('The third argument must be an integer.')

    if i < 0 or i > 3:
        raise ValueError('i must be an integer between 0 and 3 inclusive.')

    if Z.shape[0] != R.shape[0]:
        raise ValueError("The inconsistency matrix and linkage matrix each "
                         "have a different number of rows.")

    n = Z.shape[0] + 1
    MR = np.zeros((n - 1,))
    Z = np.asarray(Z)
    R = np.asarray(R)
    _hierarchy.get_max_Rfield_for_each_cluster(Z, R, MR, int(n), i)
    MR = xp.asarray(MR)
    return MR


def leaders(Z, T):
    """
    Return the root nodes in a hierarchical clustering.

    Returns the root nodes in a hierarchical clustering corresponding
    to a cut defined by a flat cluster assignment vector ``T``. See
    the ``fcluster`` function for more information on the format of ``T``.

    For each flat cluster :math:`j` of the :math:`k` flat clusters
    represented in the n-sized flat cluster assignment vector ``T``,
    this function finds the lowest cluster node :math:`i` in the linkage
    tree Z, such that:

      * leaf descendants belong only to flat cluster j
        (i.e., ``T[p]==j`` for all :math:`p` in :math:`S(i)`, where
        :math:`S(i)` is the set of leaf ids of descendant leaf nodes
        with cluster node :math:`i`)

      * there does not exist a leaf that is not a descendant with
        :math:`i` that also belongs to cluster :math:`j`
        (i.e., ``T[q]!=j`` for all :math:`q` not in :math:`S(i)`). If
        this condition is violated, ``T`` is not a valid cluster
        assignment vector, and an exception will be thrown.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a matrix. See
        `linkage` for more information.
    T : ndarray
        The flat cluster assignment vector.

    Returns
    -------
    L : ndarray
        The leader linkage node id's stored as a k-element 1-D array,
        where ``k`` is the number of flat clusters found in ``T``.

        ``L[j]=i`` is the linkage cluster node id that is the
        leader of flat cluster with id M[j]. If ``i < n``, ``i``
        corresponds to an original observation, otherwise it
        corresponds to a non-singleton cluster.
    M : ndarray
        The leader linkage node id's stored as a k-element 1-D array, where
        ``k`` is the number of flat clusters found in ``T``. This allows the
        set of flat cluster ids to be any arbitrary set of ``k`` integers.

        For example: if ``L[3]=2`` and ``M[3]=8``, the flat cluster with
        id 8's leader is linkage node 2.

    See Also
    --------
    fcluster : for the creation of flat cluster assignments.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, fcluster, leaders
    >>> from scipy.spatial.distance import pdist

    Given a linkage matrix ``Z`` - obtained after apply a clustering method
    to a dataset ``X`` - and a flat cluster assignment array ``T``:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])

    >>> T = fcluster(Z, 3, criterion='distance')
    >>> T
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)

    `scipy.cluster.hierarchy.leaders` returns the indices of the nodes
    in the dendrogram that are the leaders of each flat cluster:

    >>> L, M = leaders(Z, T)
    >>> L
    array([16, 17, 18, 19], dtype=int32)

    (remember that indices 0-11 point to the 12 data points in ``X``,
    whereas indices 12-22 point to the 11 rows of ``Z``)

    `scipy.cluster.hierarchy.leaders` also returns the indices of
    the flat clusters in ``T``:

    >>> M
    array([1, 2, 3, 4], dtype=int32)

    """
    xp = array_namespace(Z, T)
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    T = _asarray(T, order='C', xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')

    if T.dtype != xp.int32:
        raise TypeError('T must be a 1-D array of dtype int32.')

    if T.shape[0] != Z.shape[0] + 1:
        raise ValueError('Mismatch: len(T)!=Z.shape[0] + 1.')

    n_clusters = int(xp.unique_values(T).shape[0])
    n_obs = int(Z.shape[0] + 1)
    L = np.zeros(n_clusters, dtype=np.int32)
    M = np.zeros(n_clusters, dtype=np.int32)
    Z = np.asarray(Z)
    T = np.asarray(T, dtype=np.int32)
    s = _hierarchy.leaders(Z, T, L, M, n_clusters, n_obs)
    if s >= 0:
        raise ValueError(('T is not a valid assignment vector. Error found '
                          'when examining linkage node %d (< 2n-1).') % s)
    L, M = xp.asarray(L), xp.asarray(M)
    return (L, M)
<|endoftext|># Warnings


class DegenerateDataWarning(RuntimeWarning):
    """Warns when data is degenerate and results may not be reliable."""
    def __init__(self, msg=None):
        if msg is None:
            msg = ("Degenerate data encountered; results may not be reliable.")
        self.args = (msg,)


class ConstantInputWarning(DegenerateDataWarning):
    """Warns when all values in data are exactly equal."""
    def __init__(self, msg=None):
        if msg is None:
            msg = ("All values in data are exactly equal; "
                   "results may not be reliable.")
        self.args = (msg,)


class NearConstantInputWarning(DegenerateDataWarning):
    """Warns when all values in data are nearly equal."""
    def __init__(self, msg=None):
        if msg is None:
            msg = ("All values in data are nearly equal; "
                   "results may not be reliable.")
        self.args = (msg,)


# Errors


class FitError(RuntimeError):
    """Represents an error condition when fitting a distribution to data."""
    def __init__(self, msg=None):
        if msg is None:
            msg = ("An error occurred when fitting a distribution to data.")
        self.args = (msg,)
<|endoftext|>from functools import cached_property

import numpy as np
from scipy import linalg
from scipy.stats import _multivariate


__all__ = ["Covariance"]


class Covariance:
    """
    Representation of a covariance matrix

    Calculations involving covariance matrices (e.g. data whitening,
    multivariate normal function evaluation) are often performed more
    efficiently using a decomposition of the covariance matrix instead of the
    covariance matrix itself. This class allows the user to construct an
    object representing a covariance matrix using any of several
    decompositions and perform calculations using a common interface.

    .. note::

        The `Covariance` class cannot be instantiated directly. Instead, use
        one of the factory methods (e.g. `Covariance.from_diagonal`).

    Examples
    --------
    The `Covariance` class is used by calling one of its
    factory methods to create a `Covariance` object, then pass that
    representation of the `Covariance` matrix as a shape parameter of a
    multivariate distribution.

    For instance, the multivariate normal distribution can accept an array
    representing a covariance matrix:

    >>> from scipy import stats
    >>> import numpy as np
    >>> d = [1, 2, 3]
    >>> A = np.diag(d)  # a diagonal covariance matrix
    >>> x = [4, -2, 5]  # a point of interest
    >>> dist = stats.multivariate_normal(mean=[0, 0, 0], cov=A)
    >>> dist.pdf(x)
    4.9595685102808205e-08

    but the calculations are performed in a very generic way that does not
    take advantage of any special properties of the covariance matrix. Because
    our covariance matrix is diagonal, we can use ``Covariance.from_diagonal``
    to create an object representing the covariance matrix, and
    `multivariate_normal` can use this to compute the probability density
    function more efficiently.

    >>> cov = stats.Covariance.from_diagonal(d)
    >>> dist = stats.multivariate_normal(mean=[0, 0, 0], cov=cov)
    >>> dist.pdf(x)
    4.9595685102808205e-08

    """
    def __init__(self):
        message = ("The `Covariance` class cannot be instantiated directly. "
                   "Please use one of the factory methods "
                   "(e.g. `Covariance.from_diagonal`).")
        raise NotImplementedError(message)

    @staticmethod
    def from_diagonal(diagonal):
        r"""
        Return a representation of a covariance matrix from its diagonal.

        Parameters
        ----------
        diagonal : array_like
            The diagonal elements of a diagonal matrix.

        Notes
        -----
        Let the diagonal elements of a diagonal covariance matrix :math:`D` be
        stored in the vector :math:`d`.

        When all elements of :math:`d` are strictly positive, whitening of a
        data point :math:`x` is performed by computing
        :math:`x \cdot d^{-1/2}`, where the inverse square root can be taken
        element-wise.
        :math:`\log\det{D}` is calculated as :math:`-2 \sum(\log{d})`,
        where the :math:`\log` operation is performed element-wise.

        This `Covariance` class supports singular covariance matrices. When
        computing ``_log_pdet``, non-positive elements of :math:`d` are
        ignored. Whitening is not well defined when the point to be whitened
        does not lie in the span of the columns of the covariance matrix. The
        convention taken here is to treat the inverse square root of
        non-positive elements of :math:`d` as zeros.

        Examples
        --------
        Prepare a symmetric positive definite covariance matrix ``A`` and a
        data point ``x``.

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> A = np.diag(rng.random(n))
        >>> x = rng.random(size=n)

        Extract the diagonal from ``A`` and create the `Covariance` object.

        >>> d = np.diag(A)
        >>> cov = stats.Covariance.from_diagonal(d)

        Compare the functionality of the `Covariance` object against a
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = np.diag(d**-0.5) @ x
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
        return CovViaDiagonal(diagonal)

    @staticmethod
    def from_precision(precision, covariance=None):
        r"""
        Return a representation of a covariance from its precision matrix.

        Parameters
        ----------
        precision : array_like
            The precision matrix; that is, the inverse of a square, symmetric,
            positive definite covariance matrix.
        covariance : array_like, optional
            The square, symmetric, positive definite covariance matrix. If not
            provided, this may need to be calculated (e.g. to evaluate the
            cumulative distribution function of
            `scipy.stats.multivariate_normal`) by inverting `precision`.

        Notes
        -----
        Let the covariance matrix be :math:`A`, its precision matrix be
        :math:`P = A^{-1}`, and :math:`L` be the lower Cholesky factor such
        that :math:`L L^T = P`.
        Whitening of a data point :math:`x` is performed by computing
        :math:`x^T L`. :math:`\log\det{A}` is calculated as
        :math:`-2tr(\log{L})`, where the :math:`\log` operation is performed
        element-wise.

        This `Covariance` class does not support singular covariance matrices
        because the precision matrix does not exist for a singular covariance
        matrix.

        Examples
        --------
        Prepare a symmetric positive definite precision matrix ``P`` and a
        data point ``x``. (If the precision matrix is not already available,
        consider the other factory methods of the `Covariance` class.)

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> P = rng.random(size=(n, n))
        >>> P = P @ P.T  # a precision matrix must be positive definite
        >>> x = rng.random(size=n)

        Create the `Covariance` object.

        >>> cov = stats.Covariance.from_precision(P)

        Compare the functionality of the `Covariance` object against
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = x @ np.linalg.cholesky(P)
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = -np.linalg.slogdet(P)[-1]
        >>> np.allclose(res, ref)
        True

        """
        return CovViaPrecision(precision, covariance)

    @staticmethod
    def from_cholesky(cholesky):
        r"""
        Representation of a covariance provided via the (lower) Cholesky factor

        Parameters
        ----------
        cholesky : array_like
            The lower triangular Cholesky factor of the covariance matrix.

        Notes
        -----
        Let the covariance matrix be :math:`A` and :math:`L` be the lower
        Cholesky factor such that :math:`L L^T = A`.
        Whitening of a data point :math:`x` is performed by computing
        :math:`L^{-1} x`. :math:`\log\det{A}` is calculated as
        :math:`2tr(\log{L})`, where the :math:`\log` operation is performed
        element-wise.

        This `Covariance` class does not support singular covariance matrices
        because the Cholesky decomposition does not exist for a singular
        covariance matrix.

        Examples
        --------
        Prepare a symmetric positive definite covariance matrix ``A`` and a
        data point ``x``.

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> A = rng.random(size=(n, n))
        >>> A = A @ A.T  # make the covariance symmetric positive definite
        >>> x = rng.random(size=n)

        Perform the Cholesky decomposition of ``A`` and create the
        `Covariance` object.

        >>> L = np.linalg.cholesky(A)
        >>> cov = stats.Covariance.from_cholesky(L)

        Compare the functionality of the `Covariance` object against
        reference implementation.

        >>> from scipy.linalg import solve_triangular
        >>> res = cov.whiten(x)
        >>> ref = solve_triangular(L, x, lower=True)
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
        return CovViaCholesky(cholesky)

    @staticmethod
    def from_eigendecomposition(eigendecomposition):
        r"""
        Representation of a covariance provided via eigendecomposition

        Parameters
        ----------
        eigendecomposition : sequence
            A sequence (nominally a tuple) containing the eigenvalue and
            eigenvector arrays as computed by `scipy.linalg.eigh` or
            `numpy.linalg.eigh`.

        Notes
        -----
        Let the covariance matrix be :math:`A`, let :math:`V` be matrix of
        eigenvectors, and let :math:`W` be the diagonal matrix of eigenvalues
        such that `V W V^T = A`.

        When all of the eigenvalues are strictly positive, whitening of a
        data point :math:`x` is performed by computing
        :math:`x^T (V W^{-1/2})`, where the inverse square root can be taken
        element-wise.
        :math:`\log\det{A}` is calculated as  :math:`tr(\log{W})`,
        where the :math:`\log` operation is performed element-wise.

        This `Covariance` class supports singular covariance matrices. When
        computing ``_log_pdet``, non-positive eigenvalues are ignored.
        Whitening is not well defined when the point to be whitened
        does not lie in the span of the columns of the covariance matrix. The
        convention taken here is to treat the inverse square root of
        non-positive eigenvalues as zeros.

        Examples
        --------
        Prepare a symmetric positive definite covariance matrix ``A`` and a
        data point ``x``.

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> A = rng.random(size=(n, n))
        >>> A = A @ A.T  # make the covariance symmetric positive definite
        >>> x = rng.random(size=n)

        Perform the eigendecomposition of ``A`` and create the `Covariance`
        object.

        >>> w, v = np.linalg.eigh(A)
        >>> cov = stats.Covariance.from_eigendecomposition((w, v))

        Compare the functionality of the `Covariance` object against
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = x @ (v @ np.diag(w**-0.5))
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
        return CovViaEigendecomposition(eigendecomposition)

    def whiten(self, x):
        """
        Perform a whitening transformation on data.

        "Whitening" ("white" as in "white noise", in which each frequency has
        equal magnitude) transforms a set of random variables into a new set of
        random variables with unit-diagonal covariance. When a whitening
        transform is applied to a sample of points distributed according to
        a multivariate normal distribution with zero mean, the covariance of
        the transformed sample is approximately the identity matrix.

        Parameters
        ----------
        x : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.

        Returns
        -------
        x_ : array_like
            The transformed array of points.

        References
        ----------
        .. [1] "Whitening Transformation". Wikipedia.
               https://en.wikipedia.org/wiki/Whitening_transformation
        .. [2] Novak, Lukas, and Miroslav Vorechovsky. "Generalization of
               coloring linear transformation". Transactions of VSB 18.2
               (2018): 31-35. :doi:`10.31490/tces-2018-0013`

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 3
        >>> A = rng.random(size=(n, n))
        >>> cov_array = A @ A.T  # make matrix symmetric positive definite
        >>> precision = np.linalg.inv(cov_array)
        >>> cov_object = stats.Covariance.from_precision(precision)
        >>> x = rng.multivariate_normal(np.zeros(n), cov_array, size=(10000))
        >>> x_ = cov_object.whiten(x)
        >>> np.cov(x_, rowvar=False)  # near-identity covariance
        array([[0.97862122, 0.00893147, 0.02430451],
               [0.00893147, 0.96719062, 0.02201312],
               [0.02430451, 0.02201312, 0.99206881]])

        """
        return self._whiten(np.asarray(x))

    def colorize(self, x):
        """
        Perform a colorizing transformation on data.

        "Colorizing" ("color" as in "colored noise", in which different
        frequencies may have different magnitudes) transforms a set of
        uncorrelated random variables into a new set of random variables with
        the desired covariance. When a coloring transform is applied to a
        sample of points distributed according to a multivariate normal
        distribution with identity covariance and zero mean, the covariance of
        the transformed sample is approximately the covariance matrix used
        in the coloring transform.

        Parameters
        ----------
        x : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.

        Returns
        -------
        x_ : array_like
            The transformed array of points.

        References
        ----------
        .. [1] "Whitening Transformation". Wikipedia.
               https://en.wikipedia.org/wiki/Whitening_transformation
        .. [2] Novak, Lukas, and Miroslav Vorechovsky. "Generalization of
               coloring linear transformation". Transactions of VSB 18.2
               (2018): 31-35. :doi:`10.31490/tces-2018-0013`

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng(1638083107694713882823079058616272161)
        >>> n = 3
        >>> A = rng.random(size=(n, n))
        >>> cov_array = A @ A.T  # make matrix symmetric positive definite
        >>> cholesky = np.linalg.cholesky(cov_array)
        >>> cov_object = stats.Covariance.from_cholesky(cholesky)
        >>> x = rng.multivariate_normal(np.zeros(n), np.eye(n), size=(10000))
        >>> x_ = cov_object.colorize(x)
        >>> cov_data = np.cov(x_, rowvar=False)
        >>> np.allclose(cov_data, cov_array, rtol=3e-2)
        True
        """
        return self._colorize(np.asarray(x))

    @property
    def log_pdet(self):
        """
        Log of the pseudo-determinant of the covariance matrix
        """
        return np.array(self._log_pdet, dtype=float)[()]

    @property
    def rank(self):
        """
        Rank of the covariance matrix
        """
        return np.array(self._rank, dtype=int)[()]

    @property
    def covariance(self):
        """
        Explicit representation of the covariance matrix
        """
        return self._covariance

    @property
    def shape(self):
        """
        Shape of the covariance array
        """
        return self._shape

    def _validate_matrix(self, A, name):
        A = np.atleast_2d(A)
        m, n = A.shape[-2:]
        if m != n or A.ndim != 2 or not (np.issubdtype(A.dtype, np.integer) or
                                         np.issubdtype(A.dtype, np.floating)):
            message = (f"The input `{name}` must be a square, "
                       "two-dimensional array of real numbers.")
            raise ValueError(message)
        return A

    def _validate_vector(self, A, name):
        A = np.atleast_1d(A)
        if A.ndim != 1 or not (np.issubdtype(A.dtype, np.integer) or
                               np.issubdtype(A.dtype, np.floating)):
            message = (f"The input `{name}` must be a one-dimensional array "
                       "of real numbers.")
            raise ValueError(message)
        return A


class CovViaPrecision(Covariance):

    def __init__(self, precision, covariance=None):
        precision = self._validate_matrix(precision, 'precision')
        if covariance is not None:
            covariance = self._validate_matrix(covariance, 'covariance')
            message = "`precision.shape` must equal `covariance.shape`."
            if precision.shape != covariance.shape:
                raise ValueError(message)

        self._chol_P = np.linalg.cholesky(precision)
        self._log_pdet = -2*np.log(np.diag(self._chol_P)).sum(axis=-1)
        self._rank = precision.shape[-1]  # must be full rank if invertible
        self._precision = precision
        self._cov_matrix = covariance
        self._shape = precision.shape
        self._allow_singular = False

    def _whiten(self, x):
        return x @ self._chol_P

    @cached_property
    def _covariance(self):
        n = self._shape[-1]
        return (linalg.cho_solve((self._chol_P, True), np.eye(n))
                if self._cov_matrix is None else self._cov_matrix)

    def _colorize(self, x):
        return linalg.solve_triangular(self._chol_P.T, x.T, lower=False).T


def _dot_diag(x, d):
    # If d were a full diagonal matrix, x @ d would always do what we want.
    # Special treatment is needed for n-dimensional `d` in which each row
    # includes only the diagonal elements of a covariance matrix.
    return x * d if x.ndim < 2 else x * np.expand_dims(d, -2)


class CovViaDiagonal(Covariance):

    def __init__(self, diagonal):
        diagonal = self._validate_vector(diagonal, 'diagonal')

        i_zero = diagonal <= 0
        positive_diagonal = np.array(diagonal, dtype=np.float64)

        positive_diagonal[i_zero] = 1  # ones don't affect determinant
        self._log_pdet = np.sum(np.log(positive_diagonal), axis=-1)

        psuedo_reciprocals = 1 / np.sqrt(positive_diagonal)
        psuedo_reciprocals[i_zero] = 0

        self._sqrt_diagonal = np.sqrt(diagonal)
        self._LP = psuedo_reciprocals
        self._rank = positive_diagonal.shape[-1] - i_zero.sum(axis=-1)
        self._covariance = np.apply_along_axis(np.diag, -1, diagonal)
        self._i_zero = i_zero
        self._shape = self._covariance.shape
        self._allow_singular = True

    def _whiten(self, x):
        return _dot_diag(x, self._LP)

    def _colorize(self, x):
        return _dot_diag(x, self._sqrt_diagonal)

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        return ~np.any(_dot_diag(x, self._i_zero), axis=-1)


class CovViaCholesky(Covariance):

    def __init__(self, cholesky):
        L = self._validate_matrix(cholesky, 'cholesky')

        self._factor = L
        self._log_pdet = 2*np.log(np.diag(self._factor)).sum(axis=-1)
        self._rank = L.shape[-1]  # must be full rank for cholesky
        self._shape = L.shape
        self._allow_singular = False

    @cached_property
    def _covariance(self):
        return self._factor @ self._factor.T

    def _whiten(self, x):
        res = linalg.solve_triangular(self._factor, x.T, lower=True).T
        return res

    def _colorize(self, x):
        return x @ self._factor.T


class CovViaEigendecomposition(Covariance):

    def __init__(self, eigendecomposition):
        eigenvalues, eigenvectors = eigendecomposition
        eigenvalues = self._validate_vector(eigenvalues, 'eigenvalues')
        eigenvectors = self._validate_matrix(eigenvectors, 'eigenvectors')
        message = ("The shapes of `eigenvalues` and `eigenvectors` "
                   "must be compatible.")
        try:
            eigenvalues = np.expand_dims(eigenvalues, -2)
            eigenvectors, eigenvalues = np.broadcast_arrays(eigenvectors,
                                                            eigenvalues)
            eigenvalues = eigenvalues[..., 0, :]
        except ValueError:
            raise ValueError(message)

        i_zero = eigenvalues <= 0
        positive_eigenvalues = np.array(eigenvalues, dtype=np.float64)

        positive_eigenvalues[i_zero] = 1  # ones don't affect determinant
        self._log_pdet = np.sum(np.log(positive_eigenvalues), axis=-1)

        psuedo_reciprocals = 1 / np.sqrt(positive_eigenvalues)
        psuedo_reciprocals[i_zero] = 0

        self._LP = eigenvectors * psuedo_reciprocals
        self._LA = eigenvectors * np.sqrt(eigenvalues)
        self._rank = positive_eigenvalues.shape[-1] - i_zero.sum(axis=-1)
        self._w = eigenvalues
        self._v = eigenvectors
        self._shape = eigenvectors.shape
        self._null_basis = eigenvectors * i_zero
        # This is only used for `_support_mask`, not to decide whether
        # the covariance is singular or not.
        self._eps = _multivariate._eigvalsh_to_eps(eigenvalues) * 10**3
        self._allow_singular = True

    def _whiten(self, x):
        return x @ self._LP

    def _colorize(self, x):
        return x @ self._LA.T

    @cached_property
    def _covariance(self):
        return (self._v * self._w) @ self._v.T

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        residual = np.linalg.norm(x @ self._null_basis, axis=-1)
        in_support = residual < self._eps
        return in_support


class CovViaPSD(Covariance):
    """
    Representation of a covariance provided via an instance of _PSD
    """

    def __init__(self, psd):
        self._LP = psd.U
        self._log_pdet = psd.log_pdet
        self._rank = psd.rank
        self._covariance = psd._M
        self._shape = psd._M.shape
        self._psd = psd
        self._allow_singular = False  # by default

    def _whiten(self, x):
        return x @ self._LP

    def _support_mask(self, x):
        return self._psd._support_mask(x)
<|endoftext|>import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._stats_py import _var
from scipy._lib._util import _transition_to_rng, DecimalNumber, SeedType


if TYPE_CHECKING:
    import numpy.typing as npt


__all__ = [
    'dunnett'
]


@dataclass
class DunnettResult:
    """Result object returned by `scipy.stats.dunnett`.

    Attributes
    ----------
    statistic : float ndarray
        The computed statistic of the test for each comparison. The element
        at index ``i`` is the statistic for the comparison between
        groups ``i`` and the control.
    pvalue : float ndarray
        The computed p-value of the test for each comparison. The element
        at index ``i`` is the p-value for the comparison between
        group ``i`` and the control.
    """
    statistic: np.ndarray
    pvalue: np.ndarray
    _alternative: Literal['two-sided', 'less', 'greater'] = field(repr=False)
    _rho: np.ndarray = field(repr=False)
    _df: int = field(repr=False)
    _std: float = field(repr=False)
    _mean_samples: np.ndarray = field(repr=False)
    _mean_control: np.ndarray = field(repr=False)
    _n_samples: np.ndarray = field(repr=False)
    _n_control: int = field(repr=False)
    _rng: SeedType = field(repr=False)
    _ci: ConfidenceInterval | None = field(default=None, repr=False)
    _ci_cl: DecimalNumber | None = field(default=None, repr=False)

    def __str__(self):
        # Note: `__str__` prints the confidence intervals from the most
        # recent call to `confidence_interval`. If it has not been called,
        # it will be called with the default CL of .95.
        if self._ci is None:
            self.confidence_interval(confidence_level=.95)
        s = (
            "Dunnett's test"
            f" ({self._ci_cl*100:.1f}% Confidence Interval)\n"
            "Comparison               Statistic  p-value  Lower CI  Upper CI\n"
        )
        for i in range(self.pvalue.size):
            s += (f" (Sample {i} - Control) {self.statistic[i]:>10.3f}"
                  f"{self.pvalue[i]:>10.3f}"
                  f"{self._ci.low[i]:>10.3f}"
                  f"{self._ci.high[i]:>10.3f}\n")

        return s

    def _allowance(
        self, confidence_level: DecimalNumber = 0.95, tol: DecimalNumber = 1e-3
    ) -> float:
        """Allowance.

        It is the quantity to add/subtract from the observed difference
        between the means of observed groups and the mean of the control
        group. The result gives confidence limits.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval.
            Default is .95.
        tol : float, optional
            A tolerance for numerical optimization: the allowance will produce
            a confidence within ``10*tol*(1 - confidence_level)`` of the
            specified level, or a warning will be emitted. Tight tolerances
            may be impractical due to noisy evaluation of the objective.
            Default is 1e-3.

        Returns
        -------
        allowance : float
            Allowance around the mean.
        """
        alpha = 1 - confidence_level

        def pvalue_from_stat(statistic):
            statistic = np.array(statistic)
            sf = _pvalue_dunnett(
                rho=self._rho, df=self._df,
                statistic=statistic, alternative=self._alternative,
                rng=self._rng
            )
            return abs(sf - alpha)/alpha

        # Evaluation of `pvalue_from_stat` is noisy due to the use of RQMC to
        # evaluate `multivariate_t.cdf`. `minimize_scalar` is not designed
        # to tolerate a noisy objective function and may fail to find the
        # minimum accurately. We mitigate this possibility with the validation
        # step below, but implementation of a noise-tolerant root finder or
        # minimizer would be a welcome enhancement. See gh-18150.
        res = minimize_scalar(pvalue_from_stat, method='brent', tol=tol)
        critical_value = res.x

        # validation
        # tol*10 because tol=1e-3 means we tolerate a 1% change at most
        if res.success is False or res.fun >= tol*10:
            warnings.warn(
                "Computation of the confidence interval did not converge to "
                "the desired level. The confidence level corresponding with "
                f"the returned interval is approximately {alpha*(1+res.fun)}.",
                stacklevel=3
            )

        # From [1] p. 1101 between (1) and (3)
        allowance = critical_value*self._std*np.sqrt(
            1/self._n_samples + 1/self._n_control
        )
        return abs(allowance)

    def confidence_interval(
        self, confidence_level: DecimalNumber = 0.95
    ) -> ConfidenceInterval:
        """Compute the confidence interval for the specified confidence level.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval.
            Default is .95.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence intervals for each
            comparison. The high and low values are accessible for each
            comparison at index ``i`` for each group ``i``.

        """
        # check to see if the supplied confidence level matches that of the
        # previously computed CI.
        if (self._ci is not None) and (confidence_level == self._ci_cl):
            return self._ci

        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1.")

        allowance = self._allowance(confidence_level=confidence_level)
        diff_means = self._mean_samples - self._mean_control

        low = diff_means-allowance
        high = diff_means+allowance

        if self._alternative == 'greater':
            high = [np.inf] * len(diff_means)
        elif self._alternative == 'less':
            low = [-np.inf] * len(diff_means)

        self._ci_cl = confidence_level
        self._ci = ConfidenceInterval(
            low=low,
            high=high
        )
        return self._ci


@_transition_to_rng('random_state', replace_doc=False)
def dunnett(
    *samples: "npt.ArrayLike",  # noqa: D417
    control: "npt.ArrayLike",
    alternative: Literal['two-sided', 'less', 'greater'] = "two-sided",
    rng: SeedType = None
) -> DunnettResult:
    """Dunnett's test: multiple comparisons of means against a control group.

    This is an implementation of Dunnett's original, single-step test as
    described in [1]_.

    Parameters
    ----------
    sample1, sample2, ... : 1D array_like
        The sample measurements for each experimental group.
    control : 1D array_like
        The sample measurements for the control group.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.

        The null hypothesis is that the means of the distributions underlying
        the samples and control are equal. The following alternative
        hypotheses are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          and control are unequal.
        * 'less': the means of the distributions underlying the samples
          are less than the mean of the distribution underlying the control.
        * 'greater': the means of the distributions underlying the
          samples are greater than the mean of the distribution underlying
          the control.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this keyword was changed from `random_state` to
            `rng`. For an interim period, both keywords will continue to work, although
            only one may be specified at a time. After the interim period, function
            calls using the `random_state` keyword will emit warnings. Following a
            deprecation period, the `random_state` keyword will be removed.

    Returns
    -------
    res : `~scipy.stats._result_classes.DunnettResult`
        An object containing attributes:

        statistic : float ndarray
            The computed statistic of the test for each comparison. The element
            at index ``i`` is the statistic for the comparison between
            groups ``i`` and the control.
        pvalue : float ndarray
            The computed p-value of the test for each comparison. The element
            at index ``i`` is the p-value for the comparison between
            group ``i`` and the control.

        And the following method:

        confidence_interval(confidence_level=0.95) :
            Compute the difference in means of the groups
            with the control +- the allowance.

    See Also
    --------
    tukey_hsd : performs pairwise comparison of means.
    :ref:`hypothesis_dunnett` : Extended example

    Notes
    -----
    Like the independent-sample t-test, Dunnett's test [1]_ is used to make
    inferences about the means of distributions from which samples were drawn.
    However, when multiple t-tests are performed at a fixed significance level,
    the "family-wise error rate" - the probability of incorrectly rejecting the
    null hypothesis in at least one test - will exceed the significance level.
    Dunnett's test is designed to perform multiple comparisons while
    controlling the family-wise error rate.

    Dunnett's test compares the means of multiple experimental groups
    against a single control group. Tukey's Honestly Significant Difference Test
    is another multiple-comparison test that controls the family-wise error
    rate, but `tukey_hsd` performs *all* pairwise comparisons between groups.
    When pairwise comparisons between experimental groups are not needed,
    Dunnett's test is preferable due to its higher power.

    The use of this test relies on several assumptions.

    1. The observations are independent within and among groups.
    2. The observations within each group are normally distributed.
    3. The distributions from which the samples are drawn have the same finite
       variance.

    References
    ----------
    .. [1] Dunnett, Charles W. (1955) "A Multiple Comparison Procedure for
           Comparing Several Treatments with a Control." Journal of the American
           Statistical Association, 50:272, 1096-1121,
           :doi:`10.1080/01621459.1955.10501294`
    .. [2] Thomson, M. L., & Short, M. D. (1969). Mucociliary function in
           health, chronic obstructive airway disease, and asbestosis. Journal
           of applied physiology, 26(5), 535-539.
           :doi:`10.1152/jappl.1969.26.5.535`

    Examples
    --------
    We'll use data from [2]_, Table 1. The null hypothesis is that the means of
    the distributions underlying the samples and control are equal.

    First, we test that the means of the distributions underlying the samples
    and control are unequal (``alternative='two-sided'``, the default).

    >>> import numpy as np
    >>> from scipy.stats import dunnett
    >>> samples = [[3.8, 2.7, 4.0, 2.4], [2.8, 3.4, 3.7, 2.2, 2.0]]
    >>> control = [2.9, 3.0, 2.5, 2.6, 3.2]
    >>> res = dunnett(*samples, control=control)
    >>> res.statistic
    array([ 0.90874545, -0.05007117])
    >>> res.pvalue
    array([0.58325114, 0.99819341])

    Now, we test that the means of the distributions underlying the samples are
    greater than the mean of the distribution underlying the control.

    >>> res = dunnett(*samples, control=control, alternative='greater')
    >>> res.statistic
    array([ 0.90874545, -0.05007117])
    >>> res.pvalue
    array([0.30230596, 0.69115597])

    For a more detailed example, see :ref:`hypothesis_dunnett`.
    """
    samples_, control_, rng = _iv_dunnett(
        samples=samples, control=control,
        alternative=alternative, rng=rng
    )

    rho, df, n_group, n_samples, n_control = _params_dunnett(
        samples=samples_, control=control_
    )

    statistic, std, mean_control, mean_samples = _statistic_dunnett(
        samples_, control_, df, n_samples, n_control
    )

    pvalue = _pvalue_dunnett(
        rho=rho, df=df, statistic=statistic, alternative=alternative, rng=rng
    )

    return DunnettResult(
        statistic=statistic, pvalue=pvalue,
        _alternative=alternative,
        _rho=rho, _df=df, _std=std,
        _mean_samples=mean_samples,
        _mean_control=mean_control,
        _n_samples=n_samples,
        _n_control=n_control,
        _rng=rng
    )


def _iv_dunnett(
    samples: Sequence["npt.ArrayLike"],
    control: "npt.ArrayLike",
    alternative: Literal['two-sided', 'less', 'greater'],
    rng: SeedType
) -> tuple[list[np.ndarray], np.ndarray, SeedType]:
    """Input validation for Dunnett's test."""
    rng = check_random_state(rng)

    if alternative not in {'two-sided', 'less', 'greater'}:
        raise ValueError(
            "alternative must be 'less', 'greater' or 'two-sided'"
        )

    ndim_msg = "Control and samples groups must be 1D arrays"
    n_obs_msg = "Control and samples groups must have at least 1 observation"

    control = np.asarray(control)
    samples_ = [np.asarray(sample) for sample in samples]

    # samples checks
    samples_control: list[np.ndarray] = samples_ + [control]
    for sample in samples_control:
        if sample.ndim > 1:
            raise ValueError(ndim_msg)

        if sample.size < 1:
            raise ValueError(n_obs_msg)

    return samples_, control, rng


def _params_dunnett(
    samples: list[np.ndarray], control: np.ndarray
) -> tuple[np.ndarray, int, int, np.ndarray, int]:
    """Specific parameters for Dunnett's test.

    Degree of freedom is the number of observations minus the number of groups
    including the control.
    """
    n_samples = np.array([sample.size for sample in samples])

    # From [1] p. 1100 d.f. = (sum N)-(p+1)
    n_sample = n_samples.sum()
    n_control = control.size
    n = n_sample + n_control
    n_groups = len(samples)
    df = n - n_groups - 1

    # From [1] p. 1103 rho_ij = 1/sqrt((N0/Ni+1)(N0/Nj+1))
    rho = n_control/n_samples + 1
    rho = 1/np.sqrt(rho[:, None] * rho[None, :])
    np.fill_diagonal(rho, 1)

    return rho, df, n_groups, n_samples, n_control


def _statistic_dunnett(
    samples: list[np.ndarray], control: np.ndarray, df: int,
    n_samples: np.ndarray, n_control: int
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Statistic of Dunnett's test.

    Computation based on the original single-step test from [1].
    """
    mean_control = np.mean(control)
    mean_samples = np.array([np.mean(sample) for sample in samples])
    all_samples = [control] + samples
    all_means = np.concatenate([[mean_control], mean_samples])

    # Variance estimate s^2 from [1] Eq. 1
    s2 = np.sum([_var(sample, mean=mean)*sample.size
                 for sample, mean in zip(all_samples, all_means)]) / df
    std = np.sqrt(s2)

    # z score inferred from [1] unlabeled equation after Eq. 1
    z = (mean_samples - mean_control) / np.sqrt(1/n_samples + 1/n_control)

    return z / std, std, mean_control, mean_samples


def _pvalue_dunnett(
    rho: np.ndarray, df: int, statistic: np.ndarray,
    alternative: Literal['two-sided', 'less', 'greater'],
    rng: SeedType = None
) -> np.ndarray:
    """pvalue from the multivariate t-distribution.

    Critical values come from the multivariate student-t distribution.
    """
    statistic = statistic.reshape(-1, 1)

    mvt = stats.multivariate_t(shape=rho, df=df, seed=rng)
    if alternative == "two-sided":
        statistic = abs(statistic)
        pvalue = 1 - mvt.cdf(statistic, lower_limit=-statistic)
    elif alternative == "greater":
        pvalue = 1 - mvt.cdf(statistic, lower_limit=-np.inf)
    else:
        pvalue = 1 - mvt.cdf(np.inf, lower_limit=statistic)

    return np.atleast_1d(pvalue)
<|endoftext|>import warnings
import numpy as np

from scipy._lib._util import check_random_state, MapWrapper, rng_integers, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements

from ._stats import _local_correlations  # type: ignore[import-not-found]
from . import distributions

__all__ = ['multiscale_graphcorr']

# FROM MGCPY: https://github.com/neurodata/mgcpy


class _ParallelP:
    """Helper function to calculate parallel p-value."""

    def __init__(self, x, y, random_states):
        self.x = x
        self.y = y
        self.random_states = random_states

    def __call__(self, index):
        order = self.random_states[index].permutation(self.y.shape[0])
        permy = self.y[order][:, order]

        # calculate permuted stats, store in null distribution
        perm_stat = _mgc_stat(self.x, permy)[0]

        return perm_stat


def _perm_test(x, y, stat, reps=1000, workers=-1, random_state=None):
    r"""Helper function that calculates the p-value. See below for uses.

    Parameters
    ----------
    x, y : ndarray
        `x` and `y` have shapes ``(n, p)`` and ``(n, q)``.
    stat : float
        The sample test statistic.
    reps : int, optional
        The number of replications used to estimate the null when using the
        permutation test. The default is 1000 replications.
    workers : int or map-like callable, optional
        If `workers` is an int the population is subdivided into `workers`
        sections and evaluated in parallel (uses
        `multiprocessing.Pool <multiprocessing>`). Supply `-1` to use all cores
        available to the Process. Alternatively supply a map-like callable,
        such as `multiprocessing.Pool.map` for evaluating the population in
        parallel. This evaluation is carried out as `workers(func, iterable)`.
        Requires that `func` be pickleable.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    pvalue : float
        The sample test p-value.
    null_dist : list
        The approximated null distribution.

    """
    # generate seeds for each rep (change to new parallel random number
    # capabilities in numpy >= 1.17+)
    random_state = check_random_state(random_state)
    random_states = [np.random.RandomState(rng_integers(random_state, 1 << 32,
                     size=4, dtype=np.uint32)) for _ in range(reps)]

    # parallelizes with specified workers over number of reps and set seeds
    parallelp = _ParallelP(x=x, y=y, random_states=random_states)
    with MapWrapper(workers) as mapwrapper:
        null_dist = np.array(list(mapwrapper(parallelp, range(reps))))

    # calculate p-value and significant permutation map through list
    pvalue = (1 + (null_dist >= stat).sum()) / (1 + reps)

    return pvalue, null_dist


def _euclidean_dist(x):
    return cdist(x, x)


MGCResult = _make_tuple_bunch('MGCResult',
                              ['statistic', 'pvalue', 'mgc_dict'], [])


def multiscale_graphcorr(x, y, compute_distance=_euclidean_dist, reps=1000,
                         workers=1, is_twosamp=False, random_state=None):
    r"""Computes the Multiscale Graph Correlation (MGC) test statistic.

    Specifically, for each point, MGC finds the :math:`k`-nearest neighbors for
    one property (e.g. cloud density), and the :math:`l`-nearest neighbors for
    the other property (e.g. grass wetness) [1]_. This pair :math:`(k, l)` is
    called the "scale". A priori, however, it is not know which scales will be
    most informative. So, MGC computes all distance pairs, and then efficiently
    computes the distance correlations for all scales. The local correlations
    illustrate which scales are relatively informative about the relationship.
    The key, therefore, to successfully discover and decipher relationships
    between disparate data modalities is to adaptively determine which scales
    are the most informative, and the geometric implication for the most
    informative scales. Doing so not only provides an estimate of whether the
    modalities are related, but also provides insight into how the
    determination was made. This is especially important in high-dimensional
    data, where simple visualizations do not reveal relationships to the
    unaided human eye. Characterizations of this implementation in particular
    have been derived from and benchmarked within in [2]_.

    Parameters
    ----------
    x, y : ndarray
        If ``x`` and ``y`` have shapes ``(n, p)`` and ``(n, q)`` where `n` is
        the number of samples and `p` and `q` are the number of dimensions,
        then the MGC independence test will be run.  Alternatively, ``x`` and
        ``y`` can have shapes ``(n, n)`` if they are distance or similarity
        matrices, and ``compute_distance`` must be sent to ``None``. If ``x``
        and ``y`` have shapes ``(n, p)`` and ``(m, p)``, an unpaired
        two-sample MGC test will be run.
    compute_distance : callable, optional
        A function that computes the distance or similarity among the samples
        within each data matrix. Set to ``None`` if ``x`` and ``y`` are
        already distance matrices. The default uses the euclidean norm metric.
        If you are calling a custom function, either create the distance
        matrix before-hand or create a function of the form
        ``compute_distance(x)`` where `x` is the data matrix for which
        pairwise distances are calculated.
    reps : int, optional
        The number of replications used to estimate the null when using the
        permutation test. The default is ``1000``.
    workers : int or map-like callable, optional
        If ``workers`` is an int the population is subdivided into ``workers``
        sections and evaluated in parallel (uses ``multiprocessing.Pool
        <multiprocessing>``). Supply ``-1`` to use all cores available to the
        Process. Alternatively supply a map-like callable, such as
        ``multiprocessing.Pool.map`` for evaluating the p-value in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
        Requires that `func` be pickleable. The default is ``1``.
    is_twosamp : bool, optional
        If `True`, a two sample test will be run. If ``x`` and ``y`` have
        shapes ``(n, p)`` and ``(m, p)``, this optional will be overridden and
        set to ``True``. Set to ``True`` if ``x`` and ``y`` both have shapes
        ``(n, p)`` and a two sample test is desired. The default is ``False``.
        Note that this will not run if inputs are distance matrices.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    res : MGCResult
        An object containing attributes:

        statistic : float
            The sample MGC test statistic within ``[-1, 1]``.
        pvalue : float
            The p-value obtained via permutation.
        mgc_dict : dict
            Contains additional useful results:

                - mgc_map : ndarray
                    A 2D representation of the latent geometry of the
                    relationship.
                - opt_scale : (int, int)
                    The estimated optimal scale as a ``(x, y)`` pair.
                - null_dist : list
                    The null distribution derived from the permuted matrices.

    See Also
    --------
    pearsonr : Pearson correlation coefficient and p-value for testing
               non-correlation.
    kendalltau : Calculates Kendall's tau.
    spearmanr : Calculates a Spearman rank-order correlation coefficient.

    Notes
    -----
    A description of the process of MGC and applications on neuroscience data
    can be found in [1]_. It is performed using the following steps:

    #. Two distance matrices :math:`D^X` and :math:`D^Y` are computed and
       modified to be mean zero columnwise. This results in two
       :math:`n \times n` distance matrices :math:`A` and :math:`B` (the
       centering and unbiased modification) [3]_.

    #. For all values :math:`k` and :math:`l` from :math:`1, ..., n`,

       * The :math:`k`-nearest neighbor and :math:`l`-nearest neighbor graphs
         are calculated for each property. Here, :math:`G_k (i, j)` indicates
         the :math:`k`-smallest values of the :math:`i`-th row of :math:`A`
         and :math:`H_l (i, j)` indicates the :math:`l` smallested values of
         the :math:`i`-th row of :math:`B`

       * Let :math:`\circ` denotes the entry-wise matrix product, then local
         correlations are summed and normalized using the following statistic:

    .. math::

        c^{kl} = \frac{\sum_{ij} A G_k B H_l}
                      {\sqrt{\sum_{ij} A^2 G_k \times \sum_{ij} B^2 H_l}}

    #. The MGC test statistic is the smoothed optimal local correlation of
       :math:`\{ c^{kl} \}`. Denote the smoothing operation as :math:`R(\cdot)`
       (which essentially set all isolated large correlations) as 0 and
       connected large correlations the same as before, see [3]_.) MGC is,

    .. math::

        MGC_n (x, y) = \max_{(k, l)} R \left(c^{kl} \left( x_n, y_n \right)
                                                    \right)

    The test statistic returns a value between :math:`(-1, 1)` since it is
    normalized.

    The p-value returned is calculated using a permutation test. This process
    is completed by first randomly permuting :math:`y` to estimate the null
    distribution and then calculating the probability of observing a test
    statistic, under the null, at least as extreme as the observed test
    statistic.

    MGC requires at least 5 samples to run with reliable results. It can also
    handle high-dimensional data sets.
    In addition, by manipulating the input data matrices, the two-sample
    testing problem can be reduced to the independence testing problem [4]_.
    Given sample data :math:`U` and :math:`V` of sizes :math:`p \times n`
    :math:`p \times m`, data matrix :math:`X` and :math:`Y` can be created as
    follows:

    .. math::

        X = [U | V] \in \mathcal{R}^{p \times (n + m)}
        Y = [0_{1 \times n} | 1_{1 \times m}] \in \mathcal{R}^{(n + m)}

    Then, the MGC statistic can be calculated as normal. This methodology can
    be extended to similar tests such as distance correlation [4]_.

    .. versionadded:: 1.4.0

    References
    ----------
    .. [1] Vogelstein, J. T., Bridgeford, E. W., Wang, Q., Priebe, C. E.,
           Maggioni, M., & Shen, C. (2019). Discovering and deciphering
           relationships across disparate data modalities. ELife.
    .. [2] Panda, S., Palaniappan, S., Xiong, J., Swaminathan, A.,
           Ramachandran, S., Bridgeford, E. W., ... Vogelstein, J. T. (2019).
           mgcpy: A Comprehensive High Dimensional Independence Testing Python
           Package. :arXiv:`1907.02088`
    .. [3] Shen, C., Priebe, C.E., & Vogelstein, J. T. (2019). From distance
           correlation to multiscale graph correlation. Journal of the American
           Statistical Association.
    .. [4] Shen, C. & Vogelstein, J. T. (2018). The Exact Equivalence of
           Distance and Kernel Methods for Hypothesis Testing.
           :arXiv:`1806.05514`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import multiscale_graphcorr
    >>> x = np.arange(100)
    >>> y = x
    >>> res = multiscale_graphcorr(x, y)
    >>> res.statistic, res.pvalue
    (1.0, 0.001)

    To run an unpaired two-sample test,

    >>> x = np.arange(100)
    >>> y = np.arange(79)
    >>> res = multiscale_graphcorr(x, y)
    >>> res.statistic, res.pvalue  # doctest: +SKIP
    (0.033258146255703246, 0.023)

    or, if shape of the inputs are the same,

    >>> x = np.arange(100)
    >>> y = x
    >>> res = multiscale_graphcorr(x, y, is_twosamp=True)
    >>> res.statistic, res.pvalue  # doctest: +SKIP
    (-0.008021809890200488, 1.0)

    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x and y must be ndarrays")

    # convert arrays of type (n,) to (n, 1)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.ndim != 2:
        raise ValueError(f"Expected a 2-D array `x`, found shape {x.shape}")
    if y.ndim == 1:
        y = y[:, np.newaxis]
    elif y.ndim != 2:
        raise ValueError(f"Expected a 2-D array `y`, found shape {y.shape}")

    nx, px = x.shape
    ny, py = y.shape

    # check for NaNs
    _contains_nan(x, nan_policy='raise')
    _contains_nan(y, nan_policy='raise')

    # check for positive or negative infinity and raise error
    if np.sum(np.isinf(x)) > 0 or np.sum(np.isinf(y)) > 0:
        raise ValueError("Inputs contain infinities")

    if nx != ny:
        if px == py:
            # reshape x and y for two sample testing
            is_twosamp = True
        else:
            raise ValueError("Shape mismatch, x and y must have shape [n, p] "
                             "and [n, q] or have shape [n, p] and [m, p].")

    if nx < 5 or ny < 5:
        raise ValueError("MGC requires at least 5 samples to give reasonable "
                         "results.")

    # convert x and y to float
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # check if compute_distance_matrix if a callable()
    if not callable(compute_distance) and compute_distance is not None:
        raise ValueError("Compute_distance must be a function.")

    # check if number of reps exists, integer, or > 0 (if under 1000 raises
    # warning)
    if not isinstance(reps, int) or reps < 0:
        raise ValueError("Number of reps must be an integer greater than 0.")
    elif reps < 1000:
        msg = ("The number of replications is low (under 1000), and p-value "
               "calculations may be unreliable. Use the p-value result, with "
               "caution!")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if is_twosamp:
        if compute_distance is None:
            raise ValueError("Cannot run if inputs are distance matrices")
        x, y = _two_sample_transform(x, y)

    if compute_distance is not None:
        # compute distance matrices for x and y
        x = compute_distance(x)
        y = compute_distance(y)

    # calculate MGC stat
    stat, stat_dict = _mgc_stat(x, y)
    stat_mgc_map = stat_dict["stat_mgc_map"]
    opt_scale = stat_dict["opt_scale"]

    # calculate permutation MGC p-value
    pvalue, null_dist = _perm_test(x, y, stat, reps=reps, workers=workers,
                                   random_state=random_state)

    # save all stats (other than stat/p-value) in dictionary
    mgc_dict = {"mgc_map": stat_mgc_map,
                "opt_scale": opt_scale,
                "null_dist": null_dist}

    # create result object with alias for backward compatibility
    res = MGCResult(stat, pvalue, mgc_dict)
    res.stat = stat
    return res


def _mgc_stat(distx, disty):
    r"""Helper function that calculates the MGC stat. See above for use.

    Parameters
    ----------
    distx, disty : ndarray
        `distx` and `disty` have shapes ``(n, p)`` and ``(n, q)`` or
        ``(n, n)`` and ``(n, n)``
        if distance matrices.

    Returns
    -------
    stat : float
        The sample MGC test statistic within ``[-1, 1]``.
    stat_dict : dict
        Contains additional useful additional returns containing the following
        keys:

            - stat_mgc_map : ndarray
                MGC-map of the statistics.
            - opt_scale : (float, float)
                The estimated optimal scale as a ``(x, y)`` pair.

    """
    # calculate MGC map and optimal scale
    stat_mgc_map = _local_correlations(distx, disty, global_corr='mgc')

    n, m = stat_mgc_map.shape
    if m == 1 or n == 1:
        # the global scale at is the statistic calculated at maximal nearest
        # neighbors. There is not enough local scale to search over, so
        # default to global scale
        stat = stat_mgc_map[m - 1][n - 1]
        opt_scale = m * n
    else:
        samp_size = len(distx) - 1

        # threshold to find connected region of significant local correlations
        sig_connect = _threshold_mgc_map(stat_mgc_map, samp_size)

        # maximum within the significant region
        stat, opt_scale = _smooth_mgc_map(sig_connect, stat_mgc_map)

    stat_dict = {"stat_mgc_map": stat_mgc_map,
                 "opt_scale": opt_scale}

    return stat, stat_dict


def _threshold_mgc_map(stat_mgc_map, samp_size):
    r"""
    Finds a connected region of significance in the MGC-map by thresholding.

    Parameters
    ----------
    stat_mgc_map : ndarray
        All local correlations within ``[-1,1]``.
    samp_size : int
        The sample size of original data.

    Returns
    -------
    sig_connect : ndarray
        A binary matrix with 1's indicating the significant region.

    """
    m, n = stat_mgc_map.shape

    # 0.02 is simply an empirical threshold, this can be set to 0.01 or 0.05
    # with varying levels of performance. Threshold is based on a beta
    # approximation.
    per_sig = 1 - (0.02 / samp_size)  # Percentile to consider as significant
    threshold = samp_size * (samp_size - 3)/4 - 1/2  # Beta approximation
    threshold = distributions.beta.ppf(per_sig, threshold, threshold) * 2 - 1

    # the global scale at is the statistic calculated at maximal nearest
    # neighbors. Threshold is the maximum on the global and local scales
    threshold = max(threshold, stat_mgc_map[m - 1][n - 1])

    # find the largest connected component of significant correlations
    sig_connect = stat_mgc_map > threshold
    if np.sum(sig_connect) > 0:
        sig_connect, _ = _measurements.label(sig_connect)
        _, label_counts = np.unique(sig_connect, return_counts=True)

        # skip the first element in label_counts, as it is count(zeros)
        max_label = np.argmax(label_counts[1:]) + 1
        sig_connect = sig_connect == max_label
    else:
        sig_connect = np.array([[False]])

    return sig_connect


def _smooth_mgc_map(sig_connect, stat_mgc_map):
    """Finds the smoothed maximal within the significant region R.

    If area of R is too small it returns the last local correlation. Otherwise,
    returns the maximum within significant_connected_region.

    Parameters
    ----------
    sig_connect : ndarray
        A binary matrix with 1's indicating the significant region.
    stat_mgc_map : ndarray
        All local correlations within ``[-1, 1]``.

    Returns
    -------
    stat : float
        The sample MGC statistic within ``[-1, 1]``.
    opt_scale: (float, float)
        The estimated optimal scale as an ``(x, y)`` pair.

    """
    m, n = stat_mgc_map.shape

    # the global scale at is the statistic calculated at maximal nearest
    # neighbors. By default, statistic and optimal scale are global.
    stat = stat_mgc_map[m - 1][n - 1]
    opt_scale = [m, n]

    if np.linalg.norm(sig_connect) != 0:
        # proceed only when the connected region's area is sufficiently large
        # 0.02 is simply an empirical threshold, this can be set to 0.01 or 0.05
        # with varying levels of performance
        if np.sum(sig_connect) >= np.ceil(0.02 * max(m, n)) * min(m, n):
            max_corr = max(stat_mgc_map[sig_connect])

            # find all scales within significant_connected_region that maximize
            # the local correlation
            max_corr_index = np.where((stat_mgc_map >= max_corr) & sig_connect)

            if max_corr >= stat:
                stat = max_corr

                k, l = max_corr_index
                one_d_indices = k * n + l  # 2D to 1D indexing
                k = np.max(one_d_indices) // n
                l = np.max(one_d_indices) % n
                opt_scale = [k+1, l+1]  # adding 1s to match R indexing

    return stat, opt_scale


def _two_sample_transform(u, v):
    """Helper function that concatenates x and y for two sample MGC stat.

    See above for use.

    Parameters
    ----------
    u, v : ndarray
        `u` and `v` have shapes ``(n, p)`` and ``(m, p)``.

    Returns
    -------
    x : ndarray
        Concatenate `u` and `v` along the ``axis = 0``. `x` thus has shape
        ``(2n, p)``.
    y : ndarray
        Label matrix for `x` where 0 refers to samples that comes from `u` and
        1 refers to samples that come from `v`. `y` thus has shape ``(2n, 1)``.

    """
    nx = u.shape[0]
    ny = v.shape[0]
    x = np.concatenate([u, v], axis=0)
    y = np.concatenate([np.zeros(nx), np.ones(ny)], axis=0).reshape(-1, 1)
    return x, y
<|endoftext|>"""
Contingency table functions (:mod:`scipy.stats.contingency`)
============================================================

Functions for creating and analyzing contingency tables.

.. currentmodule:: scipy.stats.contingency

.. autosummary::
   :toctree: generated/

   chi2_contingency
   relative_risk
   odds_ratio
   crosstab
   association

   expected_freq
   margins

"""


from functools import reduce
import math
import numpy as np
from ._stats_py import power_divergence, _untabulate
from ._relative_risk import relative_risk
from ._crosstab import crosstab
from ._odds_ratio import odds_ratio
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats


__all__ = ['margins', 'expected_freq', 'chi2_contingency', 'crosstab',
           'association', 'relative_risk', 'odds_ratio']


def margins(a):
    """Return a list of the marginal sums of the array `a`.

    Parameters
    ----------
    a : ndarray
        The array for which to compute the marginal sums.

    Returns
    -------
    margsums : list of ndarrays
        A list of length `a.ndim`.  `margsums[k]` is the result
        of summing `a` over all axes except `k`; it has the same
        number of dimensions as `a`, but the length of each axis
        except axis `k` will be 1.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.contingency import margins

    >>> a = np.arange(12).reshape(2, 6)
    >>> a
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11]])
    >>> m0, m1 = margins(a)
    >>> m0
    array([[15],
           [51]])
    >>> m1
    array([[ 6,  8, 10, 12, 14, 16]])

    >>> b = np.arange(24).reshape(2,3,4)
    >>> m0, m1, m2 = margins(b)
    >>> m0
    array([[[ 66]],
           [[210]]])
    >>> m1
    array([[[ 60],
            [ 92],
            [124]]])
    >>> m2
    array([[[60, 66, 72, 78]]])
    """
    margsums = []
    ranged = list(range(a.ndim))
    for k in ranged:
        marg = np.apply_over_axes(np.sum, a, [j for j in ranged if j != k])
        margsums.append(marg)
    return margsums


def expected_freq(observed):
    """
    Compute the expected frequencies from a contingency table.

    Given an n-dimensional contingency table of observed frequencies,
    compute the expected frequencies for the table based on the marginal
    sums under the assumption that the groups associated with each
    dimension are independent.

    Parameters
    ----------
    observed : array_like
        The table of observed frequencies.  (While this function can handle
        a 1-D array, that case is trivial.  Generally `observed` is at
        least 2-D.)

    Returns
    -------
    expected : ndarray of float64
        The expected frequencies, based on the marginal sums of the table.
        Same shape as `observed`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.contingency import expected_freq
    >>> observed = np.array([[10, 10, 20],[20, 20, 20]])
    >>> expected_freq(observed)
    array([[ 12.,  12.,  16.],
           [ 18.,  18.,  24.]])

    """
    # Typically `observed` is an integer array. If `observed` has a large
    # number of dimensions or holds large values, some of the following
    # computations may overflow, so we first switch to floating point.
    observed = np.asarray(observed, dtype=np.float64)

    # Create a list of the marginal sums.
    margsums = margins(observed)

    # Create the array of expected frequencies.  The shapes of the
    # marginal sums returned by apply_over_axes() are just what we
    # need for broadcasting in the following product.
    d = observed.ndim
    expected = reduce(np.multiply, margsums) / observed.sum() ** (d - 1)
    return expected


Chi2ContingencyResult = _make_tuple_bunch(
    'Chi2ContingencyResult',
    ['statistic', 'pvalue', 'dof', 'expected_freq'], []
)


def chi2_contingency(observed, correction=True, lambda_=None, *, method=None):
    """Chi-square test of independence of variables in a contingency table.

    This function computes the chi-square statistic and p-value for the
    hypothesis test of independence of the observed frequencies in the
    contingency table [1]_ `observed`.  The expected frequencies are computed
    based on the marginal sums under the assumption of independence; see
    `scipy.stats.contingency.expected_freq`.  The number of degrees of
    freedom is (expressed using numpy functions and attributes)::

        dof = observed.size - sum(observed.shape) + observed.ndim - 1


    Parameters
    ----------
    observed : array_like
        The contingency table. The table contains the observed frequencies
        (i.e. number of occurrences) in each category.  In the two-dimensional
        case, the table is often described as an "R x C table".
    correction : bool, optional
        If True, *and* the degrees of freedom is 1, apply Yates' correction
        for continuity.  The effect of the correction is to adjust each
        observed value by 0.5 towards the corresponding expected value.
    lambda_ : float or str, optional
        By default, the statistic computed in this test is Pearson's
        chi-squared statistic [2]_.  `lambda_` allows a statistic from the
        Cressie-Read power divergence family [3]_ to be used instead.  See
        `scipy.stats.power_divergence` for details.
    method : ResamplingMethod, optional
        Defines the method used to compute the p-value. Compatible only with
        `correction=False`,  default `lambda_`, and two-way tables.
        If `method` is an instance of `PermutationMethod`/`MonteCarloMethod`,
        the p-value is computed using
        `scipy.stats.permutation_test`/`scipy.stats.monte_carlo_test` with the
        provided configuration options and other appropriate settings.
        Otherwise, the p-value is computed as documented in the notes.
        Note that if `method` is an instance of `MonteCarloMethod`, the ``rvs``
        attribute must be left unspecified; Monte Carlo samples are always drawn
        using the ``rvs`` method of `scipy.stats.random_table`.

        .. versionadded:: 1.15.0


    Returns
    -------
    res : Chi2ContingencyResult
        An object containing attributes:

        statistic : float
            The test statistic.
        pvalue : float
            The p-value of the test.
        dof : int
            The degrees of freedom. NaN if `method` is not ``None``.
        expected_freq : ndarray, same shape as `observed`
            The expected frequencies, based on the marginal sums of the table.

    See Also
    --------
    scipy.stats.contingency.expected_freq
    scipy.stats.fisher_exact
    scipy.stats.chisquare
    scipy.stats.power_divergence
    scipy.stats.barnard_exact
    scipy.stats.boschloo_exact
    :ref:`hypothesis_chi2_contingency` : Extended example

    Notes
    -----
    An often quoted guideline for the validity of this calculation is that
    the test should be used only if the observed and expected frequencies
    in each cell are at least 5.

    This is a test for the independence of different categories of a
    population. The test is only meaningful when the dimension of
    `observed` is two or more.  Applying the test to a one-dimensional
    table will always result in `expected` equal to `observed` and a
    chi-square statistic equal to 0.

    This function does not handle masked arrays, because the calculation
    does not make sense with missing values.

    Like `scipy.stats.chisquare`, this function computes a chi-square
    statistic; the convenience this function provides is to figure out the
    expected frequencies and degrees of freedom from the given contingency
    table. If these were already known, and if the Yates' correction was not
    required, one could use `scipy.stats.chisquare`.  That is, if one calls::

        res = chi2_contingency(obs, correction=False)

    then the following is true::

        (res.statistic, res.pvalue) == stats.chisquare(obs.ravel(),
                                                       f_exp=ex.ravel(),
                                                       ddof=obs.size - 1 - dof)

    The `lambda_` argument was added in version 0.13.0 of scipy.

    References
    ----------
    .. [1] "Contingency table",
           https://en.wikipedia.org/wiki/Contingency_table
    .. [2] "Pearson's chi-squared test",
           https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    .. [3] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
           Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
           pp. 440-464.

    Examples
    --------
    A two-way example (2 x 3):

    >>> import numpy as np
    >>> from scipy.stats import chi2_contingency
    >>> obs = np.array([[10, 10, 20], [20, 20, 20]])
    >>> res = chi2_contingency(obs)
    >>> res.statistic
    2.7777777777777777
    >>> res.pvalue
    0.24935220877729619
    >>> res.dof
    2
    >>> res.expected_freq
    array([[ 12.,  12.,  16.],
           [ 18.,  18.,  24.]])

    Perform the test using the log-likelihood ratio (i.e. the "G-test")
    instead of Pearson's chi-squared statistic.

    >>> res = chi2_contingency(obs, lambda_="log-likelihood")
    >>> res.statistic
    2.7688587616781319
    >>> res.pvalue
    0.25046668010954165

    A four-way example (2 x 2 x 2 x 2):

    >>> obs = np.array(
    ...     [[[[12, 17],
    ...        [11, 16]],
    ...       [[11, 12],
    ...        [15, 16]]],
    ...      [[[23, 15],
    ...        [30, 22]],
    ...       [[14, 17],
    ...        [15, 16]]]])
    >>> res = chi2_contingency(obs)
    >>> res.statistic
    8.7584514426741897
    >>> res.pvalue
    0.64417725029295503

    When the sum of the elements in a two-way table is small, the p-value
    produced by the default asymptotic approximation may be inaccurate.
    Consider passing a `PermutationMethod` or `MonteCarloMethod` as the
    `method` parameter with `correction=False`.

    >>> from scipy.stats import PermutationMethod
    >>> obs = np.asarray([[12, 3],
    ...                   [17, 16]])
    >>> res = chi2_contingency(obs, correction=False)
    >>> ref = chi2_contingency(obs, correction=False, method=PermutationMethod())
    >>> res.pvalue, ref.pvalue
    (0.0614122539870913, 0.1074)  # may vary

    For a more detailed example, see :ref:`hypothesis_chi2_contingency`.

    """
    observed = np.asarray(observed)
    if np.any(observed < 0):
        raise ValueError("All values in `observed` must be nonnegative.")
    if observed.size == 0:
        raise ValueError("No data; `observed` has size 0.")

    expected = expected_freq(observed)
    if np.any(expected == 0):
        # Include one of the positions where expected is zero in
        # the exception message.
        zeropos = list(zip(*np.nonzero(expected == 0)))[0]
        raise ValueError("The internally computed table of expected "
                         f"frequencies has a zero element at {zeropos}.")

    if method is not None:
        return _chi2_resampling_methods(observed, expected, correction, lambda_, method)

    # The degrees of freedom
    dof = expected.size - sum(expected.shape) + expected.ndim - 1

    if dof == 0:
        # Degenerate case; this occurs when `observed` is 1D (or, more
        # generally, when it has only one nontrivial dimension).  In this
        # case, we also have observed == expected, so chi2 is 0.
        chi2 = 0.0
        p = 1.0
    else:
        if dof == 1 and correction:
            # Adjust `observed` according to Yates' correction for continuity.
            # Magnitude of correction no bigger than difference; see gh-13875
            diff = expected - observed
            direction = np.sign(diff)
            magnitude = np.minimum(0.5, np.abs(diff))
            observed = observed + magnitude * direction

        chi2, p = power_divergence(observed, expected,
                                   ddof=observed.size - 1 - dof, axis=None,
                                   lambda_=lambda_)

    return Chi2ContingencyResult(chi2, p, dof, expected)


def _chi2_resampling_methods(observed, expected, correction, lambda_, method):

    if observed.ndim != 2:
        message = 'Use of `method` is only compatible with two-way tables.'
        raise ValueError(message)

    if correction:
        message = f'`{correction=}` is not compatible with `{method=}.`'
        raise ValueError(message)

    if lambda_ is not None:
        message = f'`{lambda_=}` is not compatible with `{method=}.`'
        raise ValueError(message)

    if isinstance(method, stats.PermutationMethod):
        res = _chi2_permutation_method(observed, expected, method)
    elif isinstance(method, stats.MonteCarloMethod):
        res = _chi2_monte_carlo_method(observed, expected, method)
    else:
        message = (f'`{method=}` not recognized; if provided, `method` must be an '
                   'instance of `PermutationMethod` or `MonteCarloMethod`.')
        raise ValueError(message)

    return Chi2ContingencyResult(res.statistic, res.pvalue, np.nan, expected)


def _chi2_permutation_method(observed, expected, method):
    x, y = _untabulate(observed)
    # `permutation_test` with `permutation_type='pairings' permutes the order of `x`,
    # which pairs observations in `x` with different observations in `y`.
    def statistic(x):
        # crosstab the resample and compute the statistic
        table = crosstab(x, y)[1]
        return np.sum((table - expected)**2/expected)

    return stats.permutation_test((x,), statistic, permutation_type='pairings',
                                  alternative='greater', **method._asdict())


def _chi2_monte_carlo_method(observed, expected, method):
    method = method._asdict()

    if method.pop('rvs', None) is not None:
        message = ('If the `method` argument of `chi2_contingency` is an '
                   'instance of `MonteCarloMethod`, its `rvs` attribute '
                   'must be unspecified. Use the `MonteCarloMethod` `rng` argument '
                   'to control the random state.')
        raise ValueError(message)
    rng = np.random.default_rng(method.pop('rng', None))

    # `random_table.rvs` produces random contingency tables with the given marginals
    # under the null hypothesis of independence
    rowsums, colsums = stats.contingency.margins(observed)
    X = stats.random_table(rowsums.ravel(), colsums.ravel(), seed=rng)
    def rvs(size):
        n_resamples = size[0]
        return X.rvs(size=n_resamples).reshape(size)

    expected = expected.ravel()
    def statistic(table, axis):
        return np.sum((table - expected)**2/expected, axis=axis)

    return stats.monte_carlo_test(observed.ravel(), rvs, statistic,
                                  alternative='greater', **method)


def association(observed, method="cramer", correction=False, lambda_=None):
    """Calculates degree of association between two nominal variables.

    The function provides the option for computing one of three measures of
    association between two nominal variables from the data given in a 2d
    contingency table: Tschuprow's T, Pearson's Contingency Coefficient
    and Cramer's V.

    Parameters
    ----------
    observed : array-like
        The array of observed values
    method : {"cramer", "tschuprow", "pearson"} (default = "cramer")
        The association test statistic.
    correction : bool, optional
        Inherited from `scipy.stats.contingency.chi2_contingency()`
    lambda_ : float or str, optional
        Inherited from `scipy.stats.contingency.chi2_contingency()`

    Returns
    -------
    statistic : float
        Value of the test statistic

    Notes
    -----
    Cramer's V, Tschuprow's T and Pearson's Contingency Coefficient, all
    measure the degree to which two nominal or ordinal variables are related,
    or the level of their association. This differs from correlation, although
    many often mistakenly consider them equivalent. Correlation measures in
    what way two variables are related, whereas, association measures how
    related the variables are. As such, association does not subsume
    independent variables, and is rather a test of independence. A value of
    1.0 indicates perfect association, and 0.0 means the variables have no
    association.

    Both the Cramer's V and Tschuprow's T are extensions of the phi
    coefficient.  Moreover, due to the close relationship between the
    Cramer's V and Tschuprow's T the returned values can often be similar
    or even equivalent.  They are likely to diverge more as the array shape
    diverges from a 2x2.

    References
    ----------
    .. [1] "Tschuprow's T",
           https://en.wikipedia.org/wiki/Tschuprow's_T
    .. [2] Tschuprow, A. A. (1939)
           Principles of the Mathematical Theory of Correlation;
           translated by M. Kantorowitsch. W. Hodge & Co.
    .. [3] "Cramer's V", https://en.wikipedia.org/wiki/Cramer's_V
    .. [4] "Nominal Association: Phi and Cramer's V",
           http://www.people.vcu.edu/~pdattalo/702SuppRead/MeasAssoc/NominalAssoc.html
    .. [5] Gingrich, Paul, "Association Between Variables",
           http://uregina.ca/~gingrich/ch11a.pdf

    Examples
    --------
    An example with a 4x2 contingency table:

    >>> import numpy as np
    >>> from scipy.stats.contingency import association
    >>> obs4x2 = np.array([[100, 150], [203, 322], [420, 700], [320, 210]])

    Pearson's contingency coefficient

    >>> association(obs4x2, method="pearson")
    0.18303298140595667

    Cramer's V

    >>> association(obs4x2, method="cramer")
    0.18617813077483678

    Tschuprow's T

    >>> association(obs4x2, method="tschuprow")
    0.14146478765062995
    """
    arr = np.asarray(observed)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("`observed` must be an integer array.")

    if len(arr.shape) != 2:
        raise ValueError("method only accepts 2d arrays")

    chi2_stat = chi2_contingency(arr, correction=correction,
                                 lambda_=lambda_)

    phi2 = chi2_stat.statistic / arr.sum()
    n_rows, n_cols = arr.shape
    if method == "cramer":
        value = phi2 / min(n_cols - 1, n_rows - 1)
    elif method == "tschuprow":
        value = phi2 / math.sqrt((n_rows - 1) * (n_cols - 1))
    elif method == 'pearson':
        value = phi2 / (1 + phi2)
    else:
        raise ValueError("Invalid argument value: 'method' argument must "
                         "be 'cramer', 'tschuprow', or 'pearson'")

    return math.sqrt(value)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'argstoarray',
    'count_tied_groups',
    'describe',
    'f_oneway', 'find_repeats','friedmanchisquare',
    'kendalltau','kendalltau_seasonal','kruskal','kruskalwallis',
    'ks_twosamp', 'ks_2samp', 'kurtosis', 'kurtosistest',
    'ks_1samp', 'kstest',
    'linregress',
    'mannwhitneyu', 'meppf','mode','moment','mquantiles','msign',
    'normaltest',
    'obrientransform',
    'pearsonr','plotting_positions','pointbiserialr',
    'rankdata',
    'scoreatpercentile','sem',
    'sen_seasonal_slopes','skew','skewtest','spearmanr',
    'siegelslopes', 'theilslopes',
    'tmax','tmean','tmin','trim','trimboth',
    'trimtail','trima','trimr','trimmed_mean','trimmed_std',
    'trimmed_stde','trimmed_var','tsem','ttest_1samp','ttest_onesamp',
    'ttest_ind','ttest_rel','tvar',
    'variation',
    'winsorize',
    'brunnermunzel',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="mstats_basic",
                                   private_modules=["_mstats_basic"], all=__all__,
                                   attribute=name, correct_module="mstats")
<|endoftext|>from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
import warnings

import numpy as np
from scipy import special, interpolate, stats
from scipy.stats._censored_data import CensoredData
from scipy.stats._common import ConfidenceInterval

if TYPE_CHECKING:
    import numpy.typing as npt


__all__ = ['ecdf', 'logrank']


@dataclass
class EmpiricalDistributionFunction:
    """An empirical distribution function produced by `scipy.stats.ecdf`

    Attributes
    ----------
    quantiles : ndarray
        The unique values of the sample from which the
        `EmpiricalDistributionFunction` was estimated.
    probabilities : ndarray
        The point estimates of the cumulative distribution function (CDF) or
        its complement, the survival function (SF), corresponding with
        `quantiles`.
    """
    quantiles: np.ndarray
    probabilities: np.ndarray
    # Exclude these from __str__
    _n: np.ndarray = field(repr=False)  # number "at risk"
    _d: np.ndarray = field(repr=False)  # number of "deaths"
    _sf: np.ndarray = field(repr=False)  # survival function for var estimate
    _kind: str = field(repr=False)  # type of function: "cdf" or "sf"

    def __init__(self, q, p, n, d, kind):
        self.probabilities = p
        self.quantiles = q
        self._n = n
        self._d = d
        self._sf = p if kind == 'sf' else 1 - p
        self._kind = kind

        f0 = 1 if kind == 'sf' else 0  # leftmost function value
        f1 = 1 - f0
        # fill_value can't handle edge cases at infinity
        x = np.insert(q, [0, len(q)], [-np.inf, np.inf])
        y = np.insert(p, [0, len(p)], [f0, f1])
        # `or` conditions handle the case of empty x, points
        self._f = interpolate.interp1d(x, y, kind='previous',
                                       assume_sorted=True)

    def evaluate(self, x):
        """Evaluate the empirical CDF/SF function at the input.

        Parameters
        ----------
        x : ndarray
            Argument to the CDF/SF

        Returns
        -------
        y : ndarray
            The CDF/SF evaluated at the input
        """
        return self._f(x)

    def plot(self, ax=None, **matplotlib_kwargs):
        """Plot the empirical distribution function

        Available only if ``matplotlib`` is installed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to draw the plot onto, otherwise uses the current Axes.

        **matplotlib_kwargs : dict, optional
            Keyword arguments passed directly to `matplotlib.axes.Axes.step`.
            Unless overridden, ``where='post'``.

        Returns
        -------
        lines : list of `matplotlib.lines.Line2D`
            Objects representing the plotted data
        """
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError as exc:
            message = "matplotlib must be installed to use method `plot`."
            raise ModuleNotFoundError(message) from exc

        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        kwargs = {'where': 'post'}
        kwargs.update(matplotlib_kwargs)

        delta = np.ptp(self.quantiles)*0.05  # how far past sample edge to plot
        q = self.quantiles
        q = [q[0] - delta] + list(q) + [q[-1] + delta]

        return ax.step(q, self.evaluate(q), **kwargs)

    def confidence_interval(self, confidence_level=0.95, *, method='linear'):
        """Compute a confidence interval around the CDF/SF point estimate

        Parameters
        ----------
        confidence_level : float, default: 0.95
            Confidence level for the computed confidence interval

        method : str, {"linear", "log-log"}
            Method used to compute the confidence interval. Options are
            "linear" for the conventional Greenwood confidence interval
            (default)  and "log-log" for the "exponential Greenwood",
            log-negative-log-transformed confidence interval.

        Returns
        -------
        ci : ``ConfidenceInterval``
            An object with attributes ``low`` and ``high``, instances of
            `~scipy.stats._result_classes.EmpiricalDistributionFunction` that
            represent the lower and upper bounds (respectively) of the
            confidence interval.

        Notes
        -----
        Confidence intervals are computed according to the Greenwood formula
        (``method='linear'``) or the more recent "exponential Greenwood"
        formula (``method='log-log'``) as described in [1]_. The conventional
        Greenwood formula can result in lower confidence limits less than 0
        and upper confidence limits greater than 1; these are clipped to the
        unit interval. NaNs may be produced by either method; these are
        features of the formulas.

        References
        ----------
        .. [1] Sawyer, Stanley. "The Greenwood and Exponential Greenwood
               Confidence Intervals in Survival Analysis."
               https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf

        """
        message = ("Confidence interval bounds do not implement a "
                   "`confidence_interval` method.")
        if self._n is None:
            raise NotImplementedError(message)

        methods = {'linear': self._linear_ci,
                   'log-log': self._loglog_ci}

        message = f"`method` must be one of {set(methods)}."
        if method.lower() not in methods:
            raise ValueError(message)

        message = "`confidence_level` must be a scalar between 0 and 1."
        confidence_level = np.asarray(confidence_level)[()]
        if confidence_level.shape or not (0 <= confidence_level <= 1):
            raise ValueError(message)

        method_fun = methods[method.lower()]
        low, high = method_fun(confidence_level)

        message = ("The confidence interval is undefined at some observations."
                   " This is a feature of the mathematical formula used, not"
                   " an error in its implementation.")
        if np.any(np.isnan(low) | np.isnan(high)):
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)
        low = EmpiricalDistributionFunction(self.quantiles, low, None, None,
                                            self._kind)
        high = EmpiricalDistributionFunction(self.quantiles, high, None, None,
                                             self._kind)
        return ConfidenceInterval(low, high)

    def _linear_ci(self, confidence_level):
        sf, d, n = self._sf, self._d, self._n
        # When n == d, Greenwood's formula divides by zero.
        # When s != 0, this can be ignored: var == inf, and CI is [0, 1]
        # When s == 0, this results in NaNs. Produce an informative warning.
        with np.errstate(divide='ignore', invalid='ignore'):
            var = sf ** 2 * np.cumsum(d / (n * (n - d)))

        se = np.sqrt(var)
        z = special.ndtri(1 / 2 + confidence_level / 2)

        z_se = z * se
        low = self.probabilities - z_se
        high = self.probabilities + z_se

        return low, high

    def _loglog_ci(self, confidence_level):
        sf, d, n = self._sf, self._d, self._n

        with np.errstate(divide='ignore', invalid='ignore'):
            var = 1 / np.log(sf) ** 2 * np.cumsum(d / (n * (n - d)))

        se = np.sqrt(var)
        z = special.ndtri(1 / 2 + confidence_level / 2)

        with np.errstate(divide='ignore'):
            lnl_points = np.log(-np.log(sf))

        z_se = z * se
        low = np.exp(-np.exp(lnl_points + z_se))
        high = np.exp(-np.exp(lnl_points - z_se))
        if self._kind == "cdf":
            low, high = 1-high, 1-low

        return low, high


@dataclass
class ECDFResult:
    """ Result object returned by `scipy.stats.ecdf`

    Attributes
    ----------
    cdf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
        An object representing the empirical cumulative distribution function.
    sf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
        An object representing the complement of the empirical cumulative
        distribution function.
    """
    cdf: EmpiricalDistributionFunction
    sf: EmpiricalDistributionFunction

    def __init__(self, q, cdf, sf, n, d):
        self.cdf = EmpiricalDistributionFunction(q, cdf, n, d, "cdf")
        self.sf = EmpiricalDistributionFunction(q, sf, n, d, "sf")


def _iv_CensoredData(
    sample: "npt.ArrayLike | CensoredData", param_name: str = "sample"
) -> CensoredData:
    """Attempt to convert `sample` to `CensoredData`."""
    if not isinstance(sample, CensoredData):
        try:  # takes care of input standardization/validation
            sample = CensoredData(uncensored=sample)
        except ValueError as e:
            message = str(e).replace('uncensored', param_name)
            raise type(e)(message) from e
    return sample


def ecdf(sample: "npt.ArrayLike | CensoredData") -> ECDFResult:
    """Empirical cumulative distribution function of a sample.

    The empirical cumulative distribution function (ECDF) is a step function
    estimate of the CDF of the distribution underlying a sample. This function
    returns objects representing both the empirical distribution function and
    its complement, the empirical survival function.

    Parameters
    ----------
    sample : 1D array_like or `scipy.stats.CensoredData`
        Besides array_like, instances of `scipy.stats.CensoredData` containing
        uncensored and right-censored observations are supported. Currently,
        other instances of `scipy.stats.CensoredData` will result in a
        ``NotImplementedError``.

    Returns
    -------
    res : `~scipy.stats._result_classes.ECDFResult`
        An object with the following attributes.

        cdf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
            An object representing the empirical cumulative distribution
            function.
        sf : `~scipy.stats._result_classes.EmpiricalDistributionFunction`
            An object representing the empirical survival function.

        The `cdf` and `sf` attributes themselves have the following attributes.

        quantiles : ndarray
            The unique values in the sample that defines the empirical CDF/SF.
        probabilities : ndarray
            The point estimates of the probabilities corresponding with
            `quantiles`.

        And the following methods:

        evaluate(x) :
            Evaluate the CDF/SF at the argument.

        plot(ax) :
            Plot the CDF/SF on the provided axes.

        confidence_interval(confidence_level=0.95) :
            Compute the confidence interval around the CDF/SF at the values in
            `quantiles`.

    Notes
    -----
    When each observation of the sample is a precise measurement, the ECDF
    steps up by ``1/len(sample)`` at each of the observations [1]_.

    When observations are lower bounds, upper bounds, or both upper and lower
    bounds, the data is said to be "censored", and `sample` may be provided as
    an instance of `scipy.stats.CensoredData`.

    For right-censored data, the ECDF is given by the Kaplan-Meier estimator
    [2]_; other forms of censoring are not supported at this time.

    Confidence intervals are computed according to the Greenwood formula or the
    more recent "Exponential Greenwood" formula as described in [4]_.

    References
    ----------
    .. [1] Conover, William Jay. Practical nonparametric statistics. Vol. 350.
           John Wiley & Sons, 1999.

    .. [2] Kaplan, Edward L., and Paul Meier. "Nonparametric estimation from
           incomplete observations." Journal of the American statistical
           association 53.282 (1958): 457-481.

    .. [3] Goel, Manish Kumar, Pardeep Khanna, and Jugal Kishore.
           "Understanding survival analysis: Kaplan-Meier estimate."
           International journal of Ayurveda research 1.4 (2010): 274.

    .. [4] Sawyer, Stanley. "The Greenwood and Exponential Greenwood Confidence
           Intervals in Survival Analysis."
           https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf

    Examples
    --------
    **Uncensored Data**

    As in the example from [1]_ page 79, five boys were selected at random from
    those in a single high school. Their one-mile run times were recorded as
    follows.

    >>> sample = [6.23, 5.58, 7.06, 6.42, 5.20]  # one-mile run times (minutes)

    The empirical distribution function, which approximates the distribution
    function of one-mile run times of the population from which the boys were
    sampled, is calculated as follows.

    >>> from scipy import stats
    >>> res = stats.ecdf(sample)
    >>> res.cdf.quantiles
    array([5.2 , 5.58, 6.23, 6.42, 7.06])
    >>> res.cdf.probabilities
    array([0.2, 0.4, 0.6, 0.8, 1. ])

    To plot the result as a step function:

    >>> import matplotlib.pyplot as plt
    >>> ax = plt.subplot()
    >>> res.cdf.plot(ax)
    >>> ax.set_xlabel('One-Mile Run Time (minutes)')
    >>> ax.set_ylabel('Empirical CDF')
    >>> plt.show()

    **Right-censored Data**

    As in the example from [1]_ page 91, the lives of ten car fanbelts were
    tested. Five tests concluded because the fanbelt being tested broke, but
    the remaining tests concluded for other reasons (e.g. the study ran out of
    funding, but the fanbelt was still functional). The mileage driven
    with the fanbelts were recorded as follows.

    >>> broken = [77, 47, 81, 56, 80]  # in thousands of miles driven
    >>> unbroken = [62, 60, 43, 71, 37]

    Precise survival times of the fanbelts that were still functional at the
    end of the tests are unknown, but they are known to exceed the values
    recorded in ``unbroken``. Therefore, these observations are said to be
    "right-censored", and the data is represented using
    `scipy.stats.CensoredData`.

    >>> sample = stats.CensoredData(uncensored=broken, right=unbroken)

    The empirical survival function is calculated as follows.

    >>> res = stats.ecdf(sample)
    >>> res.sf.quantiles
    array([37., 43., 47., 56., 60., 62., 71., 77., 80., 81.])
    >>> res.sf.probabilities
    array([1.   , 1.   , 0.875, 0.75 , 0.75 , 0.75 , 0.75 , 0.5  , 0.25 , 0.   ])

    To plot the result as a step function:

    >>> ax = plt.subplot()
    >>> res.sf.plot(ax)
    >>> ax.set_xlabel('Fanbelt Survival Time (thousands of miles)')
    >>> ax.set_ylabel('Empirical SF')
    >>> plt.show()

    """
    sample = _iv_CensoredData(sample)

    if sample.num_censored() == 0:
        res = _ecdf_uncensored(sample._uncensor())
    elif sample.num_censored() == sample._right.size:
        res = _ecdf_right_censored(sample)
    else:
        # Support additional censoring options in follow-up PRs
        message = ("Currently, only uncensored and right-censored data is "
                   "supported.")
        raise NotImplementedError(message)

    t, cdf, sf, n, d = res
    return ECDFResult(t, cdf, sf, n, d)


def _ecdf_uncensored(sample):
    sample = np.sort(sample)
    x, counts = np.unique(sample, return_counts=True)

    # [1].81 "the fraction of [observations] that are less than or equal to x
    events = np.cumsum(counts)
    n = sample.size
    cdf = events / n

    # [1].89 "the relative frequency of the sample that exceeds x in value"
    sf = 1 - cdf

    at_risk = np.concatenate(([n], n - events[:-1]))
    return x, cdf, sf, at_risk, counts


def _ecdf_right_censored(sample):
    # It is conventional to discuss right-censored data in terms of
    # "survival time", "death", and "loss" (e.g. [2]). We'll use that
    # terminology here.
    # This implementation was influenced by the references cited and also
    # https://www.youtube.com/watch?v=lxoWsVco_iM
    # https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator
    # In retrospect it is probably most easily compared against [3].
    # Ultimately, the data needs to be sorted, so this implementation is
    # written to avoid a separate call to `unique` after sorting. In hope of
    # better performance on large datasets, it also computes survival
    # probabilities at unique times only rather than at each observation.
    tod = sample._uncensored  # time of "death"
    tol = sample._right  # time of "loss"
    times = np.concatenate((tod, tol))
    died = np.asarray([1]*tod.size + [0]*tol.size)

    # sort by times
    i = np.argsort(times)
    times = times[i]
    died = died[i]
    at_risk = np.arange(times.size, 0, -1)

    # logical indices of unique times
    j = np.diff(times, prepend=-np.inf, append=np.inf) > 0
    j_l = j[:-1]  # first instances of unique times
    j_r = j[1:]  # last instances of unique times

    # get number at risk and deaths at each unique time
    t = times[j_l]  # unique times
    n = at_risk[j_l]  # number at risk at each unique time
    cd = np.cumsum(died)[j_r]  # cumulative deaths up to/including unique times
    d = np.diff(cd, prepend=0)  # deaths at each unique time

    # compute survival function
    sf = np.cumprod((n - d) / n)
    cdf = 1 - sf
    return t, cdf, sf, n, d


@dataclass
class LogRankResult:
    """Result object returned by `scipy.stats.logrank`.

    Attributes
    ----------
    statistic : float ndarray
        The computed statistic (defined below). Its magnitude is the
        square root of the magnitude returned by most other logrank test
        implementations.
    pvalue : float ndarray
        The computed p-value of the test.
    """
    statistic: np.ndarray
    pvalue: np.ndarray


def logrank(
    x: "npt.ArrayLike | CensoredData",
    y: "npt.ArrayLike | CensoredData",
    alternative: Literal['two-sided', 'less', 'greater'] = "two-sided"
) -> LogRankResult:
    r"""Compare the survival distributions of two samples via the logrank test.

    Parameters
    ----------
    x, y : array_like or CensoredData
        Samples to compare based on their empirical survival functions.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.

        The null hypothesis is that the survival distributions of the two
        groups, say *X* and *Y*, are identical.

        The following alternative hypotheses [4]_ are available (default is
        'two-sided'):

        * 'two-sided': the survival distributions of the two groups are not
          identical.
        * 'less': survival of group *X* is favored: the group *X* failure rate
          function is less than the group *Y* failure rate function at some
          times.
        * 'greater': survival of group *Y* is favored: the group *X* failure
          rate function is greater than the group *Y* failure rate function at
          some times.

    Returns
    -------
    res : `~scipy.stats._result_classes.LogRankResult`
        An object containing attributes:

        statistic : float ndarray
            The computed statistic (defined below). Its magnitude is the
            square root of the magnitude returned by most other logrank test
            implementations.
        pvalue : float ndarray
            The computed p-value of the test.

    See Also
    --------
    scipy.stats.ecdf

    Notes
    -----
    The logrank test [1]_ compares the observed number of events to
    the expected number of events under the null hypothesis that the two
    samples were drawn from the same distribution. The statistic is

    .. math::

        Z_i = \frac{\sum_{j=1}^J(O_{i,j}-E_{i,j})}{\sqrt{\sum_{j=1}^J V_{i,j}}}
        \rightarrow \mathcal{N}(0,1)

    where

    .. math::

        E_{i,j} = O_j \frac{N_{i,j}}{N_j},
        \qquad
        V_{i,j} = E_{i,j} \left(\frac{N_j-O_j}{N_j}\right)
        \left(\frac{N_j-N_{i,j}}{N_j-1}\right),

    :math:`i` denotes the group (i.e. it may assume values :math:`x` or
    :math:`y`, or it may be omitted to refer to the combined sample)
    :math:`j` denotes the time (at which an event occurred),
    :math:`N` is the number of subjects at risk just before an event occurred,
    and :math:`O` is the observed number of events at that time.

    The ``statistic`` :math:`Z_x` returned by `logrank` is the (signed) square
    root of the statistic returned by many other implementations. Under the
    null hypothesis, :math:`Z_x**2` is asymptotically distributed according to
    the chi-squared distribution with one degree of freedom. Consequently,
    :math:`Z_x` is asymptotically distributed according to the standard normal
    distribution. The advantage of using :math:`Z_x` is that the sign
    information (i.e. whether the observed number of events tends to be less
    than or greater than the number expected under the null hypothesis) is
    preserved, allowing `scipy.stats.logrank` to offer one-sided alternative
    hypotheses.

    References
    ----------
    .. [1] Mantel N. "Evaluation of survival data and two new rank order
           statistics arising in its consideration."
           Cancer Chemotherapy Reports, 50(3):163-170, PMID: 5910392, 1966
    .. [2] Bland, Altman, "The logrank test", BMJ, 328:1073,
           :doi:`10.1136/bmj.328.7447.1073`, 2004
    .. [3] "Logrank test", Wikipedia,
           https://en.wikipedia.org/wiki/Logrank_test
    .. [4] Brown, Mark. "On the choice of variance for the log rank test."
           Biometrika 71.1 (1984): 65-74.
    .. [5] Klein, John P., and Melvin L. Moeschberger. Survival analysis:
           techniques for censored and truncated data. Vol. 1230. New York:
           Springer, 2003.

    Examples
    --------
    Reference [2]_ compared the survival times of patients with two different
    types of recurrent malignant gliomas. The samples below record the time
    (number of weeks) for which each patient participated in the study. The
    `scipy.stats.CensoredData` class is used because the data is
    right-censored: the uncensored observations correspond with observed deaths
    whereas the censored observations correspond with the patient leaving the
    study for another reason.

    >>> from scipy import stats
    >>> x = stats.CensoredData(
    ...     uncensored=[6, 13, 21, 30, 37, 38, 49, 50,
    ...                 63, 79, 86, 98, 202, 219],
    ...     right=[31, 47, 80, 82, 82, 149]
    ... )
    >>> y = stats.CensoredData(
    ...     uncensored=[10, 10, 12, 13, 14, 15, 16, 17, 18, 20, 24, 24,
    ...                 25, 28,30, 33, 35, 37, 40, 40, 46, 48, 76, 81,
    ...                 82, 91, 112, 181],
    ...     right=[34, 40, 70]
    ... )

    We can calculate and visualize the empirical survival functions
    of both groups as follows.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> ax = plt.subplot()
    >>> ecdf_x = stats.ecdf(x)
    >>> ecdf_x.sf.plot(ax, label='Astrocytoma')
    >>> ecdf_y = stats.ecdf(y)
    >>> ecdf_y.sf.plot(ax, label='Glioblastoma')
    >>> ax.set_xlabel('Time to death (weeks)')
    >>> ax.set_ylabel('Empirical SF')
    >>> plt.legend()
    >>> plt.show()

    Visual inspection of the empirical survival functions suggests that the
    survival times tend to be different between the two groups. To formally
    assess whether the difference is significant at the 1% level, we use the
    logrank test.

    >>> res = stats.logrank(x=x, y=y)
    >>> res.statistic
    -2.73799
    >>> res.pvalue
    0.00618

    The p-value is less than 1%, so we can consider the data to be evidence
    against the null hypothesis in favor of the alternative that there is a
    difference between the two survival functions.

    """
    # Input validation. `alternative` IV handled in `_get_pvalue` below.
    x = _iv_CensoredData(sample=x, param_name='x')
    y = _iv_CensoredData(sample=y, param_name='y')

    # Combined sample. (Under H0, the two groups are identical.)
    xy = CensoredData(
        uncensored=np.concatenate((x._uncensored, y._uncensored)),
        right=np.concatenate((x._right, y._right))
    )

    # Extract data from the combined sample
    res = ecdf(xy)
    idx = res.sf._d.astype(bool)  # indices of observed events
    times_xy = res.sf.quantiles[idx]  # unique times of observed events
    at_risk_xy = res.sf._n[idx]  # combined number of subjects at risk
    deaths_xy = res.sf._d[idx]  # combined number of events

    # Get the number at risk within each sample.
    # First compute the number at risk in group X at each of the `times_xy`.
    # Could use `interpolate_1d`, but this is more compact.
    res_x = ecdf(x)
    i = np.searchsorted(res_x.sf.quantiles, times_xy)
    at_risk_x = np.append(res_x.sf._n, 0)[i]  # 0 at risk after last time
    # Subtract from the combined number at risk to get number at risk in Y
    at_risk_y = at_risk_xy - at_risk_x

    # Compute the variance.
    num = at_risk_x * at_risk_y * deaths_xy * (at_risk_xy - deaths_xy)
    den = at_risk_xy**2 * (at_risk_xy - 1)
    # Note: when `at_risk_xy == 1`, we would have `at_risk_xy - 1 == 0` in the
    # numerator and denominator. Simplifying the fraction symbolically, we
    # would always find the overall quotient to be zero, so don't compute it.
    i = at_risk_xy > 1
    sum_var = np.sum(num[i]/den[i])

    # Get the observed and expected number of deaths in group X
    n_died_x = x._uncensored.size
    sum_exp_deaths_x = np.sum(at_risk_x * (deaths_xy/at_risk_xy))

    # Compute the statistic. This is the square root of that in references.
    statistic = (n_died_x - sum_exp_deaths_x)/np.sqrt(sum_var)

    # Equivalent to chi2(df=1).sf(statistic**2) when alternative='two-sided'
    norm = stats._stats_py._SimpleNormal()
    pvalue = stats._stats_py._get_pvalue(statistic, norm, alternative, xp=np)

    return LogRankResult(statistic=statistic[()], pvalue=pvalue[()])
<|endoftext|>#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#
# NOTE: To look at history using `git blame`, use `git blame -M -C -C`
#       instead of `git blame -Lxxx,+x`.
#
from ._distn_infrastructure import (rv_discrete, rv_continuous, rv_frozen)  # noqa: F401

from . import _continuous_distns
from . import _discrete_distns

from ._continuous_distns import *  # noqa: F403
from ._levy_stable import levy_stable
from ._discrete_distns import *  # noqa: F403
from ._entropy import entropy

# For backwards compatibility e.g. pymc expects distributions.__all__.
__all__ = ['rv_discrete', 'rv_continuous', 'rv_histogram', 'entropy']  # noqa: F405

# Add only the distribution names, not the *_gen names.
__all__ += _continuous_distns._distn_names
__all__ += ['levy_stable']
__all__ += _discrete_distns._distn_names
<|endoftext|>import numpy as np
from numpy import poly1d
from scipy.special import beta


# The following code was used to generate the Pade coefficients for the
# Tukey Lambda variance function.  Version 0.17 of mpmath was used.
#---------------------------------------------------------------------------
# import mpmath as mp
#
# mp.mp.dps = 60
#
# one   = mp.mpf(1)
# two   = mp.mpf(2)
#
# def mpvar(lam):
#     if lam == 0:
#         v = mp.pi**2 / three
#     else:
#         v = (two / lam**2) * (one / (one + two*lam) -
#                               mp.beta(lam + one, lam + one))
#     return v
#
# t = mp.taylor(mpvar, 0, 8)
# p, q = mp.pade(t, 4, 4)
# print("p =", [mp.fp.mpf(c) for c in p])
# print("q =", [mp.fp.mpf(c) for c in q])
#---------------------------------------------------------------------------

# Pade coefficients for the Tukey Lambda variance function.
_tukeylambda_var_pc = [3.289868133696453, 0.7306125098871127,
                       -0.5370742306855439, 0.17292046290190008,
                       -0.02371146284628187]
_tukeylambda_var_qc = [1.0, 3.683605511659861, 4.184152498888124,
                       1.7660926747377275, 0.2643989311168465]

# numpy.poly1d instances for the numerator and denominator of the
# Pade approximation to the Tukey Lambda variance.
_tukeylambda_var_p = poly1d(_tukeylambda_var_pc[::-1])
_tukeylambda_var_q = poly1d(_tukeylambda_var_qc[::-1])


def tukeylambda_variance(lam):
    """Variance of the Tukey Lambda distribution.

    Parameters
    ----------
    lam : array_like
        The lambda values at which to compute the variance.

    Returns
    -------
    v : ndarray
        The variance.  For lam < -0.5, the variance is not defined, so
        np.nan is returned.  For lam = 0.5, np.inf is returned.

    Notes
    -----
    In an interval around lambda=0, this function uses the [4,4] Pade
    approximation to compute the variance.  Otherwise it uses the standard
    formula (https://en.wikipedia.org/wiki/Tukey_lambda_distribution).  The
    Pade approximation is used because the standard formula has a removable
    discontinuity at lambda = 0, and does not produce accurate numerical
    results near lambda = 0.
    """
    lam = np.asarray(lam)
    shp = lam.shape
    lam = np.atleast_1d(lam).astype(np.float64)

    # For absolute values of lam less than threshold, use the Pade
    # approximation.
    threshold = 0.075

    # Play games with masks to implement the conditional evaluation of
    # the distribution.
    # lambda < -0.5:  var = nan
    low_mask = lam < -0.5
    # lambda == -0.5: var = inf
    neghalf_mask = lam == -0.5
    # abs(lambda) < threshold:  use Pade approximation
    small_mask = np.abs(lam) < threshold
    # else the "regular" case:  use the explicit formula.
    reg_mask = ~(low_mask | neghalf_mask | small_mask)

    # Get the 'lam' values for the cases where they are needed.
    small = lam[small_mask]
    reg = lam[reg_mask]

    # Compute the function for each case.
    v = np.empty_like(lam)
    v[low_mask] = np.nan
    v[neghalf_mask] = np.inf
    if small.size > 0:
        # Use the Pade approximation near lambda = 0.
        v[small_mask] = _tukeylambda_var_p(small) / _tukeylambda_var_q(small)
    if reg.size > 0:
        v[reg_mask] = (2.0 / reg**2) * (1.0 / (1.0 + 2 * reg) -
                                        beta(reg + 1, reg + 1))
    v.shape = shp
    return v


# The following code was used to generate the Pade coefficients for the
# Tukey Lambda kurtosis function.  Version 0.17 of mpmath was used.
#---------------------------------------------------------------------------
# import mpmath as mp
#
# mp.mp.dps = 60
#
# one   = mp.mpf(1)
# two   = mp.mpf(2)
# three = mp.mpf(3)
# four  = mp.mpf(4)
#
# def mpkurt(lam):
#     if lam == 0:
#         k = mp.mpf(6)/5
#     else:
#         numer = (one/(four*lam+one) - four*mp.beta(three*lam+one, lam+one) +
#                  three*mp.beta(two*lam+one, two*lam+one))
#         denom = two*(one/(two*lam+one) - mp.beta(lam+one,lam+one))**2
#         k = numer / denom - three
#     return k
#
# # There is a bug in mpmath 0.17: when we use the 'method' keyword of the
# # taylor function and we request a degree 9 Taylor polynomial, we actually
# # get degree 8.
# t = mp.taylor(mpkurt, 0, 9, method='quad', radius=0.01)
# t = [mp.chop(c, tol=1e-15) for c in t]
# p, q = mp.pade(t, 4, 4)
# print("p =", [mp.fp.mpf(c) for c in p])
# print("q =", [mp.fp.mpf(c) for c in q])
#---------------------------------------------------------------------------

# Pade coefficients for the Tukey Lambda kurtosis function.
_tukeylambda_kurt_pc = [1.2, -5.853465139719495, -22.653447381131077,
                        0.20601184383406815, 4.59796302262789]
_tukeylambda_kurt_qc = [1.0, 7.171149192233599, 12.96663094361842,
                        0.43075235247853005, -2.789746758009912]

# numpy.poly1d instances for the numerator and denominator of the
# Pade approximation to the Tukey Lambda kurtosis.
_tukeylambda_kurt_p = poly1d(_tukeylambda_kurt_pc[::-1])
_tukeylambda_kurt_q = poly1d(_tukeylambda_kurt_qc[::-1])


def tukeylambda_kurtosis(lam):
    """Kurtosis of the Tukey Lambda distribution.

    Parameters
    ----------
    lam : array_like
        The lambda values at which to compute the variance.

    Returns
    -------
    v : ndarray
        The variance.  For lam < -0.25, the variance is not defined, so
        np.nan is returned.  For lam = 0.25, np.inf is returned.

    """
    lam = np.asarray(lam)
    shp = lam.shape
    lam = np.atleast_1d(lam).astype(np.float64)

    # For absolute values of lam less than threshold, use the Pade
    # approximation.
    threshold = 0.055

    # Use masks to implement the conditional evaluation of the kurtosis.
    # lambda < -0.25:  kurtosis = nan
    low_mask = lam < -0.25
    # lambda == -0.25: kurtosis = inf
    negqrtr_mask = lam == -0.25
    # lambda near 0:  use Pade approximation
    small_mask = np.abs(lam) < threshold
    # else the "regular" case:  use the explicit formula.
    reg_mask = ~(low_mask | negqrtr_mask | small_mask)

    # Get the 'lam' values for the cases where they are needed.
    small = lam[small_mask]
    reg = lam[reg_mask]

    # Compute the function for each case.
    k = np.empty_like(lam)
    k[low_mask] = np.nan
    k[negqrtr_mask] = np.inf
    if small.size > 0:
        k[small_mask] = _tukeylambda_kurt_p(small) / _tukeylambda_kurt_q(small)
    if reg.size > 0:
        numer = (1.0 / (4 * reg + 1) - 4 * beta(3 * reg + 1, reg + 1) +
                 3 * beta(2 * reg + 1, 2 * reg + 1))
        denom = 2 * (1.0/(2 * reg + 1) - beta(reg + 1, reg + 1))**2
        k[reg_mask] = numer / denom - 3

    # The return value will be a numpy array; resetting the shape ensures that
    # if `lam` was a scalar, the return value is a 0-d array.
    k.shape = shp
    return k
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="mvn",
                                   private_modules=["_mvn"], all=__all__,
                                   attribute=name)
<|endoftext|>import numpy as np
from functools import partial
from scipy import stats


def _bws_input_validation(x, y, alternative, method):
    ''' Input validation and standardization for bws test'''
    x, y = np.atleast_1d(x, y)
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError('`x` and `y` must be exactly one-dimensional.')
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError('`x` and `y` must not contain NaNs.')
    if np.size(x) == 0 or np.size(y) == 0:
        raise ValueError('`x` and `y` must be of nonzero size.')

    z = stats.rankdata(np.concatenate((x, y)))
    x, y = z[:len(x)], z[len(x):]

    alternatives = {'two-sided', 'less', 'greater'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}.')

    method = stats.PermutationMethod() if method is None else method
    if not isinstance(method, stats.PermutationMethod):
        raise ValueError('`method` must be an instance of '
                         '`scipy.stats.PermutationMethod`')

    return x, y, alternative, method


def _bws_statistic(x, y, alternative, axis):
    '''Compute the BWS test statistic for two independent samples'''
    # Public function currently does not accept `axis`, but `permutation_test`
    # uses `axis` to make vectorized call.

    Ri, Hj = np.sort(x, axis=axis), np.sort(y, axis=axis)
    n, m = Ri.shape[axis], Hj.shape[axis]
    i, j = np.arange(1, n+1), np.arange(1, m+1)

    Bx_num = Ri - (m + n)/n * i
    By_num = Hj - (m + n)/m * j

    if alternative == 'two-sided':
        Bx_num *= Bx_num
        By_num *= By_num
    else:
        Bx_num *= np.abs(Bx_num)
        By_num *= np.abs(By_num)

    Bx_den = i/(n+1) * (1 - i/(n+1)) * m*(m+n)/n
    By_den = j/(m+1) * (1 - j/(m+1)) * n*(m+n)/m

    Bx = 1/n * np.sum(Bx_num/Bx_den, axis=axis)
    By = 1/m * np.sum(By_num/By_den, axis=axis)

    B = (Bx + By) / 2 if alternative == 'two-sided' else (Bx - By) / 2

    return B


def bws_test(x, y, *, alternative="two-sided", method=None):
    r'''Perform the Baumgartner-Weiss-Schindler test on two independent samples.

    The Baumgartner-Weiss-Schindler (BWS) test is a nonparametric test of 
    the null hypothesis that the distribution underlying sample `x` 
    is the same as the distribution underlying sample `y`. Unlike 
    the Kolmogorov-Smirnov, Wilcoxon, and Cramer-Von Mises tests, 
    the BWS test weights the integral by the variance of the difference
    in cumulative distribution functions (CDFs), emphasizing the tails of the
    distributions, which increases the power of the test in many applications.

    Parameters
    ----------
    x, y : array-like
        1-d arrays of samples.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        Let *F(u)* and *G(u)* be the cumulative distribution functions of the
        distributions underlying `x` and `y`, respectively. Then the following
        alternative hypotheses are available:

        * 'two-sided': the distributions are not equal, i.e. *F(u) ≠ G(u)* for
          at least one *u*.
        * 'less': the distribution underlying `x` is stochastically less than
          the distribution underlying `y`, i.e. *F(u) >= G(u)* for all *u*.
        * 'greater': the distribution underlying `x` is stochastically greater
          than the distribution underlying `y`, i.e. *F(u) <= G(u)* for all
          *u*.

        Under a more restrictive set of assumptions, the alternative hypotheses
        can be expressed in terms of the locations of the distributions;
        see [2] section 5.1.
    method : PermutationMethod, optional
        Configures the method used to compute the p-value. The default is
        the default `PermutationMethod` object.

    Returns
    -------
    res : PermutationTestResult
    An object with attributes:

    statistic : float
        The observed test statistic of the data.
    pvalue : float
        The p-value for the given alternative.
    null_distribution : ndarray
        The values of the test statistic generated under the null hypothesis.

    See also
    --------
    scipy.stats.wilcoxon, scipy.stats.mannwhitneyu, scipy.stats.ttest_ind

    Notes
    -----
    When ``alternative=='two-sided'``, the statistic is defined by the
    equations given in [1]_ Section 2. This statistic is not appropriate for
    one-sided alternatives; in that case, the statistic is the *negative* of
    that given by the equations in [1]_ Section 2. Consequently, when the
    distribution of the first sample is stochastically greater than that of the
    second sample, the statistic will tend to be positive.

    References
    ----------
    .. [1] Neuhäuser, M. (2005). Exact Tests Based on the
           Baumgartner-Weiss-Schindler Statistic: A Survey. Statistical Papers,
           46(1), 1-29.
    .. [2] Fay, M. P., & Proschan, M. A. (2010). Wilcoxon-Mann-Whitney or t-test?
           On assumptions for hypothesis tests and multiple interpretations of 
           decision rules. Statistics surveys, 4, 1.

    Examples
    --------
    We follow the example of table 3 in [1]_: Fourteen children were divided
    randomly into two groups. Their ranks at performing a specific tests are
    as follows.

    >>> import numpy as np
    >>> x = [1, 2, 3, 4, 6, 7, 8]
    >>> y = [5, 9, 10, 11, 12, 13, 14]

    We use the BWS test to assess whether there is a statistically significant
    difference between the two groups.
    The null hypothesis is that there is no difference in the distributions of
    performance between the two groups. We decide that a significance level of
    1% is required to reject the null hypothesis in favor of the alternative
    that the distributions are different.
    Since the number of samples is very small, we can compare the observed test
    statistic against the *exact* distribution of the test statistic under the
    null hypothesis.

    >>> from scipy.stats import bws_test
    >>> res = bws_test(x, y)
    >>> print(res.statistic)
    5.132167152575315

    This agrees with :math:`B = 5.132` reported in [1]_. The *p*-value produced
    by `bws_test` also agrees with :math:`p = 0.0029` reported in [1]_.

    >>> print(res.pvalue)
    0.002913752913752914

    Because the p-value is below our threshold of 1%, we take this as evidence
    against the null hypothesis in favor of the alternative that there is a
    difference in performance between the two groups.
    '''

    x, y, alternative, method = _bws_input_validation(x, y, alternative,
                                                      method)
    bws_statistic = partial(_bws_statistic, alternative=alternative)

    permutation_alternative = 'less' if alternative == 'less' else 'greater'
    res = stats.permutation_test((x, y), bws_statistic,
                                 alternative=permutation_alternative,
                                 **method._asdict())

    return res
<|endoftext|>import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state, _transition_to_rng


def _combine_bounds(name, user_bounds, shape_domain, integral):
    """Intersection of user-defined bounds and distribution PDF/PMF domain"""

    user_bounds = np.atleast_1d(user_bounds)

    if user_bounds[0] > user_bounds[1]:
        message = (f"There are no values for `{name}` on the interval "
                   f"{list(user_bounds)}.")
        raise ValueError(message)

    bounds = (max(user_bounds[0], shape_domain[0]),
              min(user_bounds[1], shape_domain[1]))

    if integral and (np.ceil(bounds[0]) > np.floor(bounds[1])):
        message = (f"There are no integer values for `{name}` on the interval "
                   f"defined by the user-provided bounds and the domain "
                   "of the distribution.")
        raise ValueError(message)
    elif not integral and (bounds[0] > bounds[1]):
        message = (f"There are no values for `{name}` on the interval "
                   f"defined by the user-provided bounds and the domain "
                   "of the distribution.")
        raise ValueError(message)

    if not np.all(np.isfinite(bounds)):
        message = (f"The intersection of user-provided bounds for `{name}` "
                   f"and the domain of the distribution is not finite. Please "
                   f"provide finite bounds for shape `{name}` in `bounds`.")
        raise ValueError(message)

    return bounds


class FitResult:
    r"""Result of fitting a discrete or continuous distribution to data

    Attributes
    ----------
    params : namedtuple
        A namedtuple containing the maximum likelihood estimates of the
        shape parameters, location, and (if applicable) scale of the
        distribution.
    success : bool or None
        Whether the optimizer considered the optimization to terminate
        successfully or not.
    message : str or None
        Any status message provided by the optimizer.

    """

    def __init__(self, dist, data, discrete, res):
        self._dist = dist
        self._data = data
        self.discrete = discrete
        self.pxf = getattr(dist, "pmf", None) or getattr(dist, "pdf", None)

        shape_names = [] if dist.shapes is None else dist.shapes.split(", ")
        if not discrete:
            FitParams = namedtuple('FitParams', shape_names + ['loc', 'scale'])
        else:
            FitParams = namedtuple('FitParams', shape_names + ['loc'])

        self.params = FitParams(*res.x)

        # Optimizer can report success even when nllf is infinite
        if res.success and not np.isfinite(self.nllf()):
            res.success = False
            res.message = ("Optimization converged to parameter values that "
                           "are inconsistent with the data.")
        self.success = getattr(res, "success", None)
        self.message = getattr(res, "message", None)

    def __repr__(self):
        keys = ["params", "success", "message"]
        m = max(map(len, keys)) + 1
        return '\n'.join([key.rjust(m) + ': ' + repr(getattr(self, key))
                          for key in keys if getattr(self, key) is not None])

    def nllf(self, params=None, data=None):
        """Negative log-likelihood function

        Evaluates the negative of the log-likelihood function of the provided
        data at the provided parameters.

        Parameters
        ----------
        params : tuple, optional
            The shape parameters, location, and (if applicable) scale of the
            distribution as a single tuple. Default is the maximum likelihood
            estimates (``self.params``).
        data : array_like, optional
            The data for which the log-likelihood function is to be evaluated.
            Default is the data to which the distribution was fit.

        Returns
        -------
        nllf : float
            The negative of the log-likelihood function.

        """
        params = params if params is not None else self.params
        data = data if data is not None else self._data
        return self._dist.nnlf(theta=params, x=data)

    def plot(self, ax=None, *, plot_type="hist"):
        """Visually compare the data against the fitted distribution.

        Available only if `matplotlib` is installed.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes object to draw the plot onto, otherwise uses the current Axes.
        plot_type : {"hist", "qq", "pp", "cdf"}
            Type of plot to draw. Options include:

            - "hist": Superposes the PDF/PMF of the fitted distribution
              over a normalized histogram of the data.
            - "qq": Scatter plot of theoretical quantiles against the
              empirical quantiles. Specifically, the x-coordinates are the
              values of the fitted distribution PPF evaluated at the
              percentiles ``(np.arange(1, n) - 0.5)/n``, where ``n`` is the
              number of data points, and the y-coordinates are the sorted
              data points.
            - "pp": Scatter plot of theoretical percentiles against the
              observed percentiles. Specifically, the x-coordinates are the
              percentiles ``(np.arange(1, n) - 0.5)/n``, where ``n`` is
              the number of data points, and the y-coordinates are the values
              of the fitted distribution CDF evaluated at the sorted
              data points.
            - "cdf": Superposes the CDF of the fitted distribution over the
              empirical CDF. Specifically, the x-coordinates of the empirical
              CDF are the sorted data points, and the y-coordinates are the
              percentiles ``(np.arange(1, n) - 0.5)/n``, where ``n`` is
              the number of data points.

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            The matplotlib Axes object on which the plot was drawn.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import stats
        >>> import matplotlib.pyplot as plt  # matplotlib must be installed
        >>> rng = np.random.default_rng()
        >>> data = stats.nbinom(5, 0.5).rvs(size=1000, random_state=rng)
        >>> bounds = [(0, 30), (0, 1)]
        >>> res = stats.fit(stats.nbinom, data, bounds)
        >>> ax = res.plot()  # save matplotlib Axes object

        The `matplotlib.axes.Axes` object can be used to customize the plot.
        See `matplotlib.axes.Axes` documentation for details.

        >>> ax.set_xlabel('number of trials')  # customize axis label
        >>> ax.get_children()[0].set_linewidth(5)  # customize line widths
        >>> ax.legend()
        >>> plt.show()
        """
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError as exc:
            message = "matplotlib must be installed to use method `plot`."
            raise ModuleNotFoundError(message) from exc

        plots = {'histogram': self._hist_plot, 'qq': self._qq_plot,
                 'pp': self._pp_plot, 'cdf': self._cdf_plot,
                 'hist': self._hist_plot}
        if plot_type.lower() not in plots:
            message = f"`plot_type` must be one of {set(plots.keys())}"
            raise ValueError(message)
        plot = plots[plot_type.lower()]

        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        fit_params = np.atleast_1d(self.params)

        return plot(ax=ax, fit_params=fit_params)

    def _hist_plot(self, ax, fit_params):
        from matplotlib.ticker import MaxNLocator

        support = self._dist.support(*fit_params)
        lb = support[0] if np.isfinite(support[0]) else min(self._data)
        ub = support[1] if np.isfinite(support[1]) else max(self._data)
        pxf = "PMF" if self.discrete else "PDF"

        if self.discrete:
            x = np.arange(lb, ub + 2)
            y = self.pxf(x, *fit_params)
            ax.vlines(x[:-1], 0, y[:-1], label='Fitted Distribution PMF',
                      color='C0')
            options = dict(density=True, bins=x, align='left', color='C1')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel('k')
            ax.set_ylabel('PMF')
        else:
            x = np.linspace(lb, ub, 200)
            y = self.pxf(x, *fit_params)
            ax.plot(x, y, '--', label='Fitted Distribution PDF', color='C0')
            options = dict(density=True, bins=50, align='mid', color='C1')
            ax.set_xlabel('x')
            ax.set_ylabel('PDF')

        if len(self._data) > 50 or self.discrete:
            ax.hist(self._data, label="Histogram of Data", **options)
        else:
            ax.plot(self._data, np.zeros_like(self._data), "*",
                    label='Data', color='C1')

        ax.set_title(rf"Fitted $\tt {self._dist.name}$ {pxf} and Histogram")
        ax.legend(*ax.get_legend_handles_labels())
        return ax

    def _qp_plot(self, ax, fit_params, qq):
        data = np.sort(self._data)
        ps = self._plotting_positions(len(self._data))

        if qq:
            qp = "Quantiles"
            plot_type = 'Q-Q'
            x = self._dist.ppf(ps, *fit_params)
            y = data
        else:
            qp = "Percentiles"
            plot_type = 'P-P'
            x = ps
            y = self._dist.cdf(data, *fit_params)

        ax.plot(x, y, '.', label=f'Fitted Distribution {plot_type}',
                color='C0', zorder=1)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        if not qq:
            lim = max(lim[0], 0), min(lim[1], 1)

        if self.discrete and qq:
            q_min, q_max = int(lim[0]), int(lim[1]+1)
            q_ideal = np.arange(q_min, q_max)
            # q_ideal = np.unique(self._dist.ppf(ps, *fit_params))
            ax.plot(q_ideal, q_ideal, 'o', label='Reference', color='k',
                    alpha=0.25, markerfacecolor='none', clip_on=True)
        elif self.discrete and not qq:
            # The intent of this is to match the plot that would be produced
            # if x were continuous on [0, 1] and y were cdf(ppf(x)).
            # It can be approximated by letting x = np.linspace(0, 1, 1000),
            # but this might not look great when zooming in. The vertical
            # portions are included to indicate where the transition occurs
            # where the data completely obscures the horizontal portions.
            p_min, p_max = lim
            a, b = self._dist.support(*fit_params)
            p_min = max(p_min, 0 if np.isfinite(a) else 1e-3)
            p_max = min(p_max, 1 if np.isfinite(b) else 1-1e-3)
            q_min, q_max = self._dist.ppf([p_min, p_max], *fit_params)
            qs = np.arange(q_min-1, q_max+1)
            ps = self._dist.cdf(qs, *fit_params)
            ax.step(ps, ps, '-', label='Reference', color='k', alpha=0.25,
                    clip_on=True)
        else:
            ax.plot(lim, lim, '-', label='Reference', color='k', alpha=0.25,
                    clip_on=True)

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(rf"Fitted $\tt {self._dist.name}$ Theoretical {qp}")
        ax.set_ylabel(f"Data {qp}")
        ax.set_title(rf"Fitted $\tt {self._dist.name}$ {plot_type} Plot")
        ax.legend(*ax.get_legend_handles_labels())
        ax.set_aspect('equal')
        return ax

    def _qq_plot(self, **kwargs):
        return self._qp_plot(qq=True, **kwargs)

    def _pp_plot(self, **kwargs):
        return self._qp_plot(qq=False, **kwargs)

    def _plotting_positions(self, n, a=.5):
        # See https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Plotting_positions
        k = np.arange(1, n+1)
        return (k-a) / (n + 1 - 2*a)

    def _cdf_plot(self, ax, fit_params):
        data = np.sort(self._data)
        ecdf = self._plotting_positions(len(self._data))
        ls = '--' if len(np.unique(data)) < 30 else '.'
        xlabel = 'k' if self.discrete else 'x'
        ax.step(data, ecdf, ls, label='Empirical CDF', color='C1', zorder=0)

        xlim = ax.get_xlim()
        q = np.linspace(*xlim, 300)
        tcdf = self._dist.cdf(q, *fit_params)

        ax.plot(q, tcdf, label='Fitted Distribution CDF', color='C0', zorder=1)
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_title(rf"Fitted $\tt {self._dist.name}$ and Empirical CDF")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
        return ax


def fit(dist, data, bounds=None, *, guess=None, method='mle',
        optimizer=optimize.differential_evolution):
    r"""Fit a discrete or continuous distribution to data

    Given a distribution, data, and bounds on the parameters of the
    distribution, return maximum likelihood estimates of the parameters.

    Parameters
    ----------
    dist : `scipy.stats.rv_continuous` or `scipy.stats.rv_discrete`
        The object representing the distribution to be fit to the data.
    data : 1D array_like
        The data to which the distribution is to be fit. If the data contain
        any of ``np.nan``, ``np.inf``, or -``np.inf``, the fit method will
        raise a ``ValueError``.
    bounds : dict or sequence of tuples, optional
        If a dictionary, each key is the name of a parameter of the
        distribution, and the corresponding value is a tuple containing the
        lower and upper bound on that parameter.  If the distribution is
        defined only for a finite range of values of that parameter, no entry
        for that parameter is required; e.g., some distributions have
        parameters which must be on the interval [0, 1]. Bounds for parameters
        location (``loc``) and scale (``scale``) are optional; by default,
        they are fixed to 0 and 1, respectively.

        If a sequence, element *i* is a tuple containing the lower and upper
        bound on the *i*\ th parameter of the distribution. In this case,
        bounds for *all* distribution shape parameters must be provided.
        Optionally, bounds for location and scale may follow the
        distribution shape parameters.

        If a shape is to be held fixed (e.g. if it is known), the
        lower and upper bounds may be equal. If a user-provided lower or upper
        bound is beyond a bound of the domain for which the distribution is
        defined, the bound of the distribution's domain will replace the
        user-provided value. Similarly, parameters which must be integral
        will be constrained to integral values within the user-provided bounds.
    guess : dict or array_like, optional
        If a dictionary, each key is the name of a parameter of the
        distribution, and the corresponding value is a guess for the value
        of the parameter.

        If a sequence, element *i* is a guess for the *i*\ th parameter of the
        distribution. In this case, guesses for *all* distribution shape
        parameters must be provided.

        If `guess` is not provided, guesses for the decision variables will
        not be passed to the optimizer. If `guess` is provided, guesses for
        any missing parameters will be set at the mean of the lower and
        upper bounds. Guesses for parameters which must be integral will be
        rounded to integral values, and guesses that lie outside the
        intersection of the user-provided bounds and the domain of the
        distribution will be clipped.
    method : {'mle', 'mse'}
        With ``method="mle"`` (default), the fit is computed by minimizing
        the negative log-likelihood function. A large, finite penalty
        (rather than infinite negative log-likelihood) is applied for
        observations beyond the support of the distribution.
        With ``method="mse"``, the fit is computed by minimizing
        the negative log-product spacing function. The same penalty is applied
        for observations beyond the support. We follow the approach of [1]_,
        which is generalized for samples with repeated observations.
    optimizer : callable, optional
        `optimizer` is a callable that accepts the following positional
        argument.

        fun : callable
            The objective function to be optimized. `fun` accepts one argument
            ``x``, candidate shape parameters of the distribution, and returns
            the objective function value given ``x``, `dist`, and the provided
            `data`.
            The job of `optimizer` is to find values of the decision variables
            that minimizes `fun`.

        `optimizer` must also accept the following keyword argument.

        bounds : sequence of tuples
            The bounds on values of the decision variables; each element will
            be a tuple containing the lower and upper bound on a decision
            variable.

        If `guess` is provided, `optimizer` must also accept the following
        keyword argument.

        x0 : array_like
            The guesses for each decision variable.

        If the distribution has any shape parameters that must be integral or
        if the distribution is discrete and the location parameter is not
        fixed, `optimizer` must also accept the following keyword argument.

        integrality : array_like of bools
            For each decision variable, True if the decision variable
            must be constrained to integer values and False if the decision
            variable is continuous.

        `optimizer` must return an object, such as an instance of
        `scipy.optimize.OptimizeResult`, which holds the optimal values of
        the decision variables in an attribute ``x``. If attributes
        ``fun``, ``status``, or ``message`` are provided, they will be
        included in the result object returned by `fit`.

    Returns
    -------
    result : `~scipy.stats._result_classes.FitResult`
        An object with the following fields.

        params : namedtuple
            A namedtuple containing the maximum likelihood estimates of the
            shape parameters, location, and (if applicable) scale of the
            distribution.
        success : bool or None
            Whether the optimizer considered the optimization to terminate
            successfully or not.
        message : str or None
            Any status message provided by the optimizer.

        The object has the following method:

        nllf(params=None, data=None)
            By default, the negative log-likelihood function at the fitted
            `params` for the given `data`. Accepts a tuple containing
            alternative shapes, location, and scale of the distribution and
            an array of alternative data.

        plot(ax=None)
            Superposes the PDF/PMF of the fitted distribution over a normalized
            histogram of the data.

    See Also
    --------
    rv_continuous,  rv_discrete

    Notes
    -----
    Optimization is more likely to converge to the maximum likelihood estimate
    when the user provides tight bounds containing the maximum likelihood
    estimate. For example, when fitting a binomial distribution to data, the
    number of experiments underlying each sample may be known, in which case
    the corresponding shape parameter ``n`` can be fixed.

    References
    ----------
    .. [1] Shao, Yongzhao, and Marjorie G. Hahn. "Maximum product of spacings
           method: a unified formulation with illustration of strong
           consistency." Illinois Journal of Mathematics 43.3 (1999): 489-499.

    Examples
    --------
    Suppose we wish to fit a distribution to the following data.

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> dist = stats.nbinom
    >>> shapes = (5, 0.5)
    >>> data = dist.rvs(*shapes, size=1000, random_state=rng)

    Suppose we do not know how the data were generated, but we suspect that
    it follows a negative binomial distribution with parameters *n* and *p*\.
    (See `scipy.stats.nbinom`.) We believe that the parameter *n* was fewer
    than 30, and we know that the parameter *p* must lie on the interval
    [0, 1]. We record this information in a variable `bounds` and pass
    this information to `fit`.

    >>> bounds = [(0, 30), (0, 1)]
    >>> res = stats.fit(dist, data, bounds)

    `fit` searches within the user-specified `bounds` for the
    values that best match the data (in the sense of maximum likelihood
    estimation). In this case, it found shape values similar to those
    from which the data were actually generated.

    >>> res.params
    FitParams(n=5.0, p=0.5028157644634368, loc=0.0)  # may vary

    We can visualize the results by superposing the probability mass function
    of the distribution (with the shapes fit to the data) over a normalized
    histogram of the data.

    >>> import matplotlib.pyplot as plt  # matplotlib must be installed to plot
    >>> res.plot()
    >>> plt.show()

    Note that the estimate for *n* was exactly integral; this is because
    the domain of the `nbinom` PMF includes only integral *n*, and the `nbinom`
    object "knows" that. `nbinom` also knows that the shape *p* must be a
    value between 0 and 1. In such a case - when the domain of the distribution
    with respect to a parameter is finite - we are not required to specify
    bounds for the parameter.

    >>> bounds = {'n': (0, 30)}  # omit parameter p using a `dict`
    >>> res2 = stats.fit(dist, data, bounds)
    >>> res2.params
    FitParams(n=5.0, p=0.5016492009232932, loc=0.0)  # may vary

    If we wish to force the distribution to be fit with *n* fixed at 6, we can
    set both the lower and upper bounds on *n* to 6. Note, however, that the
    value of the objective function being optimized is typically worse (higher)
    in this case.

    >>> bounds = {'n': (6, 6)}  # fix parameter `n`
    >>> res3 = stats.fit(dist, data, bounds)
    >>> res3.params
    FitParams(n=6.0, p=0.5486556076755706, loc=0.0)  # may vary
    >>> res3.nllf() > res.nllf()
    True  # may vary

    Note that the numerical results of the previous examples are typical, but
    they may vary because the default optimizer used by `fit`,
    `scipy.optimize.differential_evolution`, is stochastic. However, we can
    customize the settings used by the optimizer to ensure reproducibility -
    or even use a different optimizer entirely - using the `optimizer`
    parameter.

    >>> from scipy.optimize import differential_evolution
    >>> rng = np.random.default_rng(767585560716548)
    >>> def optimizer(fun, bounds, *, integrality):
    ...     return differential_evolution(fun, bounds, strategy='best2bin',
    ...                                   rng=rng, integrality=integrality)
    >>> bounds = [(0, 30), (0, 1)]
    >>> res4 = stats.fit(dist, data, bounds, optimizer=optimizer)
    >>> res4.params
    FitParams(n=5.0, p=0.5015183149259951, loc=0.0)

    """
    # --- Input Validation / Standardization --- #
    user_bounds = bounds
    user_guess = guess

    # distribution input validation and information collection
    if hasattr(dist, "pdf"):  # can't use isinstance for types
        default_bounds = {'loc': (0, 0), 'scale': (1, 1)}
        discrete = False
    elif hasattr(dist, "pmf"):
        default_bounds = {'loc': (0, 0)}
        discrete = True
    else:
        message = ("`dist` must be an instance of `rv_continuous` "
                   "or `rv_discrete.`")
        raise ValueError(message)

    try:
        param_info = dist._param_info()
    except AttributeError as e:
        message = (f"Distribution `{dist.name}` is not yet supported by "
                   "`scipy.stats.fit` because shape information has "
                   "not been defined.")
        raise ValueError(message) from e

    # data input validation
    data = np.asarray(data)
    if data.ndim != 1:
        message = "`data` must be exactly one-dimensional."
        raise ValueError(message)
    if not (np.issubdtype(data.dtype, np.number)
            and np.all(np.isfinite(data))):
        message = "All elements of `data` must be finite numbers."
        raise ValueError(message)

    # bounds input validation and information collection
    n_params = len(param_info)
    n_shapes = n_params - (1 if discrete else 2)
    param_list = [param.name for param in param_info]
    param_names = ", ".join(param_list)
    shape_names = ", ".join(param_list[:n_shapes])

    if user_bounds is None:
        user_bounds = {}

    if isinstance(user_bounds, dict):
        default_bounds.update(user_bounds)
        user_bounds = default_bounds
        user_bounds_array = np.empty((n_params, 2))
        for i in range(n_params):
            param_name = param_info[i].name
            user_bound = user_bounds.pop(param_name, None)
            if user_bound is None:
                user_bound = param_info[i].domain
            user_bounds_array[i] = user_bound
        if user_bounds:
            message = ("Bounds provided for the following unrecognized "
                       f"parameters will be ignored: {set(user_bounds)}")
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    else:
        try:
            user_bounds = np.asarray(user_bounds, dtype=float)
            if user_bounds.size == 0:
                user_bounds = np.empty((0, 2))
        except ValueError as e:
            message = ("Each element of a `bounds` sequence must be a tuple "
                       "containing two elements: the lower and upper bound of "
                       "a distribution parameter.")
            raise ValueError(message) from e
        if (user_bounds.ndim != 2 or user_bounds.shape[1] != 2):
            message = ("Each element of `bounds` must be a tuple specifying "
                       "the lower and upper bounds of a shape parameter")
            raise ValueError(message)
        if user_bounds.shape[0] < n_shapes:
            message = (f"A `bounds` sequence must contain at least {n_shapes} "
                       "elements: tuples specifying the lower and upper "
                       f"bounds of all shape parameters {shape_names}.")
            raise ValueError(message)
        if user_bounds.shape[0] > n_params:
            message = ("A `bounds` sequence may not contain more than "
                       f"{n_params} elements: tuples specifying the lower and "
                       "upper bounds of distribution parameters "
                       f"{param_names}.")
            raise ValueError(message)

        user_bounds_array = np.empty((n_params, 2))
        user_bounds_array[n_shapes:] = list(default_bounds.values())
        user_bounds_array[:len(user_bounds)] = user_bounds

    user_bounds = user_bounds_array
    validated_bounds = []
    for i in range(n_params):
        name = param_info[i].name
        user_bound = user_bounds_array[i]
        param_domain = param_info[i].domain
        integral = param_info[i].integrality
        combined = _combine_bounds(name, user_bound, param_domain, integral)
        validated_bounds.append(combined)

    bounds = np.asarray(validated_bounds)
    integrality = [param.integrality for param in param_info]

    # guess input validation

    if user_guess is None:
        guess_array = None
    elif isinstance(user_guess, dict):
        default_guess = {param.name: np.mean(bound)
                         for param, bound in zip(param_info, bounds)}
        unrecognized = set(user_guess) - set(default_guess)
        if unrecognized:
            message = ("Guesses provided for the following unrecognized "
                       f"parameters will be ignored: {unrecognized}")
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        default_guess.update(user_guess)

        message = ("Each element of `guess` must be a scalar "
                   "guess for a distribution parameter.")
        try:
            guess_array = np.asarray([default_guess[param.name]
                                      for param in param_info], dtype=float)
        except ValueError as e:
            raise ValueError(message) from e

    else:
        message = ("Each element of `guess` must be a scalar "
                   "guess for a distribution parameter.")
        try:
            user_guess = np.asarray(user_guess, dtype=float)
        except ValueError as e:
            raise ValueError(message) from e
        if user_guess.ndim != 1:
            raise ValueError(message)
        if user_guess.shape[0] < n_shapes:
            message = (f"A `guess` sequence must contain at least {n_shapes} "
                       "elements: scalar guesses for the distribution shape "
                       f"parameters {shape_names}.")
            raise ValueError(message)
        if user_guess.shape[0] > n_params:
            message = ("A `guess` sequence may not contain more than "
                       f"{n_params} elements: scalar guesses for the "
                       f"distribution parameters {param_names}.")
            raise ValueError(message)

        guess_array = np.mean(bounds, axis=1)
        guess_array[:len(user_guess)] = user_guess

    if guess_array is not None:
        guess_rounded = guess_array.copy()

        guess_rounded[integrality] = np.round(guess_rounded[integrality])
        rounded = np.where(guess_rounded != guess_array)[0]
        for i in rounded:
            message = (f"Guess for parameter `{param_info[i].name}` "
                       f"rounded from {guess_array[i]} to {guess_rounded[i]}.")
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        guess_clipped = np.clip(guess_rounded, bounds[:, 0], bounds[:, 1])
        clipped = np.where(guess_clipped != guess_rounded)[0]
        for i in clipped:
            message = (f"Guess for parameter `{param_info[i].name}` "
                       f"clipped from {guess_rounded[i]} to "
                       f"{guess_clipped[i]}.")
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        guess = guess_clipped
    else:
        guess = None

    # --- Fitting --- #
    def nllf(free_params, data=data):  # bind data NOW
        with np.errstate(invalid='ignore', divide='ignore'):
            return dist._penalized_nnlf(free_params, data)

    def nlpsf(free_params, data=data):  # bind data NOW
        with np.errstate(invalid='ignore', divide='ignore'):
            return dist._penalized_nlpsf(free_params, data)

    methods = {'mle': nllf, 'mse': nlpsf}
    objective = methods[method.lower()]

    with np.errstate(invalid='ignore', divide='ignore'):
        kwds = {}
        if bounds is not None:
            kwds['bounds'] = bounds
        if np.any(integrality):
            kwds['integrality'] = integrality
        if guess is not None:
            kwds['x0'] = guess
        res = optimizer(objective, **kwds)

    return FitResult(dist, data, discrete, res)


GoodnessOfFitResult = namedtuple('GoodnessOfFitResult',
                                 ('fit_result', 'statistic', 'pvalue',
                                  'null_distribution'))


@_transition_to_rng('random_state')
def goodness_of_fit(dist, data, *, known_params=None, fit_params=None,
                    guessed_params=None, statistic='ad', n_mc_samples=9999,
                    rng=None):
    r"""
    Perform a goodness of fit test comparing data to a distribution family.

    Given a distribution family and data, perform a test of the null hypothesis
    that the data were drawn from a distribution in that family. Any known
    parameters of the distribution may be specified. Remaining parameters of
    the distribution will be fit to the data, and the p-value of the test
    is computed accordingly. Several statistics for comparing the distribution
    to data are available.

    Parameters
    ----------
    dist : `scipy.stats.rv_continuous`
        The object representing the distribution family under the null
        hypothesis.
    data : 1D array_like
        Finite, uncensored data to be tested.
    known_params : dict, optional
        A dictionary containing name-value pairs of known distribution
        parameters. Monte Carlo samples are randomly drawn from the
        null-hypothesized distribution with these values of the parameters.
        Before the statistic is evaluated for the observed `data` and each
        Monte Carlo sample, only remaining unknown parameters of the
        null-hypothesized distribution family are fit to the samples; the
        known parameters are held fixed. If all parameters of the distribution
        family are known, then the step of fitting the distribution family to
        each sample is omitted.
    fit_params : dict, optional
        A dictionary containing name-value pairs of distribution parameters
        that have already been fit to the data, e.g. using `scipy.stats.fit`
        or the ``fit`` method of `dist`. Monte Carlo samples are drawn from the
        null-hypothesized distribution with these specified values of the
        parameter. However, these and all other unknown parameters of the
        null-hypothesized distribution family are always fit to the sample,
        whether that is the observed `data` or a Monte Carlo sample, before
        the statistic is evaluated.
    guessed_params : dict, optional
        A dictionary containing name-value pairs of distribution parameters
        which have been guessed. These parameters are always considered as
        free parameters and are fit both to the provided `data` as well as
        to the Monte Carlo samples drawn from the null-hypothesized
        distribution. The purpose of these `guessed_params` is to be used as
        initial values for the numerical fitting procedure.
    statistic : {"ad", "ks", "cvm", "filliben"} or callable, optional
        The statistic used to compare data to a distribution after fitting
        unknown parameters of the distribution family to the data. The
        Anderson-Darling ("ad") [1]_, Kolmogorov-Smirnov ("ks") [1]_,
        Cramer-von Mises ("cvm") [1]_, and Filliben ("filliben") [7]_
        statistics are available.  Alternatively, a callable with signature
        ``(dist, data, axis)`` may be supplied to compute the statistic. Here
        ``dist`` is a frozen distribution object (potentially with array
        parameters), ``data`` is an array of Monte Carlo samples (of
        compatible shape), and ``axis`` is the axis of ``data`` along which
        the statistic must be computed.
    n_mc_samples : int, default: 9999
        The number of Monte Carlo samples drawn from the null hypothesized
        distribution to form the null distribution of the statistic. The
        sample size of each is the same as the given `data`.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

    Returns
    -------
    res : GoodnessOfFitResult
        An object with the following attributes.

        fit_result : `~scipy.stats._result_classes.FitResult`
            An object representing the fit of the provided `dist` to `data`.
            This  object includes the values of distribution family parameters
            that fully define the null-hypothesized distribution, that is,
            the distribution from which Monte Carlo samples are drawn.
        statistic : float
            The value of the statistic comparing provided `data` to the
            null-hypothesized distribution.
        pvalue : float
            The proportion of elements in the null distribution with
            statistic values at least as extreme as the statistic value of the
            provided `data`.
        null_distribution : ndarray
            The value of the statistic for each Monte Carlo sample
            drawn from the null-hypothesized distribution.

    Notes
    -----
    This is a generalized Monte Carlo goodness-of-fit procedure, special cases
    of which correspond with various Anderson-Darling tests, Lilliefors' test,
    etc. The test is described in [2]_, [3]_, and [4]_ as a parametric
    bootstrap test. This is a Monte Carlo test in which parameters that
    specify the distribution from which samples are drawn have been estimated
    from the data. We describe the test using "Monte Carlo" rather than
    "parametric bootstrap" throughout to avoid confusion with the more familiar
    nonparametric bootstrap, and describe how the test is performed below.

    *Traditional goodness of fit tests*

    Traditionally, critical values corresponding with a fixed set of
    significance levels are pre-calculated using Monte Carlo methods. Users
    perform the test by calculating the value of the test statistic only for
    their observed `data` and comparing this value to tabulated critical
    values. This practice is not very flexible, as tables are not available for
    all distributions and combinations of known and unknown parameter values.
    Also, results can be inaccurate when critical values are interpolated from
    limited tabulated data to correspond with the user's sample size and
    fitted parameter values. To overcome these shortcomings, this function
    allows the user to perform the Monte Carlo trials adapted to their
    particular data.

    *Algorithmic overview*

    In brief, this routine executes the following steps:

      1. Fit unknown parameters to the given `data`, thereby forming the
         "null-hypothesized" distribution, and compute the statistic of
         this pair of data and distribution.
      2. Draw random samples from this null-hypothesized distribution.
      3. Fit the unknown parameters to each random sample.
      4. Calculate the statistic between each sample and the distribution that
         has been fit to the sample.
      5. Compare the value of the statistic corresponding with `data` from (1)
         against the values of the statistic corresponding with the random
         samples from (4). The p-value is the proportion of samples with a
         statistic value greater than or equal to the statistic of the observed
         data.

    In more detail, the steps are as follows.

    First, any unknown parameters of the distribution family specified by
    `dist` are fit to the provided `data` using maximum likelihood estimation.
    (One exception is the normal distribution with unknown location and scale:
    we use the bias-corrected standard deviation ``np.std(data, ddof=1)`` for
    the scale as recommended in [1]_.)
    These values of the parameters specify a particular member of the
    distribution family referred to as the "null-hypothesized distribution",
    that is, the distribution from which the data were sampled under the null
    hypothesis. The `statistic`, which compares data to a distribution, is
    computed between `data` and the null-hypothesized distribution.

    Next, many (specifically `n_mc_samples`) new samples, each containing the
    same number of observations as `data`, are drawn from the
    null-hypothesized distribution. All unknown parameters of the distribution
