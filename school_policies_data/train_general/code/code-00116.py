def system_install_dir_patterns(desc: HeaderDescriptor) -> tuple[str, ...]:
    """Return platform-aware, expanded system include-directory patterns."""
    patterns: list[str] = []
    if IS_WINDOWS:
        patterns.extend(os.path.expandvars(pattern) for pattern in desc.system_install_dirs_windows)
        return tuple(patterns)
    if desc.use_linux_multiarch_include_dir:
        multiarch = sysconfig.get_config_var("MULTIARCH")
        if isinstance(multiarch, str) and multiarch:
            patterns.append(os.path.join("/usr/include", multiarch))
    patterns.extend(os.path.expandvars(pattern) for pattern in desc.system_install_dirs)
    return tuple(patterns)


def resolve_conda_anchor(desc: HeaderDescriptor, conda_prefix: str) -> str | None:
    """Resolve the conda anchor point for header search on the current platform.

    Returns the directory that ``_locate_in_anchor_layout`` should use as
    *anchor_point*, or ``None`` if the conda layout is not usable.
    """
    if IS_WINDOWS:
        anchor = os.path.join(conda_prefix, "Library")
        return anchor if os.path.isdir(anchor) else None
    if desc.conda_targets_layout:
        targets_include_path = glob.glob(os.path.join(conda_prefix, "targets", "*", "include"))
        if not targets_include_path or len(targets_include_path) != 1:
            return None
        return os.path.dirname(targets_include_path[0])
    return conda_prefix
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy table exports derived from the authored header descriptor catalog.

The canonical data entry point is :mod:`header_descriptor_catalog`. This module
keeps historical constant names for backward compatibility by deriving them
from the catalog.
"""

from __future__ import annotations

from typing import Final

from cuda.pathfinder._headers.header_descriptor import system_install_dir_patterns
from cuda.pathfinder._headers.header_descriptor_catalog import HEADER_DESCRIPTOR_CATALOG
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

_CTK_DESCRIPTORS = tuple(desc for desc in HEADER_DESCRIPTOR_CATALOG if desc.packaged_with == "ctk")
_NON_CTK_DESCRIPTORS = tuple(desc for desc in HEADER_DESCRIPTOR_CATALOG if desc.packaged_with == "other")

SUPPORTED_HEADERS_CTK_COMMON: Final[dict[str, str]] = {
    desc.name: desc.header_basename
    for desc in _CTK_DESCRIPTORS
    if desc.available_on_linux and desc.available_on_windows
}
SUPPORTED_HEADERS_CTK_LINUX_ONLY: Final[dict[str, str]] = {
    desc.name: desc.header_basename
    for desc in _CTK_DESCRIPTORS
    if desc.available_on_linux and not desc.available_on_windows
}
SUPPORTED_HEADERS_CTK_WINDOWS_ONLY: Final[dict[str, str]] = {
    desc.name: desc.header_basename
    for desc in _CTK_DESCRIPTORS
    if desc.available_on_windows and not desc.available_on_linux
}

SUPPORTED_HEADERS_CTK_LINUX = SUPPORTED_HEADERS_CTK_COMMON | SUPPORTED_HEADERS_CTK_LINUX_ONLY
SUPPORTED_HEADERS_CTK_WINDOWS = SUPPORTED_HEADERS_CTK_COMMON | SUPPORTED_HEADERS_CTK_WINDOWS_ONLY
SUPPORTED_HEADERS_CTK_ALL = (
    SUPPORTED_HEADERS_CTK_COMMON | SUPPORTED_HEADERS_CTK_LINUX_ONLY | SUPPORTED_HEADERS_CTK_WINDOWS_ONLY
)
SUPPORTED_HEADERS_CTK: Final[dict[str, str]] = (
    SUPPORTED_HEADERS_CTK_WINDOWS if IS_WINDOWS else SUPPORTED_HEADERS_CTK_LINUX
)

SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK: Final[dict[str, tuple[str, ...]]] = {
    desc.name: desc.site_packages_dirs for desc in _CTK_DESCRIPTORS if desc.site_packages_dirs
}

SUPPORTED_HEADERS_NON_CTK_COMMON: Final[dict[str, str]] = {
    desc.name: desc.header_basename
    for desc in _NON_CTK_DESCRIPTORS
    if desc.available_on_linux and desc.available_on_windows
}
SUPPORTED_HEADERS_NON_CTK_LINUX_ONLY: Final[dict[str, str]] = {
    desc.name: desc.header_basename
    for desc in _NON_CTK_DESCRIPTORS
    if desc.available_on_linux and not desc.available_on_windows
}
SUPPORTED_HEADERS_NON_CTK_WINDOWS_ONLY: Final[dict[str, str]] = {
    desc.name: desc.header_basename
    for desc in _NON_CTK_DESCRIPTORS
    if desc.available_on_windows and not desc.available_on_linux
}

SUPPORTED_HEADERS_NON_CTK_LINUX = SUPPORTED_HEADERS_NON_CTK_COMMON | SUPPORTED_HEADERS_NON_CTK_LINUX_ONLY
SUPPORTED_HEADERS_NON_CTK_WINDOWS = SUPPORTED_HEADERS_NON_CTK_COMMON | SUPPORTED_HEADERS_NON_CTK_WINDOWS_ONLY
SUPPORTED_HEADERS_NON_CTK_ALL = (
    SUPPORTED_HEADERS_NON_CTK_COMMON | SUPPORTED_HEADERS_NON_CTK_LINUX_ONLY | SUPPORTED_HEADERS_NON_CTK_WINDOWS_ONLY
)
SUPPORTED_HEADERS_NON_CTK: Final[dict[str, str]] = (
    SUPPORTED_HEADERS_NON_CTK_WINDOWS if IS_WINDOWS else SUPPORTED_HEADERS_NON_CTK_LINUX
)

SUPPORTED_SITE_PACKAGE_HEADER_DIRS_NON_CTK: Final[dict[str, tuple[str, ...]]] = {
    desc.name: desc.site_packages_dirs for desc in _NON_CTK_DESCRIPTORS if desc.site_packages_dirs
}

SUPPORTED_INSTALL_DIRS_NON_CTK: Final[dict[str, tuple[str, ...]]] = {
    desc.name: patterns for desc in _NON_CTK_DESCRIPTORS if (patterns := system_install_dir_patterns(desc))
}
<|endoftext|>"""Functions to convert NetworkX graphs to and from common data containers
like numpy arrays, scipy sparse arrays, and pandas DataFrames.

The preferred way of converting data to a NetworkX graph is through the
graph constructor.  The constructor calls the `~networkx.convert.to_networkx_graph`
function which attempts to guess the input type and convert it automatically.

Examples
--------
Create a 10 node random graph from a numpy array

>>> import numpy as np
>>> rng = np.random.default_rng()
>>> a = rng.integers(low=0, high=2, size=(10, 10))
>>> DG = nx.from_numpy_array(a, create_using=nx.DiGraph)

or equivalently:

>>> DG = nx.DiGraph(a)

which calls `from_numpy_array` internally based on the type of ``a``.

See Also
--------
nx_agraph, nx_pydot
"""

import itertools
from collections import defaultdict

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "from_pandas_adjacency",
    "to_pandas_adjacency",
    "from_pandas_edgelist",
    "to_pandas_edgelist",
    "from_scipy_sparse_array",
    "to_scipy_sparse_array",
    "from_numpy_array",
    "to_numpy_array",
]


@nx._dispatchable(edge_attrs="weight")
def to_pandas_adjacency(
    G,
    nodelist=None,
    dtype=None,
    order=None,
    multigraph_weight=sum,
    weight="weight",
    nonedge=0.0,
):
    """Returns the graph adjacency matrix as a Pandas DataFrame.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the Pandas DataFrame.

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in `nodelist`.
       If `nodelist` is None, then the ordering is produced by G.nodes().

    multigraph_weight : {sum, min, max}, optional
        An operator that determines how weights in multigraphs are handled.
        The default is to sum the weights of the multiple edges.

    weight : string or None, optional
        The edge attribute that holds the numerical value used for
        the edge weight.  If an edge does not have that attribute, then the
        value 1 is used instead.

    nonedge : float, optional
        The matrix values corresponding to nonedges are typically set to zero.
        However, this could be undesirable if there are matrix values
        corresponding to actual edges that also have the value zero. If so,
        one might prefer nonedges to have some other value, such as nan.

    Returns
    -------
    df : Pandas DataFrame
       Graph adjacency matrix

    Notes
    -----
    For directed graphs, entry i,j corresponds to an edge from i to j.

    The DataFrame entries are assigned to the weight edge attribute. When
    an edge does not have a weight attribute, the value of the entry is set to
    the number 1.  For multiple (parallel) edges, the values of the entries
    are determined by the 'multigraph_weight' parameter.  The default is to
    sum the weight attributes for each of the parallel edges.

    When `nodelist` does not contain every node in `G`, the matrix is built
    from the subgraph of `G` that is induced by the nodes in `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal matrix entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute).  If the
    alternate convention of doubling the edge weight is desired the
    resulting Pandas DataFrame can be modified as follows::

        >>> import pandas as pd
        >>> G = nx.Graph([(1, 1), (2, 2)])
        >>> df = nx.to_pandas_adjacency(G)
        >>> df
             1    2
        1  1.0  0.0
        2  0.0  1.0
        >>> diag_idx = list(range(len(df)))
        >>> df.iloc[diag_idx, diag_idx] *= 2
        >>> df
             1    2
        1  2.0  0.0
        2  0.0  2.0

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, weight=2)
    0
    >>> G.add_edge(1, 0)
    0
    >>> G.add_edge(2, 2, weight=3)
    0
    >>> G.add_edge(2, 2)
    1
    >>> nx.to_pandas_adjacency(G, nodelist=[0, 1, 2], dtype=int)
       0  1  2
    0  0  2  0
    1  1  0  0
    2  0  0  4

    """
    import pandas as pd

    M = to_numpy_array(
        G,
        nodelist=nodelist,
        dtype=dtype,
        order=order,
        multigraph_weight=multigraph_weight,
        weight=weight,
        nonedge=nonedge,
    )
    if nodelist is None:
        nodelist = list(G)
    return pd.DataFrame(data=M, index=nodelist, columns=nodelist)


@nx._dispatchable(graphs=None, returns_graph=True)
def from_pandas_adjacency(df, create_using=None):
    r"""Returns a graph from Pandas DataFrame.

    The Pandas DataFrame is interpreted as an adjacency matrix for the graph.

    Parameters
    ----------
    df : Pandas DataFrame
      An adjacency matrix representation of a graph

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of df corresponds to an edge from i to j.

    If `df` has a single data type for each entry it will be converted to an
    appropriate Python data type.

    If you have node attributes stored in a separate dataframe `df_nodes`,
    you can load those attributes to the graph `G` using the following code:

    ```
    df_nodes = pd.DataFrame({"node_id": [1, 2, 3], "attribute1": ["A", "B", "C"]})
    G.add_nodes_from((n, dict(d)) for n, d in df_nodes.iterrows())
    ```

    If `df` has a user-specified compound data type the names
    of the data fields will be used as attribute keys in the resulting
    NetworkX graph.

    See Also
    --------
    to_pandas_adjacency

    Examples
    --------
    Simple integer weights on edges:

    >>> import pandas as pd
    >>> pd.options.display.max_columns = 20
    >>> df = pd.DataFrame([[1, 1], [2, 1]])
    >>> df
       0  1
    0  1  1
    1  2  1
    >>> G = nx.from_pandas_adjacency(df)
    >>> G.name = "Graph from pandas adjacency matrix"
    >>> print(G)
    Graph named 'Graph from pandas adjacency matrix' with 2 nodes and 3 edges
    """

    try:
        df = df[df.index]
    except Exception as err:
        missing = list(set(df.index).difference(set(df.columns)))
        msg = f"{missing} not in columns"
        raise nx.NetworkXError("Columns must match Indices.", msg) from err

    A = df.values
    G = from_numpy_array(A, create_using=create_using, nodelist=df.columns)

    return G


@nx._dispatchable(preserve_edge_attrs=True)
def to_pandas_edgelist(
    G,
    source="source",
    target="target",
    nodelist=None,
    dtype=None,
    edge_key=None,
):
    """Returns the graph edge list as a Pandas DataFrame.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the Pandas DataFrame.

    source : str or int, optional
        A valid column name (string or integer) for the source nodes (for the
        directed case).

    target : str or int, optional
        A valid column name (string or integer) for the target nodes (for the
        directed case).

    nodelist : list, optional
       Use only nodes specified in nodelist

    dtype : dtype, default None
        Use to create the DataFrame. Data type to force.
        Only a single dtype is allowed. If None, infer.

    edge_key : str or int or None, optional (default=None)
        A valid column name (string or integer) for the edge keys (for the
        multigraph case). If None, edge keys are not stored in the DataFrame.

    Returns
    -------
    df : Pandas DataFrame
       Graph edge list

    Examples
    --------
    >>> G = nx.Graph(
    ...     [
    ...         ("A", "B", {"cost": 1, "weight": 7}),
    ...         ("C", "E", {"cost": 9, "weight": 10}),
    ...     ]
    ... )
    >>> df = nx.to_pandas_edgelist(G, nodelist=["A", "C"])
    >>> df[["source", "target", "cost", "weight"]]
      source target  cost  weight
    0      A      B     1       7
    1      C      E     9      10

    >>> G = nx.MultiGraph([("A", "B", {"cost": 1}), ("A", "B", {"cost": 9})])
    >>> df = nx.to_pandas_edgelist(G, nodelist=["A", "C"], edge_key="ekey")
    >>> df[["source", "target", "cost", "ekey"]]
      source target  cost  ekey
    0      A      B     1     0
    1      A      B     9     1

    """
    import pandas as pd

    if nodelist is None:
        edgelist = G.edges(data=True)
    else:
        edgelist = G.edges(nodelist, data=True)
    source_nodes = [s for s, _, _ in edgelist]
    target_nodes = [t for _, t, _ in edgelist]

    all_attrs = set().union(*(d.keys() for _, _, d in edgelist))
    if source in all_attrs:
        raise nx.NetworkXError(f"Source name {source!r} is an edge attr name")
    if target in all_attrs:
        raise nx.NetworkXError(f"Target name {target!r} is an edge attr name")

    nan = float("nan")
    edge_attr = {k: [d.get(k, nan) for _, _, d in edgelist] for k in all_attrs}

    if G.is_multigraph() and edge_key is not None:
        if edge_key in all_attrs:
            raise nx.NetworkXError(f"Edge key name {edge_key!r} is an edge attr name")
        edge_keys = [k for _, _, k in G.edges(keys=True)]
        edgelistdict = {source: source_nodes, target: target_nodes, edge_key: edge_keys}
    else:
        edgelistdict = {source: source_nodes, target: target_nodes}

    edgelistdict.update(edge_attr)
    return pd.DataFrame(edgelistdict, dtype=dtype)


@nx._dispatchable(graphs=None, returns_graph=True)
def from_pandas_edgelist(
    df,
    source="source",
    target="target",
    edge_attr=None,
    create_using=None,
    edge_key=None,
):
    """Returns a graph from Pandas DataFrame containing an edge list.

    The Pandas DataFrame should contain at least two columns of node names and
    zero or more columns of edge attributes. Each row will be processed as one
    edge instance.

    Note: This function iterates over DataFrame.values, which is not
    guaranteed to retain the data type across columns in the row. This is only
    a problem if your row is entirely numeric and a mix of ints and floats. In
    that case, all values will be returned as floats. See the
    DataFrame.iterrows documentation for an example.

    Parameters
    ----------
    df : Pandas DataFrame
        An edge list representation of a graph

    source : str or int
        A valid column name (string or integer) for the source nodes (for the
        directed case).

    target : str or int
        A valid column name (string or integer) for the target nodes (for the
        directed case).

    edge_attr : str or int, iterable, True, or None
        A valid column name (str or int) or iterable of column names that are
        used to retrieve items and add them to the graph as edge attributes.
        If `True`, all columns will be added except `source`, `target` and `edge_key`.
        If `None`, no edge attributes are added to the graph.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    edge_key : str or None, optional (default=None)
        A valid column name for the edge keys (for a MultiGraph). The values in
        this column are used for the edge keys when adding edges if create_using
        is a multigraph.

    If you have node attributes stored in a separate dataframe `df_nodes`,
    you can load those attributes to the graph `G` using the following code:

    ```
    df_nodes = pd.DataFrame({"node_id": [1, 2, 3], "attribute1": ["A", "B", "C"]})
    G.add_nodes_from((n, dict(d)) for n, d in df_nodes.iterrows())
    ```

    See Also
    --------
    to_pandas_edgelist

    Examples
    --------
    Simple integer weights on edges:

    >>> import pandas as pd
    >>> pd.options.display.max_columns = 20
    >>> import numpy as np
    >>> rng = np.random.RandomState(seed=5)
    >>> ints = rng.randint(1, 11, size=(3, 2))
    >>> a = ["A", "B", "C"]
    >>> b = ["D", "A", "E"]
    >>> df = pd.DataFrame(ints, columns=["weight", "cost"])
    >>> df[0] = a
    >>> df["b"] = b
    >>> df[["weight", "cost", 0, "b"]]
       weight  cost  0  b
    0       4     7  A  D
    1       7     1  B  A
    2      10     9  C  E
    >>> G = nx.from_pandas_edgelist(df, 0, "b", ["weight", "cost"])
    >>> G["E"]["C"]["weight"]
    10
    >>> G["E"]["C"]["cost"]
    9
    >>> edges = pd.DataFrame(
    ...     {
    ...         "source": [0, 1, 2],
    ...         "target": [2, 2, 3],
    ...         "weight": [3, 4, 5],
    ...         "color": ["red", "blue", "blue"],
    ...     }
    ... )
    >>> G = nx.from_pandas_edgelist(edges, edge_attr=True)
    >>> G[0][2]["color"]
    'red'

    Build multigraph with custom keys:

    >>> edges = pd.DataFrame(
    ...     {
    ...         "source": [0, 1, 2, 0],
    ...         "target": [2, 2, 3, 2],
    ...         "my_edge_key": ["A", "B", "C", "D"],
    ...         "weight": [3, 4, 5, 6],
    ...         "color": ["red", "blue", "blue", "blue"],
    ...     }
    ... )
    >>> G = nx.from_pandas_edgelist(
    ...     edges,
    ...     edge_key="my_edge_key",
    ...     edge_attr=["weight", "color"],
    ...     create_using=nx.MultiGraph(),
    ... )
    >>> G[0][2]
    AtlasView({'A': {'weight': 3, 'color': 'red'}, 'D': {'weight': 6, 'color': 'blue'}})


    """
    g = nx.empty_graph(0, create_using)

    if edge_attr is None:
        if g.is_multigraph() and edge_key is not None:
            for u, v, k in zip(df[source], df[target], df[edge_key]):
                g.add_edge(u, v, k)
        else:
            g.add_edges_from(zip(df[source], df[target]))
        return g

    reserved_columns = [source, target]
    if g.is_multigraph() and edge_key is not None:
        reserved_columns.append(edge_key)

    # Additional columns requested
    attr_col_headings = []
    attribute_data = []
    if edge_attr is True:
        attr_col_headings = [c for c in df.columns if c not in reserved_columns]
    elif isinstance(edge_attr, list | tuple):
        attr_col_headings = edge_attr
    else:
        attr_col_headings = [edge_attr]
    if len(attr_col_headings) == 0:
        raise nx.NetworkXError(
            f"Invalid edge_attr argument: No columns found with name: {attr_col_headings}"
        )

    try:
        attribute_data = zip(*[df[col] for col in attr_col_headings])
    except (KeyError, TypeError) as err:
        msg = f"Invalid edge_attr argument: {edge_attr}"
        raise nx.NetworkXError(msg) from err

    if g.is_multigraph():
        # => append the edge keys from the df to the bundled data
        if edge_key is not None:
            try:
                multigraph_edge_keys = df[edge_key]
                attribute_data = zip(attribute_data, multigraph_edge_keys)
            except (KeyError, TypeError) as err:
                msg = f"Invalid edge_key argument: {edge_key}"
                raise nx.NetworkXError(msg) from err

        for s, t, attrs in zip(df[source], df[target], attribute_data):
            if edge_key is not None:
                attrs, multigraph_edge_key = attrs
                key = g.add_edge(s, t, key=multigraph_edge_key)
            else:
                key = g.add_edge(s, t)

            g[s][t][key].update(zip(attr_col_headings, attrs))
    else:
        for s, t, attrs in zip(df[source], df[target], attribute_data):
            g.add_edge(s, t)
            g[s][t].update(zip(attr_col_headings, attrs))

    return g


@nx._dispatchable(edge_attrs="weight")
def to_scipy_sparse_array(G, nodelist=None, dtype=None, weight="weight", format="csr"):
    """Returns the graph adjacency matrix as a SciPy sparse array.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the sparse array.

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in `nodelist`.
       If `nodelist` is None, then the ordering is produced by ``G.nodes()``.

    dtype : NumPy data-type, optional
        A valid NumPy dtype used to initialize the array. If None, then the
        NumPy default is used.

    weight : string or None, optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None then all edge weights are 1.

    format : str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}
        The format of the sparse array to be returned (default 'csr').  For
        some algorithms different implementations of sparse arrays
        can perform better.  See [1]_ for details.

    Returns
    -------
    A : SciPy sparse array
       Graph adjacency matrix.

    Notes
    -----
    For directed graphs, matrix entry ``i, j`` corresponds to an edge from
    ``i`` to ``j``.

    The values of the adjacency matrix are populated using the edge attribute held in
    parameter `weight`. When an edge does not have that attribute, the
    value of the entry is 1.

    For multiple edges the matrix values are the sums of the edge weights.

    When `nodelist` does not contain every node in `G`, the adjacency matrix
    is built from the subgraph of `G` that is induced by the nodes in
    `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal matrix entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute).  If the
    alternate convention of doubling the edge weight is desired the
    resulting array can be modified as follows::

        >>> G = nx.Graph([(1, 1)])
        >>> A = nx.to_scipy_sparse_array(G)
        >>> A.toarray()
        array([[1]])
        >>> A.setdiag(A.diagonal() * 2)
        >>> A.toarray()
        array([[2]])

    Examples
    --------

    Basic usage:

    >>> G = nx.path_graph(4)
    >>> A = nx.to_scipy_sparse_array(G)
    >>> A  # doctest: +SKIP
    <Compressed Sparse Row sparse array of dtype 'int64'
        with 6 stored elements and shape (4, 4)>

    >>> A.toarray()
    array([[0, 1, 0, 0],
           [1, 0, 1, 0],
           [0, 1, 0, 1],
           [0, 0, 1, 0]])

    .. note:: The `toarray` method is used in these examples to better visualize
       the adjacancy matrix. For a dense representation of the adjaceny matrix,
       use `to_numpy_array` instead.

    Directed graphs:

    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    >>> nx.to_scipy_sparse_array(G).toarray()
    array([[0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 0]])

    >>> H = G.reverse()
    >>> H.edges
    OutEdgeView([(1, 0), (2, 1), (3, 2)])
    >>> nx.to_scipy_sparse_array(H).toarray()
    array([[0, 0, 0, 0],
           [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0]])

    By default, the order of the rows/columns of the adjacency matrix is determined
    by the ordering of the nodes in `G`:

    >>> G = nx.Graph()
    >>> G.add_nodes_from([3, 5, 0, 1])
    >>> G.add_edges_from([(1, 3), (1, 5)])
    >>> nx.to_scipy_sparse_array(G).toarray()
    array([[0, 0, 0, 1],
           [0, 0, 0, 1],
           [0, 0, 0, 0],
           [1, 1, 0, 0]])

    The ordering of the rows can be changed with `nodelist`:

    >>> ordered = [0, 1, 3, 5]
    >>> nx.to_scipy_sparse_array(G, nodelist=ordered).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 1, 1],
           [0, 1, 0, 0],
           [0, 1, 0, 0]])

    If `nodelist` contains a subset of the nodes in `G`, the adjacency matrix
    for the node-induced subgraph is produced:

    >>> nx.to_scipy_sparse_array(G, nodelist=[1, 3, 5]).toarray()
    array([[0, 1, 1],
           [1, 0, 0],
           [1, 0, 0]])

    The values of the adjacency matrix are drawn from the edge attribute
    specified by the `weight` parameter:

    >>> G = nx.path_graph(4)
    >>> nx.set_edge_attributes(
    ...     G, values={(0, 1): 1, (1, 2): 10, (2, 3): 2}, name="weight"
    ... )
    >>> nx.set_edge_attributes(
    ...     G, values={(0, 1): 50, (1, 2): 35, (2, 3): 10}, name="capacity"
    ... )
    >>> nx.to_scipy_sparse_array(G).toarray()  # Default weight="weight"
    array([[ 0,  1,  0,  0],
           [ 1,  0, 10,  0],
           [ 0, 10,  0,  2],
           [ 0,  0,  2,  0]])
    >>> nx.to_scipy_sparse_array(G, weight="capacity").toarray()
    array([[ 0, 50,  0,  0],
           [50,  0, 35,  0],
           [ 0, 35,  0, 10],
           [ 0,  0, 10,  0]])

    Any edges that don't have a `weight` attribute default to 1:

    >>> G[1][2].pop("capacity")
    35
    >>> nx.to_scipy_sparse_array(G, weight="capacity").toarray()
    array([[ 0, 50,  0,  0],
           [50,  0,  1,  0],
           [ 0,  1,  0, 10],
           [ 0,  0, 10,  0]])

    When `G` is a multigraph, the values in the adjacency matrix are given by
    the sum of the `weight` edge attribute over each edge key:

    >>> G = nx.MultiDiGraph([(0, 1), (0, 1), (0, 1), (2, 0)])
    >>> nx.to_scipy_sparse_array(G).toarray()
    array([[0, 3, 0],
           [0, 0, 0],
           [1, 0, 0]])

    References
    ----------
    .. [1] Scipy Dev. References, "Sparse Arrays",
       https://docs.scipy.org/doc/scipy/reference/sparse.html
    """
    import scipy as sp

    if len(G) == 0:
        raise nx.NetworkXError("Graph has no nodes or edges")

    if nodelist is None:
        nodelist = list(G)
        nlen = len(G)
    else:
        nlen = len(nodelist)
        if nlen == 0:
            raise nx.NetworkXError("nodelist has no nodes")
        nodeset = set(G.nbunch_iter(nodelist))
        if nlen != len(nodeset):
            for n in nodelist:
                if n not in G:
                    raise nx.NetworkXError(f"Node {n} in nodelist is not in G")
            raise nx.NetworkXError("nodelist contains duplicates.")
        if nlen < len(G):
            G = G.subgraph(nodelist)

    index = dict(zip(nodelist, range(nlen)))
    coefficients = zip(
        *((index[u], index[v], wt) for u, v, wt in G.edges(data=weight, default=1))
    )
    try:
        row, col, data = coefficients
    except ValueError:
        # there is no edge in the subgraph
        row, col, data = [], [], []

    if G.is_directed():
        A = sp.sparse.coo_array((data, (row, col)), shape=(nlen, nlen), dtype=dtype)
    else:
        # symmetrize matrix
        d = data + data
        r = row + col
        c = col + row
        # selfloop entries get double counted when symmetrizing
        # so we subtract the data on the diagonal
        selfloops = list(nx.selfloop_edges(G, data=weight, default=1))
        if selfloops:
            diag_index, diag_data = zip(*((index[u], -wt) for u, v, wt in selfloops))
            d += diag_data
            r += diag_index
            c += diag_index
        A = sp.sparse.coo_array((d, (r, c)), shape=(nlen, nlen), dtype=dtype)
    try:
        return A.asformat(format)
    except ValueError as err:
        raise nx.NetworkXError(f"Unknown sparse matrix format: {format}") from err


def _csr_gen_triples(A):
    """Converts a SciPy sparse array in **Compressed Sparse Row** format to
    an iterable of weighted edge triples.

    """
    nrows = A.shape[0]
    indptr, dst_indices, data = A.indptr, A.indices, A.data
    import numpy as np

    src_indices = np.repeat(np.arange(nrows), np.diff(indptr))
    return zip(src_indices.tolist(), dst_indices.tolist(), A.data.tolist())


def _csc_gen_triples(A):
    """Converts a SciPy sparse array in **Compressed Sparse Column** format to
    an iterable of weighted edge triples.

    """
    ncols = A.shape[1]
    indptr, src_indices, data = A.indptr, A.indices, A.data
    import numpy as np

    dst_indices = np.repeat(np.arange(ncols), np.diff(indptr))
    return zip(src_indices.tolist(), dst_indices.tolist(), A.data.tolist())


def _coo_gen_triples(A):
    """Converts a SciPy sparse array in **Coordinate** format to an iterable
    of weighted edge triples.

    """
    return zip(A.row.tolist(), A.col.tolist(), A.data.tolist())


def _dok_gen_triples(A):
    """Converts a SciPy sparse array in **Dictionary of Keys** format to an
    iterable of weighted edge triples.

    """
    for (r, c), v in A.items():
        # Use `v.item()` to convert a NumPy scalar to the appropriate Python scalar
        yield int(r), int(c), v.item()


def _generate_weighted_edges(A):
    """Returns an iterable over (u, v, w) triples, where u and v are adjacent
    vertices and w is the weight of the edge joining u and v.

    `A` is a SciPy sparse array (in any format).

    """
    if A.format == "csr":
        return _csr_gen_triples(A)
    if A.format == "csc":
        return _csc_gen_triples(A)
    if A.format == "dok":
        return _dok_gen_triples(A)
    # If A is in any other format (including COO), convert it to COO format.
    return _coo_gen_triples(A.tocoo())


@nx._dispatchable(graphs=None, returns_graph=True)
def from_scipy_sparse_array(
    A, parallel_edges=False, create_using=None, edge_attribute="weight"
):
    """Creates a new graph from an adjacency matrix given as a SciPy sparse
    array.

    Parameters
    ----------
    A: scipy.sparse array
      An adjacency matrix representation of a graph

    parallel_edges : Boolean
      If this is True, `create_using` is a multigraph, and `A` is an
      integer matrix, then entry *(i, j)* in the matrix is interpreted as the
      number of parallel edges joining vertices *i* and *j* in the graph.
      If it is False, then the entries in the matrix are interpreted as
      the weight of a single edge joining the vertices.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    edge_attribute: string
       Name of edge attribute to store matrix numeric value. The data will
       have the same type as the matrix entry (int, float, (real,imag)).

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of A corresponds to an edge from i to j.

    If `create_using` is :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph`, `parallel_edges` is True, and the
    entries of `A` are of type :class:`int`, then this function returns a
    multigraph (constructed from `create_using`) with parallel edges.
    In this case, `edge_attribute` will be ignored.

    If `create_using` indicates an undirected multigraph, then only the edges
    indicated by the upper triangle of the matrix `A` will be added to the
    graph.

    Examples
    --------
    >>> import scipy as sp
    >>> A = sp.sparse.eye(2, 2, 1)
    >>> G = nx.from_scipy_sparse_array(A)

    If `create_using` indicates a multigraph and the matrix has only integer
    entries and `parallel_edges` is False, then the entries will be treated
    as weights for edges joining the nodes (without creating parallel edges):

    >>> A = sp.sparse.csr_array([[1, 1], [1, 2]])
    >>> G = nx.from_scipy_sparse_array(A, create_using=nx.MultiGraph)
    >>> G[1][1]
    AtlasView({0: {'weight': 2}})

    If `create_using` indicates a multigraph and the matrix has only integer
    entries and `parallel_edges` is True, then the entries will be treated
    as the number of parallel edges joining those two vertices:

    >>> A = sp.sparse.csr_array([[1, 1], [1, 2]])
    >>> G = nx.from_scipy_sparse_array(
    ...     A, parallel_edges=True, create_using=nx.MultiGraph
    ... )
    >>> G[1][1]
    AtlasView({0: {'weight': 1}, 1: {'weight': 1}})

    """
    G = nx.empty_graph(0, create_using)
    n, m = A.shape
    if n != m:
        raise nx.NetworkXError(f"Adjacency matrix not square: nx,ny={A.shape}")
    # Make sure we get even the isolated nodes of the graph.
    G.add_nodes_from(range(n))
    # Create an iterable over (u, v, w) triples and for each triple, add an
    # edge from u to v with weight w.
    triples = _generate_weighted_edges(A)
    # If the entries in the adjacency matrix are integers, the graph is a
    # multigraph, and parallel_edges is True, then create parallel edges, each
    # with weight 1, for each entry in the adjacency matrix. Otherwise, create
    # one edge for each positive entry in the adjacency matrix and set the
    # weight of that edge to be the entry in the matrix.
    if A.dtype.kind in ("i", "u") and G.is_multigraph() and parallel_edges:
        chain = itertools.chain.from_iterable
        # The following line is equivalent to:
        #
        #     for (u, v) in edges:
        #         for d in range(A[u, v]):
        #             G.add_edge(u, v, weight=1)
        #
        triples = chain(((u, v, 1) for d in range(w)) for (u, v, w) in triples)
    # If we are creating an undirected multigraph, only add the edges from the
    # upper triangle of the matrix. Otherwise, add all the edges. This relies
    # on the fact that the vertices created in the
    # `_generated_weighted_edges()` function are actually the row/column
    # indices for the matrix `A`.
    #
    # Without this check, we run into a problem where each edge is added twice
    # when `G.add_weighted_edges_from()` is invoked below.
    if G.is_multigraph() and not G.is_directed():
        triples = ((u, v, d) for u, v, d in triples if u <= v)
    G.add_weighted_edges_from(triples, weight=edge_attribute)
    return G


@nx._dispatchable(edge_attrs="weight")  # edge attrs may also be obtained from `dtype`
def to_numpy_array(
    G,
    nodelist=None,
    dtype=None,
    order=None,
    multigraph_weight=sum,
    weight="weight",
    nonedge=0.0,
):
    """Returns the graph adjacency matrix as a NumPy array.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the NumPy array.

    nodelist : list, optional
        The rows and columns are ordered according to the nodes in `nodelist`.
        If `nodelist` is ``None``, then the ordering is produced by ``G.nodes()``.

    dtype : NumPy data type, optional
        A NumPy data type used to initialize the array. If None, then the NumPy
        default is used. The dtype can be structured if `weight=None`, in which
        case the dtype field names are used to look up edge attributes. The
        result is a structured array where each named field in the dtype
        corresponds to the adjacency for that edge attribute. See examples for
        details.

    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. If None, then the NumPy default
        is used.

    multigraph_weight : callable, optional
        An function that determines how weights in multigraphs are handled.
        The function should accept a sequence of weights and return a single
        value. The default is to sum the weights of the multiple edges.

    weight : string or None optional (default = 'weight')
        The edge attribute that holds the numerical value used for
        the edge weight. If an edge does not have that attribute, then the
        value 1 is used instead. `weight` must be ``None`` if a structured
        dtype is used.

    nonedge : array_like (default = 0.0)
        The value used to represent non-edges in the adjacency matrix.
        The array values corresponding to nonedges are typically set to zero.
        However, this could be undesirable if there are array values
        corresponding to actual edges that also have the value zero. If so,
        one might prefer nonedges to have some other value, such as ``nan``.

    Returns
    -------
    A : NumPy ndarray
        Graph adjacency matrix

    Raises
    ------
    NetworkXError
        If `dtype` is a structured dtype and `G` is a multigraph
    ValueError
        If `dtype` is a structured dtype and `weight` is not `None`

    See Also
    --------
    from_numpy_array

    Notes
    -----
    For directed graphs, entry ``i, j`` corresponds to an edge from ``i`` to ``j``.

    Entries in the adjacency matrix are given by the `weight` edge attribute.
    When an edge does not have a weight attribute, the value of the entry is
    set to the number 1.  For multiple (parallel) edges, the values of the
    entries are determined by the `multigraph_weight` parameter. The default is
    to sum the weight attributes for each of the parallel edges.

    When `nodelist` does not contain every node in `G`, the adjacency matrix is
    built from the subgraph of `G` that is induced by the nodes in `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal array entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute). If the
    alternate convention of doubling the edge weight is desired the
    resulting NumPy array can be modified as follows:

    >>> import numpy as np
    >>> G = nx.Graph([(1, 1)])
    >>> A = nx.to_numpy_array(G)
    >>> A
    array([[1.]])
    >>> A[np.diag_indices_from(A)] *= 2
    >>> A
    array([[2.]])

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, weight=2)
    0
    >>> G.add_edge(1, 0)
    0
    >>> G.add_edge(2, 2, weight=3)
    0
    >>> G.add_edge(2, 2)
    1
    >>> nx.to_numpy_array(G, nodelist=[0, 1, 2])
    array([[0., 2., 0.],
           [1., 0., 0.],
           [0., 0., 4.]])

    When `nodelist` argument is used, nodes of `G` which do not appear in the `nodelist`
    and their edges are not included in the adjacency matrix. Here is an example:

    >>> G = nx.Graph()
    >>> G.add_edge(3, 1)
    >>> G.add_edge(2, 0)
    >>> G.add_edge(2, 1)
    >>> G.add_edge(3, 0)
    >>> nx.to_numpy_array(G, nodelist=[1, 2, 3])
    array([[0., 1., 1.],
           [1., 0., 0.],
           [1., 0., 0.]])

    This function can also be used to create adjacency matrices for multiple
    edge attributes with structured dtypes:

    >>> G = nx.Graph()
    >>> G.add_edge(0, 1, weight=10)
    >>> G.add_edge(1, 2, cost=5)
    >>> G.add_edge(2, 3, weight=3, cost=-4.0)
    >>> dtype = np.dtype([("weight", int), ("cost", float)])
    >>> A = nx.to_numpy_array(G, dtype=dtype, weight=None)
    >>> A["weight"]
    array([[ 0, 10,  0,  0],
           [10,  0,  1,  0],
           [ 0,  1,  0,  3],
           [ 0,  0,  3,  0]])
    >>> A["cost"]
    array([[ 0.,  1.,  0.,  0.],
           [ 1.,  0.,  5.,  0.],
           [ 0.,  5.,  0., -4.],
           [ 0.,  0., -4.,  0.]])

    As stated above, the argument "nonedge" is useful especially when there are
    actually edges with weight 0 in the graph. Setting a nonedge value different than 0,
    makes it much clearer to differentiate such 0-weighted edges and actual nonedge values.

    >>> G = nx.Graph()
    >>> G.add_edge(3, 1, weight=2)
    >>> G.add_edge(2, 0, weight=0)
    >>> G.add_edge(2, 1, weight=0)
    >>> G.add_edge(3, 0, weight=1)
    >>> nx.to_numpy_array(G, nonedge=-1.0)
    array([[-1.,  2., -1.,  1.],
           [ 2., -1.,  0., -1.],
           [-1.,  0., -1.,  0.],
           [ 1., -1.,  0., -1.]])
    """
    import numpy as np

    if nodelist is None:
        nodelist = list(G)
    nlen = len(nodelist)

    # Input validation
    nodeset = set(nodelist)
    if nodeset - set(G):
        raise nx.NetworkXError(f"Nodes {nodeset - set(G)} in nodelist is not in G")
    if len(nodeset) < nlen:
        raise nx.NetworkXError("nodelist contains duplicates.")

    A = np.full((nlen, nlen), fill_value=nonedge, dtype=dtype, order=order)

    # Corner cases: empty nodelist or graph without any edges
    if nlen == 0 or G.number_of_edges() == 0:
        return A

    # If dtype is structured and weight is None, use dtype field names as
    # edge attributes
    edge_attrs = None  # Only single edge attribute by default
    if A.dtype.names:
        if weight is None:
            edge_attrs = dtype.names
        else:
            raise ValueError(
                "Specifying `weight` not supported for structured dtypes\n."
                "To create adjacency matrices from structured dtypes, use `weight=None`."
            )

    # Map nodes to row/col in matrix
    idx = dict(zip(nodelist, range(nlen)))
    if len(nodelist) < len(G):
        G = G.subgraph(nodelist).copy()

    # Collect all edge weights and reduce with `multigraph_weights`
    if G.is_multigraph():
        if edge_attrs:
            raise nx.NetworkXError(
                "Structured arrays are not supported for MultiGraphs"
            )
        d = defaultdict(list)
        for u, v, wt in G.edges(data=weight, default=1.0):
            d[(idx[u], idx[v])].append(wt)
        i, j = np.array(list(d.keys())).T  # indices
        wts = [multigraph_weight(ws) for ws in d.values()]  # reduced weights
    else:
        i, j, wts = [], [], []

        # Special branch: multi-attr adjacency from structured dtypes
        if edge_attrs:
            # Extract edges with all data
            for u, v, data in G.edges(data=True):
                i.append(idx[u])
                j.append(idx[v])
                wts.append(data)
            # Map each attribute to the appropriate named field in the
            # structured dtype
            for attr in edge_attrs:
                attr_data = [wt.get(attr, 1.0) for wt in wts]
                A[attr][i, j] = attr_data
                if not G.is_directed():
                    A[attr][j, i] = attr_data
            return A

        for u, v, wt in G.edges(data=weight, default=1.0):
            i.append(idx[u])
            j.append(idx[v])
            wts.append(wt)

    # Set array values with advanced indexing
    A[i, j] = wts
    if not G.is_directed():
        A[j, i] = wts

    return A


@nx._dispatchable(graphs=None, returns_graph=True)
def from_numpy_array(
    A, parallel_edges=False, create_using=None, edge_attr="weight", *, nodelist=None
):
    """Returns a graph from a 2D NumPy array.

    The 2D NumPy array is interpreted as an adjacency matrix for the graph.

    Parameters
    ----------
    A : a 2D numpy.ndarray
        An adjacency matrix representation of a graph

    parallel_edges : Boolean
        If this is True, `create_using` is a multigraph, and `A` is an
        integer array, then entry *(i, j)* in the array is interpreted as the
        number of parallel edges joining vertices *i* and *j* in the graph.
        If it is False, then the entries in the array are interpreted as
        the weight of a single edge joining the vertices.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    edge_attr : String, optional (default="weight")
        The attribute to which the array values are assigned on each edge. If
        it is None, edge attributes will not be assigned.

    nodelist : sequence of nodes, optional
        A sequence of objects to use as the nodes in the graph. If provided, the
        list of nodes must be the same length as the dimensions of `A`. The
        default is `None`, in which case the nodes are drawn from ``range(n)``.

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of A corresponds to an edge from i to j.

    If `create_using` is :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph`, `parallel_edges` is True, and the
    entries of `A` are of type :class:`int`, then this function returns a
    multigraph (of the same type as `create_using`) with parallel edges.

    If `create_using` indicates an undirected multigraph, then only the edges
    indicated by the upper triangle of the array `A` will be added to the
    graph.

    If `edge_attr` is Falsy (False or None), edge attributes will not be
    assigned, and the array data will be treated like a binary mask of
    edge presence or absence. Otherwise, the attributes will be assigned
    as follows:

    If the NumPy array has a single data type for each array entry it
    will be converted to an appropriate Python data type.

    If the NumPy array has a user-specified compound data type the names
    of the data fields will be used as attribute keys in the resulting
    NetworkX graph.

    See Also
    --------
    to_numpy_array

    Examples
    --------
    Simple integer weights on edges:

    >>> import numpy as np
    >>> A = np.array([[1, 1], [2, 1]])
    >>> G = nx.from_numpy_array(A)
    >>> G.edges(data=True)
    EdgeDataView([(0, 0, {'weight': 1}), (0, 1, {'weight': 2}), (1, 1, {'weight': 1})])

    If `create_using` indicates a multigraph and the array has only integer
    entries and `parallel_edges` is False, then the entries will be treated
    as weights for edges joining the nodes (without creating parallel edges):

    >>> A = np.array([[1, 1], [1, 2]])
    >>> G = nx.from_numpy_array(A, create_using=nx.MultiGraph)
    >>> G[1][1]
    AtlasView({0: {'weight': 2}})

    If `create_using` indicates a multigraph and the array has only integer
    entries and `parallel_edges` is True, then the entries will be treated
    as the number of parallel edges joining those two vertices:

    >>> A = np.array([[1, 1], [1, 2]])
    >>> temp = nx.MultiGraph()
    >>> G = nx.from_numpy_array(A, parallel_edges=True, create_using=temp)
    >>> G[1][1]
    AtlasView({0: {'weight': 1}, 1: {'weight': 1}})

    User defined compound data type on edges:

    >>> dt = [("weight", float), ("cost", int)]
    >>> A = np.array([[(1.0, 2)]], dtype=dt)
    >>> G = nx.from_numpy_array(A)
    >>> G.edges()
    EdgeView([(0, 0)])
    >>> G[0][0]["cost"]
    2
    >>> G[0][0]["weight"]
    1.0

    """
    kind_to_python_type = {
        "f": float,
        "i": int,
        "u": int,
        "b": bool,
        "c": complex,
        "S": str,
        "U": str,
        "V": "void",
    }
    G = nx.empty_graph(0, create_using)
    if A.ndim != 2:
        raise nx.NetworkXError(f"Input array must be 2D, not {A.ndim}")
    n, m = A.shape
    if n != m:
        raise nx.NetworkXError(f"Adjacency matrix not square: nx,ny={A.shape}")
    dt = A.dtype
    try:
        python_type = kind_to_python_type[dt.kind]
    except Exception as err:
        raise TypeError(f"Unknown numpy data type: {dt}") from err
    if _default_nodes := (nodelist is None):
        nodelist = range(n)
    else:
        if len(nodelist) != n:
            raise ValueError("nodelist must have the same length as A.shape[0]")

    # Make sure we get even the isolated nodes of the graph.
    G.add_nodes_from(nodelist)
    # Get a list of all the entries in the array with nonzero entries. These
    # coordinates become edges in the graph. (convert to int from np.int64)
    edges = ((int(e[0]), int(e[1])) for e in zip(*A.nonzero()))
    # handle numpy constructed data type
    if python_type == "void":
        # Sort the fields by their offset, then by dtype, then by name.
        fields = sorted(
            (offset, dtype, name) for name, (dtype, offset) in A.dtype.fields.items()
        )
        triples = (
            (
                u,
                v,
                {}
                if edge_attr in [False, None]
                else {
                    name: kind_to_python_type[dtype.kind](val)
                    for (_, dtype, name), val in zip(fields, A[u, v])
                },
            )
            for u, v in edges
        )
    # If the entries in the adjacency matrix are integers, the graph is a
    # multigraph, and parallel_edges is True, then create parallel edges, each
    # with weight 1, for each entry in the adjacency matrix. Otherwise, create
    # one edge for each positive entry in the adjacency matrix and set the
    # weight of that edge to be the entry in the matrix.
    elif python_type is int and G.is_multigraph() and parallel_edges:
        chain = itertools.chain.from_iterable
        # The following line is equivalent to:
        #
        #     for (u, v) in edges:
        #         for d in range(A[u, v]):
        #             G.add_edge(u, v, weight=1)
        #
        if edge_attr in [False, None]:
            triples = chain(((u, v, {}) for d in range(A[u, v])) for (u, v) in edges)
        else:
            triples = chain(
                ((u, v, {edge_attr: 1}) for d in range(A[u, v])) for (u, v) in edges
            )
    else:  # basic data type
        if edge_attr in [False, None]:
            triples = ((u, v, {}) for u, v in edges)
        else:
            triples = ((u, v, {edge_attr: python_type(A[u, v])}) for u, v in edges)
    # If we are creating an undirected multigraph, only add the edges from the
    # upper triangle of the matrix. Otherwise, add all the edges. This relies
    # on the fact that the vertices created in the
    # `_generated_weighted_edges()` function are actually the row/column
    # indices for the matrix `A`.
    #
    # Without this check, we run into a problem where each edge is added twice
    # when `G.add_edges_from()` is invoked below.
    if G.is_multigraph() and not G.is_directed():
        triples = ((u, v, d) for u, v, d in triples if u <= v)
    # Remap nodes if user provided custom `nodelist`
    if not _default_nodes:
        idx_to_node = dict(enumerate(nodelist))
        triples = ((idx_to_node[u], idx_to_node[v], d) for u, v, d in triples)
    G.add_edges_from(triples)
    return G
<|endoftext|>"""Functions to convert NetworkX graphs to and from other formats.

The preferred way of converting data to a NetworkX graph is through the
graph constructor.  The constructor calls the to_networkx_graph() function
which attempts to guess the input type and convert it automatically.

Examples
--------
Create a graph with a single edge from a dictionary of dictionaries

>>> d = {0: {1: 1}}  # dict-of-dicts single edge (0,1)
>>> G = nx.Graph(d)

See Also
--------
nx_agraph, nx_pydot
"""

import warnings
from collections.abc import Collection, Generator, Iterator

import networkx as nx

__all__ = [
    "to_networkx_graph",
    "from_dict_of_dicts",
    "to_dict_of_dicts",
    "from_dict_of_lists",
    "to_dict_of_lists",
    "from_edgelist",
    "to_edgelist",
]


def to_networkx_graph(data, create_using=None, multigraph_input=False):
    """Make a NetworkX graph from a known data structure.

    The preferred way to call this is automatically
    from the class constructor

    >>> d = {0: {1: {"weight": 1}}}  # dict-of-dicts single edge (0,1)
    >>> G = nx.Graph(d)

    instead of the equivalent

    >>> G = nx.from_dict_of_dicts(d)

    Parameters
    ----------
    data : object to be converted

        Current known types are:
         any NetworkX graph
         dict-of-dicts
         dict-of-lists
         container (e.g. set, list, tuple) of edges
         iterator (e.g. itertools.chain) that produces edges
         generator of edges
         Pandas DataFrame (row per edge)
         2D numpy array
         scipy sparse array
         pygraphviz agraph

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    multigraph_input : bool (default False)
        If True and  data is a dict_of_dicts,
        try to create a multigraph assuming dict_of_dict_of_lists.
        If data and create_using are both multigraphs then create
        a multigraph from a multigraph.

    """
    # NX graph
    if hasattr(data, "adj"):
        try:
            result = from_dict_of_dicts(
                data.adj,
                create_using=create_using,
                multigraph_input=data.is_multigraph(),
            )
            # data.graph should be dict-like
            result.graph.update(data.graph)
            # data.nodes should be dict-like
            # result.add_node_from(data.nodes.items()) possible but
            # for custom node_attr_dict_factory which may be hashable
            # will be unexpected behavior
            for n, dd in data.nodes.items():
                result._node[n].update(dd)
            return result
        except Exception as err:
            raise nx.NetworkXError("Input is not a correct NetworkX graph.") from err

    # dict of dicts/lists
    if isinstance(data, dict):
        try:
            return from_dict_of_dicts(
                data, create_using=create_using, multigraph_input=multigraph_input
            )
        except Exception as err1:
            if multigraph_input is True:
                raise nx.NetworkXError(
                    f"converting multigraph_input raised:\n{type(err1)}: {err1}"
                )
            try:
                return from_dict_of_lists(data, create_using=create_using)
            except Exception as err2:
                raise TypeError("Input is not known type.") from err2

    # edgelists
    if isinstance(data, list | tuple | nx.reportviews.EdgeViewABC | Iterator):
        try:
            return from_edgelist(data, create_using=create_using)
        except:
            pass

    # pygraphviz  agraph
    if hasattr(data, "is_strict"):
        try:
            return nx.nx_agraph.from_agraph(data, create_using=create_using)
        except Exception as err:
            raise nx.NetworkXError("Input is not a correct pygraphviz graph.") from err

    # Pandas DataFrame
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            if data.shape[0] == data.shape[1]:
                try:
                    return nx.from_pandas_adjacency(data, create_using=create_using)
                except Exception as err:
                    msg = "Input is not a correct Pandas DataFrame adjacency matrix."
                    raise nx.NetworkXError(msg) from err
            else:
                try:
                    return nx.from_pandas_edgelist(
                        data, edge_attr=True, create_using=create_using
                    )
                except Exception as err:
                    msg = "Input is not a correct Pandas DataFrame edge-list."
                    raise nx.NetworkXError(msg) from err
    except ImportError:
        pass

    # numpy array
    try:
        import numpy as np

        if isinstance(data, np.ndarray):
            try:
                return nx.from_numpy_array(data, create_using=create_using)
            except Exception as err:
                raise nx.NetworkXError(
                    f"Failed to interpret array as an adjacency matrix."
                ) from err
    except ImportError:
        pass

    # scipy sparse array - any format
    try:
        import scipy

        if hasattr(data, "format"):
            try:
                return nx.from_scipy_sparse_array(data, create_using=create_using)
            except Exception as err:
                raise nx.NetworkXError(
                    "Input is not a correct scipy sparse array type."
                ) from err
    except ImportError:
        pass

    # Note: most general check - should remain last in order of execution
    # Includes containers (e.g. list, set, dict, etc.), generators, and
    # iterators (e.g. itertools.chain) of edges

    if isinstance(data, Collection | Generator | Iterator):
        try:
            return from_edgelist(data, create_using=create_using)
        except Exception as err:
            raise nx.NetworkXError("Input is not a valid edge list") from err

    raise nx.NetworkXError("Input is not a known data type for conversion.")


@nx._dispatchable
def to_dict_of_lists(G, nodelist=None):
    """Returns adjacency representation of graph as a dictionary of lists.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list
       Use only nodes specified in nodelist

    Notes
    -----
    Completely ignores edge data for MultiGraph and MultiDiGraph.

    """
    if nodelist is None:
        nodelist = G

    d = {}
    for n in nodelist:
        d[n] = [nbr for nbr in G.neighbors(n) if nbr in nodelist]
    return d


@nx._dispatchable(graphs=None, returns_graph=True)
def from_dict_of_lists(d, create_using=None):
    """Returns a graph from a dictionary of lists.

    Parameters
    ----------
    d : dictionary of lists
      A dictionary of lists adjacency representation.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Examples
    --------
    >>> dol = {0: [1]}  # single edge (0,1)
    >>> G = nx.from_dict_of_lists(dol)

    or

    >>> G = nx.Graph(dol)  # use Graph constructor

    """
    G = nx.empty_graph(0, create_using)
    G.add_nodes_from(d)
    if G.is_multigraph() and not G.is_directed():
        # a dict_of_lists can't show multiedges.  BUT for undirected graphs,
        # each edge shows up twice in the dict_of_lists.
        # So we need to treat this case separately.
        seen = {}
        for node, nbrlist in d.items():
            for nbr in nbrlist:
                if nbr not in seen:
                    G.add_edge(node, nbr)
            seen[node] = 1  # don't allow reverse edge to show up
    else:
        G.add_edges_from(
            ((node, nbr) for node, nbrlist in d.items() for nbr in nbrlist)
        )
    return G


def to_dict_of_dicts(G, nodelist=None, edge_data=None):
    """Returns adjacency representation of graph as a dictionary of dictionaries.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list
       Use only nodes specified in nodelist

    edge_data : scalar, optional
       If provided, the value of the dictionary will be set to `edge_data` for
       all edges. Usual values could be `1` or `True`. If `edge_data` is
       `None` (the default), the edgedata in `G` is used, resulting in a
       dict-of-dict-of-dicts. If `G` is a MultiGraph, the result will be a
       dict-of-dict-of-dict-of-dicts. See Notes for an approach to customize
       handling edge data. `edge_data` should *not* be a container.

    Returns
    -------
    dod : dict
       A nested dictionary representation of `G`. Note that the level of
       nesting depends on the type of `G` and the value of `edge_data`
       (see Examples).

    See Also
    --------
    from_dict_of_dicts, to_dict_of_lists

    Notes
    -----
    For a more custom approach to handling edge data, try::

        dod = {
            n: {nbr: custom(n, nbr, dd) for nbr, dd in nbrdict.items()}
            for n, nbrdict in G.adj.items()
        }

    where `custom` returns the desired edge data for each edge between `n` and
    `nbr`, given existing edge data `dd`.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> nx.to_dict_of_dicts(G)
    {0: {1: {}}, 1: {0: {}, 2: {}}, 2: {1: {}}}

    Edge data is preserved by default (``edge_data=None``), resulting
    in dict-of-dict-of-dicts where the innermost dictionary contains the
    edge data:

    >>> G = nx.Graph()
    >>> G.add_edges_from(
    ...     [
    ...         (0, 1, {"weight": 1.0}),
    ...         (1, 2, {"weight": 2.0}),
    ...         (2, 0, {"weight": 1.0}),
    ...     ]
    ... )
    >>> d = nx.to_dict_of_dicts(G)
    >>> d  # doctest: +SKIP
    {0: {1: {'weight': 1.0}, 2: {'weight': 1.0}},
     1: {0: {'weight': 1.0}, 2: {'weight': 2.0}},
     2: {1: {'weight': 2.0}, 0: {'weight': 1.0}}}
    >>> d[1][2]["weight"]
    2.0

    If `edge_data` is not `None`, edge data in the original graph (if any) is
    replaced:

    >>> d = nx.to_dict_of_dicts(G, edge_data=1)
    >>> d
    {0: {1: 1, 2: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 0: 1}}
    >>> d[1][2]
    1

    This also applies to MultiGraphs: edge data is preserved by default:

    >>> G = nx.MultiGraph()
    >>> G.add_edge(0, 1, key="a", weight=1.0)
    'a'
    >>> G.add_edge(0, 1, key="b", weight=5.0)
    'b'
    >>> d = nx.to_dict_of_dicts(G)
    >>> d  # doctest: +SKIP
    {0: {1: {'a': {'weight': 1.0}, 'b': {'weight': 5.0}}},
     1: {0: {'a': {'weight': 1.0}, 'b': {'weight': 5.0}}}}
    >>> d[0][1]["b"]["weight"]
    5.0

    But multi edge data is lost if `edge_data` is not `None`:

    >>> d = nx.to_dict_of_dicts(G, edge_data=10)
    >>> d
    {0: {1: 10}, 1: {0: 10}}
    """
    dod = {}
    if nodelist is None:
        if edge_data is None:
            for u, nbrdict in G.adjacency():
                dod[u] = nbrdict.copy()
        else:  # edge_data is not None
            for u, nbrdict in G.adjacency():
                dod[u] = dod.fromkeys(nbrdict, edge_data)
    else:  # nodelist is not None
        if edge_data is None:
            for u in nodelist:
                dod[u] = {}
                for v, data in ((v, data) for v, data in G[u].items() if v in nodelist):
                    dod[u][v] = data
        else:  # nodelist and edge_data are not None
            for u in nodelist:
                dod[u] = {}
                for v in (v for v in G[u] if v in nodelist):
                    dod[u][v] = edge_data
    return dod


@nx._dispatchable(graphs=None, returns_graph=True)
def from_dict_of_dicts(d, create_using=None, multigraph_input=False):
    """Returns a graph from a dictionary of dictionaries.

    Parameters
    ----------
    d : dictionary of dictionaries
      A dictionary of dictionaries adjacency representation.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    multigraph_input : bool (default False)
       When True, the dict `d` is assumed
       to be a dict-of-dict-of-dict-of-dict structure keyed by
       node to neighbor to edge keys to edge data for multi-edges.
       Otherwise this routine assumes dict-of-dict-of-dict keyed by
       node to neighbor to edge data.

    Examples
    --------
    >>> dod = {0: {1: {"weight": 1}}}  # single edge (0,1)
    >>> G = nx.from_dict_of_dicts(dod)

    or

    >>> G = nx.Graph(dod)  # use Graph constructor

    """
    G = nx.empty_graph(0, create_using)
    G.add_nodes_from(d)
    # does dict d represent a MultiGraph or MultiDiGraph?
    if multigraph_input:
        if G.is_directed():
            if G.is_multigraph():
                G.add_edges_from(
                    (u, v, key, data)
                    for u, nbrs in d.items()
                    for v, datadict in nbrs.items()
                    for key, data in datadict.items()
                )
            else:
                G.add_edges_from(
                    (u, v, data)
                    for u, nbrs in d.items()
                    for v, datadict in nbrs.items()
                    for key, data in datadict.items()
                )
        else:  # Undirected
            if G.is_multigraph():
                seen = set()  # don't add both directions of undirected graph
                for u, nbrs in d.items():
                    for v, datadict in nbrs.items():
                        if (u, v) not in seen:
                            G.add_edges_from(
                                (u, v, key, data) for key, data in datadict.items()
                            )
                            seen.add((v, u))
            else:
                seen = set()  # don't add both directions of undirected graph
                for u, nbrs in d.items():
                    for v, datadict in nbrs.items():
                        if (u, v) not in seen:
                            G.add_edges_from(
                                (u, v, data) for key, data in datadict.items()
                            )
                            seen.add((v, u))

    else:  # not a multigraph to multigraph transfer
        if G.is_multigraph() and not G.is_directed():
            # d can have both representations u-v, v-u in dict.  Only add one.
            # We don't need this check for digraphs since we add both directions,
            # or for Graph() since it is done implicitly (parallel edges not allowed)
            seen = set()
            for u, nbrs in d.items():
                for v, data in nbrs.items():
                    if (u, v) not in seen:
                        G.add_edge(u, v, key=0)
                        G[u][v][0].update(data)
                    seen.add((v, u))
        else:
            G.add_edges_from(
                ((u, v, data) for u, nbrs in d.items() for v, data in nbrs.items())
            )
    return G


@nx._dispatchable(preserve_edge_attrs=True)
def to_edgelist(G, nodelist=None):
    """Returns a list of edges in the graph.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list
       Use only nodes specified in nodelist

    """
    if nodelist is None:
        return G.edges(data=True)
    return G.edges(nodelist, data=True)


@nx._dispatchable(graphs=None, returns_graph=True)
def from_edgelist(edgelist, create_using=None):
    """Returns a graph from a list of edges.

    Parameters
    ----------
    edgelist : list or iterator
      Edge tuples

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Examples
    --------
    >>> edgelist = [(0, 1)]  # single edge (0,1)
    >>> G = nx.from_edgelist(edgelist)

    or

    >>> G = nx.Graph(edgelist)  # use Graph constructor

    """
    G = nx.empty_graph(0, create_using)
    G.add_edges_from(edgelist)
    return G
<|endoftext|>"""
**********
Exceptions
**********

Base exceptions and errors for NetworkX.
"""

__all__ = [
    "HasACycle",
    "NodeNotFound",
    "PowerIterationFailedConvergence",
    "ExceededMaxIterations",
    "AmbiguousSolution",
    "NetworkXAlgorithmError",
    "NetworkXException",
    "NetworkXError",
    "NetworkXNoCycle",
    "NetworkXNoPath",
    "NetworkXNotImplemented",
    "NetworkXPointlessConcept",
    "NetworkXUnbounded",
    "NetworkXUnfeasible",
]


class NetworkXException(Exception):
    """Base class for exceptions in NetworkX."""


class NetworkXError(NetworkXException):
    """Exception for a serious error in NetworkX"""


class NetworkXPointlessConcept(NetworkXException):
    """Raised when a null graph is provided as input to an algorithm
    that cannot use it.

    The null graph is sometimes considered a pointless concept [1]_,
    thus the name of the exception.

    Notes
    -----
    Null graphs and empty graphs are often used interchangeably but they
    are well defined in NetworkX. An ``empty_graph`` is a graph with ``n`` nodes
    and 0 edges, and a ``null_graph`` is a graph with 0 nodes and 0 edges.

    References
    ----------
    .. [1] Harary, F. and Read, R. "Is the Null Graph a Pointless
       Concept?"  In Graphs and Combinatorics Conference, George
       Washington University.  New York: Springer-Verlag, 1973.

    """


class NetworkXAlgorithmError(NetworkXException):
    """Exception for unexpected termination of algorithms."""


class NetworkXUnfeasible(NetworkXAlgorithmError):
    """Exception raised by algorithms trying to solve a problem
    instance that has no feasible solution."""


class NetworkXNoPath(NetworkXUnfeasible):
    """Exception for algorithms that should return a path when running
    on graphs where such a path does not exist."""


class NetworkXNoCycle(NetworkXUnfeasible):
    """Exception for algorithms that should return a cycle when running
    on graphs where such a cycle does not exist."""


class HasACycle(NetworkXException):
    """Raised if a graph has a cycle when an algorithm expects that it
    will have no cycles.

    """


class NetworkXUnbounded(NetworkXAlgorithmError):
    """Exception raised by algorithms trying to solve a maximization
    or a minimization problem instance that is unbounded."""


class NetworkXNotImplemented(NetworkXException):
    """Exception raised by algorithms not implemented for a type of graph."""


class NodeNotFound(NetworkXException):
    """Exception raised if requested node is not present in the graph"""


class AmbiguousSolution(NetworkXException):
    """Raised if more than one valid solution exists for an intermediary step
    of an algorithm.

    In the face of ambiguity, refuse the temptation to guess.
    This may occur, for example, when trying to determine the
    bipartite node sets in a disconnected bipartite graph when
    computing bipartite matchings.

    """


class ExceededMaxIterations(NetworkXException):
    """Raised if a loop iterates too many times without breaking.

    This may occur, for example, in an algorithm that computes
    progressively better approximations to a value but exceeds an
    iteration bound specified by the user.

    """


class PowerIterationFailedConvergence(ExceededMaxIterations):
    """Raised when the power iteration method fails to converge within a
    specified iteration limit.

    `num_iterations` is the number of iterations that have been
    completed when this exception was raised.

    """

    def __init__(self, num_iterations, *args, **kw):
        msg = f"power iteration failed to converge within {num_iterations} iterations"
        exception_message = msg
        superinit = super().__init__
        superinit(self, exception_message, *args, **kw)
<|endoftext|>"""
NetworkX
========

NetworkX is a Python package for the creation, manipulation, and study of the
structure, dynamics, and functions of complex networks.

See https://networkx.org for complete documentation.
"""

__version__ = "3.4.2"


# These are imported in order as listed
from networkx.lazy_imports import _lazy_import

from networkx.exception import *

from networkx import utils
from networkx.utils import _clear_cache, _dispatchable

# load_and_call entry_points, set configs
config = utils.backends._set_configs_from_environment()
utils.config = utils.configs.config = config  # type: ignore[attr-defined]

from networkx import classes
from networkx.classes import filters
from networkx.classes import *

from networkx import convert
from networkx.convert import *

from networkx import convert_matrix
from networkx.convert_matrix import *

from networkx import relabel
from networkx.relabel import *

from networkx import generators
from networkx.generators import *

from networkx import readwrite
from networkx.readwrite import *

# Need to test with SciPy, when available
from networkx import algorithms
from networkx.algorithms import *

from networkx import linalg
from networkx.linalg import *

from networkx import drawing
from networkx.drawing import *
<|endoftext|>import networkx as nx

__all__ = ["convert_node_labels_to_integers", "relabel_nodes"]


@nx._dispatchable(
    preserve_all_attrs=True, mutates_input={"not copy": 2}, returns_graph=True
)
def relabel_nodes(G, mapping, copy=True):
    """Relabel the nodes of the graph G according to a given mapping.

    The original node ordering may not be preserved if `copy` is `False` and the
    mapping includes overlap between old and new labels.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    mapping : dictionary
       A dictionary with the old labels as keys and new labels as values.
       A partial mapping is allowed. Mapping 2 nodes to a single node is allowed.
       Any non-node keys in the mapping are ignored.

    copy : bool (optional, default=True)
       If True return a copy, or if False relabel the nodes in place.

    Examples
    --------
    To create a new graph with nodes relabeled according to a given
    dictionary:

    >>> G = nx.path_graph(3)
    >>> sorted(G)
    [0, 1, 2]
    >>> mapping = {0: "a", 1: "b", 2: "c"}
    >>> H = nx.relabel_nodes(G, mapping)
    >>> sorted(H)
    ['a', 'b', 'c']

    Nodes can be relabeled with any hashable object, including numbers
    and strings:

    >>> import string
    >>> G = nx.path_graph(26)  # nodes are integers 0 through 25
    >>> sorted(G)[:3]
    [0, 1, 2]
    >>> mapping = dict(zip(G, string.ascii_lowercase))
    >>> G = nx.relabel_nodes(G, mapping)  # nodes are characters a through z
    >>> sorted(G)[:3]
    ['a', 'b', 'c']
    >>> mapping = dict(zip(G, range(1, 27)))
    >>> G = nx.relabel_nodes(G, mapping)  # nodes are integers 1 through 26
    >>> sorted(G)[:3]
    [1, 2, 3]

    To perform a partial in-place relabeling, provide a dictionary
    mapping only a subset of the nodes, and set the `copy` keyword
    argument to False:

    >>> G = nx.path_graph(3)  # nodes 0-1-2
    >>> mapping = {0: "a", 1: "b"}  # 0->'a' and 1->'b'
    >>> G = nx.relabel_nodes(G, mapping, copy=False)
    >>> sorted(G, key=str)
    [2, 'a', 'b']

    A mapping can also be given as a function:

    >>> G = nx.path_graph(3)
    >>> H = nx.relabel_nodes(G, lambda x: x**2)
    >>> list(H)
    [0, 1, 4]

    In a multigraph, relabeling two or more nodes to the same new node
    will retain all edges, but may change the edge keys in the process:

    >>> G = nx.MultiGraph()
    >>> G.add_edge(0, 1, value="a")  # returns the key for this edge
    0
    >>> G.add_edge(0, 2, value="b")
    0
    >>> G.add_edge(0, 3, value="c")
    0
    >>> mapping = {1: 4, 2: 4, 3: 4}
    >>> H = nx.relabel_nodes(G, mapping, copy=True)
    >>> print(H[0])
    {4: {0: {'value': 'a'}, 1: {'value': 'b'}, 2: {'value': 'c'}}}

    This works for in-place relabeling too:

    >>> G = nx.relabel_nodes(G, mapping, copy=False)
    >>> print(G[0])
    {4: {0: {'value': 'a'}, 1: {'value': 'b'}, 2: {'value': 'c'}}}

    Notes
    -----
    Only the nodes specified in the mapping will be relabeled.
    Any non-node keys in the mapping are ignored.

    The keyword setting copy=False modifies the graph in place.
    Relabel_nodes avoids naming collisions by building a
    directed graph from ``mapping`` which specifies the order of
    relabelings. Naming collisions, such as a->b, b->c, are ordered
    such that "b" gets renamed to "c" before "a" gets renamed "b".
    In cases of circular mappings (e.g. a->b, b->a), modifying the
    graph is not possible in-place and an exception is raised.
    In that case, use copy=True.

    If a relabel operation on a multigraph would cause two or more
    edges to have the same source, target and key, the second edge must
    be assigned a new key to retain all edges. The new key is set
    to the lowest non-negative integer not already used as a key
    for edges between these two nodes. Note that this means non-numeric
    keys may be replaced by numeric keys.

    See Also
    --------
    convert_node_labels_to_integers
    """
    # you can pass any callable e.g. f(old_label) -> new_label or
    # e.g. str(old_label) -> new_label, but we'll just make a dictionary here regardless
    m = {n: mapping(n) for n in G} if callable(mapping) else mapping

    if copy:
        return _relabel_copy(G, m)
    else:
        return _relabel_inplace(G, m)


def _relabel_inplace(G, mapping):
    if len(mapping.keys() & mapping.values()) > 0:
        # labels sets overlap
        # can we topological sort and still do the relabeling?
        D = nx.DiGraph(list(mapping.items()))
        D.remove_edges_from(nx.selfloop_edges(D))
        try:
            nodes = reversed(list(nx.topological_sort(D)))
        except nx.NetworkXUnfeasible as err:
            raise nx.NetworkXUnfeasible(
                "The node label sets are overlapping and no ordering can "
                "resolve the mapping. Use copy=True."
            ) from err
    else:
        # non-overlapping label sets, sort them in the order of G nodes
        nodes = [n for n in G if n in mapping]

    multigraph = G.is_multigraph()
    directed = G.is_directed()

    for old in nodes:
        # Test that old is in both mapping and G, otherwise ignore.
        try:
            new = mapping[old]
            G.add_node(new, **G.nodes[old])
        except KeyError:
            continue
        if new == old:
            continue
        if multigraph:
            new_edges = [
                (new, new if old == target else target, key, data)
                for (_, target, key, data) in G.edges(old, data=True, keys=True)
            ]
            if directed:
                new_edges += [
                    (new if old == source else source, new, key, data)
                    for (source, _, key, data) in G.in_edges(old, data=True, keys=True)
                ]
            # Ensure new edges won't overwrite existing ones
            seen = set()
            for i, (source, target, key, data) in enumerate(new_edges):
                if target in G[source] and key in G[source][target]:
                    new_key = 0 if not isinstance(key, int | float) else key
                    while new_key in G[source][target] or (target, new_key) in seen:
                        new_key += 1
                    new_edges[i] = (source, target, new_key, data)
                    seen.add((target, new_key))
        else:
            new_edges = [
                (new, new if old == target else target, data)
                for (_, target, data) in G.edges(old, data=True)
            ]
            if directed:
                new_edges += [
                    (new if old == source else source, new, data)
                    for (source, _, data) in G.in_edges(old, data=True)
                ]
        G.remove_node(old)
        G.add_edges_from(new_edges)
    return G


def _relabel_copy(G, mapping):
    H = G.__class__()
    H.add_nodes_from(mapping.get(n, n) for n in G)
    H._node.update((mapping.get(n, n), d.copy()) for n, d in G.nodes.items())
    if G.is_multigraph():
        new_edges = [
            (mapping.get(n1, n1), mapping.get(n2, n2), k, d.copy())
            for (n1, n2, k, d) in G.edges(keys=True, data=True)
        ]

        # check for conflicting edge-keys
        undirected = not G.is_directed()
        seen_edges = set()
        for i, (source, target, key, data) in enumerate(new_edges):
            while (source, target, key) in seen_edges:
                if not isinstance(key, int | float):
                    key = 0
                key += 1
            seen_edges.add((source, target, key))
            if undirected:
                seen_edges.add((target, source, key))
            new_edges[i] = (source, target, key, data)

        H.add_edges_from(new_edges)
    else:
        H.add_edges_from(
            (mapping.get(n1, n1), mapping.get(n2, n2), d.copy())
            for (n1, n2, d) in G.edges(data=True)
        )
    H.graph.update(G.graph)
    return H


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def convert_node_labels_to_integers(
    G, first_label=0, ordering="default", label_attribute=None
):
    """Returns a copy of the graph G with the nodes relabeled using
    consecutive integers.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    first_label : int, optional (default=0)
       An integer specifying the starting offset in numbering nodes.
       The new integer labels are numbered first_label, ..., n-1+first_label.

    ordering : string
       "default" : inherit node ordering from G.nodes()
       "sorted"  : inherit node ordering from sorted(G.nodes())
       "increasing degree" : nodes are sorted by increasing degree
       "decreasing degree" : nodes are sorted by decreasing degree

    label_attribute : string, optional (default=None)
       Name of node attribute to store old label.  If None no attribute
       is created.

    Notes
    -----
    Node and edge attribute data are copied to the new (relabeled) graph.

    There is no guarantee that the relabeling of nodes to integers will
    give the same two integers for two (even identical graphs).
    Use the `ordering` argument to try to preserve the order.

    See Also
    --------
    relabel_nodes
    """
    N = G.number_of_nodes() + first_label
    if ordering == "default":
        mapping = dict(zip(G.nodes(), range(first_label, N)))
    elif ordering == "sorted":
        nlist = sorted(G.nodes())
        mapping = dict(zip(nlist, range(first_label, N)))
    elif ordering == "increasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    elif ordering == "decreasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        dv_pairs.reverse()
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    else:
        raise nx.NetworkXError(f"Unknown node ordering: {ordering}")
    H = relabel_nodes(G, mapping)
    # create node attribute with the old label
    if label_attribute is not None:
        nx.set_node_attributes(H, {v: k for k, v in mapping.items()}, label_attribute)
    return H
<|endoftext|>import importlib
import importlib.util
import inspect
import os
import sys
import types

__all__ = ["attach", "_lazy_import"]


def attach(module_name, submodules=None, submod_attrs=None):
    """Attach lazily loaded submodules, and functions or other attributes.

    Typically, modules import submodules and attributes as follows::

      import mysubmodule
      import anothersubmodule

      from .foo import someattr

    The idea of  this function is to replace the `__init__.py`
    module's `__getattr__`, `__dir__`, and `__all__` attributes such that
    all imports work exactly the way they normally would, except that the
    actual import is delayed until the resulting module object is first used.

    The typical way to call this function, replacing the above imports, is::

      __getattr__, __lazy_dir__, __all__ = lazy.attach(
          __name__, ["mysubmodule", "anothersubmodule"], {"foo": "someattr"}
      )

    This functionality requires Python 3.7 or higher.

    Parameters
    ----------
    module_name : str
        Typically use __name__.
    submodules : set
        List of submodules to lazily import.
    submod_attrs : dict
        Dictionary of submodule -> list of attributes / functions.
        These attributes are imported as they are used.

    Returns
    -------
    __getattr__, __dir__, __all__

    """
    if submod_attrs is None:
        submod_attrs = {}

    if submodules is None:
        submodules = set()
    else:
        submodules = set(submodules)

    attr_to_modules = {
        attr: mod for mod, attrs in submod_attrs.items() for attr in attrs
    }

    __all__ = list(submodules | attr_to_modules.keys())

    def __getattr__(name):
        if name in submodules:
            return importlib.import_module(f"{module_name}.{name}")
        elif name in attr_to_modules:
            submod = importlib.import_module(f"{module_name}.{attr_to_modules[name]}")
            return getattr(submod, name)
        else:
            raise AttributeError(f"No {module_name} attribute {name}")

    def __dir__():
        return __all__

    if os.environ.get("EAGER_IMPORT", ""):
        for attr in set(attr_to_modules.keys()) | submodules:
            __getattr__(attr)

    return __getattr__, __dir__, list(__all__)


class DelayedImportErrorModule(types.ModuleType):
    def __init__(self, frame_data, *args, **kwargs):
        self.__frame_data = frame_data
        super().__init__(*args, **kwargs)

    def __getattr__(self, x):
        if x in ("__class__", "__file__", "__frame_data"):
            super().__getattr__(x)
        else:
            fd = self.__frame_data
            raise ModuleNotFoundError(
                f"No module named '{fd['spec']}'\n\n"
                "This error is lazily reported, having originally occurred in\n"
                f'  File {fd["filename"]}, line {fd["lineno"]}, in {fd["function"]}\n\n'
                f'----> {"".join(fd["code_context"] or "").strip()}'
            )


def _lazy_import(fullname):
    """Return a lazily imported proxy for a module or library.

    Warning
    -------
    Importing using this function can currently cause trouble
    when the user tries to import from a subpackage of a module before
    the package is fully imported. In particular, this idiom may not work:

      np = lazy_import("numpy")
      from numpy.lib import recfunctions

    This is due to a difference in the way Python's LazyLoader handles
    subpackage imports compared to the normal import process. Hopefully
    we will get Python's LazyLoader to fix this, or find a workaround.
    In the meantime, this is a potential problem.

    The workaround is to import numpy before importing from the subpackage.

    Notes
    -----
    We often see the following pattern::

      def myfunc():
          import scipy as sp
          sp.argmin(...)
          ....

    This is to prevent a library, in this case `scipy`, from being
    imported at function definition time, since that can be slow.

    This function provides a proxy module that, upon access, imports
    the actual module.  So the idiom equivalent to the above example is::

      sp = lazy.load("scipy")

      def myfunc():
          sp.argmin(...)
          ....

    The initial import time is fast because the actual import is delayed
    until the first attribute is requested. The overall import time may
    decrease as well for users that don't make use of large portions
    of the library.

    Parameters
    ----------
    fullname : str
        The full name of the package or subpackage to import.  For example::

          sp = lazy.load("scipy")  # import scipy as sp
          spla = lazy.load("scipy.linalg")  # import scipy.linalg as spla

    Returns
    -------
    pm : importlib.util._LazyModule
        Proxy module. Can be used like any regularly imported module.
        Actual loading of the module occurs upon first attribute request.

    """
    try:
        return sys.modules[fullname]
    except:
        pass

    # Not previously loaded -- look it up
    spec = importlib.util.find_spec(fullname)

    if spec is None:
        try:
            parent = inspect.stack()[1]
            frame_data = {
                "spec": fullname,
                "filename": parent.filename,
                "lineno": parent.lineno,
                "function": parent.function,
                "code_context": parent.code_context,
            }
            return DelayedImportErrorModule(frame_data, "DelayedImportErrorModule")
        finally:
            del parent

    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module

    loader = importlib.util.LazyLoader(spec.loader)
    loader.exec_module(module)

    return module
<|endoftext|>"""Generators for geometric graphs."""

import math
from bisect import bisect_left
from itertools import accumulate, combinations, product

import networkx as nx
from networkx.utils import py_random_state

__all__ = [
    "geometric_edges",
    "geographical_threshold_graph",
    "navigable_small_world_graph",
    "random_geometric_graph",
    "soft_random_geometric_graph",
    "thresholded_random_geometric_graph",
    "waxman_graph",
    "geometric_soft_configuration_graph",
]


@nx._dispatchable(node_attrs="pos_name")
def geometric_edges(G, radius, p=2, *, pos_name="pos"):
    """Returns edge list of node pairs within `radius` of each other.

    Parameters
    ----------
    G : networkx graph
        The graph from which to generate the edge list. The nodes in `G` should
        have an attribute ``pos`` corresponding to the node position, which is
        used to compute the distance to other nodes.
    radius : scalar
        The distance threshold. Edges are included in the edge list if the
        distance between the two nodes is less than `radius`.
    pos_name : string, default="pos"
        The name of the node attribute which represents the position of each
        node in 2D coordinates. Every node in the Graph must have this attribute.
    p : scalar, default=2
        The `Minkowski distance metric
        <https://en.wikipedia.org/wiki/Minkowski_distance>`_ used to compute
        distances. The default value is 2, i.e. Euclidean distance.

    Returns
    -------
    edges : list
        List of edges whose distances are less than `radius`

    Notes
    -----
    Radius uses Minkowski distance metric `p`.
    If scipy is available, `scipy.spatial.cKDTree` is used to speed computation.

    Examples
    --------
    Create a graph with nodes that have a "pos" attribute representing 2D
    coordinates.

    >>> G = nx.Graph()
    >>> G.add_nodes_from(
    ...     [
    ...         (0, {"pos": (0, 0)}),
    ...         (1, {"pos": (3, 0)}),
    ...         (2, {"pos": (8, 0)}),
    ...     ]
    ... )
    >>> nx.geometric_edges(G, radius=1)
    []
    >>> nx.geometric_edges(G, radius=4)
    [(0, 1)]
    >>> nx.geometric_edges(G, radius=6)
    [(0, 1), (1, 2)]
    >>> nx.geometric_edges(G, radius=9)
    [(0, 1), (0, 2), (1, 2)]
    """
    # Input validation - every node must have a "pos" attribute
    for n, pos in G.nodes(data=pos_name):
        if pos is None:
            raise nx.NetworkXError(
                f"Node {n} (and all nodes) must have a '{pos_name}' attribute."
            )

    # NOTE: See _geometric_edges for the actual implementation. The reason this
    # is split into two functions is to avoid the overhead of input validation
    # every time the function is called internally in one of the other
    # geometric generators
    return _geometric_edges(G, radius, p, pos_name)


def _geometric_edges(G, radius, p, pos_name):
    """
    Implements `geometric_edges` without input validation. See `geometric_edges`
    for complete docstring.
    """
    nodes_pos = G.nodes(data=pos_name)
    try:
        import scipy as sp
    except ImportError:
        # no scipy KDTree so compute by for-loop
        radius_p = radius**p
        edges = [
            (u, v)
            for (u, pu), (v, pv) in combinations(nodes_pos, 2)
            if sum(abs(a - b) ** p for a, b in zip(pu, pv)) <= radius_p
        ]
        return edges
    # scipy KDTree is available
    nodes, coords = list(zip(*nodes_pos))
    kdtree = sp.spatial.cKDTree(coords)  # Cannot provide generator.
    edge_indexes = kdtree.query_pairs(radius, p)
    edges = [(nodes[u], nodes[v]) for u, v in sorted(edge_indexes)]
    return edges


@py_random_state(5)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_geometric_graph(
    n, radius, dim=2, pos=None, p=2, seed=None, *, pos_name="pos"
):
    """Returns a random geometric graph in the unit cube of dimensions `dim`.

    The random geometric graph model places `n` nodes uniformly at
    random in the unit cube. Two nodes are joined by an edge if the
    distance between the nodes is at most `radius`.

    Edges are determined using a KDTree when SciPy is available.
    This reduces the time complexity from $O(n^2)$ to $O(n)$.

    Parameters
    ----------
    n : int or iterable
        Number of nodes or iterable of nodes
    radius: float
        Distance threshold value
    dim : int, optional
        Dimension of graph
    pos : dict, optional
        A dictionary keyed by node with node positions as values.
    p : float, optional
        Which Minkowski distance metric to use.  `p` has to meet the condition
        ``1 <= p <= infinity``.

        If this argument is not specified, the :math:`L^2` metric
        (the Euclidean distance metric), p = 2 is used.
        This should not be confused with the `p` of an Erdős-Rényi random
        graph, which represents probability.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    pos_name : string, default="pos"
        The name of the node attribute which represents the position
        in 2D coordinates of the node in the returned graph.

    Returns
    -------
    Graph
        A random geometric graph, undirected and without self-loops.
        Each node has a node attribute ``'pos'`` that stores the
        position of that node in Euclidean space as provided by the
        ``pos`` keyword argument or, if ``pos`` was not provided, as
        generated by this function.

    Examples
    --------
    Create a random geometric graph on twenty nodes where nodes are joined by
    an edge if their distance is at most 0.1::

    >>> G = nx.random_geometric_graph(20, 0.1)

    Notes
    -----
    This uses a *k*-d tree to build the graph.

    The `pos` keyword argument can be used to specify node positions so you
    can create an arbitrary distribution and domain for positions.

    For example, to use a 2D Gaussian distribution of node positions with mean
    (0, 0) and standard deviation 2::

    >>> import random
    >>> n = 20
    >>> pos = {i: (random.gauss(0, 2), random.gauss(0, 2)) for i in range(n)}
    >>> G = nx.random_geometric_graph(n, 0.2, pos=pos)

    References
    ----------
    .. [1] Penrose, Mathew, *Random Geometric Graphs*,
           Oxford Studies in Probability, 5, 2003.

    """
    # TODO Is this function just a special case of the geographical
    # threshold graph?
    #
    #     half_radius = {v: radius / 2 for v in n}
    #     return geographical_threshold_graph(nodes, theta=1, alpha=1,
    #                                         weight=half_radius)
    #
    G = nx.empty_graph(n)
    # If no positions are provided, choose uniformly random vectors in
    # Euclidean space of the specified dimension.
    if pos is None:
        pos = {v: [seed.random() for i in range(dim)] for v in G}
    nx.set_node_attributes(G, pos, pos_name)

    G.add_edges_from(_geometric_edges(G, radius, p, pos_name))
    return G


@py_random_state(6)
@nx._dispatchable(graphs=None, returns_graph=True)
def soft_random_geometric_graph(
    n, radius, dim=2, pos=None, p=2, p_dist=None, seed=None, *, pos_name="pos"
):
    r"""Returns a soft random geometric graph in the unit cube.

    The soft random geometric graph [1] model places `n` nodes uniformly at
    random in the unit cube in dimension `dim`. Two nodes of distance, `dist`,
    computed by the `p`-Minkowski distance metric are joined by an edge with
    probability `p_dist` if the computed distance metric value of the nodes
    is at most `radius`, otherwise they are not joined.

    Edges within `radius` of each other are determined using a KDTree when
    SciPy is available. This reduces the time complexity from :math:`O(n^2)`
    to :math:`O(n)`.

    Parameters
    ----------
    n : int or iterable
        Number of nodes or iterable of nodes
    radius: float
        Distance threshold value
    dim : int, optional
        Dimension of graph
    pos : dict, optional
        A dictionary keyed by node with node positions as values.
    p : float, optional
        Which Minkowski distance metric to use.
        `p` has to meet the condition ``1 <= p <= infinity``.

        If this argument is not specified, the :math:`L^2` metric
        (the Euclidean distance metric), p = 2 is used.

        This should not be confused with the `p` of an Erdős-Rényi random
        graph, which represents probability.
    p_dist : function, optional
        A probability density function computing the probability of
        connecting two nodes that are of distance, dist, computed by the
        Minkowski distance metric. The probability density function, `p_dist`,
        must be any function that takes the metric value as input
        and outputs a single probability value between 0-1. The scipy.stats
        package has many probability distribution functions implemented and
        tools for custom probability distribution definitions [2], and passing
        the .pdf method of scipy.stats distributions can be used here.  If the
        probability function, `p_dist`, is not supplied, the default function
        is an exponential distribution with rate parameter :math:`\lambda=1`.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    pos_name : string, default="pos"
        The name of the node attribute which represents the position
        in 2D coordinates of the node in the returned graph.

    Returns
    -------
    Graph
        A soft random geometric graph, undirected and without self-loops.
        Each node has a node attribute ``'pos'`` that stores the
        position of that node in Euclidean space as provided by the
        ``pos`` keyword argument or, if ``pos`` was not provided, as
        generated by this function.

    Examples
    --------
    Default Graph:

    G = nx.soft_random_geometric_graph(50, 0.2)

    Custom Graph:

    Create a soft random geometric graph on 100 uniformly distributed nodes
    where nodes are joined by an edge with probability computed from an
    exponential distribution with rate parameter :math:`\lambda=1` if their
    Euclidean distance is at most 0.2.

    Notes
    -----
    This uses a *k*-d tree to build the graph.

    The `pos` keyword argument can be used to specify node positions so you
    can create an arbitrary distribution and domain for positions.

    For example, to use a 2D Gaussian distribution of node positions with mean
    (0, 0) and standard deviation 2

    The scipy.stats package can be used to define the probability distribution
    with the .pdf method used as `p_dist`.

    ::

    >>> import random
    >>> import math
    >>> n = 100
    >>> pos = {i: (random.gauss(0, 2), random.gauss(0, 2)) for i in range(n)}
    >>> p_dist = lambda dist: math.exp(-dist)
    >>> G = nx.soft_random_geometric_graph(n, 0.2, pos=pos, p_dist=p_dist)

    References
    ----------
    .. [1] Penrose, Mathew D. "Connectivity of soft random geometric graphs."
           The Annals of Applied Probability 26.2 (2016): 986-1028.
    .. [2] scipy.stats -
           https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html

    """
    G = nx.empty_graph(n)
    G.name = f"soft_random_geometric_graph({n}, {radius}, {dim})"
    # If no positions are provided, choose uniformly random vectors in
    # Euclidean space of the specified dimension.
    if pos is None:
        pos = {v: [seed.random() for i in range(dim)] for v in G}
    nx.set_node_attributes(G, pos, pos_name)

    # if p_dist function not supplied the default function is an exponential
    # distribution with rate parameter :math:`\lambda=1`.
    if p_dist is None:

        def p_dist(dist):
            return math.exp(-dist)

    def should_join(edge):
        u, v = edge
        dist = (sum(abs(a - b) ** p for a, b in zip(pos[u], pos[v]))) ** (1 / p)
        return seed.random() < p_dist(dist)

    G.add_edges_from(filter(should_join, _geometric_edges(G, radius, p, pos_name)))
    return G


@py_random_state(7)
@nx._dispatchable(graphs=None, returns_graph=True)
def geographical_threshold_graph(
    n,
    theta,
    dim=2,
    pos=None,
    weight=None,
    metric=None,
    p_dist=None,
    seed=None,
    *,
    pos_name="pos",
    weight_name="weight",
):
    r"""Returns a geographical threshold graph.

    The geographical threshold graph model places $n$ nodes uniformly at
    random in a rectangular domain.  Each node $u$ is assigned a weight
    $w_u$. Two nodes $u$ and $v$ are joined by an edge if

    .. math::

       (w_u + w_v)p_{dist}(r) \ge \theta

    where `r` is the distance between `u` and `v`, `p_dist` is any function of
    `r`, and :math:`\theta` as the threshold parameter. `p_dist` is used to
    give weight to the distance between nodes when deciding whether or not
    they should be connected. The larger `p_dist` is, the more prone nodes
    separated by `r` are to be connected, and vice versa.

    Parameters
    ----------
    n : int or iterable
        Number of nodes or iterable of nodes
    theta: float
        Threshold value
    dim : int, optional
        Dimension of graph
    pos : dict
        Node positions as a dictionary of tuples keyed by node.
    weight : dict
        Node weights as a dictionary of numbers keyed by node.
    metric : function
        A metric on vectors of numbers (represented as lists or
        tuples). This must be a function that accepts two lists (or
        tuples) as input and yields a number as output. The function
        must also satisfy the four requirements of a `metric`_.
        Specifically, if $d$ is the function and $x$, $y$,
        and $z$ are vectors in the graph, then $d$ must satisfy

        1. $d(x, y) \ge 0$,
        2. $d(x, y) = 0$ if and only if $x = y$,
        3. $d(x, y) = d(y, x)$,
        4. $d(x, z) \le d(x, y) + d(y, z)$.

        If this argument is not specified, the Euclidean distance metric is
        used.

        .. _metric: https://en.wikipedia.org/wiki/Metric_%28mathematics%29
    p_dist : function, optional
        Any function used to give weight to the distance between nodes when
        deciding whether or not they should be connected. `p_dist` was
        originally conceived as a probability density function giving the
        probability of connecting two nodes that are of metric distance `r`
        apart. The implementation here allows for more arbitrary definitions
        of `p_dist` that do not need to correspond to valid probability
        density functions. The :mod:`scipy.stats` package has many
        probability density functions implemented and tools for custom
        probability density definitions, and passing the ``.pdf`` method of
        scipy.stats distributions can be used here. If ``p_dist=None``
        (the default), the exponential function :math:`r^{-2}` is used.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    pos_name : string, default="pos"
        The name of the node attribute which represents the position
        in 2D coordinates of the node in the returned graph.
    weight_name : string, default="weight"
        The name of the node attribute which represents the weight
        of the node in the returned graph.

    Returns
    -------
    Graph
        A random geographic threshold graph, undirected and without
        self-loops.

        Each node has a node attribute ``pos`` that stores the
        position of that node in Euclidean space as provided by the
        ``pos`` keyword argument or, if ``pos`` was not provided, as
        generated by this function. Similarly, each node has a node
        attribute ``weight`` that stores the weight of that node as
        provided or as generated.

    Examples
    --------
    Specify an alternate distance metric using the ``metric`` keyword
    argument. For example, to use the `taxicab metric`_ instead of the
    default `Euclidean metric`_::

        >>> dist = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        >>> G = nx.geographical_threshold_graph(10, 0.1, metric=dist)

    .. _taxicab metric: https://en.wikipedia.org/wiki/Taxicab_geometry
    .. _Euclidean metric: https://en.wikipedia.org/wiki/Euclidean_distance

    Notes
    -----
    If weights are not specified they are assigned to nodes by drawing randomly
    from the exponential distribution with rate parameter $\lambda=1$.
    To specify weights from a different distribution, use the `weight` keyword
    argument::

    >>> import random
    >>> n = 20
    >>> w = {i: random.expovariate(5.0) for i in range(n)}
    >>> G = nx.geographical_threshold_graph(20, 50, weight=w)

    If node positions are not specified they are randomly assigned from the
    uniform distribution.

    References
    ----------
    .. [1] Masuda, N., Miwa, H., Konno, N.:
       Geographical threshold graphs with small-world and scale-free
       properties.
       Physical Review E 71, 036108 (2005)
    .. [2]  Milan Bradonjić, Aric Hagberg and Allon G. Percus,
       Giant component and connectivity in geographical threshold graphs,
       in Algorithms and Models for the Web-Graph (WAW 2007),
       Antony Bonato and Fan Chung (Eds), pp. 209--216, 2007
    """
    G = nx.empty_graph(n)
    # If no weights are provided, choose them from an exponential
    # distribution.
    if weight is None:
        weight = {v: seed.expovariate(1) for v in G}
    # If no positions are provided, choose uniformly random vectors in
    # Euclidean space of the specified dimension.
    if pos is None:
        pos = {v: [seed.random() for i in range(dim)] for v in G}
    # If no distance metric is provided, use Euclidean distance.
    if metric is None:
        metric = math.dist
    nx.set_node_attributes(G, weight, weight_name)
    nx.set_node_attributes(G, pos, pos_name)

    # if p_dist is not supplied, use default r^-2
    if p_dist is None:

        def p_dist(r):
            return r**-2

    # Returns ``True`` if and only if the nodes whose attributes are
    # ``du`` and ``dv`` should be joined, according to the threshold
    # condition.
    def should_join(pair):
        u, v = pair
        u_pos, v_pos = pos[u], pos[v]
        u_weight, v_weight = weight[u], weight[v]
        return (u_weight + v_weight) * p_dist(metric(u_pos, v_pos)) >= theta

    G.add_edges_from(filter(should_join, combinations(G, 2)))
    return G


@py_random_state(6)
@nx._dispatchable(graphs=None, returns_graph=True)
def waxman_graph(
    n,
    beta=0.4,
    alpha=0.1,
    L=None,
    domain=(0, 0, 1, 1),
    metric=None,
    seed=None,
    *,
    pos_name="pos",
):
    r"""Returns a Waxman random graph.

    The Waxman random graph model places `n` nodes uniformly at random
    in a rectangular domain. Each pair of nodes at distance `d` is
    joined by an edge with probability

    .. math::
            p = \beta \exp(-d / \alpha L).

    This function implements both Waxman models, using the `L` keyword
    argument.

    * Waxman-1: if `L` is not specified, it is set to be the maximum distance
      between any pair of nodes.
    * Waxman-2: if `L` is specified, the distance between a pair of nodes is
      chosen uniformly at random from the interval `[0, L]`.

    Parameters
    ----------
    n : int or iterable
        Number of nodes or iterable of nodes
    beta: float
        Model parameter
    alpha: float
        Model parameter
    L : float, optional
        Maximum distance between nodes.  If not specified, the actual distance
        is calculated.
    domain : four-tuple of numbers, optional
        Domain size, given as a tuple of the form `(x_min, y_min, x_max,
        y_max)`.
    metric : function
        A metric on vectors of numbers (represented as lists or
        tuples). This must be a function that accepts two lists (or
        tuples) as input and yields a number as output. The function
        must also satisfy the four requirements of a `metric`_.
        Specifically, if $d$ is the function and $x$, $y$,
        and $z$ are vectors in the graph, then $d$ must satisfy

        1. $d(x, y) \ge 0$,
        2. $d(x, y) = 0$ if and only if $x = y$,
        3. $d(x, y) = d(y, x)$,
        4. $d(x, z) \le d(x, y) + d(y, z)$.

        If this argument is not specified, the Euclidean distance metric is
        used.

        .. _metric: https://en.wikipedia.org/wiki/Metric_%28mathematics%29

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    pos_name : string, default="pos"
        The name of the node attribute which represents the position
        in 2D coordinates of the node in the returned graph.

    Returns
    -------
    Graph
        A random Waxman graph, undirected and without self-loops. Each
        node has a node attribute ``'pos'`` that stores the position of
        that node in Euclidean space as generated by this function.

    Examples
    --------
    Specify an alternate distance metric using the ``metric`` keyword
    argument. For example, to use the "`taxicab metric`_" instead of the
    default `Euclidean metric`_::

        >>> dist = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        >>> G = nx.waxman_graph(10, 0.5, 0.1, metric=dist)

    .. _taxicab metric: https://en.wikipedia.org/wiki/Taxicab_geometry
    .. _Euclidean metric: https://en.wikipedia.org/wiki/Euclidean_distance

    Notes
    -----
    Starting in NetworkX 2.0 the parameters alpha and beta align with their
    usual roles in the probability distribution. In earlier versions their
    positions in the expression were reversed. Their position in the calling
    sequence reversed as well to minimize backward incompatibility.

    References
    ----------
    .. [1]  B. M. Waxman, *Routing of multipoint connections*.
       IEEE J. Select. Areas Commun. 6(9),(1988) 1617--1622.
    """
    G = nx.empty_graph(n)
    (xmin, ymin, xmax, ymax) = domain
    # Each node gets a uniformly random position in the given rectangle.
    pos = {v: (seed.uniform(xmin, xmax), seed.uniform(ymin, ymax)) for v in G}
    nx.set_node_attributes(G, pos, pos_name)
    # If no distance metric is provided, use Euclidean distance.
    if metric is None:
        metric = math.dist
    # If the maximum distance L is not specified (that is, we are in the
    # Waxman-1 model), then find the maximum distance between any pair
    # of nodes.
    #
    # In the Waxman-1 model, join nodes randomly based on distance. In
    # the Waxman-2 model, join randomly based on random l.
    if L is None:
        L = max(metric(x, y) for x, y in combinations(pos.values(), 2))

        def dist(u, v):
            return metric(pos[u], pos[v])

    else:

        def dist(u, v):
            return seed.random() * L

    # `pair` is the pair of nodes to decide whether to join.
    def should_join(pair):
        return seed.random() < beta * math.exp(-dist(*pair) / (alpha * L))

    G.add_edges_from(filter(should_join, combinations(G, 2)))
    return G


@py_random_state(5)
@nx._dispatchable(graphs=None, returns_graph=True)
def navigable_small_world_graph(n, p=1, q=1, r=2, dim=2, seed=None):
    r"""Returns a navigable small-world graph.

    A navigable small-world graph is a directed grid with additional long-range
    connections that are chosen randomly.

      [...] we begin with a set of nodes [...] that are identified with the set
      of lattice points in an $n \times n$ square,
      $\{(i, j): i \in \{1, 2, \ldots, n\}, j \in \{1, 2, \ldots, n\}\}$,
      and we define the *lattice distance* between two nodes $(i, j)$ and
      $(k, l)$ to be the number of "lattice steps" separating them:
      $d((i, j), (k, l)) = |k - i| + |l - j|$.

      For a universal constant $p >= 1$, the node $u$ has a directed edge to
      every other node within lattice distance $p$---these are its *local
      contacts*. For universal constants $q >= 0$ and $r >= 0$ we also
      construct directed edges from $u$ to $q$ other nodes (the *long-range
      contacts*) using independent random trials; the $i$th directed edge from
      $u$ has endpoint $v$ with probability proportional to $[d(u,v)]^{-r}$.

      -- [1]_

    Parameters
    ----------
    n : int
        The length of one side of the lattice; the number of nodes in
        the graph is therefore $n^2$.
    p : int
        The diameter of short range connections. Each node is joined with every
        other node within this lattice distance.
    q : int
        The number of long-range connections for each node.
    r : float
        Exponent for decaying probability of connections.  The probability of
        connecting to a node at lattice distance $d$ is $1/d^r$.
    dim : int
        Dimension of grid
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    References
    ----------
    .. [1] J. Kleinberg. The small-world phenomenon: An algorithmic
       perspective. Proc. 32nd ACM Symposium on Theory of Computing, 2000.
    """
    if p < 1:
        raise nx.NetworkXException("p must be >= 1")
    if q < 0:
        raise nx.NetworkXException("q must be >= 0")
    if r < 0:
        raise nx.NetworkXException("r must be >= 0")

    G = nx.DiGraph()
    nodes = list(product(range(n), repeat=dim))
    for p1 in nodes:
        probs = [0]
        for p2 in nodes:
            if p1 == p2:
                continue
            d = sum((abs(b - a) for a, b in zip(p1, p2)))
            if d <= p:
                G.add_edge(p1, p2)
            probs.append(d**-r)
        cdf = list(accumulate(probs))
        for _ in range(q):
            target = nodes[bisect_left(cdf, seed.uniform(0, cdf[-1]))]
            G.add_edge(p1, target)
    return G


@py_random_state(7)
@nx._dispatchable(graphs=None, returns_graph=True)
def thresholded_random_geometric_graph(
    n,
    radius,
    theta,
    dim=2,
    pos=None,
    weight=None,
    p=2,
    seed=None,
    *,
    pos_name="pos",
    weight_name="weight",
):
    r"""Returns a thresholded random geometric graph in the unit cube.

    The thresholded random geometric graph [1] model places `n` nodes
    uniformly at random in the unit cube of dimensions `dim`. Each node
    `u` is assigned a weight :math:`w_u`. Two nodes `u` and `v` are
    joined by an edge if they are within the maximum connection distance,
    `radius` computed by the `p`-Minkowski distance and the summation of
    weights :math:`w_u` + :math:`w_v` is greater than or equal
    to the threshold parameter `theta`.

    Edges within `radius` of each other are determined using a KDTree when
    SciPy is available. This reduces the time complexity from :math:`O(n^2)`
    to :math:`O(n)`.

    Parameters
    ----------
    n : int or iterable
        Number of nodes or iterable of nodes
    radius: float
        Distance threshold value
    theta: float
        Threshold value
    dim : int, optional
        Dimension of graph
    pos : dict, optional
        A dictionary keyed by node with node positions as values.
    weight : dict, optional
        Node weights as a dictionary of numbers keyed by node.
    p : float, optional (default 2)
        Which Minkowski distance metric to use.  `p` has to meet the condition
        ``1 <= p <= infinity``.

        If this argument is not specified, the :math:`L^2` metric
        (the Euclidean distance metric), p = 2 is used.

        This should not be confused with the `p` of an Erdős-Rényi random
        graph, which represents probability.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    pos_name : string, default="pos"
        The name of the node attribute which represents the position
        in 2D coordinates of the node in the returned graph.
    weight_name : string, default="weight"
        The name of the node attribute which represents the weight
        of the node in the returned graph.

    Returns
    -------
    Graph
        A thresholded random geographic graph, undirected and without
        self-loops.

        Each node has a node attribute ``'pos'`` that stores the
        position of that node in Euclidean space as provided by the
        ``pos`` keyword argument or, if ``pos`` was not provided, as
        generated by this function. Similarly, each node has a nodethre
        attribute ``'weight'`` that stores the weight of that node as
        provided or as generated.

    Examples
    --------
    Default Graph:

    G = nx.thresholded_random_geometric_graph(50, 0.2, 0.1)

    Custom Graph:

    Create a thresholded random geometric graph on 50 uniformly distributed
    nodes where nodes are joined by an edge if their sum weights drawn from
    a exponential distribution with rate = 5 are >= theta = 0.1 and their
    Euclidean distance is at most 0.2.

    Notes
    -----
    This uses a *k*-d tree to build the graph.

    The `pos` keyword argument can be used to specify node positions so you
    can create an arbitrary distribution and domain for positions.

    For example, to use a 2D Gaussian distribution of node positions with mean
    (0, 0) and standard deviation 2

    If weights are not specified they are assigned to nodes by drawing randomly
    from the exponential distribution with rate parameter :math:`\lambda=1`.
    To specify weights from a different distribution, use the `weight` keyword
    argument::

    ::

    >>> import random
    >>> import math
    >>> n = 50
    >>> pos = {i: (random.gauss(0, 2), random.gauss(0, 2)) for i in range(n)}
    >>> w = {i: random.expovariate(5.0) for i in range(n)}
    >>> G = nx.thresholded_random_geometric_graph(n, 0.2, 0.1, 2, pos, w)

    References
    ----------
    .. [1] http://cole-maclean.github.io/blog/files/thesis.pdf

    """
    G = nx.empty_graph(n)
    G.name = f"thresholded_random_geometric_graph({n}, {radius}, {theta}, {dim})"
    # If no weights are provided, choose them from an exponential
    # distribution.
    if weight is None:
        weight = {v: seed.expovariate(1) for v in G}
    # If no positions are provided, choose uniformly random vectors in
    # Euclidean space of the specified dimension.
    if pos is None:
        pos = {v: [seed.random() for i in range(dim)] for v in G}
    # If no distance metric is provided, use Euclidean distance.
    nx.set_node_attributes(G, weight, weight_name)
    nx.set_node_attributes(G, pos, pos_name)

    edges = (
        (u, v)
        for u, v in _geometric_edges(G, radius, p, pos_name)
        if weight[u] + weight[v] >= theta
    )
    G.add_edges_from(edges)
    return G


@py_random_state(5)
@nx._dispatchable(graphs=None, returns_graph=True)
def geometric_soft_configuration_graph(
    *, beta, n=None, gamma=None, mean_degree=None, kappas=None, seed=None
):
    r"""Returns a random graph from the geometric soft configuration model.

    The $\mathbb{S}^1$ model [1]_ is the geometric soft configuration model
    which is able to explain many fundamental features of real networks such as
    small-world property, heteregenous degree distributions, high level of
    clustering, and self-similarity.

    In the geometric soft configuration model, a node $i$ is assigned two hidden
    variables: a hidden degree $\kappa_i$, quantifying its popularity, influence,
    or importance, and an angular position $\theta_i$ in a circle abstracting the
    similarity space, where angular distances between nodes are a proxy for their
    similarity. Focusing on the angular position, this model is often called
    the $\mathbb{S}^1$ model (a one-dimensional sphere). The circle's radius is
    adjusted to $R = N/2\pi$, where $N$ is the number of nodes, so that the density
    is set to 1 without loss of generality.

    The connection probability between any pair of nodes increases with
    the product of their hidden degrees (i.e., their combined popularities),
    and decreases with the angular distance between the two nodes.
    Specifically, nodes $i$ and $j$ are connected with the probability

    $p_{ij} = \frac{1}{1 + \frac{d_{ij}^\beta}{\left(\mu \kappa_i \kappa_j\right)^{\max(1, \beta)}}}$

    where $d_{ij} = R\Delta\theta_{ij}$ is the arc length of the circle between
    nodes $i$ and $j$ separated by an angular distance $\Delta\theta_{ij}$.
    Parameters $\mu$ and $\beta$ (also called inverse temperature) control the
    average degree and the clustering coefficient, respectively.

    It can be shown [2]_ that the model undergoes a structural phase transition
    at $\beta=1$ so that for $\beta<1$ networks are unclustered in the thermodynamic
    limit (when $N\to \infty$) whereas for $\beta>1$ the ensemble generates
    networks with finite clustering coefficient.

    The $\mathbb{S}^1$ model can be expressed as a purely geometric model
    $\mathbb{H}^2$ in the hyperbolic plane [3]_ by mapping the hidden degree of
    each node into a radial coordinate as

    $r_i = \hat{R} - \frac{2 \max(1, \beta)}{\beta \zeta} \ln \left(\frac{\kappa_i}{\kappa_0}\right)$

    where $\hat{R}$ is the radius of the hyperbolic disk and $\zeta$ is the curvature,

    $\hat{R} = \frac{2}{\zeta} \ln \left(\frac{N}{\pi}\right)
    - \frac{2\max(1, \beta)}{\beta \zeta} \ln (\mu \kappa_0^2)$

    The connection probability then reads

    $p_{ij} = \frac{1}{1 + \exp\left({\frac{\beta\zeta}{2} (x_{ij} - \hat{R})}\right)}$

    where

    $x_{ij} = r_i + r_j + \frac{2}{\zeta} \ln \frac{\Delta\theta_{ij}}{2}$

    is a good approximation of the hyperbolic distance between two nodes separated
    by an angular distance $\Delta\theta_{ij}$ with radial coordinates $r_i$ and $r_j$.
    For $\beta > 1$, the curvature $\zeta = 1$, for $\beta < 1$, $\zeta = \beta^{-1}$.


    Parameters
    ----------
    Either `n`, `gamma`, `mean_degree` are provided or `kappas`. The values of
    `n`, `gamma`, `mean_degree` (if provided) are used to construct a random
    kappa-dict keyed by node with values sampled from a power-law distribution.

    beta : positive number
        Inverse temperature, controlling the clustering coefficient.
    n : int (default: None)
        Size of the network (number of nodes).
        If not provided, `kappas` must be provided and holds the nodes.
    gamma : float (default: None)
        Exponent of the power-law distribution for hidden degrees `kappas`.
        If not provided, `kappas` must be provided directly.
    mean_degree : float (default: None)
        The mean degree in the network.
        If not provided, `kappas` must be provided directly.
    kappas : dict (default: None)
        A dict keyed by node to its hidden degree value.
        If not provided, random values are computed based on a power-law
        distribution using `n`, `gamma` and `mean_degree`.
    seed : int, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    Graph
        A random geometric soft configuration graph (undirected with no self-loops).
        Each node has three node-attributes:

        - ``kappa`` that represents the hidden degree.

        - ``theta`` the position in the similarity space ($\mathbb{S}^1$) which is
          also the angular position in the hyperbolic plane.

        - ``radius`` the radial position in the hyperbolic plane
          (based on the hidden degree).


    Examples
    --------
    Generate a network with specified parameters:

    >>> G = nx.geometric_soft_configuration_graph(
    ...     beta=1.5, n=100, gamma=2.7, mean_degree=5
    ... )

    Create a geometric soft configuration graph with 100 nodes. The $\beta$ parameter
    is set to 1.5 and the exponent of the powerlaw distribution of the hidden
    degrees is 2.7 with mean value of 5.

    Generate a network with predefined hidden degrees:

    >>> kappas = {i: 10 for i in range(100)}
    >>> G = nx.geometric_soft_configuration_graph(beta=2.5, kappas=kappas)

    Create a geometric soft configuration graph with 100 nodes. The $\beta$ parameter
    is set to 2.5 and all nodes with hidden degree $\kappa=10$.


    References
    ----------
    .. [1] Serrano, M. Á., Krioukov, D., & Boguñá, M. (2008). Self-similarity
       of complex networks and hidden metric spaces. Physical review letters, 100(7), 078701.

    .. [2] van der Kolk, J., Serrano, M. Á., & Boguñá, M. (2022). An anomalous
       topological phase transition in spatial random graphs. Communications Physics, 5(1), 245.

    .. [3] Krioukov, D., Papadopoulos, F., Kitsak, M., Vahdat, A., & Boguná, M. (2010).
       Hyperbolic geometry of complex networks. Physical Review E, 82(3), 036106.

    """
    if beta <= 0:
        raise nx.NetworkXError("The parameter beta cannot be smaller or equal to 0.")

    if kappas is not None:
        if not all((n is None, gamma is None, mean_degree is None)):
            raise nx.NetworkXError(
                "When kappas is input, n, gamma and mean_degree must not be."
            )

        n = len(kappas)
        mean_degree = sum(kappas) / len(kappas)
    else:
        if any((n is None, gamma is None, mean_degree is None)):
            raise nx.NetworkXError(
                "Please provide either kappas, or all 3 of: n, gamma and mean_degree."
            )

        # Generate `n` hidden degrees from a powerlaw distribution
        # with given exponent `gamma` and mean value `mean_degree`
        gam_ratio = (gamma - 2) / (gamma - 1)
        kappa_0 = mean_degree * gam_ratio * (1 - 1 / n) / (1 - 1 / n**gam_ratio)
        base = 1 - 1 / n
        power = 1 / (1 - gamma)
        kappas = {i: kappa_0 * (1 - seed.random() * base) ** power for i in range(n)}

    G = nx.Graph()
    R = n / (2 * math.pi)

    # Approximate values for mu in the thermodynamic limit (when n -> infinity)
    if beta > 1:
        mu = beta * math.sin(math.pi / beta) / (2 * math.pi * mean_degree)
    elif beta == 1:
        mu = 1 / (2 * mean_degree * math.log(n))
    else:
        mu = (1 - beta) / (2**beta * mean_degree * n ** (1 - beta))

    # Generate random positions on a circle
    thetas = {k: seed.uniform(0, 2 * math.pi) for k in kappas}

    for u in kappas:
        for v in list(G):
            angle = math.pi - math.fabs(math.pi - math.fabs(thetas[u] - thetas[v]))
            dij = math.pow(R * angle, beta)
            mu_kappas = math.pow(mu * kappas[u] * kappas[v], max(1, beta))
            p_ij = 1 / (1 + dij / mu_kappas)

            # Create an edge with a certain connection probability
            if seed.random() < p_ij:
                G.add_edge(u, v)
        G.add_node(u)

    nx.set_node_attributes(G, thetas, "theta")
    nx.set_node_attributes(G, kappas, "kappa")

    # Map hidden degrees into the radial coordinates
    zeta = 1 if beta > 1 else 1 / beta
    kappa_min = min(kappas.values())
    R_c = 2 * max(1, beta) / (beta * zeta)
    R_hat = (2 / zeta) * math.log(n / math.pi) - R_c * math.log(mu * kappa_min)
    radii = {node: R_hat - R_c * math.log(kappa) for node, kappa in kappas.items()}
    nx.set_node_attributes(G, radii, "radius")

    return G
<|endoftext|>"""Provides explicit constructions of expander graphs."""

import itertools

import networkx as nx

__all__ = [
    "margulis_gabber_galil_graph",
    "chordal_cycle_graph",
    "paley_graph",
    "maybe_regular_expander",
    "is_regular_expander",
    "random_regular_expander_graph",
]


# Other discrete torus expanders can be constructed by using the following edge
# sets. For more information, see Chapter 4, "Expander Graphs", in
# "Pseudorandomness", by Salil Vadhan.
#
# For a directed expander, add edges from (x, y) to:
#
#     (x, y),
#     ((x + 1) % n, y),
#     (x, (y + 1) % n),
#     (x, (x + y) % n),
#     (-y % n, x)
#
# For an undirected expander, add the reverse edges.
#
# Also appearing in the paper of Gabber and Galil:
#
#     (x, y),
#     (x, (x + y) % n),
#     (x, (x + y + 1) % n),
#     ((x + y) % n, y),
#     ((x + y + 1) % n, y)
#
# and:
#
#     (x, y),
#     ((x + 2*y) % n, y),
#     ((x + (2*y + 1)) % n, y),
#     ((x + (2*y + 2)) % n, y),
#     (x, (y + 2*x) % n),
#     (x, (y + (2*x + 1)) % n),
#     (x, (y + (2*x + 2)) % n),
#
@nx._dispatchable(graphs=None, returns_graph=True)
def margulis_gabber_galil_graph(n, create_using=None):
    r"""Returns the Margulis-Gabber-Galil undirected MultiGraph on `n^2` nodes.

    The undirected MultiGraph is regular with degree `8`. Nodes are integer
    pairs. The second-largest eigenvalue of the adjacency matrix of the graph
    is at most `5 \sqrt{2}`, regardless of `n`.

    Parameters
    ----------
    n : int
        Determines the number of nodes in the graph: `n^2`.
    create_using : NetworkX graph constructor, optional (default MultiGraph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : graph
        The constructed undirected multigraph.

    Raises
    ------
    NetworkXError
        If the graph is directed or not a multigraph.

    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed() or not G.is_multigraph():
        msg = "`create_using` must be an undirected multigraph."
        raise nx.NetworkXError(msg)

    for x, y in itertools.product(range(n), repeat=2):
        for u, v in (
            ((x + 2 * y) % n, y),
            ((x + (2 * y + 1)) % n, y),
            (x, (y + 2 * x) % n),
            (x, (y + (2 * x + 1)) % n),
        ):
            G.add_edge((x, y), (u, v))
    G.graph["name"] = f"margulis_gabber_galil_graph({n})"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def chordal_cycle_graph(p, create_using=None):
    """Returns the chordal cycle graph on `p` nodes.

    The returned graph is a cycle graph on `p` nodes with chords joining each
    vertex `x` to its inverse modulo `p`. This graph is a (mildly explicit)
    3-regular expander [1]_.

    `p` *must* be a prime number.

    Parameters
    ----------
    p : a prime number

        The number of vertices in the graph. This also indicates where the
        chordal edges in the cycle will be created.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : graph
        The constructed undirected multigraph.

    Raises
    ------
    NetworkXError

        If `create_using` indicates directed or not a multigraph.

    References
    ----------

    .. [1] Theorem 4.4.2 in A. Lubotzky. "Discrete groups, expanding graphs and
           invariant measures", volume 125 of Progress in Mathematics.
           Birkhäuser Verlag, Basel, 1994.

    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed() or not G.is_multigraph():
        msg = "`create_using` must be an undirected multigraph."
        raise nx.NetworkXError(msg)

    for x in range(p):
        left = (x - 1) % p
        right = (x + 1) % p
        # Here we apply Fermat's Little Theorem to compute the multiplicative
        # inverse of x in Z/pZ. By Fermat's Little Theorem,
        #
        #     x^p = x (mod p)
        #
        # Therefore,
        #
        #     x * x^(p - 2) = 1 (mod p)
        #
        # The number 0 is a special case: we just let its inverse be itself.
        chord = pow(x, p - 2, p) if x > 0 else 0
        for y in (left, right, chord):
            G.add_edge(x, y)
    G.graph["name"] = f"chordal_cycle_graph({p})"
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def paley_graph(p, create_using=None):
    r"""Returns the Paley $\frac{(p-1)}{2}$ -regular graph on $p$ nodes.

    The returned graph is a graph on $\mathbb{Z}/p\mathbb{Z}$ with edges between $x$ and $y$
    if and only if $x-y$ is a nonzero square in $\mathbb{Z}/p\mathbb{Z}$.

    If $p \equiv 1  \pmod 4$, $-1$ is a square in $\mathbb{Z}/p\mathbb{Z}$ and therefore $x-y$ is a square if and
    only if $y-x$ is also a square, i.e the edges in the Paley graph are symmetric.

    If $p \equiv 3 \pmod 4$, $-1$ is not a square in $\mathbb{Z}/p\mathbb{Z}$ and therefore either $x-y$ or $y-x$
    is a square in $\mathbb{Z}/p\mathbb{Z}$ but not both.

    Note that a more general definition of Paley graphs extends this construction
    to graphs over $q=p^n$ vertices, by using the finite field $F_q$ instead of $\mathbb{Z}/p\mathbb{Z}$.
    This construction requires to compute squares in general finite fields and is
    not what is implemented here (i.e `paley_graph(25)` does not return the true
    Paley graph associated with $5^2$).

    Parameters
    ----------
    p : int, an odd prime number.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : graph
        The constructed directed graph.

    Raises
    ------
    NetworkXError
        If the graph is a multigraph.

    References
    ----------
    Chapter 13 in B. Bollobas, Random Graphs. Second edition.
    Cambridge Studies in Advanced Mathematics, 73.
    Cambridge University Press, Cambridge (2001).
    """
    G = nx.empty_graph(0, create_using, default=nx.DiGraph)
    if G.is_multigraph():
        msg = "`create_using` cannot be a multigraph."
        raise nx.NetworkXError(msg)

    # Compute the squares in Z/pZ.
    # Make it a set to uniquify (there are exactly (p-1)/2 squares in Z/pZ
    # when is prime).
    square_set = {(x**2) % p for x in range(1, p) if (x**2) % p != 0}

    for x in range(p):
        for x2 in square_set:
            G.add_edge(x, (x + x2) % p)
    G.graph["name"] = f"paley({p})"
    return G


@nx.utils.decorators.np_random_state("seed")
@nx._dispatchable(graphs=None, returns_graph=True)
def maybe_regular_expander(n, d, *, create_using=None, max_tries=100, seed=None):
    r"""Utility for creating a random regular expander.

    Returns a random $d$-regular graph on $n$ nodes which is an expander
    graph with very good probability.

    Parameters
    ----------
    n : int
      The number of nodes.
    d : int
      The degree of each node.
    create_using : Graph Instance or Constructor
      Indicator of type of graph to return.
      If a Graph-type instance, then clear and use it.
      If a constructor, call it to create an empty graph.
      Use the Graph constructor by default.
    max_tries : int. (default: 100)
      The number of allowed loops when generating each independent cycle
    seed : (default: None)
      Seed used to set random number generation state. See :ref`Randomness<randomness>`.

    Notes
    -----
    The nodes are numbered from $0$ to $n - 1$.

    The graph is generated by taking $d / 2$ random independent cycles.

    Joel Friedman proved that in this model the resulting
    graph is an expander with probability
    $1 - O(n^{-\tau})$ where $\tau = \lceil (\sqrt{d - 1}) / 2 \rceil - 1$. [1]_

    Examples
    --------
    >>> G = nx.maybe_regular_expander(n=200, d=6, seed=8020)

    Returns
    -------
    G : graph
        The constructed undirected graph.

    Raises
    ------
    NetworkXError
        If $d % 2 != 0$ as the degree must be even.
        If $n - 1$ is less than $ 2d $ as the graph is complete at most.
        If max_tries is reached

    See Also
    --------
    is_regular_expander
    random_regular_expander_graph

    References
    ----------
    .. [1] Joel Friedman,
       A Proof of Alon’s Second Eigenvalue Conjecture and Related Problems, 2004
       https://arxiv.org/abs/cs/0405020

    """

    import numpy as np

    if n < 1:
        raise nx.NetworkXError("n must be a positive integer")

    if not (d >= 2):
        raise nx.NetworkXError("d must be greater than or equal to 2")

    if not (d % 2 == 0):
        raise nx.NetworkXError("d must be even")

    if not (n - 1 >= d):
        raise nx.NetworkXError(
            f"Need n-1>= d to have room for {d//2} independent cycles with {n} nodes"
        )

    G = nx.empty_graph(n, create_using)

    if n < 2:
        return G

    cycles = []
    edges = set()

    # Create d / 2 cycles
    for i in range(d // 2):
        iterations = max_tries
        # Make sure the cycles are independent to have a regular graph
        while len(edges) != (i + 1) * n:
            iterations -= 1
            # Faster than random.permutation(n) since there are only
            # (n-1)! distinct cycles against n! permutations of size n
            cycle = seed.permutation(n - 1).tolist()
            cycle.append(n - 1)

            new_edges = {
                (u, v)
                for u, v in nx.utils.pairwise(cycle, cyclic=True)
                if (u, v) not in edges and (v, u) not in edges
            }
            # If the new cycle has no edges in common with previous cycles
            # then add it to the list otherwise try again
            if len(new_edges) == n:
                cycles.append(cycle)
                edges.update(new_edges)

            if iterations == 0:
                raise nx.NetworkXError("Too many iterations in maybe_regular_expander")

    G.add_edges_from(edges)

    return G


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
@nx._dispatchable(preserve_edge_attrs={"G": {"weight": 1}})
def is_regular_expander(G, *, epsilon=0):
    r"""Determines whether the graph G is a regular expander. [1]_

    An expander graph is a sparse graph with strong connectivity properties.

    More precisely, this helper checks whether the graph is a
    regular $(n, d, \lambda)$-expander with $\lambda$ close to
    the Alon-Boppana bound and given by
    $\lambda = 2 \sqrt{d - 1} + \epsilon$. [2]_

    In the case where $\epsilon = 0$ then if the graph successfully passes the test
    it is a Ramanujan graph. [3]_

    A Ramanujan graph has spectral gap almost as large as possible, which makes them
    excellent expanders.

    Parameters
    ----------
    G : NetworkX graph
    epsilon : int, float, default=0

    Returns
    -------
    bool
        Whether the given graph is a regular $(n, d, \lambda)$-expander
        where $\lambda = 2 \sqrt{d - 1} + \epsilon$.

    Examples
    --------
    >>> G = nx.random_regular_expander_graph(20, 4)
    >>> nx.is_regular_expander(G)
    True

    See Also
    --------
    maybe_regular_expander
    random_regular_expander_graph

    References
    ----------
    .. [1] Expander graph, https://en.wikipedia.org/wiki/Expander_graph
    .. [2] Alon-Boppana bound, https://en.wikipedia.org/wiki/Alon%E2%80%93Boppana_bound
    .. [3] Ramanujan graphs, https://en.wikipedia.org/wiki/Ramanujan_graph

    """

    import numpy as np
    from scipy.sparse.linalg import eigsh

    if epsilon < 0:
        raise nx.NetworkXError("epsilon must be non negative")

    if not nx.is_regular(G):
        return False

    _, d = nx.utils.arbitrary_element(G.degree)

    A = nx.adjacency_matrix(G, dtype=float)
    lams = eigsh(A, which="LM", k=2, return_eigenvectors=False)

    # lambda2 is the second biggest eigenvalue
    lambda2 = min(lams)

    # Use bool() to convert numpy scalar to Python Boolean
    return bool(abs(lambda2) < 2 ** np.sqrt(d - 1) + epsilon)


@nx.utils.decorators.np_random_state("seed")
@nx._dispatchable(graphs=None, returns_graph=True)
def random_regular_expander_graph(
    n, d, *, epsilon=0, create_using=None, max_tries=100, seed=None
):
    r"""Returns a random regular expander graph on $n$ nodes with degree $d$.

    An expander graph is a sparse graph with strong connectivity properties. [1]_

    More precisely the returned graph is a $(n, d, \lambda)$-expander with
    $\lambda = 2 \sqrt{d - 1} + \epsilon$, close to the Alon-Boppana bound. [2]_

    In the case where $\epsilon = 0$ it returns a Ramanujan graph.
    A Ramanujan graph has spectral gap almost as large as possible,
    which makes them excellent expanders. [3]_

    Parameters
    ----------
    n : int
      The number of nodes.
    d : int
      The degree of each node.
    epsilon : int, float, default=0
    max_tries : int, (default: 100)
      The number of allowed loops, also used in the maybe_regular_expander utility
    seed : (default: None)
      Seed used to set random number generation state. See :ref`Randomness<randomness>`.

    Raises
    ------
    NetworkXError
        If max_tries is reached

    Examples
    --------
    >>> G = nx.random_regular_expander_graph(20, 4)
    >>> nx.is_regular_expander(G)
    True

    Notes
    -----
    This loops over `maybe_regular_expander` and can be slow when
    $n$ is too big or $\epsilon$ too small.

    See Also
    --------
    maybe_regular_expander
    is_regular_expander

    References
    ----------
    .. [1] Expander graph, https://en.wikipedia.org/wiki/Expander_graph
    .. [2] Alon-Boppana bound, https://en.wikipedia.org/wiki/Alon%E2%80%93Boppana_bound
    .. [3] Ramanujan graphs, https://en.wikipedia.org/wiki/Ramanujan_graph

    """
    G = maybe_regular_expander(
        n, d, create_using=create_using, max_tries=max_tries, seed=seed
    )
    iterations = max_tries

    while not is_regular_expander(G, epsilon=epsilon):
        iterations -= 1
        G = maybe_regular_expander(
            n=n, d=d, create_using=create_using, max_tries=max_tries, seed=seed
        )

        if iterations == 0:
            raise nx.NetworkXError(
                "Too many iterations in random_regular_expander_graph"
            )

    return G
<|endoftext|>"""Functions related to the Mycielski Operation and the Mycielskian family
of graphs.

"""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["mycielskian", "mycielski_graph"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(returns_graph=True)
def mycielskian(G, iterations=1):
    r"""Returns the Mycielskian of a simple, undirected graph G

    The Mycielskian of graph preserves a graph's triangle free
    property while increasing the chromatic number by 1.

    The Mycielski Operation on a graph, :math:`G=(V, E)`, constructs a new
    graph with :math:`2|V| + 1` nodes and :math:`3|E| + |V|` edges.

    The construction is as follows:

    Let :math:`V = {0, ..., n-1}`. Construct another vertex set
    :math:`U = {n, ..., 2n}` and a vertex, `w`.
    Construct a new graph, `M`, with vertices :math:`U \bigcup V \bigcup w`.
    For edges, :math:`(u, v) \in E` add edges :math:`(u, v), (u, v + n)`, and
    :math:`(u + n, v)` to M. Finally, for all vertices :math:`u \in U`, add
    edge :math:`(u, w)` to M.

    The Mycielski Operation can be done multiple times by repeating the above
    process iteratively.

    More information can be found at https://en.wikipedia.org/wiki/Mycielskian

    Parameters
    ----------
    G : graph
        A simple, undirected NetworkX graph
    iterations : int
        The number of iterations of the Mycielski operation to
        perform on G. Defaults to 1. Must be a non-negative integer.

    Returns
    -------
    M : graph
        The Mycielskian of G after the specified number of iterations.

    Notes
    -----
    Graph, node, and edge data are not necessarily propagated to the new graph.

    """

    M = nx.convert_node_labels_to_integers(G)

    for i in range(iterations):
        n = M.number_of_nodes()
        M.add_nodes_from(range(n, 2 * n))
        old_edges = list(M.edges())
        M.add_edges_from((u, v + n) for u, v in old_edges)
        M.add_edges_from((u + n, v) for u, v in old_edges)
        M.add_node(2 * n)
        M.add_edges_from((u + n, 2 * n) for u in range(n))

    return M


@nx._dispatchable(graphs=None, returns_graph=True)
def mycielski_graph(n):
    """Generator for the n_th Mycielski Graph.

    The Mycielski family of graphs is an infinite set of graphs.
    :math:`M_1` is the singleton graph, :math:`M_2` is two vertices with an
    edge, and, for :math:`i > 2`, :math:`M_i` is the Mycielskian of
    :math:`M_{i-1}`.

    More information can be found at
    http://mathworld.wolfram.com/MycielskiGraph.html

    Parameters
    ----------
    n : int
        The desired Mycielski Graph.

    Returns
    -------
    M : graph
        The n_th Mycielski Graph

    Notes
    -----
    The first graph in the Mycielski sequence is the singleton graph.
    The Mycielskian of this graph is not the :math:`P_2` graph, but rather the
    :math:`P_2` graph with an extra, isolated vertex. The second Mycielski
    graph is the :math:`P_2` graph, so the first two are hard coded.
    The remaining graphs are generated using the Mycielski operation.

    """

    if n < 1:
        raise nx.NetworkXError("must satisfy n >= 1")

    if n == 1:
        return nx.empty_graph(1)

    else:
        return mycielskian(nx.path_graph(2), n - 2)
<|endoftext|>"""Functions for generating line graphs."""

from collections import defaultdict
from functools import partial
from itertools import combinations

import networkx as nx
from networkx.utils import arbitrary_element
from networkx.utils.decorators import not_implemented_for

__all__ = ["line_graph", "inverse_line_graph"]


@nx._dispatchable(returns_graph=True)
def line_graph(G, create_using=None):
    r"""Returns the line graph of the graph or digraph `G`.

    The line graph of a graph `G` has a node for each edge in `G` and an
    edge joining those nodes if the two edges in `G` share a common node. For
    directed graphs, nodes are adjacent exactly when the edges they represent
    form a directed path of length two.

    The nodes of the line graph are 2-tuples of nodes in the original graph (or
    3-tuples for multigraphs, with the key of the edge as the third element).

    For information about self-loops and more discussion, see the **Notes**
    section below.

    Parameters
    ----------
    G : graph
        A NetworkX Graph, DiGraph, MultiGraph, or MultiDigraph.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    L : graph
        The line graph of G.

    Examples
    --------
    >>> G = nx.star_graph(3)
    >>> L = nx.line_graph(G)
    >>> print(sorted(map(sorted, L.edges())))  # makes a 3-clique, K3
    [[(0, 1), (0, 2)], [(0, 1), (0, 3)], [(0, 2), (0, 3)]]

    Edge attributes from `G` are not copied over as node attributes in `L`, but
    attributes can be copied manually:

    >>> G = nx.path_graph(4)
    >>> G.add_edges_from((u, v, {"tot": u + v}) for u, v in G.edges)
    >>> G.edges(data=True)
    EdgeDataView([(0, 1, {'tot': 1}), (1, 2, {'tot': 3}), (2, 3, {'tot': 5})])
    >>> H = nx.line_graph(G)
    >>> H.add_nodes_from((node, G.edges[node]) for node in H)
    >>> H.nodes(data=True)
    NodeDataView({(0, 1): {'tot': 1}, (2, 3): {'tot': 5}, (1, 2): {'tot': 3}})

    Notes
    -----
    Graph, node, and edge data are not propagated to the new graph. For
    undirected graphs, the nodes in G must be sortable, otherwise the
    constructed line graph may not be correct.

    *Self-loops in undirected graphs*

    For an undirected graph `G` without multiple edges, each edge can be
    written as a set `\{u, v\}`.  Its line graph `L` has the edges of `G` as
    its nodes. If `x` and `y` are two nodes in `L`, then `\{x, y\}` is an edge
    in `L` if and only if the intersection of `x` and `y` is nonempty. Thus,
    the set of all edges is determined by the set of all pairwise intersections
    of edges in `G`.

    Trivially, every edge in G would have a nonzero intersection with itself,
    and so every node in `L` should have a self-loop. This is not so
    interesting, and the original context of line graphs was with simple
    graphs, which had no self-loops or multiple edges. The line graph was also
    meant to be a simple graph and thus, self-loops in `L` are not part of the
    standard definition of a line graph. In a pairwise intersection matrix,
    this is analogous to excluding the diagonal entries from the line graph
    definition.

    Self-loops and multiple edges in `G` add nodes to `L` in a natural way, and
    do not require any fundamental changes to the definition. It might be
    argued that the self-loops we excluded before should now be included.
    However, the self-loops are still "trivial" in some sense and thus, are
    usually excluded.

    *Self-loops in directed graphs*

    For a directed graph `G` without multiple edges, each edge can be written
    as a tuple `(u, v)`. Its line graph `L` has the edges of `G` as its
    nodes. If `x` and `y` are two nodes in `L`, then `(x, y)` is an edge in `L`
    if and only if the tail of `x` matches the head of `y`, for example, if `x
    = (a, b)` and `y = (b, c)` for some vertices `a`, `b`, and `c` in `G`.

    Due to the directed nature of the edges, it is no longer the case that
    every edge in `G` should have a self-loop in `L`. Now, the only time
    self-loops arise is if a node in `G` itself has a self-loop.  So such
    self-loops are no longer "trivial" but instead, represent essential
    features of the topology of `G`. For this reason, the historical
    development of line digraphs is such that self-loops are included. When the
    graph `G` has multiple edges, once again only superficial changes are
    required to the definition.

    References
    ----------
    * Harary, Frank, and Norman, Robert Z., "Some properties of line digraphs",
      Rend. Circ. Mat. Palermo, II. Ser. 9 (1960), 161--168.
    * Hemminger, R. L.; Beineke, L. W. (1978), "Line graphs and line digraphs",
      in Beineke, L. W.; Wilson, R. J., Selected Topics in Graph Theory,
      Academic Press Inc., pp. 271--305.

    """
    if G.is_directed():
        L = _lg_directed(G, create_using=create_using)
    else:
        L = _lg_undirected(G, selfloops=False, create_using=create_using)
    return L


def _lg_directed(G, create_using=None):
    """Returns the line graph L of the (multi)digraph G.

    Edges in G appear as nodes in L, represented as tuples of the form (u,v)
    or (u,v,key) if G is a multidigraph. A node in L corresponding to the edge
    (u,v) is connected to every node corresponding to an edge (v,w).

    Parameters
    ----------
    G : digraph
        A directed graph or directed multigraph.
    create_using : NetworkX graph constructor, optional
       Graph type to create. If graph instance, then cleared before populated.
       Default is to use the same graph class as `G`.

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    # Create a graph specific edge function.
    get_edges = partial(G.edges, keys=True) if G.is_multigraph() else G.edges

    for from_node in get_edges():
        # from_node is: (u,v) or (u,v,key)
        L.add_node(from_node)
        for to_node in get_edges(from_node[1]):
            L.add_edge(from_node, to_node)

    return L


def _lg_undirected(G, selfloops=False, create_using=None):
    """Returns the line graph L of the (multi)graph G.

    Edges in G appear as nodes in L, represented as sorted tuples of the form
    (u,v), or (u,v,key) if G is a multigraph. A node in L corresponding to
    the edge {u,v} is connected to every node corresponding to an edge that
    involves u or v.

    Parameters
    ----------
    G : graph
        An undirected graph or multigraph.
    selfloops : bool
        If `True`, then self-loops are included in the line graph. If `False`,
        they are excluded.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    The standard algorithm for line graphs of undirected graphs does not
    produce self-loops.

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    # Graph specific functions for edges.
    get_edges = partial(G.edges, keys=True) if G.is_multigraph() else G.edges

    # Determine if we include self-loops or not.
    shift = 0 if selfloops else 1

    # Introduce numbering of nodes
    node_index = {n: i for i, n in enumerate(G)}

    # Lift canonical representation of nodes to edges in line graph
    edge_key_function = lambda edge: (node_index[edge[0]], node_index[edge[1]])

    edges = set()
    for u in G:
        # Label nodes as a sorted tuple of nodes in original graph.
        # Decide on representation of {u, v} as (u, v) or (v, u) depending on node_index.
        # -> This ensures a canonical representation and avoids comparing values of different types.
        nodes = [tuple(sorted(x[:2], key=node_index.get)) + x[2:] for x in get_edges(u)]

        if len(nodes) == 1:
            # Then the edge will be an isolated node in L.
            L.add_node(nodes[0])

        # Add a clique of `nodes` to graph. To prevent double adding edges,
        # especially important for multigraphs, we store the edges in
        # canonical form in a set.
        for i, a in enumerate(nodes):
            edges.update(
                [
                    tuple(sorted((a, b), key=edge_key_function))
                    for b in nodes[i + shift :]
                ]
            )

    L.add_edges_from(edges)
    return L


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(returns_graph=True)
def inverse_line_graph(G):
    """Returns the inverse line graph of graph G.

    If H is a graph, and G is the line graph of H, such that G = L(H).
    Then H is the inverse line graph of G.

    Not all graphs are line graphs and these do not have an inverse line graph.
    In these cases this function raises a NetworkXError.

    Parameters
    ----------
    G : graph
        A NetworkX Graph

    Returns
    -------
    H : graph
        The inverse line graph of G.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed or a multigraph

    NetworkXError
        If G is not a line graph

    Notes
    -----
    This is an implementation of the Roussopoulos algorithm[1]_.

    If G consists of multiple components, then the algorithm doesn't work.
    You should invert every component separately:

    >>> K5 = nx.complete_graph(5)
    >>> P4 = nx.Graph([("a", "b"), ("b", "c"), ("c", "d")])
    >>> G = nx.union(K5, P4)
    >>> root_graphs = []
    >>> for comp in nx.connected_components(G):
    ...     root_graphs.append(nx.inverse_line_graph(G.subgraph(comp)))
    >>> len(root_graphs)
    2

    References
    ----------
    .. [1] Roussopoulos, N.D. , "A max {m, n} algorithm for determining the graph H from
       its line graph G", Information Processing Letters 2, (1973), 108--112, ISSN 0020-0190,
       `DOI link <https://doi.org/10.1016/0020-0190(73)90029-X>`_

    """
    if G.number_of_nodes() == 0:
        return nx.empty_graph(1)
    elif G.number_of_nodes() == 1:
        v = arbitrary_element(G)
        a = (v, 0)
        b = (v, 1)
        H = nx.Graph([(a, b)])
        return H
    elif G.number_of_nodes() > 1 and G.number_of_edges() == 0:
        msg = (
            "inverse_line_graph() doesn't work on an edgeless graph. "
            "Please use this function on each component separately."
        )
        raise nx.NetworkXError(msg)

    if nx.number_of_selfloops(G) != 0:
        msg = (
            "A line graph as generated by NetworkX has no selfloops, so G has no "
            "inverse line graph. Please remove the selfloops from G and try again."
        )
        raise nx.NetworkXError(msg)

    starting_cell = _select_starting_cell(G)
    P = _find_partition(G, starting_cell)
    # count how many times each vertex appears in the partition set
    P_count = {u: 0 for u in G.nodes}
    for p in P:
        for u in p:
            P_count[u] += 1

    if max(P_count.values()) > 2:
        msg = "G is not a line graph (vertex found in more than two partition cells)"
        raise nx.NetworkXError(msg)
    W = tuple((u,) for u in P_count if P_count[u] == 1)
    H = nx.Graph()
    H.add_nodes_from(P)
    H.add_nodes_from(W)
    for a, b in combinations(H.nodes, 2):
        if any(a_bit in b for a_bit in a):
            H.add_edge(a, b)
    return H


def _triangles(G, e):
    """Return list of all triangles containing edge e"""
    u, v = e
    if u not in G:
        raise nx.NetworkXError(f"Vertex {u} not in graph")
    if v not in G[u]:
        raise nx.NetworkXError(f"Edge ({u}, {v}) not in graph")
    triangle_list = []
    for x in G[u]:
        if x in G[v]:
            triangle_list.append((u, v, x))
    return triangle_list


def _odd_triangle(G, T):
    """Test whether T is an odd triangle in G

    Parameters
    ----------
    G : NetworkX Graph
    T : 3-tuple of vertices forming triangle in G

    Returns
    -------
    True is T is an odd triangle
    False otherwise

    Raises
    ------
    NetworkXError
        T is not a triangle in G

    Notes
    -----
    An odd triangle is one in which there exists another vertex in G which is
    adjacent to either exactly one or exactly all three of the vertices in the
    triangle.

    """
    for u in T:
        if u not in G.nodes():
            raise nx.NetworkXError(f"Vertex {u} not in graph")
    for e in list(combinations(T, 2)):
        if e[0] not in G[e[1]]:
            raise nx.NetworkXError(f"Edge ({e[0]}, {e[1]}) not in graph")

    T_nbrs = defaultdict(int)
    for t in T:
        for v in G[t]:
            if v not in T:
                T_nbrs[v] += 1
    return any(T_nbrs[v] in [1, 3] for v in T_nbrs)


def _find_partition(G, starting_cell):
    """Find a partition of the vertices of G into cells of complete graphs

    Parameters
    ----------
    G : NetworkX Graph
    starting_cell : tuple of vertices in G which form a cell

    Returns
    -------
    List of tuples of vertices of G

    Raises
    ------
    NetworkXError
        If a cell is not a complete subgraph then G is not a line graph
    """
    G_partition = G.copy()
    P = [starting_cell]  # partition set
    G_partition.remove_edges_from(list(combinations(starting_cell, 2)))
    # keep list of partitioned nodes which might have an edge in G_partition
    partitioned_vertices = list(starting_cell)
    while G_partition.number_of_edges() > 0:
        # there are still edges left and so more cells to be made
        u = partitioned_vertices.pop()
        deg_u = len(G_partition[u])
        if deg_u != 0:
            # if u still has edges then we need to find its other cell
            # this other cell must be a complete subgraph or else G is
            # not a line graph
            new_cell = [u] + list(G_partition[u])
            for u in new_cell:
                for v in new_cell:
                    if (u != v) and (v not in G_partition[u]):
                        msg = (
                            "G is not a line graph "
                            "(partition cell not a complete subgraph)"
                        )
                        raise nx.NetworkXError(msg)
            P.append(tuple(new_cell))
            G_partition.remove_edges_from(list(combinations(new_cell, 2)))
            partitioned_vertices += new_cell
    return P


def _select_starting_cell(G, starting_edge=None):
    """Select a cell to initiate _find_partition

    Parameters
    ----------
    G : NetworkX Graph
    starting_edge: an edge to build the starting cell from

    Returns
    -------
    Tuple of vertices in G

    Raises
    ------
    NetworkXError
        If it is determined that G is not a line graph

    Notes
    -----
    If starting edge not specified then pick an arbitrary edge - doesn't
    matter which. However, this function may call itself requiring a
    specific starting edge. Note that the r, s notation for counting
    triangles is the same as in the Roussopoulos paper cited above.
    """
    if starting_edge is None:
        e = arbitrary_element(G.edges())
    else:
        e = starting_edge
        if e[0] not in G.nodes():
            raise nx.NetworkXError(f"Vertex {e[0]} not in graph")
        if e[1] not in G[e[0]]:
            msg = f"starting_edge ({e[0]}, {e[1]}) is not in the Graph"
            raise nx.NetworkXError(msg)
    e_triangles = _triangles(G, e)
    r = len(e_triangles)
    if r == 0:
        # there are no triangles containing e, so the starting cell is just e
        starting_cell = e
    elif r == 1:
        # there is exactly one triangle, T, containing e. If other 2 edges
        # of T belong only to this triangle then T is starting cell
        T = e_triangles[0]
        a, b, c = T
        # ab was original edge so check the other 2 edges
        ac_edges = len(_triangles(G, (a, c)))
        bc_edges = len(_triangles(G, (b, c)))
        if ac_edges == 1:
            if bc_edges == 1:
                starting_cell = T
            else:
                return _select_starting_cell(G, starting_edge=(b, c))
        else:
            return _select_starting_cell(G, starting_edge=(a, c))
    else:
        # r >= 2 so we need to count the number of odd triangles, s
        s = 0
        odd_triangles = []
        for T in e_triangles:
            if _odd_triangle(G, T):
                s += 1
                odd_triangles.append(T)
        if r == 2 and s == 0:
            # in this case either triangle works, so just use T
            starting_cell = T
        elif r - 1 <= s <= r:
            # check if odd triangles containing e form complete subgraph
            triangle_nodes = set()
            for T in odd_triangles:
                for x in T:
                    triangle_nodes.add(x)

            for u in triangle_nodes:
                for v in triangle_nodes:
                    if u != v and (v not in G[u]):
                        msg = (
                            "G is not a line graph (odd triangles "
                            "do not form complete subgraph)"
                        )
                        raise nx.NetworkXError(msg)
            # otherwise then we can use this as the starting cell
            starting_cell = tuple(triangle_nodes)

        else:
            msg = (
                "G is not a line graph (incorrect number of "
                "odd triangles around starting edge)"
            )
            raise nx.NetworkXError(msg)
    return starting_cell
<|endoftext|>"""
Implementation of the Wright, Richmond, Odlyzko and McKay (WROM)
algorithm for the enumeration of all non-isomorphic free trees of a
given order.  Rooted trees are represented by level sequences, i.e.,
lists in which the i-th element specifies the distance of vertex i to
the root.

"""

__all__ = ["nonisomorphic_trees", "number_of_nonisomorphic_trees"]

import networkx as nx


@nx._dispatchable(graphs=None, returns_graph=True)
def nonisomorphic_trees(order, create="graph"):
    """Generates lists of nonisomorphic trees

    Parameters
    ----------
    order : int
       order of the desired tree(s)

    create : one of {"graph", "matrix"} (default="graph")
       If ``"graph"`` is selected a list of ``Graph`` instances will be returned,
       if matrix is selected a list of adjacency matrices will be returned.

       .. deprecated:: 3.3

          The `create` argument is deprecated and will be removed in NetworkX
          version 3.5. In the future, `nonisomorphic_trees` will yield graph
          instances by default. To generate adjacency matrices, call
          ``nx.to_numpy_array`` on the output, e.g.::

             [nx.to_numpy_array(G) for G in nx.nonisomorphic_trees(N)]

    Yields
    ------
    list
       A list of nonisomorphic trees, in one of two formats depending on the
       value of the `create` parameter:
       - ``create="graph"``: yields a list of `networkx.Graph` instances
       - ``create="matrix"``: yields a list of list-of-lists representing adjacency matrices
    """

    if order < 2:
        raise ValueError
    # start at the path graph rooted at its center
    layout = list(range(order // 2 + 1)) + list(range(1, (order + 1) // 2))

    while layout is not None:
        layout = _next_tree(layout)
        if layout is not None:
            if create == "graph":
                yield _layout_to_graph(layout)
            elif create == "matrix":
                import warnings

                warnings.warn(
                    (
                        "\n\nThe 'create=matrix' argument of nonisomorphic_trees\n"
                        "is deprecated and will be removed in version 3.5.\n"
                        "Use ``nx.to_numpy_array`` to convert graphs to adjacency "
                        "matrices, e.g.::\n\n"
                        "   [nx.to_numpy_array(G) for G in nx.nonisomorphic_trees(N)]"
                    ),
                    category=DeprecationWarning,
                    stacklevel=2,
                )

                yield _layout_to_matrix(layout)
            layout = _next_rooted_tree(layout)


@nx._dispatchable(graphs=None)
def number_of_nonisomorphic_trees(order):
    """Returns the number of nonisomorphic trees

    Parameters
    ----------
    order : int
      order of the desired tree(s)

    Returns
    -------
    length : Number of nonisomorphic graphs for the given order

    References
    ----------

    """
    return sum(1 for _ in nonisomorphic_trees(order))


def _next_rooted_tree(predecessor, p=None):
    """One iteration of the Beyer-Hedetniemi algorithm."""

    if p is None:
        p = len(predecessor) - 1
        while predecessor[p] == 1:
            p -= 1
    if p == 0:
        return None

    q = p - 1
    while predecessor[q] != predecessor[p] - 1:
        q -= 1
    result = list(predecessor)
    for i in range(p, len(result)):
        result[i] = result[i - p + q]
    return result


def _next_tree(candidate):
    """One iteration of the Wright, Richmond, Odlyzko and McKay
    algorithm."""

    # valid representation of a free tree if:
    # there are at least two vertices at layer 1
    # (this is always the case because we start at the path graph)
    left, rest = _split_tree(candidate)

    # and the left subtree of the root
    # is less high than the tree with the left subtree removed
    left_height = max(left)
    rest_height = max(rest)
    valid = rest_height >= left_height

    if valid and rest_height == left_height:
        # and, if left and rest are of the same height,
        # if left does not encompass more vertices
        if len(left) > len(rest):
            valid = False
        # and, if they have the same number or vertices,
        # if left does not come after rest lexicographically
        elif len(left) == len(rest) and left > rest:
            valid = False

    if valid:
        return candidate
    else:
        # jump to the next valid free tree
        p = len(left)
        new_candidate = _next_rooted_tree(candidate, p)
        if candidate[p] > 2:
            new_left, new_rest = _split_tree(new_candidate)
            new_left_height = max(new_left)
            suffix = range(1, new_left_height + 2)
            new_candidate[-len(suffix) :] = suffix
        return new_candidate


def _split_tree(layout):
    """Returns a tuple of two layouts, one containing the left
    subtree of the root vertex, and one containing the original tree
    with the left subtree removed."""

    one_found = False
    m = None
    for i in range(len(layout)):
        if layout[i] == 1:
            if one_found:
                m = i
                break
            else:
                one_found = True

    if m is None:
        m = len(layout)

    left = [layout[i] - 1 for i in range(1, m)]
    rest = [0] + [layout[i] for i in range(m, len(layout))]
    return (left, rest)


def _layout_to_matrix(layout):
    """Create the adjacency matrix for the tree specified by the
    given layout (level sequence)."""

    result = [[0] * len(layout) for i in range(len(layout))]
    stack = []
    for i in range(len(layout)):
        i_level = layout[i]
        if stack:
            j = stack[-1]
            j_level = layout[j]
            while j_level >= i_level:
                stack.pop()
                j = stack[-1]
                j_level = layout[j]
            result[i][j] = result[j][i] = 1
        stack.append(i)
    return result


def _layout_to_graph(layout):
    """Create a NetworkX Graph for the tree specified by the
    given layout(level sequence)"""
    G = nx.Graph()
    stack = []
    for i in range(len(layout)):
        i_level = layout[i]
        if stack:
            j = stack[-1]
            j_level = layout[j]
            while j_level >= i_level:
                stack.pop()
                j = stack[-1]
                j_level = layout[j]
            G.add_edge(i, j)
        stack.append(i)
    return G
<|endoftext|>"""
Generators for random intersection graphs.
"""

import networkx as nx
from networkx.utils import py_random_state

__all__ = [
    "uniform_random_intersection_graph",
    "k_random_intersection_graph",
    "general_random_intersection_graph",
]


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def uniform_random_intersection_graph(n, m, p, seed=None):
    """Returns a uniform random intersection graph.

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set (nodes)
    m : int
        The number of nodes in the second bipartite set (attributes)
    p : float
        Probability of connecting nodes between bipartite sets
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    gnp_random_graph

    References
    ----------
    .. [1] K.B. Singer-Cohen, Random Intersection Graphs, 1995,
       PhD thesis, Johns Hopkins University
    .. [2] Fill, J. A., Scheinerman, E. R., and Singer-Cohen, K. B.,
       Random intersection graphs when m = !(n):
       An equivalence theorem relating the evolution of the g(n, m, p)
       and g(n, p) models. Random Struct. Algorithms 16, 2 (2000), 156–176.
    """
    from networkx.algorithms import bipartite

    G = bipartite.random_graph(n, m, p, seed)
    return nx.projected_graph(G, range(n))


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def k_random_intersection_graph(n, m, k, seed=None):
    """Returns a intersection graph with randomly chosen attribute sets for
    each node that are of equal size (k).

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set (nodes)
    m : int
        The number of nodes in the second bipartite set (attributes)
    k : float
        Size of attribute set to assign to each node.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    gnp_random_graph, uniform_random_intersection_graph

    References
    ----------
    .. [1] Godehardt, E., and Jaworski, J.
       Two models of random intersection graphs and their applications.
       Electronic Notes in Discrete Mathematics 10 (2001), 129--132.
    """
    G = nx.empty_graph(n + m)
    mset = range(n, n + m)
    for v in range(n):
        targets = seed.sample(mset, k)
        G.add_edges_from(zip([v] * len(targets), targets))
    return nx.projected_graph(G, range(n))


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def general_random_intersection_graph(n, m, p, seed=None):
    """Returns a random intersection graph with independent probabilities
    for connections between node and attribute sets.

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set (nodes)
    m : int
        The number of nodes in the second bipartite set (attributes)
    p : list of floats of length m
        Probabilities for connecting nodes to each attribute
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    gnp_random_graph, uniform_random_intersection_graph

    References
    ----------
    .. [1] Nikoletseas, S. E., Raptopoulos, C., and Spirakis, P. G.
       The existence and efficient construction of large independent sets
       in general random intersection graphs. In ICALP (2004), J. D´ıaz,
       J. Karhum¨aki, A. Lepist¨o, and D. Sannella, Eds., vol. 3142
       of Lecture Notes in Computer Science, Springer, pp. 1029–1040.
    """
    if len(p) != m:
        raise ValueError("Probability list p must have m elements.")
    G = nx.empty_graph(n + m)
    mset = range(n, n + m)
    for u in range(n):
        for v, q in zip(mset, p):
            if seed.random() < q:
                G.add_edge(u, v)
    return nx.projected_graph(G, range(n))
<|endoftext|>"""
Generators for interval graph.
"""

from collections.abc import Sequence

import networkx as nx

__all__ = ["interval_graph"]


@nx._dispatchable(graphs=None, returns_graph=True)
def interval_graph(intervals):
    """Generates an interval graph for a list of intervals given.

    In graph theory, an interval graph is an undirected graph formed from a set
    of closed intervals on the real line, with a vertex for each interval
    and an edge between vertices whose intervals intersect.
    It is the intersection graph of the intervals.

    More information can be found at:
    https://en.wikipedia.org/wiki/Interval_graph

    Parameters
    ----------
    intervals : a sequence of intervals, say (l, r) where l is the left end,
    and r is the right end of the closed interval.

    Returns
    -------
    G : networkx graph

    Examples
    --------
    >>> intervals = [(-2, 3), [1, 4], (2, 3), (4, 6)]
    >>> G = nx.interval_graph(intervals)
    >>> sorted(G.edges)
    [((-2, 3), (1, 4)), ((-2, 3), (2, 3)), ((1, 4), (2, 3)), ((1, 4), (4, 6))]

    Raises
    ------
    :exc:`TypeError`
        if `intervals` contains None or an element which is not
        collections.abc.Sequence or not a length of 2.
    :exc:`ValueError`
        if `intervals` contains an interval such that min1 > max1
        where min1,max1 = interval
    """
    intervals = list(intervals)
    for interval in intervals:
        if not (isinstance(interval, Sequence) and len(interval) == 2):
            raise TypeError(
                "Each interval must have length 2, and be a "
                "collections.abc.Sequence such as tuple or list."
            )
        if interval[0] > interval[1]:
            raise ValueError(f"Interval must have lower value first. Got {interval}")

    graph = nx.Graph()

    tupled_intervals = [tuple(interval) for interval in intervals]
    graph.add_nodes_from(tupled_intervals)

    while tupled_intervals:
        min1, max1 = interval1 = tupled_intervals.pop()
        for interval2 in tupled_intervals:
            min2, max2 = interval2
            if max1 >= min2 and max2 >= min1:
                graph.add_edge(interval1, interval2)
    return graph
<|endoftext|>"""Generates graphs resembling the Internet Autonomous System network"""

import networkx as nx
from networkx.utils import py_random_state

__all__ = ["random_internet_as_graph"]


def uniform_int_from_avg(a, m, seed):
    """Pick a random integer with uniform probability.

    Returns a random integer uniformly taken from a distribution with
    minimum value 'a' and average value 'm', X~U(a,b), E[X]=m, X in N where
    b = 2*m - a.

    Notes
    -----
    p = (b-floor(b))/2
    X = X1 + X2; X1~U(a,floor(b)), X2~B(p)
    E[X] = E[X1] + E[X2] = (floor(b)+a)/2 + (b-floor(b))/2 = (b+a)/2 = m
    """

    from math import floor

    assert m >= a
    b = 2 * m - a
    p = (b - floor(b)) / 2
    X1 = round(seed.random() * (floor(b) - a) + a)
    if seed.random() < p:
        X2 = 1
    else:
        X2 = 0
    return X1 + X2


def choose_pref_attach(degs, seed):
    """Pick a random value, with a probability given by its weight.

    Returns a random choice among degs keys, each of which has a
    probability proportional to the corresponding dictionary value.

    Parameters
    ----------
    degs: dictionary
        It contains the possible values (keys) and the corresponding
        probabilities (values)
    seed: random state

    Returns
    -------
    v: object
        A key of degs or None if degs is empty
    """

    if len(degs) == 0:
        return None
    s = sum(degs.values())
    if s == 0:
        return seed.choice(list(degs.keys()))
    v = seed.random() * s

    nodes = list(degs.keys())
    i = 0
    acc = degs[nodes[i]]
    while v > acc:
        i += 1
        acc += degs[nodes[i]]
    return nodes[i]


class AS_graph_generator:
    """Generates random internet AS graphs."""

    def __init__(self, n, seed):
        """Initializes variables. Immediate numbers are taken from [1].

        Parameters
        ----------
        n: integer
            Number of graph nodes
        seed: random state
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Returns
        -------
        GG: AS_graph_generator object

        References
        ----------
        [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
        BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
        in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
        """

        self.seed = seed
        self.n_t = min(n, round(self.seed.random() * 2 + 4))  # num of T nodes
        self.n_m = round(0.15 * n)  # number of M nodes
        self.n_cp = round(0.05 * n)  # number of CP nodes
        self.n_c = max(0, n - self.n_t - self.n_m - self.n_cp)  # number of C nodes

        self.d_m = 2 + (2.5 * n) / 10000  # average multihoming degree for M nodes
        self.d_cp = 2 + (1.5 * n) / 10000  # avg multihoming degree for CP nodes
        self.d_c = 1 + (5 * n) / 100000  # average multihoming degree for C nodes

        self.p_m_m = 1 + (2 * n) / 10000  # avg num of peer edges between M and M
        self.p_cp_m = 0.2 + (2 * n) / 10000  # avg num of peer edges between CP, M
        self.p_cp_cp = 0.05 + (2 * n) / 100000  # avg num of peer edges btwn CP, CP

        self.t_m = 0.375  # probability M's provider is T
        self.t_cp = 0.375  # probability CP's provider is T
        self.t_c = 0.125  # probability C's provider is T

    def t_graph(self):
        """Generates the core mesh network of tier one nodes of a AS graph.

        Returns
        -------
        G: Networkx Graph
            Core network
        """

        self.G = nx.Graph()
        for i in range(self.n_t):
            self.G.add_node(i, type="T")
            for r in self.regions:
                self.regions[r].add(i)
            for j in self.G.nodes():
                if i != j:
                    self.add_edge(i, j, "peer")
            self.customers[i] = set()
            self.providers[i] = set()
        return self.G

    def add_edge(self, i, j, kind):
        if kind == "transit":
            customer = str(i)
        else:
            customer = "none"
        self.G.add_edge(i, j, type=kind, customer=customer)

    def choose_peer_pref_attach(self, node_list):
        """Pick a node with a probability weighted by its peer degree.

        Pick a node from node_list with preferential attachment
        computed only on their peer degree
        """

        d = {}
        for n in node_list:
            d[n] = self.G.nodes[n]["peers"]
        return choose_pref_attach(d, self.seed)

    def choose_node_pref_attach(self, node_list):
        """Pick a node with a probability weighted by its degree.

        Pick a node from node_list with preferential attachment
        computed on their degree
        """

        degs = dict(self.G.degree(node_list))
        return choose_pref_attach(degs, self.seed)

    def add_customer(self, i, j):
        """Keep the dictionaries 'customers' and 'providers' consistent."""

        self.customers[j].add(i)
        self.providers[i].add(j)
        for z in self.providers[j]:
            self.customers[z].add(i)
            self.providers[i].add(z)

    def add_node(self, i, kind, reg2prob, avg_deg, t_edge_prob):
        """Add a node and its customer transit edges to the graph.

        Parameters
        ----------
        i: object
            Identifier of the new node
        kind: string
            Type of the new node. Options are: 'M' for middle node, 'CP' for
            content provider and 'C' for customer.
        reg2prob: float
            Probability the new node can be in two different regions.
        avg_deg: float
            Average number of transit nodes of which node i is customer.
        t_edge_prob: float
            Probability node i establish a customer transit edge with a tier
            one (T) node

        Returns
        -------
        i: object
            Identifier of the new node
        """

        regs = 1  # regions in which node resides
        if self.seed.random() < reg2prob:  # node is in two regions
            regs = 2
        node_options = set()

        self.G.add_node(i, type=kind, peers=0)
        self.customers[i] = set()
        self.providers[i] = set()
        self.nodes[kind].add(i)
        for r in self.seed.sample(list(self.regions), regs):
            node_options = node_options.union(self.regions[r])
            self.regions[r].add(i)

        edge_num = uniform_int_from_avg(1, avg_deg, self.seed)

        t_options = node_options.intersection(self.nodes["T"])
        m_options = node_options.intersection(self.nodes["M"])
        if i in m_options:
            m_options.remove(i)
        d = 0
        while d < edge_num and (len(t_options) > 0 or len(m_options) > 0):
            if len(m_options) == 0 or (
                len(t_options) > 0 and self.seed.random() < t_edge_prob
            ):  # add edge to a T node
                j = self.choose_node_pref_attach(t_options)
                t_options.remove(j)
            else:
                j = self.choose_node_pref_attach(m_options)
                m_options.remove(j)
            self.add_edge(i, j, "transit")
            self.add_customer(i, j)
            d += 1

        return i

    def add_m_peering_link(self, m, to_kind):
        """Add a peering link between two middle tier (M) nodes.

        Target node j is drawn considering a preferential attachment based on
        other M node peering degree.

        Parameters
        ----------
        m: object
            Node identifier
        to_kind: string
            type for target node j (must be always M)

        Returns
        -------
        success: boolean
        """

        # candidates are of type 'M' and are not customers of m
        node_options = self.nodes["M"].difference(self.customers[m])
        # candidates are not providers of m
        node_options = node_options.difference(self.providers[m])
        # remove self
        if m in node_options:
            node_options.remove(m)

        # remove candidates we are already connected to
        for j in self.G.neighbors(m):
            if j in node_options:
                node_options.remove(j)

        if len(node_options) > 0:
            j = self.choose_peer_pref_attach(node_options)
            self.add_edge(m, j, "peer")
            self.G.nodes[m]["peers"] += 1
            self.G.nodes[j]["peers"] += 1
            return True
        else:
            return False

    def add_cp_peering_link(self, cp, to_kind):
        """Add a peering link to a content provider (CP) node.

        Target node j can be CP or M and it is drawn uniformly among the nodes
        belonging to the same region as cp.

        Parameters
        ----------
        cp: object
            Node identifier
        to_kind: string
            type for target node j (must be M or CP)

        Returns
        -------
        success: boolean
        """

        node_options = set()
        for r in self.regions:  # options include nodes in the same region(s)
            if cp in self.regions[r]:
                node_options = node_options.union(self.regions[r])

        # options are restricted to the indicated kind ('M' or 'CP')
        node_options = self.nodes[to_kind].intersection(node_options)

        # remove self
        if cp in node_options:
            node_options.remove(cp)

        # remove nodes that are cp's providers
        node_options = node_options.difference(self.providers[cp])

        # remove nodes we are already connected to
        for j in self.G.neighbors(cp):
            if j in node_options:
                node_options.remove(j)

        if len(node_options) > 0:
            j = self.seed.sample(list(node_options), 1)[0]
            self.add_edge(cp, j, "peer")
            self.G.nodes[cp]["peers"] += 1
            self.G.nodes[j]["peers"] += 1
            return True
        else:
            return False

    def graph_regions(self, rn):
        """Initializes AS network regions.

        Parameters
        ----------
        rn: integer
            Number of regions
        """

        self.regions = {}
        for i in range(rn):
            self.regions["REG" + str(i)] = set()

    def add_peering_links(self, from_kind, to_kind):
        """Utility function to add peering links among node groups."""
        peer_link_method = None
        if from_kind == "M":
            peer_link_method = self.add_m_peering_link
            m = self.p_m_m
        if from_kind == "CP":
            peer_link_method = self.add_cp_peering_link
            if to_kind == "M":
                m = self.p_cp_m
            else:
                m = self.p_cp_cp

        for i in self.nodes[from_kind]:
            num = uniform_int_from_avg(0, m, self.seed)
            for _ in range(num):
                peer_link_method(i, to_kind)

    def generate(self):
        """Generates a random AS network graph as described in [1].

        Returns
        -------
        G: Graph object

        Notes
        -----
        The process steps are the following: first we create the core network
        of tier one nodes, then we add the middle tier (M), the content
        provider (CP) and the customer (C) nodes along with their transit edges
        (link i,j means i is customer of j). Finally we add peering links
        between M nodes, between M and CP nodes and between CP node couples.
        For a detailed description of the algorithm, please refer to [1].

        References
        ----------
        [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
        BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
        in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
        """

        self.graph_regions(5)
        self.customers = {}
        self.providers = {}
        self.nodes = {"T": set(), "M": set(), "CP": set(), "C": set()}

        self.t_graph()
        self.nodes["T"] = set(self.G.nodes())

        i = len(self.nodes["T"])
        for _ in range(self.n_m):
            self.nodes["M"].add(self.add_node(i, "M", 0.2, self.d_m, self.t_m))
            i += 1
        for _ in range(self.n_cp):
            self.nodes["CP"].add(self.add_node(i, "CP", 0.05, self.d_cp, self.t_cp))
            i += 1
        for _ in range(self.n_c):
            self.nodes["C"].add(self.add_node(i, "C", 0, self.d_c, self.t_c))
            i += 1

        self.add_peering_links("M", "M")
        self.add_peering_links("CP", "M")
        self.add_peering_links("CP", "CP")

        return self.G


@py_random_state(1)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_internet_as_graph(n, seed=None):
    """Generates a random undirected graph resembling the Internet AS network

    Parameters
    ----------
    n: integer in [1000, 10000]
        Number of graph nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G: Networkx Graph object
        A randomly generated undirected graph

    Notes
    -----
    This algorithm returns an undirected graph resembling the Internet
    Autonomous System (AS) network, it uses the approach by Elmokashfi et al.
    [1]_ and it grants the properties described in the related paper [1]_.

    Each node models an autonomous system, with an attribute 'type' specifying
    its kind; tier-1 (T), mid-level (M), customer (C) or content-provider (CP).
    Each edge models an ADV communication link (hence, bidirectional) with
    attributes:

      - type: transit|peer, the kind of commercial agreement between nodes;
      - customer: <node id>, the identifier of the node acting as customer
        ('none' if type is peer).

    References
    ----------
    .. [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
       BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
       in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
    """

    GG = AS_graph_generator(n, seed)
    G = GG.generate()
    return G
<|endoftext|>"""Generators for some classic graphs.

The typical graph builder function is called as follows:

>>> G = nx.complete_graph(100)

returning the complete graph on n nodes labeled 0, .., 99
as a simple graph. Except for `empty_graph`, all the functions
in this module return a Graph class (i.e. a simple, undirected graph).

"""

import itertools
import numbers

import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise

__all__ = [
    "balanced_tree",
    "barbell_graph",
    "binomial_tree",
    "complete_graph",
    "complete_multipartite_graph",
    "circular_ladder_graph",
    "circulant_graph",
    "cycle_graph",
    "dorogovtsev_goltsev_mendes_graph",
    "empty_graph",
    "full_rary_tree",
    "kneser_graph",
    "ladder_graph",
    "lollipop_graph",
    "null_graph",
    "path_graph",
    "star_graph",
    "tadpole_graph",
    "trivial_graph",
    "turan_graph",
    "wheel_graph",
]


# -------------------------------------------------------------------
#   Some Classic Graphs
# -------------------------------------------------------------------


def _tree_edges(n, r):
    if n == 0:
        return
    # helper function for trees
    # yields edges in rooted tree at 0 with n nodes and branching ratio r
    nodes = iter(range(n))
    parents = [next(nodes)]  # stack of max length r
    while parents:
        source = parents.pop(0)
        for i in range(r):
            try:
                target = next(nodes)
                parents.append(target)
                yield source, target
            except StopIteration:
                break


@nx._dispatchable(graphs=None, returns_graph=True)
def full_rary_tree(r, n, create_using=None):
    """Creates a full r-ary tree of `n` nodes.

    Sometimes called a k-ary, n-ary, or m-ary tree.
    "... all non-leaf nodes have exactly r children and all levels
    are full except for some rightmost position of the bottom level
    (if a leaf at the bottom level is missing, then so are all of the
    leaves to its right." [1]_

    .. plot::

        >>> nx.draw(nx.full_rary_tree(2, 10))

    Parameters
    ----------
    r : int
        branching factor of the tree
    n : int
        Number of nodes in the tree
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        An r-ary tree with n nodes

    References
    ----------
    .. [1] An introduction to data structures and algorithms,
           James Andrew Storer,  Birkhauser Boston 2001, (page 225).
    """
    G = empty_graph(n, create_using)
    G.add_edges_from(_tree_edges(n, r))
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def kneser_graph(n, k):
    """Returns the Kneser Graph with parameters `n` and `k`.

    The Kneser Graph has nodes that are k-tuples (subsets) of the integers
    between 0 and ``n-1``. Nodes are adjacent if their corresponding sets are disjoint.

    Parameters
    ----------
    n: int
        Number of integers from which to make node subsets.
        Subsets are drawn from ``set(range(n))``.
    k: int
        Size of the subsets.

    Returns
    -------
    G : NetworkX Graph

    Examples
    --------
    >>> G = nx.kneser_graph(5, 2)
    >>> G.number_of_nodes()
    10
    >>> G.number_of_edges()
    15
    >>> nx.is_isomorphic(G, nx.petersen_graph())
    True
    """
    if n <= 0:
        raise NetworkXError("n should be greater than zero")
    if k <= 0 or k > n:
        raise NetworkXError("k should be greater than zero and smaller than n")

    G = nx.Graph()
    # Create all k-subsets of [0, 1, ..., n-1]
    subsets = list(itertools.combinations(range(n), k))

    if 2 * k > n:
        G.add_nodes_from(subsets)

    universe = set(range(n))
    comb = itertools.combinations  # only to make it all fit on one line
    G.add_edges_from((s, t) for s in subsets for t in comb(universe - set(s), k))
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def balanced_tree(r, h, create_using=None):
    """Returns the perfectly balanced `r`-ary tree of height `h`.

    .. plot::

        >>> nx.draw(nx.balanced_tree(2, 3))

    Parameters
    ----------
    r : int
        Branching factor of the tree; each node will have `r`
        children.

    h : int
        Height of the tree.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : NetworkX graph
        A balanced `r`-ary tree of height `h`.

    Notes
    -----
    This is the rooted tree where all leaves are at distance `h` from
    the root. The root has degree `r` and all other internal nodes
    have degree `r + 1`.

    Node labels are integers, starting from zero.

    A balanced tree is also known as a *complete r-ary tree*.

    """
    # The number of nodes in the balanced tree is `1 + r + ... + r^h`,
    # which is computed by using the closed-form formula for a geometric
    # sum with ratio `r`. In the special case that `r` is 1, the number
    # of nodes is simply `h + 1` (since the tree is actually a path
    # graph).
    if r == 1:
        n = h + 1
    else:
        # This must be an integer if both `r` and `h` are integers. If
        # they are not, we force integer division anyway.
        n = (1 - r ** (h + 1)) // (1 - r)
    return full_rary_tree(r, n, create_using=create_using)


@nx._dispatchable(graphs=None, returns_graph=True)
def barbell_graph(m1, m2, create_using=None):
    """Returns the Barbell Graph: two complete graphs connected by a path.

    .. plot::

        >>> nx.draw(nx.barbell_graph(4, 2))

    Parameters
    ----------
    m1 : int
        Size of the left and right barbells, must be greater than 2.

    m2 : int
        Length of the path connecting the barbells.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.
       Only undirected Graphs are supported.

    Returns
    -------
    G : NetworkX graph
        A barbell graph.

    Notes
    -----


    Two identical complete graphs $K_{m1}$ form the left and right bells,
    and are connected by a path $P_{m2}$.

    The `2*m1+m2`  nodes are numbered
        `0, ..., m1-1` for the left barbell,
        `m1, ..., m1+m2-1` for the path,
        and `m1+m2, ..., 2*m1+m2-1` for the right barbell.

    The 3 subgraphs are joined via the edges `(m1-1, m1)` and
    `(m1+m2-1, m1+m2)`. If `m2=0`, this is merely two complete
    graphs joined together.

    This graph is an extremal example in David Aldous
    and Jim Fill's e-text on Random Walks on Graphs.

    """
    if m1 < 2:
        raise NetworkXError("Invalid graph description, m1 should be >=2")
    if m2 < 0:
        raise NetworkXError("Invalid graph description, m2 should be >=0")

    # left barbell
    G = complete_graph(m1, create_using)
    if G.is_directed():
        raise NetworkXError("Directed Graph not supported")

    # connecting path
    G.add_nodes_from(range(m1, m1 + m2 - 1))
    if m2 > 1:
        G.add_edges_from(pairwise(range(m1, m1 + m2)))

    # right barbell
    G.add_edges_from(
        (u, v) for u in range(m1 + m2, 2 * m1 + m2) for v in range(u + 1, 2 * m1 + m2)
    )

    # connect it up
    G.add_edge(m1 - 1, m1)
    if m2 > 0:
        G.add_edge(m1 + m2 - 1, m1 + m2)

    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def binomial_tree(n, create_using=None):
    """Returns the Binomial Tree of order n.

    The binomial tree of order 0 consists of a single node. A binomial tree of order k
    is defined recursively by linking two binomial trees of order k-1: the root of one is
    the leftmost child of the root of the other.

    .. plot::

        >>> nx.draw(nx.binomial_tree(3))

    Parameters
    ----------
    n : int
        Order of the binomial tree.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : NetworkX graph
        A binomial tree of $2^n$ nodes and $2^n - 1$ edges.

    """
    G = nx.empty_graph(1, create_using)

    N = 1
    for i in range(n):
        # Use G.edges() to ensure 2-tuples. G.edges is 3-tuple for MultiGraph
        edges = [(u + N, v + N) for (u, v) in G.edges()]
        G.add_edges_from(edges)
        G.add_edge(0, N)
        N *= 2
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number(0)
def complete_graph(n, create_using=None):
    """Return the complete graph `K_n` with n nodes.

    A complete graph on `n` nodes means that all pairs
    of distinct nodes have an edge connecting them.

    .. plot::

        >>> nx.draw(nx.complete_graph(5))

    Parameters
    ----------
    n : int or iterable container of nodes
        If n is an integer, nodes are from range(n).
        If n is a container of nodes, those nodes appear in the graph.
        Warning: n is not checked for duplicates and if present the
        resulting graph may not be as desired. Make sure you have no duplicates.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Examples
    --------
    >>> G = nx.complete_graph(9)
    >>> len(G)
    9
    >>> G.size()
    36
    >>> G = nx.complete_graph(range(11, 14))
    >>> list(G.nodes())
    [11, 12, 13]
    >>> G = nx.complete_graph(4, nx.DiGraph())
    >>> G.is_directed()
    True

    """
    _, nodes = n
    G = empty_graph(nodes, create_using)
    if len(nodes) > 1:
        if G.is_directed():
            edges = itertools.permutations(nodes, 2)
        else:
            edges = itertools.combinations(nodes, 2)
        G.add_edges_from(edges)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def circular_ladder_graph(n, create_using=None):
    """Returns the circular ladder graph $CL_n$ of length n.

    $CL_n$ consists of two concentric n-cycles in which
    each of the n pairs of concentric nodes are joined by an edge.

    Node labels are the integers 0 to n-1

