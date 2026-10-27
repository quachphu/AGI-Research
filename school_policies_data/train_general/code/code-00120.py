            if "unhashable" in str(err):
                msg = " : Could be b/c data=True or your values are unhashable"
                raise TypeError(str(err) + msg) from err
            raise

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        data = self._data
        if data is False:
            return iter(self._nodes)
        if data is True:
            return iter(self._nodes.items())
        return (
            (n, dd[data] if data in dd else self._default)
            for n, dd in self._nodes.items()
        )

    def __contains__(self, n):
        try:
            node_in = n in self._nodes
        except TypeError:
            n, d = n
            return n in self._nodes and self[n] == d
        if node_in is True:
            return node_in
        try:
            n, d = n
        except (TypeError, ValueError):
            return False
        return n in self._nodes and self[n] == d

    def __getitem__(self, n):
        if isinstance(n, slice):
            raise nx.NetworkXError(
                f"{type(self).__name__} does not support slicing, "
                f"try list(G.nodes.data())[{n.start}:{n.stop}:{n.step}]"
            )
        ddict = self._nodes[n]
        data = self._data
        if data is False or data is True:
            return ddict
        return ddict[data] if data in ddict else self._default

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        name = self.__class__.__name__
        if self._data is False:
            return f"{name}({tuple(self)})"
        if self._data is True:
            return f"{name}({dict(self)})"
        return f"{name}({dict(self)}, data={self._data!r})"


# DegreeViews
class DiDegreeView:
    """A View class for degree of nodes in a NetworkX Graph

    The functionality is like dict.items() with (node, degree) pairs.
    Additional functionality includes read-only lookup of node degree,
    and calling with optional features nbunch (for only a subset of nodes)
    and weight (use edge weights to compute degree).

    Parameters
    ==========
    graph : NetworkX graph-like class
    nbunch : node, container of nodes, or None meaning all nodes (default=None)
    weight : bool or string (default=None)

    Notes
    -----
    DegreeView can still lookup any node even if nbunch is specified.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> DV = G.degree()
    >>> assert DV[2] == 1
    >>> assert sum(deg for n, deg in DV) == 4

    >>> DVweight = G.degree(weight="span")
    >>> G.add_edge(1, 2, span=34)
    >>> DVweight[2]
    34
    >>> DVweight[0]  #  default edge weight is 1
    1
    >>> sum(span for n, span in DVweight)  # sum weighted degrees
    70

    >>> DVnbunch = G.degree(nbunch=(1, 2))
    >>> assert len(list(DVnbunch)) == 2  # iteration over nbunch only
    """

    def __init__(self, G, nbunch=None, weight=None):
        self._graph = G
        self._succ = G._succ if hasattr(G, "_succ") else G._adj
        self._pred = G._pred if hasattr(G, "_pred") else G._adj
        self._nodes = self._succ if nbunch is None else list(G.nbunch_iter(nbunch))
        self._weight = weight

    def __call__(self, nbunch=None, weight=None):
        if nbunch is None:
            if weight == self._weight:
                return self
            return self.__class__(self._graph, None, weight)
        try:
            if nbunch in self._nodes:
                if weight == self._weight:
                    return self[nbunch]
                return self.__class__(self._graph, None, weight)[nbunch]
        except TypeError:
            pass
        return self.__class__(self._graph, nbunch, weight)

    def __getitem__(self, n):
        weight = self._weight
        succs = self._succ[n]
        preds = self._pred[n]
        if weight is None:
            return len(succs) + len(preds)
        return sum(dd.get(weight, 1) for dd in succs.values()) + sum(
            dd.get(weight, 1) for dd in preds.values()
        )

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                succs = self._succ[n]
                preds = self._pred[n]
                yield (n, len(succs) + len(preds))
        else:
            for n in self._nodes:
                succs = self._succ[n]
                preds = self._pred[n]
                deg = sum(dd.get(weight, 1) for dd in succs.values()) + sum(
                    dd.get(weight, 1) for dd in preds.values()
                )
                yield (n, deg)

    def __len__(self):
        return len(self._nodes)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return f"{self.__class__.__name__}({dict(self)})"


class DegreeView(DiDegreeView):
    """A DegreeView class to act as G.degree for a NetworkX Graph

    Typical usage focuses on iteration over `(node, degree)` pairs.
    The degree is by default the number of edges incident to the node.
    Optional argument `weight` enables weighted degree using the edge
    attribute named in the `weight` argument.  Reporting and iteration
    can also be restricted to a subset of nodes using `nbunch`.

    Additional functionality include node lookup so that `G.degree[n]`
    reported the (possibly weighted) degree of node `n`. Calling the
    view creates a view with different arguments `nbunch` or `weight`.

    Parameters
    ==========
    graph : NetworkX graph-like class
    nbunch : node, container of nodes, or None meaning all nodes (default=None)
    weight : string or None (default=None)

    Notes
    -----
    DegreeView can still lookup any node even if nbunch is specified.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> DV = G.degree()
    >>> assert DV[2] == 1
    >>> assert G.degree[2] == 1
    >>> assert sum(deg for n, deg in DV) == 4

    >>> DVweight = G.degree(weight="span")
    >>> G.add_edge(1, 2, span=34)
    >>> DVweight[2]
    34
    >>> DVweight[0]  #  default edge weight is 1
    1
    >>> sum(span for n, span in DVweight)  # sum weighted degrees
    70

    >>> DVnbunch = G.degree(nbunch=(1, 2))
    >>> assert len(list(DVnbunch)) == 2  # iteration over nbunch only
    """

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._succ[n]
        if weight is None:
            return len(nbrs) + (n in nbrs)
        return sum(dd.get(weight, 1) for dd in nbrs.values()) + (
            n in nbrs and nbrs[n].get(weight, 1)
        )

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                nbrs = self._succ[n]
                yield (n, len(nbrs) + (n in nbrs))
        else:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(dd.get(weight, 1) for dd in nbrs.values()) + (
                    n in nbrs and nbrs[n].get(weight, 1)
                )
                yield (n, deg)


class OutDegreeView(DiDegreeView):
    """A DegreeView class to report out_degree for a DiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._succ[n]
        if self._weight is None:
            return len(nbrs)
        return sum(dd.get(self._weight, 1) for dd in nbrs.values())

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                succs = self._succ[n]
                yield (n, len(succs))
        else:
            for n in self._nodes:
                succs = self._succ[n]
                deg = sum(dd.get(weight, 1) for dd in succs.values())
                yield (n, deg)


class InDegreeView(DiDegreeView):
    """A DegreeView class to report in_degree for a DiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._pred[n]
        if weight is None:
            return len(nbrs)
        return sum(dd.get(weight, 1) for dd in nbrs.values())

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                preds = self._pred[n]
                yield (n, len(preds))
        else:
            for n in self._nodes:
                preds = self._pred[n]
                deg = sum(dd.get(weight, 1) for dd in preds.values())
                yield (n, deg)


class MultiDegreeView(DiDegreeView):
    """A DegreeView class for undirected multigraphs; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._succ[n]
        if weight is None:
            return sum(len(keys) for keys in nbrs.values()) + (
                n in nbrs and len(nbrs[n])
            )
        # edge weighted graph - degree is sum of nbr edge weights
        deg = sum(
            d.get(weight, 1) for key_dict in nbrs.values() for d in key_dict.values()
        )
        if n in nbrs:
            deg += sum(d.get(weight, 1) for d in nbrs[n].values())
        return deg

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(len(keys) for keys in nbrs.values()) + (
                    n in nbrs and len(nbrs[n])
                )
                yield (n, deg)
        else:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in nbrs.values()
                    for d in key_dict.values()
                )
                if n in nbrs:
                    deg += sum(d.get(weight, 1) for d in nbrs[n].values())
                yield (n, deg)


class DiMultiDegreeView(DiDegreeView):
    """A DegreeView class for MultiDiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        succs = self._succ[n]
        preds = self._pred[n]
        if weight is None:
            return sum(len(keys) for keys in succs.values()) + sum(
                len(keys) for keys in preds.values()
            )
        # edge weighted graph - degree is sum of nbr edge weights
        deg = sum(
            d.get(weight, 1) for key_dict in succs.values() for d in key_dict.values()
        ) + sum(
            d.get(weight, 1) for key_dict in preds.values() for d in key_dict.values()
        )
        return deg

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                succs = self._succ[n]
                preds = self._pred[n]
                deg = sum(len(keys) for keys in succs.values()) + sum(
                    len(keys) for keys in preds.values()
                )
                yield (n, deg)
        else:
            for n in self._nodes:
                succs = self._succ[n]
                preds = self._pred[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in succs.values()
                    for d in key_dict.values()
                ) + sum(
                    d.get(weight, 1)
                    for key_dict in preds.values()
                    for d in key_dict.values()
                )
                yield (n, deg)


class InMultiDegreeView(DiDegreeView):
    """A DegreeView class for inward degree of MultiDiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._pred[n]
        if weight is None:
            return sum(len(data) for data in nbrs.values())
        # edge weighted graph - degree is sum of nbr edge weights
        return sum(
            d.get(weight, 1) for key_dict in nbrs.values() for d in key_dict.values()
        )

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                nbrs = self._pred[n]
                deg = sum(len(data) for data in nbrs.values())
                yield (n, deg)
        else:
            for n in self._nodes:
                nbrs = self._pred[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in nbrs.values()
                    for d in key_dict.values()
                )
                yield (n, deg)


class OutMultiDegreeView(DiDegreeView):
    """A DegreeView class for outward degree of MultiDiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._succ[n]
        if weight is None:
            return sum(len(data) for data in nbrs.values())
        # edge weighted graph - degree is sum of nbr edge weights
        return sum(
            d.get(weight, 1) for key_dict in nbrs.values() for d in key_dict.values()
        )

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(len(data) for data in nbrs.values())
                yield (n, deg)
        else:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in nbrs.values()
                    for d in key_dict.values()
                )
                yield (n, deg)


# A base class for all edge views. Ensures all edge view and edge data view
# objects/classes are captured by `isinstance(obj, EdgeViewABC)` and
# `issubclass(cls, EdgeViewABC)` respectively
class EdgeViewABC(ABC):
    pass


# EdgeDataViews
class OutEdgeDataView(EdgeViewABC):
    """EdgeDataView for outward edges of DiGraph; See EdgeDataView"""

    __slots__ = (
        "_viewer",
        "_nbunch",
        "_data",
        "_default",
        "_adjdict",
        "_nodes_nbrs",
        "_report",
    )

    def __getstate__(self):
        return {
            "viewer": self._viewer,
            "nbunch": self._nbunch,
            "data": self._data,
            "default": self._default,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, viewer, nbunch=None, data=False, *, default=None):
        self._viewer = viewer
        adjdict = self._adjdict = viewer._adjdict
        if nbunch is None:
            self._nodes_nbrs = adjdict.items
        else:
            # dict retains order of nodes but acts like a set
            nbunch = dict.fromkeys(viewer._graph.nbunch_iter(nbunch))
            self._nodes_nbrs = lambda: [(n, adjdict[n]) for n in nbunch]
        self._nbunch = nbunch
        self._data = data
        self._default = default
        # Set _report based on data and default
        if data is True:
            self._report = lambda n, nbr, dd: (n, nbr, dd)
        elif data is False:
            self._report = lambda n, nbr, dd: (n, nbr)
        else:  # data is attribute name
            self._report = (
                lambda n, nbr, dd: (n, nbr, dd[data])
                if data in dd
                else (n, nbr, default)
            )

    def __len__(self):
        return sum(len(nbrs) for n, nbrs in self._nodes_nbrs())

    def __iter__(self):
        return (
            self._report(n, nbr, dd)
            for n, nbrs in self._nodes_nbrs()
            for nbr, dd in nbrs.items()
        )

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and u not in self._nbunch:
            return False  # this edge doesn't start in nbunch
        try:
            ddict = self._adjdict[u][v]
        except KeyError:
            return False
        return e == self._report(u, v, ddict)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)})"


class EdgeDataView(OutEdgeDataView):
    """A EdgeDataView class for edges of Graph

    This view is primarily used to iterate over the edges reporting
    edges as node-tuples with edge data optionally reported. The
    argument `nbunch` allows restriction to edges incident to nodes
    in that container/singleton. The default (nbunch=None)
    reports all edges. The arguments `data` and `default` control
    what edge data is reported. The default `data is False` reports
    only node-tuples for each edge. If `data is True` the entire edge
    data dict is returned. Otherwise `data` is assumed to hold the name
    of the edge attribute to report with default `default` if  that
    edge attribute is not present.

    Parameters
    ----------
    nbunch : container of nodes, node or None (default None)
    data : False, True or string (default False)
    default : default value (default None)

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> G.add_edge(1, 2, foo="bar")
    >>> list(G.edges(data="foo", default="biz"))
    [(0, 1, 'biz'), (1, 2, 'bar')]
    >>> assert (0, 1, "biz") in G.edges(data="foo", default="biz")
    """

    __slots__ = ()

    def __len__(self):
        return sum(1 for e in self)

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr, dd in nbrs.items():
                if nbr not in seen:
                    yield self._report(n, nbr, dd)
            seen[n] = 1
        del seen

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and u not in self._nbunch and v not in self._nbunch:
            return False  # this edge doesn't start and it doesn't end in nbunch
        try:
            ddict = self._adjdict[u][v]
        except KeyError:
            return False
        return e == self._report(u, v, ddict)


class InEdgeDataView(OutEdgeDataView):
    """An EdgeDataView class for outward edges of DiGraph; See EdgeDataView"""

    __slots__ = ()

    def __iter__(self):
        return (
            self._report(nbr, n, dd)
            for n, nbrs in self._nodes_nbrs()
            for nbr, dd in nbrs.items()
        )

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and v not in self._nbunch:
            return False  # this edge doesn't end in nbunch
        try:
            ddict = self._adjdict[v][u]
        except KeyError:
            return False
        return e == self._report(u, v, ddict)


class OutMultiEdgeDataView(OutEdgeDataView):
    """An EdgeDataView for outward edges of MultiDiGraph; See EdgeDataView"""

    __slots__ = ("keys",)

    def __getstate__(self):
        return {
            "viewer": self._viewer,
            "nbunch": self._nbunch,
            "keys": self.keys,
            "data": self._data,
            "default": self._default,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, viewer, nbunch=None, data=False, *, default=None, keys=False):
        self._viewer = viewer
        adjdict = self._adjdict = viewer._adjdict
        self.keys = keys
        if nbunch is None:
            self._nodes_nbrs = adjdict.items
        else:
            # dict retains order of nodes but acts like a set
            nbunch = dict.fromkeys(viewer._graph.nbunch_iter(nbunch))
            self._nodes_nbrs = lambda: [(n, adjdict[n]) for n in nbunch]
        self._nbunch = nbunch
        self._data = data
        self._default = default
        # Set _report based on data and default
        if data is True:
            if keys is True:
                self._report = lambda n, nbr, k, dd: (n, nbr, k, dd)
            else:
                self._report = lambda n, nbr, k, dd: (n, nbr, dd)
        elif data is False:
            if keys is True:
                self._report = lambda n, nbr, k, dd: (n, nbr, k)
            else:
                self._report = lambda n, nbr, k, dd: (n, nbr)
        else:  # data is attribute name
            if keys is True:
                self._report = (
                    lambda n, nbr, k, dd: (n, nbr, k, dd[data])
                    if data in dd
                    else (n, nbr, k, default)
                )
            else:
                self._report = (
                    lambda n, nbr, k, dd: (n, nbr, dd[data])
                    if data in dd
                    else (n, nbr, default)
                )

    def __len__(self):
        return sum(1 for e in self)

    def __iter__(self):
        return (
            self._report(n, nbr, k, dd)
            for n, nbrs in self._nodes_nbrs()
            for nbr, kd in nbrs.items()
            for k, dd in kd.items()
        )

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and u not in self._nbunch:
            return False  # this edge doesn't start in nbunch
        try:
            kdict = self._adjdict[u][v]
        except KeyError:
            return False
        if self.keys is True:
            k = e[2]
            try:
                dd = kdict[k]
            except KeyError:
                return False
            return e == self._report(u, v, k, dd)
        return any(e == self._report(u, v, k, dd) for k, dd in kdict.items())


class MultiEdgeDataView(OutMultiEdgeDataView):
    """An EdgeDataView class for edges of MultiGraph; See EdgeDataView"""

    __slots__ = ()

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr, kd in nbrs.items():
                if nbr not in seen:
                    for k, dd in kd.items():
                        yield self._report(n, nbr, k, dd)
            seen[n] = 1
        del seen

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and u not in self._nbunch and v not in self._nbunch:
            return False  # this edge doesn't start and doesn't end in nbunch
        try:
            kdict = self._adjdict[u][v]
        except KeyError:
            try:
                kdict = self._adjdict[v][u]
            except KeyError:
                return False
        if self.keys is True:
            k = e[2]
            try:
                dd = kdict[k]
            except KeyError:
                return False
            return e == self._report(u, v, k, dd)
        return any(e == self._report(u, v, k, dd) for k, dd in kdict.items())


class InMultiEdgeDataView(OutMultiEdgeDataView):
    """An EdgeDataView for inward edges of MultiDiGraph; See EdgeDataView"""

    __slots__ = ()

    def __iter__(self):
        return (
            self._report(nbr, n, k, dd)
            for n, nbrs in self._nodes_nbrs()
            for nbr, kd in nbrs.items()
            for k, dd in kd.items()
        )

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and v not in self._nbunch:
            return False  # this edge doesn't end in nbunch
        try:
            kdict = self._adjdict[v][u]
        except KeyError:
            return False
        if self.keys is True:
            k = e[2]
            dd = kdict[k]
            return e == self._report(u, v, k, dd)
        return any(e == self._report(u, v, k, dd) for k, dd in kdict.items())


# EdgeViews    have set operations and no data reported
class OutEdgeView(Set, Mapping, EdgeViewABC):
    """A EdgeView class for outward edges of a DiGraph"""

    __slots__ = ("_adjdict", "_graph", "_nodes_nbrs")

    def __getstate__(self):
        return {"_graph": self._graph, "_adjdict": self._adjdict}

    def __setstate__(self, state):
        self._graph = state["_graph"]
        self._adjdict = state["_adjdict"]
        self._nodes_nbrs = self._adjdict.items

    @classmethod
    def _from_iterable(cls, it):
        return set(it)

    dataview = OutEdgeDataView

    def __init__(self, G):
        self._graph = G
        self._adjdict = G._succ if hasattr(G, "succ") else G._adj
        self._nodes_nbrs = self._adjdict.items

    # Set methods
    def __len__(self):
        return sum(len(nbrs) for n, nbrs in self._nodes_nbrs())

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr in nbrs:
                yield (n, nbr)

    def __contains__(self, e):
        try:
            u, v = e
            return v in self._adjdict[u]
        except KeyError:
            return False

    # Mapping Methods
    def __getitem__(self, e):
        if isinstance(e, slice):
            raise nx.NetworkXError(
                f"{type(self).__name__} does not support slicing, "
                f"try list(G.edges)[{e.start}:{e.stop}:{e.step}]"
            )
        u, v = e
        try:
            return self._adjdict[u][v]
        except KeyError as ex:  # Customize msg to indicate exception origin
            raise KeyError(f"The edge {e} is not in the graph.")

    # EdgeDataView methods
    def __call__(self, nbunch=None, data=False, *, default=None):
        if nbunch is None and data is False:
            return self
        return self.dataview(self, nbunch, data, default=default)

    def data(self, data=True, default=None, nbunch=None):
        """
        Return a read-only view of edge data.

        Parameters
        ----------
        data : bool or edge attribute key
            If ``data=True``, then the data view maps each edge to a dictionary
            containing all of its attributes. If `data` is a key in the edge
            dictionary, then the data view maps each edge to its value for
            the keyed attribute. In this case, if the edge doesn't have the
            attribute, the `default` value is returned.
        default : object, default=None
            The value used when an edge does not have a specific attribute
        nbunch : container of nodes, optional (default=None)
            Allows restriction to edges only involving certain nodes. All edges
            are considered by default.

        Returns
        -------
        dataview
            Returns an `EdgeDataView` for undirected Graphs, `OutEdgeDataView`
            for DiGraphs, `MultiEdgeDataView` for MultiGraphs and
            `OutMultiEdgeDataView` for MultiDiGraphs.

        Notes
        -----
        If ``data=False``, returns an `EdgeView` without any edge data.

        See Also
        --------
        EdgeDataView
        OutEdgeDataView
        MultiEdgeDataView
        OutMultiEdgeDataView

        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_edges_from(
        ...     [
        ...         (0, 1, {"dist": 3, "capacity": 20}),
        ...         (1, 2, {"dist": 4}),
        ...         (2, 0, {"dist": 5}),
        ...     ]
        ... )

        Accessing edge data with ``data=True`` (the default) returns an
        edge data view object listing each edge with all of its attributes:

        >>> G.edges.data()
        EdgeDataView([(0, 1, {'dist': 3, 'capacity': 20}), (0, 2, {'dist': 5}), (1, 2, {'dist': 4})])

        If `data` represents a key in the edge attribute dict, a dataview listing
        each edge with its value for that specific key is returned:

        >>> G.edges.data("dist")
        EdgeDataView([(0, 1, 3), (0, 2, 5), (1, 2, 4)])

        `nbunch` can be used to limit the edges:

        >>> G.edges.data("dist", nbunch=[0])
        EdgeDataView([(0, 1, 3), (0, 2, 5)])

        If a specific key is not found in an edge attribute dict, the value
        specified by `default` is used:

        >>> G.edges.data("capacity")
        EdgeDataView([(0, 1, 20), (0, 2, None), (1, 2, None)])

        Note that there is no check that the `data` key is present in any of
        the edge attribute dictionaries:

        >>> G.edges.data("speed")
        EdgeDataView([(0, 1, None), (0, 2, None), (1, 2, None)])
        """
        if nbunch is None and data is False:
            return self
        return self.dataview(self, nbunch, data, default=default)

    # String Methods
    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)})"


class EdgeView(OutEdgeView):
    """A EdgeView class for edges of a Graph

    This densely packed View allows iteration over edges, data lookup
    like a dict and set operations on edges represented by node-tuples.
    In addition, edge data can be controlled by calling this object
    possibly creating an EdgeDataView. Typically edges are iterated over
    and reported as `(u, v)` node tuples or `(u, v, key)` node/key tuples
    for multigraphs. Those edge representations can also be using to
    lookup the data dict for any edge. Set operations also are available
    where those tuples are the elements of the set.
    Calling this object with optional arguments `data`, `default` and `keys`
    controls the form of the tuple (see EdgeDataView). Optional argument
    `nbunch` allows restriction to edges only involving certain nodes.

    If `data is False` (the default) then iterate over 2-tuples `(u, v)`.
    If `data is True` iterate over 3-tuples `(u, v, datadict)`.
    Otherwise iterate over `(u, v, datadict.get(data, default))`.
    For Multigraphs, if `keys is True`, replace `u, v` with `u, v, key` above.

    Parameters
    ==========
    graph : NetworkX graph-like class
    nbunch : (default= all nodes in graph) only report edges with these nodes
    keys : (only for MultiGraph. default=False) report edge key in tuple
    data : bool or string (default=False) see above
    default : object (default=None)

    Examples
    ========
    >>> G = nx.path_graph(4)
    >>> EV = G.edges()
    >>> (2, 3) in EV
    True
    >>> for u, v in EV:
    ...     print((u, v))
    (0, 1)
    (1, 2)
    (2, 3)
    >>> assert EV & {(1, 2), (3, 4)} == {(1, 2)}

    >>> EVdata = G.edges(data="color", default="aqua")
    >>> G.add_edge(2, 3, color="blue")
    >>> assert (2, 3, "blue") in EVdata
    >>> for u, v, c in EVdata:
    ...     print(f"({u}, {v}) has color: {c}")
    (0, 1) has color: aqua
    (1, 2) has color: aqua
    (2, 3) has color: blue

    >>> EVnbunch = G.edges(nbunch=2)
    >>> assert (2, 3) in EVnbunch
    >>> assert (0, 1) not in EVnbunch
    >>> for u, v in EVnbunch:
    ...     assert u == 2 or v == 2

    >>> MG = nx.path_graph(4, create_using=nx.MultiGraph)
    >>> EVmulti = MG.edges(keys=True)
    >>> (2, 3, 0) in EVmulti
    True
    >>> (2, 3) in EVmulti  # 2-tuples work even when keys is True
    True
    >>> key = MG.add_edge(2, 3)
    >>> for u, v, k in EVmulti:
    ...     print((u, v, k))
    (0, 1, 0)
    (1, 2, 0)
    (2, 3, 0)
    (2, 3, 1)
    """

    __slots__ = ()

    dataview = EdgeDataView

    def __len__(self):
        num_nbrs = (len(nbrs) + (n in nbrs) for n, nbrs in self._nodes_nbrs())
        return sum(num_nbrs) // 2

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr in list(nbrs):
                if nbr not in seen:
                    yield (n, nbr)
            seen[n] = 1
        del seen

    def __contains__(self, e):
        try:
            u, v = e[:2]
            return v in self._adjdict[u] or u in self._adjdict[v]
        except (KeyError, ValueError):
            return False


class InEdgeView(OutEdgeView):
    """A EdgeView class for inward edges of a DiGraph"""

    __slots__ = ()

    def __setstate__(self, state):
        self._graph = state["_graph"]
        self._adjdict = state["_adjdict"]
        self._nodes_nbrs = self._adjdict.items

    dataview = InEdgeDataView

    def __init__(self, G):
        self._graph = G
        self._adjdict = G._pred if hasattr(G, "pred") else G._adj
        self._nodes_nbrs = self._adjdict.items

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr in nbrs:
                yield (nbr, n)

    def __contains__(self, e):
        try:
            u, v = e
            return u in self._adjdict[v]
        except KeyError:
            return False

    def __getitem__(self, e):
        if isinstance(e, slice):
            raise nx.NetworkXError(
                f"{type(self).__name__} does not support slicing, "
                f"try list(G.in_edges)[{e.start}:{e.stop}:{e.step}]"
            )
        u, v = e
        return self._adjdict[v][u]


class OutMultiEdgeView(OutEdgeView):
    """A EdgeView class for outward edges of a MultiDiGraph"""

    __slots__ = ()

    dataview = OutMultiEdgeDataView

    def __len__(self):
        return sum(
            len(kdict) for n, nbrs in self._nodes_nbrs() for nbr, kdict in nbrs.items()
        )

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr, kdict in nbrs.items():
                for key in kdict:
                    yield (n, nbr, key)

    def __contains__(self, e):
        N = len(e)
        if N == 3:
            u, v, k = e
        elif N == 2:
            u, v = e
            k = 0
        else:
            raise ValueError("MultiEdge must have length 2 or 3")
        try:
            return k in self._adjdict[u][v]
        except KeyError:
            return False

    def __getitem__(self, e):
        if isinstance(e, slice):
            raise nx.NetworkXError(
                f"{type(self).__name__} does not support slicing, "
                f"try list(G.edges)[{e.start}:{e.stop}:{e.step}]"
            )
        u, v, k = e
        return self._adjdict[u][v][k]

    def __call__(self, nbunch=None, data=False, *, default=None, keys=False):
        if nbunch is None and data is False and keys is True:
            return self
        return self.dataview(self, nbunch, data, default=default, keys=keys)

    def data(self, data=True, default=None, nbunch=None, keys=False):
        if nbunch is None and data is False and keys is True:
            return self
        return self.dataview(self, nbunch, data, default=default, keys=keys)


class MultiEdgeView(OutMultiEdgeView):
    """A EdgeView class for edges of a MultiGraph"""

    __slots__ = ()

    dataview = MultiEdgeDataView

    def __len__(self):
        return sum(1 for e in self)

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr, kd in nbrs.items():
                if nbr not in seen:
                    for k, dd in kd.items():
                        yield (n, nbr, k)
            seen[n] = 1
        del seen


class InMultiEdgeView(OutMultiEdgeView):
    """A EdgeView class for inward edges of a MultiDiGraph"""

    __slots__ = ()

    def __setstate__(self, state):
        self._graph = state["_graph"]
        self._adjdict = state["_adjdict"]
        self._nodes_nbrs = self._adjdict.items

    dataview = InMultiEdgeDataView

    def __init__(self, G):
        self._graph = G
        self._adjdict = G._pred if hasattr(G, "pred") else G._adj
        self._nodes_nbrs = self._adjdict.items

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr, kdict in nbrs.items():
                for key in kdict:
                    yield (nbr, n, key)

    def __contains__(self, e):
        N = len(e)
        if N == 3:
            u, v, k = e
        elif N == 2:
            u, v = e
            k = 0
        else:
            raise ValueError("MultiEdge must have length 2 or 3")
        try:
            return k in self._adjdict[v][u]
        except KeyError:
            return False

    def __getitem__(self, e):
        if isinstance(e, slice):
            raise nx.NetworkXError(
                f"{type(self).__name__} does not support slicing, "
                f"try list(G.in_edges)[{e.start}:{e.stop}:{e.step}]"
            )
        u, v, k = e
        return self._adjdict[v][u][k]
<|endoftext|>"""
Cuthill-McKee ordering of graph nodes to produce sparse matrices
"""

from collections import deque
from operator import itemgetter

import networkx as nx

from ..utils import arbitrary_element

__all__ = ["cuthill_mckee_ordering", "reverse_cuthill_mckee_ordering"]


def cuthill_mckee_ordering(G, heuristic=None):
    """Generate an ordering (permutation) of the graph nodes to make
    a sparse matrix.

    Uses the Cuthill-McKee heuristic (based on breadth-first search) [1]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    heuristic : function, optional
      Function to choose starting node for RCM algorithm.  If None
      a node from a pseudo-peripheral pair is used.  A user-defined function
      can be supplied that takes a graph object and returns a single node.

    Returns
    -------
    nodes : generator
       Generator of nodes in Cuthill-McKee ordering.

    Examples
    --------
    >>> from networkx.utils import cuthill_mckee_ordering
    >>> G = nx.path_graph(4)
    >>> rcm = list(cuthill_mckee_ordering(G))
    >>> A = nx.adjacency_matrix(G, nodelist=rcm)

    Smallest degree node as heuristic function:

    >>> def smallest_degree(G):
    ...     return min(G, key=G.degree)
    >>> rcm = list(cuthill_mckee_ordering(G, heuristic=smallest_degree))


    See Also
    --------
    reverse_cuthill_mckee_ordering

    Notes
    -----
    The optimal solution the bandwidth reduction is NP-complete [2]_.


    References
    ----------
    .. [1] E. Cuthill and J. McKee.
       Reducing the bandwidth of sparse symmetric matrices,
       In Proc. 24th Nat. Conf. ACM, pages 157-172, 1969.
       http://doi.acm.org/10.1145/800195.805928
    .. [2]  Steven S. Skiena. 1997. The Algorithm Design Manual.
       Springer-Verlag New York, Inc., New York, NY, USA.
    """
    for c in nx.connected_components(G):
        yield from connected_cuthill_mckee_ordering(G.subgraph(c), heuristic)


def reverse_cuthill_mckee_ordering(G, heuristic=None):
    """Generate an ordering (permutation) of the graph nodes to make
    a sparse matrix.

    Uses the reverse Cuthill-McKee heuristic (based on breadth-first search)
    [1]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    heuristic : function, optional
      Function to choose starting node for RCM algorithm.  If None
      a node from a pseudo-peripheral pair is used.  A user-defined function
      can be supplied that takes a graph object and returns a single node.

    Returns
    -------
    nodes : generator
       Generator of nodes in reverse Cuthill-McKee ordering.

    Examples
    --------
    >>> from networkx.utils import reverse_cuthill_mckee_ordering
    >>> G = nx.path_graph(4)
    >>> rcm = list(reverse_cuthill_mckee_ordering(G))
    >>> A = nx.adjacency_matrix(G, nodelist=rcm)

    Smallest degree node as heuristic function:

    >>> def smallest_degree(G):
    ...     return min(G, key=G.degree)
    >>> rcm = list(reverse_cuthill_mckee_ordering(G, heuristic=smallest_degree))


    See Also
    --------
    cuthill_mckee_ordering

    Notes
    -----
    The optimal solution the bandwidth reduction is NP-complete [2]_.

    References
    ----------
    .. [1] E. Cuthill and J. McKee.
       Reducing the bandwidth of sparse symmetric matrices,
       In Proc. 24th Nat. Conf. ACM, pages 157-72, 1969.
       http://doi.acm.org/10.1145/800195.805928
    .. [2]  Steven S. Skiena. 1997. The Algorithm Design Manual.
       Springer-Verlag New York, Inc., New York, NY, USA.
    """
    return reversed(list(cuthill_mckee_ordering(G, heuristic=heuristic)))


def connected_cuthill_mckee_ordering(G, heuristic=None):
    # the cuthill mckee algorithm for connected graphs
    if heuristic is None:
        start = pseudo_peripheral_node(G)
    else:
        start = heuristic(G)
    visited = {start}
    queue = deque([start])
    while queue:
        parent = queue.popleft()
        yield parent
        nd = sorted(G.degree(set(G[parent]) - visited), key=itemgetter(1))
        children = [n for n, d in nd]
        visited.update(children)
        queue.extend(children)


def pseudo_peripheral_node(G):
    # helper for cuthill-mckee to find a node in a "pseudo peripheral pair"
    # to use as good starting node
    u = arbitrary_element(G)
    lp = 0
    v = u
    while True:
        spl = dict(nx.shortest_path_length(G, v))
        l = max(spl.values())
        if l <= lp:
            break
        lp = l
        farthest = (n for n, dist in spl.items() if dist == l)
        v, deg = min(G.degree(farthest), key=itemgetter(1))
    return v
<|endoftext|>import bz2
import collections
import gzip
import inspect
import itertools
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter, signature
from os.path import splitext
from pathlib import Path

import networkx as nx
from networkx.utils import create_py_random_state, create_random_state

__all__ = [
    "not_implemented_for",
    "open_file",
    "nodes_or_number",
    "np_random_state",
    "py_random_state",
    "argmap",
]


def not_implemented_for(*graph_types):
    """Decorator to mark algorithms as not implemented

    Parameters
    ----------
    graph_types : container of strings
        Entries must be one of "directed", "undirected", "multigraph", or "graph".

    Returns
    -------
    _require : function
        The decorated function.

    Raises
    ------
    NetworkXNotImplemented
    If any of the packages cannot be imported

    Notes
    -----
    Multiple types are joined logically with "and".
    For "or" use multiple @not_implemented_for() lines.

    Examples
    --------
    Decorate functions like this::

       @not_implemented_for("directed")
       def sp_function(G):
           pass


       # rule out MultiDiGraph
       @not_implemented_for("directed", "multigraph")
       def sp_np_function(G):
           pass


       # rule out all except DiGraph
       @not_implemented_for("undirected")
       @not_implemented_for("multigraph")
       def sp_np_function(G):
           pass
    """
    if ("directed" in graph_types) and ("undirected" in graph_types):
        raise ValueError("Function not implemented on directed AND undirected graphs?")
    if ("multigraph" in graph_types) and ("graph" in graph_types):
        raise ValueError("Function not implemented on graph AND multigraphs?")
    if not set(graph_types) < {"directed", "undirected", "multigraph", "graph"}:
        raise KeyError(
            "use one or more of directed, undirected, multigraph, graph.  "
            f"You used {graph_types}"
        )

    # 3-way logic: True if "directed" input, False if "undirected" input, else None
    dval = ("directed" in graph_types) or "undirected" not in graph_types and None
    mval = ("multigraph" in graph_types) or "graph" not in graph_types and None
    errmsg = f"not implemented for {' '.join(graph_types)} type"

    def _not_implemented_for(g):
        if (mval is None or mval == g.is_multigraph()) and (
            dval is None or dval == g.is_directed()
        ):
            raise nx.NetworkXNotImplemented(errmsg)

        return g

    return argmap(_not_implemented_for, 0)


# To handle new extensions, define a function accepting a `path` and `mode`.
# Then add the extension to _dispatch_dict.
fopeners = {
    ".gz": gzip.open,
    ".gzip": gzip.open,
    ".bz2": bz2.BZ2File,
}
_dispatch_dict = defaultdict(lambda: open, **fopeners)


def open_file(path_arg, mode="r"):
    """Decorator to ensure clean opening and closing of files.

    Parameters
    ----------
    path_arg : string or int
        Name or index of the argument that is a path.

    mode : str
        String for opening mode.

    Returns
    -------
    _open_file : function
        Function which cleanly executes the io.

    Examples
    --------
    Decorate functions like this::

       @open_file(0, "r")
       def read_function(pathname):
           pass


       @open_file(1, "w")
       def write_function(G, pathname):
           pass


       @open_file(1, "w")
       def write_function(G, pathname="graph.dot"):
           pass


       @open_file("pathname", "w")
       def write_function(G, pathname="graph.dot"):
           pass


       @open_file("path", "w+")
       def another_function(arg, **kwargs):
           path = kwargs["path"]
           pass

    Notes
    -----
    Note that this decorator solves the problem when a path argument is
    specified as a string, but it does not handle the situation when the
    function wants to accept a default of None (and then handle it).

    Here is an example of how to handle this case::

      @open_file("path")
      def some_function(arg1, arg2, path=None):
          if path is None:
              fobj = tempfile.NamedTemporaryFile(delete=False)
          else:
              # `path` could have been a string or file object or something
              # similar. In any event, the decorator has given us a file object
              # and it will close it for us, if it should.
              fobj = path

          try:
              fobj.write("blah")
          finally:
              if path is None:
                  fobj.close()

    Normally, we'd want to use "with" to ensure that fobj gets closed.
    However, the decorator will make `path` a file object for us,
    and using "with" would undesirably close that file object.
    Instead, we use a try block, as shown above.
    When we exit the function, fobj will be closed, if it should be, by the decorator.
    """

    def _open_file(path):
        # Now we have the path_arg. There are two types of input to consider:
        #   1) string representing a path that should be opened
        #   2) an already opened file object
        if isinstance(path, str):
            ext = splitext(path)[1]
        elif isinstance(path, Path):
            # path is a pathlib reference to a filename
            ext = path.suffix
            path = str(path)
        else:
            # could be None, or a file handle, in which case the algorithm will deal with it
            return path, lambda: None

        fobj = _dispatch_dict[ext](path, mode=mode)
        return fobj, lambda: fobj.close()

    return argmap(_open_file, path_arg, try_finally=True)


def nodes_or_number(which_args):
    """Decorator to allow number of nodes or container of nodes.

    With this decorator, the specified argument can be either a number or a container
    of nodes. If it is a number, the nodes used are `range(n)`.
    This allows `nx.complete_graph(50)` in place of `nx.complete_graph(list(range(50)))`.
    And it also allows `nx.complete_graph(any_list_of_nodes)`.

    Parameters
    ----------
    which_args : string or int or sequence of strings or ints
        If string, the name of the argument to be treated.
        If int, the index of the argument to be treated.
        If more than one node argument is allowed, can be a list of locations.

    Returns
    -------
    _nodes_or_numbers : function
        Function which replaces int args with ranges.

    Examples
    --------
    Decorate functions like this::

       @nodes_or_number("nodes")
       def empty_graph(nodes):
           # nodes is converted to a list of nodes

       @nodes_or_number(0)
       def empty_graph(nodes):
           # nodes is converted to a list of nodes

       @nodes_or_number(["m1", "m2"])
       def grid_2d_graph(m1, m2, periodic=False):
           # m1 and m2 are each converted to a list of nodes

       @nodes_or_number([0, 1])
       def grid_2d_graph(m1, m2, periodic=False):
           # m1 and m2 are each converted to a list of nodes

       @nodes_or_number(1)
       def full_rary_tree(r, n)
           # presumably r is a number. It is not handled by this decorator.
           # n is converted to a list of nodes
    """

    def _nodes_or_number(n):
        try:
            nodes = list(range(n))
        except TypeError:
            nodes = tuple(n)
        else:
            if n < 0:
                raise nx.NetworkXError(f"Negative number of nodes not valid: {n}")
        return (n, nodes)

    try:
        iter_wa = iter(which_args)
    except TypeError:
        iter_wa = (which_args,)

    return argmap(_nodes_or_number, *iter_wa)


def np_random_state(random_state_argument):
    """Decorator to generate a numpy RandomState or Generator instance.

    The decorator processes the argument indicated by `random_state_argument`
    using :func:`nx.utils.create_random_state`.
    The argument value can be a seed (integer), or a `numpy.random.RandomState`
    or `numpy.random.RandomState` instance or (`None` or `numpy.random`).
    The latter two options use the global random number generator for `numpy.random`.

    The returned instance is a `numpy.random.RandomState` or `numpy.random.Generator`.

    Parameters
    ----------
    random_state_argument : string or int
        The name or index of the argument to be converted
        to a `numpy.random.RandomState` instance.

    Returns
    -------
    _random_state : function
        Function whose random_state keyword argument is a RandomState instance.

    Examples
    --------
    Decorate functions like this::

       @np_random_state("seed")
       def random_float(seed=None):
           return seed.rand()


       @np_random_state(0)
       def random_float(rng=None):
           return rng.rand()


       @np_random_state(1)
       def random_array(dims, random_state=1):
           return random_state.rand(*dims)

    See Also
    --------
    py_random_state
    """
    return argmap(create_random_state, random_state_argument)


def py_random_state(random_state_argument):
    """Decorator to generate a random.Random instance (or equiv).

    This decorator processes `random_state_argument` using
    :func:`nx.utils.create_py_random_state`.
    The input value can be a seed (integer), or a random number generator::

        If int, return a random.Random instance set with seed=int.
        If random.Random instance, return it.
        If None or the `random` package, return the global random number
        generator used by `random`.
        If np.random package, or the default numpy RandomState instance,
        return the default numpy random number generator wrapped in a
        `PythonRandomViaNumpyBits`  class.
        If np.random.Generator instance, return it wrapped in a
        `PythonRandomViaNumpyBits`  class.

        # Legacy options
        If np.random.RandomState instance, return it wrapped in a
        `PythonRandomInterface` class.
        If a `PythonRandomInterface` instance, return it

    Parameters
    ----------
    random_state_argument : string or int
        The name of the argument or the index of the argument in args that is
        to be converted to the random.Random instance or numpy.random.RandomState
        instance that mimics basic methods of random.Random.

    Returns
    -------
    _random_state : function
        Function whose random_state_argument is converted to a Random instance.

    Examples
    --------
    Decorate functions like this::

       @py_random_state("random_state")
       def random_float(random_state=None):
           return random_state.rand()


       @py_random_state(0)
       def random_float(rng=None):
           return rng.rand()


       @py_random_state(1)
       def random_array(dims, seed=12345):
           return seed.rand(*dims)

    See Also
    --------
    np_random_state
    """

    return argmap(create_py_random_state, random_state_argument)


class argmap:
    """A decorator to apply a map to arguments before calling the function

    This class provides a decorator that maps (transforms) arguments of the function
    before the function is called. Thus for example, we have similar code
    in many functions to determine whether an argument is the number of nodes
    to be created, or a list of nodes to be handled. The decorator provides
    the code to accept either -- transforming the indicated argument into a
    list of nodes before the actual function is called.

    This decorator class allows us to process single or multiple arguments.
    The arguments to be processed can be specified by string, naming the argument,
    or by index, specifying the item in the args list.

    Parameters
    ----------
    func : callable
        The function to apply to arguments

    *args : iterable of (int, str or tuple)
        A list of parameters, specified either as strings (their names), ints
        (numerical indices) or tuples, which may contain ints, strings, and
        (recursively) tuples. Each indicates which parameters the decorator
        should map. Tuples indicate that the map function takes (and returns)
        multiple parameters in the same order and nested structure as indicated
        here.

    try_finally : bool (default: False)
        When True, wrap the function call in a try-finally block with code
        for the finally block created by `func`. This is used when the map
        function constructs an object (like a file handle) that requires
        post-processing (like closing).

        Note: try_finally decorators cannot be used to decorate generator
        functions.

    Examples
    --------
    Most of these examples use `@argmap(...)` to apply the decorator to
    the function defined on the next line.
    In the NetworkX codebase however, `argmap` is used within a function to
    construct a decorator. That is, the decorator defines a mapping function
    and then uses `argmap` to build and return a decorated function.
    A simple example is a decorator that specifies which currency to report money.
    The decorator (named `convert_to`) would be used like::

        @convert_to("US_Dollars", "income")
        def show_me_the_money(name, income):
            print(f"{name} : {income}")

    And the code to create the decorator might be::

        def convert_to(currency, which_arg):
            def _convert(amount):
                if amount.currency != currency:
                    amount = amount.to_currency(currency)
                return amount

            return argmap(_convert, which_arg)

    Despite this common idiom for argmap, most of the following examples
    use the `@argmap(...)` idiom to save space.

    Here's an example use of argmap to sum the elements of two of the functions
    arguments. The decorated function::

        @argmap(sum, "xlist", "zlist")
        def foo(xlist, y, zlist):
            return xlist - y + zlist

    is syntactic sugar for::

        def foo(xlist, y, zlist):
            x = sum(xlist)
            z = sum(zlist)
            return x - y + z

    and is equivalent to (using argument indexes)::

        @argmap(sum, "xlist", 2)
        def foo(xlist, y, zlist):
            return xlist - y + zlist

    or::

        @argmap(sum, "zlist", 0)
        def foo(xlist, y, zlist):
            return xlist - y + zlist

    Transforming functions can be applied to multiple arguments, such as::

        def swap(x, y):
            return y, x

        # the 2-tuple tells argmap that the map `swap` has 2 inputs/outputs.
        @argmap(swap, ("a", "b")):
        def foo(a, b, c):
            return a / b * c

    is equivalent to::

        def foo(a, b, c):
            a, b = swap(a, b)
            return a / b * c

    More generally, the applied arguments can be nested tuples of strings or ints.
    The syntax `@argmap(some_func, ("a", ("b", "c")))` would expect `some_func` to
    accept 2 inputs with the second expected to be a 2-tuple. It should then return
    2 outputs with the second a 2-tuple. The returns values would replace input "a"
    "b" and "c" respectively. Similarly for `@argmap(some_func, (0, ("b", 2)))`.

    Also, note that an index larger than the number of named parameters is allowed
    for variadic functions. For example::

        def double(a):
            return 2 * a


        @argmap(double, 3)
        def overflow(a, *args):
            return a, args


        print(overflow(1, 2, 3, 4, 5, 6))  # output is 1, (2, 3, 8, 5, 6)

    **Try Finally**

    Additionally, this `argmap` class can be used to create a decorator that
    initiates a try...finally block. The decorator must be written to return
    both the transformed argument and a closing function.
    This feature was included to enable the `open_file` decorator which might
    need to close the file or not depending on whether it had to open that file.
    This feature uses the keyword-only `try_finally` argument to `@argmap`.

    For example this map opens a file and then makes sure it is closed::

        def open_file(fn):
            f = open(fn)
            return f, lambda: f.close()

    The decorator applies that to the function `foo`::

        @argmap(open_file, "file", try_finally=True)
        def foo(file):
            print(file.read())

    is syntactic sugar for::

        def foo(file):
            file, close_file = open_file(file)
            try:
                print(file.read())
            finally:
                close_file()

    and is equivalent to (using indexes)::

        @argmap(open_file, 0, try_finally=True)
        def foo(file):
            print(file.read())

    Here's an example of the try_finally feature used to create a decorator::

        def my_closing_decorator(which_arg):
            def _opener(path):
                if path is None:
                    path = open(path)
                    fclose = path.close
                else:
                    # assume `path` handles the closing
                    fclose = lambda: None
                return path, fclose

            return argmap(_opener, which_arg, try_finally=True)

    which can then be used as::

        @my_closing_decorator("file")
        def fancy_reader(file=None):
            # this code doesn't need to worry about closing the file
            print(file.read())

    Decorators with try_finally = True cannot be used with generator functions,
    because the `finally` block is evaluated before the generator is exhausted::

        @argmap(open_file, "file", try_finally=True)
        def file_to_lines(file):
            for line in file.readlines():
                yield line

    is equivalent to::

        def file_to_lines_wrapped(file):
            for line in file.readlines():
                yield line


        def file_to_lines_wrapper(file):
            try:
                file = open_file(file)
                return file_to_lines_wrapped(file)
            finally:
                file.close()

    which behaves similarly to::

        def file_to_lines_whoops(file):
            file = open_file(file)
            file.close()
            for line in file.readlines():
                yield line

    because the `finally` block of `file_to_lines_wrapper` is executed before
    the caller has a chance to exhaust the iterator.

    Notes
    -----
    An object of this class is callable and intended to be used when
    defining a decorator. Generally, a decorator takes a function as input
    and constructs a function as output. Specifically, an `argmap` object
    returns the input function decorated/wrapped so that specified arguments
    are mapped (transformed) to new values before the decorated function is called.

    As an overview, the argmap object returns a new function with all the
    dunder values of the original function (like `__doc__`, `__name__`, etc).
    Code for this decorated function is built based on the original function's
    signature. It starts by mapping the input arguments to potentially new
    values. Then it calls the decorated function with these new values in place
    of the indicated arguments that have been mapped. The return value of the
    original function is then returned. This new function is the function that
    is actually called by the user.

    Three additional features are provided.
        1) The code is lazily compiled. That is, the new function is returned
        as an object without the code compiled, but with all information
        needed so it can be compiled upon it's first invocation. This saves
        time on import at the cost of additional time on the first call of
        the function. Subsequent calls are then just as fast as normal.

        2) If the "try_finally" keyword-only argument is True, a try block
        follows each mapped argument, matched on the other side of the wrapped
        call, by a finally block closing that mapping.  We expect func to return
        a 2-tuple: the mapped value and a function to be called in the finally
        clause.  This feature was included so the `open_file` decorator could
        provide a file handle to the decorated function and close the file handle
        after the function call. It even keeps track of whether to close the file
        handle or not based on whether it had to open the file or the input was
        already open. So, the decorated function does not need to include any
        code to open or close files.

        3) The maps applied can process multiple arguments. For example,
        you could swap two arguments using a mapping, or transform
        them to their sum and their difference. This was included to allow
        a decorator in the `quality.py` module that checks that an input
        `partition` is a valid partition of the nodes of the input graph `G`.
        In this example, the map has inputs `(G, partition)`. After checking
        for a valid partition, the map either raises an exception or leaves
        the inputs unchanged. Thus many functions that make this check can
        use the decorator rather than copy the checking code into each function.
        More complicated nested argument structures are described below.

    The remaining notes describe the code structure and methods for this
    class in broad terms to aid in understanding how to use it.

    Instantiating an `argmap` object simply stores the mapping function and
    the input identifiers of which arguments to map. The resulting decorator
    is ready to use this map to decorate any function. Calling that object
    (`argmap.__call__`, but usually done via `@my_decorator`) a lazily
    compiled thin wrapper of the decorated function is constructed,
    wrapped with the necessary function dunder attributes like `__doc__`
    and `__name__`. That thinly wrapped function is returned as the
    decorated function. When that decorated function is called, the thin
    wrapper of code calls `argmap._lazy_compile` which compiles the decorated
    function (using `argmap.compile`) and replaces the code of the thin
    wrapper with the newly compiled code. This saves the compilation step
    every import of networkx, at the cost of compiling upon the first call
    to the decorated function.

    When the decorated function is compiled, the code is recursively assembled
    using the `argmap.assemble` method. The recursive nature is needed in
    case of nested decorators. The result of the assembly is a number of
    useful objects.

      sig : the function signature of the original decorated function as
          constructed by :func:`argmap.signature`. This is constructed
          using `inspect.signature` but enhanced with attribute
          strings `sig_def` and `sig_call`, and other information
          specific to mapping arguments of this function.
          This information is used to construct a string of code defining
          the new decorated function.

      wrapped_name : a unique internally used name constructed by argmap
          for the decorated function.

      functions : a dict of the functions used inside the code of this
          decorated function, to be used as `globals` in `exec`.
          This dict is recursively updated to allow for nested decorating.

      mapblock : code (as a list of strings) to map the incoming argument
          values to their mapped values.

      finallys : code (as a list of strings) to provide the possibly nested
          set of finally clauses if needed.

      mutable_args : a bool indicating whether the `sig.args` tuple should be
          converted to a list so mutation can occur.

    After this recursive assembly process, the `argmap.compile` method
    constructs code (as strings) to convert the tuple `sig.args` to a list
    if needed. It joins the defining code with appropriate indents and
    compiles the result.  Finally, this code is evaluated and the original
    wrapper's implementation is replaced with the compiled version (see
    `argmap._lazy_compile` for more details).

    Other `argmap` methods include `_name` and `_count` which allow internally
    generated names to be unique within a python session.
    The methods `_flatten` and `_indent` process the nested lists of strings
    into properly indented python code ready to be compiled.

    More complicated nested tuples of arguments also allowed though
    usually not used. For the simple 2 argument case, the argmap
    input ("a", "b") implies the mapping function will take 2 arguments
    and return a 2-tuple of mapped values. A more complicated example
    with argmap input `("a", ("b", "c"))` requires the mapping function
    take 2 inputs, with the second being a 2-tuple. It then must output
    the 3 mapped values in the same nested structure `(newa, (newb, newc))`.
    This level of generality is not often needed, but was convenient
    to implement when handling the multiple arguments.

    See Also
    --------
    not_implemented_for
    open_file
    nodes_or_number
    py_random_state
    networkx.algorithms.community.quality.require_partition

    """

    def __init__(self, func, *args, try_finally=False):
        self._func = func
        self._args = args
        self._finally = try_finally

    @staticmethod
    def _lazy_compile(func):
        """Compile the source of a wrapped function

        Assemble and compile the decorated function, and intrusively replace its
        code with the compiled version's.  The thinly wrapped function becomes
        the decorated function.

        Parameters
        ----------
        func : callable
            A function returned by argmap.__call__ which is in the process
            of being called for the first time.

        Returns
        -------
        func : callable
            The same function, with a new __code__ object.

        Notes
        -----
        It was observed in NetworkX issue #4732 [1] that the import time of
        NetworkX was significantly bloated by the use of decorators: over half
        of the import time was being spent decorating functions.  This was
        somewhat improved by a change made to the `decorator` library, at the
        cost of a relatively heavy-weight call to `inspect.Signature.bind`
        for each call to the decorated function.

        The workaround we arrived at is to do minimal work at the time of
        decoration.  When the decorated function is called for the first time,
        we compile a function with the same function signature as the wrapped
        function.  The resulting decorated function is faster than one made by
        the `decorator` library, so that the overhead of the first call is
        'paid off' after a small number of calls.

        References
        ----------

        [1] https://github.com/networkx/networkx/issues/4732

        """
        real_func = func.__argmap__.compile(func.__wrapped__)
        func.__code__ = real_func.__code__
        func.__globals__.update(real_func.__globals__)
        func.__dict__.update(real_func.__dict__)
        return func

    def __call__(self, f):
        """Construct a lazily decorated wrapper of f.

        The decorated function will be compiled when it is called for the first time,
        and it will replace its own __code__ object so subsequent calls are fast.

        Parameters
        ----------
        f : callable
            A function to be decorated.

        Returns
        -------
        func : callable
            The decorated function.

        See Also
        --------
        argmap._lazy_compile
        """

        def func(*args, __wrapper=None, **kwargs):
            return argmap._lazy_compile(__wrapper)(*args, **kwargs)

        # standard function-wrapping stuff
        func.__name__ = f.__name__
        func.__doc__ = f.__doc__
        func.__defaults__ = f.__defaults__
        func.__kwdefaults__.update(f.__kwdefaults__ or {})
        func.__module__ = f.__module__
        func.__qualname__ = f.__qualname__
        func.__dict__.update(f.__dict__)
        func.__wrapped__ = f

        # now that we've wrapped f, we may have picked up some __dict__ or
        # __kwdefaults__ items that were set by a previous argmap.  Thus, we set
        # these values after those update() calls.

        # If we attempt to access func from within itself, that happens through
        # a closure -- which trips an error when we replace func.__code__.  The
        # standard workaround for functions which can't see themselves is to use
        # a Y-combinator, as we do here.
        func.__kwdefaults__["_argmap__wrapper"] = func

        # this self-reference is here because functools.wraps preserves
        # everything in __dict__, and we don't want to mistake a non-argmap
        # wrapper for an argmap wrapper
        func.__self__ = func

        # this is used to variously call self.assemble and self.compile
        func.__argmap__ = self

        if hasattr(f, "__argmap__"):
            func.__is_generator = f.__is_generator
        else:
            func.__is_generator = inspect.isgeneratorfunction(f)

        if self._finally and func.__is_generator:
            raise nx.NetworkXError("argmap cannot decorate generators with try_finally")

        return func

    __count = 0

    @classmethod
    def _count(cls):
        """Maintain a globally-unique identifier for function names and "file" names

        Note that this counter is a class method reporting a class variable
        so the count is unique within a Python session. It could differ from
        session to session for a specific decorator depending on the order
        that the decorators are created. But that doesn't disrupt `argmap`.

        This is used in two places: to construct unique variable names
        in the `_name` method and to construct unique fictitious filenames
        in the `_compile` method.

        Returns
        -------
        count : int
            An integer unique to this Python session (simply counts from zero)
        """
        cls.__count += 1
        return cls.__count

    _bad_chars = re.compile("[^a-zA-Z0-9_]")

    @classmethod
    def _name(cls, f):
        """Mangle the name of a function to be unique but somewhat human-readable

        The names are unique within a Python session and set using `_count`.

        Parameters
        ----------
        f : str or object

        Returns
        -------
        name : str
            The mangled version of `f.__name__` (if `f.__name__` exists) or `f`

        """
        f = f.__name__ if hasattr(f, "__name__") else f
        fname = re.sub(cls._bad_chars, "_", f)
        return f"argmap_{fname}_{cls._count()}"

    def compile(self, f):
        """Compile the decorated function.

        Called once for a given decorated function -- collects the code from all
        argmap decorators in the stack, and compiles the decorated function.

        Much of the work done here uses the `assemble` method to allow recursive
        treatment of multiple argmap decorators on a single decorated function.
        That flattens the argmap decorators, collects the source code to construct
        a single decorated function, then compiles/executes/returns that function.

        The source code for the decorated function is stored as an attribute
        `_code` on the function object itself.

        Note that Python's `compile` function requires a filename, but this
        code is constructed without a file, so a fictitious filename is used
        to describe where the function comes from. The name is something like:
        "argmap compilation 4".

        Parameters
        ----------
        f : callable
            The function to be decorated

        Returns
        -------
        func : callable
            The decorated file

        """
        sig, wrapped_name, functions, mapblock, finallys, mutable_args = self.assemble(
            f
        )

        call = f"{sig.call_sig.format(wrapped_name)}#"
        mut_args = f"{sig.args} = list({sig.args})" if mutable_args else ""
        body = argmap._indent(sig.def_sig, mut_args, mapblock, call, finallys)
        code = "\n".join(body)

        locl = {}
        globl = dict(functions.values())
        filename = f"{self.__class__} compilation {self._count()}"
        compiled = compile(code, filename, "exec")
        exec(compiled, globl, locl)
        func = locl[sig.name]
        func._code = code
        return func

    def assemble(self, f):
        """Collects components of the source for the decorated function wrapping f.

        If `f` has multiple argmap decorators, we recursively assemble the stack of
        decorators into a single flattened function.

        This method is part of the `compile` method's process yet separated
        from that method to allow recursive processing. The outputs are
        strings, dictionaries and lists that collect needed info to
        flatten any nested argmap-decoration.

        Parameters
        ----------
        f : callable
            The function to be decorated.  If f is argmapped, we assemble it.

        Returns
        -------
        sig : argmap.Signature
            The function signature as an `argmap.Signature` object.
        wrapped_name : str
            The mangled name used to represent the wrapped function in the code
            being assembled.
        functions : dict
            A dictionary mapping id(g) -> (mangled_name(g), g) for functions g
            referred to in the code being assembled. These need to be present
            in the ``globals`` scope of ``exec`` when defining the decorated
            function.
        mapblock : list of lists and/or strings
            Code that implements mapping of parameters including any try blocks
            if needed. This code will precede the decorated function call.
        finallys : list of lists and/or strings
            Code that implements the finally blocks to post-process the
            arguments (usually close any files if needed) after the
            decorated function is called.
        mutable_args : bool
            True if the decorator needs to modify positional arguments
            via their indices. The compile method then turns the argument
            tuple into a list so that the arguments can be modified.
        """

        # first, we check if f is already argmapped -- if that's the case,
        # build up the function recursively.
        # > mapblock is generally a list of function calls of the sort
        #     arg = func(arg)
        # in addition to some try-blocks if needed.
        # > finallys is a recursive list of finally blocks of the sort
        #         finally:
        #             close_func_1()
        #     finally:
        #         close_func_2()
        # > functions is a dict of functions used in the scope of our decorated
        # function. It will be used to construct globals used in compilation.
        # We make functions[id(f)] = name_of_f, f to ensure that a given
        # function is stored and named exactly once even if called by
        # nested decorators.
        if hasattr(f, "__argmap__") and f.__self__ is f:
            (
                sig,
                wrapped_name,
                functions,
                mapblock,
                finallys,
                mutable_args,
            ) = f.__argmap__.assemble(f.__wrapped__)
            functions = dict(functions)  # shallow-copy just in case
        else:
            sig = self.signature(f)
            wrapped_name = self._name(f)
            mapblock, finallys = [], []
            functions = {id(f): (wrapped_name, f)}
            mutable_args = False

        if id(self._func) in functions:
            fname, _ = functions[id(self._func)]
        else:
            fname, _ = functions[id(self._func)] = self._name(self._func), self._func

        # this is a bit complicated -- we can call functions with a variety of
        # nested arguments, so long as their input and output are tuples with
        # the same nested structure. e.g. ("a", "b") maps arguments a and b.
        # A more complicated nesting like (0, (3, 4)) maps arguments 0, 3, 4
        # expecting the mapping to output new values in the same nested shape.
        # The ability to argmap multiple arguments was necessary for
        # the decorator `nx.algorithms.community.quality.require_partition`, and
        # while we're not taking full advantage of the ability to handle
        # multiply-nested tuples, it was convenient to implement this in
        # generality because the recursive call to `get_name` is necessary in
        # any case.
        applied = set()

        def get_name(arg, first=True):
            nonlocal mutable_args
            if isinstance(arg, tuple):
                name = ", ".join(get_name(x, False) for x in arg)
                return name if first else f"({name})"
            if arg in applied:
                raise nx.NetworkXError(f"argument {arg} is specified multiple times")
            applied.add(arg)
            if arg in sig.names:
                return sig.names[arg]
            elif isinstance(arg, str):
                if sig.kwargs is None:
                    raise nx.NetworkXError(
                        f"name {arg} is not a named parameter and this function doesn't have kwargs"
                    )
                return f"{sig.kwargs}[{arg!r}]"
            else:
                if sig.args is None:
                    raise nx.NetworkXError(
                        f"index {arg} not a parameter index and this function doesn't have args"
                    )
                mutable_args = True
                return f"{sig.args}[{arg - sig.n_positional}]"

        if self._finally:
            # here's where we handle try_finally decorators.  Such a decorator
            # returns a mapped argument and a function to be called in a
            # finally block.  This feature was required by the open_file
            # decorator.  The below generates the code
            #
            # name, final = func(name)                   #<--append to mapblock
            # try:                                       #<--append to mapblock
            #     ... more argmapping and try blocks
            #     return WRAPPED_FUNCTION(...)
            #     ... more finally blocks
            # finally:                                   #<--prepend to finallys
            #     final()                                #<--prepend to finallys
            #
            for a in self._args:
                name = get_name(a)
                final = self._name(name)
                mapblock.append(f"{name}, {final} = {fname}({name})")
                mapblock.append("try:")
                finallys = ["finally:", f"{final}()#", "#", finallys]
        else:
            mapblock.extend(
                f"{name} = {fname}({name})" for name in map(get_name, self._args)
            )

        return sig, wrapped_name, functions, mapblock, finallys, mutable_args

    @classmethod
    def signature(cls, f):
        r"""Construct a Signature object describing `f`

        Compute a Signature so that we can write a function wrapping f with
        the same signature and call-type.

        Parameters
        ----------
        f : callable
            A function to be decorated

        Returns
        -------
        sig : argmap.Signature
            The Signature of f

        Notes
        -----
        The Signature is a namedtuple with names:

            name : a unique version of the name of the decorated function
            signature : the inspect.signature of the decorated function
            def_sig : a string used as code to define the new function
            call_sig : a string used as code to call the decorated function
            names : a dict keyed by argument name and index to the argument's name
            n_positional : the number of positional arguments in the signature
            args : the name of the VAR_POSITIONAL argument if any, i.e. \*theseargs
            kwargs : the name of the VAR_KEYWORDS argument if any, i.e. \*\*kwargs

        These named attributes of the signature are used in `assemble` and `compile`
        to construct a string of source code for the decorated function.

        """
        sig = inspect.signature(f, follow_wrapped=False)
        def_sig = []
        call_sig = []
        names = {}

        kind = None
        args = None
        kwargs = None
        npos = 0
        for i, param in enumerate(sig.parameters.values()):
            # parameters can be position-only, keyword-or-position, keyword-only
            # in any combination, but only in the order as above.  we do edge
            # detection to add the appropriate punctuation
            prev = kind
            kind = param.kind
            if prev == param.POSITIONAL_ONLY != kind:
                # the last token was position-only, but this one isn't
                def_sig.append("/")
            if (
                param.VAR_POSITIONAL
                != prev
                != param.KEYWORD_ONLY
                == kind
                != param.VAR_POSITIONAL
            ):
                # param is the first keyword-only arg and isn't starred
                def_sig.append("*")

            # star arguments as appropriate
            if kind == param.VAR_POSITIONAL:
                name = "*" + param.name
                args = param.name
                count = 0
            elif kind == param.VAR_KEYWORD:
                name = "**" + param.name
                kwargs = param.name
                count = 0
            else:
                names[i] = names[param.name] = param.name
                name = param.name
                count = 1

            # assign to keyword-only args in the function call
            if kind == param.KEYWORD_ONLY:
                call_sig.append(f"{name} = {name}")
            else:
                npos += count
                call_sig.append(name)

            def_sig.append(name)

        fname = cls._name(f)
        def_sig = f'def {fname}({", ".join(def_sig)}):'

        call_sig = f"return {{}}({', '.join(call_sig)})"

        return cls.Signature(fname, sig, def_sig, call_sig, names, npos, args, kwargs)

    Signature = collections.namedtuple(
        "Signature",
        [
            "name",
            "signature",
            "def_sig",
            "call_sig",
            "names",
            "n_positional",
            "args",
            "kwargs",
        ],
    )

    @staticmethod
    def _flatten(nestlist, visited):
        """flattens a recursive list of lists that doesn't have cyclic references

        Parameters
        ----------
        nestlist : iterable
            A recursive list of objects to be flattened into a single iterable

        visited : set
            A set of object ids which have been walked -- initialize with an
            empty set

        Yields
        ------
        Non-list objects contained in nestlist

        """
        for thing in nestlist:
            if isinstance(thing, list):
                if id(thing) in visited:
                    raise ValueError("A cycle was found in nestlist.  Be a tree.")
                else:
                    visited.add(id(thing))
                yield from argmap._flatten(thing, visited)
            else:
                yield thing

    _tabs = " " * 64

    @staticmethod
    def _indent(*lines):
        """Indent list of code lines to make executable Python code

        Indents a tree-recursive list of strings, following the rule that one
        space is added to the tab after a line that ends in a colon, and one is
        removed after a line that ends in an hashmark.

        Parameters
        ----------
        *lines : lists and/or strings
            A recursive list of strings to be assembled into properly indented
            code.

        Returns
        -------
        code : str

        Examples
        --------

            argmap._indent(*["try:", "try:", "pass#", "finally:", "pass#", "#",
                             "finally:", "pass#"])

        renders to

            '''try:
             try:
              pass#
             finally:
              pass#
             #
            finally:
             pass#'''
        """
        depth = 0
        for line in argmap._flatten(lines, set()):
            yield f"{argmap._tabs[:depth]}{line}"
            depth += (line[-1:] == ":") - (line[-1:] == "#")
<|endoftext|>"""
Miscellaneous Helpers for NetworkX.

These are not imported into the base networkx namespace but
can be accessed, for example, as

>>> import networkx
>>> networkx.utils.make_list_of_ints({1, 2, 3})
[1, 2, 3]
>>> networkx.utils.arbitrary_element({5, 1, 7})  # doctest: +SKIP
1
"""

import random
import sys
import uuid
import warnings
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Sized
from itertools import chain, tee

import networkx as nx

__all__ = [
    "flatten",
    "make_list_of_ints",
    "dict_to_numpy_array",
    "arbitrary_element",
    "pairwise",
    "groups",
    "create_random_state",
    "create_py_random_state",
    "PythonRandomInterface",
    "PythonRandomViaNumpyBits",
    "nodes_equal",
    "edges_equal",
    "graphs_equal",
    "_clear_cache",
]


# some cookbook stuff
# used in deciding whether something is a bunch of nodes, edges, etc.
# see G.add_nodes and others in Graph Class in networkx/base.py


def flatten(obj, result=None):
    """Return flattened version of (possibly nested) iterable object."""
    if not isinstance(obj, Iterable | Sized) or isinstance(obj, str):
        return obj
    if result is None:
        result = []
    for item in obj:
        if not isinstance(item, Iterable | Sized) or isinstance(item, str):
            result.append(item)
        else:
            flatten(item, result)
    return tuple(result)


def make_list_of_ints(sequence):
    """Return list of ints from sequence of integral numbers.

    All elements of the sequence must satisfy int(element) == element
    or a ValueError is raised. Sequence is iterated through once.

    If sequence is a list, the non-int values are replaced with ints.
    So, no new list is created
    """
    if not isinstance(sequence, list):
        result = []
        for i in sequence:
            errmsg = f"sequence is not all integers: {i}"
            try:
                ii = int(i)
            except ValueError:
                raise nx.NetworkXError(errmsg) from None
            if ii != i:
                raise nx.NetworkXError(errmsg)
            result.append(ii)
        return result
    # original sequence is a list... in-place conversion to ints
    for indx, i in enumerate(sequence):
        errmsg = f"sequence is not all integers: {i}"
        if isinstance(i, int):
            continue
        try:
            ii = int(i)
        except ValueError:
            raise nx.NetworkXError(errmsg) from None
        if ii != i:
            raise nx.NetworkXError(errmsg)
        sequence[indx] = ii
    return sequence


def dict_to_numpy_array(d, mapping=None):
    """Convert a dictionary of dictionaries to a numpy array
    with optional mapping."""
    try:
        return _dict_to_numpy_array2(d, mapping)
    except (AttributeError, TypeError):
        # AttributeError is when no mapping was provided and v.keys() fails.
        # TypeError is when a mapping was provided and d[k1][k2] fails.
        return _dict_to_numpy_array1(d, mapping)


def _dict_to_numpy_array2(d, mapping=None):
    """Convert a dictionary of dictionaries to a 2d numpy array
    with optional mapping.

    """
    import numpy as np

    if mapping is None:
        s = set(d.keys())
        for k, v in d.items():
            s.update(v.keys())
        mapping = dict(zip(s, range(len(s))))
    n = len(mapping)
    a = np.zeros((n, n))
    for k1, i in mapping.items():
        for k2, j in mapping.items():
            try:
                a[i, j] = d[k1][k2]
            except KeyError:
                pass
    return a


def _dict_to_numpy_array1(d, mapping=None):
    """Convert a dictionary of numbers to a 1d numpy array with optional mapping."""
    import numpy as np

    if mapping is None:
        s = set(d.keys())
        mapping = dict(zip(s, range(len(s))))
    n = len(mapping)
    a = np.zeros(n)
    for k1, i in mapping.items():
        i = mapping[k1]
        a[i] = d[k1]
    return a


def arbitrary_element(iterable):
    """Returns an arbitrary element of `iterable` without removing it.

    This is most useful for "peeking" at an arbitrary element of a set,
    but can be used for any list, dictionary, etc., as well.

    Parameters
    ----------
    iterable : `abc.collections.Iterable` instance
        Any object that implements ``__iter__``, e.g. set, dict, list, tuple,
        etc.

    Returns
    -------
    The object that results from ``next(iter(iterable))``

    Raises
    ------
    ValueError
        If `iterable` is an iterator (because the current implementation of
        this function would consume an element from the iterator).

    Examples
    --------
    Arbitrary elements from common Iterable objects:

    >>> nx.utils.arbitrary_element([1, 2, 3])  # list
    1
    >>> nx.utils.arbitrary_element((1, 2, 3))  # tuple
    1
    >>> nx.utils.arbitrary_element({1, 2, 3})  # set
    1
    >>> d = {k: v for k, v in zip([1, 2, 3], [3, 2, 1])}
    >>> nx.utils.arbitrary_element(d)  # dict_keys
    1
    >>> nx.utils.arbitrary_element(d.values())  # dict values
    3

    `str` is also an Iterable:

    >>> nx.utils.arbitrary_element("hello")
    'h'

    :exc:`ValueError` is raised if `iterable` is an iterator:

    >>> iterator = iter([1, 2, 3])  # Iterator, *not* Iterable
    >>> nx.utils.arbitrary_element(iterator)
    Traceback (most recent call last):
        ...
    ValueError: cannot return an arbitrary item from an iterator

    Notes
    -----
    This function does not return a *random* element. If `iterable` is
    ordered, sequential calls will return the same value::

        >>> l = [1, 2, 3]
        >>> nx.utils.arbitrary_element(l)
        1
        >>> nx.utils.arbitrary_element(l)
        1

    """
    if isinstance(iterable, Iterator):
        raise ValueError("cannot return an arbitrary item from an iterator")
    # Another possible implementation is ``for x in iterable: return x``.
    return next(iter(iterable))


# Recipe from the itertools documentation.
def pairwise(iterable, cyclic=False):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = tee(iterable)
    first = next(b, None)
    if cyclic is True:
        return zip(a, chain(b, (first,)))
    return zip(a, b)


def groups(many_to_one):
    """Converts a many-to-one mapping into a one-to-many mapping.

    `many_to_one` must be a dictionary whose keys and values are all
    :term:`hashable`.

    The return value is a dictionary mapping values from `many_to_one`
    to sets of keys from `many_to_one` that have that value.

    Examples
    --------
    >>> from networkx.utils import groups
    >>> many_to_one = {"a": 1, "b": 1, "c": 2, "d": 3, "e": 3}
    >>> groups(many_to_one)  # doctest: +SKIP
    {1: {'a', 'b'}, 2: {'c'}, 3: {'e', 'd'}}
    """
    one_to_many = defaultdict(set)
    for v, k in many_to_one.items():
        one_to_many[k].add(v)
    return dict(one_to_many)


def create_random_state(random_state=None):
    """Returns a numpy.random.RandomState or numpy.random.Generator instance
    depending on input.

    Parameters
    ----------
    random_state : int or NumPy RandomState or Generator instance, optional (default=None)
        If int, return a numpy.random.RandomState instance set with seed=int.
        if `numpy.random.RandomState` instance, return it.
        if `numpy.random.Generator` instance, return it.
        if None or numpy.random, return the global random number generator used
        by numpy.random.
    """
    import numpy as np

    if random_state is None or random_state is np.random:
        return np.random.mtrand._rand
    if isinstance(random_state, np.random.RandomState):
        return random_state
    if isinstance(random_state, int):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.Generator):
        return random_state
    msg = (
        f"{random_state} cannot be used to create a numpy.random.RandomState or\n"
        "numpy.random.Generator instance"
    )
    raise ValueError(msg)


class PythonRandomViaNumpyBits(random.Random):
    """Provide the random.random algorithms using a numpy.random bit generator

    The intent is to allow people to contribute code that uses Python's random
    library, but still allow users to provide a single easily controlled random
    bit-stream for all work with NetworkX. This implementation is based on helpful
    comments and code from Robert Kern on NumPy's GitHub Issue #24458.

    This implementation supersedes that of `PythonRandomInterface` which rewrote
    methods to account for subtle differences in API between `random` and
    `numpy.random`. Instead this subclasses `random.Random` and overwrites
    the methods `random`, `getrandbits`, `getstate`, `setstate` and `seed`.
    It makes them use the rng values from an input numpy `RandomState` or `Generator`.
    Those few methods allow the rest of the `random.Random` methods to provide
    the API interface of `random.random` while using randomness generated by
    a numpy generator.
    """

    def __init__(self, rng=None):
        try:
            import numpy as np
        except ImportError:
            msg = "numpy not found, only random.random available."
            warnings.warn(msg, ImportWarning)

        if rng is None:
            self._rng = np.random.mtrand._rand
        else:
            self._rng = rng

        # Not necessary, given our overriding of gauss() below, but it's
        # in the superclass and nominally public, so initialize it here.
        self.gauss_next = None

    def random(self):
        """Get the next random number in the range 0.0 <= X < 1.0."""
        return self._rng.random()

    def getrandbits(self, k):
        """getrandbits(k) -> x.  Generates an int with k random bits."""
        if k < 0:
            raise ValueError("number of bits must be non-negative")
        numbytes = (k + 7) // 8  # bits / 8 and rounded up
        x = int.from_bytes(self._rng.bytes(numbytes), "big")
        return x >> (numbytes * 8 - k)  # trim excess bits

    def getstate(self):
        return self._rng.__getstate__()

    def setstate(self, state):
        self._rng.__setstate__(state)

    def seed(self, *args, **kwds):
        "Do nothing override method."
        raise NotImplementedError("seed() not implemented in PythonRandomViaNumpyBits")


##################################################################
class PythonRandomInterface:
    """PythonRandomInterface is included for backward compatibility
    New code should use PythonRandomViaNumpyBits instead.
    """

    def __init__(self, rng=None):
        try:
            import numpy as np
        except ImportError:
            msg = "numpy not found, only random.random available."
            warnings.warn(msg, ImportWarning)

        if rng is None:
            self._rng = np.random.mtrand._rand
        else:
            self._rng = rng

    def random(self):
        return self._rng.random()

    def uniform(self, a, b):
        return a + (b - a) * self._rng.random()

    def randrange(self, a, b=None):
        import numpy as np

        if b is None:
            a, b = 0, a
        if b > 9223372036854775807:  # from np.iinfo(np.int64).max
            tmp_rng = PythonRandomViaNumpyBits(self._rng)
            return tmp_rng.randrange(a, b)

        if isinstance(self._rng, np.random.Generator):
            return self._rng.integers(a, b)
        return self._rng.randint(a, b)

    # NOTE: the numpy implementations of `choice` don't support strings, so
    # this cannot be replaced with self._rng.choice
    def choice(self, seq):
        import numpy as np

        if isinstance(self._rng, np.random.Generator):
            idx = self._rng.integers(0, len(seq))
        else:
            idx = self._rng.randint(0, len(seq))
        return seq[idx]

    def gauss(self, mu, sigma):
        return self._rng.normal(mu, sigma)

    def shuffle(self, seq):
        return self._rng.shuffle(seq)

    #    Some methods don't match API for numpy RandomState.
    #    Commented out versions are not used by NetworkX

    def sample(self, seq, k):
        return self._rng.choice(list(seq), size=(k,), replace=False)

    def randint(self, a, b):
        import numpy as np

        if b > 9223372036854775807:  # from np.iinfo(np.int64).max
            tmp_rng = PythonRandomViaNumpyBits(self._rng)
            return tmp_rng.randint(a, b)

        if isinstance(self._rng, np.random.Generator):
            return self._rng.integers(a, b + 1)
        return self._rng.randint(a, b + 1)

    #    exponential as expovariate with 1/argument,
    def expovariate(self, scale):
        return self._rng.exponential(1 / scale)

    #    pareto as paretovariate with 1/argument,
    def paretovariate(self, shape):
        return self._rng.pareto(shape)


#    weibull as weibullvariate multiplied by beta,
#    def weibullvariate(self, alpha, beta):
#        return self._rng.weibull(alpha) * beta
#
#    def triangular(self, low, high, mode):
#        return self._rng.triangular(low, mode, high)
#
#    def choices(self, seq, weights=None, cum_weights=None, k=1):
#        return self._rng.choice(seq


def create_py_random_state(random_state=None):
    """Returns a random.Random instance depending on input.

    Parameters
    ----------
    random_state : int or random number generator or None (default=None)
        - If int, return a `random.Random` instance set with seed=int.
        - If `random.Random` instance, return it.
        - If None or the `np.random` package, return the global random number
          generator used by `np.random`.
        - If an `np.random.Generator` instance, or the `np.random` package, or
          the global numpy random number generator, then return it.
          wrapped in a `PythonRandomViaNumpyBits` class.
        - If a `PythonRandomViaNumpyBits` instance, return it.
        - If a `PythonRandomInterface` instance, return it.
        - If a `np.random.RandomState` instance and not the global numpy default,
          return it wrapped in `PythonRandomInterface` for backward bit-stream
          matching with legacy code.

    Notes
    -----
    - A diagram intending to illustrate the relationships behind our support
      for numpy random numbers is called
      `NetworkX Numpy Random Numbers <https://excalidraw.com/#room=b5303f2b03d3af7ccc6a,e5ZDIWdWWCTTsg8OqoRvPA>`_.
    - More discussion about this support also appears in
      `gh-6869#comment <https://github.com/networkx/networkx/pull/6869#issuecomment-1944799534>`_.
    - Wrappers of numpy.random number generators allow them to mimic the Python random
      number generation algorithms. For example, Python can create arbitrarily large
      random ints, and the wrappers use Numpy bit-streams with CPython's random module
      to choose arbitrarily large random integers too.
    - We provide two wrapper classes:
      `PythonRandomViaNumpyBits` is usually what you want and is always used for
      `np.Generator` instances. But for users who need to recreate random numbers
      produced in NetworkX 3.2 or earlier, we maintain the `PythonRandomInterface`
      wrapper as well. We use it only used if passed a (non-default) `np.RandomState`
      instance pre-initialized from a seed. Otherwise the newer wrapper is used.
    """
    if random_state is None or random_state is random:
        return random._inst
    if isinstance(random_state, random.Random):
        return random_state
    if isinstance(random_state, int):
        return random.Random(random_state)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if isinstance(random_state, PythonRandomInterface | PythonRandomViaNumpyBits):
            return random_state
        if isinstance(random_state, np.random.Generator):
            return PythonRandomViaNumpyBits(random_state)
        if random_state is np.random:
            return PythonRandomViaNumpyBits(np.random.mtrand._rand)

        if isinstance(random_state, np.random.RandomState):
            if random_state is np.random.mtrand._rand:
                return PythonRandomViaNumpyBits(random_state)
            # Only need older interface if specially constructed RandomState used
            return PythonRandomInterface(random_state)

    msg = f"{random_state} cannot be used to generate a random.Random instance"
    raise ValueError(msg)


def nodes_equal(nodes1, nodes2):
    """Check if nodes are equal.

    Equality here means equal as Python objects.
    Node data must match if included.
    The order of nodes is not relevant.

    Parameters
    ----------
    nodes1, nodes2 : iterables of nodes, or (node, datadict) tuples

    Returns
    -------
    bool
        True if nodes are equal, False otherwise.
    """
    nlist1 = list(nodes1)
    nlist2 = list(nodes2)
    try:
        d1 = dict(nlist1)
        d2 = dict(nlist2)
    except (ValueError, TypeError):
        d1 = dict.fromkeys(nlist1)
        d2 = dict.fromkeys(nlist2)
    return d1 == d2


def edges_equal(edges1, edges2):
    """Check if edges are equal.

    Equality here means equal as Python objects.
    Edge data must match if included.
    The order of the edges is not relevant.

    Parameters
    ----------
    edges1, edges2 : iterables of with u, v nodes as
        edge tuples (u, v), or
        edge tuples with data dicts (u, v, d), or
        edge tuples with keys and data dicts (u, v, k, d)

    Returns
    -------
    bool
        True if edges are equal, False otherwise.
    """
    from collections import defaultdict

    d1 = defaultdict(dict)
    d2 = defaultdict(dict)
    c1 = 0
    for c1, e in enumerate(edges1):
        u, v = e[0], e[1]
        data = [e[2:]]
        if v in d1[u]:
            data = d1[u][v] + data
        d1[u][v] = data
        d1[v][u] = data
    c2 = 0
    for c2, e in enumerate(edges2):
        u, v = e[0], e[1]
        data = [e[2:]]
        if v in d2[u]:
            data = d2[u][v] + data
        d2[u][v] = data
        d2[v][u] = data
    if c1 != c2:
        return False
    # can check one direction because lengths are the same.
    for n, nbrdict in d1.items():
        for nbr, datalist in nbrdict.items():
            if n not in d2:
                return False
            if nbr not in d2[n]:
                return False
            d2datalist = d2[n][nbr]
            for data in datalist:
                if datalist.count(data) != d2datalist.count(data):
                    return False
    return True


def graphs_equal(graph1, graph2):
    """Check if graphs are equal.

    Equality here means equal as Python objects (not isomorphism).
    Node, edge and graph data must match.

    Parameters
    ----------
    graph1, graph2 : graph

    Returns
    -------
    bool
        True if graphs are equal, False otherwise.
    """
    return (
        graph1.adj == graph2.adj
        and graph1.nodes == graph2.nodes
        and graph1.graph == graph2.graph
    )


def _clear_cache(G):
    """Clear the cache of a graph (currently stores converted graphs).

    Caching is controlled via ``nx.config.cache_converted_graphs`` configuration.
    """
    if cache := getattr(G, "__networkx_cache__", None):
        cache.clear()


def check_create_using(create_using, *, directed=None, multigraph=None, default=None):
    """Assert that create_using has good properties

    This checks for desired directedness and multi-edge properties.
    It returns `create_using` unless that is `None` when it returns
    the optionally specified default value.

    Parameters
    ----------
    create_using : None, graph class or instance
        The input value of create_using for a function.
    directed : None or bool
        Whether to check `create_using.is_directed() == directed`.
        If None, do not assert directedness.
    multigraph : None or bool
        Whether to check `create_using.is_multigraph() == multigraph`.
        If None, do not assert multi-edge property.
    default : None or graph class
        The graph class to return if create_using is None.

    Returns
    -------
    create_using : graph class or instance
        The provided graph class or instance, or if None, the `default` value.

    Raises
    ------
    NetworkXError
        When `create_using` doesn't match the properties specified by `directed`
        or `multigraph` parameters.
    """
    if default is None:
        default = nx.Graph
    G = create_using if create_using is not None else default

    G_directed = G.is_directed(None) if isinstance(G, type) else G.is_directed()
    G_multigraph = G.is_multigraph(None) if isinstance(G, type) else G.is_multigraph()

    if directed is not None:
        if directed and not G_directed:
            raise nx.NetworkXError("create_using must be directed")
        if not directed and G_directed:
            raise nx.NetworkXError("create_using must not be directed")

    if multigraph is not None:
        if multigraph and not G_multigraph:
            raise nx.NetworkXError("create_using must be a multi-graph")
        if not multigraph and G_multigraph:
            raise nx.NetworkXError("create_using must not be a multi-graph")
    return G
<|endoftext|>from networkx.utils.misc import *
from networkx.utils.decorators import *
from networkx.utils.random_sequence import *
from networkx.utils.union_find import *
from networkx.utils.rcm import *
from networkx.utils.heaps import *
from networkx.utils.configs import *
from networkx.utils.backends import *
<|endoftext|>"""
Union-find data structure.
"""

from networkx.utils import groups


class UnionFind:
    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.

      Union-find data structure. Based on Josiah Carlson's code,
      https://code.activestate.com/recipes/215912/
      with significant additional changes by D. Eppstein.
      http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py

    """

    def __init__(self, elements=None):
        """Create a new empty union-find structure.

        If *elements* is an iterable, this structure will be initialized
        with the discrete partition on the given set of elements.

        """
        if elements is None:
            elements = ()
        self.parents = {}
        self.weights = {}
        for x in elements:
            self.weights[x] = 1
            self.parents[x] = x

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = []
        root = self.parents[object]
        while root != object:
            path.append(object)
            object = root
            root = self.parents[object]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def to_sets(self):
        """Iterates over the sets stored in this structure.

        For example::

            >>> partition = UnionFind("xyz")
            >>> sorted(map(sorted, partition.to_sets()))
            [['x'], ['y'], ['z']]
            >>> partition.union("x", "y")
            >>> sorted(map(sorted, partition.to_sets()))
            [['x', 'y'], ['z']]

        """
        # Ensure fully pruned paths
        for x in self.parents:
            _ = self[x]  # Evaluated for side-effect only

        yield from groups(self.parents).values()

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        # Find the heaviest root according to its weight.
        roots = iter(
            sorted(
                {self[x] for x in objects}, key=lambda r: self.weights[r], reverse=True
            )
        )
        try:
            root = next(roots)
        except StopIteration:
            return

        for r in roots:
            self.weights[root] += self.weights[r]
            self.parents[r] = root
<|endoftext|>"""
Docs for backend users
~~~~~~~~~~~~~~~~~~~~~~
NetworkX utilizes a plugin-dispatch architecture. A valid NetworkX backend
specifies `entry points
<https://packaging.python.org/en/latest/specifications/entry-points>`_, named
``networkx.backends`` and an optional ``networkx.backend_info`` when it is
installed (not imported). This allows NetworkX to dispatch (redirect) function
calls to the backend so the execution flows to the designated backend
implementation. This design enhances flexibility and integration, making
NetworkX more adaptable and efficient.

NetworkX can dispatch to backends **explicitly** (this requires changing code)
or **automatically** (this requires setting configuration or environment
variables). The best way to use a backend depends on the backend, your use
case, and whether you want to automatically convert to or from backend
graphs. Automatic conversions of graphs is always opt-in.

To explicitly dispatch to a backend, use the `backend=` keyword argument in a
dispatchable function. This will convert (and cache by default) input NetworkX
graphs to backend graphs and call the backend implementation. Another explicit
way to use a backend is to create a backend graph directly--for example,
perhaps the backend has its own functions for loading data and creating
graphs--and pass that graph to a dispatchable function, which will then call
the backend implementation without converting.

Using automatic dispatch requires setting configuration options. Every NetworkX
configuration may also be set from an environment variable and are processed at
the time networkx is imported.  The following configuration variables are
supported:

* ``nx.config.backend_priority`` (``NETWORKX_BACKEND_PRIORITY`` env var), a
  list of backends, controls dispatchable functions that don't return graphs
  such as e.g. ``nx.pagerank``. When one of these functions is called with
  NetworkX graphs as input, the dispatcher iterates over the backends listed in
  this backend_priority config and will use the first backend that implements
  this function. The input NetworkX graphs are converted (and cached by
  default) to backend graphs. Using this configuration can allow you to use the
  full flexibility of NetworkX graphs and the performance of backend
  implementations, but possible downsides are that creating NetworkX graphs,
  converting to backend graphs, and caching backend graphs may all be
  expensive.

* ``nx.config.backend_priority.algos`` (``NETWORKX_BACKEND_PRIORITY_ALGOS`` env
  var), can be used instead of ``nx.config.backend_priority``
  (``NETWORKX_BACKEND_PRIORITY`` env var) to emphasize that the setting only
  affects the dispatching of algorithm functions as described above.

* ``nx.config.backend_priority.generators``
  (``NETWORKX_BACKEND_PRIORITY_GENERATORS`` env var), a list of backends,
  controls dispatchable functions that return graphs such as
  nx.from_pandas_edgelist and nx.empty_graph. When one of these functions is
  called, the first backend listed in this backend_priority config that
  implements this function will be used and will return a backend graph. When
  this backend graph is passed to other dispatchable NetworkX functions, it
  will use the backend implementation if it exists or raise by default unless
  nx.config.fallback_to_nx is True (default is False). Using this configuration
  avoids creating NetworkX graphs, which subsequently avoids the need to
  convert to and cache backend graphs as when using
  nx.config.backend_priority.algos, but possible downsides are that the backend
  graph may not behave the same as a NetworkX graph and the backend may not
  implement all algorithms that you use, which may break your workflow.

* ``nx.config.fallback_to_nx`` (``NETWORKX_FALLBACK_TO_NX`` env var), a boolean
  (default False), controls what happens when a backend graph is passed to a
  dispatchable function that is not implemented by that backend. The default
  behavior when False is to raise. If True, then the backend graph will be
  converted (and cached by default) to a NetworkX graph and will run with the
  default NetworkX implementation. Enabling this configuration can allow
  workflows to complete if the backend does not implement all algorithms used
  by the workflow, but a possible downside is that it may require converting
  the input backend graph to a NetworkX graph, which may be expensive. If a
  backend graph is duck-type compatible as a NetworkX graph, then the backend
  may choose not to convert to a NetworkX graph and use the incoming graph
  as-is.

* ``nx.config.cache_converted_graphs`` (``NETWORKX_CACHE_CONVERTED_GRAPHS`` env
  var), a boolean (default True), controls whether graph conversions are cached
  to G.__networkx_cache__ or not. Caching can improve performance by avoiding
  repeated conversions, but it uses more memory.

.. note:: Backends *should* follow the NetworkX backend naming convention. For
   example, if a backend is named ``parallel`` and specified using
   ``backend=parallel`` or ``NETWORKX_BACKEND_PRIORITY=parallel``, the package
   installed is ``nx-parallel``, and we would use ``import nx_parallel`` if we
   were to import the backend package directly.

Backends are encouraged to document how they recommend to be used and whether
their graph types are duck-type compatible as NetworkX graphs. If backend
graphs are NetworkX-compatible and you want your workflow to automatically
"just work" with a backend--converting and caching if necessary--then use all
of the above configurations. Automatically converting graphs is opt-in, and
configuration gives the user control.

Examples:
---------

Use the ``cugraph`` backend for every algorithm function it supports. This will
allow for fall back to the default NetworkX implementations for algorithm calls
not supported by cugraph because graph generator functions are still returning
NetworkX graphs.

.. code-block:: bash

   bash> NETWORKX_BACKEND_PRIORITY=cugraph python my_networkx_script.py

Explicitly use the ``parallel`` backend for a function call.

.. code-block:: python

    nx.betweenness_centrality(G, k=10, backend="parallel")

Explicitly use the ``parallel`` backend for a function call by passing an
instance of the backend graph type to the function.

.. code-block:: python

   H = nx_parallel.ParallelGraph(G)
   nx.betweenness_centrality(H, k=10)

Explicitly use the ``parallel`` backend and pass additional backend-specific
arguments. Here, ``get_chunks`` is an argument unique to the ``parallel``
backend.

.. code-block:: python

   nx.betweenness_centrality(G, k=10, backend="parallel", get_chunks=get_chunks)

Automatically dispatch the ``cugraph`` backend for all NetworkX algorithms and
generators, and allow the backend graph object returned from generators to be
passed to NetworkX functions the backend does not support.

.. code-block:: bash

   bash> NETWORKX_BACKEND_PRIORITY_ALGOS=cugraph \\
         NETWORKX_BACKEND_PRIORITY_GENERATORS=cugraph \\
         NETWORKX_FALLBACK_TO_NX=True \\
         python my_networkx_script.py

How does this work?
-------------------

If you've looked at functions in the NetworkX codebase, you might have seen the
``@nx._dispatchable`` decorator on most of the functions. This decorator allows the NetworkX
function to dispatch to the corresponding backend function if available. When the decorated
function is called, it first checks for a backend to run the function, and if no appropriate
backend is specified or available, it runs the NetworkX version of the function.

Backend Keyword Argument
^^^^^^^^^^^^^^^^^^^^^^^^

When a decorated function is called with the ``backend`` kwarg provided, it checks
if the specified backend is installed, and loads it. Next it checks whether to convert
input graphs by first resolving the backend of each input graph by looking
for an attribute named ``__networkx_backend__`` that holds the backend name for that
graph type. If all input graphs backend matches the ``backend`` kwarg, the backend's
function is called with the original inputs. If any of the input graphs do not match
the ``backend`` kwarg, they are converted to the backend graph type before calling.
Exceptions are raised if any step is not possible, e.g. if the backend does not
implement this function.

Finding a Backend
^^^^^^^^^^^^^^^^^

When a decorated function is called without a ``backend`` kwarg, it tries to find a
dispatchable backend function.
The backend type of each input graph parameter is resolved (using the
``__networkx_backend__`` attribute) and if they all agree, that backend's function
is called if possible. Otherwise the backends listed in the config ``backend_priority``
are considered one at a time in order. If that backend supports the function and
can convert the input graphs to its backend type, that backend function is called.
Otherwise the next backend is considered.

During this process, the backends can provide helpful information to the dispatcher
via helper methods in the backend's interface. Backend methods ``can_run`` and
``should_run`` are used by the dispatcher to determine whether to use the backend
function. If the number of nodes is small, it might be faster to run the NetworkX
version of the function. This is how backends can provide info about whether to run.

Falling Back to NetworkX
^^^^^^^^^^^^^^^^^^^^^^^^

If none of the backends are appropriate, we "fall back" to the NetworkX function.
That means we resolve the backends of all input graphs and if all are NetworkX
graphs we call the NetworkX function. If any are not NetworkX graphs, we raise
an exception unless the `fallback_to_nx` config is set. If it is, we convert all
graph types to NetworkX graph types before calling the NetworkX function.

Functions that mutate the graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any function decorated with the option that indicates it mutates the graph goes through
a slightly different path to automatically find backends. These functions typically
generate a graph, or add attributes or change the graph structure. The config
`backend_priority.generators` holds a list of backend names similar to the config
`backend_priority`. The process is similar for finding a matching backend. Once found,
the backend function is called and a backend graph is returned (instead of a NetworkX
graph). You can then use this backend graph in any function supported by the backend.
And you can use it for functions not supported by the backend if you set the config
`fallback_to_nx` to allow it to convert the backend graph to a NetworkX graph before
calling the function.

Optional keyword arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^

Backends can add optional keyword parameters to NetworkX functions to allow you to
control aspects of the backend algorithm. Thus the function signatures can be extended
beyond the NetworkX function signature. For example, the ``parallel`` backend might
have a parameter to specify how many CPUs to use. These parameters are collected
by the dispatchable decorator code at the start of the function call and used when
calling the backend function.

Existing Backends
^^^^^^^^^^^^^^^^^

NetworkX does not know all the backends that have been created.  In fact, the
NetworkX library does not need to know that a backend exists for it to work. As
long as the backend package creates the ``entry_point``, and provides the
correct interface, it will be called when the user requests it using one of the
three approaches described above. Some backends have been working with the
NetworkX developers to ensure smooth operation.

Refer to the :doc:`/backends` section to see a list of available backends known
to work with the current stable release of NetworkX.

.. _introspect:

Introspection and Logging
-------------------------
Introspection techniques aim to demystify dispatching and backend graph conversion behaviors.

The primary way to see what the dispatch machinery is doing is by enabling logging.
This can help you verify that the backend you specified is being used.
You can enable NetworkX's backend logger to print to ``sys.stderr`` like this::

    import logging
    nxl = logging.getLogger("networkx")
    nxl.addHandler(logging.StreamHandler())
    nxl.setLevel(logging.DEBUG)

And you can disable it by running this::

    nxl.setLevel(logging.CRITICAL)

Refer to :external+python:mod:`logging` to learn more about the logging facilities in Python.

By looking at the ``.backends`` attribute, you can get the set of all currently
installed backends that implement a particular function. For example::

    >>> nx.betweenness_centrality.backends  # doctest: +SKIP
    {'parallel'}

The function docstring will also show which installed backends support it
along with any backend-specific notes and keyword arguments::

    >>> help(nx.betweenness_centrality)  # doctest: +SKIP
    ...
    Backends
    --------
    parallel : Parallel backend for NetworkX algorithms
      The parallel computation is implemented by dividing the nodes into chunks
      and computing betweenness centrality for each chunk concurrently.
    ...

The NetworkX documentation website also includes info about trusted backends of NetworkX in function references.
For example, see :func:`~networkx.algorithms.shortest_paths.weighted.all_pairs_bellman_ford_path_length`.

Introspection capabilities are currently limited, but we are working to improve them.
We plan to make it easier to answer questions such as:

- What happened (and why)?
- What *will* happen (and why)?
- Where was time spent (including conversions)?
- What is in the cache and how much memory is it using?

Transparency is essential to allow for greater understanding, debug-ability,
and customization. After all, NetworkX dispatching is extremely flexible and can
support advanced workflows with multiple backends and fine-tuned configuration,
but introspection can be helpful by describing *when* and *how* to evolve your workflow
to meet your needs. If you have suggestions for how to improve introspection, please
`let us know <https://github.com/networkx/networkx/issues/new>`_!

Docs for backend developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a custom backend
-------------------------

1.  Defining a ``BackendInterface`` object:

    Note that the ``BackendInterface`` doesn't need to must be a class. It can be an
    instance of a class, or a module as well. You can define the following methods or
    functions in your backend's ``BackendInterface`` object.:

    1. ``convert_from_nx`` and ``convert_to_nx`` methods or functions are required for
       backend dispatching to work. The arguments to ``convert_from_nx`` are:

       - ``G`` : NetworkX Graph
       - ``edge_attrs`` : dict, optional
            Dictionary mapping edge attributes to default values if missing in ``G``.
            If None, then no edge attributes will be converted and default may be 1.
       - ``node_attrs``: dict, optional
            Dictionary mapping node attributes to default values if missing in ``G``.
            If None, then no node attributes will be converted.
       - ``preserve_edge_attrs`` : bool
            Whether to preserve all edge attributes.
       - ``preserve_node_attrs`` : bool
            Whether to preserve all node attributes.
       - ``preserve_graph_attrs`` : bool
            Whether to preserve all graph attributes.
       - ``preserve_all_attrs`` : bool
            Whether to preserve all graph, node, and edge attributes.
       - ``name`` : str
            The name of the algorithm.
       - ``graph_name`` : str
            The name of the graph argument being converted.

    2. ``can_run`` (Optional):
          If your backend only partially implements an algorithm, you can define
          a ``can_run(name, args, kwargs)`` function in your ``BackendInterface`` object that
          returns True or False indicating whether the backend can run the algorithm with
          the given arguments or not. Instead of a boolean you can also return a string
          message to inform the user why that algorithm can't be run.

    3. ``should_run`` (Optional):
          A backend may also define ``should_run(name, args, kwargs)``
          that is similar to ``can_run``, but answers whether the backend *should* be run.
          ``should_run`` is only run when performing backend graph conversions. Like
          ``can_run``, it receives the original arguments so it can decide whether it
          should be run by inspecting the arguments. ``can_run`` runs before
          ``should_run``, so ``should_run`` may assume ``can_run`` is True. If not
          implemented by the backend, ``can_run``and ``should_run`` are assumed to
          always return True if the backend implements the algorithm.

    4. ``on_start_tests`` (Optional):
          A special ``on_start_tests(items)`` function may be defined by the backend.
          It will be called with the list of NetworkX tests discovered. Each item
          is a test object that can be marked as xfail if the backend does not support
          the test using ``item.add_marker(pytest.mark.xfail(reason=...))``.

2.  Adding entry points

    To be discoverable by NetworkX, your package must register an
    `entry-point <https://packaging.python.org/en/latest/specifications/entry-points>`_
    ``networkx.backends`` in the package's metadata, with a `key pointing to your
    dispatch object <https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata>`_ .
    For example, if you are using ``setuptools`` to manage your backend package,
    you can `add the following to your pyproject.toml file <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_::

        [project.entry-points."networkx.backends"]
        backend_name = "your_backend_interface_object"

    You can also add the ``backend_info`` entry-point. It points towards the ``get_info``
    function that returns all the backend information, which is then used to build the
    "Additional Backend Implementation" box at the end of algorithm's documentation
    page. Note that the `get_info` function shouldn't import your backend package.::

        [project.entry-points."networkx.backend_info"]
        backend_name = "your_get_info_function"

    The ``get_info`` should return a dictionary with following key-value pairs:
        - ``backend_name`` : str or None
            It is the name passed in the ``backend`` kwarg.
        - ``project`` : str or None
            The name of your backend project.
        - ``package`` : str or None
            The name of your backend package.
        - ``url`` : str or None
            This is the url to either your backend's codebase or documentation, and
            will be displayed as a hyperlink to the ``backend_name``, in the
            "Additional backend implementations" section.
        - ``short_summary`` : str or None
            One line summary of your backend which will be displayed in the
            "Additional backend implementations" section.
        - ``default_config`` : dict
            A dictionary mapping the backend config parameter names to their default values.
            This is used to automatically initialize the default configs for all the
            installed backends at the time of networkx's import.

            .. seealso:: `~networkx.utils.configs.Config`

        - ``functions`` : dict or None
            A dictionary mapping function names to a dictionary of information
            about the function. The information can include the following keys:

            - ``url`` : str or None
              The url to ``function``'s source code or documentation.
            - ``additional_docs`` : str or None
              A short description or note about the backend function's
              implementation.
            - ``additional_parameters`` : dict or None
              A dictionary mapping additional parameters headers to their
              short descriptions. For example::

                  "additional_parameters": {
                      'param1 : str, function (default = "chunks")' : "...",
                      'param2 : int' : "...",
                  }

            If any of these keys are not present, the corresponding information
            will not be displayed in the "Additional backend implementations"
            section on NetworkX docs website.

        Note that your backend's docs would only appear on the official NetworkX docs only
        if your backend is a trusted backend of NetworkX, and is present in the
        `.circleci/config.yml` and `.github/workflows/deploy-docs.yml` files in the
        NetworkX repository.

3.  Defining a Backend Graph class

    The backend must create an object with an attribute ``__networkx_backend__`` that holds
    a string with the entry point name::

        class BackendGraph:
            __networkx_backend__ = "backend_name"
            ...

    A backend graph instance may have a ``G.__networkx_cache__`` dict to enable
    caching, and care should be taken to clear the cache when appropriate.

Testing the Custom backend
--------------------------

To test your custom backend, you can run the NetworkX test suite on your backend.
This also ensures that the custom backend is compatible with NetworkX's API.
The following steps will help you run the tests:

1. Setting Backend Environment Variables:
    - ``NETWORKX_TEST_BACKEND`` : Setting this to your backend's ``backend_name`` will
      let NetworkX's dispatch machinery to automatically convert a regular NetworkX
      ``Graph``, ``DiGraph``, ``MultiGraph``, etc. to their backend equivalents, using
      ``your_backend_interface_object.convert_from_nx(G, ...)`` function.
    - ``NETWORKX_FALLBACK_TO_NX`` (default=False) : Setting this variable to `True` will
      instruct tests to use a NetworkX ``Graph`` for algorithms not implemented by your
      custom backend. Setting this to `False` will only run the tests for algorithms
      implemented by your custom backend and tests for other algorithms will ``xfail``.

2. Running Tests:
    You can invoke NetworkX tests for your custom backend with the following commands::

        NETWORKX_TEST_BACKEND=<backend_name>
        NETWORKX_FALLBACK_TO_NX=True # or False
        pytest --pyargs networkx

How tests are run?
------------------

1. While dispatching to the backend implementation the ``_convert_and_call`` function
   is used and while testing the ``_convert_and_call_for_tests`` function is used.
   Other than testing it also checks for functions that return numpy scalars, and
   for functions that return graphs it runs the backend implementation and the
   networkx implementation and then converts the backend graph into a NetworkX graph
   and then compares them, and returns the networkx graph. This can be regarded as
   (pragmatic) technical debt. We may replace these checks in the future.

2. Conversions while running tests:
    - Convert NetworkX graphs using ``<your_backend_interface_object>.convert_from_nx(G, ...)`` into
      the backend graph.
    - Pass the backend graph objects to the backend implementation of the algorithm.
    - Convert the result back to a form expected by NetworkX tests using
      ``<your_backend_interface_object>.convert_to_nx(result, ...)``.
    - For nx_loopback, the graph is copied using the dispatchable metadata

3. Dispatchable algorithms that are not implemented by the backend
   will cause a ``pytest.xfail``, when the ``NETWORKX_FALLBACK_TO_NX``
   environment variable is set to ``False``, giving some indication that
   not all tests are running, while avoiding causing an explicit failure.
"""

import inspect
import itertools
import logging
import os
import warnings
from functools import partial
from importlib.metadata import entry_points

import networkx as nx

from .configs import BackendPriorities, Config, NetworkXConfig
from .decorators import argmap

__all__ = ["_dispatchable"]

_logger = logging.getLogger(__name__)


def _do_nothing():
    """This does nothing at all, yet it helps turn `_dispatchable` into functions."""


def _get_backends(group, *, load_and_call=False):
    """
    Retrieve NetworkX ``backends`` and ``backend_info`` from the entry points.

    Parameters
    -----------
    group : str
        The entry_point to be retrieved.
    load_and_call : bool, optional
        If True, load and call the backend. Defaults to False.

    Returns
    --------
    dict
        A dictionary mapping backend names to their respective backend objects.

    Notes
    ------
    If a backend is defined more than once, a warning is issued.
    The `nx_loopback` backend is removed if it exists, as it is only available during testing.
    A warning is displayed if an error occurs while loading a backend.
    """
    items = entry_points(group=group)
    rv = {}
    for ep in items:
        if ep.name in rv:
            warnings.warn(
                f"networkx backend defined more than once: {ep.name}",
                RuntimeWarning,
                stacklevel=2,
            )
        elif load_and_call:
            try:
                rv[ep.name] = ep.load()()
            except Exception as exc:
                warnings.warn(
                    f"Error encountered when loading info for backend {ep.name}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            rv[ep.name] = ep
    rv.pop("nx_loopback", None)
    return rv


# Note: "networkx" will be in `backend_info`, but not `backends` or `config.backends`.
# It is valid to use "networkx"` as backend argument and in `config.backend_priority`.
# We may make "networkx" a "proper" backend and have it in `backends` and `config.backends`.
backends = _get_backends("networkx.backends")
backend_info = {}  # fill backend_info after networkx is imported in __init__.py

# Load and cache backends on-demand
_loaded_backends = {}  # type: ignore[var-annotated]
_registered_algorithms = {}


# Get default configuration from environment variables at import time
def _comma_sep_to_list(string):
    return [stripped for x in string.strip().split(",") if (stripped := x.strip())]


def _set_configs_from_environment():
    """Initialize ``config.backend_priority``, load backend_info and config.

    This gets default values from environment variables (see ``nx.config`` for details).
    This function is run at the very end of importing networkx. It is run at this time
    to avoid loading backend_info before the rest of networkx is imported in case a
    backend uses networkx for its backend_info (e.g. subclassing the Config class.)
    """
    # backend_info is defined above as empty dict. Fill it after import finishes.
    backend_info.update(_get_backends("networkx.backend_info", load_and_call=True))
    backend_info.update(
        (backend, {}) for backend in backends.keys() - backend_info.keys()
    )

    # set up config based on backend_info and environment
    config = NetworkXConfig(
        backend_priority=BackendPriorities(
            algos=[],
            generators=[],
        ),
        backends=Config(
            **{
                backend: (
                    cfg
                    if isinstance(cfg := info["default_config"], Config)
                    else Config(**cfg)
                )
                if "default_config" in info
                else Config()
                for backend, info in backend_info.items()
            }
        ),
        cache_converted_graphs=bool(
            os.environ.get("NETWORKX_CACHE_CONVERTED_GRAPHS", True)
        ),
        fallback_to_nx=bool(os.environ.get("NETWORKX_FALLBACK_TO_NX", False)),
        warnings_to_ignore={
            x.strip()
            for x in os.environ.get("NETWORKX_WARNINGS_TO_IGNORE", "").split(",")
            if x.strip()
        },
    )
    backend_info["networkx"] = {}
    type(config.backends).__doc__ = "All installed NetworkX backends and their configs."

    # NETWORKX_BACKEND_PRIORITY is the same as NETWORKX_BACKEND_PRIORITY_ALGOS
    priorities = {
        key[26:].lower(): val
        for key, val in os.environ.items()
        if key.startswith("NETWORKX_BACKEND_PRIORITY_")
    }
    backend_priority = config.backend_priority
    backend_priority.algos = (
        _comma_sep_to_list(priorities.pop("algos"))
        if "algos" in priorities
        else _comma_sep_to_list(
            os.environ.get(
                "NETWORKX_BACKEND_PRIORITY",
                os.environ.get("NETWORKX_AUTOMATIC_BACKENDS", ""),
            )
        )
    )
    backend_priority.generators = _comma_sep_to_list(priorities.pop("generators", ""))
    for key in sorted(priorities):
        backend_priority[key] = _comma_sep_to_list(priorities[key])

    return config


def _always_run(name, args, kwargs):
    return True


def _load_backend(backend_name):
    if backend_name in _loaded_backends:
        return _loaded_backends[backend_name]
    if backend_name not in backends:
        raise ImportError(f"'{backend_name}' backend is not installed")
    rv = _loaded_backends[backend_name] = backends[backend_name].load()
    if not hasattr(rv, "can_run"):
        rv.can_run = _always_run
    if not hasattr(rv, "should_run"):
        rv.should_run = _always_run
    return rv


class _dispatchable:
    _is_testing = False

    class _fallback_to_nx:
        """Class property that returns ``nx.config.fallback_to_nx``."""

        def __get__(self, instance, owner=None):
            warnings.warn(
                "`_dispatchable._fallback_to_nx` is deprecated and will be removed "
                "in NetworkX v3.5. Use `nx.config.fallback_to_nx` instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return nx.config.fallback_to_nx

    # Note that chaining `@classmethod` and `@property` was removed in Python 3.13
    _fallback_to_nx = _fallback_to_nx()  # type: ignore[assignment,misc]

    def __new__(
        cls,
        func=None,
        *,
        name=None,
        graphs="G",
        edge_attrs=None,
        node_attrs=None,
        preserve_edge_attrs=False,
        preserve_node_attrs=False,
        preserve_graph_attrs=False,
        preserve_all_attrs=False,
        mutates_input=False,
        returns_graph=False,
    ):
        """A decorator function that is used to redirect the execution of ``func``
        function to its backend implementation.

        This decorator function dispatches to
        a different backend implementation based on the input graph types, and it also
        manages all the ``backend_kwargs``. Usage can be any of the following decorator
        forms:

        - ``@_dispatchable``
        - ``@_dispatchable()``
        - ``@_dispatchable(name="override_name")``
        - ``@_dispatchable(graphs="graph_var_name")``
        - ``@_dispatchable(edge_attrs="weight")``
        - ``@_dispatchable(graphs={"G": 0, "H": 1}, edge_attrs={"weight": "default"})``
            with 0 and 1 giving the position in the signature function for graph
            objects. When ``edge_attrs`` is a dict, keys are keyword names and values
            are defaults.

        Parameters
        ----------
        func : callable, optional
            The function to be decorated. If ``func`` is not provided, returns a
            partial object that can be used to decorate a function later. If ``func``
            is provided, returns a new callable object that dispatches to a backend
            algorithm based on input graph types.

        name : str, optional
            The name of the algorithm to use for dispatching. If not provided,
            the name of ``func`` will be used. ``name`` is useful to avoid name
            conflicts, as all dispatched algorithms live in a single namespace.
            For example, ``tournament.is_strongly_connected`` had a name conflict
            with the standard ``nx.is_strongly_connected``, so we used
            ``@_dispatchable(name="tournament_is_strongly_connected")``.

        graphs : str or dict or None, default "G"
            If a string, the parameter name of the graph, which must be the first
            argument of the wrapped function. If more than one graph is required
            for the algorithm (or if the graph is not the first argument), provide
            a dict keyed to argument names with argument position as values for each
            graph argument. For example, ``@_dispatchable(graphs={"G": 0, "auxiliary?": 4})``
            indicates the 0th parameter ``G`` of the function is a required graph,
            and the 4th parameter ``auxiliary?`` is an optional graph.
            To indicate that an argument is a list of graphs, do ``"[graphs]"``.
            Use ``graphs=None``, if *no* arguments are NetworkX graphs such as for
            graph generators, readers, and conversion functions.

        edge_attrs : str or dict, optional
            ``edge_attrs`` holds information about edge attribute arguments
            and default values for those edge attributes.
            If a string, ``edge_attrs`` holds the function argument name that
            indicates a single edge attribute to include in the converted graph.
            The default value for this attribute is 1. To indicate that an argument
            is a list of attributes (all with default value 1), use e.g. ``"[attrs]"``.
            If a dict, ``edge_attrs`` holds a dict keyed by argument names, with
            values that are either the default value or, if a string, the argument
            name that indicates the default value.

        node_attrs : str or dict, optional
            Like ``edge_attrs``, but for node attributes.

        preserve_edge_attrs : bool or str or dict, optional
            For bool, whether to preserve all edge attributes.
            For str, the parameter name that may indicate (with ``True`` or a
            callable argument) whether all edge attributes should be preserved
            when converting.
            For dict of ``{graph_name: {attr: default}}``, indicate pre-determined
            edge attributes (and defaults) to preserve for input graphs.

        preserve_node_attrs : bool or str or dict, optional
            Like ``preserve_edge_attrs``, but for node attributes.

        preserve_graph_attrs : bool or set
            For bool, whether to preserve all graph attributes.
            For set, which input graph arguments to preserve graph attributes.

        preserve_all_attrs : bool
            Whether to preserve all edge, node and graph attributes.
            This overrides all the other preserve_*_attrs.

        mutates_input : bool or dict, default False
            For bool, whether the function mutates an input graph argument.
            For dict of ``{arg_name: arg_pos}``, arguments that indicate whether an
            input graph will be mutated, and ``arg_name`` may begin with ``"not "``
            to negate the logic (for example, this is used by ``copy=`` arguments).
            By default, dispatching doesn't convert input graphs to a different
            backend for functions that mutate input graphs.

        returns_graph : bool, default False
            Whether the function can return or yield a graph object. By default,
            dispatching doesn't convert input graphs to a different backend for
            functions that return graphs.
        """
        if func is None:
            return partial(
                _dispatchable,
                name=name,
                graphs=graphs,
                edge_attrs=edge_attrs,
                node_attrs=node_attrs,
                preserve_edge_attrs=preserve_edge_attrs,
                preserve_node_attrs=preserve_node_attrs,
                preserve_graph_attrs=preserve_graph_attrs,
                preserve_all_attrs=preserve_all_attrs,
                mutates_input=mutates_input,
                returns_graph=returns_graph,
            )
        if isinstance(func, str):
            raise TypeError("'name' and 'graphs' must be passed by keyword") from None
        # If name not provided, use the name of the function
        if name is None:
            name = func.__name__

        self = object.__new__(cls)

        # standard function-wrapping stuff
        # __annotations__ not used
        self.__name__ = func.__name__
        # self.__doc__ = func.__doc__  # __doc__ handled as cached property
        self.__defaults__ = func.__defaults__
        # We "magically" add `backend=` keyword argument to allow backend to be specified
        if func.__kwdefaults__:
            self.__kwdefaults__ = {**func.__kwdefaults__, "backend": None}
        else:
            self.__kwdefaults__ = {"backend": None}
        self.__module__ = func.__module__
        self.__qualname__ = func.__qualname__
        self.__dict__.update(func.__dict__)
        self.__wrapped__ = func

        # Supplement docstring with backend info; compute and cache when needed
        self._orig_doc = func.__doc__
        self._cached_doc = None

        self.orig_func = func
        self.name = name
        self.edge_attrs = edge_attrs
        self.node_attrs = node_attrs
        self.preserve_edge_attrs = preserve_edge_attrs or preserve_all_attrs
        self.preserve_node_attrs = preserve_node_attrs or preserve_all_attrs
        self.preserve_graph_attrs = preserve_graph_attrs or preserve_all_attrs
        self.mutates_input = mutates_input
        # Keep `returns_graph` private for now, b/c we may extend info on return types
        self._returns_graph = returns_graph

        if edge_attrs is not None and not isinstance(edge_attrs, str | dict):
            raise TypeError(
                f"Bad type for edge_attrs: {type(edge_attrs)}. Expected str or dict."
            ) from None
        if node_attrs is not None and not isinstance(node_attrs, str | dict):
            raise TypeError(
                f"Bad type for node_attrs: {type(node_attrs)}. Expected str or dict."
            ) from None
        if not isinstance(self.preserve_edge_attrs, bool | str | dict):
            raise TypeError(
                f"Bad type for preserve_edge_attrs: {type(self.preserve_edge_attrs)}."
                " Expected bool, str, or dict."
            ) from None
        if not isinstance(self.preserve_node_attrs, bool | str | dict):
            raise TypeError(
                f"Bad type for preserve_node_attrs: {type(self.preserve_node_attrs)}."
                " Expected bool, str, or dict."
            ) from None
        if not isinstance(self.preserve_graph_attrs, bool | set):
            raise TypeError(
                f"Bad type for preserve_graph_attrs: {type(self.preserve_graph_attrs)}."
                " Expected bool or set."
            ) from None
        if not isinstance(self.mutates_input, bool | dict):
            raise TypeError(
                f"Bad type for mutates_input: {type(self.mutates_input)}."
                " Expected bool or dict."
            ) from None
        if not isinstance(self._returns_graph, bool):
            raise TypeError(
                f"Bad type for returns_graph: {type(self._returns_graph)}."
                " Expected bool."
            ) from None

        if isinstance(graphs, str):
            graphs = {graphs: 0}
        elif graphs is None:
            pass
        elif not isinstance(graphs, dict):
            raise TypeError(
                f"Bad type for graphs: {type(graphs)}. Expected str or dict."
            ) from None
        elif len(graphs) == 0:
            raise KeyError("'graphs' must contain at least one variable name") from None

        # This dict comprehension is complicated for better performance; equivalent shown below.
        self.optional_graphs = set()
        self.list_graphs = set()
        if graphs is None:
            self.graphs = {}
        else:
            self.graphs = {
                self.optional_graphs.add(val := k[:-1]) or val
                if (last := k[-1]) == "?"
                else self.list_graphs.add(val := k[1:-1]) or val
                if last == "]"
                else k: v
                for k, v in graphs.items()
            }
        # The above is equivalent to:
        # self.optional_graphs = {k[:-1] for k in graphs if k[-1] == "?"}
        # self.list_graphs = {k[1:-1] for k in graphs if k[-1] == "]"}
        # self.graphs = {k[:-1] if k[-1] == "?" else k: v for k, v in graphs.items()}

        # Compute and cache the signature on-demand
        self._sig = None

        # Which backends implement this function?
        self.backends = {
            backend
            for backend, info in backend_info.items()
            if "functions" in info and name in info["functions"]
        }

        if name in _registered_algorithms:
            raise KeyError(
                f"Algorithm already exists in dispatch registry: {name}"
            ) from None
        # Use the magic of `argmap` to turn `self` into a function. This does result
        # in small additional overhead compared to calling `_dispatchable` directly,
        # but `argmap` has the magical property that it can stack with other `argmap`
        # decorators "for free". Being a function is better for REPRs and type-checkers.
        self = argmap(_do_nothing)(self)
        _registered_algorithms[name] = self
        return self

    @property
    def __doc__(self):
        """If the cached documentation exists, it is returned.
        Otherwise, the documentation is generated using _make_doc() method,
        cached, and then returned."""

        if (rv := self._cached_doc) is not None:
            return rv
        rv = self._cached_doc = self._make_doc()
        return rv

    @__doc__.setter
    def __doc__(self, val):
        """Sets the original documentation to the given value and resets the
        cached documentation."""

        self._orig_doc = val
        self._cached_doc = None

    @property
    def __signature__(self):
        """Return the signature of the original function, with the addition of
        the `backend` and `backend_kwargs` parameters."""

        if self._sig is None:
            sig = inspect.signature(self.orig_func)
            # `backend` is now a reserved argument used by dispatching.
            # assert "backend" not in sig.parameters
            if not any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                sig = sig.replace(
                    parameters=[
                        *sig.parameters.values(),
                        inspect.Parameter(
                            "backend", inspect.Parameter.KEYWORD_ONLY, default=None
                        ),
                        inspect.Parameter(
                            "backend_kwargs", inspect.Parameter.VAR_KEYWORD
                        ),
                    ]
                )
            else:
                *parameters, var_keyword = sig.parameters.values()
                sig = sig.replace(
                    parameters=[
                        *parameters,
                        inspect.Parameter(
                            "backend", inspect.Parameter.KEYWORD_ONLY, default=None
                        ),
                        var_keyword,
                    ]
                )
            self._sig = sig
        return self._sig

    def __call__(self, /, *args, backend=None, **kwargs):
        """Returns the result of the original function, or the backend function if
        the backend is specified and that backend implements `func`."""

        if not backends:
            # Fast path if no backends are installed
            if backend is not None and backend != "networkx":
                raise ImportError(f"'{backend}' backend is not installed")
            return self.orig_func(*args, **kwargs)

        # Use `backend_name` in this function instead of `backend`.
        # This is purely for aesthetics and to make it easier to search for this
        # variable since "backend" is used in many comments and log/error messages.
        backend_name = backend
        if backend_name is not None and backend_name not in backend_info:
            raise ImportError(f"'{backend_name}' backend is not installed")

        graphs_resolved = {}
        for gname, pos in self.graphs.items():
            if pos < len(args):
                if gname in kwargs:
                    raise TypeError(f"{self.name}() got multiple values for {gname!r}")
                graph = args[pos]
            elif gname in kwargs:
                graph = kwargs[gname]
            elif gname not in self.optional_graphs:
                raise TypeError(
                    f"{self.name}() missing required graph argument: {gname}"
                )
            else:
                continue
            if graph is None:
                if gname not in self.optional_graphs:
                    raise TypeError(
                        f"{self.name}() required graph argument {gname!r} is None; must be a graph"
                    )
            else:
                graphs_resolved[gname] = graph

        # Alternative to the above that does not check duplicated args or missing required graphs.
        # graphs_resolved = {
        #     gname: graph
        #     for gname, pos in self.graphs.items()
        #     if (graph := args[pos] if pos < len(args) else kwargs.get(gname)) is not None
        # }

        # Check if any graph comes from a backend
        if self.list_graphs:
            # Make sure we don't lose values by consuming an iterator
            args = list(args)
            for gname in self.list_graphs & graphs_resolved.keys():
                list_of_graphs = list(graphs_resolved[gname])
                graphs_resolved[gname] = list_of_graphs
                if gname in kwargs:
                    kwargs[gname] = list_of_graphs
                else:
                    args[self.graphs[gname]] = list_of_graphs

            graph_backend_names = {
                getattr(g, "__networkx_backend__", None)
                for gname, g in graphs_resolved.items()
                if gname not in self.list_graphs
            }
            for gname in self.list_graphs & graphs_resolved.keys():
                graph_backend_names.update(
                    getattr(g, "__networkx_backend__", None)
                    for g in graphs_resolved[gname]
                )
        else:
            graph_backend_names = {
                getattr(g, "__networkx_backend__", None)
                for g in graphs_resolved.values()
            }

        backend_priority = nx.config.backend_priority.get(
            self.name,
            nx.config.backend_priority.generators
            if self._returns_graph
            else nx.config.backend_priority.algos,
        )
        if self._is_testing and backend_priority and backend_name is None:
            # Special path if we are running networkx tests with a backend.
            # This even runs for (and handles) functions that mutate input graphs.
            return self._convert_and_call_for_tests(
                backend_priority[0],
                args,
                kwargs,
                fallback_to_nx=nx.config.fallback_to_nx,
            )

        graph_backend_names.discard(None)
        if backend_name is not None:
            # Must run with the given backend.
            # `can_run` only used for better log and error messages.
            # Check `mutates_input` for logging, not behavior.
            blurb = (
                "No other backends will be attempted, because the backend was "
                f"specified with the `backend='{backend_name}'` keyword argument."
            )
            extra_message = (
                f"'{backend_name}' backend raised NotImplementedError when calling "
                f"`{self.name}'. {blurb}"
            )
            if not graph_backend_names or graph_backend_names == {backend_name}:
                # All graphs are backend graphs--no need to convert!
                if self._can_backend_run(backend_name, args, kwargs):
                    return self._call_with_backend(
                        backend_name, args, kwargs, extra_message=extra_message
                    )
                if self._does_backend_have(backend_name):
                    extra = " for the given arguments"
                else:
                    extra = ""
                raise NotImplementedError(
                    f"`{self.name}' is not implemented by '{backend_name}' backend"
                    f"{extra}. {blurb}"
                )
            if self._can_convert(backend_name, graph_backend_names):
                if self._can_backend_run(backend_name, args, kwargs):
                    if self._will_call_mutate_input(args, kwargs):
                        _logger.debug(
                            "`%s' will mutate an input graph. This prevents automatic conversion "
                            "to, and use of, backends listed in `nx.config.backend_priority`. "
                            "Using backend specified by the "
                            "`backend='%s'` keyword argument. This may change behavior by not "
                            "mutating inputs.",
                            self.name,
                            backend_name,
                        )
                        mutations = []
                    else:
                        mutations = None
                    rv = self._convert_and_call(
                        backend_name,
                        graph_backend_names,
                        args,
                        kwargs,
                        extra_message=extra_message,
                        mutations=mutations,
                    )
                    if mutations:
                        for cache, key in mutations:
                            # If the call mutates inputs, then remove all inputs gotten
                            # from cache. We do this after all conversions (and call) so
                            # that a graph can be gotten from a cache multiple times.
                            cache.pop(key, None)
                    return rv
                if self._does_backend_have(backend_name):
                    extra = " for the given arguments"
                else:
                    extra = ""
                raise NotImplementedError(
                    f"`{self.name}' is not implemented by '{backend_name}' backend"
                    f"{extra}. {blurb}"
                )
            if len(graph_backend_names) == 1:
                maybe_s = ""
                graph_backend_names = f"'{next(iter(graph_backend_names))}'"
            else:
                maybe_s = "s"
            raise TypeError(
                f"`{self.name}' is unable to convert graph from backend{maybe_s} "
                f"{graph_backend_names} to '{backend_name}' backend, which was "
                f"specified with the `backend='{backend_name}'` keyword argument. "
                f"{blurb}"
            )

        if self._will_call_mutate_input(args, kwargs):
            # The current behavior for functions that mutate input graphs:
            #
            # 1. If backend is specified by `backend=` keyword, use it (done above).
            # 2. If inputs are from one backend, try to use it.
            # 3. If all input graphs are instances of `nx.Graph`, then run with the
            #    default "networkx" implementation.
            #
            # Do not automatically convert if a call will mutate inputs, because doing
            # so would change behavior. Hence, we should fail if there are multiple input
            # backends or if the input backend does not implement the function. However,
            # we offer a way for backends to circumvent this if they do not implement
            # this function: we will fall back to the default "networkx" implementation
            # without using conversions if all input graphs are subclasses of `nx.Graph`.
            blurb = (
                "conversions between backends (if configured) will not be attempted, "
                "because this may change behavior. You may specify a backend to use "
                "by passing e.g. `backend='networkx'` keyword, but this may also "
                "change behavior by not mutating inputs."
            )
            fallback_blurb = (
                "This call will mutate inputs, so fall back to 'networkx' "
                "backend (without converting) since all input graphs are "
                "instances of nx.Graph and are hopefully compatible.",
            )
            if len(graph_backend_names) == 1:
                [backend_name] = graph_backend_names
                msg_template = (
                    f"Backend '{backend_name}' does not implement `{self.name}'%s. "
                    f"This call will mutate an input, so automatic {blurb}"
                )
                # `can_run` is only used for better log and error messages
                try:
                    if self._can_backend_run(backend_name, args, kwargs):
                        return self._call_with_backend(
                            backend_name,
                            args,
                            kwargs,
                            extra_message=msg_template % " with these arguments",
                        )
                except NotImplementedError as exc:
                    if all(isinstance(g, nx.Graph) for g in graphs_resolved.values()):
                        _logger.debug(
                            "Backend '%s' raised when calling `%s': %s. %s",
                            backend_name,
                            self.name,
                            exc,
                            fallback_blurb,
                        )
                    else:
                        raise
                else:
                    if nx.config.fallback_to_nx and all(
                        # Consider dropping the `isinstance` check here to allow
                        # duck-type graphs, but let's wait for a backend to ask us.
                        isinstance(g, nx.Graph)
                        for g in graphs_resolved.values()
                    ):
                        # Log that we are falling back to networkx
                        _logger.debug(
                            "Backend '%s' can't run `%s'. %s",
                            backend_name,
                            self.name,
                            fallback_blurb,
                        )
                    else:
                        if self._does_backend_have(backend_name):
                            extra = " with these arguments"
                        else:
                            extra = ""
                        raise NotImplementedError(msg_template % extra)
            elif nx.config.fallback_to_nx and all(
                # Consider dropping the `isinstance` check here to allow
                # duck-type graphs, but let's wait for a backend to ask us.
                isinstance(g, nx.Graph)
                for g in graphs_resolved.values()
            ):
                # Log that we are falling back to networkx
                _logger.debug(
                    "`%s' was called with inputs from multiple backends: %s. %s",
                    self.name,
                    graph_backend_names,
                    fallback_blurb,
                )
            else:
                raise RuntimeError(
                    f"`{self.name}' will mutate an input, but it was called with inputs "
                    f"from multiple backends: {graph_backend_names}. Automatic {blurb}"
                )
            # At this point, no backends are available to handle the call with
            # the input graph types, but if the input graphs are compatible
            # nx.Graph instances, fall back to networkx without converting.
            return self.orig_func(*args, **kwargs)

        # We may generalize fallback configuration as e.g. `nx.config.backend_fallback`
        if nx.config.fallback_to_nx or not graph_backend_names:
            # Use "networkx" by default if there are no inputs from backends.
            # For example, graph generators should probably return NetworkX graphs
            # instead of raising NotImplementedError.
            backend_fallback = ["networkx"]
        else:
            backend_fallback = []

        # ##########################
        # # How this behaves today #
        # ##########################
        #
        # The prose below describes the implementation and a *possible* way to
        # generalize "networkx" as "just another backend". The code is structured
        # to perhaps someday support backend-to-backend conversions (including
        # simply passing objects from one backend directly to another backend;
        # the dispatch machinery does not necessarily need to perform conversions),
        # but since backend-to-backend matching is not yet supported, the following
        # code is merely a convenient way to implement dispatch behaviors that have
        # been carefully developed since NetworkX 3.0 and to include falling back
        # to the default NetworkX implementation.
        #
        # The current behavior for functions that don't mutate input graphs:
        #
        # 1. If backend is specified by `backend=` keyword, use it (done above).
        # 2. If input is from a backend other than "networkx", try to use it.
        #    - Note: if present, "networkx" graphs will be converted to the backend.
        # 3. If input is from "networkx" (or no backend), try to use backends from
        #    `backend_priority` before running with the default "networkx" implementation.
        # 4. If configured, "fall back" and run with the default "networkx" implementation.
        #
        # ################################################
        # # How this is implemented and may work someday #
        # ################################################
        #
        # Let's determine the order of backends we should try according
        # to `backend_priority`, `backend_fallback`, and input backends.
        # There are two† dimensions of priorities to consider:
        #   backend_priority > unspecified > backend_fallback
        # and
        #   backend of an input > not a backend of an input
        # These are combined to form five groups of priorities as such:
        #
        #                    input   ~input
        #                  +-------+-------+
        # backend_priority |   1   |   2   |
        #      unspecified |   3   |  N/A  | (if only 1)
        # backend_fallback |   4   |   5   |
        #                  +-------+-------+
        #
        # This matches the behaviors we developed in versions 3.0 to 3.2, it
        # ought to cover virtually all use cases we expect, and I (@eriknw) don't
        # think it can be done any simpler (although it can be generalized further
        # and made to be more complicated to capture 100% of *possible* use cases).
        # Some observations:
        #
        #   1. If an input is in `backend_priority`, it will be used before trying a
        #      backend that is higher priority in `backend_priority` and not an input.
        #   2. To prioritize converting from one backend to another even if both implement
        #      a function, list one in `backend_priority` and one in `backend_fallback`.
        #   3. To disable conversions, set `backend_priority` and `backend_fallback` to [].
        #
        # †: There is actually a third dimension of priorities:
        #        should_run == True > should_run == False
        #    Backends with `can_run == True` and `should_run == False` are tried last.
        #
        seen = set()
        group1 = []  # In backend_priority, and an input
        group2 = []  # In backend_priority, but not an input
        for name in backend_priority:
            if name in seen:
                continue
            seen.add(name)
            if name in graph_backend_names:
                group1.append(name)
            else:
                group2.append(name)
        group4 = []  # In backend_fallback, and an input
        group5 = []  # In backend_fallback, but not an input
        for name in backend_fallback:
            if name in seen:
                continue
            seen.add(name)
            if name in graph_backend_names:
                group4.append(name)
            else:
                group5.append(name)
        # An input, but not in backend_priority or backend_fallback.
        group3 = graph_backend_names - seen
        if len(group3) > 1:
            # `group3` backends are not configured for automatic conversion or fallback.
            # There are at least two issues if this group contains multiple backends:
            #
            #   1. How should we prioritize them? We have no good way to break ties.
            #      Although we could arbitrarily choose alphabetical or left-most,
            #      let's follow the Zen of Python and refuse the temptation to guess.
            #   2. We probably shouldn't automatically convert to these backends,
            #      because we are not configured to do so.
            #
            # (2) is important to allow disabling all conversions by setting both
            # `nx.config.backend_priority` and `nx.config.backend_fallback` to [].
            #
            # If there is a single backend in `group3`, then giving it priority over
            # the fallback backends is what is generally expected. For example, this
            # allows input graphs of `backend_fallback` backends (such as "networkx")
            # to be converted to, and run with, the unspecified backend.
            _logger.debug(
                "Call to `%s' has inputs from multiple backends, %s, that "
                "have no priority set in `nx.config.backend_priority`, "
                "so automatic conversions to "
                "these backends will not be attempted.",
                self.name,
                group3,
            )
            group3 = ()

        try_order = list(itertools.chain(group1, group2, group3, group4, group5))
        if len(try_order) > 1:
            # Should we consider adding an option for more verbose logging?
            # For example, we could explain the order of `try_order` in detail.
            _logger.debug(
                "Call to `%s' has inputs from %s backends, and will try to use "
                "backends in the following order: %s",
                self.name,
                graph_backend_names or "no",
                try_order,
            )
        backends_to_try_again = []
        for is_not_first, backend_name in enumerate(try_order):
            if is_not_first:
                _logger.debug("Trying next backend: '%s'", backend_name)
            try:
                if not graph_backend_names or graph_backend_names == {backend_name}:
                    if self._can_backend_run(backend_name, args, kwargs):
                        return self._call_with_backend(backend_name, args, kwargs)
                elif self._can_convert(
                    backend_name, graph_backend_names
                ) and self._can_backend_run(backend_name, args, kwargs):
                    if self._should_backend_run(backend_name, args, kwargs):
                        rv = self._convert_and_call(
                            backend_name, graph_backend_names, args, kwargs
                        )
                        if (
                            self._returns_graph
                            and graph_backend_names
                            and backend_name not in graph_backend_names
                        ):
                            # If the function has graph inputs and graph output, we try
                            # to make it so the backend of the return type will match the
                            # backend of the input types. In case this is not possible,
                            # let's tell the user that the backend of the return graph
                            # has changed. Perhaps we could try to convert back, but
                            # "fallback" backends for graph generators should typically
                            # be compatible with NetworkX graphs.
                            _logger.debug(
                                "Call to `%s' is returning a graph from a different "
                                "backend! It has inputs from %s backends, but ran with "
                                "'%s' backend and is returning graph from '%s' backend",
                                self.name,
                                graph_backend_names,
                                backend_name,
                                backend_name,
                            )
                        return rv
                    # `should_run` is False, but `can_run` is True, so try again later
                    backends_to_try_again.append(backend_name)
            except NotImplementedError as exc:
                _logger.debug(
                    "Backend '%s' raised when calling `%s': %s",
                    backend_name,
                    self.name,
                    exc,
                )

        # We are about to fail. Let's try backends with can_run=True and should_run=False.
        # This is unlikely to help today since we try to run with "networkx" before this.
        for backend_name in backends_to_try_again:
            _logger.debug(
                "Trying backend: '%s' (ignoring `should_run=False`)", backend_name
            )
            try:
                rv = self._convert_and_call(
                    backend_name, graph_backend_names, args, kwargs
                )
                if (
                    self._returns_graph
                    and graph_backend_names
                    and backend_name not in graph_backend_names
                ):
                    _logger.debug(
                        "Call to `%s' is returning a graph from a different "
                        "backend! It has inputs from %s backends, but ran with "
                        "'%s' backend and is returning graph from '%s' backend",
                        self.name,
                        graph_backend_names,
                        backend_name,
                        backend_name,
                    )
                return rv
            except NotImplementedError as exc:
                _logger.debug(
                    "Backend '%s' raised when calling `%s': %s",
                    backend_name,
                    self.name,
                    exc,
                )
        # As a final effort, we could try to convert and run with `group3` backends
        # that we discarded when `len(group3) > 1`, but let's not consider doing
        # so until there is a reasonable request for it.

        if len(unspecified_backends := graph_backend_names - seen) > 1:
            raise TypeError(
                f"Unable to convert inputs from {graph_backend_names} backends and "
                f"run `{self.name}'. NetworkX is configured to automatically convert "
                f"to {try_order} backends. To remedy this, you may enable automatic "
                f"conversion to {unspecified_backends} backends by adding them to "
                "`nx.config.backend_priority`, or you "
                "may specify a backend to use with the `backend=` keyword argument."
            )
        raise NotImplementedError(
            f"`{self.name}' is not implemented by {try_order} backends. To remedy "
            "this, you may enable automatic conversion to more backends (including "
            "'networkx') by adding them to `nx.config.backend_priority`, "
            "or you may specify a backend to use with "
            "the `backend=` keyword argument."
        )

    def _will_call_mutate_input(self, args, kwargs):
        return (mutates_input := self.mutates_input) and (
            mutates_input is True
            or any(
                # If `mutates_input` begins with "not ", then assume the argument is bool,
                # otherwise treat it as a node or edge attribute if it's not None.
                not (
                    args[arg_pos]
                    if len(args) > arg_pos
                    # This assumes that e.g. `copy=True` is the default
                    else kwargs.get(arg_name[4:], True)
                )
                if arg_name.startswith("not ")
                else (args[arg_pos] if len(args) > arg_pos else kwargs.get(arg_name))
                is not None
                for arg_name, arg_pos in mutates_input.items()
            )
        )

    def _can_convert(self, backend_name, graph_backend_names):
        # Backend-to-backend conversion not supported yet.
        # We can only convert to and from networkx.
        rv = backend_name == "networkx" or graph_backend_names.issubset(
            {"networkx", backend_name}
        )
        if not rv:
            _logger.debug(
                "Unable to convert from %s backends to '%s' backend",
                graph_backend_names,
                backend_name,
            )
        return rv

    def _does_backend_have(self, backend_name):
        """Does the specified backend have this algorithm?"""
        if backend_name == "networkx":
            return True
        # Inspect the backend; don't trust metadata used to create `self.backends`
        backend = _load_backend(backend_name)
        return hasattr(backend, self.name)

    def _can_backend_run(self, backend_name, args, kwargs):
        """Can the specified backend run this algorithm with these arguments?"""
        if backend_name == "networkx":
            return True
        backend = _load_backend(backend_name)
        # `backend.can_run` and `backend.should_run` may return strings that describe
        # why they can't or shouldn't be run.
        if not hasattr(backend, self.name):
            _logger.debug(
                "Backend '%s' does not implement `%s'", backend_name, self.name
            )
            return False
        can_run = backend.can_run(self.name, args, kwargs)
        if isinstance(can_run, str) or not can_run:
            reason = f", because: {can_run}" if isinstance(can_run, str) else ""
            _logger.debug(
                "Backend '%s' can't run `%s` with arguments: %s%s",
                backend_name,
                self.name,
                _LazyArgsRepr(self, args, kwargs),
                reason,
            )
            return False
        return True

    def _should_backend_run(self, backend_name, args, kwargs):
        """Should the specified backend run this algorithm with these arguments?

        Note that this does not check ``backend.can_run``.
        """
        # `backend.can_run` and `backend.should_run` may return strings that describe
        # why they can't or shouldn't be run.
        if backend_name == "networkx":
            return True
        backend = _load_backend(backend_name)
        should_run = backend.should_run(self.name, args, kwargs)
        if isinstance(should_run, str) or not should_run:
            reason = f", because: {should_run}" if isinstance(should_run, str) else ""
            _logger.debug(
                "Backend '%s' shouldn't run `%s` with arguments: %s%s",
                backend_name,
                self.name,
                _LazyArgsRepr(self, args, kwargs),
                reason,
            )
            return False
        return True

    def _convert_arguments(self, backend_name, args, kwargs, *, use_cache, mutations):
        """Convert graph arguments to the specified backend.

        Returns
        -------
        args tuple and kwargs dict
        """
        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        if not self.graphs:
            bound_kwargs = bound.kwargs
            del bound_kwargs["backend"]
            return bound.args, bound_kwargs
        if backend_name == "networkx":
            # `backend_interface.convert_from_nx` preserves everything
            preserve_edge_attrs = preserve_node_attrs = preserve_graph_attrs = True
        else:
            preserve_edge_attrs = self.preserve_edge_attrs
            preserve_node_attrs = self.preserve_node_attrs
            preserve_graph_attrs = self.preserve_graph_attrs
            edge_attrs = self.edge_attrs
            node_attrs = self.node_attrs
        # Convert graphs into backend graph-like object
        # Include the edge and/or node labels if provided to the algorithm
        if preserve_edge_attrs is False:
            # e.g. `preserve_edge_attrs=False`
            pass
        elif preserve_edge_attrs is True:
            # e.g. `preserve_edge_attrs=True`
            edge_attrs = None
        elif isinstance(preserve_edge_attrs, str):
            if bound.arguments[preserve_edge_attrs] is True or callable(
                bound.arguments[preserve_edge_attrs]
            ):
                # e.g. `preserve_edge_attrs="attr"` and `func(attr=True)`
                # e.g. `preserve_edge_attrs="attr"` and `func(attr=myfunc)`
                preserve_edge_attrs = True
                edge_attrs = None
            elif bound.arguments[preserve_edge_attrs] is False and (
                isinstance(edge_attrs, str)
                and edge_attrs == preserve_edge_attrs
                or isinstance(edge_attrs, dict)
                and preserve_edge_attrs in edge_attrs
            ):
                # e.g. `preserve_edge_attrs="attr"` and `func(attr=False)`
                # Treat `False` argument as meaning "preserve_edge_data=False"
                # and not `False` as the edge attribute to use.
                preserve_edge_attrs = False
                edge_attrs = None
            else:
                # e.g. `preserve_edge_attrs="attr"` and `func(attr="weight")`
                preserve_edge_attrs = False
        # Else: e.g. `preserve_edge_attrs={"G": {"weight": 1}}`

        if edge_attrs is None:
            # May have been set to None above b/c all attributes are preserved
            pass
        elif isinstance(edge_attrs, str):
            if edge_attrs[0] == "[":
                # e.g. `edge_attrs="[edge_attributes]"` (argument of list of attributes)
                # e.g. `func(edge_attributes=["foo", "bar"])`
                edge_attrs = {
                    edge_attr: 1 for edge_attr in bound.arguments[edge_attrs[1:-1]]
                }
            elif callable(bound.arguments[edge_attrs]):
                # e.g. `edge_attrs="weight"` and `func(weight=myfunc)`
                preserve_edge_attrs = True
                edge_attrs = None
            elif bound.arguments[edge_attrs] is not None:
                # e.g. `edge_attrs="weight"` and `func(weight="foo")` (default of 1)
                edge_attrs = {bound.arguments[edge_attrs]: 1}
            elif self.name == "to_numpy_array" and hasattr(
                bound.arguments["dtype"], "names"
            ):
                # Custom handling: attributes may be obtained from `dtype`
                edge_attrs = {
                    edge_attr: 1 for edge_attr in bound.arguments["dtype"].names
                }
            else:
                # e.g. `edge_attrs="weight"` and `func(weight=None)`
                edge_attrs = None
        else:
            # e.g. `edge_attrs={"attr": "default"}` and `func(attr="foo", default=7)`
            # e.g. `edge_attrs={"attr": 0}` and `func(attr="foo")`
            edge_attrs = {
                edge_attr: bound.arguments.get(val, 1) if isinstance(val, str) else val
                for key, val in edge_attrs.items()
                if (edge_attr := bound.arguments[key]) is not None
            }

        if preserve_node_attrs is False:
            # e.g. `preserve_node_attrs=False`
            pass
        elif preserve_node_attrs is True:
            # e.g. `preserve_node_attrs=True`
            node_attrs = None
        elif isinstance(preserve_node_attrs, str):
            if bound.arguments[preserve_node_attrs] is True or callable(
                bound.arguments[preserve_node_attrs]
            ):
                # e.g. `preserve_node_attrs="attr"` and `func(attr=True)`
                # e.g. `preserve_node_attrs="attr"` and `func(attr=myfunc)`
                preserve_node_attrs = True
                node_attrs = None
            elif bound.arguments[preserve_node_attrs] is False and (
                isinstance(node_attrs, str)
                and node_attrs == preserve_node_attrs
                or isinstance(node_attrs, dict)
                and preserve_node_attrs in node_attrs
            ):
                # e.g. `preserve_node_attrs="attr"` and `func(attr=False)`
                # Treat `False` argument as meaning "preserve_node_data=False"
                # and not `False` as the node attribute to use. Is this used?
                preserve_node_attrs = False
                node_attrs = None
            else:
                # e.g. `preserve_node_attrs="attr"` and `func(attr="weight")`
                preserve_node_attrs = False
        # Else: e.g. `preserve_node_attrs={"G": {"pos": None}}`

        if node_attrs is None:
            # May have been set to None above b/c all attributes are preserved
            pass
        elif isinstance(node_attrs, str):
            if node_attrs[0] == "[":
                # e.g. `node_attrs="[node_attributes]"` (argument of list of attributes)
                # e.g. `func(node_attributes=["foo", "bar"])`
                node_attrs = {
                    node_attr: None for node_attr in bound.arguments[node_attrs[1:-1]]
                }
            elif callable(bound.arguments[node_attrs]):
                # e.g. `node_attrs="weight"` and `func(weight=myfunc)`
                preserve_node_attrs = True
                node_attrs = None
            elif bound.arguments[node_attrs] is not None:
                # e.g. `node_attrs="weight"` and `func(weight="foo")`
                node_attrs = {bound.arguments[node_attrs]: None}
            else:
                # e.g. `node_attrs="weight"` and `func(weight=None)`
                node_attrs = None
        else:
            # e.g. `node_attrs={"attr": "default"}` and `func(attr="foo", default=7)`
            # e.g. `node_attrs={"attr": 0}` and `func(attr="foo")`
            node_attrs = {
                node_attr: bound.arguments.get(val) if isinstance(val, str) else val
                for key, val in node_attrs.items()
                if (node_attr := bound.arguments[key]) is not None
            }

        # It should be safe to assume that we either have networkx graphs or backend graphs.
        # Future work: allow conversions between backends.
        for gname in self.graphs:
            if gname in self.list_graphs:
                bound.arguments[gname] = [
                    self._convert_graph(
                        backend_name,
                        g,
                        edge_attrs=edge_attrs,
                        node_attrs=node_attrs,
                        preserve_edge_attrs=preserve_edge_attrs,
                        preserve_node_attrs=preserve_node_attrs,
                        preserve_graph_attrs=preserve_graph_attrs,
                        graph_name=gname,
                        use_cache=use_cache,
                        mutations=mutations,
                    )
                    if getattr(g, "__networkx_backend__", "networkx") != backend_name
                    else g
                    for g in bound.arguments[gname]
                ]
            else:
                graph = bound.arguments[gname]
                if graph is None:
                    if gname in self.optional_graphs:
                        continue
                    raise TypeError(
                        f"Missing required graph argument `{gname}` in {self.name} function"
                    )
                if isinstance(preserve_edge_attrs, dict):
                    preserve_edges = False
                    edges = preserve_edge_attrs.get(gname, edge_attrs)
                else:
                    preserve_edges = preserve_edge_attrs
                    edges = edge_attrs
                if isinstance(preserve_node_attrs, dict):
                    preserve_nodes = False
                    nodes = preserve_node_attrs.get(gname, node_attrs)
                else:
                    preserve_nodes = preserve_node_attrs
                    nodes = node_attrs
                if isinstance(preserve_graph_attrs, set):
                    preserve_graph = gname in preserve_graph_attrs
                else:
                    preserve_graph = preserve_graph_attrs
                if getattr(graph, "__networkx_backend__", "networkx") != backend_name:
                    bound.arguments[gname] = self._convert_graph(
                        backend_name,
                        graph,
                        edge_attrs=edges,
                        node_attrs=nodes,
                        preserve_edge_attrs=preserve_edges,
                        preserve_node_attrs=preserve_nodes,
                        preserve_graph_attrs=preserve_graph,
                        graph_name=gname,
                        use_cache=use_cache,
                        mutations=mutations,
                    )
        bound_kwargs = bound.kwargs
        del bound_kwargs["backend"]
        return bound.args, bound_kwargs

    def _convert_graph(
        self,
        backend_name,
        graph,
        *,
        edge_attrs,
        node_attrs,
        preserve_edge_attrs,
        preserve_node_attrs,
        preserve_graph_attrs,
        graph_name,
        use_cache,
        mutations,
    ):
        if (
            use_cache
            and (nx_cache := getattr(graph, "__networkx_cache__", None)) is not None
        ):
            cache = nx_cache.setdefault("backends", {}).setdefault(backend_name, {})
            key = _get_cache_key(
                edge_attrs=edge_attrs,
                node_attrs=node_attrs,
                preserve_edge_attrs=preserve_edge_attrs,
                preserve_node_attrs=preserve_node_attrs,
                preserve_graph_attrs=preserve_graph_attrs,
            )
            compat_key, rv = _get_from_cache(cache, key, mutations=mutations)
            if rv is not None:
                if "cache" not in nx.config.warnings_to_ignore:
                    warnings.warn(
                        "Note: conversions to backend graphs are saved to cache "
                        "(`G.__networkx_cache__` on the original graph) by default."
                        "\n\nThis warning means the cached graph is being used "
                        f"for the {backend_name!r} backend in the "
                        f"call to {self.name}.\n\nFor the cache to be consistent "
                        "(i.e., correct), the input graph must not have been "
                        "manually mutated since the cached graph was created. "
                        "Examples of manually mutating the graph data structures "
                        "resulting in an inconsistent cache include:\n\n"
                        "    >>> G[u][v][key] = val\n\n"
                        "and\n\n"
                        "    >>> for u, v, d in G.edges(data=True):\n"
                        "    ...     d[key] = val\n\n"
                        "Using methods such as `G.add_edge(u, v, weight=val)` "
                        "will correctly clear the cache to keep it consistent. "
                        "You may also use `G.__networkx_cache__.clear()` to "
                        "manually clear the cache, or set `G.__networkx_cache__` "
                        "to None to disable caching for G. Enable or disable caching "
                        "globally via `nx.config.cache_converted_graphs` config.\n\n"
                        "To disable this warning:\n\n"
                        '    >>> nx.config.warnings_to_ignore.add("cache")\n'
                    )
                _logger.debug(
                    "Using cached converted graph (from '%s' to '%s' backend) "
                    "in call to `%s' for '%s' argument",
                    getattr(graph, "__networkx_backend__", None),
                    backend_name,
                    self.name,
                    graph_name,
                )
                return rv

        if backend_name == "networkx":
            # Perhaps we should check that "__networkx_backend__" attribute exists
            # and return the original object if not.
            if not hasattr(graph, "__networkx_backend__"):
                _logger.debug(
                    "Unable to convert input to 'networkx' backend in call to `%s' for "
                    "'%s argument, because it is not from a backend (i.e., it does not "
                    "have `G.__networkx_backend__` attribute). Using the original "
                    "object: %s",
                    self.name,
                    graph_name,
                    graph,
                )
                # This may fail, but let it fail in the networkx function
                return graph
            backend = _load_backend(graph.__networkx_backend__)
            rv = backend.convert_to_nx(graph)
        else:
            backend = _load_backend(backend_name)
            rv = backend.convert_from_nx(
                graph,
                edge_attrs=edge_attrs,
                node_attrs=node_attrs,
                preserve_edge_attrs=preserve_edge_attrs,
                preserve_node_attrs=preserve_node_attrs,
                # Always preserve graph attrs when we are caching b/c this should be
                # cheap and may help prevent extra (unnecessary) conversions. Because
                # we do this, we don't need `preserve_graph_attrs` in the cache key.
                preserve_graph_attrs=preserve_graph_attrs or use_cache,
                name=self.name,
                graph_name=graph_name,
            )
        if use_cache and nx_cache is not None and mutations is None:
            _set_to_cache(cache, key, rv)
            _logger.debug(
                "Caching converted graph (from '%s' to '%s' backend) "
                "in call to `%s' for '%s' argument",
                getattr(graph, "__networkx_backend__", None),
                backend_name,
                self.name,
                graph_name,
            )

        return rv

    def _call_with_backend(self, backend_name, args, kwargs, *, extra_message=None):
        """Call this dispatchable function with a backend without converting inputs."""
        if backend_name == "networkx":
            return self.orig_func(*args, **kwargs)
        backend = _load_backend(backend_name)
        _logger.debug(
            "Using backend '%s' for call to `%s' with arguments: %s",
            backend_name,
            self.name,
            _LazyArgsRepr(self, args, kwargs),
        )
        try:
            return getattr(backend, self.name)(*args, **kwargs)
        except NotImplementedError as exc:
            if extra_message is not None:
                _logger.debug(
                    "Backend '%s' raised when calling `%s': %s",
                    backend_name,
                    self.name,
                    exc,
                )
                raise NotImplementedError(extra_message) from exc
            raise

    def _convert_and_call(
        self,
        backend_name,
        input_backend_names,
        args,
        kwargs,
        *,
        extra_message=None,
        mutations=None,
    ):
        """Call this dispatchable function with a backend after converting inputs.

        Parameters
        ----------
        backend_name : str
        input_backend_names : set[str]
        args : arguments tuple
        kwargs : keywords dict
        extra_message : str, optional
            Additional message to log if NotImplementedError is raised by backend.
        mutations : list, optional
            Used to clear objects gotten from cache if inputs will be mutated.
        """
        if backend_name == "networkx":
            func = self.orig_func
        else:
            backend = _load_backend(backend_name)
            func = getattr(backend, self.name)
        other_backend_names = input_backend_names - {backend_name}
        _logger.debug(
            "Converting input graphs from %s backend%s to '%s' backend for call to `%s'",
            other_backend_names
            if len(other_backend_names) > 1
            else f"'{next(iter(other_backend_names))}'",
            "s" if len(other_backend_names) > 1 else "",
            backend_name,
            self.name,
        )
        try:
            converted_args, converted_kwargs = self._convert_arguments(
                backend_name,
                args,
                kwargs,
                use_cache=nx.config.cache_converted_graphs,
                mutations=mutations,
            )
        except NotImplementedError as exc:
            # Only log the exception if we are adding an extra message
            # because we don't want to lose any information.
            _logger.debug(
