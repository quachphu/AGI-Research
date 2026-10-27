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

        >>> G = nx.Graph([(0, 1), (1, 2), (3, 4)])
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
                self._adj[n] = self.adjlist_inner_dict_factory()
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
        adj = self._adj
        try:
            nbrs = list(adj[n])  # list handles self-loops (allows mutation)
            del self._node[n]
        except KeyError as err:  # NetworkXError if n not in self
            raise NetworkXError(f"The node {n} is not in the graph.") from err
        for u in nbrs:
            del adj[u][n]  # remove all edges n-u in graph
        del adj[n]  # now remove node
        nx._clear_cache(self)

    def remove_nodes_from(self, nodes):
        """Remove multiple nodes.

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, set, etc.).  If a node
            in the container is not in the graph it is silently
            ignored.

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

        >>> G = nx.Graph([(0, 1), (1, 2), (3, 4)])
        >>> # this command will fail, as the graph's dict is modified during iteration
        >>> # G.remove_nodes_from(n for n in G.nodes if n < 2)
        >>> # this command will work, since the dictionary underlying graph is not modified
        >>> G.remove_nodes_from(list(n for n in G.nodes if n < 2))
        """
        adj = self._adj
        for n in nodes:
            try:
                del self._node[n]
                for u in list(adj[n]):  # list handles self-loops
                    del adj[u][n]  # (allows mutation of dict in loop)
                del adj[n]
            except KeyError:
                pass
        nx._clear_cache(self)

    @cached_property
    def nodes(self):
        """A NodeView of the Graph as G.nodes or G.nodes().

        Can be used as `G.nodes` for data lookup and for set-like operations.
        Can also be used as `G.nodes(data='color', default=None)` to return a
        NodeDataView which reports specific node data but no set operations.
        It presents a dict-like interface as well with `G.nodes.items()`
        iterating over `(node, nodedata)` 2-tuples and `G.nodes[3]['foo']`
        providing the value of the `foo` attribute for node `3`. In addition,
        a view `G.nodes.data('foo')` provides a dict-like interface to the
        `foo` attribute of each node. `G.nodes.data('foo', default=1)`
        provides a default for nodes that do not have attribute `foo`.

        Parameters
        ----------
        data : string or bool, optional (default=False)
            The node attribute returned in 2-tuple (n, ddict[data]).
            If True, return entire node attribute dict as (n, ddict).
            If False, return just the nodes n.

        default : value, optional (default=None)
            Value used for nodes that don't have the requested attribute.
            Only relevant if data is not True or False.

        Returns
        -------
        NodeView
            Allows set-like operations over the nodes as well as node
            attribute dict lookup and calling to get a NodeDataView.
            A NodeDataView iterates over `(n, data)` and has no set operations.
            A NodeView iterates over `n` and includes set operations.

            When called, if data is False, an iterator over nodes.
            Otherwise an iterator of 2-tuples (node, attribute value)
            where the attribute is specified in `data`.
            If data is True then the attribute becomes the
            entire data dictionary.

        Notes
        -----
        If your node data is not needed, it is simpler and equivalent
        to use the expression ``for n in G``, or ``list(G)``.

        Examples
        --------
        There are two simple ways of getting a list of all nodes in the graph:

        >>> G = nx.path_graph(3)
        >>> list(G.nodes)
        [0, 1, 2]
        >>> list(G)
        [0, 1, 2]

        To get the node data along with the nodes:

        >>> G.add_node(1, time="5pm")
        >>> G.nodes[0]["foo"] = "bar"
        >>> list(G.nodes(data=True))
        [(0, {'foo': 'bar'}), (1, {'time': '5pm'}), (2, {})]
        >>> list(G.nodes.data())
        [(0, {'foo': 'bar'}), (1, {'time': '5pm'}), (2, {})]

        >>> list(G.nodes(data="foo"))
        [(0, 'bar'), (1, None), (2, None)]
        >>> list(G.nodes.data("foo"))
        [(0, 'bar'), (1, None), (2, None)]

        >>> list(G.nodes(data="time"))
        [(0, None), (1, '5pm'), (2, None)]
        >>> list(G.nodes.data("time"))
        [(0, None), (1, '5pm'), (2, None)]

        >>> list(G.nodes(data="time", default="Not Available"))
        [(0, 'Not Available'), (1, '5pm'), (2, 'Not Available')]
        >>> list(G.nodes.data("time", default="Not Available"))
        [(0, 'Not Available'), (1, '5pm'), (2, 'Not Available')]

        If some of your nodes have an attribute and the rest are assumed
        to have a default attribute value you can create a dictionary
        from node/attribute pairs using the `default` keyword argument
        to guarantee the value is never None::

            >>> G = nx.Graph()
            >>> G.add_node(0)
            >>> G.add_node(1, weight=2)
            >>> G.add_node(2, weight=3)
            >>> dict(G.nodes(data="weight", default=1))
            {0: 1, 1: 2, 2: 3}

        """
        return NodeView(self)

    def number_of_nodes(self):
        """Returns the number of nodes in the graph.

        Returns
        -------
        nnodes : int
            The number of nodes in the graph.

        See Also
        --------
        order: identical method
        __len__: identical method

        Examples
        --------
        >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.number_of_nodes()
        3
        """
        return len(self._node)

    def order(self):
        """Returns the number of nodes in the graph.

        Returns
        -------
        nnodes : int
            The number of nodes in the graph.

        See Also
        --------
        number_of_nodes: identical method
        __len__: identical method

        Examples
        --------
        >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.order()
        3
        """
        return len(self._node)

    def has_node(self, n):
        """Returns True if the graph contains the node n.

        Identical to `n in G`

        Parameters
        ----------
        n : node

        Examples
        --------
        >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.has_node(0)
        True

        It is more readable and simpler to use

        >>> 0 in G
        True

        """
        try:
            return n in self._node
        except TypeError:
            return False

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
        if u not in self._node:
            if u is None:
                raise ValueError("None cannot be a node")
            self._adj[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._node:
            if v is None:
                raise ValueError("None cannot be a node")
            self._adj[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        # add the edge
        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
        datadict.update(attr)
        self._adj[u][v] = datadict
        self._adj[v][u] = datadict
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

        >>> G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        >>> # Grow graph by one new node, adding edges to all existing nodes.
        >>> # wrong way - will raise RuntimeError
        >>> # G.add_edges_from(((5, n) for n in G.nodes))
        >>> # correct way - note that there will be no self-edge for node 5
        >>> G.add_edges_from(list((5, n) for n in G.nodes))
        """
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}  # doesn't need edge_attr_dict_factory
            else:
                raise NetworkXError(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
            if u not in self._node:
                if u is None:
                    raise ValueError("None cannot be a node")
                self._adj[u] = self.adjlist_inner_dict_factory()
                self._node[u] = self.node_attr_dict_factory()
            if v not in self._node:
                if v is None:
                    raise ValueError("None cannot be a node")
                self._adj[v] = self.adjlist_inner_dict_factory()
                self._node[v] = self.node_attr_dict_factory()
            datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
            datadict.update(attr)
            datadict.update(dd)
            self._adj[u][v] = datadict
            self._adj[v][u] = datadict
        nx._clear_cache(self)

    def add_weighted_edges_from(self, ebunch_to_add, weight="weight", **attr):
        """Add weighted edges in `ebunch_to_add` with specified weight attr

        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the list or container will be added
            to the graph. The edges must be given as 3-tuples (u, v, w)
            where w is a number.
        weight : string, optional (default= 'weight')
            The attribute name for the edge weights to be added.
        attr : keyword arguments, optional (default= no attributes)
            Edge attributes to add/update for all edges.

        See Also
        --------
        add_edge : add a single edge
        add_edges_from : add multiple edges

        Notes
        -----
        Adding the same edge twice for Graph/DiGraph simply updates
        the edge data. For MultiGraph/MultiDiGraph, duplicate edges
        are stored.

        When adding edges from an iterator over the graph you are changing,
        a `RuntimeError` can be raised with message:
        `RuntimeError: dictionary changed size during iteration`. This
        happens when the graph's underlying dictionary is modified during
        iteration. To avoid this error, evaluate the iterator into a separate
        object, e.g. by using `list(iterator_of_edges)`, and pass this
        object to `G.add_weighted_edges_from`.

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])

        Evaluate an iterator over edges before passing it

        >>> G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        >>> weight = 0.1
        >>> # Grow graph by one new node, adding edges to all existing nodes.
        >>> # wrong way - will raise RuntimeError
        >>> # G.add_weighted_edges_from(((5, n, weight) for n in G.nodes))
        >>> # correct way - note that there will be no self-edge for node 5
        >>> G.add_weighted_edges_from(list((5, n, weight) for n in G.nodes))
        """
        self.add_edges_from(((u, v, {weight: d}) for u, v, d in ebunch_to_add), **attr)
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
        >>> G = nx.path_graph(4)  # or DiGraph, etc
        >>> G.remove_edge(0, 1)
        >>> e = (1, 2)
        >>> G.remove_edge(*e)  # unpacks e from an edge tuple
        >>> e = (2, 3, {"weight": 7})  # an edge with attribute data
        >>> G.remove_edge(*e[:2])  # select first part of edge tuple
        """
        try:
            del self._adj[u][v]
            if u != v:  # self-loop needs only one entry removed
                del self._adj[v][u]
        except KeyError as err:
            raise NetworkXError(f"The edge {u}-{v} is not in the graph") from err
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
        adj = self._adj
        for e in ebunch:
            u, v = e[:2]  # ignore edge data if present
            if u in adj and v in adj[u]:
                del adj[u][v]
                if u != v:  # self loop needs only one entry removed
                    del adj[v][u]
        nx._clear_cache(self)

    def update(self, edges=None, nodes=None):
        """Update the graph using nodes/edges/graphs as input.

        Like dict.update, this method takes a graph as input, adding the
        graph's nodes and edges to this graph. It can also take two inputs:
        edges and nodes. Finally it can take either edges or nodes.
        To specify only nodes the keyword `nodes` must be used.

        The collections of edges and nodes are treated similarly to
        the add_edges_from/add_nodes_from methods. When iterated, they
        should yield 2-tuples (u, v) or 3-tuples (u, v, datadict).

        Parameters
        ----------
        edges : Graph object, collection of edges, or None
            The first parameter can be a graph or some edges. If it has
            attributes `nodes` and `edges`, then it is taken to be a
            Graph-like object and those attributes are used as collections
            of nodes and edges to be added to the graph.
            If the first parameter does not have those attributes, it is
            treated as a collection of edges and added to the graph.
            If the first argument is None, no edges are added.
        nodes : collection of nodes, or None
            The second parameter is treated as a collection of nodes
            to be added to the graph unless it is None.
            If `edges is None` and `nodes is None` an exception is raised.
            If the first parameter is a Graph, then `nodes` is ignored.

        Examples
        --------
        >>> G = nx.path_graph(5)
        >>> G.update(nx.complete_graph(range(4, 10)))
        >>> from itertools import combinations
        >>> edges = (
        ...     (u, v, {"power": u * v})
        ...     for u, v in combinations(range(10, 20), 2)
        ...     if u * v < 225
        ... )
        >>> nodes = [1000]  # for singleton, use a container
        >>> G.update(edges, nodes)

        Notes
        -----
        It you want to update the graph using an adjacency structure
        it is straightforward to obtain the edges/nodes from adjacency.
        The following examples provide common cases, your adjacency may
        be slightly different and require tweaks of these examples::

        >>> # dict-of-set/list/tuple
        >>> adj = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
        >>> e = [(u, v) for u, nbrs in adj.items() for v in nbrs]
        >>> G.update(edges=e, nodes=adj)

        >>> DG = nx.DiGraph()
        >>> # dict-of-dict-of-attribute
        >>> adj = {1: {2: 1.3, 3: 0.7}, 2: {1: 1.4}, 3: {1: 0.7}}
        >>> e = [
        ...     (u, v, {"weight": d})
        ...     for u, nbrs in adj.items()
        ...     for v, d in nbrs.items()
        ... ]
        >>> DG.update(edges=e, nodes=adj)

        >>> # dict-of-dict-of-dict
        >>> adj = {1: {2: {"weight": 1.3}, 3: {"color": 0.7, "weight": 1.2}}}
        >>> e = [
        ...     (u, v, {"weight": d})
        ...     for u, nbrs in adj.items()
        ...     for v, d in nbrs.items()
        ... ]
        >>> DG.update(edges=e, nodes=adj)

        >>> # predecessor adjacency (dict-of-set)
        >>> pred = {1: {2, 3}, 2: {3}, 3: {3}}
        >>> e = [(v, u) for u, nbrs in pred.items() for v in nbrs]

        >>> # MultiGraph dict-of-dict-of-dict-of-attribute
        >>> MDG = nx.MultiDiGraph()
        >>> adj = {
        ...     1: {2: {0: {"weight": 1.3}, 1: {"weight": 1.2}}},
        ...     3: {2: {0: {"weight": 0.7}}},
        ... }
        >>> e = [
        ...     (u, v, ekey, d)
        ...     for u, nbrs in adj.items()
        ...     for v, keydict in nbrs.items()
        ...     for ekey, d in keydict.items()
        ... ]
        >>> MDG.update(edges=e)

        See Also
        --------
        add_edges_from: add multiple edges to a graph
        add_nodes_from: add multiple nodes to a graph
        """
        if edges is not None:
            if nodes is not None:
                self.add_nodes_from(nodes)
                self.add_edges_from(edges)
            else:
                # check if edges is a Graph object
                try:
                    graph_nodes = edges.nodes
                    graph_edges = edges.edges
                except AttributeError:
                    # edge not Graph-like
                    self.add_edges_from(edges)
                else:  # edges is Graph-like
                    self.add_nodes_from(graph_nodes.data())
                    self.add_edges_from(graph_edges.data())
                    self.graph.update(edges.graph)
        elif nodes is not None:
            self.add_nodes_from(nodes)
        else:
            raise NetworkXError("update needs nodes or edges input")

    def has_edge(self, u, v):
        """Returns True if the edge (u, v) is in the graph.

        This is the same as `v in G[u]` without KeyError exceptions.

        Parameters
        ----------
        u, v : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        edge_ind : bool
            True if edge is in the graph, False otherwise.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.has_edge(0, 1)  # using two nodes
        True
        >>> e = (0, 1)
        >>> G.has_edge(*e)  #  e is a 2-tuple (u, v)
        True
        >>> e = (0, 1, {"weight": 7})
        >>> G.has_edge(*e[:2])  # e is a 3-tuple (u, v, data_dictionary)
        True

        The following syntax are equivalent:

        >>> G.has_edge(0, 1)
        True
        >>> 1 in G[0]  # though this gives KeyError if 0 not in G
        True

        """
        try:
            return v in self._adj[u]
        except KeyError:
            return False

    def neighbors(self, n):
        """Returns an iterator over all neighbors of node n.

        This is identical to `iter(G[n])`

        Parameters
        ----------
        n : node
           A node in the graph

        Returns
        -------
        neighbors : iterator
            An iterator over all neighbors of node n

        Raises
        ------
        NetworkXError
            If the node n is not in the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> [n for n in G.neighbors(0)]
        [1]

        Notes
        -----
        Alternate ways to access the neighbors are ``G.adj[n]`` or ``G[n]``:

        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edge("a", "b", weight=7)
        >>> G["a"]
        AtlasView({'b': {'weight': 7}})
        >>> G = nx.path_graph(4)
        >>> [n for n in G[0]]
        [1]
        """
        try:
            return iter(self._adj[n])
        except KeyError as err:
            raise NetworkXError(f"The node {n} is not in the graph.") from err

    @cached_property
    def edges(self):
        """An EdgeView of the Graph as G.edges or G.edges().

        edges(self, nbunch=None, data=False, default=None)

        The EdgeView provides set-like operations on the edge-tuples
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
        edges : EdgeView
            A view of edge attributes, usually it iterates over (u, v)
            or (u, v, d) tuples of edges, but can also be used for
            attribute lookup as `edges[u, v]['foo']`.

        Notes
        -----
        Nodes in nbunch that are not in the graph will be (quietly) ignored.
        For directed graphs this returns the out-edges.

        Examples
        --------
        >>> G = nx.path_graph(3)  # or MultiGraph, etc
        >>> G.add_edge(2, 3, weight=5)
        >>> [e for e in G.edges]
        [(0, 1), (1, 2), (2, 3)]
        >>> G.edges.data()  # default data is {} (empty dict)
        EdgeDataView([(0, 1, {}), (1, 2, {}), (2, 3, {'weight': 5})])
        >>> G.edges.data("weight", default=1)
        EdgeDataView([(0, 1, 1), (1, 2, 1), (2, 3, 5)])
        >>> G.edges([0, 3])  # only edges from these nodes
        EdgeDataView([(0, 1), (3, 2)])
        >>> G.edges(0)  # only edges from node 0
        EdgeDataView([(0, 1)])
        """
        return EdgeView(self)

    def get_edge_data(self, u, v, default=None):
        """Returns the attribute dictionary associated with edge (u, v).

        This is identical to `G[u][v]` except the default is returned
        instead of an exception if the edge doesn't exist.

        Parameters
        ----------
        u, v : nodes
        default:  any Python object (default=None)
            Value to return if the edge (u, v) is not found.

        Returns
        -------
        edge_dict : dictionary
            The edge attribute dictionary.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G[0][1]
        {}

        Warning: Assigning to `G[u][v]` is not permitted.
        But it is safe to assign attributes `G[u][v]['foo']`

        >>> G[0][1]["weight"] = 7
        >>> G[0][1]["weight"]
        7
        >>> G[1][0]["weight"]
        7

        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.get_edge_data(0, 1)  # default edge data is {}
        {}
        >>> e = (0, 1)
        >>> G.get_edge_data(*e)  # tuple form
        {}
        >>> G.get_edge_data("a", "b", default=0)  # edge not in graph, return 0
        0
        """
        try:
            return self._adj[u][v]
        except KeyError:
            return default

    def adjacency(self):
        """Returns an iterator over (node, adjacency dict) tuples for all nodes.

        For directed graphs, only outgoing neighbors/adjacencies are included.

        Returns
        -------
        adj_iter : iterator
           An iterator over (node, adjacency dictionary) for all nodes in
           the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> [(n, nbrdict) for n, nbrdict in G.adjacency()]
        [(0, {1: {}}), (1, {0: {}, 2: {}}), (2, {1: {}, 3: {}}), (3, {2: {}})]

        """
        return iter(self._adj.items())

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
        DegreeView or int
            If multiple nodes are requested (the default), returns a `DegreeView`
            mapping nodes to their degree.
            If a single node is requested, returns the degree of the node as an integer.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.degree[0]  # node 0 has degree 1
        1
        >>> list(G.degree([0, 1, 2]))
        [(0, 1), (1, 2), (2, 2)]
        """
        return DegreeView(self)

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
        self._adj.clear()
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
        for nbr_dict in self._adj.values():
            nbr_dict.clear()
        nx._clear_cache(self)

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return False

    def is_directed(self):
        """Returns True if graph is directed, False otherwise."""
        return False

    def copy(self, as_view=False):
        """Returns a copy of the graph.

        The copy method by default returns an independent shallow copy
        of the graph and attributes. That is, if an attribute is a
        container, that container is shared by the original an the copy.
        Use Python's `copy.deepcopy` for new containers.

        If `as_view` is True then a view is returned instead of a copy.

        Notes
        -----
        All copies reproduce the graph structure, but data attributes
        may be handled in different ways. There are four types of copies
        of a graph that people might want.

        Deepcopy -- A "deepcopy" copies the graph structure as well as
        all data attributes and any objects they might contain.
        The entire graph object is new so that changes in the copy
        do not affect the original object. (see Python's copy.deepcopy)

        Data Reference (Shallow) -- For a shallow copy the graph structure
        is copied but the edge, node and graph attribute dicts are
        references to those in the original graph. This saves
        time and memory but could cause confusion if you change an attribute
        in one graph and it changes the attribute in the other.
        NetworkX does not provide this level of shallow copy.

        Independent Shallow -- This copy creates new independent attribute
        dicts and then does a shallow copy of the attributes. That is, any
        attributes that are containers are shared between the new graph
        and the original. This is exactly what `dict.copy()` provides.
        You can obtain this style copy using:

            >>> G = nx.path_graph(5)
            >>> H = G.copy()
            >>> H = G.copy(as_view=False)
            >>> H = nx.Graph(G)
            >>> H = G.__class__(G)

        Fresh Data -- For fresh data, the graph structure is copied while
        new empty data attribute dicts are created. The resulting graph
        is independent of the original and it has no edge, node or graph
        attributes. Fresh copies are not enabled. Instead use:

            >>> H = G.__class__()
            >>> H.add_nodes_from(G)
            >>> H.add_edges_from(G.edges)

        View -- Inspired by dict-views, graph-views act like read-only
        versions of the original graph, providing a copy of the original
        structure without requiring any memory for copying the information.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Parameters
        ----------
        as_view : bool, optional (default=False)
            If True, the returned graph-view provides a read-only view
            of the original graph without actually copying any data.

        Returns
        -------
        G : Graph
            A copy of the graph.

        See Also
        --------
        to_directed: return a directed copy of the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> H = G.copy()

        """
        if as_view is True:
            return nx.graphviews.generic_graph_view(self)
        G = self.__class__()
        G.graph.update(self.graph)
        G.add_nodes_from((n, d.copy()) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, datadict.copy())
            for u, nbrs in self._adj.items()
            for v, datadict in nbrs.items()
        )
        return G

    def to_directed(self, as_view=False):
        """Returns a directed representation of the graph.

        Returns
        -------
        G : DiGraph
            A directed graph with the same name, same nodes, and with
            each edge (u, v, data) replaced by two directed edges
            (u, v, data) and (v, u, data).

        Notes
        -----
        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.

        This is in contrast to the similar D=DiGraph(G) which returns a
        shallow copy of the data.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Warning: If you have subclassed Graph to use dict-like objects
        in the data structure, those changes do not transfer to the
        DiGraph created by this method.

        Examples
        --------
        >>> G = nx.Graph()  # or MultiGraph, etc
        >>> G.add_edge(0, 1)
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1), (1, 0)]

        If already directed, return a (deep) copy

        >>> G = nx.DiGraph()  # or MultiDiGraph, etc
        >>> G.add_edge(0, 1)
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1)]
        """
        graph_class = self.to_directed_class()
        if as_view is True:
            return nx.graphviews.generic_graph_view(self, graph_class)
        # deepcopy when not a view
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, deepcopy(data))
            for u, nbrs in self._adj.items()
            for v, data in nbrs.items()
        )
        return G

    def to_undirected(self, as_view=False):
        """Returns an undirected copy of the graph.

        Parameters
        ----------
        as_view : bool (optional, default=False)
          If True return a view of the original undirected graph.

        Returns
        -------
        G : Graph/MultiGraph
            A deepcopy of the graph.

        See Also
        --------
        Graph, copy, add_edge, add_edges_from

        Notes
        -----
        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.

        This is in contrast to the similar `G = nx.DiGraph(D)` which returns a
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
        G.add_edges_from(
            (u, v, deepcopy(d))
            for u, nbrs in self._adj.items()
            for v, d in nbrs.items()
        )
        return G

    def subgraph(self, nodes):
        """Returns a SubGraph view of the subgraph induced on `nodes`.

        The induced subgraph of the graph contains the nodes in `nodes`
        and the edges between those nodes.

        Parameters
        ----------
        nodes : list, iterable
            A container of nodes which will be iterated through once.

        Returns
        -------
        G : SubGraph View
            A subgraph view of the graph. The graph structure cannot be
            changed but node/edge attributes can and are shared with the
            original graph.

        Notes
        -----
        The graph, edge and node attributes are shared with the original graph.
        Changes to the graph structure is ruled out by the view, but changes
        to attributes are reflected in the original graph.

        To create a subgraph with its own copy of the edge/node attributes use:
        G.subgraph(nodes).copy()

        For an inplace reduction of a graph to a subgraph you can remove nodes:
        G.remove_nodes_from([n for n in G if n not in set(nodes)])

        Subgraph views are sometimes NOT what you want. In most cases where
        you want to do more than simply look at the induced edges, it makes
        more sense to just create the subgraph as its own graph with code like:

        ::

            # Create a subgraph SG based on a (possibly multigraph) G
            SG = G.__class__()
            SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
            if SG.is_multigraph():
                SG.add_edges_from(
                    (n, nbr, key, d)
                    for n, nbrs in G.adj.items()
                    if n in largest_wcc
                    for nbr, keydict in nbrs.items()
                    if nbr in largest_wcc
                    for key, d in keydict.items()
                )
            else:
                SG.add_edges_from(
                    (n, nbr, d)
                    for n, nbrs in G.adj.items()
                    if n in largest_wcc
                    for nbr, d in nbrs.items()
                    if nbr in largest_wcc
                )
            SG.graph.update(G.graph)

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> H = G.subgraph([0, 1, 2])
        >>> list(H.edges)
        [(0, 1), (1, 2)]
        """
        induced_nodes = nx.filters.show_nodes(self.nbunch_iter(nodes))
        # if already a subgraph, don't make a chain
        subgraph = nx.subgraph_view
        if hasattr(self, "_NODE_OK"):
            return subgraph(
                self._graph, filter_node=induced_nodes, filter_edge=self._EDGE_OK
            )
        return subgraph(self, filter_node=induced_nodes)

    def edge_subgraph(self, edges):
        """Returns the subgraph induced by the specified edges.

        The induced subgraph contains each edge in `edges` and each
        node incident to any one of those edges.

        Parameters
        ----------
        edges : iterable
            An iterable of edges in this graph.

        Returns
        -------
        G : Graph
            An edge-induced subgraph of this graph with the same edge
            attributes.

        Notes
        -----
        The graph, edge, and node attributes in the returned subgraph
        view are references to the corresponding attributes in the original
        graph. The view is read-only.

        To create a full graph version of the subgraph with its own copy
        of the edge or node attributes, use::

            G.edge_subgraph(edges).copy()

        Examples
        --------
        >>> G = nx.path_graph(5)
        >>> H = G.edge_subgraph([(0, 1), (3, 4)])
        >>> list(H.nodes)
        [0, 1, 3, 4]
        >>> list(H.edges)
        [(0, 1), (3, 4)]

        """
        return nx.edge_subgraph(self, edges)

    def size(self, weight=None):
        """Returns the number of edges or total of all edge weights.

        Parameters
        ----------
        weight : string or None, optional (default=None)
            The edge attribute that holds the numerical value used
            as a weight. If None, then each edge has weight 1.

        Returns
        -------
        size : numeric
            The number of edges or
            (if weight keyword is provided) the total weight sum.

            If weight is None, returns an int. Otherwise a float
            (or more general numeric if the weights are more general).

        See Also
        --------
        number_of_edges

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.size()
        3

        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edge("a", "b", weight=2)
        >>> G.add_edge("b", "c", weight=4)
        >>> G.size()
        2
        >>> G.size(weight="weight")
        6.0
        """
        s = sum(d for v, d in self.degree(weight=weight))
        # If `weight` is None, the sum of the degrees is guaranteed to be
        # even, so we can perform integer division and hence return an
        # integer. Otherwise, the sum of the weighted degrees is not
        # guaranteed to be an integer, so we perform "real" division.
        return s // 2 if weight is None else s / 2

    def number_of_edges(self, u=None, v=None):
        """Returns the number of edges between two nodes.

        Parameters
        ----------
        u, v : nodes, optional (default=all edges)
            If u and v are specified, return the number of edges between
            u and v. Otherwise return the total number of all edges.

        Returns
        -------
        nedges : int
            The number of edges in the graph.  If nodes `u` and `v` are
            specified return the number of edges between those nodes. If
            the graph is directed, this only returns the number of edges
            from `u` to `v`.

        See Also
        --------
        size

        Examples
        --------
        For undirected graphs, this method counts the total number of
        edges in the graph:

        >>> G = nx.path_graph(4)
        >>> G.number_of_edges()
        3

        If you specify two nodes, this counts the total number of edges
        joining the two nodes:

        >>> G.number_of_edges(0, 1)
        1

        For directed graphs, this method can count the total number of
        directed edges from `u` to `v`:

        >>> G = nx.DiGraph()
        >>> G.add_edge(0, 1)
        >>> G.add_edge(1, 0)
        >>> G.number_of_edges(0, 1)
        1

        """
        if u is None:
            return int(self.size())
        if v in self._adj[u]:
            return 1
        return 0

    def nbunch_iter(self, nbunch=None):
        """Returns an iterator over nodes contained in nbunch that are
        also in the graph.

        The nodes in nbunch are checked for membership in the graph
        and if not are silently ignored.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.

        Returns
        -------
        niter : iterator
            An iterator over nodes in nbunch that are also in the graph.
            If nbunch is None, iterate over all nodes in the graph.

        Raises
        ------
        NetworkXError
            If nbunch is not a node or sequence of nodes.
            If a node in nbunch is not hashable.

        See Also
        --------
        Graph.__iter__

        Notes
        -----
        When nbunch is an iterator, the returned iterator yields values
        directly from nbunch, becoming exhausted when nbunch is exhausted.

        To test whether nbunch is a single node, one can use
        "if nbunch in self:", even after processing with this routine.

        If nbunch is not a node or a (possibly empty) sequence/iterator
        or None, a :exc:`NetworkXError` is raised.  Also, if any object in
        nbunch is not hashable, a :exc:`NetworkXError` is raised.
        """
        if nbunch is None:  # include all nodes via iterator
            bunch = iter(self._adj)
        elif nbunch in self:  # if nbunch is a single node
            bunch = iter([nbunch])
        else:  # if nbunch is a sequence of nodes

            def bunch_iter(nlist, adj):
                try:
                    for n in nlist:
                        if n in adj:
                            yield n
                except TypeError as err:
                    exc, message = err, err.args[0]
                    # capture error for non-sequence/iterator nbunch.
                    if "iter" in message:
                        exc = NetworkXError(
                            "nbunch is not a node or a sequence of nodes."
                        )
                    # capture error for unhashable node.
                    if "hashable" in message:
                        exc = NetworkXError(
                            f"Node {n} in sequence nbunch is not a valid node."
                        )
                    raise exc

            bunch = bunch_iter(nbunch, self._adj)
        return bunch
<|endoftext|>from .graph import Graph
from .digraph import DiGraph
from .multigraph import MultiGraph
from .multidigraph import MultiDiGraph

from .function import *
from .graphviews import subgraph_view, reverse_view

from networkx.classes import filters

from networkx.classes import coreviews
from networkx.classes import graphviews
from networkx.classes import reportviews
<|endoftext|>"""Base class for MultiGraph."""

from copy import deepcopy
from functools import cached_property

import networkx as nx
from networkx import NetworkXError, convert
from networkx.classes.coreviews import MultiAdjacencyView
from networkx.classes.graph import Graph
from networkx.classes.reportviews import MultiDegreeView, MultiEdgeView

__all__ = ["MultiGraph"]


class MultiGraph(Graph):
    """
    An undirected graph class that can store multiedges.

    Multiedges are multiple edges between two nodes.  Each edge
    can hold optional data or attributes.

    A MultiGraph holds undirected edges.  Self loops are allowed.

    Nodes can be arbitrary (hashable) Python objects with optional
    key/value attributes. By convention `None` is not used as a node.

    Edges are represented as links between nodes with optional
    key/value attributes, in a MultiGraph each edge has a key to
    distinguish between multiple edges that have the same source and
    destination nodes.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array,
        SciPy sparse array, or PyGraphviz graph.

    multigraph_input : bool or None (default None)
        Note: Only used when `incoming_graph_data` is a dict.
        If True, `incoming_graph_data` is assumed to be a
        dict-of-dict-of-dict-of-dict structure keyed by
        node to neighbor to edge keys to edge data for multi-edges.
        A NetworkXError is raised if this is not the case.
        If False, :func:`to_networkx_graph` is used to try to determine
        the dict's graph data structure as either a dict-of-dict-of-dict
        keyed by node to neighbor to edge data, or a dict-of-iterable
        keyed by node to neighbors.
        If None, the treatment for True is tried, but if it fails,
        the treatment for False is tried.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    Graph
    DiGraph
    MultiDiGraph

    Examples
    --------
    Create an empty graph structure (a "null graph") with no nodes and
    no edges.

    >>> G = nx.MultiGraph()

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

    >>> key = G.add_edge(1, 2)

    a list of edges,

    >>> keys = G.add_edges_from([(1, 2), (1, 3)])

    or a collection of edges,

    >>> keys = G.add_edges_from(H.edges)

    If some edges connect nodes not yet in the graph, the nodes
    are added automatically.  If an edge already exists, an additional
    edge is created and stored using a key to identify the edge.
    By default the key is the lowest unused integer.

    >>> keys = G.add_edges_from([(4, 5, {"route": 28}), (4, 5, {"route": 37})])
    >>> G[4]
    AdjacencyView({3: {0: {}}, 5: {0: {}, 1: {'route': 28}, 2: {'route': 37}}})

    **Attributes:**

    Each graph, node, and edge can hold key/value attribute pairs
    in an associated attribute dictionary (the keys must be hashable).
    By default these are empty, but can be added or changed using
    add_edge, add_node or direct manipulation of the attribute
    dictionaries named graph, node and edge respectively.

    >>> G = nx.MultiGraph(day="Friday")
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

    >>> key = G.add_edge(1, 2, weight=4.7)
    >>> keys = G.add_edges_from([(3, 4), (4, 5)], color="red")
    >>> keys = G.add_edges_from([(1, 2, {"color": "blue"}), (2, 3, {"weight": 8})])
    >>> G[1][2][0]["weight"] = 4.7
    >>> G.edges[1, 2, 0]["weight"] = 4

    Warning: we protect the graph data structure by making `G.edges[1,
    2, 0]` a read-only dict-like structure. However, you can assign to
    attributes in e.g. `G.edges[1, 2, 0]`. Thus, use 2 sets of brackets
    to add/change data attributes: `G.edges[1, 2, 0]['weight'] = 4`.

    **Shortcuts:**

    Many common graph features allow python syntax to speed reporting.

    >>> 1 in G  # check if node in graph
    True
    >>> [n for n in G if n < 3]  # iterate through nodes
    [1, 2]
    >>> len(G)  # number of nodes in graph
    5
    >>> G[1]  # adjacency dict-like view mapping neighbor -> edge key -> edge attributes
    AdjacencyView({2: {0: {'weight': 4}, 1: {'color': 'blue'}}})

    Often the best way to traverse all edges of a graph is via the neighbors.
    The neighbors are reported as an adjacency-dict `G.adj` or `G.adjacency()`.

    >>> for n, nbrsdict in G.adjacency():
    ...     for nbr, keydict in nbrsdict.items():
    ...         for key, eattr in keydict.items():
    ...             if "weight" in eattr:
    ...                 # Do something useful with the edges
    ...                 pass

    But the edges() method is often more convenient:

    >>> for u, v, keys, weight in G.edges(data="weight", keys=True):
    ...     if weight is not None:
    ...         # Do something useful with the edges
    ...         pass

    **Reporting:**

    Simple graph information is obtained using methods and object-attributes.
    Reporting usually provides views instead of containers to reduce memory
    usage. The views update as the graph is updated similarly to dict-views.
    The objects `nodes`, `edges` and `adj` provide access to data attributes
    via lookup (e.g. `nodes[n]`, `edges[u, v, k]`, `adj[u][v]`) and iteration
    (e.g. `nodes.items()`, `nodes.data('color')`,
    `nodes.data('color', default='blue')` and similarly for `edges`)
    Views exist for `nodes`, `edges`, `neighbors()`/`adj` and `degree`.

    For details on these and other miscellaneous methods, see below.

    **Subclasses (Advanced):**

    The MultiGraph class uses a dict-of-dict-of-dict-of-dict data structure.
    The outer dict (node_dict) holds adjacency information keyed by node.
    The next dict (adjlist_dict) represents the adjacency information
    and holds edge_key dicts keyed by neighbor. The edge_key dict holds
    each edge_attr dict keyed by edge key. The inner dict
    (edge_attr_dict) represents the edge data and holds edge attribute
    values keyed by attribute names.

    Each of these four dicts in the dict-of-dict-of-dict-of-dict
    structure can be replaced by a user defined dict-like object.
    In general, the dict-like features should be maintained but
    extra features can be added. To replace one of the dicts create
    a new graph class by changing the class(!) variable holding the
    factory for that dict-like structure. The variable names are
    node_dict_factory, node_attr_dict_factory, adjlist_inner_dict_factory,
    adjlist_outer_dict_factory, edge_key_dict_factory, edge_attr_dict_factory
    and graph_attr_dict_factory.

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
        dict which holds multiedge key dicts keyed by neighbor.
        It should require no arguments and return a dict-like object.

    edge_key_dict_factory : function, (default: dict)
        Factory function to be used to create the edge key dict
        which holds edge data keyed by edge key.
        It should require no arguments and return a dict-like object.

    edge_attr_dict_factory : function, (default: dict)
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

    # node_dict_factory = dict    # already assigned in Graph
    # adjlist_outer_dict_factory = dict
    # adjlist_inner_dict_factory = dict
    edge_key_dict_factory = dict
    # edge_attr_dict_factory = dict

    def to_directed_class(self):
        """Returns the class to use for empty directed copies.

        If you subclass the base classes, use this to designate
        what directed class to use for `to_directed()` copies.
        """
        return nx.MultiDiGraph

    def to_undirected_class(self):
        """Returns the class to use for empty undirected copies.

        If you subclass the base classes, use this to designate
        what directed class to use for `to_directed()` copies.
        """
        return MultiGraph

    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        """Initialize a graph with edges, name, or graph attributes.

        Parameters
        ----------
        incoming_graph_data : input graph
            Data to initialize graph.  If incoming_graph_data=None (default)
            an empty graph is created.  The data can be an edge list, or any
            NetworkX graph object.  If the corresponding optional Python
            packages are installed the data can also be a 2D NumPy array, a
            SciPy sparse array, or a PyGraphviz graph.

        multigraph_input : bool or None (default None)
            Note: Only used when `incoming_graph_data` is a dict.
            If True, `incoming_graph_data` is assumed to be a
            dict-of-dict-of-dict-of-dict structure keyed by
            node to neighbor to edge keys to edge data for multi-edges.
            A NetworkXError is raised if this is not the case.
            If False, :func:`to_networkx_graph` is used to try to determine
            the dict's graph data structure as either a dict-of-dict-of-dict
            keyed by node to neighbor to edge data, or a dict-of-iterable
            keyed by node to neighbors.
            If None, the treatment for True is tried, but if it fails,
            the treatment for False is tried.

        attr : keyword arguments, optional (default= no attributes)
            Attributes to add to graph as key=value pairs.

        See Also
        --------
        convert

        Examples
        --------
        >>> G = nx.MultiGraph()
        >>> G = nx.MultiGraph(name="my graph")
        >>> e = [(1, 2), (1, 2), (2, 3), (3, 4)]  # list of edges
        >>> G = nx.MultiGraph(e)

        Arbitrary graph attribute pairs (key=value) may be assigned

        >>> G = nx.MultiGraph(e, day="Friday")
        >>> G.graph
        {'day': 'Friday'}

        """
        # multigraph_input can be None/True/False. So check "is not False"
        if isinstance(incoming_graph_data, dict) and multigraph_input is not False:
            Graph.__init__(self)
            try:
                convert.from_dict_of_dicts(
                    incoming_graph_data, create_using=self, multigraph_input=True
                )
                self.graph.update(attr)
            except Exception as err:
                if multigraph_input is True:
                    raise nx.NetworkXError(
                        f"converting multigraph_input raised:\n{type(err)}: {err}"
                    )
                Graph.__init__(self, incoming_graph_data, **attr)
        else:
            Graph.__init__(self, incoming_graph_data, **attr)

    @cached_property
    def adj(self):
        """Graph adjacency object holding the neighbors of each node.

        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edgekey-data-dict.  So `G.adj[3][2][0]['color'] = 'blue'` sets
        the color of the edge `(3, 2, 0)` to `"blue"`.

        Iterating over G.adj behaves like a dict. Useful idioms include
        `for nbr, edgesdict in G.adj[n].items():`.

        The neighbor information is also provided by subscripting the graph.

        Examples
        --------
        >>> e = [(1, 2), (1, 2), (1, 3), (3, 4)]  # list of edges
        >>> G = nx.MultiGraph(e)
        >>> G.edges[1, 2, 0]["weight"] = 3
        >>> result = set()
        >>> for edgekey, data in G[1][2].items():
        ...     result.add(data.get("weight", 1))
        >>> result
        {1, 3}

        For directed graphs, `G.adj` holds outgoing (successor) info.
        """
        return MultiAdjacencyView(self._adj)

    def new_edge_key(self, u, v):
        """Returns an unused key for edges between nodes `u` and `v`.

        The nodes `u` and `v` do not need to be already in the graph.

        Notes
        -----
        In the standard MultiGraph class the new key is the number of existing
        edges between `u` and `v` (increased if necessary to ensure unused).
        The first edge will have key 0, then 1, etc. If an edge is removed
        further new_edge_keys may not be in this order.

        Parameters
        ----------
        u, v : nodes

        Returns
        -------
        key : int
        """
        try:
            keydict = self._adj[u][v]
        except KeyError:
            return 0
        key = len(keydict)
        while key in keydict:
            key += 1
        return key

    def add_edge(self, u_for_edge, v_for_edge, key=None, **attr):
        """Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Edge attributes can be specified with keywords or by directly
        accessing the edge's attribute dictionary. See examples below.

        Parameters
        ----------
        u_for_edge, v_for_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        key : hashable identifier, optional (default=lowest unused integer)
            Used to distinguish multiedges between a pair of nodes.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        The edge key assigned to the edge.

        See Also
        --------
        add_edges_from : add a collection of edges

        Notes
        -----
        To replace/update edge data, use the optional key argument
        to identify a unique edge.  Otherwise a new edge will be created.

        NetworkX algorithms designed for weighted graphs cannot use
        multigraphs directly because it is not clear how to handle
        multiedge weights.  Convert to Graph using edge attribute
        'weight' to enable weighted graph algorithms.

        Default keys are generated using the method `new_edge_key()`.
        This method can be overridden by subclassing the base class and
        providing a custom `new_edge_key()` method.

        Examples
        --------
        The following each add an additional edge e=(1, 2) to graph G:

        >>> G = nx.MultiGraph()
        >>> e = (1, 2)
        >>> ekey = G.add_edge(1, 2)  # explicit two-node form
        >>> G.add_edge(*e)  # single edge as tuple of two nodes
        1
        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container
        [2]

        Associate data to edges using keywords:

        >>> ekey = G.add_edge(1, 2, weight=3)
        >>> ekey = G.add_edge(1, 2, key=0, weight=4)  # update data for key=0
        >>> ekey = G.add_edge(1, 3, weight=7, capacity=15, length=342.7)

        For non-string attribute keys, use subscript notation.

        >>> ekey = G.add_edge(1, 2)
        >>> G[1][2][0].update({0: 5})
        >>> G.edges[1, 2, 0].update({0: 5})
        """
        u, v = u_for_edge, v_for_edge
        # add nodes
        if u not in self._adj:
            if u is None:
                raise ValueError("None cannot be a node")
            self._adj[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._adj:
            if v is None:
                raise ValueError("None cannot be a node")
            self._adj[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        if key is None:
            key = self.new_edge_key(u, v)
        if v in self._adj[u]:
            keydict = self._adj[u][v]
            datadict = keydict.get(key, self.edge_attr_dict_factory())
            datadict.update(attr)
            keydict[key] = datadict
        else:
            # selfloops work this way without special treatment
            datadict = self.edge_attr_dict_factory()
            datadict.update(attr)
            keydict = self.edge_key_dict_factory()
            keydict[key] = datadict
            self._adj[u][v] = keydict
            self._adj[v][u] = keydict
        nx._clear_cache(self)
        return key

    def add_edges_from(self, ebunch_to_add, **attr):
        """Add all the edges in ebunch_to_add.

        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the container will be added to the
            graph. The edges can be:

                - 2-tuples (u, v) or
                - 3-tuples (u, v, d) for an edge data dict d, or
                - 3-tuples (u, v, k) for not iterable key k, or
                - 4-tuples (u, v, k, d) for an edge with data and key k

        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        A list of edge keys assigned to the edges in `ebunch`.

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

        Default keys are generated using the method ``new_edge_key()``.
        This method can be overridden by subclassing the base class and
        providing a custom ``new_edge_key()`` method.

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

        >>> G = nx.MultiGraph([(1, 2), (2, 3), (3, 4)])
        >>> # Grow graph by one new node, adding edges to all existing nodes.
        >>> # wrong way - will raise RuntimeError
        >>> # G.add_edges_from(((5, n) for n in G.nodes))
        >>> # right way - note that there will be no self-edge for node 5
        >>> assigned_keys = G.add_edges_from(list((5, n) for n in G.nodes))
        """
        keylist = []
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 4:
                u, v, key, dd = e
            elif ne == 3:
                u, v, dd = e
                key = None
            elif ne == 2:
                u, v = e
                dd = {}
                key = None
            else:
                msg = f"Edge tuple {e} must be a 2-tuple, 3-tuple or 4-tuple."
                raise NetworkXError(msg)
            ddd = {}
            ddd.update(attr)
            try:
                ddd.update(dd)
            except (TypeError, ValueError):
                if ne != 3:
                    raise
                key = dd  # ne == 3 with 3rd value not dict, must be a key
            key = self.add_edge(u, v, key)
            self[u][v][key].update(ddd)
            keylist.append(key)
        nx._clear_cache(self)
        return keylist

    def remove_edge(self, u, v, key=None):
        """Remove an edge between u and v.

        Parameters
        ----------
        u, v : nodes
            Remove an edge between nodes u and v.
        key : hashable identifier, optional (default=None)
            Used to distinguish multiple edges between a pair of nodes.
            If None, remove a single edge between u and v. If there are
            multiple edges, removes the last edge added in terms of
            insertion order.

        Raises
        ------
        NetworkXError
            If there is not an edge between u and v, or
            if there is no edge with the specified key.

        See Also
        --------
        remove_edges_from : remove a collection of edges

        Examples
        --------
        >>> G = nx.MultiGraph()
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.remove_edge(0, 1)
        >>> e = (1, 2)
        >>> G.remove_edge(*e)  # unpacks e from an edge tuple

        For multiple edges

        >>> G = nx.MultiGraph()  # or MultiDiGraph, etc
        >>> G.add_edges_from([(1, 2), (1, 2), (1, 2)])  # key_list returned
        [0, 1, 2]

        When ``key=None`` (the default), edges are removed in the opposite
        order that they were added:

        >>> G.remove_edge(1, 2)
        >>> G.edges(keys=True)
        MultiEdgeView([(1, 2, 0), (1, 2, 1)])
        >>> G.remove_edge(2, 1)  # edges are not directed
        >>> G.edges(keys=True)
        MultiEdgeView([(1, 2, 0)])

        For edges with keys

        >>> G = nx.MultiGraph()
        >>> G.add_edge(1, 2, key="first")
        'first'
        >>> G.add_edge(1, 2, key="second")
        'second'
        >>> G.remove_edge(1, 2, key="first")
        >>> G.edges(keys=True)
        MultiEdgeView([(1, 2, 'second')])

        """
        try:
            d = self._adj[u][v]
        except KeyError as err:
            raise NetworkXError(f"The edge {u}-{v} is not in the graph.") from err
        # remove the edge with specified data
        if key is None:
            d.popitem()
        else:
            try:
                del d[key]
            except KeyError as err:
                msg = f"The edge {u}-{v} with key {key} is not in the graph."
                raise NetworkXError(msg) from err
        if len(d) == 0:
            # remove the key entries if last edge
            del self._adj[u][v]
            if u != v:  # check for selfloop
                del self._adj[v][u]
        nx._clear_cache(self)

    def remove_edges_from(self, ebunch):
        """Remove all edges specified in ebunch.

        Parameters
        ----------
        ebunch: list or container of edge tuples
            Each edge given in the list or container will be removed
            from the graph. The edges can be:

                - 2-tuples (u, v) A single edge between u and v is removed.
                - 3-tuples (u, v, key) The edge identified by key is removed.
                - 4-tuples (u, v, key, data) where data is ignored.

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

        Removing multiple copies of edges

        >>> G = nx.MultiGraph()
        >>> keys = G.add_edges_from([(1, 2), (1, 2), (1, 2)])
        >>> G.remove_edges_from([(1, 2), (2, 1)])  # edges aren't directed
        >>> list(G.edges())
        [(1, 2)]
        >>> G.remove_edges_from([(1, 2), (1, 2)])  # silently ignore extra copy
        >>> list(G.edges)  # now empty graph
        []

        When the edge is a 2-tuple ``(u, v)`` but there are multiple edges between
        u and v in the graph, the most recent edge (in terms of insertion
        order) is removed.

        >>> G = nx.MultiGraph()
        >>> for key in ("x", "y", "a"):
        ...     k = G.add_edge(0, 1, key=key)
        >>> G.edges(keys=True)
        MultiEdgeView([(0, 1, 'x'), (0, 1, 'y'), (0, 1, 'a')])
        >>> G.remove_edges_from([(0, 1)])
        >>> G.edges(keys=True)
        MultiEdgeView([(0, 1, 'x'), (0, 1, 'y')])

        """
        for e in ebunch:
            try:
                self.remove_edge(*e[:3])
            except NetworkXError:
                pass
        nx._clear_cache(self)

    def has_edge(self, u, v, key=None):
        """Returns True if the graph has an edge between nodes u and v.

        This is the same as `v in G[u] or key in G[u][v]`
        without KeyError exceptions.

        Parameters
        ----------
        u, v : nodes
            Nodes can be, for example, strings or numbers.

        key : hashable identifier, optional (default=None)
            If specified return True only if the edge with
            key is found.

        Returns
        -------
        edge_ind : bool
            True if edge is in the graph, False otherwise.

        Examples
        --------
        Can be called either using two nodes u, v, an edge tuple (u, v),
        or an edge tuple (u, v, key).

        >>> G = nx.MultiGraph()  # or MultiDiGraph
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.has_edge(0, 1)  # using two nodes
        True
        >>> e = (0, 1)
        >>> G.has_edge(*e)  #  e is a 2-tuple (u, v)
        True
        >>> G.add_edge(0, 1, key="a")
        'a'
        >>> G.has_edge(0, 1, key="a")  # specify key
        True
        >>> G.has_edge(1, 0, key="a")  # edges aren't directed
        True
        >>> e = (0, 1, "a")
        >>> G.has_edge(*e)  # e is a 3-tuple (u, v, 'a')
        True

        The following syntax are equivalent:

        >>> G.has_edge(0, 1)
        True
        >>> 1 in G[0]  # though this gives :exc:`KeyError` if 0 not in G
        True
        >>> 0 in G[1]  # other order; also gives :exc:`KeyError` if 0 not in G
        True

        """
        try:
            if key is None:
                return v in self._adj[u]
            else:
                return key in self._adj[u][v]
        except KeyError:
            return False

    @cached_property
    def edges(self):
        """Returns an iterator over the edges.

        edges(self, nbunch=None, data=False, keys=False, default=None)

        The MultiEdgeView provides set-like operations on the edge-tuples
        as well as edge attribute lookup. When called, it also provides
        an EdgeDataView object which allows control of access to edge
        attributes (but does not provide set-like operations).
        Hence, ``G.edges[u, v, k]['color']`` provides the value of the color
        attribute for the edge from ``u`` to ``v`` with key ``k`` while
        ``for (u, v, k, c) in G.edges(data='color', keys=True, default="red"):``
        iterates through all the edges yielding the color attribute with
        default `'red'` if no color attribute exists.

        Edges are returned as tuples with optional data and keys
        in the order (node, neighbor, key, data). If ``keys=True`` is not
        provided, the tuples will just be (node, neighbor, data), but
        multiple tuples with the same node and neighbor will be generated
        when multiple edges exist between two nodes.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges from these nodes.
        data : string or bool, optional (default=False)
            The edge attribute returned in 3-tuple (u, v, ddict[data]).
            If True, return edge attribute dict in 3-tuple (u, v, ddict).
            If False, return 2-tuple (u, v).
        keys : bool, optional (default=False)
            If True, return edge keys with each edge, creating (u, v, k)
            tuples or (u, v, k, d) tuples if data is also requested.
        default : value, optional (default=None)
            Value used for edges that don't have the requested attribute.
            Only relevant if data is not True or False.

        Returns
        -------
        edges : MultiEdgeView
            A view of edge attributes, usually it iterates over (u, v)
            (u, v, k) or (u, v, k, d) tuples of edges, but can also be
            used for attribute lookup as ``edges[u, v, k]['foo']``.

        Notes
        -----
        Nodes in nbunch that are not in the graph will be (quietly) ignored.
        For directed graphs this returns the out-edges.

        Examples
        --------
        >>> G = nx.MultiGraph()
        >>> nx.add_path(G, [0, 1, 2])
        >>> key = G.add_edge(2, 3, weight=5)
        >>> key2 = G.add_edge(2, 1, weight=2)  # multi-edge
        >>> [e for e in G.edges()]
        [(0, 1), (1, 2), (1, 2), (2, 3)]
        >>> G.edges.data()  # default data is {} (empty dict)
        MultiEdgeDataView([(0, 1, {}), (1, 2, {}), (1, 2, {'weight': 2}), (2, 3, {'weight': 5})])
        >>> G.edges.data("weight", default=1)
        MultiEdgeDataView([(0, 1, 1), (1, 2, 1), (1, 2, 2), (2, 3, 5)])
        >>> G.edges(keys=True)  # default keys are integers
        MultiEdgeView([(0, 1, 0), (1, 2, 0), (1, 2, 1), (2, 3, 0)])
        >>> G.edges.data(keys=True)
        MultiEdgeDataView([(0, 1, 0, {}), (1, 2, 0, {}), (1, 2, 1, {'weight': 2}), (2, 3, 0, {'weight': 5})])
        >>> G.edges.data("weight", default=1, keys=True)
        MultiEdgeDataView([(0, 1, 0, 1), (1, 2, 0, 1), (1, 2, 1, 2), (2, 3, 0, 5)])
        >>> G.edges([0, 3])  # Note ordering of tuples from listed sources
        MultiEdgeDataView([(0, 1), (3, 2)])
        >>> G.edges([0, 3, 2, 1])  # Note ordering of tuples
        MultiEdgeDataView([(0, 1), (3, 2), (2, 1), (2, 1)])
        >>> G.edges(0)
        MultiEdgeDataView([(0, 1)])
        """
        return MultiEdgeView(self)

    def get_edge_data(self, u, v, key=None, default=None):
        """Returns the attribute dictionary associated with edge (u, v,
        key).

        If a key is not provided, returns a dictionary mapping edge keys
        to attribute dictionaries for each edge between u and v.

        This is identical to `G[u][v][key]` except the default is returned
        instead of an exception is the edge doesn't exist.

        Parameters
        ----------
        u, v : nodes

        default :  any Python object (default=None)
            Value to return if the specific edge (u, v, key) is not
            found, OR if there are no edges between u and v and no key
            is specified.

        key : hashable identifier, optional (default=None)
            Return data only for the edge with specified key, as an
            attribute dictionary (rather than a dictionary mapping keys
            to attribute dictionaries).

        Returns
        -------
        edge_dict : dictionary
            The edge attribute dictionary, OR a dictionary mapping edge
            keys to attribute dictionaries for each of those edges if no
            specific key is provided (even if there's only one edge
            between u and v).

        Examples
        --------
        >>> G = nx.MultiGraph()  # or MultiDiGraph
        >>> key = G.add_edge(0, 1, key="a", weight=7)
        >>> G[0][1]["a"]  # key='a'
        {'weight': 7}
        >>> G.edges[0, 1, "a"]  # key='a'
        {'weight': 7}

        Warning: we protect the graph data structure by making
        `G.edges` and `G[1][2]` read-only dict-like structures.
        However, you can assign values to attributes in e.g.
        `G.edges[1, 2, 'a']` or `G[1][2]['a']` using an additional
        bracket as shown next. You need to specify all edge info
        to assign to the edge data associated with an edge.

        >>> G[0][1]["a"]["weight"] = 10
        >>> G.edges[0, 1, "a"]["weight"] = 10
        >>> G[0][1]["a"]["weight"]
        10
        >>> G.edges[1, 0, "a"]["weight"]
        10

        >>> G = nx.MultiGraph()  # or MultiDiGraph
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.edges[0, 1, 0]["weight"] = 5
        >>> G.get_edge_data(0, 1)
        {0: {'weight': 5}}
        >>> e = (0, 1)
        >>> G.get_edge_data(*e)  # tuple form
        {0: {'weight': 5}}
        >>> G.get_edge_data(3, 0)  # edge not in graph, returns None
        >>> G.get_edge_data(3, 0, default=0)  # edge not in graph, return default
        0
        >>> G.get_edge_data(1, 0, 0)  # specific key gives back
        {'weight': 5}
        """
        try:
            if key is None:
                return self._adj[u][v]
            else:
                return self._adj[u][v][key]
        except KeyError:
            return default

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
        MultiDegreeView or int
            If multiple nodes are requested (the default), returns a `MultiDegreeView`
            mapping nodes to their degree.
            If a single node is requested, returns the degree of the node as an integer.

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.degree(0)  # node 0 with degree 1
        1
        >>> list(G.degree([0, 1]))
        [(0, 1), (1, 2)]

        """
        return MultiDegreeView(self)

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return True

    def is_directed(self):
        """Returns True if graph is directed, False otherwise."""
        return False

    def copy(self, as_view=False):
        """Returns a copy of the graph.

        The copy method by default returns an independent shallow copy
        of the graph and attributes. That is, if an attribute is a
        container, that container is shared by the original an the copy.
        Use Python's `copy.deepcopy` for new containers.

        If `as_view` is True then a view is returned instead of a copy.

        Notes
        -----
        All copies reproduce the graph structure, but data attributes
        may be handled in different ways. There are four types of copies
        of a graph that people might want.

        Deepcopy -- A "deepcopy" copies the graph structure as well as
        all data attributes and any objects they might contain.
        The entire graph object is new so that changes in the copy
        do not affect the original object. (see Python's copy.deepcopy)

        Data Reference (Shallow) -- For a shallow copy the graph structure
        is copied but the edge, node and graph attribute dicts are
        references to those in the original graph. This saves
        time and memory but could cause confusion if you change an attribute
        in one graph and it changes the attribute in the other.
        NetworkX does not provide this level of shallow copy.

        Independent Shallow -- This copy creates new independent attribute
        dicts and then does a shallow copy of the attributes. That is, any
        attributes that are containers are shared between the new graph
        and the original. This is exactly what `dict.copy()` provides.
        You can obtain this style copy using:

            >>> G = nx.path_graph(5)
            >>> H = G.copy()
            >>> H = G.copy(as_view=False)
            >>> H = nx.Graph(G)
            >>> H = G.__class__(G)

        Fresh Data -- For fresh data, the graph structure is copied while
        new empty data attribute dicts are created. The resulting graph
        is independent of the original and it has no edge, node or graph
        attributes. Fresh copies are not enabled. Instead use:

            >>> H = G.__class__()
            >>> H.add_nodes_from(G)
            >>> H.add_edges_from(G.edges)

        View -- Inspired by dict-views, graph-views act like read-only
        versions of the original graph, providing a copy of the original
        structure without requiring any memory for copying the information.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Parameters
        ----------
        as_view : bool, optional (default=False)
            If True, the returned graph-view provides a read-only view
            of the original graph without actually copying any data.

        Returns
        -------
        G : Graph
            A copy of the graph.

        See Also
        --------
        to_directed: return a directed copy of the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> H = G.copy()

        """
        if as_view is True:
            return nx.graphviews.generic_graph_view(self)
        G = self.__class__()
        G.graph.update(self.graph)
        G.add_nodes_from((n, d.copy()) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, key, datadict.copy())
            for u, nbrs in self._adj.items()
            for v, keydict in nbrs.items()
            for key, datadict in keydict.items()
        )
        return G

    def to_directed(self, as_view=False):
        """Returns a directed representation of the graph.

        Returns
        -------
        G : MultiDiGraph
            A directed graph with the same name, same nodes, and with
            each edge (u, v, k, data) replaced by two directed edges
            (u, v, k, data) and (v, u, k, data).

        Notes
        -----
        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.

        This is in contrast to the similar D=MultiDiGraph(G) which
        returns a shallow copy of the data.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Warning: If you have subclassed MultiGraph to use dict-like objects
        in the data structure, those changes do not transfer to the
        MultiDiGraph created by this method.

        Examples
        --------
        >>> G = nx.MultiGraph()
        >>> G.add_edge(0, 1)
        0
        >>> G.add_edge(0, 1)
        1
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)]

        If already directed, return a (deep) copy

        >>> G = nx.MultiDiGraph()
        >>> G.add_edge(0, 1)
        0
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1, 0)]
        """
        graph_class = self.to_directed_class()
        if as_view is True:
            return nx.graphviews.generic_graph_view(self, graph_class)
        # deepcopy when not a view
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, key, deepcopy(datadict))
            for u, nbrs in self.adj.items()
            for v, keydict in nbrs.items()
            for key, datadict in keydict.items()
        )
        return G

    def to_undirected(self, as_view=False):
        """Returns an undirected copy of the graph.

        Returns
        -------
        G : Graph/MultiGraph
            A deepcopy of the graph.

        See Also
        --------
        copy, add_edge, add_edges_from

        Notes
        -----
        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.

        This is in contrast to the similar `G = nx.MultiGraph(D)`
        which returns a shallow copy of the data.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Warning: If you have subclassed MultiGraph to use dict-like
        objects in the data structure, those changes do not transfer
        to the MultiGraph created by this method.

        Examples
        --------
        >>> G = nx.MultiGraph([(0, 1), (0, 1), (1, 2)])
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 2, 0), (2, 1, 0)]
        >>> G2 = H.to_undirected()
        >>> list(G2.edges)
        [(0, 1, 0), (0, 1, 1), (1, 2, 0)]
        """
        graph_class = self.to_undirected_class()
        if as_view is True:
            return nx.graphviews.generic_graph_view(self, graph_class)
        # deepcopy when not a view
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, key, deepcopy(datadict))
            for u, nbrs in self._adj.items()
            for v, keydict in nbrs.items()
            for key, datadict in keydict.items()
        )
        return G

    def number_of_edges(self, u=None, v=None):
        """Returns the number of edges between two nodes.

        Parameters
        ----------
        u, v : nodes, optional (Default=all edges)
            If u and v are specified, return the number of edges between
            u and v. Otherwise return the total number of all edges.

        Returns
        -------
        nedges : int
            The number of edges in the graph.  If nodes `u` and `v` are
            specified return the number of edges between those nodes. If
            the graph is directed, this only returns the number of edges
            from `u` to `v`.

        See Also
        --------
        size

        Examples
        --------
        For undirected multigraphs, this method counts the total number
        of edges in the graph::

            >>> G = nx.MultiGraph()
            >>> G.add_edges_from([(0, 1), (0, 1), (1, 2)])
            [0, 1, 0]
            >>> G.number_of_edges()
            3

        If you specify two nodes, this counts the total number of edges
        joining the two nodes::

            >>> G.number_of_edges(0, 1)
            2

        For directed multigraphs, this method can count the total number
        of directed edges from `u` to `v`::

            >>> G = nx.MultiDiGraph()
            >>> G.add_edges_from([(0, 1), (0, 1), (1, 0)])
            [0, 1, 0]
            >>> G.number_of_edges(0, 1)
            2
            >>> G.number_of_edges(1, 0)
            1

        """
        if u is None:
            return self.size()
        try:
            edgedata = self._adj[u][v]
        except KeyError:
            return 0  # no such edge
        return len(edgedata)
<|endoftext|>"""Filter factories to hide or show sets of nodes and edges.

These filters return the function used when creating `SubGraph`.
"""

__all__ = [
    "no_filter",
    "hide_nodes",
    "hide_edges",
    "hide_multiedges",
    "hide_diedges",
    "hide_multidiedges",
    "show_nodes",
    "show_edges",
    "show_multiedges",
    "show_diedges",
    "show_multidiedges",
]


def no_filter(*items):
    """Returns a filter function that always evaluates to True."""
    return True


def hide_nodes(nodes):
    """Returns a filter function that hides specific nodes."""
    nodes = set(nodes)
    return lambda node: node not in nodes


def hide_diedges(edges):
    """Returns a filter function that hides specific directed edges."""
    edges = {(u, v) for u, v in edges}
    return lambda u, v: (u, v) not in edges


def hide_edges(edges):
    """Returns a filter function that hides specific undirected edges."""
    alledges = set(edges) | {(v, u) for (u, v) in edges}
    return lambda u, v: (u, v) not in alledges


def hide_multidiedges(edges):
    """Returns a filter function that hides specific multi-directed edges."""
    edges = {(u, v, k) for u, v, k in edges}
    return lambda u, v, k: (u, v, k) not in edges


def hide_multiedges(edges):
    """Returns a filter function that hides specific multi-undirected edges."""
    alledges = set(edges) | {(v, u, k) for (u, v, k) in edges}
    return lambda u, v, k: (u, v, k) not in alledges


# write show_nodes as a class to make SubGraph pickleable
class show_nodes:
    """Filter class to show specific nodes.

    Attach the set of nodes as an attribute to speed up this commonly used filter

    Note that another allowed attribute for filters is to store the number of nodes
    on the filter as attribute `length` (used in `__len__`). It is a user
    responsibility to ensure this attribute is accurate if present.
    """

    def __init__(self, nodes):
        self.nodes = set(nodes)

    def __call__(self, node):
        return node in self.nodes


def show_diedges(edges):
    """Returns a filter function that shows specific directed edges."""
    edges = {(u, v) for u, v in edges}
    return lambda u, v: (u, v) in edges


def show_edges(edges):
    """Returns a filter function that shows specific undirected edges."""
    alledges = set(edges) | {(v, u) for (u, v) in edges}
    return lambda u, v: (u, v) in alledges


def show_multidiedges(edges):
    """Returns a filter function that shows specific multi-directed edges."""
    edges = {(u, v, k) for u, v, k in edges}
    return lambda u, v, k: (u, v, k) in edges


def show_multiedges(edges):
    """Returns a filter function that shows specific multi-undirected edges."""
    alledges = set(edges) | {(v, u, k) for (u, v, k) in edges}
    return lambda u, v, k: (u, v, k) in alledges
<|endoftext|>"""Functional interface to graph methods and assorted utilities."""

from collections import Counter
from itertools import chain

import networkx as nx
from networkx.utils import not_implemented_for, pairwise

__all__ = [
    "nodes",
    "edges",
    "degree",
    "degree_histogram",
    "neighbors",
    "number_of_nodes",
    "number_of_edges",
    "density",
    "is_directed",
    "freeze",
    "is_frozen",
    "subgraph",
    "induced_subgraph",
    "edge_subgraph",
    "restricted_view",
    "to_directed",
    "to_undirected",
    "add_star",
    "add_path",
    "add_cycle",
    "create_empty_copy",
    "set_node_attributes",
    "get_node_attributes",
    "remove_node_attributes",
    "set_edge_attributes",
    "get_edge_attributes",
    "remove_edge_attributes",
    "all_neighbors",
    "non_neighbors",
    "non_edges",
    "common_neighbors",
    "is_weighted",
    "is_negatively_weighted",
    "is_empty",
    "selfloop_edges",
    "nodes_with_selfloops",
    "number_of_selfloops",
    "path_weight",
    "is_path",
]


def nodes(G):
    """Returns a NodeView over the graph nodes.

    This function wraps the :func:`G.nodes <networkx.Graph.nodes>` property.
    """
    return G.nodes()


def edges(G, nbunch=None):
    """Returns an edge view of edges incident to nodes in nbunch.

    Return all edges if nbunch is unspecified or nbunch=None.

    For digraphs, edges=out_edges

    This function wraps the :func:`G.edges <networkx.Graph.edges>` property.
    """
    return G.edges(nbunch)


def degree(G, nbunch=None, weight=None):
    """Returns a degree view of single node or of nbunch of nodes.
    If nbunch is omitted, then return degrees of *all* nodes.

    This function wraps the :func:`G.degree <networkx.Graph.degree>` property.
    """
    return G.degree(nbunch, weight)


def neighbors(G, n):
    """Returns an iterator over all neighbors of node n.

    This function wraps the :func:`G.neighbors <networkx.Graph.neighbors>` function.
    """
    return G.neighbors(n)


def number_of_nodes(G):
    """Returns the number of nodes in the graph.

    This function wraps the :func:`G.number_of_nodes <networkx.Graph.number_of_nodes>` function.
    """
    return G.number_of_nodes()


def number_of_edges(G):
    """Returns the number of edges in the graph.

    This function wraps the :func:`G.number_of_edges <networkx.Graph.number_of_edges>` function.
    """
    return G.number_of_edges()


def density(G):
    r"""Returns the density of a graph.

    The density for undirected graphs is

    .. math::

       d = \frac{2m}{n(n-1)},

    and for directed graphs is

    .. math::

       d = \frac{m}{n(n-1)},

    where `n` is the number of nodes and `m`  is the number of edges in `G`.

    Notes
    -----
    The density is 0 for a graph without edges and 1 for a complete graph.
    The density of multigraphs can be higher than 1.

    Self loops are counted in the total number of edges so graphs with self
    loops can have density higher than 1.
    """
    n = number_of_nodes(G)
    m = number_of_edges(G)
    if m == 0 or n <= 1:
        return 0
    d = m / (n * (n - 1))
    if not G.is_directed():
        d *= 2
    return d


def degree_histogram(G):
    """Returns a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    counts = Counter(d for n, d in G.degree())
    return [counts.get(i, 0) for i in range(max(counts) + 1 if counts else 0)]


def is_directed(G):
    """Return True if graph is directed."""
    return G.is_directed()


def frozen(*args, **kwargs):
    """Dummy method for raising errors when trying to modify frozen graphs"""
    raise nx.NetworkXError("Frozen graph can't be modified")


def freeze(G):
    """Modify graph to prevent further change by adding or removing
    nodes or edges.

    Node and edge data can still be modified.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> G = nx.freeze(G)
    >>> try:
    ...     G.add_edge(4, 5)
    ... except nx.NetworkXError as err:
    ...     print(str(err))
    Frozen graph can't be modified

    Notes
    -----
    To "unfreeze" a graph you must make a copy by creating a new graph object:

    >>> graph = nx.path_graph(4)
    >>> frozen_graph = nx.freeze(graph)
    >>> unfrozen_graph = nx.Graph(frozen_graph)
    >>> nx.is_frozen(unfrozen_graph)
    False

    See Also
    --------
    is_frozen
    """
    G.add_node = frozen
    G.add_nodes_from = frozen
    G.remove_node = frozen
    G.remove_nodes_from = frozen
    G.add_edge = frozen
    G.add_edges_from = frozen
    G.add_weighted_edges_from = frozen
    G.remove_edge = frozen
    G.remove_edges_from = frozen
    G.clear = frozen
    G.clear_edges = frozen
    G.frozen = True
    return G


def is_frozen(G):
    """Returns True if graph is frozen.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    See Also
    --------
    freeze
    """
    try:
        return G.frozen
    except AttributeError:
        return False


def add_star(G_to_add_to, nodes_for_star, **attr):
    """Add a star to Graph G_to_add_to.

    The first node in `nodes_for_star` is the middle of the star.
    It is connected to all other nodes.

    Parameters
    ----------
    G_to_add_to : graph
        A NetworkX graph
    nodes_for_star : iterable container
        A container of nodes.
    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to every edge in star.

    See Also
    --------
    add_path, add_cycle

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_star(G, [0, 1, 2, 3])
    >>> nx.add_star(G, [10, 11, 12], weight=2)
    """
    nlist = iter(nodes_for_star)
    try:
        v = next(nlist)
    except StopIteration:
        return
    G_to_add_to.add_node(v)
    edges = ((v, n) for n in nlist)
    G_to_add_to.add_edges_from(edges, **attr)


def add_path(G_to_add_to, nodes_for_path, **attr):
    """Add a path to the Graph G_to_add_to.

    Parameters
    ----------
    G_to_add_to : graph
        A NetworkX graph
    nodes_for_path : iterable container
        A container of nodes.  A path will be constructed from
        the nodes (in order) and added to the graph.
    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to every edge in path.

    See Also
    --------
    add_star, add_cycle

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_path(G, [0, 1, 2, 3])
    >>> nx.add_path(G, [10, 11, 12], weight=7)
    """
    nlist = iter(nodes_for_path)
    try:
        first_node = next(nlist)
    except StopIteration:
        return
    G_to_add_to.add_node(first_node)
    G_to_add_to.add_edges_from(pairwise(chain((first_node,), nlist)), **attr)


def add_cycle(G_to_add_to, nodes_for_cycle, **attr):
    """Add a cycle to the Graph G_to_add_to.

    Parameters
    ----------
    G_to_add_to : graph
        A NetworkX graph
    nodes_for_cycle: iterable container
        A container of nodes.  A cycle will be constructed from
        the nodes (in order) and added to the graph.
    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to every edge in cycle.

    See Also
    --------
    add_path, add_star

    Examples
    --------
    >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    >>> nx.add_cycle(G, [0, 1, 2, 3])
    >>> nx.add_cycle(G, [10, 11, 12], weight=7)
    """
    nlist = iter(nodes_for_cycle)
    try:
        first_node = next(nlist)
    except StopIteration:
        return
    G_to_add_to.add_node(first_node)
    G_to_add_to.add_edges_from(
        pairwise(chain((first_node,), nlist), cyclic=True), **attr
    )


def subgraph(G, nbunch):
    """Returns the subgraph induced on nodes in nbunch.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nbunch : list, iterable
       A container of nodes that will be iterated through once (thus
       it should be an iterator or be iterable).  Each element of the
       container should be a valid node type: any hashable type except
       None.  If nbunch is None, return all edges data in the graph.
       Nodes in nbunch that are not in the graph will be (quietly)
       ignored.

    Notes
    -----
    subgraph(G) calls G.subgraph()
    """
    return G.subgraph(nbunch)


def induced_subgraph(G, nbunch):
    """Returns a SubGraph view of `G` showing only nodes in nbunch.

    The induced subgraph of a graph on a set of nodes N is the
    graph with nodes N and edges from G which have both ends in N.

    Parameters
    ----------
    G : NetworkX Graph
    nbunch : node, container of nodes or None (for all nodes)

    Returns
    -------
    subgraph : SubGraph View
        A read-only view of the subgraph in `G` induced by the nodes.
        Changes to the graph `G` will be reflected in the view.

    Notes
    -----
    To create a mutable subgraph with its own copies of nodes
    edges and attributes use `subgraph.copy()` or `Graph(subgraph)`

    For an inplace reduction of a graph to a subgraph you can remove nodes:
    `G.remove_nodes_from(n in G if n not in set(nbunch))`

    If you are going to compute subgraphs of your subgraphs you could
    end up with a chain of views that can be very slow once the chain
    has about 15 views in it. If they are all induced subgraphs, you
    can short-cut the chain by making them all subgraphs of the original
    graph. The graph class method `G.subgraph` does this when `G` is
    a subgraph. In contrast, this function allows you to choose to build
    chains or not, as you wish. The returned subgraph is a view on `G`.

    Examples
    --------
    >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
    >>> H = nx.induced_subgraph(G, [0, 1, 3])
    >>> list(H.edges)
    [(0, 1)]
    >>> list(H.nodes)
    [0, 1, 3]
    """
    induced_nodes = nx.filters.show_nodes(G.nbunch_iter(nbunch))
    return nx.subgraph_view(G, filter_node=induced_nodes)


def edge_subgraph(G, edges):
    """Returns a view of the subgraph induced by the specified edges.

    The induced subgraph contains each edge in `edges` and each
    node incident to any of those edges.

    Parameters
    ----------
    G : NetworkX Graph
    edges : iterable
        An iterable of edges. Edges not present in `G` are ignored.

    Returns
    -------
    subgraph : SubGraph View
        A read-only edge-induced subgraph of `G`.
        Changes to `G` are reflected in the view.

    Notes
    -----
    To create a mutable subgraph with its own copies of nodes
    edges and attributes use `subgraph.copy()` or `Graph(subgraph)`

    If you create a subgraph of a subgraph recursively you can end up
    with a chain of subgraphs that becomes very slow with about 15
    nested subgraph views. Luckily the edge_subgraph filter nests
    nicely so you can use the original graph as G in this function
    to avoid chains. We do not rule out chains programmatically so
    that odd cases like an `edge_subgraph` of a `restricted_view`
    can be created.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> H = G.edge_subgraph([(0, 1), (3, 4)])
    >>> list(H.nodes)
    [0, 1, 3, 4]
    >>> list(H.edges)
    [(0, 1), (3, 4)]
    """
    nxf = nx.filters
    edges = set(edges)
    nodes = set()
    for e in edges:
        nodes.update(e[:2])
    induced_nodes = nxf.show_nodes(nodes)
    if G.is_multigraph():
        if G.is_directed():
            induced_edges = nxf.show_multidiedges(edges)
        else:
            induced_edges = nxf.show_multiedges(edges)
    else:
        if G.is_directed():
            induced_edges = nxf.show_diedges(edges)
        else:
            induced_edges = nxf.show_edges(edges)
    return nx.subgraph_view(G, filter_node=induced_nodes, filter_edge=induced_edges)


def restricted_view(G, nodes, edges):
    """Returns a view of `G` with hidden nodes and edges.

    The resulting subgraph filters out node `nodes` and edges `edges`.
    Filtered out nodes also filter out any of their edges.

    Parameters
    ----------
    G : NetworkX Graph
    nodes : iterable
        An iterable of nodes. Nodes not present in `G` are ignored.
    edges : iterable
        An iterable of edges. Edges not present in `G` are ignored.

    Returns
    -------
    subgraph : SubGraph View
        A read-only restricted view of `G` filtering out nodes and edges.
        Changes to `G` are reflected in the view.

    Notes
    -----
    To create a mutable subgraph with its own copies of nodes
    edges and attributes use `subgraph.copy()` or `Graph(subgraph)`

    If you create a subgraph of a subgraph recursively you may end up
    with a chain of subgraph views. Such chains can get quite slow
    for lengths near 15. To avoid long chains, try to make your subgraph
    based on the original graph.  We do not rule out chains programmatically
    so that odd cases like an `edge_subgraph` of a `restricted_view`
    can be created.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> H = nx.restricted_view(G, [0], [(1, 2), (3, 4)])
    >>> list(H.nodes)
    [1, 2, 3, 4]
    >>> list(H.edges)
    [(2, 3)]
    """
    nxf = nx.filters
    hide_nodes = nxf.hide_nodes(nodes)
    if G.is_multigraph():
        if G.is_directed():
            hide_edges = nxf.hide_multidiedges(edges)
        else:
            hide_edges = nxf.hide_multiedges(edges)
    else:
        if G.is_directed():
            hide_edges = nxf.hide_diedges(edges)
        else:
            hide_edges = nxf.hide_edges(edges)
    return nx.subgraph_view(G, filter_node=hide_nodes, filter_edge=hide_edges)


def to_directed(graph):
    """Returns a directed view of the graph `graph`.

    Identical to graph.to_directed(as_view=True)
    Note that graph.to_directed defaults to `as_view=False`
    while this function always provides a view.
    """
    return graph.to_directed(as_view=True)


def to_undirected(graph):
    """Returns an undirected view of the graph `graph`.

    Identical to graph.to_undirected(as_view=True)
    Note that graph.to_undirected defaults to `as_view=False`
    while this function always provides a view.
    """
    return graph.to_undirected(as_view=True)


def create_empty_copy(G, with_data=True):
    """Returns a copy of the graph G with all of the edges removed.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    with_data :  bool (default=True)
       Propagate Graph and Nodes data to the new graph.

    See Also
    --------
    empty_graph

    """
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=with_data))
    if with_data:
        H.graph.update(G.graph)
    return H


def set_node_attributes(G, values, name=None):
    """Sets node attributes from a given value or dictionary of values.

    .. Warning:: The call order of arguments `values` and `name`
        switched between v1.x & v2.x.

    Parameters
    ----------
    G : NetworkX Graph

    values : scalar value, dict-like
        What the node attribute should be set to.  If `values` is
        not a dictionary, then it is treated as a single attribute value
        that is then applied to every node in `G`.  This means that if
        you provide a mutable object, like a list, updates to that object
        will be reflected in the node attribute for every node.
        The attribute name will be `name`.

        If `values` is a dict or a dict of dict, it should be keyed
        by node to either an attribute value or a dict of attribute key/value
        pairs used to update the node's attributes.

    name : string (optional, default=None)
        Name of the node attribute to set if values is a scalar.

    Examples
    --------
    After computing some property of the nodes of a graph, you may want
    to assign a node attribute to store the value of that property for
    each node::

        >>> G = nx.path_graph(3)
        >>> bb = nx.betweenness_centrality(G)
        >>> isinstance(bb, dict)
        True
        >>> nx.set_node_attributes(G, bb, "betweenness")
        >>> G.nodes[1]["betweenness"]
        1.0

    If you provide a list as the second argument, updates to the list
    will be reflected in the node attribute for each node::

        >>> G = nx.path_graph(3)
        >>> labels = []
        >>> nx.set_node_attributes(G, labels, "labels")
        >>> labels.append("foo")
        >>> G.nodes[0]["labels"]
        ['foo']
        >>> G.nodes[1]["labels"]
        ['foo']
        >>> G.nodes[2]["labels"]
        ['foo']

    If you provide a dictionary of dictionaries as the second argument,
    the outer dictionary is assumed to be keyed by node to an inner
    dictionary of node attributes for that node::

        >>> G = nx.path_graph(3)
        >>> attrs = {0: {"attr1": 20, "attr2": "nothing"}, 1: {"attr2": 3}}
        >>> nx.set_node_attributes(G, attrs)
        >>> G.nodes[0]["attr1"]
        20
        >>> G.nodes[0]["attr2"]
        'nothing'
        >>> G.nodes[1]["attr2"]
        3
        >>> G.nodes[2]
        {}

    Note that if the dictionary contains nodes that are not in `G`, the
    values are silently ignored::

        >>> G = nx.Graph()
        >>> G.add_node(0)
        >>> nx.set_node_attributes(G, {0: "red", 1: "blue"}, name="color")
        >>> G.nodes[0]["color"]
        'red'
        >>> 1 in G.nodes
        False

    """
    # Set node attributes based on type of `values`
    if name is not None:  # `values` must not be a dict of dict
        try:  # `values` is a dict
            for n, v in values.items():
                try:
                    G.nodes[n][name] = values[n]
                except KeyError:
                    pass
        except AttributeError:  # `values` is a constant
            for n in G:
                G.nodes[n][name] = values
    else:  # `values` must be dict of dict
        for n, d in values.items():
            try:
                G.nodes[n].update(d)
            except KeyError:
                pass
    nx._clear_cache(G)


def get_node_attributes(G, name, default=None):
    """Get node attributes from graph

    Parameters
    ----------
    G : NetworkX Graph

    name : string
       Attribute name

    default: object (default=None)
       Default value of the node attribute if there is no value set for that
       node in graph. If `None` then nodes without this attribute are not
       included in the returned dict.

    Returns
    -------
    Dictionary of attributes keyed by node.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_nodes_from([1, 2, 3], color="red")
    >>> color = nx.get_node_attributes(G, "color")
    >>> color[1]
    'red'
    >>> G.add_node(4)
    >>> color = nx.get_node_attributes(G, "color", default="yellow")
    >>> color[4]
    'yellow'
    """
    if default is not None:
        return {n: d.get(name, default) for n, d in G.nodes.items()}
    return {n: d[name] for n, d in G.nodes.items() if name in d}


def remove_node_attributes(G, *attr_names, nbunch=None):
    """Remove node attributes from all nodes in the graph.

    Parameters
    ----------
    G : NetworkX Graph

    *attr_names : List of Strings
        The attribute names to remove from the graph.

    nbunch : List of Nodes
        Remove the node attributes only from the nodes in this list.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_nodes_from([1, 2, 3], color="blue")
    >>> nx.get_node_attributes(G, "color")
    {1: 'blue', 2: 'blue', 3: 'blue'}
    >>> nx.remove_node_attributes(G, "color")
    >>> nx.get_node_attributes(G, "color")
    {}
    """

    if nbunch is None:
        nbunch = G.nodes()

    for attr in attr_names:
        for n, d in G.nodes(data=True):
            if n in nbunch:
                try:
                    del d[attr]
                except KeyError:
                    pass


def set_edge_attributes(G, values, name=None):
    """Sets edge attributes from a given value or dictionary of values.

    .. Warning:: The call order of arguments `values` and `name`
        switched between v1.x & v2.x.

    Parameters
    ----------
    G : NetworkX Graph

    values : scalar value, dict-like
        What the edge attribute should be set to.  If `values` is
        not a dictionary, then it is treated as a single attribute value
        that is then applied to every edge in `G`.  This means that if
        you provide a mutable object, like a list, updates to that object
        will be reflected in the edge attribute for each edge.  The attribute
        name will be `name`.

        If `values` is a dict or a dict of dict, it should be keyed
        by edge tuple to either an attribute value or a dict of attribute
        key/value pairs used to update the edge's attributes.
        For multigraphs, the edge tuples must be of the form ``(u, v, key)``,
        where `u` and `v` are nodes and `key` is the edge key.
        For non-multigraphs, the keys must be tuples of the form ``(u, v)``.

    name : string (optional, default=None)
        Name of the edge attribute to set if values is a scalar.

    Examples
    --------
    After computing some property of the edges of a graph, you may want
    to assign a edge attribute to store the value of that property for
    each edge::

        >>> G = nx.path_graph(3)
        >>> bb = nx.edge_betweenness_centrality(G, normalized=False)
        >>> nx.set_edge_attributes(G, bb, "betweenness")
        >>> G.edges[1, 2]["betweenness"]
        2.0

    If you provide a list as the second argument, updates to the list
    will be reflected in the edge attribute for each edge::

        >>> labels = []
        >>> nx.set_edge_attributes(G, labels, "labels")
        >>> labels.append("foo")
        >>> G.edges[0, 1]["labels"]
        ['foo']
        >>> G.edges[1, 2]["labels"]
        ['foo']

    If you provide a dictionary of dictionaries as the second argument,
    the entire dictionary will be used to update edge attributes::

        >>> G = nx.path_graph(3)
        >>> attrs = {(0, 1): {"attr1": 20, "attr2": "nothing"}, (1, 2): {"attr2": 3}}
        >>> nx.set_edge_attributes(G, attrs)
        >>> G[0][1]["attr1"]
        20
        >>> G[0][1]["attr2"]
        'nothing'
        >>> G[1][2]["attr2"]
        3

    The attributes of one Graph can be used to set those of another.

        >>> H = nx.path_graph(3)
        >>> nx.set_edge_attributes(H, G.edges)

    Note that if the dict contains edges that are not in `G`, they are
    silently ignored::

        >>> G = nx.Graph([(0, 1)])
        >>> nx.set_edge_attributes(G, {(1, 2): {"weight": 2.0}})
        >>> (1, 2) in G.edges()
        False

    For multigraphs, the `values` dict is expected to be keyed by 3-tuples
    including the edge key::

        >>> MG = nx.MultiGraph()
        >>> edges = [(0, 1), (0, 1)]
        >>> MG.add_edges_from(edges)  # Returns list of edge keys
        [0, 1]
        >>> attributes = {(0, 1, 0): {"cost": 21}, (0, 1, 1): {"cost": 7}}
        >>> nx.set_edge_attributes(MG, attributes)
        >>> MG[0][1][0]["cost"]
        21
        >>> MG[0][1][1]["cost"]
        7

    If MultiGraph attributes are desired for a Graph, you must convert the 3-tuple
    multiedge to a 2-tuple edge and the last multiedge's attribute value will
    overwrite the previous values. Continuing from the previous case we get::

        >>> H = nx.path_graph([0, 1, 2])
        >>> nx.set_edge_attributes(H, {(u, v): ed for u, v, ed in MG.edges.data()})
        >>> nx.get_edge_attributes(H, "cost")
        {(0, 1): 7}

    """
    if name is not None:
        # `values` does not contain attribute names
        try:
            # if `values` is a dict using `.items()` => {edge: value}
            if G.is_multigraph():
                for (u, v, key), value in values.items():
                    try:
                        G._adj[u][v][key][name] = value
                    except KeyError:
                        pass
            else:
                for (u, v), value in values.items():
                    try:
                        G._adj[u][v][name] = value
                    except KeyError:
                        pass
        except AttributeError:
            # treat `values` as a constant
            for u, v, data in G.edges(data=True):
                data[name] = values
    else:
        # `values` consists of doct-of-dict {edge: {attr: value}} shape
        if G.is_multigraph():
            for (u, v, key), d in values.items():
                try:
                    G._adj[u][v][key].update(d)
                except KeyError:
                    pass
        else:
            for (u, v), d in values.items():
                try:
                    G._adj[u][v].update(d)
                except KeyError:
                    pass
    nx._clear_cache(G)


def get_edge_attributes(G, name, default=None):
    """Get edge attributes from graph

    Parameters
    ----------
    G : NetworkX Graph

    name : string
       Attribute name

    default: object (default=None)
       Default value of the edge attribute if there is no value set for that
       edge in graph. If `None` then edges without this attribute are not
       included in the returned dict.

    Returns
    -------
    Dictionary of attributes keyed by edge. For (di)graphs, the keys are
    2-tuples of the form: (u, v). For multi(di)graphs, the keys are 3-tuples of
    the form: (u, v, key).

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_path(G, [1, 2, 3], color="red")
    >>> color = nx.get_edge_attributes(G, "color")
    >>> color[(1, 2)]
    'red'
    >>> G.add_edge(3, 4)
    >>> color = nx.get_edge_attributes(G, "color", default="yellow")
    >>> color[(3, 4)]
    'yellow'
    """
    if G.is_multigraph():
        edges = G.edges(keys=True, data=True)
    else:
        edges = G.edges(data=True)
    if default is not None:
        return {x[:-1]: x[-1].get(name, default) for x in edges}
    return {x[:-1]: x[-1][name] for x in edges if name in x[-1]}


def remove_edge_attributes(G, *attr_names, ebunch=None):
    """Remove edge attributes from all edges in the graph.

    Parameters
    ----------
    G : NetworkX Graph

    *attr_names : List of Strings
        The attribute names to remove from the graph.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> nx.set_edge_attributes(G, {(u, v): u + v for u, v in G.edges()}, name="weight")
    >>> nx.get_edge_attributes(G, "weight")
    {(0, 1): 1, (1, 2): 3}
    >>> remove_edge_attributes(G, "weight")
    >>> nx.get_edge_attributes(G, "weight")
    {}
    """
    if ebunch is None:
        ebunch = G.edges(keys=True) if G.is_multigraph() else G.edges()

    for attr in attr_names:
        edges = (
            G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
        )
        for *e, d in edges:
            if tuple(e) in ebunch:
                try:
                    del d[attr]
                except KeyError:
                    pass


def all_neighbors(graph, node):
    """Returns all of the neighbors of a node in the graph.

    If the graph is directed returns predecessors as well as successors.

    Parameters
    ----------
    graph : NetworkX graph
        Graph to find neighbors.

    node : node
        The node whose neighbors will be returned.

    Returns
    -------
    neighbors : iterator
        Iterator of neighbors
    """
    if graph.is_directed():
        values = chain(graph.predecessors(node), graph.successors(node))
    else:
        values = graph.neighbors(node)
    return values


def non_neighbors(graph, node):
    """Returns the non-neighbors of the node in the graph.

    Parameters
    ----------
    graph : NetworkX graph
        Graph to find neighbors.

    node : node
        The node whose neighbors will be returned.

    Returns
    -------
    non_neighbors : set
        Set of nodes in the graph that are not neighbors of the node.
    """
    return graph._adj.keys() - graph._adj[node].keys() - {node}


def non_edges(graph):
    """Returns the nonexistent edges in the graph.

    Parameters
    ----------
    graph : NetworkX graph.
        Graph to find nonexistent edges.

    Returns
    -------
    non_edges : iterator
        Iterator of edges that are not in the graph.
    """
    if graph.is_directed():
        for u in graph:
            for v in non_neighbors(graph, u):
                yield (u, v)
    else:
        nodes = set(graph)
        while nodes:
            u = nodes.pop()
            for v in nodes - set(graph[u]):
                yield (u, v)


@not_implemented_for("directed")
def common_neighbors(G, u, v):
    """Returns the common neighbors of two nodes in a graph.

    Parameters
    ----------
    G : graph
        A NetworkX undirected graph.

    u, v : nodes
        Nodes in the graph.

    Returns
    -------
    cnbors : set
        Set of common neighbors of u and v in the graph.

    Raises
    ------
    NetworkXError
        If u or v is not a node in the graph.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> sorted(nx.common_neighbors(G, 0, 1))
    [2, 3, 4]
    """
    if u not in G:
        raise nx.NetworkXError("u is not in the graph.")
    if v not in G:
        raise nx.NetworkXError("v is not in the graph.")

    return G._adj[u].keys() & G._adj[v].keys() - {u, v}


def is_weighted(G, edge=None, weight="weight"):
    """Returns True if `G` has weighted edges.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    edge : tuple, optional
        A 2-tuple specifying the only edge in `G` that will be tested. If
        None, then every edge in `G` is tested.

    weight: string, optional
        The attribute name used to query for edge weights.

    Returns
    -------
    bool
        A boolean signifying if `G`, or the specified edge, is weighted.

    Raises
    ------
    NetworkXError
        If the specified edge does not exist.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.is_weighted(G)
    False
    >>> nx.is_weighted(G, (2, 3))
    False

    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2, weight=1)
    >>> nx.is_weighted(G)
    True

    """
    if edge is not None:
        data = G.get_edge_data(*edge)
        if data is None:
            msg = f"Edge {edge!r} does not exist."
            raise nx.NetworkXError(msg)
        return weight in data

    if is_empty(G):
        # Special handling required since: all([]) == True
        return False

    return all(weight in data for u, v, data in G.edges(data=True))


@nx._dispatchable(edge_attrs="weight")
def is_negatively_weighted(G, edge=None, weight="weight"):
    """Returns True if `G` has negatively weighted edges.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    edge : tuple, optional
        A 2-tuple specifying the only edge in `G` that will be tested. If
        None, then every edge in `G` is tested.

    weight: string, optional
        The attribute name used to query for edge weights.

    Returns
    -------
    bool
        A boolean signifying if `G`, or the specified edge, is negatively
        weighted.

    Raises
    ------
    NetworkXError
        If the specified edge does not exist.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 3), (2, 4), (2, 6)])
    >>> G.add_edge(1, 2, weight=4)
    >>> nx.is_negatively_weighted(G, (1, 2))
    False
    >>> G[2][4]["weight"] = -2
    >>> nx.is_negatively_weighted(G)
    True
    >>> G = nx.DiGraph()
    >>> edges = [("0", "3", 3), ("0", "1", -5), ("1", "0", -2)]
    >>> G.add_weighted_edges_from(edges)
    >>> nx.is_negatively_weighted(G)
    True

    """
    if edge is not None:
        data = G.get_edge_data(*edge)
        if data is None:
            msg = f"Edge {edge!r} does not exist."
            raise nx.NetworkXError(msg)
        return weight in data and data[weight] < 0

    return any(weight in data and data[weight] < 0 for u, v, data in G.edges(data=True))


def is_empty(G):
    """Returns True if `G` has no edges.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    Returns
    -------
    bool
        True if `G` has no edges, and False otherwise.

    Notes
    -----
    An empty graph can have nodes but not edges. The empty graph with zero
    nodes is known as the null graph. This is an $O(n)$ operation where n
    is the number of nodes in the graph.

    """
    return not any(G._adj.values())


def nodes_with_selfloops(G):
    """Returns an iterator over nodes with self loops.

    A node with a self loop has an edge with both ends adjacent
    to that node.

    Returns
    -------
    nodelist : iterator
        A iterator over nodes with self loops.

    See Also
    --------
    selfloop_edges, number_of_selfloops

    Examples
    --------
    >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    >>> G.add_edge(1, 1)
    >>> G.add_edge(1, 2)
    >>> list(nx.nodes_with_selfloops(G))
    [1]

    """
    return (n for n, nbrs in G._adj.items() if n in nbrs)


def selfloop_edges(G, data=False, keys=False, default=None):
    """Returns an iterator over selfloop edges.

    A selfloop edge has the same node at both ends.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    data : string or bool, optional (default=False)
        Return selfloop edges as two tuples (u, v) (data=False)
        or three-tuples (u, v, datadict) (data=True)
        or three-tuples (u, v, datavalue) (data='attrname')
    keys : bool, optional (default=False)
        If True, return edge keys with each edge.
    default : value, optional (default=None)
        Value used for edges that don't have the requested attribute.
        Only relevant if data is not True or False.

    Returns
    -------
    edgeiter : iterator over edge tuples
        An iterator over all selfloop edges.

    See Also
    --------
    nodes_with_selfloops, number_of_selfloops

    Examples
    --------
    >>> G = nx.MultiGraph()  # or Graph, DiGraph, MultiDiGraph, etc
    >>> ekey = G.add_edge(1, 1)
    >>> ekey = G.add_edge(1, 2)
    >>> list(nx.selfloop_edges(G))
    [(1, 1)]
    >>> list(nx.selfloop_edges(G, data=True))
    [(1, 1, {})]
    >>> list(nx.selfloop_edges(G, keys=True))
    [(1, 1, 0)]
    >>> list(nx.selfloop_edges(G, keys=True, data=True))
    [(1, 1, 0, {})]
    """
    if data is True:
        if G.is_multigraph():
            if keys is True:
                return (
                    (n, n, k, d)
                    for n, nbrs in G._adj.items()
                    if n in nbrs
                    for k, d in nbrs[n].items()
                )
            else:
                return (
                    (n, n, d)
                    for n, nbrs in G._adj.items()
                    if n in nbrs
                    for d in nbrs[n].values()
                )
        else:
            return ((n, n, nbrs[n]) for n, nbrs in G._adj.items() if n in nbrs)
    elif data is not False:
        if G.is_multigraph():
            if keys is True:
                return (
                    (n, n, k, d.get(data, default))
                    for n, nbrs in G._adj.items()
                    if n in nbrs
                    for k, d in nbrs[n].items()
                )
            else:
                return (
                    (n, n, d.get(data, default))
                    for n, nbrs in G._adj.items()
                    if n in nbrs
                    for d in nbrs[n].values()
                )
        else:
            return (
                (n, n, nbrs[n].get(data, default))
                for n, nbrs in G._adj.items()
                if n in nbrs
            )
    else:
        if G.is_multigraph():
            if keys is True:
                return (
                    (n, n, k)
                    for n, nbrs in G._adj.items()
                    if n in nbrs
                    for k in nbrs[n]
                )
            else:
                return (
                    (n, n)
                    for n, nbrs in G._adj.items()
                    if n in nbrs
                    for i in range(len(nbrs[n]))  # for easy edge removal (#4068)
                )
        else:
            return ((n, n) for n, nbrs in G._adj.items() if n in nbrs)


def number_of_selfloops(G):
    """Returns the number of selfloop edges.

    A selfloop edge has the same node at both ends.

    Returns
    -------
    nloops : int
        The number of selfloops.

    See Also
    --------
    nodes_with_selfloops, selfloop_edges

    Examples
    --------
    >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    >>> G.add_edge(1, 1)
    >>> G.add_edge(1, 2)
    >>> nx.number_of_selfloops(G)
    1
    """
    return sum(1 for _ in nx.selfloop_edges(G))


def is_path(G, path):
    """Returns whether or not the specified path exists.

    For it to return True, every node on the path must exist and
    each consecutive pair must be connected via one or more edges.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    path : list
        A list of nodes which defines the path to traverse

    Returns
    -------
    bool
        True if `path` is a valid path in `G`

    """
    try:
        return all(nbr in G._adj[node] for node, nbr in nx.utils.pairwise(path))
    except (KeyError, TypeError):
        return False


def path_weight(G, path, weight):
    """Returns total cost associated with specified path and weight

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    path: list
        A list of node labels which defines the path to traverse

    weight: string
        A string indicating which edge attribute to use for path cost

    Returns
    -------
    cost: int or float
        An integer or a float representing the total cost with respect to the
        specified weight of the specified path

    Raises
    ------
    NetworkXNoPath
        If the specified edge does not exist.
    """
    multigraph = G.is_multigraph()
    cost = 0

    if not nx.is_path(G, path):
        raise nx.NetworkXNoPath("path does not exist")
    for node, nbr in nx.utils.pairwise(path):
        if multigraph:
            cost += min(v[weight] for v in G._adj[node][nbr].values())
        else:
            cost += G._adj[node][nbr][weight]
    return cost
<|endoftext|>"""Views of core data structures such as nested Mappings (e.g. dict-of-dicts).
These ``Views`` often restrict element access, with either the entire view or
layers of nested mappings being read-only.
"""

from collections.abc import Mapping

__all__ = [
    "AtlasView",
    "AdjacencyView",
    "MultiAdjacencyView",
    "UnionAtlas",
    "UnionAdjacency",
    "UnionMultiInner",
    "UnionMultiAdjacency",
    "FilterAtlas",
    "FilterAdjacency",
    "FilterMultiInner",
    "FilterMultiAdjacency",
]


class AtlasView(Mapping):
    """An AtlasView is a Read-only Mapping of Mappings.

    It is a View into a dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer level is read-only.

    See Also
    ========
    AdjacencyView: View into dict-of-dict-of-dict
    MultiAdjacencyView: View into dict-of-dict-of-dict-of-dict
    """

    __slots__ = ("_atlas",)

    def __getstate__(self):
        return {"_atlas": self._atlas}

    def __setstate__(self, state):
        self._atlas = state["_atlas"]

    def __init__(self, d):
        self._atlas = d

    def __len__(self):
        return len(self._atlas)

    def __iter__(self):
        return iter(self._atlas)

    def __getitem__(self, key):
        return self._atlas[key]

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}

    def __str__(self):
        return str(self._atlas)  # {nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f"{self.__class__.__name__}({self._atlas!r})"


class AdjacencyView(AtlasView):
    """An AdjacencyView is a Read-only Map of Maps of Maps.

    It is a View into a dict-of-dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer levels are read-only.

    See Also
    ========
    AtlasView: View into dict-of-dict
    MultiAdjacencyView: View into dict-of-dict-of-dict-of-dict
    """

    __slots__ = ()  # Still uses AtlasView slots names _atlas

    def __getitem__(self, name):
        return AtlasView(self._atlas[name])

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}


class MultiAdjacencyView(AdjacencyView):
    """An MultiAdjacencyView is a Read-only Map of Maps of Maps of Maps.

    It is a View into a dict-of-dict-of-dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer levels are read-only.

    See Also
    ========
    AtlasView: View into dict-of-dict
    AdjacencyView: View into dict-of-dict-of-dict
    """

    __slots__ = ()  # Still uses AtlasView slots names _atlas

    def __getitem__(self, name):
        return AdjacencyView(self._atlas[name])

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}


class UnionAtlas(Mapping):
    """A read-only union of two atlases (dict-of-dict).

    The two dict-of-dicts represent the inner dict of
    an Adjacency:  `G.succ[node]` and `G.pred[node]`.
    The inner level of dict of both hold attribute key:value
    pairs and is read-write. But the outer level is read-only.

    See Also
    ========
    UnionAdjacency: View into dict-of-dict-of-dict
    UnionMultiAdjacency: View into dict-of-dict-of-dict-of-dict
    """

    __slots__ = ("_succ", "_pred")

    def __getstate__(self):
        return {"_succ": self._succ, "_pred": self._pred}

    def __setstate__(self, state):
        self._succ = state["_succ"]
        self._pred = state["_pred"]

    def __init__(self, succ, pred):
        self._succ = succ
        self._pred = pred

    def __len__(self):
        return len(self._succ.keys() | self._pred.keys())

    def __iter__(self):
        return iter(set(self._succ.keys()) | set(self._pred.keys()))

    def __getitem__(self, key):
        try:
            return self._succ[key]
        except KeyError:
            return self._pred[key]

    def copy(self):
        result = {nbr: dd.copy() for nbr, dd in self._succ.items()}
        for nbr, dd in self._pred.items():
            if nbr in result:
                result[nbr].update(dd)
            else:
                result[nbr] = dd.copy()
        return result

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f"{self.__class__.__name__}({self._succ!r}, {self._pred!r})"


class UnionAdjacency(Mapping):
    """A read-only union of dict Adjacencies as a Map of Maps of Maps.

    The two input dict-of-dict-of-dicts represent the union of
    `G.succ` and `G.pred`. Return values are UnionAtlas
    The inner level of dict is read-write. But the
    middle and outer levels are read-only.

    succ : a dict-of-dict-of-dict {node: nbrdict}
    pred : a dict-of-dict-of-dict {node: nbrdict}
    The keys for the two dicts should be the same

    See Also
    ========
    UnionAtlas: View into dict-of-dict
    UnionMultiAdjacency: View into dict-of-dict-of-dict-of-dict
    """

    __slots__ = ("_succ", "_pred")

    def __getstate__(self):
        return {"_succ": self._succ, "_pred": self._pred}

    def __setstate__(self, state):
        self._succ = state["_succ"]
        self._pred = state["_pred"]

    def __init__(self, succ, pred):
        # keys must be the same for two input dicts
        assert len(set(succ.keys()) ^ set(pred.keys())) == 0
        self._succ = succ
        self._pred = pred

    def __len__(self):
        return len(self._succ)  # length of each dict should be the same

    def __iter__(self):
        return iter(self._succ)

    def __getitem__(self, nbr):
        return UnionAtlas(self._succ[nbr], self._pred[nbr])

    def copy(self):
        return {n: self[n].copy() for n in self._succ}

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f"{self.__class__.__name__}({self._succ!r}, {self._pred!r})"


class UnionMultiInner(UnionAtlas):
    """A read-only union of two inner dicts of MultiAdjacencies.

    The two input dict-of-dict-of-dicts represent the union of
    `G.succ[node]` and `G.pred[node]` for MultiDiGraphs.
    Return values are UnionAtlas.
    The inner level of dict is read-write. But the outer levels are read-only.

    See Also
    ========
    UnionAtlas: View into dict-of-dict
    UnionAdjacency:  View into dict-of-dict-of-dict
    UnionMultiAdjacency:  View into dict-of-dict-of-dict-of-dict
    """

    __slots__ = ()  # Still uses UnionAtlas slots names _succ, _pred

    def __getitem__(self, node):
        in_succ = node in self._succ
        in_pred = node in self._pred
        if in_succ:
            if in_pred:
                return UnionAtlas(self._succ[node], self._pred[node])
            return UnionAtlas(self._succ[node], {})
        return UnionAtlas({}, self._pred[node])

    def copy(self):
        nodes = set(self._succ.keys()) | set(self._pred.keys())
        return {n: self[n].copy() for n in nodes}


class UnionMultiAdjacency(UnionAdjacency):
    """A read-only union of two dict MultiAdjacencies.

    The two input dict-of-dict-of-dict-of-dicts represent the union of
    `G.succ` and `G.pred` for MultiDiGraphs. Return values are UnionAdjacency.
    The inner level of dict is read-write. But the outer levels are read-only.

    See Also
    ========
    UnionAtlas:  View into dict-of-dict
    UnionMultiInner:  View into dict-of-dict-of-dict
    """

    __slots__ = ()  # Still uses UnionAdjacency slots names _succ, _pred

    def __getitem__(self, node):
        return UnionMultiInner(self._succ[node], self._pred[node])


class FilterAtlas(Mapping):  # nodedict, nbrdict, keydict
    """A read-only Mapping of Mappings with filtering criteria for nodes.

    It is a view into a dict-of-dict data structure, and it selects only
    nodes that meet the criteria defined by ``NODE_OK``.

    See Also
    ========
    FilterAdjacency
    FilterMultiInner
    FilterMultiAdjacency
    """

    def __init__(self, d, NODE_OK):
        self._atlas = d
        self.NODE_OK = NODE_OK

    def __len__(self):
        # check whether NODE_OK stores the number of nodes as `length`
        # or the nodes themselves as a set `nodes`. If not, count the nodes.
        if hasattr(self.NODE_OK, "length"):
            return self.NODE_OK.length
        if hasattr(self.NODE_OK, "nodes"):
            return len(self.NODE_OK.nodes & self._atlas.keys())
        return sum(1 for n in self._atlas if self.NODE_OK(n))

    def __iter__(self):
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return (n for n in self.NODE_OK.nodes if n in self._atlas)
        return (n for n in self._atlas if self.NODE_OK(n))

    def __getitem__(self, key):
        if key in self._atlas and self.NODE_OK(key):
            return self._atlas[key]
        raise KeyError(f"Key {key} not found")

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f"{self.__class__.__name__}({self._atlas!r}, {self.NODE_OK!r})"


class FilterAdjacency(Mapping):  # edgedict
    """A read-only Mapping of Mappings with filtering criteria for nodes and edges.

    It is a view into a dict-of-dict-of-dict data structure, and it selects nodes
    and edges that satisfy specific criteria defined by ``NODE_OK`` and ``EDGE_OK``,
    respectively.

    See Also
    ========
    FilterAtlas
    FilterMultiInner
    FilterMultiAdjacency
    """

    def __init__(self, d, NODE_OK, EDGE_OK):
        self._atlas = d
        self.NODE_OK = NODE_OK
        self.EDGE_OK = EDGE_OK

    def __len__(self):
        # check whether NODE_OK stores the number of nodes as `length`
        # or the nodes themselves as a set `nodes`. If not, count the nodes.
        if hasattr(self.NODE_OK, "length"):
            return self.NODE_OK.length
        if hasattr(self.NODE_OK, "nodes"):
            return len(self.NODE_OK.nodes & self._atlas.keys())
        return sum(1 for n in self._atlas if self.NODE_OK(n))

    def __iter__(self):
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return (n for n in self.NODE_OK.nodes if n in self._atlas)
        return (n for n in self._atlas if self.NODE_OK(n))

    def __getitem__(self, node):
        if node in self._atlas and self.NODE_OK(node):

            def new_node_ok(nbr):
                return self.NODE_OK(nbr) and self.EDGE_OK(node, nbr)

            return FilterAtlas(self._atlas[node], new_node_ok)
        raise KeyError(f"Key {node} not found")

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({self._atlas!r}, {self.NODE_OK!r}, {self.EDGE_OK!r})"


class FilterMultiInner(FilterAdjacency):  # muliedge_seconddict
    """A read-only Mapping of Mappings with filtering criteria for nodes and edges.

    It is a view into a dict-of-dict-of-dict-of-dict data structure, and it selects nodes
    and edges that meet specific criteria defined by ``NODE_OK`` and ``EDGE_OK``.

    See Also
    ========
    FilterAtlas
    FilterAdjacency
    FilterMultiAdjacency
    """

    def __iter__(self):
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            my_nodes = (n for n in self.NODE_OK.nodes if n in self._atlas)
        else:
            my_nodes = (n for n in self._atlas if self.NODE_OK(n))
        for n in my_nodes:
            some_keys_ok = False
            for key in self._atlas[n]:
                if self.EDGE_OK(n, key):
                    some_keys_ok = True
                    break
            if some_keys_ok is True:
                yield n

    def __getitem__(self, nbr):
        if nbr in self._atlas and self.NODE_OK(nbr):

            def new_node_ok(key):
                return self.EDGE_OK(nbr, key)

            return FilterAtlas(self._atlas[nbr], new_node_ok)
        raise KeyError(f"Key {nbr} not found")


class FilterMultiAdjacency(FilterAdjacency):  # multiedgedict
    """A read-only Mapping of Mappings with filtering criteria
    for nodes and edges.

    It is a view into a dict-of-dict-of-dict-of-dict data structure,
    and it selects nodes and edges that satisfy specific criteria
    defined by ``NODE_OK`` and ``EDGE_OK``, respectively.

    See Also
    ========
    FilterAtlas
    FilterAdjacency
    FilterMultiInner
    """

    def __getitem__(self, node):
        if node in self._atlas and self.NODE_OK(node):

            def edge_ok(nbr, key):
                return self.NODE_OK(nbr) and self.EDGE_OK(node, nbr, key)

            return FilterMultiInner(self._atlas[node], self.NODE_OK, edge_ok)
        raise KeyError(f"Key {node} not found")
<|endoftext|>"""Base class for MultiDiGraph."""

from copy import deepcopy
from functools import cached_property

import networkx as nx
from networkx import convert
from networkx.classes.coreviews import MultiAdjacencyView
from networkx.classes.digraph import DiGraph
from networkx.classes.multigraph import MultiGraph
from networkx.classes.reportviews import (
    DiMultiDegreeView,
    InMultiDegreeView,
    InMultiEdgeView,
    OutMultiDegreeView,
    OutMultiEdgeView,
)
from networkx.exception import NetworkXError

__all__ = ["MultiDiGraph"]


class MultiDiGraph(MultiGraph, DiGraph):
    """A directed graph class that can store multiedges.

    Multiedges are multiple edges between two nodes.  Each edge
    can hold optional data or attributes.

    A MultiDiGraph holds directed edges.  Self loops are allowed.

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

    multigraph_input : bool or None (default None)
        Note: Only used when `incoming_graph_data` is a dict.
        If True, `incoming_graph_data` is assumed to be a
        dict-of-dict-of-dict-of-dict structure keyed by
        node to neighbor to edge keys to edge data for multi-edges.
        A NetworkXError is raised if this is not the case.
        If False, :func:`to_networkx_graph` is used to try to determine
        the dict's graph data structure as either a dict-of-dict-of-dict
        keyed by node to neighbor to edge data, or a dict-of-iterable
        keyed by node to neighbors.
        If None, the treatment for True is tried, but if it fails,
        the treatment for False is tried.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    Graph
    DiGraph
    MultiGraph

    Examples
    --------
    Create an empty graph structure (a "null graph") with no nodes and
    no edges.

    >>> G = nx.MultiDiGraph()

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

    >>> key = G.add_edge(1, 2)

    a list of edges,

    >>> keys = G.add_edges_from([(1, 2), (1, 3)])

    or a collection of edges,

    >>> keys = G.add_edges_from(H.edges)

    If some edges connect nodes not yet in the graph, the nodes
    are added automatically.  If an edge already exists, an additional
    edge is created and stored using a key to identify the edge.
    By default the key is the lowest unused integer.

    >>> keys = G.add_edges_from([(4, 5, dict(route=282)), (4, 5, dict(route=37))])
    >>> G[4]
    AdjacencyView({5: {0: {}, 1: {'route': 282}, 2: {'route': 37}}})

    **Attributes:**

    Each graph, node, and edge can hold key/value attribute pairs
    in an associated attribute dictionary (the keys must be hashable).
    By default these are empty, but can be added or changed using
    add_edge, add_node or direct manipulation of the attribute
    dictionaries named graph, node and edge respectively.

    >>> G = nx.MultiDiGraph(day="Friday")
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

    >>> key = G.add_edge(1, 2, weight=4.7)
    >>> keys = G.add_edges_from([(3, 4), (4, 5)], color="red")
    >>> keys = G.add_edges_from([(1, 2, {"color": "blue"}), (2, 3, {"weight": 8})])
    >>> G[1][2][0]["weight"] = 4.7
    >>> G.edges[1, 2, 0]["weight"] = 4

    Warning: we protect the graph data structure by making `G.edges[1,
    2, 0]` a read-only dict-like structure. However, you can assign to
    attributes in e.g. `G.edges[1, 2, 0]`. Thus, use 2 sets of brackets
    to add/change data attributes: `G.edges[1, 2, 0]['weight'] = 4`
    (for multigraphs the edge key is required: `MG.edges[u, v,
    key][name] = value`).

    **Shortcuts:**

    Many common graph features allow python syntax to speed reporting.

    >>> 1 in G  # check if node in graph
    True
    >>> [n for n in G if n < 3]  # iterate through nodes
    [1, 2]
    >>> len(G)  # number of nodes in graph
    5
    >>> G[1]  # adjacency dict-like view mapping neighbor -> edge key -> edge attributes
    AdjacencyView({2: {0: {'weight': 4}, 1: {'color': 'blue'}}})

    Often the best way to traverse all edges of a graph is via the neighbors.
    The neighbors are available as an adjacency-view `G.adj` object or via
    the method `G.adjacency()`.

    >>> for n, nbrsdict in G.adjacency():
    ...     for nbr, keydict in nbrsdict.items():
    ...         for key, eattr in keydict.items():
    ...             if "weight" in eattr:
    ...                 # Do something useful with the edges
    ...                 pass

    But the edges() method is often more convenient:

    >>> for u, v, keys, weight in G.edges(data="weight", keys=True):
    ...     if weight is not None:
    ...         # Do something useful with the edges
    ...         pass

    **Reporting:**

    Simple graph information is obtained using methods and object-attributes.
    Reporting usually provides views instead of containers to reduce memory
    usage. The views update as the graph is updated similarly to dict-views.
    The objects `nodes`, `edges` and `adj` provide access to data attributes
    via lookup (e.g. `nodes[n]`, `edges[u, v, k]`, `adj[u][v]`) and iteration
    (e.g. `nodes.items()`, `nodes.data('color')`,
    `nodes.data('color', default='blue')` and similarly for `edges`)
    Views exist for `nodes`, `edges`, `neighbors()`/`adj` and `degree`.

    For details on these and other miscellaneous methods, see below.

    **Subclasses (Advanced):**

    The MultiDiGraph class uses a dict-of-dict-of-dict-of-dict structure.
    The outer dict (node_dict) holds adjacency information keyed by node.
    The next dict (adjlist_dict) represents the adjacency information
    and holds edge_key dicts keyed by neighbor. The edge_key dict holds
    each edge_attr dict keyed by edge key. The inner dict
    (edge_attr_dict) represents the edge data and holds edge attribute
    values keyed by attribute names.

    Each of these four dicts in the dict-of-dict-of-dict-of-dict
    structure can be replaced by a user defined dict-like object.
    In general, the dict-like features should be maintained but
    extra features can be added. To replace one of the dicts create
    a new graph class by changing the class(!) variable holding the
    factory for that dict-like structure. The variable names are
    node_dict_factory, node_attr_dict_factory, adjlist_inner_dict_factory,
    adjlist_outer_dict_factory, edge_key_dict_factory, edge_attr_dict_factory
    and graph_attr_dict_factory.

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
        dict which holds multiedge key dicts keyed by neighbor.
        It should require no arguments and return a dict-like object.

    edge_key_dict_factory : function, (default: dict)
        Factory function to be used to create the edge key dict
        which holds edge data keyed by edge key.
        It should require no arguments and return a dict-like object.

    edge_attr_dict_factory : function, (default: dict)
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

    # node_dict_factory = dict    # already assigned in Graph
    # adjlist_outer_dict_factory = dict
    # adjlist_inner_dict_factory = dict
    edge_key_dict_factory = dict
    # edge_attr_dict_factory = dict

    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        """Initialize a graph with edges, name, or graph attributes.

        Parameters
        ----------
        incoming_graph_data : input graph
            Data to initialize graph.  If incoming_graph_data=None (default)
            an empty graph is created.  The data can be an edge list, or any
            NetworkX graph object.  If the corresponding optional Python
            packages are installed the data can also be a 2D NumPy array, a
            SciPy sparse array, or a PyGraphviz graph.

        multigraph_input : bool or None (default None)
            Note: Only used when `incoming_graph_data` is a dict.
            If True, `incoming_graph_data` is assumed to be a
            dict-of-dict-of-dict-of-dict structure keyed by
            node to neighbor to edge keys to edge data for multi-edges.
            A NetworkXError is raised if this is not the case.
            If False, :func:`to_networkx_graph` is used to try to determine
            the dict's graph data structure as either a dict-of-dict-of-dict
            keyed by node to neighbor to edge data, or a dict-of-iterable
            keyed by node to neighbors.
            If None, the treatment for True is tried, but if it fails,
            the treatment for False is tried.

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
        # multigraph_input can be None/True/False. So check "is not False"
        if isinstance(incoming_graph_data, dict) and multigraph_input is not False:
            DiGraph.__init__(self)
            try:
                convert.from_dict_of_dicts(
                    incoming_graph_data, create_using=self, multigraph_input=True
                )
                self.graph.update(attr)
            except Exception as err:
                if multigraph_input is True:
                    raise nx.NetworkXError(
                        f"converting multigraph_input raised:\n{type(err)}: {err}"
                    )
                DiGraph.__init__(self, incoming_graph_data, **attr)
        else:
            DiGraph.__init__(self, incoming_graph_data, **attr)

    @cached_property
    def adj(self):
        """Graph adjacency object holding the neighbors of each node.

        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edgekey-dict.  So `G.adj[3][2][0]['color'] = 'blue'` sets
        the color of the edge `(3, 2, 0)` to `"blue"`.

        Iterating over G.adj behaves like a dict. Useful idioms include
        `for nbr, datadict in G.adj[n].items():`.

        The neighbor information is also provided by subscripting the graph.
        So `for nbr, foovalue in G[node].data('foo', default=1):` works.

        For directed graphs, `G.adj` holds outgoing (successor) info.
        """
        return MultiAdjacencyView(self._succ)

    @cached_property
    def succ(self):
        """Graph adjacency object holding the successors of each node.

        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edgekey-dict.  So `G.adj[3][2][0]['color'] = 'blue'` sets
        the color of the edge `(3, 2, 0)` to `"blue"`.

        Iterating over G.adj behaves like a dict. Useful idioms include
        `for nbr, datadict in G.adj[n].items():`.

        The neighbor information is also provided by subscripting the graph.
        So `for nbr, foovalue in G[node].data('foo', default=1):` works.

        For directed graphs, `G.succ` is identical to `G.adj`.
        """
        return MultiAdjacencyView(self._succ)

    @cached_property
    def pred(self):
        """Graph adjacency object holding the predecessors of each node.

        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edgekey-dict.  So `G.adj[3][2][0]['color'] = 'blue'` sets
        the color of the edge `(3, 2, 0)` to `"blue"`.

        Iterating over G.adj behaves like a dict. Useful idioms include
        `for nbr, datadict in G.adj[n].items():`.
        """
        return MultiAdjacencyView(self._pred)

    def add_edge(self, u_for_edge, v_for_edge, key=None, **attr):
        """Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Edge attributes can be specified with keywords or by directly
        accessing the edge's attribute dictionary. See examples below.

        Parameters
        ----------
        u_for_edge, v_for_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        key : hashable identifier, optional (default=lowest unused integer)
            Used to distinguish multiedges between a pair of nodes.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        The edge key assigned to the edge.

        See Also
        --------
        add_edges_from : add a collection of edges

        Notes
        -----
        To replace/update edge data, use the optional key argument
        to identify a unique edge.  Otherwise a new edge will be created.

        NetworkX algorithms designed for weighted graphs cannot use
        multigraphs directly because it is not clear how to handle
        multiedge weights.  Convert to Graph using edge attribute
        'weight' to enable weighted graph algorithms.

        Default keys are generated using the method `new_edge_key()`.
        This method can be overridden by subclassing the base class and
        providing a custom `new_edge_key()` method.

        Examples
        --------
        The following all add the edge e=(1, 2) to graph G:

        >>> G = nx.MultiDiGraph()
        >>> e = (1, 2)
        >>> key = G.add_edge(1, 2)  # explicit two-node form
        >>> G.add_edge(*e)  # single edge as tuple of two nodes
        1
        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container
        [2]

        Associate data to edges using keywords:

        >>> key = G.add_edge(1, 2, weight=3)
        >>> key = G.add_edge(1, 2, key=0, weight=4)  # update data for key=0
        >>> key = G.add_edge(1, 3, weight=7, capacity=15, length=342.7)

        For non-string attribute keys, use subscript notation.

        >>> ekey = G.add_edge(1, 2)
        >>> G[1][2][0].update({0: 5})
        >>> G.edges[1, 2, 0].update({0: 5})
        """
        u, v = u_for_edge, v_for_edge
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
        if key is None:
            key = self.new_edge_key(u, v)
        if v in self._succ[u]:
            keydict = self._adj[u][v]
            datadict = keydict.get(key, self.edge_attr_dict_factory())
            datadict.update(attr)
            keydict[key] = datadict
        else:
            # selfloops work this way without special treatment
            datadict = self.edge_attr_dict_factory()
            datadict.update(attr)
            keydict = self.edge_key_dict_factory()
            keydict[key] = datadict
            self._succ[u][v] = keydict
            self._pred[v][u] = keydict
        nx._clear_cache(self)
        return key

    def remove_edge(self, u, v, key=None):
        """Remove an edge between u and v.

        Parameters
        ----------
        u, v : nodes
            Remove an edge between nodes u and v.
        key : hashable identifier, optional (default=None)
            Used to distinguish multiple edges between a pair of nodes.
            If None, remove a single edge between u and v. If there are
            multiple edges, removes the last edge added in terms of
            insertion order.

        Raises
        ------
        NetworkXError
            If there is not an edge between u and v, or
            if there is no edge with the specified key.

        See Also
        --------
        remove_edges_from : remove a collection of edges

        Examples
        --------
        >>> G = nx.MultiDiGraph()
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.remove_edge(0, 1)
        >>> e = (1, 2)
        >>> G.remove_edge(*e)  # unpacks e from an edge tuple

        For multiple edges

        >>> G = nx.MultiDiGraph()
        >>> G.add_edges_from([(1, 2), (1, 2), (1, 2)])  # key_list returned
        [0, 1, 2]

        When ``key=None`` (the default), edges are removed in the opposite
        order that they were added:

        >>> G.remove_edge(1, 2)
        >>> G.edges(keys=True)
        OutMultiEdgeView([(1, 2, 0), (1, 2, 1)])

        For edges with keys

        >>> G = nx.MultiDiGraph()
        >>> G.add_edge(1, 2, key="first")
        'first'
        >>> G.add_edge(1, 2, key="second")
        'second'
        >>> G.remove_edge(1, 2, key="first")
        >>> G.edges(keys=True)
        OutMultiEdgeView([(1, 2, 'second')])

        """
        try:
            d = self._adj[u][v]
        except KeyError as err:
            raise NetworkXError(f"The edge {u}-{v} is not in the graph.") from err
        # remove the edge with specified data
        if key is None:
            d.popitem()
        else:
            try:
                del d[key]
            except KeyError as err:
                msg = f"The edge {u}-{v} with key {key} is not in the graph."
                raise NetworkXError(msg) from err
        if len(d) == 0:
            # remove the key entries if last edge
            del self._succ[u][v]
            del self._pred[v][u]
        nx._clear_cache(self)

    @cached_property
    def edges(self):
        """An OutMultiEdgeView of the Graph as G.edges or G.edges().

        edges(self, nbunch=None, data=False, keys=False, default=None)

        The OutMultiEdgeView provides set-like operations on the edge-tuples
        as well as edge attribute lookup. When called, it also provides
        an EdgeDataView object which allows control of access to edge
        attributes (but does not provide set-like operations).
        Hence, ``G.edges[u, v, k]['color']`` provides the value of the color
        attribute for the edge from ``u`` to ``v`` with key ``k`` while
        ``for (u, v, k, c) in G.edges(data='color', default='red', keys=True):``
        iterates through all the edges yielding the color attribute with
        default `'red'` if no color attribute exists.

        Edges are returned as tuples with optional data and keys
        in the order (node, neighbor, key, data). If ``keys=True`` is not
        provided, the tuples will just be (node, neighbor, data), but
        multiple tuples with the same node and neighbor will be
        generated when multiple edges between two nodes exist.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges from these nodes.
        data : string or bool, optional (default=False)
            The edge attribute returned in 3-tuple (u, v, ddict[data]).
            If True, return edge attribute dict in 3-tuple (u, v, ddict).
            If False, return 2-tuple (u, v).
        keys : bool, optional (default=False)
            If True, return edge keys with each edge, creating (u, v, k,
            d) tuples when data is also requested (the default) and (u,
            v, k) tuples when data is not requested.
        default : value, optional (default=None)
            Value used for edges that don't have the requested attribute.
            Only relevant if data is not True or False.

        Returns
        -------
        edges : OutMultiEdgeView
            A view of edge attributes, usually it iterates over (u, v)
            (u, v, k) or (u, v, k, d) tuples of edges, but can also be
            used for attribute lookup as ``edges[u, v, k]['foo']``.

        Notes
        -----
        Nodes in nbunch that are not in the graph will be (quietly) ignored.
        For directed graphs this returns the out-edges.

        Examples
        --------
        >>> G = nx.MultiDiGraph()
        >>> nx.add_path(G, [0, 1, 2])
        >>> key = G.add_edge(2, 3, weight=5)
        >>> key2 = G.add_edge(1, 2)  # second edge between these nodes
        >>> [e for e in G.edges()]
        [(0, 1), (1, 2), (1, 2), (2, 3)]
        >>> list(G.edges(data=True))  # default data is {} (empty dict)
        [(0, 1, {}), (1, 2, {}), (1, 2, {}), (2, 3, {'weight': 5})]
        >>> list(G.edges(data="weight", default=1))
        [(0, 1, 1), (1, 2, 1), (1, 2, 1), (2, 3, 5)]
        >>> list(G.edges(keys=True))  # default keys are integers
        [(0, 1, 0), (1, 2, 0), (1, 2, 1), (2, 3, 0)]
        >>> list(G.edges(data=True, keys=True))
        [(0, 1, 0, {}), (1, 2, 0, {}), (1, 2, 1, {}), (2, 3, 0, {'weight': 5})]
        >>> list(G.edges(data="weight", default=1, keys=True))
        [(0, 1, 0, 1), (1, 2, 0, 1), (1, 2, 1, 1), (2, 3, 0, 5)]
        >>> list(G.edges([0, 2]))
        [(0, 1), (2, 3)]
        >>> list(G.edges(0))
        [(0, 1)]
        >>> list(G.edges(1))
        [(1, 2), (1, 2)]

        See Also
        --------
        in_edges, out_edges
        """
        return OutMultiEdgeView(self)

    # alias out_edges to edges
    @cached_property
    def out_edges(self):
        return OutMultiEdgeView(self)

    out_edges.__doc__ = edges.__doc__

    @cached_property
    def in_edges(self):
        """A view of the in edges of the graph as G.in_edges or G.in_edges().

        in_edges(self, nbunch=None, data=False, keys=False, default=None)

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.
        data : string or bool, optional (default=False)
            The edge attribute returned in 3-tuple (u, v, ddict[data]).
            If True, return edge attribute dict in 3-tuple (u, v, ddict).
            If False, return 2-tuple (u, v).
        keys : bool, optional (default=False)
            If True, return edge keys with each edge, creating 3-tuples
            (u, v, k) or with data, 4-tuples (u, v, k, d).
        default : value, optional (default=None)
            Value used for edges that don't have the requested attribute.
            Only relevant if data is not True or False.

        Returns
        -------
        in_edges : InMultiEdgeView or InMultiEdgeDataView
            A view of edge attributes, usually it iterates over (u, v)
            or (u, v, k) or (u, v, k, d) tuples of edges, but can also be
            used for attribute lookup as `edges[u, v, k]['foo']`.

        See Also
        --------
        edges
        """
        return InMultiEdgeView(self)

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
        DiMultiDegreeView or int
            If multiple nodes are requested (the default), returns a `DiMultiDegreeView`
            mapping nodes to their degree.
            If a single node is requested, returns the degree of the node as an integer.

        See Also
        --------
        out_degree, in_degree

        Examples
        --------
        >>> G = nx.MultiDiGraph()
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.degree(0)  # node 0 with degree 1
        1
        >>> list(G.degree([0, 1, 2]))
        [(0, 1), (1, 2), (2, 2)]
        >>> G.add_edge(0, 1)  # parallel edge
        1
        >>> list(G.degree([0, 1, 2]))  # parallel edges are counted
        [(0, 2), (1, 3), (2, 2)]

        """
        return DiMultiDegreeView(self)

    @cached_property
    def in_degree(self):
        """A DegreeView for (node, in_degree) or in_degree for single node.

        The node in-degree is the number of edges pointing into the node.
        The weighted node degree is the sum of the edge weights for
        edges incident to that node.

        This object provides an iterator for (node, degree) as well as
        lookup for the degree for a single node.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.

        weight : string or None, optional (default=None)
           The edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights adjacent to the node.

        Returns
        -------
        If a single node is requested
        deg : int
            Degree of the node

        OR if multiple nodes are requested
        nd_iter : iterator
            The iterator returns two-tuples of (node, in-degree).

        See Also
        --------
        degree, out_degree

        Examples
        --------
        >>> G = nx.MultiDiGraph()
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.in_degree(0)  # node 0 with degree 0
        0
        >>> list(G.in_degree([0, 1, 2]))
        [(0, 0), (1, 1), (2, 1)]
        >>> G.add_edge(0, 1)  # parallel edge
        1
        >>> list(G.in_degree([0, 1, 2]))  # parallel edges counted
        [(0, 0), (1, 2), (2, 1)]

        """
        return InMultiDegreeView(self)

    @cached_property
    def out_degree(self):
        """Returns an iterator for (node, out-degree) or out-degree for single node.

        out_degree(self, nbunch=None, weight=None)

        The node out-degree is the number of edges pointing out of the node.
        This function returns the out-degree for a single node or an iterator
        for a bunch of nodes or if nothing is passed as argument.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.

        weight : string or None, optional (default=None)
           The edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights.

        Returns
        -------
        If a single node is requested
        deg : int
            Degree of the node

        OR if multiple nodes are requested
        nd_iter : iterator
            The iterator returns two-tuples of (node, out-degree).

        See Also
        --------
        degree, in_degree

        Examples
        --------
        >>> G = nx.MultiDiGraph()
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.out_degree(0)  # node 0 with degree 1
        1
        >>> list(G.out_degree([0, 1, 2]))
        [(0, 1), (1, 1), (2, 1)]
        >>> G.add_edge(0, 1)  # parallel edge
        1
        >>> list(G.out_degree([0, 1, 2]))  # counts parallel edges
        [(0, 2), (1, 1), (2, 1)]

        """
        return OutMultiDegreeView(self)

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return True

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
        G : MultiGraph
            An undirected graph with the same name and nodes and
            with edge (u, v, data) if either (u, v, data) or (v, u, data)
            is in the digraph.  If both edges exist in digraph and
            their edge data is different, only one edge is created
            with an arbitrary choice of which edge data to use.
            You must check and correct for this manually if desired.

        See Also
        --------
        MultiGraph, copy, add_edge, add_edges_from

        Notes
        -----
        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.

        This is in contrast to the similar D=MultiDiGraph(G) which
        returns a shallow copy of the data.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Warning: If you have subclassed MultiDiGraph to use dict-like
        objects in the data structure, those changes do not transfer
        to the MultiGraph created by this method.

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
                (u, v, key, deepcopy(data))
                for u, nbrs in self._adj.items()
                for v, keydict in nbrs.items()
                for key, data in keydict.items()
                if v in self._pred[u] and key in self._pred[u][v]
            )
        else:
            G.add_edges_from(
                (u, v, key, deepcopy(data))
                for u, nbrs in self._adj.items()
                for v, keydict in nbrs.items()
                for key, data in keydict.items()
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
            H.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
            H.add_edges_from(
                (v, u, k, deepcopy(d))
                for u, v, k, d in self.edges(keys=True, data=True)
            )
            return H
        return nx.reverse_view(self)
<|endoftext|>"""
View Classes provide node, edge and degree "views" of a graph.

Views for nodes, edges and degree are provided for all base graph classes.
A view means a read-only object that is quick to create, automatically
updated when the graph changes, and provides basic access like `n in V`,
`for n in V`, `V[n]` and sometimes set operations.

The views are read-only iterable containers that are updated as the
graph is updated. As with dicts, the graph should not be updated
while iterating through the view. Views can be iterated multiple times.

Edge and Node views also allow data attribute lookup.
The resulting attribute dict is writable as `G.edges[3, 4]['color']='red'`
Degree views allow lookup of degree values for single nodes.
Weighted degree is supported with the `weight` argument.

NodeView
========

    `V = G.nodes` (or `V = G.nodes()`) allows `len(V)`, `n in V`, set
    operations e.g. "G.nodes & H.nodes", and `dd = G.nodes[n]`, where
    `dd` is the node data dict. Iteration is over the nodes by default.

NodeDataView
============

    To iterate over (node, data) pairs, use arguments to `G.nodes()`
    to create a DataView e.g. `DV = G.nodes(data='color', default='red')`.
    The DataView iterates as `for n, color in DV` and allows
    `(n, 'red') in DV`. Using `DV = G.nodes(data=True)`, the DataViews
    use the full datadict in writeable form also allowing contain testing as
    `(n, {'color': 'red'}) in VD`. DataViews allow set operations when
    data attributes are hashable.

DegreeView
==========

    `V = G.degree` allows iteration over (node, degree) pairs as well
    as lookup: `deg=V[n]`. There are many flavors of DegreeView
    for In/Out/Directed/Multi. For Directed Graphs, `G.degree`
    counts both in and out going edges. `G.out_degree` and
    `G.in_degree` count only specific directions.
    Weighted degree using edge data attributes is provide via
    `V = G.degree(weight='attr_name')` where any string with the
    attribute name can be used. `weight=None` is the default.
    No set operations are implemented for degrees, use NodeView.

    The argument `nbunch` restricts iteration to nodes in nbunch.
    The DegreeView can still lookup any node even if nbunch is specified.

EdgeView
========

    `V = G.edges` or `V = G.edges()` allows iteration over edges as well as
    `e in V`, set operations and edge data lookup `dd = G.edges[2, 3]`.
    Iteration is over 2-tuples `(u, v)` for Graph/DiGraph. For multigraphs
    edges 3-tuples `(u, v, key)` are the default but 2-tuples can be obtained
    via `V = G.edges(keys=False)`.

    Set operations for directed graphs treat the edges as a set of 2-tuples.
    For undirected graphs, 2-tuples are not a unique representation of edges.
    So long as the set being compared to contains unique representations
    of its edges, the set operations will act as expected. If the other
    set contains both `(0, 1)` and `(1, 0)` however, the result of set
    operations may contain both representations of the same edge.

EdgeDataView
============

    Edge data can be reported using an EdgeDataView typically created
    by calling an EdgeView: `DV = G.edges(data='weight', default=1)`.
    The EdgeDataView allows iteration over edge tuples, membership checking
    but no set operations.

    Iteration depends on `data` and `default` and for multigraph `keys`
    If `data is False` (the default) then iterate over 2-tuples `(u, v)`.
    If `data is True` iterate over 3-tuples `(u, v, datadict)`.
    Otherwise iterate over `(u, v, datadict.get(data, default))`.
    For Multigraphs, if `keys is True`, replace `u, v` with `u, v, key`
    to create 3-tuples and 4-tuples.

    The argument `nbunch` restricts edges to those incident to nodes in nbunch.
"""

from abc import ABC
from collections.abc import Mapping, Set

import networkx as nx

__all__ = [
    "NodeView",
    "NodeDataView",
    "EdgeView",
    "OutEdgeView",
    "InEdgeView",
    "EdgeDataView",
    "OutEdgeDataView",
    "InEdgeDataView",
    "MultiEdgeView",
    "OutMultiEdgeView",
    "InMultiEdgeView",
    "MultiEdgeDataView",
    "OutMultiEdgeDataView",
    "InMultiEdgeDataView",
    "DegreeView",
    "DiDegreeView",
    "InDegreeView",
    "OutDegreeView",
    "MultiDegreeView",
    "DiMultiDegreeView",
    "InMultiDegreeView",
    "OutMultiDegreeView",
]


# NodeViews
class NodeView(Mapping, Set):
    """A NodeView class to act as G.nodes for a NetworkX Graph

    Set operations act on the nodes without considering data.
    Iteration is over nodes. Node data can be looked up like a dict.
    Use NodeDataView to iterate over node data or to specify a data
    attribute for lookup. NodeDataView is created by calling the NodeView.

    Parameters
    ----------
    graph : NetworkX graph-like class

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> NV = G.nodes()
    >>> 2 in NV
    True
    >>> for n in NV:
    ...     print(n)
    0
    1
    2
    >>> assert NV & {1, 2, 3} == {1, 2}

    >>> G.add_node(2, color="blue")
    >>> NV[2]
    {'color': 'blue'}
    >>> G.add_node(8, color="red")
    >>> NDV = G.nodes(data=True)
    >>> (2, NV[2]) in NDV
    True
    >>> for n, dd in NDV:
    ...     print((n, dd.get("color", "aqua")))
    (0, 'aqua')
    (1, 'aqua')
    (2, 'blue')
    (8, 'red')
    >>> NDV[2] == NV[2]
    True

    >>> NVdata = G.nodes(data="color", default="aqua")
    >>> (2, NVdata[2]) in NVdata
    True
    >>> for n, dd in NVdata:
    ...     print((n, dd))
    (0, 'aqua')
    (1, 'aqua')
    (2, 'blue')
    (8, 'red')
    >>> NVdata[2] == NV[2]  # NVdata gets 'color', NV gets datadict
    False
    """

    __slots__ = ("_nodes",)

    def __getstate__(self):
        return {"_nodes": self._nodes}

    def __setstate__(self, state):
        self._nodes = state["_nodes"]

    def __init__(self, graph):
        self._nodes = graph._node

    # Mapping methods
    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, n):
        if isinstance(n, slice):
            raise nx.NetworkXError(
                f"{type(self).__name__} does not support slicing, "
                f"try list(G.nodes)[{n.start}:{n.stop}:{n.step}]"
            )
        return self._nodes[n]

    # Set methods
    def __contains__(self, n):
        return n in self._nodes

    @classmethod
    def _from_iterable(cls, it):
        return set(it)

    # DataView method
    def __call__(self, data=False, default=None):
        if data is False:
            return self
        return NodeDataView(self._nodes, data, default)

    def data(self, data=True, default=None):
        """
        Return a read-only view of node data.

        Parameters
        ----------
        data : bool or node data key, default=True
            If ``data=True`` (the default), return a `NodeDataView` object that
            maps each node to *all* of its attributes. `data` may also be an
            arbitrary key, in which case the `NodeDataView` maps each node to
            the value for the keyed attribute. In this case, if a node does
            not have the `data` attribute, the `default` value is used.
        default : object, default=None
            The value used when a node does not have a specific attribute.

        Returns
        -------
        NodeDataView
            The layout of the returned NodeDataView depends on the value of the
            `data` parameter.

        Notes
        -----
        If ``data=False``, returns a `NodeView` object without data.

        See Also
        --------
        NodeDataView

        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_nodes_from(
        ...     [
        ...         (0, {"color": "red", "weight": 10}),
        ...         (1, {"color": "blue"}),
        ...         (2, {"color": "yellow", "weight": 2}),
        ...     ]
        ... )

        Accessing node data with ``data=True`` (the default) returns a
        NodeDataView mapping each node to all of its attributes:

        >>> G.nodes.data()
        NodeDataView({0: {'color': 'red', 'weight': 10}, 1: {'color': 'blue'}, 2: {'color': 'yellow', 'weight': 2}})

        If `data` represents  a key in the node attribute dict, a NodeDataView mapping
        the nodes to the value for that specific key is returned:

        >>> G.nodes.data("color")
        NodeDataView({0: 'red', 1: 'blue', 2: 'yellow'}, data='color')

        If a specific key is not found in an attribute dict, the value specified
        by `default` is returned:

        >>> G.nodes.data("weight", default=-999)
        NodeDataView({0: 10, 1: -999, 2: 2}, data='weight')

        Note that there is no check that the `data` key is in any of the
        node attribute dictionaries:

        >>> G.nodes.data("height")
        NodeDataView({0: None, 1: None, 2: None}, data='height')
        """
        if data is False:
            return self
        return NodeDataView(self._nodes, data, default)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return f"{self.__class__.__name__}({tuple(self)})"


class NodeDataView(Set):
    """A DataView class for nodes of a NetworkX Graph

    The main use for this class is to iterate through node-data pairs.
    The data can be the entire data-dictionary for each node, or it
    can be a specific attribute (with default) for each node.
    Set operations are enabled with NodeDataView, but don't work in
    cases where the data is not hashable. Use with caution.
    Typically, set operations on nodes use NodeView, not NodeDataView.
    That is, they use `G.nodes` instead of `G.nodes(data='foo')`.

    Parameters
    ==========
    graph : NetworkX graph-like class
    data : bool or string (default=False)
    default : object (default=None)
    """

    __slots__ = ("_nodes", "_data", "_default")

    def __getstate__(self):
        return {"_nodes": self._nodes, "_data": self._data, "_default": self._default}

    def __setstate__(self, state):
        self._nodes = state["_nodes"]
        self._data = state["_data"]
        self._default = state["_default"]

    def __init__(self, nodedict, data=False, default=None):
        self._nodes = nodedict
        self._data = data
        self._default = default

    @classmethod
    def _from_iterable(cls, it):
        try:
            return set(it)
        except TypeError as err:
