            shape = viz.get("shape")
            if shape is not None:
                if shape.startswith("http"):
                    e = Element(
                        f"{{{self.NS_VIZ}}}shape", value="image", uri=str(shape)
                    )
                else:
                    e = Element(f"{{{self.NS_VIZ}}}shape", value=str(shape))
                element.append(e)

            position = viz.get("position")
            if position is not None:
                e = Element(
                    f"{{{self.NS_VIZ}}}position",
                    x=str(position.get("x")),
                    y=str(position.get("y")),
                    z=str(position.get("z")),
                )
                element.append(e)
        return node_data

    def add_parents(self, node_element, node_data):
        parents = node_data.pop("parents", False)
        if parents:
            parents_element = Element("parents")
            for p in parents:
                e = Element("parent")
                e.attrib["for"] = str(p)
                parents_element.append(e)
            node_element.append(parents_element)
        return node_data

    def add_slices(self, node_or_edge_element, node_or_edge_data):
        slices = node_or_edge_data.pop("slices", False)
        if slices:
            slices_element = Element("slices")
            for start, end in slices:
                e = Element("slice", start=str(start), end=str(end))
                slices_element.append(e)
            node_or_edge_element.append(slices_element)
        return node_or_edge_data

    def add_spells(self, node_or_edge_element, node_or_edge_data):
        spells = node_or_edge_data.pop("spells", False)
        if spells:
            spells_element = Element("spells")
            for start, end in spells:
                e = Element("spell")
                if start is not None:
                    e.attrib["start"] = str(start)
                    self.alter_graph_mode_timeformat(start)
                if end is not None:
                    e.attrib["end"] = str(end)
                    self.alter_graph_mode_timeformat(end)
                spells_element.append(e)
            node_or_edge_element.append(spells_element)
        return node_or_edge_data

    def alter_graph_mode_timeformat(self, start_or_end):
        # If 'start' or 'end' appears, alter Graph mode to dynamic and
        # set timeformat
        if self.graph_element.get("mode") == "static":
            if start_or_end is not None:
                if isinstance(start_or_end, str):
                    timeformat = "date"
                elif isinstance(start_or_end, float):
                    timeformat = "double"
                elif isinstance(start_or_end, int):
                    timeformat = "long"
                else:
                    raise nx.NetworkXError(
                        "timeformat should be of the type int, float or str"
                    )
                self.graph_element.set("timeformat", timeformat)
                self.graph_element.set("mode", "dynamic")

    def write(self, fh):
        # Serialize graph G in GEXF to the open fh
        if self.prettyprint:
            self.indent(self.xml)
        document = ElementTree(self.xml)
        document.write(fh, encoding=self.encoding, xml_declaration=True)

    def indent(self, elem, level=0):
        # in-place prettyprint formatter
        i = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


class GEXFReader(GEXF):
    # Class to read GEXF format files
    # use read_gexf() function
    def __init__(self, node_type=None, version="1.2draft"):
        self.construct_types()
        self.node_type = node_type
        # assume simple graph and test for multigraph on read
        self.simple_graph = True
        self.set_version(version)

    def __call__(self, stream):
        self.xml = ElementTree(file=stream)
        g = self.xml.find(f"{{{self.NS_GEXF}}}graph")
        if g is not None:
            return self.make_graph(g)
        # try all the versions
        for version in self.versions:
            self.set_version(version)
            g = self.xml.find(f"{{{self.NS_GEXF}}}graph")
            if g is not None:
                return self.make_graph(g)
        raise nx.NetworkXError("No <graph> element in GEXF file.")

    def make_graph(self, graph_xml):
        # start with empty DiGraph or MultiDiGraph
        edgedefault = graph_xml.get("defaultedgetype", None)
        if edgedefault == "directed":
            G = nx.MultiDiGraph()
        else:
            G = nx.MultiGraph()

        # graph attributes
        graph_name = graph_xml.get("name", "")
        if graph_name != "":
            G.graph["name"] = graph_name
        graph_start = graph_xml.get("start")
        if graph_start is not None:
            G.graph["start"] = graph_start
        graph_end = graph_xml.get("end")
        if graph_end is not None:
            G.graph["end"] = graph_end
        graph_mode = graph_xml.get("mode", "")
        if graph_mode == "dynamic":
            G.graph["mode"] = "dynamic"
        else:
            G.graph["mode"] = "static"

        # timeformat
        self.timeformat = graph_xml.get("timeformat")
        if self.timeformat == "date":
            self.timeformat = "string"

        # node and edge attributes
        attributes_elements = graph_xml.findall(f"{{{self.NS_GEXF}}}attributes")
        # dictionaries to hold attributes and attribute defaults
        node_attr = {}
        node_default = {}
        edge_attr = {}
        edge_default = {}
        for a in attributes_elements:
            attr_class = a.get("class")
            if attr_class == "node":
                na, nd = self.find_gexf_attributes(a)
                node_attr.update(na)
                node_default.update(nd)
                G.graph["node_default"] = node_default
            elif attr_class == "edge":
                ea, ed = self.find_gexf_attributes(a)
                edge_attr.update(ea)
                edge_default.update(ed)
                G.graph["edge_default"] = edge_default
            else:
                raise  # unknown attribute class

        # Hack to handle Gephi0.7beta bug
        # add weight attribute
        ea = {"weight": {"type": "double", "mode": "static", "title": "weight"}}
        ed = {}
        edge_attr.update(ea)
        edge_default.update(ed)
        G.graph["edge_default"] = edge_default

        # add nodes
        nodes_element = graph_xml.find(f"{{{self.NS_GEXF}}}nodes")
        if nodes_element is not None:
            for node_xml in nodes_element.findall(f"{{{self.NS_GEXF}}}node"):
                self.add_node(G, node_xml, node_attr)

        # add edges
        edges_element = graph_xml.find(f"{{{self.NS_GEXF}}}edges")
        if edges_element is not None:
            for edge_xml in edges_element.findall(f"{{{self.NS_GEXF}}}edge"):
                self.add_edge(G, edge_xml, edge_attr)

        # switch to Graph or DiGraph if no parallel edges were found.
        if self.simple_graph:
            if G.is_directed():
                G = nx.DiGraph(G)
            else:
                G = nx.Graph(G)
        return G

    def add_node(self, G, node_xml, node_attr, node_pid=None):
        # add a single node with attributes to the graph

        # get attributes and subattributues for node
        data = self.decode_attr_elements(node_attr, node_xml)
        data = self.add_parents(data, node_xml)  # add any parents
        if self.VERSION == "1.1":
            data = self.add_slices(data, node_xml)  # add slices
        else:
            data = self.add_spells(data, node_xml)  # add spells
        data = self.add_viz(data, node_xml)  # add viz
        data = self.add_start_end(data, node_xml)  # add start/end

        # find the node id and cast it to the appropriate type
        node_id = node_xml.get("id")
        if self.node_type is not None:
            node_id = self.node_type(node_id)

        # every node should have a label
        node_label = node_xml.get("label")
        data["label"] = node_label

        # parent node id
        node_pid = node_xml.get("pid", node_pid)
        if node_pid is not None:
            data["pid"] = node_pid

        # check for subnodes, recursive
        subnodes = node_xml.find(f"{{{self.NS_GEXF}}}nodes")
        if subnodes is not None:
            for node_xml in subnodes.findall(f"{{{self.NS_GEXF}}}node"):
                self.add_node(G, node_xml, node_attr, node_pid=node_id)

        G.add_node(node_id, **data)

    def add_start_end(self, data, xml):
        # start and end times
        ttype = self.timeformat
        node_start = xml.get("start")
        if node_start is not None:
            data["start"] = self.python_type[ttype](node_start)
        node_end = xml.get("end")
        if node_end is not None:
            data["end"] = self.python_type[ttype](node_end)
        return data

    def add_viz(self, data, node_xml):
        # add viz element for node
        viz = {}
        color = node_xml.find(f"{{{self.NS_VIZ}}}color")
        if color is not None:
            if self.VERSION == "1.1":
                viz["color"] = {
                    "r": int(color.get("r")),
                    "g": int(color.get("g")),
                    "b": int(color.get("b")),
                }
            else:
                viz["color"] = {
                    "r": int(color.get("r")),
                    "g": int(color.get("g")),
                    "b": int(color.get("b")),
                    "a": float(color.get("a", 1)),
                }

        size = node_xml.find(f"{{{self.NS_VIZ}}}size")
        if size is not None:
            viz["size"] = float(size.get("value"))

        thickness = node_xml.find(f"{{{self.NS_VIZ}}}thickness")
        if thickness is not None:
            viz["thickness"] = float(thickness.get("value"))

        shape = node_xml.find(f"{{{self.NS_VIZ}}}shape")
        if shape is not None:
            viz["shape"] = shape.get("shape")
            if viz["shape"] == "image":
                viz["shape"] = shape.get("uri")

        position = node_xml.find(f"{{{self.NS_VIZ}}}position")
        if position is not None:
            viz["position"] = {
                "x": float(position.get("x", 0)),
                "y": float(position.get("y", 0)),
                "z": float(position.get("z", 0)),
            }

        if len(viz) > 0:
            data["viz"] = viz
        return data

    def add_parents(self, data, node_xml):
        parents_element = node_xml.find(f"{{{self.NS_GEXF}}}parents")
        if parents_element is not None:
            data["parents"] = []
            for p in parents_element.findall(f"{{{self.NS_GEXF}}}parent"):
                parent = p.get("for")
                data["parents"].append(parent)
        return data

    def add_slices(self, data, node_or_edge_xml):
        slices_element = node_or_edge_xml.find(f"{{{self.NS_GEXF}}}slices")
        if slices_element is not None:
            data["slices"] = []
            for s in slices_element.findall(f"{{{self.NS_GEXF}}}slice"):
                start = s.get("start")
                end = s.get("end")
                data["slices"].append((start, end))
        return data

    def add_spells(self, data, node_or_edge_xml):
        spells_element = node_or_edge_xml.find(f"{{{self.NS_GEXF}}}spells")
        if spells_element is not None:
            data["spells"] = []
            ttype = self.timeformat
            for s in spells_element.findall(f"{{{self.NS_GEXF}}}spell"):
                start = self.python_type[ttype](s.get("start"))
                end = self.python_type[ttype](s.get("end"))
                data["spells"].append((start, end))
        return data

    def add_edge(self, G, edge_element, edge_attr):
        # add an edge to the graph

        # raise error if we find mixed directed and undirected edges
        edge_direction = edge_element.get("type")
        if G.is_directed() and edge_direction == "undirected":
            raise nx.NetworkXError("Undirected edge found in directed graph.")
        if (not G.is_directed()) and edge_direction == "directed":
            raise nx.NetworkXError("Directed edge found in undirected graph.")

        # Get source and target and recast type if required
        source = edge_element.get("source")
        target = edge_element.get("target")
        if self.node_type is not None:
            source = self.node_type(source)
            target = self.node_type(target)

        data = self.decode_attr_elements(edge_attr, edge_element)
        data = self.add_start_end(data, edge_element)

        if self.VERSION == "1.1":
            data = self.add_slices(data, edge_element)  # add slices
        else:
            data = self.add_spells(data, edge_element)  # add spells

        # GEXF stores edge ids as an attribute
        # NetworkX uses them as keys in multigraphs
        # if networkx_key is not specified as an attribute
        edge_id = edge_element.get("id")
        if edge_id is not None:
            data["id"] = edge_id

        # check if there is a 'multigraph_key' and use that as edge_id
        multigraph_key = data.pop("networkx_key", None)
        if multigraph_key is not None:
            edge_id = multigraph_key

        weight = edge_element.get("weight")
        if weight is not None:
            data["weight"] = float(weight)

        edge_label = edge_element.get("label")
        if edge_label is not None:
            data["label"] = edge_label

        if G.has_edge(source, target):
            # seen this edge before - this is a multigraph
            self.simple_graph = False
        G.add_edge(source, target, key=edge_id, **data)
        if edge_direction == "mutual":
            G.add_edge(target, source, key=edge_id, **data)

    def decode_attr_elements(self, gexf_keys, obj_xml):
        # Use the key information to decode the attr XML
        attr = {}
        # look for outer '<attvalues>' element
        attr_element = obj_xml.find(f"{{{self.NS_GEXF}}}attvalues")
        if attr_element is not None:
            # loop over <attvalue> elements
            for a in attr_element.findall(f"{{{self.NS_GEXF}}}attvalue"):
                key = a.get("for")  # for is required
                try:  # should be in our gexf_keys dictionary
                    title = gexf_keys[key]["title"]
                except KeyError as err:
                    raise nx.NetworkXError(f"No attribute defined for={key}.") from err
                atype = gexf_keys[key]["type"]
                value = a.get("value")
                if atype == "boolean":
                    value = self.convert_bool[value]
                else:
                    value = self.python_type[atype](value)
                if gexf_keys[key]["mode"] == "dynamic":
                    # for dynamic graphs use list of three-tuples
                    # [(value1,start1,end1), (value2,start2,end2), etc]
                    ttype = self.timeformat
                    start = self.python_type[ttype](a.get("start"))
                    end = self.python_type[ttype](a.get("end"))
                    if title in attr:
                        attr[title].append((value, start, end))
                    else:
                        attr[title] = [(value, start, end)]
                else:
                    # for static graphs just assign the value
                    attr[title] = value
        return attr

    def find_gexf_attributes(self, attributes_element):
        # Extract all the attributes and defaults
        attrs = {}
        defaults = {}
        mode = attributes_element.get("mode")
        for k in attributes_element.findall(f"{{{self.NS_GEXF}}}attribute"):
            attr_id = k.get("id")
            title = k.get("title")
            atype = k.get("type")
            attrs[attr_id] = {"title": title, "type": atype, "mode": mode}
            # check for the 'default' subelement of key element and add
            default = k.find(f"{{{self.NS_GEXF}}}default")
            if default is not None:
                if atype == "boolean":
                    value = self.convert_bool[default.text]
                else:
                    value = self.python_type[atype](default.text)
                defaults[title] = value
        return attrs, defaults


def relabel_gexf_graph(G):
    """Relabel graph using "label" node keyword for node label.

    Parameters
    ----------
    G : graph
       A NetworkX graph read from GEXF data

    Returns
    -------
    H : graph
      A NetworkX graph with relabeled nodes

    Raises
    ------
    NetworkXError
        If node labels are missing or not unique while relabel=True.

    Notes
    -----
    This function relabels the nodes in a NetworkX graph with the
    "label" attribute.  It also handles relabeling the specific GEXF
    node attributes "parents", and "pid".
    """
    # build mapping of node labels, do some error checking
    try:
        mapping = [(u, G.nodes[u]["label"]) for u in G]
    except KeyError as err:
        raise nx.NetworkXError(
            "Failed to relabel nodes: missing node labels found. Use relabel=False."
        ) from err
    x, y = zip(*mapping)
    if len(set(y)) != len(G):
        raise nx.NetworkXError(
            "Failed to relabel nodes: "
            "duplicate node labels found. "
            "Use relabel=False."
        )
    mapping = dict(mapping)
    H = nx.relabel_nodes(G, mapping)
    # relabel attributes
    for n in G:
        m = mapping[n]
        H.nodes[m]["id"] = n
        H.nodes[m].pop("label")
        if "pid" in H.nodes[m]:
            H.nodes[m]["pid"] = mapping[G.nodes[n]["pid"]]
        if "parents" in H.nodes[m]:
            H.nodes[m]["parents"] = [mapping[p] for p in G.nodes[n]["parents"]]
    return H
<|endoftext|>"""
Read graphs in GML format.

"GML, the Graph Modelling Language, is our proposal for a portable
file format for graphs. GML's key features are portability, simple
syntax, extensibility and flexibility. A GML file consists of a
hierarchical key-value lists. Graphs can be annotated with arbitrary
data structures. The idea for a common file format was born at the
GD'95; this proposal is the outcome of many discussions. GML is the
standard file format in the Graphlet graph editor system. It has been
overtaken and adapted by several other systems for drawing graphs."

GML files are stored using a 7-bit ASCII encoding with any extended
ASCII characters (iso8859-1) appearing as HTML character entities.
You will need to give some thought into how the exported data should
interact with different languages and even different Python versions.
Re-importing from gml is also a concern.

Without specifying a `stringizer`/`destringizer`, the code is capable of
writing `int`/`float`/`str`/`dict`/`list` data as required by the GML
specification.  For writing other data types, and for reading data other
than `str` you need to explicitly supply a `stringizer`/`destringizer`.

For additional documentation on the GML file format, please see the
`GML website <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.

Several example graphs in GML format may be found on Mark Newman's
`Network data page <http://www-personal.umich.edu/~mejn/netdata/>`_.
"""

import html.entities as htmlentitydefs
import re
import warnings
from ast import literal_eval
from collections import defaultdict
from enum import Enum
from io import StringIO
from typing import Any, NamedTuple

import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file

__all__ = ["read_gml", "parse_gml", "generate_gml", "write_gml"]


def escape(text):
    """Use XML character references to escape characters.

    Use XML character references for unprintable or non-ASCII
    characters, double quotes and ampersands in a string
    """

    def fixup(m):
        ch = m.group(0)
        return "&#" + str(ord(ch)) + ";"

    text = re.sub('[^ -~]|[&"]', fixup, text)
    return text if isinstance(text, str) else str(text)


def unescape(text):
    """Replace XML character references with the referenced characters"""

    def fixup(m):
        text = m.group(0)
        if text[1] == "#":
            # Character reference
            if text[2] == "x":
                code = int(text[3:-1], 16)
            else:
                code = int(text[2:-1])
        else:
            # Named entity
            try:
                code = htmlentitydefs.name2codepoint[text[1:-1]]
            except KeyError:
                return text  # leave unchanged
        try:
            return chr(code)
        except (ValueError, OverflowError):
            return text  # leave unchanged

    return re.sub("&(?:[0-9A-Za-z]+|#(?:[0-9]+|x[0-9A-Fa-f]+));", fixup, text)


def literal_destringizer(rep):
    """Convert a Python literal to the value it represents.

    Parameters
    ----------
    rep : string
        A Python literal.

    Returns
    -------
    value : object
        The value of the Python literal.

    Raises
    ------
    ValueError
        If `rep` is not a Python literal.
    """
    if isinstance(rep, str):
        orig_rep = rep
        try:
            return literal_eval(rep)
        except SyntaxError as err:
            raise ValueError(f"{orig_rep!r} is not a valid Python literal") from err
    else:
        raise ValueError(f"{rep!r} is not a string")


@open_file(0, mode="rb")
@nx._dispatchable(graphs=None, returns_graph=True)
def read_gml(path, label="label", destringizer=None):
    """Read graph in GML format from `path`.

    Parameters
    ----------
    path : filename or filehandle
        The filename or filehandle to read from.

    label : string, optional
        If not None, the parsed nodes will be renamed according to node
        attributes indicated by `label`. Default value: 'label'.

    destringizer : callable, optional
        A `destringizer` that recovers values stored as strings in GML. If it
        cannot convert a string to a value, a `ValueError` is raised. Default
        value : None.

    Returns
    -------
    G : NetworkX graph
        The parsed graph.

    Raises
    ------
    NetworkXError
        If the input cannot be parsed.

    See Also
    --------
    write_gml, parse_gml
    literal_destringizer

    Notes
    -----
    GML files are stored using a 7-bit ASCII encoding with any extended
    ASCII characters (iso8859-1) appearing as HTML character entities.
    Without specifying a `stringizer`/`destringizer`, the code is capable of
    writing `int`/`float`/`str`/`dict`/`list` data as required by the GML
    specification.  For writing other data types, and for reading data other
    than `str` you need to explicitly supply a `stringizer`/`destringizer`.

    For additional documentation on the GML file format, please see the
    `GML url <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.

    See the module docstring :mod:`networkx.readwrite.gml` for more details.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_gml(G, "test.gml")

    GML values are interpreted as strings by default:

    >>> H = nx.read_gml("test.gml")
    >>> H.nodes
    NodeView(('0', '1', '2', '3'))

    When a `destringizer` is provided, GML values are converted to the provided type.
    For example, integer nodes can be recovered as shown below:

    >>> J = nx.read_gml("test.gml", destringizer=int)
    >>> J.nodes
    NodeView((0, 1, 2, 3))

    """

    def filter_lines(lines):
        for line in lines:
            try:
                line = line.decode("ascii")
            except UnicodeDecodeError as err:
                raise NetworkXError("input is not ASCII-encoded") from err
            if not isinstance(line, str):
                lines = str(lines)
            if line and line[-1] == "\n":
                line = line[:-1]
            yield line

    G = parse_gml_lines(filter_lines(path), label, destringizer)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_gml(lines, label="label", destringizer=None):
    """Parse GML graph from a string or iterable.

    Parameters
    ----------
    lines : string or iterable of strings
       Data in GML format.

    label : string, optional
        If not None, the parsed nodes will be renamed according to node
        attributes indicated by `label`. Default value: 'label'.

    destringizer : callable, optional
        A `destringizer` that recovers values stored as strings in GML. If it
        cannot convert a string to a value, a `ValueError` is raised. Default
        value : None.

    Returns
    -------
    G : NetworkX graph
        The parsed graph.

    Raises
    ------
    NetworkXError
        If the input cannot be parsed.

    See Also
    --------
    write_gml, read_gml

    Notes
    -----
    This stores nested GML attributes as dictionaries in the NetworkX graph,
    node, and edge attribute structures.

    GML files are stored using a 7-bit ASCII encoding with any extended
    ASCII characters (iso8859-1) appearing as HTML character entities.
    Without specifying a `stringizer`/`destringizer`, the code is capable of
    writing `int`/`float`/`str`/`dict`/`list` data as required by the GML
    specification.  For writing other data types, and for reading data other
    than `str` you need to explicitly supply a `stringizer`/`destringizer`.

    For additional documentation on the GML file format, please see the
    `GML url <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.

    See the module docstring :mod:`networkx.readwrite.gml` for more details.
    """

    def decode_line(line):
        if isinstance(line, bytes):
            try:
                line.decode("ascii")
            except UnicodeDecodeError as err:
                raise NetworkXError("input is not ASCII-encoded") from err
        if not isinstance(line, str):
            line = str(line)
        return line

    def filter_lines(lines):
        if isinstance(lines, str):
            lines = decode_line(lines)
            lines = lines.splitlines()
            yield from lines
        else:
            for line in lines:
                line = decode_line(line)
                if line and line[-1] == "\n":
                    line = line[:-1]
                if line.find("\n") != -1:
                    raise NetworkXError("input line contains newline")
                yield line

    G = parse_gml_lines(filter_lines(lines), label, destringizer)
    return G


class Pattern(Enum):
    """encodes the index of each token-matching pattern in `tokenize`."""

    KEYS = 0
    REALS = 1
    INTS = 2
    STRINGS = 3
    DICT_START = 4
    DICT_END = 5
    COMMENT_WHITESPACE = 6


class Token(NamedTuple):
    category: Pattern
    value: Any
    line: int
    position: int


LIST_START_VALUE = "_networkx_list_start"


def parse_gml_lines(lines, label, destringizer):
    """Parse GML `lines` into a graph."""

    def tokenize():
        patterns = [
            r"[A-Za-z][0-9A-Za-z_]*\b",  # keys
            # reals
            r"[+-]?(?:[0-9]*\.[0-9]+|[0-9]+\.[0-9]*|INF)(?:[Ee][+-]?[0-9]+)?",
            r"[+-]?[0-9]+",  # ints
            r'".*?"',  # strings
            r"\[",  # dict start
            r"\]",  # dict end
            r"#.*$|\s+",  # comments and whitespaces
        ]
        tokens = re.compile("|".join(f"({pattern})" for pattern in patterns))
        lineno = 0
        multilines = []  # entries spread across multiple lines
        for line in lines:
            pos = 0

            # deal with entries spread across multiple lines
            #
            # should we actually have to deal with escaped "s then do it here
            if multilines:
                multilines.append(line.strip())
                if line[-1] == '"':  # closing multiline entry
                    # multiline entries will be joined by space. cannot
                    # reintroduce newlines as this will break the tokenizer
                    line = " ".join(multilines)
                    multilines = []
                else:  # continued multiline entry
                    lineno += 1
                    continue
            else:
                if line.count('"') == 1:  # opening multiline entry
                    if line.strip()[0] != '"' and line.strip()[-1] != '"':
                        # since we expect something like key "value", the " should not be found at ends
                        # otherwise tokenizer will pick up the formatting mistake.
                        multilines = [line.rstrip()]
                        lineno += 1
                        continue

            length = len(line)

            while pos < length:
                match = tokens.match(line, pos)
                if match is None:
                    m = f"cannot tokenize {line[pos:]} at ({lineno + 1}, {pos + 1})"
                    raise NetworkXError(m)
                for i in range(len(patterns)):
                    group = match.group(i + 1)
                    if group is not None:
                        if i == 0:  # keys
                            value = group.rstrip()
                        elif i == 1:  # reals
                            value = float(group)
                        elif i == 2:  # ints
                            value = int(group)
                        else:
                            value = group
                        if i != 6:  # comments and whitespaces
                            yield Token(Pattern(i), value, lineno + 1, pos + 1)
                        pos += len(group)
                        break
            lineno += 1
        yield Token(None, None, lineno + 1, 1)  # EOF

    def unexpected(curr_token, expected):
        category, value, lineno, pos = curr_token
        value = repr(value) if value is not None else "EOF"
        raise NetworkXError(f"expected {expected}, found {value} at ({lineno}, {pos})")

    def consume(curr_token, category, expected):
        if curr_token.category == category:
            return next(tokens)
        unexpected(curr_token, expected)

    def parse_kv(curr_token):
        dct = defaultdict(list)
        while curr_token.category == Pattern.KEYS:
            key = curr_token.value
            curr_token = next(tokens)
            category = curr_token.category
            if category == Pattern.REALS or category == Pattern.INTS:
                value = curr_token.value
                curr_token = next(tokens)
            elif category == Pattern.STRINGS:
                value = unescape(curr_token.value[1:-1])
                if destringizer:
                    try:
                        value = destringizer(value)
                    except ValueError:
                        pass
                # Special handling for empty lists and tuples
                if value == "()":
                    value = ()
                if value == "[]":
                    value = []
                curr_token = next(tokens)
            elif category == Pattern.DICT_START:
                curr_token, value = parse_dict(curr_token)
            else:
                # Allow for string convertible id and label values
                if key in ("id", "label", "source", "target"):
                    try:
                        # String convert the token value
                        value = unescape(str(curr_token.value))
                        if destringizer:
                            try:
                                value = destringizer(value)
                            except ValueError:
                                pass
                        curr_token = next(tokens)
                    except Exception:
                        msg = (
                            "an int, float, string, '[' or string"
                            + " convertible ASCII value for node id or label"
                        )
                        unexpected(curr_token, msg)
                # Special handling for nan and infinity.  Since the gml language
                # defines unquoted strings as keys, the numeric and string branches
                # are skipped and we end up in this special branch, so we need to
                # convert the current token value to a float for NAN and plain INF.
                # +/-INF are handled in the pattern for 'reals' in tokenize().  This
                # allows labels and values to be nan or infinity, but not keys.
                elif curr_token.value in {"NAN", "INF"}:
                    value = float(curr_token.value)
                    curr_token = next(tokens)
                else:  # Otherwise error out
                    unexpected(curr_token, "an int, float, string or '['")
            dct[key].append(value)

        def clean_dict_value(value):
            if not isinstance(value, list):
                return value
            if len(value) == 1:
                return value[0]
            if value[0] == LIST_START_VALUE:
                return value[1:]
            return value

        dct = {key: clean_dict_value(value) for key, value in dct.items()}
        return curr_token, dct

    def parse_dict(curr_token):
        # dict start
        curr_token = consume(curr_token, Pattern.DICT_START, "'['")
        # dict contents
        curr_token, dct = parse_kv(curr_token)
        # dict end
        curr_token = consume(curr_token, Pattern.DICT_END, "']'")
        return curr_token, dct

    def parse_graph():
        curr_token, dct = parse_kv(next(tokens))
        if curr_token.category is not None:  # EOF
            unexpected(curr_token, "EOF")
        if "graph" not in dct:
            raise NetworkXError("input contains no graph")
        graph = dct["graph"]
        if isinstance(graph, list):
            raise NetworkXError("input contains more than one graph")
        return graph

    tokens = tokenize()
    graph = parse_graph()

    directed = graph.pop("directed", False)
    multigraph = graph.pop("multigraph", False)
    if not multigraph:
        G = nx.DiGraph() if directed else nx.Graph()
    else:
        G = nx.MultiDiGraph() if directed else nx.MultiGraph()
    graph_attr = {k: v for k, v in graph.items() if k not in ("node", "edge")}
    G.graph.update(graph_attr)

    def pop_attr(dct, category, attr, i):
        try:
            return dct.pop(attr)
        except KeyError as err:
            raise NetworkXError(f"{category} #{i} has no {attr!r} attribute") from err

    nodes = graph.get("node", [])
    mapping = {}
    node_labels = set()
    for i, node in enumerate(nodes if isinstance(nodes, list) else [nodes]):
        id = pop_attr(node, "node", "id", i)
        if id in G:
            raise NetworkXError(f"node id {id!r} is duplicated")
        if label is not None and label != "id":
            node_label = pop_attr(node, "node", label, i)
            if node_label in node_labels:
                raise NetworkXError(f"node label {node_label!r} is duplicated")
            node_labels.add(node_label)
            mapping[id] = node_label
        G.add_node(id, **node)

    edges = graph.get("edge", [])
    for i, edge in enumerate(edges if isinstance(edges, list) else [edges]):
        source = pop_attr(edge, "edge", "source", i)
        target = pop_attr(edge, "edge", "target", i)
        if source not in G:
            raise NetworkXError(f"edge #{i} has undefined source {source!r}")
        if target not in G:
            raise NetworkXError(f"edge #{i} has undefined target {target!r}")
        if not multigraph:
            if not G.has_edge(source, target):
                G.add_edge(source, target, **edge)
            else:
                arrow = "->" if directed else "--"
                msg = f"edge #{i} ({source!r}{arrow}{target!r}) is duplicated"
                raise nx.NetworkXError(msg)
        else:
            key = edge.pop("key", None)
            if key is not None and G.has_edge(source, target, key):
                arrow = "->" if directed else "--"
                msg = f"edge #{i} ({source!r}{arrow}{target!r}, {key!r})"
                msg2 = 'Hint: If multigraph add "multigraph 1" to file header.'
                raise nx.NetworkXError(msg + " is duplicated\n" + msg2)
            G.add_edge(source, target, key, **edge)

    if label is not None and label != "id":
        G = nx.relabel_nodes(G, mapping)
    return G


def literal_stringizer(value):
    """Convert a `value` to a Python literal in GML representation.

    Parameters
    ----------
    value : object
        The `value` to be converted to GML representation.

    Returns
    -------
    rep : string
        A double-quoted Python literal representing value. Unprintable
        characters are replaced by XML character references.

    Raises
    ------
    ValueError
        If `value` cannot be converted to GML.

    Notes
    -----
    The original value can be recovered using the
    :func:`networkx.readwrite.gml.literal_destringizer` function.
    """

    def stringize(value):
        if isinstance(value, int | bool) or value is None:
            if value is True:  # GML uses 1/0 for boolean values.
                buf.write(str(1))
            elif value is False:
                buf.write(str(0))
            else:
                buf.write(str(value))
        elif isinstance(value, str):
            text = repr(value)
            if text[0] != "u":
                try:
                    value.encode("latin1")
                except UnicodeEncodeError:
                    text = "u" + text
            buf.write(text)
        elif isinstance(value, float | complex | str | bytes):
            buf.write(repr(value))
        elif isinstance(value, list):
            buf.write("[")
            first = True
            for item in value:
                if not first:
                    buf.write(",")
                else:
                    first = False
                stringize(item)
            buf.write("]")
        elif isinstance(value, tuple):
            if len(value) > 1:
                buf.write("(")
                first = True
                for item in value:
                    if not first:
                        buf.write(",")
                    else:
                        first = False
                    stringize(item)
                buf.write(")")
            elif value:
                buf.write("(")
                stringize(value[0])
                buf.write(",)")
            else:
                buf.write("()")
        elif isinstance(value, dict):
            buf.write("{")
            first = True
            for key, value in value.items():
                if not first:
                    buf.write(",")
                else:
                    first = False
                stringize(key)
                buf.write(":")
                stringize(value)
            buf.write("}")
        elif isinstance(value, set):
            buf.write("{")
            first = True
            for item in value:
                if not first:
                    buf.write(",")
                else:
                    first = False
                stringize(item)
            buf.write("}")
        else:
            msg = f"{value!r} cannot be converted into a Python literal"
            raise ValueError(msg)

    buf = StringIO()
    stringize(value)
    return buf.getvalue()


def generate_gml(G, stringizer=None):
    r"""Generate a single entry of the graph `G` in GML format.

    Parameters
    ----------
    G : NetworkX graph
        The graph to be converted to GML.

    stringizer : callable, optional
        A `stringizer` which converts non-int/non-float/non-dict values into
        strings. If it cannot convert a value into a string, it should raise a
        `ValueError` to indicate that. Default value: None.

    Returns
    -------
    lines: generator of strings
        Lines of GML data. Newlines are not appended.

    Raises
    ------
    NetworkXError
        If `stringizer` cannot convert a value into a string, or the value to
        convert is not a string while `stringizer` is None.

    See Also
    --------
    literal_stringizer

    Notes
    -----
    Graph attributes named 'directed', 'multigraph', 'node' or
    'edge', node attributes named 'id' or 'label', edge attributes
    named 'source' or 'target' (or 'key' if `G` is a multigraph)
    are ignored because these attribute names are used to encode the graph
    structure.

    GML files are stored using a 7-bit ASCII encoding with any extended
    ASCII characters (iso8859-1) appearing as HTML character entities.
    Without specifying a `stringizer`/`destringizer`, the code is capable of
    writing `int`/`float`/`str`/`dict`/`list` data as required by the GML
    specification.  For writing other data types, and for reading data other
    than `str` you need to explicitly supply a `stringizer`/`destringizer`.

    For additional documentation on the GML file format, please see the
    `GML url <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.

    See the module docstring :mod:`networkx.readwrite.gml` for more details.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_node("1")
    >>> print("\n".join(nx.generate_gml(G)))
    graph [
      node [
        id 0
        label "1"
      ]
    ]
    >>> G = nx.MultiGraph([("a", "b"), ("a", "b")])
    >>> print("\n".join(nx.generate_gml(G)))
    graph [
      multigraph 1
      node [
        id 0
        label "a"
      ]
      node [
        id 1
        label "b"
      ]
      edge [
        source 0
        target 1
        key 0
      ]
      edge [
        source 0
        target 1
        key 1
      ]
    ]
    """
    valid_keys = re.compile("^[A-Za-z][0-9A-Za-z_]*$")

    def stringize(key, value, ignored_keys, indent, in_list=False):
        if not isinstance(key, str):
            raise NetworkXError(f"{key!r} is not a string")
        if not valid_keys.match(key):
            raise NetworkXError(f"{key!r} is not a valid key")
        if not isinstance(key, str):
            key = str(key)
        if key not in ignored_keys:
            if isinstance(value, int | bool):
                if key == "label":
                    yield indent + key + ' "' + str(value) + '"'
                elif value is True:
                    # python bool is an instance of int
                    yield indent + key + " 1"
                elif value is False:
                    yield indent + key + " 0"
                # GML only supports signed 32-bit integers
                elif value < -(2**31) or value >= 2**31:
                    yield indent + key + ' "' + str(value) + '"'
                else:
                    yield indent + key + " " + str(value)
            elif isinstance(value, float):
                text = repr(value).upper()
                # GML matches INF to keys, so prepend + to INF. Use repr(float(*))
                # instead of string literal to future proof against changes to repr.
                if text == repr(float("inf")).upper():
                    text = "+" + text
                else:
                    # GML requires that a real literal contain a decimal point, but
                    # repr may not output a decimal point when the mantissa is
                    # integral and hence needs fixing.
                    epos = text.rfind("E")
                    if epos != -1 and text.find(".", 0, epos) == -1:
                        text = text[:epos] + "." + text[epos:]
                if key == "label":
                    yield indent + key + ' "' + text + '"'
                else:
                    yield indent + key + " " + text
            elif isinstance(value, dict):
                yield indent + key + " ["
                next_indent = indent + "  "
                for key, value in value.items():
                    yield from stringize(key, value, (), next_indent)
                yield indent + "]"
            elif isinstance(value, tuple) and key == "label":
                yield indent + key + f" \"({','.join(repr(v) for v in value)})\""
            elif isinstance(value, list | tuple) and key != "label" and not in_list:
                if len(value) == 0:
                    yield indent + key + " " + f'"{value!r}"'
                if len(value) == 1:
                    yield indent + key + " " + f'"{LIST_START_VALUE}"'
                for val in value:
                    yield from stringize(key, val, (), indent, True)
            else:
                if stringizer:
                    try:
                        value = stringizer(value)
                    except ValueError as err:
                        raise NetworkXError(
                            f"{value!r} cannot be converted into a string"
                        ) from err
                if not isinstance(value, str):
                    raise NetworkXError(f"{value!r} is not a string")
                yield indent + key + ' "' + escape(value) + '"'

    multigraph = G.is_multigraph()
    yield "graph ["

    # Output graph attributes
    if G.is_directed():
        yield "  directed 1"
    if multigraph:
        yield "  multigraph 1"
    ignored_keys = {"directed", "multigraph", "node", "edge"}
    for attr, value in G.graph.items():
        yield from stringize(attr, value, ignored_keys, "  ")

    # Output node data
    node_id = dict(zip(G, range(len(G))))
    ignored_keys = {"id", "label"}
    for node, attrs in G.nodes.items():
        yield "  node ["
        yield "    id " + str(node_id[node])
        yield from stringize("label", node, (), "    ")
        for attr, value in attrs.items():
            yield from stringize(attr, value, ignored_keys, "    ")
        yield "  ]"

    # Output edge data
    ignored_keys = {"source", "target"}
    kwargs = {"data": True}
    if multigraph:
        ignored_keys.add("key")
        kwargs["keys"] = True
    for e in G.edges(**kwargs):
        yield "  edge ["
        yield "    source " + str(node_id[e[0]])
        yield "    target " + str(node_id[e[1]])
        if multigraph:
            yield from stringize("key", e[2], (), "    ")
        for attr, value in e[-1].items():
            yield from stringize(attr, value, ignored_keys, "    ")
        yield "  ]"
    yield "]"


@open_file(1, mode="wb")
def write_gml(G, path, stringizer=None):
    """Write a graph `G` in GML format to the file or file handle `path`.

    Parameters
    ----------
    G : NetworkX graph
        The graph to be converted to GML.

    path : filename or filehandle
        The filename or filehandle to write. Files whose names end with .gz or
        .bz2 will be compressed.

    stringizer : callable, optional
        A `stringizer` which converts non-int/non-float/non-dict values into
        strings. If it cannot convert a value into a string, it should raise a
        `ValueError` to indicate that. Default value: None.

    Raises
    ------
    NetworkXError
        If `stringizer` cannot convert a value into a string, or the value to
        convert is not a string while `stringizer` is None.

    See Also
    --------
    read_gml, generate_gml
    literal_stringizer

    Notes
    -----
    Graph attributes named 'directed', 'multigraph', 'node' or
    'edge', node attributes named 'id' or 'label', edge attributes
    named 'source' or 'target' (or 'key' if `G` is a multigraph)
    are ignored because these attribute names are used to encode the graph
    structure.

    GML files are stored using a 7-bit ASCII encoding with any extended
    ASCII characters (iso8859-1) appearing as HTML character entities.
    Without specifying a `stringizer`/`destringizer`, the code is capable of
    writing `int`/`float`/`str`/`dict`/`list` data as required by the GML
    specification.  For writing other data types, and for reading data other
    than `str` you need to explicitly supply a `stringizer`/`destringizer`.

    Note that while we allow non-standard GML to be read from a file, we make
    sure to write GML format. In particular, underscores are not allowed in
    attribute names.
    For additional documentation on the GML file format, please see the
    `GML url <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.

    See the module docstring :mod:`networkx.readwrite.gml` for more details.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_gml(G, "test.gml")

    Filenames ending in .gz or .bz2 will be compressed.

    >>> nx.write_gml(G, "test.gml.gz")
    """
    for line in generate_gml(G, stringizer):
        path.write((line + "\n").encode("ascii"))
<|endoftext|>"""
**********
Edge Lists
**********
Read and write NetworkX graphs as edge lists.

The multi-line adjacency list format is useful for graphs with nodes
that can be meaningfully represented as strings.  With the edgelist
format simple edge data can be stored but node or graph data is not.
There is no way of representing isolated nodes unless the node has a
self-loop edge.

Format
------
You can read or write three formats of edge lists with these functions.

Node pairs with no data::

 1 2

Python dictionary as data::

 1 2 {'weight':7, 'color':'green'}

Arbitrary data::

 1 2 7 green
"""

__all__ = [
    "generate_edgelist",
    "write_edgelist",
    "parse_edgelist",
    "read_edgelist",
    "read_weighted_edgelist",
    "write_weighted_edgelist",
]

import networkx as nx
from networkx.utils import open_file


def generate_edgelist(G, delimiter=" ", data=True):
    """Generate a single line of the graph G in edge list format.

    Parameters
    ----------
    G : NetworkX graph

    delimiter : string, optional
       Separator for node labels

    data : bool or list of keys
       If False generate no edge data.  If True use a dictionary
       representation of edge data.  If a list of keys use a list of data
       values corresponding to the keys.

    Returns
    -------
    lines : string
        Lines of data in adjlist format.

    Examples
    --------
    >>> G = nx.lollipop_graph(4, 3)
    >>> G[1][2]["weight"] = 3
    >>> G[3][4]["capacity"] = 12
    >>> for line in nx.generate_edgelist(G, data=False):
    ...     print(line)
    0 1
    0 2
    0 3
    1 2
    1 3
    2 3
    3 4
    4 5
    5 6

    >>> for line in nx.generate_edgelist(G):
    ...     print(line)
    0 1 {}
    0 2 {}
    0 3 {}
    1 2 {'weight': 3}
    1 3 {}
    2 3 {}
    3 4 {'capacity': 12}
    4 5 {}
    5 6 {}

    >>> for line in nx.generate_edgelist(G, data=["weight"]):
    ...     print(line)
    0 1
    0 2
    0 3
    1 2 3
    1 3
    2 3
    3 4
    4 5
    5 6

    See Also
    --------
    write_adjlist, read_adjlist
    """
    if data is True:
        for u, v, d in G.edges(data=True):
            e = u, v, dict(d)
            yield delimiter.join(map(str, e))
    elif data is False:
        for u, v in G.edges(data=False):
            e = u, v
            yield delimiter.join(map(str, e))
    else:
        for u, v, d in G.edges(data=True):
            e = [u, v]
            try:
                e.extend(d[k] for k in data)
            except KeyError:
                pass  # missing data for this edge, should warn?
            yield delimiter.join(map(str, e))


@open_file(1, mode="wb")
def write_edgelist(G, path, comments="#", delimiter=" ", data=True, encoding="utf-8"):
    """Write graph as a list of edges.

    Parameters
    ----------
    G : graph
       A NetworkX graph
    path : file or string
       File or filename to write. If a file is provided, it must be
       opened in 'wb' mode. Filenames ending in .gz or .bz2 will be compressed.
    comments : string, optional
       The character used to indicate the start of a comment
    delimiter : string, optional
       The string used to separate values.  The default is whitespace.
    data : bool or list, optional
       If False write no edge data.
       If True write a string representation of the edge data dictionary..
       If a list (or other iterable) is provided, write the  keys specified
       in the list.
    encoding: string, optional
       Specify which encoding to use when writing file.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_edgelist(G, "test.edgelist")
    >>> G = nx.path_graph(4)
    >>> fh = open("test.edgelist", "wb")
    >>> nx.write_edgelist(G, fh)
    >>> nx.write_edgelist(G, "test.edgelist.gz")
    >>> nx.write_edgelist(G, "test.edgelist.gz", data=False)

    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=7, color="red")
    >>> nx.write_edgelist(G, "test.edgelist", data=False)
    >>> nx.write_edgelist(G, "test.edgelist", data=["color"])
    >>> nx.write_edgelist(G, "test.edgelist", data=["color", "weight"])

    See Also
    --------
    read_edgelist
    write_weighted_edgelist
    """

    for line in generate_edgelist(G, delimiter, data):
        line += "\n"
        path.write(line.encode(encoding))


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_edgelist(
    lines, comments="#", delimiter=None, create_using=None, nodetype=None, data=True
):
    """Parse lines of an edge list representation of a graph.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in edgelist format
    comments : string, optional
       Marker for comment lines. Default is `'#'`. To specify that no character
       should be treated as a comment, use ``comments=None``.
    delimiter : string, optional
       Separator for node labels. Default is `None`, meaning any whitespace.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.
    nodetype : Python type, optional
       Convert nodes to this type. Default is `None`, meaning no conversion is
       performed.
    data : bool or list of (label,type) tuples
       If `False` generate no edge data or if `True` use a dictionary
       representation of edge data or a list tuples specifying dictionary
       key names and types for edge data.

    Returns
    -------
    G: NetworkX Graph
        The graph corresponding to lines

    Examples
    --------
    Edgelist with no data:

    >>> lines = ["1 2", "2 3", "3 4"]
    >>> G = nx.parse_edgelist(lines, nodetype=int)
    >>> list(G)
    [1, 2, 3, 4]
    >>> list(G.edges())
    [(1, 2), (2, 3), (3, 4)]

    Edgelist with data in Python dictionary representation:

    >>> lines = ["1 2 {'weight': 3}", "2 3 {'weight': 27}", "3 4 {'weight': 3.0}"]
    >>> G = nx.parse_edgelist(lines, nodetype=int)
    >>> list(G)
    [1, 2, 3, 4]
    >>> list(G.edges(data=True))
    [(1, 2, {'weight': 3}), (2, 3, {'weight': 27}), (3, 4, {'weight': 3.0})]

    Edgelist with data in a list:

    >>> lines = ["1 2 3", "2 3 27", "3 4 3.0"]
    >>> G = nx.parse_edgelist(lines, nodetype=int, data=(("weight", float),))
    >>> list(G)
    [1, 2, 3, 4]
    >>> list(G.edges(data=True))
    [(1, 2, {'weight': 3.0}), (2, 3, {'weight': 27.0}), (3, 4, {'weight': 3.0})]

    See Also
    --------
    read_weighted_edgelist
    """
    from ast import literal_eval

    G = nx.empty_graph(0, create_using)
    for line in lines:
        if comments is not None:
            p = line.find(comments)
            if p >= 0:
                line = line[:p]
            if not line:
                continue
        # split line, should have 2 or more
        s = line.rstrip("\n").split(delimiter)
        if len(s) < 2:
            continue
        u = s.pop(0)
        v = s.pop(0)
        d = s
        if nodetype is not None:
            try:
                u = nodetype(u)
                v = nodetype(v)
            except Exception as err:
                raise TypeError(
                    f"Failed to convert nodes {u},{v} to type {nodetype}."
                ) from err

        if len(d) == 0 or data is False:
            # no data or data type specified
            edgedata = {}
        elif data is True:
            # no edge types specified
            try:  # try to evaluate as dictionary
                if delimiter == ",":
                    edgedata_str = ",".join(d)
                else:
                    edgedata_str = " ".join(d)
                edgedata = dict(literal_eval(edgedata_str.strip()))
            except Exception as err:
                raise TypeError(
                    f"Failed to convert edge data ({d}) to dictionary."
                ) from err
        else:
            # convert edge data to dictionary with specified keys and type
            if len(d) != len(data):
                raise IndexError(
                    f"Edge data {d} and data_keys {data} are not the same length"
                )
            edgedata = {}
            for (edge_key, edge_type), edge_value in zip(data, d):
                try:
                    edge_value = edge_type(edge_value)
                except Exception as err:
                    raise TypeError(
                        f"Failed to convert {edge_key} data {edge_value} "
                        f"to type {edge_type}."
                    ) from err
                edgedata.update({edge_key: edge_value})
        G.add_edge(u, v, **edgedata)
    return G


@open_file(0, mode="rb")
@nx._dispatchable(graphs=None, returns_graph=True)
def read_edgelist(
    path,
    comments="#",
    delimiter=None,
    create_using=None,
    nodetype=None,
    data=True,
    edgetype=None,
    encoding="utf-8",
):
    """Read a graph from a list of edges.

    Parameters
    ----------
    path : file or string
       File or filename to read. If a file is provided, it must be
       opened in 'rb' mode.
       Filenames ending in .gz or .bz2 will be uncompressed.
    comments : string, optional
       The character used to indicate the start of a comment. To specify that
       no character should be treated as a comment, use ``comments=None``.
    delimiter : string, optional
       The string used to separate values.  The default is whitespace.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.
    nodetype : int, float, str, Python type, optional
       Convert node data from strings to specified type
    data : bool or list of (label,type) tuples
       Tuples specifying dictionary key names and types for edge data
    edgetype : int, float, str, Python type, optional OBSOLETE
       Convert edge data from strings to specified type and use as 'weight'
    encoding: string, optional
       Specify which encoding to use when reading file.

    Returns
    -------
    G : graph
       A networkx Graph or other type specified with create_using

    Examples
    --------
    >>> nx.write_edgelist(nx.path_graph(4), "test.edgelist")
    >>> G = nx.read_edgelist("test.edgelist")

    >>> fh = open("test.edgelist", "rb")
    >>> G = nx.read_edgelist(fh)
    >>> fh.close()

    >>> G = nx.read_edgelist("test.edgelist", nodetype=int)
    >>> G = nx.read_edgelist("test.edgelist", create_using=nx.DiGraph)

    Edgelist with data in a list:

    >>> textline = "1 2 3"
    >>> fh = open("test.edgelist", "w")
    >>> d = fh.write(textline)
    >>> fh.close()
    >>> G = nx.read_edgelist("test.edgelist", nodetype=int, data=(("weight", float),))
    >>> list(G)
    [1, 2]
    >>> list(G.edges(data=True))
    [(1, 2, {'weight': 3.0})]

    See parse_edgelist() for more examples of formatting.

    See Also
    --------
    parse_edgelist
    write_edgelist

    Notes
    -----
    Since nodes must be hashable, the function nodetype must return hashable
    types (e.g. int, float, str, frozenset - or tuples of those, etc.)
    """
    lines = (line if isinstance(line, str) else line.decode(encoding) for line in path)
    return parse_edgelist(
        lines,
        comments=comments,
        delimiter=delimiter,
        create_using=create_using,
        nodetype=nodetype,
        data=data,
    )


def write_weighted_edgelist(G, path, comments="#", delimiter=" ", encoding="utf-8"):
    """Write graph G as a list of edges with numeric weights.

    Parameters
    ----------
    G : graph
       A NetworkX graph
    path : file or string
       File or filename to write. If a file is provided, it must be
       opened in 'wb' mode.
       Filenames ending in .gz or .bz2 will be compressed.
    comments : string, optional
       The character used to indicate the start of a comment
    delimiter : string, optional
       The string used to separate values.  The default is whitespace.
    encoding: string, optional
       Specify which encoding to use when writing file.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=7)
    >>> nx.write_weighted_edgelist(G, "test.weighted.edgelist")

    See Also
    --------
    read_edgelist
    write_edgelist
    read_weighted_edgelist
    """
    write_edgelist(
        G,
        path,
        comments=comments,
        delimiter=delimiter,
        data=("weight",),
        encoding=encoding,
    )


@nx._dispatchable(graphs=None, returns_graph=True)
def read_weighted_edgelist(
    path,
    comments="#",
    delimiter=None,
    create_using=None,
    nodetype=None,
    encoding="utf-8",
):
    """Read a graph as list of edges with numeric weights.

    Parameters
    ----------
    path : file or string
       File or filename to read. If a file is provided, it must be
       opened in 'rb' mode.
       Filenames ending in .gz or .bz2 will be uncompressed.
    comments : string, optional
       The character used to indicate the start of a comment.
    delimiter : string, optional
       The string used to separate values.  The default is whitespace.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.
    nodetype : int, float, str, Python type, optional
       Convert node data from strings to specified type
    encoding: string, optional
       Specify which encoding to use when reading file.

    Returns
    -------
    G : graph
       A networkx Graph or other type specified with create_using

    Notes
    -----
    Since nodes must be hashable, the function nodetype must return hashable
    types (e.g. int, float, str, frozenset - or tuples of those, etc.)

    Example edgelist file format.

    With numeric edge data::

     # read with
     # >>> G=nx.read_weighted_edgelist(fh)
     # source target data
     a b 1
     a c 3.14159
     d e 42

    See Also
    --------
    write_weighted_edgelist
    """
    return read_edgelist(
        path,
        comments=comments,
        delimiter=delimiter,
        create_using=create_using,
        nodetype=nodetype,
        data=(("weight", float),),
        encoding=encoding,
    )
<|endoftext|># Original author: D. Eppstein, UC Irvine, August 12, 2003.
# The original code at http://www.ics.uci.edu/~eppstein/PADS/ is public domain.
"""Functions for reading and writing graphs in the *graph6* format.

The *graph6* file format is suitable for small graphs or large dense
graphs. For large sparse graphs, use the *sparse6* format.

For more information, see the `graph6`_ homepage.

.. _graph6: http://users.cecs.anu.edu.au/~bdm/data/formats.html

"""

from itertools import islice

import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for, open_file

__all__ = ["from_graph6_bytes", "read_graph6", "to_graph6_bytes", "write_graph6"]


def _generate_graph6_bytes(G, nodes, header):
    """Yield bytes in the graph6 encoding of a graph.

    `G` is an undirected simple graph. `nodes` is the list of nodes for
    which the node-induced subgraph will be encoded; if `nodes` is the
    list of all nodes in the graph, the entire graph will be
    encoded. `header` is a Boolean that specifies whether to generate
    the header ``b'>>graph6<<'`` before the remaining data.

    This function generates `bytes` objects in the following order:

    1. the header (if requested),
    2. the encoding of the number of nodes,
    3. each character, one-at-a-time, in the encoding of the requested
       node-induced subgraph,
    4. a newline character.

    This function raises :exc:`ValueError` if the graph is too large for
    the graph6 format (that is, greater than ``2 ** 36`` nodes).

    """
    n = len(G)
    if n >= 2**36:
        raise ValueError(
            "graph6 is only defined if number of nodes is less than 2 ** 36"
        )
    if header:
        yield b">>graph6<<"
    for d in n_to_data(n):
        yield str.encode(chr(d + 63))
    # This generates the same as `(v in G[u] for u, v in combinations(G, 2))`,
    # but in "column-major" order instead of "row-major" order.
    bits = (nodes[j] in G[nodes[i]] for j in range(1, n) for i in range(j))
    chunk = list(islice(bits, 6))
    while chunk:
        d = sum(b << 5 - i for i, b in enumerate(chunk))
        yield str.encode(chr(d + 63))
        chunk = list(islice(bits, 6))
    yield b"\n"


@nx._dispatchable(graphs=None, returns_graph=True)
def from_graph6_bytes(bytes_in):
    """Read a simple undirected graph in graph6 format from bytes.

    Parameters
    ----------
    bytes_in : bytes
       Data in graph6 format, without a trailing newline.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If bytes_in is unable to be parsed in graph6 format

    ValueError
        If any character ``c`` in bytes_in does not satisfy
        ``63 <= ord(c) < 127``.

    Examples
    --------
    >>> G = nx.from_graph6_bytes(b"A_")
    >>> sorted(G.edges())
    [(0, 1)]

    See Also
    --------
    read_graph6, write_graph6

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """

    def bits():
        """Returns sequence of individual bits from 6-bit-per-value
        list of data values."""
        for d in data:
            for i in [5, 4, 3, 2, 1, 0]:
                yield (d >> i) & 1

    if bytes_in.startswith(b">>graph6<<"):
        bytes_in = bytes_in[10:]

    data = [c - 63 for c in bytes_in]
    if any(c > 63 for c in data):
        raise ValueError("each input character must be in range(63, 127)")

    n, data = data_to_n(data)
    nd = (n * (n - 1) // 2 + 5) // 6
    if len(data) != nd:
        raise NetworkXError(
            f"Expected {n * (n - 1) // 2} bits but got {len(data) * 6} in graph6"
        )

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), b in zip(((i, j) for j in range(1, n) for i in range(j)), bits()):
        if b:
            G.add_edge(i, j)

    return G


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def to_graph6_bytes(G, nodes=None, header=True):
    """Convert a simple undirected graph to bytes in graph6 format.

    Parameters
    ----------
    G : Graph (undirected)

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by ``G.nodes()`` is used.

    header: bool
       If True add '>>graph6<<' bytes to head of data.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    ValueError
        If the graph has at least ``2 ** 36`` nodes; the graph6 format
        is only defined for graphs of order less than ``2 ** 36``.

    Examples
    --------
    >>> nx.to_graph6_bytes(nx.path_graph(2))
    b'>>graph6<<A_\\n'

    See Also
    --------
    from_graph6_bytes, read_graph6, write_graph6_bytes

    Notes
    -----
    The returned bytes end with a newline character.

    The format does not support edge or node labels, parallel edges or
    self loops. If self loops are present they are silently ignored.

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    if nodes is not None:
        G = G.subgraph(nodes)
    H = nx.convert_node_labels_to_integers(G)
    nodes = sorted(H.nodes())
    return b"".join(_generate_graph6_bytes(H, nodes, header))


@open_file(0, mode="rb")
@nx._dispatchable(graphs=None, returns_graph=True)
def read_graph6(path):
    """Read simple undirected graphs in graph6 format from path.

    Parameters
    ----------
    path : file or string
       File or filename to write.

    Returns
    -------
    G : Graph or list of Graphs
       If the file contains multiple lines then a list of graphs is returned

    Raises
    ------
    NetworkXError
        If the string is unable to be parsed in graph6 format

    Examples
    --------
    You can read a graph6 file by giving the path to the file::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False) as f:
        ...     _ = f.write(b">>graph6<<A_\\n")
        ...     _ = f.seek(0)
        ...     G = nx.read_graph6(f.name)
        >>> list(G.edges())
        [(0, 1)]

    You can also read a graph6 file by giving an open file-like object::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     _ = f.write(b">>graph6<<A_\\n")
        ...     _ = f.seek(0)
        ...     G = nx.read_graph6(f)
        >>> list(G.edges())
        [(0, 1)]

    See Also
    --------
    from_graph6_bytes, write_graph6

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    glist = []
    for line in path:
        line = line.strip()
        if not len(line):
            continue
        glist.append(from_graph6_bytes(line))
    if len(glist) == 1:
        return glist[0]
    else:
        return glist


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@open_file(1, mode="wb")
def write_graph6(G, path, nodes=None, header=True):
    """Write a simple undirected graph to a path in graph6 format.

    Parameters
    ----------
    G : Graph (undirected)

    path : str
       The path naming the file to which to write the graph.

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by ``G.nodes()`` is used.

    header: bool
       If True add '>>graph6<<' string to head of data

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    ValueError
        If the graph has at least ``2 ** 36`` nodes; the graph6 format
        is only defined for graphs of order less than ``2 ** 36``.

    Examples
    --------
    You can write a graph6 file by giving the path to a file::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False) as f:
        ...     nx.write_graph6(nx.path_graph(2), f.name)
        ...     _ = f.seek(0)
        ...     print(f.read())
        b'>>graph6<<A_\\n'

    See Also
    --------
    from_graph6_bytes, read_graph6

    Notes
    -----
    The function writes a newline character after writing the encoding
    of the graph.

    The format does not support edge or node labels, parallel edges or
    self loops.  If self loops are present they are silently ignored.

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    return write_graph6_file(G, path, nodes=nodes, header=header)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def write_graph6_file(G, f, nodes=None, header=True):
    """Write a simple undirected graph to a file-like object in graph6 format.

    Parameters
    ----------
    G : Graph (undirected)

    f : file-like object
       The file to write.

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by ``G.nodes()`` is used.

    header: bool
       If True add '>>graph6<<' string to head of data

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    ValueError
        If the graph has at least ``2 ** 36`` nodes; the graph6 format
        is only defined for graphs of order less than ``2 ** 36``.

    Examples
    --------
    You can write a graph6 file by giving an open file-like object::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     nx.write_graph6(nx.path_graph(2), f)
        ...     _ = f.seek(0)
        ...     print(f.read())
        b'>>graph6<<A_\\n'

    See Also
    --------
    from_graph6_bytes, read_graph6

    Notes
    -----
    The function writes a newline character after writing the encoding
    of the graph.

    The format does not support edge or node labels, parallel edges or
    self loops.  If self loops are present they are silently ignored.

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    if nodes is not None:
        G = G.subgraph(nodes)
    H = nx.convert_node_labels_to_integers(G)
    nodes = sorted(H.nodes())
    for b in _generate_graph6_bytes(H, nodes, header):
        f.write(b)


def data_to_n(data):
    """Read initial one-, four- or eight-unit value from graph6
    integer sequence.

    Return (value, rest of seq.)"""
    if data[0] <= 62:
        return data[0], data[1:]
    if data[1] <= 62:
        return (data[1] << 12) + (data[2] << 6) + data[3], data[4:]
    return (
        (data[2] << 30)
        + (data[3] << 24)
        + (data[4] << 18)
        + (data[5] << 12)
        + (data[6] << 6)
        + data[7],
        data[8:],
    )


def n_to_data(n):
    """Convert an integer to one-, four- or eight-unit graph6 sequence.

    This function is undefined if `n` is not in ``range(2 ** 36)``.

    """
    if n <= 62:
        return [n]
    elif n <= 258047:
        return [63, (n >> 12) & 0x3F, (n >> 6) & 0x3F, n & 0x3F]
    else:  # if n <= 68719476735:
        return [
            63,
            63,
            (n >> 30) & 0x3F,
            (n >> 24) & 0x3F,
            (n >> 18) & 0x3F,
            (n >> 12) & 0x3F,
            (n >> 6) & 0x3F,
            n & 0x3F,
        ]
<|endoftext|>"""
A package for reading and writing graphs in various formats.

"""

from networkx.readwrite.adjlist import *
from networkx.readwrite.multiline_adjlist import *
from networkx.readwrite.edgelist import *
from networkx.readwrite.pajek import *
from networkx.readwrite.leda import *
from networkx.readwrite.sparse6 import *
from networkx.readwrite.graph6 import *
from networkx.readwrite.gml import *
from networkx.readwrite.graphml import *
from networkx.readwrite.gexf import *
from networkx.readwrite.json_graph import *
from networkx.readwrite.text import *
<|endoftext|>"""
This module provides the following: read and write of p2g format
used in metabolic pathway studies.

See https://web.archive.org/web/20080626113807/http://www.cs.purdue.edu/homes/koyuturk/pathway/ for a description.

The summary is included here:

A file that describes a uniquely labeled graph (with extension ".gr")
format looks like the following:


name
3 4
a
1 2
b

c
0 2

"name" is simply a description of what the graph corresponds to. The
second line displays the number of nodes and number of edges,
respectively. This sample graph contains three nodes labeled "a", "b",
and "c". The rest of the graph contains two lines for each node. The
first line for a node contains the node label. After the declaration
of the node label, the out-edges of that node in the graph are
provided. For instance, "a" is linked to nodes 1 and 2, which are
labeled "b" and "c", while the node labeled "b" has no outgoing
edges. Observe that node labeled "c" has an outgoing edge to
itself. Indeed, self-loops are allowed. Node index starts from 0.

"""

import networkx as nx
from networkx.utils import open_file


@open_file(1, mode="w")
def write_p2g(G, path, encoding="utf-8"):
    """Write NetworkX graph in p2g format.

    Notes
    -----
    This format is meant to be used with directed graphs with
    possible self loops.
    """
    path.write((f"{G.name}\n").encode(encoding))
    path.write((f"{G.order()} {G.size()}\n").encode(encoding))
    nodes = list(G)
    # make dictionary mapping nodes to integers
    nodenumber = dict(zip(nodes, range(len(nodes))))
    for n in nodes:
        path.write((f"{n}\n").encode(encoding))
        for nbr in G.neighbors(n):
            path.write((f"{nodenumber[nbr]} ").encode(encoding))
        path.write("\n".encode(encoding))


@open_file(0, mode="r")
@nx._dispatchable(graphs=None, returns_graph=True)
def read_p2g(path, encoding="utf-8"):
    """Read graph in p2g format from path.

    Returns
    -------
    MultiDiGraph

    Notes
    -----
    If you want a DiGraph (with no self loops allowed and no edge data)
    use D=nx.DiGraph(read_p2g(path))
    """
    lines = (line.decode(encoding) for line in path)
    G = parse_p2g(lines)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_p2g(lines):
    """Parse p2g format graph from string or iterable.

    Returns
    -------
    MultiDiGraph
    """
    description = next(lines).strip()
    # are multiedges (parallel edges) allowed?
    G = nx.MultiDiGraph(name=description, selfloops=True)
    nnodes, nedges = map(int, next(lines).split())
    nodelabel = {}
    nbrs = {}
    # loop over the nodes keeping track of node labels and out neighbors
    # defer adding edges until all node labels are known
    for i in range(nnodes):
        n = next(lines).strip()
        nodelabel[i] = n
        G.add_node(n)
        nbrs[n] = map(int, next(lines).split())
    # now we know all of the node labels so we can add the edges
    # with the correct labels
    for n in G:
        for nbr in nbrs[n]:
            G.add_edge(n, nodelabel[nbr])
    return G
<|endoftext|>import networkx as nx

__all__ = ["adjacency_data", "adjacency_graph"]

_attrs = {"id": "id", "key": "key"}


def adjacency_data(G, attrs=_attrs):
    """Returns data in adjacency format that is suitable for JSON serialization
    and use in JavaScript documents.

    Parameters
    ----------
    G : NetworkX graph

    attrs : dict
        A dictionary that contains two keys 'id' and 'key'. The corresponding
        values provide the attribute names for storing NetworkX-internal graph
        data. The values should be unique. Default value:
        :samp:`dict(id='id', key='key')`.

        If some user-defined graph data use these attribute names as data keys,
        they may be silently dropped.

    Returns
    -------
    data : dict
       A dictionary with adjacency formatted data.

    Raises
    ------
    NetworkXError
        If values in attrs are not unique.

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.Graph([(1, 2)])
    >>> data = json_graph.adjacency_data(G)

    To serialize with json

    >>> import json
    >>> s = json.dumps(data)

    Notes
    -----
    Graph, node, and link attributes will be written when using this format
    but attribute keys must be strings if you want to serialize the resulting
    data with JSON.

    The default value of attrs will be changed in a future release of NetworkX.

    See Also
    --------
    adjacency_graph, node_link_data, tree_data
    """
    multigraph = G.is_multigraph()
    id_ = attrs["id"]
    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None if not multigraph else attrs["key"]
    if id_ == key:
        raise nx.NetworkXError("Attribute names are not unique.")
    data = {}
    data["directed"] = G.is_directed()
    data["multigraph"] = multigraph
    data["graph"] = list(G.graph.items())
    data["nodes"] = []
    data["adjacency"] = []
    for n, nbrdict in G.adjacency():
        data["nodes"].append({**G.nodes[n], id_: n})
        adj = []
        if multigraph:
            for nbr, keys in nbrdict.items():
                for k, d in keys.items():
                    adj.append({**d, id_: nbr, key: k})
        else:
            for nbr, d in nbrdict.items():
                adj.append({**d, id_: nbr})
        data["adjacency"].append(adj)
    return data


@nx._dispatchable(graphs=None, returns_graph=True)
def adjacency_graph(data, directed=False, multigraph=True, attrs=_attrs):
    """Returns graph from adjacency data format.

    Parameters
    ----------
    data : dict
        Adjacency list formatted graph data

    directed : bool
        If True, and direction not specified in data, return a directed graph.

    multigraph : bool
        If True, and multigraph not specified in data, return a multigraph.

    attrs : dict
        A dictionary that contains two keys 'id' and 'key'. The corresponding
        values provide the attribute names for storing NetworkX-internal graph
        data. The values should be unique. Default value:
        :samp:`dict(id='id', key='key')`.

    Returns
    -------
    G : NetworkX graph
       A NetworkX graph object

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.Graph([(1, 2)])
    >>> data = json_graph.adjacency_data(G)
    >>> H = json_graph.adjacency_graph(data)

    Notes
    -----
    The default value of attrs will be changed in a future release of NetworkX.

    See Also
    --------
    adjacency_graph, node_link_data, tree_data
    """
    multigraph = data.get("multigraph", multigraph)
    directed = data.get("directed", directed)
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()
    id_ = attrs["id"]
    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None if not multigraph else attrs["key"]
    graph.graph = dict(data.get("graph", []))
    mapping = []
    for d in data["nodes"]:
        node_data = d.copy()
        node = node_data.pop(id_)
        mapping.append(node)
        graph.add_node(node)
        graph.nodes[node].update(node_data)
    for i, d in enumerate(data["adjacency"]):
        source = mapping[i]
        for tdata in d:
            target_data = tdata.copy()
            target = target_data.pop(id_)
            if not multigraph:
                graph.add_edge(source, target)
                graph[source][target].update(target_data)
            else:
                ky = target_data.pop(key, None)
                graph.add_edge(source, target, key=ky)
                graph[source][target][ky].update(target_data)
    return graph
<|endoftext|>import networkx as nx

__all__ = ["cytoscape_data", "cytoscape_graph"]


def cytoscape_data(G, name="name", ident="id"):
    """Returns data in Cytoscape JSON format (cyjs).

    Parameters
    ----------
    G : NetworkX Graph
        The graph to convert to cytoscape format
    name : string
        A string which is mapped to the 'name' node element in cyjs format.
        Must not have the same value as `ident`.
    ident : string
        A string which is mapped to the 'id' node element in cyjs format.
        Must not have the same value as `name`.

    Returns
    -------
    data: dict
        A dictionary with cyjs formatted data.

    Raises
    ------
    NetworkXError
        If the values for `name` and `ident` are identical.

    See Also
    --------
    cytoscape_graph: convert a dictionary in cyjs format to a graph

    References
    ----------
    .. [1] Cytoscape user's manual:
       http://manual.cytoscape.org/en/stable/index.html

    Examples
    --------
    >>> G = nx.path_graph(2)
    >>> nx.cytoscape_data(G)  # doctest: +SKIP
    {'data': [],
     'directed': False,
     'multigraph': False,
     'elements': {'nodes': [{'data': {'id': '0', 'value': 0, 'name': '0'}},
       {'data': {'id': '1', 'value': 1, 'name': '1'}}],
      'edges': [{'data': {'source': 0, 'target': 1}}]}}
    """
    if name == ident:
        raise nx.NetworkXError("name and ident must be different.")

    jsondata = {"data": list(G.graph.items())}
    jsondata["directed"] = G.is_directed()
    jsondata["multigraph"] = G.is_multigraph()
    jsondata["elements"] = {"nodes": [], "edges": []}
    nodes = jsondata["elements"]["nodes"]
    edges = jsondata["elements"]["edges"]

    for i, j in G.nodes.items():
        n = {"data": j.copy()}
        n["data"]["id"] = j.get(ident) or str(i)
        n["data"]["value"] = i
        n["data"]["name"] = j.get(name) or str(i)
        nodes.append(n)

    if G.is_multigraph():
        for e in G.edges(keys=True):
            n = {"data": G.adj[e[0]][e[1]][e[2]].copy()}
            n["data"]["source"] = e[0]
            n["data"]["target"] = e[1]
            n["data"]["key"] = e[2]
            edges.append(n)
    else:
        for e in G.edges():
            n = {"data": G.adj[e[0]][e[1]].copy()}
            n["data"]["source"] = e[0]
            n["data"]["target"] = e[1]
            edges.append(n)
    return jsondata


@nx._dispatchable(graphs=None, returns_graph=True)
def cytoscape_graph(data, name="name", ident="id"):
    """
    Create a NetworkX graph from a dictionary in cytoscape JSON format.

    Parameters
    ----------
    data : dict
        A dictionary of data conforming to cytoscape JSON format.
    name : string
        A string which is mapped to the 'name' node element in cyjs format.
        Must not have the same value as `ident`.
    ident : string
        A string which is mapped to the 'id' node element in cyjs format.
        Must not have the same value as `name`.

    Returns
    -------
    graph : a NetworkX graph instance
        The `graph` can be an instance of `Graph`, `DiGraph`, `MultiGraph`, or
        `MultiDiGraph` depending on the input data.

    Raises
    ------
    NetworkXError
        If the `name` and `ident` attributes are identical.

    See Also
    --------
    cytoscape_data: convert a NetworkX graph to a dict in cyjs format

    References
    ----------
    .. [1] Cytoscape user's manual:
       http://manual.cytoscape.org/en/stable/index.html

    Examples
    --------
    >>> data_dict = {
    ...     "data": [],
    ...     "directed": False,
    ...     "multigraph": False,
    ...     "elements": {
    ...         "nodes": [
    ...             {"data": {"id": "0", "value": 0, "name": "0"}},
    ...             {"data": {"id": "1", "value": 1, "name": "1"}},
    ...         ],
    ...         "edges": [{"data": {"source": 0, "target": 1}}],
    ...     },
    ... }
    >>> G = nx.cytoscape_graph(data_dict)
    >>> G.name
    ''
    >>> G.nodes()
    NodeView((0, 1))
    >>> G.nodes(data=True)[0]
    {'id': '0', 'value': 0, 'name': '0'}
    >>> G.edges(data=True)
    EdgeDataView([(0, 1, {'source': 0, 'target': 1})])
    """
    if name == ident:
        raise nx.NetworkXError("name and ident must be different.")

    multigraph = data.get("multigraph")
    directed = data.get("directed")
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()
    graph.graph = dict(data.get("data"))
    for d in data["elements"]["nodes"]:
        node_data = d["data"].copy()
        node = d["data"]["value"]

        if d["data"].get(name):
            node_data[name] = d["data"].get(name)
        if d["data"].get(ident):
            node_data[ident] = d["data"].get(ident)

        graph.add_node(node)
        graph.nodes[node].update(node_data)

    for d in data["elements"]["edges"]:
        edge_data = d["data"].copy()
        sour = d["data"]["source"]
        targ = d["data"]["target"]
        if multigraph:
            key = d["data"].get("key", 0)
            graph.add_edge(sour, targ, key=key)
            graph.edges[sour, targ, key].update(edge_data)
        else:
            graph.add_edge(sour, targ)
            graph.edges[sour, targ].update(edge_data)
    return graph
<|endoftext|>"""
*********
JSON data
*********
Generate and parse JSON serializable data for NetworkX graphs.

These formats are suitable for use with the d3.js examples https://d3js.org/

The three formats that you can generate with NetworkX are:

 - node-link like in the d3.js example https://bl.ocks.org/mbostock/4062045
 - tree like in the d3.js example https://bl.ocks.org/mbostock/4063550
 - adjacency like in the d3.js example https://bost.ocks.org/mike/miserables/
"""

from networkx.readwrite.json_graph.node_link import *
from networkx.readwrite.json_graph.adjacency import *
from networkx.readwrite.json_graph.tree import *
from networkx.readwrite.json_graph.cytoscape import *
<|endoftext|>import warnings
from itertools import count

import networkx as nx

__all__ = ["node_link_data", "node_link_graph"]


def _to_tuple(x):
    """Converts lists to tuples, including nested lists.

    All other non-list inputs are passed through unmodified. This function is
    intended to be used to convert potentially nested lists from json files
    into valid nodes.

    Examples
    --------
    >>> _to_tuple([1, 2, [3, 4]])
    (1, 2, (3, 4))
    """
    if not isinstance(x, tuple | list):
        return x
    return tuple(map(_to_tuple, x))


def node_link_data(
    G,
    *,
    source="source",
    target="target",
    name="id",
    key="key",
    edges=None,
    nodes="nodes",
    link=None,
):
    """Returns data in node-link format that is suitable for JSON serialization
    and use in JavaScript documents.

    Parameters
    ----------
    G : NetworkX graph
    source : string
        A string that provides the 'source' attribute name for storing NetworkX-internal graph data.
    target : string
        A string that provides the 'target' attribute name for storing NetworkX-internal graph data.
    name : string
        A string that provides the 'name' attribute name for storing NetworkX-internal graph data.
    key : string
        A string that provides the 'key' attribute name for storing NetworkX-internal graph data.
    edges : string
        A string that provides the 'edges' attribute name for storing NetworkX-internal graph data.
    nodes : string
        A string that provides the 'nodes' attribute name for storing NetworkX-internal graph data.
    link : string
        .. deprecated:: 3.4

           The `link` argument is deprecated and will be removed in version `3.6`.
           Use the `edges` keyword instead.

        A string that provides the 'edges' attribute name for storing NetworkX-internal graph data.

    Returns
    -------
    data : dict
       A dictionary with node-link formatted data.

    Raises
    ------
    NetworkXError
        If the values of 'source', 'target' and 'key' are not unique.

    Examples
    --------
    >>> from pprint import pprint
    >>> G = nx.Graph([("A", "B")])
    >>> data1 = nx.node_link_data(G, edges="edges")
    >>> pprint(data1)
    {'directed': False,
     'edges': [{'source': 'A', 'target': 'B'}],
     'graph': {},
     'multigraph': False,
     'nodes': [{'id': 'A'}, {'id': 'B'}]}

    To serialize with JSON

    >>> import json
    >>> s1 = json.dumps(data1)
    >>> s1
    '{"directed": false, "multigraph": false, "graph": {}, "nodes": [{"id": "A"}, {"id": "B"}], "edges": [{"source": "A", "target": "B"}]}'

    A graph can also be serialized by passing `node_link_data` as an encoder function.

    >>> s1 = json.dumps(G, default=nx.node_link_data)
    >>> s1
    '{"directed": false, "multigraph": false, "graph": {}, "nodes": [{"id": "A"}, {"id": "B"}], "links": [{"source": "A", "target": "B"}]}'

    The attribute names for storing NetworkX-internal graph data can
    be specified as keyword options.

    >>> H = nx.gn_graph(2)
    >>> data2 = nx.node_link_data(
    ...     H, edges="links", source="from", target="to", nodes="vertices"
    ... )
    >>> pprint(data2)
    {'directed': True,
     'graph': {},
     'links': [{'from': 1, 'to': 0}],
     'multigraph': False,
     'vertices': [{'id': 0}, {'id': 1}]}

    Notes
    -----
    Graph, node, and link attributes are stored in this format.  Note that
    attribute keys will be converted to strings in order to comply with JSON.

    Attribute 'key' is only used for multigraphs.

    To use `node_link_data` in conjunction with `node_link_graph`,
    the keyword names for the attributes must match.

    See Also
    --------
    node_link_graph, adjacency_data, tree_data
    """
    # TODO: Remove between the lines when `link` deprecation expires
    # -------------------------------------------------------------
    if link is not None:
        warnings.warn(
            "Keyword argument 'link' is deprecated; use 'edges' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if edges is not None:
            raise ValueError(
                "Both 'edges' and 'link' are specified. Use 'edges', 'link' will be remove in a future release"
            )
        else:
            edges = link
    else:
        if edges is None:
            warnings.warn(
                (
                    '\nThe default value will be `edges="edges" in NetworkX 3.6.\n\n'
                    "To make this warning go away, explicitly set the edges kwarg, e.g.:\n\n"
                    '  nx.node_link_data(G, edges="links") to preserve current behavior, or\n'
                    '  nx.node_link_data(G, edges="edges") for forward compatibility.'
                ),
                FutureWarning,
            )
            edges = "links"
    # ------------------------------------------------------------

    multigraph = G.is_multigraph()

    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None if not multigraph else key
    if len({source, target, key}) < 3:
        raise nx.NetworkXError("Attribute names are not unique.")
    data = {
        "directed": G.is_directed(),
        "multigraph": multigraph,
        "graph": G.graph,
        nodes: [{**G.nodes[n], name: n} for n in G],
    }
    if multigraph:
        data[edges] = [
            {**d, source: u, target: v, key: k}
            for u, v, k, d in G.edges(keys=True, data=True)
        ]
    else:
        data[edges] = [{**d, source: u, target: v} for u, v, d in G.edges(data=True)]
    return data


@nx._dispatchable(graphs=None, returns_graph=True)
def node_link_graph(
    data,
    directed=False,
    multigraph=True,
    *,
    source="source",
    target="target",
    name="id",
    key="key",
    edges=None,
    nodes="nodes",
    link=None,
):
    """Returns graph from node-link data format.

    Useful for de-serialization from JSON.

    Parameters
    ----------
    data : dict
        node-link formatted graph data

    directed : bool
        If True, and direction not specified in data, return a directed graph.

    multigraph : bool
        If True, and multigraph not specified in data, return a multigraph.

    source : string
        A string that provides the 'source' attribute name for storing NetworkX-internal graph data.
    target : string
        A string that provides the 'target' attribute name for storing NetworkX-internal graph data.
    name : string
        A string that provides the 'name' attribute name for storing NetworkX-internal graph data.
    key : string
        A string that provides the 'key' attribute name for storing NetworkX-internal graph data.
    edges : string
        A string that provides the 'edges' attribute name for storing NetworkX-internal graph data.
    nodes : string
        A string that provides the 'nodes' attribute name for storing NetworkX-internal graph data.
    link : string
        .. deprecated:: 3.4

           The `link` argument is deprecated and will be removed in version `3.6`.
           Use the `edges` keyword instead.

        A string that provides the 'edges' attribute name for storing NetworkX-internal graph data.

    Returns
    -------
    G : NetworkX graph
        A NetworkX graph object

    Examples
    --------

    Create data in node-link format by converting a graph.

    >>> from pprint import pprint
    >>> G = nx.Graph([("A", "B")])
    >>> data = nx.node_link_data(G, edges="edges")
    >>> pprint(data)
    {'directed': False,
     'edges': [{'source': 'A', 'target': 'B'}],
     'graph': {},
     'multigraph': False,
     'nodes': [{'id': 'A'}, {'id': 'B'}]}

    Revert data in node-link format to a graph.

    >>> H = nx.node_link_graph(data, edges="edges")
    >>> print(H.edges)
    [('A', 'B')]

    To serialize and deserialize a graph with JSON,

    >>> import json
    >>> d = json.dumps(nx.node_link_data(G, edges="edges"))
    >>> H = nx.node_link_graph(json.loads(d), edges="edges")
    >>> print(G.edges, H.edges)
    [('A', 'B')] [('A', 'B')]


    Notes
    -----
    Attribute 'key' is only used for multigraphs.

    To use `node_link_data` in conjunction with `node_link_graph`,
    the keyword names for the attributes must match.

    See Also
    --------
    node_link_data, adjacency_data, tree_data
    """
    # TODO: Remove between the lines when `link` deprecation expires
    # -------------------------------------------------------------
    if link is not None:
        warnings.warn(
            "Keyword argument 'link' is deprecated; use 'edges' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if edges is not None:
            raise ValueError(
                "Both 'edges' and 'link' are specified. Use 'edges', 'link' will be remove in a future release"
            )
        else:
            edges = link
    else:
        if edges is None:
            warnings.warn(
                (
                    '\nThe default value will be changed to `edges="edges" in NetworkX 3.6.\n\n'
                    "To make this warning go away, explicitly set the edges kwarg, e.g.:\n\n"
                    '  nx.node_link_graph(data, edges="links") to preserve current behavior, or\n'
                    '  nx.node_link_graph(data, edges="edges") for forward compatibility.'
                ),
                FutureWarning,
            )
            edges = "links"
    # -------------------------------------------------------------

    multigraph = data.get("multigraph", multigraph)
    directed = data.get("directed", directed)
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()

    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None if not multigraph else key
    graph.graph = data.get("graph", {})
    c = count()
    for d in data[nodes]:
        node = _to_tuple(d.get(name, next(c)))
        nodedata = {str(k): v for k, v in d.items() if k != name}
        graph.add_node(node, **nodedata)
    for d in data[edges]:
        src = tuple(d[source]) if isinstance(d[source], list) else d[source]
        tgt = tuple(d[target]) if isinstance(d[target], list) else d[target]
        if not multigraph:
            edgedata = {str(k): v for k, v in d.items() if k != source and k != target}
            graph.add_edge(src, tgt, **edgedata)
        else:
            ky = d.get(key, None)
            edgedata = {
                str(k): v
                for k, v in d.items()
                if k != source and k != target and k != key
            }
            graph.add_edge(src, tgt, ky, **edgedata)
    return graph
<|endoftext|>from itertools import chain

import networkx as nx

__all__ = ["tree_data", "tree_graph"]


def tree_data(G, root, ident="id", children="children"):
    """Returns data in tree format that is suitable for JSON serialization
    and use in JavaScript documents.

    Parameters
    ----------
    G : NetworkX graph
       G must be an oriented tree

    root : node
       The root of the tree

    ident : string
        Attribute name for storing NetworkX-internal graph data. `ident` must
        have a different value than `children`. The default is 'id'.

    children : string
        Attribute name for storing NetworkX-internal graph data. `children`
        must have a different value than `ident`. The default is 'children'.

    Returns
    -------
    data : dict
       A dictionary with node-link formatted data.

    Raises
    ------
    NetworkXError
        If `children` and `ident` attributes are identical.

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.DiGraph([(1, 2)])
    >>> data = json_graph.tree_data(G, root=1)

    To serialize with json

    >>> import json
    >>> s = json.dumps(data)

    Notes
    -----
    Node attributes are stored in this format but keys
    for attributes must be strings if you want to serialize with JSON.

    Graph and edge attributes are not stored.

    See Also
    --------
    tree_graph, node_link_data, adjacency_data
    """
    if G.number_of_nodes() != G.number_of_edges() + 1:
        raise TypeError("G is not a tree.")
    if not G.is_directed():
        raise TypeError("G is not directed.")
    if not nx.is_weakly_connected(G):
        raise TypeError("G is not weakly connected.")

    if ident == children:
        raise nx.NetworkXError("The values for `id` and `children` must be different.")

    def add_children(n, G):
        nbrs = G[n]
        if len(nbrs) == 0:
            return []
        children_ = []
        for child in nbrs:
            d = {**G.nodes[child], ident: child}
            c = add_children(child, G)
            if c:
                d[children] = c
            children_.append(d)
        return children_

    return {**G.nodes[root], ident: root, children: add_children(root, G)}


@nx._dispatchable(graphs=None, returns_graph=True)
def tree_graph(data, ident="id", children="children"):
    """Returns graph from tree data format.

    Parameters
    ----------
    data : dict
        Tree formatted graph data

    ident : string
        Attribute name for storing NetworkX-internal graph data. `ident` must
        have a different value than `children`. The default is 'id'.

    children : string
        Attribute name for storing NetworkX-internal graph data. `children`
        must have a different value than `ident`. The default is 'children'.

    Returns
    -------
    G : NetworkX DiGraph

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.DiGraph([(1, 2)])
    >>> data = json_graph.tree_data(G, root=1)
    >>> H = json_graph.tree_graph(data)

    See Also
    --------
    tree_data, node_link_data, adjacency_data
    """
    graph = nx.DiGraph()

    def add_children(parent, children_):
        for data in children_:
            child = data[ident]
            graph.add_edge(parent, child)
            grandchildren = data.get(children, [])
            if grandchildren:
                add_children(child, grandchildren)
            nodedata = {
                str(k): v for k, v in data.items() if k != ident and k != children
            }
            graph.add_node(child, **nodedata)

    root = data[ident]
    children_ = data.get(children, [])
    nodedata = {str(k): v for k, v in data.items() if k != ident and k != children}
    graph.add_node(root, **nodedata)
    add_children(root, children_)
    return graph
<|endoftext|>"""Modularity matrix of graphs."""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["modularity_matrix", "directed_modularity_matrix"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def modularity_matrix(G, nodelist=None, weight=None):
    r"""Returns the modularity matrix of G.

    The modularity matrix is the matrix B = A - <A>, where A is the adjacency
    matrix and <A> is the average adjacency matrix, assuming that the graph
    is described by the configuration model.

    More specifically, the element B_ij of B is defined as

    .. math::
        A_{ij} - {k_i k_j \over 2 m}

    where k_i is the degree of node i, and where m is the number of edges
    in the graph. When weight is set to a name of an attribute edge, Aij, k_i,
    k_j and m are computed using its value.

    Parameters
    ----------
    G : Graph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used for
       the edge weight.  If None then all edge weights are 1.

    Returns
    -------
    B : Numpy array
      The modularity matrix of G.

    Examples
    --------
    >>> k = [3, 2, 2, 1, 0]
    >>> G = nx.havel_hakimi_graph(k)
    >>> B = nx.modularity_matrix(G)


    See Also
    --------
    to_numpy_array
    modularity_spectrum
    adjacency_matrix
    directed_modularity_matrix

    References
    ----------
    .. [1] M. E. J. Newman, "Modularity and community structure in networks",
           Proc. Natl. Acad. Sci. USA, vol. 103, pp. 8577-8582, 2006.
    """
    import numpy as np

    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format="csr")
    k = A.sum(axis=1)
    m = k.sum() * 0.5
    # Expected adjacency matrix
    X = np.outer(k, k) / (2 * m)

    return A - X


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def directed_modularity_matrix(G, nodelist=None, weight=None):
    """Returns the directed modularity matrix of G.

    The modularity matrix is the matrix B = A - <A>, where A is the adjacency
    matrix and <A> is the expected adjacency matrix, assuming that the graph
    is described by the configuration model.

    More specifically, the element B_ij of B is defined as

    .. math::
        B_{ij} = A_{ij} - k_i^{out} k_j^{in} / m

    where :math:`k_i^{in}` is the in degree of node i, and :math:`k_j^{out}` is the out degree
    of node j, with m the number of edges in the graph. When weight is set
    to a name of an attribute edge, Aij, k_i, k_j and m are computed using
    its value.

    Parameters
    ----------
    G : DiGraph
       A NetworkX DiGraph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used for
       the edge weight.  If None then all edge weights are 1.

    Returns
    -------
    B : Numpy array
      The modularity matrix of G.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from(
    ...     (
    ...         (1, 2),
    ...         (1, 3),
    ...         (3, 1),
    ...         (3, 2),
    ...         (3, 5),
    ...         (4, 5),
    ...         (4, 6),
    ...         (5, 4),
    ...         (5, 6),
    ...         (6, 4),
    ...     )
    ... )
    >>> B = nx.directed_modularity_matrix(G)


    Notes
    -----
    NetworkX defines the element A_ij of the adjacency matrix as 1 if there
    is a link going from node i to node j. Leicht and Newman use the opposite
    definition. This explains the different expression for B_ij.

    See Also
    --------
    to_numpy_array
    modularity_spectrum
    adjacency_matrix
    modularity_matrix

    References
    ----------
    .. [1] E. A. Leicht, M. E. J. Newman,
        "Community structure in directed networks",
        Phys. Rev Lett., vol. 100, no. 11, p. 118703, 2008.
    """
    import numpy as np

    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format="csr")
    k_in = A.sum(axis=0)
    k_out = A.sum(axis=1)
    m = k_in.sum()
    # Expected adjacency matrix
    X = np.outer(k_out, k_in) / m

    return A - X
<|endoftext|>"""
Adjacency matrix and incidence matrix of graphs.
"""

import networkx as nx

__all__ = ["incidence_matrix", "adjacency_matrix"]


@nx._dispatchable(edge_attrs="weight")
def incidence_matrix(
    G, nodelist=None, edgelist=None, oriented=False, weight=None, *, dtype=None
):
    """Returns incidence matrix of G.

    The incidence matrix assigns each row to a node and each column to an edge.
    For a standard incidence matrix a 1 appears wherever a row's node is
    incident on the column's edge.  For an oriented incidence matrix each
    edge is assigned an orientation (arbitrarily for undirected and aligning to
    direction for directed).  A -1 appears for the source (tail) of an edge and
    1 for the destination (head) of the edge.  The elements are zero otherwise.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional   (default= all nodes in G)
       The rows are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    edgelist : list, optional (default= all edges in G)
       The columns are ordered according to the edges in edgelist.
       If edgelist is None, then the ordering is produced by G.edges().

    oriented: bool, optional (default=False)
       If True, matrix elements are +1 or -1 for the head or tail node
       respectively of each edge.  If False, +1 occurs at both nodes.

    weight : string or None, optional (default=None)
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.  Edge weights, if used,
       should be positive so that the orientation can provide the sign.

    dtype : a NumPy dtype or None (default=None)
        The dtype of the output sparse array. This type should be a compatible
        type of the weight argument, eg. if weight would return a float this
        argument should also be a float.
        If None, then the default for SciPy is used.

    Returns
    -------
    A : SciPy sparse array
      The incidence matrix of G.

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges in edgelist should be
    (u,v,key) 3-tuples.

    "Networks are the best discrete model for so many problems in
    applied mathematics" [1]_.

    References
    ----------
    .. [1] Gil Strang, Network applications: A = incidence matrix,
       http://videolectures.net/mit18085f07_strang_lec03/
    """
    import scipy as sp

    if nodelist is None:
        nodelist = list(G)
    if edgelist is None:
        if G.is_multigraph():
            edgelist = list(G.edges(keys=True))
        else:
            edgelist = list(G.edges())
    A = sp.sparse.lil_array((len(nodelist), len(edgelist)), dtype=dtype)
    node_index = {node: i for i, node in enumerate(nodelist)}
    for ei, e in enumerate(edgelist):
        (u, v) = e[:2]
        if u == v:
            continue  # self loops give zero column
        try:
            ui = node_index[u]
            vi = node_index[v]
        except KeyError as err:
            raise nx.NetworkXError(
                f"node {u} or {v} in edgelist but not in nodelist"
            ) from err
        if weight is None:
            wt = 1
        else:
            if G.is_multigraph():
                ekey = e[2]
                wt = G[u][v][ekey].get(weight, 1)
            else:
                wt = G[u][v].get(weight, 1)
        if oriented:
            A[ui, ei] = -wt
            A[vi, ei] = wt
        else:
            A[ui, ei] = wt
            A[vi, ei] = wt
    return A.asformat("csc")


@nx._dispatchable(edge_attrs="weight")
def adjacency_matrix(G, nodelist=None, dtype=None, weight="weight"):
    """Returns adjacency matrix of `G`.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in `nodelist`.
       If ``nodelist=None`` (the default), then the ordering is produced by
       ``G.nodes()``.

    dtype : NumPy data-type, optional
        The desired data-type for the array.
        If `None`, then the NumPy default is used.

    weight : string or None, optional (default='weight')
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    A : SciPy sparse array
      Adjacency matrix representation of G.

    Notes
    -----
    For directed graphs, entry ``i, j`` corresponds to an edge from ``i`` to ``j``.

    If you want a pure Python adjacency matrix representation try
    :func:`~networkx.convert.to_dict_of_dicts` which will return a
    dictionary-of-dictionaries format that can be addressed as a
    sparse matrix.

    For multigraphs with parallel edges the weights are summed.
    See :func:`networkx.convert_matrix.to_numpy_array` for other options.

    The convention used for self-loop edges in graphs is to assign the
    diagonal matrix entry value to the edge weight attribute
    (or the number 1 if the edge has no weight attribute).  If the
    alternate convention of doubling the edge weight is desired the
    resulting SciPy sparse array can be modified as follows::

        >>> G = nx.Graph([(1, 1)])
        >>> A = nx.adjacency_matrix(G)
        >>> A.toarray()
        array([[1]])
        >>> A.setdiag(A.diagonal() * 2)
        >>> A.toarray()
        array([[2]])

    See Also
    --------
    to_numpy_array
    to_scipy_sparse_array
    to_dict_of_dicts
    adjacency_spectrum
    """
    return nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=dtype, weight=weight)
<|endoftext|>from networkx.linalg.attrmatrix import *
from networkx.linalg import attrmatrix
from networkx.linalg.spectrum import *
from networkx.linalg import spectrum
from networkx.linalg.graphmatrix import *
from networkx.linalg import graphmatrix
from networkx.linalg.laplacianmatrix import *
from networkx.linalg import laplacianmatrix
from networkx.linalg.algebraicconnectivity import *
from networkx.linalg.modularitymatrix import *
from networkx.linalg import modularitymatrix
from networkx.linalg.bethehessianmatrix import *
from networkx.linalg import bethehessianmatrix
<|endoftext|>"""
Functions for constructing matrix-like objects from graph attributes.
"""

import networkx as nx

__all__ = ["attr_matrix", "attr_sparse_matrix"]


def _node_value(G, node_attr):
    """Returns a function that returns a value from G.nodes[u].

    We return a function expecting a node as its sole argument. Then, in the
    simplest scenario, the returned function will return G.nodes[u][node_attr].
    However, we also handle the case when `node_attr` is None or when it is a
    function itself.

    Parameters
    ----------
    G : graph
        A NetworkX graph

    node_attr : {None, str, callable}
        Specification of how the value of the node attribute should be obtained
        from the node attribute dictionary.

    Returns
    -------
    value : function
        A function expecting a node as its sole argument. The function will
        returns a value from G.nodes[u] that depends on `edge_attr`.

    """
    if node_attr is None:

        def value(u):
            return u

    elif not callable(node_attr):
        # assume it is a key for the node attribute dictionary
        def value(u):
            return G.nodes[u][node_attr]

    else:
        # Advanced:  Allow users to specify something else.
        #
        # For example,
        #     node_attr = lambda u: G.nodes[u].get('size', .5) * 3
        #
        value = node_attr

    return value


def _edge_value(G, edge_attr):
    """Returns a function that returns a value from G[u][v].

    Suppose there exists an edge between u and v.  Then we return a function
    expecting u and v as arguments.  For Graph and DiGraph, G[u][v] is
    the edge attribute dictionary, and the function (essentially) returns
    G[u][v][edge_attr].  However, we also handle cases when `edge_attr` is None
    and when it is a function itself. For MultiGraph and MultiDiGraph, G[u][v]
    is a dictionary of all edges between u and v.  In this case, the returned
    function sums the value of `edge_attr` for every edge between u and v.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    edge_attr : {None, str, callable}
        Specification of how the value of the edge attribute should be obtained
        from the edge attribute dictionary, G[u][v].  For multigraphs, G[u][v]
        is a dictionary of all the edges between u and v.  This allows for
        special treatment of multiedges.

    Returns
    -------
    value : function
        A function expecting two nodes as parameters. The nodes should
        represent the from- and to- node of an edge. The function will
        return a value from G[u][v] that depends on `edge_attr`.

    """

    if edge_attr is None:
        # topological count of edges

        if G.is_multigraph():

            def value(u, v):
                return len(G[u][v])

        else:

            def value(u, v):
                return 1

    elif not callable(edge_attr):
        # assume it is a key for the edge attribute dictionary

        if edge_attr == "weight":
            # provide a default value
            if G.is_multigraph():

                def value(u, v):
                    return sum(d.get(edge_attr, 1) for d in G[u][v].values())

            else:

                def value(u, v):
                    return G[u][v].get(edge_attr, 1)

        else:
            # otherwise, the edge attribute MUST exist for each edge
            if G.is_multigraph():

                def value(u, v):
                    return sum(d[edge_attr] for d in G[u][v].values())

            else:

                def value(u, v):
                    return G[u][v][edge_attr]

    else:
        # Advanced:  Allow users to specify something else.
        #
        # Alternative default value:
        #     edge_attr = lambda u,v: G[u][v].get('thickness', .5)
        #
        # Function on an attribute:
        #     edge_attr = lambda u,v: abs(G[u][v]['weight'])
        #
        # Handle Multi(Di)Graphs differently:
        #     edge_attr = lambda u,v: numpy.prod([d['size'] for d in G[u][v].values()])
        #
        # Ignore multiple edges
        #     edge_attr = lambda u,v: 1 if len(G[u][v]) else 0
        #
        value = edge_attr

    return value


@nx._dispatchable(edge_attrs={"edge_attr": None}, node_attrs="node_attr")
def attr_matrix(
    G,
    edge_attr=None,
    node_attr=None,
    normalized=False,
    rc_order=None,
    dtype=None,
    order=None,
):
    """Returns the attribute matrix using attributes from `G` as a numpy array.

    If only `G` is passed in, then the adjacency matrix is constructed.

    Let A be a discrete set of values for the node attribute `node_attr`. Then
    the elements of A represent the rows and columns of the constructed matrix.
    Now, iterate through every edge e=(u,v) in `G` and consider the value
    of the edge attribute `edge_attr`.  If ua and va are the values of the
    node attribute `node_attr` for u and v, respectively, then the value of
    the edge attribute is added to the matrix element at (ua, va).

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the attribute matrix.

    edge_attr : str, optional
        Each element of the matrix represents a running total of the
        specified edge attribute for edges whose node attributes correspond
        to the rows/cols of the matrix. The attribute must be present for
        all edges in the graph. If no attribute is specified, then we
        just count the number of edges whose node attributes correspond
        to the matrix element.

    node_attr : str, optional
        Each row and column in the matrix represents a particular value
        of the node attribute.  The attribute must be present for all nodes
        in the graph. Note, the values of this attribute should be reliably
        hashable. So, float values are not recommended. If no attribute is
        specified, then the rows and columns will be the nodes of the graph.

    normalized : bool, optional
        If True, then each row is normalized by the summation of its values.

    rc_order : list, optional
        A list of the node attribute values. This list specifies the ordering
        of rows and columns of the array. If no ordering is provided, then
        the ordering will be random (and also, a return value).

    Other Parameters
    ----------------
    dtype : NumPy data-type, optional
        A valid NumPy dtype used to initialize the array. Keep in mind certain
        dtypes can yield unexpected results if the array is to be normalized.
        The parameter is passed to numpy.zeros(). If unspecified, the NumPy
        default is used.

    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. This parameter is passed to
        numpy.zeros(). If unspecified, the NumPy default is used.

    Returns
    -------
    M : 2D NumPy ndarray
        The attribute matrix.

    ordering : list
        If `rc_order` was specified, then only the attribute matrix is returned.
        However, if `rc_order` was None, then the ordering used to construct
        the matrix is returned as well.

    Examples
    --------
    Construct an adjacency matrix:

    >>> G = nx.Graph()
    >>> G.add_edge(0, 1, thickness=1, weight=3)
    >>> G.add_edge(0, 2, thickness=2)
    >>> G.add_edge(1, 2, thickness=3)
    >>> nx.attr_matrix(G, rc_order=[0, 1, 2])
    array([[0., 1., 1.],
           [1., 0., 1.],
           [1., 1., 0.]])

    Alternatively, we can obtain the matrix describing edge thickness.

    >>> nx.attr_matrix(G, edge_attr="thickness", rc_order=[0, 1, 2])
    array([[0., 1., 2.],
           [1., 0., 3.],
           [2., 3., 0.]])

    We can also color the nodes and ask for the probability distribution over
    all edges (u,v) describing:

        Pr(v has color Y | u has color X)

    >>> G.nodes[0]["color"] = "red"
    >>> G.nodes[1]["color"] = "red"
    >>> G.nodes[2]["color"] = "blue"
    >>> rc = ["red", "blue"]
    >>> nx.attr_matrix(G, node_attr="color", normalized=True, rc_order=rc)
    array([[0.33333333, 0.66666667],
           [1.        , 0.        ]])

    For example, the above tells us that for all edges (u,v):

        Pr( v is red  | u is red)  = 1/3
        Pr( v is blue | u is red)  = 2/3

        Pr( v is red  | u is blue) = 1
        Pr( v is blue | u is blue) = 0

    Finally, we can obtain the total weights listed by the node colors.

    >>> nx.attr_matrix(G, edge_attr="weight", node_attr="color", rc_order=rc)
    array([[3., 2.],
           [2., 0.]])

    Thus, the total weight over all edges (u,v) with u and v having colors:

        (red, red)   is 3   # the sole contribution is from edge (0,1)
        (red, blue)  is 2   # contributions from edges (0,2) and (1,2)
        (blue, red)  is 2   # same as (red, blue) since graph is undirected
        (blue, blue) is 0   # there are no edges with blue endpoints

    """
    import numpy as np

    edge_value = _edge_value(G, edge_attr)
    node_value = _node_value(G, node_attr)

    if rc_order is None:
        ordering = list({node_value(n) for n in G})
    else:
        ordering = rc_order

    N = len(ordering)
    undirected = not G.is_directed()
    index = dict(zip(ordering, range(N)))
    M = np.zeros((N, N), dtype=dtype, order=order)

    seen = set()
    for u, nbrdict in G.adjacency():
        for v in nbrdict:
            # Obtain the node attribute values.
            i, j = index[node_value(u)], index[node_value(v)]
            if v not in seen:
                M[i, j] += edge_value(u, v)
                if undirected:
                    M[j, i] = M[i, j]

        if undirected:
            seen.add(u)

    if normalized:
        M /= M.sum(axis=1).reshape((N, 1))

    if rc_order is None:
        return M, ordering
    else:
        return M


@nx._dispatchable(edge_attrs={"edge_attr": None}, node_attrs="node_attr")
def attr_sparse_matrix(
    G, edge_attr=None, node_attr=None, normalized=False, rc_order=None, dtype=None
):
    """Returns a SciPy sparse array using attributes from G.

    If only `G` is passed in, then the adjacency matrix is constructed.

    Let A be a discrete set of values for the node attribute `node_attr`. Then
    the elements of A represent the rows and columns of the constructed matrix.
    Now, iterate through every edge e=(u,v) in `G` and consider the value
    of the edge attribute `edge_attr`.  If ua and va are the values of the
    node attribute `node_attr` for u and v, respectively, then the value of
    the edge attribute is added to the matrix element at (ua, va).

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the NumPy matrix.

    edge_attr : str, optional
        Each element of the matrix represents a running total of the
        specified edge attribute for edges whose node attributes correspond
        to the rows/cols of the matrix. The attribute must be present for
        all edges in the graph. If no attribute is specified, then we
        just count the number of edges whose node attributes correspond
        to the matrix element.

    node_attr : str, optional
        Each row and column in the matrix represents a particular value
        of the node attribute.  The attribute must be present for all nodes
        in the graph. Note, the values of this attribute should be reliably
        hashable. So, float values are not recommended. If no attribute is
        specified, then the rows and columns will be the nodes of the graph.

    normalized : bool, optional
        If True, then each row is normalized by the summation of its values.

    rc_order : list, optional
        A list of the node attribute values. This list specifies the ordering
        of rows and columns of the array. If no ordering is provided, then
        the ordering will be random (and also, a return value).

    Other Parameters
    ----------------
    dtype : NumPy data-type, optional
        A valid NumPy dtype used to initialize the array. Keep in mind certain
        dtypes can yield unexpected results if the array is to be normalized.
        The parameter is passed to numpy.zeros(). If unspecified, the NumPy
        default is used.

    Returns
    -------
    M : SciPy sparse array
        The attribute matrix.

    ordering : list
        If `rc_order` was specified, then only the matrix is returned.
        However, if `rc_order` was None, then the ordering used to construct
        the matrix is returned as well.

    Examples
    --------
    Construct an adjacency matrix:

    >>> G = nx.Graph()
    >>> G.add_edge(0, 1, thickness=1, weight=3)
    >>> G.add_edge(0, 2, thickness=2)
    >>> G.add_edge(1, 2, thickness=3)
    >>> M = nx.attr_sparse_matrix(G, rc_order=[0, 1, 2])
    >>> M.toarray()
    array([[0., 1., 1.],
           [1., 0., 1.],
           [1., 1., 0.]])

    Alternatively, we can obtain the matrix describing edge thickness.

    >>> M = nx.attr_sparse_matrix(G, edge_attr="thickness", rc_order=[0, 1, 2])
    >>> M.toarray()
    array([[0., 1., 2.],
           [1., 0., 3.],
           [2., 3., 0.]])

    We can also color the nodes and ask for the probability distribution over
    all edges (u,v) describing:

        Pr(v has color Y | u has color X)

    >>> G.nodes[0]["color"] = "red"
    >>> G.nodes[1]["color"] = "red"
    >>> G.nodes[2]["color"] = "blue"
    >>> rc = ["red", "blue"]
    >>> M = nx.attr_sparse_matrix(G, node_attr="color", normalized=True, rc_order=rc)
    >>> M.toarray()
    array([[0.33333333, 0.66666667],
           [1.        , 0.        ]])

    For example, the above tells us that for all edges (u,v):

        Pr( v is red  | u is red)  = 1/3
        Pr( v is blue | u is red)  = 2/3

        Pr( v is red  | u is blue) = 1
        Pr( v is blue | u is blue) = 0

    Finally, we can obtain the total weights listed by the node colors.

    >>> M = nx.attr_sparse_matrix(G, edge_attr="weight", node_attr="color", rc_order=rc)
    >>> M.toarray()
    array([[3., 2.],
           [2., 0.]])

    Thus, the total weight over all edges (u,v) with u and v having colors:

        (red, red)   is 3   # the sole contribution is from edge (0,1)
        (red, blue)  is 2   # contributions from edges (0,2) and (1,2)
        (blue, red)  is 2   # same as (red, blue) since graph is undirected
        (blue, blue) is 0   # there are no edges with blue endpoints

    """
    import numpy as np
    import scipy as sp

    edge_value = _edge_value(G, edge_attr)
    node_value = _node_value(G, node_attr)

    if rc_order is None:
        ordering = list({node_value(n) for n in G})
    else:
        ordering = rc_order

    N = len(ordering)
    undirected = not G.is_directed()
    index = dict(zip(ordering, range(N)))
    M = sp.sparse.lil_array((N, N), dtype=dtype)

    seen = set()
    for u, nbrdict in G.adjacency():
        for v in nbrdict:
            # Obtain the node attribute values.
            i, j = index[node_value(u)], index[node_value(v)]
            if v not in seen:
                M[i, j] += edge_value(u, v)
                if undirected:
                    M[j, i] = M[i, j]

        if undirected:
            seen.add(u)

    if normalized:
        M *= 1 / M.sum(axis=1)[:, np.newaxis]  # in-place mult preserves sparse

    if rc_order is None:
        return M, ordering
    else:
        return M
<|endoftext|>"""Bethe Hessian or deformed Laplacian matrix of graphs."""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["bethe_hessian_matrix"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def bethe_hessian_matrix(G, r=None, nodelist=None):
    r"""Returns the Bethe Hessian matrix of G.

    The Bethe Hessian is a family of matrices parametrized by r, defined as
    H(r) = (r^2 - 1) I - r A + D where A is the adjacency matrix, D is the
    diagonal matrix of node degrees, and I is the identify matrix. It is equal
    to the graph laplacian when the regularizer r = 1.

    The default choice of regularizer should be the ratio [2]_

    .. math::
      r_m = \left(\sum k_i \right)^{-1}\left(\sum k_i^2 \right) - 1

    Parameters
    ----------
    G : Graph
       A NetworkX graph
    r : float
       Regularizer parameter
    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by ``G.nodes()``.

    Returns
    -------
    H : scipy.sparse.csr_array
      The Bethe Hessian matrix of `G`, with parameter `r`.

    Examples
    --------
    >>> k = [3, 2, 2, 1, 0]
    >>> G = nx.havel_hakimi_graph(k)
    >>> H = nx.bethe_hessian_matrix(G)
    >>> H.toarray()
    array([[ 3.5625, -1.25  , -1.25  , -1.25  ,  0.    ],
           [-1.25  ,  2.5625, -1.25  ,  0.    ,  0.    ],
           [-1.25  , -1.25  ,  2.5625,  0.    ,  0.    ],
           [-1.25  ,  0.    ,  0.    ,  1.5625,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.5625]])

    See Also
    --------
    bethe_hessian_spectrum
    adjacency_matrix
    laplacian_matrix

    References
    ----------
    .. [1] A. Saade, F. Krzakala and L. Zdeborová
       "Spectral Clustering of Graphs with the Bethe Hessian",
       Advances in Neural Information Processing Systems, 2014.
    .. [2] C. M. Le, E. Levina
       "Estimating the number of communities in networks by spectral methods"
       arXiv:1507.00827, 2015.
    """
    import scipy as sp

    if nodelist is None:
        nodelist = list(G)
    if r is None:
        r = sum(d**2 for v, d in nx.degree(G)) / sum(d for v, d in nx.degree(G)) - 1
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, format="csr")
    n, m = A.shape
    # TODO: Rm csr_array wrapper when spdiags array creation becomes available
    D = sp.sparse.csr_array(sp.sparse.spdiags(A.sum(axis=1), 0, m, n, format="csr"))
    # TODO: Rm csr_array wrapper when eye array creation becomes available
    I = sp.sparse.csr_array(sp.sparse.eye(m, n, format="csr"))
    return (r**2 - 1) * I - r * A + D
<|endoftext|>"""
Eigenvalue spectrum of graphs.
"""

import networkx as nx

__all__ = [
    "laplacian_spectrum",
    "adjacency_spectrum",
    "modularity_spectrum",
    "normalized_laplacian_spectrum",
    "bethe_hessian_spectrum",
]


@nx._dispatchable(edge_attrs="weight")
def laplacian_spectrum(G, weight="weight"):
    """Returns eigenvalues of the Laplacian of G

    Parameters
    ----------
    G : graph
       A NetworkX graph

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges weights are summed.
    See :func:`~networkx.convert_matrix.to_numpy_array` for other options.

    See Also
    --------
    laplacian_matrix

    Examples
    --------
    The multiplicity of 0 as an eigenvalue of the laplacian matrix is equal
    to the number of connected components of G.

    >>> G = nx.Graph()  # Create a graph with 5 nodes and 3 connected components
    >>> G.add_nodes_from(range(5))
    >>> G.add_edges_from([(0, 2), (3, 4)])
    >>> nx.laplacian_spectrum(G)
    array([0., 0., 0., 2., 2.])

    """
    import scipy as sp

    return sp.linalg.eigvalsh(nx.laplacian_matrix(G, weight=weight).todense())


@nx._dispatchable(edge_attrs="weight")
def normalized_laplacian_spectrum(G, weight="weight"):
    """Return eigenvalues of the normalized Laplacian of G

    Parameters
    ----------
    G : graph
       A NetworkX graph

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges weights are summed.
    See to_numpy_array for other options.

    See Also
    --------
    normalized_laplacian_matrix
    """
    import scipy as sp

    return sp.linalg.eigvalsh(
        nx.normalized_laplacian_matrix(G, weight=weight).todense()
    )


@nx._dispatchable(edge_attrs="weight")
def adjacency_spectrum(G, weight="weight"):
    """Returns eigenvalues of the adjacency matrix of G.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges weights are summed.
    See to_numpy_array for other options.

    See Also
    --------
    adjacency_matrix
    """
    import scipy as sp

    return sp.linalg.eigvals(nx.adjacency_matrix(G, weight=weight).todense())


@nx._dispatchable
def modularity_spectrum(G):
    """Returns eigenvalues of the modularity matrix of G.

    Parameters
    ----------
    G : Graph
       A NetworkX Graph or DiGraph

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    See Also
    --------
    modularity_matrix

    References
    ----------
    .. [1] M. E. J. Newman, "Modularity and community structure in networks",
       Proc. Natl. Acad. Sci. USA, vol. 103, pp. 8577-8582, 2006.
    """
    import scipy as sp

    if G.is_directed():
        return sp.linalg.eigvals(nx.directed_modularity_matrix(G))
    else:
        return sp.linalg.eigvals(nx.modularity_matrix(G))


@nx._dispatchable
def bethe_hessian_spectrum(G, r=None):
    """Returns eigenvalues of the Bethe Hessian matrix of G.

    Parameters
    ----------
    G : Graph
       A NetworkX Graph or DiGraph

    r : float
       Regularizer parameter

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    See Also
    --------
    bethe_hessian_matrix

    References
    ----------
    .. [1] A. Saade, F. Krzakala and L. Zdeborová
       "Spectral clustering of graphs with the bethe hessian",
       Advances in Neural Information Processing Systems. 2014.
    """
    import scipy as sp

    return sp.linalg.eigvalsh(nx.bethe_hessian_matrix(G, r).todense())
<|endoftext|>"""
Algebraic connectivity and Fiedler vectors of undirected graphs.
"""

from functools import partial

import networkx as nx
from networkx.utils import (
    not_implemented_for,
    np_random_state,
    reverse_cuthill_mckee_ordering,
)

__all__ = [
    "algebraic_connectivity",
    "fiedler_vector",
    "spectral_ordering",
    "spectral_bisection",
]


class _PCGSolver:
    """Preconditioned conjugate gradient method.

    To solve Ax = b:
        M = A.diagonal() # or some other preconditioner
        solver = _PCGSolver(lambda x: A * x, lambda x: M * x)
        x = solver.solve(b)

    The inputs A and M are functions which compute
    matrix multiplication on the argument.
    A - multiply by the matrix A in Ax=b
    M - multiply by M, the preconditioner surrogate for A

    Warning: There is no limit on number of iterations.
    """

    def __init__(self, A, M):
        self._A = A
        self._M = M

    def solve(self, B, tol):
        import numpy as np

        # Densifying step - can this be kept sparse?
        B = np.asarray(B)
        X = np.ndarray(B.shape, order="F")
        for j in range(B.shape[1]):
            X[:, j] = self._solve(B[:, j], tol)
        return X

    def _solve(self, b, tol):
        import numpy as np
        import scipy as sp

        A = self._A
        M = self._M
        tol *= sp.linalg.blas.dasum(b)
        # Initialize.
        x = np.zeros(b.shape)
        r = b.copy()
        z = M(r)
        rz = sp.linalg.blas.ddot(r, z)
        p = z.copy()
        # Iterate.
        while True:
            Ap = A(p)
            alpha = rz / sp.linalg.blas.ddot(p, Ap)
            x = sp.linalg.blas.daxpy(p, x, a=alpha)
            r = sp.linalg.blas.daxpy(Ap, r, a=-alpha)
            if sp.linalg.blas.dasum(r) < tol:
                return x
            z = M(r)
            beta = sp.linalg.blas.ddot(r, z)
            beta, rz = beta / rz, beta
            p = sp.linalg.blas.daxpy(p, z, a=beta)


class _LUSolver:
    """LU factorization.

    To solve Ax = b:
        solver = _LUSolver(A)
        x = solver.solve(b)

    optional argument `tol` on solve method is ignored but included
    to match _PCGsolver API.
    """

    def __init__(self, A):
        import scipy as sp

        self._LU = sp.sparse.linalg.splu(
            A,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"Equil": True, "SymmetricMode": True},
        )

    def solve(self, B, tol=None):
        import numpy as np

        B = np.asarray(B)
        X = np.ndarray(B.shape, order="F")
        for j in range(B.shape[1]):
            X[:, j] = self._LU.solve(B[:, j])
        return X


def _preprocess_graph(G, weight):
    """Compute edge weights and eliminate zero-weight edges."""
    if G.is_directed():
        H = nx.MultiGraph()
        H.add_nodes_from(G)
        H.add_weighted_edges_from(
            ((u, v, e.get(weight, 1.0)) for u, v, e in G.edges(data=True) if u != v),
            weight=weight,
        )
        G = H
    if not G.is_multigraph():
        edges = (
            (u, v, abs(e.get(weight, 1.0))) for u, v, e in G.edges(data=True) if u != v
        )
    else:
        edges = (
            (u, v, sum(abs(e.get(weight, 1.0)) for e in G[u][v].values()))
            for u, v in G.edges()
            if u != v
        )
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_weighted_edges_from((u, v, e) for u, v, e in edges if e != 0)
    return H


def _rcm_estimate(G, nodelist):
    """Estimate the Fiedler vector using the reverse Cuthill-McKee ordering."""
    import numpy as np

    G = G.subgraph(nodelist)
    order = reverse_cuthill_mckee_ordering(G)
    n = len(nodelist)
    index = dict(zip(nodelist, range(n)))
    x = np.ndarray(n, dtype=float)
    for i, u in enumerate(order):
        x[index[u]] = i
    x -= (n - 1) / 2.0
    return x


def _tracemin_fiedler(L, X, normalized, tol, method):
    """Compute the Fiedler vector of L using the TraceMIN-Fiedler algorithm.

    The Fiedler vector of a connected undirected graph is the eigenvector
    corresponding to the second smallest eigenvalue of the Laplacian matrix
    of the graph. This function starts with the Laplacian L, not the Graph.

    Parameters
    ----------
    L : Laplacian of a possibly weighted or normalized, but undirected graph

    X : Initial guess for a solution. Usually a matrix of random numbers.
        This function allows more than one column in X to identify more than
        one eigenvector if desired.

    normalized : bool
        Whether the normalized Laplacian matrix is used.

    tol : float
        Tolerance of relative residual in eigenvalue computation.
        Warning: There is no limit on number of iterations.

    method : string
        Should be 'tracemin_pcg' or 'tracemin_lu'.
        Otherwise exception is raised.

    Returns
    -------
    sigma, X : Two NumPy arrays of floats.
        The lowest eigenvalues and corresponding eigenvectors of L.
        The size of input X determines the size of these outputs.
        As this is for Fiedler vectors, the zero eigenvalue (and
        constant eigenvector) are avoided.
    """
    import numpy as np
    import scipy as sp

    n = X.shape[0]

    if normalized:
        # Form the normalized Laplacian matrix and determine the eigenvector of
        # its nullspace.
        e = np.sqrt(L.diagonal())
        # TODO: rm csr_array wrapper when spdiags array creation becomes available
        D = sp.sparse.csr_array(sp.sparse.spdiags(1 / e, 0, n, n, format="csr"))
        L = D @ L @ D
        e *= 1.0 / np.linalg.norm(e, 2)

    if normalized:

        def project(X):
            """Make X orthogonal to the nullspace of L."""
            X = np.asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= (X[:, j] @ e) * e

    else:

        def project(X):
            """Make X orthogonal to the nullspace of L."""
            X = np.asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= X[:, j].sum() / n

    if method == "tracemin_pcg":
        D = L.diagonal().astype(float)
        solver = _PCGSolver(lambda x: L @ x, lambda x: D * x)
    elif method == "tracemin_lu":
        # Convert A to CSC to suppress SparseEfficiencyWarning.
        A = sp.sparse.csc_array(L, dtype=float, copy=True)
        # Force A to be nonsingular. Since A is the Laplacian matrix of a
        # connected graph, its rank deficiency is one, and thus one diagonal
        # element needs to modified. Changing to infinity forces a zero in the
        # corresponding element in the solution.
        i = (A.indptr[1:] - A.indptr[:-1]).argmax()
        A[i, i] = np.inf
        solver = _LUSolver(A)
    else:
        raise nx.NetworkXError(f"Unknown linear system solver: {method}")

    # Initialize.
    Lnorm = abs(L).sum(axis=1).flatten().max()
    project(X)
    W = np.ndarray(X.shape, order="F")

    while True:
        # Orthonormalize X.
        X = np.linalg.qr(X)[0]
        # Compute iteration matrix H.
        W[:, :] = L @ X
        H = X.T @ W
        sigma, Y = sp.linalg.eigh(H, overwrite_a=True)
        # Compute the Ritz vectors.
        X = X @ Y
        # Test for convergence exploiting the fact that L * X == W * Y.
        res = sp.linalg.blas.dasum(W @ Y[:, 0] - sigma[0] * X[:, 0]) / Lnorm
        if res < tol:
            break
        # Compute X = L \ X / (X' * (L \ X)).
        # L \ X can have an arbitrary projection on the nullspace of L,
        # which will be eliminated.
        W[:, :] = solver.solve(X, tol)
        X = (sp.linalg.inv(W.T @ X) @ W.T).T  # Preserves Fortran storage order.
        project(X)

    return sigma, np.asarray(X)


def _get_fiedler_func(method):
    """Returns a function that solves the Fiedler eigenvalue problem."""
    import numpy as np

    if method == "tracemin":  # old style keyword <v2.1
        method = "tracemin_pcg"
    if method in ("tracemin_pcg", "tracemin_lu"):

        def find_fiedler(L, x, normalized, tol, seed):
            q = 1 if method == "tracemin_pcg" else min(4, L.shape[0] - 1)
            X = np.asarray(seed.normal(size=(q, L.shape[0]))).T
            sigma, X = _tracemin_fiedler(L, X, normalized, tol, method)
            return sigma[0], X[:, 0]

    elif method == "lanczos" or method == "lobpcg":

        def find_fiedler(L, x, normalized, tol, seed):
            import scipy as sp

            L = sp.sparse.csc_array(L, dtype=float)
            n = L.shape[0]
            if normalized:
                # TODO: rm csc_array wrapping when spdiags array becomes available
                D = sp.sparse.csc_array(
                    sp.sparse.spdiags(
                        1.0 / np.sqrt(L.diagonal()), [0], n, n, format="csc"
                    )
                )
                L = D @ L @ D
            if method == "lanczos" or n < 10:
                # Avoid LOBPCG when n < 10 due to
                # https://github.com/scipy/scipy/issues/3592
                # https://github.com/scipy/scipy/pull/3594
                sigma, X = sp.sparse.linalg.eigsh(
                    L, 2, which="SM", tol=tol, return_eigenvectors=True
                )
                return sigma[1], X[:, 1]
            else:
                X = np.asarray(np.atleast_2d(x).T)
                # TODO: rm csr_array wrapping when spdiags array becomes available
                M = sp.sparse.csr_array(sp.sparse.spdiags(1.0 / L.diagonal(), 0, n, n))
                Y = np.ones(n)
                if normalized:
                    Y /= D.diagonal()
                sigma, X = sp.sparse.linalg.lobpcg(
                    L, X, M=M, Y=np.atleast_2d(Y).T, tol=tol, maxiter=n, largest=False
                )
                return sigma[0], X[:, 0]

    else:
        raise nx.NetworkXError(f"unknown method {method!r}.")

    return find_fiedler


@not_implemented_for("directed")
@np_random_state(5)
@nx._dispatchable(edge_attrs="weight")
def algebraic_connectivity(
    G, weight="weight", normalized=False, tol=1e-8, method="tracemin_pcg", seed=None
):
    r"""Returns the algebraic connectivity of an undirected graph.

    The algebraic connectivity of a connected undirected graph is the second
    smallest eigenvalue of its Laplacian matrix.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    weight : object, optional (default: None)
        The data key used to determine the weight of each edge. If None, then
        each edge has unit weight.

    normalized : bool, optional (default: False)
        Whether the normalized Laplacian matrix is used.

    tol : float, optional (default: 1e-8)
        Tolerance of relative residual in eigenvalue computation.

    method : string, optional (default: 'tracemin_pcg')
        Method of eigenvalue computation. It must be one of the tracemin
        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
        or 'lobpcg' (LOBPCG).

        The TraceMIN algorithm uses a linear system solver. The following
        values allow specifying the solver to be used.

        =============== ========================================
        Value           Solver
        =============== ========================================
        'tracemin_pcg'  Preconditioned conjugate gradient method
        'tracemin_lu'   LU factorization
        =============== ========================================

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    algebraic_connectivity : float
        Algebraic connectivity.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    NetworkXError
        If G has less than two nodes.

    Notes
    -----
    Edge weights are interpreted by their absolute values. For MultiGraph's,
    weights of parallel edges are summed. Zero-weighted edges are ignored.

    See Also
    --------
    laplacian_matrix

    Examples
    --------
    For undirected graphs algebraic connectivity can tell us if a graph is connected or not
    `G` is connected iff  ``algebraic_connectivity(G) > 0``:

    >>> G = nx.complete_graph(5)
    >>> nx.algebraic_connectivity(G) > 0
    True
    >>> G.add_node(10)  # G is no longer connected
    >>> nx.algebraic_connectivity(G) > 0
    False

    """
    if len(G) < 2:
        raise nx.NetworkXError("graph has less than two nodes.")
    G = _preprocess_graph(G, weight)
    if not nx.is_connected(G):
        return 0.0

    L = nx.laplacian_matrix(G)
    if L.shape[0] == 2:
        return 2.0 * float(L[0, 0]) if not normalized else 2.0

    find_fiedler = _get_fiedler_func(method)
    x = None if method != "lobpcg" else _rcm_estimate(G, G)
    sigma, fiedler = find_fiedler(L, x, normalized, tol, seed)
    return float(sigma)


@not_implemented_for("directed")
@np_random_state(5)
@nx._dispatchable(edge_attrs="weight")
def fiedler_vector(
    G, weight="weight", normalized=False, tol=1e-8, method="tracemin_pcg", seed=None
):
    """Returns the Fiedler vector of a connected undirected graph.

    The Fiedler vector of a connected undirected graph is the eigenvector
    corresponding to the second smallest eigenvalue of the Laplacian matrix
    of the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    weight : object, optional (default: None)
        The data key used to determine the weight of each edge. If None, then
        each edge has unit weight.

    normalized : bool, optional (default: False)
        Whether the normalized Laplacian matrix is used.

    tol : float, optional (default: 1e-8)
        Tolerance of relative residual in eigenvalue computation.

    method : string, optional (default: 'tracemin_pcg')
        Method of eigenvalue computation. It must be one of the tracemin
        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
        or 'lobpcg' (LOBPCG).

        The TraceMIN algorithm uses a linear system solver. The following
        values allow specifying the solver to be used.

        =============== ========================================
        Value           Solver
        =============== ========================================
        'tracemin_pcg'  Preconditioned conjugate gradient method
        'tracemin_lu'   LU factorization
        =============== ========================================

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    fiedler_vector : NumPy array of floats.
        Fiedler vector.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    NetworkXError
        If G has less than two nodes or is not connected.

    Notes
    -----
    Edge weights are interpreted by their absolute values. For MultiGraph's,
    weights of parallel edges are summed. Zero-weighted edges are ignored.

    See Also
    --------
    laplacian_matrix

    Examples
    --------
    Given a connected graph the signs of the values in the Fiedler vector can be
    used to partition the graph into two components.

    >>> G = nx.barbell_graph(5, 0)
    >>> nx.fiedler_vector(G, normalized=True, seed=1)
    array([-0.32864129, -0.32864129, -0.32864129, -0.32864129, -0.26072899,
            0.26072899,  0.32864129,  0.32864129,  0.32864129,  0.32864129])

    The connected components are the two 5-node cliques of the barbell graph.
    """
    import numpy as np

    if len(G) < 2:
        raise nx.NetworkXError("graph has less than two nodes.")
    G = _preprocess_graph(G, weight)
    if not nx.is_connected(G):
        raise nx.NetworkXError("graph is not connected.")

    if len(G) == 2:
        return np.array([1.0, -1.0])

    find_fiedler = _get_fiedler_func(method)
    L = nx.laplacian_matrix(G)
    x = None if method != "lobpcg" else _rcm_estimate(G, G)
    sigma, fiedler = find_fiedler(L, x, normalized, tol, seed)
    return fiedler


@np_random_state(5)
@nx._dispatchable(edge_attrs="weight")
def spectral_ordering(
    G, weight="weight", normalized=False, tol=1e-8, method="tracemin_pcg", seed=None
):
    """Compute the spectral_ordering of a graph.

    The spectral ordering of a graph is an ordering of its nodes where nodes
    in the same weakly connected components appear contiguous and ordered by
    their corresponding elements in the Fiedler vector of the component.

    Parameters
    ----------
    G : NetworkX graph
        A graph.

    weight : object, optional (default: None)
        The data key used to determine the weight of each edge. If None, then
        each edge has unit weight.

    normalized : bool, optional (default: False)
        Whether the normalized Laplacian matrix is used.

    tol : float, optional (default: 1e-8)
        Tolerance of relative residual in eigenvalue computation.

    method : string, optional (default: 'tracemin_pcg')
        Method of eigenvalue computation. It must be one of the tracemin
        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
        or 'lobpcg' (LOBPCG).

        The TraceMIN algorithm uses a linear system solver. The following
        values allow specifying the solver to be used.

        =============== ========================================
        Value           Solver
        =============== ========================================
        'tracemin_pcg'  Preconditioned conjugate gradient method
        'tracemin_lu'   LU factorization
        =============== ========================================

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    spectral_ordering : NumPy array of floats.
        Spectral ordering of nodes.

    Raises
    ------
    NetworkXError
        If G is empty.

    Notes
    -----
    Edge weights are interpreted by their absolute values. For MultiGraph's,
    weights of parallel edges are summed. Zero-weighted edges are ignored.

    See Also
    --------
    laplacian_matrix
    """
    if len(G) == 0:
        raise nx.NetworkXError("graph is empty.")
    G = _preprocess_graph(G, weight)

    find_fiedler = _get_fiedler_func(method)
    order = []
    for component in nx.connected_components(G):
        size = len(component)
        if size > 2:
            L = nx.laplacian_matrix(G, component)
            x = None if method != "lobpcg" else _rcm_estimate(G, component)
            sigma, fiedler = find_fiedler(L, x, normalized, tol, seed)
            sort_info = zip(fiedler, range(size), component)
            order.extend(u for x, c, u in sorted(sort_info))
        else:
            order.extend(component)

    return order


@nx._dispatchable(edge_attrs="weight")
def spectral_bisection(
    G, weight="weight", normalized=False, tol=1e-8, method="tracemin_pcg", seed=None
):
    """Bisect the graph using the Fiedler vector.

    This method uses the Fiedler vector to bisect a graph.
    The partition is defined by the nodes which are associated with
    either positive or negative values in the vector.

    Parameters
    ----------
    G : NetworkX Graph

    weight : str, optional (default: weight)
        The data key used to determine the weight of each edge. If None, then
        each edge has unit weight.

    normalized : bool, optional (default: False)
        Whether the normalized Laplacian matrix is used.

    tol : float, optional (default: 1e-8)
        Tolerance of relative residual in eigenvalue computation.

    method : string, optional (default: 'tracemin_pcg')
        Method of eigenvalue computation. It must be one of the tracemin
        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
        or 'lobpcg' (LOBPCG).

        The TraceMIN algorithm uses a linear system solver. The following
        values allow specifying the solver to be used.

        =============== ========================================
        Value           Solver
        =============== ========================================
        'tracemin_pcg'  Preconditioned conjugate gradient method
        'tracemin_lu'   LU factorization
        =============== ========================================

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    bisection : tuple of sets
        Sets with the bisection of nodes

    Examples
    --------
    >>> G = nx.barbell_graph(3, 0)
    >>> nx.spectral_bisection(G)
    ({0, 1, 2}, {3, 4, 5})

    References
    ----------
    .. [1] M. E. J Newman 'Networks: An Introduction', pages 364-370
       Oxford University Press 2011.
    """
    import numpy as np

    v = nx.fiedler_vector(G, weight, normalized, tol, method, seed)
    nodes = np.array(list(G))
    pos_vals = v >= 0

    return set(nodes[~pos_vals].tolist()), set(nodes[pos_vals].tolist())
<|endoftext|>import asyncio
import enum
import sys
from types import TracebackType
from typing import Optional, Type, final


__version__ = "5.0.1"


__all__ = ("timeout", "timeout_at", "Timeout")


def timeout(delay: Optional[float]) -> "Timeout":
    """timeout context manager.

    Useful in cases when you want to apply timeout logic around block
    of code or in cases when asyncio.wait_for is not suitable. For example:

    >>> async with timeout(0.001):
    ...     async with aiohttp.get('https://github.com') as r:
    ...         await r.text()


    delay - value in seconds or None to disable timeout logic
    """
    loop = asyncio.get_running_loop()
    if delay is not None:
        deadline = loop.time() + delay  # type: Optional[float]
    else:
        deadline = None
    return Timeout(deadline, loop)


def timeout_at(deadline: Optional[float]) -> "Timeout":
    """Schedule the timeout at absolute time.

    deadline argument points on the time in the same clock system
    as loop.time().

    Please note: it is not POSIX time but a time with
    undefined starting base, e.g. the time of the system power on.

    >>> async with timeout_at(loop.time() + 10):
    ...     async with aiohttp.get('https://github.com') as r:
    ...         await r.text()


    """
    loop = asyncio.get_running_loop()
    return Timeout(deadline, loop)


class _State(enum.Enum):
    INIT = "INIT"
    ENTER = "ENTER"
    TIMEOUT = "TIMEOUT"
    EXIT = "EXIT"


if sys.version_info >= (3, 11):

    class _Expired:
        __slots__ = ("_val",)

        def __init__(self, val: bool) -> None:
            self._val = val

        def __call__(self) -> bool:
            return self._val

        def __bool__(self) -> bool:
            return self._val

        def __repr__(self) -> str:
            return repr(self._val)

        def __str__(self) -> str:
            return str(self._val)

    @final
    class Timeout(asyncio.Timeout):  # type: ignore[misc]
        # Supports full asyncio.Timeout API.
        # Also provides several asyncio_timeout specific methods
        # for backward compatibility.
        def __init__(
            self, deadline: Optional[float], loop: asyncio.AbstractEventLoop
        ) -> None:
            super().__init__(deadline)

        @property
        def expired(self) -> _Expired:
            # a hacky property hat can provide both roles:
            # timeout.expired()  from asyncio
            # timeout.expired    from asyncio_timeout
            return _Expired(super().expired())

        @property
        def deadline(self) -> Optional[float]:
            return self.when()

        def reject(self) -> None:
            """Reject scheduled timeout if any."""
            # cancel is maybe better name but
            # task.cancel() raises CancelledError in asyncio world.
            self.reschedule(None)

        def shift(self, delay: float) -> None:
            """Advance timeout on delay seconds.

            The delay can be negative.

            Raise RuntimeError if shift is called when deadline is not scheduled
            """
            deadline = self.when()
            if deadline is None:
                raise RuntimeError("cannot shift timeout if deadline is not scheduled")
            self.reschedule(deadline + delay)

        def update(self, deadline: float) -> None:
            """Set deadline to absolute value.

            deadline argument points on the time in the same clock system
            as loop.time().

            If new deadline is in the past the timeout is raised immediately.

            Please note: it is not POSIX time but a time with
            undefined starting base, e.g. the time of the system power on.
            """
            self.reschedule(deadline)

else:

    @final
    class Timeout:
        # Internal class, please don't instantiate it directly
        # Use timeout() and timeout_at() public factories instead.
        #
        # Implementation note: `async with timeout()` is preferred
        # over `with timeout()`.
        # While technically the Timeout class implementation
        # doesn't need to be async at all,
        # the `async with` statement explicitly points that
        # the context manager should be used from async function context.
        #
        # This design allows to avoid many silly misusages.
        #
        # TimeoutError is raised immediately when scheduled
        # if the deadline is passed.
        # The purpose is to time out as soon as possible
        # without waiting for the next await expression.

        __slots__ = ("_deadline", "_loop", "_state", "_timeout_handler", "_task")

        def __init__(
            self, deadline: Optional[float], loop: asyncio.AbstractEventLoop
        ) -> None:
            self._loop = loop
            self._state = _State.INIT

            self._task: Optional["asyncio.Task[object]"] = None
            self._timeout_handler = None  # type: Optional[asyncio.Handle]
            if deadline is None:
                self._deadline = None  # type: Optional[float]
            else:
                self.update(deadline)

        async def __aenter__(self) -> "Timeout":
            self._do_enter()
            return self

        async def __aexit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
        ) -> Optional[bool]:
            self._do_exit(exc_type)
            return None

        @property
        def expired(self) -> bool:
            """Is timeout expired during execution?"""
            return self._state == _State.TIMEOUT

        @property
        def deadline(self) -> Optional[float]:
            return self._deadline

        def reject(self) -> None:
            """Reject scheduled timeout if any."""
            # cancel is maybe better name but
            # task.cancel() raises CancelledError in asyncio world.
            if self._state not in (_State.INIT, _State.ENTER):
                raise RuntimeError(f"invalid state {self._state.value}")
            self._reject()

        def _reject(self) -> None:
            self._task = None
            if self._timeout_handler is not None:
                self._timeout_handler.cancel()
                self._timeout_handler = None

        def shift(self, delay: float) -> None:
            """Advance timeout on delay seconds.

            The delay can be negative.

            Raise RuntimeError if shift is called when deadline is not scheduled
            """
            deadline = self._deadline
            if deadline is None:
                raise RuntimeError("cannot shift timeout if deadline is not scheduled")
            self.update(deadline + delay)

        def update(self, deadline: float) -> None:
            """Set deadline to absolute value.

            deadline argument points on the time in the same clock system
            as loop.time().

            If new deadline is in the past the timeout is raised immediately.

            Please note: it is not POSIX time but a time with
            undefined starting base, e.g. the time of the system power on.
            """
            if self._state == _State.EXIT:
                raise RuntimeError("cannot reschedule after exit from context manager")
            if self._state == _State.TIMEOUT:
                raise RuntimeError("cannot reschedule expired timeout")
            if self._timeout_handler is not None:
                self._timeout_handler.cancel()
            self._deadline = deadline
            if self._state != _State.INIT:
                self._reschedule()

        def _reschedule(self) -> None:
            assert self._state == _State.ENTER
            deadline = self._deadline
            if deadline is None:
                return

            now = self._loop.time()
            if self._timeout_handler is not None:
                self._timeout_handler.cancel()

            self._task = asyncio.current_task()
            if deadline <= now:
                self._timeout_handler = self._loop.call_soon(self._on_timeout)
            else:
                self._timeout_handler = self._loop.call_at(deadline, self._on_timeout)

        def _do_enter(self) -> None:
            if self._state != _State.INIT:
                raise RuntimeError(f"invalid state {self._state.value}")
            self._state = _State.ENTER
            self._reschedule()

        def _do_exit(self, exc_type: Optional[Type[BaseException]]) -> None:
            if exc_type is asyncio.CancelledError and self._state == _State.TIMEOUT:
                assert self._task is not None
                self._timeout_handler = None
                self._task = None
                raise asyncio.TimeoutError
            # timeout has not expired
            self._state = _State.EXIT
            self._reject()
            return None

        def _on_timeout(self) -> None:
            assert self._task is not None
            self._task.cancel()
            self._state = _State.TIMEOUT
            # drop the reference early
            self._timeout_handler = None
<|endoftext|>from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from .specifiers import Specifier

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping


@dataclass
class Token:
    __slots__ = ("name", "position", "text")

    name: str
    text: str
    position: int


class ParserSyntaxError(Exception):
    """The provided source text could not be parsed correctly."""

    def __init__(
        self,
        message: str,
        *,
        source: str,
        span: tuple[int, int],
    ) -> None:
        self.span = span
        self.message = message
        self.source = source

        super().__init__()

    def __str__(self) -> str:
        marker = " " * self.span[0] + "~" * (self.span[1] - self.span[0]) + "^"
        return f"{self.message}\n    {self.source}\n    {marker}"


DEFAULT_RULES: dict[str, re.Pattern[str]] = {
    "LEFT_PARENTHESIS": re.compile(r"\("),
    "RIGHT_PARENTHESIS": re.compile(r"\)"),
    "LEFT_BRACKET": re.compile(r"\["),
    "RIGHT_BRACKET": re.compile(r"\]"),
    "SEMICOLON": re.compile(r";"),
    "COMMA": re.compile(r","),
    "QUOTED_STRING": re.compile(
        r"""
            (
                ('[^']*')
                |
                ("[^"]*")
            )
        """,
        re.VERBOSE,
    ),
    "OP": re.compile(r"(===|==|~=|!=|<=|>=|<|>)"),
    "BOOLOP": re.compile(r"\b(or|and)\b"),
    "IN": re.compile(r"\bin\b"),
    "NOT": re.compile(r"\bnot\b"),
    "VARIABLE": re.compile(
        r"""
            \b(
                python_version
                |python_full_version
                |os[._]name
                |sys[._]platform
                |platform_(release|system)
                |platform[._](version|machine|python_implementation)
                |python_implementation
                |implementation_(name|version)
                |extras?
                |dependency_groups
            )\b
        """,
        re.VERBOSE,
    ),
    "SPECIFIER": re.compile(
        Specifier._specifier_regex_str,
        re.VERBOSE | re.IGNORECASE,
    ),
    "AT": re.compile(r"\@"),
    "URL": re.compile(r"[^ \t]+"),
    "IDENTIFIER": re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9._-]*\b"),
    "VERSION_PREFIX_TRAIL": re.compile(r"\.\*"),
    "VERSION_LOCAL_LABEL_TRAIL": re.compile(r"\+[a-z0-9]+(?:[-_\.][a-z0-9]+)*"),
    "WS": re.compile(r"[ \t]+"),
    "END": re.compile(r"\Z"),
}


class Tokenizer:
    """Context-sensitive token parsing.

    Provides methods to examine the input stream to check whether the next token
    matches.
    """

    __slots__ = ("next_token", "position", "rules", "source")

    def __init__(
        self,
        source: str,
        *,
        rules: Mapping[str, re.Pattern[str]],
    ) -> None:
        self.source = source
        self.rules = rules
        self.next_token: Token | None = None
        self.position = 0

    def consume(self, name: str) -> None:
        """Move beyond provided token name, if at current position."""
        if self.check(name):
            self.read()

    def check(self, name: str, *, peek: bool = False) -> bool:
        """Check whether the next token has the provided name.

        By default, if the check succeeds, the token *must* be read before
        another check. If `peek` is set to `True`, the token is not loaded and
        would need to be checked again.
        """
        assert self.next_token is None, (
            f"Cannot check for {name!r}, already have {self.next_token!r}"
        )
        assert name in self.rules, f"Unknown token name: {name!r}"

        expression = self.rules[name]

        match = expression.match(self.source, self.position)
        if match is None:
            return False
        if not peek:
            self.next_token = Token(name, match[0], self.position)
        return True

    def expect(self, name: str, *, expected: str) -> Token:
        """Expect a certain token name next, failing with a syntax error otherwise.

        The token is read and returned.
        """
        if not self.check(name):
            raise self.raise_syntax_error(f"Expected {expected}")
        return self.read()

    def read(self) -> Token:
        """Consume the next token and return it."""
        token = self.next_token
        assert token is not None

        self.position += len(token.text)
        self.next_token = None

        return token

    def raise_syntax_error(
        self,
        message: str,
        *,
        span_start: int | None = None,
        span_end: int | None = None,
    ) -> NoReturn:
        """Raise ParserSyntaxError at the given position."""
        span = (
            self.position if span_start is None else span_start,
            self.position if span_end is None else span_end,
        )
        raise ParserSyntaxError(
            message,
            source=self.source,
            span=span,
        )

    @contextlib.contextmanager
    def enclosing_tokens(
        self, open_token: str, close_token: str, *, around: str
    ) -> Generator[None, None, None]:
        if self.check(open_token):
            open_position = self.position
            self.read()
        else:
            open_position = None

        yield

        if open_position is None:
            return

        if not self.check(close_token):
            self.raise_syntax_error(
                f"Expected matching {close_token} for {open_token}, after {around}",
                span_start=open_position,
            )

        self.read()
<|endoftext|># This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
from __future__ import annotations

from typing import TYPE_CHECKING

from ._parser import parse_requirement as _parse_requirement
from ._tokenizer import ParserSyntaxError
from .markers import Marker, _normalize_extra_values
from .specifiers import InvalidSpecifier, SpecifierSet
from .utils import canonicalize_name

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "InvalidRequirement",
    "Requirement",
]


def __dir__() -> list[str]:
    return __all__


class InvalidRequirement(ValueError):
    """
    An invalid requirement was found, users should refer to PEP 508.

    .. versionadded:: 16.1
    """


class Requirement:
    """Parse a requirement.

    Parse a given requirement string into its parts, such as name, specifier,
    URL, and extras. Raises InvalidRequirement on a badly-formed requirement
    string.

    .. versionadded:: 16.1

    .. versionchanged:: 22.0
        Added equality (``__eq__``) and hashing (``__hash__``) so requirements
        can be compared and stored in sets / dicts.

    .. versionchanged:: 23.2
        Equality and hashing began canonicalizing requirement names, so
        requirements whose names differ only by normalization (e.g.
        ``Requirement("Foo")`` vs ``Requirement("foo")``) now compare and hash
        equal.

    Instances are safe to serialize with :mod:`pickle`. They use a stable
    format so the same pickle can be loaded in future packaging releases.

    .. versionchanged:: 26.2

        Added a stable pickle format. Pickles created with packaging 26.2+ can
        be unpickled with future releases.  Backward compatibility with pickles
        from packaging < 26.2 is supported but may be removed in a future
        release.

    .. versionchanged:: 26.3

        The dedicated pickle support introduced in 26.2 did not preserve the
        specifier's explicit :attr:`~packaging.specifiers.SpecifierSet.prereleases`
        override; it is now included again.

        Equality and hashing normalize requirement names, extras, and
        equivalent specifiers. The string representation still preserves the
        parsed name and extras spelling.
    """

    # TODO: Can we test whether something is contained within a requirement?
    #       If so how do we do that? Do we need to test against the _name_ of
    #       the thing as well as the version? What about the markers?
    # TODO: Can we normalize the name and extra name?

    __slots__ = ("extras", "marker", "name", "specifier", "url")

    def __init__(self, requirement_string: str) -> None:
        try:
            parsed = _parse_requirement(requirement_string)
        except ParserSyntaxError as e:
            raise InvalidRequirement(str(e)) from e

        self.name: str = parsed.name
        self.url: str | None = parsed.url or None
        self.extras: set[str] = set(parsed.extras)
        try:
            self.specifier: SpecifierSet = SpecifierSet(parsed.specifier)
        except InvalidSpecifier as e:
            raise InvalidRequirement(str(e)) from e
        self.marker: Marker | None = None
        if parsed.marker is not None:
            self.marker = Marker.__new__(Marker)
            self.marker._markers = _normalize_extra_values(parsed.marker)

    def _iter_parts(self, name: str) -> Iterator[str]:
        yield name

        if self.extras:
            formatted_extras = ",".join(sorted(self.extras))
            yield f"[{formatted_extras}]"

        if self.specifier:
            yield str(self.specifier)

        if self.url:
            yield f" @ {self.url}"
            if self.marker:
                yield " "

        if self.marker:
            yield f"; {self.marker}"

    def __getstate__(self) -> tuple[str, bool | None]:
        # Return the requirement string for compactness and stability, paired
        # with the specifier's explicit prereleases override, which is not
        # captured by the string form. Re-parsed on load to reconstruct all
        # other fields.
        return (str(self), self.specifier._prereleases)

    def __setstate__(self, state: object) -> None:
        if isinstance(state, str):
            # Format (26.2): just the requirement string.
            requirement_string: str = state
            prereleases: bool | None = None
        elif (
            isinstance(state, tuple)
            and len(state) == 2
            and isinstance(state[0], str)
            and (state[1] is None or isinstance(state[1], bool))
        ):
            # New format (26.3+): (requirement string, specifier prereleases).
            requirement_string, prereleases = state
        elif isinstance(state, dict) and state.keys() >= set(self.__slots__):
            # Old format (packaging <= 26.1, no __slots__): plain __dict__.
            for key in self.__slots__:
                setattr(self, key, state[key])
            return
        else:
            raise TypeError(f"Cannot restore Requirement from {state!r}")

        try:
            tmp = Requirement(requirement_string)
        except InvalidRequirement as exc:
            raise TypeError(f"Cannot restore Requirement from {state!r}") from exc
        self.name = tmp.name
        self.url = tmp.url
        self.extras = tmp.extras
        self.specifier = tmp.specifier
        self.specifier._prereleases = prereleases
        self.marker = tmp.marker

    def __str__(self) -> str:
        return "".join(self._iter_parts(self.name))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({str(self)!r})>"

    def __hash__(self) -> int:
        # Mirror __eq__ by hashing the canonical specifier object rather than
        # its raw string. ``_iter_parts`` yields ``str(self.specifier)``, which
        # is non-canonical, so trailing-zero-equivalent requirements such as
        # ``foo==1.0.0`` and ``foo==1.0.0.0`` (which compare equal) would
        # otherwise hash differently, breaking the hash/__eq__ invariant.
        return hash(
            (
                canonicalize_name(self.name),
                frozenset(canonicalize_name(e) for e in self.extras),
                self.specifier,
                self.url,
                self.marker,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Requirement):
            return NotImplemented

        # Extras must be normalized before comparison as per PEP 685.
        self_extras = frozenset(canonicalize_name(e) for e in self.extras)
        other_extras = frozenset(canonicalize_name(e) for e in other.extras)
        return (
            canonicalize_name(self.name) == canonicalize_name(other.name)
            and self_extras == other_extras
            and self.specifier == other.specifier
            and self.url == other.url
            and self.marker == other.marker
        )
<|endoftext|># This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

"""Backward-compatibility shim for unpickling Version objects serialized before
packaging 26.1.

Old pickles reference ``packaging._structures.InfinityType`` and
``packaging._structures.NegativeInfinityType``.  This module provides minimal
stand-in classes so that ``pickle.loads()`` can resolve those references.
The deserialized objects are not used for comparisons — ``Version.__setstate__``
discards the stale ``_key`` cache and recomputes it from the core version fields.
"""

from __future__ import annotations


class InfinityType:
    """Stand-in for the removed ``InfinityType`` used in old comparison keys."""

    def __repr__(self) -> str:
        return "Infinity"


class NegativeInfinityType:
    """Stand-in for the removed ``NegativeInfinityType`` used in old comparison keys."""

    def __repr__(self) -> str:
        return "-Infinity"


Infinity = InfinityType()
NegativeInfinity = NegativeInfinityType()
<|endoftext|>from __future__ import annotations

import contextlib
import dataclasses
import sys
import typing

__all__ = ["ExceptionGroup"]


def __dir__() -> list[str]:
    return __all__


if sys.version_info >= (3, 11):  # pragma: no cover
    from builtins import ExceptionGroup
else:  # pragma: no cover

    class ExceptionGroup(Exception):
        """A minimal implementation of :external:exc:`ExceptionGroup` from Python 3.11.

        If :external:exc:`ExceptionGroup` is already defined by Python itself,
        that version is used instead.
        """

        message: str
        exceptions: list[Exception]

        def __init__(self, message: str, exceptions: list[Exception]) -> None:
            self.message = message
            self.exceptions = exceptions

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self.message!r}, {self.exceptions!r})"


@dataclasses.dataclass
class _ErrorCollector:
    """
    Collect errors into ExceptionGroups.

    Used like this:

        collector = _ErrorCollector()
        # Add a single exception
        collector.error(ValueError("one"))

        # Supports nesting, including combining ExceptionGroups
        with collector.collect():
            raise ValueError("two")
        collector.finalize("Found some errors")

    Since making a collector and then calling finalize later is a common pattern,
    a convenience method ``on_exit`` is provided.
    """

    errors: list[Exception] = dataclasses.field(default_factory=list, init=False)

    def finalize(self, msg: str) -> None:
        """Raise a group exception if there are any errors."""
        if self.errors:
            raise ExceptionGroup(msg, self.errors)

    @contextlib.contextmanager
    def on_exit(self, msg: str) -> typing.Generator[_ErrorCollector, None, None]:
        """
        Calls finalize if no uncollected errors were present.

        Uncollected errors are raised normally.
        """
        yield self
        self.finalize(msg)

    @contextlib.contextmanager
    def collect(self, *err_cls: type[Exception]) -> typing.Generator[None, None, None]:
        """
        Context manager to collect errors into the error list.

        Must be inside loops, as only one error can be collected at a time.
        """
        error_classes = err_cls or (Exception,)
        try:
            yield
        except ExceptionGroup as error:
            self.errors.extend(error.exceptions)
        except error_classes as error:
            self.errors.append(error)

    def error(
        self,
        error: Exception,
    ) -> None:
        """Add an error to the list."""
        self.errors.append(error)
<|endoftext|>from __future__ import annotations

import dataclasses
import re
import urllib.parse
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    import sys
    from collections.abc import Collection
    from urllib.parse import SplitResult

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

__all__ = [
    "ArchiveInfo",
    "DirInfo",
    "DirectUrl",
    "DirectUrlValidationError",
    "VcsInfo",
]


def __dir__() -> list[str]:
    return __all__


_T = TypeVar("_T")


class _FromMappingProtocol(Protocol):  # pragma: no cover
    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self: ...


_FromMappingProtocolT = TypeVar("_FromMappingProtocolT", bound=_FromMappingProtocol)


def _json_dict_factory(data: list[tuple[str, Any]]) -> dict[str, Any]:
    return {key: value for key, value in data if value is not None}


def _get(d: Mapping[str, Any], expected_type: type[_T], key: str) -> _T | None:
    """Get a value from the dictionary and verify it's the expected type."""
    if (value := d.get(key)) is None:
        return None
    if not isinstance(value, expected_type):
        raise DirectUrlValidationError(
            f"Unexpected type {type(value).__name__} "
            f"(expected {expected_type.__name__})",
            context=key,
        )
    return value


def _get_required(d: Mapping[str, Any], expected_type: type[_T], key: str) -> _T:
    """Get a required value from the dictionary and verify it's the expected type."""
    if (value := _get(d, expected_type, key)) is None:
        raise _DirectUrlRequiredKeyError(key)
    return value


def _get_object(
    d: Mapping[str, Any], target_type: type[_FromMappingProtocolT], key: str
) -> _FromMappingProtocolT | None:
    """Get a dictionary value from the dictionary and convert it to a dataclass."""
    if (value := _get(d, Mapping, key)) is None:  # type: ignore[type-abstract]
        return None
    try:
        return target_type._from_dict(value)
    except Exception as e:
        raise DirectUrlValidationError(e, context=key) from e


_PEP610_USER_PASS_ENV_VARS_REGEX = re.compile(
    r"^\$\{[A-Za-z0-9-_]+\}(:\$\{[A-Za-z0-9-_]+\})?$"
)


def _strip_auth_from_netloc(netloc: str, safe_user_passwords: Collection[str]) -> str:
    if "@" not in netloc:
        return netloc
    user_pass, netloc_no_user_pass = netloc.rsplit("@", 1)
    if user_pass in safe_user_passwords:
        return netloc
    if _PEP610_USER_PASS_ENV_VARS_REGEX.match(user_pass):
        return netloc
    return netloc_no_user_pass


def _strip_url(url: str, safe_user_passwords: Collection[str]) -> str:
    """url with user:password part removed unless it is formed with
    environment variables as specified in PEP 610, or it is a safe user:password
    such as `git`.
    """
    parsed_url = urllib.parse.urlsplit(url)
    netloc = _strip_auth_from_netloc(parsed_url.netloc, safe_user_passwords)
    return urllib.parse.urlunsplit(
        (
            parsed_url.scheme,
            netloc,
            parsed_url.path,
            parsed_url.query,
            parsed_url.fragment,
        )
    )


def _file_url_has_absolute_path(parsed_url: SplitResult) -> bool:
    return parsed_url.path.startswith("/")


class DirectUrlValidationError(Exception):
    """Raised when when input data is not spec-compliant.

    .. versionadded:: 26.1
    """

    context: str | None = None
    message: str

    def __init__(
        self,
        cause: str | Exception,
        *,
        context: str | None = None,
    ) -> None:
        if isinstance(cause, DirectUrlValidationError):
            if cause.context:
                self.context = (
                    f"{context}.{cause.context}" if context else cause.context
                )
            else:
                self.context = context  # pragma: no cover
            self.message = cause.message
        else:
            self.context = context
            self.message = str(cause)

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} in {self.context!r}"
        return self.message


class _DirectUrlRequiredKeyError(DirectUrlValidationError):
    def __init__(self, key: str) -> None:
        super().__init__("Missing required value", context=key)


@dataclasses.dataclass(frozen=True, init=False)
class VcsInfo:
    """The version control information of a :class:`DirectUrl`."""

    vcs: str
    commit_id: str
    requested_revision: str | None = None

    def __init__(
        self,
        *,
        vcs: str,
        commit_id: str,
        requested_revision: str | None = None,
    ) -> None:
        object.__setattr__(self, "vcs", vcs)
        object.__setattr__(self, "commit_id", commit_id)
        object.__setattr__(self, "requested_revision", requested_revision)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        # We can't validate vcs value because is not closed.
        return cls(
            vcs=_get_required(d, str, "vcs"),
            requested_revision=_get(d, str, "requested_revision"),
            commit_id=_get_required(d, str, "commit_id"),
        )


@dataclasses.dataclass(frozen=True, init=False)
class ArchiveInfo:
    """The archive information of a :class:`DirectUrl`."""

    hashes: Mapping[str, str] | None = None

    def __init__(
        self,
        *,
        hashes: Mapping[str, str] | None = None,
    ) -> None:
        object.__setattr__(self, "hashes", hashes)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        hashes = _get(d, Mapping, "hashes")  # type: ignore[type-abstract]
        if hashes is not None and not all(isinstance(h, str) for h in hashes.values()):
            raise DirectUrlValidationError(
                "Hash values must be strings", context="hashes"
            )
        legacy_hash = _get(d, str, "hash")
        if legacy_hash is not None:
            if "=" not in legacy_hash:
                raise DirectUrlValidationError(
                    "Invalid hash format (expected '<algorithm>=<hash>')",
                    context="hash",
                )
            hash_algorithm, hash_value = legacy_hash.split("=", 1)
            if hashes is None:
                # if `hashes` are not present, we can derive it from the legacy `hash`
                hashes = {hash_algorithm: hash_value}
            else:
                # if `hashes` are present, the legacy `hash` must match one of them
                if hash_algorithm not in hashes:
                    raise DirectUrlValidationError(
                        f"Algorithm {hash_algorithm!r} used in hash field "
                        f"is not present in hashes field",
                        context="hashes",
                    )
                if hashes[hash_algorithm] != hash_value:
                    raise DirectUrlValidationError(
                        f"Algorithm {hash_algorithm!r} used in hash field "
                        f"has different value in hashes field",
                        context="hash",
                    )
        return cls(hashes=hashes)


@dataclasses.dataclass(frozen=True, init=False)
class DirInfo:
    """The local directory information of a :class:`DirectUrl`."""

    editable: bool | None = None

    def __init__(
        self,
        *,
        editable: bool | None = None,
    ) -> None:
        object.__setattr__(self, "editable", editable)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls(
            editable=_get(d, bool, "editable"),
        )


@dataclasses.dataclass(frozen=True, init=False)
class DirectUrl:
    """A class representing a direct URL.

    .. versionadded:: 26.1
    """

    url: str
    archive_info: ArchiveInfo | None = None
    vcs_info: VcsInfo | None = None
    dir_info: DirInfo | None = None
    subdirectory: str | None = None  # XXX Path or str?

    def __init__(
        self,
        *,
        url: str,
        archive_info: ArchiveInfo | None = None,
        vcs_info: VcsInfo | None = None,
        dir_info: DirInfo | None = None,
        subdirectory: str | None = None,
    ) -> None:
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "archive_info", archive_info)
        object.__setattr__(self, "vcs_info", vcs_info)
        object.__setattr__(self, "dir_info", dir_info)
        object.__setattr__(self, "subdirectory", subdirectory)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        direct_url = cls(
            url=_get_required(d, str, "url"),
            archive_info=_get_object(d, ArchiveInfo, "archive_info"),
            vcs_info=_get_object(d, VcsInfo, "vcs_info"),
            dir_info=_get_object(d, DirInfo, "dir_info"),
            subdirectory=_get(d, str, "subdirectory"),
        )
        if (
            bool(direct_url.vcs_info)
            + bool(direct_url.archive_info)
            + bool(direct_url.dir_info)
        ) != 1:
            raise DirectUrlValidationError(
                "Exactly one of vcs_info, archive_info, dir_info must be present"
            )
        if direct_url.dir_info is not None:
            parsed_url = urllib.parse.urlsplit(direct_url.url)
            if parsed_url.scheme != "file":
                raise DirectUrlValidationError(
                    "URL scheme must be file:// when dir_info is present",
                    context="url",
                )
            if not _file_url_has_absolute_path(parsed_url):
                raise DirectUrlValidationError(
                    "File URL must be absolute when dir_info is present",
                    context="url",
                )
        # XXX subdirectory must be relative, can we, should we validate that here?
        return direct_url

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], /) -> Self:
        """Create and validate a DirectUrl instance from a JSON dictionary."""
        return cls._from_dict(d)

    def to_dict(
        self,
        *,
        generate_legacy_hash: bool = False,
        strip_user_password: bool = True,
        safe_user_passwords: Collection[str] = ("git",),
    ) -> Mapping[str, Any]:
        """Convert the DirectUrl instance to a JSON dictionary.

        :param generate_legacy_hash: If True, include a legacy `hash` field in
            `archive_info` for backward compatibility with tools that don't
            support the `hashes` field.
        :param strip_user_password: If True, strip user:password from the URL
            unless it is formed with environment variables as specified in PEP
            610, or it is a safe user:password such as `git`.
        :param safe_user_passwords: A collection of user:password strings that
            should not be stripped from the URL even if `strip_user_password` is
            True.
        """
        res = dataclasses.asdict(self, dict_factory=_json_dict_factory)
        if generate_legacy_hash and self.archive_info and self.archive_info.hashes:
            hash_algorithm, hash_value = next(iter(self.archive_info.hashes.items()))
            res["archive_info"]["hash"] = f"{hash_algorithm}={hash_value}"
        if strip_user_password:
            res["url"] = _strip_url(self.url, safe_user_passwords)
        return res

    def validate(self) -> None:
        """Validate the DirectUrl instance against the specification.

        Raises :class:`DirectUrlValidationError` if invalid.
        """
        self.from_dict(self.to_dict())
<|endoftext|># This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from __future__ import annotations

import functools
import operator
import os
import platform
import sys
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Callable, Literal, TypedDict, Union, cast

from ._parser import MarkerAtom, MarkerList, Op, Value, Variable
from ._parser import parse_marker as _parse_marker
from ._tokenizer import ParserSyntaxError
from .specifiers import InvalidSpecifier, Specifier
from .utils import canonicalize_name

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "Environment",
    "EvaluateContext",
    "InvalidMarker",
    "Marker",
    "UndefinedComparison",
    "UndefinedEnvironmentName",
    "default_environment",
]


def __dir__() -> list[str]:
    return __all__


Operator = Callable[[str, Union[str, AbstractSet[str]]], bool]
EvaluateContext = Literal["metadata", "lock_file", "requirement"]
"""A ``typing.Literal`` enumerating valid marker evaluation contexts.

Valid values for the ``context`` passed to :meth:`Marker.evaluate` are:

* ``"metadata"`` (for core metadata; default)
* ``"lock_file"`` (for lock files)
* ``"requirement"`` (i.e. all other situations)

.. versionadded:: 25.0
"""

MARKERS_ALLOWING_SET = {"extras", "dependency_groups"}
MARKERS_REQUIRING_VERSION = {
    "implementation_version",
    "platform_release",
    "python_full_version",
    "python_version",
}


class InvalidMarker(ValueError):
    """Raised when attempting to create a :class:`Marker` from invalid input.

    This error indicates that the given marker string does not conform to the
    :ref:`specification of dependency specifiers <pypug:dependency-specifiers>`.
    """


class UndefinedComparison(ValueError):
    """Raised when evaluating an unsupported marker comparison.

    This can happen when marker values are compared as versions but do not
    conform to the :ref:`specification of version specifiers
    <pypug:version-specifiers>`.
    """


class UndefinedEnvironmentName(KeyError):
    """Raised when evaluating a marker that references a missing environment key.

    Subclasses :class:`KeyError` so that code catching the bare ``KeyError`` that
    a missing environment lookup historically produced keeps working.

    .. versionchanged:: 26.3
        Now subclasses :class:`KeyError` (was :class:`ValueError`) and is raised by
        :meth:`Marker.evaluate` for missing environment keys, where a bare
        ``KeyError`` was raised before.
    """


class Environment(TypedDict):
    """
    A dictionary that represents a Python environment as captured by
    :func:`default_environment`. All fields are required.
    """

    implementation_name: str
    """The implementation's identifier, e.g. ``'cpython'``."""

    implementation_version: str
    """
    The implementation's version, e.g. ``'3.13.0a2'`` for CPython 3.13.0a2, or
    ``'7.3.13'`` for PyPy3.10 v7.3.13.
    """

    os_name: str
    """
    The value of :py:data:`os.name`. The name of the operating system dependent module
    imported, e.g. ``'posix'``.
    """

    platform_machine: str
    """
    Returns the machine type, e.g. ``'i386'``.

    An empty string if the value cannot be determined.
    """

    platform_release: str
    """
    The system's release, e.g. ``'2.2.0'`` or ``'NT'``.

    An empty string if the value cannot be determined.
    """

    platform_system: str
    """
    The system/OS name, e.g. ``'Linux'``, ``'Windows'`` or ``'Java'``.

    An empty string if the value cannot be determined.
    """

    platform_version: str
    """
    The system's release version, e.g. ``'#3 on degas'``.

