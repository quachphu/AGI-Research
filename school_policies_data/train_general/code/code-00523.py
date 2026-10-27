def _get_writer(file_or_filename, encoding):
    # returns text write method and release all resources after using
    try:
        write = file_or_filename.write
    except AttributeError:
        # file_or_filename is a file name
        if encoding.lower() == "unicode":
            encoding="utf-8"
        with open(file_or_filename, "w", encoding=encoding,
                  errors="xmlcharrefreplace") as file:
            yield file.write, encoding
    else:
        # file_or_filename is a file-like object
        # encoding determines if it is a text or binary writer
        if encoding.lower() == "unicode":
            # use a text writer as is
            yield write, getattr(file_or_filename, "encoding", None) or "utf-8"
        else:
            # wrap a binary writer with TextIOWrapper
            with contextlib.ExitStack() as stack:
                if isinstance(file_or_filename, io.BufferedIOBase):
                    file = file_or_filename
                elif isinstance(file_or_filename, io.RawIOBase):
                    file = io.BufferedWriter(file_or_filename)
                    # Keep the original file open when the BufferedWriter is
                    # destroyed
                    stack.callback(file.detach)
                else:
                    # This is to handle passed objects that aren't in the
                    # IOBase hierarchy, but just have a write method
                    file = io.BufferedIOBase()
                    file.writable = lambda: True
                    file.write = write
                    try:
                        # TextIOWrapper uses this methods to determine
                        # if BOM (for UTF-16, etc) should be added
                        file.seekable = file_or_filename.seekable
                        file.tell = file_or_filename.tell
                    except AttributeError:
                        pass
                file = io.TextIOWrapper(file,
                                        encoding=encoding,
                                        errors="xmlcharrefreplace",
                                        newline="\n")
                # Keep the original file open when the TextIOWrapper is
                # destroyed
                stack.callback(file.detach)
                yield file.write, encoding

def _namespaces(elem, default_namespace=None):
    # identify namespaces used in this tree

    # maps qnames to *encoded* prefix:local names
    qnames = {None: None}

    # maps uri:s to prefixes
    namespaces = {}
    if default_namespace:
        namespaces[default_namespace] = ""

    def add_qname(qname):
        # calculate serialized qname representation
        try:
            if qname[:1] == "{":
                uri, tag = qname[1:].rsplit("}", 1)
                prefix = namespaces.get(uri)
                if prefix is None:
                    prefix = _namespace_map.get(uri)
                    if prefix is None:
                        prefix = "ns%d" % len(namespaces)
                    if prefix != "xml":
                        namespaces[uri] = prefix
                if prefix:
                    qnames[qname] = "%s:%s" % (prefix, tag)
                else:
                    qnames[qname] = tag # default element
            else:
                if default_namespace:
                    # FIXME: can this be handled in XML 1.0?
                    raise ValueError(
                        "cannot use non-qualified names with "
                        "default_namespace option"
                        )
                qnames[qname] = qname
        except TypeError:
            _raise_serialization_error(qname)

    # populate qname and namespaces table
    for elem in elem.iter():
        tag = elem.tag
        if isinstance(tag, QName):
            if tag.text not in qnames:
                add_qname(tag.text)
        elif isinstance(tag, str):
            if tag not in qnames:
                add_qname(tag)
        elif tag is not None and tag is not Comment and tag is not PI:
            _raise_serialization_error(tag)
        for key, value in elem.items():
            if isinstance(key, QName):
                key = key.text
            if key not in qnames:
                add_qname(key)
            if isinstance(value, QName) and value.text not in qnames:
                add_qname(value.text)
        text = elem.text
        if isinstance(text, QName) and text.text not in qnames:
            add_qname(text.text)
    return qnames, namespaces

def _serialize_xml(write, elem, qnames, namespaces,
                   short_empty_elements, **kwargs):
    tag = elem.tag
    text = elem.text
    if tag is Comment:
        write("<!--%s-->" % text)
    elif tag is ProcessingInstruction:
        write("<?%s?>" % text)
    else:
        tag = qnames[tag]
        if tag is None:
            if text:
                write(_escape_cdata(text))
            for e in elem:
                _serialize_xml(write, e, qnames, None,
                               short_empty_elements=short_empty_elements)
        else:
            write("<" + tag)
            items = list(elem.items())
            if items or namespaces:
                if namespaces:
                    for v, k in sorted(namespaces.items(),
                                       key=lambda x: x[1]):  # sort on prefix
                        if k:
                            k = ":" + k
                        write(" xmlns%s=\"%s\"" % (
                            k,
                            _escape_attrib(v)
                            ))
                for k, v in items:
                    if isinstance(k, QName):
                        k = k.text
                    if isinstance(v, QName):
                        v = qnames[v.text]
                    else:
                        v = _escape_attrib(v)
                    write(" %s=\"%s\"" % (qnames[k], v))
            if text or len(elem) or not short_empty_elements:
                write(">")
                if text:
                    write(_escape_cdata(text))
                for e in elem:
                    _serialize_xml(write, e, qnames, None,
                                   short_empty_elements=short_empty_elements)
                write("</" + tag + ">")
            else:
                write(" />")
    if elem.tail:
        write(_escape_cdata(elem.tail))

HTML_EMPTY = ("area", "base", "basefont", "br", "col", "frame", "hr",
              "img", "input", "isindex", "link", "meta", "param")

try:
    HTML_EMPTY = set(HTML_EMPTY)
except NameError:
    pass

def _serialize_html(write, elem, qnames, namespaces, **kwargs):
    tag = elem.tag
    text = elem.text
    if tag is Comment:
        write("<!--%s-->" % _escape_cdata(text))
    elif tag is ProcessingInstruction:
        write("<?%s?>" % _escape_cdata(text))
    else:
        tag = qnames[tag]
        if tag is None:
            if text:
                write(_escape_cdata(text))
            for e in elem:
                _serialize_html(write, e, qnames, None)
        else:
            write("<" + tag)
            items = list(elem.items())
            if items or namespaces:
                if namespaces:
                    for v, k in sorted(namespaces.items(),
                                       key=lambda x: x[1]):  # sort on prefix
                        if k:
                            k = ":" + k
                        write(" xmlns%s=\"%s\"" % (
                            k,
                            _escape_attrib(v)
                            ))
                for k, v in items:
                    if isinstance(k, QName):
                        k = k.text
                    if isinstance(v, QName):
                        v = qnames[v.text]
                    else:
                        v = _escape_attrib_html(v)
                    # FIXME: handle boolean attributes
                    write(" %s=\"%s\"" % (qnames[k], v))
            write(">")
            ltag = tag.lower()
            if text:
                if ltag == "script" or ltag == "style":
                    write(text)
                else:
                    write(_escape_cdata(text))
            for e in elem:
                _serialize_html(write, e, qnames, None)
            if ltag not in HTML_EMPTY:
                write("</" + tag + ">")
    if elem.tail:
        write(_escape_cdata(elem.tail))

def _serialize_text(write, elem):
    for part in elem.itertext():
        write(part)
    if elem.tail:
        write(elem.tail)

_serialize = {
    "xml": _serialize_xml,
    "html": _serialize_html,
    "text": _serialize_text,
# this optional method is imported at the end of the module
#   "c14n": _serialize_c14n,
}


def register_namespace(prefix, uri):
    """Register a namespace prefix.

    The registry is global, and any existing mapping for either the
    given prefix or the namespace URI will be removed.

    *prefix* is the namespace prefix, *uri* is a namespace uri. Tags and
    attributes in this namespace will be serialized with prefix if possible.

    ValueError is raised if prefix is reserved or is invalid.

    """
    if re.match(r"ns\d+$", prefix):
        raise ValueError("Prefix format reserved for internal use")
    for k, v in list(_namespace_map.items()):
        if k == uri or v == prefix:
            del _namespace_map[k]
    _namespace_map[uri] = prefix

_namespace_map = {
    # "well-known" namespace prefixes
    "http://www.w3.org/XML/1998/namespace": "xml",
    "http://www.w3.org/1999/xhtml": "html",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
    "http://schemas.xmlsoap.org/wsdl/": "wsdl",
    # xml schema
    "http://www.w3.org/2001/XMLSchema": "xs",
    "http://www.w3.org/2001/XMLSchema-instance": "xsi",
    # dublin core
    "http://purl.org/dc/elements/1.1/": "dc",
}
# For tests and troubleshooting
register_namespace._namespace_map = _namespace_map

def _raise_serialization_error(text):
    raise TypeError(
        "cannot serialize %r (type %s)" % (text, type(text).__name__)
        )

def _escape_cdata(text):
    # escape character data
    try:
        # it's worth avoiding do-nothing calls for strings that are
        # shorter than 500 characters, or so.  assume that's, by far,
        # the most common case in most applications.
        if "&" in text:
            text = text.replace("&", "&amp;")
        if "<" in text:
            text = text.replace("<", "&lt;")
        if ">" in text:
            text = text.replace(">", "&gt;")
        return text
    except (TypeError, AttributeError):
        _raise_serialization_error(text)

def _escape_attrib(text):
    # escape attribute value
    try:
        if "&" in text:
            text = text.replace("&", "&amp;")
        if "<" in text:
            text = text.replace("<", "&lt;")
        if ">" in text:
            text = text.replace(">", "&gt;")
        if "\"" in text:
            text = text.replace("\"", "&quot;")
        # Although section 2.11 of the XML specification states that CR or
        # CR LN should be replaced with just LN, it applies only to EOLNs
        # which take part of organizing file into lines. Within attributes,
        # we are replacing these with entity numbers, so they do not count.
        # http://www.w3.org/TR/REC-xml/#sec-line-ends
        # The current solution, contained in following six lines, was
        # discussed in issue 17582 and 39011.
        if "\r" in text:
            text = text.replace("\r", "&#13;")
        if "\n" in text:
            text = text.replace("\n", "&#10;")
        if "\t" in text:
            text = text.replace("\t", "&#09;")
        return text
    except (TypeError, AttributeError):
        _raise_serialization_error(text)

def _escape_attrib_html(text):
    # escape attribute value
    try:
        if "&" in text:
            text = text.replace("&", "&amp;")
        if ">" in text:
            text = text.replace(">", "&gt;")
        if "\"" in text:
            text = text.replace("\"", "&quot;")
        return text
    except (TypeError, AttributeError):
        _raise_serialization_error(text)

# --------------------------------------------------------------------

def tostring(element, encoding=None, method=None, *,
             xml_declaration=None, default_namespace=None,
             short_empty_elements=True):
    """Generate string representation of XML element.

    All subelements are included.  If encoding is "unicode", a string
    is returned. Otherwise a bytestring is returned.

    *element* is an Element instance, *encoding* is an optional output
    encoding defaulting to US-ASCII, *method* is an optional output which can
    be one of "xml" (default), "html", "text" or "c14n", *default_namespace*
    sets the default XML namespace (for "xmlns").

    Returns an (optionally) encoded string containing the XML data.

    """
    stream = io.StringIO() if encoding == 'unicode' else io.BytesIO()
    ElementTree(element).write(stream, encoding,
                               xml_declaration=xml_declaration,
                               default_namespace=default_namespace,
                               method=method,
                               short_empty_elements=short_empty_elements)
    return stream.getvalue()

class _ListDataStream(io.BufferedIOBase):
    """An auxiliary stream accumulating into a list reference."""
    def __init__(self, lst):
        self.lst = lst

    def writable(self):
        return True

    def seekable(self):
        return True

    def write(self, b):
        self.lst.append(b)

    def tell(self):
        return len(self.lst)

def tostringlist(element, encoding=None, method=None, *,
                 xml_declaration=None, default_namespace=None,
                 short_empty_elements=True):
    lst = []
    stream = _ListDataStream(lst)
    ElementTree(element).write(stream, encoding,
                               xml_declaration=xml_declaration,
                               default_namespace=default_namespace,
                               method=method,
                               short_empty_elements=short_empty_elements)
    return lst


def dump(elem):
    """Write element tree or element structure to sys.stdout.

    This function should be used for debugging only.

    *elem* is either an ElementTree, or a single Element.  The exact output
    format is implementation dependent.  In this version, it's written as an
    ordinary XML file.

    """
    # debugging
    if not isinstance(elem, ElementTree):
        elem = ElementTree(elem)
    elem.write(sys.stdout, encoding="unicode")
    tail = elem.getroot().tail
    if not tail or tail[-1] != "\n":
        sys.stdout.write("\n")


def indent(tree, space="  ", level=0):
    """Indent an XML document by inserting newlines and indentation space
    after elements.

    *tree* is the ElementTree or Element to modify.  The (root) element
    itself will not be changed, but the tail text of all elements in its
    subtree will be adapted.

    *space* is the whitespace to insert for each indentation level, two
    space characters by default.

    *level* is the initial indentation level. Setting this to a higher
    value than 0 can be used for indenting subtrees that are more deeply
    nested inside of a document.
    """
    if isinstance(tree, ElementTree):
        tree = tree.getroot()
    if level < 0:
        raise ValueError(f"Initial indentation level must be >= 0, got {level}")
    if not len(tree):
        return

    # Reduce the memory consumption by reusing indentation strings.
    indentations = ["\n" + level * space]

    def _indent_children(elem, level):
        # Start a new indentation level for the first child.
        child_level = level + 1
        try:
            child_indentation = indentations[child_level]
        except IndexError:
            child_indentation = indentations[level] + space
            indentations.append(child_indentation)

        if not elem.text or not elem.text.strip():
            elem.text = child_indentation

        for child in elem:
            if len(child):
                _indent_children(child, child_level)
            if not child.tail or not child.tail.strip():
                child.tail = child_indentation

        # Dedent after the last child by overwriting the previous indentation.
        if not child.tail.strip():
            child.tail = indentations[level]

    _indent_children(tree, 0)


# --------------------------------------------------------------------
# parsing


def parse(source, parser=None):
    """Parse XML document into element tree.

    *source* is a filename or file object containing XML data,
    *parser* is an optional parser instance defaulting to XMLParser.

    Return an ElementTree instance.

    """
    tree = ElementTree()
    tree.parse(source, parser)
    return tree


def iterparse(source, events=None, parser=None):
    """Incrementally parse XML document into ElementTree.

    This class also reports what's going on to the user based on the
    *events* it is initialized with.  The supported events are the strings
    "start", "end", "start-ns" and "end-ns" (the "ns" events are used to get
    detailed namespace information).  If *events* is omitted, only
    "end" events are reported.

    *source* is a filename or file object containing XML data, *events* is
    a list of events to report back, *parser* is an optional parser instance.

    Returns an iterator providing (event, elem) pairs.

    """
    # Use the internal, undocumented _parser argument for now; When the
    # parser argument of iterparse is removed, this can be killed.
    pullparser = XMLPullParser(events=events, _parser=parser)

    def iterator(source):
        close_source = False
        try:
            if not hasattr(source, "read"):
                source = open(source, "rb")
                close_source = True
            yield None
            while True:
                yield from pullparser.read_events()
                # load event buffer
                data = source.read(16 * 1024)
                if not data:
                    break
                pullparser.feed(data)
            root = pullparser._close_and_return_root()
            yield from pullparser.read_events()
            it.root = root
        finally:
            if close_source:
                source.close()

    class IterParseIterator(collections.abc.Iterator):
        __next__ = iterator(source).__next__
    it = IterParseIterator()
    it.root = None
    del iterator, IterParseIterator

    next(it)
    return it


class XMLPullParser:

    def __init__(self, events=None, *, _parser=None):
        # The _parser argument is for internal use only and must not be relied
        # upon in user code. It will be removed in a future release.
        # See https://bugs.python.org/issue17741 for more details.

        self._events_queue = collections.deque()
        self._parser = _parser or XMLParser(target=TreeBuilder())
        # wire up the parser for event reporting
        if events is None:
            events = ("end",)
        self._parser._setevents(self._events_queue, events)

    def feed(self, data):
        """Feed encoded data to parser."""
        if self._parser is None:
            raise ValueError("feed() called after end of stream")
        if data:
            try:
                self._parser.feed(data)
            except SyntaxError as exc:
                self._events_queue.append(exc)

    def _close_and_return_root(self):
        # iterparse needs this to set its root attribute properly :(
        root = self._parser.close()
        self._parser = None
        return root

    def close(self):
        """Finish feeding data to parser.

        Unlike XMLParser, does not return the root element. Use
        read_events() to consume elements from XMLPullParser.
        """
        self._close_and_return_root()

    def read_events(self):
        """Return an iterator over currently available (event, elem) pairs.

        Events are consumed from the internal event queue as they are
        retrieved from the iterator.
        """
        events = self._events_queue
        while events:
            event = events.popleft()
            if isinstance(event, Exception):
                raise event
            else:
                yield event


def XML(text, parser=None):
    """Parse XML document from string constant.

    This function can be used to embed "XML Literals" in Python code.

    *text* is a string containing XML data, *parser* is an
    optional parser instance, defaulting to the standard XMLParser.

    Returns an Element instance.

    """
    if not parser:
        parser = XMLParser(target=TreeBuilder())
    parser.feed(text)
    return parser.close()


def XMLID(text, parser=None):
    """Parse XML document from string constant for its IDs.

    *text* is a string containing XML data, *parser* is an
    optional parser instance, defaulting to the standard XMLParser.

    Returns an (Element, dict) tuple, in which the
    dict maps element id:s to elements.

    """
    if not parser:
        parser = XMLParser(target=TreeBuilder())
    parser.feed(text)
    tree = parser.close()
    ids = {}
    for elem in tree.iter():
        id = elem.get("id")
        if id:
            ids[id] = elem
    return tree, ids

# Parse XML document from string constant.  Alias for XML().
fromstring = XML

def fromstringlist(sequence, parser=None):
    """Parse XML document from sequence of string fragments.

    *sequence* is a list of other sequence, *parser* is an optional parser
    instance, defaulting to the standard XMLParser.

    Returns an Element instance.

    """
    if not parser:
        parser = XMLParser(target=TreeBuilder())
    for text in sequence:
        parser.feed(text)
    return parser.close()

# --------------------------------------------------------------------


class TreeBuilder:
    """Generic element structure builder.

    This builder converts a sequence of start, data, and end method
    calls to a well-formed element structure.

    You can use this class to build an element structure using a custom XML
    parser, or a parser for some other XML-like format.

    *element_factory* is an optional element factory which is called
    to create new Element instances, as necessary.

    *comment_factory* is a factory to create comments to be used instead of
    the standard factory.  If *insert_comments* is false (the default),
    comments will not be inserted into the tree.

    *pi_factory* is a factory to create processing instructions to be used
    instead of the standard factory.  If *insert_pis* is false (the default),
    processing instructions will not be inserted into the tree.
    """
    def __init__(self, element_factory=None, *,
                 comment_factory=None, pi_factory=None,
                 insert_comments=False, insert_pis=False):
        self._data = [] # data collector
        self._elem = [] # element stack
        self._last = None # last element
        self._root = None # root element
        self._tail = None # true if we're after an end tag
        if comment_factory is None:
            comment_factory = Comment
        self._comment_factory = comment_factory
        self.insert_comments = insert_comments
        if pi_factory is None:
            pi_factory = ProcessingInstruction
        self._pi_factory = pi_factory
        self.insert_pis = insert_pis
        if element_factory is None:
            element_factory = Element
        self._factory = element_factory

    def close(self):
        """Flush builder buffers and return toplevel document Element."""
        assert len(self._elem) == 0, "missing end tags"
        assert self._root is not None, "missing toplevel element"
        return self._root

    def _flush(self):
        if self._data:
            if self._last is not None:
                text = "".join(self._data)
                if self._tail:
                    assert self._last.tail is None, "internal error (tail)"
                    self._last.tail = text
                else:
                    assert self._last.text is None, "internal error (text)"
                    self._last.text = text
            self._data = []

    def data(self, data):
        """Add text to current element."""
        self._data.append(data)

    def start(self, tag, attrs):
        """Open new element and return it.

        *tag* is the element name, *attrs* is a dict containing element
        attributes.

        """
        self._flush()
        self._last = elem = self._factory(tag, attrs)
        if self._elem:
            self._elem[-1].append(elem)
        elif self._root is None:
            self._root = elem
        self._elem.append(elem)
        self._tail = 0
        return elem

    def end(self, tag):
        """Close and return current Element.

        *tag* is the element name.

        """
        self._flush()
        self._last = self._elem.pop()
        assert self._last.tag == tag,\
               "end tag mismatch (expected %s, got %s)" % (
                   self._last.tag, tag)
        self._tail = 1
        return self._last

    def comment(self, text):
        """Create a comment using the comment_factory.

        *text* is the text of the comment.
        """
        return self._handle_single(
            self._comment_factory, self.insert_comments, text)

    def pi(self, target, text=None):
        """Create a processing instruction using the pi_factory.

        *target* is the target name of the processing instruction.
        *text* is the data of the processing instruction, or ''.
        """
        return self._handle_single(
            self._pi_factory, self.insert_pis, target, text)

    def _handle_single(self, factory, insert, *args):
        elem = factory(*args)
        if insert:
            self._flush()
            self._last = elem
            if self._elem:
                self._elem[-1].append(elem)
            self._tail = 1
        return elem


# also see ElementTree and TreeBuilder
class XMLParser:
    """Element structure builder for XML source data based on the expat parser.

    *target* is an optional target object which defaults to an instance of the
    standard TreeBuilder class, *encoding* is an optional encoding string
    which if given, overrides the encoding specified in the XML file:
    http://www.iana.org/assignments/character-sets

    """

    def __init__(self, *, target=None, encoding=None):
        try:
            from xml.parsers import expat
        except ImportError:
            try:
                import pyexpat as expat
            except ImportError:
                raise ImportError(
                    "No module named expat; use SimpleXMLTreeBuilder instead"
                    )
        parser = expat.ParserCreate(encoding, "}")
        if target is None:
            target = TreeBuilder()
        # underscored names are provided for compatibility only
        self.parser = self._parser = parser
        self.target = self._target = target
        self._error = expat.error
        self._names = {} # name memo cache
        # main callbacks
        parser.DefaultHandlerExpand = self._default
        if hasattr(target, 'start'):
            parser.StartElementHandler = self._start
        if hasattr(target, 'end'):
            parser.EndElementHandler = self._end
        if hasattr(target, 'start_ns'):
            parser.StartNamespaceDeclHandler = self._start_ns
        if hasattr(target, 'end_ns'):
            parser.EndNamespaceDeclHandler = self._end_ns
        if hasattr(target, 'data'):
            parser.CharacterDataHandler = target.data
        # miscellaneous callbacks
        if hasattr(target, 'comment'):
            parser.CommentHandler = target.comment
        if hasattr(target, 'pi'):
            parser.ProcessingInstructionHandler = target.pi
        # Configure pyexpat: buffering, new-style attribute handling.
        parser.buffer_text = 1
        parser.ordered_attributes = 1
        self._doctype = None
        self.entity = {}
        try:
            self.version = "Expat %d.%d.%d" % expat.version_info
        except AttributeError:
            pass # unknown

    def _setevents(self, events_queue, events_to_report):
        # Internal API for XMLPullParser
        # events_to_report: a list of events to report during parsing (same as
        # the *events* of XMLPullParser's constructor.
        # events_queue: a list of actual parsing events that will be populated
        # by the underlying parser.
        #
        parser = self._parser
        append = events_queue.append
        for event_name in events_to_report:
            if event_name == "start":
                parser.ordered_attributes = 1
                def handler(tag, attrib_in, event=event_name, append=append,
                            start=self._start):
                    append((event, start(tag, attrib_in)))
                parser.StartElementHandler = handler
            elif event_name == "end":
                def handler(tag, event=event_name, append=append,
                            end=self._end):
                    append((event, end(tag)))
                parser.EndElementHandler = handler
            elif event_name == "start-ns":
                # TreeBuilder does not implement .start_ns()
                if hasattr(self.target, "start_ns"):
                    def handler(prefix, uri, event=event_name, append=append,
                                start_ns=self._start_ns):
                        append((event, start_ns(prefix, uri)))
                else:
                    def handler(prefix, uri, event=event_name, append=append):
                        append((event, (prefix or '', uri or '')))
                parser.StartNamespaceDeclHandler = handler
            elif event_name == "end-ns":
                # TreeBuilder does not implement .end_ns()
                if hasattr(self.target, "end_ns"):
                    def handler(prefix, event=event_name, append=append,
                                end_ns=self._end_ns):
                        append((event, end_ns(prefix)))
                else:
                    def handler(prefix, event=event_name, append=append):
                        append((event, None))
                parser.EndNamespaceDeclHandler = handler
            elif event_name == 'comment':
                def handler(text, event=event_name, append=append, self=self):
                    append((event, self.target.comment(text)))
                parser.CommentHandler = handler
            elif event_name == 'pi':
                def handler(pi_target, data, event=event_name, append=append,
                            self=self):
                    append((event, self.target.pi(pi_target, data)))
                parser.ProcessingInstructionHandler = handler
            else:
                raise ValueError("unknown event %r" % event_name)

    def _raiseerror(self, value):
        err = ParseError(value)
        err.code = value.code
        err.position = value.lineno, value.offset
        raise err

    def _fixname(self, key):
        # expand qname, and convert name string to ascii, if possible
        try:
            name = self._names[key]
        except KeyError:
            name = key
            if "}" in name:
                name = "{" + name
            self._names[key] = name
        return name

    def _start_ns(self, prefix, uri):
        return self.target.start_ns(prefix or '', uri or '')

    def _end_ns(self, prefix):
        return self.target.end_ns(prefix or '')

    def _start(self, tag, attr_list):
        # Handler for expat's StartElementHandler. Since ordered_attributes
        # is set, the attributes are reported as a list of alternating
        # attribute name,value.
        fixname = self._fixname
        tag = fixname(tag)
        attrib = {}
        if attr_list:
            for i in range(0, len(attr_list), 2):
                attrib[fixname(attr_list[i])] = attr_list[i+1]
        return self.target.start(tag, attrib)

    def _end(self, tag):
        return self.target.end(self._fixname(tag))

    def _default(self, text):
        prefix = text[:1]
        if prefix == "&":
            # deal with undefined entities
            try:
                data_handler = self.target.data
            except AttributeError:
                return
            try:
                data_handler(self.entity[text[1:-1]])
            except KeyError:
                from xml.parsers import expat
                err = expat.error(
                    "undefined entity %s: line %d, column %d" %
                    (text, self.parser.ErrorLineNumber,
                    self.parser.ErrorColumnNumber)
                    )
                err.code = 11 # XML_ERROR_UNDEFINED_ENTITY
                err.lineno = self.parser.ErrorLineNumber
                err.offset = self.parser.ErrorColumnNumber
                raise err
        elif prefix == "<" and text[:9] == "<!DOCTYPE":
            self._doctype = [] # inside a doctype declaration
        elif self._doctype is not None:
            # parse doctype contents
            if prefix == ">":
                self._doctype = None
                return
            text = text.strip()
            if not text:
                return
            self._doctype.append(text)
            n = len(self._doctype)
            if n > 2:
                type = self._doctype[1]
                if type == "PUBLIC" and n == 4:
                    name, type, pubid, system = self._doctype
                    if pubid:
                        pubid = pubid[1:-1]
                elif type == "SYSTEM" and n == 3:
                    name, type, system = self._doctype
                    pubid = None
                else:
                    return
                if hasattr(self.target, "doctype"):
                    self.target.doctype(name, pubid, system[1:-1])
                elif hasattr(self, "doctype"):
                    warnings.warn(
                        "The doctype() method of XMLParser is ignored.  "
                        "Define doctype() method on the TreeBuilder target.",
                        RuntimeWarning)

                self._doctype = None

    def feed(self, data):
        """Feed encoded data to parser."""
        try:
            self.parser.Parse(data, False)
        except self._error as v:
            self._raiseerror(v)

    def close(self):
        """Finish feeding data to parser and return element structure."""
        try:
            self.parser.Parse(b"", True) # end of data
        except self._error as v:
            self._raiseerror(v)
        try:
            close_handler = self.target.close
        except AttributeError:
            pass
        else:
            return close_handler()
        finally:
            # get rid of circular references
            del self.parser, self._parser
            del self.target, self._target


# --------------------------------------------------------------------
# C14N 2.0

def canonicalize(xml_data=None, *, out=None, from_file=None, **options):
    """Convert XML to its C14N 2.0 serialised form.

    If *out* is provided, it must be a file or file-like object that receives
    the serialised canonical XML output (text, not bytes) through its ``.write()``
    method.  To write to a file, open it in text mode with encoding "utf-8".
    If *out* is not provided, this function returns the output as text string.

    Either *xml_data* (an XML string) or *from_file* (a file path or
    file-like object) must be provided as input.

    The configuration options are the same as for the ``C14NWriterTarget``.
    """
    if xml_data is None and from_file is None:
        raise ValueError("Either 'xml_data' or 'from_file' must be provided as input")
    sio = None
    if out is None:
        sio = out = io.StringIO()

    parser = XMLParser(target=C14NWriterTarget(out.write, **options))

    if xml_data is not None:
        parser.feed(xml_data)
        parser.close()
    elif from_file is not None:
        parse(from_file, parser=parser)

    return sio.getvalue() if sio is not None else None


_looks_like_prefix_name = re.compile(r'^\w+:\w+$', re.UNICODE).match


class C14NWriterTarget:
    """
    Canonicalization writer target for the XMLParser.

    Serialises parse events to XML C14N 2.0.

    The *write* function is used for writing out the resulting data stream
    as text (not bytes).  To write to a file, open it in text mode with encoding
    "utf-8" and pass its ``.write`` method.

    Configuration options:

    - *with_comments*: set to true to include comments
    - *strip_text*: set to true to strip whitespace before and after text content
    - *rewrite_prefixes*: set to true to replace namespace prefixes by "n{number}"
    - *qname_aware_tags*: a set of qname aware tag names in which prefixes
                          should be replaced in text content
    - *qname_aware_attrs*: a set of qname aware attribute names in which prefixes
                           should be replaced in text content
    - *exclude_attrs*: a set of attribute names that should not be serialised
    - *exclude_tags*: a set of tag names that should not be serialised
    """
    def __init__(self, write, *,
                 with_comments=False, strip_text=False, rewrite_prefixes=False,
                 qname_aware_tags=None, qname_aware_attrs=None,
                 exclude_attrs=None, exclude_tags=None):
        self._write = write
        self._data = []
        self._with_comments = with_comments
        self._strip_text = strip_text
        self._exclude_attrs = set(exclude_attrs) if exclude_attrs else None
        self._exclude_tags = set(exclude_tags) if exclude_tags else None

        self._rewrite_prefixes = rewrite_prefixes
        if qname_aware_tags:
            self._qname_aware_tags = set(qname_aware_tags)
        else:
            self._qname_aware_tags = None
        if qname_aware_attrs:
            self._find_qname_aware_attrs = set(qname_aware_attrs).intersection
        else:
            self._find_qname_aware_attrs = None

        # Stack with globally and newly declared namespaces as (uri, prefix) pairs.
        self._declared_ns_stack = [[
            ("http://www.w3.org/XML/1998/namespace", "xml"),
        ]]
        # Stack with user declared namespace prefixes as (uri, prefix) pairs.
        self._ns_stack = []
        if not rewrite_prefixes:
            self._ns_stack.append(list(_namespace_map.items()))
        self._ns_stack.append([])
        self._prefix_map = {}
        self._preserve_space = [False]
        self._pending_start = None
        self._root_seen = False
        self._root_done = False
        self._ignored_depth = 0

    def _iter_namespaces(self, ns_stack, _reversed=reversed):
        for namespaces in _reversed(ns_stack):
            if namespaces:  # almost no element declares new namespaces
                yield from namespaces

    def _resolve_prefix_name(self, prefixed_name):
        prefix, name = prefixed_name.split(':', 1)
        for uri, p in self._iter_namespaces(self._ns_stack):
            if p == prefix:
                return f'{{{uri}}}{name}'
        raise ValueError(f'Prefix {prefix} of QName "{prefixed_name}" is not declared in scope')

    def _qname(self, qname, uri=None):
        if uri is None:
            uri, tag = qname[1:].rsplit('}', 1) if qname[:1] == '{' else ('', qname)
        else:
            tag = qname

        prefixes_seen = set()
        for u, prefix in self._iter_namespaces(self._declared_ns_stack):
            if u == uri and prefix not in prefixes_seen:
                return f'{prefix}:{tag}' if prefix else tag, tag, uri
            prefixes_seen.add(prefix)

        # Not declared yet => add new declaration.
        if self._rewrite_prefixes:
            if uri in self._prefix_map:
                prefix = self._prefix_map[uri]
            else:
                prefix = self._prefix_map[uri] = f'n{len(self._prefix_map)}'
            self._declared_ns_stack[-1].append((uri, prefix))
            return f'{prefix}:{tag}', tag, uri

        if not uri and '' not in prefixes_seen:
            # No default namespace declared => no prefix needed.
            return tag, tag, uri

        for u, prefix in self._iter_namespaces(self._ns_stack):
            if u == uri:
                self._declared_ns_stack[-1].append((uri, prefix))
                return f'{prefix}:{tag}' if prefix else tag, tag, uri

        if not uri:
            # As soon as a default namespace is defined,
            # anything that has no namespace (and thus, no prefix) goes there.
            return tag, tag, uri

        raise ValueError(f'Namespace "{uri}" is not declared in scope')

    def data(self, data):
        if not self._ignored_depth:
            self._data.append(data)

    def _flush(self, _join_text=''.join):
        data = _join_text(self._data)
        del self._data[:]
        if self._strip_text and not self._preserve_space[-1]:
            data = data.strip()
        if self._pending_start is not None:
            args, self._pending_start = self._pending_start, None
            qname_text = data if data and _looks_like_prefix_name(data) else None
            self._start(*args, qname_text)
            if qname_text is not None:
                return
        if data and self._root_seen:
            self._write(_escape_cdata_c14n(data))

    def start_ns(self, prefix, uri):
        if self._ignored_depth:
            return
        # we may have to resolve qnames in text content
        if self._data:
            self._flush()
        self._ns_stack[-1].append((uri, prefix))

    def start(self, tag, attrs):
        if self._exclude_tags is not None and (
                self._ignored_depth or tag in self._exclude_tags):
            self._ignored_depth += 1
            return
        if self._data:
            self._flush()

        new_namespaces = []
        self._declared_ns_stack.append(new_namespaces)

        if self._qname_aware_tags is not None and tag in self._qname_aware_tags:
            # Need to parse text first to see if it requires a prefix declaration.
            self._pending_start = (tag, attrs, new_namespaces)
            return
        self._start(tag, attrs, new_namespaces)

    def _start(self, tag, attrs, new_namespaces, qname_text=None):
        if self._exclude_attrs is not None and attrs:
            attrs = {k: v for k, v in attrs.items() if k not in self._exclude_attrs}

        qnames = {tag, *attrs}
        resolved_names = {}

        # Resolve prefixes in attribute and tag text.
        if qname_text is not None:
            qname = resolved_names[qname_text] = self._resolve_prefix_name(qname_text)
            qnames.add(qname)
        if self._find_qname_aware_attrs is not None and attrs:
            qattrs = self._find_qname_aware_attrs(attrs)
            if qattrs:
                for attr_name in qattrs:
                    value = attrs[attr_name]
                    if _looks_like_prefix_name(value):
                        qname = resolved_names[value] = self._resolve_prefix_name(value)
                        qnames.add(qname)
            else:
                qattrs = None
        else:
            qattrs = None

        # Assign prefixes in lexicographical order of used URIs.
        parse_qname = self._qname
        parsed_qnames = {n: parse_qname(n) for n in sorted(
            qnames, key=lambda n: n.split('}', 1))}

        # Write namespace declarations in prefix order ...
        if new_namespaces:
            attr_list = [
                ('xmlns:' + prefix if prefix else 'xmlns', uri)
                for uri, prefix in new_namespaces
            ]
            attr_list.sort()
        else:
            # almost always empty
            attr_list = []

        # ... followed by attributes in URI+name order
        if attrs:
            for k, v in sorted(attrs.items()):
                if qattrs is not None and k in qattrs and v in resolved_names:
                    v = parsed_qnames[resolved_names[v]][0]
                attr_qname, attr_name, uri = parsed_qnames[k]
                # No prefix for attributes in default ('') namespace.
                attr_list.append((attr_qname if uri else attr_name, v))

        # Honour xml:space attributes.
        space_behaviour = attrs.get('{http://www.w3.org/XML/1998/namespace}space')
        self._preserve_space.append(
            space_behaviour == 'preserve' if space_behaviour
            else self._preserve_space[-1])

        # Write the tag.
        write = self._write
        write('<' + parsed_qnames[tag][0])
        if attr_list:
            write(''.join([f' {k}="{_escape_attrib_c14n(v)}"' for k, v in attr_list]))
        write('>')

        # Write the resolved qname text content.
        if qname_text is not None:
            write(_escape_cdata_c14n(parsed_qnames[resolved_names[qname_text]][0]))

        self._root_seen = True
        self._ns_stack.append([])

    def end(self, tag):
        if self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._data:
            self._flush()
        self._write(f'</{self._qname(tag)[0]}>')
        self._preserve_space.pop()
        self._root_done = len(self._preserve_space) == 1
        self._declared_ns_stack.pop()
        self._ns_stack.pop()

    def comment(self, text):
        if not self._with_comments:
            return
        if self._ignored_depth:
            return
        if self._root_done:
            self._write('\n')
        elif self._root_seen and self._data:
            self._flush()
        self._write(f'<!--{_escape_cdata_c14n(text)}-->')
        if not self._root_seen:
            self._write('\n')

    def pi(self, target, data):
        if self._ignored_depth:
            return
        if self._root_done:
            self._write('\n')
        elif self._root_seen and self._data:
            self._flush()
        self._write(
            f'<?{target} {_escape_cdata_c14n(data)}?>' if data else f'<?{target}?>')
        if not self._root_seen:
            self._write('\n')


def _escape_cdata_c14n(text):
    # escape character data
    try:
        # it's worth avoiding do-nothing calls for strings that are
        # shorter than 500 character, or so.  assume that's, by far,
        # the most common case in most applications.
        if '&' in text:
            text = text.replace('&', '&amp;')
        if '<' in text:
            text = text.replace('<', '&lt;')
        if '>' in text:
            text = text.replace('>', '&gt;')
        if '\r' in text:
            text = text.replace('\r', '&#xD;')
        return text
    except (TypeError, AttributeError):
        _raise_serialization_error(text)


def _escape_attrib_c14n(text):
    # escape attribute value
    try:
        if '&' in text:
            text = text.replace('&', '&amp;')
        if '<' in text:
            text = text.replace('<', '&lt;')
        if '"' in text:
            text = text.replace('"', '&quot;')
        if '\t' in text:
            text = text.replace('\t', '&#x9;')
        if '\n' in text:
            text = text.replace('\n', '&#xA;')
        if '\r' in text:
            text = text.replace('\r', '&#xD;')
        return text
    except (TypeError, AttributeError):
        _raise_serialization_error(text)


# --------------------------------------------------------------------

# Import the C accelerators
try:
    # Element is going to be shadowed by the C implementation. We need to keep
    # the Python version of it accessible for some "creative" by external code
    # (see tests)
    _Element_Py = Element

    # Element, SubElement, ParseError, TreeBuilder, XMLParser, _set_factories
    from _elementtree import *
    from _elementtree import _set_factories
except ImportError:
    pass
else:
    _set_factories(Comment, ProcessingInstruction)
<|endoftext|>#
# ElementTree
# $Id: ElementInclude.py 3375 2008-02-13 08:05:08Z fredrik $
#
# limited xinclude support for element trees
#
# history:
# 2003-08-15 fl   created
# 2003-11-14 fl   fixed default loader
#
# Copyright (c) 2003-2004 by Fredrik Lundh.  All rights reserved.
#
# fredrik@pythonware.com
# http://www.pythonware.com
#
# --------------------------------------------------------------------
# The ElementTree toolkit is
#
# Copyright (c) 1999-2008 by Fredrik Lundh
#
# By obtaining, using, and/or copying this software and/or its
# associated documentation, you agree that you have read, understood,
# and will comply with the following terms and conditions:
#
# Permission to use, copy, modify, and distribute this software and
# its associated documentation for any purpose and without fee is
# hereby granted, provided that the above copyright notice appears in
# all copies, and that both that copyright notice and this permission
# notice appear in supporting documentation, and that the name of
# Secret Labs AB or the author not be used in advertising or publicity
# pertaining to distribution of the software without specific, written
# prior permission.
#
# SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD
# TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANT-
# ABILITY AND FITNESS.  IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR
# BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THIS SOFTWARE.
# --------------------------------------------------------------------

# Licensed to PSF under a Contributor Agreement.
# See https://www.python.org/psf/license for licensing details.

##
# Limited XInclude support for the ElementTree package.
##

import copy
from . import ElementTree
from urllib.parse import urljoin

XINCLUDE = "{http://www.w3.org/2001/XInclude}"

XINCLUDE_INCLUDE = XINCLUDE + "include"
XINCLUDE_FALLBACK = XINCLUDE + "fallback"

# For security reasons, the inclusion depth is limited to this read-only value by default.
DEFAULT_MAX_INCLUSION_DEPTH = 6


##
# Fatal include error.

class FatalIncludeError(SyntaxError):
    pass


class LimitedRecursiveIncludeError(FatalIncludeError):
    pass


##
# Default loader.  This loader reads an included resource from disk.
#
# @param href Resource reference.
# @param parse Parse mode.  Either "xml" or "text".
# @param encoding Optional text encoding (UTF-8 by default for "text").
# @return The expanded resource.  If the parse mode is "xml", this
#    is an ElementTree instance.  If the parse mode is "text", this
#    is a Unicode string.  If the loader fails, it can return None
#    or raise an OSError exception.
# @throws OSError If the loader fails to load the resource.

def default_loader(href, parse, encoding=None):
    if parse == "xml":
        with open(href, 'rb') as file:
            data = ElementTree.parse(file).getroot()
    else:
        if not encoding:
            encoding = 'UTF-8'
        with open(href, 'r', encoding=encoding) as file:
            data = file.read()
    return data

##
# Expand XInclude directives.
#
# @param elem Root element.
# @param loader Optional resource loader.  If omitted, it defaults
#     to {@link default_loader}.  If given, it should be a callable
#     that implements the same interface as <b>default_loader</b>.
# @param base_url The base URL of the original file, to resolve
#     relative include file references.
# @param max_depth The maximum number of recursive inclusions.
#     Limited to reduce the risk of malicious content explosion.
#     Pass a negative value to disable the limitation.
# @throws LimitedRecursiveIncludeError If the {@link max_depth} was exceeded.
# @throws FatalIncludeError If the function fails to include a given
#     resource, or if the tree contains malformed XInclude elements.
# @throws IOError If the function fails to load a given resource.
# @returns the node or its replacement if it was an XInclude node

def include(elem, loader=None, base_url=None,
            max_depth=DEFAULT_MAX_INCLUSION_DEPTH):
    if max_depth is None:
        max_depth = -1
    elif max_depth < 0:
        raise ValueError("expected non-negative depth or None for 'max_depth', got %r" % max_depth)

    if hasattr(elem, 'getroot'):
        elem = elem.getroot()
    if loader is None:
        loader = default_loader

    _include(elem, loader, base_url, max_depth, set())


def _include(elem, loader, base_url, max_depth, _parent_hrefs):
    # look for xinclude elements
    i = 0
    while i < len(elem):
        e = elem[i]
        if e.tag == XINCLUDE_INCLUDE:
            # process xinclude directive
            href = e.get("href")
            if base_url:
                href = urljoin(base_url, href)
            parse = e.get("parse", "xml")
            if parse == "xml":
                if href in _parent_hrefs:
                    raise FatalIncludeError("recursive include of %s" % href)
                if max_depth == 0:
                    raise LimitedRecursiveIncludeError(
                        "maximum xinclude depth reached when including file %s" % href)
                _parent_hrefs.add(href)
                node = loader(href, parse)
                if node is None:
                    raise FatalIncludeError(
                        "cannot load %r as %r" % (href, parse)
                        )
                node = copy.copy(node)  # FIXME: this makes little sense with recursive includes
                _include(node, loader, href, max_depth - 1, _parent_hrefs)
                _parent_hrefs.remove(href)
                if e.tail:
                    node.tail = (node.tail or "") + e.tail
                elem[i] = node
            elif parse == "text":
                text = loader(href, parse, e.get("encoding"))
                if text is None:
                    raise FatalIncludeError(
                        "cannot load %r as %r" % (href, parse)
                        )
                if e.tail:
                    text += e.tail
                if i:
                    node = elem[i-1]
                    node.tail = (node.tail or "") + text
                else:
                    elem.text = (elem.text or "") + text
                del elem[i]
                continue
            else:
                raise FatalIncludeError(
                    "unknown parse type in xi:include tag (%r)" % parse
                )
        elif e.tag == XINCLUDE_FALLBACK:
            raise FatalIncludeError(
                "xi:fallback tag must be child of xi:include (%r)" % e.tag
                )
        else:
            _include(e, loader, base_url, max_depth, _parent_hrefs)
        i += 1
<|endoftext|>#
# ElementTree
# $Id: ElementPath.py 3375 2008-02-13 08:05:08Z fredrik $
#
# limited xpath support for element trees
#
# history:
# 2003-05-23 fl   created
# 2003-05-28 fl   added support for // etc
# 2003-08-27 fl   fixed parsing of periods in element names
# 2007-09-10 fl   new selection engine
# 2007-09-12 fl   fixed parent selector
# 2007-09-13 fl   added iterfind; changed findall to return a list
# 2007-11-30 fl   added namespaces support
# 2009-10-30 fl   added child element value filter
#
# Copyright (c) 2003-2009 by Fredrik Lundh.  All rights reserved.
#
# fredrik@pythonware.com
# http://www.pythonware.com
#
# --------------------------------------------------------------------
# The ElementTree toolkit is
#
# Copyright (c) 1999-2009 by Fredrik Lundh
#
# By obtaining, using, and/or copying this software and/or its
# associated documentation, you agree that you have read, understood,
# and will comply with the following terms and conditions:
#
# Permission to use, copy, modify, and distribute this software and
# its associated documentation for any purpose and without fee is
# hereby granted, provided that the above copyright notice appears in
# all copies, and that both that copyright notice and this permission
# notice appear in supporting documentation, and that the name of
# Secret Labs AB or the author not be used in advertising or publicity
# pertaining to distribution of the software without specific, written
# prior permission.
#
# SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD
# TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANT-
# ABILITY AND FITNESS.  IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR
# BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THIS SOFTWARE.
# --------------------------------------------------------------------

# Licensed to PSF under a Contributor Agreement.
# See https://www.python.org/psf/license for licensing details.

##
# Implementation module for XPath support.  There's usually no reason
# to import this module directly; the <b>ElementTree</b> does this for
# you, if needed.
##

import re

xpath_tokenizer_re = re.compile(
    r"("
    r"'[^']*'|\"[^\"]*\"|"
    r"::|"
    r"//?|"
    r"\.\.|"
    r"\(\)|"
    r"!=|"
    r"[/.*:\[\]\(\)@=])|"
    r"((?:\{[^}]+\})?[^/\[\]\(\)@!=\s]+)|"
    r"\s+"
    )

def xpath_tokenizer(pattern, namespaces=None):
    default_namespace = namespaces.get('') if namespaces else None
    parsing_attribute = False
    for token in xpath_tokenizer_re.findall(pattern):
        ttype, tag = token
        if tag and tag[0] != "{":
            if ":" in tag:
                prefix, uri = tag.split(":", 1)
                try:
                    if not namespaces:
                        raise KeyError
                    yield ttype, "{%s}%s" % (namespaces[prefix], uri)
                except KeyError:
                    raise SyntaxError("prefix %r not found in prefix map" % prefix) from None
            elif default_namespace and not parsing_attribute:
                yield ttype, "{%s}%s" % (default_namespace, tag)
            else:
                yield token
            parsing_attribute = False
        else:
            yield token
            parsing_attribute = ttype == '@'


def get_parent_map(context):
    parent_map = context.parent_map
    if parent_map is None:
        context.parent_map = parent_map = {}
        for p in context.root.iter():
            for e in p:
                parent_map[e] = p
    return parent_map


def _is_wildcard_tag(tag):
    return tag[:3] == '{*}' or tag[-2:] == '}*'


def _prepare_tag(tag):
    _isinstance, _str = isinstance, str
    if tag == '{*}*':
        # Same as '*', but no comments or processing instructions.
        # It can be a surprise that '*' includes those, but there is no
        # justification for '{*}*' doing the same.
        def select(context, result):
            for elem in result:
                if _isinstance(elem.tag, _str):
                    yield elem
    elif tag == '{}*':
        # Any tag that is not in a namespace.
        def select(context, result):
            for elem in result:
                el_tag = elem.tag
                if _isinstance(el_tag, _str) and el_tag[0] != '{':
                    yield elem
    elif tag[:3] == '{*}':
        # The tag in any (or no) namespace.
        suffix = tag[2:]  # '}name'
        no_ns = slice(-len(suffix), None)
        tag = tag[3:]
        def select(context, result):
            for elem in result:
                el_tag = elem.tag
                if el_tag == tag or _isinstance(el_tag, _str) and el_tag[no_ns] == suffix:
                    yield elem
    elif tag[-2:] == '}*':
        # Any tag in the given namespace.
        ns = tag[:-1]
        ns_only = slice(None, len(ns))
        def select(context, result):
            for elem in result:
                el_tag = elem.tag
                if _isinstance(el_tag, _str) and el_tag[ns_only] == ns:
                    yield elem
    else:
        raise RuntimeError(f"internal parser error, got {tag}")
    return select


def prepare_child(next, token):
    tag = token[1]
    if _is_wildcard_tag(tag):
        select_tag = _prepare_tag(tag)
        def select(context, result):
            def select_child(result):
                for elem in result:
                    yield from elem
            return select_tag(context, select_child(result))
    else:
        if tag[:2] == '{}':
            tag = tag[2:]  # '{}tag' == 'tag'
        def select(context, result):
            for elem in result:
                for e in elem:
                    if e.tag == tag:
                        yield e
    return select

def prepare_star(next, token):
    def select(context, result):
        for elem in result:
            yield from elem
    return select

def prepare_self(next, token):
    def select(context, result):
        yield from result
    return select

def prepare_descendant(next, token):
    try:
        token = next()
    except StopIteration:
        return
    if token[0] == "*":
        tag = "*"
    elif not token[0]:
        tag = token[1]
    else:
        raise SyntaxError("invalid descendant")

    if _is_wildcard_tag(tag):
        select_tag = _prepare_tag(tag)
        def select(context, result):
            def select_child(result):
                for elem in result:
                    for e in elem.iter():
                        if e is not elem:
                            yield e
            return select_tag(context, select_child(result))
    else:
        if tag[:2] == '{}':
            tag = tag[2:]  # '{}tag' == 'tag'
        def select(context, result):
            for elem in result:
                for e in elem.iter(tag):
                    if e is not elem:
                        yield e
    return select

def prepare_parent(next, token):
    def select(context, result):
        # FIXME: raise error if .. is applied at toplevel?
        parent_map = get_parent_map(context)
        result_map = {}
        for elem in result:
            if elem in parent_map:
                parent = parent_map[elem]
                if parent not in result_map:
                    result_map[parent] = None
                    yield parent
    return select

def prepare_predicate(next, token):
    # FIXME: replace with real parser!!! refs:
    # http://javascript.crockford.com/tdop/tdop.html
    signature = []
    predicate = []
    while 1:
        try:
            token = next()
        except StopIteration:
            return
        if token[0] == "]":
            break
        if token == ('', ''):
            # ignore whitespace
            continue
        if token[0] and token[0][:1] in "'\"":
            token = "'", token[0][1:-1]
        signature.append(token[0] or "-")
        predicate.append(token[1])
    signature = "".join(signature)
    # use signature to determine predicate type
    if signature == "@-":
        # [@attribute] predicate
        key = predicate[1]
        def select(context, result):
            for elem in result:
                if elem.get(key) is not None:
                    yield elem
        return select
    if signature == "@-='" or signature == "@-!='":
        # [@attribute='value'] or [@attribute!='value']
        key = predicate[1]
        value = predicate[-1]
        def select(context, result):
            for elem in result:
                if elem.get(key) == value:
                    yield elem
        def select_negated(context, result):
            for elem in result:
                if (attr_value := elem.get(key)) is not None and attr_value != value:
                    yield elem
        return select_negated if '!=' in signature else select
    if signature == "-" and not re.match(r"\-?\d+$", predicate[0]):
        # [tag]
        tag = predicate[0]
        def select(context, result):
            for elem in result:
                if elem.find(tag) is not None:
                    yield elem
        return select
    if signature == ".='" or signature == ".!='" or (
            (signature == "-='" or signature == "-!='")
            and not re.match(r"\-?\d+$", predicate[0])):
        # [.='value'] or [tag='value'] or [.!='value'] or [tag!='value']
        tag = predicate[0]
        value = predicate[-1]
        if tag:
            def select(context, result):
                for elem in result:
                    for e in elem.findall(tag):
                        if "".join(e.itertext()) == value:
                            yield elem
                            break
            def select_negated(context, result):
                for elem in result:
                    for e in elem.iterfind(tag):
                        if "".join(e.itertext()) != value:
                            yield elem
                            break
        else:
            def select(context, result):
                for elem in result:
                    if "".join(elem.itertext()) == value:
                        yield elem
            def select_negated(context, result):
                for elem in result:
                    if "".join(elem.itertext()) != value:
                        yield elem
        return select_negated if '!=' in signature else select
    if signature == "-" or signature == "-()" or signature == "-()-":
        # [index] or [last()] or [last()-index]
        if signature == "-":
            # [index]
            index = int(predicate[0]) - 1
            if index < 0:
                raise SyntaxError("XPath position >= 1 expected")
        else:
            if predicate[0] != "last":
                raise SyntaxError("unsupported function")
            if signature == "-()-":
                try:
                    index = int(predicate[2]) - 1
                except ValueError:
                    raise SyntaxError("unsupported expression")
                if index > -2:
                    raise SyntaxError("XPath offset from last() must be negative")
            else:
                index = -1
        def select(context, result):
            parent_map = get_parent_map(context)
            for elem in result:
                try:
                    parent = parent_map[elem]
                    # FIXME: what if the selector is "*" ?
                    elems = list(parent.findall(elem.tag))
                    if elems[index] is elem:
                        yield elem
                except (IndexError, KeyError):
                    pass
        return select
    raise SyntaxError("invalid predicate")

ops = {
    "": prepare_child,
    "*": prepare_star,
    ".": prepare_self,
    "..": prepare_parent,
    "//": prepare_descendant,
    "[": prepare_predicate,
    }

_cache = {}

class _SelectorContext:
    parent_map = None
    def __init__(self, root):
        self.root = root

# --------------------------------------------------------------------

##
# Generate all matching objects.

def iterfind(elem, path, namespaces=None):
    # compile selector pattern
    if path[-1:] == "/":
        path = path + "*" # implicit all (FIXME: keep this?)

    cache_key = (path,)
    if namespaces:
        cache_key += tuple(sorted(namespaces.items()))

    try:
        selector = _cache[cache_key]
    except KeyError:
        if len(_cache) > 100:
            _cache.clear()
        if path[:1] == "/":
            raise SyntaxError("cannot use absolute path on element")
        next = iter(xpath_tokenizer(path, namespaces)).__next__
        try:
            token = next()
        except StopIteration:
            return
        selector = []
        while 1:
            try:
                selector.append(ops[token[0]](next, token))
            except StopIteration:
                raise SyntaxError("invalid path") from None
            try:
                token = next()
                if token[0] == "/":
                    token = next()
            except StopIteration:
                break
        _cache[cache_key] = selector
    # execute selector pattern
    result = [elem]
    context = _SelectorContext(elem)
    for select in selector:
        result = select(context, result)
    return result

##
# Find first matching object.

def find(elem, path, namespaces=None):
    return next(iterfind(elem, path, namespaces), None)

##
# Find all matching objects.

def findall(elem, path, namespaces=None):
    return list(iterfind(elem, path, namespaces))

##
# Find text for first matching object.

def findtext(elem, path, default=None, namespaces=None):
    try:
        elem = next(iterfind(elem, path, namespaces))
        return elem.text or ""
    except StopIteration:
        return default
<|endoftext|>"""\
A library of useful helper classes to the SAX classes, for the
convenience of application and driver writers.
"""

import os, urllib.parse, urllib.request
import io
import codecs
from . import handler
from . import xmlreader

def __dict_replace(s, d):
    """Replace substrings of a string using a dictionary."""
    for key, value in d.items():
        s = s.replace(key, value)
    return s

def escape(data, entities={}):
    """Escape &, <, and > in a string of data.

    You can escape other strings of data by passing a dictionary as
    the optional entities parameter.  The keys and values must all be
    strings; each key will be replaced with its corresponding value.
    """

    # must do ampersand first
    data = data.replace("&", "&amp;")
    data = data.replace(">", "&gt;")
    data = data.replace("<", "&lt;")
    if entities:
        data = __dict_replace(data, entities)
    return data

def unescape(data, entities={}):
    """Unescape &amp;, &lt;, and &gt; in a string of data.

    You can unescape other strings of data by passing a dictionary as
    the optional entities parameter.  The keys and values must all be
    strings; each key will be replaced with its corresponding value.
    """
    data = data.replace("&lt;", "<")
    data = data.replace("&gt;", ">")
    if entities:
        data = __dict_replace(data, entities)
    # must do ampersand last
    return data.replace("&amp;", "&")

def quoteattr(data, entities={}):
    """Escape and quote an attribute value.

    Escape &, <, and > in a string of data, then quote it for use as
    an attribute value.  The \" character will be escaped as well, if
    necessary.

    You can escape other strings of data by passing a dictionary as
    the optional entities parameter.  The keys and values must all be
    strings; each key will be replaced with its corresponding value.
    """
    entities = {**entities, '\n': '&#10;', '\r': '&#13;', '\t':'&#9;'}
    data = escape(data, entities)
    if '"' in data:
        if "'" in data:
            data = '"%s"' % data.replace('"', "&quot;")
        else:
            data = "'%s'" % data
    else:
        data = '"%s"' % data
    return data


def _gettextwriter(out, encoding):
    if out is None:
        import sys
        return sys.stdout

    if isinstance(out, io.TextIOBase):
        # use a text writer as is
        return out

    if isinstance(out, (codecs.StreamWriter, codecs.StreamReaderWriter)):
        # use a codecs stream writer as is
        return out

    # wrap a binary writer with TextIOWrapper
    if isinstance(out, io.RawIOBase):
        # Keep the original file open when the TextIOWrapper is
        # destroyed
        class _wrapper:
            __class__ = out.__class__
            def __getattr__(self, name):
                return getattr(out, name)
        buffer = _wrapper()
        buffer.close = lambda: None
    else:
        # This is to handle passed objects that aren't in the
        # IOBase hierarchy, but just have a write method
        buffer = io.BufferedIOBase()
        buffer.writable = lambda: True
        buffer.write = out.write
        try:
            # TextIOWrapper uses this methods to determine
            # if BOM (for UTF-16, etc) should be added
            buffer.seekable = out.seekable
            buffer.tell = out.tell
        except AttributeError:
            pass
    return io.TextIOWrapper(buffer, encoding=encoding,
                            errors='xmlcharrefreplace',
                            newline='\n',
                            write_through=True)

class XMLGenerator(handler.ContentHandler):

    def __init__(self, out=None, encoding="iso-8859-1", short_empty_elements=False):
        handler.ContentHandler.__init__(self)
        out = _gettextwriter(out, encoding)
        self._write = out.write
        self._flush = out.flush
        self._ns_contexts = [{}] # contains uri -> prefix dicts
        self._current_context = self._ns_contexts[-1]
        self._undeclared_ns_maps = []
        self._encoding = encoding
        self._short_empty_elements = short_empty_elements
        self._pending_start_element = False

    def _qname(self, name):
        """Builds a qualified name from a (ns_url, localname) pair"""
        if name[0]:
            # Per http://www.w3.org/XML/1998/namespace, The 'xml' prefix is
            # bound by definition to http://www.w3.org/XML/1998/namespace.  It
            # does not need to be declared and will not usually be found in
            # self._current_context.
            if 'http://www.w3.org/XML/1998/namespace' == name[0]:
                return 'xml:' + name[1]
            # The name is in a non-empty namespace
            prefix = self._current_context[name[0]]
            if prefix:
                # If it is not the default namespace, prepend the prefix
                return prefix + ":" + name[1]
        # Return the unqualified name
        return name[1]

    def _finish_pending_start_element(self,endElement=False):
        if self._pending_start_element:
            self._write('>')
            self._pending_start_element = False

    # ContentHandler methods

    def startDocument(self):
        self._write('<?xml version="1.0" encoding="%s"?>\n' %
                        self._encoding)

    def endDocument(self):
        self._flush()

    def startPrefixMapping(self, prefix, uri):
        self._ns_contexts.append(self._current_context.copy())
        self._current_context[uri] = prefix
        self._undeclared_ns_maps.append((prefix, uri))

    def endPrefixMapping(self, prefix):
        self._current_context = self._ns_contexts[-1]
        del self._ns_contexts[-1]

    def startElement(self, name, attrs):
        self._finish_pending_start_element()
        self._write('<' + name)
        for (name, value) in attrs.items():
            self._write(' %s=%s' % (name, quoteattr(value)))
        if self._short_empty_elements:
            self._pending_start_element = True
        else:
            self._write(">")

    def endElement(self, name):
        if self._pending_start_element:
            self._write('/>')
            self._pending_start_element = False
        else:
            self._write('</%s>' % name)

    def startElementNS(self, name, qname, attrs):
        self._finish_pending_start_element()
        self._write('<' + self._qname(name))

        for prefix, uri in self._undeclared_ns_maps:
            if prefix:
                self._write(' xmlns:%s="%s"' % (prefix, uri))
            else:
                self._write(' xmlns="%s"' % uri)
        self._undeclared_ns_maps = []

        for (name, value) in attrs.items():
            self._write(' %s=%s' % (self._qname(name), quoteattr(value)))
        if self._short_empty_elements:
            self._pending_start_element = True
        else:
            self._write(">")

    def endElementNS(self, name, qname):
        if self._pending_start_element:
            self._write('/>')
            self._pending_start_element = False
        else:
            self._write('</%s>' % self._qname(name))

    def characters(self, content):
        if content:
            self._finish_pending_start_element()
            if not isinstance(content, str):
                content = str(content, self._encoding)
            self._write(escape(content))

    def ignorableWhitespace(self, content):
        if content:
            self._finish_pending_start_element()
            if not isinstance(content, str):
                content = str(content, self._encoding)
            self._write(content)

    def processingInstruction(self, target, data):
        self._finish_pending_start_element()
        self._write('<?%s %s?>' % (target, data))


class XMLFilterBase(xmlreader.XMLReader):
    """This class is designed to sit between an XMLReader and the
    client application's event handlers.  By default, it does nothing
    but pass requests up to the reader and events on to the handlers
    unmodified, but subclasses can override specific methods to modify
    the event stream or the configuration requests as they pass
    through."""

    def __init__(self, parent = None):
        xmlreader.XMLReader.__init__(self)
        self._parent = parent

    # ErrorHandler methods

    def error(self, exception):
        self._err_handler.error(exception)

    def fatalError(self, exception):
        self._err_handler.fatalError(exception)

    def warning(self, exception):
        self._err_handler.warning(exception)

    # ContentHandler methods

    def setDocumentLocator(self, locator):
        self._cont_handler.setDocumentLocator(locator)

    def startDocument(self):
        self._cont_handler.startDocument()

    def endDocument(self):
        self._cont_handler.endDocument()

    def startPrefixMapping(self, prefix, uri):
        self._cont_handler.startPrefixMapping(prefix, uri)

    def endPrefixMapping(self, prefix):
        self._cont_handler.endPrefixMapping(prefix)

    def startElement(self, name, attrs):
        self._cont_handler.startElement(name, attrs)

    def endElement(self, name):
        self._cont_handler.endElement(name)

    def startElementNS(self, name, qname, attrs):
        self._cont_handler.startElementNS(name, qname, attrs)

    def endElementNS(self, name, qname):
        self._cont_handler.endElementNS(name, qname)

    def characters(self, content):
        self._cont_handler.characters(content)

    def ignorableWhitespace(self, chars):
        self._cont_handler.ignorableWhitespace(chars)

    def processingInstruction(self, target, data):
        self._cont_handler.processingInstruction(target, data)

    def skippedEntity(self, name):
        self._cont_handler.skippedEntity(name)

    # DTDHandler methods

    def notationDecl(self, name, publicId, systemId):
        self._dtd_handler.notationDecl(name, publicId, systemId)

    def unparsedEntityDecl(self, name, publicId, systemId, ndata):
        self._dtd_handler.unparsedEntityDecl(name, publicId, systemId, ndata)

    # EntityResolver methods

    def resolveEntity(self, publicId, systemId):
        return self._ent_handler.resolveEntity(publicId, systemId)

    # XMLReader methods

    def parse(self, source):
        self._parent.setContentHandler(self)
        self._parent.setErrorHandler(self)
        self._parent.setEntityResolver(self)
        self._parent.setDTDHandler(self)
        self._parent.parse(source)

    def setLocale(self, locale):
        self._parent.setLocale(locale)

    def getFeature(self, name):
        return self._parent.getFeature(name)

    def setFeature(self, name, state):
        self._parent.setFeature(name, state)

    def getProperty(self, name):
        return self._parent.getProperty(name)

    def setProperty(self, name, value):
        self._parent.setProperty(name, value)

    # XMLFilter methods

    def getParent(self):
        return self._parent

    def setParent(self, parent):
        self._parent = parent

# --- Utility functions

def prepare_input_source(source, base=""):
    """This function takes an InputSource and an optional base URL and
    returns a fully resolved InputSource object ready for reading."""

    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if isinstance(source, str):
        source = xmlreader.InputSource(source)
    elif hasattr(source, "read"):
        f = source
        source = xmlreader.InputSource()
        if isinstance(f.read(0), str):
            source.setCharacterStream(f)
        else:
            source.setByteStream(f)
        if hasattr(f, "name") and isinstance(f.name, str):
            source.setSystemId(f.name)

    if source.getCharacterStream() is None and source.getByteStream() is None:
        sysid = source.getSystemId()
        basehead = os.path.dirname(os.path.normpath(base))
        sysidfilename = os.path.join(basehead, sysid)
        if os.path.isfile(sysidfilename):
            source.setSystemId(sysidfilename)
            f = open(sysidfilename, "rb")
        else:
            source.setSystemId(urllib.parse.urljoin(base, sysid))
            f = urllib.request.urlopen(source.getSystemId())

        source.setByteStream(f)

    return source
<|endoftext|>"""An XML Reader is the SAX 2 name for an XML parser. XML Parsers
should be based on this code. """

from . import handler

from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException


# ===== XMLREADER =====

class XMLReader:
    """Interface for reading an XML document using callbacks.

    XMLReader is the interface that an XML parser's SAX2 driver must
    implement. This interface allows an application to set and query
    features and properties in the parser, to register event handlers
    for document processing, and to initiate a document parse.

    All SAX interfaces are assumed to be synchronous: the parse
    methods must not return until parsing is complete, and readers
    must wait for an event-handler callback to return before reporting
    the next event."""

    def __init__(self):
        self._cont_handler = handler.ContentHandler()
        self._dtd_handler = handler.DTDHandler()
        self._ent_handler = handler.EntityResolver()
        self._err_handler = handler.ErrorHandler()

    def parse(self, source):
        "Parse an XML document from a system identifier or an InputSource."
        raise NotImplementedError("This method must be implemented!")

    def getContentHandler(self):
        "Returns the current ContentHandler."
        return self._cont_handler

    def setContentHandler(self, handler):
        "Registers a new object to receive document content events."
        self._cont_handler = handler

    def getDTDHandler(self):
        "Returns the current DTD handler."
        return self._dtd_handler

    def setDTDHandler(self, handler):
        "Register an object to receive basic DTD-related events."
        self._dtd_handler = handler

    def getEntityResolver(self):
        "Returns the current EntityResolver."
        return self._ent_handler

    def setEntityResolver(self, resolver):
        "Register an object to resolve external entities."
        self._ent_handler = resolver

    def getErrorHandler(self):
        "Returns the current ErrorHandler."
        return self._err_handler

    def setErrorHandler(self, handler):
        "Register an object to receive error-message events."
        self._err_handler = handler

    def setLocale(self, locale):
        """Allow an application to set the locale for errors and warnings.

        SAX parsers are not required to provide localization for errors
        and warnings; if they cannot support the requested locale,
        however, they must raise a SAX exception. Applications may
        request a locale change in the middle of a parse."""
        raise SAXNotSupportedException("Locale support not implemented")

    def getFeature(self, name):
        "Looks up and returns the state of a SAX2 feature."
        raise SAXNotRecognizedException("Feature '%s' not recognized" % name)

    def setFeature(self, name, state):
        "Sets the state of a SAX2 feature."
        raise SAXNotRecognizedException("Feature '%s' not recognized" % name)

    def getProperty(self, name):
        "Looks up and returns the value of a SAX2 property."
        raise SAXNotRecognizedException("Property '%s' not recognized" % name)

    def setProperty(self, name, value):
        "Sets the value of a SAX2 property."
        raise SAXNotRecognizedException("Property '%s' not recognized" % name)

class IncrementalParser(XMLReader):
    """This interface adds three extra methods to the XMLReader
    interface that allow XML parsers to support incremental
    parsing. Support for this interface is optional, since not all
    underlying XML parsers support this functionality.

    When the parser is instantiated it is ready to begin accepting
    data from the feed method immediately. After parsing has been
    finished with a call to close the reset method must be called to
    make the parser ready to accept new data, either from feed or
    using the parse method.

    Note that these methods must _not_ be called during parsing, that
    is, after parse has been called and before it returns.

    By default, the class also implements the parse method of the XMLReader
    interface using the feed, close and reset methods of the
    IncrementalParser interface as a convenience to SAX 2.0 driver
    writers."""

    def __init__(self, bufsize=2**16):
        self._bufsize = bufsize
        XMLReader.__init__(self)

    def parse(self, source):
        from . import saxutils
        source = saxutils.prepare_input_source(source)

        self.prepareParser(source)
        file = source.getCharacterStream()
        if file is None:
            file = source.getByteStream()
        buffer = file.read(self._bufsize)
        while buffer:
            self.feed(buffer)
            buffer = file.read(self._bufsize)
        self.close()

    def feed(self, data):
        """This method gives the raw XML data in the data parameter to
        the parser and makes it parse the data, emitting the
        corresponding events. It is allowed for XML constructs to be
        split across several calls to feed.

        feed may raise SAXException."""
        raise NotImplementedError("This method must be implemented!")

    def prepareParser(self, source):
        """This method is called by the parse implementation to allow
        the SAX 2.0 driver to prepare itself for parsing."""
        raise NotImplementedError("prepareParser must be overridden!")

    def close(self):
        """This method is called when the entire XML document has been
        passed to the parser through the feed method, to notify the
        parser that there are no more data. This allows the parser to
        do the final checks on the document and empty the internal
        data buffer.

        The parser will not be ready to parse another document until
        the reset method has been called.

        close may raise SAXException."""
        raise NotImplementedError("This method must be implemented!")

    def reset(self):
        """This method is called after close has been called to reset
        the parser so that it is ready to parse new documents. The
        results of calling parse or feed after close without calling
        reset are undefined."""
        raise NotImplementedError("This method must be implemented!")

# ===== LOCATOR =====

class Locator:
    """Interface for associating a SAX event with a document
    location. A locator object will return valid results only during
    calls to DocumentHandler methods; at any other time, the
    results are unpredictable."""

    def getColumnNumber(self):
        "Return the column number where the current event ends."
        return -1

    def getLineNumber(self):
        "Return the line number where the current event ends."
        return -1

    def getPublicId(self):
        "Return the public identifier for the current event."
        return None

    def getSystemId(self):
        "Return the system identifier for the current event."
        return None

# ===== INPUTSOURCE =====

class InputSource:
    """Encapsulation of the information needed by the XMLReader to
    read entities.

    This class may include information about the public identifier,
    system identifier, byte stream (possibly with character encoding
    information) and/or the character stream of an entity.

    Applications will create objects of this class for use in the
    XMLReader.parse method and for returning from
    EntityResolver.resolveEntity.

    An InputSource belongs to the application, the XMLReader is not
    allowed to modify InputSource objects passed to it from the
    application, although it may make copies and modify those."""

    def __init__(self, system_id = None):
        self.__system_id = system_id
        self.__public_id = None
        self.__encoding  = None
        self.__bytefile  = None
        self.__charfile  = None

    def setPublicId(self, public_id):
        "Sets the public identifier of this InputSource."
        self.__public_id = public_id

    def getPublicId(self):
        "Returns the public identifier of this InputSource."
        return self.__public_id

    def setSystemId(self, system_id):
        "Sets the system identifier of this InputSource."
        self.__system_id = system_id

    def getSystemId(self):
        "Returns the system identifier of this InputSource."
        return self.__system_id

    def setEncoding(self, encoding):
        """Sets the character encoding of this InputSource.

        The encoding must be a string acceptable for an XML encoding
        declaration (see section 4.3.3 of the XML recommendation).

        The encoding attribute of the InputSource is ignored if the
        InputSource also contains a character stream."""
        self.__encoding = encoding

    def getEncoding(self):
        "Get the character encoding of this InputSource."
        return self.__encoding

    def setByteStream(self, bytefile):
        """Set the byte stream (a Python file-like object which does
        not perform byte-to-character conversion) for this input
        source.

        The SAX parser will ignore this if there is also a character
        stream specified, but it will use a byte stream in preference
        to opening a URI connection itself.

        If the application knows the character encoding of the byte
        stream, it should set it with the setEncoding method."""
        self.__bytefile = bytefile

    def getByteStream(self):
        """Get the byte stream for this input source.

        The getEncoding method will return the character encoding for
        this byte stream, or None if unknown."""
        return self.__bytefile

    def setCharacterStream(self, charfile):
        """Set the character stream for this input source. (The stream
        must be a Python 2.0 Unicode-wrapped file-like that performs
        conversion to Unicode strings.)

        If there is a character stream specified, the SAX parser will
        ignore any byte stream and will not attempt to open a URI
        connection to the system identifier."""
        self.__charfile = charfile

    def getCharacterStream(self):
        "Get the character stream for this input source."
        return self.__charfile

# ===== ATTRIBUTESIMPL =====

class AttributesImpl:

    def __init__(self, attrs):
        """Non-NS-aware implementation.

        attrs should be of the form {name : value}."""
        self._attrs = attrs

    def getLength(self):
        return len(self._attrs)

    def getType(self, name):
        return "CDATA"

    def getValue(self, name):
        return self._attrs[name]

    def getValueByQName(self, name):
        return self._attrs[name]

    def getNameByQName(self, name):
        if name not in self._attrs:
            raise KeyError(name)
        return name

    def getQNameByName(self, name):
        if name not in self._attrs:
            raise KeyError(name)
        return name

    def getNames(self):
        return list(self._attrs.keys())

    def getQNames(self):
        return list(self._attrs.keys())

    def __len__(self):
        return len(self._attrs)

    def __getitem__(self, name):
        return self._attrs[name]

    def keys(self):
        return list(self._attrs.keys())

    def __contains__(self, name):
        return name in self._attrs

    def get(self, name, alternative=None):
        return self._attrs.get(name, alternative)

    def copy(self):
        return self.__class__(self._attrs)

    def items(self):
        return list(self._attrs.items())

    def values(self):
        return list(self._attrs.values())

# ===== ATTRIBUTESNSIMPL =====

class AttributesNSImpl(AttributesImpl):

    def __init__(self, attrs, qnames):
        """NS-aware implementation.

        attrs should be of the form {(ns_uri, lname): value, ...}.
        qnames of the form {(ns_uri, lname): qname, ...}."""
        self._attrs = attrs
        self._qnames = qnames

    def getValueByQName(self, name):
        for (nsname, qname) in self._qnames.items():
            if qname == name:
                return self._attrs[nsname]

        raise KeyError(name)

    def getNameByQName(self, name):
        for (nsname, qname) in self._qnames.items():
            if qname == name:
                return nsname

        raise KeyError(name)

    def getQNameByName(self, name):
        return self._qnames[name]

    def getQNames(self):
        return list(self._qnames.values())

    def copy(self):
        return self.__class__(self._attrs, self._qnames)


def _test():
    XMLReader()
    IncrementalParser()
    Locator()

if __name__ == "__main__":
    _test()
<|endoftext|>"""
SAX driver for the pyexpat C module.  This driver works with
pyexpat.__version__ == '2.22'.
"""

version = "0.20"

from xml.sax._exceptions import *
from xml.sax.handler import feature_validation, feature_namespaces
from xml.sax.handler import feature_namespace_prefixes
from xml.sax.handler import feature_external_ges, feature_external_pes
from xml.sax.handler import feature_string_interning
from xml.sax.handler import property_xml_string, property_interning_dict

# xml.parsers.expat does not raise ImportError in Jython
import sys
if sys.platform[:4] == "java":
    raise SAXReaderNotAvailable("expat not available in Java", None)
del sys

try:
    from xml.parsers import expat
except ImportError:
    raise SAXReaderNotAvailable("expat not supported", None)
else:
    if not hasattr(expat, "ParserCreate"):
        raise SAXReaderNotAvailable("expat not supported", None)
from xml.sax import xmlreader, saxutils, handler

AttributesImpl = xmlreader.AttributesImpl
AttributesNSImpl = xmlreader.AttributesNSImpl

# If we're using a sufficiently recent version of Python, we can use
# weak references to avoid cycles between the parser and content
# handler, otherwise we'll just have to pretend.
try:
    import _weakref
except ImportError:
    def _mkproxy(o):
        return o
else:
    import weakref
    _mkproxy = weakref.proxy
    del weakref, _weakref

class _ClosedParser:
    pass

# --- ExpatLocator

class ExpatLocator(xmlreader.Locator):
    """Locator for use with the ExpatParser class.

    This uses a weak reference to the parser object to avoid creating
    a circular reference between the parser and the content handler.
    """
    def __init__(self, parser):
        self._ref = _mkproxy(parser)

    def getColumnNumber(self):
        parser = self._ref
        if parser._parser is None:
            return None
        return parser._parser.ErrorColumnNumber

    def getLineNumber(self):
        parser = self._ref
        if parser._parser is None:
            return 1
        return parser._parser.ErrorLineNumber

    def getPublicId(self):
        parser = self._ref
        if parser is None:
            return None
        return parser._source.getPublicId()

    def getSystemId(self):
        parser = self._ref
        if parser is None:
            return None
        return parser._source.getSystemId()


# --- ExpatParser

class ExpatParser(xmlreader.IncrementalParser, xmlreader.Locator):
    """SAX driver for the pyexpat C module."""

    def __init__(self, namespaceHandling=0, bufsize=2**16-20):
        xmlreader.IncrementalParser.__init__(self, bufsize)
        self._source = xmlreader.InputSource()
        self._parser = None
        self._namespaces = namespaceHandling
        self._lex_handler_prop = None
        self._parsing = False
        self._entity_stack = []
        self._external_ges = 0
        self._interning = None

    # XMLReader methods

    def parse(self, source):
        "Parse an XML document from a URL or an InputSource."
        source = saxutils.prepare_input_source(source)

        self._source = source
        try:
            self.reset()
            self._cont_handler.setDocumentLocator(ExpatLocator(self))
            xmlreader.IncrementalParser.parse(self, source)
        except:
            # bpo-30264: Close the source on error to not leak resources:
            # xml.sax.parse() doesn't give access to the underlying parser
            # to the caller
            self._close_source()
            raise

    def prepareParser(self, source):
        if source.getSystemId() is not None:
            self._parser.SetBase(source.getSystemId())

    # Redefined setContentHandler to allow changing handlers during parsing

    def setContentHandler(self, handler):
        xmlreader.IncrementalParser.setContentHandler(self, handler)
        if self._parsing:
            self._reset_cont_handler()

    def getFeature(self, name):
        if name == feature_namespaces:
            return self._namespaces
        elif name == feature_string_interning:
            return self._interning is not None
        elif name in (feature_validation, feature_external_pes,
                      feature_namespace_prefixes):
            return 0
        elif name == feature_external_ges:
            return self._external_ges
        raise SAXNotRecognizedException("Feature '%s' not recognized" % name)

    def setFeature(self, name, state):
        if self._parsing:
            raise SAXNotSupportedException("Cannot set features while parsing")

        if name == feature_namespaces:
            self._namespaces = state
        elif name == feature_external_ges:
            self._external_ges = state
        elif name == feature_string_interning:
            if state:
                if self._interning is None:
                    self._interning = {}
            else:
                self._interning = None
        elif name == feature_validation:
            if state:
                raise SAXNotSupportedException(
                    "expat does not support validation")
        elif name == feature_external_pes:
            if state:
                raise SAXNotSupportedException(
                    "expat does not read external parameter entities")
        elif name == feature_namespace_prefixes:
            if state:
                raise SAXNotSupportedException(
                    "expat does not report namespace prefixes")
        else:
            raise SAXNotRecognizedException(
                "Feature '%s' not recognized" % name)

    def getProperty(self, name):
        if name == handler.property_lexical_handler:
            return self._lex_handler_prop
        elif name == property_interning_dict:
            return self._interning
        elif name == property_xml_string:
            if self._parser:
                if hasattr(self._parser, "GetInputContext"):
                    return self._parser.GetInputContext()
                else:
                    raise SAXNotRecognizedException(
                        "This version of expat does not support getting"
                        " the XML string")
            else:
                raise SAXNotSupportedException(
                    "XML string cannot be returned when not parsing")
        raise SAXNotRecognizedException("Property '%s' not recognized" % name)

    def setProperty(self, name, value):
        if name == handler.property_lexical_handler:
            self._lex_handler_prop = value
            if self._parsing:
                self._reset_lex_handler_prop()
        elif name == property_interning_dict:
            self._interning = value
        elif name == property_xml_string:
            raise SAXNotSupportedException("Property '%s' cannot be set" %
                                           name)
        else:
            raise SAXNotRecognizedException("Property '%s' not recognized" %
                                            name)

    # IncrementalParser methods

    def feed(self, data, isFinal=False):
        if not self._parsing:
            self.reset()
            self._parsing = True
            self._cont_handler.startDocument()

        try:
            # The isFinal parameter is internal to the expat reader.
            # If it is set to true, expat will check validity of the entire
            # document. When feeding chunks, they are not normally final -
            # except when invoked from close.
            self._parser.Parse(data, isFinal)
        except expat.error as e:
            exc = SAXParseException(expat.ErrorString(e.code), e, self)
            # FIXME: when to invoke error()?
            self._err_handler.fatalError(exc)

    def _close_source(self):
        source = self._source
        try:
            file = source.getCharacterStream()
            if file is not None:
                file.close()
        finally:
            file = source.getByteStream()
            if file is not None:
                file.close()

    def close(self):
        if (self._entity_stack or self._parser is None or
            isinstance(self._parser, _ClosedParser)):
            # If we are completing an external entity, do nothing here
            return
        try:
            self.feed(b"", isFinal=True)
            self._cont_handler.endDocument()
            self._parsing = False
            # break cycle created by expat handlers pointing to our methods
            self._parser = None
        finally:
            self._parsing = False
            if self._parser is not None:
                # Keep ErrorColumnNumber and ErrorLineNumber after closing.
                parser = _ClosedParser()
                parser.ErrorColumnNumber = self._parser.ErrorColumnNumber
                parser.ErrorLineNumber = self._parser.ErrorLineNumber
                self._parser = parser
            self._close_source()

    def _reset_cont_handler(self):
        self._parser.ProcessingInstructionHandler = \
                                    self._cont_handler.processingInstruction
        self._parser.CharacterDataHandler = self._cont_handler.characters

    def _reset_lex_handler_prop(self):
        lex = self._lex_handler_prop
        parser = self._parser
        if lex is None:
            parser.CommentHandler = None
            parser.StartCdataSectionHandler = None
            parser.EndCdataSectionHandler = None
            parser.StartDoctypeDeclHandler = None
            parser.EndDoctypeDeclHandler = None
        else:
            parser.CommentHandler = lex.comment
            parser.StartCdataSectionHandler = lex.startCDATA
            parser.EndCdataSectionHandler = lex.endCDATA
            parser.StartDoctypeDeclHandler = self.start_doctype_decl
            parser.EndDoctypeDeclHandler = lex.endDTD

    def reset(self):
        if self._namespaces:
            self._parser = expat.ParserCreate(self._source.getEncoding(), " ",
                                              intern=self._interning)
            self._parser.namespace_prefixes = 1
            self._parser.StartElementHandler = self.start_element_ns
            self._parser.EndElementHandler = self.end_element_ns
        else:
            self._parser = expat.ParserCreate(self._source.getEncoding(),
                                              intern = self._interning)
            self._parser.StartElementHandler = self.start_element
            self._parser.EndElementHandler = self.end_element

        self._reset_cont_handler()
        self._parser.UnparsedEntityDeclHandler = self.unparsed_entity_decl
        self._parser.NotationDeclHandler = self.notation_decl
        self._parser.StartNamespaceDeclHandler = self.start_namespace_decl
        self._parser.EndNamespaceDeclHandler = self.end_namespace_decl

        self._decl_handler_prop = None
        if self._lex_handler_prop:
            self._reset_lex_handler_prop()
#         self._parser.DefaultHandler =
#         self._parser.DefaultHandlerExpand =
#         self._parser.NotStandaloneHandler =
        self._parser.ExternalEntityRefHandler = self.external_entity_ref
        try:
            self._parser.SkippedEntityHandler = self.skipped_entity_handler
        except AttributeError:
            # This pyexpat does not support SkippedEntity
            pass
        self._parser.SetParamEntityParsing(
            expat.XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE)

        self._parsing = False
        self._entity_stack = []

    # Locator methods

    def getColumnNumber(self):
        if self._parser is None:
            return None
        return self._parser.ErrorColumnNumber

    def getLineNumber(self):
        if self._parser is None:
            return 1
        return self._parser.ErrorLineNumber

    def getPublicId(self):
        return self._source.getPublicId()

    def getSystemId(self):
        return self._source.getSystemId()

    # event handlers
    def start_element(self, name, attrs):
        self._cont_handler.startElement(name, AttributesImpl(attrs))

    def end_element(self, name):
        self._cont_handler.endElement(name)

    def start_element_ns(self, name, attrs):
        pair = name.split()
        if len(pair) == 1:
            # no namespace
            pair = (None, name)
        elif len(pair) == 3:
            pair = pair[0], pair[1]
        else:
            # default namespace
            pair = tuple(pair)

        newattrs = {}
        qnames = {}
        for (aname, value) in attrs.items():
            parts = aname.split()
            length = len(parts)
            if length == 1:
                # no namespace
                qname = aname
                apair = (None, aname)
            elif length == 3:
                qname = "%s:%s" % (parts[2], parts[1])
                apair = parts[0], parts[1]
            else:
                # default namespace
                qname = parts[1]
                apair = tuple(parts)

            newattrs[apair] = value
            qnames[apair] = qname

        self._cont_handler.startElementNS(pair, None,
                                          AttributesNSImpl(newattrs, qnames))

    def end_element_ns(self, name):
        pair = name.split()
        if len(pair) == 1:
            pair = (None, name)
        elif len(pair) == 3:
            pair = pair[0], pair[1]
        else:
            pair = tuple(pair)

        self._cont_handler.endElementNS(pair, None)

    # this is not used (call directly to ContentHandler)
    def processing_instruction(self, target, data):
        self._cont_handler.processingInstruction(target, data)

    # this is not used (call directly to ContentHandler)
    def character_data(self, data):
        self._cont_handler.characters(data)

    def start_namespace_decl(self, prefix, uri):
        self._cont_handler.startPrefixMapping(prefix, uri)

    def end_namespace_decl(self, prefix):
        self._cont_handler.endPrefixMapping(prefix)

    def start_doctype_decl(self, name, sysid, pubid, has_internal_subset):
        self._lex_handler_prop.startDTD(name, pubid, sysid)

    def unparsed_entity_decl(self, name, base, sysid, pubid, notation_name):
        self._dtd_handler.unparsedEntityDecl(name, pubid, sysid, notation_name)

    def notation_decl(self, name, base, sysid, pubid):
        self._dtd_handler.notationDecl(name, pubid, sysid)

    def external_entity_ref(self, context, base, sysid, pubid):
        if not self._external_ges:
            return 1

        source = self._ent_handler.resolveEntity(pubid, sysid)
        source = saxutils.prepare_input_source(source,
                                               self._source.getSystemId() or
                                               "")

        self._entity_stack.append((self._parser, self._source))
        self._parser = self._parser.ExternalEntityParserCreate(context)
        self._source = source

        try:
            xmlreader.IncrementalParser.parse(self, source)
        except:
            return 0  # FIXME: save error info here?

        (self._parser, self._source) = self._entity_stack[-1]
        del self._entity_stack[-1]
        return 1

    def skipped_entity_handler(self, name, is_pe):
        if is_pe:
            # The SAX spec requires to report skipped PEs with a '%'
            name = '%'+name
        self._cont_handler.skippedEntity(name)

# ---

def create_parser(*args, **kwargs):
    return ExpatParser(*args, **kwargs)

# ---

if __name__ == "__main__":
    import xml.sax.saxutils
    p = create_parser()
    p.setContentHandler(xml.sax.saxutils.XMLGenerator())
    p.setErrorHandler(xml.sax.ErrorHandler())
    p.parse("http://www.ibiblio.org/xml/examples/shakespeare/hamlet.xml")
<|endoftext|>"""Simple API for XML (SAX) implementation for Python.

This module provides an implementation of the SAX 2 interface;
information about the Java version of the interface can be found at
http://www.megginson.com/SAX/.  The Python version of the interface is
documented at <...>.

This package contains the following modules:

handler -- Base classes and constants which define the SAX 2 API for
           the 'client-side' of SAX for Python.

saxutils -- Implementation of the convenience classes commonly used to
            work with SAX.

xmlreader -- Base classes and constants which define the SAX 2 API for
             the parsers used with SAX for Python.

expatreader -- Driver that allows use of the Expat parser with SAX.
"""

from .xmlreader import InputSource
from .handler import ContentHandler, ErrorHandler
from ._exceptions import SAXException, SAXNotRecognizedException, \
                        SAXParseException, SAXNotSupportedException, \
                        SAXReaderNotAvailable


def parse(source, handler, errorHandler=ErrorHandler()):
    parser = make_parser()
    parser.setContentHandler(handler)
    parser.setErrorHandler(errorHandler)
    parser.parse(source)

def parseString(string, handler, errorHandler=ErrorHandler()):
    import io
    if errorHandler is None:
        errorHandler = ErrorHandler()
    parser = make_parser()
    parser.setContentHandler(handler)
    parser.setErrorHandler(errorHandler)

    inpsrc = InputSource()
    if isinstance(string, str):
        inpsrc.setCharacterStream(io.StringIO(string))
    else:
        inpsrc.setByteStream(io.BytesIO(string))
    parser.parse(inpsrc)

# this is the parser list used by the make_parser function if no
# alternatives are given as parameters to the function

default_parser_list = ["xml.sax.expatreader"]

# tell modulefinder that importing sax potentially imports expatreader
_false = 0
if _false:
    import xml.sax.expatreader

import os, sys
if not sys.flags.ignore_environment and "PY_SAX_PARSER" in os.environ:
    default_parser_list = os.environ["PY_SAX_PARSER"].split(",")
del os

_key = "python.xml.sax.parser"
if sys.platform[:4] == "java" and sys.registry.containsKey(_key):
    default_parser_list = sys.registry.getProperty(_key).split(",")


def make_parser(parser_list=()):
    """Creates and returns a SAX parser.

    Creates the first parser it is able to instantiate of the ones
    given in the iterable created by chaining parser_list and
    default_parser_list.  The iterables must contain the names of Python
    modules containing both a SAX parser and a create_parser function."""

    for parser_name in list(parser_list) + default_parser_list:
        try:
            return _create_parser(parser_name)
        except ImportError:
            import sys
            if parser_name in sys.modules:
                # The parser module was found, but importing it
                # failed unexpectedly, pass this exception through
                raise
        except SAXReaderNotAvailable:
            # The parser module detected that it won't work properly,
            # so try the next one
            pass

    raise SAXReaderNotAvailable("No parsers found", None)

# --- Internal utility methods used by make_parser

if sys.platform[ : 4] == "java":
    def _create_parser(parser_name):
        from org.python.core import imp
        drv_module = imp.importName(parser_name, 0, globals())
        return drv_module.create_parser()

else:
    def _create_parser(parser_name):
        drv_module = __import__(parser_name,{},{},['create_parser'])
        return drv_module.create_parser()

del sys
<|endoftext|>"""Different kinds of SAX Exceptions"""
import sys
if sys.platform[:4] == "java":
    from java.lang import Exception
del sys

# ===== SAXEXCEPTION =====

class SAXException(Exception):
    """Encapsulate an XML error or warning. This class can contain
    basic error or warning information from either the XML parser or
    the application: you can subclass it to provide additional
    functionality, or to add localization. Note that although you will
    receive a SAXException as the argument to the handlers in the
    ErrorHandler interface, you are not actually required to raise
    the exception; instead, you can simply read the information in
    it."""

    def __init__(self, msg, exception=None):
        """Creates an exception. The message is required, but the exception
        is optional."""
        self._msg = msg
        self._exception = exception
        Exception.__init__(self, msg)

    def getMessage(self):
        "Return a message for this exception."
        return self._msg

    def getException(self):
        "Return the embedded exception, or None if there was none."
        return self._exception

    def __str__(self):
        "Create a string representation of the exception."
        return self._msg

    def __getitem__(self, ix):
        """Avoids weird error messages if someone does exception[ix] by
        mistake, since Exception has __getitem__ defined."""
        raise AttributeError("__getitem__")


# ===== SAXPARSEEXCEPTION =====

class SAXParseException(SAXException):
    """Encapsulate an XML parse error or warning.

    This exception will include information for locating the error in
    the original XML document. Note that although the application will
    receive a SAXParseException as the argument to the handlers in the
    ErrorHandler interface, the application is not actually required
    to raise the exception; instead, it can simply read the
    information in it and take a different action.

    Since this exception is a subclass of SAXException, it inherits
    the ability to wrap another exception."""

    def __init__(self, msg, exception, locator):
        "Creates the exception. The exception parameter is allowed to be None."
        SAXException.__init__(self, msg, exception)
        self._locator = locator

        # We need to cache this stuff at construction time.
        # If this exception is raised, the objects through which we must
        # traverse to get this information may be deleted by the time
        # it gets caught.
        self._systemId = self._locator.getSystemId()
        self._colnum = self._locator.getColumnNumber()
        self._linenum = self._locator.getLineNumber()

    def getColumnNumber(self):
        """The column number of the end of the text where the exception
        occurred."""
        return self._colnum

    def getLineNumber(self):
        "The line number of the end of the text where the exception occurred."
        return self._linenum

    def getPublicId(self):
        "Get the public identifier of the entity where the exception occurred."
        return self._locator.getPublicId()

    def getSystemId(self):
        "Get the system identifier of the entity where the exception occurred."
        return self._systemId

    def __str__(self):
        "Create a string representation of the exception."
        sysid = self.getSystemId()
        if sysid is None:
            sysid = "<unknown>"
        linenum = self.getLineNumber()
        if linenum is None:
            linenum = "?"
        colnum = self.getColumnNumber()
        if colnum is None:
            colnum = "?"
        return "%s:%s:%s: %s" % (sysid, linenum, colnum, self._msg)


# ===== SAXNOTRECOGNIZEDEXCEPTION =====

class SAXNotRecognizedException(SAXException):
    """Exception class for an unrecognized identifier.

    An XMLReader will raise this exception when it is confronted with an
    unrecognized feature or property. SAX applications and extensions may
    use this class for similar purposes."""


# ===== SAXNOTSUPPORTEDEXCEPTION =====

class SAXNotSupportedException(SAXException):
    """Exception class for an unsupported operation.

    An XMLReader will raise this exception when a service it cannot
    perform is requested (specifically setting a state or value). SAX
    applications and extensions may use this class for similar
    purposes."""

# ===== SAXNOTSUPPORTEDEXCEPTION =====

class SAXReaderNotAvailable(SAXNotSupportedException):
    """Exception class for a missing driver.

    An XMLReader module (driver) should raise this exception when it
    is first imported, e.g. when a support module cannot be imported.
    It also may be raised during parsing, e.g. if executing an external
    program is not permitted."""
<|endoftext|>"""
This module contains the core classes of version 2.0 of SAX for Python.
This file provides only default classes with absolutely minimum
functionality, from which drivers and applications can be subclassed.

Many of these classes are empty and are included only as documentation
of the interfaces.

$Id$
"""

version = '2.0beta'

#============================================================================
#
# HANDLER INTERFACES
#
#============================================================================

# ===== ERRORHANDLER =====

class ErrorHandler:
    """Basic interface for SAX error handlers.

    If you create an object that implements this interface, then
    register the object with your XMLReader, the parser will call the
    methods in your object to report all warnings and errors. There
    are three levels of errors available: warnings, (possibly)
    recoverable errors, and unrecoverable errors. All methods take a
    SAXParseException as the only parameter."""

    def error(self, exception):
        "Handle a recoverable error."
        raise exception

    def fatalError(self, exception):
        "Handle a non-recoverable error."
        raise exception

    def warning(self, exception):
        "Handle a warning."
        print(exception)


# ===== CONTENTHANDLER =====

class ContentHandler:
    """Interface for receiving logical document content events.

    This is the main callback interface in SAX, and the one most
    important to applications. The order of events in this interface
    mirrors the order of the information in the document."""

    def __init__(self):
        self._locator = None

    def setDocumentLocator(self, locator):
        """Called by the parser to give the application a locator for
        locating the origin of document events.

        SAX parsers are strongly encouraged (though not absolutely
        required) to supply a locator: if it does so, it must supply
        the locator to the application by invoking this method before
        invoking any of the other methods in the DocumentHandler
        interface.

        The locator allows the application to determine the end
        position of any document-related event, even if the parser is
        not reporting an error. Typically, the application will use
        this information for reporting its own errors (such as
        character content that does not match an application's
        business rules). The information returned by the locator is
        probably not sufficient for use with a search engine.

        Note that the locator will return correct information only
        during the invocation of the events in this interface. The
        application should not attempt to use it at any other time."""
        self._locator = locator

    def startDocument(self):
        """Receive notification of the beginning of a document.

        The SAX parser will invoke this method only once, before any
        other methods in this interface or in DTDHandler (except for
        setDocumentLocator)."""

    def endDocument(self):
        """Receive notification of the end of a document.

        The SAX parser will invoke this method only once, and it will
        be the last method invoked during the parse. The parser shall
        not invoke this method until it has either abandoned parsing
        (because of an unrecoverable error) or reached the end of
        input."""

    def startPrefixMapping(self, prefix, uri):
        """Begin the scope of a prefix-URI Namespace mapping.

        The information from this event is not necessary for normal
        Namespace processing: the SAX XML reader will automatically
        replace prefixes for element and attribute names when the
        http://xml.org/sax/features/namespaces feature is true (the
        default).

        There are cases, however, when applications need to use
        prefixes in character data or in attribute values, where they
        cannot safely be expanded automatically; the
        start/endPrefixMapping event supplies the information to the
        application to expand prefixes in those contexts itself, if
        necessary.

        Note that start/endPrefixMapping events are not guaranteed to
        be properly nested relative to each-other: all
        startPrefixMapping events will occur before the corresponding
        startElement event, and all endPrefixMapping events will occur
        after the corresponding endElement event, but their order is
        not guaranteed."""

    def endPrefixMapping(self, prefix):
        """End the scope of a prefix-URI mapping.

        See startPrefixMapping for details. This event will always
        occur after the corresponding endElement event, but the order
        of endPrefixMapping events is not otherwise guaranteed."""

    def startElement(self, name, attrs):
        """Signals the start of an element in non-namespace mode.

        The name parameter contains the raw XML 1.0 name of the
        element type as a string and the attrs parameter holds an
        instance of the Attributes class containing the attributes of
        the element."""

    def endElement(self, name):
        """Signals the end of an element in non-namespace mode.

        The name parameter contains the name of the element type, just
        as with the startElement event."""

    def startElementNS(self, name, qname, attrs):
        """Signals the start of an element in namespace mode.

        The name parameter contains the name of the element type as a
        (uri, localname) tuple, the qname parameter the raw XML 1.0
        name used in the source document, and the attrs parameter
        holds an instance of the Attributes class containing the
        attributes of the element.

        The uri part of the name tuple is None for elements which have
        no namespace."""

    def endElementNS(self, name, qname):
        """Signals the end of an element in namespace mode.

        The name parameter contains the name of the element type, just
        as with the startElementNS event."""

    def characters(self, content):
        """Receive notification of character data.

        The Parser will call this method to report each chunk of
        character data. SAX parsers may return all contiguous
        character data in a single chunk, or they may split it into
        several chunks; however, all of the characters in any single
        event must come from the same external entity so that the
        Locator provides useful information."""

    def ignorableWhitespace(self, whitespace):
        """Receive notification of ignorable whitespace in element content.

        Validating Parsers must use this method to report each chunk
        of ignorable whitespace (see the W3C XML 1.0 recommendation,
        section 2.10): non-validating parsers may also use this method
        if they are capable of parsing and using content models.

        SAX parsers may return all contiguous whitespace in a single
        chunk, or they may split it into several chunks; however, all
        of the characters in any single event must come from the same
        external entity, so that the Locator provides useful
        information."""

    def processingInstruction(self, target, data):
        """Receive notification of a processing instruction.

        The Parser will invoke this method once for each processing
        instruction found: note that processing instructions may occur
        before or after the main document element.

        A SAX parser should never report an XML declaration (XML 1.0,
        section 2.8) or a text declaration (XML 1.0, section 4.3.1)
        using this method."""

    def skippedEntity(self, name):
        """Receive notification of a skipped entity.

        The Parser will invoke this method once for each entity
        skipped. Non-validating processors may skip entities if they
        have not seen the declarations (because, for example, the
        entity was declared in an external DTD subset). All processors
        may skip external entities, depending on the values of the
        http://xml.org/sax/features/external-general-entities and the
        http://xml.org/sax/features/external-parameter-entities
        properties."""


# ===== DTDHandler =====

class DTDHandler:
    """Handle DTD events.

    This interface specifies only those DTD events required for basic
    parsing (unparsed entities and attributes)."""

    def notationDecl(self, name, publicId, systemId):
        "Handle a notation declaration event."

    def unparsedEntityDecl(self, name, publicId, systemId, ndata):
        "Handle an unparsed entity declaration event."


# ===== ENTITYRESOLVER =====

class EntityResolver:
    """Basic interface for resolving entities. If you create an object
    implementing this interface, then register the object with your
    Parser, the parser will call the method in your object to
    resolve all external entities. Note that DefaultHandler implements
    this interface with the default behaviour."""

    def resolveEntity(self, publicId, systemId):
        """Resolve the system identifier of an entity and return either
        the system identifier to read from as a string, or an InputSource
        to read from."""
        return systemId


#============================================================================
#
# CORE FEATURES
#
#============================================================================

feature_namespaces = "http://xml.org/sax/features/namespaces"
# true: Perform Namespace processing (default).
# false: Optionally do not perform Namespace processing
#        (implies namespace-prefixes).
# access: (parsing) read-only; (not parsing) read/write

feature_namespace_prefixes = "http://xml.org/sax/features/namespace-prefixes"
# true: Report the original prefixed names and attributes used for Namespace
#       declarations.
# false: Do not report attributes used for Namespace declarations, and
#        optionally do not report original prefixed names (default).
# access: (parsing) read-only; (not parsing) read/write

feature_string_interning = "http://xml.org/sax/features/string-interning"
# true: All element names, prefixes, attribute names, Namespace URIs, and
#       local names are interned using the built-in intern function.
# false: Names are not necessarily interned, although they may be (default).
# access: (parsing) read-only; (not parsing) read/write

feature_validation = "http://xml.org/sax/features/validation"
# true: Report all validation errors (implies external-general-entities and
#       external-parameter-entities).
# false: Do not report validation errors.
# access: (parsing) read-only; (not parsing) read/write

feature_external_ges = "http://xml.org/sax/features/external-general-entities"
# true: Include all external general (text) entities.
# false: Do not include external general entities.
# access: (parsing) read-only; (not parsing) read/write

feature_external_pes = "http://xml.org/sax/features/external-parameter-entities"
# true: Include all external parameter entities, including the external
#       DTD subset.
# false: Do not include any external parameter entities, even the external
#        DTD subset.
# access: (parsing) read-only; (not parsing) read/write

all_features = [feature_namespaces,
                feature_namespace_prefixes,
                feature_string_interning,
                feature_validation,
                feature_external_ges,
                feature_external_pes]


#============================================================================
#
# CORE PROPERTIES
#
#============================================================================

property_lexical_handler = "http://xml.org/sax/properties/lexical-handler"
# data type: xml.sax.sax2lib.LexicalHandler
# description: An optional extension handler for lexical events like comments.
# access: read/write

property_declaration_handler = "http://xml.org/sax/properties/declaration-handler"
# data type: xml.sax.sax2lib.DeclHandler
# description: An optional extension handler for DTD-related events other
#              than notations and unparsed entities.
# access: read/write

property_dom_node = "http://xml.org/sax/properties/dom-node"
# data type: org.w3c.dom.Node
# description: When parsing, the current DOM node being visited if this is
#              a DOM iterator; when not parsing, the root DOM node for
#              iteration.
# access: (parsing) read-only; (not parsing) read/write

property_xml_string = "http://xml.org/sax/properties/xml-string"
# data type: String
# description: The literal string of characters that was the source for
#              the current event.
# access: read-only

property_encoding = "http://www.python.org/sax/properties/encoding"
# data type: String
# description: The name of the encoding to assume for input data.
# access: write: set the encoding, e.g. established by a higher-level
#                protocol. May change during parsing (e.g. after
#                processing a META tag)
#         read:  return the current encoding (possibly established through
#                auto-detection.
# initial value: UTF-8
#

property_interning_dict = "http://www.python.org/sax/properties/interning-dict"
# data type: Dictionary
# description: The dictionary used to intern common strings in the document
# access: write: Request that the parser uses a specific dictionary, to
#                allow interning across different documents
#         read:  return the current interning dictionary, or None
#

all_properties = [property_lexical_handler,
                  property_dom_node,
                  property_declaration_handler,
                  property_xml_string,
                  property_encoding,
                  property_interning_dict]


class LexicalHandler:
    """Optional SAX2 handler for lexical events.

    This handler is used to obtain lexical information about an XML
    document, that is, information about how the document was encoded
    (as opposed to what it contains, which is reported to the
    ContentHandler), such as comments and CDATA marked section
    boundaries.

    To set the LexicalHandler of an XMLReader, use the setProperty
    method with the property identifier
    'http://xml.org/sax/properties/lexical-handler'."""

    def comment(self, content):
        """Reports a comment anywhere in the document (including the
        DTD and outside the document element).

        content is a string that holds the contents of the comment."""

    def startDTD(self, name, public_id, system_id):
        """Report the start of the DTD declarations, if the document
        has an associated DTD.

        A startEntity event will be reported before declaration events
        from the external DTD subset are reported, and this can be
        used to infer from which subset DTD declarations derive.

        name is the name of the document element type, public_id the
        public identifier of the DTD (or None if none were supplied)
        and system_id the system identfier of the external subset (or
        None if none were supplied)."""

    def endDTD(self):
        """Signals the end of DTD declarations."""

    def startCDATA(self):
        """Reports the beginning of a CDATA marked section.

        The contents of the CDATA marked section will be reported
        through the characters event."""

    def endCDATA(self):
        """Reports the end of a CDATA marked section."""
<|endoftext|>"""Constants and membership tests for ASCII characters"""

NUL     = 0x00  # ^@
SOH     = 0x01  # ^A
STX     = 0x02  # ^B
ETX     = 0x03  # ^C
EOT     = 0x04  # ^D
ENQ     = 0x05  # ^E
ACK     = 0x06  # ^F
BEL     = 0x07  # ^G
BS      = 0x08  # ^H
TAB     = 0x09  # ^I
HT      = 0x09  # ^I
LF      = 0x0a  # ^J
NL      = 0x0a  # ^J
VT      = 0x0b  # ^K
FF      = 0x0c  # ^L
CR      = 0x0d  # ^M
SO      = 0x0e  # ^N
SI      = 0x0f  # ^O
DLE     = 0x10  # ^P
DC1     = 0x11  # ^Q
DC2     = 0x12  # ^R
DC3     = 0x13  # ^S
DC4     = 0x14  # ^T
NAK     = 0x15  # ^U
SYN     = 0x16  # ^V
ETB     = 0x17  # ^W
CAN     = 0x18  # ^X
EM      = 0x19  # ^Y
SUB     = 0x1a  # ^Z
ESC     = 0x1b  # ^[
FS      = 0x1c  # ^\
GS      = 0x1d  # ^]
RS      = 0x1e  # ^^
US      = 0x1f  # ^_
SP      = 0x20  # space
DEL     = 0x7f  # delete

controlnames = [
"NUL", "SOH", "STX", "ETX", "EOT", "ENQ", "ACK", "BEL",
"BS",  "HT",  "LF",  "VT",  "FF",  "CR",  "SO",  "SI",
"DLE", "DC1", "DC2", "DC3", "DC4", "NAK", "SYN", "ETB",
"CAN", "EM",  "SUB", "ESC", "FS",  "GS",  "RS",  "US",
"SP"
]

def _ctoi(c):
    if type(c) == type(""):
        return ord(c)
    else:
        return c

def isalnum(c): return isalpha(c) or isdigit(c)
def isalpha(c): return isupper(c) or islower(c)
def isascii(c): return 0 <= _ctoi(c) <= 127          # ?
def isblank(c): return _ctoi(c) in (9, 32)
def iscntrl(c): return 0 <= _ctoi(c) <= 31 or _ctoi(c) == 127
def isdigit(c): return 48 <= _ctoi(c) <= 57
def isgraph(c): return 33 <= _ctoi(c) <= 126
def islower(c): return 97 <= _ctoi(c) <= 122
def isprint(c): return 32 <= _ctoi(c) <= 126
def ispunct(c): return isgraph(c) and not isalnum(c)
def isspace(c): return _ctoi(c) in (9, 10, 11, 12, 13, 32)
def isupper(c): return 65 <= _ctoi(c) <= 90
def isxdigit(c): return isdigit(c) or \
    (65 <= _ctoi(c) <= 70) or (97 <= _ctoi(c) <= 102)
def isctrl(c): return 0 <= _ctoi(c) < 32
def ismeta(c): return _ctoi(c) > 127

def ascii(c):
    if type(c) == type(""):
        return chr(_ctoi(c) & 0x7f)
    else:
        return _ctoi(c) & 0x7f

def ctrl(c):
    if type(c) == type(""):
        return chr(_ctoi(c) & 0x1f)
    else:
        return _ctoi(c) & 0x1f

def alt(c):
    if type(c) == type(""):
        return chr(_ctoi(c) | 0x80)
    else:
        return _ctoi(c) | 0x80

def unctrl(c):
    bits = _ctoi(c)
    if bits == 0x7f:
        rep = "^?"
    elif isprint(bits & 0x7f):
        rep = chr(bits & 0x7f)
    else:
        rep = "^" + chr(((bits & 0x7f) | 0x20) + 0x20)
    if bits & 0x80:
        return "!" + rep
    return rep
<|endoftext|>"""Simple textbox editing widget with Emacs-like keybindings."""

import curses
import curses.ascii

def rectangle(win, uly, ulx, lry, lrx):
    """Draw a rectangle with corners at the provided upper-left
    and lower-right coordinates.
    """
    win.vline(uly+1, ulx, curses.ACS_VLINE, lry - uly - 1)
    win.hline(uly, ulx+1, curses.ACS_HLINE, lrx - ulx - 1)
    win.hline(lry, ulx+1, curses.ACS_HLINE, lrx - ulx - 1)
    win.vline(uly+1, lrx, curses.ACS_VLINE, lry - uly - 1)
    win.addch(uly, ulx, curses.ACS_ULCORNER)
    win.addch(uly, lrx, curses.ACS_URCORNER)
    win.addch(lry, lrx, curses.ACS_LRCORNER)
    win.addch(lry, ulx, curses.ACS_LLCORNER)

class Textbox:
    """Editing widget using the interior of a window object.
     Supports the following Emacs-like key bindings:

    Ctrl-A      Go to left edge of window.
    Ctrl-B      Cursor left, wrapping to previous line if appropriate.
    Ctrl-D      Delete character under cursor.
    Ctrl-E      Go to right edge (stripspaces off) or end of line (stripspaces on).
    Ctrl-F      Cursor right, wrapping to next line when appropriate.
    Ctrl-G      Terminate, returning the window contents.
    Ctrl-H      Delete character backward.
    Ctrl-J      Terminate if the window is 1 line, otherwise insert newline.
    Ctrl-K      If line is blank, delete it, otherwise clear to end of line.
    Ctrl-L      Refresh screen.
    Ctrl-N      Cursor down; move down one line.
    Ctrl-O      Insert a blank line at cursor location.
    Ctrl-P      Cursor up; move up one line.

    Move operations do nothing if the cursor is at an edge where the movement
    is not possible.  The following synonyms are supported where possible:

    KEY_LEFT = Ctrl-B, KEY_RIGHT = Ctrl-F, KEY_UP = Ctrl-P, KEY_DOWN = Ctrl-N
    KEY_BACKSPACE = Ctrl-h
    """
    def __init__(self, win, insert_mode=False):
        self.win = win
        self.insert_mode = insert_mode
        self._update_max_yx()
        self.stripspaces = 1
        self.lastcmd = None
        win.keypad(1)

    def _update_max_yx(self):
        maxy, maxx = self.win.getmaxyx()
        self.maxy = maxy - 1
        self.maxx = maxx - 1

    def _end_of_line(self, y):
        """Go to the location of the first blank on the given line,
        returning the index of the last non-blank character."""
        self._update_max_yx()
        last = self.maxx
        while True:
            if curses.ascii.ascii(self.win.inch(y, last)) != curses.ascii.SP:
                last = min(self.maxx, last+1)
                break
            elif last == 0:
                break
            last = last - 1
        return last

    def _insert_printable_char(self, ch):
        self._update_max_yx()
        (y, x) = self.win.getyx()
        backyx = None
        while y < self.maxy or x < self.maxx:
            if self.insert_mode:
                oldch = self.win.inch()
            # The try-catch ignores the error we trigger from some curses
            # versions by trying to write into the lowest-rightmost spot
            # in the window.
            try:
                self.win.addch(ch)
            except curses.error:
                pass
            if not self.insert_mode or not curses.ascii.isprint(oldch):
                break
            ch = oldch
            (y, x) = self.win.getyx()
            # Remember where to put the cursor back since we are in insert_mode
            if backyx is None:
                backyx = y, x

        if backyx is not None:
            self.win.move(*backyx)

    def do_command(self, ch):
        "Process a single editing command."
        self._update_max_yx()
        (y, x) = self.win.getyx()
        self.lastcmd = ch
        if curses.ascii.isprint(ch):
            if y < self.maxy or x < self.maxx:
                self._insert_printable_char(ch)
        elif ch == curses.ascii.SOH:                           # ^a
            self.win.move(y, 0)
        elif ch in (curses.ascii.STX,curses.KEY_LEFT, curses.ascii.BS,curses.KEY_BACKSPACE):
            if x > 0:
                self.win.move(y, x-1)
            elif y == 0:
                pass
            elif self.stripspaces:
                self.win.move(y-1, self._end_of_line(y-1))
            else:
                self.win.move(y-1, self.maxx)
            if ch in (curses.ascii.BS, curses.KEY_BACKSPACE):
                self.win.delch()
        elif ch == curses.ascii.EOT:                           # ^d
            self.win.delch()
        elif ch == curses.ascii.ENQ:                           # ^e
            if self.stripspaces:
                self.win.move(y, self._end_of_line(y))
            else:
                self.win.move(y, self.maxx)
        elif ch in (curses.ascii.ACK, curses.KEY_RIGHT):       # ^f
            if x < self.maxx:
                self.win.move(y, x+1)
            elif y == self.maxy:
                pass
            else:
                self.win.move(y+1, 0)
        elif ch == curses.ascii.BEL:                           # ^g
            return 0
        elif ch == curses.ascii.NL:                            # ^j
            if self.maxy == 0:
                return 0
            elif y < self.maxy:
                self.win.move(y+1, 0)
        elif ch == curses.ascii.VT:                            # ^k
            if x == 0 and self._end_of_line(y) == 0:
                self.win.deleteln()
            else:
                # first undo the effect of self._end_of_line
                self.win.move(y, x)
                self.win.clrtoeol()
        elif ch == curses.ascii.FF:                            # ^l
            self.win.refresh()
        elif ch in (curses.ascii.SO, curses.KEY_DOWN):         # ^n
            if y < self.maxy:
                self.win.move(y+1, x)
                if x > self._end_of_line(y+1):
                    self.win.move(y+1, self._end_of_line(y+1))
        elif ch == curses.ascii.SI:                            # ^o
            self.win.insertln()
        elif ch in (curses.ascii.DLE, curses.KEY_UP):          # ^p
            if y > 0:
                self.win.move(y-1, x)
                if x > self._end_of_line(y-1):
                    self.win.move(y-1, self._end_of_line(y-1))
        return 1

    def gather(self):
        "Collect and return the contents of the window."
        result = ""
        self._update_max_yx()
        for y in range(self.maxy+1):
            self.win.move(y, 0)
            stop = self._end_of_line(y)
            if stop == 0 and self.stripspaces:
                continue
            for x in range(self.maxx+1):
                if self.stripspaces and x > stop:
                    break
                result = result + chr(curses.ascii.ascii(self.win.inch(y, x)))
            if self.maxy > 0:
                result = result + "\n"
        return result

    def edit(self, validate=None):
        "Edit in the widget window and collect the results."
        while 1:
            ch = self.win.getch()
            if validate:
                ch = validate(ch)
            if not ch:
                continue
            if not self.do_command(ch):
                break
            self.win.refresh()
        return self.gather()

if __name__ == '__main__':
    def test_editbox(stdscr):
        ncols, nlines = 9, 4
        uly, ulx = 15, 20
        stdscr.addstr(uly-2, ulx, "Use Ctrl-G to end editing.")
        win = curses.newwin(nlines, ncols, uly, ulx)
        rectangle(stdscr, uly-1, ulx-1, uly + nlines, ulx + ncols)
        stdscr.refresh()
        return Textbox(win).edit()

    str = curses.wrapper(test_editbox)
    print('Contents of text box:', repr(str))
<|endoftext|>"""curses

The main package for curses support for Python.  Normally used by importing
the package, and perhaps a particular module inside it.

   import curses
   from curses import textpad
   curses.initscr()
   ...

"""

from _curses import *
import os as _os
import sys as _sys

# Some constants, most notably the ACS_* ones, are only added to the C
# _curses module's dictionary after initscr() is called.  (Some
# versions of SGI's curses don't define values for those constants
# until initscr() has been called.)  This wrapper function calls the
# underlying C initscr(), and then copies the constants from the
# _curses module to the curses package's dictionary.  Don't do 'from
# curses import *' if you'll be needing the ACS_* constants.

def initscr():
    import _curses, curses
    # we call setupterm() here because it raises an error
    # instead of calling exit() in error cases.
    setupterm(term=_os.environ.get("TERM", "unknown"),
              fd=_sys.__stdout__.fileno())
    stdscr = _curses.initscr()
    for key, value in _curses.__dict__.items():
        if key[0:4] == 'ACS_' or key in ('LINES', 'COLS'):
            setattr(curses, key, value)

    return stdscr

# This is a similar wrapper for start_color(), which adds the COLORS and
# COLOR_PAIRS variables which are only available after start_color() is
# called.

def start_color():
    import _curses, curses
    retval = _curses.start_color()
    if hasattr(_curses, 'COLORS'):
        curses.COLORS = _curses.COLORS
    if hasattr(_curses, 'COLOR_PAIRS'):
        curses.COLOR_PAIRS = _curses.COLOR_PAIRS
    return retval

# Import Python has_key() implementation if _curses doesn't contain has_key()

try:
    has_key
except NameError:
    from .has_key import has_key

# Wrapper for the entire curses-based application.  Runs a function which
# should be the rest of your curses-based application.  If the application
# raises an exception, wrapper() will restore the terminal to a sane state so
# you can read the resulting traceback.

def wrapper(func, /, *args, **kwds):
    """Wrapper function that initializes curses and calls another function,
    restoring normal keyboard/screen behavior on error.
    The callable object 'func' is then passed the main window 'stdscr'
    as its first argument, followed by any other arguments passed to
    wrapper().
    """

    try:
        # Initialize curses
        stdscr = initscr()

        # Turn off echoing of keys, and enter cbreak mode,
        # where no buffering is performed on keyboard input
        noecho()
        cbreak()

        # In keypad mode, escape sequences for special keys
        # (like the cursor keys) will be interpreted and
        # a special value like curses.KEY_LEFT will be returned
        stdscr.keypad(1)

        # Start color, too.  Harmless if the terminal doesn't have
        # color; user can test with has_color() later on.  The try/catch
        # works around a minor bit of over-conscientiousness in the curses
        # module -- the error return from C start_color() is ignorable.
        try:
            start_color()
        except:
            pass

        return func(stdscr, *args, **kwds)
    finally:
        # Set everything back to normal
        if 'stdscr' in locals():
            stdscr.keypad(0)
            echo()
            nocbreak()
            endwin()
<|endoftext|>
#
# Emulation of has_key() function for platforms that don't use ncurses
#

import _curses

# Table mapping curses keys to the terminfo capability name

_capability_names = {
    _curses.KEY_A1: 'ka1',
    _curses.KEY_A3: 'ka3',
    _curses.KEY_B2: 'kb2',
    _curses.KEY_BACKSPACE: 'kbs',
    _curses.KEY_BEG: 'kbeg',
    _curses.KEY_BTAB: 'kcbt',
    _curses.KEY_C1: 'kc1',
    _curses.KEY_C3: 'kc3',
    _curses.KEY_CANCEL: 'kcan',
    _curses.KEY_CATAB: 'ktbc',
    _curses.KEY_CLEAR: 'kclr',
    _curses.KEY_CLOSE: 'kclo',
    _curses.KEY_COMMAND: 'kcmd',
    _curses.KEY_COPY: 'kcpy',
    _curses.KEY_CREATE: 'kcrt',
    _curses.KEY_CTAB: 'kctab',
    _curses.KEY_DC: 'kdch1',
    _curses.KEY_DL: 'kdl1',
    _curses.KEY_DOWN: 'kcud1',
    _curses.KEY_EIC: 'krmir',
    _curses.KEY_END: 'kend',
    _curses.KEY_ENTER: 'kent',
    _curses.KEY_EOL: 'kel',
    _curses.KEY_EOS: 'ked',
    _curses.KEY_EXIT: 'kext',
    _curses.KEY_F0: 'kf0',
    _curses.KEY_F1: 'kf1',
    _curses.KEY_F10: 'kf10',
    _curses.KEY_F11: 'kf11',
    _curses.KEY_F12: 'kf12',
    _curses.KEY_F13: 'kf13',
    _curses.KEY_F14: 'kf14',
    _curses.KEY_F15: 'kf15',
    _curses.KEY_F16: 'kf16',
    _curses.KEY_F17: 'kf17',
    _curses.KEY_F18: 'kf18',
    _curses.KEY_F19: 'kf19',
    _curses.KEY_F2: 'kf2',
    _curses.KEY_F20: 'kf20',
    _curses.KEY_F21: 'kf21',
    _curses.KEY_F22: 'kf22',
    _curses.KEY_F23: 'kf23',
    _curses.KEY_F24: 'kf24',
    _curses.KEY_F25: 'kf25',
    _curses.KEY_F26: 'kf26',
    _curses.KEY_F27: 'kf27',
    _curses.KEY_F28: 'kf28',
    _curses.KEY_F29: 'kf29',
    _curses.KEY_F3: 'kf3',
    _curses.KEY_F30: 'kf30',
    _curses.KEY_F31: 'kf31',
    _curses.KEY_F32: 'kf32',
    _curses.KEY_F33: 'kf33',
    _curses.KEY_F34: 'kf34',
    _curses.KEY_F35: 'kf35',
    _curses.KEY_F36: 'kf36',
    _curses.KEY_F37: 'kf37',
    _curses.KEY_F38: 'kf38',
    _curses.KEY_F39: 'kf39',
    _curses.KEY_F4: 'kf4',
    _curses.KEY_F40: 'kf40',
    _curses.KEY_F41: 'kf41',
    _curses.KEY_F42: 'kf42',
    _curses.KEY_F43: 'kf43',
    _curses.KEY_F44: 'kf44',
    _curses.KEY_F45: 'kf45',
    _curses.KEY_F46: 'kf46',
    _curses.KEY_F47: 'kf47',
    _curses.KEY_F48: 'kf48',
    _curses.KEY_F49: 'kf49',
    _curses.KEY_F5: 'kf5',
    _curses.KEY_F50: 'kf50',
    _curses.KEY_F51: 'kf51',
    _curses.KEY_F52: 'kf52',
    _curses.KEY_F53: 'kf53',
    _curses.KEY_F54: 'kf54',
    _curses.KEY_F55: 'kf55',
    _curses.KEY_F56: 'kf56',
    _curses.KEY_F57: 'kf57',
    _curses.KEY_F58: 'kf58',
    _curses.KEY_F59: 'kf59',
    _curses.KEY_F6: 'kf6',
    _curses.KEY_F60: 'kf60',
    _curses.KEY_F61: 'kf61',
    _curses.KEY_F62: 'kf62',
    _curses.KEY_F63: 'kf63',
    _curses.KEY_F7: 'kf7',
    _curses.KEY_F8: 'kf8',
    _curses.KEY_F9: 'kf9',
    _curses.KEY_FIND: 'kfnd',
    _curses.KEY_HELP: 'khlp',
    _curses.KEY_HOME: 'khome',
    _curses.KEY_IC: 'kich1',
    _curses.KEY_IL: 'kil1',
    _curses.KEY_LEFT: 'kcub1',
    _curses.KEY_LL: 'kll',
    _curses.KEY_MARK: 'kmrk',
    _curses.KEY_MESSAGE: 'kmsg',
    _curses.KEY_MOVE: 'kmov',
    _curses.KEY_NEXT: 'knxt',
    _curses.KEY_NPAGE: 'knp',
    _curses.KEY_OPEN: 'kopn',
    _curses.KEY_OPTIONS: 'kopt',
    _curses.KEY_PPAGE: 'kpp',
    _curses.KEY_PREVIOUS: 'kprv',
    _curses.KEY_PRINT: 'kprt',
    _curses.KEY_REDO: 'krdo',
    _curses.KEY_REFERENCE: 'kref',
    _curses.KEY_REFRESH: 'krfr',
    _curses.KEY_REPLACE: 'krpl',
    _curses.KEY_RESTART: 'krst',
    _curses.KEY_RESUME: 'kres',
    _curses.KEY_RIGHT: 'kcuf1',
    _curses.KEY_SAVE: 'ksav',
    _curses.KEY_SBEG: 'kBEG',
    _curses.KEY_SCANCEL: 'kCAN',
    _curses.KEY_SCOMMAND: 'kCMD',
    _curses.KEY_SCOPY: 'kCPY',
    _curses.KEY_SCREATE: 'kCRT',
    _curses.KEY_SDC: 'kDC',
    _curses.KEY_SDL: 'kDL',
    _curses.KEY_SELECT: 'kslt',
    _curses.KEY_SEND: 'kEND',
    _curses.KEY_SEOL: 'kEOL',
    _curses.KEY_SEXIT: 'kEXT',
    _curses.KEY_SF: 'kind',
    _curses.KEY_SFIND: 'kFND',
    _curses.KEY_SHELP: 'kHLP',
    _curses.KEY_SHOME: 'kHOM',
    _curses.KEY_SIC: 'kIC',
    _curses.KEY_SLEFT: 'kLFT',
    _curses.KEY_SMESSAGE: 'kMSG',
    _curses.KEY_SMOVE: 'kMOV',
    _curses.KEY_SNEXT: 'kNXT',
    _curses.KEY_SOPTIONS: 'kOPT',
    _curses.KEY_SPREVIOUS: 'kPRV',
    _curses.KEY_SPRINT: 'kPRT',
    _curses.KEY_SR: 'kri',
    _curses.KEY_SREDO: 'kRDO',
    _curses.KEY_SREPLACE: 'kRPL',
    _curses.KEY_SRIGHT: 'kRIT',
    _curses.KEY_SRSUME: 'kRES',
    _curses.KEY_SSAVE: 'kSAV',
    _curses.KEY_SSUSPEND: 'kSPD',
    _curses.KEY_STAB: 'khts',
    _curses.KEY_SUNDO: 'kUND',
    _curses.KEY_SUSPEND: 'kspd',
    _curses.KEY_UNDO: 'kund',
    _curses.KEY_UP: 'kcuu1'
    }

def has_key(ch):
    if isinstance(ch, str):
        ch = ord(ch)

    # Figure out the correct capability name for the keycode.
    capability_name = _capability_names.get(ch)
    if capability_name is None:
        return False

    #Check the current terminal description for that capability;
    #if present, return true, else return false.
    if _curses.tigetstr( capability_name ):
        return True
    else:
        return False

if __name__ == '__main__':
    # Compare the output of this implementation and the ncurses has_key,
    # on platforms where has_key is already available
    try:
        L = []
        _curses.initscr()
        for key in _capability_names.keys():
            system = _curses.has_key(key)
            python = has_key(key)
            if system != python:
                L.append( 'Mismatch for key %s, system=%i, Python=%i'
                          % (_curses.keyname( key ), system, python) )
    finally:
        _curses.endwin()
        for i in L: print(i)
<|endoftext|>"""Miscellaneous WSGI-related Utilities"""

import posixpath

__all__ = [
    'FileWrapper', 'guess_scheme', 'application_uri', 'request_uri',
    'shift_path_info', 'setup_testing_defaults',
]


class FileWrapper:
    """Wrapper to convert file-like objects to iterables"""

    def __init__(self, filelike, blksize=8192):
        self.filelike = filelike
        self.blksize = blksize
        if hasattr(filelike,'close'):
            self.close = filelike.close

    def __getitem__(self,key):
        import warnings
        warnings.warn(
            "FileWrapper's __getitem__ method ignores 'key' parameter. "
            "Use iterator protocol instead.",
            DeprecationWarning,
            stacklevel=2
        )
        data = self.filelike.read(self.blksize)
        if data:
            return data
        raise IndexError

    def __iter__(self):
        return self

    def __next__(self):
        data = self.filelike.read(self.blksize)
        if data:
            return data
        raise StopIteration

def guess_scheme(environ):
    """Return a guess for whether 'wsgi.url_scheme' should be 'http' or 'https'
    """
    if environ.get("HTTPS") in ('yes','on','1'):
        return 'https'
    else:
        return 'http'

def application_uri(environ):
    """Return the application's base URI (no PATH_INFO or QUERY_STRING)"""
    url = environ['wsgi.url_scheme']+'://'
    from urllib.parse import quote

    if environ.get('HTTP_HOST'):
        url += environ['HTTP_HOST']
    else:
        url += environ['SERVER_NAME']

        if environ['wsgi.url_scheme'] == 'https':
            if environ['SERVER_PORT'] != '443':
                url += ':' + environ['SERVER_PORT']
        else:
            if environ['SERVER_PORT'] != '80':
                url += ':' + environ['SERVER_PORT']

    url += quote(environ.get('SCRIPT_NAME') or '/', encoding='latin1')
    return url

def request_uri(environ, include_query=True):
    """Return the full request URI, optionally including the query string"""
    url = application_uri(environ)
    from urllib.parse import quote
    path_info = quote(environ.get('PATH_INFO',''), safe='/;=,', encoding='latin1')
    if not environ.get('SCRIPT_NAME'):
        url += path_info[1:]
    else:
        url += path_info
    if include_query and environ.get('QUERY_STRING'):
        url += '?' + environ['QUERY_STRING']
    return url

def shift_path_info(environ):
    """Shift a name from PATH_INFO to SCRIPT_NAME, returning it

    If there are no remaining path segments in PATH_INFO, return None.
    Note: 'environ' is modified in-place; use a copy if you need to keep
    the original PATH_INFO or SCRIPT_NAME.

    Note: when PATH_INFO is just a '/', this returns '' and appends a trailing
    '/' to SCRIPT_NAME, even though empty path segments are normally ignored,
    and SCRIPT_NAME doesn't normally end in a '/'.  This is intentional
    behavior, to ensure that an application can tell the difference between
    '/x' and '/x/' when traversing to objects.
    """
    path_info = environ.get('PATH_INFO','')
    if not path_info:
        return None

    path_parts = path_info.split('/')
    path_parts[1:-1] = [p for p in path_parts[1:-1] if p and p != '.']
    name = path_parts[1]
    del path_parts[1]

    script_name = environ.get('SCRIPT_NAME','')
    script_name = posixpath.normpath(script_name+'/'+name)
    if script_name.endswith('/'):
        script_name = script_name[:-1]
    if not name and not script_name.endswith('/'):
        script_name += '/'

    environ['SCRIPT_NAME'] = script_name
    environ['PATH_INFO']   = '/'.join(path_parts)

    # Special case: '/.' on PATH_INFO doesn't get stripped,
    # because we don't strip the last element of PATH_INFO
    # if there's only one path part left.  Instead of fixing this
    # above, we fix it here so that PATH_INFO gets normalized to
    # an empty string in the environ.
    if name=='.':
        name = None
    return name

def setup_testing_defaults(environ):
    """Update 'environ' with trivial defaults for testing purposes

    This adds various parameters required for WSGI, including HTTP_HOST,
    SERVER_NAME, SERVER_PORT, REQUEST_METHOD, SCRIPT_NAME, PATH_INFO,
    and all of the wsgi.* variables.  It only supplies default values,
    and does not replace any existing settings for these variables.

    This routine is intended to make it easier for unit tests of WSGI
    servers and applications to set up dummy environments.  It should *not*
    be used by actual WSGI servers or applications, since the data is fake!
    """

    environ.setdefault('SERVER_NAME','127.0.0.1')
    environ.setdefault('SERVER_PROTOCOL','HTTP/1.0')

    environ.setdefault('HTTP_HOST',environ['SERVER_NAME'])
    environ.setdefault('REQUEST_METHOD','GET')

    if 'SCRIPT_NAME' not in environ and 'PATH_INFO' not in environ:
        environ.setdefault('SCRIPT_NAME','')
        environ.setdefault('PATH_INFO','/')

    environ.setdefault('wsgi.version', (1,0))
    environ.setdefault('wsgi.run_once', 0)
    environ.setdefault('wsgi.multithread', 0)
    environ.setdefault('wsgi.multiprocess', 0)

    from io import StringIO, BytesIO
    environ.setdefault('wsgi.input', BytesIO())
    environ.setdefault('wsgi.errors', StringIO())
    environ.setdefault('wsgi.url_scheme',guess_scheme(environ))

    if environ['wsgi.url_scheme']=='http':
        environ.setdefault('SERVER_PORT', '80')
    elif environ['wsgi.url_scheme']=='https':
        environ.setdefault('SERVER_PORT', '443')



_hoppish = {
    'connection', 'keep-alive', 'proxy-authenticate',
    'proxy-authorization', 'te', 'trailers', 'transfer-encoding',
    'upgrade'
}.__contains__

def is_hop_by_hop(header_name):
    """Return true if 'header_name' is an HTTP/1.1 "Hop-by-Hop" header"""
    return _hoppish(header_name.lower())
<|endoftext|>"""BaseHTTPServer that implements the Python WSGI protocol (PEP 3333)

This is both an example of how WSGI can be implemented, and a basis for running
simple web applications on a local machine, such as might be done when testing
or debugging an application.  It has not been reviewed for security issues,
however, and we strongly recommend that you use a "real" web server for
production use.

For example usage, see the 'if __name__=="__main__"' block at the end of the
module.  See also the BaseHTTPServer module docs for other API information.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import urllib.parse
from wsgiref.handlers import SimpleHandler
from platform import python_implementation

__version__ = "0.2"
__all__ = ['WSGIServer', 'WSGIRequestHandler', 'demo_app', 'make_server']


server_version = "WSGIServer/" + __version__
sys_version = python_implementation() + "/" + sys.version.split()[0]
software_version = server_version + ' ' + sys_version


class ServerHandler(SimpleHandler):

    server_software = software_version

    def close(self):
        try:
            self.request_handler.log_request(
                self.status.split(' ',1)[0], self.bytes_sent
            )
        finally:
            SimpleHandler.close(self)



class WSGIServer(HTTPServer):

    """BaseHTTPServer that implements the Python WSGI protocol"""

    application = None

    def server_bind(self):
        """Override server_bind to store the server name."""
        HTTPServer.server_bind(self)
        self.setup_environ()

    def setup_environ(self):
        # Set up base environment
        env = self.base_environ = {}
        env['SERVER_NAME'] = self.server_name
        env['GATEWAY_INTERFACE'] = 'CGI/1.1'
        env['SERVER_PORT'] = str(self.server_port)
        env['REMOTE_HOST']=''
        env['CONTENT_LENGTH']=''
        env['SCRIPT_NAME'] = ''

    def get_app(self):
        return self.application

    def set_app(self,application):
        self.application = application



class WSGIRequestHandler(BaseHTTPRequestHandler):

    server_version = "WSGIServer/" + __version__

    def get_environ(self):
        env = self.server.base_environ.copy()
        env['SERVER_PROTOCOL'] = self.request_version
        env['SERVER_SOFTWARE'] = self.server_version
        env['REQUEST_METHOD'] = self.command
        if '?' in self.path:
            path,query = self.path.split('?',1)
        else:
            path,query = self.path,''

        env['PATH_INFO'] = urllib.parse.unquote(path, 'iso-8859-1')
        env['QUERY_STRING'] = query

        host = self.address_string()
        if host != self.client_address[0]:
            env['REMOTE_HOST'] = host
        env['REMOTE_ADDR'] = self.client_address[0]

        if self.headers.get('content-type') is None:
            env['CONTENT_TYPE'] = self.headers.get_content_type()
        else:
            env['CONTENT_TYPE'] = self.headers['content-type']

        length = self.headers.get('content-length')
        if length:
            env['CONTENT_LENGTH'] = length

        for k, v in self.headers.items():
            k=k.replace('-','_').upper(); v=v.strip()
            if k in env:
                continue                    # skip content length, type,etc.
            if 'HTTP_'+k in env:
                env['HTTP_'+k] += ','+v     # comma-separate multiple headers
            else:
                env['HTTP_'+k] = v
        return env

    def get_stderr(self):
        return sys.stderr

    def handle(self):
        """Handle a single HTTP request"""

        self.raw_requestline = self.rfile.readline(65537)
        if len(self.raw_requestline) > 65536:
            self.requestline = ''
            self.request_version = ''
            self.command = ''
            self.send_error(414)
            return

        if not self.parse_request(): # An error code has been sent, just exit
            return

        handler = ServerHandler(
            self.rfile, self.wfile, self.get_stderr(), self.get_environ(),
            multithread=False,
        )
        handler.request_handler = self      # backpointer for logging
        handler.run(self.server.get_app())



def demo_app(environ,start_response):
    from io import StringIO
    stdout = StringIO()
    print("Hello world!", file=stdout)
    print(file=stdout)
    h = sorted(environ.items())
    for k,v in h:
        print(k,'=',repr(v), file=stdout)
    start_response("200 OK", [('Content-Type','text/plain; charset=utf-8')])
    return [stdout.getvalue().encode("utf-8")]


def make_server(
    host, port, app, server_class=WSGIServer, handler_class=WSGIRequestHandler
):
    """Create a new WSGI server listening on `host` and `port` for `app`"""
    server = server_class((host, port), handler_class)
    server.set_app(app)
    return server


if __name__ == '__main__':
    with make_server('', 8000, demo_app) as httpd:
        sa = httpd.socket.getsockname()
        print("Serving HTTP on", sa[0], "port", sa[1], "...")
        import webbrowser
        webbrowser.open('http://localhost:8000/xyz?abc')
        httpd.handle_request()  # serve one request, then exit
<|endoftext|># (c) 2005 Ian Bicking and contributors; written for Paste (http://pythonpaste.org)
# Licensed under the MIT license: https://opensource.org/licenses/mit-license.php
# Also licenced under the Apache License, 2.0: https://opensource.org/licenses/apache2.0.php
# Licensed to PSF under a Contributor Agreement
"""
Middleware to check for obedience to the WSGI specification.

Some of the things this checks:

* Signature of the application and start_response (including that
  keyword arguments are not used).

* Environment checks:

  - Environment is a dictionary (and not a subclass).

  - That all the required keys are in the environment: REQUEST_METHOD,
    SERVER_NAME, SERVER_PORT, wsgi.version, wsgi.input, wsgi.errors,
    wsgi.multithread, wsgi.multiprocess, wsgi.run_once

  - That HTTP_CONTENT_TYPE and HTTP_CONTENT_LENGTH are not in the
    environment (these headers should appear as CONTENT_LENGTH and
    CONTENT_TYPE).

  - Warns if QUERY_STRING is missing, as the cgi module acts
    unpredictably in that case.

  - That CGI-style variables (that don't contain a .) have
    (non-unicode) string values

  - That wsgi.version is a tuple

  - That wsgi.url_scheme is 'http' or 'https' (@@: is this too
    restrictive?)

  - Warns if the REQUEST_METHOD is not known (@@: probably too
    restrictive).

  - That SCRIPT_NAME and PATH_INFO are empty or start with /

  - That at least one of SCRIPT_NAME or PATH_INFO are set.

  - That CONTENT_LENGTH is a positive integer.

  - That SCRIPT_NAME is not '/' (it should be '', and PATH_INFO should
    be '/').

  - That wsgi.input has the methods read, readline, readlines, and
    __iter__

  - That wsgi.errors has the methods flush, write, writelines

* The status is a string, contains a space, starts with an integer,
  and that integer is in range (> 100).

* That the headers is a list (not a subclass, not another kind of
  sequence).

* That the items of the headers are tuples of strings.

* That there is no 'status' header (that is used in CGI, but not in
  WSGI).

* That the headers don't contain newlines or colons, end in _ or -, or
  contain characters codes below 037.

* That Content-Type is given if there is content (CGI often has a
  default content type, but WSGI does not).

* That no Content-Type is given when there is no content (@@: is this
  too restrictive?)

* That the exc_info argument to start_response is a tuple or None.

* That all calls to the writer are with strings, and no other methods
  on the writer are accessed.

* That wsgi.input is used properly:

  - .read() is called with exactly one argument

  - That it returns a string

  - That readline, readlines, and __iter__ return strings

  - That .close() is not called

  - No other methods are provided

* That wsgi.errors is used properly:

  - .write() and .writelines() is called with a string

  - That .close() is not called, and no other methods are provided.

* The response iterator:

  - That it is not a string (it should be a list of a single string; a
    string will work, but perform horribly).

  - That .__next__() returns a string

  - That the iterator is not iterated over until start_response has
    been called (that can signal either a server or application
    error).

  - That .close() is called (doesn't raise exception, only prints to
    sys.stderr, because we only know it isn't called when the object
    is garbage collected).
"""
__all__ = ['validator']


import re
import sys
import warnings

header_re = re.compile(r'^[a-zA-Z][a-zA-Z0-9\-_]*$')
bad_header_value_re = re.compile(r'[\000-\037]')

class WSGIWarning(Warning):
    """
    Raised in response to WSGI-spec-related warnings
    """

def assert_(cond, *args):
    if not cond:
        raise AssertionError(*args)

def check_string_type(value, title):
    if type (value) is str:
        return value
    raise AssertionError(
        "{0} must be of type str (got {1})".format(title, repr(value)))

def validator(application):

    """
    When applied between a WSGI server and a WSGI application, this
    middleware will check for WSGI compliance on a number of levels.
    This middleware does not modify the request or response in any
    way, but will raise an AssertionError if anything seems off
    (except for a failure to close the application iterator, which
    will be printed to stderr -- there's no way to raise an exception
    at that point).
    """

    def lint_app(*args, **kw):
        assert_(len(args) == 2, "Two arguments required")
        assert_(not kw, "No keyword arguments allowed")
        environ, start_response = args

        check_environ(environ)

        # We use this to check if the application returns without
        # calling start_response:
        start_response_started = []

        def start_response_wrapper(*args, **kw):
            assert_(len(args) == 2 or len(args) == 3, (
                "Invalid number of arguments: %s" % (args,)))
            assert_(not kw, "No keyword arguments allowed")
            status = args[0]
            headers = args[1]
            if len(args) == 3:
                exc_info = args[2]
            else:
                exc_info = None

            check_status(status)
            check_headers(headers)
            check_content_type(status, headers)
            check_exc_info(exc_info)

            start_response_started.append(None)
            return WriteWrapper(start_response(*args))

        environ['wsgi.input'] = InputWrapper(environ['wsgi.input'])
        environ['wsgi.errors'] = ErrorWrapper(environ['wsgi.errors'])

        iterator = application(environ, start_response_wrapper)
        assert_(iterator is not None and iterator != False,
            "The application must return an iterator, if only an empty list")

        check_iterator(iterator)

        return IteratorWrapper(iterator, start_response_started)

    return lint_app

class InputWrapper:

    def __init__(self, wsgi_input):
        self.input = wsgi_input

    def read(self, *args):
        assert_(len(args) == 1)
        v = self.input.read(*args)
        assert_(type(v) is bytes)
        return v

    def readline(self, *args):
        assert_(len(args) <= 1)
        v = self.input.readline(*args)
        assert_(type(v) is bytes)
        return v

    def readlines(self, *args):
        assert_(len(args) <= 1)
        lines = self.input.readlines(*args)
        assert_(type(lines) is list)
        for line in lines:
            assert_(type(line) is bytes)
        return lines

    def __iter__(self):
        while 1:
            line = self.readline()
            if not line:
                return
            yield line

    def close(self):
        assert_(0, "input.close() must not be called")

class ErrorWrapper:

    def __init__(self, wsgi_errors):
        self.errors = wsgi_errors

    def write(self, s):
        assert_(type(s) is str)
        self.errors.write(s)

    def flush(self):
        self.errors.flush()

    def writelines(self, seq):
        for line in seq:
            self.write(line)

    def close(self):
        assert_(0, "errors.close() must not be called")

class WriteWrapper:

    def __init__(self, wsgi_writer):
        self.writer = wsgi_writer

    def __call__(self, s):
        assert_(type(s) is bytes)
        self.writer(s)

class PartialIteratorWrapper:

    def __init__(self, wsgi_iterator):
        self.iterator = wsgi_iterator

    def __iter__(self):
        # We want to make sure __iter__ is called
        return IteratorWrapper(self.iterator, None)

class IteratorWrapper:

    def __init__(self, wsgi_iterator, check_start_response):
        self.original_iterator = wsgi_iterator
        self.iterator = iter(wsgi_iterator)
        self.closed = False
        self.check_start_response = check_start_response

    def __iter__(self):
        return self

    def __next__(self):
        assert_(not self.closed,
            "Iterator read after closed")
        v = next(self.iterator)
        if type(v) is not bytes:
            assert_(False, "Iterator yielded non-bytestring (%r)" % (v,))
        if self.check_start_response is not None:
            assert_(self.check_start_response,
                "The application returns and we started iterating over its body, but start_response has not yet been called")
            self.check_start_response = None
        return v

    def close(self):
        self.closed = True
        if hasattr(self.original_iterator, 'close'):
            self.original_iterator.close()

    def __del__(self):
        if not self.closed:
            sys.stderr.write(
                "Iterator garbage collected without being closed")
        assert_(self.closed,
            "Iterator garbage collected without being closed")

def check_environ(environ):
    assert_(type(environ) is dict,
        "Environment is not of the right type: %r (environment: %r)"
        % (type(environ), environ))

    for key in ['REQUEST_METHOD', 'SERVER_NAME', 'SERVER_PORT',
                'wsgi.version', 'wsgi.input', 'wsgi.errors',
                'wsgi.multithread', 'wsgi.multiprocess',
                'wsgi.run_once']:
        assert_(key in environ,
            "Environment missing required key: %r" % (key,))

    for key in ['HTTP_CONTENT_TYPE', 'HTTP_CONTENT_LENGTH']:
        assert_(key not in environ,
            "Environment should not have the key: %s "
            "(use %s instead)" % (key, key[5:]))

    if 'QUERY_STRING' not in environ:
        warnings.warn(
            'QUERY_STRING is not in the WSGI environment; the cgi '
            'module will use sys.argv when this variable is missing, '
            'so application errors are more likely',
            WSGIWarning)

    for key in environ.keys():
        if '.' in key:
            # Extension, we don't care about its type
            continue
        assert_(type(environ[key]) is str,
            "Environmental variable %s is not a string: %r (value: %r)"
            % (key, type(environ[key]), environ[key]))

    assert_(type(environ['wsgi.version']) is tuple,
        "wsgi.version should be a tuple (%r)" % (environ['wsgi.version'],))
    assert_(environ['wsgi.url_scheme'] in ('http', 'https'),
        "wsgi.url_scheme unknown: %r" % environ['wsgi.url_scheme'])

    check_input(environ['wsgi.input'])
    check_errors(environ['wsgi.errors'])

    # @@: these need filling out:
    if environ['REQUEST_METHOD'] not in (
        'GET', 'HEAD', 'POST', 'OPTIONS', 'PATCH', 'PUT', 'DELETE', 'TRACE'):
        warnings.warn(
            "Unknown REQUEST_METHOD: %r" % environ['REQUEST_METHOD'],
            WSGIWarning)

    assert_(not environ.get('SCRIPT_NAME')
            or environ['SCRIPT_NAME'].startswith('/'),
        "SCRIPT_NAME doesn't start with /: %r" % environ['SCRIPT_NAME'])
    assert_(not environ.get('PATH_INFO')
            or environ['PATH_INFO'].startswith('/'),
        "PATH_INFO doesn't start with /: %r" % environ['PATH_INFO'])
    if environ.get('CONTENT_LENGTH'):
        assert_(int(environ['CONTENT_LENGTH']) >= 0,
            "Invalid CONTENT_LENGTH: %r" % environ['CONTENT_LENGTH'])

    if not environ.get('SCRIPT_NAME'):
        assert_('PATH_INFO' in environ,
            "One of SCRIPT_NAME or PATH_INFO are required (PATH_INFO "
            "should at least be '/' if SCRIPT_NAME is empty)")
    assert_(environ.get('SCRIPT_NAME') != '/',
        "SCRIPT_NAME cannot be '/'; it should instead be '', and "
        "PATH_INFO should be '/'")

def check_input(wsgi_input):
    for attr in ['read', 'readline', 'readlines', '__iter__']:
        assert_(hasattr(wsgi_input, attr),
            "wsgi.input (%r) doesn't have the attribute %s"
            % (wsgi_input, attr))

def check_errors(wsgi_errors):
    for attr in ['flush', 'write', 'writelines']:
        assert_(hasattr(wsgi_errors, attr),
            "wsgi.errors (%r) doesn't have the attribute %s"
            % (wsgi_errors, attr))

def check_status(status):
    status = check_string_type(status, "Status")
    # Implicitly check that we can turn it into an integer:
    status_code = status.split(None, 1)[0]
    assert_(len(status_code) == 3,
        "Status codes must be three characters: %r" % status_code)
    status_int = int(status_code)
    assert_(status_int >= 100, "Status code is invalid: %r" % status_int)
    if len(status) < 4 or status[3] != ' ':
        warnings.warn(
            "The status string (%r) should be a three-digit integer "
            "followed by a single space and a status explanation"
            % status, WSGIWarning)

def check_headers(headers):
    assert_(type(headers) is list,
        "Headers (%r) must be of type list: %r"
        % (headers, type(headers)))
    for item in headers:
        assert_(type(item) is tuple,
            "Individual headers (%r) must be of type tuple: %r"
            % (item, type(item)))
        assert_(len(item) == 2)
        name, value = item
        name = check_string_type(name, "Header name")
        value = check_string_type(value, "Header value")
        assert_(name.lower() != 'status',
            "The Status header cannot be used; it conflicts with CGI "
            "script, and HTTP status is not given through headers "
            "(value: %r)." % value)
        assert_('\n' not in name and ':' not in name,
            "Header names may not contain ':' or '\\n': %r" % name)
        assert_(header_re.search(name), "Bad header name: %r" % name)
        assert_(not name.endswith('-') and not name.endswith('_'),
            "Names may not end in '-' or '_': %r" % name)
        if bad_header_value_re.search(value):
            assert_(0, "Bad header value: %r (bad char: %r)"
            % (value, bad_header_value_re.search(value).group(0)))

def check_content_type(status, headers):
    status = check_string_type(status, "Status")
    code = int(status.split(None, 1)[0])
    # @@: need one more person to verify this interpretation of RFC 2616
    #     http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html
    NO_MESSAGE_BODY = (204, 304)
    for name, value in headers:
        name = check_string_type(name, "Header name")
        if name.lower() == 'content-type':
            if code not in NO_MESSAGE_BODY:
                return
            assert_(0, ("Content-Type header found in a %s response, "
                        "which must not return content.") % code)
    if code not in NO_MESSAGE_BODY:
        assert_(0, "No Content-Type header found in headers (%s)" % headers)

def check_exc_info(exc_info):
    assert_(exc_info is None or type(exc_info) is tuple,
        "exc_info (%r) is not a tuple: %r" % (exc_info, type(exc_info)))
    # More exc_info checks?

def check_iterator(iterator):
    # Technically a bytestring is legal, which is why it's a really bad
    # idea, because it may cause the response to be returned
    # character-by-character
    assert_(not isinstance(iterator, (str, bytes)),
        "You should not return a string as your application iterator, "
        "instead return a single-item list containing a bytestring.")
<|endoftext|>"""Manage HTTP Response Headers

Much of this module is red-handedly pilfered from email.message in the stdlib,
so portions are Copyright (C) 2001,2002 Python Software Foundation, and were
written by Barry Warsaw.
"""

# Regular expression that matches `special' characters in parameters, the
# existence of which force quoting of the parameter value.
import re
tspecials = re.compile(r'[ \(\)<>@,;:\\"/\[\]\?=]')
# Disallowed characters for headers and values.
# HTAB (\x09) is allowed in header values, but
# not in header names. (RFC 9110 Section 5.5)
_name_disallowed_re = re.compile(r'[\x00-\x1F\x7F]')
_value_disallowed_re = re.compile(r'[\x00-\x08\x0A-\x1F\x7F]')

def _formatparam(param, value=None, quote=1):
    """Convenience function to format and return a key=value pair.

    This will quote the value if needed or if quote is true.
    """
    if value is not None and len(value) > 0:
        if quote or tspecials.search(value):
            value = value.replace('\\', '\\\\').replace('"', r'\"')
            return '%s="%s"' % (param, value)
        else:
            return '%s=%s' % (param, value)
    else:
        return param


class Headers:
    """Manage a collection of HTTP response headers"""

    def __init__(self, headers=None):
        headers = headers if headers is not None else []
        if type(headers) is not list:
            raise TypeError("Headers must be a list of name/value tuples")
        self._headers = headers
        if __debug__:
            for k, v in headers:
                self._convert_string_type(k, name=True)
                self._convert_string_type(v, name=False)

    def _convert_string_type(self, value, *, name):
        """Convert/check value type."""
        if type(value) is str:
            regex = (_name_disallowed_re if name else _value_disallowed_re)
            if regex.search(value):
                raise ValueError("Control characters not allowed in headers")
            return value
        raise AssertionError("Header names/values must be"
            " of type str (got {0})".format(repr(value)))

    def __len__(self):
        """Return the total number of headers, including duplicates."""
        return len(self._headers)

    def __setitem__(self, name, val):
        """Set the value of a header."""
        del self[name]
        self._headers.append(
            (self._convert_string_type(name, name=True), self._convert_string_type(val, name=False)))

    def __delitem__(self,name):
        """Delete all occurrences of a header, if present.

        Does *not* raise an exception if the header is missing.
        """
        name = self._convert_string_type(name.lower(), name=True)
        self._headers[:] = [kv for kv in self._headers if kv[0].lower() != name]

    def __getitem__(self,name):
        """Get the first header value for 'name'

        Return None if the header is missing instead of raising an exception.

        Note that if the header appeared multiple times, the first exactly which
        occurrence gets returned is undefined.  Use getall() to get all
        the values matching a header field name.
        """
        return self.get(name)

    def __contains__(self, name):
        """Return true if the message contains the header."""
        return self.get(name) is not None


    def get_all(self, name):
        """Return a list of all the values for the named field.

        These will be sorted in the order they appeared in the original header
        list or were added to this instance, and may contain duplicates.  Any
        fields deleted and re-inserted are always appended to the header list.
        If no fields exist with the given name, returns an empty list.
        """
        name = self._convert_string_type(name.lower(), name=True)
        return [kv[1] for kv in self._headers if kv[0].lower()==name]


    def get(self,name,default=None):
        """Get the first header value for 'name', or return 'default'"""
        name = self._convert_string_type(name.lower(), name=True)
        for k,v in self._headers:
            if k.lower()==name:
                return v
        return default


    def keys(self):
        """Return a list of all the header field names.

        These will be sorted in the order they appeared in the original header
        list, or were added to this instance, and may contain duplicates.
        Any fields deleted and re-inserted are always appended to the header
        list.
        """
        return [k for k, v in self._headers]

    def values(self):
        """Return a list of all header values.

        These will be sorted in the order they appeared in the original header
        list, or were added to this instance, and may contain duplicates.
        Any fields deleted and re-inserted are always appended to the header
        list.
        """
        return [v for k, v in self._headers]

    def items(self):
        """Get all the header fields and values.

        These will be sorted in the order they were in the original header
        list, or were added to this instance, and may contain duplicates.
        Any fields deleted and re-inserted are always appended to the header
        list.
        """
        return self._headers[:]

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._headers)

    def __str__(self):
        """str() returns the formatted headers, complete with end line,
        suitable for direct HTTP transmission."""
        return '\r\n'.join(["%s: %s" % kv for kv in self._headers]+['',''])

    def __bytes__(self):
        return str(self).encode('iso-8859-1')

    def setdefault(self,name,value):
        """Return first matching header value for 'name', or 'value'

        If there is no header named 'name', add a new header with name 'name'
        and value 'value'."""
        result = self.get(name)
        if result is None:
            self._headers.append((self._convert_string_type(name, name=True),
                self._convert_string_type(value, name=False)))
            return value
        else:
            return result

    def add_header(self, _name, _value, **_params):
        """Extended header setting.

        _name is the header field to add.  keyword arguments can be used to set
        additional parameters for the header field, with underscores converted
        to dashes.  Normally the parameter will be added as key="value" unless
        value is None, in which case only the key will be added.

        Example:

        h.add_header('content-disposition', 'attachment', filename='bud.gif')

        Note that unlike the corresponding 'email.message' method, this does
        *not* handle '(charset, language, value)' tuples: all values must be
        strings or None.
        """
        parts = []
        if _value is not None:
            _value = self._convert_string_type(_value, name=False)
            parts.append(_value)
        for k, v in _params.items():
            k = self._convert_string_type(k, name=True)
            if v is None:
                parts.append(k.replace('_', '-'))
            else:
                v = self._convert_string_type(v, name=False)
                parts.append(_formatparam(k.replace('_', '-'), v))
        self._headers.append((self._convert_string_type(_name, name=True), "; ".join(parts)))
<|endoftext|>"""Base classes for server/gateway implementations"""

from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers

import sys, os, time

__all__ = [
    'BaseHandler', 'SimpleHandler', 'BaseCGIHandler', 'CGIHandler',
    'IISCGIHandler', 'read_environ'
]

# Weekday and month names for HTTP date/time formatting; always English!
_weekdayname = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_monthname = [None, # Dummy so we can use 1-based month numbers
              "Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def format_date_time(timestamp):
    year, month, day, hh, mm, ss, wd, y, z = time.gmtime(timestamp)
    return "%s, %02d %3s %4d %02d:%02d:%02d GMT" % (
        _weekdayname[wd], day, _monthname[month], year, hh, mm, ss
    )

_is_request = {
    'SCRIPT_NAME', 'PATH_INFO', 'QUERY_STRING', 'REQUEST_METHOD', 'AUTH_TYPE',
    'CONTENT_TYPE', 'CONTENT_LENGTH', 'HTTPS', 'REMOTE_USER', 'REMOTE_IDENT',
}.__contains__

def _needs_transcode(k):
    return _is_request(k) or k.startswith('HTTP_') or k.startswith('SSL_') \
        or (k.startswith('REDIRECT_') and _needs_transcode(k[9:]))

def read_environ():
    """Read environment, fixing HTTP variables"""
    enc = sys.getfilesystemencoding()
    esc = 'surrogateescape'
    try:
        ''.encode('utf-8', esc)
    except LookupError:
        esc = 'replace'
    environ = {}

    # Take the basic environment from native-unicode os.environ. Attempt to
    # fix up the variables that come from the HTTP request to compensate for
    # the bytes->unicode decoding step that will already have taken place.
    for k, v in os.environ.items():
        if _needs_transcode(k):

            # On win32, the os.environ is natively Unicode. Different servers
            # decode the request bytes using different encodings.
            if sys.platform == 'win32':
                software = os.environ.get('SERVER_SOFTWARE', '').lower()

                # On IIS, the HTTP request will be decoded as UTF-8 as long
                # as the input is a valid UTF-8 sequence. Otherwise it is
                # decoded using the system code page (mbcs), with no way to
                # detect this has happened. Because UTF-8 is the more likely
                # encoding, and mbcs is inherently unreliable (an mbcs string
                # that happens to be valid UTF-8 will not be decoded as mbcs)
                # always recreate the original bytes as UTF-8.
                if software.startswith('microsoft-iis/'):
                    v = v.encode('utf-8').decode('iso-8859-1')

                # Apache mod_cgi writes bytes-as-unicode (as if ISO-8859-1) direct
                # to the Unicode environ. No modification needed.
                elif software.startswith('apache/'):
                    pass

                # Python 3's http.server.CGIHTTPRequestHandler decodes
                # using the urllib.unquote default of UTF-8, amongst other
                # issues.
                elif (
                    software.startswith('simplehttp/')
                    and 'python/3' in software
                ):
                    v = v.encode('utf-8').decode('iso-8859-1')

                # For other servers, guess that they have written bytes to
                # the environ using stdio byte-oriented interfaces, ending up
                # with the system code page.
                else:
                    v = v.encode(enc, 'replace').decode('iso-8859-1')

            # Recover bytes from unicode environ, using surrogate escapes
            # where available (Python 3.1+).
            else:
                v = v.encode(enc, esc).decode('iso-8859-1')

        environ[k] = v
    return environ


class BaseHandler:
    """Manage the invocation of a WSGI application"""

    # Configuration parameters; can override per-subclass or per-instance
    wsgi_version = (1,0)
    wsgi_multithread = True
    wsgi_multiprocess = True
    wsgi_run_once = False

    origin_server = True    # We are transmitting direct to client
    http_version  = "1.0"   # Version that should be used for response
    server_software = None  # String name of server software, if any

    # os_environ is used to supply configuration from the OS environment:
    # by default it's a copy of 'os.environ' as of import time, but you can
    # override this in e.g. your __init__ method.
    os_environ= read_environ()

    # Collaborator classes
    wsgi_file_wrapper = FileWrapper     # set to None to disable
    headers_class = Headers             # must be a Headers-like class

    # Error handling (also per-subclass or per-instance)
    traceback_limit = None  # Print entire traceback to self.get_stderr()
    error_status = "500 Internal Server Error"
    error_headers = [('Content-Type','text/plain')]
    error_body = b"A server error occurred.  Please contact the administrator."

    # State variables (don't mess with these)
    status = result = None
    headers_sent = False
    headers = None
    bytes_sent = 0

    def run(self, application):
        """Invoke the application"""
        # Note to self: don't move the close()!  Asynchronous servers shouldn't
        # call close() from finish_response(), so if you close() anywhere but
        # the double-error branch here, you'll break asynchronous servers by
        # prematurely closing.  Async servers must return from 'run()' without
        # closing if there might still be output to iterate over.
        try:
            self.setup_environ()
            self.result = application(self.environ, self.start_response)
            self.finish_response()
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            # We expect the client to close the connection abruptly from time
            # to time.
            return
        except:
            try:
                self.handle_error()
            except:
                # If we get an error handling an error, just give up already!
                self.close()
                raise   # ...and let the actual server figure it out.


    def setup_environ(self):
        """Set up the environment for one request"""

        env = self.environ = self.os_environ.copy()
        self.add_cgi_vars()

        env['wsgi.input']        = self.get_stdin()
        env['wsgi.errors']       = self.get_stderr()
        env['wsgi.version']      = self.wsgi_version
        env['wsgi.run_once']     = self.wsgi_run_once
        env['wsgi.url_scheme']   = self.get_scheme()
        env['wsgi.multithread']  = self.wsgi_multithread
        env['wsgi.multiprocess'] = self.wsgi_multiprocess

        if self.wsgi_file_wrapper is not None:
            env['wsgi.file_wrapper'] = self.wsgi_file_wrapper

        if self.origin_server and self.server_software:
            env.setdefault('SERVER_SOFTWARE',self.server_software)


    def finish_response(self):
        """Send any iterable data, then close self and the iterable

        Subclasses intended for use in asynchronous servers will
        want to redefine this method, such that it sets up callbacks
        in the event loop to iterate over the data, and to call
        'self.close()' once the response is finished.
        """
        try:
            if not self.result_is_file() or not self.sendfile():
                for data in self.result:
                    self.write(data)
                self.finish_content()
        except:
            # Call close() on the iterable returned by the WSGI application
            # in case of an exception.
            if hasattr(self.result, 'close'):
                self.result.close()
            raise
        else:
            # We only call close() when no exception is raised, because it
            # will set status, result, headers, and environ fields to None.
            # See bpo-29183 for more details.
            self.close()


    def get_scheme(self):
        """Return the URL scheme being used"""
        return guess_scheme(self.environ)


    def set_content_length(self):
        """Compute Content-Length or switch to chunked encoding if possible"""
        try:
            blocks = len(self.result)
        except (TypeError,AttributeError,NotImplementedError):
            pass
        else:
            if blocks==1:
                self.headers['Content-Length'] = str(self.bytes_sent)
                return
        # XXX Try for chunked encoding if origin server and client is 1.1


    def cleanup_headers(self):
        """Make any necessary header changes or defaults

        Subclasses can extend this to add other defaults.
        """
        if 'Content-Length' not in self.headers:
            self.set_content_length()

    def start_response(self, status, headers,exc_info=None):
        """'start_response()' callable as specified by PEP 3333"""

        if exc_info:
            try:
                if self.headers_sent:
                    # Re-raise original exception if headers sent
                    raise exc_info[0](exc_info[1]).with_traceback(exc_info[2])
            finally:
                exc_info = None        # avoid dangling circular ref
        elif self.headers is not None:
            raise AssertionError("Headers already set!")

        self.status = status
        self.headers = self.headers_class(headers)
        status = self._convert_string_type(status, "Status")
        assert len(status)>=4,"Status must be at least 4 characters"
        assert status[:3].isdigit(), "Status message must begin w/3-digit code"
        assert status[3]==" ", "Status message must have a space after code"

        if __debug__:
            for name, val in headers:
                name = self._convert_string_type(name, "Header name")
                val = self._convert_string_type(val, "Header value")
                assert not is_hop_by_hop(name),\
                       f"Hop-by-hop header, '{name}: {val}', not allowed"

        return self.write

    def _convert_string_type(self, value, title):
        """Convert/check value type."""
        if type(value) is str:
            return value
        raise AssertionError(
            "{0} must be of type str (got {1})".format(title, repr(value))
        )

    def send_preamble(self):
        """Transmit version/status/date/server, via self._write()"""
        if self.origin_server:
            if self.client_is_modern():
                self._write(('HTTP/%s %s\r\n' % (self.http_version,self.status)).encode('iso-8859-1'))
                if 'Date' not in self.headers:
                    self._write(
                        ('Date: %s\r\n' % format_date_time(time.time())).encode('iso-8859-1')
                    )
                if self.server_software and 'Server' not in self.headers:
                    self._write(('Server: %s\r\n' % self.server_software).encode('iso-8859-1'))
        else:
            self._write(('Status: %s\r\n' % self.status).encode('iso-8859-1'))

    def write(self, data):
        """'write()' callable as specified by PEP 3333"""

        assert type(data) is bytes, \
            "write() argument must be a bytes instance"

        if not self.status:
            raise AssertionError("write() before start_response()")

        elif not self.headers_sent:
            # Before the first output, send the stored headers
            self.bytes_sent = len(data)    # make sure we know content-length
            self.send_headers()
        else:
            self.bytes_sent += len(data)

        # XXX check Content-Length and truncate if too many bytes written?
        self._write(data)
        self._flush()


    def sendfile(self):
        """Platform-specific file transmission

        Override this method in subclasses to support platform-specific
        file transmission.  It is only called if the application's
        return iterable ('self.result') is an instance of
        'self.wsgi_file_wrapper'.

        This method should return a true value if it was able to actually
        transmit the wrapped file-like object using a platform-specific
        approach.  It should return a false value if normal iteration
        should be used instead.  An exception can be raised to indicate
        that transmission was attempted, but failed.

        NOTE: this method should call 'self.send_headers()' if
        'self.headers_sent' is false and it is going to attempt direct
        transmission of the file.
        """
        return False   # No platform-specific transmission by default


    def finish_content(self):
        """Ensure headers and content have both been sent"""
        if not self.headers_sent:
            # Only zero Content-Length if not set by the application (so
            # that HEAD requests can be satisfied properly, see #3839)
            self.headers.setdefault('Content-Length', "0")
            self.send_headers()
        else:
            pass # XXX check if content-length was too short?

    def close(self):
        """Close the iterable (if needed) and reset all instance vars

        Subclasses may want to also drop the client connection.
        """
        try:
            if hasattr(self.result,'close'):
