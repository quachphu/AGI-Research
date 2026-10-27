        strict_parsing: flag indicating what to do with parsing errors.
            If false (the default), errors are silently ignored.
            If true, errors raise a ValueError exception.

        separator: str. The symbol to use for separating the query arguments.
            Defaults to &.
    """
    if fp is None:
        fp = sys.stdin

    # field keys and values (except for files) are returned as strings
    # an encoding is required to decode the bytes read from self.fp
    if hasattr(fp,'encoding'):
        encoding = fp.encoding
    else:
        encoding = 'latin-1'

    # fp.read() must return bytes
    if isinstance(fp, TextIOWrapper):
        fp = fp.buffer

    if not 'REQUEST_METHOD' in environ:
        environ['REQUEST_METHOD'] = 'GET'       # For testing stand-alone
    if environ['REQUEST_METHOD'] == 'POST':
        ctype, pdict = parse_header(environ['CONTENT_TYPE'])
        if ctype == 'multipart/form-data':
            return parse_multipart(fp, pdict, separator=separator)
        elif ctype == 'application/x-www-form-urlencoded':
            clength = int(environ['CONTENT_LENGTH'])
            if maxlen and clength > maxlen:
                raise ValueError('Maximum content length exceeded')
            qs = fp.read(clength).decode(encoding)
        else:
            qs = ''                     # Unknown content-type
        if 'QUERY_STRING' in environ:
            if qs: qs = qs + '&'
            qs = qs + environ['QUERY_STRING']
        elif sys.argv[1:]:
            if qs: qs = qs + '&'
            qs = qs + sys.argv[1]
        environ['QUERY_STRING'] = qs    # XXX Shouldn't, really
    elif 'QUERY_STRING' in environ:
        qs = environ['QUERY_STRING']
    else:
        if sys.argv[1:]:
            qs = sys.argv[1]
        else:
            qs = ""
        environ['QUERY_STRING'] = qs    # XXX Shouldn't, really
    return urllib.parse.parse_qs(qs, keep_blank_values, strict_parsing,
                                 encoding=encoding, separator=separator)


def parse_multipart(fp, pdict, encoding="utf-8", errors="replace", separator='&'):
    """Parse multipart input.

    Arguments:
    fp   : input file
    pdict: dictionary containing other parameters of content-type header
    encoding, errors: request encoding and error handler, passed to
        FieldStorage

    Returns a dictionary just like parse_qs(): keys are the field names, each
    value is a list of values for that field. For non-file fields, the value
    is a list of strings.
    """
    # RFC 2046, Section 5.1 : The "multipart" boundary delimiters are always
    # represented as 7bit US-ASCII.
    boundary = pdict['boundary'].decode('ascii')
    ctype = "multipart/form-data; boundary={}".format(boundary)
    headers = Message()
    headers.set_type(ctype)
    try:
        headers['Content-Length'] = pdict['CONTENT-LENGTH']
    except KeyError:
        pass
    fs = FieldStorage(fp, headers=headers, encoding=encoding, errors=errors,
        environ={'REQUEST_METHOD': 'POST'}, separator=separator)
    return {k: fs.getlist(k) for k in fs}

def _parseparam(s):
    while s[:1] == ';':
        s = s[1:]
        end = s.find(';')
        while end > 0 and (s.count('"', 0, end) - s.count('\\"', 0, end)) % 2:
            end = s.find(';', end + 1)
        if end < 0:
            end = len(s)
        f = s[:end]
        yield f.strip()
        s = s[end:]

def parse_header(line):
    """Parse a Content-type like header.

    Return the main content-type and a dictionary of options.

    """
    parts = _parseparam(';' + line)
    key = parts.__next__()
    pdict = {}
    for p in parts:
        i = p.find('=')
        if i >= 0:
            name = p[:i].strip().lower()
            value = p[i+1:].strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
                value = value.replace('\\\\', '\\').replace('\\"', '"')
            pdict[name] = value
    return key, pdict


# Classes for field storage
# =========================

class MiniFieldStorage:

    """Like FieldStorage, for use when no file uploads are possible."""

    # Dummy attributes
    filename = None
    list = None
    type = None
    file = None
    type_options = {}
    disposition = None
    disposition_options = {}
    headers = {}

    def __init__(self, name, value):
        """Constructor from field name and value."""
        self.name = name
        self.value = value
        # self.file = StringIO(value)

    def __repr__(self):
        """Return printable representation."""
        return "MiniFieldStorage(%r, %r)" % (self.name, self.value)


class FieldStorage:

    """Store a sequence of fields, reading multipart/form-data.

    This class provides naming, typing, files stored on disk, and
    more.  At the top level, it is accessible like a dictionary, whose
    keys are the field names.  (Note: None can occur as a field name.)
    The items are either a Python list (if there's multiple values) or
    another FieldStorage or MiniFieldStorage object.  If it's a single
    object, it has the following attributes:

    name: the field name, if specified; otherwise None

    filename: the filename, if specified; otherwise None; this is the
        client side filename, *not* the file name on which it is
        stored (that's a temporary file you don't deal with)

    value: the value as a *string*; for file uploads, this
        transparently reads the file every time you request the value
        and returns *bytes*

    file: the file(-like) object from which you can read the data *as
        bytes* ; None if the data is stored a simple string

    type: the content-type, or None if not specified

    type_options: dictionary of options specified on the content-type
        line

    disposition: content-disposition, or None if not specified

    disposition_options: dictionary of corresponding options

    headers: a dictionary(-like) object (sometimes email.message.Message or a
        subclass thereof) containing *all* headers

    The class is subclassable, mostly for the purpose of overriding
    the make_file() method, which is called internally to come up with
    a file open for reading and writing.  This makes it possible to
    override the default choice of storing all files in a temporary
    directory and unlinking them as soon as they have been opened.

    """
    def __init__(self, fp=None, headers=None, outerboundary=b'',
                 environ=os.environ, keep_blank_values=0, strict_parsing=0,
                 limit=None, encoding='utf-8', errors='replace',
                 max_num_fields=None, separator='&'):
        """Constructor.  Read multipart/* until last part.

        Arguments, all optional:

        fp              : file pointer; default: sys.stdin.buffer
            (not used when the request method is GET)
            Can be :
            1. a TextIOWrapper object
            2. an object whose read() and readline() methods return bytes

        headers         : header dictionary-like object; default:
            taken from environ as per CGI spec

        outerboundary   : terminating multipart boundary
            (for internal use only)

        environ         : environment dictionary; default: os.environ

        keep_blank_values: flag indicating whether blank values in
            percent-encoded forms should be treated as blank strings.
            A true value indicates that blanks should be retained as
            blank strings.  The default false value indicates that
            blank values are to be ignored and treated as if they were
            not included.

        strict_parsing: flag indicating what to do with parsing errors.
            If false (the default), errors are silently ignored.
            If true, errors raise a ValueError exception.

        limit : used internally to read parts of multipart/form-data forms,
            to exit from the reading loop when reached. It is the difference
            between the form content-length and the number of bytes already
            read

        encoding, errors : the encoding and error handler used to decode the
            binary stream to strings. Must be the same as the charset defined
            for the page sending the form (content-type : meta http-equiv or
            header)

        max_num_fields: int. If set, then __init__ throws a ValueError
            if there are more than n fields read by parse_qsl().

        """
        method = 'GET'
        self.keep_blank_values = keep_blank_values
        self.strict_parsing = strict_parsing
        self.max_num_fields = max_num_fields
        self.separator = separator
        if 'REQUEST_METHOD' in environ:
            method = environ['REQUEST_METHOD'].upper()
        self.qs_on_post = None
        if method == 'GET' or method == 'HEAD':
            if 'QUERY_STRING' in environ:
                qs = environ['QUERY_STRING']
            elif sys.argv[1:]:
                qs = sys.argv[1]
            else:
                qs = ""
            qs = qs.encode(locale.getpreferredencoding(), 'surrogateescape')
            fp = BytesIO(qs)
            if headers is None:
                headers = {'content-type':
                           "application/x-www-form-urlencoded"}
        if headers is None:
            headers = {}
            if method == 'POST':
                # Set default content-type for POST to what's traditional
                headers['content-type'] = "application/x-www-form-urlencoded"
            if 'CONTENT_TYPE' in environ:
                headers['content-type'] = environ['CONTENT_TYPE']
            if 'QUERY_STRING' in environ:
                self.qs_on_post = environ['QUERY_STRING']
            if 'CONTENT_LENGTH' in environ:
                headers['content-length'] = environ['CONTENT_LENGTH']
        else:
            if not (isinstance(headers, (Mapping, Message))):
                raise TypeError("headers must be mapping or an instance of "
                                "email.message.Message")
        self.headers = headers
        if fp is None:
            self.fp = sys.stdin.buffer
        # self.fp.read() must return bytes
        elif isinstance(fp, TextIOWrapper):
            self.fp = fp.buffer
        else:
            if not (hasattr(fp, 'read') and hasattr(fp, 'readline')):
                raise TypeError("fp must be file pointer")
            self.fp = fp

        self.encoding = encoding
        self.errors = errors

        if not isinstance(outerboundary, bytes):
            raise TypeError('outerboundary must be bytes, not %s'
                            % type(outerboundary).__name__)
        self.outerboundary = outerboundary

        self.bytes_read = 0
        self.limit = limit

        # Process content-disposition header
        cdisp, pdict = "", {}
        if 'content-disposition' in self.headers:
            cdisp, pdict = parse_header(self.headers['content-disposition'])
        self.disposition = cdisp
        self.disposition_options = pdict
        self.name = None
        if 'name' in pdict:
            self.name = pdict['name']
        self.filename = None
        if 'filename' in pdict:
            self.filename = pdict['filename']
        self._binary_file = self.filename is not None

        # Process content-type header
        #
        # Honor any existing content-type header.  But if there is no
        # content-type header, use some sensible defaults.  Assume
        # outerboundary is "" at the outer level, but something non-false
        # inside a multi-part.  The default for an inner part is text/plain,
        # but for an outer part it should be urlencoded.  This should catch
        # bogus clients which erroneously forget to include a content-type
        # header.
        #
        # See below for what we do if there does exist a content-type header,
        # but it happens to be something we don't understand.
        if 'content-type' in self.headers:
            ctype, pdict = parse_header(self.headers['content-type'])
        elif self.outerboundary or method != 'POST':
            ctype, pdict = "text/plain", {}
        else:
            ctype, pdict = 'application/x-www-form-urlencoded', {}
        self.type = ctype
        self.type_options = pdict
        if 'boundary' in pdict:
            self.innerboundary = pdict['boundary'].encode(self.encoding,
                                                          self.errors)
        else:
            self.innerboundary = b""

        clen = -1
        if 'content-length' in self.headers:
            try:
                clen = int(self.headers['content-length'])
            except ValueError:
                pass
            if maxlen and clen > maxlen:
                raise ValueError('Maximum content length exceeded')
        self.length = clen
        if self.limit is None and clen >= 0:
            self.limit = clen

        self.list = self.file = None
        self.done = 0
        if ctype == 'application/x-www-form-urlencoded':
            self.read_urlencoded()
        elif ctype[:10] == 'multipart/':
            self.read_multi(environ, keep_blank_values, strict_parsing)
        else:
            self.read_single()

    def __del__(self):
        try:
            self.file.close()
        except AttributeError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def __repr__(self):
        """Return a printable representation."""
        return "FieldStorage(%r, %r, %r)" % (
                self.name, self.filename, self.value)

    def __iter__(self):
        return iter(self.keys())

    def __getattr__(self, name):
        if name != 'value':
            raise AttributeError(name)
        if self.file:
            self.file.seek(0)
            value = self.file.read()
            self.file.seek(0)
        elif self.list is not None:
            value = self.list
        else:
            value = None
        return value

    def __getitem__(self, key):
        """Dictionary style indexing."""
        if self.list is None:
            raise TypeError("not indexable")
        found = []
        for item in self.list:
            if item.name == key: found.append(item)
        if not found:
            raise KeyError(key)
        if len(found) == 1:
            return found[0]
        else:
            return found

    def getvalue(self, key, default=None):
        """Dictionary style get() method, including 'value' lookup."""
        if key in self:
            value = self[key]
            if isinstance(value, list):
                return [x.value for x in value]
            else:
                return value.value
        else:
            return default

    def getfirst(self, key, default=None):
        """ Return the first value received."""
        if key in self:
            value = self[key]
            if isinstance(value, list):
                return value[0].value
            else:
                return value.value
        else:
            return default

    def getlist(self, key):
        """ Return list of received values."""
        if key in self:
            value = self[key]
            if isinstance(value, list):
                return [x.value for x in value]
            else:
                return [value.value]
        else:
            return []

    def keys(self):
        """Dictionary style keys() method."""
        if self.list is None:
            raise TypeError("not indexable")
        return list(set(item.name for item in self.list))

    def __contains__(self, key):
        """Dictionary style __contains__ method."""
        if self.list is None:
            raise TypeError("not indexable")
        return any(item.name == key for item in self.list)

    def __len__(self):
        """Dictionary style len(x) support."""
        return len(self.keys())

    def __bool__(self):
        if self.list is None:
            raise TypeError("Cannot be converted to bool.")
        return bool(self.list)

    def read_urlencoded(self):
        """Internal: read data in query string format."""
        qs = self.fp.read(self.length)
        if not isinstance(qs, bytes):
            raise ValueError("%s should return bytes, got %s" \
                             % (self.fp, type(qs).__name__))
        qs = qs.decode(self.encoding, self.errors)
        if self.qs_on_post:
            qs += '&' + self.qs_on_post
        query = urllib.parse.parse_qsl(
            qs, self.keep_blank_values, self.strict_parsing,
            encoding=self.encoding, errors=self.errors,
            max_num_fields=self.max_num_fields, separator=self.separator)
        self.list = [MiniFieldStorage(key, value) for key, value in query]
        self.skip_lines()

    FieldStorageClass = None

    def read_multi(self, environ, keep_blank_values, strict_parsing):
        """Internal: read a part that is itself multipart."""
        ib = self.innerboundary
        if not valid_boundary(ib):
            raise ValueError('Invalid boundary in multipart form: %r' % (ib,))
        self.list = []
        if self.qs_on_post:
            query = urllib.parse.parse_qsl(
                self.qs_on_post, self.keep_blank_values, self.strict_parsing,
                encoding=self.encoding, errors=self.errors,
                max_num_fields=self.max_num_fields, separator=self.separator)
            self.list.extend(MiniFieldStorage(key, value) for key, value in query)

        klass = self.FieldStorageClass or self.__class__
        first_line = self.fp.readline() # bytes
        if not isinstance(first_line, bytes):
            raise ValueError("%s should return bytes, got %s" \
                             % (self.fp, type(first_line).__name__))
        self.bytes_read += len(first_line)

        # Ensure that we consume the file until we've hit our inner boundary
        while (first_line.strip() != (b"--" + self.innerboundary) and
                first_line):
            first_line = self.fp.readline()
            self.bytes_read += len(first_line)

        # Propagate max_num_fields into the sub class appropriately
        max_num_fields = self.max_num_fields
        if max_num_fields is not None:
            max_num_fields -= len(self.list)

        while True:
            parser = FeedParser()
            hdr_text = b""
            while True:
                data = self.fp.readline()
                hdr_text += data
                if not data.strip():
                    break
            if not hdr_text:
                break
            # parser takes strings, not bytes
            self.bytes_read += len(hdr_text)
            parser.feed(hdr_text.decode(self.encoding, self.errors))
            headers = parser.close()

            # Some clients add Content-Length for part headers, ignore them
            if 'content-length' in headers:
                del headers['content-length']

            limit = None if self.limit is None \
                else self.limit - self.bytes_read
            part = klass(self.fp, headers, ib, environ, keep_blank_values,
                         strict_parsing, limit,
                         self.encoding, self.errors, max_num_fields, self.separator)

            if max_num_fields is not None:
                max_num_fields -= 1
                if part.list:
                    max_num_fields -= len(part.list)
                if max_num_fields < 0:
                    raise ValueError('Max number of fields exceeded')

            self.bytes_read += part.bytes_read
            self.list.append(part)
            if part.done or self.bytes_read >= self.length > 0:
                break
        self.skip_lines()

    def read_single(self):
        """Internal: read an atomic part."""
        if self.length >= 0:
            self.read_binary()
            self.skip_lines()
        else:
            self.read_lines()
        self.file.seek(0)

    bufsize = 8*1024            # I/O buffering size for copy to file

    def read_binary(self):
        """Internal: read binary data."""
        self.file = self.make_file()
        todo = self.length
        if todo >= 0:
            while todo > 0:
                data = self.fp.read(min(todo, self.bufsize)) # bytes
                if not isinstance(data, bytes):
                    raise ValueError("%s should return bytes, got %s"
                                     % (self.fp, type(data).__name__))
                self.bytes_read += len(data)
                if not data:
                    self.done = -1
                    break
                self.file.write(data)
                todo = todo - len(data)

    def read_lines(self):
        """Internal: read lines until EOF or outerboundary."""
        if self._binary_file:
            self.file = self.__file = BytesIO() # store data as bytes for files
        else:
            self.file = self.__file = StringIO() # as strings for other fields
        if self.outerboundary:
            self.read_lines_to_outerboundary()
        else:
            self.read_lines_to_eof()

    def __write(self, line):
        """line is always bytes, not string"""
        if self.__file is not None:
            if self.__file.tell() + len(line) > 1000:
                self.file = self.make_file()
                data = self.__file.getvalue()
                self.file.write(data)
                self.__file = None
        if self._binary_file:
            # keep bytes
            self.file.write(line)
        else:
            # decode to string
            self.file.write(line.decode(self.encoding, self.errors))

    def read_lines_to_eof(self):
        """Internal: read lines until EOF."""
        while 1:
            line = self.fp.readline(1<<16) # bytes
            self.bytes_read += len(line)
            if not line:
                self.done = -1
                break
            self.__write(line)

    def read_lines_to_outerboundary(self):
        """Internal: read lines until outerboundary.
        Data is read as bytes: boundaries and line ends must be converted
        to bytes for comparisons.
        """
        next_boundary = b"--" + self.outerboundary
        last_boundary = next_boundary + b"--"
        delim = b""
        last_line_lfend = True
        _read = 0
        while 1:

            if self.limit is not None and 0 <= self.limit <= _read:
                break
            line = self.fp.readline(1<<16) # bytes
            self.bytes_read += len(line)
            _read += len(line)
            if not line:
                self.done = -1
                break
            if delim == b"\r":
                line = delim + line
                delim = b""
            if line.startswith(b"--") and last_line_lfend:
                strippedline = line.rstrip()
                if strippedline == next_boundary:
                    break
                if strippedline == last_boundary:
                    self.done = 1
                    break
            odelim = delim
            if line.endswith(b"\r\n"):
                delim = b"\r\n"
                line = line[:-2]
                last_line_lfend = True
            elif line.endswith(b"\n"):
                delim = b"\n"
                line = line[:-1]
                last_line_lfend = True
            elif line.endswith(b"\r"):
                # We may interrupt \r\n sequences if they span the 2**16
                # byte boundary
                delim = b"\r"
                line = line[:-1]
                last_line_lfend = False
            else:
                delim = b""
                last_line_lfend = False
            self.__write(odelim + line)

    def skip_lines(self):
        """Internal: skip lines until outer boundary if defined."""
        if not self.outerboundary or self.done:
            return
        next_boundary = b"--" + self.outerboundary
        last_boundary = next_boundary + b"--"
        last_line_lfend = True
        while True:
            line = self.fp.readline(1<<16)
            self.bytes_read += len(line)
            if not line:
                self.done = -1
                break
            if line.endswith(b"--") and last_line_lfend:
                strippedline = line.strip()
                if strippedline == next_boundary:
                    break
                if strippedline == last_boundary:
                    self.done = 1
                    break
            last_line_lfend = line.endswith(b'\n')

    def make_file(self):
        """Overridable: return a readable & writable file.

        The file will be used as follows:
        - data is written to it
        - seek(0)
        - data is read from it

        The file is opened in binary mode for files, in text mode
        for other fields

        This version opens a temporary file for reading and writing,
        and immediately deletes (unlinks) it.  The trick (on Unix!) is
        that the file can still be used, but it can't be opened by
        another process, and it will automatically be deleted when it
        is closed or when the current process terminates.

        If you want a more permanent file, you derive a class which
        overrides this method.  If you want a visible temporary file
        that is nevertheless automatically deleted when the script
        terminates, try defining a __del__ method in a derived class
        which unlinks the temporary files you have created.

        """
        if self._binary_file:
            return tempfile.TemporaryFile("wb+")
        else:
            return tempfile.TemporaryFile("w+",
                encoding=self.encoding, newline = '\n')


# Test/debug code
# ===============

def test(environ=os.environ):
    """Robust test CGI script, usable as main program.

    Write minimal HTTP headers and dump all information provided to
    the script in HTML form.

    """
    print("Content-type: text/html")
    print()
    sys.stderr = sys.stdout
    try:
        form = FieldStorage()   # Replace with other classes to test those
        print_directory()
        print_arguments()
        print_form(form)
        print_environ(environ)
        print_environ_usage()
        def f():
            exec("testing print_exception() -- <I>italics?</I>")
        def g(f=f):
            f()
        print("<H3>What follows is a test, not an actual exception:</H3>")
        g()
    except:
        print_exception()

    print("<H1>Second try with a small maxlen...</H1>")

    global maxlen
    maxlen = 50
    try:
        form = FieldStorage()   # Replace with other classes to test those
        print_directory()
        print_arguments()
        print_form(form)
        print_environ(environ)
    except:
        print_exception()

def print_exception(type=None, value=None, tb=None, limit=None):
    if type is None:
        type, value, tb = sys.exc_info()
    import traceback
    print()
    print("<H3>Traceback (most recent call last):</H3>")
    list = traceback.format_tb(tb, limit) + \
           traceback.format_exception_only(type, value)
    print("<PRE>%s<B>%s</B></PRE>" % (
        html.escape("".join(list[:-1])),
        html.escape(list[-1]),
        ))
    del tb

def print_environ(environ=os.environ):
    """Dump the shell environment as HTML."""
    keys = sorted(environ.keys())
    print()
    print("<H3>Shell Environment:</H3>")
    print("<DL>")
    for key in keys:
        print("<DT>", html.escape(key), "<DD>", html.escape(environ[key]))
    print("</DL>")
    print()

def print_form(form):
    """Dump the contents of a form as HTML."""
    keys = sorted(form.keys())
    print()
    print("<H3>Form Contents:</H3>")
    if not keys:
        print("<P>No form fields.")
    print("<DL>")
    for key in keys:
        print("<DT>" + html.escape(key) + ":", end=' ')
        value = form[key]
        print("<i>" + html.escape(repr(type(value))) + "</i>")
        print("<DD>" + html.escape(repr(value)))
    print("</DL>")
    print()

def print_directory():
    """Dump the current directory as HTML."""
    print()
    print("<H3>Current Working Directory:</H3>")
    try:
        pwd = os.getcwd()
    except OSError as msg:
        print("OSError:", html.escape(str(msg)))
    else:
        print(html.escape(pwd))
    print()

def print_arguments():
    print()
    print("<H3>Command Line Arguments:</H3>")
    print()
    print(sys.argv)
    print()

def print_environ_usage():
    """Dump a list of environment variables used by CGI as HTML."""
    print("""
<H3>These environment variables could have been set:</H3>
<UL>
<LI>AUTH_TYPE
<LI>CONTENT_LENGTH
<LI>CONTENT_TYPE
<LI>DATE_GMT
<LI>DATE_LOCAL
<LI>DOCUMENT_NAME
<LI>DOCUMENT_ROOT
<LI>DOCUMENT_URI
<LI>GATEWAY_INTERFACE
<LI>LAST_MODIFIED
<LI>PATH
<LI>PATH_INFO
<LI>PATH_TRANSLATED
<LI>QUERY_STRING
<LI>REMOTE_ADDR
<LI>REMOTE_HOST
<LI>REMOTE_IDENT
<LI>REMOTE_USER
<LI>REQUEST_METHOD
<LI>SCRIPT_NAME
<LI>SERVER_NAME
<LI>SERVER_PORT
<LI>SERVER_PROTOCOL
<LI>SERVER_ROOT
<LI>SERVER_SOFTWARE
</UL>
In addition, HTTP headers sent by the server may be passed in the
environment as well.  Here are some common variable names:
<UL>
<LI>HTTP_ACCEPT
<LI>HTTP_CONNECTION
<LI>HTTP_HOST
<LI>HTTP_PRAGMA
<LI>HTTP_REFERER
<LI>HTTP_USER_AGENT
</UL>
""")


# Utilities
# =========

def valid_boundary(s):
    import re
    if isinstance(s, bytes):
        _vb_pattern = b"^[ -~]{0,200}[!-~]$"
    else:
        _vb_pattern = "^[ -~]{0,200}[!-~]$"
    return re.match(_vb_pattern, s)

# Invoke mainline
# ===============

# Call test() when this file is run as a script (not imported as a module)
if __name__ == '__main__':
    test()
<|endoftext|>"""zipimport provides support for importing Python modules from Zip archives.

This module exports three objects:
- zipimporter: a class; its constructor takes a path to a Zip archive.
- ZipImportError: exception raised by zipimporter objects. It's a
  subclass of ImportError, so it can be caught as ImportError, too.
- _zip_directory_cache: a dict, mapping archive paths to zip directory
  info dicts, as used in zipimporter._files.

It is usually not needed to use the zipimport module explicitly; it is
used by the builtin import mechanism for sys.path items that are paths
to Zip archives.
"""

#from importlib import _bootstrap_external
#from importlib import _bootstrap  # for _verbose_message
import _frozen_importlib_external as _bootstrap_external
from _frozen_importlib_external import _unpack_uint16, _unpack_uint32
import _frozen_importlib as _bootstrap  # for _verbose_message
import _imp  # for check_hash_based_pycs
import _io  # for open
import marshal  # for loads
import sys  # for modules
import time  # for mktime
import _warnings  # For warn()

__all__ = ['ZipImportError', 'zipimporter']


path_sep = _bootstrap_external.path_sep
alt_path_sep = _bootstrap_external.path_separators[1:]


class ZipImportError(ImportError):
    pass

# _read_directory() cache
_zip_directory_cache = {}

_module_type = type(sys)

END_CENTRAL_DIR_SIZE = 22
STRING_END_ARCHIVE = b'PK\x05\x06'
MAX_COMMENT_LEN = (1 << 16) - 1

class zipimporter(_bootstrap_external._LoaderBasics):
    """zipimporter(archivepath) -> zipimporter object

    Create a new zipimporter instance. 'archivepath' must be a path to
    a zipfile, or to a specific path inside a zipfile. For example, it can be
    '/tmp/myimport.zip', or '/tmp/myimport.zip/mydirectory', if mydirectory is a
    valid directory inside the archive.

    'ZipImportError is raised if 'archivepath' doesn't point to a valid Zip
    archive.

    The 'archive' attribute of zipimporter objects contains the name of the
    zipfile targeted.
    """

    # Split the "subdirectory" from the Zip archive path, lookup a matching
    # entry in sys.path_importer_cache, fetch the file directory from there
    # if found, or else read it from the archive.
    def __init__(self, path):
        if not isinstance(path, str):
            import os
            path = os.fsdecode(path)
        if not path:
            raise ZipImportError('archive path is empty', path=path)
        if alt_path_sep:
            path = path.replace(alt_path_sep, path_sep)

        prefix = []
        while True:
            try:
                st = _bootstrap_external._path_stat(path)
            except (OSError, ValueError):
                # On Windows a ValueError is raised for too long paths.
                # Back up one path element.
                dirname, basename = _bootstrap_external._path_split(path)
                if dirname == path:
                    raise ZipImportError('not a Zip file', path=path)
                path = dirname
                prefix.append(basename)
            else:
                # it exists
                if (st.st_mode & 0o170000) != 0o100000:  # stat.S_ISREG
                    # it's a not file
                    raise ZipImportError('not a Zip file', path=path)
                break

        try:
            files = _zip_directory_cache[path]
        except KeyError:
            files = _read_directory(path)
            _zip_directory_cache[path] = files
        self._files = files
        self.archive = path
        # a prefix directory following the ZIP file path.
        self.prefix = _bootstrap_external._path_join(*prefix[::-1])
        if self.prefix:
            self.prefix += path_sep


    # Check whether we can satisfy the import of the module named by
    # 'fullname', or whether it could be a portion of a namespace
    # package. Return self if we can load it, a string containing the
    # full path if it's a possible namespace portion, None if we
    # can't load it.
    def find_loader(self, fullname, path=None):
        """find_loader(fullname, path=None) -> self, str or None.

        Search for a module specified by 'fullname'. 'fullname' must be the
        fully qualified (dotted) module name. It returns the zipimporter
        instance itself if the module was found, a string containing the
        full path name if it's possibly a portion of a namespace package,
        or None otherwise. The optional 'path' argument is ignored -- it's
        there for compatibility with the importer protocol.

        Deprecated since Python 3.10. Use find_spec() instead.
        """
        _warnings.warn("zipimporter.find_loader() is deprecated and slated for "
                       "removal in Python 3.12; use find_spec() instead",
                       DeprecationWarning)
        mi = _get_module_info(self, fullname)
        if mi is not None:
            # This is a module or package.
            return self, []

        # Not a module or regular package. See if this is a directory, and
        # therefore possibly a portion of a namespace package.

        # We're only interested in the last path component of fullname
        # earlier components are recorded in self.prefix.
        modpath = _get_module_path(self, fullname)
        if _is_dir(self, modpath):
            # This is possibly a portion of a namespace
            # package. Return the string representing its path,
            # without a trailing separator.
            return None, [f'{self.archive}{path_sep}{modpath}']

        return None, []


    # Check whether we can satisfy the import of the module named by
    # 'fullname'. Return self if we can, None if we can't.
    def find_module(self, fullname, path=None):
        """find_module(fullname, path=None) -> self or None.

        Search for a module specified by 'fullname'. 'fullname' must be the
        fully qualified (dotted) module name. It returns the zipimporter
        instance itself if the module was found, or None if it wasn't.
        The optional 'path' argument is ignored -- it's there for compatibility
        with the importer protocol.

        Deprecated since Python 3.10. Use find_spec() instead.
        """
        _warnings.warn("zipimporter.find_module() is deprecated and slated for "
                       "removal in Python 3.12; use find_spec() instead",
                       DeprecationWarning)
        return self.find_loader(fullname, path)[0]

    def find_spec(self, fullname, target=None):
        """Create a ModuleSpec for the specified module.

        Returns None if the module cannot be found.
        """
        module_info = _get_module_info(self, fullname)
        if module_info is not None:
            return _bootstrap.spec_from_loader(fullname, self, is_package=module_info)
        else:
            # Not a module or regular package. See if this is a directory, and
            # therefore possibly a portion of a namespace package.

            # We're only interested in the last path component of fullname
            # earlier components are recorded in self.prefix.
            modpath = _get_module_path(self, fullname)
            if _is_dir(self, modpath):
                # This is possibly a portion of a namespace
                # package. Return the string representing its path,
                # without a trailing separator.
                path = f'{self.archive}{path_sep}{modpath}'
                spec = _bootstrap.ModuleSpec(name=fullname, loader=None,
                                             is_package=True)
                spec.submodule_search_locations.append(path)
                return spec
            else:
                return None

    def get_code(self, fullname):
        """get_code(fullname) -> code object.

        Return the code object for the specified module. Raise ZipImportError
        if the module couldn't be imported.
        """
        code, ispackage, modpath = _get_module_code(self, fullname)
        return code


    def get_data(self, pathname):
        """get_data(pathname) -> string with file data.

        Return the data associated with 'pathname'. Raise OSError if
        the file wasn't found.
        """
        if alt_path_sep:
            pathname = pathname.replace(alt_path_sep, path_sep)

        key = pathname
        if pathname.startswith(self.archive + path_sep):
            key = pathname[len(self.archive + path_sep):]

        try:
            toc_entry = self._files[key]
        except KeyError:
            raise OSError(0, '', key)
        return _get_data(self.archive, toc_entry)


    # Return a string matching __file__ for the named module
    def get_filename(self, fullname):
        """get_filename(fullname) -> filename string.

        Return the filename for the specified module or raise ZipImportError
        if it couldn't be imported.
        """
        # Deciding the filename requires working out where the code
        # would come from if the module was actually loaded
        code, ispackage, modpath = _get_module_code(self, fullname)
        return modpath


    def get_source(self, fullname):
        """get_source(fullname) -> source string.

        Return the source code for the specified module. Raise ZipImportError
        if the module couldn't be found, return None if the archive does
        contain the module, but has no source for it.
        """
        mi = _get_module_info(self, fullname)
        if mi is None:
            raise ZipImportError(f"can't find module {fullname!r}", name=fullname)

        path = _get_module_path(self, fullname)
        if mi:
            fullpath = _bootstrap_external._path_join(path, '__init__.py')
        else:
            fullpath = f'{path}.py'

        try:
            toc_entry = self._files[fullpath]
        except KeyError:
            # we have the module, but no source
            return None
        return _get_data(self.archive, toc_entry).decode()


    # Return a bool signifying whether the module is a package or not.
    def is_package(self, fullname):
        """is_package(fullname) -> bool.

        Return True if the module specified by fullname is a package.
        Raise ZipImportError if the module couldn't be found.
        """
        mi = _get_module_info(self, fullname)
        if mi is None:
            raise ZipImportError(f"can't find module {fullname!r}", name=fullname)
        return mi


    # Load and return the module named by 'fullname'.
    def load_module(self, fullname):
        """load_module(fullname) -> module.

        Load the module specified by 'fullname'. 'fullname' must be the
        fully qualified (dotted) module name. It returns the imported
        module, or raises ZipImportError if it could not be imported.

        Deprecated since Python 3.10. Use exec_module() instead.
        """
        msg = ("zipimport.zipimporter.load_module() is deprecated and slated for "
               "removal in Python 3.12; use exec_module() instead")
        _warnings.warn(msg, DeprecationWarning)
        code, ispackage, modpath = _get_module_code(self, fullname)
        mod = sys.modules.get(fullname)
        if mod is None or not isinstance(mod, _module_type):
            mod = _module_type(fullname)
            sys.modules[fullname] = mod
        mod.__loader__ = self

        try:
            if ispackage:
                # add __path__ to the module *before* the code gets
                # executed
                path = _get_module_path(self, fullname)
                fullpath = _bootstrap_external._path_join(self.archive, path)
                mod.__path__ = [fullpath]

            if not hasattr(mod, '__builtins__'):
                mod.__builtins__ = __builtins__
            _bootstrap_external._fix_up_module(mod.__dict__, fullname, modpath)
            exec(code, mod.__dict__)
        except:
            del sys.modules[fullname]
            raise

        try:
            mod = sys.modules[fullname]
        except KeyError:
            raise ImportError(f'Loaded module {fullname!r} not found in sys.modules')
        _bootstrap._verbose_message('import {} # loaded from Zip {}', fullname, modpath)
        return mod


    def get_resource_reader(self, fullname):
        """Return the ResourceReader for a package in a zip file.

        If 'fullname' is a package within the zip file, return the
        'ResourceReader' object for the package.  Otherwise return None.
        """
        try:
            if not self.is_package(fullname):
                return None
        except ZipImportError:
            return None
        from importlib.readers import ZipReader
        return ZipReader(self, fullname)


    def invalidate_caches(self):
        """Reload the file data of the archive path."""
        try:
            self._files = _read_directory(self.archive)
            _zip_directory_cache[self.archive] = self._files
        except ZipImportError:
            _zip_directory_cache.pop(self.archive, None)
            self._files = {}


    def __repr__(self):
        return f'<zipimporter object "{self.archive}{path_sep}{self.prefix}">'


# _zip_searchorder defines how we search for a module in the Zip
# archive: we first search for a package __init__, then for
# non-package .pyc, and .py entries. The .pyc entries
# are swapped by initzipimport() if we run in optimized mode. Also,
# '/' is replaced by path_sep there.
_zip_searchorder = (
    (path_sep + '__init__.pyc', True, True),
    (path_sep + '__init__.py', False, True),
    ('.pyc', True, False),
    ('.py', False, False),
)

# Given a module name, return the potential file path in the
# archive (without extension).
def _get_module_path(self, fullname):
    return self.prefix + fullname.rpartition('.')[2]

# Does this path represent a directory?
def _is_dir(self, path):
    # See if this is a "directory". If so, it's eligible to be part
    # of a namespace package. We test by seeing if the name, with an
    # appended path separator, exists.
    dirpath = path + path_sep
    # If dirpath is present in self._files, we have a directory.
    return dirpath in self._files

# Return some information about a module.
def _get_module_info(self, fullname):
    path = _get_module_path(self, fullname)
    for suffix, isbytecode, ispackage in _zip_searchorder:
        fullpath = path + suffix
        if fullpath in self._files:
            return ispackage
    return None


# implementation

# _read_directory(archive) -> files dict (new reference)
#
# Given a path to a Zip archive, build a dict, mapping file names
# (local to the archive, using SEP as a separator) to toc entries.
#
# A toc_entry is a tuple:
#
# (__file__,        # value to use for __file__, available for all files,
#                   # encoded to the filesystem encoding
#  compress,        # compression kind; 0 for uncompressed
#  data_size,       # size of compressed data on disk
#  file_size,       # size of decompressed data
#  file_offset,     # offset of file header from start of archive
#  time,            # mod time of file (in dos format)
#  date,            # mod data of file (in dos format)
#  crc,             # crc checksum of the data
# )
#
# Directories can be recognized by the trailing path_sep in the name,
# data_size and file_offset are 0.
def _read_directory(archive):
    try:
        fp = _io.open_code(archive)
    except OSError:
        raise ZipImportError(f"can't open Zip file: {archive!r}", path=archive)

    with fp:
        # GH-87235: On macOS all file descriptors for /dev/fd/N share the same
        # file offset, reset the file offset after scanning the zipfile diretory
        # to not cause problems when some runs 'python3 /dev/fd/9 9<some_script'
        start_offset = fp.tell()
        try:
            try:
                fp.seek(-END_CENTRAL_DIR_SIZE, 2)
                header_position = fp.tell()
                buffer = fp.read(END_CENTRAL_DIR_SIZE)
            except OSError:
                raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
            if len(buffer) != END_CENTRAL_DIR_SIZE:
                raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
            if buffer[:4] != STRING_END_ARCHIVE:
                # Bad: End of Central Dir signature
                # Check if there's a comment.
                try:
                    fp.seek(0, 2)
                    file_size = fp.tell()
                except OSError:
                    raise ZipImportError(f"can't read Zip file: {archive!r}",
                                         path=archive)
                max_comment_start = max(file_size - MAX_COMMENT_LEN -
                                        END_CENTRAL_DIR_SIZE, 0)
                try:
                    fp.seek(max_comment_start)
                    data = fp.read()
                except OSError:
                    raise ZipImportError(f"can't read Zip file: {archive!r}",
                                         path=archive)
                pos = data.rfind(STRING_END_ARCHIVE)
                if pos < 0:
                    raise ZipImportError(f'not a Zip file: {archive!r}',
                                         path=archive)
                buffer = data[pos:pos+END_CENTRAL_DIR_SIZE]
                if len(buffer) != END_CENTRAL_DIR_SIZE:
                    raise ZipImportError(f"corrupt Zip file: {archive!r}",
                                         path=archive)
                header_position = file_size - len(data) + pos

            header_size = _unpack_uint32(buffer[12:16])
            header_offset = _unpack_uint32(buffer[16:20])
            if header_position < header_size:
                raise ZipImportError(f'bad central directory size: {archive!r}', path=archive)
            if header_position < header_offset:
                raise ZipImportError(f'bad central directory offset: {archive!r}', path=archive)
            header_position -= header_size
            arc_offset = header_position - header_offset
            if arc_offset < 0:
                raise ZipImportError(f'bad central directory size or offset: {archive!r}', path=archive)

            files = {}
            # Start of Central Directory
            count = 0
            try:
                fp.seek(header_position)
            except OSError:
                raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
            while True:
                buffer = fp.read(46)
                if len(buffer) < 4:
                    raise EOFError('EOF read where not expected')
                # Start of file header
                if buffer[:4] != b'PK\x01\x02':
                    break                                # Bad: Central Dir File Header
                if len(buffer) != 46:
                    raise EOFError('EOF read where not expected')
                flags = _unpack_uint16(buffer[8:10])
                compress = _unpack_uint16(buffer[10:12])
                time = _unpack_uint16(buffer[12:14])
                date = _unpack_uint16(buffer[14:16])
                crc = _unpack_uint32(buffer[16:20])
                data_size = _unpack_uint32(buffer[20:24])
                file_size = _unpack_uint32(buffer[24:28])
                name_size = _unpack_uint16(buffer[28:30])
                extra_size = _unpack_uint16(buffer[30:32])
                comment_size = _unpack_uint16(buffer[32:34])
                file_offset = _unpack_uint32(buffer[42:46])
                header_size = name_size + extra_size + comment_size
                if file_offset > header_offset:
                    raise ZipImportError(f'bad local header offset: {archive!r}', path=archive)
                file_offset += arc_offset

                try:
                    name = fp.read(name_size)
                except OSError:
                    raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
                if len(name) != name_size:
                    raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
                # On Windows, calling fseek to skip over the fields we don't use is
                # slower than reading the data because fseek flushes stdio's
                # internal buffers.    See issue #8745.
                try:
                    if len(fp.read(header_size - name_size)) != header_size - name_size:
                        raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
                except OSError:
                    raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)

                if flags & 0x800:
                    # UTF-8 file names extension
                    name = name.decode()
                else:
                    # Historical ZIP filename encoding
                    try:
                        name = name.decode('ascii')
                    except UnicodeDecodeError:
                        name = name.decode('latin1').translate(cp437_table)

                name = name.replace('/', path_sep)
                path = _bootstrap_external._path_join(archive, name)
                t = (path, compress, data_size, file_size, file_offset, time, date, crc)
                files[name] = t
                count += 1
        finally:
            fp.seek(start_offset)
    _bootstrap._verbose_message('zipimport: found {} names in {!r}', count, archive)
    return files

# During bootstrap, we may need to load the encodings
# package from a ZIP file. But the cp437 encoding is implemented
# in Python in the encodings package.
#
# Break out of this dependency by using the translation table for
# the cp437 encoding.
cp437_table = (
    # ASCII part, 8 rows x 16 chars
    '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
    '\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'
    ' !"#$%&\'()*+,-./'
    '0123456789:;<=>?'
    '@ABCDEFGHIJKLMNO'
    'PQRSTUVWXYZ[\\]^_'
    '`abcdefghijklmno'
    'pqrstuvwxyz{|}~\x7f'
    # non-ASCII part, 16 rows x 8 chars
    '\xc7\xfc\xe9\xe2\xe4\xe0\xe5\xe7'
    '\xea\xeb\xe8\xef\xee\xec\xc4\xc5'
    '\xc9\xe6\xc6\xf4\xf6\xf2\xfb\xf9'
    '\xff\xd6\xdc\xa2\xa3\xa5\u20a7\u0192'
    '\xe1\xed\xf3\xfa\xf1\xd1\xaa\xba'
    '\xbf\u2310\xac\xbd\xbc\xa1\xab\xbb'
    '\u2591\u2592\u2593\u2502\u2524\u2561\u2562\u2556'
    '\u2555\u2563\u2551\u2557\u255d\u255c\u255b\u2510'
    '\u2514\u2534\u252c\u251c\u2500\u253c\u255e\u255f'
    '\u255a\u2554\u2569\u2566\u2560\u2550\u256c\u2567'
    '\u2568\u2564\u2565\u2559\u2558\u2552\u2553\u256b'
    '\u256a\u2518\u250c\u2588\u2584\u258c\u2590\u2580'
    '\u03b1\xdf\u0393\u03c0\u03a3\u03c3\xb5\u03c4'
    '\u03a6\u0398\u03a9\u03b4\u221e\u03c6\u03b5\u2229'
    '\u2261\xb1\u2265\u2264\u2320\u2321\xf7\u2248'
    '\xb0\u2219\xb7\u221a\u207f\xb2\u25a0\xa0'
)

_importing_zlib = False

# Return the zlib.decompress function object, or NULL if zlib couldn't
# be imported. The function is cached when found, so subsequent calls
# don't import zlib again.
def _get_decompress_func():
    global _importing_zlib
    if _importing_zlib:
        # Someone has a zlib.py[co] in their Zip file
        # let's avoid a stack overflow.
        _bootstrap._verbose_message('zipimport: zlib UNAVAILABLE')
        raise ZipImportError("can't decompress data; zlib not available")

    _importing_zlib = True
    try:
        from zlib import decompress
    except Exception:
        _bootstrap._verbose_message('zipimport: zlib UNAVAILABLE')
        raise ZipImportError("can't decompress data; zlib not available")
    finally:
        _importing_zlib = False

    _bootstrap._verbose_message('zipimport: zlib available')
    return decompress

# Given a path to a Zip file and a toc_entry, return the (uncompressed) data.
def _get_data(archive, toc_entry):
    datapath, compress, data_size, file_size, file_offset, time, date, crc = toc_entry
    if data_size < 0:
        raise ZipImportError('negative data size')

    with _io.open_code(archive) as fp:
        # Check to make sure the local file header is correct
        try:
            fp.seek(file_offset)
        except OSError:
            raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
        buffer = fp.read(30)
        if len(buffer) != 30:
            raise EOFError('EOF read where not expected')

        if buffer[:4] != b'PK\x03\x04':
            # Bad: Local File Header
            raise ZipImportError(f'bad local file header: {archive!r}', path=archive)

        name_size = _unpack_uint16(buffer[26:28])
        extra_size = _unpack_uint16(buffer[28:30])
        header_size = 30 + name_size + extra_size
        file_offset += header_size  # Start of file data
        try:
            fp.seek(file_offset)
        except OSError:
            raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
        raw_data = fp.read(data_size)
        if len(raw_data) != data_size:
            raise OSError("zipimport: can't read data")

    if compress == 0:
        # data is not compressed
        return raw_data

    # Decompress with zlib
    try:
        decompress = _get_decompress_func()
    except Exception:
        raise ZipImportError("can't decompress data; zlib not available")
    return decompress(raw_data, -15)


# Lenient date/time comparison function. The precision of the mtime
# in the archive is lower than the mtime stored in a .pyc: we
# must allow a difference of at most one second.
def _eq_mtime(t1, t2):
    # dostime only stores even seconds, so be lenient
    return abs(t1 - t2) <= 1


# Given the contents of a .py[co] file, unmarshal the data
# and return the code object. Raises ImportError it the magic word doesn't
# match, or if the recorded .py[co] metadata does not match the source.
def _unmarshal_code(self, pathname, fullpath, fullname, data):
    exc_details = {
        'name': fullname,
        'path': fullpath,
    }

    flags = _bootstrap_external._classify_pyc(data, fullname, exc_details)

    hash_based = flags & 0b1 != 0
    if hash_based:
        check_source = flags & 0b10 != 0
        if (_imp.check_hash_based_pycs != 'never' and
                (check_source or _imp.check_hash_based_pycs == 'always')):
            source_bytes = _get_pyc_source(self, fullpath)
            if source_bytes is not None:
                source_hash = _imp.source_hash(
                    _bootstrap_external._RAW_MAGIC_NUMBER,
                    source_bytes,
                )

                _bootstrap_external._validate_hash_pyc(
                    data, source_hash, fullname, exc_details)
    else:
        source_mtime, source_size = \
            _get_mtime_and_size_of_source(self, fullpath)

        if source_mtime:
            # We don't use _bootstrap_external._validate_timestamp_pyc
            # to allow for a more lenient timestamp check.
            if (not _eq_mtime(_unpack_uint32(data[8:12]), source_mtime) or
                    _unpack_uint32(data[12:16]) != source_size):
                _bootstrap._verbose_message(
                    f'bytecode is stale for {fullname!r}')
                return None

    code = marshal.loads(data[16:])
    if not isinstance(code, _code_type):
        raise TypeError(f'compiled module {pathname!r} is not a code object')
    return code

_code_type = type(_unmarshal_code.__code__)


# Replace any occurrences of '\r\n?' in the input string with '\n'.
# This converts DOS and Mac line endings to Unix line endings.
def _normalize_line_endings(source):
    source = source.replace(b'\r\n', b'\n')
    source = source.replace(b'\r', b'\n')
    return source

# Given a string buffer containing Python source code, compile it
# and return a code object.
def _compile_source(pathname, source):
    source = _normalize_line_endings(source)
    return compile(source, pathname, 'exec', dont_inherit=True)

# Convert the date/time values found in the Zip archive to a value
# that's compatible with the time stamp stored in .pyc files.
def _parse_dostime(d, t):
    return time.mktime((
        (d >> 9) + 1980,    # bits 9..15: year
        (d >> 5) & 0xF,     # bits 5..8: month
        d & 0x1F,           # bits 0..4: day
        t >> 11,            # bits 11..15: hours
        (t >> 5) & 0x3F,    # bits 8..10: minutes
        (t & 0x1F) * 2,     # bits 0..7: seconds / 2
        -1, -1, -1))

# Given a path to a .pyc file in the archive, return the
# modification time of the matching .py file and its size,
# or (0, 0) if no source is available.
def _get_mtime_and_size_of_source(self, path):
    try:
        # strip 'c' or 'o' from *.py[co]
        assert path[-1:] in ('c', 'o')
        path = path[:-1]
        toc_entry = self._files[path]
        # fetch the time stamp of the .py file for comparison
        # with an embedded pyc time stamp
        time = toc_entry[5]
        date = toc_entry[6]
        uncompressed_size = toc_entry[3]
        return _parse_dostime(date, time), uncompressed_size
    except (KeyError, IndexError, TypeError):
        return 0, 0


# Given a path to a .pyc file in the archive, return the
# contents of the matching .py file, or None if no source
# is available.
def _get_pyc_source(self, path):
    # strip 'c' or 'o' from *.py[co]
    assert path[-1:] in ('c', 'o')
    path = path[:-1]

    try:
        toc_entry = self._files[path]
    except KeyError:
        return None
    else:
        return _get_data(self.archive, toc_entry)


# Get the code object associated with the module specified by
# 'fullname'.
def _get_module_code(self, fullname):
    path = _get_module_path(self, fullname)
    import_error = None
    for suffix, isbytecode, ispackage in _zip_searchorder:
        fullpath = path + suffix
        _bootstrap._verbose_message('trying {}{}{}', self.archive, path_sep, fullpath, verbosity=2)
        try:
            toc_entry = self._files[fullpath]
        except KeyError:
            pass
        else:
            modpath = toc_entry[0]
            data = _get_data(self.archive, toc_entry)
            code = None
            if isbytecode:
                try:
                    code = _unmarshal_code(self, modpath, fullpath, fullname, data)
                except ImportError as exc:
                    import_error = exc
            else:
                code = _compile_source(modpath, data)
            if code is None:
                # bad magic number or non-matching mtime
                # in byte code, try next
                continue
            modpath = toc_entry[0]
            return code, ispackage, modpath
    else:
        if import_error:
            msg = f"module load failed: {import_error}"
            raise ZipImportError(msg, name=fullname) from import_error
        else:
            raise ZipImportError(f"can't find module {fullname!r}", name=fullname)
<|endoftext|># Wrapper module for _socket, providing some additional facilities
# implemented in Python.

"""\
This module provides socket operations and some related functions.
On Unix, it supports IP (Internet Protocol) and Unix domain sockets.
On other systems, it only supports IP. Functions specific for a
socket are available as methods of the socket object.

Functions:

socket() -- create a new socket object
socketpair() -- create a pair of new socket objects [*]
fromfd() -- create a socket object from an open file descriptor [*]
send_fds() -- Send file descriptor to the socket.
recv_fds() -- Recieve file descriptors from the socket.
fromshare() -- create a socket object from data received from socket.share() [*]
gethostname() -- return the current hostname
gethostbyname() -- map a hostname to its IP number
gethostbyaddr() -- map an IP number or hostname to DNS info
getservbyname() -- map a service name and a protocol name to a port number
getprotobyname() -- map a protocol name (e.g. 'tcp') to a number
ntohs(), ntohl() -- convert 16, 32 bit int from network to host byte order
htons(), htonl() -- convert 16, 32 bit int from host to network byte order
inet_aton() -- convert IP addr string (123.45.67.89) to 32-bit packed format
inet_ntoa() -- convert 32-bit packed format IP to string (123.45.67.89)
socket.getdefaulttimeout() -- get the default timeout value
socket.setdefaulttimeout() -- set the default timeout value
create_connection() -- connects to an address, with an optional timeout and
                       optional source address.

 [*] not available on all platforms!

Special objects:

SocketType -- type object for socket objects
error -- exception raised for I/O errors
has_ipv6 -- boolean value indicating if IPv6 is supported

IntEnum constants:

AF_INET, AF_UNIX -- socket domains (first argument to socket() call)
SOCK_STREAM, SOCK_DGRAM, SOCK_RAW -- socket types (second argument)

Integer constants:

Many other constants may be defined; these may be used in calls to
the setsockopt() and getsockopt() methods.
"""

import _socket
from _socket import *

import os, sys, io, selectors
from enum import IntEnum, IntFlag

try:
    import errno
except ImportError:
    errno = None
EBADF = getattr(errno, 'EBADF', 9)
EAGAIN = getattr(errno, 'EAGAIN', 11)
EWOULDBLOCK = getattr(errno, 'EWOULDBLOCK', 11)

__all__ = ["fromfd", "getfqdn", "create_connection", "create_server",
           "has_dualstack_ipv6", "AddressFamily", "SocketKind"]
__all__.extend(os._get_exports_list(_socket))

# Set up the socket.AF_* socket.SOCK_* constants as members of IntEnums for
# nicer string representations.
# Note that _socket only knows about the integer values. The public interface
# in this module understands the enums and translates them back from integers
# where needed (e.g. .family property of a socket object).

IntEnum._convert_(
        'AddressFamily',
        __name__,
        lambda C: C.isupper() and C.startswith('AF_'))

IntEnum._convert_(
        'SocketKind',
        __name__,
        lambda C: C.isupper() and C.startswith('SOCK_'))

IntFlag._convert_(
        'MsgFlag',
        __name__,
        lambda C: C.isupper() and C.startswith('MSG_'))

IntFlag._convert_(
        'AddressInfo',
        __name__,
        lambda C: C.isupper() and C.startswith('AI_'))

_LOCALHOST    = '127.0.0.1'
_LOCALHOST_V6 = '::1'


def _intenum_converter(value, enum_klass):
    """Convert a numeric family value to an IntEnum member.

    If it's not a known member, return the numeric value itself.
    """
    try:
        return enum_klass(value)
    except ValueError:
        return value


# WSA error codes
if sys.platform.lower().startswith("win"):
    errorTab = {}
    errorTab[6] = "Specified event object handle is invalid."
    errorTab[8] = "Insufficient memory available."
    errorTab[87] = "One or more parameters are invalid."
    errorTab[995] = "Overlapped operation aborted."
    errorTab[996] = "Overlapped I/O event object not in signaled state."
    errorTab[997] = "Overlapped operation will complete later."
    errorTab[10004] = "The operation was interrupted."
    errorTab[10009] = "A bad file handle was passed."
    errorTab[10013] = "Permission denied."
    errorTab[10014] = "A fault occurred on the network??"  # WSAEFAULT
    errorTab[10022] = "An invalid operation was attempted."
    errorTab[10024] = "Too many open files."
    errorTab[10035] = "The socket operation would block"
    errorTab[10036] = "A blocking operation is already in progress."
    errorTab[10037] = "Operation already in progress."
    errorTab[10038] = "Socket operation on nonsocket."
    errorTab[10039] = "Destination address required."
    errorTab[10040] = "Message too long."
    errorTab[10041] = "Protocol wrong type for socket."
    errorTab[10042] = "Bad protocol option."
    errorTab[10043] = "Protocol not supported."
    errorTab[10044] = "Socket type not supported."
    errorTab[10045] = "Operation not supported."
    errorTab[10046] = "Protocol family not supported."
    errorTab[10047] = "Address family not supported by protocol family."
    errorTab[10048] = "The network address is in use."
    errorTab[10049] = "Cannot assign requested address."
    errorTab[10050] = "Network is down."
    errorTab[10051] = "Network is unreachable."
    errorTab[10052] = "Network dropped connection on reset."
    errorTab[10053] = "Software caused connection abort."
    errorTab[10054] = "The connection has been reset."
    errorTab[10055] = "No buffer space available."
    errorTab[10056] = "Socket is already connected."
    errorTab[10057] = "Socket is not connected."
    errorTab[10058] = "The network has been shut down."
    errorTab[10059] = "Too many references."
    errorTab[10060] = "The operation timed out."
    errorTab[10061] = "Connection refused."
    errorTab[10062] = "Cannot translate name."
    errorTab[10063] = "The name is too long."
    errorTab[10064] = "The host is down."
    errorTab[10065] = "The host is unreachable."
    errorTab[10066] = "Directory not empty."
    errorTab[10067] = "Too many processes."
    errorTab[10068] = "User quota exceeded."
    errorTab[10069] = "Disk quota exceeded."
    errorTab[10070] = "Stale file handle reference."
    errorTab[10071] = "Item is remote."
    errorTab[10091] = "Network subsystem is unavailable."
    errorTab[10092] = "Winsock.dll version out of range."
    errorTab[10093] = "Successful WSAStartup not yet performed."
    errorTab[10101] = "Graceful shutdown in progress."
    errorTab[10102] = "No more results from WSALookupServiceNext."
    errorTab[10103] = "Call has been canceled."
    errorTab[10104] = "Procedure call table is invalid."
    errorTab[10105] = "Service provider is invalid."
    errorTab[10106] = "Service provider failed to initialize."
    errorTab[10107] = "System call failure."
    errorTab[10108] = "Service not found."
    errorTab[10109] = "Class type not found."
    errorTab[10110] = "No more results from WSALookupServiceNext."
    errorTab[10111] = "Call was canceled."
    errorTab[10112] = "Database query was refused."
    errorTab[11001] = "Host not found."
    errorTab[11002] = "Nonauthoritative host not found."
    errorTab[11003] = "This is a nonrecoverable error."
    errorTab[11004] = "Valid name, no data record requested type."
    errorTab[11005] = "QoS receivers."
    errorTab[11006] = "QoS senders."
    errorTab[11007] = "No QoS senders."
    errorTab[11008] = "QoS no receivers."
    errorTab[11009] = "QoS request confirmed."
    errorTab[11010] = "QoS admission error."
    errorTab[11011] = "QoS policy failure."
    errorTab[11012] = "QoS bad style."
    errorTab[11013] = "QoS bad object."
    errorTab[11014] = "QoS traffic control error."
    errorTab[11015] = "QoS generic error."
    errorTab[11016] = "QoS service type error."
    errorTab[11017] = "QoS flowspec error."
    errorTab[11018] = "Invalid QoS provider buffer."
    errorTab[11019] = "Invalid QoS filter style."
    errorTab[11020] = "Invalid QoS filter style."
    errorTab[11021] = "Incorrect QoS filter count."
    errorTab[11022] = "Invalid QoS object length."
    errorTab[11023] = "Incorrect QoS flow count."
    errorTab[11024] = "Unrecognized QoS object."
    errorTab[11025] = "Invalid QoS policy object."
    errorTab[11026] = "Invalid QoS flow descriptor."
    errorTab[11027] = "Invalid QoS provider-specific flowspec."
    errorTab[11028] = "Invalid QoS provider-specific filterspec."
    errorTab[11029] = "Invalid QoS shape discard mode object."
    errorTab[11030] = "Invalid QoS shaping rate object."
    errorTab[11031] = "Reserved policy QoS element type."
    __all__.append("errorTab")


class _GiveupOnSendfile(Exception): pass


class socket(_socket.socket):

    """A subclass of _socket.socket adding the makefile() method."""

    __slots__ = ["__weakref__", "_io_refs", "_closed"]

    def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
        # For user code address family and type values are IntEnum members, but
        # for the underlying _socket.socket they're just integers. The
        # constructor of _socket.socket converts the given argument to an
        # integer automatically.
        if fileno is None:
            if family == -1:
                family = AF_INET
            if type == -1:
                type = SOCK_STREAM
            if proto == -1:
                proto = 0
        _socket.socket.__init__(self, family, type, proto, fileno)
        self._io_refs = 0
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not self._closed:
            self.close()

    def __repr__(self):
        """Wrap __repr__() to reveal the real class name and socket
        address(es).
        """
        closed = getattr(self, '_closed', False)
        s = "<%s.%s%s fd=%i, family=%s, type=%s, proto=%i" \
            % (self.__class__.__module__,
               self.__class__.__qualname__,
               " [closed]" if closed else "",
               self.fileno(),
               self.family,
               self.type,
               self.proto)
        if not closed:
            try:
                laddr = self.getsockname()
                if laddr:
                    s += ", laddr=%s" % str(laddr)
            except error:
                pass
            try:
                raddr = self.getpeername()
                if raddr:
                    s += ", raddr=%s" % str(raddr)
            except error:
                pass
        s += '>'
        return s

    def __getstate__(self):
        raise TypeError(f"cannot pickle {self.__class__.__name__!r} object")

    def dup(self):
        """dup() -> socket object

        Duplicate the socket. Return a new socket object connected to the same
        system resource. The new socket is non-inheritable.
        """
        fd = dup(self.fileno())
        sock = self.__class__(self.family, self.type, self.proto, fileno=fd)
        sock.settimeout(self.gettimeout())
        return sock

    def accept(self):
        """accept() -> (socket object, address info)

        Wait for an incoming connection.  Return a new socket
        representing the connection, and the address of the client.
        For IP sockets, the address info is a pair (hostaddr, port).
        """
        fd, addr = self._accept()
        sock = socket(self.family, self.type, self.proto, fileno=fd)
        # Issue #7995: if no default timeout is set and the listening
        # socket had a (non-zero) timeout, force the new socket in blocking
        # mode to override platform-specific socket flags inheritance.
        if getdefaulttimeout() is None and self.gettimeout():
            sock.setblocking(True)
        return sock, addr

    def makefile(self, mode="r", buffering=None, *,
                 encoding=None, errors=None, newline=None):
        """makefile(...) -> an I/O stream connected to the socket

        The arguments are as for io.open() after the filename, except the only
        supported mode values are 'r' (default), 'w' and 'b'.
        """
        # XXX refactor to share code?
        if not set(mode) <= {"r", "w", "b"}:
            raise ValueError("invalid mode %r (only r, w, b allowed)" % (mode,))
        writing = "w" in mode
        reading = "r" in mode or not writing
        assert reading or writing
        binary = "b" in mode
        rawmode = ""
        if reading:
            rawmode += "r"
        if writing:
            rawmode += "w"
        raw = SocketIO(self, rawmode)
        self._io_refs += 1
        if buffering is None:
            buffering = -1
        if buffering < 0:
            buffering = io.DEFAULT_BUFFER_SIZE
        if buffering == 0:
            if not binary:
                raise ValueError("unbuffered streams must be binary")
            return raw
        if reading and writing:
            buffer = io.BufferedRWPair(raw, raw, buffering)
        elif reading:
            buffer = io.BufferedReader(raw, buffering)
        else:
            assert writing
            buffer = io.BufferedWriter(raw, buffering)
        if binary:
            return buffer
        encoding = io.text_encoding(encoding)
        text = io.TextIOWrapper(buffer, encoding, errors, newline)
        text.mode = mode
        return text

    if hasattr(os, 'sendfile'):

        def _sendfile_use_sendfile(self, file, offset=0, count=None):
            self._check_sendfile_params(file, offset, count)
            sockno = self.fileno()
            try:
                fileno = file.fileno()
            except (AttributeError, io.UnsupportedOperation) as err:
                raise _GiveupOnSendfile(err)  # not a regular file
            try:
                fsize = os.fstat(fileno).st_size
            except OSError as err:
                raise _GiveupOnSendfile(err)  # not a regular file
            if not fsize:
                return 0  # empty file
            # Truncate to 1GiB to avoid OverflowError, see bpo-38319.
            blocksize = min(count or fsize, 2 ** 30)
            timeout = self.gettimeout()
            if timeout == 0:
                raise ValueError("non-blocking sockets are not supported")
            # poll/select have the advantage of not requiring any
            # extra file descriptor, contrarily to epoll/kqueue
            # (also, they require a single syscall).
            if hasattr(selectors, 'PollSelector'):
                selector = selectors.PollSelector()
            else:
                selector = selectors.SelectSelector()
            selector.register(sockno, selectors.EVENT_WRITE)

            total_sent = 0
            # localize variable access to minimize overhead
            selector_select = selector.select
            os_sendfile = os.sendfile
            try:
                while True:
                    if timeout and not selector_select(timeout):
                        raise TimeoutError('timed out')
                    if count:
                        blocksize = count - total_sent
                        if blocksize <= 0:
                            break
                    try:
                        sent = os_sendfile(sockno, fileno, offset, blocksize)
                    except BlockingIOError:
                        if not timeout:
                            # Block until the socket is ready to send some
                            # data; avoids hogging CPU resources.
                            selector_select()
                        continue
                    except OSError as err:
                        if total_sent == 0:
                            # We can get here for different reasons, the main
                            # one being 'file' is not a regular mmap(2)-like
                            # file, in which case we'll fall back on using
                            # plain send().
                            raise _GiveupOnSendfile(err)
                        raise err from None
                    else:
                        if sent == 0:
                            break  # EOF
                        offset += sent
                        total_sent += sent
                return total_sent
            finally:
                if total_sent > 0 and hasattr(file, 'seek'):
                    file.seek(offset)
    else:
        def _sendfile_use_sendfile(self, file, offset=0, count=None):
            raise _GiveupOnSendfile(
                "os.sendfile() not available on this platform")

    def _sendfile_use_send(self, file, offset=0, count=None):
        self._check_sendfile_params(file, offset, count)
        if self.gettimeout() == 0:
            raise ValueError("non-blocking sockets are not supported")
        if offset:
            file.seek(offset)
        blocksize = min(count, 8192) if count else 8192
        total_sent = 0
        # localize variable access to minimize overhead
        file_read = file.read
        sock_send = self.send
        try:
            while True:
                if count:
                    blocksize = min(count - total_sent, blocksize)
                    if blocksize <= 0:
                        break
                data = memoryview(file_read(blocksize))
                if not data:
                    break  # EOF
                while True:
                    try:
                        sent = sock_send(data)
                    except BlockingIOError:
                        continue
                    else:
                        total_sent += sent
                        if sent < len(data):
                            data = data[sent:]
                        else:
                            break
            return total_sent
        finally:
            if total_sent > 0 and hasattr(file, 'seek'):
                file.seek(offset + total_sent)

    def _check_sendfile_params(self, file, offset, count):
        if 'b' not in getattr(file, 'mode', 'b'):
            raise ValueError("file should be opened in binary mode")
        if not self.type & SOCK_STREAM:
            raise ValueError("only SOCK_STREAM type sockets are supported")
        if count is not None:
            if not isinstance(count, int):
                raise TypeError(
                    "count must be a positive integer (got {!r})".format(count))
            if count <= 0:
                raise ValueError(
                    "count must be a positive integer (got {!r})".format(count))

    def sendfile(self, file, offset=0, count=None):
        """sendfile(file[, offset[, count]]) -> sent

        Send a file until EOF is reached by using high-performance
        os.sendfile() and return the total number of bytes which
        were sent.
        *file* must be a regular file object opened in binary mode.
        If os.sendfile() is not available (e.g. Windows) or file is
        not a regular file socket.send() will be used instead.
        *offset* tells from where to start reading the file.
        If specified, *count* is the total number of bytes to transmit
        as opposed to sending the file until EOF is reached.
        File position is updated on return or also in case of error in
        which case file.tell() can be used to figure out the number of
        bytes which were sent.
        The socket must be of SOCK_STREAM type.
        Non-blocking sockets are not supported.
        """
        try:
            return self._sendfile_use_sendfile(file, offset, count)
        except _GiveupOnSendfile:
            return self._sendfile_use_send(file, offset, count)

    def _decref_socketios(self):
        if self._io_refs > 0:
            self._io_refs -= 1
        if self._closed:
            self.close()

    def _real_close(self, _ss=_socket.socket):
        # This function should not reference any globals. See issue #808164.
        _ss.close(self)

    def close(self):
        # This function should not reference any globals. See issue #808164.
        self._closed = True
        if self._io_refs <= 0:
            self._real_close()

    def detach(self):
        """detach() -> file descriptor

        Close the socket object without closing the underlying file descriptor.
        The object cannot be used after this call, but the file descriptor
        can be reused for other purposes.  The file descriptor is returned.
        """
        self._closed = True
        return super().detach()

    @property
    def family(self):
        """Read-only access to the address family for this socket.
        """
        return _intenum_converter(super().family, AddressFamily)

    @property
    def type(self):
        """Read-only access to the socket type.
        """
        return _intenum_converter(super().type, SocketKind)

    if os.name == 'nt':
        def get_inheritable(self):
            return os.get_handle_inheritable(self.fileno())
        def set_inheritable(self, inheritable):
            os.set_handle_inheritable(self.fileno(), inheritable)
    else:
        def get_inheritable(self):
            return os.get_inheritable(self.fileno())
        def set_inheritable(self, inheritable):
            os.set_inheritable(self.fileno(), inheritable)
    get_inheritable.__doc__ = "Get the inheritable flag of the socket"
    set_inheritable.__doc__ = "Set the inheritable flag of the socket"

def fromfd(fd, family, type, proto=0):
    """ fromfd(fd, family, type[, proto]) -> socket object

    Create a socket object from a duplicate of the given file
    descriptor.  The remaining arguments are the same as for socket().
    """
    nfd = dup(fd)
    return socket(family, type, proto, nfd)

if hasattr(_socket.socket, "sendmsg"):
    import array

    def send_fds(sock, buffers, fds, flags=0, address=None):
        """ send_fds(sock, buffers, fds[, flags[, address]]) -> integer

        Send the list of file descriptors fds over an AF_UNIX socket.
        """
        return sock.sendmsg(buffers, [(_socket.SOL_SOCKET,
            _socket.SCM_RIGHTS, array.array("i", fds))])
    __all__.append("send_fds")

if hasattr(_socket.socket, "recvmsg"):
    import array

    def recv_fds(sock, bufsize, maxfds, flags=0):
        """ recv_fds(sock, bufsize, maxfds[, flags]) -> (data, list of file
        descriptors, msg_flags, address)

        Receive up to maxfds file descriptors returning the message
        data and a list containing the descriptors.
        """
        # Array of ints
        fds = array.array("i")
        msg, ancdata, flags, addr = sock.recvmsg(bufsize,
            _socket.CMSG_LEN(maxfds * fds.itemsize))
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if (cmsg_level == _socket.SOL_SOCKET and cmsg_type == _socket.SCM_RIGHTS):
                fds.frombytes(cmsg_data[:
                        len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])

        return msg, list(fds), flags, addr
    __all__.append("recv_fds")

if hasattr(_socket.socket, "share"):
    def fromshare(info):
        """ fromshare(info) -> socket object

        Create a socket object from the bytes object returned by
        socket.share(pid).
        """
        return socket(0, 0, 0, info)
    __all__.append("fromshare")

if hasattr(_socket, "socketpair"):

    def socketpair(family=None, type=SOCK_STREAM, proto=0):
        """socketpair([family[, type[, proto]]]) -> (socket object, socket object)

        Create a pair of socket objects from the sockets returned by the platform
        socketpair() function.
        The arguments are the same as for socket() except the default family is
        AF_UNIX if defined on the platform; otherwise, the default is AF_INET.
        """
        if family is None:
            try:
                family = AF_UNIX
            except NameError:
                family = AF_INET
        a, b = _socket.socketpair(family, type, proto)
        a = socket(family, type, proto, a.detach())
        b = socket(family, type, proto, b.detach())
        return a, b

else:

    # Origin: https://gist.github.com/4325783, by Geert Jansen.  Public domain.
    def socketpair(family=AF_INET, type=SOCK_STREAM, proto=0):
        if family == AF_INET:
            host = _LOCALHOST
        elif family == AF_INET6:
            host = _LOCALHOST_V6
        else:
            raise ValueError("Only AF_INET and AF_INET6 socket address families "
                             "are supported")
        if type != SOCK_STREAM:
            raise ValueError("Only SOCK_STREAM socket type is supported")
        if proto != 0:
            raise ValueError("Only protocol zero is supported")

        # We create a connected TCP socket. Note the trick with
        # setblocking(False) that prevents us from having to create a thread.
        lsock = socket(family, type, proto)
        try:
            lsock.bind((host, 0))
            lsock.listen()
            # On IPv6, ignore flow_info and scope_id
            addr, port = lsock.getsockname()[:2]
            csock = socket(family, type, proto)
            try:
                csock.setblocking(False)
                try:
                    csock.connect((addr, port))
                except (BlockingIOError, InterruptedError):
                    pass
                csock.setblocking(True)
                ssock, _ = lsock.accept()
            except:
                csock.close()
                raise
        finally:
            lsock.close()
        return (ssock, csock)
    __all__.append("socketpair")

socketpair.__doc__ = """socketpair([family[, type[, proto]]]) -> (socket object, socket object)
Create a pair of socket objects from the sockets returned by the platform
socketpair() function.
The arguments are the same as for socket() except the default family is AF_UNIX
if defined on the platform; otherwise, the default is AF_INET.
"""

_blocking_errnos = { EAGAIN, EWOULDBLOCK }

class SocketIO(io.RawIOBase):

    """Raw I/O implementation for stream sockets.

    This class supports the makefile() method on sockets.  It provides
    the raw I/O interface on top of a socket object.
    """

    # One might wonder why not let FileIO do the job instead.  There are two
    # main reasons why FileIO is not adapted:
    # - it wouldn't work under Windows (where you can't used read() and
    #   write() on a socket handle)
    # - it wouldn't work with socket timeouts (FileIO would ignore the
    #   timeout and consider the socket non-blocking)

    # XXX More docs

    def __init__(self, sock, mode):
        if mode not in ("r", "w", "rw", "rb", "wb", "rwb"):
            raise ValueError("invalid mode: %r" % mode)
        io.RawIOBase.__init__(self)
        self._sock = sock
        if "b" not in mode:
            mode += "b"
        self._mode = mode
        self._reading = "r" in mode
        self._writing = "w" in mode
        self._timeout_occurred = False

    def readinto(self, b):
        """Read up to len(b) bytes into the writable buffer *b* and return
        the number of bytes read.  If the socket is non-blocking and no bytes
        are available, None is returned.

        If *b* is non-empty, a 0 return value indicates that the connection
        was shutdown at the other end.
        """
        self._checkClosed()
        self._checkReadable()
        if self._timeout_occurred:
            raise OSError("cannot read from timed out object")
        while True:
            try:
                return self._sock.recv_into(b)
            except timeout:
                self._timeout_occurred = True
                raise
            except error as e:
                if e.errno in _blocking_errnos:
                    return None
                raise

    def write(self, b):
        """Write the given bytes or bytearray object *b* to the socket
        and return the number of bytes written.  This can be less than
        len(b) if not all data could be written.  If the socket is
        non-blocking and no bytes could be written None is returned.
        """
        self._checkClosed()
        self._checkWritable()
        try:
            return self._sock.send(b)
        except error as e:
            # XXX what about EINTR?
            if e.errno in _blocking_errnos:
                return None
            raise

    def readable(self):
        """True if the SocketIO is open for reading.
        """
        if self.closed:
            raise ValueError("I/O operation on closed socket.")
        return self._reading

    def writable(self):
        """True if the SocketIO is open for writing.
        """
        if self.closed:
            raise ValueError("I/O operation on closed socket.")
        return self._writing

    def seekable(self):
        """True if the SocketIO is open for seeking.
        """
        if self.closed:
            raise ValueError("I/O operation on closed socket.")
        return super().seekable()

    def fileno(self):
        """Return the file descriptor of the underlying socket.
        """
        self._checkClosed()
        return self._sock.fileno()

    @property
    def name(self):
        if not self.closed:
            return self.fileno()
        else:
            return -1

    @property
    def mode(self):
        return self._mode

    def close(self):
        """Close the SocketIO object.  This doesn't close the underlying
        socket, except if all references to it have disappeared.
        """
        if self.closed:
            return
        io.RawIOBase.close(self)
        self._sock._decref_socketios()
        self._sock = None


def getfqdn(name=''):
    """Get fully qualified domain name from name.

    An empty argument is interpreted as meaning the local host.

    First the hostname returned by gethostbyaddr() is checked, then
    possibly existing aliases. In case no FQDN is available and `name`
    was given, it is returned unchanged. If `name` was empty, '0.0.0.0' or '::',
    hostname from gethostname() is returned.
    """
    name = name.strip()
    if not name or name in ('0.0.0.0', '::'):
        name = gethostname()
    try:
        hostname, aliases, ipaddrs = gethostbyaddr(name)
    except error:
        pass
    else:
        aliases.insert(0, hostname)
        for name in aliases:
            if '.' in name:
                break
        else:
            name = hostname
    return name


_GLOBAL_DEFAULT_TIMEOUT = object()

def create_connection(address, timeout=_GLOBAL_DEFAULT_TIMEOUT,
                      source_address=None):
    """Connect to *address* and return the socket object.

    Convenience function.  Connect to *address* (a 2-tuple ``(host,
    port)``) and return the socket object.  Passing the optional
    *timeout* parameter will set the timeout on the socket instance
    before attempting to connect.  If no *timeout* is supplied, the
    global default timeout setting returned by :func:`getdefaulttimeout`
    is used.  If *source_address* is set it must be a tuple of (host, port)
    for the socket to bind as a source address before making the connection.
    A host of '' or port 0 tells the OS to use the default.
    """

    host, port = address
    err = None
    for res in getaddrinfo(host, port, 0, SOCK_STREAM):
        af, socktype, proto, canonname, sa = res
        sock = None
        try:
            sock = socket(af, socktype, proto)
            if timeout is not _GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sa)
            # Break explicitly a reference cycle
            err = None
            return sock

        except error as _:
            err = _
            if sock is not None:
                sock.close()

    if err is not None:
        try:
            raise err
        finally:
            # Break explicitly a reference cycle
            err = None
    else:
        raise error("getaddrinfo returns an empty list")


def has_dualstack_ipv6():
    """Return True if the platform supports creating a SOCK_STREAM socket
    which can handle both AF_INET and AF_INET6 (IPv4 / IPv6) connections.
    """
    if not has_ipv6 \
            or not hasattr(_socket, 'IPPROTO_IPV6') \
            or not hasattr(_socket, 'IPV6_V6ONLY'):
        return False
    try:
        with socket(AF_INET6, SOCK_STREAM) as sock:
            sock.setsockopt(IPPROTO_IPV6, IPV6_V6ONLY, 0)
            return True
    except error:
        return False


def create_server(address, *, family=AF_INET, backlog=None, reuse_port=False,
                  dualstack_ipv6=False):
    """Convenience function which creates a SOCK_STREAM type socket
    bound to *address* (a 2-tuple (host, port)) and return the socket
    object.

    *family* should be either AF_INET or AF_INET6.
    *backlog* is the queue size passed to socket.listen().
    *reuse_port* dictates whether to use the SO_REUSEPORT socket option.
    *dualstack_ipv6*: if true and the platform supports it, it will
    create an AF_INET6 socket able to accept both IPv4 or IPv6
    connections. When false it will explicitly disable this option on
    platforms that enable it by default (e.g. Linux).

    >>> with create_server(('', 8000)) as server:
    ...     while True:
    ...         conn, addr = server.accept()
    ...         # handle new connection
    """
    if reuse_port and not hasattr(_socket, "SO_REUSEPORT"):
        raise ValueError("SO_REUSEPORT not supported on this platform")
    if dualstack_ipv6:
        if not has_dualstack_ipv6():
            raise ValueError("dualstack_ipv6 not supported on this platform")
        if family != AF_INET6:
            raise ValueError("dualstack_ipv6 requires AF_INET6 family")
    sock = socket(family, SOCK_STREAM)
    try:
        # Note about Windows. We don't set SO_REUSEADDR because:
        # 1) It's unnecessary: bind() will succeed even in case of a
        # previous closed socket on the same address and still in
        # TIME_WAIT state.
        # 2) If set, another socket is free to bind() on the same
        # address, effectively preventing this one from accepting
        # connections. Also, it may set the process in a state where
        # it'll no longer respond to any signals or graceful kills.
        # See: https://learn.microsoft.com/windows/win32/winsock/using-so-reuseaddr-and-so-exclusiveaddruse
        if os.name not in ('nt', 'cygwin') and \
                hasattr(_socket, 'SO_REUSEADDR'):
            try:
                sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
            except error:
                # Fail later on bind(), for platforms which may not
                # support this option.
                pass
        if reuse_port:
            sock.setsockopt(SOL_SOCKET, SO_REUSEPORT, 1)
        if has_ipv6 and family == AF_INET6:
            if dualstack_ipv6:
                sock.setsockopt(IPPROTO_IPV6, IPV6_V6ONLY, 0)
            elif hasattr(_socket, "IPV6_V6ONLY") and \
                    hasattr(_socket, "IPPROTO_IPV6"):
                sock.setsockopt(IPPROTO_IPV6, IPV6_V6ONLY, 1)
        try:
            sock.bind(address)
        except error as err:
            msg = '%s (while attempting to bind on address %r)' % \
                (err.strerror, address)
            raise error(err.errno, msg) from None
        if backlog is None:
            sock.listen()
        else:
            sock.listen(backlog)
        return sock
    except error:
        sock.close()
        raise


def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """Resolve host and port into list of address info entries.

    Translate the host/port argument into a sequence of 5-tuples that contain
    all the necessary arguments for creating a socket connected to that service.
    host is a domain name, a string representation of an IPv4/v6 address or
    None. port is a string service name such as 'http', a numeric port number or
    None. By passing None as the value of host and port, you can pass NULL to
    the underlying C API.

    The family, type and proto arguments can be optionally specified in order to
    narrow the list of addresses returned. Passing zero as a value for each of
    these arguments selects the full range of results.
    """
    # We override this function since we want to translate the numeric family
    # and socket type values to enum constants.
    addrlist = []
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
        af, socktype, proto, canonname, sa = res
        addrlist.append((_intenum_converter(af, AddressFamily),
                         _intenum_converter(socktype, SocketKind),
                         proto, canonname, sa))
    return addrlist
<|endoftext|>"""Internal classes used by the gzip, lzma and bz2 modules"""

import io
import sys

BUFFER_SIZE = io.DEFAULT_BUFFER_SIZE  # Compressed data read chunk size


class BaseStream(io.BufferedIOBase):
    """Mode-checking helper functions."""

    def _check_not_closed(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")

    def _check_can_read(self):
        if not self.readable():
            raise io.UnsupportedOperation("File not open for reading")

    def _check_can_write(self):
        if not self.writable():
            raise io.UnsupportedOperation("File not open for writing")

    def _check_can_seek(self):
        if not self.readable():
            raise io.UnsupportedOperation("Seeking is only supported "
                                          "on files open for reading")
        if not self.seekable():
            raise io.UnsupportedOperation("The underlying file object "
                                          "does not support seeking")


class DecompressReader(io.RawIOBase):
    """Adapts the decompressor API to a RawIOBase reader API"""

    def readable(self):
        return True

    def __init__(self, fp, decomp_factory, trailing_error=(), **decomp_args):
        self._fp = fp
        self._eof = False
        self._pos = 0  # Current offset in decompressed stream

        # Set to size of decompressed stream once it is known, for SEEK_END
        self._size = -1

        # Save the decompressor factory and arguments.
        # If the file contains multiple compressed streams, each
        # stream will need a separate decompressor object. A new decompressor
        # object is also needed when implementing a backwards seek().
        self._decomp_factory = decomp_factory
        self._decomp_args = decomp_args
        self._decompressor = self._decomp_factory(**self._decomp_args)

        # Exception class to catch from decompressor signifying invalid
        # trailing data to ignore
        self._trailing_error = trailing_error

    def close(self):
        self._decompressor = None
        return super().close()

    def seekable(self):
        return self._fp.seekable()

    def readinto(self, b):
        with memoryview(b) as view, view.cast("B") as byte_view:
            data = self.read(len(byte_view))
            byte_view[:len(data)] = data
        return len(data)

    def read(self, size=-1):
        if size < 0:
            return self.readall()

        if not size or self._eof:
            return b""
        data = None  # Default if EOF is encountered
        # Depending on the input data, our call to the decompressor may not
        # return any data. In this case, try again after reading another block.
        while True:
            if self._decompressor.eof:
                rawblock = (self._decompressor.unused_data or
                            self._fp.read(BUFFER_SIZE))
                if not rawblock:
                    break
                # Continue to next stream.
                self._decompressor = self._decomp_factory(
                    **self._decomp_args)
                try:
                    data = self._decompressor.decompress(rawblock, size)
                except self._trailing_error:
                    # Trailing data isn't a valid compressed stream; ignore it.
                    break
            else:
                if self._decompressor.needs_input:
                    rawblock = self._fp.read(BUFFER_SIZE)
                    if not rawblock:
                        raise EOFError("Compressed file ended before the "
                                       "end-of-stream marker was reached")
                else:
                    rawblock = b""
                data = self._decompressor.decompress(rawblock, size)
            if data:
                break
        if not data:
            self._eof = True
            self._size = self._pos
            return b""
        self._pos += len(data)
        return data

    def readall(self):
        chunks = []
        # sys.maxsize means the max length of output buffer is unlimited,
        # so that the whole input buffer can be decompressed within one
        # .decompress() call.
        while data := self.read(sys.maxsize):
            chunks.append(data)

        return b"".join(chunks)

    # Rewind the file to the beginning of the data stream.
    def _rewind(self):
        self._fp.seek(0)
        self._eof = False
        self._pos = 0
        self._decompressor = self._decomp_factory(**self._decomp_args)

    def seek(self, offset, whence=io.SEEK_SET):
        # Recalculate offset as an absolute file position.
        if whence == io.SEEK_SET:
            pass
        elif whence == io.SEEK_CUR:
            offset = self._pos + offset
        elif whence == io.SEEK_END:
            # Seeking relative to EOF - we need to know the file's size.
            if self._size < 0:
                while self.read(io.DEFAULT_BUFFER_SIZE):
                    pass
            offset = self._size + offset
        else:
            raise ValueError("Invalid value for whence: {}".format(whence))

        # Make it so that offset is the number of bytes to skip forward.
        if offset < self._pos:
            self._rewind()
        else:
            offset -= self._pos

        # Read and discard data until we reach the desired position.
        while offset > 0:
            data = self.read(min(io.DEFAULT_BUFFER_SIZE, offset))
            if not data:
                break
            offset -= len(data)

        return self._pos

    def tell(self):
        """Return the current file position."""
        return self._pos
<|endoftext|>"""An object-oriented interface to .netrc files."""

# Module and documentation by Eric S. Raymond, 21 Dec 1998

import os, shlex, stat

__all__ = ["netrc", "NetrcParseError"]


class NetrcParseError(Exception):
    """Exception raised on syntax errors in the .netrc file."""
    def __init__(self, msg, filename=None, lineno=None):
        self.filename = filename
        self.lineno = lineno
        self.msg = msg
        Exception.__init__(self, msg)

    def __str__(self):
        return "%s (%s, line %s)" % (self.msg, self.filename, self.lineno)


class netrc:
    def __init__(self, file=None):
        default_netrc = file is None
        if file is None:
            file = os.path.join(os.path.expanduser("~"), ".netrc")
        self.hosts = {}
        self.macros = {}
        try:
            with open(file, encoding="utf-8") as fp:
                self._parse(file, fp, default_netrc)
        except UnicodeDecodeError:
            with open(file, encoding="locale") as fp:
                self._parse(file, fp, default_netrc)

    def _parse(self, file, fp, default_netrc):
        lexer = shlex.shlex(fp)
        lexer.wordchars += r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        lexer.commenters = lexer.commenters.replace('#', '')
        while 1:
            # Look for a machine, default, or macdef top-level keyword
            saved_lineno = lexer.lineno
            toplevel = tt = lexer.get_token()
            if not tt:
                break
            elif tt[0] == '#':
                if lexer.lineno == saved_lineno and len(tt) == 1:
                    lexer.instream.readline()
                continue
            elif tt == 'machine':
                entryname = lexer.get_token()
            elif tt == 'default':
                entryname = 'default'
            elif tt == 'macdef':                # Just skip to end of macdefs
                entryname = lexer.get_token()
                self.macros[entryname] = []
                lexer.whitespace = ' \t'
                while 1:
                    line = lexer.instream.readline()
                    if not line or line == '\012':
                        lexer.whitespace = ' \t\r\n'
                        break
                    self.macros[entryname].append(line)
                continue
            else:
                raise NetrcParseError(
                    "bad toplevel token %r" % tt, file, lexer.lineno)

            # We're looking at start of an entry for a named machine or default.
            login = ''
            account = password = None
            self.hosts[entryname] = {}
            while 1:
                tt = lexer.get_token()
                if (tt.startswith('#') or
                    tt in {'', 'machine', 'default', 'macdef'}):
                    if password:
                        self.hosts[entryname] = (login, account, password)
                        lexer.push_token(tt)
                        break
                    else:
                        raise NetrcParseError(
                            "malformed %s entry %s terminated by %s"
                            % (toplevel, entryname, repr(tt)),
                            file, lexer.lineno)
                elif tt == 'login' or tt == 'user':
                    login = lexer.get_token()
                elif tt == 'account':
                    account = lexer.get_token()
                elif tt == 'password':
                    if os.name == 'posix' and default_netrc:
                        prop = os.fstat(fp.fileno())
                        if prop.st_uid != os.getuid():
                            import pwd
                            try:
                                fowner = pwd.getpwuid(prop.st_uid)[0]
                            except KeyError:
                                fowner = 'uid %s' % prop.st_uid
                            try:
                                user = pwd.getpwuid(os.getuid())[0]
                            except KeyError:
                                user = 'uid %s' % os.getuid()
                            raise NetrcParseError(
                                ("~/.netrc file owner (%s) does not match"
                                 " current user (%s)") % (fowner, user),
                                file, lexer.lineno)
                        if (prop.st_mode & (stat.S_IRWXG | stat.S_IRWXO)):
                            raise NetrcParseError(
                               "~/.netrc access too permissive: access"
                               " permissions must restrict access to only"
                               " the owner", file, lexer.lineno)
                    password = lexer.get_token()
                else:
                    raise NetrcParseError("bad follower token %r" % tt,
                                          file, lexer.lineno)

    def authenticators(self, host):
        """Return a (user, account, password) tuple for given host."""
        if host in self.hosts:
            return self.hosts[host]
        elif 'default' in self.hosts:
            return self.hosts['default']
        else:
            return None

    def __repr__(self):
        """Dump the class data in the format of a .netrc file."""
        rep = ""
        for host in self.hosts.keys():
            attrs = self.hosts[host]
            rep += f"machine {host}\n\tlogin {attrs[0]}\n"
            if attrs[1]:
                rep += f"\taccount {attrs[1]}\n"
            rep += f"\tpassword {attrs[2]}\n"
        for macro in self.macros.keys():
            rep += f"macdef {macro}\n"
            for line in self.macros[macro]:
                rep += line
            rep += "\n"
        return rep

if __name__ == '__main__':
    print(netrc())
<|endoftext|>"""Interface to the compiler's internal symbol tables"""

import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
     DEF_IMPORT, DEF_BOUND, DEF_ANNOT, SCOPE_OFF, SCOPE_MASK, FREE,
     LOCAL, GLOBAL_IMPLICIT, GLOBAL_EXPLICIT, CELL)

import weakref

__all__ = ["symtable", "SymbolTable", "Class", "Function", "Symbol"]

def symtable(code, filename, compile_type):
    """ Return the toplevel *SymbolTable* for the source code.

    *filename* is the name of the file with the code
    and *compile_type* is the *compile()* mode argument.
    """
    top = _symtable.symtable(code, filename, compile_type)
    return _newSymbolTable(top, filename)

class SymbolTableFactory:
    def __init__(self):
        self.__memo = weakref.WeakValueDictionary()

    def new(self, table, filename):
        if table.type == _symtable.TYPE_FUNCTION:
            return Function(table, filename)
        if table.type == _symtable.TYPE_CLASS:
            return Class(table, filename)
        return SymbolTable(table, filename)

    def __call__(self, table, filename):
        key = table, filename
        obj = self.__memo.get(key, None)
        if obj is None:
            obj = self.__memo[key] = self.new(table, filename)
        return obj

_newSymbolTable = SymbolTableFactory()


class SymbolTable:

    def __init__(self, raw_table, filename):
        self._table = raw_table
        self._filename = filename
        self._symbols = {}

    def __repr__(self):
        if self.__class__ == SymbolTable:
            kind = ""
        else:
            kind = "%s " % self.__class__.__name__

        if self._table.name == "top":
            return "<{0}SymbolTable for module {1}>".format(kind, self._filename)
        else:
            return "<{0}SymbolTable for {1} in {2}>".format(kind,
                                                            self._table.name,
                                                            self._filename)

    def get_type(self):
        """Return the type of the symbol table.

        The values retuned are 'class', 'module' and
        'function'.
        """
        if self._table.type == _symtable.TYPE_MODULE:
            return "module"
        if self._table.type == _symtable.TYPE_FUNCTION:
            return "function"
        if self._table.type == _symtable.TYPE_CLASS:
            return "class"
        assert self._table.type in (1, 2, 3), \
               "unexpected type: {0}".format(self._table.type)

    def get_id(self):
        """Return an identifier for the table.
        """
        return self._table.id

    def get_name(self):
        """Return the table's name.

        This corresponds to the name of the class, function
        or 'top' if the table is for a class, function or
        global respectively.
        """
        return self._table.name

    def get_lineno(self):
        """Return the number of the first line in the
        block for the table.
        """
        return self._table.lineno

    def is_optimized(self):
        """Return *True* if the locals in the table
        are optimizable.
        """
        return bool(self._table.type == _symtable.TYPE_FUNCTION)

    def is_nested(self):
        """Return *True* if the block is a nested class
        or function."""
        return bool(self._table.nested)

    def has_children(self):
        """Return *True* if the block has nested namespaces.
        """
        return bool(self._table.children)

    def get_identifiers(self):
        """Return a view object containing the names of symbols in the table.
        """
        return self._table.symbols.keys()

    def lookup(self, name):
        """Lookup a *name* in the table.

        Returns a *Symbol* instance.
        """
        sym = self._symbols.get(name)
        if sym is None:
            flags = self._table.symbols[name]
            namespaces = self.__check_children(name)
            module_scope = (self._table.name == "top")
            sym = self._symbols[name] = Symbol(name, flags, namespaces,
                                               module_scope=module_scope)
        return sym

    def get_symbols(self):
        """Return a list of *Symbol* instances for
        names in the table.
        """
        return [self.lookup(ident) for ident in self.get_identifiers()]

    def __check_children(self, name):
        return [_newSymbolTable(st, self._filename)
                for st in self._table.children
                if st.name == name]

    def get_children(self):
        """Return a list of the nested symbol tables.
        """
        return [_newSymbolTable(st, self._filename)
                for st in self._table.children]


class Function(SymbolTable):

    # Default values for instance variables
    __params = None
    __locals = None
    __frees = None
    __globals = None
    __nonlocals = None

    def __idents_matching(self, test_func):
        return tuple(ident for ident in self.get_identifiers()
                     if test_func(self._table.symbols[ident]))

    def get_parameters(self):
        """Return a tuple of parameters to the function.
        """
        if self.__params is None:
            self.__params = self.__idents_matching(lambda x:x & DEF_PARAM)
        return self.__params

    def get_locals(self):
        """Return a tuple of locals in the function.
        """
        if self.__locals is None:
            locs = (LOCAL, CELL)
            test = lambda x: ((x >> SCOPE_OFF) & SCOPE_MASK) in locs
            self.__locals = self.__idents_matching(test)
        return self.__locals

    def get_globals(self):
        """Return a tuple of globals in the function.
        """
        if self.__globals is None:
            glob = (GLOBAL_IMPLICIT, GLOBAL_EXPLICIT)
            test = lambda x:((x >> SCOPE_OFF) & SCOPE_MASK) in glob
            self.__globals = self.__idents_matching(test)
        return self.__globals

    def get_nonlocals(self):
        """Return a tuple of nonlocals in the function.
        """
        if self.__nonlocals is None:
            self.__nonlocals = self.__idents_matching(lambda x:x & DEF_NONLOCAL)
        return self.__nonlocals

    def get_frees(self):
        """Return a tuple of free variables in the function.
        """
        if self.__frees is None:
            is_free = lambda x:((x >> SCOPE_OFF) & SCOPE_MASK) == FREE
            self.__frees = self.__idents_matching(is_free)
        return self.__frees


class Class(SymbolTable):

    __methods = None

    def get_methods(self):
        """Return a tuple of methods declared in the class.
        """
        if self.__methods is None:
            d = {}
            for st in self._table.children:
                d[st.name] = 1
            self.__methods = tuple(d)
        return self.__methods


class Symbol:

    def __init__(self, name, flags, namespaces=None, *, module_scope=False):
        self.__name = name
        self.__flags = flags
        self.__scope = (flags >> SCOPE_OFF) & SCOPE_MASK # like PyST_GetScope()
        self.__namespaces = namespaces or ()
        self.__module_scope = module_scope

    def __repr__(self):
        return "<symbol {0!r}>".format(self.__name)

    def get_name(self):
        """Return a name of a symbol.
        """
        return self.__name

    def is_referenced(self):
        """Return *True* if the symbol is used in
        its block.
        """
        return bool(self.__flags & _symtable.USE)

    def is_parameter(self):
        """Return *True* if the symbol is a parameter.
        """
        return bool(self.__flags & DEF_PARAM)

    def is_global(self):
        """Return *True* if the sysmbol is global.
        """
        return bool(self.__scope in (GLOBAL_IMPLICIT, GLOBAL_EXPLICIT)
                    or (self.__module_scope and self.__flags & DEF_BOUND))

    def is_nonlocal(self):
        """Return *True* if the symbol is nonlocal."""
        return bool(self.__flags & DEF_NONLOCAL)

    def is_declared_global(self):
        """Return *True* if the symbol is declared global
        with a global statement."""
        return bool(self.__scope == GLOBAL_EXPLICIT)

    def is_local(self):
        """Return *True* if the symbol is local.
        """
        return bool(self.__scope in (LOCAL, CELL)
                    or (self.__module_scope and self.__flags & DEF_BOUND))

    def is_annotated(self):
        """Return *True* if the symbol is annotated.
        """
        return bool(self.__flags & DEF_ANNOT)

    def is_free(self):
        """Return *True* if a referenced symbol is
        not assigned to.
        """
        return bool(self.__scope == FREE)

    def is_imported(self):
        """Return *True* if the symbol is created from
        an import statement.
        """
        return bool(self.__flags & DEF_IMPORT)

    def is_assigned(self):
        """Return *True* if a symbol is assigned to."""
        return bool(self.__flags & DEF_LOCAL)

    def is_namespace(self):
        """Returns *True* if name binding introduces new namespace.

        If the name is used as the target of a function or class
        statement, this will be true.

        Note that a single name can be bound to multiple objects.  If
        is_namespace() is true, the name may also be bound to other
        objects, like an int or list, that does not introduce a new
        namespace.
        """
        return bool(self.__namespaces)

    def get_namespaces(self):
        """Return a list of namespaces bound to this name"""
        return self.__namespaces

    def get_namespace(self):
        """Return the single namespace bound to this name.

        Raises ValueError if the name is bound to multiple namespaces.
        """
        if len(self.__namespaces) != 1:
            raise ValueError("name is bound to multiple namespaces")
        return self.__namespaces[0]

if __name__ == "__main__":
    import os, sys
    with open(sys.argv[0]) as f:
        src = f.read()
    mod = symtable(src, os.path.split(sys.argv[0])[1], "exec")
    for ident in mod.get_identifiers():
        info = mod.lookup(ident)
        print(info, info.is_local(), info.is_namespace())
<|endoftext|>"""Cache lines from Python source files.

This is intended to read lines from modules imported -- hence if a filename
is not found, it will look down the module search path for a file by
that name.
"""

import functools
import sys
import os
import tokenize

__all__ = ["getline", "clearcache", "checkcache", "lazycache"]


# The cache. Maps filenames to either a thunk which will provide source code,
# or a tuple (size, mtime, lines, fullname) once loaded.
cache = {}


def clearcache():
    """Clear the cache entirely."""
    cache.clear()


def getline(filename, lineno, module_globals=None):
    """Get a line for a Python source file from the cache.
    Update the cache if it doesn't contain an entry for this file already."""

    lines = getlines(filename, module_globals)
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1]
    return ''


def getlines(filename, module_globals=None):
    """Get the lines for a Python source file from the cache.
    Update the cache if it doesn't contain an entry for this file already."""

    if filename in cache:
        entry = cache[filename]
        if len(entry) != 1:
            return cache[filename][2]

    try:
        return updatecache(filename, module_globals)
    except MemoryError:
        clearcache()
        return []


def checkcache(filename=None):
    """Discard cache entries that are out of date.
    (This is not checked upon each call!)"""

    if filename is None:
        filenames = list(cache.keys())
    elif filename in cache:
        filenames = [filename]
    else:
        return

    for filename in filenames:
        entry = cache[filename]
        if len(entry) == 1:
            # lazy cache entry, leave it lazy.
            continue
        size, mtime, lines, fullname = entry
        if mtime is None:
            continue   # no-op for files loaded via a __loader__
        try:
            stat = os.stat(fullname)
        except OSError:
            cache.pop(filename, None)
            continue
        if size != stat.st_size or mtime != stat.st_mtime:
            cache.pop(filename, None)


def updatecache(filename, module_globals=None):
    """Update a cache entry and return its list of lines.
    If something's wrong, print a message, discard the cache entry,
    and return an empty list."""

    if filename in cache:
        if len(cache[filename]) != 1:
            cache.pop(filename, None)
    if not filename or (filename.startswith('<') and filename.endswith('>')):
        return []

    fullname = filename
    try:
        stat = os.stat(fullname)
    except OSError:
        basename = filename

        # Realise a lazy loader based lookup if there is one
        # otherwise try to lookup right now.
        if lazycache(filename, module_globals):
            try:
                data = cache[filename][0]()
            except (ImportError, OSError):
                pass
            else:
                if data is None:
                    # No luck, the PEP302 loader cannot find the source
                    # for this module.
                    return []
                cache[filename] = (
                    len(data),
                    None,
                    [line + '\n' for line in data.splitlines()],
                    fullname
                )
                return cache[filename][2]

        # Try looking through the module search path, which is only useful
        # when handling a relative filename.
        if os.path.isabs(filename):
            return []

        for dirname in sys.path:
            try:
                fullname = os.path.join(dirname, basename)
            except (TypeError, AttributeError):
                # Not sufficiently string-like to do anything useful with.
                continue
            try:
                stat = os.stat(fullname)
                break
            except OSError:
                pass
        else:
            return []
    try:
        with tokenize.open(fullname) as fp:
            lines = fp.readlines()
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    size, mtime = stat.st_size, stat.st_mtime
    cache[filename] = size, mtime, lines, fullname
    return lines


def lazycache(filename, module_globals):
    """Seed the cache for filename with module_globals.

    The module loader will be asked for the source only when getlines is
    called, not immediately.

    If there is an entry in the cache already, it is not altered.

    :return: True if a lazy load is registered in the cache,
        otherwise False. To register such a load a module loader with a
        get_source method must be found, the filename must be a cacheable
        filename, and the filename must not be already cached.
    """
    if filename in cache:
        if len(cache[filename]) == 1:
            return True
        else:
            return False
    if not filename or (filename.startswith('<') and filename.endswith('>')):
        return False
    # Try for a __loader__, if available
    if module_globals and '__name__' in module_globals:
        name = module_globals['__name__']
        if (loader := module_globals.get('__loader__')) is None:
            if spec := module_globals.get('__spec__'):
                try:
                    loader = spec.loader
                except AttributeError:
                    pass
        get_source = getattr(loader, 'get_source', None)

        if name and get_source:
            get_lines = functools.partial(get_source, name)
            cache[filename] = (get_lines,)
            return True
    return False
<|endoftext|>r"""Utilities to compile possibly incomplete Python source code.

This module provides two interfaces, broadly similar to the builtin
function compile(), which take program text, a filename and a 'mode'
and:

- Return code object if the command is complete and valid
- Return None if the command is incomplete
- Raise SyntaxError, ValueError or OverflowError if the command is a
  syntax error (OverflowError and ValueError can be produced by
  malformed literals).

The two interfaces are:

compile_command(source, filename, symbol):

    Compiles a single command in the manner described above.

CommandCompiler():

    Instances of this class have __call__ methods identical in
    signature to compile_command; the difference is that if the
    instance compiles program text containing a __future__ statement,
    the instance 'remembers' and compiles all subsequent program texts
    with the statement in force.

The module also provides another class:

Compile():

    Instances of this class act like the built-in function compile,
    but with 'memory' in the sense described above.
"""

import __future__
import warnings

_features = [getattr(__future__, fname)
             for fname in __future__.all_feature_names]

__all__ = ["compile_command", "Compile", "CommandCompiler"]

# The following flags match the values from Include/cpython/compile.h
# Caveat emptor: These flags are undocumented on purpose and depending
# on their effect outside the standard library is **unsupported**.
PyCF_DONT_IMPLY_DEDENT = 0x200          
PyCF_ALLOW_INCOMPLETE_INPUT = 0x4000

def _maybe_compile(compiler, source, filename, symbol):
    # Check for source consisting of only blank lines and comments.
    for line in source.split("\n"):
        line = line.strip()
        if line and line[0] != '#':
            break               # Leave it alone.
    else:
        if symbol != "eval":
            source = "pass"     # Replace it with a 'pass' statement

    # Disable compiler warnings when checking for incomplete input.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
        try:
            compiler(source, filename, symbol)
        except SyntaxError:  # Let other compile() errors propagate.
            try:
                compiler(source + "\n", filename, symbol)
                return None
            except SyntaxError as e:
                if "incomplete input" in str(e):
                    return None
                # fallthrough

    return compiler(source, filename, symbol)


def _is_syntax_error(err1, err2):
    rep1 = repr(err1)
    rep2 = repr(err2)
    if "was never closed" in rep1 and "was never closed" in rep2:
        return False
    if rep1 == rep2:
        return True
    return False

def _compile(source, filename, symbol):
    return compile(source, filename, symbol, PyCF_DONT_IMPLY_DEDENT | PyCF_ALLOW_INCOMPLETE_INPUT)

def compile_command(source, filename="<input>", symbol="single"):
    r"""Compile a command and determine whether it is incomplete.

    Arguments:

    source -- the source string; may contain \n characters
    filename -- optional filename from which source was read; default
                "<input>"
    symbol -- optional grammar start symbol; "single" (default), "exec"
              or "eval"

    Return value / exceptions raised:

    - Return a code object if the command is complete and valid
    - Return None if the command is incomplete
    - Raise SyntaxError, ValueError or OverflowError if the command is a
      syntax error (OverflowError and ValueError can be produced by
      malformed literals).
    """
    return _maybe_compile(_compile, source, filename, symbol)

class Compile:
    """Instances of this class behave much like the built-in compile
    function, but if one is used to compile text containing a future
    statement, it "remembers" and compiles all subsequent program texts
    with the statement in force."""
    def __init__(self):
        self.flags = PyCF_DONT_IMPLY_DEDENT | PyCF_ALLOW_INCOMPLETE_INPUT

    def __call__(self, source, filename, symbol):
        codeob = compile(source, filename, symbol, self.flags, True)
        for feature in _features:
            if codeob.co_flags & feature.compiler_flag:
                self.flags |= feature.compiler_flag
        return codeob

class CommandCompiler:
    """Instances of this class have __call__ methods identical in
    signature to compile_command; the difference is that if the
    instance compiles program text containing a __future__ statement,
    the instance 'remembers' and compiles all subsequent program texts
    with the statement in force."""

    def __init__(self,):
        self.compiler = Compile()

    def __call__(self, source, filename="<input>", symbol="single"):
        r"""Compile a command and determine whether it is incomplete.

        Arguments:

        source -- the source string; may contain \n characters
        filename -- optional filename from which source was read;
                    default "<input>"
        symbol -- optional grammar start symbol; "single" (default) or
                  "eval"

        Return value / exceptions raised:

        - Return a code object if the command is complete and valid
        - Return None if the command is incomplete
        - Raise SyntaxError, ValueError or OverflowError if the command is a
          syntax error (OverflowError and ValueError can be produced by
          malformed literals).
        """
        return _maybe_compile(self.compiler, source, filename, symbol)
<|endoftext|>#!/usr/bin/env python3
#-------------------------------------------------------------------
# tarfile.py
#-------------------------------------------------------------------
# Copyright (C) 2002 Lars Gustaebel <lars@gustaebel.de>
# All rights reserved.
#
# Permission  is  hereby granted,  free  of charge,  to  any person
# obtaining a  copy of  this software  and associated documentation
# files  (the  "Software"),  to   deal  in  the  Software   without
# restriction,  including  without limitation  the  rights to  use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies  of  the  Software,  and to  permit  persons  to  whom the
# Software  is  furnished  to  do  so,  subject  to  the  following
# conditions:
#
# The above copyright  notice and this  permission notice shall  be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS  IS", WITHOUT WARRANTY OF ANY  KIND,
# EXPRESS OR IMPLIED, INCLUDING  BUT NOT LIMITED TO  THE WARRANTIES
# OF  MERCHANTABILITY,  FITNESS   FOR  A  PARTICULAR   PURPOSE  AND
# NONINFRINGEMENT.  IN  NO  EVENT SHALL  THE  AUTHORS  OR COPYRIGHT
# HOLDERS  BE LIABLE  FOR ANY  CLAIM, DAMAGES  OR OTHER  LIABILITY,
# WHETHER  IN AN  ACTION OF  CONTRACT, TORT  OR OTHERWISE,  ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
"""Read from and write to tar format archives.
"""

version     = "0.9.0"
__author__  = "Lars Gust\u00e4bel (lars@gustaebel.de)"
__credits__ = "Gustavo Niemeyer, Niels Gust\u00e4bel, Richard Townsend."

#---------
# Imports
#---------
from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings

try:
    import pwd
except ImportError:
    pwd = None
try:
    import grp
except ImportError:
    grp = None

# os.symlink on Windows prior to 6.0 raises NotImplementedError
symlink_exception = (AttributeError, NotImplementedError)
try:
    # OSError (winerror=1314) will be raised if the caller does not hold the
    # SeCreateSymbolicLinkPrivilege privilege
    symlink_exception += (OSError,)
except NameError:
    pass

# from tarfile import *
__all__ = ["TarFile", "TarInfo", "is_tarfile", "TarError", "ReadError",
           "CompressionError", "StreamError", "ExtractError", "HeaderError",
           "ENCODING", "USTAR_FORMAT", "GNU_FORMAT", "PAX_FORMAT",
           "DEFAULT_FORMAT", "open"]


#---------------------------------------------------------
# tar constants
#---------------------------------------------------------
NUL = b"\0"                     # the null character
BLOCKSIZE = 512                 # length of processing blocks
RECORDSIZE = BLOCKSIZE * 20     # length of records
GNU_MAGIC = b"ustar  \0"        # magic gnu tar string
POSIX_MAGIC = b"ustar\x0000"    # magic posix tar string

LENGTH_NAME = 100               # maximum length of a filename
LENGTH_LINK = 100               # maximum length of a linkname
LENGTH_PREFIX = 155             # maximum length of the prefix field

REGTYPE = b"0"                  # regular file
AREGTYPE = b"\0"                # regular file
LNKTYPE = b"1"                  # link (inside tarfile)
SYMTYPE = b"2"                  # symbolic link
CHRTYPE = b"3"                  # character special device
BLKTYPE = b"4"                  # block special device
DIRTYPE = b"5"                  # directory
FIFOTYPE = b"6"                 # fifo special device
CONTTYPE = b"7"                 # contiguous file

GNUTYPE_LONGNAME = b"L"         # GNU tar longname
GNUTYPE_LONGLINK = b"K"         # GNU tar longlink
GNUTYPE_SPARSE = b"S"           # GNU tar sparse file

XHDTYPE = b"x"                  # POSIX.1-2001 extended header
XGLTYPE = b"g"                  # POSIX.1-2001 global header
SOLARIS_XHDTYPE = b"X"          # Solaris extended header

USTAR_FORMAT = 0                # POSIX.1-1988 (ustar) format
GNU_FORMAT = 1                  # GNU tar format
PAX_FORMAT = 2                  # POSIX.1-2001 (pax) format
DEFAULT_FORMAT = PAX_FORMAT

#---------------------------------------------------------
# tarfile constants
#---------------------------------------------------------
# File types that tarfile supports:
SUPPORTED_TYPES = (REGTYPE, AREGTYPE, LNKTYPE,
                   SYMTYPE, DIRTYPE, FIFOTYPE,
                   CONTTYPE, CHRTYPE, BLKTYPE,
                   GNUTYPE_LONGNAME, GNUTYPE_LONGLINK,
                   GNUTYPE_SPARSE)

# File types that will be treated as a regular file.
REGULAR_TYPES = (REGTYPE, AREGTYPE,
                 CONTTYPE, GNUTYPE_SPARSE)

# File types that are part of the GNU tar format.
GNU_TYPES = (GNUTYPE_LONGNAME, GNUTYPE_LONGLINK,
             GNUTYPE_SPARSE)

# Fields from a pax header that override a TarInfo attribute.
PAX_FIELDS = ("path", "linkpath", "size", "mtime",
              "uid", "gid", "uname", "gname")

# Fields from a pax header that are affected by hdrcharset.
PAX_NAME_FIELDS = {"path", "linkpath", "uname", "gname"}

# Fields in a pax header that are numbers, all other fields
# are treated as strings.
PAX_NUMBER_FIELDS = {
    "atime": float,
    "ctime": float,
    "mtime": float,
    "uid": int,
    "gid": int,
    "size": int
}

#---------------------------------------------------------
# initialization
#---------------------------------------------------------
if os.name == "nt":
    ENCODING = "utf-8"
else:
    ENCODING = sys.getfilesystemencoding()

#---------------------------------------------------------
# Some useful functions
#---------------------------------------------------------

def stn(s, length, encoding, errors):
    """Convert a string to a null-terminated bytes object.
    """
    if s is None:
        raise ValueError("metadata cannot contain None")
    s = s.encode(encoding, errors)
    return s[:length] + (length - len(s)) * NUL

def nts(s, encoding, errors):
    """Convert a null-terminated bytes object to a string.
    """
    p = s.find(b"\0")
    if p != -1:
        s = s[:p]
    return s.decode(encoding, errors)

def nti(s):
    """Convert a number field to a python number.
    """
    # There are two possible encodings for a number field, see
    # itn() below.
    if s[0] in (0o200, 0o377):
        n = 0
        for i in range(len(s) - 1):
            n <<= 8
            n += s[i + 1]
        if s[0] == 0o377:
            n = -(256 ** (len(s) - 1) - n)
    else:
        try:
            s = nts(s, "ascii", "strict")
            n = int(s.strip() or "0", 8)
        except ValueError:
            raise InvalidHeaderError("invalid header")
    return n

def itn(n, digits=8, format=DEFAULT_FORMAT):
    """Convert a python number to a number field.
    """
    # POSIX 1003.1-1988 requires numbers to be encoded as a string of
    # octal digits followed by a null-byte, this allows values up to
    # (8**(digits-1))-1. GNU tar allows storing numbers greater than
    # that if necessary. A leading 0o200 or 0o377 byte indicate this
    # particular encoding, the following digits-1 bytes are a big-endian
    # base-256 representation. This allows values up to (256**(digits-1))-1.
    # A 0o200 byte indicates a positive number, a 0o377 byte a negative
    # number.
    original_n = n
    n = int(n)
    if 0 <= n < 8 ** (digits - 1):
        s = bytes("%0*o" % (digits - 1, n), "ascii") + NUL
    elif format == GNU_FORMAT and -256 ** (digits - 1) <= n < 256 ** (digits - 1):
        if n >= 0:
            s = bytearray([0o200])
        else:
            s = bytearray([0o377])
            n = 256 ** digits + n

        for i in range(digits - 1):
            s.insert(1, n & 0o377)
            n >>= 8
    else:
        raise ValueError("overflow in number field")

    return s

def calc_chksums(buf):
    """Calculate the checksum for a member's header by summing up all
       characters except for the chksum field which is treated as if
       it was filled with spaces. According to the GNU tar sources,
       some tars (Sun and NeXT) calculate chksum with signed char,
       which will be different if there are chars in the buffer with
       the high bit set. So we calculate two checksums, unsigned and
       signed.
    """
    unsigned_chksum = 256 + sum(struct.unpack_from("148B8x356B", buf))
    signed_chksum = 256 + sum(struct.unpack_from("148b8x356b", buf))
    return unsigned_chksum, signed_chksum

def copyfileobj(src, dst, length=None, exception=OSError, bufsize=None):
    """Copy length bytes from fileobj src to fileobj dst.
       If length is None, copy the entire content.
    """
    bufsize = bufsize or 16 * 1024
    if length == 0:
        return
    if length is None:
        shutil.copyfileobj(src, dst, bufsize)
        return

    blocks, remainder = divmod(length, bufsize)
    for b in range(blocks):
        buf = src.read(bufsize)
        if len(buf) < bufsize:
            raise exception("unexpected end of data")
        dst.write(buf)

    if remainder != 0:
        buf = src.read(remainder)
        if len(buf) < remainder:
            raise exception("unexpected end of data")
        dst.write(buf)
    return

def _safe_print(s):
    encoding = getattr(sys.stdout, 'encoding', None)
    if encoding is not None:
        s = s.encode(encoding, 'backslashreplace').decode(encoding)
    print(s, end=' ')


class TarError(Exception):
    """Base exception."""
    pass
class ExtractError(TarError):
    """General exception for extract errors."""
    pass
class ReadError(TarError):
    """Exception for unreadable tar archives."""
    pass
class CompressionError(TarError):
    """Exception for unavailable compression methods."""
    pass
class StreamError(TarError):
    """Exception for unsupported operations on stream-like TarFiles."""
    pass
class HeaderError(TarError):
    """Base exception for header errors."""
    pass
class EmptyHeaderError(HeaderError):
    """Exception for empty headers."""
    pass
class TruncatedHeaderError(HeaderError):
    """Exception for truncated headers."""
    pass
class EOFHeaderError(HeaderError):
    """Exception for end of file headers."""
    pass
class InvalidHeaderError(HeaderError):
    """Exception for invalid headers."""
    pass
class SubsequentHeaderError(HeaderError):
    """Exception for missing and invalid extended headers."""
    pass

#---------------------------
# internal stream interface
#---------------------------
class _LowLevelFile:
    """Low-level file object. Supports reading and writing.
       It is used instead of a regular file object for streaming
       access.
    """

    def __init__(self, name, mode):
        mode = {
            "r": os.O_RDONLY,
            "w": os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        }[mode]
        if hasattr(os, "O_BINARY"):
            mode |= os.O_BINARY
        self.fd = os.open(name, mode, 0o666)

    def close(self):
        os.close(self.fd)

    def read(self, size):
        return os.read(self.fd, size)

    def write(self, s):
        os.write(self.fd, s)

class _Stream:
    """Class that serves as an adapter between TarFile and
       a stream-like object.  The stream-like object only
       needs to have a read() or write() method and is accessed
       blockwise.  Use of gzip or bzip2 compression is possible.
       A stream-like object could be for example: sys.stdin,
       sys.stdout, a socket, a tape device etc.

       _Stream is intended to be used only internally.
    """

    def __init__(self, name, mode, comptype, fileobj, bufsize):
        """Construct a _Stream object.
        """
        self._extfileobj = True
        if fileobj is None:
            fileobj = _LowLevelFile(name, mode)
            self._extfileobj = False

        if comptype == '*':
            # Enable transparent compression detection for the
            # stream interface
            fileobj = _StreamProxy(fileobj)
            comptype = fileobj.getcomptype()

        self.name     = name or ""
        self.mode     = mode
        self.comptype = comptype
        self.fileobj  = fileobj
        self.bufsize  = bufsize
        self.buf      = b""
        self.pos      = 0
        self.closed   = False

        try:
            if comptype == "gz":
                try:
                    import zlib
                except ImportError:
                    raise CompressionError("zlib module is not available") from None
                self.zlib = zlib
                self.crc = zlib.crc32(b"")
                if mode == "r":
                    self._init_read_gz()
                    self.exception = zlib.error
                else:
                    self._init_write_gz()

            elif comptype == "bz2":
                try:
                    import bz2
                except ImportError:
                    raise CompressionError("bz2 module is not available") from None
                if mode == "r":
                    self.dbuf = b""
                    self.cmp = bz2.BZ2Decompressor()
                    self.exception = OSError
                else:
                    self.cmp = bz2.BZ2Compressor()

            elif comptype == "xz":
                try:
                    import lzma
                except ImportError:
                    raise CompressionError("lzma module is not available") from None
                if mode == "r":
                    self.dbuf = b""
                    self.cmp = lzma.LZMADecompressor()
                    self.exception = lzma.LZMAError
                else:
                    self.cmp = lzma.LZMACompressor()

            elif comptype != "tar":
                raise CompressionError("unknown compression type %r" % comptype)

        except:
            if not self._extfileobj:
                self.fileobj.close()
            self.closed = True
            raise

    def __del__(self):
        if hasattr(self, "closed") and not self.closed:
            self.close()

    def _init_write_gz(self):
        """Initialize for writing with gzip compression.
        """
        self.cmp = self.zlib.compressobj(9, self.zlib.DEFLATED,
                                            -self.zlib.MAX_WBITS,
                                            self.zlib.DEF_MEM_LEVEL,
                                            0)
        timestamp = struct.pack("<L", int(time.time()))
        self.__write(b"\037\213\010\010" + timestamp + b"\002\377")
        if self.name.endswith(".gz"):
            self.name = self.name[:-3]
        # Honor "directory components removed" from RFC1952
        self.name = os.path.basename(self.name)
        # RFC1952 says we must use ISO-8859-1 for the FNAME field.
        self.__write(self.name.encode("iso-8859-1", "replace") + NUL)

    def write(self, s):
        """Write string s to the stream.
        """
        if self.comptype == "gz":
            self.crc = self.zlib.crc32(s, self.crc)
        self.pos += len(s)
        if self.comptype != "tar":
            s = self.cmp.compress(s)
        self.__write(s)

    def __write(self, s):
        """Write string s to the stream if a whole new block
           is ready to be written.
        """
        self.buf += s
        while len(self.buf) > self.bufsize:
            self.fileobj.write(self.buf[:self.bufsize])
            self.buf = self.buf[self.bufsize:]

    def close(self):
        """Close the _Stream object. No operation should be
           done on it afterwards.
        """
        if self.closed:
            return

        self.closed = True
        try:
            if self.mode == "w" and self.comptype != "tar":
                self.buf += self.cmp.flush()

            if self.mode == "w" and self.buf:
                self.fileobj.write(self.buf)
                self.buf = b""
                if self.comptype == "gz":
                    self.fileobj.write(struct.pack("<L", self.crc))
                    self.fileobj.write(struct.pack("<L", self.pos & 0xffffFFFF))
        finally:
            if not self._extfileobj:
                self.fileobj.close()

    def _init_read_gz(self):
        """Initialize for reading a gzip compressed fileobj.
        """
        self.cmp = self.zlib.decompressobj(-self.zlib.MAX_WBITS)
        self.dbuf = b""

        # taken from gzip.GzipFile with some alterations
        if self.__read(2) != b"\037\213":
            raise ReadError("not a gzip file")
        if self.__read(1) != b"\010":
            raise CompressionError("unsupported compression method")

        flag = ord(self.__read(1))
        self.__read(6)

        if flag & 4:
            xlen = ord(self.__read(1)) + 256 * ord(self.__read(1))
            self.read(xlen)
        if flag & 8:
            while True:
                s = self.__read(1)
                if not s or s == NUL:
                    break
        if flag & 16:
            while True:
                s = self.__read(1)
                if not s or s == NUL:
                    break
        if flag & 2:
            self.__read(2)

    def tell(self):
        """Return the stream's file pointer position.
        """
        return self.pos

    def seek(self, pos=0):
        """Set the stream's file pointer to pos. Negative seeking
           is forbidden.
        """
        if pos - self.pos >= 0:
            blocks, remainder = divmod(pos - self.pos, self.bufsize)
            for i in range(blocks):
                self.read(self.bufsize)
            self.read(remainder)
        else:
            raise StreamError("seeking backwards is not allowed")
        return self.pos

    def read(self, size):
        """Return the next size number of bytes from the stream."""
        assert size is not None
        buf = self._read(size)
        self.pos += len(buf)
        return buf

    def _read(self, size):
        """Return size bytes from the stream.
        """
        if self.comptype == "tar":
            return self.__read(size)

        c = len(self.dbuf)
        t = [self.dbuf]
        while c < size:
            # Skip underlying buffer to avoid unaligned double buffering.
            if self.buf:
                buf = self.buf
                self.buf = b""
            else:
                buf = self.fileobj.read(self.bufsize)
                if not buf:
                    break
            try:
                buf = self.cmp.decompress(buf)
            except self.exception as e:
                raise ReadError("invalid compressed data") from e
            t.append(buf)
            c += len(buf)
        t = b"".join(t)
        self.dbuf = t[size:]
        return t[:size]

    def __read(self, size):
        """Return size bytes from stream. If internal buffer is empty,
           read another block from the stream.
        """
        c = len(self.buf)
        t = [self.buf]
        while c < size:
            buf = self.fileobj.read(self.bufsize)
            if not buf:
                break
            t.append(buf)
            c += len(buf)
        t = b"".join(t)
        self.buf = t[size:]
        return t[:size]
# class _Stream

class _StreamProxy(object):
    """Small proxy class that enables transparent compression
       detection for the Stream interface (mode 'r|*').
    """

    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.buf = self.fileobj.read(BLOCKSIZE)

    def read(self, size):
        self.read = self.fileobj.read
        return self.buf

    def getcomptype(self):
        if self.buf.startswith(b"\x1f\x8b\x08"):
            return "gz"
        elif self.buf[0:3] == b"BZh" and self.buf[4:10] == b"1AY&SY":
            return "bz2"
        elif self.buf.startswith((b"\x5d\x00\x00\x80", b"\xfd7zXZ")):
            return "xz"
        else:
            return "tar"

    def close(self):
        self.fileobj.close()
# class StreamProxy

#------------------------
# Extraction file object
#------------------------
class _FileInFile(object):
    """A thin wrapper around an existing file object that
       provides a part of its data as an individual file
       object.
    """

    def __init__(self, fileobj, offset, size, blockinfo=None):
        self.fileobj = fileobj
        self.offset = offset
        self.size = size
        self.position = 0
        self.name = getattr(fileobj, "name", None)
        self.closed = False

        if blockinfo is None:
            blockinfo = [(0, size)]

        # Construct a map with data and zero blocks.
        self.map_index = 0
        self.map = []
        lastpos = 0
        realpos = self.offset
        for offset, size in blockinfo:
            if offset > lastpos:
                self.map.append((False, lastpos, offset, None))
            self.map.append((True, offset, offset + size, realpos))
            realpos += size
            lastpos = offset + size
        if lastpos < self.size:
            self.map.append((False, lastpos, self.size, None))

    def flush(self):
        pass

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return self.fileobj.seekable()

    def tell(self):
        """Return the current file position.
        """
        return self.position

    def seek(self, position, whence=io.SEEK_SET):
        """Seek to a position in the file.
        """
        if whence == io.SEEK_SET:
            self.position = min(max(position, 0), self.size)
        elif whence == io.SEEK_CUR:
            if position < 0:
                self.position = max(self.position + position, 0)
            else:
                self.position = min(self.position + position, self.size)
        elif whence == io.SEEK_END:
            self.position = max(min(self.size + position, self.size), 0)
        else:
            raise ValueError("Invalid argument")
        return self.position

    def read(self, size=None):
        """Read data from the file.
        """
        if size is None:
            size = self.size - self.position
        else:
            size = min(size, self.size - self.position)

        buf = b""
        while size > 0:
            while True:
                data, start, stop, offset = self.map[self.map_index]
                if start <= self.position < stop:
                    break
                else:
                    self.map_index += 1
                    if self.map_index == len(self.map):
                        self.map_index = 0
            length = min(size, stop - self.position)
            if data:
                self.fileobj.seek(offset + (self.position - start))
                b = self.fileobj.read(length)
                if len(b) != length:
                    raise ReadError("unexpected end of data")
                buf += b
            else:
                buf += NUL * length
            size -= length
            self.position += length
        return buf

    def readinto(self, b):
        buf = self.read(len(b))
        b[:len(buf)] = buf
        return len(buf)

    def close(self):
        self.closed = True
#class _FileInFile

class ExFileObject(io.BufferedReader):

    def __init__(self, tarfile, tarinfo):
        fileobj = _FileInFile(tarfile.fileobj, tarinfo.offset_data,
                tarinfo.size, tarinfo.sparse)
        super().__init__(fileobj)
#class ExFileObject


#-----------------------------
# extraction filters (PEP 706)
#-----------------------------

class FilterError(TarError):
    pass

class AbsolutePathError(FilterError):
    def __init__(self, tarinfo):
        self.tarinfo = tarinfo
        super().__init__(f'member {tarinfo.name!r} has an absolute path')

class OutsideDestinationError(FilterError):
    def __init__(self, tarinfo, path):
        self.tarinfo = tarinfo
        self._path = path
        super().__init__(f'{tarinfo.name!r} would be extracted to {path!r}, '
                         + 'which is outside the destination')

class SpecialFileError(FilterError):
    def __init__(self, tarinfo):
        self.tarinfo = tarinfo
        super().__init__(f'{tarinfo.name!r} is a special file')

class AbsoluteLinkError(FilterError):
    def __init__(self, tarinfo):
        self.tarinfo = tarinfo
        super().__init__(f'{tarinfo.name!r} is a symlink to an absolute path')

class LinkOutsideDestinationError(FilterError):
    def __init__(self, tarinfo, path):
        self.tarinfo = tarinfo
        self._path = path
        super().__init__(f'{tarinfo.name!r} would link to {path!r}, '
                         + 'which is outside the destination')

def _get_filtered_attrs(member, dest_path, for_data=True):
    new_attrs = {}
    name = member.name
    dest_path = os.path.realpath(dest_path)
    # Strip leading / (tar's directory separator) from filenames.
    # Include os.sep (target OS directory separator) as well.
    if name.startswith(('/', os.sep)):
        name = new_attrs['name'] = member.path.lstrip('/' + os.sep)
    if os.path.isabs(name):
        # Path is absolute even after stripping.
        # For example, 'C:/foo' on Windows.
        raise AbsolutePathError(member)
    # Ensure we stay in the destination
    target_path = os.path.realpath(os.path.join(dest_path, name))
    if os.path.commonpath([target_path, dest_path]) != dest_path:
        raise OutsideDestinationError(member, target_path)
    # Limit permissions (no high bits, and go-w)
    mode = member.mode
    if mode is not None:
        # Strip high bits & group/other write bits
        mode = mode & 0o755
        if for_data:
            # For data, handle permissions & file types
            if member.isreg() or member.islnk():
                if not mode & 0o100:
                    # Clear executable bits if not executable by user
                    mode &= ~0o111
                # Ensure owner can read & write
                mode |= 0o600
            elif member.isdir() or member.issym():
                # Ignore mode for directories & symlinks
                mode = None
            else:
                # Reject special files
                raise SpecialFileError(member)
        if mode != member.mode:
            new_attrs['mode'] = mode
    if for_data:
        # Ignore ownership for 'data'
        if member.uid is not None:
            new_attrs['uid'] = None
        if member.gid is not None:
            new_attrs['gid'] = None
        if member.uname is not None:
            new_attrs['uname'] = None
        if member.gname is not None:
            new_attrs['gname'] = None
        # Check link destination for 'data'
        if member.islnk() or member.issym():
            if os.path.isabs(member.linkname):
                raise AbsoluteLinkError(member)
            target_path = os.path.realpath(os.path.join(dest_path, member.linkname))
            if os.path.commonpath([target_path, dest_path]) != dest_path:
                raise LinkOutsideDestinationError(member, target_path)
    return new_attrs

def fully_trusted_filter(member, dest_path):
    return member

def tar_filter(member, dest_path):
    new_attrs = _get_filtered_attrs(member, dest_path, False)
    if new_attrs:
        return member.replace(**new_attrs, deep=False)
    return member

def data_filter(member, dest_path):
    new_attrs = _get_filtered_attrs(member, dest_path, True)
    if new_attrs:
        return member.replace(**new_attrs, deep=False)
    return member

_NAMED_FILTERS = {
    "fully_trusted": fully_trusted_filter,
    "tar": tar_filter,
    "data": data_filter,
}

#------------------
# Exported Classes
#------------------

# Sentinel for replace() defaults, meaning "don't change the attribute"
_KEEP = object()

# Header length is digits followed by a space.
_header_length_prefix_re = re.compile(br"([0-9]{1,20}) ")

class TarInfo(object):
    """Informational class which holds the details about an
       archive member given by a tar header block.
       TarInfo objects are returned by TarFile.getmember(),
       TarFile.getmembers() and TarFile.gettarinfo() and are
       usually created internally.
    """

    __slots__ = dict(
        name = 'Name of the archive member.',
        mode = 'Permission bits.',
        uid = 'User ID of the user who originally stored this member.',
        gid = 'Group ID of the user who originally stored this member.',
        size = 'Size in bytes.',
        mtime = 'Time of last modification.',
        chksum = 'Header checksum.',
        type = ('File type. type is usually one of these constants: '
                'REGTYPE, AREGTYPE, LNKTYPE, SYMTYPE, DIRTYPE, FIFOTYPE, '
                'CONTTYPE, CHRTYPE, BLKTYPE, GNUTYPE_SPARSE.'),
        linkname = ('Name of the target file name, which is only present '
                    'in TarInfo objects of type LNKTYPE and SYMTYPE.'),
        uname = 'User name.',
        gname = 'Group name.',
        devmajor = 'Device major number.',
        devminor = 'Device minor number.',
        offset = 'The tar header starts here.',
        offset_data = "The file's data starts here.",
        pax_headers = ('A dictionary containing key-value pairs of an '
                       'associated pax extended header.'),
        sparse = 'Sparse member information.',
        tarfile = None,
        _sparse_structs = None,
        _link_target = None,
        )

    def __init__(self, name=""):
        """Construct a TarInfo object. name is the optional name
           of the member.
        """
        self.name = name        # member name
        self.mode = 0o644       # file permissions
        self.uid = 0            # user id
        self.gid = 0            # group id
        self.size = 0           # file size
        self.mtime = 0          # modification time
        self.chksum = 0         # header checksum
        self.type = REGTYPE     # member type
        self.linkname = ""      # link name
        self.uname = ""         # user name
        self.gname = ""         # group name
        self.devmajor = 0       # device major number
        self.devminor = 0       # device minor number

        self.offset = 0         # the tar header starts here
        self.offset_data = 0    # the file's data starts here

        self.sparse = None      # sparse member information
        self.pax_headers = {}   # pax header information

    @property
    def path(self):
        'In pax headers, "name" is called "path".'
        return self.name

    @path.setter
    def path(self, name):
        self.name = name

    @property
    def linkpath(self):
        'In pax headers, "linkname" is called "linkpath".'
        return self.linkname

    @linkpath.setter
    def linkpath(self, linkname):
        self.linkname = linkname

    def __repr__(self):
        return "<%s %r at %#x>" % (self.__class__.__name__,self.name,id(self))

    def replace(self, *,
                name=_KEEP, mtime=_KEEP, mode=_KEEP, linkname=_KEEP,
                uid=_KEEP, gid=_KEEP, uname=_KEEP, gname=_KEEP,
                deep=True, _KEEP=_KEEP):
        """Return a deep copy of self with the given attributes replaced.
        """
        if deep:
            result = copy.deepcopy(self)
        else:
            result = copy.copy(self)
        if name is not _KEEP:
            result.name = name
        if mtime is not _KEEP:
            result.mtime = mtime
        if mode is not _KEEP:
            result.mode = mode
        if linkname is not _KEEP:
            result.linkname = linkname
        if uid is not _KEEP:
            result.uid = uid
        if gid is not _KEEP:
            result.gid = gid
        if uname is not _KEEP:
            result.uname = uname
        if gname is not _KEEP:
            result.gname = gname
        return result

    def get_info(self):
        """Return the TarInfo's attributes as a dictionary.
        """
        if self.mode is None:
            mode = None
        else:
            mode = self.mode & 0o7777
        info = {
            "name":     self.name,
            "mode":     mode,
            "uid":      self.uid,
            "gid":      self.gid,
            "size":     self.size,
            "mtime":    self.mtime,
            "chksum":   self.chksum,
            "type":     self.type,
            "linkname": self.linkname,
            "uname":    self.uname,
            "gname":    self.gname,
            "devmajor": self.devmajor,
            "devminor": self.devminor
        }

        if info["type"] == DIRTYPE and not info["name"].endswith("/"):
            info["name"] += "/"

        return info

    def tobuf(self, format=DEFAULT_FORMAT, encoding=ENCODING, errors="surrogateescape"):
        """Return a tar header as a string of 512 byte blocks.
        """
        info = self.get_info()
        for name, value in info.items():
            if value is None:
                raise ValueError("%s may not be None" % name)

        if format == USTAR_FORMAT:
            return self.create_ustar_header(info, encoding, errors)
        elif format == GNU_FORMAT:
            return self.create_gnu_header(info, encoding, errors)
        elif format == PAX_FORMAT:
            return self.create_pax_header(info, encoding)
        else:
            raise ValueError("invalid format")

    def create_ustar_header(self, info, encoding, errors):
        """Return the object as a ustar header block.
        """
        info["magic"] = POSIX_MAGIC

        if len(info["linkname"].encode(encoding, errors)) > LENGTH_LINK:
            raise ValueError("linkname is too long")

        if len(info["name"].encode(encoding, errors)) > LENGTH_NAME:
            info["prefix"], info["name"] = self._posix_split_name(info["name"], encoding, errors)

        return self._create_header(info, USTAR_FORMAT, encoding, errors)

    def create_gnu_header(self, info, encoding, errors):
        """Return the object as a GNU header block sequence.
        """
        info["magic"] = GNU_MAGIC

        buf = b""
        if len(info["linkname"].encode(encoding, errors)) > LENGTH_LINK:
            buf += self._create_gnu_long_header(info["linkname"], GNUTYPE_LONGLINK, encoding, errors)

        if len(info["name"].encode(encoding, errors)) > LENGTH_NAME:
            buf += self._create_gnu_long_header(info["name"], GNUTYPE_LONGNAME, encoding, errors)

        return buf + self._create_header(info, GNU_FORMAT, encoding, errors)

    def create_pax_header(self, info, encoding):
        """Return the object as a ustar header block. If it cannot be
           represented this way, prepend a pax extended header sequence
           with supplement information.
        """
        info["magic"] = POSIX_MAGIC
        pax_headers = self.pax_headers.copy()

        # Test string fields for values that exceed the field length or cannot
        # be represented in ASCII encoding.
        for name, hname, length in (
                ("name", "path", LENGTH_NAME), ("linkname", "linkpath", LENGTH_LINK),
                ("uname", "uname", 32), ("gname", "gname", 32)):

            if hname in pax_headers:
                # The pax header has priority.
                continue

            # Try to encode the string as ASCII.
            try:
                info[name].encode("ascii", "strict")
            except UnicodeEncodeError:
                pax_headers[hname] = info[name]
                continue

            if len(info[name]) > length:
                pax_headers[hname] = info[name]

        # Test number fields for values that exceed the field limit or values
        # that like to be stored as float.
        for name, digits in (("uid", 8), ("gid", 8), ("size", 12), ("mtime", 12)):
            needs_pax = False

            val = info[name]
            val_is_float = isinstance(val, float)
            val_int = round(val) if val_is_float else val
            if not 0 <= val_int < 8 ** (digits - 1):
                # Avoid overflow.
                info[name] = 0
                needs_pax = True
            elif val_is_float:
                # Put rounded value in ustar header, and full
                # precision value in pax header.
                info[name] = val_int
                needs_pax = True

            # The existing pax header has priority.
            if needs_pax and name not in pax_headers:
                pax_headers[name] = str(val)

        # Create a pax extended header if necessary.
        if pax_headers:
            buf = self._create_pax_generic_header(pax_headers, XHDTYPE, encoding)
        else:
            buf = b""

        return buf + self._create_header(info, USTAR_FORMAT, "ascii", "replace")

    @classmethod
    def create_pax_global_header(cls, pax_headers):
        """Return the object as a pax global header block sequence.
        """
        return cls._create_pax_generic_header(pax_headers, XGLTYPE, "utf-8")

    def _posix_split_name(self, name, encoding, errors):
        """Split a name longer than 100 chars into a prefix
           and a name part.
        """
        components = name.split("/")
        for i in range(1, len(components)):
            prefix = "/".join(components[:i])
            name = "/".join(components[i:])
            if len(prefix.encode(encoding, errors)) <= LENGTH_PREFIX and \
                    len(name.encode(encoding, errors)) <= LENGTH_NAME:
                break
        else:
            raise ValueError("name is too long")

        return prefix, name

    @staticmethod
    def _create_header(info, format, encoding, errors):
        """Return a header block. info is a dictionary with file
           information, format must be one of the *_FORMAT constants.
        """
        has_device_fields = info.get("type") in (CHRTYPE, BLKTYPE)
        if has_device_fields:
            devmajor = itn(info.get("devmajor", 0), 8, format)
            devminor = itn(info.get("devminor", 0), 8, format)
        else:
            devmajor = stn("", 8, encoding, errors)
            devminor = stn("", 8, encoding, errors)

        # None values in metadata should cause ValueError.
        # itn()/stn() do this for all fields except type.
        filetype = info.get("type", REGTYPE)
        if filetype is None:
            raise ValueError("TarInfo.type must not be None")

        parts = [
            stn(info.get("name", ""), 100, encoding, errors),
            itn(info.get("mode", 0) & 0o7777, 8, format),
            itn(info.get("uid", 0), 8, format),
            itn(info.get("gid", 0), 8, format),
            itn(info.get("size", 0), 12, format),
            itn(info.get("mtime", 0), 12, format),
            b"        ", # checksum field
            filetype,
            stn(info.get("linkname", ""), 100, encoding, errors),
            info.get("magic", POSIX_MAGIC),
            stn(info.get("uname", ""), 32, encoding, errors),
            stn(info.get("gname", ""), 32, encoding, errors),
            devmajor,
            devminor,
            stn(info.get("prefix", ""), 155, encoding, errors)
        ]

        buf = struct.pack("%ds" % BLOCKSIZE, b"".join(parts))
        chksum = calc_chksums(buf[-BLOCKSIZE:])[0]
        buf = buf[:-364] + bytes("%06o\0" % chksum, "ascii") + buf[-357:]
        return buf

    @staticmethod
    def _create_payload(payload):
        """Return the string payload filled with zero bytes
           up to the next 512 byte border.
        """
        blocks, remainder = divmod(len(payload), BLOCKSIZE)
        if remainder > 0:
            payload += (BLOCKSIZE - remainder) * NUL
        return payload

    @classmethod
    def _create_gnu_long_header(cls, name, type, encoding, errors):
        """Return a GNUTYPE_LONGNAME or GNUTYPE_LONGLINK sequence
           for name.
        """
        name = name.encode(encoding, errors) + NUL

        info = {}
        info["name"] = "././@LongLink"
        info["type"] = type
        info["size"] = len(name)
        info["magic"] = GNU_MAGIC

        # create extended header + name blocks.
        return cls._create_header(info, USTAR_FORMAT, encoding, errors) + \
                cls._create_payload(name)

    @classmethod
    def _create_pax_generic_header(cls, pax_headers, type, encoding):
        """Return a POSIX.1-2008 extended or global header sequence
           that contains a list of keyword, value pairs. The values
           must be strings.
        """
        # Check if one of the fields contains surrogate characters and thereby
        # forces hdrcharset=BINARY, see _proc_pax() for more information.
        binary = False
        for keyword, value in pax_headers.items():
            try:
                value.encode("utf-8", "strict")
            except UnicodeEncodeError:
                binary = True
                break

        records = b""
        if binary:
            # Put the hdrcharset field at the beginning of the header.
            records += b"21 hdrcharset=BINARY\n"

        for keyword, value in pax_headers.items():
            keyword = keyword.encode("utf-8")
            if binary:
                # Try to restore the original byte representation of `value'.
                # Needless to say, that the encoding must match the string.
                value = value.encode(encoding, "surrogateescape")
            else:
                value = value.encode("utf-8")

            l = len(keyword) + len(value) + 3   # ' ' + '=' + '\n'
            n = p = 0
            while True:
                n = l + len(str(p))
                if n == p:
                    break
                p = n
            records += bytes(str(p), "ascii") + b" " + keyword + b"=" + value + b"\n"

        # We use a hardcoded "././@PaxHeader" name like star does
        # instead of the one that POSIX recommends.
        info = {}
        info["name"] = "././@PaxHeader"
        info["type"] = type
        info["size"] = len(records)
        info["magic"] = POSIX_MAGIC

        # Create pax header + record blocks.
        return cls._create_header(info, USTAR_FORMAT, "ascii", "replace") + \
                cls._create_payload(records)

    @classmethod
    def frombuf(cls, buf, encoding, errors):
        """Construct a TarInfo object from a 512 byte bytes object.

        To support the old v7 tar format AREGTYPE headers are
        transformed to DIRTYPE headers if their name ends in '/'.
        """
        return cls._frombuf(buf, encoding, errors)

    @classmethod
    def _frombuf(cls, buf, encoding, errors, *, dircheck=True):
        """Construct a TarInfo object from a 512 byte bytes object.

        If ``dircheck`` is set to ``True`` then ``AREGTYPE`` headers will
        be normalized to ``DIRTYPE`` if the name ends in a trailing slash.
        ``dircheck`` must be set to ``False`` if this function is called
        on a follow-up header such as ``GNUTYPE_LONGNAME``.
        """
        if len(buf) == 0:
            raise EmptyHeaderError("empty header")
        if len(buf) != BLOCKSIZE:
            raise TruncatedHeaderError("truncated header")
        if buf.count(NUL) == BLOCKSIZE:
            raise EOFHeaderError("end of file header")

        chksum = nti(buf[148:156])
        if chksum not in calc_chksums(buf):
            raise InvalidHeaderError("bad checksum")

        obj = cls()
        obj.name = nts(buf[0:100], encoding, errors)
        obj.mode = nti(buf[100:108])
        obj.uid = nti(buf[108:116])
        obj.gid = nti(buf[116:124])
        obj.size = nti(buf[124:136])
        obj.mtime = nti(buf[136:148])
        obj.chksum = chksum
        obj.type = buf[156:157]
        obj.linkname = nts(buf[157:257], encoding, errors)
        obj.uname = nts(buf[265:297], encoding, errors)
        obj.gname = nts(buf[297:329], encoding, errors)
        obj.devmajor = nti(buf[329:337])
        obj.devminor = nti(buf[337:345])
        prefix = nts(buf[345:500], encoding, errors)

        # Old V7 tar format represents a directory as a regular
        # file with a trailing slash.
        if dircheck and obj.type == AREGTYPE and obj.name.endswith("/"):
            obj.type = DIRTYPE

        # The old GNU sparse format occupies some of the unused
        # space in the buffer for up to 4 sparse structures.
        # Save them for later processing in _proc_sparse().
        if obj.type == GNUTYPE_SPARSE:
            pos = 386
            structs = []
            for i in range(4):
                try:
                    offset = nti(buf[pos:pos + 12])
                    numbytes = nti(buf[pos + 12:pos + 24])
                except ValueError:
                    break
                structs.append((offset, numbytes))
                pos += 24
            isextended = bool(buf[482])
            origsize = nti(buf[483:495])
            obj._sparse_structs = (structs, isextended, origsize)

        # Remove redundant slashes from directories.
        if obj.isdir():
            obj.name = obj.name.rstrip("/")

        # Reconstruct a ustar longname.
        if prefix and obj.type not in GNU_TYPES:
            obj.name = prefix + "/" + obj.name
        return obj

    @classmethod
    def fromtarfile(cls, tarfile):
        """Return the next TarInfo object from TarFile object
           tarfile.
        """
        return cls._fromtarfile(tarfile)

    @classmethod
    def _fromtarfile(cls, tarfile, *, dircheck=True):
        """
        See dircheck documentation in _frombuf().
        """
        buf = tarfile.fileobj.read(BLOCKSIZE)
        obj = cls._frombuf(buf, tarfile.encoding, tarfile.errors, dircheck=dircheck)
        obj.offset = tarfile.fileobj.tell() - BLOCKSIZE
        return obj._proc_member(tarfile)

    #--------------------------------------------------------------------------
    # The following are methods that are called depending on the type of a
    # member. The entry point is _proc_member() which can be overridden in a
    # subclass to add custom _proc_*() methods. A _proc_*() method MUST
    # implement the following
    # operations:
    # 1. Set self.offset_data to the position where the data blocks begin,
    #    if there is data that follows.
    # 2. Set tarfile.offset to the position where the next member's header will
    #    begin.
    # 3. Return self or another valid TarInfo object.
    def _proc_member(self, tarfile):
        """Choose the right processing method depending on
           the type and call it.
        """
        if self.type in (GNUTYPE_LONGNAME, GNUTYPE_LONGLINK):
            return self._proc_gnulong(tarfile)
        elif self.type == GNUTYPE_SPARSE:
            return self._proc_sparse(tarfile)
        elif self.type in (XHDTYPE, XGLTYPE, SOLARIS_XHDTYPE):
            return self._proc_pax(tarfile)
        else:
            return self._proc_builtin(tarfile)

    def _proc_builtin(self, tarfile):
        """Process a builtin type or an unknown type which
           will be treated as a regular file.
        """
        self.offset_data = tarfile.fileobj.tell()
        offset = self.offset_data
        if self.isreg() or self.type not in SUPPORTED_TYPES:
            # Skip the following data blocks.
            offset += self._block(self.size)
        tarfile.offset = offset

        # Patch the TarInfo object with saved global
        # header information.
        self._apply_pax_info(tarfile.pax_headers, tarfile.encoding, tarfile.errors)

        # Remove redundant slashes from directories. This is to be consistent
        # with frombuf().
        if self.isdir():
            self.name = self.name.rstrip("/")

        return self

    def _proc_gnulong(self, tarfile):
        """Process the blocks that hold a GNU longname
           or longlink member.
        """
        buf = tarfile.fileobj.read(self._block(self.size))

        # Fetch the next header and process it.
        try:
            next = self._fromtarfile(tarfile, dircheck=False)
        except HeaderError as e:
            raise SubsequentHeaderError(str(e)) from None

        # Patch the TarInfo object from the next header with
        # the longname information.
        next.offset = self.offset
        if self.type == GNUTYPE_LONGNAME:
            next.name = nts(buf, tarfile.encoding, tarfile.errors)
        elif self.type == GNUTYPE_LONGLINK:
            next.linkname = nts(buf, tarfile.encoding, tarfile.errors)

        # Remove redundant slashes from directories. This is to be consistent
        # with frombuf().
        if next.isdir():
            next.name = next.name.removesuffix("/")

        return next

    def _proc_sparse(self, tarfile):
        """Process a GNU sparse header plus extra headers.
        """
        # We already collected some sparse structures in frombuf().
        structs, isextended, origsize = self._sparse_structs
        del self._sparse_structs

        # Collect sparse structures from extended header blocks.
        while isextended:
            buf = tarfile.fileobj.read(BLOCKSIZE)
            pos = 0
            for i in range(21):
                try:
                    offset = nti(buf[pos:pos + 12])
                    numbytes = nti(buf[pos + 12:pos + 24])
                except ValueError:
                    break
                if offset and numbytes:
                    structs.append((offset, numbytes))
                pos += 24
            isextended = bool(buf[504])
        self.sparse = structs

        self.offset_data = tarfile.fileobj.tell()
        tarfile.offset = self.offset_data + self._block(self.size)
        self.size = origsize
        return self

    def _proc_pax(self, tarfile):
        """Process an extended or global header as described in
           POSIX.1-2008.
        """
        # Read the header information.
        buf = tarfile.fileobj.read(self._block(self.size))

        # A pax header stores supplemental information for either
        # the following file (extended) or all following files
        # (global).
        if self.type == XGLTYPE:
            pax_headers = tarfile.pax_headers
        else:
            pax_headers = tarfile.pax_headers.copy()

        # Parse pax header information. A record looks like that:
        # "%d %s=%s\n" % (length, keyword, value). length is the size
        # of the complete record including the length field itself and
        # the newline.
        pos = 0
        encoding = None
        raw_headers = []
        while len(buf) > pos and buf[pos] != 0x00:
            if not (match := _header_length_prefix_re.match(buf, pos)):
                raise InvalidHeaderError("invalid header")
            try:
                length = int(match.group(1))
            except ValueError:
                raise InvalidHeaderError("invalid header")
            # Headers must be at least 5 bytes, shortest being '5 x=\n'.
            # Value is allowed to be empty.
            if length < 5:
                raise InvalidHeaderError("invalid header")
            if pos + length > len(buf):
                raise InvalidHeaderError("invalid header")

            header_value_end_offset = match.start(1) + length - 1  # Last byte of the header
            keyword_and_value = buf[match.end(1) + 1:header_value_end_offset]
            raw_keyword, equals, raw_value = keyword_and_value.partition(b"=")

            # Check the framing of the header. The last character must be '\n' (0x0A)
            if not raw_keyword or equals != b"=" or buf[header_value_end_offset] != 0x0A:
                raise InvalidHeaderError("invalid header")
            raw_headers.append((length, raw_keyword, raw_value))

            # Check if the pax header contains a hdrcharset field. This tells us
            # the encoding of the path, linkpath, uname and gname fields. Normally,
            # these fields are UTF-8 encoded but since POSIX.1-2008 tar
            # implementations are allowed to store them as raw binary strings if
            # the translation to UTF-8 fails. For the time being, we don't care about
            # anything other than "BINARY". The only other value that is currently
            # allowed by the standard is "ISO-IR 10646 2000 UTF-8" in other words UTF-8.
            # Note that we only follow the initial 'hdrcharset' setting to preserve
            # the initial behavior of the 'tarfile' module.
            if raw_keyword == b"hdrcharset" and encoding is None:
                if raw_value == b"BINARY":
                    encoding = tarfile.encoding
                else:  # This branch ensures only the first 'hdrcharset' header is used.
                    encoding = "utf-8"

            pos += length

        # If no explicit hdrcharset is set, we use UTF-8 as a default.
        if encoding is None:
            encoding = "utf-8"

        # After parsing the raw headers we can decode them to text.
        for length, raw_keyword, raw_value in raw_headers:
            # Normally, we could just use "utf-8" as the encoding and "strict"
            # as the error handler, but we better not take the risk. For
            # example, GNU tar <= 1.23 is known to store filenames it cannot
            # translate to UTF-8 as raw strings (unfortunately without a
            # hdrcharset=BINARY header).
            # We first try the strict standard encoding, and if that fails we
            # fall back on the user's encoding and error handler.
            keyword = self._decode_pax_field(raw_keyword, "utf-8", "utf-8",
                    tarfile.errors)
            if keyword in PAX_NAME_FIELDS:
                value = self._decode_pax_field(raw_value, encoding, tarfile.encoding,
                        tarfile.errors)
            else:
                value = self._decode_pax_field(raw_value, "utf-8", "utf-8",
                        tarfile.errors)

            pax_headers[keyword] = value

        # Fetch the next header.
        try:
            next = self._fromtarfile(tarfile, dircheck=False)
        except HeaderError as e:
            raise SubsequentHeaderError(str(e)) from None

        # Process GNU sparse information.
        if "GNU.sparse.map" in pax_headers:
            # GNU extended sparse format version 0.1.
            self._proc_gnusparse_01(next, pax_headers)

        elif "GNU.sparse.size" in pax_headers:
            # GNU extended sparse format version 0.0.
            self._proc_gnusparse_00(next, raw_headers)

        elif pax_headers.get("GNU.sparse.major") == "1" and pax_headers.get("GNU.sparse.minor") == "0":
            # GNU extended sparse format version 1.0.
            self._proc_gnusparse_10(next, pax_headers, tarfile)

        if self.type in (XHDTYPE, SOLARIS_XHDTYPE):
            # Patch the TarInfo object with the extended header info.
            next._apply_pax_info(pax_headers, tarfile.encoding, tarfile.errors)
            next.offset = self.offset

            if "size" in pax_headers:
                # If the extended header replaces the size field,
                # we need to recalculate the offset where the next
                # header starts.
                offset = next.offset_data
                if next.isreg() or next.type not in SUPPORTED_TYPES:
                    offset += next._block(next.size)
                tarfile.offset = offset

        return next

    def _proc_gnusparse_00(self, next, raw_headers):
        """Process a GNU tar extended sparse header, version 0.0.
        """
        offsets = []
        numbytes = []
        for _, keyword, value in raw_headers:
            if keyword == b"GNU.sparse.offset":
                try:
                    offsets.append(int(value.decode()))
                except ValueError:
                    raise InvalidHeaderError("invalid header")

            elif keyword == b"GNU.sparse.numbytes":
                try:
                    numbytes.append(int(value.decode()))
                except ValueError:
                    raise InvalidHeaderError("invalid header")

        next.sparse = list(zip(offsets, numbytes))

    def _proc_gnusparse_01(self, next, pax_headers):
        """Process a GNU tar extended sparse header, version 0.1.
        """
        sparse = [int(x) for x in pax_headers["GNU.sparse.map"].split(",")]
        next.sparse = list(zip(sparse[::2], sparse[1::2]))

    def _proc_gnusparse_10(self, next, pax_headers, tarfile):
        """Process a GNU tar extended sparse header, version 1.0.
        """
        fields = None
        sparse = []
        buf = tarfile.fileobj.read(BLOCKSIZE)
        fields, buf = buf.split(b"\n", 1)
        fields = int(fields)
        while len(sparse) < fields * 2:
            if b"\n" not in buf:
                buf += tarfile.fileobj.read(BLOCKSIZE)
            number, buf = buf.split(b"\n", 1)
            sparse.append(int(number))
        next.offset_data = tarfile.fileobj.tell()
        next.sparse = list(zip(sparse[::2], sparse[1::2]))

    def _apply_pax_info(self, pax_headers, encoding, errors):
        """Replace fields with supplemental information from a previous
           pax extended or global header.
        """
        for keyword, value in pax_headers.items():
            if keyword == "GNU.sparse.name":
                setattr(self, "path", value)
            elif keyword == "GNU.sparse.size":
                setattr(self, "size", int(value))
            elif keyword == "GNU.sparse.realsize":
                setattr(self, "size", int(value))
            elif keyword in PAX_FIELDS:
                if keyword in PAX_NUMBER_FIELDS:
                    try:
                        value = PAX_NUMBER_FIELDS[keyword](value)
                    except ValueError:
                        value = 0
                if keyword == "path":
                    value = value.rstrip("/")
                setattr(self, keyword, value)

        self.pax_headers = pax_headers.copy()

    def _decode_pax_field(self, value, encoding, fallback_encoding, fallback_errors):
        """Decode a single field from a pax record.
        """
        try:
            return value.decode(encoding, "strict")
        except UnicodeDecodeError:
            return value.decode(fallback_encoding, fallback_errors)

    def _block(self, count):
        """Round up a byte count by BLOCKSIZE and return it,
           e.g. _block(834) => 1024.
        """
        # Only non-negative offsets are allowed
        if count < 0:
            raise InvalidHeaderError("invalid offset")
        blocks, remainder = divmod(count, BLOCKSIZE)
        if remainder:
            blocks += 1
        return blocks * BLOCKSIZE

    def isreg(self):
        'Return True if the Tarinfo object is a regular file.'
        return self.type in REGULAR_TYPES

    def isfile(self):
        'Return True if the Tarinfo object is a regular file.'
        return self.isreg()

    def isdir(self):
        'Return True if it is a directory.'
        return self.type == DIRTYPE

    def issym(self):
        'Return True if it is a symbolic link.'
        return self.type == SYMTYPE

    def islnk(self):
        'Return True if it is a hard link.'
        return self.type == LNKTYPE

    def ischr(self):
        'Return True if it is a character device.'
        return self.type == CHRTYPE

    def isblk(self):
        'Return True if it is a block device.'
        return self.type == BLKTYPE

    def isfifo(self):
        'Return True if it is a FIFO.'
        return self.type == FIFOTYPE

    def issparse(self):
        return self.sparse is not None

    def isdev(self):
        'Return True if it is one of character device, block device or FIFO.'
        return self.type in (CHRTYPE, BLKTYPE, FIFOTYPE)
# class TarInfo

class TarFile(object):
    """The TarFile Class provides an interface to tar archives.
    """

    debug = 0                   # May be set from 0 (no msgs) to 3 (all msgs)

    dereference = False         # If true, add content of linked file to the
                                # tar file, else the link.

    ignore_zeros = False        # If true, skips empty or invalid blocks and
                                # continues processing.

    errorlevel = 1              # If 0, fatal errors only appear in debug
                                # messages (if debug >= 0). If > 0, errors
                                # are passed to the caller as exceptions.

    format = DEFAULT_FORMAT     # The format to use when creating an archive.

    encoding = ENCODING         # Encoding for 8-bit character strings.

    errors = None               # Error handler for unicode conversion.

    tarinfo = TarInfo           # The default TarInfo class to use.

    fileobject = ExFileObject   # The file-object for extractfile().

    extraction_filter = None    # The default filter for extraction.

    def __init__(self, name=None, mode="r", fileobj=None, format=None,
            tarinfo=None, dereference=None, ignore_zeros=None, encoding=None,
            errors="surrogateescape", pax_headers=None, debug=None,
            errorlevel=None, copybufsize=None):
        """Open an (uncompressed) tar archive `name'. `mode' is either 'r' to
           read from an existing archive, 'a' to append data to an existing
           file or 'w' to create a new file overwriting an existing one. `mode'
           defaults to 'r'.
           If `fileobj' is given, it is used for reading or writing data. If it
           can be determined, `mode' is overridden by `fileobj's mode.
           `fileobj' is not closed, when TarFile is closed.
        """
        modes = {"r": "rb", "a": "r+b", "w": "wb", "x": "xb"}
        if mode not in modes:
            raise ValueError("mode must be 'r', 'a', 'w' or 'x'")
        self.mode = mode
        self._mode = modes[mode]

        if not fileobj:
            if self.mode == "a" and not os.path.exists(name):
                # Create nonexistent files in append mode.
                self.mode = "w"
                self._mode = "wb"
            fileobj = bltn_open(name, self._mode)
            self._extfileobj = False
        else:
            if (name is None and hasattr(fileobj, "name") and
                isinstance(fileobj.name, (str, bytes))):
                name = fileobj.name
            if hasattr(fileobj, "mode"):
                self._mode = fileobj.mode
            self._extfileobj = True
        self.name = os.path.abspath(name) if name else None
        self.fileobj = fileobj

        # Init attributes.
        if format is not None:
            self.format = format
        if tarinfo is not None:
            self.tarinfo = tarinfo
        if dereference is not None:
            self.dereference = dereference
        if ignore_zeros is not None:
            self.ignore_zeros = ignore_zeros
        if encoding is not None:
            self.encoding = encoding
        self.errors = errors

        if pax_headers is not None and self.format == PAX_FORMAT:
            self.pax_headers = pax_headers
        else:
            self.pax_headers = {}

        if debug is not None:
            self.debug = debug
        if errorlevel is not None:
            self.errorlevel = errorlevel

        # Init datastructures.
        self.copybufsize = copybufsize
        self.closed = False
        self.members = []       # list of members as TarInfo objects
        self._loaded = False    # flag if all members have been read
        self.offset = self.fileobj.tell()
                                # current position in the archive file
        self.inodes = {}        # dictionary caching the inodes of
                                # archive members already added

        try:
            if self.mode == "r":
                self.firstmember = None
                self.firstmember = self.next()

            if self.mode == "a":
                # Move to the end of the archive,
                # before the first empty block.
                while True:
                    self.fileobj.seek(self.offset)
                    try:
                        tarinfo = self.tarinfo.fromtarfile(self)
                        self.members.append(tarinfo)
                    except EOFHeaderError:
                        self.fileobj.seek(self.offset)
                        break
                    except HeaderError as e:
                        raise ReadError(str(e)) from None

            if self.mode in ("a", "w", "x"):
                self._loaded = True

                if self.pax_headers:
                    buf = self.tarinfo.create_pax_global_header(self.pax_headers.copy())
                    self.fileobj.write(buf)
                    self.offset += len(buf)
        except:
            if not self._extfileobj:
                self.fileobj.close()
            self.closed = True
            raise

    #--------------------------------------------------------------------------
    # Below are the classmethods which act as alternate constructors to the
    # TarFile class. The open() method is the only one that is needed for
    # public use; it is the "super"-constructor and is able to select an
    # adequate "sub"-constructor for a particular compression using the mapping
    # from OPEN_METH.
    #
    # This concept allows one to subclass TarFile without losing the comfort of
    # the super-constructor. A sub-constructor is registered and made available
    # by adding it to the mapping in OPEN_METH.

    @classmethod
    def open(cls, name=None, mode="r", fileobj=None, bufsize=RECORDSIZE, **kwargs):
        """Open a tar archive for reading, writing or appending. Return
           an appropriate TarFile class.

           mode:
           'r' or 'r:*' open for reading with transparent compression
           'r:'         open for reading exclusively uncompressed
           'r:gz'       open for reading with gzip compression
           'r:bz2'      open for reading with bzip2 compression
           'r:xz'       open for reading with lzma compression
           'a' or 'a:'  open for appending, creating the file if necessary
           'w' or 'w:'  open for writing without compression
           'w:gz'       open for writing with gzip compression
           'w:bz2'      open for writing with bzip2 compression
           'w:xz'       open for writing with lzma compression

           'x' or 'x:'  create a tarfile exclusively without compression, raise
                        an exception if the file is already created
           'x:gz'       create a gzip compressed tarfile, raise an exception
                        if the file is already created
           'x:bz2'      create a bzip2 compressed tarfile, raise an exception
                        if the file is already created
           'x:xz'       create an lzma compressed tarfile, raise an exception
                        if the file is already created

           'r|*'        open a stream of tar blocks with transparent compression
           'r|'         open an uncompressed stream of tar blocks for reading
           'r|gz'       open a gzip compressed stream of tar blocks
           'r|bz2'      open a bzip2 compressed stream of tar blocks
           'r|xz'       open an lzma compressed stream of tar blocks
           'w|'         open an uncompressed stream for writing
           'w|gz'       open a gzip compressed stream for writing
           'w|bz2'      open a bzip2 compressed stream for writing
           'w|xz'       open an lzma compressed stream for writing
        """

        if not name and not fileobj:
            raise ValueError("nothing to open")

        if mode in ("r", "r:*"):
            # Find out which *open() is appropriate for opening the file.
            def not_compressed(comptype):
                return cls.OPEN_METH[comptype] == 'taropen'
            error_msgs = []
            for comptype in sorted(cls.OPEN_METH, key=not_compressed):
                func = getattr(cls, cls.OPEN_METH[comptype])
                if fileobj is not None:
                    saved_pos = fileobj.tell()
                try:
                    return func(name, "r", fileobj, **kwargs)
                except (ReadError, CompressionError) as e:
                    error_msgs.append(f'- method {comptype}: {e!r}')
                    if fileobj is not None:
                        fileobj.seek(saved_pos)
                    continue
            error_msgs_summary = '\n'.join(error_msgs)
            raise ReadError(f"file could not be opened successfully:\n{error_msgs_summary}")

        elif ":" in mode:
            filemode, comptype = mode.split(":", 1)
            filemode = filemode or "r"
            comptype = comptype or "tar"

            # Select the *open() function according to
            # given compression.
            if comptype in cls.OPEN_METH:
                func = getattr(cls, cls.OPEN_METH[comptype])
            else:
                raise CompressionError("unknown compression type %r" % comptype)
            return func(name, filemode, fileobj, **kwargs)

        elif "|" in mode:
            filemode, comptype = mode.split("|", 1)
            filemode = filemode or "r"
            comptype = comptype or "tar"

            if filemode not in ("r", "w"):
                raise ValueError("mode must be 'r' or 'w'")

            stream = _Stream(name, filemode, comptype, fileobj, bufsize)
            try:
                t = cls(name, filemode, stream, **kwargs)
            except:
                stream.close()
                raise
            t._extfileobj = False
            return t

        elif mode in ("a", "w", "x"):
            return cls.taropen(name, mode, fileobj, **kwargs)

