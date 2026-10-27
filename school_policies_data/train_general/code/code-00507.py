    driven.  Encoding suffixes are case sensitive; type suffixes are
    first tried case sensitive, then case insensitive.

    The suffixes .tgz, .taz and .tz (case sensitive!) are all mapped
    to ".tar.gz".  (This is table-driven too, using the dictionary
    suffix_map).

    Optional `strict' argument when false adds a bunch of commonly found, but
    non-standard types.
    """
    if _db is None:
        init()
    return _db.guess_type(url, strict)


def guess_all_extensions(type, strict=True):
    """Guess the extensions for a file based on its MIME type.

    Return value is a list of strings giving the possible filename
    extensions, including the leading dot ('.').  The extension is not
    guaranteed to have been associated with any particular data
    stream, but would be mapped to the MIME type `type' by
    guess_type().  If no extension can be guessed for `type', None
    is returned.

    Optional `strict' argument when false adds a bunch of commonly found,
    but non-standard types.
    """
    if _db is None:
        init()
    return _db.guess_all_extensions(type, strict)

def guess_extension(type, strict=True):
    """Guess the extension for a file based on its MIME type.

    Return value is a string giving a filename extension, including the
    leading dot ('.').  The extension is not guaranteed to have been
    associated with any particular data stream, but would be mapped to the
    MIME type `type' by guess_type().  If no extension can be guessed for
    `type', None is returned.

    Optional `strict' argument when false adds a bunch of commonly found,
    but non-standard types.
    """
    if _db is None:
        init()
    return _db.guess_extension(type, strict)

def add_type(type, ext, strict=True):
    """Add a mapping between a type and an extension.

    When the extension is already known, the new
    type will replace the old one. When the type
    is already known the extension will be added
    to the list of known extensions.

    If strict is true, information will be added to
    list of standard types, else to the list of non-standard
    types.
    """
    if _db is None:
        init()
    return _db.add_type(type, ext, strict)


def init(files=None):
    global suffix_map, types_map, encodings_map, common_types
    global inited, _db
    inited = True    # so that MimeTypes.__init__() doesn't call us again

    if files is None or _db is None:
        db = MimeTypes()
        # Quick return if not supported
        db.read_windows_registry()

        if files is None:
            files = knownfiles
        else:
            files = knownfiles + list(files)
    else:
        db = _db

    for file in files:
        if os.path.isfile(file):
            db.read(file)
    encodings_map = db.encodings_map
    suffix_map = db.suffix_map
    types_map = db.types_map[True]
    common_types = db.types_map[False]
    # Make the DB a global variable now that it is fully initialized
    _db = db


def read_mime_types(file):
    try:
        f = open(file, encoding='utf-8')
    except OSError:
        return None
    with f:
        db = MimeTypes()
        db.readfp(f, True)
        return db.types_map[True]


def _default_mime_types():
    global suffix_map, _suffix_map_default
    global encodings_map, _encodings_map_default
    global types_map, _types_map_default
    global common_types, _common_types_default

    suffix_map = _suffix_map_default = {
        '.svgz': '.svg.gz',
        '.tgz': '.tar.gz',
        '.taz': '.tar.gz',
        '.tz': '.tar.gz',
        '.tbz2': '.tar.bz2',
        '.txz': '.tar.xz',
        }

    encodings_map = _encodings_map_default = {
        '.gz': 'gzip',
        '.Z': 'compress',
        '.bz2': 'bzip2',
        '.xz': 'xz',
        '.br': 'br',
        }

    # Before adding new types, make sure they are either registered with IANA,
    # at http://www.iana.org/assignments/media-types
    # or extensions, i.e. using the x- prefix

    # If you add to these, please keep them sorted by mime type.
    # Make sure the entry with the preferred file extension for a particular mime type
    # appears before any others of the same mimetype.
    types_map = _types_map_default = {
        '.js'     : 'application/javascript',
        '.mjs'    : 'application/javascript',
        '.json'   : 'application/json',
        '.webmanifest': 'application/manifest+json',
        '.doc'    : 'application/msword',
        '.dot'    : 'application/msword',
        '.wiz'    : 'application/msword',
        '.bin'    : 'application/octet-stream',
        '.a'      : 'application/octet-stream',
        '.dll'    : 'application/octet-stream',
        '.exe'    : 'application/octet-stream',
        '.o'      : 'application/octet-stream',
        '.obj'    : 'application/octet-stream',
        '.so'     : 'application/octet-stream',
        '.oda'    : 'application/oda',
        '.pdf'    : 'application/pdf',
        '.p7c'    : 'application/pkcs7-mime',
        '.ps'     : 'application/postscript',
        '.ai'     : 'application/postscript',
        '.eps'    : 'application/postscript',
        '.m3u'    : 'application/vnd.apple.mpegurl',
        '.m3u8'   : 'application/vnd.apple.mpegurl',
        '.xls'    : 'application/vnd.ms-excel',
        '.xlb'    : 'application/vnd.ms-excel',
        '.ppt'    : 'application/vnd.ms-powerpoint',
        '.pot'    : 'application/vnd.ms-powerpoint',
        '.ppa'    : 'application/vnd.ms-powerpoint',
        '.pps'    : 'application/vnd.ms-powerpoint',
        '.pwz'    : 'application/vnd.ms-powerpoint',
        '.wasm'   : 'application/wasm',
        '.bcpio'  : 'application/x-bcpio',
        '.cpio'   : 'application/x-cpio',
        '.csh'    : 'application/x-csh',
        '.dvi'    : 'application/x-dvi',
        '.gtar'   : 'application/x-gtar',
        '.hdf'    : 'application/x-hdf',
        '.h5'     : 'application/x-hdf5',
        '.latex'  : 'application/x-latex',
        '.mif'    : 'application/x-mif',
        '.cdf'    : 'application/x-netcdf',
        '.nc'     : 'application/x-netcdf',
        '.p12'    : 'application/x-pkcs12',
        '.pfx'    : 'application/x-pkcs12',
        '.ram'    : 'application/x-pn-realaudio',
        '.pyc'    : 'application/x-python-code',
        '.pyo'    : 'application/x-python-code',
        '.sh'     : 'application/x-sh',
        '.shar'   : 'application/x-shar',
        '.swf'    : 'application/x-shockwave-flash',
        '.sv4cpio': 'application/x-sv4cpio',
        '.sv4crc' : 'application/x-sv4crc',
        '.tar'    : 'application/x-tar',
        '.tcl'    : 'application/x-tcl',
        '.tex'    : 'application/x-tex',
        '.texi'   : 'application/x-texinfo',
        '.texinfo': 'application/x-texinfo',
        '.roff'   : 'application/x-troff',
        '.t'      : 'application/x-troff',
        '.tr'     : 'application/x-troff',
        '.man'    : 'application/x-troff-man',
        '.me'     : 'application/x-troff-me',
        '.ms'     : 'application/x-troff-ms',
        '.ustar'  : 'application/x-ustar',
        '.src'    : 'application/x-wais-source',
        '.xsl'    : 'application/xml',
        '.rdf'    : 'application/xml',
        '.wsdl'   : 'application/xml',
        '.xpdl'   : 'application/xml',
        '.zip'    : 'application/zip',
        '.3gp'    : 'audio/3gpp',
        '.3gpp'   : 'audio/3gpp',
        '.3g2'    : 'audio/3gpp2',
        '.3gpp2'  : 'audio/3gpp2',
        '.aac'    : 'audio/aac',
        '.adts'   : 'audio/aac',
        '.loas'   : 'audio/aac',
        '.ass'    : 'audio/aac',
        '.au'     : 'audio/basic',
        '.snd'    : 'audio/basic',
        '.mp3'    : 'audio/mpeg',
        '.mp2'    : 'audio/mpeg',
        '.opus'   : 'audio/opus',
        '.aif'    : 'audio/x-aiff',
        '.aifc'   : 'audio/x-aiff',
        '.aiff'   : 'audio/x-aiff',
        '.ra'     : 'audio/x-pn-realaudio',
        '.wav'    : 'audio/x-wav',
        '.bmp'    : 'image/bmp',
        '.gif'    : 'image/gif',
        '.ief'    : 'image/ief',
        '.jpg'    : 'image/jpeg',
        '.jpe'    : 'image/jpeg',
        '.jpeg'   : 'image/jpeg',
        '.heic'   : 'image/heic',
        '.heif'   : 'image/heif',
        '.png'    : 'image/png',
        '.svg'    : 'image/svg+xml',
        '.tiff'   : 'image/tiff',
        '.tif'    : 'image/tiff',
        '.ico'    : 'image/vnd.microsoft.icon',
        '.ras'    : 'image/x-cmu-raster',
        '.bmp'    : 'image/x-ms-bmp',
        '.pnm'    : 'image/x-portable-anymap',
        '.pbm'    : 'image/x-portable-bitmap',
        '.pgm'    : 'image/x-portable-graymap',
        '.ppm'    : 'image/x-portable-pixmap',
        '.rgb'    : 'image/x-rgb',
        '.xbm'    : 'image/x-xbitmap',
        '.xpm'    : 'image/x-xpixmap',
        '.xwd'    : 'image/x-xwindowdump',
        '.eml'    : 'message/rfc822',
        '.mht'    : 'message/rfc822',
        '.mhtml'  : 'message/rfc822',
        '.nws'    : 'message/rfc822',
        '.css'    : 'text/css',
        '.csv'    : 'text/csv',
        '.html'   : 'text/html',
        '.htm'    : 'text/html',
        '.txt'    : 'text/plain',
        '.bat'    : 'text/plain',
        '.c'      : 'text/plain',
        '.h'      : 'text/plain',
        '.ksh'    : 'text/plain',
        '.pl'     : 'text/plain',
        '.rtx'    : 'text/richtext',
        '.tsv'    : 'text/tab-separated-values',
        '.py'     : 'text/x-python',
        '.etx'    : 'text/x-setext',
        '.sgm'    : 'text/x-sgml',
        '.sgml'   : 'text/x-sgml',
        '.vcf'    : 'text/x-vcard',
        '.xml'    : 'text/xml',
        '.mp4'    : 'video/mp4',
        '.mpeg'   : 'video/mpeg',
        '.m1v'    : 'video/mpeg',
        '.mpa'    : 'video/mpeg',
        '.mpe'    : 'video/mpeg',
        '.mpg'    : 'video/mpeg',
        '.mov'    : 'video/quicktime',
        '.qt'     : 'video/quicktime',
        '.webm'   : 'video/webm',
        '.avi'    : 'video/x-msvideo',
        '.movie'  : 'video/x-sgi-movie',
        }

    # These are non-standard types, commonly found in the wild.  They will
    # only match if strict=0 flag is given to the API methods.

    # Please sort these too
    common_types = _common_types_default = {
        '.rtf' : 'application/rtf',
        '.midi': 'audio/midi',
        '.mid' : 'audio/midi',
        '.jpg' : 'image/jpg',
        '.pict': 'image/pict',
        '.pct' : 'image/pict',
        '.pic' : 'image/pict',
        '.xul' : 'text/xul',
        }


_default_mime_types()


def _main():
    import getopt

    USAGE = """\
Usage: mimetypes.py [options] type

Options:
    --help / -h       -- print this message and exit
    --lenient / -l    -- additionally search of some common, but non-standard
                         types.
    --extension / -e  -- guess extension instead of type

More than one type argument may be given.
"""

    def usage(code, msg=''):
        print(USAGE)
        if msg: print(msg)
        sys.exit(code)

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hle',
                                   ['help', 'lenient', 'extension'])
    except getopt.error as msg:
        usage(1, msg)

    strict = 1
    extension = 0
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage(0)
        elif opt in ('-l', '--lenient'):
            strict = 0
        elif opt in ('-e', '--extension'):
            extension = 1
    for gtype in args:
        if extension:
            guess = guess_extension(gtype, strict)
            if not guess: print("I don't know anything about type", gtype)
            else: print(guess)
        else:
            guess, encoding = guess_type(gtype, strict)
            if not guess: print("I don't know anything about type", gtype)
            else: print('type:', guess, 'encoding:', encoding)


if __name__ == '__main__':
    _main()
<|endoftext|>"""
    ast
    ~~~

    The `ast` module helps Python applications to process trees of the Python
    abstract syntax grammar.  The abstract syntax itself might change with
    each Python release; this module helps to find out programmatically what
    the current grammar looks like and allows modifications of it.

    An abstract syntax tree can be generated by passing `ast.PyCF_ONLY_AST` as
    a flag to the `compile()` builtin function or by using the `parse()`
    function from this module.  The result will be a tree of objects whose
    classes all inherit from `ast.AST`.

    A modified abstract syntax tree can be compiled into a Python code object
    using the built-in `compile()` function.

    Additionally various helper functions are provided that make working with
    the trees simpler.  The main intention of the helper functions and this
    module in general is to provide an easy to use interface for libraries
    that work tightly with the python syntax (template engines for example).


    :copyright: Copyright 2008 by Armin Ronacher.
    :license: Python License.
"""
import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto


def parse(source, filename='<unknown>', mode='exec', *,
          type_comments=False, feature_version=None):
    """
    Parse the source into an AST node.
    Equivalent to compile(source, filename, mode, PyCF_ONLY_AST).
    Pass type_comments=True to get back type comments where the syntax allows.
    """
    flags = PyCF_ONLY_AST
    if type_comments:
        flags |= PyCF_TYPE_COMMENTS
    if isinstance(feature_version, tuple):
        major, minor = feature_version  # Should be a 2-tuple.
        assert major == 3
        feature_version = minor
    elif feature_version is None:
        feature_version = -1
    # Else it should be an int giving the minor version for 3.x.
    return compile(source, filename, mode, flags,
                   _feature_version=feature_version)


def literal_eval(node_or_string):
    """
    Evaluate an expression node or a string containing only a Python
    expression.  The string or node provided may only consist of the following
    Python literal structures: strings, bytes, numbers, tuples, lists, dicts,
    sets, booleans, and None.

    Caution: A complex expression can overflow the C stack and cause a crash.
    """
    if isinstance(node_or_string, str):
        node_or_string = parse(node_or_string.lstrip(" \t"), mode='eval')
    if isinstance(node_or_string, Expression):
        node_or_string = node_or_string.body
    def _raise_malformed_node(node):
        msg = "malformed node or string"
        if lno := getattr(node, 'lineno', None):
            msg += f' on line {lno}'
        raise ValueError(msg + f': {node!r}')
    def _convert_num(node):
        if not isinstance(node, Constant) or type(node.value) not in (int, float, complex):
            _raise_malformed_node(node)
        return node.value
    def _convert_signed_num(node):
        if isinstance(node, UnaryOp) and isinstance(node.op, (UAdd, USub)):
            operand = _convert_num(node.operand)
            if isinstance(node.op, UAdd):
                return + operand
            else:
                return - operand
        return _convert_num(node)
    def _convert(node):
        if isinstance(node, Constant):
            return node.value
        elif isinstance(node, Tuple):
            return tuple(map(_convert, node.elts))
        elif isinstance(node, List):
            return list(map(_convert, node.elts))
        elif isinstance(node, Set):
            return set(map(_convert, node.elts))
        elif (isinstance(node, Call) and isinstance(node.func, Name) and
              node.func.id == 'set' and node.args == node.keywords == []):
            return set()
        elif isinstance(node, Dict):
            if len(node.keys) != len(node.values):
                _raise_malformed_node(node)
            return dict(zip(map(_convert, node.keys),
                            map(_convert, node.values)))
        elif isinstance(node, BinOp) and isinstance(node.op, (Add, Sub)):
            left = _convert_signed_num(node.left)
            right = _convert_num(node.right)
            if isinstance(left, (int, float)) and isinstance(right, complex):
                if isinstance(node.op, Add):
                    return left + right
                else:
                    return left - right
        return _convert_signed_num(node)
    return _convert(node_or_string)


def dump(node, annotate_fields=True, include_attributes=False, *, indent=None):
    """
    Return a formatted dump of the tree in node.  This is mainly useful for
    debugging purposes.  If annotate_fields is true (by default),
    the returned string will show the names and the values for fields.
    If annotate_fields is false, the result string will be more compact by
    omitting unambiguous field names.  Attributes such as line
    numbers and column offsets are not dumped by default.  If this is wanted,
    include_attributes can be set to true.  If indent is a non-negative
    integer or string, then the tree will be pretty-printed with that indent
    level. None (the default) selects the single line representation.
    """
    def _format(node, level=0):
        if indent is not None:
            level += 1
            prefix = '\n' + indent * level
            sep = ',\n' + indent * level
        else:
            prefix = ''
            sep = ', '
        if isinstance(node, AST):
            cls = type(node)
            args = []
            allsimple = True
            keywords = annotate_fields
            for name in node._fields:
                try:
                    value = getattr(node, name)
                except AttributeError:
                    keywords = True
                    continue
                if value is None and getattr(cls, name, ...) is None:
                    keywords = True
                    continue
                value, simple = _format(value, level)
                allsimple = allsimple and simple
                if keywords:
                    args.append('%s=%s' % (name, value))
                else:
                    args.append(value)
            if include_attributes and node._attributes:
                for name in node._attributes:
                    try:
                        value = getattr(node, name)
                    except AttributeError:
                        continue
                    if value is None and getattr(cls, name, ...) is None:
                        continue
                    value, simple = _format(value, level)
                    allsimple = allsimple and simple
                    args.append('%s=%s' % (name, value))
            if allsimple and len(args) <= 3:
                return '%s(%s)' % (node.__class__.__name__, ', '.join(args)), not args
            return '%s(%s%s)' % (node.__class__.__name__, prefix, sep.join(args)), False
        elif isinstance(node, list):
            if not node:
                return '[]', True
            return '[%s%s]' % (prefix, sep.join(_format(x, level)[0] for x in node)), False
        return repr(node), True

    if not isinstance(node, AST):
        raise TypeError('expected AST, got %r' % node.__class__.__name__)
    if indent is not None and not isinstance(indent, str):
        indent = ' ' * indent
    return _format(node)[0]


def copy_location(new_node, old_node):
    """
    Copy source location (`lineno`, `col_offset`, `end_lineno`, and `end_col_offset`
    attributes) from *old_node* to *new_node* if possible, and return *new_node*.
    """
    for attr in 'lineno', 'col_offset', 'end_lineno', 'end_col_offset':
        if attr in old_node._attributes and attr in new_node._attributes:
            value = getattr(old_node, attr, None)
            # end_lineno and end_col_offset are optional attributes, and they
            # should be copied whether the value is None or not.
            if value is not None or (
                hasattr(old_node, attr) and attr.startswith("end_")
            ):
                setattr(new_node, attr, value)
    return new_node


def fix_missing_locations(node):
    """
    When you compile a node tree with compile(), the compiler expects lineno and
    col_offset attributes for every node that supports them.  This is rather
    tedious to fill in for generated nodes, so this helper adds these attributes
    recursively where not already set, by setting them to the values of the
    parent node.  It works recursively starting at *node*.
    """
    def _fix(node, lineno, col_offset, end_lineno, end_col_offset):
        if 'lineno' in node._attributes:
            if not hasattr(node, 'lineno'):
                node.lineno = lineno
            else:
                lineno = node.lineno
        if 'end_lineno' in node._attributes:
            if getattr(node, 'end_lineno', None) is None:
                node.end_lineno = end_lineno
            else:
                end_lineno = node.end_lineno
        if 'col_offset' in node._attributes:
            if not hasattr(node, 'col_offset'):
                node.col_offset = col_offset
            else:
                col_offset = node.col_offset
        if 'end_col_offset' in node._attributes:
            if getattr(node, 'end_col_offset', None) is None:
                node.end_col_offset = end_col_offset
            else:
                end_col_offset = node.end_col_offset
        for child in iter_child_nodes(node):
            _fix(child, lineno, col_offset, end_lineno, end_col_offset)
    _fix(node, 1, 0, 1, 0)
    return node


def increment_lineno(node, n=1):
    """
    Increment the line number and end line number of each node in the tree
    starting at *node* by *n*. This is useful to "move code" to a different
    location in a file.
    """
    for child in walk(node):
        # TypeIgnore is a special case where lineno is not an attribute
        # but rather a field of the node itself.
        if isinstance(child, TypeIgnore):
            child.lineno = getattr(child, 'lineno', 0) + n
            continue

        if 'lineno' in child._attributes:
            child.lineno = getattr(child, 'lineno', 0) + n
        if (
            "end_lineno" in child._attributes
            and (end_lineno := getattr(child, "end_lineno", 0)) is not None
        ):
            child.end_lineno = end_lineno + n
    return node


def iter_fields(node):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


def iter_child_nodes(node):
    """
    Yield all direct child nodes of *node*, that is, all fields that are nodes
    and all items of fields that are lists of nodes.
    """
    for name, field in iter_fields(node):
        if isinstance(field, AST):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, AST):
                    yield item


def get_docstring(node, clean=True):
    """
    Return the docstring for the given node or None if no docstring can
    be found.  If the node provided does not have docstrings a TypeError
    will be raised.

    If *clean* is `True`, all tabs are expanded to spaces and any whitespace
    that can be uniformly removed from the second line onwards is removed.
    """
    if not isinstance(node, (AsyncFunctionDef, FunctionDef, ClassDef, Module)):
        raise TypeError("%r can't have docstrings" % node.__class__.__name__)
    if not(node.body and isinstance(node.body[0], Expr)):
        return None
    node = node.body[0].value
    if isinstance(node, Str):
        text = node.s
    elif isinstance(node, Constant) and isinstance(node.value, str):
        text = node.value
    else:
        return None
    if clean:
        import inspect
        text = inspect.cleandoc(text)
    return text


def _splitlines_no_ff(source):
    """Split a string into lines ignoring form feed and other chars.

    This mimics how the Python parser splits source code.
    """
    idx = 0
    lines = []
    next_line = ''
    while idx < len(source):
        c = source[idx]
        next_line += c
        idx += 1
        # Keep \r\n together
        if c == '\r' and idx < len(source) and source[idx] == '\n':
            next_line += '\n'
            idx += 1
        if c in '\r\n':
            lines.append(next_line)
            next_line = ''

    if next_line:
        lines.append(next_line)
    return lines


def _pad_whitespace(source):
    r"""Replace all chars except '\f\t' in a line with spaces."""
    result = ''
    for c in source:
        if c in '\f\t':
            result += c
        else:
            result += ' '
    return result


def get_source_segment(source, node, *, padded=False):
    """Get source code segment of the *source* that generated *node*.

    If some location information (`lineno`, `end_lineno`, `col_offset`,
    or `end_col_offset`) is missing, return None.

    If *padded* is `True`, the first line of a multi-line statement will
    be padded with spaces to match its original position.
    """
    try:
        if node.end_lineno is None or node.end_col_offset is None:
            return None
        lineno = node.lineno - 1
        end_lineno = node.end_lineno - 1
        col_offset = node.col_offset
        end_col_offset = node.end_col_offset
    except AttributeError:
        return None

    lines = _splitlines_no_ff(source)
    if end_lineno == lineno:
        return lines[lineno].encode()[col_offset:end_col_offset].decode()

    if padded:
        padding = _pad_whitespace(lines[lineno].encode()[:col_offset].decode())
    else:
        padding = ''

    first = padding + lines[lineno].encode()[col_offset:].decode()
    last = lines[end_lineno].encode()[:end_col_offset].decode()
    lines = lines[lineno+1:end_lineno]

    lines.insert(0, first)
    lines.append(last)
    return ''.join(lines)


def walk(node):
    """
    Recursively yield all descendant nodes in the tree starting at *node*
    (including *node* itself), in no specified order.  This is useful if you
    only want to modify nodes in place and don't care about the context.
    """
    from collections import deque
    todo = deque([node])
    while todo:
        node = todo.popleft()
        todo.extend(iter_child_nodes(node))
        yield node


class NodeVisitor(object):
    """
    A node visitor base class that walks the abstract syntax tree and calls a
    visitor function for every node found.  This function may return a value
    which is forwarded by the `visit` method.

    This class is meant to be subclassed, with the subclass adding visitor
    methods.

    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node.  So a `TryFinally` node visit function would
    be `visit_TryFinally`.  This behavior can be changed by overriding
    the `visit` method.  If no visitor function exists for a node
    (return value `None`) the `generic_visit` visitor is used instead.

    Don't use the `NodeVisitor` if you want to apply changes to nodes during
    traversing.  For this a special visitor exists (`NodeTransformer`) that
    allows modifications.
    """

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        self.visit(item)
            elif isinstance(value, AST):
                self.visit(value)

    def visit_Constant(self, node):
        value = node.value
        type_name = _const_node_type_names.get(type(value))
        if type_name is None:
            for cls, name in _const_node_type_names.items():
                if isinstance(value, cls):
                    type_name = name
                    break
        if type_name is not None:
            method = 'visit_' + type_name
            try:
                visitor = getattr(self, method)
            except AttributeError:
                pass
            else:
                import warnings
                warnings.warn(f"{method} is deprecated; add visit_Constant",
                              DeprecationWarning, 2)
                return visitor(node)
        return self.generic_visit(node)


class NodeTransformer(NodeVisitor):
    """
    A :class:`NodeVisitor` subclass that walks the abstract syntax tree and
    allows modification of nodes.

    The `NodeTransformer` will walk the AST and use the return value of the
    visitor methods to replace or remove the old node.  If the return value of
    the visitor method is ``None``, the node will be removed from its location,
    otherwise it is replaced with the return value.  The return value may be the
    original node in which case no replacement takes place.

    Here is an example transformer that rewrites all occurrences of name lookups
    (``foo``) to ``data['foo']``::

       class RewriteName(NodeTransformer):

           def visit_Name(self, node):
               return Subscript(
                   value=Name(id='data', ctx=Load()),
                   slice=Constant(value=node.id),
                   ctx=node.ctx
               )

    Keep in mind that if the node you're operating on has child nodes you must
    either transform the child nodes yourself or call the :meth:`generic_visit`
    method for the node first.

    For nodes that were part of a collection of statements (that applies to all
    statement nodes), the visitor may also return a list of nodes rather than
    just a single node.

    Usually you use the transformer like this::

       node = YourTransformer().visit(node)
    """

    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


# If the ast module is loaded more than once, only add deprecated methods once
if not hasattr(Constant, 'n'):
    # The following code is for backward compatibility.
    # It will be removed in future.

    def _getter(self):
        """Deprecated. Use value instead."""
        return self.value

    def _setter(self, value):
        self.value = value

    Constant.n = property(_getter, _setter)
    Constant.s = property(_getter, _setter)

class _ABC(type):

    def __init__(cls, *args):
        cls.__doc__ = """Deprecated AST node class. Use ast.Constant instead"""

    def __instancecheck__(cls, inst):
        if not isinstance(inst, Constant):
            return False
        if cls in _const_types:
            try:
                value = inst.value
            except AttributeError:
                return False
            else:
                return (
                    isinstance(value, _const_types[cls]) and
                    not isinstance(value, _const_types_not.get(cls, ()))
                )
        return type.__instancecheck__(cls, inst)

def _new(cls, *args, **kwargs):
    for key in kwargs:
        if key not in cls._fields:
            # arbitrary keyword arguments are accepted
            continue
        pos = cls._fields.index(key)
        if pos < len(args):
            raise TypeError(f"{cls.__name__} got multiple values for argument {key!r}")
    if cls in _const_types:
        return Constant(*args, **kwargs)
    return Constant.__new__(cls, *args, **kwargs)

class Num(Constant, metaclass=_ABC):
    _fields = ('n',)
    __new__ = _new

class Str(Constant, metaclass=_ABC):
    _fields = ('s',)
    __new__ = _new

class Bytes(Constant, metaclass=_ABC):
    _fields = ('s',)
    __new__ = _new

class NameConstant(Constant, metaclass=_ABC):
    __new__ = _new

class Ellipsis(Constant, metaclass=_ABC):
    _fields = ()

    def __new__(cls, *args, **kwargs):
        if cls is Ellipsis:
            return Constant(..., *args, **kwargs)
        return Constant.__new__(cls, *args, **kwargs)

_const_types = {
    Num: (int, float, complex),
    Str: (str,),
    Bytes: (bytes,),
    NameConstant: (type(None), bool),
    Ellipsis: (type(...),),
}
_const_types_not = {
    Num: (bool,),
}

_const_node_type_names = {
    bool: 'NameConstant',  # should be before int
    type(None): 'NameConstant',
    int: 'Num',
    float: 'Num',
    complex: 'Num',
    str: 'Str',
    bytes: 'Bytes',
    type(...): 'Ellipsis',
}

class slice(AST):
    """Deprecated AST node class."""

class Index(slice):
    """Deprecated AST node class. Use the index value directly instead."""
    def __new__(cls, value, **kwargs):
        return value

class ExtSlice(slice):
    """Deprecated AST node class. Use ast.Tuple instead."""
    def __new__(cls, dims=(), **kwargs):
        return Tuple(list(dims), Load(), **kwargs)

# If the ast module is loaded more than once, only add deprecated methods once
if not hasattr(Tuple, 'dims'):
    # The following code is for backward compatibility.
    # It will be removed in future.

    def _dims_getter(self):
        """Deprecated. Use elts instead."""
        return self.elts

    def _dims_setter(self, value):
        self.elts = value

    Tuple.dims = property(_dims_getter, _dims_setter)

class Suite(mod):
    """Deprecated AST node class.  Unused in Python 3."""

class AugLoad(expr_context):
    """Deprecated AST node class.  Unused in Python 3."""

class AugStore(expr_context):
    """Deprecated AST node class.  Unused in Python 3."""

class Param(expr_context):
    """Deprecated AST node class.  Unused in Python 3."""


# Large float and imaginary literals get turned into infinities in the AST.
# We unparse those infinities to INFSTR.
_INFSTR = "1e" + repr(sys.float_info.max_10_exp + 1)

class _Precedence(IntEnum):
    """Precedence table that originated from python grammar."""

    TUPLE = auto()
    YIELD = auto()           # 'yield', 'yield from'
    TEST = auto()            # 'if'-'else', 'lambda'
    OR = auto()              # 'or'
    AND = auto()             # 'and'
    NOT = auto()             # 'not'
    CMP = auto()             # '<', '>', '==', '>=', '<=', '!=',
                             # 'in', 'not in', 'is', 'is not'
    EXPR = auto()
    BOR = EXPR               # '|'
    BXOR = auto()            # '^'
    BAND = auto()            # '&'
    SHIFT = auto()           # '<<', '>>'
    ARITH = auto()           # '+', '-'
    TERM = auto()            # '*', '@', '/', '%', '//'
    FACTOR = auto()          # unary '+', '-', '~'
    POWER = auto()           # '**'
    AWAIT = auto()           # 'await'
    ATOM = auto()

    def next(self):
        try:
            return self.__class__(self + 1)
        except ValueError:
            return self


_SINGLE_QUOTES = ("'", '"')
_MULTI_QUOTES = ('"""', "'''")
_ALL_QUOTES = (*_SINGLE_QUOTES, *_MULTI_QUOTES)

class _Unparser(NodeVisitor):
    """Methods in this class recursively traverse an AST and
    output source code for the abstract syntax; original formatting
    is disregarded."""

    def __init__(self, *, _avoid_backslashes=False):
        self._source = []
        self._buffer = []
        self._precedences = {}
        self._type_ignores = {}
        self._indent = 0
        self._avoid_backslashes = _avoid_backslashes

    def interleave(self, inter, f, seq):
        """Call f on each item in seq, calling inter() in between."""
        seq = iter(seq)
        try:
            f(next(seq))
        except StopIteration:
            pass
        else:
            for x in seq:
                inter()
                f(x)

    def items_view(self, traverser, items):
        """Traverse and separate the given *items* with a comma and append it to
        the buffer. If *items* is a single item sequence, a trailing comma
        will be added."""
        if len(items) == 1:
            traverser(items[0])
            self.write(",")
        else:
            self.interleave(lambda: self.write(", "), traverser, items)

    def maybe_newline(self):
        """Adds a newline if it isn't the start of generated source"""
        if self._source:
            self.write("\n")

    def fill(self, text=""):
        """Indent a piece of text and append it, according to the current
        indentation level"""
        self.maybe_newline()
        self.write("    " * self._indent + text)

    def write(self, text):
        """Append a piece of text"""
        self._source.append(text)

    def buffer_writer(self, text):
        self._buffer.append(text)

    @property
    def buffer(self):
        value = "".join(self._buffer)
        self._buffer.clear()
        return value

    @contextmanager
    def block(self, *, extra = None):
        """A context manager for preparing the source for blocks. It adds
        the character':', increases the indentation on enter and decreases
        the indentation on exit. If *extra* is given, it will be directly
        appended after the colon character.
        """
        self.write(":")
        if extra:
            self.write(extra)
        self._indent += 1
        yield
        self._indent -= 1

    @contextmanager
    def delimit(self, start, end):
        """A context manager for preparing the source for expressions. It adds
        *start* to the buffer and enters, after exit it adds *end*."""

        self.write(start)
        yield
        self.write(end)

    def delimit_if(self, start, end, condition):
        if condition:
            return self.delimit(start, end)
        else:
            return nullcontext()

    def require_parens(self, precedence, node):
        """Shortcut to adding precedence related parens"""
        return self.delimit_if("(", ")", self.get_precedence(node) > precedence)

    def get_precedence(self, node):
        return self._precedences.get(node, _Precedence.TEST)

    def set_precedence(self, precedence, *nodes):
        for node in nodes:
            self._precedences[node] = precedence

    def get_raw_docstring(self, node):
        """If a docstring node is found in the body of the *node* parameter,
        return that docstring node, None otherwise.

        Logic mirrored from ``_PyAST_GetDocString``."""
        if not isinstance(
            node, (AsyncFunctionDef, FunctionDef, ClassDef, Module)
        ) or len(node.body) < 1:
            return None
        node = node.body[0]
        if not isinstance(node, Expr):
            return None
        node = node.value
        if isinstance(node, Constant) and isinstance(node.value, str):
            return node

    def get_type_comment(self, node):
        comment = self._type_ignores.get(node.lineno) or node.type_comment
        if comment is not None:
            return f" # type: {comment}"

    def traverse(self, node):
        if isinstance(node, list):
            for item in node:
                self.traverse(item)
        else:
            super().visit(node)

    # Note: as visit() resets the output text, do NOT rely on
    # NodeVisitor.generic_visit to handle any nodes (as it calls back in to
    # the subclass visit() method, which resets self._source to an empty list)
    def visit(self, node):
        """Outputs a source code string that, if converted back to an ast
        (using ast.parse) will generate an AST equivalent to *node*"""
        self._source = []
        self.traverse(node)
        return "".join(self._source)

    def _write_docstring_and_traverse_body(self, node):
        if (docstring := self.get_raw_docstring(node)):
            self._write_docstring(docstring)
            self.traverse(node.body[1:])
        else:
            self.traverse(node.body)

    def visit_Module(self, node):
        self._type_ignores = {
            ignore.lineno: f"ignore{ignore.tag}"
            for ignore in node.type_ignores
        }
        self._write_docstring_and_traverse_body(node)
        self._type_ignores.clear()

    def visit_FunctionType(self, node):
        with self.delimit("(", ")"):
            self.interleave(
                lambda: self.write(", "), self.traverse, node.argtypes
            )

        self.write(" -> ")
        self.traverse(node.returns)

    def visit_Expr(self, node):
        self.fill()
        self.set_precedence(_Precedence.YIELD, node.value)
        self.traverse(node.value)

    def visit_NamedExpr(self, node):
        with self.require_parens(_Precedence.TUPLE, node):
            self.set_precedence(_Precedence.ATOM, node.target, node.value)
            self.traverse(node.target)
            self.write(" := ")
            self.traverse(node.value)

    def visit_Import(self, node):
        self.fill("import ")
        self.interleave(lambda: self.write(", "), self.traverse, node.names)

    def visit_ImportFrom(self, node):
        self.fill("from ")
        self.write("." * (node.level or 0))
        if node.module:
            self.write(node.module)
        self.write(" import ")
        self.interleave(lambda: self.write(", "), self.traverse, node.names)

    def visit_Assign(self, node):
        self.fill()
        for target in node.targets:
            self.traverse(target)
            self.write(" = ")
        self.traverse(node.value)
        if type_comment := self.get_type_comment(node):
            self.write(type_comment)

    def visit_AugAssign(self, node):
        self.fill()
        self.traverse(node.target)
        self.write(" " + self.binop[node.op.__class__.__name__] + "= ")
        self.traverse(node.value)

    def visit_AnnAssign(self, node):
        self.fill()
        with self.delimit_if("(", ")", not node.simple and isinstance(node.target, Name)):
            self.traverse(node.target)
        self.write(": ")
        self.traverse(node.annotation)
        if node.value:
            self.write(" = ")
            self.traverse(node.value)

    def visit_Return(self, node):
        self.fill("return")
        if node.value:
            self.write(" ")
            self.traverse(node.value)

    def visit_Pass(self, node):
        self.fill("pass")

    def visit_Break(self, node):
        self.fill("break")

    def visit_Continue(self, node):
        self.fill("continue")

    def visit_Delete(self, node):
        self.fill("del ")
        self.interleave(lambda: self.write(", "), self.traverse, node.targets)

    def visit_Assert(self, node):
        self.fill("assert ")
        self.traverse(node.test)
        if node.msg:
            self.write(", ")
            self.traverse(node.msg)

    def visit_Global(self, node):
        self.fill("global ")
        self.interleave(lambda: self.write(", "), self.write, node.names)

    def visit_Nonlocal(self, node):
        self.fill("nonlocal ")
        self.interleave(lambda: self.write(", "), self.write, node.names)

    def visit_Await(self, node):
        with self.require_parens(_Precedence.AWAIT, node):
            self.write("await")
            if node.value:
                self.write(" ")
                self.set_precedence(_Precedence.ATOM, node.value)
                self.traverse(node.value)

    def visit_Yield(self, node):
        with self.require_parens(_Precedence.YIELD, node):
            self.write("yield")
            if node.value:
                self.write(" ")
                self.set_precedence(_Precedence.ATOM, node.value)
                self.traverse(node.value)

    def visit_YieldFrom(self, node):
        with self.require_parens(_Precedence.YIELD, node):
            self.write("yield from ")
            if not node.value:
                raise ValueError("Node can't be used without a value attribute.")
            self.set_precedence(_Precedence.ATOM, node.value)
            self.traverse(node.value)

    def visit_Raise(self, node):
        self.fill("raise")
        if not node.exc:
            if node.cause:
                raise ValueError(f"Node can't use cause without an exception.")
            return
        self.write(" ")
        self.traverse(node.exc)
        if node.cause:
            self.write(" from ")
            self.traverse(node.cause)

    def visit_Try(self, node):
        self.fill("try")
        with self.block():
            self.traverse(node.body)
        for ex in node.handlers:
            self.traverse(ex)
        if node.orelse:
            self.fill("else")
            with self.block():
                self.traverse(node.orelse)
        if node.finalbody:
            self.fill("finally")
            with self.block():
                self.traverse(node.finalbody)

    def visit_ExceptHandler(self, node):
        self.fill("except")
        if node.type:
            self.write(" ")
            self.traverse(node.type)
        if node.name:
            self.write(" as ")
            self.write(node.name)
        with self.block():
            self.traverse(node.body)

    def visit_ClassDef(self, node):
        self.maybe_newline()
        for deco in node.decorator_list:
            self.fill("@")
            self.traverse(deco)
        self.fill("class " + node.name)
        with self.delimit_if("(", ")", condition = node.bases or node.keywords):
            comma = False
            for e in node.bases:
                if comma:
                    self.write(", ")
                else:
                    comma = True
                self.traverse(e)
            for e in node.keywords:
                if comma:
                    self.write(", ")
                else:
                    comma = True
                self.traverse(e)

        with self.block():
            self._write_docstring_and_traverse_body(node)

    def visit_FunctionDef(self, node):
        self._function_helper(node, "def")

    def visit_AsyncFunctionDef(self, node):
        self._function_helper(node, "async def")

    def _function_helper(self, node, fill_suffix):
        self.maybe_newline()
        for deco in node.decorator_list:
            self.fill("@")
            self.traverse(deco)
        def_str = fill_suffix + " " + node.name
        self.fill(def_str)
        with self.delimit("(", ")"):
            self.traverse(node.args)
        if node.returns:
            self.write(" -> ")
            self.traverse(node.returns)
        with self.block(extra=self.get_type_comment(node)):
            self._write_docstring_and_traverse_body(node)

    def visit_For(self, node):
        self._for_helper("for ", node)

    def visit_AsyncFor(self, node):
        self._for_helper("async for ", node)

    def _for_helper(self, fill, node):
        self.fill(fill)
        self.traverse(node.target)
        self.write(" in ")
        self.traverse(node.iter)
        with self.block(extra=self.get_type_comment(node)):
            self.traverse(node.body)
        if node.orelse:
            self.fill("else")
            with self.block():
                self.traverse(node.orelse)

    def visit_If(self, node):
        self.fill("if ")
        self.traverse(node.test)
        with self.block():
            self.traverse(node.body)
        # collapse nested ifs into equivalent elifs.
        while node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], If):
            node = node.orelse[0]
            self.fill("elif ")
            self.traverse(node.test)
            with self.block():
                self.traverse(node.body)
        # final else
        if node.orelse:
            self.fill("else")
            with self.block():
                self.traverse(node.orelse)

    def visit_While(self, node):
        self.fill("while ")
        self.traverse(node.test)
        with self.block():
            self.traverse(node.body)
        if node.orelse:
            self.fill("else")
            with self.block():
                self.traverse(node.orelse)

    def visit_With(self, node):
        self.fill("with ")
        self.interleave(lambda: self.write(", "), self.traverse, node.items)
        with self.block(extra=self.get_type_comment(node)):
            self.traverse(node.body)

    def visit_AsyncWith(self, node):
        self.fill("async with ")
        self.interleave(lambda: self.write(", "), self.traverse, node.items)
        with self.block(extra=self.get_type_comment(node)):
            self.traverse(node.body)

    def _str_literal_helper(
        self, string, *, quote_types=_ALL_QUOTES, escape_special_whitespace=False
    ):
        """Helper for writing string literals, minimizing escapes.
        Returns the tuple (string literal to write, possible quote types).
        """
        def escape_char(c):
            # \n and \t are non-printable, but we only escape them if
            # escape_special_whitespace is True
            if not escape_special_whitespace and c in "\n\t":
                return c
            # Always escape backslashes and other non-printable characters
            if c == "\\" or not c.isprintable():
                return c.encode("unicode_escape").decode("ascii")
            return c

        escaped_string = "".join(map(escape_char, string))
        possible_quotes = quote_types
        if "\n" in escaped_string:
            possible_quotes = [q for q in possible_quotes if q in _MULTI_QUOTES]
        possible_quotes = [q for q in possible_quotes if q not in escaped_string]
        if not possible_quotes:
            # If there aren't any possible_quotes, fallback to using repr
            # on the original string. Try to use a quote from quote_types,
            # e.g., so that we use triple quotes for docstrings.
            string = repr(string)
            quote = next((q for q in quote_types if string[0] in q), string[0])
            return string[1:-1], [quote]
        if escaped_string:
            # Sort so that we prefer '''"''' over """\""""
            possible_quotes.sort(key=lambda q: q[0] == escaped_string[-1])
            # If we're using triple quotes and we'd need to escape a final
            # quote, escape it
            if possible_quotes[0][0] == escaped_string[-1]:
                assert len(possible_quotes[0]) == 3
                escaped_string = escaped_string[:-1] + "\\" + escaped_string[-1]
        return escaped_string, possible_quotes

    def _write_str_avoiding_backslashes(self, string, *, quote_types=_ALL_QUOTES):
        """Write string literal value with a best effort attempt to avoid backslashes."""
        string, quote_types = self._str_literal_helper(string, quote_types=quote_types)
        quote_type = quote_types[0]
        self.write(f"{quote_type}{string}{quote_type}")

    def visit_JoinedStr(self, node):
        self.write("f")
        if self._avoid_backslashes:
            self._fstring_JoinedStr(node, self.buffer_writer)
            self._write_str_avoiding_backslashes(self.buffer)
            return

        # If we don't need to avoid backslashes globally (i.e., we only need
        # to avoid them inside FormattedValues), it's cosmetically preferred
        # to use escaped whitespace. That is, it's preferred to use backslashes
        # for cases like: f"{x}\n". To accomplish this, we keep track of what
        # in our buffer corresponds to FormattedValues and what corresponds to
        # Constant parts of the f-string, and allow escapes accordingly.
        buffer = []
        for value in node.values:
            meth = getattr(self, "_fstring_" + type(value).__name__)
            meth(value, self.buffer_writer)
            buffer.append((self.buffer, isinstance(value, Constant)))
        new_buffer = []
        quote_types = _ALL_QUOTES
        for value, is_constant in buffer:
            # Repeatedly narrow down the list of possible quote_types
            value, quote_types = self._str_literal_helper(
                value, quote_types=quote_types,
                escape_special_whitespace=is_constant
            )
            new_buffer.append(value)
        value = "".join(new_buffer)
        quote_type = quote_types[0]
        self.write(f"{quote_type}{value}{quote_type}")

    def visit_FormattedValue(self, node):
        self.write("f")
        self._fstring_FormattedValue(node, self.buffer_writer)
        self._write_str_avoiding_backslashes(self.buffer)

    def _fstring_JoinedStr(self, node, write):
        for value in node.values:
            meth = getattr(self, "_fstring_" + type(value).__name__)
            meth(value, write)

    def _fstring_Constant(self, node, write):
        if not isinstance(node.value, str):
            raise ValueError("Constants inside JoinedStr should be a string.")
        value = node.value.replace("{", "{{").replace("}", "}}")
        write(value)

    def _fstring_FormattedValue(self, node, write):
        write("{")
        unparser = type(self)(_avoid_backslashes=True)
        unparser.set_precedence(_Precedence.TEST.next(), node.value)
        expr = unparser.visit(node.value)
        if expr.startswith("{"):
            write(" ")  # Separate pair of opening brackets as "{ {"
        if "\\" in expr:
            raise ValueError("Unable to avoid backslash in f-string expression part")
        write(expr)
        if node.conversion != -1:
            conversion = chr(node.conversion)
            if conversion not in "sra":
                raise ValueError("Unknown f-string conversion.")
            write(f"!{conversion}")
        if node.format_spec:
            write(":")
            meth = getattr(self, "_fstring_" + type(node.format_spec).__name__)
            meth(node.format_spec, write)
        write("}")

    def visit_Name(self, node):
        self.write(node.id)

    def _write_docstring(self, node):
        self.fill()
        if node.kind == "u":
            self.write("u")
        self._write_str_avoiding_backslashes(node.value, quote_types=_MULTI_QUOTES)

    def _write_constant(self, value):
        if isinstance(value, (float, complex)):
            # Substitute overflowing decimal literal for AST infinities,
            # and inf - inf for NaNs.
            self.write(
                repr(value)
                .replace("inf", _INFSTR)
                .replace("nan", f"({_INFSTR}-{_INFSTR})")
            )
        elif self._avoid_backslashes and isinstance(value, str):
            self._write_str_avoiding_backslashes(value)
        else:
            self.write(repr(value))

    def visit_Constant(self, node):
        value = node.value
        if isinstance(value, tuple):
            with self.delimit("(", ")"):
                self.items_view(self._write_constant, value)
        elif value is ...:
            self.write("...")
        else:
            if node.kind == "u":
                self.write("u")
            self._write_constant(node.value)

    def visit_List(self, node):
        with self.delimit("[", "]"):
            self.interleave(lambda: self.write(", "), self.traverse, node.elts)

    def visit_ListComp(self, node):
        with self.delimit("[", "]"):
            self.traverse(node.elt)
            for gen in node.generators:
                self.traverse(gen)

    def visit_GeneratorExp(self, node):
        with self.delimit("(", ")"):
            self.traverse(node.elt)
            for gen in node.generators:
                self.traverse(gen)

    def visit_SetComp(self, node):
        with self.delimit("{", "}"):
            self.traverse(node.elt)
            for gen in node.generators:
                self.traverse(gen)

    def visit_DictComp(self, node):
        with self.delimit("{", "}"):
            self.traverse(node.key)
            self.write(": ")
            self.traverse(node.value)
            for gen in node.generators:
                self.traverse(gen)

    def visit_comprehension(self, node):
        if node.is_async:
            self.write(" async for ")
        else:
            self.write(" for ")
        self.set_precedence(_Precedence.TUPLE, node.target)
        self.traverse(node.target)
        self.write(" in ")
        self.set_precedence(_Precedence.TEST.next(), node.iter, *node.ifs)
        self.traverse(node.iter)
        for if_clause in node.ifs:
            self.write(" if ")
            self.traverse(if_clause)

    def visit_IfExp(self, node):
        with self.require_parens(_Precedence.TEST, node):
            self.set_precedence(_Precedence.TEST.next(), node.body, node.test)
            self.traverse(node.body)
            self.write(" if ")
            self.traverse(node.test)
            self.write(" else ")
            self.set_precedence(_Precedence.TEST, node.orelse)
            self.traverse(node.orelse)

    def visit_Set(self, node):
        if node.elts:
            with self.delimit("{", "}"):
                self.interleave(lambda: self.write(", "), self.traverse, node.elts)
        else:
            # `{}` would be interpreted as a dictionary literal, and
            # `set` might be shadowed. Thus:
            self.write('{*()}')

    def visit_Dict(self, node):
        def write_key_value_pair(k, v):
            self.traverse(k)
            self.write(": ")
            self.traverse(v)

        def write_item(item):
            k, v = item
            if k is None:
                # for dictionary unpacking operator in dicts {**{'y': 2}}
                # see PEP 448 for details
                self.write("**")
                self.set_precedence(_Precedence.EXPR, v)
                self.traverse(v)
            else:
                write_key_value_pair(k, v)

        with self.delimit("{", "}"):
            self.interleave(
                lambda: self.write(", "), write_item, zip(node.keys, node.values)
            )

    def visit_Tuple(self, node):
        with self.delimit("(", ")"):
            self.items_view(self.traverse, node.elts)

    unop = {"Invert": "~", "Not": "not", "UAdd": "+", "USub": "-"}
    unop_precedence = {
        "not": _Precedence.NOT,
        "~": _Precedence.FACTOR,
        "+": _Precedence.FACTOR,
        "-": _Precedence.FACTOR,
    }

    def visit_UnaryOp(self, node):
        operator = self.unop[node.op.__class__.__name__]
        operator_precedence = self.unop_precedence[operator]
        with self.require_parens(operator_precedence, node):
            self.write(operator)
            # factor prefixes (+, -, ~) shouldn't be seperated
            # from the value they belong, (e.g: +1 instead of + 1)
            if operator_precedence is not _Precedence.FACTOR:
                self.write(" ")
            self.set_precedence(operator_precedence, node.operand)
            self.traverse(node.operand)

    binop = {
        "Add": "+",
        "Sub": "-",
        "Mult": "*",
        "MatMult": "@",
        "Div": "/",
        "Mod": "%",
        "LShift": "<<",
        "RShift": ">>",
        "BitOr": "|",
        "BitXor": "^",
        "BitAnd": "&",
        "FloorDiv": "//",
        "Pow": "**",
    }

    binop_precedence = {
        "+": _Precedence.ARITH,
        "-": _Precedence.ARITH,
        "*": _Precedence.TERM,
        "@": _Precedence.TERM,
        "/": _Precedence.TERM,
        "%": _Precedence.TERM,
        "<<": _Precedence.SHIFT,
        ">>": _Precedence.SHIFT,
        "|": _Precedence.BOR,
        "^": _Precedence.BXOR,
        "&": _Precedence.BAND,
        "//": _Precedence.TERM,
        "**": _Precedence.POWER,
    }

    binop_rassoc = frozenset(("**",))
    def visit_BinOp(self, node):
        operator = self.binop[node.op.__class__.__name__]
        operator_precedence = self.binop_precedence[operator]
        with self.require_parens(operator_precedence, node):
            if operator in self.binop_rassoc:
                left_precedence = operator_precedence.next()
                right_precedence = operator_precedence
            else:
                left_precedence = operator_precedence
                right_precedence = operator_precedence.next()

            self.set_precedence(left_precedence, node.left)
            self.traverse(node.left)
            self.write(f" {operator} ")
            self.set_precedence(right_precedence, node.right)
            self.traverse(node.right)

    cmpops = {
        "Eq": "==",
        "NotEq": "!=",
        "Lt": "<",
        "LtE": "<=",
        "Gt": ">",
        "GtE": ">=",
        "Is": "is",
        "IsNot": "is not",
        "In": "in",
        "NotIn": "not in",
    }

    def visit_Compare(self, node):
        with self.require_parens(_Precedence.CMP, node):
            self.set_precedence(_Precedence.CMP.next(), node.left, *node.comparators)
            self.traverse(node.left)
            for o, e in zip(node.ops, node.comparators):
                self.write(" " + self.cmpops[o.__class__.__name__] + " ")
                self.traverse(e)

    boolops = {"And": "and", "Or": "or"}
    boolop_precedence = {"and": _Precedence.AND, "or": _Precedence.OR}

    def visit_BoolOp(self, node):
        operator = self.boolops[node.op.__class__.__name__]
        operator_precedence = self.boolop_precedence[operator]

        def increasing_level_traverse(node):
            nonlocal operator_precedence
            operator_precedence = operator_precedence.next()
            self.set_precedence(operator_precedence, node)
            self.traverse(node)

        with self.require_parens(operator_precedence, node):
            s = f" {operator} "
            self.interleave(lambda: self.write(s), increasing_level_traverse, node.values)

    def visit_Attribute(self, node):
        self.set_precedence(_Precedence.ATOM, node.value)
        self.traverse(node.value)
        # Special case: 3.__abs__() is a syntax error, so if node.value
        # is an integer literal then we need to either parenthesize
        # it or add an extra space to get 3 .__abs__().
        if isinstance(node.value, Constant) and isinstance(node.value.value, int):
            self.write(" ")
        self.write(".")
        self.write(node.attr)

    def visit_Call(self, node):
        self.set_precedence(_Precedence.ATOM, node.func)
        self.traverse(node.func)
        with self.delimit("(", ")"):
            comma = False
            for e in node.args:
                if comma:
                    self.write(", ")
                else:
                    comma = True
                self.traverse(e)
            for e in node.keywords:
                if comma:
                    self.write(", ")
                else:
                    comma = True
                self.traverse(e)

    def visit_Subscript(self, node):
        def is_simple_tuple(slice_value):
            # when unparsing a non-empty tuple, the parentheses can be safely
            # omitted if there aren't any elements that explicitly requires
            # parentheses (such as starred expressions).
            return (
                isinstance(slice_value, Tuple)
                and slice_value.elts
                and not any(isinstance(elt, Starred) for elt in slice_value.elts)
            )

        self.set_precedence(_Precedence.ATOM, node.value)
        self.traverse(node.value)
        with self.delimit("[", "]"):
            if is_simple_tuple(node.slice):
                self.items_view(self.traverse, node.slice.elts)
            else:
                self.traverse(node.slice)

    def visit_Starred(self, node):
        self.write("*")
        self.set_precedence(_Precedence.EXPR, node.value)
        self.traverse(node.value)

    def visit_Ellipsis(self, node):
        self.write("...")

    def visit_Slice(self, node):
        if node.lower:
            self.traverse(node.lower)
        self.write(":")
        if node.upper:
            self.traverse(node.upper)
        if node.step:
            self.write(":")
            self.traverse(node.step)

    def visit_Match(self, node):
        self.fill("match ")
        self.traverse(node.subject)
        with self.block():
            for case in node.cases:
                self.traverse(case)

    def visit_arg(self, node):
        self.write(node.arg)
        if node.annotation:
            self.write(": ")
            self.traverse(node.annotation)

    def visit_arguments(self, node):
        first = True
        # normal arguments
        all_args = node.posonlyargs + node.args
        defaults = [None] * (len(all_args) - len(node.defaults)) + node.defaults
        for index, elements in enumerate(zip(all_args, defaults), 1):
            a, d = elements
            if first:
                first = False
            else:
                self.write(", ")
            self.traverse(a)
            if d:
                self.write("=")
                self.traverse(d)
            if index == len(node.posonlyargs):
                self.write(", /")

        # varargs, or bare '*' if no varargs but keyword-only arguments present
        if node.vararg or node.kwonlyargs:
            if first:
                first = False
            else:
                self.write(", ")
            self.write("*")
            if node.vararg:
                self.write(node.vararg.arg)
                if node.vararg.annotation:
                    self.write(": ")
                    self.traverse(node.vararg.annotation)

        # keyword-only arguments
        if node.kwonlyargs:
            for a, d in zip(node.kwonlyargs, node.kw_defaults):
                self.write(", ")
                self.traverse(a)
                if d:
                    self.write("=")
                    self.traverse(d)

        # kwargs
        if node.kwarg:
            if first:
                first = False
            else:
                self.write(", ")
            self.write("**" + node.kwarg.arg)
            if node.kwarg.annotation:
                self.write(": ")
                self.traverse(node.kwarg.annotation)

    def visit_keyword(self, node):
        if node.arg is None:
            self.write("**")
        else:
            self.write(node.arg)
            self.write("=")
        self.traverse(node.value)

    def visit_Lambda(self, node):
        with self.require_parens(_Precedence.TEST, node):
            self.write("lambda ")
            self.traverse(node.args)
            self.write(": ")
            self.set_precedence(_Precedence.TEST, node.body)
            self.traverse(node.body)

    def visit_alias(self, node):
        self.write(node.name)
        if node.asname:
            self.write(" as " + node.asname)

    def visit_withitem(self, node):
        self.traverse(node.context_expr)
        if node.optional_vars:
            self.write(" as ")
            self.traverse(node.optional_vars)

    def visit_match_case(self, node):
        self.fill("case ")
        self.traverse(node.pattern)
        if node.guard:
            self.write(" if ")
            self.traverse(node.guard)
        with self.block():
            self.traverse(node.body)

    def visit_MatchValue(self, node):
        self.traverse(node.value)

    def visit_MatchSingleton(self, node):
        self._write_constant(node.value)

    def visit_MatchSequence(self, node):
        with self.delimit("[", "]"):
            self.interleave(
                lambda: self.write(", "), self.traverse, node.patterns
            )

    def visit_MatchStar(self, node):
        name = node.name
        if name is None:
            name = "_"
        self.write(f"*{name}")

    def visit_MatchMapping(self, node):
        def write_key_pattern_pair(pair):
            k, p = pair
            self.traverse(k)
            self.write(": ")
            self.traverse(p)

        with self.delimit("{", "}"):
            keys = node.keys
            self.interleave(
                lambda: self.write(", "),
                write_key_pattern_pair,
                zip(keys, node.patterns, strict=True),
            )
            rest = node.rest
            if rest is not None:
                if keys:
                    self.write(", ")
                self.write(f"**{rest}")

    def visit_MatchClass(self, node):
        self.set_precedence(_Precedence.ATOM, node.cls)
        self.traverse(node.cls)
        with self.delimit("(", ")"):
            patterns = node.patterns
            self.interleave(
                lambda: self.write(", "), self.traverse, patterns
            )
            attrs = node.kwd_attrs
            if attrs:
                def write_attr_pattern(pair):
                    attr, pattern = pair
                    self.write(f"{attr}=")
                    self.traverse(pattern)

                if patterns:
                    self.write(", ")
                self.interleave(
                    lambda: self.write(", "),
                    write_attr_pattern,
                    zip(attrs, node.kwd_patterns, strict=True),
                )

    def visit_MatchAs(self, node):
        name = node.name
        pattern = node.pattern
        if name is None:
            self.write("_")
        elif pattern is None:
            self.write(node.name)
        else:
            with self.require_parens(_Precedence.TEST, node):
                self.set_precedence(_Precedence.BOR, node.pattern)
                self.traverse(node.pattern)
                self.write(f" as {node.name}")

    def visit_MatchOr(self, node):
        with self.require_parens(_Precedence.BOR, node):
            self.set_precedence(_Precedence.BOR.next(), *node.patterns)
            self.interleave(lambda: self.write(" | "), self.traverse, node.patterns)

def unparse(ast_obj):
    unparser = _Unparser()
    return unparser.visit(ast_obj)


def main():
    import argparse

    parser = argparse.ArgumentParser(prog='python -m ast')
    parser.add_argument('infile', type=argparse.FileType(mode='rb'), nargs='?',
                        default='-',
                        help='the file to parse; defaults to stdin')
    parser.add_argument('-m', '--mode', default='exec',
                        choices=('exec', 'single', 'eval', 'func_type'),
                        help='specify what kind of code must be parsed')
    parser.add_argument('--no-type-comments', default=True, action='store_false',
                        help="don't add information about type comments")
    parser.add_argument('-a', '--include-attributes', action='store_true',
                        help='include attributes such as line numbers and '
                             'column offsets')
    parser.add_argument('-i', '--indent', type=int, default=3,
                        help='indentation of nodes (number of spaces)')
    args = parser.parse_args()

    with args.infile as infile:
        source = infile.read()
    tree = parse(source, args.infile.name, args.mode, type_comments=args.no_type_comments)
    print(dump(tree, include_attributes=args.include_attributes, indent=args.indent))

if __name__ == '__main__':
    main()
<|endoftext|>#! /usr/bin/env python3
"""An RFC 5321 smtp proxy with optional RFC 1870 and RFC 6531 extensions.

Usage: %(program)s [options] [localhost:localport [remotehost:remoteport]]

Options:

    --nosetuid
    -n
        This program generally tries to setuid `nobody', unless this flag is
        set.  The setuid call will fail if this program is not run as root (in
        which case, use this flag).

    --version
    -V
        Print the version number and exit.

    --class classname
    -c classname
        Use `classname' as the concrete SMTP proxy class.  Uses `PureProxy' by
        default.

    --size limit
    -s limit
        Restrict the total size of the incoming message to "limit" number of
        bytes via the RFC 1870 SIZE extension.  Defaults to 33554432 bytes.

    --smtputf8
    -u
        Enable the SMTPUTF8 extension and behave as an RFC 6531 smtp proxy.

    --debug
    -d
        Turn on debugging prints.

    --help
    -h
        Print this message and exit.

Version: %(__version__)s

If localhost is not given then `localhost' is used, and if localport is not
given then 8025 is used.  If remotehost is not given then `localhost' is used,
and if remoteport is not given, then 25 is used.
"""

# Overview:
#
# This file implements the minimal SMTP protocol as defined in RFC 5321.  It
# has a hierarchy of classes which implement the backend functionality for the
# smtpd.  A number of classes are provided:
#
#   SMTPServer - the base class for the backend.  Raises NotImplementedError
#   if you try to use it.
#
#   DebuggingServer - simply prints each message it receives on stdout.
#
#   PureProxy - Proxies all messages to a real smtpd which does final
#   delivery.  One known problem with this class is that it doesn't handle
#   SMTP errors from the backend server at all.  This should be fixed
#   (contributions are welcome!).
#
#   MailmanProxy - An experimental hack to work with GNU Mailman
#   <www.list.org>.  Using this server as your real incoming smtpd, your
#   mailhost will automatically recognize and accept mail destined to Mailman
#   lists when those lists are created.  Every message not destined for a list
#   gets forwarded to a real backend smtpd, as with PureProxy.  Again, errors
#   are not handled correctly yet.
#
#
# Author: Barry Warsaw <barry@python.org>
#
# TODO:
#
# - support mailbox delivery
# - alias files
# - Handle more ESMTP extensions
# - handle error codes from the backend smtpd

import sys
import os
import errno
import getopt
import time
import socket
import collections
from warnings import warn
from email._header_value_parser import get_addr_spec, get_angle_addr

__all__ = [
    "SMTPChannel", "SMTPServer", "DebuggingServer", "PureProxy",
    "MailmanProxy",
]

warn(
    'The smtpd module is deprecated and unmaintained and will be removed '
    'in Python 3.12.  Please see aiosmtpd '
    '(https://aiosmtpd.readthedocs.io/) for the recommended replacement.',
    DeprecationWarning,
    stacklevel=2)


# These are imported after the above warning so that users get the correct
# deprecation warning.
import asyncore
import asynchat


program = sys.argv[0]
__version__ = 'Python SMTP proxy version 0.3'


class Devnull:
    def write(self, msg): pass
    def flush(self): pass


DEBUGSTREAM = Devnull()
NEWLINE = '\n'
COMMASPACE = ', '
DATA_SIZE_DEFAULT = 33554432


def usage(code, msg=''):
    print(__doc__ % globals(), file=sys.stderr)
    if msg:
        print(msg, file=sys.stderr)
    sys.exit(code)


class SMTPChannel(asynchat.async_chat):
    COMMAND = 0
    DATA = 1

    command_size_limit = 512
    command_size_limits = collections.defaultdict(lambda x=command_size_limit: x)

    @property
    def max_command_size_limit(self):
        try:
            return max(self.command_size_limits.values())
        except ValueError:
            return self.command_size_limit

    def __init__(self, server, conn, addr, data_size_limit=DATA_SIZE_DEFAULT,
                 map=None, enable_SMTPUTF8=False, decode_data=False):
        asynchat.async_chat.__init__(self, conn, map=map)
        self.smtp_server = server
        self.conn = conn
        self.addr = addr
        self.data_size_limit = data_size_limit
        self.enable_SMTPUTF8 = enable_SMTPUTF8
        self._decode_data = decode_data
        if enable_SMTPUTF8 and decode_data:
            raise ValueError("decode_data and enable_SMTPUTF8 cannot"
                             " be set to True at the same time")
        if decode_data:
            self._emptystring = ''
            self._linesep = '\r\n'
            self._dotsep = '.'
            self._newline = NEWLINE
        else:
            self._emptystring = b''
            self._linesep = b'\r\n'
            self._dotsep = ord(b'.')
            self._newline = b'\n'
        self._set_rset_state()
        self.seen_greeting = ''
        self.extended_smtp = False
        self.command_size_limits.clear()
        self.fqdn = socket.getfqdn()
        try:
            self.peer = conn.getpeername()
        except OSError as err:
            # a race condition  may occur if the other end is closing
            # before we can get the peername
            self.close()
            if err.errno != errno.ENOTCONN:
                raise
            return
        print('Peer:', repr(self.peer), file=DEBUGSTREAM)
        self.push('220 %s %s' % (self.fqdn, __version__))

    def _set_post_data_state(self):
        """Reset state variables to their post-DATA state."""
        self.smtp_state = self.COMMAND
        self.mailfrom = None
        self.rcpttos = []
        self.require_SMTPUTF8 = False
        self.num_bytes = 0
        self.set_terminator(b'\r\n')

    def _set_rset_state(self):
        """Reset all state variables except the greeting."""
        self._set_post_data_state()
        self.received_data = ''
        self.received_lines = []


    # properties for backwards-compatibility
    @property
    def __server(self):
        warn("Access to __server attribute on SMTPChannel is deprecated, "
            "use 'smtp_server' instead", DeprecationWarning, 2)
        return self.smtp_server
    @__server.setter
    def __server(self, value):
        warn("Setting __server attribute on SMTPChannel is deprecated, "
            "set 'smtp_server' instead", DeprecationWarning, 2)
        self.smtp_server = value

    @property
    def __line(self):
        warn("Access to __line attribute on SMTPChannel is deprecated, "
            "use 'received_lines' instead", DeprecationWarning, 2)
        return self.received_lines
    @__line.setter
    def __line(self, value):
        warn("Setting __line attribute on SMTPChannel is deprecated, "
            "set 'received_lines' instead", DeprecationWarning, 2)
        self.received_lines = value

    @property
    def __state(self):
        warn("Access to __state attribute on SMTPChannel is deprecated, "
            "use 'smtp_state' instead", DeprecationWarning, 2)
        return self.smtp_state
    @__state.setter
    def __state(self, value):
        warn("Setting __state attribute on SMTPChannel is deprecated, "
            "set 'smtp_state' instead", DeprecationWarning, 2)
        self.smtp_state = value

    @property
    def __greeting(self):
        warn("Access to __greeting attribute on SMTPChannel is deprecated, "
            "use 'seen_greeting' instead", DeprecationWarning, 2)
        return self.seen_greeting
    @__greeting.setter
    def __greeting(self, value):
        warn("Setting __greeting attribute on SMTPChannel is deprecated, "
            "set 'seen_greeting' instead", DeprecationWarning, 2)
        self.seen_greeting = value

    @property
    def __mailfrom(self):
        warn("Access to __mailfrom attribute on SMTPChannel is deprecated, "
            "use 'mailfrom' instead", DeprecationWarning, 2)
        return self.mailfrom
    @__mailfrom.setter
    def __mailfrom(self, value):
        warn("Setting __mailfrom attribute on SMTPChannel is deprecated, "
            "set 'mailfrom' instead", DeprecationWarning, 2)
        self.mailfrom = value

    @property
    def __rcpttos(self):
        warn("Access to __rcpttos attribute on SMTPChannel is deprecated, "
            "use 'rcpttos' instead", DeprecationWarning, 2)
        return self.rcpttos
    @__rcpttos.setter
    def __rcpttos(self, value):
        warn("Setting __rcpttos attribute on SMTPChannel is deprecated, "
            "set 'rcpttos' instead", DeprecationWarning, 2)
        self.rcpttos = value

    @property
    def __data(self):
        warn("Access to __data attribute on SMTPChannel is deprecated, "
            "use 'received_data' instead", DeprecationWarning, 2)
        return self.received_data
    @__data.setter
    def __data(self, value):
        warn("Setting __data attribute on SMTPChannel is deprecated, "
            "set 'received_data' instead", DeprecationWarning, 2)
        self.received_data = value

    @property
    def __fqdn(self):
        warn("Access to __fqdn attribute on SMTPChannel is deprecated, "
            "use 'fqdn' instead", DeprecationWarning, 2)
        return self.fqdn
    @__fqdn.setter
    def __fqdn(self, value):
        warn("Setting __fqdn attribute on SMTPChannel is deprecated, "
            "set 'fqdn' instead", DeprecationWarning, 2)
        self.fqdn = value

    @property
    def __peer(self):
        warn("Access to __peer attribute on SMTPChannel is deprecated, "
            "use 'peer' instead", DeprecationWarning, 2)
        return self.peer
    @__peer.setter
    def __peer(self, value):
        warn("Setting __peer attribute on SMTPChannel is deprecated, "
            "set 'peer' instead", DeprecationWarning, 2)
        self.peer = value

    @property
    def __conn(self):
        warn("Access to __conn attribute on SMTPChannel is deprecated, "
            "use 'conn' instead", DeprecationWarning, 2)
        return self.conn
    @__conn.setter
    def __conn(self, value):
        warn("Setting __conn attribute on SMTPChannel is deprecated, "
            "set 'conn' instead", DeprecationWarning, 2)
        self.conn = value

    @property
    def __addr(self):
        warn("Access to __addr attribute on SMTPChannel is deprecated, "
            "use 'addr' instead", DeprecationWarning, 2)
        return self.addr
    @__addr.setter
    def __addr(self, value):
        warn("Setting __addr attribute on SMTPChannel is deprecated, "
            "set 'addr' instead", DeprecationWarning, 2)
        self.addr = value

    # Overrides base class for convenience.
    def push(self, msg):
        asynchat.async_chat.push(self, bytes(
            msg + '\r\n', 'utf-8' if self.require_SMTPUTF8 else 'ascii'))

    # Implementation of base class abstract method
    def collect_incoming_data(self, data):
        limit = None
        if self.smtp_state == self.COMMAND:
            limit = self.max_command_size_limit
        elif self.smtp_state == self.DATA:
            limit = self.data_size_limit
        if limit and self.num_bytes > limit:
            return
        elif limit:
            self.num_bytes += len(data)
        if self._decode_data:
            self.received_lines.append(str(data, 'utf-8'))
        else:
            self.received_lines.append(data)

    # Implementation of base class abstract method
    def found_terminator(self):
        line = self._emptystring.join(self.received_lines)
        print('Data:', repr(line), file=DEBUGSTREAM)
        self.received_lines = []
        if self.smtp_state == self.COMMAND:
            sz, self.num_bytes = self.num_bytes, 0
            if not line:
                self.push('500 Error: bad syntax')
                return
            if not self._decode_data:
                line = str(line, 'utf-8')
            i = line.find(' ')
            if i < 0:
                command = line.upper()
                arg = None
            else:
                command = line[:i].upper()
                arg = line[i+1:].strip()
            max_sz = (self.command_size_limits[command]
                        if self.extended_smtp else self.command_size_limit)
            if sz > max_sz:
                self.push('500 Error: line too long')
                return
            method = getattr(self, 'smtp_' + command, None)
            if not method:
                self.push('500 Error: command "%s" not recognized' % command)
                return
            method(arg)
            return
        else:
            if self.smtp_state != self.DATA:
                self.push('451 Internal confusion')
                self.num_bytes = 0
                return
            if self.data_size_limit and self.num_bytes > self.data_size_limit:
                self.push('552 Error: Too much mail data')
                self.num_bytes = 0
                return
            # Remove extraneous carriage returns and de-transparency according
            # to RFC 5321, Section 4.5.2.
            data = []
            for text in line.split(self._linesep):
                if text and text[0] == self._dotsep:
                    data.append(text[1:])
                else:
                    data.append(text)
            self.received_data = self._newline.join(data)
            args = (self.peer, self.mailfrom, self.rcpttos, self.received_data)
            kwargs = {}
            if not self._decode_data:
                kwargs = {
                    'mail_options': self.mail_options,
                    'rcpt_options': self.rcpt_options,
                }
            status = self.smtp_server.process_message(*args, **kwargs)
            self._set_post_data_state()
            if not status:
                self.push('250 OK')
            else:
                self.push(status)

    # SMTP and ESMTP commands
    def smtp_HELO(self, arg):
        if not arg:
            self.push('501 Syntax: HELO hostname')
            return
        # See issue #21783 for a discussion of this behavior.
        if self.seen_greeting:
            self.push('503 Duplicate HELO/EHLO')
            return
        self._set_rset_state()
        self.seen_greeting = arg
        self.push('250 %s' % self.fqdn)

    def smtp_EHLO(self, arg):
        if not arg:
            self.push('501 Syntax: EHLO hostname')
            return
        # See issue #21783 for a discussion of this behavior.
        if self.seen_greeting:
            self.push('503 Duplicate HELO/EHLO')
            return
        self._set_rset_state()
        self.seen_greeting = arg
        self.extended_smtp = True
        self.push('250-%s' % self.fqdn)
        if self.data_size_limit:
            self.push('250-SIZE %s' % self.data_size_limit)
            self.command_size_limits['MAIL'] += 26
        if not self._decode_data:
            self.push('250-8BITMIME')
        if self.enable_SMTPUTF8:
            self.push('250-SMTPUTF8')
            self.command_size_limits['MAIL'] += 10
        self.push('250 HELP')

    def smtp_NOOP(self, arg):
        if arg:
            self.push('501 Syntax: NOOP')
        else:
            self.push('250 OK')

    def smtp_QUIT(self, arg):
        # args is ignored
        self.push('221 Bye')
        self.close_when_done()

    def _strip_command_keyword(self, keyword, arg):
        keylen = len(keyword)
        if arg[:keylen].upper() == keyword:
            return arg[keylen:].strip()
        return ''

    def _getaddr(self, arg):
        if not arg:
            return '', ''
        if arg.lstrip().startswith('<'):
            address, rest = get_angle_addr(arg)
        else:
            address, rest = get_addr_spec(arg)
        if not address:
            return address, rest
        return address.addr_spec, rest

    def _getparams(self, params):
        # Return params as dictionary. Return None if not all parameters
        # appear to be syntactically valid according to RFC 1869.
        result = {}
        for param in params:
            param, eq, value = param.partition('=')
            if not param.isalnum() or eq and not value:
                return None
            result[param] = value if eq else True
        return result

    def smtp_HELP(self, arg):
        if arg:
            extended = ' [SP <mail-parameters>]'
            lc_arg = arg.upper()
            if lc_arg == 'EHLO':
                self.push('250 Syntax: EHLO hostname')
            elif lc_arg == 'HELO':
                self.push('250 Syntax: HELO hostname')
            elif lc_arg == 'MAIL':
                msg = '250 Syntax: MAIL FROM: <address>'
                if self.extended_smtp:
                    msg += extended
                self.push(msg)
            elif lc_arg == 'RCPT':
                msg = '250 Syntax: RCPT TO: <address>'
                if self.extended_smtp:
                    msg += extended
                self.push(msg)
            elif lc_arg == 'DATA':
                self.push('250 Syntax: DATA')
            elif lc_arg == 'RSET':
                self.push('250 Syntax: RSET')
            elif lc_arg == 'NOOP':
                self.push('250 Syntax: NOOP')
            elif lc_arg == 'QUIT':
                self.push('250 Syntax: QUIT')
            elif lc_arg == 'VRFY':
                self.push('250 Syntax: VRFY <address>')
            else:
                self.push('501 Supported commands: EHLO HELO MAIL RCPT '
                          'DATA RSET NOOP QUIT VRFY')
        else:
            self.push('250 Supported commands: EHLO HELO MAIL RCPT DATA '
                      'RSET NOOP QUIT VRFY')

    def smtp_VRFY(self, arg):
        if arg:
            address, params = self._getaddr(arg)
            if address:
                self.push('252 Cannot VRFY user, but will accept message '
                          'and attempt delivery')
            else:
                self.push('502 Could not VRFY %s' % arg)
        else:
            self.push('501 Syntax: VRFY <address>')

    def smtp_MAIL(self, arg):
        if not self.seen_greeting:
            self.push('503 Error: send HELO first')
            return
        print('===> MAIL', arg, file=DEBUGSTREAM)
        syntaxerr = '501 Syntax: MAIL FROM: <address>'
        if self.extended_smtp:
            syntaxerr += ' [SP <mail-parameters>]'
        if arg is None:
            self.push(syntaxerr)
            return
        arg = self._strip_command_keyword('FROM:', arg)
        address, params = self._getaddr(arg)
        if not address:
            self.push(syntaxerr)
            return
        if not self.extended_smtp and params:
            self.push(syntaxerr)
            return
        if self.mailfrom:
            self.push('503 Error: nested MAIL command')
            return
        self.mail_options = params.upper().split()
        params = self._getparams(self.mail_options)
        if params is None:
            self.push(syntaxerr)
            return
        if not self._decode_data:
            body = params.pop('BODY', '7BIT')
            if body not in ['7BIT', '8BITMIME']:
                self.push('501 Error: BODY can only be one of 7BIT, 8BITMIME')
                return
        if self.enable_SMTPUTF8:
            smtputf8 = params.pop('SMTPUTF8', False)
            if smtputf8 is True:
                self.require_SMTPUTF8 = True
            elif smtputf8 is not False:
                self.push('501 Error: SMTPUTF8 takes no arguments')
                return
        size = params.pop('SIZE', None)
        if size:
            if not size.isdigit():
                self.push(syntaxerr)
                return
            elif self.data_size_limit and int(size) > self.data_size_limit:
                self.push('552 Error: message size exceeds fixed maximum message size')
                return
        if len(params.keys()) > 0:
            self.push('555 MAIL FROM parameters not recognized or not implemented')
            return
        self.mailfrom = address
        print('sender:', self.mailfrom, file=DEBUGSTREAM)
        self.push('250 OK')

    def smtp_RCPT(self, arg):
        if not self.seen_greeting:
            self.push('503 Error: send HELO first');
            return
        print('===> RCPT', arg, file=DEBUGSTREAM)
        if not self.mailfrom:
            self.push('503 Error: need MAIL command')
            return
        syntaxerr = '501 Syntax: RCPT TO: <address>'
        if self.extended_smtp:
            syntaxerr += ' [SP <mail-parameters>]'
        if arg is None:
            self.push(syntaxerr)
            return
        arg = self._strip_command_keyword('TO:', arg)
        address, params = self._getaddr(arg)
        if not address:
            self.push(syntaxerr)
            return
        if not self.extended_smtp and params:
            self.push(syntaxerr)
            return
        self.rcpt_options = params.upper().split()
        params = self._getparams(self.rcpt_options)
        if params is None:
            self.push(syntaxerr)
            return
        # XXX currently there are no options we recognize.
        if len(params.keys()) > 0:
            self.push('555 RCPT TO parameters not recognized or not implemented')
            return
        self.rcpttos.append(address)
        print('recips:', self.rcpttos, file=DEBUGSTREAM)
        self.push('250 OK')

    def smtp_RSET(self, arg):
        if arg:
            self.push('501 Syntax: RSET')
            return
        self._set_rset_state()
        self.push('250 OK')

    def smtp_DATA(self, arg):
        if not self.seen_greeting:
            self.push('503 Error: send HELO first');
            return
        if not self.rcpttos:
            self.push('503 Error: need RCPT command')
            return
        if arg:
            self.push('501 Syntax: DATA')
            return
        self.smtp_state = self.DATA
        self.set_terminator(b'\r\n.\r\n')
        self.push('354 End data with <CR><LF>.<CR><LF>')

    # Commands that have not been implemented
    def smtp_EXPN(self, arg):
        self.push('502 EXPN not implemented')


class SMTPServer(asyncore.dispatcher):
    # SMTPChannel class to use for managing client connections
    channel_class = SMTPChannel

    def __init__(self, localaddr, remoteaddr,
                 data_size_limit=DATA_SIZE_DEFAULT, map=None,
                 enable_SMTPUTF8=False, decode_data=False):
        self._localaddr = localaddr
        self._remoteaddr = remoteaddr
        self.data_size_limit = data_size_limit
        self.enable_SMTPUTF8 = enable_SMTPUTF8
        self._decode_data = decode_data
        if enable_SMTPUTF8 and decode_data:
            raise ValueError("decode_data and enable_SMTPUTF8 cannot"
                             " be set to True at the same time")
        asyncore.dispatcher.__init__(self, map=map)
        try:
            gai_results = socket.getaddrinfo(*localaddr,
                                             type=socket.SOCK_STREAM)
            self.create_socket(gai_results[0][0], gai_results[0][1])
            # try to re-use a server port if possible
            self.set_reuse_addr()
            self.bind(localaddr)
            self.listen(5)
        except:
            self.close()
            raise
        else:
            print('%s started at %s\n\tLocal addr: %s\n\tRemote addr:%s' % (
                self.__class__.__name__, time.ctime(time.time()),
                localaddr, remoteaddr), file=DEBUGSTREAM)

    def handle_accepted(self, conn, addr):
        print('Incoming connection from %s' % repr(addr), file=DEBUGSTREAM)
        channel = self.channel_class(self,
                                     conn,
                                     addr,
                                     self.data_size_limit,
                                     self._map,
                                     self.enable_SMTPUTF8,
                                     self._decode_data)

    # API for "doing something useful with the message"
    def process_message(self, peer, mailfrom, rcpttos, data, **kwargs):
        """Override this abstract method to handle messages from the client.

        peer is a tuple containing (ipaddr, port) of the client that made the
        socket connection to our smtp port.

        mailfrom is the raw address the client claims the message is coming
        from.

        rcpttos is a list of raw addresses the client wishes to deliver the
        message to.

        data is a string containing the entire full text of the message,
        headers (if supplied) and all.  It has been `de-transparencied'
        according to RFC 821, Section 4.5.2.  In other words, a line
        containing a `.' followed by other text has had the leading dot
        removed.

        kwargs is a dictionary containing additional information.  It is
        empty if decode_data=True was given as init parameter, otherwise
        it will contain the following keys:
            'mail_options': list of parameters to the mail command.  All
                            elements are uppercase strings.  Example:
                            ['BODY=8BITMIME', 'SMTPUTF8'].
            'rcpt_options': same, for the rcpt command.

        This function should return None for a normal `250 Ok' response;
        otherwise, it should return the desired response string in RFC 821
        format.

        """
        raise NotImplementedError


class DebuggingServer(SMTPServer):

    def _print_message_content(self, peer, data):
        inheaders = 1
        lines = data.splitlines()
        for line in lines:
            # headers first
            if inheaders and not line:
                peerheader = 'X-Peer: ' + peer[0]
                if not isinstance(data, str):
                    # decoded_data=false; make header match other binary output
                    peerheader = repr(peerheader.encode('utf-8'))
                print(peerheader)
                inheaders = 0
            if not isinstance(data, str):
                # Avoid spurious 'str on bytes instance' warning.
                line = repr(line)
            print(line)

    def process_message(self, peer, mailfrom, rcpttos, data, **kwargs):
        print('---------- MESSAGE FOLLOWS ----------')
        if kwargs:
            if kwargs.get('mail_options'):
                print('mail options: %s' % kwargs['mail_options'])
            if kwargs.get('rcpt_options'):
                print('rcpt options: %s\n' % kwargs['rcpt_options'])
        self._print_message_content(peer, data)
        print('------------ END MESSAGE ------------')


class PureProxy(SMTPServer):
    def __init__(self, *args, **kwargs):
        if 'enable_SMTPUTF8' in kwargs and kwargs['enable_SMTPUTF8']:
            raise ValueError("PureProxy does not support SMTPUTF8.")
        super(PureProxy, self).__init__(*args, **kwargs)

    def process_message(self, peer, mailfrom, rcpttos, data):
        lines = data.split('\n')
        # Look for the last header
        i = 0
        for line in lines:
            if not line:
                break
            i += 1
        lines.insert(i, 'X-Peer: %s' % peer[0])
        data = NEWLINE.join(lines)
        refused = self._deliver(mailfrom, rcpttos, data)
        # TBD: what to do with refused addresses?
        print('we got some refusals:', refused, file=DEBUGSTREAM)

    def _deliver(self, mailfrom, rcpttos, data):
        import smtplib
        refused = {}
        try:
            s = smtplib.SMTP()
            s.connect(self._remoteaddr[0], self._remoteaddr[1])
            try:
                refused = s.sendmail(mailfrom, rcpttos, data)
            finally:
                s.quit()
        except smtplib.SMTPRecipientsRefused as e:
            print('got SMTPRecipientsRefused', file=DEBUGSTREAM)
            refused = e.recipients
        except (OSError, smtplib.SMTPException) as e:
            print('got', e.__class__, file=DEBUGSTREAM)
            # All recipients were refused.  If the exception had an associated
            # error code, use it.  Otherwise,fake it with a non-triggering
            # exception code.
            errcode = getattr(e, 'smtp_code', -1)
            errmsg = getattr(e, 'smtp_error', 'ignore')
            for r in rcpttos:
                refused[r] = (errcode, errmsg)
        return refused


class MailmanProxy(PureProxy):
    def __init__(self, *args, **kwargs):
        warn('MailmanProxy is deprecated and will be removed '
             'in future', DeprecationWarning, 2)
        if 'enable_SMTPUTF8' in kwargs and kwargs['enable_SMTPUTF8']:
            raise ValueError("MailmanProxy does not support SMTPUTF8.")
        super(PureProxy, self).__init__(*args, **kwargs)

    def process_message(self, peer, mailfrom, rcpttos, data):
        from io import StringIO
        from Mailman import Utils
        from Mailman import Message
        from Mailman import MailList
        # If the message is to a Mailman mailing list, then we'll invoke the
        # Mailman script directly, without going through the real smtpd.
        # Otherwise we'll forward it to the local proxy for disposition.
        listnames = []
        for rcpt in rcpttos:
            local = rcpt.lower().split('@')[0]
            # We allow the following variations on the theme
            #   listname
            #   listname-admin
            #   listname-owner
            #   listname-request
            #   listname-join
            #   listname-leave
            parts = local.split('-')
            if len(parts) > 2:
                continue
            listname = parts[0]
            if len(parts) == 2:
                command = parts[1]
            else:
                command = ''
            if not Utils.list_exists(listname) or command not in (
                    '', 'admin', 'owner', 'request', 'join', 'leave'):
                continue
            listnames.append((rcpt, listname, command))
        # Remove all list recipients from rcpttos and forward what we're not
        # going to take care of ourselves.  Linear removal should be fine
        # since we don't expect a large number of recipients.
        for rcpt, listname, command in listnames:
            rcpttos.remove(rcpt)
        # If there's any non-list destined recipients left,
        print('forwarding recips:', ' '.join(rcpttos), file=DEBUGSTREAM)
        if rcpttos:
            refused = self._deliver(mailfrom, rcpttos, data)
            # TBD: what to do with refused addresses?
            print('we got refusals:', refused, file=DEBUGSTREAM)
        # Now deliver directly to the list commands
        mlists = {}
        s = StringIO(data)
        msg = Message.Message(s)
        # These headers are required for the proper execution of Mailman.  All
        # MTAs in existence seem to add these if the original message doesn't
        # have them.
        if not msg.get('from'):
            msg['From'] = mailfrom
        if not msg.get('date'):
            msg['Date'] = time.ctime(time.time())
        for rcpt, listname, command in listnames:
            print('sending message to', rcpt, file=DEBUGSTREAM)
            mlist = mlists.get(listname)
            if not mlist:
                mlist = MailList.MailList(listname, lock=0)
                mlists[listname] = mlist
            # dispatch on the type of command
            if command == '':
                # post
                msg.Enqueue(mlist, tolist=1)
            elif command == 'admin':
                msg.Enqueue(mlist, toadmin=1)
            elif command == 'owner':
                msg.Enqueue(mlist, toowner=1)
            elif command == 'request':
                msg.Enqueue(mlist, torequest=1)
            elif command in ('join', 'leave'):
                # TBD: this is a hack!
                if command == 'join':
                    msg['Subject'] = 'subscribe'
                else:
                    msg['Subject'] = 'unsubscribe'
                msg.Enqueue(mlist, torequest=1)


class Options:
    setuid = True
    classname = 'PureProxy'
    size_limit = None
    enable_SMTPUTF8 = False


def parseargs():
    global DEBUGSTREAM
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], 'nVhc:s:du',
            ['class=', 'nosetuid', 'version', 'help', 'size=', 'debug',
             'smtputf8'])
    except getopt.error as e:
        usage(1, e)

    options = Options()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage(0)
        elif opt in ('-V', '--version'):
            print(__version__)
            sys.exit(0)
        elif opt in ('-n', '--nosetuid'):
            options.setuid = False
        elif opt in ('-c', '--class'):
            options.classname = arg
        elif opt in ('-d', '--debug'):
            DEBUGSTREAM = sys.stderr
        elif opt in ('-u', '--smtputf8'):
            options.enable_SMTPUTF8 = True
        elif opt in ('-s', '--size'):
            try:
                int_size = int(arg)
                options.size_limit = int_size
            except:
                print('Invalid size: ' + arg, file=sys.stderr)
                sys.exit(1)

    # parse the rest of the arguments
    if len(args) < 1:
        localspec = 'localhost:8025'
        remotespec = 'localhost:25'
    elif len(args) < 2:
        localspec = args[0]
        remotespec = 'localhost:25'
    elif len(args) < 3:
        localspec = args[0]
        remotespec = args[1]
    else:
        usage(1, 'Invalid arguments: %s' % COMMASPACE.join(args))

    # split into host/port pairs
    i = localspec.find(':')
    if i < 0:
        usage(1, 'Bad local spec: %s' % localspec)
    options.localhost = localspec[:i]
    try:
        options.localport = int(localspec[i+1:])
    except ValueError:
        usage(1, 'Bad local port: %s' % localspec)
    i = remotespec.find(':')
    if i < 0:
        usage(1, 'Bad remote spec: %s' % remotespec)
    options.remotehost = remotespec[:i]
    try:
        options.remoteport = int(remotespec[i+1:])
    except ValueError:
        usage(1, 'Bad remote port: %s' % remotespec)
    return options


if __name__ == '__main__':
    options = parseargs()
    # Become nobody
    classname = options.classname
    if "." in classname:
        lastdot = classname.rfind(".")
        mod = __import__(classname[:lastdot], globals(), locals(), [""])
        classname = classname[lastdot+1:]
    else:
        import __main__ as mod
    class_ = getattr(mod, classname)
    proxy = class_((options.localhost, options.localport),
                   (options.remotehost, options.remoteport),
                   options.size_limit, enable_SMTPUTF8=options.enable_SMTPUTF8)
    if options.setuid:
        try:
            import pwd
        except ImportError:
            print('Cannot import module "pwd"; try running with -n option.', file=sys.stderr)
            sys.exit(1)
        nobody = pwd.getpwnam('nobody')[2]
        try:
            os.setuid(nobody)
        except PermissionError:
            print('Cannot setuid "nobody"; try running with -n option.', file=sys.stderr)
            sys.exit(1)
    try:
        asyncore.loop()
    except KeyboardInterrupt:
        pass
<|endoftext|>"""Extract, format and print information about Python stack traces."""

import collections
import itertools
import linecache
import sys

__all__ = ['extract_stack', 'extract_tb', 'format_exception',
           'format_exception_only', 'format_list', 'format_stack',
           'format_tb', 'print_exc', 'format_exc', 'print_exception',
           'print_last', 'print_stack', 'print_tb', 'clear_frames',
           'FrameSummary', 'StackSummary', 'TracebackException',
           'walk_stack', 'walk_tb']

#
# Formatting and printing lists of traceback lines.
#

def print_list(extracted_list, file=None):
    """Print the list of tuples as returned by extract_tb() or
    extract_stack() as a formatted stack trace to the given file."""
    if file is None:
        file = sys.stderr
    for item in StackSummary.from_list(extracted_list).format():
        print(item, file=file, end="")

def format_list(extracted_list):
    """Format a list of tuples or FrameSummary objects for printing.

    Given a list of tuples or FrameSummary objects as returned by
    extract_tb() or extract_stack(), return a list of strings ready
    for printing.

    Each string in the resulting list corresponds to the item with the
    same index in the argument list.  Each string ends in a newline;
    the strings may contain internal newlines as well, for those items
    whose source text line is not None.
    """
    return StackSummary.from_list(extracted_list).format()

#
# Printing and Extracting Tracebacks.
#

def print_tb(tb, limit=None, file=None):
    """Print up to 'limit' stack trace entries from the traceback 'tb'.

    If 'limit' is omitted or None, all entries are printed.  If 'file'
    is omitted or None, the output goes to sys.stderr; otherwise
    'file' should be an open file or file-like object with a write()
    method.
    """
    print_list(extract_tb(tb, limit=limit), file=file)

def format_tb(tb, limit=None):
    """A shorthand for 'format_list(extract_tb(tb, limit))'."""
    return extract_tb(tb, limit=limit).format()

def extract_tb(tb, limit=None):
    """
    Return a StackSummary object representing a list of
    pre-processed entries from traceback.

    This is useful for alternate formatting of stack traces.  If
    'limit' is omitted or None, all entries are extracted.  A
    pre-processed stack trace entry is a FrameSummary object
    containing attributes filename, lineno, name, and line
    representing the information that is usually printed for a stack
    trace.  The line is a string with leading and trailing
    whitespace stripped; if the source is not available it is None.
    """
    return StackSummary.extract(walk_tb(tb), limit=limit)

#
# Exception formatting and output.
#

_cause_message = (
    "\nThe above exception was the direct cause "
    "of the following exception:\n\n")

_context_message = (
    "\nDuring handling of the above exception, "
    "another exception occurred:\n\n")


class _Sentinel:
    def __repr__(self):
        return "<implicit>"

_sentinel = _Sentinel()

def _parse_value_tb(exc, value, tb):
    if (value is _sentinel) != (tb is _sentinel):
        raise ValueError("Both or neither of value and tb must be given")
    if value is tb is _sentinel:
        if exc is not None:
            return exc, exc.__traceback__
        else:
            return None, None
    return value, tb


def print_exception(exc, /, value=_sentinel, tb=_sentinel, limit=None, \
                    file=None, chain=True):
    """Print exception up to 'limit' stack trace entries from 'tb' to 'file'.

    This differs from print_tb() in the following ways: (1) if
    traceback is not None, it prints a header "Traceback (most recent
    call last):"; (2) it prints the exception type and value after the
    stack trace; (3) if type is SyntaxError and value has the
    appropriate format, it prints the line where the syntax error
    occurred with a caret on the next line indicating the approximate
    position of the error.
    """
    value, tb = _parse_value_tb(exc, value, tb)
    if file is None:
        file = sys.stderr
    te = TracebackException(type(value), value, tb, limit=limit, compact=True)
    for line in te.format(chain=chain):
        print(line, file=file, end="")


def format_exception(exc, /, value=_sentinel, tb=_sentinel, limit=None, \
                     chain=True):
    """Format a stack trace and the exception information.

    The arguments have the same meaning as the corresponding arguments
    to print_exception().  The return value is a list of strings, each
    ending in a newline and some containing internal newlines.  When
    these lines are concatenated and printed, exactly the same text is
    printed as does print_exception().
    """
    value, tb = _parse_value_tb(exc, value, tb)
    te = TracebackException(type(value), value, tb, limit=limit, compact=True)
    return list(te.format(chain=chain))


def format_exception_only(exc, /, value=_sentinel):
    """Format the exception part of a traceback.

    The return value is a list of strings, each ending in a newline.

    Normally, the list contains a single string; however, for
    SyntaxError exceptions, it contains several lines that (when
    printed) display detailed information about where the syntax
    error occurred.

    The message indicating which exception occurred is always the last
    string in the list.

    """
    if value is _sentinel:
        value = exc
    te = TracebackException(type(value), value, None, compact=True)
    return list(te.format_exception_only())


# -- not official API but folk probably use these two functions.

def _format_final_exc_line(etype, value):
    valuestr = _some_str(value)
    if value is None or not valuestr:
        line = "%s\n" % etype
    else:
        line = "%s: %s\n" % (etype, valuestr)
    return line

def _some_str(value):
    try:
        return str(value)
    except:
        return '<unprintable %s object>' % type(value).__name__

# --

def print_exc(limit=None, file=None, chain=True):
    """Shorthand for 'print_exception(*sys.exc_info(), limit, file)'."""
    print_exception(*sys.exc_info(), limit=limit, file=file, chain=chain)

def format_exc(limit=None, chain=True):
    """Like print_exc() but return a string."""
    return "".join(format_exception(*sys.exc_info(), limit=limit, chain=chain))

def print_last(limit=None, file=None, chain=True):
    """This is a shorthand for 'print_exception(sys.last_type,
    sys.last_value, sys.last_traceback, limit, file)'."""
    if not hasattr(sys, "last_type"):
        raise ValueError("no last exception")
    print_exception(sys.last_type, sys.last_value, sys.last_traceback,
                    limit, file, chain)

#
# Printing and Extracting Stacks.
#

def print_stack(f=None, limit=None, file=None):
    """Print a stack trace from its invocation point.

    The optional 'f' argument can be used to specify an alternate
    stack frame at which to start. The optional 'limit' and 'file'
    arguments have the same meaning as for print_exception().
    """
    if f is None:
        f = sys._getframe().f_back
    print_list(extract_stack(f, limit=limit), file=file)


def format_stack(f=None, limit=None):
    """Shorthand for 'format_list(extract_stack(f, limit))'."""
    if f is None:
        f = sys._getframe().f_back
    return format_list(extract_stack(f, limit=limit))


def extract_stack(f=None, limit=None):
    """Extract the raw traceback from the current stack frame.

    The return value has the same format as for extract_tb().  The
    optional 'f' and 'limit' arguments have the same meaning as for
    print_stack().  Each item in the list is a quadruple (filename,
    line number, function name, text), and the entries are in order
    from oldest to newest stack frame.
    """
    if f is None:
        f = sys._getframe().f_back
    stack = StackSummary.extract(walk_stack(f), limit=limit)
    stack.reverse()
    return stack


def clear_frames(tb):
    "Clear all references to local variables in the frames of a traceback."
    while tb is not None:
        try:
            tb.tb_frame.clear()
        except RuntimeError:
            # Ignore the exception raised if the frame is still executing.
            pass
        tb = tb.tb_next


class FrameSummary:
    """A single frame from a traceback.

    - :attr:`filename` The filename for the frame.
    - :attr:`lineno` The line within filename for the frame that was
      active when the frame was captured.
    - :attr:`name` The name of the function or method that was executing
      when the frame was captured.
    - :attr:`line` The text from the linecache module for the
      of code that was running when the frame was captured.
    - :attr:`locals` Either None if locals were not supplied, or a dict
      mapping the name to the repr() of the variable.
    """

    __slots__ = ('filename', 'lineno', 'name', '_line', 'locals')

    def __init__(self, filename, lineno, name, *, lookup_line=True,
            locals=None, line=None):
        """Construct a FrameSummary.

        :param lookup_line: If True, `linecache` is consulted for the source
            code line. Otherwise, the line will be looked up when first needed.
        :param locals: If supplied the frame locals, which will be captured as
            object representations.
        :param line: If provided, use this instead of looking up the line in
            the linecache.
        """
        self.filename = filename
        self.lineno = lineno
        self.name = name
        self._line = line
        if lookup_line:
            self.line
        self.locals = {k: repr(v) for k, v in locals.items()} if locals else None

    def __eq__(self, other):
        if isinstance(other, FrameSummary):
            return (self.filename == other.filename and
                    self.lineno == other.lineno and
                    self.name == other.name and
                    self.locals == other.locals)
        if isinstance(other, tuple):
            return (self.filename, self.lineno, self.name, self.line) == other
        return NotImplemented

    def __getitem__(self, pos):
        return (self.filename, self.lineno, self.name, self.line)[pos]

    def __iter__(self):
        return iter([self.filename, self.lineno, self.name, self.line])

    def __repr__(self):
        return "<FrameSummary file {filename}, line {lineno} in {name}>".format(
            filename=self.filename, lineno=self.lineno, name=self.name)

    def __len__(self):
        return 4

    @property
    def line(self):
        if self._line is None:
            if self.lineno is None:
                return None
            self._line = linecache.getline(self.filename, self.lineno)
        return self._line.strip()

def walk_stack(f):
    """Walk a stack yielding the frame and line number for each frame.

    This will follow f.f_back from the given frame. If no frame is given, the
    current stack is used. Usually used with StackSummary.extract.
    """
    if f is None:
        f = sys._getframe().f_back.f_back
    while f is not None:
        yield f, f.f_lineno
        f = f.f_back


def walk_tb(tb):
    """Walk a traceback yielding the frame and line number for each frame.

    This will follow tb.tb_next (and thus is in the opposite order to
    walk_stack). Usually used with StackSummary.extract.
    """
    while tb is not None:
        yield tb.tb_frame, tb.tb_lineno
        tb = tb.tb_next


_RECURSIVE_CUTOFF = 3 # Also hardcoded in traceback.c.

class StackSummary(list):
    """A stack of frames."""

    @classmethod
    def extract(klass, frame_gen, *, limit=None, lookup_lines=True,
            capture_locals=False):
        """Create a StackSummary from a traceback or stack object.

        :param frame_gen: A generator that yields (frame, lineno) tuples to
            include in the stack.
        :param limit: None to include all frames or the number of frames to
            include.
        :param lookup_lines: If True, lookup lines for each frame immediately,
            otherwise lookup is deferred until the frame is rendered.
        :param capture_locals: If True, the local variables from each frame will
            be captured as object representations into the FrameSummary.
        """
        if limit is None:
            limit = getattr(sys, 'tracebacklimit', None)
            if limit is not None and limit < 0:
                limit = 0
        if limit is not None:
            if limit >= 0:
                frame_gen = itertools.islice(frame_gen, limit)
            else:
                frame_gen = collections.deque(frame_gen, maxlen=-limit)

        result = klass()
        fnames = set()
        for f, lineno in frame_gen:
            co = f.f_code
            filename = co.co_filename
            name = co.co_name

            fnames.add(filename)
            linecache.lazycache(filename, f.f_globals)
            # Must defer line lookups until we have called checkcache.
            if capture_locals:
                f_locals = f.f_locals
            else:
                f_locals = None
            result.append(FrameSummary(
                filename, lineno, name, lookup_line=False, locals=f_locals))
        for filename in fnames:
            linecache.checkcache(filename)
        # If immediate lookup was desired, trigger lookups now.
        if lookup_lines:
            for f in result:
                f.line
        return result

    @classmethod
    def from_list(klass, a_list):
        """
        Create a StackSummary object from a supplied list of
        FrameSummary objects or old-style list of tuples.
        """
        # While doing a fast-path check for isinstance(a_list, StackSummary) is
        # appealing, idlelib.run.cleanup_traceback and other similar code may
        # break this by making arbitrary frames plain tuples, so we need to
        # check on a frame by frame basis.
        result = StackSummary()
        for frame in a_list:
            if isinstance(frame, FrameSummary):
                result.append(frame)
            else:
                filename, lineno, name, line = frame
                result.append(FrameSummary(filename, lineno, name, line=line))
        return result

    def format(self):
        """Format the stack ready for printing.

        Returns a list of strings ready for printing.  Each string in the
        resulting list corresponds to a single frame from the stack.
        Each string ends in a newline; the strings may contain internal
        newlines as well, for those items with source text lines.

        For long sequences of the same frame and line, the first few
        repetitions are shown, followed by a summary line stating the exact
        number of further repetitions.
        """
        result = []
        last_file = None
        last_line = None
        last_name = None
        count = 0
        for frame in self:
            if (last_file is None or last_file != frame.filename or
                last_line is None or last_line != frame.lineno or
                last_name is None or last_name != frame.name):
                if count > _RECURSIVE_CUTOFF:
                    count -= _RECURSIVE_CUTOFF
                    result.append(
                        f'  [Previous line repeated {count} more '
                        f'time{"s" if count > 1 else ""}]\n'
                    )
                last_file = frame.filename
                last_line = frame.lineno
                last_name = frame.name
                count = 0
            count += 1
            if count > _RECURSIVE_CUTOFF:
                continue
            row = []
            row.append('  File "{}", line {}, in {}\n'.format(
                frame.filename, frame.lineno, frame.name))
            if frame.line:
                row.append('    {}\n'.format(frame.line.strip()))
            if frame.locals:
                for name, value in sorted(frame.locals.items()):
                    row.append('    {name} = {value}\n'.format(name=name, value=value))
            result.append(''.join(row))
        if count > _RECURSIVE_CUTOFF:
            count -= _RECURSIVE_CUTOFF
            result.append(
                f'  [Previous line repeated {count} more '
                f'time{"s" if count > 1 else ""}]\n'
            )
        return result


class TracebackException:
    """An exception ready for rendering.

    The traceback module captures enough attributes from the original exception
    to this intermediary form to ensure that no references are held, while
    still being able to fully print or format it.

    Use `from_exception` to create TracebackException instances from exception
    objects, or the constructor to create TracebackException instances from
    individual components.

    - :attr:`__cause__` A TracebackException of the original *__cause__*.
    - :attr:`__context__` A TracebackException of the original *__context__*.
    - :attr:`__suppress_context__` The *__suppress_context__* value from the
      original exception.
    - :attr:`stack` A `StackSummary` representing the traceback.
    - :attr:`exc_type` The class of the original traceback.
    - :attr:`filename` For syntax errors - the filename where the error
      occurred.
    - :attr:`lineno` For syntax errors - the linenumber where the error
      occurred.
    - :attr:`end_lineno` For syntax errors - the end linenumber where the error
      occurred. Can be `None` if not present.
    - :attr:`text` For syntax errors - the text where the error
      occurred.
    - :attr:`offset` For syntax errors - the offset into the text where the
      error occurred.
    - :attr:`end_offset` For syntax errors - the end offset into the text where
      the error occurred. Can be `None` if not present.
    - :attr:`msg` For syntax errors - the compiler error message.
    """

    def __init__(self, exc_type, exc_value, exc_traceback, *, limit=None,
            lookup_lines=True, capture_locals=False, compact=False,
            _seen=None):
        # NB: we need to accept exc_traceback, exc_value, exc_traceback to
        # permit backwards compat with the existing API, otherwise we
        # need stub thunk objects just to glue it together.
        # Handle loops in __cause__ or __context__.
        is_recursive_call = _seen is not None
        if _seen is None:
            _seen = set()
        _seen.add(id(exc_value))

        # TODO: locals.
        self.stack = StackSummary.extract(
            walk_tb(exc_traceback), limit=limit, lookup_lines=lookup_lines,
            capture_locals=capture_locals)
        self.exc_type = exc_type
        # Capture now to permit freeing resources: only complication is in the
        # unofficial API _format_final_exc_line
        self._str = _some_str(exc_value)
        if exc_type and issubclass(exc_type, SyntaxError):
            # Handle SyntaxError's specially
            self.filename = exc_value.filename
            lno = exc_value.lineno
            self.lineno = str(lno) if lno is not None else None
            end_lno = exc_value.end_lineno
            self.end_lineno = str(end_lno) if end_lno is not None else None
            self.text = exc_value.text
            self.offset = exc_value.offset
            self.end_offset = exc_value.end_offset
            self.msg = exc_value.msg
        if lookup_lines:
            self._load_lines()
        self.__suppress_context__ = \
            exc_value.__suppress_context__ if exc_value is not None else False

        # Convert __cause__ and __context__ to `TracebackExceptions`s, use a
        # queue to avoid recursion (only the top-level call gets _seen == None)
        if not is_recursive_call:
            queue = [(self, exc_value)]
            while queue:
                te, e = queue.pop()
                if (e and e.__cause__ is not None
                    and id(e.__cause__) not in _seen):
                    cause = TracebackException(
                        type(e.__cause__),
                        e.__cause__,
                        e.__cause__.__traceback__,
                        limit=limit,
                        lookup_lines=lookup_lines,
                        capture_locals=capture_locals,
                        _seen=_seen)
                else:
                    cause = None

                if compact:
                    need_context = (cause is None and
                                    e is not None and
                                    not e.__suppress_context__)
                else:
                    need_context = True
                if (e and e.__context__ is not None
                    and need_context and id(e.__context__) not in _seen):
                    context = TracebackException(
                        type(e.__context__),
                        e.__context__,
                        e.__context__.__traceback__,
                        limit=limit,
                        lookup_lines=lookup_lines,
                        capture_locals=capture_locals,
                        _seen=_seen)
                else:
                    context = None
                te.__cause__ = cause
                te.__context__ = context
                if cause:
                    queue.append((te.__cause__, e.__cause__))
                if context:
                    queue.append((te.__context__, e.__context__))

    @classmethod
    def from_exception(cls, exc, *args, **kwargs):
        """Create a TracebackException from an exception."""
        return cls(type(exc), exc, exc.__traceback__, *args, **kwargs)

    def _load_lines(self):
        """Private API. force all lines in the stack to be loaded."""
        for frame in self.stack:
            frame.line

    def __eq__(self, other):
        if isinstance(other, TracebackException):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __str__(self):
        return self._str

    def format_exception_only(self):
        """Format the exception part of the traceback.

        The return value is a generator of strings, each ending in a newline.

        Normally, the generator emits a single string; however, for
        SyntaxError exceptions, it emits several lines that (when
        printed) display detailed information about where the syntax
        error occurred.

        The message indicating which exception occurred is always the last
        string in the output.
        """
        if self.exc_type is None:
            yield _format_final_exc_line(None, self._str)
            return

        stype = self.exc_type.__qualname__
        smod = self.exc_type.__module__
        if smod not in ("__main__", "builtins"):
            if not isinstance(smod, str):
                smod = "<unknown>"
            stype = smod + '.' + stype

        if not issubclass(self.exc_type, SyntaxError):
            yield _format_final_exc_line(stype, self._str)
        else:
            yield from self._format_syntax_error(stype)

    def _format_syntax_error(self, stype):
        """Format SyntaxError exceptions (internal helper)."""
        # Show exactly where the problem was found.
        filename_suffix = ''
        if self.lineno is not None:
            yield '  File "{}", line {}\n'.format(
                self.filename or "<string>", self.lineno)
        elif self.filename is not None:
            filename_suffix = ' ({})'.format(self.filename)

        text = self.text
        if text is not None:
            # text  = "   foo\n"
            # rtext = "   foo"
            # ltext =    "foo"
            rtext = text.rstrip('\n')
            ltext = rtext.lstrip(' \n\f')
            spaces = len(rtext) - len(ltext)
            yield '    {}\n'.format(ltext)

            if self.offset is not None:
                offset = self.offset
                end_offset = self.end_offset if self.end_offset not in {None, 0} else offset
                if offset == end_offset or end_offset == -1:
                    end_offset = offset + 1

                # Convert 1-based column offset to 0-based index into stripped text
                colno = offset - 1 - spaces
                end_colno = end_offset - 1 - spaces
                if colno >= 0:
                    # non-space whitespace (likes tabs) must be kept for alignment
                    caretspace = ((c if c.isspace() else ' ') for c in ltext[:colno])
                    yield '    {}{}'.format("".join(caretspace), ('^' * (end_colno - colno) + "\n"))
        msg = self.msg or "<no detail available>"
        yield "{}: {}{}\n".format(stype, msg, filename_suffix)

    def format(self, *, chain=True):
        """Format the exception.

        If chain is not *True*, *__cause__* and *__context__* will not be formatted.

        The return value is a generator of strings, each ending in a newline and
        some containing internal newlines. `print_exception` is a wrapper around
        this method which just prints the lines to a file.

        The message indicating which exception occurred is always the last
        string in the output.
        """

        output = []
        exc = self
        while exc:
            if chain:
                if exc.__cause__ is not None:
                    chained_msg = _cause_message
                    chained_exc = exc.__cause__
                elif (exc.__context__  is not None and
                      not exc.__suppress_context__):
                    chained_msg = _context_message
                    chained_exc = exc.__context__
                else:
                    chained_msg = None
                    chained_exc = None

                output.append((chained_msg, exc))
                exc = chained_exc
            else:
                output.append((None, exc))
                exc = None

        for msg, exc in reversed(output):
            if msg is not None:
                yield msg
            if exc.stack:
                yield 'Traceback (most recent call last):\n'
                yield from exc.stack.format()
            yield from exc.format_exception_only()
<|endoftext|>"""Find modules used by a script, using introspection."""

import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys


LOAD_CONST = dis.opmap['LOAD_CONST']
IMPORT_NAME = dis.opmap['IMPORT_NAME']
STORE_NAME = dis.opmap['STORE_NAME']
STORE_GLOBAL = dis.opmap['STORE_GLOBAL']
STORE_OPS = STORE_NAME, STORE_GLOBAL
EXTENDED_ARG = dis.EXTENDED_ARG

# Old imp constants:

_SEARCH_ERROR = 0
_PY_SOURCE = 1
_PY_COMPILED = 2
_C_EXTENSION = 3
_PKG_DIRECTORY = 5
_C_BUILTIN = 6
_PY_FROZEN = 7

# Modulefinder does a good job at simulating Python's, but it can not
# handle __path__ modifications packages make at runtime.  Therefore there
# is a mechanism whereby you can register extra paths in this map for a
# package, and it will be honored.

# Note this is a mapping is lists of paths.
packagePathMap = {}

# A Public interface
def AddPackagePath(packagename, path):
    packagePathMap.setdefault(packagename, []).append(path)

replacePackageMap = {}

# This ReplacePackage mechanism allows modulefinder to work around
# situations in which a package injects itself under the name
# of another package into sys.modules at runtime by calling
# ReplacePackage("real_package_name", "faked_package_name")
# before running ModuleFinder.

def ReplacePackage(oldname, newname):
    replacePackageMap[oldname] = newname


def _find_module(name, path=None):
    """An importlib reimplementation of imp.find_module (for our purposes)."""

    # It's necessary to clear the caches for our Finder first, in case any
    # modules are being added/deleted/modified at runtime. In particular,
    # test_modulefinder.py changes file tree contents in a cache-breaking way:

    importlib.machinery.PathFinder.invalidate_caches()

    spec = importlib.machinery.PathFinder.find_spec(name, path)

    if spec is None:
        raise ImportError("No module named {name!r}".format(name=name), name=name)

    # Some special cases:

    if spec.loader is importlib.machinery.BuiltinImporter:
        return None, None, ("", "", _C_BUILTIN)

    if spec.loader is importlib.machinery.FrozenImporter:
        return None, None, ("", "", _PY_FROZEN)

    file_path = spec.origin

    if spec.loader.is_package(name):
        return None, os.path.dirname(file_path), ("", "", _PKG_DIRECTORY)

    if isinstance(spec.loader, importlib.machinery.SourceFileLoader):
        kind = _PY_SOURCE

    elif isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        kind = _C_EXTENSION

    elif isinstance(spec.loader, importlib.machinery.SourcelessFileLoader):
        kind = _PY_COMPILED

    else:  # Should never happen.
        return None, None, ("", "", _SEARCH_ERROR)

    file = io.open_code(file_path)
    suffix = os.path.splitext(file_path)[-1]

    return file, file_path, (suffix, "rb", kind)


class Module:

    def __init__(self, name, file=None, path=None):
        self.__name__ = name
        self.__file__ = file
        self.__path__ = path
        self.__code__ = None
        # The set of global names that are assigned to in the module.
        # This includes those names imported through starimports of
        # Python modules.
        self.globalnames = {}
        # The set of starimports this module did that could not be
        # resolved, ie. a starimport from a non-Python module.
        self.starimports = {}

    def __repr__(self):
        s = "Module(%r" % (self.__name__,)
        if self.__file__ is not None:
            s = s + ", %r" % (self.__file__,)
        if self.__path__ is not None:
            s = s + ", %r" % (self.__path__,)
        s = s + ")"
        return s

class ModuleFinder:

    def __init__(self, path=None, debug=0, excludes=None, replace_paths=None):
        if path is None:
            path = sys.path
        self.path = path
        self.modules = {}
        self.badmodules = {}
        self.debug = debug
        self.indent = 0
        self.excludes = excludes if excludes is not None else []
        self.replace_paths = replace_paths if replace_paths is not None else []
        self.processed_paths = []   # Used in debugging only

    def msg(self, level, str, *args):
        if level <= self.debug:
            for i in range(self.indent):
                print("   ", end=' ')
            print(str, end=' ')
            for arg in args:
                print(repr(arg), end=' ')
            print()

    def msgin(self, *args):
        level = args[0]
        if level <= self.debug:
            self.indent = self.indent + 1
            self.msg(*args)

    def msgout(self, *args):
        level = args[0]
        if level <= self.debug:
            self.indent = self.indent - 1
            self.msg(*args)

    def run_script(self, pathname):
        self.msg(2, "run_script", pathname)
        with io.open_code(pathname) as fp:
            stuff = ("", "rb", _PY_SOURCE)
            self.load_module('__main__', fp, pathname, stuff)

    def load_file(self, pathname):
        dir, name = os.path.split(pathname)
        name, ext = os.path.splitext(name)
        with io.open_code(pathname) as fp:
            stuff = (ext, "rb", _PY_SOURCE)
            self.load_module(name, fp, pathname, stuff)

    def import_hook(self, name, caller=None, fromlist=None, level=-1):
        self.msg(3, "import_hook", name, caller, fromlist, level)
        parent = self.determine_parent(caller, level=level)
        q, tail = self.find_head_package(parent, name)
        m = self.load_tail(q, tail)
        if not fromlist:
            return q
        if m.__path__:
            self.ensure_fromlist(m, fromlist)
        return None

    def determine_parent(self, caller, level=-1):
        self.msgin(4, "determine_parent", caller, level)
        if not caller or level == 0:
            self.msgout(4, "determine_parent -> None")
            return None
        pname = caller.__name__
        if level >= 1: # relative import
            if caller.__path__:
                level -= 1
            if level == 0:
                parent = self.modules[pname]
                assert parent is caller
                self.msgout(4, "determine_parent ->", parent)
                return parent
            if pname.count(".") < level:
                raise ImportError("relative importpath too deep")
            pname = ".".join(pname.split(".")[:-level])
            parent = self.modules[pname]
            self.msgout(4, "determine_parent ->", parent)
            return parent
        if caller.__path__:
            parent = self.modules[pname]
            assert caller is parent
            self.msgout(4, "determine_parent ->", parent)
            return parent
        if '.' in pname:
            i = pname.rfind('.')
            pname = pname[:i]
            parent = self.modules[pname]
            assert parent.__name__ == pname
            self.msgout(4, "determine_parent ->", parent)
            return parent
        self.msgout(4, "determine_parent -> None")
        return None

    def find_head_package(self, parent, name):
        self.msgin(4, "find_head_package", parent, name)
        if '.' in name:
            i = name.find('.')
            head = name[:i]
            tail = name[i+1:]
        else:
            head = name
            tail = ""
        if parent:
            qname = "%s.%s" % (parent.__name__, head)
        else:
            qname = head
        q = self.import_module(head, qname, parent)
        if q:
            self.msgout(4, "find_head_package ->", (q, tail))
            return q, tail
        if parent:
            qname = head
            parent = None
            q = self.import_module(head, qname, parent)
            if q:
                self.msgout(4, "find_head_package ->", (q, tail))
                return q, tail
        self.msgout(4, "raise ImportError: No module named", qname)
        raise ImportError("No module named " + qname)

    def load_tail(self, q, tail):
        self.msgin(4, "load_tail", q, tail)
        m = q
        while tail:
            i = tail.find('.')
            if i < 0: i = len(tail)
            head, tail = tail[:i], tail[i+1:]
            mname = "%s.%s" % (m.__name__, head)
            m = self.import_module(head, mname, m)
            if not m:
                self.msgout(4, "raise ImportError: No module named", mname)
                raise ImportError("No module named " + mname)
        self.msgout(4, "load_tail ->", m)
        return m

    def ensure_fromlist(self, m, fromlist, recursive=0):
        self.msg(4, "ensure_fromlist", m, fromlist, recursive)
        for sub in fromlist:
            if sub == "*":
                if not recursive:
                    all = self.find_all_submodules(m)
                    if all:
                        self.ensure_fromlist(m, all, 1)
            elif not hasattr(m, sub):
                subname = "%s.%s" % (m.__name__, sub)
                submod = self.import_module(sub, subname, m)
                if not submod:
                    raise ImportError("No module named " + subname)

    def find_all_submodules(self, m):
        if not m.__path__:
            return
        modules = {}
        # 'suffixes' used to be a list hardcoded to [".py", ".pyc"].
        # But we must also collect Python extension modules - although
        # we cannot separate normal dlls from Python extensions.
        suffixes = []
        suffixes += importlib.machinery.EXTENSION_SUFFIXES[:]
        suffixes += importlib.machinery.SOURCE_SUFFIXES[:]
        suffixes += importlib.machinery.BYTECODE_SUFFIXES[:]
        for dir in m.__path__:
            try:
                names = os.listdir(dir)
            except OSError:
                self.msg(2, "can't list directory", dir)
                continue
            for name in names:
                mod = None
                for suff in suffixes:
                    n = len(suff)
                    if name[-n:] == suff:
                        mod = name[:-n]
                        break
                if mod and mod != "__init__":
                    modules[mod] = mod
        return modules.keys()

    def import_module(self, partname, fqname, parent):
        self.msgin(3, "import_module", partname, fqname, parent)
        try:
            m = self.modules[fqname]
        except KeyError:
            pass
        else:
            self.msgout(3, "import_module ->", m)
            return m
        if fqname in self.badmodules:
            self.msgout(3, "import_module -> None")
            return None
        if parent and parent.__path__ is None:
            self.msgout(3, "import_module -> None")
            return None
        try:
            fp, pathname, stuff = self.find_module(partname,
                                                   parent and parent.__path__, parent)
        except ImportError:
            self.msgout(3, "import_module ->", None)
            return None

        try:
            m = self.load_module(fqname, fp, pathname, stuff)
        finally:
            if fp:
                fp.close()
        if parent:
            setattr(parent, partname, m)
        self.msgout(3, "import_module ->", m)
        return m

    def load_module(self, fqname, fp, pathname, file_info):
        suffix, mode, type = file_info
        self.msgin(2, "load_module", fqname, fp and "fp", pathname)
        if type == _PKG_DIRECTORY:
            m = self.load_package(fqname, pathname)
            self.msgout(2, "load_module ->", m)
            return m
        if type == _PY_SOURCE:
            co = compile(fp.read(), pathname, 'exec')
        elif type == _PY_COMPILED:
            try:
                data = fp.read()
                importlib._bootstrap_external._classify_pyc(data, fqname, {})
            except ImportError as exc:
                self.msgout(2, "raise ImportError: " + str(exc), pathname)
                raise
            co = marshal.loads(memoryview(data)[16:])
        else:
            co = None
        m = self.add_module(fqname)
        m.__file__ = pathname
        if co:
            if self.replace_paths:
                co = self.replace_paths_in_code(co)
            m.__code__ = co
            self.scan_code(co, m)
        self.msgout(2, "load_module ->", m)
        return m

    def _add_badmodule(self, name, caller):
        if name not in self.badmodules:
            self.badmodules[name] = {}
        if caller:
            self.badmodules[name][caller.__name__] = 1
        else:
            self.badmodules[name]["-"] = 1

    def _safe_import_hook(self, name, caller, fromlist, level=-1):
        # wrapper for self.import_hook() that won't raise ImportError
        if name in self.badmodules:
            self._add_badmodule(name, caller)
            return
        try:
            self.import_hook(name, caller, level=level)
        except ImportError as msg:
            self.msg(2, "ImportError:", str(msg))
            self._add_badmodule(name, caller)
        except SyntaxError as msg:
            self.msg(2, "SyntaxError:", str(msg))
            self._add_badmodule(name, caller)
        else:
            if fromlist:
                for sub in fromlist:
                    fullname = name + "." + sub
                    if fullname in self.badmodules:
                        self._add_badmodule(fullname, caller)
                        continue
                    try:
                        self.import_hook(name, caller, [sub], level=level)
                    except ImportError as msg:
                        self.msg(2, "ImportError:", str(msg))
                        self._add_badmodule(fullname, caller)

    def scan_opcodes(self, co):
        # Scan the code, and yield 'interesting' opcode combinations
        code = co.co_code
        names = co.co_names
        consts = co.co_consts
        opargs = [(op, arg) for _, op, arg in dis._unpack_opargs(code)
                  if op != EXTENDED_ARG]
        for i, (op, oparg) in enumerate(opargs):
            if op in STORE_OPS:
                yield "store", (names[oparg],)
                continue
            if (op == IMPORT_NAME and i >= 2
                    and opargs[i-1][0] == opargs[i-2][0] == LOAD_CONST):
                level = consts[opargs[i-2][1]]
                fromlist = consts[opargs[i-1][1]]
                if level == 0: # absolute import
                    yield "absolute_import", (fromlist, names[oparg])
                else: # relative import
                    yield "relative_import", (level, fromlist, names[oparg])
                continue

    def scan_code(self, co, m):
        code = co.co_code
        scanner = self.scan_opcodes
        for what, args in scanner(co):
            if what == "store":
                name, = args
                m.globalnames[name] = 1
            elif what == "absolute_import":
                fromlist, name = args
                have_star = 0
                if fromlist is not None:
                    if "*" in fromlist:
                        have_star = 1
                    fromlist = [f for f in fromlist if f != "*"]
                self._safe_import_hook(name, m, fromlist, level=0)
                if have_star:
                    # We've encountered an "import *". If it is a Python module,
                    # the code has already been parsed and we can suck out the
                    # global names.
                    mm = None
                    if m.__path__:
                        # At this point we don't know whether 'name' is a
                        # submodule of 'm' or a global module. Let's just try
                        # the full name first.
                        mm = self.modules.get(m.__name__ + "." + name)
                    if mm is None:
                        mm = self.modules.get(name)
                    if mm is not None:
                        m.globalnames.update(mm.globalnames)
                        m.starimports.update(mm.starimports)
                        if mm.__code__ is None:
                            m.starimports[name] = 1
                    else:
                        m.starimports[name] = 1
            elif what == "relative_import":
                level, fromlist, name = args
                if name:
                    self._safe_import_hook(name, m, fromlist, level=level)
                else:
                    parent = self.determine_parent(m, level=level)
                    self._safe_import_hook(parent.__name__, None, fromlist, level=0)
            else:
                # We don't expect anything else from the generator.
                raise RuntimeError(what)

        for c in co.co_consts:
            if isinstance(c, type(co)):
                self.scan_code(c, m)

    def load_package(self, fqname, pathname):
        self.msgin(2, "load_package", fqname, pathname)
        newname = replacePackageMap.get(fqname)
        if newname:
            fqname = newname
        m = self.add_module(fqname)
        m.__file__ = pathname
        m.__path__ = [pathname]

        # As per comment at top of file, simulate runtime __path__ additions.
        m.__path__ = m.__path__ + packagePathMap.get(fqname, [])

        fp, buf, stuff = self.find_module("__init__", m.__path__)
        try:
            self.load_module(fqname, fp, buf, stuff)
            self.msgout(2, "load_package ->", m)
            return m
        finally:
            if fp:
                fp.close()

    def add_module(self, fqname):
        if fqname in self.modules:
            return self.modules[fqname]
        self.modules[fqname] = m = Module(fqname)
        return m

    def find_module(self, name, path, parent=None):
        if parent is not None:
            # assert path is not None
            fullname = parent.__name__+'.'+name
        else:
            fullname = name
        if fullname in self.excludes:
            self.msgout(3, "find_module -> Excluded", fullname)
            raise ImportError(name)

        if path is None:
            if name in sys.builtin_module_names:
                return (None, None, ("", "", _C_BUILTIN))

            path = self.path

        return _find_module(name, path)

    def report(self):
        """Print a report to stdout, listing the found modules with their
        paths, as well as modules that are missing, or seem to be missing.
        """
        print()
        print("  %-25s %s" % ("Name", "File"))
        print("  %-25s %s" % ("----", "----"))
        # Print modules found
        keys = sorted(self.modules.keys())
        for key in keys:
            m = self.modules[key]
            if m.__path__:
                print("P", end=' ')
            else:
                print("m", end=' ')
            print("%-25s" % key, m.__file__ or "")

        # Print missing modules
        missing, maybe = self.any_missing_maybe()
        if missing:
            print()
            print("Missing modules:")
            for name in missing:
                mods = sorted(self.badmodules[name].keys())
                print("?", name, "imported from", ', '.join(mods))
        # Print modules that may be missing, but then again, maybe not...
        if maybe:
            print()
            print("Submodules that appear to be missing, but could also be", end=' ')
            print("global names in the parent package:")
            for name in maybe:
                mods = sorted(self.badmodules[name].keys())
                print("?", name, "imported from", ', '.join(mods))

    def any_missing(self):
        """Return a list of modules that appear to be missing. Use
        any_missing_maybe() if you want to know which modules are
        certain to be missing, and which *may* be missing.
        """
        missing, maybe = self.any_missing_maybe()
        return missing + maybe

    def any_missing_maybe(self):
        """Return two lists, one with modules that are certainly missing
        and one with modules that *may* be missing. The latter names could
        either be submodules *or* just global names in the package.

        The reason it can't always be determined is that it's impossible to
        tell which names are imported when "from module import *" is done
        with an extension module, short of actually importing it.
        """
        missing = []
        maybe = []
        for name in self.badmodules:
            if name in self.excludes:
                continue
            i = name.rfind(".")
            if i < 0:
                missing.append(name)
                continue
            subname = name[i+1:]
            pkgname = name[:i]
            pkg = self.modules.get(pkgname)
            if pkg is not None:
                if pkgname in self.badmodules[name]:
                    # The package tried to import this module itself and
                    # failed. It's definitely missing.
                    missing.append(name)
                elif subname in pkg.globalnames:
                    # It's a global in the package: definitely not missing.
                    pass
                elif pkg.starimports:
                    # It could be missing, but the package did an "import *"
                    # from a non-Python module, so we simply can't be sure.
                    maybe.append(name)
                else:
                    # It's not a global in the package, the package didn't
                    # do funny star imports, it's very likely to be missing.
                    # The symbol could be inserted into the package from the
                    # outside, but since that's not good style we simply list
                    # it missing.
                    missing.append(name)
            else:
                missing.append(name)
        missing.sort()
        maybe.sort()
        return missing, maybe

    def replace_paths_in_code(self, co):
        new_filename = original_filename = os.path.normpath(co.co_filename)
        for f, r in self.replace_paths:
            if original_filename.startswith(f):
                new_filename = r + original_filename[len(f):]
                break

        if self.debug and original_filename not in self.processed_paths:
            if new_filename != original_filename:
                self.msgout(2, "co_filename %r changed to %r" \
                                    % (original_filename,new_filename,))
            else:
                self.msgout(2, "co_filename %r remains unchanged" \
                                    % (original_filename,))
            self.processed_paths.append(original_filename)

        consts = list(co.co_consts)
        for i in range(len(consts)):
            if isinstance(consts[i], type(co)):
                consts[i] = self.replace_paths_in_code(consts[i])

        return co.replace(co_consts=tuple(consts), co_filename=new_filename)


def test():
    # Parse command line
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], "dmp:qx:")
    except getopt.error as msg:
        print(msg)
        return

    # Process options
    debug = 1
    domods = 0
    addpath = []
    exclude = []
    for o, a in opts:
        if o == '-d':
            debug = debug + 1
        if o == '-m':
            domods = 1
        if o == '-p':
            addpath = addpath + a.split(os.pathsep)
        if o == '-q':
            debug = 0
        if o == '-x':
            exclude.append(a)

    # Provide default arguments
    if not args:
        script = "hello.py"
    else:
        script = args[0]

    # Set the path based on sys.path and the script directory
    path = sys.path[:]
    path[0] = os.path.dirname(script)
    path = addpath + path
    if debug > 1:
        print("path:")
        for item in path:
            print("   ", repr(item))

    # Create the module finder and turn its crank
    mf = ModuleFinder(path, debug, exclude)
    for arg in args[1:]:
        if arg == '-m':
            domods = 1
            continue
        if domods:
            if arg[-2:] == '.*':
                mf.import_hook(arg[:-2], None, ["*"])
            else:
                mf.import_hook(arg)
        else:
            mf.load_file(arg)
    mf.run_script(script)
    mf.report()
    return mf  # for -i debugging


if __name__ == '__main__':
    try:
        mf = test()
    except KeyboardInterrupt:
        print("\n[interrupted]")
<|endoftext|>"""A generally useful event scheduler class.

Each instance of this class manages its own queue.
No multi-threading is implied; you are supposed to hack that
yourself, or use a single instance per application.

Each instance is parametrized with two functions, one that is
supposed to return the current time, one that is supposed to
implement a delay.  You can implement real-time scheduling by
substituting time and sleep from built-in module time, or you can
implement simulated time by writing your own functions.  This can
also be used to integrate scheduling with STDWIN events; the delay
function is allowed to modify the queue.  Time can be expressed as
integers or floating point numbers, as long as it is consistent.

Events are specified by tuples (time, priority, action, argument, kwargs).
As in UNIX, lower priority numbers mean higher priority; in this
way the queue can be maintained as a priority queue.  Execution of the
event means calling the action function, passing it the argument
sequence in "argument" (remember that in Python, multiple function
arguments are be packed in a sequence) and keyword parameters in "kwargs".
The action function may be an instance method so it
has another way to reference private data (besides global variables).
"""

import time
import heapq
from collections import namedtuple
from itertools import count
import threading
from time import monotonic as _time

__all__ = ["scheduler"]

Event = namedtuple('Event', 'time, priority, sequence, action, argument, kwargs')
Event.time.__doc__ = ('''Numeric type compatible with the return value of the
timefunc function passed to the constructor.''')
Event.priority.__doc__ = ('''Events scheduled for the same time will be executed
in the order of their priority.''')
Event.sequence.__doc__ = ('''A continually increasing sequence number that
    separates events if time and priority are equal.''')
Event.action.__doc__ = ('''Executing the event means executing
action(*argument, **kwargs)''')
Event.argument.__doc__ = ('''argument is a sequence holding the positional
arguments for the action.''')
Event.kwargs.__doc__ = ('''kwargs is a dictionary holding the keyword
arguments for the action.''')

_sentinel = object()

class scheduler:

    def __init__(self, timefunc=_time, delayfunc=time.sleep):
        """Initialize a new instance, passing the time and delay
        functions"""
        self._queue = []
        self._lock = threading.RLock()
        self.timefunc = timefunc
        self.delayfunc = delayfunc
        self._sequence_generator = count()

    def enterabs(self, time, priority, action, argument=(), kwargs=_sentinel):
        """Enter a new event in the queue at an absolute time.

        Returns an ID for the event which can be used to remove it,
        if necessary.

        """
        if kwargs is _sentinel:
            kwargs = {}

        with self._lock:
            event = Event(time, priority, next(self._sequence_generator),
                          action, argument, kwargs)
            heapq.heappush(self._queue, event)
        return event # The ID

    def enter(self, delay, priority, action, argument=(), kwargs=_sentinel):
        """A variant that specifies the time as a relative time.

        This is actually the more commonly used interface.

        """
        time = self.timefunc() + delay
        return self.enterabs(time, priority, action, argument, kwargs)

    def cancel(self, event):
        """Remove an event from the queue.

        This must be presented the ID as returned by enter().
        If the event is not in the queue, this raises ValueError.

        """
        with self._lock:
            self._queue.remove(event)
            heapq.heapify(self._queue)

    def empty(self):
        """Check whether the queue is empty."""
        with self._lock:
            return not self._queue

    def run(self, blocking=True):
        """Execute events until the queue is empty.
        If blocking is False executes the scheduled events due to
        expire soonest (if any) and then return the deadline of the
        next scheduled call in the scheduler.

        When there is a positive delay until the first event, the
        delay function is called and the event is left in the queue;
        otherwise, the event is removed from the queue and executed
        (its action function is called, passing it the argument).  If
        the delay function returns prematurely, it is simply
        restarted.

        It is legal for both the delay function and the action
        function to modify the queue or to raise an exception;
        exceptions are not caught but the scheduler's state remains
        well-defined so run() may be called again.

        A questionable hack is added to allow other threads to run:
        just after an event is executed, a delay of 0 is executed, to
        avoid monopolizing the CPU when other threads are also
        runnable.

        """
        # localize variable access to minimize overhead
        # and to improve thread safety
        lock = self._lock
        q = self._queue
        delayfunc = self.delayfunc
        timefunc = self.timefunc
        pop = heapq.heappop
        while True:
            with lock:
                if not q:
                    break
                (time, priority, sequence, action,
                 argument, kwargs) = q[0]
                now = timefunc()
                if time > now:
                    delay = True
                else:
                    delay = False
                    pop(q)
            if delay:
                if not blocking:
                    return time - now
                delayfunc(time - now)
            else:
                action(*argument, **kwargs)
                delayfunc(0)   # Let other threads run

    @property
    def queue(self):
        """An ordered list of upcoming events.

        Events are named tuples with fields for:
            time, priority, action, arguments, kwargs

        """
        # Use heapq to sort the queue rather than using 'sorted(self._queue)'.
        # With heapq, two events scheduled at the same time will show in
        # the actual order they would be retrieved.
        with self._lock:
            events = self._queue[:]
        return list(map(heapq.heappop, [events]*len(events)))
<|endoftext|>#! /usr/bin/python3.10

"""Base16, Base32, Base64 (RFC 3548), Base85 and Ascii85 data encodings"""

# Modified 04-Oct-1995 by Jack Jansen to use binascii module
# Modified 30-Dec-2003 by Barry Warsaw to add full RFC 3548 support
# Modified 22-May-2007 by Guido van Rossum to use bytes everywhere

import re
import struct
import binascii


__all__ = [
    # Legacy interface exports traditional RFC 2045 Base64 encodings
    'encode', 'decode', 'encodebytes', 'decodebytes',
    # Generalized interface for other encodings
    'b64encode', 'b64decode', 'b32encode', 'b32decode',
    'b32hexencode', 'b32hexdecode', 'b16encode', 'b16decode',
    # Base85 and Ascii85 encodings
    'b85encode', 'b85decode', 'a85encode', 'a85decode',
    # Standard Base64 encoding
    'standard_b64encode', 'standard_b64decode',
    # Some common Base64 alternatives.  As referenced by RFC 3458, see thread
    # starting at:
    #
    # http://zgp.org/pipermail/p2p-hackers/2001-September/000316.html
    'urlsafe_b64encode', 'urlsafe_b64decode',
    ]


bytes_types = (bytes, bytearray)  # Types acceptable as binary data

def _bytes_from_decode_data(s):
    if isinstance(s, str):
        try:
            return s.encode('ascii')
        except UnicodeEncodeError:
            raise ValueError('string argument should contain only ASCII characters')
    if isinstance(s, bytes_types):
        return s
    try:
        return memoryview(s).tobytes()
    except TypeError:
        raise TypeError("argument should be a bytes-like object or ASCII "
                        "string, not %r" % s.__class__.__name__) from None


# Base64 encoding/decoding uses binascii

def b64encode(s, altchars=None):
    """Encode the bytes-like object s using Base64 and return a bytes object.

    Optional altchars should be a byte string of length 2 which specifies an
    alternative alphabet for the '+' and '/' characters.  This allows an
    application to e.g. generate url or filesystem safe Base64 strings.
    """
    encoded = binascii.b2a_base64(s, newline=False)
    if altchars is not None:
        assert len(altchars) == 2, repr(altchars)
        return encoded.translate(bytes.maketrans(b'+/', altchars))
    return encoded


def b64decode(s, altchars=None, validate=False):
    """Decode the Base64 encoded bytes-like object or ASCII string s.

    Optional altchars must be a bytes-like object or ASCII string of length 2
    which specifies the alternative alphabet used instead of the '+' and '/'
    characters.

    The result is returned as a bytes object.  A binascii.Error is raised if
    s is incorrectly padded.

    If validate is False (the default), characters that are neither in the
    normal base-64 alphabet nor the alternative alphabet are discarded prior
    to the padding check.  If validate is True, these non-alphabet characters
    in the input result in a binascii.Error.
    """
    s = _bytes_from_decode_data(s)
    if altchars is not None:
        altchars = _bytes_from_decode_data(altchars)
        assert len(altchars) == 2, repr(altchars)
        s = s.translate(bytes.maketrans(altchars, b'+/'))
    if validate and not re.fullmatch(b'[A-Za-z0-9+/]*={0,2}', s):
        raise binascii.Error('Non-base64 digit found')
    return binascii.a2b_base64(s)


def standard_b64encode(s):
    """Encode bytes-like object s using the standard Base64 alphabet.

    The result is returned as a bytes object.
    """
    return b64encode(s)

def standard_b64decode(s):
    """Decode bytes encoded with the standard Base64 alphabet.

    Argument s is a bytes-like object or ASCII string to decode.  The result
    is returned as a bytes object.  A binascii.Error is raised if the input
    is incorrectly padded.  Characters that are not in the standard alphabet
    are discarded prior to the padding check.
    """
    return b64decode(s)


_urlsafe_encode_translation = bytes.maketrans(b'+/', b'-_')
_urlsafe_decode_translation = bytes.maketrans(b'-_', b'+/')

def urlsafe_b64encode(s):
    """Encode bytes using the URL- and filesystem-safe Base64 alphabet.

    Argument s is a bytes-like object to encode.  The result is returned as a
    bytes object.  The alphabet uses '-' instead of '+' and '_' instead of
    '/'.
    """
    return b64encode(s).translate(_urlsafe_encode_translation)

def urlsafe_b64decode(s):
    """Decode bytes using the URL- and filesystem-safe Base64 alphabet.

    Argument s is a bytes-like object or ASCII string to decode.  The result
    is returned as a bytes object.  A binascii.Error is raised if the input
    is incorrectly padded.  Characters that are not in the URL-safe base-64
    alphabet, and are not a plus '+' or slash '/', are discarded prior to the
    padding check.

    The alphabet uses '-' instead of '+' and '_' instead of '/'.
    """
    s = _bytes_from_decode_data(s)
    s = s.translate(_urlsafe_decode_translation)
    return b64decode(s)



# Base32 encoding/decoding must be done in Python
_B32_ENCODE_DOCSTRING = '''
Encode the bytes-like objects using {encoding} and return a bytes object.
'''
_B32_DECODE_DOCSTRING = '''
Decode the {encoding} encoded bytes-like object or ASCII string s.

Optional casefold is a flag specifying whether a lowercase alphabet is
acceptable as input.  For security purposes, the default is False.
{extra_args}
The result is returned as a bytes object.  A binascii.Error is raised if
the input is incorrectly padded or if there are non-alphabet
characters present in the input.
'''
_B32_DECODE_MAP01_DOCSTRING = '''
RFC 3548 allows for optional mapping of the digit 0 (zero) to the
letter O (oh), and for optional mapping of the digit 1 (one) to
either the letter I (eye) or letter L (el).  The optional argument
map01 when not None, specifies which letter the digit 1 should be
mapped to (when map01 is not None, the digit 0 is always mapped to
the letter O).  For security purposes the default is None, so that
0 and 1 are not allowed in the input.
'''
_b32alphabet = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'
_b32hexalphabet = b'0123456789ABCDEFGHIJKLMNOPQRSTUV'
_b32tab2 = {}
_b32rev = {}

def _b32encode(alphabet, s):
    global _b32tab2
    # Delay the initialization of the table to not waste memory
    # if the function is never called
    if alphabet not in _b32tab2:
        b32tab = [bytes((i,)) for i in alphabet]
        _b32tab2[alphabet] = [a + b for a in b32tab for b in b32tab]
        b32tab = None

    if not isinstance(s, bytes_types):
        s = memoryview(s).tobytes()
    leftover = len(s) % 5
    # Pad the last quantum with zero bits if necessary
    if leftover:
        s = s + b'\0' * (5 - leftover)  # Don't use += !
    encoded = bytearray()
    from_bytes = int.from_bytes
    b32tab2 = _b32tab2[alphabet]
    for i in range(0, len(s), 5):
        c = from_bytes(s[i: i + 5], 'big')
        encoded += (b32tab2[c >> 30] +           # bits 1 - 10
                    b32tab2[(c >> 20) & 0x3ff] + # bits 11 - 20
                    b32tab2[(c >> 10) & 0x3ff] + # bits 21 - 30
                    b32tab2[c & 0x3ff]           # bits 31 - 40
                   )
    # Adjust for any leftover partial quanta
    if leftover == 1:
        encoded[-6:] = b'======'
    elif leftover == 2:
        encoded[-4:] = b'===='
    elif leftover == 3:
        encoded[-3:] = b'==='
    elif leftover == 4:
        encoded[-1:] = b'='
    return bytes(encoded)

def _b32decode(alphabet, s, casefold=False, map01=None):
    global _b32rev
    # Delay the initialization of the table to not waste memory
    # if the function is never called
    if alphabet not in _b32rev:
        _b32rev[alphabet] = {v: k for k, v in enumerate(alphabet)}
    s = _bytes_from_decode_data(s)
    if len(s) % 8:
        raise binascii.Error('Incorrect padding')
    # Handle section 2.4 zero and one mapping.  The flag map01 will be either
    # False, or the character to map the digit 1 (one) to.  It should be
    # either L (el) or I (eye).
    if map01 is not None:
        map01 = _bytes_from_decode_data(map01)
        assert len(map01) == 1, repr(map01)
        s = s.translate(bytes.maketrans(b'01', b'O' + map01))
    if casefold:
        s = s.upper()
    # Strip off pad characters from the right.  We need to count the pad
    # characters because this will tell us how many null bytes to remove from
    # the end of the decoded string.
    l = len(s)
    s = s.rstrip(b'=')
    padchars = l - len(s)
    # Now decode the full quanta
    decoded = bytearray()
    b32rev = _b32rev[alphabet]
    for i in range(0, len(s), 8):
        quanta = s[i: i + 8]
        acc = 0
        try:
            for c in quanta:
                acc = (acc << 5) + b32rev[c]
        except KeyError:
            raise binascii.Error('Non-base32 digit found') from None
        decoded += acc.to_bytes(5, 'big')
    # Process the last, partial quanta
    if l % 8 or padchars not in {0, 1, 3, 4, 6}:
        raise binascii.Error('Incorrect padding')
    if padchars and decoded:
        acc <<= 5 * padchars
        last = acc.to_bytes(5, 'big')
        leftover = (43 - 5 * padchars) // 8  # 1: 4, 3: 3, 4: 2, 6: 1
        decoded[-5:] = last[:leftover]
    return bytes(decoded)


def b32encode(s):
    return _b32encode(_b32alphabet, s)
b32encode.__doc__ = _B32_ENCODE_DOCSTRING.format(encoding='base32')

def b32decode(s, casefold=False, map01=None):
    return _b32decode(_b32alphabet, s, casefold, map01)
b32decode.__doc__ = _B32_DECODE_DOCSTRING.format(encoding='base32',
                                        extra_args=_B32_DECODE_MAP01_DOCSTRING)

def b32hexencode(s):
    return _b32encode(_b32hexalphabet, s)
b32hexencode.__doc__ = _B32_ENCODE_DOCSTRING.format(encoding='base32hex')

def b32hexdecode(s, casefold=False):
    # base32hex does not have the 01 mapping
    return _b32decode(_b32hexalphabet, s, casefold)
b32hexdecode.__doc__ = _B32_DECODE_DOCSTRING.format(encoding='base32hex',
                                                    extra_args='')


# RFC 3548, Base 16 Alphabet specifies uppercase, but hexlify() returns
# lowercase.  The RFC also recommends against accepting input case
# insensitively.
def b16encode(s):
    """Encode the bytes-like object s using Base16 and return a bytes object.
    """
    return binascii.hexlify(s).upper()


def b16decode(s, casefold=False):
    """Decode the Base16 encoded bytes-like object or ASCII string s.

    Optional casefold is a flag specifying whether a lowercase alphabet is
    acceptable as input.  For security purposes, the default is False.

    The result is returned as a bytes object.  A binascii.Error is raised if
    s is incorrectly padded or if there are non-alphabet characters present
    in the input.
    """
    s = _bytes_from_decode_data(s)
    if casefold:
        s = s.upper()
    if re.search(b'[^0-9A-F]', s):
        raise binascii.Error('Non-base16 digit found')
    return binascii.unhexlify(s)

#
# Ascii85 encoding/decoding
#

_a85chars = None
_a85chars2 = None
_A85START = b"<~"
_A85END = b"~>"

def _85encode(b, chars, chars2, pad=False, foldnuls=False, foldspaces=False):
    # Helper function for a85encode and b85encode
    if not isinstance(b, bytes_types):
        b = memoryview(b).tobytes()

    padding = (-len(b)) % 4
    if padding:
        b = b + b'\0' * padding
    words = struct.Struct('!%dI' % (len(b) // 4)).unpack(b)

    chunks = [b'z' if foldnuls and not word else
              b'y' if foldspaces and word == 0x20202020 else
              (chars2[word // 614125] +
               chars2[word // 85 % 7225] +
               chars[word % 85])
              for word in words]

    if padding and not pad:
        if chunks[-1] == b'z':
            chunks[-1] = chars[0] * 5
        chunks[-1] = chunks[-1][:-padding]

    return b''.join(chunks)

def a85encode(b, *, foldspaces=False, wrapcol=0, pad=False, adobe=False):
    """Encode bytes-like object b using Ascii85 and return a bytes object.

    foldspaces is an optional flag that uses the special short sequence 'y'
    instead of 4 consecutive spaces (ASCII 0x20) as supported by 'btoa'. This
    feature is not supported by the "standard" Adobe encoding.

    wrapcol controls whether the output should have newline (b'\\n') characters
    added to it. If this is non-zero, each output line will be at most this
    many characters long.

    pad controls whether the input is padded to a multiple of 4 before
    encoding. Note that the btoa implementation always pads.

    adobe controls whether the encoded byte sequence is framed with <~ and ~>,
    which is used by the Adobe implementation.
    """
    global _a85chars, _a85chars2
    # Delay the initialization of tables to not waste memory
    # if the function is never called
    if _a85chars2 is None:
        _a85chars = [bytes((i,)) for i in range(33, 118)]
        _a85chars2 = [(a + b) for a in _a85chars for b in _a85chars]

    result = _85encode(b, _a85chars, _a85chars2, pad, True, foldspaces)

    if adobe:
        result = _A85START + result
    if wrapcol:
        wrapcol = max(2 if adobe else 1, wrapcol)
        chunks = [result[i: i + wrapcol]
                  for i in range(0, len(result), wrapcol)]
        if adobe:
            if len(chunks[-1]) + 2 > wrapcol:
                chunks.append(b'')
        result = b'\n'.join(chunks)
    if adobe:
        result += _A85END

    return result

def a85decode(b, *, foldspaces=False, adobe=False, ignorechars=b' \t\n\r\v'):
    """Decode the Ascii85 encoded bytes-like object or ASCII string b.

    foldspaces is a flag that specifies whether the 'y' short sequence should be
    accepted as shorthand for 4 consecutive spaces (ASCII 0x20). This feature is
    not supported by the "standard" Adobe encoding.

    adobe controls whether the input sequence is in Adobe Ascii85 format (i.e.
    is framed with <~ and ~>).

    ignorechars should be a byte string containing characters to ignore from the
    input. This should only contain whitespace characters, and by default
    contains all whitespace characters in ASCII.

    The result is returned as a bytes object.
    """
    b = _bytes_from_decode_data(b)
    if adobe:
        if not b.endswith(_A85END):
            raise ValueError(
                "Ascii85 encoded byte sequences must end "
                "with {!r}".format(_A85END)
                )
        if b.startswith(_A85START):
            b = b[2:-2]  # Strip off start/end markers
        else:
            b = b[:-2]
    #
    # We have to go through this stepwise, so as to ignore spaces and handle
    # special short sequences
    #
    packI = struct.Struct('!I').pack
    decoded = []
    decoded_append = decoded.append
    curr = []
    curr_append = curr.append
    curr_clear = curr.clear
    for x in b + b'u' * 4:
        if b'!'[0] <= x <= b'u'[0]:
            curr_append(x)
            if len(curr) == 5:
                acc = 0
                for x in curr:
                    acc = 85 * acc + (x - 33)
                try:
                    decoded_append(packI(acc))
                except struct.error:
                    raise ValueError('Ascii85 overflow') from None
                curr_clear()
        elif x == b'z'[0]:
            if curr:
                raise ValueError('z inside Ascii85 5-tuple')
            decoded_append(b'\0\0\0\0')
        elif foldspaces and x == b'y'[0]:
            if curr:
                raise ValueError('y inside Ascii85 5-tuple')
            decoded_append(b'\x20\x20\x20\x20')
        elif x in ignorechars:
            # Skip whitespace
            continue
        else:
            raise ValueError('Non-Ascii85 digit found: %c' % x)

    result = b''.join(decoded)
    padding = 4 - len(curr)
    if padding:
        # Throw away the extra padding
        result = result[:-padding]
    return result

# The following code is originally taken (with permission) from Mercurial

_b85alphabet = (b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                b"abcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~")
_b85chars = None
_b85chars2 = None
_b85dec = None

def b85encode(b, pad=False):
    """Encode bytes-like object b in base85 format and return a bytes object.

    If pad is true, the input is padded with b'\\0' so its length is a multiple of
    4 bytes before encoding.
    """
    global _b85chars, _b85chars2
    # Delay the initialization of tables to not waste memory
    # if the function is never called
    if _b85chars2 is None:
        _b85chars = [bytes((i,)) for i in _b85alphabet]
        _b85chars2 = [(a + b) for a in _b85chars for b in _b85chars]
    return _85encode(b, _b85chars, _b85chars2, pad)

def b85decode(b):
    """Decode the base85-encoded bytes-like object or ASCII string b

    The result is returned as a bytes object.
    """
    global _b85dec
    # Delay the initialization of tables to not waste memory
    # if the function is never called
    if _b85dec is None:
        _b85dec = [None] * 256
        for i, c in enumerate(_b85alphabet):
            _b85dec[c] = i

    b = _bytes_from_decode_data(b)
    padding = (-len(b)) % 5
    b = b + b'~' * padding
    out = []
    packI = struct.Struct('!I').pack
    for i in range(0, len(b), 5):
        chunk = b[i:i + 5]
        acc = 0
        try:
            for c in chunk:
                acc = acc * 85 + _b85dec[c]
        except TypeError:
            for j, c in enumerate(chunk):
                if _b85dec[c] is None:
                    raise ValueError('bad base85 character at position %d'
                                    % (i + j)) from None
            raise
        try:
            out.append(packI(acc))
        except struct.error:
            raise ValueError('base85 overflow in hunk starting at byte %d'
                             % i) from None

    result = b''.join(out)
    if padding:
        result = result[:-padding]
    return result

# Legacy interface.  This code could be cleaned up since I don't believe
# binascii has any line length limitations.  It just doesn't seem worth it
# though.  The files should be opened in binary mode.

MAXLINESIZE = 76 # Excluding the CRLF
MAXBINSIZE = (MAXLINESIZE//4)*3

def encode(input, output):
    """Encode a file; input and output are binary files."""
    while True:
        s = input.read(MAXBINSIZE)
        if not s:
            break
        while len(s) < MAXBINSIZE:
            ns = input.read(MAXBINSIZE-len(s))
            if not ns:
                break
            s += ns
        line = binascii.b2a_base64(s)
        output.write(line)


def decode(input, output):
    """Decode a file; input and output are binary files."""
    while True:
        line = input.readline()
        if not line:
            break
        s = binascii.a2b_base64(line)
        output.write(s)

def _input_type_check(s):
    try:
        m = memoryview(s)
    except TypeError as err:
        msg = "expected bytes-like object, not %s" % s.__class__.__name__
        raise TypeError(msg) from err
    if m.format not in ('c', 'b', 'B'):
        msg = ("expected single byte elements, not %r from %s" %
                                          (m.format, s.__class__.__name__))
        raise TypeError(msg)
    if m.ndim != 1:
        msg = ("expected 1-D data, not %d-D data from %s" %
                                          (m.ndim, s.__class__.__name__))
        raise TypeError(msg)


def encodebytes(s):
    """Encode a bytestring into a bytes object containing multiple lines
    of base-64 data."""
    _input_type_check(s)
    pieces = []
    for i in range(0, len(s), MAXBINSIZE):
        chunk = s[i : i + MAXBINSIZE]
        pieces.append(binascii.b2a_base64(chunk))
    return b"".join(pieces)


def decodebytes(s):
    """Decode a bytestring of base-64 data into a bytes object."""
    _input_type_check(s)
    return binascii.a2b_base64(s)


# Usable as a script...
def main():
    """Small main program"""
    import sys, getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'deut')
    except getopt.error as msg:
        sys.stdout = sys.stderr
        print(msg)
        print("""usage: %s [-d|-e|-u|-t] [file|-]
        -d, -u: decode
        -e: encode (default)
        -t: encode and decode string 'Aladdin:open sesame'"""%sys.argv[0])
        sys.exit(2)
    func = encode
    for o, a in opts:
        if o == '-e': func = encode
        if o == '-d': func = decode
        if o == '-u': func = decode
        if o == '-t': test(); return
    if args and args[0] != '-':
        with open(args[0], 'rb') as f:
            func(f, sys.stdout.buffer)
    else:
        func(sys.stdin.buffer, sys.stdout.buffer)


def test():
    s0 = b"Aladdin:open sesame"
    print(repr(s0))
    s1 = encodebytes(s0)
    print(repr(s1))
    s2 = decodebytes(s1)
    print(repr(s2))
    assert s0 == s2


if __name__ == '__main__':
    main()
<|endoftext|>"""Temporary files.

This module provides generic, low- and high-level interfaces for
creating temporary files and directories.  All of the interfaces
provided by this module can be used without fear of race conditions
except for 'mktemp'.  'mktemp' is subject to race conditions and
should not be used; it is provided for backward compatibility only.

The default path names are returned as str.  If you supply bytes as
input, all return values will be in bytes.  Ex:

    >>> tempfile.mkstemp()
    (4, '/tmp/tmptpu9nin8')
    >>> tempfile.mkdtemp(suffix=b'')
    b'/tmp/tmppbi8f0hy'

This module also provides some data items to the user:

  TMP_MAX  - maximum number of names that will be tried before
             giving up.
  tempdir  - If this is set to a string before the first use of
             any routine from this module, it will be considered as
             another candidate location to store temporary files.
"""

__all__ = [
    "NamedTemporaryFile", "TemporaryFile", # high level safe interfaces
    "SpooledTemporaryFile", "TemporaryDirectory",
    "mkstemp", "mkdtemp",                  # low level safe interfaces
    "mktemp",                              # deprecated unsafe interface
    "TMP_MAX", "gettempprefix",            # constants
    "tempdir", "gettempdir",
    "gettempprefixb", "gettempdirb",
   ]


# Imports.

import functools as _functools
import warnings as _warnings
import io as _io
import os as _os
try:
  import shutil as _shutil
  _rmtree = _shutil.rmtree
except ImportError:
  import sys as _sys
  import stat as _stat
  # version vulnerable to race conditions
  def _rmtree_unsafe(path, onerror):
    try:
        if _os.path.islink(path):
            # symlinks to directories are forbidden, see bug #1669
            raise OSError("Cannot call rmtree on a symbolic link")
    except OSError:
        onerror(_os.path.islink, path, _sys.exc_info())
        # can't continue even if onerror hook returns
        return
    names = []
    try:
        names = _os.listdir(path)
    except OSError:
        onerror(_os.listdir, path, _sys.exc_info())
    for name in names:
        fullname = _os.path.join(path, name)
        try:
            mode = _os.lstat(fullname).st_mode
        except OSError:
            mode = 0
        if _stat.S_ISDIR(mode):
            _rmtree_unsafe(fullname, onerror)
        else:
            try:
                _os.unlink(fullname)
            except OSError:
                onerror(_os.unlink, fullname, _sys.exc_info())
    try:
        _os.rmdir(path)
    except OSError:
        onerror(_os.rmdir, path, _sys.exc_info())

  # Version using fd-based APIs to protect against races
  def _rmtree_safe_fd(topfd, path, onerror):
    names = []
    try:
        names = _os.listdir(topfd)
    except OSError as err:
        err.filename = path
        onerror(_os.listdir, path, _sys.exc_info())
    for name in names:
        fullname = _os.path.join(path, name)
        try:
            orig_st = _os.stat(name, dir_fd=topfd, follow_symlinks=False)
            mode = orig_st.st_mode
        except OSError:
            mode = 0
        if _stat.S_ISDIR(mode):
            try:
                dirfd = _os.open(name, _os.O_RDONLY, dir_fd=topfd)
            except OSError:
                onerror(_os.open, fullname, _sys.exc_info())
            else:
                try:
                    if _os.path.samestat(orig_st, _os.fstat(dirfd)):
                        _rmtree_safe_fd(dirfd, fullname, onerror)
                        try:
                            _os.rmdir(name, dir_fd=topfd)
                        except OSError:
                            onerror(_os.rmdir, fullname, _sys.exc_info())
                    else:
                        try:
                            # This can only happen if someone replaces
                            # a directory with a symlink after the call to
                            # stat.S_ISDIR above.
                            raise OSError("Cannot call rmtree on a symbolic "
                                          "link")
                        except OSError:
                            onerror(_os.path.islink, fullname, _sys.exc_info())
                finally:
                    _os.close(dirfd)
        else:
            try:
                _os.unlink(name, dir_fd=topfd)
            except OSError:
                onerror(_os.unlink, fullname, _sys.exc_info())

  _use_fd_functions = ({_os.open, _os.stat, _os.unlink, _os.rmdir} <=
                     _os.supports_dir_fd and
                     _os.listdir in _os.supports_fd and
                     _os.stat in _os.supports_follow_symlinks)

  def _rmtree(path, ignore_errors=False, onerror=None):
    """Recursively delete a directory tree.

    If ignore_errors is set, errors are ignored; otherwise, if onerror
    is set, it is called to handle the error with arguments (func,
    path, exc_info) where func is platform and implementation dependent;
    path is the argument to that function that caused it to fail; and
    exc_info is a tuple returned by sys.exc_info().  If ignore_errors
    is false and onerror is None, an exception is raised.

    """
    if ignore_errors:
        def onerror(*args):
            pass
    elif onerror is None:
        def onerror(*args):
            raise
    if _use_fd_functions:
        # While the unsafe rmtree works fine on bytes, the fd based does not.
        if isinstance(path, bytes):
            path = _os.fsdecode(path)
        # Note: To guard against symlink races, we use the standard
        # lstat()/open()/fstat() trick.
        try:
            orig_st = _os.lstat(path)
        except Exception:
            onerror(_os.lstat, path, _sys.exc_info())
            return
        try:
            fd = _os.open(path, _os.O_RDONLY)
        except Exception:
            onerror(_os.lstat, path, _sys.exc_info())
            return
        try:
            if _os.path.samestat(orig_st, _os.fstat(fd)):
                _rmtree_safe_fd(fd, path, onerror)
                try:
                    _os.rmdir(path)
                except OSError:
                    onerror(_os.rmdir, path, _sys.exc_info())
            else:
                try:
                    # symlinks to directories are forbidden, see bug #1669
                    raise OSError("Cannot call rmtree on a symbolic link")
                except OSError:
                    onerror(_os.path.islink, path, _sys.exc_info())
        finally:
            _os.close(fd)
    else:
        return _rmtree_unsafe(path, onerror)

import errno as _errno
from random import Random as _Random
import sys as _sys
import types as _types
import weakref as _weakref
import _thread
_allocate_lock = _thread.allocate_lock

_text_openflags = _os.O_RDWR | _os.O_CREAT | _os.O_EXCL
if hasattr(_os, 'O_NOFOLLOW'):
    _text_openflags |= _os.O_NOFOLLOW

_bin_openflags = _text_openflags
if hasattr(_os, 'O_BINARY'):
    _bin_openflags |= _os.O_BINARY

if hasattr(_os, 'TMP_MAX'):
    TMP_MAX = _os.TMP_MAX
else:
    TMP_MAX = 10000

# This variable _was_ unused for legacy reasons, see issue 10354.
# But as of 3.5 we actually use it at runtime so changing it would
# have a possibly desirable side effect...  But we do not want to support
# that as an API.  It is undocumented on purpose.  Do not depend on this.
template = "tmp"

# Internal routines.

_once_lock = _allocate_lock()


def _exists(fn):
    try:
        _os.lstat(fn)
    except OSError:
        return False
    else:
        return True


def _infer_return_type(*args):
    """Look at the type of all args and divine their implied return type."""
    return_type = None
    for arg in args:
        if arg is None:
            continue

        if isinstance(arg, _os.PathLike):
            arg = _os.fspath(arg)

        if isinstance(arg, bytes):
            if return_type is str:
                raise TypeError("Can't mix bytes and non-bytes in "
                                "path components.")
            return_type = bytes
        else:
            if return_type is bytes:
                raise TypeError("Can't mix bytes and non-bytes in "
                                "path components.")
            return_type = str
    if return_type is None:
        if tempdir is None or isinstance(tempdir, str):
            return str  # tempfile APIs return a str by default.
        else:
            # we could check for bytes but it'll fail later on anyway
            return bytes
    return return_type


def _sanitize_params(prefix, suffix, dir):
    """Common parameter processing for most APIs in this module."""
    output_type = _infer_return_type(prefix, suffix, dir)
    if suffix is None:
        suffix = output_type()
    if prefix is None:
        if output_type is str:
            prefix = template
        else:
            prefix = _os.fsencode(template)
    if dir is None:
        if output_type is str:
            dir = gettempdir()
        else:
            dir = gettempdirb()
    return prefix, suffix, dir, output_type


class _RandomNameSequence:
    """An instance of _RandomNameSequence generates an endless
    sequence of unpredictable strings which can safely be incorporated
    into file names.  Each string is eight characters long.  Multiple
    threads can safely use the same instance at the same time.

    _RandomNameSequence is an iterator."""

    characters = "abcdefghijklmnopqrstuvwxyz0123456789_"

    @property
    def rng(self):
        cur_pid = _os.getpid()
        if cur_pid != getattr(self, '_rng_pid', None):
            self._rng = _Random()
            self._rng_pid = cur_pid
        return self._rng

    def __iter__(self):
        return self

    def __next__(self):
        return ''.join(self.rng.choices(self.characters, k=8))

def _candidate_tempdir_list():
    """Generate a list of candidate temporary directories which
    _get_default_tempdir will try."""

    dirlist = []

    # First, try the environment.
    for envname in 'TMPDIR', 'TEMP', 'TMP':
        dirname = _os.getenv(envname)
        if dirname: dirlist.append(dirname)

    # Failing that, try OS-specific locations.
    if _os.name == 'nt':
        dirlist.extend([ _os.path.expanduser(r'~\AppData\Local\Temp'),
                         _os.path.expandvars(r'%SYSTEMROOT%\Temp'),
                         r'c:\temp', r'c:\tmp', r'\temp', r'\tmp' ])
    else:
        dirlist.extend([ '/tmp', '/var/tmp', '/usr/tmp' ])

    # As a last resort, the current directory.
    try:
        dirlist.append(_os.getcwd())
    except (AttributeError, OSError):
        dirlist.append(_os.curdir)

    return dirlist

def _get_default_tempdir():
    """Calculate the default directory to use for temporary files.
    This routine should be called exactly once.

    We determine whether or not a candidate temp dir is usable by
    trying to create and write to a file in that directory.  If this
    is successful, the test file is deleted.  To prevent denial of
    service, the name of the test file must be randomized."""

    namer = _RandomNameSequence()
    dirlist = _candidate_tempdir_list()

    for dir in dirlist:
        if dir != _os.curdir:
            dir = _os.path.abspath(dir)
        # Try only a few names per directory.
        for seq in range(100):
            name = next(namer)
            filename = _os.path.join(dir, name)
            try:
                fd = _os.open(filename, _bin_openflags, 0o600)
                try:
                    try:
                        _os.write(fd, b'blat')
                    finally:
                        _os.close(fd)
                finally:
                    _os.unlink(filename)
                return dir
            except FileExistsError:
                pass
            except PermissionError:
                # This exception is thrown when a directory with the chosen name
                # already exists on windows.
                if (_os.name == 'nt' and _os.path.isdir(dir) and
                    _os.access(dir, _os.W_OK)):
                    continue
                break   # no point trying more names in this directory
            except OSError:
                break   # no point trying more names in this directory
    raise FileNotFoundError(_errno.ENOENT,
                            "No usable temporary directory found in %s" %
                            dirlist)

_name_sequence = None

def _get_candidate_names():
    """Common setup sequence for all user-callable interfaces."""

    global _name_sequence
    if _name_sequence is None:
        _once_lock.acquire()
        try:
            if _name_sequence is None:
                _name_sequence = _RandomNameSequence()
        finally:
            _once_lock.release()
    return _name_sequence


def _mkstemp_inner(dir, pre, suf, flags, output_type):
    """Code common to mkstemp, TemporaryFile, and NamedTemporaryFile."""

    dir = _os.path.abspath(dir)
    names = _get_candidate_names()
    if output_type is bytes:
        names = map(_os.fsencode, names)

    for seq in range(TMP_MAX):
        name = next(names)
        file = _os.path.join(dir, pre + name + suf)
        _sys.audit("tempfile.mkstemp", file)
        try:
            fd = _os.open(file, flags, 0o600)
        except FileExistsError:
            continue    # try again
        except PermissionError:
            # This exception is thrown when a directory with the chosen name
            # already exists on windows.
            if (_os.name == 'nt' and _os.path.isdir(dir) and
                _os.access(dir, _os.W_OK)):
                continue
            else:
                raise
        return fd, file

    raise FileExistsError(_errno.EEXIST,
                          "No usable temporary file name found")

def _dont_follow_symlinks(func, path, *args):
    # Pass follow_symlinks=False, unless not supported on this platform.
    if func in _os.supports_follow_symlinks:
        func(path, *args, follow_symlinks=False)
    elif _os.name == 'nt' or not _os.path.islink(path):
        func(path, *args)

def _resetperms(path):
    try:
        chflags = _os.chflags
    except AttributeError:
        pass
    else:
        _dont_follow_symlinks(chflags, path, 0)
    _dont_follow_symlinks(_os.chmod, path, 0o700)


# User visible interfaces.

def gettempprefix():
    """The default prefix for temporary directories as string."""
    return _os.fsdecode(template)

def gettempprefixb():
    """The default prefix for temporary directories as bytes."""
    return _os.fsencode(template)

tempdir = None

def _gettempdir():
    """Private accessor for tempfile.tempdir."""
    global tempdir
    if tempdir is None:
        _once_lock.acquire()
        try:
            if tempdir is None:
                tempdir = _get_default_tempdir()
        finally:
            _once_lock.release()
    return tempdir

def gettempdir():
    """Returns tempfile.tempdir as str."""
    return _os.fsdecode(_gettempdir())

def gettempdirb():
    """Returns tempfile.tempdir as bytes."""
    return _os.fsencode(_gettempdir())

