        M._mesg('%s => %s %s' % (cmd, typ, dat))
        if typ == 'NO': raise dat[0]
        return dat

    try:
        if stream_command:
            M = IMAP4_stream(stream_command)
        else:
            M = IMAP4(host)
        if M.state == 'AUTH':
            test_seq1 = test_seq1[1:]   # Login not needed
        M._mesg('PROTOCOL_VERSION = %s' % M.PROTOCOL_VERSION)
        M._mesg('CAPABILITIES = %r' % (M.capabilities,))

        for cmd,args in test_seq1:
            run(cmd, args)

        for ml in run('list', ('/tmp/', 'yy%')):
            mo = re.match(r'.*"([^"]+)"$', ml)
            if mo: path = mo.group(1)
            else: path = ml.split()[-1]
            run('delete', (path,))

        for cmd,args in test_seq2:
            dat = run(cmd, args)

            if (cmd,args) != ('uid', ('SEARCH', 'ALL')):
                continue

            uid = dat[-1].split()
            if not uid: continue
            run('uid', ('FETCH', '%s' % uid[-1],
                    '(FLAGS INTERNALDATE RFC822.SIZE RFC822.HEADER RFC822.TEXT)'))

        print('\nAll tests OK.')

    except:
        print('\nTests failed.')

        if not Debug:
            print('''
If you would like to see debugging output,
try: %s -d5
''' % sys.argv[0])

        raise
<|endoftext|># This module is used to map the old Python 2 names to the new names used in
# Python 3 for the pickle module.  This needed to make pickle streams
# generated with Python 2 loadable by Python 3.

# This is a copy of lib2to3.fixes.fix_imports.MAPPING.  We cannot import
# lib2to3 and use the mapping defined there, because lib2to3 uses pickle.
# Thus, this could cause the module to be imported recursively.
IMPORT_MAPPING = {
    '__builtin__' : 'builtins',
    'copy_reg': 'copyreg',
    'Queue': 'queue',
    'SocketServer': 'socketserver',
    'ConfigParser': 'configparser',
    'repr': 'reprlib',
    'tkFileDialog': 'tkinter.filedialog',
    'tkSimpleDialog': 'tkinter.simpledialog',
    'tkColorChooser': 'tkinter.colorchooser',
    'tkCommonDialog': 'tkinter.commondialog',
    'Dialog': 'tkinter.dialog',
    'Tkdnd': 'tkinter.dnd',
    'tkFont': 'tkinter.font',
    'tkMessageBox': 'tkinter.messagebox',
    'ScrolledText': 'tkinter.scrolledtext',
    'Tkconstants': 'tkinter.constants',
    'Tix': 'tkinter.tix',
    'ttk': 'tkinter.ttk',
    'Tkinter': 'tkinter',
    'markupbase': '_markupbase',
    '_winreg': 'winreg',
    'thread': '_thread',
    'dummy_thread': '_dummy_thread',
    'dbhash': 'dbm.bsd',
    'dumbdbm': 'dbm.dumb',
    'dbm': 'dbm.ndbm',
    'gdbm': 'dbm.gnu',
    'xmlrpclib': 'xmlrpc.client',
    'SimpleXMLRPCServer': 'xmlrpc.server',
    'httplib': 'http.client',
    'htmlentitydefs' : 'html.entities',
    'HTMLParser' : 'html.parser',
    'Cookie': 'http.cookies',
    'cookielib': 'http.cookiejar',
    'BaseHTTPServer': 'http.server',
    'test.test_support': 'test.support',
    'commands': 'subprocess',
    'urlparse' : 'urllib.parse',
    'robotparser' : 'urllib.robotparser',
    'urllib2': 'urllib.request',
    'anydbm': 'dbm',
    '_abcoll' : 'collections.abc',
}


# This contains rename rules that are easy to handle.  We ignore the more
# complex stuff (e.g. mapping the names in the urllib and types modules).
# These rules should be run before import names are fixed.
NAME_MAPPING = {
    ('__builtin__', 'xrange'):     ('builtins', 'range'),
    ('__builtin__', 'reduce'):     ('functools', 'reduce'),
    ('__builtin__', 'intern'):     ('sys', 'intern'),
    ('__builtin__', 'unichr'):     ('builtins', 'chr'),
    ('__builtin__', 'unicode'):    ('builtins', 'str'),
    ('__builtin__', 'long'):       ('builtins', 'int'),
    ('itertools', 'izip'):         ('builtins', 'zip'),
    ('itertools', 'imap'):         ('builtins', 'map'),
    ('itertools', 'ifilter'):      ('builtins', 'filter'),
    ('itertools', 'ifilterfalse'): ('itertools', 'filterfalse'),
    ('itertools', 'izip_longest'): ('itertools', 'zip_longest'),
    ('UserDict', 'IterableUserDict'): ('collections', 'UserDict'),
    ('UserList', 'UserList'): ('collections', 'UserList'),
    ('UserString', 'UserString'): ('collections', 'UserString'),
    ('whichdb', 'whichdb'): ('dbm', 'whichdb'),
    ('_socket', 'fromfd'): ('socket', 'fromfd'),
    ('_multiprocessing', 'Connection'): ('multiprocessing.connection', 'Connection'),
    ('multiprocessing.process', 'Process'): ('multiprocessing.context', 'Process'),
    ('multiprocessing.forking', 'Popen'): ('multiprocessing.popen_fork', 'Popen'),
    ('urllib', 'ContentTooShortError'): ('urllib.error', 'ContentTooShortError'),
    ('urllib', 'getproxies'): ('urllib.request', 'getproxies'),
    ('urllib', 'pathname2url'): ('urllib.request', 'pathname2url'),
    ('urllib', 'quote_plus'): ('urllib.parse', 'quote_plus'),
    ('urllib', 'quote'): ('urllib.parse', 'quote'),
    ('urllib', 'unquote_plus'): ('urllib.parse', 'unquote_plus'),
    ('urllib', 'unquote'): ('urllib.parse', 'unquote'),
    ('urllib', 'url2pathname'): ('urllib.request', 'url2pathname'),
    ('urllib', 'urlcleanup'): ('urllib.request', 'urlcleanup'),
    ('urllib', 'urlencode'): ('urllib.parse', 'urlencode'),
    ('urllib', 'urlopen'): ('urllib.request', 'urlopen'),
    ('urllib', 'urlretrieve'): ('urllib.request', 'urlretrieve'),
    ('urllib2', 'HTTPError'): ('urllib.error', 'HTTPError'),
    ('urllib2', 'URLError'): ('urllib.error', 'URLError'),
}

PYTHON2_EXCEPTIONS = (
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "BufferError",
    "BytesWarning",
    "DeprecationWarning",
    "EOFError",
    "EnvironmentError",
    "Exception",
    "FloatingPointError",
    "FutureWarning",
    "GeneratorExit",
    "IOError",
    "ImportError",
    "ImportWarning",
    "IndentationError",
    "IndexError",
    "KeyError",
    "KeyboardInterrupt",
    "LookupError",
    "MemoryError",
    "NameError",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "PendingDeprecationWarning",
    "ReferenceError",
    "RuntimeError",
    "RuntimeWarning",
    # StandardError is gone in Python 3, so we map it to Exception
    "StopIteration",
    "SyntaxError",
    "SyntaxWarning",
    "SystemError",
    "SystemExit",
    "TabError",
    "TypeError",
    "UnboundLocalError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeError",
    "UnicodeTranslateError",
    "UnicodeWarning",
    "UserWarning",
    "ValueError",
    "Warning",
    "ZeroDivisionError",
)

try:
    WindowsError
except NameError:
    pass
else:
    PYTHON2_EXCEPTIONS += ("WindowsError",)

for excname in PYTHON2_EXCEPTIONS:
    NAME_MAPPING[("exceptions", excname)] = ("builtins", excname)

MULTIPROCESSING_EXCEPTIONS = (
    'AuthenticationError',
    'BufferTooShort',
    'ProcessError',
    'TimeoutError',
)

for excname in MULTIPROCESSING_EXCEPTIONS:
    NAME_MAPPING[("multiprocessing", excname)] = ("multiprocessing.context", excname)

# Same, but for 3.x to 2.x
REVERSE_IMPORT_MAPPING = dict((v, k) for (k, v) in IMPORT_MAPPING.items())
assert len(REVERSE_IMPORT_MAPPING) == len(IMPORT_MAPPING)
REVERSE_NAME_MAPPING = dict((v, k) for (k, v) in NAME_MAPPING.items())
assert len(REVERSE_NAME_MAPPING) == len(NAME_MAPPING)

# Non-mutual mappings.

IMPORT_MAPPING.update({
    'cPickle': 'pickle',
    '_elementtree': 'xml.etree.ElementTree',
    'FileDialog': 'tkinter.filedialog',
    'SimpleDialog': 'tkinter.simpledialog',
    'DocXMLRPCServer': 'xmlrpc.server',
    'SimpleHTTPServer': 'http.server',
    'CGIHTTPServer': 'http.server',
    # For compatibility with broken pickles saved in old Python 3 versions
    'UserDict': 'collections',
    'UserList': 'collections',
    'UserString': 'collections',
    'whichdb': 'dbm',
    'StringIO':  'io',
    'cStringIO': 'io',
})

REVERSE_IMPORT_MAPPING.update({
    '_bz2': 'bz2',
    '_dbm': 'dbm',
    '_functools': 'functools',
    '_gdbm': 'gdbm',
    '_pickle': 'pickle',
})

NAME_MAPPING.update({
    ('__builtin__', 'basestring'): ('builtins', 'str'),
    ('exceptions', 'StandardError'): ('builtins', 'Exception'),
    ('UserDict', 'UserDict'): ('collections', 'UserDict'),
    ('socket', '_socketobject'): ('socket', 'SocketType'),
})

REVERSE_NAME_MAPPING.update({
    ('_functools', 'reduce'): ('__builtin__', 'reduce'),
    ('tkinter.filedialog', 'FileDialog'): ('FileDialog', 'FileDialog'),
    ('tkinter.filedialog', 'LoadFileDialog'): ('FileDialog', 'LoadFileDialog'),
    ('tkinter.filedialog', 'SaveFileDialog'): ('FileDialog', 'SaveFileDialog'),
    ('tkinter.simpledialog', 'SimpleDialog'): ('SimpleDialog', 'SimpleDialog'),
    ('xmlrpc.server', 'ServerHTMLDoc'): ('DocXMLRPCServer', 'ServerHTMLDoc'),
    ('xmlrpc.server', 'XMLRPCDocGenerator'):
        ('DocXMLRPCServer', 'XMLRPCDocGenerator'),
    ('xmlrpc.server', 'DocXMLRPCRequestHandler'):
        ('DocXMLRPCServer', 'DocXMLRPCRequestHandler'),
    ('xmlrpc.server', 'DocXMLRPCServer'):
        ('DocXMLRPCServer', 'DocXMLRPCServer'),
    ('xmlrpc.server', 'DocCGIXMLRPCRequestHandler'):
        ('DocXMLRPCServer', 'DocCGIXMLRPCRequestHandler'),
    ('http.server', 'SimpleHTTPRequestHandler'):
        ('SimpleHTTPServer', 'SimpleHTTPRequestHandler'),
    ('http.server', 'CGIHTTPRequestHandler'):
        ('CGIHTTPServer', 'CGIHTTPRequestHandler'),
    ('_socket', 'socket'): ('socket', '_socketobject'),
})

PYTHON3_OSERROR_EXCEPTIONS = (
    'BrokenPipeError',
    'ChildProcessError',
    'ConnectionAbortedError',
    'ConnectionError',
    'ConnectionRefusedError',
    'ConnectionResetError',
    'FileExistsError',
    'FileNotFoundError',
    'InterruptedError',
    'IsADirectoryError',
    'NotADirectoryError',
    'PermissionError',
    'ProcessLookupError',
    'TimeoutError',
)

for excname in PYTHON3_OSERROR_EXCEPTIONS:
    REVERSE_NAME_MAPPING[('builtins', excname)] = ('exceptions', 'OSError')

PYTHON3_IMPORTERROR_EXCEPTIONS = (
    'ModuleNotFoundError',
)

for excname in PYTHON3_IMPORTERROR_EXCEPTIONS:
    REVERSE_NAME_MAPPING[('builtins', excname)] = ('exceptions', 'ImportError')
<|endoftext|>__all__ = [
    # Functions
    'calcsize', 'pack', 'pack_into', 'unpack', 'unpack_from',
    'iter_unpack',

    # Classes
    'Struct',

    # Exceptions
    'error'
    ]

from _struct import *
from _struct import _clearcache
from _struct import __doc__
<|endoftext|>"""Routines to help recognizing sound files.

Function whathdr() recognizes various types of sound file headers.
It understands almost all headers that SOX can decode.

The return tuple contains the following items, in this order:
- file type (as SOX understands it)
- sampling rate (0 if unknown or hard to decode)
- number of channels (0 if unknown or hard to decode)
- number of frames in the file (-1 if unknown or hard to decode)
- number of bits/sample, or 'U' for U-LAW, or 'A' for A-LAW

If the file doesn't have a recognizable type, it returns None.
If the file can't be opened, OSError is raised.

To compute the total time, divide the number of frames by the
sampling rate (a frame contains a sample for each channel).

Function what() calls whathdr().  (It used to also use some
heuristics for raw data, but this doesn't work very well.)

Finally, the function test() is a simple main program that calls
what() for all files mentioned on the argument list.  For directory
arguments it calls what() for all files in that directory.  Default
argument is "." (testing all files in the current directory).  The
option -r tells it to recurse down directories found inside
explicitly given directories.
"""

# The file structure is top-down except that the test program and its
# subroutine come last.

__all__ = ['what', 'whathdr']

from collections import namedtuple

SndHeaders = namedtuple('SndHeaders',
                        'filetype framerate nchannels nframes sampwidth')

SndHeaders.filetype.__doc__ = ("""The value for type indicates the data type
and will be one of the strings 'aifc', 'aiff', 'au','hcom',
'sndr', 'sndt', 'voc', 'wav', '8svx', 'sb', 'ub', or 'ul'.""")
SndHeaders.framerate.__doc__ = ("""The sampling_rate will be either the actual
value or 0 if unknown or difficult to decode.""")
SndHeaders.nchannels.__doc__ = ("""The number of channels or 0 if it cannot be
determined or if the value is difficult to decode.""")
SndHeaders.nframes.__doc__ = ("""The value for frames will be either the number
of frames or -1.""")
SndHeaders.sampwidth.__doc__ = ("""Either the sample size in bits or
'A' for A-LAW or 'U' for u-LAW.""")

def what(filename):
    """Guess the type of a sound file."""
    res = whathdr(filename)
    return res


def whathdr(filename):
    """Recognize sound headers."""
    with open(filename, 'rb') as f:
        h = f.read(512)
        for tf in tests:
            res = tf(h, f)
            if res:
                return SndHeaders(*res)
        return None


#-----------------------------------#
# Subroutines per sound header type #
#-----------------------------------#

tests = []

def test_aifc(h, f):
    import aifc
    if not h.startswith(b'FORM'):
        return None
    if h[8:12] == b'AIFC':
        fmt = 'aifc'
    elif h[8:12] == b'AIFF':
        fmt = 'aiff'
    else:
        return None
    f.seek(0)
    try:
        a = aifc.open(f, 'r')
    except (EOFError, aifc.Error):
        return None
    return (fmt, a.getframerate(), a.getnchannels(),
            a.getnframes(), 8 * a.getsampwidth())

tests.append(test_aifc)


def test_au(h, f):
    if h.startswith(b'.snd'):
        func = get_long_be
    elif h[:4] in (b'\0ds.', b'dns.'):
        func = get_long_le
    else:
        return None
    filetype = 'au'
    hdr_size = func(h[4:8])
    data_size = func(h[8:12])
    encoding = func(h[12:16])
    rate = func(h[16:20])
    nchannels = func(h[20:24])
    sample_size = 1 # default
    if encoding == 1:
        sample_bits = 'U'
    elif encoding == 2:
        sample_bits = 8
    elif encoding == 3:
        sample_bits = 16
        sample_size = 2
    else:
        sample_bits = '?'
    frame_size = sample_size * nchannels
    if frame_size:
        nframe = data_size / frame_size
    else:
        nframe = -1
    return filetype, rate, nchannels, nframe, sample_bits

tests.append(test_au)


def test_hcom(h, f):
    if h[65:69] != b'FSSD' or h[128:132] != b'HCOM':
        return None
    divisor = get_long_be(h[144:148])
    if divisor:
        rate = 22050 / divisor
    else:
        rate = 0
    return 'hcom', rate, 1, -1, 8

tests.append(test_hcom)


def test_voc(h, f):
    if not h.startswith(b'Creative Voice File\032'):
        return None
    sbseek = get_short_le(h[20:22])
    rate = 0
    if 0 <= sbseek < 500 and h[sbseek] == 1:
        ratecode = 256 - h[sbseek+4]
        if ratecode:
            rate = int(1000000.0 / ratecode)
    return 'voc', rate, 1, -1, 8

tests.append(test_voc)


def test_wav(h, f):
    import wave
    # 'RIFF' <len> 'WAVE' 'fmt ' <len>
    if not h.startswith(b'RIFF') or h[8:12] != b'WAVE' or h[12:16] != b'fmt ':
        return None
    f.seek(0)
    try:
        w = wave.open(f, 'r')
    except (EOFError, wave.Error):
        return None
    return ('wav', w.getframerate(), w.getnchannels(),
                   w.getnframes(), 8*w.getsampwidth())

tests.append(test_wav)


def test_8svx(h, f):
    if not h.startswith(b'FORM') or h[8:12] != b'8SVX':
        return None
    # Should decode it to get #channels -- assume always 1
    return '8svx', 0, 1, 0, 8

tests.append(test_8svx)


def test_sndt(h, f):
    if h.startswith(b'SOUND'):
        nsamples = get_long_le(h[8:12])
        rate = get_short_le(h[20:22])
        return 'sndt', rate, 1, nsamples, 8

tests.append(test_sndt)


def test_sndr(h, f):
    if h.startswith(b'\0\0'):
        rate = get_short_le(h[2:4])
        if 4000 <= rate <= 25000:
            return 'sndr', rate, 1, -1, 8

tests.append(test_sndr)


#-------------------------------------------#
# Subroutines to extract numbers from bytes #
#-------------------------------------------#

def get_long_be(b):
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]

def get_long_le(b):
    return (b[3] << 24) | (b[2] << 16) | (b[1] << 8) | b[0]

def get_short_be(b):
    return (b[0] << 8) | b[1]

def get_short_le(b):
    return (b[1] << 8) | b[0]


#--------------------#
# Small test program #
#--------------------#

def test():
    import sys
    recursive = 0
    if sys.argv[1:] and sys.argv[1] == '-r':
        del sys.argv[1:2]
        recursive = 1
    try:
        if sys.argv[1:]:
            testall(sys.argv[1:], recursive, 1)
        else:
            testall(['.'], recursive, 1)
    except KeyboardInterrupt:
        sys.stderr.write('\n[Interrupted]\n')
        sys.exit(1)

def testall(list, recursive, toplevel):
    import sys
    import os
    for filename in list:
        if os.path.isdir(filename):
            print(filename + '/:', end=' ')
            if recursive or toplevel:
                print('recursing down:')
                import glob
                names = glob.glob(os.path.join(glob.escape(filename), '*'))
                testall(names, recursive, 0)
            else:
                print('*** directory (use -r) ***')
        else:
            print(filename + ':', end=' ')
            sys.stdout.flush()
            try:
                print(what(filename))
            except OSError:
                print('*** not found ***')

if __name__ == '__main__':
    test()
<|endoftext|>"""More comprehensive traceback formatting for Python scripts.

To enable this module, do:

    import cgitb; cgitb.enable()

at the top of your script.  The optional arguments to enable() are:

    display     - if true, tracebacks are displayed in the web browser
    logdir      - if set, tracebacks are written to files in this directory
    context     - number of lines of source code to show for each stack frame
    format      - 'text' or 'html' controls the output format

By default, tracebacks are displayed but not saved, the context is 5 lines
and the output format is 'html' (for backwards compatibility with the
original use of this module)

Alternatively, if you have caught an exception and want cgitb to display it
for you, call cgitb.handler().  The optional argument to handler() is a
3-item tuple (etype, evalue, etb) just like the value of sys.exc_info().
The default handler displays output as HTML.

"""
import inspect
import keyword
import linecache
import os
import pydoc
import sys
import tempfile
import time
import tokenize
import traceback

def reset():
    """Return a string that resets the CGI and browser to a known state."""
    return '''<!--: spam
Content-Type: text/html

<body bgcolor="#f0f0f8"><font color="#f0f0f8" size="-5"> -->
<body bgcolor="#f0f0f8"><font color="#f0f0f8" size="-5"> --> -->
</font> </font> </font> </script> </object> </blockquote> </pre>
</table> </table> </table> </table> </table> </font> </font> </font>'''

__UNDEF__ = []                          # a special sentinel object
def small(text):
    if text:
        return '<small>' + text + '</small>'
    else:
        return ''

def strong(text):
    if text:
        return '<strong>' + text + '</strong>'
    else:
        return ''

def grey(text):
    if text:
        return '<font color="#909090">' + text + '</font>'
    else:
        return ''

def lookup(name, frame, locals):
    """Find the value for a given name in the given environment."""
    if name in locals:
        return 'local', locals[name]
    if name in frame.f_globals:
        return 'global', frame.f_globals[name]
    if '__builtins__' in frame.f_globals:
        builtins = frame.f_globals['__builtins__']
        if type(builtins) is type({}):
            if name in builtins:
                return 'builtin', builtins[name]
        else:
            if hasattr(builtins, name):
                return 'builtin', getattr(builtins, name)
    return None, __UNDEF__

def scanvars(reader, frame, locals):
    """Scan one logical line of Python and look up values of variables used."""
    vars, lasttoken, parent, prefix, value = [], None, None, '', __UNDEF__
    for ttype, token, start, end, line in tokenize.generate_tokens(reader):
        if ttype == tokenize.NEWLINE: break
        if ttype == tokenize.NAME and token not in keyword.kwlist:
            if lasttoken == '.':
                if parent is not __UNDEF__:
                    value = getattr(parent, token, __UNDEF__)
                    vars.append((prefix + token, prefix, value))
            else:
                where, value = lookup(token, frame, locals)
                vars.append((token, where, value))
        elif token == '.':
            prefix += lasttoken + '.'
            parent = value
        else:
            parent, prefix = None, ''
        lasttoken = token
    return vars

def html(einfo, context=5):
    """Return a nice HTML document describing a given traceback."""
    etype, evalue, etb = einfo
    if isinstance(etype, type):
        etype = etype.__name__
    pyver = 'Python ' + sys.version.split()[0] + ': ' + sys.executable
    date = time.ctime(time.time())
    head = '<body bgcolor="#f0f0f8">' + pydoc.html.heading(
        '<big><big>%s</big></big>' %
        strong(pydoc.html.escape(str(etype))),
        '#ffffff', '#6622aa', pyver + '<br>' + date) + '''
<p>A problem occurred in a Python script.  Here is the sequence of
function calls leading up to the error, in the order they occurred.</p>'''

    indent = '<tt>' + small('&nbsp;' * 5) + '&nbsp;</tt>'
    frames = []
    records = inspect.getinnerframes(etb, context)
    for frame, file, lnum, func, lines, index in records:
        if file:
            file = os.path.abspath(file)
            link = '<a href="file://%s">%s</a>' % (file, pydoc.html.escape(file))
        else:
            file = link = '?'
        args, varargs, varkw, locals = inspect.getargvalues(frame)
        call = ''
        if func != '?':
            call = 'in ' + strong(pydoc.html.escape(func))
            if func != "<module>":
                call += inspect.formatargvalues(args, varargs, varkw, locals,
                    formatvalue=lambda value: '=' + pydoc.html.repr(value))

        highlight = {}
        def reader(lnum=[lnum]):
            highlight[lnum[0]] = 1
            try: return linecache.getline(file, lnum[0])
            finally: lnum[0] += 1
        vars = scanvars(reader, frame, locals)

        rows = ['<tr><td bgcolor="#d8bbff">%s%s %s</td></tr>' %
                ('<big>&nbsp;</big>', link, call)]
        if index is not None:
            i = lnum - index
            for line in lines:
                num = small('&nbsp;' * (5-len(str(i))) + str(i)) + '&nbsp;'
                if i in highlight:
                    line = '<tt>=&gt;%s%s</tt>' % (num, pydoc.html.preformat(line))
                    rows.append('<tr><td bgcolor="#ffccee">%s</td></tr>' % line)
                else:
                    line = '<tt>&nbsp;&nbsp;%s%s</tt>' % (num, pydoc.html.preformat(line))
                    rows.append('<tr><td>%s</td></tr>' % grey(line))
                i += 1

        done, dump = {}, []
        for name, where, value in vars:
            if name in done: continue
            done[name] = 1
            if value is not __UNDEF__:
                if where in ('global', 'builtin'):
                    name = ('<em>%s</em> ' % where) + strong(name)
                elif where == 'local':
                    name = strong(name)
                else:
                    name = where + strong(name.split('.')[-1])
                dump.append('%s&nbsp;= %s' % (name, pydoc.html.repr(value)))
            else:
                dump.append(name + ' <em>undefined</em>')

        rows.append('<tr><td>%s</td></tr>' % small(grey(', '.join(dump))))
        frames.append('''
<table width="100%%" cellspacing=0 cellpadding=0 border=0>
%s</table>''' % '\n'.join(rows))

    exception = ['<p>%s: %s' % (strong(pydoc.html.escape(str(etype))),
                                pydoc.html.escape(str(evalue)))]
    for name in dir(evalue):
        if name[:1] == '_': continue
        value = pydoc.html.repr(getattr(evalue, name))
        exception.append('\n<br>%s%s&nbsp;=\n%s' % (indent, name, value))

    return head + ''.join(frames) + ''.join(exception) + '''


<!-- The above is a description of an error in a Python program, formatted
     for a web browser because the 'cgitb' module was enabled.  In case you
     are not reading this in a web browser, here is the original traceback:

%s
-->
''' % pydoc.html.escape(
          ''.join(traceback.format_exception(etype, evalue, etb)))

def text(einfo, context=5):
    """Return a plain text document describing a given traceback."""
    etype, evalue, etb = einfo
    if isinstance(etype, type):
        etype = etype.__name__
    pyver = 'Python ' + sys.version.split()[0] + ': ' + sys.executable
    date = time.ctime(time.time())
    head = "%s\n%s\n%s\n" % (str(etype), pyver, date) + '''
A problem occurred in a Python script.  Here is the sequence of
function calls leading up to the error, in the order they occurred.
'''

    frames = []
    records = inspect.getinnerframes(etb, context)
    for frame, file, lnum, func, lines, index in records:
        file = file and os.path.abspath(file) or '?'
        args, varargs, varkw, locals = inspect.getargvalues(frame)
        call = ''
        if func != '?':
            call = 'in ' + func
            if func != "<module>":
                call += inspect.formatargvalues(args, varargs, varkw, locals,
                    formatvalue=lambda value: '=' + pydoc.text.repr(value))

        highlight = {}
        def reader(lnum=[lnum]):
            highlight[lnum[0]] = 1
            try: return linecache.getline(file, lnum[0])
            finally: lnum[0] += 1
        vars = scanvars(reader, frame, locals)

        rows = [' %s %s' % (file, call)]
        if index is not None:
            i = lnum - index
            for line in lines:
                num = '%5d ' % i
                rows.append(num+line.rstrip())
                i += 1

        done, dump = {}, []
        for name, where, value in vars:
            if name in done: continue
            done[name] = 1
            if value is not __UNDEF__:
                if where == 'global': name = 'global ' + name
                elif where != 'local': name = where + name.split('.')[-1]
                dump.append('%s = %s' % (name, pydoc.text.repr(value)))
            else:
                dump.append(name + ' undefined')

        rows.append('\n'.join(dump))
        frames.append('\n%s\n' % '\n'.join(rows))

    exception = ['%s: %s' % (str(etype), str(evalue))]
    for name in dir(evalue):
        value = pydoc.text.repr(getattr(evalue, name))
        exception.append('\n%s%s = %s' % (" "*4, name, value))

    return head + ''.join(frames) + ''.join(exception) + '''

The above is a description of an error in a Python program.  Here is
the original traceback:

%s
''' % ''.join(traceback.format_exception(etype, evalue, etb))

class Hook:
    """A hook to replace sys.excepthook that shows tracebacks in HTML."""

    def __init__(self, display=1, logdir=None, context=5, file=None,
                 format="html"):
        self.display = display          # send tracebacks to browser if true
        self.logdir = logdir            # log tracebacks to files if not None
        self.context = context          # number of source code lines per frame
        self.file = file or sys.stdout  # place to send the output
        self.format = format

    def __call__(self, etype, evalue, etb):
        self.handle((etype, evalue, etb))

    def handle(self, info=None):
        info = info or sys.exc_info()
        if self.format == "html":
            self.file.write(reset())

        formatter = (self.format=="html") and html or text
        plain = False
        try:
            doc = formatter(info, self.context)
        except:                         # just in case something goes wrong
            doc = ''.join(traceback.format_exception(*info))
            plain = True

        if self.display:
            if plain:
                doc = pydoc.html.escape(doc)
                self.file.write('<pre>' + doc + '</pre>\n')
            else:
                self.file.write(doc + '\n')
        else:
            self.file.write('<p>A problem occurred in a Python script.\n')

        if self.logdir is not None:
            suffix = ['.txt', '.html'][self.format=="html"]
            (fd, path) = tempfile.mkstemp(suffix=suffix, dir=self.logdir)

            try:
                with os.fdopen(fd, 'w') as file:
                    file.write(doc)
                msg = '%s contains the description of this error.' % path
            except:
                msg = 'Tried to save traceback to %s, but failed.' % path

            if self.format == 'html':
                self.file.write('<p>%s</p>\n' % msg)
            else:
                self.file.write(msg + '\n')
        try:
            self.file.flush()
        except: pass

handler = Hook().handle
def enable(display=1, logdir=None, context=5, format="html"):
    """Install an exception handler that formats tracebacks as HTML.

    The optional argument 'display' can be set to 0 to suppress sending the
    traceback to the browser, and 'logdir' can be set to a directory to cause
    tracebacks to be written to files there."""
    sys.excepthook = Hook(display=display, logdir=logdir,
                          context=context, format=format)
<|endoftext|>#
# Secret Labs' Regular Expression Engine
#
# re-compatible interface for the sre matching engine
#
# Copyright (c) 1998-2001 by Secret Labs AB.  All rights reserved.
#
# This version of the SRE library can be redistributed under CNRI's
# Python 1.6 license.  For any other use, please contact Secret Labs
# AB (info@pythonware.com).
#
# Portions of this engine have been developed in cooperation with
# CNRI.  Hewlett-Packard provided funding for 1.6 integration and
# other compatibility work.
#

r"""Support for regular expressions (RE).

This module provides regular expression matching operations similar to
those found in Perl.  It supports both 8-bit and Unicode strings; both
the pattern and the strings being processed can contain null bytes and
characters outside the US ASCII range.

Regular expressions can contain both special and ordinary characters.
Most ordinary characters, like "A", "a", or "0", are the simplest
regular expressions; they simply match themselves.  You can
concatenate ordinary characters, so last matches the string 'last'.

The special characters are:
    "."      Matches any character except a newline.
    "^"      Matches the start of the string.
    "$"      Matches the end of the string or just before the newline at
             the end of the string.
    "*"      Matches 0 or more (greedy) repetitions of the preceding RE.
             Greedy means that it will match as many repetitions as possible.
    "+"      Matches 1 or more (greedy) repetitions of the preceding RE.
    "?"      Matches 0 or 1 (greedy) of the preceding RE.
    *?,+?,?? Non-greedy versions of the previous three special characters.
    {m,n}    Matches from m to n repetitions of the preceding RE.
    {m,n}?   Non-greedy version of the above.
    "\\"     Either escapes special characters or signals a special sequence.
    []       Indicates a set of characters.
             A "^" as the first character indicates a complementing set.
    "|"      A|B, creates an RE that will match either A or B.
    (...)    Matches the RE inside the parentheses.
             The contents can be retrieved or matched later in the string.
    (?aiLmsux) The letters set the corresponding flags defined below.
    (?:...)  Non-grouping version of regular parentheses.
    (?P<name>...) The substring matched by the group is accessible by name.
    (?P=name)     Matches the text matched earlier by the group named name.
    (?#...)  A comment; ignored.
    (?=...)  Matches if ... matches next, but doesn't consume the string.
    (?!...)  Matches if ... doesn't match next.
    (?<=...) Matches if preceded by ... (must be fixed length).
    (?<!...) Matches if not preceded by ... (must be fixed length).
    (?(id/name)yes|no) Matches yes pattern if the group with id/name matched,
                       the (optional) no pattern otherwise.

The special sequences consist of "\\" and a character from the list
below.  If the ordinary character is not on the list, then the
resulting RE will match the second character.
    \number  Matches the contents of the group of the same number.
    \A       Matches only at the start of the string.
    \Z       Matches only at the end of the string.
    \b       Matches the empty string, but only at the start or end of a word.
    \B       Matches the empty string, but not at the start or end of a word.
    \d       Matches any decimal digit; equivalent to the set [0-9] in
             bytes patterns or string patterns with the ASCII flag.
             In string patterns without the ASCII flag, it will match the whole
             range of Unicode digits.
    \D       Matches any non-digit character; equivalent to [^\d].
    \s       Matches any whitespace character; equivalent to [ \t\n\r\f\v] in
             bytes patterns or string patterns with the ASCII flag.
             In string patterns without the ASCII flag, it will match the whole
             range of Unicode whitespace characters.
    \S       Matches any non-whitespace character; equivalent to [^\s].
    \w       Matches any alphanumeric character; equivalent to [a-zA-Z0-9_]
             in bytes patterns or string patterns with the ASCII flag.
             In string patterns without the ASCII flag, it will match the
             range of Unicode alphanumeric characters (letters plus digits
             plus underscore).
             With LOCALE, it will match the set [0-9_] plus characters defined
             as letters for the current locale.
    \W       Matches the complement of \w.
    \\       Matches a literal backslash.

This module exports the following functions:
    match     Match a regular expression pattern to the beginning of a string.
    fullmatch Match a regular expression pattern to all of a string.
    search    Search a string for the presence of a pattern.
    sub       Substitute occurrences of a pattern found in a string.
    subn      Same as sub, but also return the number of substitutions made.
    split     Split a string by the occurrences of a pattern.
    findall   Find all occurrences of a pattern in a string.
    finditer  Return an iterator yielding a Match object for each match.
    compile   Compile a pattern into a Pattern object.
    purge     Clear the regular expression cache.
    escape    Backslash all non-alphanumerics in a string.

Each function other than purge and escape can take an optional 'flags' argument
consisting of one or more of the following module constants, joined by "|".
A, L, and U are mutually exclusive.
    A  ASCII       For string patterns, make \w, \W, \b, \B, \d, \D
                   match the corresponding ASCII character categories
                   (rather than the whole Unicode categories, which is the
                   default).
                   For bytes patterns, this flag is the only available
                   behaviour and needn't be specified.
    I  IGNORECASE  Perform case-insensitive matching.
    L  LOCALE      Make \w, \W, \b, \B, dependent on the current locale.
    M  MULTILINE   "^" matches the beginning of lines (after a newline)
                   as well as the string.
                   "$" matches the end of lines (before a newline) as well
                   as the end of the string.
    S  DOTALL      "." matches any character at all, including the newline.
    X  VERBOSE     Ignore whitespace and comments for nicer looking RE's.
    U  UNICODE     For compatibility only. Ignored for string patterns (it
                   is the default), and forbidden for bytes patterns.

This module also defines an exception 'error'.

"""

import enum
import sre_compile
import sre_parse
import functools
try:
    import _locale
except ImportError:
    _locale = None


# public symbols
__all__ = [
    "match", "fullmatch", "search", "sub", "subn", "split",
    "findall", "finditer", "compile", "purge", "template", "escape",
    "error", "Pattern", "Match", "A", "I", "L", "M", "S", "X", "U",
    "ASCII", "IGNORECASE", "LOCALE", "MULTILINE", "DOTALL", "VERBOSE",
    "UNICODE",
]

__version__ = "2.2.1"

class RegexFlag(enum.IntFlag):
    ASCII = A = sre_compile.SRE_FLAG_ASCII # assume ascii "locale"
    IGNORECASE = I = sre_compile.SRE_FLAG_IGNORECASE # ignore case
    LOCALE = L = sre_compile.SRE_FLAG_LOCALE # assume current 8-bit locale
    UNICODE = U = sre_compile.SRE_FLAG_UNICODE # assume unicode "locale"
    MULTILINE = M = sre_compile.SRE_FLAG_MULTILINE # make anchors look for newline
    DOTALL = S = sre_compile.SRE_FLAG_DOTALL # make dot match newline
    VERBOSE = X = sre_compile.SRE_FLAG_VERBOSE # ignore whitespace and comments
    # sre extensions (experimental, don't rely on these)
    TEMPLATE = T = sre_compile.SRE_FLAG_TEMPLATE # disable backtracking
    DEBUG = sre_compile.SRE_FLAG_DEBUG # dump pattern after compilation

    def __repr__(self):
        if self._name_ is not None:
            return f're.{self._name_}'
        value = self._value_
        members = []
        negative = value < 0
        if negative:
            value = ~value
        for m in self.__class__:
            if value & m._value_:
                value &= ~m._value_
                members.append(f're.{m._name_}')
        if value:
            members.append(hex(value))
        res = '|'.join(members)
        if negative:
            if len(members) > 1:
                res = f'~({res})'
            else:
                res = f'~{res}'
        return res
    __str__ = object.__str__
globals().update(RegexFlag.__members__)

# sre exception
error = sre_compile.error

# --------------------------------------------------------------------
# public interface

def match(pattern, string, flags=0):
    """Try to apply the pattern at the start of the string, returning
    a Match object, or None if no match was found."""
    return _compile(pattern, flags).match(string)

def fullmatch(pattern, string, flags=0):
    """Try to apply the pattern to all of the string, returning
    a Match object, or None if no match was found."""
    return _compile(pattern, flags).fullmatch(string)

def search(pattern, string, flags=0):
    """Scan through string looking for a match to the pattern, returning
    a Match object, or None if no match was found."""
    return _compile(pattern, flags).search(string)

def sub(pattern, repl, string, count=0, flags=0):
    """Return the string obtained by replacing the leftmost
    non-overlapping occurrences of the pattern in string by the
    replacement repl.  repl can be either a string or a callable;
    if a string, backslash escapes in it are processed.  If it is
    a callable, it's passed the Match object and must return
    a replacement string to be used."""
    return _compile(pattern, flags).sub(repl, string, count)

def subn(pattern, repl, string, count=0, flags=0):
    """Return a 2-tuple containing (new_string, number).
    new_string is the string obtained by replacing the leftmost
    non-overlapping occurrences of the pattern in the source
    string by the replacement repl.  number is the number of
    substitutions that were made. repl can be either a string or a
    callable; if a string, backslash escapes in it are processed.
    If it is a callable, it's passed the Match object and must
    return a replacement string to be used."""
    return _compile(pattern, flags).subn(repl, string, count)

def split(pattern, string, maxsplit=0, flags=0):
    """Split the source string by the occurrences of the pattern,
    returning a list containing the resulting substrings.  If
    capturing parentheses are used in pattern, then the text of all
    groups in the pattern are also returned as part of the resulting
    list.  If maxsplit is nonzero, at most maxsplit splits occur,
    and the remainder of the string is returned as the final element
    of the list."""
    return _compile(pattern, flags).split(string, maxsplit)

def findall(pattern, string, flags=0):
    """Return a list of all non-overlapping matches in the string.

    If one or more capturing groups are present in the pattern, return
    a list of groups; this will be a list of tuples if the pattern
    has more than one group.

    Empty matches are included in the result."""
    return _compile(pattern, flags).findall(string)

def finditer(pattern, string, flags=0):
    """Return an iterator over all non-overlapping matches in the
    string.  For each match, the iterator returns a Match object.

    Empty matches are included in the result."""
    return _compile(pattern, flags).finditer(string)

def compile(pattern, flags=0):
    "Compile a regular expression pattern, returning a Pattern object."
    return _compile(pattern, flags)

def purge():
    "Clear the regular expression caches"
    _cache.clear()
    _compile_repl.cache_clear()

def template(pattern, flags=0):
    "Compile a template pattern, returning a Pattern object"
    return _compile(pattern, flags|T)

# SPECIAL_CHARS
# closing ')', '}' and ']'
# '-' (a range in character set)
# '&', '~', (extended character set operations)
# '#' (comment) and WHITESPACE (ignored) in verbose mode
_special_chars_map = {i: '\\' + chr(i) for i in b'()[]{}?*+-|^$\\.&~# \t\n\r\v\f'}

def escape(pattern):
    """
    Escape special characters in a string.
    """
    if isinstance(pattern, str):
        return pattern.translate(_special_chars_map)
    else:
        pattern = str(pattern, 'latin1')
        return pattern.translate(_special_chars_map).encode('latin1')

Pattern = type(sre_compile.compile('', 0))
Match = type(sre_compile.compile('', 0).match(''))

# --------------------------------------------------------------------
# internals

_cache = {}  # ordered!

_MAXCACHE = 512
def _compile(pattern, flags):
    # internal: compile pattern
    if isinstance(flags, RegexFlag):
        flags = flags.value
    try:
        return _cache[type(pattern), pattern, flags]
    except KeyError:
        pass
    if isinstance(pattern, Pattern):
        if flags:
            raise ValueError(
                "cannot process flags argument with a compiled pattern")
        return pattern
    if not sre_compile.isstring(pattern):
        raise TypeError("first argument must be string or compiled pattern")
    p = sre_compile.compile(pattern, flags)
    if not (flags & DEBUG):
        if len(_cache) >= _MAXCACHE:
            # Drop the oldest item
            try:
                del _cache[next(iter(_cache))]
            except (StopIteration, RuntimeError, KeyError):
                pass
        _cache[type(pattern), pattern, flags] = p
    return p

@functools.lru_cache(_MAXCACHE)
def _compile_repl(repl, pattern):
    # internal: compile replacement pattern
    return sre_parse.parse_template(repl, pattern)

def _expand(pattern, match, template):
    # internal: Match.expand implementation hook
    template = sre_parse.parse_template(template, pattern)
    return sre_parse.expand_template(template, match)

def _subx(pattern, template):
    # internal: Pattern.sub/subn implementation helper
    template = _compile_repl(template, pattern)
    if not template[0] and len(template[1]) == 1:
        # literal replacement
        return template[1][0]
    def filter(match, template=template):
        return sre_parse.expand_template(template, match)
    return filter

# register myself for pickling

import copyreg

def _pickle(p):
    return _compile, (p.pattern, p.flags)

copyreg.pickle(Pattern, _pickle, _compile)

# --------------------------------------------------------------------
# experimental stuff (see python-dev discussions for details)

class Scanner:
    def __init__(self, lexicon, flags=0):
        from sre_constants import BRANCH, SUBPATTERN
        if isinstance(flags, RegexFlag):
            flags = flags.value
        self.lexicon = lexicon
        # combine phrases into a compound pattern
        p = []
        s = sre_parse.State()
        s.flags = flags
        for phrase, action in lexicon:
            gid = s.opengroup()
            p.append(sre_parse.SubPattern(s, [
                (SUBPATTERN, (gid, 0, 0, sre_parse.parse(phrase, flags))),
                ]))
            s.closegroup(gid, p[-1])
        p = sre_parse.SubPattern(s, [(BRANCH, (None, p))])
        self.scanner = sre_compile.compile(p)
    def scan(self, string):
        result = []
        append = result.append
        match = self.scanner.scanner(string).match
        i = 0
        while True:
            m = match()
            if not m:
                break
            j = m.end()
            if i == j:
                break
            action = self.lexicon[m.lastindex-1][1]
            if callable(action):
                self.match = m
                action = action(self, m.group())
            if action is not None:
                append(action)
            i = j
        return result, string[i:]
<|endoftext|>"""Heap queue algorithm (a.k.a. priority queue).

Heaps are arrays for which a[k] <= a[2*k+1] and a[k] <= a[2*k+2] for
all k, counting elements from 0.  For the sake of comparison,
non-existing elements are considered to be infinite.  The interesting
property of a heap is that a[0] is always its smallest element.

Usage:

heap = []            # creates an empty heap
heappush(heap, item) # pushes a new item on the heap
item = heappop(heap) # pops the smallest item from the heap
item = heap[0]       # smallest item on the heap without popping it
heapify(x)           # transforms list into a heap, in-place, in linear time
item = heapreplace(heap, item) # pops and returns smallest item, and adds
                               # new item; the heap size is unchanged

Our API differs from textbook heap algorithms as follows:

- We use 0-based indexing.  This makes the relationship between the
  index for a node and the indexes for its children slightly less
  obvious, but is more suitable since Python uses 0-based indexing.

- Our heappop() method returns the smallest item, not the largest.

These two make it possible to view the heap as a regular Python list
without surprises: heap[0] is the smallest item, and heap.sort()
maintains the heap invariant!
"""

# Original code by Kevin O'Connor, augmented by Tim Peters and Raymond Hettinger

__about__ = """Heap queues

[explanation by François Pinard]

Heaps are arrays for which a[k] <= a[2*k+1] and a[k] <= a[2*k+2] for
all k, counting elements from 0.  For the sake of comparison,
non-existing elements are considered to be infinite.  The interesting
property of a heap is that a[0] is always its smallest element.

The strange invariant above is meant to be an efficient memory
representation for a tournament.  The numbers below are `k', not a[k]:

                                   0

                  1                                 2

          3               4                5               6

      7       8       9       10      11      12      13      14

    15 16   17 18   19 20   21 22   23 24   25 26   27 28   29 30


In the tree above, each cell `k' is topping `2*k+1' and `2*k+2'.  In
a usual binary tournament we see in sports, each cell is the winner
over the two cells it tops, and we can trace the winner down the tree
to see all opponents s/he had.  However, in many computer applications
of such tournaments, we do not need to trace the history of a winner.
To be more memory efficient, when a winner is promoted, we try to
replace it by something else at a lower level, and the rule becomes
that a cell and the two cells it tops contain three different items,
but the top cell "wins" over the two topped cells.

If this heap invariant is protected at all time, index 0 is clearly
the overall winner.  The simplest algorithmic way to remove it and
find the "next" winner is to move some loser (let's say cell 30 in the
diagram above) into the 0 position, and then percolate this new 0 down
the tree, exchanging values, until the invariant is re-established.
This is clearly logarithmic on the total number of items in the tree.
By iterating over all items, you get an O(n ln n) sort.

A nice feature of this sort is that you can efficiently insert new
items while the sort is going on, provided that the inserted items are
not "better" than the last 0'th element you extracted.  This is
especially useful in simulation contexts, where the tree holds all
incoming events, and the "win" condition means the smallest scheduled
time.  When an event schedule other events for execution, they are
scheduled into the future, so they can easily go into the heap.  So, a
heap is a good structure for implementing schedulers (this is what I
used for my MIDI sequencer :-).

Various structures for implementing schedulers have been extensively
studied, and heaps are good for this, as they are reasonably speedy,
the speed is almost constant, and the worst case is not much different
than the average case.  However, there are other representations which
are more efficient overall, yet the worst cases might be terrible.

Heaps are also very useful in big disk sorts.  You most probably all
know that a big sort implies producing "runs" (which are pre-sorted
sequences, which size is usually related to the amount of CPU memory),
followed by a merging passes for these runs, which merging is often
very cleverly organised[1].  It is very important that the initial
sort produces the longest runs possible.  Tournaments are a good way
to that.  If, using all the memory available to hold a tournament, you
replace and percolate items that happen to fit the current run, you'll
produce runs which are twice the size of the memory for random input,
and much better for input fuzzily ordered.

Moreover, if you output the 0'th item on disk and get an input which
may not fit in the current tournament (because the value "wins" over
the last output value), it cannot fit in the heap, so the size of the
heap decreases.  The freed memory could be cleverly reused immediately
for progressively building a second heap, which grows at exactly the
same rate the first heap is melting.  When the first heap completely
vanishes, you switch heaps and start a new run.  Clever and quite
effective!

In a word, heaps are useful memory structures to know.  I use them in
a few applications, and I think it is good to keep a `heap' module
around. :-)

--------------------
[1] The disk balancing algorithms which are current, nowadays, are
more annoying than clever, and this is a consequence of the seeking
capabilities of the disks.  On devices which cannot seek, like big
tape drives, the story was quite different, and one had to be very
clever to ensure (far in advance) that each tape movement will be the
most effective possible (that is, will best participate at
"progressing" the merge).  Some tapes were even able to read
backwards, and this was also used to avoid the rewinding time.
Believe me, real good tape sorts were quite spectacular to watch!
From all times, sorting has always been a Great Art! :-)
"""

__all__ = ['heappush', 'heappop', 'heapify', 'heapreplace', 'merge',
           'nlargest', 'nsmallest', 'heappushpop']

def heappush(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)

def heappop(heap):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt

def heapreplace(heap, item):
    """Pop and return the current smallest value, and add the new item.

    This is more efficient than heappop() followed by heappush(), and can be
    more appropriate when using a fixed-size heap.  Note that the value
    returned may be larger than item!  That constrains reasonable uses of
    this routine unless written as part of a conditional replacement:

        if item > heap[0]:
            item = heapreplace(heap, item)
    """
    returnitem = heap[0]    # raises appropriate IndexError if heap is empty
    heap[0] = item
    _siftup(heap, 0)
    return returnitem

def heappushpop(heap, item):
    """Fast version of a heappush followed by a heappop."""
    if heap and heap[0] < item:
        item, heap[0] = heap[0], item
        _siftup(heap, 0)
    return item

def heapify(x):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    n = len(x)
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, so must have 2*i + 1 < n,
    # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
    # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
    # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    for i in reversed(range(n//2)):
        _siftup(x, i)

def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt

def _heapreplace_max(heap, item):
    """Maxheap version of a heappop followed by a heappush."""
    returnitem = heap[0]    # raises appropriate IndexError if heap is empty
    heap[0] = item
    _siftup_max(heap, 0)
    return returnitem

def _heapify_max(x):
    """Transform list into a maxheap, in-place, in O(len(x)) time."""
    n = len(x)
    for i in reversed(range(n//2)):
        _siftup_max(x, i)

# 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
# is the index of a leaf with a possibly out-of-order value.  Restore the
# heap invariant.
def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

# The child indices of heap index pos are already heaps, and we want to make
# a heap at index pos too.  We do this by bubbling the smaller child of
# pos up (and so on with that child's children, etc) until hitting a leaf,
# then using _siftdown to move the oddball originally at index pos into place.
#
# We *could* break out of the loop as soon as we find a pos where newitem <=
# both its children, but turns out that's not a good idea, and despite that
# many books write the algorithm that way.  During a heap pop, the last array
# element is sifted in, and that tends to be large, so that comparing it
# against values starting from the root usually doesn't pay (= usually doesn't
# get us out of the loop early).  See Knuth, Volume 3, where this is
# explained and quantified in an exercise.
#
# Cutting the # of comparisons is important, since these routines have no
# way to extract "the priority" from an array element, so that intelligence
# is likely to be hiding in custom comparison methods, or in array elements
# storing (priority, record) tuples.  Comparisons are thus potentially
# expensive.
#
# On random arrays of length 1000, making this change cut the number of
# comparisons made by heapify() a little, and those made by exhaustive
# heappop() a lot, in accord with theory.  Here are typical results from 3
# runs (3 just to demonstrate how small the variance is):
#
# Compares needed by heapify     Compares needed by 1000 heappops
# --------------------------     --------------------------------
# 1837 cut to 1663               14996 cut to 8680
# 1855 cut to 1659               14966 cut to 8678
# 1847 cut to 1660               15024 cut to 8703
#
# Building the heap by using heappush() 1000 times instead required
# 2198, 2148, and 2219 compares:  heapify() is more efficient, when
# you can use it.
#
# The total compares needed by list.sort() on the same lists were 8627,
# 8627, and 8632 (this should be compared to the sum of heapify() and
# heappop() compares):  list.sort() is (unsurprisingly!) more efficient
# for sorting.

def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)

def _siftdown_max(heap, startpos, pos):
    'Maxheap variant of _siftdown'
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if parent < newitem:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

def _siftup_max(heap, pos):
    'Maxheap variant of _siftup'
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the larger child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of larger child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[rightpos] < heap[childpos]:
            childpos = rightpos
        # Move the larger child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown_max(heap, startpos, pos)

def merge(*iterables, key=None, reverse=False):
    '''Merge multiple sorted inputs into a single sorted output.

    Similar to sorted(itertools.chain(*iterables)) but returns a generator,
    does not pull the data into memory all at once, and assumes that each of
    the input streams is already sorted (smallest to largest).

    >>> list(merge([1,3,5,7], [0,2,4,8], [5,10,15,20], [], [25]))
    [0, 1, 2, 3, 4, 5, 5, 7, 8, 10, 15, 20, 25]

    If *key* is not None, applies a key function to each element to determine
    its sort order.

    >>> list(merge(['dog', 'horse'], ['cat', 'fish', 'kangaroo'], key=len))
    ['dog', 'cat', 'fish', 'horse', 'kangaroo']

    '''

    h = []
    h_append = h.append

    if reverse:
        _heapify = _heapify_max
        _heappop = _heappop_max
        _heapreplace = _heapreplace_max
        direction = -1
    else:
        _heapify = heapify
        _heappop = heappop
        _heapreplace = heapreplace
        direction = 1

    if key is None:
        for order, it in enumerate(map(iter, iterables)):
            try:
                next = it.__next__
                h_append([next(), order * direction, next])
            except StopIteration:
                pass
        _heapify(h)
        while len(h) > 1:
            try:
                while True:
                    value, order, next = s = h[0]
                    yield value
                    s[0] = next()           # raises StopIteration when exhausted
                    _heapreplace(h, s)      # restore heap condition
            except StopIteration:
                _heappop(h)                 # remove empty iterator
        if h:
            # fast case when only a single iterator remains
            value, order, next = h[0]
            yield value
            yield from next.__self__
        return

    for order, it in enumerate(map(iter, iterables)):
        try:
            next = it.__next__
            value = next()
            h_append([key(value), order * direction, value, next])
        except StopIteration:
            pass
    _heapify(h)
    while len(h) > 1:
        try:
            while True:
                key_value, order, value, next = s = h[0]
                yield value
                value = next()
                s[0] = key(value)
                s[2] = value
                _heapreplace(h, s)
        except StopIteration:
            _heappop(h)
    if h:
        key_value, order, value, next = h[0]
        yield value
        yield from next.__self__


# Algorithm notes for nlargest() and nsmallest()
# ==============================================
#
# Make a single pass over the data while keeping the k most extreme values
# in a heap.  Memory consumption is limited to keeping k values in a list.
#
# Measured performance for random inputs:
#
#                                   number of comparisons
#    n inputs     k-extreme values  (average of 5 trials)   % more than min()
# -------------   ----------------  ---------------------   -----------------
#      1,000           100                  3,317               231.7%
#     10,000           100                 14,046                40.5%
#    100,000           100                105,749                 5.7%
#  1,000,000           100              1,007,751                 0.8%
# 10,000,000           100             10,009,401                 0.1%
#
# Theoretical number of comparisons for k smallest of n random inputs:
#
# Step   Comparisons                  Action
# ----   --------------------------   ---------------------------
#  1     1.66 * k                     heapify the first k-inputs
#  2     n - k                        compare remaining elements to top of heap
#  3     k * (1 + lg2(k)) * ln(n/k)   replace the topmost value on the heap
#  4     k * lg2(k) - (k/2)           final sort of the k most extreme values
#
# Combining and simplifying for a rough estimate gives:
#
#        comparisons = n + k * (log(k, 2) * log(n/k) + log(k, 2) + log(n/k))
#
# Computing the number of comparisons for step 3:
# -----------------------------------------------
# * For the i-th new value from the iterable, the probability of being in the
#   k most extreme values is k/i.  For example, the probability of the 101st
#   value seen being in the 100 most extreme values is 100/101.
# * If the value is a new extreme value, the cost of inserting it into the
#   heap is 1 + log(k, 2).
# * The probability times the cost gives:
#            (k/i) * (1 + log(k, 2))
# * Summing across the remaining n-k elements gives:
#            sum((k/i) * (1 + log(k, 2)) for i in range(k+1, n+1))
# * This reduces to:
#            (H(n) - H(k)) * k * (1 + log(k, 2))
# * Where H(n) is the n-th harmonic number estimated by:
#            gamma = 0.5772156649
#            H(n) = log(n, e) + gamma + 1 / (2 * n)
#   http://en.wikipedia.org/wiki/Harmonic_series_(mathematics)#Rate_of_divergence
# * Substituting the H(n) formula:
#            comparisons = k * (1 + log(k, 2)) * (log(n/k, e) + (1/n - 1/k) / 2)
#
# Worst-case for step 3:
# ----------------------
# In the worst case, the input data is reversed sorted so that every new element
# must be inserted in the heap:
#
#             comparisons = 1.66 * k + log(k, 2) * (n - k)
#
# Alternative Algorithms
# ----------------------
# Other algorithms were not used because they:
# 1) Took much more auxiliary memory,
# 2) Made multiple passes over the data.
# 3) Made more comparisons in common cases (small k, large n, semi-random input).
# See the more detailed comparison of approach at:
# http://code.activestate.com/recipes/577573-compare-algorithms-for-heapqsmallest

def nsmallest(n, iterable, key=None):
    """Find the n smallest elements in a dataset.

    Equivalent to:  sorted(iterable, key=key)[:n]
    """

    # Short-cut for n==1 is to use min()
    if n == 1:
        it = iter(iterable)
        sentinel = object()
        result = min(it, default=sentinel, key=key)
        return [] if result is sentinel else [result]

    # When n>=size, it's faster to use sorted()
    try:
        size = len(iterable)
    except (TypeError, AttributeError):
        pass
    else:
        if n >= size:
            return sorted(iterable, key=key)[:n]

    # When key is none, use simpler decoration
    if key is None:
        it = iter(iterable)
        # put the range(n) first so that zip() doesn't
        # consume one too many elements from the iterator
        result = [(elem, i) for i, elem in zip(range(n), it)]
        if not result:
            return result
        _heapify_max(result)
        top = result[0][0]
        order = n
        _heapreplace = _heapreplace_max
        for elem in it:
            if elem < top:
                _heapreplace(result, (elem, order))
                top, _order = result[0]
                order += 1
        result.sort()
        return [elem for (elem, order) in result]

    # General case, slowest method
    it = iter(iterable)
    result = [(key(elem), i, elem) for i, elem in zip(range(n), it)]
    if not result:
        return result
    _heapify_max(result)
    top = result[0][0]
    order = n
    _heapreplace = _heapreplace_max
    for elem in it:
        k = key(elem)
        if k < top:
            _heapreplace(result, (k, order, elem))
            top, _order, _elem = result[0]
            order += 1
    result.sort()
    return [elem for (k, order, elem) in result]

def nlargest(n, iterable, key=None):
    """Find the n largest elements in a dataset.

    Equivalent to:  sorted(iterable, key=key, reverse=True)[:n]
    """

    # Short-cut for n==1 is to use max()
    if n == 1:
        it = iter(iterable)
        sentinel = object()
        result = max(it, default=sentinel, key=key)
        return [] if result is sentinel else [result]

    # When n>=size, it's faster to use sorted()
    try:
        size = len(iterable)
    except (TypeError, AttributeError):
        pass
    else:
        if n >= size:
            return sorted(iterable, key=key, reverse=True)[:n]

    # When key is none, use simpler decoration
    if key is None:
        it = iter(iterable)
        result = [(elem, i) for i, elem in zip(range(0, -n, -1), it)]
        if not result:
            return result
        heapify(result)
        top = result[0][0]
        order = -n
        _heapreplace = heapreplace
        for elem in it:
            if top < elem:
                _heapreplace(result, (elem, order))
                top, _order = result[0]
                order -= 1
        result.sort(reverse=True)
        return [elem for (elem, order) in result]

    # General case, slowest method
    it = iter(iterable)
    result = [(key(elem), i, elem) for i, elem in zip(range(0, -n, -1), it)]
    if not result:
        return result
    heapify(result)
    top = result[0][0]
    order = -n
    _heapreplace = heapreplace
    for elem in it:
        k = key(elem)
        if top < k:
            _heapreplace(result, (k, order, elem))
            top, _order, _elem = result[0]
            order -= 1
    result.sort(reverse=True)
    return [elem for (k, order, elem) in result]

# If available, use C implementation
try:
    from _heapq import *
except ImportError:
    pass
try:
    from _heapq import _heapreplace_max
except ImportError:
    pass
try:
    from _heapq import _heapify_max
except ImportError:
    pass
try:
    from _heapq import _heappop_max
except ImportError:
    pass


if __name__ == "__main__":

    import doctest # pragma: no cover
    print(doctest.testmod()) # pragma: no cover
<|endoftext|># Copyright 2007 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Abstract Base Classes (ABCs) for collections, according to PEP 3119.

Unit tests are in test_collections.
"""

from abc import ABCMeta, abstractmethod
import sys

GenericAlias = type(list[int])
EllipsisType = type(...)
def _f(): pass
FunctionType = type(_f)
del _f

__all__ = ["Awaitable", "Coroutine",
           "AsyncIterable", "AsyncIterator", "AsyncGenerator",
           "Hashable", "Iterable", "Iterator", "Generator", "Reversible",
           "Sized", "Container", "Callable", "Collection",
           "Set", "MutableSet",
           "Mapping", "MutableMapping",
           "MappingView", "KeysView", "ItemsView", "ValuesView",
           "Sequence", "MutableSequence",
           "ByteString",
           ]

# This module has been renamed from collections.abc to _collections_abc to
# speed up interpreter startup. Some of the types such as MutableMapping are
# required early but collections module imports a lot of other modules.
# See issue #19218
__name__ = "collections.abc"

# Private list of types that we want to register with the various ABCs
# so that they will pass tests like:
#       it = iter(somebytearray)
#       assert isinstance(it, Iterable)
# Note:  in other implementations, these types might not be distinct
# and they may have their own implementation specific types that
# are not included on this list.
bytes_iterator = type(iter(b''))
bytearray_iterator = type(iter(bytearray()))
#callable_iterator = ???
dict_keyiterator = type(iter({}.keys()))
dict_valueiterator = type(iter({}.values()))
dict_itemiterator = type(iter({}.items()))
list_iterator = type(iter([]))
list_reverseiterator = type(iter(reversed([])))
range_iterator = type(iter(range(0)))
longrange_iterator = type(iter(range(1 << 1000)))
set_iterator = type(iter(set()))
str_iterator = type(iter(""))
tuple_iterator = type(iter(()))
zip_iterator = type(iter(zip()))
## views ##
dict_keys = type({}.keys())
dict_values = type({}.values())
dict_items = type({}.items())
## misc ##
mappingproxy = type(type.__dict__)
generator = type((lambda: (yield))())
## coroutine ##
async def _coro(): pass
_coro = _coro()
coroutine = type(_coro)
_coro.close()  # Prevent ResourceWarning
del _coro
## asynchronous generator ##
async def _ag(): yield
_ag = _ag()
async_generator = type(_ag)
del _ag


### ONE-TRICK PONIES ###

def _check_methods(C, *methods):
    mro = C.__mro__
    for method in methods:
        for B in mro:
            if method in B.__dict__:
                if B.__dict__[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True

class Hashable(metaclass=ABCMeta):

    __slots__ = ()

    @abstractmethod
    def __hash__(self):
        return 0

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Hashable:
            return _check_methods(C, "__hash__")
        return NotImplemented


class Awaitable(metaclass=ABCMeta):

    __slots__ = ()

    @abstractmethod
    def __await__(self):
        yield

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Awaitable:
            return _check_methods(C, "__await__")
        return NotImplemented

    __class_getitem__ = classmethod(GenericAlias)


class Coroutine(Awaitable):

    __slots__ = ()

    @abstractmethod
    def send(self, value):
        """Send a value into the coroutine.
        Return next yielded value or raise StopIteration.
        """
        raise StopIteration

    @abstractmethod
    def throw(self, typ, val=None, tb=None):
        """Raise an exception in the coroutine.
        Return next yielded value or raise StopIteration.
        """
        if val is None:
            if tb is None:
                raise typ
            val = typ()
        if tb is not None:
            val = val.with_traceback(tb)
        raise val

    def close(self):
        """Raise GeneratorExit inside coroutine.
        """
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            pass
        else:
            raise RuntimeError("coroutine ignored GeneratorExit")

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Coroutine:
            return _check_methods(C, '__await__', 'send', 'throw', 'close')
        return NotImplemented


Coroutine.register(coroutine)


class AsyncIterable(metaclass=ABCMeta):

    __slots__ = ()

    @abstractmethod
    def __aiter__(self):
        return AsyncIterator()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AsyncIterable:
            return _check_methods(C, "__aiter__")
        return NotImplemented

    __class_getitem__ = classmethod(GenericAlias)


class AsyncIterator(AsyncIterable):

    __slots__ = ()

    @abstractmethod
    async def __anext__(self):
        """Return the next item or raise StopAsyncIteration when exhausted."""
        raise StopAsyncIteration

    def __aiter__(self):
        return self

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AsyncIterator:
            return _check_methods(C, "__anext__", "__aiter__")
        return NotImplemented


class AsyncGenerator(AsyncIterator):

    __slots__ = ()

    async def __anext__(self):
        """Return the next item from the asynchronous generator.
        When exhausted, raise StopAsyncIteration.
        """
        return await self.asend(None)

    @abstractmethod
    async def asend(self, value):
        """Send a value into the asynchronous generator.
        Return next yielded value or raise StopAsyncIteration.
        """
        raise StopAsyncIteration

    @abstractmethod
    async def athrow(self, typ, val=None, tb=None):
        """Raise an exception in the asynchronous generator.
        Return next yielded value or raise StopAsyncIteration.
        """
        if val is None:
            if tb is None:
                raise typ
            val = typ()
        if tb is not None:
            val = val.with_traceback(tb)
        raise val

    async def aclose(self):
        """Raise GeneratorExit inside coroutine.
        """
        try:
            await self.athrow(GeneratorExit)
        except (GeneratorExit, StopAsyncIteration):
            pass
        else:
            raise RuntimeError("asynchronous generator ignored GeneratorExit")

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AsyncGenerator:
            return _check_methods(C, '__aiter__', '__anext__',
                                  'asend', 'athrow', 'aclose')
        return NotImplemented


AsyncGenerator.register(async_generator)


class Iterable(metaclass=ABCMeta):

    __slots__ = ()

    @abstractmethod
    def __iter__(self):
        while False:
            yield None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Iterable:
            return _check_methods(C, "__iter__")
        return NotImplemented

    __class_getitem__ = classmethod(GenericAlias)


class Iterator(Iterable):

    __slots__ = ()

    @abstractmethod
    def __next__(self):
        'Return the next item from the iterator. When exhausted, raise StopIteration'
        raise StopIteration

    def __iter__(self):
        return self

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Iterator:
            return _check_methods(C, '__iter__', '__next__')
        return NotImplemented


Iterator.register(bytes_iterator)
Iterator.register(bytearray_iterator)
#Iterator.register(callable_iterator)
Iterator.register(dict_keyiterator)
Iterator.register(dict_valueiterator)
Iterator.register(dict_itemiterator)
Iterator.register(list_iterator)
Iterator.register(list_reverseiterator)
Iterator.register(range_iterator)
Iterator.register(longrange_iterator)
Iterator.register(set_iterator)
Iterator.register(str_iterator)
Iterator.register(tuple_iterator)
Iterator.register(zip_iterator)


class Reversible(Iterable):

    __slots__ = ()

    @abstractmethod
    def __reversed__(self):
        while False:
            yield None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Reversible:
            return _check_methods(C, "__reversed__", "__iter__")
        return NotImplemented


class Generator(Iterator):

    __slots__ = ()

    def __next__(self):
        """Return the next item from the generator.
        When exhausted, raise StopIteration.
        """
        return self.send(None)

    @abstractmethod
    def send(self, value):
        """Send a value into the generator.
        Return next yielded value or raise StopIteration.
        """
        raise StopIteration

    @abstractmethod
    def throw(self, typ, val=None, tb=None):
        """Raise an exception in the generator.
        Return next yielded value or raise StopIteration.
        """
        if val is None:
            if tb is None:
                raise typ
            val = typ()
        if tb is not None:
            val = val.with_traceback(tb)
        raise val

    def close(self):
        """Raise GeneratorExit inside generator.
        """
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            pass
        else:
            raise RuntimeError("generator ignored GeneratorExit")

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Generator:
            return _check_methods(C, '__iter__', '__next__',
                                  'send', 'throw', 'close')
        return NotImplemented


Generator.register(generator)


class Sized(metaclass=ABCMeta):

    __slots__ = ()

    @abstractmethod
    def __len__(self):
        return 0

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Sized:
            return _check_methods(C, "__len__")
        return NotImplemented


class Container(metaclass=ABCMeta):

    __slots__ = ()

    @abstractmethod
    def __contains__(self, x):
        return False

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Container:
            return _check_methods(C, "__contains__")
        return NotImplemented

    __class_getitem__ = classmethod(GenericAlias)


class Collection(Sized, Iterable, Container):

    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Collection:
            return _check_methods(C,  "__len__", "__iter__", "__contains__")
        return NotImplemented


class _CallableGenericAlias(GenericAlias):
    """ Represent `Callable[argtypes, resulttype]`.

    This sets ``__args__`` to a tuple containing the flattened ``argtypes``
    followed by ``resulttype``.

    Example: ``Callable[[int, str], float]`` sets ``__args__`` to
    ``(int, str, float)``.
    """

    __slots__ = ()

    def __new__(cls, origin, args):
        if not (isinstance(args, tuple) and len(args) == 2):
            raise TypeError(
                "Callable must be used as Callable[[arg, ...], result].")
        t_args, t_result = args
        if isinstance(t_args, list):
            args = (*t_args, t_result)
        elif not _is_param_expr(t_args):
            raise TypeError(f"Expected a list of types, an ellipsis, "
                            f"ParamSpec, or Concatenate. Got {t_args}")
        return super().__new__(cls, origin, args)

    @property
    def __parameters__(self):
        params = []
        for arg in self.__args__:
            if isinstance(arg, type) and not isinstance(arg, GenericAlias):
                continue
            # Looks like a genericalias
            if hasattr(arg, "__parameters__") and isinstance(arg.__parameters__, tuple):
                params.extend(arg.__parameters__)
            else:
                if _is_typevarlike(arg):
                    params.append(arg)
        return tuple(dict.fromkeys(params))

    def __repr__(self):
        if len(self.__args__) == 2 and _is_param_expr(self.__args__[0]):
            return super().__repr__()
        return (f'collections.abc.Callable'
                f'[[{", ".join([_type_repr(a) for a in self.__args__[:-1]])}], '
                f'{_type_repr(self.__args__[-1])}]')

    def __reduce__(self):
        args = self.__args__
        if not (len(args) == 2 and _is_param_expr(args[0])):
            args = list(args[:-1]), args[-1]
        return _CallableGenericAlias, (Callable, args)

    def __getitem__(self, item):
        # Called during TypeVar substitution, returns the custom subclass
        # rather than the default types.GenericAlias object.  Most of the
        # code is copied from typing's _GenericAlias and the builtin
        # types.GenericAlias.

        # A special case in PEP 612 where if X = Callable[P, int],
        # then X[int, str] == X[[int, str]].
        param_len = len(self.__parameters__)
        if param_len == 0:
            raise TypeError(f'{self} is not a generic class')
        if not isinstance(item, tuple):
            item = (item,)
        if (param_len == 1 and _is_param_expr(self.__parameters__[0])
                and item and not _is_param_expr(item[0])):
            item = (list(item),)
        item_len = len(item)
        if item_len != param_len:
            raise TypeError(f'Too {"many" if item_len > param_len else "few"}'
                            f' arguments for {self};'
                            f' actual {item_len}, expected {param_len}')
        subst = dict(zip(self.__parameters__, item))
        new_args = []
        for arg in self.__args__:
            if isinstance(arg, type) and not isinstance(arg, GenericAlias):
                new_args.append(arg)
                continue
            if _is_typevarlike(arg):
                if _is_param_expr(arg):
                    arg = subst[arg]
                    if not _is_param_expr(arg):
                        raise TypeError(f"Expected a list of types, an ellipsis, "
                                        f"ParamSpec, or Concatenate. Got {arg}")
                else:
                    arg = subst[arg]
            # Looks like a GenericAlias
            elif hasattr(arg, '__parameters__') and isinstance(arg.__parameters__, tuple):
                subparams = arg.__parameters__
                if subparams:
                    subargs = tuple(subst[x] for x in subparams)
                    arg = arg[subargs]
            if isinstance(arg, tuple):
                new_args.extend(arg)
            else:
                new_args.append(arg)

        # args[0] occurs due to things like Z[[int, str, bool]] from PEP 612
        if not isinstance(new_args[0], list):
            t_result = new_args[-1]
            t_args = new_args[:-1]
            new_args = (t_args, t_result)
        return _CallableGenericAlias(Callable, tuple(new_args))


def _is_typevarlike(arg):
    obj = type(arg)
    # looks like a TypeVar/ParamSpec
    return (obj.__module__ == 'typing'
            and obj.__name__ in {'ParamSpec', 'TypeVar'})

def _is_param_expr(obj):
    """Checks if obj matches either a list of types, ``...``, ``ParamSpec`` or
    ``_ConcatenateGenericAlias`` from typing.py
    """
    if obj is Ellipsis:
        return True
    if isinstance(obj, list):
        return True
    obj = type(obj)
    names = ('ParamSpec', '_ConcatenateGenericAlias')
    return obj.__module__ == 'typing' and any(obj.__name__ == name for name in names)

def _type_repr(obj):
    """Return the repr() of an object, special-casing types (internal helper).

    Copied from :mod:`typing` since collections.abc
    shouldn't depend on that module.
    """
    if isinstance(obj, GenericAlias):
        return repr(obj)
    if isinstance(obj, type):
        if obj.__module__ == 'builtins':
            return obj.__qualname__
        return f'{obj.__module__}.{obj.__qualname__}'
    if obj is Ellipsis:
        return '...'
    if isinstance(obj, FunctionType):
        return obj.__name__
    return repr(obj)


class Callable(metaclass=ABCMeta):

    __slots__ = ()

    @abstractmethod
    def __call__(self, *args, **kwds):
        return False

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Callable:
            return _check_methods(C, "__call__")
        return NotImplemented

    __class_getitem__ = classmethod(_CallableGenericAlias)


### SETS ###


class Set(Collection):
    """A set is a finite, iterable container.

    This class provides concrete generic implementations of all
    methods except for __contains__, __iter__ and __len__.

    To override the comparisons (presumably for speed, as the
    semantics are fixed), redefine __le__ and __ge__,
    then the other operations will automatically follow suit.
    """

    __slots__ = ()

    def __le__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        if len(self) > len(other):
            return False
        for elem in self:
            if elem not in other:
                return False
        return True

    def __lt__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        return len(self) < len(other) and self.__le__(other)

    def __gt__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        return len(self) > len(other) and self.__ge__(other)

    def __ge__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        if len(self) < len(other):
            return False
        for elem in other:
            if elem not in self:
                return False
        return True

    def __eq__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        return len(self) == len(other) and self.__le__(other)

    @classmethod
    def _from_iterable(cls, it):
        '''Construct an instance of the class from any iterable input.

        Must override this method if the class constructor signature
        does not accept an iterable for an input.
        '''
        return cls(it)

    def __and__(self, other):
        if not isinstance(other, Iterable):
            return NotImplemented
        return self._from_iterable(value for value in other if value in self)

    __rand__ = __and__

    def isdisjoint(self, other):
        'Return True if two sets have a null intersection.'
        for value in other:
            if value in self:
                return False
        return True

    def __or__(self, other):
        if not isinstance(other, Iterable):
            return NotImplemented
        chain = (e for s in (self, other) for e in s)
        return self._from_iterable(chain)

    __ror__ = __or__

    def __sub__(self, other):
        if not isinstance(other, Set):
            if not isinstance(other, Iterable):
                return NotImplemented
            other = self._from_iterable(other)
        return self._from_iterable(value for value in self
                                   if value not in other)

    def __rsub__(self, other):
        if not isinstance(other, Set):
            if not isinstance(other, Iterable):
                return NotImplemented
            other = self._from_iterable(other)
        return self._from_iterable(value for value in other
                                   if value not in self)

    def __xor__(self, other):
        if not isinstance(other, Set):
            if not isinstance(other, Iterable):
                return NotImplemented
            other = self._from_iterable(other)
        return (self - other) | (other - self)

    __rxor__ = __xor__

    def _hash(self):
        """Compute the hash value of a set.

        Note that we don't define __hash__: not all sets are hashable.
        But if you define a hashable set type, its __hash__ should
        call this function.

        This must be compatible __eq__.

        All sets ought to compare equal if they contain the same
        elements, regardless of how they are implemented, and
        regardless of the order of the elements; so there's not much
        freedom for __eq__ or __hash__.  We match the algorithm used
        by the built-in frozenset type.
        """
        MAX = sys.maxsize
        MASK = 2 * MAX + 1
        n = len(self)
        h = 1927868237 * (n + 1)
        h &= MASK
        for x in self:
            hx = hash(x)
            h ^= (hx ^ (hx << 16) ^ 89869747)  * 3644798167
            h &= MASK
        h ^= (h >> 11) ^ (h >> 25)
        h = h * 69069 + 907133923
        h &= MASK
        if h > MAX:
            h -= MASK + 1
        if h == -1:
            h = 590923713
        return h


Set.register(frozenset)


class MutableSet(Set):
    """A mutable set is a finite, iterable container.

    This class provides concrete generic implementations of all
    methods except for __contains__, __iter__, __len__,
    add(), and discard().

    To override the comparisons (presumably for speed, as the
    semantics are fixed), all you have to do is redefine __le__ and
    then the other operations will automatically follow suit.
    """

    __slots__ = ()

    @abstractmethod
    def add(self, value):
        """Add an element."""
        raise NotImplementedError

    @abstractmethod
    def discard(self, value):
        """Remove an element.  Do not raise an exception if absent."""
        raise NotImplementedError

    def remove(self, value):
        """Remove an element. If not a member, raise a KeyError."""
        if value not in self:
            raise KeyError(value)
        self.discard(value)

    def pop(self):
        """Return the popped value.  Raise KeyError if empty."""
        it = iter(self)
        try:
            value = next(it)
        except StopIteration:
            raise KeyError from None
        self.discard(value)
        return value

    def clear(self):
        """This is slow (creates N new iterators!) but effective."""
        try:
            while True:
                self.pop()
        except KeyError:
            pass

    def __ior__(self, it):
        for value in it:
            self.add(value)
        return self

    def __iand__(self, it):
        for value in (self - it):
            self.discard(value)
        return self

    def __ixor__(self, it):
        if it is self:
            self.clear()
        else:
            if not isinstance(it, Set):
                it = self._from_iterable(it)
            for value in it:
                if value in self:
                    self.discard(value)
                else:
                    self.add(value)
        return self

    def __isub__(self, it):
        if it is self:
            self.clear()
        else:
            for value in it:
                self.discard(value)
        return self


MutableSet.register(set)


### MAPPINGS ###

class Mapping(Collection):
    """A Mapping is a generic container for associating key/value
    pairs.

    This class provides concrete generic implementations of all
    methods except for __getitem__, __iter__, and __len__.
    """

    __slots__ = ()

    # Tell ABCMeta.__new__ that this class should have TPFLAGS_MAPPING set.
    __abc_tpflags__ = 1 << 6 # Py_TPFLAGS_MAPPING

    @abstractmethod
    def __getitem__(self, key):
        raise KeyError

    def get(self, key, default=None):
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def keys(self):
        "D.keys() -> a set-like object providing a view on D's keys"
        return KeysView(self)

    def items(self):
        "D.items() -> a set-like object providing a view on D's items"
        return ItemsView(self)

    def values(self):
        "D.values() -> an object providing a view on D's values"
        return ValuesView(self)

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    __reversed__ = None

Mapping.register(mappingproxy)


class MappingView(Sized):

    __slots__ = '_mapping',

    def __init__(self, mapping):
        self._mapping = mapping

    def __len__(self):
        return len(self._mapping)

    def __repr__(self):
        return '{0.__class__.__name__}({0._mapping!r})'.format(self)

    __class_getitem__ = classmethod(GenericAlias)


class KeysView(MappingView, Set):

    __slots__ = ()

    @classmethod
    def _from_iterable(cls, it):
        return set(it)

    def __contains__(self, key):
        return key in self._mapping

    def __iter__(self):
        yield from self._mapping


KeysView.register(dict_keys)


class ItemsView(MappingView, Set):

    __slots__ = ()

    @classmethod
    def _from_iterable(cls, it):
        return set(it)

    def __contains__(self, item):
        key, value = item
        try:
            v = self._mapping[key]
        except KeyError:
            return False
        else:
            return v is value or v == value

    def __iter__(self):
        for key in self._mapping:
            yield (key, self._mapping[key])


ItemsView.register(dict_items)


class ValuesView(MappingView, Collection):

    __slots__ = ()

    def __contains__(self, value):
        for key in self._mapping:
            v = self._mapping[key]
            if v is value or v == value:
                return True
        return False

    def __iter__(self):
        for key in self._mapping:
            yield self._mapping[key]


ValuesView.register(dict_values)


class MutableMapping(Mapping):
    """A MutableMapping is a generic container for associating
    key/value pairs.

    This class provides concrete generic implementations of all
    methods except for __getitem__, __setitem__, __delitem__,
    __iter__, and __len__.
    """

    __slots__ = ()

    @abstractmethod
    def __setitem__(self, key, value):
        raise KeyError

    @abstractmethod
    def __delitem__(self, key):
        raise KeyError

    __marker = object()

    def pop(self, key, default=__marker):
        '''D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
          If key is not found, d is returned if given, otherwise KeyError is raised.
        '''
        try:
            value = self[key]
        except KeyError:
            if default is self.__marker:
                raise
            return default
        else:
            del self[key]
            return value

    def popitem(self):
        '''D.popitem() -> (k, v), remove and return some (key, value) pair
           as a 2-tuple; but raise KeyError if D is empty.
        '''
        try:
            key = next(iter(self))
        except StopIteration:
            raise KeyError from None
        value = self[key]
        del self[key]
        return key, value

    def clear(self):
        'D.clear() -> None.  Remove all items from D.'
        try:
            while True:
                self.popitem()
        except KeyError:
            pass

    def update(self, other=(), /, **kwds):
        ''' D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.
            If E present and has a .keys() method, does:     for k in E: D[k] = E[k]
            If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
            In either case, this is followed by: for k, v in F.items(): D[k] = v
        '''
        if isinstance(other, Mapping):
            for key in other:
                self[key] = other[key]
        elif hasattr(other, "keys"):
            for key in other.keys():
                self[key] = other[key]
        else:
            for key, value in other:
                self[key] = value
        for key, value in kwds.items():
            self[key] = value

    def setdefault(self, key, default=None):
        'D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D'
        try:
            return self[key]
        except KeyError:
            self[key] = default
        return default


MutableMapping.register(dict)


### SEQUENCES ###

class Sequence(Reversible, Collection):
    """All the operations on a read-only sequence.

    Concrete subclasses must override __new__ or __init__,
    __getitem__, and __len__.
    """

    __slots__ = ()

    # Tell ABCMeta.__new__ that this class should have TPFLAGS_SEQUENCE set.
    __abc_tpflags__ = 1 << 5 # Py_TPFLAGS_SEQUENCE

    @abstractmethod
    def __getitem__(self, index):
        raise IndexError

    def __iter__(self):
        i = 0
        try:
            while True:
                v = self[i]
                yield v
                i += 1
        except IndexError:
            return

    def __contains__(self, value):
        for v in self:
            if v is value or v == value:
                return True
        return False

    def __reversed__(self):
        for i in reversed(range(len(self))):
            yield self[i]

    def index(self, value, start=0, stop=None):
        '''S.index(value, [start, [stop]]) -> integer -- return first index of value.
           Raises ValueError if the value is not present.

           Supporting start and stop arguments is optional, but
           recommended.
        '''
        if start is not None and start < 0:
            start = max(len(self) + start, 0)
        if stop is not None and stop < 0:
            stop += len(self)

        i = start
        while stop is None or i < stop:
            try:
                v = self[i]
                if v is value or v == value:
                    return i
            except IndexError:
                break
            i += 1
        raise ValueError

    def count(self, value):
        'S.count(value) -> integer -- return number of occurrences of value'
        return sum(1 for v in self if v is value or v == value)

Sequence.register(tuple)
Sequence.register(str)
Sequence.register(range)
Sequence.register(memoryview)


class ByteString(Sequence):
    """This unifies bytes and bytearray.

    XXX Should add all their methods.
    """

    __slots__ = ()

ByteString.register(bytes)
ByteString.register(bytearray)


class MutableSequence(Sequence):
    """All the operations on a read-write sequence.

    Concrete subclasses must provide __new__ or __init__,
    __getitem__, __setitem__, __delitem__, __len__, and insert().
    """

    __slots__ = ()

    @abstractmethod
    def __setitem__(self, index, value):
        raise IndexError

    @abstractmethod
    def __delitem__(self, index):
        raise IndexError

    @abstractmethod
    def insert(self, index, value):
        'S.insert(index, value) -- insert value before index'
        raise IndexError

    def append(self, value):
        'S.append(value) -- append value to the end of the sequence'
        self.insert(len(self), value)

    def clear(self):
        'S.clear() -> None -- remove all items from S'
        try:
            while True:
                self.pop()
        except IndexError:
            pass

    def reverse(self):
        'S.reverse() -- reverse *IN PLACE*'
        n = len(self)
        for i in range(n//2):
            self[i], self[n-i-1] = self[n-i-1], self[i]

    def extend(self, values):
        'S.extend(iterable) -- extend sequence by appending elements from the iterable'
        if values is self:
            values = list(values)
        for v in values:
            self.append(v)

    def pop(self, index=-1):
        '''S.pop([index]) -> item -- remove and return item at index (default last).
           Raise IndexError if list is empty or index is out of range.
        '''
        v = self[index]
        del self[index]
        return v

    def remove(self, value):
        '''S.remove(value) -- remove first occurrence of value.
           Raise ValueError if the value is not present.
        '''
        del self[self.index(value)]

    def __iadd__(self, values):
        self.extend(values)
        return self


MutableSequence.register(list)
MutableSequence.register(bytearray)  # Multiply inheriting, see ByteString
<|endoftext|>"""Parse a Python module and describe its classes and functions.

Parse enough of a Python file to recognize imports and class and
function definitions, and to find out the superclasses of a class.

The interface consists of a single function:
    readmodule_ex(module, path=None)
where module is the name of a Python module, and path is an optional
list of directories where the module is to be searched.  If present,
path is prepended to the system search path sys.path.  The return value
is a dictionary.  The keys of the dictionary are the names of the
classes and functions defined in the module (including classes that are
defined via the from XXX import YYY construct).  The values are
instances of classes Class and Function.  One special key/value pair is
present for packages: the key '__path__' has a list as its value which
contains the package search path.

Classes and Functions have a common superclass: _Object.  Every instance
has the following attributes:
    module  -- name of the module;
    name    -- name of the object;
    file    -- file in which the object is defined;
    lineno  -- line in the file where the object's definition starts;
    end_lineno -- line in the file where the object's definition ends;
    parent  -- parent of this object, if any;
    children -- nested objects contained in this object.
The 'children' attribute is a dictionary mapping names to objects.

Instances of Function describe functions with the attributes from _Object,
plus the following:
    is_async -- if a function is defined with an 'async' prefix

Instances of Class describe classes with the attributes from _Object,
plus the following:
    super   -- list of super classes (Class instances if possible);
    methods -- mapping of method names to beginning line numbers.
If the name of a super class is not recognized, the corresponding
entry in the list of super classes is not a class instance but a
string giving the name of the super class.  Since import statements
are recognized and imported modules are scanned as well, this
shouldn't happen often.
"""

import ast
import sys
import importlib.util

__all__ = ["readmodule", "readmodule_ex", "Class", "Function"]

_modules = {}  # Initialize cache of modules we've seen.


class _Object:
    "Information about Python class or function."
    def __init__(self, module, name, file, lineno, end_lineno, parent):
        self.module = module
        self.name = name
        self.file = file
        self.lineno = lineno
        self.end_lineno = end_lineno
        self.parent = parent
        self.children = {}
        if parent is not None:
            parent.children[name] = self


# Odd Function and Class signatures are for back-compatibility.
class Function(_Object):
    "Information about a Python function, including methods."
    def __init__(self, module, name, file, lineno,
                 parent=None, is_async=False, *, end_lineno=None):
        super().__init__(module, name, file, lineno, end_lineno, parent)
        self.is_async = is_async
        if isinstance(parent, Class):
            parent.methods[name] = lineno


class Class(_Object):
    "Information about a Python class."
    def __init__(self, module, name, super_, file, lineno,
                 parent=None, *, end_lineno=None):
        super().__init__(module, name, file, lineno, end_lineno, parent)
        self.super = super_ or []
        self.methods = {}


# These 2 functions are used in these tests
# Lib/test/test_pyclbr, Lib/idlelib/idle_test/test_browser.py
def _nest_function(ob, func_name, lineno, end_lineno, is_async=False):
    "Return a Function after nesting within ob."
    return Function(ob.module, func_name, ob.file, lineno,
                    parent=ob, is_async=is_async, end_lineno=end_lineno)

def _nest_class(ob, class_name, lineno, end_lineno, super=None):
    "Return a Class after nesting within ob."
    return Class(ob.module, class_name, super, ob.file, lineno,
                 parent=ob, end_lineno=end_lineno)


def readmodule(module, path=None):
    """Return Class objects for the top-level classes in module.

    This is the original interface, before Functions were added.
    """

    res = {}
    for key, value in _readmodule(module, path or []).items():
        if isinstance(value, Class):
            res[key] = value
    return res

def readmodule_ex(module, path=None):
    """Return a dictionary with all functions and classes in module.

    Search for module in PATH + sys.path.
    If possible, include imported superclasses.
    Do this by reading source, without importing (and executing) it.
    """
    return _readmodule(module, path or [])


def _readmodule(module, path, inpackage=None):
    """Do the hard work for readmodule[_ex].

    If inpackage is given, it must be the dotted name of the package in
    which we are searching for a submodule, and then PATH must be the
    package search path; otherwise, we are searching for a top-level
    module, and path is combined with sys.path.
    """
    # Compute the full module name (prepending inpackage if set).
    if inpackage is not None:
        fullmodule = "%s.%s" % (inpackage, module)
    else:
        fullmodule = module

    # Check in the cache.
    if fullmodule in _modules:
        return _modules[fullmodule]

    # Initialize the dict for this module's contents.
    tree = {}

    # Check if it is a built-in module; we don't do much for these.
    if module in sys.builtin_module_names and inpackage is None:
        _modules[module] = tree
        return tree

    # Check for a dotted module name.
    i = module.rfind('.')
    if i >= 0:
        package = module[:i]
        submodule = module[i+1:]
        parent = _readmodule(package, path, inpackage)
        if inpackage is not None:
            package = "%s.%s" % (inpackage, package)
        if not '__path__' in parent:
            raise ImportError('No package named {}'.format(package))
        return _readmodule(submodule, parent['__path__'], package)

    # Search the path for the module.
    f = None
    if inpackage is not None:
        search_path = path
    else:
        search_path = path + sys.path
    spec = importlib.util._find_spec_from_path(fullmodule, search_path)
    if spec is None:
        raise ModuleNotFoundError(f"no module named {fullmodule!r}", name=fullmodule)
    _modules[fullmodule] = tree
    # Is module a package?
    if spec.submodule_search_locations is not None:
        tree['__path__'] = spec.submodule_search_locations
    try:
        source = spec.loader.get_source(fullmodule)
    except (AttributeError, ImportError):
        # If module is not Python source, we cannot do anything.
        return tree
    else:
        if source is None:
            return tree

    fname = spec.loader.get_filename(fullmodule)
    return _create_tree(fullmodule, path, fname, source, tree, inpackage)


class _ModuleBrowser(ast.NodeVisitor):
    def __init__(self, module, path, file, tree, inpackage):
        self.path = path
        self.tree = tree
        self.file = file
        self.module = module
        self.inpackage = inpackage
        self.stack = []

    def visit_ClassDef(self, node):
        bases = []
        for base in node.bases:
            name = ast.unparse(base)
            if name in self.tree:
                # We know this super class.
                bases.append(self.tree[name])
            elif len(names := name.split(".")) > 1:
                # Super class form is module.class:
                # look in module for class.
                *_, module, class_ = names
                if module in _modules:
                    bases.append(_modules[module].get(class_, name))
            else:
                bases.append(name)

        parent = self.stack[-1] if self.stack else None
        class_ = Class(self.module, node.name, bases, self.file, node.lineno,
                       parent=parent, end_lineno=node.end_lineno)
        if parent is None:
            self.tree[node.name] = class_
        self.stack.append(class_)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node, *, is_async=False):
        parent = self.stack[-1] if self.stack else None
        function = Function(self.module, node.name, self.file, node.lineno,
                            parent, is_async, end_lineno=node.end_lineno)
        if parent is None:
            self.tree[node.name] = function
        self.stack.append(function)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node, is_async=True)

    def visit_Import(self, node):
        if node.col_offset != 0:
            return

        for module in node.names:
            try:
                try:
                    _readmodule(module.name, self.path, self.inpackage)
                except ImportError:
                    _readmodule(module.name, [])
            except (ImportError, SyntaxError):
                # If we can't find or parse the imported module,
                # too bad -- don't die here.
                continue

    def visit_ImportFrom(self, node):
        if node.col_offset != 0:
            return
        try:
            module = "." * node.level
            if node.module:
                module += node.module
            module = _readmodule(module, self.path, self.inpackage)
        except (ImportError, SyntaxError):
            return

        for name in node.names:
            if name.name in module:
                self.tree[name.asname or name.name] = module[name.name]
            elif name.name == "*":
                for import_name, import_value in module.items():
                    if import_name.startswith("_"):
                        continue
                    self.tree[import_name] = import_value


def _create_tree(fullmodule, path, fname, source, tree, inpackage):
    mbrowser = _ModuleBrowser(fullmodule, path, fname, tree, inpackage)
    mbrowser.visit(ast.parse(source))
    return mbrowser.tree


def _main():
    "Print module output (default this file) for quick visual check."
    import os
    try:
        mod = sys.argv[1]
    except:
        mod = __file__
    if os.path.exists(mod):
        path = [os.path.dirname(mod)]
        mod = os.path.basename(mod)
        if mod.lower().endswith(".py"):
            mod = mod[:-3]
    else:
        path = []
    tree = readmodule_ex(mod, path)
    lineno_key = lambda a: getattr(a, 'lineno', 0)
    objs = sorted(tree.values(), key=lineno_key, reverse=True)
    indent_level = 2
    while objs:
        obj = objs.pop()
        if isinstance(obj, list):
            # Value is a __path__ key.
            continue
        if not hasattr(obj, 'indent'):
            obj.indent = 0

        if isinstance(obj, _Object):
            new_objs = sorted(obj.children.values(),
                              key=lineno_key, reverse=True)
            for ob in new_objs:
                ob.indent = obj.indent + indent_level
            objs.extend(new_objs)
        if isinstance(obj, Class):
            print("{}class {} {} {}"
                  .format(' ' * obj.indent, obj.name, obj.super, obj.lineno))
        elif isinstance(obj, Function):
            print("{}def {} {}".format(' ' * obj.indent, obj.name, obj.lineno))

if __name__ == "__main__":
    _main()
<|endoftext|>"""
The objects used by the site module to add custom builtins.
"""

# Those objects are almost immortal and they keep a reference to their module
# globals.  Defining them in the site module would keep too many references
# alive.
# Note this means this module should also avoid keep things alive in its
# globals.

import sys

class Quitter(object):
    def __init__(self, name, eof):
        self.name = name
        self.eof = eof
    def __repr__(self):
        return 'Use %s() or %s to exit' % (self.name, self.eof)
    def __call__(self, code=None):
        # Shells like IDLE catch the SystemExit, but listen when their
        # stdin wrapper is closed.
        try:
            sys.stdin.close()
        except:
            pass
        raise SystemExit(code)


class _Printer(object):
    """interactive prompt objects for printing the license text, a list of
    contributors and the copyright notice."""

    MAXLINES = 23

    def __init__(self, name, data, files=(), dirs=()):
        import os
        self.__name = name
        self.__data = data
        self.__lines = None
        self.__filenames = [os.path.join(dir, filename)
                            for dir in dirs
                            for filename in files]

    def __setup(self):
        if self.__lines:
            return
        data = None
        for filename in self.__filenames:
            try:
                with open(filename, encoding='utf-8') as fp:
                    data = fp.read()
                break
            except OSError:
                pass
        if not data:
            data = self.__data
        self.__lines = data.split('\n')
        self.__linecnt = len(self.__lines)

    def __repr__(self):
        self.__setup()
        if len(self.__lines) <= self.MAXLINES:
            return "\n".join(self.__lines)
        else:
            return "Type %s() to see the full %s text" % ((self.__name,)*2)

    def __call__(self):
        self.__setup()
        prompt = 'Hit Return for more, or q (and Return) to quit: '
        lineno = 0
        while 1:
            try:
                for i in range(lineno, lineno + self.MAXLINES):
                    print(self.__lines[i])
            except IndexError:
                break
            else:
                lineno += self.MAXLINES
                key = None
                while key is None:
                    key = input(prompt)
                    if key not in ('', 'q'):
                        key = None
                if key == 'q':
                    break


class _Helper(object):
    """Define the builtin 'help'.

    This is a wrapper around pydoc.help that provides a helpful message
    when 'help' is typed at the Python interactive prompt.

    Calling help() at the Python prompt starts an interactive help session.
    Calling help(thing) prints help for the python object 'thing'.
    """

    def __repr__(self):
        return "Type help() for interactive help, " \
               "or help(object) for help about object."
    def __call__(self, *args, **kwds):
        import pydoc
        return pydoc.help(*args, **kwds)
<|endoftext|>"""Strptime-related classes and functions.

CLASSES:
    LocaleTime -- Discovers and stores locale-specific time information
    TimeRE -- Creates regexes for pattern matching a string of text containing
                time information

FUNCTIONS:
    _getlang -- Figure out what language is being used for the locale
    strptime -- Calculates the time struct represented by the passed-in string

"""
import time
import locale
import calendar
from re import compile as re_compile
from re import IGNORECASE
from re import escape as re_escape
from datetime import (date as datetime_date,
                      timedelta as datetime_timedelta,
                      timezone as datetime_timezone)
from _thread import allocate_lock as _thread_allocate_lock

__all__ = []

def _getlang():
    # Figure out what the current language is set to.
    return locale.getlocale(locale.LC_TIME)

class LocaleTime(object):
    """Stores and handles locale-specific information related to time.

    ATTRIBUTES:
        f_weekday -- full weekday names (7-item list)
        a_weekday -- abbreviated weekday names (7-item list)
        f_month -- full month names (13-item list; dummy value in [0], which
                    is added by code)
        a_month -- abbreviated month names (13-item list, dummy value in
                    [0], which is added by code)
        am_pm -- AM/PM representation (2-item list)
        LC_date_time -- format string for date/time representation (string)
        LC_date -- format string for date representation (string)
        LC_time -- format string for time representation (string)
        timezone -- daylight- and non-daylight-savings timezone representation
                    (2-item list of sets)
        lang -- Language used by instance (2-item tuple)
    """

    def __init__(self):
        """Set all attributes.

        Order of methods called matters for dependency reasons.

        The locale language is set at the offset and then checked again before
        exiting.  This is to make sure that the attributes were not set with a
        mix of information from more than one locale.  This would most likely
        happen when using threads where one thread calls a locale-dependent
        function while another thread changes the locale while the function in
        the other thread is still running.  Proper coding would call for
        locks to prevent changing the locale while locale-dependent code is
        running.  The check here is done in case someone does not think about
        doing this.

        Only other possible issue is if someone changed the timezone and did
        not call tz.tzset .  That is an issue for the programmer, though,
        since changing the timezone is worthless without that call.

        """
        self.lang = _getlang()
        self.__calc_weekday()
        self.__calc_month()
        self.__calc_am_pm()
        self.__calc_timezone()
        self.__calc_date_time()
        if _getlang() != self.lang:
            raise ValueError("locale changed during initialization")
        if time.tzname != self.tzname or time.daylight != self.daylight:
            raise ValueError("timezone changed during initialization")

    def __calc_weekday(self):
        # Set self.a_weekday and self.f_weekday using the calendar
        # module.
        a_weekday = [calendar.day_abbr[i].lower() for i in range(7)]
        f_weekday = [calendar.day_name[i].lower() for i in range(7)]
        self.a_weekday = a_weekday
        self.f_weekday = f_weekday

    def __calc_month(self):
        # Set self.f_month and self.a_month using the calendar module.
        a_month = [calendar.month_abbr[i].lower() for i in range(13)]
        f_month = [calendar.month_name[i].lower() for i in range(13)]
        self.a_month = a_month
        self.f_month = f_month

    def __calc_am_pm(self):
        # Set self.am_pm by using time.strftime().

        # The magic date (1999,3,17,hour,44,55,2,76,0) is not really that
        # magical; just happened to have used it everywhere else where a
        # static date was needed.
        am_pm = []
        for hour in (1, 22):
            time_tuple = time.struct_time((1999,3,17,hour,44,55,2,76,0))
            am_pm.append(time.strftime("%p", time_tuple).lower())
        self.am_pm = am_pm

    def __calc_date_time(self):
        # Set self.date_time, self.date, & self.time by using
        # time.strftime().

        # Use (1999,3,17,22,44,55,2,76,0) for magic date because the amount of
        # overloaded numbers is minimized.  The order in which searches for
        # values within the format string is very important; it eliminates
        # possible ambiguity for what something represents.
        time_tuple = time.struct_time((1999,3,17,22,44,55,2,76,0))
        date_time = [None, None, None]
        date_time[0] = time.strftime("%c", time_tuple).lower()
        date_time[1] = time.strftime("%x", time_tuple).lower()
        date_time[2] = time.strftime("%X", time_tuple).lower()
        replacement_pairs = [('%', '%%'), (self.f_weekday[2], '%A'),
                    (self.f_month[3], '%B'), (self.a_weekday[2], '%a'),
                    (self.a_month[3], '%b'), (self.am_pm[1], '%p'),
                    ('1999', '%Y'), ('99', '%y'), ('22', '%H'),
                    ('44', '%M'), ('55', '%S'), ('76', '%j'),
                    ('17', '%d'), ('03', '%m'), ('3', '%m'),
                    # '3' needed for when no leading zero.
                    ('2', '%w'), ('10', '%I')]
        replacement_pairs.extend([(tz, "%Z") for tz_values in self.timezone
                                                for tz in tz_values])
        for offset,directive in ((0,'%c'), (1,'%x'), (2,'%X')):
            current_format = date_time[offset]
            for old, new in replacement_pairs:
                # Must deal with possible lack of locale info
                # manifesting itself as the empty string (e.g., Swedish's
                # lack of AM/PM info) or a platform returning a tuple of empty
                # strings (e.g., MacOS 9 having timezone as ('','')).
                if old:
                    current_format = current_format.replace(old, new)
            # If %W is used, then Sunday, 2005-01-03 will fall on week 0 since
            # 2005-01-03 occurs before the first Monday of the year.  Otherwise
            # %U is used.
            time_tuple = time.struct_time((1999,1,3,1,1,1,6,3,0))
            if '00' in time.strftime(directive, time_tuple):
                U_W = '%W'
            else:
                U_W = '%U'
            date_time[offset] = current_format.replace('11', U_W)
        self.LC_date_time = date_time[0]
        self.LC_date = date_time[1]
        self.LC_time = date_time[2]

    def __calc_timezone(self):
        # Set self.timezone by using time.tzname.
        # Do not worry about possibility of time.tzname[0] == time.tzname[1]
        # and time.daylight; handle that in strptime.
        try:
            time.tzset()
        except AttributeError:
            pass
        self.tzname = time.tzname
        self.daylight = time.daylight
        no_saving = frozenset({"utc", "gmt", self.tzname[0].lower()})
        if self.daylight:
            has_saving = frozenset({self.tzname[1].lower()})
        else:
            has_saving = frozenset()
        self.timezone = (no_saving, has_saving)


class TimeRE(dict):
    """Handle conversion from format directives to regexes."""

    def __init__(self, locale_time=None):
        """Create keys/values.

        Order of execution is important for dependency reasons.

        """
        if locale_time:
            self.locale_time = locale_time
        else:
            self.locale_time = LocaleTime()
        base = super()
        base.__init__({
            # The " [1-9]" part of the regex is to make %c from ANSI C work
            'd': r"(?P<d>3[0-1]|[1-2]\d|0[1-9]|[1-9]| [1-9])",
            'f': r"(?P<f>[0-9]{1,6})",
            'H': r"(?P<H>2[0-3]|[0-1]\d|\d)",
            'I': r"(?P<I>1[0-2]|0[1-9]|[1-9])",
            'G': r"(?P<G>\d\d\d\d)",
            'j': r"(?P<j>36[0-6]|3[0-5]\d|[1-2]\d\d|0[1-9]\d|00[1-9]|[1-9]\d|0[1-9]|[1-9])",
            'm': r"(?P<m>1[0-2]|0[1-9]|[1-9])",
            'M': r"(?P<M>[0-5]\d|\d)",
            'S': r"(?P<S>6[0-1]|[0-5]\d|\d)",
            'U': r"(?P<U>5[0-3]|[0-4]\d|\d)",
            'w': r"(?P<w>[0-6])",
            'u': r"(?P<u>[1-7])",
            'V': r"(?P<V>5[0-3]|0[1-9]|[1-4]\d|\d)",
            # W is set below by using 'U'
            'y': r"(?P<y>\d\d)",
            #XXX: Does 'Y' need to worry about having less or more than
            #     4 digits?
            'Y': r"(?P<Y>\d\d\d\d)",
            'z': r"(?P<z>[+-]\d\d:?[0-5]\d(:?[0-5]\d(\.\d{1,6})?)?|(?-i:Z))",
            'A': self.__seqToRE(self.locale_time.f_weekday, 'A'),
            'a': self.__seqToRE(self.locale_time.a_weekday, 'a'),
            'B': self.__seqToRE(self.locale_time.f_month[1:], 'B'),
            'b': self.__seqToRE(self.locale_time.a_month[1:], 'b'),
            'p': self.__seqToRE(self.locale_time.am_pm, 'p'),
            'Z': self.__seqToRE((tz for tz_names in self.locale_time.timezone
                                        for tz in tz_names),
                                'Z'),
            '%': '%'})
        base.__setitem__('W', base.__getitem__('U').replace('U', 'W'))
        base.__setitem__('c', self.pattern(self.locale_time.LC_date_time))
        base.__setitem__('x', self.pattern(self.locale_time.LC_date))
        base.__setitem__('X', self.pattern(self.locale_time.LC_time))

    def __seqToRE(self, to_convert, directive):
        """Convert a list to a regex string for matching a directive.

        Want possible matching values to be from longest to shortest.  This
        prevents the possibility of a match occurring for a value that also
        a substring of a larger value that should have matched (e.g., 'abc'
        matching when 'abcdef' should have been the match).

        """
        to_convert = sorted(to_convert, key=len, reverse=True)
        for value in to_convert:
            if value != '':
                break
        else:
            return ''
        regex = '|'.join(re_escape(stuff) for stuff in to_convert)
        regex = '(?P<%s>%s' % (directive, regex)
        return '%s)' % regex

    def pattern(self, format):
        """Return regex pattern for the format string.

        Need to make sure that any characters that might be interpreted as
        regex syntax are escaped.

        """
        processed_format = ''
        # The sub() call escapes all characters that might be misconstrued
        # as regex syntax.  Cannot use re.escape since we have to deal with
        # format directives (%m, etc.).
        regex_chars = re_compile(r"([\\.^$*+?\(\){}\[\]|])")
        format = regex_chars.sub(r"\\\1", format)
        whitespace_replacement = re_compile(r'\s+')
        format = whitespace_replacement.sub(r'\\s+', format)
        while '%' in format:
            directive_index = format.index('%')+1
            processed_format = "%s%s%s" % (processed_format,
                                           format[:directive_index-1],
                                           self[format[directive_index]])
            format = format[directive_index+1:]
        return "%s%s" % (processed_format, format)

    def compile(self, format):
        """Return a compiled re object for the format string."""
        return re_compile(self.pattern(format), IGNORECASE)

_cache_lock = _thread_allocate_lock()
# DO NOT modify _TimeRE_cache or _regex_cache without acquiring the cache lock
# first!
_TimeRE_cache = TimeRE()
_CACHE_MAX_SIZE = 5 # Max number of regexes stored in _regex_cache
_regex_cache = {}

def _calc_julian_from_U_or_W(year, week_of_year, day_of_week, week_starts_Mon):
    """Calculate the Julian day based on the year, week of the year, and day of
    the week, with week_start_day representing whether the week of the year
    assumes the week starts on Sunday or Monday (6 or 0)."""
    first_weekday = datetime_date(year, 1, 1).weekday()
    # If we are dealing with the %U directive (week starts on Sunday), it's
    # easier to just shift the view to Sunday being the first day of the
    # week.
    if not week_starts_Mon:
        first_weekday = (first_weekday + 1) % 7
        day_of_week = (day_of_week + 1) % 7
    # Need to watch out for a week 0 (when the first day of the year is not
    # the same as that specified by %U or %W).
    week_0_length = (7 - first_weekday) % 7
    if week_of_year == 0:
        return 1 + day_of_week - first_weekday
    else:
        days_to_week = week_0_length + (7 * (week_of_year - 1))
        return 1 + days_to_week + day_of_week


def _calc_julian_from_V(iso_year, iso_week, iso_weekday):
    """Calculate the Julian day based on the ISO 8601 year, week, and weekday.
    ISO weeks start on Mondays, with week 01 being the week containing 4 Jan.
    ISO week days range from 1 (Monday) to 7 (Sunday).
    """
    correction = datetime_date(iso_year, 1, 4).isoweekday() + 3
    ordinal = (iso_week * 7) + iso_weekday - correction
    # ordinal may be negative or 0 now, which means the date is in the previous
    # calendar year
    if ordinal < 1:
        ordinal += datetime_date(iso_year, 1, 1).toordinal()
        iso_year -= 1
        ordinal -= datetime_date(iso_year, 1, 1).toordinal()
    return iso_year, ordinal


def _strptime(data_string, format="%a %b %d %H:%M:%S %Y"):
    """Return a 2-tuple consisting of a time struct and an int containing
    the number of microseconds based on the input string and the
    format string."""

    for index, arg in enumerate([data_string, format]):
        if not isinstance(arg, str):
            msg = "strptime() argument {} must be str, not {}"
            raise TypeError(msg.format(index, type(arg)))

    global _TimeRE_cache, _regex_cache
    with _cache_lock:
        locale_time = _TimeRE_cache.locale_time
        if (_getlang() != locale_time.lang or
            time.tzname != locale_time.tzname or
            time.daylight != locale_time.daylight):
            _TimeRE_cache = TimeRE()
            _regex_cache.clear()
            locale_time = _TimeRE_cache.locale_time
        if len(_regex_cache) > _CACHE_MAX_SIZE:
            _regex_cache.clear()
        format_regex = _regex_cache.get(format)
        if not format_regex:
            try:
                format_regex = _TimeRE_cache.compile(format)
            # KeyError raised when a bad format is found; can be specified as
            # \\, in which case it was a stray % but with a space after it
            except KeyError as err:
                bad_directive = err.args[0]
                if bad_directive == "\\":
                    bad_directive = "%"
                del err
                raise ValueError("'%s' is a bad directive in format '%s'" %
                                    (bad_directive, format)) from None
            # IndexError only occurs when the format string is "%"
            except IndexError:
                raise ValueError("stray %% in format '%s'" % format) from None
            _regex_cache[format] = format_regex
    found = format_regex.match(data_string)
    if not found:
        raise ValueError("time data %r does not match format %r" %
                         (data_string, format))
    if len(data_string) != found.end():
        raise ValueError("unconverted data remains: %s" %
                          data_string[found.end():])

    iso_year = year = None
    month = day = 1
    hour = minute = second = fraction = 0
    tz = -1
    gmtoff = None
    gmtoff_fraction = 0
    # Default to -1 to signify that values not known; not critical to have,
    # though
    iso_week = week_of_year = None
    week_of_year_start = None
    # weekday and julian defaulted to None so as to signal need to calculate
    # values
    weekday = julian = None
    found_dict = found.groupdict()
    for group_key in found_dict.keys():
        # Directives not explicitly handled below:
        #   c, x, X
        #      handled by making out of other directives
        #   U, W
        #      worthless without day of the week
        if group_key == 'y':
            year = int(found_dict['y'])
            # Open Group specification for strptime() states that a %y
            #value in the range of [00, 68] is in the century 2000, while
            #[69,99] is in the century 1900
            if year <= 68:
                year += 2000
            else:
                year += 1900
        elif group_key == 'Y':
            year = int(found_dict['Y'])
        elif group_key == 'G':
            iso_year = int(found_dict['G'])
        elif group_key == 'm':
            month = int(found_dict['m'])
        elif group_key == 'B':
            month = locale_time.f_month.index(found_dict['B'].lower())
        elif group_key == 'b':
            month = locale_time.a_month.index(found_dict['b'].lower())
        elif group_key == 'd':
            day = int(found_dict['d'])
        elif group_key == 'H':
            hour = int(found_dict['H'])
        elif group_key == 'I':
            hour = int(found_dict['I'])
            ampm = found_dict.get('p', '').lower()
            # If there was no AM/PM indicator, we'll treat this like AM
            if ampm in ('', locale_time.am_pm[0]):
                # We're in AM so the hour is correct unless we're
                # looking at 12 midnight.
                # 12 midnight == 12 AM == hour 0
                if hour == 12:
                    hour = 0
            elif ampm == locale_time.am_pm[1]:
                # We're in PM so we need to add 12 to the hour unless
                # we're looking at 12 noon.
                # 12 noon == 12 PM == hour 12
                if hour != 12:
                    hour += 12
        elif group_key == 'M':
            minute = int(found_dict['M'])
        elif group_key == 'S':
            second = int(found_dict['S'])
        elif group_key == 'f':
            s = found_dict['f']
            # Pad to always return microseconds.
            s += "0" * (6 - len(s))
            fraction = int(s)
        elif group_key == 'A':
            weekday = locale_time.f_weekday.index(found_dict['A'].lower())
        elif group_key == 'a':
            weekday = locale_time.a_weekday.index(found_dict['a'].lower())
        elif group_key == 'w':
            weekday = int(found_dict['w'])
            if weekday == 0:
                weekday = 6
            else:
                weekday -= 1
        elif group_key == 'u':
            weekday = int(found_dict['u'])
            weekday -= 1
        elif group_key == 'j':
            julian = int(found_dict['j'])
        elif group_key in ('U', 'W'):
            week_of_year = int(found_dict[group_key])
            if group_key == 'U':
                # U starts week on Sunday.
                week_of_year_start = 6
            else:
                # W starts week on Monday.
                week_of_year_start = 0
        elif group_key == 'V':
            iso_week = int(found_dict['V'])
        elif group_key == 'z':
            z = found_dict['z']
            if z == 'Z':
                gmtoff = 0
            else:
                if z[3] == ':':
                    z = z[:3] + z[4:]
                    if len(z) > 5:
                        if z[5] != ':':
                            msg = f"Inconsistent use of : in {found_dict['z']}"
                            raise ValueError(msg)
                        z = z[:5] + z[6:]
                hours = int(z[1:3])
                minutes = int(z[3:5])
                seconds = int(z[5:7] or 0)
                gmtoff = (hours * 60 * 60) + (minutes * 60) + seconds
                gmtoff_remainder = z[8:]
                # Pad to always return microseconds.
                gmtoff_remainder_padding = "0" * (6 - len(gmtoff_remainder))
                gmtoff_fraction = int(gmtoff_remainder + gmtoff_remainder_padding)
                if z.startswith("-"):
                    gmtoff = -gmtoff
                    gmtoff_fraction = -gmtoff_fraction
        elif group_key == 'Z':
            # Since -1 is default value only need to worry about setting tz if
            # it can be something other than -1.
            found_zone = found_dict['Z'].lower()
            for value, tz_values in enumerate(locale_time.timezone):
                if found_zone in tz_values:
                    # Deal with bad locale setup where timezone names are the
                    # same and yet time.daylight is true; too ambiguous to
                    # be able to tell what timezone has daylight savings
                    if (time.tzname[0] == time.tzname[1] and
                       time.daylight and found_zone not in ("utc", "gmt")):
                        break
                    else:
                        tz = value
                        break
    # Deal with the cases where ambiguities arize
    # don't assume default values for ISO week/year
    if year is None and iso_year is not None:
        if iso_week is None or weekday is None:
            raise ValueError("ISO year directive '%G' must be used with "
                             "the ISO week directive '%V' and a weekday "
                             "directive ('%A', '%a', '%w', or '%u').")
        if julian is not None:
            raise ValueError("Day of the year directive '%j' is not "
                             "compatible with ISO year directive '%G'. "
                             "Use '%Y' instead.")
    elif week_of_year is None and iso_week is not None:
        if weekday is None:
            raise ValueError("ISO week directive '%V' must be used with "
                             "the ISO year directive '%G' and a weekday "
                             "directive ('%A', '%a', '%w', or '%u').")
        else:
            raise ValueError("ISO week directive '%V' is incompatible with "
                             "the year directive '%Y'. Use the ISO year '%G' "
                             "instead.")

    leap_year_fix = False
    if year is None and month == 2 and day == 29:
        year = 1904  # 1904 is first leap year of 20th century
        leap_year_fix = True
    elif year is None:
        year = 1900


    # If we know the week of the year and what day of that week, we can figure
    # out the Julian day of the year.
    if julian is None and weekday is not None:
        if week_of_year is not None:
            week_starts_Mon = True if week_of_year_start == 0 else False
            julian = _calc_julian_from_U_or_W(year, week_of_year, weekday,
                                                week_starts_Mon)
        elif iso_year is not None and iso_week is not None:
            year, julian = _calc_julian_from_V(iso_year, iso_week, weekday + 1)
        if julian is not None and julian <= 0:
            year -= 1
            yday = 366 if calendar.isleap(year) else 365
            julian += yday

    if julian is None:
        # Cannot pre-calculate datetime_date() since can change in Julian
        # calculation and thus could have different value for the day of
        # the week calculation.
        # Need to add 1 to result since first day of the year is 1, not 0.
        julian = datetime_date(year, month, day).toordinal() - \
                  datetime_date(year, 1, 1).toordinal() + 1
    else:  # Assume that if they bothered to include Julian day (or if it was
           # calculated above with year/week/weekday) it will be accurate.
        datetime_result = datetime_date.fromordinal(
                            (julian - 1) +
                            datetime_date(year, 1, 1).toordinal())
        year = datetime_result.year
        month = datetime_result.month
        day = datetime_result.day
    if weekday is None:
        weekday = datetime_date(year, month, day).weekday()
    # Add timezone info
    tzname = found_dict.get("Z")

    if leap_year_fix:
        # the caller didn't supply a year but asked for Feb 29th. We couldn't
        # use the default of 1900 for computations. We set it back to ensure
        # that February 29th is smaller than March 1st.
        year = 1900

    return (year, month, day,
            hour, minute, second,
            weekday, julian, tz, tzname, gmtoff), fraction, gmtoff_fraction

def _strptime_time(data_string, format="%a %b %d %H:%M:%S %Y"):
    """Return a time struct based on the input string and the
    format string."""
    tt = _strptime(data_string, format)[0]
    return time.struct_time(tt[:time._STRUCT_TM_ITEMS])

def _strptime_datetime(cls, data_string, format="%a %b %d %H:%M:%S %Y"):
    """Return a class cls instance based on the input string and the
    format string."""
    tt, fraction, gmtoff_fraction = _strptime(data_string, format)
    tzname, gmtoff = tt[-2:]
    args = tt[:6] + (fraction,)
    if gmtoff is not None:
        tzdelta = datetime_timedelta(seconds=gmtoff, microseconds=gmtoff_fraction)
        if tzname:
            tz = datetime_timezone(tzdelta, tzname)
        else:
            tz = datetime_timezone(tzdelta)
        args += (tz,)

    return cls(*args)
<|endoftext|>#! /usr/bin/python3.10

""" This module tries to retrieve as much platform-identifying data as
    possible. It makes this information available via function APIs.

    If called from the command line, it prints the platform
    information concatenated as single string to stdout. The output
    format is useable as part of a filename.

"""
#    This module is maintained by Marc-Andre Lemburg <mal@egenix.com>.
#    If you find problems, please submit bug reports/patches via the
#    Python bug tracker (http://bugs.python.org) and assign them to "lemburg".
#
#    Still needed:
#    * support for MS-DOS (PythonDX ?)
#    * support for Amiga and other still unsupported platforms running Python
#    * support for additional Linux distributions
#
#    Many thanks to all those who helped adding platform-specific
#    checks (in no particular order):
#
#      Charles G Waldman, David Arnold, Gordon McMillan, Ben Darnell,
#      Jeff Bauer, Cliff Crawford, Ivan Van Laningham, Josef
#      Betancourt, Randall Hopper, Karl Putland, John Farrell, Greg
#      Andruk, Just van Rossum, Thomas Heller, Mark R. Levinson, Mark
#      Hammond, Bill Tutt, Hans Nowak, Uwe Zessin (OpenVMS support),
#      Colin Kong, Trent Mick, Guido van Rossum, Anthony Baxter, Steve
#      Dower
#
#    History:
#
#    <see CVS and SVN checkin messages for history>
#
#    1.0.8 - changed Windows support to read version from kernel32.dll
#    1.0.7 - added DEV_NULL
#    1.0.6 - added linux_distribution()
#    1.0.5 - fixed Java support to allow running the module on Jython
#    1.0.4 - added IronPython support
#    1.0.3 - added normalization of Windows system name
#    1.0.2 - added more Windows support
#    1.0.1 - reformatted to make doc.py happy
#    1.0.0 - reformatted a bit and checked into Python CVS
#    0.8.0 - added sys.version parser and various new access
#            APIs (python_version(), python_compiler(), etc.)
#    0.7.2 - fixed architecture() to use sizeof(pointer) where available
#    0.7.1 - added support for Caldera OpenLinux
#    0.7.0 - some fixes for WinCE; untabified the source file
#    0.6.2 - support for OpenVMS - requires version 1.5.2-V006 or higher and
#            vms_lib.getsyi() configured
#    0.6.1 - added code to prevent 'uname -p' on platforms which are
#            known not to support it
#    0.6.0 - fixed win32_ver() to hopefully work on Win95,98,NT and Win2k;
#            did some cleanup of the interfaces - some APIs have changed
#    0.5.5 - fixed another type in the MacOS code... should have
#            used more coffee today ;-)
#    0.5.4 - fixed a few typos in the MacOS code
#    0.5.3 - added experimental MacOS support; added better popen()
#            workarounds in _syscmd_ver() -- still not 100% elegant
#            though
#    0.5.2 - fixed uname() to return '' instead of 'unknown' in all
#            return values (the system uname command tends to return
#            'unknown' instead of just leaving the field empty)
#    0.5.1 - included code for slackware dist; added exception handlers
#            to cover up situations where platforms don't have os.popen
#            (e.g. Mac) or fail on socket.gethostname(); fixed libc
#            detection RE
#    0.5.0 - changed the API names referring to system commands to *syscmd*;
#            added java_ver(); made syscmd_ver() a private
#            API (was system_ver() in previous versions) -- use uname()
#            instead; extended the win32_ver() to also return processor
#            type information
#    0.4.0 - added win32_ver() and modified the platform() output for WinXX
#    0.3.4 - fixed a bug in _follow_symlinks()
#    0.3.3 - fixed popen() and "file" command invocation bugs
#    0.3.2 - added architecture() API and support for it in platform()
#    0.3.1 - fixed syscmd_ver() RE to support Windows NT
#    0.3.0 - added system alias support
#    0.2.3 - removed 'wince' again... oh well.
#    0.2.2 - added 'wince' to syscmd_ver() supported platforms
#    0.2.1 - added cache logic and changed the platform string format
#    0.2.0 - changed the API to use functions instead of module globals
#            since some action take too long to be run on module import
#    0.1.0 - first release
#
#    You can always get the latest version of this module at:
#
#             http://www.egenix.com/files/python/platform.py
#
#    If that URL should fail, try contacting the author.

__copyright__ = """
    Copyright (c) 1999-2000, Marc-Andre Lemburg; mailto:mal@lemburg.com
    Copyright (c) 2000-2010, eGenix.com Software GmbH; mailto:info@egenix.com

    Permission to use, copy, modify, and distribute this software and its
    documentation for any purpose and without fee or royalty is hereby granted,
    provided that the above copyright notice appear in all copies and that
    both that copyright notice and this permission notice appear in
    supporting documentation or portions thereof, including modifications,
    that you make.

    EGENIX.COM SOFTWARE GMBH DISCLAIMS ALL WARRANTIES WITH REGARD TO
    THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
    FITNESS, IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL,
    INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
    FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
    NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
    WITH THE USE OR PERFORMANCE OF THIS SOFTWARE !

"""

__version__ = '1.0.8'

import collections
import os
import re
import sys
import subprocess
import functools
import itertools

### Globals & Constants

# Helper for comparing two version number strings.
# Based on the description of the PHP's version_compare():
# http://php.net/manual/en/function.version-compare.php

_ver_stages = {
    # any string not found in this dict, will get 0 assigned
    'dev': 10,
    'alpha': 20, 'a': 20,
    'beta': 30, 'b': 30,
    'c': 40,
    'RC': 50, 'rc': 50,
    # number, will get 100 assigned
    'pl': 200, 'p': 200,
}

_component_re = re.compile(r'([0-9]+|[._+-])')

def _comparable_version(version):
    result = []
    for v in _component_re.split(version):
        if v not in '._+-':
            try:
                v = int(v, 10)
                t = 100
            except ValueError:
                t = _ver_stages.get(v, 0)
            result.extend((t, v))
    return result

### Platform specific APIs

_libc_search = re.compile(b'(__libc_init)'
                          b'|'
                          b'(GLIBC_([0-9.]+))'
                          b'|'
                          br'(libc(_\w+)?\.so(?:\.(\d[0-9.]*))?)', re.ASCII)

def libc_ver(executable=None, lib='', version='', chunksize=16384):

    """ Tries to determine the libc version that the file executable
        (which defaults to the Python interpreter) is linked against.

        Returns a tuple of strings (lib,version) which default to the
        given parameters in case the lookup fails.

        Note that the function has intimate knowledge of how different
        libc versions add symbols to the executable and thus is probably
        only useable for executables compiled using gcc.

        The file is read and scanned in chunks of chunksize bytes.

    """
    if not executable:
        try:
            ver = os.confstr('CS_GNU_LIBC_VERSION')
            # parse 'glibc 2.28' as ('glibc', '2.28')
            parts = ver.split(maxsplit=1)
            if len(parts) == 2:
                return tuple(parts)
        except (AttributeError, ValueError, OSError):
            # os.confstr() or CS_GNU_LIBC_VERSION value not available
            pass

        executable = sys.executable

    V = _comparable_version
    if hasattr(os.path, 'realpath'):
        # Python 2.2 introduced os.path.realpath(); it is used
        # here to work around problems with Cygwin not being
        # able to open symlinks for reading
        executable = os.path.realpath(executable)
    with open(executable, 'rb') as f:
        binary = f.read(chunksize)
        pos = 0
        while pos < len(binary):
            if b'libc' in binary or b'GLIBC' in binary:
                m = _libc_search.search(binary, pos)
            else:
                m = None
            if not m or m.end() == len(binary):
                chunk = f.read(chunksize)
                if chunk:
                    binary = binary[max(pos, len(binary) - 1000):] + chunk
                    pos = 0
                    continue
                if not m:
                    break
            libcinit, glibc, glibcversion, so, threads, soversion = [
                s.decode('latin1') if s is not None else s
                for s in m.groups()]
            if libcinit and not lib:
                lib = 'libc'
            elif glibc:
                if lib != 'glibc':
                    lib = 'glibc'
                    version = glibcversion
                elif V(glibcversion) > V(version):
                    version = glibcversion
            elif so:
                if lib != 'glibc':
                    lib = 'libc'
                    if soversion and (not version or V(soversion) > V(version)):
                        version = soversion
                    if threads and version[-len(threads):] != threads:
                        version = version + threads
            pos = m.end()
    return lib, version

def _norm_version(version, build=''):

    """ Normalize the version and build strings and return a single
        version string using the format major.minor.build (or patchlevel).
    """
    l = version.split('.')
    if build:
        l.append(build)
    try:
        strings = list(map(str, map(int, l)))
    except ValueError:
        strings = l
    version = '.'.join(strings[:3])
    return version

_ver_output = re.compile(r'(?:([\w ]+) ([\w.]+) '
                         r'.*'
                         r'\[.* ([\d.]+)\])')

# Examples of VER command output:
#
#   Windows 2000:  Microsoft Windows 2000 [Version 5.00.2195]
#   Windows XP:    Microsoft Windows XP [Version 5.1.2600]
#   Windows Vista: Microsoft Windows [Version 6.0.6002]
#
# Note that the "Version" string gets localized on different
# Windows versions.

def _syscmd_ver(system='', release='', version='',

               supported_platforms=('win32', 'win16', 'dos')):

    """ Tries to figure out the OS version used and returns
        a tuple (system, release, version).

        It uses the "ver" shell command for this which is known
        to exists on Windows, DOS. XXX Others too ?

        In case this fails, the given parameters are used as
        defaults.

    """
    if sys.platform not in supported_platforms:
        return system, release, version

    # Try some common cmd strings
    import subprocess
    for cmd in ('ver', 'command /c ver', 'cmd /c ver'):
        try:
            info = subprocess.check_output(cmd,
                                           stdin=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL,
                                           text=True,
                                           shell=True)
        except (OSError, subprocess.CalledProcessError) as why:
            #print('Command %s failed: %s' % (cmd, why))
            continue
        else:
            break
    else:
        return system, release, version

    # Parse the output
    info = info.strip()
    m = _ver_output.match(info)
    if m is not None:
        system, release, version = m.groups()
        # Strip trailing dots from version and release
        if release[-1] == '.':
            release = release[:-1]
        if version[-1] == '.':
            version = version[:-1]
        # Normalize the version and build strings (eliminating additional
        # zeros)
        version = _norm_version(version)
    return system, release, version

_WIN32_CLIENT_RELEASES = {
    (5, 0): "2000",
    (5, 1): "XP",
    # Strictly, 5.2 client is XP 64-bit, but platform.py historically
    # has always called it 2003 Server
    (5, 2): "2003Server",
    (5, None): "post2003",

    (6, 0): "Vista",
    (6, 1): "7",
    (6, 2): "8",
    (6, 3): "8.1",
    (6, None): "post8.1",

    (10, 0): "10",
    (10, None): "post10",
}

# Server release name lookup will default to client names if necessary
_WIN32_SERVER_RELEASES = {
    (5, 2): "2003Server",

    (6, 0): "2008Server",
    (6, 1): "2008ServerR2",
    (6, 2): "2012Server",
    (6, 3): "2012ServerR2",
    (6, None): "post2012ServerR2",
}

def win32_is_iot():
    return win32_edition() in ('IoTUAP', 'NanoServer', 'WindowsCoreHeadless', 'IoTEdgeOS')

def win32_edition():
    try:
        try:
            import winreg
        except ImportError:
            import _winreg as winreg
    except ImportError:
        pass
    else:
        try:
            cvkey = r'SOFTWARE\Microsoft\Windows NT\CurrentVersion'
            with winreg.OpenKeyEx(winreg.HKEY_LOCAL_MACHINE, cvkey) as key:
                return winreg.QueryValueEx(key, 'EditionId')[0]
        except OSError:
            pass

    return None

def win32_ver(release='', version='', csd='', ptype=''):
    try:
        from sys import getwindowsversion
    except ImportError:
        return release, version, csd, ptype

    winver = getwindowsversion()
    try:
        major, minor, build = map(int, _syscmd_ver()[2].split('.'))
    except ValueError:
        major, minor, build = winver.platform_version or winver[:3]
    version = '{0}.{1}.{2}'.format(major, minor, build)

    release = (_WIN32_CLIENT_RELEASES.get((major, minor)) or
               _WIN32_CLIENT_RELEASES.get((major, None)) or
               release)

    # getwindowsversion() reflect the compatibility mode Python is
    # running under, and so the service pack value is only going to be
    # valid if the versions match.
    if winver[:2] == (major, minor):
        try:
            csd = 'SP{}'.format(winver.service_pack_major)
        except AttributeError:
            if csd[:13] == 'Service Pack ':
                csd = 'SP' + csd[13:]

    # VER_NT_SERVER = 3
    if getattr(winver, 'product_type', None) == 3:
        release = (_WIN32_SERVER_RELEASES.get((major, minor)) or
                   _WIN32_SERVER_RELEASES.get((major, None)) or
                   release)

    try:
        try:
            import winreg
        except ImportError:
            import _winreg as winreg
    except ImportError:
        pass
    else:
        try:
            cvkey = r'SOFTWARE\Microsoft\Windows NT\CurrentVersion'
            with winreg.OpenKeyEx(winreg.HKEY_LOCAL_MACHINE, cvkey) as key:
                ptype = winreg.QueryValueEx(key, 'CurrentType')[0]
        except OSError:
            pass

    return release, version, csd, ptype


def _mac_ver_xml():
    fn = '/System/Library/CoreServices/SystemVersion.plist'
    if not os.path.exists(fn):
        return None

    try:
        import plistlib
    except ImportError:
        return None

    with open(fn, 'rb') as f:
        pl = plistlib.load(f)
    release = pl['ProductVersion']
    versioninfo = ('', '', '')
    machine = os.uname().machine
    if machine in ('ppc', 'Power Macintosh'):
        # Canonical name
        machine = 'PowerPC'

    return release, versioninfo, machine


def mac_ver(release='', versioninfo=('', '', ''), machine=''):

    """ Get macOS version information and return it as tuple (release,
        versioninfo, machine) with versioninfo being a tuple (version,
        dev_stage, non_release_version).

        Entries which cannot be determined are set to the parameter values
        which default to ''. All tuple entries are strings.
    """

    # First try reading the information from an XML file which should
    # always be present
    info = _mac_ver_xml()
    if info is not None:
        return info

    # If that also doesn't work return the default values
    return release, versioninfo, machine

def _java_getprop(name, default):

    from java.lang import System
    try:
        value = System.getProperty(name)
        if value is None:
            return default
        return value
    except AttributeError:
        return default

def java_ver(release='', vendor='', vminfo=('', '', ''), osinfo=('', '', '')):

    """ Version interface for Jython.

        Returns a tuple (release, vendor, vminfo, osinfo) with vminfo being
        a tuple (vm_name, vm_release, vm_vendor) and osinfo being a
        tuple (os_name, os_version, os_arch).

        Values which cannot be determined are set to the defaults
        given as parameters (which all default to '').

    """
    # Import the needed APIs
    try:
        import java.lang
    except ImportError:
        return release, vendor, vminfo, osinfo

    vendor = _java_getprop('java.vendor', vendor)
    release = _java_getprop('java.version', release)
    vm_name, vm_release, vm_vendor = vminfo
    vm_name = _java_getprop('java.vm.name', vm_name)
    vm_vendor = _java_getprop('java.vm.vendor', vm_vendor)
    vm_release = _java_getprop('java.vm.version', vm_release)
    vminfo = vm_name, vm_release, vm_vendor
    os_name, os_version, os_arch = osinfo
    os_arch = _java_getprop('java.os.arch', os_arch)
    os_name = _java_getprop('java.os.name', os_name)
    os_version = _java_getprop('java.os.version', os_version)
    osinfo = os_name, os_version, os_arch

    return release, vendor, vminfo, osinfo

### System name aliasing

def system_alias(system, release, version):

    """ Returns (system, release, version) aliased to common
        marketing names used for some systems.

        It also does some reordering of the information in some cases
        where it would otherwise cause confusion.

    """
    if system == 'SunOS':
        # Sun's OS
        if release < '5':
            # These releases use the old name SunOS
            return system, release, version
        # Modify release (marketing release = SunOS release - 3)
        l = release.split('.')
        if l:
            try:
                major = int(l[0])
            except ValueError:
                pass
            else:
                major = major - 3
                l[0] = str(major)
                release = '.'.join(l)
        if release < '6':
            system = 'Solaris'
        else:
            # XXX Whatever the new SunOS marketing name is...
            system = 'Solaris'

    elif system in ('win32', 'win16'):
        # In case one of the other tricks
        system = 'Windows'

    # bpo-35516: Don't replace Darwin with macOS since input release and
    # version arguments can be different than the currently running version.

    return system, release, version

### Various internal helpers

def _platform(*args):

    """ Helper to format the platform string in a filename
        compatible format e.g. "system-version-machine".
    """
    # Format the platform string
    platform = '-'.join(x.strip() for x in filter(len, args))

    # Cleanup some possible filename obstacles...
    platform = platform.replace(' ', '_')
    platform = platform.replace('/', '-')
    platform = platform.replace('\\', '-')
    platform = platform.replace(':', '-')
    platform = platform.replace(';', '-')
    platform = platform.replace('"', '-')
    platform = platform.replace('(', '-')
    platform = platform.replace(')', '-')

    # No need to report 'unknown' information...
    platform = platform.replace('unknown', '')

    # Fold '--'s and remove trailing '-'
    while 1:
        cleaned = platform.replace('--', '-')
        if cleaned == platform:
            break
        platform = cleaned
    while platform[-1] == '-':
        platform = platform[:-1]

    return platform

def _node(default=''):

    """ Helper to determine the node name of this machine.
    """
    try:
        import socket
    except ImportError:
        # No sockets...
        return default
    try:
        return socket.gethostname()
    except OSError:
        # Still not working...
        return default

def _follow_symlinks(filepath):

    """ In case filepath is a symlink, follow it until a
        real file is reached.
    """
    filepath = os.path.abspath(filepath)
    while os.path.islink(filepath):
        filepath = os.path.normpath(
            os.path.join(os.path.dirname(filepath), os.readlink(filepath)))
    return filepath


def _syscmd_file(target, default=''):

    """ Interface to the system's file command.

        The function uses the -b option of the file command to have it
        omit the filename in its output. Follow the symlinks. It returns
        default in case the command should fail.

    """
    if sys.platform in ('dos', 'win32', 'win16'):
        # XXX Others too ?
        return default

    import subprocess
    target = _follow_symlinks(target)
    # "file" output is locale dependent: force the usage of the C locale
    # to get deterministic behavior.
    env = dict(os.environ, LC_ALL='C')
    try:
        # -b: do not prepend filenames to output lines (brief mode)
        output = subprocess.check_output(['file', '-b', target],
                                         stderr=subprocess.DEVNULL,
                                         env=env)
    except (OSError, subprocess.CalledProcessError):
        return default
    if not output:
        return default
    # With the C locale, the output should be mostly ASCII-compatible.
    # Decode from Latin-1 to prevent Unicode decode error.
    return output.decode('latin-1')

### Information about the used architecture

# Default values for architecture; non-empty strings override the
# defaults given as parameters
_default_architecture = {
    'win32': ('', 'WindowsPE'),
    'win16': ('', 'Windows'),
    'dos': ('', 'MSDOS'),
}

def architecture(executable=sys.executable, bits='', linkage=''):

    """ Queries the given executable (defaults to the Python interpreter
        binary) for various architecture information.

        Returns a tuple (bits, linkage) which contains information about
        the bit architecture and the linkage format used for the
        executable. Both values are returned as strings.

        Values that cannot be determined are returned as given by the
        parameter presets. If bits is given as '', the sizeof(pointer)
        (or sizeof(long) on Python version < 1.5.2) is used as
        indicator for the supported pointer size.

        The function relies on the system's "file" command to do the
        actual work. This is available on most if not all Unix
        platforms. On some non-Unix platforms where the "file" command
        does not exist and the executable is set to the Python interpreter
        binary defaults from _default_architecture are used.

    """
    # Use the sizeof(pointer) as default number of bits if nothing
    # else is given as default.
    if not bits:
        import struct
        size = struct.calcsize('P')
        bits = str(size * 8) + 'bit'

    # Get data from the 'file' system command
    if executable:
        fileout = _syscmd_file(executable, '')
    else:
        fileout = ''

    if not fileout and \
       executable == sys.executable:
        # "file" command did not return anything; we'll try to provide
        # some sensible defaults then...
        if sys.platform in _default_architecture:
            b, l = _default_architecture[sys.platform]
            if b:
                bits = b
            if l:
                linkage = l
        return bits, linkage

    if 'executable' not in fileout and 'shared object' not in fileout:
        # Format not supported
        return bits, linkage

    # Bits
    if '32-bit' in fileout:
        bits = '32bit'
    elif '64-bit' in fileout:
        bits = '64bit'

    # Linkage
    if 'ELF' in fileout:
        linkage = 'ELF'
    elif 'PE' in fileout:
        # E.g. Windows uses this format
        if 'Windows' in fileout:
            linkage = 'WindowsPE'
        else:
            linkage = 'PE'
    elif 'COFF' in fileout:
        linkage = 'COFF'
    elif 'MS-DOS' in fileout:
        linkage = 'MSDOS'
    else:
        # XXX the A.OUT format also falls under this class...
        pass

    return bits, linkage


def _get_machine_win32():
    # Try to use the PROCESSOR_* environment variables
    # available on Win XP and later; see
    # http://support.microsoft.com/kb/888731 and
    # http://www.geocities.com/rick_lively/MANUALS/ENV/MSWIN/PROCESSI.HTM

    # WOW64 processes mask the native architecture
    return (
        os.environ.get('PROCESSOR_ARCHITEW6432', '') or
        os.environ.get('PROCESSOR_ARCHITECTURE', '')
    )


class _Processor:
    @classmethod
    def get(cls):
        func = getattr(cls, f'get_{sys.platform}', cls.from_subprocess)
        return func() or ''

    def get_win32():
        return os.environ.get('PROCESSOR_IDENTIFIER', _get_machine_win32())

    def get_OpenVMS():
        try:
            import vms_lib
        except ImportError:
            pass
        else:
            csid, cpu_number = vms_lib.getsyi('SYI$_CPU', 0)
            return 'Alpha' if cpu_number >= 128 else 'VAX'

    def from_subprocess():
        """
        Fall back to `uname -p`
        """
        try:
            return subprocess.check_output(
                ['uname', '-p'],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass


def _unknown_as_blank(val):
    return '' if val == 'unknown' else val


### Portable uname() interface

class uname_result(
    collections.namedtuple(
        "uname_result_base",
        "system node release version machine")
        ):
    """
    A uname_result that's largely compatible with a
    simple namedtuple except that 'processor' is
    resolved late and cached to avoid calling "uname"
    except when needed.
    """

    _fields = ('system', 'node', 'release', 'version', 'machine', 'processor')

    @functools.cached_property
    def processor(self):
        return _unknown_as_blank(_Processor.get())

    def __iter__(self):
        return itertools.chain(
            super().__iter__(),
            (self.processor,)
        )

    @classmethod
    def _make(cls, iterable):
        # override factory to affect length check
        num_fields = len(cls._fields) - 1
        result = cls.__new__(cls, *iterable)
        if len(result) != num_fields + 1:
            msg = f'Expected {num_fields} arguments, got {len(result)}'
            raise TypeError(msg)
        return result

    def __getitem__(self, key):
        return tuple(self)[key]

    def __len__(self):
        return len(tuple(iter(self)))

    def __reduce__(self):
        return uname_result, tuple(self)[:len(self._fields) - 1]


_uname_cache = None


def uname():

    """ Fairly portable uname interface. Returns a tuple
        of strings (system, node, release, version, machine, processor)
        identifying the underlying platform.

        Note that unlike the os.uname function this also returns
        possible processor information as an additional tuple entry.

        Entries which cannot be determined are set to ''.

    """
    global _uname_cache

    if _uname_cache is not None:
        return _uname_cache

    # Get some infos from the builtin os.uname API...
    try:
        system, node, release, version, machine = infos = os.uname()
    except AttributeError:
        system = sys.platform
        node = _node()
        release = version = machine = ''
        infos = ()

    if not any(infos):
        # uname is not available

        # Try win32_ver() on win32 platforms
        if system == 'win32':
            release, version, csd, ptype = win32_ver()
            machine = machine or _get_machine_win32()

        # Try the 'ver' system command available on some
        # platforms
        if not (release and version):
            system, release, version = _syscmd_ver(system)
            # Normalize system to what win32_ver() normally returns
            # (_syscmd_ver() tends to return the vendor name as well)
            if system == 'Microsoft Windows':
                system = 'Windows'
            elif system == 'Microsoft' and release == 'Windows':
                # Under Windows Vista and Windows Server 2008,
                # Microsoft changed the output of the ver command. The
                # release is no longer printed.  This causes the
                # system and release to be misidentified.
                system = 'Windows'
                if '6.0' == version[:3]:
                    release = 'Vista'
                else:
                    release = ''

        # In case we still don't know anything useful, we'll try to
        # help ourselves
        if system in ('win32', 'win16'):
            if not version:
                if system == 'win32':
                    version = '32bit'
                else:
                    version = '16bit'
            system = 'Windows'

        elif system[:4] == 'java':
            release, vendor, vminfo, osinfo = java_ver()
            system = 'Java'
            version = ', '.join(vminfo)
            if not version:
                version = vendor

    # System specific extensions
    if system == 'OpenVMS':
        # OpenVMS seems to have release and version mixed up
        if not release or release == '0':
            release = version
            version = ''

    #  normalize name
    if system == 'Microsoft' and release == 'Windows':
        system = 'Windows'
        release = 'Vista'

    vals = system, node, release, version, machine
    # Replace 'unknown' values with the more portable ''
    _uname_cache = uname_result(*map(_unknown_as_blank, vals))
    return _uname_cache

### Direct interfaces to some of the uname() return values

def system():

    """ Returns the system/OS name, e.g. 'Linux', 'Windows' or 'Java'.

        An empty string is returned if the value cannot be determined.

    """
    return uname().system

def node():

    """ Returns the computer's network name (which may not be fully
        qualified)

        An empty string is returned if the value cannot be determined.

    """
    return uname().node

def release():

    """ Returns the system's release, e.g. '2.2.0' or 'NT'

        An empty string is returned if the value cannot be determined.

    """
    return uname().release

def version():

    """ Returns the system's release version, e.g. '#3 on degas'

        An empty string is returned if the value cannot be determined.

    """
    return uname().version

def machine():

    """ Returns the machine type, e.g. 'i386'

        An empty string is returned if the value cannot be determined.

    """
    return uname().machine

def processor():

    """ Returns the (true) processor name, e.g. 'amdk6'

        An empty string is returned if the value cannot be
        determined. Note that many platforms do not provide this
        information or simply return the same value as for machine(),
        e.g.  NetBSD does this.

    """
    return uname().processor

### Various APIs for extracting information from sys.version

_sys_version_parser = re.compile(
    r'([\w.+]+)\s*'  # "version<space>"
    r'\(#?([^,]+)'  # "(#buildno"
    r'(?:,\s*([\w ]*)'  # ", builddate"
    r'(?:,\s*([\w :]*))?)?\)\s*'  # ", buildtime)<space>"
    r'\[([^\]]+)\]?', re.ASCII)  # "[compiler]"

_ironpython_sys_version_parser = re.compile(
    r'IronPython\s*'
    r'([\d\.]+)'
    r'(?: \(([\d\.]+)\))?'
    r' on (.NET [\d\.]+)', re.ASCII)

# IronPython covering 2.6 and 2.7
_ironpython26_sys_version_parser = re.compile(
    r'([\d.]+)\s*'
    r'\(IronPython\s*'
    r'[\d.]+\s*'
    r'\(([\d.]+)\) on ([\w.]+ [\d.]+(?: \(\d+-bit\))?)\)'
)

_pypy_sys_version_parser = re.compile(
    r'([\w.+]+)\s*'
    r'\(#?([^,]+),\s*([\w ]+),\s*([\w :]+)\)\s*'
    r'\[PyPy [^\]]+\]?')

_sys_version_cache = {}

def _sys_version(sys_version=None):

    """ Returns a parsed version of Python's sys.version as tuple
        (name, version, branch, revision, buildno, builddate, compiler)
        referring to the Python implementation name, version, branch,
        revision, build number, build date/time as string and the compiler
        identification string.

        Note that unlike the Python sys.version, the returned value
        for the Python version will always include the patchlevel (it
        defaults to '.0').

        The function returns empty strings for tuple entries that
        cannot be determined.

        sys_version may be given to parse an alternative version
        string, e.g. if the version was read from a different Python
        interpreter.

    """
    # Get the Python version
    if sys_version is None:
        sys_version = sys.version

    # Try the cache first
    result = _sys_version_cache.get(sys_version, None)
    if result is not None:
        return result

    # Parse it
    if 'IronPython' in sys_version:
        # IronPython
        name = 'IronPython'
        if sys_version.startswith('IronPython'):
            match = _ironpython_sys_version_parser.match(sys_version)
        else:
            match = _ironpython26_sys_version_parser.match(sys_version)

        if match is None:
            raise ValueError(
                'failed to parse IronPython sys.version: %s' %
                repr(sys_version))

        version, alt_version, compiler = match.groups()
        buildno = ''
        builddate = ''

    elif sys.platform.startswith('java'):
        # Jython
        name = 'Jython'
        match = _sys_version_parser.match(sys_version)
        if match is None:
            raise ValueError(
                'failed to parse Jython sys.version: %s' %
                repr(sys_version))
        version, buildno, builddate, buildtime, _ = match.groups()
        if builddate is None:
            builddate = ''
        compiler = sys.platform

    elif "PyPy" in sys_version:
        # PyPy
        name = "PyPy"
        match = _pypy_sys_version_parser.match(sys_version)
        if match is None:
            raise ValueError("failed to parse PyPy sys.version: %s" %
                             repr(sys_version))
        version, buildno, builddate, buildtime = match.groups()
        compiler = ""

    else:
        # CPython
        match = _sys_version_parser.match(sys_version)
        if match is None:
            raise ValueError(
                'failed to parse CPython sys.version: %s' %
                repr(sys_version))
        version, buildno, builddate, buildtime, compiler = \
              match.groups()
        name = 'CPython'
        if builddate is None:
            builddate = ''
        elif buildtime:
            builddate = builddate + ' ' + buildtime

    if hasattr(sys, '_git'):
        _, branch, revision = sys._git
    elif hasattr(sys, '_mercurial'):
        _, branch, revision = sys._mercurial
    else:
        branch = ''
        revision = ''

    # Add the patchlevel version if missing
    l = version.split('.')
    if len(l) == 2:
        l.append('0')
        version = '.'.join(l)

    # Build and cache the result
    result = (name, version, branch, revision, buildno, builddate, compiler)
    _sys_version_cache[sys_version] = result
    return result

def python_implementation():

    """ Returns a string identifying the Python implementation.

        Currently, the following implementations are identified:
          'CPython' (C implementation of Python),
          'IronPython' (.NET implementation of Python),
          'Jython' (Java implementation of Python),
          'PyPy' (Python implementation of Python).

    """
    return _sys_version()[0]

def python_version():

    """ Returns the Python version as string 'major.minor.patchlevel'

        Note that unlike the Python sys.version, the returned value
        will always include the patchlevel (it defaults to 0).

    """
    return _sys_version()[1]

def python_version_tuple():

    """ Returns the Python version as tuple (major, minor, patchlevel)
        of strings.

        Note that unlike the Python sys.version, the returned value
        will always include the patchlevel (it defaults to 0).

    """
    return tuple(_sys_version()[1].split('.'))

def python_branch():

    """ Returns a string identifying the Python implementation
        branch.

        For CPython this is the SCM branch from which the
        Python binary was built.

        If not available, an empty string is returned.

    """

    return _sys_version()[2]

def python_revision():

    """ Returns a string identifying the Python implementation
        revision.

        For CPython this is the SCM revision from which the
        Python binary was built.

        If not available, an empty string is returned.

    """
    return _sys_version()[3]

def python_build():

    """ Returns a tuple (buildno, builddate) stating the Python
        build number and date as strings.

    """
    return _sys_version()[4:6]

def python_compiler():

    """ Returns a string identifying the compiler used for compiling
        Python.

    """
    return _sys_version()[6]

### The Opus Magnum of platform strings :-)

_platform_cache = {}

def platform(aliased=0, terse=0):

    """ Returns a single string identifying the underlying platform
        with as much useful information as possible (but no more :).

        The output is intended to be human readable rather than
        machine parseable. It may look different on different
        platforms and this is intended.

        If "aliased" is true, the function will use aliases for
        various platforms that report system names which differ from
        their common names, e.g. SunOS will be reported as
        Solaris. The system_alias() function is used to implement
        this.

        Setting terse to true causes the function to return only the
        absolute minimum information needed to identify the platform.

    """
    result = _platform_cache.get((aliased, terse), None)
    if result is not None:
        return result

    # Get uname information and then apply platform specific cosmetics
    # to it...
    system, node, release, version, machine, processor = uname()
    if machine == processor:
        processor = ''
    if aliased:
        system, release, version = system_alias(system, release, version)

    if system == 'Darwin':
        # macOS (darwin kernel)
        macos_release = mac_ver()[0]
        if macos_release:
            system = 'macOS'
            release = macos_release

    if system == 'Windows':
        # MS platforms
        rel, vers, csd, ptype = win32_ver(version)
        if terse:
            platform = _platform(system, release)
        else:
            platform = _platform(system, release, version, csd)

    elif system in ('Linux',):
        # check for libc vs. glibc
        libcname, libcversion = libc_ver()
        platform = _platform(system, release, machine, processor,
                             'with',
                             libcname+libcversion)
    elif system == 'Java':
        # Java platforms
        r, v, vminfo, (os_name, os_version, os_arch) = java_ver()
        if terse or not os_name:
            platform = _platform(system, release, version)
        else:
            platform = _platform(system, release, version,
                                 'on',
                                 os_name, os_version, os_arch)

    else:
        # Generic handler
        if terse:
            platform = _platform(system, release)
        else:
            bits, linkage = architecture(sys.executable)
            platform = _platform(system, release, machine,
                                 processor, bits, linkage)

    _platform_cache[(aliased, terse)] = platform
    return platform

### freedesktop.org os-release standard
# https://www.freedesktop.org/software/systemd/man/os-release.html

# NAME=value with optional quotes (' or "). The regular expression is less
# strict than shell lexer, but that's ok.
_os_release_line = re.compile(
    "^(?P<name>[a-zA-Z0-9_]+)=(?P<quote>[\"\']?)(?P<value>.*)(?P=quote)$"
)
# unescape five special characters mentioned in the standard
_os_release_unescape = re.compile(r"\\([\\\$\"\'`])")
# /etc takes precedence over /usr/lib
_os_release_candidates = ("/etc/os-release", "/usr/lib/os-release")
_os_release_cache = None


def _parse_os_release(lines):
    # These fields are mandatory fields with well-known defaults
    # in practice all Linux distributions override NAME, ID, and PRETTY_NAME.
    info = {
        "NAME": "Linux",
        "ID": "linux",
        "PRETTY_NAME": "Linux",
    }

    for line in lines:
        mo = _os_release_line.match(line)
        if mo is not None:
            info[mo.group('name')] = _os_release_unescape.sub(
                r"\1", mo.group('value')
            )

    return info


def freedesktop_os_release():
    """Return operation system identification from freedesktop.org os-release
    """
    global _os_release_cache

    if _os_release_cache is None:
        errno = None
        for candidate in _os_release_candidates:
            try:
                with open(candidate, encoding="utf-8") as f:
                    _os_release_cache = _parse_os_release(f)
                break
            except OSError as e:
                errno = e.errno
        else:
            raise OSError(
                errno,
                f"Unable to read files {', '.join(_os_release_candidates)}"
            )

    return _os_release_cache.copy()


### Command line interface

if __name__ == '__main__':
    # Default is to print the aliased verbose platform string
    terse = ('terse' in sys.argv or '--terse' in sys.argv)
    aliased = (not 'nonaliased' in sys.argv and not '--nonaliased' in sys.argv)
    print(platform(aliased, terse))
    sys.exit(0)
<|endoftext|>"""runpy.py - locating and running Python code using the module namespace

Provides support for locating and running Python scripts using the Python
module namespace instead of the native filesystem.

This allows Python code to play nicely with non-filesystem based PEP 302
importers when locating support scripts as well as when importing modules.
"""
# Written by Nick Coghlan <ncoghlan at gmail.com>
#    to implement PEP 338 (Executing Modules as Scripts)


import sys
import importlib.machinery # importlib first so we can test #15386 via -m
import importlib.util
import io
import types
import os

__all__ = [
    "run_module", "run_path",
]

class _TempModule(object):
    """Temporarily replace a module in sys.modules with an empty namespace"""
    def __init__(self, mod_name):
        self.mod_name = mod_name
        self.module = types.ModuleType(mod_name)
        self._saved_module = []

    def __enter__(self):
        mod_name = self.mod_name
        try:
            self._saved_module.append(sys.modules[mod_name])
        except KeyError:
            pass
        sys.modules[mod_name] = self.module
        return self

    def __exit__(self, *args):
        if self._saved_module:
            sys.modules[self.mod_name] = self._saved_module[0]
        else:
            del sys.modules[self.mod_name]
        self._saved_module = []

class _ModifiedArgv0(object):
    def __init__(self, value):
        self.value = value
        self._saved_value = self._sentinel = object()

    def __enter__(self):
        if self._saved_value is not self._sentinel:
            raise RuntimeError("Already preserving saved value")
        self._saved_value = sys.argv[0]
        sys.argv[0] = self.value

    def __exit__(self, *args):
        self.value = self._sentinel
        sys.argv[0] = self._saved_value

# TODO: Replace these helpers with importlib._bootstrap_external functions.
def _run_code(code, run_globals, init_globals=None,
              mod_name=None, mod_spec=None,
              pkg_name=None, script_name=None):
    """Helper to run code in nominated namespace"""
    if init_globals is not None:
        run_globals.update(init_globals)
    if mod_spec is None:
        loader = None
        fname = script_name
        cached = None
    else:
        loader = mod_spec.loader
        fname = mod_spec.origin
        cached = mod_spec.cached
        if pkg_name is None:
            pkg_name = mod_spec.parent
    run_globals.update(__name__ = mod_name,
                       __file__ = fname,
                       __cached__ = cached,
                       __doc__ = None,
                       __loader__ = loader,
                       __package__ = pkg_name,
                       __spec__ = mod_spec)
    exec(code, run_globals)
    return run_globals

def _run_module_code(code, init_globals=None,
                    mod_name=None, mod_spec=None,
                    pkg_name=None, script_name=None):
    """Helper to run code in new namespace with sys modified"""
    fname = script_name if mod_spec is None else mod_spec.origin
    with _TempModule(mod_name) as temp_module, _ModifiedArgv0(fname):
        mod_globals = temp_module.module.__dict__
        _run_code(code, mod_globals, init_globals,
                  mod_name, mod_spec, pkg_name, script_name)
    # Copy the globals of the temporary module, as they
    # may be cleared when the temporary module goes away
    return mod_globals.copy()

# Helper to get the full name, spec and code for a module
def _get_module_details(mod_name, error=ImportError):
    if mod_name.startswith("."):
        raise error("Relative module names not supported")
    pkg_name, _, _ = mod_name.rpartition(".")
    if pkg_name:
        # Try importing the parent to avoid catching initialization errors
        try:
            __import__(pkg_name)
        except ImportError as e:
            # If the parent or higher ancestor package is missing, let the
            # error be raised by find_spec() below and then be caught. But do
            # not allow other errors to be caught.
            if e.name is None or (e.name != pkg_name and
                    not pkg_name.startswith(e.name + ".")):
                raise
        # Warn if the module has already been imported under its normal name
        existing = sys.modules.get(mod_name)
        if existing is not None and not hasattr(existing, "__path__"):
            from warnings import warn
            msg = "{mod_name!r} found in sys.modules after import of " \
                "package {pkg_name!r}, but prior to execution of " \
                "{mod_name!r}; this may result in unpredictable " \
                "behaviour".format(mod_name=mod_name, pkg_name=pkg_name)
            warn(RuntimeWarning(msg))

    try:
        spec = importlib.util.find_spec(mod_name)
    except (ImportError, AttributeError, TypeError, ValueError) as ex:
        # This hack fixes an impedance mismatch between pkgutil and
        # importlib, where the latter raises other errors for cases where
        # pkgutil previously raised ImportError
        msg = "Error while finding module specification for {!r} ({}: {})"
        if mod_name.endswith(".py"):
            msg += (f". Try using '{mod_name[:-3]}' instead of "
                    f"'{mod_name}' as the module name.")
        raise error(msg.format(mod_name, type(ex).__name__, ex)) from ex
    if spec is None:
        raise error("No module named %s" % mod_name)
    if spec.submodule_search_locations is not None:
        if mod_name == "__main__" or mod_name.endswith(".__main__"):
            raise error("Cannot use package as __main__ module")
        try:
            pkg_main_name = mod_name + ".__main__"
            return _get_module_details(pkg_main_name, error)
        except error as e:
            if mod_name not in sys.modules:
                raise  # No module loaded; being a package is irrelevant
            raise error(("%s; %r is a package and cannot " +
                               "be directly executed") %(e, mod_name))
    loader = spec.loader
    if loader is None:
        raise error("%r is a namespace package and cannot be executed"
                                                                 % mod_name)
    try:
        code = loader.get_code(mod_name)
    except ImportError as e:
        raise error(format(e)) from e
    if code is None:
        raise error("No code object available for %s" % mod_name)
    return mod_name, spec, code

class _Error(Exception):
    """Error that _run_module_as_main() should report without a traceback"""

# XXX ncoghlan: Should this be documented and made public?
# (Current thoughts: don't repeat the mistake that lead to its
# creation when run_module() no longer met the needs of
# mainmodule.c, but couldn't be changed because it was public)
def _run_module_as_main(mod_name, alter_argv=True):
    """Runs the designated module in the __main__ namespace

       Note that the executed module will have full access to the
       __main__ namespace. If this is not desirable, the run_module()
       function should be used to run the module code in a fresh namespace.

       At the very least, these variables in __main__ will be overwritten:
           __name__
           __file__
           __cached__
           __loader__
           __package__
    """
    try:
        if alter_argv or mod_name != "__main__": # i.e. -m switch
            mod_name, mod_spec, code = _get_module_details(mod_name, _Error)
        else:          # i.e. directory or zipfile execution
            mod_name, mod_spec, code = _get_main_module_details(_Error)
    except _Error as exc:
        msg = "%s: %s" % (sys.executable, exc)
        sys.exit(msg)
    main_globals = sys.modules["__main__"].__dict__
    if alter_argv:
        sys.argv[0] = mod_spec.origin
    return _run_code(code, main_globals, None,
                     "__main__", mod_spec)

def run_module(mod_name, init_globals=None,
               run_name=None, alter_sys=False):
    """Execute a module's code without importing it.

       mod_name -- an absolute module name or package name.

       Optional arguments:
       init_globals -- dictionary used to pre-populate the module’s
       globals dictionary before the code is executed.

       run_name -- if not None, this will be used for setting __name__;
       otherwise, __name__ will be set to mod_name + '__main__' if the
       named module is a package and to just mod_name otherwise.

       alter_sys -- if True, sys.argv[0] is updated with the value of
       __file__ and sys.modules[__name__] is updated with a temporary
       module object for the module being executed. Both are
       restored to their original values before the function returns.

       Returns the resulting module globals dictionary.
    """
    mod_name, mod_spec, code = _get_module_details(mod_name)
    if run_name is None:
        run_name = mod_name
    if alter_sys:
        return _run_module_code(code, init_globals, run_name, mod_spec)
    else:
        # Leave the sys module alone
        return _run_code(code, {}, init_globals, run_name, mod_spec)

def _get_main_module_details(error=ImportError):
    # Helper that gives a nicer error message when attempting to
    # execute a zipfile or directory by invoking __main__.py
    # Also moves the standard __main__ out of the way so that the
    # preexisting __loader__ entry doesn't cause issues
    main_name = "__main__"
    saved_main = sys.modules[main_name]
    del sys.modules[main_name]
    try:
        return _get_module_details(main_name)
    except ImportError as exc:
        if main_name in str(exc):
            raise error("can't find %r module in %r" %
                              (main_name, sys.path[0])) from exc
        raise
    finally:
        sys.modules[main_name] = saved_main


def _get_code_from_file(run_name, fname):
    # Check for a compiled file first
    from pkgutil import read_code
    decoded_path = os.path.abspath(os.fsdecode(fname))
    with io.open_code(decoded_path) as f:
        code = read_code(f)
    if code is None:
        # That didn't work, so try it as normal source code
        with io.open_code(decoded_path) as f:
            code = compile(f.read(), fname, 'exec')
    return code, fname

def run_path(path_name, init_globals=None, run_name=None):
    """Execute code located at the specified filesystem location.

       path_name -- filesystem location of a Python script, zipfile,
       or directory containing a top level __main__.py script.

       Optional arguments:
       init_globals -- dictionary used to pre-populate the module’s
       globals dictionary before the code is executed.

       run_name -- if not None, this will be used to set __name__;
       otherwise, '<run_path>' will be used for __name__.

       Returns the resulting module globals dictionary.
    """
    if run_name is None:
        run_name = "<run_path>"
    pkg_name = run_name.rpartition(".")[0]
    from pkgutil import get_importer
    importer = get_importer(path_name)
    # Trying to avoid importing imp so as to not consume the deprecation warning.
    is_NullImporter = False
    if type(importer).__module__ == 'imp':
        if type(importer).__name__ == 'NullImporter':
            is_NullImporter = True
    if isinstance(importer, type(None)) or is_NullImporter:
        # Not a valid sys.path entry, so run the code directly
        # execfile() doesn't help as we want to allow compiled files
        code, fname = _get_code_from_file(run_name, path_name)
        return _run_module_code(code, init_globals, run_name,
                                pkg_name=pkg_name, script_name=fname)
    else:
        # Finder is defined for path, so add it to
        # the start of sys.path
        sys.path.insert(0, path_name)
        try:
            # Here's where things are a little different from the run_module
            # case. There, we only had to replace the module in sys while the
            # code was running and doing so was somewhat optional. Here, we
            # have no choice and we have to remove it even while we read the
            # code. If we don't do this, a __loader__ attribute in the
            # existing __main__ module may prevent location of the new module.
            mod_name, mod_spec, code = _get_main_module_details()
            with _TempModule(run_name) as temp_module, \
                 _ModifiedArgv0(path_name):
                mod_globals = temp_module.module.__dict__
                return _run_code(code, mod_globals, init_globals,
                                    run_name, mod_spec, pkg_name).copy()
        finally:
            try:
                sys.path.remove(path_name)
            except ValueError:
                pass


if __name__ == "__main__":
    # Run the module specified as the next command line argument
    if len(sys.argv) < 2:
        print("No module specified for execution", file=sys.stderr)
    else:
        del sys.argv[0] # Make the requested module sys.argv[0]
        _run_module_as_main(sys.argv[0])
<|endoftext|>#! /usr/local/bin/python

# NOTE: the above "/usr/local/bin/python" is NOT a mistake.  It is
# intentionally NOT "/usr/bin/env python".  On many systems
# (e.g. Solaris), /usr/local/bin is not in $PATH as passed to CGI
# scripts, and /usr/local/bin is the default directory where Python is
# installed, so /usr/bin/env would be unable to find python.  Granted,
# binary installations by Linux vendors often install Python in
# /usr/bin.  So let those vendors patch cgi.py to match their choice
# of installation.

"""Support module for CGI (Common Gateway Interface) scripts.

This module defines a number of utilities for use by CGI scripts
written in Python.
"""

# History
# -------
#
# Michael McLay started this module.  Steve Majewski changed the
# interface to SvFormContentDict and FormContentDict.  The multipart
# parsing was inspired by code submitted by Andreas Paepcke.  Guido van
# Rossum rewrote, reformatted and documented the module and is currently
# responsible for its maintenance.
#

__version__ = "2.6"


# Imports
# =======

from io import StringIO, BytesIO, TextIOWrapper
from collections.abc import Mapping
import sys
import os
import urllib.parse
from email.parser import FeedParser
from email.message import Message
import html
import locale
import tempfile
import warnings

__all__ = ["MiniFieldStorage", "FieldStorage", "parse", "parse_multipart",
           "parse_header", "test", "print_exception", "print_environ",
           "print_form", "print_directory", "print_arguments",
           "print_environ_usage"]

# Logging support
# ===============

logfile = ""            # Filename to log to, if not empty
logfp = None            # File object to log to, if not None

def initlog(*allargs):
    """Write a log message, if there is a log file.

    Even though this function is called initlog(), you should always
    use log(); log is a variable that is set either to initlog
    (initially), to dolog (once the log file has been opened), or to
    nolog (when logging is disabled).

    The first argument is a format string; the remaining arguments (if
    any) are arguments to the % operator, so e.g.
        log("%s: %s", "a", "b")
    will write "a: b" to the log file, followed by a newline.

    If the global logfp is not None, it should be a file object to
    which log data is written.

    If the global logfp is None, the global logfile may be a string
    giving a filename to open, in append mode.  This file should be
    world writable!!!  If the file can't be opened, logging is
    silently disabled (since there is no safe place where we could
    send an error message).

    """
    global log, logfile, logfp
    warnings.warn("cgi.log() is deprecated as of 3.10. Use logging instead",
                  DeprecationWarning, stacklevel=2)
    if logfile and not logfp:
        try:
            logfp = open(logfile, "a", encoding="locale")
        except OSError:
            pass
    if not logfp:
        log = nolog
    else:
        log = dolog
    log(*allargs)

def dolog(fmt, *args):
    """Write a log message to the log file.  See initlog() for docs."""
    logfp.write(fmt%args + "\n")

def nolog(*allargs):
    """Dummy function, assigned to log when logging is disabled."""
    pass

def closelog():
    """Close the log file."""
    global log, logfile, logfp
    logfile = ''
    if logfp:
        logfp.close()
        logfp = None
    log = initlog

log = initlog           # The current logging function


# Parsing functions
# =================

# Maximum input we will accept when REQUEST_METHOD is POST
# 0 ==> unlimited input
maxlen = 0

def parse(fp=None, environ=os.environ, keep_blank_values=0,
          strict_parsing=0, separator='&'):
    """Parse a query in the environment or from a file (default stdin)

        Arguments, all optional:

        fp              : file pointer; default: sys.stdin.buffer

        environ         : environment dictionary; default: os.environ

        keep_blank_values: flag indicating whether blank values in
            percent-encoded forms should be treated as blank strings.
            A true value indicates that blanks should be retained as
            blank strings.  The default false value indicates that
            blank values are to be ignored and treated as if they were
            not included.

