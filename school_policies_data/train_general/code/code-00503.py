                    events |= EVENT_READ

                key = self._key_from_fd(fd)
                if key:
                    ready.append((key, events & key.events))
            return ready

        def close(self):
            self._selector.close()
            super().close()


if hasattr(select, 'devpoll'):

    class DevpollSelector(_PollLikeSelector):
        """Solaris /dev/poll selector."""
        _selector_cls = select.devpoll
        _EVENT_READ = select.POLLIN
        _EVENT_WRITE = select.POLLOUT

        def fileno(self):
            return self._selector.fileno()

        def close(self):
            self._selector.close()
            super().close()


if hasattr(select, 'kqueue'):

    class KqueueSelector(_BaseSelectorImpl):
        """Kqueue-based selector."""

        def __init__(self):
            super().__init__()
            self._selector = select.kqueue()

        def fileno(self):
            return self._selector.fileno()

        def register(self, fileobj, events, data=None):
            key = super().register(fileobj, events, data)
            try:
                if events & EVENT_READ:
                    kev = select.kevent(key.fd, select.KQ_FILTER_READ,
                                        select.KQ_EV_ADD)
                    self._selector.control([kev], 0, 0)
                if events & EVENT_WRITE:
                    kev = select.kevent(key.fd, select.KQ_FILTER_WRITE,
                                        select.KQ_EV_ADD)
                    self._selector.control([kev], 0, 0)
            except:
                super().unregister(fileobj)
                raise
            return key

        def unregister(self, fileobj):
            key = super().unregister(fileobj)
            if key.events & EVENT_READ:
                kev = select.kevent(key.fd, select.KQ_FILTER_READ,
                                    select.KQ_EV_DELETE)
                try:
                    self._selector.control([kev], 0, 0)
                except OSError:
                    # This can happen if the FD was closed since it
                    # was registered.
                    pass
            if key.events & EVENT_WRITE:
                kev = select.kevent(key.fd, select.KQ_FILTER_WRITE,
                                    select.KQ_EV_DELETE)
                try:
                    self._selector.control([kev], 0, 0)
                except OSError:
                    # See comment above.
                    pass
            return key

        def select(self, timeout=None):
            timeout = None if timeout is None else max(timeout, 0)
            # If max_ev is 0, kqueue will ignore the timeout. For consistent
            # behavior with the other selector classes, we prevent that here
            # (using max). See https://bugs.python.org/issue29255
            max_ev = max(len(self._fd_to_key), 1)
            ready = []
            try:
                kev_list = self._selector.control(None, max_ev, timeout)
            except InterruptedError:
                return ready
            for kev in kev_list:
                fd = kev.ident
                flag = kev.filter
                events = 0
                if flag == select.KQ_FILTER_READ:
                    events |= EVENT_READ
                if flag == select.KQ_FILTER_WRITE:
                    events |= EVENT_WRITE

                key = self._key_from_fd(fd)
                if key:
                    ready.append((key, events & key.events))
            return ready

        def close(self):
            self._selector.close()
            super().close()


def _can_use(method):
    """Check if we can use the selector depending upon the
    operating system. """
    # Implementation based upon https://github.com/sethmlarson/selectors2/blob/master/selectors2.py
    selector = getattr(select, method, None)
    if selector is None:
        # select module does not implement method
        return False
    # check if the OS and Kernel actually support the method. Call may fail with
    # OSError: [Errno 38] Function not implemented
    try:
        selector_obj = selector()
        if method == 'poll':
            # check that poll actually works
            selector_obj.poll(0)
        else:
            # close epoll, kqueue, and devpoll fd
            selector_obj.close()
        return True
    except OSError:
        return False


# Choose the best implementation, roughly:
#    epoll|kqueue|devpoll > poll > select.
# select() also can't accept a FD > FD_SETSIZE (usually around 1024)
if _can_use('kqueue'):
    DefaultSelector = KqueueSelector
elif _can_use('epoll'):
    DefaultSelector = EpollSelector
elif _can_use('devpoll'):
    DefaultSelector = DevpollSelector
elif _can_use('poll'):
    DefaultSelector = PollSelector
else:
    DefaultSelector = SelectSelector
<|endoftext|># Module doctest.
# Released to the public domain 16-Jan-2001, by Tim Peters (tim@python.org).
# Major enhancements and refactoring by:
#     Jim Fulton
#     Edward Loper

# Provided as-is; use at your own risk; no warranty; no promises; enjoy!

r"""Module doctest -- a framework for running examples in docstrings.

In simplest use, end each module M to be tested with:

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()

Then running the module as a script will cause the examples in the
docstrings to get executed and verified:

python M.py

This won't display anything unless an example fails, in which case the
failing example(s) and the cause(s) of the failure(s) are printed to stdout
(why not stderr? because stderr is a lame hack <0.2 wink>), and the final
line of output is "Test failed.".

Run it with the -v switch instead:

python M.py -v

and a detailed report of all examples tried is printed to stdout, along
with assorted summaries at the end.

You can force verbose mode by passing "verbose=True" to testmod, or prohibit
it by passing "verbose=False".  In either of those cases, sys.argv is not
examined by testmod.

There are a variety of other ways to run doctests, including integration
with the unittest framework, and support for running non-Python text
files containing doctests.  There are also many ways to override parts
of doctest's default behaviors.  See the Library Reference Manual for
details.
"""

__docformat__ = 'reStructuredText en'

__all__ = [
    # 0, Option Flags
    'register_optionflag',
    'DONT_ACCEPT_TRUE_FOR_1',
    'DONT_ACCEPT_BLANKLINE',
    'NORMALIZE_WHITESPACE',
    'ELLIPSIS',
    'SKIP',
    'IGNORE_EXCEPTION_DETAIL',
    'COMPARISON_FLAGS',
    'REPORT_UDIFF',
    'REPORT_CDIFF',
    'REPORT_NDIFF',
    'REPORT_ONLY_FIRST_FAILURE',
    'REPORTING_FLAGS',
    'FAIL_FAST',
    # 1. Utility Functions
    # 2. Example & DocTest
    'Example',
    'DocTest',
    # 3. Doctest Parser
    'DocTestParser',
    # 4. Doctest Finder
    'DocTestFinder',
    # 5. Doctest Runner
    'DocTestRunner',
    'OutputChecker',
    'DocTestFailure',
    'UnexpectedException',
    'DebugRunner',
    # 6. Test Functions
    'testmod',
    'testfile',
    'run_docstring_examples',
    # 7. Unittest Support
    'DocTestSuite',
    'DocFileSuite',
    'set_unittest_reportflags',
    # 8. Debugging Support
    'script_from_examples',
    'testsource',
    'debug_src',
    'debug',
]

import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple

TestResults = namedtuple('TestResults', 'failed attempted')

# There are 4 basic classes:
#  - Example: a <source, want> pair, plus an intra-docstring line number.
#  - DocTest: a collection of examples, parsed from a docstring, plus
#    info about where the docstring came from (name, filename, lineno).
#  - DocTestFinder: extracts DocTests from a given object's docstring and
#    its contained objects' docstrings.
#  - DocTestRunner: runs DocTest cases, and accumulates statistics.
#
# So the basic picture is:
#
#                             list of:
# +------+                   +---------+                   +-------+
# |object| --DocTestFinder-> | DocTest | --DocTestRunner-> |results|
# +------+                   +---------+                   +-------+
#                            | Example |
#                            |   ...   |
#                            | Example |
#                            +---------+

# Option constants.

OPTIONFLAGS_BY_NAME = {}
def register_optionflag(name):
    # Create a new flag unless `name` is already known.
    return OPTIONFLAGS_BY_NAME.setdefault(name, 1 << len(OPTIONFLAGS_BY_NAME))

DONT_ACCEPT_TRUE_FOR_1 = register_optionflag('DONT_ACCEPT_TRUE_FOR_1')
DONT_ACCEPT_BLANKLINE = register_optionflag('DONT_ACCEPT_BLANKLINE')
NORMALIZE_WHITESPACE = register_optionflag('NORMALIZE_WHITESPACE')
ELLIPSIS = register_optionflag('ELLIPSIS')
SKIP = register_optionflag('SKIP')
IGNORE_EXCEPTION_DETAIL = register_optionflag('IGNORE_EXCEPTION_DETAIL')

COMPARISON_FLAGS = (DONT_ACCEPT_TRUE_FOR_1 |
                    DONT_ACCEPT_BLANKLINE |
                    NORMALIZE_WHITESPACE |
                    ELLIPSIS |
                    SKIP |
                    IGNORE_EXCEPTION_DETAIL)

REPORT_UDIFF = register_optionflag('REPORT_UDIFF')
REPORT_CDIFF = register_optionflag('REPORT_CDIFF')
REPORT_NDIFF = register_optionflag('REPORT_NDIFF')
REPORT_ONLY_FIRST_FAILURE = register_optionflag('REPORT_ONLY_FIRST_FAILURE')
FAIL_FAST = register_optionflag('FAIL_FAST')

REPORTING_FLAGS = (REPORT_UDIFF |
                   REPORT_CDIFF |
                   REPORT_NDIFF |
                   REPORT_ONLY_FIRST_FAILURE |
                   FAIL_FAST)

# Special string markers for use in `want` strings:
BLANKLINE_MARKER = '<BLANKLINE>'
ELLIPSIS_MARKER = '...'

######################################################################
## Table of Contents
######################################################################
#  1. Utility Functions
#  2. Example & DocTest -- store test cases
#  3. DocTest Parser -- extracts examples from strings
#  4. DocTest Finder -- extracts test cases from objects
#  5. DocTest Runner -- runs test cases
#  6. Test Functions -- convenient wrappers for testing
#  7. Unittest Support
#  8. Debugging Support
#  9. Example Usage

######################################################################
## 1. Utility Functions
######################################################################

def _extract_future_flags(globs):
    """
    Return the compiler-flags associated with the future features that
    have been imported into the given namespace (globs).
    """
    flags = 0
    for fname in __future__.all_feature_names:
        feature = globs.get(fname, None)
        if feature is getattr(__future__, fname):
            flags |= feature.compiler_flag
    return flags

def _normalize_module(module, depth=2):
    """
    Return the module specified by `module`.  In particular:
      - If `module` is a module, then return module.
      - If `module` is a string, then import and return the
        module with that name.
      - If `module` is None, then return the calling module.
        The calling module is assumed to be the module of
        the stack frame at the given depth in the call stack.
    """
    if inspect.ismodule(module):
        return module
    elif isinstance(module, str):
        return __import__(module, globals(), locals(), ["*"])
    elif module is None:
        return sys.modules[sys._getframe(depth).f_globals['__name__']]
    else:
        raise TypeError("Expected a module, string, or None")

def _newline_convert(data):
    # The IO module provides a handy decoder for universal newline conversion
    return IncrementalNewlineDecoder(None, True).decode(data, True)

def _load_testfile(filename, package, module_relative, encoding):
    if module_relative:
        package = _normalize_module(package, 3)
        filename = _module_relative_path(package, filename)
        if (loader := getattr(package, '__loader__', None)) is None:
            try:
                loader = package.__spec__.loader
            except AttributeError:
                pass
        if hasattr(loader, 'get_data'):
            file_contents = loader.get_data(filename)
            file_contents = file_contents.decode(encoding)
            # get_data() opens files as 'rb', so one must do the equivalent
            # conversion as universal newlines would do.
            return _newline_convert(file_contents), filename
    with open(filename, encoding=encoding) as f:
        return f.read(), filename

def _indent(s, indent=4):
    """
    Add the given number of space characters to the beginning of
    every non-blank line in `s`, and return the result.
    """
    # This regexp matches the start of non-blank lines:
    return re.sub('(?m)^(?!$)', indent*' ', s)

def _exception_traceback(exc_info):
    """
    Return a string containing a traceback message for the given
    exc_info tuple (as returned by sys.exc_info()).
    """
    # Get a traceback message.
    excout = StringIO()
    exc_type, exc_val, exc_tb = exc_info
    traceback.print_exception(exc_type, exc_val, exc_tb, file=excout)
    return excout.getvalue()

# Override some StringIO methods.
class _SpoofOut(StringIO):
    def getvalue(self):
        result = StringIO.getvalue(self)
        # If anything at all was written, make sure there's a trailing
        # newline.  There's no way for the expected output to indicate
        # that a trailing newline is missing.
        if result and not result.endswith("\n"):
            result += "\n"
        return result

    def truncate(self, size=None):
        self.seek(size)
        StringIO.truncate(self)

# Worst-case linear-time ellipsis matching.
def _ellipsis_match(want, got):
    """
    Essentially the only subtle case:
    >>> _ellipsis_match('aa...aa', 'aaa')
    False
    """
    if ELLIPSIS_MARKER not in want:
        return want == got

    # Find "the real" strings.
    ws = want.split(ELLIPSIS_MARKER)
    assert len(ws) >= 2

    # Deal with exact matches possibly needed at one or both ends.
    startpos, endpos = 0, len(got)
    w = ws[0]
    if w:   # starts with exact match
        if got.startswith(w):
            startpos = len(w)
            del ws[0]
        else:
            return False
    w = ws[-1]
    if w:   # ends with exact match
        if got.endswith(w):
            endpos -= len(w)
            del ws[-1]
        else:
            return False

    if startpos > endpos:
        # Exact end matches required more characters than we have, as in
        # _ellipsis_match('aa...aa', 'aaa')
        return False

    # For the rest, we only need to find the leftmost non-overlapping
    # match for each piece.  If there's no overall match that way alone,
    # there's no overall match period.
    for w in ws:
        # w may be '' at times, if there are consecutive ellipses, or
        # due to an ellipsis at the start or end of `want`.  That's OK.
        # Search for an empty string succeeds, and doesn't change startpos.
        startpos = got.find(w, startpos, endpos)
        if startpos < 0:
            return False
        startpos += len(w)

    return True

def _comment_line(line):
    "Return a commented form of the given line"
    line = line.rstrip()
    if line:
        return '# '+line
    else:
        return '#'

def _strip_exception_details(msg):
    # Support for IGNORE_EXCEPTION_DETAIL.
    # Get rid of everything except the exception name; in particular, drop
    # the possibly dotted module path (if any) and the exception message (if
    # any).  We assume that a colon is never part of a dotted name, or of an
    # exception name.
    # E.g., given
    #    "foo.bar.MyError: la di da"
    # return "MyError"
    # Or for "abc.def" or "abc.def:\n" return "def".

    start, end = 0, len(msg)
    # The exception name must appear on the first line.
    i = msg.find("\n")
    if i >= 0:
        end = i
    # retain up to the first colon (if any)
    i = msg.find(':', 0, end)
    if i >= 0:
        end = i
    # retain just the exception name
    i = msg.rfind('.', 0, end)
    if i >= 0:
        start = i+1
    return msg[start: end]

class _OutputRedirectingPdb(pdb.Pdb):
    """
    A specialized version of the python debugger that redirects stdout
    to a given stream when interacting with the user.  Stdout is *not*
    redirected when traced code is executed.
    """
    def __init__(self, out):
        self.__out = out
        self.__debugger_used = False
        # do not play signal games in the pdb
        pdb.Pdb.__init__(self, stdout=out, nosigint=True)
        # still use input() to get user input
        self.use_rawinput = 1

    def set_trace(self, frame=None):
        self.__debugger_used = True
        if frame is None:
            frame = sys._getframe().f_back
        pdb.Pdb.set_trace(self, frame)

    def set_continue(self):
        # Calling set_continue unconditionally would break unit test
        # coverage reporting, as Bdb.set_continue calls sys.settrace(None).
        if self.__debugger_used:
            pdb.Pdb.set_continue(self)

    def trace_dispatch(self, *args):
        # Redirect stdout to the given stream.
        save_stdout = sys.stdout
        sys.stdout = self.__out
        # Call Pdb's trace dispatch method.
        try:
            return pdb.Pdb.trace_dispatch(self, *args)
        finally:
            sys.stdout = save_stdout

# [XX] Normalize with respect to os.path.pardir?
def _module_relative_path(module, test_path):
    if not inspect.ismodule(module):
        raise TypeError('Expected a module: %r' % module)
    if test_path.startswith('/'):
        raise ValueError('Module-relative files may not have absolute paths')

    # Normalize the path. On Windows, replace "/" with "\".
    test_path = os.path.join(*(test_path.split('/')))

    # Find the base directory for the path.
    if hasattr(module, '__file__'):
        # A normal module/package
        basedir = os.path.split(module.__file__)[0]
    elif module.__name__ == '__main__':
        # An interactive session.
        if len(sys.argv)>0 and sys.argv[0] != '':
            basedir = os.path.split(sys.argv[0])[0]
        else:
            basedir = os.curdir
    else:
        if hasattr(module, '__path__'):
            for directory in module.__path__:
                fullpath = os.path.join(directory, test_path)
                if os.path.exists(fullpath):
                    return fullpath

        # A module w/o __file__ (this includes builtins)
        raise ValueError("Can't resolve paths relative to the module "
                         "%r (it has no __file__)"
                         % module.__name__)

    # Combine the base directory and the test path.
    return os.path.join(basedir, test_path)

######################################################################
## 2. Example & DocTest
######################################################################
## - An "example" is a <source, want> pair, where "source" is a
##   fragment of source code, and "want" is the expected output for
##   "source."  The Example class also includes information about
##   where the example was extracted from.
##
## - A "doctest" is a collection of examples, typically extracted from
##   a string (such as an object's docstring).  The DocTest class also
##   includes information about where the string was extracted from.

class Example:
    """
    A single doctest example, consisting of source code and expected
    output.  `Example` defines the following attributes:

      - source: A single Python statement, always ending with a newline.
        The constructor adds a newline if needed.

      - want: The expected output from running the source code (either
        from stdout, or a traceback in case of exception).  `want` ends
        with a newline unless it's empty, in which case it's an empty
        string.  The constructor adds a newline if needed.

      - exc_msg: The exception message generated by the example, if
        the example is expected to generate an exception; or `None` if
        it is not expected to generate an exception.  This exception
        message is compared against the return value of
        `traceback.format_exception_only()`.  `exc_msg` ends with a
        newline unless it's `None`.  The constructor adds a newline
        if needed.

      - lineno: The line number within the DocTest string containing
        this Example where the Example begins.  This line number is
        zero-based, with respect to the beginning of the DocTest.

      - indent: The example's indentation in the DocTest string.
        I.e., the number of space characters that precede the
        example's first prompt.

      - options: A dictionary mapping from option flags to True or
        False, which is used to override default options for this
        example.  Any option flags not contained in this dictionary
        are left at their default value (as specified by the
        DocTestRunner's optionflags).  By default, no options are set.
    """
    def __init__(self, source, want, exc_msg=None, lineno=0, indent=0,
                 options=None):
        # Normalize inputs.
        if not source.endswith('\n'):
            source += '\n'
        if want and not want.endswith('\n'):
            want += '\n'
        if exc_msg is not None and not exc_msg.endswith('\n'):
            exc_msg += '\n'
        # Store properties.
        self.source = source
        self.want = want
        self.lineno = lineno
        self.indent = indent
        if options is None: options = {}
        self.options = options
        self.exc_msg = exc_msg

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented

        return self.source == other.source and \
               self.want == other.want and \
               self.lineno == other.lineno and \
               self.indent == other.indent and \
               self.options == other.options and \
               self.exc_msg == other.exc_msg

    def __hash__(self):
        return hash((self.source, self.want, self.lineno, self.indent,
                     self.exc_msg))

class DocTest:
    """
    A collection of doctest examples that should be run in a single
    namespace.  Each `DocTest` defines the following attributes:

      - examples: the list of examples.

      - globs: The namespace (aka globals) that the examples should
        be run in.

      - name: A name identifying the DocTest (typically, the name of
        the object whose docstring this DocTest was extracted from).

      - filename: The name of the file that this DocTest was extracted
        from, or `None` if the filename is unknown.

      - lineno: The line number within filename where this DocTest
        begins, or `None` if the line number is unavailable.  This
        line number is zero-based, with respect to the beginning of
        the file.

      - docstring: The string that the examples were extracted from,
        or `None` if the string is unavailable.
    """
    def __init__(self, examples, globs, name, filename, lineno, docstring):
        """
        Create a new DocTest containing the given examples.  The
        DocTest's globals are initialized with a copy of `globs`.
        """
        assert not isinstance(examples, str), \
               "DocTest no longer accepts str; use DocTestParser instead"
        self.examples = examples
        self.docstring = docstring
        self.globs = globs.copy()
        self.name = name
        self.filename = filename
        self.lineno = lineno

    def __repr__(self):
        if len(self.examples) == 0:
            examples = 'no examples'
        elif len(self.examples) == 1:
            examples = '1 example'
        else:
            examples = '%d examples' % len(self.examples)
        return ('<%s %s from %s:%s (%s)>' %
                (self.__class__.__name__,
                 self.name, self.filename, self.lineno, examples))

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented

        return self.examples == other.examples and \
               self.docstring == other.docstring and \
               self.globs == other.globs and \
               self.name == other.name and \
               self.filename == other.filename and \
               self.lineno == other.lineno

    def __hash__(self):
        return hash((self.docstring, self.name, self.filename, self.lineno))

    # This lets us sort tests by name:
    def __lt__(self, other):
        if not isinstance(other, DocTest):
            return NotImplemented
        return ((self.name, self.filename, self.lineno, id(self))
                <
                (other.name, other.filename, other.lineno, id(other)))

######################################################################
## 3. DocTestParser
######################################################################

class DocTestParser:
    """
    A class used to parse strings containing doctest examples.
    """
    # This regular expression is used to find doctest examples in a
    # string.  It defines three groups: `source` is the source code
    # (including leading indentation and prompts); `indent` is the
    # indentation of the first (PS1) line of the source code; and
    # `want` is the expected output (including leading indentation).
    _EXAMPLE_RE = re.compile(r'''
        # Source consists of a PS1 line followed by zero or more PS2 lines.
        (?P<source>
            (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line
            (?:\n           [ ]*  \.\.\. .*)*)  # PS2 lines
        \n?
        # Want consists of any non-blank lines that do not start with PS1.
        (?P<want> (?:(?![ ]*$)    # Not a blank line
                     (?![ ]*>>>)  # Not a line starting with PS1
                     .+$\n?       # But any other line
                  )*)
        ''', re.MULTILINE | re.VERBOSE)

    # A regular expression for handling `want` strings that contain
    # expected exceptions.  It divides `want` into three pieces:
    #    - the traceback header line (`hdr`)
    #    - the traceback stack (`stack`)
    #    - the exception message (`msg`), as generated by
    #      traceback.format_exception_only()
    # `msg` may have multiple lines.  We assume/require that the
    # exception message is the first non-indented line starting with a word
    # character following the traceback header line.
    _EXCEPTION_RE = re.compile(r"""
        # Grab the traceback header.  Different versions of Python have
        # said different things on the first traceback line.
        ^(?P<hdr> Traceback\ \(
            (?: most\ recent\ call\ last
            |   innermost\ last
            ) \) :
        )
        \s* $                # toss trailing whitespace on the header.
        (?P<stack> .*?)      # don't blink: absorb stuff until...
        ^ (?P<msg> \w+ .*)   #     a line *starts* with alphanum.
        """, re.VERBOSE | re.MULTILINE | re.DOTALL)

    # A callable returning a true value iff its argument is a blank line
    # or contains a single comment.
    _IS_BLANK_OR_COMMENT = re.compile(r'^[ ]*(#.*)?$').match

    def parse(self, string, name='<string>'):
        """
        Divide the given string into examples and intervening text,
        and return them as a list of alternating Examples and strings.
        Line numbers for the Examples are 0-based.  The optional
        argument `name` is a name identifying this string, and is only
        used for error messages.
        """
        string = string.expandtabs()
        # If all lines begin with the same indentation, then strip it.
        min_indent = self._min_indent(string)
        if min_indent > 0:
            string = '\n'.join([l[min_indent:] for l in string.split('\n')])

        output = []
        charno, lineno = 0, 0
        # Find all doctest examples in the string:
        for m in self._EXAMPLE_RE.finditer(string):
            # Add the pre-example text to `output`.
            output.append(string[charno:m.start()])
            # Update lineno (lines before this example)
            lineno += string.count('\n', charno, m.start())
            # Extract info from the regexp match.
            (source, options, want, exc_msg) = \
                     self._parse_example(m, name, lineno)
            # Create an Example, and add it to the list.
            if not self._IS_BLANK_OR_COMMENT(source):
                output.append( Example(source, want, exc_msg,
                                    lineno=lineno,
                                    indent=min_indent+len(m.group('indent')),
                                    options=options) )
            # Update lineno (lines inside this example)
            lineno += string.count('\n', m.start(), m.end())
            # Update charno.
            charno = m.end()
        # Add any remaining post-example text to `output`.
        output.append(string[charno:])
        return output

    def get_doctest(self, string, globs, name, filename, lineno):
        """
        Extract all doctest examples from the given string, and
        collect them into a `DocTest` object.

        `globs`, `name`, `filename`, and `lineno` are attributes for
        the new `DocTest` object.  See the documentation for `DocTest`
        for more information.
        """
        return DocTest(self.get_examples(string, name), globs,
                       name, filename, lineno, string)

    def get_examples(self, string, name='<string>'):
        """
        Extract all doctest examples from the given string, and return
        them as a list of `Example` objects.  Line numbers are
        0-based, because it's most common in doctests that nothing
        interesting appears on the same line as opening triple-quote,
        and so the first interesting line is called \"line 1\" then.

        The optional argument `name` is a name identifying this
        string, and is only used for error messages.
        """
        return [x for x in self.parse(string, name)
                if isinstance(x, Example)]

    def _parse_example(self, m, name, lineno):
        """
        Given a regular expression match from `_EXAMPLE_RE` (`m`),
        return a pair `(source, want)`, where `source` is the matched
        example's source code (with prompts and indentation stripped);
        and `want` is the example's expected output (with indentation
        stripped).

        `name` is the string's name, and `lineno` is the line number
        where the example starts; both are used for error messages.
        """
        # Get the example's indentation level.
        indent = len(m.group('indent'))

        # Divide source into lines; check that they're properly
        # indented; and then strip their indentation & prompts.
        source_lines = m.group('source').split('\n')
        self._check_prompt_blank(source_lines, indent, name, lineno)
        self._check_prefix(source_lines[1:], ' '*indent + '.', name, lineno)
        source = '\n'.join([sl[indent+4:] for sl in source_lines])

        # Divide want into lines; check that it's properly indented; and
        # then strip the indentation.  Spaces before the last newline should
        # be preserved, so plain rstrip() isn't good enough.
        want = m.group('want')
        want_lines = want.split('\n')
        if len(want_lines) > 1 and re.match(r' *$', want_lines[-1]):
            del want_lines[-1]  # forget final newline & spaces after it
        self._check_prefix(want_lines, ' '*indent, name,
                           lineno + len(source_lines))
        want = '\n'.join([wl[indent:] for wl in want_lines])

        # If `want` contains a traceback message, then extract it.
        m = self._EXCEPTION_RE.match(want)
        if m:
            exc_msg = m.group('msg')
        else:
            exc_msg = None

        # Extract options from the source.
        options = self._find_options(source, name, lineno)

        return source, options, want, exc_msg

    # This regular expression looks for option directives in the
    # source code of an example.  Option directives are comments
    # starting with "doctest:".  Warning: this may give false
    # positives for string-literals that contain the string
    # "#doctest:".  Eliminating these false positives would require
    # actually parsing the string; but we limit them by ignoring any
    # line containing "#doctest:" that is *followed* by a quote mark.
    _OPTION_DIRECTIVE_RE = re.compile(r'#\s*doctest:\s*([^\n\'"]*)$',
                                      re.MULTILINE)

    def _find_options(self, source, name, lineno):
        """
        Return a dictionary containing option overrides extracted from
        option directives in the given source string.

        `name` is the string's name, and `lineno` is the line number
        where the example starts; both are used for error messages.
        """
        options = {}
        # (note: with the current regexp, this will match at most once:)
        for m in self._OPTION_DIRECTIVE_RE.finditer(source):
            option_strings = m.group(1).replace(',', ' ').split()
            for option in option_strings:
                if (option[0] not in '+-' or
                    option[1:] not in OPTIONFLAGS_BY_NAME):
                    raise ValueError('line %r of the doctest for %s '
                                     'has an invalid option: %r' %
                                     (lineno+1, name, option))
                flag = OPTIONFLAGS_BY_NAME[option[1:]]
                options[flag] = (option[0] == '+')
        if options and self._IS_BLANK_OR_COMMENT(source):
            raise ValueError('line %r of the doctest for %s has an option '
                             'directive on a line with no example: %r' %
                             (lineno, name, source))
        return options

    # This regular expression finds the indentation of every non-blank
    # line in a string.
    _INDENT_RE = re.compile(r'^([ ]*)(?=\S)', re.MULTILINE)

    def _min_indent(self, s):
        "Return the minimum indentation of any non-blank line in `s`"
        indents = [len(indent) for indent in self._INDENT_RE.findall(s)]
        if len(indents) > 0:
            return min(indents)
        else:
            return 0

    def _check_prompt_blank(self, lines, indent, name, lineno):
        """
        Given the lines of a source string (including prompts and
        leading indentation), check to make sure that every prompt is
        followed by a space character.  If any line is not followed by
        a space character, then raise ValueError.
        """
        for i, line in enumerate(lines):
            if len(line) >= indent+4 and line[indent+3] != ' ':
                raise ValueError('line %r of the docstring for %s '
                                 'lacks blank after %s: %r' %
                                 (lineno+i+1, name,
                                  line[indent:indent+3], line))

    def _check_prefix(self, lines, prefix, name, lineno):
        """
        Check that every line in the given list starts with the given
        prefix; if any line does not, then raise a ValueError.
        """
        for i, line in enumerate(lines):
            if line and not line.startswith(prefix):
                raise ValueError('line %r of the docstring for %s has '
                                 'inconsistent leading whitespace: %r' %
                                 (lineno+i+1, name, line))


######################################################################
## 4. DocTest Finder
######################################################################

class DocTestFinder:
    """
    A class used to extract the DocTests that are relevant to a given
    object, from its docstring and the docstrings of its contained
    objects.  Doctests can currently be extracted from the following
    object types: modules, functions, classes, methods, staticmethods,
    classmethods, and properties.
    """

    def __init__(self, verbose=False, parser=DocTestParser(),
                 recurse=True, exclude_empty=True):
        """
        Create a new doctest finder.

        The optional argument `parser` specifies a class or
        function that should be used to create new DocTest objects (or
        objects that implement the same interface as DocTest).  The
        signature for this factory function should match the signature
        of the DocTest constructor.

        If the optional argument `recurse` is false, then `find` will
        only examine the given object, and not any contained objects.

        If the optional argument `exclude_empty` is false, then `find`
        will include tests for objects with empty docstrings.
        """
        self._parser = parser
        self._verbose = verbose
        self._recurse = recurse
        self._exclude_empty = exclude_empty

    def find(self, obj, name=None, module=None, globs=None, extraglobs=None):
        """
        Return a list of the DocTests that are defined by the given
        object's docstring, or by any of its contained objects'
        docstrings.

        The optional parameter `module` is the module that contains
        the given object.  If the module is not specified or is None, then
        the test finder will attempt to automatically determine the
        correct module.  The object's module is used:

            - As a default namespace, if `globs` is not specified.
            - To prevent the DocTestFinder from extracting DocTests
              from objects that are imported from other modules.
            - To find the name of the file containing the object.
            - To help find the line number of the object within its
              file.

        Contained objects whose module does not match `module` are ignored.

        If `module` is False, no attempt to find the module will be made.
        This is obscure, of use mostly in tests:  if `module` is False, or
        is None but cannot be found automatically, then all objects are
        considered to belong to the (non-existent) module, so all contained
        objects will (recursively) be searched for doctests.

        The globals for each DocTest is formed by combining `globs`
        and `extraglobs` (bindings in `extraglobs` override bindings
        in `globs`).  A new copy of the globals dictionary is created
        for each DocTest.  If `globs` is not specified, then it
        defaults to the module's `__dict__`, if specified, or {}
        otherwise.  If `extraglobs` is not specified, then it defaults
        to {}.

        """
        # If name was not specified, then extract it from the object.
        if name is None:
            name = getattr(obj, '__name__', None)
            if name is None:
                raise ValueError("DocTestFinder.find: name must be given "
                        "when obj.__name__ doesn't exist: %r" %
                                 (type(obj),))

        # Find the module that contains the given object (if obj is
        # a module, then module=obj.).  Note: this may fail, in which
        # case module will be None.
        if module is False:
            module = None
        elif module is None:
            module = inspect.getmodule(obj)

        # Read the module's source code.  This is used by
        # DocTestFinder._find_lineno to find the line number for a
        # given object's docstring.
        try:
            file = inspect.getsourcefile(obj)
        except TypeError:
            source_lines = None
        else:
            if not file:
                # Check to see if it's one of our special internal "files"
                # (see __patched_linecache_getlines).
                file = inspect.getfile(obj)
                if not file[0]+file[-2:] == '<]>': file = None
            if file is None:
                source_lines = None
            else:
                if module is not None:
                    # Supply the module globals in case the module was
                    # originally loaded via a PEP 302 loader and
                    # file is not a valid filesystem path
                    source_lines = linecache.getlines(file, module.__dict__)
                else:
                    # No access to a loader, so assume it's a normal
                    # filesystem path
                    source_lines = linecache.getlines(file)
                if not source_lines:
                    source_lines = None

        # Initialize globals, and merge in extraglobs.
        if globs is None:
            if module is None:
                globs = {}
            else:
                globs = module.__dict__.copy()
        else:
            globs = globs.copy()
        if extraglobs is not None:
            globs.update(extraglobs)
        if '__name__' not in globs:
            globs['__name__'] = '__main__'  # provide a default module name

        # Recursively explore `obj`, extracting DocTests.
        tests = []
        self._find(tests, obj, name, module, source_lines, globs, {})
        # Sort the tests by alpha order of names, for consistency in
        # verbose-mode output.  This was a feature of doctest in Pythons
        # <= 2.3 that got lost by accident in 2.4.  It was repaired in
        # 2.4.4 and 2.5.
        tests.sort()
        return tests

    def _from_module(self, module, object):
        """
        Return true if the given object is defined in the given
        module.
        """
        if module is None:
            return True
        elif inspect.getmodule(object) is not None:
            return module is inspect.getmodule(object)
        elif inspect.isfunction(object):
            return module.__dict__ is object.__globals__
        elif inspect.ismethoddescriptor(object):
            if hasattr(object, '__objclass__'):
                obj_mod = object.__objclass__.__module__
            elif hasattr(object, '__module__'):
                obj_mod = object.__module__
            else:
                return True # [XX] no easy way to tell otherwise
            return module.__name__ == obj_mod
        elif inspect.isclass(object):
            return module.__name__ == object.__module__
        elif hasattr(object, '__module__'):
            return module.__name__ == object.__module__
        elif isinstance(object, property):
            return True # [XX] no way not be sure.
        else:
            raise ValueError("object must be a class or function")

    def _is_routine(self, obj):
        """
        Safely unwrap objects and determine if they are functions.
        """
        maybe_routine = obj
        try:
            maybe_routine = inspect.unwrap(maybe_routine)
        except ValueError:
            pass
        return inspect.isroutine(maybe_routine)

    def _find(self, tests, obj, name, module, source_lines, globs, seen):
        """
        Find tests for the given object and any contained objects, and
        add them to `tests`.
        """
        if self._verbose:
            print('Finding tests in %s' % name)

        # If we've already processed this object, then ignore it.
        if id(obj) in seen:
            return
        seen[id(obj)] = 1

        # Find a test for this object, and add it to the list of tests.
        test = self._get_test(obj, name, module, globs, source_lines)
        if test is not None:
            tests.append(test)

        # Look for tests in a module's contained objects.
        if inspect.ismodule(obj) and self._recurse:
            for valname, val in obj.__dict__.items():
                valname = '%s.%s' % (name, valname)

                # Recurse to functions & classes.
                if ((self._is_routine(val) or inspect.isclass(val)) and
                    self._from_module(module, val)):
                    self._find(tests, val, valname, module, source_lines,
                               globs, seen)

        # Look for tests in a module's __test__ dictionary.
        if inspect.ismodule(obj) and self._recurse:
            for valname, val in getattr(obj, '__test__', {}).items():
                if not isinstance(valname, str):
                    raise ValueError("DocTestFinder.find: __test__ keys "
                                     "must be strings: %r" %
                                     (type(valname),))
                if not (inspect.isroutine(val) or inspect.isclass(val) or
                        inspect.ismodule(val) or isinstance(val, str)):
                    raise ValueError("DocTestFinder.find: __test__ values "
                                     "must be strings, functions, methods, "
                                     "classes, or modules: %r" %
                                     (type(val),))
                valname = '%s.__test__.%s' % (name, valname)
                self._find(tests, val, valname, module, source_lines,
                           globs, seen)

        # Look for tests in a class's contained objects.
        if inspect.isclass(obj) and self._recurse:
            for valname, val in obj.__dict__.items():
                # Special handling for staticmethod/classmethod.
                if isinstance(val, (staticmethod, classmethod)):
                    val = val.__func__

                # Recurse to methods, properties, and nested classes.
                if ((inspect.isroutine(val) or inspect.isclass(val) or
                      isinstance(val, property)) and
                      self._from_module(module, val)):
                    valname = '%s.%s' % (name, valname)
                    self._find(tests, val, valname, module, source_lines,
                               globs, seen)

    def _get_test(self, obj, name, module, globs, source_lines):
        """
        Return a DocTest for the given object, if it defines a docstring;
        otherwise, return None.
        """
        # Extract the object's docstring.  If it doesn't have one,
        # then return None (no test for this object).
        if isinstance(obj, str):
            docstring = obj
        else:
            try:
                if obj.__doc__ is None:
                    docstring = ''
                else:
                    docstring = obj.__doc__
                    if not isinstance(docstring, str):
                        docstring = str(docstring)
            except (TypeError, AttributeError):
                docstring = ''

        # Find the docstring's location in the file.
        lineno = self._find_lineno(obj, source_lines)

        # Don't bother if the docstring is empty.
        if self._exclude_empty and not docstring:
            return None

        # Return a DocTest for this object.
        if module is None:
            filename = None
        else:
            # __file__ can be None for namespace packages.
            filename = getattr(module, '__file__', None) or module.__name__
            if filename[-4:] == ".pyc":
                filename = filename[:-1]
        return self._parser.get_doctest(docstring, globs, name,
                                        filename, lineno)

    def _find_lineno(self, obj, source_lines):
        """
        Return a line number of the given object's docstring.

        Returns `None` if the given object does not have a docstring.
        """
        lineno = None
        docstring = getattr(obj, '__doc__', None)

        # Find the line number for modules.
        if inspect.ismodule(obj) and docstring is not None:
            lineno = 0

        # Find the line number for classes.
        # Note: this could be fooled if a class is defined multiple
        # times in a single file.
        if inspect.isclass(obj) and docstring is not None:
            if source_lines is None:
                return None
            pat = re.compile(r'^\s*class\s*%s\b' %
                             getattr(obj, '__name__', '-'))
            for i, line in enumerate(source_lines):
                if pat.match(line):
                    lineno = i
                    break

        # Find the line number for functions & methods.
        if inspect.ismethod(obj): obj = obj.__func__
        if inspect.isfunction(obj) and getattr(obj, '__doc__', None):
            # We don't use `docstring` var here, because `obj` can be changed.
            obj = obj.__code__
        if inspect.istraceback(obj): obj = obj.tb_frame
        if inspect.isframe(obj): obj = obj.f_code
        if inspect.iscode(obj):
            lineno = getattr(obj, 'co_firstlineno', None)-1

        # Find the line number where the docstring starts.  Assume
        # that it's the first line that begins with a quote mark.
        # Note: this could be fooled by a multiline function
        # signature, where a continuation line begins with a quote
        # mark.
        if lineno is not None:
            if source_lines is None:
                return lineno+1
            pat = re.compile(r'(^|.*:)\s*\w*("|\')')
            for lineno in range(lineno, len(source_lines)):
                if pat.match(source_lines[lineno]):
                    return lineno

        # We couldn't find the line number.
        return None

######################################################################
## 5. DocTest Runner
######################################################################

class DocTestRunner:
    """
    A class used to run DocTest test cases, and accumulate statistics.
    The `run` method is used to process a single DocTest case.  It
    returns a tuple `(f, t)`, where `t` is the number of test cases
    tried, and `f` is the number of test cases that failed.

        >>> tests = DocTestFinder().find(_TestClass)
        >>> runner = DocTestRunner(verbose=False)
        >>> tests.sort(key = lambda test: test.name)
        >>> for test in tests:
        ...     print(test.name, '->', runner.run(test))
        _TestClass -> TestResults(failed=0, attempted=2)
        _TestClass.__init__ -> TestResults(failed=0, attempted=2)
        _TestClass.get -> TestResults(failed=0, attempted=2)
        _TestClass.square -> TestResults(failed=0, attempted=1)

    The `summarize` method prints a summary of all the test cases that
    have been run by the runner, and returns an aggregated `(f, t)`
    tuple:

        >>> runner.summarize(verbose=1)
        4 items passed all tests:
           2 tests in _TestClass
           2 tests in _TestClass.__init__
           2 tests in _TestClass.get
           1 tests in _TestClass.square
        7 tests in 4 items.
        7 passed and 0 failed.
        Test passed.
        TestResults(failed=0, attempted=7)

    The aggregated number of tried examples and failed examples is
    also available via the `tries` and `failures` attributes:

        >>> runner.tries
        7
        >>> runner.failures
        0

    The comparison between expected outputs and actual outputs is done
    by an `OutputChecker`.  This comparison may be customized with a
    number of option flags; see the documentation for `testmod` for
    more information.  If the option flags are insufficient, then the
    comparison may also be customized by passing a subclass of
    `OutputChecker` to the constructor.

    The test runner's display output can be controlled in two ways.
    First, an output function (`out) can be passed to
    `TestRunner.run`; this function will be called with strings that
    should be displayed.  It defaults to `sys.stdout.write`.  If
    capturing the output is not sufficient, then the display output
    can be also customized by subclassing DocTestRunner, and
    overriding the methods `report_start`, `report_success`,
    `report_unexpected_exception`, and `report_failure`.
    """
    # This divider string is used to separate failure messages, and to
    # separate sections of the summary.
    DIVIDER = "*" * 70

    def __init__(self, checker=None, verbose=None, optionflags=0):
        """
        Create a new test runner.

        Optional keyword arg `checker` is the `OutputChecker` that
        should be used to compare the expected outputs and actual
        outputs of doctest examples.

        Optional keyword arg 'verbose' prints lots of stuff if true,
        only failures if false; by default, it's true iff '-v' is in
        sys.argv.

        Optional argument `optionflags` can be used to control how the
        test runner compares expected output to actual output, and how
        it displays failures.  See the documentation for `testmod` for
        more information.
        """
        self._checker = checker or OutputChecker()
        if verbose is None:
            verbose = '-v' in sys.argv
        self._verbose = verbose
        self.optionflags = optionflags
        self.original_optionflags = optionflags

        # Keep track of the examples we've run.
        self.tries = 0
        self.failures = 0
        self._name2ft = {}

        # Create a fake output target for capturing doctest output.
        self._fakeout = _SpoofOut()

    #/////////////////////////////////////////////////////////////////
    # Reporting methods
    #/////////////////////////////////////////////////////////////////

    def report_start(self, out, test, example):
        """
        Report that the test runner is about to process the given
        example.  (Only displays a message if verbose=True)
        """
        if self._verbose:
            if example.want:
                out('Trying:\n' + _indent(example.source) +
                    'Expecting:\n' + _indent(example.want))
            else:
                out('Trying:\n' + _indent(example.source) +
                    'Expecting nothing\n')

    def report_success(self, out, test, example, got):
        """
        Report that the given example ran successfully.  (Only
        displays a message if verbose=True)
        """
        if self._verbose:
            out("ok\n")

    def report_failure(self, out, test, example, got):
        """
        Report that the given example failed.
        """
        out(self._failure_header(test, example) +
            self._checker.output_difference(example, got, self.optionflags))

    def report_unexpected_exception(self, out, test, example, exc_info):
        """
        Report that the given example raised an unexpected exception.
        """
        out(self._failure_header(test, example) +
            'Exception raised:\n' + _indent(_exception_traceback(exc_info)))

    def _failure_header(self, test, example):
        out = [self.DIVIDER]
        if test.filename:
            if test.lineno is not None and example.lineno is not None:
                lineno = test.lineno + example.lineno + 1
            else:
                lineno = '?'
            out.append('File "%s", line %s, in %s' %
                       (test.filename, lineno, test.name))
        else:
            out.append('Line %s, in %s' % (example.lineno+1, test.name))
        out.append('Failed example:')
        source = example.source
        out.append(_indent(source))
        return '\n'.join(out)

    #/////////////////////////////////////////////////////////////////
    # DocTest Running
    #/////////////////////////////////////////////////////////////////

    def __run(self, test, compileflags, out):
        """
        Run the examples in `test`.  Write the outcome of each example
        with one of the `DocTestRunner.report_*` methods, using the
        writer function `out`.  `compileflags` is the set of compiler
        flags that should be used to execute examples.  Return a tuple
        `(f, t)`, where `t` is the number of examples tried, and `f`
        is the number of examples that failed.  The examples are run
        in the namespace `test.globs`.
        """
        # Keep track of the number of failures and tries.
        failures = tries = 0

        # Save the option flags (since option directives can be used
        # to modify them).
        original_optionflags = self.optionflags

        SUCCESS, FAILURE, BOOM = range(3) # `outcome` state

        check = self._checker.check_output

        # Process each example.
        for examplenum, example in enumerate(test.examples):

            # If REPORT_ONLY_FIRST_FAILURE is set, then suppress
            # reporting after the first failure.
            quiet = (self.optionflags & REPORT_ONLY_FIRST_FAILURE and
                     failures > 0)

            # Merge in the example's options.
            self.optionflags = original_optionflags
            if example.options:
                for (optionflag, val) in example.options.items():
                    if val:
                        self.optionflags |= optionflag
                    else:
                        self.optionflags &= ~optionflag

            # If 'SKIP' is set, then skip this example.
            if self.optionflags & SKIP:
                continue

            # Record that we started this example.
            tries += 1
            if not quiet:
                self.report_start(out, test, example)

            # Use a special filename for compile(), so we can retrieve
            # the source code during interactive debugging (see
            # __patched_linecache_getlines).
            filename = '<doctest %s[%d]>' % (test.name, examplenum)

            # Run the example in the given context (globs), and record
            # any exception that gets raised.  (But don't intercept
            # keyboard interrupts.)
            try:
                # Don't blink!  This is where the user's code gets run.
                exec(compile(example.source, filename, "single",
                             compileflags, True), test.globs)
                self.debugger.set_continue() # ==== Example Finished ====
                exception = None
            except KeyboardInterrupt:
                raise
            except:
                exception = sys.exc_info()
                self.debugger.set_continue() # ==== Example Finished ====

            got = self._fakeout.getvalue()  # the actual output
            self._fakeout.truncate(0)
            outcome = FAILURE   # guilty until proved innocent or insane

            # If the example executed without raising any exceptions,
            # verify its output.
            if exception is None:
                if check(example.want, got, self.optionflags):
                    outcome = SUCCESS

            # The example raised an exception:  check if it was expected.
            else:
                exc_msg = traceback.format_exception_only(*exception[:2])[-1]
                if not quiet:
                    got += _exception_traceback(exception)

                # If `example.exc_msg` is None, then we weren't expecting
                # an exception.
                if example.exc_msg is None:
                    outcome = BOOM

                # We expected an exception:  see whether it matches.
                elif check(example.exc_msg, exc_msg, self.optionflags):
                    outcome = SUCCESS

                # Another chance if they didn't care about the detail.
                elif self.optionflags & IGNORE_EXCEPTION_DETAIL:
                    if check(_strip_exception_details(example.exc_msg),
                             _strip_exception_details(exc_msg),
                             self.optionflags):
                        outcome = SUCCESS

            # Report the outcome.
            if outcome is SUCCESS:
                if not quiet:
                    self.report_success(out, test, example, got)
            elif outcome is FAILURE:
                if not quiet:
                    self.report_failure(out, test, example, got)
                failures += 1
            elif outcome is BOOM:
                if not quiet:
                    self.report_unexpected_exception(out, test, example,
                                                     exception)
                failures += 1
            else:
                assert False, ("unknown outcome", outcome)

            if failures and self.optionflags & FAIL_FAST:
                break

        # Restore the option flags (in case they were modified)
        self.optionflags = original_optionflags

        # Record and return the number of failures and tries.
        self.__record_outcome(test, failures, tries)
        return TestResults(failures, tries)

    def __record_outcome(self, test, f, t):
        """
        Record the fact that the given DocTest (`test`) generated `f`
        failures out of `t` tried examples.
        """
        f2, t2 = self._name2ft.get(test.name, (0,0))
        self._name2ft[test.name] = (f+f2, t+t2)
        self.failures += f
        self.tries += t

    __LINECACHE_FILENAME_RE = re.compile(r'<doctest '
                                         r'(?P<name>.+)'
                                         r'\[(?P<examplenum>\d+)\]>$')
    def __patched_linecache_getlines(self, filename, module_globals=None):
        m = self.__LINECACHE_FILENAME_RE.match(filename)
        if m and m.group('name') == self.test.name:
            example = self.test.examples[int(m.group('examplenum'))]
            return example.source.splitlines(keepends=True)
        else:
            return self.save_linecache_getlines(filename, module_globals)

    def run(self, test, compileflags=None, out=None, clear_globs=True):
        """
        Run the examples in `test`, and display the results using the
        writer function `out`.

        The examples are run in the namespace `test.globs`.  If
        `clear_globs` is true (the default), then this namespace will
        be cleared after the test runs, to help with garbage
        collection.  If you would like to examine the namespace after
        the test completes, then use `clear_globs=False`.

        `compileflags` gives the set of flags that should be used by
        the Python compiler when running the examples.  If not
        specified, then it will default to the set of future-import
        flags that apply to `globs`.

        The output of each example is checked using
        `DocTestRunner.check_output`, and the results are formatted by
        the `DocTestRunner.report_*` methods.
        """
        self.test = test

        if compileflags is None:
            compileflags = _extract_future_flags(test.globs)

        save_stdout = sys.stdout
        if out is None:
            encoding = save_stdout.encoding
            if encoding is None or encoding.lower() == 'utf-8':
                out = save_stdout.write
            else:
                # Use backslashreplace error handling on write
                def out(s):
                    s = str(s.encode(encoding, 'backslashreplace'), encoding)
                    save_stdout.write(s)
        sys.stdout = self._fakeout

        # Patch pdb.set_trace to restore sys.stdout during interactive
        # debugging (so it's not still redirected to self._fakeout).
        # Note that the interactive output will go to *our*
        # save_stdout, even if that's not the real sys.stdout; this
        # allows us to write test cases for the set_trace behavior.
        save_trace = sys.gettrace()
        save_set_trace = pdb.set_trace
        self.debugger = _OutputRedirectingPdb(save_stdout)
        self.debugger.reset()
        pdb.set_trace = self.debugger.set_trace

        # Patch linecache.getlines, so we can see the example's source
        # when we're inside the debugger.
        self.save_linecache_getlines = linecache.getlines
        linecache.getlines = self.__patched_linecache_getlines

        # Make sure sys.displayhook just prints the value to stdout
        save_displayhook = sys.displayhook
        sys.displayhook = sys.__displayhook__

        try:
            return self.__run(test, compileflags, out)
        finally:
            sys.stdout = save_stdout
            pdb.set_trace = save_set_trace
            sys.settrace(save_trace)
            linecache.getlines = self.save_linecache_getlines
            sys.displayhook = save_displayhook
            if clear_globs:
                test.globs.clear()
                import builtins
                builtins._ = None

    #/////////////////////////////////////////////////////////////////
    # Summarization
    #/////////////////////////////////////////////////////////////////
    def summarize(self, verbose=None):
        """
        Print a summary of all the test cases that have been run by
        this DocTestRunner, and return a tuple `(f, t)`, where `f` is
        the total number of failed examples, and `t` is the total
        number of tried examples.

        The optional `verbose` argument controls how detailed the
        summary is.  If the verbosity is not specified, then the
        DocTestRunner's verbosity is used.
        """
        if verbose is None:
            verbose = self._verbose
        notests = []
        passed = []
        failed = []
        totalt = totalf = 0
        for x in self._name2ft.items():
            name, (f, t) = x
            assert f <= t
            totalt += t
            totalf += f
            if t == 0:
                notests.append(name)
            elif f == 0:
                passed.append( (name, t) )
            else:
                failed.append(x)
        if verbose:
            if notests:
                print(len(notests), "items had no tests:")
                notests.sort()
                for thing in notests:
                    print("   ", thing)
            if passed:
                print(len(passed), "items passed all tests:")
                passed.sort()
                for thing, count in passed:
                    print(" %3d tests in %s" % (count, thing))
        if failed:
            print(self.DIVIDER)
            print(len(failed), "items had failures:")
            failed.sort()
            for thing, (f, t) in failed:
                print(" %3d of %3d in %s" % (f, t, thing))
        if verbose:
            print(totalt, "tests in", len(self._name2ft), "items.")
            print(totalt - totalf, "passed and", totalf, "failed.")
        if totalf:
            print("***Test Failed***", totalf, "failures.")
        elif verbose:
            print("Test passed.")
        return TestResults(totalf, totalt)

    #/////////////////////////////////////////////////////////////////
    # Backward compatibility cruft to maintain doctest.master.
    #/////////////////////////////////////////////////////////////////
    def merge(self, other):
        d = self._name2ft
        for name, (f, t) in other._name2ft.items():
            if name in d:
                # Don't print here by default, since doing
                #     so breaks some of the buildbots
                #print("*** DocTestRunner.merge: '" + name + "' in both" \
                #    " testers; summing outcomes.")
                f2, t2 = d[name]
                f = f + f2
                t = t + t2
            d[name] = f, t

class OutputChecker:
    """
    A class used to check the whether the actual output from a doctest
    example matches the expected output.  `OutputChecker` defines two
    methods: `check_output`, which compares a given pair of outputs,
    and returns true if they match; and `output_difference`, which
    returns a string describing the differences between two outputs.
    """
    def _toAscii(self, s):
        """
        Convert string to hex-escaped ASCII string.
        """
        return str(s.encode('ASCII', 'backslashreplace'), "ASCII")

    def check_output(self, want, got, optionflags):
        """
        Return True iff the actual output from an example (`got`)
        matches the expected output (`want`).  These strings are
        always considered to match if they are identical; but
        depending on what option flags the test runner is using,
        several non-exact match types are also possible.  See the
        documentation for `TestRunner` for more information about
        option flags.
        """

        # If `want` contains hex-escaped character such as "\u1234",
        # then `want` is a string of six characters(e.g. [\,u,1,2,3,4]).
        # On the other hand, `got` could be another sequence of
        # characters such as [\u1234], so `want` and `got` should
        # be folded to hex-escaped ASCII string to compare.
        got = self._toAscii(got)
        want = self._toAscii(want)

        # Handle the common case first, for efficiency:
        # if they're string-identical, always return true.
        if got == want:
            return True

        # The values True and False replaced 1 and 0 as the return
        # value for boolean comparisons in Python 2.3.
        if not (optionflags & DONT_ACCEPT_TRUE_FOR_1):
            if (got,want) == ("True\n", "1\n"):
                return True
            if (got,want) == ("False\n", "0\n"):
                return True

        # <BLANKLINE> can be used as a special sequence to signify a
        # blank line, unless the DONT_ACCEPT_BLANKLINE flag is used.
        if not (optionflags & DONT_ACCEPT_BLANKLINE):
            # Replace <BLANKLINE> in want with a blank line.
            want = re.sub(r'(?m)^%s\s*?$' % re.escape(BLANKLINE_MARKER),
                          '', want)
            # If a line in got contains only spaces, then remove the
            # spaces.
            got = re.sub(r'(?m)^[^\S\n]+$', '', got)
            if got == want:
                return True

        # This flag causes doctest to ignore any differences in the
        # contents of whitespace strings.  Note that this can be used
        # in conjunction with the ELLIPSIS flag.
        if optionflags & NORMALIZE_WHITESPACE:
            got = ' '.join(got.split())
            want = ' '.join(want.split())
            if got == want:
                return True

        # The ELLIPSIS flag says to let the sequence "..." in `want`
        # match any substring in `got`.
        if optionflags & ELLIPSIS:
            if _ellipsis_match(want, got):
                return True

        # We didn't find any match; return false.
        return False

    # Should we do a fancy diff?
    def _do_a_fancy_diff(self, want, got, optionflags):
        # Not unless they asked for a fancy diff.
        if not optionflags & (REPORT_UDIFF |
                              REPORT_CDIFF |
                              REPORT_NDIFF):
            return False

        # If expected output uses ellipsis, a meaningful fancy diff is
        # too hard ... or maybe not.  In two real-life failures Tim saw,
        # a diff was a major help anyway, so this is commented out.
        # [todo] _ellipsis_match() knows which pieces do and don't match,
        # and could be the basis for a kick-ass diff in this case.
        ##if optionflags & ELLIPSIS and ELLIPSIS_MARKER in want:
        ##    return False

        # ndiff does intraline difference marking, so can be useful even
        # for 1-line differences.
        if optionflags & REPORT_NDIFF:
            return True

        # The other diff types need at least a few lines to be helpful.
        return want.count('\n') > 2 and got.count('\n') > 2

    def output_difference(self, example, got, optionflags):
        """
        Return a string describing the differences between the
        expected output for a given example (`example`) and the actual
        output (`got`).  `optionflags` is the set of option flags used
        to compare `want` and `got`.
        """
        want = example.want
        # If <BLANKLINE>s are being used, then replace blank lines
        # with <BLANKLINE> in the actual output string.
        if not (optionflags & DONT_ACCEPT_BLANKLINE):
            got = re.sub('(?m)^[ ]*(?=\n)', BLANKLINE_MARKER, got)

        # Check if we should use diff.
        if self._do_a_fancy_diff(want, got, optionflags):
            # Split want & got into lines.
            want_lines = want.splitlines(keepends=True)
            got_lines = got.splitlines(keepends=True)
            # Use difflib to find their differences.
            if optionflags & REPORT_UDIFF:
                diff = difflib.unified_diff(want_lines, got_lines, n=2)
                diff = list(diff)[2:] # strip the diff header
                kind = 'unified diff with -expected +actual'
            elif optionflags & REPORT_CDIFF:
                diff = difflib.context_diff(want_lines, got_lines, n=2)
                diff = list(diff)[2:] # strip the diff header
                kind = 'context diff with expected followed by actual'
            elif optionflags & REPORT_NDIFF:
                engine = difflib.Differ(charjunk=difflib.IS_CHARACTER_JUNK)
                diff = list(engine.compare(want_lines, got_lines))
                kind = 'ndiff with -expected +actual'
            else:
                assert 0, 'Bad diff option'
            return 'Differences (%s):\n' % kind + _indent(''.join(diff))

        # If we're not using diff, then simply list the expected
        # output followed by the actual output.
        if want and got:
            return 'Expected:\n%sGot:\n%s' % (_indent(want), _indent(got))
        elif want:
            return 'Expected:\n%sGot nothing\n' % _indent(want)
        elif got:
            return 'Expected nothing\nGot:\n%s' % _indent(got)
        else:
            return 'Expected nothing\nGot nothing\n'

class DocTestFailure(Exception):
    """A DocTest example has failed in debugging mode.

    The exception instance has variables:

    - test: the DocTest object being run

    - example: the Example object that failed

    - got: the actual output
    """
    def __init__(self, test, example, got):
        self.test = test
        self.example = example
        self.got = got

    def __str__(self):
        return str(self.test)

class UnexpectedException(Exception):
    """A DocTest example has encountered an unexpected exception

    The exception instance has variables:

    - test: the DocTest object being run

    - example: the Example object that failed

    - exc_info: the exception info
    """
    def __init__(self, test, example, exc_info):
        self.test = test
        self.example = example
        self.exc_info = exc_info

    def __str__(self):
        return str(self.test)

class DebugRunner(DocTestRunner):
    r"""Run doc tests but raise an exception as soon as there is a failure.

       If an unexpected exception occurs, an UnexpectedException is raised.
       It contains the test, the example, and the original exception:

         >>> runner = DebugRunner(verbose=False)
         >>> test = DocTestParser().get_doctest('>>> raise KeyError\n42',
         ...                                    {}, 'foo', 'foo.py', 0)
         >>> try:
         ...     runner.run(test)
         ... except UnexpectedException as f:
         ...     failure = f

         >>> failure.test is test
         True

         >>> failure.example.want
         '42\n'

         >>> exc_info = failure.exc_info
         >>> raise exc_info[1] # Already has the traceback
         Traceback (most recent call last):
         ...
         KeyError

       We wrap the original exception to give the calling application
       access to the test and example information.

       If the output doesn't match, then a DocTestFailure is raised:

         >>> test = DocTestParser().get_doctest('''
         ...      >>> x = 1
         ...      >>> x
         ...      2
         ...      ''', {}, 'foo', 'foo.py', 0)

         >>> try:
         ...    runner.run(test)
         ... except DocTestFailure as f:
         ...    failure = f

       DocTestFailure objects provide access to the test:

         >>> failure.test is test
         True

       As well as to the example:

         >>> failure.example.want
         '2\n'

       and the actual output:

         >>> failure.got
         '1\n'

       If a failure or error occurs, the globals are left intact:

         >>> del test.globs['__builtins__']
         >>> test.globs
         {'x': 1}

         >>> test = DocTestParser().get_doctest('''
         ...      >>> x = 2
         ...      >>> raise KeyError
         ...      ''', {}, 'foo', 'foo.py', 0)

         >>> runner.run(test)
         Traceback (most recent call last):
         ...
         doctest.UnexpectedException: <DocTest foo from foo.py:0 (2 examples)>

         >>> del test.globs['__builtins__']
         >>> test.globs
         {'x': 2}

       But the globals are cleared if there is no error:

         >>> test = DocTestParser().get_doctest('''
         ...      >>> x = 2
         ...      ''', {}, 'foo', 'foo.py', 0)

         >>> runner.run(test)
         TestResults(failed=0, attempted=1)

         >>> test.globs
         {}

       """

    def run(self, test, compileflags=None, out=None, clear_globs=True):
        r = DocTestRunner.run(self, test, compileflags, out, False)
        if clear_globs:
            test.globs.clear()
        return r

    def report_unexpected_exception(self, out, test, example, exc_info):
        raise UnexpectedException(test, example, exc_info)

    def report_failure(self, out, test, example, got):
        raise DocTestFailure(test, example, got)

######################################################################
## 6. Test Functions
######################################################################
# These should be backwards compatible.

# For backward compatibility, a global instance of a DocTestRunner
# class, updated by testmod.
master = None

def testmod(m=None, name=None, globs=None, verbose=None,
            report=True, optionflags=0, extraglobs=None,
            raise_on_error=False, exclude_empty=False):
    """m=None, name=None, globs=None, verbose=None, report=True,
       optionflags=0, extraglobs=None, raise_on_error=False,
       exclude_empty=False

    Test examples in docstrings in functions and classes reachable
    from module m (or the current module if m is not supplied), starting
    with m.__doc__.

    Also test examples reachable from dict m.__test__ if it exists and is
    not None.  m.__test__ maps names to functions, classes and strings;
    function and class docstrings are tested even if the name is private;
    strings are tested directly, as if they were docstrings.

    Return (#failures, #tests).

    See help(doctest) for an overview.

    Optional keyword arg "name" gives the name of the module; by default
    use m.__name__.

    Optional keyword arg "globs" gives a dict to be used as the globals
    when executing examples; by default, use m.__dict__.  A copy of this
    dict is actually used for each docstring, so that each docstring's
    examples start with a clean slate.

    Optional keyword arg "extraglobs" gives a dictionary that should be
    merged into the globals that are used to execute examples.  By
    default, no extra globals are used.  This is new in 2.4.

    Optional keyword arg "verbose" prints lots of stuff if true, prints
    only failures if false; by default, it's true iff "-v" is in sys.argv.

    Optional keyword arg "report" prints a summary at the end when true,
    else prints nothing at the end.  In verbose mode, the summary is
    detailed, else very brief (in fact, empty if all tests passed).

    Optional keyword arg "optionflags" or's together module constants,
    and defaults to 0.  This is new in 2.3.  Possible values (see the
    docs for details):

        DONT_ACCEPT_TRUE_FOR_1
        DONT_ACCEPT_BLANKLINE
        NORMALIZE_WHITESPACE
        ELLIPSIS
        SKIP
        IGNORE_EXCEPTION_DETAIL
        REPORT_UDIFF
        REPORT_CDIFF
        REPORT_NDIFF
        REPORT_ONLY_FIRST_FAILURE

    Optional keyword arg "raise_on_error" raises an exception on the
    first unexpected exception or failure. This allows failures to be
    post-mortem debugged.

    Advanced tomfoolery:  testmod runs methods of a local instance of
    class doctest.Tester, then merges the results into (or creates)
    global Tester instance doctest.master.  Methods of doctest.master
    can be called directly too, if you want to do something unusual.
    Passing report=0 to testmod is especially useful then, to delay
    displaying a summary.  Invoke doctest.master.summarize(verbose)
    when you're done fiddling.
    """
    global master

    # If no module was given, then use __main__.
    if m is None:
        # DWA - m will still be None if this wasn't invoked from the command
        # line, in which case the following TypeError is about as good an error
        # as we should expect
        m = sys.modules.get('__main__')

    # Check that we were actually given a module.
    if not inspect.ismodule(m):
        raise TypeError("testmod: module required; %r" % (m,))

    # If no name was given, then use the module's name.
    if name is None:
        name = m.__name__

    # Find, parse, and run all tests in the given module.
    finder = DocTestFinder(exclude_empty=exclude_empty)

    if raise_on_error:
        runner = DebugRunner(verbose=verbose, optionflags=optionflags)
    else:
        runner = DocTestRunner(verbose=verbose, optionflags=optionflags)

    for test in finder.find(m, name, globs=globs, extraglobs=extraglobs):
        runner.run(test)

    if report:
        runner.summarize()

    if master is None:
        master = runner
    else:
        master.merge(runner)

    return TestResults(runner.failures, runner.tries)

def testfile(filename, module_relative=True, name=None, package=None,
             globs=None, verbose=None, report=True, optionflags=0,
             extraglobs=None, raise_on_error=False, parser=DocTestParser(),
             encoding=None):
    """
    Test examples in the given file.  Return (#failures, #tests).

    Optional keyword arg "module_relative" specifies how filenames
    should be interpreted:

      - If "module_relative" is True (the default), then "filename"
         specifies a module-relative path.  By default, this path is
         relative to the calling module's directory; but if the
         "package" argument is specified, then it is relative to that
         package.  To ensure os-independence, "filename" should use
         "/" characters to separate path segments, and should not
         be an absolute path (i.e., it may not begin with "/").

      - If "module_relative" is False, then "filename" specifies an
        os-specific path.  The path may be absolute or relative (to
        the current working directory).

    Optional keyword arg "name" gives the name of the test; by default
    use the file's basename.

    Optional keyword argument "package" is a Python package or the
    name of a Python package whose directory should be used as the
    base directory for a module relative filename.  If no package is
    specified, then the calling module's directory is used as the base
    directory for module relative filenames.  It is an error to
    specify "package" if "module_relative" is False.

    Optional keyword arg "globs" gives a dict to be used as the globals
    when executing examples; by default, use {}.  A copy of this dict
    is actually used for each docstring, so that each docstring's
    examples start with a clean slate.

    Optional keyword arg "extraglobs" gives a dictionary that should be
    merged into the globals that are used to execute examples.  By
    default, no extra globals are used.

    Optional keyword arg "verbose" prints lots of stuff if true, prints
    only failures if false; by default, it's true iff "-v" is in sys.argv.

    Optional keyword arg "report" prints a summary at the end when true,
    else prints nothing at the end.  In verbose mode, the summary is
    detailed, else very brief (in fact, empty if all tests passed).

    Optional keyword arg "optionflags" or's together module constants,
    and defaults to 0.  Possible values (see the docs for details):

        DONT_ACCEPT_TRUE_FOR_1
        DONT_ACCEPT_BLANKLINE
        NORMALIZE_WHITESPACE
        ELLIPSIS
        SKIP
        IGNORE_EXCEPTION_DETAIL
        REPORT_UDIFF
        REPORT_CDIFF
        REPORT_NDIFF
        REPORT_ONLY_FIRST_FAILURE

    Optional keyword arg "raise_on_error" raises an exception on the
    first unexpected exception or failure. This allows failures to be
    post-mortem debugged.

    Optional keyword arg "parser" specifies a DocTestParser (or
    subclass) that should be used to extract tests from the files.

    Optional keyword arg "encoding" specifies an encoding that should
    be used to convert the file to unicode.

    Advanced tomfoolery:  testmod runs methods of a local instance of
    class doctest.Tester, then merges the results into (or creates)
    global Tester instance doctest.master.  Methods of doctest.master
    can be called directly too, if you want to do something unusual.
    Passing report=0 to testmod is especially useful then, to delay
    displaying a summary.  Invoke doctest.master.summarize(verbose)
    when you're done fiddling.
    """
    global master

    if package and not module_relative:
        raise ValueError("Package may only be specified for module-"
                         "relative paths.")

    # Relativize the path
    text, filename = _load_testfile(filename, package, module_relative,
                                    encoding or "utf-8")

    # If no name was given, then use the file's name.
    if name is None:
        name = os.path.basename(filename)

    # Assemble the globals.
    if globs is None:
        globs = {}
    else:
        globs = globs.copy()
    if extraglobs is not None:
        globs.update(extraglobs)
    if '__name__' not in globs:
        globs['__name__'] = '__main__'

    if raise_on_error:
        runner = DebugRunner(verbose=verbose, optionflags=optionflags)
    else:
        runner = DocTestRunner(verbose=verbose, optionflags=optionflags)

    # Read the file, convert it to a test, and run it.
    test = parser.get_doctest(text, globs, name, filename, 0)
    runner.run(test)

    if report:
        runner.summarize()

    if master is None:
        master = runner
    else:
        master.merge(runner)

    return TestResults(runner.failures, runner.tries)

def run_docstring_examples(f, globs, verbose=False, name="NoName",
                           compileflags=None, optionflags=0):
    """
    Test examples in the given object's docstring (`f`), using `globs`
    as globals.  Optional argument `name` is used in failure messages.
    If the optional argument `verbose` is true, then generate output
    even if there are no failures.

    `compileflags` gives the set of flags that should be used by the
    Python compiler when running the examples.  If not specified, then
    it will default to the set of future-import flags that apply to
    `globs`.

    Optional keyword arg `optionflags` specifies options for the
    testing and output.  See the documentation for `testmod` for more
    information.
    """
    # Find, parse, and run all tests in the given module.
    finder = DocTestFinder(verbose=verbose, recurse=False)
    runner = DocTestRunner(verbose=verbose, optionflags=optionflags)
    for test in finder.find(f, name, globs=globs):
        runner.run(test, compileflags=compileflags)

######################################################################
## 7. Unittest Support
######################################################################

_unittest_reportflags = 0

def set_unittest_reportflags(flags):
    """Sets the unittest option flags.

    The old flag is returned so that a runner could restore the old
    value if it wished to:

      >>> import doctest
      >>> old = doctest._unittest_reportflags
      >>> doctest.set_unittest_reportflags(REPORT_NDIFF |
      ...                          REPORT_ONLY_FIRST_FAILURE) == old
      True

      >>> doctest._unittest_reportflags == (REPORT_NDIFF |
      ...                                   REPORT_ONLY_FIRST_FAILURE)
      True

    Only reporting flags can be set:

      >>> doctest.set_unittest_reportflags(ELLIPSIS)
      Traceback (most recent call last):
      ...
      ValueError: ('Only reporting flags allowed', 8)

      >>> doctest.set_unittest_reportflags(old) == (REPORT_NDIFF |
      ...                                   REPORT_ONLY_FIRST_FAILURE)
      True
    """
    global _unittest_reportflags

    if (flags & REPORTING_FLAGS) != flags:
        raise ValueError("Only reporting flags allowed", flags)
    old = _unittest_reportflags
    _unittest_reportflags = flags
    return old


class DocTestCase(unittest.TestCase):

    def __init__(self, test, optionflags=0, setUp=None, tearDown=None,
                 checker=None):

        unittest.TestCase.__init__(self)
        self._dt_optionflags = optionflags
        self._dt_checker = checker
        self._dt_globs = test.globs.copy()
        self._dt_test = test
        self._dt_setUp = setUp
        self._dt_tearDown = tearDown

    def setUp(self):
        test = self._dt_test

        if self._dt_setUp is not None:
            self._dt_setUp(test)

    def tearDown(self):
        test = self._dt_test

        if self._dt_tearDown is not None:
            self._dt_tearDown(test)

        # restore the original globs
        test.globs.clear()
        test.globs.update(self._dt_globs)

    def runTest(self):
        test = self._dt_test
        old = sys.stdout
        new = StringIO()
        optionflags = self._dt_optionflags

        if not (optionflags & REPORTING_FLAGS):
            # The option flags don't include any reporting flags,
            # so add the default reporting flags
            optionflags |= _unittest_reportflags

        runner = DocTestRunner(optionflags=optionflags,
                               checker=self._dt_checker, verbose=False)

        try:
            runner.DIVIDER = "-"*70
            failures, tries = runner.run(
                test, out=new.write, clear_globs=False)
        finally:
            sys.stdout = old

        if failures:
            raise self.failureException(self.format_failure(new.getvalue()))

    def format_failure(self, err):
        test = self._dt_test
        if test.lineno is None:
            lineno = 'unknown line number'
        else:
            lineno = '%s' % test.lineno
        lname = '.'.join(test.name.split('.')[-1:])
        return ('Failed doctest test for %s\n'
                '  File "%s", line %s, in %s\n\n%s'
                % (test.name, test.filename, lineno, lname, err)
                )

    def debug(self):
        r"""Run the test case without results and without catching exceptions

           The unit test framework includes a debug method on test cases
           and test suites to support post-mortem debugging.  The test code
           is run in such a way that errors are not caught.  This way a
           caller can catch the errors and initiate post-mortem debugging.

           The DocTestCase provides a debug method that raises
           UnexpectedException errors if there is an unexpected
           exception:

             >>> test = DocTestParser().get_doctest('>>> raise KeyError\n42',
             ...                {}, 'foo', 'foo.py', 0)
             >>> case = DocTestCase(test)
             >>> try:
             ...     case.debug()
             ... except UnexpectedException as f:
             ...     failure = f

           The UnexpectedException contains the test, the example, and
           the original exception:

             >>> failure.test is test
             True

             >>> failure.example.want
             '42\n'

             >>> exc_info = failure.exc_info
             >>> raise exc_info[1] # Already has the traceback
             Traceback (most recent call last):
             ...
             KeyError

           If the output doesn't match, then a DocTestFailure is raised:

             >>> test = DocTestParser().get_doctest('''
             ...      >>> x = 1
             ...      >>> x
             ...      2
             ...      ''', {}, 'foo', 'foo.py', 0)
             >>> case = DocTestCase(test)

             >>> try:
             ...    case.debug()
             ... except DocTestFailure as f:
             ...    failure = f

           DocTestFailure objects provide access to the test:

             >>> failure.test is test
             True

           As well as to the example:

             >>> failure.example.want
             '2\n'

           and the actual output:

             >>> failure.got
             '1\n'

           """

        self.setUp()
        runner = DebugRunner(optionflags=self._dt_optionflags,
                             checker=self._dt_checker, verbose=False)
        runner.run(self._dt_test, clear_globs=False)
        self.tearDown()

    def id(self):
        return self._dt_test.name

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented

        return self._dt_test == other._dt_test and \
               self._dt_optionflags == other._dt_optionflags and \
               self._dt_setUp == other._dt_setUp and \
               self._dt_tearDown == other._dt_tearDown and \
               self._dt_checker == other._dt_checker

    def __hash__(self):
        return hash((self._dt_optionflags, self._dt_setUp, self._dt_tearDown,
                     self._dt_checker))

    def __repr__(self):
        name = self._dt_test.name.split('.')
        return "%s (%s)" % (name[-1], '.'.join(name[:-1]))

    __str__ = object.__str__

    def shortDescription(self):
        return "Doctest: " + self._dt_test.name

class SkipDocTestCase(DocTestCase):
    def __init__(self, module):
        self.module = module
        DocTestCase.__init__(self, None)

    def setUp(self):
        self.skipTest("DocTestSuite will not work with -O2 and above")

    def test_skip(self):
        pass

    def shortDescription(self):
        return "Skipping tests from %s" % self.module.__name__

    __str__ = shortDescription


class _DocTestSuite(unittest.TestSuite):

    def _removeTestAtIndex(self, index):
        pass


def DocTestSuite(module=None, globs=None, extraglobs=None, test_finder=None,
                 **options):
    """
    Convert doctest tests for a module to a unittest test suite.

    This converts each documentation string in a module that
    contains doctest tests to a unittest test case.  If any of the
    tests in a doc string fail, then the test case fails.  An exception
    is raised showing the name of the file containing the test and a
    (sometimes approximate) line number.

    The `module` argument provides the module to be tested.  The argument
    can be either a module or a module name.

    If no argument is given, the calling module is used.

    A number of options may be provided as keyword arguments:

    setUp
      A set-up function.  This is called before running the
      tests in each file. The setUp function will be passed a DocTest
      object.  The setUp function can access the test globals as the
      globs attribute of the test passed.

    tearDown
      A tear-down function.  This is called after running the
      tests in each file.  The tearDown function will be passed a DocTest
      object.  The tearDown function can access the test globals as the
      globs attribute of the test passed.

    globs
      A dictionary containing initial global variables for the tests.

    optionflags
       A set of doctest option flags expressed as an integer.
    """

    if test_finder is None:
        test_finder = DocTestFinder()

    module = _normalize_module(module)
    tests = test_finder.find(module, globs=globs, extraglobs=extraglobs)

    if not tests and sys.flags.optimize >=2:
        # Skip doctests when running with -O2
        suite = _DocTestSuite()
        suite.addTest(SkipDocTestCase(module))
        return suite

    tests.sort()
    suite = _DocTestSuite()

    for test in tests:
        if len(test.examples) == 0:
            continue
        if not test.filename:
            filename = module.__file__
            if filename[-4:] == ".pyc":
                filename = filename[:-1]
            test.filename = filename
        suite.addTest(DocTestCase(test, **options))

    return suite

class DocFileCase(DocTestCase):

    def id(self):
        return '_'.join(self._dt_test.name.split('.'))

    def __repr__(self):
        return self._dt_test.filename

    def format_failure(self, err):
        return ('Failed doctest test for %s\n  File "%s", line 0\n\n%s'
                % (self._dt_test.name, self._dt_test.filename, err)
                )

def DocFileTest(path, module_relative=True, package=None,
                globs=None, parser=DocTestParser(),
                encoding=None, **options):
    if globs is None:
        globs = {}
    else:
        globs = globs.copy()

    if package and not module_relative:
        raise ValueError("Package may only be specified for module-"
                         "relative paths.")

    # Relativize the path.
    doc, path = _load_testfile(path, package, module_relative,
                               encoding or "utf-8")

    if "__file__" not in globs:
        globs["__file__"] = path

    # Find the file and read it.
    name = os.path.basename(path)

    # Convert it to a test, and wrap it in a DocFileCase.
    test = parser.get_doctest(doc, globs, name, path, 0)
    return DocFileCase(test, **options)

def DocFileSuite(*paths, **kw):
    """A unittest suite for one or more doctest files.

    The path to each doctest file is given as a string; the
    interpretation of that string depends on the keyword argument
    "module_relative".

    A number of options may be provided as keyword arguments:

    module_relative
      If "module_relative" is True, then the given file paths are
      interpreted as os-independent module-relative paths.  By
      default, these paths are relative to the calling module's
      directory; but if the "package" argument is specified, then
      they are relative to that package.  To ensure os-independence,
      "filename" should use "/" characters to separate path
      segments, and may not be an absolute path (i.e., it may not
      begin with "/").

      If "module_relative" is False, then the given file paths are
      interpreted as os-specific paths.  These paths may be absolute
      or relative (to the current working directory).

    package
      A Python package or the name of a Python package whose directory
      should be used as the base directory for module relative paths.
      If "package" is not specified, then the calling module's
      directory is used as the base directory for module relative
      filenames.  It is an error to specify "package" if
      "module_relative" is False.

    setUp
      A set-up function.  This is called before running the
      tests in each file. The setUp function will be passed a DocTest
      object.  The setUp function can access the test globals as the
      globs attribute of the test passed.

    tearDown
      A tear-down function.  This is called after running the
      tests in each file.  The tearDown function will be passed a DocTest
      object.  The tearDown function can access the test globals as the
      globs attribute of the test passed.

    globs
      A dictionary containing initial global variables for the tests.

    optionflags
      A set of doctest option flags expressed as an integer.

    parser
      A DocTestParser (or subclass) that should be used to extract
      tests from the files.

    encoding
      An encoding that will be used to convert the files to unicode.
    """
    suite = _DocTestSuite()

    # We do this here so that _normalize_module is called at the right
    # level.  If it were called in DocFileTest, then this function
    # would be the caller and we might guess the package incorrectly.
    if kw.get('module_relative', True):
        kw['package'] = _normalize_module(kw.get('package'))

    for path in paths:
        suite.addTest(DocFileTest(path, **kw))

    return suite

######################################################################
## 8. Debugging Support
######################################################################

def script_from_examples(s):
    r"""Extract script from text with examples.

       Converts text with examples to a Python script.  Example input is
       converted to regular code.  Example output and all other words
       are converted to comments:

       >>> text = '''
       ...       Here are examples of simple math.
       ...
       ...           Python has super accurate integer addition
       ...
       ...           >>> 2 + 2
       ...           5
       ...
       ...           And very friendly error messages:
       ...
       ...           >>> 1/0
       ...           To Infinity
       ...           And
       ...           Beyond
       ...
       ...           You can use logic if you want:
       ...
       ...           >>> if 0:
       ...           ...    blah
       ...           ...    blah
       ...           ...
       ...
       ...           Ho hum
       ...           '''

       >>> print(script_from_examples(text))
       # Here are examples of simple math.
       #
       #     Python has super accurate integer addition
       #
       2 + 2
       # Expected:
       ## 5
       #
       #     And very friendly error messages:
       #
       1/0
       # Expected:
       ## To Infinity
       ## And
       ## Beyond
       #
       #     You can use logic if you want:
       #
       if 0:
          blah
          blah
       #
       #     Ho hum
       <BLANKLINE>
       """
    output = []
    for piece in DocTestParser().parse(s):
        if isinstance(piece, Example):
            # Add the example's source code (strip trailing NL)
            output.append(piece.source[:-1])
            # Add the expected output:
            want = piece.want
            if want:
                output.append('# Expected:')
                output += ['## '+l for l in want.split('\n')[:-1]]
        else:
            # Add non-example text.
            output += [_comment_line(l)
                       for l in piece.split('\n')[:-1]]

    # Trim junk on both ends.
    while output and output[-1] == '#':
        output.pop()
    while output and output[0] == '#':
        output.pop(0)
    # Combine the output, and return it.
    # Add a courtesy newline to prevent exec from choking (see bug #1172785)
    return '\n'.join(output) + '\n'

def testsource(module, name):
    """Extract the test sources from a doctest docstring as a script.

    Provide the module (or dotted name of the module) containing the
    test to be debugged and the name (within the module) of the object
    with the doc string with tests to be debugged.
    """
    module = _normalize_module(module)
    tests = DocTestFinder().find(module)
    test = [t for t in tests if t.name == name]
    if not test:
        raise ValueError(name, "not found in tests")
    test = test[0]
    testsrc = script_from_examples(test.docstring)
    return testsrc

def debug_src(src, pm=False, globs=None):
    """Debug a single doctest docstring, in argument `src`'"""
    testsrc = script_from_examples(src)
    debug_script(testsrc, pm, globs)

def debug_script(src, pm=False, globs=None):
    "Debug a test script.  `src` is the script, as a string."
    import pdb

    if globs:
        globs = globs.copy()
    else:
        globs = {}

    if pm:
        try:
            exec(src, globs, globs)
        except:
            print(sys.exc_info()[1])
            p = pdb.Pdb(nosigint=True)
            p.reset()
            p.interaction(None, sys.exc_info()[2])
    else:
        pdb.Pdb(nosigint=True).run("exec(%r)" % src, globs, globs)

def debug(module, name, pm=False):
    """Debug a single doctest docstring.

    Provide the module (or dotted name of the module) containing the
    test to be debugged and the name (within the module) of the object
    with the docstring with tests to be debugged.
    """
    module = _normalize_module(module)
    testsrc = testsource(module, name)
    debug_script(testsrc, pm, module.__dict__)

######################################################################
## 9. Example Usage
######################################################################
class _TestClass:
    """
    A pointless class, for sanity-checking of docstring testing.

    Methods:
        square()
        get()

    >>> _TestClass(13).get() + _TestClass(-12).get()
    1
    >>> hex(_TestClass(13).square().get())
    '0xa9'
    """

    def __init__(self, val):
        """val -> _TestClass object with associated value val.

        >>> t = _TestClass(123)
        >>> print(t.get())
        123
        """

        self.val = val

    def square(self):
        """square() -> square TestClass's associated value

        >>> _TestClass(13).square().get()
        169
        """

        self.val = self.val ** 2
        return self

    def get(self):
        """get() -> return TestClass's associated value.

        >>> x = _TestClass(-42)
        >>> print(x.get())
        -42
        """

        return self.val

__test__ = {"_TestClass": _TestClass,
            "string": r"""
                      Example of a string object, searched as-is.
                      >>> x = 1; y = 2
                      >>> x + y, x * y
                      (3, 2)
                      """,

            "bool-int equivalence": r"""
                                    In 2.2, boolean expressions displayed
                                    0 or 1.  By default, we still accept
                                    them.  This can be disabled by passing
                                    DONT_ACCEPT_TRUE_FOR_1 to the new
                                    optionflags argument.
                                    >>> 4 == 4
                                    1
                                    >>> 4 == 4
                                    True
                                    >>> 4 > 4
                                    0
                                    >>> 4 > 4
                                    False
                                    """,

            "blank lines": r"""
                Blank lines can be marked with <BLANKLINE>:
                    >>> print('foo\n\nbar\n')
                    foo
                    <BLANKLINE>
                    bar
                    <BLANKLINE>
            """,

            "ellipsis": r"""
                If the ellipsis flag is used, then '...' can be used to
                elide substrings in the desired output:
                    >>> print(list(range(1000))) #doctest: +ELLIPSIS
                    [0, 1, 2, ..., 999]
            """,

            "whitespace normalization": r"""
                If the whitespace normalization flag is used, then
                differences in whitespace are ignored.
                    >>> print(list(range(30))) #doctest: +NORMALIZE_WHITESPACE
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                     27, 28, 29]
            """,
           }


def _test():
    import argparse

    parser = argparse.ArgumentParser(description="doctest runner")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print very verbose output for all tests')
    parser.add_argument('-o', '--option', action='append',
                        choices=OPTIONFLAGS_BY_NAME.keys(), default=[],
                        help=('specify a doctest option flag to apply'
                              ' to the test run; may be specified more'
                              ' than once to apply multiple options'))
    parser.add_argument('-f', '--fail-fast', action='store_true',
                        help=('stop running tests after first failure (this'
                              ' is a shorthand for -o FAIL_FAST, and is'
                              ' in addition to any other -o options)'))
    parser.add_argument('file', nargs='+',
                        help='file containing the tests to run')
    args = parser.parse_args()
    testfiles = args.file
    # Verbose used to be handled by the "inspect argv" magic in DocTestRunner,
    # but since we are using argparse we are passing it manually now.
    verbose = args.verbose
    options = 0
    for option in args.option:
        options |= OPTIONFLAGS_BY_NAME[option]
    if args.fail_fast:
        options |= FAIL_FAST
    for filename in testfiles:
        if filename.endswith(".py"):
            # It is a module -- insert its dir into sys.path and try to
            # import it. If it is part of a package, that possibly
            # won't work because of package imports.
            dirname, filename = os.path.split(filename)
            sys.path.insert(0, dirname)
            m = __import__(filename[:-3])
            del sys.path[0]
            failures, _ = testmod(m, verbose=verbose, optionflags=options)
        else:
            failures, _ = testfile(filename, module_relative=False,
                                     verbose=verbose, optionflags=options)
        if failures:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(_test())
<|endoftext|>'''"Executable documentation" for the pickle module.

Extensive comments about the pickle protocols and pickle-machine opcodes
can be found here.  Some functions meant for external use:

genops(pickle)
   Generate all the opcodes in a pickle, as (opcode, arg, position) triples.

dis(pickle, out=None, memo=None, indentlevel=4)
   Print a symbolic disassembly of a pickle.
'''

import codecs
import io
import pickle
import re
import sys

__all__ = ['dis', 'genops', 'optimize']

bytes_types = pickle.bytes_types

# Other ideas:
#
# - A pickle verifier:  read a pickle and check it exhaustively for
#   well-formedness.  dis() does a lot of this already.
#
# - A protocol identifier:  examine a pickle and return its protocol number
#   (== the highest .proto attr value among all the opcodes in the pickle).
#   dis() already prints this info at the end.
#
# - A pickle optimizer:  for example, tuple-building code is sometimes more
#   elaborate than necessary, catering for the possibility that the tuple
#   is recursive.  Or lots of times a PUT is generated that's never accessed
#   by a later GET.


# "A pickle" is a program for a virtual pickle machine (PM, but more accurately
# called an unpickling machine).  It's a sequence of opcodes, interpreted by the
# PM, building an arbitrarily complex Python object.
#
# For the most part, the PM is very simple:  there are no looping, testing, or
# conditional instructions, no arithmetic and no function calls.  Opcodes are
# executed once each, from first to last, until a STOP opcode is reached.
#
# The PM has two data areas, "the stack" and "the memo".
#
# Many opcodes push Python objects onto the stack; e.g., INT pushes a Python
# integer object on the stack, whose value is gotten from a decimal string
# literal immediately following the INT opcode in the pickle bytestream.  Other
# opcodes take Python objects off the stack.  The result of unpickling is
# whatever object is left on the stack when the final STOP opcode is executed.
#
# The memo is simply an array of objects, or it can be implemented as a dict
# mapping little integers to objects.  The memo serves as the PM's "long term
# memory", and the little integers indexing the memo are akin to variable
# names.  Some opcodes pop a stack object into the memo at a given index,
# and others push a memo object at a given index onto the stack again.
#
# At heart, that's all the PM has.  Subtleties arise for these reasons:
#
# + Object identity.  Objects can be arbitrarily complex, and subobjects
#   may be shared (for example, the list [a, a] refers to the same object a
#   twice).  It can be vital that unpickling recreate an isomorphic object
#   graph, faithfully reproducing sharing.
#
# + Recursive objects.  For example, after "L = []; L.append(L)", L is a
#   list, and L[0] is the same list.  This is related to the object identity
#   point, and some sequences of pickle opcodes are subtle in order to
#   get the right result in all cases.
#
# + Things pickle doesn't know everything about.  Examples of things pickle
#   does know everything about are Python's builtin scalar and container
#   types, like ints and tuples.  They generally have opcodes dedicated to
#   them.  For things like module references and instances of user-defined
#   classes, pickle's knowledge is limited.  Historically, many enhancements
#   have been made to the pickle protocol in order to do a better (faster,
#   and/or more compact) job on those.
#
# + Backward compatibility and micro-optimization.  As explained below,
#   pickle opcodes never go away, not even when better ways to do a thing
#   get invented.  The repertoire of the PM just keeps growing over time.
#   For example, protocol 0 had two opcodes for building Python integers (INT
#   and LONG), protocol 1 added three more for more-efficient pickling of short
#   integers, and protocol 2 added two more for more-efficient pickling of
#   long integers (before protocol 2, the only ways to pickle a Python long
#   took time quadratic in the number of digits, for both pickling and
#   unpickling).  "Opcode bloat" isn't so much a subtlety as a source of
#   wearying complication.
#
#
# Pickle protocols:
#
# For compatibility, the meaning of a pickle opcode never changes.  Instead new
# pickle opcodes get added, and each version's unpickler can handle all the
# pickle opcodes in all protocol versions to date.  So old pickles continue to
# be readable forever.  The pickler can generally be told to restrict itself to
# the subset of opcodes available under previous protocol versions too, so that
# users can create pickles under the current version readable by older
# versions.  However, a pickle does not contain its version number embedded
# within it.  If an older unpickler tries to read a pickle using a later
# protocol, the result is most likely an exception due to seeing an unknown (in
# the older unpickler) opcode.
#
# The original pickle used what's now called "protocol 0", and what was called
# "text mode" before Python 2.3.  The entire pickle bytestream is made up of
# printable 7-bit ASCII characters, plus the newline character, in protocol 0.
# That's why it was called text mode.  Protocol 0 is small and elegant, but
# sometimes painfully inefficient.
#
# The second major set of additions is now called "protocol 1", and was called
# "binary mode" before Python 2.3.  This added many opcodes with arguments
# consisting of arbitrary bytes, including NUL bytes and unprintable "high bit"
# bytes.  Binary mode pickles can be substantially smaller than equivalent
# text mode pickles, and sometimes faster too; e.g., BININT represents a 4-byte
# int as 4 bytes following the opcode, which is cheaper to unpickle than the
# (perhaps) 11-character decimal string attached to INT.  Protocol 1 also added
# a number of opcodes that operate on many stack elements at once (like APPENDS
# and SETITEMS), and "shortcut" opcodes (like EMPTY_DICT and EMPTY_TUPLE).
#
# The third major set of additions came in Python 2.3, and is called "protocol
# 2".  This added:
#
# - A better way to pickle instances of new-style classes (NEWOBJ).
#
# - A way for a pickle to identify its protocol (PROTO).
#
# - Time- and space- efficient pickling of long ints (LONG{1,4}).
#
# - Shortcuts for small tuples (TUPLE{1,2,3}}.
#
# - Dedicated opcodes for bools (NEWTRUE, NEWFALSE).
#
# - The "extension registry", a vector of popular objects that can be pushed
#   efficiently by index (EXT{1,2,4}).  This is akin to the memo and GET, but
#   the registry contents are predefined (there's nothing akin to the memo's
#   PUT).
#
# Another independent change with Python 2.3 is the abandonment of any
# pretense that it might be safe to load pickles received from untrusted
# parties -- no sufficient security analysis has been done to guarantee
# this and there isn't a use case that warrants the expense of such an
# analysis.
#
# To this end, all tests for __safe_for_unpickling__ or for
# copyreg.safe_constructors are removed from the unpickling code.
# References to these variables in the descriptions below are to be seen
# as describing unpickling in Python 2.2 and before.


# Meta-rule:  Descriptions are stored in instances of descriptor objects,
# with plain constructors.  No meta-language is defined from which
# descriptors could be constructed.  If you want, e.g., XML, write a little
# program to generate XML from the objects.

##############################################################################
# Some pickle opcodes have an argument, following the opcode in the
# bytestream.  An argument is of a specific type, described by an instance
# of ArgumentDescriptor.  These are not to be confused with arguments taken
# off the stack -- ArgumentDescriptor applies only to arguments embedded in
# the opcode stream, immediately following an opcode.

# Represents the number of bytes consumed by an argument delimited by the
# next newline character.
UP_TO_NEWLINE = -1

# Represents the number of bytes consumed by a two-argument opcode where
# the first argument gives the number of bytes in the second argument.
TAKEN_FROM_ARGUMENT1  = -2   # num bytes is 1-byte unsigned int
TAKEN_FROM_ARGUMENT4  = -3   # num bytes is 4-byte signed little-endian int
TAKEN_FROM_ARGUMENT4U = -4   # num bytes is 4-byte unsigned little-endian int
TAKEN_FROM_ARGUMENT8U = -5   # num bytes is 8-byte unsigned little-endian int

class ArgumentDescriptor(object):
    __slots__ = (
        # name of descriptor record, also a module global name; a string
        'name',

        # length of argument, in bytes; an int; UP_TO_NEWLINE and
        # TAKEN_FROM_ARGUMENT{1,4,8} are negative values for variable-length
        # cases
        'n',

        # a function taking a file-like object, reading this kind of argument
        # from the object at the current position, advancing the current
        # position by n bytes, and returning the value of the argument
        'reader',

        # human-readable docs for this arg descriptor; a string
        'doc',
    )

    def __init__(self, name, n, reader, doc):
        assert isinstance(name, str)
        self.name = name

        assert isinstance(n, int) and (n >= 0 or
                                       n in (UP_TO_NEWLINE,
                                             TAKEN_FROM_ARGUMENT1,
                                             TAKEN_FROM_ARGUMENT4,
                                             TAKEN_FROM_ARGUMENT4U,
                                             TAKEN_FROM_ARGUMENT8U))
        self.n = n

        self.reader = reader

        assert isinstance(doc, str)
        self.doc = doc

from struct import unpack as _unpack

def read_uint1(f):
    r"""
    >>> import io
    >>> read_uint1(io.BytesIO(b'\xff'))
    255
    """

    data = f.read(1)
    if data:
        return data[0]
    raise ValueError("not enough data in stream to read uint1")

uint1 = ArgumentDescriptor(
            name='uint1',
            n=1,
            reader=read_uint1,
            doc="One-byte unsigned integer.")


def read_uint2(f):
    r"""
    >>> import io
    >>> read_uint2(io.BytesIO(b'\xff\x00'))
    255
    >>> read_uint2(io.BytesIO(b'\xff\xff'))
    65535
    """

    data = f.read(2)
    if len(data) == 2:
        return _unpack("<H", data)[0]
    raise ValueError("not enough data in stream to read uint2")

uint2 = ArgumentDescriptor(
            name='uint2',
            n=2,
            reader=read_uint2,
            doc="Two-byte unsigned integer, little-endian.")


def read_int4(f):
    r"""
    >>> import io
    >>> read_int4(io.BytesIO(b'\xff\x00\x00\x00'))
    255
    >>> read_int4(io.BytesIO(b'\x00\x00\x00\x80')) == -(2**31)
    True
    """

    data = f.read(4)
    if len(data) == 4:
        return _unpack("<i", data)[0]
    raise ValueError("not enough data in stream to read int4")

int4 = ArgumentDescriptor(
           name='int4',
           n=4,
           reader=read_int4,
           doc="Four-byte signed integer, little-endian, 2's complement.")


def read_uint4(f):
    r"""
    >>> import io
    >>> read_uint4(io.BytesIO(b'\xff\x00\x00\x00'))
    255
    >>> read_uint4(io.BytesIO(b'\x00\x00\x00\x80')) == 2**31
    True
    """

    data = f.read(4)
    if len(data) == 4:
        return _unpack("<I", data)[0]
    raise ValueError("not enough data in stream to read uint4")

uint4 = ArgumentDescriptor(
            name='uint4',
            n=4,
            reader=read_uint4,
            doc="Four-byte unsigned integer, little-endian.")


def read_uint8(f):
    r"""
    >>> import io
    >>> read_uint8(io.BytesIO(b'\xff\x00\x00\x00\x00\x00\x00\x00'))
    255
    >>> read_uint8(io.BytesIO(b'\xff' * 8)) == 2**64-1
    True
    """

    data = f.read(8)
    if len(data) == 8:
        return _unpack("<Q", data)[0]
    raise ValueError("not enough data in stream to read uint8")

uint8 = ArgumentDescriptor(
            name='uint8',
            n=8,
            reader=read_uint8,
            doc="Eight-byte unsigned integer, little-endian.")


def read_stringnl(f, decode=True, stripquotes=True):
    r"""
    >>> import io
    >>> read_stringnl(io.BytesIO(b"'abcd'\nefg\n"))
    'abcd'

    >>> read_stringnl(io.BytesIO(b"\n"))
    Traceback (most recent call last):
    ...
    ValueError: no string quotes around b''

    >>> read_stringnl(io.BytesIO(b"\n"), stripquotes=False)
    ''

    >>> read_stringnl(io.BytesIO(b"''\n"))
    ''

    >>> read_stringnl(io.BytesIO(b'"abcd"'))
    Traceback (most recent call last):
    ...
    ValueError: no newline found when trying to read stringnl

    Embedded escapes are undone in the result.
    >>> read_stringnl(io.BytesIO(br"'a\n\\b\x00c\td'" + b"\n'e'"))
    'a\n\\b\x00c\td'
    """

    data = f.readline()
    if not data.endswith(b'\n'):
        raise ValueError("no newline found when trying to read stringnl")
    data = data[:-1]    # lose the newline

    if stripquotes:
        for q in (b'"', b"'"):
            if data.startswith(q):
                if not data.endswith(q):
                    raise ValueError("strinq quote %r not found at both "
                                     "ends of %r" % (q, data))
                data = data[1:-1]
                break
        else:
            raise ValueError("no string quotes around %r" % data)

    if decode:
        data = codecs.escape_decode(data)[0].decode("ascii")
    return data

stringnl = ArgumentDescriptor(
               name='stringnl',
               n=UP_TO_NEWLINE,
               reader=read_stringnl,
               doc="""A newline-terminated string.

                   This is a repr-style string, with embedded escapes, and
                   bracketing quotes.
                   """)

def read_stringnl_noescape(f):
    return read_stringnl(f, stripquotes=False)

stringnl_noescape = ArgumentDescriptor(
                        name='stringnl_noescape',
                        n=UP_TO_NEWLINE,
                        reader=read_stringnl_noescape,
                        doc="""A newline-terminated string.

                        This is a str-style string, without embedded escapes,
                        or bracketing quotes.  It should consist solely of
                        printable ASCII characters.
                        """)

def read_stringnl_noescape_pair(f):
    r"""
    >>> import io
    >>> read_stringnl_noescape_pair(io.BytesIO(b"Queue\nEmpty\njunk"))
    'Queue Empty'
    """

    return "%s %s" % (read_stringnl_noescape(f), read_stringnl_noescape(f))

stringnl_noescape_pair = ArgumentDescriptor(
                             name='stringnl_noescape_pair',
                             n=UP_TO_NEWLINE,
                             reader=read_stringnl_noescape_pair,
                             doc="""A pair of newline-terminated strings.

                             These are str-style strings, without embedded
                             escapes, or bracketing quotes.  They should
                             consist solely of printable ASCII characters.
                             The pair is returned as a single string, with
                             a single blank separating the two strings.
                             """)


def read_string1(f):
    r"""
    >>> import io
    >>> read_string1(io.BytesIO(b"\x00"))
    ''
    >>> read_string1(io.BytesIO(b"\x03abcdef"))
    'abc'
    """

    n = read_uint1(f)
    assert n >= 0
    data = f.read(n)
    if len(data) == n:
        return data.decode("latin-1")
    raise ValueError("expected %d bytes in a string1, but only %d remain" %
                     (n, len(data)))

string1 = ArgumentDescriptor(
              name="string1",
              n=TAKEN_FROM_ARGUMENT1,
              reader=read_string1,
              doc="""A counted string.

              The first argument is a 1-byte unsigned int giving the number
              of bytes in the string, and the second argument is that many
              bytes.
              """)


def read_string4(f):
    r"""
    >>> import io
    >>> read_string4(io.BytesIO(b"\x00\x00\x00\x00abc"))
    ''
    >>> read_string4(io.BytesIO(b"\x03\x00\x00\x00abcdef"))
    'abc'
    >>> read_string4(io.BytesIO(b"\x00\x00\x00\x03abcdef"))
    Traceback (most recent call last):
    ...
    ValueError: expected 50331648 bytes in a string4, but only 6 remain
    """

    n = read_int4(f)
    if n < 0:
        raise ValueError("string4 byte count < 0: %d" % n)
    data = f.read(n)
    if len(data) == n:
        return data.decode("latin-1")
    raise ValueError("expected %d bytes in a string4, but only %d remain" %
                     (n, len(data)))

string4 = ArgumentDescriptor(
              name="string4",
              n=TAKEN_FROM_ARGUMENT4,
              reader=read_string4,
              doc="""A counted string.

              The first argument is a 4-byte little-endian signed int giving
              the number of bytes in the string, and the second argument is
              that many bytes.
              """)


def read_bytes1(f):
    r"""
    >>> import io
    >>> read_bytes1(io.BytesIO(b"\x00"))
    b''
    >>> read_bytes1(io.BytesIO(b"\x03abcdef"))
    b'abc'
    """

    n = read_uint1(f)
    assert n >= 0
    data = f.read(n)
    if len(data) == n:
        return data
    raise ValueError("expected %d bytes in a bytes1, but only %d remain" %
                     (n, len(data)))

bytes1 = ArgumentDescriptor(
              name="bytes1",
              n=TAKEN_FROM_ARGUMENT1,
              reader=read_bytes1,
              doc="""A counted bytes string.

              The first argument is a 1-byte unsigned int giving the number
              of bytes, and the second argument is that many bytes.
              """)


def read_bytes4(f):
    r"""
    >>> import io
    >>> read_bytes4(io.BytesIO(b"\x00\x00\x00\x00abc"))
    b''
    >>> read_bytes4(io.BytesIO(b"\x03\x00\x00\x00abcdef"))
    b'abc'
    >>> read_bytes4(io.BytesIO(b"\x00\x00\x00\x03abcdef"))
    Traceback (most recent call last):
    ...
    ValueError: expected 50331648 bytes in a bytes4, but only 6 remain
    """

    n = read_uint4(f)
    assert n >= 0
    if n > sys.maxsize:
        raise ValueError("bytes4 byte count > sys.maxsize: %d" % n)
    data = f.read(n)
    if len(data) == n:
        return data
    raise ValueError("expected %d bytes in a bytes4, but only %d remain" %
                     (n, len(data)))

bytes4 = ArgumentDescriptor(
              name="bytes4",
              n=TAKEN_FROM_ARGUMENT4U,
              reader=read_bytes4,
              doc="""A counted bytes string.

              The first argument is a 4-byte little-endian unsigned int giving
              the number of bytes, and the second argument is that many bytes.
              """)


def read_bytes8(f):
    r"""
    >>> import io, struct, sys
    >>> read_bytes8(io.BytesIO(b"\x00\x00\x00\x00\x00\x00\x00\x00abc"))
    b''
    >>> read_bytes8(io.BytesIO(b"\x03\x00\x00\x00\x00\x00\x00\x00abcdef"))
    b'abc'
    >>> bigsize8 = struct.pack("<Q", sys.maxsize//3)
    >>> read_bytes8(io.BytesIO(bigsize8 + b"abcdef"))  #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: expected ... bytes in a bytes8, but only 6 remain
    """

    n = read_uint8(f)
    assert n >= 0
    if n > sys.maxsize:
        raise ValueError("bytes8 byte count > sys.maxsize: %d" % n)
    data = f.read(n)
    if len(data) == n:
        return data
    raise ValueError("expected %d bytes in a bytes8, but only %d remain" %
                     (n, len(data)))

bytes8 = ArgumentDescriptor(
              name="bytes8",
              n=TAKEN_FROM_ARGUMENT8U,
              reader=read_bytes8,
              doc="""A counted bytes string.

              The first argument is an 8-byte little-endian unsigned int giving
              the number of bytes, and the second argument is that many bytes.
              """)


def read_bytearray8(f):
    r"""
    >>> import io, struct, sys
    >>> read_bytearray8(io.BytesIO(b"\x00\x00\x00\x00\x00\x00\x00\x00abc"))
    bytearray(b'')
    >>> read_bytearray8(io.BytesIO(b"\x03\x00\x00\x00\x00\x00\x00\x00abcdef"))
    bytearray(b'abc')
    >>> bigsize8 = struct.pack("<Q", sys.maxsize//3)
    >>> read_bytearray8(io.BytesIO(bigsize8 + b"abcdef"))  #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: expected ... bytes in a bytearray8, but only 6 remain
    """

    n = read_uint8(f)
    assert n >= 0
    if n > sys.maxsize:
        raise ValueError("bytearray8 byte count > sys.maxsize: %d" % n)
    data = f.read(n)
    if len(data) == n:
        return bytearray(data)
    raise ValueError("expected %d bytes in a bytearray8, but only %d remain" %
                     (n, len(data)))

bytearray8 = ArgumentDescriptor(
              name="bytearray8",
              n=TAKEN_FROM_ARGUMENT8U,
              reader=read_bytearray8,
              doc="""A counted bytearray.

              The first argument is an 8-byte little-endian unsigned int giving
              the number of bytes, and the second argument is that many bytes.
              """)

def read_unicodestringnl(f):
    r"""
    >>> import io
    >>> read_unicodestringnl(io.BytesIO(b"abc\\uabcd\njunk")) == 'abc\uabcd'
    True
    """

    data = f.readline()
    if not data.endswith(b'\n'):
        raise ValueError("no newline found when trying to read "
                         "unicodestringnl")
    data = data[:-1]    # lose the newline
    return str(data, 'raw-unicode-escape')

unicodestringnl = ArgumentDescriptor(
                      name='unicodestringnl',
                      n=UP_TO_NEWLINE,
                      reader=read_unicodestringnl,
                      doc="""A newline-terminated Unicode string.

                      This is raw-unicode-escape encoded, so consists of
                      printable ASCII characters, and may contain embedded
                      escape sequences.
                      """)


def read_unicodestring1(f):
    r"""
    >>> import io
    >>> s = 'abcd\uabcd'
    >>> enc = s.encode('utf-8')
    >>> enc
    b'abcd\xea\xaf\x8d'
    >>> n = bytes([len(enc)])  # little-endian 1-byte length
    >>> t = read_unicodestring1(io.BytesIO(n + enc + b'junk'))
    >>> s == t
    True

    >>> read_unicodestring1(io.BytesIO(n + enc[:-1]))
    Traceback (most recent call last):
    ...
    ValueError: expected 7 bytes in a unicodestring1, but only 6 remain
    """

    n = read_uint1(f)
    assert n >= 0
    data = f.read(n)
    if len(data) == n:
        return str(data, 'utf-8', 'surrogatepass')
    raise ValueError("expected %d bytes in a unicodestring1, but only %d "
                     "remain" % (n, len(data)))

unicodestring1 = ArgumentDescriptor(
                    name="unicodestring1",
                    n=TAKEN_FROM_ARGUMENT1,
                    reader=read_unicodestring1,
                    doc="""A counted Unicode string.

                    The first argument is a 1-byte little-endian signed int
                    giving the number of bytes in the string, and the second
                    argument-- the UTF-8 encoding of the Unicode string --
                    contains that many bytes.
                    """)


def read_unicodestring4(f):
    r"""
    >>> import io
    >>> s = 'abcd\uabcd'
    >>> enc = s.encode('utf-8')
    >>> enc
    b'abcd\xea\xaf\x8d'
    >>> n = bytes([len(enc), 0, 0, 0])  # little-endian 4-byte length
    >>> t = read_unicodestring4(io.BytesIO(n + enc + b'junk'))
    >>> s == t
    True

    >>> read_unicodestring4(io.BytesIO(n + enc[:-1]))
    Traceback (most recent call last):
    ...
    ValueError: expected 7 bytes in a unicodestring4, but only 6 remain
    """

    n = read_uint4(f)
    assert n >= 0
    if n > sys.maxsize:
        raise ValueError("unicodestring4 byte count > sys.maxsize: %d" % n)
    data = f.read(n)
    if len(data) == n:
        return str(data, 'utf-8', 'surrogatepass')
    raise ValueError("expected %d bytes in a unicodestring4, but only %d "
                     "remain" % (n, len(data)))

unicodestring4 = ArgumentDescriptor(
                    name="unicodestring4",
                    n=TAKEN_FROM_ARGUMENT4U,
                    reader=read_unicodestring4,
                    doc="""A counted Unicode string.

                    The first argument is a 4-byte little-endian signed int
                    giving the number of bytes in the string, and the second
                    argument-- the UTF-8 encoding of the Unicode string --
                    contains that many bytes.
                    """)


def read_unicodestring8(f):
    r"""
    >>> import io
    >>> s = 'abcd\uabcd'
    >>> enc = s.encode('utf-8')
    >>> enc
    b'abcd\xea\xaf\x8d'
    >>> n = bytes([len(enc)]) + b'\0' * 7  # little-endian 8-byte length
    >>> t = read_unicodestring8(io.BytesIO(n + enc + b'junk'))
    >>> s == t
    True

    >>> read_unicodestring8(io.BytesIO(n + enc[:-1]))
    Traceback (most recent call last):
    ...
    ValueError: expected 7 bytes in a unicodestring8, but only 6 remain
    """

    n = read_uint8(f)
    assert n >= 0
    if n > sys.maxsize:
        raise ValueError("unicodestring8 byte count > sys.maxsize: %d" % n)
    data = f.read(n)
    if len(data) == n:
        return str(data, 'utf-8', 'surrogatepass')
    raise ValueError("expected %d bytes in a unicodestring8, but only %d "
                     "remain" % (n, len(data)))

unicodestring8 = ArgumentDescriptor(
                    name="unicodestring8",
                    n=TAKEN_FROM_ARGUMENT8U,
                    reader=read_unicodestring8,
                    doc="""A counted Unicode string.

                    The first argument is an 8-byte little-endian signed int
                    giving the number of bytes in the string, and the second
                    argument-- the UTF-8 encoding of the Unicode string --
                    contains that many bytes.
                    """)


def read_decimalnl_short(f):
    r"""
    >>> import io
    >>> read_decimalnl_short(io.BytesIO(b"1234\n56"))
    1234

    >>> read_decimalnl_short(io.BytesIO(b"1234L\n56"))
    Traceback (most recent call last):
    ...
    ValueError: invalid literal for int() with base 10: b'1234L'
    """

    s = read_stringnl(f, decode=False, stripquotes=False)

    # There's a hack for True and False here.
    if s == b"00":
        return False
    elif s == b"01":
        return True

    return int(s)

def read_decimalnl_long(f):
    r"""
    >>> import io

    >>> read_decimalnl_long(io.BytesIO(b"1234L\n56"))
    1234

    >>> read_decimalnl_long(io.BytesIO(b"123456789012345678901234L\n6"))
    123456789012345678901234
    """

    s = read_stringnl(f, decode=False, stripquotes=False)
    if s[-1:] == b'L':
        s = s[:-1]
    return int(s)


decimalnl_short = ArgumentDescriptor(
                      name='decimalnl_short',
                      n=UP_TO_NEWLINE,
                      reader=read_decimalnl_short,
                      doc="""A newline-terminated decimal integer literal.

                          This never has a trailing 'L', and the integer fit
                          in a short Python int on the box where the pickle
                          was written -- but there's no guarantee it will fit
                          in a short Python int on the box where the pickle
                          is read.
                          """)

decimalnl_long = ArgumentDescriptor(
                     name='decimalnl_long',
                     n=UP_TO_NEWLINE,
                     reader=read_decimalnl_long,
                     doc="""A newline-terminated decimal integer literal.

                         This has a trailing 'L', and can represent integers
                         of any size.
                         """)


def read_floatnl(f):
    r"""
    >>> import io
    >>> read_floatnl(io.BytesIO(b"-1.25\n6"))
    -1.25
    """
    s = read_stringnl(f, decode=False, stripquotes=False)
    return float(s)

floatnl = ArgumentDescriptor(
              name='floatnl',
              n=UP_TO_NEWLINE,
              reader=read_floatnl,
              doc="""A newline-terminated decimal floating literal.

              In general this requires 17 significant digits for roundtrip
              identity, and pickling then unpickling infinities, NaNs, and
              minus zero doesn't work across boxes, or on some boxes even
              on itself (e.g., Windows can't read the strings it produces
              for infinities or NaNs).
              """)

def read_float8(f):
    r"""
    >>> import io, struct
    >>> raw = struct.pack(">d", -1.25)
    >>> raw
    b'\xbf\xf4\x00\x00\x00\x00\x00\x00'
    >>> read_float8(io.BytesIO(raw + b"\n"))
    -1.25
    """

    data = f.read(8)
    if len(data) == 8:
        return _unpack(">d", data)[0]
    raise ValueError("not enough data in stream to read float8")


float8 = ArgumentDescriptor(
             name='float8',
             n=8,
             reader=read_float8,
             doc="""An 8-byte binary representation of a float, big-endian.

             The format is unique to Python, and shared with the struct
             module (format string '>d') "in theory" (the struct and pickle
             implementations don't share the code -- they should).  It's
             strongly related to the IEEE-754 double format, and, in normal
             cases, is in fact identical to the big-endian 754 double format.
             On other boxes the dynamic range is limited to that of a 754
             double, and "add a half and chop" rounding is used to reduce
             the precision to 53 bits.  However, even on a 754 box,
             infinities, NaNs, and minus zero may not be handled correctly
             (may not survive roundtrip pickling intact).
             """)

# Protocol 2 formats

from pickle import decode_long

def read_long1(f):
    r"""
    >>> import io
    >>> read_long1(io.BytesIO(b"\x00"))
    0
    >>> read_long1(io.BytesIO(b"\x02\xff\x00"))
    255
    >>> read_long1(io.BytesIO(b"\x02\xff\x7f"))
    32767
    >>> read_long1(io.BytesIO(b"\x02\x00\xff"))
    -256
    >>> read_long1(io.BytesIO(b"\x02\x00\x80"))
    -32768
    """

    n = read_uint1(f)
    data = f.read(n)
    if len(data) != n:
        raise ValueError("not enough data in stream to read long1")
    return decode_long(data)

long1 = ArgumentDescriptor(
    name="long1",
    n=TAKEN_FROM_ARGUMENT1,
    reader=read_long1,
    doc="""A binary long, little-endian, using 1-byte size.

    This first reads one byte as an unsigned size, then reads that
    many bytes and interprets them as a little-endian 2's-complement long.
    If the size is 0, that's taken as a shortcut for the long 0L.
    """)

def read_long4(f):
    r"""
    >>> import io
    >>> read_long4(io.BytesIO(b"\x02\x00\x00\x00\xff\x00"))
    255
    >>> read_long4(io.BytesIO(b"\x02\x00\x00\x00\xff\x7f"))
    32767
    >>> read_long4(io.BytesIO(b"\x02\x00\x00\x00\x00\xff"))
    -256
    >>> read_long4(io.BytesIO(b"\x02\x00\x00\x00\x00\x80"))
    -32768
    >>> read_long1(io.BytesIO(b"\x00\x00\x00\x00"))
    0
    """

    n = read_int4(f)
    if n < 0:
        raise ValueError("long4 byte count < 0: %d" % n)
    data = f.read(n)
    if len(data) != n:
        raise ValueError("not enough data in stream to read long4")
    return decode_long(data)

long4 = ArgumentDescriptor(
    name="long4",
    n=TAKEN_FROM_ARGUMENT4,
    reader=read_long4,
    doc="""A binary representation of a long, little-endian.

    This first reads four bytes as a signed size (but requires the
    size to be >= 0), then reads that many bytes and interprets them
    as a little-endian 2's-complement long.  If the size is 0, that's taken
    as a shortcut for the int 0, although LONG1 should really be used
    then instead (and in any case where # of bytes < 256).
    """)


##############################################################################
# Object descriptors.  The stack used by the pickle machine holds objects,
# and in the stack_before and stack_after attributes of OpcodeInfo
# descriptors we need names to describe the various types of objects that can
# appear on the stack.

class StackObject(object):
    __slots__ = (
        # name of descriptor record, for info only
        'name',

        # type of object, or tuple of type objects (meaning the object can
        # be of any type in the tuple)
        'obtype',

        # human-readable docs for this kind of stack object; a string
        'doc',
    )

    def __init__(self, name, obtype, doc):
        assert isinstance(name, str)
        self.name = name

        assert isinstance(obtype, type) or isinstance(obtype, tuple)
        if isinstance(obtype, tuple):
            for contained in obtype:
                assert isinstance(contained, type)
        self.obtype = obtype

        assert isinstance(doc, str)
        self.doc = doc

    def __repr__(self):
        return self.name


pyint = pylong = StackObject(
    name='int',
    obtype=int,
    doc="A Python integer object.")

pyinteger_or_bool = StackObject(
    name='int_or_bool',
    obtype=(int, bool),
    doc="A Python integer or boolean object.")

pybool = StackObject(
    name='bool',
    obtype=bool,
    doc="A Python boolean object.")

pyfloat = StackObject(
    name='float',
    obtype=float,
    doc="A Python float object.")

pybytes_or_str = pystring = StackObject(
    name='bytes_or_str',
    obtype=(bytes, str),
    doc="A Python bytes or (Unicode) string object.")

pybytes = StackObject(
    name='bytes',
    obtype=bytes,
    doc="A Python bytes object.")

pybytearray = StackObject(
    name='bytearray',
    obtype=bytearray,
    doc="A Python bytearray object.")

pyunicode = StackObject(
    name='str',
    obtype=str,
    doc="A Python (Unicode) string object.")

pynone = StackObject(
    name="None",
    obtype=type(None),
    doc="The Python None object.")

pytuple = StackObject(
    name="tuple",
    obtype=tuple,
    doc="A Python tuple object.")

pylist = StackObject(
    name="list",
    obtype=list,
    doc="A Python list object.")

pydict = StackObject(
    name="dict",
    obtype=dict,
    doc="A Python dict object.")

pyset = StackObject(
    name="set",
    obtype=set,
    doc="A Python set object.")

pyfrozenset = StackObject(
    name="frozenset",
    obtype=set,
    doc="A Python frozenset object.")

pybuffer = StackObject(
    name='buffer',
    obtype=object,
    doc="A Python buffer-like object.")

anyobject = StackObject(
    name='any',
    obtype=object,
    doc="Any kind of object whatsoever.")

markobject = StackObject(
    name="mark",
    obtype=StackObject,
    doc="""'The mark' is a unique object.

Opcodes that operate on a variable number of objects
generally don't embed the count of objects in the opcode,
or pull it off the stack.  Instead the MARK opcode is used
to push a special marker object on the stack, and then
some other opcodes grab all the objects from the top of
the stack down to (but not including) the topmost marker
object.
""")

stackslice = StackObject(
    name="stackslice",
    obtype=StackObject,
    doc="""An object representing a contiguous slice of the stack.

This is used in conjunction with markobject, to represent all
of the stack following the topmost markobject.  For example,
the POP_MARK opcode changes the stack from

    [..., markobject, stackslice]
to
    [...]

No matter how many object are on the stack after the topmost
markobject, POP_MARK gets rid of all of them (including the
topmost markobject too).
""")

##############################################################################
# Descriptors for pickle opcodes.

class OpcodeInfo(object):

    __slots__ = (
        # symbolic name of opcode; a string
        'name',

        # the code used in a bytestream to represent the opcode; a
        # one-character string
        'code',

        # If the opcode has an argument embedded in the byte string, an
        # instance of ArgumentDescriptor specifying its type.  Note that
        # arg.reader(s) can be used to read and decode the argument from
        # the bytestream s, and arg.doc documents the format of the raw
        # argument bytes.  If the opcode doesn't have an argument embedded
        # in the bytestream, arg should be None.
        'arg',

        # what the stack looks like before this opcode runs; a list
        'stack_before',

        # what the stack looks like after this opcode runs; a list
        'stack_after',

        # the protocol number in which this opcode was introduced; an int
        'proto',

        # human-readable docs for this opcode; a string
        'doc',
    )

    def __init__(self, name, code, arg,
                 stack_before, stack_after, proto, doc):
        assert isinstance(name, str)
        self.name = name

        assert isinstance(code, str)
        assert len(code) == 1
        self.code = code

        assert arg is None or isinstance(arg, ArgumentDescriptor)
        self.arg = arg

        assert isinstance(stack_before, list)
        for x in stack_before:
            assert isinstance(x, StackObject)
        self.stack_before = stack_before

        assert isinstance(stack_after, list)
        for x in stack_after:
            assert isinstance(x, StackObject)
        self.stack_after = stack_after

        assert isinstance(proto, int) and 0 <= proto <= pickle.HIGHEST_PROTOCOL
        self.proto = proto

        assert isinstance(doc, str)
        self.doc = doc

I = OpcodeInfo
opcodes = [

    # Ways to spell integers.

    I(name='INT',
      code='I',
      arg=decimalnl_short,
      stack_before=[],
      stack_after=[pyinteger_or_bool],
      proto=0,
      doc="""Push an integer or bool.

      The argument is a newline-terminated decimal literal string.

      The intent may have been that this always fit in a short Python int,
      but INT can be generated in pickles written on a 64-bit box that
      require a Python long on a 32-bit box.  The difference between this
      and LONG then is that INT skips a trailing 'L', and produces a short
      int whenever possible.

      Another difference is due to that, when bool was introduced as a
      distinct type in 2.3, builtin names True and False were also added to
      2.2.2, mapping to ints 1 and 0.  For compatibility in both directions,
      True gets pickled as INT + "I01\\n", and False as INT + "I00\\n".
      Leading zeroes are never produced for a genuine integer.  The 2.3
      (and later) unpicklers special-case these and return bool instead;
      earlier unpicklers ignore the leading "0" and return the int.
      """),

    I(name='BININT',
      code='J',
      arg=int4,
      stack_before=[],
      stack_after=[pyint],
      proto=1,
      doc="""Push a four-byte signed integer.

      This handles the full range of Python (short) integers on a 32-bit
      box, directly as binary bytes (1 for the opcode and 4 for the integer).
      If the integer is non-negative and fits in 1 or 2 bytes, pickling via
      BININT1 or BININT2 saves space.
      """),

    I(name='BININT1',
      code='K',
      arg=uint1,
      stack_before=[],
      stack_after=[pyint],
      proto=1,
      doc="""Push a one-byte unsigned integer.

      This is a space optimization for pickling very small non-negative ints,
      in range(256).
      """),

    I(name='BININT2',
      code='M',
      arg=uint2,
      stack_before=[],
      stack_after=[pyint],
      proto=1,
      doc="""Push a two-byte unsigned integer.

      This is a space optimization for pickling small positive ints, in
      range(256, 2**16).  Integers in range(256) can also be pickled via
      BININT2, but BININT1 instead saves a byte.
      """),

    I(name='LONG',
      code='L',
      arg=decimalnl_long,
      stack_before=[],
      stack_after=[pyint],
      proto=0,
      doc="""Push a long integer.

      The same as INT, except that the literal ends with 'L', and always
      unpickles to a Python long.  There doesn't seem a real purpose to the
      trailing 'L'.

      Note that LONG takes time quadratic in the number of digits when
      unpickling (this is simply due to the nature of decimal->binary
      conversion).  Proto 2 added linear-time (in C; still quadratic-time
      in Python) LONG1 and LONG4 opcodes.
      """),

    I(name="LONG1",
      code='\x8a',
      arg=long1,
      stack_before=[],
      stack_after=[pyint],
      proto=2,
      doc="""Long integer using one-byte length.

      A more efficient encoding of a Python long; the long1 encoding
      says it all."""),

    I(name="LONG4",
      code='\x8b',
      arg=long4,
      stack_before=[],
      stack_after=[pyint],
      proto=2,
      doc="""Long integer using found-byte length.

      A more efficient encoding of a Python long; the long4 encoding
      says it all."""),

    # Ways to spell strings (8-bit, not Unicode).

    I(name='STRING',
      code='S',
      arg=stringnl,
      stack_before=[],
      stack_after=[pybytes_or_str],
      proto=0,
      doc="""Push a Python string object.

      The argument is a repr-style string, with bracketing quote characters,
      and perhaps embedded escapes.  The argument extends until the next
      newline character.  These are usually decoded into a str instance
      using the encoding given to the Unpickler constructor. or the default,
      'ASCII'.  If the encoding given was 'bytes' however, they will be
      decoded as bytes object instead.
      """),

    I(name='BINSTRING',
      code='T',
      arg=string4,
      stack_before=[],
      stack_after=[pybytes_or_str],
      proto=1,
      doc="""Push a Python string object.

      There are two arguments: the first is a 4-byte little-endian
      signed int giving the number of bytes in the string, and the
      second is that many bytes, which are taken literally as the string
      content.  These are usually decoded into a str instance using the
      encoding given to the Unpickler constructor. or the default,
      'ASCII'.  If the encoding given was 'bytes' however, they will be
      decoded as bytes object instead.
      """),

    I(name='SHORT_BINSTRING',
      code='U',
      arg=string1,
      stack_before=[],
      stack_after=[pybytes_or_str],
      proto=1,
      doc="""Push a Python string object.

      There are two arguments: the first is a 1-byte unsigned int giving
      the number of bytes in the string, and the second is that many
      bytes, which are taken literally as the string content.  These are
      usually decoded into a str instance using the encoding given to
      the Unpickler constructor. or the default, 'ASCII'.  If the
      encoding given was 'bytes' however, they will be decoded as bytes
      object instead.
      """),

    # Bytes (protocol 3 and higher)

    I(name='BINBYTES',
      code='B',
      arg=bytes4,
      stack_before=[],
      stack_after=[pybytes],
      proto=3,
      doc="""Push a Python bytes object.

      There are two arguments:  the first is a 4-byte little-endian unsigned int
      giving the number of bytes, and the second is that many bytes, which are
      taken literally as the bytes content.
      """),

    I(name='SHORT_BINBYTES',
      code='C',
      arg=bytes1,
      stack_before=[],
      stack_after=[pybytes],
      proto=3,
      doc="""Push a Python bytes object.

      There are two arguments:  the first is a 1-byte unsigned int giving
      the number of bytes, and the second is that many bytes, which are taken
      literally as the string content.
      """),

    I(name='BINBYTES8',
      code='\x8e',
      arg=bytes8,
      stack_before=[],
      stack_after=[pybytes],
      proto=4,
      doc="""Push a Python bytes object.

      There are two arguments:  the first is an 8-byte unsigned int giving
      the number of bytes in the string, and the second is that many bytes,
      which are taken literally as the string content.
      """),

    # Bytearray (protocol 5 and higher)

    I(name='BYTEARRAY8',
      code='\x96',
      arg=bytearray8,
      stack_before=[],
      stack_after=[pybytearray],
      proto=5,
      doc="""Push a Python bytearray object.

      There are two arguments:  the first is an 8-byte unsigned int giving
      the number of bytes in the bytearray, and the second is that many bytes,
      which are taken literally as the bytearray content.
      """),

    # Out-of-band buffer (protocol 5 and higher)

    I(name='NEXT_BUFFER',
      code='\x97',
      arg=None,
      stack_before=[],
      stack_after=[pybuffer],
      proto=5,
      doc="Push an out-of-band buffer object."),

    I(name='READONLY_BUFFER',
      code='\x98',
      arg=None,
      stack_before=[pybuffer],
      stack_after=[pybuffer],
      proto=5,
      doc="Make an out-of-band buffer object read-only."),

    # Ways to spell None.

    I(name='NONE',
      code='N',
      arg=None,
      stack_before=[],
      stack_after=[pynone],
      proto=0,
      doc="Push None on the stack."),

    # Ways to spell bools, starting with proto 2.  See INT for how this was
    # done before proto 2.

    I(name='NEWTRUE',
      code='\x88',
      arg=None,
      stack_before=[],
      stack_after=[pybool],
      proto=2,
      doc="Push True onto the stack."),

    I(name='NEWFALSE',
      code='\x89',
      arg=None,
      stack_before=[],
      stack_after=[pybool],
      proto=2,
      doc="Push False onto the stack."),

    # Ways to spell Unicode strings.

    I(name='UNICODE',
      code='V',
      arg=unicodestringnl,
      stack_before=[],
      stack_after=[pyunicode],
      proto=0,  # this may be pure-text, but it's a later addition
      doc="""Push a Python Unicode string object.

      The argument is a raw-unicode-escape encoding of a Unicode string,
      and so may contain embedded escape sequences.  The argument extends
      until the next newline character.
      """),

    I(name='SHORT_BINUNICODE',
      code='\x8c',
      arg=unicodestring1,
      stack_before=[],
      stack_after=[pyunicode],
      proto=4,
      doc="""Push a Python Unicode string object.

      There are two arguments:  the first is a 1-byte little-endian signed int
      giving the number of bytes in the string.  The second is that many
      bytes, and is the UTF-8 encoding of the Unicode string.
      """),

    I(name='BINUNICODE',
      code='X',
      arg=unicodestring4,
      stack_before=[],
      stack_after=[pyunicode],
      proto=1,
      doc="""Push a Python Unicode string object.

      There are two arguments:  the first is a 4-byte little-endian unsigned int
      giving the number of bytes in the string.  The second is that many
      bytes, and is the UTF-8 encoding of the Unicode string.
      """),

    I(name='BINUNICODE8',
      code='\x8d',
      arg=unicodestring8,
      stack_before=[],
      stack_after=[pyunicode],
      proto=4,
      doc="""Push a Python Unicode string object.

      There are two arguments:  the first is an 8-byte little-endian signed int
      giving the number of bytes in the string.  The second is that many
      bytes, and is the UTF-8 encoding of the Unicode string.
      """),

    # Ways to spell floats.

    I(name='FLOAT',
      code='F',
      arg=floatnl,
      stack_before=[],
      stack_after=[pyfloat],
      proto=0,
      doc="""Newline-terminated decimal float literal.

      The argument is repr(a_float), and in general requires 17 significant
      digits for roundtrip conversion to be an identity (this is so for
      IEEE-754 double precision values, which is what Python float maps to
      on most boxes).

      In general, FLOAT cannot be used to transport infinities, NaNs, or
      minus zero across boxes (or even on a single box, if the platform C
      library can't read the strings it produces for such things -- Windows
      is like that), but may do less damage than BINFLOAT on boxes with
      greater precision or dynamic range than IEEE-754 double.
      """),

    I(name='BINFLOAT',
      code='G',
      arg=float8,
      stack_before=[],
      stack_after=[pyfloat],
      proto=1,
      doc="""Float stored in binary form, with 8 bytes of data.

      This generally requires less than half the space of FLOAT encoding.
      In general, BINFLOAT cannot be used to transport infinities, NaNs, or
      minus zero, raises an exception if the exponent exceeds the range of
      an IEEE-754 double, and retains no more than 53 bits of precision (if
      there are more than that, "add a half and chop" rounding is used to
      cut it back to 53 significant bits).
      """),

    # Ways to build lists.

    I(name='EMPTY_LIST',
      code=']',
      arg=None,
      stack_before=[],
      stack_after=[pylist],
      proto=1,
      doc="Push an empty list."),

    I(name='APPEND',
      code='a',
      arg=None,
      stack_before=[pylist, anyobject],
      stack_after=[pylist],
      proto=0,
      doc="""Append an object to a list.

      Stack before:  ... pylist anyobject
      Stack after:   ... pylist+[anyobject]

      although pylist is really extended in-place.
      """),

    I(name='APPENDS',
      code='e',
      arg=None,
      stack_before=[pylist, markobject, stackslice],
      stack_after=[pylist],
      proto=1,
      doc="""Extend a list by a slice of stack objects.

      Stack before:  ... pylist markobject stackslice
      Stack after:   ... pylist+stackslice

      although pylist is really extended in-place.
      """),

    I(name='LIST',
      code='l',
      arg=None,
      stack_before=[markobject, stackslice],
      stack_after=[pylist],
      proto=0,
      doc="""Build a list out of the topmost stack slice, after markobject.

      All the stack entries following the topmost markobject are placed into
      a single Python list, which single list object replaces all of the
      stack from the topmost markobject onward.  For example,

      Stack before: ... markobject 1 2 3 'abc'
      Stack after:  ... [1, 2, 3, 'abc']
      """),

    # Ways to build tuples.

    I(name='EMPTY_TUPLE',
      code=')',
      arg=None,
      stack_before=[],
      stack_after=[pytuple],
      proto=1,
      doc="Push an empty tuple."),

    I(name='TUPLE',
      code='t',
      arg=None,
      stack_before=[markobject, stackslice],
      stack_after=[pytuple],
      proto=0,
      doc="""Build a tuple out of the topmost stack slice, after markobject.

      All the stack entries following the topmost markobject are placed into
      a single Python tuple, which single tuple object replaces all of the
      stack from the topmost markobject onward.  For example,

      Stack before: ... markobject 1 2 3 'abc'
      Stack after:  ... (1, 2, 3, 'abc')
      """),

    I(name='TUPLE1',
      code='\x85',
      arg=None,
      stack_before=[anyobject],
      stack_after=[pytuple],
      proto=2,
      doc="""Build a one-tuple out of the topmost item on the stack.

      This code pops one value off the stack and pushes a tuple of
      length 1 whose one item is that value back onto it.  In other
      words:

          stack[-1] = tuple(stack[-1:])
      """),

    I(name='TUPLE2',
      code='\x86',
      arg=None,
      stack_before=[anyobject, anyobject],
      stack_after=[pytuple],
      proto=2,
      doc="""Build a two-tuple out of the top two items on the stack.

      This code pops two values off the stack and pushes a tuple of
      length 2 whose items are those values back onto it.  In other
      words:

          stack[-2:] = [tuple(stack[-2:])]
      """),

    I(name='TUPLE3',
      code='\x87',
      arg=None,
      stack_before=[anyobject, anyobject, anyobject],
      stack_after=[pytuple],
      proto=2,
      doc="""Build a three-tuple out of the top three items on the stack.

      This code pops three values off the stack and pushes a tuple of
      length 3 whose items are those values back onto it.  In other
      words:

          stack[-3:] = [tuple(stack[-3:])]
      """),

    # Ways to build dicts.

    I(name='EMPTY_DICT',
      code='}',
      arg=None,
      stack_before=[],
      stack_after=[pydict],
      proto=1,
      doc="Push an empty dict."),

    I(name='DICT',
      code='d',
      arg=None,
      stack_before=[markobject, stackslice],
      stack_after=[pydict],
      proto=0,
      doc="""Build a dict out of the topmost stack slice, after markobject.

      All the stack entries following the topmost markobject are placed into
      a single Python dict, which single dict object replaces all of the
      stack from the topmost markobject onward.  The stack slice alternates
      key, value, key, value, ....  For example,

      Stack before: ... markobject 1 2 3 'abc'
      Stack after:  ... {1: 2, 3: 'abc'}
      """),

    I(name='SETITEM',
      code='s',
      arg=None,
      stack_before=[pydict, anyobject, anyobject],
      stack_after=[pydict],
      proto=0,
      doc="""Add a key+value pair to an existing dict.

      Stack before:  ... pydict key value
      Stack after:   ... pydict

      where pydict has been modified via pydict[key] = value.
      """),

    I(name='SETITEMS',
      code='u',
      arg=None,
      stack_before=[pydict, markobject, stackslice],
      stack_after=[pydict],
      proto=1,
      doc="""Add an arbitrary number of key+value pairs to an existing dict.

      The slice of the stack following the topmost markobject is taken as
      an alternating sequence of keys and values, added to the dict
      immediately under the topmost markobject.  Everything at and after the
      topmost markobject is popped, leaving the mutated dict at the top
      of the stack.

      Stack before:  ... pydict markobject key_1 value_1 ... key_n value_n
      Stack after:   ... pydict

      where pydict has been modified via pydict[key_i] = value_i for i in
      1, 2, ..., n, and in that order.
      """),

    # Ways to build sets

    I(name='EMPTY_SET',
      code='\x8f',
      arg=None,
      stack_before=[],
      stack_after=[pyset],
      proto=4,
      doc="Push an empty set."),

    I(name='ADDITEMS',
      code='\x90',
      arg=None,
      stack_before=[pyset, markobject, stackslice],
      stack_after=[pyset],
      proto=4,
      doc="""Add an arbitrary number of items to an existing set.

      The slice of the stack following the topmost markobject is taken as
      a sequence of items, added to the set immediately under the topmost
      markobject.  Everything at and after the topmost markobject is popped,
      leaving the mutated set at the top of the stack.

      Stack before:  ... pyset markobject item_1 ... item_n
      Stack after:   ... pyset

      where pyset has been modified via pyset.add(item_i) = item_i for i in
      1, 2, ..., n, and in that order.
      """),

    # Way to build frozensets

    I(name='FROZENSET',
      code='\x91',
      arg=None,
      stack_before=[markobject, stackslice],
      stack_after=[pyfrozenset],
      proto=4,
      doc="""Build a frozenset out of the topmost slice, after markobject.

      All the stack entries following the topmost markobject are placed into
      a single Python frozenset, which single frozenset object replaces all
      of the stack from the topmost markobject onward.  For example,

      Stack before: ... markobject 1 2 3
      Stack after:  ... frozenset({1, 2, 3})
      """),

    # Stack manipulation.

    I(name='POP',
      code='0',
      arg=None,
      stack_before=[anyobject],
      stack_after=[],
      proto=0,
      doc="Discard the top stack item, shrinking the stack by one item."),

    I(name='DUP',
      code='2',
      arg=None,
      stack_before=[anyobject],
      stack_after=[anyobject, anyobject],
      proto=0,
      doc="Push the top stack item onto the stack again, duplicating it."),

    I(name='MARK',
      code='(',
      arg=None,
      stack_before=[],
      stack_after=[markobject],
      proto=0,
      doc="""Push markobject onto the stack.

      markobject is a unique object, used by other opcodes to identify a
      region of the stack containing a variable number of objects for them
      to work on.  See markobject.doc for more detail.
      """),

    I(name='POP_MARK',
      code='1',
      arg=None,
      stack_before=[markobject, stackslice],
      stack_after=[],
      proto=1,
      doc="""Pop all the stack objects at and above the topmost markobject.

      When an opcode using a variable number of stack objects is done,
      POP_MARK is used to remove those objects, and to remove the markobject
      that delimited their starting position on the stack.
      """),

    # Memo manipulation.  There are really only two operations (get and put),
    # each in all-text, "short binary", and "long binary" flavors.

    I(name='GET',
      code='g',
      arg=decimalnl_short,
      stack_before=[],
      stack_after=[anyobject],
      proto=0,
      doc="""Read an object from the memo and push it on the stack.

      The index of the memo object to push is given by the newline-terminated
      decimal string following.  BINGET and LONG_BINGET are space-optimized
      versions.
      """),

    I(name='BINGET',
      code='h',
      arg=uint1,
      stack_before=[],
      stack_after=[anyobject],
      proto=1,
      doc="""Read an object from the memo and push it on the stack.

      The index of the memo object to push is given by the 1-byte unsigned
      integer following.
      """),

    I(name='LONG_BINGET',
      code='j',
      arg=uint4,
      stack_before=[],
      stack_after=[anyobject],
      proto=1,
      doc="""Read an object from the memo and push it on the stack.

      The index of the memo object to push is given by the 4-byte unsigned
      little-endian integer following.
      """),

    I(name='PUT',
      code='p',
      arg=decimalnl_short,
      stack_before=[],
      stack_after=[],
      proto=0,
      doc="""Store the stack top into the memo.  The stack is not popped.

      The index of the memo location to write into is given by the newline-
      terminated decimal string following.  BINPUT and LONG_BINPUT are
      space-optimized versions.
      """),

    I(name='BINPUT',
      code='q',
      arg=uint1,
      stack_before=[],
      stack_after=[],
      proto=1,
      doc="""Store the stack top into the memo.  The stack is not popped.

      The index of the memo location to write into is given by the 1-byte
      unsigned integer following.
      """),

    I(name='LONG_BINPUT',
      code='r',
      arg=uint4,
      stack_before=[],
      stack_after=[],
      proto=1,
      doc="""Store the stack top into the memo.  The stack is not popped.

      The index of the memo location to write into is given by the 4-byte
      unsigned little-endian integer following.
      """),

    I(name='MEMOIZE',
      code='\x94',
      arg=None,
      stack_before=[anyobject],
      stack_after=[anyobject],
      proto=4,
      doc="""Store the stack top into the memo.  The stack is not popped.

      The index of the memo location to write is the number of
      elements currently present in the memo.
      """),

    # Access the extension registry (predefined objects).  Akin to the GET
    # family.

    I(name='EXT1',
      code='\x82',
      arg=uint1,
      stack_before=[],
      stack_after=[anyobject],
      proto=2,
      doc="""Extension code.

      This code and the similar EXT2 and EXT4 allow using a registry
      of popular objects that are pickled by name, typically classes.
      It is envisioned that through a global negotiation and
      registration process, third parties can set up a mapping between
      ints and object names.

      In order to guarantee pickle interchangeability, the extension
      code registry ought to be global, although a range of codes may
      be reserved for private use.

      EXT1 has a 1-byte integer argument.  This is used to index into the
      extension registry, and the object at that index is pushed on the stack.
      """),

    I(name='EXT2',
      code='\x83',
      arg=uint2,
      stack_before=[],
      stack_after=[anyobject],
      proto=2,
      doc="""Extension code.

      See EXT1.  EXT2 has a two-byte integer argument.
      """),

    I(name='EXT4',
      code='\x84',
      arg=int4,
      stack_before=[],
      stack_after=[anyobject],
      proto=2,
      doc="""Extension code.

      See EXT1.  EXT4 has a four-byte integer argument.
      """),

    # Push a class object, or module function, on the stack, via its module
    # and name.

    I(name='GLOBAL',
      code='c',
      arg=stringnl_noescape_pair,
      stack_before=[],
      stack_after=[anyobject],
      proto=0,
      doc="""Push a global object (module.attr) on the stack.

      Two newline-terminated strings follow the GLOBAL opcode.  The first is
      taken as a module name, and the second as a class name.  The class
      object module.class is pushed on the stack.  More accurately, the
      object returned by self.find_class(module, class) is pushed on the
      stack, so unpickling subclasses can override this form of lookup.
      """),

    I(name='STACK_GLOBAL',
      code='\x93',
      arg=None,
      stack_before=[pyunicode, pyunicode],
      stack_after=[anyobject],
      proto=4,
      doc="""Push a global object (module.attr) on the stack.
      """),

    # Ways to build objects of classes pickle doesn't know about directly
    # (user-defined classes).  I despair of documenting this accurately
    # and comprehensibly -- you really have to read the pickle code to
    # find all the special cases.

    I(name='REDUCE',
      code='R',
      arg=None,
      stack_before=[anyobject, anyobject],
      stack_after=[anyobject],
      proto=0,
      doc="""Push an object built from a callable and an argument tuple.

      The opcode is named to remind of the __reduce__() method.

      Stack before: ... callable pytuple
      Stack after:  ... callable(*pytuple)

      The callable and the argument tuple are the first two items returned
      by a __reduce__ method.  Applying the callable to the argtuple is
      supposed to reproduce the original object, or at least get it started.
      If the __reduce__ method returns a 3-tuple, the last component is an
      argument to be passed to the object's __setstate__, and then the REDUCE
      opcode is followed by code to create setstate's argument, and then a
      BUILD opcode to apply  __setstate__ to that argument.

      If not isinstance(callable, type), REDUCE complains unless the
      callable has been registered with the copyreg module's
      safe_constructors dict, or the callable has a magic
      '__safe_for_unpickling__' attribute with a true value.  I'm not sure
      why it does this, but I've sure seen this complaint often enough when
      I didn't want to <wink>.
      """),

    I(name='BUILD',
      code='b',
      arg=None,
      stack_before=[anyobject, anyobject],
      stack_after=[anyobject],
      proto=0,
      doc="""Finish building an object, via __setstate__ or dict update.

      Stack before: ... anyobject argument
      Stack after:  ... anyobject

      where anyobject may have been mutated, as follows:

      If the object has a __setstate__ method,

          anyobject.__setstate__(argument)

      is called.

      Else the argument must be a dict, the object must have a __dict__, and
      the object is updated via

          anyobject.__dict__.update(argument)
      """),

    I(name='INST',
      code='i',
      arg=stringnl_noescape_pair,
      stack_before=[markobject, stackslice],
      stack_after=[anyobject],
      proto=0,
      doc="""Build a class instance.

      This is the protocol 0 version of protocol 1's OBJ opcode.
      INST is followed by two newline-terminated strings, giving a
      module and class name, just as for the GLOBAL opcode (and see
      GLOBAL for more details about that).  self.find_class(module, name)
      is used to get a class object.

      In addition, all the objects on the stack following the topmost
      markobject are gathered into a tuple and popped (along with the
      topmost markobject), just as for the TUPLE opcode.

      Now it gets complicated.  If all of these are true:

        + The argtuple is empty (markobject was at the top of the stack
          at the start).

        + The class object does not have a __getinitargs__ attribute.

      then we want to create an old-style class instance without invoking
      its __init__() method (pickle has waffled on this over the years; not
      calling __init__() is current wisdom).  In this case, an instance of
      an old-style dummy class is created, and then we try to rebind its
      __class__ attribute to the desired class object.  If this succeeds,
      the new instance object is pushed on the stack, and we're done.

      Else (the argtuple is not empty, it's not an old-style class object,
      or the class object does have a __getinitargs__ attribute), the code
      first insists that the class object have a __safe_for_unpickling__
      attribute.  Unlike as for the __safe_for_unpickling__ check in REDUCE,
      it doesn't matter whether this attribute has a true or false value, it
      only matters whether it exists (XXX this is a bug).  If
      __safe_for_unpickling__ doesn't exist, UnpicklingError is raised.

      Else (the class object does have a __safe_for_unpickling__ attr),
      the class object obtained from INST's arguments is applied to the
      argtuple obtained from the stack, and the resulting instance object
      is pushed on the stack.

      NOTE:  checks for __safe_for_unpickling__ went away in Python 2.3.
      NOTE:  the distinction between old-style and new-style classes does
             not make sense in Python 3.
      """),

    I(name='OBJ',
      code='o',
      arg=None,
      stack_before=[markobject, anyobject, stackslice],
      stack_after=[anyobject],
      proto=1,
      doc="""Build a class instance.

      This is the protocol 1 version of protocol 0's INST opcode, and is
      very much like it.  The major difference is that the class object
      is taken off the stack, allowing it to be retrieved from the memo
      repeatedly if several instances of the same class are created.  This
      can be much more efficient (in both time and space) than repeatedly
      embedding the module and class names in INST opcodes.

      Unlike INST, OBJ takes no arguments from the opcode stream.  Instead
      the class object is taken off the stack, immediately above the
      topmost markobject:

      Stack before: ... markobject classobject stackslice
      Stack after:  ... new_instance_object

      As for INST, the remainder of the stack above the markobject is
      gathered into an argument tuple, and then the logic seems identical,
      except that no __safe_for_unpickling__ check is done (XXX this is
      a bug).  See INST for the gory details.

      NOTE:  In Python 2.3, INST and OBJ are identical except for how they
      get the class object.  That was always the intent; the implementations
      had diverged for accidental reasons.
      """),

    I(name='NEWOBJ',
      code='\x81',
      arg=None,
      stack_before=[anyobject, anyobject],
      stack_after=[anyobject],
      proto=2,
      doc="""Build an object instance.

      The stack before should be thought of as containing a class
      object followed by an argument tuple (the tuple being the stack
      top).  Call these cls and args.  They are popped off the stack,
      and the value returned by cls.__new__(cls, *args) is pushed back
      onto the stack.
      """),

    I(name='NEWOBJ_EX',
      code='\x92',
      arg=None,
      stack_before=[anyobject, anyobject, anyobject],
      stack_after=[anyobject],
      proto=4,
      doc="""Build an object instance.

      The stack before should be thought of as containing a class
      object followed by an argument tuple and by a keyword argument dict
      (the dict being the stack top).  Call these cls and args.  They are
      popped off the stack, and the value returned by
      cls.__new__(cls, *args, *kwargs) is  pushed back  onto the stack.
      """),

    # Machine control.

    I(name='PROTO',
      code='\x80',
      arg=uint1,
      stack_before=[],
      stack_after=[],
      proto=2,
      doc="""Protocol version indicator.

      For protocol 2 and above, a pickle must start with this opcode.
      The argument is the protocol version, an int in range(2, 256).
      """),

    I(name='STOP',
      code='.',
      arg=None,
      stack_before=[anyobject],
      stack_after=[],
      proto=0,
      doc="""Stop the unpickling machine.

      Every pickle ends with this opcode.  The object at the top of the stack
      is popped, and that's the result of unpickling.  The stack should be
      empty then.
      """),

    # Framing support.

    I(name='FRAME',
      code='\x95',
      arg=uint8,
      stack_before=[],
      stack_after=[],
      proto=4,
      doc="""Indicate the beginning of a new frame.

      The unpickler may use this opcode to safely prefetch data from its
      underlying stream.
      """),

    # Ways to deal with persistent IDs.

    I(name='PERSID',
      code='P',
      arg=stringnl_noescape,
      stack_before=[],
      stack_after=[anyobject],
      proto=0,
      doc="""Push an object identified by a persistent ID.

      The pickle module doesn't define what a persistent ID means.  PERSID's
      argument is a newline-terminated str-style (no embedded escapes, no
      bracketing quote characters) string, which *is* "the persistent ID".
      The unpickler passes this string to self.persistent_load().  Whatever
      object that returns is pushed on the stack.  There is no implementation
      of persistent_load() in Python's unpickler:  it must be supplied by an
      unpickler subclass.
      """),

    I(name='BINPERSID',
      code='Q',
      arg=None,
      stack_before=[anyobject],
      stack_after=[anyobject],
      proto=1,
      doc="""Push an object identified by a persistent ID.

      Like PERSID, except the persistent ID is popped off the stack (instead
      of being a string embedded in the opcode bytestream).  The persistent
      ID is passed to self.persistent_load(), and whatever object that
      returns is pushed on the stack.  See PERSID for more detail.
      """),
]
del I

# Verify uniqueness of .name and .code members.
name2i = {}
code2i = {}

for i, d in enumerate(opcodes):
    if d.name in name2i:
        raise ValueError("repeated name %r at indices %d and %d" %
                         (d.name, name2i[d.name], i))
    if d.code in code2i:
        raise ValueError("repeated code %r at indices %d and %d" %
                         (d.code, code2i[d.code], i))

    name2i[d.name] = i
    code2i[d.code] = i

del name2i, code2i, i, d

##############################################################################
# Build a code2op dict, mapping opcode characters to OpcodeInfo records.
# Also ensure we've got the same stuff as pickle.py, although the
# introspection here is dicey.

code2op = {}
for d in opcodes:
    code2op[d.code] = d
del d

def assure_pickle_consistency(verbose=False):

    copy = code2op.copy()
    for name in pickle.__all__:
        if not re.match("[A-Z][A-Z0-9_]+$", name):
            if verbose:
                print("skipping %r: it doesn't look like an opcode name" % name)
            continue
        picklecode = getattr(pickle, name)
        if not isinstance(picklecode, bytes) or len(picklecode) != 1:
            if verbose:
                print(("skipping %r: value %r doesn't look like a pickle "
                       "code" % (name, picklecode)))
            continue
        picklecode = picklecode.decode("latin-1")
        if picklecode in copy:
            if verbose:
                print("checking name %r w/ code %r for consistency" % (
                      name, picklecode))
            d = copy[picklecode]
            if d.name != name:
                raise ValueError("for pickle code %r, pickle.py uses name %r "
                                 "but we're using name %r" % (picklecode,
                                                              name,
                                                              d.name))
            # Forget this one.  Any left over in copy at the end are a problem
            # of a different kind.
            del copy[picklecode]
        else:
            raise ValueError("pickle.py appears to have a pickle opcode with "
                             "name %r and code %r, but we don't" %
                             (name, picklecode))
    if copy:
        msg = ["we appear to have pickle opcodes that pickle.py doesn't have:"]
        for code, d in copy.items():
            msg.append("    name %r with code %r" % (d.name, code))
        raise ValueError("\n".join(msg))

assure_pickle_consistency()
del assure_pickle_consistency

##############################################################################
# A pickle opcode generator.

def _genops(data, yield_end_pos=False):
    if isinstance(data, bytes_types):
        data = io.BytesIO(data)

    if hasattr(data, "tell"):
        getpos = data.tell
    else:
        getpos = lambda: None

    while True:
        pos = getpos()
        code = data.read(1)
        opcode = code2op.get(code.decode("latin-1"))
        if opcode is None:
            if code == b"":
                raise ValueError("pickle exhausted before seeing STOP")
            else:
                raise ValueError("at position %s, opcode %r unknown" % (
                                 "<unknown>" if pos is None else pos,
                                 code))
        if opcode.arg is None:
            arg = None
        else:
            arg = opcode.arg.reader(data)
        if yield_end_pos:
            yield opcode, arg, pos, getpos()
        else:
            yield opcode, arg, pos
        if code == b'.':
            assert opcode.name == 'STOP'
            break

def genops(pickle):
    """Generate all the opcodes in a pickle.

    'pickle' is a file-like object, or string, containing the pickle.

    Each opcode in the pickle is generated, from the current pickle position,
    stopping after a STOP opcode is delivered.  A triple is generated for
    each opcode:

        opcode, arg, pos

    opcode is an OpcodeInfo record, describing the current opcode.

    If the opcode has an argument embedded in the pickle, arg is its decoded
    value, as a Python object.  If the opcode doesn't have an argument, arg
    is None.

    If the pickle has a tell() method, pos was the value of pickle.tell()
    before reading the current opcode.  If the pickle is a bytes object,
    it's wrapped in a BytesIO object, and the latter's tell() result is
    used.  Else (the pickle doesn't have a tell(), and it's not obvious how
    to query its current position) pos is None.
    """
    return _genops(pickle)

##############################################################################
# A pickle optimizer.

def optimize(p):
    'Optimize a pickle string by removing unused PUT opcodes'
    put = 'PUT'
    get = 'GET'
    oldids = set()          # set of all PUT ids
    newids = {}             # set of ids used by a GET opcode
    opcodes = []            # (op, idx) or (pos, end_pos)
    proto = 0
    protoheader = b''
    for opcode, arg, pos, end_pos in _genops(p, yield_end_pos=True):
        if 'PUT' in opcode.name:
            oldids.add(arg)
            opcodes.append((put, arg))
        elif opcode.name == 'MEMOIZE':
            idx = len(oldids)
            oldids.add(idx)
            opcodes.append((put, idx))
        elif 'FRAME' in opcode.name:
            pass
        elif 'GET' in opcode.name:
            if opcode.proto > proto:
                proto = opcode.proto
            newids[arg] = None
            opcodes.append((get, arg))
        elif opcode.name == 'PROTO':
            if arg > proto:
                proto = arg
            if pos == 0:
                protoheader = p[pos:end_pos]
            else:
                opcodes.append((pos, end_pos))
        else:
            opcodes.append((pos, end_pos))
    del oldids

    # Copy the opcodes except for PUTS without a corresponding GET
    out = io.BytesIO()
    # Write the PROTO header before any framing
    out.write(protoheader)
    pickler = pickle._Pickler(out, proto)
    if proto >= 4:
        pickler.framer.start_framing()
    idx = 0
    for op, arg in opcodes:
        frameless = False
        if op is put:
            if arg not in newids:
                continue
            data = pickler.put(idx)
            newids[arg] = idx
            idx += 1
        elif op is get:
            data = pickler.get(newids[arg])
        else:
            data = p[op:arg]
            frameless = len(data) > pickler.framer._FRAME_SIZE_TARGET
        pickler.framer.commit_frame(force=frameless)
        if frameless:
            pickler.framer.file_write(data)
        else:
            pickler.write(data)
    pickler.framer.end_framing()
    return out.getvalue()

##############################################################################
# A symbolic pickle disassembler.

def dis(pickle, out=None, memo=None, indentlevel=4, annotate=0):
    """Produce a symbolic disassembly of a pickle.

    'pickle' is a file-like object, or string, containing a (at least one)
    pickle.  The pickle is disassembled from the current position, through
    the first STOP opcode encountered.

    Optional arg 'out' is a file-like object to which the disassembly is
    printed.  It defaults to sys.stdout.

    Optional arg 'memo' is a Python dict, used as the pickle's memo.  It
    may be mutated by dis(), if the pickle contains PUT or BINPUT opcodes.
    Passing the same memo object to another dis() call then allows disassembly
    to proceed across multiple pickles that were all created by the same
    pickler with the same memo.  Ordinarily you don't need to worry about this.

    Optional arg 'indentlevel' is the number of blanks by which to indent
    a new MARK level.  It defaults to 4.

    Optional arg 'annotate' if nonzero instructs dis() to add short
    description of the opcode on each line of disassembled output.
    The value given to 'annotate' must be an integer and is used as a
    hint for the column where annotation should start.  The default
    value is 0, meaning no annotations.

    In addition to printing the disassembly, some sanity checks are made:

    + All embedded opcode arguments "make sense".

    + Explicit and implicit pop operations have enough items on the stack.

    + When an opcode implicitly refers to a markobject, a markobject is
      actually on the stack.

    + A memo entry isn't referenced before it's defined.

    + The markobject isn't stored in the memo.

    + A memo entry isn't redefined.
    """

    # Most of the hair here is for sanity checks, but most of it is needed
    # anyway to detect when a protocol 0 POP takes a MARK off the stack
    # (which in turn is needed to indent MARK blocks correctly).

    stack = []          # crude emulation of unpickler stack
    if memo is None:
        memo = {}       # crude emulation of unpickler memo
    maxproto = -1       # max protocol number seen
    markstack = []      # bytecode positions of MARK opcodes
    indentchunk = ' ' * indentlevel
    errormsg = None
    annocol = annotate  # column hint for annotations
    for opcode, arg, pos in genops(pickle):
        if pos is not None:
            print("%5d:" % pos, end=' ', file=out)

        line = "%-4s %s%s" % (repr(opcode.code)[1:-1],
                              indentchunk * len(markstack),
                              opcode.name)

        maxproto = max(maxproto, opcode.proto)
        before = opcode.stack_before    # don't mutate
        after = opcode.stack_after      # don't mutate
        numtopop = len(before)

        # See whether a MARK should be popped.
        markmsg = None
        if markobject in before or (opcode.name == "POP" and
                                    stack and
                                    stack[-1] is markobject):
            assert markobject not in after
            if __debug__:
                if markobject in before:
                    assert before[-1] is stackslice
            if markstack:
                markpos = markstack.pop()
                if markpos is None:
                    markmsg = "(MARK at unknown opcode offset)"
                else:
                    markmsg = "(MARK at %d)" % markpos
                # Pop everything at and after the topmost markobject.
                while stack[-1] is not markobject:
                    stack.pop()
                stack.pop()
                # Stop later code from popping too much.
                try:
                    numtopop = before.index(markobject)
                except ValueError:
                    assert opcode.name == "POP"
                    numtopop = 0
            else:
                errormsg = markmsg = "no MARK exists on stack"

        # Check for correct memo usage.
        if opcode.name in ("PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"):
            if opcode.name == "MEMOIZE":
                memo_idx = len(memo)
                markmsg = "(as %d)" % memo_idx
            else:
                assert arg is not None
                memo_idx = arg
            if memo_idx in memo:
                errormsg = "memo key %r already defined" % arg
            elif not stack:
                errormsg = "stack is empty -- can't store into memo"
            elif stack[-1] is markobject:
                errormsg = "can't store markobject in the memo"
            else:
                memo[memo_idx] = stack[-1]
        elif opcode.name in ("GET", "BINGET", "LONG_BINGET"):
            if arg in memo:
                assert len(after) == 1
                after = [memo[arg]]     # for better stack emulation
            else:
                errormsg = "memo key %r has never been stored into" % arg

        if arg is not None or markmsg:
            # make a mild effort to align arguments
            line += ' ' * (10 - len(opcode.name))
            if arg is not None:
                line += ' ' + repr(arg)
            if markmsg:
                line += ' ' + markmsg
        if annotate:
            line += ' ' * (annocol - len(line))
            # make a mild effort to align annotations
            annocol = len(line)
            if annocol > 50:
                annocol = annotate
            line += ' ' + opcode.doc.split('\n', 1)[0]
        print(line, file=out)

        if errormsg:
            # Note that we delayed complaining until the offending opcode
            # was printed.
            raise ValueError(errormsg)

        # Emulate the stack effects.
        if len(stack) < numtopop:
            raise ValueError("tries to pop %d items from stack with "
                             "only %d items" % (numtopop, len(stack)))
        if numtopop:
            del stack[-numtopop:]
        if markobject in after:
            assert markobject not in before
            markstack.append(pos)

        stack.extend(after)

    print("highest protocol among opcodes =", maxproto, file=out)
    if stack:
        raise ValueError("stack not empty after STOP: %r" % stack)

# For use in the doctest, simply as an example of a class to pickle.
class _Example:
    def __init__(self, value):
        self.value = value

_dis_test = r"""
>>> import pickle
>>> x = [1, 2, (3, 4), {b'abc': "def"}]
>>> pkl0 = pickle.dumps(x, 0)
>>> dis(pkl0)
    0: (    MARK
    1: l        LIST       (MARK at 0)
    2: p    PUT        0
    5: I    INT        1
    8: a    APPEND
    9: I    INT        2
   12: a    APPEND
   13: (    MARK
   14: I        INT        3
   17: I        INT        4
   20: t        TUPLE      (MARK at 13)
   21: p    PUT        1
   24: a    APPEND
   25: (    MARK
   26: d        DICT       (MARK at 25)
   27: p    PUT        2
   30: c    GLOBAL     '_codecs encode'
   46: p    PUT        3
   49: (    MARK
   50: V        UNICODE    'abc'
   55: p        PUT        4
   58: V        UNICODE    'latin1'
   66: p        PUT        5
   69: t        TUPLE      (MARK at 49)
   70: p    PUT        6
   73: R    REDUCE
   74: p    PUT        7
   77: V    UNICODE    'def'
   82: p    PUT        8
   85: s    SETITEM
   86: a    APPEND
   87: .    STOP
highest protocol among opcodes = 0

Try again with a "binary" pickle.

>>> pkl1 = pickle.dumps(x, 1)
>>> dis(pkl1)
    0: ]    EMPTY_LIST
    1: q    BINPUT     0
    3: (    MARK
    4: K        BININT1    1
    6: K        BININT1    2
    8: (        MARK
    9: K            BININT1    3
   11: K            BININT1    4
   13: t            TUPLE      (MARK at 8)
   14: q        BINPUT     1
   16: }        EMPTY_DICT
   17: q        BINPUT     2
   19: c        GLOBAL     '_codecs encode'
   35: q        BINPUT     3
   37: (        MARK
   38: X            BINUNICODE 'abc'
   46: q            BINPUT     4
   48: X            BINUNICODE 'latin1'
   59: q            BINPUT     5
   61: t            TUPLE      (MARK at 37)
   62: q        BINPUT     6
   64: R        REDUCE
   65: q        BINPUT     7
   67: X        BINUNICODE 'def'
   75: q        BINPUT     8
   77: s        SETITEM
   78: e        APPENDS    (MARK at 3)
   79: .    STOP
highest protocol among opcodes = 1

Exercise the INST/OBJ/BUILD family.

>>> import pickletools
>>> dis(pickle.dumps(pickletools.dis, 0))
    0: c    GLOBAL     'pickletools dis'
   17: p    PUT        0
   20: .    STOP
highest protocol among opcodes = 0

>>> from pickletools import _Example
>>> x = [_Example(42)] * 2
>>> dis(pickle.dumps(x, 0))
    0: (    MARK
    1: l        LIST       (MARK at 0)
    2: p    PUT        0
    5: c    GLOBAL     'copy_reg _reconstructor'
   30: p    PUT        1
   33: (    MARK
   34: c        GLOBAL     'pickletools _Example'
   56: p        PUT        2
   59: c        GLOBAL     '__builtin__ object'
   79: p        PUT        3
   82: N        NONE
   83: t        TUPLE      (MARK at 33)
   84: p    PUT        4
   87: R    REDUCE
   88: p    PUT        5
   91: (    MARK
   92: d        DICT       (MARK at 91)
   93: p    PUT        6
   96: V    UNICODE    'value'
  103: p    PUT        7
  106: I    INT        42
  110: s    SETITEM
  111: b    BUILD
  112: a    APPEND
  113: g    GET        5
  116: a    APPEND
  117: .    STOP
highest protocol among opcodes = 0

>>> dis(pickle.dumps(x, 1))
    0: ]    EMPTY_LIST
    1: q    BINPUT     0
    3: (    MARK
    4: c        GLOBAL     'copy_reg _reconstructor'
   29: q        BINPUT     1
   31: (        MARK
   32: c            GLOBAL     'pickletools _Example'
   54: q            BINPUT     2
   56: c            GLOBAL     '__builtin__ object'
   76: q            BINPUT     3
   78: N            NONE
   79: t            TUPLE      (MARK at 31)
   80: q        BINPUT     4
   82: R        REDUCE
   83: q        BINPUT     5
   85: }        EMPTY_DICT
   86: q        BINPUT     6
   88: X        BINUNICODE 'value'
   98: q        BINPUT     7
  100: K        BININT1    42
  102: s        SETITEM
  103: b        BUILD
  104: h        BINGET     5
  106: e        APPENDS    (MARK at 3)
  107: .    STOP
highest protocol among opcodes = 1

Try "the canonical" recursive-object test.

>>> L = []
>>> T = L,
>>> L.append(T)
>>> L[0] is T
True
>>> T[0] is L
True
>>> L[0][0] is L
True
>>> T[0][0] is T
True
>>> dis(pickle.dumps(L, 0))
    0: (    MARK
    1: l        LIST       (MARK at 0)
    2: p    PUT        0
    5: (    MARK
    6: g        GET        0
    9: t        TUPLE      (MARK at 5)
   10: p    PUT        1
   13: a    APPEND
   14: .    STOP
highest protocol among opcodes = 0

>>> dis(pickle.dumps(L, 1))
    0: ]    EMPTY_LIST
    1: q    BINPUT     0
    3: (    MARK
    4: h        BINGET     0
    6: t        TUPLE      (MARK at 3)
    7: q    BINPUT     1
    9: a    APPEND
   10: .    STOP
highest protocol among opcodes = 1

Note that, in the protocol 0 pickle of the recursive tuple, the disassembler
has to emulate the stack in order to realize that the POP opcode at 16 gets
rid of the MARK at 0.

>>> dis(pickle.dumps(T, 0))
    0: (    MARK
    1: (        MARK
    2: l            LIST       (MARK at 1)
    3: p        PUT        0
    6: (        MARK
    7: g            GET        0
   10: t            TUPLE      (MARK at 6)
   11: p        PUT        1
   14: a        APPEND
   15: 0        POP
   16: 0        POP        (MARK at 0)
   17: g    GET        1
   20: .    STOP
highest protocol among opcodes = 0

>>> dis(pickle.dumps(T, 1))
    0: (    MARK
    1: ]        EMPTY_LIST
    2: q        BINPUT     0
    4: (        MARK
    5: h            BINGET     0
    7: t            TUPLE      (MARK at 4)
    8: q        BINPUT     1
   10: a        APPEND
   11: 1        POP_MARK   (MARK at 0)
   12: h    BINGET     1
   14: .    STOP
highest protocol among opcodes = 1

Try protocol 2.

>>> dis(pickle.dumps(L, 2))
    0: \x80 PROTO      2
    2: ]    EMPTY_LIST
    3: q    BINPUT     0
    5: h    BINGET     0
    7: \x85 TUPLE1
    8: q    BINPUT     1
   10: a    APPEND
   11: .    STOP
highest protocol among opcodes = 2

>>> dis(pickle.dumps(T, 2))
    0: \x80 PROTO      2
    2: ]    EMPTY_LIST
    3: q    BINPUT     0
    5: h    BINGET     0
    7: \x85 TUPLE1
    8: q    BINPUT     1
   10: a    APPEND
   11: 0    POP
