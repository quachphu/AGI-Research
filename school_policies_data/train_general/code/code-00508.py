def mkstemp(suffix=None, prefix=None, dir=None, text=False):
    """User-callable function to create and return a unique temporary
    file.  The return value is a pair (fd, name) where fd is the
    file descriptor returned by os.open, and name is the filename.

    If 'suffix' is not None, the file name will end with that suffix,
    otherwise there will be no suffix.

    If 'prefix' is not None, the file name will begin with that prefix,
    otherwise a default prefix is used.

    If 'dir' is not None, the file will be created in that directory,
    otherwise a default directory is used.

    If 'text' is specified and true, the file is opened in text
    mode.  Else (the default) the file is opened in binary mode.

    If any of 'suffix', 'prefix' and 'dir' are not None, they must be the
    same type.  If they are bytes, the returned name will be bytes; str
    otherwise.

    The file is readable and writable only by the creating user ID.
    If the operating system uses permission bits to indicate whether a
    file is executable, the file is executable by no one. The file
    descriptor is not inherited by children of this process.

    Caller is responsible for deleting the file when done with it.
    """

    prefix, suffix, dir, output_type = _sanitize_params(prefix, suffix, dir)

    if text:
        flags = _text_openflags
    else:
        flags = _bin_openflags

    return _mkstemp_inner(dir, prefix, suffix, flags, output_type)


def mkdtemp(suffix=None, prefix=None, dir=None):
    """User-callable function to create and return a unique temporary
    directory.  The return value is the pathname of the directory.

    Arguments are as for mkstemp, except that the 'text' argument is
    not accepted.

    The directory is readable, writable, and searchable only by the
    creating user.

    Caller is responsible for deleting the directory when done with it.
    """

    prefix, suffix, dir, output_type = _sanitize_params(prefix, suffix, dir)

    names = _get_candidate_names()
    if output_type is bytes:
        names = map(_os.fsencode, names)

    for seq in range(TMP_MAX):
        name = next(names)
        file = _os.path.join(dir, prefix + name + suffix)
        _sys.audit("tempfile.mkdtemp", file)
        try:
            _os.mkdir(file, 0o700)
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
        return file

    raise FileExistsError(_errno.EEXIST,
                          "No usable temporary directory name found")

def mktemp(suffix="", prefix=template, dir=None):
    """User-callable function to return a unique temporary file name.  The
    file is not created.

    Arguments are similar to mkstemp, except that the 'text' argument is
    not accepted, and suffix=None, prefix=None and bytes file names are not
    supported.

    THIS FUNCTION IS UNSAFE AND SHOULD NOT BE USED.  The file name may
    refer to a file that did not exist at some point, but by the time
    you get around to creating it, someone else may have beaten you to
    the punch.
    """

##    from warnings import warn as _warn
##    _warn("mktemp is a potential security risk to your program",
##          RuntimeWarning, stacklevel=2)

    if dir is None:
        dir = gettempdir()

    names = _get_candidate_names()
    for seq in range(TMP_MAX):
        name = next(names)
        file = _os.path.join(dir, prefix + name + suffix)
        if not _exists(file):
            return file

    raise FileExistsError(_errno.EEXIST,
                          "No usable temporary filename found")


class _TemporaryFileCloser:
    """A separate object allowing proper closing of a temporary file's
    underlying file object, without adding a __del__ method to the
    temporary file."""

    file = None  # Set here since __del__ checks it
    close_called = False

    def __init__(self, file, name, delete=True):
        self.file = file
        self.name = name
        self.delete = delete

    # NT provides delete-on-close as a primitive, so we don't need
    # the wrapper to do anything special.  We still use it so that
    # file.name is useful (i.e. not "(fdopen)") with NamedTemporaryFile.
    if _os.name != 'nt':
        # Cache the unlinker so we don't get spurious errors at
        # shutdown when the module-level "os" is None'd out.  Note
        # that this must be referenced as self.unlink, because the
        # name TemporaryFileWrapper may also get None'd out before
        # __del__ is called.

        def close(self, unlink=_os.unlink):
            if not self.close_called and self.file is not None:
                self.close_called = True
                try:
                    self.file.close()
                finally:
                    if self.delete:
                        unlink(self.name)

        # Need to ensure the file is deleted on __del__
        def __del__(self):
            self.close()

    else:
        def close(self):
            if not self.close_called:
                self.close_called = True
                self.file.close()


class _TemporaryFileWrapper:
    """Temporary file wrapper

    This class provides a wrapper around files opened for
    temporary use.  In particular, it seeks to automatically
    remove the file when it is no longer needed.
    """

    def __init__(self, file, name, delete=True):
        self.file = file
        self.name = name
        self.delete = delete
        self._closer = _TemporaryFileCloser(file, name, delete)

    def __getattr__(self, name):
        # Attribute lookups are delegated to the underlying file
        # and cached for non-numeric results
        # (i.e. methods are cached, closed and friends are not)
        file = self.__dict__['file']
        a = getattr(file, name)
        if hasattr(a, '__call__'):
            func = a
            @_functools.wraps(func)
            def func_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            # Avoid closing the file as long as the wrapper is alive,
            # see issue #18879.
            func_wrapper._closer = self._closer
            a = func_wrapper
        if not isinstance(a, int):
            setattr(self, name, a)
        return a

    # The underlying __enter__ method returns the wrong object
    # (self.file) so override it to return the wrapper
    def __enter__(self):
        self.file.__enter__()
        return self

    # Need to trap __exit__ as well to ensure the file gets
    # deleted when used in a with statement
    def __exit__(self, exc, value, tb):
        result = self.file.__exit__(exc, value, tb)
        self.close()
        return result

    def close(self):
        """
        Close the temporary file, possibly deleting it.
        """
        self._closer.close()

    # iter() doesn't use __getattr__ to find the __iter__ method
    def __iter__(self):
        # Don't return iter(self.file), but yield from it to avoid closing
        # file as long as it's being used as iterator (see issue #23700).  We
        # can't use 'yield from' here because iter(file) returns the file
        # object itself, which has a close method, and thus the file would get
        # closed when the generator is finalized, due to PEP380 semantics.
        for line in self.file:
            yield line


def NamedTemporaryFile(mode='w+b', buffering=-1, encoding=None,
                       newline=None, suffix=None, prefix=None,
                       dir=None, delete=True, *, errors=None):
    """Create and return a temporary file.
    Arguments:
    'prefix', 'suffix', 'dir' -- as for mkstemp.
    'mode' -- the mode argument to io.open (default "w+b").
    'buffering' -- the buffer size argument to io.open (default -1).
    'encoding' -- the encoding argument to io.open (default None)
    'newline' -- the newline argument to io.open (default None)
    'delete' -- whether the file is deleted on close (default True).
    'errors' -- the errors argument to io.open (default None)
    The file is created as mkstemp() would do it.

    Returns an object with a file-like interface; the name of the file
    is accessible as its 'name' attribute.  The file will be automatically
    deleted when it is closed unless the 'delete' argument is set to False.
    """

    prefix, suffix, dir, output_type = _sanitize_params(prefix, suffix, dir)

    flags = _bin_openflags

    # Setting O_TEMPORARY in the flags causes the OS to delete
    # the file when it is closed.  This is only supported by Windows.
    if _os.name == 'nt' and delete:
        flags |= _os.O_TEMPORARY

    if "b" not in mode:
        encoding = _io.text_encoding(encoding)

    name = None
    def opener(*args):
        nonlocal name
        fd, name = _mkstemp_inner(dir, prefix, suffix, flags, output_type)
        return fd
    try:
        file = _io.open(dir, mode, buffering=buffering,
                        newline=newline, encoding=encoding, errors=errors,
                        opener=opener)
        try:
            raw = getattr(file, 'buffer', file)
            raw = getattr(raw, 'raw', raw)
            raw.name = name
            return _TemporaryFileWrapper(file, name, delete)
        except:
            file.close()
            raise
    except:
        if name is not None and not (_os.name == 'nt' and delete):
            _os.unlink(name)
        raise

if _os.name != 'posix' or _sys.platform == 'cygwin':
    # On non-POSIX and Cygwin systems, assume that we cannot unlink a file
    # while it is open.
    TemporaryFile = NamedTemporaryFile

else:
    # Is the O_TMPFILE flag available and does it work?
    # The flag is set to False if os.open(dir, os.O_TMPFILE) raises an
    # IsADirectoryError exception
    _O_TMPFILE_WORKS = hasattr(_os, 'O_TMPFILE')

    def TemporaryFile(mode='w+b', buffering=-1, encoding=None,
                      newline=None, suffix=None, prefix=None,
                      dir=None, *, errors=None):
        """Create and return a temporary file.
        Arguments:
        'prefix', 'suffix', 'dir' -- as for mkstemp.
        'mode' -- the mode argument to io.open (default "w+b").
        'buffering' -- the buffer size argument to io.open (default -1).
        'encoding' -- the encoding argument to io.open (default None)
        'newline' -- the newline argument to io.open (default None)
        'errors' -- the errors argument to io.open (default None)
        The file is created as mkstemp() would do it.

        Returns an object with a file-like interface.  The file has no
        name, and will cease to exist when it is closed.
        """
        global _O_TMPFILE_WORKS

        if "b" not in mode:
            encoding = _io.text_encoding(encoding)

        prefix, suffix, dir, output_type = _sanitize_params(prefix, suffix, dir)

        flags = _bin_openflags
        if _O_TMPFILE_WORKS:
            fd = None
            def opener(*args):
                nonlocal fd
                flags2 = (flags | _os.O_TMPFILE) & ~_os.O_CREAT
                fd = _os.open(dir, flags2, 0o600)
                return fd
            try:
                file = _io.open(dir, mode, buffering=buffering,
                                newline=newline, encoding=encoding,
                                errors=errors, opener=opener)
                raw = getattr(file, 'buffer', file)
                raw = getattr(raw, 'raw', raw)
                raw.name = fd
                return file
            except IsADirectoryError:
                # Linux kernel older than 3.11 ignores the O_TMPFILE flag:
                # O_TMPFILE is read as O_DIRECTORY. Trying to open a directory
                # with O_RDWR|O_DIRECTORY fails with IsADirectoryError, a
                # directory cannot be open to write. Set flag to False to not
                # try again.
                _O_TMPFILE_WORKS = False
            except OSError:
                # The filesystem of the directory does not support O_TMPFILE.
                # For example, OSError(95, 'Operation not supported').
                #
                # On Linux kernel older than 3.11, trying to open a regular
                # file (or a symbolic link to a regular file) with O_TMPFILE
                # fails with NotADirectoryError, because O_TMPFILE is read as
                # O_DIRECTORY.
                pass
            # Fallback to _mkstemp_inner().

        fd = None
        def opener(*args):
            nonlocal fd
            fd, name = _mkstemp_inner(dir, prefix, suffix, flags, output_type)
            try:
                _os.unlink(name)
            except BaseException as e:
                _os.close(fd)
                raise
            return fd
        file = _io.open(dir, mode, buffering=buffering,
                        newline=newline, encoding=encoding, errors=errors,
                        opener=opener)
        raw = getattr(file, 'buffer', file)
        raw = getattr(raw, 'raw', raw)
        raw.name = fd
        return file

class SpooledTemporaryFile:
    """Temporary file wrapper, specialized to switch from BytesIO
    or StringIO to a real file when it exceeds a certain size or
    when a fileno is needed.
    """
    _rolled = False

    def __init__(self, max_size=0, mode='w+b', buffering=-1,
                 encoding=None, newline=None,
                 suffix=None, prefix=None, dir=None, *, errors=None):
        if 'b' in mode:
            self._file = _io.BytesIO()
        else:
            encoding = _io.text_encoding(encoding)
            self._file = _io.TextIOWrapper(_io.BytesIO(),
                            encoding=encoding, errors=errors,
                            newline=newline)
        self._max_size = max_size
        self._rolled = False
        self._TemporaryFileArgs = {'mode': mode, 'buffering': buffering,
                                   'suffix': suffix, 'prefix': prefix,
                                   'encoding': encoding, 'newline': newline,
                                   'dir': dir, 'errors': errors}

    __class_getitem__ = classmethod(_types.GenericAlias)

    def _check(self, file):
        if self._rolled: return
        max_size = self._max_size
        if max_size and file.tell() > max_size:
            self.rollover()

    def rollover(self):
        if self._rolled: return
        file = self._file
        newfile = self._file = TemporaryFile(**self._TemporaryFileArgs)
        del self._TemporaryFileArgs

        pos = file.tell()
        if hasattr(newfile, 'buffer'):
            newfile.buffer.write(file.detach().getvalue())
        else:
            newfile.write(file.getvalue())
        newfile.seek(pos, 0)

        self._rolled = True

    # The method caching trick from NamedTemporaryFile
    # won't work here, because _file may change from a
    # BytesIO/StringIO instance to a real file. So we list
    # all the methods directly.

    # Context management protocol
    def __enter__(self):
        if self._file.closed:
            raise ValueError("Cannot enter context with closed file")
        return self

    def __exit__(self, exc, value, tb):
        self._file.close()

    # file protocol
    def __iter__(self):
        return self._file.__iter__()

    def close(self):
        self._file.close()

    @property
    def closed(self):
        return self._file.closed

    @property
    def encoding(self):
        return self._file.encoding

    @property
    def errors(self):
        return self._file.errors

    def fileno(self):
        self.rollover()
        return self._file.fileno()

    def flush(self):
        self._file.flush()

    def isatty(self):
        return self._file.isatty()

    @property
    def mode(self):
        try:
            return self._file.mode
        except AttributeError:
            return self._TemporaryFileArgs['mode']

    @property
    def name(self):
        try:
            return self._file.name
        except AttributeError:
            return None

    @property
    def newlines(self):
        return self._file.newlines

    def read(self, *args):
        return self._file.read(*args)

    def readline(self, *args):
        return self._file.readline(*args)

    def readlines(self, *args):
        return self._file.readlines(*args)

    def seek(self, *args):
        return self._file.seek(*args)

    def tell(self):
        return self._file.tell()

    def truncate(self, size=None):
        if size is None:
            self._file.truncate()
        else:
            if size > self._max_size:
                self.rollover()
            self._file.truncate(size)

    def write(self, s):
        file = self._file
        rv = file.write(s)
        self._check(file)
        return rv

    def writelines(self, iterable):
        file = self._file
        rv = file.writelines(iterable)
        self._check(file)
        return rv


class TemporaryDirectory:
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    def __init__(self, suffix=None, prefix=None, dir=None,
                 ignore_cleanup_errors=False):
        self.name = mkdtemp(suffix, prefix, dir)
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._finalizer = _weakref.finalize(
            self, self._cleanup, self.name,
            warn_message="Implicitly cleaning up {!r}".format(self),
            ignore_errors=self._ignore_cleanup_errors)

    @classmethod
    def _rmtree(cls, name, ignore_errors=False):
        def onerror(func, path, exc_info):
            if issubclass(exc_info[0], PermissionError):
                try:
                    if path != name:
                        _resetperms(_os.path.dirname(path))
                    _resetperms(path)

                    try:
                        _os.unlink(path)
                    # PermissionError is raised on FreeBSD for directories
                    except (IsADirectoryError, PermissionError):
                        cls._rmtree(path, ignore_errors=ignore_errors)
                except FileNotFoundError:
                    pass
            elif issubclass(exc_info[0], FileNotFoundError):
                pass
            else:
                if not ignore_errors:
                    raise

        _rmtree(name, onerror=onerror)

    @classmethod
    def _cleanup(cls, name, warn_message, ignore_errors=False):
        cls._rmtree(name, ignore_errors=ignore_errors)
        _warnings.warn(warn_message, ResourceWarning)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def cleanup(self):
        if self._finalizer.detach() or _os.path.exists(self.name):
            self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)

    __class_getitem__ = classmethod(_types.GenericAlias)
<|endoftext|>
"""
csv.py - read/write/investigate CSV files
"""

import re
from _csv import Error, __version__, writer, reader, register_dialect, \
                 unregister_dialect, get_dialect, list_dialects, \
                 field_size_limit, \
                 QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONNUMERIC, QUOTE_NONE, \
                 __doc__
from _csv import Dialect as _Dialect

from io import StringIO

__all__ = ["QUOTE_MINIMAL", "QUOTE_ALL", "QUOTE_NONNUMERIC", "QUOTE_NONE",
           "Error", "Dialect", "__doc__", "excel", "excel_tab",
           "field_size_limit", "reader", "writer",
           "register_dialect", "get_dialect", "list_dialects", "Sniffer",
           "unregister_dialect", "__version__", "DictReader", "DictWriter",
           "unix_dialect"]

class Dialect:
    """Describe a CSV dialect.

    This must be subclassed (see csv.excel).  Valid attributes are:
    delimiter, quotechar, escapechar, doublequote, skipinitialspace,
    lineterminator, quoting.

    """
    _name = ""
    _valid = False
    # placeholders
    delimiter = None
    quotechar = None
    escapechar = None
    doublequote = None
    skipinitialspace = None
    lineterminator = None
    quoting = None

    def __init__(self):
        if self.__class__ != Dialect:
            self._valid = True
        self._validate()

    def _validate(self):
        try:
            _Dialect(self)
        except TypeError as e:
            # We do this for compatibility with py2.3
            raise Error(str(e))

class excel(Dialect):
    """Describe the usual properties of Excel-generated CSV files."""
    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\r\n'
    quoting = QUOTE_MINIMAL
register_dialect("excel", excel)

class excel_tab(excel):
    """Describe the usual properties of Excel-generated TAB-delimited files."""
    delimiter = '\t'
register_dialect("excel-tab", excel_tab)

class unix_dialect(Dialect):
    """Describe the usual properties of Unix-generated CSV files."""
    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\n'
    quoting = QUOTE_ALL
register_dialect("unix", unix_dialect)


class DictReader:
    def __init__(self, f, fieldnames=None, restkey=None, restval=None,
                 dialect="excel", *args, **kwds):
        self._fieldnames = fieldnames   # list of keys for the dict
        self.restkey = restkey          # key to catch long rows
        self.restval = restval          # default value for short rows
        self.reader = reader(f, dialect, *args, **kwds)
        self.dialect = dialect
        self.line_num = 0

    def __iter__(self):
        return self

    @property
    def fieldnames(self):
        if self._fieldnames is None:
            try:
                self._fieldnames = next(self.reader)
            except StopIteration:
                pass
        self.line_num = self.reader.line_num
        return self._fieldnames

    @fieldnames.setter
    def fieldnames(self, value):
        self._fieldnames = value

    def __next__(self):
        if self.line_num == 0:
            # Used only for its side effect.
            self.fieldnames
        row = next(self.reader)
        self.line_num = self.reader.line_num

        # unlike the basic reader, we prefer not to return blanks,
        # because we will typically wind up with a dict full of None
        # values
        while row == []:
            row = next(self.reader)
        d = dict(zip(self.fieldnames, row))
        lf = len(self.fieldnames)
        lr = len(row)
        if lf < lr:
            d[self.restkey] = row[lf:]
        elif lf > lr:
            for key in self.fieldnames[lr:]:
                d[key] = self.restval
        return d


class DictWriter:
    def __init__(self, f, fieldnames, restval="", extrasaction="raise",
                 dialect="excel", *args, **kwds):
        self.fieldnames = fieldnames    # list of keys for the dict
        self.restval = restval          # for writing short dicts
        if extrasaction.lower() not in ("raise", "ignore"):
            raise ValueError("extrasaction (%s) must be 'raise' or 'ignore'"
                             % extrasaction)
        self.extrasaction = extrasaction
        self.writer = writer(f, dialect, *args, **kwds)

    def writeheader(self):
        header = dict(zip(self.fieldnames, self.fieldnames))
        return self.writerow(header)

    def _dict_to_list(self, rowdict):
        if self.extrasaction == "raise":
            wrong_fields = rowdict.keys() - self.fieldnames
            if wrong_fields:
                raise ValueError("dict contains fields not in fieldnames: "
                                 + ", ".join([repr(x) for x in wrong_fields]))
        return (rowdict.get(key, self.restval) for key in self.fieldnames)

    def writerow(self, rowdict):
        return self.writer.writerow(self._dict_to_list(rowdict))

    def writerows(self, rowdicts):
        return self.writer.writerows(map(self._dict_to_list, rowdicts))

# Guard Sniffer's type checking against builds that exclude complex()
try:
    complex
except NameError:
    complex = float

class Sniffer:
    '''
    "Sniffs" the format of a CSV file (i.e. delimiter, quotechar)
    Returns a Dialect object.
    '''
    def __init__(self):
        # in case there is more than one possible delimiter
        self.preferred = [',', '\t', ';', ' ', ':']


    def sniff(self, sample, delimiters=None):
        """
        Returns a dialect (or None) corresponding to the sample
        """

        quotechar, doublequote, delimiter, skipinitialspace = \
                   self._guess_quote_and_delimiter(sample, delimiters)
        if not delimiter:
            delimiter, skipinitialspace = self._guess_delimiter(sample,
                                                                delimiters)

        if not delimiter:
            raise Error("Could not determine delimiter")

        class dialect(Dialect):
            _name = "sniffed"
            lineterminator = '\r\n'
            quoting = QUOTE_MINIMAL
            # escapechar = ''

        dialect.doublequote = doublequote
        dialect.delimiter = delimiter
        # _csv.reader won't accept a quotechar of ''
        dialect.quotechar = quotechar or '"'
        dialect.skipinitialspace = skipinitialspace

        return dialect


    def _guess_quote_and_delimiter(self, data, delimiters):
        """
        Looks for text enclosed between two identical quotes
        (the probable quotechar) which are preceded and followed
        by the same character (the probable delimiter).
        For example:
                         ,'some text',
        The quote with the most wins, same with the delimiter.
        If there is no quotechar the delimiter can't be determined
        this way.
        """

        matches = []
        for restr in (r'(?P<delim>[^\w\n"\'])(?P<space> ?)(?P<quote>["\']).*?(?P=quote)(?P=delim)', # ,".*?",
                      r'(?:^|\n)(?P<quote>["\']).*?(?P=quote)(?P<delim>[^\w\n"\'])(?P<space> ?)',   #  ".*?",
                      r'(?P<delim>[^\w\n"\'])(?P<space> ?)(?P<quote>["\']).*?(?P=quote)(?:$|\n)',   # ,".*?"
                      r'(?:^|\n)(?P<quote>["\']).*?(?P=quote)(?:$|\n)'):                            #  ".*?" (no delim, no space)
            regexp = re.compile(restr, re.DOTALL | re.MULTILINE)
            matches = regexp.findall(data)
            if matches:
                break

        if not matches:
            # (quotechar, doublequote, delimiter, skipinitialspace)
            return ('', False, None, 0)
        quotes = {}
        delims = {}
        spaces = 0
        groupindex = regexp.groupindex
        for m in matches:
            n = groupindex['quote'] - 1
            key = m[n]
            if key:
                quotes[key] = quotes.get(key, 0) + 1
            try:
                n = groupindex['delim'] - 1
                key = m[n]
            except KeyError:
                continue
            if key and (delimiters is None or key in delimiters):
                delims[key] = delims.get(key, 0) + 1
            try:
                n = groupindex['space'] - 1
            except KeyError:
                continue
            if m[n]:
                spaces += 1

        quotechar = max(quotes, key=quotes.get)

        if delims:
            delim = max(delims, key=delims.get)
            skipinitialspace = delims[delim] == spaces
            if delim == '\n': # most likely a file with a single column
                delim = ''
        else:
            # there is *no* delimiter, it's a single column of quoted data
            delim = ''
            skipinitialspace = 0

        # if we see an extra quote between delimiters, we've got a
        # double quoted format
        dq_regexp = re.compile(
                               r"((%(delim)s)|^)\W*%(quote)s[^%(delim)s\n]*%(quote)s[^%(delim)s\n]*%(quote)s\W*((%(delim)s)|$)" % \
                               {'delim':re.escape(delim), 'quote':quotechar}, re.MULTILINE)



        if dq_regexp.search(data):
            doublequote = True
        else:
            doublequote = False

        return (quotechar, doublequote, delim, skipinitialspace)


    def _guess_delimiter(self, data, delimiters):
        """
        The delimiter /should/ occur the same number of times on
        each row. However, due to malformed data, it may not. We don't want
        an all or nothing approach, so we allow for small variations in this
        number.
          1) build a table of the frequency of each character on every line.
          2) build a table of frequencies of this frequency (meta-frequency?),
             e.g.  'x occurred 5 times in 10 rows, 6 times in 1000 rows,
             7 times in 2 rows'
          3) use the mode of the meta-frequency to determine the /expected/
             frequency for that character
          4) find out how often the character actually meets that goal
          5) the character that best meets its goal is the delimiter
        For performance reasons, the data is evaluated in chunks, so it can
        try and evaluate the smallest portion of the data possible, evaluating
        additional chunks as necessary.
        """

        data = list(filter(None, data.split('\n')))

        ascii = [chr(c) for c in range(127)] # 7-bit ASCII

        # build frequency tables
        chunkLength = min(10, len(data))
        iteration = 0
        charFrequency = {}
        modes = {}
        delims = {}
        start, end = 0, chunkLength
        while start < len(data):
            iteration += 1
            for line in data[start:end]:
                for char in ascii:
                    metaFrequency = charFrequency.get(char, {})
                    # must count even if frequency is 0
                    freq = line.count(char)
                    # value is the mode
                    metaFrequency[freq] = metaFrequency.get(freq, 0) + 1
                    charFrequency[char] = metaFrequency

            for char in charFrequency.keys():
                items = list(charFrequency[char].items())
                if len(items) == 1 and items[0][0] == 0:
                    continue
                # get the mode of the frequencies
                if len(items) > 1:
                    modes[char] = max(items, key=lambda x: x[1])
                    # adjust the mode - subtract the sum of all
                    # other frequencies
                    items.remove(modes[char])
                    modes[char] = (modes[char][0], modes[char][1]
                                   - sum(item[1] for item in items))
                else:
                    modes[char] = items[0]

            # build a list of possible delimiters
            modeList = modes.items()
            total = float(min(chunkLength * iteration, len(data)))
            # (rows of consistent data) / (number of rows) = 100%
            consistency = 1.0
            # minimum consistency threshold
            threshold = 0.9
            while len(delims) == 0 and consistency >= threshold:
                for k, v in modeList:
                    if v[0] > 0 and v[1] > 0:
                        if ((v[1]/total) >= consistency and
                            (delimiters is None or k in delimiters)):
                            delims[k] = v
                consistency -= 0.01

            if len(delims) == 1:
                delim = list(delims.keys())[0]
                skipinitialspace = (data[0].count(delim) ==
                                    data[0].count("%c " % delim))
                return (delim, skipinitialspace)

            # analyze another chunkLength lines
            start = end
            end += chunkLength

        if not delims:
            return ('', 0)

        # if there's more than one, fall back to a 'preferred' list
        if len(delims) > 1:
            for d in self.preferred:
                if d in delims.keys():
                    skipinitialspace = (data[0].count(d) ==
                                        data[0].count("%c " % d))
                    return (d, skipinitialspace)

        # nothing else indicates a preference, pick the character that
        # dominates(?)
        items = [(v,k) for (k,v) in delims.items()]
        items.sort()
        delim = items[-1][1]

        skipinitialspace = (data[0].count(delim) ==
                            data[0].count("%c " % delim))
        return (delim, skipinitialspace)


    def has_header(self, sample):
        # Creates a dictionary of types of data in each column. If any
        # column is of a single type (say, integers), *except* for the first
        # row, then the first row is presumed to be labels. If the type
        # can't be determined, it is assumed to be a string in which case
        # the length of the string is the determining factor: if all of the
        # rows except for the first are the same length, it's a header.
        # Finally, a 'vote' is taken at the end for each column, adding or
        # subtracting from the likelihood of the first row being a header.

        rdr = reader(StringIO(sample), self.sniff(sample))

        header = next(rdr) # assume first row is header

        columns = len(header)
        columnTypes = {}
        for i in range(columns): columnTypes[i] = None

        checked = 0
        for row in rdr:
            # arbitrary number of rows to check, to keep it sane
            if checked > 20:
                break
            checked += 1

            if len(row) != columns:
                continue # skip rows that have irregular number of columns

            for col in list(columnTypes.keys()):
                thisType = complex
                try:
                    thisType(row[col])
                except (ValueError, OverflowError):
                    # fallback to length of string
                    thisType = len(row[col])

                if thisType != columnTypes[col]:
                    if columnTypes[col] is None: # add new column type
                        columnTypes[col] = thisType
                    else:
                        # type is inconsistent, remove column from
                        # consideration
                        del columnTypes[col]

        # finally, compare results against first row and "vote"
        # on whether it's a header
        hasHeader = 0
        for col, colType in columnTypes.items():
            if type(colType) == type(0): # it's a length
                if len(header[col]) != colType:
                    hasHeader += 1
                else:
                    hasHeader -= 1
            else: # attempt typecast
                try:
                    colType(header[col])
                except (ValueError, TypeError):
                    hasHeader += 1
                else:
                    hasHeader -= 1

        return hasHeader > 0
<|endoftext|>"""
Basic statistics module.

This module provides functions for calculating statistics of data, including
averages, variance, and standard deviation.

Calculating averages
--------------------

==================  ==================================================
Function            Description
==================  ==================================================
mean                Arithmetic mean (average) of data.
fmean               Fast, floating point arithmetic mean.
geometric_mean      Geometric mean of data.
harmonic_mean       Harmonic mean of data.
median              Median (middle value) of data.
median_low          Low median of data.
median_high         High median of data.
median_grouped      Median, or 50th percentile, of grouped data.
mode                Mode (most common value) of data.
multimode           List of modes (most common values of data).
quantiles           Divide data into intervals with equal probability.
==================  ==================================================

Calculate the arithmetic mean ("the average") of data:

>>> mean([-1.0, 2.5, 3.25, 5.75])
2.625


Calculate the standard median of discrete data:

>>> median([2, 3, 4, 5])
3.5


Calculate the median, or 50th percentile, of data grouped into class intervals
centred on the data values provided. E.g. if your data points are rounded to
the nearest whole number:

>>> median_grouped([2, 2, 3, 3, 3, 4])  #doctest: +ELLIPSIS
2.8333333333...

This should be interpreted in this way: you have two data points in the class
interval 1.5-2.5, three data points in the class interval 2.5-3.5, and one in
the class interval 3.5-4.5. The median of these data points is 2.8333...


Calculating variability or spread
---------------------------------

==================  =============================================
Function            Description
==================  =============================================
pvariance           Population variance of data.
variance            Sample variance of data.
pstdev              Population standard deviation of data.
stdev               Sample standard deviation of data.
==================  =============================================

Calculate the standard deviation of sample data:

>>> stdev([2.5, 3.25, 5.5, 11.25, 11.75])  #doctest: +ELLIPSIS
4.38961843444...

If you have previously calculated the mean, you can pass it as the optional
second argument to the four "spread" functions to avoid recalculating it:

>>> data = [1, 2, 2, 4, 4, 4, 5, 6]
>>> mu = mean(data)
>>> pvariance(data, mu)
2.5


Statistics for relations between two inputs
-------------------------------------------

==================  ====================================================
Function            Description
==================  ====================================================
covariance          Sample covariance for two variables.
correlation         Pearson's correlation coefficient for two variables.
linear_regression   Intercept and slope for simple linear regression.
==================  ====================================================

Calculate covariance, Pearson's correlation, and simple linear regression
for two inputs:

>>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
>>> covariance(x, y)
0.75
>>> correlation(x, y)  #doctest: +ELLIPSIS
0.31622776601...
>>> linear_regression(x, y)  #doctest:
LinearRegression(slope=0.1, intercept=1.5)


Exceptions
----------

A single exception is defined: StatisticsError is a subclass of ValueError.

"""

__all__ = [
    'NormalDist',
    'StatisticsError',
    'correlation',
    'covariance',
    'fmean',
    'geometric_mean',
    'harmonic_mean',
    'linear_regression',
    'mean',
    'median',
    'median_grouped',
    'median_high',
    'median_low',
    'mode',
    'multimode',
    'pstdev',
    'pvariance',
    'quantiles',
    'stdev',
    'variance',
]

import math
import numbers
import random

from fractions import Fraction
from decimal import Decimal
from itertools import groupby, repeat
from bisect import bisect_left, bisect_right
from math import hypot, sqrt, fabs, exp, erf, tau, log, fsum
from operator import itemgetter
from collections import Counter, namedtuple

# === Exceptions ===

class StatisticsError(ValueError):
    pass


# === Private utilities ===

def _sum(data):
    """_sum(data) -> (type, sum, count)

    Return a high-precision sum of the given numeric data as a fraction,
    together with the type to be converted to and the count of items.

    Examples
    --------

    >>> _sum([3, 2.25, 4.5, -0.5, 0.25])
    (<class 'float'>, Fraction(19, 2), 5)

    Some sources of round-off error will be avoided:

    # Built-in sum returns zero.
    >>> _sum([1e50, 1, -1e50] * 1000)
    (<class 'float'>, Fraction(1000, 1), 3000)

    Fractions and Decimals are also supported:

    >>> from fractions import Fraction as F
    >>> _sum([F(2, 3), F(7, 5), F(1, 4), F(5, 6)])
    (<class 'fractions.Fraction'>, Fraction(63, 20), 4)

    >>> from decimal import Decimal as D
    >>> data = [D("0.1375"), D("0.2108"), D("0.3061"), D("0.0419")]
    >>> _sum(data)
    (<class 'decimal.Decimal'>, Fraction(6963, 10000), 4)

    Mixed types are currently treated as an error, except that int is
    allowed.
    """
    count = 0
    partials = {}
    partials_get = partials.get
    T = int
    for typ, values in groupby(data, type):
        T = _coerce(T, typ)  # or raise TypeError
        for n, d in map(_exact_ratio, values):
            count += 1
            partials[d] = partials_get(d, 0) + n
    if None in partials:
        # The sum will be a NAN or INF. We can ignore all the finite
        # partials, and just look at this special one.
        total = partials[None]
        assert not _isfinite(total)
    else:
        # Sum all the partial sums using builtin sum.
        total = sum(Fraction(n, d) for d, n in partials.items())
    return (T, total, count)


def _isfinite(x):
    try:
        return x.is_finite()  # Likely a Decimal.
    except AttributeError:
        return math.isfinite(x)  # Coerces to float first.


def _coerce(T, S):
    """Coerce types T and S to a common type, or raise TypeError.

    Coercion rules are currently an implementation detail. See the CoerceTest
    test class in test_statistics for details.
    """
    # See http://bugs.python.org/issue24068.
    assert T is not bool, "initial type T is bool"
    # If the types are the same, no need to coerce anything. Put this
    # first, so that the usual case (no coercion needed) happens as soon
    # as possible.
    if T is S:  return T
    # Mixed int & other coerce to the other type.
    if S is int or S is bool:  return T
    if T is int:  return S
    # If one is a (strict) subclass of the other, coerce to the subclass.
    if issubclass(S, T):  return S
    if issubclass(T, S):  return T
    # Ints coerce to the other type.
    if issubclass(T, int):  return S
    if issubclass(S, int):  return T
    # Mixed fraction & float coerces to float (or float subclass).
    if issubclass(T, Fraction) and issubclass(S, float):
        return S
    if issubclass(T, float) and issubclass(S, Fraction):
        return T
    # Any other combination is disallowed.
    msg = "don't know how to coerce %s and %s"
    raise TypeError(msg % (T.__name__, S.__name__))


def _exact_ratio(x):
    """Return Real number x to exact (numerator, denominator) pair.

    >>> _exact_ratio(0.25)
    (1, 4)

    x is expected to be an int, Fraction, Decimal or float.
    """
    try:
        return x.as_integer_ratio()
    except AttributeError:
        pass
    except (OverflowError, ValueError):
        # float NAN or INF.
        assert not _isfinite(x)
        return (x, None)
    try:
        # x may be an Integral ABC.
        return (x.numerator, x.denominator)
    except AttributeError:
        msg = f"can't convert type '{type(x).__name__}' to numerator/denominator"
        raise TypeError(msg)


def _convert(value, T):
    """Convert value to given numeric type T."""
    if type(value) is T:
        # This covers the cases where T is Fraction, or where value is
        # a NAN or INF (Decimal or float).
        return value
    if issubclass(T, int) and value.denominator != 1:
        T = float
    try:
        # FIXME: what do we do if this overflows?
        return T(value)
    except TypeError:
        if issubclass(T, Decimal):
            return T(value.numerator) / T(value.denominator)
        else:
            raise


def _find_lteq(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


def _find_rteq(a, l, x):
    'Locate the rightmost value exactly equal to x'
    i = bisect_right(a, x, lo=l)
    if i != (len(a) + 1) and a[i - 1] == x:
        return i - 1
    raise ValueError


def _fail_neg(values, errmsg='negative value'):
    """Iterate over values, failing if any are less than zero."""
    for x in values:
        if x < 0:
            raise StatisticsError(errmsg)
        yield x


# === Measures of central tendency (averages) ===

def mean(data):
    """Return the sample arithmetic mean of data.

    >>> mean([1, 2, 3, 4, 4])
    2.8

    >>> from fractions import Fraction as F
    >>> mean([F(3, 7), F(1, 21), F(5, 3), F(1, 3)])
    Fraction(13, 21)

    >>> from decimal import Decimal as D
    >>> mean([D("0.5"), D("0.75"), D("0.625"), D("0.375")])
    Decimal('0.5625')

    If ``data`` is empty, StatisticsError will be raised.
    """
    if iter(data) is data:
        data = list(data)
    n = len(data)
    if n < 1:
        raise StatisticsError('mean requires at least one data point')
    T, total, count = _sum(data)
    assert count == n
    return _convert(total / n, T)


def fmean(data):
    """Convert data to floats and compute the arithmetic mean.

    This runs faster than the mean() function and it always returns a float.
    If the input dataset is empty, it raises a StatisticsError.

    >>> fmean([3.5, 4.0, 5.25])
    4.25
    """
    try:
        n = len(data)
    except TypeError:
        # Handle iterators that do not define __len__().
        n = 0
        def count(iterable):
            nonlocal n
            for n, x in enumerate(iterable, start=1):
                yield x
        total = fsum(count(data))
    else:
        total = fsum(data)
    try:
        return total / n
    except ZeroDivisionError:
        raise StatisticsError('fmean requires at least one data point') from None


def geometric_mean(data):
    """Convert data to floats and compute the geometric mean.

    Raises a StatisticsError if the input dataset is empty,
    if it contains a zero, or if it contains a negative value.

    No special efforts are made to achieve exact results.
    (However, this may change in the future.)

    >>> round(geometric_mean([54, 24, 36]), 9)
    36.0
    """
    try:
        return exp(fmean(map(log, data)))
    except ValueError:
        raise StatisticsError('geometric mean requires a non-empty dataset '
                              'containing positive numbers') from None


def harmonic_mean(data, weights=None):
    """Return the harmonic mean of data.

    The harmonic mean is the reciprocal of the arithmetic mean of the
    reciprocals of the data.  It can be used for averaging ratios or
    rates, for example speeds.

    Suppose a car travels 40 km/hr for 5 km and then speeds-up to
    60 km/hr for another 5 km. What is the average speed?

        >>> harmonic_mean([40, 60])
        48.0

    Suppose a car travels 40 km/hr for 5 km, and when traffic clears,
    speeds-up to 60 km/hr for the remaining 30 km of the journey. What
    is the average speed?

        >>> harmonic_mean([40, 60], weights=[5, 30])
        56.0

    If ``data`` is empty, or any element is less than zero,
    ``harmonic_mean`` will raise ``StatisticsError``.
    """
    if iter(data) is data:
        data = list(data)
    errmsg = 'harmonic mean does not support negative values'
    n = len(data)
    if n < 1:
        raise StatisticsError('harmonic_mean requires at least one data point')
    elif n == 1 and weights is None:
        x = data[0]
        if isinstance(x, (numbers.Real, Decimal)):
            if x < 0:
                raise StatisticsError(errmsg)
            return x
        else:
            raise TypeError('unsupported type')
    if weights is None:
        weights = repeat(1, n)
        sum_weights = n
    else:
        if iter(weights) is weights:
            weights = list(weights)
        if len(weights) != n:
            raise StatisticsError('Number of weights does not match data size')
        _, sum_weights, _ = _sum(w for w in _fail_neg(weights, errmsg))
    try:
        data = _fail_neg(data, errmsg)
        T, total, count = _sum(w / x if w else 0 for w, x in zip(weights, data))
    except ZeroDivisionError:
        return 0
    if total <= 0:
        raise StatisticsError('Weighted sum must be positive')
    return _convert(sum_weights / total, T)

# FIXME: investigate ways to calculate medians without sorting? Quickselect?
def median(data):
    """Return the median (middle value) of numeric data.

    When the number of data points is odd, return the middle data point.
    When the number of data points is even, the median is interpolated by
    taking the average of the two middle values:

    >>> median([1, 3, 5])
    3
    >>> median([1, 3, 5, 7])
    4.0

    """
    data = sorted(data)
    n = len(data)
    if n == 0:
        raise StatisticsError("no median for empty data")
    if n % 2 == 1:
        return data[n // 2]
    else:
        i = n // 2
        return (data[i - 1] + data[i]) / 2


def median_low(data):
    """Return the low median of numeric data.

    When the number of data points is odd, the middle value is returned.
    When it is even, the smaller of the two middle values is returned.

    >>> median_low([1, 3, 5])
    3
    >>> median_low([1, 3, 5, 7])
    3

    """
    data = sorted(data)
    n = len(data)
    if n == 0:
        raise StatisticsError("no median for empty data")
    if n % 2 == 1:
        return data[n // 2]
    else:
        return data[n // 2 - 1]


def median_high(data):
    """Return the high median of data.

    When the number of data points is odd, the middle value is returned.
    When it is even, the larger of the two middle values is returned.

    >>> median_high([1, 3, 5])
    3
    >>> median_high([1, 3, 5, 7])
    5

    """
    data = sorted(data)
    n = len(data)
    if n == 0:
        raise StatisticsError("no median for empty data")
    return data[n // 2]


def median_grouped(data, interval=1):
    """Return the 50th percentile (median) of grouped continuous data.

    >>> median_grouped([1, 2, 2, 3, 4, 4, 4, 4, 4, 5])
    3.7
    >>> median_grouped([52, 52, 53, 54])
    52.5

    This calculates the median as the 50th percentile, and should be
    used when your data is continuous and grouped. In the above example,
    the values 1, 2, 3, etc. actually represent the midpoint of classes
    0.5-1.5, 1.5-2.5, 2.5-3.5, etc. The middle value falls somewhere in
    class 3.5-4.5, and interpolation is used to estimate it.

    Optional argument ``interval`` represents the class interval, and
    defaults to 1. Changing the class interval naturally will change the
    interpolated 50th percentile value:

    >>> median_grouped([1, 3, 3, 5, 7], interval=1)
    3.25
    >>> median_grouped([1, 3, 3, 5, 7], interval=2)
    3.5

    This function does not check whether the data points are at least
    ``interval`` apart.
    """
    data = sorted(data)
    n = len(data)
    if n == 0:
        raise StatisticsError("no median for empty data")
    elif n == 1:
        return data[0]
    # Find the value at the midpoint. Remember this corresponds to the
    # centre of the class interval.
    x = data[n // 2]
    for obj in (x, interval):
        if isinstance(obj, (str, bytes)):
            raise TypeError('expected number but got %r' % obj)
    try:
        L = x - interval / 2  # The lower limit of the median interval.
    except TypeError:
        # Mixed type. For now we just coerce to float.
        L = float(x) - float(interval) / 2

    # Uses bisection search to search for x in data with log(n) time complexity
    # Find the position of leftmost occurrence of x in data
    l1 = _find_lteq(data, x)
    # Find the position of rightmost occurrence of x in data[l1...len(data)]
    # Assuming always l1 <= l2
    l2 = _find_rteq(data, l1, x)
    cf = l1
    f = l2 - l1 + 1
    return L + interval * (n / 2 - cf) / f


def mode(data):
    """Return the most common data point from discrete or nominal data.

    ``mode`` assumes discrete data, and returns a single value. This is the
    standard treatment of the mode as commonly taught in schools:

        >>> mode([1, 1, 2, 3, 3, 3, 3, 4])
        3

    This also works with nominal (non-numeric) data:

        >>> mode(["red", "blue", "blue", "red", "green", "red", "red"])
        'red'

    If there are multiple modes with same frequency, return the first one
    encountered:

        >>> mode(['red', 'red', 'green', 'blue', 'blue'])
        'red'

    If *data* is empty, ``mode``, raises StatisticsError.

    """
    pairs = Counter(iter(data)).most_common(1)
    try:
        return pairs[0][0]
    except IndexError:
        raise StatisticsError('no mode for empty data') from None


def multimode(data):
    """Return a list of the most frequently occurring values.

    Will return more than one result if there are multiple modes
    or an empty list if *data* is empty.

    >>> multimode('aabbbbbbbbcc')
    ['b']
    >>> multimode('aabbbbccddddeeffffgg')
    ['b', 'd', 'f']
    >>> multimode('')
    []
    """
    counts = Counter(iter(data)).most_common()
    maxcount, mode_items = next(groupby(counts, key=itemgetter(1)), (0, []))
    return list(map(itemgetter(0), mode_items))


# Notes on methods for computing quantiles
# ----------------------------------------
#
# There is no one perfect way to compute quantiles.  Here we offer
# two methods that serve common needs.  Most other packages
# surveyed offered at least one or both of these two, making them
# "standard" in the sense of "widely-adopted and reproducible".
# They are also easy to explain, easy to compute manually, and have
# straight-forward interpretations that aren't surprising.

# The default method is known as "R6", "PERCENTILE.EXC", or "expected
# value of rank order statistics". The alternative method is known as
# "R7", "PERCENTILE.INC", or "mode of rank order statistics".

# For sample data where there is a positive probability for values
# beyond the range of the data, the R6 exclusive method is a
# reasonable choice.  Consider a random sample of nine values from a
# population with a uniform distribution from 0.0 to 1.0.  The
# distribution of the third ranked sample point is described by
# betavariate(alpha=3, beta=7) which has mode=0.250, median=0.286, and
# mean=0.300.  Only the latter (which corresponds with R6) gives the
# desired cut point with 30% of the population falling below that
# value, making it comparable to a result from an inv_cdf() function.
# The R6 exclusive method is also idempotent.

# For describing population data where the end points are known to
# be included in the data, the R7 inclusive method is a reasonable
# choice.  Instead of the mean, it uses the mode of the beta
# distribution for the interior points.  Per Hyndman & Fan, "One nice
# property is that the vertices of Q7(p) divide the range into n - 1
# intervals, and exactly 100p% of the intervals lie to the left of
# Q7(p) and 100(1 - p)% of the intervals lie to the right of Q7(p)."

# If needed, other methods could be added.  However, for now, the
# position is that fewer options make for easier choices and that
# external packages can be used for anything more advanced.

def quantiles(data, *, n=4, method='exclusive'):
    """Divide *data* into *n* continuous intervals with equal probability.

    Returns a list of (n - 1) cut points separating the intervals.

    Set *n* to 4 for quartiles (the default).  Set *n* to 10 for deciles.
    Set *n* to 100 for percentiles which gives the 99 cuts points that
    separate *data* in to 100 equal sized groups.

    The *data* can be any iterable containing sample.
    The cut points are linearly interpolated between data points.

    If *method* is set to *inclusive*, *data* is treated as population
    data.  The minimum value is treated as the 0th percentile and the
    maximum value is treated as the 100th percentile.
    """
    if n < 1:
        raise StatisticsError('n must be at least 1')
    data = sorted(data)
    ld = len(data)
    if ld < 2:
        raise StatisticsError('must have at least two data points')
    if method == 'inclusive':
        m = ld - 1
        result = []
        for i in range(1, n):
            j, delta = divmod(i * m, n)
            interpolated = (data[j] * (n - delta) + data[j + 1] * delta) / n
            result.append(interpolated)
        return result
    if method == 'exclusive':
        m = ld + 1
        result = []
        for i in range(1, n):
            j = i * m // n                               # rescale i to m/n
            j = 1 if j < 1 else ld-1 if j > ld-1 else j  # clamp to 1 .. ld-1
            delta = i*m - j*n                            # exact integer math
            interpolated = (data[j - 1] * (n - delta) + data[j] * delta) / n
            result.append(interpolated)
        return result
    raise ValueError(f'Unknown method: {method!r}')


# === Measures of spread ===

# See http://mathworld.wolfram.com/Variance.html
#     http://mathworld.wolfram.com/SampleVariance.html
#     http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
#
# Under no circumstances use the so-called "computational formula for
# variance", as that is only suitable for hand calculations with a small
# amount of low-precision data. It has terrible numeric properties.
#
# See a comparison of three computational methods here:
# http://www.johndcook.com/blog/2008/09/26/comparing-three-methods-of-computing-standard-deviation/

def _ss(data, c=None):
    """Return sum of square deviations of sequence data.

    If ``c`` is None, the mean is calculated in one pass, and the deviations
    from the mean are calculated in a second pass. Otherwise, deviations are
    calculated from ``c`` as given. Use the second case with care, as it can
    lead to garbage results.
    """
    if c is not None:
        T, total, count = _sum((x-c)**2 for x in data)
        return (T, total)
    T, total, count = _sum(data)
    mean_n, mean_d = (total / count).as_integer_ratio()
    partials = Counter()
    for n, d in map(_exact_ratio, data):
        diff_n = n * mean_d - d * mean_n
        diff_d = d * mean_d
        partials[diff_d * diff_d] += diff_n * diff_n
    if None in partials:
        # The sum will be a NAN or INF. We can ignore all the finite
        # partials, and just look at this special one.
        total = partials[None]
        assert not _isfinite(total)
    else:
        total = sum(Fraction(n, d) for d, n in partials.items())
    return (T, total)


def variance(data, xbar=None):
    """Return the sample variance of data.

    data should be an iterable of Real-valued numbers, with at least two
    values. The optional argument xbar, if given, should be the mean of
    the data. If it is missing or None, the mean is automatically calculated.

    Use this function when your data is a sample from a population. To
    calculate the variance from the entire population, see ``pvariance``.

    Examples:

    >>> data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
    >>> variance(data)
    1.3720238095238095

    If you have already calculated the mean of your data, you can pass it as
    the optional second argument ``xbar`` to avoid recalculating it:

    >>> m = mean(data)
    >>> variance(data, m)
    1.3720238095238095

    This function does not check that ``xbar`` is actually the mean of
    ``data``. Giving arbitrary values for ``xbar`` may lead to invalid or
    impossible results.

    Decimals and Fractions are supported:

    >>> from decimal import Decimal as D
    >>> variance([D("27.5"), D("30.25"), D("30.25"), D("34.5"), D("41.75")])
    Decimal('31.01875')

    >>> from fractions import Fraction as F
    >>> variance([F(1, 6), F(1, 2), F(5, 3)])
    Fraction(67, 108)

    """
    if iter(data) is data:
        data = list(data)
    n = len(data)
    if n < 2:
        raise StatisticsError('variance requires at least two data points')
    T, ss = _ss(data, xbar)
    return _convert(ss / (n - 1), T)


def pvariance(data, mu=None):
    """Return the population variance of ``data``.

    data should be a sequence or iterable of Real-valued numbers, with at least one
    value. The optional argument mu, if given, should be the mean of
    the data. If it is missing or None, the mean is automatically calculated.

    Use this function to calculate the variance from the entire population.
    To estimate the variance from a sample, the ``variance`` function is
    usually a better choice.

    Examples:

    >>> data = [0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25]
    >>> pvariance(data)
    1.25

    If you have already calculated the mean of the data, you can pass it as
    the optional second argument to avoid recalculating it:

    >>> mu = mean(data)
    >>> pvariance(data, mu)
    1.25

    Decimals and Fractions are supported:

    >>> from decimal import Decimal as D
    >>> pvariance([D("27.5"), D("30.25"), D("30.25"), D("34.5"), D("41.75")])
    Decimal('24.815')

    >>> from fractions import Fraction as F
    >>> pvariance([F(1, 4), F(5, 4), F(1, 2)])
    Fraction(13, 72)

    """
    if iter(data) is data:
        data = list(data)
    n = len(data)
    if n < 1:
        raise StatisticsError('pvariance requires at least one data point')
    T, ss = _ss(data, mu)
    return _convert(ss / n, T)


def stdev(data, xbar=None):
    """Return the square root of the sample variance.

    See ``variance`` for arguments and other details.

    >>> stdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])
    1.0810874155219827

    """
    # Fixme: Despite the exact sum of squared deviations, some inaccuracy
    # remain because there are two rounding steps.  The first occurs in
    # the _convert() step for variance(), the second occurs in math.sqrt().
    var = variance(data, xbar)
    try:
        return var.sqrt()
    except AttributeError:
        return math.sqrt(var)


def pstdev(data, mu=None):
    """Return the square root of the population variance.

    See ``pvariance`` for arguments and other details.

    >>> pstdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])
    0.986893273527251

    """
    # Fixme: Despite the exact sum of squared deviations, some inaccuracy
    # remain because there are two rounding steps.  The first occurs in
    # the _convert() step for pvariance(), the second occurs in math.sqrt().
    var = pvariance(data, mu)
    try:
        return var.sqrt()
    except AttributeError:
        return math.sqrt(var)


# === Statistics for relations between two inputs ===

# See https://en.wikipedia.org/wiki/Covariance
#     https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
#     https://en.wikipedia.org/wiki/Simple_linear_regression


def covariance(x, y, /):
    """Covariance

    Return the sample covariance of two inputs *x* and *y*. Covariance
    is a measure of the joint variability of two inputs.

    >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    >>> covariance(x, y)
    0.75
    >>> z = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    >>> covariance(x, z)
    -7.5
    >>> covariance(z, x)
    -7.5

    """
    n = len(x)
    if len(y) != n:
        raise StatisticsError('covariance requires that both inputs have same number of data points')
    if n < 2:
        raise StatisticsError('covariance requires at least two data points')
    xbar = fsum(x) / n
    ybar = fsum(y) / n
    sxy = fsum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y))
    return sxy / (n - 1)


def correlation(x, y, /):
    """Pearson's correlation coefficient

    Return the Pearson's correlation coefficient for two inputs. Pearson's
    correlation coefficient *r* takes values between -1 and +1. It measures the
    strength and direction of the linear relationship, where +1 means very
    strong, positive linear relationship, -1 very strong, negative linear
    relationship, and 0 no linear relationship.

    >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> y = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    >>> correlation(x, x)
    1.0
    >>> correlation(x, y)
    -1.0

    """
    n = len(x)
    if len(y) != n:
        raise StatisticsError('correlation requires that both inputs have same number of data points')
    if n < 2:
        raise StatisticsError('correlation requires at least two data points')
    xbar = fsum(x) / n
    ybar = fsum(y) / n
    sxy = fsum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y))
    sxx = fsum((xi - xbar) ** 2.0 for xi in x)
    syy = fsum((yi - ybar) ** 2.0 for yi in y)
    try:
        return sxy / sqrt(sxx * syy)
    except ZeroDivisionError:
        raise StatisticsError('at least one of the inputs is constant')


LinearRegression = namedtuple('LinearRegression', ('slope', 'intercept'))


def linear_regression(x, y, /):
    """Slope and intercept for simple linear regression.

    Return the slope and intercept of simple linear regression
    parameters estimated using ordinary least squares. Simple linear
    regression describes relationship between an independent variable
    *x* and a dependent variable *y* in terms of linear function:

        y = slope * x + intercept + noise

    where *slope* and *intercept* are the regression parameters that are
    estimated, and noise represents the variability of the data that was
    not explained by the linear regression (it is equal to the
    difference between predicted and actual values of the dependent
    variable).

    The parameters are returned as a named tuple.

    >>> x = [1, 2, 3, 4, 5]
    >>> noise = NormalDist().samples(5, seed=42)
    >>> y = [3 * x[i] + 2 + noise[i] for i in range(5)]
    >>> linear_regression(x, y)  #doctest: +ELLIPSIS
    LinearRegression(slope=3.09078914170..., intercept=1.75684970486...)

    """
    n = len(x)
    if len(y) != n:
        raise StatisticsError('linear regression requires that both inputs have same number of data points')
    if n < 2:
        raise StatisticsError('linear regression requires at least two data points')
    xbar = fsum(x) / n
    ybar = fsum(y) / n
    sxy = fsum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y))
    sxx = fsum((xi - xbar) ** 2.0 for xi in x)
    try:
        slope = sxy / sxx   # equivalent to:  covariance(x, y) / variance(x)
    except ZeroDivisionError:
        raise StatisticsError('x is constant')
    intercept = ybar - slope * xbar
    return LinearRegression(slope=slope, intercept=intercept)


## Normal Distribution #####################################################


def _normal_dist_inv_cdf(p, mu, sigma):
    # There is no closed-form solution to the inverse CDF for the normal
    # distribution, so we use a rational approximation instead:
    # Wichura, M.J. (1988). "Algorithm AS241: The Percentage Points of the
    # Normal Distribution".  Applied Statistics. Blackwell Publishing. 37
    # (3): 477–484. doi:10.2307/2347330. JSTOR 2347330.
    q = p - 0.5
    if fabs(q) <= 0.425:
        r = 0.180625 - q * q
        # Hash sum: 55.88319_28806_14901_4439
        num = (((((((2.50908_09287_30122_6727e+3 * r +
                     3.34305_75583_58812_8105e+4) * r +
                     6.72657_70927_00870_0853e+4) * r +
                     4.59219_53931_54987_1457e+4) * r +
                     1.37316_93765_50946_1125e+4) * r +
                     1.97159_09503_06551_4427e+3) * r +
                     1.33141_66789_17843_7745e+2) * r +
                     3.38713_28727_96366_6080e+0) * q
        den = (((((((5.22649_52788_52854_5610e+3 * r +
                     2.87290_85735_72194_2674e+4) * r +
                     3.93078_95800_09271_0610e+4) * r +
                     2.12137_94301_58659_5867e+4) * r +
                     5.39419_60214_24751_1077e+3) * r +
                     6.87187_00749_20579_0830e+2) * r +
                     4.23133_30701_60091_1252e+1) * r +
                     1.0)
        x = num / den
        return mu + (x * sigma)
    r = p if q <= 0.0 else 1.0 - p
    r = sqrt(-log(r))
    if r <= 5.0:
        r = r - 1.6
        # Hash sum: 49.33206_50330_16102_89036
        num = (((((((7.74545_01427_83414_07640e-4 * r +
                     2.27238_44989_26918_45833e-2) * r +
                     2.41780_72517_74506_11770e-1) * r +
                     1.27045_82524_52368_38258e+0) * r +
                     3.64784_83247_63204_60504e+0) * r +
                     5.76949_72214_60691_40550e+0) * r +
                     4.63033_78461_56545_29590e+0) * r +
                     1.42343_71107_49683_57734e+0)
        den = (((((((1.05075_00716_44416_84324e-9 * r +
                     5.47593_80849_95344_94600e-4) * r +
                     1.51986_66563_61645_71966e-2) * r +
                     1.48103_97642_74800_74590e-1) * r +
                     6.89767_33498_51000_04550e-1) * r +
                     1.67638_48301_83803_84940e+0) * r +
                     2.05319_16266_37758_82187e+0) * r +
                     1.0)
    else:
        r = r - 5.0
        # Hash sum: 47.52583_31754_92896_71629
        num = (((((((2.01033_43992_92288_13265e-7 * r +
                     2.71155_55687_43487_57815e-5) * r +
                     1.24266_09473_88078_43860e-3) * r +
                     2.65321_89526_57612_30930e-2) * r +
                     2.96560_57182_85048_91230e-1) * r +
                     1.78482_65399_17291_33580e+0) * r +
                     5.46378_49111_64114_36990e+0) * r +
                     6.65790_46435_01103_77720e+0)
        den = (((((((2.04426_31033_89939_78564e-15 * r +
                     1.42151_17583_16445_88870e-7) * r +
                     1.84631_83175_10054_68180e-5) * r +
                     7.86869_13114_56132_59100e-4) * r +
                     1.48753_61290_85061_48525e-2) * r +
                     1.36929_88092_27358_05310e-1) * r +
                     5.99832_20655_58879_37690e-1) * r +
                     1.0)
    x = num / den
    if q < 0.0:
        x = -x
    return mu + (x * sigma)


# If available, use C implementation
try:
    from _statistics import _normal_dist_inv_cdf
except ImportError:
    pass


class NormalDist:
    "Normal distribution of a random variable"
    # https://en.wikipedia.org/wiki/Normal_distribution
    # https://en.wikipedia.org/wiki/Variance#Properties

    __slots__ = {
        '_mu': 'Arithmetic mean of a normal distribution',
        '_sigma': 'Standard deviation of a normal distribution',
    }

    def __init__(self, mu=0.0, sigma=1.0):
        "NormalDist where mu is the mean and sigma is the standard deviation."
        if sigma < 0.0:
            raise StatisticsError('sigma must be non-negative')
        self._mu = float(mu)
        self._sigma = float(sigma)

    @classmethod
    def from_samples(cls, data):
        "Make a normal distribution instance from sample data."
        if not isinstance(data, (list, tuple)):
            data = list(data)
        xbar = fmean(data)
        return cls(xbar, stdev(data, xbar))

    def samples(self, n, *, seed=None):
        "Generate *n* samples for a given mean and standard deviation."
        gauss = random.gauss if seed is None else random.Random(seed).gauss
        mu, sigma = self._mu, self._sigma
        return [gauss(mu, sigma) for i in range(n)]

    def pdf(self, x):
        "Probability density function.  P(x <= X < x+dx) / dx"
        variance = self._sigma ** 2.0
        if not variance:
            raise StatisticsError('pdf() not defined when sigma is zero')
        return exp((x - self._mu)**2.0 / (-2.0*variance)) / sqrt(tau*variance)

    def cdf(self, x):
        "Cumulative distribution function.  P(X <= x)"
        if not self._sigma:
            raise StatisticsError('cdf() not defined when sigma is zero')
        return 0.5 * (1.0 + erf((x - self._mu) / (self._sigma * sqrt(2.0))))

    def inv_cdf(self, p):
        """Inverse cumulative distribution function.  x : P(X <= x) = p

        Finds the value of the random variable such that the probability of
        the variable being less than or equal to that value equals the given
        probability.

        This function is also called the percent point function or quantile
        function.
        """
        if p <= 0.0 or p >= 1.0:
            raise StatisticsError('p must be in the range 0.0 < p < 1.0')
        if self._sigma <= 0.0:
            raise StatisticsError('cdf() not defined when sigma at or below zero')
        return _normal_dist_inv_cdf(p, self._mu, self._sigma)

    def quantiles(self, n=4):
        """Divide into *n* continuous intervals with equal probability.

        Returns a list of (n - 1) cut points separating the intervals.

        Set *n* to 4 for quartiles (the default).  Set *n* to 10 for deciles.
        Set *n* to 100 for percentiles which gives the 99 cuts points that
        separate the normal distribution in to 100 equal sized groups.
        """
        return [self.inv_cdf(i / n) for i in range(1, n)]

    def overlap(self, other):
        """Compute the overlapping coefficient (OVL) between two normal distributions.

        Measures the agreement between two normal probability distributions.
        Returns a value between 0.0 and 1.0 giving the overlapping area in
        the two underlying probability density functions.

            >>> N1 = NormalDist(2.4, 1.6)
            >>> N2 = NormalDist(3.2, 2.0)
            >>> N1.overlap(N2)
            0.8035050657330205
        """
        # See: "The overlapping coefficient as a measure of agreement between
        # probability distributions and point estimation of the overlap of two
        # normal densities" -- Henry F. Inman and Edwin L. Bradley Jr
        # http://dx.doi.org/10.1080/03610928908830127
        if not isinstance(other, NormalDist):
            raise TypeError('Expected another NormalDist instance')
        X, Y = self, other
        if (Y._sigma, Y._mu) < (X._sigma, X._mu):  # sort to assure commutativity
            X, Y = Y, X
        X_var, Y_var = X.variance, Y.variance
        if not X_var or not Y_var:
            raise StatisticsError('overlap() not defined when sigma is zero')
        dv = Y_var - X_var
        dm = fabs(Y._mu - X._mu)
        if not dv:
            return 1.0 - erf(dm / (2.0 * X._sigma * sqrt(2.0)))
        a = X._mu * Y_var - Y._mu * X_var
        b = X._sigma * Y._sigma * sqrt(dm**2.0 + dv * log(Y_var / X_var))
        x1 = (a + b) / dv
        x2 = (a - b) / dv
        return 1.0 - (fabs(Y.cdf(x1) - X.cdf(x1)) + fabs(Y.cdf(x2) - X.cdf(x2)))

    def zscore(self, x):
        """Compute the Standard Score.  (x - mean) / stdev

        Describes *x* in terms of the number of standard deviations
        above or below the mean of the normal distribution.
        """
        # https://www.statisticshowto.com/probability-and-statistics/z-score/
        if not self._sigma:
            raise StatisticsError('zscore() not defined when sigma is zero')
        return (x - self._mu) / self._sigma

    @property
    def mean(self):
        "Arithmetic mean of the normal distribution."
        return self._mu

    @property
    def median(self):
        "Return the median of the normal distribution"
        return self._mu

    @property
    def mode(self):
        """Return the mode of the normal distribution

        The mode is the value x where which the probability density
        function (pdf) takes its maximum value.
        """
        return self._mu

    @property
    def stdev(self):
        "Standard deviation of the normal distribution."
        return self._sigma

    @property
    def variance(self):
        "Square of the standard deviation."
        return self._sigma ** 2.0

    def __add__(x1, x2):
        """Add a constant or another NormalDist instance.

        If *other* is a constant, translate mu by the constant,
        leaving sigma unchanged.

        If *other* is a NormalDist, add both the means and the variances.
        Mathematically, this works only if the two distributions are
        independent or if they are jointly normally distributed.
        """
        if isinstance(x2, NormalDist):
            return NormalDist(x1._mu + x2._mu, hypot(x1._sigma, x2._sigma))
        return NormalDist(x1._mu + x2, x1._sigma)

    def __sub__(x1, x2):
        """Subtract a constant or another NormalDist instance.

        If *other* is a constant, translate by the constant mu,
        leaving sigma unchanged.

        If *other* is a NormalDist, subtract the means and add the variances.
        Mathematically, this works only if the two distributions are
        independent or if they are jointly normally distributed.
        """
        if isinstance(x2, NormalDist):
            return NormalDist(x1._mu - x2._mu, hypot(x1._sigma, x2._sigma))
        return NormalDist(x1._mu - x2, x1._sigma)

    def __mul__(x1, x2):
        """Multiply both mu and sigma by a constant.

        Used for rescaling, perhaps to change measurement units.
        Sigma is scaled with the absolute value of the constant.
        """
        return NormalDist(x1._mu * x2, x1._sigma * fabs(x2))

    def __truediv__(x1, x2):
        """Divide both mu and sigma by a constant.

        Used for rescaling, perhaps to change measurement units.
        Sigma is scaled with the absolute value of the constant.
        """
        return NormalDist(x1._mu / x2, x1._sigma / fabs(x2))

    def __pos__(x1):
        "Return a copy of the instance."
        return NormalDist(x1._mu, x1._sigma)

    def __neg__(x1):
        "Negates mu while keeping sigma the same."
        return NormalDist(-x1._mu, x1._sigma)

    __radd__ = __add__

    def __rsub__(x1, x2):
        "Subtract a NormalDist from a constant or another NormalDist."
        return -(x1 - x2)

    __rmul__ = __mul__

    def __eq__(x1, x2):
        "Two NormalDist objects are equal if their mu and sigma are both equal."
        if not isinstance(x2, NormalDist):
            return NotImplemented
        return x1._mu == x2._mu and x1._sigma == x2._sigma

    def __hash__(self):
        "NormalDist objects hash equal if their mu and sigma are both equal."
        return hash((self._mu, self._sigma))

    def __repr__(self):
        return f'{type(self).__name__}(mu={self._mu!r}, sigma={self._sigma!r})'

    def __getstate__(self):
        return self._mu, self._sigma

    def __setstate__(self, state):
        self._mu, self._sigma = state
<|endoftext|>"""Recognize image file formats based on their first few bytes."""

from os import PathLike

__all__ = ["what"]

#-------------------------#
# Recognize image headers #
#-------------------------#

def what(file, h=None):
    f = None
    try:
        if h is None:
            if isinstance(file, (str, PathLike)):
                f = open(file, 'rb')
                h = f.read(32)
            else:
                location = file.tell()
                h = file.read(32)
                file.seek(location)
        for tf in tests:
            res = tf(h, f)
            if res:
                return res
    finally:
        if f: f.close()
    return None


#---------------------------------#
# Subroutines per image file type #
#---------------------------------#

tests = []

def test_jpeg(h, f):
    """JPEG data in JFIF or Exif format"""
    if h[6:10] in (b'JFIF', b'Exif'):
        return 'jpeg'

tests.append(test_jpeg)

def test_png(h, f):
    if h.startswith(b'\211PNG\r\n\032\n'):
        return 'png'

tests.append(test_png)

def test_gif(h, f):
    """GIF ('87 and '89 variants)"""
    if h[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'

tests.append(test_gif)

def test_tiff(h, f):
    """TIFF (can be in Motorola or Intel byte order)"""
    if h[:2] in (b'MM', b'II'):
        return 'tiff'

tests.append(test_tiff)

def test_rgb(h, f):
    """SGI image library"""
    if h.startswith(b'\001\332'):
        return 'rgb'

tests.append(test_rgb)

def test_pbm(h, f):
    """PBM (portable bitmap)"""
    if len(h) >= 3 and \
        h[0] == ord(b'P') and h[1] in b'14' and h[2] in b' \t\n\r':
        return 'pbm'

tests.append(test_pbm)

def test_pgm(h, f):
    """PGM (portable graymap)"""
    if len(h) >= 3 and \
        h[0] == ord(b'P') and h[1] in b'25' and h[2] in b' \t\n\r':
        return 'pgm'

tests.append(test_pgm)

def test_ppm(h, f):
    """PPM (portable pixmap)"""
    if len(h) >= 3 and \
        h[0] == ord(b'P') and h[1] in b'36' and h[2] in b' \t\n\r':
        return 'ppm'

tests.append(test_ppm)

def test_rast(h, f):
    """Sun raster file"""
    if h.startswith(b'\x59\xA6\x6A\x95'):
        return 'rast'

tests.append(test_rast)

def test_xbm(h, f):
    """X bitmap (X10 or X11)"""
    if h.startswith(b'#define '):
        return 'xbm'

tests.append(test_xbm)

def test_bmp(h, f):
    if h.startswith(b'BM'):
        return 'bmp'

tests.append(test_bmp)

def test_webp(h, f):
    if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
        return 'webp'

tests.append(test_webp)

def test_exr(h, f):
    if h.startswith(b'\x76\x2f\x31\x01'):
        return 'exr'

tests.append(test_exr)

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
<|endoftext|>"""Interface to the libbzip2 compression library.

This module provides a file interface, classes for incremental
(de)compression, and functions for one-shot (de)compression.
"""

__all__ = ["BZ2File", "BZ2Compressor", "BZ2Decompressor",
           "open", "compress", "decompress"]

__author__ = "Nadeem Vawda <nadeem.vawda@gmail.com>"

from builtins import open as _builtin_open
import io
import os
import _compression

from _bz2 import BZ2Compressor, BZ2Decompressor


_MODE_CLOSED   = 0
_MODE_READ     = 1
# Value 2 no longer used
_MODE_WRITE    = 3


class BZ2File(_compression.BaseStream):

    """A file object providing transparent bzip2 (de)compression.

    A BZ2File can act as a wrapper for an existing file object, or refer
    directly to a named file on disk.

    Note that BZ2File provides a *binary* file interface - data read is
    returned as bytes, and data to be written should be given as bytes.
    """

    def __init__(self, filename, mode="r", *, compresslevel=9):
        """Open a bzip2-compressed file.

        If filename is a str, bytes, or PathLike object, it gives the
        name of the file to be opened. Otherwise, it should be a file
        object, which will be used to read or write the compressed data.

        mode can be 'r' for reading (default), 'w' for (over)writing,
        'x' for creating exclusively, or 'a' for appending. These can
        equivalently be given as 'rb', 'wb', 'xb', and 'ab'.

        If mode is 'w', 'x' or 'a', compresslevel can be a number between 1
        and 9 specifying the level of compression: 1 produces the least
        compression, and 9 (default) produces the most compression.

        If mode is 'r', the input file may be the concatenation of
        multiple compressed streams.
        """
        self._fp = None
        self._closefp = False
        self._mode = _MODE_CLOSED

        if not (1 <= compresslevel <= 9):
            raise ValueError("compresslevel must be between 1 and 9")

        if mode in ("", "r", "rb"):
            mode = "rb"
            mode_code = _MODE_READ
        elif mode in ("w", "wb"):
            mode = "wb"
            mode_code = _MODE_WRITE
            self._compressor = BZ2Compressor(compresslevel)
        elif mode in ("x", "xb"):
            mode = "xb"
            mode_code = _MODE_WRITE
            self._compressor = BZ2Compressor(compresslevel)
        elif mode in ("a", "ab"):
            mode = "ab"
            mode_code = _MODE_WRITE
            self._compressor = BZ2Compressor(compresslevel)
        else:
            raise ValueError("Invalid mode: %r" % (mode,))

        if isinstance(filename, (str, bytes, os.PathLike)):
            self._fp = _builtin_open(filename, mode)
            self._closefp = True
            self._mode = mode_code
        elif hasattr(filename, "read") or hasattr(filename, "write"):
            self._fp = filename
            self._mode = mode_code
        else:
            raise TypeError("filename must be a str, bytes, file or PathLike object")

        if self._mode == _MODE_READ:
            raw = _compression.DecompressReader(self._fp,
                BZ2Decompressor, trailing_error=OSError)
            self._buffer = io.BufferedReader(raw)
        else:
            self._pos = 0

    def close(self):
        """Flush and close the file.

        May be called more than once without error. Once the file is
        closed, any other operation on it will raise a ValueError.
        """
        if self._mode == _MODE_CLOSED:
            return
        try:
            if self._mode == _MODE_READ:
                self._buffer.close()
            elif self._mode == _MODE_WRITE:
                self._fp.write(self._compressor.flush())
                self._compressor = None
        finally:
            try:
                if self._closefp:
                    self._fp.close()
            finally:
                self._fp = None
                self._closefp = False
                self._mode = _MODE_CLOSED
                self._buffer = None

    @property
    def closed(self):
        """True if this file is closed."""
        return self._mode == _MODE_CLOSED

    def fileno(self):
        """Return the file descriptor for the underlying file."""
        self._check_not_closed()
        return self._fp.fileno()

    def seekable(self):
        """Return whether the file supports seeking."""
        return self.readable() and self._buffer.seekable()

    def readable(self):
        """Return whether the file was opened for reading."""
        self._check_not_closed()
        return self._mode == _MODE_READ

    def writable(self):
        """Return whether the file was opened for writing."""
        self._check_not_closed()
        return self._mode == _MODE_WRITE

    def peek(self, n=0):
        """Return buffered data without advancing the file position.

        Always returns at least one byte of data, unless at EOF.
        The exact number of bytes returned is unspecified.
        """
        self._check_can_read()
        # Relies on the undocumented fact that BufferedReader.peek()
        # always returns at least one byte (except at EOF), independent
        # of the value of n
        return self._buffer.peek(n)

    def read(self, size=-1):
        """Read up to size uncompressed bytes from the file.

        If size is negative or omitted, read until EOF is reached.
        Returns b'' if the file is already at EOF.
        """
        self._check_can_read()
        return self._buffer.read(size)

    def read1(self, size=-1):
        """Read up to size uncompressed bytes, while trying to avoid
        making multiple reads from the underlying stream. Reads up to a
        buffer's worth of data if size is negative.

        Returns b'' if the file is at EOF.
        """
        self._check_can_read()
        if size < 0:
            size = io.DEFAULT_BUFFER_SIZE
        return self._buffer.read1(size)

    def readinto(self, b):
        """Read bytes into b.

        Returns the number of bytes read (0 for EOF).
        """
        self._check_can_read()
        return self._buffer.readinto(b)

    def readline(self, size=-1):
        """Read a line of uncompressed bytes from the file.

        The terminating newline (if present) is retained. If size is
        non-negative, no more than size bytes will be read (in which
        case the line may be incomplete). Returns b'' if already at EOF.
        """
        if not isinstance(size, int):
            if not hasattr(size, "__index__"):
                raise TypeError("Integer argument expected")
            size = size.__index__()
        self._check_can_read()
        return self._buffer.readline(size)

    def readlines(self, size=-1):
        """Read a list of lines of uncompressed bytes from the file.

        size can be specified to control the number of lines read: no
        further lines will be read once the total size of the lines read
        so far equals or exceeds size.
        """
        if not isinstance(size, int):
            if not hasattr(size, "__index__"):
                raise TypeError("Integer argument expected")
            size = size.__index__()
        self._check_can_read()
        return self._buffer.readlines(size)

    def write(self, data):
        """Write a byte string to the file.

        Returns the number of uncompressed bytes written, which is
        always the length of data in bytes. Note that due to buffering,
        the file on disk may not reflect the data written until close()
        is called.
        """
        self._check_can_write()
        if isinstance(data, (bytes, bytearray)):
            length = len(data)
        else:
            # accept any data that supports the buffer protocol
            data = memoryview(data)
            length = data.nbytes

        compressed = self._compressor.compress(data)
        self._fp.write(compressed)
        self._pos += length
        return length

    def writelines(self, seq):
        """Write a sequence of byte strings to the file.

        Returns the number of uncompressed bytes written.
        seq can be any iterable yielding byte strings.

        Line separators are not added between the written byte strings.
        """
        return _compression.BaseStream.writelines(self, seq)

    def seek(self, offset, whence=io.SEEK_SET):
        """Change the file position.

        The new position is specified by offset, relative to the
        position indicated by whence. Values for whence are:

            0: start of stream (default); offset must not be negative
            1: current stream position
            2: end of stream; offset must not be positive

        Returns the new file position.

        Note that seeking is emulated, so depending on the parameters,
        this operation may be extremely slow.
        """
        self._check_can_seek()
        return self._buffer.seek(offset, whence)

    def tell(self):
        """Return the current file position."""
        self._check_not_closed()
        if self._mode == _MODE_READ:
            return self._buffer.tell()
        return self._pos


def open(filename, mode="rb", compresslevel=9,
         encoding=None, errors=None, newline=None):
    """Open a bzip2-compressed file in binary or text mode.

    The filename argument can be an actual filename (a str, bytes, or
    PathLike object), or an existing file object to read from or write
    to.

    The mode argument can be "r", "rb", "w", "wb", "x", "xb", "a" or
    "ab" for binary mode, or "rt", "wt", "xt" or "at" for text mode.
    The default mode is "rb", and the default compresslevel is 9.

    For binary mode, this function is equivalent to the BZ2File
    constructor: BZ2File(filename, mode, compresslevel). In this case,
    the encoding, errors and newline arguments must not be provided.

    For text mode, a BZ2File object is created, and wrapped in an
    io.TextIOWrapper instance with the specified encoding, error
    handling behavior, and line ending(s).

    """
    if "t" in mode:
        if "b" in mode:
            raise ValueError("Invalid mode: %r" % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if errors is not None:
            raise ValueError("Argument 'errors' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")

    bz_mode = mode.replace("t", "")
    binary_file = BZ2File(filename, bz_mode, compresslevel=compresslevel)

    if "t" in mode:
        encoding = io.text_encoding(encoding)
        return io.TextIOWrapper(binary_file, encoding, errors, newline)
    else:
        return binary_file


def compress(data, compresslevel=9):
    """Compress a block of data.

    compresslevel, if given, must be a number between 1 and 9.

    For incremental compression, use a BZ2Compressor object instead.
    """
    comp = BZ2Compressor(compresslevel)
    return comp.compress(data) + comp.flush()


def decompress(data):
    """Decompress a block of data.

    For incremental decompression, use a BZ2Decompressor object instead.
    """
    results = []
    while data:
        decomp = BZ2Decompressor()
        try:
            res = decomp.decompress(data)
        except OSError:
            if results:
                break  # Leftover data is not a valid bzip2 stream; ignore it.
            else:
                raise  # Error on the first iteration; bail out.
        results.append(res)
        if not decomp.eof:
            raise ValueError("Compressed data ended before the "
                             "end-of-stream marker was reached")
        data = decomp.unused_data
    return b"".join(results)
<|endoftext|>"""
Basic subprocess implementation for POSIX which only uses os functions. Only
implement features required by setup.py to build C extension modules when
subprocess is unavailable. setup.py is not used on Windows.
"""
import os


# distutils.spawn used by distutils.command.build_ext
# calls subprocess.Popen().wait()
class Popen:
    def __init__(self, cmd, env=None):
        self._cmd = cmd
        self._env = env
        self.returncode = None

    def wait(self):
        pid = os.fork()
        if pid == 0:
            # Child process
            try:
                if self._env is not None:
                    os.execve(self._cmd[0], self._cmd, self._env)
                else:
                    os.execv(self._cmd[0], self._cmd)
            finally:
                os._exit(1)
        else:
            # Parent process
            _, status = os.waitpid(pid, 0)
            self.returncode = os.waitstatus_to_exitcode(status)

        return self.returncode


def _check_cmd(cmd):
    # Use regex [a-zA-Z0-9./-]+: reject empty string, space, etc.
    safe_chars = []
    for first, last in (("a", "z"), ("A", "Z"), ("0", "9")):
        for ch in range(ord(first), ord(last) + 1):
            safe_chars.append(chr(ch))
    safe_chars.append("./-")
    safe_chars = ''.join(safe_chars)

    if isinstance(cmd, (tuple, list)):
        check_strs = cmd
    elif isinstance(cmd, str):
        check_strs = [cmd]
    else:
        return False

    for arg in check_strs:
        if not isinstance(arg, str):
            return False
        if not arg:
            # reject empty string
            return False
        for ch in arg:
            if ch not in safe_chars:
                return False

    return True


# _aix_support used by distutil.util calls subprocess.check_output()
def check_output(cmd, **kwargs):
    if kwargs:
        raise NotImplementedError(repr(kwargs))

    if not _check_cmd(cmd):
        raise ValueError(f"unsupported command: {cmd!r}")

    tmp_filename = "check_output.tmp"
    if not isinstance(cmd, str):
        cmd = " ".join(cmd)
    cmd = f"{cmd} >{tmp_filename}"

    try:
        # system() spawns a shell
        status = os.system(cmd)
        exitcode = os.waitstatus_to_exitcode(status)
        if exitcode:
            raise ValueError(f"Command {cmd!r} returned non-zero "
                             f"exit status {exitcode!r}")

        try:
            with open(tmp_filename, "rb") as fp:
                stdout = fp.read()
        except FileNotFoundError:
            stdout = b''
    finally:
        try:
            os.unlink(tmp_filename)
        except OSError:
            pass

    return stdout
<|endoftext|>"""Locale support module.

The module provides low-level access to the C lib's locale APIs and adds high
level number formatting APIs as well as a locale aliasing engine to complement
these.

The aliasing engine includes support for many commonly used locale names and
maps them to values suitable for passing to the C lib's setlocale() function. It
also includes default encodings for all supported locale names.

"""

import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools

# Try importing the _locale module.
#
# If this fails, fall back on a basic 'C' locale emulation.

# Yuck:  LC_MESSAGES is non-standard:  can't tell whether it exists before
# trying the import.  So __all__ is also fiddled at the end of the file.
__all__ = ["getlocale", "getdefaultlocale", "getpreferredencoding", "Error",
           "setlocale", "resetlocale", "localeconv", "strcoll", "strxfrm",
           "str", "atof", "atoi", "format", "format_string", "currency",
           "normalize", "LC_CTYPE", "LC_COLLATE", "LC_TIME", "LC_MONETARY",
           "LC_NUMERIC", "LC_ALL", "CHAR_MAX"]

def _strcoll(a,b):
    """ strcoll(string,string) -> int.
        Compares two strings according to the locale.
    """
    return (a > b) - (a < b)

def _strxfrm(s):
    """ strxfrm(string) -> string.
        Returns a string that behaves for cmp locale-aware.
    """
    return s

try:

    from _locale import *

except ImportError:

    # Locale emulation

    CHAR_MAX = 127
    LC_ALL = 6
    LC_COLLATE = 3
    LC_CTYPE = 0
    LC_MESSAGES = 5
    LC_MONETARY = 4
    LC_NUMERIC = 1
    LC_TIME = 2
    Error = ValueError

    def localeconv():
        """ localeconv() -> dict.
            Returns numeric and monetary locale-specific parameters.
        """
        # 'C' locale default values
        return {'grouping': [127],
                'currency_symbol': '',
                'n_sign_posn': 127,
                'p_cs_precedes': 127,
                'n_cs_precedes': 127,
                'mon_grouping': [],
                'n_sep_by_space': 127,
                'decimal_point': '.',
                'negative_sign': '',
                'positive_sign': '',
                'p_sep_by_space': 127,
                'int_curr_symbol': '',
                'p_sign_posn': 127,
                'thousands_sep': '',
                'mon_thousands_sep': '',
                'frac_digits': 127,
                'mon_decimal_point': '',
                'int_frac_digits': 127}

    def setlocale(category, value=None):
        """ setlocale(integer,string=None) -> string.
            Activates/queries locale processing.
        """
        if value not in (None, '', 'C'):
            raise Error('_locale emulation only supports "C" locale')
        return 'C'

# These may or may not exist in _locale, so be sure to set them.
if 'strxfrm' not in globals():
    strxfrm = _strxfrm
if 'strcoll' not in globals():
    strcoll = _strcoll


_localeconv = localeconv

# With this dict, you can override some items of localeconv's return value.
# This is useful for testing purposes.
_override_localeconv = {}

@functools.wraps(_localeconv)
def localeconv():
    d = _localeconv()
    if _override_localeconv:
        d.update(_override_localeconv)
    return d


### Number formatting APIs

# Author: Martin von Loewis
# improved by Georg Brandl

# Iterate over grouping intervals
def _grouping_intervals(grouping):
    last_interval = None
    for interval in grouping:
        # if grouping is -1, we are done
        if interval == CHAR_MAX:
            return
        # 0: re-use last group ad infinitum
        if interval == 0:
            if last_interval is None:
                raise ValueError("invalid grouping")
            while True:
                yield last_interval
        yield interval
        last_interval = interval

#perform the grouping from right to left
def _group(s, monetary=False):
    conv = localeconv()
    thousands_sep = conv[monetary and 'mon_thousands_sep' or 'thousands_sep']
    grouping = conv[monetary and 'mon_grouping' or 'grouping']
    if not grouping:
        return (s, 0)
    if s[-1] == ' ':
        stripped = s.rstrip()
        right_spaces = s[len(stripped):]
        s = stripped
    else:
        right_spaces = ''
    left_spaces = ''
    groups = []
    for interval in _grouping_intervals(grouping):
        if not s or s[-1] not in "0123456789":
            # only non-digit characters remain (sign, spaces)
            left_spaces = s
            s = ''
            break
        groups.append(s[-interval:])
        s = s[:-interval]
    if s:
        groups.append(s)
    groups.reverse()
    return (
        left_spaces + thousands_sep.join(groups) + right_spaces,
        len(thousands_sep) * (len(groups) - 1)
    )

# Strip a given amount of excess padding from the given string
def _strip_padding(s, amount):
    lpos = 0
    while amount and s[lpos] == ' ':
        lpos += 1
        amount -= 1
    rpos = len(s) - 1
    while amount and s[rpos] == ' ':
        rpos -= 1
        amount -= 1
    return s[lpos:rpos+1]

_percent_re = re.compile(r'%(?:\((?P<key>.*?)\))?'
                         r'(?P<modifiers>[-#0-9 +*.hlL]*?)[eEfFgGdiouxXcrs%]')

def _format(percent, value, grouping=False, monetary=False, *additional):
    if additional:
        formatted = percent % ((value,) + additional)
    else:
        formatted = percent % value
    if percent[-1] in 'eEfFgGdiu':
        formatted = _localize(formatted, grouping, monetary)
    return formatted

# Transform formatted as locale number according to the locale settings
def _localize(formatted, grouping=False, monetary=False):
    # floats and decimal ints need special action!
    if '.' in formatted:
        seps = 0
        parts = formatted.split('.')
        if grouping:
            parts[0], seps = _group(parts[0], monetary=monetary)
        decimal_point = localeconv()[monetary and 'mon_decimal_point'
                                              or 'decimal_point']
        formatted = decimal_point.join(parts)
        if seps:
            formatted = _strip_padding(formatted, seps)
    else:
        seps = 0
        if grouping:
            formatted, seps = _group(formatted, monetary=monetary)
        if seps:
            formatted = _strip_padding(formatted, seps)
    return formatted

def format_string(f, val, grouping=False, monetary=False):
    """Formats a string in the same way that the % formatting would use,
    but takes the current locale into account.

    Grouping is applied if the third parameter is true.
    Conversion uses monetary thousands separator and grouping strings if
    forth parameter monetary is true."""
    percents = list(_percent_re.finditer(f))
    new_f = _percent_re.sub('%s', f)

    if isinstance(val, _collections_abc.Mapping):
        new_val = []
        for perc in percents:
            if perc.group()[-1]=='%':
                new_val.append('%')
            else:
                new_val.append(_format(perc.group(), val, grouping, monetary))
    else:
        if not isinstance(val, tuple):
            val = (val,)
        new_val = []
        i = 0
        for perc in percents:
            if perc.group()[-1]=='%':
                new_val.append('%')
            else:
                starcount = perc.group('modifiers').count('*')
                new_val.append(_format(perc.group(),
                                      val[i],
                                      grouping,
                                      monetary,
                                      *val[i+1:i+1+starcount]))
                i += (1 + starcount)
    val = tuple(new_val)

    return new_f % val

def format(percent, value, grouping=False, monetary=False, *additional):
    """Deprecated, use format_string instead."""
    import warnings
    warnings.warn(
        "This method will be removed in a future version of Python. "
        "Use 'locale.format_string()' instead.",
        DeprecationWarning, stacklevel=2
    )

    match = _percent_re.match(percent)
    if not match or len(match.group())!= len(percent):
        raise ValueError(("format() must be given exactly one %%char "
                         "format specifier, %s not valid") % repr(percent))
    return _format(percent, value, grouping, monetary, *additional)

def currency(val, symbol=True, grouping=False, international=False):
    """Formats val according to the currency settings
    in the current locale."""
    conv = localeconv()

    # check for illegal values
    digits = conv[international and 'int_frac_digits' or 'frac_digits']
    if digits == 127:
        raise ValueError("Currency formatting is not possible using "
                         "the 'C' locale.")

    s = _localize(f'{abs(val):.{digits}f}', grouping, monetary=True)
    # '<' and '>' are markers if the sign must be inserted between symbol and value
    s = '<' + s + '>'

    if symbol:
        smb = conv[international and 'int_curr_symbol' or 'currency_symbol']
        precedes = conv[val<0 and 'n_cs_precedes' or 'p_cs_precedes']
        separated = conv[val<0 and 'n_sep_by_space' or 'p_sep_by_space']

        if precedes:
            s = smb + (separated and ' ' or '') + s
        else:
            if international and smb[-1] == ' ':
                smb = smb[:-1]
            s = s + (separated and ' ' or '') + smb

    sign_pos = conv[val<0 and 'n_sign_posn' or 'p_sign_posn']
    sign = conv[val<0 and 'negative_sign' or 'positive_sign']

    if sign_pos == 0:
        s = '(' + s + ')'
    elif sign_pos == 1:
        s = sign + s
    elif sign_pos == 2:
        s = s + sign
    elif sign_pos == 3:
        s = s.replace('<', sign)
    elif sign_pos == 4:
        s = s.replace('>', sign)
    else:
        # the default if nothing specified;
        # this should be the most fitting sign position
        s = sign + s

    return s.replace('<', '').replace('>', '')

def str(val):
    """Convert float to string, taking the locale into account."""
    return _format("%.12g", val)

def delocalize(string):
    "Parses a string as a normalized number according to the locale settings."

    conv = localeconv()

    #First, get rid of the grouping
    ts = conv['thousands_sep']
    if ts:
        string = string.replace(ts, '')

    #next, replace the decimal point with a dot
    dd = conv['decimal_point']
    if dd:
        string = string.replace(dd, '.')
    return string

def localize(string, grouping=False, monetary=False):
    """Parses a string as locale number according to the locale settings."""
    return _localize(string, grouping, monetary)

def atof(string, func=float):
    "Parses a string as a float according to the locale settings."
    return func(delocalize(string))

def atoi(string):
    "Converts a string to an integer according to the locale settings."
    return int(delocalize(string))

def _test():
    setlocale(LC_ALL, "")
    #do grouping
    s1 = format_string("%d", 123456789,1)
    print(s1, "is", atoi(s1))
    #standard formatting
    s1 = str(3.14)
    print(s1, "is", atof(s1))

### Locale name aliasing engine

# Author: Marc-Andre Lemburg, mal@lemburg.com
# Various tweaks by Fredrik Lundh <fredrik@pythonware.com>

# store away the low-level version of setlocale (it's
# overridden below)
_setlocale = setlocale

def _replace_encoding(code, encoding):
    if '.' in code:
        langname = code[:code.index('.')]
    else:
        langname = code
    # Convert the encoding to a C lib compatible encoding string
    norm_encoding = encodings.normalize_encoding(encoding)
    #print('norm encoding: %r' % norm_encoding)
    norm_encoding = encodings.aliases.aliases.get(norm_encoding.lower(),
                                                  norm_encoding)
    #print('aliased encoding: %r' % norm_encoding)
    encoding = norm_encoding
    norm_encoding = norm_encoding.lower()
    if norm_encoding in locale_encoding_alias:
        encoding = locale_encoding_alias[norm_encoding]
    else:
        norm_encoding = norm_encoding.replace('_', '')
        norm_encoding = norm_encoding.replace('-', '')
        if norm_encoding in locale_encoding_alias:
            encoding = locale_encoding_alias[norm_encoding]
    #print('found encoding %r' % encoding)
    return langname + '.' + encoding

def _append_modifier(code, modifier):
    if modifier == 'euro':
        if '.' not in code:
            return code + '.ISO8859-15'
        _, _, encoding = code.partition('.')
        if encoding in ('ISO8859-15', 'UTF-8'):
            return code
        if encoding == 'ISO8859-1':
            return _replace_encoding(code, 'ISO8859-15')
    return code + '@' + modifier

def normalize(localename):

    """ Returns a normalized locale code for the given locale
        name.

        The returned locale code is formatted for use with
        setlocale().

        If normalization fails, the original name is returned
        unchanged.

        If the given encoding is not known, the function defaults to
        the default encoding for the locale code just like setlocale()
        does.

    """
    # Normalize the locale name and extract the encoding and modifier
    code = localename.lower()
    if ':' in code:
        # ':' is sometimes used as encoding delimiter.
        code = code.replace(':', '.')
    if '@' in code:
        code, modifier = code.split('@', 1)
    else:
        modifier = ''
    if '.' in code:
        langname, encoding = code.split('.')[:2]
    else:
        langname = code
        encoding = ''

    # First lookup: fullname (possibly with encoding and modifier)
    lang_enc = langname
    if encoding:
        norm_encoding = encoding.replace('-', '')
        norm_encoding = norm_encoding.replace('_', '')
        lang_enc += '.' + norm_encoding
    lookup_name = lang_enc
    if modifier:
        lookup_name += '@' + modifier
    code = locale_alias.get(lookup_name, None)
    if code is not None:
        return code
    #print('first lookup failed')

    if modifier:
        # Second try: fullname without modifier (possibly with encoding)
        code = locale_alias.get(lang_enc, None)
        if code is not None:
            #print('lookup without modifier succeeded')
            if '@' not in code:
                return _append_modifier(code, modifier)
            if code.split('@', 1)[1].lower() == modifier:
                return code
        #print('second lookup failed')

    if encoding:
        # Third try: langname (without encoding, possibly with modifier)
        lookup_name = langname
        if modifier:
            lookup_name += '@' + modifier
        code = locale_alias.get(lookup_name, None)
        if code is not None:
            #print('lookup without encoding succeeded')
            if '@' not in code:
                return _replace_encoding(code, encoding)
            code, modifier = code.split('@', 1)
            return _replace_encoding(code, encoding) + '@' + modifier

        if modifier:
            # Fourth try: langname (without encoding and modifier)
            code = locale_alias.get(langname, None)
            if code is not None:
                #print('lookup without modifier and encoding succeeded')
                if '@' not in code:
                    code = _replace_encoding(code, encoding)
                    return _append_modifier(code, modifier)
                code, defmod = code.split('@', 1)
                if defmod.lower() == modifier:
                    return _replace_encoding(code, encoding) + '@' + defmod

    return localename

def _parse_localename(localename):

    """ Parses the locale code for localename and returns the
        result as tuple (language code, encoding).

        The localename is normalized and passed through the locale
        alias engine. A ValueError is raised in case the locale name
        cannot be parsed.

        The language code corresponds to RFC 1766.  code and encoding
        can be None in case the values cannot be determined or are
        unknown to this implementation.

    """
    code = normalize(localename)
    if '@' in code:
        # Deal with locale modifiers
        code, modifier = code.split('@', 1)
        if modifier == 'euro' and '.' not in code:
            # Assume Latin-9 for @euro locales. This is bogus,
            # since some systems may use other encodings for these
            # locales. Also, we ignore other modifiers.
            return code, 'iso-8859-15'

    if '.' in code:
        return tuple(code.split('.')[:2])
    elif code == 'C':
        return None, None
    elif code == 'UTF-8':
        # On macOS "LC_CTYPE=UTF-8" is a valid locale setting
        # for getting UTF-8 handling for text.
        return None, 'UTF-8'
    raise ValueError('unknown locale: %s' % localename)

def _build_localename(localetuple):

    """ Builds a locale code from the given tuple (language code,
        encoding).

        No aliasing or normalizing takes place.

    """
    try:
        language, encoding = localetuple

        if language is None:
            language = 'C'
        if encoding is None:
            return language
        else:
            return language + '.' + encoding
    except (TypeError, ValueError):
        raise TypeError('Locale must be None, a string, or an iterable of '
                        'two strings -- language code, encoding.') from None

def getdefaultlocale(envvars=('LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE')):

    """ Tries to determine the default locale settings and returns
        them as tuple (language code, encoding).

        According to POSIX, a program which has not called
        setlocale(LC_ALL, "") runs using the portable 'C' locale.
        Calling setlocale(LC_ALL, "") lets it use the default locale as
        defined by the LANG variable. Since we don't want to interfere
        with the current locale setting we thus emulate the behavior
        in the way described above.

        To maintain compatibility with other platforms, not only the
        LANG variable is tested, but a list of variables given as
        envvars parameter. The first found to be defined will be
        used. envvars defaults to the search path used in GNU gettext;
        it must always contain the variable name 'LANG'.

        Except for the code 'C', the language code corresponds to RFC
        1766.  code and encoding can be None in case the values cannot
        be determined.

    """

    try:
        # check if it's supported by the _locale module
        import _locale
        code, encoding = _locale._getdefaultlocale()
    except (ImportError, AttributeError):
        pass
    else:
        # make sure the code/encoding values are valid
        if sys.platform == "win32" and code and code[:2] == "0x":
            # map windows language identifier to language name
            code = windows_locale.get(int(code, 0))
        # ...add other platform-specific processing here, if
        # necessary...
        return code, encoding

    # fall back on POSIX behaviour
    import os
    lookup = os.environ.get
    for variable in envvars:
        localename = lookup(variable,None)
        if localename:
            if variable == 'LANGUAGE':
                localename = localename.split(':')[0]
            break
    else:
        localename = 'C'
    return _parse_localename(localename)


def getlocale(category=LC_CTYPE):

    """ Returns the current setting for the given locale category as
        tuple (language code, encoding).

        category may be one of the LC_* value except LC_ALL. It
        defaults to LC_CTYPE.

        Except for the code 'C', the language code corresponds to RFC
        1766.  code and encoding can be None in case the values cannot
        be determined.

    """
    localename = _setlocale(category)
    if category == LC_ALL and ';' in localename:
        raise TypeError('category LC_ALL is not supported')
    return _parse_localename(localename)

def setlocale(category, locale=None):

    """ Set the locale for the given category.  The locale can be
        a string, an iterable of two strings (language code and encoding),
        or None.

        Iterables are converted to strings using the locale aliasing
        engine.  Locale strings are passed directly to the C lib.

        category may be given as one of the LC_* values.

    """
    if locale and not isinstance(locale, _builtin_str):
        # convert to string
        locale = normalize(_build_localename(locale))
    return _setlocale(category, locale)

def resetlocale(category=LC_ALL):

    """ Sets the locale for category to the default setting.

        The default setting is determined by calling
        getdefaultlocale(). category defaults to LC_ALL.

    """
    _setlocale(category, _build_localename(getdefaultlocale()))


try:
    from _locale import _get_locale_encoding
except ImportError:
    def _get_locale_encoding():
        if hasattr(sys, 'getandroidapilevel'):
            # On Android langinfo.h and CODESET are missing, and UTF-8 is
            # always used in mbstowcs() and wcstombs().
            return 'UTF-8'
        if sys.flags.utf8_mode:
            return 'UTF-8'
        encoding = getdefaultlocale()[1]
        if encoding is None:
            # LANG not set, default conservatively to ASCII
            encoding = 'ascii'
        return encoding

try:
    CODESET
except NameError:
    def getpreferredencoding(do_setlocale=True):
        """Return the charset that the user is likely using."""
        return _get_locale_encoding()
else:
    # On Unix, if CODESET is available, use that.
    def getpreferredencoding(do_setlocale=True):
        """Return the charset that the user is likely using,
        according to the system configuration."""
        if sys.flags.utf8_mode:
            return 'UTF-8'

        if not do_setlocale:
            return _get_locale_encoding()

        old_loc = setlocale(LC_CTYPE)
        try:
            try:
                setlocale(LC_CTYPE, "")
            except Error:
                pass
            return _get_locale_encoding()
        finally:
            setlocale(LC_CTYPE, old_loc)


### Database
#
# The following data was extracted from the locale.alias file which
# comes with X11 and then hand edited removing the explicit encoding
# definitions and adding some more aliases. The file is usually
# available as /usr/lib/X11/locale/locale.alias.
#

#
# The local_encoding_alias table maps lowercase encoding alias names
# to C locale encoding names (case-sensitive). Note that normalize()
# first looks up the encoding in the encodings.aliases dictionary and
# then applies this mapping to find the correct C lib name for the
# encoding.
#
locale_encoding_alias = {

    # Mappings for non-standard encoding names used in locale names
    '437':                          'C',
    'c':                            'C',
    'en':                           'ISO8859-1',
    'jis':                          'JIS7',
    'jis7':                         'JIS7',
    'ajec':                         'eucJP',
    'koi8c':                        'KOI8-C',
    'microsoftcp1251':              'CP1251',
    'microsoftcp1255':              'CP1255',
    'microsoftcp1256':              'CP1256',
    '88591':                        'ISO8859-1',
    '88592':                        'ISO8859-2',
    '88595':                        'ISO8859-5',
    '885915':                       'ISO8859-15',

    # Mappings from Python codec names to C lib encoding names
    'ascii':                        'ISO8859-1',
    'latin_1':                      'ISO8859-1',
    'iso8859_1':                    'ISO8859-1',
    'iso8859_10':                   'ISO8859-10',
    'iso8859_11':                   'ISO8859-11',
    'iso8859_13':                   'ISO8859-13',
    'iso8859_14':                   'ISO8859-14',
    'iso8859_15':                   'ISO8859-15',
    'iso8859_16':                   'ISO8859-16',
    'iso8859_2':                    'ISO8859-2',
    'iso8859_3':                    'ISO8859-3',
    'iso8859_4':                    'ISO8859-4',
    'iso8859_5':                    'ISO8859-5',
    'iso8859_6':                    'ISO8859-6',
    'iso8859_7':                    'ISO8859-7',
    'iso8859_8':                    'ISO8859-8',
    'iso8859_9':                    'ISO8859-9',
    'iso2022_jp':                   'JIS7',
    'shift_jis':                    'SJIS',
    'tactis':                       'TACTIS',
    'euc_jp':                       'eucJP',
    'euc_kr':                       'eucKR',
    'utf_8':                        'UTF-8',
    'koi8_r':                       'KOI8-R',
    'koi8_t':                       'KOI8-T',
    'koi8_u':                       'KOI8-U',
    'kz1048':                       'RK1048',
    'cp1251':                       'CP1251',
    'cp1255':                       'CP1255',
    'cp1256':                       'CP1256',

    # XXX This list is still incomplete. If you know more
    # mappings, please file a bug report. Thanks.
}

for k, v in sorted(locale_encoding_alias.items()):
    k = k.replace('_', '')
    locale_encoding_alias.setdefault(k, v)

#
# The locale_alias table maps lowercase alias names to C locale names
# (case-sensitive). Encodings are always separated from the locale
# name using a dot ('.'); they should only be given in case the
# language name is needed to interpret the given encoding alias
# correctly (CJK codes often have this need).
#
# Note that the normalize() function which uses this tables
# removes '_' and '-' characters from the encoding part of the
# locale name before doing the lookup. This saves a lot of
# space in the table.
#
# MAL 2004-12-10:
# Updated alias mapping to most recent locale.alias file
# from X.org distribution using makelocalealias.py.
#
# These are the differences compared to the old mapping (Python 2.4
# and older):
#
#    updated 'bg' -> 'bg_BG.ISO8859-5' to 'bg_BG.CP1251'
#    updated 'bg_bg' -> 'bg_BG.ISO8859-5' to 'bg_BG.CP1251'
#    updated 'bulgarian' -> 'bg_BG.ISO8859-5' to 'bg_BG.CP1251'
#    updated 'cz' -> 'cz_CZ.ISO8859-2' to 'cs_CZ.ISO8859-2'
#    updated 'cz_cz' -> 'cz_CZ.ISO8859-2' to 'cs_CZ.ISO8859-2'
#    updated 'czech' -> 'cs_CS.ISO8859-2' to 'cs_CZ.ISO8859-2'
#    updated 'dutch' -> 'nl_BE.ISO8859-1' to 'nl_NL.ISO8859-1'
#    updated 'et' -> 'et_EE.ISO8859-4' to 'et_EE.ISO8859-15'
#    updated 'et_ee' -> 'et_EE.ISO8859-4' to 'et_EE.ISO8859-15'
#    updated 'fi' -> 'fi_FI.ISO8859-1' to 'fi_FI.ISO8859-15'
#    updated 'fi_fi' -> 'fi_FI.ISO8859-1' to 'fi_FI.ISO8859-15'
#    updated 'iw' -> 'iw_IL.ISO8859-8' to 'he_IL.ISO8859-8'
#    updated 'iw_il' -> 'iw_IL.ISO8859-8' to 'he_IL.ISO8859-8'
#    updated 'japanese' -> 'ja_JP.SJIS' to 'ja_JP.eucJP'
#    updated 'lt' -> 'lt_LT.ISO8859-4' to 'lt_LT.ISO8859-13'
#    updated 'lv' -> 'lv_LV.ISO8859-4' to 'lv_LV.ISO8859-13'
#    updated 'sl' -> 'sl_CS.ISO8859-2' to 'sl_SI.ISO8859-2'
#    updated 'slovene' -> 'sl_CS.ISO8859-2' to 'sl_SI.ISO8859-2'
#    updated 'th_th' -> 'th_TH.TACTIS' to 'th_TH.ISO8859-11'
#    updated 'zh_cn' -> 'zh_CN.eucCN' to 'zh_CN.gb2312'
#    updated 'zh_cn.big5' -> 'zh_TW.eucTW' to 'zh_TW.big5'
#    updated 'zh_tw' -> 'zh_TW.eucTW' to 'zh_TW.big5'
#
# MAL 2008-05-30:
# Updated alias mapping to most recent locale.alias file
# from X.org distribution using makelocalealias.py.
#
# These are the differences compared to the old mapping (Python 2.5
# and older):
#
#    updated 'cs_cs.iso88592' -> 'cs_CZ.ISO8859-2' to 'cs_CS.ISO8859-2'
#    updated 'serbocroatian' -> 'sh_YU.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sh' -> 'sh_YU.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sh_hr.iso88592' -> 'sh_HR.ISO8859-2' to 'hr_HR.ISO8859-2'
#    updated 'sh_sp' -> 'sh_YU.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sh_yu' -> 'sh_YU.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sp' -> 'sp_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sp_yu' -> 'sp_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr@cyrillic' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr_sp' -> 'sr_SP.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sr_yu' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr_yu.cp1251@cyrillic' -> 'sr_YU.CP1251' to 'sr_CS.CP1251'
#    updated 'sr_yu.iso88592' -> 'sr_YU.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sr_yu.iso88595' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr_yu.iso88595@cyrillic' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr_yu.microsoftcp1251@cyrillic' -> 'sr_YU.CP1251' to 'sr_CS.CP1251'
#    updated 'sr_yu.utf8@cyrillic' -> 'sr_YU.UTF-8' to 'sr_CS.UTF-8'
#    updated 'sr_yu@cyrillic' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#
# AP 2010-04-12:
# Updated alias mapping to most recent locale.alias file
# from X.org distribution using makelocalealias.py.
#
# These are the differences compared to the old mapping (Python 2.6.5
# and older):
#
#    updated 'ru' -> 'ru_RU.ISO8859-5' to 'ru_RU.UTF-8'
#    updated 'ru_ru' -> 'ru_RU.ISO8859-5' to 'ru_RU.UTF-8'
#    updated 'serbocroatian' -> 'sr_CS.ISO8859-2' to 'sr_RS.UTF-8@latin'
#    updated 'sh' -> 'sr_CS.ISO8859-2' to 'sr_RS.UTF-8@latin'
#    updated 'sh_yu' -> 'sr_CS.ISO8859-2' to 'sr_RS.UTF-8@latin'
#    updated 'sr' -> 'sr_CS.ISO8859-5' to 'sr_RS.UTF-8'
#    updated 'sr@cyrillic' -> 'sr_CS.ISO8859-5' to 'sr_RS.UTF-8'
#    updated 'sr@latn' -> 'sr_CS.ISO8859-2' to 'sr_RS.UTF-8@latin'
#    updated 'sr_cs.utf8@latn' -> 'sr_CS.UTF-8' to 'sr_RS.UTF-8@latin'
#    updated 'sr_cs@latn' -> 'sr_CS.ISO8859-2' to 'sr_RS.UTF-8@latin'
#    updated 'sr_yu' -> 'sr_CS.ISO8859-5' to 'sr_RS.UTF-8@latin'
#    updated 'sr_yu.utf8@cyrillic' -> 'sr_CS.UTF-8' to 'sr_RS.UTF-8'
#    updated 'sr_yu@cyrillic' -> 'sr_CS.ISO8859-5' to 'sr_RS.UTF-8'
#
# SS 2013-12-20:
# Updated alias mapping to most recent locale.alias file
# from X.org distribution using makelocalealias.py.
#
# These are the differences compared to the old mapping (Python 3.3.3
# and older):
#
#    updated 'a3' -> 'a3_AZ.KOI8-C' to 'az_AZ.KOI8-C'
#    updated 'a3_az' -> 'a3_AZ.KOI8-C' to 'az_AZ.KOI8-C'
#    updated 'a3_az.koi8c' -> 'a3_AZ.KOI8-C' to 'az_AZ.KOI8-C'
#    updated 'cs_cs.iso88592' -> 'cs_CS.ISO8859-2' to 'cs_CZ.ISO8859-2'
#    updated 'hebrew' -> 'iw_IL.ISO8859-8' to 'he_IL.ISO8859-8'
#    updated 'hebrew.iso88598' -> 'iw_IL.ISO8859-8' to 'he_IL.ISO8859-8'
#    updated 'sd' -> 'sd_IN@devanagari.UTF-8' to 'sd_IN.UTF-8'
#    updated 'sr@latn' -> 'sr_RS.UTF-8@latin' to 'sr_CS.UTF-8@latin'
#    updated 'sr_cs' -> 'sr_RS.UTF-8' to 'sr_CS.UTF-8'
#    updated 'sr_cs.utf8@latn' -> 'sr_RS.UTF-8@latin' to 'sr_CS.UTF-8@latin'
#    updated 'sr_cs@latn' -> 'sr_RS.UTF-8@latin' to 'sr_CS.UTF-8@latin'
#
# SS 2014-10-01:
# Updated alias mapping with glibc 2.19 supported locales.
#
# SS 2018-05-05:
# Updated alias mapping with glibc 2.27 supported locales.
#
# These are the differences compared to the old mapping (Python 3.6.5
# and older):
#
#    updated 'ca_es@valencia' -> 'ca_ES.ISO8859-15@valencia' to 'ca_ES.UTF-8@valencia'
#    updated 'kk_kz' -> 'kk_KZ.RK1048' to 'kk_KZ.ptcp154'
#    updated 'russian' -> 'ru_RU.ISO8859-5' to 'ru_RU.KOI8-R'

locale_alias = {
    'a3':                                   'az_AZ.KOI8-C',
    'a3_az':                                'az_AZ.KOI8-C',
    'a3_az.koic':                           'az_AZ.KOI8-C',
    'aa_dj':                                'aa_DJ.ISO8859-1',
    'aa_er':                                'aa_ER.UTF-8',
    'aa_et':                                'aa_ET.UTF-8',
    'af':                                   'af_ZA.ISO8859-1',
    'af_za':                                'af_ZA.ISO8859-1',
    'agr_pe':                               'agr_PE.UTF-8',
    'ak_gh':                                'ak_GH.UTF-8',
    'am':                                   'am_ET.UTF-8',
    'am_et':                                'am_ET.UTF-8',
    'american':                             'en_US.ISO8859-1',
    'an_es':                                'an_ES.ISO8859-15',
    'anp_in':                               'anp_IN.UTF-8',
    'ar':                                   'ar_AA.ISO8859-6',
    'ar_aa':                                'ar_AA.ISO8859-6',
    'ar_ae':                                'ar_AE.ISO8859-6',
    'ar_bh':                                'ar_BH.ISO8859-6',
    'ar_dz':                                'ar_DZ.ISO8859-6',
    'ar_eg':                                'ar_EG.ISO8859-6',
    'ar_in':                                'ar_IN.UTF-8',
    'ar_iq':                                'ar_IQ.ISO8859-6',
    'ar_jo':                                'ar_JO.ISO8859-6',
    'ar_kw':                                'ar_KW.ISO8859-6',
    'ar_lb':                                'ar_LB.ISO8859-6',
    'ar_ly':                                'ar_LY.ISO8859-6',
    'ar_ma':                                'ar_MA.ISO8859-6',
    'ar_om':                                'ar_OM.ISO8859-6',
    'ar_qa':                                'ar_QA.ISO8859-6',
    'ar_sa':                                'ar_SA.ISO8859-6',
    'ar_sd':                                'ar_SD.ISO8859-6',
    'ar_ss':                                'ar_SS.UTF-8',
    'ar_sy':                                'ar_SY.ISO8859-6',
    'ar_tn':                                'ar_TN.ISO8859-6',
    'ar_ye':                                'ar_YE.ISO8859-6',
    'arabic':                               'ar_AA.ISO8859-6',
    'as':                                   'as_IN.UTF-8',
    'as_in':                                'as_IN.UTF-8',
    'ast_es':                               'ast_ES.ISO8859-15',
    'ayc_pe':                               'ayc_PE.UTF-8',
    'az':                                   'az_AZ.ISO8859-9E',
    'az_az':                                'az_AZ.ISO8859-9E',
    'az_az.iso88599e':                      'az_AZ.ISO8859-9E',
    'az_ir':                                'az_IR.UTF-8',
    'be':                                   'be_BY.CP1251',
    'be@latin':                             'be_BY.UTF-8@latin',
    'be_bg.utf8':                           'bg_BG.UTF-8',
    'be_by':                                'be_BY.CP1251',
    'be_by@latin':                          'be_BY.UTF-8@latin',
    'bem_zm':                               'bem_ZM.UTF-8',
    'ber_dz':                               'ber_DZ.UTF-8',
    'ber_ma':                               'ber_MA.UTF-8',
    'bg':                                   'bg_BG.CP1251',
    'bg_bg':                                'bg_BG.CP1251',
    'bhb_in.utf8':                          'bhb_IN.UTF-8',
    'bho_in':                               'bho_IN.UTF-8',
    'bho_np':                               'bho_NP.UTF-8',
    'bi_vu':                                'bi_VU.UTF-8',
    'bn_bd':                                'bn_BD.UTF-8',
    'bn_in':                                'bn_IN.UTF-8',
    'bo_cn':                                'bo_CN.UTF-8',
    'bo_in':                                'bo_IN.UTF-8',
    'bokmal':                               'nb_NO.ISO8859-1',
    'bokm\xe5l':                            'nb_NO.ISO8859-1',
    'br':                                   'br_FR.ISO8859-1',
    'br_fr':                                'br_FR.ISO8859-1',
    'brx_in':                               'brx_IN.UTF-8',
    'bs':                                   'bs_BA.ISO8859-2',
    'bs_ba':                                'bs_BA.ISO8859-2',
    'bulgarian':                            'bg_BG.CP1251',
    'byn_er':                               'byn_ER.UTF-8',
    'c':                                    'C',
    'c-french':                             'fr_CA.ISO8859-1',
    'c.ascii':                              'C',
    'c.en':                                 'C',
    'c.iso88591':                           'en_US.ISO8859-1',
    'c.utf8':                               'en_US.UTF-8',
    'c_c':                                  'C',
    'c_c.c':                                'C',
    'ca':                                   'ca_ES.ISO8859-1',
    'ca_ad':                                'ca_AD.ISO8859-1',
    'ca_es':                                'ca_ES.ISO8859-1',
    'ca_es@valencia':                       'ca_ES.UTF-8@valencia',
    'ca_fr':                                'ca_FR.ISO8859-1',
    'ca_it':                                'ca_IT.ISO8859-1',
    'catalan':                              'ca_ES.ISO8859-1',
    'ce_ru':                                'ce_RU.UTF-8',
    'cextend':                              'en_US.ISO8859-1',
    'chinese-s':                            'zh_CN.eucCN',
    'chinese-t':                            'zh_TW.eucTW',
    'chr_us':                               'chr_US.UTF-8',
    'ckb_iq':                               'ckb_IQ.UTF-8',
    'cmn_tw':                               'cmn_TW.UTF-8',
    'crh_ua':                               'crh_UA.UTF-8',
    'croatian':                             'hr_HR.ISO8859-2',
    'cs':                                   'cs_CZ.ISO8859-2',
    'cs_cs':                                'cs_CZ.ISO8859-2',
    'cs_cz':                                'cs_CZ.ISO8859-2',
    'csb_pl':                               'csb_PL.UTF-8',
    'cv_ru':                                'cv_RU.UTF-8',
    'cy':                                   'cy_GB.ISO8859-1',
    'cy_gb':                                'cy_GB.ISO8859-1',
    'cz':                                   'cs_CZ.ISO8859-2',
    'cz_cz':                                'cs_CZ.ISO8859-2',
    'czech':                                'cs_CZ.ISO8859-2',
    'da':                                   'da_DK.ISO8859-1',
    'da_dk':                                'da_DK.ISO8859-1',
    'danish':                               'da_DK.ISO8859-1',
    'dansk':                                'da_DK.ISO8859-1',
    'de':                                   'de_DE.ISO8859-1',
    'de_at':                                'de_AT.ISO8859-1',
    'de_be':                                'de_BE.ISO8859-1',
    'de_ch':                                'de_CH.ISO8859-1',
    'de_de':                                'de_DE.ISO8859-1',
    'de_it':                                'de_IT.ISO8859-1',
    'de_li.utf8':                           'de_LI.UTF-8',
    'de_lu':                                'de_LU.ISO8859-1',
    'deutsch':                              'de_DE.ISO8859-1',
    'doi_in':                               'doi_IN.UTF-8',
    'dutch':                                'nl_NL.ISO8859-1',
    'dutch.iso88591':                       'nl_BE.ISO8859-1',
    'dv_mv':                                'dv_MV.UTF-8',
    'dz_bt':                                'dz_BT.UTF-8',
    'ee':                                   'ee_EE.ISO8859-4',
    'ee_ee':                                'ee_EE.ISO8859-4',
    'eesti':                                'et_EE.ISO8859-1',
    'el':                                   'el_GR.ISO8859-7',
    'el_cy':                                'el_CY.ISO8859-7',
    'el_gr':                                'el_GR.ISO8859-7',
    'el_gr@euro':                           'el_GR.ISO8859-15',
    'en':                                   'en_US.ISO8859-1',
    'en_ag':                                'en_AG.UTF-8',
    'en_au':                                'en_AU.ISO8859-1',
    'en_be':                                'en_BE.ISO8859-1',
    'en_bw':                                'en_BW.ISO8859-1',
    'en_ca':                                'en_CA.ISO8859-1',
    'en_dk':                                'en_DK.ISO8859-1',
    'en_dl.utf8':                           'en_DL.UTF-8',
    'en_gb':                                'en_GB.ISO8859-1',
    'en_hk':                                'en_HK.ISO8859-1',
    'en_ie':                                'en_IE.ISO8859-1',
    'en_il':                                'en_IL.UTF-8',
    'en_in':                                'en_IN.ISO8859-1',
    'en_ng':                                'en_NG.UTF-8',
    'en_nz':                                'en_NZ.ISO8859-1',
    'en_ph':                                'en_PH.ISO8859-1',
    'en_sc.utf8':                           'en_SC.UTF-8',
    'en_sg':                                'en_SG.ISO8859-1',
    'en_uk':                                'en_GB.ISO8859-1',
    'en_us':                                'en_US.ISO8859-1',
    'en_us@euro@euro':                      'en_US.ISO8859-15',
    'en_za':                                'en_ZA.ISO8859-1',
    'en_zm':                                'en_ZM.UTF-8',
    'en_zw':                                'en_ZW.ISO8859-1',
    'en_zw.utf8':                           'en_ZS.UTF-8',
    'eng_gb':                               'en_GB.ISO8859-1',
    'english':                              'en_EN.ISO8859-1',
    'english.iso88591':                     'en_US.ISO8859-1',
    'english_uk':                           'en_GB.ISO8859-1',
    'english_united-states':                'en_US.ISO8859-1',
    'english_united-states.437':            'C',
    'english_us':                           'en_US.ISO8859-1',
    'eo':                                   'eo_XX.ISO8859-3',
    'eo.utf8':                              'eo.UTF-8',
    'eo_eo':                                'eo_EO.ISO8859-3',
    'eo_us.utf8':                           'eo_US.UTF-8',
    'eo_xx':                                'eo_XX.ISO8859-3',
    'es':                                   'es_ES.ISO8859-1',
    'es_ar':                                'es_AR.ISO8859-1',
    'es_bo':                                'es_BO.ISO8859-1',
    'es_cl':                                'es_CL.ISO8859-1',
    'es_co':                                'es_CO.ISO8859-1',
    'es_cr':                                'es_CR.ISO8859-1',
    'es_cu':                                'es_CU.UTF-8',
    'es_do':                                'es_DO.ISO8859-1',
    'es_ec':                                'es_EC.ISO8859-1',
    'es_es':                                'es_ES.ISO8859-1',
    'es_gt':                                'es_GT.ISO8859-1',
    'es_hn':                                'es_HN.ISO8859-1',
    'es_mx':                                'es_MX.ISO8859-1',
    'es_ni':                                'es_NI.ISO8859-1',
    'es_pa':                                'es_PA.ISO8859-1',
    'es_pe':                                'es_PE.ISO8859-1',
    'es_pr':                                'es_PR.ISO8859-1',
    'es_py':                                'es_PY.ISO8859-1',
    'es_sv':                                'es_SV.ISO8859-1',
    'es_us':                                'es_US.ISO8859-1',
    'es_uy':                                'es_UY.ISO8859-1',
    'es_ve':                                'es_VE.ISO8859-1',
    'estonian':                             'et_EE.ISO8859-1',
    'et':                                   'et_EE.ISO8859-15',
    'et_ee':                                'et_EE.ISO8859-15',
    'eu':                                   'eu_ES.ISO8859-1',
    'eu_es':                                'eu_ES.ISO8859-1',
    'eu_fr':                                'eu_FR.ISO8859-1',
    'fa':                                   'fa_IR.UTF-8',
    'fa_ir':                                'fa_IR.UTF-8',
    'fa_ir.isiri3342':                      'fa_IR.ISIRI-3342',
    'ff_sn':                                'ff_SN.UTF-8',
    'fi':                                   'fi_FI.ISO8859-15',
    'fi_fi':                                'fi_FI.ISO8859-15',
    'fil_ph':                               'fil_PH.UTF-8',
    'finnish':                              'fi_FI.ISO8859-1',
    'fo':                                   'fo_FO.ISO8859-1',
    'fo_fo':                                'fo_FO.ISO8859-1',
    'fr':                                   'fr_FR.ISO8859-1',
    'fr_be':                                'fr_BE.ISO8859-1',
    'fr_ca':                                'fr_CA.ISO8859-1',
    'fr_ch':                                'fr_CH.ISO8859-1',
    'fr_fr':                                'fr_FR.ISO8859-1',
    'fr_lu':                                'fr_LU.ISO8859-1',
    'fran\xe7ais':                          'fr_FR.ISO8859-1',
    'fre_fr':                               'fr_FR.ISO8859-1',
    'french':                               'fr_FR.ISO8859-1',
    'french.iso88591':                      'fr_CH.ISO8859-1',
    'french_france':                        'fr_FR.ISO8859-1',
    'fur_it':                               'fur_IT.UTF-8',
    'fy_de':                                'fy_DE.UTF-8',
    'fy_nl':                                'fy_NL.UTF-8',
    'ga':                                   'ga_IE.ISO8859-1',
    'ga_ie':                                'ga_IE.ISO8859-1',
    'galego':                               'gl_ES.ISO8859-1',
    'galician':                             'gl_ES.ISO8859-1',
    'gd':                                   'gd_GB.ISO8859-1',
    'gd_gb':                                'gd_GB.ISO8859-1',
    'ger_de':                               'de_DE.ISO8859-1',
    'german':                               'de_DE.ISO8859-1',
    'german.iso88591':                      'de_CH.ISO8859-1',
    'german_germany':                       'de_DE.ISO8859-1',
    'gez_er':                               'gez_ER.UTF-8',
    'gez_et':                               'gez_ET.UTF-8',
    'gl':                                   'gl_ES.ISO8859-1',
    'gl_es':                                'gl_ES.ISO8859-1',
    'greek':                                'el_GR.ISO8859-7',
    'gu_in':                                'gu_IN.UTF-8',
    'gv':                                   'gv_GB.ISO8859-1',
    'gv_gb':                                'gv_GB.ISO8859-1',
    'ha_ng':                                'ha_NG.UTF-8',
    'hak_tw':                               'hak_TW.UTF-8',
    'he':                                   'he_IL.ISO8859-8',
    'he_il':                                'he_IL.ISO8859-8',
    'hebrew':                               'he_IL.ISO8859-8',
    'hi':                                   'hi_IN.ISCII-DEV',
    'hi_in':                                'hi_IN.ISCII-DEV',
    'hi_in.isciidev':                       'hi_IN.ISCII-DEV',
    'hif_fj':                               'hif_FJ.UTF-8',
    'hne':                                  'hne_IN.UTF-8',
    'hne_in':                               'hne_IN.UTF-8',
    'hr':                                   'hr_HR.ISO8859-2',
    'hr_hr':                                'hr_HR.ISO8859-2',
    'hrvatski':                             'hr_HR.ISO8859-2',
    'hsb_de':                               'hsb_DE.ISO8859-2',
    'ht_ht':                                'ht_HT.UTF-8',
    'hu':                                   'hu_HU.ISO8859-2',
    'hu_hu':                                'hu_HU.ISO8859-2',
    'hungarian':                            'hu_HU.ISO8859-2',
    'hy_am':                                'hy_AM.UTF-8',
    'hy_am.armscii8':                       'hy_AM.ARMSCII_8',
    'ia':                                   'ia.UTF-8',
    'ia_fr':                                'ia_FR.UTF-8',
    'icelandic':                            'is_IS.ISO8859-1',
    'id':                                   'id_ID.ISO8859-1',
    'id_id':                                'id_ID.ISO8859-1',
    'ig_ng':                                'ig_NG.UTF-8',
    'ik_ca':                                'ik_CA.UTF-8',
    'in':                                   'id_ID.ISO8859-1',
    'in_id':                                'id_ID.ISO8859-1',
    'is':                                   'is_IS.ISO8859-1',
    'is_is':                                'is_IS.ISO8859-1',
    'iso-8859-1':                           'en_US.ISO8859-1',
    'iso-8859-15':                          'en_US.ISO8859-15',
    'iso8859-1':                            'en_US.ISO8859-1',
    'iso8859-15':                           'en_US.ISO8859-15',
    'iso_8859_1':                           'en_US.ISO8859-1',
    'iso_8859_15':                          'en_US.ISO8859-15',
    'it':                                   'it_IT.ISO8859-1',
    'it_ch':                                'it_CH.ISO8859-1',
    'it_it':                                'it_IT.ISO8859-1',
    'italian':                              'it_IT.ISO8859-1',
    'iu':                                   'iu_CA.NUNACOM-8',
    'iu_ca':                                'iu_CA.NUNACOM-8',
    'iu_ca.nunacom8':                       'iu_CA.NUNACOM-8',
    'iw':                                   'he_IL.ISO8859-8',
    'iw_il':                                'he_IL.ISO8859-8',
    'iw_il.utf8':                           'iw_IL.UTF-8',
    'ja':                                   'ja_JP.eucJP',
    'ja_jp':                                'ja_JP.eucJP',
    'ja_jp.euc':                            'ja_JP.eucJP',
    'ja_jp.mscode':                         'ja_JP.SJIS',
    'ja_jp.pck':                            'ja_JP.SJIS',
    'japan':                                'ja_JP.eucJP',
    'japanese':                             'ja_JP.eucJP',
    'japanese-euc':                         'ja_JP.eucJP',
    'japanese.euc':                         'ja_JP.eucJP',
    'jp_jp':                                'ja_JP.eucJP',
    'ka':                                   'ka_GE.GEORGIAN-ACADEMY',
    'ka_ge':                                'ka_GE.GEORGIAN-ACADEMY',
    'ka_ge.georgianacademy':                'ka_GE.GEORGIAN-ACADEMY',
    'ka_ge.georgianps':                     'ka_GE.GEORGIAN-PS',
    'ka_ge.georgianrs':                     'ka_GE.GEORGIAN-ACADEMY',
    'kab_dz':                               'kab_DZ.UTF-8',
    'kk_kz':                                'kk_KZ.ptcp154',
    'kl':                                   'kl_GL.ISO8859-1',
    'kl_gl':                                'kl_GL.ISO8859-1',
    'km_kh':                                'km_KH.UTF-8',
    'kn':                                   'kn_IN.UTF-8',
    'kn_in':                                'kn_IN.UTF-8',
    'ko':                                   'ko_KR.eucKR',
    'ko_kr':                                'ko_KR.eucKR',
    'ko_kr.euc':                            'ko_KR.eucKR',
    'kok_in':                               'kok_IN.UTF-8',
    'korean':                               'ko_KR.eucKR',
    'korean.euc':                           'ko_KR.eucKR',
    'ks':                                   'ks_IN.UTF-8',
    'ks_in':                                'ks_IN.UTF-8',
    'ks_in@devanagari.utf8':                'ks_IN.UTF-8@devanagari',
    'ku_tr':                                'ku_TR.ISO8859-9',
    'kw':                                   'kw_GB.ISO8859-1',
    'kw_gb':                                'kw_GB.ISO8859-1',
    'ky':                                   'ky_KG.UTF-8',
    'ky_kg':                                'ky_KG.UTF-8',
    'lb_lu':                                'lb_LU.UTF-8',
    'lg_ug':                                'lg_UG.ISO8859-10',
    'li_be':                                'li_BE.UTF-8',
    'li_nl':                                'li_NL.UTF-8',
    'lij_it':                               'lij_IT.UTF-8',
    'lithuanian':                           'lt_LT.ISO8859-13',
    'ln_cd':                                'ln_CD.UTF-8',
    'lo':                                   'lo_LA.MULELAO-1',
    'lo_la':                                'lo_LA.MULELAO-1',
    'lo_la.cp1133':                         'lo_LA.IBM-CP1133',
    'lo_la.ibmcp1133':                      'lo_LA.IBM-CP1133',
    'lo_la.mulelao1':                       'lo_LA.MULELAO-1',
    'lt':                                   'lt_LT.ISO8859-13',
    'lt_lt':                                'lt_LT.ISO8859-13',
    'lv':                                   'lv_LV.ISO8859-13',
    'lv_lv':                                'lv_LV.ISO8859-13',
    'lzh_tw':                               'lzh_TW.UTF-8',
    'mag_in':                               'mag_IN.UTF-8',
    'mai':                                  'mai_IN.UTF-8',
    'mai_in':                               'mai_IN.UTF-8',
    'mai_np':                               'mai_NP.UTF-8',
    'mfe_mu':                               'mfe_MU.UTF-8',
    'mg_mg':                                'mg_MG.ISO8859-15',
    'mhr_ru':                               'mhr_RU.UTF-8',
    'mi':                                   'mi_NZ.ISO8859-1',
    'mi_nz':                                'mi_NZ.ISO8859-1',
    'miq_ni':                               'miq_NI.UTF-8',
    'mjw_in':                               'mjw_IN.UTF-8',
    'mk':                                   'mk_MK.ISO8859-5',
    'mk_mk':                                'mk_MK.ISO8859-5',
    'ml':                                   'ml_IN.UTF-8',
    'ml_in':                                'ml_IN.UTF-8',
    'mn_mn':                                'mn_MN.UTF-8',
    'mni_in':                               'mni_IN.UTF-8',
    'mr':                                   'mr_IN.UTF-8',
    'mr_in':                                'mr_IN.UTF-8',
    'ms':                                   'ms_MY.ISO8859-1',
    'ms_my':                                'ms_MY.ISO8859-1',
    'mt':                                   'mt_MT.ISO8859-3',
    'mt_mt':                                'mt_MT.ISO8859-3',
    'my_mm':                                'my_MM.UTF-8',
    'nan_tw':                               'nan_TW.UTF-8',
    'nb':                                   'nb_NO.ISO8859-1',
    'nb_no':                                'nb_NO.ISO8859-1',
    'nds_de':                               'nds_DE.UTF-8',
    'nds_nl':                               'nds_NL.UTF-8',
    'ne_np':                                'ne_NP.UTF-8',
    'nhn_mx':                               'nhn_MX.UTF-8',
    'niu_nu':                               'niu_NU.UTF-8',
    'niu_nz':                               'niu_NZ.UTF-8',
    'nl':                                   'nl_NL.ISO8859-1',
    'nl_aw':                                'nl_AW.UTF-8',
    'nl_be':                                'nl_BE.ISO8859-1',
    'nl_nl':                                'nl_NL.ISO8859-1',
    'nn':                                   'nn_NO.ISO8859-1',
    'nn_no':                                'nn_NO.ISO8859-1',
    'no':                                   'no_NO.ISO8859-1',
    'no@nynorsk':                           'ny_NO.ISO8859-1',
    'no_no':                                'no_NO.ISO8859-1',
    'no_no.iso88591@bokmal':                'no_NO.ISO8859-1',
    'no_no.iso88591@nynorsk':               'no_NO.ISO8859-1',
    'norwegian':                            'no_NO.ISO8859-1',
    'nr':                                   'nr_ZA.ISO8859-1',
    'nr_za':                                'nr_ZA.ISO8859-1',
    'nso':                                  'nso_ZA.ISO8859-15',
    'nso_za':                               'nso_ZA.ISO8859-15',
    'ny':                                   'ny_NO.ISO8859-1',
    'ny_no':                                'ny_NO.ISO8859-1',
    'nynorsk':                              'nn_NO.ISO8859-1',
    'oc':                                   'oc_FR.ISO8859-1',
    'oc_fr':                                'oc_FR.ISO8859-1',
    'om_et':                                'om_ET.UTF-8',
    'om_ke':                                'om_KE.ISO8859-1',
    'or':                                   'or_IN.UTF-8',
    'or_in':                                'or_IN.UTF-8',
    'os_ru':                                'os_RU.UTF-8',
    'pa':                                   'pa_IN.UTF-8',
    'pa_in':                                'pa_IN.UTF-8',
    'pa_pk':                                'pa_PK.UTF-8',
    'pap_an':                               'pap_AN.UTF-8',
    'pap_aw':                               'pap_AW.UTF-8',
    'pap_cw':                               'pap_CW.UTF-8',
    'pd':                                   'pd_US.ISO8859-1',
    'pd_de':                                'pd_DE.ISO8859-1',
    'pd_us':                                'pd_US.ISO8859-1',
    'ph':                                   'ph_PH.ISO8859-1',
    'ph_ph':                                'ph_PH.ISO8859-1',
    'pl':                                   'pl_PL.ISO8859-2',
    'pl_pl':                                'pl_PL.ISO8859-2',
    'polish':                               'pl_PL.ISO8859-2',
    'portuguese':                           'pt_PT.ISO8859-1',
    'portuguese_brazil':                    'pt_BR.ISO8859-1',
    'posix':                                'C',
    'posix-utf2':                           'C',
    'pp':                                   'pp_AN.ISO8859-1',
    'pp_an':                                'pp_AN.ISO8859-1',
    'ps_af':                                'ps_AF.UTF-8',
    'pt':                                   'pt_PT.ISO8859-1',
    'pt_br':                                'pt_BR.ISO8859-1',
    'pt_pt':                                'pt_PT.ISO8859-1',
    'quz_pe':                               'quz_PE.UTF-8',
    'raj_in':                               'raj_IN.UTF-8',
    'ro':                                   'ro_RO.ISO8859-2',
    'ro_ro':                                'ro_RO.ISO8859-2',
    'romanian':                             'ro_RO.ISO8859-2',
    'ru':                                   'ru_RU.UTF-8',
    'ru_ru':                                'ru_RU.UTF-8',
    'ru_ua':                                'ru_UA.KOI8-U',
    'rumanian':                             'ro_RO.ISO8859-2',
    'russian':                              'ru_RU.KOI8-R',
    'rw':                                   'rw_RW.ISO8859-1',
    'rw_rw':                                'rw_RW.ISO8859-1',
    'sa_in':                                'sa_IN.UTF-8',
    'sat_in':                               'sat_IN.UTF-8',
    'sc_it':                                'sc_IT.UTF-8',
    'sd':                                   'sd_IN.UTF-8',
    'sd_in':                                'sd_IN.UTF-8',
    'sd_in@devanagari.utf8':                'sd_IN.UTF-8@devanagari',
    'sd_pk':                                'sd_PK.UTF-8',
    'se_no':                                'se_NO.UTF-8',
    'serbocroatian':                        'sr_RS.UTF-8@latin',
    'sgs_lt':                               'sgs_LT.UTF-8',
    'sh':                                   'sr_RS.UTF-8@latin',
    'sh_ba.iso88592@bosnia':                'sr_CS.ISO8859-2',
    'sh_hr':                                'sh_HR.ISO8859-2',
    'sh_hr.iso88592':                       'hr_HR.ISO8859-2',
    'sh_sp':                                'sr_CS.ISO8859-2',
    'sh_yu':                                'sr_RS.UTF-8@latin',
    'shn_mm':                               'shn_MM.UTF-8',
    'shs_ca':                               'shs_CA.UTF-8',
    'si':                                   'si_LK.UTF-8',
    'si_lk':                                'si_LK.UTF-8',
    'sid_et':                               'sid_ET.UTF-8',
    'sinhala':                              'si_LK.UTF-8',
    'sk':                                   'sk_SK.ISO8859-2',
    'sk_sk':                                'sk_SK.ISO8859-2',
    'sl':                                   'sl_SI.ISO8859-2',
    'sl_cs':                                'sl_CS.ISO8859-2',
    'sl_si':                                'sl_SI.ISO8859-2',
    'slovak':                               'sk_SK.ISO8859-2',
    'slovene':                              'sl_SI.ISO8859-2',
    'slovenian':                            'sl_SI.ISO8859-2',
    'sm_ws':                                'sm_WS.UTF-8',
    'so_dj':                                'so_DJ.ISO8859-1',
    'so_et':                                'so_ET.UTF-8',
    'so_ke':                                'so_KE.ISO8859-1',
    'so_so':                                'so_SO.ISO8859-1',
    'sp':                                   'sr_CS.ISO8859-5',
    'sp_yu':                                'sr_CS.ISO8859-5',
    'spanish':                              'es_ES.ISO8859-1',
    'spanish_spain':                        'es_ES.ISO8859-1',
    'sq':                                   'sq_AL.ISO8859-2',
    'sq_al':                                'sq_AL.ISO8859-2',
    'sq_mk':                                'sq_MK.UTF-8',
    'sr':                                   'sr_RS.UTF-8',
    'sr@cyrillic':                          'sr_RS.UTF-8',
    'sr@latn':                              'sr_CS.UTF-8@latin',
    'sr_cs':                                'sr_CS.UTF-8',
    'sr_cs.iso88592@latn':                  'sr_CS.ISO8859-2',
    'sr_cs@latn':                           'sr_CS.UTF-8@latin',
    'sr_me':                                'sr_ME.UTF-8',
    'sr_rs':                                'sr_RS.UTF-8',
    'sr_rs@latn':                           'sr_RS.UTF-8@latin',
    'sr_sp':                                'sr_CS.ISO8859-2',
    'sr_yu':                                'sr_RS.UTF-8@latin',
    'sr_yu.cp1251@cyrillic':                'sr_CS.CP1251',
    'sr_yu.iso88592':                       'sr_CS.ISO8859-2',
    'sr_yu.iso88595':                       'sr_CS.ISO8859-5',
    'sr_yu.iso88595@cyrillic':              'sr_CS.ISO8859-5',
    'sr_yu.microsoftcp1251@cyrillic':       'sr_CS.CP1251',
    'sr_yu.utf8':                           'sr_RS.UTF-8',
    'sr_yu.utf8@cyrillic':                  'sr_RS.UTF-8',
    'sr_yu@cyrillic':                       'sr_RS.UTF-8',
    'ss':                                   'ss_ZA.ISO8859-1',
    'ss_za':                                'ss_ZA.ISO8859-1',
    'st':                                   'st_ZA.ISO8859-1',
    'st_za':                                'st_ZA.ISO8859-1',
    'sv':                                   'sv_SE.ISO8859-1',
    'sv_fi':                                'sv_FI.ISO8859-1',
    'sv_se':                                'sv_SE.ISO8859-1',
    'sw_ke':                                'sw_KE.UTF-8',
    'sw_tz':                                'sw_TZ.UTF-8',
    'swedish':                              'sv_SE.ISO8859-1',
    'szl_pl':                               'szl_PL.UTF-8',
    'ta':                                   'ta_IN.TSCII-0',
    'ta_in':                                'ta_IN.TSCII-0',
    'ta_in.tscii':                          'ta_IN.TSCII-0',
    'ta_in.tscii0':                         'ta_IN.TSCII-0',
    'ta_lk':                                'ta_LK.UTF-8',
    'tcy_in.utf8':                          'tcy_IN.UTF-8',
    'te':                                   'te_IN.UTF-8',
    'te_in':                                'te_IN.UTF-8',
    'tg':                                   'tg_TJ.KOI8-C',
    'tg_tj':                                'tg_TJ.KOI8-C',
    'th':                                   'th_TH.ISO8859-11',
    'th_th':                                'th_TH.ISO8859-11',
    'th_th.tactis':                         'th_TH.TIS620',
    'th_th.tis620':                         'th_TH.TIS620',
    'thai':                                 'th_TH.ISO8859-11',
    'the_np':                               'the_NP.UTF-8',
    'ti_er':                                'ti_ER.UTF-8',
    'ti_et':                                'ti_ET.UTF-8',
    'tig_er':                               'tig_ER.UTF-8',
    'tk_tm':                                'tk_TM.UTF-8',
    'tl':                                   'tl_PH.ISO8859-1',
    'tl_ph':                                'tl_PH.ISO8859-1',
    'tn':                                   'tn_ZA.ISO8859-15',
    'tn_za':                                'tn_ZA.ISO8859-15',
    'to_to':                                'to_TO.UTF-8',
    'tpi_pg':                               'tpi_PG.UTF-8',
    'tr':                                   'tr_TR.ISO8859-9',
    'tr_cy':                                'tr_CY.ISO8859-9',
    'tr_tr':                                'tr_TR.ISO8859-9',
    'ts':                                   'ts_ZA.ISO8859-1',
    'ts_za':                                'ts_ZA.ISO8859-1',
    'tt':                                   'tt_RU.TATAR-CYR',
    'tt_ru':                                'tt_RU.TATAR-CYR',
    'tt_ru.tatarcyr':                       'tt_RU.TATAR-CYR',
    'tt_ru@iqtelif':                        'tt_RU.UTF-8@iqtelif',
    'turkish':                              'tr_TR.ISO8859-9',
    'ug_cn':                                'ug_CN.UTF-8',
    'uk':                                   'uk_UA.KOI8-U',
    'uk_ua':                                'uk_UA.KOI8-U',
    'univ':                                 'en_US.UTF-8',
    'universal':                            'en_US.UTF-8',
    'universal.utf8@ucs4':                  'en_US.UTF-8',
    'unm_us':                               'unm_US.UTF-8',
    'ur':                                   'ur_PK.CP1256',
    'ur_in':                                'ur_IN.UTF-8',
    'ur_pk':                                'ur_PK.CP1256',
    'uz':                                   'uz_UZ.UTF-8',
    'uz_uz':                                'uz_UZ.UTF-8',
    'uz_uz@cyrillic':                       'uz_UZ.UTF-8',
    've':                                   've_ZA.UTF-8',
    've_za':                                've_ZA.UTF-8',
    'vi':                                   'vi_VN.TCVN',
    'vi_vn':                                'vi_VN.TCVN',
    'vi_vn.tcvn':                           'vi_VN.TCVN',
    'vi_vn.tcvn5712':                       'vi_VN.TCVN',
    'vi_vn.viscii':                         'vi_VN.VISCII',
    'vi_vn.viscii111':                      'vi_VN.VISCII',
    'wa':                                   'wa_BE.ISO8859-1',
    'wa_be':                                'wa_BE.ISO8859-1',
    'wae_ch':                               'wae_CH.UTF-8',
    'wal_et':                               'wal_ET.UTF-8',
    'wo_sn':                                'wo_SN.UTF-8',
    'xh':                                   'xh_ZA.ISO8859-1',
    'xh_za':                                'xh_ZA.ISO8859-1',
    'yi':                                   'yi_US.CP1255',
    'yi_us':                                'yi_US.CP1255',
    'yo_ng':                                'yo_NG.UTF-8',
    'yue_hk':                               'yue_HK.UTF-8',
    'yuw_pg':                               'yuw_PG.UTF-8',
    'zh':                                   'zh_CN.eucCN',
    'zh_cn':                                'zh_CN.gb2312',
    'zh_cn.big5':                           'zh_TW.big5',
    'zh_cn.euc':                            'zh_CN.eucCN',
    'zh_hk':                                'zh_HK.big5hkscs',
    'zh_hk.big5hk':                         'zh_HK.big5hkscs',
    'zh_sg':                                'zh_SG.GB2312',
    'zh_sg.gbk':                            'zh_SG.GBK',
    'zh_tw':                                'zh_TW.big5',
    'zh_tw.euc':                            'zh_TW.eucTW',
    'zh_tw.euctw':                          'zh_TW.eucTW',
    'zu':                                   'zu_ZA.ISO8859-1',
    'zu_za':                                'zu_ZA.ISO8859-1',
}

#
# This maps Windows language identifiers to locale strings.
#
# This list has been updated from
# http://msdn.microsoft.com/library/default.asp?url=/library/en-us/intl/nls_238z.asp
# to include every locale up to Windows Vista.
#
# NOTE: this mapping is incomplete.  If your language is missing, please
# submit a bug report to the Python bug tracker at http://bugs.python.org/
# Make sure you include the missing language identifier and the suggested
# locale code.
#

windows_locale = {
    0x0436: "af_ZA", # Afrikaans
    0x041c: "sq_AL", # Albanian
    0x0484: "gsw_FR",# Alsatian - France
    0x045e: "am_ET", # Amharic - Ethiopia
    0x0401: "ar_SA", # Arabic - Saudi Arabia
    0x0801: "ar_IQ", # Arabic - Iraq
    0x0c01: "ar_EG", # Arabic - Egypt
    0x1001: "ar_LY", # Arabic - Libya
    0x1401: "ar_DZ", # Arabic - Algeria
    0x1801: "ar_MA", # Arabic - Morocco
    0x1c01: "ar_TN", # Arabic - Tunisia
    0x2001: "ar_OM", # Arabic - Oman
    0x2401: "ar_YE", # Arabic - Yemen
    0x2801: "ar_SY", # Arabic - Syria
    0x2c01: "ar_JO", # Arabic - Jordan
    0x3001: "ar_LB", # Arabic - Lebanon
    0x3401: "ar_KW", # Arabic - Kuwait
    0x3801: "ar_AE", # Arabic - United Arab Emirates
    0x3c01: "ar_BH", # Arabic - Bahrain
    0x4001: "ar_QA", # Arabic - Qatar
    0x042b: "hy_AM", # Armenian
    0x044d: "as_IN", # Assamese - India
    0x042c: "az_AZ", # Azeri - Latin
    0x082c: "az_AZ", # Azeri - Cyrillic
    0x046d: "ba_RU", # Bashkir
    0x042d: "eu_ES", # Basque - Russia
    0x0423: "be_BY", # Belarusian
    0x0445: "bn_IN", # Begali
    0x201a: "bs_BA", # Bosnian - Cyrillic
    0x141a: "bs_BA", # Bosnian - Latin
    0x047e: "br_FR", # Breton - France
    0x0402: "bg_BG", # Bulgarian
#    0x0455: "my_MM", # Burmese - Not supported
    0x0403: "ca_ES", # Catalan
    0x0004: "zh_CHS",# Chinese - Simplified
    0x0404: "zh_TW", # Chinese - Taiwan
    0x0804: "zh_CN", # Chinese - PRC
    0x0c04: "zh_HK", # Chinese - Hong Kong S.A.R.
    0x1004: "zh_SG", # Chinese - Singapore
    0x1404: "zh_MO", # Chinese - Macao S.A.R.
    0x7c04: "zh_CHT",# Chinese - Traditional
    0x0483: "co_FR", # Corsican - France
    0x041a: "hr_HR", # Croatian
    0x101a: "hr_BA", # Croatian - Bosnia
    0x0405: "cs_CZ", # Czech
    0x0406: "da_DK", # Danish
    0x048c: "gbz_AF",# Dari - Afghanistan
    0x0465: "div_MV",# Divehi - Maldives
    0x0413: "nl_NL", # Dutch - The Netherlands
    0x0813: "nl_BE", # Dutch - Belgium
    0x0409: "en_US", # English - United States
    0x0809: "en_GB", # English - United Kingdom
    0x0c09: "en_AU", # English - Australia
    0x1009: "en_CA", # English - Canada
    0x1409: "en_NZ", # English - New Zealand
    0x1809: "en_IE", # English - Ireland
    0x1c09: "en_ZA", # English - South Africa
    0x2009: "en_JA", # English - Jamaica
    0x2409: "en_CB", # English - Caribbean
    0x2809: "en_BZ", # English - Belize
    0x2c09: "en_TT", # English - Trinidad
    0x3009: "en_ZW", # English - Zimbabwe
    0x3409: "en_PH", # English - Philippines
    0x4009: "en_IN", # English - India
    0x4409: "en_MY", # English - Malaysia
    0x4809: "en_IN", # English - Singapore
    0x0425: "et_EE", # Estonian
    0x0438: "fo_FO", # Faroese
    0x0464: "fil_PH",# Filipino
    0x040b: "fi_FI", # Finnish
    0x040c: "fr_FR", # French - France
    0x080c: "fr_BE", # French - Belgium
    0x0c0c: "fr_CA", # French - Canada
    0x100c: "fr_CH", # French - Switzerland
    0x140c: "fr_LU", # French - Luxembourg
    0x180c: "fr_MC", # French - Monaco
    0x0462: "fy_NL", # Frisian - Netherlands
    0x0456: "gl_ES", # Galician
    0x0437: "ka_GE", # Georgian
    0x0407: "de_DE", # German - Germany
    0x0807: "de_CH", # German - Switzerland
    0x0c07: "de_AT", # German - Austria
    0x1007: "de_LU", # German - Luxembourg
    0x1407: "de_LI", # German - Liechtenstein
    0x0408: "el_GR", # Greek
    0x046f: "kl_GL", # Greenlandic - Greenland
    0x0447: "gu_IN", # Gujarati
    0x0468: "ha_NG", # Hausa - Latin
    0x040d: "he_IL", # Hebrew
    0x0439: "hi_IN", # Hindi
    0x040e: "hu_HU", # Hungarian
    0x040f: "is_IS", # Icelandic
    0x0421: "id_ID", # Indonesian
    0x045d: "iu_CA", # Inuktitut - Syllabics
    0x085d: "iu_CA", # Inuktitut - Latin
    0x083c: "ga_IE", # Irish - Ireland
    0x0410: "it_IT", # Italian - Italy
    0x0810: "it_CH", # Italian - Switzerland
    0x0411: "ja_JP", # Japanese
    0x044b: "kn_IN", # Kannada - India
    0x043f: "kk_KZ", # Kazakh
    0x0453: "kh_KH", # Khmer - Cambodia
    0x0486: "qut_GT",# K'iche - Guatemala
    0x0487: "rw_RW", # Kinyarwanda - Rwanda
    0x0457: "kok_IN",# Konkani
    0x0412: "ko_KR", # Korean
    0x0440: "ky_KG", # Kyrgyz
    0x0454: "lo_LA", # Lao - Lao PDR
    0x0426: "lv_LV", # Latvian
    0x0427: "lt_LT", # Lithuanian
    0x082e: "dsb_DE",# Lower Sorbian - Germany
    0x046e: "lb_LU", # Luxembourgish
    0x042f: "mk_MK", # FYROM Macedonian
    0x043e: "ms_MY", # Malay - Malaysia
    0x083e: "ms_BN", # Malay - Brunei Darussalam
    0x044c: "ml_IN", # Malayalam - India
    0x043a: "mt_MT", # Maltese
    0x0481: "mi_NZ", # Maori
    0x047a: "arn_CL",# Mapudungun
    0x044e: "mr_IN", # Marathi
    0x047c: "moh_CA",# Mohawk - Canada
    0x0450: "mn_MN", # Mongolian - Cyrillic
    0x0850: "mn_CN", # Mongolian - PRC
    0x0461: "ne_NP", # Nepali
    0x0414: "nb_NO", # Norwegian - Bokmal
    0x0814: "nn_NO", # Norwegian - Nynorsk
    0x0482: "oc_FR", # Occitan - France
    0x0448: "or_IN", # Oriya - India
    0x0463: "ps_AF", # Pashto - Afghanistan
    0x0429: "fa_IR", # Persian
    0x0415: "pl_PL", # Polish
    0x0416: "pt_BR", # Portuguese - Brazil
    0x0816: "pt_PT", # Portuguese - Portugal
    0x0446: "pa_IN", # Punjabi
    0x046b: "quz_BO",# Quechua (Bolivia)
    0x086b: "quz_EC",# Quechua (Ecuador)
    0x0c6b: "quz_PE",# Quechua (Peru)
    0x0418: "ro_RO", # Romanian - Romania
    0x0417: "rm_CH", # Romansh
    0x0419: "ru_RU", # Russian
    0x243b: "smn_FI",# Sami Finland
    0x103b: "smj_NO",# Sami Norway
    0x143b: "smj_SE",# Sami Sweden
    0x043b: "se_NO", # Sami Northern Norway
    0x083b: "se_SE", # Sami Northern Sweden
    0x0c3b: "se_FI", # Sami Northern Finland
    0x203b: "sms_FI",# Sami Skolt
    0x183b: "sma_NO",# Sami Southern Norway
    0x1c3b: "sma_SE",# Sami Southern Sweden
    0x044f: "sa_IN", # Sanskrit
    0x0c1a: "sr_SP", # Serbian - Cyrillic
    0x1c1a: "sr_BA", # Serbian - Bosnia Cyrillic
    0x081a: "sr_SP", # Serbian - Latin
    0x181a: "sr_BA", # Serbian - Bosnia Latin
    0x045b: "si_LK", # Sinhala - Sri Lanka
    0x046c: "ns_ZA", # Northern Sotho
    0x0432: "tn_ZA", # Setswana - Southern Africa
    0x041b: "sk_SK", # Slovak
    0x0424: "sl_SI", # Slovenian
    0x040a: "es_ES", # Spanish - Spain
    0x080a: "es_MX", # Spanish - Mexico
    0x0c0a: "es_ES", # Spanish - Spain (Modern)
    0x100a: "es_GT", # Spanish - Guatemala
    0x140a: "es_CR", # Spanish - Costa Rica
    0x180a: "es_PA", # Spanish - Panama
    0x1c0a: "es_DO", # Spanish - Dominican Republic
    0x200a: "es_VE", # Spanish - Venezuela
    0x240a: "es_CO", # Spanish - Colombia
    0x280a: "es_PE", # Spanish - Peru
    0x2c0a: "es_AR", # Spanish - Argentina
    0x300a: "es_EC", # Spanish - Ecuador
    0x340a: "es_CL", # Spanish - Chile
    0x380a: "es_UR", # Spanish - Uruguay
    0x3c0a: "es_PY", # Spanish - Paraguay
    0x400a: "es_BO", # Spanish - Bolivia
    0x440a: "es_SV", # Spanish - El Salvador
    0x480a: "es_HN", # Spanish - Honduras
    0x4c0a: "es_NI", # Spanish - Nicaragua
    0x500a: "es_PR", # Spanish - Puerto Rico
    0x540a: "es_US", # Spanish - United States
#    0x0430: "", # Sutu - Not supported
    0x0441: "sw_KE", # Swahili
    0x041d: "sv_SE", # Swedish - Sweden
    0x081d: "sv_FI", # Swedish - Finland
    0x045a: "syr_SY",# Syriac
    0x0428: "tg_TJ", # Tajik - Cyrillic
    0x085f: "tmz_DZ",# Tamazight - Latin
    0x0449: "ta_IN", # Tamil
    0x0444: "tt_RU", # Tatar
    0x044a: "te_IN", # Telugu
    0x041e: "th_TH", # Thai
    0x0851: "bo_BT", # Tibetan - Bhutan
    0x0451: "bo_CN", # Tibetan - PRC
    0x041f: "tr_TR", # Turkish
    0x0442: "tk_TM", # Turkmen - Cyrillic
    0x0480: "ug_CN", # Uighur - Arabic
    0x0422: "uk_UA", # Ukrainian
    0x042e: "wen_DE",# Upper Sorbian - Germany
    0x0420: "ur_PK", # Urdu
    0x0820: "ur_IN", # Urdu - India
    0x0443: "uz_UZ", # Uzbek - Latin
    0x0843: "uz_UZ", # Uzbek - Cyrillic
    0x042a: "vi_VN", # Vietnamese
    0x0452: "cy_GB", # Welsh
    0x0488: "wo_SN", # Wolof - Senegal
    0x0434: "xh_ZA", # Xhosa - South Africa
    0x0485: "sah_RU",# Yakut - Cyrillic
    0x0478: "ii_CN", # Yi - PRC
    0x046a: "yo_NG", # Yoruba - Nigeria
    0x0435: "zu_ZA", # Zulu
}

def _print_locale():

    """ Test function.
    """
    categories = {}
    def _init_categories(categories=categories):
        for k,v in globals().items():
            if k[:3] == 'LC_':
                categories[k] = v
    _init_categories()
    del categories['LC_ALL']

    print('Locale defaults as determined by getdefaultlocale():')
    print('-'*72)
    lang, enc = getdefaultlocale()
    print('Language: ', lang or '(undefined)')
    print('Encoding: ', enc or '(undefined)')
    print()

    print('Locale settings on startup:')
    print('-'*72)
    for name,category in categories.items():
        print(name, '...')
        lang, enc = getlocale(category)
        print('   Language: ', lang or '(undefined)')
        print('   Encoding: ', enc or '(undefined)')
        print()

    print()
    print('Locale settings after calling resetlocale():')
    print('-'*72)
    resetlocale()
    for name,category in categories.items():
        print(name, '...')
        lang, enc = getlocale(category)
        print('   Language: ', lang or '(undefined)')
        print('   Encoding: ', enc or '(undefined)')
        print()

    try:
        setlocale(LC_ALL, "")
    except:
        print('NOTE:')
        print('setlocale(LC_ALL, "") does not support the default locale')
        print('given in the OS environment variables.')
    else:
        print()
        print('Locale settings after calling setlocale(LC_ALL, ""):')
        print('-'*72)
        for name,category in categories.items():
            print(name, '...')
            lang, enc = getlocale(category)
            print('   Language: ', lang or '(undefined)')
            print('   Encoding: ', enc or '(undefined)')
            print()

###

try:
    LC_MESSAGES
except NameError:
    pass
else:
    __all__.append("LC_MESSAGES")

if __name__=='__main__':
    print('Locale aliasing:')
    print()
    _print_locale()
    print()
    print('Number formatting:')
    print()
    _test()
<|endoftext|>"""
Read and write ZIP files.

XXX references to utf-8 need further investigation.
"""
import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import re
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib

try:
    import zlib # We may need its compression method
    crc32 = zlib.crc32
except ImportError:
    zlib = None
    crc32 = binascii.crc32

try:
    import bz2 # We may need its compression method
except ImportError:
    bz2 = None

try:
    import lzma # We may need its compression method
except ImportError:
    lzma = None

__all__ = ["BadZipFile", "BadZipfile", "error",
           "ZIP_STORED", "ZIP_DEFLATED", "ZIP_BZIP2", "ZIP_LZMA",
           "is_zipfile", "ZipInfo", "ZipFile", "PyZipFile", "LargeZipFile",
           "Path"]

class BadZipFile(Exception):
    pass


class LargeZipFile(Exception):
    """
    Raised when writing a zipfile, the zipfile requires ZIP64 extensions
    and those extensions are disabled.
    """

error = BadZipfile = BadZipFile      # Pre-3.2 compatibility names


ZIP64_LIMIT = (1 << 31) - 1
ZIP_FILECOUNT_LIMIT = (1 << 16) - 1
ZIP_MAX_COMMENT = (1 << 16) - 1

# constants for Zip file compression methods
ZIP_STORED = 0
ZIP_DEFLATED = 8
ZIP_BZIP2 = 12
ZIP_LZMA = 14
# Other ZIP compression methods not supported

DEFAULT_VERSION = 20
ZIP64_VERSION = 45
BZIP2_VERSION = 46
LZMA_VERSION = 63
# we recognize (but not necessarily support) all features up to that version
MAX_EXTRACT_VERSION = 63

# Below are some formats and associated data for reading/writing headers using
# the struct module.  The names and structures of headers/records are those used
# in the PKWARE description of the ZIP file format:
#     http://www.pkware.com/documents/casestudies/APPNOTE.TXT
# (URL valid as of January 2008)

# The "end of central directory" structure, magic number, size, and indices
# (section V.I in the format document)
structEndArchive = b"<4s4H2LH"
stringEndArchive = b"PK\005\006"
sizeEndCentDir = struct.calcsize(structEndArchive)

_ECD_SIGNATURE = 0
_ECD_DISK_NUMBER = 1
_ECD_DISK_START = 2
_ECD_ENTRIES_THIS_DISK = 3
_ECD_ENTRIES_TOTAL = 4
_ECD_SIZE = 5
_ECD_OFFSET = 6
_ECD_COMMENT_SIZE = 7
# These last two indices are not part of the structure as defined in the
# spec, but they are used internally by this module as a convenience
_ECD_COMMENT = 8
_ECD_LOCATION = 9

# The "central directory" structure, magic number, size, and indices
# of entries in the structure (section V.F in the format document)
structCentralDir = "<4s4B4HL2L5H2L"
stringCentralDir = b"PK\001\002"
sizeCentralDir = struct.calcsize(structCentralDir)

# indexes of entries in the central directory structure
_CD_SIGNATURE = 0
_CD_CREATE_VERSION = 1
_CD_CREATE_SYSTEM = 2
_CD_EXTRACT_VERSION = 3
_CD_EXTRACT_SYSTEM = 4
_CD_FLAG_BITS = 5
_CD_COMPRESS_TYPE = 6
_CD_TIME = 7
_CD_DATE = 8
_CD_CRC = 9
_CD_COMPRESSED_SIZE = 10
_CD_UNCOMPRESSED_SIZE = 11
_CD_FILENAME_LENGTH = 12
_CD_EXTRA_FIELD_LENGTH = 13
_CD_COMMENT_LENGTH = 14
_CD_DISK_NUMBER_START = 15
_CD_INTERNAL_FILE_ATTRIBUTES = 16
_CD_EXTERNAL_FILE_ATTRIBUTES = 17
_CD_LOCAL_HEADER_OFFSET = 18

# The "local file header" structure, magic number, size, and indices
# (section V.A in the format document)
structFileHeader = "<4s2B4HL2L2H"
stringFileHeader = b"PK\003\004"
sizeFileHeader = struct.calcsize(structFileHeader)

_FH_SIGNATURE = 0
_FH_EXTRACT_VERSION = 1
_FH_EXTRACT_SYSTEM = 2
_FH_GENERAL_PURPOSE_FLAG_BITS = 3
_FH_COMPRESSION_METHOD = 4
_FH_LAST_MOD_TIME = 5
_FH_LAST_MOD_DATE = 6
_FH_CRC = 7
_FH_COMPRESSED_SIZE = 8
_FH_UNCOMPRESSED_SIZE = 9
_FH_FILENAME_LENGTH = 10
_FH_EXTRA_FIELD_LENGTH = 11

# The "Zip64 end of central directory locator" structure, magic number, and size
structEndArchive64Locator = "<4sLQL"
stringEndArchive64Locator = b"PK\x06\x07"
sizeEndCentDir64Locator = struct.calcsize(structEndArchive64Locator)

# The "Zip64 end of central directory" record, magic number, size, and indices
# (section V.G in the format document)
structEndArchive64 = "<4sQ2H2L4Q"
stringEndArchive64 = b"PK\x06\x06"
sizeEndCentDir64 = struct.calcsize(structEndArchive64)

_CD64_SIGNATURE = 0
_CD64_DIRECTORY_RECSIZE = 1
_CD64_CREATE_VERSION = 2
_CD64_EXTRACT_VERSION = 3
_CD64_DISK_NUMBER = 4
_CD64_DISK_NUMBER_START = 5
_CD64_NUMBER_ENTRIES_THIS_DISK = 6
_CD64_NUMBER_ENTRIES_TOTAL = 7
_CD64_DIRECTORY_SIZE = 8
_CD64_OFFSET_START_CENTDIR = 9

_DD_SIGNATURE = 0x08074b50

_EXTRA_FIELD_STRUCT = struct.Struct('<HH')

def _strip_extra(extra, xids):
    # Remove Extra Fields with specified IDs.
    unpack = _EXTRA_FIELD_STRUCT.unpack
    modified = False
    buffer = []
    start = i = 0
    while i + 4 <= len(extra):
        xid, xlen = unpack(extra[i : i + 4])
        j = i + 4 + xlen
        if xid in xids:
            if i != start:
                buffer.append(extra[start : i])
            start = j
            modified = True
        i = j
    if not modified:
        return extra
    if start != len(extra):
        buffer.append(extra[start:])
    return b''.join(buffer)

def _check_zipfile(fp):
    try:
        if _EndRecData(fp):
            return True         # file has correct magic number
    except OSError:
        pass
    return False

def is_zipfile(filename):
    """Quickly see if a file is a ZIP file by checking the magic number.

    The filename argument may be a file or file-like object too.
    """
    result = False
    try:
        if hasattr(filename, "read"):
            result = _check_zipfile(fp=filename)
        else:
            with open(filename, "rb") as fp:
                result = _check_zipfile(fp)
    except (OSError, BadZipFile):
        pass
    return result

def _EndRecData64(fpin, offset, endrec):
    """
    Read the ZIP64 end-of-archive records and use that to update endrec
    """
    offset -= sizeEndCentDir64Locator
    if offset < 0:
        # The file is not large enough to contain a ZIP64
        # end-of-archive record, so just return the end record we were given.
        return endrec
    fpin.seek(offset)
    data = fpin.read(sizeEndCentDir64Locator)
    if len(data) != sizeEndCentDir64Locator:
        raise OSError("Unknown I/O error")
    sig, diskno, reloff, disks = struct.unpack(structEndArchive64Locator, data)
    if sig != stringEndArchive64Locator:
        return endrec

    if diskno != 0 or disks > 1:
        raise BadZipFile("zipfiles that span multiple disks are not supported")

    offset -= sizeEndCentDir64
    if reloff > offset:
        raise BadZipFile("Corrupt zip64 end of central directory locator")
    # First, check the assumption that there is no prepended data.
    fpin.seek(reloff)
    extrasz = offset - reloff
    data = fpin.read(sizeEndCentDir64)
    if len(data) != sizeEndCentDir64:
        raise OSError("Unknown I/O error")
    if not data.startswith(stringEndArchive64) and reloff != offset:
        # Since we already have seen the Zip64 EOCD Locator, it's
        # possible we got here because there is prepended data.
        # Assume no 'zip64 extensible data'
        fpin.seek(offset)
        extrasz = 0
        data = fpin.read(sizeEndCentDir64)
        if len(data) != sizeEndCentDir64:
            raise OSError("Unknown I/O error")
    if not data.startswith(stringEndArchive64):
        raise BadZipFile("Zip64 end of central directory record not found")

    sig, sz, create_version, read_version, disk_num, disk_dir, \
        dircount, dircount2, dirsize, diroffset = \
        struct.unpack(structEndArchive64, data)
    if (diroffset + dirsize != reloff or
        sz + 12 != sizeEndCentDir64 + extrasz):
        raise BadZipFile("Corrupt zip64 end of central directory record")

    # Update the original endrec using data from the ZIP64 record
    endrec[_ECD_SIGNATURE] = sig
    endrec[_ECD_DISK_NUMBER] = disk_num
    endrec[_ECD_DISK_START] = disk_dir
    endrec[_ECD_ENTRIES_THIS_DISK] = dircount
    endrec[_ECD_ENTRIES_TOTAL] = dircount2
    endrec[_ECD_SIZE] = dirsize
    endrec[_ECD_OFFSET] = diroffset
    endrec[_ECD_LOCATION] = offset - extrasz
    return endrec


def _EndRecData(fpin):
    """Return data from the "End of Central Directory" record, or None.

    The data is a list of the nine items in the ZIP "End of central dir"
    record followed by a tenth item, the file seek offset of this record."""

    # Determine file size
    fpin.seek(0, 2)
    filesize = fpin.tell()

    # Check to see if this is ZIP file with no archive comment (the
    # "end of central directory" structure should be the last item in the
    # file if this is the case).
    try:
        fpin.seek(-sizeEndCentDir, 2)
    except OSError:
        return None
    data = fpin.read()
    if (len(data) == sizeEndCentDir and
        data[0:4] == stringEndArchive and
        data[-2:] == b"\000\000"):
        # the signature is correct and there's no comment, unpack structure
        endrec = struct.unpack(structEndArchive, data)
        endrec=list(endrec)

        # Append a blank comment and record start offset
        endrec.append(b"")
        endrec.append(filesize - sizeEndCentDir)

        # Try to read the "Zip64 end of central directory" structure
        return _EndRecData64(fpin, filesize - sizeEndCentDir, endrec)

    # Either this is not a ZIP file, or it is a ZIP file with an archive
    # comment.  Search the end of the file for the "end of central directory"
    # record signature. The comment is the last item in the ZIP file and may be
    # up to 64K long.  It is assumed that the "end of central directory" magic
    # number does not appear in the comment.
    maxCommentStart = max(filesize - (1 << 16) - sizeEndCentDir, 0)
    fpin.seek(maxCommentStart, 0)
    data = fpin.read()
    start = data.rfind(stringEndArchive)
    if start >= 0:
        # found the magic number; attempt to unpack and interpret
        recData = data[start:start+sizeEndCentDir]
        if len(recData) != sizeEndCentDir:
            # Zip file is corrupted.
            return None
        endrec = list(struct.unpack(structEndArchive, recData))
        commentSize = endrec[_ECD_COMMENT_SIZE] #as claimed by the zip file
        comment = data[start+sizeEndCentDir:start+sizeEndCentDir+commentSize]
        endrec.append(comment)
        endrec.append(maxCommentStart + start)

        # Try to read the "Zip64 end of central directory" structure
        return _EndRecData64(fpin, maxCommentStart + start, endrec)

    # Unable to find a valid end of central directory structure
    return None


class ZipInfo (object):
    """Class with attributes describing each file in the ZIP archive."""

    __slots__ = (
        'orig_filename',
        'filename',
        'date_time',
        'compress_type',
        '_compresslevel',
        'comment',
        'extra',
        'create_system',
        'create_version',
        'extract_version',
        'reserved',
        'flag_bits',
        'volume',
        'internal_attr',
        'external_attr',
        'header_offset',
        'CRC',
        'compress_size',
        'file_size',
        '_raw_time',
        '_end_offset',
    )

    def __init__(self, filename="NoName", date_time=(1980,1,1,0,0,0)):
        self.orig_filename = filename   # Original file name in archive

        # Terminate the file name at the first null byte.  Null bytes in file
        # names are used as tricks by viruses in archives.
        null_byte = filename.find(chr(0))
        if null_byte >= 0:
            filename = filename[0:null_byte]
        # This is used to ensure paths in generated ZIP files always use
        # forward slashes as the directory separator, as required by the
        # ZIP format specification.
        if os.sep != "/" and os.sep in filename:
            filename = filename.replace(os.sep, "/")

        self.filename = filename        # Normalized file name
        self.date_time = date_time      # year, month, day, hour, min, sec

        if date_time[0] < 1980:
            raise ValueError('ZIP does not support timestamps before 1980')

        # Standard values:
        self.compress_type = ZIP_STORED # Type of compression for the file
        self._compresslevel = None      # Level for the compressor
        self.comment = b""              # Comment for each file
        self.extra = b""                # ZIP extra data
        if sys.platform == 'win32':
            self.create_system = 0          # System which created ZIP archive
        else:
            # Assume everything else is unix-y
            self.create_system = 3          # System which created ZIP archive
        self.create_version = DEFAULT_VERSION  # Version which created ZIP archive
        self.extract_version = DEFAULT_VERSION # Version needed to extract archive
        self.reserved = 0               # Must be zero
        self.flag_bits = 0              # ZIP flag bits
        self.volume = 0                 # Volume number of file header
        self.internal_attr = 0          # Internal attributes
        self.external_attr = 0          # External file attributes
        self.compress_size = 0          # Size of the compressed file
        self.file_size = 0              # Size of the uncompressed file
        self._end_offset = None         # Start of the next local header or central directory
        # Other attributes are set by class ZipFile:
        # header_offset         Byte offset to the file header
        # CRC                   CRC-32 of the uncompressed file

    def __repr__(self):
        result = ['<%s filename=%r' % (self.__class__.__name__, self.filename)]
        if self.compress_type != ZIP_STORED:
            result.append(' compress_type=%s' %
                          compressor_names.get(self.compress_type,
                                               self.compress_type))
        hi = self.external_attr >> 16
        lo = self.external_attr & 0xFFFF
        if hi:
            result.append(' filemode=%r' % stat.filemode(hi))
        if lo:
            result.append(' external_attr=%#x' % lo)
        isdir = self.is_dir()
        if not isdir or self.file_size:
            result.append(' file_size=%r' % self.file_size)
        if ((not isdir or self.compress_size) and
            (self.compress_type != ZIP_STORED or
             self.file_size != self.compress_size)):
            result.append(' compress_size=%r' % self.compress_size)
        result.append('>')
        return ''.join(result)

    def FileHeader(self, zip64=None):
        """Return the per-file header as a bytes object."""
        dt = self.date_time
        dosdate = (dt[0] - 1980) << 9 | dt[1] << 5 | dt[2]
        dostime = dt[3] << 11 | dt[4] << 5 | (dt[5] // 2)
        if self.flag_bits & 0x08:
            # Set these to zero because we write them after the file data
            CRC = compress_size = file_size = 0
        else:
            CRC = self.CRC
            compress_size = self.compress_size
            file_size = self.file_size

        extra = self.extra

        min_version = 0
        if zip64 is None:
            zip64 = file_size > ZIP64_LIMIT or compress_size > ZIP64_LIMIT
        if zip64:
            fmt = '<HHQQ'
            extra = extra + struct.pack(fmt,
                                        1, struct.calcsize(fmt)-4, file_size, compress_size)
        if file_size > ZIP64_LIMIT or compress_size > ZIP64_LIMIT:
            if not zip64:
                raise LargeZipFile("Filesize would require ZIP64 extensions")
            # File is larger than what fits into a 4 byte integer,
            # fall back to the ZIP64 extension
            file_size = 0xffffffff
            compress_size = 0xffffffff
            min_version = ZIP64_VERSION

        if self.compress_type == ZIP_BZIP2:
            min_version = max(BZIP2_VERSION, min_version)
        elif self.compress_type == ZIP_LZMA:
            min_version = max(LZMA_VERSION, min_version)

        self.extract_version = max(min_version, self.extract_version)
        self.create_version = max(min_version, self.create_version)
        filename, flag_bits = self._encodeFilenameFlags()
        header = struct.pack(structFileHeader, stringFileHeader,
                             self.extract_version, self.reserved, flag_bits,
                             self.compress_type, dostime, dosdate, CRC,
                             compress_size, file_size,
                             len(filename), len(extra))
        return header + filename + extra

    def _encodeFilenameFlags(self):
        try:
            return self.filename.encode('ascii'), self.flag_bits
        except UnicodeEncodeError:
            return self.filename.encode('utf-8'), self.flag_bits | 0x800

    def _decodeExtra(self):
        # Try to decode the extra field.
        extra = self.extra
        unpack = struct.unpack
        while len(extra) >= 4:
            tp, ln = unpack('<HH', extra[:4])
            if ln+4 > len(extra):
                raise BadZipFile("Corrupt extra field %04x (size=%d)" % (tp, ln))
            if tp == 0x0001:
                data = extra[4:ln+4]
                # ZIP64 extension (large files and/or large archives)
                try:
                    if self.file_size in (0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF):
                        field = "File size"
                        self.file_size, = unpack('<Q', data[:8])
                        data = data[8:]
                    if self.compress_size == 0xFFFF_FFFF:
                        field = "Compress size"
                        self.compress_size, = unpack('<Q', data[:8])
                        data = data[8:]
                    if self.header_offset == 0xFFFF_FFFF:
                        field = "Header offset"
                        self.header_offset, = unpack('<Q', data[:8])
                except struct.error:
                    raise BadZipFile(f"Corrupt zip64 extra field. "
                                     f"{field} not found.") from None

            extra = extra[ln+4:]

    @classmethod
    def from_file(cls, filename, arcname=None, *, strict_timestamps=True):
        """Construct an appropriate ZipInfo for a file on the filesystem.

        filename should be the path to a file or directory on the filesystem.

        arcname is the name which it will have within the archive (by default,
        this will be the same as filename, but without a drive letter and with
        leading path separators removed).
        """
        if isinstance(filename, os.PathLike):
            filename = os.fspath(filename)
        st = os.stat(filename)
        isdir = stat.S_ISDIR(st.st_mode)
        mtime = time.localtime(st.st_mtime)
        date_time = mtime[0:6]
        if not strict_timestamps and date_time[0] < 1980:
            date_time = (1980, 1, 1, 0, 0, 0)
        elif not strict_timestamps and date_time[0] > 2107:
            date_time = (2107, 12, 31, 23, 59, 59)
        # Create ZipInfo instance to store file information
        if arcname is None:
            arcname = filename
        arcname = os.path.normpath(os.path.splitdrive(arcname)[1])
        while arcname[0] in (os.sep, os.altsep):
            arcname = arcname[1:]
        if isdir:
            arcname += '/'
        zinfo = cls(arcname, date_time)
        zinfo.external_attr = (st.st_mode & 0xFFFF) << 16  # Unix attributes
        if isdir:
            zinfo.file_size = 0
            zinfo.external_attr |= 0x10  # MS-DOS directory flag
        else:
            zinfo.file_size = st.st_size

        return zinfo

    def is_dir(self):
        """Return True if this archive member is a directory."""
        return self.filename[-1] == '/'


# ZIP encryption uses the CRC32 one-byte primitive for scrambling some
# internal keys. We noticed that a direct implementation is faster than
# relying on binascii.crc32().

_crctable = None
def _gen_crc(crc):
    for j in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xEDB88320
        else:
            crc >>= 1
    return crc

# ZIP supports a password-based form of encryption. Even though known
# plaintext attacks have been found against it, it is still useful
# to be able to get data out of such a file.
#
# Usage:
#     zd = _ZipDecrypter(mypwd)
#     plain_bytes = zd(cypher_bytes)

def _ZipDecrypter(pwd):
    key0 = 305419896
    key1 = 591751049
    key2 = 878082192

    global _crctable
    if _crctable is None:
        _crctable = list(map(_gen_crc, range(256)))
    crctable = _crctable

    def crc32(ch, crc):
        """Compute the CRC32 primitive on one byte."""
        return (crc >> 8) ^ crctable[(crc ^ ch) & 0xFF]

    def update_keys(c):
        nonlocal key0, key1, key2
        key0 = crc32(c, key0)
        key1 = (key1 + (key0 & 0xFF)) & 0xFFFFFFFF
        key1 = (key1 * 134775813 + 1) & 0xFFFFFFFF
        key2 = crc32(key1 >> 24, key2)

    for p in pwd:
        update_keys(p)

    def decrypter(data):
        """Decrypt a bytes object."""
        result = bytearray()
        append = result.append
        for c in data:
            k = key2 | 2
            c ^= ((k * (k^1)) >> 8) & 0xFF
            update_keys(c)
            append(c)
        return bytes(result)

    return decrypter


class LZMACompressor:

    def __init__(self):
        self._comp = None

    def _init(self):
        props = lzma._encode_filter_properties({'id': lzma.FILTER_LZMA1})
        self._comp = lzma.LZMACompressor(lzma.FORMAT_RAW, filters=[
            lzma._decode_filter_properties(lzma.FILTER_LZMA1, props)
        ])
        return struct.pack('<BBH', 9, 4, len(props)) + props

    def compress(self, data):
        if self._comp is None:
            return self._init() + self._comp.compress(data)
        return self._comp.compress(data)

    def flush(self):
        if self._comp is None:
            return self._init() + self._comp.flush()
        return self._comp.flush()


class LZMADecompressor:

    def __init__(self):
        self._decomp = None
        self._unconsumed = b''
        self.eof = False

    def decompress(self, data):
        if self._decomp is None:
            self._unconsumed += data
            if len(self._unconsumed) <= 4:
                return b''
            psize, = struct.unpack('<H', self._unconsumed[2:4])
            if len(self._unconsumed) <= 4 + psize:
                return b''

            self._decomp = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[
                lzma._decode_filter_properties(lzma.FILTER_LZMA1,
                                               self._unconsumed[4:4 + psize])
            ])
            data = self._unconsumed[4 + psize:]
            del self._unconsumed

        result = self._decomp.decompress(data)
        self.eof = self._decomp.eof
        return result


compressor_names = {
    0: 'store',
    1: 'shrink',
    2: 'reduce',
    3: 'reduce',
    4: 'reduce',
    5: 'reduce',
    6: 'implode',
    7: 'tokenize',
    8: 'deflate',
    9: 'deflate64',
    10: 'implode',
    12: 'bzip2',
    14: 'lzma',
    18: 'terse',
    19: 'lz77',
    97: 'wavpack',
    98: 'ppmd',
}

def _check_compression(compression):
    if compression == ZIP_STORED:
        pass
    elif compression == ZIP_DEFLATED:
        if not zlib:
            raise RuntimeError(
                "Compression requires the (missing) zlib module")
    elif compression == ZIP_BZIP2:
        if not bz2:
            raise RuntimeError(
                "Compression requires the (missing) bz2 module")
    elif compression == ZIP_LZMA:
        if not lzma:
            raise RuntimeError(
                "Compression requires the (missing) lzma module")
    else:
        raise NotImplementedError("That compression method is not supported")


def _get_compressor(compress_type, compresslevel=None):
    if compress_type == ZIP_DEFLATED:
        if compresslevel is not None:
            return zlib.compressobj(compresslevel, zlib.DEFLATED, -15)
        return zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    elif compress_type == ZIP_BZIP2:
        if compresslevel is not None:
            return bz2.BZ2Compressor(compresslevel)
        return bz2.BZ2Compressor()
    # compresslevel is ignored for ZIP_LZMA
    elif compress_type == ZIP_LZMA:
        return LZMACompressor()
    else:
        return None


def _get_decompressor(compress_type):
    _check_compression(compress_type)
    if compress_type == ZIP_STORED:
        return None
    elif compress_type == ZIP_DEFLATED:
        return zlib.decompressobj(-15)
    elif compress_type == ZIP_BZIP2:
        return bz2.BZ2Decompressor()
    elif compress_type == ZIP_LZMA:
        return LZMADecompressor()
    else:
