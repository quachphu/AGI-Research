# Indices for stat struct members in the tuple returned by os.stat()

ST_MODE  = 0
ST_INO   = 1
ST_DEV   = 2
ST_NLINK = 3
ST_UID   = 4
ST_GID   = 5
ST_SIZE  = 6
ST_ATIME = 7
ST_MTIME = 8
ST_CTIME = 9

# Extract bits from the mode

def S_IMODE(mode):
    """Return the portion of the file's mode that can be set by
    os.chmod().
    """
    return mode & 0o7777

def S_IFMT(mode):
    """Return the portion of the file's mode that describes the
    file type.
    """
    return mode & 0o170000

# Constants used as S_IFMT() for various file types
# (not all are implemented on all systems)

S_IFDIR  = 0o040000  # directory
S_IFCHR  = 0o020000  # character device
S_IFBLK  = 0o060000  # block device
S_IFREG  = 0o100000  # regular file
S_IFIFO  = 0o010000  # fifo (named pipe)
S_IFLNK  = 0o120000  # symbolic link
S_IFSOCK = 0o140000  # socket file
# Fallbacks for uncommon platform-specific constants
S_IFDOOR = 0
S_IFPORT = 0
S_IFWHT = 0

# Functions to test for each file type

def S_ISDIR(mode):
    """Return True if mode is from a directory."""
    return S_IFMT(mode) == S_IFDIR

def S_ISCHR(mode):
    """Return True if mode is from a character special device file."""
    return S_IFMT(mode) == S_IFCHR

def S_ISBLK(mode):
    """Return True if mode is from a block special device file."""
    return S_IFMT(mode) == S_IFBLK

def S_ISREG(mode):
    """Return True if mode is from a regular file."""
    return S_IFMT(mode) == S_IFREG

def S_ISFIFO(mode):
    """Return True if mode is from a FIFO (named pipe)."""
    return S_IFMT(mode) == S_IFIFO

def S_ISLNK(mode):
    """Return True if mode is from a symbolic link."""
    return S_IFMT(mode) == S_IFLNK

def S_ISSOCK(mode):
    """Return True if mode is from a socket."""
    return S_IFMT(mode) == S_IFSOCK

def S_ISDOOR(mode):
    """Return True if mode is from a door."""
    return False

def S_ISPORT(mode):
    """Return True if mode is from an event port."""
    return False

def S_ISWHT(mode):
    """Return True if mode is from a whiteout."""
    return False

# Names for permission bits

S_ISUID = 0o4000  # set UID bit
S_ISGID = 0o2000  # set GID bit
S_ENFMT = S_ISGID # file locking enforcement
S_ISVTX = 0o1000  # sticky bit
S_IREAD = 0o0400  # Unix V7 synonym for S_IRUSR
S_IWRITE = 0o0200 # Unix V7 synonym for S_IWUSR
S_IEXEC = 0o0100  # Unix V7 synonym for S_IXUSR
S_IRWXU = 0o0700  # mask for owner permissions
S_IRUSR = 0o0400  # read by owner
S_IWUSR = 0o0200  # write by owner
S_IXUSR = 0o0100  # execute by owner
S_IRWXG = 0o0070  # mask for group permissions
S_IRGRP = 0o0040  # read by group
S_IWGRP = 0o0020  # write by group
S_IXGRP = 0o0010  # execute by group
S_IRWXO = 0o0007  # mask for others (not in group) permissions
S_IROTH = 0o0004  # read by others
S_IWOTH = 0o0002  # write by others
S_IXOTH = 0o0001  # execute by others

# Names for file flags

UF_NODUMP    = 0x00000001  # do not dump file
UF_IMMUTABLE = 0x00000002  # file may not be changed
UF_APPEND    = 0x00000004  # file may only be appended to
UF_OPAQUE    = 0x00000008  # directory is opaque when viewed through a union stack
UF_NOUNLINK  = 0x00000010  # file may not be renamed or deleted
UF_COMPRESSED = 0x00000020 # OS X: file is hfs-compressed
UF_HIDDEN    = 0x00008000  # OS X: file should not be displayed
SF_ARCHIVED  = 0x00010000  # file may be archived
SF_IMMUTABLE = 0x00020000  # file may not be changed
SF_APPEND    = 0x00040000  # file may only be appended to
SF_NOUNLINK  = 0x00100000  # file may not be renamed or deleted
SF_SNAPSHOT  = 0x00200000  # file is a snapshot file


_filemode_table = (
    ((S_IFLNK,         "l"),
     (S_IFSOCK,        "s"),  # Must appear before IFREG and IFDIR as IFSOCK == IFREG | IFDIR
     (S_IFREG,         "-"),
     (S_IFBLK,         "b"),
     (S_IFDIR,         "d"),
     (S_IFCHR,         "c"),
     (S_IFIFO,         "p")),

    ((S_IRUSR,         "r"),),
    ((S_IWUSR,         "w"),),
    ((S_IXUSR|S_ISUID, "s"),
     (S_ISUID,         "S"),
     (S_IXUSR,         "x")),

    ((S_IRGRP,         "r"),),
    ((S_IWGRP,         "w"),),
    ((S_IXGRP|S_ISGID, "s"),
     (S_ISGID,         "S"),
     (S_IXGRP,         "x")),

    ((S_IROTH,         "r"),),
    ((S_IWOTH,         "w"),),
    ((S_IXOTH|S_ISVTX, "t"),
     (S_ISVTX,         "T"),
     (S_IXOTH,         "x"))
)

def filemode(mode):
    """Convert a file's mode to a string of the form '-rwxrwxrwx'."""
    perm = []
    for table in _filemode_table:
        for bit, char in table:
            if mode & bit == bit:
                perm.append(char)
                break
        else:
            perm.append("-")
    return "".join(perm)


# Windows FILE_ATTRIBUTE constants for interpreting os.stat()'s
# "st_file_attributes" member

FILE_ATTRIBUTE_ARCHIVE = 32
FILE_ATTRIBUTE_COMPRESSED = 2048
FILE_ATTRIBUTE_DEVICE = 64
FILE_ATTRIBUTE_DIRECTORY = 16
FILE_ATTRIBUTE_ENCRYPTED = 16384
FILE_ATTRIBUTE_HIDDEN = 2
FILE_ATTRIBUTE_INTEGRITY_STREAM = 32768
FILE_ATTRIBUTE_NORMAL = 128
FILE_ATTRIBUTE_NOT_CONTENT_INDEXED = 8192
FILE_ATTRIBUTE_NO_SCRUB_DATA = 131072
FILE_ATTRIBUTE_OFFLINE = 4096
FILE_ATTRIBUTE_READONLY = 1
FILE_ATTRIBUTE_REPARSE_POINT = 1024
FILE_ATTRIBUTE_SPARSE_FILE = 512
FILE_ATTRIBUTE_SYSTEM = 4
FILE_ATTRIBUTE_TEMPORARY = 256
FILE_ATTRIBUTE_VIRTUAL = 65536


# If available, use C implementation
try:
    from _stat import *
except ImportError:
    pass
<|endoftext|>from _weakrefset import WeakSet


def get_cache_token():
    """Returns the current ABC cache token.

    The token is an opaque object (supporting equality testing) identifying the
    current version of the ABC cache for virtual subclasses. The token changes
    with every call to ``register()`` on any ABC.
    """
    return ABCMeta._abc_invalidation_counter


class ABCMeta(type):
    """Metaclass for defining Abstract Base Classes (ABCs).

    Use this metaclass to create an ABC.  An ABC can be subclassed
    directly, and then acts as a mix-in class.  You can also register
    unrelated concrete classes (even built-in classes) and unrelated
    ABCs as 'virtual subclasses' -- these and their descendants will
    be considered subclasses of the registering ABC by the built-in
    issubclass() function, but the registering ABC won't show up in
    their MRO (Method Resolution Order) nor will method
    implementations defined by the registering ABC be callable (not
    even via super()).
    """

    # A global counter that is incremented each time a class is
    # registered as a virtual subclass of anything.  It forces the
    # negative cache to be cleared before its next use.
    # Note: this counter is private. Use `abc.get_cache_token()` for
    #       external code.
    _abc_invalidation_counter = 0

    def __new__(mcls, name, bases, namespace, /, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        # Compute set of abstract method names
        abstracts = {name
                     for name, value in namespace.items()
                     if getattr(value, "__isabstractmethod__", False)}
        for base in bases:
            for name in getattr(base, "__abstractmethods__", set()):
                value = getattr(cls, name, None)
                if getattr(value, "__isabstractmethod__", False):
                    abstracts.add(name)
        cls.__abstractmethods__ = frozenset(abstracts)
        # Set up inheritance registry
        cls._abc_registry = WeakSet()
        cls._abc_cache = WeakSet()
        cls._abc_negative_cache = WeakSet()
        cls._abc_negative_cache_version = ABCMeta._abc_invalidation_counter
        return cls

    def register(cls, subclass):
        """Register a virtual subclass of an ABC.

        Returns the subclass, to allow usage as a class decorator.
        """
        if not isinstance(subclass, type):
            raise TypeError("Can only register classes")
        if issubclass(subclass, cls):
            return subclass  # Already a subclass
        # Subtle: test for cycles *after* testing for "already a subclass";
        # this means we allow X.register(X) and interpret it as a no-op.
        if issubclass(cls, subclass):
            # This would create a cycle, which is bad for the algorithm below
            raise RuntimeError("Refusing to create an inheritance cycle")
        cls._abc_registry.add(subclass)
        ABCMeta._abc_invalidation_counter += 1  # Invalidate negative cache
        return subclass

    def _dump_registry(cls, file=None):
        """Debug helper to print the ABC registry."""
        print(f"Class: {cls.__module__}.{cls.__qualname__}", file=file)
        print(f"Inv. counter: {get_cache_token()}", file=file)
        for name in cls.__dict__:
            if name.startswith("_abc_"):
                value = getattr(cls, name)
                if isinstance(value, WeakSet):
                    value = set(value)
                print(f"{name}: {value!r}", file=file)

    def _abc_registry_clear(cls):
        """Clear the registry (for debugging or testing)."""
        cls._abc_registry.clear()

    def _abc_caches_clear(cls):
        """Clear the caches (for debugging or testing)."""
        cls._abc_cache.clear()
        cls._abc_negative_cache.clear()

    def __instancecheck__(cls, instance):
        """Override for isinstance(instance, cls)."""
        # Inline the cache checking
        subclass = instance.__class__
        if subclass in cls._abc_cache:
            return True
        subtype = type(instance)
        if subtype is subclass:
            if (cls._abc_negative_cache_version ==
                ABCMeta._abc_invalidation_counter and
                subclass in cls._abc_negative_cache):
                return False
            # Fall back to the subclass check.
            return cls.__subclasscheck__(subclass)
        return any(cls.__subclasscheck__(c) for c in (subclass, subtype))

    def __subclasscheck__(cls, subclass):
        """Override for issubclass(subclass, cls)."""
        if not isinstance(subclass, type):
            raise TypeError('issubclass() arg 1 must be a class')
        # Check cache
        if subclass in cls._abc_cache:
            return True
        # Check negative cache; may have to invalidate
        if cls._abc_negative_cache_version < ABCMeta._abc_invalidation_counter:
            # Invalidate the negative cache
            cls._abc_negative_cache = WeakSet()
            cls._abc_negative_cache_version = ABCMeta._abc_invalidation_counter
        elif subclass in cls._abc_negative_cache:
            return False
        # Check the subclass hook
        ok = cls.__subclasshook__(subclass)
        if ok is not NotImplemented:
            assert isinstance(ok, bool)
            if ok:
                cls._abc_cache.add(subclass)
            else:
                cls._abc_negative_cache.add(subclass)
            return ok
        # Check if it's a direct subclass
        if cls in getattr(subclass, '__mro__', ()):
            cls._abc_cache.add(subclass)
            return True
        # Check if it's a subclass of a registered class (recursive)
        for rcls in cls._abc_registry:
            if issubclass(subclass, rcls):
                cls._abc_cache.add(subclass)
                return True
        # Check if it's a subclass of a subclass (recursive)
        for scls in cls.__subclasses__():
            if issubclass(subclass, scls):
                cls._abc_cache.add(subclass)
                return True
        # No dice; update negative cache
        cls._abc_negative_cache.add(subclass)
        return False
<|endoftext|>"""Module/script to byte-compile all .py files to .pyc files.

When called as a script with arguments, this compiles the directories
given as arguments recursively; the -l option prevents it from
recursing into directories.

Without arguments, if compiles all modules on sys.path, without
recursing into subdirectories.  (Even though it should do so for
packages -- for now, you'll have to deal with packages separately.)

See module py_compile for details of the actual byte-compilation.
"""
import os
import sys
import importlib.util
import py_compile
import struct
import filecmp

from functools import partial
from pathlib import Path

__all__ = ["compile_dir","compile_file","compile_path"]

def _walk_dir(dir, maxlevels, quiet=0):
    if quiet < 2 and isinstance(dir, os.PathLike):
        dir = os.fspath(dir)
    if not quiet:
        print('Listing {!r}...'.format(dir))
    try:
        names = os.listdir(dir)
    except OSError:
        if quiet < 2:
            print("Can't list {!r}".format(dir))
        names = []
    names.sort()
    for name in names:
        if name == '__pycache__':
            continue
        fullname = os.path.join(dir, name)
        if not os.path.isdir(fullname):
            yield fullname
        elif (maxlevels > 0 and name != os.curdir and name != os.pardir and
              os.path.isdir(fullname) and not os.path.islink(fullname)):
            yield from _walk_dir(fullname, maxlevels=maxlevels - 1,
                                 quiet=quiet)

def compile_dir(dir, maxlevels=None, ddir=None, force=False,
                rx=None, quiet=0, legacy=False, optimize=-1, workers=1,
                invalidation_mode=None, *, stripdir=None,
                prependdir=None, limit_sl_dest=None, hardlink_dupes=False):
    """Byte-compile all modules in the given directory tree.

    Arguments (only dir is required):

    dir:       the directory to byte-compile
    maxlevels: maximum recursion level (default `sys.getrecursionlimit()`)
    ddir:      the directory that will be prepended to the path to the
               file as it is compiled into each byte-code file.
    force:     if True, force compilation, even if timestamps are up-to-date
    quiet:     full output with False or 0, errors only with 1,
               no output with 2
    legacy:    if True, produce legacy pyc paths instead of PEP 3147 paths
    optimize:  int or list of optimization levels or -1 for level of
               the interpreter. Multiple levels leads to multiple compiled
               files each with one optimization level.
    workers:   maximum number of parallel workers
    invalidation_mode: how the up-to-dateness of the pyc will be checked
    stripdir:  part of path to left-strip from source file path
    prependdir: path to prepend to beginning of original file path, applied
               after stripdir
    limit_sl_dest: ignore symlinks if they are pointing outside of
                   the defined path
    hardlink_dupes: hardlink duplicated pyc files
    """
    ProcessPoolExecutor = None
    if ddir is not None and (stripdir is not None or prependdir is not None):
        raise ValueError(("Destination dir (ddir) cannot be used "
                          "in combination with stripdir or prependdir"))
    if ddir is not None:
        stripdir = dir
        prependdir = ddir
        ddir = None
    if workers < 0:
        raise ValueError('workers must be greater or equal to 0')
    if workers != 1:
        # Check if this is a system where ProcessPoolExecutor can function.
        from concurrent.futures.process import _check_system_limits
        try:
            _check_system_limits()
        except NotImplementedError:
            workers = 1
        else:
            from concurrent.futures import ProcessPoolExecutor
    if maxlevels is None:
        maxlevels = sys.getrecursionlimit()
    files = _walk_dir(dir, quiet=quiet, maxlevels=maxlevels)
    success = True
    if workers != 1 and ProcessPoolExecutor is not None:
        # If workers == 0, let ProcessPoolExecutor choose
        workers = workers or None
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(partial(compile_file,
                                           ddir=ddir, force=force,
                                           rx=rx, quiet=quiet,
                                           legacy=legacy,
                                           optimize=optimize,
                                           invalidation_mode=invalidation_mode,
                                           stripdir=stripdir,
                                           prependdir=prependdir,
                                           limit_sl_dest=limit_sl_dest,
                                           hardlink_dupes=hardlink_dupes),
                                   files)
            success = min(results, default=True)
    else:
        for file in files:
            if not compile_file(file, ddir, force, rx, quiet,
                                legacy, optimize, invalidation_mode,
                                stripdir=stripdir, prependdir=prependdir,
                                limit_sl_dest=limit_sl_dest,
                                hardlink_dupes=hardlink_dupes):
                success = False
    return success

def compile_file(fullname, ddir=None, force=False, rx=None, quiet=0,
                 legacy=False, optimize=-1,
                 invalidation_mode=None, *, stripdir=None, prependdir=None,
                 limit_sl_dest=None, hardlink_dupes=False):
    """Byte-compile one file.

    Arguments (only fullname is required):

    fullname:  the file to byte-compile
    ddir:      if given, the directory name compiled in to the
               byte-code file.
    force:     if True, force compilation, even if timestamps are up-to-date
    quiet:     full output with False or 0, errors only with 1,
               no output with 2
    legacy:    if True, produce legacy pyc paths instead of PEP 3147 paths
    optimize:  int or list of optimization levels or -1 for level of
               the interpreter. Multiple levels leads to multiple compiled
               files each with one optimization level.
    invalidation_mode: how the up-to-dateness of the pyc will be checked
    stripdir:  part of path to left-strip from source file path
    prependdir: path to prepend to beginning of original file path, applied
               after stripdir
    limit_sl_dest: ignore symlinks if they are pointing outside of
                   the defined path.
    hardlink_dupes: hardlink duplicated pyc files
    """

    if ddir is not None and (stripdir is not None or prependdir is not None):
        raise ValueError(("Destination dir (ddir) cannot be used "
                          "in combination with stripdir or prependdir"))

    success = True
    fullname = os.fspath(fullname)
    stripdir = os.fspath(stripdir) if stripdir is not None else None
    name = os.path.basename(fullname)

    dfile = None

    if ddir is not None:
        dfile = os.path.join(ddir, name)

    if stripdir is not None:
        fullname_parts = fullname.split(os.path.sep)
        stripdir_parts = stripdir.split(os.path.sep)
        ddir_parts = list(fullname_parts)

        for spart, opart in zip(stripdir_parts, fullname_parts):
            if spart == opart:
                ddir_parts.remove(spart)

        dfile = os.path.join(*ddir_parts)

    if prependdir is not None:
        if dfile is None:
            dfile = os.path.join(prependdir, fullname)
        else:
            dfile = os.path.join(prependdir, dfile)

    if isinstance(optimize, int):
        optimize = [optimize]

    # Use set() to remove duplicates.
    # Use sorted() to create pyc files in a deterministic order.
    optimize = sorted(set(optimize))

    if hardlink_dupes and len(optimize) < 2:
        raise ValueError("Hardlinking of duplicated bytecode makes sense "
                          "only for more than one optimization level")

    if rx is not None:
        mo = rx.search(fullname)
        if mo:
            return success

    if limit_sl_dest is not None and os.path.islink(fullname):
        if Path(limit_sl_dest).resolve() not in Path(fullname).resolve().parents:
            return success

    opt_cfiles = {}

    if os.path.isfile(fullname):
        for opt_level in optimize:
            if legacy:
                opt_cfiles[opt_level] = fullname + 'c'
            else:
                if opt_level >= 0:
                    opt = opt_level if opt_level >= 1 else ''
                    cfile = (importlib.util.cache_from_source(
                             fullname, optimization=opt))
                    opt_cfiles[opt_level] = cfile
                else:
                    cfile = importlib.util.cache_from_source(fullname)
                    opt_cfiles[opt_level] = cfile

        head, tail = name[:-3], name[-3:]
        if tail == '.py':
            if not force:
                try:
                    mtime = int(os.stat(fullname).st_mtime)
                    expect = struct.pack('<4sLL', importlib.util.MAGIC_NUMBER,
                                         0, mtime & 0xFFFF_FFFF)
                    for cfile in opt_cfiles.values():
                        with open(cfile, 'rb') as chandle:
                            actual = chandle.read(12)
                        if expect != actual:
                            break
                    else:
                        return success
                except OSError:
                    pass
            if not quiet:
                print('Compiling {!r}...'.format(fullname))
            try:
                for index, opt_level in enumerate(optimize):
                    cfile = opt_cfiles[opt_level]
                    ok = py_compile.compile(fullname, cfile, dfile, True,
                                            optimize=opt_level,
                                            invalidation_mode=invalidation_mode)
                    if index > 0 and hardlink_dupes:
                        previous_cfile = opt_cfiles[optimize[index - 1]]
                        if filecmp.cmp(cfile, previous_cfile, shallow=False):
                            os.unlink(cfile)
                            os.link(previous_cfile, cfile)
            except py_compile.PyCompileError as err:
                success = False
                if quiet >= 2:
                    return success
                elif quiet:
                    print('*** Error compiling {!r}...'.format(fullname))
                else:
                    print('*** ', end='')
                # escape non-printable characters in msg
                encoding = sys.stdout.encoding or sys.getdefaultencoding()
                msg = err.msg.encode(encoding, errors='backslashreplace').decode(encoding)
                print(msg)
            except (SyntaxError, UnicodeError, OSError) as e:
                success = False
                if quiet >= 2:
                    return success
                elif quiet:
                    print('*** Error compiling {!r}...'.format(fullname))
                else:
                    print('*** ', end='')
                print(e.__class__.__name__ + ':', e)
            else:
                if ok == 0:
                    success = False
    return success

def compile_path(skip_curdir=1, maxlevels=0, force=False, quiet=0,
                 legacy=False, optimize=-1,
                 invalidation_mode=None):
    """Byte-compile all module on sys.path.

    Arguments (all optional):

    skip_curdir: if true, skip current directory (default True)
    maxlevels:   max recursion level (default 0)
    force: as for compile_dir() (default False)
    quiet: as for compile_dir() (default 0)
    legacy: as for compile_dir() (default False)
    optimize: as for compile_dir() (default -1)
    invalidation_mode: as for compiler_dir()
    """
    success = True
    for dir in sys.path:
        if (not dir or dir == os.curdir) and skip_curdir:
            if quiet < 2:
                print('Skipping current directory')
        else:
            success = success and compile_dir(
                dir,
                maxlevels,
                None,
                force,
                quiet=quiet,
                legacy=legacy,
                optimize=optimize,
                invalidation_mode=invalidation_mode,
            )
    return success


def main():
    """Script main program."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Utilities to support installing Python libraries.')
    parser.add_argument('-l', action='store_const', const=0,
                        default=None, dest='maxlevels',
                        help="don't recurse into subdirectories")
    parser.add_argument('-r', type=int, dest='recursion',
                        help=('control the maximum recursion level. '
                              'if `-l` and `-r` options are specified, '
                              'then `-r` takes precedence.'))
    parser.add_argument('-f', action='store_true', dest='force',
                        help='force rebuild even if timestamps are up to date')
    parser.add_argument('-q', action='count', dest='quiet', default=0,
                        help='output only error messages; -qq will suppress '
                             'the error messages as well.')
    parser.add_argument('-b', action='store_true', dest='legacy',
                        help='use legacy (pre-PEP3147) compiled file locations')
    parser.add_argument('-d', metavar='DESTDIR',  dest='ddir', default=None,
                        help=('directory to prepend to file paths for use in '
                              'compile-time tracebacks and in runtime '
                              'tracebacks in cases where the source file is '
                              'unavailable'))
    parser.add_argument('-s', metavar='STRIPDIR',  dest='stripdir',
                        default=None,
                        help=('part of path to left-strip from path '
                              'to source file - for example buildroot. '
                              '`-d` and `-s` options cannot be '
                              'specified together.'))
    parser.add_argument('-p', metavar='PREPENDDIR',  dest='prependdir',
                        default=None,
                        help=('path to add as prefix to path '
                              'to source file - for example / to make '
                              'it absolute when some part is removed '
                              'by `-s` option. '
                              '`-d` and `-p` options cannot be '
                              'specified together.'))
    parser.add_argument('-x', metavar='REGEXP', dest='rx', default=None,
                        help=('skip files matching the regular expression; '
                              'the regexp is searched for in the full path '
                              'of each file considered for compilation'))
    parser.add_argument('-i', metavar='FILE', dest='flist',
                        help=('add all the files and directories listed in '
                              'FILE to the list considered for compilation; '
                              'if "-", names are read from stdin'))
    parser.add_argument('compile_dest', metavar='FILE|DIR', nargs='*',
                        help=('zero or more file and directory names '
                              'to compile; if no arguments given, defaults '
                              'to the equivalent of -l sys.path'))
    parser.add_argument('-j', '--workers', default=1,
                        type=int, help='Run compileall concurrently')
    invalidation_modes = [mode.name.lower().replace('_', '-')
                          for mode in py_compile.PycInvalidationMode]
    parser.add_argument('--invalidation-mode',
                        choices=sorted(invalidation_modes),
                        help=('set .pyc invalidation mode; defaults to '
                              '"checked-hash" if the SOURCE_DATE_EPOCH '
                              'environment variable is set, and '
                              '"timestamp" otherwise.'))
    parser.add_argument('-o', action='append', type=int, dest='opt_levels',
                        help=('Optimization levels to run compilation with. '
                              'Default is -1 which uses the optimization level '
                              'of the Python interpreter itself (see -O).'))
    parser.add_argument('-e', metavar='DIR', dest='limit_sl_dest',
                        help='Ignore symlinks pointing outsite of the DIR')
    parser.add_argument('--hardlink-dupes', action='store_true',
                        dest='hardlink_dupes',
                        help='Hardlink duplicated pyc files')

    args = parser.parse_args()
    compile_dests = args.compile_dest

    if args.rx:
        import re
        args.rx = re.compile(args.rx)

    if args.limit_sl_dest == "":
        args.limit_sl_dest = None

    if args.recursion is not None:
        maxlevels = args.recursion
    else:
        maxlevels = args.maxlevels

    if args.opt_levels is None:
        args.opt_levels = [-1]

    if len(args.opt_levels) == 1 and args.hardlink_dupes:
        parser.error(("Hardlinking of duplicated bytecode makes sense "
                      "only for more than one optimization level."))

    if args.ddir is not None and (
        args.stripdir is not None or args.prependdir is not None
    ):
        parser.error("-d cannot be used in combination with -s or -p")

    # if flist is provided then load it
    if args.flist:
        try:
            with (sys.stdin if args.flist=='-' else
                    open(args.flist, encoding="utf-8")) as f:
                for line in f:
                    compile_dests.append(line.strip())
        except OSError:
            if args.quiet < 2:
                print("Error reading file list {}".format(args.flist))
            return False

    if args.invalidation_mode:
        ivl_mode = args.invalidation_mode.replace('-', '_').upper()
        invalidation_mode = py_compile.PycInvalidationMode[ivl_mode]
    else:
        invalidation_mode = None

    success = True
    try:
        if compile_dests:
            for dest in compile_dests:
                if os.path.isfile(dest):
                    if not compile_file(dest, args.ddir, args.force, args.rx,
                                        args.quiet, args.legacy,
                                        invalidation_mode=invalidation_mode,
                                        stripdir=args.stripdir,
                                        prependdir=args.prependdir,
                                        optimize=args.opt_levels,
                                        limit_sl_dest=args.limit_sl_dest,
                                        hardlink_dupes=args.hardlink_dupes):
                        success = False
                else:
                    if not compile_dir(dest, maxlevels, args.ddir,
                                       args.force, args.rx, args.quiet,
                                       args.legacy, workers=args.workers,
                                       invalidation_mode=invalidation_mode,
                                       stripdir=args.stripdir,
                                       prependdir=args.prependdir,
                                       optimize=args.opt_levels,
                                       limit_sl_dest=args.limit_sl_dest,
                                       hardlink_dupes=args.hardlink_dupes):
                        success = False
            return success
        else:
            return compile_path(legacy=args.legacy, force=args.force,
                                quiet=args.quiet,
                                invalidation_mode=invalidation_mode)
    except KeyboardInterrupt:
        if args.quiet < 2:
            print("\n[interrupted]")
        return False
    return True


if __name__ == '__main__':
    exit_status = int(not main())
    sys.exit(exit_status)
<|endoftext|>"""Helper class to quickly write a loop over all standard input files.

Typical use is:

    import fileinput
    for line in fileinput.input(encoding="utf-8"):
        process(line)

This iterates over the lines of all files listed in sys.argv[1:],
defaulting to sys.stdin if the list is empty.  If a filename is '-' it
is also replaced by sys.stdin and the optional arguments mode and
openhook are ignored.  To specify an alternative list of filenames,
pass it as the argument to input().  A single file name is also allowed.

Functions filename(), lineno() return the filename and cumulative line
number of the line that has just been read; filelineno() returns its
line number in the current file; isfirstline() returns true iff the
line just read is the first line of its file; isstdin() returns true
iff the line was read from sys.stdin.  Function nextfile() closes the
current file so that the next iteration will read the first line from
the next file (if any); lines not read from the file will not count
towards the cumulative line count; the filename is not changed until
after the first line of the next file has been read.  Function close()
closes the sequence.

Before any lines have been read, filename() returns None and both line
numbers are zero; nextfile() has no effect.  After all lines have been
read, filename() and the line number functions return the values
pertaining to the last line read; nextfile() has no effect.

All files are opened in text mode by default, you can override this by
setting the mode parameter to input() or FileInput.__init__().
If an I/O error occurs during opening or reading a file, the OSError
exception is raised.

If sys.stdin is used more than once, the second and further use will
return no lines, except perhaps for interactive use, or if it has been
explicitly reset (e.g. using sys.stdin.seek(0)).

Empty files are opened and immediately closed; the only time their
presence in the list of filenames is noticeable at all is when the
last file opened is empty.

It is possible that the last line of a file doesn't end in a newline
character; otherwise lines are returned including the trailing
newline.

Class FileInput is the implementation; its methods filename(),
lineno(), fileline(), isfirstline(), isstdin(), nextfile() and close()
correspond to the functions in the module.  In addition it has a
readline() method which returns the next input line, and a
__getitem__() method which implements the sequence behavior.  The
sequence must be accessed in strictly sequential order; sequence
access and readline() cannot be mixed.

Optional in-place filtering: if the keyword argument inplace=1 is
passed to input() or to the FileInput constructor, the file is moved
to a backup file and standard output is directed to the input file.
This makes it possible to write a filter that rewrites its input file
in place.  If the keyword argument backup=".<some extension>" is also
given, it specifies the extension for the backup file, and the backup
file remains around; by default, the extension is ".bak" and it is
deleted when the output file is closed.  In-place filtering is
disabled when standard input is read.  XXX The current implementation
does not work for MS-DOS 8+3 filesystems.
"""

import io
import sys, os
from types import GenericAlias

__all__ = ["input", "close", "nextfile", "filename", "lineno", "filelineno",
           "fileno", "isfirstline", "isstdin", "FileInput", "hook_compressed",
           "hook_encoded"]

_state = None

def input(files=None, inplace=False, backup="", *, mode="r", openhook=None,
          encoding=None, errors=None):
    """Return an instance of the FileInput class, which can be iterated.

    The parameters are passed to the constructor of the FileInput class.
    The returned instance, in addition to being an iterator,
    keeps global state for the functions of this module,.
    """
    global _state
    if _state and _state._file:
        raise RuntimeError("input() already active")
    _state = FileInput(files, inplace, backup, mode=mode, openhook=openhook,
                       encoding=encoding, errors=errors)
    return _state

def close():
    """Close the sequence."""
    global _state
    state = _state
    _state = None
    if state:
        state.close()

def nextfile():
    """
    Close the current file so that the next iteration will read the first
    line from the next file (if any); lines not read from the file will
    not count towards the cumulative line count. The filename is not
    changed until after the first line of the next file has been read.
    Before the first line has been read, this function has no effect;
    it cannot be used to skip the first file. After the last line of the
    last file has been read, this function has no effect.
    """
    if not _state:
        raise RuntimeError("no active input()")
    return _state.nextfile()

def filename():
    """
    Return the name of the file currently being read.
    Before the first line has been read, returns None.
    """
    if not _state:
        raise RuntimeError("no active input()")
    return _state.filename()

def lineno():
    """
    Return the cumulative line number of the line that has just been read.
    Before the first line has been read, returns 0. After the last line
    of the last file has been read, returns the line number of that line.
    """
    if not _state:
        raise RuntimeError("no active input()")
    return _state.lineno()

def filelineno():
    """
    Return the line number in the current file. Before the first line
    has been read, returns 0. After the last line of the last file has
    been read, returns the line number of that line within the file.
    """
    if not _state:
        raise RuntimeError("no active input()")
    return _state.filelineno()

def fileno():
    """
    Return the file number of the current file. When no file is currently
    opened, returns -1.
    """
    if not _state:
        raise RuntimeError("no active input()")
    return _state.fileno()

def isfirstline():
    """
    Returns true the line just read is the first line of its file,
    otherwise returns false.
    """
    if not _state:
        raise RuntimeError("no active input()")
    return _state.isfirstline()

def isstdin():
    """
    Returns true if the last line was read from sys.stdin,
    otherwise returns false.
    """
    if not _state:
        raise RuntimeError("no active input()")
    return _state.isstdin()

class FileInput:
    """FileInput([files[, inplace[, backup]]], *, mode=None, openhook=None)

    Class FileInput is the implementation of the module; its methods
    filename(), lineno(), fileline(), isfirstline(), isstdin(), fileno(),
    nextfile() and close() correspond to the functions of the same name
    in the module.
    In addition it has a readline() method which returns the next
    input line, and a __getitem__() method which implements the
    sequence behavior. The sequence must be accessed in strictly
    sequential order; random access and readline() cannot be mixed.
    """

    def __init__(self, files=None, inplace=False, backup="", *,
                 mode="r", openhook=None, encoding=None, errors=None):
        if isinstance(files, str):
            files = (files,)
        elif isinstance(files, os.PathLike):
            files = (os.fspath(files), )
        else:
            if files is None:
                files = sys.argv[1:]
            if not files:
                files = ('-',)
            else:
                files = tuple(files)
        self._files = files
        self._inplace = inplace
        self._backup = backup
        self._savestdout = None
        self._output = None
        self._filename = None
        self._startlineno = 0
        self._filelineno = 0
        self._file = None
        self._isstdin = False
        self._backupfilename = None
        self._encoding = encoding
        self._errors = errors

        # We can not use io.text_encoding() here because old openhook doesn't
        # take encoding parameter.
        if (sys.flags.warn_default_encoding and
                "b" not in mode and encoding is None and openhook is None):
            import warnings
            warnings.warn("'encoding' argument not specified.",
                          EncodingWarning, 2)

        # restrict mode argument to reading modes
        if mode not in ('r', 'rU', 'U', 'rb'):
            raise ValueError("FileInput opening mode must be one of "
                             "'r', 'rU', 'U' and 'rb'")
        if 'U' in mode:
            import warnings
            warnings.warn("'U' mode is deprecated",
                          DeprecationWarning, 2)
        self._mode = mode
        self._write_mode = mode.replace('r', 'w') if 'U' not in mode else 'w'
        if openhook:
            if inplace:
                raise ValueError("FileInput cannot use an opening hook in inplace mode")
            if not callable(openhook):
                raise ValueError("FileInput openhook must be callable")
        self._openhook = openhook

    def __del__(self):
        self.close()

    def close(self):
        try:
            self.nextfile()
        finally:
            self._files = ()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            line = self._readline()
            if line:
                self._filelineno += 1
                return line
            if not self._file:
                raise StopIteration
            self.nextfile()
            # repeat with next file

    def __getitem__(self, i):
        import warnings
        warnings.warn(
            "Support for indexing FileInput objects is deprecated. "
            "Use iterator protocol instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if i != self.lineno():
            raise RuntimeError("accessing lines out of order")
        try:
            return self.__next__()
        except StopIteration:
            raise IndexError("end of input reached")

    def nextfile(self):
        savestdout = self._savestdout
        self._savestdout = None
        if savestdout:
            sys.stdout = savestdout

        output = self._output
        self._output = None
        try:
            if output:
                output.close()
        finally:
            file = self._file
            self._file = None
            try:
                del self._readline  # restore FileInput._readline
            except AttributeError:
                pass
            try:
                if file and not self._isstdin:
                    file.close()
            finally:
                backupfilename = self._backupfilename
                self._backupfilename = None
                if backupfilename and not self._backup:
                    try: os.unlink(backupfilename)
                    except OSError: pass

                self._isstdin = False

    def readline(self):
        while True:
            line = self._readline()
            if line:
                self._filelineno += 1
                return line
            if not self._file:
                return line
            self.nextfile()
            # repeat with next file

    def _readline(self):
        if not self._files:
            if 'b' in self._mode:
                return b''
            else:
                return ''
        self._filename = self._files[0]
        self._files = self._files[1:]
        self._startlineno = self.lineno()
        self._filelineno = 0
        self._file = None
        self._isstdin = False
        self._backupfilename = 0

        # EncodingWarning is emitted in __init__() already
        if "b" not in self._mode:
            encoding = self._encoding or "locale"
        else:
            encoding = None

        if self._filename == '-':
            self._filename = '<stdin>'
            if 'b' in self._mode:
                self._file = getattr(sys.stdin, 'buffer', sys.stdin)
            else:
                self._file = sys.stdin
            self._isstdin = True
        else:
            if self._inplace:
                self._backupfilename = (
                    os.fspath(self._filename) + (self._backup or ".bak"))
                try:
                    os.unlink(self._backupfilename)
                except OSError:
                    pass
                # The next few lines may raise OSError
                os.rename(self._filename, self._backupfilename)
                self._file = open(self._backupfilename, self._mode,
                                  encoding=encoding, errors=self._errors)
                try:
                    perm = os.fstat(self._file.fileno()).st_mode
                except OSError:
                    self._output = open(self._filename, self._write_mode,
                                        encoding=encoding, errors=self._errors)
                else:
                    mode = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
                    if hasattr(os, 'O_BINARY'):
                        mode |= os.O_BINARY

                    fd = os.open(self._filename, mode, perm)
                    self._output = os.fdopen(fd, self._write_mode,
                                             encoding=encoding, errors=self._errors)
                    try:
                        os.chmod(self._filename, perm)
                    except OSError:
                        pass
                self._savestdout = sys.stdout
                sys.stdout = self._output
            else:
                # This may raise OSError
                if self._openhook:
                    # Custom hooks made previous to Python 3.10 didn't have
                    # encoding argument
                    if self._encoding is None:
                        self._file = self._openhook(self._filename, self._mode)
                    else:
                        self._file = self._openhook(
                            self._filename, self._mode, encoding=self._encoding, errors=self._errors)
                else:
                    self._file = open(self._filename, self._mode, encoding=encoding, errors=self._errors)
        self._readline = self._file.readline  # hide FileInput._readline
        return self._readline()

    def filename(self):
        return self._filename

    def lineno(self):
        return self._startlineno + self._filelineno

    def filelineno(self):
        return self._filelineno

    def fileno(self):
        if self._file:
            try:
                return self._file.fileno()
            except ValueError:
                return -1
        else:
            return -1

    def isfirstline(self):
        return self._filelineno == 1

    def isstdin(self):
        return self._isstdin

    __class_getitem__ = classmethod(GenericAlias)


def hook_compressed(filename, mode, *, encoding=None, errors=None):
    if encoding is None and "b" not in mode:  # EncodingWarning is emitted in FileInput() already.
        encoding = "locale"
    ext = os.path.splitext(filename)[1]
    if ext == '.gz':
        import gzip
        stream = gzip.open(filename, mode)
    elif ext == '.bz2':
        import bz2
        stream = bz2.BZ2File(filename, mode)
    else:
        return open(filename, mode, encoding=encoding, errors=errors)

    # gzip and bz2 are binary mode by default.
    if "b" not in mode:
        stream = io.TextIOWrapper(stream, encoding=encoding, errors=errors)
    return stream


def hook_encoded(encoding, errors=None):
    def openhook(filename, mode):
        return open(filename, mode, encoding=encoding, errors=errors)
    return openhook


def _test():
    import getopt
    inplace = False
    backup = False
    opts, args = getopt.getopt(sys.argv[1:], "ib:")
    for o, a in opts:
        if o == '-i': inplace = True
        if o == '-b': backup = a
    for line in input(args, inplace=inplace, backup=backup):
        if line[-1:] == '\n': line = line[:-1]
        if line[-1:] == '\r': line = line[:-1]
        print("%d: %s[%d]%s %s" % (lineno(), filename(), filelineno(),
                                   isfirstline() and "*" or "", line))
    print("%d: %s[%d]" % (lineno(), filename(), filelineno()))

if __name__ == '__main__':
    _test()
<|endoftext|>r"""UUID objects (universally unique identifiers) according to RFC 4122.

This module provides immutable UUID objects (class UUID) and the functions
uuid1(), uuid3(), uuid4(), uuid5() for generating version 1, 3, 4, and 5
UUIDs as specified in RFC 4122.

If all you want is a unique ID, you should probably call uuid1() or uuid4().
Note that uuid1() may compromise privacy since it creates a UUID containing
the computer's network address.  uuid4() creates a random UUID.

Typical usage:

    >>> import uuid

    # make a UUID based on the host ID and current time
    >>> uuid.uuid1()    # doctest: +SKIP
    UUID('a8098c1a-f86e-11da-bd1a-00112444be1e')

    # make a UUID using an MD5 hash of a namespace UUID and a name
    >>> uuid.uuid3(uuid.NAMESPACE_DNS, 'python.org')
    UUID('6fa459ea-ee8a-3ca4-894e-db77e160355e')

    # make a random UUID
    >>> uuid.uuid4()    # doctest: +SKIP
    UUID('16fd2706-8baf-433b-82eb-8c7fada847da')

    # make a UUID using a SHA-1 hash of a namespace UUID and a name
    >>> uuid.uuid5(uuid.NAMESPACE_DNS, 'python.org')
    UUID('886313e1-3b8a-5372-9b90-0c9aee199e5d')

    # make a UUID from a string of hex digits (braces and hyphens ignored)
    >>> x = uuid.UUID('{00010203-0405-0607-0809-0a0b0c0d0e0f}')

    # convert a UUID to a string of hex digits in standard form
    >>> str(x)
    '00010203-0405-0607-0809-0a0b0c0d0e0f'

    # get the raw 16 bytes of the UUID
    >>> x.bytes
    b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'

    # make a UUID from a 16-byte string
    >>> uuid.UUID(bytes=x.bytes)
    UUID('00010203-0405-0607-0809-0a0b0c0d0e0f')
"""

import os
import sys

from enum import Enum


__author__ = 'Ka-Ping Yee <ping@zesty.ca>'

# The recognized platforms - known behaviors
if sys.platform in ('win32', 'darwin'):
    _AIX = _LINUX = False
else:
    import platform
    _platform_system = platform.system()
    _AIX     = _platform_system == 'AIX'
    _LINUX   = _platform_system == 'Linux'

_MAC_DELIM = b':'
_MAC_OMITS_LEADING_ZEROES = False
if _AIX:
    _MAC_DELIM = b'.'
    _MAC_OMITS_LEADING_ZEROES = True

RESERVED_NCS, RFC_4122, RESERVED_MICROSOFT, RESERVED_FUTURE = [
    'reserved for NCS compatibility', 'specified in RFC 4122',
    'reserved for Microsoft compatibility', 'reserved for future definition']

int_ = int      # The built-in int type
bytes_ = bytes  # The built-in bytes type


class SafeUUID(Enum):
    safe = 0
    unsafe = -1
    unknown = None


class UUID:
    """Instances of the UUID class represent UUIDs as specified in RFC 4122.
    UUID objects are immutable, hashable, and usable as dictionary keys.
    Converting a UUID to a string with str() yields something in the form
    '12345678-1234-1234-1234-123456789abc'.  The UUID constructor accepts
    five possible forms: a similar string of hexadecimal digits, or a tuple
    of six integer fields (with 32-bit, 16-bit, 16-bit, 8-bit, 8-bit, and
    48-bit values respectively) as an argument named 'fields', or a string
    of 16 bytes (with all the integer fields in big-endian order) as an
    argument named 'bytes', or a string of 16 bytes (with the first three
    fields in little-endian order) as an argument named 'bytes_le', or a
    single 128-bit integer as an argument named 'int'.

    UUIDs have these read-only attributes:

        bytes       the UUID as a 16-byte string (containing the six
                    integer fields in big-endian byte order)

        bytes_le    the UUID as a 16-byte string (with time_low, time_mid,
                    and time_hi_version in little-endian byte order)

        fields      a tuple of the six integer fields of the UUID,
                    which are also available as six individual attributes
                    and two derived attributes:

            time_low                the first 32 bits of the UUID
            time_mid                the next 16 bits of the UUID
            time_hi_version         the next 16 bits of the UUID
            clock_seq_hi_variant    the next 8 bits of the UUID
            clock_seq_low           the next 8 bits of the UUID
            node                    the last 48 bits of the UUID

            time                    the 60-bit timestamp
            clock_seq               the 14-bit sequence number

        hex         the UUID as a 32-character hexadecimal string

        int         the UUID as a 128-bit integer

        urn         the UUID as a URN as specified in RFC 4122

        variant     the UUID variant (one of the constants RESERVED_NCS,
                    RFC_4122, RESERVED_MICROSOFT, or RESERVED_FUTURE)

        version     the UUID version number (1 through 5, meaningful only
                    when the variant is RFC_4122)

        is_safe     An enum indicating whether the UUID has been generated in
                    a way that is safe for multiprocessing applications, via
                    uuid_generate_time_safe(3).
    """

    __slots__ = ('int', 'is_safe', '__weakref__')

    def __init__(self, hex=None, bytes=None, bytes_le=None, fields=None,
                       int=None, version=None,
                       *, is_safe=SafeUUID.unknown):
        r"""Create a UUID from either a string of 32 hexadecimal digits,
        a string of 16 bytes as the 'bytes' argument, a string of 16 bytes
        in little-endian order as the 'bytes_le' argument, a tuple of six
        integers (32-bit time_low, 16-bit time_mid, 16-bit time_hi_version,
        8-bit clock_seq_hi_variant, 8-bit clock_seq_low, 48-bit node) as
        the 'fields' argument, or a single 128-bit integer as the 'int'
        argument.  When a string of hex digits is given, curly braces,
        hyphens, and a URN prefix are all optional.  For example, these
        expressions all yield the same UUID:

        UUID('{12345678-1234-5678-1234-567812345678}')
        UUID('12345678123456781234567812345678')
        UUID('urn:uuid:12345678-1234-5678-1234-567812345678')
        UUID(bytes='\x12\x34\x56\x78'*4)
        UUID(bytes_le='\x78\x56\x34\x12\x34\x12\x78\x56' +
                      '\x12\x34\x56\x78\x12\x34\x56\x78')
        UUID(fields=(0x12345678, 0x1234, 0x5678, 0x12, 0x34, 0x567812345678))
        UUID(int=0x12345678123456781234567812345678)

        Exactly one of 'hex', 'bytes', 'bytes_le', 'fields', or 'int' must
        be given.  The 'version' argument is optional; if given, the resulting
        UUID will have its variant and version set according to RFC 4122,
        overriding the given 'hex', 'bytes', 'bytes_le', 'fields', or 'int'.

        is_safe is an enum exposed as an attribute on the instance.  It
        indicates whether the UUID has been generated in a way that is safe
        for multiprocessing applications, via uuid_generate_time_safe(3).
        """

        if [hex, bytes, bytes_le, fields, int].count(None) != 4:
            raise TypeError('one of the hex, bytes, bytes_le, fields, '
                            'or int arguments must be given')
        if hex is not None:
            hex = hex.replace('urn:', '').replace('uuid:', '')
            hex = hex.strip('{}').replace('-', '')
            if len(hex) != 32:
                raise ValueError('badly formed hexadecimal UUID string')
            int = int_(hex, 16)
        if bytes_le is not None:
            if len(bytes_le) != 16:
                raise ValueError('bytes_le is not a 16-char string')
            bytes = (bytes_le[4-1::-1] + bytes_le[6-1:4-1:-1] +
                     bytes_le[8-1:6-1:-1] + bytes_le[8:])
        if bytes is not None:
            if len(bytes) != 16:
                raise ValueError('bytes is not a 16-char string')
            assert isinstance(bytes, bytes_), repr(bytes)
            int = int_.from_bytes(bytes, byteorder='big')
        if fields is not None:
            if len(fields) != 6:
                raise ValueError('fields is not a 6-tuple')
            (time_low, time_mid, time_hi_version,
             clock_seq_hi_variant, clock_seq_low, node) = fields
            if not 0 <= time_low < 1<<32:
                raise ValueError('field 1 out of range (need a 32-bit value)')
            if not 0 <= time_mid < 1<<16:
                raise ValueError('field 2 out of range (need a 16-bit value)')
            if not 0 <= time_hi_version < 1<<16:
                raise ValueError('field 3 out of range (need a 16-bit value)')
            if not 0 <= clock_seq_hi_variant < 1<<8:
                raise ValueError('field 4 out of range (need an 8-bit value)')
            if not 0 <= clock_seq_low < 1<<8:
                raise ValueError('field 5 out of range (need an 8-bit value)')
            if not 0 <= node < 1<<48:
                raise ValueError('field 6 out of range (need a 48-bit value)')
            clock_seq = (clock_seq_hi_variant << 8) | clock_seq_low
            int = ((time_low << 96) | (time_mid << 80) |
                   (time_hi_version << 64) | (clock_seq << 48) | node)
        if int is not None:
            if not 0 <= int < 1<<128:
                raise ValueError('int is out of range (need a 128-bit value)')
        if version is not None:
            if not 1 <= version <= 5:
                raise ValueError('illegal version number')
            # Set the variant to RFC 4122.
            int &= ~(0xc000 << 48)
            int |= 0x8000 << 48
            # Set the version number.
            int &= ~(0xf000 << 64)
            int |= version << 76
        object.__setattr__(self, 'int', int)
        object.__setattr__(self, 'is_safe', is_safe)

    def __getstate__(self):
        d = {'int': self.int}
        if self.is_safe != SafeUUID.unknown:
            # is_safe is a SafeUUID instance.  Return just its value, so that
            # it can be un-pickled in older Python versions without SafeUUID.
            d['is_safe'] = self.is_safe.value
        return d

    def __setstate__(self, state):
        object.__setattr__(self, 'int', state['int'])
        # is_safe was added in 3.7; it is also omitted when it is "unknown"
        object.__setattr__(self, 'is_safe',
                           SafeUUID(state['is_safe'])
                           if 'is_safe' in state else SafeUUID.unknown)

    def __eq__(self, other):
        if isinstance(other, UUID):
            return self.int == other.int
        return NotImplemented

    # Q. What's the value of being able to sort UUIDs?
    # A. Use them as keys in a B-Tree or similar mapping.

    def __lt__(self, other):
        if isinstance(other, UUID):
            return self.int < other.int
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, UUID):
            return self.int > other.int
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, UUID):
            return self.int <= other.int
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, UUID):
            return self.int >= other.int
        return NotImplemented

    def __hash__(self):
        return hash(self.int)

    def __int__(self):
        return self.int

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, str(self))

    def __setattr__(self, name, value):
        raise TypeError('UUID objects are immutable')

    def __str__(self):
        hex = '%032x' % self.int
        return '%s-%s-%s-%s-%s' % (
            hex[:8], hex[8:12], hex[12:16], hex[16:20], hex[20:])

    @property
    def bytes(self):
        return self.int.to_bytes(16, 'big')

    @property
    def bytes_le(self):
        bytes = self.bytes
        return (bytes[4-1::-1] + bytes[6-1:4-1:-1] + bytes[8-1:6-1:-1] +
                bytes[8:])

    @property
    def fields(self):
        return (self.time_low, self.time_mid, self.time_hi_version,
                self.clock_seq_hi_variant, self.clock_seq_low, self.node)

    @property
    def time_low(self):
        return self.int >> 96

    @property
    def time_mid(self):
        return (self.int >> 80) & 0xffff

    @property
    def time_hi_version(self):
        return (self.int >> 64) & 0xffff

    @property
    def clock_seq_hi_variant(self):
        return (self.int >> 56) & 0xff

    @property
    def clock_seq_low(self):
        return (self.int >> 48) & 0xff

    @property
    def time(self):
        return (((self.time_hi_version & 0x0fff) << 48) |
                (self.time_mid << 32) | self.time_low)

    @property
    def clock_seq(self):
        return (((self.clock_seq_hi_variant & 0x3f) << 8) |
                self.clock_seq_low)

    @property
    def node(self):
        return self.int & 0xffffffffffff

    @property
    def hex(self):
        return '%032x' % self.int

    @property
    def urn(self):
        return 'urn:uuid:' + str(self)

    @property
    def variant(self):
        if not self.int & (0x8000 << 48):
            return RESERVED_NCS
        elif not self.int & (0x4000 << 48):
            return RFC_4122
        elif not self.int & (0x2000 << 48):
            return RESERVED_MICROSOFT
        else:
            return RESERVED_FUTURE

    @property
    def version(self):
        # The version bits are only meaningful for RFC 4122 UUIDs.
        if self.variant == RFC_4122:
            return int((self.int >> 76) & 0xf)


def _get_command_stdout(command, *args):
    import io, os, shutil, subprocess

    try:
        path_dirs = os.environ.get('PATH', os.defpath).split(os.pathsep)
        path_dirs.extend(['/sbin', '/usr/sbin'])
        executable = shutil.which(command, path=os.pathsep.join(path_dirs))
        if executable is None:
            return None
        # LC_ALL=C to ensure English output, stderr=DEVNULL to prevent output
        # on stderr (Note: we don't have an example where the words we search
        # for are actually localized, but in theory some system could do so.)
        env = dict(os.environ)
        env['LC_ALL'] = 'C'
        # Empty strings will be quoted by popen so we should just ommit it
        if args != ('',):
            command = (executable, *args)
        else:
            command = (executable,)
        proc = subprocess.Popen(command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                env=env)
        if not proc:
            return None
        stdout, stderr = proc.communicate()
        return io.BytesIO(stdout)
    except (OSError, subprocess.SubprocessError):
        return None


# For MAC (a.k.a. IEEE 802, or EUI-48) addresses, the second least significant
# bit of the first octet signifies whether the MAC address is universally (0)
# or locally (1) administered.  Network cards from hardware manufacturers will
# always be universally administered to guarantee global uniqueness of the MAC
# address, but any particular machine may have other interfaces which are
# locally administered.  An example of the latter is the bridge interface to
# the Touch Bar on MacBook Pros.
#
# This bit works out to be the 42nd bit counting from 1 being the least
# significant, or 1<<41.  We'll prefer universally administered MAC addresses
# over locally administered ones since the former are globally unique, but
# we'll return the first of the latter found if that's all the machine has.
#
# See https://en.wikipedia.org/wiki/MAC_address#Universal_vs._local

def _is_universal(mac):
    return not (mac & (1 << 41))


def _find_mac_near_keyword(command, args, keywords, get_word_index):
    """Searches a command's output for a MAC address near a keyword.

    Each line of words in the output is case-insensitively searched for
    any of the given keywords.  Upon a match, get_word_index is invoked
    to pick a word from the line, given the index of the match.  For
    example, lambda i: 0 would get the first word on the line, while
    lambda i: i - 1 would get the word preceding the keyword.
    """
    stdout = _get_command_stdout(command, args)
    if stdout is None:
        return None

    first_local_mac = None
    for line in stdout:
        words = line.lower().rstrip().split()
        for i in range(len(words)):
            if words[i] in keywords:
                try:
                    word = words[get_word_index(i)]
                    mac = int(word.replace(_MAC_DELIM, b''), 16)
                except (ValueError, IndexError):
                    # Virtual interfaces, such as those provided by
                    # VPNs, do not have a colon-delimited MAC address
                    # as expected, but a 16-byte HWAddr separated by
                    # dashes. These should be ignored in favor of a
                    # real MAC address
                    pass
                else:
                    if _is_universal(mac):
                        return mac
                    first_local_mac = first_local_mac or mac
    return first_local_mac or None


def _parse_mac(word):
    # Accept 'HH:HH:HH:HH:HH:HH' MAC address (ex: '52:54:00:9d:0e:67'),
    # but reject IPv6 address (ex: 'fe80::5054:ff:fe9' or '123:2:3:4:5:6:7:8').
    #
    # Virtual interfaces, such as those provided by VPNs, do not have a
    # colon-delimited MAC address as expected, but a 16-byte HWAddr separated
    # by dashes. These should be ignored in favor of a real MAC address
    parts = word.split(_MAC_DELIM)
    if len(parts) != 6:
        return
    if _MAC_OMITS_LEADING_ZEROES:
        # (Only) on AIX the macaddr value given is not prefixed by 0, e.g.
        # en0   1500  link#2      fa.bc.de.f7.62.4 110854824     0 160133733     0     0
        # not
        # en0   1500  link#2      fa.bc.de.f7.62.04 110854824     0 160133733     0     0
        if not all(1 <= len(part) <= 2 for part in parts):
            return
        hexstr = b''.join(part.rjust(2, b'0') for part in parts)
    else:
        if not all(len(part) == 2 for part in parts):
            return
        hexstr = b''.join(parts)
    try:
        return int(hexstr, 16)
    except ValueError:
        return


def _find_mac_under_heading(command, args, heading):
    """Looks for a MAC address under a heading in a command's output.

    The first line of words in the output is searched for the given
    heading. Words at the same word index as the heading in subsequent
    lines are then examined to see if they look like MAC addresses.
    """
    stdout = _get_command_stdout(command, args)
    if stdout is None:
        return None

    keywords = stdout.readline().rstrip().split()
    try:
        column_index = keywords.index(heading)
    except ValueError:
        return None

    first_local_mac = None
    for line in stdout:
        words = line.rstrip().split()
        try:
            word = words[column_index]
        except IndexError:
            continue

        mac = _parse_mac(word)
        if mac is None:
            continue
        if _is_universal(mac):
            return mac
        if first_local_mac is None:
            first_local_mac = mac

    return first_local_mac


# The following functions call external programs to 'get' a macaddr value to
# be used as basis for an uuid
def _ifconfig_getnode():
    """Get the hardware address on Unix by running ifconfig."""
    # This works on Linux ('' or '-a'), Tru64 ('-av'), but not all Unixes.
    keywords = (b'hwaddr', b'ether', b'address:', b'lladdr')
    for args in ('', '-a', '-av'):
        mac = _find_mac_near_keyword('ifconfig', args, keywords, lambda i: i+1)
        if mac:
            return mac
    return None

def _ip_getnode():
    """Get the hardware address on Unix by running ip."""
    # This works on Linux with iproute2.
    mac = _find_mac_near_keyword('ip', 'link', [b'link/ether'], lambda i: i+1)
    if mac:
        return mac
    return None

def _arp_getnode():
    """Get the hardware address on Unix by running arp."""
    import os, socket
    try:
        ip_addr = socket.gethostbyname(socket.gethostname())
    except OSError:
        return None

    # Try getting the MAC addr from arp based on our IP address (Solaris).
    mac = _find_mac_near_keyword('arp', '-an', [os.fsencode(ip_addr)], lambda i: -1)
    if mac:
        return mac

    # This works on OpenBSD
    mac = _find_mac_near_keyword('arp', '-an', [os.fsencode(ip_addr)], lambda i: i+1)
    if mac:
        return mac

    # This works on Linux, FreeBSD and NetBSD
    mac = _find_mac_near_keyword('arp', '-an', [os.fsencode('(%s)' % ip_addr)],
                    lambda i: i+2)
    # Return None instead of 0.
    if mac:
        return mac
    return None

def _lanscan_getnode():
    """Get the hardware address on Unix by running lanscan."""
    # This might work on HP-UX.
    return _find_mac_near_keyword('lanscan', '-ai', [b'lan0'], lambda i: 0)

def _netstat_getnode():
    """Get the hardware address on Unix by running netstat."""
    # This works on AIX and might work on Tru64 UNIX.
    return _find_mac_under_heading('netstat', '-ian', b'Address')

def _ipconfig_getnode():
    """[DEPRECATED] Get the hardware address on Windows."""
    # bpo-40501: UuidCreateSequential() is now the only supported approach
    return _windll_getnode()

def _netbios_getnode():
    """[DEPRECATED] Get the hardware address on Windows."""
    # bpo-40501: UuidCreateSequential() is now the only supported approach
    return _windll_getnode()


# Import optional C extension at toplevel, to help disabling it when testing
try:
    import _uuid
    _generate_time_safe = getattr(_uuid, "generate_time_safe", None)
    _UuidCreate = getattr(_uuid, "UuidCreate", None)
    _has_uuid_generate_time_safe = _uuid.has_uuid_generate_time_safe
except ImportError:
    _uuid = None
    _generate_time_safe = None
    _UuidCreate = None
    _has_uuid_generate_time_safe = None


def _load_system_functions():
    """[DEPRECATED] Platform-specific functions loaded at import time"""


def _unix_getnode():
    """Get the hardware address on Unix using the _uuid extension module."""
    if _generate_time_safe:
        uuid_time, _ = _generate_time_safe()
        return UUID(bytes=uuid_time).node

def _windll_getnode():
    """Get the hardware address on Windows using the _uuid extension module."""
    if _UuidCreate:
        uuid_bytes = _UuidCreate()
        return UUID(bytes_le=uuid_bytes).node

def _random_getnode():
    """Get a random node ID."""
    # RFC 4122, $4.1.6 says "For systems with no IEEE address, a randomly or
    # pseudo-randomly generated value may be used; see Section 4.5.  The
    # multicast bit must be set in such addresses, in order that they will
    # never conflict with addresses obtained from network cards."
    #
    # The "multicast bit" of a MAC address is defined to be "the least
    # significant bit of the first octet".  This works out to be the 41st bit
    # counting from 1 being the least significant bit, or 1<<40.
    #
    # See https://en.wikipedia.org/wiki/MAC_address#Unicast_vs._multicast
    import random
    return random.getrandbits(48) | (1 << 40)


# _OS_GETTERS, when known, are targeted for a specific OS or platform.
# The order is by 'common practice' on the specified platform.
# Note: 'posix' and 'windows' _OS_GETTERS are prefixed by a dll/dlload() method
# which, when successful, means none of these "external" methods are called.
# _GETTERS is (also) used by test_uuid.py to SkipUnless(), e.g.,
#     @unittest.skipUnless(_uuid._ifconfig_getnode in _uuid._GETTERS, ...)
if _LINUX:
    _OS_GETTERS = [_ip_getnode, _ifconfig_getnode]
elif sys.platform == 'darwin':
    _OS_GETTERS = [_ifconfig_getnode, _arp_getnode, _netstat_getnode]
elif sys.platform == 'win32':
    # bpo-40201: _windll_getnode will always succeed, so these are not needed
    _OS_GETTERS = []
elif _AIX:
    _OS_GETTERS = [_netstat_getnode]
else:
    _OS_GETTERS = [_ifconfig_getnode, _ip_getnode, _arp_getnode,
                   _netstat_getnode, _lanscan_getnode]
if os.name == 'posix':
    _GETTERS = [_unix_getnode] + _OS_GETTERS
elif os.name == 'nt':
    _GETTERS = [_windll_getnode] + _OS_GETTERS
else:
    _GETTERS = _OS_GETTERS

_node = None

def getnode():
    """Get the hardware address as a 48-bit positive integer.

    The first time this runs, it may launch a separate program, which could
    be quite slow.  If all attempts to obtain the hardware address fail, we
    choose a random 48-bit number with its eighth bit set to 1 as recommended
    in RFC 4122.
    """
    global _node
    if _node is not None:
        return _node

    for getter in _GETTERS + [_random_getnode]:
        try:
            _node = getter()
        except:
            continue
        if (_node is not None) and (0 <= _node < (1 << 48)):
            return _node
    assert False, '_random_getnode() returned invalid value: {}'.format(_node)


_last_timestamp = None

def uuid1(node=None, clock_seq=None):
    """Generate a UUID from a host ID, sequence number, and the current time.
    If 'node' is not given, getnode() is used to obtain the hardware
    address.  If 'clock_seq' is given, it is used as the sequence number;
    otherwise a random 14-bit sequence number is chosen."""

    # When the system provides a version-1 UUID generator, use it (but don't
    # use UuidCreate here because its UUIDs don't conform to RFC 4122).
    if _generate_time_safe is not None and node is clock_seq is None:
        uuid_time, safely_generated = _generate_time_safe()
        try:
            is_safe = SafeUUID(safely_generated)
        except ValueError:
            is_safe = SafeUUID.unknown
        return UUID(bytes=uuid_time, is_safe=is_safe)

    global _last_timestamp
    import time
    nanoseconds = time.time_ns()
    # 0x01b21dd213814000 is the number of 100-ns intervals between the
    # UUID epoch 1582-10-15 00:00:00 and the Unix epoch 1970-01-01 00:00:00.
    timestamp = nanoseconds // 100 + 0x01b21dd213814000
    if _last_timestamp is not None and timestamp <= _last_timestamp:
        timestamp = _last_timestamp + 1
    _last_timestamp = timestamp
    if clock_seq is None:
        import random
        clock_seq = random.getrandbits(14) # instead of stable storage
    time_low = timestamp & 0xffffffff
    time_mid = (timestamp >> 32) & 0xffff
    time_hi_version = (timestamp >> 48) & 0x0fff
    clock_seq_low = clock_seq & 0xff
    clock_seq_hi_variant = (clock_seq >> 8) & 0x3f
    if node is None:
        node = getnode()
    return UUID(fields=(time_low, time_mid, time_hi_version,
                        clock_seq_hi_variant, clock_seq_low, node), version=1)

def uuid3(namespace, name):
    """Generate a UUID from the MD5 hash of a namespace UUID and a name."""
    from hashlib import md5
    digest = md5(
        namespace.bytes + bytes(name, "utf-8"),
        usedforsecurity=False
    ).digest()
    return UUID(bytes=digest[:16], version=3)

def uuid4():
    """Generate a random UUID."""
    return UUID(bytes=os.urandom(16), version=4)

def uuid5(namespace, name):
    """Generate a UUID from the SHA-1 hash of a namespace UUID and a name."""
    from hashlib import sha1
    hash = sha1(namespace.bytes + bytes(name, "utf-8")).digest()
    return UUID(bytes=hash[:16], version=5)

# The following standard UUIDs are for use with uuid3() or uuid5().

NAMESPACE_DNS = UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
NAMESPACE_URL = UUID('6ba7b811-9dad-11d1-80b4-00c04fd430c8')
NAMESPACE_OID = UUID('6ba7b812-9dad-11d1-80b4-00c04fd430c8')
NAMESPACE_X500 = UUID('6ba7b814-9dad-11d1-80b4-00c04fd430c8')
<|endoftext|>"""A lexical analyzer class for simple shell-like syntaxes."""

# Module and documentation by Eric S. Raymond, 21 Dec 1998
# Input stacking and error message cleanup added by ESR, March 2000
# push_source() and pop_source() made explicit by ESR, January 2001.
# Posix compliance, split(), string arguments, and
# iterator interface by Gustavo Niemeyer, April 2003.
# changes to tokenize more like Posix shells by Vinay Sajip, July 2016.

import os
import re
import sys
from collections import deque

from io import StringIO

__all__ = ["shlex", "split", "quote", "join"]

class shlex:
    "A lexical analyzer class for simple shell-like syntaxes."
    def __init__(self, instream=None, infile=None, posix=False,
                 punctuation_chars=False):
        if isinstance(instream, str):
            instream = StringIO(instream)
        if instream is not None:
            self.instream = instream
            self.infile = infile
        else:
            self.instream = sys.stdin
            self.infile = None
        self.posix = posix
        if posix:
            self.eof = None
        else:
            self.eof = ''
        self.commenters = '#'
        self.wordchars = ('abcdfeghijklmnopqrstuvwxyz'
                          'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
        if self.posix:
            self.wordchars += ('ßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
                               'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ')
        self.whitespace = ' \t\r\n'
        self.whitespace_split = False
        self.quotes = '\'"'
        self.escape = '\\'
        self.escapedquotes = '"'
        self.state = ' '
        self.pushback = deque()
        self.lineno = 1
        self.debug = 0
        self.token = ''
        self.filestack = deque()
        self.source = None
        if not punctuation_chars:
            punctuation_chars = ''
        elif punctuation_chars is True:
            punctuation_chars = '();<>|&'
        self._punctuation_chars = punctuation_chars
        if punctuation_chars:
            # _pushback_chars is a push back queue used by lookahead logic
            self._pushback_chars = deque()
            # these chars added because allowed in file names, args, wildcards
            self.wordchars += '~-./*?='
            #remove any punctuation chars from wordchars
            t = self.wordchars.maketrans(dict.fromkeys(punctuation_chars))
            self.wordchars = self.wordchars.translate(t)

    @property
    def punctuation_chars(self):
        return self._punctuation_chars

    def push_token(self, tok):
        "Push a token onto the stack popped by the get_token method"
        if self.debug >= 1:
            print("shlex: pushing token " + repr(tok))
        self.pushback.appendleft(tok)

    def push_source(self, newstream, newfile=None):
        "Push an input source onto the lexer's input source stack."
        if isinstance(newstream, str):
            newstream = StringIO(newstream)
        self.filestack.appendleft((self.infile, self.instream, self.lineno))
        self.infile = newfile
        self.instream = newstream
        self.lineno = 1
        if self.debug:
            if newfile is not None:
                print('shlex: pushing to file %s' % (self.infile,))
            else:
                print('shlex: pushing to stream %s' % (self.instream,))

    def pop_source(self):
        "Pop the input source stack."
        self.instream.close()
        (self.infile, self.instream, self.lineno) = self.filestack.popleft()
        if self.debug:
            print('shlex: popping to %s, line %d' \
                  % (self.instream, self.lineno))
        self.state = ' '

    def get_token(self):
        "Get a token from the input stream (or from stack if it's nonempty)"
        if self.pushback:
            tok = self.pushback.popleft()
            if self.debug >= 1:
                print("shlex: popping token " + repr(tok))
            return tok
        # No pushback.  Get a token.
        raw = self.read_token()
        # Handle inclusions
        if self.source is not None:
            while raw == self.source:
                spec = self.sourcehook(self.read_token())
                if spec:
                    (newfile, newstream) = spec
                    self.push_source(newstream, newfile)
                raw = self.get_token()
        # Maybe we got EOF instead?
        while raw == self.eof:
            if not self.filestack:
                return self.eof
            else:
                self.pop_source()
                raw = self.get_token()
        # Neither inclusion nor EOF
        if self.debug >= 1:
            if raw != self.eof:
                print("shlex: token=" + repr(raw))
            else:
                print("shlex: token=EOF")
        return raw

    def read_token(self):
        quoted = False
        escapedstate = ' '
        while True:
            if self.punctuation_chars and self._pushback_chars:
                nextchar = self._pushback_chars.pop()
            else:
                nextchar = self.instream.read(1)
            if nextchar == '\n':
                self.lineno += 1
            if self.debug >= 3:
                print("shlex: in state %r I see character: %r" % (self.state,
                                                                  nextchar))
            if self.state is None:
                self.token = ''        # past end of file
                break
            elif self.state == ' ':
                if not nextchar:
                    self.state = None  # end of file
                    break
                elif nextchar in self.whitespace:
                    if self.debug >= 2:
                        print("shlex: I see whitespace in whitespace state")
                    if self.token or (self.posix and quoted):
                        break   # emit current token
                    else:
                        continue
                elif nextchar in self.commenters:
                    self.instream.readline()
                    self.lineno += 1
                elif self.posix and nextchar in self.escape:
                    escapedstate = 'a'
                    self.state = nextchar
                elif nextchar in self.wordchars:
                    self.token = nextchar
                    self.state = 'a'
                elif nextchar in self.punctuation_chars:
                    self.token = nextchar
                    self.state = 'c'
                elif nextchar in self.quotes:
                    if not self.posix:
                        self.token = nextchar
                    self.state = nextchar
                elif self.whitespace_split:
                    self.token = nextchar
                    self.state = 'a'
                else:
                    self.token = nextchar
                    if self.token or (self.posix and quoted):
                        break   # emit current token
                    else:
                        continue
            elif self.state in self.quotes:
                quoted = True
                if not nextchar:      # end of file
                    if self.debug >= 2:
                        print("shlex: I see EOF in quotes state")
                    # XXX what error should be raised here?
                    raise ValueError("No closing quotation")
                if nextchar == self.state:
                    if not self.posix:
                        self.token += nextchar
                        self.state = ' '
                        break
                    else:
                        self.state = 'a'
                elif (self.posix and nextchar in self.escape and self.state
                      in self.escapedquotes):
                    escapedstate = self.state
                    self.state = nextchar
                else:
                    self.token += nextchar
            elif self.state in self.escape:
                if not nextchar:      # end of file
                    if self.debug >= 2:
                        print("shlex: I see EOF in escape state")
                    # XXX what error should be raised here?
                    raise ValueError("No escaped character")
                # In posix shells, only the quote itself or the escape
                # character may be escaped within quotes.
                if (escapedstate in self.quotes and
                        nextchar != self.state and nextchar != escapedstate):
                    self.token += self.state
                self.token += nextchar
                self.state = escapedstate
            elif self.state in ('a', 'c'):
                if not nextchar:
                    self.state = None   # end of file
                    break
                elif nextchar in self.whitespace:
                    if self.debug >= 2:
                        print("shlex: I see whitespace in word state")
                    self.state = ' '
                    if self.token or (self.posix and quoted):
                        break   # emit current token
                    else:
                        continue
                elif nextchar in self.commenters:
                    self.instream.readline()
                    self.lineno += 1
                    if self.posix:
                        self.state = ' '
                        if self.token or (self.posix and quoted):
                            break   # emit current token
                        else:
                            continue
                elif self.state == 'c':
                    if nextchar in self.punctuation_chars:
                        self.token += nextchar
                    else:
                        if nextchar not in self.whitespace:
                            self._pushback_chars.append(nextchar)
                        self.state = ' '
                        break
                elif self.posix and nextchar in self.quotes:
                    self.state = nextchar
                elif self.posix and nextchar in self.escape:
                    escapedstate = 'a'
                    self.state = nextchar
                elif (nextchar in self.wordchars or nextchar in self.quotes
                      or (self.whitespace_split and
                          nextchar not in self.punctuation_chars)):
                    self.token += nextchar
                else:
                    if self.punctuation_chars:
                        self._pushback_chars.append(nextchar)
                    else:
                        self.pushback.appendleft(nextchar)
                    if self.debug >= 2:
                        print("shlex: I see punctuation in word state")
                    self.state = ' '
                    if self.token or (self.posix and quoted):
                        break   # emit current token
                    else:
                        continue
        result = self.token
        self.token = ''
        if self.posix and not quoted and result == '':
            result = None
        if self.debug > 1:
            if result:
                print("shlex: raw token=" + repr(result))
            else:
                print("shlex: raw token=EOF")
        return result

    def sourcehook(self, newfile):
        "Hook called on a filename to be sourced."
        if newfile[0] == '"':
            newfile = newfile[1:-1]
        # This implements cpp-like semantics for relative-path inclusion.
        if isinstance(self.infile, str) and not os.path.isabs(newfile):
            newfile = os.path.join(os.path.dirname(self.infile), newfile)
        return (newfile, open(newfile, "r"))

    def error_leader(self, infile=None, lineno=None):
        "Emit a C-compiler-like, Emacs-friendly error-message leader."
        if infile is None:
            infile = self.infile
        if lineno is None:
            lineno = self.lineno
        return "\"%s\", line %d: " % (infile, lineno)

    def __iter__(self):
        return self

    def __next__(self):
        token = self.get_token()
        if token == self.eof:
            raise StopIteration
        return token

def split(s, comments=False, posix=True):
    """Split the string *s* using shell-like syntax."""
    if s is None:
        import warnings
        warnings.warn("Passing None for 's' to shlex.split() is deprecated.",
                      DeprecationWarning, stacklevel=2)
    lex = shlex(s, posix=posix)
    lex.whitespace_split = True
    if not comments:
        lex.commenters = ''
    return list(lex)


def join(split_command):
    """Return a shell-escaped string from *split_command*."""
    return ' '.join(quote(arg) for arg in split_command)


_find_unsafe = re.compile(r'[^\w@%+=:,./-]', re.ASCII).search

def quote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"
    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _print_tokens(lexer):
    while 1:
        tt = lexer.get_token()
        if not tt:
            break
        print("Token: " + repr(tt))

if __name__ == '__main__':
    if len(sys.argv) == 1:
        _print_tokens(shlex())
    else:
        fn = sys.argv[1]
        with open(fn) as f:
            _print_tokens(shlex(f, fn))
<|endoftext|>#! /usr/bin/env python3
"""Interfaces for launching and remotely controlling web browsers."""
# Maintained by Georg Brandl.

import os
import shlex
import shutil
import sys
import subprocess
import threading

__all__ = ["Error", "open", "open_new", "open_new_tab", "get", "register"]

class Error(Exception):
    pass

_lock = threading.RLock()
_browsers = {}                  # Dictionary of available browser controllers
_tryorder = None                # Preference order of available browsers
_os_preferred_browser = None    # The preferred browser

def register(name, klass, instance=None, *, preferred=False):
    """Register a browser connector."""
    with _lock:
        if _tryorder is None:
            register_standard_browsers()
        _browsers[name.lower()] = [klass, instance]

        # Preferred browsers go to the front of the list.
        # Need to match to the default browser returned by xdg-settings, which
        # may be of the form e.g. "firefox.desktop".
        if preferred or (_os_preferred_browser and name in _os_preferred_browser):
            _tryorder.insert(0, name)
        else:
            _tryorder.append(name)

def get(using=None):
    """Return a browser launcher instance appropriate for the environment."""
    if _tryorder is None:
        with _lock:
            if _tryorder is None:
                register_standard_browsers()
    if using is not None:
        alternatives = [using]
    else:
        alternatives = _tryorder
    for browser in alternatives:
        if '%s' in browser:
            # User gave us a command line, split it into name and args
            browser = shlex.split(browser)
            if browser[-1] == '&':
                return BackgroundBrowser(browser[:-1])
            else:
                return GenericBrowser(browser)
        else:
            # User gave us a browser name or path.
            try:
                command = _browsers[browser.lower()]
            except KeyError:
                command = _synthesize(browser)
            if command[1] is not None:
                return command[1]
            elif command[0] is not None:
                return command[0]()
    raise Error("could not locate runnable browser")

# Please note: the following definition hides a builtin function.
# It is recommended one does "import webbrowser" and uses webbrowser.open(url)
# instead of "from webbrowser import *".

def open(url, new=0, autoraise=True):
    """Display url using the default browser.

    If possible, open url in a location determined by new.
    - 0: the same browser window (the default).
    - 1: a new browser window.
    - 2: a new browser page ("tab").
    If possible, autoraise raises the window (the default) or not.
    """
    if _tryorder is None:
        with _lock:
            if _tryorder is None:
                register_standard_browsers()
    for name in _tryorder:
        browser = get(name)
        if browser.open(url, new, autoraise):
            return True
    return False

def open_new(url):
    """Open url in a new window of the default browser.

    If not possible, then open url in the only browser window.
    """
    return open(url, 1)

def open_new_tab(url):
    """Open url in a new page ("tab") of the default browser.

    If not possible, then the behavior becomes equivalent to open_new().
    """
    return open(url, 2)


def _synthesize(browser, *, preferred=False):
    """Attempt to synthesize a controller based on existing controllers.

    This is useful to create a controller when a user specifies a path to
    an entry in the BROWSER environment variable -- we can copy a general
    controller to operate using a specific installation of the desired
    browser in this way.

    If we can't create a controller in this way, or if there is no
    executable for the requested browser, return [None, None].

    """
    cmd = browser.split()[0]
    if not shutil.which(cmd):
        return [None, None]
    name = os.path.basename(cmd)
    try:
        command = _browsers[name.lower()]
    except KeyError:
        return [None, None]
    # now attempt to clone to fit the new name:
    controller = command[1]
    if controller and name.lower() == controller.basename:
        import copy
        controller = copy.copy(controller)
        controller.name = browser
        controller.basename = os.path.basename(browser)
        register(browser, None, instance=controller, preferred=preferred)
        return [None, controller]
    return [None, None]


# General parent classes

class BaseBrowser(object):
    """Parent class for all browsers. Do not use directly."""

    args = ['%s']

    def __init__(self, name=""):
        self.name = name
        self.basename = name

    def open(self, url, new=0, autoraise=True):
        raise NotImplementedError

    def open_new(self, url):
        return self.open(url, 1)

    def open_new_tab(self, url):
        return self.open(url, 2)

    @staticmethod
    def _check_url(url):
        """Ensures that the URL is safe to pass to subprocesses as a parameter"""
        if url and url.lstrip().startswith("-"):
            raise ValueError(f"Invalid URL (leading dash disallowed): {url!r}")


class GenericBrowser(BaseBrowser):
    """Class for all browsers started with a command
       and without remote functionality."""

    def __init__(self, name):
        if isinstance(name, str):
            self.name = name
            self.args = ["%s"]
        else:
            # name should be a list with arguments
            self.name = name[0]
            self.args = name[1:]
        self.basename = os.path.basename(self.name)

    def open(self, url, new=0, autoraise=True):
        sys.audit("webbrowser.open", url)
        self._check_url(url)
        cmdline = [self.name] + [arg.replace("%s", url)
                                 for arg in self.args]
        try:
            if sys.platform[:3] == 'win':
                p = subprocess.Popen(cmdline)
            else:
                p = subprocess.Popen(cmdline, close_fds=True)
            return not p.wait()
        except OSError:
            return False


class BackgroundBrowser(GenericBrowser):
    """Class for all browsers which are to be started in the
       background."""

    def open(self, url, new=0, autoraise=True):
        cmdline = [self.name] + [arg.replace("%s", url)
                                 for arg in self.args]
        sys.audit("webbrowser.open", url)
        self._check_url(url)
        try:
            if sys.platform[:3] == 'win':
                p = subprocess.Popen(cmdline)
            else:
                p = subprocess.Popen(cmdline, close_fds=True,
                                     start_new_session=True)
            return (p.poll() is None)
        except OSError:
            return False


class UnixBrowser(BaseBrowser):
    """Parent class for all Unix browsers with remote functionality."""

    raise_opts = None
    background = False
    redirect_stdout = True
    # In remote_args, %s will be replaced with the requested URL.  %action will
    # be replaced depending on the value of 'new' passed to open.
    # remote_action is used for new=0 (open).  If newwin is not None, it is
    # used for new=1 (open_new).  If newtab is not None, it is used for
    # new=3 (open_new_tab).  After both substitutions are made, any empty
    # strings in the transformed remote_args list will be removed.
    remote_args = ['%action', '%s']
    remote_action = None
    remote_action_newwin = None
    remote_action_newtab = None

    def _invoke(self, args, remote, autoraise, url=None):
        raise_opt = []
        if remote and self.raise_opts:
            # use autoraise argument only for remote invocation
            autoraise = int(autoraise)
            opt = self.raise_opts[autoraise]
            if opt: raise_opt = [opt]

        cmdline = [self.name] + raise_opt + args

        if remote or self.background:
            inout = subprocess.DEVNULL
        else:
            # for TTY browsers, we need stdin/out
            inout = None
        p = subprocess.Popen(cmdline, close_fds=True, stdin=inout,
                             stdout=(self.redirect_stdout and inout or None),
                             stderr=inout, start_new_session=True)
        if remote:
            # wait at most five seconds. If the subprocess is not finished, the
            # remote invocation has (hopefully) started a new instance.
            try:
                rc = p.wait(5)
                # if remote call failed, open() will try direct invocation
                return not rc
            except subprocess.TimeoutExpired:
                return True
        elif self.background:
            if p.poll() is None:
                return True
            else:
                return False
        else:
            return not p.wait()

    def open(self, url, new=0, autoraise=True):
        sys.audit("webbrowser.open", url)
        if new == 0:
            action = self.remote_action
        elif new == 1:
            action = self.remote_action_newwin
        elif new == 2:
            if self.remote_action_newtab is None:
                action = self.remote_action_newwin
            else:
                action = self.remote_action_newtab
        else:
            raise Error("Bad 'new' parameter to open(); " +
                        "expected 0, 1, or 2, got %s" % new)

        self._check_url(url.replace("%action", action))

        args = [arg.replace("%action", action).replace("%s", url)
                for arg in self.remote_args]
        args = [arg for arg in args if arg]
        success = self._invoke(args, True, autoraise, url)
        if not success:
            # remote invocation failed, try straight way
            args = [arg.replace("%s", url) for arg in self.args]
            return self._invoke(args, False, False)
        else:
            return True


class Mozilla(UnixBrowser):
    """Launcher class for Mozilla browsers."""

    remote_args = ['%action', '%s']
    remote_action = ""
    remote_action_newwin = "-new-window"
    remote_action_newtab = "-new-tab"
    background = True


class Netscape(UnixBrowser):
    """Launcher class for Netscape browser."""

    raise_opts = ["-noraise", "-raise"]
    remote_args = ['-remote', 'openURL(%s%action)']
    remote_action = ""
    remote_action_newwin = ",new-window"
    remote_action_newtab = ",new-tab"
    background = True


class Galeon(UnixBrowser):
    """Launcher class for Galeon/Epiphany browsers."""

    raise_opts = ["-noraise", ""]
    remote_args = ['%action', '%s']
    remote_action = "-n"
    remote_action_newwin = "-w"
    background = True


class Chrome(UnixBrowser):
    "Launcher class for Google Chrome browser."

    remote_args = ['%action', '%s']
    remote_action = ""
    remote_action_newwin = "--new-window"
    remote_action_newtab = ""
    background = True

Chromium = Chrome


class Opera(UnixBrowser):
    "Launcher class for Opera browser."

    remote_args = ['%action', '%s']
    remote_action = ""
    remote_action_newwin = "--new-window"
    remote_action_newtab = ""
    background = True


class Elinks(UnixBrowser):
    "Launcher class for Elinks browsers."

    remote_args = ['-remote', 'openURL(%s%action)']
    remote_action = ""
    remote_action_newwin = ",new-window"
    remote_action_newtab = ",new-tab"
    background = False

    # elinks doesn't like its stdout to be redirected -
    # it uses redirected stdout as a signal to do -dump
    redirect_stdout = False


class Konqueror(BaseBrowser):
    """Controller for the KDE File Manager (kfm, or Konqueror).

    See the output of ``kfmclient --commands``
    for more information on the Konqueror remote-control interface.
    """

    def open(self, url, new=0, autoraise=True):
        sys.audit("webbrowser.open", url)
        self._check_url(url)
        # XXX Currently I know no way to prevent KFM from opening a new win.
        if new == 2:
            action = "newTab"
        else:
            action = "openURL"

        devnull = subprocess.DEVNULL

        try:
            p = subprocess.Popen(["kfmclient", action, url],
                                 close_fds=True, stdin=devnull,
                                 stdout=devnull, stderr=devnull)
        except OSError:
            # fall through to next variant
            pass
        else:
            p.wait()
            # kfmclient's return code unfortunately has no meaning as it seems
            return True

        try:
            p = subprocess.Popen(["konqueror", "--silent", url],
                                 close_fds=True, stdin=devnull,
                                 stdout=devnull, stderr=devnull,
                                 start_new_session=True)
        except OSError:
            # fall through to next variant
            pass
        else:
            if p.poll() is None:
                # Should be running now.
                return True

        try:
            p = subprocess.Popen(["kfm", "-d", url],
                                 close_fds=True, stdin=devnull,
                                 stdout=devnull, stderr=devnull,
                                 start_new_session=True)
        except OSError:
            return False
        else:
            return (p.poll() is None)


class Grail(BaseBrowser):
    # There should be a way to maintain a connection to Grail, but the
    # Grail remote control protocol doesn't really allow that at this
    # point.  It probably never will!
    def _find_grail_rc(self):
        import glob
        import pwd
        import socket
        import tempfile
        tempdir = os.path.join(tempfile.gettempdir(),
                               ".grail-unix")
        user = pwd.getpwuid(os.getuid())[0]
        filename = os.path.join(glob.escape(tempdir), glob.escape(user) + "-*")
        maybes = glob.glob(filename)
        if not maybes:
            return None
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        for fn in maybes:
            # need to PING each one until we find one that's live
            try:
                s.connect(fn)
            except OSError:
                # no good; attempt to clean it out, but don't fail:
                try:
                    os.unlink(fn)
                except OSError:
                    pass
            else:
                return s

    def _remote(self, action):
        s = self._find_grail_rc()
        if not s:
            return 0
        s.send(action)
        s.close()
        return 1

    def open(self, url, new=0, autoraise=True):
        sys.audit("webbrowser.open", url)
        self._check_url(url)
        if new:
            ok = self._remote("LOADNEW " + url)
        else:
            ok = self._remote("LOAD " + url)
        return ok


#
# Platform support for Unix
#

# These are the right tests because all these Unix browsers require either
# a console terminal or an X display to run.

def register_X_browsers():

    # use xdg-open if around
    if shutil.which("xdg-open"):
        register("xdg-open", None, BackgroundBrowser("xdg-open"))

    # The default GNOME3 browser
    if "GNOME_DESKTOP_SESSION_ID" in os.environ and shutil.which("gvfs-open"):
        register("gvfs-open", None, BackgroundBrowser("gvfs-open"))

    # The default GNOME browser
    if "GNOME_DESKTOP_SESSION_ID" in os.environ and shutil.which("gnome-open"):
        register("gnome-open", None, BackgroundBrowser("gnome-open"))

    # The default KDE browser
    if "KDE_FULL_SESSION" in os.environ and shutil.which("kfmclient"):
        register("kfmclient", Konqueror, Konqueror("kfmclient"))

    if shutil.which("x-www-browser"):
        register("x-www-browser", None, BackgroundBrowser("x-www-browser"))

    # The Mozilla browsers
    for browser in ("firefox", "iceweasel", "iceape", "seamonkey"):
        if shutil.which(browser):
            register(browser, None, Mozilla(browser))

    # The Netscape and old Mozilla browsers
    for browser in ("mozilla-firefox",
                    "mozilla-firebird", "firebird",
                    "mozilla", "netscape"):
        if shutil.which(browser):
            register(browser, None, Netscape(browser))

    # Konqueror/kfm, the KDE browser.
    if shutil.which("kfm"):
        register("kfm", Konqueror, Konqueror("kfm"))
    elif shutil.which("konqueror"):
        register("konqueror", Konqueror, Konqueror("konqueror"))

    # Gnome's Galeon and Epiphany
    for browser in ("galeon", "epiphany"):
        if shutil.which(browser):
            register(browser, None, Galeon(browser))

    # Skipstone, another Gtk/Mozilla based browser
    if shutil.which("skipstone"):
        register("skipstone", None, BackgroundBrowser("skipstone"))

    # Google Chrome/Chromium browsers
    for browser in ("google-chrome", "chrome", "chromium", "chromium-browser"):
        if shutil.which(browser):
            register(browser, None, Chrome(browser))

    # Opera, quite popular
    if shutil.which("opera"):
        register("opera", None, Opera("opera"))

    # Next, Mosaic -- old but still in use.
    if shutil.which("mosaic"):
        register("mosaic", None, BackgroundBrowser("mosaic"))

    # Grail, the Python browser. Does anybody still use it?
    if shutil.which("grail"):
        register("grail", Grail, None)

def register_standard_browsers():
    global _tryorder
    _tryorder = []

    if sys.platform == 'darwin':
        register("MacOSX", None, MacOSXOSAScript('default'))
        register("chrome", None, MacOSXOSAScript('chrome'))
        register("firefox", None, MacOSXOSAScript('firefox'))
        register("safari", None, MacOSXOSAScript('safari'))
        # OS X can use below Unix support (but we prefer using the OS X
        # specific stuff)

    if sys.platform == "serenityos":
        # SerenityOS webbrowser, simply called "Browser".
        register("Browser", None, BackgroundBrowser("Browser"))

    if sys.platform[:3] == "win":
        # First try to use the default Windows browser
        register("windows-default", WindowsDefault)

        # Detect some common Windows browsers, fallback to IE
        iexplore = os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"),
                                "Internet Explorer\\IEXPLORE.EXE")
        for browser in ("firefox", "firebird", "seamonkey", "mozilla",
                        "netscape", "opera", iexplore):
            if shutil.which(browser):
                register(browser, None, BackgroundBrowser(browser))
    else:
        # Prefer X browsers if present
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            try:
                cmd = "xdg-settings get default-web-browser".split()
                raw_result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                result = raw_result.decode().strip()
            except (FileNotFoundError, subprocess.CalledProcessError, PermissionError, NotADirectoryError) :
                pass
            else:
                global _os_preferred_browser
                _os_preferred_browser = result

            register_X_browsers()

        # Also try console browsers
        if os.environ.get("TERM"):
            if shutil.which("www-browser"):
                register("www-browser", None, GenericBrowser("www-browser"))
            # The Links/elinks browsers <http://artax.karlin.mff.cuni.cz/~mikulas/links/>
            if shutil.which("links"):
                register("links", None, GenericBrowser("links"))
            if shutil.which("elinks"):
                register("elinks", None, Elinks("elinks"))
            # The Lynx browser <http://lynx.isc.org/>, <http://lynx.browser.org/>
            if shutil.which("lynx"):
                register("lynx", None, GenericBrowser("lynx"))
            # The w3m browser <http://w3m.sourceforge.net/>
            if shutil.which("w3m"):
                register("w3m", None, GenericBrowser("w3m"))

    # OK, now that we know what the default preference orders for each
    # platform are, allow user to override them with the BROWSER variable.
    if "BROWSER" in os.environ:
        userchoices = os.environ["BROWSER"].split(os.pathsep)
        userchoices.reverse()

        # Treat choices in same way as if passed into get() but do register
        # and prepend to _tryorder
        for cmdline in userchoices:
            if cmdline != '':
                cmd = _synthesize(cmdline, preferred=True)
                if cmd[1] is None:
                    register(cmdline, None, GenericBrowser(cmdline), preferred=True)

    # what to do if _tryorder is now empty?


#
# Platform support for Windows
#

if sys.platform[:3] == "win":
    class WindowsDefault(BaseBrowser):
        def open(self, url, new=0, autoraise=True):
            sys.audit("webbrowser.open", url)
            self._check_url(url)
            try:
                os.startfile(url)
            except OSError:
                # [Error 22] No application is associated with the specified
                # file for this operation: '<URL>'
                return False
            else:
                return True

#
# Platform support for MacOS
#

if sys.platform == 'darwin':
    # Adapted from patch submitted to SourceForge by Steven J. Burr
    class MacOSX(BaseBrowser):
        """Launcher class for Aqua browsers on Mac OS X

        Optionally specify a browser name on instantiation.  Note that this
        will not work for Aqua browsers if the user has moved the application
        package after installation.

        If no browser is specified, the default browser, as specified in the
        Internet System Preferences panel, will be used.
        """
        def __init__(self, name):
            self.name = name

        def open(self, url, new=0, autoraise=True):
            sys.audit("webbrowser.open", url)
            self._check_url(url)
            assert "'" not in url
            # hack for local urls
            if not ':' in url:
                url = 'file:'+url

            # new must be 0 or 1
            new = int(bool(new))
            if self.name == "default":
                # User called open, open_new or get without a browser parameter
                script = 'open location "%s"' % url.replace('"', '%22') # opens in default browser
            else:
                # User called get and chose a browser
                if self.name == "OmniWeb":
                    toWindow = ""
                else:
                    # Include toWindow parameter of OpenURL command for browsers
                    # that support it.  0 == new window; -1 == existing
                    toWindow = "toWindow %d" % (new - 1)
                cmd = 'OpenURL "%s"' % url.replace('"', '%22')
                script = '''tell application "%s"
                                activate
                                %s %s
                            end tell''' % (self.name, cmd, toWindow)
            # Open pipe to AppleScript through osascript command
            osapipe = os.popen("osascript", "w")
            if osapipe is None:
                return False
            # Write script to osascript's stdin
            osapipe.write(script)
            rc = osapipe.close()
            return not rc

    class MacOSXOSAScript(BaseBrowser):
        def __init__(self, name):
            self._name = name

        def open(self, url, new=0, autoraise=True):
            if self._name == 'default':
                script = 'open location "%s"' % url.replace('"', '%22') # opens in default browser
            else:
                script = '''
                   tell application "%s"
                       activate
                       open location "%s"
                   end
                   '''%(self._name, url.replace('"', '%22'))

            osapipe = os.popen("osascript", "w")
            if osapipe is None:
                return False

            osapipe.write(script)
            rc = osapipe.close()
            return not rc


def main():
    import getopt
    usage = """Usage: %s [-n | -t] url
    -n: open new window
    -t: open new tab""" % sys.argv[0]
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'ntd')
    except getopt.error as msg:
        print(msg, file=sys.stderr)
        print(usage, file=sys.stderr)
        sys.exit(1)
    new_win = 0
    for o, a in opts:
        if o == '-n': new_win = 1
        elif o == '-t': new_win = 2
    if len(args) != 1:
        print(usage, file=sys.stderr)
        sys.exit(1)

    url = args[0]
    open(url, new_win)

    print("\a")

if __name__ == "__main__":
    main()
<|endoftext|>"""Thread module emulating a subset of Java's threading model."""

import os as _os
import sys as _sys
import _thread
import functools

from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
try:
    from _collections import deque as _deque
except ImportError:
    from collections import deque as _deque

# Note regarding PEP 8 compliant names
#  This threading model was originally inspired by Java, and inherited
# the convention of camelCase function and method names from that
# language. Those original names are not in any imminent danger of
# being deprecated (even for Py3k),so this module provides them as an
# alias for the PEP 8 compliant names
# Note that using the new PEP 8 compliant names facilitates substitution
# with the multiprocessing module, which doesn't provide the old
# Java inspired names.

__all__ = ['get_ident', 'active_count', 'Condition', 'current_thread',
           'enumerate', 'main_thread', 'TIMEOUT_MAX',
           'Event', 'Lock', 'RLock', 'Semaphore', 'BoundedSemaphore', 'Thread',
           'Barrier', 'BrokenBarrierError', 'Timer', 'ThreadError',
           'setprofile', 'settrace', 'local', 'stack_size',
           'excepthook', 'ExceptHookArgs', 'gettrace', 'getprofile']

# Rename some stuff so "from threading import *" is safe
_start_new_thread = _thread.start_new_thread
_allocate_lock = _thread.allocate_lock
_set_sentinel = _thread._set_sentinel
get_ident = _thread.get_ident
try:
    get_native_id = _thread.get_native_id
    _HAVE_THREAD_NATIVE_ID = True
    __all__.append('get_native_id')
except AttributeError:
    _HAVE_THREAD_NATIVE_ID = False
ThreadError = _thread.error
try:
    _CRLock = _thread.RLock
except AttributeError:
    _CRLock = None
TIMEOUT_MAX = _thread.TIMEOUT_MAX
del _thread


# Support for profile and trace hooks

_profile_hook = None
_trace_hook = None

def setprofile(func):
    """Set a profile function for all threads started from the threading module.

    The func will be passed to sys.setprofile() for each thread, before its
    run() method is called.

    """
    global _profile_hook
    _profile_hook = func

def getprofile():
    """Get the profiler function as set by threading.setprofile()."""
    return _profile_hook

def settrace(func):
    """Set a trace function for all threads started from the threading module.

    The func will be passed to sys.settrace() for each thread, before its run()
    method is called.

    """
    global _trace_hook
    _trace_hook = func

def gettrace():
    """Get the trace function as set by threading.settrace()."""
    return _trace_hook

# Synchronization classes

Lock = _allocate_lock

def RLock(*args, **kwargs):
    """Factory function that returns a new reentrant lock.

    A reentrant lock must be released by the thread that acquired it. Once a
    thread has acquired a reentrant lock, the same thread may acquire it again
    without blocking; the thread must release it once for each time it has
    acquired it.

    """
    if _CRLock is None:
        return _PyRLock(*args, **kwargs)
    return _CRLock(*args, **kwargs)

class _RLock:
    """This class implements reentrant lock objects.

    A reentrant lock must be released by the thread that acquired it. Once a
    thread has acquired a reentrant lock, the same thread may acquire it
    again without blocking; the thread must release it once for each time it
    has acquired it.

    """

    def __init__(self):
        self._block = _allocate_lock()
        self._owner = None
        self._count = 0

    def __repr__(self):
        owner = self._owner
        try:
            owner = _active[owner].name
        except KeyError:
            pass
        return "<%s %s.%s object owner=%r count=%d at %s>" % (
            "locked" if self._block.locked() else "unlocked",
            self.__class__.__module__,
            self.__class__.__qualname__,
            owner,
            self._count,
            hex(id(self))
        )

    def _at_fork_reinit(self):
        self._block._at_fork_reinit()
        self._owner = None
        self._count = 0

    def acquire(self, blocking=True, timeout=-1):
        """Acquire a lock, blocking or non-blocking.

        When invoked without arguments: if this thread already owns the lock,
        increment the recursion level by one, and return immediately. Otherwise,
        if another thread owns the lock, block until the lock is unlocked. Once
        the lock is unlocked (not owned by any thread), then grab ownership, set
        the recursion level to one, and return. If more than one thread is
        blocked waiting until the lock is unlocked, only one at a time will be
        able to grab ownership of the lock. There is no return value in this
        case.

        When invoked with the blocking argument set to true, do the same thing
        as when called without arguments, and return true.

        When invoked with the blocking argument set to false, do not block. If a
        call without an argument would block, return false immediately;
        otherwise, do the same thing as when called without arguments, and
        return true.

        When invoked with the floating-point timeout argument set to a positive
        value, block for at most the number of seconds specified by timeout
        and as long as the lock cannot be acquired.  Return true if the lock has
        been acquired, false if the timeout has elapsed.

        """
        me = get_ident()
        if self._owner == me:
            self._count += 1
            return 1
        rc = self._block.acquire(blocking, timeout)
        if rc:
            self._owner = me
            self._count = 1
        return rc

    __enter__ = acquire

    def release(self):
        """Release a lock, decrementing the recursion level.

        If after the decrement it is zero, reset the lock to unlocked (not owned
        by any thread), and if any other threads are blocked waiting for the
        lock to become unlocked, allow exactly one of them to proceed. If after
        the decrement the recursion level is still nonzero, the lock remains
        locked and owned by the calling thread.

        Only call this method when the calling thread owns the lock. A
        RuntimeError is raised if this method is called when the lock is
        unlocked.

        There is no return value.

        """
        if self._owner != get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._count = count = self._count - 1
        if not count:
            self._owner = None
            self._block.release()

    def __exit__(self, t, v, tb):
        self.release()

    # Internal methods used by condition variables

    def _acquire_restore(self, state):
        self._block.acquire()
        self._count, self._owner = state

    def _release_save(self):
        if self._count == 0:
            raise RuntimeError("cannot release un-acquired lock")
        count = self._count
        self._count = 0
        owner = self._owner
        self._owner = None
        self._block.release()
        return (count, owner)

    def _is_owned(self):
        return self._owner == get_ident()

_PyRLock = _RLock


class Condition:
    """Class that implements a condition variable.

    A condition variable allows one or more threads to wait until they are
    notified by another thread.

    If the lock argument is given and not None, it must be a Lock or RLock
    object, and it is used as the underlying lock. Otherwise, a new RLock object
    is created and used as the underlying lock.

    """

    def __init__(self, lock=None):
        if lock is None:
            lock = RLock()
        self._lock = lock
        # Export the lock's acquire() and release() methods
        self.acquire = lock.acquire
        self.release = lock.release
        # If the lock defines _release_save() and/or _acquire_restore(),
        # these override the default implementations (which just call
        # release() and acquire() on the lock).  Ditto for _is_owned().
        try:
            self._release_save = lock._release_save
        except AttributeError:
            pass
        try:
            self._acquire_restore = lock._acquire_restore
        except AttributeError:
            pass
        try:
            self._is_owned = lock._is_owned
        except AttributeError:
            pass
        self._waiters = _deque()

    def _at_fork_reinit(self):
        self._lock._at_fork_reinit()
        self._waiters.clear()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def __repr__(self):
        return "<Condition(%s, %d)>" % (self._lock, len(self._waiters))

    def _release_save(self):
        self._lock.release()           # No state to save

    def _acquire_restore(self, x):
        self._lock.acquire()           # Ignore saved state

    def _is_owned(self):
        # Return True if lock is owned by current_thread.
        # This method is called only if _lock doesn't have _is_owned().
        if self._lock.acquire(False):
            self._lock.release()
            return False
        else:
            return True

    def wait(self, timeout=None):
        """Wait until notified or until a timeout occurs.

        If the calling thread has not acquired the lock when this method is
        called, a RuntimeError is raised.

        This method releases the underlying lock, and then blocks until it is
        awakened by a notify() or notify_all() call for the same condition
        variable in another thread, or until the optional timeout occurs. Once
        awakened or timed out, it re-acquires the lock and returns.

        When the timeout argument is present and not None, it should be a
        floating point number specifying a timeout for the operation in seconds
        (or fractions thereof).

        When the underlying lock is an RLock, it is not released using its
        release() method, since this may not actually unlock the lock when it
        was acquired multiple times recursively. Instead, an internal interface
        of the RLock class is used, which really unlocks it even when it has
        been recursively acquired several times. Another internal interface is
        then used to restore the recursion level when the lock is reacquired.

        """
        if not self._is_owned():
            raise RuntimeError("cannot wait on un-acquired lock")
        waiter = _allocate_lock()
        waiter.acquire()
        self._waiters.append(waiter)
        saved_state = self._release_save()
        gotit = False
        try:    # restore state no matter what (e.g., KeyboardInterrupt)
            if timeout is None:
                waiter.acquire()
                gotit = True
            else:
                if timeout > 0:
                    gotit = waiter.acquire(True, timeout)
                else:
                    gotit = waiter.acquire(False)
            return gotit
        finally:
            self._acquire_restore(saved_state)
            if not gotit:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass

    def wait_for(self, predicate, timeout=None):
        """Wait until a condition evaluates to True.

        predicate should be a callable which result will be interpreted as a
        boolean value.  A timeout may be provided giving the maximum time to
        wait.

        """
        endtime = None
        waittime = timeout
        result = predicate()
        while not result:
            if waittime is not None:
                if endtime is None:
                    endtime = _time() + waittime
                else:
                    waittime = endtime - _time()
                    if waittime <= 0:
                        break
            self.wait(waittime)
            result = predicate()
        return result

    def notify(self, n=1):
        """Wake up one or more threads waiting on this condition, if any.

        If the calling thread has not acquired the lock when this method is
        called, a RuntimeError is raised.

        This method wakes up at most n of the threads waiting for the condition
        variable; it is a no-op if no threads are waiting.

        """
        if not self._is_owned():
            raise RuntimeError("cannot notify on un-acquired lock")
        waiters = self._waiters
        while waiters and n > 0:
            waiter = waiters[0]
            try:
                waiter.release()
            except RuntimeError:
                # gh-92530: The previous call of notify() released the lock,
                # but was interrupted before removing it from the queue.
                # It can happen if a signal handler raises an exception,
                # like CTRL+C which raises KeyboardInterrupt.
                pass
            else:
                n -= 1
            try:
                waiters.remove(waiter)
            except ValueError:
                pass

    def notify_all(self):
        """Wake up all threads waiting on this condition.

        If the calling thread has not acquired the lock when this method
        is called, a RuntimeError is raised.

        """
        self.notify(len(self._waiters))

    def notifyAll(self):
        """Wake up all threads waiting on this condition.

        This method is deprecated, use notify_all() instead.

        """
        import warnings
        warnings.warn('notifyAll() is deprecated, use notify_all() instead',
                      DeprecationWarning, stacklevel=2)
        self.notify_all()


class Semaphore:
    """This class implements semaphore objects.

    Semaphores manage a counter representing the number of release() calls minus
    the number of acquire() calls, plus an initial value. The acquire() method
    blocks if necessary until it can return without making the counter
    negative. If not given, value defaults to 1.

    """

    # After Tim Peters' semaphore class, but not quite the same (no maximum)

    def __init__(self, value=1):
        if value < 0:
            raise ValueError("semaphore initial value must be >= 0")
        self._cond = Condition(Lock())
        self._value = value

    def acquire(self, blocking=True, timeout=None):
        """Acquire a semaphore, decrementing the internal counter by one.

        When invoked without arguments: if the internal counter is larger than
        zero on entry, decrement it by one and return immediately. If it is zero
        on entry, block, waiting until some other thread has called release() to
        make it larger than zero. This is done with proper interlocking so that
        if multiple acquire() calls are blocked, release() will wake exactly one
        of them up. The implementation may pick one at random, so the order in
        which blocked threads are awakened should not be relied on. There is no
        return value in this case.

        When invoked with blocking set to true, do the same thing as when called
        without arguments, and return true.

        When invoked with blocking set to false, do not block. If a call without
        an argument would block, return false immediately; otherwise, do the
        same thing as when called without arguments, and return true.

        When invoked with a timeout other than None, it will block for at
        most timeout seconds.  If acquire does not complete successfully in
        that interval, return false.  Return true otherwise.

        """
        if not blocking and timeout is not None:
            raise ValueError("can't specify timeout for non-blocking acquire")
        rc = False
        endtime = None
        with self._cond:
            while self._value == 0:
                if not blocking:
                    break
                if timeout is not None:
                    if endtime is None:
                        endtime = _time() + timeout
                    else:
                        timeout = endtime - _time()
                        if timeout <= 0:
                            break
                self._cond.wait(timeout)
            else:
                self._value -= 1
                rc = True
        return rc

    __enter__ = acquire

    def release(self, n=1):
        """Release a semaphore, incrementing the internal counter by one or more.

        When the counter is zero on entry and another thread is waiting for it
        to become larger than zero again, wake up that thread.

        """
        if n < 1:
            raise ValueError('n must be one or more')
        with self._cond:
            self._value += n
            for i in range(n):
                self._cond.notify()

    def __exit__(self, t, v, tb):
        self.release()


class BoundedSemaphore(Semaphore):
    """Implements a bounded semaphore.

    A bounded semaphore checks to make sure its current value doesn't exceed its
    initial value. If it does, ValueError is raised. In most situations
    semaphores are used to guard resources with limited capacity.

    If the semaphore is released too many times it's a sign of a bug. If not
    given, value defaults to 1.

    Like regular semaphores, bounded semaphores manage a counter representing
    the number of release() calls minus the number of acquire() calls, plus an
    initial value. The acquire() method blocks if necessary until it can return
    without making the counter negative. If not given, value defaults to 1.

    """

    def __init__(self, value=1):
        Semaphore.__init__(self, value)
        self._initial_value = value

    def release(self, n=1):
        """Release a semaphore, incrementing the internal counter by one or more.

        When the counter is zero on entry and another thread is waiting for it
        to become larger than zero again, wake up that thread.

        If the number of releases exceeds the number of acquires,
        raise a ValueError.

        """
        if n < 1:
            raise ValueError('n must be one or more')
        with self._cond:
            if self._value + n > self._initial_value:
                raise ValueError("Semaphore released too many times")
            self._value += n
            for i in range(n):
                self._cond.notify()


class Event:
    """Class implementing event objects.

    Events manage a flag that can be set to true with the set() method and reset
    to false with the clear() method. The wait() method blocks until the flag is
    true.  The flag is initially false.

    """

    # After Tim Peters' event class (without is_posted())

    def __init__(self):
        self._cond = Condition(Lock())
        self._flag = False

    def _at_fork_reinit(self):
        # Private method called by Thread._reset_internal_locks()
        self._cond._at_fork_reinit()

    def is_set(self):
        """Return true if and only if the internal flag is true."""
        return self._flag

    def isSet(self):
        """Return true if and only if the internal flag is true.

        This method is deprecated, use is_set() instead.

        """
        import warnings
        warnings.warn('isSet() is deprecated, use is_set() instead',
                      DeprecationWarning, stacklevel=2)
        return self.is_set()

    def set(self):
        """Set the internal flag to true.

        All threads waiting for it to become true are awakened. Threads
        that call wait() once the flag is true will not block at all.

        """
        with self._cond:
            self._flag = True
            self._cond.notify_all()

    def clear(self):
        """Reset the internal flag to false.

        Subsequently, threads calling wait() will block until set() is called to
        set the internal flag to true again.

        """
        with self._cond:
            self._flag = False

    def wait(self, timeout=None):
        """Block until the internal flag is true.

        If the internal flag is true on entry, return immediately. Otherwise,
        block until another thread calls set() to set the flag to true, or until
        the optional timeout occurs.

        When the timeout argument is present and not None, it should be a
        floating point number specifying a timeout for the operation in seconds
        (or fractions thereof).

        This method returns the internal flag on exit, so it will always return
        True except if a timeout is given and the operation times out.

        """
        with self._cond:
            signaled = self._flag
            if not signaled:
                signaled = self._cond.wait(timeout)
            return signaled


# A barrier class.  Inspired in part by the pthread_barrier_* api and
# the CyclicBarrier class from Java.  See
# http://sourceware.org/pthreads-win32/manual/pthread_barrier_init.html and
# http://java.sun.com/j2se/1.5.0/docs/api/java/util/concurrent/
#        CyclicBarrier.html
# for information.
# We maintain two main states, 'filling' and 'draining' enabling the barrier
# to be cyclic.  Threads are not allowed into it until it has fully drained
# since the previous cycle.  In addition, a 'resetting' state exists which is
# similar to 'draining' except that threads leave with a BrokenBarrierError,
# and a 'broken' state in which all threads get the exception.
class Barrier:
    """Implements a Barrier.

    Useful for synchronizing a fixed number of threads at known synchronization
    points.  Threads block on 'wait()' and are simultaneously awoken once they
    have all made that call.

    """

    def __init__(self, parties, action=None, timeout=None):
        """Create a barrier, initialised to 'parties' threads.

        'action' is a callable which, when supplied, will be called by one of
        the threads after they have all entered the barrier and just prior to
        releasing them all. If a 'timeout' is provided, it is used as the
        default for all subsequent 'wait()' calls.

        """
        self._cond = Condition(Lock())
        self._action = action
        self._timeout = timeout
        self._parties = parties
        self._state = 0  # 0 filling, 1 draining, -1 resetting, -2 broken
        self._count = 0

    def wait(self, timeout=None):
        """Wait for the barrier.

        When the specified number of threads have started waiting, they are all
        simultaneously awoken. If an 'action' was provided for the barrier, one
        of the threads will have executed that callback prior to returning.
        Returns an individual index number from 0 to 'parties-1'.

        """
        if timeout is None:
            timeout = self._timeout
        with self._cond:
            self._enter() # Block while the barrier drains.
            index = self._count
            self._count += 1
            try:
                if index + 1 == self._parties:
                    # We release the barrier
                    self._release()
                else:
                    # We wait until someone releases us
                    self._wait(timeout)
                return index
            finally:
                self._count -= 1
                # Wake up any threads waiting for barrier to drain.
                self._exit()

    # Block until the barrier is ready for us, or raise an exception
    # if it is broken.
    def _enter(self):
        while self._state in (-1, 1):
            # It is draining or resetting, wait until done
            self._cond.wait()
        #see if the barrier is in a broken state
        if self._state < 0:
            raise BrokenBarrierError
        assert self._state == 0

    # Optionally run the 'action' and release the threads waiting
    # in the barrier.
    def _release(self):
        try:
            if self._action:
                self._action()
            # enter draining state
            self._state = 1
            self._cond.notify_all()
        except:
            #an exception during the _action handler.  Break and reraise
            self._break()
            raise

    # Wait in the barrier until we are released.  Raise an exception
    # if the barrier is reset or broken.
    def _wait(self, timeout):
        if not self._cond.wait_for(lambda : self._state != 0, timeout):
            #timed out.  Break the barrier
            self._break()
            raise BrokenBarrierError
        if self._state < 0:
            raise BrokenBarrierError
        assert self._state == 1

    # If we are the last thread to exit the barrier, signal any threads
    # waiting for the barrier to drain.
    def _exit(self):
        if self._count == 0:
            if self._state in (-1, 1):
                #resetting or draining
                self._state = 0
                self._cond.notify_all()

    def reset(self):
        """Reset the barrier to the initial state.

        Any threads currently waiting will get the BrokenBarrier exception
        raised.

        """
        with self._cond:
            if self._count > 0:
                if self._state == 0:
                    #reset the barrier, waking up threads
                    self._state = -1
                elif self._state == -2:
                    #was broken, set it to reset state
                    #which clears when the last thread exits
                    self._state = -1
            else:
                self._state = 0
            self._cond.notify_all()

    def abort(self):
        """Place the barrier into a 'broken' state.

        Useful in case of error.  Any currently waiting threads and threads
        attempting to 'wait()' will have BrokenBarrierError raised.

        """
        with self._cond:
            self._break()

    def _break(self):
        # An internal error was detected.  The barrier is set to
        # a broken state all parties awakened.
        self._state = -2
        self._cond.notify_all()

    @property
    def parties(self):
        """Return the number of threads required to trip the barrier."""
        return self._parties

    @property
    def n_waiting(self):
        """Return the number of threads currently waiting at the barrier."""
        # We don't need synchronization here since this is an ephemeral result
        # anyway.  It returns the correct value in the steady state.
        if self._state == 0:
            return self._count
        return 0

    @property
    def broken(self):
        """Return True if the barrier is in a broken state."""
        return self._state == -2

# exception raised by the Barrier class
class BrokenBarrierError(RuntimeError):
    pass


# Helper to generate new thread names
_counter = _count(1).__next__
def _newname(name_template):
    return name_template % _counter()

# Active thread administration.
#
# bpo-44422: Use a reentrant lock to allow reentrant calls to functions like
# threading.enumerate().
_active_limbo_lock = RLock()
_active = {}    # maps thread id to Thread object
_limbo = {}
_dangling = WeakSet()

# Set of Thread._tstate_lock locks of non-daemon threads used by _shutdown()
# to wait until all Python thread states get deleted:
# see Thread._set_tstate_lock().
_shutdown_locks_lock = _allocate_lock()
_shutdown_locks = set()

def _maintain_shutdown_locks():
    """
    Drop any shutdown locks that don't correspond to running threads anymore.

    Calling this from time to time avoids an ever-growing _shutdown_locks
    set when Thread objects are not joined explicitly. See bpo-37788.

    This must be called with _shutdown_locks_lock acquired.
    """
    # If a lock was released, the corresponding thread has exited
    to_remove = [lock for lock in _shutdown_locks if not lock.locked()]
    _shutdown_locks.difference_update(to_remove)


# Main class for threads

class Thread:
    """A class that represents a thread of control.

    This class can be safely subclassed in a limited fashion. There are two ways
    to specify the activity: by passing a callable object to the constructor, or
    by overriding the run() method in a subclass.

    """

    _initialized = False

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None):
        """This constructor should always be called with keyword arguments. Arguments are:

        *group* should be None; reserved for future extension when a ThreadGroup
        class is implemented.

        *target* is the callable object to be invoked by the run()
        method. Defaults to None, meaning nothing is called.

        *name* is the thread name. By default, a unique name is constructed of
        the form "Thread-N" where N is a small decimal number.

        *args* is the argument tuple for the target invocation. Defaults to ().

        *kwargs* is a dictionary of keyword arguments for the target
        invocation. Defaults to {}.

        If a subclass overrides the constructor, it must make sure to invoke
        the base class constructor (Thread.__init__()) before doing anything
        else to the thread.

        """
        assert group is None, "group argument must be None for now"
        if kwargs is None:
            kwargs = {}
        if name:
            name = str(name)
        else:
            name = _newname("Thread-%d")
            if target is not None:
                try:
                    target_name = target.__name__
                    name += f" ({target_name})"
                except AttributeError:
                    pass

        self._target = target
        self._name = name
        self._args = args
        self._kwargs = kwargs
        if daemon is not None:
            self._daemonic = daemon
        else:
            self._daemonic = current_thread().daemon
        self._ident = None
        if _HAVE_THREAD_NATIVE_ID:
            self._native_id = None
        self._tstate_lock = None
        self._started = Event()
        self._is_stopped = False
        self._initialized = True
        # Copy of sys.stderr used by self._invoke_excepthook()
        self._stderr = _sys.stderr
        self._invoke_excepthook = _make_invoke_excepthook()
        # For debugging and _after_fork()
        _dangling.add(self)

    def _reset_internal_locks(self, is_alive):
        # private!  Called by _after_fork() to reset our internal locks as
        # they may be in an invalid state leading to a deadlock or crash.
        self._started._at_fork_reinit()
        if is_alive:
            # bpo-42350: If the fork happens when the thread is already stopped
            # (ex: after threading._shutdown() has been called), _tstate_lock
            # is None. Do nothing in this case.
            if self._tstate_lock is not None:
                self._tstate_lock._at_fork_reinit()
                self._tstate_lock.acquire()
        else:
            # The thread isn't alive after fork: it doesn't have a tstate
            # anymore.
            self._is_stopped = True
            self._tstate_lock = None

    def __repr__(self):
        assert self._initialized, "Thread.__init__() was not called"
        status = "initial"
        if self._started.is_set():
            status = "started"
        self.is_alive() # easy way to get ._is_stopped set when appropriate
        if self._is_stopped:
            status = "stopped"
        if self._daemonic:
            status += " daemon"
        if self._ident is not None:
            status += " %s" % self._ident
        return "<%s(%s, %s)>" % (self.__class__.__name__, self._name, status)

    def start(self):
        """Start the thread's activity.

        It must be called at most once per thread object. It arranges for the
        object's run() method to be invoked in a separate thread of control.

        This method will raise a RuntimeError if called more than once on the
        same thread object.

        """
        if not self._initialized:
            raise RuntimeError("thread.__init__() not called")

        if self._started.is_set():
            raise RuntimeError("threads can only be started once")

        with _active_limbo_lock:
            _limbo[self] = self
        try:
            _start_new_thread(self._bootstrap, ())
        except Exception:
            with _active_limbo_lock:
                del _limbo[self]
            raise
        self._started.wait()

    def run(self):
        """Method representing the thread's activity.

        You may override this method in a subclass. The standard run() method
        invokes the callable object passed to the object's constructor as the
        target argument, if any, with sequential and keyword arguments taken
        from the args and kwargs arguments, respectively.

        """
        try:
            if self._target is not None:
                self._target(*self._args, **self._kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def _bootstrap(self):
        # Wrapper around the real bootstrap code that ignores
        # exceptions during interpreter cleanup.  Those typically
        # happen when a daemon thread wakes up at an unfortunate
        # moment, finds the world around it destroyed, and raises some
        # random exception *** while trying to report the exception in
        # _bootstrap_inner() below ***.  Those random exceptions
        # don't help anybody, and they confuse users, so we suppress
        # them.  We suppress them only when it appears that the world
        # indeed has already been destroyed, so that exceptions in
        # _bootstrap_inner() during normal business hours are properly
        # reported.  Also, we only suppress them for daemonic threads;
        # if a non-daemonic encounters this, something else is wrong.
        try:
            self._bootstrap_inner()
        except:
            if self._daemonic and _sys is None:
                return
            raise

    def _set_ident(self):
        self._ident = get_ident()

    if _HAVE_THREAD_NATIVE_ID:
        def _set_native_id(self):
            self._native_id = get_native_id()

    def _set_tstate_lock(self):
        """
        Set a lock object which will be released by the interpreter when
        the underlying thread state (see pystate.h) gets deleted.
        """
        self._tstate_lock = _set_sentinel()
        self._tstate_lock.acquire()

        if not self.daemon:
            with _shutdown_locks_lock:
                _maintain_shutdown_locks()
                _shutdown_locks.add(self._tstate_lock)

    def _bootstrap_inner(self):
        try:
            self._set_ident()
            self._set_tstate_lock()
            if _HAVE_THREAD_NATIVE_ID:
                self._set_native_id()
            self._started.set()
            with _active_limbo_lock:
                _active[self._ident] = self
                del _limbo[self]

            if _trace_hook:
                _sys.settrace(_trace_hook)
            if _profile_hook:
                _sys.setprofile(_profile_hook)

            try:
                self.run()
            except:
                self._invoke_excepthook(self)
        finally:
            with _active_limbo_lock:
                try:
                    # We don't call self._delete() because it also
                    # grabs _active_limbo_lock.
                    del _active[get_ident()]
                except:
                    pass

    def _stop(self):
        # After calling ._stop(), .is_alive() returns False and .join() returns
        # immediately.  ._tstate_lock must be released before calling ._stop().
        #
        # Normal case:  C code at the end of the thread's life
        # (release_sentinel in _threadmodule.c) releases ._tstate_lock, and
        # that's detected by our ._wait_for_tstate_lock(), called by .join()
        # and .is_alive().  Any number of threads _may_ call ._stop()
        # simultaneously (for example, if multiple threads are blocked in
        # .join() calls), and they're not serialized.  That's harmless -
        # they'll just make redundant rebindings of ._is_stopped and
        # ._tstate_lock.  Obscure:  we rebind ._tstate_lock last so that the
        # "assert self._is_stopped" in ._wait_for_tstate_lock() always works
        # (the assert is executed only if ._tstate_lock is None).
        #
        # Special case:  _main_thread releases ._tstate_lock via this
        # module's _shutdown() function.
        lock = self._tstate_lock
        if lock is not None:
            assert not lock.locked()
        self._is_stopped = True
        self._tstate_lock = None
        if not self.daemon:
            with _shutdown_locks_lock:
                # Remove our lock and other released locks from _shutdown_locks
                _maintain_shutdown_locks()

    def _delete(self):
        "Remove current thread from the dict of currently running threads."
        with _active_limbo_lock:
            del _active[get_ident()]
            # There must not be any python code between the previous line
            # and after the lock is released.  Otherwise a tracing function
            # could try to acquire the lock again in the same thread, (in
            # current_thread()), and would block.

    def join(self, timeout=None):
        """Wait until the thread terminates.

        This blocks the calling thread until the thread whose join() method is
        called terminates -- either normally or through an unhandled exception
        or until the optional timeout occurs.

        When the timeout argument is present and not None, it should be a
        floating point number specifying a timeout for the operation in seconds
        (or fractions thereof). As join() always returns None, you must call
        is_alive() after join() to decide whether a timeout happened -- if the
        thread is still alive, the join() call timed out.

        When the timeout argument is not present or None, the operation will
        block until the thread terminates.

        A thread can be join()ed many times.

        join() raises a RuntimeError if an attempt is made to join the current
        thread as that would cause a deadlock. It is also an error to join() a
        thread before it has been started and attempts to do so raises the same
        exception.

        """
        if not self._initialized:
            raise RuntimeError("Thread.__init__() not called")
        if not self._started.is_set():
            raise RuntimeError("cannot join thread before it is started")
        if self is current_thread():
            raise RuntimeError("cannot join current thread")

        if timeout is None:
            self._wait_for_tstate_lock()
        else:
            # the behavior of a negative timeout isn't documented, but
            # historically .join(timeout=x) for x<0 has acted as if timeout=0
            self._wait_for_tstate_lock(timeout=max(timeout, 0))

    def _wait_for_tstate_lock(self, block=True, timeout=-1):
        # Issue #18808: wait for the thread state to be gone.
        # At the end of the thread's life, after all knowledge of the thread
        # is removed from C data structures, C code releases our _tstate_lock.
        # This method passes its arguments to _tstate_lock.acquire().
        # If the lock is acquired, the C code is done, and self._stop() is
        # called.  That sets ._is_stopped to True, and ._tstate_lock to None.
        lock = self._tstate_lock
        if lock is None:
            # already determined that the C code is done
            assert self._is_stopped
            return

        try:
            if lock.acquire(block, timeout):
                lock.release()
                self._stop()
        except:
            if lock.locked():
                # bpo-45274: lock.acquire() acquired the lock, but the function
                # was interrupted with an exception before reaching the
                # lock.release(). It can happen if a signal handler raises an
                # exception, like CTRL+C which raises KeyboardInterrupt.
                lock.release()
                self._stop()
            raise

    @property
    def name(self):
        """A string used for identification purposes only.

        It has no semantics. Multiple threads may be given the same name. The
        initial name is set by the constructor.

        """
        assert self._initialized, "Thread.__init__() not called"
        return self._name

    @name.setter
    def name(self, name):
        assert self._initialized, "Thread.__init__() not called"
        self._name = str(name)

    @property
    def ident(self):
        """Thread identifier of this thread or None if it has not been started.

        This is a nonzero integer. See the get_ident() function. Thread
        identifiers may be recycled when a thread exits and another thread is
        created. The identifier is available even after the thread has exited.

        """
        assert self._initialized, "Thread.__init__() not called"
        return self._ident

    if _HAVE_THREAD_NATIVE_ID:
        @property
        def native_id(self):
            """Native integral thread ID of this thread, or None if it has not been started.

            This is a non-negative integer. See the get_native_id() function.
            This represents the Thread ID as reported by the kernel.

            """
            assert self._initialized, "Thread.__init__() not called"
            return self._native_id

    def is_alive(self):
        """Return whether the thread is alive.

        This method returns True just before the run() method starts until just
        after the run() method terminates. See also the module function
        enumerate().

        """
        assert self._initialized, "Thread.__init__() not called"
        if self._is_stopped or not self._started.is_set():
            return False
        self._wait_for_tstate_lock(False)
        return not self._is_stopped

    @property
    def daemon(self):
        """A boolean value indicating whether this thread is a daemon thread.

        This must be set before start() is called, otherwise RuntimeError is
        raised. Its initial value is inherited from the creating thread; the
        main thread is not a daemon thread and therefore all threads created in
        the main thread default to daemon = False.

        The entire Python program exits when only daemon threads are left.

        """
        assert self._initialized, "Thread.__init__() not called"
        return self._daemonic

    @daemon.setter
    def daemon(self, daemonic):
        if not self._initialized:
            raise RuntimeError("Thread.__init__() not called")
        if self._started.is_set():
            raise RuntimeError("cannot set daemon status of active thread")
        self._daemonic = daemonic

    def isDaemon(self):
        """Return whether this thread is a daemon.

        This method is deprecated, use the daemon attribute instead.

        """
        import warnings
        warnings.warn('isDaemon() is deprecated, get the daemon attribute instead',
                      DeprecationWarning, stacklevel=2)
        return self.daemon

    def setDaemon(self, daemonic):
        """Set whether this thread is a daemon.

        This method is deprecated, use the .daemon property instead.

        """
        import warnings
        warnings.warn('setDaemon() is deprecated, set the daemon attribute instead',
                      DeprecationWarning, stacklevel=2)
        self.daemon = daemonic

    def getName(self):
        """Return a string used for identification purposes only.

        This method is deprecated, use the name attribute instead.

        """
        import warnings
        warnings.warn('getName() is deprecated, get the name attribute instead',
                      DeprecationWarning, stacklevel=2)
        return self.name

    def setName(self, name):
        """Set the name string for this thread.

        This method is deprecated, use the name attribute instead.

        """
        import warnings
        warnings.warn('setName() is deprecated, set the name attribute instead',
                      DeprecationWarning, stacklevel=2)
        self.name = name


try:
    from _thread import (_excepthook as excepthook,
                         _ExceptHookArgs as ExceptHookArgs)
except ImportError:
    # Simple Python implementation if _thread._excepthook() is not available
    from traceback import print_exception as _print_exception
    from collections import namedtuple

    _ExceptHookArgs = namedtuple(
        'ExceptHookArgs',
        'exc_type exc_value exc_traceback thread')

    def ExceptHookArgs(args):
        return _ExceptHookArgs(*args)

    def excepthook(args, /):
        """
        Handle uncaught Thread.run() exception.
        """
        if args.exc_type == SystemExit:
            # silently ignore SystemExit
            return

        if _sys is not None and _sys.stderr is not None:
            stderr = _sys.stderr
        elif args.thread is not None:
            stderr = args.thread._stderr
            if stderr is None:
                # do nothing if sys.stderr is None and sys.stderr was None
                # when the thread was created
                return
        else:
            # do nothing if sys.stderr is None and args.thread is None
            return

        if args.thread is not None:
            name = args.thread.name
        else:
            name = get_ident()
        print(f"Exception in thread {name}:",
              file=stderr, flush=True)
        _print_exception(args.exc_type, args.exc_value, args.exc_traceback,
                         file=stderr)
        stderr.flush()


# Original value of threading.excepthook
__excepthook__ = excepthook


def _make_invoke_excepthook():
    # Create a local namespace to ensure that variables remain alive
    # when _invoke_excepthook() is called, even if it is called late during
    # Python shutdown. It is mostly needed for daemon threads.

    old_excepthook = excepthook
    old_sys_excepthook = _sys.excepthook
    if old_excepthook is None:
        raise RuntimeError("threading.excepthook is None")
    if old_sys_excepthook is None:
        raise RuntimeError("sys.excepthook is None")

    sys_exc_info = _sys.exc_info
    local_print = print
    local_sys = _sys

    def invoke_excepthook(thread):
        global excepthook
        try:
            hook = excepthook
            if hook is None:
                hook = old_excepthook

            args = ExceptHookArgs([*sys_exc_info(), thread])

            hook(args)
        except Exception as exc:
            exc.__suppress_context__ = True
            del exc

            if local_sys is not None and local_sys.stderr is not None:
                stderr = local_sys.stderr
            else:
                stderr = thread._stderr

            local_print("Exception in threading.excepthook:",
                        file=stderr, flush=True)

            if local_sys is not None and local_sys.excepthook is not None:
                sys_excepthook = local_sys.excepthook
            else:
                sys_excepthook = old_sys_excepthook

            sys_excepthook(*sys_exc_info())
        finally:
            # Break reference cycle (exception stored in a variable)
            args = None

    return invoke_excepthook


# The timer class was contributed by Itamar Shtull-Trauring

class Timer(Thread):
    """Call a function after a specified number of seconds:

            t = Timer(30.0, f, args=None, kwargs=None)
            t.start()
            t.cancel()     # stop the timer's action if it's still waiting

    """

    def __init__(self, interval, function, args=None, kwargs=None):
        Thread.__init__(self)
        self.interval = interval
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.finished = Event()

    def cancel(self):
        """Stop the timer if it hasn't finished yet."""
        self.finished.set()

    def run(self):
        self.finished.wait(self.interval)
        if not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
        self.finished.set()


# Special thread class to represent the main thread

class _MainThread(Thread):

    def __init__(self):
        Thread.__init__(self, name="MainThread", daemon=False)
        self._set_tstate_lock()
        self._started.set()
        self._set_ident()
        if _HAVE_THREAD_NATIVE_ID:
            self._set_native_id()
        with _active_limbo_lock:
            _active[self._ident] = self


# Dummy thread class to represent threads not started here.
# These aren't garbage collected when they die, nor can they be waited for.
# If they invoke anything in threading.py that calls current_thread(), they
# leave an entry in the _active dict forever after.
# Their purpose is to return *something* from current_thread().
# They are marked as daemon threads so we won't wait for them
# when we exit (conform previous semantics).

class _DummyThread(Thread):

    def __init__(self):
        Thread.__init__(self, name=_newname("Dummy-%d"), daemon=True)

        self._started.set()
        self._set_ident()
        if _HAVE_THREAD_NATIVE_ID:
            self._set_native_id()
        with _active_limbo_lock:
            _active[self._ident] = self

    def _stop(self):
        pass

    def is_alive(self):
        assert not self._is_stopped and self._started.is_set()
        return True

    def join(self, timeout=None):
        assert False, "cannot join a dummy thread"


# Global API functions

def current_thread():
    """Return the current Thread object, corresponding to the caller's thread of control.

    If the caller's thread of control was not created through the threading
    module, a dummy thread object with limited functionality is returned.

    """
    try:
        return _active[get_ident()]
    except KeyError:
        return _DummyThread()

def currentThread():
    """Return the current Thread object, corresponding to the caller's thread of control.

    This function is deprecated, use current_thread() instead.

    """
    import warnings
    warnings.warn('currentThread() is deprecated, use current_thread() instead',
                  DeprecationWarning, stacklevel=2)
    return current_thread()

def active_count():
    """Return the number of Thread objects currently alive.

    The returned count is equal to the length of the list returned by
    enumerate().

    """
    with _active_limbo_lock:
        return len(_active) + len(_limbo)

def activeCount():
    """Return the number of Thread objects currently alive.

    This function is deprecated, use active_count() instead.

    """
    import warnings
    warnings.warn('activeCount() is deprecated, use active_count() instead',
                  DeprecationWarning, stacklevel=2)
    return active_count()

def _enumerate():
    # Same as enumerate(), but without the lock. Internal use only.
    return list(_active.values()) + list(_limbo.values())

def enumerate():
    """Return a list of all Thread objects currently alive.

    The list includes daemonic threads, dummy thread objects created by
    current_thread(), and the main thread. It excludes terminated threads and
    threads that have not yet been started.

    """
    with _active_limbo_lock:
        return list(_active.values()) + list(_limbo.values())


_threading_atexits = []
_SHUTTING_DOWN = False

def _register_atexit(func, *arg, **kwargs):
    """CPython internal: register *func* to be called before joining threads.

    The registered *func* is called with its arguments just before all
    non-daemon threads are joined in `_shutdown()`. It provides a similar
    purpose to `atexit.register()`, but its functions are called prior to
    threading shutdown instead of interpreter shutdown.

    For similarity to atexit, the registered functions are called in reverse.
    """
    if _SHUTTING_DOWN:
        raise RuntimeError("can't register atexit after shutdown")

    call = functools.partial(func, *arg, **kwargs)
    _threading_atexits.append(call)


from _thread import stack_size

# Create the main thread object,
# and make it available for the interpreter
# (Py_Main) as threading._shutdown.

_main_thread = _MainThread()

def _shutdown():
    """
    Wait until the Python thread state of all non-daemon threads get deleted.
    """
    # Obscure:  other threads may be waiting to join _main_thread.  That's
    # dubious, but some code does it.  We can't wait for C code to release
    # the main thread's tstate_lock - that won't happen until the interpreter
    # is nearly dead.  So we release it here.  Note that just calling _stop()
    # isn't enough:  other threads may already be waiting on _tstate_lock.
    if _main_thread._is_stopped:
        # _shutdown() was already called
        return

    global _SHUTTING_DOWN
    _SHUTTING_DOWN = True

    # Call registered threading atexit functions before threads are joined.
    # Order is reversed, similar to atexit.
    for atexit_call in reversed(_threading_atexits):
        atexit_call()

    # Main thread
    if _main_thread.ident == get_ident():
        tlock = _main_thread._tstate_lock
        # The main thread isn't finished yet, so its thread state lock can't
        # have been released.
        assert tlock is not None
        assert tlock.locked()
        tlock.release()
        _main_thread._stop()
    else:
        # bpo-1596321: _shutdown() must be called in the main thread.
        # If the threading module was not imported by the main thread,
        # _main_thread is the thread which imported the threading module.
        # In this case, ignore _main_thread, similar behavior than for threads
        # spawned by C libraries or using _thread.start_new_thread().
        pass

    # Join all non-deamon threads
    while True:
        with _shutdown_locks_lock:
            locks = list(_shutdown_locks)
            _shutdown_locks.clear()

        if not locks:
            break

        for lock in locks:
            # mimic Thread.join()
            lock.acquire()
            lock.release()

        # new threads can be spawned while we were waiting for the other
        # threads to complete


def main_thread():
    """Return the main thread object.

    In normal conditions, the main thread is the thread from which the
    Python interpreter was started.
    """
    return _main_thread

# get thread-local implementation, either from the thread
# module, or from the python fallback

try:
    from _thread import _local as local
except ImportError:
    from _threading_local import local


def _after_fork():
    """
    Cleanup threading module state that should not exist after a fork.
    """
    # Reset _active_limbo_lock, in case we forked while the lock was held
    # by another (non-forked) thread.  http://bugs.python.org/issue874900
    global _active_limbo_lock, _main_thread
    global _shutdown_locks_lock, _shutdown_locks
    _active_limbo_lock = RLock()

    # fork() only copied the current thread; clear references to others.
    new_active = {}

    try:
        current = _active[get_ident()]
    except KeyError:
        # fork() was called in a thread which was not spawned
        # by threading.Thread. For example, a thread spawned
        # by thread.start_new_thread().
        current = _MainThread()

    _main_thread = current

    # reset _shutdown() locks: threads re-register their _tstate_lock below
    _shutdown_locks_lock = _allocate_lock()
    _shutdown_locks = set()

    with _active_limbo_lock:
        # Dangling thread instances must still have their locks reset,
        # because someone may join() them.
        threads = set(_enumerate())
        threads.update(_dangling)
        for thread in threads:
            # Any lock/condition variable may be currently locked or in an
            # invalid state, so we reinitialize them.
            if thread is current:
                # There is only one active thread. We reset the ident to
                # its new value since it can have changed.
                thread._reset_internal_locks(True)
                ident = get_ident()
                thread._ident = ident
                new_active[ident] = thread
            else:
                # All the others are already stopped.
                thread._reset_internal_locks(False)
                thread._stop()

        _limbo.clear()
        _active.clear()
        _active.update(new_active)
        assert len(_active) == 1


if hasattr(_os, "register_at_fork"):
    _os.register_at_fork(after_in_child=_after_fork)
<|endoftext|>"""A generic class to build line-oriented command interpreters.

Interpreters constructed with this class obey the following conventions:

1. End of file on input is processed as the command 'EOF'.
2. A command is parsed out of each line by collecting the prefix composed
   of characters in the identchars member.
3. A command `foo' is dispatched to a method 'do_foo()'; the do_ method
   is passed a single argument consisting of the remainder of the line.
4. Typing an empty line repeats the last command.  (Actually, it calls the
   method `emptyline', which may be overridden in a subclass.)
5. There is a predefined `help' method.  Given an argument `topic', it
   calls the command `help_topic'.  With no arguments, it lists all topics
   with defined help_ functions, broken into up to three topics; documented
   commands, miscellaneous help topics, and undocumented commands.
6. The command '?' is a synonym for `help'.  The command '!' is a synonym
   for `shell', if a do_shell method exists.
7. If completion is enabled, completing commands will be done automatically,
   and completing of commands args is done by calling complete_foo() with
   arguments text, line, begidx, endidx.  text is string we are matching
   against, all returned matches must begin with it.  line is the current
   input line (lstripped), begidx and endidx are the beginning and end
   indexes of the text being matched, which could be used to provide
   different completion depending upon which position the argument is in.

The `default' method may be overridden to intercept commands for which there
is no do_ method.

The `completedefault' method may be overridden to intercept completions for
commands that have no complete_ method.

The data member `self.ruler' sets the character used to draw separator lines
in the help messages.  If empty, no ruler line is drawn.  It defaults to "=".

If the value of `self.intro' is nonempty when the cmdloop method is called,
it is printed out on interpreter startup.  This value may be overridden
via an optional argument to the cmdloop() method.

The data members `self.doc_header', `self.misc_header', and
`self.undoc_header' set the headers used for the help function's
listings of documented functions, miscellaneous topics, and undocumented
functions respectively.
"""

import string, sys

__all__ = ["Cmd"]

PROMPT = '(Cmd) '
IDENTCHARS = string.ascii_letters + string.digits + '_'

class Cmd:
    """A simple framework for writing line-oriented command interpreters.

    These are often useful for test harnesses, administrative tools, and
    prototypes that will later be wrapped in a more sophisticated interface.

    A Cmd instance or subclass instance is a line-oriented interpreter
    framework.  There is no good reason to instantiate Cmd itself; rather,
    it's useful as a superclass of an interpreter class you define yourself
    in order to inherit Cmd's methods and encapsulate action methods.

    """
    prompt = PROMPT
    identchars = IDENTCHARS
    ruler = '='
    lastcmd = ''
    intro = None
    doc_leader = ""
    doc_header = "Documented commands (type help <topic>):"
    misc_header = "Miscellaneous help topics:"
    undoc_header = "Undocumented commands:"
    nohelp = "*** No help on %s"
    use_rawinput = 1

    def __init__(self, completekey='tab', stdin=None, stdout=None):
        """Instantiate a line-oriented interpreter framework.

        The optional argument 'completekey' is the readline name of a
        completion key; it defaults to the Tab key. If completekey is
        not None and the readline module is available, command completion
        is done automatically. The optional arguments stdin and stdout
        specify alternate input and output file objects; if not specified,
        sys.stdin and sys.stdout are used.

        """
        if stdin is not None:
            self.stdin = stdin
        else:
            self.stdin = sys.stdin
        if stdout is not None:
            self.stdout = stdout
        else:
            self.stdout = sys.stdout
        self.cmdqueue = []
        self.completekey = completekey

    def cmdloop(self, intro=None):
        """Repeatedly issue a prompt, accept input, parse an initial prefix
        off the received input, and dispatch to action methods, passing them
        the remainder of the line as argument.

        """

        self.preloop()
        if self.use_rawinput and self.completekey:
            try:
                import readline
                self.old_completer = readline.get_completer()
                readline.set_completer(self.complete)
                readline.parse_and_bind(self.completekey+": complete")
            except ImportError:
                pass
        try:
            if intro is not None:
                self.intro = intro
            if self.intro:
                self.stdout.write(str(self.intro)+"\n")
            stop = None
            while not stop:
                if self.cmdqueue:
                    line = self.cmdqueue.pop(0)
                else:
                    if self.use_rawinput:
                        try:
                            line = input(self.prompt)
                        except EOFError:
                            line = 'EOF'
                    else:
                        self.stdout.write(self.prompt)
                        self.stdout.flush()
                        line = self.stdin.readline()
                        if not len(line):
                            line = 'EOF'
                        else:
                            line = line.rstrip('\r\n')
                line = self.precmd(line)
                stop = self.onecmd(line)
                stop = self.postcmd(stop, line)
            self.postloop()
        finally:
            if self.use_rawinput and self.completekey:
                try:
                    import readline
                    readline.set_completer(self.old_completer)
                except ImportError:
                    pass


    def precmd(self, line):
        """Hook method executed just before the command line is
        interpreted, but after the input prompt is generated and issued.

        """
        return line

    def postcmd(self, stop, line):
        """Hook method executed just after a command dispatch is finished."""
        return stop

    def preloop(self):
        """Hook method executed once when the cmdloop() method is called."""
        pass

    def postloop(self):
        """Hook method executed once when the cmdloop() method is about to
        return.

        """
        pass

    def parseline(self, line):
        """Parse the line into a command name and a string containing
        the arguments.  Returns a tuple containing (command, args, line).
        'command' and 'args' may be None if the line couldn't be parsed.
        """
        line = line.strip()
        if not line:
            return None, None, line
        elif line[0] == '?':
            line = 'help ' + line[1:]
        elif line[0] == '!':
            if hasattr(self, 'do_shell'):
                line = 'shell ' + line[1:]
            else:
                return None, None, line
        i, n = 0, len(line)
        while i < n and line[i] in self.identchars: i = i+1
        cmd, arg = line[:i], line[i:].strip()
        return cmd, arg, line

    def onecmd(self, line):
        """Interpret the argument as though it had been typed in response
        to the prompt.

        This may be overridden, but should not normally need to be;
        see the precmd() and postcmd() methods for useful execution hooks.
        The return value is a flag indicating whether interpretation of
        commands by the interpreter should stop.

        """
        cmd, arg, line = self.parseline(line)
        if not line:
            return self.emptyline()
        if cmd is None:
            return self.default(line)
        self.lastcmd = line
        if line == 'EOF' :
            self.lastcmd = ''
        if cmd == '':
            return self.default(line)
        else:
            try:
                func = getattr(self, 'do_' + cmd)
            except AttributeError:
                return self.default(line)
            return func(arg)

    def emptyline(self):
        """Called when an empty line is entered in response to the prompt.

        If this method is not overridden, it repeats the last nonempty
        command entered.

        """
        if self.lastcmd:
            return self.onecmd(self.lastcmd)

    def default(self, line):
        """Called on an input line when the command prefix is not recognized.

        If this method is not overridden, it prints an error message and
        returns.

        """
        self.stdout.write('*** Unknown syntax: %s\n'%line)

    def completedefault(self, *ignored):
        """Method called to complete an input line when no command-specific
        complete_*() method is available.

        By default, it returns an empty list.

        """
        return []

    def completenames(self, text, *ignored):
        dotext = 'do_'+text
        return [a[3:] for a in self.get_names() if a.startswith(dotext)]

    def complete(self, text, state):
        """Return the next possible completion for 'text'.

        If a command has not been entered, then complete against command list.
        Otherwise try to call complete_<command> to get list of completions.
        """
        if state == 0:
            import readline
            origline = readline.get_line_buffer()
            line = origline.lstrip()
            stripped = len(origline) - len(line)
            begidx = readline.get_begidx() - stripped
            endidx = readline.get_endidx() - stripped
            if begidx>0:
                cmd, args, foo = self.parseline(line)
                if cmd == '':
                    compfunc = self.completedefault
                else:
                    try:
                        compfunc = getattr(self, 'complete_' + cmd)
                    except AttributeError:
                        compfunc = self.completedefault
            else:
                compfunc = self.completenames
            self.completion_matches = compfunc(text, line, begidx, endidx)
        try:
            return self.completion_matches[state]
        except IndexError:
            return None

    def get_names(self):
        # This method used to pull in base class attributes
        # at a time dir() didn't do it yet.
        return dir(self.__class__)

    def complete_help(self, *args):
        commands = set(self.completenames(*args))
        topics = set(a[5:] for a in self.get_names()
                     if a.startswith('help_' + args[0]))
        return list(commands | topics)

    def do_help(self, arg):
        'List available commands with "help" or detailed help with "help cmd".'
        if arg:
            # XXX check arg syntax
            try:
                func = getattr(self, 'help_' + arg)
            except AttributeError:
                try:
                    doc=getattr(self, 'do_' + arg).__doc__
                    if doc:
                        self.stdout.write("%s\n"%str(doc))
                        return
                except AttributeError:
                    pass
                self.stdout.write("%s\n"%str(self.nohelp % (arg,)))
                return
            func()
        else:
            names = self.get_names()
            cmds_doc = []
            cmds_undoc = []
            help = {}
            for name in names:
                if name[:5] == 'help_':
                    help[name[5:]]=1
            names.sort()
            # There can be duplicates if routines overridden
            prevname = ''
            for name in names:
                if name[:3] == 'do_':
                    if name == prevname:
                        continue
                    prevname = name
                    cmd=name[3:]
                    if cmd in help:
                        cmds_doc.append(cmd)
                        del help[cmd]
                    elif getattr(self, name).__doc__:
                        cmds_doc.append(cmd)
                    else:
                        cmds_undoc.append(cmd)
            self.stdout.write("%s\n"%str(self.doc_leader))
            self.print_topics(self.doc_header,   cmds_doc,   15,80)
            self.print_topics(self.misc_header,  list(help.keys()),15,80)
            self.print_topics(self.undoc_header, cmds_undoc, 15,80)

    def print_topics(self, header, cmds, cmdlen, maxcol):
        if cmds:
            self.stdout.write("%s\n"%str(header))
            if self.ruler:
                self.stdout.write("%s\n"%str(self.ruler * len(header)))
            self.columnize(cmds, maxcol-1)
            self.stdout.write("\n")

    def columnize(self, list, displaywidth=80):
        """Display a list of strings as a compact set of columns.

        Each column is only as wide as necessary.
        Columns are separated by two spaces (one was not legible enough).
        """
        if not list:
            self.stdout.write("<empty>\n")
            return

        nonstrings = [i for i in range(len(list))
                        if not isinstance(list[i], str)]
        if nonstrings:
            raise TypeError("list[i] not a string for i in %s"
                            % ", ".join(map(str, nonstrings)))
        size = len(list)
        if size == 1:
            self.stdout.write('%s\n'%str(list[0]))
            return
        # Try every row count from 1 upwards
        for nrows in range(1, len(list)):
            ncols = (size+nrows-1) // nrows
            colwidths = []
            totwidth = -2
            for col in range(ncols):
                colwidth = 0
                for row in range(nrows):
                    i = row + nrows*col
                    if i >= size:
                        break
                    x = list[i]
                    colwidth = max(colwidth, len(x))
                colwidths.append(colwidth)
                totwidth += colwidth + 2
                if totwidth > displaywidth:
                    break
            if totwidth <= displaywidth:
                break
        else:
            nrows = len(list)
            ncols = 1
            colwidths = [0]
        for row in range(nrows):
            texts = []
            for col in range(ncols):
                i = row + nrows*col
                if i >= size:
                    x = ""
                else:
                    x = list[i]
                texts.append(x)
            while texts and not texts[-1]:
                del texts[-1]
            for col in range(len(texts)):
                texts[col] = texts[col].ljust(colwidths[col])
            self.stdout.write("%s\n"%str("  ".join(texts)))
<|endoftext|>"""Filename matching with shell patterns.

fnmatch(FILENAME, PATTERN) matches according to the local convention.
fnmatchcase(FILENAME, PATTERN) always takes case in account.

The functions operate by translating the pattern into a regular
expression.  They cache the compiled regular expressions for speed.

The function translate(PATTERN) returns a regular expression
corresponding to PATTERN.  (It does not compile it.)
"""
import os
import posixpath
import re
import functools

__all__ = ["filter", "fnmatch", "fnmatchcase", "translate"]

# Build a thread-safe incrementing counter to help create unique regexp group
# names across calls.
from itertools import count
_nextgroupnum = count().__next__
del count

def fnmatch(name, pat):
    """Test whether FILENAME matches PATTERN.

    Patterns are Unix shell style:

    *       matches everything
    ?       matches any single character
    [seq]   matches any character in seq
    [!seq]  matches any char not in seq

    An initial period in FILENAME is not special.
    Both FILENAME and PATTERN are first case-normalized
    if the operating system requires it.
    If you don't want this, use fnmatchcase(FILENAME, PATTERN).
    """
    name = os.path.normcase(name)
    pat = os.path.normcase(pat)
    return fnmatchcase(name, pat)

@functools.lru_cache(maxsize=256, typed=True)
def _compile_pattern(pat):
    if isinstance(pat, bytes):
        pat_str = str(pat, 'ISO-8859-1')
        res_str = translate(pat_str)
        res = bytes(res_str, 'ISO-8859-1')
    else:
        res = translate(pat)
    return re.compile(res).match

def filter(names, pat):
    """Construct a list from those elements of the iterable NAMES that match PAT."""
    result = []
    pat = os.path.normcase(pat)
    match = _compile_pattern(pat)
    if os.path is posixpath:
        # normcase on posix is NOP. Optimize it away from the loop.
        for name in names:
            if match(name):
                result.append(name)
    else:
        for name in names:
            if match(os.path.normcase(name)):
                result.append(name)
    return result

def fnmatchcase(name, pat):
    """Test whether FILENAME matches PATTERN, including case.

    This is a version of fnmatch() which doesn't case-normalize
    its arguments.
    """
    match = _compile_pattern(pat)
    return match(name) is not None


def translate(pat):
    """Translate a shell PATTERN to a regular expression.

    There is no way to quote meta-characters.
    """

    STAR = object()
    res = []
    add = res.append
    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        i = i+1
        if c == '*':
            # compress consecutive `*` into one
            if (not res) or res[-1] is not STAR:
                add(STAR)
        elif c == '?':
            add('.')
        elif c == '[':
            j = i
            if j < n and pat[j] == '!':
                j = j+1
            if j < n and pat[j] == ']':
                j = j+1
            while j < n and pat[j] != ']':
                j = j+1
            if j >= n:
                add('\\[')
            else:
                stuff = pat[i:j]
                if '-' not in stuff:
                    stuff = stuff.replace('\\', r'\\')
                else:
                    chunks = []
                    k = i+2 if pat[i] == '!' else i+1
                    while True:
                        k = pat.find('-', k, j)
                        if k < 0:
                            break
                        chunks.append(pat[i:k])
                        i = k+1
                        k = k+3
                    chunk = pat[i:j]
                    if chunk:
                        chunks.append(chunk)
                    else:
                        chunks[-1] += '-'
                    # Remove empty ranges -- invalid in RE.
                    for k in range(len(chunks)-1, 0, -1):
                        if chunks[k-1][-1] > chunks[k][0]:
                            chunks[k-1] = chunks[k-1][:-1] + chunks[k][1:]
                            del chunks[k]
                    # Escape backslashes and hyphens for set difference (--).
                    # Hyphens that create ranges shouldn't be escaped.
                    stuff = '-'.join(s.replace('\\', r'\\').replace('-', r'\-')
                                     for s in chunks)
                # Escape set operations (&&, ~~ and ||).
                stuff = re.sub(r'([&~|])', r'\\\1', stuff)
                i = j+1
                if not stuff:
                    # Empty range: never match.
                    add('(?!)')
                elif stuff == '!':
                    # Negated empty range: match any character.
                    add('.')
                else:
                    if stuff[0] == '!':
                        stuff = '^' + stuff[1:]
                    elif stuff[0] in ('^', '['):
                        stuff = '\\' + stuff
                    add(f'[{stuff}]')
        else:
            add(re.escape(c))
    assert i == n

    # Deal with STARs.
    inp = res
    res = []
    add = res.append
    i, n = 0, len(inp)
    # Fixed pieces at the start?
    while i < n and inp[i] is not STAR:
        add(inp[i])
        i += 1
    # Now deal with STAR fixed STAR fixed ...
    # For an interior `STAR fixed` pairing, we want to do a minimal
    # .*? match followed by `fixed`, with no possibility of backtracking.
    # We can't spell that directly, but can trick it into working by matching
    #    .*?fixed
    # in a lookahead assertion, save the matched part in a group, then
    # consume that group via a backreference. If the overall match fails,
    # the lookahead assertion won't try alternatives. So the translation is:
    #     (?=(?P<name>.*?fixed))(?P=name)
    # Group names are created as needed: g0, g1, g2, ...
    # The numbers are obtained from _nextgroupnum() to ensure they're unique
    # across calls and across threads. This is because people rely on the
    # undocumented ability to join multiple translate() results together via
    # "|" to build large regexps matching "one of many" shell patterns.
    while i < n:
        assert inp[i] is STAR
        i += 1
        if i == n:
            add(".*")
            break
        assert inp[i] is not STAR
        fixed = []
        while i < n and inp[i] is not STAR:
            fixed.append(inp[i])
            i += 1
        fixed = "".join(fixed)
        if i == n:
            add(".*")
            add(fixed)
        else:
            groupnum = _nextgroupnum()
            add(f"(?=(?P<g{groupnum}>.*?{fixed}))(?P=g{groupnum})")
    assert i == n
    res = "".join(res)
    return fr'(?s:{res})\Z'
<|endoftext|>"""A powerful, extensible, and easy-to-use option parser.

By Greg Ward <gward@python.net>

Originally distributed as Optik.

For support, use the optik-users@lists.sourceforge.net mailing list
(http://lists.sourceforge.net/lists/listinfo/optik-users).

Simple usage example:

   from optparse import OptionParser

   parser = OptionParser()
   parser.add_option("-f", "--file", dest="filename",
                     help="write report to FILE", metavar="FILE")
   parser.add_option("-q", "--quiet",
                     action="store_false", dest="verbose", default=True,
                     help="don't print status messages to stdout")

   (options, args) = parser.parse_args()
"""

__version__ = "1.5.3"

__all__ = ['Option',
           'make_option',
           'SUPPRESS_HELP',
           'SUPPRESS_USAGE',
           'Values',
           'OptionContainer',
           'OptionGroup',
           'OptionParser',
           'HelpFormatter',
           'IndentedHelpFormatter',
           'TitledHelpFormatter',
           'OptParseError',
           'OptionError',
           'OptionConflictError',
           'OptionValueError',
           'BadOptionError',
           'check_choice']

__copyright__ = """
Copyright (c) 2001-2006 Gregory P. Ward.  All rights reserved.
Copyright (c) 2002-2006 Python Software Foundation.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

  * Neither the name of the author nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys, os
import textwrap

def _repr(self):
    return "<%s at 0x%x: %s>" % (self.__class__.__name__, id(self), self)


# This file was generated from:
#   Id: option_parser.py 527 2006-07-23 15:21:30Z greg
#   Id: option.py 522 2006-06-11 16:22:03Z gward
#   Id: help.py 527 2006-07-23 15:21:30Z greg
#   Id: errors.py 509 2006-04-20 00:58:24Z gward

try:
    from gettext import gettext, ngettext
except ImportError:
    def gettext(message):
        return message

    def ngettext(singular, plural, n):
        if n == 1:
            return singular
        return plural

_ = gettext


class OptParseError (Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class OptionError (OptParseError):
    """
    Raised if an Option instance is created with invalid or
    inconsistent arguments.
    """

    def __init__(self, msg, option):
        self.msg = msg
        self.option_id = str(option)

    def __str__(self):
        if self.option_id:
            return "option %s: %s" % (self.option_id, self.msg)
        else:
            return self.msg

class OptionConflictError (OptionError):
    """
    Raised if conflicting options are added to an OptionParser.
    """

class OptionValueError (OptParseError):
    """
    Raised if an invalid option value is encountered on the command
    line.
    """

class BadOptionError (OptParseError):
    """
    Raised if an invalid option is seen on the command line.
    """
    def __init__(self, opt_str):
        self.opt_str = opt_str

    def __str__(self):
        return _("no such option: %s") % self.opt_str

class AmbiguousOptionError (BadOptionError):
    """
    Raised if an ambiguous option is seen on the command line.
    """
    def __init__(self, opt_str, possibilities):
        BadOptionError.__init__(self, opt_str)
        self.possibilities = possibilities

    def __str__(self):
        return (_("ambiguous option: %s (%s?)")
                % (self.opt_str, ", ".join(self.possibilities)))


class HelpFormatter:

    """
    Abstract base class for formatting option help.  OptionParser
    instances should use one of the HelpFormatter subclasses for
    formatting help; by default IndentedHelpFormatter is used.

    Instance attributes:
      parser : OptionParser
        the controlling OptionParser instance
      indent_increment : int
        the number of columns to indent per nesting level
      max_help_position : int
        the maximum starting column for option help text
      help_position : int
        the calculated starting column for option help text;
        initially the same as the maximum
      width : int
        total number of columns for output (pass None to constructor for
        this value to be taken from the $COLUMNS environment variable)
      level : int
        current indentation level
      current_indent : int
        current indentation level (in columns)
      help_width : int
        number of columns available for option help text (calculated)
      default_tag : str
        text to replace with each option's default value, "%default"
        by default.  Set to false value to disable default value expansion.
      option_strings : { Option : str }
        maps Option instances to the snippet of help text explaining
        the syntax of that option, e.g. "-h, --help" or
        "-fFILE, --file=FILE"
      _short_opt_fmt : str
        format string controlling how short options with values are
        printed in help text.  Must be either "%s%s" ("-fFILE") or
        "%s %s" ("-f FILE"), because those are the two syntaxes that
        Optik supports.
      _long_opt_fmt : str
        similar but for long options; must be either "%s %s" ("--file FILE")
        or "%s=%s" ("--file=FILE").
    """

    NO_DEFAULT_VALUE = "none"

    def __init__(self,
                 indent_increment,
                 max_help_position,
                 width,
                 short_first):
        self.parser = None
        self.indent_increment = indent_increment
        if width is None:
            try:
                width = int(os.environ['COLUMNS'])
            except (KeyError, ValueError):
                width = 80
            width -= 2
        self.width = width
