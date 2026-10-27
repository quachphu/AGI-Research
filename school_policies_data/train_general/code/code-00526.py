            if not is_reload and name in sys.modules:
                module = sys.modules[name]
                try:
                    __spec__ = module.__spec__
                except AttributeError:
                    # We use the found spec since that is the one that
                    # we would have used if the parent module hadn't
                    # beaten us to the punch.
                    return spec
                else:
                    if __spec__ is None:
                        return spec
                    else:
                        return __spec__
            else:
                return spec
    else:
        return None


def _sanity_check(name, package, level):
    """Verify arguments are "sane"."""
    if not isinstance(name, str):
        raise TypeError('module name must be str, not {}'.format(type(name)))
    if level < 0:
        raise ValueError('level must be >= 0')
    if level > 0:
        if not isinstance(package, str):
            raise TypeError('__package__ not set to a string')
        elif not package:
            raise ImportError('attempted relative import with no known parent '
                              'package')
    if not name and level == 0:
        raise ValueError('Empty module name')


_ERR_MSG_PREFIX = 'No module named '
_ERR_MSG = _ERR_MSG_PREFIX + '{!r}'

def _find_and_load_unlocked(name, import_):
    path = None
    parent = name.rpartition('.')[0]
    if parent:
        if parent not in sys.modules:
            _call_with_frames_removed(import_, parent)
        # Crazy side-effects!
        if name in sys.modules:
            return sys.modules[name]
        parent_module = sys.modules[parent]
        try:
            path = parent_module.__path__
        except AttributeError:
            msg = (_ERR_MSG + '; {!r} is not a package').format(name, parent)
            raise ModuleNotFoundError(msg, name=name) from None
    spec = _find_spec(name, path)
    if spec is None:
        raise ModuleNotFoundError(_ERR_MSG.format(name), name=name)
    else:
        module = _load_unlocked(spec)
    if parent:
        # Set the module as an attribute on its parent.
        parent_module = sys.modules[parent]
        child = name.rpartition('.')[2]
        try:
            setattr(parent_module, child, module)
        except AttributeError:
            msg = f"Cannot set an attribute on {parent!r} for child module {child!r}"
            _warnings.warn(msg, ImportWarning)
    return module


_NEEDS_LOADING = object()


def _find_and_load(name, import_):
    """Find and load the module."""
    with _ModuleLockManager(name):
        module = sys.modules.get(name, _NEEDS_LOADING)
        if module is _NEEDS_LOADING:
            return _find_and_load_unlocked(name, import_)

    if module is None:
        message = ('import of {} halted; '
                   'None in sys.modules'.format(name))
        raise ModuleNotFoundError(message, name=name)

    _lock_unlock_module(name)
    return module


def _gcd_import(name, package=None, level=0):
    """Import and return the module based on its name, the package the call is
    being made from, and the level adjustment.

    This function represents the greatest common denominator of functionality
    between import_module and __import__. This includes setting __package__ if
    the loader did not.

    """
    _sanity_check(name, package, level)
    if level > 0:
        name = _resolve_name(name, package, level)
    return _find_and_load(name, _gcd_import)


def _handle_fromlist(module, fromlist, import_, *, recursive=False):
    """Figure out what __import__ should return.

    The import_ parameter is a callable which takes the name of module to
    import. It is required to decouple the function from assuming importlib's
    import implementation is desired.

    """
    # The hell that is fromlist ...
    # If a package was imported, try to import stuff from fromlist.
    for x in fromlist:
        if not isinstance(x, str):
            if recursive:
                where = module.__name__ + '.__all__'
            else:
                where = "``from list''"
            raise TypeError(f"Item in {where} must be str, "
                            f"not {type(x).__name__}")
        elif x == '*':
            if not recursive and hasattr(module, '__all__'):
                _handle_fromlist(module, module.__all__, import_,
                                 recursive=True)
        elif not hasattr(module, x):
            from_name = '{}.{}'.format(module.__name__, x)
            try:
                _call_with_frames_removed(import_, from_name)
            except ModuleNotFoundError as exc:
                # Backwards-compatibility dictates we ignore failed
                # imports triggered by fromlist for modules that don't
                # exist.
                if (exc.name == from_name and
                    sys.modules.get(from_name, _NEEDS_LOADING) is not None):
                    continue
                raise
    return module


def _calc___package__(globals):
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    """
    package = globals.get('__package__')
    spec = globals.get('__spec__')
    if package is not None:
        if spec is not None and package != spec.parent:
            _warnings.warn("__package__ != __spec__.parent "
                           f"({package!r} != {spec.parent!r})",
                           ImportWarning, stacklevel=3)
        return package
    elif spec is not None:
        return spec.parent
    else:
        _warnings.warn("can't resolve package from __spec__ or __package__, "
                       "falling back on __name__ and __path__",
                       ImportWarning, stacklevel=3)
        package = globals['__name__']
        if '__path__' not in globals:
            package = package.rpartition('.')[0]
    return package


def __import__(name, globals=None, locals=None, fromlist=(), level=0):
    """Import a module.

    The 'globals' argument is used to infer where the import is occurring from
    to handle relative imports. The 'locals' argument is ignored. The
    'fromlist' argument specifies what should exist as attributes on the module
    being imported (e.g. ``from module import <fromlist>``).  The 'level'
    argument represents the package location to import from in a relative
    import (e.g. ``from ..pkg import mod`` would have a 'level' of 2).

    """
    if level == 0:
        module = _gcd_import(name)
    else:
        globals_ = globals if globals is not None else {}
        package = _calc___package__(globals_)
        module = _gcd_import(name, package, level)
    if not fromlist:
        # Return up to the first dot in 'name'. This is complicated by the fact
        # that 'name' may be relative.
        if level == 0:
            return _gcd_import(name.partition('.')[0])
        elif not name:
            return module
        else:
            # Figure out where to slice the module's name up to the first dot
            # in 'name'.
            cut_off = len(name) - len(name.partition('.')[0])
            # Slice end needs to be positive to alleviate need to special-case
            # when ``'.' not in name``.
            return sys.modules[module.__name__[:len(module.__name__)-cut_off]]
    elif hasattr(module, '__path__'):
        return _handle_fromlist(module, fromlist, _gcd_import)
    else:
        return module


def _builtin_from_name(name):
    spec = BuiltinImporter.find_spec(name)
    if spec is None:
        raise ImportError('no built-in module named ' + name)
    return _load_unlocked(spec)


def _setup(sys_module, _imp_module):
    """Setup importlib by importing needed built-in modules and injecting them
    into the global namespace.

    As sys is needed for sys.modules access and _imp is needed to load built-in
    modules, those two modules must be explicitly passed in.

    """
    global _imp, sys
    _imp = _imp_module
    sys = sys_module

    # Set up the spec for existing builtin/frozen modules.
    module_type = type(sys)
    for name, module in sys.modules.items():
        if isinstance(module, module_type):
            if name in sys.builtin_module_names:
                loader = BuiltinImporter
            elif _imp.is_frozen(name):
                loader = FrozenImporter
            else:
                continue
            spec = _spec_from_module(module, loader)
            _init_module_attrs(spec, module)

    # Directly load built-in modules needed during bootstrap.
    self_module = sys.modules[__name__]
    for builtin_name in ('_thread', '_warnings', '_weakref'):
        if builtin_name not in sys.modules:
            builtin_module = _builtin_from_name(builtin_name)
        else:
            builtin_module = sys.modules[builtin_name]
        setattr(self_module, builtin_name, builtin_module)


def _install(sys_module, _imp_module):
    """Install importers for builtin and frozen modules"""
    _setup(sys_module, _imp_module)

    sys.meta_path.append(BuiltinImporter)
    sys.meta_path.append(FrozenImporter)


def _install_external_importers():
    """Install importers that require external filesystem access"""
    global _bootstrap_external
    import _frozen_importlib_external
    _bootstrap_external = _frozen_importlib_external
    _frozen_importlib_external._install(sys.modules[__name__])
<|endoftext|>"""Subset of importlib.abc used to reduce importlib.util imports."""
from . import _bootstrap
import abc
import warnings


class Loader(metaclass=abc.ABCMeta):

    """Abstract base class for import loaders."""

    def create_module(self, spec):
        """Return a module to initialize and into which to load.

        This method should raise ImportError if anything prevents it
        from creating a new module.  It may return None to indicate
        that the spec should create the new module.
        """
        # By default, defer to default semantics for the new module.
        return None

    # We don't define exec_module() here since that would break
    # hasattr checks we do to support backward compatibility.

    def load_module(self, fullname):
        """Return the loaded module.

        The module must be added to sys.modules and have import-related
        attributes set properly.  The fullname is a str.

        ImportError is raised on failure.

        This method is deprecated in favor of loader.exec_module(). If
        exec_module() exists then it is used to provide a backwards-compatible
        functionality for this method.

        """
        if not hasattr(self, 'exec_module'):
            raise ImportError
        # Warning implemented in _load_module_shim().
        return _bootstrap._load_module_shim(self, fullname)

    def module_repr(self, module):
        """Return a module's repr.

        Used by the module type when the method does not raise
        NotImplementedError.

        This method is deprecated.

        """
        warnings.warn("importlib.abc.Loader.module_repr() is deprecated and "
                      "slated for removal in Python 3.12", DeprecationWarning)
        # The exception will cause ModuleType.__repr__ to ignore this method.
        raise NotImplementedError
<|endoftext|>from contextlib import suppress

from . import abc


class SpecLoaderAdapter:
    """
    Adapt a package spec to adapt the underlying loader.
    """

    def __init__(self, spec, adapter=lambda spec: spec.loader):
        self.spec = spec
        self.loader = adapter(spec)

    def __getattr__(self, name):
        return getattr(self.spec, name)


class TraversableResourcesLoader:
    """
    Adapt a loader to provide TraversableResources.
    """

    def __init__(self, spec):
        self.spec = spec

    def get_resource_reader(self, name):
        return DegenerateFiles(self.spec)._native()


class DegenerateFiles:
    """
    Adapter for an existing or non-existant resource reader
    to provide a degenerate .files().
    """

    class Path(abc.Traversable):
        def iterdir(self):
            return iter(())

        def is_dir(self):
            return False

        is_file = exists = is_dir  # type: ignore

        def joinpath(self, other):
            return DegenerateFiles.Path()

        @property
        def name(self):
            return ''

        def open(self, mode='rb', *args, **kwargs):
            raise ValueError()

    def __init__(self, spec):
        self.spec = spec

    @property
    def _reader(self):
        with suppress(AttributeError):
            return self.spec.loader.get_resource_reader(self.spec.name)

    def _native(self):
        """
        Return the native reader if it supports files().
        """
        reader = self._reader
        return reader if hasattr(reader, 'files') else self

    def __getattr__(self, attr):
        return getattr(self._reader, attr)

    def files(self):
        return DegenerateFiles.Path()


def wrap_spec(package):
    """
    Construct a package spec with traversable compatibility
    on the spec/loader/reader.
    """
    return SpecLoaderAdapter(package.__spec__, TraversableResourcesLoader)
<|endoftext|>import os
import pathlib
import tempfile
import functools
import contextlib
import types
import importlib

from typing import Union, Any, Optional
from .abc import ResourceReader, Traversable

from ._adapters import wrap_spec

Package = Union[types.ModuleType, str]


def files(package):
    # type: (Package) -> Traversable
    """
    Get a Traversable resource from a package
    """
    return from_package(get_package(package))


def normalize_path(path):
    # type: (Any) -> str
    """Normalize a path by ensuring it is a string.

    If the resulting string contains path separators, an exception is raised.
    """
    str_path = str(path)
    parent, file_name = os.path.split(str_path)
    if parent:
        raise ValueError(f'{path!r} must be only a file name')
    return file_name


def get_resource_reader(package):
    # type: (types.ModuleType) -> Optional[ResourceReader]
    """
    Return the package's loader if it's a ResourceReader.
    """
    # We can't use
    # a issubclass() check here because apparently abc.'s __subclasscheck__()
    # hook wants to create a weak reference to the object, but
    # zipimport.zipimporter does not support weak references, resulting in a
    # TypeError.  That seems terrible.
    spec = package.__spec__
    reader = getattr(spec.loader, 'get_resource_reader', None)  # type: ignore
    if reader is None:
        return None
    return reader(spec.name)  # type: ignore


def resolve(cand):
    # type: (Package) -> types.ModuleType
    return cand if isinstance(cand, types.ModuleType) else importlib.import_module(cand)


def get_package(package):
    # type: (Package) -> types.ModuleType
    """Take a package name or module object and return the module.

    Raise an exception if the resolved module is not a package.
    """
    resolved = resolve(package)
    if wrap_spec(resolved).submodule_search_locations is None:
        raise TypeError(f'{package!r} is not a package')
    return resolved


def from_package(package):
    """
    Return a Traversable object for the given package.

    """
    spec = wrap_spec(package)
    reader = spec.loader.get_resource_reader(spec.name)
    return reader.files()


@contextlib.contextmanager
def _tempfile(reader, suffix='',
              # gh-93353: Keep a reference to call os.remove() in late Python
              # finalization.
              *, _os_remove=os.remove):
    # Not using tempfile.NamedTemporaryFile as it leads to deeper 'try'
    # blocks due to the need to close the temporary file to work on Windows
    # properly.
    fd, raw_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, reader())
        os.close(fd)
        del reader
        yield pathlib.Path(raw_path)
    finally:
        try:
            _os_remove(raw_path)
        except FileNotFoundError:
            pass


@functools.singledispatch
def as_file(path):
    """
    Given a Traversable object, return that object as a
    path on the local file system in a context manager.
    """
    return _tempfile(path.read_bytes, suffix=path.name)


@as_file.register(pathlib.Path)
@contextlib.contextmanager
def _(path):
    """
    Degenerate behavior for pathlib.Path objects.
    """
    yield path
<|endoftext|>"""Core implementation of path-based import.

This module is NOT meant to be directly imported! It has been designed such
that it can be bootstrapped into Python as the implementation of import. As
such it requires the injection of specific modules and attributes in order to
work. One should use importlib as the public-facing version of this module.

"""
# IMPORTANT: Whenever making changes to this module, be sure to run a top-level
# `make regen-importlib` followed by `make` in order to get the frozen version
# of the module updated. Not doing so will result in the Makefile to fail for
# all others who don't have a ./python around to freeze the module in the early
# stages of compilation.
#

# See importlib._setup() for what is injected into the global namespace.

# When editing this code be aware that code executed at import time CANNOT
# reference any injected objects! This includes not only global code but also
# anything specified at the class level.

# Module injected manually by _set_bootstrap_module()
_bootstrap = None

# Import builtin modules
import _imp
import _io
import sys
import _warnings
import marshal


_MS_WINDOWS = (sys.platform == 'win32')
if _MS_WINDOWS:
    import nt as _os
    import winreg
else:
    import posix as _os


if _MS_WINDOWS:
    path_separators = ['\\', '/']
else:
    path_separators = ['/']
# Assumption made in _path_join()
assert all(len(sep) == 1 for sep in path_separators)
path_sep = path_separators[0]
path_sep_tuple = tuple(path_separators)
path_separators = ''.join(path_separators)
_pathseps_with_colon = {f':{s}' for s in path_separators}


# Bootstrap-related code ######################################################
_CASE_INSENSITIVE_PLATFORMS_STR_KEY = 'win',
_CASE_INSENSITIVE_PLATFORMS_BYTES_KEY = 'cygwin', 'darwin'
_CASE_INSENSITIVE_PLATFORMS =  (_CASE_INSENSITIVE_PLATFORMS_BYTES_KEY
                                + _CASE_INSENSITIVE_PLATFORMS_STR_KEY)


def _make_relax_case():
    if sys.platform.startswith(_CASE_INSENSITIVE_PLATFORMS):
        if sys.platform.startswith(_CASE_INSENSITIVE_PLATFORMS_STR_KEY):
            key = 'PYTHONCASEOK'
        else:
            key = b'PYTHONCASEOK'

        def _relax_case():
            """True if filenames must be checked case-insensitively and ignore environment flags are not set."""
            return not sys.flags.ignore_environment and key in _os.environ
    else:
        def _relax_case():
            """True if filenames must be checked case-insensitively."""
            return False
    return _relax_case

_relax_case = _make_relax_case()


def _pack_uint32(x):
    """Convert a 32-bit integer to little-endian."""
    return (int(x) & 0xFFFFFFFF).to_bytes(4, 'little')


def _unpack_uint32(data):
    """Convert 4 bytes in little-endian to an integer."""
    assert len(data) == 4
    return int.from_bytes(data, 'little')

def _unpack_uint16(data):
    """Convert 2 bytes in little-endian to an integer."""
    assert len(data) == 2
    return int.from_bytes(data, 'little')


if _MS_WINDOWS:
    def _path_join(*path_parts):
        """Replacement for os.path.join()."""
        if not path_parts:
            return ""
        if len(path_parts) == 1:
            return path_parts[0]
        root = ""
        path = []
        for new_root, tail in map(_os._path_splitroot, path_parts):
            if new_root.startswith(path_sep_tuple) or new_root.endswith(path_sep_tuple):
                root = new_root.rstrip(path_separators) or root
                path = [path_sep + tail]
            elif new_root.endswith(':'):
                if root.casefold() != new_root.casefold():
                    # Drive relative paths have to be resolved by the OS, so we reset the
                    # tail but do not add a path_sep prefix.
                    root = new_root
                    path = [tail]
                else:
                    path.append(tail)
            else:
                root = new_root or root
                path.append(tail)
        path = [p.rstrip(path_separators) for p in path if p]
        if len(path) == 1 and not path[0]:
            # Avoid losing the root's trailing separator when joining with nothing
            return root + path_sep
        return root + path_sep.join(path)

else:
    def _path_join(*path_parts):
        """Replacement for os.path.join()."""
        return path_sep.join([part.rstrip(path_separators)
                              for part in path_parts if part])


def _path_split(path):
    """Replacement for os.path.split()."""
    i = max(path.rfind(p) for p in path_separators)
    if i < 0:
        return '', path
    return path[:i], path[i + 1:]


def _path_stat(path):
    """Stat the path.

    Made a separate function to make it easier to override in experiments
    (e.g. cache stat results).

    """
    return _os.stat(path)


def _path_is_mode_type(path, mode):
    """Test whether the path is the specified mode type."""
    try:
        stat_info = _path_stat(path)
    except OSError:
        return False
    return (stat_info.st_mode & 0o170000) == mode


def _path_isfile(path):
    """Replacement for os.path.isfile."""
    return _path_is_mode_type(path, 0o100000)


def _path_isdir(path):
    """Replacement for os.path.isdir."""
    if not path:
        path = _os.getcwd()
    return _path_is_mode_type(path, 0o040000)


if _MS_WINDOWS:
    def _path_isabs(path):
        """Replacement for os.path.isabs."""
        if not path:
            return False
        root = _os._path_splitroot(path)[0].replace('/', '\\')
        return len(root) > 1 and (root.startswith('\\\\') or root.endswith('\\'))

else:
    def _path_isabs(path):
        """Replacement for os.path.isabs."""
        return path.startswith(path_separators)


def _write_atomic(path, data, mode=0o666):
    """Best-effort function to write data to a path atomically.
    Be prepared to handle a FileExistsError if concurrent writing of the
    temporary file is attempted."""
    # id() is used to generate a pseudo-random filename.
    path_tmp = '{}.{}'.format(path, id(path))
    fd = _os.open(path_tmp,
                  _os.O_EXCL | _os.O_CREAT | _os.O_WRONLY, mode & 0o666)
    try:
        # We first write data to a temporary file, and then use os.replace() to
        # perform an atomic rename.
        with _io.FileIO(fd, 'wb') as file:
            file.write(data)
        _os.replace(path_tmp, path)
    except OSError:
        try:
            _os.unlink(path_tmp)
        except OSError:
            pass
        raise


_code_type = type(_write_atomic.__code__)


# Finder/loader utility code ###############################################

# Magic word to reject .pyc files generated by other Python versions.
# It should change for each incompatible change to the bytecode.
#
# The value of CR and LF is incorporated so if you ever read or write
# a .pyc file in text mode the magic number will be wrong; also, the
# Apple MPW compiler swaps their values, botching string constants.
#
# There were a variety of old schemes for setting the magic number.
# The current working scheme is to increment the previous value by
# 10.
#
# Starting with the adoption of PEP 3147 in Python 3.2, every bump in magic
# number also includes a new "magic tag", i.e. a human readable string used
# to represent the magic number in __pycache__ directories.  When you change
# the magic number, you must also set a new unique magic tag.  Generally this
# can be named after the Python major version of the magic number bump, but
# it can really be anything, as long as it's different than anything else
# that's come before.  The tags are included in the following table, starting
# with Python 3.2a0.
#
# Known values:
#  Python 1.5:   20121
#  Python 1.5.1: 20121
#     Python 1.5.2: 20121
#     Python 1.6:   50428
#     Python 2.0:   50823
#     Python 2.0.1: 50823
#     Python 2.1:   60202
#     Python 2.1.1: 60202
#     Python 2.1.2: 60202
#     Python 2.2:   60717
#     Python 2.3a0: 62011
#     Python 2.3a0: 62021
#     Python 2.3a0: 62011 (!)
#     Python 2.4a0: 62041
#     Python 2.4a3: 62051
#     Python 2.4b1: 62061
#     Python 2.5a0: 62071
#     Python 2.5a0: 62081 (ast-branch)
#     Python 2.5a0: 62091 (with)
#     Python 2.5a0: 62092 (changed WITH_CLEANUP opcode)
#     Python 2.5b3: 62101 (fix wrong code: for x, in ...)
#     Python 2.5b3: 62111 (fix wrong code: x += yield)
#     Python 2.5c1: 62121 (fix wrong lnotab with for loops and
#                          storing constants that should have been removed)
#     Python 2.5c2: 62131 (fix wrong code: for x, in ... in listcomp/genexp)
#     Python 2.6a0: 62151 (peephole optimizations and STORE_MAP opcode)
#     Python 2.6a1: 62161 (WITH_CLEANUP optimization)
#     Python 2.7a0: 62171 (optimize list comprehensions/change LIST_APPEND)
#     Python 2.7a0: 62181 (optimize conditional branches:
#                          introduce POP_JUMP_IF_FALSE and POP_JUMP_IF_TRUE)
#     Python 2.7a0  62191 (introduce SETUP_WITH)
#     Python 2.7a0  62201 (introduce BUILD_SET)
#     Python 2.7a0  62211 (introduce MAP_ADD and SET_ADD)
#     Python 3000:   3000
#                    3010 (removed UNARY_CONVERT)
#                    3020 (added BUILD_SET)
#                    3030 (added keyword-only parameters)
#                    3040 (added signature annotations)
#                    3050 (print becomes a function)
#                    3060 (PEP 3115 metaclass syntax)
#                    3061 (string literals become unicode)
#                    3071 (PEP 3109 raise changes)
#                    3081 (PEP 3137 make __file__ and __name__ unicode)
#                    3091 (kill str8 interning)
#                    3101 (merge from 2.6a0, see 62151)
#                    3103 (__file__ points to source file)
#     Python 3.0a4: 3111 (WITH_CLEANUP optimization).
#     Python 3.0b1: 3131 (lexical exception stacking, including POP_EXCEPT
                          #3021)
#     Python 3.1a1: 3141 (optimize list, set and dict comprehensions:
#                         change LIST_APPEND and SET_ADD, add MAP_ADD #2183)
#     Python 3.1a1: 3151 (optimize conditional branches:
#                         introduce POP_JUMP_IF_FALSE and POP_JUMP_IF_TRUE
                          #4715)
#     Python 3.2a1: 3160 (add SETUP_WITH #6101)
#                   tag: cpython-32
#     Python 3.2a2: 3170 (add DUP_TOP_TWO, remove DUP_TOPX and ROT_FOUR #9225)
#                   tag: cpython-32
#     Python 3.2a3  3180 (add DELETE_DEREF #4617)
#     Python 3.3a1  3190 (__class__ super closure changed)
#     Python 3.3a1  3200 (PEP 3155 __qualname__ added #13448)
#     Python 3.3a1  3210 (added size modulo 2**32 to the pyc header #13645)
#     Python 3.3a2  3220 (changed PEP 380 implementation #14230)
#     Python 3.3a4  3230 (revert changes to implicit __class__ closure #14857)
#     Python 3.4a1  3250 (evaluate positional default arguments before
#                        keyword-only defaults #16967)
#     Python 3.4a1  3260 (add LOAD_CLASSDEREF; allow locals of class to override
#                        free vars #17853)
#     Python 3.4a1  3270 (various tweaks to the __class__ closure #12370)
#     Python 3.4a1  3280 (remove implicit class argument)
#     Python 3.4a4  3290 (changes to __qualname__ computation #19301)
#     Python 3.4a4  3300 (more changes to __qualname__ computation #19301)
#     Python 3.4rc2 3310 (alter __qualname__ computation #20625)
#     Python 3.5a1  3320 (PEP 465: Matrix multiplication operator #21176)
#     Python 3.5b1  3330 (PEP 448: Additional Unpacking Generalizations #2292)
#     Python 3.5b2  3340 (fix dictionary display evaluation order #11205)
#     Python 3.5b3  3350 (add GET_YIELD_FROM_ITER opcode #24400)
#     Python 3.5.2  3351 (fix BUILD_MAP_UNPACK_WITH_CALL opcode #27286)
#     Python 3.6a0  3360 (add FORMAT_VALUE opcode #25483)
#     Python 3.6a1  3361 (lineno delta of code.co_lnotab becomes signed #26107)
#     Python 3.6a2  3370 (16 bit wordcode #26647)
#     Python 3.6a2  3371 (add BUILD_CONST_KEY_MAP opcode #27140)
#     Python 3.6a2  3372 (MAKE_FUNCTION simplification, remove MAKE_CLOSURE
#                         #27095)
#     Python 3.6b1  3373 (add BUILD_STRING opcode #27078)
#     Python 3.6b1  3375 (add SETUP_ANNOTATIONS and STORE_ANNOTATION opcodes
#                         #27985)
#     Python 3.6b1  3376 (simplify CALL_FUNCTIONs & BUILD_MAP_UNPACK_WITH_CALL
                          #27213)
#     Python 3.6b1  3377 (set __class__ cell from type.__new__ #23722)
#     Python 3.6b2  3378 (add BUILD_TUPLE_UNPACK_WITH_CALL #28257)
#     Python 3.6rc1 3379 (more thorough __class__ validation #23722)
#     Python 3.7a1  3390 (add LOAD_METHOD and CALL_METHOD opcodes #26110)
#     Python 3.7a2  3391 (update GET_AITER #31709)
#     Python 3.7a4  3392 (PEP 552: Deterministic pycs #31650)
#     Python 3.7b1  3393 (remove STORE_ANNOTATION opcode #32550)
#     Python 3.7b5  3394 (restored docstring as the first stmt in the body;
#                         this might affected the first line number #32911)
#     Python 3.8a1  3400 (move frame block handling to compiler #17611)
#     Python 3.8a1  3401 (add END_ASYNC_FOR #33041)
#     Python 3.8a1  3410 (PEP570 Python Positional-Only Parameters #36540)
#     Python 3.8b2  3411 (Reverse evaluation order of key: value in dict
#                         comprehensions #35224)
#     Python 3.8b2  3412 (Swap the position of positional args and positional
#                         only args in ast.arguments #37593)
#     Python 3.8b4  3413 (Fix "break" and "continue" in "finally" #37830)
#     Python 3.9a0  3420 (add LOAD_ASSERTION_ERROR #34880)
#     Python 3.9a0  3421 (simplified bytecode for with blocks #32949)
#     Python 3.9a0  3422 (remove BEGIN_FINALLY, END_FINALLY, CALL_FINALLY, POP_FINALLY bytecodes #33387)
#     Python 3.9a2  3423 (add IS_OP, CONTAINS_OP and JUMP_IF_NOT_EXC_MATCH bytecodes #39156)
#     Python 3.9a2  3424 (simplify bytecodes for *value unpacking)
#     Python 3.9a2  3425 (simplify bytecodes for **value unpacking)
#     Python 3.10a1 3430 (Make 'annotations' future by default)
#     Python 3.10a1 3431 (New line number table format -- PEP 626)
#     Python 3.10a2 3432 (Function annotation for MAKE_FUNCTION is changed from dict to tuple bpo-42202)
#     Python 3.10a2 3433 (RERAISE restores f_lasti if oparg != 0)
#     Python 3.10a6 3434 (PEP 634: Structural Pattern Matching)
#     Python 3.10a7 3435 Use instruction offsets (as opposed to byte offsets).
#     Python 3.10b1 3436 (Add GEN_START bytecode #43683)
#     Python 3.10b1 3437 (Undo making 'annotations' future by default - We like to dance among core devs!)
#     Python 3.10b1 3438 Safer line number table handling.
#     Python 3.10b1 3439 (Add ROT_N)

#
# MAGIC must change whenever the bytecode emitted by the compiler may no
# longer be understood by older implementations of the eval loop (usually
# due to the addition of new opcodes).
#
# Whenever MAGIC_NUMBER is changed, the ranges in the magic_values array
# in PC/launcher.c must also be updated.

MAGIC_NUMBER = (3439).to_bytes(2, 'little') + b'\r\n'
_RAW_MAGIC_NUMBER = int.from_bytes(MAGIC_NUMBER, 'little')  # For import.c

_PYCACHE = '__pycache__'
_OPT = 'opt-'

SOURCE_SUFFIXES = ['.py']
if _MS_WINDOWS:
    SOURCE_SUFFIXES.append('.pyw')

EXTENSION_SUFFIXES = _imp.extension_suffixes()

BYTECODE_SUFFIXES = ['.pyc']
# Deprecated.
DEBUG_BYTECODE_SUFFIXES = OPTIMIZED_BYTECODE_SUFFIXES = BYTECODE_SUFFIXES

def cache_from_source(path, debug_override=None, *, optimization=None):
    """Given the path to a .py file, return the path to its .pyc file.

    The .py file does not need to exist; this simply returns the path to the
    .pyc file calculated as if the .py file were imported.

    The 'optimization' parameter controls the presumed optimization level of
    the bytecode file. If 'optimization' is not None, the string representation
    of the argument is taken and verified to be alphanumeric (else ValueError
    is raised).

    The debug_override parameter is deprecated. If debug_override is not None,
    a True value is the same as setting 'optimization' to the empty string
    while a False value is equivalent to setting 'optimization' to '1'.

    If sys.implementation.cache_tag is None then NotImplementedError is raised.

    """
    if debug_override is not None:
        _warnings.warn('the debug_override parameter is deprecated; use '
                       "'optimization' instead", DeprecationWarning)
        if optimization is not None:
            message = 'debug_override or optimization must be set to None'
            raise TypeError(message)
        optimization = '' if debug_override else 1
    path = _os.fspath(path)
    head, tail = _path_split(path)
    base, sep, rest = tail.rpartition('.')
    tag = sys.implementation.cache_tag
    if tag is None:
        raise NotImplementedError('sys.implementation.cache_tag is None')
    almost_filename = ''.join([(base if base else rest), sep, tag])
    if optimization is None:
        if sys.flags.optimize == 0:
            optimization = ''
        else:
            optimization = sys.flags.optimize
    optimization = str(optimization)
    if optimization != '':
        if not optimization.isalnum():
            raise ValueError('{!r} is not alphanumeric'.format(optimization))
        almost_filename = '{}.{}{}'.format(almost_filename, _OPT, optimization)
    filename = almost_filename + BYTECODE_SUFFIXES[0]
    if sys.pycache_prefix is not None:
        # We need an absolute path to the py file to avoid the possibility of
        # collisions within sys.pycache_prefix, if someone has two different
        # `foo/bar.py` on their system and they import both of them using the
        # same sys.pycache_prefix. Let's say sys.pycache_prefix is
        # `C:\Bytecode`; the idea here is that if we get `Foo\Bar`, we first
        # make it absolute (`C:\Somewhere\Foo\Bar`), then make it root-relative
        # (`Somewhere\Foo\Bar`), so we end up placing the bytecode file in an
        # unambiguous `C:\Bytecode\Somewhere\Foo\Bar\`.
        if not _path_isabs(head):
            head = _path_join(_os.getcwd(), head)

        # Strip initial drive from a Windows path. We know we have an absolute
        # path here, so the second part of the check rules out a POSIX path that
        # happens to contain a colon at the second character.
        if head[1] == ':' and head[0] not in path_separators:
            head = head[2:]

        # Strip initial path separator from `head` to complete the conversion
        # back to a root-relative path before joining.
        return _path_join(
            sys.pycache_prefix,
            head.lstrip(path_separators),
            filename,
        )
    return _path_join(head, _PYCACHE, filename)


def source_from_cache(path):
    """Given the path to a .pyc. file, return the path to its .py file.

    The .pyc file does not need to exist; this simply returns the path to
    the .py file calculated to correspond to the .pyc file.  If path does
    not conform to PEP 3147/488 format, ValueError will be raised. If
    sys.implementation.cache_tag is None then NotImplementedError is raised.

    """
    if sys.implementation.cache_tag is None:
        raise NotImplementedError('sys.implementation.cache_tag is None')
    path = _os.fspath(path)
    head, pycache_filename = _path_split(path)
    found_in_pycache_prefix = False
    if sys.pycache_prefix is not None:
        stripped_path = sys.pycache_prefix.rstrip(path_separators)
        if head.startswith(stripped_path + path_sep):
            head = head[len(stripped_path):]
            found_in_pycache_prefix = True
    if not found_in_pycache_prefix:
        head, pycache = _path_split(head)
        if pycache != _PYCACHE:
            raise ValueError(f'{_PYCACHE} not bottom-level directory in '
                             f'{path!r}')
    dot_count = pycache_filename.count('.')
    if dot_count not in {2, 3}:
        raise ValueError(f'expected only 2 or 3 dots in {pycache_filename!r}')
    elif dot_count == 3:
        optimization = pycache_filename.rsplit('.', 2)[-2]
        if not optimization.startswith(_OPT):
            raise ValueError("optimization portion of filename does not start "
                             f"with {_OPT!r}")
        opt_level = optimization[len(_OPT):]
        if not opt_level.isalnum():
            raise ValueError(f"optimization level {optimization!r} is not an "
                             "alphanumeric value")
    base_filename = pycache_filename.partition('.')[0]
    return _path_join(head, base_filename + SOURCE_SUFFIXES[0])


def _get_sourcefile(bytecode_path):
    """Convert a bytecode file path to a source path (if possible).

    This function exists purely for backwards-compatibility for
    PyImport_ExecCodeModuleWithFilenames() in the C API.

    """
    if len(bytecode_path) == 0:
        return None
    rest, _, extension = bytecode_path.rpartition('.')
    if not rest or extension.lower()[-3:-1] != 'py':
        return bytecode_path
    try:
        source_path = source_from_cache(bytecode_path)
    except (NotImplementedError, ValueError):
        source_path = bytecode_path[:-1]
    return source_path if _path_isfile(source_path) else bytecode_path


def _get_cached(filename):
    if filename.endswith(tuple(SOURCE_SUFFIXES)):
        try:
            return cache_from_source(filename)
        except NotImplementedError:
            pass
    elif filename.endswith(tuple(BYTECODE_SUFFIXES)):
        return filename
    else:
        return None


def _calc_mode(path):
    """Calculate the mode permissions for a bytecode file."""
    try:
        mode = _path_stat(path).st_mode
    except OSError:
        mode = 0o666
    # We always ensure write access so we can update cached files
    # later even when the source files are read-only on Windows (#6074)
    mode |= 0o200
    return mode


def _check_name(method):
    """Decorator to verify that the module being requested matches the one the
    loader can handle.

    The first argument (self) must define _name which the second argument is
    compared against. If the comparison fails then ImportError is raised.

    """
    def _check_name_wrapper(self, name=None, *args, **kwargs):
        if name is None:
            name = self.name
        elif self.name != name:
            raise ImportError('loader for %s cannot handle %s' %
                                (self.name, name), name=name)
        return method(self, name, *args, **kwargs)

    # FIXME: @_check_name is used to define class methods before the
    # _bootstrap module is set by _set_bootstrap_module().
    if _bootstrap is not None:
        _wrap = _bootstrap._wrap
    else:
        def _wrap(new, old):
            for replace in ['__module__', '__name__', '__qualname__', '__doc__']:
                if hasattr(old, replace):
                    setattr(new, replace, getattr(old, replace))
            new.__dict__.update(old.__dict__)

    _wrap(_check_name_wrapper, method)
    return _check_name_wrapper


def _find_module_shim(self, fullname):
    """Try to find a loader for the specified module by delegating to
    self.find_loader().

    This method is deprecated in favor of finder.find_spec().

    """
    _warnings.warn("find_module() is deprecated and "
                   "slated for removal in Python 3.12; use find_spec() instead",
                   DeprecationWarning)
    # Call find_loader(). If it returns a string (indicating this
    # is a namespace package portion), generate a warning and
    # return None.
    loader, portions = self.find_loader(fullname)
    if loader is None and len(portions):
        msg = 'Not importing directory {}: missing __init__'
        _warnings.warn(msg.format(portions[0]), ImportWarning)
    return loader


def _classify_pyc(data, name, exc_details):
    """Perform basic validity checking of a pyc header and return the flags field,
    which determines how the pyc should be further validated against the source.

    *data* is the contents of the pyc file. (Only the first 16 bytes are
    required, though.)

    *name* is the name of the module being imported. It is used for logging.

    *exc_details* is a dictionary passed to ImportError if it raised for
    improved debugging.

    ImportError is raised when the magic number is incorrect or when the flags
    field is invalid. EOFError is raised when the data is found to be truncated.

    """
    magic = data[:4]
    if magic != MAGIC_NUMBER:
        message = f'bad magic number in {name!r}: {magic!r}'
        _bootstrap._verbose_message('{}', message)
        raise ImportError(message, **exc_details)
    if len(data) < 16:
        message = f'reached EOF while reading pyc header of {name!r}'
        _bootstrap._verbose_message('{}', message)
        raise EOFError(message)
    flags = _unpack_uint32(data[4:8])
    # Only the first two flags are defined.
    if flags & ~0b11:
        message = f'invalid flags {flags!r} in {name!r}'
        raise ImportError(message, **exc_details)
    return flags


def _validate_timestamp_pyc(data, source_mtime, source_size, name,
                            exc_details):
    """Validate a pyc against the source last-modified time.

    *data* is the contents of the pyc file. (Only the first 16 bytes are
    required.)

    *source_mtime* is the last modified timestamp of the source file.

    *source_size* is None or the size of the source file in bytes.

    *name* is the name of the module being imported. It is used for logging.

    *exc_details* is a dictionary passed to ImportError if it raised for
    improved debugging.

    An ImportError is raised if the bytecode is stale.

    """
    if _unpack_uint32(data[8:12]) != (source_mtime & 0xFFFFFFFF):
        message = f'bytecode is stale for {name!r}'
        _bootstrap._verbose_message('{}', message)
        raise ImportError(message, **exc_details)
    if (source_size is not None and
        _unpack_uint32(data[12:16]) != (source_size & 0xFFFFFFFF)):
        raise ImportError(f'bytecode is stale for {name!r}', **exc_details)


def _validate_hash_pyc(data, source_hash, name, exc_details):
    """Validate a hash-based pyc by checking the real source hash against the one in
    the pyc header.

    *data* is the contents of the pyc file. (Only the first 16 bytes are
    required.)

    *source_hash* is the importlib.util.source_hash() of the source file.

    *name* is the name of the module being imported. It is used for logging.

    *exc_details* is a dictionary passed to ImportError if it raised for
    improved debugging.

    An ImportError is raised if the bytecode is stale.

    """
    if data[8:16] != source_hash:
        raise ImportError(
            f'hash in bytecode doesn\'t match hash of source {name!r}',
            **exc_details,
        )


def _compile_bytecode(data, name=None, bytecode_path=None, source_path=None):
    """Compile bytecode as found in a pyc."""
    code = marshal.loads(data)
    if isinstance(code, _code_type):
        _bootstrap._verbose_message('code object from {!r}', bytecode_path)
        if source_path is not None:
            _imp._fix_co_filename(code, source_path)
        return code
    else:
        raise ImportError('Non-code object in {!r}'.format(bytecode_path),
                          name=name, path=bytecode_path)


def _code_to_timestamp_pyc(code, mtime=0, source_size=0):
    "Produce the data for a timestamp-based pyc."
    data = bytearray(MAGIC_NUMBER)
    data.extend(_pack_uint32(0))
    data.extend(_pack_uint32(mtime))
    data.extend(_pack_uint32(source_size))
    data.extend(marshal.dumps(code))
    return data


def _code_to_hash_pyc(code, source_hash, checked=True):
    "Produce the data for a hash-based pyc."
    data = bytearray(MAGIC_NUMBER)
    flags = 0b1 | checked << 1
    data.extend(_pack_uint32(flags))
    assert len(source_hash) == 8
    data.extend(source_hash)
    data.extend(marshal.dumps(code))
    return data


def decode_source(source_bytes):
    """Decode bytes representing source code and return the string.

    Universal newline support is used in the decoding.
    """
    import tokenize  # To avoid bootstrap issues.
    source_bytes_readline = _io.BytesIO(source_bytes).readline
    encoding = tokenize.detect_encoding(source_bytes_readline)
    newline_decoder = _io.IncrementalNewlineDecoder(None, True)
    return newline_decoder.decode(source_bytes.decode(encoding[0]))


# Module specifications #######################################################

_POPULATE = object()


def spec_from_file_location(name, location=None, *, loader=None,
                            submodule_search_locations=_POPULATE):
    """Return a module spec based on a file location.

    To indicate that the module is a package, set
    submodule_search_locations to a list of directory paths.  An
    empty list is sufficient, though its not otherwise useful to the
    import system.

    The loader must take a spec as its only __init__() arg.

    """
    if location is None:
        # The caller may simply want a partially populated location-
        # oriented spec.  So we set the location to a bogus value and
        # fill in as much as we can.
        location = '<unknown>'
        if hasattr(loader, 'get_filename'):
            # ExecutionLoader
            try:
                location = loader.get_filename(name)
            except ImportError:
                pass
    else:
        location = _os.fspath(location)
        if not _path_isabs(location):
            try:
                location = _path_join(_os.getcwd(), location)
            except OSError:
                pass

    # If the location is on the filesystem, but doesn't actually exist,
    # we could return None here, indicating that the location is not
    # valid.  However, we don't have a good way of testing since an
    # indirect location (e.g. a zip file or URL) will look like a
    # non-existent file relative to the filesystem.

    spec = _bootstrap.ModuleSpec(name, loader, origin=location)
    spec._set_fileattr = True

    # Pick a loader if one wasn't provided.
    if loader is None:
        for loader_class, suffixes in _get_supported_file_loaders():
            if location.endswith(tuple(suffixes)):
                loader = loader_class(name, location)
                spec.loader = loader
                break
        else:
            return None

    # Set submodule_search_paths appropriately.
    if submodule_search_locations is _POPULATE:
        # Check the loader.
        if hasattr(loader, 'is_package'):
            try:
                is_package = loader.is_package(name)
            except ImportError:
                pass
            else:
                if is_package:
                    spec.submodule_search_locations = []
    else:
        spec.submodule_search_locations = submodule_search_locations
    if spec.submodule_search_locations == []:
        if location:
            dirname = _path_split(location)[0]
            spec.submodule_search_locations.append(dirname)

    return spec


# Loaders #####################################################################

class WindowsRegistryFinder:

    """Meta path finder for modules declared in the Windows registry."""

    REGISTRY_KEY = (
        'Software\\Python\\PythonCore\\{sys_version}'
        '\\Modules\\{fullname}')
    REGISTRY_KEY_DEBUG = (
        'Software\\Python\\PythonCore\\{sys_version}'
        '\\Modules\\{fullname}\\Debug')
    DEBUG_BUILD = (_MS_WINDOWS and '_d.pyd' in EXTENSION_SUFFIXES)

    @staticmethod
    def _open_registry(key):
        try:
            return winreg.OpenKey(winreg.HKEY_CURRENT_USER, key)
        except OSError:
            return winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key)

    @classmethod
    def _search_registry(cls, fullname):
        if cls.DEBUG_BUILD:
            registry_key = cls.REGISTRY_KEY_DEBUG
        else:
            registry_key = cls.REGISTRY_KEY
        key = registry_key.format(fullname=fullname,
                                  sys_version='%d.%d' % sys.version_info[:2])
        try:
            with cls._open_registry(key) as hkey:
                filepath = winreg.QueryValue(hkey, '')
        except OSError:
            return None
        return filepath

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        filepath = cls._search_registry(fullname)
        if filepath is None:
            return None
        try:
            _path_stat(filepath)
        except OSError:
            return None
        for loader, suffixes in _get_supported_file_loaders():
            if filepath.endswith(tuple(suffixes)):
                spec = _bootstrap.spec_from_loader(fullname,
                                                   loader(fullname, filepath),
                                                   origin=filepath)
                return spec

    @classmethod
    def find_module(cls, fullname, path=None):
        """Find module named in the registry.

        This method is deprecated.  Use find_spec() instead.

        """
        _warnings.warn("WindowsRegistryFinder.find_module() is deprecated and "
                       "slated for removal in Python 3.12; use find_spec() instead",
                       DeprecationWarning)
        spec = cls.find_spec(fullname, path)
        if spec is not None:
            return spec.loader
        else:
            return None


class _LoaderBasics:

    """Base class of common code needed by both SourceLoader and
    SourcelessFileLoader."""

    def is_package(self, fullname):
        """Concrete implementation of InspectLoader.is_package by checking if
        the path returned by get_filename has a filename of '__init__.py'."""
        filename = _path_split(self.get_filename(fullname))[1]
        filename_base = filename.rsplit('.', 1)[0]
        tail_name = fullname.rpartition('.')[2]
        return filename_base == '__init__' and tail_name != '__init__'

    def create_module(self, spec):
        """Use default semantics for module creation."""

    def exec_module(self, module):
        """Execute the module."""
        code = self.get_code(module.__name__)
        if code is None:
            raise ImportError('cannot load module {!r} when get_code() '
                              'returns None'.format(module.__name__))
        _bootstrap._call_with_frames_removed(exec, code, module.__dict__)

    def load_module(self, fullname):
        """This method is deprecated."""
        # Warning implemented in _load_module_shim().
        return _bootstrap._load_module_shim(self, fullname)


class SourceLoader(_LoaderBasics):

    def path_mtime(self, path):
        """Optional method that returns the modification time (an int) for the
        specified path (a str).

        Raises OSError when the path cannot be handled.
        """
        raise OSError

    def path_stats(self, path):
        """Optional method returning a metadata dict for the specified
        path (a str).

        Possible keys:
        - 'mtime' (mandatory) is the numeric timestamp of last source
          code modification;
        - 'size' (optional) is the size in bytes of the source code.

        Implementing this method allows the loader to read bytecode files.
        Raises OSError when the path cannot be handled.
        """
        return {'mtime': self.path_mtime(path)}

    def _cache_bytecode(self, source_path, cache_path, data):
        """Optional method which writes data (bytes) to a file path (a str).

        Implementing this method allows for the writing of bytecode files.

        The source path is needed in order to correctly transfer permissions
        """
        # For backwards compatibility, we delegate to set_data()
        return self.set_data(cache_path, data)

    def set_data(self, path, data):
        """Optional method which writes data (bytes) to a file path (a str).

        Implementing this method allows for the writing of bytecode files.
        """


    def get_source(self, fullname):
        """Concrete implementation of InspectLoader.get_source."""
        path = self.get_filename(fullname)
        try:
            source_bytes = self.get_data(path)
        except OSError as exc:
            raise ImportError('source not available through get_data()',
                              name=fullname) from exc
        return decode_source(source_bytes)

    def source_to_code(self, data, path, *, _optimize=-1):
        """Return the code object compiled from source.

        The 'data' argument can be any object type that compile() supports.
        """
        return _bootstrap._call_with_frames_removed(compile, data, path, 'exec',
                                        dont_inherit=True, optimize=_optimize)

    def get_code(self, fullname):
        """Concrete implementation of InspectLoader.get_code.

        Reading of bytecode requires path_stats to be implemented. To write
        bytecode, set_data must also be implemented.

        """
        source_path = self.get_filename(fullname)
        source_mtime = None
        source_bytes = None
        source_hash = None
        hash_based = False
        check_source = True
        try:
            bytecode_path = cache_from_source(source_path)
        except NotImplementedError:
            bytecode_path = None
        else:
            try:
                st = self.path_stats(source_path)
            except OSError:
                pass
            else:
                source_mtime = int(st['mtime'])
                try:
                    data = self.get_data(bytecode_path)
                except OSError:
                    pass
                else:
                    exc_details = {
                        'name': fullname,
                        'path': bytecode_path,
                    }
                    try:
                        flags = _classify_pyc(data, fullname, exc_details)
                        bytes_data = memoryview(data)[16:]
                        hash_based = flags & 0b1 != 0
                        if hash_based:
                            check_source = flags & 0b10 != 0
                            if (_imp.check_hash_based_pycs != 'never' and
                                (check_source or
                                 _imp.check_hash_based_pycs == 'always')):
                                source_bytes = self.get_data(source_path)
                                source_hash = _imp.source_hash(
                                    _RAW_MAGIC_NUMBER,
                                    source_bytes,
                                )
                                _validate_hash_pyc(data, source_hash, fullname,
                                                   exc_details)
                        else:
                            _validate_timestamp_pyc(
                                data,
                                source_mtime,
                                st['size'],
                                fullname,
                                exc_details,
                            )
                    except (ImportError, EOFError):
                        pass
                    else:
                        _bootstrap._verbose_message('{} matches {}', bytecode_path,
                                                    source_path)
                        return _compile_bytecode(bytes_data, name=fullname,
                                                 bytecode_path=bytecode_path,
                                                 source_path=source_path)
        if source_bytes is None:
            source_bytes = self.get_data(source_path)
        code_object = self.source_to_code(source_bytes, source_path)
        _bootstrap._verbose_message('code object from {}', source_path)
        if (not sys.dont_write_bytecode and bytecode_path is not None and
                source_mtime is not None):
            if hash_based:
                if source_hash is None:
                    source_hash = _imp.source_hash(source_bytes)
                data = _code_to_hash_pyc(code_object, source_hash, check_source)
            else:
                data = _code_to_timestamp_pyc(code_object, source_mtime,
                                              len(source_bytes))
            try:
                self._cache_bytecode(source_path, bytecode_path, data)
            except NotImplementedError:
                pass
        return code_object


class FileLoader:

    """Base file loader class which implements the loader protocol methods that
    require file system usage."""

    def __init__(self, fullname, path):
        """Cache the module name and the path to the file found by the
        finder."""
        self.name = fullname
        self.path = path

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self.__dict__ == other.__dict__)

    def __hash__(self):
        return hash(self.name) ^ hash(self.path)

    @_check_name
    def load_module(self, fullname):
        """Load a module from a file.

        This method is deprecated.  Use exec_module() instead.

        """
        # The only reason for this method is for the name check.
        # Issue #14857: Avoid the zero-argument form of super so the implementation
        # of that form can be updated without breaking the frozen module.
        return super(FileLoader, self).load_module(fullname)

    @_check_name
    def get_filename(self, fullname):
        """Return the path to the source file as found by the finder."""
        return self.path

    def get_data(self, path):
        """Return the data from path as raw bytes."""
        if isinstance(self, (SourceLoader, SourcelessFileLoader, ExtensionFileLoader)):
            with _io.open_code(str(path)) as file:
                return file.read()
        else:
            with _io.FileIO(path, 'r') as file:
                return file.read()

    @_check_name
    def get_resource_reader(self, module):
        from importlib.readers import FileReader
        return FileReader(self)


class SourceFileLoader(FileLoader, SourceLoader):

    """Concrete implementation of SourceLoader using the file system."""

    def path_stats(self, path):
        """Return the metadata for the path."""
        st = _path_stat(path)
        return {'mtime': st.st_mtime, 'size': st.st_size}

    def _cache_bytecode(self, source_path, bytecode_path, data):
        # Adapt between the two APIs
        mode = _calc_mode(source_path)
        return self.set_data(bytecode_path, data, _mode=mode)

    def set_data(self, path, data, *, _mode=0o666):
        """Write bytes data to a file."""
        parent, filename = _path_split(path)
        path_parts = []
        # Figure out what directories are missing.
        while parent and not _path_isdir(parent):
            parent, part = _path_split(parent)
            path_parts.append(part)
        # Create needed directories.
        for part in reversed(path_parts):
            parent = _path_join(parent, part)
            try:
                _os.mkdir(parent)
            except FileExistsError:
                # Probably another Python process already created the dir.
                continue
            except OSError as exc:
                # Could be a permission error, read-only filesystem: just forget
                # about writing the data.
                _bootstrap._verbose_message('could not create {!r}: {!r}',
                                            parent, exc)
                return
        try:
            _write_atomic(path, data, _mode)
            _bootstrap._verbose_message('created {!r}', path)
        except OSError as exc:
            # Same as above: just don't write the bytecode.
            _bootstrap._verbose_message('could not create {!r}: {!r}', path,
                                        exc)


class SourcelessFileLoader(FileLoader, _LoaderBasics):

    """Loader which handles sourceless file imports."""

    def get_code(self, fullname):
        path = self.get_filename(fullname)
        data = self.get_data(path)
        # Call _classify_pyc to do basic validation of the pyc but ignore the
        # result. There's no source to check against.
        exc_details = {
            'name': fullname,
            'path': path,
        }
        _classify_pyc(data, fullname, exc_details)
        return _compile_bytecode(
            memoryview(data)[16:],
            name=fullname,
            bytecode_path=path,
        )

    def get_source(self, fullname):
        """Return None as there is no source code."""
        return None


class ExtensionFileLoader(FileLoader, _LoaderBasics):

    """Loader for extension modules.

    The constructor is designed to work with FileFinder.

    """

    def __init__(self, name, path):
        self.name = name
        self.path = path

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self.__dict__ == other.__dict__)

    def __hash__(self):
        return hash(self.name) ^ hash(self.path)

    def create_module(self, spec):
        """Create an unitialized extension module"""
        module = _bootstrap._call_with_frames_removed(
            _imp.create_dynamic, spec)
        _bootstrap._verbose_message('extension module {!r} loaded from {!r}',
                         spec.name, self.path)
        return module

    def exec_module(self, module):
        """Initialize an extension module"""
        _bootstrap._call_with_frames_removed(_imp.exec_dynamic, module)
        _bootstrap._verbose_message('extension module {!r} executed from {!r}',
                         self.name, self.path)

    def is_package(self, fullname):
        """Return True if the extension module is a package."""
        file_name = _path_split(self.path)[1]
        return any(file_name == '__init__' + suffix
                   for suffix in EXTENSION_SUFFIXES)

    def get_code(self, fullname):
        """Return None as an extension module cannot create a code object."""
        return None

    def get_source(self, fullname):
        """Return None as extension modules have no source code."""
        return None

    @_check_name
    def get_filename(self, fullname):
        """Return the path to the source file as found by the finder."""
        return self.path


class _NamespacePath:
    """Represents a namespace package's path.  It uses the module name
    to find its parent module, and from there it looks up the parent's
    __path__.  When this changes, the module's own path is recomputed,
    using path_finder.  For top-level modules, the parent module's path
    is sys.path."""

    # When invalidate_caches() is called, this epoch is incremented
    # https://bugs.python.org/issue45703
    _epoch = 0

    def __init__(self, name, path, path_finder):
        self._name = name
        self._path = path
        self._last_parent_path = tuple(self._get_parent_path())
        self._last_epoch = self._epoch
        self._path_finder = path_finder

    def _find_parent_path_names(self):
        """Returns a tuple of (parent-module-name, parent-path-attr-name)"""
        parent, dot, me = self._name.rpartition('.')
        if dot == '':
            # This is a top-level module. sys.path contains the parent path.
            return 'sys', 'path'
        # Not a top-level module. parent-module.__path__ contains the
        #  parent path.
        return parent, '__path__'

    def _get_parent_path(self):
        parent_module_name, path_attr_name = self._find_parent_path_names()
        return getattr(sys.modules[parent_module_name], path_attr_name)

    def _recalculate(self):
        # If the parent's path has changed, recalculate _path
        parent_path = tuple(self._get_parent_path()) # Make a copy
        if parent_path != self._last_parent_path or self._epoch != self._last_epoch:
            spec = self._path_finder(self._name, parent_path)
            # Note that no changes are made if a loader is returned, but we
            #  do remember the new parent path
            if spec is not None and spec.loader is None:
                if spec.submodule_search_locations:
                    self._path = spec.submodule_search_locations
            self._last_parent_path = parent_path     # Save the copy
            self._last_epoch = self._epoch
        return self._path

    def __iter__(self):
        return iter(self._recalculate())

    def __getitem__(self, index):
        return self._recalculate()[index]

    def __setitem__(self, index, path):
        self._path[index] = path

    def __len__(self):
        return len(self._recalculate())

    def __repr__(self):
        return '_NamespacePath({!r})'.format(self._path)

    def __contains__(self, item):
        return item in self._recalculate()

    def append(self, item):
        self._path.append(item)


# We use this exclusively in module_from_spec() for backward-compatibility.
class _NamespaceLoader:
    def __init__(self, name, path, path_finder):
        self._path = _NamespacePath(name, path, path_finder)

    @staticmethod
    def module_repr(module):
        """Return repr for the module.

        The method is deprecated.  The import machinery does the job itself.

        """
        _warnings.warn("_NamespaceLoader.module_repr() is deprecated and "
                       "slated for removal in Python 3.12", DeprecationWarning)
        return '<module {!r} (namespace)>'.format(module.__name__)

    def is_package(self, fullname):
        return True

    def get_source(self, fullname):
        return ''

    def get_code(self, fullname):
        return compile('', '<string>', 'exec', dont_inherit=True)

    def create_module(self, spec):
        """Use default semantics for module creation."""

    def exec_module(self, module):
        pass

    def load_module(self, fullname):
        """Load a namespace module.

        This method is deprecated.  Use exec_module() instead.

        """
        # The import system never calls this method.
        _bootstrap._verbose_message('namespace module loaded with path {!r}',
                                    self._path)
        # Warning implemented in _load_module_shim().
        return _bootstrap._load_module_shim(self, fullname)

    def get_resource_reader(self, module):
        from importlib.readers import NamespaceReader
        return NamespaceReader(self._path)


# Finders #####################################################################

class PathFinder:

    """Meta path finder for sys.path and package __path__ attributes."""

    @staticmethod
    def invalidate_caches():
        """Call the invalidate_caches() method on all path entry finders
        stored in sys.path_importer_caches (where implemented)."""
        for name, finder in list(sys.path_importer_cache.items()):
            if finder is None:
                del sys.path_importer_cache[name]
            elif hasattr(finder, 'invalidate_caches'):
                finder.invalidate_caches()
        # Also invalidate the caches of _NamespacePaths
        # https://bugs.python.org/issue45703
        _NamespacePath._epoch += 1

    @staticmethod
    def _path_hooks(path):
        """Search sys.path_hooks for a finder for 'path'."""
        if sys.path_hooks is not None and not sys.path_hooks:
            _warnings.warn('sys.path_hooks is empty', ImportWarning)
        for hook in sys.path_hooks:
            try:
                return hook(path)
            except ImportError:
                continue
        else:
            return None

    @classmethod
    def _path_importer_cache(cls, path):
        """Get the finder for the path entry from sys.path_importer_cache.

        If the path entry is not in the cache, find the appropriate finder
        and cache it. If no finder is available, store None.

        """
        if path == '':
            try:
                path = _os.getcwd()
            except FileNotFoundError:
                # Don't cache the failure as the cwd can easily change to
                # a valid directory later on.
                return None
        try:
            finder = sys.path_importer_cache[path]
        except KeyError:
            finder = cls._path_hooks(path)
            sys.path_importer_cache[path] = finder
        return finder

    @classmethod
    def _legacy_get_spec(cls, fullname, finder):
        # This would be a good place for a DeprecationWarning if
        # we ended up going that route.
        if hasattr(finder, 'find_loader'):
            msg = (f"{_bootstrap._object_name(finder)}.find_spec() not found; "
                    "falling back to find_loader()")
            _warnings.warn(msg, ImportWarning)
            loader, portions = finder.find_loader(fullname)
        else:
            msg = (f"{_bootstrap._object_name(finder)}.find_spec() not found; "
                    "falling back to find_module()")
            _warnings.warn(msg, ImportWarning)
            loader = finder.find_module(fullname)
            portions = []
        if loader is not None:
            return _bootstrap.spec_from_loader(fullname, loader)
        spec = _bootstrap.ModuleSpec(fullname, None)
        spec.submodule_search_locations = portions
        return spec

    @classmethod
    def _get_spec(cls, fullname, path, target=None):
        """Find the loader or namespace_path for this module/package name."""
        # If this ends up being a namespace package, namespace_path is
        #  the list of paths that will become its __path__
        namespace_path = []
        for entry in path:
            if not isinstance(entry, (str, bytes)):
                continue
            finder = cls._path_importer_cache(entry)
            if finder is not None:
                if hasattr(finder, 'find_spec'):
                    spec = finder.find_spec(fullname, target)
                else:
                    spec = cls._legacy_get_spec(fullname, finder)
                if spec is None:
                    continue
                if spec.loader is not None:
                    return spec
                portions = spec.submodule_search_locations
                if portions is None:
                    raise ImportError('spec missing loader')
                # This is possibly part of a namespace package.
                #  Remember these path entries (if any) for when we
                #  create a namespace package, and continue iterating
                #  on path.
                namespace_path.extend(portions)
        else:
            spec = _bootstrap.ModuleSpec(fullname, None)
            spec.submodule_search_locations = namespace_path
            return spec

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        """Try to find a spec for 'fullname' on sys.path or 'path'.

        The search is based on sys.path_hooks and sys.path_importer_cache.
        """
        if path is None:
            path = sys.path
        spec = cls._get_spec(fullname, path, target)
        if spec is None:
            return None
        elif spec.loader is None:
            namespace_path = spec.submodule_search_locations
            if namespace_path:
                # We found at least one namespace path.  Return a spec which
                # can create the namespace package.
                spec.origin = None
                spec.submodule_search_locations = _NamespacePath(fullname, namespace_path, cls._get_spec)
                return spec
            else:
                return None
        else:
            return spec

    @classmethod
    def find_module(cls, fullname, path=None):
        """find the module on sys.path or 'path' based on sys.path_hooks and
        sys.path_importer_cache.

        This method is deprecated.  Use find_spec() instead.

        """
        _warnings.warn("PathFinder.find_module() is deprecated and "
                       "slated for removal in Python 3.12; use find_spec() instead",
                       DeprecationWarning)
        spec = cls.find_spec(fullname, path)
        if spec is None:
            return None
        return spec.loader

    @staticmethod
    def find_distributions(*args, **kwargs):
        """
        Find distributions.

        Return an iterable of all Distribution instances capable of
        loading the metadata for packages matching ``context.name``
        (or all names if ``None`` indicated) along the paths in the list
        of directories ``context.path``.
        """
        from importlib.metadata import MetadataPathFinder
        return MetadataPathFinder.find_distributions(*args, **kwargs)


class FileFinder:

    """File-based finder.

    Interactions with the file system are cached for performance, being
    refreshed when the directory the finder is handling has been modified.

    """

    def __init__(self, path, *loader_details):
        """Initialize with the path to search on and a variable number of
        2-tuples containing the loader and the file suffixes the loader
        recognizes."""
        loaders = []
        for loader, suffixes in loader_details:
            loaders.extend((suffix, loader) for suffix in suffixes)
        self._loaders = loaders
        # Base (directory) path
        self.path = path or '.'
        if not _path_isabs(self.path):
            self.path = _path_join(_os.getcwd(), self.path)
        self._path_mtime = -1
        self._path_cache = set()
        self._relaxed_path_cache = set()

    def invalidate_caches(self):
        """Invalidate the directory mtime."""
        self._path_mtime = -1

    find_module = _find_module_shim

    def find_loader(self, fullname):
        """Try to find a loader for the specified module, or the namespace
        package portions. Returns (loader, list-of-portions).

        This method is deprecated.  Use find_spec() instead.

        """
        _warnings.warn("FileFinder.find_loader() is deprecated and "
                       "slated for removal in Python 3.12; use find_spec() instead",
                       DeprecationWarning)
        spec = self.find_spec(fullname)
        if spec is None:
            return None, []
        return spec.loader, spec.submodule_search_locations or []

    def _get_spec(self, loader_class, fullname, path, smsl, target):
        loader = loader_class(fullname, path)
        return spec_from_file_location(fullname, path, loader=loader,
                                       submodule_search_locations=smsl)

    def find_spec(self, fullname, target=None):
        """Try to find a spec for the specified module.

        Returns the matching spec, or None if not found.
        """
        is_namespace = False
        tail_module = fullname.rpartition('.')[2]
        try:
            mtime = _path_stat(self.path or _os.getcwd()).st_mtime
        except OSError:
            mtime = -1
        if mtime != self._path_mtime:
            self._fill_cache()
            self._path_mtime = mtime
        # tail_module keeps the original casing, for __file__ and friends
        if _relax_case():
            cache = self._relaxed_path_cache
            cache_module = tail_module.lower()
        else:
            cache = self._path_cache
            cache_module = tail_module
        # Check if the module is the name of a directory (and thus a package).
        if cache_module in cache:
            base_path = _path_join(self.path, tail_module)
            for suffix, loader_class in self._loaders:
                init_filename = '__init__' + suffix
                full_path = _path_join(base_path, init_filename)
                if _path_isfile(full_path):
                    return self._get_spec(loader_class, fullname, full_path, [base_path], target)
            else:
                # If a namespace package, return the path if we don't
                #  find a module in the next section.
                is_namespace = _path_isdir(base_path)
        # Check for a file w/ a proper suffix exists.
        for suffix, loader_class in self._loaders:
            try:
                full_path = _path_join(self.path, tail_module + suffix)
            except ValueError:
                return None
            _bootstrap._verbose_message('trying {}', full_path, verbosity=2)
            if cache_module + suffix in cache:
                if _path_isfile(full_path):
                    return self._get_spec(loader_class, fullname, full_path,
                                          None, target)
        if is_namespace:
            _bootstrap._verbose_message('possible namespace for {}', base_path)
            spec = _bootstrap.ModuleSpec(fullname, None)
            spec.submodule_search_locations = [base_path]
            return spec
        return None

    def _fill_cache(self):
        """Fill the cache of potential modules and packages for this directory."""
        path = self.path
        try:
            contents = _os.listdir(path or _os.getcwd())
        except (FileNotFoundError, PermissionError, NotADirectoryError):
            # Directory has either been removed, turned into a file, or made
            # unreadable.
            contents = []
        # We store two cached versions, to handle runtime changes of the
        # PYTHONCASEOK environment variable.
        if not sys.platform.startswith('win'):
            self._path_cache = set(contents)
        else:
            # Windows users can import modules with case-insensitive file
            # suffixes (for legacy reasons). Make the suffix lowercase here
            # so it's done once instead of for every import. This is safe as
            # the specified suffixes to check against are always specified in a
            # case-sensitive manner.
            lower_suffix_contents = set()
            for item in contents:
                name, dot, suffix = item.partition('.')
                if dot:
                    new_name = '{}.{}'.format(name, suffix.lower())
                else:
                    new_name = name
                lower_suffix_contents.add(new_name)
            self._path_cache = lower_suffix_contents
        if sys.platform.startswith(_CASE_INSENSITIVE_PLATFORMS):
            self._relaxed_path_cache = {fn.lower() for fn in contents}

    @classmethod
    def path_hook(cls, *loader_details):
        """A class method which returns a closure to use on sys.path_hook
        which will return an instance using the specified loaders and the path
        called on the closure.

        If the path called on the closure is not a directory, ImportError is
        raised.

        """
        def path_hook_for_FileFinder(path):
            """Path hook for importlib.machinery.FileFinder."""
            if not _path_isdir(path):
                raise ImportError('only directories are supported', path=path)
            return cls(path, *loader_details)

        return path_hook_for_FileFinder

    def __repr__(self):
        return 'FileFinder({!r})'.format(self.path)


# Import setup ###############################################################

def _fix_up_module(ns, name, pathname, cpathname=None):
    # This function is used by PyImport_ExecCodeModuleObject().
    loader = ns.get('__loader__')
    spec = ns.get('__spec__')
    if not loader:
        if spec:
            loader = spec.loader
        elif pathname == cpathname:
            loader = SourcelessFileLoader(name, pathname)
        else:
            loader = SourceFileLoader(name, pathname)
    if not spec:
        spec = spec_from_file_location(name, pathname, loader=loader)
    try:
        ns['__spec__'] = spec
        ns['__loader__'] = loader
        ns['__file__'] = pathname
        ns['__cached__'] = cpathname
    except Exception:
        # Not important enough to report.
        pass


def _get_supported_file_loaders():
    """Returns a list of file-based module loaders.

    Each item is a tuple (loader, suffixes).
    """
    extensions = ExtensionFileLoader, _imp.extension_suffixes()
    source = SourceFileLoader, SOURCE_SUFFIXES
    bytecode = SourcelessFileLoader, BYTECODE_SUFFIXES
    return [extensions, source, bytecode]


def _set_bootstrap_module(_bootstrap_module):
    global _bootstrap
    _bootstrap = _bootstrap_module


def _install(_bootstrap_module):
    """Install the path-based import components."""
    _set_bootstrap_module(_bootstrap_module)
    supported_loaders = _get_supported_file_loaders()
    sys.path_hooks.extend([FileFinder.path_hook(*supported_loaders)])
    sys.meta_path.append(PathFinder)
<|endoftext|>import collections


# from jaraco.collections 3.3
class FreezableDefaultDict(collections.defaultdict):
    """
    Often it is desirable to prevent the mutation of
    a default dict after its initial construction, such
    as to prevent mutation during iteration.

    >>> dd = FreezableDefaultDict(list)
    >>> dd[0].append('1')
    >>> dd.freeze()
    >>> dd[1]
    []
    >>> len(dd)
    1
    """

    def __missing__(self, key):
        return getattr(self, '_frozen', super().__missing__)(key)

    def freeze(self):
        self._frozen = lambda key: self.default_factory()


class Pair(collections.namedtuple('Pair', 'name value')):
    @classmethod
    def parse(cls, text):
        return cls(*map(str.strip, text.split("=", 1)))
<|endoftext|>from typing import Any, Dict, Iterator, List, Protocol, TypeVar, Union


_T = TypeVar("_T")


class PackageMetadata(Protocol):
    def __len__(self) -> int:
        ...  # pragma: no cover

    def __contains__(self, item: str) -> bool:
        ...  # pragma: no cover

    def __getitem__(self, key: str) -> str:
        ...  # pragma: no cover

    def __iter__(self) -> Iterator[str]:
        ...  # pragma: no cover

    def get_all(self, name: str, failobj: _T = ...) -> Union[List[Any], _T]:
        """
        Return all values associated with a possibly multi-valued key.
        """

    @property
    def json(self) -> Dict[str, Union[str, List[str]]]:
        """
        A JSON-compatible form of the metadata.
        """


class SimplePath(Protocol):
    """
    A minimal subset of pathlib.Path required by PathDistribution.
    """

    def joinpath(self) -> 'SimplePath':
        ...  # pragma: no cover

    def __div__(self) -> 'SimplePath':
        ...  # pragma: no cover

    def parent(self) -> 'SimplePath':
        ...  # pragma: no cover

    def read_text(self) -> str:
        ...  # pragma: no cover
<|endoftext|>import re

from ._functools import method_cache


# from jaraco.text 3.5
class FoldedCase(str):
    """
    A case insensitive string class; behaves just like str
    except compares equal when the only variation is case.

    >>> s = FoldedCase('hello world')

    >>> s == 'Hello World'
    True

    >>> 'Hello World' == s
    True

    >>> s != 'Hello World'
    False

    >>> s.index('O')
    4

    >>> s.split('O')
    ['hell', ' w', 'rld']

    >>> sorted(map(FoldedCase, ['GAMMA', 'alpha', 'Beta']))
    ['alpha', 'Beta', 'GAMMA']

    Sequence membership is straightforward.

    >>> "Hello World" in [s]
    True
    >>> s in ["Hello World"]
    True

    You may test for set inclusion, but candidate and elements
    must both be folded.

    >>> FoldedCase("Hello World") in {s}
    True
    >>> s in {FoldedCase("Hello World")}
    True

    String inclusion works as long as the FoldedCase object
    is on the right.

    >>> "hello" in FoldedCase("Hello World")
    True

    But not if the FoldedCase object is on the left:

    >>> FoldedCase('hello') in 'Hello World'
    False

    In that case, use in_:

    >>> FoldedCase('hello').in_('Hello World')
    True

    >>> FoldedCase('hello') > FoldedCase('Hello')
    False
    """

    def __lt__(self, other):
        return self.lower() < other.lower()

    def __gt__(self, other):
        return self.lower() > other.lower()

    def __eq__(self, other):
        return self.lower() == other.lower()

    def __ne__(self, other):
        return self.lower() != other.lower()

    def __hash__(self):
        return hash(self.lower())

    def __contains__(self, other):
        return super(FoldedCase, self).lower().__contains__(other.lower())

    def in_(self, other):
        "Does self appear in other?"
        return self in FoldedCase(other)

    # cache lower since it's likely to be called frequently.
    @method_cache
    def lower(self):
        return super(FoldedCase, self).lower()

    def index(self, sub):
        return self.lower().index(sub.lower())

    def split(self, splitter=' ', maxsplit=0):
        pattern = re.compile(re.escape(splitter), re.I)
        return pattern.split(self, maxsplit)
<|endoftext|>import os
import re
import abc
import csv
import sys
import email
import pathlib
import zipfile
import operator
import textwrap
import warnings
import functools
import itertools
import posixpath
import collections

from . import _adapters, _meta
from ._meta import PackageMetadata
from ._collections import FreezableDefaultDict, Pair
from ._functools import method_cache, pass_none
from ._itertools import unique_everseen
from ._meta import PackageMetadata, SimplePath

from contextlib import suppress
from importlib import import_module
from importlib.abc import MetaPathFinder
from itertools import starmap
from typing import List, Mapping, Optional, Union


__all__ = [
    'Distribution',
    'DistributionFinder',
    'PackageMetadata',
    'PackageNotFoundError',
    'distribution',
    'distributions',
    'entry_points',
    'files',
    'metadata',
    'packages_distributions',
    'requires',
    'version',
]


class PackageNotFoundError(ModuleNotFoundError):
    """The package was not found."""

    def __str__(self):
        return f"No package metadata was found for {self.name}"

    @property
    def name(self):
        (name,) = self.args
        return name


class Sectioned:
    """
    A simple entry point config parser for performance

    >>> for item in Sectioned.read(Sectioned._sample):
    ...     print(item)
    Pair(name='sec1', value='# comments ignored')
    Pair(name='sec1', value='a = 1')
    Pair(name='sec1', value='b = 2')
    Pair(name='sec2', value='a = 2')

    >>> res = Sectioned.section_pairs(Sectioned._sample)
    >>> item = next(res)
    >>> item.name
    'sec1'
    >>> item.value
    Pair(name='a', value='1')
    >>> item = next(res)
    >>> item.value
    Pair(name='b', value='2')
    >>> item = next(res)
    >>> item.name
    'sec2'
    >>> item.value
    Pair(name='a', value='2')
    >>> list(res)
    []
    """

    _sample = textwrap.dedent(
        """
        [sec1]
        # comments ignored
        a = 1
        b = 2

        [sec2]
        a = 2
        """
    ).lstrip()

    @classmethod
    def section_pairs(cls, text):
        return (
            section._replace(value=Pair.parse(section.value))
            for section in cls.read(text, filter_=cls.valid)
            if section.name is not None
        )

    @staticmethod
    def read(text, filter_=None):
        lines = filter(filter_, map(str.strip, text.splitlines()))
        name = None
        for value in lines:
            section_match = value.startswith('[') and value.endswith(']')
            if section_match:
                name = value.strip('[]')
                continue
            yield Pair(name, value)

    @staticmethod
    def valid(line):
        return line and not line.startswith('#')


class EntryPoint(
        collections.namedtuple('EntryPointBase', 'name value group')):
    """An entry point as defined by Python packaging conventions.

    See `the packaging docs on entry points
    <https://packaging.python.org/specifications/entry-points/>`_
    for more information.

    >>> ep = EntryPoint(
    ...     name=None, group=None, value='package.module:attr [extra1, extra2]')
    >>> ep.module
    'package.module'
    >>> ep.attr
    'attr'
    >>> ep.extras
    ['extra1', 'extra2']
    """

    pattern = re.compile(
        r'(?P<module>[\w.]+)\s*'
        r'(:\s*(?P<attr>[\w.]+)\s*)?'
        r'((?P<extras>\[.*\])\s*)?$'
    )
    """
    A regular expression describing the syntax for an entry point,
    which might look like:

        - module
        - package.module
        - package.module:attribute
        - package.module:object.attribute
        - package.module:attr [extra1, extra2]

    Other combinations are possible as well.

    The expression is lenient about whitespace around the ':',
    following the attr, and following any extras.
    """

    dist: Optional['Distribution'] = None

    def load(self):
        """Load the entry point from its definition. If only a module
        is indicated by the value, return that module. Otherwise,
        return the named object.
        """
        match = self.pattern.match(self.value)
        module = import_module(match.group('module'))
        attrs = filter(None, (match.group('attr') or '').split('.'))
        return functools.reduce(getattr, attrs, module)

    @property
    def module(self):
        match = self.pattern.match(self.value)
        return match.group('module')

    @property
    def attr(self):
        match = self.pattern.match(self.value)
        return match.group('attr')

    @property
    def extras(self):
        match = self.pattern.match(self.value)
        return re.findall(r'\w+', match.group('extras') or '')

    def _for(self, dist):
        self.dist = dist
        return self

    def __iter__(self):
        """
        Supply iter so one may construct dicts of EntryPoints by name.
        """
        msg = (
            "Construction of dict of EntryPoints is deprecated in "
            "favor of EntryPoints."
        )
        warnings.warn(msg, DeprecationWarning)
        return iter((self.name, self))

    def __reduce__(self):
        return (
            self.__class__,
            (self.name, self.value, self.group),
        )

    def matches(self, **params):
        """
        EntryPoint matches the given parameters.

        >>> ep = EntryPoint(group='foo', name='bar', value='bing:bong [extra1, extra2]')
        >>> ep.matches(group='foo')
        True
        >>> ep.matches(name='bar', value='bing:bong [extra1, extra2]')
        True
        >>> ep.matches(group='foo', name='other')
        False
        >>> ep.matches()
        True
        >>> ep.matches(extras=['extra1', 'extra2'])
        True
        >>> ep.matches(module='bing')
        True
        >>> ep.matches(attr='bong')
        True
        """
        attrs = (getattr(self, param) for param in params)
        return all(map(operator.eq, params.values(), attrs))


class DeprecatedList(list):
    """
    Allow an otherwise immutable object to implement mutability
    for compatibility.

    >>> recwarn = getfixture('recwarn')
    >>> dl = DeprecatedList(range(3))
    >>> dl[0] = 1
    >>> dl.append(3)
    >>> del dl[3]
    >>> dl.reverse()
    >>> dl.sort()
    >>> dl.extend([4])
    >>> dl.pop(-1)
    4
    >>> dl.remove(1)
    >>> dl += [5]
    >>> dl + [6]
    [1, 2, 5, 6]
    >>> dl + (6,)
    [1, 2, 5, 6]
    >>> dl.insert(0, 0)
    >>> dl
    [0, 1, 2, 5]
    >>> dl == [0, 1, 2, 5]
    True
    >>> dl == (0, 1, 2, 5)
    True
    >>> len(recwarn)
    1
    """

    __slots__ = ()

    _warn = functools.partial(
        warnings.warn,
        "EntryPoints list interface is deprecated. Cast to list if needed.",
        DeprecationWarning,
        stacklevel=2,
    )

    def __setitem__(self, *args, **kwargs):
        self._warn()
        return super().__setitem__(*args, **kwargs)

    def __delitem__(self, *args, **kwargs):
        self._warn()
        return super().__delitem__(*args, **kwargs)

    def append(self, *args, **kwargs):
        self._warn()
        return super().append(*args, **kwargs)

    def reverse(self, *args, **kwargs):
        self._warn()
        return super().reverse(*args, **kwargs)

    def extend(self, *args, **kwargs):
        self._warn()
        return super().extend(*args, **kwargs)

    def pop(self, *args, **kwargs):
        self._warn()
        return super().pop(*args, **kwargs)

    def remove(self, *args, **kwargs):
        self._warn()
        return super().remove(*args, **kwargs)

    def __iadd__(self, *args, **kwargs):
        self._warn()
        return super().__iadd__(*args, **kwargs)

    def __add__(self, other):
        if not isinstance(other, tuple):
            self._warn()
            other = tuple(other)
        return self.__class__(tuple(self) + other)

    def insert(self, *args, **kwargs):
        self._warn()
        return super().insert(*args, **kwargs)

    def sort(self, *args, **kwargs):
        self._warn()
        return super().sort(*args, **kwargs)

    def __eq__(self, other):
        if not isinstance(other, tuple):
            self._warn()
            other = tuple(other)

        return tuple(self).__eq__(other)


class EntryPoints(DeprecatedList):
    """
    An immutable collection of selectable EntryPoint objects.
    """

    __slots__ = ()

    def __getitem__(self, name):  # -> EntryPoint:
        """
        Get the EntryPoint in self matching name.
        """
        if isinstance(name, int):
            warnings.warn(
                "Accessing entry points by index is deprecated. "
                "Cast to tuple if needed.",
                DeprecationWarning,
                stacklevel=2,
            )
            return super().__getitem__(name)
        try:
            return next(iter(self.select(name=name)))
        except StopIteration:
            raise KeyError(name)

    def select(self, **params):
        """
        Select entry points from self that match the
        given parameters (typically group and/or name).
        """
        return EntryPoints(ep for ep in self if ep.matches(**params))

    @property
    def names(self):
        """
        Return the set of all names of all entry points.
        """
        return set(ep.name for ep in self)

    @property
    def groups(self):
        """
        Return the set of all groups of all entry points.

        For coverage while SelectableGroups is present.
        >>> EntryPoints().groups
        set()
        """
        return set(ep.group for ep in self)

    @classmethod
    def _from_text_for(cls, text, dist):
        return cls(ep._for(dist) for ep in cls._from_text(text))

    @classmethod
    def _from_text(cls, text):
        return itertools.starmap(EntryPoint, cls._parse_groups(text or ''))

    @staticmethod
    def _parse_groups(text):
        return (
            (item.value.name, item.value.value, item.name)
            for item in Sectioned.section_pairs(text)
        )


class Deprecated:
    """
    Compatibility add-in for mapping to indicate that
    mapping behavior is deprecated.

    >>> recwarn = getfixture('recwarn')
    >>> class DeprecatedDict(Deprecated, dict): pass
    >>> dd = DeprecatedDict(foo='bar')
    >>> dd.get('baz', None)
    >>> dd['foo']
    'bar'
    >>> list(dd)
    ['foo']
    >>> list(dd.keys())
    ['foo']
    >>> 'foo' in dd
    True
    >>> list(dd.values())
    ['bar']
    >>> len(recwarn)
    1
    """

    _warn = functools.partial(
        warnings.warn,
        "SelectableGroups dict interface is deprecated. Use select.",
        DeprecationWarning,
        stacklevel=2,
    )

    def __getitem__(self, name):
        self._warn()
        return super().__getitem__(name)

    def get(self, name, default=None):
        self._warn()
        return super().get(name, default)

    def __iter__(self):
        self._warn()
        return super().__iter__()

    def __contains__(self, *args):
        self._warn()
        return super().__contains__(*args)

    def keys(self):
        self._warn()
        return super().keys()

    def values(self):
        self._warn()
        return super().values()


class SelectableGroups(Deprecated, dict):
    """
    A backward- and forward-compatible result from
    entry_points that fully implements the dict interface.
    """

    @classmethod
    def load(cls, eps):
        by_group = operator.attrgetter('group')
        ordered = sorted(eps, key=by_group)
        grouped = itertools.groupby(ordered, by_group)
        return cls((group, EntryPoints(eps)) for group, eps in grouped)

    @property
    def _all(self):
        """
        Reconstruct a list of all entrypoints from the groups.
        """
        groups = super(Deprecated, self).values()
        return EntryPoints(itertools.chain.from_iterable(groups))

    @property
    def groups(self):
        return self._all.groups

    @property
    def names(self):
        """
        for coverage:
        >>> SelectableGroups().names
        set()
        """
        return self._all.names

    def select(self, **params):
        if not params:
            return self
        return self._all.select(**params)


class PackagePath(pathlib.PurePosixPath):
    """A reference to a path in a package"""

    def read_text(self, encoding='utf-8'):
        with self.locate().open(encoding=encoding) as stream:
            return stream.read()

    def read_binary(self):
        with self.locate().open('rb') as stream:
            return stream.read()

    def locate(self):
        """Return a path-like object for this path"""
        return self.dist.locate_file(self)


class FileHash:
    def __init__(self, spec):
        self.mode, _, self.value = spec.partition('=')

    def __repr__(self):
        return f'<FileHash mode: {self.mode} value: {self.value}>'


class Distribution:
    """A Python distribution package."""

    @abc.abstractmethod
    def read_text(self, filename):
        """Attempt to load metadata file given by the name.

        :param filename: The name of the file in the distribution info.
        :return: The text if found, otherwise None.
        """

    @abc.abstractmethod
    def locate_file(self, path):
        """
        Given a path to a file in this distribution, return a path
        to it.
        """

    @classmethod
    def from_name(cls, name):
        """Return the Distribution for the given package name.

        :param name: The name of the distribution package to search for.
        :return: The Distribution instance (or subclass thereof) for the named
            package, if found.
        :raises PackageNotFoundError: When the named package's distribution
            metadata cannot be found.
        """
        for resolver in cls._discover_resolvers():
            dists = resolver(DistributionFinder.Context(name=name))
            dist = next(iter(dists), None)
            if dist is not None:
                return dist
        else:
            raise PackageNotFoundError(name)

    @classmethod
    def discover(cls, **kwargs):
        """Return an iterable of Distribution objects for all packages.

        Pass a ``context`` or pass keyword arguments for constructing
        a context.

        :context: A ``DistributionFinder.Context`` object.
        :return: Iterable of Distribution objects for all packages.
        """
        context = kwargs.pop('context', None)
        if context and kwargs:
            raise ValueError("cannot accept context and kwargs")
        context = context or DistributionFinder.Context(**kwargs)
        return itertools.chain.from_iterable(
            resolver(context) for resolver in cls._discover_resolvers()
        )

    @staticmethod
    def at(path):
        """Return a Distribution for the indicated metadata path

        :param path: a string or path-like object
        :return: a concrete Distribution instance for the path
        """
        return PathDistribution(pathlib.Path(path))

    @staticmethod
    def _discover_resolvers():
        """Search the meta_path for resolvers."""
        declared = (
            getattr(finder, 'find_distributions', None) for finder in sys.meta_path
        )
        return filter(None, declared)

    @classmethod
    def _local(cls, root='.'):
        from pep517 import build, meta

        system = build.compat_system(root)
        builder = functools.partial(
            meta.build,
            source_dir=root,
            system=system,
        )
        return PathDistribution(zipfile.Path(meta.build_as_zip(builder)))

    @property
    def metadata(self) -> _meta.PackageMetadata:
        """Return the parsed metadata for this Distribution.

        The returned object will have keys that name the various bits of
        metadata.  See PEP 566 for details.
        """
        text = (
            self.read_text('METADATA')
            or self.read_text('PKG-INFO')
            # This last clause is here to support old egg-info files.  Its
            # effect is to just end up using the PathDistribution's self._path
            # (which points to the egg-info file) attribute unchanged.
            or self.read_text('')
        )
        return _adapters.Message(email.message_from_string(text))

    @property
    def name(self):
        """Return the 'Name' metadata for the distribution package."""
        return self.metadata['Name']

    @property
    def _normalized_name(self):
        """Return a normalized version of the name."""
        return Prepared.normalize(self.name)

    @property
    def version(self):
        """Return the 'Version' metadata for the distribution package."""
        return self.metadata['Version']

    @property
    def entry_points(self):
        return EntryPoints._from_text_for(self.read_text('entry_points.txt'), self)

    @property
    def files(self):
        """Files in this distribution.

        :return: List of PackagePath for this distribution or None

        Result is `None` if the metadata file that enumerates files
        (i.e. RECORD for dist-info or SOURCES.txt for egg-info) is
        missing.
        Result may be empty if the metadata exists but is empty.
        """
        file_lines = self._read_files_distinfo() or self._read_files_egginfo()

        def make_file(name, hash=None, size_str=None):
            result = PackagePath(name)
            result.hash = FileHash(hash) if hash else None
            result.size = int(size_str) if size_str else None
            result.dist = self
            return result

        return file_lines and list(starmap(make_file, csv.reader(file_lines)))

    def _read_files_distinfo(self):
        """
        Read the lines of RECORD
        """
        text = self.read_text('RECORD')
        return text and text.splitlines()

    def _read_files_egginfo(self):
        """
        SOURCES.txt might contain literal commas, so wrap each line
        in quotes.
        """
        text = self.read_text('SOURCES.txt')
        return text and map('"{}"'.format, text.splitlines())

    @property
    def requires(self):
        """Generated requirements specified for this Distribution"""
        reqs = self._read_dist_info_reqs() or self._read_egg_info_reqs()
        return reqs and list(reqs)

    def _read_dist_info_reqs(self):
        return self.metadata.get_all('Requires-Dist')

    def _read_egg_info_reqs(self):
        source = self.read_text('requires.txt')
        return None if source is None else self._deps_from_requires_text(source)

    @classmethod
    def _deps_from_requires_text(cls, source):
        return cls._convert_egg_info_reqs_to_simple_reqs(Sectioned.read(source))

    @staticmethod
    def _convert_egg_info_reqs_to_simple_reqs(sections):
        """
        Historically, setuptools would solicit and store 'extra'
        requirements, including those with environment markers,
        in separate sections. More modern tools expect each
        dependency to be defined separately, with any relevant
        extras and environment markers attached directly to that
        requirement. This method converts the former to the
        latter. See _test_deps_from_requires_text for an example.
        """

        def make_condition(name):
            return name and f'extra == "{name}"'

        def quoted_marker(section):
            section = section or ''
            extra, sep, markers = section.partition(':')
            if extra and markers:
                markers = f'({markers})'
            conditions = list(filter(None, [markers, make_condition(extra)]))
            return '; ' + ' and '.join(conditions) if conditions else ''

        def url_req_space(req):
            """
            PEP 508 requires a space between the url_spec and the quoted_marker.
            Ref python/importlib_metadata#357.
            """
            # '@' is uniquely indicative of a url_req.
            return ' ' * ('@' in req)

        for section in sections:
            space = url_req_space(section.value)
            yield section.value + space + quoted_marker(section.name)


class DistributionFinder(MetaPathFinder):
    """
    A MetaPathFinder capable of discovering installed distributions.
    """

    class Context:
        """
        Keyword arguments presented by the caller to
        ``distributions()`` or ``Distribution.discover()``
        to narrow the scope of a search for distributions
        in all DistributionFinders.

        Each DistributionFinder may expect any parameters
        and should attempt to honor the canonical
        parameters defined below when appropriate.
        """

        name = None
        """
        Specific name for which a distribution finder should match.
        A name of ``None`` matches all distributions.
        """

        def __init__(self, **kwargs):
            vars(self).update(kwargs)

        @property
        def path(self):
            """
            The sequence of directory path that a distribution finder
            should search.

            Typically refers to Python installed package paths such as
            "site-packages" directories and defaults to ``sys.path``.
            """
            return vars(self).get('path', sys.path)

    @abc.abstractmethod
    def find_distributions(self, context=Context()):
        """
        Find distributions.

        Return an iterable of all Distribution instances capable of
        loading the metadata for packages matching the ``context``,
        a DistributionFinder.Context instance.
        """


class FastPath:
    """
    Micro-optimized class for searching a path for
    children.
    """

    @functools.lru_cache()  # type: ignore
    def __new__(cls, root):
        return super().__new__(cls)

    def __init__(self, root):
        self.root = root

    def joinpath(self, child):
        return pathlib.Path(self.root, child)

    def children(self):
        with suppress(Exception):
            return os.listdir(self.root or '.')
        with suppress(Exception):
            return self.zip_children()
        return []

    def zip_children(self):
        zip_path = zipfile.Path(self.root)
        names = zip_path.root.namelist()
        self.joinpath = zip_path.joinpath

        return dict.fromkeys(child.split(posixpath.sep, 1)[0] for child in names)

    def search(self, name):
        return self.lookup(self.mtime).search(name)

    @property
    def mtime(self):
        with suppress(OSError):
            return os.stat(self.root).st_mtime
        self.lookup.cache_clear()

    @method_cache
    def lookup(self, mtime):
        return Lookup(self)


class Lookup:
    def __init__(self, path: FastPath):
        base = os.path.basename(path.root).lower()
        base_is_egg = base.endswith(".egg")
        self.infos = FreezableDefaultDict(list)
        self.eggs = FreezableDefaultDict(list)

        for child in path.children():
            low = child.lower()
            if low.endswith((".dist-info", ".egg-info")):
                # rpartition is faster than splitext and suitable for this purpose.
                name = low.rpartition(".")[0].partition("-")[0]
                normalized = Prepared.normalize(name)
                self.infos[normalized].append(path.joinpath(child))
            elif base_is_egg and low == "egg-info":
                name = base.rpartition(".")[0].partition("-")[0]
                legacy_normalized = Prepared.legacy_normalize(name)
                self.eggs[legacy_normalized].append(path.joinpath(child))

        self.infos.freeze()
        self.eggs.freeze()

    def search(self, prepared):
        infos = (
            self.infos[prepared.normalized]
            if prepared
            else itertools.chain.from_iterable(self.infos.values())
        )
        eggs = (
            self.eggs[prepared.legacy_normalized]
            if prepared
            else itertools.chain.from_iterable(self.eggs.values())
        )
        return itertools.chain(infos, eggs)


class Prepared:
    """
    A prepared search for metadata on a possibly-named package.
    """

    normalized = None
    legacy_normalized = None

    def __init__(self, name):
        self.name = name
        if name is None:
            return
        self.normalized = self.normalize(name)
        self.legacy_normalized = self.legacy_normalize(name)

    @staticmethod
    def normalize(name):
        """
        PEP 503 normalization plus dashes as underscores.
        """
        return re.sub(r"[-_.]+", "-", name).lower().replace('-', '_')

    @staticmethod
    def legacy_normalize(name):
        """
        Normalize the package name as found in the convention in
        older packaging tools versions and specs.
        """
        return name.lower().replace('-', '_')

    def __bool__(self):
        return bool(self.name)


class MetadataPathFinder(DistributionFinder):
    @classmethod
    def find_distributions(cls, context=DistributionFinder.Context()):
        """
        Find distributions.

        Return an iterable of all Distribution instances capable of
        loading the metadata for packages matching ``context.name``
        (or all names if ``None`` indicated) along the paths in the list
        of directories ``context.path``.
        """
        found = cls._search_paths(context.name, context.path)
        return map(PathDistribution, found)

    @classmethod
    def _search_paths(cls, name, paths):
        """Find metadata directories in paths heuristically."""
        prepared = Prepared(name)
        return itertools.chain.from_iterable(
            path.search(prepared) for path in map(FastPath, paths)
        )

    def invalidate_caches(cls):
        FastPath.__new__.cache_clear()


class PathDistribution(Distribution):
    def __init__(self, path: SimplePath):
        """Construct a distribution.

        :param path: SimplePath indicating the metadata directory.
        """
        self._path = path

    def read_text(self, filename):
        with suppress(
            FileNotFoundError,
            IsADirectoryError,
            KeyError,
            NotADirectoryError,
            PermissionError,
        ):
            return self._path.joinpath(filename).read_text(encoding='utf-8')

    read_text.__doc__ = Distribution.read_text.__doc__

    def locate_file(self, path):
        return self._path.parent / path

    @property
    def _normalized_name(self):
        """
        Performance optimization: where possible, resolve the
        normalized name from the file system path.
        """
        stem = os.path.basename(str(self._path))
        return (
            pass_none(Prepared.normalize)(self._name_from_stem(stem))
            or super()._normalized_name
        )

    @staticmethod
    def _name_from_stem(stem):
        """
        >>> PathDistribution._name_from_stem('foo-3.0.egg-info')
        'foo'
        >>> PathDistribution._name_from_stem('CherryPy-3.0.dist-info')
        'CherryPy'
        >>> PathDistribution._name_from_stem('face.egg-info')
        'face'
        """
        filename, ext = os.path.splitext(stem)
        if ext not in ('.dist-info', '.egg-info'):
            return
        name, sep, rest = filename.partition('-')
        return name


def distribution(distribution_name):
    """Get the ``Distribution`` instance for the named package.

    :param distribution_name: The name of the distribution package as a string.
    :return: A ``Distribution`` instance (or subclass thereof).
    """
    return Distribution.from_name(distribution_name)


def distributions(**kwargs):
    """Get all ``Distribution`` instances in the current environment.

    :return: An iterable of ``Distribution`` instances.
    """
    return Distribution.discover(**kwargs)


def metadata(distribution_name) -> _meta.PackageMetadata:
    """Get the metadata for the named package.

    :param distribution_name: The name of the distribution package to query.
    :return: A PackageMetadata containing the parsed metadata.
    """
    return Distribution.from_name(distribution_name).metadata


def version(distribution_name):
    """Get the version string for the named package.

    :param distribution_name: The name of the distribution package to query.
    :return: The version string for the package as defined in the package's
        "Version" metadata key.
    """
    return distribution(distribution_name).version


def entry_points(**params) -> Union[EntryPoints, SelectableGroups]:
    """Return EntryPoint objects for all installed packages.

    Pass selection parameters (group or name) to filter the
    result to entry points matching those properties (see
    EntryPoints.select()).

    For compatibility, returns ``SelectableGroups`` object unless
    selection parameters are supplied. In the future, this function
    will return ``EntryPoints`` instead of ``SelectableGroups``
    even when no selection parameters are supplied.

    For maximum future compatibility, pass selection parameters
    or invoke ``.select`` with parameters on the result.

    :return: EntryPoints or SelectableGroups for all installed packages.
    """
    norm_name = operator.attrgetter('_normalized_name')
    unique = functools.partial(unique_everseen, key=norm_name)
    eps = itertools.chain.from_iterable(
        dist.entry_points for dist in unique(distributions())
    )
    return SelectableGroups.load(eps).select(**params)


def files(distribution_name):
    """Return a list of files for the named package.

    :param distribution_name: The name of the distribution package to query.
    :return: List of files composing the distribution.
    """
    return distribution(distribution_name).files


def requires(distribution_name):
    """
    Return a list of requirements for the named package.

    :return: An iterator of requirements, suitable for
        packaging.requirement.Requirement.
    """
    return distribution(distribution_name).requires


def packages_distributions() -> Mapping[str, List[str]]:
    """
    Return a mapping of top-level packages to their
    distributions.

    >>> import collections.abc
    >>> pkgs = packages_distributions()
    >>> all(isinstance(dist, collections.abc.Sequence) for dist in pkgs.values())
    True
    """
    pkg_to_dist = collections.defaultdict(list)
    for dist in distributions():
        for pkg in (dist.read_text('top_level.txt') or '').split():
            pkg_to_dist[pkg].append(dist.metadata['Name'])
    return dict(pkg_to_dist)
<|endoftext|>import re
import textwrap
import email.message

from ._text import FoldedCase


class Message(email.message.Message):
    multiple_use_keys = set(
        map(
            FoldedCase,
            [
                'Classifier',
                'Obsoletes-Dist',
                'Platform',
                'Project-URL',
                'Provides-Dist',
                'Provides-Extra',
                'Requires-Dist',
                'Requires-External',
                'Supported-Platform',
                'Dynamic',
            ],
        )
    )
    """
    Keys that may be indicated multiple times per PEP 566.
    """

    def __new__(cls, orig: email.message.Message):
        res = super().__new__(cls)
        vars(res).update(vars(orig))
        return res

    def __init__(self, *args, **kwargs):
        self._headers = self._repair_headers()

    # suppress spurious error from mypy
    def __iter__(self):
        return super().__iter__()

    def _repair_headers(self):
        def redent(value):
            "Correct for RFC822 indentation"
            if not value or '\n' not in value:
                return value
            return textwrap.dedent(' ' * 8 + value)

        headers = [(key, redent(value)) for key, value in vars(self)['_headers']]
        if self._payload:
            headers.append(('Description', self.get_payload()))
        return headers

    @property
    def json(self):
        """
        Convert PackageMetadata to a JSON-compatible format
        per PEP 0566.
        """

        def transform(key):
            value = self.get_all(key) if key in self.multiple_use_keys else self[key]
            if key == 'Keywords':
                value = re.split(r'\s+', value)
            tk = key.lower().replace('-', '_')
            return tk, value

        return dict(map(transform, map(FoldedCase, self)))
<|endoftext|>from itertools import filterfalse


def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element
<|endoftext|>import types
import functools


# from jaraco.functools 3.3
def method_cache(method, cache_wrapper=None):
    """
    Wrap lru_cache to support storing the cache data in the object instances.

    Abstracts the common paradigm where the method explicitly saves an
    underscore-prefixed protected property on first call and returns that
    subsequently.

    >>> class MyClass:
    ...     calls = 0
    ...
    ...     @method_cache
    ...     def method(self, value):
    ...         self.calls += 1
    ...         return value

    >>> a = MyClass()
    >>> a.method(3)
    3
    >>> for x in range(75):
    ...     res = a.method(x)
    >>> a.calls
    75

    Note that the apparent behavior will be exactly like that of lru_cache
    except that the cache is stored on each instance, so values in one
    instance will not flush values from another, and when an instance is
    deleted, so are the cached values for that instance.

    >>> b = MyClass()
    >>> for x in range(35):
    ...     res = b.method(x)
    >>> b.calls
    35
    >>> a.method(0)
    0
    >>> a.calls
    75

    Note that if method had been decorated with ``functools.lru_cache()``,
    a.calls would have been 76 (due to the cached value of 0 having been
    flushed by the 'b' instance).

    Clear the cache with ``.cache_clear()``

    >>> a.method.cache_clear()

    Same for a method that hasn't yet been called.

    >>> c = MyClass()
    >>> c.method.cache_clear()

    Another cache wrapper may be supplied:

    >>> cache = functools.lru_cache(maxsize=2)
    >>> MyClass.method2 = method_cache(lambda self: 3, cache_wrapper=cache)
    >>> a = MyClass()
    >>> a.method2()
    3

    Caution - do not subsequently wrap the method with another decorator, such
    as ``@property``, which changes the semantics of the function.

    See also
    http://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/
    for another implementation and additional justification.
    """
    cache_wrapper = cache_wrapper or functools.lru_cache()

    def wrapper(self, *args, **kwargs):
        # it's the first call, replace the method with a cached, bound method
        bound_method = types.MethodType(method, self)
        cached_method = cache_wrapper(bound_method)
        setattr(self, method.__name__, cached_method)
        return cached_method(*args, **kwargs)

    # Support cache clear even before cache has been created.
    wrapper.cache_clear = lambda: None

    return wrapper


# From jaraco.functools 3.3
def pass_none(func):
    """
    Wrap func so it's not called if its first param is None

    >>> print_text = pass_none(print)
    >>> print_text('text')
    text
    >>> print_text(None)
    """

    @functools.wraps(func)
    def wrapper(param, *args, **kwargs):
        if param is not None:
            return func(param, *args, **kwargs)

    return wrapper
<|endoftext|>import os
import shutil
import subprocess
import sys

# find_library(name) returns the pathname of a library, or None.
if os.name == "nt":

    def _get_build_version():
        """Return the version of MSVC that was used to build Python.

        For Python 2.3 and up, the version number is included in
        sys.version.  For earlier versions, assume the compiler is MSVC 6.
        """
        # This function was copied from Lib/distutils/msvccompiler.py
        prefix = "MSC v."
        i = sys.version.find(prefix)
        if i == -1:
            return 6
        i = i + len(prefix)
        s, rest = sys.version[i:].split(" ", 1)
        majorVersion = int(s[:-2]) - 6
        if majorVersion >= 13:
            majorVersion += 1
        minorVersion = int(s[2:3]) / 10.0
        # I don't think paths are affected by minor version in version 6
        if majorVersion == 6:
            minorVersion = 0
        if majorVersion >= 6:
            return majorVersion + minorVersion
        # else we don't know what version of the compiler this is
        return None

    def find_msvcrt():
        """Return the name of the VC runtime dll"""
        version = _get_build_version()
        if version is None:
            # better be safe than sorry
            return None
        if version <= 6:
            clibname = 'msvcrt'
        elif version <= 13:
            clibname = 'msvcr%d' % (version * 10)
        else:
            # CRT is no longer directly loadable. See issue23606 for the
            # discussion about alternative approaches.
            return None

        # If python was built with in debug mode
        import importlib.machinery
        if '_d.pyd' in importlib.machinery.EXTENSION_SUFFIXES:
            clibname += 'd'
        return clibname+'.dll'

    def find_library(name):
        if name in ('c', 'm'):
            return find_msvcrt()
        # See MSDN for the REAL search order.
        for directory in os.environ['PATH'].split(os.pathsep):
            fname = os.path.join(directory, name)
            if os.path.isfile(fname):
                return fname
            if fname.lower().endswith(".dll"):
                continue
            fname = fname + ".dll"
            if os.path.isfile(fname):
                return fname
        return None

elif os.name == "posix" and sys.platform == "darwin":
    from ctypes.macholib.dyld import dyld_find as _dyld_find
    def find_library(name):
        possible = ['lib%s.dylib' % name,
                    '%s.dylib' % name,
                    '%s.framework/%s' % (name, name)]
        for name in possible:
            try:
                return _dyld_find(name)
            except ValueError:
                continue
        return None

elif sys.platform.startswith("aix"):
    # AIX has two styles of storing shared libraries
    # GNU auto_tools refer to these as svr4 and aix
    # svr4 (System V Release 4) is a regular file, often with .so as suffix
    # AIX style uses an archive (suffix .a) with members (e.g., shr.o, libssl.so)
    # see issue#26439 and _aix.py for more details

    from ctypes._aix import find_library

elif os.name == "posix":
    # Andreas Degert's find functions, using gcc, /sbin/ldconfig, objdump
    import re, tempfile

    def _is_elf(filename):
        "Return True if the given file is an ELF file"
        elf_header = b'\x7fELF'
        with open(filename, 'br') as thefile:
            return thefile.read(4) == elf_header

    def _findLib_gcc(name):
        # Run GCC's linker with the -t (aka --trace) option and examine the
        # library name it prints out. The GCC command will fail because we
        # haven't supplied a proper program with main(), but that does not
        # matter.
        expr = os.fsencode(r'[^\(\)\s]*lib%s\.[^\(\)\s]*' % re.escape(name))

        c_compiler = shutil.which('gcc')
        if not c_compiler:
            c_compiler = shutil.which('cc')
        if not c_compiler:
            # No C compiler available, give up
            return None

        temp = tempfile.NamedTemporaryFile()
        try:
            args = [c_compiler, '-Wl,-t', '-o', temp.name, '-l' + name]

            env = dict(os.environ)
            env['LC_ALL'] = 'C'
            env['LANG'] = 'C'
            try:
                proc = subprocess.Popen(args,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        env=env)
            except OSError:  # E.g. bad executable
                return None
            with proc:
                trace = proc.stdout.read()
        finally:
            try:
                temp.close()
            except FileNotFoundError:
                # Raised if the file was already removed, which is the normal
                # behaviour of GCC if linking fails
                pass
        res = re.findall(expr, trace)
        if not res:
            return None

        for file in res:
            # Check if the given file is an elf file: gcc can report
            # some files that are linker scripts and not actual
            # shared objects. See bpo-41976 for more details
            if not _is_elf(file):
                continue
            return os.fsdecode(file)


    if sys.platform == "sunos5":
        # use /usr/ccs/bin/dump on solaris
        def _get_soname(f):
            if not f:
                return None

            try:
                proc = subprocess.Popen(("/usr/ccs/bin/dump", "-Lpv", f),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
            except OSError:  # E.g. command not found
                return None
            with proc:
                data = proc.stdout.read()
            res = re.search(br'\[.*\]\sSONAME\s+([^\s]+)', data)
            if not res:
                return None
            return os.fsdecode(res.group(1))
    else:
        def _get_soname(f):
            # assuming GNU binutils / ELF
            if not f:
                return None
            objdump = shutil.which('objdump')
            if not objdump:
                # objdump is not available, give up
                return None

            try:
                proc = subprocess.Popen((objdump, '-p', '-j', '.dynamic', f),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
            except OSError:  # E.g. bad executable
                return None
            with proc:
                dump = proc.stdout.read()
            res = re.search(br'\sSONAME\s+([^\s]+)', dump)
            if not res:
                return None
            return os.fsdecode(res.group(1))

    if sys.platform.startswith(("freebsd", "openbsd", "dragonfly")):

        def _num_version(libname):
            # "libxyz.so.MAJOR.MINOR" => [ MAJOR, MINOR ]
            parts = libname.split(b".")
            nums = []
            try:
                while parts:
                    nums.insert(0, int(parts.pop()))
            except ValueError:
                pass
            return nums or [sys.maxsize]

        def find_library(name):
            ename = re.escape(name)
            expr = r':-l%s\.\S+ => \S*/(lib%s\.\S+)' % (ename, ename)
            expr = os.fsencode(expr)

            try:
                proc = subprocess.Popen(('/sbin/ldconfig', '-r'),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
            except OSError:  # E.g. command not found
                data = b''
            else:
                with proc:
                    data = proc.stdout.read()

            res = re.findall(expr, data)
            if not res:
                return _get_soname(_findLib_gcc(name))
            res.sort(key=_num_version)
            return os.fsdecode(res[-1])

    elif sys.platform == "sunos5":

        def _findLib_crle(name, is64):
            if not os.path.exists('/usr/bin/crle'):
                return None

            env = dict(os.environ)
            env['LC_ALL'] = 'C'

            if is64:
                args = ('/usr/bin/crle', '-64')
            else:
                args = ('/usr/bin/crle',)

            paths = None
            try:
                proc = subprocess.Popen(args,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL,
                                        env=env)
            except OSError:  # E.g. bad executable
                return None
            with proc:
                for line in proc.stdout:
                    line = line.strip()
                    if line.startswith(b'Default Library Path (ELF):'):
                        paths = os.fsdecode(line).split()[4]

            if not paths:
                return None

            for dir in paths.split(":"):
                libfile = os.path.join(dir, "lib%s.so" % name)
                if os.path.exists(libfile):
                    return libfile

            return None

        def find_library(name, is64 = False):
            return _get_soname(_findLib_crle(name, is64) or _findLib_gcc(name))

    else:

        def _findSoname_ldconfig(name):
            import struct
            # XXX this code assumes that we know all unames and that a single
            # ABI is supported per uname; instead we should find what the
            # ABI is (e.g. check ABI of current process) or simply ask libc
            # to load the library for us
            uname = os.uname()
            # ARM has a variety of unames, e.g. armv7l
            if uname.machine.startswith("arm"):
                machine = "arm"
            if struct.calcsize('l') == 4:
                machine = uname.machine + '-32'
            else:
                machine = uname.machine + '-64'
            mach_map = {
                'x86_64-64': 'libc6,x86-64',
                'ppc64-64': 'libc6,64bit',
                'sparc64-64': 'libc6,64bit',
                's390x-64': 'libc6,64bit',
                'ia64-64': 'libc6,IA-64',
                # this actually breaks on biarch or multiarch as the first
                # library wins; uname doesn't tell us which ABI we're using
                'arm-32': 'libc6(,hard-float)?',
                }
            abi_type = mach_map.get(machine, 'libc6')

            # XXX assuming GLIBC's ldconfig (with option -p)
            regex = r'\s+(lib%s\.[^\s]+)\s+\(%s'
            regex = os.fsencode(regex % (re.escape(name), abi_type))
            try:
                with subprocess.Popen(['/sbin/ldconfig', '-p'],
                                      stdin=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE,
                                      env={'LC_ALL': 'C', 'LANG': 'C'}) as p:
                    res = re.search(regex, p.stdout.read())
                    if res:
                        return os.fsdecode(res.group(1))
            except OSError:
                pass

        def _findLib_ld(name):
            # See issue #9998 for why this is needed
            expr = r'[^\(\)\s]*lib%s\.[^\(\)\s]*' % re.escape(name)
            cmd = ['ld', '-t']
            libpath = os.environ.get('LD_LIBRARY_PATH')
            if libpath:
                for d in libpath.split(':'):
                    cmd.extend(['-L', d])
            cmd.extend(['-o', os.devnull, '-l%s' % name])
            result = None
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
                out, _ = p.communicate()
                res = re.findall(expr, os.fsdecode(out))
                for file in res:
                    # Check if the given file is an elf file: gcc can report
                    # some files that are linker scripts and not actual
                    # shared objects. See bpo-41976 for more details
                    if not _is_elf(file):
                        continue
                    return os.fsdecode(file)
            except Exception:
                pass  # result will be None
            return result

        def find_library(name):
            # See issue #9998
            return _findSoname_ldconfig(name) or \
                   _get_soname(_findLib_gcc(name)) or _get_soname(_findLib_ld(name))

################################################################
# test code

def test():
    from ctypes import cdll
    if os.name == "nt":
        print(cdll.msvcrt)
        print(cdll.load("msvcrt"))
        print(find_library("msvcrt"))

    if os.name == "posix":
        # find and load_version
        print(find_library("m"))
        print(find_library("c"))
        print(find_library("bz2"))

        # load
        if sys.platform == "darwin":
            print(cdll.LoadLibrary("libm.dylib"))
            print(cdll.LoadLibrary("libcrypto.dylib"))
            print(cdll.LoadLibrary("libSystem.dylib"))
            print(cdll.LoadLibrary("System.framework/System"))
        # issue-26439 - fix broken test call for AIX
        elif sys.platform.startswith("aix"):
            from ctypes import CDLL
            if sys.maxsize < 2**32:
                print(f"Using CDLL(name, os.RTLD_MEMBER): {CDLL('libc.a(shr.o)', os.RTLD_MEMBER)}")
                print(f"Using cdll.LoadLibrary(): {cdll.LoadLibrary('libc.a(shr.o)')}")
                # librpm.so is only available as 32-bit shared library
                print(find_library("rpm"))
                print(cdll.LoadLibrary("librpm.so"))
            else:
                print(f"Using CDLL(name, os.RTLD_MEMBER): {CDLL('libc.a(shr_64.o)', os.RTLD_MEMBER)}")
                print(f"Using cdll.LoadLibrary(): {cdll.LoadLibrary('libc.a(shr_64.o)')}")
            print(f"crypt\t:: {find_library('crypt')}")
            print(f"crypt\t:: {cdll.LoadLibrary(find_library('crypt'))}")
            print(f"crypto\t:: {find_library('crypto')}")
            print(f"crypto\t:: {cdll.LoadLibrary(find_library('crypto'))}")
        else:
            print(cdll.LoadLibrary("libm.so"))
            print(cdll.LoadLibrary("libcrypt.so"))
            print(find_library("crypt"))

if __name__ == "__main__":
    test()
<|endoftext|>"""create and manipulate C data types in Python"""

import os as _os, sys as _sys
import types as _types

__version__ = "1.1.0"

from _ctypes import Union, Structure, Array
from _ctypes import _Pointer
from _ctypes import CFuncPtr as _CFuncPtr
from _ctypes import __version__ as _ctypes_version
from _ctypes import RTLD_LOCAL, RTLD_GLOBAL
from _ctypes import ArgumentError

from struct import calcsize as _calcsize

if __version__ != _ctypes_version:
    raise Exception("Version number mismatch", __version__, _ctypes_version)

if _os.name == "nt":
    from _ctypes import FormatError

DEFAULT_MODE = RTLD_LOCAL
if _os.name == "posix" and _sys.platform == "darwin":
    # On OS X 10.3, we use RTLD_GLOBAL as default mode
    # because RTLD_LOCAL does not work at least on some
    # libraries.  OS X 10.3 is Darwin 7, so we check for
    # that.

    if int(_os.uname().release.split('.')[0]) < 8:
        DEFAULT_MODE = RTLD_GLOBAL

from _ctypes import FUNCFLAG_CDECL as _FUNCFLAG_CDECL, \
     FUNCFLAG_PYTHONAPI as _FUNCFLAG_PYTHONAPI, \
     FUNCFLAG_USE_ERRNO as _FUNCFLAG_USE_ERRNO, \
     FUNCFLAG_USE_LASTERROR as _FUNCFLAG_USE_LASTERROR

# WINOLEAPI -> HRESULT
# WINOLEAPI_(type)
#
# STDMETHODCALLTYPE
#
# STDMETHOD(name)
# STDMETHOD_(type, name)
#
# STDAPICALLTYPE

def create_string_buffer(init, size=None):
    """create_string_buffer(aBytes) -> character array
    create_string_buffer(anInteger) -> character array
    create_string_buffer(aBytes, anInteger) -> character array
    """
    if isinstance(init, bytes):
        if size is None:
            size = len(init)+1
        _sys.audit("ctypes.create_string_buffer", init, size)
        buftype = c_char * size
        buf = buftype()
        buf.value = init
        return buf
    elif isinstance(init, int):
        _sys.audit("ctypes.create_string_buffer", None, init)
        buftype = c_char * init
        buf = buftype()
        return buf
    raise TypeError(init)

def c_buffer(init, size=None):
##    "deprecated, use create_string_buffer instead"
##    import warnings
##    warnings.warn("c_buffer is deprecated, use create_string_buffer instead",
##                  DeprecationWarning, stacklevel=2)
    return create_string_buffer(init, size)

_c_functype_cache = {}
def CFUNCTYPE(restype, *argtypes, **kw):
    """CFUNCTYPE(restype, *argtypes,
                 use_errno=False, use_last_error=False) -> function prototype.

    restype: the result type
    argtypes: a sequence specifying the argument types

    The function prototype can be called in different ways to create a
    callable object:

    prototype(integer address) -> foreign function
    prototype(callable) -> create and return a C callable function from callable
    prototype(integer index, method name[, paramflags]) -> foreign function calling a COM method
    prototype((ordinal number, dll object)[, paramflags]) -> foreign function exported by ordinal
    prototype((function name, dll object)[, paramflags]) -> foreign function exported by name
    """
    flags = _FUNCFLAG_CDECL
    if kw.pop("use_errno", False):
        flags |= _FUNCFLAG_USE_ERRNO
    if kw.pop("use_last_error", False):
        flags |= _FUNCFLAG_USE_LASTERROR
    if kw:
        raise ValueError("unexpected keyword argument(s) %s" % kw.keys())
    try:
        return _c_functype_cache[(restype, argtypes, flags)]
    except KeyError:
        class CFunctionType(_CFuncPtr):
            _argtypes_ = argtypes
            _restype_ = restype
            _flags_ = flags
        _c_functype_cache[(restype, argtypes, flags)] = CFunctionType
        return CFunctionType

if _os.name == "nt":
    from _ctypes import LoadLibrary as _dlopen
    from _ctypes import FUNCFLAG_STDCALL as _FUNCFLAG_STDCALL

    _win_functype_cache = {}
    def WINFUNCTYPE(restype, *argtypes, **kw):
        # docstring set later (very similar to CFUNCTYPE.__doc__)
        flags = _FUNCFLAG_STDCALL
        if kw.pop("use_errno", False):
            flags |= _FUNCFLAG_USE_ERRNO
        if kw.pop("use_last_error", False):
            flags |= _FUNCFLAG_USE_LASTERROR
        if kw:
            raise ValueError("unexpected keyword argument(s) %s" % kw.keys())
        try:
            return _win_functype_cache[(restype, argtypes, flags)]
        except KeyError:
            class WinFunctionType(_CFuncPtr):
                _argtypes_ = argtypes
                _restype_ = restype
                _flags_ = flags
            _win_functype_cache[(restype, argtypes, flags)] = WinFunctionType
            return WinFunctionType
    if WINFUNCTYPE.__doc__:
        WINFUNCTYPE.__doc__ = CFUNCTYPE.__doc__.replace("CFUNCTYPE", "WINFUNCTYPE")

elif _os.name == "posix":
    from _ctypes import dlopen as _dlopen

from _ctypes import sizeof, byref, addressof, alignment, resize
from _ctypes import get_errno, set_errno
from _ctypes import _SimpleCData

def _check_size(typ, typecode=None):
    # Check if sizeof(ctypes_type) against struct.calcsize.  This
    # should protect somewhat against a misconfigured libffi.
    from struct import calcsize
    if typecode is None:
        # Most _type_ codes are the same as used in struct
        typecode = typ._type_
    actual, required = sizeof(typ), calcsize(typecode)
    if actual != required:
        raise SystemError("sizeof(%s) wrong: %d instead of %d" % \
                          (typ, actual, required))

class py_object(_SimpleCData):
    _type_ = "O"
    def __repr__(self):
        try:
            return super().__repr__()
        except ValueError:
            return "%s(<NULL>)" % type(self).__name__
_check_size(py_object, "P")

class c_short(_SimpleCData):
    _type_ = "h"
_check_size(c_short)

class c_ushort(_SimpleCData):
    _type_ = "H"
_check_size(c_ushort)

class c_long(_SimpleCData):
    _type_ = "l"
_check_size(c_long)

class c_ulong(_SimpleCData):
    _type_ = "L"
_check_size(c_ulong)

if _calcsize("i") == _calcsize("l"):
    # if int and long have the same size, make c_int an alias for c_long
    c_int = c_long
    c_uint = c_ulong
else:
    class c_int(_SimpleCData):
        _type_ = "i"
    _check_size(c_int)

    class c_uint(_SimpleCData):
        _type_ = "I"
    _check_size(c_uint)

class c_float(_SimpleCData):
    _type_ = "f"
_check_size(c_float)

class c_double(_SimpleCData):
    _type_ = "d"
_check_size(c_double)

class c_longdouble(_SimpleCData):
    _type_ = "g"
if sizeof(c_longdouble) == sizeof(c_double):
    c_longdouble = c_double

if _calcsize("l") == _calcsize("q"):
    # if long and long long have the same size, make c_longlong an alias for c_long
    c_longlong = c_long
    c_ulonglong = c_ulong
else:
    class c_longlong(_SimpleCData):
        _type_ = "q"
    _check_size(c_longlong)

    class c_ulonglong(_SimpleCData):
        _type_ = "Q"
    ##    def from_param(cls, val):
    ##        return ('d', float(val), val)
    ##    from_param = classmethod(from_param)
    _check_size(c_ulonglong)

class c_ubyte(_SimpleCData):
    _type_ = "B"
c_ubyte.__ctype_le__ = c_ubyte.__ctype_be__ = c_ubyte
# backward compatibility:
##c_uchar = c_ubyte
_check_size(c_ubyte)

class c_byte(_SimpleCData):
    _type_ = "b"
c_byte.__ctype_le__ = c_byte.__ctype_be__ = c_byte
_check_size(c_byte)

class c_char(_SimpleCData):
    _type_ = "c"
c_char.__ctype_le__ = c_char.__ctype_be__ = c_char
_check_size(c_char)

class c_char_p(_SimpleCData):
    _type_ = "z"
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, c_void_p.from_buffer(self).value)
_check_size(c_char_p, "P")

class c_void_p(_SimpleCData):
    _type_ = "P"
c_voidp = c_void_p # backwards compatibility (to a bug)
_check_size(c_void_p)

class c_bool(_SimpleCData):
    _type_ = "?"

from _ctypes import POINTER, pointer, _pointer_type_cache

class c_wchar_p(_SimpleCData):
    _type_ = "Z"
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, c_void_p.from_buffer(self).value)

class c_wchar(_SimpleCData):
    _type_ = "u"

def _reset_cache():
    _pointer_type_cache.clear()
    _c_functype_cache.clear()
    if _os.name == "nt":
        _win_functype_cache.clear()
    # _SimpleCData.c_wchar_p_from_param
    POINTER(c_wchar).from_param = c_wchar_p.from_param
    # _SimpleCData.c_char_p_from_param
    POINTER(c_char).from_param = c_char_p.from_param
    _pointer_type_cache[None] = c_void_p

def create_unicode_buffer(init, size=None):
    """create_unicode_buffer(aString) -> character array
    create_unicode_buffer(anInteger) -> character array
    create_unicode_buffer(aString, anInteger) -> character array
    """
    if isinstance(init, str):
        if size is None:
            if sizeof(c_wchar) == 2:
                # UTF-16 requires a surrogate pair (2 wchar_t) for non-BMP
                # characters (outside [U+0000; U+FFFF] range). +1 for trailing
                # NUL character.
                size = sum(2 if ord(c) > 0xFFFF else 1 for c in init) + 1
            else:
                # 32-bit wchar_t (1 wchar_t per Unicode character). +1 for
                # trailing NUL character.
                size = len(init) + 1
        _sys.audit("ctypes.create_unicode_buffer", init, size)
        buftype = c_wchar * size
        buf = buftype()
        buf.value = init
        return buf
    elif isinstance(init, int):
        _sys.audit("ctypes.create_unicode_buffer", None, init)
        buftype = c_wchar * init
        buf = buftype()
        return buf
    raise TypeError(init)


# XXX Deprecated
def SetPointerType(pointer, cls):
    if _pointer_type_cache.get(cls, None) is not None:
        raise RuntimeError("This type already exists in the cache")
    if id(pointer) not in _pointer_type_cache:
        raise RuntimeError("What's this???")
    pointer.set_type(cls)
    _pointer_type_cache[cls] = pointer
    del _pointer_type_cache[id(pointer)]

# XXX Deprecated
def ARRAY(typ, len):
    return typ * len

################################################################


class CDLL(object):
    """An instance of this class represents a loaded dll/shared
    library, exporting functions using the standard C calling
    convention (named 'cdecl' on Windows).

    The exported functions can be accessed as attributes, or by
    indexing with the function name.  Examples:

    <obj>.qsort -> callable object
    <obj>['qsort'] -> callable object

    Calling the functions releases the Python GIL during the call and
    reacquires it afterwards.
    """
    _func_flags_ = _FUNCFLAG_CDECL
    _func_restype_ = c_int
    # default values for repr
    _name = '<uninitialized>'
    _handle = 0
    _FuncPtr = None

    def __init__(self, name, mode=DEFAULT_MODE, handle=None,
                 use_errno=False,
                 use_last_error=False,
                 winmode=None):
        self._name = name
        flags = self._func_flags_
        if use_errno:
            flags |= _FUNCFLAG_USE_ERRNO
        if use_last_error:
            flags |= _FUNCFLAG_USE_LASTERROR
        if _sys.platform.startswith("aix"):
            """When the name contains ".a(" and ends with ")",
               e.g., "libFOO.a(libFOO.so)" - this is taken to be an
               archive(member) syntax for dlopen(), and the mode is adjusted.
               Otherwise, name is presented to dlopen() as a file argument.
            """
            if name and name.endswith(")") and ".a(" in name:
                mode |= ( _os.RTLD_MEMBER | _os.RTLD_NOW )
        if _os.name == "nt":
            if winmode is not None:
                mode = winmode
            else:
                import nt
                mode = nt._LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
                if '/' in name or '\\' in name:
                    self._name = nt._getfullpathname(self._name)
                    mode |= nt._LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR

        class _FuncPtr(_CFuncPtr):
            _flags_ = flags
            _restype_ = self._func_restype_
        self._FuncPtr = _FuncPtr

        if handle is None:
            self._handle = _dlopen(self._name, mode)
        else:
            self._handle = handle

    def __repr__(self):
        return "<%s '%s', handle %x at %#x>" % \
               (self.__class__.__name__, self._name,
                (self._handle & (_sys.maxsize*2 + 1)),
                id(self) & (_sys.maxsize*2 + 1))

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        func = self.__getitem__(name)
        setattr(self, name, func)
        return func

    def __getitem__(self, name_or_ordinal):
        func = self._FuncPtr((name_or_ordinal, self))
        if not isinstance(name_or_ordinal, int):
            func.__name__ = name_or_ordinal
        return func

class PyDLL(CDLL):
    """This class represents the Python library itself.  It allows
    accessing Python API functions.  The GIL is not released, and
    Python exceptions are handled correctly.
    """
    _func_flags_ = _FUNCFLAG_CDECL | _FUNCFLAG_PYTHONAPI

if _os.name == "nt":

    class WinDLL(CDLL):
        """This class represents a dll exporting functions using the
        Windows stdcall calling convention.
        """
        _func_flags_ = _FUNCFLAG_STDCALL

    # XXX Hm, what about HRESULT as normal parameter?
    # Mustn't it derive from c_long then?
    from _ctypes import _check_HRESULT, _SimpleCData
    class HRESULT(_SimpleCData):
        _type_ = "l"
        # _check_retval_ is called with the function's result when it
        # is used as restype.  It checks for the FAILED bit, and
        # raises an OSError if it is set.
        #
        # The _check_retval_ method is implemented in C, so that the
        # method definition itself is not included in the traceback
        # when it raises an error - that is what we want (and Python
        # doesn't have a way to raise an exception in the caller's
        # frame).
        _check_retval_ = _check_HRESULT

    class OleDLL(CDLL):
        """This class represents a dll exporting functions using the
        Windows stdcall calling convention, and returning HRESULT.
        HRESULT error values are automatically raised as OSError
        exceptions.
        """
        _func_flags_ = _FUNCFLAG_STDCALL
        _func_restype_ = HRESULT

class LibraryLoader(object):
    def __init__(self, dlltype):
        self._dlltype = dlltype

    def __getattr__(self, name):
        if name[0] == '_':
            raise AttributeError(name)
        dll = self._dlltype(name)
        setattr(self, name, dll)
        return dll

    def __getitem__(self, name):
        return getattr(self, name)

    def LoadLibrary(self, name):
        return self._dlltype(name)

    __class_getitem__ = classmethod(_types.GenericAlias)

cdll = LibraryLoader(CDLL)
pydll = LibraryLoader(PyDLL)

if _os.name == "nt":
    pythonapi = PyDLL("python dll", None, _sys.dllhandle)
elif _sys.platform == "cygwin":
    pythonapi = PyDLL("libpython%d.%d.dll" % _sys.version_info[:2])
else:
    pythonapi = PyDLL(None)


if _os.name == "nt":
    windll = LibraryLoader(WinDLL)
    oledll = LibraryLoader(OleDLL)

    GetLastError = windll.kernel32.GetLastError
    from _ctypes import get_last_error, set_last_error

    def WinError(code=None, descr=None):
        if code is None:
            code = GetLastError()
        if descr is None:
            descr = FormatError(code).strip()
        return OSError(None, descr, None, code)

if sizeof(c_uint) == sizeof(c_void_p):
    c_size_t = c_uint
    c_ssize_t = c_int
elif sizeof(c_ulong) == sizeof(c_void_p):
    c_size_t = c_ulong
    c_ssize_t = c_long
elif sizeof(c_ulonglong) == sizeof(c_void_p):
    c_size_t = c_ulonglong
    c_ssize_t = c_longlong

# functions

from _ctypes import _memmove_addr, _memset_addr, _string_at_addr, _cast_addr

## void *memmove(void *, const void *, size_t);
memmove = CFUNCTYPE(c_void_p, c_void_p, c_void_p, c_size_t)(_memmove_addr)

## void *memset(void *, int, size_t)
memset = CFUNCTYPE(c_void_p, c_void_p, c_int, c_size_t)(_memset_addr)

def PYFUNCTYPE(restype, *argtypes):
    class CFunctionType(_CFuncPtr):
        _argtypes_ = argtypes
        _restype_ = restype
        _flags_ = _FUNCFLAG_CDECL | _FUNCFLAG_PYTHONAPI
    return CFunctionType

_cast = PYFUNCTYPE(py_object, c_void_p, py_object, py_object)(_cast_addr)
def cast(obj, typ):
    return _cast(obj, obj, typ)

_string_at = PYFUNCTYPE(py_object, c_void_p, c_int)(_string_at_addr)
def string_at(ptr, size=-1):
    """string_at(addr[, size]) -> string

    Return the string at addr."""
    return _string_at(ptr, size)

try:
    from _ctypes import _wstring_at_addr
except ImportError:
    pass
else:
    _wstring_at = PYFUNCTYPE(py_object, c_void_p, c_int)(_wstring_at_addr)
    def wstring_at(ptr, size=-1):
        """wstring_at(addr[, size]) -> string

        Return the string at addr."""
        return _wstring_at(ptr, size)


if _os.name == "nt": # COM stuff
    def DllGetClassObject(rclsid, riid, ppv):
        try:
            ccom = __import__("comtypes.server.inprocserver", globals(), locals(), ['*'])
        except ImportError:
            return -2147221231 # CLASS_E_CLASSNOTAVAILABLE
        else:
            return ccom.DllGetClassObject(rclsid, riid, ppv)

    def DllCanUnloadNow():
        try:
            ccom = __import__("comtypes.server.inprocserver", globals(), locals(), ['*'])
        except ImportError:
            return 0 # S_OK
        return ccom.DllCanUnloadNow()

from ctypes._endian import BigEndianStructure, LittleEndianStructure

# Fill in specifically-sized types
c_int8 = c_byte
c_uint8 = c_ubyte
for kind in [c_short, c_int, c_long, c_longlong]:
    if sizeof(kind) == 2: c_int16 = kind
    elif sizeof(kind) == 4: c_int32 = kind
    elif sizeof(kind) == 8: c_int64 = kind
for kind in [c_ushort, c_uint, c_ulong, c_ulonglong]:
    if sizeof(kind) == 2: c_uint16 = kind
    elif sizeof(kind) == 4: c_uint32 = kind
    elif sizeof(kind) == 8: c_uint64 = kind
del(kind)

_reset_cache()
<|endoftext|># The most useful windows datatypes
import ctypes

BYTE = ctypes.c_byte
WORD = ctypes.c_ushort
DWORD = ctypes.c_ulong

#UCHAR = ctypes.c_uchar
CHAR = ctypes.c_char
WCHAR = ctypes.c_wchar
UINT = ctypes.c_uint
INT = ctypes.c_int

DOUBLE = ctypes.c_double
FLOAT = ctypes.c_float

BOOLEAN = BYTE
BOOL = ctypes.c_long

class VARIANT_BOOL(ctypes._SimpleCData):
    _type_ = "v"
    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.value)

ULONG = ctypes.c_ulong
LONG = ctypes.c_long

USHORT = ctypes.c_ushort
SHORT = ctypes.c_short

# in the windows header files, these are structures.
_LARGE_INTEGER = LARGE_INTEGER = ctypes.c_longlong
_ULARGE_INTEGER = ULARGE_INTEGER = ctypes.c_ulonglong

LPCOLESTR = LPOLESTR = OLESTR = ctypes.c_wchar_p
LPCWSTR = LPWSTR = ctypes.c_wchar_p
LPCSTR = LPSTR = ctypes.c_char_p
LPCVOID = LPVOID = ctypes.c_void_p

# WPARAM is defined as UINT_PTR (unsigned type)
# LPARAM is defined as LONG_PTR (signed type)
if ctypes.sizeof(ctypes.c_long) == ctypes.sizeof(ctypes.c_void_p):
    WPARAM = ctypes.c_ulong
    LPARAM = ctypes.c_long
elif ctypes.sizeof(ctypes.c_longlong) == ctypes.sizeof(ctypes.c_void_p):
    WPARAM = ctypes.c_ulonglong
    LPARAM = ctypes.c_longlong

ATOM = WORD
LANGID = WORD

COLORREF = DWORD
LGRPID = DWORD
LCTYPE = DWORD

LCID = DWORD

################################################################
# HANDLE types
HANDLE = ctypes.c_void_p # in the header files: void *

HACCEL = HANDLE
HBITMAP = HANDLE
HBRUSH = HANDLE
HCOLORSPACE = HANDLE
HDC = HANDLE
HDESK = HANDLE
HDWP = HANDLE
HENHMETAFILE = HANDLE
HFONT = HANDLE
HGDIOBJ = HANDLE
HGLOBAL = HANDLE
HHOOK = HANDLE
HICON = HANDLE
HINSTANCE = HANDLE
HKEY = HANDLE
HKL = HANDLE
HLOCAL = HANDLE
HMENU = HANDLE
HMETAFILE = HANDLE
HMODULE = HANDLE
HMONITOR = HANDLE
HPALETTE = HANDLE
HPEN = HANDLE
HRGN = HANDLE
HRSRC = HANDLE
HSTR = HANDLE
HTASK = HANDLE
HWINSTA = HANDLE
HWND = HANDLE
SC_HANDLE = HANDLE
SERVICE_STATUS_HANDLE = HANDLE

################################################################
# Some important structure definitions

class RECT(ctypes.Structure):
    _fields_ = [("left", LONG),
                ("top", LONG),
                ("right", LONG),
                ("bottom", LONG)]
tagRECT = _RECTL = RECTL = RECT

class _SMALL_RECT(ctypes.Structure):
    _fields_ = [('Left', SHORT),
                ('Top', SHORT),
                ('Right', SHORT),
                ('Bottom', SHORT)]
SMALL_RECT = _SMALL_RECT

class _COORD(ctypes.Structure):
    _fields_ = [('X', SHORT),
                ('Y', SHORT)]

class POINT(ctypes.Structure):
    _fields_ = [("x", LONG),
                ("y", LONG)]
tagPOINT = _POINTL = POINTL = POINT

class SIZE(ctypes.Structure):
    _fields_ = [("cx", LONG),
                ("cy", LONG)]
tagSIZE = SIZEL = SIZE

def RGB(red, green, blue):
    return red + (green << 8) + (blue << 16)

class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", DWORD),
                ("dwHighDateTime", DWORD)]
_FILETIME = FILETIME

class MSG(ctypes.Structure):
    _fields_ = [("hWnd", HWND),
                ("message", UINT),
                ("wParam", WPARAM),
                ("lParam", LPARAM),
                ("time", DWORD),
                ("pt", POINT)]
tagMSG = MSG
MAX_PATH = 260

class WIN32_FIND_DATAA(ctypes.Structure):
    _fields_ = [("dwFileAttributes", DWORD),
                ("ftCreationTime", FILETIME),
                ("ftLastAccessTime", FILETIME),
                ("ftLastWriteTime", FILETIME),
                ("nFileSizeHigh", DWORD),
                ("nFileSizeLow", DWORD),
                ("dwReserved0", DWORD),
                ("dwReserved1", DWORD),
                ("cFileName", CHAR * MAX_PATH),
                ("cAlternateFileName", CHAR * 14)]

class WIN32_FIND_DATAW(ctypes.Structure):
    _fields_ = [("dwFileAttributes", DWORD),
                ("ftCreationTime", FILETIME),
                ("ftLastAccessTime", FILETIME),
                ("ftLastWriteTime", FILETIME),
                ("nFileSizeHigh", DWORD),
                ("nFileSizeLow", DWORD),
                ("dwReserved0", DWORD),
                ("dwReserved1", DWORD),
                ("cFileName", WCHAR * MAX_PATH),
                ("cAlternateFileName", WCHAR * 14)]

################################################################
# Pointer types

LPBOOL = PBOOL = ctypes.POINTER(BOOL)
PBOOLEAN = ctypes.POINTER(BOOLEAN)
LPBYTE = PBYTE = ctypes.POINTER(BYTE)
PCHAR = ctypes.POINTER(CHAR)
LPCOLORREF = ctypes.POINTER(COLORREF)
LPDWORD = PDWORD = ctypes.POINTER(DWORD)
LPFILETIME = PFILETIME = ctypes.POINTER(FILETIME)
PFLOAT = ctypes.POINTER(FLOAT)
LPHANDLE = PHANDLE = ctypes.POINTER(HANDLE)
PHKEY = ctypes.POINTER(HKEY)
LPHKL = ctypes.POINTER(HKL)
LPINT = PINT = ctypes.POINTER(INT)
PLARGE_INTEGER = ctypes.POINTER(LARGE_INTEGER)
PLCID = ctypes.POINTER(LCID)
LPLONG = PLONG = ctypes.POINTER(LONG)
LPMSG = PMSG = ctypes.POINTER(MSG)
LPPOINT = PPOINT = ctypes.POINTER(POINT)
PPOINTL = ctypes.POINTER(POINTL)
LPRECT = PRECT = ctypes.POINTER(RECT)
LPRECTL = PRECTL = ctypes.POINTER(RECTL)
LPSC_HANDLE = ctypes.POINTER(SC_HANDLE)
PSHORT = ctypes.POINTER(SHORT)
LPSIZE = PSIZE = ctypes.POINTER(SIZE)
LPSIZEL = PSIZEL = ctypes.POINTER(SIZEL)
PSMALL_RECT = ctypes.POINTER(SMALL_RECT)
LPUINT = PUINT = ctypes.POINTER(UINT)
PULARGE_INTEGER = ctypes.POINTER(ULARGE_INTEGER)
PULONG = ctypes.POINTER(ULONG)
PUSHORT = ctypes.POINTER(USHORT)
PWCHAR = ctypes.POINTER(WCHAR)
LPWIN32_FIND_DATAA = PWIN32_FIND_DATAA = ctypes.POINTER(WIN32_FIND_DATAA)
LPWIN32_FIND_DATAW = PWIN32_FIND_DATAW = ctypes.POINTER(WIN32_FIND_DATAW)
LPWORD = PWORD = ctypes.POINTER(WORD)
<|endoftext|>import sys
from ctypes import *

_array_type = type(Array)

def _other_endian(typ):
    """Return the type with the 'other' byte order.  Simple types like
    c_int and so on already have __ctype_be__ and __ctype_le__
    attributes which contain the types, for more complicated types
    arrays and structures are supported.
    """
    # check _OTHER_ENDIAN attribute (present if typ is primitive type)
    if hasattr(typ, _OTHER_ENDIAN):
        return getattr(typ, _OTHER_ENDIAN)
    # if typ is array
    if isinstance(typ, _array_type):
        return _other_endian(typ._type_) * typ._length_
    # if typ is structure
    if issubclass(typ, Structure):
        return typ
    raise TypeError("This type does not support other endian: %s" % typ)

class _swapped_meta(type(Structure)):
    def __setattr__(self, attrname, value):
        if attrname == "_fields_":
            fields = []
            for desc in value:
                name = desc[0]
                typ = desc[1]
                rest = desc[2:]
                fields.append((name, _other_endian(typ)) + rest)
            value = fields
        super().__setattr__(attrname, value)

################################################################

# Note: The Structure metaclass checks for the *presence* (not the
# value!) of a _swapped_bytes_ attribute to determine the bit order in
# structures containing bit fields.

if sys.byteorder == "little":
    _OTHER_ENDIAN = "__ctype_be__"

    LittleEndianStructure = Structure

    class BigEndianStructure(Structure, metaclass=_swapped_meta):
        """Structure with big endian byte order"""
        __slots__ = ()
        _swappedbytes_ = None

elif sys.byteorder == "big":
    _OTHER_ENDIAN = "__ctype_le__"

    BigEndianStructure = Structure
    class LittleEndianStructure(Structure, metaclass=_swapped_meta):
        """Structure with little endian byte order"""
        __slots__ = ()
        _swappedbytes_ = None

else:
    raise RuntimeError("Invalid byteorder")
<|endoftext|>"""
Lib/ctypes.util.find_library() support for AIX
Similar approach as done for Darwin support by using separate files
but unlike Darwin - no extension such as ctypes.macholib.*

dlopen() is an interface to AIX initAndLoad() - primary documentation at:
https://www.ibm.com/support/knowledgecenter/en/ssw_aix_61/com.ibm.aix.basetrf1/dlopen.htm
https://www.ibm.com/support/knowledgecenter/en/ssw_aix_61/com.ibm.aix.basetrf1/load.htm

AIX supports two styles for dlopen(): svr4 (System V Release 4) which is common on posix
platforms, but also a BSD style - aka SVR3.

From AIX 5.3 Difference Addendum (December 2004)
2.9 SVR4 linking affinity
Nowadays, there are two major object file formats used by the operating systems:
XCOFF: The COFF enhanced by IBM and others. The original COFF (Common
Object File Format) was the base of SVR3 and BSD 4.2 systems.
ELF:   Executable and Linking Format that was developed by AT&T and is a
base for SVR4 UNIX.

While the shared library content is identical on AIX - one is located as a filepath name
(svr4 style) and the other is located as a member of an archive (and the archive
is located as a filepath name).

The key difference arises when supporting multiple abi formats (i.e., 32 and 64 bit).
For svr4 either only one ABI is supported, or there are two directories, or there
are different file names. The most common solution for multiple ABI is multiple
directories.

For the XCOFF (aka AIX) style - one directory (one archive file) is sufficient
as multiple shared libraries can be in the archive - even sharing the same name.
In documentation the archive is also referred to as the "base" and the shared
library object is referred to as the "member".

For dlopen() on AIX (read initAndLoad()) the calls are similar.
Default activity occurs when no path information is provided. When path
information is provided dlopen() does not search any other directories.

For SVR4 - the shared library name is the name of the file expected: libFOO.so
For AIX - the shared library is expressed as base(member). The search is for the
base (e.g., libFOO.a) and once the base is found the shared library - identified by
member (e.g., libFOO.so, or shr.o) is located and loaded.

The mode bit RTLD_MEMBER tells initAndLoad() that it needs to use the AIX (SVR3)
naming style.
"""
__author__ = "Michael Felt <aixtools@felt.demon.nl>"

import re
from os import environ, path
from sys import executable
from ctypes import c_void_p, sizeof
from subprocess import Popen, PIPE, DEVNULL

# Executable bit size - 32 or 64
# Used to filter the search in an archive by size, e.g., -X64
AIX_ABI = sizeof(c_void_p) * 8


from sys import maxsize
def _last_version(libnames, sep):
    def _num_version(libname):
        # "libxyz.so.MAJOR.MINOR" => [MAJOR, MINOR]
        parts = libname.split(sep)
        nums = []
        try:
            while parts:
                nums.insert(0, int(parts.pop()))
        except ValueError:
            pass
        return nums or [maxsize]
    return max(reversed(libnames), key=_num_version)

def get_ld_header(p):
    # "nested-function, but placed at module level
    ld_header = None
    for line in p.stdout:
        if line.startswith(('/', './', '../')):
            ld_header = line
        elif "INDEX" in line:
            return ld_header.rstrip('\n')
    return None

def get_ld_header_info(p):
    # "nested-function, but placed at module level
    # as an ld_header was found, return known paths, archives and members
    # these lines start with a digit
    info = []
    for line in p.stdout:
        if re.match("[0-9]", line):
            info.append(line)
        else:
            # blank line (separator), consume line and end for loop
            break
    return info

def get_ld_headers(file):
    """
    Parse the header of the loader section of executable and archives
    This function calls /usr/bin/dump -H as a subprocess
    and returns a list of (ld_header, ld_header_info) tuples.
    """
    # get_ld_headers parsing:
    # 1. Find a line that starts with /, ./, or ../ - set as ld_header
    # 2. If "INDEX" in occurs in a following line - return ld_header
    # 3. get info (lines starting with [0-9])
    ldr_headers = []
    p = Popen(["/usr/bin/dump", f"-X{AIX_ABI}", "-H", file],
        universal_newlines=True, stdout=PIPE, stderr=DEVNULL)
    # be sure to read to the end-of-file - getting all entries
    while True:
        ld_header = get_ld_header(p)
        if ld_header:
            ldr_headers.append((ld_header, get_ld_header_info(p)))
        else:
            break
    p.stdout.close()
    p.wait()
    return ldr_headers

def get_shared(ld_headers):
    """
    extract the shareable objects from ld_headers
    character "[" is used to strip off the path information.
    Note: the "[" and "]" characters that are part of dump -H output
    are not removed here.
    """
    shared = []
    for (line, _) in ld_headers:
        # potential member lines contain "["
        # otherwise, no processing needed
        if "[" in line:
            # Strip off trailing colon (:)
            shared.append(line[line.index("["):-1])
    return shared

def get_one_match(expr, lines):
    """
    Must be only one match, otherwise result is None.
    When there is a match, strip leading "[" and trailing "]"
    """
    # member names in the ld_headers output are between square brackets
    expr = rf'\[({expr})\]'
    matches = list(filter(None, (re.search(expr, line) for line in lines)))
    if len(matches) == 1:
        return matches[0].group(1)
    else:
        return None

# additional processing to deal with AIX legacy names for 64-bit members
def get_legacy(members):
    """
    This routine provides historical aka legacy naming schemes started
    in AIX4 shared library support for library members names.
    e.g., in /usr/lib/libc.a the member name shr.o for 32-bit binary and
    shr_64.o for 64-bit binary.
    """
    if AIX_ABI == 64:
        # AIX 64-bit member is one of shr64.o, shr_64.o, or shr4_64.o
        expr = r'shr4?_?64\.o'
        member = get_one_match(expr, members)
        if member:
            return member
    else:
        # 32-bit legacy names - both shr.o and shr4.o exist.
        # shr.o is the preferred name so we look for shr.o first
        #  i.e., shr4.o is returned only when shr.o does not exist
        for name in ['shr.o', 'shr4.o']:
            member = get_one_match(re.escape(name), members)
            if member:
                return member
    return None

def get_version(name, members):
    """
    Sort list of members and return highest numbered version - if it exists.
    This function is called when an unversioned libFOO.a(libFOO.so) has
    not been found.

    Versioning for the member name is expected to follow
    GNU LIBTOOL conventions: the highest version (x, then X.y, then X.Y.z)
     * find [libFoo.so.X]
     * find [libFoo.so.X.Y]
     * find [libFoo.so.X.Y.Z]

    Before the GNU convention became the standard scheme regardless of
    binary size AIX packagers used GNU convention "as-is" for 32-bit
    archive members but used an "distinguishing" name for 64-bit members.
    This scheme inserted either 64 or _64 between libFOO and .so
    - generally libFOO_64.so, but occasionally libFOO64.so
    """
    # the expression ending for versions must start as
    # '.so.[0-9]', i.e., *.so.[at least one digit]
    # while multiple, more specific expressions could be specified
    # to search for .so.X, .so.X.Y and .so.X.Y.Z
    # after the first required 'dot' digit
    # any combination of additional 'dot' digits pairs are accepted
    # anything more than libFOO.so.digits.digits.digits
    # should be seen as a member name outside normal expectations
    exprs = [rf'lib{name}\.so\.[0-9]+[0-9.]*',
        rf'lib{name}_?64\.so\.[0-9]+[0-9.]*']
    for expr in exprs:
        versions = []
        for line in members:
            m = re.search(expr, line)
            if m:
                versions.append(m.group(0))
        if versions:
            return _last_version(versions, '.')
    return None

def get_member(name, members):
    """
    Return an archive member matching the request in name.
    Name is the library name without any prefix like lib, suffix like .so,
    or version number.
    Given a list of members find and return the most appropriate result
    Priority is given to generic libXXX.so, then a versioned libXXX.so.a.b.c
    and finally, legacy AIX naming scheme.
    """
    # look first for a generic match - prepend lib and append .so
    expr = rf'lib{name}\.so'
    member = get_one_match(expr, members)
    if member:
        return member
    elif AIX_ABI == 64:
        expr = rf'lib{name}64\.so'
        member = get_one_match(expr, members)
    if member:
        return member
    # since an exact match with .so as suffix was not found
    # look for a versioned name
    # If a versioned name is not found, look for AIX legacy member name
    member = get_version(name, members)
    if member:
        return member
    else:
        return get_legacy(members)

def get_libpaths():
    """
    On AIX, the buildtime searchpath is stored in the executable.
    as "loader header information".
    The command /usr/bin/dump -H extracts this info.
    Prefix searched libraries with LD_LIBRARY_PATH (preferred),
    or LIBPATH if defined. These paths are appended to the paths
    to libraries the python executable is linked with.
    This mimics AIX dlopen() behavior.
    """
    libpaths = environ.get("LD_LIBRARY_PATH")
    if libpaths is None:
        libpaths = environ.get("LIBPATH")
    if libpaths is None:
        libpaths = []
    else:
        libpaths = libpaths.split(":")
    objects = get_ld_headers(executable)
    for (_, lines) in objects:
        for line in lines:
            # the second (optional) argument is PATH if it includes a /
            path = line.split()[1]
            if "/" in path:
                libpaths.extend(path.split(":"))
    return libpaths

def find_shared(paths, name):
    """
    paths is a list of directories to search for an archive.
    name is the abbreviated name given to find_library().
    Process: search "paths" for archive, and if an archive is found
    return the result of get_member().
    If an archive is not found then return None
    """
    for dir in paths:
        # /lib is a symbolic link to /usr/lib, skip it
        if dir == "/lib":
            continue
        # "lib" is prefixed to emulate compiler name resolution,
        # e.g., -lc to libc
        base = f'lib{name}.a'
        archive = path.join(dir, base)
        if path.exists(archive):
            members = get_shared(get_ld_headers(archive))
            member = get_member(re.escape(name), members)
            if member is not None:
                return (base, member)
            else:
                return (None, None)
    return (None, None)

def find_library(name):
    """AIX implementation of ctypes.util.find_library()
    Find an archive member that will dlopen(). If not available,
    also search for a file (or link) with a .so suffix.

    AIX supports two types of schemes that can be used with dlopen().
    The so-called SystemV Release4 (svr4) format is commonly suffixed
    with .so while the (default) AIX scheme has the library (archive)
    ending with the suffix .a
    As an archive has multiple members (e.g., 32-bit and 64-bit) in one file
    the argument passed to dlopen must include both the library and
    the member names in a single string.

    find_library() looks first for an archive (.a) with a suitable member.
    If no archive+member pair is found, look for a .so file.
    """

    libpaths = get_libpaths()
    (base, member) = find_shared(libpaths, name)
    if base is not None:
        return f"{base}({member})"

    # To get here, a member in an archive has not been found
    # In other words, either:
    # a) a .a file was not found
    # b) a .a file did not have a suitable member
    # So, look for a .so file
    # Check libpaths for .so file
    # Note, the installation must prepare a link from a .so
    # to a versioned file
    # This is common practice by GNU libtool on other platforms
    soname = f"lib{name}.so"
    for dir in libpaths:
        # /lib is a symbolic link to /usr/lib, skip it
        if dir == "/lib":
            continue
        shlib = path.join(dir, soname)
        if path.exists(shlib):
            return soname
    # if we are here, we have not found anything plausible
    return None
<|endoftext|>"""distutils

The main package for the Python Module Distribution Utilities.  Normally
used from a setup script as

   from distutils.core import setup

   setup (...)
"""

import sys
import warnings

__version__ = sys.version[:sys.version.index(' ')]

_DEPRECATION_MESSAGE = ("The distutils package is deprecated and slated for "
                        "removal in Python 3.12. Use setuptools or check "
                        "PEP 632 for potential alternatives")
warnings.warn(_DEPRECATION_MESSAGE,
              DeprecationWarning, 2)
<|endoftext|>#
# distutils/version.py
#
# Implements multiple version numbering conventions for the
# Python Module Distribution Utilities.
#
# $Id$
#

"""Provides classes to represent module version numbers (one class for
each style of version numbering).  There are currently two such classes
implemented: StrictVersion and LooseVersion.

Every version number class implements the following interface:
  * the 'parse' method takes a string and parses it to some internal
    representation; if the string is an invalid version number,
    'parse' raises a ValueError exception
  * the class constructor takes an optional string argument which,
    if supplied, is passed to 'parse'
  * __str__ reconstructs the string that was passed to 'parse' (or
    an equivalent string -- ie. one that will generate an equivalent
    version number instance)
  * __repr__ generates Python code to recreate the version number instance
  * _cmp compares the current instance with either another instance
    of the same class or a string (which will be parsed to an instance
    of the same class, thus must follow the same rules)
"""

import re

class Version:
    """Abstract base class for version numbering classes.  Just provides
    constructor (__init__) and reproducer (__repr__), because those
    seem to be the same for all version numbering classes; and route
    rich comparisons to _cmp.
    """

    def __init__ (self, vstring=None):
        if vstring:
            self.parse(vstring)

    def __repr__ (self):
        return "%s ('%s')" % (self.__class__.__name__, str(self))

    def __eq__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c == 0

    def __lt__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c < 0

    def __le__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c <= 0

    def __gt__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c > 0

    def __ge__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c >= 0


# Interface for version-number classes -- must be implemented
# by the following classes (the concrete ones -- Version should
# be treated as an abstract class).
#    __init__ (string) - create and take same action as 'parse'
#                        (string parameter is optional)
#    parse (string)    - convert a string representation to whatever
#                        internal representation is appropriate for
#                        this style of version numbering
#    __str__ (self)    - convert back to a string; should be very similar
#                        (if not identical to) the string supplied to parse
#    __repr__ (self)   - generate Python code to recreate
#                        the instance
#    _cmp (self, other) - compare two version numbers ('other' may
#                        be an unparsed version string, or another
#                        instance of your version class)


class StrictVersion (Version):

    """Version numbering for anal retentives and software idealists.
    Implements the standard interface for version number classes as
    described above.  A version number consists of two or three
    dot-separated numeric components, with an optional "pre-release" tag
    on the end.  The pre-release tag consists of the letter 'a' or 'b'
    followed by a number.  If the numeric components of two version
    numbers are equal, then one with a pre-release tag will always
    be deemed earlier (lesser) than one without.

    The following are valid version numbers (shown in the order that
    would be obtained by sorting according to the supplied cmp function):

        0.4       0.4.0  (these two are equivalent)
        0.4.1
        0.5a1
        0.5b3
        0.5
        0.9.6
        1.0
        1.0.4a3
        1.0.4b1
        1.0.4

    The following are examples of invalid version numbers:

        1
        2.7.2.2
        1.3.a4
        1.3pl1
        1.3c4

    The rationale for this version numbering system will be explained
    in the distutils documentation.
    """

    version_re = re.compile(r'^(\d+) \. (\d+) (\. (\d+))? ([ab](\d+))?$',
                            re.VERBOSE | re.ASCII)


    def parse (self, vstring):
        match = self.version_re.match(vstring)
        if not match:
            raise ValueError("invalid version number '%s'" % vstring)

        (major, minor, patch, prerelease, prerelease_num) = \
            match.group(1, 2, 4, 5, 6)

        if patch:
            self.version = tuple(map(int, [major, minor, patch]))
        else:
            self.version = tuple(map(int, [major, minor])) + (0,)

        if prerelease:
            self.prerelease = (prerelease[0], int(prerelease_num))
        else:
            self.prerelease = None


    def __str__ (self):

        if self.version[2] == 0:
            vstring = '.'.join(map(str, self.version[0:2]))
        else:
            vstring = '.'.join(map(str, self.version))

        if self.prerelease:
            vstring = vstring + self.prerelease[0] + str(self.prerelease[1])

        return vstring


    def _cmp (self, other):
        if isinstance(other, str):
            other = StrictVersion(other)
        elif not isinstance(other, StrictVersion):
            return NotImplemented

        if self.version != other.version:
            # numeric versions don't match
            # prerelease stuff doesn't matter
            if self.version < other.version:
                return -1
            else:
                return 1

        # have to compare prerelease
        # case 1: neither has prerelease; they're equal
        # case 2: self has prerelease, other doesn't; other is greater
        # case 3: self doesn't have prerelease, other does: self is greater
        # case 4: both have prerelease: must compare them!

        if (not self.prerelease and not other.prerelease):
            return 0
        elif (self.prerelease and not other.prerelease):
            return -1
        elif (not self.prerelease and other.prerelease):
            return 1
        elif (self.prerelease and other.prerelease):
            if self.prerelease == other.prerelease:
                return 0
            elif self.prerelease < other.prerelease:
                return -1
            else:
                return 1
        else:
            assert False, "never get here"

# end class StrictVersion


# The rules according to Greg Stein:
# 1) a version number has 1 or more numbers separated by a period or by
#    sequences of letters. If only periods, then these are compared
#    left-to-right to determine an ordering.
# 2) sequences of letters are part of the tuple for comparison and are
#    compared lexicographically
# 3) recognize the numeric components may have leading zeroes
#
# The LooseVersion class below implements these rules: a version number
# string is split up into a tuple of integer and string components, and
# comparison is a simple tuple comparison.  This means that version
# numbers behave in a predictable and obvious way, but a way that might
# not necessarily be how people *want* version numbers to behave.  There
# wouldn't be a problem if people could stick to purely numeric version
# numbers: just split on period and compare the numbers as tuples.
# However, people insist on putting letters into their version numbers;
# the most common purpose seems to be:
#   - indicating a "pre-release" version
#     ('alpha', 'beta', 'a', 'b', 'pre', 'p')
#   - indicating a post-release patch ('p', 'pl', 'patch')
# but of course this can't cover all version number schemes, and there's
# no way to know what a programmer means without asking him.
#
# The problem is what to do with letters (and other non-numeric
# characters) in a version number.  The current implementation does the
# obvious and predictable thing: keep them as strings and compare
# lexically within a tuple comparison.  This has the desired effect if
# an appended letter sequence implies something "post-release":
# eg. "0.99" < "0.99pl14" < "1.0", and "5.001" < "5.001m" < "5.002".
#
# However, if letters in a version number imply a pre-release version,
# the "obvious" thing isn't correct.  Eg. you would expect that
# "1.5.1" < "1.5.2a2" < "1.5.2", but under the tuple/lexical comparison
# implemented here, this just isn't so.
#
# Two possible solutions come to mind.  The first is to tie the
# comparison algorithm to a particular set of semantic rules, as has
# been done in the StrictVersion class above.  This works great as long
# as everyone can go along with bondage and discipline.  Hopefully a
# (large) subset of Python module programmers will agree that the
# particular flavour of bondage and discipline provided by StrictVersion
# provides enough benefit to be worth using, and will submit their
# version numbering scheme to its domination.  The free-thinking
# anarchists in the lot will never give in, though, and something needs
# to be done to accommodate them.
#
# Perhaps a "moderately strict" version class could be implemented that
# lets almost anything slide (syntactically), and makes some heuristic
# assumptions about non-digits in version number strings.  This could
# sink into special-case-hell, though; if I was as talented and
# idiosyncratic as Larry Wall, I'd go ahead and implement a class that
# somehow knows that "1.2.1" < "1.2.2a2" < "1.2.2" < "1.2.2pl3", and is
# just as happy dealing with things like "2g6" and "1.13++".  I don't
# think I'm smart enough to do it right though.
#
# In any case, I've coded the test suite for this module (see
# ../test/test_version.py) specifically to fail on things like comparing
# "1.2a2" and "1.2".  That's not because the *code* is doing anything
# wrong, it's because the simple, obvious design doesn't match my
# complicated, hairy expectations for real-world version numbers.  It
# would be a snap to fix the test suite to say, "Yep, LooseVersion does
# the Right Thing" (ie. the code matches the conception).  But I'd rather
# have a conception that matches common notions about version numbers.

class LooseVersion (Version):

    """Version numbering for anarchists and software realists.
    Implements the standard interface for version number classes as
    described above.  A version number consists of a series of numbers,
    separated by either periods or strings of letters.  When comparing
    version numbers, the numeric components will be compared
    numerically, and the alphabetic components lexically.  The following
    are all valid version numbers, in no particular order:

        1.5.1
        1.5.2b2
        161
        3.10a
        8.02
        3.4j
        1996.07.12
        3.2.pl0
        3.1.1.6
        2g6
        11g
        0.960923
        2.2beta29
        1.13++
        5.5.kw
        2.0b1pl0

    In fact, there is no such thing as an invalid version number under
    this scheme; the rules for comparison are simple and predictable,
    but may not always give the results you want (for some definition
    of "want").
    """

    component_re = re.compile(r'(\d+ | [a-z]+ | \.)', re.VERBOSE)

    def __init__ (self, vstring=None):
        if vstring:
            self.parse(vstring)


    def parse (self, vstring):
        # I've given up on thinking I can reconstruct the version string
        # from the parsed tuple -- so I just store the string here for
        # use by __str__
        self.vstring = vstring
        components = [x for x in self.component_re.split(vstring)
                              if x and x != '.']
        for i, obj in enumerate(components):
            try:
                components[i] = int(obj)
            except ValueError:
                pass

        self.version = components


    def __str__ (self):
        return self.vstring


    def __repr__ (self):
        return "LooseVersion ('%s')" % str(self)


    def _cmp (self, other):
        if isinstance(other, str):
            other = LooseVersion(other)
        elif not isinstance(other, LooseVersion):
            return NotImplemented

        if self.version == other.version:
            return 0
        if self.version < other.version:
            return -1
        if self.version > other.version:
            return 1


# end class LooseVersion
<|endoftext|>"""Python 'hex_codec' Codec - 2-digit hex content transfer encoding.

This codec de/encodes from bytes to bytes.

Written by Marc-Andre Lemburg (mal@lemburg.com).
"""

import codecs
import binascii

### Codec APIs

def hex_encode(input, errors='strict'):
    assert errors == 'strict'
    return (binascii.b2a_hex(input), len(input))

def hex_decode(input, errors='strict'):
    assert errors == 'strict'
    return (binascii.a2b_hex(input), len(input))

class Codec(codecs.Codec):
    def encode(self, input, errors='strict'):
        return hex_encode(input, errors)
    def decode(self, input, errors='strict'):
        return hex_decode(input, errors)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        assert self.errors == 'strict'
        return binascii.b2a_hex(input)

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        assert self.errors == 'strict'
        return binascii.a2b_hex(input)

class StreamWriter(Codec, codecs.StreamWriter):
    charbuffertype = bytes

class StreamReader(Codec, codecs.StreamReader):
    charbuffertype = bytes

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='hex',
        encode=hex_encode,
        decode=hex_decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
        _is_text_encoding=False,
    )
<|endoftext|>""" Python Character Mapping Codec iso8859_7 generated from 'MAPPINGS/ISO8859/8859-7.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='iso8859-7',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> NULL
    '\x01'     #  0x01 -> START OF HEADING
    '\x02'     #  0x02 -> START OF TEXT
    '\x03'     #  0x03 -> END OF TEXT
    '\x04'     #  0x04 -> END OF TRANSMISSION
    '\x05'     #  0x05 -> ENQUIRY
    '\x06'     #  0x06 -> ACKNOWLEDGE
    '\x07'     #  0x07 -> BELL
    '\x08'     #  0x08 -> BACKSPACE
    '\t'       #  0x09 -> HORIZONTAL TABULATION
    '\n'       #  0x0A -> LINE FEED
    '\x0b'     #  0x0B -> VERTICAL TABULATION
    '\x0c'     #  0x0C -> FORM FEED
    '\r'       #  0x0D -> CARRIAGE RETURN
    '\x0e'     #  0x0E -> SHIFT OUT
    '\x0f'     #  0x0F -> SHIFT IN
    '\x10'     #  0x10 -> DATA LINK ESCAPE
    '\x11'     #  0x11 -> DEVICE CONTROL ONE
    '\x12'     #  0x12 -> DEVICE CONTROL TWO
    '\x13'     #  0x13 -> DEVICE CONTROL THREE
    '\x14'     #  0x14 -> DEVICE CONTROL FOUR
    '\x15'     #  0x15 -> NEGATIVE ACKNOWLEDGE
    '\x16'     #  0x16 -> SYNCHRONOUS IDLE
    '\x17'     #  0x17 -> END OF TRANSMISSION BLOCK
    '\x18'     #  0x18 -> CANCEL
    '\x19'     #  0x19 -> END OF MEDIUM
    '\x1a'     #  0x1A -> SUBSTITUTE
    '\x1b'     #  0x1B -> ESCAPE
    '\x1c'     #  0x1C -> FILE SEPARATOR
    '\x1d'     #  0x1D -> GROUP SEPARATOR
    '\x1e'     #  0x1E -> RECORD SEPARATOR
    '\x1f'     #  0x1F -> UNIT SEPARATOR
    ' '        #  0x20 -> SPACE
    '!'        #  0x21 -> EXCLAMATION MARK
    '"'        #  0x22 -> QUOTATION MARK
    '#'        #  0x23 -> NUMBER SIGN
    '$'        #  0x24 -> DOLLAR SIGN
    '%'        #  0x25 -> PERCENT SIGN
    '&'        #  0x26 -> AMPERSAND
    "'"        #  0x27 -> APOSTROPHE
    '('        #  0x28 -> LEFT PARENTHESIS
    ')'        #  0x29 -> RIGHT PARENTHESIS
    '*'        #  0x2A -> ASTERISK
    '+'        #  0x2B -> PLUS SIGN
    ','        #  0x2C -> COMMA
    '-'        #  0x2D -> HYPHEN-MINUS
    '.'        #  0x2E -> FULL STOP
    '/'        #  0x2F -> SOLIDUS
    '0'        #  0x30 -> DIGIT ZERO
    '1'        #  0x31 -> DIGIT ONE
    '2'        #  0x32 -> DIGIT TWO
    '3'        #  0x33 -> DIGIT THREE
    '4'        #  0x34 -> DIGIT FOUR
    '5'        #  0x35 -> DIGIT FIVE
    '6'        #  0x36 -> DIGIT SIX
    '7'        #  0x37 -> DIGIT SEVEN
    '8'        #  0x38 -> DIGIT EIGHT
    '9'        #  0x39 -> DIGIT NINE
    ':'        #  0x3A -> COLON
    ';'        #  0x3B -> SEMICOLON
    '<'        #  0x3C -> LESS-THAN SIGN
    '='        #  0x3D -> EQUALS SIGN
    '>'        #  0x3E -> GREATER-THAN SIGN
    '?'        #  0x3F -> QUESTION MARK
    '@'        #  0x40 -> COMMERCIAL AT
    'A'        #  0x41 -> LATIN CAPITAL LETTER A
    'B'        #  0x42 -> LATIN CAPITAL LETTER B
    'C'        #  0x43 -> LATIN CAPITAL LETTER C
    'D'        #  0x44 -> LATIN CAPITAL LETTER D
    'E'        #  0x45 -> LATIN CAPITAL LETTER E
    'F'        #  0x46 -> LATIN CAPITAL LETTER F
    'G'        #  0x47 -> LATIN CAPITAL LETTER G
    'H'        #  0x48 -> LATIN CAPITAL LETTER H
    'I'        #  0x49 -> LATIN CAPITAL LETTER I
    'J'        #  0x4A -> LATIN CAPITAL LETTER J
    'K'        #  0x4B -> LATIN CAPITAL LETTER K
    'L'        #  0x4C -> LATIN CAPITAL LETTER L
    'M'        #  0x4D -> LATIN CAPITAL LETTER M
    'N'        #  0x4E -> LATIN CAPITAL LETTER N
    'O'        #  0x4F -> LATIN CAPITAL LETTER O
    'P'        #  0x50 -> LATIN CAPITAL LETTER P
    'Q'        #  0x51 -> LATIN CAPITAL LETTER Q
    'R'        #  0x52 -> LATIN CAPITAL LETTER R
    'S'        #  0x53 -> LATIN CAPITAL LETTER S
    'T'        #  0x54 -> LATIN CAPITAL LETTER T
    'U'        #  0x55 -> LATIN CAPITAL LETTER U
    'V'        #  0x56 -> LATIN CAPITAL LETTER V
    'W'        #  0x57 -> LATIN CAPITAL LETTER W
    'X'        #  0x58 -> LATIN CAPITAL LETTER X
    'Y'        #  0x59 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x5A -> LATIN CAPITAL LETTER Z
    '['        #  0x5B -> LEFT SQUARE BRACKET
    '\\'       #  0x5C -> REVERSE SOLIDUS
    ']'        #  0x5D -> RIGHT SQUARE BRACKET
    '^'        #  0x5E -> CIRCUMFLEX ACCENT
    '_'        #  0x5F -> LOW LINE
    '`'        #  0x60 -> GRAVE ACCENT
    'a'        #  0x61 -> LATIN SMALL LETTER A
    'b'        #  0x62 -> LATIN SMALL LETTER B
    'c'        #  0x63 -> LATIN SMALL LETTER C
    'd'        #  0x64 -> LATIN SMALL LETTER D
    'e'        #  0x65 -> LATIN SMALL LETTER E
    'f'        #  0x66 -> LATIN SMALL LETTER F
    'g'        #  0x67 -> LATIN SMALL LETTER G
    'h'        #  0x68 -> LATIN SMALL LETTER H
    'i'        #  0x69 -> LATIN SMALL LETTER I
    'j'        #  0x6A -> LATIN SMALL LETTER J
    'k'        #  0x6B -> LATIN SMALL LETTER K
    'l'        #  0x6C -> LATIN SMALL LETTER L
    'm'        #  0x6D -> LATIN SMALL LETTER M
    'n'        #  0x6E -> LATIN SMALL LETTER N
    'o'        #  0x6F -> LATIN SMALL LETTER O
    'p'        #  0x70 -> LATIN SMALL LETTER P
    'q'        #  0x71 -> LATIN SMALL LETTER Q
    'r'        #  0x72 -> LATIN SMALL LETTER R
    's'        #  0x73 -> LATIN SMALL LETTER S
    't'        #  0x74 -> LATIN SMALL LETTER T
    'u'        #  0x75 -> LATIN SMALL LETTER U
    'v'        #  0x76 -> LATIN SMALL LETTER V
    'w'        #  0x77 -> LATIN SMALL LETTER W
    'x'        #  0x78 -> LATIN SMALL LETTER X
    'y'        #  0x79 -> LATIN SMALL LETTER Y
    'z'        #  0x7A -> LATIN SMALL LETTER Z
    '{'        #  0x7B -> LEFT CURLY BRACKET
    '|'        #  0x7C -> VERTICAL LINE
    '}'        #  0x7D -> RIGHT CURLY BRACKET
    '~'        #  0x7E -> TILDE
    '\x7f'     #  0x7F -> DELETE
    '\x80'     #  0x80 -> <control>
    '\x81'     #  0x81 -> <control>
    '\x82'     #  0x82 -> <control>
    '\x83'     #  0x83 -> <control>
    '\x84'     #  0x84 -> <control>
    '\x85'     #  0x85 -> <control>
    '\x86'     #  0x86 -> <control>
    '\x87'     #  0x87 -> <control>
    '\x88'     #  0x88 -> <control>
    '\x89'     #  0x89 -> <control>
    '\x8a'     #  0x8A -> <control>
    '\x8b'     #  0x8B -> <control>
    '\x8c'     #  0x8C -> <control>
    '\x8d'     #  0x8D -> <control>
    '\x8e'     #  0x8E -> <control>
    '\x8f'     #  0x8F -> <control>
    '\x90'     #  0x90 -> <control>
    '\x91'     #  0x91 -> <control>
    '\x92'     #  0x92 -> <control>
    '\x93'     #  0x93 -> <control>
    '\x94'     #  0x94 -> <control>
    '\x95'     #  0x95 -> <control>
    '\x96'     #  0x96 -> <control>
    '\x97'     #  0x97 -> <control>
    '\x98'     #  0x98 -> <control>
    '\x99'     #  0x99 -> <control>
    '\x9a'     #  0x9A -> <control>
    '\x9b'     #  0x9B -> <control>
    '\x9c'     #  0x9C -> <control>
    '\x9d'     #  0x9D -> <control>
    '\x9e'     #  0x9E -> <control>
    '\x9f'     #  0x9F -> <control>
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\u2018'   #  0xA1 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0xA2 -> RIGHT SINGLE QUOTATION MARK
    '\xa3'     #  0xA3 -> POUND SIGN
    '\u20ac'   #  0xA4 -> EURO SIGN
    '\u20af'   #  0xA5 -> DRACHMA SIGN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u037a'   #  0xAA -> GREEK YPOGEGRAMMENI
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xAC -> NOT SIGN
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\ufffe'
    '\u2015'   #  0xAF -> HORIZONTAL BAR
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    '\xb3'     #  0xB3 -> SUPERSCRIPT THREE
    '\u0384'   #  0xB4 -> GREEK TONOS
    '\u0385'   #  0xB5 -> GREEK DIALYTIKA TONOS
    '\u0386'   #  0xB6 -> GREEK CAPITAL LETTER ALPHA WITH TONOS
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\u0388'   #  0xB8 -> GREEK CAPITAL LETTER EPSILON WITH TONOS
    '\u0389'   #  0xB9 -> GREEK CAPITAL LETTER ETA WITH TONOS
    '\u038a'   #  0xBA -> GREEK CAPITAL LETTER IOTA WITH TONOS
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u038c'   #  0xBC -> GREEK CAPITAL LETTER OMICRON WITH TONOS
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\u038e'   #  0xBE -> GREEK CAPITAL LETTER UPSILON WITH TONOS
    '\u038f'   #  0xBF -> GREEK CAPITAL LETTER OMEGA WITH TONOS
    '\u0390'   #  0xC0 -> GREEK SMALL LETTER IOTA WITH DIALYTIKA AND TONOS
    '\u0391'   #  0xC1 -> GREEK CAPITAL LETTER ALPHA
    '\u0392'   #  0xC2 -> GREEK CAPITAL LETTER BETA
    '\u0393'   #  0xC3 -> GREEK CAPITAL LETTER GAMMA
    '\u0394'   #  0xC4 -> GREEK CAPITAL LETTER DELTA
    '\u0395'   #  0xC5 -> GREEK CAPITAL LETTER EPSILON
    '\u0396'   #  0xC6 -> GREEK CAPITAL LETTER ZETA
    '\u0397'   #  0xC7 -> GREEK CAPITAL LETTER ETA
    '\u0398'   #  0xC8 -> GREEK CAPITAL LETTER THETA
    '\u0399'   #  0xC9 -> GREEK CAPITAL LETTER IOTA
    '\u039a'   #  0xCA -> GREEK CAPITAL LETTER KAPPA
    '\u039b'   #  0xCB -> GREEK CAPITAL LETTER LAMDA
    '\u039c'   #  0xCC -> GREEK CAPITAL LETTER MU
    '\u039d'   #  0xCD -> GREEK CAPITAL LETTER NU
    '\u039e'   #  0xCE -> GREEK CAPITAL LETTER XI
    '\u039f'   #  0xCF -> GREEK CAPITAL LETTER OMICRON
    '\u03a0'   #  0xD0 -> GREEK CAPITAL LETTER PI
    '\u03a1'   #  0xD1 -> GREEK CAPITAL LETTER RHO
    '\ufffe'
    '\u03a3'   #  0xD3 -> GREEK CAPITAL LETTER SIGMA
    '\u03a4'   #  0xD4 -> GREEK CAPITAL LETTER TAU
    '\u03a5'   #  0xD5 -> GREEK CAPITAL LETTER UPSILON
    '\u03a6'   #  0xD6 -> GREEK CAPITAL LETTER PHI
    '\u03a7'   #  0xD7 -> GREEK CAPITAL LETTER CHI
    '\u03a8'   #  0xD8 -> GREEK CAPITAL LETTER PSI
    '\u03a9'   #  0xD9 -> GREEK CAPITAL LETTER OMEGA
    '\u03aa'   #  0xDA -> GREEK CAPITAL LETTER IOTA WITH DIALYTIKA
    '\u03ab'   #  0xDB -> GREEK CAPITAL LETTER UPSILON WITH DIALYTIKA
    '\u03ac'   #  0xDC -> GREEK SMALL LETTER ALPHA WITH TONOS
    '\u03ad'   #  0xDD -> GREEK SMALL LETTER EPSILON WITH TONOS
    '\u03ae'   #  0xDE -> GREEK SMALL LETTER ETA WITH TONOS
